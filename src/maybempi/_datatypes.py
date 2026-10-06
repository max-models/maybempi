"""Datatypes of the serial stand-in: the named placeholders and derived types.

A datatype describes where its elements lie, in units of its base type (for
example ``DOUBLE``): a list of runs ``(start, length)`` and an extent, the
distance between consecutive copies. Named types are one element with extent
one; ``Create_contiguous``, ``Create_vector`` and ``Create_subarray`` build the
runs of derived types. Buffers are read and written along these runs, in the
array's own elements, so a derived type must be built on the array's type.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import Any

from maybempi._handles import _Constant

#: Row-major (C) and column-major (Fortran) order for ``Create_subarray``.
ORDER_C = 0
ORDER_F = 1

Run = tuple[int, int]


def _merge(runs: Iterable[Run]) -> tuple[Run, ...]:
    """Join runs that continue each other, keeping their order."""
    merged: list[list[int]] = []
    for start, length in runs:
        if merged and merged[-1][0] + merged[-1][1] == start:
            merged[-1][1] += length
        else:
            merged.append([start, length])
    return tuple((start, length) for start, length in merged)


class _Datatype(_Constant):
    """Placeholder for an MPI datatype (``isinstance(t, MPI.Datatype)`` holds).

    Derived types know their layout, so buffers and file views built from
    them select the right elements.
    """

    __slots__ = ("_base", "_extent", "_itemsize", "_runs")

    def __init__(
        self,
        name: str,
        itemsize: int = 1,
        base: _Datatype | None = None,
        runs: Sequence[Run] = ((0, 1),),
        extent: int = 1,
    ) -> None:
        super().__init__(name)
        self._itemsize = itemsize
        self._base = self if base is None else base
        self._runs = tuple(runs)
        self._extent = extent

    # ------------------------------------------------------------ queries
    @property
    def _elements(self) -> int:
        """Number of base elements in one copy of the type."""
        return sum(length for _, length in self._runs)

    @property
    def _named(self) -> bool:
        """Whether this is a named (predefined) type, one element."""
        return self._base is self

    @property
    def size(self) -> int:
        """Bytes of data in one copy of the type."""
        return self._elements * self._itemsize

    @property
    def extent(self) -> int:
        """Bytes from the start of one copy to the start of the next."""
        return self._extent * self._itemsize

    @property
    def lb(self) -> int:
        """The lower bound: 0."""
        return 0

    @property
    def ub(self) -> int:
        """The upper bound: the extent."""
        return self.extent

    def Get_size(self) -> int:
        """Return the bytes of data in one copy of the type."""
        return self.size

    def Get_extent(self) -> tuple[int, int]:
        """Return ``(lower bound, extent)`` in bytes."""
        return 0, self.extent

    def Get_name(self) -> str:
        """Return the name, e.g. ``"DOUBLE"``."""
        return self.name

    # ------------------------------------------------------- life cycle
    def Commit(self) -> _Datatype:
        """Return the type itself: nothing to commit."""
        return self

    def Free(self) -> None:
        """Do nothing."""
        return None

    def Dup(self) -> _Datatype:
        """Return a copy of the type."""
        return _Datatype(
            self.name, self._itemsize, self._base, self._runs, self._extent
        )

    # ---------------------------------------------------- derived types
    def _derived(self, name: str, places: Iterable[int], extent: int) -> _Datatype:
        """Return the type made of copies of this one at element `places`."""
        runs = _merge(
            (place * self._extent + start, length)
            for place in places
            for start, length in self._runs
        )
        return _Datatype(name, self._itemsize, self._base, runs, extent * self._extent)

    def Create_contiguous(self, count: int) -> _Datatype:
        """Return `count` copies of this type, one after the other."""
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        return self._derived(f"contiguous({count}, {self.name})", range(count), count)

    def Create_vector(self, count: int, blocklength: int, stride: int) -> _Datatype:
        """Return `count` blocks of `blocklength` copies, `stride` copies apart."""
        if count < 0 or blocklength < 0 or stride < 0:
            raise ValueError(
                "count, blocklength and stride must be non-negative (negative strides "
                "are not supported by the serial stand-in)",
            )
        places = (i * stride + j for i in range(count) for j in range(blocklength))
        extent = (count - 1) * stride + blocklength if count else 0
        return self._derived(
            f"vector({count}, {blocklength}, {stride}, {self.name})", places, extent
        )

    def Create_subarray(
        self,
        sizes: Sequence[int],
        subsizes: Sequence[int],
        starts: Sequence[int],
        order: int = ORDER_C,
    ) -> _Datatype:
        """Return the block ``starts:starts+subsizes`` of an array of shape `sizes`."""
        sizes, subsizes, starts = list(sizes), list(subsizes), list(starts)
        if not len(sizes) == len(subsizes) == len(starts) or not sizes:
            raise ValueError("sizes, subsizes and starts need the same, nonzero length")
        for size, subsize, start in zip(sizes, subsizes, starts, strict=True):
            if subsize < 1 or start < 0 or start + subsize > size:
                raise ValueError(
                    f"a subarray of {subsizes} at {starts} does not fit into {sizes}"
                )
        if order == ORDER_F:
            sizes, subsizes, starts = sizes[::-1], subsizes[::-1], starts[::-1]
        elif order != ORDER_C:
            raise ValueError(f"order must be ORDER_C or ORDER_F, got {order}")
        strides = [math.prod(sizes[axis + 1 :]) for axis in range(len(sizes))]
        rows = itertools.product(
            *(range(s, s + n) for s, n in zip(starts[:-1], subsizes[:-1], strict=True))
        )
        places = (
            sum(i * stride for i, stride in zip(row, strides[:-1], strict=True))
            + starts[-1]
            + k
            for row in rows
            for k in range(subsizes[-1])
        )
        return self._derived(
            f"subarray({sizes}, {subsizes}, {starts}, {self.name})",
            places,
            math.prod(sizes),
        )


def _buffer_runs(spec: Any) -> tuple[Any, tuple[Run, ...]]:
    """Return the array of a point-to-point buffer spec and the runs of elements it covers.

    The spec is an array, ``[array, datatype]``, ``[array, count, datatype]`` or
    ``[array, (count, displacement), datatype]``, as mpi4py takes them. Counts
    and displacements count copies of the datatype.

    Raises:
        ValueError: If the datatype does not fit into the array.
    """
    if not isinstance(spec, (list, tuple)):
        return spec, ((0, spec.size),) if spec.size else ()
    array, *rest = spec
    datatype = rest.pop() if rest and isinstance(rest[-1], _Datatype) else None
    count: int | None = None
    displacement = 0
    if rest and rest[0] is not None:
        if isinstance(rest[0], (list, tuple)):
            count, displacement = (int(value) for value in rest[0])
        else:
            count = int(rest[0])
    extent = 1 if datatype is None else datatype._extent
    first = displacement * extent
    if datatype is None or datatype._named:
        count = array.size - first if count is None else count
        runs = ((first, count),) if count > 0 else ()
    else:
        if count is None:
            count = (array.size - first) // extent if extent else 0
        runs = _merge(
            (first + copy * extent + start, length)
            for copy in range(count)
            for start, length in datatype._runs
        )
    end = max((start + length for start, length in runs), default=0)
    if first < 0 or end > array.size:
        raise ValueError(
            f"the datatype covers {end} elements, but the buffer has {array.size}"
        )
    return array, runs
