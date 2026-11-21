"""Collectives on NumPy arrays that every rank calls, for any communicator or none.

``comm`` is an mpi4py communicator, the serial stand-in (:class:`maybempi.SerialComm`),
a :class:`maybempi.testing.ThreadComm`, or ``None`` for one process without MPI; with one
rank every function returns at once, without communicating. The module needs NumPy,
which ``import maybempi`` itself does not import::

    from maybempi import MPI
    from maybempi.collectives import allsum, block, gather_columns

    comm = MPI.COMM_WORLD
    mine = block(len(points), comm)        # this rank's share of the items
    values = evaluate(points[mine])        # the work, split
    everything, _ = gather_columns(comm, values)  # the whole result on every rank
    total = allsum(comm, values.sum())
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from maybempi.serial import SerialMPI


def size(comm: Any) -> int:
    """The number of ranks of ``comm``, 1 for ``None``."""
    return 1 if comm is None else int(comm.Get_size())


def rank(comm: Any) -> int:
    """This process's rank in ``comm``, 0 for ``None``."""
    return 0 if comm is None else int(comm.Get_rank())


def is_parallel(comm: Any) -> bool:
    """Whether ``comm`` has more than one rank."""
    return size(comm) > 1


def _op(comm: Any, name: str) -> Any:
    """The reduction ``name`` (``"SUM"``, ``"MAX"``) of the MPI that ``comm`` belongs to."""
    module = sys.modules.get(type(comm).__module__)
    op = getattr(module, name, None)  # mpi4py.MPI.SUM for an mpi4py communicator
    return getattr(SerialMPI, name) if op is None else op


def block(n: int, comm: Any) -> slice:
    """This rank's contiguous share of ``range(n)``; the shares differ by one at most."""
    ranks, me = size(comm), rank(comm)
    base, extra = divmod(n, ranks)
    start = me * base + min(me, extra)
    return slice(start, start + base + (me < extra))


def gather_columns(comm: Any, local: Any) -> tuple[np.ndarray, int]:
    """Every rank's columns, in rank order, and the offset of this rank's first.

    ``local`` is ``(k, n_rank)`` or ``(n_rank,)``, as floats; the result is ``(k, N)``
    or ``(N,)`` with ``N`` the sum of the ranks' ``n_rank``, the same on every rank
    (``MPI_Allgatherv``). The ranks may hold different numbers of columns, also none.
    """
    local = np.asarray(local, dtype=float)
    if not is_parallel(comm):
        return local.copy(), 0
    flat = local.ndim == 1
    rows = local.reshape(1, -1) if flat else local
    k, n = rows.shape
    counts = np.asarray(comm.allgather(n), dtype=np.int64)
    offset = int(counts[: rank(comm)].sum())
    send = np.ascontiguousarray(rows.T)  # (n, k): one rank's block is contiguous
    received = np.empty((int(counts.sum()), k))
    comm.Allgatherv(send, [received, counts * k])
    gathered = np.ascontiguousarray(received.T)
    return (gathered[0] if flat else gathered), offset


def gather_blocks(comm: Any, local: Any, n: int) -> np.ndarray:
    """The ``(..., n)`` whole of per-rank ``block(n, comm)`` pieces ``(..., n_rank)``."""
    if not is_parallel(comm):
        return np.asarray(local)
    gathered, _ = gather_columns(comm, local)
    if gathered.shape[-1] != n:
        raise ValueError(
            f"the ranks' pieces hold {gathered.shape[-1]} columns, not {n}"
        )
    return gathered


def allsum(comm: Any, value: Any) -> Any:
    """The sum of ``value`` (a number or an array) over the ranks, on every rank."""
    if not is_parallel(comm):
        return value
    if np.ndim(value):
        array = np.ascontiguousarray(value, dtype=float)
        out = np.empty_like(array)
        comm.Allreduce(array, out, op=_op(comm, "SUM"))
        return out
    return comm.allreduce(value, op=_op(comm, "SUM"))


def allmax(comm: Any, value: Any) -> Any:
    """The largest ``value`` (a number) over the ranks, on every rank."""
    if not is_parallel(comm):
        return value
    return comm.allreduce(value, op=_op(comm, "MAX"))


__all__ = [
    "allmax",
    "allsum",
    "block",
    "gather_blocks",
    "gather_columns",
    "is_parallel",
    "rank",
    "size",
]
