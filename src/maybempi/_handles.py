"""Placeholders for MPI handles and constants, shared by the serial stand-in's modules."""

from __future__ import annotations

import sys
from typing import Any


class _Constant:
    """A named placeholder for an MPI constant (an op, a datatype, ``IN_PLACE``, ...)."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"SerialMPI.{self.name}"


class _Null(_Constant):
    """A null handle (``COMM_NULL``, ``DATATYPE_NULL``, ...): false, like mpi4py's."""

    __slots__ = ()

    def __bool__(self) -> bool:
        return False


_IN_PLACE = _Constant("IN_PLACE")
_COMM_NULL = _Null("COMM_NULL")
_PROC_NULL = -2
_ANY_SOURCE = -1
_ANY_TAG = -1
_UNDEFINED = -32766


def _mpi4py_constant(name: str) -> Any:
    """Return ``mpi4py.MPI.<name>`` if the application imported mpi4py, else None."""
    module = sys.modules.get("mpi4py.MPI")
    return getattr(module, name, None) if module is not None else None


def _is_proc_null(rank: Any) -> bool:
    """Tell whether `rank` is ``PROC_NULL``, the stand-in's or mpi4py's."""
    return rank == _PROC_NULL or rank == _mpi4py_constant("PROC_NULL")


def _is_any_tag(tag: Any) -> bool:
    """Tell whether `tag` is ``ANY_TAG``, the stand-in's or mpi4py's."""
    return tag == _ANY_TAG or tag == _mpi4py_constant("ANY_TAG")
