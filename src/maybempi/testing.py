"""Ranks as threads of one process, to test parallel code paths under plain pytest.

:func:`run_ranks` calls a function on several :class:`ThreadComm` ranks at once. They
implement the common collectives (``allgather``, ``allreduce``, ``bcast``, ``Allgatherv``,
``Allreduce``, ``Bcast``, ``Barrier``), enough for code built on
:mod:`maybempi.collectives`, without mpirun or mpi4py::

    from maybempi.testing import run_ranks

    def work(comm):
        return my_parallel_sum(data, comm)

    assert run_ranks(3, work) == [expected] * 3

A rank that raises aborts the others' waits, and :func:`run_ranks` re-raises its error.
The reductions take the stand-in's ops (``maybempi.SerialMPI.SUM``, ``MAX``, ``MIN``).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from maybempi.serial import SerialMPI


class _World:
    def __init__(self, size: int, timeout: float):
        self.size = size
        self.barrier = threading.Barrier(size, timeout=timeout)
        self.slots: list = [None] * size


def _reduce(values: list, op: Any) -> Any:
    """``values`` combined by ``op`` (sum unless MAX or MIN)."""
    if op is SerialMPI.MAX:
        return np.max(values, axis=0) if np.ndim(values[0]) else max(values)
    if op is SerialMPI.MIN:
        return np.min(values, axis=0) if np.ndim(values[0]) else min(values)
    return np.sum(values, axis=0) if np.ndim(values[0]) else sum(values)


class ThreadComm:
    """Rank ``rank`` of a group of threads that stand for MPI ranks (see :func:`run_ranks`)."""

    def __init__(self, world: _World, rank: int):
        """Rank ``rank`` of ``world``; :func:`run_ranks` makes them."""
        self._world = world
        self._rank = rank

    def Get_rank(self) -> int:
        """This rank."""
        return self._rank

    def Get_size(self) -> int:
        """The number of ranks."""
        return self._world.size

    def _exchange(self, value: Any) -> list:
        """Every rank's ``value``, in rank order."""
        world = self._world
        world.slots[self._rank] = value
        world.barrier.wait()
        values = list(world.slots)
        world.barrier.wait()
        return values

    def Barrier(self) -> None:
        """Wait for every rank."""
        self._world.barrier.wait()

    def allgather(self, sendobj: Any) -> list:
        """Every rank's object, in rank order."""
        return self._exchange(sendobj)

    def allreduce(self, sendobj: Any, op: Any = None) -> Any:
        """The ranks' objects combined by ``op`` (default sum)."""
        return _reduce(self._exchange(sendobj), op)

    def bcast(self, obj: Any, root: int = 0) -> Any:
        """``root``'s object, on every rank."""
        return self._exchange(obj)[root]

    def Allgatherv(self, sendbuf: Any, recvbuf: Any) -> None:
        """Every rank's buffer, concatenated in rank order, into ``recvbuf[0]``."""
        buffer = recvbuf[0] if isinstance(recvbuf, (list, tuple)) else recvbuf
        parts = self._exchange(np.array(sendbuf, copy=True).ravel())
        buffer.reshape(-1)[:] = np.concatenate(parts)

    def Allreduce(self, sendbuf: Any, recvbuf: Any, op: Any = None) -> None:
        """The ranks' buffers combined element-wise by ``op`` (default sum)."""
        recvbuf[...] = _reduce(self._exchange(np.array(sendbuf, copy=True)), op)

    def Bcast(self, buf: Any, root: int = 0) -> None:
        """``root``'s buffer, copied into every rank's ``buf``."""
        buf[...] = self._exchange(np.array(buf, copy=True))[root]


def run_ranks(
    size: int, function: Callable[[ThreadComm], Any], timeout: float = 120.0
) -> list:
    """``function(comm)`` on ``size`` thread ranks at once; their results in rank order.

    A collective that some rank never reaches fails after ``timeout`` seconds instead
    of hanging. The first error raised on a rank is re-raised here.
    """
    world = _World(size, timeout)
    results: list = [None] * size
    errors: list = []

    def target(rank: int) -> None:
        try:
            results[rank] = function(ThreadComm(world, rank))
        except BaseException as error:  # noqa: BLE001 -- re-raised below
            errors.append(error)
            world.barrier.abort()

    threads = [threading.Thread(target=target, args=(r,)) for r in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


__all__ = ["ThreadComm", "run_ranks"]
