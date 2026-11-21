"""maybempi.testing: thread ranks behave like MPI ranks."""

import sys

import numpy as np
import pytest

from maybempi import SerialMPI
from maybempi.testing import run_ranks

# Pyodide (the browser) has no threads
pytestmark = pytest.mark.skipif(sys.platform == "emscripten", reason="no threads")


def test_collectives():
    def work(comm):
        me = comm.Get_rank()
        comm.Barrier()
        buf = np.full(2, float(me))
        comm.Bcast(buf, root=2)
        out = np.empty(2)
        comm.Allreduce(np.array([me, -me], dtype=float), out, op=SerialMPI.MAX)
        low = np.empty(2)
        comm.Allreduce(np.array([me, -me], dtype=float), low, op=SerialMPI.MIN)
        gathered = np.empty(6)
        comm.Allgatherv(np.full(me + 1, float(me)), gathered)
        return (
            comm.Get_size(),
            comm.allgather(me),
            comm.allreduce(me),
            comm.allreduce(me, op=SerialMPI.MAX),
            comm.allreduce(me, op=SerialMPI.MIN),
            comm.bcast(f"from {me}", root=1),
            buf,
            out,
            low,
            gathered,
        )

    for result in run_ranks(3, work):
        size, every, total, largest, smallest, message, buf, out, low, gathered = result
        assert size == 3 and every == [0, 1, 2]
        assert (total, largest, smallest) == (3, 2, 0)
        assert message == "from 1"
        assert np.array_equal(buf, [2.0, 2.0])
        assert np.array_equal(out, [2.0, 0.0]) and np.array_equal(low, [0.0, -2.0])
        assert np.array_equal(gathered, [0, 1, 1, 2, 2, 2])


def test_an_error_on_one_rank_is_raised_and_frees_the_others():
    def work(comm):
        if comm.Get_rank() == 1:
            raise RuntimeError("rank 1 failed")
        comm.Barrier()

    with pytest.raises(RuntimeError, match="rank 1"):
        run_ranks(2, work)


def test_a_missed_collective_times_out():
    def work(comm):
        if comm.Get_rank() == 0:
            comm.Barrier()

    with pytest.raises(Exception):  # noqa: B017 -- threading.BrokenBarrierError
        run_ranks(2, work, timeout=0.2)
