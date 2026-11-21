"""maybempi.collectives on one process, the serial stand-in and thread ranks."""

import numpy as np
import pytest

from maybempi import SerialMPI
from maybempi.collectives import (
    _op,
    allmax,
    allsum,
    block,
    gather_blocks,
    gather_columns,
    is_parallel,
    rank,
    size,
)
from maybempi.testing import run_ranks


@pytest.mark.parametrize("comm", [None, SerialMPI.COMM_WORLD])
def test_one_rank_returns_at_once(comm):
    local = np.arange(6.0).reshape(2, 3)
    gathered, offset = gather_columns(comm, local)
    assert offset == 0 and np.array_equal(gathered, local)
    assert gathered is not local
    assert allsum(comm, 2.5) == 2.5
    assert allmax(comm, 4) == 4
    assert block(7, comm) == slice(0, 7)
    assert gather_blocks(comm, local, 3) is local
    assert (size(comm), rank(comm), is_parallel(comm)) == (1, 0, False)


def test_ops_follow_the_communicator():
    assert _op(SerialMPI.COMM_WORLD, "SUM") is SerialMPI.SUM
    assert _op(object(), "MAX") is SerialMPI.MAX


def test_on_thread_ranks():
    def work(comm):
        me = comm.Get_rank()
        share = block(10, comm)
        columns, offset = gather_columns(comm, np.full((2, me + 1), me))
        flat, _ = gather_columns(comm, np.full(me, float(me)))
        pieces = gather_blocks(comm, np.arange(10.0)[share], 10)
        return (
            share,
            columns,
            offset,
            flat,
            pieces,
            allsum(comm, me),
            allsum(comm, np.array([me, 1.0])),
            allmax(comm, 5 * me),
        )

    results = run_ranks(3, work)
    assert [r[0] for r in results] == [slice(0, 4), slice(4, 7), slice(7, 10)]
    assert [r[2] for r in results] == [0, 1, 3]
    for _, columns, _, flat, pieces, count, array, largest in results:
        assert np.array_equal(columns, np.repeat([[0, 1, 1, 2, 2, 2]], 2, axis=0))
        assert np.array_equal(flat, [1.0, 2.0, 2.0])
        assert np.array_equal(pieces, np.arange(10.0))
        assert count == 3 and np.array_equal(array, [3.0, 3.0])
        assert largest == 10


def test_blocks_must_cover_the_range():
    def work(comm):
        with pytest.raises(ValueError, match="columns"):
            gather_blocks(comm, np.zeros(2), 5)

    run_ranks(2, work)


@pytest.mark.parametrize("n", [0, 1, 2, 9])
def test_blocks_partition_the_range(n):
    shares = run_ranks(4, lambda comm: block(n, comm))
    covered = [i for share in shares for i in range(n)[share]]
    assert covered == list(range(n))
    assert (
        max(len(range(n)[s]) for s in shares) - min(len(range(n)[s]) for s in shares)
        <= 1
    )
