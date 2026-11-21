---
title: Array collectives
description: Split work over the ranks, gather and reduce NumPy arrays, test with thread ranks.
---

`maybempi.collectives` holds the few collectives that code which splits its work over the ranks
keeps writing: a share of the items per rank, the gather of per-rank results, sums and maxima.
They take any communicator (mpi4py, the serial stand-in, the thread ranks below) or `None`,
and with one rank they return at once without communicating. The module needs NumPy, which
`import maybempi` does not import.

```python
from maybempi import MPI
from maybempi.collectives import allsum, block, gather_blocks

comm = MPI.COMM_WORLD
mine = block(len(points), comm)  # this rank's contiguous share
values = evaluate(points[mine])  # each rank does its share
values = gather_blocks(comm, values, len(points))  # the whole result on every rank
total = allsum(comm, values.sum())
```

| Function | Result, on every rank |
|---|---|
| `block(n, comm)` | this rank's `slice` of `range(n)`; the shares differ by one at most |
| `gather_columns(comm, local)` | the ranks' `(k, n_rank)` or `(n_rank,)` arrays joined along the last axis in rank order (`Allgatherv`), and this rank's offset in it |
| `gather_blocks(comm, local, n)` | the `(..., n)` whole of per-rank `block(n, comm)` pieces |
| `allsum(comm, value)` | the sum of a number or an array over the ranks |
| `allmax(comm, value)` | the largest number over the ranks |
| `size(comm)`, `rank(comm)`, `is_parallel(comm)` | 1, 0 and `False` for `None` |

Every rank must call them, also a rank with nothing to contribute: pass an empty array then.

## Testing with thread ranks

`maybempi.testing.run_ranks(size, function)` calls `function(comm)` on `size` threads of one
process at once, each with a `ThreadComm` that implements `allgather`, `allreduce`, `bcast`,
`Allgatherv`, `Allreduce`, `Bcast` and `Barrier`. So the parallel branches of a library run and
count towards coverage under plain pytest, without mpirun or mpi4py:

```python
import numpy as np
from maybempi.testing import run_ranks


def test_parallel_sum_matches_serial():
    data = np.arange(100.0)
    serial = my_sum(data, comm=None)
    assert run_ranks(3, lambda comm: my_sum(data, comm=comm)) == [serial] * 3
```

A rank that raises frees the others, and `run_ranks` re-raises its error; a collective that
some rank never reaches fails after `timeout` seconds instead of hanging. Keep one test under
a real `mpiexec` as well: threads share memory and run one at a time where the code holds the
GIL, so they check the logic of the collectives, not MPI itself.
