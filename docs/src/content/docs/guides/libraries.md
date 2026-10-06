---
title: Using maybempi in a library
description: Decide once, share the communicator, and tell serial from MPI runs.
---

A library that supports both serial and MPI runs gets its MPI module once, at import time:

```python
# mylib/_mpi.py
import maybempi

MPI = maybempi.get_mpi()


def default_comm():
    return MPI.COMM_WORLD
```

All modules import `MPI` from there. Because `get_mpi()` decides once per process, the
library and the application using it always agree, even if both call it.

## Type annotations

For type checkers, annotate with mpi4py's types and fall back to the stand-in at run time:

```python
from typing import TYPE_CHECKING

import maybempi

if TYPE_CHECKING:
    from mpi4py import MPI
else:
    MPI = maybempi.get_mpi()

Comm = "MPI.Comm | maybempi.SerialComm"
```

`MPI.Intracomm | None` also works at run time with the stand-in, so annotations evaluated at
definition time do not break.

## Serial-only behavior

Use `maybempi.is_serial` rather than `comm.Get_size() == 1` when the code must know that MPI is
not running at all, for example before calling into a C library that needs `MPI_Init`:

```python
if maybempi.is_serial(comm):
    run_serial_solver()
else:
    run_distributed_solver(comm)
```

## Tests

Tests can force either mode without a launcher: `maybempi.get_mpi(False)` returns the stand-in,
and `MAYBEMPI=0` keeps a whole `mpirun` job serial. To run a test suite on several ranks:

```bash
mpiexec -n 2 python -m pytest tests/mpi
```
