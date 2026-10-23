---
title: Using maybempi in a library
description: Decide once, share the communicator, and tell serial from MPI runs.
---

A library that supports both serial and MPI runs imports the MPI module from maybempi wherever
it would import mpi4py's:

```python
from maybempi import MPI


def default_comm():
    return MPI.COMM_WORLD
```

`MPI` is decided once per process, so the library, the libraries it uses and the application
always agree, even if each of them imports it.

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

## PETSc

petsc4py takes mpi4py communicators but not the stand-in. `maybempi.petsc_comm` converts
either: the stand-in, and any communicator of one process, become `PETSc.COMM_SELF`, so a
serial run does not import mpi4py; larger communicators are wrapped with `PETSc.Comm(comm)`,
which shares the MPI communicator rather than duplicating it:

```python
from petsc4py import PETSc

import maybempi

comm = maybempi.petsc_comm(MPI.COMM_WORLD)  # or petsc_comm() for COMM_WORLD
matrix = PETSc.Mat().createAIJ(size=((local, total), (local, total)), comm=comm)
```

petsc4py is not a dependency of maybempi; `petsc_comm` imports it when called and raises an
`ImportError` that says how to install it otherwise. Importing `petsc4py.PETSc` initializes
PETSc, and MPI too if PETSc was built with MPI, so call it only where PETSc is used.

## Tests

Tests can force either mode without a launcher: `maybempi.get_mpi(False)` returns the stand-in,
and `MAYBEMPI=0` keeps a whole `mpirun` job serial. To run a test suite on several ranks:

```bash
mpiexec -n 2 python -m pytest tests/mpi
```
