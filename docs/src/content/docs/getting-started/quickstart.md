---
title: Quickstart
description: Replace the mpi4py import with maybempi.get_mpi().
---

Replace

```python
from mpi4py import MPI
```

with

```python
import maybempi

MPI = maybempi.get_mpi()
```

The rest of the code stays as it is. Under `mpirun`, `mpiexec` or `srun`, `MPI` is the
`mpi4py.MPI` module. In a plain `python script.py`, it is a serial stand-in, and mpi4py is never
imported:

```python
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()  # 0 and 1 in a serial run

total = comm.allreduce(local_total, op=MPI.SUM)  # local_total in a serial run
everything = comm.gather(local_result, root=0)  # [local_result]
comm.Allreduce(MPI.IN_PLACE, density, op=MPI.SUM)  # does nothing on one process
```

Tell the two apart with `maybempi.is_serial(MPI)` (or `maybempi.is_serial(comm)`), for example to
skip writing per-rank files.

## Choose the GPU before MPI starts

`maybempi.local_rank()` reads the node-local rank from the launcher's environment, so a code can
pick its device before `MPI_Init`:

```python
import maybempi

device = maybempi.local_rank() % n_gpus_per_node
# select `device`, then:
MPI = maybempi.get_mpi()
```

## Next

- [How the decision is made](/maybempi/guides/detection/): which launchers are recognized, and
  how to override the decision.
- [The serial stand-in](/maybempi/guides/serial-stand-in/): what each call does on one process.
- [The tutorial](/maybempi/tutorials/one-code-path/), executed with its outputs.
- The [API reference](/maybempi/api/maybempi/).
