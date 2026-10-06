---
title: The serial stand-in
description: What SerialMPI and SerialComm do with each call on one process.
---

In a serial run, `maybempi.get_mpi()` returns a `SerialMPI` object in place of the `mpi4py.MPI`
module. Its `COMM_WORLD` and `COMM_SELF` are `SerialComm` objects: communicators of size 1.

Every call does what MPI does on one process. Calls that return a value always return it,
never `None` in its place. Calls that are not implemented raise `AttributeError`: a stand-in
that silently does nothing would hide bugs, for example a halo exchange that never happens.

## Object collectives (lowercase)

| Call                                          | Returns in a serial run              |
| --------------------------------------------- | ------------------------------------ |
| `bcast(obj)`, `reduce(x)`, `allreduce(x)`, `scan(x)` | the value itself              |
| `exscan(x)`                                   | `None` (undefined on rank 0, as in mpi4py) |
| `gather(x)`, `allgather(x)`                   | `[x]`                                |
| `scatter([x])`                                | `x` (`ValueError` unless one item)   |
| `alltoall([x])`                               | `[x]` (`ValueError` unless one item) |
| `sendrecv(x, dest=0, source=0)`               | `x`; `None` with `PROC_NULL`         |
| `ibcast`, `iallreduce`                        | a completed request; `wait()` returns the value |
| `Barrier()`, `barrier()`, `Ibarrier()`        | return at once                       |

The `op` argument is accepted and ignored: on one process every reduction returns its input.

## Buffer collectives (uppercase)

| Call                                                     | Does in a serial run                      |
| -------------------------------------------------------- | ----------------------------------------- |
| `Allreduce`, `Reduce`, `Scan`, `Allgather`, `Gather`, `Alltoall`, `Scatter` | copies the send buffer into the receive buffer |
| the same with `MPI.IN_PLACE`                             | nothing                                   |
| `Gatherv`, `Allgatherv`                                  | copies at the displacement of rank 0      |
| `Alltoallv`                                              | copies the part for rank 0 to the displacement of rank 0 |
| `Scatterv`                                               | copies the part at the displacement of rank 0 |
| `Bcast`, `Exscan`                                        | nothing                                   |
| `Sendrecv` to and from rank 0                            | copies; nothing with `PROC_NULL`          |
| `Ibcast`, `Iallreduce`, `Iallgather`                     | as above, and returns a completed request |

Buffers are NumPy-like arrays (with `reshape`, `flags` and `size`) or mpi4py buffer
specifications such as `[array, MPI.DOUBLE]` and `[array, counts, displs, MPI.DOUBLE]`. The
receive buffer must be C-contiguous and large enough, otherwise `ValueError` is raised.

### Device arrays

A send buffer with a `.get()` method, such as a CuPy array, is copied to a host receive buffer
with `.get()`, and a host send buffer is copied into a device receive buffer. An application that
counts host-device transfers can register a hook:

```python
import maybempi


def count_copy(kind, array):  # kind is "to_host" or "to_device"
    print(kind, array.nbytes)


maybempi.set_copy_hook(count_copy)
```

## Next to real MPI

A `SerialComm` can also be used inside an MPI job, for work that one rank does on its own (for
example post-processing on rank 0 while `MPI` is mpi4py). It then accepts mpi4py's own
`MPI.IN_PLACE`, `MPI.PROC_NULL` and `MPI.ANY_SOURCE` as well as its own, so the same calls work
with either module's constants:

```python
from maybempi import MPI, SerialComm

comm = SerialComm()  # size 1, even under mpirun
comm.Reduce(MPI.IN_PLACE, totals, op=MPI.SUM, root=0)  # nothing to do
```

## Ranks

Only rank 0 exists. A `root`, `dest` or `source` other than 0, `PROC_NULL` or `ANY_SOURCE`
raises `ValueError`.

## Communicators, requests and the module

- `Get_rank()` is 0, `Get_size()` is 1, and the attributes `rank` and `size` match.
- `Dup()`, `Clone()` and `Split()` return new serial communicators. `Split(MPI.UNDEFINED)`
  returns `COMM_NULL`.
- `Abort(code)` raises `SystemExit(code)`.
- Requests are complete: `Wait`, `Waitall`, `Test` and `Testall` return at once.
- `MPI.Is_initialized()` is False: MPI is never started. `Wtime`, `Wtick`,
  `Get_processor_name` and `Query_thread` work as usual.
- The reduction operations (`SUM`, `MAX`, ...) and datatypes (`DOUBLE`, `INT64_T`, ...) are
  placeholders: `isinstance(MPI.DOUBLE, MPI.Datatype)` and `isinstance(MPI.SUM, MPI.Op)` hold.
  `MPI._typedict` maps NumPy type characters to datatypes, like mpi4py's.
- Null handles (`COMM_NULL`, `DATATYPE_NULL`, ...) are false, communicators true.
- The integer constants (`PROC_NULL`, `ANY_SOURCE`, `ROOT`, `UNDEFINED`, ...) are only
  guaranteed to be consistent within the stand-in. Their values in mpi4py depend on the MPI
  library.

## Not implemented

Point-to-point messages to other ranks (`Send`, `Recv`, `Isend`, `Irecv`, ...), topologies
(`Create_cart`, ...), groups, one-sided communication (`Win`) and I/O raise `AttributeError`.
Codes that need them in a serial run usually take a separate serial path, for example passing
`comm=None` to their domain decomposition.
