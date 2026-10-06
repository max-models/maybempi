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
| `sendrecv(x, dest=0, source=0)`               | a copy of `x` (see [Point-to-point](#point-to-point)) |
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
| `Sendrecv` to and from rank 0                            | copies (see [Point-to-point](#point-to-point)) |
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

## Point-to-point

Rank 0 can send to itself. A send copies the data into a message that the communicator keeps
until a receive with a matching tag (or `ANY_TAG`) takes it; messages with the same tag are
received in the order they were sent, and messages on different communicators (e.g. after
`Dup()`) stay apart. A posted receive (`Irecv`, `irecv`) completes as soon as its message is
sent.

```python
comm = MPI.COMM_WORLD
request = comm.Irecv(halo, source=comm.rank, tag=1)  # posted first
comm.Send(edge, dest=comm.rank, tag=1)  # completes the receive
request.Wait()
```

| Call                                                | Does in a serial run                              |
| --------------------------------------------------- | ------------------------------------------------- |
| `Send`, `Isend`, `Bsend`, `Rsend`, `send`, `isend`  | keeps a copy of the data; the request is complete |
| `Recv`, `recv`                                      | takes the first matching message (`RuntimeError` if none was sent) |
| `Irecv`, `irecv`                                    | takes a matching message, or completes when one is sent |
| `Ssend`, `ssend`                                    | needs a posted matching receive (`RuntimeError` otherwise) |
| `Issend`, `issend`                                  | completes when a receive takes the message        |
| `Sendrecv`, `Sendrecv_replace`, `sendrecv`          | a send, then a receive                            |
| `Iprobe`, `Probe`                                   | look for a matching message (`Probe` raises if none) |
| `Send_init`, `Recv_init`, `Prequest.Start`          | persistent requests: each `Start` posts again     |
| to or from `PROC_NULL`                              | nothing; a receive reports source `PROC_NULL`     |

On one process MPI would wait forever for a receive that nothing can match; the stand-in
raises `RuntimeError` instead, and withdraws the receive. `Wait`, `Waitall`, `Waitany` and
`Waitsome` raise the same way for a request that can never complete. `MPI.Status` objects get
the source, tag and size of the received message (`Get_count(MPI.DOUBLE)`, ...).

Buffers can be plain arrays, `[array, count, datatype]`, `[array, (count, displacement),
datatype]`, or use derived datatypes, e.g. a subarray for a halo slab:

```python
edge = MPI.DOUBLE.Create_subarray([ny, nx], [ny, 1], [0, nx - 2]).Commit()
halo = MPI.DOUBLE.Create_subarray([ny, nx], [ny, 1], [0, 0]).Commit()
comm.Sendrecv([grid, 1, edge], dest=0, recvbuf=[grid, 1, halo], source=0)  # periodic
```

Object messages are pickled when sent, as in mpi4py: the receiver gets a copy.

## Groups and topologies

- `Get_group()` is the group of rank 0. `MPI.Group.Translate_ranks`, `Incl`, `Excl`,
  `Compare`, `Union`, `Intersection` and `Difference` work on it and on `MPI.GROUP_EMPTY`.
  `Create(group)` and `Create_group(group)` return a new communicator (`COMM_NULL` for the
  empty group).
- `Create_cart(dims, periods)` takes dimensions that multiply to 1 and returns a
  `Cartcomm`: `Get_topo()`, `Get_coords`, `Get_cart_rank` (coordinates wrap along periodic
  dimensions), `Sub`, and `Shift`, which gives rank 0 along periodic dimensions and
  `PROC_NULL` along the others, so halo code needs no serial special case.
- `Split_type(MPI.COMM_TYPE_SHARED)` returns a new communicator; `MPI.Compute_dims(1, n)` is
  all ones.

## Datatypes

The named datatypes have their sizes (`MPI.DOUBLE.Get_size()` is 8). `Create_contiguous`,
`Create_vector` (non-negative strides) and `Create_subarray` (C or Fortran order) build
derived types with the right `size` and `extent`; `Commit` and `Free` do nothing. A derived
type selects elements of a buffer counted in the array's own elements, so it must be built
on the array's datatype.

## Files

`MPI.File.Open(comm, path, amode)` opens a file with the MPI access modes (`MODE_RDONLY`,
`MODE_WRONLY`, `MODE_RDWR`, `MODE_CREATE`, `MODE_EXCL`, `MODE_APPEND`,
`MODE_DELETE_ON_CLOSE`). `Set_view(disp, etype, filetype)` places the data as MPI does, for
example a subarray file type for this rank's block of a global array; offsets and file
pointers count elementary types. `Write_at`, `Read_at`, `Write`, `Read`, their `_all` and
non-blocking (`Iwrite_at`, ...) forms, `Seek`, `Get_position`, `Get_size`, `Set_size`,
`Preallocate`, `Sync` and `Close` (or a `with` block) work on host arrays; reading past the
end of the file leaves the rest of the buffer unchanged, and the status counts what was
read. Only the `"native"` data representation is supported.

## Ranks

Only rank 0 exists. A `root`, `dest` or `source` other than 0, `PROC_NULL` or `ANY_SOURCE`
raises `ValueError`.

## Communicators, requests and the module

- `Get_rank()` is 0, `Get_size()` is 1, and the attributes `rank` and `size` match.
- `Dup()`, `Clone()` and `Split()` return new serial communicators. `Split(MPI.UNDEFINED)`
  returns `COMM_NULL`.
- `Abort(code)` raises `SystemExit(code)`.
- Requests of collectives and sends are complete; receives complete when their message
  is sent (see [Point-to-point](#point-to-point)).
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

One-sided communication (`Win`), graph topologies (`Create_graph`, `Create_dist_graph`) and
neighbourhood collectives, intercommunicators, matched probes (`Mprobe`) and file data
representations other than `"native"` raise `AttributeError` (or `ValueError`). Messages to
or from ranks other than 0 raise `ValueError`.
