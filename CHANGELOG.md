# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Point-to-point messages from rank 0 to itself: `Send`/`Recv`, `Isend`/`Irecv`,
  `Ssend`/`Issend`, `Bsend`/`Rsend`, the object versions (`send`, `recv`, `isend`, `irecv`,
  `ssend`, `issend`), `Sendrecv_replace`, `Iprobe`/`Probe` and persistent requests
  (`Send_init`, `Recv_init`, `Prequest.Start`). A send is kept, per communicator, until a
  receive with a matching tag takes it, in the order sent; a receive that no message can
  match raises `RuntimeError` where MPI would wait forever.
- Requests with state: a receive completes when its message is sent; `Waitall`, `Waitany`,
  `Waitsome`, `Testany`, `Testsome` and `testall`.
- Groups (`Get_group`, `MPI.Group` with `Translate_ranks`, `Incl`, `Excl`, `Compare`, ...),
  Cartesian communicators (`Create_cart`, `MPI.Cartcomm` with `Shift`, `Get_topo`,
  `Get_cart_rank`, `Sub`, ...), `Split_type`, `Create`, `Create_group` and
  `MPI.Compute_dims`.
- Datatype sizes and derived datatypes (`Create_contiguous`, `Create_vector`,
  `Create_subarray`), which select the elements of point-to-point buffers and file views.
- `MPI.File`: MPI-IO on one process, with views, explicit offsets and file pointers.
- `MPI.Status` records the source, tag and size of a received message.

### Changed

- `sendrecv`/`Sendrecv` go through the message queue: the object version returns a copy (as
  mpi4py does), and a receive from rank 0 with nothing sent (e.g. `dest=PROC_NULL,
  source=0`) raises `RuntimeError` instead of doing nothing. The default `source` and
  `recvtag` are `ANY_SOURCE` and `ANY_TAG`, as in mpi4py.

### Fixed

- `maybempi.__version__` is read from the installed package metadata (the version in
  `pyproject.toml`) instead of a second hard-coded copy; 0.1.1 reported 0.1.0. `CITATION.cff`
  says 0.1.1.

## [0.1.1] - 2026-10-06

### Added

- `SerialComm.Alltoallv`.

## [0.1.0] - 2026-10-06

### Added

- `get_mpi()`: `mpi4py.MPI` when the process was started by an MPI launcher, a serial stand-in
  otherwise, decided once per process without importing mpi4py. `from maybempi import MPI` is
  the drop-in replacement for `from mpi4py import MPI`.
- `launched_under_mpi()`, `launcher_variable()` and `local_rank()`, from the environment
  variables of Open MPI, MPICH, Intel MPI, PMIx/`srun`, MVAPICH2, Hydra and Cray ALPS/PALS. A
  plain `sbatch` batch script is not treated as an MPI launch. `MAYBEMPI=0`/`1` overrides the
  decision.
- `SerialMPI` and `SerialComm`: the module constants and communicator methods a serial run
  needs, with the results MPI gives on one process. Unsupported methods raise `AttributeError`.
  Buffers may be NumPy arrays, device arrays with `.get()` (such as CuPy) or mpi4py buffer
  specifications; `set_copy_hook()` reports host-device copies. Inside an MPI job, a
  `SerialComm` accepts mpi4py's `IN_PLACE`, `PROC_NULL` and `ANY_SOURCE` too.
- `is_serial()` to tell the stand-in from mpi4py.
- The `maybempi` command (and `python -m maybempi`), which prints the decision and why.

The code comes from `cunumpy.mpi` (cunumpy 0.5) and `feectools.ddm.mpi`, where the override
variables were `CUNUMPY_MPI`, `STRUPHY_MPI` and `FEECTOOLS_MPI`.
