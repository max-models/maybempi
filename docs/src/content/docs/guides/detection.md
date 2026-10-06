---
title: How the decision is made
description: Launcher variables, the MAYBEMPI override, Slurm, and when the decision is taken.
---

`maybempi.get_mpi()` imports mpi4py only if `maybempi.launched_under_mpi()` is true. That
function never imports mpi4py: it reads the environment that the launcher exports to each
process it starts.

## Launcher variables

A process belongs to an MPI job if one of these variables is set
(`maybempi.LAUNCHER_VARIABLES`):

| Variable               | Set by                                          |
| ---------------------- | ----------------------------------------------- |
| `OMPI_COMM_WORLD_RANK` | Open MPI and its derivatives                    |
| `PMI_RANK`             | MPICH, Intel MPI, MS-MPI, Cray, `srun --mpi=pmi2` |
| `PMIX_RANK`            | PMIx: `srun --mpi=pmix`, Open MPI 5             |
| `MV2_COMM_WORLD_RANK`  | MVAPICH2                                        |
| `MPI_LOCALRANKID`      | Hydra (`mpiexec.hydra`)                         |
| `ALPS_APP_PE`          | Cray ALPS `aprun`                               |
| `PALS_RANKID`          | Cray PALS                                       |

`maybempi.launcher_variable()` returns the one that was found, and the `maybempi` command
prints it.

The process is also treated as an MPI process if the application imported mpi4py itself and
MPI is initialized: using it then costs nothing more.

## Slurm

`SLURM_PROCID` is deliberately not in the list. Slurm also sets it for the batch script of a
plain `sbatch` job, which is not an MPI launch: a script that runs `python serial.py` inside a
job allocation stays serial. Processes started by `srun` get the PMI or PMIx variables from
Slurm's MPI plugin, and are recognized through those.

## Override

`MAYBEMPI` forces the decision, for a launcher whose variables are not in the list, or to keep
a launched process serial:

```bash
MAYBEMPI=1 my-launcher -n 4 python script.py   # use MPI
MAYBEMPI=0 mpirun -n 4 python script.py         # every rank serial (rank 0 of 1)
```

`1`, `true`, `yes` and `on` mean MPI; `0`, `false`, `no` and `off` mean serial. Other values are
ignored. In code, `get_mpi(True)` always imports mpi4py, and `get_mpi(False)` always returns the
stand-in.

## Once per process

`get_mpi()` decides the first time it is called and returns the same object afterwards, so all
parts of a program agree on whether MPI is used. Libraries can call it at import time.

If a launcher is detected but mpi4py is not installed, `get_mpi()` warns with a
`RuntimeWarning` and returns the stand-in: every process then runs as rank 0 of 1.

## Node-local rank

`maybempi.local_rank()` reads the rank of the process within its node from
`OMPI_COMM_WORLD_LOCAL_RANK`, `MV2_COMM_WORLD_LOCAL_RANK`, `MPI_LOCALRANKID`, `PMI_LOCAL_RANK`,
`PALS_LOCAL_RANKID`, `SLURM_LOCALID` or `LOCAL_RANK` (`maybempi.LOCAL_RANK_VARIABLES`), and
returns 0 if none is set. These are set before `MPI_Init`, so a code can choose its GPU before
MPI starts.
