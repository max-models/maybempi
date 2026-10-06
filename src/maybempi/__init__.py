"""Use MPI only when the process was launched under MPI, and a serial stand-in otherwise.

Importing ``mpi4py.MPI`` starts MPI, which costs close to a second and makes every
collective cost something, even on one process. maybempi decides from the
environment the launcher sets up, without importing mpi4py, and returns either
``mpi4py.MPI`` or a serial stand-in with the same interface::

    import maybempi

    MPI = maybempi.get_mpi()  # mpi4py.MPI under mpirun/srun, else the stand-in
    comm = MPI.COMM_WORLD
    total = comm.allreduce(local_total, op=MPI.SUM)

:mod:`maybempi.launch` holds the detection, :mod:`maybempi.serial` the stand-in.
"""

from maybempi.launch import (
    LAUNCHER_VARIABLES,
    LOCAL_RANK_VARIABLES,
    OVERRIDE_VARIABLE,
    get_mpi,
    is_serial,
    launched_under_mpi,
    launcher_variable,
    local_rank,
)
from maybempi.serial import (
    SerialComm,
    SerialMPI,
    SerialPrequest,
    SerialRequest,
    SerialStatus,
    set_copy_hook,
)

__version__ = "0.1.0"

__all__ = [
    "LAUNCHER_VARIABLES",
    "LOCAL_RANK_VARIABLES",
    "OVERRIDE_VARIABLE",
    "SerialComm",
    "SerialMPI",
    "SerialPrequest",
    "SerialRequest",
    "SerialStatus",
    "__version__",
    "get_mpi",
    "is_serial",
    "launched_under_mpi",
    "launcher_variable",
    "local_rank",
    "set_copy_hook",
]
