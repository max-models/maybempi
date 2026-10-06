"""Use MPI only when the process was launched under MPI, and a serial stand-in otherwise.

Importing ``mpi4py.MPI`` starts MPI, which costs close to a second and makes every
collective cost something, even on one process. maybempi decides from the
environment the launcher sets up, without importing mpi4py, and returns either
``mpi4py.MPI`` or a serial stand-in with the same interface::

    import maybempi

    MPI = maybempi.get_mpi()  # mpi4py.MPI under mpirun/srun, else the stand-in
    comm = MPI.COMM_WORLD
    total = comm.allreduce(local_total, op=MPI.SUM)

``from maybempi import MPI`` is the drop-in replacement for ``from mpi4py import
MPI``: the name resolves to ``get_mpi()`` when it is first imported.

:mod:`maybempi.launch` holds the detection, :mod:`maybempi.serial` the stand-in.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

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

try:
    __version__ = version("maybempi")  # the version in pyproject.toml
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"

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


def __getattr__(name: str) -> Any:
    """Resolve ``maybempi.MPI`` (and ``from maybempi import MPI``) to :func:`get_mpi`."""
    if name == "MPI":
        return get_mpi()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
