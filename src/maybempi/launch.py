"""Decide whether to use MPI, from the environment the launcher sets up.

Importing ``mpi4py.MPI`` calls ``MPI_Init``, which can take close to a second and
makes every collective cost something, even on one process. A plain
``python script.py`` should therefore not touch MPI, even when mpi4py is
installed. :func:`launched_under_mpi` tells, without importing mpi4py, whether
the process was started by ``mpirun``/``mpiexec``/``srun``. :func:`get_mpi`
returns ``mpi4py.MPI`` then, and the serial stand-in
:class:`~maybempi.serial.SerialMPI` otherwise, so one code path serves both::

    import maybempi

    MPI = maybempi.get_mpi()
    comm = MPI.COMM_WORLD
    total = comm.allreduce(local_total, op=MPI.SUM)  # local_total on one process
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any

from maybempi.serial import SerialComm, SerialMPI

__all__ = [
    "LAUNCHER_VARIABLES",
    "LOCAL_RANK_VARIABLES",
    "OVERRIDE_VARIABLE",
    "get_mpi",
    "is_serial",
    "launched_under_mpi",
    "launcher_variable",
    "local_rank",
]

#: Per-rank variables exported by the process managers behind common launchers.
#: Each is set only for processes started *by* a launcher. ``SLURM_PROCID`` is
#: deliberately absent: it is also set for the batch script of a plain
#: ``sbatch`` job, which is not an MPI launch (``srun`` exports the PMI/PMIx
#: variables).
LAUNCHER_VARIABLES = (
    "OMPI_COMM_WORLD_RANK",  # Open MPI (and derivatives)
    "PMI_RANK",  # MPICH, Intel MPI, MS-MPI, Cray, srun --mpi=pmi2
    "PMIX_RANK",  # PMIx: srun --mpi=pmix, Open MPI 5
    "MV2_COMM_WORLD_RANK",  # MVAPICH2
    "MPI_LOCALRANKID",  # Hydra (mpiexec.hydra)
    "ALPS_APP_PE",  # Cray ALPS aprun
    "PALS_RANKID",  # Cray PALS
)

#: Node-local rank of the process, as exported by common launchers. They are set
#: before ``MPI_Init``, so e.g. a GPU can be chosen before MPI starts.
LOCAL_RANK_VARIABLES = (
    "OMPI_COMM_WORLD_LOCAL_RANK",  # Open MPI
    "MV2_COMM_WORLD_LOCAL_RANK",  # MVAPICH2
    "MPI_LOCALRANKID",  # Intel MPI, MPICH (Hydra)
    "PMI_LOCAL_RANK",  # MPICH / PMI
    "PALS_LOCAL_RANKID",  # Cray PALS
    "SLURM_LOCALID",  # Slurm (srun)
    "LOCAL_RANK",  # torchrun and others
)

#: Environment variable that forces the decision of :func:`launched_under_mpi`
#: (``1``/``true``/``yes``/``on`` or ``0``/``false``/``no``/``off``).
OVERRIDE_VARIABLE = "MAYBEMPI"

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")

_SERIAL_MPI = SerialMPI()
_AUTO_MPI: Any = None  # the result of get_mpi(None), decided once per process


def _env_flag(name: str) -> bool | None:
    """Return the boolean value of the environment variable `name`, or None if unset or unknown."""
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return None


def local_rank() -> int:
    """Return the rank of this process within its node, from the launcher's environment.

    Reads the node-local rank that common launchers export (Open MPI, MVAPICH2,
    Intel MPI/MPICH, PMI, Cray PALS, Slurm, ``LOCAL_RANK``, see
    :data:`LOCAL_RANK_VARIABLES`). These are set before ``MPI_Init``, so this works
    before MPI is initialized and without importing mpi4py.

    Returns:
        The node-local rank, or 0 if no launcher variable is set (a serial run).
    """
    for variable in LOCAL_RANK_VARIABLES:
        value = os.environ.get(variable)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


def launcher_variable() -> str | None:
    """Return the first variable of :data:`LAUNCHER_VARIABLES` that is set, if any.

    Useful to see why :func:`launched_under_mpi` decided as it did.

    Returns:
        The name of the variable, or None if none is set.
    """
    return next((v for v in LAUNCHER_VARIABLES if v in os.environ), None)


def launched_under_mpi() -> bool:
    """Tell whether this process was started by an MPI launcher, without importing mpi4py.

    True if a per-rank variable of a common launcher is set (Open MPI, MPICH,
    Intel MPI, PMIx/``srun``, MVAPICH2, Hydra, Cray ALPS/PALS; see
    :data:`LAUNCHER_VARIABLES`), or if mpi4py is already imported and MPI
    initialized (using it then costs nothing more). ``MAYBEMPI=1``/``0``
    overrides the detection, e.g. for a launcher whose variables are not known.

    Returns:
        Whether the process belongs to an MPI job.
    """
    override = _env_flag(OVERRIDE_VARIABLE)
    if override is not None:
        return override
    if launcher_variable() is not None:
        return True
    # only look at mpi4py if the application imported it: importing it here is
    # what must be avoided
    mpi = sys.modules.get("mpi4py.MPI")
    if mpi is not None:
        try:
            return bool(mpi.Is_initialized())
        except AttributeError:
            return False
    return False


def get_mpi(use_mpi: bool | None = None) -> Any:
    """Return ``mpi4py.MPI`` for an MPI run, else the serial stand-in.

    Args:
        use_mpi: ``None`` (the default) decides with :func:`launched_under_mpi`,
            once per process. ``True`` imports mpi4py (``ImportError`` if it is
            not installed). ``False`` returns the stand-in without importing
            mpi4py.

    Returns:
        ``mpi4py.MPI``, or the one :class:`~maybempi.serial.SerialMPI` object,
        which has the attributes of the module that a serial run needs.
        :func:`is_serial` tells which one it is.

    Warns:
        RuntimeWarning: Launched under MPI, but mpi4py is not installed. Every
            process then runs as if it were alone, as rank 0 of 1.
    """
    global _AUTO_MPI
    if use_mpi is True:
        from mpi4py import MPI  # pyright: ignore[reportMissingImports]

        return MPI
    if use_mpi is False:
        return _SERIAL_MPI
    if _AUTO_MPI is None:
        if launched_under_mpi():
            try:
                from mpi4py import MPI as mpi  # pyright: ignore[reportMissingImports]
            except ImportError:
                warnings.warn(
                    "launched under an MPI launcher, but mpi4py is not installed: "
                    "every process runs serially as rank 0 of 1 (pip install mpi4py)",
                    RuntimeWarning,
                    stacklevel=2,
                )
                mpi = _SERIAL_MPI
            _AUTO_MPI = mpi
        else:
            _AUTO_MPI = _SERIAL_MPI
    return _AUTO_MPI


def is_serial(obj: Any) -> bool:
    """Tell whether `obj` is the serial stand-in (the module or a communicator).

    Args:
        obj: What :func:`get_mpi` returned, or a communicator such as
            ``MPI.COMM_WORLD``.

    Returns:
        True for :class:`~maybempi.serial.SerialMPI` and
        :class:`~maybempi.serial.SerialComm` objects, False for mpi4py's.
    """
    return isinstance(obj, (SerialMPI, SerialComm))
