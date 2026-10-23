"""Hand a maybempi communicator to PETSc.

petsc4py takes mpi4py communicators, but not maybempi's serial stand-in.
:func:`petsc_comm` turns either into a ``petsc4py.PETSc.Comm``: the stand-in, and
any communicator of one process, become ``PETSc.COMM_SELF``, so a serial run
never imports mpi4py; larger mpi4py communicators are wrapped with
``PETSc.Comm(comm)``.

petsc4py is optional: it is imported only when :func:`petsc_comm` is called.
"""

from __future__ import annotations

from typing import Any

from maybempi.launch import get_mpi, is_serial

_INSTALL_HINT = (
    "petsc_comm needs petsc4py: `pip install petsc4py` "
    "(or `conda install -c conda-forge petsc4py`)"
)


def _petsc() -> Any:
    """``petsc4py.PETSc``, or an ImportError that says how to get it."""
    try:
        from petsc4py import PETSc  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise ImportError(_INSTALL_HINT) from error
    return PETSc


def petsc_comm(comm: Any = None) -> Any:
    """Return the ``petsc4py.PETSc.Comm`` for a maybempi or mpi4py communicator.

    Importing ``petsc4py.PETSc`` initializes PETSc (and, with a PETSc built
    against MPI, MPI itself), so call this only when PETSc is needed.

    Args:
        comm: The communicator: the serial stand-in's, an mpi4py communicator,
            or a ``PETSc.Comm`` (returned as it is). ``None`` means
            ``COMM_WORLD`` of :func:`~maybempi.get_mpi`.

    Returns:
        ``PETSc.COMM_SELF`` for the serial stand-in and for any communicator of
        one process, otherwise ``PETSc.Comm(comm)``, which shares the MPI
        communicator (it is not duplicated).

    Raises:
        ImportError: If petsc4py is not installed.
    """
    PETSc = _petsc()  # noqa: N806 - PETSc's own name
    if comm is None:
        comm = get_mpi().COMM_WORLD
    if isinstance(comm, PETSc.Comm):
        return comm
    if is_serial(comm) or comm.Get_size() == 1:
        return PETSc.COMM_SELF
    return PETSc.Comm(comm)
