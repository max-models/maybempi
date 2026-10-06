"""The ``maybempi`` command: show what maybempi decides in this environment.

Run it the way the application is run, to see whether it would use MPI::

    maybempi                  # serial
    mpirun -n 2 maybempi      # each rank reports its launcher variable
    mpirun -n 2 maybempi --init   # also start MPI and print rank and size
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import socket

from maybempi import __version__
from maybempi.launch import (
    OVERRIDE_VARIABLE,
    get_mpi,
    is_serial,
    launched_under_mpi,
    launcher_variable,
    local_rank,
)


def report(init: bool = False) -> str:
    """Describe the decision of :func:`~maybempi.launch.launched_under_mpi` here.

    Args:
        init: Also call :func:`~maybempi.launch.get_mpi` (which starts MPI under a
            launcher) and add the rank and size of ``COMM_WORLD``.

    Returns:
        One line per item, ``name: value``.
    """
    mpi4py_installed = importlib.util.find_spec("mpi4py") is not None
    lines = {
        "maybempi": __version__,
        "host": socket.gethostname(),
        "launched under MPI": launched_under_mpi(),
        "launcher variable": launcher_variable() or "-",
        f"{OVERRIDE_VARIABLE} override": os.environ.get(OVERRIDE_VARIABLE, "-"),
        "local rank": local_rank(),
        "mpi4py installed": mpi4py_installed,
    }
    if init:
        mpi = get_mpi()
        comm = mpi.COMM_WORLD
        lines["MPI"] = "serial stand-in" if is_serial(mpi) else "mpi4py"
        lines["rank"] = f"{comm.Get_rank()} of {comm.Get_size()}"
    return "\n".join(f"{name}: {value}" for name, value in lines.items())


def main(argv: list[str] | None = None) -> None:
    """Print :func:`report`; the entry point of the ``maybempi`` command."""
    parser = argparse.ArgumentParser(
        prog="maybempi",
        description="Show whether this process would use MPI, and why.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="also start MPI (under a launcher) and print the rank and size",
    )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    print(report(init=args.init), flush=True)


if __name__ == "__main__":
    main()
