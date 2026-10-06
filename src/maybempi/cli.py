"""The ``maybempi`` command: show what maybempi decides in this environment.

Run it the way the application is run, to see whether it would use MPI::

    maybempi                      # serial: one table
    mpirun -n 2 maybempi          # one table per rank (MPI is not started)
    mpirun -n 2 maybempi --init   # start MPI: rank 0 prints one table for all ranks
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import socket
from collections.abc import Sequence
from typing import Any

from maybempi import __version__
from maybempi.launch import (
    OVERRIDE_VARIABLE,
    get_mpi,
    is_serial,
    launched_under_mpi,
    launcher_variable,
    local_rank,
)

# columns of the per-rank table, in this order; other items become columns only
# if they differ between ranks
_RANK_COLUMNS = ("rank", "host", "local rank", "launcher variable")


def info(init: bool = False) -> dict[str, Any]:
    """Collect what maybempi decides in this process, and why.

    Args:
        init: Also call :func:`~maybempi.launch.get_mpi` (which starts MPI under a
            launcher) and add the rank and size of ``COMM_WORLD``.

    Returns:
        The items of the report, by name.
    """
    items: dict[str, Any] = {
        "maybempi": __version__,
        "host": socket.gethostname(),
        "launched under MPI": launched_under_mpi(),
        "launcher variable": launcher_variable() or "-",
        f"{OVERRIDE_VARIABLE} override": os.environ.get(OVERRIDE_VARIABLE, "-"),
        "local rank": local_rank(),
        "mpi4py installed": importlib.util.find_spec("mpi4py") is not None,
    }
    if init:
        mpi = get_mpi()
        comm = mpi.COMM_WORLD
        items["MPI"] = "serial stand-in" if is_serial(mpi) else "mpi4py"
        items["rank"] = comm.Get_rank()
        items["size"] = comm.Get_size()
    return items


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Format `rows` as a plain-text table with aligned columns."""
    cells = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(str(h)), *(len(row[i]) for row in cells)) for i, h in enumerate(header)
    ]

    def line(values: Sequence[str]) -> str:
        return "  ".join(
            v.ljust(w) for v, w in zip(values, widths, strict=True)
        ).rstrip()

    return "\n".join(
        [line(header), line(["-" * w for w in widths]), *(line(row) for row in cells)]
    )


def format_report(items: dict[str, Any]) -> str:
    """Format the report of one process as a two-column table.

    Args:
        items: What :func:`info` returned.

    Returns:
        The table, one item per line.
    """
    items = dict(items)
    if "rank" in items:
        items["rank"] = f"{items['rank']} of {items.pop('size')}"
    return _table(("item", "value"), list(items.items()))


def format_reports(reports: Sequence[dict[str, Any]]) -> str:
    """Format the reports of all ranks: the common items, then one row per rank.

    Args:
        reports: What :func:`info` returned on each rank, in rank order.

    Returns:
        A table of the items that are the same on every rank, and a table with one
        row per rank for the others (rank, host, local rank, launcher variable and
        whatever else differs).
    """
    names = list(reports[0])
    differs = {n for n in names if any(r.get(n) != reports[0].get(n) for r in reports)}
    columns = [n for n in _RANK_COLUMNS if n in names]
    columns += [n for n in names if n in differs and n not in columns]
    common = [(n, reports[0][n]) for n in names if n not in columns]
    rows = [[report.get(n, "-") for n in columns] for report in reports]
    return f"{_table(('item', 'value'), common)}\n\n{_table(columns, rows)}"


def report(init: bool = False) -> str:
    """Return the report of this process as a table (see :func:`info`)."""
    return format_report(info(init))


def main(argv: list[str] | None = None) -> None:
    """Print the report; the entry point of the ``maybempi`` command."""
    parser = argparse.ArgumentParser(
        prog="maybempi",
        description="Show whether this process would use MPI, and why.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="also start MPI (under a launcher); rank 0 then prints one table for all ranks",
    )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    items = info(init=args.init)
    if args.init and items["size"] > 1:
        reports = get_mpi().COMM_WORLD.gather(items, root=0)
        if items["rank"] == 0:
            print(format_reports(reports), flush=True)
        return
    print(format_report(items), flush=True)


if __name__ == "__main__":
    main()
