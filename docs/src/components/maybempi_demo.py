"""The Python side of the live demo: one code path, split over ranks or run serially.

It runs in the browser (Pyodide), where maybempi decides on the serial stand-in. ``run`` takes
the values of the sliders, splits ``n`` cells over ``ranks`` ranks with
:func:`maybempi.collectives.block`, integrates 4 / (1 + x^2) over each rank's block (the
midpoint rule, pi in total) and sums the parts with the stand-in's ``allreduce``.
"""

import numpy as np

import maybempi
from maybempi.collectives import allsum, block


class _Rank:
    """Rank ``rank`` of ``size``: what :func:`block` asks of a communicator."""

    def __init__(self, rank, size):
        self._rank, self._size = rank, size

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size


def _part(n, comm):
    """One rank's share of the midpoint rule for pi on n cells."""
    cells = block(n, comm)
    x = (np.arange(cells.start, cells.stop) + 0.5) / n
    return float(np.sum(4.0 / (1.0 + x * x)) / n), cells


def run(n=40, ranks=4):
    """Split ``n`` cells over ``ranks`` ranks and add up their parts of pi.

    Args:
        n: Number of cells of the midpoint rule.
        ranks: Number of ranks the cells are split over.

    Returns:
        The cells of each rank as a bar chart, and the result.
    """
    n, ranks = int(n), int(ranks)
    MPI = maybempi.get_mpi()  # the serial stand-in: no mpirun in a browser
    comm = MPI.COMM_WORLD
    parts, shares = [], []
    for r in range(ranks):  # under mpirun each process computes only its own r
        part, cells = _part(n, _Rank(r, ranks))
        parts.append(part)
        shares.append(cells)
    total = allsum(comm, sum(parts))  # one rank here: allreduce returns the value itself
    return {
        "figure": {
            "data": [
                {
                    "type": "bar",
                    "orientation": "h",
                    "y": [f"rank {r}" for r in range(ranks)],
                    "x": [s.stop - s.start for s in shares],
                    "base": [s.start for s in shares],
                    "text": [f"{s.start}–{s.stop - 1}" if s.stop > s.start else "" for s in shares],
                    "textposition": "inside",
                    "name": "cells",
                }
            ],
            "layout": {
                "xaxis": {"title": {"text": "cell index"}, "range": [0, max(n, 1)]},
                "yaxis": {"autorange": "reversed"},
                "showlegend": False,
            },
        },
        "numbers": {
            "π from the parts": total,
            "Error": abs(total - np.pi),
            "Launched under MPI": str(maybempi.launched_under_mpi()),
            "Size of COMM_WORLD": comm.Get_size(),
        },
    }
