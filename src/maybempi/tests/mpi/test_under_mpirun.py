"""Tests that run under a real MPI launcher, e.g. ``mpiexec -n 2 python -m pytest``.

Outside a launcher they are skipped, except :func:`test_mpiexec`, which starts
``mpiexec`` itself when it and mpi4py are installed.
"""

import importlib.util
import shutil
import subprocess
import sys

import numpy as np
import pytest

import maybempi

under_mpi = pytest.mark.skipif(
    not maybempi.launched_under_mpi(), reason="not launched under an MPI launcher"
)
has_mpi = importlib.util.find_spec("mpi4py") is not None and shutil.which("mpiexec")


@under_mpi
def test_get_mpi_returns_mpi4py():
    MPI = maybempi.get_mpi()
    assert not maybempi.is_serial(MPI)
    assert MPI.Is_initialized()
    assert maybempi.launcher_variable() is not None


@under_mpi
def test_collectives_across_ranks():
    MPI = maybempi.get_mpi()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    assert comm.allreduce(rank, op=MPI.SUM) == size * (size - 1) // 2
    recv = np.zeros(1)
    comm.Allreduce(np.array([1.0]), recv, op=MPI.SUM)
    assert recv[0] == size


@under_mpi
def test_local_rank_matches_the_node_communicator():
    MPI = maybempi.get_mpi()
    node = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    assert maybempi.local_rank() == node.Get_rank()


@pytest.mark.skipif(maybempi.launched_under_mpi(), reason="already under a launcher")
@pytest.mark.skipif(not has_mpi, reason="needs mpiexec and mpi4py")
def test_mpiexec():
    """``mpiexec -n 2 maybempi --init``: rank 0 prints one table for both ranks."""
    command = ["mpiexec", "-n", "2"]
    help_text = subprocess.run(
        ["mpiexec", "--help"], capture_output=True, text=True, check=False
    ).stdout
    if "--oversubscribe" in help_text:  # Open MPI: allow more ranks than cores
        command.append("--oversubscribe")
    # a session of its own: the launcher may signal its process group when it ends
    result = subprocess.run(
        [*command, sys.executable, "-m", "maybempi", "--init"],
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    assert result.returncode == 0, (
        f"{' '.join(command)} exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    common, per_rank = result.stdout.strip().split("\n\n")
    assert result.stdout.count("item ") == 1, result.stdout  # one table, from rank 0
    common_rows = [line.split() for line in common.splitlines()]
    assert ["MPI", "mpi4py"] in common_rows and ["size", "2"] in common_rows
    ranks = [line.split()[0] for line in per_rank.splitlines()[2:]]
    assert ranks == ["0", "1"], result.stdout
