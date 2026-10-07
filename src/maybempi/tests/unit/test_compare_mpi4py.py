"""The stand-in's point-to-point, topology, datatype and file calls against mpi4py's COMM_SELF."""

import subprocess
import sys

import pytest

_SCRIPT = r"""
import os
import sys
import tempfile

import numpy as np
from mpi4py import MPI as real

from maybempi import get_mpi

serial = get_mpi(False)


def run(MPI, comm):
    out = {}
    recv = np.zeros(3)
    request = comm.Irecv(recv, source=0, tag=4)
    comm.Send(np.arange(3.0), dest=0, tag=4)
    status = MPI.Status()
    request.Wait(status)
    out["p2p"] = (recv.tolist(), status.Get_source(), status.Get_tag(),
                  status.Get_count(MPI.DOUBLE))
    # non-blocking sends: a blocking send to oneself before the receive is
    # posted may wait forever in MPI (MPICH does)
    sends = [comm.isend({"a": 1}, dest=0, tag=2), comm.isend("b", dest=0, tag=1)]
    out["objects"] = (comm.recv(source=0, tag=1), comm.recv(source=0, tag=2))
    MPI.Request.waitall(sends)
    out["iprobe"] = comm.Iprobe(source=0, tag=9)
    grid = np.arange(20.0).reshape(4, 5)
    edge = MPI.DOUBLE.Create_subarray([4, 5], [4, 1], [0, 3]).Commit()
    halo = MPI.DOUBLE.Create_subarray([4, 5], [4, 1], [0, 0]).Commit()
    comm.Sendrecv([grid, 1, edge], dest=0, recvbuf=[grid, 1, halo], source=0)
    out["subarray"] = grid[:, 0].tolist()
    vector = MPI.DOUBLE.Create_vector(3, 2, 4).Commit()
    out["types"] = (edge.Get_size(), edge.Get_extent(), vector.Get_size(),
                    vector.Get_extent(), MPI.DOUBLE.Create_contiguous(3).Get_extent())
    part = np.zeros(6)
    comm.Sendrecv([np.arange(12.0), 1, vector], dest=0, recvbuf=part, source=0)
    out["vector"] = part.tolist()
    cart = comm.Create_cart([1, 1], periods=[True, False])
    # the value of PROC_NULL depends on the MPI library (-1 in MPICH, -2 in Open MPI)
    shifts = [["PROC_NULL" if r == MPI.PROC_NULL else r for r in cart.Shift(d, 1)]
              for d in (0, 1)]
    out["cart"] = (cart.Get_topo(), shifts, cart.Get_cart_rank([0, 0]),
                   cart.Get_topology() == MPI.CART)
    group = comm.Get_group()
    out["group"] = (group.Get_size(), group.Get_rank(),
                    MPI.Group.Translate_ranks(group, [0], cart.Get_group()))
    path = os.path.join(tempfile.mkdtemp(), "data.bin")
    handle = MPI.File.Open(comm, path, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    block = MPI.DOUBLE.Create_subarray([4, 5], [2, 3], [1, 1]).Commit()
    # bytes nobody writes are undefined in MPI-IO (MPICH leaves 0xff in the gaps of
    # the view), so the whole region is written once first
    handle.Write_at(0, np.zeros(1 + 4 * 5))
    handle.Set_view(8, MPI.DOUBLE, block)
    handle.Write_all(np.arange(6.0) + 1)
    handle.Close()
    with open(path, "rb") as file:
        out["file"] = file.read()
    for datatype in (edge, halo, vector, block):
        datatype.Free()
    return out


expected = run(real, real.COMM_SELF)
got = run(serial, serial.COMM_SELF)
for key in expected:
    assert expected[key] == got[key], (key, expected[key], got[key])
print("same results")
"""


def test_point_to_point_topology_and_io_match_mpi4py():
    """Runs in a child process: MPI_Init inside pytest is fragile with some MPI libraries."""
    pytest.importorskip("mpi4py")
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "same results" in result.stdout
