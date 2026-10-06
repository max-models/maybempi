"""Tests for the serial stand-in: SerialMPI, SerialComm and the copy hook."""

import numpy as np
import pytest

from maybempi import get_mpi, set_copy_hook
from maybempi.serial import SerialMPI

MPI = get_mpi(False)
comm = MPI.COMM_WORLD


def test_queries():
    assert comm.rank == 0 and comm.size == 1
    assert comm.Get_rank() == 0 and comm.Get_size() == 1
    assert isinstance(comm, MPI.Comm) and isinstance(comm, MPI.Intracomm)
    assert comm.Is_intra() and not comm.Is_inter()
    assert MPI.COMM_SELF.Get_size() == 1 and MPI.COMM_SELF.Get_name() == "COMM_SELF"
    assert repr(comm) == "SerialComm(COMM_WORLD)"


def test_object_collectives_return_the_value():
    value = {"a": 1}
    assert comm.bcast(value) is value
    assert comm.allreduce(5, op=MPI.SUM) == 5
    assert comm.allreduce(2.5, op=MPI.MAX) == 2.5
    assert comm.reduce(7) == 7
    assert comm.scan(3) == 3
    assert comm.exscan(3) is None  # as mpi4py on rank 0
    assert comm.gather(value) == [value]
    assert comm.allgather(value) == [value]
    assert comm.scatter([value]) is value
    assert comm.alltoall([value]) == [value]
    assert comm.sendrecv(value, dest=0, source=0) is value
    assert comm.ibcast(value).wait() is value
    assert comm.iallreduce(4).wait() == 4
    assert comm.Barrier() is None and comm.barrier() is None


def test_object_collectives_check_sizes_and_ranks():
    with pytest.raises(ValueError, match="1 item"):
        comm.scatter([1, 2])
    with pytest.raises(ValueError, match="1 item"):
        comm.alltoall([1, 2])
    with pytest.raises(ValueError, match="only rank 0"):
        comm.bcast(1, root=1)
    with pytest.raises(ValueError, match="only rank 0"):
        comm.sendrecv(1, dest=1)
    assert comm.sendrecv(1, dest=MPI.PROC_NULL, source=MPI.PROC_NULL) is None
    assert comm.sendrecv(1, dest=MPI.PROC_NULL, source=0) is None


def test_unknown_methods_raise():
    # a stand-in returning None for everything hides missing support
    with pytest.raises(AttributeError):
        _ = comm.Create_cart  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = MPI.Win  # type: ignore[attr-defined]


def test_buffer_collectives_copy():
    send = np.arange(6.0).reshape(2, 3)
    for call in (
        lambda r: comm.Allreduce(send, r, op=MPI.SUM),
        lambda r: comm.Reduce(send, r, op=MPI.SUM, root=0),
        lambda r: comm.Allgather(send, r),
        lambda r: comm.Gather(send, r),
        lambda r: comm.Scatter(send, r),
        lambda r: comm.Alltoall(send, r),
        lambda r: comm.Scan(send, r),
        lambda r: comm.Sendrecv(send, dest=0, recvbuf=r, source=0),
        lambda r: comm.Iallreduce(send, r).Wait(),
        lambda r: comm.Iallgather(send, r).Wait(),
    ):
        recv = np.zeros_like(send)
        call(recv)
        np.testing.assert_array_equal(recv, send)


def test_buffer_specs_and_in_place():
    data = np.arange(4.0)
    recv = np.zeros(4)
    comm.Allreduce([data, MPI.DOUBLE], [recv, MPI.DOUBLE], op=MPI.SUM)
    np.testing.assert_array_equal(recv, data)
    before = data.copy()
    comm.Allreduce(MPI.IN_PLACE, data, op=MPI.SUM)
    comm.Bcast(data, root=0)
    comm.Ibcast(data).Wait()
    comm.Scatter(None, MPI.IN_PLACE)
    comm.Scatterv(None, MPI.IN_PLACE)
    np.testing.assert_array_equal(data, before)
    comm.Exscan(data, recv)  # undefined on rank 0: untouched
    np.testing.assert_array_equal(recv, before)


def test_vector_collectives_use_the_displacement():
    send = np.array([1.0, 2.0])
    recv = np.zeros(5)
    comm.Allgatherv(send, [recv, [2], [3], MPI.DOUBLE])
    np.testing.assert_array_equal(recv, [0, 0, 0, 1, 2])
    recv[:] = 0
    comm.Gatherv(send, [recv, [2], [1]])
    np.testing.assert_array_equal(recv, [0, 1, 2, 0, 0])
    recv[:] = 0
    comm.Gatherv(send, [recv, [2], None])
    np.testing.assert_array_equal(recv, [1, 2, 0, 0, 0])
    part = np.zeros(2)
    comm.Scatterv([np.arange(5.0), [2], [2], MPI.DOUBLE], part)
    np.testing.assert_array_equal(part, [2, 3])


def test_alltoallv_copies_the_part_for_rank_0():
    send = np.arange(5.0)
    recv = np.zeros(5)
    comm.Alltoallv([send, [3], [1], MPI.DOUBLE], [recv, [3], [2], MPI.DOUBLE])
    np.testing.assert_array_equal(recv, [0, 0, 1, 2, 3])
    recv[:] = 0
    comm.Alltoallv([send, [5], None, None], [recv, [5], None, None])
    np.testing.assert_array_equal(recv, send)
    recv[:] = 7
    comm.Alltoallv(MPI.IN_PLACE, [recv, [5], None, None])
    np.testing.assert_array_equal(recv, 7)


def test_buffer_errors():
    with pytest.raises(ValueError, match="too small"):
        comm.Allreduce(np.ones(3), np.zeros(2))
    with pytest.raises(ValueError, match="C-contiguous"):
        comm.Allreduce(np.ones(2), np.zeros((2, 2))[:, 0])
    with pytest.raises(ValueError, match="only rank 0"):
        comm.Sendrecv(np.ones(2), dest=1, recvbuf=np.zeros(2))
    comm.Sendrecv(np.ones(2), dest=MPI.PROC_NULL, recvbuf=np.zeros(2))


class _DeviceArray:
    """Duck-typed device array (like CuPy): `.get()` copies it to the host."""

    def __init__(self, data):
        self._data = np.asarray(data)
        self.flags = self._data.flags

    def get(self):
        return self._data.copy()

    def reshape(self, *shape):
        return _DeviceArray(self._data.reshape(*shape))

    def __setitem__(self, key, value):
        self._data[key] = value

    @property
    def size(self):
        return self._data.size


def test_device_arrays_and_the_copy_hook():
    copies = []
    set_copy_hook(lambda kind, array: copies.append((kind, array.size)))
    try:
        recv = np.zeros(3)
        comm.Allreduce(_DeviceArray([1.0, 2.0, 3.0]), recv)
        np.testing.assert_array_equal(recv, [1, 2, 3])
        device = _DeviceArray(np.zeros(3))
        comm.Allgather(np.ones(3), device)
        np.testing.assert_array_equal(device.get(), [1, 1, 1])
        comm.Allreduce(np.ones(3), np.zeros(3))  # host to host: not reported
    finally:
        set_copy_hook(None)
    assert copies == [("to_host", 3), ("to_device", 3)]
    comm.Allreduce(_DeviceArray([1.0]), np.zeros(1))  # no hook: nothing recorded
    assert len(copies) == 2


def test_split_dup_abort_and_requests():
    assert comm.Split(0, 0).Get_size() == 1
    assert comm.Split(MPI.UNDEFINED) is MPI.COMM_NULL
    assert comm.Dup().Get_rank() == 0 and comm.Clone().Get_size() == 1
    assert comm.Free() is None
    with pytest.raises(SystemExit):
        comm.Abort(3)
    requests = [comm.Ibarrier(), comm.Ibcast(np.zeros(1))]
    assert MPI.Request.Waitall(requests) is None
    assert MPI.Request.Testall(requests) is True
    assert (
        MPI.Request.Waitany(requests) == 0 and MPI.Request.Waitany([]) == MPI.UNDEFINED
    )
    assert MPI.Request.waitall([comm.ibcast(1)]) == [1]
    request = comm.iallreduce(2)
    assert request.Test() and request.test() == (True, 2) and request.Wait() is None
    assert request.Free() is None and request.Cancel() is None
    status = MPI.Status()
    assert status.Get_source() == 0 and status.Get_tag() == 0
    assert status.Get_count(MPI.DOUBLE) == 0


def test_module_functions():
    assert MPI.Is_initialized() is False and MPI.Is_finalized() is False
    assert MPI.Init() is None and MPI.Finalize() is None
    assert MPI.Wtime() > 0 and MPI.Wtick() > 0
    assert isinstance(MPI.Get_processor_name(), str)
    assert MPI.Query_thread() == 0
    assert repr(MPI.SUM) == "SerialMPI.SUM"
    assert "serial stand-in" in repr(MPI)
    assert MPI.PROC_NULL == -2 and MPI.ANY_SOURCE == -1 and MPI.ROOT == -3
    assert isinstance(MPI, SerialMPI)


def test_constants_and_handles():
    assert isinstance(MPI.DOUBLE, MPI.Datatype)
    assert not isinstance(MPI.SUM, MPI.Datatype)
    assert isinstance(MPI.LOR, MPI.Op)
    assert MPI._typedict[np.dtype(np.float64).char] is MPI.DOUBLE
    # null handles are false, communicators true, like in mpi4py
    assert not MPI.COMM_NULL and not MPI.DATATYPE_NULL
    assert comm and comm != MPI.COMM_NULL
    MPI.Prequest.Startall([])
    assert MPI.Prequest.Waitall([]) is None
    assert MPI.Prequest().Start() is None
    # usable in annotations evaluated at definition time
    assert (MPI.Intracomm | None) is not None


_COMPARE_WITH_MPI4PY = """
import numpy as np
from mpi4py import MPI as real

from maybempi import get_mpi

MPI = get_mpi(False)
comm, self_comm = MPI.COMM_WORLD, real.COMM_SELF
for call in (
    lambda c: c.allreduce(5), lambda c: c.reduce(5), lambda c: c.scan(5),
    lambda c: c.gather(3), lambda c: c.allgather(3), lambda c: c.scatter([4]),
    lambda c: c.alltoall([4]), lambda c: c.exscan(3),
    lambda c: c.sendrecv(6, dest=0, source=0), lambda c: c.Get_rank(), lambda c: c.Get_size(),
):
    assert call(self_comm) == call(comm)
for name in ("Allreduce", "Reduce", "Scan", "Allgather", "Gather", "Alltoall"):
    send = np.arange(3.0)
    recv_real, recv_serial = np.zeros(3), np.zeros(3)
    getattr(self_comm, name)(send, recv_real)
    getattr(comm, name)(send, recv_serial)
    np.testing.assert_array_equal(recv_real, recv_serial, err_msg=name)
print("same results")
"""  # fmt: skip


def test_matches_mpi4py_on_one_process():
    """The same calls on mpi4py's COMM_SELF give the same results (if mpi4py is there).

    Runs in a child process: starting MPI inside pytest (a singleton MPI_Init) is
    fragile with some MPI libraries (MPICH exits with an error code at shutdown).
    """
    import subprocess
    import sys

    pytest.importorskip("mpi4py")
    result = subprocess.run(
        [sys.executable, "-c", _COMPARE_WITH_MPI4PY],
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "same results" in result.stdout


def test_mpi4py_constants_next_to_real_mpi(monkeypatch):
    """A SerialComm used in an MPI job gets mpi4py's constants (MPICH: PROC_NULL is -1)."""
    import sys
    import types

    real = types.ModuleType("mpi4py.MPI")
    real.IN_PLACE = object()  # type: ignore[attr-defined]
    real.PROC_NULL = -1  # type: ignore[attr-defined]
    real.ANY_SOURCE = -7  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", real)
    data = np.arange(3.0)
    comm.Reduce(real.IN_PLACE, data, op=MPI.SUM, root=0)
    comm.Allreduce(real.IN_PLACE, data)
    comm.Scatter(data, real.IN_PLACE)
    comm.Scatterv(data, real.IN_PLACE)
    np.testing.assert_array_equal(data, [0, 1, 2])
    assert comm.sendrecv(1, dest=real.PROC_NULL, source=0) is None
    assert comm.sendrecv(1, dest=0, source=real.ANY_SOURCE) == 1
    recv = np.zeros(3)
    comm.Sendrecv(data, dest=real.PROC_NULL, recvbuf=recv)
    np.testing.assert_array_equal(recv, 0)
    with pytest.raises(ValueError, match="only rank 0"):
        comm.bcast(1, root=5)
