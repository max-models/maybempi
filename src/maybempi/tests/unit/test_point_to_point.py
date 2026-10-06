"""Point-to-point messages of the serial stand-in: rank 0 sends to itself."""

import numpy as np
import pytest

from maybempi import get_mpi, set_copy_hook
from maybempi.serial import SerialComm

MPI = get_mpi(False)


@pytest.fixture
def comm():
    """A fresh communicator, so that no message is left over between tests."""
    return SerialComm()


def test_send_then_receive(comm):
    comm.Send(np.arange(3.0), dest=0, tag=4)
    recv = np.zeros(3)
    status = MPI.Status()
    comm.Recv(recv, source=0, tag=4, status=status)
    np.testing.assert_array_equal(recv, [0, 1, 2])
    assert status.Get_source() == 0 and status.Get_tag() == 4
    assert status.Get_count() == 24 and status.Get_count(MPI.DOUBLE) == 3
    assert status.count == 24 and status.Get_elements(MPI.DOUBLE) == 3
    assert status.Get_error() == 0 and not status.Is_cancelled()


def test_the_data_are_copied_when_sent(comm):
    data = np.arange(3.0)
    request = comm.Isend(data, dest=0)
    assert request.Test()
    data[:] = -1  # after the send: the message keeps the old values
    recv = np.zeros(3)
    comm.Recv(recv)
    np.testing.assert_array_equal(recv, [0, 1, 2])


def test_receive_posted_first_completes_on_the_send(comm):
    recv = np.zeros(2)
    request = comm.Irecv(recv, source=0, tag=7)
    assert not request.Test()
    assert request.test() == (False, None)
    comm.Isend(np.ones(2), dest=0, tag=3)  # another tag: kept
    assert not request.Test()
    comm.Send(np.full(2, 5.0), dest=0, tag=7)
    status = MPI.Status()
    assert request.Test(status) and status.Get_tag() == 7
    request.Wait()
    np.testing.assert_array_equal(recv, 5)
    comm.Recv(recv, tag=3)
    np.testing.assert_array_equal(recv, 1)


def test_messages_are_received_in_order_and_by_tag(comm):
    for value, tag in [(1, 1), (2, 2), (3, 1)]:
        comm.send(value, dest=0, tag=tag)
    assert comm.recv(tag=2) == 2
    assert comm.recv(tag=1) == 1
    assert comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG) == 3


def test_object_messages(comm):
    value = {"a": [1, 2]}
    comm.isend(value, dest=0, tag=1).Wait()
    value["a"].append(3)  # pickled when sent
    request = comm.irecv(source=0, tag=1)
    assert request.wait() == {"a": [1, 2]}
    later = comm.irecv(tag=9)
    comm.send("x", dest=0, tag=9)
    assert later.test() == (True, "x")
    comm.bsend(4, dest=0)
    comm.ibsend(5, dest=0)
    assert [comm.recv(), comm.recv()] == [4, 5]
    comm.send([1], dest=0)
    with pytest.raises(TypeError, match="Recv/Irecv|recv/irecv"):
        comm.Recv(np.zeros(1))
    comm.Send(np.zeros(1), dest=0)
    with pytest.raises(TypeError, match="Recv/Irecv"):
        comm.recv()


def test_a_receive_without_a_message_raises_instead_of_hanging(comm):
    with pytest.raises(RuntimeError, match="wait forever"):
        comm.Recv(np.zeros(1), source=0, tag=2)
    with pytest.raises(RuntimeError, match="any tag"):
        comm.recv()
    # the failed receives are withdrawn: a later send is kept for a later receive
    comm.Send(np.ones(1), dest=0, tag=2)
    assert comm.Iprobe(tag=2)
    request = comm.Irecv(np.zeros(1), tag=5)
    with pytest.raises(RuntimeError, match="tag 5"):
        request.Wait()
    with pytest.raises(RuntimeError, match="wait forever"):
        request.wait()
    with pytest.raises(RuntimeError, match="wait forever"):
        MPI.Request.Waitall([request])
    request.Cancel()
    assert request.Test()  # cancelled: complete


def test_proc_null_does_nothing(comm):
    comm.Send(np.ones(2), dest=MPI.PROC_NULL)
    comm.send(1, dest=MPI.PROC_NULL)
    comm.Isend(np.ones(2), dest=MPI.PROC_NULL).Wait()
    comm.Issend(np.ones(2), dest=MPI.PROC_NULL).Wait()
    comm.Ssend(np.ones(2), dest=MPI.PROC_NULL)
    recv = np.zeros(2)
    status = MPI.Status()
    comm.Recv(recv, source=MPI.PROC_NULL, status=status)
    assert status.Get_source() == MPI.PROC_NULL and status.Get_count() == 0
    assert comm.recv(source=MPI.PROC_NULL) is None
    assert comm.Iprobe(source=MPI.PROC_NULL)
    np.testing.assert_array_equal(recv, 0)
    assert not comm.Iprobe()  # nothing was kept


def test_other_ranks_raise(comm):
    with pytest.raises(ValueError, match="only rank 0"):
        comm.Send(np.ones(1), dest=1)
    with pytest.raises(ValueError, match="only rank 0"):
        comm.Irecv(np.ones(1), source=2)


def test_synchronous_sends_need_a_posted_receive(comm):
    with pytest.raises(RuntimeError, match="no receive for tag 3"):
        comm.Ssend(np.ones(1), dest=0, tag=3)
    with pytest.raises(RuntimeError, match="no receive"):
        comm.ssend(1, dest=0)
    assert not comm.Iprobe()  # nothing was sent
    recv = np.zeros(1)
    receive = comm.Irecv(recv, tag=3)
    comm.Ssend(np.ones(1), dest=0, tag=3)
    assert receive.Test()
    pending = comm.Issend(np.full(1, 2.0), dest=0, tag=4)
    assert not pending.Test()
    comm.Recv(recv, tag=4)
    assert pending.Test() and recv[0] == 2
    waiting = comm.irecv(tag=1)
    comm.ssend("y", dest=0, tag=1)
    assert waiting.wait() == "y"
    object_pending = comm.issend("z", dest=0)
    assert not object_pending.Test()
    assert comm.recv() == "z" and object_pending.Test()


def test_sendrecv_variants(comm):
    recv = np.zeros(3)
    comm.Sendrecv(np.arange(3.0), dest=0, sendtag=1, recvbuf=recv, source=0, recvtag=1)
    np.testing.assert_array_equal(recv, [0, 1, 2])
    buf = np.arange(3.0)
    comm.Sendrecv_replace(buf, dest=0, source=0)
    np.testing.assert_array_equal(buf, [0, 1, 2])
    assert comm.sendrecv("a", dest=0, source=0) == "a"
    # a receive from PROC_NULL leaves the sent message for a later receive
    comm.Sendrecv(np.ones(3), dest=0, recvbuf=recv, source=MPI.PROC_NULL)
    comm.Recv(recv)
    np.testing.assert_array_equal(recv, 1)


def test_probe(comm):
    with pytest.raises(RuntimeError, match="Probe on rank 0 would wait forever"):
        comm.Probe(tag=1)
    comm.Send(np.ones(4, dtype=np.int32), dest=0, tag=1)
    status = MPI.Status()
    assert comm.Probe(tag=1, status=status) and comm.iprobe(tag=1)
    assert status.Get_count(MPI.INT32_T) == 4 and status.Get_tag() == 1
    assert not comm.Iprobe(tag=2)
    assert comm.probe()


def test_buffer_specs_and_truncation(comm):
    data = np.arange(6.0)
    comm.Send([data, 2, MPI.DOUBLE], dest=0)
    comm.Send([data, (2, 3), MPI.DOUBLE], dest=0)  # 2 values from index 3
    comm.Send([data, MPI.DOUBLE], dest=0)
    recv = np.zeros(6)
    comm.Recv([recv, (2, 1), MPI.DOUBLE])
    np.testing.assert_array_equal(recv, [0, 0, 1, 0, 0, 0])
    comm.Recv(recv)
    np.testing.assert_array_equal(recv[:2], [3, 4])
    with pytest.raises(ValueError, match="truncated"):
        comm.Recv(np.zeros(5))
    with pytest.raises(ValueError, match="C-contiguous"):
        comm.Send(data, dest=0)
        comm.Recv(np.zeros((6, 2))[:, 0])
    with pytest.raises(ValueError, match="covers 8 elements"):
        comm.Send([data, 8, MPI.DOUBLE], dest=0)
    empty = np.zeros(0)
    comm.Send(empty, dest=0)
    comm.Recv(empty)


def test_subarray_buffers(comm):
    grid = np.arange(20.0).reshape(4, 5)
    edge = MPI.DOUBLE.Create_subarray([4, 5], [4, 1], [0, 3]).Commit()
    halo = MPI.DOUBLE.Create_subarray([4, 5], [4, 1], [0, 0]).Commit()
    comm.Sendrecv([grid, 1, edge], dest=0, recvbuf=[grid, 1, halo], source=0)
    np.testing.assert_array_equal(grid[:, 0], grid[:, 3])
    column = np.zeros(4)
    comm.Send([grid, 1, edge], dest=0)
    comm.Recv(column)
    np.testing.assert_array_equal(column, [3, 8, 13, 18])
    edge.Free()


def test_persistent_requests(comm):
    send = np.zeros(1)
    recv = np.zeros(1)
    outgoing = comm.Send_init(send, dest=0, tag=2)
    incoming = comm.Recv_init(recv, source=0, tag=2)
    assert incoming.Test()  # inactive
    incoming.Wait()
    for step in range(3):
        send[0] = step
        MPI.Prequest.Startall([incoming, outgoing])
        MPI.Request.Waitall([incoming, outgoing])
        assert recv[0] == step
    waiting = comm.Recv_init(recv, tag=8)
    waiting.Start()
    assert not waiting.Test()
    waiting.Cancel()
    assert waiting.Test()
    synchronous = comm.Ssend_init(send, dest=0, tag=6)
    synchronous.Start()
    assert not synchronous.Test()
    comm.Recv(recv, tag=6)
    synchronous.Wait()
    comm.Bsend_init(send, dest=0).Start()
    comm.Rsend_init(send, dest=0).Start()
    comm.Recv(recv)
    comm.Recv(recv)
    MPI.Prequest().Cancel()


def test_request_collections(comm):
    done = comm.Isend(np.ones(1), dest=0, tag=1)
    pending = comm.Irecv(np.zeros(1), tag=2)
    requests = [pending, done]
    assert not MPI.Request.Testall(requests)
    assert MPI.Request.testall(requests) == (False, None)
    assert MPI.Request.Waitany(requests) == 1
    assert MPI.Request.Testany(requests) == (1, True)
    assert MPI.Request.Testany([pending]) == (MPI.UNDEFINED, False)
    assert MPI.Request.Testany([]) == (MPI.UNDEFINED, True)
    assert MPI.Request.Waitsome(requests) == [1]
    assert MPI.Request.Testsome(requests) == [1]
    assert MPI.Request.Waitsome([]) is None and MPI.Request.Testsome([]) is None
    with pytest.raises(RuntimeError, match="wait forever"):
        MPI.Request.Waitany([pending])
    with pytest.raises(RuntimeError, match="wait forever"):
        MPI.Request.Waitsome([pending])
    comm.Send(np.ones(1), dest=0, tag=2)
    statuses = [MPI.Status(), MPI.Status()]
    MPI.Request.Waitall(requests, statuses)
    assert [s.Get_tag() for s in statuses] == [2, 0]
    comm.Recv(np.zeros(1), tag=1)  # the message of `done`
    assert MPI.Request.testall([comm.isend(1, dest=0)]) == (True, [None])
    assert MPI.Request.waitall([comm.irecv()], [MPI.Status()]) == [1]


def test_messages_stay_on_their_communicator(comm):
    other = comm.Dup()
    comm.send(1, dest=0)
    assert not other.Iprobe()
    assert comm.recv() == 1


class _DeviceArray:
    """A duck-typed device array (like CuPy): `.get()` copies it to the host."""

    def __init__(self, data):
        self._data = np.asarray(data)

    def __getattr__(self, name):
        return getattr(self._data, name)

    def __getitem__(self, index):
        return _DeviceArray(self._data[index])

    def __setitem__(self, index, value):
        self._data[index] = value._data if isinstance(value, _DeviceArray) else value

    def reshape(self, *shape):
        return _DeviceArray(self._data.reshape(*shape))

    def copy(self):
        return _DeviceArray(self._data.copy())

    def get(self):
        return self._data.copy()


def test_device_buffers_and_the_copy_hook(comm):
    copies = []
    set_copy_hook(lambda kind, array: copies.append(kind))
    try:
        comm.Send(_DeviceArray(np.arange(3.0)), dest=0)
        recv = np.zeros(3)
        comm.Recv(recv)
        np.testing.assert_array_equal(recv, [0, 1, 2])
        device = _DeviceArray(np.zeros(3))
        comm.Send(np.ones(3), dest=0)
        comm.Recv(device)
        np.testing.assert_array_equal(device.get(), 1)
        comm.Send(_DeviceArray(np.full(3, 2.0)), dest=0)
        comm.Recv(device)  # device to device: no copy through the host
        np.testing.assert_array_equal(device.get(), 2)
    finally:
        set_copy_hook(None)
    assert copies == ["to_host", "to_device"]
    comm.Send(_DeviceArray(np.arange(3.0)), dest=0)
    comm.Recv(np.zeros(3))  # without a hook
    comm.Send(np.ones(3), dest=0)
    comm.Recv(_DeviceArray(np.zeros(3)))
