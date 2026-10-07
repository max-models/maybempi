"""A serial stand-in for ``mpi4py.MPI``: a communicator of size 1.

:class:`SerialMPI` has the attributes of the ``mpi4py.MPI`` module that a serial
run needs, and :class:`SerialComm` the methods of a communicator. Collectives
return (object methods) or copy (buffer methods) what they would on one rank,
never ``None`` in place of a value. Methods that are not implemented raise
``AttributeError`` instead of silently doing nothing.

Point-to-point messages go from rank 0 to itself: a send is kept until a
matching receive takes it, in order, by tag. A receive that no message can
ever match raises ``RuntimeError`` where MPI would wait forever. Groups,
Cartesian topologies, derived datatypes and ``MPI.File`` work as on one
process.

Buffers are NumPy-like arrays (``.reshape``, ``.flags``, ``.size``) or mpi4py
buffer specifications (``[array, MPI.DOUBLE]``). A :class:`SerialComm` may also
be used next to real MPI (e.g. for work done by one rank of an MPI job): it then
accepts mpi4py's own ``IN_PLACE``, ``PROC_NULL`` and ``ANY_SOURCE`` as well. Arrays with a ``.get()``
method, such as CuPy arrays, are copied to a host receive buffer with
``.get()``; :func:`set_copy_hook` reports such copies to the application.
"""

from __future__ import annotations

import ctypes
import math
import pickle
import socket
import time
from collections.abc import Callable, Sequence
from types import MappingProxyType
from typing import Any

from maybempi import _file
from maybempi._datatypes import ORDER_C, ORDER_F, _buffer_runs, _Datatype
from maybempi._file import SerialFile
from maybempi._handles import (
    _ANY_SOURCE,
    _ANY_TAG,
    _COMM_NULL,
    _IN_PLACE,
    _PROC_NULL,
    _UNDEFINED,
    _Constant,
    _is_any_tag,
    _is_proc_null,
    _mpi4py_constant,
    _Null,
)

__all__ = [
    "SerialCartcomm",
    "SerialComm",
    "SerialFile",
    "SerialGroup",
    "SerialMPI",
    "SerialPrequest",
    "SerialRequest",
    "SerialStatus",
    "set_copy_hook",
]

_copy_hook: Callable[[str, Any], None] | None = None


def set_copy_hook(hook: Callable[[str, Any], None] | None) -> None:
    """Report copies between device and host arrays in buffer collectives.

    A receive into a host array from a device array (one with ``.get()``, e.g.
    a CuPy array), or the other way round, is a transfer that an application
    may want to count. The hook is called after each such copy.

    Args:
        hook: Called as ``hook(kind, array)`` with ``kind`` ``"to_host"`` or
            ``"to_device"`` and the host array that was copied. ``None``
            removes the hook.
    """
    global _copy_hook
    _copy_hook = hook


_CART = 1
_GRAPH = 2
_DIST_GRAPH = 3


class _Op(_Constant):
    """Placeholder for an MPI reduction operation."""

    __slots__ = ()


def _is_in_place(obj: Any) -> bool:
    """Tell whether `obj` is ``IN_PLACE``, the stand-in's or mpi4py's."""
    return obj is _IN_PLACE or (obj is not None and obj is _mpi4py_constant("IN_PLACE"))


def _buffer(spec: Any) -> Any:
    """Return the array of an mpi4py buffer specification (``buf`` or ``[buf, ...]``)."""
    if isinstance(spec, (list, tuple)):
        return spec[0]
    return spec


def _displacement(spec: Any) -> int:
    """Return the displacement of rank 0 in a vector buffer spec ``[buf, counts, displs, type]``."""
    if isinstance(spec, (list, tuple)) and len(spec) >= 3:
        displs = spec[2]
        if displs is not None and not isinstance(displs, _Constant):
            return int(displs[0])
    return 0


def _count(spec: Any) -> int | None:
    """Return the count of rank 0 in a vector buffer spec ``[buf, counts, displs, type]``."""
    if isinstance(spec, (list, tuple)) and len(spec) >= 2:
        counts = spec[1]
        if counts is not None and not isinstance(counts, _Constant):
            return int(counts[0])
    return None


def _copy(source: Any, target: Any, offset: int = 0) -> None:
    """Copy the elements of the array `source` into `target`, starting at `offset`.

    Both are flattened (C order). A device source (with ``.get()``) is copied to
    the host for a host target.
    """
    if _is_in_place(source) or source is None or target is None:
        return
    to_host = hasattr(source, "get") and not hasattr(target, "get")
    to_device = not hasattr(source, "get") and hasattr(target, "get")
    if to_host:
        source = source.get()
    if not target.flags.c_contiguous:
        raise ValueError("the receive buffer must be C-contiguous")
    flat = target.reshape(-1)
    source = source.reshape(-1)
    if offset + source.size > flat.size:
        raise ValueError(
            f"receive buffer too small: {flat.size} elements for {source.size} "
            f"at offset {offset}",
        )
    flat[offset : offset + source.size] = source
    if _copy_hook is not None and (to_host or to_device):
        _copy_hook("to_host" if to_host else "to_device", source)


def _check_rank(rank: int, what: str) -> None:
    if rank in (0, _PROC_NULL, _ANY_SOURCE) or _is_proc_null(rank):
        return
    if rank != _mpi4py_constant("ANY_SOURCE"):
        raise ValueError(f"{what}={rank}: a serial communicator has only rank 0")


def _fill_status(status: Any, source: int, tag: int, nbytes: int) -> None:
    """Record the source, tag and size of a message in a status object, if given."""
    if status is not None:
        status._set(source, tag, nbytes)


class SerialRequest:
    """A request of a non-blocking call of :class:`SerialComm`.

    Collectives and sends complete at once. A receive (``Irecv``, ``irecv``)
    or a synchronous send (``Issend``) completes when its matching message (or
    receive) is posted. Waiting for a request that can never complete raises
    ``RuntimeError``: on one process, MPI would wait forever.
    """

    def __init__(
        self, result: Any = None, *, complete: bool = True, waiting_for: str = ""
    ) -> None:
        """Create a request; complete unless `complete` is False."""
        self._result = result
        self._complete = complete
        self._waiting_for = waiting_for
        self._status: tuple[int, int, int] | None = None
        self._cancel: Callable[[], None] | None = None

    def _finish(
        self, result: Any = None, status: tuple[int, int, int] | None = None
    ) -> None:
        """Complete the request with a result and the status of its message."""
        self._result, self._status, self._complete = result, status, True

    def _check(self, call: str) -> None:
        if not self._complete:
            raise RuntimeError(
                f"{call} on rank 0 would wait forever: {self._waiting_for}. "
                "In a serial run the matching call must already have been made.",
            )

    def _report(self, status: Any) -> None:
        if self._status is not None:
            _fill_status(status, *self._status)

    def Wait(self, status: Any = None) -> None:
        """Return once the operation is complete (``RuntimeError`` if it never can)."""
        self._check("Wait")
        self._report(status)

    def Test(self, status: Any = None) -> bool:
        """Return whether the operation is complete."""
        if self._complete:
            self._report(status)
        return self._complete

    def wait(self, status: Any = None) -> Any:
        """Return the result of the non-blocking object call (``RuntimeError`` if never)."""
        self._check("wait")
        self._report(status)
        return self._result

    def test(self, status: Any = None) -> tuple[bool, Any]:
        """Return ``(True, result)`` when complete, else ``(False, None)``."""
        if not self.Test(status):
            return False, None
        return True, self._result

    def Free(self) -> None:
        """Do nothing."""
        return None

    def Cancel(self) -> None:
        """Cancel a receive that has no message yet; nothing for a complete request."""
        if not self._complete and self._cancel is not None:
            self._cancel()
            self._finish(None, (_ANY_SOURCE, _ANY_TAG, 0))

    @staticmethod
    def Waitall(requests: Sequence[Any], statuses: Any = None) -> None:
        """Wait for every request (``RuntimeError`` if one can never complete)."""
        for index, request in enumerate(requests):
            request.Wait(None if statuses is None else statuses[index])

    @staticmethod
    def waitall(requests: Sequence[Any], statuses: Any = None) -> list[Any]:
        """Return the results of the requests."""
        return [
            request.wait(None if statuses is None else statuses[index])
            for index, request in enumerate(requests)
        ]

    @staticmethod
    def Testall(requests: Sequence[Any], statuses: Any = None) -> bool:
        """Return whether every request is complete."""
        return all([request.Test() for request in requests])

    @staticmethod
    def testall(requests: Sequence[Any], statuses: Any = None) -> tuple[bool, Any]:
        """Return ``(True, results)`` when all are complete, else ``(False, None)``."""
        if not SerialRequest.Testall(requests):
            return False, None
        return True, SerialRequest.waitall(requests, statuses)

    @staticmethod
    def Waitany(requests: Sequence[Any], status: Any = None) -> int:
        """Return the index of a complete request, or ``UNDEFINED`` for no requests."""
        if not requests:
            return _UNDEFINED
        for index, request in enumerate(requests):
            if request.Test(status):
                return index
        requests[0].Wait()  # raises: none can complete
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def Testany(requests: Sequence[Any], status: Any = None) -> tuple[int, bool]:
        """Return ``(index, True)`` for a complete request, else ``(UNDEFINED, ...)``."""
        for index, request in enumerate(requests):
            if request.Test(status):
                return index, True
        return _UNDEFINED, not requests

    @staticmethod
    def Waitsome(requests: Sequence[Any], statuses: Any = None) -> list[int] | None:
        """Return the indices of the complete requests (``None`` for no requests)."""
        if not requests:
            return None
        done = SerialRequest.Testsome(requests, statuses)
        if not done:
            requests[0].Wait()  # raises: none can complete
        return done

    @staticmethod
    def Testsome(requests: Sequence[Any], statuses: Any = None) -> list[int] | None:
        """Return the indices of the complete requests (``None`` for no requests)."""
        if not requests:
            return None
        return [index for index, request in enumerate(requests) if request.Test()]


class SerialPrequest(SerialRequest):
    """A persistent request (``Send_init``/``Recv_init``): each ``Start`` posts it again.

    Before the first ``Start``, the request is inactive and complete.
    """

    def __init__(self, start: Callable[[], SerialRequest] | None = None) -> None:
        """Create a persistent request that `start` posts."""
        super().__init__()
        self._start = start
        self._active: SerialRequest | None = None

    def Start(self) -> None:
        """Post the operation (a send or a receive)."""
        if self._start is not None:
            self._active = self._start()

    @staticmethod
    def Startall(requests: Sequence[Any]) -> None:
        """Start every request."""
        for request in requests:
            request.Start()

    def Wait(self, status: Any = None) -> None:
        """Wait for the started operation (at once if inactive)."""
        if self._active is not None:
            self._active.Wait(status)

    def Test(self, status: Any = None) -> bool:
        """Return whether the started operation is complete (True if inactive)."""
        return self._active is None or self._active.Test(status)

    def Cancel(self) -> None:
        """Cancel the started operation."""
        if self._active is not None:
            self._active.Cancel()


def _complete_now(request: SerialRequest, call: str) -> SerialRequest:
    """Return a complete request, or withdraw it and raise: a blocking call never returns."""
    if not request._complete:
        request.Cancel()
        raise RuntimeError(
            f"{call} on rank 0 would wait forever: {request._waiting_for}. In a "
            "serial run the matching call must already have been made.",
        )
    return request


class _Message:
    """A message from rank 0 to itself, kept until a receive takes it."""

    def __init__(self, tag: int, data: Any, nbytes: int, is_object: bool) -> None:
        self.tag = tag
        self.data = data  # copies of the sent elements, or a pickle
        self.nbytes = nbytes
        self.is_object = is_object
        self.on_receive: Callable[[], None] | None = None

    @classmethod
    def of_buffer(cls, buf: Any, tag: int) -> _Message:
        array, runs = _buffer_runs(buf)
        flat = array.reshape(-1)
        pieces = [flat[start : start + length].copy() for start, length in runs]
        count = sum(length for _, length in runs)
        return cls(tag, pieces, count * array.itemsize, is_object=False)

    @classmethod
    def of_object(cls, obj: Any, tag: int) -> _Message:
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return cls(tag, data, len(data), is_object=True)

    def status(self) -> tuple[int, int, int]:
        return 0, self.tag, self.nbytes

    def unpack_buffer(self, buf: Any) -> None:
        """Copy the message into the elements of the receive buffer `buf`."""
        if self.is_object:
            raise TypeError(
                "a message sent with send/isend is received with recv/irecv"
            )
        array, runs = _buffer_runs(buf)
        if not array.flags.c_contiguous:
            raise ValueError("the receive buffer must be C-contiguous")
        received = sum(piece.size for piece in self.data)
        room = sum(length for _, length in runs)
        if received > room:
            raise ValueError(
                f"message truncated: {received} elements for a receive buffer of {room}"
            )
        flat = array.reshape(-1)
        targets = [flat[start : start + length] for start, length in runs]
        _copy_pieces(self.data, targets)

    def unpack_object(self) -> Any:
        """Return the sent object."""
        if not self.is_object:
            raise TypeError(
                "a message sent with Send/Isend is received with Recv/Irecv"
            )
        return pickle.loads(self.data)


def _copy_pieces(sources: list[Any], targets: list[Any]) -> None:
    """Copy the elements of the 1-D `sources` into the 1-D `targets`, in order."""
    target_index, filled = 0, 0
    for source in sources:
        if hasattr(source, "get") and targets and not hasattr(targets[0], "get"):
            source = source.get()
            if _copy_hook is not None:
                _copy_hook("to_host", source)
        elif targets and hasattr(targets[0], "get") and not hasattr(source, "get"):
            if _copy_hook is not None:
                _copy_hook("to_device", source)
        used = 0
        while used < source.size:
            target = targets[target_index]
            take = min(source.size - used, target.size - filled)
            target[filled : filled + take] = source[used : used + take]
            used += take
            filled += take
            if filled == target.size:
                target_index, filled = target_index + 1, 0


class SerialComm:
    """A communicator of size 1, with the mpi4py ``Comm`` methods a serial run needs.

    Collectives return (object methods) or copy (buffer methods) what they would
    on one rank: ``allreduce(x)`` is ``x``, ``gather(x)`` is ``[x]``,
    ``Allreduce(send, recv)`` copies `send` into `recv` (nothing with
    ``IN_PLACE``), ``Bcast`` does nothing. Point-to-point messages go to and
    come from rank 0 itself or ``PROC_NULL``: a send is kept, per
    communicator, until a receive with a matching tag takes it, in the order
    sent. A blocking receive with no matching message raises ``RuntimeError``
    (MPI would wait forever). Other methods raise ``AttributeError``.

    Attributes:
        rank: Always 0.
        size: Always 1.
    """

    rank = 0
    size = 1

    def __init__(self, name: str = "COMM_WORLD") -> None:
        """Create a communicator named `name` (as ``Get_name`` returns it)."""
        self._name = name
        self._messages: list[_Message] = []  # sent, not yet received
        self._receives: list[tuple[int, Callable[[_Message], None]]] = []

    def __repr__(self) -> str:
        """Return ``SerialComm(<name>)``."""
        return f"SerialComm({self._name})"

    # ---------------------------------------------------------------- queries
    def Get_rank(self) -> int:
        """Return 0."""
        return 0

    def Get_size(self) -> int:
        """Return 1."""
        return 1

    def Get_name(self) -> str:
        """Return the name of the communicator, e.g. ``"COMM_WORLD"``."""
        return self._name

    def Is_inter(self) -> bool:
        """Return False."""
        return False

    def Is_intra(self) -> bool:
        """Return True."""
        return True

    def Get_group(self) -> SerialGroup:
        """Return the group of the communicator: rank 0 only."""
        return SerialGroup()

    @property
    def group(self) -> SerialGroup:
        """The group of the communicator."""
        return self.Get_group()

    def Get_topology(self) -> int:
        """Return ``UNDEFINED``: no topology."""
        return _UNDEFINED

    @property
    def topology(self) -> int:
        """The topology: ``UNDEFINED``."""
        return self.Get_topology()

    # -------------------------------------------------- communicator creation
    def Dup(self, info: Any = None) -> SerialComm:
        """Return a new serial communicator with the same name."""
        return SerialComm(self._name)

    Clone = Dup

    def Split(self, color: int = 0, key: int = 0) -> Any:
        """Return a new serial communicator, or ``COMM_NULL`` for ``color=UNDEFINED``."""
        if color == _UNDEFINED:
            return _COMM_NULL
        return SerialComm(self._name)

    def Split_type(self, split_type: int, key: int = 0, info: Any = None) -> Any:
        """Return a new serial communicator, or ``COMM_NULL`` for ``UNDEFINED``."""
        return self.Split(split_type, key)

    def Create(self, group: SerialGroup) -> Any:
        """Return a communicator for `group`: a new one, or ``COMM_NULL`` if empty."""
        return SerialComm(self._name) if group.Get_size() else _COMM_NULL

    def Create_group(self, group: SerialGroup, tag: int = 0) -> Any:
        """Return a communicator for `group`: a new one, or ``COMM_NULL`` if empty."""
        return self.Create(group)

    def Create_cart(
        self,
        dims: Sequence[int],
        periods: Sequence[bool] | None = None,
        reorder: bool = False,
    ) -> SerialCartcomm:
        """Return a Cartesian communicator; the dimensions must multiply to 1.

        Raises:
            ValueError: If the grid needs more than one process.
        """
        dims = [int(n) for n in dims]
        if math.prod(dims) != 1:
            raise ValueError(f"a {dims} process grid needs more than 1 process")
        periods = [False] * len(dims) if periods is None else list(periods)
        if len(periods) != len(dims):
            raise ValueError(f"{len(periods)} periods for {len(dims)} dimensions")
        return SerialCartcomm(dims, periods, self._name)

    def Free(self) -> None:
        """Do nothing."""
        return None

    def Abort(self, errorcode: int = 0) -> None:
        """Exit the process with `errorcode` (``SystemExit``)."""
        raise SystemExit(errorcode)

    # ------------------------------------------------------ synchronization
    def Barrier(self) -> None:
        """Return at once."""
        return None

    barrier = Barrier

    def Ibarrier(self) -> SerialRequest:
        """Return a completed request."""
        return SerialRequest()

    # ------------------------------------------------- collectives, objects
    def bcast(self, obj: Any, root: int = 0) -> Any:
        """Return `obj`."""
        _check_rank(root, "root")
        return obj

    def reduce(self, sendobj: Any, op: Any = None, root: int = 0) -> Any:
        """Return `sendobj`."""
        _check_rank(root, "root")
        return sendobj

    def allreduce(self, sendobj: Any, op: Any = None) -> Any:
        """Return `sendobj`."""
        return sendobj

    def scan(self, sendobj: Any, op: Any = None) -> Any:
        """Return `sendobj`."""
        return sendobj

    def exscan(self, sendobj: Any, op: Any = None) -> None:
        """Return None: the result is undefined on rank 0 (``None`` in mpi4py)."""
        return None

    def gather(self, sendobj: Any, root: int = 0) -> list[Any]:
        """Return ``[sendobj]``."""
        _check_rank(root, "root")
        return [sendobj]

    def allgather(self, sendobj: Any) -> list[Any]:
        """Return ``[sendobj]``."""
        return [sendobj]

    def scatter(self, sendobj: Any, root: int = 0) -> Any:
        """Return the only item of `sendobj` (``ValueError`` unless it has one)."""
        _check_rank(root, "root")
        items = list(sendobj)
        if len(items) != 1:
            raise ValueError(f"scatter on 1 process needs 1 item, got {len(items)}")
        return items[0]

    def alltoall(self, sendobj: Any) -> list[Any]:
        """Return `sendobj` as a list (``ValueError`` unless it has one item)."""
        items = list(sendobj)
        if len(items) != 1:
            raise ValueError(f"alltoall on 1 process needs 1 item, got {len(items)}")
        return items

    def sendrecv(
        self,
        sendobj: Any,
        dest: int,
        sendtag: int = 0,
        recvbuf: Any = None,
        source: int = _ANY_SOURCE,
        recvtag: int = _ANY_TAG,
        status: Any = None,
    ) -> Any:
        """Send `sendobj` to `dest` and return the object received from `source`.

        With ``dest=0, source=0`` that is a copy of `sendobj` (unless an earlier
        message matches first); ``None`` with ``source=PROC_NULL``.
        """
        self.send(sendobj, dest, sendtag)
        return self.recv(recvbuf, source, recvtag, status)

    def ibcast(self, obj: Any, root: int = 0) -> SerialRequest:
        """Return a completed request whose ``wait()`` returns `obj`."""
        return SerialRequest(self.bcast(obj, root))

    def iallreduce(self, sendobj: Any, op: Any = None) -> SerialRequest:
        """Return a completed request whose ``wait()`` returns `sendobj`."""
        return SerialRequest(sendobj)

    # ------------------------------------------------- collectives, buffers
    def Bcast(self, buf: Any, root: int = 0) -> None:
        """Do nothing: rank 0 already has the data."""
        _check_rank(root, "root")

    def Reduce(self, sendbuf: Any, recvbuf: Any, op: Any = None, root: int = 0) -> None:
        """Copy `sendbuf` into `recvbuf` (nothing with ``IN_PLACE``)."""
        _check_rank(root, "root")
        _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Allreduce(self, sendbuf: Any, recvbuf: Any, op: Any = None) -> None:
        """Copy `sendbuf` into `recvbuf` (nothing with ``IN_PLACE``)."""
        _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Scan(self, sendbuf: Any, recvbuf: Any, op: Any = None) -> None:
        """Copy `sendbuf` into `recvbuf` (nothing with ``IN_PLACE``)."""
        _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Exscan(self, sendbuf: Any, recvbuf: Any, op: Any = None) -> None:
        """Do nothing: the receive buffer of rank 0 is undefined."""
        return None

    def Gather(self, sendbuf: Any, recvbuf: Any, root: int = 0) -> None:
        """Copy `sendbuf` into `recvbuf`."""
        _check_rank(root, "root")
        _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Gatherv(self, sendbuf: Any, recvbuf: Any, root: int = 0) -> None:
        """Copy `sendbuf` into `recvbuf`, at the displacement of rank 0."""
        _check_rank(root, "root")
        _copy(_buffer(sendbuf), _buffer(recvbuf), _displacement(recvbuf))

    def Allgather(self, sendbuf: Any, recvbuf: Any) -> None:
        """Copy `sendbuf` into `recvbuf`."""
        _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Allgatherv(self, sendbuf: Any, recvbuf: Any) -> None:
        """Copy `sendbuf` into `recvbuf`, at the displacement of rank 0."""
        _copy(_buffer(sendbuf), _buffer(recvbuf), _displacement(recvbuf))

    def Scatter(self, sendbuf: Any, recvbuf: Any, root: int = 0) -> None:
        """Copy `sendbuf` into `recvbuf` (nothing with ``IN_PLACE``)."""
        _check_rank(root, "root")
        if not _is_in_place(recvbuf):
            _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Scatterv(self, sendbuf: Any, recvbuf: Any, root: int = 0) -> None:
        """Copy the part of `sendbuf` at the displacement of rank 0 into `recvbuf`."""
        _check_rank(root, "root")
        if _is_in_place(recvbuf):
            return
        source = _buffer(sendbuf).reshape(-1)
        start = _displacement(sendbuf)
        target = _buffer(recvbuf)
        _copy(source[start : start + target.size], target)

    def Alltoall(self, sendbuf: Any, recvbuf: Any) -> None:
        """Copy `sendbuf` into `recvbuf`."""
        _copy(_buffer(sendbuf), _buffer(recvbuf))

    def Alltoallv(self, sendbuf: Any, recvbuf: Any) -> None:
        """Copy the part of `sendbuf` for rank 0 into `recvbuf`, at the displacement of rank 0."""
        if _is_in_place(sendbuf):
            return
        source = _buffer(sendbuf).reshape(-1)
        start = _displacement(sendbuf)
        count = _count(sendbuf)
        stop = source.size if count is None else start + count
        _copy(source[start:stop], _buffer(recvbuf), _displacement(recvbuf))

    def Sendrecv(
        self,
        sendbuf: Any,
        dest: int,
        sendtag: int = 0,
        recvbuf: Any = None,
        source: int = _ANY_SOURCE,
        recvtag: int = _ANY_TAG,
        status: Any = None,
    ) -> None:
        """Send `sendbuf` to `dest` and receive into `recvbuf` from `source`.

        With ``dest=0, source=0`` that copies `sendbuf` into `recvbuf`; with
        ``PROC_NULL`` the send or the receive does nothing.
        """
        self.Send(sendbuf, dest, sendtag)
        self.Recv(recvbuf, source, recvtag, status)

    def Sendrecv_replace(
        self,
        buf: Any,
        dest: int,
        sendtag: int = 0,
        source: int = _ANY_SOURCE,
        recvtag: int = _ANY_TAG,
        status: Any = None,
    ) -> None:
        """Send `buf` to `dest`, then receive into `buf` from `source`."""
        self.Sendrecv(buf, dest, sendtag, buf, source, recvtag, status)

    # ------------------------------------------------------- point-to-point
    def _post(self, message: _Message) -> None:
        """Hand a message to the first matching receive, or keep it."""
        for index, (tag, deliver) in enumerate(self._receives):
            if _is_any_tag(tag) or tag == message.tag:
                del self._receives[index]
                deliver(message)
                return
        self._messages.append(message)

    def _take(self, tag: int) -> _Message | None:
        """Remove and return the first kept message with a matching tag."""
        for index, message in enumerate(self._messages):
            if _is_any_tag(tag) or tag == message.tag:
                return self._messages.pop(index)
        return None

    def _waiting(self, call: str, tag: int) -> str:
        which = "any tag" if _is_any_tag(tag) else f"tag {tag}"
        return f"{call} waits for a message with {which} that rank 0 has not sent"

    def _receive(
        self,
        source: int,
        tag: int,
        unpack: Callable[[_Message], Any],
        call: str,
    ) -> SerialRequest:
        """Post a receive; the request completes with ``unpack(message)``."""
        _check_rank(source, "source")
        request = SerialRequest(complete=False, waiting_for=self._waiting(call, tag))
        if _is_proc_null(source):
            request._finish(None, (_PROC_NULL, _ANY_TAG, 0))
            return request

        def deliver(message: _Message) -> None:
            result = unpack(message)
            request._finish(result, message.status())
            if message.on_receive is not None:
                message.on_receive()

        message = self._take(tag)
        if message is not None:
            deliver(message)
        else:
            entry = (tag, deliver)
            self._receives.append(entry)
            request._cancel = lambda: self._receives.remove(entry)
        return request

    def _send(self, message: _Message, dest: int, synchronous: bool) -> SerialRequest:
        """Post a send; a synchronous one completes when a receive takes it."""
        _check_rank(dest, "dest")
        request = SerialRequest()
        if _is_proc_null(dest):
            return request
        if synchronous:
            request = SerialRequest(
                complete=False,
                waiting_for=f"a synchronous send with tag {message.tag} waits for "
                "a receive that rank 0 has not posted",
            )
            message.on_receive = request._finish
        self._post(message)
        return request

    def Send(self, buf: Any, dest: int, tag: int = 0) -> None:
        """Send `buf` to `dest` (0, or ``PROC_NULL``); kept until received."""
        self.Isend(buf, dest, tag)

    Bsend = Rsend = Send

    def _check_receiver(self, call: str, dest: int, tag: int) -> None:
        """Raise unless a posted receive would take a synchronous send at once."""
        _check_rank(dest, "dest")
        if _is_proc_null(dest):
            return
        if not any(_is_any_tag(want) or want == tag for want, _ in self._receives):
            raise RuntimeError(
                f"{call} on rank 0 would wait forever: no receive for tag {tag} "
                "is posted. In a serial run the receive must be posted first "
                "(Irecv), or use Send.",
            )

    def Ssend(self, buf: Any, dest: int, tag: int = 0) -> None:
        """Send synchronously: a matching receive must already be posted."""
        self._check_receiver("Ssend", dest, tag)
        self.Send(buf, dest, tag)

    def Isend(self, buf: Any, dest: int, tag: int = 0) -> SerialRequest:
        """Send `buf` and return a completed request (the data are copied)."""
        if _is_proc_null(dest):
            return self._send(_Message(tag, [], 0, False), dest, False)
        return self._send(_Message.of_buffer(buf, tag), dest, synchronous=False)

    Ibsend = Irsend = Isend

    def Issend(self, buf: Any, dest: int, tag: int = 0) -> SerialRequest:
        """Send synchronously: the request completes when a receive takes it."""
        if _is_proc_null(dest):
            return self._send(_Message(tag, [], 0, False), dest, False)
        return self._send(_Message.of_buffer(buf, tag), dest, synchronous=True)

    def Recv(
        self,
        buf: Any,
        source: int = _ANY_SOURCE,
        tag: int = _ANY_TAG,
        status: Any = None,
    ) -> None:
        """Receive into `buf` (``RuntimeError`` if no matching message was sent)."""
        _complete_now(self.Irecv(buf, source, tag), "Recv").Wait(status)

    def Irecv(
        self, buf: Any, source: int = _ANY_SOURCE, tag: int = _ANY_TAG
    ) -> SerialRequest:
        """Post a receive into `buf`; complete once a matching message is sent."""
        return self._receive(
            source, tag, lambda message: message.unpack_buffer(buf), "a receive"
        )

    def send(self, obj: Any, dest: int, tag: int = 0) -> None:
        """Send a (pickled) object to `dest`."""
        self.isend(obj, dest, tag)

    bsend = send

    def ssend(self, obj: Any, dest: int, tag: int = 0) -> None:
        """Send an object synchronously: a matching receive must be posted."""
        self._check_receiver("ssend", dest, tag)
        self.send(obj, dest, tag)

    def isend(self, obj: Any, dest: int, tag: int = 0) -> SerialRequest:
        """Send a (pickled) object and return a completed request."""
        return self._send(_Message.of_object(obj, tag), dest, synchronous=False)

    ibsend = isend

    def issend(self, obj: Any, dest: int, tag: int = 0) -> SerialRequest:
        """Send an object synchronously: complete when a receive takes it."""
        return self._send(_Message.of_object(obj, tag), dest, synchronous=True)

    def recv(
        self,
        buf: Any = None,
        source: int = _ANY_SOURCE,
        tag: int = _ANY_TAG,
        status: Any = None,
    ) -> Any:
        """Return a received object (``None`` from ``PROC_NULL``)."""
        return _complete_now(self.irecv(buf, source, tag), "recv").wait(status)

    def irecv(
        self, buf: Any = None, source: int = _ANY_SOURCE, tag: int = _ANY_TAG
    ) -> SerialRequest:
        """Post an object receive; ``wait()`` returns the object."""
        return self._receive(
            source, tag, lambda message: message.unpack_object(), "an object receive"
        )

    def Iprobe(
        self, source: int = _ANY_SOURCE, tag: int = _ANY_TAG, status: Any = None
    ) -> bool:
        """Return whether a matching message is waiting, without receiving it."""
        _check_rank(source, "source")
        if _is_proc_null(source):
            _fill_status(status, _PROC_NULL, _ANY_TAG, 0)
            return True
        for message in self._messages:
            if _is_any_tag(tag) or tag == message.tag:
                _fill_status(status, *message.status())
                return True
        return False

    iprobe = Iprobe

    def Probe(
        self, source: int = _ANY_SOURCE, tag: int = _ANY_TAG, status: Any = None
    ) -> bool:
        """Return True for a waiting message (``RuntimeError`` if none was sent)."""
        if not self.Iprobe(source, tag, status):
            raise RuntimeError(
                f"Probe on rank 0 would wait forever: {self._waiting('it', tag)}"
            )
        return True

    probe = Probe

    def Send_init(self, buf: Any, dest: int, tag: int = 0) -> SerialPrequest:
        """Return a persistent send: each ``Start`` sends `buf`."""
        return SerialPrequest(lambda: self.Isend(buf, dest, tag))

    Bsend_init = Rsend_init = Send_init

    def Ssend_init(self, buf: Any, dest: int, tag: int = 0) -> SerialPrequest:
        """Return a persistent synchronous send."""
        return SerialPrequest(lambda: self.Issend(buf, dest, tag))

    def Recv_init(
        self, buf: Any, source: int = _ANY_SOURCE, tag: int = _ANY_TAG
    ) -> SerialPrequest:
        """Return a persistent receive: each ``Start`` posts a receive into `buf`."""
        return SerialPrequest(lambda: self.Irecv(buf, source, tag))

    def Ibcast(self, buf: Any, root: int = 0) -> SerialRequest:
        """Return a completed request (``Bcast`` does nothing)."""
        self.Bcast(buf, root)
        return SerialRequest()

    def Iallreduce(self, sendbuf: Any, recvbuf: Any, op: Any = None) -> SerialRequest:
        """Copy like ``Allreduce`` and return a completed request."""
        self.Allreduce(sendbuf, recvbuf, op)
        return SerialRequest()

    def Iallgather(self, sendbuf: Any, recvbuf: Any) -> SerialRequest:
        """Copy like ``Allgather`` and return a completed request."""
        self.Allgather(sendbuf, recvbuf)
        return SerialRequest()


class SerialStatus:
    """Stand-in for ``MPI.Status``: the source, tag and size of a received message."""

    def __init__(self) -> None:
        """Create a status of an empty message from rank 0 with tag 0."""
        self.source = 0
        self.tag = 0
        self.error = 0
        self.count = 0  # bytes, as mpi4py's Status.count

    def _set(self, source: int, tag: int, nbytes: int) -> None:
        self.source, self.tag, self.count = source, tag, nbytes

    def Get_source(self) -> int:
        """Return the source rank (0, or ``PROC_NULL``)."""
        return self.source

    def Get_tag(self) -> int:
        """Return the tag of the message."""
        return self.tag

    def Get_error(self) -> int:
        """Return 0 (``SUCCESS``)."""
        return self.error

    def Get_count(self, datatype: Any = None) -> int:
        """Return the number of `datatype` items received (bytes by default)."""
        size = 1 if datatype is None else datatype.Get_size()
        return self.count // size if size else 0

    Get_elements = Get_count

    def Is_cancelled(self) -> bool:
        """Return False."""
        return False


class SerialGroup:
    """Stand-in for ``MPI.Group``: rank 0 alone, or the empty group."""

    def __init__(self, size: int = 1) -> None:
        """Create the group of rank 0 (`size` 1) or the empty group (`size` 0)."""
        self._size = size

    def __repr__(self) -> str:
        """Return ``SerialGroup(size=<size>)``."""
        return f"SerialGroup(size={self._size})"

    def Get_size(self) -> int:
        """Return 1, or 0 for the empty group."""
        return self._size

    def Get_rank(self) -> int:
        """Return 0, or ``UNDEFINED`` for the empty group."""
        return 0 if self._size else _UNDEFINED

    @property
    def size(self) -> int:
        """The number of ranks."""
        return self.Get_size()

    @property
    def rank(self) -> int:
        """The rank of this process in the group."""
        return self.Get_rank()

    def Translate_ranks(
        self, ranks: Sequence[int] | None = None, group: SerialGroup | None = None
    ) -> list[int]:
        """Return the ranks in `group` of `ranks` of this group (0 stays 0)."""
        ranks = list(range(self._size)) if ranks is None else list(ranks)
        target = 1 if group is None else group.Get_size()
        translated = []
        for rank in ranks:
            if _is_proc_null(rank):
                translated.append(rank)
            elif rank == 0 and self._size:
                translated.append(0 if target else _UNDEFINED)
            else:
                raise ValueError(f"rank {rank} is not in a group of size {self._size}")
        return translated

    def Compare(self, group: SerialGroup) -> int:
        """Return ``IDENT`` for groups of the same size, else ``UNEQUAL``."""
        return 0 if self._size == group.Get_size() else 3

    def Incl(self, ranks: Sequence[int]) -> SerialGroup:
        """Return the group of `ranks`: ``[0]`` or ``[]``."""
        self.Translate_ranks(ranks)
        return SerialGroup(1 if list(ranks) else 0)

    def Excl(self, ranks: Sequence[int]) -> SerialGroup:
        """Return the group without `ranks`."""
        self.Translate_ranks(ranks)
        return SerialGroup(0 if list(ranks) else self._size)

    @staticmethod
    def Union(group1: SerialGroup, group2: SerialGroup) -> SerialGroup:
        """Return the union of two groups."""
        return SerialGroup(max(group1.Get_size(), group2.Get_size()))

    @staticmethod
    def Intersection(group1: SerialGroup, group2: SerialGroup) -> SerialGroup:
        """Return the intersection of two groups."""
        return SerialGroup(min(group1.Get_size(), group2.Get_size()))

    @staticmethod
    def Difference(group1: SerialGroup, group2: SerialGroup) -> SerialGroup:
        """Return the ranks of `group1` that are not in `group2`."""
        return SerialGroup(group1.Get_size() if not group2.Get_size() else 0)

    def Dup(self) -> SerialGroup:
        """Return a copy."""
        return SerialGroup(self._size)

    def Free(self) -> None:
        """Do nothing."""
        return None


class SerialCartcomm(SerialComm):
    """Stand-in for ``MPI.Cartcomm``: a Cartesian grid of one process.

    Every dimension has length 1. Along a periodic dimension rank 0 is its own
    neighbour; along the others ``Shift`` gives ``PROC_NULL``.
    """

    def __init__(
        self,
        dims: Sequence[int],
        periods: Sequence[bool | int],
        name: str = "COMM_WORLD",
    ) -> None:
        """Create a grid with `dims` (all 1) and `periods`."""
        super().__init__(name)
        self._dims = [int(n) for n in dims]
        self._periods = [int(bool(p)) for p in periods]

    def __repr__(self) -> str:
        """Return ``SerialCartcomm(<name>, dims=..., periods=...)``."""
        return (
            f"SerialCartcomm({self._name}, dims={self._dims}, periods={self._periods})"
        )

    def Dup(self, info: Any = None) -> SerialCartcomm:
        """Return a new Cartesian communicator with the same grid."""
        return SerialCartcomm(self._dims, self._periods, self._name)

    Clone = Dup

    def Get_topology(self) -> int:
        """Return ``CART``."""
        return _CART

    def Get_dim(self) -> int:
        """Return the number of dimensions."""
        return len(self._dims)

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        return self.Get_dim()

    @property
    def dims(self) -> list[int]:
        """The processes along each dimension (all 1)."""
        return list(self._dims)

    @property
    def periods(self) -> list[int]:
        """Whether each dimension is periodic, as 0 or 1."""
        return list(self._periods)

    @property
    def coords(self) -> list[int]:
        """The coordinates of this process (all 0)."""
        return [0] * len(self._dims)

    def Get_topo(self) -> tuple[list[int], list[int], list[int]]:
        """Return ``(dims, periods, coords)``."""
        return self.dims, self.periods, self.coords

    @property
    def topo(self) -> tuple[list[int], list[int], list[int]]:
        """``(dims, periods, coords)``."""
        return self.Get_topo()

    def Get_coords(self, rank: int) -> list[int]:
        """Return the coordinates of `rank` (only 0 exists)."""
        if rank != 0:
            raise ValueError(f"rank={rank}: a serial communicator has only rank 0")
        return self.coords

    def Get_cart_rank(self, coords: Sequence[int]) -> int:
        """Return 0 for coordinates on the grid (any along periodic dimensions)."""
        coords = list(coords)
        if len(coords) != len(self._dims):
            raise ValueError(
                f"{len(coords)} coordinates for {len(self._dims)} dimensions"
            )
        for coord, periodic in zip(coords, self._periods, strict=True):
            if coord != 0 and not periodic:
                raise ValueError(f"coordinates {coords} are outside the grid")
        return 0

    def Shift(self, direction: int, disp: int) -> tuple[int, int]:
        """Return ``(source, dest)``: 0 along periodic dimensions, else ``PROC_NULL``."""
        if not 0 <= direction < len(self._dims):
            raise ValueError(f"direction {direction} for {len(self._dims)} dimensions")
        if self._periods[direction] or disp == 0:
            return 0, 0
        return _PROC_NULL, _PROC_NULL

    def Sub(self, remain_dims: Sequence[bool]) -> SerialCartcomm:
        """Return the grid of the dimensions in `remain_dims`."""
        keep = list(remain_dims)
        if len(keep) != len(self._dims):
            raise ValueError(f"{len(keep)} flags for {len(self._dims)} dimensions")
        return SerialCartcomm(
            [n for n, k in zip(self._dims, keep, strict=True) if k],
            [bool(p) for p, k in zip(self._periods, keep, strict=True) if k],
            self._name,
        )


class SerialMPI:
    """Stand-in for the ``mpi4py.MPI`` module in a serial run.

    :func:`~maybempi.launch.get_mpi` returns one instance, and
    :func:`~maybempi.launch.is_serial` tells it from ``mpi4py.MPI``.
    ``COMM_WORLD`` and ``COMM_SELF`` are :class:`SerialComm` objects; the reduction
    operations, datatypes and other constants are placeholders that
    :class:`SerialComm` accepts. ``Is_initialized()`` is False: MPI itself is
    never started.
    """

    COMM_WORLD = SerialComm("COMM_WORLD")
    COMM_SELF = SerialComm("COMM_SELF")
    COMM_NULL = _COMM_NULL
    Comm = Intracomm = SerialComm
    Cartcomm = SerialCartcomm
    Group = SerialGroup
    Request = SerialRequest
    Prequest = SerialPrequest
    Status = SerialStatus
    Datatype = _Datatype
    Op = _Op
    File = SerialFile

    GROUP_EMPTY = SerialGroup(0)
    GROUP_NULL = _Null("GROUP_NULL")
    FILE_NULL = _file.FILE_NULL
    INFO_NULL = _Null("INFO_NULL")
    INFO_ENV = _Constant("INFO_ENV")

    # comparisons of groups and communicators
    IDENT = 0
    CONGRUENT = 1
    SIMILAR = 2
    UNEQUAL = 3

    # topologies and communicator splitting
    CART = _CART
    GRAPH = _GRAPH
    DIST_GRAPH = _DIST_GRAPH
    COMM_TYPE_SHARED = 0

    # thread support levels
    THREAD_SINGLE = 0
    THREAD_FUNNELED = 1
    THREAD_SERIALIZED = 2
    THREAD_MULTIPLE = 3

    # subarray order and MPI-IO modes
    ORDER_C = ORDER_C
    ORDER_F = ORDER_F
    ORDER_FORTRAN = ORDER_F
    MODE_CREATE = _file.MODE_CREATE
    MODE_RDONLY = _file.MODE_RDONLY
    MODE_WRONLY = _file.MODE_WRONLY
    MODE_RDWR = _file.MODE_RDWR
    MODE_DELETE_ON_CLOSE = _file.MODE_DELETE_ON_CLOSE
    MODE_UNIQUE_OPEN = _file.MODE_UNIQUE_OPEN
    MODE_EXCL = _file.MODE_EXCL
    MODE_APPEND = _file.MODE_APPEND
    MODE_SEQUENTIAL = _file.MODE_SEQUENTIAL
    SEEK_SET = _file.SEEK_SET
    SEEK_CUR = _file.SEEK_CUR
    SEEK_END = _file.SEEK_END

    IN_PLACE = _IN_PLACE
    BOTTOM = _Constant("BOTTOM")
    DATATYPE_NULL = _Null("DATATYPE_NULL")
    REQUEST_NULL = _Null("REQUEST_NULL")
    OP_NULL = _Null("OP_NULL")
    PROC_NULL = _PROC_NULL
    ANY_SOURCE = _ANY_SOURCE
    ANY_TAG = _ANY_TAG
    ROOT = -3
    UNDEFINED = _UNDEFINED
    SUCCESS = 0

    # reduction operations
    SUM = _Op("SUM")
    PROD = _Op("PROD")
    MAX = _Op("MAX")
    MIN = _Op("MIN")
    LAND = _Op("LAND")
    LOR = _Op("LOR")
    LXOR = _Op("LXOR")
    BAND = _Op("BAND")
    BOR = _Op("BOR")
    BXOR = _Op("BXOR")
    MAXLOC = _Op("MAXLOC")
    MINLOC = _Op("MINLOC")
    REPLACE = _Op("REPLACE")

    # datatypes
    BYTE = _Datatype("BYTE", 1)
    CHAR = _Datatype("CHAR", 1)
    BOOL = _Datatype("BOOL", 1)
    C_BOOL = _Datatype("C_BOOL", 1)
    INT = _Datatype("INT", ctypes.sizeof(ctypes.c_int))
    LONG = _Datatype("LONG", ctypes.sizeof(ctypes.c_long))
    LONG_LONG = _Datatype("LONG_LONG", 8)
    UNSIGNED = _Datatype("UNSIGNED", ctypes.sizeof(ctypes.c_uint))
    UNSIGNED_LONG = _Datatype("UNSIGNED_LONG", ctypes.sizeof(ctypes.c_ulong))
    INT8_T = _Datatype("INT8_T", 1)
    INT16_T = _Datatype("INT16_T", 2)
    INT32_T = _Datatype("INT32_T", 4)
    INT64_T = _Datatype("INT64_T", 8)
    UINT8_T = _Datatype("UINT8_T", 1)
    UINT16_T = _Datatype("UINT16_T", 2)
    UINT32_T = _Datatype("UINT32_T", 4)
    UINT64_T = _Datatype("UINT64_T", 8)
    FLOAT = _Datatype("FLOAT", 4)
    DOUBLE = _Datatype("DOUBLE", 8)
    LONG_DOUBLE = _Datatype("LONG_DOUBLE", ctypes.sizeof(ctypes.c_longdouble))
    C_FLOAT_COMPLEX = _Datatype("C_FLOAT_COMPLEX", 8)
    C_DOUBLE_COMPLEX = _Datatype("C_DOUBLE_COMPLEX", 16)
    COMPLEX = _Datatype("COMPLEX", 8)
    DOUBLE_COMPLEX = _Datatype("DOUBLE_COMPLEX", 16)

    # NumPy type characters to datatypes, like mpi4py's (private) MPI._typedict
    _typedict = MappingProxyType({
        "b": INT8_T, "h": INT16_T, "i": INT32_T, "l": LONG, "q": INT64_T,
        "B": UINT8_T, "H": UINT16_T, "I": UINT32_T, "L": UNSIGNED_LONG, "Q": UINT64_T,
        "f": FLOAT, "d": DOUBLE, "g": LONG_DOUBLE, "?": C_BOOL,
        "F": C_FLOAT_COMPLEX, "D": C_DOUBLE_COMPLEX,
    })  # fmt: skip

    def __repr__(self) -> str:
        """Return a description that names the stand-in."""
        return "<SerialMPI: serial stand-in for mpi4py.MPI>"

    @staticmethod
    def Wtime() -> float:
        """Return the wall-clock time in seconds (``time.time()``)."""
        return time.time()

    @staticmethod
    def Wtick() -> float:
        """Return the resolution of :meth:`Wtime` in seconds."""
        return time.get_clock_info("time").resolution

    @staticmethod
    def Is_initialized() -> bool:
        """Return False: MPI is never started."""
        return False

    @staticmethod
    def Is_finalized() -> bool:
        """Return False."""
        return False

    @staticmethod
    def Init() -> None:
        """Do nothing."""
        return None

    @staticmethod
    def Finalize() -> None:
        """Do nothing."""
        return None

    @staticmethod
    def Get_processor_name() -> str:
        """Return the host name."""
        return socket.gethostname()

    @staticmethod
    def Get_version() -> tuple[int, int]:
        """Return ``(4, 0)``: the MPI standard the stand-in follows."""
        return 4, 0

    @staticmethod
    def Compute_dims(nnodes: int, dims: int | Sequence[int]) -> list[int]:
        """Return a process grid for `nnodes` (1) processes: all ones.

        Raises:
            ValueError: For a number of processes other than 1.
        """
        count = dims if isinstance(dims, int) else len(dims)
        if nnodes != 1:
            raise ValueError(f"a serial run has 1 process, not {nnodes}")
        given = [0] * count if isinstance(dims, int) else list(dims)
        if any(n not in (0, 1) for n in given):
            raise ValueError(f"dims {given} do not fit 1 process")
        return [1] * count

    @staticmethod
    def Query_thread() -> int:
        """Return 0 (``THREAD_SINGLE``)."""
        return 0
