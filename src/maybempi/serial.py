"""A serial stand-in for ``mpi4py.MPI``: a communicator of size 1.

:class:`SerialMPI` has the attributes of the ``mpi4py.MPI`` module that a serial
run needs, and :class:`SerialComm` the methods of a communicator. Collectives
return (object methods) or copy (buffer methods) what they would on one rank,
never ``None`` in place of a value. Methods that are not implemented raise
``AttributeError`` instead of silently doing nothing.

Buffers are NumPy-like arrays (``.reshape``, ``.flags``, ``.size``) or mpi4py
buffer specifications (``[array, MPI.DOUBLE]``). A :class:`SerialComm` may also
be used next to real MPI (e.g. for work done by one rank of an MPI job): it then
accepts mpi4py's own ``IN_PLACE``, ``PROC_NULL`` and ``ANY_SOURCE`` as well. Arrays with a ``.get()``
method, such as CuPy arrays, are copied to a host receive buffer with
``.get()``; :func:`set_copy_hook` reports such copies to the application.
"""

from __future__ import annotations

import socket
import sys
import time
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

__all__ = [
    "SerialComm",
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


class _Constant:
    """A named placeholder for an MPI constant (an op, a datatype, ``IN_PLACE``, ...)."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"SerialMPI.{self.name}"


class _Datatype(_Constant):
    """Placeholder for an MPI datatype (``isinstance(t, MPI.Datatype)`` holds)."""

    __slots__ = ()


class _Op(_Constant):
    """Placeholder for an MPI reduction operation."""

    __slots__ = ()


class _Null(_Constant):
    """A null handle (``COMM_NULL``, ``DATATYPE_NULL``, ...): false, like mpi4py's."""

    __slots__ = ()

    def __bool__(self) -> bool:
        return False


_IN_PLACE = _Constant("IN_PLACE")
_COMM_NULL = _Null("COMM_NULL")
_PROC_NULL = -2
_ANY_SOURCE = -1
_UNDEFINED = -32766


def _mpi4py_constant(name: str) -> Any:
    """Return ``mpi4py.MPI.<name>`` if the application imported mpi4py, else None."""
    module = sys.modules.get("mpi4py.MPI")
    return getattr(module, name, None) if module is not None else None


def _is_in_place(obj: Any) -> bool:
    """Tell whether `obj` is ``IN_PLACE``, the stand-in's or mpi4py's."""
    return obj is _IN_PLACE or (obj is not None and obj is _mpi4py_constant("IN_PLACE"))


def _is_proc_null(rank: Any) -> bool:
    """Tell whether `rank` is ``PROC_NULL``, the stand-in's or mpi4py's."""
    return rank == _PROC_NULL or rank == _mpi4py_constant("PROC_NULL")


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


class SerialRequest:
    """A completed request, returned by the non-blocking calls of :class:`SerialComm`."""

    def __init__(self, result: Any = None) -> None:
        """Create a request whose ``wait()`` returns `result`."""
        self._result = result

    def Wait(self, status: Any = None) -> None:
        """Return at once: the operation is complete."""
        return None

    def Test(self, status: Any = None) -> bool:
        """Return True: the operation is complete."""
        return True

    def wait(self, status: Any = None) -> Any:
        """Return the result of the non-blocking object call."""
        return self._result

    def test(self, status: Any = None) -> tuple[bool, Any]:
        """Return ``(True, result)``."""
        return True, self._result

    def Free(self) -> None:
        """Do nothing."""
        return None

    def Cancel(self) -> None:
        """Do nothing: the operation is complete."""
        return None

    @staticmethod
    def Waitall(requests: Any, statuses: Any = None) -> None:
        """Return at once: all requests are complete."""
        return None

    @staticmethod
    def waitall(requests: Any, statuses: Any = None) -> list[Any]:
        """Return the results of the requests."""
        return [request.wait() for request in requests]

    @staticmethod
    def Testall(requests: Any, statuses: Any = None) -> bool:
        """Return True: all requests are complete."""
        return True

    @staticmethod
    def Waitany(requests: Any, status: Any = None) -> int:
        """Return 0, or ``UNDEFINED`` for no requests."""
        return 0 if requests else _UNDEFINED


class SerialPrequest(SerialRequest):
    """A persistent request (``Send_init``/``Recv_init``), for ``Startall``/``Waitall``."""

    def Start(self) -> None:
        """Do nothing."""
        return None

    @staticmethod
    def Startall(requests: Any) -> None:
        """Do nothing."""
        return None


class SerialComm:
    """A communicator of size 1, with the mpi4py ``Comm`` methods a serial run needs.

    Collectives return (object methods) or copy (buffer methods) what they would
    on one rank: ``allreduce(x)`` is ``x``, ``gather(x)`` is ``[x]``,
    ``Allreduce(send, recv)`` copies `send` into `recv` (nothing with
    ``IN_PLACE``), ``Bcast`` does nothing. Point-to-point calls are only
    supported to and from rank 0 itself (``sendrecv``, ``Sendrecv``) or
    ``PROC_NULL``. Other methods raise ``AttributeError``.

    Attributes:
        rank: Always 0.
        size: Always 1.
    """

    rank = 0
    size = 1

    def __init__(self, name: str = "COMM_WORLD") -> None:
        """Create a communicator named `name` (as ``Get_name`` returns it)."""
        self._name = name

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
        dest: int = 0,
        sendtag: int = 0,
        recvbuf: Any = None,
        source: int = 0,
        recvtag: int = 0,
        status: Any = None,
    ) -> Any:
        """Return `sendobj` (sent to and received from rank 0), or None with ``PROC_NULL``."""
        _check_rank(dest, "dest")
        _check_rank(source, "source")
        if _is_proc_null(source):
            return None
        return None if _is_proc_null(dest) else sendobj

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

    def Sendrecv(
        self,
        sendbuf: Any,
        dest: int = 0,
        sendtag: int = 0,
        recvbuf: Any = None,
        source: int = 0,
        recvtag: int = 0,
        status: Any = None,
    ) -> None:
        """Copy `sendbuf` into `recvbuf` (sent to and received from rank 0); nothing with ``PROC_NULL``."""
        _check_rank(dest, "dest")
        _check_rank(source, "source")
        if _is_proc_null(dest) or _is_proc_null(source):
            return
        _copy(_buffer(sendbuf), _buffer(recvbuf))

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
    """Stand-in for ``MPI.Status`` (source 0, tag 0)."""

    source = 0
    tag = 0
    error = 0

    def Get_source(self) -> int:
        """Return 0."""
        return 0

    def Get_tag(self) -> int:
        """Return 0."""
        return 0

    def Get_count(self, datatype: Any = None) -> int:
        """Return 0."""
        return 0


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
    Request = SerialRequest
    Prequest = SerialPrequest
    Status = SerialStatus
    Datatype = _Datatype
    Op = _Op

    IN_PLACE = _IN_PLACE
    BOTTOM = _Constant("BOTTOM")
    DATATYPE_NULL = _Null("DATATYPE_NULL")
    REQUEST_NULL = _Null("REQUEST_NULL")
    OP_NULL = _Null("OP_NULL")
    PROC_NULL = _PROC_NULL
    ANY_SOURCE = _ANY_SOURCE
    ANY_TAG = -1
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
    BYTE = _Datatype("BYTE")
    CHAR = _Datatype("CHAR")
    BOOL = _Datatype("BOOL")
    C_BOOL = _Datatype("C_BOOL")
    INT = _Datatype("INT")
    LONG = _Datatype("LONG")
    LONG_LONG = _Datatype("LONG_LONG")
    UNSIGNED = _Datatype("UNSIGNED")
    UNSIGNED_LONG = _Datatype("UNSIGNED_LONG")
    INT8_T = _Datatype("INT8_T")
    INT16_T = _Datatype("INT16_T")
    INT32_T = _Datatype("INT32_T")
    INT64_T = _Datatype("INT64_T")
    UINT8_T = _Datatype("UINT8_T")
    UINT16_T = _Datatype("UINT16_T")
    UINT32_T = _Datatype("UINT32_T")
    UINT64_T = _Datatype("UINT64_T")
    FLOAT = _Datatype("FLOAT")
    DOUBLE = _Datatype("DOUBLE")
    LONG_DOUBLE = _Datatype("LONG_DOUBLE")
    C_FLOAT_COMPLEX = _Datatype("C_FLOAT_COMPLEX")
    C_DOUBLE_COMPLEX = _Datatype("C_DOUBLE_COMPLEX")
    COMPLEX = _Datatype("COMPLEX")
    DOUBLE_COMPLEX = _Datatype("DOUBLE_COMPLEX")

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
    def Query_thread() -> int:
        """Return 0 (``THREAD_SINGLE``)."""
        return 0
