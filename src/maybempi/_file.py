"""``MPI.File`` for the serial stand-in: MPI-IO on one process, with Python file I/O.

A view (``Set_view``) works as in MPI: a displacement in bytes, an elementary
type, and a file type built from it (e.g. ``Create_subarray``) that selects
where in the file the data goes. Offsets and file pointers count elementary
types within the view. Buffers must be host arrays (NumPy-like, with ``view``).
"""

from __future__ import annotations

import bisect
import os
from typing import Any

from maybempi._datatypes import _buffer_runs, _Datatype
from maybempi._handles import _Null

MODE_CREATE = 1
MODE_RDONLY = 2
MODE_WRONLY = 4
MODE_RDWR = 8
MODE_DELETE_ON_CLOSE = 16
MODE_UNIQUE_OPEN = 32
MODE_EXCL = 64
MODE_APPEND = 128
MODE_SEQUENTIAL = 256
SEEK_SET = 600
SEEK_CUR = 602
SEEK_END = 604
FILE_NULL = _Null("FILE_NULL")

_BYTE = _Datatype("BYTE", 1)


def _byte_view(array: Any, writable: bool) -> memoryview:
    """Return the bytes of a host array as a flat memoryview."""
    if hasattr(array, "get"):
        if writable:
            raise TypeError("MPI.File in a serial run reads only into host arrays")
        array = array.get()
    if not array.flags.c_contiguous:
        if writable:
            raise ValueError("the read buffer must be C-contiguous")
        array = array.copy()
    return memoryview(array.reshape(-1).view("u1"))


class SerialFile:
    """Stand-in for ``mpi4py.MPI.File``: a file opened by the only process."""

    def __init__(self, handle: Any, path: str, amode: int) -> None:
        """Wrap an open Python file; use :meth:`Open`."""
        self._handle = handle
        self._path = path
        self._amode = amode
        self._atomic = False
        self.Set_view()
        if amode & MODE_APPEND:
            self._position = self.Get_size() // self._etype.size

    def __repr__(self) -> str:
        """Return ``SerialFile(<path>)``."""
        return f"SerialFile({self._path!r})"

    def __enter__(self) -> SerialFile:
        """Return the file, to close it at the end of a ``with`` block."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the file."""
        self.Close()

    # ---------------------------------------------------- opening, closing
    @classmethod
    def Open(
        cls,
        comm: Any,
        filename: str | os.PathLike[str],
        amode: int = MODE_RDONLY,
        info: Any = None,
    ) -> SerialFile:
        """Open `filename` with the MPI access mode `amode` (``MODE_RDONLY``, ...).

        Raises:
            ValueError: For an invalid combination of modes.
            OSError: If the file cannot be opened (e.g. ``FileNotFoundError``).
        """
        access = amode & (MODE_RDONLY | MODE_WRONLY | MODE_RDWR)
        if access not in (MODE_RDONLY, MODE_WRONLY, MODE_RDWR):
            raise ValueError(
                "amode needs exactly one of MODE_RDONLY, MODE_WRONLY, MODE_RDWR"
            )
        if access == MODE_RDONLY and amode & (MODE_CREATE | MODE_EXCL):
            raise ValueError("MODE_CREATE and MODE_EXCL need write access")
        flags = {
            MODE_RDONLY: os.O_RDONLY,
            MODE_WRONLY: os.O_WRONLY,
            MODE_RDWR: os.O_RDWR,
        }[access] | getattr(os, "O_BINARY", 0)
        if amode & MODE_CREATE:
            flags |= os.O_CREAT
        if amode & MODE_EXCL:
            flags |= os.O_EXCL
        path = os.fspath(filename)
        descriptor = os.open(path, flags, 0o666)
        mode = {MODE_RDONLY: "rb", MODE_WRONLY: "wb", MODE_RDWR: "r+b"}[access]
        # a descriptor opened without O_TRUNC is not truncated by "wb"
        return cls(open(descriptor, mode, buffering=0), path, amode)

    def Close(self) -> None:
        """Close the file (and delete it with ``MODE_DELETE_ON_CLOSE``)."""
        self._handle.close()
        if self._amode & MODE_DELETE_ON_CLOSE:
            os.remove(self._path)

    @staticmethod
    def Delete(filename: str | os.PathLike[str], info: Any = None) -> None:
        """Delete a file."""
        os.remove(os.fspath(filename))

    # -------------------------------------------------------------- queries
    def Get_amode(self) -> int:
        """Return the access mode the file was opened with."""
        return self._amode

    def Get_size(self) -> int:
        """Return the size of the file in bytes."""
        return os.fstat(self._handle.fileno()).st_size

    def Set_size(self, size: int) -> None:
        """Truncate or extend the file to `size` bytes."""
        self._handle.truncate(size)

    def Preallocate(self, size: int) -> None:
        """Extend the file to at least `size` bytes."""
        if self.Get_size() < size:
            self._handle.truncate(size)

    def Sync(self) -> None:
        """Write the file to the storage device."""
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def Set_atomicity(self, flag: bool) -> None:
        """Record the atomicity flag (one process is always atomic)."""
        self._atomic = bool(flag)

    def Get_atomicity(self) -> bool:
        """Return the atomicity flag."""
        return self._atomic

    # ----------------------------------------------------------------- view
    def Set_view(
        self,
        disp: int = 0,
        etype: _Datatype | None = None,
        filetype: _Datatype | None = None,
        datarep: str = "native",
        info: Any = None,
    ) -> None:
        """Set the view: data from byte `disp` on, where `filetype` places it.

        Resets the file pointer to 0.

        Raises:
            ValueError: For a data representation other than ``"native"``, or
                a file type not built from the elementary type.
        """
        etype = _BYTE if etype is None else etype
        filetype = etype if filetype is None else filetype
        if datarep != "native":
            raise ValueError(
                f"the serial stand-in supports datarep 'native' only, not {datarep!r}"
            )
        if not etype._named:
            raise ValueError("the elementary type must be a named type")
        if filetype._itemsize != etype.size:
            raise ValueError(
                f"the file type ({filetype.name}) is not built from the elementary "
                f"type ({etype.name})",
            )
        self._disp, self._etype, self._filetype, self._datarep = (
            int(disp),
            etype,
            filetype,
            datarep,
        )
        self._starts = [0]
        for _, length in filetype._runs:
            self._starts.append(self._starts[-1] + length)
        self._position = 0

    def Get_view(self) -> tuple[int, _Datatype, _Datatype, str]:
        """Return ``(disp, etype, filetype, datarep)``."""
        return self._disp, self._etype, self._filetype, self._datarep

    def Get_byte_offset(self, offset: int) -> int:
        """Return the byte position of the elementary type at `offset` in the view."""
        if self._starts[-1] == 0:
            return self._disp
        return self._segments(offset, 1)[0][0]

    def _segments(self, offset: int, count: int) -> list[tuple[int, int]]:
        """Return the ``(byte position, bytes)`` pieces of `count` etypes from `offset`."""
        per_copy = self._starts[-1]
        if count and not per_copy:
            raise ValueError("the file type of the view holds no data")
        size, extent = self._etype.size, self._filetype._extent
        segments: list[list[int]] = []
        while count > 0:
            copy, within = divmod(offset, per_copy)
            run = bisect.bisect_right(self._starts, within) - 1
            start, length = self._filetype._runs[run]
            inside = within - self._starts[run]
            take = min(length - inside, count)
            position = self._disp + (copy * extent + start + inside) * size
            if segments and segments[-1][0] + segments[-1][1] == position:
                segments[-1][1] += take * size
            else:
                segments.append([position, take * size])
            offset += take
            count -= take
        return [(position, nbytes) for position, nbytes in segments]

    # ---------------------------------------------------------- data access
    def _memory(self, buf: Any, writable: bool) -> tuple[memoryview, list[slice]]:
        """Return the bytes of a buffer and the byte slices it covers, in order."""
        array, runs = _buffer_runs(buf)
        data = _byte_view(array, writable)
        item = array.itemsize
        return data, [
            slice(start * item, (start + length) * item) for start, length in runs
        ]

    def _write(self, offset: int, buf: Any, status: Any) -> int:
        """Write `buf` at etype `offset` of the view; return the etypes written."""
        data, pieces = self._memory(buf, writable=False)
        payload = b"".join(data[piece] for piece in pieces)
        count, rest = divmod(len(payload), self._etype.size)
        if rest:
            raise ValueError("the buffer is not a whole number of elementary types")
        done = 0
        for position, nbytes in self._segments(offset, count):
            self._handle.seek(position)
            self._handle.write(payload[done : done + nbytes])
            done += nbytes
        _fill_status(status, len(payload))
        return count

    def _read(self, offset: int, buf: Any, status: Any) -> int:
        """Read into `buf` from etype `offset` of the view; return the etypes read."""
        data, pieces = self._memory(buf, writable=True)
        wanted = sum(piece.stop - piece.start for piece in pieces)
        count = wanted // self._etype.size
        payload = bytearray()
        for position, nbytes in self._segments(offset, count):
            self._handle.seek(position)
            chunk = self._handle.read(nbytes)
            payload += chunk
            if len(chunk) < nbytes:  # the end of the file
                break
        payload = payload[: len(payload) - len(payload) % self._etype.size]
        done = 0
        for piece in pieces:
            take = min(piece.stop - piece.start, len(payload) - done)
            if take <= 0:
                break
            data[piece.start : piece.start + take] = payload[done : done + take]
            done += take
        _fill_status(status, len(payload))
        return len(payload) // self._etype.size

    def Write_at(self, offset: int, buf: Any, status: Any = None) -> None:
        """Write `buf` at the explicit offset `offset` (in etypes of the view)."""
        self._write(offset, buf, status)

    Write_at_all = Write_at

    def Read_at(self, offset: int, buf: Any, status: Any = None) -> None:
        """Read into `buf` from the explicit offset `offset` (in etypes of the view)."""
        self._read(offset, buf, status)

    Read_at_all = Read_at

    def Write(self, buf: Any, status: Any = None) -> None:
        """Write `buf` at the file pointer and advance it."""
        self._position += self._write(self._position, buf, status)

    Write_all = Write

    def Read(self, buf: Any, status: Any = None) -> None:
        """Read into `buf` from the file pointer and advance it."""
        self._position += self._read(self._position, buf, status)

    Read_all = Read

    def Iwrite_at(self, offset: int, buf: Any) -> Any:
        """Write like ``Write_at`` and return a completed request."""
        from maybempi.serial import SerialRequest

        self.Write_at(offset, buf)
        return SerialRequest()

    def Iread_at(self, offset: int, buf: Any) -> Any:
        """Read like ``Read_at`` and return a completed request."""
        from maybempi.serial import SerialRequest

        self.Read_at(offset, buf)
        return SerialRequest()

    def Iwrite(self, buf: Any) -> Any:
        """Write like ``Write`` and return a completed request."""
        from maybempi.serial import SerialRequest

        self.Write(buf)
        return SerialRequest()

    def Iread(self, buf: Any) -> Any:
        """Read like ``Read`` and return a completed request."""
        from maybempi.serial import SerialRequest

        self.Read(buf)
        return SerialRequest()

    # --------------------------------------------------------- file pointer
    def Seek(self, offset: int, whence: int = SEEK_SET) -> None:
        """Move the file pointer (in etypes of the view).

        Raises:
            ValueError: For ``SEEK_END`` in a view with gaps, or a negative position.
        """
        if whence == SEEK_SET:
            position = offset
        elif whence == SEEK_CUR:
            position = self._position + offset
        elif whence == SEEK_END:
            if self._filetype._runs != ((0, self._filetype._extent),):
                raise ValueError("SEEK_END needs a contiguous view in a serial run")
            end = max(self.Get_size() - self._disp, 0) // self._etype.size
            position = end + offset
        else:
            raise ValueError(
                f"whence must be SEEK_SET, SEEK_CUR or SEEK_END, not {whence}"
            )
        if position < 0:
            raise ValueError(f"the file pointer would be negative: {position}")
        self._position = position

    def Get_position(self) -> int:
        """Return the file pointer, in etypes of the view."""
        return self._position


def _fill_status(status: Any, nbytes: int) -> None:
    """Record a transfer of `nbytes` in an ``MPI.Status``-like object, if given."""
    if status is not None:
        status._set(0, 0, nbytes)
