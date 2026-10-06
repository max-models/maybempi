"""MPI.File of the serial stand-in: MPI-IO on one process."""

import numpy as np
import pytest

from maybempi import get_mpi

MPI = get_mpi(False)
comm = MPI.COMM_WORLD


def test_write_and_read_back(tmp_path):
    path = tmp_path / "data.bin"
    data = np.arange(10.0)
    handle = MPI.File.Open(comm, path, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    status = MPI.Status()
    handle.Write_at(0, data, status)
    assert status.Get_count(MPI.DOUBLE) == 10
    handle.Write_at_all(80, np.array([1.5]))  # bytes after the data
    assert handle.Get_size() == 88
    assert handle.Get_amode() == MPI.MODE_WRONLY | MPI.MODE_CREATE
    handle.Close()
    np.testing.assert_array_equal(np.fromfile(path)[:10], data)
    with MPI.File.Open(comm, str(path)) as readable:
        back = np.zeros(11)
        readable.Read_at_all(0, back)
        assert back[-1] == 1.5
        readable.Read_at(8, back[:2])
        np.testing.assert_array_equal(back[:2], [1, 2])


def test_a_subarray_view_writes_the_block_where_numpy_has_it(tmp_path):
    path = tmp_path / "grid.bin"
    full = np.arange(20.0).reshape(4, 5)
    with MPI.File.Open(comm, path, MPI.MODE_RDWR | MPI.MODE_CREATE) as handle:
        handle.Set_size(16 + full.nbytes)
        for start in (0, 2):  # two blocks of rows, as two ranks would write them
            block = MPI.DOUBLE.Create_subarray([4, 5], [2, 3], [start, 1]).Commit()
            handle.Set_view(16, MPI.DOUBLE, block)
            handle.Write_all(np.ascontiguousarray(full[start : start + 2, 1:4]))
            assert handle.Get_position() == 6
        handle.Set_view(16, MPI.DOUBLE)
        assert handle.Get_view()[0] == 16 and handle.Get_view()[3] == "native"
        everything = np.zeros(20)
        handle.Read_all(everything)
    written = np.fromfile(path)[2:].reshape(4, 5)
    np.testing.assert_array_equal(written[:, 1:4], full[:, 1:4])
    np.testing.assert_array_equal(written[:, [0, 4]], 0)
    np.testing.assert_array_equal(everything.reshape(4, 5), written)
    with MPI.File.Open(comm, path) as handle:
        block = MPI.DOUBLE.Create_subarray([4, 5], [2, 3], [2, 1]).Commit()
        handle.Set_view(16, MPI.DOUBLE, block)
        assert handle.Get_byte_offset(0) == 16 + (2 * 5 + 1) * 8
        part = np.zeros((2, 3))
        handle.Read(part)
        np.testing.assert_array_equal(part, full[2:4, 1:4])


def test_file_pointer_and_seek(tmp_path):
    path = tmp_path / "pointer.bin"
    with MPI.File.Open(comm, path, MPI.MODE_RDWR | MPI.MODE_CREATE) as handle:
        handle.Set_view(0, MPI.INT32_T)
        handle.Write(np.arange(4, dtype=np.int32))
        handle.Iwrite(np.arange(4, 6, dtype=np.int32)).Wait()
        assert handle.Get_position() == 6
        handle.Seek(1)
        value = np.zeros(1, dtype=np.int32)
        handle.Iread(value).Wait()
        assert value[0] == 1 and handle.Get_position() == 2
        handle.Seek(1, MPI.SEEK_CUR)
        handle.Read(value)
        assert value[0] == 3
        handle.Seek(-1, MPI.SEEK_END)
        handle.Read(value)
        assert value[0] == 5
        handle.Iwrite_at(0, np.array([9], dtype=np.int32)).Wait()
        handle.Iread_at(0, value).Wait()
        assert value[0] == 9
        with pytest.raises(ValueError, match="negative"):
            handle.Seek(-1)
        with pytest.raises(ValueError, match="whence"):
            handle.Seek(0, 1)
        handle.Set_view(0, MPI.INT32_T, MPI.INT32_T.Create_vector(2, 1, 2))
        with pytest.raises(ValueError, match="contiguous view"):
            handle.Seek(0, MPI.SEEK_END)
        handle.Sync()
        handle.Set_atomicity(True)
        assert handle.Get_atomicity()


def test_reading_past_the_end(tmp_path):
    path = tmp_path / "short.bin"
    np.arange(3.0).tofile(path)
    with MPI.File.Open(comm, path) as handle:
        handle.Set_view(0, MPI.DOUBLE)
        back = np.full(5, -1.0)
        status = MPI.Status()
        handle.Read_at(1, back, status)
        assert status.Get_count(MPI.DOUBLE) == 2
        np.testing.assert_array_equal(back, [1, 2, -1, -1, -1])
        handle.Set_view(0, MPI.DOUBLE, MPI.DOUBLE.Create_vector(2, 1, 2))
        handle.Read_at(0, back[:3])  # elements 0, 2 and then 4, past the end
        np.testing.assert_array_equal(back[:3], [0, 2, -1])  # the third is past the end
        handle.Set_view(16, MPI.DOUBLE)  # one element left in the file
        spread = np.full(6, -1.0)
        handle.Read_at(0, [spread, 1, MPI.DOUBLE.Create_vector(3, 1, 2)])
        np.testing.assert_array_equal(spread, [2, -1, -1, -1, -1, -1])


def test_modes(tmp_path):
    path = tmp_path / "modes.bin"
    with pytest.raises(FileNotFoundError):
        MPI.File.Open(comm, path)
    with pytest.raises(ValueError, match="exactly one"):
        MPI.File.Open(comm, path, MPI.MODE_RDONLY | MPI.MODE_RDWR)
    with pytest.raises(ValueError, match="write access"):
        MPI.File.Open(comm, path, MPI.MODE_RDONLY | MPI.MODE_CREATE)
    handle = MPI.File.Open(
        comm, path, MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_EXCL
    )
    handle.Write(np.arange(2.0))
    handle.Close()
    with pytest.raises(FileExistsError):
        MPI.File.Open(comm, path, MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_EXCL)
    with MPI.File.Open(comm, path, MPI.MODE_WRONLY | MPI.MODE_APPEND) as handle:
        assert handle.Get_position() == 16  # in bytes: the default view
        handle.Write(np.array([7.0]))
    np.testing.assert_array_equal(np.fromfile(path), [0, 1, 7])  # not truncated
    with MPI.File.Open(comm, path, MPI.MODE_RDWR) as handle:
        handle.Preallocate(64)
        handle.Preallocate(8)  # never shrinks
        assert handle.Get_size() == 64
        assert "modes.bin" in repr(handle)
    handle = MPI.File.Open(comm, path, MPI.MODE_RDWR | MPI.MODE_DELETE_ON_CLOSE)
    handle.Close()
    assert not path.exists()
    np.arange(2.0).tofile(path)
    MPI.File.Delete(path)
    assert not path.exists() and not MPI.FILE_NULL


def test_view_and_buffer_errors(tmp_path):
    path = tmp_path / "errors.bin"
    with MPI.File.Open(comm, path, MPI.MODE_RDWR | MPI.MODE_CREATE) as handle:
        with pytest.raises(ValueError, match="native"):
            handle.Set_view(0, MPI.DOUBLE, datarep="external32")
        with pytest.raises(ValueError, match="named type"):
            handle.Set_view(0, MPI.DOUBLE.Create_contiguous(2))
        with pytest.raises(ValueError, match="not built from"):
            handle.Set_view(0, MPI.DOUBLE, MPI.INT32_T.Create_contiguous(2))
        handle.Set_view(0, MPI.DOUBLE, MPI.DOUBLE.Create_contiguous(0))
        assert handle.Get_byte_offset(3) == 0
        with pytest.raises(ValueError, match="holds no data"):
            handle.Write(np.ones(1))
        handle.Set_view(0, MPI.DOUBLE)
        with pytest.raises(ValueError, match="whole number"):
            handle.Write(np.ones(1, dtype=np.int32))
        with pytest.raises(ValueError, match="C-contiguous"):
            handle.Read(np.zeros((2, 2))[:, 0])
        handle.Write(np.ones((2, 2))[:, 0])  # copied for writing
        assert handle.Get_size() == 16

        class Device:
            def __init__(self, data):
                self.data = data
                self.size = data.size
                self.itemsize = data.itemsize

            def get(self):
                return self.data

        handle.Write_at(0, Device(np.full(2, 3.0)))
        back = np.zeros(2)
        handle.Read_at(0, back)
        np.testing.assert_array_equal(back, 3)
        with pytest.raises(TypeError, match="host arrays"):
            handle.Read_at(0, Device(np.zeros(2)))
