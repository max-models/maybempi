"""Tests for petsc_comm, with a stand-in for petsc4py and, if installed, the real one."""

import importlib.util
import sys
import types

import pytest

import maybempi
from maybempi import petsc_comm

has_petsc4py = importlib.util.find_spec("petsc4py") is not None


class FakeComm:
    """Records the communicator it wraps, as ``PETSc.Comm(comm)`` does."""

    def __init__(self, comm=None):
        self.wrapped = comm


class FakeMPIComm:
    """An mpi4py-like communicator of ``size`` processes."""

    def __init__(self, size):
        self.size = size

    def Get_size(self):
        return self.size


@pytest.fixture
def fake_petsc(monkeypatch):
    """A petsc4py package whose PETSc module has Comm and COMM_SELF."""
    module = types.ModuleType("petsc4py.PETSc")
    module.Comm = FakeComm  # type: ignore[attr-defined]
    module.COMM_SELF = FakeComm("self")  # type: ignore[attr-defined]
    package = types.ModuleType("petsc4py")
    package.PETSc = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "petsc4py", package)
    monkeypatch.setitem(sys.modules, "petsc4py.PETSc", module)
    return module


def test_serial_comm_is_comm_self(fake_petsc):
    serial = maybempi.get_mpi(False)
    assert petsc_comm(serial.COMM_WORLD) is fake_petsc.COMM_SELF
    assert petsc_comm(serial.COMM_SELF) is fake_petsc.COMM_SELF


def test_default_is_comm_world_of_get_mpi(fake_petsc, monkeypatch):
    monkeypatch.setattr(
        maybempi.petsc, "get_mpi", lambda: maybempi.get_mpi(False)
    )
    assert petsc_comm() is fake_petsc.COMM_SELF


def test_one_process_is_comm_self(fake_petsc):
    assert petsc_comm(FakeMPIComm(1)) is fake_petsc.COMM_SELF


def test_several_processes_are_wrapped(fake_petsc):
    comm = FakeMPIComm(4)
    result = petsc_comm(comm)
    assert isinstance(result, FakeComm)
    assert result.wrapped is comm


def test_petsc_comm_passes_through(fake_petsc):
    comm = FakeComm("world")
    assert petsc_comm(comm) is comm


def test_missing_petsc4py_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "petsc4py", None)  # import raises ImportError
    with pytest.raises(ImportError, match="pip install petsc4py"):
        petsc_comm(maybempi.get_mpi(False).COMM_WORLD)


@pytest.mark.skipif(not has_petsc4py, reason="needs petsc4py")
def test_real_petsc_on_the_serial_stand_in():
    from petsc4py import PETSc

    comm = petsc_comm(maybempi.get_mpi(False).COMM_WORLD)
    assert comm == PETSc.COMM_SELF
    assert comm.getSize() == 1
    vector = PETSc.Vec().createMPI(5, comm=comm)
    vector.set(2.0)
    assert vector.sum() == pytest.approx(10.0)
    vector.destroy()
