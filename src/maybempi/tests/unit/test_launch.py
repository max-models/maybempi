"""Tests for the decision: launched_under_mpi, get_mpi, local_rank, is_serial."""

import sys
import types

import pytest

import maybempi
from maybempi import launch
from maybempi.launch import get_mpi, is_serial, launched_under_mpi, local_rank
from maybempi.serial import SerialMPI


@pytest.fixture
def clean_env(monkeypatch):
    """No launcher variables, no override, no mpi4py imported, no cached decision."""
    for variable in (
        *launch.LAUNCHER_VARIABLES,
        *launch.LOCAL_RANK_VARIABLES,
        launch.OVERRIDE_VARIABLE,
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    monkeypatch.setattr(launch, "_AUTO_MPI", None)
    return monkeypatch


@pytest.fixture
def fake_mpi4py(clean_env):
    """A stand-in for an installed mpi4py (without starting MPI)."""
    module = types.ModuleType("mpi4py.MPI")
    module.Is_initialized = lambda: True  # type: ignore[attr-defined]
    package = types.ModuleType("mpi4py")
    package.MPI = module  # type: ignore[attr-defined]
    clean_env.setitem(sys.modules, "mpi4py", package)
    clean_env.setitem(sys.modules, "mpi4py.MPI", module)
    return module


@pytest.fixture
def no_mpi4py(clean_env):
    clean_env.setitem(sys.modules, "mpi4py", None)
    clean_env.setitem(sys.modules, "mpi4py.MPI", None)
    return clean_env


def test_serial_by_default(clean_env):
    assert launched_under_mpi() is False
    assert launch.launcher_variable() is None
    assert isinstance(get_mpi(), SerialMPI)
    assert "mpi4py.MPI" not in sys.modules  # deciding never imports mpi4py


@pytest.mark.parametrize("variable", launch.LAUNCHER_VARIABLES)
def test_launcher_variables(clean_env, variable):
    clean_env.setenv(variable, "3")
    assert launched_under_mpi() is True
    assert launch.launcher_variable() == variable


def test_slurm_batch_script_is_not_an_mpi_launch(clean_env):
    clean_env.setenv("SLURM_PROCID", "0")
    assert launched_under_mpi() is False


@pytest.mark.parametrize(
    ("value", "launcher", "expected"),
    [
        ("1", False, True), ("on", False, True), (" TRUE ", False, True),
        ("0", True, False), ("no", True, False),
        ("maybe", True, True), ("maybe", False, False),
    ],
)  # fmt: skip
def test_override(clean_env, value, launcher, expected):
    clean_env.setenv("MAYBEMPI", value)
    if launcher:
        clean_env.setenv("PMI_RANK", "0")
    assert launched_under_mpi() is expected


def test_initialized_mpi4py_counts_as_mpi(fake_mpi4py):
    assert launched_under_mpi() is True
    fake_mpi4py.Is_initialized = lambda: False
    assert launched_under_mpi() is False
    del fake_mpi4py.Is_initialized
    assert launched_under_mpi() is False


def test_get_mpi_under_a_launcher(fake_mpi4py, clean_env):
    fake_mpi4py.Is_initialized = lambda: False
    clean_env.setenv("OMPI_COMM_WORLD_RANK", "0")
    assert get_mpi() is fake_mpi4py
    clean_env.delenv("OMPI_COMM_WORLD_RANK")
    assert get_mpi() is fake_mpi4py  # decided once per process


def test_get_mpi_explicit(fake_mpi4py):
    assert get_mpi(True) is fake_mpi4py
    assert isinstance(get_mpi(False), SerialMPI)
    assert get_mpi(False) is get_mpi(False)


def test_launcher_without_mpi4py_warns(no_mpi4py):
    no_mpi4py.setenv("PMI_RANK", "0")
    with pytest.warns(RuntimeWarning, match="mpi4py is not installed"):
        assert isinstance(get_mpi(), SerialMPI)


def test_explicit_mpi_without_mpi4py_raises(no_mpi4py):
    with pytest.raises(ImportError):
        get_mpi(True)


def test_local_rank(clean_env):
    assert local_rank() == 0
    clean_env.setenv("SLURM_LOCALID", "not a number")
    assert local_rank() == 0
    clean_env.setenv("LOCAL_RANK", "5")
    assert local_rank() == 5
    clean_env.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "2")  # earlier in the list
    assert local_rank() == 2


def test_is_serial(fake_mpi4py):
    serial = get_mpi(False)
    assert is_serial(serial) and is_serial(serial.COMM_WORLD)
    assert is_serial(serial.COMM_WORLD.Split(0))
    assert not is_serial(fake_mpi4py) and not is_serial(None)


def test_top_level_exports():
    for name in maybempi.__all__:
        assert hasattr(maybempi, name), name
    assert maybempi.get_mpi is get_mpi
    assert maybempi.OVERRIDE_VARIABLE == "MAYBEMPI"
