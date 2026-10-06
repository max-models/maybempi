"""Tests for the ``maybempi`` command."""

import subprocess
import sys
from pathlib import Path

import pytest

import maybempi
from maybempi import __version__, cli, launch


@pytest.fixture
def serial_env(monkeypatch):
    for variable in (*launch.LAUNCHER_VARIABLES, *launch.LOCAL_RANK_VARIABLES):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv(launch.OVERRIDE_VARIABLE, raising=False)
    monkeypatch.setattr(launch, "_AUTO_MPI", None)
    return monkeypatch


def test_report_serial(serial_env, capsys):
    cli.main([])
    out = capsys.readouterr().out
    assert f"maybempi: {__version__}" in out
    assert "launched under MPI: False" in out
    assert "launcher variable: -" in out
    assert "MAYBEMPI override: -" in out
    assert "\nrank:" not in out


def test_report_init_and_override(serial_env, capsys):
    serial_env.setenv("MAYBEMPI", "0")
    serial_env.setenv("PMI_RANK", "1")
    cli.main(["--init"])
    out = capsys.readouterr().out
    assert "launched under MPI: False" in out
    assert "launcher variable: PMI_RANK" in out
    assert "MAYBEMPI override: 0" in out
    assert "MPI: serial stand-in" in out
    assert "rank: 0 of 1" in out


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert capsys.readouterr().out.strip() == __version__


def test_python_dash_m():
    result = subprocess.run(
        [sys.executable, "-m", "maybempi", "--init"],
        capture_output=True,
        text=True,
        check=True,
        env={"MAYBEMPI": "0", "PYTHONPATH": str(Path(maybempi.__file__).parents[1])},
    )
    assert "MPI: serial stand-in" in result.stdout
