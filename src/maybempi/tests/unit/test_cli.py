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


def _rows(out):
    """The table rows as lists of cells (two or more spaces separate the cells)."""
    import re

    return [re.split(r"\s{2,}", line.strip()) for line in out.strip().splitlines()]


def test_report_serial(serial_env, capsys):
    cli.main([])
    rows = _rows(capsys.readouterr().out)
    assert rows[0] == ["item", "value"] and set(rows[1][0]) == {"-"}
    assert ["maybempi", __version__] in rows
    assert ["launched under MPI", "False"] in rows
    assert ["launcher variable", "-"] in rows
    assert ["MAYBEMPI override", "-"] in rows
    assert not any(row[0] == "rank" for row in rows)


def test_report_init_and_override(serial_env, capsys):
    serial_env.setenv("MAYBEMPI", "0")
    serial_env.setenv("PMI_RANK", "1")
    cli.main(["--init"])
    rows = _rows(capsys.readouterr().out)
    assert ["launched under MPI", "False"] in rows
    assert ["launcher variable", "PMI_RANK"] in rows
    assert ["MAYBEMPI override", "0"] in rows
    assert ["MPI", "serial stand-in"] in rows
    assert ["rank", "0 of 1"] in rows


def test_columns_are_aligned():
    table = cli._table(("a", "longer"), [["xxxxx", 1], ["y", 22]])
    assert table.splitlines() == [
        "a      longer",
        "-----  ------",
        "xxxxx  1",
        "y      22",
    ]


def test_reports_of_all_ranks():
    base = cli.info(init=False) | {"MPI": "mpi4py", "size": 2}
    reports = [
        base | {"rank": 0, "local rank": 0, "launcher variable": "OMPI_COMM_WORLD_RANK"},
        base | {"rank": 1, "local rank": 1, "launcher variable": "OMPI_COMM_WORLD_RANK",
                "mpi4py installed": False},
    ]  # fmt: skip
    common, per_rank = cli.format_reports(reports).split("\n\n")
    common_rows = _rows(common)
    assert ["size", "2"] in common_rows and ["MPI", "mpi4py"] in common_rows
    assert not any(
        row[0] in ("rank", "host", "mpi4py installed") for row in common_rows
    )
    header, _, *rows = _rows(per_rank)
    # the fixed columns first, then what differs between ranks
    assert header == [
        "rank",
        "host",
        "local rank",
        "launcher variable",
        "mpi4py installed",
    ]
    assert rows[0][0] == "0" and rows[1][0] == "1"
    assert rows[1][-1] == "False"


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
    assert ["MPI", "serial stand-in"] in _rows(result.stdout)
