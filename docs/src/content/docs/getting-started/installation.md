---
title: Installation
description: Install maybempi, with or without mpi4py.
---

maybempi is pure Python with no dependencies. It needs Python 3.10 or newer.

```bash
pip install maybempi
```

To run under MPI, install mpi4py as well, built against the MPI library of your system:

```bash
pip install "maybempi[mpi]"   # same as: pip install maybempi mpi4py
```

Without mpi4py, every process of an `mpirun` job runs serially as rank 0 of 1, and
`maybempi.get_mpi()` warns about it.

Check what maybempi decides in an environment with the `maybempi` command (or
`python -m maybempi`):

```bash
maybempi                      # in a terminal: serial
mpirun -n 2 maybempi --init   # each rank prints its rank and the launcher variable it saw
```

## From source

```bash
git clone https://github.com/max-models/maybempi.git
cd maybempi
pip install -e ".[dev]"
```

| Extra  | Installs                                                  |
| ------ | --------------------------------------------------------- |
| `mpi`  | `mpi4py`                                                  |
| `test` | `pytest`, `pytest-cov` and NumPy                          |
| `docs` | the notebook runner, NumPy and griffe for the API reference |
| `dev`  | formatters and linters, plus the `test` and `docs` extras |
