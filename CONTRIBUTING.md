# Contributing

Thanks for taking the time to contribute.

## Set up

```bash
git clone https://github.com/max-models/maybempi.git
cd maybempi
make install          # uv sync with the dev extra and the pre-commit hooks
```

Without uv:

```bash
python -m venv env && source env/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Develop

- Format and lint with [ruff](https://docs.astral.sh/ruff/); the pre-commit hooks run it for you.
- Type-check with `pyright src/`.
- Add tests under `src/maybempi/tests/unit/` and run `pytest .`. Tests that need several
  ranks go in `src/maybempi/tests/mpi/`; run them with
  `mpiexec -n 2 python -m pytest src/maybempi/tests/mpi` (CI does, with Open MPI and MPICH).
- Every public module, class and function needs a Google-style docstring. They are rendered
  in the API reference and checked by ruff's `D` rules.
- Preview the documentation with `make docs-dev`; see the
  [docs guide](https://max-models.github.io/maybempi/development/docs/).

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `ci:`. A `feat!:` or a
`BREAKING CHANGE:` footer marks a breaking change. Before a release, update the package version
and changelog manually.

## Pull requests

Open the PR against `devel`. CI runs the tests on every supported Python version, ruff, pyright,
the tutorials and the documentation build. One approving review is required to merge.

## Releases

Before merging a release to `main`, update the versions in `pyproject.toml`,
`src/maybempi/__init__.py` and `CITATION.cff` (including its release date), and add the release
notes to `CHANGELOG.md`. The push creates a GitHub release with a version tag and publishes the
package to PyPI.
