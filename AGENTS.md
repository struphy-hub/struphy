# Guidelines for coding agents in Struphy

These are the conventions for AI coding agents (Claude Code, Codex, Copilot, Cursor, ...) working on Struphy.
Human contributors should follow them too. Each rule gives the reason, so it can be applied sensibly to cases it doesn't list.
General contributor information is in `CONTRIBUTING.md` and `doc/sections/dev_guide.rst`.

## Repository layout

- Python package in `src/struphy/`; the FEEC library `feectools` is a git submodule in `feectools/`
  (run `git submodule update --init` after cloning or creating a worktree).
- Performance-critical code lives in `*_kernels.py` files, which Pyccel translates to Fortran with `struphy compile`.
- Examples in `examples/`, notebooks in `tutorials/`, Sphinx docs in `doc/`.

## Branches and pull requests

- `main` (releases) and `devel` are protected. Work on a feature branch and open PRs into **`devel`**.
- Never push to `main`/`devel`, never force-push a shared branch. Merge `devel` into a feature branch instead of rebasing it.
- Keep unrelated work out of an open PR: put it on a separate (possibly stacked) branch.
- Fetch before pushing. GitHub "Update branch" creates merge commits on the remote.

## Adding a model or propagator

Before creating, renaming or substantially changing a model or a propagator, **read the matching guide first**:
- models: `src/struphy/models/AGENTS.md`
- propagators: `src/struphy/propagators/AGENTS.md`

They list the required structure, documentation, registration and tests; CI checks several of them.

## Code style

- Format and lint with ruff before pushing (settings in `pyproject.toml`, line length 120), since CI runs
  `ruff format --check` and `ruff check`:
  ```bash
  ruff check --fix
  ruff format
  ```
- Match the surrounding code: naming, docstring style (numpy-style `Parameters` / `Returns`), comment density.
- After adding, renaming or removing a model, propagator or domain, run `struphy build-init-files`.
  The `__init__.py` files in `models/`, `propagators/` and `geometry/domains/` are generated, so don't edit them by hand.
- New models go in `doc/sections/subsections/models-all.rst` and new propagator modules in
  `doc/sections/subsections/propagators-avail.rst`. CI (`doc/check_models_doc.py`, `doc/check_propagator_doc.py`) fails otherwise.

## MPI

- Import MPI as `from maybempi import MPI` (also `SerialComm`, `launched_under_mpi`), in library code **and** tests.
  Never `from mpi4py import MPI`, because Struphy must run without mpi4py installed.
  Exception: annotation-only imports under `if TYPE_CHECKING:`.
- Collective operations must run on **all** ranks. This includes `Derham.create_spline_function`, evaluating a
  `SplineFunction`, mass-matrix/projector solves, `AccumulatorVector`, and `inner`/norms.
  Never put them inside `if rank == 0:`, not even hidden in an f-string of a `print`, or the run hangs under
  `mpirun -n >1`. Compute on all ranks first, then restrict only printing and plotting to rank 0.
- In serial runs the comm is a stand-in. Don't rely on collectives returning useful values there; when in doubt,
  guard with `if comm.Get_size() > 1`.

## Pyccel kernels

- Kernels are compiled code. After editing a `*_kernels.py` file, or after merging `devel`, recompile with
  `struphy compile -y` and check `struphy compile --status` (all kernels must show as compiled; a parallel `-j N`
  build can skip some silently). Unexpected kernel test failures after a merge usually mean stale `.so` files.
- Pyccel names Fortran modules after the file **basename**. A kernel that other kernels import must live in a
  file whose basename is unique among all `*kernels.py` files (there are e.g. three `utilities_kernels.py`).
- Adding a new cross-module kernel import may need a second `struphy compile` run, since dependencies are read from the old build.
- Don't call `inspect.signature` on kernels: compiled kernels are Fortran builtins without a signature.

## Tests

- CI does **not** auto-discover test folders. When adding a `src/struphy/<pkg>/tests/` folder, add its path
  (relative to `src/struphy/`) in all three places:
  1. `.github/workflows/reusable-unit-testing.yml` (shard `test_files`)
  2. `.github/workflows/reusable-scheduled.yml` (same shard, `TEST_FILES`)
  3. `src/struphy/console/test.py` (`list_of_tests`)
- Mark tests that don't touch parallel code (pure numpy, CLI, I/O helpers, metadata) with
  `@pytest.mark.mpi_skip`. Parallel code means Derham/splines, Particles/PIC, mass operators, projectors and simulations.
- Run MPI tests like CI does: spline/FEEC tests on **2** ranks, particle/PIC tests on **4** ranks.
  FEEC tests on 4 ranks can fail spuriously ("Local number of elements ... too small").
  ```bash
  mpirun -n 2 python -m pytest -p no:xdist --with-mpi src/struphy/feec/tests/...
  ```
  `-p no:xdist` is required: with pytest-xdist loaded, `mpi`-marked tests are skipped.
- Keep local runs small (e.g. ≤ 32³ cells). Never use the full-size profiling cases as tests.

## Performance pitfalls

- When timing under MPI, limit BLAS threads (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, exported with `mpirun -x`).
  Oversubscribed threaded BLAS makes timings meaningless (issue #743).

## Usage guidelines (shipped with the package)

This file is about changing Struphy's code. How to *use* Struphy (writing params files, running, post-processing)
is in `src/struphy/agent_guides/`, which is part of the pip package so that users' agents get the guide that
matches their installed version:
- `USAGE.md`: the entry point, printed by `struphy agents`;
- `skills/setup-simulation/` and `skills/postprocess-simulation/`: skills, linked into `.claude/skills/` of
  this repo and installed into users' projects by `struphy agents --install`.

Keep them in sync with the code. A PR that changes user-facing API (params-file classes, `Simulation`, `Output`,
the CLI) must update the affected guide in the same PR. The guides must work without a source checkout: refer
to CLI commands, importable names and documentation URLs, not to repository paths.
Coding guidelines (`AGENTS.md`/`CLAUDE.md`) must not go into the package.

## Changes to the feectools submodule

1. Commit on a feature branch in `feectools/` and open the feectools PR into **`devel-tiny`**.
   Every PR there must raise the **minor** version in feectools' `pyproject.toml` (e.g. 0.1.11 → 0.2.0).
   CI fails if the version is not higher than the latest release on PyPI.
2. Open a paired struphy PR into `devel` whose submodule pointer is the feectools feature-branch commit,
   so the full struphy test suite validates the change.
3. Mention each PR in the other's description. Don't raise struphy's `feectools` version pin before the new
   version is on PyPI.
