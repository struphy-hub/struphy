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
- In PR descriptions (GitHub Markdown), escape `*` and `_` in formulas as `\*` and `\_` (e.g. `M\_2`, `dt\*\*2`),
  otherwise GitHub renders them as italics/bold; or put the formula in backticks.

## Adding a model or propagator

Before creating, renaming or substantially changing a model or a propagator, **read the matching guide first**:
- models: `src/struphy/models/AGENTS.md`
- propagators: `src/struphy/propagators/AGENTS.md`

They list the required structure, documentation, registration and tests; CI checks several of them.

## Profiling cases

Before adding or changing a profiling case (a `submit_*.py` script next to an example, run by the driver in
`profiling/` and uploaded to struphy-hub/profiling-data), read the skill `.claude/skills/profiling-case/SKILL.md`.

## Studies

Studies (pages with runs, findings and videos on https://struphy-hub.github.io/struphy-studies/) are made in the
repository struphy-hub/struphy-studies, see the skill `.claude/skills/create-study/SKILL.md`.

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

- Test files go into `tests/` folders. CI does **not** auto-discover test folders: when adding a
  `src/struphy/<pkg>/tests/` folder or a `src/struphy/models/tests/test_*.py` file, add its path (relative to
  `src/struphy/`) to a shard in `src/struphy/console/unit_test_shards.py`. The CI workflows and `struphy test unit`
  read that list, and `console/tests/test_unit_test_shards.py` fails if a test file is not covered.
- Markers must be registered in `pyproject.toml` (`--strict-markers`).
- The weekly workflow `coverage.yml` reports the line coverage of the Python code by the unit tests (job summary and
  HTML artifact; kernels are compiled and not measured). Run it on a branch with `gh workflow run coverage.yml --ref <branch>`.
- Tests that write output (`Simulation`, `EnvironmentOptions(out_folders=...)`) use the `mpi_out_folder` fixture
  (`src/struphy/conftest.py`): a folder per test in `$STRUPHY_TEST_TMPDIR` (default: system temp dir), the same on all
  MPI ranks, removed unless the test failed on some rank (then kept, with its path in a warning). Don't write into the current
  working directory and don't remove output folders in the test itself. Don't combine it with pytest's `tmp_path`
  for files that other ranks read (each rank has its own `tmp_path`).
- NumPy's global random generator is seeded before every test (per rank); use a seed for any other generator.
- Failed scheduled runs and failed runs on `devel` open (or comment on) an issue labelled `ci-failure`.
  Close it once the workflow is green again.
- Mark tests that don't touch parallel code (pure numpy, CLI, I/O helpers, metadata) with
  `@pytest.mark.mpi_skip`. Parallel code means Derham/splines, Particles/PIC, mass operators, projectors and simulations.
- Run MPI tests like CI does: spline/FEEC tests on **2** ranks, particle/PIC tests on **4** ranks.
  FEEC tests on 4 ranks can fail spuriously ("Local number of elements ... too small").
  ```bash
  mpirun -n 2 python -m pytest -p no:xdist --with-mpi src/struphy/feec/tests/...
  ```
  `-p no:xdist` is required: with pytest-xdist loaded, `mpi`-marked tests are skipped.
- The model tests in `src/struphy/models/tests/regression/` compare every model against stored baselines (1 and 2
  ranks). A change that intentionally changes results must update the baselines in the same PR, see
  `src/struphy/models/AGENTS.md`.
- The verification tests in `src/struphy/models/tests/verification/` run on PRs only if marked `verif_pr` (a small
  core set) or if their file is changed by the PR; all of them run in the scheduled CI (Sun/Wed on `devel`).
  Before merging a PR that changes a model or shared numerics, run the affected verification tests locally, or run all
  of them on the branch with `gh workflow run test-PR-models.yml --ref <branch>` (a manual run selects everything).
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
