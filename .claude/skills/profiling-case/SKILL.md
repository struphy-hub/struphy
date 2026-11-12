---
name: profiling-case
description: Use when adding or changing a Struphy profiling case - a submit_*.py script next to an example in examples/ that runs the example on several MPI rank counts (strong/weak scaling, CPU vs GPU, preconditioner or discretization comparisons) via the driver in profiling/, and packages and uploads the results to struphy-hub/profiling-data. Triggers on requests like "make a scaling study of example X", "add a profiling case", "draft a submit script", "compare preconditioners on N ranks", "upload to profiling-data".
---

# Profiling cases

A profiling case is an example of `examples/<Model>/<case>/` that is run on several rank counts (or backends) by a
`submit_<case>*.py` script next to it. The driver in `profiling/` (`profiling_job.py`: `ProfilingCase`, `sweep.py`,
`clusters.py`, `upload.py`) builds one job script per run, submits it under SLURM (or runs it locally), and packages
each finished run (scope-profiler data, run metadata, the files of `<sim_dir>/results/`) into a case folder of the
`struphy-hub/profiling-data` repo. Each **case** (label) is one series of runs there, e.g. one preconditioner on 4 to
64 ranks.

Templates (read one before writing a new script):
- `examples/Poisson/gaussian_hollow_cylinder/submit_gaussian_hollow_cylinder.py`: strong scaling, one case per
  `--variant` (preconditioner), `VARIANTS = {key: (label, name, description, flags)}`;
- `examples/LinearMHDDriftkineticCC/lowmode_tae/submit_lowmode_tae_cpu_vs_gpu.py` with its params and pproc files: CPU
  vs GPU, `--setup profiling` in the params file and `--save-results` in the pproc script;
- `examples/ShearAlfven/itpa_tae_benchmark/submit_ShearAlfven_scaling.py` (from #932): variants with their own grid,
  rank counts and time limit, `--ranks`, `--steps`, `--time`, `--slurm-preset`, solver iterations in the summary;
- `examples/Poisson/sweep_poisson.py`: all variants of several scripts in one process (`profiling/sweep.py`).

## 1. Agree on the design with the user first

Ask (or propose) before writing: the cluster and partition (is there a preset in `profiling/clusters.py`?), rank
counts and nodes, the variants (one case each: space, preconditioner, grid, ...), the grid of each variant, the
number of time steps per run, and whether every run gets a whole node (comparable conditions vs. queue time). The
grid must be large enough for the largest rank count (see 4.) and physically meaningful (a variant that gives the
wrong physics on its grid is not worth profiling).

## 2. The params file: `--setup profiling`

The job script runs the params file (`params_<case>.py`) directly with the case's `param_flags`, plus
`--out-folder` and `--sim-folder` set by the driver; the pproc script imports `args` and `parser` from it. So parse
with `parser.parse_known_args()` (unknown flags, e.g. those of the pproc, are ignored) and create the parser with
`add_help=__name__ == "__main__"`.
- Add `--setup {example,profiling}` and a step count (e.g. `--profiling-steps`, default 50): with `profiling`, set
  `Time(dt, Tend=round(steps * dt, 14))`, `save_restart=False`, and nothing else that differs from the example
  physics. The first step includes one-time work (allocations, first solver calls), so take the median of the
  others.
- Run with **`sim.run(profiling_activated=True)`** in the `__main__` block: without it no scope-profiler data is
  written and nothing is packaged.
- Give `ProfilingOptions(label=...)` a label such as `"<N> ranks"` (from `MPI.COMM_WORLD.Get_size()`, with
  `from maybempi import MPI, SerialMPI`): it names the run in the comparison plots of the profiling-data page.
- Solver iterations are the key quantity of preconditioner comparisons: set `SolverParameters(info=profiling)` for the
  solver in question and make sure the struphy logger is at INFO level (`set_logging_level(logging.INFO)` in the
  params file; the default is WARNING, and then nothing is logged). Many propagators then log
  `Iterations for <Name>: N` on rank 0, but the name and format differ between propagators (some log the whole info
  dict, some nothing): check the propagator's code.
- Limit BLAS threads in the params file before NumPy is imported (`os.environ.setdefault("OMP_NUM_THREADS", "1")`
  etc., see `examples/Poisson/gaussian_cube/params_gaussian_cube.py`): the driver does not, and oversubscribed
  threaded BLAS makes MPI timings meaningless (issue #743).
- Every variant must be selectable by flags (`--precond`, `--b-space`, `--num-elements`, ...), with a `parser.error`
  for invalid combinations.

## 3. The pproc script: `--save-results`

The driver runs `python pproc_<case>.py <sim_dir> <param_flags>` serially after each run; everything it writes into
`<sim_dir>/results/` (figures, arrays, `.html`, `.json`) is packaged. **Put `--save-results` into the `param_flags` of
the submit script** (the params file ignores it). With it (error if the run was not `--setup profiling`), write:
- the scope-profiler report: `create_html_report(Output(path).profile.results, results/"report.html")`;
- `results/summary.json`: the variant, ranks, grid, steps, the time per step from the `"model.integrate"` region (max
  over ranks per step), its median without the first step, and the solver iterations per step.

Read the iterations from `<sim_dir>/struphy.out` (the stdout of the run, written by the driver): `struphy.log` is a
rotating log (10 kB per file, 3 backups) and loses the early steps of a long run.

## 4. The submit script

- `sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "profiling"))`, then `from profiling_job import
  ProfilingCase`.
- Keep the flags `--upload`, `--language`, `--compiler` of the templates; add `--variant` (choices `VARIANTS`) and, as
  needed, `--ranks`, `--steps`, `--time`, `--slurm-preset`.
- `VARIANTS`: one entry per case, keyed by a name of the variant (and grid, e.g. `hdiv_stiffness_24x88x8`), with the
  flags, grid, rank counts and a SLURM time limit; `sweep.py` reads the keys.
- `ProfilingCase(label=..., name=..., description=..., physics_problem=..., struphy_model_used=..., params_source=...,
  pproc_source=..., language=..., compiler=..., upload=...)`. Labels are permanent folder names in profiling-data:
  lowercase, model and case, then the variant. The description states the domain, grid, degrees, dt, steps and solver.
- `profiling_case.launch(n, num_nodes=1, param_flags=..., slurm_preset=..., slurm_options={"time": ...})` per rank
  count, then `profiling_case.finalize_run()`.
- Time limits: measure the cost per step of each variant on a few steps first, size the limit for the smallest rank
  count with a margin, and scale it with `--steps`. A run that hits the limit is lost and its pproc does not run.
- For all variants at once, add a `sweep_<case>.py` next to the case folder (see `examples/Poisson/sweep_poisson.py`):
  `run_sweep(variant_entries(<submit script>), upload=..., dry_run=...)`.
- `submit_*.py` (and `sweep_*.py`, since #932) are excluded from the struphy-examples sync
  (`.github/workflows/create-PR-examples.yml`), since they need `profiling/`; check that a new kind of script is too.

Clusters: `launch` takes the preset `clusters.SLURM_PRESETS[slurm_preset or detect_machine_name()]` and, without
`num_nodes`, `HARDWARE_INFO[...]["cpus_per_node"]`. On a cluster with a small home file system, add a
`SCRATCH_DIRS` entry so that the runs go to the scratch file system (Raven: `/ptmp/<user>/`). A detected cluster
without a preset raises a KeyError: add a `SLURM_PRESETS` entry (and a `HARDWARE_INFO` entry, or always pass
`num_nodes`), or pass `slurm_preset`. Check the first jobs of a new preset (allocation, memory, time limits) before
submitting everything.

Decomposition and memory:
- Check that every rank count decomposes the grid with at least `degree` elements per rank and direction:
  `feectools.ddm.partition.compute_dims_general(n, num_elements, mpi_dims_mask)`. Polar splines need enough elements
  along eta1 on the pole rank and a multiple of 3 elements along eta2.
- Preconditioners that hold global dense data on every rank (e.g. `ToroidalFourierPreconditioner`: one dense poloidal
  factorization per toroidal mode) do not shrink with the rank count: estimate the memory per rank from a smaller
  grid and limit their rank counts.
- Under SLURM, jobs on a fraction of a node may share it with other jobs (memory bandwidth, cache), which disturbs
  the timings of the small rank counts; a preset can request the whole node (e.g. its full memory with `mem`).

## 5. Test locally before submitting

With the venv activated (`VIRTUAL_ENV` must be set; the driver compiles the kernels if needed):
```bash
STRUPHY_PROFILING_LOCAL=1 python examples/<Model>/<case>/submit_<case>.py --variant <v> --ranks 2 --steps 3
```
This runs the job scripts locally; rank counts above `os.cpu_count()` are skipped, and `mpirun` may also refuse
counts above the physical cores. Output, in the profiling folder `<base>` (`$STRUPHY_PROFILING_DIR` if set, else the
`clusters.SCRATCH_DIRS` entry of the detected cluster, e.g. `/ptmp/<user>/struphy_profiling` on Raven, else
`_profiling_jobs/` in the repo):
- the job scripts and SLURM logs: `<base>/job_profile_*.sh`, `<base>/profiling_*.out|err`;
- the runs: `<base>/<timestamp>-<commit>/<label>/sim_01/` (`struphy.out`, `pproc.out`, `results/`);
- the packaged case: `<base>/profiling-data/<timestamp>-<commit>-<label>/results-run01/` (`summary.json`,
  `report.html`, the `.h5`).

Every call without `--upload` empties `<base>/profiling-data/` first (and `--upload` re-clones it): push a
packaged case you want to keep with `profiling/upload.py <case folder>` before the next call.

Check `struphy.out`, `pproc.out` and `summary.json` (iterations of every step present). Test each variant at least
once on a small grid, the full grid on a few steps for the cost per step and the memory, and `python
sweep_<case>.py --dry-run`. Also run the example's `--test-run` (used by `test_examples.py`), which must behave as
before. Ask the user before local runs that take long.

## 6. Submit and upload

On the cluster (login node, venv activated): `python sweep_<case>.py --upload` or `python submit_<case>.py --variant
<v> --upload`, kept running (e.g. in tmux) until all jobs are packaged; `--upload` clones
`git@github.com:struphy-hub/profiling-data.git` (or `$PROFILING_DATA_REPO_URL`) up front and pushes every run as its
job finishes. Without `--upload`, the script prints how to push a packaged case later with `profiling/upload.py`.

## 7. PR

New cases go into a PR into `devel` (params/pproc changes, submit and sweep scripts, preset changes in
`profiling/clusters.py`), with the variants, the measured cost per step and iterations, and the local tests in the
description. Run `ruff check` and `ruff format` on the changed files.
