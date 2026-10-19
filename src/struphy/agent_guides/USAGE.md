# Using Struphy: guide for AI coding agents

Struphy solves plasma physics PDEs with structure-preserving discretizations (FEEC, PIC, SPH).
Users set up a simulation in a Python script (`params_<name>.py`), run it serially or with MPI, and
post-process the saved output with the `Output` API. This guide ships with the installed package and
matches its version; `struphy agents` prints it. It is about *using* Struphy. Rules for changing
Struphy's own source code are in `AGENTS.md` of the Struphy repository.

## Skills

Detailed instructions are in the skills next to this file (`skills/<name>/SKILL.md`).
Read the matching one before writing code:

- **setup-simulation**: write or edit a params file (model, domain, grid, time stepping, species,
  backgrounds, perturbations, propagator options) and run it.
- **postprocess-simulation**: read, evaluate, plot and analyse a run's output folder.

`struphy agents --install` copies them into `.claude/skills/` of a project and adds a pointer to the
project's `AGENTS.md` and `CLAUDE.md`.

## Check the installation first

```bash
struphy --version
struphy compile --status   # all kernels must be compiled; otherwise run `struphy compile`
```

Struphy's compute kernels are compiled after `pip install` with `struphy compile`. A kernel that isn't
compiled fails at runtime. Install help: https://struphy-hub.github.io/struphy/sections/install.html.

## Command line

| Command | Purpose |
| --- | --- |
| `struphy params --help` | list all models |
| `struphy --fluid` / `--kinetic` / `--hybrid` | list one group of models |
| `struphy params <Model> [-y]` | write a default `params_<Model>.py` |
| `struphy params <Model> --check-file <file>` | validate a params file |
| `struphy output info\|keys\|pproc\|report <folder>` | inspect, post-process or report a run's output |
| `struphy compile [--status]` | compile the kernels, or show their status |
| `struphy agents [--install [DIR]]` | print this guide, or install the skills into a project |

## Running

```bash
python params_<name>.py                 # serial
mpirun -n 4 python params_<name>.py     # MPI (needs `pip install mpi4py`); the script is unchanged
```

- Keep `sim.run()` under `if __name__ == "__main__":` so that importing the script doesn't start a run.
- Start with a small grid and a short end time to check the setup, then scale up.
- When timing MPI runs, set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`; oversubscribed threaded BLAS
  makes timings meaningless.

## Ground rules

- Don't invent model, species, variable, propagator or option names. Get them from
  `struphy params <Model>`, `help(<class>)` or a matching example, because every model differs.
- Examples (`examples/<Model>/<case>/`) are the best starting point. They are in the Struphy repository,
  not in the pip package; `struphy agents` prints the URL for the installed version.
- Post-processing reads the output folder (`Output("sim_1")`); never re-run or import the params file for it.
- Under MPI, call collective operations (`sim.run()`, `out.pproc()`, `out.evaluate()`) on all ranks.
  Restrict only printing and plotting to rank 0.

## Documentation

- Quickstart: https://struphy-hub.github.io/struphy/sections/quickstart.html
- User guide: https://struphy-hub.github.io/struphy/sections/userguide.html
- Models: https://struphy-hub.github.io/struphy/sections/models.html
- Python API: https://struphy-hub.github.io/struphy/sections/api_guide.html
