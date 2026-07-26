# Struphy Profiling System

This directory contains the infrastructure for running profiling jobs on SLURM clusters, and on a laptop or workstation without a batch system. The system is designed to be easily extensible for adding new profiling cases.

## Directory Structure

```
profiling/
├── README.md                      # This file
├── profiling_job.py              # Core profiling infrastructure (shared)
├── package_profiling_results.py  # Results packaging utilities
├── common_commands.sh            # Common shell commands
├── submit_<model>_job.py         # Job submission scripts (one per model/case)
├── run_<model>.py                # Job execution scripts (one per model/case)
├── examples/                     # Profiling-specific parameter files
│   └── <Model>/
│       └── <case_name>/
│           └── params_<case>.py  # Parameter file for the profiling case
├── results/                      # Output directory (gitignored)
│   └── profiling/
│       └── <timestamp>-<commit>/
│           └── <case_label>/
└── tests/                        # Tests for profiling infrastructure
```

## How It Works

The profiling system consists of three main components:

1. **Core Infrastructure** (`profiling_job.py`): Provides the `ProfilingCase` dataclass and `run_profiling_job()` function that handle:
   - Compiling Struphy kernels
   - Generating the job script (SLURM batch script, or plain bash when there is no batch system)
   - Submitting the job to the cluster, or running it locally
   - Waiting for completion
   - Packaging and uploading results

2. **Job Submission Scripts** (`submit_<model>_job.py`): Define the specific profiling cases for a model and delegate to the core infrastructure.

3. **Job Execution Scripts** (`run_<model>.py`): Execute the actual simulation with profiling enabled. These are called by the SLURM jobs.

## Adding a New Profiling Job

Follow these steps to add a new profiling job for a different model or physics case:

### 1. Prepare Your Parameter File

Copy the parameter file you want to profile to the appropriate location:

```bash
mkdir -p profiling/examples/<ModelName>/<case_name>/
cp <source_params>.py profiling/examples/<ModelName>/<case_name>/params_<case>.py
```

**Example:**
```bash
mkdir -p profiling/examples/VlasovMaxwell/landau_damping/
cp examples/VlasovMaxwell/landau_damping/params_landau.py \
   profiling/examples/VlasovMaxwell/landau_damping/
```

### 2. Create the Execution Script

Create `profiling/run_<model>.py` based on the template below:

```python
"""Run the <model> profiling case."""
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

from struphy import EnvironmentOptions, set_logging_level

set_logging_level(logging.INFO)

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
params_dir = script_dir / "examples" / "<ModelName>" / "<case_name>"
default_out_root = repo_root / "results" / "profiling" / "manual" / "<case_label>"

sys.path.insert(0, str(params_dir))
from params_<case> import sim


def main() -> None:
    parser = ArgumentParser(description="Run the <model> profiling case.")
    parser.add_argument("nranks", type=int, help="Number of MPI ranks used for the run")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=default_out_root,
        help="Output root for this testcase. Each rank writes under OUT_ROOT/sim_ranks<N>.",
    )
    args = parser.parse_args()
    num_ranks = args.nranks

    env = EnvironmentOptions(
        out_folders=str(args.out_root.expanduser().resolve()),
        sim_folder=f"sim_ranks{num_ranks}",
        profiling_activated=True,
        profiling_trace=True,
    )

    sim.env = env
    sim._setup_folders()

    sim.run(one_time_step=True)  # Or adjust as needed for your case


if __name__ == "__main__":
    main()
```

**Key points:**
- The script must accept a positional `nranks` argument
- The script must accept an `--out-root` option
- Set `profiling_activated=True` and `profiling_trace=True` in `EnvironmentOptions`
- Use `sim_folder=f"sim_ranks{num_ranks}"` to organize output by rank count

### 3. Create the Submission Script

Create `profiling/submit_<model>_job.py` based on this template:

```python
"""Submit the <model> profiling job.

Defines the <model>-specific `ProfilingCase`(s) and delegates all submission,
packaging, and upload logic to `profiling_job.run_profiling_job`. See
`profiling_job.py` for the shared machinery, and use this file as a template
for submitting other profiling jobs.
"""

from pathlib import Path

from profiling_job import ProfilingCase, run_profiling_job

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent

CASES = [
    ProfilingCase(
        label="<case_label>",  # Used for directory names and job names
        name="<Human-readable name>",
        description="<Detailed description of what this case tests>",
        physics_problem="<Physics problem being simulated>",
        struphy_model_used="<ModelName>",
        ranks=(1, 2, 4),  # Tuple of MPI rank counts to test
        params_source=(script_dir / "examples" / "<ModelName>" / "<case_name>" / "params_<case>.py"),
        run_script=script_dir / "run_<model>.py",
    ),
    # Add more cases here if needed
]


def main() -> None:
    run_profiling_job(CASES, description="Submit the <model> profiling job.")


if __name__ == "__main__":
    main()
```

**ProfilingCase Parameters:**
- `label`: Short identifier used in directory names and job names (e.g., `"landau_damping_scaling"`)
- `name`: Human-readable name for reports (e.g., `"Landau Damping Scaling Test"`)
- `description`: Detailed description of what the case tests
- `physics_problem`: Brief description of the physics (e.g., `"Electron Landau damping in 1D"`)
- `struphy_model_used`: The Struphy model name (e.g., `"VlasovMaxwell"`)
- `ranks`: Tuple of MPI rank counts to test (e.g., `(1, 2, 4, 8)`)
- `params_source`: Path to the parameter file in `profiling/examples/`
- `run_script`: Path to the execution script (`run_<model>.py`)

### 4. Submit Your Job

```bash
# Activate your virtual environment
source .venv/bin/activate

# Submit the job (default: pitagora cluster, GNU compiler, Fortran)
python profiling/submit_<model>_job.py

# Or with custom options:
python profiling/submit_<model>_job.py \
    --cluster tok \
    --compiler intel \
    --language c
```

**Available Options:**
- `--cluster`: SLURM cluster preset (`pitagora` or `tok`)
- `--compiler`: Compiler family (`GNU`, `intel`, `PGI`, `nvidia`, `LLVM`)
- `--language`: Pyccel language (`fortran` or `c`)
- `--results-root`: Custom output directory (default: auto-generated timestamp-based)

## Running Locally

The same command works on a laptop — there is no separate local mode to select. If
`sbatch` is not on `PATH`, `run_profiling_job` runs the case directly on this machine
instead of submitting it:

```bash
source .venv/bin/activate
python profiling/submit_<model>_job.py
```

What changes, and what does not:

| | Cluster | Local |
| --- | --- | --- |
| Job script | SBATCH pragmas, submitted with `sbatch`, waited on with `squeue` | plain `#!/bin/bash`, run with `bash` |
| MPI launcher | `srun -n <N>` | `mpirun -n <N>` (or `mpiexec`) |
| Environment modules | `module purge` / `source ./setup/modules.sh load` | omitted when there is no module system |
| Rank counts | all of `case.ranks` | only those that fit in `os.cpu_count()`; larger ones are skipped with a message |
| whereami, profiling, packaging, upload | | identical |

The script is still written to `job_profile_<case_label>.sh` in the repo root, so you
can inspect or rerun it by hand. Detection is per capability rather than per machine:
the MPI launcher comes from `detect_launcher()`, the module lines from
`has_module_system()`, so a cluster login node without Lmod, or a workstation with
`mpirun` only, both work.

Because a laptop typically has far fewer cores than a compute node, a case declaring
`ranks=(2, 4, 8, 16, 32, 64)` runs `(2, 4, 8)` on an 8-core machine. Oversubscribing
would just make `mpirun` refuse to start; if no rank count fits, the run raises.
`job_information.scheduler` in the packaged metadata records which path was taken.

## Example: Diocotron Instability

The diocotron instability profiling job is provided as a complete example:

- **Parameter file**: `profiling/examples/ToyGyrokinetic/diocotron_instability/params_diocotron.py`
- **Execution script**: `profiling/run_diocotron.py`
- **Submission script**: `profiling/submit_diocotron_job.py`

Study these files to understand the pattern, then follow the steps above to add your own profiling cases.

## Output Structure

After a profiling run completes, results are organized as:

```
results/profiling/<timestamp>-<commit>/
├── <case_label>/
│   ├── profiling_case_info.json      # Metadata about the case
│   ├── machine_params.json           # `whereami` export for the compute node
│   ├── sim_ranks1/                   # Output for 1 MPI rank
│   │   ├── profiling_data.h5         # Raw profiling data
│   │   ├── run_metadata.json         # Metadata for this run
│   │   └── *.png                     # Processed plots
│   ├── sim_ranks2/                   # Output for 2 MPI ranks
│   ├── sim_ranks4/                   # Output for 4 MPI ranks
│   └── figures/                      # Comparison plots across ranks
└── latest_run_root.txt               # Pointer to most recent run
```

### Packaged metadata (`case_metadata.json`)

`package_profiling_results.py` copies the `.h5` files into
`profiling-results-export/<timestamp>-<commit>-<case>-<language>/` and writes a
`case_metadata.json` next to them. Every value is stored exactly once, in one of five
top-level sections:

| Section | Contents |
| --- | --- |
| `general_information` | Timestamp, user, test case identity/description, model, simulation name and description read from `parameters.py`, source results root |
| `hardware_information` | Cluster name, platform, hostname, uname, `lscpu` output, resolved node hostnames, and the name of the packaged `whereami` export (see below) |
| `software_information` | Struphy commit, pyccel language/compiler family and remaining compiler options, parameter file paths, loaded modules, environment variables, `pip freeze` |
| `job_information` | Scheduler (`slurm` or `local`), job script path and contents, SBATCH pragmas, `SLURM_*` variables |
| `files` | One entry per packaged `.h5` file: source path, rank count, destination file name, and the name of that run's packaged `run_metadata.json` |

Each run also contributes its own `run_metadata.json` (written by `Simulation.run()`
into `sim_ranks<N>/`). It is copied next to the corresponding `.h5` file and renamed to
match it, and both file names are listed in the `files` entry for that run:

```json
{
  "ranks": "4",
  "destination": "diocotron_poisson_scaling-ranks0004-fortran.h5",
  "run_metadata_destination": "diocotron_poisson_scaling-ranks0004-fortran-run_metadata.json"
}
```

So looping over the runs of a case needs nothing but the metadata file:

```python
case_dir = Path("20260724T135741Z-d045c53a-diocotron_poisson_scaling-fortran")
metadata = json.loads((case_dir / "case_metadata.json").read_text())

for entry in metadata["files"]:
    profiling_data = case_dir / entry["destination"]
    run_metadata = json.loads((case_dir / entry["run_metadata_destination"]).read_text())
    print(entry["ranks"], run_metadata["data"]["mpi_ranks"])
```

`run_metadata_destination` is `null` for a run that produced no metadata, so guard the
lookup if you cannot rule that out.

Notes on where things live, to avoid re-adding duplicates:

- `SLURM_*` variables appear only in `job_information.variables`, not in
  `software_information.environment_variables`.
- `LOADEDMODULES` is expanded into `software_information.modules` and is not repeated
  as an environment variable.
- The job script is stored once as `job_information.script`; the generator's
  `custom_commands` list is dropped because it is already contained in that script.
- The raw `profiling_case_info.json` is not embedded; its fields are hoisted into the
  sections above.

### Machine parameters (`machine_params.json`)

CPU/GPU details are not written into `case_metadata.json`. Instead
[`whereami`](https://github.com/max-models/whereami) exports the parameters of the
compute node the job actually runs on.

Because compute nodes have no outbound network access, `run_profiling_job` installs
`whereami` from the **login node**, before `sbatch`:

```bash
curl -fsSL https://raw.githubusercontent.com/max-models/whereami/main/install.sh | bash -s -- "$VIRTUAL_ENV/bin"
```

The venv is on a shared filesystem, so the batch script only has to run:

```bash
whereami --output <case dir>/machine_params.json
```

The packaging step copies that file verbatim next to the `.h5` files and records its
name in `hardware_information.machine_params_file`. A failed install is not fatal: the
job then writes no export, packaging regenerates it with `whereami` from `PATH`, and if
that is unavailable too `machine_params_file` is `null`.

Choosing the SLURM preset at submission time does not depend on `whereami` being
installed: `detect_machine_name()` in `package_profiling_results.py` is a Python port of
the machine-detection table of `whereami` (`MACHINE_NAME` only). `detect_cluster_name`
matches the detected name ("Pitagora (DCGP)", "TOK", ...) against the keys of
`CLUSTER_PRESETS` and falls back to `default_cluster_name` on an unknown machine, e.g.
when submitting from a laptop. Keep the port in sync when `whereami` learns about a new
machine.

## Cluster Configuration

Cluster-specific SLURM settings are defined in `profiling_job.py` in the `CLUSTER_PRESETS` dictionary. To add a new cluster:

1. Add an entry to `CLUSTER_PRESETS` with appropriate SLURM parameters
2. Add the cluster name to the `--cluster` choices in `build_arg_parser()`

## Testing

Before submitting a new profiling job to the cluster, test it locally:

```bash
# Test the execution script directly
python profiling/run_<model>.py 1 --out-root /tmp/test_profiling
```

This will run a single-rank simulation with profiling enabled and save output to `/tmp/test_profiling/sim_ranks1/`.

## Troubleshooting

**Job fails immediately:**
- Check that your parameter file is valid and can be imported
- Verify that all paths in your submission script are correct
- Check SLURM logs in the repo root: `job_profile_<case_label>.<job_id>.{out,err}`

**No profiling data generated:**
- Ensure `profiling_activated=True` and `profiling_trace=True` in `EnvironmentOptions`
- Check that `scope-profiler` is installed in your virtual environment
- Verify that the simulation actually runs (check SLURM output files)

**Import errors:**
- Make sure your parameter file is in `profiling/examples/` and not in `examples/`
- Check that `sys.path.insert(0, str(params_dir))` is present in your execution script

## Contributing

When adding new profiling cases:
1. Follow the naming conventions (`submit_<model>_job.py`, `run_<model>.py`)
2. Use the diocotron example as a template
3. Document any model-specific requirements in comments
4. Test locally before submitting to the cluster
5. Update this README if you add new features or patterns
