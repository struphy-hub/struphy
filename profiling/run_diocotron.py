"""Diocotron profiling case.

This file defines everything there is to know about the diocotron profiling case
(the `ProfilingCase`, its cluster presets, and the MPI worker that runs one rank
count) and is the single entry point for it:

- Running it directly (``python run_diocotron.py``) submits the profiling job:
  `profiling_job.run_profiling_job` builds one SLURM script per rank count in
  `profiling_case.ranks`, submits all of them, waits for all to finish, and then
  packages/uploads the results.
- Each generated script instead invokes this same file with ``--worker``, which runs
  the diocotron simulation for the one rank count it was launched with.

Use this file as a template for defining other profiling cases.
"""

import argparse
import logging
import sys
from pathlib import Path

from mpi4py import MPI
from profiling_job import ProfilingCase, run_profiling_job

from struphy import EnvironmentOptions, set_logging_level

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
params_dir = script_dir / "examples" / "ToyGyrokinetic" / "diocotron_instability"

sys.path.insert(0, str(params_dir))
from params_diocotron import sim

# Static SLURM settings per cluster. `job_name` and `ntasks_per_node` are filled in
# per rank count at submission time.
CLUSTER_PRESETS: dict[str, dict] = {
    "pitagora": {
        "nodes": 1,
        "cpus_per_task": 1,
        "mem": "480GB",
        "partition": "dcgp_fua_dbg",
        "account": "FUSIO_HLST_7",
        "output": "./%x.%j.out",
        "error": "./%x.%j.err",
        "mail_type": "none",
        "time": "00:15:00",
    },
    "tok": {
        "nodes": 1,
        "cpus_per_task": 1,
        "mem_per_cpu": "1GB",
        "partition": "s.tok",
        "qos": "tok.debug",
        "chdir": "./",
        "output": "./%x.%j.out",
        "error": "./%x.%j.err",
        "mail_type": "none",
        "time": "00:15:00",
    },
}

profiling_case = ProfilingCase(
    label="diocotron_instability",
    name="Diocotron instability",
    description="Scaling test running the diocotron profiling setup with multiple MPI ranks.",
    physics_problem="Diocotron instability in a non-neutral plasma.",
    struphy_model_used="ToyDrift",
    ranks=(2, 4, 8, 16, 32, 64),
    params_source=params_dir / "params_diocotron.py",
    run_script=Path(__file__).resolve(),
    cluster_presets=CLUSTER_PRESETS,
)


def run_worker(out_root: Path) -> None:
    """Run the diocotron simulation for one MPI rank count.

    Invoked (via `srun`/`mpirun`) by the scripts that `run_profiling_job` generates,
    one per rank count in `profiling_case.ranks`.
    """
    set_logging_level(logging.INFO)

    comm = MPI.COMM_WORLD
    num_ranks = comm.Get_size()

    env = EnvironmentOptions(
        out_folders=str(out_root.expanduser().resolve()),
        sim_folder=f"sim_ranks{num_ranks}",
        profiling_activated=True,
        profiling_trace=True,
    )

    sim.env = env
    sim.run(one_time_step=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diocotron profiling case.")
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal: run the simulation for one rank count instead of submitting the job.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root for this rank count's run. Required with --worker.",
    )
    args = parser.parse_args()

    if args.worker:
        if args.out_root is None:
            parser.error("--out-root is required with --worker")
        run_worker(args.out_root)
        return

    run_profiling_job(profiling_case)


if __name__ == "__main__":
    main()
