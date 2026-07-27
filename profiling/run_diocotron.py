"""Diocotron profiling case.

This file defines the diocotron profiling case (the `ProfilingCase` and its cluster
presets) and submits it: `profiling_job.run_profiling_job` builds one SLURM script
per rank count in `profiling_case.ranks`, submits all of them, waits for all to
finish, and then packages/uploads the results. Each generated script runs the
simulation itself by invoking `params_diocotron.py` directly (its `__main__` block
is the worker).

Use this file as a template for defining other profiling cases.
"""

from pathlib import Path

from profiling_job import ProfilingCase, run_profiling_job

script_dir = Path(__file__).resolve().parent
params_dir = script_dir / "examples" / "ToyGyrokinetic" / "diocotron_instability"

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
    cluster_presets=CLUSTER_PRESETS,
)


def main() -> None:
    run_profiling_job(profiling_case)


if __name__ == "__main__":
    main()
