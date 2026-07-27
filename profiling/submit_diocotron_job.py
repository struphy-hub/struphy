"""Submit the diocotron profiling job.

Defines the diocotron-specific `ProfilingCase`(s) and delegates all submission,
packaging, and upload logic to `profiling_job.run_profiling_job`. See
`profiling_job.py` for the shared machinery, and use this file as a template
for submitting other profiling jobs.
"""

from pathlib import Path

from profiling_job import ProfilingCase, run_profiling_job

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent

# Static SLURM settings per cluster. `job_name`, `ntasks_per_node`, and
# `custom_commands` are filled in per profiling case at submission time.
CLUSTER_PRESETS: dict[str, dict] = {
    "pitagora": {
        "nodes": 1,
        "cpus_per_task": 1,
        # "mem_per_cpu": "8GB",
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
    params_source=(script_dir / "examples" / "ToyGyrokinetic" / "diocotron_instability" / "params_diocotron.py"),
    run_script=script_dir / "run_diocotron.py",
    cluster_presets=CLUSTER_PRESETS,
)


def main() -> None:
    run_profiling_job(profiling_case)


if __name__ == "__main__":
    main()
