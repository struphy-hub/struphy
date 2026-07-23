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

CASES = [
    ProfilingCase(
        label="diocotron_poisson_scaling",
        name="Diocotron Poisson scaling",
        description="Scaling test running the diocotron profiling setup with multiple MPI ranks.",
        physics_problem="Diocotron instability in a non-neutral plasma.",
        struphy_model_used="ToyDrift",
        ranks=(1, 2, 4),  # , 8),
        params_source=(repo_root / "examples" / "ToyGyrokinetic" / "diocotron_instability" / "params_diocotron.py"),
        run_script=script_dir / "run_diocotron.py",
    ),
]


def main() -> None:
    run_profiling_job(CASES, description="Submit the diocotron profiling job.")


if __name__ == "__main__":
    main()
