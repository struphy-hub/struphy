"""Diocotron profiling case.

This file defines the diocotron profiling case (the `ProfilingCase`) and submits
it: for each rank count, `ProfilingCase.launch` builds and submits a SLURM script
(using `clusters.SLURM_PRESETS` by default), or, without a batch system,
runs directly on this machine. `finalize_run` then packages and uploads each run
as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_diocotron.py`
directly (its `__main__` block is the worker).

Use this file as a template for defining other profiling cases.
"""

import argparse
from pathlib import Path

from profiling_job import ProfilingCase
from utils import _get_profiling_args

def main() -> None:

    args = _get_profiling_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "examples" / "ToyGyrokinetic" / "diocotron_instability"

    profiling_case = ProfilingCase(
        label="diocotron_instability",
        name="Diocotron instability",
        description="Scaling test running the diocotron profiling setup with multiple MPI ranks.",
        physics_problem="Diocotron instability in a non-neutral plasma.",
        struphy_model_used="ToyDrift",
        params_source=params_dir / "params_diocotron.py",
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )


    # Launch one run per rank count
    for num_tasks in (2, 4):  # , 8, 16, 32, 64, 128, 256):
        profiling_case.launch(num_tasks)

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
