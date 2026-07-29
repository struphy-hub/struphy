"""Diocotron profiling case.

This file defines the diocotron profiling case (the `ProfilingCase`) and submits
it: for each rank count, `ProfilingCase.launch` builds and submits a SLURM script
(using `clusters.SLURM_PRESETS` by default), or, without a batch system,
runs directly on this machine. Once every rank count has finished running, the
comparison plot across rank counts is built, and the case is packaged/uploaded.
Each generated script runs the simulation itself by invoking `params_diocotron.py`
directly (its `__main__` block is the worker).

Use this file as a template for defining other profiling cases.
"""

import argparse
from pathlib import Path

from profiling_job import ProfilingCase

def main() -> None:

    # Parse arguments, do not remove --upload
    parser = argparse.ArgumentParser(
        description=("Submit profiling jobs to a SLURM cluster and package the results for upload."),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the packaged profiling results to the profiling-data repo.",
    )
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "examples" / "Poisson" / "cube_strong_scaling"

    profiling_case = ProfilingCase(
        label="poisson_cube_strong_scaling",
        name="Poisson on cube strong scaling test",
        description="Strong scaling of the Poisson model with manufactured solution on 3D cube.",
        physics_problem="Occurs in many plasma applications.",
        struphy_model_used="Poisson",
        params_source=params_dir / "params_poisson.py",
        language="fortran",
        compiler="GNU",
    )

    # Launch one run per rank count
    for num_tasks in (1, 2, ):#4, 8, 16, 32, 64, 128, 256):
        profiling_case.launch(num_tasks)

    # Wait for all jobs to finish, and then build the comparison plot and package the case.
    profiling_case.finalize_run(upload=args.upload)


if __name__ == "__main__":
    main()
