"""Poisson strong scaling profiling case.

This file defines the Poisson strong scaling profiling case (the `ProfilingCase`)
and submits it: for each rank count, `ProfilingCase.launch` builds and submits a
SLURM script (using `clusters.SLURM_PRESETS` by default), or, without a batch
system, runs directly on this machine. `finalize_run` then packages and uploads
each run as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_LinearMHDDriftkineticCC.py`
directly (its `__main__` block is the worker).
"""

import argparse
from pathlib import Path

from profiling_job import ProfilingCase
from clusters import SLURM_PRESETS

cpu_preset = SLURM_PRESETS.get("pitagora_dcgp")
gpu_preset = SLURM_PRESETS.get("pitagora_booster")

slurm_presets = {
    "numpy": cpu_preset,
    "cupy": gpu_preset,
}


def main() -> None:

    # Parse arguments, do not remove --upload
    parser = argparse.ArgumentParser(
        description=(
            "Submit profiling jobs to a SLURM cluster and package the results for upload."
        ),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the packaged profiling results to the profiling-data repo.",
    )
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = (
        script_dir / "examples" / "LinearMHDDriftkineticCC" / "cube_strong_scaling"
    )

    profiling_case = ProfilingCase(
        label="linearmhd_numpy_vs_cupy",
        name="Linear MHD on cube",
        description="Linear MHD model with manufactured solution on 3D cube.",
        physics_problem="Occurs in many plasma applications.",
        struphy_model_used="LinearMHDDriftkineticCC",
        params_source=params_dir / "params_LinearMHDDriftkineticCC.py",
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )

    # Launch one run per rank count
    for num_tasks in (1,):
        for backend in ("numpy", "cupy"):
            profiling_case.launch(
                num_tasks,
                param_flags=["--backend", backend],
                slurm_preset=slurm_presets[backend],
            )

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
