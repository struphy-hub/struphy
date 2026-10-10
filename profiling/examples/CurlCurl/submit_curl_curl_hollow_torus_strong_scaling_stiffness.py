"""Curl-curl strong scaling profiling case on hollow torus, stiffness preconditioner.

This file defines the curl-curl strong scaling profiling case (the `ProfilingCase`)
and submits it: for each rank count, `ProfilingCase.launch` builds and submits a
SLURM script (using `clusters.SLURM_PRESETS` by default), or, without a batch
system, runs directly on this machine. `finalize_run` then packages and uploads
each run as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_curl_curl.py`
directly (its `__main__` block is the worker).
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/, two levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from profiling_job import ProfilingCase  # noqa: E402


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
    parser.add_argument(
        "--language",
        default="fortran",
        choices=["fortran", "c"],
        help="Pyccel language to compile the Struphy kernels with.",
    )
    parser.add_argument(
        "--compiler",
        default="GNU",
        help='Pyccel compiler family ("GNU", "intel", "PGI", "nvidia" or "LLVM").',
    )
    args = parser.parse_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "hollow_torus_strong_scaling_stiffness"

    profiling_case = ProfilingCase(
        label="curl_curl_hollow_torus_strong_scaling_stiffness",
        name="Curl-curl on hollow torus strong scaling test, stiffness preconditioner",
        description="Strong scaling of the CurlCurl model with manufactured solution on a full hollow torus (HollowTorus, a1=1e-2, a2=1, R0=4), solved with CG preconditioned by the Kronecker approximation of the inverse curl-curl operator with gradient correction (StiffnessPreconditioner).",
        physics_problem="Occurs in implicit time stepping of Maxwell's equations and in magnetostatics.",
        struphy_model_used="CurlCurl",
        params_source=params_dir / "params_curl_curl.py",
        language=args.language,
        compiler=args.compiler,
        upload=args.upload,
    )

    # Launch one run per rank count
    for num_tasks in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        profiling_case.launch(num_tasks)

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
