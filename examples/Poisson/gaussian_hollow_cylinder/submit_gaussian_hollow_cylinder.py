"""Poisson strong scaling profiling case on hollow cylinder.

This file defines the Poisson strong scaling profiling case (the `ProfilingCase`)
and submits it: for each rank count, `ProfilingCase.launch` builds and submits a
SLURM script (using `clusters.SLURM_PRESETS` by default), or, without a batch
system, runs directly on this machine. `finalize_run` then packages and uploads
each run as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_gaussian_hollow_cylinder.py`
directly (its `__main__` block is the worker), and then post-processes it with `pproc_gaussian_hollow_cylinder.py`.

The variant (preconditioner) is chosen with `--variant`; run this script once per variant.
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/ at the repo root, three levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "profiling"))

from profiling_job import ProfilingCase  # noqa: E402

# variant -> (label, name, description, flags of the params file); the labels identify the
# cases in the profiling-data repo and are kept from the former one-case-per-script layout
VARIANTS = {
    "none": (
        "poisson_hollow_cylinder_strong_scaling",
        "Poisson on hollow cylinder strong scaling test, no preconditioner",
        "Strong scaling of the Poisson model with manufactured solution on a hollow cylinder (HollowCylinder, a1=1e-2, a2=1, Lz=4), solved with unpreconditioned CG.",
        [],
    ),
    "multigrid": (
        "poisson_hollow_cylinder_strong_scaling_multigrid",
        "Poisson on hollow cylinder strong scaling test, multigrid preconditioner",
        "Strong scaling of the Poisson model with manufactured solution on a hollow cylinder (HollowCylinder, a1=1e-2, a2=1, Lz=4), solved with CG preconditioned by geometric multigrid.",
        ["--precond", "MultiGrid"],
    ),
    "stiffness": (
        "poisson_hollow_cylinder_strong_scaling_stiffness",
        "Poisson on hollow cylinder strong scaling test, stiffness preconditioner",
        "Strong scaling of the Poisson model with manufactured solution on a hollow cylinder (HollowCylinder, a1=1e-2, a2=1, Lz=4), solved with CG preconditioned by the Kronecker approximation of the inverse Laplacian (StiffnessPreconditioner).",
        ["--precond", "StiffnessPreconditioner"],
    ),
}


def main() -> None:

    # Parse arguments, do not remove --upload
    parser = argparse.ArgumentParser(
        description=("Submit profiling jobs to a SLURM cluster and package the results for upload."),
    )
    parser.add_argument(
        "--variant",
        default="none",
        choices=list(VARIANTS),
        help="Preconditioner of the runs.",
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
    label, name, description, param_flags = VARIANTS[args.variant]

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent

    profiling_case = ProfilingCase(
        label=label,
        name=name,
        description=description,
        physics_problem="Occurs in many plasma applications.",
        struphy_model_used="Poisson",
        params_source=script_dir / "params_gaussian_hollow_cylinder.py",
        pproc_source=script_dir / "pproc_gaussian_hollow_cylinder.py",
        language=args.language,
        compiler=args.compiler,
        upload=args.upload,
    )

    # Launch one run per rank count
    for num_tasks in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        profiling_case.launch(num_tasks, param_flags=param_flags)

    # Package and push each run as its own job finishes.
    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
