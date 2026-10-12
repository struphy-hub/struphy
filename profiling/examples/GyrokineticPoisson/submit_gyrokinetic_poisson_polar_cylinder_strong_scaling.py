"""GyrokineticPoisson polar cylinder strong scaling profiling case: polar splines versus tensor-product splines on a cylinder with a hole.

This file defines one variant of the GyrokineticPoisson strong scaling profiling case on a cylinder (radius 1, length 2*pi) in a screw pinch equilibrium (the
`ProfilingCase`) and submits it: for each rank count, `ProfilingCase.launch` builds and submits a
SLURM script (using `clusters.SLURM_PRESETS` by default), or, without a batch system, runs directly
on this machine. `finalize_run` then packages and uploads each run as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_gyrokinetic_poisson.py` directly (its `__main__`
block is the worker), with the flags of the variant.

The variant (spline space and preconditioner) is chosen with `--variant`; run this script once per variant.
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/, two levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from profiling_job import ProfilingCase  # noqa: E402

# variant -> (description, flags of params_gyrokinetic_poisson.py)
VARIANTS = {
    "tensor_none": (
        "tensor-product splines, no preconditioner",
        ["--splines", "tensor", "--precond", "None"],
    ),
    "tensor_stiffness": (
        "tensor-product splines, stiffness preconditioner",
        ["--splines", "tensor", "--precond", "StiffnessPreconditioner"],
    ),
    "tensor_multigrid": (
        "tensor-product splines, multigrid preconditioner",
        ["--splines", "tensor", "--precond", "MultiGrid"],
    ),
    "polar_none": (
        "polar splines, no preconditioner",
        ["--splines", "polar", "--precond", "None"],
    ),
    "polar_tensor_mass": (
        "polar splines, tensor-product mass matrix preconditioner",
        ["--splines", "polar", "--precond", "MassMatrixPreconditioner"],
    ),
    "polar_mass": (
        "polar splines, polar mass matrix preconditioner (Schur)",
        ["--splines", "polar", "--precond", "PolarMassMatrixPreconditioner"],
    ),
    "polar_mass_ilu": (
        "polar splines, polar mass matrix preconditioner (ILU)",
        ["--splines", "polar", "--precond", "PolarMassMatrixPreconditioner", "--poloidal-solver", "ilu"],
    ),
    "polar_stiffness": (
        "polar splines, polar stiffness preconditioner (Schur)",
        ["--splines", "polar", "--precond", "StiffnessPreconditioner"],
    ),
    "polar_stiffness_ilu": (
        "polar splines, polar stiffness preconditioner (ILU)",
        ["--splines", "polar", "--precond", "StiffnessPreconditioner", "--poloidal-solver", "ilu"],
    ),
}


def main() -> None:

    # Parse arguments, do not remove --upload
    parser = argparse.ArgumentParser(
        description=("Submit profiling jobs to a SLURM cluster and package the results for upload."),
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=list(VARIANTS),
        help="Spline space and preconditioner of the runs.",
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
    variant_description, param_flags = VARIANTS[args.variant]

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "polar_cylinder_strong_scaling"

    profiling_case = ProfilingCase(
        label=f"gyrokinetic_poisson_polar_cylinder_strong_scaling_{args.variant}",
        name=f"GyrokineticPoisson on a cylinder strong scaling test, {variant_description}",
        description=(
            "Strong scaling of the GyrokineticPoisson model with a manufactured solution whose gradient is largest on the axis, "
            "comparing C1 polar splines (IGAPolarCylinder) with tensor-product splines on a cylinder with a hole "
            f"(HollowCylinder, a1=1e-2): {variant_description}."
        ),
        physics_problem="Field equation of gyrokinetic simulations with adiabatic electrons.",
        struphy_model_used="GyrokineticPoisson",
        params_source=params_dir / "params_gyrokinetic_poisson.py",
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
