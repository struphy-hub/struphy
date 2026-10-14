"""GuidingCenter polar torus strong scaling profiling case: long-time canonical Maxwellian equilibrium in a tokamak.

This file defines one variant of the GuidingCenter strong scaling profiling case in a full torus without hole
(IGAPolarTorus, C1 polar splines) with an ad hoc tokamak equilibrium (the `ProfilingCase`): a canonical Maxwellian
(moments as functions of the canonical toroidal momentum) is an exact equilibrium and must stay so for long times,
while markers drift through the axis; a gyro-Maxwellian with the same profiles in the flux label of the position is
not an equilibrium and serves as a control. For each rank count, `ProfilingCase.launch` builds and submits a SLURM
script (using `clusters.SLURM_PRESETS` by default), or, without a batch system, runs directly on this machine.
`finalize_run` then packages and uploads each run as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_guiding_center.py` directly (its `__main__`
block is the worker), with the flags of the variant, and then post-processes it with `pproc_guiding_center.py`.

The variant (initial condition, algorithm of the guiding-center propagators) is chosen with `--variant`; run this
script once per variant.
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/ at the repo root, three levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "profiling"))

from profiling_job import ProfilingCase  # noqa: E402

# variant -> (description, flags of params_guiding_center.py)
VARIANTS = {
    "canonical_dg_1st": (
        "canonical Maxwellian, discrete gradient, 1st order (fixed point)",
        ["--init", "canonical", "--algo", "discrete_gradient_1st_order"],
    ),
    "canonical_dg_1st_newton": (
        "canonical Maxwellian, discrete gradient, 1st order (Newton)",
        ["--init", "canonical", "--algo", "discrete_gradient_1st_order_newton"],
    ),
    "canonical_dg_2nd": (
        "canonical Maxwellian, discrete gradient, 2nd order",
        ["--init", "canonical", "--algo", "discrete_gradient_2nd_order"],
    ),
    "canonical_explicit_rk4": (
        "canonical Maxwellian, explicit RK4",
        ["--init", "canonical", "--algo", "explicit", "--butcher", "rk4"],
    ),
    "canonical_explicit_heun3": (
        "canonical Maxwellian, explicit Heun3",
        ["--init", "canonical", "--algo", "explicit", "--butcher", "heun3"],
    ),
    "gyro_dg_1st": (
        "gyro-Maxwellian (control, not an equilibrium), discrete gradient, 1st order (fixed point)",
        ["--init", "gyro", "--algo", "discrete_gradient_1st_order"],
    ),
    "gyro_explicit_rk4": (
        "gyro-Maxwellian (control, not an equilibrium), explicit RK4",
        ["--init", "gyro", "--algo", "explicit", "--butcher", "rk4"],
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
        help="Initial condition and algorithm of the guiding-center propagators of the runs.",
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

    profiling_case = ProfilingCase(
        label=f"guiding_center_polar_torus_strong_scaling_{args.variant}",
        name=f"GuidingCenter ad hoc tokamak equilibrium strong scaling test, {variant_description}",
        description=(
            "Strong scaling of the GuidingCenter model in a full torus without hole (IGAPolarTorus, C1 polar splines) "
            "with an ad hoc tokamak equilibrium: a canonical Maxwellian is an exact equilibrium and must stay so for "
            "long times while markers drift through the axis (a gyro-Maxwellian serves as a control); binned "
            f"distributions in position and velocity space are compared with their initial profiles: {variant_description}."
        ),
        physics_problem="Long-time guiding-center equilibrium in a tokamak.",
        struphy_model_used="GuidingCenter",
        params_source=script_dir / "params_guiding_center.py",
        pproc_source=script_dir / "pproc_guiding_center.py",
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
