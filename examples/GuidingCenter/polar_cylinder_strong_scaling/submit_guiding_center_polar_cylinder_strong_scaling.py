"""GuidingCenter polar cylinder strong scaling profiling case: long-time gyro-Maxwellian equilibrium in a screw pinch.

This file defines one variant of the GuidingCenter strong scaling profiling case in a straight cylinder without
hole (IGAPolarCylinder, C1 polar splines) with a screw pinch equilibrium (the `ProfilingCase`): a gyro-Maxwellian
whose moments depend only on the radius is an exact equilibrium and must stay so for long times, while markers
move through the axis. For each rank count, `ProfilingCase.launch` builds and submits a SLURM script (using
`clusters.SLURM_PRESETS` by default), or, without a batch system, runs directly on this machine.
`finalize_run` then packages and uploads each run as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_guiding_center.py` directly (its `__main__`
block is the worker), with the flags of the variant, and then post-processes it with `pproc_guiding_center.py`.

The variant (algorithm of the guiding-center propagators) is chosen with `--variant`; run this script once per
variant.
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/ at the repo root, three levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "profiling"))

from profiling_job import ProfilingCase  # noqa: E402

# variant -> (description, flags of params_guiding_center.py)
VARIANTS = {
    "dg_1st": (
        "discrete gradient, 1st order (fixed point)",
        ["--algo", "discrete_gradient_1st_order"],
    ),
    "dg_1st_newton": (
        "discrete gradient, 1st order (Newton)",
        ["--algo", "discrete_gradient_1st_order_newton"],
    ),
    "dg_2nd": (
        "discrete gradient, 2nd order",
        ["--algo", "discrete_gradient_2nd_order"],
    ),
    "explicit_rk4": (
        "explicit RK4",
        ["--algo", "explicit", "--butcher", "rk4"],
    ),
    "explicit_heun3": (
        "explicit Heun3",
        ["--algo", "explicit", "--butcher", "heun3"],
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
        help="Algorithm of the guiding-center propagators of the runs.",
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
        label=f"guiding_center_polar_cylinder_strong_scaling_{args.variant}",
        name=f"GuidingCenter screw pinch equilibrium strong scaling test, {variant_description}",
        description=(
            "Strong scaling of the GuidingCenter model in a straight cylinder without hole (IGAPolarCylinder, C1 polar "
            "splines) with a screw pinch equilibrium: a gyro-Maxwellian with radial moments is an exact equilibrium and "
            "must stay so for long times while markers move through the axis; binned distributions in position and "
            f"velocity space are compared with their initial profiles: {variant_description}."
        ),
        physics_problem="Long-time guiding-center equilibrium in a screw pinch.",
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
