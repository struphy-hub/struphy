"""Vlasov cuboid strong scaling profiling case: long-time Maxwellian equilibrium in a periodic cube.

This file defines one variant of the Vlasov strong scaling profiling case (the `ProfilingCase`):
a Maxwellian with constant moments in a periodic cube with a uniform magnetic field, which must stay in
equilibrium for long times (binned distributions in position and velocity space are compared with the exact
profiles). For each rank count, `ProfilingCase.launch` builds and submits a SLURM script (using
`clusters.SLURM_PRESETS` by default), or, without a batch system, runs directly on this machine.
`finalize_run` then packages and uploads each run as soon as its own job finishes.
Each generated script runs the simulation itself by invoking `params_vlasov.py` directly (its `__main__`
block is the worker), with the flags of the variant, and then post-processes it with `pproc_vlasov.py`.

The variant (algorithms of the propagators, splitting, loading) is chosen with `--variant`; run this script once
per variant.
"""

import argparse
import sys
from pathlib import Path

# the profiling driver (profiling_job.py, clusters.py, ...) lives in profiling/ at the repo root, three levels up
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "profiling"))

from profiling_job import ProfilingCase  # noqa: E402

# variant -> (description, flags of params_vlasov.py)
VARIANTS = {
    "analytic_rk4": (
        "analytic push_vxb, RK4 push_eta",
        ["--vxb-algo", "analytic", "--butcher", "rk4"],
    ),
    "implicit_rk4": (
        "implicit push_vxb, RK4 push_eta",
        ["--vxb-algo", "implicit", "--butcher", "rk4"],
    ),
    "analytic_forward_euler": (
        "analytic push_vxb, forward Euler push_eta",
        ["--vxb-algo", "analytic", "--butcher", "forward_euler"],
    ),
    "analytic_heun2": (
        "analytic push_vxb, Heun2 push_eta",
        ["--vxb-algo", "analytic", "--butcher", "heun2"],
    ),
    "analytic_heun3": (
        "analytic push_vxb, Heun3 push_eta",
        ["--vxb-algo", "analytic", "--butcher", "heun3"],
    ),
    "analytic_rk4_strang": (
        "analytic push_vxb, RK4 push_eta, Strang splitting",
        ["--vxb-algo", "analytic", "--butcher", "rk4", "--split", "Strang"],
    ),
    "analytic_rk4_sobol": (
        "analytic push_vxb, RK4 push_eta, Sobol loading",
        ["--vxb-algo", "analytic", "--butcher", "rk4", "--loading", "sobol_standard"],
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
        help="Algorithms of the propagators, splitting and loading of the runs.",
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
        label=f"vlasov_cuboid_strong_scaling_{args.variant}",
        name=f"Vlasov Maxwellian equilibrium in a periodic cube strong scaling test, {variant_description}",
        description=(
            "Strong scaling of the Vlasov model (particles in a static, uniform magnetic field) with a Maxwellian of "
            "constant moments in a periodic cube, which must stay in equilibrium for long times; binned distributions "
            f"in position and velocity space are compared with the exact profiles: {variant_description}."
        ),
        physics_problem="Long-time Maxwellian equilibrium of charged particles in a uniform magnetic field.",
        struphy_model_used="Vlasov",
        params_source=script_dir / "params_vlasov.py",
        pproc_source=script_dir / "pproc_vlasov.py",
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
