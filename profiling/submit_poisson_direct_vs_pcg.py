"""Poisson 8x8x8 DirectSolver vs. PCG profiling case.

This submits one profiling run per solver using
``params_poisson_solver.py``. Each run is single-rank by default because the goal is to
compare repeated solve cost and DirectSolver factorization amortization, not MPI scaling.
"""

import argparse
from pathlib import Path

from profiling_job import ProfilingCase


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit Poisson 8x8x8 DirectSolver vs. PCG profiling jobs.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the packaged profiling results to the profiling-data repo.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Number of repeated Poisson solves per run.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        default=1,
        help="MPI ranks per solver run.",
    )
    args = parser.parse_args()

    resolution = (8, 8, 8)

    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "examples" / "Poisson" / "direct_vs_pcg"

    profiling_case = ProfilingCase(
        label="poisson_direct_vs_pcg_8x8x8",
        name="Poisson 8x8x8: DirectSolver vs. PCG",
        description="Single-rank repeated Poisson solves on an 8x8x8 grid, comparing cached DirectSolver against PCG.",
        physics_problem="Small manufactured Poisson field solve on a 3D cuboid.",
        struphy_model_used="Poisson",
        params_source=params_dir / "params_poisson_solver.py",
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )

    for solver in ("pcg", "direct"):
        profiling_case.launch(
            args.ranks,
            param_flags=[
                "--solver",
                solver,
                "--num-elements",
                *map(str, resolution),
                "--repeats",
                str(args.repeats),
            ],
        )

    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
