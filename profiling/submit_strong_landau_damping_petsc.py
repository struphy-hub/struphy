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
    params_dir = script_dir / "examples" / "VlasovAmpereOneSpecies" / "strong_Landau_damping"

    profiling_case = ProfilingCase(
        label="strong_landau_damping_petsc",
        name="Strong Landau damping, initial Poisson solve: pcg vs. PETSc",
        description=(
            "Strong (nonlinear) Landau damping test case for VlasovAmpereOneSpecies, solving the "
            "one-time initial Poisson problem with either feectools' native preconditioned CG "
            "(--solver pcg) or PETScSolver (KSP=cg, PC=gamg, --solver petsc)."
        ),
        physics_problem="Nonlinear Landau damping in a uniform, collisionless plasma.",
        struphy_model_used="VlasovAmpereOneSpecies",
        params_source=params_dir / "params_strong_Landau_damping.py",
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )
    profiling_case.use_slurm = False

    # Launch one run per (rank count, solver) combination, same rank counts for both solvers so
    # they are directly comparable. Each launch gets its own `--id` (hence its own `sim_<id>`
    # output folder), so the two solvers never collide even at the same rank count.
    for num_tasks in (1, ):
        for solver in ("pcg", "petsc"):
            profiling_case.launch(num_tasks, param_flags=["--solver", solver])

    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
