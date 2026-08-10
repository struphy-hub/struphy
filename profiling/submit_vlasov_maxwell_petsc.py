from pathlib import Path

from profiling_job import ProfilingCase

from utils import _get_profiling_args


def main() -> None:

    args = _get_profiling_args()

    # Paths relative to this script's location, so it can be run from anywhere.
    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "examples" / "VlasovMaxwellOneSpecies" / "weibel_instability"

    profiling_case = ProfilingCase(
        label="weibel_instability_petsc",
        name="Weibel instability, initial Poisson solve: pcg vs. PETSc",
        description=(
            "Weibel instability test case for VlasovMaxwellOneSpecies, solving the one-time "
            "initial Poisson problem with either feectools' native preconditioned CG "
            "(--solver pcg) or PETScSolver (KSP=cg, PC=gamg, --solver petsc)."
        ),
        physics_problem="Weibel instability driven by an anisotropic velocity distribution.",
        struphy_model_used="VlasovMaxwellOneSpecies",
        params_source=params_dir / "params_weibel_instability.py",
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )
    profiling_case.use_slurm = False

    # Launch one run per (rank count, solver) combination, same rank counts for both solvers so
    # they are directly comparable. Each launch gets its own `--id` (hence its own `sim_<id>`
    # output folder), so the two solvers never collide even at the same rank count.
    for num_tasks in (1, 2, 4):
        for solver in ("pcg", "petsc"):
            profiling_case.launch(num_tasks, param_flags=["--solver", solver])

    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
