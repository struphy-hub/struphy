import argparse
from pathlib import Path

from profiling_job import ProfilingCase


def main() -> None:

    parser = argparse.ArgumentParser(
        description=("Submit Poisson weak scaling profiling jobs and package the results."),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the packaged profiling results to the profiling-data repo.",
    )
    parser.add_argument(
        "--cells-per-rank",
        type=int,
        default=64,
        help="Number of cells assigned to each rank in the weak-scaling direction.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=(1, 2, 4),
        help="MPI rank counts to submit (default: 1 2 4).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    params_dir = script_dir / "examples" / "Poisson" / "cube_weak_scaling"

    profiling_case = ProfilingCase(
        label="poisson_cube_weak_scaling",
        name="Poisson on cuboid weak scaling test",
        description=(
            "Weak scaling of the Poisson model with a manufactured solution on a "
            "3D cuboid. The grid grows in x with the MPI rank count."
        ),
        physics_problem="Occurs in many plasma applications.",
        struphy_model_used="Poisson",
        params_source=params_dir / "params_poisson.py",
        language="fortran",
        compiler="GNU",
        upload=args.upload,
    )

    param_flags = ["--cells-per-rank", str(args.cells_per_rank)]

    for num_tasks in args.ranks:
        profiling_case.launch(num_tasks, param_flags=param_flags)

    profiling_case.finalize_run()


if __name__ == "__main__":
    main()
