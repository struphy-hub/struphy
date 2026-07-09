import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

from scope_profiler import ProfileManager

from struphy import EnvironmentOptions, set_logging_level

set_logging_level(logging.INFO)

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
params_dir = repo_root / "examples" / "ToyGyrokinetic" / "diocotron_instability"
default_out_root = repo_root / "results" / "profiling" / "manual" / "diocotron_poisson_scaling"

sys.path.insert(0, str(params_dir))
from params_diocotron import sim

def main() -> None:
    parser = ArgumentParser(description="Run the diocotron profiling case.")
    parser.add_argument("nranks", type=int, help="Number of MPI ranks used for the run")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=default_out_root,
        help=(
            "Output root for this testcase. Each rank writes under "
            "OUT_ROOT/sim_ranks<N> (default: results/profiling/manual/diocotron_poisson_scaling)."
        ),
    )
    args = parser.parse_args()
    num_ranks = args.nranks

    env = EnvironmentOptions(
        out_folders=str(args.out_root.expanduser().resolve()),
        sim_folder=f"sim_ranks{num_ranks}",
        profiling_activated=True,
        profiling_trace=True,
    )

    sim.env = env
    sim._setup_folders()

    # print(f"Running diocotron profiling case with {num_ranks} MPI ranks...")
    # print("Environment options:", sim.env)

    sim.run(one_time_step=True)


if __name__ == "__main__":
    main()
