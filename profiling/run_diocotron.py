import sys
from argparse import ArgumentParser
from pathlib import Path

from scope_profiler import ProfileManager

from struphy import EnvironmentOptions

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
params_dir = repo_root / "examples" / "ToyGyrokinetic" / "diocotron_instability"
default_out_root = repo_root / "profiling" / "results" / "diocotron_poisson_scaling"

sys.path.insert(0, str(params_dir))


def main() -> None:
    parser = ArgumentParser(description="Run the diocotron profiling case.")
    parser.add_argument("nranks", type=int, help="Number of MPI ranks used for the run")
    num_ranks = parser.parse_args().nranks

    env = EnvironmentOptions(
        out_folders=str(default_out_root),
        sim_folder=f"sim_ranks{num_ranks}",
        profiling_activated=True,
        profiling_trace=True,
    )

    from params_diocotron import sim

    sim.env = env
    sim.meta["output folder"] = env.path_out
    sim._setup_folders()

    sim.run(one_time_step=True)


if __name__ == "__main__":
    main()
