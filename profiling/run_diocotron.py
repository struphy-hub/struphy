from argparse import ArgumentParser
from pathlib import Path
import sys

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

    out_folders = default_out_root / f"n{num_ranks}"
    env = EnvironmentOptions(
        out_folders=str(out_folders),
        sim_folder=f"sim_{num_ranks}",
        profiling_activated=True,
        profiling_trace=True,
    )

    from params_diocotron import sim

    sim.env = env
    sim.meta["output folder"] = env.path_out
    sim._setup_folders()

    ProfileManager.finalize()
    ProfileManager.setup(
        profiling_activated=env.profiling_activated,
        time_trace=env.profiling_trace,
        use_likwid=False,
        file_path=str(Path(env.path_out) / "profiling_data.h5"),
    )

    sim.run(one_time_step=True)


if __name__ == "__main__":
    main()
