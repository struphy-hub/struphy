import logging
import sys

logger = logging.getLogger("struphy")


def struphy_profile(dirs, all=False, n_lines=20, prefix=None, savefig=None):
    """
    Show the profiling data of finished Struphy runs.

    Reads ``profiling_data.h5`` in each output folder, written by ``sim.run(profiling_activated=True)``,
    through :class:`struphy.post_processing.profiling.Profile` (also available as ``Output.profile``).

    Parameters
    ----------
    dirs : list[str]
        Simulation output folders (absolute or relative to the current directory).

    all : bool
        Show all regions instead of the ``n_lines`` most expensive ones.

    n_lines : int
        Number of regions to show (and plot), sorted by total time.

    prefix : str, optional
        Keep only regions whose name starts with this, e.g. ``"prop:"`` or ``"kernel:"``.

    savefig : str, optional
        Save a bar plot of the total time of the shown regions under this name (relative to the current directory).
    """

    from pathlib import Path

    from struphy.post_processing.profiling import Profile

    profiles = []
    for d in dirs:
        path = Path(d).resolve() / "profiling_data.h5"
        if not path.is_file():
            print(
                f"No profiling data in {d}: {path} does not exist. It is written by sim.run(profiling_activated=True).",
                file=sys.stderr,
            )
            sys.exit(1)
        profiles += [Profile(path)]

    top = None if all else n_lines

    for profile in profiles:
        print(profile.table(prefix=prefix, top=top))
        print("")

    # total time of each region side by side
    total_times = profiles[0].compare(*profiles[1:], prefix=prefix).isel(region=slice(0, top))
    if len(profiles) > 1:
        runs = [f"{run} [s]" for run in total_times.run.values]
        width = max((len(name) for name in total_times.region.values), default=6)
        col = max(len(run) for run in runs)
        print("Total time per rank of each region:")
        print("")
        print(f"{'Region':<{width}}" + "".join(f"  {run:>{col}}" for run in runs))
        print(f"{'-' * width}" + f"  {'-' * col}" * len(runs))
        for name in total_times.region.values:
            values = total_times.sel(region=name).values
            print(f"{name:<{width}}" + "".join(f"  {value:>{col}.3f}" for value in values))
        print("")

    if savefig is not None:
        import numpy as np
        from matplotlib import pyplot as plt

        regions = total_times.region.values
        n_runs = total_times.sizes["run"]
        height = 0.8 / n_runs
        fig, ax = plt.subplots(figsize=(10, 0.4 * len(regions) * n_runs + 1.5))
        for n, profile in enumerate(profiles):
            ax.barh(
                np.arange(len(regions)) + n * height,
                total_times.isel(run=n).values,
                height=height,
                label=f"{total_times.run.values[n]} ({profile.num_ranks} rank(s))",
            )
        ax.set_yticks(np.arange(len(regions)) + (n_runs - 1) * height / 2)
        ax.set_yticklabels(regions)
        ax.invert_yaxis()
        ax.set_xlabel("total time per rank [s]")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(savefig)
        plt.close(fig)
        print(f"Saved figure to {Path(savefig).resolve()}")
