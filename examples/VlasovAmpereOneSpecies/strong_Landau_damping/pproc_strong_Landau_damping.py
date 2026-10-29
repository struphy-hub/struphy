"""Post-process and plot the strong Landau damping example (VlasovAmpereOneSpecies).

Run as ``python pproc_strong_Landau_damping.py [path_out] [flags of params_strong_Landau_damping.py] [--show]``. The figures are written into
``<path_out>/results`` (``--show`` shows them instead), among them the fields along x at the first, middle and
last saved time.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output
from struphy.post_processing.slices import plot_line_profiles

SCRIPT_DIR = Path(__file__).resolve().parent

# The flags of the run (parsed from the same command line), for the default output folder.
sys.path.insert(0, str(SCRIPT_DIR))
import params_strong_Landau_damping as params  # noqa: E402

DEFAULT_OUTPUT = Path(params.args.out_folder) / params.args.sim_folder


def plot_panels(data, x, y, n_panels, ncols, title=None):
    """Snapshots of a binned distribution at evenly spaced saved times, one panel each."""
    times = np.unique(np.linspace(0, data.sizes["t"] - 1, n_panels).round().astype(int))
    grid = data.isel(t=times).plot(x=x, y=y, col="t", col_wrap=min(ncols, len(times)))
    if title is not None:
        grid.fig.suptitle(title)
    return grid


# fields shown along x at the first, middle and last saved time: (key, label, name, vector)
FIELDS = [("e_field", "E", "em_fields/e_field", True), ("phi", "phi", "em_fields/phi", False)]


def field_profiles(run, snapshots):
    """The fields along x at the first, middle and last saved time: the standard plots of runs on a 1D grid."""
    return {
        f"{key}_profiles": plot_line_profiles(
            run, run.domain, name, label, snapshots, run.field_basis(name) if vector else None
        )
        for key, label, name, vector in FIELDS
    }


def saver(run, save_figs):
    """A function saving figures as ``<name>.png`` into ``<path_out>/results`` and closing them (``save_figs``)."""
    results_dir = Path(run.path_out) / "results"
    results_dir.mkdir(exist_ok=True)

    def save(figs):
        if save_figs:
            for name, fig in figs.items():
                fig.savefig(results_dir / f"{name}.png")
                plt.close(fig)

    return save


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    run = Output(path_out)
    snapshots = run.standard_snapshots()
    # VTK files of the fields at the first, middle and last saved time (fields on a 1D grid, no 3D view)
    run.pproc(create_vtk=True, vtk_steps=snapshots)
    save = saver(run, save_figs)

    # electric field energy
    lines = run.evaluate("scalars", variables="en_E")["en_E"].plot(yscale="log")
    plt.title("Electric energy")
    save({"en_E": lines[0].figure})

    # full f in the e1-v1 plane
    data = run.evaluate("kinetic_ions/f", dataset="e1_v1_density/f")
    grid = plot_panels(data, x="eta1", y="v1", n_panels=12, ncols=4, title="full-$f$")
    save({"f_e1_v1": grid.fig})

    # the fields along x at the first, middle and last saved time
    save(field_profiles(run, snapshots))

    if not save_figs:
        plt.show()


if __name__ == "__main__":
    # the flags of the run are parsed by params_strong_Landau_damping.py, the rest here
    _, other_flags = params.parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Plot a saved simulation run.")
    parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: the output folder of the run flags, sim_data beside this script)",
    )
    parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    args = parser.parse_args(other_flags)
    main(args.path_out, save_figs=not args.show)
