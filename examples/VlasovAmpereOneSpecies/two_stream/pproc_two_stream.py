"""Post-process and plot the two stream example (VlasovAmpereOneSpecies).

Run as ``python pproc_two_stream.py [path_out] [flags of params_two_stream.py] [--show]``. The figures are written into
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
import params_two_stream as params  # noqa: E402

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

    # every scalar time series as CSV: post_processing/scalars.csv
    print(f"scalars written to {run.save_scalars()}")

    # electric field growth against the analytical growth rate gamma of the field amplitude (in Struphy units, from
    # the kinetic dispersion relation for k = 0.2): the electric energy grows as exp(2 gamma t); the initial density
    # perturbation (without velocity perturbation) puts about half of its amplitude into the growing mode
    # (cold beams: cosh(gamma t)), i.e. a quarter of the initial energy
    gamma = 0.2845
    energy = run.evaluate("scalars", variables="electric_energy")["electric_energy"]
    analytical = energy.isel(t=0) / 4 * np.exp(2 * gamma * energy.t)
    fig, ax = plt.subplots()
    ax.plot(energy.t, energy, label="numerical")
    ax.plot(energy.t, analytical, "--", label=rf"analytical, $\exp(2 \gamma t)$, $\gamma = {gamma}$")
    ax.set(xlabel="time", yscale="log", title="Electric energy")
    ax.legend()
    save({"electric_energy": fig})

    # phase space evolution
    data = run.evaluate("kinetic_ions/f", dataset="e1_v1_density/f")
    grid = plot_panels(data, x="eta1", y="v1", n_panels=12, ncols=4)
    save({"f_e1_v1": grid.fig})

    # the fields along x at the first, middle and last saved time
    save(field_profiles(run, snapshots))

    if not save_figs:
        plt.show()


if __name__ == "__main__":
    # the flags of the run are parsed by params_two_stream.py, the rest here
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
