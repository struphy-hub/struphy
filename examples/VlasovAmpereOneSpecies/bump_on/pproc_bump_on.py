import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "sim_data"


def plot_panels(data, x, y, n_panels, ncols, title=None):
    """Snapshots of a binned distribution at evenly spaced saved times, one panel each."""
    times = np.unique(np.linspace(0, data.sizes["t"] - 1, n_panels).round().astype(int))
    grid = data.isel(t=times).plot(x=x, y=y, col="t", col_wrap=min(ncols, len(times)))
    if title is not None:
        grid.fig.suptitle(title)
    return grid


def main(path_out=DEFAULT_OUTPUT):
    run = Output(path_out)

    # initial velocity distribution
    initial = run.evaluate("kinetic_ions/f", dataset="v1_density/f").isel(t=0)
    ax = initial.plot()[0].axes
    ax.set(
        xlabel="velocity $v$",
        ylabel="distribution $f(v)$",
        title="Initial velocity distribution",
    )
    plt.show()

    # electric field energy
    run.evaluate("scalars", variables="electric_energy")["electric_energy"].plot(yscale="log")
    plt.title("Electric energy")
    plt.show()

    # full f in the e1-v1 plane
    data = run.evaluate("kinetic_ions/f", dataset="e1_v1_density/f")
    plot_panels(data, x="e1", y="v1", n_panels=12, ncols=4, title="full-$f$")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a saved simulation run.")
    parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: sim_data beside this script)",
    )
    args = parser.parse_args()
    main(args.path_out)
