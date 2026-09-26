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

    # every scalar time series as CSV: post_processing/scalars.csv
    print(f"scalars written to {run.save_scalars()}")

    # electric field growth against the analytical rate (0.2845 in units of m/c)
    energy = run.evaluate("scalars", variables="electric_energy")["electric_energy"]
    analytical = 10 ** (0.2845 * energy.t - 5.3)  # t is in Struphy units
    fig, ax = plt.subplots()
    ax.plot(energy.t, energy, label="numerical")
    ax.plot(energy.t, analytical, "--", label="analytical")
    ax.set(xlabel="time", yscale="log", title="Electric energy")
    ax.legend()
    plt.show()

    # phase space evolution
    data = run.evaluate("kinetic_ions/f", dataset="e1_v1_density/f")
    plot_panels(data, x="e1", y="v1", n_panels=12, ncols=4)
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
