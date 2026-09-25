import argparse
from pathlib import Path

import cunumpy as xp
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


def E_exact(t, eps=0.001):
    """Analytical electric energy of weak Landau damping, t in normalized units."""
    r = 0.3677
    omega_r = 1.4156
    omega_i = -0.1533
    phi = 0.5362
    return (4 * eps * r * xp.exp(omega_i * t) * xp.cos(omega_r * t - phi)) ** 2 * xp.pi


def main(path_out=DEFAULT_OUTPUT, amplitude=0.001):
    run = Output(path_out)

    # electric field energy against the analytical damping
    energy = run.scalars.electric_energy
    analytical = E_exact(energy.t.values, eps=amplitude)  # t is in Struphy units
    fig, ax = plt.subplots()
    ax.plot(energy.t, energy, label="numerical")
    ax.plot(energy.t, analytical, "--", label="analytical")
    ax.set(xlabel="time", yscale="log", title="Electric energy")
    ax.legend()
    plt.show()

    # full f and delta f in the e1-v1 plane at four times
    for quantity, title in (("f", "full-$f$"), ("delta_f", r"$\delta f$")):
        plot_panels(getattr(run.kinetic_ions.e1_v1_density, quantity), x="e1", y="v1", n_panels=4, ncols=4, title=title)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a saved simulation run.")
    parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: sim_data beside this script)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.001,
        help="Initial perturbation amplitude for the analytical curve",
    )
    args = parser.parse_args()
    main(args.path_out, amplitude=args.amplitude)
