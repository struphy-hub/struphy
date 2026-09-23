import argparse
from pathlib import Path

import cunumpy as xp

from struphy import Output

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "sim_data"


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
    energy = run.scalars.electric_energy.copy()
    energy.attrs["label"] = "numerical"
    analytical = energy.copy(data=E_exact(energy.t.values, eps=amplitude))  # t is in Struphy units
    analytical.attrs["label"] = "analytical"
    energy.struphy.plot.timeseries(analytical, title="Electric energy").show()

    # full f and delta f in the e1-v1 plane at four times
    for quantity, title in (("f", "full-$f$"), ("delta_f", r"$\delta f$")):
        getattr(run.kinetic_ions.e1_v1_density, quantity).struphy.plot.panels(
            x="e1", y="v1", nrows=1, ncols=4, title=title
        ).show()


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
