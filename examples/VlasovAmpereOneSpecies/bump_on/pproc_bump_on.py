import argparse
from pathlib import Path

from matplotlib import pyplot as plt

from struphy import Output

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "sim_data"


def main(path_out=DEFAULT_OUTPUT):
    run = Output(path_out)

    # initial velocity distribution
    initial = run.evaluate("kinetic_ions/v1_density/f").isel(t=0)
    ax = initial.plot()[0].axes
    ax.set(
        xlabel="velocity $v$",
        ylabel="distribution $f(v)$",
        title="Initial velocity distribution",
    )
    plt.show()

    # electric field energy
    run.scalars.electric_energy.struphy.plot.timeseries(title="Electric energy").show()

    # full f in the e1-v1 plane
    run.kinetic_ions.e1_v1_density.f.struphy.plot.panels(x="e1", y="v1", nrows=3, ncols=4, title="full-$f$").show()


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
