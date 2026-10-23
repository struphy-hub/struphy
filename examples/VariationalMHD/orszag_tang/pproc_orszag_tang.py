"""Post-process and plot the Orszag-Tang vortex.

Run as ``python pproc_orszag_tang.py [sim_folder] [flags of params_orszag_tang.py] [--show]``; the flags
must be those of the run. The figures are written into ``<sim_folder>/results`` (``--show`` shows them instead):
density, pressure and |B| at the first, middle and last saved time, and the energies and conserved quantities
against time. The relative drifts of total energy, mass and entropy and the maximum of div B are printed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent

# The setup of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
import params_orszag_tang as params  # noqa: E402

DEFAULT_OUTPUT = Path(params.args.out_folder) / params.args.sim_folder


def relative_drift(values):
    """Maximum of |v(t) - v(0)| / |v(0)|."""
    values = np.asarray(values)
    return float(np.max(np.abs(values - values[0])) / np.abs(values[0]))


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
    snapshots = list(run.standard_snapshots())
    save = saver(run, save_figs)

    # conservation
    sc = run.scalars
    en_tot = sc["en_tot"]
    print(f"relative drift of en_tot:   {relative_drift(en_tot):.3e}")
    print(f"relative drift of dens_tot: {relative_drift(sc['dens_tot']):.3e}")
    print(f"relative drift of entr_tot: {relative_drift(sc['entr_tot']):.3e}")
    print(f"max |div B|:                {float(np.max(np.abs(sc['tot_div_B']))):.3e}")

    fig, axs = plt.subplots(2, 1, sharex=True)
    for name in ("en_U", "en_mag", "en_thermo"):
        axs[0].plot(sc.t, sc[name], label=name)
    axs[0].set(ylabel="energy")
    axs[0].legend()
    for name in ("en_tot", "dens_tot", "entr_tot"):
        axs[1].plot(sc.t, (sc[name] - sc[name][0]) / sc[name][0], label=name)
    axs[1].set(xlabel="time", ylabel="relative change")
    axs[1].legend()
    fig.tight_layout()
    save({"energies": fig})

    # density, pressure and |B| in the (x, y) plane at the first, middle and last saved time
    rho = run.evaluate("mhd/density", t=snapshots, eta3=0.5, representation="0")
    s = run.evaluate("mhd/entropy", t=snapshots, eta3=0.5, representation="0")
    b = run.evaluate("em_fields/b_field", t=snapshots, eta3=0.5, representation="v")
    absB = np.sqrt((b**2).sum("component"))
    # pressure of the entropy formulation, p = (gamma - 1) rho^gamma exp(s / rho)
    p = (params.gamma - 1.0) * rho**params.gamma * np.exp(s / rho)

    for name, field in (("density", rho), ("pressure", p), ("absB", absB)):
        grid = field.plot(x="X", y="Y", col="t")
        grid.fig.suptitle(name)
        save({name: grid.fig})

    if not save_figs:
        plt.show()


if __name__ == "__main__":
    # the flags of the run are parsed by params_orszag_tang.py, the rest here
    _, other_flags = params.parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Plot a saved simulation run.")
    parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: the output folder of the run flags, beside this script)",
    )
    parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    args = parser.parse_args(other_flags)
    main(args.path_out, save_figs=not args.show)
