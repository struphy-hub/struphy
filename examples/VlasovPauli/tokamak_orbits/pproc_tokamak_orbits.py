"""Post-process and plot the guiding-center orbits of classical Pauli particles in an ad hoc tokamak.

Run as ``python pproc_tokamak_orbits.py [sim_folder] [flags of params_tokamak_orbits.py] [--show]``; the flags
must be those of the run. The figures are written into ``<sim_folder>/results`` (``--show`` shows them instead):
the orbits of the saved markers in the (R, Z) plane, their relative energy error and their mu' / mu against time.
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
import params_tokamak_orbits as params  # noqa: E402

DEFAULT_OUTPUT = Path(params.args.out_folder) / params.args.sim_folder


def orbit_diagnostics(orbits, equil):
    """(R, Z), relative energy error and mu' / mu of the saved markers, arrays of shape (t, marker); NaN after a
    marker is lost. The energy is v^2/2 + mu |B0| with the analytic |B0| of the equilibrium."""
    x, y, z = (np.asarray(orbits[c].values) for c in ("x", "y", "z"))
    v = np.stack([np.asarray(orbits[c].values) for c in ("v1", "v2", "v3")])
    mu = np.asarray(orbits["mu"].values)

    R = np.sqrt(x**2 + y**2)
    b = np.asarray(equil.b_xyz(x.ravel(), y.ravel(), z.ravel())).reshape((3,) + x.shape)
    absB = np.sqrt(np.sum(b**2, axis=0))
    v_par = np.sum(v * b, axis=0) / absB
    v2 = np.sum(v**2, axis=0)

    energy = 0.5 * v2 + mu * absB
    en_err = np.abs(energy / energy[0] - 1.0)
    mu_prime = 0.5 * (v2 - v_par**2) / absB
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_ratio = mu_prime / mu
    return R, z, en_err, mu_ratio


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
    save = saver(run, save_figs)

    orbits = run.evaluate("kinetic_ions/orbits")
    # markers that left the plasma are saved as -1 in all columns (the full-f weights are positive): NaN from then on
    orbits = orbits.where(orbits["weight"] > 0.0)
    R, Z, en_err, mu_ratio = orbit_diagnostics(orbits, params.equil)
    t = orbits.t.values

    print(f"max relative energy error {np.nanmax(en_err):.2e}, max mu'/mu {np.nanmax(mu_ratio):.2e}")

    fig, ax = plt.subplots()
    ax.plot(R, Z, lw=0.7)
    ax.set(xlabel="R", ylabel="Z", aspect="equal", title="Guiding-center orbits of classical Pauli particles")
    save({"orbits_RZ": fig})

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].semilogy(t, np.nanmax(en_err, axis=1))
    axs[0].set(ylabel="max relative energy error")
    axs[1].semilogy(t, np.nanmax(mu_ratio, axis=1))
    axs[1].set(xlabel="time", ylabel=r"max $\mu' / \mu$")
    save({"energy_mu_prime": fig})

    if not save_figs:
        plt.show()


if __name__ == "__main__":
    # the flags of the run are parsed by params_tokamak_orbits.py, the rest here
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
