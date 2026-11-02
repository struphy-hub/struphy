"""Post-process and plot the circularly polarized Alfvén wave (PerturbativeMHDWithQ).

Run as ``python pproc_alfven_wave.py [path_out] [flags of params_alfven_wave.py] [--show]``. The components B_y, B_z of
the magnetic field perturbation are plotted at the saved times next to the exact wave
A (cos 2 pi (x - t), sin 2 pi (x - t)), together with the energies en_U, en_B and en_tot. Measured at the last saved
time: the maximal error of the magnetic field relative to A, the relative change of its amplitude, and the maximal
relative change of en_tot over the run. They are printed, saved to ``<path_out>/results/metrics.json`` and asserted
(not with ``--test-run``) after the figures are saved.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent

# The flags of the run (parsed from the same command line), for the default output folder.
sys.path.insert(0, str(SCRIPT_DIR))
from params_alfven_wave import AMP, args, parser  # noqa: E402

DEFAULT_OUTPUT = Path(args.out_folder) / args.sim_folder

NUM_EVAL = 128  # evaluation points
ERROR_TOL = 3e-2  # maximal error of B / A at the last saved time (default run: about 1e-2)
AMPLITUDE_RTOL = 1e-3  # relative change of the amplitude (default run: about 2e-6)
ENERGY_RTOL = 1e-8  # relative change of en_tot (default run: about 1e-14)


def magnetic_field(run, n):
    eta = (np.arange(NUM_EVAL) + 0.5) / NUM_EVAL
    kwargs = dict(t=n, eta1=eta, eta2=[0.5], eta3=[0.5])
    by, bz = (run.evaluate("em_fields/b_field", component=c, **kwargs).squeeze().to_numpy() for c in (1, 2))
    return eta, by, bz


def finish(fig, path, save_figs):
    if save_figs:
        fig.savefig(path)
        plt.close(fig)


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    run = Output(path_out)
    results_dir = os.path.join(run.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    steps = run.standard_snapshots()
    fig, axes = plt.subplots(1, len(steps), figsize=(4.5 * len(steps), 3.5), constrained_layout=True, squeeze=False)
    for ax, n in zip(axes[0], steps):
        t = float(run.time[n])
        x, by, bz = magnetic_field(run, n)
        ax.plot(x, by, label="B_y")
        ax.plot(x, bz, label="B_z")
        ax.plot(x, AMP * np.cos(2 * np.pi * (x - t)), "k--", label="exact")
        ax.plot(x, AMP * np.sin(2 * np.pi * (x - t)), "k--")
        ax.set(title=f"t = {t:.3g}", xlabel="x")
    axes[0, 0].legend()
    finish(fig, os.path.join(results_dir, "magnetic_field.png"), save_figs)

    t = np.asarray(run.time)
    energies = run.evaluate("scalars", variables=["en_U", "en_B", "en_tot"])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name in ("en_U", "en_B", "en_tot"):
        ax.plot(t, energies[name].to_numpy(), label=name)
    ax.set(xlabel="t", ylabel="energy")
    ax.legend()
    finish(fig, os.path.join(results_dir, "energies.png"), save_figs)

    en_tot = energies["en_tot"].to_numpy()
    energy_change = float(np.max(np.abs(en_tot - en_tot[0])) / abs(en_tot[0]))
    t_end = float(run.time[-1])
    x, by, bz = magnetic_field(run, -1)
    error = float(
        max(
            np.max(np.abs(by - AMP * np.cos(2 * np.pi * (x - t_end)))),
            np.max(np.abs(bz - AMP * np.sin(2 * np.pi * (x - t_end)))),
        )
        / AMP
    )
    amplitude_change = float(abs(np.sqrt(np.mean(by**2 + bz**2)) / AMP - 1))
    result = {"t": t_end, "b_error": error, "amplitude_change": amplitude_change, "energy_change": energy_change}
    with open(os.path.join(results_dir, "metrics.json"), "w") as file:
        json.dump(result, file, indent=2)
    print(
        f"t = {t_end:.3g}: max error of B / A {error:.3g}, relative change of the amplitude {amplitude_change:.3g}, "
        f"relative change of en_tot {energy_change:.3g}"
    )

    if not save_figs:
        plt.show()

    # not for the few steps of --test-run
    if not args.test_run:
        assert error <= ERROR_TOL, f"max error {error:.3g} of B / A above {ERROR_TOL}."
        assert amplitude_change <= AMPLITUDE_RTOL, f"amplitude change {amplitude_change:.3g} above {AMPLITUDE_RTOL}."
        assert energy_change <= ENERGY_RTOL, f"relative change {energy_change:.3g} of en_tot above {ENERGY_RTOL}."


if __name__ == "__main__":
    # the flags of the run are parsed by params_alfven_wave.py, the rest here
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process and plot a saved simulation run.")
    pproc_parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: the output folder of the run flags, sim_data beside this script)",
    )
    pproc_parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    pproc_args = pproc_parser.parse_args(other_flags)
    main(pproc_args.path_out, save_figs=not pproc_args.show)
