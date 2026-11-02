"""Post-process and plot the standing sound wave in the Colella domain (LinearEulerWithQ).

Run as ``python pproc_standing_wave_colella.py [path_out] [flags of params_standing_wave_colella.py] [--show]``. The
perturbation of q = sqrt(p) is plotted in physical coordinates at the first, middle and last saved time, next to the exact standing
wave, together with the energies en_U, en_thermal and en_tot and their exact laws en_tot sin^2(omega t),
en_tot cos^2(omega t). Measured: the relative L2 error of q at the last saved time and the maximal relative
change of en_tot. They are printed, saved to ``<path_out>/results/metrics.json`` and asserted (not with
``--test-run``) after the figures are saved.
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
from params_standing_wave_colella import args, parser  # noqa: E402

DEFAULT_OUTPUT = Path(args.out_folder) / args.sim_folder

GAMMA = 5 / 3
NUM_EVAL = 64  # evaluation points per direction
ERROR_TOL = 0.01  # relative L2 error of q at the last saved time (default run: about 3e-3)
ENERGY_RTOL = 1e-6  # relative change of en_tot (default solver tolerances: about 5e-8)


def omega_and_amplitude(run):
    """Frequency of the standing wave and amplitude of the initial perturbation of q."""
    params = run.equil.params
    p0 = params["beta"] * (params["B0x"] ** 2 + params["B0y"] ** 2 + params["B0z"] ** 2) / 2
    c_s = np.sqrt(GAMMA * p0 / params["n0"])
    return 2 * np.sqrt(2) * np.pi * c_s, 1e-2


def q_field(run, n):
    """Physical coordinates and q (scalar, representation "0") at the saved step n."""
    eta = (np.arange(NUM_EVAL) + 0.5) / NUM_EVAL
    p = run.evaluate("mhd/sqrt_p", t=n, eta1=eta, eta2=eta, eta3=[0.5], representation="0").squeeze()
    return p.X.to_numpy(), p.Y.to_numpy(), p.to_numpy()


def exact_q(run, x, y, t):
    omega, amp = omega_and_amplitude(run)
    return amp * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.cos(omega * t)


def plot_snapshots(run, results_dir, save_figs):
    steps = run.standard_snapshots()
    fig, axes = plt.subplots(2, len(steps), figsize=(4 * len(steps), 7), constrained_layout=True, squeeze=False)
    amp = omega_and_amplitude(run)[1]
    for col, n in enumerate(steps):
        t = float(run.time[n])
        x, y, p = q_field(run, n)
        for row, (values, label) in enumerate(((p, "numerical"), (exact_q(run, x, y, t), "exact"))):
            ax = axes[row, col]
            mesh = ax.pcolormesh(x, y, values, cmap="RdBu_r", vmin=-amp, vmax=amp, shading="gouraud")
            ax.set_aspect("equal")
            ax.set_title(f"{label}, t = {t:.3g}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
    fig.colorbar(mesh, ax=axes, label="perturbation of q")
    finish(fig, os.path.join(results_dir, "q_snapshots.png"), save_figs)


def plot_energies(run, results_dir, save_figs):
    omega = omega_and_amplitude(run)[0]
    t = np.asarray(run.time)
    energies = run.evaluate("scalars", variables=["en_U", "en_thermal", "en_tot"])
    en_tot0 = float(energies["en_tot"][0])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name in ("en_U", "en_thermal", "en_tot"):
        ax.plot(t, energies[name].to_numpy(), label=name)
    ax.plot(t, en_tot0 * np.sin(omega * t) ** 2, "k:", label=r"exact $E\,\sin^2(\omega t)$")
    ax.plot(t, en_tot0 * np.cos(omega * t) ** 2, "k--", label=r"exact $E\,\cos^2(\omega t)$")
    ax.set_xlabel("t")
    ax.set_ylabel("energy")
    ax.legend()
    finish(fig, os.path.join(results_dir, "energies.png"), save_figs)
    return float(np.max(np.abs(energies["en_tot"].to_numpy() - en_tot0)) / en_tot0)


def finish(fig, path, save_figs):
    if save_figs:
        fig.savefig(path)
        plt.close(fig)


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    run = Output(path_out)
    results_dir = os.path.join(run.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    plot_snapshots(run, results_dir, save_figs)
    energy_change = plot_energies(run, results_dir, save_figs)

    t = float(run.time[-1])
    x, y, p = q_field(run, -1)
    amp = omega_and_amplitude(run)[1]
    q_error = float(np.sqrt(np.mean((p - exact_q(run, x, y, t)) ** 2)) / amp)
    result = {"t": t, "q_error": q_error, "energy_change": energy_change}
    with open(os.path.join(results_dir, "metrics.json"), "w") as file:
        json.dump(result, file, indent=2)
    print(f"t = {t:.3g}: relative L2 error of q {q_error:.3g}, relative change of en_tot {energy_change:.3g}")

    if not save_figs:
        plt.show()

    # not for the few steps of --test-run
    if not args.test_run:
        assert q_error <= ERROR_TOL, f"relative L2 error {q_error:.3g} of q above {ERROR_TOL}."
        assert energy_change <= ENERGY_RTOL, f"relative change {energy_change:.3g} of en_tot above {ENERGY_RTOL:.0e}."


if __name__ == "__main__":
    # the flags of the run are parsed by params_standing_wave_colella.py, the rest here
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
