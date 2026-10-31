"""Post-process and plot the nonlinear simple sound wave (PerturbativeEulerWithQ).

Run as ``python pproc_simple_wave.py [path_out] [flags of params_simple_wave.py] [--show]``. The velocity is plotted at
the saved times next to the exact simple wave (method of characteristics) and the profile of the linear model
(propagation with c0), together with the energies en_U, en_internal and en_tot and the mass perturbation. Measured:
the maximal error of u relative to the amplitude at the last saved time, the relative change of en_tot and the change
of the mass. They are printed, saved to ``<path_out>/results/metrics.json`` and asserted (not with ``--test-run``)
after the figures are saved.
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
from params_simple_wave import AMP, C0, GAMMA, args, parser  # noqa: E402

DEFAULT_OUTPUT = Path(args.out_folder) / args.sim_folder

NUM_EVAL = 256  # evaluation points
ERROR_TOL = 2e-2  # maximal error of u / A at the last saved time (default run: about 1e-2)
ENERGY_RTOL = 1e-3  # relative change of en_tot (splitting error, default run: about 2e-5)


def exact_velocity(x, t):
    """u(x, t) = A sin(2 pi xi), x = xi + (c0 + (gamma + 1) / 2 A sin(2 pi xi)) t (Newton iteration for xi)."""
    xi = x - C0 * t
    for _ in range(50):
        u0 = AMP * np.sin(2 * np.pi * xi)
        f = xi + (C0 + (GAMMA + 1) / 2 * u0) * t - x
        xi -= f / (1 + (GAMMA + 1) / 2 * AMP * 2 * np.pi * np.cos(2 * np.pi * xi) * t)
    return AMP * np.sin(2 * np.pi * xi)


def velocity(run, n):
    eta = (np.arange(NUM_EVAL) + 0.5) / NUM_EVAL
    u = run.evaluate("mhd/velocity", component=0, t=n, eta1=eta, eta2=[0.5], eta3=[0.5]).squeeze()
    return eta, u.to_numpy()


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
        x, u = velocity(run, n)
        ax.plot(x, u, label="PerturbativeEulerWithQ")
        ax.plot(x, exact_velocity(x, t), "k--", label="exact")
        ax.plot(x, AMP * np.sin(2 * np.pi * (x - C0 * t)), "k:", label="linear")
        ax.set(title=f"t = {t:.3g}", xlabel="x", ylabel="u")
    axes[0, 0].legend()
    finish(fig, os.path.join(results_dir, "velocity.png"), save_figs)

    t = np.asarray(run.time)
    energies = run.evaluate("scalars", variables=["en_U", "en_internal", "en_tot"])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name in ("en_U", "en_internal", "en_tot"):
        ax.plot(t, energies[name].to_numpy(), label=name)
    ax.set(xlabel="t", ylabel="energy")
    ax.legend()
    finish(fig, os.path.join(results_dir, "energies.png"), save_figs)

    en_tot = energies["en_tot"].to_numpy()
    energy_change = float(np.max(np.abs(en_tot - en_tot[0])) / abs(en_tot[0]))
    t_end = float(run.time[-1])
    x, u = velocity(run, -1)
    error = float(np.max(np.abs(u - exact_velocity(x, t_end))) / AMP)
    error_linear = float(np.max(np.abs(AMP * np.sin(2 * np.pi * (x - C0 * t_end)) - exact_velocity(x, t_end))) / AMP)
    result = {"t": t_end, "u_error": error, "u_error_linear_model": error_linear, "energy_change": energy_change}
    with open(os.path.join(results_dir, "metrics.json"), "w") as file:
        json.dump(result, file, indent=2)
    print(
        f"t = {t_end:.3g}: max error of u / A {error:.3g} (linear propagation: {error_linear:.3g}), "
        f"relative change of en_tot {energy_change:.3g}"
    )

    if not save_figs:
        plt.show()

    # not for the few steps of --test-run
    if not args.test_run:
        assert error <= ERROR_TOL, f"max error {error:.3g} of u / A above {ERROR_TOL}."
        assert energy_change <= ENERGY_RTOL, f"relative change {energy_change:.3g} of en_tot above {ENERGY_RTOL}."


if __name__ == "__main__":
    # the flags of the run are parsed by params_simple_wave.py, the rest here
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
