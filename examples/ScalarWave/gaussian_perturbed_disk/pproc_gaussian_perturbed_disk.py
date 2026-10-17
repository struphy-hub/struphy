"""Post-process the ScalarWave gaussian_perturbed_disk example: phi on the cross-section at the first, middle and last
snapshot, and the kinetic energy. Run as ``python pproc_gaussian_perturbed_disk.py [sim_folder] [variant flags]``."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "sim_1"

sys.path.insert(0, str(SCRIPT_DIR))
from params_gaussian_perturbed_disk import args, parser, sim  # noqa: E402


def cross_section(out, name, step, n1=81, n2=181):
    """Values of a scalar field on the cross-section eta3 = 0.5 and the physical coordinates."""
    e1, e2, e3 = np.linspace(0, 1, n1), np.linspace(0, 1, n2), np.array([0.5])
    values = out.evaluate(name, t=step, eta1=e1, eta2=e2, eta3=e3).values.reshape(n1, n2)
    x, y, _ = (c.reshape(n1, n2) for c in sim.domain(e1, e2, e3))
    return x, y, values


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    out = Output(path_out)
    snapshots = out.standard_snapshots()
    times = [float(out.time[n]) for n in snapshots]
    results_dir = os.path.join(out.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4), squeeze=False)
    for ax, n, t in zip(axes[0], snapshots, times):
        x, y, phi = cross_section(out, "em_fields/phi", n)
        limit = np.max(np.abs(phi)) or 1.0
        ax.pcolormesh(x, y, phi, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="gouraud")
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"$t = {t:.3f}$")
    fig.suptitle(rf"$\phi$, polar splines $p = {args.degree}$, $\mathfrak{{r}} = {args.regularity}$")
    fig.tight_layout()

    scalars = out.evaluate("scalars")
    fig_e, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(scalars["t"], scalars["kinetic_energy"])
    ax.set_xlabel("$t$")
    ax.set_ylabel("kinetic energy")
    fig_e.tight_layout()

    if save_figs:
        fig.savefig(os.path.join(results_dir, "phi_snapshots.png"))
        fig_e.savefig(os.path.join(results_dir, "kinetic_energy.png"))
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process a saved simulation run.")
    pproc_parser.add_argument("path_out", nargs="?", default=DEFAULT_OUTPUT, help="Simulation output folder.")
    pproc_parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    pproc_args = pproc_parser.parse_args(other_flags)
    main(pproc_args.path_out, save_figs=not pproc_args.show)
