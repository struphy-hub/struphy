"""Post-process the Maxwell gaussian_perturbed_disk example: B_z = curl A on the cross-section at the first, middle
and last snapshot, and the energies. Run as ``python pproc_gaussian_perturbed_disk.py [sim_folder] [variant flags]``."""

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


def bz_cross_section(out, step, n1=81, n2=181):
    """Physical B_z = b_3 L_z / det(DF) on the cross-section eta3 = 0.5 (eta1 > 0)."""
    e1, e2, e3 = np.linspace(1e-3, 1, n1), np.linspace(0, 1, n2), np.array([0.5])
    b = out.evaluate("em_fields/b_field", t=step, eta1=e1, eta2=e2, eta3=e3, representation="2")
    b3 = b.sel(component=2).values.reshape(n1, n2)
    detj = sim.domain.jacobian_det(e1, e2, e3).reshape(n1, n2)
    x, y, _ = (c.reshape(n1, n2) for c in sim.domain(e1, e2, e3))
    return x, y, b3 * sim.domain.params["Lz"] / detj


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    out = Output(path_out)
    snapshots = out.standard_snapshots()
    times = [float(out.time[n]) for n in snapshots]
    results_dir = os.path.join(out.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4), squeeze=False)
    for ax, n, t in zip(axes[0], snapshots, times):
        x, y, bz = bz_cross_section(out, n)
        limit = np.max(np.abs(bz)) or 1.0
        ax.pcolormesh(x, y, bz, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="gouraud")
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"$t = {t:.3f}$")
    fig.suptitle(
        rf"$B_z = \mathrm{{curl}}\,A$, polar splines $p = {args.degree}$, $\mathfrak{{r}} = {args.regularity}$, "
        f"{args.enrichment}"
    )
    fig.tight_layout()

    scalars = out.evaluate("scalars")
    fig_e, ax = plt.subplots(figsize=(5, 3.5))
    for key in ("electric_energy", "magnetic_energy", "total_energy"):
        ax.plot(scalars["t"], scalars[key], label=key.replace("_", " "))
    ax.set_xlabel("$t$")
    ax.legend()
    fig_e.tight_layout()
    total = np.asarray(scalars["total_energy"])
    print(f"max relative variation of the total energy: {np.max(np.abs(total - total[0])) / total[0]:.2e}")

    if save_figs:
        fig.savefig(os.path.join(results_dir, "bz_snapshots.png"))
        fig_e.savefig(os.path.join(results_dir, "energies.png"))
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
