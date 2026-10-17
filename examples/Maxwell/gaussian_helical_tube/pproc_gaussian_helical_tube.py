"""Post-process the Maxwell gaussian_helical_tube example: |B| on the poloidal cut eta3 = 0 (through the pulse) at the
first, middle and last snapshot, and the energies. Run as ``python pproc_gaussian_helical_tube.py [sim_folder] [flags]``.
"""

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
from params_gaussian_helical_tube import args, parser, sim  # noqa: E402


def cut(out, step, eta3=0.0, n1=61, n2=121):
    """|B| on the poloidal cut eta3 and the coordinates of the cut in its own plane (pole at the origin)."""
    e1, e2, e3 = np.linspace(1e-3, 1, n1), np.linspace(0, 1, n2), np.array([eta3])
    b = out.evaluate("em_fields/b_field", t=step, eta1=e1, eta2=e2, eta3=e3, representation="2")
    b = np.stack([b.sel(component=c).values.reshape(n1, n2) for c in range(3)])
    # physical B = DF b / det(DF) of the logical 2-form b
    DF = np.asarray(sim.domain.jacobian(e1, e2, e3)).reshape(3, 3, n1, n2)
    detj = np.asarray(sim.domain.jacobian_det(e1, e2, e3)).reshape(n1, n2)
    magnitude = np.linalg.norm(np.einsum("ijkl,jkl->ikl", DF, b), axis=0) / np.abs(detj)
    X = np.array(sim.domain(e1, e2, e3)).reshape(3, n1, n2)
    pole = np.array(sim.domain(np.array([0.0]), np.array([0.0]), e3)).reshape(3, 1, 1)
    d = X - pole
    q1 = d[:, -1, 0] / np.linalg.norm(d[:, -1, 0])
    q2 = d[:, -1, n2 // 4] - (d[:, -1, n2 // 4] @ q1) * q1
    q2 /= np.linalg.norm(q2)
    return np.einsum("i,ijk->jk", q1, d), np.einsum("i,ijk->jk", q2, d), magnitude


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    out = Output(path_out)
    snapshots = out.standard_snapshots()
    times = [float(out.time[n]) for n in snapshots]
    results_dir = os.path.join(out.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4), squeeze=False)
    for ax, n, t in zip(axes[0], snapshots, times):
        u, v, magnitude = cut(out, n)
        ax.pcolormesh(u, v, magnitude, cmap="magma", shading="gouraud")
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"$t = {t:.3f}$")
    fig.suptitle(
        rf"$|B|$ on the poloidal cut $\zeta = 0$, polar splines $p = {args.degree}$, $\mathfrak{{r}} = {args.regularity}$"
    )
    fig.tight_layout()

    scalars = out.evaluate("scalars")
    fig_e, ax = plt.subplots(figsize=(5, 3.5))
    for key in ("electric_energy", "magnetic_energy", "total_energy"):
        ax.plot(scalars["t"], scalars[key], label=key.replace("_", " "))
    ax.set_xlabel("$t$")
    ax.legend()
    fig_e.tight_layout()

    if save_figs:
        fig.savefig(os.path.join(results_dir, "b_snapshots.png"))
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
