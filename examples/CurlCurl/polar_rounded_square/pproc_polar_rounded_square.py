"""Post-process the CurlCurl polar_rounded_square example: L2 error of curl A (computed with the polar curl of the
saved A) and plots of curl A and its error. Run as ``python pproc_polar_rounded_square.py [sim_folder] [flags]``."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "sim_1"

sys.path.insert(0, str(SCRIPT_DIR))
from params_polar_rounded_square import args, exact_b, parser, sim  # noqa: E402


def gauss_points(cells, n):
    x, w = np.polynomial.legendre.leggauss(n)
    a = np.arange(cells)[:, None] / cells
    return (a + (x[None, :] + 1) / (2 * cells)).ravel(), np.tile(w / (2 * cells), cells)


def curl_a(out):
    """Spline function of curl A (the polar curl of the saved 1-form A)."""
    a = out.spline_fields(t=-1)["em_fields"]["e_field"]
    derham = out._spline_derham
    b = derham.create_spline_function("curl_a", "Hdiv")
    b.vector = derham.curl.dot(a.vector)
    return b


def bz(b, e1, e2, e3):
    """Physical B_z = b_3 L_z / det(DF) on the tensor grid e1 x e2 x e3 (single e3)."""
    values = b(e1, e2, e3)[2].reshape(e1.size, e2.size)
    detj = sim.domain.jacobian_det(e1, e2, e3).reshape(e1.size, e2.size)
    return values * sim.domain.params["Lz"] / detj


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    out = Output(path_out)
    b = curl_a(out)
    cells = sim.grid.num_elements
    e1, w1 = gauss_points(cells[0], args.degree + 3)
    e2, w2 = gauss_points(cells[1], args.degree + 3)
    e3 = np.array([0.5])
    x, y, z = (c.reshape(e1.size, e2.size) for c in sim.domain(e1, e2, e3))
    detj = np.abs(sim.domain.jacobian_det(e1, e2, e3)).reshape(e1.size, e2.size) / sim.domain.params["Lz"]
    weights = np.outer(w1, w2) * detj
    error = np.sqrt(np.sum(weights * (bz(b, e1, e2, e3) - exact_b(x, y, z)) ** 2))
    print(f"L2 error of curl A: {error:.3e}")

    results_dir = os.path.join(out.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "errors.json"), "w") as f:
        json.dump(dict(l2_error=float(error), cells=args.cells, degree=args.degree, regularity=args.regularity), f)

    p1, p2 = np.linspace(1e-3, 1, 81), np.linspace(0, 1, 241)
    x, y, z = (c.reshape(p1.size, p2.size) for c in sim.domain(p1, p2, e3))
    values = bz(b, p1, p2, e3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, field, title in ((axes[0], values, r"$\mathrm{curl}\,A_h$"), (axes[1], values - exact_b(x, y, z), "error")):
        mesh = ax.pcolormesh(x, y, field, shading="gouraud", cmap="RdBu_r")
        plt.colorbar(mesh, ax=ax, shrink=0.8)
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.tight_layout()
    if save_figs:
        fig.savefig(os.path.join(results_dir, "curl_a.png"))
        plt.close(fig)
    else:
        plt.show()
    return error


if __name__ == "__main__":
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process a saved simulation run.")
    pproc_parser.add_argument("path_out", nargs="?", default=DEFAULT_OUTPUT, help="Simulation output folder.")
    pproc_parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    pproc_args = pproc_parser.parse_args(other_flags)
    main(pproc_args.path_out, save_figs=not pproc_args.show)
