"""Post-process the Poisson polar_rounded_square example: L2 error of phi and plots of phi and of the error.

Run as ``python pproc_polar_rounded_square.py [sim_folder] [variant flags of params_polar_rounded_square.py]``.
Errors and figures are written into ``<sim_folder>/results``.
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
DEFAULT_OUTPUT = SCRIPT_DIR / "sim_1"

sys.path.insert(0, str(SCRIPT_DIR))
from params_polar_rounded_square import args, exact_solution, parser, sim  # noqa: E402


def gauss_points(cells, n):
    """Gauss-Legendre points and weights of n points in each of `cells` cells of [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    a = np.arange(cells)[:, None] / cells
    return (a + (x[None, :] + 1) / (2 * cells)).ravel(), np.tile(w / (2 * cells), cells)


def l2_error(out, domain, n_quad):
    """Physical L2 error of phi on the cross-section (the problem does not depend on eta3)."""
    cells = sim.grid.num_elements
    e1, w1 = gauss_points(cells[0], n_quad)
    e2, w2 = gauss_points(cells[1], n_quad)
    e3 = np.array([0.5])
    phi = out.evaluate("em_fields/phi", t=-1, eta1=e1, eta2=e2, eta3=e3).values.reshape(e1.size, e2.size)
    x, y, z = (c.reshape(e1.size, e2.size) for c in domain(e1, e2, e3))
    detj = np.abs(domain.jacobian_det(e1, e2, e3)).reshape(e1.size, e2.size) / domain.params["Lz"]
    weights = np.outer(w1, w2) * detj
    error = phi - exact_solution(x, y, z)
    norm = np.sqrt(np.sum(weights * exact_solution(x, y, z) ** 2))
    return np.sqrt(np.sum(weights * error**2)), norm


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    out = Output(path_out)
    domain = sim.domain
    error, norm = l2_error(out, domain, sim.derham_opts.degree[0] + 3)
    print(f"L2 error of phi: {error:.3e} (relative {error / norm:.3e})")

    results_dir = os.path.join(out.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "errors.json"), "w") as f:
        json.dump(
            dict(
                l2_error=float(error),
                relative_l2_error=float(error / norm),
                cells=args.cells,
                degree=args.degree,
                regularity=args.regularity,
            ),
            f,
        )

    # phi and the error on the cross-section
    e1, e2, e3 = np.linspace(0, 1, 81), np.linspace(0, 1, 241), np.array([0.5])
    phi = out.evaluate("em_fields/phi", t=-1, eta1=e1, eta2=e2, eta3=e3).values.reshape(e1.size, e2.size)
    x, y, z = (c.reshape(e1.size, e2.size) for c in domain(e1, e2, e3))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, values, title in ((axes[0], phi, r"$\phi_h$"), (axes[1], phi - exact_solution(x, y, z), "error")):
        mesh = ax.pcolormesh(x, y, values, shading="gouraud", cmap="RdBu_r" if title == "error" else "viridis")
        plt.colorbar(mesh, ax=ax, shrink=0.8)
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.suptitle(f"Poisson, homotopy polar splines p = {args.degree}, r = {args.regularity}, n = {args.cells}")
    fig.tight_layout()
    if save_figs:
        fig.savefig(os.path.join(results_dir, "phi.png"))
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
