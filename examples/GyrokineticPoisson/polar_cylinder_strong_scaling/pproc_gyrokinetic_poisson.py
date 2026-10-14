"""Post-process and plot the GyrokineticPoisson polar cylinder strong scaling case.

Run as ``python pproc_gyrokinetic_poisson.py [sim_folder] [variant flags of params_gyrokinetic_poisson.py]``.
The variant flags must be those of the run; the exact solution and the run parameters are taken from
``params_gyrokinetic_poisson.py``, which reads them. Figures and errors are written into ``<sim_folder>/results``,
where the profiling packaging picks them up and uploads them with the run
(``submit_gyrokinetic_poisson_polar_cylinder_strong_scaling.py`` calls this script after each simulation).
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

# The exact solution, the variant flags and the simulation of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
from params_gyrokinetic_poisson import args, distance_to_axis, exact_solution, parser, rhs_fun, sim  # noqa: E402


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    out = Output(path_out).pproc(create_vtk=True)

    def plot_slices(num, exact, name):
        """Numerical solution, exact solution and error on the cross-section at eta3 = 0.5 and the
        half plane at eta2 = 0.5, as functions of (x, y) and (x, z)."""
        fig = plt.figure(figsize=(12, 12))
        idx3 = (slice(None), slice(None), num.shape[2] // 2)
        idx2 = (slice(None), num.shape[1] // 2, slice(None))
        for col, (idx, h, v, labels) in enumerate(
            [(idx3, X, Y, ("x", "y")), (idx2, X, Z, ("x", "z"))],
        ):
            for row, (data, what) in enumerate(
                [(num[idx], "from struphy"), (exact[idx], "exact"), (np.abs(num[idx] - exact[idx]), "error")]
            ):
                plt.subplot(3, 2, 2 * row + col + 1)
                plt.pcolormesh(h[idx], v[idx], data, shading="gouraud")
                plt.colorbar()
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
                plt.title(f"{name} {what}")
        fig.tight_layout()
        return fig

    # Raw FEEC fields are evaluated directly from the saved spline coefficients at the
    # cell centres of the simulation grid.
    rhs = out.evaluate("em_fields/source", t=0).values

    phi_data = out.evaluate("em_fields/phi", t=-1)
    phi = phi_data.values
    X, Y, Z = (phi_data.coords[c].values for c in ("X", "Y", "Z"))

    rhs_exact = rhs_fun(X, Y, Z)
    phi_exact = exact_solution(X, Y, Z)

    fig_rhs = plot_slices(rhs, rhs_exact, "RHS")
    fig_phi = plot_slices(phi, phi_exact, "Phi")

    rel_err_rhs = np.max(np.abs(rhs - rhs_exact)) / np.max(np.abs(rhs_exact))
    rel_err_phi = np.max(np.abs(phi - phi_exact)) / np.max(np.abs(phi_exact))
    # error close to the axis, where the gradient of the solution is largest
    near_axis = distance_to_axis(X, Y, Z) < 0.1
    rel_err_phi_axis = np.max(np.abs(phi - phi_exact)[near_axis]) / np.max(np.abs(phi_exact))

    print(f"Max relative error in RHS: {rel_err_rhs:.2e}")
    print(f"Max relative error in Phi: {rel_err_phi:.2e}")
    print(f"Max relative error in Phi close to the axis (r < 0.1): {rel_err_phi_axis:.2e}")

    # The profiling packaging picks these files up from here and uploads them with the run.
    results_dir = os.path.join(out.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "rel_err_rhs.npy"), rel_err_rhs)
    np.save(os.path.join(results_dir, "rel_err_phi.npy"), rel_err_phi)
    np.save(os.path.join(results_dir, "rel_err_phi_axis.npy"), rel_err_phi_axis)
    np.save(os.path.join(results_dir, "resolution.npy"), sim.grid.num_elements)
    np.save(os.path.join(results_dir, "spline_degree.npy"), sim.derham_opts.degree)
    with open(os.path.join(results_dir, "variant.json"), "w") as f:
        json.dump(
            {
                "splines": args.splines,
                "precond": args.precond,
                "poloidal_solver": args.poloidal_solver,
                "hole_bc": args.hole_bc,
            },
            f,
        )

    if save_figs:
        fig_rhs.savefig(os.path.join(results_dir, "rhs_slices.png"))
        fig_phi.savefig(os.path.join(results_dir, "phi_slices.png"))
    else:
        plt.show()


if __name__ == "__main__":
    # the variant flags are parsed by params_gyrokinetic_poisson.py, the rest here
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process and plot a saved simulation run.")
    pproc_parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: sim_1 beside this script)",
    )
    pproc_parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    pproc_args = pproc_parser.parse_args(other_flags)
    main(pproc_args.path_out, save_figs=not pproc_args.show)
