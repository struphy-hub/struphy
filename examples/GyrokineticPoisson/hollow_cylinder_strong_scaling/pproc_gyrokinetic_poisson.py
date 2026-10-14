"""Post-process and plot the GyrokineticPoisson hollow cylinder strong scaling case.

Run as ``python pproc_gyrokinetic_poisson.py [sim_folder] [variant flags of params_gyrokinetic_poisson.py]``.
The variant flags must be those of the run; the exact solution and the run parameters are taken from
``params_gyrokinetic_poisson.py``, which reads them. Figures and errors are written into ``<sim_folder>/results``,
where the profiling packaging picks them up and uploads them with the run
(``submit_gyrokinetic_poisson_hollow_cylinder_strong_scaling.py`` calls this script after each simulation).
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

# The exact solution, the variant flags and the simulation of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
from params_gyrokinetic_poisson import exact_solution, parser, rhs_fun, sim  # noqa: E402


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    out = Output(path_out).pproc(create_vtk=True)

    def slice_axes(k, X, Y, Z, E1, E2, E3):
        """Axes (horizontal, vertical, labels) for plotting the slice eta_k = 0.5: the cross-section (x, y)
        at eta3 = 0.5, the half plane (r, z) at eta2 = 0.5 and the logical (eta2, eta3) at eta1 = 0.5."""
        if k == 0:
            return E2, E3, "eta2", "eta3"
        elif k == 1:
            return np.sqrt(X**2 + Y**2), Z, "r", "z"
        else:
            return X, Y, "x", "y"

    def plot_slices(num, exact, name):
        """Numerical solution, exact solution and error on the slices eta_k = 0.5 (k = 1, 2, 3)."""
        fig = plt.figure(figsize=(16, 12))
        for k in range(3):
            idx = tuple(num.shape[k] // 2 if i == k else slice(None) for i in range(3))
            h, v, hlabel, vlabel = slice_axes(k, X[idx], Y[idx], Z[idx], E1[idx], E2[idx], E3[idx])
            eta_k = (E1, E2, E3)[k][idx].flat[0]
            for row, (data, what) in enumerate(
                [(num[idx], "from struphy"), (exact[idx], "exact"), (np.abs(num[idx] - exact[idx]), "error")]
            ):
                plt.subplot(3, 3, 3 * row + k + 1)
                plt.pcolormesh(h, v, data, shading="gouraud")
                plt.colorbar()
                plt.xlabel(hlabel)
                plt.ylabel(vlabel)
                plt.title(f"{name} {what}, slice at eta{k + 1} = {eta_k:.2f}")
        fig.tight_layout()
        return fig

    # Raw FEEC fields are evaluated directly from the saved spline coefficients at the
    # cell centres of the simulation grid.
    rhs = out.evaluate("em_fields/source", t=0).values

    phi_data = out.evaluate("em_fields/phi", t=-1)
    phi = phi_data.values

    # logical coordinates of the evaluation points and their images x = F(eta)
    E1, E2, E3 = np.meshgrid(
        *(phi_data.coords[c].values for c in ("eta1", "eta2", "eta3")),
        indexing="ij",
    )
    X, Y, Z = (phi_data.coords[c].values for c in ("X", "Y", "Z"))

    # pushed-forward exact solutions: the 0-forms at x = F(eta) are their values at eta
    rhs_exact = rhs_fun(E1, E2, E3)
    phi_exact = exact_solution(E1, E2, E3)

    fig_rhs = plot_slices(rhs, rhs_exact, "RHS")
    fig_phi = plot_slices(phi, phi_exact, "Phi")

    rel_err_rhs = np.max(np.abs(rhs - rhs_exact)) / np.max(np.abs(rhs_exact))
    rel_err_phi = np.max(np.abs(phi - phi_exact)) / np.max(np.abs(phi_exact))

    print(f"Max relative error in RHS: {rel_err_rhs:.2e}")
    print(f"Max relative error in Phi: {rel_err_phi:.2e}")

    assert rel_err_rhs < 2.2e-2, f"The computed RHS does not match the exact RHS, max rel error = {rel_err_rhs}."
    assert rel_err_phi < 1.3e-2, (
        f"The computed solution does not match the exact solution, max rel error = {rel_err_phi}."
    )

    # The profiling packaging picks these files up from here and uploads them with the run.
    results_dir = os.path.join(out.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "rel_err_rhs.npy"), rel_err_rhs)
    np.save(os.path.join(results_dir, "rel_err_phi.npy"), rel_err_phi)
    np.save(os.path.join(results_dir, "resolution.npy"), sim.grid.num_elements)
    np.save(os.path.join(results_dir, "spline_degree.npy"), sim.derham_opts.degree)

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
