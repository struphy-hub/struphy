"""Post-process and plot the GyrokineticPoisson gaussian_hollow_torus example.

Run as ``python pproc_gaussian_hollow_torus.py [sim_folder] [variant flags of params_gaussian_hollow_torus.py]``.
The variant flags must be those of the run; the exact solution and the run parameters are taken from
``params_gaussian_hollow_torus.py``, which reads them. Figures and errors are written into ``<sim_folder>/results``,
where the profiling packaging picks them up and uploads them with the run
(``submit_gaussian_hollow_torus.py`` calls this script after each simulation).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output
from struphy.post_processing.slices import plot_physical_slices

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "sim_1"

# The exact solution, the variant flags and the simulation of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
from params_gaussian_hollow_torus import args, exact_solution, parser, rhs_fun, sim  # noqa: E402


def main(path_out=DEFAULT_OUTPUT, save_figs=True, angle=np.pi):
    out = Output(path_out).pproc(create_vtk=True)

    # Raw FEEC fields are evaluated directly from the saved spline coefficients at the
    # cell centres of the simulation grid.
    rhs = out.evaluate("em_fields/source", t=0).values

    phi_data = out.evaluate("em_fields/phi", t=-1)
    phi = phi_data.values

    # logical coordinates of the evaluation points
    E1, E2, E3 = np.meshgrid(
        *(phi_data.coords[c].values for c in ("eta1", "eta2", "eta3")),
        indexing="ij",
    )

    # pushed-forward exact solutions: the 0-forms at x = F(eta) are their values at eta
    rhs_exact = rhs_fun(E1, E2, E3)
    phi_exact = exact_solution(E1, E2, E3)

    # slices in the physical planes x-y, x-z, y-z and the poloidal plane at the angle `angle`
    figs = plot_physical_slices(
        out,
        sim.domain,
        sim.grid.num_elements,
        [
            ("rhs", "RHS", "em_fields/source", 0, lambda e1, e2, e3, x, y, z: rhs_fun(e1, e2, e3), None),
            ("phi", "Phi", "em_fields/phi", -1, lambda e1, e2, e3, x, y, z: exact_solution(e1, e2, e3), None),
        ],
        phi=angle,
    )

    rel_err_rhs = np.max(np.abs(rhs - rhs_exact)) / np.max(np.abs(rhs_exact))
    rel_err_phi = np.max(np.abs(phi - phi_exact)) / np.max(np.abs(phi_exact))

    print(f"Max relative error in RHS: {rel_err_rhs:.2e}")
    print(f"Max relative error in Phi: {rel_err_phi:.2e}")

    # the thresholds hold at the profiling resolution, not at the small one of --test-run
    if not args.test_run:
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
        for fig_name, fig in figs.items():
            fig.savefig(os.path.join(results_dir, f"{fig_name}.png"))
    else:
        plt.show()


if __name__ == "__main__":
    # the variant flags are parsed by params_gaussian_hollow_torus.py, the rest here
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process and plot a saved simulation run.")
    pproc_parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: sim_1 beside this script)",
    )
    pproc_parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    pproc_parser.add_argument(
        "--phi", type=float, default=np.pi, help="Angle of the poloidal plane in the slice plots (default: pi)."
    )
    pproc_args = pproc_parser.parse_args(other_flags)
    main(pproc_args.path_out, save_figs=not pproc_args.show, angle=pproc_args.phi)
