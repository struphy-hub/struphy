"""Post-process and plot the CurlCurl gaussian_cube example.

Run as ``python pproc_gaussian_cube.py [sim_folder] [variant flags of params_gaussian_cube.py]``. The variant flags must be
those of the run; the exact solution and the run parameters are taken from ``params_gaussian_cube.py``, which reads them.
Figures, errors and the resolution are written into ``<sim_folder>/results``, where the profiling packaging picks
them up and uploads them with the run.
``submit_gaussian_cube.py`` calls this script after each simulation.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output
from struphy.post_processing.plot_3d import plot_field_3d
from struphy.post_processing.slices import plot_physical_slices

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "sim_1"

# evaluation points per cell of the VTK files, i.e. the resolution of the 3D plots
CELLDIVIDE = 2

# The exact solution, the variant flags and the simulation of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
from params_gaussian_cube import args, exact_solution, parser, rhs, sim  # noqa: E402


def exact_norm(fun):
    """Magnitude of the exact vector field with the Cartesian components fun(k)(x, y, z), as a function of
    (eta1, eta2, eta3, x, y, z) for the slice plots."""
    return lambda e1, e2, e3, x, y, z: np.linalg.norm(np.stack([fun(k)(x, y, z) for k in range(3)]), axis=0)


def main(path_out=DEFAULT_OUTPUT, save_figs=True, angle=np.pi):
    # the errors below are computed on the post-processing grid of celldivide=1 (the cell boundaries)
    out = Output(path_out).pproc(physical=True)

    # Cartesian components of the fields at the cell centres of the simulation grid
    e_xyz = out.evaluate("em_fields/e_field_xyz", t=-1)
    j_xyz = out.evaluate("em_fields/source_xyz", t=0)
    X, Y, Z = (e_xyz.coords[c_].values for c_ in ("X", "Y", "Z"))

    e = np.stack([e_xyz.isel(component=k).values for k in range(3)])
    j = np.stack([j_xyz.isel(component=k).values for k in range(3)])
    e_exact = np.stack([exact_solution(k)(X, Y, Z) for k in range(3)])
    j_exact = np.stack([rhs(k)(X, Y, Z) for k in range(3)])

    # magnitudes in the physical planes x-y, x-z, y-z and (cylinder, torus) the poloidal plane at the angle `angle`
    figs = plot_physical_slices(
        out,
        sim.domain,
        sim.grid.num_elements,
        [
            ("rhs", "|J|", "em_fields/source", 0, exact_norm(rhs), "1"),
            ("e", "|E|", "em_fields/e_field", -1, exact_norm(exact_solution), "1"),
        ],
        phi=angle,
    )

    rel_err_rhs = np.max(np.abs(j - j_exact)) / np.max(np.abs(j_exact))
    rel_err_e = np.max(np.abs(e - e_exact)) / np.max(np.abs(e_exact))

    print(f"Max relative error in J: {rel_err_rhs:.2e}")
    print(f"Max relative error in E: {rel_err_e:.2e}")

    # the thresholds hold at the profiling resolution, not at the small one of --test-run
    if not args.test_run:
        assert rel_err_rhs < 2.5e-2, f"The computed RHS does not match the exact RHS, max rel error = {rel_err_rhs}."
        assert rel_err_e < 1.7e-2, (
            f"The computed solution does not match the exact solution, max rel error = {rel_err_e}."
        )

    # The profiling packaging picks these files up from here and uploads them with the run.
    results_dir = os.path.join(out.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "rel_err_rhs.npy"), rel_err_rhs)
    np.save(os.path.join(results_dir, "rel_err_e.npy"), rel_err_e)
    np.save(os.path.join(results_dir, "resolution.npy"), sim.grid.num_elements)
    np.save(os.path.join(results_dir, "spline_degree.npy"), sim.derham_opts.degree)

    # 3D views of the fields from VTK files of a finer post-processing, saved as <field>_3d.png
    out.pproc(create_vtk=True, physical=True, celldivide=CELLDIVIDE, force=True)
    for key, variable in (("rhs", "source"), ("e", "e_field")):
        plot_field_3d(out.path_out, variable, filename=os.path.join(results_dir, f"{key}_3d.png"))

    if save_figs:
        for fig_name, fig in figs.items():
            fig.savefig(os.path.join(results_dir, f"{fig_name}.png"))
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    # the variant flags are parsed by params_gaussian_cube.py, the rest here
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
