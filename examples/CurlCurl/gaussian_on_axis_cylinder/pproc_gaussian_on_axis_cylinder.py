"""Post-process and plot the CurlCurl gaussian_on_axis_cylinder example.

Run as ``python pproc_gaussian_on_axis_cylinder.py [sim_folder] [variant flags of params_gaussian_on_axis_cylinder.py]``. The variant flags must be
those of the run; the exact solution and the run parameters are taken from ``params_gaussian_on_axis_cylinder.py``, which reads them.
Figures, errors and the resolution are written into ``<sim_folder>/results``, where the profiling packaging picks
them up and uploads them with the run.
``submit_gaussian_on_axis_cylinder.py`` calls this script after each simulation.
"""

import argparse
import json
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
from params_gaussian_on_axis_cylinder import args, distance_to_axis, exact_solution, parser, rhs, sim  # noqa: E402


def exact_components(fun):
    """The exact vector field with the Cartesian components fun(k)(x, y, z), as a function of
    (eta1, eta2, eta3, x, y, z) for the slice plots: its three components along a first axis."""
    return lambda e1, e2, e3, x, y, z: np.stack([fun(k)(x, y, z) for k in range(3)])


def save(figs, results_dir, save_figs):
    """Save the figures to ``results_dir`` and close them (``save_figs``), or keep them open for ``plt.show()``."""
    if save_figs:
        for fig_name, fig in figs.items():
            fig.savefig(os.path.join(results_dir, f"{fig_name}.png"))
            plt.close(fig)


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

    # the Cartesian components are not defined on the pole (singular mapping), which is left out
    valid = np.isfinite(e).all(axis=0) & np.isfinite(j).all(axis=0)
    rel_err_rhs = np.max(np.abs(j - j_exact)[:, valid]) / np.max(np.abs(j_exact))
    rel_err_e = np.max(np.abs(e - e_exact)[:, valid]) / np.max(np.abs(e_exact))
    # error close to the axis, where the gradient of the solution is largest
    near_axis = valid & (distance_to_axis(X, Y, Z) < 0.1)
    rel_err_e_axis = np.max(np.abs(e - e_exact)[:, near_axis]) / np.max(np.abs(e_exact))

    print(f"Max relative error in J: {rel_err_rhs:.2e}")
    print(f"Max relative error in E: {rel_err_e:.2e}")
    print(f"Max relative error in E close to the axis (r < 0.1): {rel_err_e_axis:.2e}")

    # The profiling packaging picks these files up from here and uploads them with the run.
    results_dir = os.path.join(out.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "rel_err_rhs.npy"), rel_err_rhs)
    np.save(os.path.join(results_dir, "rel_err_e.npy"), rel_err_e)
    np.save(os.path.join(results_dir, "rel_err_e_axis.npy"), rel_err_e_axis)
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

    # The Cartesian components of J (given in Cartesian components) and E (whose exact solution is Cartesian) in the
    # physical planes (see plot_physical_slices), and 3D views of their x-components from VTK files of a finer
    # post-processing, at the first, middle and last saved time, saved as e.g. e1_slices_xy_t1.5.png, e1_3d_t1.5.png
    snapshots = out.standard_snapshots()
    out.pproc(create_vtk=True, physical=True, celldivide=CELLDIVIDE, vtk_steps=snapshots, force=True)
    for n in snapshots:
        t = float(out.time[n])
        fields = [
            ("rhs", f"J at t = {t:.3g}", "em_fields/source", n, exact_components(rhs), "physical"),
            # E is the solution of the last time step; it is zero at the start, where no exact solution is shown
            (
                "e",
                f"E at t = {t:.3g}",
                "em_fields/e_field",
                n,
                exact_components(exact_solution) if n > 0 else None,
                "physical",
            ),
        ]
        figs = plot_physical_slices(out, sim.domain, sim.grid.num_elements, fields, phi=angle, components=True)
        save({f"{name}_t{t:.6g}": fig for name, fig in figs.items()}, results_dir, save_figs)
        for key, variable in (("rhs", "source"), ("e", "e_field")):
            filename = os.path.join(results_dir, f"{key}1_3d_t{t:.6g}.png")
            plot_field_3d(out.path_out, variable, snapshot=n, component=0, physical=True, filename=filename)

    if not save_figs:
        plt.show()


if __name__ == "__main__":
    # the variant flags are parsed by params_gaussian_on_axis_cylinder.py, the rest here
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
