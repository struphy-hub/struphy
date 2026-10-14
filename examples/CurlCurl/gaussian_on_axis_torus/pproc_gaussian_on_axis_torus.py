"""Post-process and plot the CurlCurl gaussian_on_axis_torus example.

Run as ``python pproc_gaussian_on_axis_torus.py [sim_folder] [variant flags of params_gaussian_on_axis_torus.py]``. The variant flags must be
those of the run; the exact solution and the run parameters are taken from ``params_gaussian_on_axis_torus.py``, which reads them.
Figures, errors and the resolution are written into ``<sim_folder>/results``, where the profiling packaging picks
them up and uploads them with the run.
``submit_gaussian_on_axis_torus.py`` calls this script after each simulation.
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
from params_gaussian_on_axis_torus import args, distance_to_axis, exact_solution, parser, rhs, sim  # noqa: E402


def plot_slices(num, exact, name, X, Y, Z):
    """Numerical solution, exact solution and error of the magnitude on the poloidal plane at eta3 = 0.5 and on
    the surface eta2 = 0.5 seen from above, as functions of (R, z) and (x, y)."""
    fig = plt.figure(figsize=(12, 12))
    idx3 = (slice(None), slice(None), num.shape[2] // 2)
    idx2 = (slice(None), num.shape[1] // 2, slice(None))
    for col, (idx, h, v, labels) in enumerate(
        [(idx3, np.sqrt(X**2 + Y**2), Z, ("R", "z")), (idx2, X, Y, ("x", "y"))],
    ):
        for row, (data, what) in enumerate(
            [(num[idx], "from struphy"), (exact[idx], "exact"), (np.abs(num[idx] - exact[idx]), "error")]
        ):
            plt.subplot(3, 2, 2 * row + col + 1)
            plt.pcolormesh(h[idx], v[idx], np.nan_to_num(data), shading="gouraud")
            plt.colorbar()
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.title(f"|{name}| {what}")
    fig.tight_layout()
    return fig


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    out = Output(path_out).pproc(create_vtk=True, physical=True)

    # Cartesian components of the fields at the cell centres of the simulation grid
    e_xyz = out.evaluate("em_fields/e_field_xyz", t=-1)
    j_xyz = out.evaluate("em_fields/source_xyz", t=0)
    X, Y, Z = (e_xyz.coords[c_].values for c_ in ("X", "Y", "Z"))

    e = np.stack([e_xyz.isel(component=k).values for k in range(3)])
    j = np.stack([j_xyz.isel(component=k).values for k in range(3)])
    e_exact = np.stack([exact_solution(k)(X, Y, Z) for k in range(3)])
    j_exact = np.stack([rhs(k)(X, Y, Z) for k in range(3)])

    fig_rhs = plot_slices(np.linalg.norm(j, axis=0), np.linalg.norm(j_exact, axis=0), "J", X, Y, Z)
    fig_e = plot_slices(np.linalg.norm(e, axis=0), np.linalg.norm(e_exact, axis=0), "E", X, Y, Z)

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

    if save_figs:
        fig_rhs.savefig(os.path.join(results_dir, "rhs_slices.png"))
        fig_e.savefig(os.path.join(results_dir, "e_slices.png"))
    else:
        plt.show()


if __name__ == "__main__":
    # the variant flags are parsed by params_gaussian_on_axis_torus.py, the rest here
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
