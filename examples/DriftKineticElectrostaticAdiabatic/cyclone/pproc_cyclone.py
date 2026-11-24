"""Post-process and plot the cyclone base case (DriftKineticElectrostaticAdiabatic).

Run as ``python pproc_cyclone.py [path_out] [flags of params_cyclone.py] [--show]``. Figures are written
into ``<path_out>/results`` (``--show`` shows them instead): the growth of the potential, the standard plots of the
examples (slices in the physical planes and 3D views of phi and rho) and the snapshots of the products below at the
first, middle and last saved time, and the marker trajectories.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt

from struphy import Output
from struphy.post_processing.plot_3d import plot_field_3d
from struphy.post_processing.slices import plot_physical_slices

SCRIPT_DIR = Path(__file__).resolve().parent

# The flags of the run (parsed from the same command line), for the default output folder.
sys.path.insert(0, str(SCRIPT_DIR))
from params_cyclone import args, parser  # noqa: E402

DEFAULT_OUTPUT = Path(args.out_folder) / args.sim_folder

# evaluation points per cell of the VTK files, i.e. the resolution of the 3D plots
CELLDIVIDE = 2

# standard plots (slices and 3D views) of the scalar fields: (key, label, species, variable)
FIELDS = [("phi", "phi", "em_fields", "phi"), ("rho", "rho", "diagnostics", "rho")]

# scalar whose exponential growth rate is fitted, and the fit window in Struphy time units
FIT_QUANTITY = "phi_integral"
FIT_WINDOW = (0.0, None)

SHOW_EQUIL_PROFILE = False

# products shown at the standard snapshots, as (product, physical plane, logical coordinate held fixed)
SNAPSHOTS = [
    ("kinetic_ions/e1_e2_density/delta_f", "RZ", {}),
    ("em_fields/phi_xyz", "RZ", {"eta3": 0}),
    ("diagnostics/rho_xyz", "RZ", {"eta3": 0}),
    ("diagnostics/rho_xyz", "XY", {"eta2": 0}),
]


def product(run, name):
    """Look up a saved product such as ``"kinetic_ions/e1_e2_density/f"`` via evaluate()."""
    species, *rest = name.split("/")
    if len(rest) == 1:
        return run.evaluate(name)
    return run.evaluate(f"{species}/{rest[-1]}", dataset="/".join(rest))


def plot_growth(series, window=(None, None)):
    """Plot a positive time series on a log axis with a fitted exponential ``exp(rate * t)``."""
    time, values = series.t.values, series.values
    lo = time[0] if window[0] is None else window[0]
    hi = time[-1] if window[1] is None else window[1]
    fig, ax = plt.subplots()
    ax.plot(time, values, label=series.name)

    mask = (time >= lo) & (time <= hi) & np.isfinite(values) & (values > 0)
    if np.count_nonzero(mask) >= 2:
        rate, intercept = np.polyfit(time[mask], np.log(values[mask]), 1)
        ax.plot(time[mask], np.exp(rate * time[mask] + intercept), "--", label=f"fit, rate = {rate:.4g}")
        print(f"{series.name}: growth rate = {rate:.6g}")

    ax.set(xlabel="time", yscale="log", title=f"Evolution of {series.name}")
    ax.legend()
    return fig, ax


def plot_plane(data, plane, fixed, n):
    """Pseudocolor plot of the saved time step ``n`` in the physical RZ or XY plane."""
    snapshot = data.isel(t=n, **fixed)
    if plane == "RZ":
        snapshot = snapshot.assign_coords(R=np.hypot(snapshot.X, snapshot.Y))
        x, y = "R", "Z"
    else:
        x, y = "X", "Y"
    fig, ax = plt.subplots()
    snapshot.plot(x=x, y=y, ax=ax)
    ax.set_aspect("equal")
    ax.set_title(f"{data.name}, t = {float(snapshot.t):.3g}")
    return fig, ax


def plot_equilibrium(path_out):
    """Radial equilibrium profiles from the geometry written at the start of the run."""
    equilibrium = pv.read(str(Path(path_out) / "geometry.vts"))
    shape = equilibrium.dimensions
    grid = np.reshape(equilibrium.points, shape + (3,))
    radius = np.sqrt(grid[..., 0] ** 2 + grid[..., 1] ** 2)[0, 0]
    pressure = np.reshape(equilibrium.point_data["p0"], shape)[0, 0]
    fig, ax = plt.subplots()
    ax.plot(radius, pressure, label=r"$p_0$")
    if "n0" in equilibrium.point_data:
        density = np.reshape(equilibrium.point_data["n0"], shape)[0, 0]
        ax.plot(radius, density, label=r"$n_0$")
        ax.plot(radius, pressure / density, label=r"$T_0$")
    ax.set(xlabel=r"$R$", title="Radial equilibrium profiles")
    ax.legend()
    return fig, ax


def plot_trajectories(orbits, max_markers=1000):
    """Marker paths in the physical XY plane."""
    selected = orbits.isel(marker=slice(0, max_markers))
    fig, ax = plt.subplots()
    ax.plot(selected.x, selected.y, lw=0.5)
    ax.set(xlabel="$x$", ylabel="$y$", title="Marker trajectories", aspect="equal")
    return fig, ax


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    run = Output(path_out)
    snapshots = run.standard_snapshots()
    # VTK files (for the 3D views) of the standard snapshots only
    run.pproc(physical=True, create_vtk=True, celldivide=CELLDIVIDE, vtk_steps=snapshots)

    # The profiling packaging picks these files up from here and uploads them with the run.
    results_dir = Path(run.path_out) / "results"
    results_dir.mkdir(exist_ok=True)

    def save(figs):
        """Save the figures as ``<name>.png`` into the results folder and close them, or keep them for plt.show()."""
        if save_figs:
            for name, fig in figs.items():
                fig.savefig(results_dir / f"{name}.png")
                plt.close(fig)

    # growth rate of the electrostatic potential
    fig, _ = plot_growth(run.evaluate("scalars", variables=FIT_QUANTITY)[FIT_QUANTITY], window=FIT_WINDOW)
    save({FIT_QUANTITY: fig})

    if SHOW_EQUIL_PROFILE:
        save({"equilibrium": plot_equilibrium(run.path_out)[0]})

    # at the first, middle and last saved time, e.g. phi_slices_xy_t1.5.png and phi_3d_t1.5.png
    for n in snapshots:
        t = float(run.time[n])
        figs = {}
        for name, plane, fixed in SNAPSHOTS:
            key = "_".join([name.replace("/", "_"), plane] + [f"{k}{v}" for k, v in fixed.items()])
            figs[f"{key}_t{t:.6g}"] = plot_plane(product(run, name), plane, fixed, n)[0]
        for key, label, species, variable in FIELDS:
            field = (key, f"{label} at t = {t:.3g}", f"{species}/{variable}", n, None, None)
            slices = plot_physical_slices(run, run.domain, run.grid.num_elements, [field], components=True)
            figs.update({f"{fig_name}_t{t:.6g}": fig for fig_name, fig in slices.items()})
            plot_field_3d(
                run.path_out, variable, species=species, snapshot=n, filename=results_dir / f"{key}_3d_t{t:.6g}.png"
            )
        save(figs)

    save({"orbits": plot_trajectories(run.evaluate("kinetic_ions/orbits"), max_markers=1000)[0]})

    if not save_figs:
        plt.show()


if __name__ == "__main__":
    # the flags of the run are parsed by params_cyclone.py, the rest here
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process and plot a saved simulation run.")
    pproc_parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: the output folder of the run flags, sim_1 beside this script)",
    )
    pproc_parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    pproc_args = pproc_parser.parse_args(other_flags)
    main(pproc_args.path_out, save_figs=not pproc_args.show)
