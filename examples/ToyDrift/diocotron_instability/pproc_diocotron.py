"""Post-process and plot the diocotron instability.

Run as ``python pproc_diocotron.py [sim_1 sim_2 ...] [flags of params_diocotron.py] [--show]`` to compare several
runs; with more than one folder only the growth-rate comparison is made. The figures are written into
``<sim_folder>/results`` of the (first) run, where the profiling packaging picks them up and uploads them with the
run (``--show`` shows them instead): the growth of the electrostatic energy, the equilibrium profiles, the standard
plots of the examples (slices in the physical planes and 3D views of phi) and the snapshots of the products below
at the first, middle and last saved time, and the marker trajectories.
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

# evaluation points per cell of the VTK files, i.e. the resolution of the 3D plots
CELLDIVIDE = 2

# The flags of the run are parsed by params_diocotron.py (imported from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
import params_diocotron as params  # noqa: E402

DEFAULT_OUTPUT = Path(params.args.out_folder) / params.args.sim_folder

# scalar whose exponential growth rate is fitted, and the fit window in Struphy time units
FIT_QUANTITY = "en_phi"
FIT_WINDOW = (0.0, 42.0)

SHOW_EQUIL_PROFILE = True

# products shown at the standard snapshots in the physical XY plane
SNAPSHOTS = [
    ("kinetic_ions", "e1_e2_density", "f"),
    ("kinetic_ions", "e1_e2_density", "delta_f"),
    ("em_fields", "phi_xyz"),
]


def product(run, parts):
    """Look up a saved product such as ``("kinetic_ions", "e1_e2_density", "f")`` via evaluate()."""
    species, *rest = parts
    if len(rest) == 1:
        return run.evaluate("/".join(parts))
    return run.evaluate(f"{species}/{rest[-1]}", dataset="/".join(rest))


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


def fit_growth(series, window=(None, None)):
    """Fit ``exp(rate * t + intercept)`` to the positive samples inside ``window``."""
    time, values = series.t.values, series.values
    lo = time[0] if window[0] is None else window[0]
    hi = time[-1] if window[1] is None else window[1]
    mask = (time >= lo) & (time <= hi) & np.isfinite(values) & (values > 0)
    if np.count_nonzero(mask) < 2:
        return None
    rate, intercept = np.polyfit(time[mask], np.log(values[mask]), 1)
    return rate, intercept, time[mask]


def main(paths=(DEFAULT_OUTPUT,), save_figs=True):
    runs = [Output(path) for path in paths]
    run = runs[0]
    snapshots = run.standard_snapshots()
    # VTK files (for the 3D views) of the standard snapshots only; the other runs only give their scalars
    run.pproc(physical=True, create_vtk=True, celldivide=CELLDIVIDE, vtk_steps=snapshots)

    # The profiling packaging picks these files up from here and uploads them with the run.
    results_dir = run.path_out / "results"
    results_dir.mkdir(exist_ok=True)

    def save(figs):
        """Save the figures as ``<name>.png`` into the results folder and close them, or keep them for plt.show()."""
        if save_figs:
            for name, fig in figs.items():
                fig.savefig(results_dir / f"{name}.png")
                plt.close(fig)

    # growth rate of the electrostatic energy, one curve per run
    fig, ax = plt.subplots()
    for each in runs:
        series = each.evaluate("scalars", variables=FIT_QUANTITY)[FIT_QUANTITY]
        (line,) = ax.plot(series.t, series, label=each.path_out.name)
        result = fit_growth(series, FIT_WINDOW)
        if result is not None:
            rate, intercept, time = result
            ax.plot(time, np.exp(rate * time + intercept), "--", color=line.get_color())
        print(f"{each.path_out.name}: growth rate = {None if result is None else result[0]}")
    ax.set(xlabel="time", yscale="log", title=f"Evolution of {FIT_QUANTITY}")
    ax.legend()
    save({FIT_QUANTITY: fig})

    if len(runs) == 1:
        if SHOW_EQUIL_PROFILE:
            save({"equilibrium": plot_equilibrium(run.path_out)[0]})

        # at the first, middle and last saved time, e.g. phi_slices_xy_t1.5.png and phi_3d_t1.5.png
        for n in snapshots:
            t = float(run.time[n])
            figs = {}
            for path in SNAPSHOTS:
                snapshot = product(run, path).isel(t=n)
                if "eta3" in snapshot.dims:
                    snapshot = snapshot.isel(eta3=0)
                fig, ax = plt.subplots()
                snapshot.plot(x="X", y="Y", ax=ax)
                ax.set(aspect="equal", title=f"{'/'.join(path)}, t = {t:.3g}")
                figs[f"{'_'.join(path)}_t{t:.6g}"] = fig
            field = ("phi", f"phi at t = {t:.3g}", "em_fields/phi", n, None, None)
            slices = plot_physical_slices(run, run.domain, run.grid.num_elements, [field], components=True)
            figs.update({f"{name}_t{t:.6g}": fig for name, fig in slices.items()})
            save(figs)
            plot_field_3d(run.path_out, "phi", snapshot=n, filename=results_dir / f"phi_3d_t{t:.6g}.png")

        orbits = run.evaluate("kinetic_ions/orbits").isel(marker=slice(0, 1000))
        fig, ax = plt.subplots()
        ax.plot(orbits.x, orbits.y, lw=0.5)
        ax.set(xlabel="$x$", ylabel="$y$", title="Marker trajectories", aspect="equal")
        save({"orbits": fig})

    if not save_figs:
        plt.show()


if __name__ == "__main__":
    # the flags of the run are parsed by params_diocotron.py, the rest here
    _, other_flags = params.parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Plot saved simulation runs.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=[DEFAULT_OUTPUT],
        help="Simulation output folders (default: the output folder of the run flags, sim_1 beside this script)",
    )
    parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    # the figures are saved by default; the flag is still passed by submit_diocotron_instability.py
    parser.add_argument("--save-results", action="store_true", help="Save the figures (the default).")
    args = parser.parse_args(other_flags)
    main(args.paths, save_figs=not args.show)
