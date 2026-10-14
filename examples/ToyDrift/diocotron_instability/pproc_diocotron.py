"""Post-process and plot the diocotron instability.

Run as ``python pproc_diocotron.py [sim_1 sim_2 ...] [flags of params_diocotron.py]`` to compare several runs;
with more than one folder only the growth-rate comparison is shown. With ``--save-results`` (passed by
``submit_diocotron_instability.py`` after each profiling run) the figures are written into
``<sim_folder>/results`` instead of being shown; the profiling packaging uploads them with the run.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt

from struphy import Output
from struphy.post_processing.plot_3d import plot_field_3d

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "sim_1"

# evaluation points per cell of the VTK files, i.e. the resolution of the 3D plots
CELLDIVIDE = 2

# The flags of the run are parsed by params_diocotron.py (imported from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
import params_diocotron as params  # noqa: E402

# scalar whose exponential growth rate is fitted, and the fit window in Struphy time units
FIT_QUANTITY = "en_phi"
FIT_WINDOW = (0.0, 42.0)

SHOW_EQUIL_PROFILE = True

# products shown at the last saved time in the physical XY plane
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


def main(paths=(DEFAULT_OUTPUT,), save_results=False):
    runs = [Output(path).pproc(physical=True, create_vtk=True, celldivide=CELLDIVIDE) for path in paths]
    run = runs[0]

    def show(fig, name):
        """Show the figure, or with ``save_results`` save it as ``<name>.png`` into the results folder of the run."""
        if save_results:
            results_dir = run.path_out / "results"
            results_dir.mkdir(exist_ok=True)
            fig.savefig(results_dir / f"{name}.png")
            plt.close(fig)
        else:
            plt.show()

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
    show(fig, FIT_QUANTITY)

    if len(runs) > 1:
        return

    if SHOW_EQUIL_PROFILE:
        fig, _ = plot_equilibrium(run.path_out)
        show(fig, "equilibrium")

    for path in SNAPSHOTS:
        data = product(run, path)
        snapshot = data.isel(t=-1)
        if "eta3" in snapshot.dims:
            snapshot = snapshot.isel(eta3=0)
        fig, ax = plt.subplots()
        snapshot.plot(x="X", y="Y", ax=ax)
        ax.set(aspect="equal", title=f"{'/'.join(path)}, t = {float(snapshot.t):.3g}")
        show(fig, "_".join(path))

    orbits = run.evaluate("kinetic_ions/orbits").isel(marker=slice(0, 1000))
    fig, ax = plt.subplots()
    ax.plot(orbits.x, orbits.y, lw=0.5)
    ax.set(xlabel="$x$", ylabel="$y$", title="Marker trajectories", aspect="equal")
    show(fig, "orbits")

    # 3D view of the potential at the last saved time (from the VTK files)
    results_dir = run.path_out / "results"
    results_dir.mkdir(exist_ok=True)
    plot_field_3d(run.path_out, "phi", filename=results_dir / "phi_3d.png")
    print(f"3D view of phi saved to {results_dir / 'phi_3d.png'}")


if __name__ == "__main__":
    # the flags of the run are parsed by params_diocotron.py, the rest here
    _, other_flags = params.parser.parse_known_args()
    parser = argparse.ArgumentParser(description="Plot saved simulation runs.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=[DEFAULT_OUTPUT],
        help="Simulation output folders (default: sim_1 beside this script)",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save the figures into the results folder of the (first) run instead of showing them.",
    )
    args = parser.parse_args(other_flags)
    main(args.paths, save_results=args.save_results)
