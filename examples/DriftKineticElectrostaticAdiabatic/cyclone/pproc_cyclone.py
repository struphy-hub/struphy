import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt

from struphy import Output

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "sim_1"

# scalar whose exponential growth rate is fitted, and the fit window in Struphy time units
FIT_QUANTITY = "phi_integral"
FIT_WINDOW = (0.0, None)

SHOW_EQUIL_PROFILE = False

# products shown at the last saved time, as (product, physical plane, logical coordinate held fixed)
SNAPSHOTS = [
    ("kinetic_ions/e1_e2_density/delta_f", "RZ", {}),
    ("em_fields/phi_xyz", "RZ", {"e3": 0}),
    ("diagnostics/rho_xyz", "RZ", {"e3": 0}),
    ("diagnostics/rho_xyz", "XY", {"e2": 0}),
]


def product(run, name):
    """Look up a saved product such as ``"kinetic_ions/e1_e2_density/f"`` by attribute access."""
    data = run
    for part in name.split("/"):
        data = getattr(data, part)
    return data


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


def plot_plane(data, plane, fixed):
    """Pseudocolor plot of the last saved time in the physical RZ or XY plane."""
    snapshot = data.isel(t=-1, **fixed)
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


def main(path_out=DEFAULT_OUTPUT):
    run = Output(path_out).pproc(physical=True)

    # growth rate of the electrostatic potential
    plot_growth(run.scalars[FIT_QUANTITY], window=FIT_WINDOW)

    if SHOW_EQUIL_PROFILE:
        plot_equilibrium(run.path_out)

    for name, plane, fixed in SNAPSHOTS:
        plot_plane(product(run, name), plane, fixed)

    plot_trajectories(run.kinetic_ions.orbits, max_markers=1000)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT)
