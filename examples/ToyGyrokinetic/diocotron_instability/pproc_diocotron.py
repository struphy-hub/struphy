"""Post-process and plot the diocotron instability.

Run as ``python pproc_diocotron.py [sim_1 sim_2 ...]`` to compare several runs; with
more than one folder only the growth-rate comparison is shown.
"""

import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt

from struphy import Output

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "sim_1"

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


def main(paths=(DEFAULT_OUTPUT,)):
    runs = [Output(path).pproc(physical=True) for path in paths]
    run = runs[0]

    # growth rate of the electrostatic energy, one curve per run
    fig, ax = plt.subplots()
    for each in runs:
        series = each.scalars[FIT_QUANTITY]
        (line,) = ax.plot(series.t, series, label=each.path_out.name)
        result = fit_growth(series, FIT_WINDOW)
        if result is not None:
            rate, intercept, time = result
            ax.plot(time, np.exp(rate * time + intercept), "--", color=line.get_color())
        print(f"{each.path_out.name}: growth rate = {None if result is None else result[0]}")
    ax.set(xlabel="time", yscale="log", title=f"Evolution of {FIT_QUANTITY}")
    ax.legend()
    plt.show()

    if len(runs) > 1:
        return

    if SHOW_EQUIL_PROFILE:
        plot_equilibrium(run.path_out)
        plt.show()

    for path in SNAPSHOTS:
        data = run
        for part in path:
            data = getattr(data, part)
        snapshot = data.isel(t=-1)
        if "e3" in snapshot.dims:
            snapshot = snapshot.isel(e3=0)
        fig, ax = plt.subplots()
        snapshot.plot(x="X", y="Y", ax=ax)
        ax.set(aspect="equal", title=f"{'/'.join(path)}, t = {float(snapshot.t):.3g}")
        plt.show()

    orbits = run.kinetic_ions.orbits.isel(marker=slice(0, 1000))
    fig, ax = plt.subplots()
    ax.plot(orbits.x, orbits.y, lw=0.5)
    ax.set(xlabel="$x$", ylabel="$y$", title="Marker trajectories", aspect="equal")
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:] or [DEFAULT_OUTPUT])
