"""Plots and movies for the ITPA TAE benchmark run (ShearAlfven).

Run ``python pproc_ShearAlfven.py [path_out]`` after ``params_ShearAlfven.py``; figures are saved
to ``<path_out>/plots/``. Fields are shown at the initial, middle and final saved time. The plots
use plasma-plots (``pip install "struphy[pproc]"``), which adds ``.plot`` to the Output and
``.plasma.plot`` to every product.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display needed
from matplotlib import pyplot as plt

from struphy import Output

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "sim_data"

# field key -> (product, representation); vector fields are shown as 2-form components
FIELDS = {
    "v": ("mhd/velocity", "2"),
    "b": ("em_fields/b_field", "2"),
}
COMPONENTS = {"r": 0, "pol": 1, "tor": 2}  # logical directions
ENERGIES = ("en_U", "en_B")  # energy scalars shown next to en_tot

SHOW = (("v", "r"), ("v", "pol"), ("b", "r"))  # (field, component) pairs to plot
ETA3 = 0.25  # toroidal position of the poloidal cross-section
# logical grids of the cross-sections, endpoints included so that the plots close in eta2
ETA1, ETA2, ETA3_GRID = np.linspace(0, 1, 101), np.linspace(0, 1, 201), np.linspace(0, 1, 51)
MAKE_MOVIE = False
MOVIE_FIELD, MOVIE_COMPONENT, MOVIE_STEP = "v", "r", 1  # every MOVIE_STEP-th saved time is a frame


def snapshot_times(run):
    """Positions of the initial, middle and final saved time."""
    last = run.time.size - 1
    return sorted({0, last // 2, last})


def get_field(run, field, comp, **selection):
    """One field component as an xarray.DataArray, evaluated from the saved spline coefficients."""
    name, representation = FIELDS[field]
    data = run.evaluate(name, representation=representation, **selection)
    label = field
    if "component" in data.dims:
        data = data.isel(component=COMPONENTS[comp], drop=True)
        label = f"{field}_{comp}"
    data.name = data.attrs["label"] = label
    return data


def minor_radius(run):
    """The minor radius of the HollowTorus as a function of eta1."""
    a1, a2 = run.domain.params["a1"], run.domain.params["a2"]
    return lambda eta1: a1 + (a2 - a1) * eta1


def save(result, run, name):
    """Save a plasma-plots figure to ``<path_out>/plots/name`` and close it."""
    path = Path(run.path_out) / "plots" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    result.fig.savefig(path, dpi=150)
    plt.close(result.fig)
    print(f"Saved {path}")


def plot_field(run, field, comp, times):
    """Radial profiles, poloidal and midplane (Z=0) cross-sections and (m, n) harmonics at the given times."""
    colors = dict(coords="physical", symmetric=True, cmap="RdBu_r", nrows=1, ncols=len(times))

    poloidal = get_field(run, field, comp, t=times, eta1=ETA1, eta2=ETA2, eta3=ETA3)
    label = poloidal.attrs["label"]
    profiles = poloidal.plasma.plot.profiles(
        x="eta1",
        at=list(range(len(times))),
        x_of=minor_radius(run),
        xlabel="minor radius r",
        title=f"{label} at eta2=0, eta3={ETA3}",
        eta2=0.0,
    )
    save(profiles, run, f"{label}_vs_r.png")
    save(poloidal.plasma.plot.panels(plane="RZ", **colors), run, f"{label}_poloidal.png")

    for side, eta2 in (("outboard", 0.0), ("inboard", 0.5)):
        midplane = get_field(run, field, comp, t=times, eta1=ETA1, eta2=eta2, eta3=ETA3_GRID)
        save(midplane.plasma.plot.panels(plane="XY", **colors), run, f"{label}_midplane_{side}.png")

    for t in times:
        data = get_field(run, field, comp, t=t)
        modes = data.plasma.plot.mode_profiles(
            scale=(1, run.domain.params["tor_period"]),
            x_of=minor_radius(run),
            xlabel="minor radius r",
            top=4,
        )
        save(modes, run, f"modes/{label}_modes_t{float(data.t):.2f}.png")


def make_movie(run, field, comp):
    """GIF of the poloidal cross-section over time."""
    data = get_field(run, field, comp, t=slice(None, None, MOVIE_STEP), eta1=ETA1, eta2=ETA2, eta3=ETA3)
    anim = data.plasma.plot.animation(coords="physical", plane="RZ", symmetric=True, robust=True, cmap="RdBu_r")
    path = Path(run.path_out) / "plots" / f"{data.attrs['label']}_movie.gif"
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer="pillow", fps=10, dpi=100)
    plt.close("all")
    print(f"Saved {path}")


def main(path_out=DEFAULT_OUTPUT):
    run = Output(path_out)

    save(run.plot.energies(parts=ENERGIES), run, "energies.png")
    times = snapshot_times(run)
    for field, comp in SHOW:
        plot_field(run, field, comp, times)
    if MAKE_MOVIE:
        make_movie(run, MOVIE_FIELD, MOVIE_COMPONENT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT)
