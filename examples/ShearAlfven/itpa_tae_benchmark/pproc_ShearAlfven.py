"""Plots and movies for the ITPA TAE benchmark run (ShearAlfven).

Run ``python pproc_ShearAlfven.py [path_out]`` after ``params_ShearAlfven.py``; figures are saved
to ``<path_out>/plots/``. The plots use plasma-plots (``pip install "struphy[pproc]"``).
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display needed
from matplotlib import pyplot as plt

from struphy import Output

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "sim6_higherResolution_ShearAlfven"

# field key -> (product, representation); vector fields are shown as 2-form components
FIELDS = {
    "v": ("mhd/velocity", "2"),
    "b": ("em_fields/b_field", "2"),
}
COMPONENTS = {"r": 0, "pol": 1, "tor": 2}  # logical directions
ENERGIES = ("en_U", "en_B")  # energy scalars shown next to en_tot

FIELD, COMPONENT = "v", "pol"
ETA3 = 0.25  # toroidal position of the poloidal cross-section
# logical grids of the cross-sections, endpoints included so that the plots close in eta2
ETA1, ETA2, ETA3_GRID = np.linspace(0, 1, 101), np.linspace(0, 1, 201), np.linspace(0, 1, 51)
T_PROFILES = (0, 10, 20)  # saved-time positions of the radial profiles
T_SNAPSHOT = 10
T_MODES = (0, 50, 100)
MAKE_MOVIE = False
MOVIE_FIELD, MOVIE_COMPONENT, MOVIE_STEP = "v", "r", 1  # every MOVIE_STEP-th saved time is a frame


def get_field(run, field, comp, **selection):
    """One field component as an xarray.DataArray, evaluated from the saved spline coefficients."""
    name, representation = FIELDS[field]
    data = run.evaluate(name, representation=representation, **selection)
    if "component" in data.dims:
        data = data.isel(component=COMPONENTS[comp], drop=True)
        data.name = f"{field}_{comp}"
    else:
        data.name = field
    return data


def minor_radius(run, eta1):
    """Minor radius of the HollowTorus at the logical coordinate eta1."""
    params = run.domain.params
    return params["a1"] + (params["a2"] - params["a1"]) * eta1


def save(fig, run, name):
    """Save a figure to ``<path_out>/plots/name`` and close it."""
    path = Path(run.path_out) / "plots" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_profiles(run, field, comp):
    """Radial profile at the outboard midplane (eta2=0, eta3=ETA3), one curve per time."""
    data = get_field(run, field, comp, t=list(T_PROFILES), eta2=0.0, eta3=ETA3)
    data = data.assign_coords(r=minor_radius(run, data.eta1))
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    data.plot.line(x="r", hue="t", ax=ax)
    ax.set(xlabel="minor radius r", title=f"{data.name} at eta2=0, eta3={ETA3}")
    save(fig, run, f"{data.name}_vs_r.png")


def plot_snapshot(run, field, comp, t):
    """Poloidal cross-section at eta3=ETA3 and midplane Z=0 (outboard and inboard) at one time."""
    poloidal = get_field(run, field, comp, t=t, eta1=ETA1, eta2=ETA2, eta3=ETA3).squeeze()
    poloidal = poloidal.assign_coords(R=np.hypot(poloidal.X, poloidal.Y))
    midplane = [get_field(run, field, comp, t=t, eta1=ETA1, eta2=eta2, eta3=ETA3_GRID).squeeze() for eta2 in (0.0, 0.5)]
    vmax = float(abs(poloidal).max()) or 1.0
    colors = dict(vmin=-vmax, vmax=vmax, cmap="RdBu_r", add_colorbar=False)
    time = float(poloidal.t)

    fig, (ax_pol, ax_mid) = plt.subplots(2, 1, figsize=(7, 12), layout="constrained")
    mesh = poloidal.plot(x="R", y="Z", ax=ax_pol, **colors)
    for half in midplane:
        half.plot(x="X", y="Y", ax=ax_mid, **colors)
    ax_pol.set(aspect="equal", title=f"{poloidal.name} at eta3={ETA3}, t={time:.2f}")
    ax_mid.set(aspect="equal", title=f"{poloidal.name} at Z=0, t={time:.2f}")
    fig.colorbar(mesh, ax=[ax_pol, ax_mid], label=poloidal.name)
    save(fig, run, f"{poloidal.name}_snapshot_t{time:.2f}.png")


def plot_modes(run, field, comp, t):
    """Radial profiles of the strongest (m, n) harmonics at one time, n counted over the full torus."""
    data = get_field(run, field, comp, t=t)
    result = data.plasma.plot.mode_profiles(
        scale=(1, run.domain.params["tor_period"]),
        x_of=lambda eta1: minor_radius(run, eta1),
        xlabel="minor radius r",
        top=4,
    )
    save(result.fig, run, f"modes/{data.name}_modes_t{float(data.t):.2f}.png")


def make_movie(run, field, comp):
    """GIF of the poloidal cross-section over time."""
    data = get_field(run, field, comp, t=slice(None, None, MOVIE_STEP), eta1=ETA1, eta2=ETA2, eta3=ETA3)
    anim = data.plasma.plot.animation(coords="physical", plane="RZ", symmetric=True, robust=True, cmap="RdBu_r")
    path = Path(run.path_out) / "plots" / f"{data.name}_movie.gif"
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer="pillow", fps=10, dpi=100)
    plt.close("all")
    print(f"Saved {path}")


def main(path_out=DEFAULT_OUTPUT):
    run = Output(path_out)

    save(run.plot.energies(parts=ENERGIES).fig, run, "energies.png")
    plot_profiles(run, FIELD, COMPONENT)
    plot_snapshot(run, FIELD, COMPONENT, T_SNAPSHOT)
    for t in T_MODES:
        plot_modes(run, "v", "r", t)
        plot_modes(run, "b", "r", t)
    if MAKE_MOVIE:
        make_movie(run, MOVIE_FIELD, MOVIE_COMPONENT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT)
