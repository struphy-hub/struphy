"""Space-charge benchmark: expansion of a uniform sheet beam in a field-free channel.

Run from the repository root with::

    python examples/IonOpticsElectrostatic/sheet_beam_expansion/sheet_beam_expansion.py

A laminar proton sheet beam (uniform in |y| <= a0, invariant in z) is injected
continuously between two grounded plates at y = ±h. By Gauss's law, the field at the edge of
a uniform slab is E = I' / (2 v) (normalized units, I' = current per unit z-length),
independent of the plates. Every ray therefore follows

    y(x) = y0 (1 + I' (x - x0)² / (4 a0 v³)),

and the beam stays uniform, with rms size a(x) / sqrt(3). This checks the charge
deposition, the Poisson solve with electrodes, and the self-consistent pusher
in two dimensions. Writes ``sheet_beam_expansion.png`` next to this file.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt

from struphy import (
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    Time,
    WeightsParameters,
    domains,
    grids,
    maxwellians,
)
from struphy.models import IonOpticsElectrostatic
from struphy.physics.ion_optics_units import IonOpticsUnits
from struphy.pic.ion_beams import LossTag, PlaneSource

UNITS = IonOpticsUnits(length=1e-3, voltage=1e3)  # mm, kV; protons


@dataclass(frozen=True)
class Case:
    """Sheet beam in SI units; the current is per metre of slit length (z)."""

    beam_energy_eV: float = 5e3
    half_width: float = 2e-3  # m
    current_per_length: float = 0.055  # A/m, i.e. 55 µA per mm of slit length
    half_gap: float = 5e-3  # m, plates at y = ±h
    length: float = 80e-3  # m
    x_start: float = 1e-3  # m


CASE = Case()


def normalized(case=CASE, units=UNITS):
    """Normalized beam parameters: a0, v0, I' (current per unit z-length), h, length, x0."""
    return {
        "a0": case.half_width / units.length,
        "v0": float(units.speed(case.beam_energy_eV)),
        # current per unit length: A/m -> (charge unit / time unit) per length unit
        "current": case.current_per_length * units.length / units.current,
        "h": case.half_gap / units.length,
        "length": case.length / units.length,
        "x0": case.x_start / units.length,
    }


def edge_envelope(x, p):
    """Analytic half-width of the laminar uniform sheet beam."""
    return p["a0"] * (1.0 + p["current"] * (x - p["x0"]) ** 2 / (4.0 * p["a0"] * p["v0"] ** 3))


def build_simulation(output_dir, num_elements=(160, 20), degree=3, dt=0.04, end_time=36.0, rate=250.0, case=CASE):
    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("Run this example on one MPI rank.")
    p = normalized(case)
    h, length = p["h"], p["length"]
    source = PlaneSource(
        rate=rate,
        current=p["current"],  # the domain is one length unit thick in z
        axis=0,
        eta_plane=p["x0"] / length,
        eta_ranges=(((h - p["a0"]) / (2 * h), (h + p["a0"]) / (2 * h)), (0.0, 1.0)),
        velocity=(p["v0"], 0.0, 0.0),
        seed=21,
    )
    model = IonOpticsElectrostatic(
        base_units=UNITS.base_units(),
        electrode_faces=((False, False), (True, True), (False, False)),  # grounded plates (phi = 0)
        source=source,
        loss_tags=(LossTag("plates", axis=1), LossTag("outlet", axis=0, side=1), LossTag("inlet", axis=0, side=0)),
        space_charge=True,
    )
    model.em_fields.e_field.save_data = False
    model.ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    model.ions.set_markers(
        loading_params=LoadingParameters(Np=1, specific_markers=((0.5, 0.5, 0.5, 0.0, 0.0, 0.0),)),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(bc=("remove", "remove", "periodic")),
        saving_params=SavingParameters(n_markers=1),
        bufsize=float(2.5 * rate * end_time),
    )
    for prop in model.prop_list:
        prop.options = prop.Options()
    return Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(output_dir), sim_folder="sheet_beam_expansion"),
        time_opts=Time(dt=dt, Tend=end_time, split_algo="Strang"),
        domain=domains.Cuboid(l1=0.0, r1=length, l2=-h, r2=h),
        grid=grids.TensorProductGrid(num_elements=(*num_elements, 1)),
        derham_opts=DerhamOptions(degree=(degree, degree, 1), bcs=(("free", "free"),) * 3),
    )


def live_beam(sim):
    """Physical (x, y) of the markers in flight."""
    p = normalized()
    particles = sim.model.ions.var.particles
    markers = particles.markers[particles.valid_mks]
    return markers[:, 0] * p["length"], -p["h"] + 2 * p["h"] * markers[:, 1]


def rms_envelope(x, y, bins):
    """RMS size of the beam in slices of x."""
    index = np.digitize(x, bins) - 1
    rms = np.array(
        [np.sqrt(np.mean(y[index == i] ** 2)) if np.any(index == i) else np.nan for i in range(len(bins) - 1)]
    )
    return 0.5 * (bins[1:] + bins[:-1]), rms


def fit_growth(x, y, p):
    """Least-squares fit of c in <y²>(x) = a0²/3 (1 + c (x - x0)²)², with a bootstrap error."""
    s = (x - p["x0"]) ** 2

    def fit(sel):
        # each marker is an unbiased sample of <3 y² / a0²> = (1 + c s)²; scan c
        ratio = 3.0 * y[sel] ** 2 / p["a0"] ** 2
        cs = np.linspace(0.0, 4 * p["current"] / (4.0 * p["a0"] * p["v0"] ** 3), 2001)
        residual = [np.mean((ratio - (1 + c * s[sel]) ** 2) ** 2) for c in cs]
        return cs[int(np.argmin(residual))]

    rng = np.random.default_rng(0)
    best = fit(np.arange(len(x)))
    boot = [fit(rng.integers(0, len(x), len(x))) for _ in range(30)]
    return best, float(np.std(boot))


def plot_results(sim, output):
    p = normalized()
    x, y = live_beam(sim)
    bins = np.linspace(p["x0"] + 1.0, p["length"] - 1.0, 40)
    centers, rms = rms_envelope(x, y, bins)
    expected = edge_envelope(centers, p) / np.sqrt(3.0)
    error = np.nanmax(np.abs(rms / expected - 1.0))
    growth = edge_envelope(p["length"], p) / p["a0"] - 1.0
    coefficient, sigma = fit_growth(x, y, p)
    analytic = p["current"] / (4.0 * p["a0"] * p["v0"] ** 3)
    print(f"Analytic edge growth over the channel: {100 * growth:.1f} %")
    print(f"Max. relative deviation of the rms size in 2 mm slices: {100 * error:.2f} %")
    print(
        f"Growth coefficient: fit {coefficient:.4e} ± {sigma:.1e}, analytic {analytic:.4e} ({(coefficient / analytic - 1) * 100:+.1f} %)"
    )
    print(f"Plate losses: {sim.model.ledger.lost_markers['plates']}")

    fig = plt.figure(figsize=(14, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1))
    ax_beam = fig.add_subplot(grid[0, :])
    ax_env, ax_profile = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    fig.suptitle(
        f"Sheet-beam expansion by space charge: {CASE.beam_energy_eV / 1e3:g} keV protons, "
        f"{CASE.current_per_length * 1e3:g} µA per mm of slit, a₀ = {CASE.half_width * 1e3:g} mm"
    )

    counts, x_edges, y_edges = np.histogram2d(x, y, bins=(160, 60), range=((0, p["length"]), (-p["h"], p["h"])))
    image = ax_beam.pcolormesh(x_edges, y_edges, counts.T, cmap="inferno")
    fig.colorbar(image, ax=ax_beam, label="Markers per bin", pad=0.01)
    x_line = np.linspace(p["x0"], p["length"], 200)
    for sign in (-1, 1):
        ax_beam.plot(
            x_line,
            sign * edge_envelope(x_line, p),
            color="#4e79a7",
            linewidth=1.5,
            label="Analytic edge" if sign > 0 else None,
        )
        ax_beam.plot(
            (0, p["length"]),
            (sign * p["h"],) * 2,
            color="0.6",
            linewidth=6,
            label="Grounded plates" if sign > 0 else None,
        )
    ax_beam.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title=f"Steady beam, {len(x)} markers in flight",
        ylim=(-p["h"] * 1.08, p["h"] * 1.08),
    )
    ax_beam.legend(loc="upper left", fontsize=8)

    ax_env.plot(centers, rms, "o", label="Struphy (rms of live markers)")
    ax_env.plot(x_line, edge_envelope(x_line, p) / np.sqrt(3), "k-", label="Analytic a(x)/√3")
    ax_env.plot(x_line, np.full_like(x_line, p["a0"] / np.sqrt(3)), "k:", label="Without space charge")
    ax_env.set(
        xlabel="x (mm)",
        ylabel="RMS beam size (mm)",
        title=f"Envelope: fitted growth {(coefficient / analytic - 1) * 100:+.1f} ± {100 * sigma / analytic:.1f} % vs analytic",
    )
    ax_env.legend(fontsize=8)

    for x_slice, colour in ((10.0, "#4e79a7"), (70.0, "#e15759")):
        select = np.abs(x - x_slice) < 2.0
        a = edge_envelope(x_slice, p)
        ax_profile.hist(
            y[select],
            bins=30,
            range=(-p["h"], p["h"]),
            histtype="step",
            color=colour,
            density=True,
            label=f"x = {x_slice:g} ± 2 mm",
        )
        ax_profile.plot((-a, -a, a, a), (0, 1 / (2 * a), 1 / (2 * a), 0), color=colour, linestyle="--")
    ax_profile.set(
        xlabel="y (mm)", ylabel="Normalized density", title="Transverse profile (dashed: analytic uniform slab)"
    )
    ax_profile.legend(fontsize=8)

    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")
    return centers, rms, expected, error, coefficient, sigma, analytic


def main():
    sim = build_simulation(Path(__file__).parent / "output")
    sim.run()
    plot_results(sim, Path(__file__).with_name("sheet_beam_expansion.png"))


if __name__ == "__main__":
    main()
