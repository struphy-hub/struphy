"""Continuously injected proton beam through the slit immersion lens (zero current).

Run from the repository root with::

    python examples/IonOpticsElectrostatic/slit_lens_injection/slit_lens_injection.py

Uses the lens and units of ``../slit_immersion_lens``. A beam with a finite width and
angular spread is injected continuously by ``InjectMarkers`` until the marker
population is steady. Removed markers are booked per boundary part
(``LossTag``): upstream electrode, downstream electrode, outlet, inlet. Checks:

* charge bookkeeping closes exactly: injected = live + sum of losses;
* the run reaches a steady state, and in it the currents to each boundary part
  agree with a Monte Carlo reference traced in the exact analytic field.

Writes ``slit_lens_injection.png`` next to this file.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slit_immersion_lens"))
from slit_immersion_lens import DESIGN, UNITS, analytic_lens  # noqa: E402

from struphy import (  # noqa: E402
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
from struphy.diagnostics.beam_diagnostics import rms_moments  # noqa: E402
from struphy.initial.perturbations import PiecewiseLinearPotential  # noqa: E402
from struphy.models import IonOpticsElectrostatic  # noqa: E402
from struphy.pic.ion_beams import LossTag, PlaneSource  # noqa: E402


@dataclass(frozen=True)
class Beam:
    """Injected beam in SI units (the lens is ``slit_immersion_lens.DESIGN``)."""

    half_width: float = 4.5e-3  # m, uniform in y at the launch plane
    angular_spread: float = 40e-3  # rad, rms of v_y / v_x
    markers_per_ns: float = 80.0  # emission rate


BEAM = Beam()
TAGS = ("electrode 1", "electrode 2", "outlet", "inlet")


def loss_tags(design=DESIGN, units=UNITS):
    """Boundary parts for loss accounting: plates split at the gap centre, and the two ends."""
    gap_center = design.lens_center / units.length
    return (
        LossTag("electrode 1", axis=1, coordinate=0, interval=(-np.inf, gap_center)),
        LossTag("electrode 2", axis=1, coordinate=0, interval=(gap_center, np.inf)),
        LossTag("outlet", axis=0, side=1),
        LossTag("inlet", axis=0, side=0),
    )


def source(beam=BEAM, design=DESIGN, units=UNITS, seed=11, current=1.0):
    """Beam source; ``current`` is normalized (``units.current`` per length unit of slit in z)."""
    length = design.length / units.length
    h = design.half_gap / units.length
    w = beam.half_width / units.length
    speed = float(units.speed(design.beam_energy_eV))
    rate = beam.markers_per_ns * 1e9 * units.time
    return PlaneSource(
        rate=rate,
        current=current,  # zero-current runs: 1, so charges are fractions of the injection
        axis=0,
        eta_plane=design.x_start / units.length / length,
        eta_ranges=(((h - w) / (2 * h), (h + w) / (2 * h)), (0.0, 1.0)),
        velocity=(speed, 0.0, 0.0),
        velocity_spread=(0.0, beam.angular_spread * speed, 0.0),
        seed=seed,
    )


def build_simulation(
    output_dir,
    num_elements=(160, 20),
    degree=3,
    dt=0.04,
    end_time=45.0,
    beam=BEAM,
    current=1.0,
    space_charge=False,
    name="slit_lens_injection",
):
    """Serial zero-current injection run (normalized units: mm, kV)."""
    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("Run this example on one MPI rank.")
    length = DESIGN.length / UNITS.length
    h = DESIGN.half_gap / UNITS.length
    lens = analytic_lens()
    src = source(beam, current=current)
    capacity = int(3 * src.rate * end_time)  # generous: markers live for about 20 time units

    model = IonOpticsElectrostatic(
        base_units=UNITS.base_units(),
        charge_number=DESIGN.charge_number,
        mass_number=DESIGN.mass_number,
        electrode_faces=((False, False), (True, True), (False, False)),
        source=src,
        space_charge=space_charge,
        loss_tags=loss_tags(),
        keep_loss_records=("outlet",),
    )
    model.em_fields.phi.add_perturbation(PiecewiseLinearPotential(lens.plate_nodes, (lens.v1, lens.v2), coordinate=0))
    model.ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    # One placeholder marker sizes the marker array; InjectMarkers removes it at start.
    model.ions.set_markers(
        loading_params=LoadingParameters(Np=1, specific_markers=((0.5, 0.5, 0.5, 0.0, 0.0, 0.0),)),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(bc=("remove", "remove", "periodic")),
        # injected markers are analysed from the final state and the ledger (n_markers=0 is not supported)
        saving_params=SavingParameters(n_markers=1),
        bufsize=float(capacity),
    )
    for prop in (model.propagators.inject, model.propagators.push_v, model.propagators.push_eta):
        prop.options = prop.Options()

    return Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(output_dir), sim_folder=name),
        time_opts=Time(dt=dt, Tend=end_time, split_algo="Strang"),
        domain=domains.Cuboid(l1=0.0, r1=length, l2=-h, r2=h),
        grid=grids.TensorProductGrid(num_elements=(*num_elements, 1)),
        derham_opts=DerhamOptions(degree=(degree, degree, 1), bcs=(("free", "free"),) * 3),
    )


def load_charges(sim):
    """Time series of the ledger scalars (normalized charge)."""
    with h5py.File(Path(sim.env.path_out) / "data" / "data_proc0.hdf5") as f:
        time = f["time/value"][:]
        scalars = {key: f["scalar"][key][:] for key in f["scalar"]}
    return time, scalars


def steady_window(time, t_start):
    return time >= t_start


def steady_currents(time, charges, t_start):
    """Mean current to each boundary part over ``t >= t_start``, as a fraction of the injected current."""
    window = steady_window(time, t_start)

    def gain(key):
        series = charges[key][window]
        return series[-1] - series[0]

    injected = gain("injected_charge")
    return {name: gain(f"lost_charge_{name}") / injected for name in TAGS} | {"live": gain("live_charge") / injected}


def binomial_error(fraction, n):
    """Standard error of a fraction estimated from ``n`` independent markers."""
    return np.sqrt(max(fraction * (1 - fraction), 0.0) / n)


def reference_fractions(n_rays=4000, beam=BEAM, seed=5, dt=0.01, t_end=40.0):
    """Fractions of rays ending on each boundary part, traced in the exact field.

    Vectorized fixed-step RK4 (dt = 0.01, i.e. about 0.05 mm per step); a ray stops at
    the first step that takes it out of the domain. The exact field is singular at the
    gap corners, which rules out a common adaptive step for all rays.
    """
    lens = analytic_lens(n_quad=16)  # same fractions as n_quad=64, 3x faster
    length = DESIGN.length / UNITS.length
    h = DESIGN.half_gap / UNITS.length
    eta, v = source(beam, seed=seed).sample(n_rays)
    state = np.column_stack([eta[:, 0] * length, -h + 2 * h * eta[:, 1], v[:, 0], v[:, 1]])
    active = np.ones(n_rays, dtype=bool)

    def rhs(s):
        ex, ey = lens.efield(s[:, 0], np.clip(s[:, 1], -h, h))
        return np.column_stack([s[:, 2], s[:, 3], ex, ey])

    for _ in range(int(round(t_end / dt))):
        s = state[active]
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * dt * k1)
        k3 = rhs(s + 0.5 * dt * k2)
        k4 = rhs(s + dt * k3)
        state[active] = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x, y = state[:, 0], state[:, 1]
        active &= (np.abs(y) < h) & (x > 0) & (x < length)
        if not active.any():
            break
    if active.any():
        raise RuntimeError("Some reference rays never leave the domain; increase t_end.")

    x_exit, y_exit = state[:, 0], state[:, 1]
    on_plate = np.abs(y_exit) >= h
    counts = {
        "electrode 1": np.sum(on_plate & (x_exit <= lens.xc)),
        "electrode 2": np.sum(on_plate & (x_exit > lens.xc)),
        "outlet": np.sum(~on_plate & (x_exit >= length)),
        "inlet": np.sum(~on_plate & (x_exit <= 0)),
    }
    return {name: count / n_rays for name, count in counts.items()}, n_rays


def plot_results(sim, output, t_start=25.0):
    model = sim.model
    model.update_ledger()
    time, charges = load_charges(sim)
    lost_total = sum(charges[f"lost_charge_{name}"] for name in (*TAGS, "other"))
    balance = charges["injected_charge"] - charges["live_charge"] - lost_total
    currents = steady_currents(time, charges, t_start)
    window = steady_window(time, t_start)
    n_window = np.ptp(charges["injected_charge"][window]) * model.propagators.inject.source.rate
    reference, n_ref = reference_fractions()
    print(f"Charge balance residual (max |injected - live - lost|): {np.max(np.abs(balance)):.2e}")
    print(f"Change of in-flight charge over the steady window: {currents['live']:+.4f} of the injected charge")
    for name in TAGS:
        sigma = binomial_error(currents[name], n_window)
        sigma_ref = binomial_error(reference[name], n_ref)
        print(
            f"Current to {name:12s}: {currents[name]:.4f} ± {sigma:.4f} "
            f"(reference {reference[name]:.4f} ± {sigma_ref:.4f})"
        )

    length = DESIGN.length / UNITS.length
    h = DESIGN.half_gap / UNITS.length
    lens = analytic_lens()
    particles = model.ions.var.particles
    live = particles.markers[particles.valid_mks]
    x_live, y_live = live[:, 0] * length, -h + 2 * h * live[:, 1]
    x_grid, y_grid = np.linspace(0, length, 321), np.linspace(-h, h, 81)
    phi = np.asarray(model.em_fields.phi.spline(x_grid / length, (y_grid + h) / (2 * h), np.array([0.5])))[:, :, 0]

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.25, 1))
    ax_beam = fig.add_subplot(grid[0, :])
    ax_charge, ax_losses, ax_exit = (fig.add_subplot(grid[1, i]) for i in range(3))
    fig.suptitle(
        f"Continuous injection, slit immersion lens: {DESIGN.beam_energy_eV / 1e3:g} keV protons, "
        f"±{BEAM.half_width * 1e3:g} mm, {BEAM.angular_spread * 1e3:g} mrad rms, zero current"
    )

    ax_beam.contour(
        x_grid,
        y_grid,
        UNITS.volts(phi.T) / 1e3,
        levels=np.arange(-9.5, 0, 1.0),
        colors="0.75",
        linewidths=0.6,
        zorder=1,
    )
    counts, x_edges, y_edges = np.histogram2d(x_live, y_live, bins=(160, 40), range=((0, length), (-h, h)))
    # Current density per unit injected current (charge per area / flight time) is proportional to counts.
    density = ax_beam.pcolormesh(x_edges, y_edges, counts.T, cmap="inferno", shading="flat", zorder=0)
    fig.colorbar(density, ax=ax_beam, label="Markers per 0.5 mm × 0.25 mm bin", pad=0.01)
    for xs, colour, label in [
        ((0.0, lens.plate_nodes[0]), "#9c9c9c", f"Electrode 1 ({DESIGN.v1 / 1e3:g} kV)"),
        ((lens.plate_nodes[1], length), "#f28e2b", f"Electrode 2 ({DESIGN.v2 / 1e3:g} kV)"),
    ]:
        for y_plate in (-h, h):
            ax_beam.plot(
                xs,
                (y_plate, y_plate),
                color=colour,
                linewidth=6,
                solid_capstyle="butt",
                label=label if y_plate < 0 else None,
            )
    ax_beam.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title=f"Steady-state beam at t = {time[-1] * UNITS.time * 1e9:.0f} ns: {len(live)} live markers (grey: equipotentials every kV)",
        xlim=(0, length),
        ylim=(-h * 1.08, h * 1.08),
    )
    ax_beam.legend(loc="upper left", fontsize=8)

    t_ns = time * UNITS.time * 1e9
    ax_charge.plot(t_ns, charges["injected_charge"], label="Injected")
    ax_charge.plot(t_ns, charges["live_charge"], label="In flight")
    for name in TAGS[:3]:
        ax_charge.plot(t_ns, charges[f"lost_charge_{name}"], label=name.capitalize())
    ax_charge.plot(t_ns, 1e6 * balance, "k:", label="Balance residual × 10⁶")
    ax_charge.axvspan(t_start * UNITS.time * 1e9, t_ns[-1], color="0.9", zorder=-1)
    ax_charge.set(
        xlabel="t (ns)",
        ylabel="Charge (injected current × time unit)",
        title="Charge bookkeeping (shaded: steady window)",
    )
    ax_charge.legend(fontsize=8)

    names = TAGS[:3]
    positions = np.arange(len(names))
    ax_losses.bar(
        positions - 0.2,
        [100 * currents[n] for n in names],
        0.4,
        yerr=[100 * binomial_error(currents[n], n_window) for n in names],
        label=f"Struphy (steady window, {n_window:.0f} markers)",
    )
    ax_losses.bar(
        positions + 0.2,
        [100 * reference[n] for n in names],
        0.4,
        yerr=[100 * np.sqrt(reference[n] * (1 - reference[n]) / n_ref) for n in names],
        label=f"Exact-field Monte Carlo ({n_ref} rays)",
    )
    ax_losses.set_xticks(positions, [n.capitalize() for n in names])
    ax_losses.set(ylabel="% of injected current", title="Where the current goes", yscale="log", ylim=(0.3, 150))
    ax_losses.legend(fontsize=8)

    exit_rows = model.ledger.records["outlet"]
    exit_rows = exit_rows[exit_rows[:, 0] > 0]
    moments = rms_moments(exit_rows[:, 1], exit_rows[:, 4] / exit_rows[:, 3])
    ax_exit.hist2d(exit_rows[:, 1], 1e3 * exit_rows[:, 4] / exit_rows[:, 3], bins=60, cmap="viridis", cmin=1)
    ax_exit.set(
        xlabel="y (mm)",
        ylabel="y' (mrad)",
        title=f"Phase space at the outlet: rms ε = {1e3 * moments['emittance']:.1f} mm·mrad",
    )

    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")
    return balance, currents, n_window, reference, n_ref


def main():
    sim = build_simulation(Path(__file__).parent / "output")
    sim.run()
    plot_results(sim, Path(__file__).with_name("slit_lens_injection.png"))


if __name__ == "__main__":
    main()
