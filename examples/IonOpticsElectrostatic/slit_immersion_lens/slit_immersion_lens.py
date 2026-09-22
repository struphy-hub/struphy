"""Zero-current proton beam through a two-electrode slit immersion lens.

Run from the repository root with::

    python examples/IonOpticsElectrostatic/slit_immersion_lens/slit_immersion_lens.py

The design case is specified in SI units (``DESIGN``) and converted with
:class:`~struphy.physics.ion_optics_units.IonOpticsUnits`. The electrode plates at
``y = ±h`` are Dirichlet boundaries whose potential jumps from ``V1`` to ``V2``
across a gap (linear-gap model); the vacuum potential is solved by the model's
Laplace solve, and a parallel beam is traced through it. Trajectories are
compared with a high-order ODE integration in the exact analytic field.
Writes ``slit_immersion_lens.png`` next to this file.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt

from analytic import SlitImmersionLens, reference_trajectories
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
from struphy.diagnostics.beam_diagnostics import plane_crossings, rms_moments
from struphy.initial.perturbations import PiecewiseLinearPotential
from struphy.models import IonOpticsElectrostatic
from struphy.physics.ion_optics_units import IonOpticsUnits


@dataclass(frozen=True)
class Design:
    """Lens and beam parameters in SI units."""

    mass_number: float = 1.0  # protons
    charge_number: int = 1
    beam_energy_eV: float = 5e3  # kinetic energy at the potential of electrode 1
    beam_half_width: float = 3e-3  # m, parallel beam
    n_rays: int = 31
    x_start: float = 1e-3  # m, launch plane
    half_gap: float = 5e-3  # m, plates at y = ±half_gap
    gap: float = 2e-3  # m, electrode gap along x
    lens_center: float = 35e-3  # m
    length: float = 80e-3  # m, domain length along the beam
    v1: float = 0.0  # V, upstream electrode
    v2: float = -10e3  # V, downstream electrode


DESIGN = Design()
UNITS = IonOpticsUnits(
    length=1e-3,
    voltage=1e3,
    mass_number=DESIGN.mass_number,
    charge_number=DESIGN.charge_number,
)


def analytic_lens(design=DESIGN, units=UNITS, n_quad=64):
    """Exact field of the design in normalized units."""
    return SlitImmersionLens(
        h=design.half_gap / units.length,
        g=design.gap / units.length,
        xc=design.lens_center / units.length,
        v1=float(units.potential(design.v1)),
        v2=float(units.potential(design.v2)),
        n_quad=n_quad,
    )


def initial_rays(design=DESIGN, units=UNITS):
    """Parallel beam as rows ``(x, y, vx, vy)`` in normalized units."""
    y0 = np.linspace(-design.beam_half_width, design.beam_half_width, design.n_rays) / units.length
    x0 = np.full_like(y0, design.x_start / units.length)
    speed = float(units.speed(design.beam_energy_eV))
    return np.column_stack([x0, y0, np.full_like(y0, speed), np.zeros_like(y0)])


def build_simulation(output_dir, num_elements=(320, 40), degree=3, dt=0.02, end_time=20.0, design=DESIGN, units=UNITS):
    """Build the serial zero-current lens simulation (normalized units: mm, kV)."""
    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("Run this small trajectory example on one MPI rank.")
    length = design.length / units.length
    h = design.half_gap / units.length
    lens = analytic_lens(design, units)
    rays = initial_rays(design, units)
    # Markers are loaded in logical coordinates (eta1, eta2, eta3) and physical velocities.
    markers = tuple((x / length, (y + h) / (2 * h), 0.5, vx, vy, 0.0) for x, y, vx, vy in rays)

    model = IonOpticsElectrostatic(
        base_units=units.base_units(),
        charge_number=design.charge_number,
        mass_number=design.mass_number,
        electrode_faces=((False, False), (True, True), (False, False)),
    )
    model.em_fields.phi.save_data = True
    model.ions.var.save_data = True
    model.em_fields.phi.add_perturbation(PiecewiseLinearPotential(lens.plate_nodes, (lens.v1, lens.v2), coordinate=0))
    model.ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    model.ions.set_markers(
        loading_params=LoadingParameters(Np=len(markers), specific_markers=markers, seed=7),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(bc=("remove", "remove", "periodic")),
        saving_params=SavingParameters(n_markers=len(markers)),
    )
    model.propagators.push_v.options = model.propagators.push_v.Options()
    model.propagators.push_eta.options = model.propagators.push_eta.Options()

    return Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(output_dir), sim_folder="slit_immersion_lens"),
        time_opts=Time(dt=dt, Tend=end_time, split_algo="Strang"),
        domain=domains.Cuboid(l1=0.0, r1=length, l2=-h, r2=h),
        grid=grids.TensorProductGrid(num_elements=(*num_elements, 1)),
        derham_opts=DerhamOptions(
            degree=(degree, degree, 1),
            bcs=(("free", "free"), ("free", "free"), ("free", "free")),
        ),
    )


def load_trajectories(sim):
    """Saved physical positions and velocities, shape ``(n_times, n_rays, 4)``.

    Rows of removed markers are set to NaN from the first step outside the domain.
    """
    with h5py.File(Path(sim.env.path_out) / "data" / "data_proc0.hdf5") as f:
        time = f["time/value"][:]
        history = f["kinetic/ions/markers"][:]
    domain = sim.domain
    x = domain.params["l1"] + (domain.params["r1"] - domain.params["l1"]) * history[:, :, 0]
    y = domain.params["l2"] + (domain.params["r2"] - domain.params["l2"]) * history[:, :, 1]
    states = np.stack([x, y, history[:, :, 3], history[:, :, 4]], axis=-1)
    lost = np.logical_or.accumulate(~_inside(history), axis=0)
    states[lost] = np.nan
    return time, states


def _inside(history):
    eta = history[:, :, :3]
    return np.all((eta >= 0.0) & (eta <= 1.0), axis=-1) & np.all(np.isfinite(history[:, :, :6]), axis=-1)


def axis_crossings(states):
    """x position where each ray first crosses the axis y = 0 (NaN for rays launched on the axis)."""
    crossings = plane_crossings(states, 0.0, axis=1)[:, 0]
    crossings[np.abs(states[0, :, 1]) < 1e-12] = np.nan
    return crossings


def compare_with_reference(time, states, design=DESIGN, units=UNITS):
    """Maximum position error against the exact-field reference, over the rays still in the domain."""
    reference = reference_trajectories(analytic_lens(design, units), initial_rays(design, units), time[-1])
    exact = np.array([reference(t) for t in time])
    inside = np.isfinite(states[..., 0])
    return np.max(np.abs(states[..., :2] - exact[..., :2])[inside]), exact


def beam_at_planes(states, planes):
    """Crossing states and rms moments of the beam at each plane ``x = const``."""
    crossings = [plane_crossings(states, x) for x in planes]
    moments = [rms_moments(c[:, 1], c[:, 3] / c[:, 2]) for c in crossings]
    return crossings, moments


def plot_results(sim, output, design=DESIGN, units=UNITS):
    time, states = load_trajectories(sim)
    error, exact = compare_with_reference(time, states, design, units)
    lens = analytic_lens(design, units)
    crossings = axis_crossings(states)
    exact_crossings = axis_crossings(exact)
    paraxial_focus = np.nanmax(exact_crossings)
    print(f"Maximum trajectory deviation from exact-field reference: {error:.3e} mm")
    print(f"Paraxial focus (innermost rays): {np.nanmax(crossings):.3f} mm (reference {paraxial_focus:.3f} mm)")

    length = design.length / units.length
    h = design.half_gap / units.length
    x_grid = np.linspace(0.0, length, 401)
    y_grid = np.linspace(-h, h, 101)
    phi = np.asarray(sim.model.em_fields.phi.spline(x_grid / length, (y_grid + h) / (2 * h), np.array([0.5])))[:, :, 0]

    envelope_planes = np.linspace(design.x_start / units.length + 0.5, length - 0.5, 150)
    _, envelope = beam_at_planes(states, envelope_planes)
    _, exact_envelope = beam_at_planes(exact, envelope_planes)
    phase_planes = (design.x_start / units.length + 1.0, lens.xc, paraxial_focus, length - 1.0)
    phase_crossings, phase_moments = beam_at_planes(states, phase_planes)

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=(1.3, 1))
    ax_beam = fig.add_subplot(grid[0, :])
    ax_axis, ax_focus, ax_env, ax_phase = (fig.add_subplot(grid[1, i]) for i in range(4))
    fig.suptitle(
        f"Slit immersion lens: {design.beam_energy_eV / 1e3:g} keV protons, "
        f"V₁ = {design.v1 / 1e3:g} kV, V₂ = {design.v2 / 1e3:g} kV, zero current — "
        f"max. deviation from exact-field rays {error * units.length * 1e6:.1f} µm"
    )

    contours = ax_beam.contourf(x_grid, y_grid, units.volts(phi.T) / 1e3, levels=40, cmap="viridis")
    ax_beam.contour(x_grid, y_grid, phi.T, levels=20, colors="white", linewidths=0.4, alpha=0.5)
    fig.colorbar(contours, ax=ax_beam, label="Potential (kV)", pad=0.01)
    for xs, colour, label in [
        ((0.0, lens.plate_nodes[0]), "#d9d9d9", f"Electrode 1 ({design.v1 / 1e3:g} kV)"),
        ((lens.plate_nodes[1], length), "#f28e2b", f"Electrode 2 ({design.v2 / 1e3:g} kV)"),
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
    ax_beam.plot(states[:, :, 0], states[:, :, 1], color="#e15759", linewidth=0.9)
    for x_plane in phase_planes:
        ax_beam.axvline(x_plane, color="white", linestyle=":", linewidth=1)
    ax_beam.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title="Ion beam trajectories in the solved vacuum potential (dotted: phase-space planes)",
        xlim=(0, length),
        ylim=(-h * 1.08, h * 1.08),
    )
    ax_beam.legend(loc="upper left", fontsize=8, framealpha=0.8)

    ax_axis.plot(x_grid, units.volts(phi[:, len(y_grid) // 2]) / 1e3, label="Struphy (Laplace solve)")
    ax_axis.plot(x_grid[::12], units.volts(lens.phi(x_grid[::12], 0.0)) / 1e3, "o", fillstyle="none", label="Analytic")
    ax_axis.set(xlabel="x (mm)", ylabel="Potential on axis (kV)", title="Axial potential")
    ax_axis.legend()

    y0 = initial_rays(design, units)[:, 1]
    ax_focus.plot(np.abs(y0), crossings, "o", label="Struphy")
    ax_focus.plot(np.abs(y0), exact_crossings, "k+", markersize=9, label="Exact-field reference")
    ax_focus.set(xlabel="Initial |y| (mm)", ylabel="Axis crossing x (mm)", title="Focus and spherical aberration")
    ax_focus.legend()

    ax_env.plot(envelope_planes, [m["size"] for m in envelope], label="RMS size (Struphy)")
    ax_env.plot(envelope_planes[::8], [m["size"] for m in exact_envelope[::8]], "k+", label="Reference")
    ax_env.set(xlabel="x (mm)", ylabel="RMS beam size (mm)", title="Beam envelope")
    ax_emit = ax_env.twinx()
    ax_emit.plot(envelope_planes, [1e3 * m["emittance"] for m in envelope], color="#59a14f", linestyle="--")
    ax_emit.set_ylabel("RMS trace-space emittance (mm·mrad), dashed", color="#59a14f")
    ax_env.legend(loc="upper center", fontsize=8)

    for x_plane, crossing, moments in zip(phase_planes, phase_crossings, phase_moments):
        ax_phase.scatter(
            crossing[:, 1],
            1e3 * crossing[:, 3] / crossing[:, 2],
            s=12,
            label=f"x = {x_plane:.1f} mm, ε = {1e3 * moments['emittance']:.2f} mm·mrad",
        )
    ax_phase.set(xlabel="y (mm)", ylabel="y' (mrad)", title="Transverse phase space at planes")
    ax_phase.legend(fontsize=7)

    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")
    return error, crossings, exact_crossings


def main():
    sim = build_simulation(Path(__file__).parent / "output")
    sim.run()
    plot_results(sim, Path(__file__).with_name("slit_immersion_lens.png"))


if __name__ == "__main__":
    main()
