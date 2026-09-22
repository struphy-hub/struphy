"""Two-stage electrostatic acceleration through two mapped slit apertures.

Run from the repository root with::

    MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/segmented_electrode_channel/double_aperture_accelerator.py

Protons enter at 2 keV. The first aperture is at -5 kV and the second/exit
electrode is at -10 kV, giving a nominal 10 keV electrostatic energy gain.
This remains a vacuum test-particle calculation; there is no space charge.
"""

from pathlib import Path

import h5py
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
    grids,
    maxwellians,
)
from struphy.geometry.domains import ElectrodeSegment, SegmentedElectrodeChannel
from struphy.models import IonOpticsElectrostatic
from struphy.physics.ion_optics_units import IonOpticsUnits

NUM_PARTICLES = 100
LENGTH_MM = 80.0
DOMAIN_RADIUS_MM = 5.0
APERTURE_RADIUS_MM = 2.25
FIRST_APERTURE_START_MM = 20.0
FIRST_APERTURE_END_MM = 25.0
SECOND_APERTURE_START_MM = 50.0
SECOND_APERTURE_END_MM = 70.0
TRANSITION_LENGTH_MM = 2.0
UNITS = IonOpticsUnits(length=1e-3, voltage=1e3, mass_number=1.0, charge_number=1)


def build_domain(num_elements=(96, 20), degree=(3, 3)):
    """Two 5 mm-wide slit apertures, stepped from 0 to -5 to -10 kV."""
    potential_aperture1 = -5e3
    potential_aperture2 = -10e3
    return SegmentedElectrodeChannel(
        length=LENGTH_MM,
        width=2.0,
        lower_profile=(
            (
                0.0,
                FIRST_APERTURE_START_MM - TRANSITION_LENGTH_MM,
                FIRST_APERTURE_START_MM + TRANSITION_LENGTH_MM,
                FIRST_APERTURE_END_MM - TRANSITION_LENGTH_MM,
                FIRST_APERTURE_END_MM + TRANSITION_LENGTH_MM,
                SECOND_APERTURE_START_MM - TRANSITION_LENGTH_MM,
                SECOND_APERTURE_START_MM + TRANSITION_LENGTH_MM,
                SECOND_APERTURE_END_MM - TRANSITION_LENGTH_MM,
                SECOND_APERTURE_END_MM + TRANSITION_LENGTH_MM,
                LENGTH_MM,
            ),
            (
                -DOMAIN_RADIUS_MM,
                -DOMAIN_RADIUS_MM,
                -APERTURE_RADIUS_MM,
                -APERTURE_RADIUS_MM,
                -DOMAIN_RADIUS_MM,
                -DOMAIN_RADIUS_MM,
                -APERTURE_RADIUS_MM,
                -APERTURE_RADIUS_MM,
                -DOMAIN_RADIUS_MM,
                -DOMAIN_RADIUS_MM,
            ),
        ),
        upper_profile=(
            (
                0.0,
                FIRST_APERTURE_START_MM - TRANSITION_LENGTH_MM,
                FIRST_APERTURE_START_MM + TRANSITION_LENGTH_MM,
                FIRST_APERTURE_END_MM - TRANSITION_LENGTH_MM,
                FIRST_APERTURE_END_MM + TRANSITION_LENGTH_MM,
                SECOND_APERTURE_START_MM - TRANSITION_LENGTH_MM,
                SECOND_APERTURE_START_MM + TRANSITION_LENGTH_MM,
                SECOND_APERTURE_END_MM - TRANSITION_LENGTH_MM,
                SECOND_APERTURE_END_MM + TRANSITION_LENGTH_MM,
                LENGTH_MM,
            ),
            (
                DOMAIN_RADIUS_MM,
                DOMAIN_RADIUS_MM,
                APERTURE_RADIUS_MM,
                APERTURE_RADIUS_MM,
                DOMAIN_RADIUS_MM,
                DOMAIN_RADIUS_MM,
                APERTURE_RADIUS_MM,
                APERTURE_RADIUS_MM,
                DOMAIN_RADIUS_MM,
                DOMAIN_RADIUS_MM,
            ),
        ),
        segments=(
            ElectrodeSegment("lower", 0.0, FIRST_APERTURE_START_MM - TRANSITION_LENGTH_MM, 0.0, "source"),
            ElectrodeSegment(
                "lower",
                FIRST_APERTURE_START_MM,
                FIRST_APERTURE_END_MM,
                UNITS.potential(potential_aperture1),
                "aperture_1",
            ),
            ElectrodeSegment(
                "lower",
                FIRST_APERTURE_END_MM + TRANSITION_LENGTH_MM,
                SECOND_APERTURE_START_MM - TRANSITION_LENGTH_MM,
                UNITS.potential(potential_aperture1),
                "interstage",
            ),
            ElectrodeSegment(
                "lower",
                SECOND_APERTURE_START_MM,
                SECOND_APERTURE_END_MM,
                UNITS.potential(potential_aperture2),
                "aperture_2",
            ),
            ElectrodeSegment(
                "lower",
                SECOND_APERTURE_END_MM + TRANSITION_LENGTH_MM,
                LENGTH_MM,
                UNITS.potential(potential_aperture2),
                "exit",
            ),
            ElectrodeSegment("upper", 0.0, FIRST_APERTURE_START_MM - TRANSITION_LENGTH_MM, 0.0, "source"),
            ElectrodeSegment(
                "upper",
                FIRST_APERTURE_START_MM,
                FIRST_APERTURE_END_MM,
                UNITS.potential(potential_aperture1),
                "aperture_1",
            ),
            ElectrodeSegment(
                "upper",
                FIRST_APERTURE_END_MM + TRANSITION_LENGTH_MM,
                SECOND_APERTURE_START_MM - TRANSITION_LENGTH_MM,
                UNITS.potential(potential_aperture1),
                "interstage",
            ),
            ElectrodeSegment(
                "upper",
                SECOND_APERTURE_START_MM,
                SECOND_APERTURE_END_MM,
                UNITS.potential(potential_aperture2),
                "aperture_2",
            ),
            ElectrodeSegment(
                "upper",
                SECOND_APERTURE_END_MM + TRANSITION_LENGTH_MM,
                LENGTH_MM,
                UNITS.potential(potential_aperture2),
                "exit",
            ),
        ),
        num_elements=num_elements,
        degree=degree,
    )


def build_simulation(output_dir, dt=0.015, end_time=26.0, num_elements=(96, 20), degree=3):
    """Build a serial, zero-current 2 keV proton accelerator simulation."""
    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("Run this small trajectory example on one MPI rank.")
    domain = build_domain(num_elements, (degree, degree))
    y0 = np.linspace(-5.0, 5.0, NUM_PARTICLES)
    speed = float(UNITS.speed(2e3))
    markers = tuple(
        (1.0 / LENGTH_MM, float((y + DOMAIN_RADIUS_MM) / (2 * DOMAIN_RADIUS_MM)), 0.5, speed, 0.0, 0.0) for y in y0
    )

    model = IonOpticsElectrostatic(
        base_units=UNITS.base_units(),
        charge_number=1,
        mass_number=1.0,
        electrode_segments=domain.segments,
        electrode_length=domain.length,
    )
    model.em_fields.phi.save_data = True
    model.ions.var.save_data = True
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
        env=EnvironmentOptions(out_folders=str(output_dir), sim_folder="double_aperture_accelerator"),
        time_opts=Time(dt=dt, Tend=end_time, split_algo="Strang"),
        domain=domain,
        grid=grids.TensorProductGrid(num_elements=(*num_elements, 1)),
        derham_opts=DerhamOptions(
            degree=(degree, degree, 1),
            bcs=(("free", "free"), ("free", "free"), ("free", "free")),
        ),
    )


def plot_results(sim, output):
    """Plot potential, trajectories, and central-ray electrostatic energy gain."""
    with h5py.File(Path(sim.env.path_out) / "data" / "data_proc0.hdf5") as data:
        history = data["kinetic/ions/markers"][:]
    eta = history[:, :, :3]
    xyz = sim.domain(eta.reshape(-1, 3), remove_outside=False).reshape(3, *eta.shape[:2])
    x, y = np.asarray(xyz[0]), np.asarray(xyz[1])
    invalid = ~np.all((eta >= 0.0) & (eta <= 1.0), axis=-1) | ~np.all(np.isfinite(history[:, :, :6]), axis=-1)
    x[invalid], y[invalid] = np.nan, np.nan

    eta_x, eta_y = np.linspace(0.0, 1.0, 321), np.linspace(0.0, 1.0, 101)
    grid_xyz = sim.domain(eta_x, eta_y, 0.5, squeeze_out=True)
    phi = np.asarray(sim.model.em_fields.phi.spline(eta_x, eta_y, np.array([0.5])))[:, :, 0]
    central = history[:, len(history[0]) // 2, 3:6]
    kinetic_eV = UNITS.kinetic_energy_eV(np.linalg.norm(central, axis=1))
    kinetic_eV[invalid[:, len(history[0]) // 2]] = np.nan

    fig, axes = plt.subplots(3, 1, figsize=(16, 16), constrained_layout=True)
    potential = axes[0].contourf(grid_xyz[0], grid_xyz[1], UNITS.volts(phi) / 1e3, levels=40, cmap="viridis")
    axes[0].contour(grid_xyz[0], grid_xyz[1], phi, levels=18, colors="white", linewidths=0.35, alpha=0.6)
    fig.colorbar(potential, ax=axes[0], label="Potential (kV)")
    axes[0].set(title="Two-stage vacuum potential", xlabel="x (mm)", ylabel="y (mm)")

    axes[1].plot(x, y, color="tab:red", linewidth=0.8)
    axes[1].plot(grid_xyz[0, :, 0], grid_xyz[1, :, 0], color="black", linewidth=1.5)
    axes[1].plot(grid_xyz[0, :, -1], grid_xyz[1, :, -1], color="black", linewidth=1.5)
    axes[1].set(title="2 keV proton trajectories", xlabel="x (mm)", ylabel="y (mm)")

    axes[2].plot(kinetic_eV / 1e3)
    axes[2].axhline(12.0, color="black", linestyle="--", label="2 keV + 10 kV nominal")
    axes[2].set(title="Central-ray kinetic energy", xlabel="Saved time index", ylabel="Energy (keV)")
    axes[2].legend()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Exit central-ray energy: {kinetic_eV[np.isfinite(kinetic_eV)][-1] / 1e3:.3f} keV")
    print(f"Wrote {output}")


def plot_grid(sim, output):
    """Save the mapped tensor-product grid used by the field solve."""
    sim.domain.show(grid_info=[96, 20, 1], show_control_pts=True, save_dir=output)
    print(f"Wrote {output}")


def main():
    sim = build_simulation(Path(__file__).parent / "output")
    sim.run()
    plot_results(sim, Path(__file__).with_name("double_aperture_accelerator.png"))
    plot_grid(sim, Path(__file__).with_name("double_aperture_accelerator_grid.png"))


if __name__ == "__main__":
    main()
