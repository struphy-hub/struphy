"""Zero-current proton beam through the segmented single-patch aperture.

Run from the repository root with::

    MPLBACKEND=Agg .venv/bin/python examples/IonOpticsElectrostatic/segmented_electrode_channel/vacuum_aperture_beam.py

The central aperture has a 2.25 mm half-height, narrower than the 5 mm domain
half-height. This is a vacuum, test-particle calculation: it solves Laplace's
equation from the segment voltages once and does not deposit space charge.
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


LENGTH_MM = 80.0
DOMAIN_RADIUS_MM = 5.0
APERTURE_RADIUS_MM = 2.25
UNITS = IonOpticsUnits(length=1e-3, voltage=1e3, mass_number=1.0, charge_number=1)


def build_domain():
    """Build in normalized millimetres; voltages are normalized kilovolts."""
    return SegmentedElectrodeChannel(
        length=LENGTH_MM,
        width=2.0,
        lower_profile=(
            (0.0, 28.0, 32.0, 48.0, 52.0, LENGTH_MM),
            (
                -DOMAIN_RADIUS_MM,
                -DOMAIN_RADIUS_MM,
                -APERTURE_RADIUS_MM,
                -APERTURE_RADIUS_MM,
                -DOMAIN_RADIUS_MM,
                -DOMAIN_RADIUS_MM,
            ),
        ),
        upper_profile=(
            (0.0, 28.0, 32.0, 48.0, 52.0, LENGTH_MM),
            (
                DOMAIN_RADIUS_MM,
                DOMAIN_RADIUS_MM,
                APERTURE_RADIUS_MM,
                APERTURE_RADIUS_MM,
                DOMAIN_RADIUS_MM,
                DOMAIN_RADIUS_MM,
            ),
        ),
        segments=(
            ElectrodeSegment("lower", 0.0, 28.0, 0.0, "entrance"),
            ElectrodeSegment("lower", 32.0, 48.0, UNITS.potential(-5e3), "aperture"),
            ElectrodeSegment("lower", 52.0, LENGTH_MM, 0.0, "exit"),
            ElectrodeSegment("upper", 0.0, 28.0, 0.0, "entrance"),
            ElectrodeSegment("upper", 32.0, 48.0, UNITS.potential(-5e3), "aperture"),
            ElectrodeSegment("upper", 52.0, LENGTH_MM, 0.0, "exit"),
        ),
        num_elements=(80, 20),
        degree=(3, 3),
    )


def build_simulation(output_dir, dt=0.02, end_time=30.0):
    """Build a serial 5 keV proton-ray simulation in the aperture domain."""
    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("Run this small trajectory example on one MPI rank.")
    domain = build_domain()
    y0 = np.linspace(-1.5, 1.5, 31)
    x0 = 1.0
    eta_y = (y0 + DOMAIN_RADIUS_MM) / (2.0 * DOMAIN_RADIUS_MM)
    speed = float(UNITS.speed(5e3))
    markers = tuple((x0 / LENGTH_MM, float(y), 0.5, speed, 0.0, 0.0) for y in eta_y)

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
        env=EnvironmentOptions(out_folders=str(output_dir), sim_folder="vacuum_aperture_beam"),
        time_opts=Time(dt=dt, Tend=end_time, split_algo="Strang"),
        domain=domain,
        grid=grids.TensorProductGrid(num_elements=(80, 20, 1)),
        derham_opts=DerhamOptions(
            degree=(3, 3, 1),
            bcs=(("free", "free"), ("free", "free"), ("free", "free")),
        ),
    )


def plot_results(sim, output):
    """Plot solved potential and physical trajectories saved by the simulation."""
    with h5py.File(Path(sim.env.path_out) / "data" / "data_proc0.hdf5") as data:
        history = data["kinetic/ions/markers"][:]

    eta = history[:, :, :3]
    xyz = sim.domain(eta.reshape(-1, 3), remove_outside=False).reshape(3, *eta.shape[:2])
    x, y = np.asarray(xyz[0]), np.asarray(xyz[1])
    invalid = ~np.all((eta >= 0.0) & (eta <= 1.0), axis=-1) | ~np.all(np.isfinite(history[:, :, :6]), axis=-1)
    x[invalid], y[invalid] = np.nan, np.nan

    eta_x = np.linspace(0.0, 1.0, 301)
    eta_y = np.linspace(0.0, 1.0, 101)
    xyz_grid = sim.domain(eta_x, eta_y, 0.5, squeeze_out=True)
    phi = np.asarray(sim.model.em_fields.phi.spline(eta_x, eta_y, np.array([0.5])))[:, :, 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    field = axes[0].contourf(xyz_grid[0], xyz_grid[1], UNITS.volts(phi) / 1e3, levels=40, cmap="viridis")
    axes[0].contour(xyz_grid[0], xyz_grid[1], phi, levels=16, colors="white", linewidths=0.35, alpha=0.6)
    fig.colorbar(field, ax=axes[0], label="Potential (kV)")
    axes[0].set(title="Vacuum potential", xlabel="x (mm)", ylabel="y (mm)")

    axes[1].plot(x, y, color="tab:red", linewidth=0.8)
    axes[1].plot(xyz_grid[0, :, 0], xyz_grid[1, :, 0], color="black", linewidth=1.5)
    axes[1].plot(xyz_grid[0, :, -1], xyz_grid[1, :, -1], color="black", linewidth=1.5)
    axes[1].set(title="5 keV proton trajectories", xlabel="x (mm)", ylabel="y (mm)")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Wrote {output}")


def main():
    sim = build_simulation(Path(__file__).parent / "output")
    sim.run()
    plot_results(sim, Path(__file__).with_name("vacuum_aperture_beam.png"))


if __name__ == "__main__":
    main()
