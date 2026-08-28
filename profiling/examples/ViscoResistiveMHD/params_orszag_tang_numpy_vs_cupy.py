"""Self-contained parameters for the Orszag--Tang NumPy-vs-CuPy profiling case.

Copied from struphy.models.tests.Test_Full_MHD.OrszagTang (the regularized
ideal-MHD Orszag--Tang verification), trimmed to the run/validate path this
profiling case actually uses (no plotting, no pytest entry point), so this
file has no import dependency on that test module. Keep the two in sync if
OrszagTang.py's simulation setup or diagnostics change.

cunumpy picks numpy or cupy during import, based on the ARRAY_BACKEND
environment variable. Everything that transitively imports cunumpy is
therefore deferred to inside main(), after ARRAY_BACKEND is set from
--backend; only stdlib and numpy are imported at module level.
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import time
from pathlib import Path

import numpy as np

name = "Orszag--Tang: NumPy vs CuPy"
description = "Fixed 96x96x1, five-step, one-rank backend comparison."

ALPHA_DIVDIV = 1.0e-2
WITH_REGULARIZATION = True

GAMMA = 5.0 / 3.0

DT = 1.0e-3
NUM_STEPS = 5
T_END = NUM_STEPS * DT

NUM_ELEMENTS = (96, 96, 1)
SPLINE_DEGREE = (3, 3, 1)

LENGTH_X = 2.0 * np.pi
LENGTH_Y = 2.0 * np.pi
LENGTH_Z = 1.0

# Set from --id in main(): OUTPUT_DIRECTORY = Path.cwd() / SIMULATION_FOLDER_NAME.
SIMULATION_FOLDER_NAME = "sim_00"
OUTPUT_DIRECTORY = Path.cwd() / SIMULATION_FOLDER_NAME


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------


def _make_initial_state_class(base_class):
    r"""Build `OrszagTangInitialState`, deriving from `base_class`.

    A factory rather than a module-level class definition because the base
    class (`CartesianFluidEquilibriumWithB`) is only importable once
    ARRAY_BACKEND is set (see main()); `class ... (base_class)` therefore
    can't run until then.

    Smooth two-dimensional Orszag--Tang initial state. The fields on the
    periodic square are

    .. math::

        \rho &= \rho_0, \\
        p &= p_0, \\
        \mathbf u &=
        u_0(-\sin y,\,\sin x,\,0), \\
        \mathbf B &=
        B_0(-\sin y,\,\sin(2x),\,0).

    Both vector fields are initially divergence-free.
    """

    class OrszagTangInitialState(base_class):
        def __init__(
            self,
            rho0: float = 1.0,
            p0: float = 1.0,
            velocity_amplitude: float = 1.0,
            magnetic_amplitude: float = 1.0,
        ):
            self.params = copy.deepcopy(locals())

        def n_xyz(self, x, y, z):
            """Constant physical density."""
            return self.params["rho0"] + 0.0 * x

        def p_xyz(self, x, y, z):
            """Constant physical pressure."""
            return self.params["p0"] + 0.0 * x

        def u_xyz(self, x, y, z):
            """Cartesian velocity."""
            amplitude = self.params["velocity_amplitude"]

            return (
                -amplitude * xp.sin(y),
                amplitude * xp.sin(x),
                0.0 * z,
            )

        def b_xyz(self, x, y, z):
            """Cartesian magnetic field."""
            amplitude = self.params["magnetic_amplitude"]

            return (
                -amplitude * xp.sin(y),
                amplitude * xp.sin(2.0 * x),
                0.0 * z,
            )

        def gradB_xyz(self, x, y, z):
            """Cartesian gradient of the magnetic-field magnitude."""
            amplitude = self.params["magnetic_amplitude"]

            sin_y = xp.sin(y)
            sin_2x = xp.sin(2.0 * x)

            square = sin_y**2 + sin_2x**2
            denominator = xp.sqrt(square)

            # Avoid division by zero at magnetic nulls.
            safe_denominator = xp.where(
                denominator > 1.0e-14,
                denominator,
                1.0,
            )

            grad_x = amplitude * 2.0 * sin_2x * xp.cos(2.0 * x) / safe_denominator
            grad_y = amplitude * sin_y * xp.cos(y) / safe_denominator

            grad_x = xp.where(denominator > 1.0e-14, grad_x, 0.0)
            grad_y = xp.where(denominator > 1.0e-14, grad_y, 0.0)

            return (
                grad_x,
                grad_y,
                0.0 * z,
            )

    return OrszagTangInitialState


# ---------------------------------------------------------------------------
# Simulation construction
# ---------------------------------------------------------------------------


def create_simulation():
    """Construct the Simulation object without running it."""
    rho0 = GAMMA**2
    p0 = GAMMA

    initial_state = OrszagTangInitialState(
        rho0=float(rho0),
        p0=float(p0),
        velocity_amplitude=1.0,
        magnetic_amplitude=1.0,
    )

    # Verify the entropy convention used by s3_monoatomic.
    specific_entropy0 = np.log(p0 / ((GAMMA - 1.0) * rho0**GAMMA))
    entropy_density0 = rho0 * specific_entropy0

    reference_entropy_density0 = GAMMA**2 * np.log(GAMMA / ((GAMMA - 1.0) * GAMMA ** (2.0 * GAMMA)))

    if not np.isclose(entropy_density0, reference_entropy_density0):
        raise RuntimeError(
            "The initial entropy convention is inconsistent.\n"
            f"Computed entropy density : {entropy_density0:.16e}\n"
            f"Reference entropy density: {reference_entropy_density0:.16e}"
        )

    model = ViscoResistiveMHD(
        with_viscosity=False,
        with_resistivity=False,
        with_regularization=WITH_REGULARIZATION,
        divdiv_alpha=ALPHA_DIVDIV,
    )

    # Full-f model: initialize all total fields.
    model.mhd.density.add_background(FieldsBackground(type="FluidEquilibrium", variable="n3"))
    model.mhd.entropy.add_background(FieldsBackground(type="FluidEquilibrium", variable="s3_monoatomic"))
    model.mhd.velocity.add_background(FieldsBackground(type="FluidEquilibrium", variable="uv"))
    model.em_fields.b_field.add_background(FieldsBackground(type="FluidEquilibrium", variable="b2"))

    # Replacing an Options object resets omitted values to their defaults.
    # Therefore all regularization options are supplied explicitly.
    model.propagators.variat_dens.options = model.propagators.variat_dens.Options(
        model="full",
        gamma=GAMMA,
        with_regularization=WITH_REGULARIZATION,
        alpha_divdiv=ALPHA_DIVDIV,
    )

    model.propagators.variat_mom.options = model.propagators.variat_mom.Options(
        with_regularization=WITH_REGULARIZATION,
        alpha_divdiv=ALPHA_DIVDIV,
    )

    model.propagators.variat_ent.options = model.propagators.variat_ent.Options(
        model="full",
        gamma=GAMMA,
        with_regularization=WITH_REGULARIZATION,
        alpha_divdiv=ALPHA_DIVDIV,
    )

    model.propagators.variat_mag.options = model.propagators.variat_mag.Options(
        model="full",
        with_regularization=WITH_REGULARIZATION,
        alpha_divdiv=ALPHA_DIVDIV,
    )

    # Fail early if one of the options was accidentally reset.
    regularized_options = (
        model.propagators.variat_dens.options,
        model.propagators.variat_mom.options,
        model.propagators.variat_ent.options,
        model.propagators.variat_mag.options,
    )

    for options in regularized_options:
        if not options.with_regularization:
            raise RuntimeError("A propagator has regularization disabled.")

        if not np.isclose(options.alpha_divdiv, ALPHA_DIVDIV):
            raise RuntimeError(f"A propagator has the wrong div-div coefficient: {options.alpha_divdiv!r}")

    env = EnvironmentOptions(
        out_folders=str(Path(OUTPUT_DIRECTORY).parent),
        sim_folder=Path(OUTPUT_DIRECTORY).name,
    )

    domain = domains.Cuboid(
        r1=float(LENGTH_X),
        r2=float(LENGTH_Y),
        r3=float(LENGTH_Z),
    )

    grid = grids.TensorProductGrid(num_elements=NUM_ELEMENTS)
    derham_opts = DerhamOptions(degree=SPLINE_DEGREE)
    time_opts = Time(dt=DT, Tend=T_END)

    simulation = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
        equil=initial_state,
        params_path=os.path.abspath(__file__),
    )

    return simulation, env


# ---------------------------------------------------------------------------
# Scalar diagnostics
# ---------------------------------------------------------------------------


def find_scalar_hdf5(output_path) -> str:
    """Find the HDF5 file containing scalar diagnostics."""
    data_directory = Path(output_path) / "data"

    candidates = sorted(glob.glob(str(data_directory / "*.hdf5")))

    if not candidates:
        raise FileNotFoundError(f"No HDF5 files found in {data_directory!s}.")

    for candidate in candidates:
        with h5py.File(candidate, "r") as file:
            if "time/value" in file and "scalar/en_tot" in file:
                return candidate

    raise FileNotFoundError("No HDF5 file containing both time and scalar diagnostics was found.")


def check_mhd_scalar_diagnostics(output_path, energy_tolerance: float | None = None) -> dict:
    """Load and check scalar diagnostics from an existing simulation."""
    hdf5_path = find_scalar_hdf5(output_path)

    print(f"Reading scalar diagnostics from:\n{hdf5_path}")

    with h5py.File(hdf5_path, "r") as file:
        time_history = np.asarray(file["time/value"]).reshape(-1)
        total_energy = np.asarray(file["scalar/en_tot"]).reshape(-1)
        kinetic_energy = np.asarray(file["scalar/en_U"]).reshape(-1)
        magnetic_energy = np.asarray(file["scalar/en_mag"]).reshape(-1)
        thermodynamic_energy = np.asarray(file["scalar/en_thermo"]).reshape(-1)
        total_mass = np.asarray(file["scalar/dens_tot"]).reshape(-1)
        total_entropy = np.asarray(file["scalar/entr_tot"]).reshape(-1)
        div_b = np.asarray(file["scalar/tot_div_B"]).reshape(-1)

    histories = {
        "time": time_history,
        "total energy": total_energy,
        "kinetic energy": kinetic_energy,
        "magnetic energy": magnetic_energy,
        "thermodynamic energy": thermodynamic_energy,
        "total mass": total_mass,
        "total entropy": total_entropy,
        "div(B)": div_b,
    }

    expected_size = time_history.size

    for name_, values in histories.items():
        if values.size != expected_size:
            raise RuntimeError(f"{name_} has {values.size} entries, but the time history has {expected_size} entries.")

        if not np.all(np.isfinite(values)):
            bad_indices = np.flatnonzero(~np.isfinite(values))
            first_bad = int(bad_indices[0])

            raise RuntimeError(
                f"Non-finite values found in {name_}.\n"
                f"First bad index: {first_bad}\n"
                f"Time: {time_history[first_bad]!r}\n"
                f"Value: {values[first_bad]!r}"
            )

    if not np.isclose(time_history[0], 0.0):
        raise RuntimeError(f"The first saved time is {time_history[0]}, not zero.")

    # Check the total-energy definition.
    energy_sum = kinetic_energy + magnetic_energy + thermodynamic_energy
    component_sum_difference = total_energy - energy_sum
    maximum_component_sum_error = float(np.max(np.abs(component_sum_difference)))
    component_sum_scale = max(abs(total_energy[0]), 1.0)
    relative_component_sum_error = maximum_component_sum_error / component_sum_scale

    if not np.allclose(total_energy, energy_sum, rtol=1.0e-12, atol=1.0e-12):
        raise RuntimeError(
            "The saved total energy is inconsistent with en_U + en_mag + en_thermo.\n"
            f"Maximum absolute difference: {maximum_component_sum_error:.12e}\n"
            f"Relative difference: {relative_component_sum_error:.12e}"
        )

    def relative_drift(values):
        scale = max(abs(values[0]), 1.0e-30)
        return float(np.max(np.abs(values - values[0])) / scale)

    energy_drift = relative_drift(total_energy)
    mass_drift = relative_drift(total_mass)
    entropy_drift = relative_drift(total_entropy)
    maximum_div_b = float(np.max(np.abs(div_b)))

    if mass_drift >= 1.0e-10:
        raise RuntimeError(f"Relative mass drift is too large: {mass_drift:.12e}")

    if entropy_drift >= 1.0e-10:
        raise RuntimeError(f"Relative total-entropy drift is too large: {entropy_drift:.12e}")

    if maximum_div_b >= 1.0e-10:
        raise RuntimeError(f"The magnetic-divergence diagnostic is too large: {maximum_div_b:.12e}")

    if energy_tolerance is not None and energy_drift >= energy_tolerance:
        raise RuntimeError(f"Relative energy drift {energy_drift:.6e} exceeds the tolerance {energy_tolerance:.6e}.")

    return {
        "hdf5_path": hdf5_path,
        "time": time_history,
        "en_tot": total_energy,
        "en_U": kinetic_energy,
        "en_mag": magnetic_energy,
        "en_thermo": thermodynamic_energy,
        "dens_tot": total_mass,
        "entr_tot": total_entropy,
        "tot_div_B": div_b,
        "energy_drift": energy_drift,
        "mass_drift": mass_drift,
        "entropy_drift": entropy_drift,
        "maximum_div_b": maximum_div_b,
        "component_sum_error": maximum_component_sum_error,
        "relative_component_sum_error": relative_component_sum_error,
    }


# ---------------------------------------------------------------------------
# Output-directory handling and run control
# ---------------------------------------------------------------------------


def prepare_output_directory(*, run: bool, overwrite: bool):
    """Validate or remove output before constructing Simulation.

    Simulation construction may create and initialize the output directory.
    Therefore an old output directory must be removed before calling
    create_simulation(), never afterward.
    """
    import shutil

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_path = Path(OUTPUT_DIRECTORY)

    output_exists = None

    if rank == 0:
        output_exists = output_path.exists()

    output_exists = comm.bcast(output_exists, root=0)

    if run:
        if output_exists and not overwrite:
            raise FileExistsError(
                "The output directory already exists:\n"
                f"{output_path}\n\n"
                "Use --overwrite to delete it and run again."
            )

        if rank == 0 and output_exists:
            print("Removing existing output because --overwrite was supplied:")
            print(output_path)
            shutil.rmtree(output_path)

        # Only create the parent here. Simulation/EnvironmentOptions will
        # initialize the actual simulation output directory.
        if rank == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if not output_exists:
            raise FileNotFoundError(f"The existing simulation output directory was not found:\n{output_path}")

    comm.Barrier()


def run_new_simulation(simulation, env):
    """Run a new simulation and post-process it."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_path = Path(env.path_out)

    # Simulation construction should normally create this directory.
    # This is a final safeguard for versions that do not.
    if rank == 0:
        output_path.mkdir(parents=True, exist_ok=True)

    comm.Barrier()

    if rank == 0:
        print("\nRUNNING A NEW SIMULATION")
        print("========================")
        print(f"Output path : {output_path}")
        print(f"Initial rho : {GAMMA**2}")
        print(f"Initial p   : {GAMMA}")
        print(f"Div-div     : {ALPHA_DIVDIV}")
        print(f"dt          : {DT}")
        print(f"T_end       : {T_END}")

    simulation.run(profiling_activated=True)

    comm.Barrier()

    if rank == 0:
        simulation.pproc(create_vtk=False)

    comm.Barrier()


def ensure_post_processing(simulation, output_path):
    """Ensure post-processed plotting data exist, generating them if not."""
    output_path = Path(output_path)
    post_processing_path = output_path / "post_processing"

    if post_processing_path.is_dir():
        return

    simulation.pproc(create_vtk=False)

    if not post_processing_path.is_dir():
        raise RuntimeError(
            f"Post-processing completed without creating the expected directory:\n{post_processing_path}"
        )


# ---------------------------------------------------------------------------
# Loading and checking post-processed fields
# ---------------------------------------------------------------------------


def extract_scalar_time_data(data):
    """Convert scalar plotting data to a time-first NumPy array."""
    times = sorted(data, key=float)
    values = np.stack([xp.to_numpy(data[time][0]) for time in times], axis=0)
    return times, values


def extract_vector_time_data(data):
    """Convert vector plotting data to ``(nt, 3, nx, ny, nz)``."""
    times = sorted(data, key=float)
    values = np.stack(
        [np.stack([xp.to_numpy(component) for component in data[time]], axis=0) for time in times],
        axis=0,
    )
    return times, values


def check_matching_times(reference_times, other_times, variable_name: str):
    """Check that two plotting histories use the same saved times."""
    if len(reference_times) != len(other_times):
        raise RuntimeError(
            f"Time-history length mismatch for {variable_name}: {len(reference_times)} != {len(other_times)}"
        )

    for index, (reference, other) in enumerate(zip(reference_times, other_times)):
        if not np.isclose(float(reference), float(other)):
            raise RuntimeError(f"Time mismatch for {variable_name} at index {index}: {reference!r} != {other!r}")


def diagnose_thermodynamic_fields(density: np.ndarray, entropy: np.ndarray, gamma: float) -> dict:
    r"""Diagnose pressure without directly overflowing ``exp``.

    Pressure is defined by :math:`p = (\gamma-1)\rho^\gamma \exp(s/\rho)`, so
    :math:`\log p = \log(\gamma-1) + \gamma\log\rho + s/\rho`. The
    logarithmic expression is only evaluated where density is finite and
    strictly positive and entropy is finite.
    """
    density = np.asarray(density, dtype=np.float64)
    entropy = np.asarray(entropy, dtype=np.float64)

    if density.shape != entropy.shape:
        raise RuntimeError(f"Density and entropy have different shapes: {density.shape} != {entropy.shape}")

    finite_density = np.isfinite(density)
    finite_entropy = np.isfinite(entropy)
    positive_density = density > 0.0

    valid_input = finite_density & finite_entropy & positive_density

    log_pressure = np.full(density.shape, np.nan, dtype=np.float64)

    # Invalid operations are prevented with the valid_input mask.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        log_pressure[valid_input] = (
            np.log(gamma - 1.0) + gamma * np.log(density[valid_input]) + entropy[valid_input] / density[valid_input]
        )

    float64_log_max = float(np.log(np.finfo(np.float64).max))
    smallest_positive = np.nextafter(np.float64(0.0), np.float64(1.0))
    float64_log_min = float(np.log(smallest_positive))

    pressure_overflow = np.isfinite(log_pressure) & (log_pressure > float64_log_max)
    pressure_underflow = np.isfinite(log_pressure) & (log_pressure < float64_log_min)
    nonfinite_log_pressure = valid_input & ~np.isfinite(log_pressure)

    physically_admissible = (
        not np.any(~finite_density)
        and not np.any(finite_density & ~positive_density)
        and not np.any(~finite_entropy)
        and not np.any(nonfinite_log_pressure)
        and not np.any(pressure_overflow)
    )

    return {
        "log_pressure": log_pressure,
        "pressure_overflow": pressure_overflow,
        "pressure_underflow": pressure_underflow,
        "physically_admissible": physically_admissible,
    }


def load_and_check_fields(simulation) -> dict:
    """Load plotting data and diagnose reconstructed MHD fields."""
    simulation.load_plotting_data()

    density_data = simulation.spline_values.mhd.density_log.data
    entropy_data = simulation.spline_values.mhd.entropy_log.data
    velocity_data = simulation.spline_values.mhd.velocity_log.data
    magnetic_data = simulation.spline_values.em_fields.b_field_log.data

    times, density_3form = extract_scalar_time_data(density_data)
    entropy_times, entropy_3form = extract_scalar_time_data(entropy_data)
    velocity_times, velocity = extract_vector_time_data(velocity_data)
    magnetic_times, magnetic = extract_vector_time_data(magnetic_data)

    check_matching_times(times, entropy_times, "entropy")
    check_matching_times(times, velocity_times, "velocity")
    check_matching_times(times, magnetic_times, "magnetic field")

    jacobian = xp.to_numpy(simulation.domain.jacobian_det(*simulation.grids_log)).astype(np.float64, copy=False)

    if not np.all(np.isfinite(jacobian)):
        raise RuntimeError("The geometry Jacobian contains non-finite values.")

    if np.any(jacobian <= 0.0):
        raise RuntimeError("The geometry Jacobian is not strictly positive.")

    # L2 variables are stored as logical 3-forms.
    density = density_3form / jacobian[None, ...]
    entropy = entropy_3form / jacobian[None, ...]

    if not np.all(np.isfinite(velocity)):
        raise RuntimeError("Velocity contains non-finite values.")

    if not np.all(np.isfinite(magnetic)):
        raise RuntimeError("Magnetic field contains non-finite values.")

    thermo = diagnose_thermodynamic_fields(density=density, entropy=entropy, gamma=GAMMA)

    return {
        "times": times,
        "density": density,
        "entropy": entropy,
        "velocity": velocity,
        "magnetic": magnetic,
        "thermo_diagnostics": thermo,
    }


def execute() -> tuple[dict, dict]:
    """Run a new case, then validate scalar diagnostics and reconstructed fields."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # This must happen before create_simulation(). Constructing Simulation
    # may create files and directories required by simulation.run().
    prepare_output_directory(run=True, overwrite=True)

    simulation, env = create_simulation()
    output_path = Path(env.path_out)

    run_new_simulation(simulation=simulation, env=env)

    comm.Barrier()

    scalar_diagnostics = None
    field_data = None

    if rank == 0:
        scalar_diagnostics = check_mhd_scalar_diagnostics(output_path, energy_tolerance=1.0e-3)
        ensure_post_processing(simulation=simulation, output_path=output_path)
        field_data = load_and_check_fields(simulation)

        if not field_data["thermo_diagnostics"]["physically_admissible"]:
            raise RuntimeError(
                "The scalar invariants are well conserved, but the saved fields contain a "
                "non-positive density, non-finite state, or unrepresentable pressure."
            )

    comm.Barrier()

    return scalar_diagnostics, field_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    args = parser.parse_args()

    # cunumpy chooses its implementation during import, so it (and everything
    # that transitively imports it) must be imported only after this is set.
    os.environ["ARRAY_BACKEND"] = args.backend

    global xp, h5py, MPI, DerhamOptions, EnvironmentOptions, FieldsBackground, Simulation, Time
    global domains, grids, ViscoResistiveMHD, OrszagTangInitialState
    global SIMULATION_FOLDER_NAME, OUTPUT_DIRECTORY

    import cunumpy as xp
    import h5py
    from feectools.ddm.mpi import mpi as MPI

    from struphy import (
        DerhamOptions,
        EnvironmentOptions,
        FieldsBackground,
        Simulation,
        Time,
        domains,
        grids,
        set_logging_level,
    )
    from struphy.fields_background.base import CartesianFluidEquilibriumWithB
    from struphy.models import ViscoResistiveMHD

    set_logging_level()
    OrszagTangInitialState = _make_initial_state_class(CartesianFluidEquilibriumWithB)

    SIMULATION_FOLDER_NAME = f"sim_{args.id:02d}"
    OUTPUT_DIRECTORY = Path.cwd() / SIMULATION_FOLDER_NAME

    print(
        "BENCHMARK_CONFIG"
        f" backend={args.backend} ranks=1 nel={NUM_ELEMENTS[0]}"
        f" steps={NUM_STEPS} dt={DT}",
    )
    started = time.perf_counter()
    execute()
    print(f"BENCHMARK_WALL_SECONDS {time.perf_counter() - started:.9f}")


if __name__ == "__main__":
    main()
