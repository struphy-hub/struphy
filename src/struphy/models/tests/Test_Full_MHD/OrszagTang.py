"""
Regularized ideal-MHD Orszag--Tang verification.

Default behavior
----------------
Load an existing simulation without rerunning it:

    python test_orszag_tang_regularized.py

Load and plot:

    python test_orszag_tang_regularized.py --plot

If raw output exists but post-processing data do not:

    python test_orszag_tang_regularized.py --post-process

Run a new simulation:

    python test_orszag_tang_regularized.py --run

Overwrite an existing simulation and rerun:

    python test_orszag_tang_regularized.py --run --overwrite

Notes
-----
Pressure is reconstructed from the entropy-density formulation as

    p = (gamma - 1) * rho**gamma * exp(s / rho).

The script evaluates log(p) first. This allows pressure overflow to be
diagnosed without immediately generating infinity.
"""

from __future__ import annotations

import argparse
import copy
import glob
import logging
import os
import shutil
from pathlib import Path

import cunumpy as xp
import h5py
import matplotlib.pyplot as plt
import numpy as np
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


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

set_logging_level()
logger = logging.getLogger("struphy")

ALPHA_DIVDIV = 1.0e-2
WITH_REGULARIZATION = True

GAMMA = 5.0 / 3.0

DT = 1.0e-3
T_END = 2.0

NUM_ELEMENTS = (32, 32, 1)
SPLINE_DEGREE = (3, 3, 1)

LENGTH_X = 2.0 * np.pi
LENGTH_Y = 2.0 * np.pi
LENGTH_Z = 1.0

# This reproduces the output layout used in the original file:
#
#   <script directory>/verification_tests_full/
#       ViscoResistiveMHD/orszag_tang_regularized
#
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
TEST_FOLDER = SCRIPT_DIRECTORY / "verification_tests_full"
MODEL_OUTPUT_FOLDER = TEST_FOLDER / "ViscoResistiveMHD"
SIMULATION_FOLDER_NAME = "orszag_tang_regularized"
OUTPUT_DIRECTORY = MODEL_OUTPUT_FOLDER / SIMULATION_FOLDER_NAME


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------


class OrszagTangInitialState(CartesianFluidEquilibriumWithB):
    r"""Smooth two-dimensional Orszag--Tang initial state.

    The fields on the periodic square are

    .. math::

        \rho &= \rho_0, \\
        p &= p_0, \\
        \mathbf u &=
        u_0(-\sin y,\,\sin x,\,0), \\
        \mathbf B &=
        B_0(-\sin y,\,\sin(2x),\,0).

    Both vector fields are initially divergence-free.
    """

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

        grad_x = (
            amplitude
            * 2.0
            * sin_2x
            * xp.cos(2.0 * x)
            / safe_denominator
        )

        grad_y = (
            amplitude
            * sin_y
            * xp.cos(y)
            / safe_denominator
        )

        grad_x = xp.where(
            denominator > 1.0e-14,
            grad_x,
            0.0,
        )

        grad_y = xp.where(
            denominator > 1.0e-14,
            grad_y,
            0.0,
        )

        return (
            grad_x,
            grad_y,
            0.0 * z,
        )


# ---------------------------------------------------------------------------
# Simulation construction
# ---------------------------------------------------------------------------


def create_simulation() -> tuple[Simulation, EnvironmentOptions]:
    """Construct the Simulation object without running it.

    Constructing this object is needed both for a new run and for loading
    existing post-processed plotting data. This function does not call
    ``simulation.run()`` and does not delete existing output.
    """
    rho0 = GAMMA**2
    p0 = GAMMA

    initial_state = OrszagTangInitialState(
        rho0=float(rho0),
        p0=float(p0),
        velocity_amplitude=1.0,
        magnetic_amplitude=1.0,
    )

    # Verify the entropy convention used by s3_monoatomic.
    specific_entropy0 = np.log(
        p0 / ((GAMMA - 1.0) * rho0**GAMMA),
    )
    entropy_density0 = rho0 * specific_entropy0

    reference_entropy_density0 = GAMMA**2 * np.log(
        GAMMA
        / (
            (GAMMA - 1.0)
            * GAMMA ** (2.0 * GAMMA)
        ),
    )

    if not np.isclose(
        entropy_density0,
        reference_entropy_density0,
    ):
        raise RuntimeError(
            "The initial entropy convention is inconsistent.\n"
            f"Computed entropy density : {entropy_density0:.16e}\n"
            f"Reference entropy density: "
            f"{reference_entropy_density0:.16e}"
        )

    model = ViscoResistiveMHD(
        with_viscosity=False,
        with_resistivity=False,
        with_regularization=WITH_REGULARIZATION,
        divdiv_alpha=ALPHA_DIVDIV,
    )

    # Full-f model: initialize all total fields.
    model.mhd.density.add_background(
        FieldsBackground(
            type="FluidEquilibrium",
            variable="n3",
        ),
    )

    model.mhd.entropy.add_background(
        FieldsBackground(
            type="FluidEquilibrium",
            variable="s3_monoatomic",
        ),
    )

    model.mhd.velocity.add_background(
        FieldsBackground(
            type="FluidEquilibrium",
            variable="uv",
        ),
    )

    model.em_fields.b_field.add_background(
        FieldsBackground(
            type="FluidEquilibrium",
            variable="b2",
        ),
    )

    # Replacing an Options object resets omitted values to their defaults.
    # Therefore all regularization options are supplied explicitly.
    model.propagators.variat_dens.options = (
        model.propagators.variat_dens.Options(
            model="full",
            gamma=GAMMA,
            with_regularization=WITH_REGULARIZATION,
            alpha_divdiv=ALPHA_DIVDIV,
        )
    )

    model.propagators.variat_mom.options = (
        model.propagators.variat_mom.Options(
            with_regularization=WITH_REGULARIZATION,
            alpha_divdiv=ALPHA_DIVDIV,
        )
    )

    model.propagators.variat_ent.options = (
        model.propagators.variat_ent.Options(
            model="full",
            gamma=GAMMA,
            with_regularization=WITH_REGULARIZATION,
            alpha_divdiv=ALPHA_DIVDIV,
        )
    )

    model.propagators.variat_mag.options = (
        model.propagators.variat_mag.Options(
            model="full",
            with_regularization=WITH_REGULARIZATION,
            alpha_divdiv=ALPHA_DIVDIV,
        )
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
            raise RuntimeError(
                "A propagator has regularization disabled."
            )

        if not np.isclose(
            options.alpha_divdiv,
            ALPHA_DIVDIV,
        ):
            raise RuntimeError(
                "A propagator has the wrong div-div coefficient: "
                f"{options.alpha_divdiv!r}"
            )

    env = EnvironmentOptions(
        out_folders=str(MODEL_OUTPUT_FOLDER),
        sim_folder=SIMULATION_FOLDER_NAME,
    )

    domain = domains.Cuboid(
        r1=float(LENGTH_X),
        r2=float(LENGTH_Y),
        r3=float(LENGTH_Z),
    )

    grid = grids.TensorProductGrid(
        num_elements=NUM_ELEMENTS,
    )

    derham_opts = DerhamOptions(
        degree=SPLINE_DEGREE,
    )

    time_opts = Time(
        dt=DT,
        Tend=T_END,
    )

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


def find_scalar_hdf5(output_path: str | Path) -> str:
    """Find the HDF5 file containing scalar diagnostics."""
    data_directory = Path(output_path) / "data"

    candidates = sorted(
        glob.glob(str(data_directory / "*.hdf5"))
    )

    if not candidates:
        raise FileNotFoundError(
            f"No HDF5 files found in {data_directory!s}."
        )

    for candidate in candidates:
        with h5py.File(candidate, "r") as file:
            if (
                "time/value" in file
                and "scalar/en_tot" in file
            ):
                return candidate

    raise FileNotFoundError(
        "No HDF5 file containing both time and scalar "
        "diagnostics was found."
    )


def check_mhd_scalar_diagnostics(
    output_path: str | Path,
    energy_tolerance: float | None = None,
) -> dict:
    """Load and check scalar diagnostics from an existing simulation."""
    hdf5_path = find_scalar_hdf5(output_path)

    print(f"Reading scalar diagnostics from:\n{hdf5_path}")

    with h5py.File(hdf5_path, "r") as file:
        time_history = np.asarray(
            file["time/value"]
        ).reshape(-1)

        total_energy = np.asarray(
            file["scalar/en_tot"]
        ).reshape(-1)

        kinetic_energy = np.asarray(
            file["scalar/en_U"]
        ).reshape(-1)

        magnetic_energy = np.asarray(
            file["scalar/en_mag"]
        ).reshape(-1)

        thermodynamic_energy = np.asarray(
            file["scalar/en_thermo"]
        ).reshape(-1)

        total_mass = np.asarray(
            file["scalar/dens_tot"]
        ).reshape(-1)

        total_entropy = np.asarray(
            file["scalar/entr_tot"]
        ).reshape(-1)

        div_b = np.asarray(
            file["scalar/tot_div_B"]
        ).reshape(-1)

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

    for name, values in histories.items():
        if values.size != expected_size:
            raise RuntimeError(
                f"{name} has {values.size} entries, but the time "
                f"history has {expected_size} entries."
            )

        if not np.all(np.isfinite(values)):
            bad_indices = np.flatnonzero(~np.isfinite(values))
            first_bad = int(bad_indices[0])

            raise RuntimeError(
                f"Non-finite values found in {name}.\n"
                f"First bad index: {first_bad}\n"
                f"Time: {time_history[first_bad]!r}\n"
                f"Value: {values[first_bad]!r}"
            )

    if not np.isclose(time_history[0], 0.0):
        raise RuntimeError(
            f"The first saved time is {time_history[0]}, not zero."
        )

    # Check the total-energy definition.
    energy_sum = (
        kinetic_energy
        + magnetic_energy
        + thermodynamic_energy
    )

    component_sum_difference = total_energy - energy_sum

    maximum_component_sum_error = float(
        np.max(np.abs(component_sum_difference))
    )

    component_sum_scale = max(
        abs(total_energy[0]),
        1.0,
    )

    relative_component_sum_error = (
        maximum_component_sum_error / component_sum_scale
    )

    if not np.allclose(
        total_energy,
        energy_sum,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "The saved total energy is inconsistent with "
            "en_U + en_mag + en_thermo.\n"
            f"Maximum absolute difference: "
            f"{maximum_component_sum_error:.12e}\n"
            f"Relative difference: "
            f"{relative_component_sum_error:.12e}"
        )

    def relative_drift(values):
        scale = max(abs(values[0]), 1.0e-30)
        return float(
            np.max(np.abs(values - values[0])) / scale
        )

    def final_relative_change(values):
        scale = max(abs(values[0]), 1.0e-30)
        return float((values[-1] - values[0]) / scale)

    def relative_variation(values):
        scale = max(abs(values[0]), 1.0e-30)
        return float(
            (np.max(values) - np.min(values)) / scale
        )

    energy_drift = relative_drift(total_energy)
    mass_drift = relative_drift(total_mass)
    entropy_drift = relative_drift(total_entropy)

    final_energy_change = final_relative_change(total_energy)
    final_mass_change = final_relative_change(total_mass)
    final_entropy_change = final_relative_change(total_entropy)

    maximum_div_b = float(np.max(np.abs(div_b)))

    kinetic_variation = relative_variation(kinetic_energy)
    magnetic_variation = relative_variation(magnetic_energy)
    thermodynamic_variation = relative_variation(
        thermodynamic_energy
    )

    print("\nREGULARIZED IDEAL-MHD SCALAR DIAGNOSTICS")
    print("========================================")
    print(
        f"Div-div alpha                  : "
        f"{ALPHA_DIVDIV:.12e}"
    )
    print(
        f"Saved states                  : {expected_size}"
    )
    print(
        f"First saved time              : "
        f"{time_history[0]:.12e}"
    )
    print(
        f"Last saved time               : "
        f"{time_history[-1]:.12e}"
    )

    print("\nDiscrete regularized energy:")
    print(
        f"Initial total energy          : "
        f"{total_energy[0]:.16e}"
    )
    print(
        f"Final total energy            : "
        f"{total_energy[-1]:.16e}"
    )
    print(
        f"Maximum relative energy drift : "
        f"{energy_drift:.12e}"
    )
    print(
        f"Signed final energy change    : "
        f"{final_energy_change:.12e}"
    )

    print("\nEnergy-definition consistency:")
    print(
        f"Maximum component-sum error   : "
        f"{maximum_component_sum_error:.12e}"
    )
    print(
        f"Relative component-sum error  : "
        f"{relative_component_sum_error:.12e}"
    )

    print("\nConserved quantities:")
    print(
        f"Maximum relative mass drift   : "
        f"{mass_drift:.12e}"
    )
    print(
        f"Signed final mass change      : "
        f"{final_mass_change:.12e}"
    )
    print(
        f"Maximum relative entropy drift: "
        f"{entropy_drift:.12e}"
    )
    print(
        f"Signed final entropy change   : "
        f"{final_entropy_change:.12e}"
    )
    print(
        f"Maximum div(B) diagnostic     : "
        f"{maximum_div_b:.12e}"
    )

    print("\nEnergy exchange:")
    print(
        f"Relative kinetic variation    : "
        f"{kinetic_variation:.12e}"
    )
    print(
        f"Relative magnetic variation   : "
        f"{magnetic_variation:.12e}"
    )
    print(
        f"Thermodynamic variation       : "
        f"{thermodynamic_variation:.12e}"
    )

    if mass_drift >= 1.0e-10:
        raise RuntimeError(
            f"Relative mass drift is too large: "
            f"{mass_drift:.12e}"
        )

    if entropy_drift >= 1.0e-10:
        raise RuntimeError(
            f"Relative total-entropy drift is too large: "
            f"{entropy_drift:.12e}"
        )

    if maximum_div_b >= 1.0e-10:
        raise RuntimeError(
            "The magnetic-divergence diagnostic is too large: "
            f"{maximum_div_b:.12e}"
        )

    if (
        energy_tolerance is not None
        and energy_drift >= energy_tolerance
    ):
        raise RuntimeError(
            f"Relative energy drift {energy_drift:.6e} exceeds "
            f"the tolerance {energy_tolerance:.6e}."
        )

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
        "final_energy_change": final_energy_change,
        "mass_drift": mass_drift,
        "entropy_drift": entropy_drift,
        "maximum_div_b": maximum_div_b,
        "component_sum_error": maximum_component_sum_error,
        "relative_component_sum_error": (
            relative_component_sum_error
        ),
        "kinetic_variation": kinetic_variation,
        "magnetic_variation": magnetic_variation,
        "thermodynamic_variation": thermodynamic_variation,
    }


# ---------------------------------------------------------------------------
# Plotting-data extraction
# ---------------------------------------------------------------------------


def extract_scalar_time_data(data: dict):
    """Convert scalar plotting data to a time-first NumPy array."""
    times = sorted(data, key=float)

    values = np.stack(
        [
            xp.to_numpy(data[time][0])
            for time in times
        ],
        axis=0,
    )

    return times, values


def extract_vector_time_data(data: dict):
    """Convert vector plotting data to ``(nt, 3, nx, ny, nz)``."""
    times = sorted(data, key=float)

    values = np.stack(
        [
            np.stack(
                [
                    xp.to_numpy(component)
                    for component in data[time]
                ],
                axis=0,
            )
            for time in times
        ],
        axis=0,
    )

    return times, values


def check_matching_times(
    reference_times,
    other_times,
    variable_name: str,
):
    """Check that two plotting histories use the same saved times."""
    if len(reference_times) != len(other_times):
        raise RuntimeError(
            f"Time-history length mismatch for {variable_name}: "
            f"{len(reference_times)} != {len(other_times)}"
        )

    for index, (reference, other) in enumerate(
        zip(reference_times, other_times)
    ):
        if not np.isclose(float(reference), float(other)):
            raise RuntimeError(
                f"Time mismatch for {variable_name} at index "
                f"{index}: {reference!r} != {other!r}"
            )


# ---------------------------------------------------------------------------
# Pressure diagnostics
# ---------------------------------------------------------------------------


def first_true_index(mask: np.ndarray):
    """Return the first true multidimensional index, or None."""
    locations = np.argwhere(mask)

    if locations.size == 0:
        return None

    return tuple(int(value) for value in locations[0])


def print_field_location(
    title: str,
    index,
    times,
    density,
    entropy,
    log_pressure,
):
    """Print values at a problematic space-time point."""
    if index is None:
        return

    it, ix, iy, iz = index

    rho_value = density[index]
    entropy_value = entropy[index]
    log_pressure_value = log_pressure[index]

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Saved-time index : {it}")
    print(f"Time             : {float(times[it]):.16e}")
    print(f"Spatial index    : ({ix}, {iy}, {iz})")
    print(f"Density          : {rho_value!r}")
    print(f"Entropy density  : {entropy_value!r}")

    if (
        np.isfinite(rho_value)
        and rho_value != 0.0
        and np.isfinite(entropy_value)
    ):
        print(
            f"Entropy / density: "
            f"{entropy_value / rho_value!r}"
        )

    print(f"log(pressure)    : {log_pressure_value!r}")


def diagnose_thermodynamic_fields(
    times,
    density: np.ndarray,
    entropy: np.ndarray,
    gamma: float,
) -> dict:
    r"""Diagnose pressure without directly overflowing ``exp``.

    Pressure is defined by

    .. math::

        p = (\gamma-1)\rho^\gamma \exp(s/\rho),

    and therefore

    .. math::

        \log p =
        \log(\gamma-1)
        + \gamma\log\rho
        + s/\rho.

    The logarithmic expression is only evaluated where density is
    finite and strictly positive and entropy is finite.
    """
    density = np.asarray(density, dtype=np.float64)
    entropy = np.asarray(entropy, dtype=np.float64)

    if density.shape != entropy.shape:
        raise RuntimeError(
            "Density and entropy have different shapes: "
            f"{density.shape} != {entropy.shape}"
        )

    finite_density = np.isfinite(density)
    finite_entropy = np.isfinite(entropy)
    positive_density = density > 0.0

    valid_input = (
        finite_density
        & finite_entropy
        & positive_density
    )

    nonfinite_density = ~finite_density
    nonfinite_entropy = ~finite_entropy
    nonpositive_density = finite_density & ~positive_density

    log_pressure = np.full(
        density.shape,
        np.nan,
        dtype=np.float64,
    )

    # Invalid operations are prevented with the valid_input mask.
    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
        under="ignore",
    ):
        log_pressure[valid_input] = (
            np.log(gamma - 1.0)
            + gamma * np.log(density[valid_input])
            + entropy[valid_input] / density[valid_input]
        )

    nonfinite_log_pressure = (
        valid_input & ~np.isfinite(log_pressure)
    )

    float64_log_max = float(
        np.log(np.finfo(np.float64).max)
    )

    # exp(x) starts returning zero near this range. Using nextafter gives
    # the smallest positive subnormal representable float.
    smallest_positive = np.nextafter(
        np.float64(0.0),
        np.float64(1.0),
    )
    float64_log_min = float(np.log(smallest_positive))

    pressure_overflow = (
        np.isfinite(log_pressure)
        & (log_pressure > float64_log_max)
    )

    pressure_underflow = (
        np.isfinite(log_pressure)
        & (log_pressure < float64_log_min)
    )

    safe_pressure = (
        np.isfinite(log_pressure)
        & ~pressure_overflow
        & ~pressure_underflow
    )

    pressure = np.full(
        density.shape,
        np.nan,
        dtype=np.float64,
    )

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        pressure[safe_pressure] = np.exp(
            log_pressure[safe_pressure]
        )

    finite_logp = np.isfinite(log_pressure)

    minimum_density = (
        float(np.min(density[finite_density]))
        if np.any(finite_density)
        else np.nan
    )

    maximum_density = (
        float(np.max(density[finite_density]))
        if np.any(finite_density)
        else np.nan
    )

    minimum_entropy = (
        float(np.min(entropy[finite_entropy]))
        if np.any(finite_entropy)
        else np.nan
    )

    maximum_entropy = (
        float(np.max(entropy[finite_entropy]))
        if np.any(finite_entropy)
        else np.nan
    )

    minimum_log_pressure = (
        float(np.min(log_pressure[finite_logp]))
        if np.any(finite_logp)
        else np.nan
    )

    maximum_log_pressure = (
        float(np.max(log_pressure[finite_logp]))
        if np.any(finite_logp)
        else np.nan
    )

    finite_pressure = np.isfinite(pressure)

    minimum_pressure = (
        float(np.min(pressure[finite_pressure]))
        if np.any(finite_pressure)
        else np.nan
    )

    maximum_pressure = (
        float(np.max(pressure[finite_pressure]))
        if np.any(finite_pressure)
        else np.nan
    )

    print("\nREGULARIZED ORSZAG--TANG FIELD DIAGNOSTICS")
    print("==========================================")
    print(
        f"Density range              : "
        f"[{minimum_density:.12e}, {maximum_density:.12e}]"
    )
    print(
        f"Entropy-density range       : "
        f"[{minimum_entropy:.12e}, {maximum_entropy:.12e}]"
    )
    print(
        f"Finite log-pressure range   : "
        f"[{minimum_log_pressure:.12e}, "
        f"{maximum_log_pressure:.12e}]"
    )
    print(
        f"Safe finite pressure range  : "
        f"[{minimum_pressure:.12e}, {maximum_pressure:.12e}]"
    )
    print(
        f"float64 maximum log-pressure: "
        f"{float64_log_max:.12e}"
    )
    print(
        f"float64 minimum log-pressure: "
        f"{float64_log_min:.12e}"
    )

    print("\nProblem counts:")
    print(
        f"Non-finite density points   : "
        f"{np.count_nonzero(nonfinite_density)}"
    )
    print(
        f"Non-positive density points : "
        f"{np.count_nonzero(nonpositive_density)}"
    )
    print(
        f"Non-finite entropy points   : "
        f"{np.count_nonzero(nonfinite_entropy)}"
    )
    print(
        f"Non-finite log-pressure     : "
        f"{np.count_nonzero(nonfinite_log_pressure)}"
    )
    print(
        f"Pressure-overflow points    : "
        f"{np.count_nonzero(pressure_overflow)}"
    )
    print(
        f"Pressure-underflow points   : "
        f"{np.count_nonzero(pressure_underflow)}"
    )

    print_field_location(
        "FIRST NON-FINITE DENSITY",
        first_true_index(nonfinite_density),
        times,
        density,
        entropy,
        log_pressure,
    )

    print_field_location(
        "FIRST NON-POSITIVE DENSITY",
        first_true_index(nonpositive_density),
        times,
        density,
        entropy,
        log_pressure,
    )

    print_field_location(
        "FIRST NON-FINITE ENTROPY",
        first_true_index(nonfinite_entropy),
        times,
        density,
        entropy,
        log_pressure,
    )

    print_field_location(
        "FIRST NON-FINITE LOG-PRESSURE",
        first_true_index(nonfinite_log_pressure),
        times,
        density,
        entropy,
        log_pressure,
    )

    print_field_location(
        "FIRST PRESSURE OVERFLOW",
        first_true_index(pressure_overflow),
        times,
        density,
        entropy,
        log_pressure,
    )

    print_field_location(
        "FIRST PRESSURE UNDERFLOW",
        first_true_index(pressure_underflow),
        times,
        density,
        entropy,
        log_pressure,
    )

    spatial_axes = tuple(range(1, density.ndim))

    minimum_density_by_time = np.min(
        density,
        axis=spatial_axes,
    )

    maximum_log_pressure_by_time = np.max(
        np.where(
            finite_logp,
            log_pressure,
            -np.inf,
        ),
        axis=spatial_axes,
    )

    bad_time_mask = (
        ~np.isfinite(minimum_density_by_time)
        | (minimum_density_by_time <= 0.0)
        | ~np.isfinite(maximum_log_pressure_by_time)
        | (
            maximum_log_pressure_by_time
            > float64_log_max
        )
    )

    first_bad_time_index = (
        int(np.flatnonzero(bad_time_mask)[0])
        if np.any(bad_time_mask)
        else None
    )

    if first_bad_time_index is None:
        print(
            "\nNo non-positive density or float64 pressure "
            "overflow was found in the saved plotting data."
        )
    else:
        print("\nFIRST PROBLEMATIC SAVED TIME")
        print("----------------------------")
        print(
            f"Index                : {first_bad_time_index}"
        )
        print(
            f"Time                 : "
            f"{float(times[first_bad_time_index]):.16e}"
        )
        print(
            f"Minimum density      : "
            f"{minimum_density_by_time[first_bad_time_index]:.16e}"
        )
        print(
            f"Maximum log-pressure : "
            f"{maximum_log_pressure_by_time[first_bad_time_index]:.16e}"
        )

    physically_admissible = (
        not np.any(nonfinite_density)
        and not np.any(nonpositive_density)
        and not np.any(nonfinite_entropy)
        and not np.any(nonfinite_log_pressure)
        and not np.any(pressure_overflow)
    )

    if physically_admissible:
        print(
            "\nFIELD STATUS: no saved-grid admissibility failure "
            "was detected."
        )
    else:
        print(
            "\nFIELD STATUS: the global invariants are not sufficient "
            "to validate this solution."
        )
        print(
            "At least one saved-grid density/entropy/pressure "
            "admissibility problem was detected."
        )

    return {
        "density": density,
        "entropy": entropy,
        "log_pressure": log_pressure,
        "pressure": pressure,
        "valid_input": valid_input,
        "pressure_overflow": pressure_overflow,
        "pressure_underflow": pressure_underflow,
        "nonpositive_density": nonpositive_density,
        "nonfinite_density": nonfinite_density,
        "nonfinite_entropy": nonfinite_entropy,
        "first_bad_time_index": first_bad_time_index,
        "minimum_density": minimum_density,
        "maximum_density": maximum_density,
        "minimum_log_pressure": minimum_log_pressure,
        "maximum_log_pressure": maximum_log_pressure,
        "float64_log_max": float64_log_max,
        "physically_admissible": physically_admissible,
    }


# ---------------------------------------------------------------------------
# Loading post-processed fields
# ---------------------------------------------------------------------------


def load_and_check_fields(
    simulation: Simulation,
) -> dict:
    """Load plotting data and diagnose reconstructed MHD fields."""
    simulation.load_plotting_data()

    density_data = (
        simulation
        .spline_values
        .mhd
        .density_log
        .data
    )

    entropy_data = (
        simulation
        .spline_values
        .mhd
        .entropy_log
        .data
    )

    velocity_data = (
        simulation
        .spline_values
        .mhd
        .velocity_log
        .data
    )

    magnetic_data = (
        simulation
        .spline_values
        .em_fields
        .b_field_log
        .data
    )

    times, density_3form = extract_scalar_time_data(
        density_data
    )

    entropy_times, entropy_3form = extract_scalar_time_data(
        entropy_data
    )

    velocity_times, velocity = extract_vector_time_data(
        velocity_data
    )

    magnetic_times, magnetic = extract_vector_time_data(
        magnetic_data
    )

    check_matching_times(
        times,
        entropy_times,
        "entropy",
    )

    check_matching_times(
        times,
        velocity_times,
        "velocity",
    )

    check_matching_times(
        times,
        magnetic_times,
        "magnetic field",
    )

    jacobian = xp.to_numpy(
        simulation.domain.jacobian_det(
            *simulation.grids_log
        ),
    ).astype(np.float64, copy=False)

    if not np.all(np.isfinite(jacobian)):
        raise RuntimeError(
            "The geometry Jacobian contains non-finite values."
        )

    if np.any(jacobian <= 0.0):
        raise RuntimeError(
            "The geometry Jacobian is not strictly positive."
        )

    # L2 variables are stored as logical 3-forms.
    density = density_3form / jacobian[None, ...]
    entropy = entropy_3form / jacobian[None, ...]

    if not np.all(np.isfinite(velocity)):
        index = tuple(
            int(value)
            for value in np.argwhere(
                ~np.isfinite(velocity)
            )[0]
        )
        raise RuntimeError(
            "Velocity contains non-finite values.\n"
            f"First bad index: {index}"
        )

    if not np.all(np.isfinite(magnetic)):
        index = tuple(
            int(value)
            for value in np.argwhere(
                ~np.isfinite(magnetic)
            )[0]
        )
        raise RuntimeError(
            "Magnetic field contains non-finite values.\n"
            f"First bad index: {index}"
        )

    thermo = diagnose_thermodynamic_fields(
        times=times,
        density=density,
        entropy=entropy,
        gamma=GAMMA,
    )

    return {
        "times": times,
        "density": density,
        "entropy": entropy,
        "velocity": velocity,
        "magnetic": magnetic,
        "log_pressure": thermo["log_pressure"],
        "pressure": thermo["pressure"],
        "thermo_diagnostics": thermo,
    }

def prepare_output_directory(
    *,
    run: bool,
    overwrite: bool,
):
    """Validate or remove output before constructing Simulation.

    Simulation construction may create and initialize the output directory.
    Therefore an old output directory must be removed before calling
    create_simulation(), never afterward.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_path = Path(OUTPUT_DIRECTORY)

    output_exists = None

    if rank == 0:
        output_exists = output_path.exists()

    output_exists = comm.bcast(
        output_exists,
        root=0,
    )

    if run:
        if output_exists and not overwrite:
            raise FileExistsError(
                "The output directory already exists:\n"
                f"{output_path}\n\n"
                "Use the default load mode to inspect it, or use\n"
                "    --run --overwrite\n"
                "to delete it and run again."
            )

        if rank == 0 and output_exists:
            print(
                "Removing existing output because --overwrite "
                "was supplied:"
            )
            print(output_path)

            shutil.rmtree(output_path)

        # Only create the parent here. Simulation/EnvironmentOptions will
        # initialize the actual simulation output directory.
        if rank == 0:
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

    else:
        if not output_exists:
            raise FileNotFoundError(
                "The existing simulation output directory was not found:\n"
                f"{output_path}\n\n"
                "Run with --run to create it, or correct "
                "OUTPUT_DIRECTORY near the top of this file."
            )

        data_path = output_path / "data"

        data_exists = None

        if rank == 0:
            data_exists = data_path.is_dir()

        data_exists = comm.bcast(
            data_exists,
            root=0,
        )

        if not data_exists:
            raise FileNotFoundError(
                "The raw simulation data directory was not found:\n"
                f"{data_path}"
            )

    comm.Barrier()
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(
    simulation: Simulation,
    scalar_diagnostics: dict,
    field_data: dict,
):
    """Plot final fields and energy histories."""
    density = field_data["density"]
    pressure = field_data["pressure"]
    log_pressure = field_data["log_pressure"]

    time_history = scalar_diagnostics["time"]
    total_energy = scalar_diagnostics["en_tot"]
    kinetic_energy = scalar_diagnostics["en_U"]
    magnetic_energy = scalar_diagnostics["en_mag"]
    thermodynamic_energy = scalar_diagnostics["en_thermo"]

    x = xp.to_numpy(simulation.grids_phy[0][:, :, 0])
    y = xp.to_numpy(simulation.grids_phy[1][:, :, 0])

    final_pressure = pressure[-1, :, :, 0]
    final_log_pressure = log_pressure[-1, :, :, 0]

    # If pressure overflowed, plot log-pressure instead.
    plot_log_pressure = not np.all(
        np.isfinite(final_pressure)
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
    )

    density_plot = axes[0, 0].contourf(
        x,
        y,
        density[-1, :, :, 0],
        levels=40,
        cmap="viridis",
    )

    fig.colorbar(
        density_plot,
        ax=axes[0, 0],
    )

    axes[0, 0].set_title("Final density")

    if plot_log_pressure:
        pressure_plot = axes[0, 1].contourf(
            x,
            y,
            np.ma.masked_invalid(final_log_pressure),
            levels=40,
            cmap="plasma",
        )

        axes[0, 1].set_title(
            "Final log-pressure\n"
            "(pressure was not finite everywhere)"
        )
    else:
        pressure_plot = axes[0, 1].contourf(
            x,
            y,
            final_pressure,
            levels=40,
            cmap="plasma",
        )

        axes[0, 1].set_title("Final pressure")

    fig.colorbar(
        pressure_plot,
        ax=axes[0, 1],
    )

    axes[1, 0].plot(
        time_history,
        kinetic_energy,
        label="regularized kinetic",
    )

    axes[1, 0].plot(
        time_history,
        magnetic_energy,
        label="magnetic",
    )

    axes[1, 0].plot(
        time_history,
        thermodynamic_energy,
        label="thermodynamic",
    )

    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("energy")
    axes[1, 0].set_title("Energy exchange")
    axes[1, 0].legend()

    relative_energy_change = (
        (total_energy - total_energy[0])
        / total_energy[0]
    )

    axes[1, 1].plot(
        time_history,
        relative_energy_change,
    )

    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel(
        r"$(E(t)-E(0))/E(0)$"
    )
    axes[1, 1].set_title(
        "Relative regularized-energy change"
    )

    for axis in axes[0, :]:
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")

    fig.suptitle(
        "Regularized ideal-MHD Orszag--Tang vortex\n"
        rf"$\alpha_{{\mathrm{{divdiv}}}}"
        rf"={ALPHA_DIVDIV}$"
    )

    fig.tight_layout()
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run/load control
# ---------------------------------------------------------------------------

def run_new_simulation(
    simulation: Simulation,
    env: EnvironmentOptions,
):
    """Run a new simulation.

    Any existing output must already have been handled before constructing
    the Simulation object.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_path = Path(env.path_out)

    # Simulation construction should normally create this directory.
    # This is a final safeguard for versions that do not.
    if rank == 0:
        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

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
        simulation.pproc(
            create_vtk=False,
        )

    comm.Barrier()

def ensure_existing_output(
    output_path: str | Path,
):
    """Check that the existing simulation output is available."""
    output_path = Path(output_path)

    if not output_path.is_dir():
        raise FileNotFoundError(
            "The existing simulation output directory was not found:\n"
            f"{output_path}\n\n"
            "Run with --run to create it, or correct "
            "OUTPUT_DIRECTORY near the top of this file."
        )

    data_path = output_path / "data"

    if not data_path.is_dir():
        raise FileNotFoundError(
            f"The raw data directory does not exist:\n{data_path}"
        )


def ensure_post_processing(
    simulation: Simulation,
    output_path: str | Path,
    allow_post_process: bool,
):
    """Ensure post-processed plotting data exist."""
    output_path = Path(output_path)
    post_processing_path = output_path / "post_processing"

    if post_processing_path.is_dir():
        return

    if not allow_post_process:
        raise FileNotFoundError(
            "Post-processed plotting data were not found:\n"
            f"{post_processing_path}\n\n"
            "The raw simulation does not need to be rerun. Use:\n"
            "    python test_orszag_tang_regularized.py "
            "--post-process\n"
            "to generate post-processing data from the existing run."
        )

    print(
        "\nPost-processing data are missing. Generating them "
        "from the existing raw output."
    )

    simulation.pproc(create_vtk=False)

    if not post_processing_path.is_dir():
        raise RuntimeError(
            "Post-processing completed without creating the expected "
            f"directory:\n{post_processing_path}"
        )

def execute(
    *,
    run: bool = False,
    overwrite: bool = False,
    post_process: bool = False,
    do_plot: bool = False,
):
    """Run a new case or load and diagnose an existing case."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # This must happen before create_simulation(). Constructing Simulation
    # may create files and directories required by simulation.run().
    prepare_output_directory(
        run=run,
        overwrite=overwrite,
    )

    simulation, env = create_simulation()
    output_path = Path(env.path_out)

    if run:
        run_new_simulation(
            simulation=simulation,
            env=env,
        )
    else:
        if rank == 0:
            print("\nLOADING EXISTING SIMULATION")
            print("===========================")
            print(f"Output path: {output_path}")
            print("The time integration will not be rerun.")

    comm.Barrier()

    scalar_diagnostics = None
    field_data = None

    if rank == 0:
        scalar_diagnostics = check_mhd_scalar_diagnostics(
            output_path,
            energy_tolerance=1.0e-3,
        )

        ensure_post_processing(
            simulation=simulation,
            output_path=output_path,
            allow_post_process=post_process,
        )

        field_data = load_and_check_fields(
            simulation,
        )

        if do_plot:
            plot_results(
                simulation=simulation,
                scalar_diagnostics=scalar_diagnostics,
                field_data=field_data,
            )

    comm.Barrier()

    return scalar_diagnostics, field_data
# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------


def test_regularized_orszag_tang_existing_output():
    """Check an existing regularized Orszag--Tang run.

    This test does not rerun the simulation and does not delete output.
    """
    scalar_diagnostics, field_data = execute(
        run=False,
        overwrite=False,
        post_process=False,
        do_plot=False,
    )

    if MPI.COMM_WORLD.Get_rank() == 0:
        assert scalar_diagnostics is not None
        assert field_data is not None

        # This assertion deliberately checks local admissibility in
        # addition to global conservation.
        assert field_data[
            "thermo_diagnostics"
        ]["physically_admissible"], (
            "The scalar invariants are well conserved, but the saved "
            "fields contain a non-positive density, non-finite state, "
            "or unrepresentable pressure."
        )


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Load or run the regularized ideal-MHD "
            "Orszag--Tang verification."
        )
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help=(
            "Run a new simulation. Without this option, the existing "
            "simulation is loaded without rerunning it."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow --run to delete and replace an existing output "
            "directory. Ignored in load mode."
        ),
    )

    parser.add_argument(
        "--post-process",
        action="store_true",
        help=(
            "Generate post-processed plotting data from existing raw "
            "output if the post_processing directory is missing."
        ),
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display density, pressure, and energy plots.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    if arguments.overwrite and not arguments.run:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(
                "Warning: --overwrite has no effect without --run."
            )

    execute(
        run=arguments.run,
        overwrite=arguments.overwrite,
        post_process=arguments.post_process,
        do_plot=arguments.plot,
    )
