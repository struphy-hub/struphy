"""Post-processing for params_OrszagTang.py.

Loads an existing Orszag--Tang ViscoResistiveMHD run, checks scalar
diagnostics (mass/entropy conservation, div(B)), reconstructs the
pressure from the density/entropy formulation, and plots the final
fields together with the energy history.

Usage
-----
Run the simulation first:

    python params_OrszagTang.py

Then post-process and plot:

    python pproc_OrszagTang.py
"""

import params_orszag_tang as params

import glob
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from feectools.ddm.mpi import mpi as MPI

GAMMA = params.GAMMA


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
    for name, values in histories.items():
        if values.size != expected_size:
            raise RuntimeError(f"{name} has {values.size} entries, but time has {expected_size} entries.")
        if not np.all(np.isfinite(values)):
            first_bad = int(np.flatnonzero(~np.isfinite(values))[0])
            raise RuntimeError(
                f"Non-finite values found in {name}.\n"
                f"First bad index: {first_bad}, time: {time_history[first_bad]!r}, value: {values[first_bad]!r}"
            )

    if not np.isclose(time_history[0], 0.0):
        raise RuntimeError(f"The first saved time is {time_history[0]}, not zero.")

    # Check the total-energy definition.
    energy_sum = kinetic_energy + magnetic_energy + thermodynamic_energy
    component_sum_error = float(np.max(np.abs(total_energy - energy_sum)))
    relative_component_sum_error = component_sum_error / max(abs(total_energy[0]), 1.0)

    if not np.allclose(total_energy, energy_sum, rtol=1.0e-12, atol=1.0e-12):
        raise RuntimeError(
            "The saved total energy is inconsistent with en_U + en_mag + en_thermo.\n"
            f"Max abs difference: {component_sum_error:.12e}, relative: {relative_component_sum_error:.12e}"
        )

    def relative_drift(values):
        return float(np.max(np.abs(values - values[0])) / max(abs(values[0]), 1.0e-30))

    def final_relative_change(values):
        return float((values[-1] - values[0]) / max(abs(values[0]), 1.0e-30))

    def relative_variation(values):
        return float((np.max(values) - np.min(values)) / max(abs(values[0]), 1.0e-30))

    energy_drift = relative_drift(total_energy)
    mass_drift = relative_drift(total_mass)
    entropy_drift = relative_drift(total_entropy)

    final_energy_change = final_relative_change(total_energy)
    final_mass_change = final_relative_change(total_mass)
    final_entropy_change = final_relative_change(total_entropy)

    maximum_div_b = float(np.max(np.abs(div_b)))

    kinetic_variation = relative_variation(kinetic_energy)
    magnetic_variation = relative_variation(magnetic_energy)
    thermodynamic_variation = relative_variation(thermodynamic_energy)

    print("\nIDEAL-MHD SCALAR DIAGNOSTICS")
    print("=============================")
    print(f"Saved states                  : {expected_size}")
    print(f"First saved time              : {time_history[0]:.12e}")
    print(f"Last saved time                : {time_history[-1]:.12e}")

    print("\nDiscrete energy:")
    print(f"Initial total energy          : {total_energy[0]:.16e}")
    print(f"Final total energy            : {total_energy[-1]:.16e}")
    print(f"Maximum relative energy drift  : {energy_drift:.12e}")
    print(f"Signed final energy change     : {final_energy_change:.12e}")

    print("\nEnergy-definition consistency:")
    print(f"Maximum component-sum error    : {component_sum_error:.12e}")
    print(f"Relative component-sum error   : {relative_component_sum_error:.12e}")

    print("\nConserved quantities:")
    print(f"Maximum relative mass drift    : {mass_drift:.12e}")
    print(f"Signed final mass change       : {final_mass_change:.12e}")
    print(f"Maximum relative entropy drift : {entropy_drift:.12e}")
    print(f"Signed final entropy change    : {final_entropy_change:.12e}")
    print(f"Maximum div(B) diagnostic      : {maximum_div_b:.12e}")

    print("\nEnergy exchange:")
    print(f"Relative kinetic variation     : {kinetic_variation:.12e}")
    print(f"Relative magnetic variation    : {magnetic_variation:.12e}")
    print(f"Thermodynamic variation        : {thermodynamic_variation:.12e}")

    if energy_tolerance is not None and energy_drift >= energy_tolerance:
        print(f"WARNING: relative energy drift {energy_drift:.6e} exceeds the tolerance {energy_tolerance:.6e}.")

    return {
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
    }


# ---------------------------------------------------------------------------
# Field extraction and pressure reconstruction
# ---------------------------------------------------------------------------


def extract_scalar_time_data(data: dict):
    """Convert time-keyed scalar plotting data to a time-first array."""
    times = sorted(data, key=float)
    values = np.stack([np.asarray(data[t][0]) for t in times], axis=0)
    return times, values


def extract_vector_time_data(data: dict):
    """Convert time-keyed vector plotting data to ``(nt, 3, nx, ny, nz)``."""
    times = sorted(data, key=float)
    values = np.stack(
        [np.stack([np.asarray(c) for c in data[t]], axis=0) for t in times],
        axis=0,
    )
    return times, values


def check_matching_times(reference_times, other_times, variable_name: str):
    """Check that two plotting histories use the same saved times."""
    if len(reference_times) != len(other_times):
        raise RuntimeError(f"Time-history length mismatch for {variable_name}.")
    for i, (ref, other) in enumerate(zip(reference_times, other_times)):
        if not np.isclose(float(ref), float(other)):
            raise RuntimeError(f"Time mismatch for {variable_name} at index {i}: {ref!r} != {other!r}")


def diagnose_thermodynamic_fields(density: np.ndarray, entropy: np.ndarray, gamma: float) -> dict:
    r"""Reconstruct pressure from density/entropy without overflowing ``exp``.

    Pressure is defined by ``p = (gamma - 1) * rho**gamma * exp(s / rho)``, so

        log(p) = log(gamma - 1) + gamma * log(rho) + s / rho

    is evaluated first, to diagnose overflow before it produces infinity.
    """
    density = np.asarray(density, dtype=np.float64)
    entropy = np.asarray(entropy, dtype=np.float64)

    finite_density = np.isfinite(density)
    finite_entropy = np.isfinite(entropy)
    positive_density = density > 0.0
    valid_input = finite_density & finite_entropy & positive_density

    log_pressure = np.full(density.shape, np.nan, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        log_pressure[valid_input] = (
            np.log(gamma - 1.0) + gamma * np.log(density[valid_input]) + entropy[valid_input] / density[valid_input]
        )

    float64_log_max = float(np.log(np.finfo(np.float64).max))
    pressure_overflow = np.isfinite(log_pressure) & (log_pressure > float64_log_max)
    safe_pressure = np.isfinite(log_pressure) & ~pressure_overflow

    pressure = np.full(density.shape, np.nan, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        pressure[safe_pressure] = np.exp(log_pressure[safe_pressure])

    physically_admissible = (
        np.all(finite_density)
        and np.all(positive_density)
        and np.all(finite_entropy)
        and not np.any(pressure_overflow)
    )

    print("\nFIELD DIAGNOSTICS")
    print("==================")
    print(f"Non-finite density points  : {np.count_nonzero(~finite_density)}")
    print(f"Non-positive density points: {np.count_nonzero(finite_density & ~positive_density)}")
    print(f"Non-finite entropy points  : {np.count_nonzero(~finite_entropy)}")
    print(f"Pressure-overflow points   : {np.count_nonzero(pressure_overflow)}")

    if physically_admissible:
        print("STATUS: no saved-grid admissibility failure was detected.")
    else:
        print("STATUS: at least one saved-grid density/entropy/pressure admissibility problem was detected.")

    return {
        "log_pressure": log_pressure,
        "pressure": pressure,
        "physically_admissible": physically_admissible,
    }


def load_and_check_fields(sim, gamma: float) -> dict:
    """Load plotting data and reconstruct density, pressure, velocity, B."""
    sim.load_plotting_data()

    density_data = sim.spline_values.mhd.density_log.data
    entropy_data = sim.spline_values.mhd.entropy_log.data
    velocity_data = sim.spline_values.mhd.velocity_log.data
    magnetic_data = sim.spline_values.em_fields.b_field_log.data

    times, density_3form = extract_scalar_time_data(density_data)
    entropy_times, entropy_3form = extract_scalar_time_data(entropy_data)
    velocity_times, velocity = extract_vector_time_data(velocity_data)
    magnetic_times, magnetic = extract_vector_time_data(magnetic_data)

    check_matching_times(times, entropy_times, "entropy")
    check_matching_times(times, velocity_times, "velocity")
    check_matching_times(times, magnetic_times, "magnetic field")

    jacobian = np.asarray(sim.domain.jacobian_det(*sim.grids_log), dtype=np.float64)
    if not np.all(np.isfinite(jacobian)) or np.any(jacobian <= 0.0):
        raise RuntimeError("The geometry Jacobian is not finite and strictly positive everywhere.")

    # L2 variables are stored as logical 3-forms.
    density = density_3form / jacobian[None, ...]
    entropy = entropy_3form / jacobian[None, ...]

    thermo = diagnose_thermodynamic_fields(density, entropy, gamma)

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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(sim, scalar_diagnostics: dict, field_data: dict):
    """Plot final density/pressure fields and energy histories."""
    density = field_data["density"]
    pressure = field_data["pressure"]
    log_pressure = field_data["log_pressure"]

    time_history = scalar_diagnostics["time"]
    total_energy = scalar_diagnostics["en_tot"]
    kinetic_energy = scalar_diagnostics["en_U"]
    magnetic_energy = scalar_diagnostics["en_mag"]
    thermodynamic_energy = scalar_diagnostics["en_thermo"]

    x = np.asarray(sim.grids_phy[0][:, :, 0])
    y = np.asarray(sim.grids_phy[1][:, :, 0])

    final_pressure = pressure[-1, :, :, 0]
    final_log_pressure = log_pressure[-1, :, :, 0]
    plot_log_pressure = not np.all(np.isfinite(final_pressure))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    density_plot = axes[0, 0].contourf(x, y, density[-1, :, :, 0], levels=40, cmap="viridis")
    fig.colorbar(density_plot, ax=axes[0, 0])
    axes[0, 0].set_title("Final density")

    if plot_log_pressure:
        pressure_plot = axes[0, 1].contourf(x, y, np.ma.masked_invalid(final_log_pressure), levels=40, cmap="plasma")
        axes[0, 1].set_title("Final log-pressure\n(pressure was not finite everywhere)")
    else:
        pressure_plot = axes[0, 1].contourf(x, y, final_pressure, levels=40, cmap="plasma")
        axes[0, 1].set_title("Final pressure")
    fig.colorbar(pressure_plot, ax=axes[0, 1])

    axes[1, 0].plot(time_history, kinetic_energy, label="kinetic")
    axes[1, 0].plot(time_history, magnetic_energy, label="magnetic")
    axes[1, 0].plot(time_history, thermodynamic_energy, label="thermodynamic")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("energy")
    axes[1, 0].set_title("Energy exchange")
    axes[1, 0].legend()

    relative_energy_change = (total_energy - total_energy[0]) / total_energy[0]
    axes[1, 1].plot(time_history, relative_energy_change)
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel(r"$(E(t)-E(0))/E(0)$")
    axes[1, 1].set_title("Relative energy change")

    for axis in axes[0, :]:
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")

    fig.suptitle("Ideal-MHD Orszag--Tang vortex")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(do_plot: bool = True):
    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    sim = params.sim
    output_path = params.env.path_out

    sim.pproc()

    scalar_diagnostics = check_mhd_scalar_diagnostics(output_path, energy_tolerance=1.0e-3)
    field_data = load_and_check_fields(sim, GAMMA)

    if do_plot:
        plot_results(sim, scalar_diagnostics, field_data)


if __name__ == "__main__":
    main(do_plot=True)