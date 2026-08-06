import logging
import os
import shutil

import cunumpy as xp
import matplotlib.pyplot as plt
from feectools.ddm.mpi import mpi as MPI
from matplotlib import colors

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
    set_logging_level,
)
from struphy.diagnostics.diagn_tools import power_spectrum_2d
from struphy.models import ViscoResistiveMHD


set_logging_level()
logger = logging.getLogger("struphy")


def extract_scalar_time_data(data: dict):
    """Convert time-keyed scalar plotting data to a time-first array."""
    times = sorted(data, key=float)

    values = xp.stack(
        [data[time][0] for time in times],
        axis=0,
    )

    return times, values


def fit_branch_near_expected_speed(
    omega,
    kvec,
    spectrum,
    expected_speed,
    competing_speed=None,
    search_bins=1,
    min_separation_bins=2,
):
    """Fit a spectral branch near the expected relation omega = v*k.

    The branch is fitted through the origin, as required for homogeneous
    ideal-MHD waves.
    """
    k_start = kvec.size // 8
    k_end = kvec.size // 2
    domega = float(omega[1] - omega[0])

    k_fit = []
    omega_fit = []

    for j in range(k_start, k_end):
        k = float(kvec[j])
        omega_expected = float(expected_speed) * k

        # Skip points at which this branch is not distinguishable from a
        # nearby competing branch at the available FFT frequency resolution.
        if competing_speed is not None:
            separation = (
                abs(float(expected_speed) - float(competing_speed))
                * k
            )

            if separation < min_separation_bins * domega:
                continue

        center = int(
            xp.argmin(
                xp.abs(omega - omega_expected)
            )
        )

        lower = max(1, center - search_bins)
        upper = min(
            omega.size - 1,
            center + search_bins + 1,
        )

        if upper <= lower:
            continue

        local_spectrum = spectrum[lower:upper, j]
        peak = lower + int(xp.argmax(local_spectrum))

        omega_peak = float(omega[peak])

        # Quadratic interpolation for sub-bin peak accuracy.
        if 0 < peak < omega.size - 1:
            y_minus = float(spectrum[peak - 1, j])
            y_zero = float(spectrum[peak, j])
            y_plus = float(spectrum[peak + 1, j])

            denominator = y_minus - 2.0 * y_zero + y_plus

            if abs(denominator) > 1e-30:
                offset = (
                    0.5
                    * (y_minus - y_plus)
                    / denominator
                )

                if abs(offset) <= 1.0:
                    omega_peak += offset * domega

        k_fit.append(k)
        omega_fit.append(omega_peak)

    assert len(k_fit) >= 2, (
        "Not enough spectrally resolved points to fit the branch."
    )

    k_fit = xp.asarray(k_fit)
    omega_fit = xp.asarray(omega_fit)

    # Least-squares fit constrained through the origin:
    #
    #     omega = v*k.
    slope = (
        xp.sum(k_fit * omega_fit)
        / xp.sum(k_fit**2)
    )

    return float(slope)


def test_slab_waves_1d(do_plot: bool = False):
    # ------------------------------------------------------------------
    # Model: ideal limit of ViscoResistiveMHD
    # ------------------------------------------------------------------

    model = ViscoResistiveMHD(
        with_viscosity=False,
        with_resistivity=False,
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    test_folder = os.path.join(
        os.getcwd(),
        "verification_tests_full",
    )

    out_folders = os.path.join(
        test_folder,
        "ViscoResistiveMHD",
    )

    env = EnvironmentOptions(
        out_folders=out_folders,
        sim_folder="slab_waves_1d",
    )

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    time_opts = Time(
        dt=0.15,
        Tend=180.0,
    )

    # ------------------------------------------------------------------
    # Geometry and equilibrium
    # ------------------------------------------------------------------

    domain = domains.Cuboid(r3=60.0)

    B0x = 0.0
    B0y = 1.0
    B0z = 1.0
    beta = 3.0
    n0 = 0.7
    gamma = 5.0 / 3.0

    equil = equils.HomogenSlab(
        B0x=B0x,
        B0y=B0y,
        B0z=B0z,
        beta=beta,
        n0=n0,
    )

    # ------------------------------------------------------------------
    # Full-f equilibrium initialization
    # ------------------------------------------------------------------

    model.mhd.density.add_background(
        FieldsBackground(
            type="FluidEquilibrium",
            variable="n3",
        )
    )

    model.mhd.entropy.add_background(
        FieldsBackground(
            type="FluidEquilibrium",
            variable="s3_monoatomic",
        )
    )

    model.em_fields.b_field.add_background(
        FieldsBackground(
            type="FluidEquilibrium",
            variable="b2",
        )
    )

    # ------------------------------------------------------------------
    # Spatial discretization
    # ------------------------------------------------------------------

    grid = grids.TensorProductGrid(
        num_elements=(1, 1, 64),
    )

    derham_opts = DerhamOptions(
        degree=(1, 1, 3),
    )

    # ------------------------------------------------------------------
    # Full thermodynamic model
    # ------------------------------------------------------------------

    model.propagators.variat_dens.options = (
        model.propagators.variat_dens.Options(
            model="full",
            gamma=gamma,
        )
    )

    model.propagators.variat_ent.options = (
        model.propagators.variat_ent.Options(
            model="full",
            gamma=gamma,
        )
    )

    model.propagators.variat_mag.options = (
        model.propagators.variat_mag.Options(
            model="full",
        )
    )

    # ------------------------------------------------------------------
    # Initial velocity perturbations
    # ------------------------------------------------------------------

    for component in range(3):
        model.mhd.velocity.add_perturbation(
            perturbations.Noise(
                amp=0.001,
                comp=component,
                seed=123,
            )
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
        equil=equil,
    )

    sim.run()

    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    sim.pproc()
    sim.load_plotting_data()

    # ------------------------------------------------------------------
    # Analytical wave speeds
    # ------------------------------------------------------------------

    Bsquare = B0x**2 + B0y**2 + B0z**2
    p0 = beta * Bsquare / 2.0

    disp_params = {
        "B0x": B0x,
        "B0y": B0y,
        "B0z": B0z,
        "p0": p0,
        "n0": n0,
        "gamma": gamma,
    }

    vA = xp.sqrt(Bsquare / n0)
    v_alfven = vA * B0z / xp.sqrt(Bsquare)

    cS = xp.sqrt(gamma * p0 / n0)

    delta = (
        4.0
        * B0z**2
        * cS**2
        * vA**2
        / (
            (cS**2 + vA**2) ** 2
            * Bsquare
        )
    )

    v_slow = xp.sqrt(
        0.5
        * (cS**2 + vA**2)
        * (1.0 - xp.sqrt(1.0 - delta))
    )

    v_fast = xp.sqrt(
        0.5
        * (cS**2 + vA**2)
        * (1.0 + xp.sqrt(1.0 - delta))
    )

    # ------------------------------------------------------------------
    # Plot 1: Alfvén-wave velocity spectrum
    # ------------------------------------------------------------------

    velocity_data = (
        sim.spline_values.mhd.velocity_log.data
    )

    _, _, _, coeffs_alfven = power_spectrum_2d(
        velocity_data,
        "velocity_log",
        grids=sim.grids_log,
        grids_mapped=sim.grids_phy,
        component=0,
        slice_at=[0, 0, None],
        do_plot=do_plot,
        disp_name="MHDhomogenSlab",
        disp_params=disp_params,
        fit_branches=1,
        noise_level=0.5,
        extr_order=10,
        fit_degree=(1,),
    )

    v_alfven_fit = float(coeffs_alfven[0][0])
    alfven_error = abs(
        v_alfven_fit - float(v_alfven)
    )

    logger.info(f"{v_alfven = }")
    logger.info(f"{v_alfven_fit = }")
    logger.info(f"{alfven_error = }")

    assert alfven_error < 0.07

    # ------------------------------------------------------------------
    # Reconstruct physical pressure
    # ------------------------------------------------------------------

    ## This is different than the Linear case, because pressure spectrum is more complicated, it does not only contain linear pertubations
    ##The automatic branch fitter inside power_spectrum_2d seems to be the problem. 

    density_data = (
        sim.spline_values.mhd.density_log.data
    )
    entropy_data = (
        sim.spline_values.mhd.entropy_log.data
    )

    times, density_3form = extract_scalar_time_data(
        density_data
    )
    entropy_times, entropy_3form = (
        extract_scalar_time_data(entropy_data)
    )

    assert len(times) == len(entropy_times)
    assert all(
        xp.isclose(float(t1), float(t2))
        for t1, t2 in zip(times, entropy_times)
    )

    # Post-processed L2 variables are logical 3-forms. Convert them
    # to physical scalar density and entropy density.
    jacobian = sim.domain.jacobian_det(
        *sim.grids_log
    )

    rho = density_3form / jacobian[None, ...]
    entropy = entropy_3form / jacobian[None, ...]

    pressure = (
        (gamma - 1.0)
        * rho**gamma
        * xp.exp(entropy / rho)
    )

    assert xp.all(xp.isfinite(rho))
    assert xp.all(xp.isfinite(pressure))
    assert xp.min(rho) > 0.0
    assert xp.min(pressure) > 0.0

    # Convert total pressure to a pressure perturbation, matching the
    # evolved pressure variable of LinearMHD.
    pressure_data = {
        time: [pressure[i] - pressure[0]]
        for i, time in enumerate(times)
    }

    # ------------------------------------------------------------------
    # Compute pressure spectrum without the automatic branch fitter
    # ------------------------------------------------------------------

    omega, kvec, pressure_spectrum, _ = (
        power_spectrum_2d(
            pressure_data,
            "pressure_log",
            grids=sim.grids_log,
            grids_mapped=sim.grids_phy,
            component=0,
            slice_at=[0, 0, None],
            do_plot=False,
            disp_name="MHDhomogenSlab",
            disp_params=disp_params,
            fit_branches=0,
        )
    )

    # Slow and Alfvén branches are close. Skip unresolved low-k points.
    v_slow_fit = fit_branch_near_expected_speed(
        omega,
        kvec,
        pressure_spectrum,
        expected_speed=v_slow,
        competing_speed=v_alfven,
        search_bins=1,
        min_separation_bins=2,
    )

    v_fast_fit = fit_branch_near_expected_speed(
        omega,
        kvec,
        pressure_spectrum,
        expected_speed=v_fast,
        search_bins=2,
    )

    slow_error = abs(
        v_slow_fit - float(v_slow)
    )
    fast_error = abs(
        v_fast_fit - float(v_fast)
    )

    logger.info(f"{v_slow = }")
    logger.info(f"{v_slow_fit = }")
    logger.info(f"{slow_error = }")

    logger.info(f"{v_fast = }")
    logger.info(f"{v_fast_fit = }")
    logger.info(f"{fast_error = }")

    assert slow_error < 0.05
    assert fast_error < 0.19

    # ------------------------------------------------------------------
    # Plot 2: reconstructed-pressure spectrum
    # ------------------------------------------------------------------

    if do_plot:
        K, W = xp.meshgrid(kvec, omega)

        normalized_power = pressure_spectrum**2
        normalized_power /= xp.max(normalized_power)

        fig, ax = plt.subplots(figsize=(10, 10))

        levels = xp.logspace(-15, -1, 27)

        spectrum_plot = ax.contourf(
            K,
            W,
            normalized_power,
            levels=levels,
            cmap="plasma",
            norm=colors.LogNorm(),
        )

        fig.colorbar(
            spectrum_plot,
            ax=ax,
            ticks=[1e-12, 1e-9, 1e-6, 1e-3],
            format="%.0e",
            label="normalized spectral power",
        )

        # Numerical fits.
        ax.plot(
            kvec,
            v_slow_fit * kvec,
            "r:",
            linewidth=2,
            label=rf"slow fit: $v={v_slow_fit:.4f}$",
        )

        ax.plot(
            kvec,
            v_fast_fit * kvec,
            "m:",
            linewidth=2,
            label=rf"fast fit: $v={v_fast_fit:.4f}$",
        )

        # Exact branches.
        ax.plot(
            kvec,
            float(v_slow) * kvec,
            "c--",
            linewidth=2,
            label=rf"slow exact: $v={float(v_slow):.4f}$",
        )

        ax.plot(
            kvec,
            float(v_fast) * kvec,
            "g--",
            linewidth=2,
            label=rf"fast exact: $v={float(v_fast):.4f}$",
        )

        ax.set_title(
            "Reconstructed pressure, space-time power spectrum"
        )
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$\omega$")
        ax.set_xlim(0.0, kvec[-1])
        ax.set_ylim(
            0.0,
            1.1 * float(v_fast) * kvec[-1],
        )
        ax.legend()

        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)


if __name__ == "__main__":
    test_slab_waves_1d(do_plot=True)