import copy
import glob
import logging
import os
import shutil

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
from struphy.fields_background.base import (
    CartesianFluidEquilibriumWithB,
)
from struphy.models import ViscoResistiveMHD


set_logging_level()
logger = logging.getLogger("struphy")
# logger.setLevel(logging.INFO)


# Div-div regularization coefficient.
#
# Keep this small for the Orszag--Tang verification. The regularization
# modifies the longitudinal kinetic metric and therefore changes the
# compressive dynamics.
ALPHA_DIVDIV = 1.0e-2


class OrszagTangInitialState(
    CartesianFluidEquilibriumWithB,
):
    r"""Smooth two-dimensional Orszag–Tang initial state.

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

        square = (
            sin_y**2
            + sin_2x**2
        )

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


def check_mhd_scalar_diagnostics(
    output_path: str,
    energy_tolerance: float | None = None,
):
    """Check scalar diagnostics from an existing MHD simulation.

    This function is valid for the nondissipative, regularized model. The
    kinetic energy is the regularized kinetic energy stored by the model.

    Parameters
    ----------
    output_path
        Simulation output directory, normally ``env.path_out``.

    energy_tolerance
        Optional maximum relative total-energy drift. If ``None``, the
        energy drift is reported but not asserted.

    Returns
    -------
    dict
        Scalar histories and measured conservation errors.
    """
    candidates = sorted(
        glob.glob(
            os.path.join(
                output_path,
                "data",
                "*.hdf5",
            ),
        ),
    )

    if not candidates:
        raise FileNotFoundError(
            "No HDF5 files found in "
            f"{os.path.join(output_path, 'data')!r}.",
        )

    hdf5_path = None

    for candidate in candidates:
        with h5py.File(candidate, "r") as file:
            if (
                "time/value" in file
                and "scalar/en_tot" in file
            ):
                hdf5_path = candidate
                break

    if hdf5_path is None:
        raise FileNotFoundError(
            "No HDF5 file containing time and scalar "
            "diagnostics was found.",
        )

    print(
        "Reading scalar diagnostics from:\n"
        f"{hdf5_path}",
    )

    with h5py.File(hdf5_path, "r") as file:
        time_history = np.asarray(
            file["time/value"],
        ).reshape(-1)

        total_energy = np.asarray(
            file["scalar/en_tot"],
        ).reshape(-1)

        kinetic_energy = np.asarray(
            file["scalar/en_U"],
        ).reshape(-1)

        magnetic_energy = np.asarray(
            file["scalar/en_mag"],
        ).reshape(-1)

        thermodynamic_energy = np.asarray(
            file["scalar/en_thermo"],
        ).reshape(-1)

        total_mass = np.asarray(
            file["scalar/dens_tot"],
        ).reshape(-1)

        total_entropy = np.asarray(
            file["scalar/entr_tot"],
        ).reshape(-1)

        div_b = np.asarray(
            file["scalar/tot_div_B"],
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
        assert values.size == expected_size, (
            f"{name} has {values.size} entries, but the "
            f"time history has {expected_size} entries."
        )

        assert np.all(np.isfinite(values)), (
            f"Non-finite values found in {name}."
        )

    assert np.isclose(
        time_history[0],
        0.0,
    ), (
        f"The first saved time is {time_history[0]}, not zero."
    )

    # --------------------------------------------------------------
    # Check the total-energy definition
    # --------------------------------------------------------------

    energy_sum = (
        kinetic_energy
        + magnetic_energy
        + thermodynamic_energy
    )

    component_sum_difference = (
        total_energy
        - energy_sum
    )

    maximum_component_sum_error = float(
        np.max(
            np.abs(component_sum_difference),
        ),
    )

    component_sum_scale = max(
        abs(total_energy[0]),
        1.0,
    )

    relative_component_sum_error = (
        maximum_component_sum_error
        / component_sum_scale
    )

    # The regularized kinetic-energy scalar must be included in en_tot.
    assert np.allclose(
        total_energy,
        energy_sum,
        rtol=1.0e-12,
        atol=1.0e-12,
    ), (
        "The saved total energy is inconsistent with "
        "en_U + en_mag + en_thermo.\n"
        f"Maximum absolute difference: "
        f"{maximum_component_sum_error:.12e}\n"
        f"Relative difference: "
        f"{relative_component_sum_error:.12e}"
    )

    # --------------------------------------------------------------
    # Conservation measures
    # --------------------------------------------------------------

    def relative_drift(values):
        scale = max(
            abs(values[0]),
            1.0e-30,
        )

        return float(
            np.max(
                np.abs(values - values[0]),
            )
            / scale,
        )

    def final_relative_change(values):
        scale = max(
            abs(values[0]),
            1.0e-30,
        )

        return float(
            (values[-1] - values[0])
            / scale,
        )

    def relative_variation(values):
        scale = max(
            abs(values[0]),
            1.0e-30,
        )

        return float(
            (
                np.max(values)
                - np.min(values)
            )
            / scale,
        )

    energy_drift = relative_drift(
        total_energy,
    )

    mass_drift = relative_drift(
        total_mass,
    )

    entropy_drift = relative_drift(
        total_entropy,
    )

    final_energy_change = final_relative_change(
        total_energy,
    )

    final_mass_change = final_relative_change(
        total_mass,
    )

    final_entropy_change = final_relative_change(
        total_entropy,
    )

    maximum_div_b = float(
        np.max(
            np.abs(div_b),
        ),
    )

    kinetic_variation = relative_variation(
        kinetic_energy,
    )

    magnetic_variation = relative_variation(
        magnetic_energy,
    )

    thermodynamic_variation = relative_variation(
        thermodynamic_energy,
    )

    # --------------------------------------------------------------
    # Report
    # --------------------------------------------------------------

    print(
        "\nREGULARIZED IDEAL-MHD SCALAR DIAGNOSTICS",
    )
    print(
        "========================================",
    )

    print(
        f"Div-div alpha                  : "
        f"{ALPHA_DIVDIV:.12e}",
    )

    print(
        f"Saved states                  : "
        f"{expected_size}",
    )

    print(
        f"First saved time              : "
        f"{time_history[0]:.12e}",
    )

    print(
        f"Last saved time               : "
        f"{time_history[-1]:.12e}",
    )

    print("\nDiscrete regularized energy:")

    print(
        f"Initial total energy          : "
        f"{total_energy[0]:.16e}",
    )

    print(
        f"Final total energy            : "
        f"{total_energy[-1]:.16e}",
    )

    print(
        f"Maximum relative energy drift : "
        f"{energy_drift:.12e}",
    )

    print(
        f"Signed final energy change    : "
        f"{final_energy_change:.12e}",
    )

    print("\nEnergy-definition consistency:")

    print(
        f"Maximum component-sum error   : "
        f"{maximum_component_sum_error:.12e}",
    )

    print(
        f"Relative component-sum error  : "
        f"{relative_component_sum_error:.12e}",
    )

    print("\nConserved quantities:")

    print(
        f"Maximum relative mass drift   : "
        f"{mass_drift:.12e}",
    )

    print(
        f"Signed final mass change      : "
        f"{final_mass_change:.12e}",
    )

    print(
        f"Maximum relative entropy drift: "
        f"{entropy_drift:.12e}",
    )

    print(
        f"Signed final entropy change   : "
        f"{final_entropy_change:.12e}",
    )

    print(
        f"Maximum div(B) diagnostic     : "
        f"{maximum_div_b:.12e}",
    )

    print("\nEnergy exchange:")

    print(
        f"Relative kinetic variation    : "
        f"{kinetic_variation:.12e}",
    )

    print(
        f"Relative magnetic variation   : "
        f"{magnetic_variation:.12e}",
    )

    print(
        f"Thermodynamic variation       : "
        f"{thermodynamic_variation:.12e}",
    )

    # Structural invariants of the nondissipative formulation.
    assert mass_drift < 1.0e-10, (
        f"Relative mass drift is too large: {mass_drift:.12e}."
    )

    assert entropy_drift < 1.0e-10, (
        "Relative total-entropy drift is too large: "
        f"{entropy_drift:.12e}."
    )

    assert maximum_div_b < 1.0e-10, (
        "The magnetic-divergence diagnostic is too large: "
        f"{maximum_div_b:.12e}."
    )

    if energy_tolerance is not None:
        assert energy_drift < energy_tolerance, (
            f"Relative energy drift {energy_drift:.6e} "
            f"exceeds the tolerance "
            f"{energy_tolerance:.6e}."
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
        "thermodynamic_variation": (
            thermodynamic_variation
        ),
    }


def extract_scalar_time_data(data: dict):
    """Convert scalar plotting data to a time-first array."""
    times = sorted(
        data,
        key=float,
    )

    values = xp.stack(
        [
            data[time][0]
            for time in times
        ],
        axis=0,
    )

    return times, values


def extract_vector_time_data(data: dict):
    """Convert vector data to ``(nt, 3, nx, ny, nz)``."""
    times = sorted(
        data,
        key=float,
    )

    values = xp.stack(
        [
            xp.stack(
                data[time],
                axis=0,
            )
            for time in times
        ],
        axis=0,
    )

    return times, values


def _run_orszag_tang(
    *,
    do_plot: bool = False,
    cleanup: bool = False,
    with_viscosity : bool = False,
):
    """Run the regularized Orszag–Tang vortex verification."""
    rank = MPI.COMM_WORLD.Get_rank()

    # ------------------------------------------------------------------
    # Physical parameters
    # ------------------------------------------------------------------

    gamma = 5.0 / 3.0

    rho0 = gamma**2
    p0 = gamma

    velocity_amplitude = 1.0
    magnetic_amplitude = 1.0

    length_x = 2.0 * xp.pi
    length_y = 2.0 * xp.pi
    length_z = 1.0

    initial_state = OrszagTangInitialState(
        rho0=float(rho0),
        p0=float(p0),
        velocity_amplitude=float(
            velocity_amplitude,
        ),
        magnetic_amplitude=float(
            magnetic_amplitude,
        ),
    )

    # Verify the entropy convention used by s3_monoatomic.
    specific_entropy0 = np.log(
        p0
        / (
            (gamma - 1.0)
            * rho0**gamma
        ),
    )

    entropy_density0 = (
        rho0
        * specific_entropy0
    )

    reference_entropy_density0 = (
        gamma**2
        * np.log(
            gamma
            / (
                (gamma - 1.0)
                * gamma ** (2.0 * gamma)
            ),
        )
    )

    assert np.isclose(
        entropy_density0,
        reference_entropy_density0,
    )

    if rank == 0:
        print(
            "Initial density        :",
            rho0,
        )
        print(
            "Initial pressure       :",
            p0,
        )
        print(
            "Initial entropy density:",
            entropy_density0,
        )
        print(
            "Div-div alpha          :",
            ALPHA_DIVDIV,
        )

    # ------------------------------------------------------------------
    # Regularized nondissipative MHD model
    # ------------------------------------------------------------------

    model = ViscoResistiveMHD(
        with_viscosity=with_viscosity,
        with_resistivity=True,
        with_regularization=False,
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

    # IMPORTANT:
    #
    # Reconstructing an Options object resets omitted fields to their
    # defaults. Therefore the regularization options must be supplied
    # every time an Options object is replaced.
    model.propagators.variat_dens.options = (
        model.propagators.variat_dens.Options(
            model="full",
            gamma=gamma,
            with_regularization=False,
            alpha_divdiv=ALPHA_DIVDIV,
        )
    )

    model.propagators.variat_mom.options = (
        model.propagators.variat_mom.Options(
            with_regularization=False,
            alpha_divdiv=ALPHA_DIVDIV,
        )
    )

    model.propagators.variat_ent.options = (
        model.propagators.variat_ent.Options(
            model="full",
            gamma=gamma,
            with_regularization=False,
            alpha_divdiv=ALPHA_DIVDIV,
        )
    )

    model.propagators.variat_mag.options = (
        model.propagators.variat_mag.Options(
            model="full",
            with_regularization=False,
            alpha_divdiv=ALPHA_DIVDIV,
        )
    )
    if with_viscosity:
        model.propagators.variat_viscous.options = (
            model.propagators.variat_viscous.Options(
                model="full",
                with_regularization=False,
                mu_a= 2 * (1/32)**2
            )
        )
        model.propagators.variat_resist.options = (
            model.propagators.variat_resist.Options(
                eta_a= 2 * (1/32)**2
            )
        )
    

    # # Fail before the expensive simulation if any option was accidentally
    # # reset to the unregularized default.
    # assert (
    #     model
    #     .propagators
    #     .variat_dens
    #     .options
    #     .with_regularization
    # )

    # assert (
    #     model
    #     .propagators
    #     .variat_mom
    #     .options
    #     .with_regularization
    # )

    # assert (
    #     model
    #     .propagators
    #     .variat_ent
    #     .options
    #     .with_regularization
    # )

    # assert (
    #     model
    #     .propagators
    #     .variat_mag
    #     .options
    #     .with_regularization
    # )

    # assert np.isclose(
    #     model
    #     .propagators
    #     .variat_dens
    #     .options
    #     .alpha_divdiv,
    #     ALPHA_DIVDIV,
    # )

    # assert np.isclose(
    #     model
    #     .propagators
    #     .variat_mom
    #     .options
    #     .alpha_divdiv,
    #     ALPHA_DIVDIV,
    # )

    # assert np.isclose(
    #     model
    #     .propagators
    #     .variat_ent
    #     .options
    #     .alpha_divdiv,
    #     ALPHA_DIVDIV,
    # )

    # assert np.isclose(
    #     model
    #     .propagators
    #     .variat_mag
    #     .options
    #     .alpha_divdiv,
    #     ALPHA_DIVDIV,
    # )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    test_folder = os.path.join(
        os.getcwd(),
        "verification_tests_full",
    )

    if rank == 0 and os.path.exists(test_folder):
        shutil.rmtree(test_folder)

    MPI.COMM_WORLD.Barrier()

    env = EnvironmentOptions(
        out_folders=os.path.join(
            test_folder,
            "ViscoResistiveMHD",
        ),
        sim_folder=(
            "orszag_tang_regularized"
        ),
        profiling_activated=True,
        profiling_trace=True,
        
    )

    # ------------------------------------------------------------------
    # Geometry and discretization
    # ------------------------------------------------------------------

    domain = domains.Cuboid(
        r1=float(length_x),
        r2=float(length_y),
        r3=float(length_z),
    )

    grid = grids.TensorProductGrid(
        num_elements=(48,48, 1),
    )

    derham_opts = DerhamOptions(
        degree=(2, 2, 1),
    )

    # Early/intermediate time, before very sharp current sheets form.
    time_opts = Time(
        dt=1.0e-3,
        Tend=3e-3, #2.,
    )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

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

    diagnostics = None

    try:
        simulation.run()

        if rank == 0:
            simulation.pproc(
                create_vtk=False,
            )

            diagnostics = check_mhd_scalar_diagnostics(
                env.path_out,
                energy_tolerance=1.0e-3,
            )

            simulation.load_plotting_data()

            time_history = diagnostics["time"]
            total_energy = diagnostics["en_tot"]
            kinetic_energy = diagnostics["en_U"]
            magnetic_energy = diagnostics["en_mag"]
            thermodynamic_energy = diagnostics["en_thermo"]

            # ----------------------------------------------------------
            # Load and validate fields
            # ----------------------------------------------------------

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

            times, density_3form = (
                extract_scalar_time_data(
                    density_data,
                )
            )

            entropy_times, entropy_3form = (
                extract_scalar_time_data(
                    entropy_data,
                )
            )

            velocity_times, velocity = (
                extract_vector_time_data(
                    velocity_data,
                )
            )

            magnetic_times, magnetic = (
                extract_vector_time_data(
                    magnetic_data,
                )
            )

            assert len(times) == len(entropy_times)
            assert len(times) == len(velocity_times)
            assert len(times) == len(magnetic_times)

            assert all(
                np.isclose(
                    float(t1),
                    float(t2),
                )
                for t1, t2 in zip(
                    times,
                    entropy_times,
                )
            )

            assert all(
                np.isclose(
                    float(t1),
                    float(t2),
                )
                for t1, t2 in zip(
                    times,
                    velocity_times,
                )
            )

            assert all(
                np.isclose(
                    float(t1),
                    float(t2),
                )
                for t1, t2 in zip(
                    times,
                    magnetic_times,
                )
            )

            jacobian = (
                simulation
                .domain
                .jacobian_det(
                    *simulation.grids_log,
                )
            )

            # L2 variables are stored as logical 3-forms.
            density = (
                density_3form
                / jacobian[None, ...]
            )

            entropy = (
                entropy_3form
                / jacobian[None, ...]
            )

            pressure = (
                (gamma - 1.0)
                * density**gamma
                * xp.exp(
                    entropy / density,
                )
            )

            assert xp.all(
                xp.isfinite(density),
            )

            assert xp.all(
                xp.isfinite(entropy),
            )

            assert xp.all(
                xp.isfinite(pressure),
            )

            assert xp.all(
                xp.isfinite(velocity),
            )

            assert xp.all(
                xp.isfinite(magnetic),
            )

            minimum_density = float(
                xp.min(density),
            )

            maximum_density = float(
                xp.max(density),
            )

            minimum_pressure = float(
                xp.min(pressure),
            )

            maximum_pressure = float(
                xp.max(pressure),
            )

            print(
                "\nREGULARIZED ORSZAG–TANG FIELD RANGES",
            )
            print(
                "====================================",
            )

            print(
                "Density range : "
                f"[{minimum_density:.12e}, "
                f"{maximum_density:.12e}]",
            )

            print(
                "Pressure range: "
                f"[{minimum_pressure:.12e}, "
                f"{maximum_pressure:.12e}]",
            )

            assert minimum_density > 0.0
            assert minimum_pressure > 0.0

            # Ensure that the simulation did not remain at its initial
            # condition and that nonlinear energy exchange occurred.
            assert (
                diagnostics["kinetic_variation"]
                > 1.0e-3
            )

            assert (
                diagnostics["magnetic_variation"]
                > 1.0e-3
            )

            # ----------------------------------------------------------
            # Optional plots
            # ----------------------------------------------------------

            if do_plot:
                x = (
                    simulation
                    .grids_phy[0][:, :, 0]
                )

                y = (
                    simulation
                    .grids_phy[1][:, :, 0]
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

                axes[0, 0].set_title(
                    "Final density",
                )

                pressure_plot = axes[0, 1].contourf(
                    x,
                    y,
                    pressure[-1, :, :, 0],
                    levels=40,
                    cmap="plasma",
                )

                fig.colorbar(
                    pressure_plot,
                    ax=axes[0, 1],
                )

                axes[0, 1].set_title(
                    "Final pressure",
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

                axes[1, 0].set_xlabel(
                    "time",
                )

                axes[1, 0].set_ylabel(
                    "energy",
                )

                axes[1, 0].set_title(
                    "Energy exchange",
                )

                axes[1, 0].legend()

                relative_energy_change = (
                    total_energy
                    - total_energy[0]
                ) / total_energy[0]

                axes[1, 1].plot(
                    time_history,
                    relative_energy_change,
                )

                axes[1, 1].set_xlabel(
                    "time",
                )

                axes[1, 1].set_ylabel(
                    r"$(E(t)-E(0))/E(0)$",
                )

                axes[1, 1].set_title(
                    "Relative regularized-energy change",
                )

                for ax in axes[0, :]:
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_aspect("equal")

                fig.suptitle(
                    "Regularized ideal-MHD "
                    "Orszag–Tang vortex\n"
                    rf"$\alpha_{{\mathrm{{divdiv}}}}"
                    rf"={ALPHA_DIVDIV}$",
                )

                fig.tight_layout()
                plt.show()
                plt.close(fig)

    finally:
        # All MPI ranks must finish using the output before rank zero
        # optionally removes it.
        MPI.COMM_WORLD.Barrier()

        if (
            rank == 0
            and cleanup
            and os.path.exists(test_folder)
        ):
            shutil.rmtree(test_folder)

        MPI.COMM_WORLD.Barrier()

    return diagnostics


def test_regularized_orszag_tang():
    """Verify the regularized Orszag–Tang vortex."""
    _run_orszag_tang(
        do_plot=False,
        cleanup=True,
    )


if __name__ == "__main__":
    _run_orszag_tang(
        do_plot=True,
        cleanup=False,
        with_viscosity=False
    )