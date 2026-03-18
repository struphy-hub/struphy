import logging
import os
import shutil

import cunumpy as xp
import pytest
from matplotlib import pyplot as plt
from scipy.special import jv, yn

from feectools.ddm.mpi import mpi as MPI
from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
)
from struphy.diagnostics.diagn_tools import power_spectrum_2d
from struphy.models import Maxwell

logger = logging.getLogger("struphy")


@pytest.mark.parametrize("algo", ["implicit", "explicit"])
def test_light_wave_1d(algo: str, do_plot: bool = False):
    # light-weight model instance
    model = Maxwell()

    # set environment options
    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "Maxwell")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="light_wave_1d")

    # time stepping
    time_opts = Time(dt=0.05, Tend=50.0)

    # geometry
    domain = domains.Cuboid(r3=20.0)

    # grid
    grid = grids.TensorProductGrid(Nel=(1, 1, 128))

    # derham options
    derham_opts = DerhamOptions(p=(1, 1, 3))

    # propagator options
    model.propagators.maxwell.options = model.propagators.maxwell.Options(algo=algo)

    # initial conditions (background + perturbation)
    model.em_fields.e_field.add_perturbation(perturbations.Noise(amp=0.1, comp=0, seed=123))
    model.em_fields.e_field.add_perturbation(perturbations.Noise(amp=0.1, comp=1, seed=123))

    # instance of simulation
    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
        verbose=True,
    )

    # run
    sim.run(verbose=True)

    # post processing
    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.pproc(verbose=True)

    # diagnostics
    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.load_plotting_data(verbose=True)

        # fft
        E_of_t = sim.spline_values.em_fields.e_field_log.data
        _1, _2, _3, coeffs = power_spectrum_2d(
            E_of_t,
            "e_field_log",
            grids=sim.grids_log,
            grids_mapped=sim.grids_phy,
            component=0,
            slice_at=[0, 0, None],
            do_plot=do_plot,
            disp_name="Maxwell1D",
            fit_branches=1,
            noise_level=0.5,
            extr_order=10,
            fit_degree=(1,),
        )

        # assert
        c_light_speed = 1.0
        assert xp.abs(coeffs[0][0] - c_light_speed) < 0.02

        shutil.rmtree(test_folder)


def test_coaxial(do_plot: bool = False):
    # light-weight model instance
    model = Maxwell()

    # environment options
    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "Maxwell")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="coaxial")

    # time
    time_opts = Time(dt=0.05, Tend=10.0)

    # geometry
    a1 = 2.326744
    a2 = 3.686839
    Lz = 2.0
    domain = domains.HollowCylinder(a1=a1, a2=a2, Lz=Lz)

    # fluid equilibrium (can be used as part of initial conditions)
    equil = equils.HomogenSlab()

    # grid
    grid = grids.TensorProductGrid(Nel=(32, 64, 1))

    # derham options
    derham_opts = DerhamOptions(
        p=(3, 3, 1),
        spl_kind=(False, True, True),
        dirichlet_bc=((True, True), (False, False), (False, False)),
    )

    # propagator options
    model.propagators.maxwell.options = model.propagators.maxwell.Options(algo="implicit")

    # initial conditions (background + perturbation)
    m = 3
    model.em_fields.e_field.add_perturbation(perturbations.CoaxialWaveguideElectric_r(m=m, a1=a1, a2=a2))
    model.em_fields.e_field.add_perturbation(perturbations.CoaxialWaveguideElectric_theta(m=m, a1=a1, a2=a2))
    model.em_fields.b_field.add_perturbation(perturbations.CoaxialWaveguideMagnetic(m=m, a1=a1, a2=a2))

    # instance of simulation
    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
        verbose=True,
    )

    # run
    sim.run(verbose=True)

    # post processing
    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.pproc(physical=True, verbose=True)

    # diagnostics
    if MPI.COMM_WORLD.Get_rank() == 0:
        # get parameters
        dt = time_opts.dt
        split_algo = time_opts.split_algo
        Nel = grid.Nel
        modes = m

        # load data
        sim.load_plotting_data(verbose=True)

        t_grid = sim.t_grid
        grids_phy = sim.grids_phy
        e_field_phy = sim.spline_values.em_fields.e_field_phy.data
        b_field_phy = sim.spline_values.em_fields.b_field_phy.data

        X = grids_phy[0][:, :, 0]
        Y = grids_phy[1][:, :, 0]

        # define analytic solution
        def B_z(X, Y, Z, m, t):
            """Magnetic field in z direction of coaxial cabel"""
            r = (X**2 + Y**2) ** 0.5
            theta = xp.arctan2(Y, X)
            return (jv(m, r) - 0.28 * yn(m, r)) * xp.cos(m * theta - t)

        def E_r(X, Y, Z, m, t):
            """Electrical field in radial direction of coaxial cabel"""
            r = (X**2 + Y**2) ** 0.5
            theta = xp.arctan2(Y, X)
            return -m / r * (jv(m, r) - 0.28 * yn(m, r)) * xp.cos(m * theta - t)

        def E_theta(X, Y, Z, m, t):
            """Electrical field in azimuthal direction of coaxial cabel"""
            r = (X**2 + Y**2) ** 0.5
            theta = xp.arctan2(Y, X)
            return ((m / r * jv(m, r) - jv(m + 1, r)) - 0.28 * (m / r * yn(m, r) - yn(m + 1, r))) * xp.sin(
                m * theta - t,
            )

        def to_E_r(X, Y, E_x, E_y):
            r = (X**2 + Y**2) ** 0.5
            theta = xp.arctan2(Y, X)
            return xp.cos(theta) * E_x + xp.sin(theta) * E_y

        def to_E_theta(X, Y, E_x, E_y):
            r = (X**2 + Y**2) ** 0.5
            theta = xp.arctan2(Y, X)
            return -xp.sin(theta) * E_x + xp.cos(theta) * E_y

        # plot
        if do_plot:
            vmin = E_theta(X, Y, grids_phy[0], modes, 0).min()
            vmax = E_theta(X, Y, grids_phy[0], modes, 0).max()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            plot_exac = ax1.contourf(
                X,
                Y,
                E_theta(X, Y, grids_phy[0], modes, t_grid[-1]),
                cmap="plasma",
                levels=100,
                vmin=vmin,
                vmax=vmax,
            )
            ax2.contourf(
                X,
                Y,
                to_E_theta(X, Y, e_field_phy[t_grid[-1]][0][:, :, 0], e_field_phy[t_grid[-1]][1][:, :, 0]),
                cmap="plasma",
                levels=100,
                vmin=vmin,
                vmax=vmax,
            )
            fig.colorbar(plot_exac, ax=[ax1, ax2], orientation="vertical", shrink=0.9)
            ax1.set_xlabel("Exact")
            ax2.set_xlabel("Numerical")
            fig.suptitle(f"Exact and Simulated $E_\\theta$ Field {dt=}, {split_algo=}, {Nel=}", fontsize=14)
            plt.show()

        # assert
        Ex_tend = e_field_phy[t_grid[-1]][0][:, :, 0]
        Ey_tend = e_field_phy[t_grid[-1]][1][:, :, 0]
        Er_exact = E_r(X, Y, grids_phy[0], modes, t_grid[-1])
        Etheta_exact = E_theta(X, Y, grids_phy[0], modes, t_grid[-1])
        Bz_tend = b_field_phy[t_grid[-1]][2][:, :, 0]
        Bz_exact = B_z(X, Y, grids_phy[0], modes, t_grid[-1])

        error_Er = xp.max(xp.abs((to_E_r(X, Y, Ex_tend, Ey_tend) - Er_exact)))
        error_Etheta = xp.max(xp.abs((to_E_theta(X, Y, Ex_tend, Ey_tend) - Etheta_exact)))
        error_Bz = xp.max(xp.abs((Bz_tend - Bz_exact)))

        rel_err_Er = error_Er / xp.max(xp.abs(Er_exact))
        rel_err_Etheta = error_Etheta / xp.max(xp.abs(Etheta_exact))
        rel_err_Bz = error_Bz / xp.max(xp.abs(Bz_exact))

        logger.info("")
        assert rel_err_Bz < 0.0021, f"Assertion for magnetic field Maxwell failed: {rel_err_Bz =}"
        logger.info(f"Assertion for magnetic field Maxwell passed ({rel_err_Bz =}).")
        assert rel_err_Etheta < 0.0021, f"Assertion for electric (E_theta) field Maxwell failed: {rel_err_Etheta =}"
        logger.info(f"Assertion for electric field Maxwell passed ({rel_err_Etheta =}).")
        assert rel_err_Er < 0.0021, f"Assertion for electric (E_r) field Maxwell failed: {rel_err_Er =}"
        logger.info(f"Assertion for electric field Maxwell passed ({rel_err_Er =}).")

        shutil.rmtree(test_folder)


if __name__ == "__main__":
    test_light_wave_1d(algo="explicit", do_plot=True)
    test_coaxial(do_plot=True)
