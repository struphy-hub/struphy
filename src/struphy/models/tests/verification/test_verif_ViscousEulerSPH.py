import logging
import os
import shutil

import cunumpy as xp
import pytest
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from struphy import (
    BaseUnits,
    BinningPlot,
    BoundaryParameters,
    EnvironmentOptions,
    KernelDensityPlot,
    LoadingParameters,
    SavingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    equils,
    perturbations,
    set_logging_level,
)
from struphy.models import ViscousEulerSPH
from struphy.initial.base import GenericPerturbation

logger = logging.getLogger("struphy")
set_logging_level(logging.INFO)


@pytest.mark.parametrize("nx", [12, 24])
@pytest.mark.parametrize("plot_pts", [11, 32])
def test_soundwave_1d(nx: int, plot_pts: int, do_plot: bool = False):
    """Verification test for SPH discretization of isthermal Euler equations.
    A standing sound wave with c_s=1 traveserses the domain once.
    """

    # environment options
    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "ViscousEulerSPH")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="soundwave_1d")

    # time stepping
    time_opts = Time(dt=0.03125, Tend=2.5, split_algo="Strang")

    # geometry
    r1 = 2.5
    domain = domains.Cuboid(r1=r1)

    # grid
    grid = None

    # derham options
    derham_opts = None

    # light-weight model instance
    model = ViscousEulerSPH(with_B0=False, with_viscosity=False)

    loading_params = LoadingParameters(ppb=8, loading="tesselation")
    weights_params = WeightsParameters()
    boundary_params = BoundaryParameters()
    sorting_params = SortingParameters(
        boxes_per_dim=(nx, 1, 1),
        dims_mask=(True, False, False),
    )

    bin_plot = BinningPlot(slice="e1", n_bins=(32,), ranges=(0.0, 1.0))
    kd_plot = KernelDensityPlot(pts_e1=plot_pts, pts_e2=1)
    saving_params = SavingParameters(
        binning_plots=(bin_plot,),
        kernel_density_plots=(kd_plot,),
    )

    model.euler_fluid.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        saving_params=saving_params,
    )

    # propagator options
    from struphy.ode.utils import ButcherTableau

    butcher = ButcherTableau(algo="forward_euler")
    model.propagators.push_eta.options = model.propagators.push_eta.Options(butcher=butcher)
    if model.with_B0:
        model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
    model.propagators.push_sph_p.options = model.propagators.push_sph_p.Options(kernel_type="gaussian_1d")

    # background, perturbations and initial conditions
    background = equils.ConstantVelocity()
    model.euler_fluid.var.add_background(background)
    perturbation = perturbations.ModesSin(ls=(1,), amps=(1.0e-2,))
    model.euler_fluid.var.add_perturbation(del_n=perturbation)

    # instance of simulation
    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )

    # run
    sim.run()

    # post processing
    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.pproc()

        # diagnostics
        sim.load_plotting_data()

        ee1, ee2, ee3 = sim.n_sph.euler_fluid.view_0.grid_n_sph
        n_sph = sim.n_sph.euler_fluid.view_0.n_sph

        if do_plot:
            ppb = 8
            dt = time_opts.dt
            end_time = time_opts.Tend
            Nt = int(end_time // dt)
            x = ee1 * r1

            plt.figure(figsize=(10, 8))
            interval = Nt / 10
            plot_ct = 0
            for i in range(0, Nt + 1):
                if i % interval == 0:
                    logger.info(f"{i =}")
                    plot_ct += 1
                    ax = plt.gca()

                    if plot_ct <= 6:
                        style = "-"
                    else:
                        style = "."
                    plt.plot(x.squeeze(), n_sph[i, :, 0, 0], style, label=f"time={i * dt:4.2f}")
                    plt.xlim(0, 2.5)
                    plt.legend()
                    ax.set_xticks(xp.linspace(0, 2.5, nx + 1))
                    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                    plt.grid(c="k")
                    plt.xlabel("x")
                    plt.ylabel(r"$\rho$")

                    plt.title(f"standing sound wave ($c_s = 1$) for {nx =} and {ppb =}")
                if plot_ct == 11:
                    break

            plt.show()

        error = xp.max(xp.abs(n_sph[0] - n_sph[-1]))
        logger.info(f"SPH sound wave {error =}.")
        assert error < 6e-4
        logger.info("Assertion passed.")

        shutil.rmtree(test_folder)


@pytest.mark.parametrize("nx", [8])
@pytest.mark.parametrize("plot_pts", [11])
def test_velocity_diffusion(nx: int, plot_pts: int, do_plot: bool = False):
    """Verification test for SPH discretization of isthermal Euler equations.
    A standing sound wave with c_s=1 is damped at the rate mu*k^2/2 by viscosity.
    """

    # environment options
    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "ViscousEulerSPH")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="velocity_diffusion")

    # time stepping
    time_opts = Time(dt=0.0025, Tend=0.1, split_algo="Strang")

    # geometry
    r1 = 1.0
    domain = domains.Cuboid(r1=r1)

    # grid
    grid = None

    # derham options
    derham_opts = None

    # light-weight model instance
    model = ViscousEulerSPH(with_B0=False, with_p=False, with_viscosity=True)

    ppb = 100  # Particles per box (controls resolution)
    loading_params = LoadingParameters(ppb=ppb, loading="tesselation")
    weights_params = WeightsParameters()
    boundary_params = BoundaryParameters()
    sorting_params = SortingParameters(
        boxes_per_dim=(nx, 1, 1),
        dims_mask=(True, False, False),
    )

    bin_plot = BinningPlot(slice="e1", n_bins=(16,), ranges=(0.0, 1.0))
    bin_plot_j1 = BinningPlot(slice="e1", n_bins=(16,), ranges=(0.0, 1.0), output_quantity="current_1")
    kd_plot = KernelDensityPlot(pts_e1=plot_pts, pts_e2=1)
    saving_params = SavingParameters(
        binning_plots=(bin_plot, bin_plot_j1),
        kernel_density_plots=(kd_plot,),
    )

    model.euler_fluid.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        saving_params=saving_params,
    )

    # propagator options
    from struphy.ode.utils import ButcherTableau

    butcher = ButcherTableau(algo="forward_euler")
    model.propagators.push_eta.options = model.propagators.push_eta.Options(butcher=butcher)
    
    mu = 1.0
    model.propagators.push_viscous.options = model.propagators.push_viscous.Options(kernel_type="gaussian_1d", mu=mu)
    
    if model.with_B0:
        model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
    if model.with_p:
        model.propagators.push_sph_p.options = model.propagators.push_sph_p.Options(kernel_type="gaussian_1d")
        

    # background, perturbations and initial conditions
    ux_mean = 0.0
    background = equils.ConstantVelocity(ux=ux_mean)
    model.euler_fluid.var.add_background(background)
    perturbation = perturbations.ModesSin(ls=(1,), amps=(0.5,))
    # perturbation = GenericPerturbation(fun=lambda e1, e2, e3: 0.2*xp.exp(-20 * (e1 - 0.5) ** 2))
    model.euler_fluid.var.add_perturbation(del_u1=perturbation)

    # instance of simulation
    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )

    # run
    sim.run()

    # post processing
    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.pproc()

        # diagnostics
        sim.load_plotting_data()

        ee1, ee2, ee3 = sim.n_sph.euler_fluid.view_0.grid_n_sph
        n_sph = sim.n_sph.euler_fluid.view_0.n_sph
        e1_binned = sim.f.euler_fluid.e1_density.grid_e1
        n_binned = sim.f.euler_fluid.e1_density.f_binned
        j1_binned = sim.f.euler_fluid.e1_current_1.f_binned
        print(f"{e1_binned.shape = }")
        print(f"{n_binned.shape = }")
        print(f"{j1_binned.shape = }")

        import numpy as np

        dt = time_opts.dt
        Nt = int(time_opts.Tend // dt)
        times = np.linspace(0.0, time_opts.Tend, Nt + 1)

        # sin(2*pi*x/r1) for mode l=1 peaks at x = 0.25*r1
        e1_np = np.asarray(e1_binned).flatten()
        idx_max = int(np.argmin(np.abs(e1_np - 0.25 * r1)))

        # amplitude time series at the peak bin
        amplitude = np.asarray(j1_binned[:, idx_max]).flatten()

        # analytical decay rate: gamma = mu * k^2, k = 2*pi/r1 for mode l=1
        k = 2.0 * np.pi / r1
        gamma_analytical = mu * 4/3 * k**2

        A0 = amplitude[0]
        amplitude_analytical = A0 * np.exp(-gamma_analytical * times)

        # numerical decay rate via linear fit to log(amplitude)
        log_amp = np.log(np.abs(amplitude) + 1e-15)
        coeffs = np.polyfit(times, log_amp, 1)
        gamma_numerical = -coeffs[0]

        logger.info(f"Analytical decay rate: gamma = mu*k^2 = {gamma_analytical:.4f}")
        logger.info(f"Numerical  decay rate: gamma           = {gamma_numerical:.4f}")

        if do_plot:
            dt = time_opts.dt
            end_time = time_opts.Tend
            Nt = int(end_time // dt)
            x = ee1 * r1

            plt.figure(figsize=(20, 40))

            plot_interval = 4
            n_rows = 8
            plot_ct = 0
            time = 0.0
            for i in range(Nt + 1):
                time = dt * i
                if i % plot_interval == 0:
                    logger.info(f"{i =}, {time =:.4f}, {plot_ct =}")
                    plt.subplot(n_rows, 3, plot_ct + 1)
                    plt.plot(x.squeeze(), n_sph[i, :, 0, 0], label=f"n_sph at time={time:.4f}", linewidth=2)
                    plt.xlim(0, r1)
                    plt.grid(c="k", linestyle="--")
                    plt.xlabel("x")
                    plt.ylim([0.8, 1.2])
                    # plt.title(f"n_sph at time={time:.4f}")
                    plt.legend()

                    plt.subplot(n_rows, 3, plot_ct + 2)
                    plt.plot(e1_binned, n_binned[i, :], label=f"n_binned at time={i * dt:4.2f}", linewidth=2)
                    plt.xlim(0, r1)
                    plt.grid(c="k", linestyle="--")
                    plt.xlabel("x")
                    plt.ylim([0.8, 1.2])
                    # plt.title(f"n_binned at time={i * dt:4.2f}")
                    plt.legend()

                    plt.subplot(n_rows, 3, plot_ct + 3)
                    plt.plot(e1_binned, j1_binned[i, :], label=f"j1_binned at time={i * dt:4.2f}", linewidth=2)
                    plt.xlim(0, r1)
                    plt.grid(c="k", linestyle="--")
                    plt.xlabel("x")
                    plt.ylim([ux_mean - 0.5, ux_mean + 0.5])
                    # plt.title(f"j1_binned at time={i * dt:4.2f}")
                    plt.legend()

                    plot_ct += 3
                    if plot_ct == n_rows * 3:
                        break

            plt.show()

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.semilogy(times, np.abs(amplitude), "o-", markersize=3,
                        label=f"Numerical (fitted rate = {gamma_numerical:.3f})")
            ax.semilogy(times, np.abs(amplitude_analytical), "--",
                        label=rf"Analytical: $\gamma = (4/3) \mu k^2 = {gamma_analytical:.3f}$")
            ax.set_xlabel("time")
            ax.set_ylabel(rf"velocity amplitude at $x = {e1_np[idx_max]:.3f}$")
            ax.set_title("Velocity diffusion: amplitude decay over time")
            ax.legend()
            ax.grid(True, which="both")
            plt.tight_layout()
            plt.show()

        error = xp.max(xp.abs(j1_binned[-1] - ux_mean))
        logger.info(f"SPH sound wave {error =}.")
        assert error < 0.0022
        logger.info("Assertion passed.")

        rel_error = abs(gamma_numerical - gamma_analytical) / gamma_analytical
        logger.info(f"Relative error in decay rate: {rel_error * 100:.2f}%")
        assert rel_error < 0.04, (
            f"Numerical decay rate {gamma_numerical:.4f} deviates {rel_error * 100:.1f}% "
            f"from analytical {gamma_analytical:.4f} (tolerance 4%)"
        )
        logger.info("Decay rate assertion passed.")

        shutil.rmtree(test_folder)


@pytest.mark.parametrize("nx", [8])
@pytest.mark.parametrize("plot_pts", [11])
def test_damped_sound_wave(nx: int, plot_pts: int, do_plot: bool = False):
    """Verification test for SPH discretization of isthermal Euler equations.
    A standing sound wave with c_s=1 is damped at the rate mu*k^2/2 by viscosity.
    """

    # environment options
    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "ViscousEulerSPH")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="damped_soundwave_1d")

    # time stepping
    time_opts = Time(dt=0.05, Tend=4.0, split_algo="Strang")

    # geometry
    r1 = 1.0
    domain = domains.Cuboid(r1=r1)

    # grid
    grid = None

    # derham options
    derham_opts = None

    # light-weight model instance
    model = ViscousEulerSPH(with_B0=False, with_p=True, with_viscosity=True)

    ppb = 100  # Particles per box (controls resolution)
    loading_params = LoadingParameters(ppb=ppb, loading="tesselation")
    weights_params = WeightsParameters()
    boundary_params = BoundaryParameters()
    sorting_params = SortingParameters(
        boxes_per_dim=(nx, 1, 1),
        dims_mask=(True, False, False),
    )

    bin_plot = BinningPlot(slice="e1", n_bins=(16,), ranges=(0.0, 1.0))
    bin_plot_j1 = BinningPlot(slice="e1", n_bins=(16,), ranges=(0.0, 1.0), output_quantity="current_1")
    kd_plot = KernelDensityPlot(pts_e1=plot_pts, pts_e2=1)
    saving_params = SavingParameters(
        binning_plots=(bin_plot, bin_plot_j1),
        kernel_density_plots=(kd_plot,),
    )

    model.euler_fluid.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        saving_params=saving_params,
    )

    # propagator options
    from struphy.ode.utils import ButcherTableau

    butcher = ButcherTableau(algo="forward_euler")
    model.propagators.push_eta.options = model.propagators.push_eta.Options(butcher=butcher)
    
    mu = 0.05
    model.propagators.push_viscous.options = model.propagators.push_viscous.Options(kernel_type="gaussian_1d", mu=mu)
    
    if model.with_B0:
        model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
    if model.with_p:
        model.propagators.push_sph_p.options = model.propagators.push_sph_p.Options(kernel_type="gaussian_1d")
        

    # background, perturbations and initial conditions
    background = equils.ConstantVelocity()
    model.euler_fluid.var.add_background(background)
    perturbation = perturbations.ModesSin(ls=(1,), amps=(1e-2,))
    model.euler_fluid.var.add_perturbation(del_n=perturbation)

    # instance of simulation
    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )

    # run
    sim.run()

    # post processing
    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.pproc()

        # diagnostics
        sim.load_plotting_data()

        ee1, ee2, ee3 = sim.n_sph.euler_fluid.view_0.grid_n_sph
        n_sph = sim.n_sph.euler_fluid.view_0.n_sph
        e1_binned = sim.f.euler_fluid.e1_density.grid_e1
        n_binned = sim.f.euler_fluid.e1_density.f_binned
        j1_binned = sim.f.euler_fluid.e1_current_1.f_binned
        print(f"{e1_binned.shape = }")
        print(f"{n_binned.shape = }")
        print(f"{j1_binned.shape = }")

        import numpy as np

        dt = time_opts.dt
        Nt = len(sim.t_grid) - 1
        times = np.linspace(0.0, time_opts.Tend, Nt + 1)

        # sin(2*pi*x/r1) for mode l=1 peaks at x = 0.25*r1
        e1_np = np.asarray(e1_binned).flatten()
        idx_max = int(np.argmin(np.abs(e1_np - 0.25 * r1)))

        # amplitude time series at the peak bin
        amplitude = np.asarray(j1_binned[:, idx_max]).flatten()

        # analytical decay rate: gamma = mu * k^2, k = 2*pi/r1 for mode l=1
        k = 2.0 * np.pi / r1
        gamma_analytical = mu * 4/3 * k**2

        A0 = amplitude[0]
        amplitude_analytical = A0 * np.exp(-gamma_analytical * times)

        # numerical decay rate via linear fit to log(amplitude)
        log_amp = np.log(np.abs(amplitude) + 1e-15)
        print(f"{times.shape = }")
        print(f"{log_amp.shape = }")
        coeffs = np.polyfit(times, log_amp, 1)
        gamma_numerical = -coeffs[0]

        logger.info(f"Analytical decay rate: gamma = mu*k^2 = {gamma_analytical:.4f}")
        logger.info(f"Numerical  decay rate: gamma           = {gamma_numerical:.4f}")

        if do_plot:
            dt = time_opts.dt
            end_time = time_opts.Tend
            Nt = int(end_time // dt)
            x = ee1 * r1

            plt.figure(figsize=(20, 40))

            plot_interval = 4
            n_rows = 8
            plot_ct = 0
            time = 0.0
            for i in range(Nt + 1):
                time = dt * i
                if i % plot_interval == 0:
                    logger.info(f"{i =}, {time =:.4f}, {plot_ct =}")
                    plt.subplot(n_rows, 3, plot_ct + 1)
                    plt.plot(x.squeeze(), n_sph[i, :, 0, 0], label=f"n_sph at time={time:.4f}", linewidth=2)
                    plt.xlim(0, r1)
                    plt.grid(c="k", linestyle="--")
                    plt.xlabel("x")
                    plt.ylim([0.98, 1.02])
                    # plt.title(f"n_sph at time={time:.4f}")
                    plt.legend()

                    plt.subplot(n_rows, 3, plot_ct + 2)
                    plt.plot(e1_binned, n_binned[i, :], label=f"n_binned at time={i * dt:4.2f}", linewidth=2)
                    plt.xlim(0, r1)
                    plt.grid(c="k", linestyle="--")
                    plt.xlabel("x")
                    plt.ylim([0.98, 1.02])
                    # plt.title(f"n_binned at time={i * dt:4.2f}")
                    plt.legend()

                    plt.subplot(n_rows, 3, plot_ct + 3)
                    plt.plot(e1_binned, j1_binned[i, :], label=f"j1_binned at time={i * dt:4.2f}", linewidth=2)
                    plt.xlim(0, r1)
                    plt.grid(c="k", linestyle="--")
                    plt.xlabel("x")
                    plt.ylim([- 0.02, 0.02])
                    # plt.title(f"j1_binned at time={i * dt:4.2f}")
                    plt.legend()

                    plot_ct += 3
                    if plot_ct == n_rows * 3:
                        break

            plt.show()

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.semilogy(times, np.abs(amplitude), "o-", markersize=3,
                        label=f"Numerical (fitted rate = {gamma_numerical:.3f})")
            ax.semilogy(times, np.abs(amplitude_analytical), "--",
                        label=rf"Analytical: $\gamma = (4/3) \mu k^2 = {gamma_analytical:.3f}$")
            ax.set_xlabel("time")
            ax.set_ylabel(rf"velocity amplitude at $x = {e1_np[idx_max]:.3f}$")
            ax.set_title("Velocity diffusion: amplitude decay over time")
            ax.legend()
            ax.grid(True, which="both")
            plt.tight_layout()
            plt.show()

        rel_error = abs(gamma_numerical - gamma_analytical) / gamma_analytical
        logger.info(f"Relative error in decay rate: {rel_error * 100:.2f}%")
        assert rel_error < 0.04, (
            f"Numerical decay rate {gamma_numerical:.4f} deviates {rel_error * 100:.1f}% "
            f"from analytical {gamma_analytical:.4f} (tolerance 4%)"
        )
        logger.info("Decay rate assertion passed.")

        shutil.rmtree(test_folder)


if __name__ == "__main__":
    # test_soundwave_1d(nx=12, plot_pts=11, do_plot=True)
    # test_velocity_diffusion(nx=8, plot_pts=11, do_plot=True)
    test_damped_sound_wave(nx=8, plot_pts=11, do_plot=True)
