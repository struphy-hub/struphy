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
@pytest.mark.parametrize("plot_pts", [21])
def test_damped_sound_wave(nx: int, plot_pts: int, do_plot: bool = False):
    """Verification test for SPH discretization of viscous isothermal Euler equations.
    A standing sound wave decays at rate mu*k^2/2.
    The numerical decay rate is extracted from local maxima of the current, analogous to Landau damping.
    """

    # environment options
    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "ViscousEulerSPH")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="damped_soundwave_1d")

    # time stepping: Tend covers ~10 oscillation periods (T=r1/c_s=1 for mode l=1)
    mu = 0.01
    time_opts = Time(dt=0.01, Tend=10.0, split_algo="Strang")

    # geometry
    r1 = 1.0
    domain = domains.Cuboid(r1=r1)

    # grid
    grid = None

    # derham options
    derham_opts = None

    # light-weight model instance (with_p=True and with_viscosity=True are the defaults)
    model = ViscousEulerSPH(with_B0=False)

    loading_params = LoadingParameters(ppb=8, loading="tesselation")
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
    model.propagators.push_sph_p.options = model.propagators.push_sph_p.Options(kernel_type="gaussian_1d")
    model.propagators.push_viscous.options = model.propagators.push_viscous.Options(
        kernel_type="gaussian_1d", mu=mu
    )

    # background and initial conditions: velocity perturbation excites the sound wave
    background = equils.ConstantVelocity()
    model.euler_fluid.var.add_background(background)
    perturbation = perturbations.ModesSin(ls=(1,), amps=(1.0e-2,))
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

        e1_binned = sim.f.euler_fluid.e1_density.grid_e1
        n_binned = sim.f.euler_fluid.e1_density.delta_f_binned
        j1_binned = sim.f.euler_fluid.e1_current_1.f_binned
        ee1, ee2, ee3 = sim.n_sph.euler_fluid.view_0.grid_n_sph
        n_sph = sim.n_sph.euler_fluid.view_0.n_sph

        print(f"{e1_binned.shape = }")
        print(f"{n_binned.shape = }")
        print(f"{j1_binned.shape = }")
        print(f"{n_sph.shape = }")

        import numpy as np

        dt = time_opts.dt
        Nt = j1_binned.shape[0] - 1
        times = np.linspace(0.0, time_opts.Tend, Nt + 1)

        # velocity antinode: sin(2*pi*x/r1) peaks closest to x=0.25*r1
        e1_np = np.asarray(e1_binned).flatten()
        idx_max = int(np.argmin(np.abs(e1_np - 0.25 * r1)))
        amplitude = np.asarray(j1_binned[:, idx_max]).flatten()

        # analytical decay rate: gamma = -mu*k^2/2 for the acoustic mode
        k = 2.0 * np.pi / r1
        gamma_analytical = -mu * 4/3 * k**2 / 2

        A0 = amplitude[0]
        amplitude_analytical = A0 * np.exp(gamma_analytical * times)

        # find local maxima of the oscillating amplitude (analogous to Landau damping)
        logA = np.log(np.abs(amplitude) + 1e-15)
        dlogA = (np.roll(logA, -1) - np.roll(logA, 1))[1:-1] / (2.0 * dt)
        zeros = dlogA * np.roll(dlogA, -1) < 0.0
        maxima_inds = np.where(np.logical_and(zeros, dlogA > 0.0))[0] + 1
        maxima = logA[maxima_inds]
        print(f"{maxima_inds = }, {times.shape = }")
        t_maxima = times[maxima_inds]

        # linear fit to log(maxima) vs time gives the decay rate
        linfit = np.polyfit(t_maxima, maxima, 1)
        gamma_numerical = linfit[0]

        logger.info(f"Analytical decay rate: gamma = -mu*k^2/2 = {gamma_analytical:.4f}")
        logger.info(f"Numerical  decay rate: gamma             = {gamma_numerical:.4f}")

        if do_plot:
            Nt_snap = int(1.0 / dt)
            snapshot_inds = np.round(np.linspace(0, Nt_snap, 12)).astype(int)
            x_sph = np.asarray(ee1).flatten() * r1
            dn_sph = np.asarray(n_sph[:, :, 0, 0]) - 1.0  # shape (Nt+1, plot_pts)
            dn_snap = dn_sph[snapshot_inds, :]
            ylim = 1.5 * np.max(np.abs(dn_snap))
            fig, axes = plt.subplots(4, 3, figsize=(12, 12), sharex=True, sharey=True)
            for ax, idx in zip(axes.flatten(), snapshot_inds):
                ax.plot(x_sph, dn_sph[idx, :])
                ax.set_title(f"$t = {times[idx]:.2f}$")
                ax.set_ylim(-ylim, ylim)
                ax.axhline(0, color="k", linewidth=0.5)
                ax.grid(True, linestyle="--", alpha=0.5)
            for ax in axes[-1, :]:
                ax.set_xlabel("$x$")
            for ax in axes[:, 0]:
                ax.set_ylabel(r"$\delta\rho$")
            fig.suptitle(r"Density fluctuations $\delta\rho = \rho - 1$ (KDE)", fontsize=13)
            plt.tight_layout()
            plt.show()

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax = axes[0]
            ax.plot(times, amplitude, label=f"numerical j1 at x={e1_np[idx_max]:.3f}")
            ax.plot(times, amplitude_analytical, "--", label=f"analytical envelope (gamma={gamma_analytical:.3f})")
            ax.plot(t_maxima, amplitude[maxima_inds], "ro", markersize=6, label="local maxima")
            ax.set_xlabel("time")
            ax.set_ylabel("velocity amplitude")
            ax.set_title("Damped sound wave: velocity at antinode")
            ax.legend()
            ax.grid(True)

            ax = axes[1]
            ax.plot(t_maxima, maxima, "ro", markersize=6, label="log(maxima)")
            ax.plot(
                times,
                np.polyval(linfit, times),
                "--",
                label=f"fit: gamma={gamma_numerical:.3f}",
            )
            ax.axline(
                (0, np.log(np.abs(A0) + 1e-15)),
                slope=gamma_analytical,
                color="k",
                linestyle=":",
                label=f"analytical: gamma={gamma_analytical:.3f}",
            )
            ax.set_xlabel("time")
            ax.set_ylabel("log(amplitude)")
            ax.set_title("Decay rate: numerical vs analytical")
            ax.legend()
            ax.grid(True)

            plt.tight_layout()
            plt.show()

        rel_error = abs(gamma_numerical - gamma_analytical) / abs(gamma_analytical)
        logger.info(f"Relative error in decay rate: {rel_error * 100:.2f}%")
        assert rel_error < 0.16, (
            f"Numerical decay rate {gamma_numerical:.4f} deviates {rel_error * 100:.1f}% "
            f"from analytical {gamma_analytical:.4f} (tolerance 16%)"
        )
        logger.info("Damped sound wave decay rate assertion passed.")

        # shutil.rmtree(test_folder)


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
@pytest.mark.parametrize("plot_pts", [21])
def test_hagen_poiseuille(nx: int, plot_pts: int, do_plot: bool = False, create_png: bool = False):
    """Verification test for SPH viscosity tensor in 2D Hagen-Poiseuille channel flow.

    Channel geometry: x ∈ [0, 1] periodic (flow direction), y ∈ [0, 1] no-slip walls.
    A constant body force g_x drives the flow; viscosity produces the parabolic
    steady-state profile u_x(y) = g_x / (2 mu) * y * (1 - y).
    """

    # environment options
    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "ViscousEulerSPH")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="hagen_poiseuille")

    # physical parameters
    mu = 0.1    # dynamic viscosity
    g_x = 0.1  # body force in x (acts as driving pressure gradient)
    H = 1.0     # channel height in y

    # time stepping: T_relax = H^2 / (pi^2 * mu) ~ 1.0, run 10x past relaxation
    time_opts = Time(dt=0.01, Tend=10.0, split_algo="Strang")

    # 2D channel: x-periodic [0, 1], y-walls [0, H=1], z-trivial
    domain = domains.Cuboid(r1=1.0, r2=H)

    # model with pressure (to maintain ~uniform density) and viscosity
    model = ViscousEulerSPH(with_B0=False, with_p=True, with_viscosity=True)

    loading_params = LoadingParameters(ppb=16, loading="tesselation")
    weights_params = WeightsParameters()
    boundary_params = BoundaryParameters(
        bc=("periodic", "reflect", "periodic"),
        bc_sph=("periodic", "noslip", "periodic"),
    )
    sorting_params = SortingParameters(
        boxes_per_dim=(nx, nx, 1),
        dims_mask=(True, True, False),
    )

    # bin current_1 (≈ rho*u_x ≈ u_x) as a function of y to get the velocity profile
    bin_plot_j1 = BinningPlot(slice="e2", n_bins=(16,), ranges=(0.0, 1.0), output_quantity="current_1")
    bin_plot_n = BinningPlot(slice="e2", n_bins=(16,), ranges=(0.0, 1.0))
    kd_plot = KernelDensityPlot(pts_e1=plot_pts, pts_e2=plot_pts, pts_e3=1)
    saving_params = SavingParameters(
        n_markers=1.0,
        binning_plots=(bin_plot_j1, bin_plot_n),
        kernel_density_plots=(kd_plot,),
    )

    model.euler_fluid.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        saving_params=saving_params,
        bufsize=2,
    )

    # propagator options: use 2D Gaussian kernel
    from struphy.ode.utils import ButcherTableau

    butcher = ButcherTableau(algo="forward_euler")
    model.propagators.push_eta.options = model.propagators.push_eta.Options(butcher=butcher)
    model.propagators.push_sph_p.options = model.propagators.push_sph_p.Options(
        kernel_type="gaussian_2d",
        gravity=(g_x, 0.0, 0.0),
    )
    model.propagators.push_viscous.options = model.propagators.push_viscous.Options(
        kernel_type="gaussian_2d",
        mu=mu,
    )

    # start from rest; body force drives the flow to the Hagen-Poiseuille steady state
    background = equils.ConstantVelocity()
    model.euler_fluid.var.add_background(background)

    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=None,
        derham_opts=None,
    )

    sim.run()

    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.pproc()
        sim.load_plotting_data()

        e2_grid = sim.f.euler_fluid.e2_current_1.grid_e2  # logical y in [0, 1]
        j1_binned = sim.f.euler_fluid.e2_current_1.f_binned  # shape (Nt+1, n_bins)

        import numpy as np

        dt = time_opts.dt
        Nt = int(time_opts.Tend / dt)
        times = np.linspace(0.0, time_opts.Tend, Nt + 1)

        e2_np = np.asarray(e2_grid).flatten()
        y_np = e2_np * H  # physical y coordinate

        # analytical Hagen-Poiseuille: u_x(y) = g_x/(2*mu) * y*(H - y)
        u_exact = g_x / (2.0 * mu) * y_np * (H - y_np)
        u_max_exact = np.max(u_exact)

        u_num_final = np.asarray(j1_binned[-1, :]).flatten()
        u_max_num = np.max(u_num_final)

        # velocity at channel centre (y=H/2) as function of time
        idx_centre = int(np.argmin(np.abs(e2_np - 0.5)))
        u_centre = np.asarray(j1_binned[:, idx_centre]).flatten()
        u_centre_exact = u_max_exact  # peak at y=H/2

        logger.info(f"Hagen-Poiseuille: analytical U_max = {u_max_exact:.6f}")
        logger.info(f"Hagen-Poiseuille: numerical  U_max = {u_max_num:.6f}")

        abs_err = np.abs(u_num_final - u_exact)
        # pointwise relative error (avoid division by zero at the walls)
        rel_err_pointwise = abs_err / u_max_exact
        # exclude wall bins where the exact value is effectively zero
        rel_error_interior = rel_err_pointwise[1:-1]  # exclude first and last bins
        rel_error_umax = abs(u_max_num - u_max_exact) / u_max_exact

        logger.info(f"Hagen-Poiseuille: mean interior relative error = {np.mean(rel_error_interior) * 100:.2f}%")
        logger.info(f"Hagen-Poiseuille: relative error in U_max = {rel_error_umax * 100:.2f}%")

        if do_plot:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # velocity profile: numerical vs analytical
            ax = axes[0]
            ax.plot(u_num_final, y_np, "o-", markersize=4, label="Numerical (SPH)")
            ax.plot(u_exact, y_np, "--", color="k", label="Analytical (Hagen-Poiseuille)")
            ax.set_xlabel(r"$u_x$")
            ax.set_ylabel(r"$y$")
            ax.set_title("Steady-state velocity profile")
            ax.legend()
            ax.grid(True)

            # pointwise relative error
            ax = axes[1]
            ax.plot(rel_err_pointwise * 100, y_np, "r-o", markersize=4)
            ax.set_xlabel(r"$|u_x^{num} - u_x^{exact}| \,/\, u_x^{exact}$ [%]")
            ax.set_ylabel(r"$y$")
            ax.set_title(f"Pointwise relative error (mean = {np.mean(rel_error_interior) * 100:.1f}%)")
            ax.grid(True)

            # time evolution of centreline velocity
            ax = axes[2]
            ax.plot(times, u_centre, label=r"Numerical $u_x(y=H/2)$")
            ax.axhline(u_centre_exact, color="k", linestyle="--",
                       label=rf"Exact $U_{{max}} = {u_centre_exact:.4f}$")
            ax.set_xlabel("time")
            ax.set_ylabel(r"$u_x(y=H/2)$")
            ax.set_title("Centreline velocity relaxation to steady state")
            ax.legend()
            ax.grid(True)

            plt.suptitle(
                rf"Hagen-Poiseuille: $\mu={mu}$, $g_x={g_x}$, $H={H}$, {nx}×{nx} boxes",
                fontsize=12,
            )
            plt.tight_layout()
            plt.show()

        if create_png:
            from matplotlib.colors import LinearSegmentedColormap
            from tqdm import tqdm as _tqdm

            orbits = np.asarray(sim.orbits.euler_fluid)  # (Nt_orb, n_markers, n_attrs)
            # attrs for vdim=2: [x, y, z, v1, v2, w, diag, id]

            Nt_orb = orbits.shape[0]
            t_orbit = np.linspace(0.0, time_opts.Tend, Nt_orb)

            # colormap: blue at walls (y=0, y=H), red at channel centre (y=H/2)
            # c_val = 1 - 2*|y/H - 0.5| maps walls→0 (blue) and centre→1 (red)
            cmap_pos = LinearSegmentedColormap.from_list("wall_centre", ["blue", "red"])
            norm = plt.Normalize(0.0, 1.0)

            # 250 equally spaced snapshot indices
            n_snaps = 250
            snap_inds = np.round(np.linspace(0, Nt_orb - 1, n_snaps)).astype(int)

            png_dir = os.path.join(out_folders, "hagen_poiseuille_pngs")
            os.makedirs(png_dir, exist_ok=True)

            for i, idx in _tqdm(enumerate(snap_inds), total=n_snaps, desc="saving PNGs"):
                c_val = 1.0 - 2.0 * np.abs(orbits[idx, :, 1] / H - 0.5)
                fig_png, ax_png = plt.subplots(figsize=(8, 6))
                sc_png = ax_png.scatter(
                    orbits[idx, :, 0],
                    orbits[idx, :, 1],
                    c=c_val,
                    cmap=cmap_pos,
                    norm=norm,
                    s=10,
                )
                ax_png.axhline(0.0, color="k", linewidth=6)
                ax_png.axhline(H, color="k", linewidth=6)
                ax_png.set_xlim(0.0, 1.0)
                ax_png.set_ylim(-0.05 * H, 1.05 * H)
                ax_png.set_xlabel("x")
                ax_png.set_ylabel("y")
                ax_png.set_title(rf"Hagen-Poiseuille markers, $t = {t_orbit[idx]:.2f}$")
                plt.colorbar(sc_png, ax=ax_png, label="steady-state velocity [a.u.]")
                plt.tight_layout()
                fig_png.savefig(os.path.join(png_dir, f"snap_{i:04d}.png"), dpi=80)
                plt.close(fig_png)

            # show last snapshot in a new figure
            fig_last, ax_last = plt.subplots(figsize=(8, 6))
            idx_last = snap_inds[-1]
            c_val_last = 1.0 - 2.0 * np.abs(orbits[idx_last, :, 1] / H - 0.5)
            sc_last = ax_last.scatter(
                orbits[idx_last, :, 0],
                orbits[idx_last, :, 1],
                c=c_val_last,
                cmap=cmap_pos,
                norm=norm,
                s=4,
            )
            ax_last.axhline(0.0, color="k", linewidth=3, label="no-slip boundary")
            ax_last.axhline(H, color="k", linewidth=3)
            ax_last.set_xlim(0.0, 1.0)
            ax_last.set_ylim(-0.05 * H, 1.05 * H)
            ax_last.set_xlabel("x")
            ax_last.set_ylabel("y")
            ax_last.set_title(rf"Last snapshot: $t = {t_orbit[idx_last]:.2f}$")
            ax_last.legend()
            plt.colorbar(sc_last, ax=ax_last, label="steady-state velocity [a.u.]")
            plt.tight_layout()
            plt.show()

        assert np.max(rel_error_interior) < 0.05, (
            f"Hagen-Poiseuille mean relative error {np.mean(rel_error_interior) * 100:.1f}% exceeds tolerance 15%"
        )
        logger.info("Hagen-Poiseuille profile assertion passed.")

        assert rel_error_umax < 0.05, (
            f"Hagen-Poiseuille U_max relative error {rel_error_umax * 100:.1f}% exceeds tolerance 10%"
        )
        logger.info("Hagen-Poiseuille U_max assertion passed.")

        # shutil.rmtree(test_folder)


if __name__ == "__main__":
    # test_soundwave_1d(nx=12, plot_pts=11, do_plot=True)
    # test_velocity_diffusion(nx=8, plot_pts=11, do_plot=True)
    # test_damped_sound_wave(nx=8, plot_pts=21, do_plot=True)
    test_hagen_poiseuille(nx=8, plot_pts=21, do_plot=True, create_png=True)

