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
from struphy.initial.base import GenericPerturbation
from struphy.models import ViscousEulerSPH

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
    model.propagators.push_viscous.options = model.propagators.push_viscous.Options(kernel_type="gaussian_1d", mu=mu)

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
        gamma_analytical = -mu * 4 / 3 * k**2 / 2

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
        gamma_analytical = mu * 4 / 3 * k**2

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
            ax.semilogy(
                times, np.abs(amplitude), "o-", markersize=3, label=f"Numerical (fitted rate = {gamma_numerical:.3f})"
            )
            ax.semilogy(
                times,
                np.abs(amplitude_analytical),
                "--",
                label=rf"Analytical: $\gamma = (4/3) \mu k^2 = {gamma_analytical:.3f}$",
            )
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
    mu = 0.1  # dynamic viscosity
    g_x = 0.1  # body force in x (acts as driving pressure gradient)
    H = 1.0  # channel height in y

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
            ax.axhline(u_centre_exact, color="k", linestyle="--", label=rf"Exact $U_{{max}} = {u_centre_exact:.4f}$")
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
            n_snaps = np.min([250, Nt_orb])
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

        assert np.mean(rel_error_interior) < 0.05, (
            f"Hagen-Poiseuille mean relative error {np.mean(rel_error_interior) * 100:.1f}% exceeds tolerance 5%"
        )
        logger.info("Hagen-Poiseuille profile assertion passed.")

        assert rel_error_umax < 0.05, (
            f"Hagen-Poiseuille U_max relative error {rel_error_umax * 100:.1f}% exceeds tolerance 10%"
        )
        logger.info("Hagen-Poiseuille U_max assertion passed.")

        # shutil.rmtree(test_folder)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("nx", [8])
@pytest.mark.parametrize("plot_pts", [21])
def test_dam_break(nx: int, plot_pts: int, do_plot: bool = False, create_png: bool = False):
    """2D dam break: a dense fluid column (left half, x < r1/2) released in a
    closed box with reflective walls, driven by pressure gradient and gravity.

    The initial density step (left: n_high, right: 1.0) is set via ConstantVelocity
    background plus a GenericPerturbation. Downward gravity drives the collapse.
    """

    # environment options
    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "ViscousEulerSPH")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="dam_break")

    # physical parameters
    # WCSPH liquid regime: kappa = c_s^2 must satisfy c_s >> U_max = sqrt(2*g*H).
    # With g=10, H=0.5: U_max ≈ 3.2.  kappa=50 → c_s≈7, Ma≈0.45 (subsonic, liquid-like).
    # Raising gravity would increase Ma and make things more gas-like — wrong direction.
    kappa = 0.2  # isothermal coefficient (= c_s^2); controls fluid stiffness
    mu = 0.05  # dynamic viscosity
    g_y = 10.0  # gravitational acceleration (downward, i.e. −y)
    r1 = 1.0  # domain width  (x-direction)
    r2 = 1.0  # domain height (y-direction)
    n_high = 0.1  # density of the fluid column (uniform → no initial pressure gradient)

    # free-fall time sqrt(2*H/g) ≈ 0.32 s; acoustic CFL: h/c_s = (1/nx)/sqrt(kappa) ≈ 0.018
    time_opts = Time(dt=0.02, Tend=3.0, split_algo="Strang")

    # 2D closed box
    domain = domains.Cuboid(r1=r1, r2=r2)

    # model with pressure and small viscosity
    model = ViscousEulerSPH(with_B0=False, with_p=True, with_viscosity=True)

    loading_params = LoadingParameters(ppb=32, loading="tesselation")
    # markers with weight ∝ 1e-8 (near-vacuum right half) are rejected;
    # left-half weights ∝ n_high × vol_per_marker ≈ n_high/(nx²×ppb) ≈ 5e-5, so 1e-6 separates cleanly
    weights_params = WeightsParameters(reject_weights=True, threshold=1e-6)
    boundary_params = BoundaryParameters(
        bc=("reflect", "reflect", "periodic"),
        bc_sph=("mirror", "mirror", "periodic"),
    )
    sorting_params = SortingParameters(
        boxes_per_dim=(nx, nx, 1),
        dims_mask=(True, True, False),
    )

    kd_plot = KernelDensityPlot(pts_e1=plot_pts, pts_e2=plot_pts, pts_e3=1)
    saving_params = SavingParameters(
        n_markers=1.0,
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

    # propagator options: 2D Gaussian kernel with downward gravity
    from struphy.ode.utils import ButcherTableau

    butcher = ButcherTableau(algo="forward_euler")
    model.propagators.push_eta.options = model.propagators.push_eta.Options(butcher=butcher)
    model.propagators.push_sph_p.options = model.propagators.push_sph_p.Options(
        kernel_type="gaussian_2d",
        gravity=(0.0, -g_y, 0.0),
        kappa=kappa,
    )
    model.propagators.push_viscous.options = model.propagators.push_viscous.Options(
        kernel_type="gaussian_2d",
        mu=mu,
    )

    # initial condition: dense column on the left half (x < r1/2), near-vacuum on the right.
    # step_function_xy places n_high where x < upper_x and 1e-8 elsewhere; the near-vacuum
    # markers are then removed by reject_weights above.
    background = equils.ConstantVelocity(
        density_profile="step_function_xy",
        n=n_high,
        upper_x=r1 / 4,
        upper_y=r2,
    )
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

        import numpy as np

        dt = time_opts.dt
        Nt = int(time_opts.Tend / dt)
        times = np.linspace(0.0, time_opts.Tend, Nt + 1)

        ee1, ee2, ee3 = sim.n_sph.euler_fluid.view_0.grid_n_sph
        n_sph = sim.n_sph.euler_fluid.view_0.n_sph  # (Nt+1, pts_e1, pts_e2, 1)

        X = np.asarray(ee1)[:, :, 0] * r1  # physical x, shape (pts_e1, pts_e2)
        Y = np.asarray(ee2)[:, :, 0] * r2  # physical y, shape (pts_e1, pts_e2)
        n_arr = np.asarray(n_sph)  # (Nt+1, pts_e1, pts_e2, 1)

        # orbits needed for both do_plot scatter overlay and create_png
        orbits = np.asarray(sim.orbits.euler_fluid)  # (Nt_orb, n_markers, n_attrs)
        Nt_orb = orbits.shape[0]
        t_orbit = np.linspace(0.0, time_opts.Tend, Nt_orb)

        # color each marker by its initial x (gradient across the left column)
        x_init = orbits[0, :, 0]
        c_val = x_init / (r1 / 2)  # 0 = left wall, 1 = dam face

        if do_plot:
            snapshot_inds = np.round(np.linspace(0, Nt, 12)).astype(int)
            # index into orbits at the same times (Nt_orb == Nt + 1 when n_markers=1.0)
            orb_inds = np.round(np.linspace(0, Nt_orb - 1, 12)).astype(int)

            vmax_plot = float(np.max(n_arr))  # density can pile above n_high as fluid collects

            fig, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True, sharey=True)
            im = None
            for ax, idx, oidx in zip(axes.flatten(), snapshot_inds, orb_inds):
                n_2d = n_arr[idx, :, :, 0]
                im = ax.pcolormesh(X, Y, n_2d, vmin=0.0, vmax=vmax_plot/2, cmap="Blues", shading="auto")
                ax.scatter(
                    orbits[oidx, :, 0],
                    orbits[oidx, :, 1],
                    c=c_val,
                    cmap="autumn",
                    s=2,
                    vmin=0.0,
                    vmax=1.0,
                    alpha=0.6,
                )
                ax.set_title(f"$t = {times[idx]:.2f}$")
                ax.set_aspect("equal")
            for ax in axes[-1, :]:
                ax.set_xlabel("$x$")
            for ax in axes[:, 0]:
                ax.set_ylabel("$y$")
            if im is not None:
                fig.colorbar(im, ax=axes.ravel().tolist(), label=r"$\rho$", shrink=0.6)
            fig.suptitle(
                rf"2D dam break: $\rho$ (KDE) + markers, $\kappa={kappa}$, $g_y={g_y}$, $\mu={mu}$, {nx}×{nx} boxes",
                fontsize=12,
            )
            plt.tight_layout()
            plt.show()

        if create_png:
            from tqdm import tqdm as _tqdm

            n_snaps = 300
            snap_inds = np.round(np.linspace(0, Nt_orb - 1, n_snaps)).astype(int)
            n_snap_inds = np.round(np.linspace(0, Nt, n_snaps)).astype(int)

            vmax_plot = float(np.max(n_arr))

            png_dir = os.path.join(out_folders, "dam_break_pngs")
            os.makedirs(png_dir, exist_ok=True)

            for i, (idx, n_idx) in _tqdm(
                enumerate(zip(snap_inds, n_snap_inds)), total=n_snaps, desc="saving PNGs"
            ):
                fig_png, ax_png = plt.subplots(figsize=(10, 5))
                n_2d = n_arr[n_idx, :, :, 0]
                im = ax_png.pcolormesh(X, Y, n_2d, vmin=0.0, vmax=vmax_plot/2, cmap="Blues", shading="auto")
                ax_png.scatter(
                    orbits[idx, :, 0],
                    orbits[idx, :, 1],
                    c=c_val,
                    cmap="autumn",
                    s=1,
                    vmin=0.0,
                    vmax=1.0,
                    alpha=0.6,
                )
                ax_png.set_xlim(0.0, r1)
                ax_png.set_ylim(0.0, r2)
                ax_png.set_xlabel("x")
                ax_png.set_ylabel("y")
                ax_png.set_title(rf"Dam break (compressible), $t = {t_orbit[idx]:.3f}$")
                ax_png.set_aspect("equal")
                plt.colorbar(im, ax=ax_png, label=r"$\rho$")
                fig_png.savefig(os.path.join(png_dir, f"snap_{i:04d}.png"), dpi=80, bbox_inches="tight", pad_inches=0.02)
                plt.close(fig_png)

        # sanity: no markers should escape the closed box (allow 1% tolerance)
        x_all = orbits[:, :, 0]
        y_all = orbits[:, :, 1]
        assert np.all(x_all >= -0.01 * r1) and np.all(x_all <= 1.01 * r1), "Markers escaped x-domain in dam break test"
        assert np.all(y_all >= -0.01 * r2) and np.all(y_all <= 1.01 * r2), "Markers escaped y-domain in dam break test"
        logger.info("Dam break domain bounds assertion passed.")


if __name__ == "__main__":
    # test_soundwave_1d(nx=12, plot_pts=11, do_plot=True)
    # test_velocity_diffusion(nx=8, plot_pts=11, do_plot=True)
    # test_damped_sound_wave(nx=8, plot_pts=21, do_plot=True)
    # test_hagen_poiseuille(nx=8, plot_pts=21, do_plot=True, create_png=True)
    test_dam_break(nx=8, plot_pts=21, do_plot=True, create_png=True)
