import logging
import os
import shutil

import cunumpy as xp
import numpy as np
import pytest
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt

from struphy import (
    BaseUnits,
    BinningPlot,
    BoundaryParameters,
    DerhamOptions,
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
    grids,
    perturbations,
    set_logging_level,
)
from struphy.initial.base import GenericPerturbation
from struphy.models import IncompressibleNavierStokesSPH
from struphy.ode.utils import ButcherTableau

logger = logging.getLogger("struphy")
set_logging_level(logging.INFO)


@pytest.mark.parametrize("nx", [8])
def test_chorin_projection_periodic_1d(nx: int, do_plot: bool = False):
    """Verification test for the Chorin projection (PoissonSolve + PushVinEfield) in a
    truly 1D periodic domain.

    Initial condition u_x(x) = U0 + A*sin(2*pi*x) has nonzero divergence everywhere.
    Under periodic boundary conditions, the only spatially-uniform-divergence-free 1D
    field is a constant, so the projection must flatten u_x(x) toward the mean flow U0,
    removing the sinusoidal (compressible) part.
    """

    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "IncompressibleNavierStokesSPH")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="chorin_projection_periodic_1d")

    time_opts = Time(dt=0.02, Tend=0.02, split_algo="LieTrotter")

    r1 = 1.0
    domain = domains.Cuboid(r1=r1)

    grid = grids.TensorProductGrid(num_elements=(nx, 1, 1))
    derham_opts = DerhamOptions(degree=(2, 1, 1))

    model = IncompressibleNavierStokesSPH(with_B0=False, with_viscosity=False)

    loading_params = LoadingParameters(ppb=16, loading="tesselation")
    weights_params = WeightsParameters()
    boundary_params = BoundaryParameters()
    sorting_params = SortingParameters(
        boxes_per_dim=(nx, 1, 1),
        dims_mask=(True, False, False),
    )

    n_bins = 32
    bin_plot_j1 = BinningPlot(slice="e1", n_bins=(n_bins,), ranges=(0.0, 1.0), output_quantity="current_1")
    saving_params = SavingParameters(binning_plots=(bin_plot_j1,))

    model.fluid.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        saving_params=saving_params,
    )

    butcher = ButcherTableau(algo="forward_euler")
    model.propagators.push_eta.options = model.propagators.push_eta.Options(butcher=butcher)

    U0 = 0.0
    A = 0.5
    background = equils.ConstantVelocity(ux=U0)
    model.fluid.density.add_background(background)
    perturbation = perturbations.ModesSin(ls=(1,), amps=(A,))
    model.fluid.density.add_perturbation(del_u1=perturbation)

    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )

    sim.run()

    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.pproc()
        sim.load_plotting_data()

        e1_grid = np.asarray(sim.f.fluid.e1_current_1.grid_e1).flatten()
        j1_binned = np.asarray(sim.f.fluid.e1_current_1.f_binned)  # (Nt+1, n_bins)

        amp_initial = 0.5 * (np.max(j1_binned[0]) - np.min(j1_binned[0]))
        amp_final = 0.5 * (np.max(j1_binned[-1]) - np.min(j1_binned[-1]))
        mean_final = np.mean(j1_binned[-1])

        logger.info(f"Chorin projection (periodic 1D): {amp_initial =:.4f}, {amp_final =:.4f}, {mean_final =:.4f}")

        if do_plot:
            plt.figure(figsize=(8, 5))
            plt.plot(e1_grid, j1_binned[0], label="t=0")
            plt.plot(e1_grid, j1_binned[-1], label=f"t={time_opts.Tend}")
            plt.axhline(U0, color="k", linestyle="--", label=f"U0={U0}")
            plt.xlabel("x")
            plt.ylabel(r"$u_x$ (current_1)")
            plt.title("Chorin projection: periodic 1D divergence cleaning")
            plt.legend()
            plt.grid(True)
            plt.show()

        rel_amp_reduction = 1.0 - amp_final / amp_initial
        logger.info(f"Relative amplitude reduction: {rel_amp_reduction * 100:.1f}%")
        assert rel_amp_reduction > 0.9, (
            f"Chorin projection did not sufficiently flatten the velocity profile: "
            f"amplitude reduced by only {rel_amp_reduction * 100:.1f}% (expected >90%)"
        )

        assert abs(mean_final - U0) < 0.05, (
            f"Mean flow after projection {mean_final:.4f} deviates from expected U0={U0}"
        )
        logger.info("Chorin projection (periodic 1D) assertions passed.")

        shutil.rmtree(test_folder)


@pytest.mark.parametrize("nx", [8])
def test_chorin_projection_reflect_1d(nx: int, do_plot: bool = False):
    """Verification test for the Chorin projection in a closed (reflecting) 1D box.

    Initial condition u_x(x) = A*sin(2*pi*x) vanishes at both walls (x=0, x=1).
    Since the walls are stationary (bc='reflect'), the only field that is both
    divergence-free AND non-penetrating at both walls is u_x=0 everywhere. The
    projection must therefore drive the entire velocity field to (near) zero,
    which is a qualitatively stronger check than the periodic case.
    """

    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "IncompressibleNavierStokesSPH")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="chorin_projection_reflect_1d")

    time_opts = Time(dt=0.02, Tend=0.4, split_algo="LieTrotter")

    r1 = 1.0
    domain = domains.Cuboid(r1=r1)

    grid = grids.TensorProductGrid(num_elements=(nx, 1, 1))
    derham_opts = DerhamOptions(degree=(2, 1, 1), bcs=(("free", "free"), None, None))

    model = IncompressibleNavierStokesSPH(with_B0=False, with_viscosity=False)

    loading_params = LoadingParameters(ppb=16, loading="tesselation")
    weights_params = WeightsParameters()
    boundary_params = BoundaryParameters(
        bc=("reflect", "periodic", "periodic"),
        bc_sph=("mirror", "periodic", "periodic"),
    )
    sorting_params = SortingParameters(
        boxes_per_dim=(nx, 1, 1),
        dims_mask=(True, False, False),
    )

    n_bins = 32
    bin_plot_j1 = BinningPlot(slice="e1", n_bins=(n_bins,), ranges=(0.0, 1.0), output_quantity="current_1")
    saving_params = SavingParameters(binning_plots=(bin_plot_j1,))

    model.fluid.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        saving_params=saving_params,
    )

    butcher = ButcherTableau(algo="forward_euler")
    model.propagators.push_eta.options = model.propagators.push_eta.Options(butcher=butcher)

    A = 0.5
    background = equils.ConstantVelocity()
    model.fluid.density.add_background(background)
    perturbation = perturbations.ModesSin(ls=(1,), amps=(A,))
    model.fluid.density.add_perturbation(del_u1=perturbation)

    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )

    sim.run()

    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.pproc()
        sim.load_plotting_data()

        e1_grid = np.asarray(sim.f.fluid.e1_current_1.grid_e1).flatten()
        j1_binned = np.asarray(sim.f.fluid.e1_current_1.f_binned)  # (Nt+1, n_bins)

        amp_initial = np.max(np.abs(j1_binned[0]))
        amp_final = np.max(np.abs(j1_binned[-1]))

        logger.info(f"Chorin projection (reflect 1D): {amp_initial =:.4f}, {amp_final =:.4f}")

        if do_plot:
            plt.figure(figsize=(8, 5))
            plt.plot(e1_grid, j1_binned[0], label="t=0")
            plt.plot(e1_grid, j1_binned[-1], label=f"t={time_opts.Tend}")
            plt.axhline(0.0, color="k", linestyle="--", label="u_x=0")
            plt.xlabel("x")
            plt.ylabel(r"$u_x$ (current_1)")
            plt.title("Chorin projection: closed (reflecting) 1D box")
            plt.legend()
            plt.grid(True)
            plt.show()

        rel_amp = amp_final / amp_initial
        logger.info(f"Final/initial amplitude ratio: {rel_amp * 100:.1f}%")
        assert rel_amp < 0.1, (
            f"Chorin projection did not drive the closed-box velocity field to zero: "
            f"final amplitude is {rel_amp * 100:.1f}% of the initial amplitude (expected <10%)"
        )
        logger.info("Chorin projection (reflect 1D) assertion passed.")

        shutil.rmtree(test_folder)


@pytest.mark.parametrize("nx", [8])
def test_channel_noslip_shear_relaxation(nx: int, do_plot: bool = False):
    """Verification test for the full incompressible propagator chain (viscosity +
    Chorin projection) under no-slip walls.

    Initial condition u_x(y) = U*sin(pi*y/H) already vanishes at both no-slip walls
    (y=0, y=H) and is exactly divergence-free (u_x depends only on y). This is a
    standard unsteady pure-shear diffusion problem: since div(u)=0 everywhere, the
    bulk-viscosity term in the stress tensor vanishes identically, giving the plain
    diffusion decay rate gamma = mu*(pi/H)**2 for the velocity amplitude. The Chorin
    projection is expected to leave this already-solenoidal, wall-bounded flow
    essentially undisturbed.
    """

    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "IncompressibleNavierStokesSPH")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="channel_noslip_shear_relaxation")

    mu = 0.1
    H = 1.0
    time_opts = Time(dt=0.01, Tend=3.0, split_algo="LieTrotter")

    domain = domains.Cuboid(r1=1.0, r2=H)

    grid = grids.TensorProductGrid(num_elements=(nx, nx, 1))
    derham_opts = DerhamOptions(degree=(2, 2, 1), bcs=(None, ("free", "free"), None))

    model = IncompressibleNavierStokesSPH(with_B0=False, with_viscosity=True)

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

    bin_plot_j1 = BinningPlot(slice="e2", n_bins=(16,), ranges=(0.0, 1.0), output_quantity="current_1")
    bin_plot_j2 = BinningPlot(slice="e2", n_bins=(16,), ranges=(0.0, 1.0), output_quantity="current_2")
    kd_plot = KernelDensityPlot(pts_e1=8, pts_e2=21, pts_e3=1)
    saving_params = SavingParameters(
        binning_plots=(bin_plot_j1, bin_plot_j2),
        kernel_density_plots=(kd_plot,),
    )

    model.fluid.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        saving_params=saving_params,
        bufsize=2,
    )

    butcher = ButcherTableau(algo="forward_euler")
    model.propagators.push_eta.options = model.propagators.push_eta.Options(butcher=butcher)
    model.propagators.push_viscous.options = model.propagators.push_viscous.Options(kernel_type="gaussian_2d", mu=mu)

    U = 0.5
    background = equils.ConstantVelocity()
    model.fluid.density.add_background(background)
    perturbation = GenericPerturbation(fun=lambda e1, e2, e3: U * xp.sin(xp.pi * e2))
    model.fluid.density.add_perturbation(del_u1=perturbation)

    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )

    sim.run()

    if MPI.COMM_WORLD.Get_rank() == 0:
        sim.pproc()
        sim.load_plotting_data()

        e2_grid = np.asarray(sim.f.fluid.e2_current_1.grid_e2).flatten()
        j1_binned = np.asarray(sim.f.fluid.e2_current_1.f_binned)  # (Nt+1, n_bins)
        j2_binned = np.asarray(sim.f.fluid.e2_current_2.f_binned)  # (Nt+1, n_bins)

        dt = time_opts.dt
        Nt = j1_binned.shape[0] - 1
        times = np.linspace(0.0, time_opts.Tend, Nt + 1)

        # probe at channel centre y=H/2, where sin(pi*y/H) peaks
        idx_centre = int(np.argmin(np.abs(e2_grid - 0.5 * H)))
        amplitude = j1_binned[:, idx_centre]

        gamma_analytical = mu * (np.pi / H) ** 2

        log_amp = np.log(np.abs(amplitude) + 1e-15)
        coeffs = np.polyfit(times, log_amp, 1)
        gamma_numerical = -coeffs[0]

        logger.info(f"Channel shear relaxation: analytical gamma = mu*(pi/H)^2 = {gamma_analytical:.4f}")
        logger.info(f"Channel shear relaxation: numerical  gamma             = {gamma_numerical:.4f}")

        if do_plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax = axes[0]
            ax.semilogy(times, np.abs(amplitude), "o-", markersize=3, label=f"numerical (rate={gamma_numerical:.3f})")
            ax.semilogy(
                times,
                np.abs(amplitude[0]) * np.exp(-gamma_analytical * times),
                "--",
                label=f"analytical (rate={gamma_analytical:.3f})",
            )
            ax.set_xlabel("time")
            ax.set_ylabel(r"$u_x(y=H/2)$")
            ax.set_title("Centreline velocity decay")
            ax.legend()
            ax.grid(True, which="both")

            ax = axes[1]
            ax.plot(e2_grid, j1_binned[0], label="t=0")
            ax.plot(e2_grid, j1_binned[-1], label=f"t={time_opts.Tend}")
            ax.set_xlabel("y")
            ax.set_ylabel(r"$u_x$ (current_1)")
            ax.set_title("Shear profile relaxation")
            ax.legend()
            ax.grid(True)

            plt.tight_layout()
            plt.show()

        rel_error = abs(gamma_numerical - gamma_analytical) / gamma_analytical
        logger.info(f"Relative error in decay rate: {rel_error * 100:.2f}%")
        assert rel_error < 0.2, (
            f"Numerical decay rate {gamma_numerical:.4f} deviates {rel_error * 100:.1f}% "
            f"from analytical {gamma_analytical:.4f} (tolerance 20%)"
        )
        logger.info("Channel shear relaxation decay rate assertion passed.")

        # Chorin projection should not spuriously inject transverse velocity
        # into this already-divergence-free flow.
        max_j2 = np.max(np.abs(j2_binned))
        max_j1 = np.max(np.abs(j1_binned))
        logger.info(f"Max |current_2| / max |current_1| = {max_j2 / max_j1:.4f}")
        assert max_j2 / max_j1 < 0.1, (
            f"Chorin projection induced significant transverse velocity: "
            f"max|current_2|/max|current_1| = {max_j2 / max_j1:.4f} (expected <0.1)"
        )
        logger.info("Transverse-velocity assertion passed.")

        shutil.rmtree(test_folder)


if __name__ == "__main__":
    test_chorin_projection_periodic_1d(nx=8, do_plot=True)
    # test_chorin_projection_reflect_1d(nx=8, do_plot=True)
    # test_channel_noslip_shear_relaxation(nx=8, do_plot=True)
