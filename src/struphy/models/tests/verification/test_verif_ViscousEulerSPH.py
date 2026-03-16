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
    Simulation,
    Time,
    WeightsParameters,
    domains,
    equils,
    perturbations,
)
from struphy.models import ViscousEulerSPH


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

    # units
    base_units = BaseUnits(kBT=1.0)

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

    # species parameters
    model.euler_fluid.set_species_properties()

    loading_params = LoadingParameters(ppb=8, loading="tesselation")
    weights_params = WeightsParameters()
    boundary_params = BoundaryParameters()
    model.euler_fluid.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
    )
    model.euler_fluid.set_sorting_boxes(
        boxes_per_dim=(nx, 1, 1),
        dims_maks=(True, False, False),
    )

    bin_plot = BinningPlot(slice="e1", n_bins=(32,), ranges=(0.0, 1.0))
    kd_plot = KernelDensityPlot(pts_e1=plot_pts, pts_e2=1)
    model.euler_fluid.set_save_data(
        binning_plots=(bin_plot,),
        kernel_density_plots=(kd_plot,),
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
        base_units=base_units,
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
        sim.load_plotting_data(env.path_out)

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
                    print(f"{i =}")
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
        print(f"SPH sound wave {error =}.")
        assert error < 6e-4
        print("Assertion passed.")

        shutil.rmtree(test_folder)


@pytest.mark.parametrize("nx", [12, 24])
@pytest.mark.parametrize("plot_pts", [11, 32])
def test_viscosity_1d(nx: int, plot_pts: int, do_plot: bool = False):
    """Verification test for SPH discretization of viscosity in Euler equations.
    A Gaussian blob in vx diffuses in periodic boundary conditions.
    """

    # environment options
    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "ViscousEulerSPH")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="viscosity_1d")

    # units
    base_units = BaseUnits(kBT=1.0)

    # time stepping
    time_opts = Time(dt=0.01, Tend=0.1, split_algo="LieTrotter")

    # geometry
    r1 = 1.0
    domain = domains.Cuboid(r1=r1)

    # grid
    grid = None

    # derham options
    derham_opts = None

    # light-weight model instance
    model = ViscousEulerSPH(with_B0=False, with_p=False, with_viscosity=True)

    # species parameters
    model.euler_fluid.set_species_properties()

    loading_params = LoadingParameters(ppb=100, loading="tesselation")
    weights_params = WeightsParameters()
    boundary_params = BoundaryParameters()
    model.euler_fluid.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
    )
    model.euler_fluid.set_sorting_boxes(
        boxes_per_dim=(nx, 1, 1),
        dims_maks=(True, False, False),
    )

    bin_plot = BinningPlot(slice="e1", n_bins=(32,), ranges=(0.0, 1.0), output_quantity="current_1")
    kd_plot = KernelDensityPlot(pts_e1=plot_pts, pts_e2=1)
    model.euler_fluid.set_save_data(
        binning_plots=(bin_plot,),
        kernel_density_plots=(kd_plot,),
    )

    # propagator options
    from struphy.ode.utils import ButcherTableau

    butcher = ButcherTableau(algo="forward_euler")
    # model.propagators.push_eta.options = model.propagators.push_eta.Options(butcher=butcher)
    if model.with_viscosity:
        model.propagators.push_viscous.options = model.propagators.push_viscous.Options(kernel_type="gaussian_1d", mu=0.001)

    # background, perturbations and initial conditions
    background = equils.ConstantVelocity()
    model.euler_fluid.var.add_background(background)
    perturbation = perturbations.GaussianBlobEta1(center=0.5, amp=1.0, sigma=0.1)
    model.euler_fluid.var.add_perturbation(del_u1=perturbation)

    # instance of simulation
    sim = Simulation(
        model=model,
        env=env,
        base_units=base_units,
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
        sim.load_plotting_data(env.path_out)

        shp = sim.t_grid.size
        grid_e1 = sim.f.euler_fluid.e1_current_1.grid_e1
        f_binned = sim.f.euler_fluid.e1_current_1.f_binned
        print(f_binned.shape)

        if do_plot:
            plt.figure(figsize=(20, 8))
            plt.subplot(1, 4, 1)
            plt.plot(grid_e1, f_binned[0, :], label=f"time {sim.t_grid[0]}")
            plt.title(f"time {sim.t_grid[0]}")
            plt.subplot(1, 4, 2)
            plt.plot(grid_e1, f_binned[shp//3, :], label=f"time {sim.t_grid[shp//3]}")
            plt.title(f"time {sim.t_grid[shp//3]}")
            plt.subplot(1, 4, 3)
            plt.plot(grid_e1, f_binned[2*shp//3, :], label=f"time {sim.t_grid[2*shp//3]}")
            plt.title(f"time {sim.t_grid[2*shp//3]}")           
            plt.subplot(1, 4, 4)
            plt.plot(grid_e1, f_binned[-1, :], label=f"time {sim.t_grid[-1]}")
            plt.title(f"time {sim.t_grid[-1]}")
            plt.show()


if __name__ == "__main__":
    # test_soundwave_1d(nx=12, plot_pts=11, do_plot=True)
    test_viscosity_1d(nx=12, plot_pts=11, do_plot=True)
