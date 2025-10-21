import os

import pytest
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from psydac.ddm.mpi import mpi as MPI

from struphy import main
from struphy.fields_background import equils
from struphy.geometry import domains
from struphy.initial import perturbations
from struphy.io.options import BaseUnits, DerhamOptions, EnvironmentOptions, FieldsBackground, Time
from struphy.kinetic_background import maxwellians
from struphy.pic.utilities import (
    BinningPlot,
    BoundaryParameters,
    KernelDensityPlot,
    LoadingParameters,
    WeightsParameters,
)
from struphy.topology import grids
from struphy.utils.arrays import xp as np

test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")


@pytest.mark.parametrize("nx", [12, 24])
@pytest.mark.parametrize("plot_pts", [11, 32])
def test_soundwave_1d(nx: int, plot_pts: int, do_plot: bool = False):
    """Verification test for SPH discretization of isthermal Euler equations.
    A standing sound wave with c_s=1 traveserses the domain once.
    """
    # import model
    from struphy.models.fluid import EulerSPH

    # environment options
    out_folders = os.path.join(test_folder, "EulerSPH")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="soundwave_1d")

    # units
    base_units = BaseUnits(kBT=1.0)

    # time stepping
    time_opts = Time(dt=0.03125, Tend=2.5, split_algo="Strang")

    # geometry
    r1 = 2.5
    domain = domains.Cuboid(r1=r1)

    # fluid equilibrium (can be used as part of initial conditions)
    equil = None

    # grid
    grid = None

    # derham options
    derham_opts = None

    # light-weight model instance
    model = EulerSPH(with_B0=False)

    # species parameters
    model.euler_fluid.set_phys_params()

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

    # start run
    main.run(
        model,
        params_path=None,
        env=env,
        base_units=base_units,
        time_opts=time_opts,
        domain=domain,
        equil=equil,
        grid=grid,
        derham_opts=derham_opts,
        verbose=True,
    )

    # post processing
    if MPI.COMM_WORLD.Get_rank() == 0:
        main.pproc(env.path_out)

        # diagnostics
        simdata = main.load_data(env.path_out)

        ee1, ee2, ee3 = simdata.n_sph["euler_fluid"]["view_0"]["grid_n_sph"]
        n_sph = simdata.n_sph["euler_fluid"]["view_0"]["n_sph"]

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
                    print(f"{i = }")
                    plot_ct += 1
                    ax = plt.gca()

                    if plot_ct <= 6:
                        style = "-"
                    else:
                        style = "."
                    plt.plot(x.squeeze(), n_sph[i, :, 0, 0], style, label=f"time={i * dt:4.2f}")
                    plt.xlim(0, 2.5)
                    plt.legend()
                    ax.set_xticks(np.linspace(0, 2.5, nx + 1))
                    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                    plt.grid(c="k")
                    plt.xlabel("x")
                    plt.ylabel(r"$\rho$")

                    plt.title(f"standing sound wave ($c_s = 1$) for {nx = } and {ppb = }")
                if plot_ct == 11:
                    break

            plt.show()

        error = np.max(np.abs(n_sph[0] - n_sph[-1]))
        print(f"SPH sound wave {error = }.")
        assert error < 6e-4
        print("Assertion passed.")


if __name__ == "__main__":
    test_soundwave_1d(nx=12, plot_pts=11, do_plot=True)
