import os
import shutil
import tempfile

import cunumpy as xp
import h5py
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


def test_weak_Landau(do_plot: bool = False):
    """Verification test for weak Landau damping.
    The computed damping rate is compared to the analytical rate.
    """
    # import model
    from struphy.models.kinetic import VlasovAmpereOneSpecies

    # environment options
    with tempfile.TemporaryDirectory() as test_folder:
        out_folders = os.path.join(test_folder, "VlasovAmpereOneSpecies")
        env = EnvironmentOptions(out_folders=out_folders, sim_folder="weak_Landau")

        # units
        base_units = BaseUnits()

        # time stepping
        time_opts = Time(dt=0.05, Tend=15)

        # geometry
        r1 = 12.56
        domain = domains.Cuboid(r1=r1)

        # fluid equilibrium (can be used as part of initial conditions)
        equil = None

        # grid
        grid = grids.TensorProductGrid(Nel=(32, 1, 1))

        # derham options
        derham_opts = DerhamOptions(p=(3, 1, 1))

        # light-weight model instance
        model = VlasovAmpereOneSpecies(with_B0=False)

        # species parameters
        model.kinetic_ions.set_phys_params(alpha=1.0, epsilon=-1.0)

        ppc = 1000
        loading_params = LoadingParameters(ppc=ppc, seed=1234)
        weights_params = WeightsParameters(control_variate=True)
        boundary_params = BoundaryParameters()
        model.kinetic_ions.set_markers(
            loading_params=loading_params,
            weights_params=weights_params,
            boundary_params=boundary_params,
            bufsize=0.4,
        )
        model.kinetic_ions.set_sorting_boxes(boxes_per_dim=(16, 1, 1), do_sort=True)

        binplot = BinningPlot(slice="e1_v1", n_bins=(128, 128), ranges=((0.0, 1.0), (-5.0, 5.0)))
        model.kinetic_ions.set_save_data(binning_plots=(binplot,))

        # propagator options
        model.propagators.push_eta.options = model.propagators.push_eta.Options()
        if model.with_B0:
            model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
        model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
        model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

        # background and initial conditions
        background = maxwellians.Maxwellian3D(n=(1.0, None))
        model.kinetic_ions.var.add_background(background)

        # if .add_initial_condition is not called, the background is the initial condition
        perturbation = perturbations.ModesCos(ls=(1,), amps=(1e-3,))
        init = maxwellians.Maxwellian3D(n=(1.0, perturbation))
        model.kinetic_ions.var.add_initial_condition(init)

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
            verbose=False,
        )

        # post processing not needed for scalar data

        # exat solution
        gamma = -0.1533

        def E_exact(t):
            eps = 0.001
            k = 0.5
            r = 0.3677
            omega = 1.4156
            phi = 0.5362
            return 16 * eps**2 * r**2 * xp.exp(2 * gamma * t) * 2 * xp.pi * xp.cos(omega * t - phi) ** 2 / 2

        # get parameters
        dt = time_opts.dt
        algo = time_opts.split_algo
        Nel = grid.Nel
        p = derham_opts.p

        # get scalar data
        if MPI.COMM_WORLD.Get_rank() == 0:
            pa_data = os.path.join(env.path_out, "data")
            with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
                time = f["time"]["value"][()]
                E = f["scalar"]["en_E"][()]
            logE = xp.log10(E)

            # find where time derivative of E is zero
            dEdt = (xp.roll(logE, -1) - xp.roll(logE, 1))[1:-1] / (2.0 * dt)
            zeros = dEdt * xp.roll(dEdt, -1) < 0.0
            maxima_inds = xp.logical_and(zeros, dEdt > 0.0)
            maxima = logE[1:-1][maxima_inds]
            t_maxima = time[1:-1][maxima_inds]

            # plot
            if do_plot:
                plt.figure(figsize=(18, 12))
                plt.plot(time, logE, label="numerical")
                plt.plot(time, xp.log10(E_exact(time)), label="exact")
                plt.legend()
                plt.title(f"{dt=}, {algo=}, {Nel=}, {p=}, {ppc=}")
                plt.xlabel("time [m/c]")
                plt.plot(t_maxima[:5], maxima[:5], "r")
                plt.plot(t_maxima[:5], maxima[:5], "or", markersize=10)
                plt.ylim([-10, -4])

                plt.show()

            # linear fit
            linfit = xp.polyfit(t_maxima[:5], maxima[:5], 1)
            gamma_num = linfit[0]

            # assert
            rel_error = xp.abs(gamma_num - gamma) / xp.abs(gamma)
            assert rel_error < 0.22, f"Assertion for weak Landau damping failed: {gamma_num =} vs. {gamma =}."
            print(f"Assertion for weak Landau damping passed ({rel_error =}).")


if __name__ == "__main__":
    test_weak_Landau(do_plot=True)
