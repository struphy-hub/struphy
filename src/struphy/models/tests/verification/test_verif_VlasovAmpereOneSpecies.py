import logging
import os
import shutil

import cunumpy as xp
import h5py
import pytest
from matplotlib import pyplot as plt
from maybempi import MPI

from struphy import (
    BaseUnits,
    BinningPlot,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    grids,
    maxwellians,
    perturbations,
)
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import VlasovAmpereOneSpecies

logger = logging.getLogger("struphy")


def test_weak_Landau(do_plot: bool = False, exit_before_run: bool = False):
    """Verification test for weak Landau damping.
    The computed damping rate is compared to the analytical rate.
    """
    # light-weight model instance
    model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)

    # environment options
    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "VlasovAmpereOneSpecies")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="weak_Landau")

    # time stepping
    time_opts = Time(dt=0.05, Tend=15)

    # geometry
    r1 = 12.56
    domain = domains.Cuboid(r1=r1)

    # grid
    grid = grids.TensorProductGrid(num_elements=(32, 1, 1))

    # derham options
    derham_opts = DerhamOptions(degree=(3, 1, 1))

    # markers
    ppc = 1000
    loading_params = LoadingParameters(ppc=ppc, seed=1234)
    weights_params = WeightsParameters(control_variate=True)
    boundary_params = BoundaryParameters()
    sorting_params = SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True)

    binplot = BinningPlot(slice="e1_v1", n_bins=(128, 128), ranges=((0.0, 1.0), (-5.0, 5.0)))
    saving_params = SavingParameters(binning_plots=(binplot,))

    model.kinetic_ions.set_markers(
        loading_params=loading_params,
        weights_params=weights_params,
        boundary_params=boundary_params,
        sorting_params=sorting_params,
        saving_params=saving_params,
        bufsize=0.4,
    )

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
    if exit_before_run:
        logger.info("Exiting before running simulation.")
        return sim

    sim.run()

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
    num_elements = grid.num_elements
    degree = derham_opts.degree

    # get scalar data
    if MPI.COMM_WORLD.Get_rank() == 0:
        pa_data = os.path.join(env.path_out, "data")
        with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
            time = f["time"]["value"][()]
            E = f["scalar"]["electric_energy"][()]
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
            plt.title(f"{dt=}, {algo=}, {num_elements=}, {degree=}, {ppc=}")
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
        logger.info(f"Assertion for weak Landau damping passed ({rel_error =}).")

        shutil.rmtree(test_folder)


@pytest.mark.parametrize("control_variate", [False, True])
def test_energy_conservation(control_variate: bool, exit_before_run: bool = False):
    """``total_energy`` is conserved from :math:`t=0` on, with full-f and with delta-f (control variate) weights.

    A time step is :class:`~struphy.propagators.push_eta.PushEta` followed by the Crank-Nicolson step
    :class:`~struphy.propagators.vlasov_ampere_coupling.VlasovAmpereCoupling` (no B0). This checks two things that
    went wrong at :math:`t=0` only:

    * the initial field ``e = -grad(phi)`` must have its ghost regions in sync, otherwise the first push evaluates a
      wrong field and the total energy jumps in the first step (by ~1e-6 relative here, ~2e-4 with 8 elements);
    * with the control variate, ``kinetic_energy`` at :math:`t=0` must be computed with the delta-f weights, like
      at all later times (it was computed with the full-f weights, i.e. the energy of the background too).

    With full-f weights the scheme conserves the total energy to the solver tolerance. With the control variate
    the weights are updated after the Crank-Nicolson step, :math:`w_p = w_{0,p} - f_0(\\mathbf v_p) / (s_{0,p} N)`,
    which changes the delta-f kinetic energy by a Monte-Carlo estimate of zero plus an :math:`O(\\Delta t^2)` term
    per step (the energy that the field gives to the background markers). So the total delta-f energy is
    conserved only up to noise and time discretization error, and is checked with a loose tolerance and more
    markers.
    """
    amplitude = 0.1
    ppc = 400 if control_variate else 40

    model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)

    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "VlasovAmpereOneSpecies")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder=f"energy_conservation_cv_{control_variate}")

    time_opts = Time(dt=0.1, Tend=0.3)

    domain = domains.Cuboid(r1=12.56)
    # 16 elements, so that every rank has at least p=3 of them with 4 ranks (CI runs with clones on 4 ranks)
    grid = grids.TensorProductGrid(num_elements=(16, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))

    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=ppc, loading="sobol_standard"),
        weights_params=WeightsParameters(control_variate=control_variate),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(),
        saving_params=SavingParameters(),
        bufsize=0.4,
    )

    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.coupling_va.options = model.propagators.coupling_va.Options(
        solver_params=SolverParameters(tol=1e-12),
    )
    model.initial_poisson.options = model.initial_poisson.Options(
        stab_mat="M0",
        stab_eps=1e-6,
        solver_params=SolverParameters(tol=1e-12),
    )

    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    perturbation = perturbations.ModesCos(ls=(1,), amps=(amplitude,))
    model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, perturbation)))

    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )

    if exit_before_run:
        logger.info("Exiting before running simulation.")
        return sim

    sim.run()

    comm = MPI.COMM_WORLD

    if comm.Get_rank() == 0:
        with h5py.File(os.path.join(env.path_out, "data", "data_proc0.hdf5"), "r") as f:
            en_E = f["scalar"]["electric_energy"][()]
            en_tot = f["scalar"]["total_energy"][()]

        # energy is exchanged between field and particles ...
        assert abs(en_E[-1] - en_E[0]) > 1e-2 * en_E[0], f"{en_E =}"

        # ... and the total is conserved, including the first step
        drift = float(xp.max(xp.abs(en_tot - en_tot[0])) / abs(en_tot[0]))
        tol = 0.1 if control_variate else 1e-10
        assert drift < tol, f"Total energy not conserved: {en_tot =}, {drift =}."
        logger.info(f"Assertion for energy conservation passed ({control_variate =}, {drift =}).")

        shutil.rmtree(env.path_out)


if __name__ == "__main__":
    test_weak_Landau(do_plot=True)
    test_energy_conservation(control_variate=False)
    test_energy_conservation(control_variate=True)
