import logging
import os
import shutil

import cunumpy as xp
import h5py
import pytest
from matplotlib import pyplot as plt
from maybempi import MPI

from struphy import (
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
    equils,
    grids,
    maxwellians,
    perturbations,
)
from struphy.models import VlasovPoissonElectrostaticAdiabatic
from struphy.propagators.base import Propagator

logger = logging.getLogger("struphy")


@pytest.mark.parametrize("quasi_neutral", [True, False])
def test_ion_acoustic_Landau(quasi_neutral: bool, do_plot: bool = False, exit_before_run: bool = False):
    r"""Verification test for Landau damping of an ion-acoustic wave along :math:`\mathbf B_0`.

    For :math:`\mathbf k \parallel \mathbf B_0` the gyration decouples and the dispersion relation of the model is

    .. math::

        \frac{\varepsilon^2 k^2}{\alpha^2 n_0} + \frac{1}{Z T_e} + \frac{1}{T_i}\left[1 + \zeta Z(\zeta)\right] = 0\,,
        \qquad \zeta = \frac{\omega}{\sqrt 2 k \sqrt{T_i}}\,.

    With :math:`Z = 1`, :math:`T_i = 1`, :math:`T_e = 4`, :math:`k = 0.5` and in the quasi-neutral limit this is
    exactly the dispersion relation of weak (electron) Landau damping at :math:`k = 0.5`, so
    :math:`\omega = 1.4157`, :math:`\gamma = -0.1534` (see ``test_verif_VlasovAmpereOneSpecies.py``).
    With the vacuum term (:math:`\varepsilon = 1`, :math:`\alpha = 2`) the root is :math:`\omega = 1.3356`,
    :math:`\gamma = -0.1936`.
    """
    # light-weight model instance (alpha only matters with the vacuum term, epsilon drops out for k parallel to B0)
    model = VlasovPoissonElectrostaticAdiabatic(alpha=2.0, epsilon=1.0)

    # environment options
    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "VlasovPoissonElectrostaticAdiabatic")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder=f"ion_acoustic_qn_{quasi_neutral}")

    # time stepping
    time_opts = Time(dt=0.05, Tend=15)

    # geometry: k = 2 pi / r1 = 0.5 along B0 = B0 e_x
    r1 = 12.56
    domain = domains.Cuboid(r1=r1)

    # fluid equilibrium: n0 = 1 and T_e = p / n0 = beta |B0|^2 / 2 = 4
    equil = equils.HomogenSlab(B0x=1.0, B0z=0.0, beta=8.0, n0=1.0)

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
    saving_params = SavingParameters()

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
    model.propagators.poisson.options = model.propagators.poisson.Options(quasi_neutral=quasi_neutral)
    model.propagators.push_v_E.options = model.propagators.push_v_E.Options()
    model.propagators.push_vxb.options = model.propagators.push_vxb.Options()

    # background (T_i = 1) and initial condition
    background = maxwellians.Maxwellian3D(n=(1.0, None))
    model.kinetic_ions.var.add_background(background)

    perturbation = perturbations.ModesCos(ls=(1,), amps=(1e-3,))
    init = maxwellians.Maxwellian3D(n=(1.0, perturbation))
    model.kinetic_ions.var.add_initial_condition(init)

    # instance of simulation
    sim = Simulation(
        model=model,
        env=env,
        time_opts=time_opts,
        equil=equil,
        domain=domain,
        grid=grid,
        derham_opts=derham_opts,
    )

    if exit_before_run:
        logger.info("Exiting before running simulation.")
        return sim

    sim.run()

    # exact damping rate
    gamma = -0.1534 if quasi_neutral else -0.1936

    dt = time_opts.dt
    if MPI.COMM_WORLD.Get_rank() == 0:
        pa_data = os.path.join(env.path_out, "data")
        with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
            time = f["time"]["value"][()]
            E = f["scalar"]["en_phi"][()]
        logE = xp.log10(E)

        # maxima of the field energy
        dEdt = (xp.roll(logE, -1) - xp.roll(logE, 1))[1:-1] / (2.0 * dt)
        zeros = dEdt * xp.roll(dEdt, -1) < 0.0
        maxima_inds = xp.logical_and(zeros, dEdt > 0.0)
        maxima = logE[1:-1][maxima_inds]
        t_maxima = time[1:-1][maxima_inds]

        if do_plot:
            plt.figure(figsize=(12, 8))
            plt.plot(time, logE, label="numerical")
            plt.plot(t_maxima[:5], maxima[:5], "or", markersize=10)
            plt.plot(time, logE[0] + 2 * gamma * time / xp.log(10), "k--", label="exact rate")
            plt.legend()
            plt.xlabel("time")
            plt.show()

        # linear fit of log10(E) ~ 2 gamma t / ln(10)
        linfit = xp.polyfit(t_maxima[:5], maxima[:5], 1)
        gamma_num = linfit[0] * xp.log(10) / 2

        rel_error = xp.abs(gamma_num - gamma) / xp.abs(gamma)
        assert rel_error < 0.22, f"Ion-acoustic Landau damping failed: {gamma_num =} vs. {gamma =}."
        logger.info(f"Ion-acoustic Landau damping passed ({gamma_num =}, {rel_error =}).")

        shutil.rmtree(test_folder)


def test_neutral_full_f_background():
    r"""A uniform full-f plasma of ions with charge number :math:`Z = 2` is neutral, so the potential is at noise level.

    Without the control variate, the Poisson solve subtracts the neutralizing background :math:`n_0` from the deposited
    ion density; the charge number is part of the source coefficient, not of the background. A background
    :math:`Z n_0` would give :math:`\phi = -Z (Z - 1) \varepsilon T_0`, here :math:`-2`, everywhere.
    """
    model = VlasovPoissonElectrostaticAdiabatic(charge_number=2, mass_number=4.0, alpha=1.0, epsilon=1.0)

    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "VlasovPoissonElectrostaticAdiabatic")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="neutral_full_f")

    # n0 = 1 and T_e = p / n0 = beta |B0|^2 / 2 = 1
    equil = equils.HomogenSlab(B0z=1.0, beta=2.0, n0=1.0)

    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=200, seed=1234),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=(8, 1, 1), do_sort=True),
        saving_params=SavingParameters(),
        bufsize=0.4,
    )
    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.poisson.options = model.propagators.poisson.Options()
    model.propagators.push_v_E.options = model.propagators.push_v_E.Options()
    model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))

    sim = Simulation(
        model=model,
        env=env,
        time_opts=Time(dt=0.05, Tend=0.05),
        equil=equil,
        domain=domains.Cuboid(),
        grid=grids.TensorProductGrid(num_elements=(16, 1, 1)),
        derham_opts=DerhamOptions(degree=(3, 1, 1)),
    )
    sim.run()

    # root mean square of phi over the unit cube (collective, on all ranks)
    phi = model.em_fields.phi.spline.vector
    phi_rms = float(xp.sqrt(Propagator.mass_ops.M0.dot_inner(phi, phi)))
    logger.info(f"{phi_rms =}")
    assert phi_rms < 0.5, f"Full-f background not neutralizing: {phi_rms =} (a background Z n0 gives 2)."

    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        shutil.rmtree(test_folder)


if __name__ == "__main__":
    test_ion_acoustic_Landau(True, do_plot=True)
    test_neutral_full_f_background()
