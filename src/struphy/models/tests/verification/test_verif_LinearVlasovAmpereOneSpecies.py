import logging
import os
import shutil

import cunumpy as xp
import h5py
import numpy as np
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
    grids,
    maxwellians,
    perturbations,
)
from struphy.models import LinearVlasovAmpereOneSpecies
from struphy.propagators.base import Propagator

logger = logging.getLogger("struphy")


def test_delta_f_initial_condition(exit_before_run: bool = False):
    """The delta-f markers must carry the density perturbation alone.

    A ``DeltaFParticles6D`` species solves for :math:`\\delta f = f - f_0`, so its weights
    are :math:`\\delta f / (s_0 N_p)` already and no control variate may be applied to them.
    Subtracting the background a second time offsets every weight by the same
    :math:`-f_0 / (s_0 N_p)`, which is a net charge: the markers then carry a uniform
    :math:`\\delta f` instead of the cosine, and the periodic Poisson problem solved in
    ``post_allocate`` becomes singular.

    This runs zero time steps -- everything asserted here is set up during allocation --
    and checks that

    1. the weights have no net charge, and
    2. the electric field built from them is the one the perturbation implies,
       :math:`\\int E^2 / 2 \\, \\textrm{d}x = A^2 r_1 / (4 k^2)`.
    """
    amplitude = 1e-3
    r1 = 12.56  # a box of length 4*pi, i.e. k = 2*pi/r1 = 0.5

    model = LinearVlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False, with_E0=False)

    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "LinearVlasovAmpereOneSpecies")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="delta_f_init")

    # no time stepping: post_allocate does the initial Poisson solve this test is about
    time_opts = Time(dt=0.05, Tend=0.0)

    domain = domains.Cuboid(r1=r1)
    grid = grids.TensorProductGrid(num_elements=(32, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))

    ppc = 2000
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=ppc, seed=1234),
        weights_params=WeightsParameters(),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True),
        saving_params=SavingParameters(),
        bufsize=0.4,
    )

    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.coupling_Eweights.options = model.propagators.coupling_Eweights.Options()
    # The constant mode of the periodic Poisson problem needs a finite stabilization: left at
    # the default the solver is free to wander along it, and the physical part of phi then
    # drowns in the solver tolerance applied to a solution of arbitrary size.
    model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0", stab_eps=1e-6)

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

    # 1. no net charge: the weights of a cosine perturbation cancel over the box, so their
    # mean is Monte-Carlo noise, small against their own spread. The control variate applied
    # by mistake made every weight equal, i.e. |mean| == max|w|.
    # Summed over the ranks of the clone: the markers are sorted into the domain of their rank,
    # so on more than two ranks each one holds a part of the cosine with a net charge of its
    # own. Each clone holds a random subset of all markers, so its mean is as good an estimator
    # of the net charge as the global one.
    particles = model.kinetic_ions.var.particles
    weights = xp.to_numpy(particles.weights)
    sums = np.array([weights.sum(), (weights**2).sum(), weights.size], dtype=float)
    if particles.mpi_comm is not None:
        particles.mpi_comm.Allreduce(MPI.IN_PLACE, sums, op=MPI.SUM)
    mean = sums[0] / sums[2]
    rms = np.sqrt(sums[1] / sums[2])
    assert abs(mean) < 0.1 * rms, (
        f"Assertion for delta-f weights failed: net charge {mean =} is not small against {rms =}; "
        "the markers do not carry the density perturbation alone."
    )
    logger.info(f"Assertion for delta-f weights passed ({mean =}, {rms =}).")

    # 2. the field energy of the cosine mode, int E^2/2 dx with E = A/k
    k = 2.0 * xp.pi / r1
    energy_exact = amplitude**2 * r1 / (4.0 * k**2)

    if comm.Get_rank() == 0:
        energy = float(model.scalars.dct["en_E"].value[0])
        rel_error = abs(energy - energy_exact) / energy_exact
        assert rel_error < 0.05, (
            f"Assertion for the initial delta-f field energy failed: {energy =} vs. {energy_exact =}."
        )
        logger.info(f"Assertion for the initial delta-f field energy passed ({rel_error =}).")

        shutil.rmtree(test_folder)


def test_energy_conservation(exit_before_run: bool = False):
    """The field-weights coupling must conserve the total energy ``en_E + en_w``.

    With only :class:`~struphy.propagators.push_eta.PushEta` and
    :class:`~struphy.propagators.efield_weights_coupling.EfieldWeightsCoupling` (no B0, no E0),
    the Crank-Nicolson step conserves ``en_tot`` up to the solver tolerance, provided the weight
    update, the accumulated matrix, the E-field source and ``en_w`` all use the same weight
    convention :math:`w_p = \\delta f_p / (s_{0,p} N_p)`. At :math:`t=0` the perturbation energy is
    :math:`\\alpha^2 v_{th}^2 / 2 \\int \\delta f^2 / f_0 \\, \\textrm{d}x \\textrm{d}v = A^2 r_1 / 4`.
    """
    amplitude = 0.1
    r1 = 12.56

    model = LinearVlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False, with_E0=False)

    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "LinearVlasovAmpereOneSpecies")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="energy_conservation")

    time_opts = Time(dt=0.1, Tend=0.3)

    domain = domains.Cuboid(r1=r1)
    grid = grids.TensorProductGrid(num_elements=(16, 1, 1))
    derham_opts = DerhamOptions(degree=(3, 1, 1))

    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=100, seed=1234),
        weights_params=WeightsParameters(),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=(8, 1, 1), do_sort=True),
        saving_params=SavingParameters(),
        bufsize=0.4,
    )

    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.coupling_Eweights.options = model.propagators.coupling_Eweights.Options()
    model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0", stab_eps=1e-6)

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
            en_E = f["scalar"]["en_E"][()]
            en_w = f["scalar"]["en_w"][()]
            en_tot = f["scalar"]["en_tot"][()]

        # 1. initial perturbation energy (was off by a factor Np^2)
        en_w_exact = amplitude**2 * r1 / 4.0
        rel_error = abs(en_w[0] - en_w_exact) / en_w_exact
        assert rel_error < 0.05, f"Initial perturbation energy failed: {en_w[0] =} vs. {en_w_exact =}."

        # 2. energy is exchanged between field and particles ...
        assert abs(en_E[-1] - en_E[0]) > 1e-3 * en_tot[0]

        # 3. ... and the total is conserved
        drift = float(xp.max(xp.abs(en_tot - en_tot[0])) / en_tot[0])
        assert drift < 1e-6, f"Total energy not conserved: {en_tot =}, {drift =}."
        logger.info(f"Assertion for energy conservation passed ({rel_error =}, {drift =}).")

        shutil.rmtree(test_folder)


def test_initial_poisson_no_background():
    r"""The delta-f markers carry f - f0, so the initial Poisson solve must not subtract the background n0.

    Without perturbation the delta-f weights vanish and, with :math:`\phi = 0` at the walls in :math:`x`, so does the
    initial potential. Subtracting :math:`n_0` gives :math:`-\Delta \phi = -\alpha^2 n_0 / \varepsilon`, i.e.
    :math:`\phi = -x(1 - x)/2` with root mean square 0.09. This runs zero time steps.
    """
    model = LinearVlasovAmpereOneSpecies(alpha=1.0, epsilon=1.0, with_B0=False, with_E0=False)

    test_folder = os.path.join(os.getcwd(), "struphy_verification_tests")
    out_folders = os.path.join(test_folder, "LinearVlasovAmpereOneSpecies")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="initial_poisson_no_background")

    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=100, seed=1234),
        weights_params=WeightsParameters(),
        boundary_params=BoundaryParameters(bc=("reflect", "periodic", "periodic")),
        sorting_params=SortingParameters(boxes_per_dim=(8, 1, 1), do_sort=True),
        saving_params=SavingParameters(),
        bufsize=0.4,
    )
    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.coupling_Eweights.options = model.propagators.coupling_Eweights.Options()
    model.initial_poisson.options = model.initial_poisson.Options()
    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    # a separate initial condition without perturbation: its density is set to zero for the delta-f markers,
    # while the background f0 keeps n0 = 1
    model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, None)))

    sim = Simulation(
        model=model,
        env=env,
        time_opts=Time(dt=0.05, Tend=0.0),
        domain=domains.Cuboid(),
        grid=grids.TensorProductGrid(num_elements=(16, 1, 1)),
        derham_opts=DerhamOptions(degree=(3, 1, 1), bcs=(("dirichlet", "dirichlet"), None, None)),
    )
    sim.run()

    # root mean square of phi over the unit cube (collective, on all ranks)
    phi = model.em_fields.phi.spline.vector
    phi_rms = float(xp.sqrt(Propagator.mass_ops.M0.dot_inner(phi, phi)))
    logger.info(f"{phi_rms =}")
    assert phi_rms < 1e-6, f"Initial potential without perturbation: {phi_rms =} (a background n0 gives 0.09)."

    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        shutil.rmtree(test_folder)


if __name__ == "__main__":
    test_delta_f_initial_condition()
    test_initial_poisson_no_background()
    test_energy_conservation()
