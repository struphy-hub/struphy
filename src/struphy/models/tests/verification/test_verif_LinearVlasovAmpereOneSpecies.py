import logging
import os
import shutil

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

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
    # Checked per rank: each one holds a random subset of the markers, so its own mean is
    # as good an estimator of the net charge as the global one.
    weights = xp.asarray(model.kinetic_ions.var.particles.weights)
    mean = float(weights.mean())
    rms = float(xp.sqrt((weights**2).mean()))
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


if __name__ == "__main__":
    test_delta_f_initial_condition()
