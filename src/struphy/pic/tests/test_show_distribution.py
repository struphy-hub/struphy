import logging

import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)

logger = logging.getLogger("struphy")


def test_gyro_maxwellian_2d(do_plot=False):

    from struphy import LoadingParameters, Simulation, equils, maxwellians
    from struphy.models import LinearMHDDriftkineticCC

    model = LinearMHDDriftkineticCC()
    equil = equils.HomogenSlab()

    sim = Simulation(
        model=model,
        equil=equil,
    )

    loading_params = LoadingParameters(Np=100000, seed=3928)
    model.energetic_ions.set_markers(loading_params=loading_params)

    # Background for kinetic species
    maxwellian_1 = maxwellians.GyroMaxwellian2D(n=(1.0, None))
    # maxwellian_2 = maxwellians.GyroMaxwellian2D(n=(0.1, None))
    # background = maxwellian_1 + maxwellian_2
    model.energetic_ions.var.add_background(maxwellian_1)

    sim.allocate()

    import numpy as np

    # the background with B0="equil" has the equilibrium of the simulation attached
    assert model.energetic_ions.var.backgrounds is maxwellian_1
    assert np.allclose(maxwellian_1.eval_B0(np.full((3, 4), 0.5)), equil.absB0(np.full((3, 4), 0.5)))

    for i in range(5):
        components = [False] * 5
        components[i] = True
        if i < 3:
            bin_edges = (np.linspace(0, 1, 32),)
        elif i == 3:
            bin_edges = (np.linspace(-5, 5, 32),)
        else:
            bin_edges = (np.linspace(0, 5, 32),)
        err = model.energetic_ions.var.particles.show_distribution_function(components, bin_edges, do_plot=do_plot)
        print(f"1d {components = }, {err = }")
        assert err < 0.05

    components = [False] * 5
    components[3] = True
    components[4] = True
    bin_edges = (np.linspace(-5, 5, 32), np.linspace(0, 5, 32))
    err = model.energetic_ions.var.particles.show_distribution_function(components, bin_edges, do_plot=do_plot)
    print(f"2d {components = }, {err = }")
    assert err < 0.05

    components = [False] * 5
    components[0] = True
    components[1] = True
    bin_edges = (np.linspace(0, 1, 32), np.linspace(0, 1, 32))
    err = model.energetic_ions.var.particles.show_distribution_function(components, bin_edges, do_plot=do_plot)
    print(f"2d {components = }, {err = }")
    # noise dominated (~100 markers per bin); ~0.31 with B0=2.0, ~0.32 with the local |B_0| = 1 of HomogenSlab
    assert err < 0.33

    # Perturbations for (some) kinetic species
    # perturbation = perturbations.TorusModesCos()
    # maxwellian_1pt = maxwellians.GyroMaxwellian2D(n=(1.0, perturbation), equil=equil)
    # init = maxwellian_1pt + maxwellian_2
    # model.energetic_ions.var.add_initial_condition(init)


if __name__ == "__main__":
    test_gyro_maxwellian_2d(do_plot=True)
