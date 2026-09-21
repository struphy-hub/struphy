import logging

import cunumpy as xp
import pytest
from feectools.ddm.mpi import MockComm
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt

from struphy import (
    BinningPlot,
    BoundaryParameters,
    LoadingParameters,
    SortingParameters,
    WeightsParameters,
    domains,
    equils,
    perturbations,
)
from struphy.fields_background.equils import ConstantVelocity
from struphy.geometry.base import Domain
from struphy.pic.particles import ParticlesSPH
from struphy import set_logging_level
set_logging_level(logging.WARNING)

logger = logging.getLogger("struphy")


@pytest.mark.parametrize("boxes_per_dim", [(24, 1, 1)])
@pytest.mark.parametrize("kernel", ["trigonometric_1d", "gaussian_1d", "linear_1d"])
@pytest.mark.parametrize("derivative", [0, 1])
@pytest.mark.parametrize("bc_x", ["periodic", "mirror", "fixed"])
@pytest.mark.parametrize("eval_pts", [11, 16])
@pytest.mark.parametrize("tesselation", [False, True])
def test_gyro_maxwellian_2d(do_plot=False):

    from struphy import Simulation
    from struphy import LoadingParameters, maxwellians
    from struphy.models import LinearMHDDriftkineticCC

    model = LinearMHDDriftkineticCC()
    equil = equils.HomogenSlab()
    
    sim = Simulation(model=model,
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
    bin_edges = (np.linspace(-5, 5, 32), np.linspace(0, 2, 32))
    err = model.energetic_ions.var.particles.show_distribution_function(components, bin_edges, do_plot=do_plot)
    print(f"2d {components = }, {err = }")
    assert err < 0.05

    components = [False] * 5
    components[0] = True
    components[1] = True
    bin_edges = (np.linspace(0, 1, 32), np.linspace(0, 1, 32))
    err = model.energetic_ions.var.particles.show_distribution_function(components, bin_edges, do_plot=do_plot)
    print(f"2d {components = }, {err = }")
    assert err < 0.31

    # Perturbations for (some) kinetic species
    # perturbation = perturbations.TorusModesCos()
    # maxwellian_1pt = maxwellians.GyroMaxwellian2D(n=(1.0, perturbation), equil=equil)
    # init = maxwellian_1pt + maxwellian_2
    # model.energetic_ions.var.add_initial_condition(init)

if __name__ == "__main__":
    test_gyro_maxwellian_2d(do_plot=True)