from time import time

import numpy as np
import pytest
from mpi4py import MPI

from struphy.feec.psydac_derham import Derham
from struphy.geometry import domains
from struphy.pic.particles import ParticlesSPH


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("Nel", [[8, 9, 10]])
@pytest.mark.parametrize("p", [[2, 3, 4]])
@pytest.mark.parametrize(
    "spl_kind", [[False, False, True], [False, True, False], [True, False, True], [True, True, False]]
)
@pytest.mark.parametrize(
    "mapping",
    [
        ["Cuboid", {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}],
    ],
)
@pytest.mark.parametrize("Np", [200000])
def test_evaluation(Nel, p, spl_kind, mapping, Np, verbose=False):
    mpi_comm = MPI.COMM_WORLD

    # DOMAIN object
    dom_type = mapping[0]
    dom_params = mapping[1]
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    params_sorting = {"nx": 3, "ny": 3, "nz": 3, "eps": 0.25}
    params_loading = {"seed": 1607, "moments": "degenerate", "spatial": "uniform"}

    bckgr_params = {
        "type": "ConstantVelocity",
        "ConstantVelocity": {"density_profile": "affine", "ux": 1.0, "uy": 0.0, "uz": 0.0, "n": 1.0, "n1": 0.1},
        "pforms": ["vol", None],
    }

    particles = ParticlesSPH(
        "test_particles",
        Np=Np,
        bc=["periodic", "periodic", "periodic"],
        loading="pseudo_random",
        eps=10., # Lots a buffering needed since only 3*3*3 box
        comm=mpi_comm,
        loading_params=params_loading,
        domain=domain,
        bckgr_params=bckgr_params,
        sorting_params=params_sorting,
    )

    particles.draw_markers(sort=False)
    particles.mpi_sort_markers()
    particles.initialize_weights()
    eta1 = np.array([0.5])
    eta2 = np.array([0.5])
    eta3 = np.array([0.5])
    test_eval = particles.eval_density(eta1, eta2, eta3, h=1/(3*mpi_comm.Get_size()))

    assert abs(test_eval[0] - 1.15) < 3.0e-2


if __name__ == "__main__":
    test_evaluation(
        [8, 9, 10],
        [2, 3, 4],
        [True, True, True],
        ["Cuboid", {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}],
        100000,
    )
