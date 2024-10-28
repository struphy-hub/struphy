from struphy.geometry import domains
import numpy as np
from mpi4py import MPI
from time import time
from struphy.feec.psydac_derham import Derham
from struphy.pic.particles import HydroParticles

import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 10]])
@pytest.mark.parametrize('p', [[2, 3, 4]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, False], [True, False, True], [True, True, False]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], ])
@pytest.mark.parametrize('Np', [10000])
def test_evaluation(Nel, p, spl_kind, mapping, Np, verbose=False):

    mpi_comm = MPI.COMM_WORLD
    # assert mpi_comm.size >= 2
    rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # DOMAIN object
    dom_type = mapping[0]
    dom_params = mapping[1]
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # DeRham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm, domain=domain)
    params_markers = {'Np': Np, 'eps': .25,
                      'loading': {'type': 'pseudo_random', 'seed': 1607, 'moments': 'degenerate', 'spatial': 'uniform'}
                      }
    params_sorting = {'nx': 3, 'ny': 3, 'nz': 3, 'eps': 0.25}

    bckgr_params = {'type': 'Constant6D',
                'Constant6D': {'n' : .1},
                'pforms' : ['vol', None]}

    particles = HydroParticles(
        'test_particles', 
        **params_markers, 
        derham=derham, 
        domain=domain,
        bckgr_params=bckgr_params, 
        sorting_params=params_sorting)
    particles.draw_markers(sort=False)
    particles.mpi_sort_markers()
    particles.initialize_weights()
    eta1 = np.array([.5])
    eta2 = np.array([.5])
    eta3 = np.array([.5])
    test_eval = particles.eval_density(eta1,eta2,eta3)

    assert abs(test_eval[0]-.1)<1e-2



if __name__ == '__main__':
    test_evaluation([8, 9, 10], [2, 3, 4], [False, True, False], ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], 10000)
