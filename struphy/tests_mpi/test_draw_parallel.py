import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 10]])
@pytest.mark.parametrize('p', [[2, 3, 4]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, False], [True, False, False]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}],
    ['ShafranovDshapedCylinder', {
        'x0': 1., 'y0': 2., 'z0': 3., 'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}]
])
def test_draw(Nel, p, spl_kind, mapping, ppc=10):
    '''Asserts whether all particles are on the correct process after `particles.send_recv_markers()`.'''

    from mpi4py import MPI
    import numpy as np

    import numpy as np

    from struphy.geometry import domains
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.pic.particles import Particles6D

    comm = MPI.COMM_WORLD
    assert comm.size >= 2
    rank = comm.Get_rank()

    seed = int(np.random.rand()*1000)
    loading_params = {'type': 'pseudo_random', 'seed': seed, 'dir_particles': 'dir',
                      'moms_params': [1., 0., 0., 0., 1., 1., 1.]}

    marker_params = {'type': 'fullorbit', 'ppc': ppc,
                     'loading': loading_params, 'n_bins': [32, 32], 'v_max': 5.}

    # Domain object
    dom_type = mapping[0]
    dom_params = mapping[1]
    domain_class = getattr(domains, dom_type)
    domain = domain_class(dom_params)

    # Psydac discrete Derham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm)

    if rank == 0:
        print()
        print('Domain decomposition according to : ')
        print(derham.domain_array)

    # create particles
    particles = Particles6D('energetic_ions', marker_params,
                            domain, derham.domain_array, comm)

    comm.Barrier()
    print('Number of particles w/wo holes on each process before sorting : ')
    print('Rank', rank, ':', particles.n_mks_loc,
          particles.n_mks_loc_with_holes)

    # sort particles according to domain decomposition
    comm.Barrier()
    particles.send_recv_markers(do_test=True)

    comm.Barrier()
    print('Number of particles w/wo holes on each process after sorting : ')
    print('Rank', rank, ':', particles.n_mks_loc,
          particles.n_mks_loc_with_holes)

    # are all markers in the correct domain?
    conds = np.logical_and(particles.markers[:, :3] > derham.domain_array[rank, ::3],
                           particles.markers[:, :3] < derham.domain_array[rank, 1::3])
    holes = particles.markers[:, 0] == -1.
    stay = np.all(conds, axis=1)

    error_mks = particles.markers[np.logical_and(~stay, ~holes)]

    print(
        f'rank {rank} | markers not on correct process: {np.nonzero(np.logical_and(~stay, ~holes))} \n corresponding positions:\n {error_mks[:, :3]}')

    assert error_mks.size == 0


if __name__ == '__main__':
    test_draw([8, 9, 10], [2, 3, 4], [False, False, True], ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}])
