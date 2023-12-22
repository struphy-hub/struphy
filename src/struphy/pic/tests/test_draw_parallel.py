import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 10]])
@pytest.mark.parametrize('p', [[1, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, False], [True, False, False]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}],
    ['ShafranovDshapedCylinder', {
        'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}]
])
def test_draw(Nel, p, spl_kind, mapping, ppc=10):
    '''Asserts whether all particles are on the correct process after `particles.mpi_sort_markers()`.'''

    from mpi4py import MPI
    import numpy as np

    import numpy as np

    from struphy.geometry import domains
    from struphy.feec.psydac_derham import Derham
    from struphy.pic.particles import Particles6D

    comm = MPI.COMM_WORLD
    assert comm.size >= 2
    rank = comm.Get_rank()

    seed = int(np.random.rand()*1000)

    # Domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # Psydac discrete Derham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm)

    if rank == 0:
        print()
        print('Domain decomposition according to : ')
        print(derham.domain_array)

    # create particles
    loading_params = {'type': 'pseudo_random',
                      'seed': seed,
                      'moments': [0., 0., 0., 1., 1., 1.],
                      'spatial': 'uniform'}
    bc_params = {'type' : ['periodic', 'periodic', 'periodic']}
    marker_params = {'ppc': ppc,
                     'eps': .25,
                     'loading': loading_params,
                     'bc': bc_params,
                     'domain': domain}
    init_params = {'type': 'Maxwellian6DUniform', 'Maxwellian6DUniform': {}}

    particles = Particles6D('energetic_ions', **marker_params, derham=derham)
    particles.draw_markers()

    # test weights
    particles.initialize_weights(init_params)
    _vdim = particles.vdim
    _w0 = particles.weights
    print('Test weights:')
    print(f'rank {rank}:', _w0.shape, np.min(_w0), np.max(_w0))

    comm.Barrier()
    print('Number of particles w/wo holes on each process before sorting : ')
    print('Rank', rank, ':', particles.n_mks_loc,
          particles.markers.shape[0])

    # sort particles according to domain decomposition
    comm.Barrier()
    particles.mpi_sort_markers(do_test=True)

    comm.Barrier()
    print('Number of particles w/wo holes on each process after sorting : ')
    print('Rank', rank, ':', particles.n_mks_loc,
          particles.markers.shape[0])

    # are all markers in the correct domain?
    conds = np.logical_and(particles.markers[:, :3] > derham.domain_array[rank, 0::3],
                           particles.markers[:, :3] < derham.domain_array[rank, 1::3])
    holes = particles.markers[:, 0] == -1.
    stay = np.all(conds, axis=1)

    error_mks = particles.markers[np.logical_and(~stay, ~holes)]

    print(
        f'rank {rank} | markers not on correct process: {np.nonzero(np.logical_and(~stay, ~holes))} \n corresponding positions:\n {error_mks[:, :3]}')

    assert error_mks.size == 0


if __name__ == '__main__':
    # test_draw([8, 9, 10], [2, 3, 4], [False, False, True], ['Cuboid', {
    #     'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}])
    test_draw([8, 9, 10], [2, 3, 4], [False, False, True], ['Cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}])
