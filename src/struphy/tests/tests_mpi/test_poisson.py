import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 10]])
@pytest.mark.parametrize('p', [[2, 3, 4]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, True], [True, False, True], [True, True, True]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}],
    ['Cuboid', {
        'l1': 0., 'r1': 2., 'l2': 0., 'r2': 3., 'l3': 0., 'r3': 4.}],
])
def test_accum_poisson(Nel, p, spl_kind, mapping, Np=1000, verbose=False):
    '''DRAFT: test the accumulation of the rhs (H1-space) in Poisson's equation .

    Tests:

        * Whether all weights are initialized as \sqrt(g) = const. (Cuboid mappings).
        * Whether the sum oaver all MC integrals is 1.
    '''

    import numpy as np
    from mpi4py import MPI

    from struphy.geometry import domains
    from struphy.feec.psydac_derham import Derham

    from struphy.pic.particles import Particles6D
    from struphy.pic.accumulation.particles_to_grid import AccumulatorVector

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()

    # domain object
    dom_type = mapping[0]
    dom_params = mapping[1]
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # DeRham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm)

    if mpi_rank == 0:
        print('Domain decomposition according to', derham.domain_array)

    # load distributed markers first and use Send/Receive to make global marker copies for the legacy routines
    params_markers = {'Np': Np,
                      'eps': .25,
                      'loading': {'type': 'pseudo_random',
                                  'seed': 1607,
                                  'moments': [0., 0., 0., 1., 1., 1.],
                                  'spatial': 'uniform'},
                      'domain': domain,
                      'derham': derham
                      }
    init_params = {'type': 'Maxwellian6DUniform', 'Maxwellian6DUniform': {}}

    particles = Particles6D('test_particles', **params_markers)
    particles.draw_markers()
    particles.mpi_sort_markers()
    particles.initialize_weights(init_params)

    _vdim = particles.vdim
    _w0 = particles.weights

    print('Test weights:')
    print(f'rank {mpi_rank}:', _w0.shape, np.min(_w0), np.max(_w0))

    _sqrtg = domain.jacobian_det(0.5, 0.5, 0.5)

    assert np.isclose(np.min(_w0), _sqrtg)
    assert np.isclose(np.max(_w0), _sqrtg)

    # instance of the accumulator
    acc = AccumulatorVector(derham, domain, 'H1', 'poisson')
    acc.accumulate(particles, 1., 1.)

    # sum all MC integrals
    _sum = np.empty(1, dtype=float)
    _sum[0] = np.sum(acc.vectors[0].toarray())
    mpi_comm.Allreduce(MPI.IN_PLACE, _sum, op=MPI.SUM)

    print(f'rank {mpi_rank}:', _sum)

    assert np.isclose(_sum, _sqrtg)


if __name__ == '__main__':
    test_accum_poisson([8, 5, 6], [2, 2, 3], [True]*3, ['Cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 2., 'l3': 0., 'r3': 1.}], Np=1000, verbose=False)
