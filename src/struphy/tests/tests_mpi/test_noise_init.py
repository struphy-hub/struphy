import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 10, 12]])
@pytest.mark.parametrize('p', [[1, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True]])
@pytest.mark.parametrize('space', ['Hcurl', 'Hdiv', 'H1vec'])
@pytest.mark.parametrize('direction', ['e1', 'e2', 'e3'])
def test_noise_init(Nel, p, spl_kind, space, direction):
    '''Only tests 1d noise ('e1', 'e2', 'e3') !!'''

    from mpi4py import MPI
    import numpy as np

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays

    comm = MPI.COMM_WORLD
    assert comm.size >= 2
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence and field of space
    derham = Derham(Nel, p, spl_kind, comm=comm)
    field = derham.create_field('field', space)
    
    derham_np = Derham(Nel, p, spl_kind, comm=None)
    field_np = derham_np.create_field('field', space)

    # initial conditions
    init_params = {
        'type': 'noise',
        'noise': {
            'comps':
                {'field': [True, False, False]},
            'variation_in': direction,
            'amp': 0.0001,
            'seed': 1234,
        }
    }
    field.initialize_coeffs(init_params)
    field_np.initialize_coeffs(init_params)
    
    # print('#'*80)
    # print(f'npts={field.vector[0].space.npts}, npts_np={field_np.vector[0].space.npts}')
    # print(f'rank={rank}: nprocs={derham.domain_array[rank]}')
    # print(f'rank={rank}, field={field.vector[0].toarray_local().shape}, field_np={field_np.vector[0].toarray_local().shape}')
    #print(f'rank={rank}: \ncomp{0}={field.vector[0].toarray_local()}, \ncomp{0}_np={field_np.vector[0].toarray_local()}')
    
    compare_arrays(field.vector, [field_np.vector[n].toarray_local() for n in range(3)], rank)

    
if __name__ == '__main__':
    test_noise_init([4, 8, 6], [1, 1, 1], [True, True, True], 'Hcurl', 'e1')
