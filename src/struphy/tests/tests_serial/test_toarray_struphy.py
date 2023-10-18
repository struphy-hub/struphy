import pytest


@pytest.mark.parametrize('Nel', [[8, 10, 4],[12,5,2]])
@pytest.mark.parametrize('p',   [[2, 3, 1],[3,4,1]])
@pytest.mark.parametrize('spl_kind', [[False, True, True],[False, False, True]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}]])
def test_toarray_struphy(Nel, p, spl_kind, mapping):
    """
    TODO
    """

    from mpi4py import MPI
    import numpy as np

    from struphy.geometry import domains
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.mass import WeightedMassOperators
    from struphy.psydac_api.utilities import create_equal_random_arrays
    from struphy.psydac_api.linear_operators import LinOpWithTransp
    from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose
    from struphy.psydac_api.linear_operators import SumLinearOperator as Sum
    from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply
    from struphy.psydac_api.linear_operators import InverseLinearOperator as Invert
    from struphy.psydac_api.linear_operators import IdentityOperator as ID
    from struphy.psydac_api.linear_operators import BoundaryOperator as Boundary

    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # create derham object
    derham = Derham(Nel, p, spl_kind, comm=MPI.COMM_WORLD)

    # assemble mass matrices in V0 and V1
    mass = WeightedMassOperators(derham, domain)
    
    M0 = mass.M0
    M1 = mass.M1
    M2 = mass.M2
    M3 = mass.M3
    
    # random vectors
    v0 = create_equal_random_arrays(derham.Vh_fem['0'], seed=4568)[1]
    v1 = create_equal_random_arrays(derham.Vh_fem['1'], seed=4568)[1]
    v2 = create_equal_random_arrays(derham.Vh_fem['2'], seed=4568)[1]
    v3 = create_equal_random_arrays(derham.Vh_fem['3'], seed=4568)[1]
    
    # ========= test toarray_struphy =================
    #Get the matrix form of the linear operators M0 to M3
    M0arr = M0.toarray_struphy()
    M1arr = M1.toarray_struphy()
    M2arr = M2.toarray_struphy()
    M3arr = M3.toarray_struphy()
    
    v0arr = v0.toarray()
    v1arr = v1.toarray()
    v2arr = v2.toarray()
    v3arr = v3.toarray()
    
    # not in-place
    assert np.allclose(M0.dot(v0).toarray(), np.matmul(M0arr,v0arr))
    assert np.allclose(M1.dot(v1).toarray(), np.matmul(M1arr,v1arr))
    assert np.allclose(M2.dot(v2).toarray(), np.matmul(M2arr,v2arr))
    assert np.allclose(M3.dot(v3).toarray(), np.matmul(M3arr,v3arr))
    
    print('test_toarry_struphy passed!')
    
    # assert np.allclose(out1.toarray(), v1.toarray(), atol=1e-5)


if __name__ == '__main__':
    test_toarray_struphy(
        [8, 10, 4], [2, 3, 2], [False, True, True], ['Colella', {
            'Lx': 1., 'Ly': 2., 'alpha': .5, 'Lz': 3.}])