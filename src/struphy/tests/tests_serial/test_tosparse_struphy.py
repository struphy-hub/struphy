import pytest
import time


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
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.utilities import create_equal_random_arrays

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
    start = time.time()
    M0arr = M0.tosparse_struphy("csr")
    M1arr = M1.tosparse_struphy("csc")
    M2arr = M2.tosparse_struphy("bsr")
    M3arr = M3.tosparse_struphy("lil")
    M0arrad = M0.tosparse_struphy("dok")
    M1arrad = M1.tosparse_struphy("coo")
    M2arrad = M2.tosparse_struphy("dia")
    end = time.time()
    print("Time converting to sparse = "+str(end-start))
    
    v0arr = v0.toarray()
    v1arr = v1.toarray()
    v2arr = v2.toarray()
    v3arr = v3.toarray()
    
    # not in-place
    assert np.allclose(M0.dot(v0).toarray(), M0arr.dot(v0arr))
    assert np.allclose(M1.dot(v1).toarray(), M1arr.dot(v1arr))
    assert np.allclose(M2.dot(v2).toarray(), M2arr.dot(v2arr))
    assert np.allclose(M3.dot(v3).toarray(), M3arr.dot(v3arr))
    assert np.allclose(M0.dot(v0).toarray(), M0arrad.dot(v0arr))
    assert np.allclose(M1.dot(v1).toarray(), M1arrad.dot(v1arr))
    assert np.allclose(M2.dot(v2).toarray(), M2arrad.dot(v2arr))
    
    print('test_tosparse_struphy passed!')
    


if __name__ == '__main__':
    test_toarray_struphy(
        [8, 10, 4], [2, 3, 2], [False, True, True], ['Colella', {
            'Lx': 1., 'Ly': 2., 'alpha': .5, 'Lz': 3.}])