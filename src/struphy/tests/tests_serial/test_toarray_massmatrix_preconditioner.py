import pytest


@pytest.mark.parametrize('Nel', [[8, 10, 4], [12, 5, 2]])
@pytest.mark.parametrize('p',   [[2, 3, 1], [3, 4, 1]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [False, False, True]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], ['Cuboid', {
            'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}]])
def test_toarray_linearsolver(Nel, p, spl_kind, mapping):
    """
    TODO
    """

    from mpi4py import MPI
    import numpy as np

    from struphy.geometry import domains
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.mass import WeightedMassOperators
    from struphy.psydac_api.utilities import create_equal_random_arrays
    from struphy.psydac_api.linear_operators import InverseLinearOperator as Invert
    from struphy.psydac_api import preconditioner

    # create domain object
    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # create derham object
    derham = Derham(Nel, p, spl_kind, comm=MPI.COMM_WORLD)

    # assemble mass matrices
    mass = WeightedMassOperators(derham, domain)

    M0 = mass.M0
    M1 = mass.M1
    M2 = mass.M2
    M3 = mass.M3

    # Assemble the inverse mass matrices
    M0i = Invert(M0, pc=None, tol=1e-9, maxiter=30000)
    M1i = Invert(M1, pc=None, tol=1e-9, maxiter=30000)
    M2i = Invert(M2, pc=None, tol=1e-9, maxiter=30000)
    M3i = Invert(M3, pc=None, tol=1e-9, maxiter=30000)

    # random vectors
    v0 = create_equal_random_arrays(derham.Vh_fem['0'], seed=4568)[1]
    v1 = create_equal_random_arrays(derham.Vh_fem['1'], seed=4568)[1]
    v2 = create_equal_random_arrays(derham.Vh_fem['2'], seed=4568)[1]
    v3 = create_equal_random_arrays(derham.Vh_fem['3'], seed=4568)[1]

    # ========= test toarray for Mass matrix preconditioner =================

    v0arr = v0.toarray()
    v1arr = v1.toarray()
    v2arr = v2.toarray()
    v3arr = v3.toarray()

    # get the mass matrix preconditioner. Which should be the inverse of that mass matrix
    pc_class = getattr(preconditioner, 'MassMatrixPreconditioner')
    pc0 = pc_class(M0)
    pc1 = pc_class(M1)
    pc2 = pc_class(M2)
    pc3 = pc_class(M3)
    # get their matrix form
    pc0arr = pc0.toarray()
    pc1arr = pc1.toarray()
    pc2arr = pc2.toarray()
    pc3arr = pc3.toarray()

    # If everything works properly both statements should be computing the product between the inverse of the mass matrix M_j
    # and the random vector v_j
    assert np.allclose(M0i.dot(v0).toarray(),
                       np.matmul(pc0arr, v0arr), atol=1e-9)
    assert np.allclose(M1i.dot(v1).toarray(),
                       np.matmul(pc1arr, v1arr), atol=1e-9)
    assert np.allclose(M2i.dot(v2).toarray(),
                       np.matmul(pc2arr, v2arr), atol=1e-9)
    assert np.allclose(M3i.dot(v3).toarray(),
                       np.matmul(pc3arr, v3arr), atol=1e-9)

    print('test_toarry for linear solvers passed!')

    # assert np.allclose(out1.toarray(), v1.toarray(), atol=1e-5)


if __name__ == '__main__':
    test_toarray_linearsolver(
        [8, 10, 4], [2, 3, 2], [False, True, True], ['Cuboid', {
            'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}])
