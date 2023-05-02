import pytest


@pytest.mark.parametrize('Nel', [[8, 10, 4]])
@pytest.mark.parametrize('p',   [[2, 3, 1]])
@pytest.mark.parametrize('spl_kind', [[False, True, True]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}],
    ['HollowTorus', {
        'a1': 1., 'a2': 2., 'R0': 3., 'tor_period': 1}],
    ['ShafranovDshapedCylinder', {
        'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}],
])
def test_composite_sum_scalar_inverse(Nel, p, spl_kind, mapping):
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

    print(f'type M0: {type(M0)}')
    print(f'shape M0: {M0.shape}')

    print(f'type M1: {type(M1)}')
    print(f'shape M1: {M1.shape}')
    
    # random vectors
    v0 = create_equal_random_arrays(derham.Vh_fem['0'], seed=4568)[1]
    v1 = create_equal_random_arrays(derham.Vh_fem['1'], seed=4568)[1]
    
    out0 = v0.space.zeros()
    out1 = v1.space.zeros()
    
    # ========= test IdentityOperator =================
    I0 = ID(M0.domain)
    I1 = ID(M1.domain)
    print(f'type I0: {type(I0)}')
    print(f'shape I0: {I0.shape}')
    print(f'type I1: {type(I1)}')
    print(f'shape I1: {I1.shape}')
    
    # not in-place
    assert np.allclose(I0.dot(v0).toarray(), v0.toarray())
    assert np.allclose(I1.dot(v1).toarray(), v1.toarray())
    # in-place
    I0.dot(v0, out=out0)
    I1.dot(v1, out=out1)
    assert np.allclose(out0.toarray(), v0.toarray())
    assert np.allclose(out1.toarray(), v1.toarray())
    
    # ========= test BoundaryOperator =================
    B0 = Boundary(M0.domain, space_id='H1')
    B1 = Boundary(M1.domain, space_id='Hcurl')
    print(f'type B0: {type(B0)}')
    print(f'shape B0: {B0.shape}')
    print(f'type B1: {type(B1)}')
    print(f'shape B1: {B1.shape}')
    
    # not in-place
    assert np.allclose(B0.dot(v0).toarray(), v0.toarray())
    assert np.allclose(B1.dot(v1).toarray(), v1.toarray())
    # in-place
    B0.dot(v0, out=out0)
    B1.dot(v1, out=out1)
    assert np.allclose(out0.toarray(), v0.toarray())
    assert np.allclose(out1.toarray(), v1.toarray())

    # ========= test CompositeLinearOperator ==========
    A = Compose(M1, derham.grad, M0)
    print(f'type A: {type(A)}')
    print(f'shape A: {A.shape}')
    
    r1 = M1.dot(derham.grad.dot(M0.dot(v0)))
    
    # not in-place
    assert np.allclose(A.dot(v0).toarray(), r1.toarray())
    # in-place
    A.dot(v0, out=out1)
    assert np.allclose(out1.toarray(), r1.toarray())
    
    # transposed
    AT = A.transpose()
    print(f'type AT: {type(AT)}')
    print(f'shape AT: {AT.shape}')
    
    r0 = M0.T.dot(derham.grad.T.dot(M1.T.dot(v1)))
    
    # not in-place
    assert np.allclose(AT.dot(v1).toarray(), r0.toarray())
    # in-place
    AT.dot(v1, out=out0)
    assert np.allclose(out0.toarray(), r0.toarray())

    
    # ========= test Sum-/ScaledLinearOperator ========== 
    B = Sum(M0, Compose(AT, A))
    BT = B.transpose()
    assert isinstance(B, LinOpWithTransp)
    assert isinstance(BT, LinOpWithTransp)

    C = Sum(M0, Multiply(-1., M0))
    
    r0 = np.zeros(*v0.toarray().shape)
    
    # not in-place
    assert np.allclose(C.dot(v0).toarray(), r0)
    # in-place
    C.dot(v0, out=out0)
    assert np.allclose(out0.toarray(), r0)

    # transposed
    CT = C.transpose()
    
    r0 = np.zeros(*v0.toarray().shape)
    
    # not in-place
    assert np.allclose(CT.dot(v0).toarray(), r0)
    # in-place
    CT.dot(v0, out=out0)
    assert np.allclose(out0.toarray(), r0)

    
    # ========= test InverseLinearOperator ========== 
    D = Compose(Invert(M0, pc=None, tol=1e-9, maxiter=30000), M0)
    
    # not-in-place
    assert np.allclose(D.dot(v0).toarray(), v0.toarray(), atol=1e-5)
    # in-place
    D.dot(v0, out=out0)
    assert np.allclose(out0.toarray(), v0.toarray(), atol=1e-5)

    # transposed
    DT = D.transpose()
    
    # not in-place
    assert np.allclose(DT.dot(v0).toarray(), v0.toarray(), atol=1e-5)
    # in-place
    DT.dot(v0, out=out0)
    assert np.allclose(out0.toarray(), v0.toarray(), atol=1e-5)

    # need smaller tolerance 1e-9 to pass assert with 1e-5 here:
    E = Compose(Invert(M1, pc=None, tol=1e-9, maxiter=30000), M1)
    print(f'type E: {type(E)}')
    print(f'shape E: {E.shape}')
    
    # not-in-place
    assert np.allclose(E.dot(v1).toarray(), v1.toarray(), atol=1e-5)
    # in-place
    E.dot(v1, out=out1)
    assert np.allclose(out1.toarray(), v1.toarray(), atol=1e-5)

    # transposed
    ET = E.transpose()
    
    # not-in-place
    assert np.allclose(ET.dot(v1).toarray(), v1.toarray(), atol=1e-5)
    # in-place
    ET.dot(v1, out=out1)
    assert np.allclose(out1.toarray(), v1.toarray(), atol=1e-5)


if __name__ == '__main__':
    test_composite_sum_scalar_inverse(
        [8, 10, 4], [2, 3, 2], [False, True, True], ['Colella', {
            'Lx': 1., 'Ly': 2., 'alpha': .5, 'Lz': 3.}])
