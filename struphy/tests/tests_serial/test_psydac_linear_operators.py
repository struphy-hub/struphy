import pytest


@pytest.mark.parametrize('Nel', [[8, 10, 4]])
@pytest.mark.parametrize('p',   [[2, 3, 1]])
@pytest.mark.parametrize('spl_kind', [[False, True, True]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}],
    ['Colella', {
        'Lx': 1., 'Ly': 2., 'alpha': .5, 'Lz': 3.}],
    ['HollowTorus', {
        'a1': 1., 'a2': 2., 'R0': 3., 'tor_period': 1}],
    ['ShafranovDshapedCylinder', {
        'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}],
])
def test_composite_sum_scalar_inverse(Nel, p, spl_kind, mapping):
    
    from mpi4py import MPI
    import numpy as np

    from struphy.geometry import domains
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.mass import WeightedMassOperators
    from struphy.psydac_api.linear_operators import LinOpWithTransp
    from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose
    from struphy.psydac_api.linear_operators import SumLinearOperator as Sum
    from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply
    from struphy.psydac_api.linear_operators import InverseLinearOperator as Invert

    from psydac.linalg.stencil import StencilVector
    from psydac.linalg.block import BlockVector

    dom_type = mapping[0]
    dom_params = mapping[1]

    domain_class = getattr(domains, dom_type)
    domain = domain_class(dom_params)

    derham = Derham(Nel, p, spl_kind, comm=MPI.COMM_WORLD)

    mass = WeightedMassOperators(derham, domain)

    print(f'type M0: {type(mass.M0)}')
    print(f'shape M0: {mass.M0.shape}')

    print(f'type M1: {type(mass.M1)}')
    print(f'shape M1: {mass.M1.shape}')

    v0 = StencilVector(derham.Vh['0'])
    v0._data = np.random.rand(*v0._data.shape)

    v1 = BlockVector(derham.Vh['1'])
    for v1i in v1:
        v1i._data = np.random.rand(*v1i._data.shape)

    A = Compose(mass.M1, derham.grad, mass.M0)
    print(f'type A: {type(A)}')
    print(f'shape A: {A.shape}')
    assert np.allclose(A.dot(v0).toarray(), mass.M1.dot(
        derham.grad.dot(mass.M0.dot(v0))).toarray())

    AT = A.transpose()
    print(f'type AT: {type(AT)}')
    print(f'shape AT: {AT.shape}')
    assert np.allclose(AT.dot(v1).toarray(), mass.M0.transpose().dot(
        derham.grad.transpose().dot(mass.M1.transpose().dot(v1))).toarray())

    B = Sum(mass.M0, Compose(AT, A))
    BT = B.transpose()
    assert isinstance(B, LinOpWithTransp)
    assert isinstance(BT, LinOpWithTransp)

    C = Sum(mass.M0, Multiply(-1., mass.M0))
    assert np.allclose(C.dot(v0).toarray(), np.zeros(*v0.toarray().shape))

    CT = C.transpose()
    assert np.allclose(CT.dot(v0).toarray(), np.zeros(*v0.toarray().shape))

    D = Compose(Invert(mass.M0, pc=None, tol=1e-9, maxiter=30000), mass.M0)
    assert np.allclose(D.dot(v0).toarray(), v0.toarray(), atol=1e-5)

    DT = D.transpose()
    assert np.allclose(DT.dot(v0).toarray(), v0.toarray(), atol=1e-5)

    # Need smaller tolerance 1e-9 to pass assert with 1e-5 here:
    E = Compose(Invert(mass.M1, pc=None, tol=1e-9, maxiter=30000), mass.M1)
    print(f'type E: {type(E)}')
    print(f'shape E: {E.shape}')
    res = E.dot(v1)
    assert np.allclose(res.toarray(), v1.toarray(), atol=1e-5)

    ET = E.transpose()
    assert np.allclose(ET.dot(v1).toarray(), v1.toarray(), atol=1e-5)


if __name__ == '__main__':
    test_composite_sum_scalar_inverse(
        [8, 10, 4], [2, 3, 2], [False, True, True], ['Colella', {
        'Lx': 1., 'Ly': 2., 'alpha': .5, 'Lz': 3.}])
