import pytest
import numpy as np

from struphy.geometry.domain_3d import Domain
from struphy.feec.psydac_derham import Derham_build
from struphy.psydac_linear_operators.linear_operators import LinOpWithTransp
from struphy.psydac_linear_operators.linear_operators import CompositeLinearOperator as Compose
from struphy.psydac_linear_operators.linear_operators import SumLinearOperator as Sum
from struphy.psydac_linear_operators.linear_operators import ScalarTimesLinearOperator as Multiply
from struphy.psydac_linear_operators.linear_operators import InverseLinearOperator as Invert

from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector


@pytest.mark.parametrize('Nel', [[8, 10, 4]])
@pytest.mark.parametrize('p',   [[2, 3, 2]])
@pytest.mark.parametrize('spl_kind', [[False, True, True]])
@pytest.mark.parametrize('mapping', [
    ['cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}],
    ['colella', {
        'Lx': 1., 'Ly': 2., 'alpha': .5, 'Lz': 3.}],
    ['hollow_torus', {
        'a1': 1., 'a2': 2., 'R0': 3.}],
    ['shafranov_dshaped', {
        'x0': 1., 'y0': 2., 'z0': 3., 'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}],
])
def test_composite_sum_scalar_inverse(Nel, p, spl_kind, mapping):

    map = mapping[0]
    params_map = mapping[1]

    DOMAIN = Domain(map, params_map)
    F_psy = DOMAIN.Psydac_mapping('F', **params_map)

    DR = Derham_build(Nel, p, spl_kind, F=F_psy)

    DR.assemble_M0()
    DR.assemble_M1()

    print(f'type M0: {type(DR.M0)}')
    print(f'shape M0: {DR.M0.shape}')

    print(f'type M1: {type(DR.M1)}')
    print(f'shape M1: {DR.M1.shape}')

    v0 = StencilVector(DR.V0.vector_space)
    v0._data = np.random.rand(*v0._data.shape)

    v1 = BlockVector(DR.V1.vector_space)
    for v1i in v1:
        v1i._data = np.random.rand(*v1i._data.shape)

    A = Compose(DR.M1, DR.grad, DR.M0)
    print(f'type A: {type(A)}')
    print(f'shape A: {A.shape}')
    assert np.allclose(A.dot(v0).toarray(), DR.M1.dot(
        DR.grad.dot(DR.M0.dot(v0))).toarray())

    AT = A.transpose()
    print(f'type AT: {type(AT)}')
    print(f'shape AT: {AT.shape}')
    assert np.allclose(AT.dot(v1).toarray(), DR.M0.transpose().dot(
        DR.grad.transpose().dot(DR.M1.transpose().dot(v1))).toarray())

    B = Sum(DR.M0, Compose(AT, A))
    BT = B.transpose()
    assert isinstance(B, LinOpWithTransp)
    assert isinstance(BT, LinOpWithTransp)

    C = Sum(DR.M0, Multiply(-1., DR.M0))
    assert np.allclose(C.dot(v0).toarray(), np.zeros(*v0.toarray().shape))

    CT = C.transpose()
    assert np.allclose(CT.dot(v0).toarray(), np.zeros(*v0.toarray().shape))

    D = Compose(Invert(DR.M0, pc=None, tol=1e-9, maxiter=30000, verbose=False), DR.M0)
    assert np.allclose(D.dot(v0).toarray(), v0.toarray(), atol=1e-5)

    DT = D.transpose()
    assert np.allclose(DT.dot(v0).toarray(), v0.toarray(), atol=1e-5)

    # Need smaller tolerance 1e-9 to pass assert with 1e-5 here:
    E = Compose(Invert(DR.M1, pc=None, tol=1e-9, maxiter=30000, verbose=False), DR.M1)
    print(f'type E: {type(E)}')
    print(f'shape E: {E.shape}')
    res = E.dot(v1)
    assert np.allclose(res.toarray(), v1.toarray(), atol=1e-5)

    ET = E.transpose()
    assert np.allclose(ET.dot(v1).toarray(), v1.toarray(), atol=1e-5)


if __name__ == '__main__':
    test_composite_sum_scalar_inverse(
        [8, 10, 4], [2, 3, 2], [False, True, True], ['colella', {
        'Lx': 1., 'Ly': 2., 'alpha': .5, 'Lz': 3.}])
