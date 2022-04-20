from mpi4py import MPI 

MPI_COMM = MPI.COMM_WORLD

import pytest
import numpy as np
import time

from struphy.geometry.domain_3d import Domain
from struphy.feec.psydac_derham import Derham_build
from struphy.psydac_linear_operators.preconditioner import MassMatrixPreConditioner as MassPre
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
def test_mass_preconditioner(Nel, p, spl_kind, mapping):

    map = mapping[0]
    params_map = mapping[1]

    DOMAIN = Domain(map, params_map)
    F_psy = DOMAIN.Psydac_mapping('F', **params_map)

    DR = Derham_build(Nel, p, spl_kind, F=F_psy, comm = MPI_COMM)

    DR.assemble_M0()
    DR.assemble_M1()

    v0 = StencilVector(DR.V0.vector_space)
    v0._data = np.random.rand(*v0._data.shape)

    v1 = BlockVector(DR.V1.vector_space)
    for v1i in v1:
        v1i._data = np.random.rand(*v1i._data.shape)

    inv_A0 = Invert(DR.M0, pc=None, tol=1e-9, maxiter=30000, verbose=False)
    A = Compose(inv_A0, DR.M0)
    assert np.allclose(A.dot(v0).toarray(), v0.toarray(), atol=1e-5)

    M0_pre = MassPre(DR.V0)
    inv_B0 = Invert(DR.M0, pc=M0_pre, tol=1e-8, maxiter=30000, verbose=False)
    B = Compose(inv_B0, DR.M0)
    assert np.allclose(B.dot(v0).toarray(), v0.toarray(), atol=1e-5)

    inv_A1 = Invert(DR.M1, pc=None, tol=1e-8, maxiter=30000, verbose=False)
    A = Compose(inv_A1, DR.M1)
    assert np.allclose(A.dot(v1).toarray(), v1.toarray(), atol=1e-5)

    M1_pre = MassPre(DR.V1)
    inv_B1 = Invert(DR.M1, pc=M1_pre, tol=1e-8, maxiter=30000, verbose=False)
    B = Compose(inv_B1, DR.M1)
    assert np.allclose(B.dot(v1).toarray(), v1.toarray(), atol=1e-5)

    print(f'Compare NUMITERS for Nel={Nel} and p={p} and mapping={map}:')
    print(f'Inverse M0 w/o pre: {inv_A0.info["niter"]}')
    print(f'Inverse M0 w/  pre: {inv_B0.info["niter"]}')
    print(f'Inverse M1 w/o pre: {inv_A1.info["niter"]}')
    print(f'Inverse M1 w/  pre: {inv_B1.info["niter"]}')

    assert inv_B0.info["niter"] < inv_A0.info["niter"]
    assert inv_B1.info["niter"] < inv_A1.info["niter"]


if __name__ == '__main__':
    test_mass_preconditioner(
        [12, 16, 4], [2, 3, 2], [False, True, True], ['cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}])