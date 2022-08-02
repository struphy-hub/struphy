import pytest


@pytest.mark.parametrize('Nel', [[8, 12, 4]])
@pytest.mark.parametrize('p',   [[2, 3, 2]])
@pytest.mark.parametrize('spl_kind', [[True, True, True]])
@pytest.mark.parametrize('mapping', [
    ['cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}],
    ['shafranov_dshaped', {
        'x0': 1., 'y0': 2., 'z0': 3., 'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}],
])
@pytest.mark.parametrize('use_fft', [True, False])
def test_mass_preconditioner(Nel, p, spl_kind, mapping, use_fft):

    import numpy as np
    from mpi4py import MPI

    from struphy.geometry.domain_3d import Domain
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.mass_psydac import WeightedMass
    from struphy.psydac_api.preconditioner import MassMatrixPreConditioner as MassPre
    from struphy.psydac_api.linear_operators import InverseLinearOperator as Invert

    from psydac.linalg.stencil import StencilVector
    from psydac.linalg.block import BlockVector

    MPI_COMM = MPI.COMM_WORLD

    map = mapping[0]
    params_map = mapping[1]

    domain = Domain(map, params_map)

    derham = Derham(Nel, p, spl_kind, comm=MPI_COMM)
    
    mass = WeightedMass(derham, domain)

    mass.assemble_M0()
    mass.assemble_M1()
    mass.assemble_M2()
    mass.assemble_M3()

    derham_spaces = [derham.V0, derham.V1, derham.V2, derham.V3]
    derham_M = [mass.M0, mass.M1, mass.M2, mass.M3]

    v = []

    v += [StencilVector(derham.V0.vector_space)]
    v[-1]._data = np.random.rand(*v[-1]._data.shape)

    v += [BlockVector(derham.V1.vector_space)]
    for v1i in v[-1]:
        v1i._data = np.random.rand(*v1i._data.shape)

    v += [BlockVector(derham.V2.vector_space)]
    for v1i in v[-1]:
        v1i._data = np.random.rand(*v1i._data.shape)

    v += [StencilVector(derham.V3.vector_space)]
    v[-1]._data = np.random.rand(*v[-1]._data.shape)

    M_pre = []
    for space in derham_spaces:
        M_pre += [MassPre(space, use_fft=use_fft)]

    for n, (M, M_p, vn) in enumerate(zip(derham_M, M_pre, v)):

        if map == 'cuboid':
            assert np.allclose(M.toarray(), M_p.matrix.toarray())
            print(f'Matrix assertion for space {n} case "cuboid" passed.')

        inv_A = Invert(M, pc=M_p, tol=1e-8, maxiter=5000)
        wn = inv_A.dot(vn)

        if map == 'cuboid':
            assert inv_A.info['niter'] == 2
            print(f'Solver assertions for space {n} case "cuboid" passed.')

        inv_A_nopc = Invert(M, pc=None, tol=1e-8, maxiter=30000)
        wn_nopc = inv_A_nopc.dot(vn)

        print(f'Inverse of M{n}: w/ pre {inv_A.info["niter"]} and w/o pre {inv_A_nopc.info["niter"]}')

        assert inv_A.info['success']
        assert inv_A.info["niter"] < inv_A_nopc.info["niter"]


if __name__ == '__main__':
    test_mass_preconditioner(
        [12, 16, 4], [2, 3, 2], [True, True, True], ['cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}], use_fft=False)