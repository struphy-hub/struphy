import pytest


@pytest.mark.parametrize('Nel', [[8, 12, 4]])
@pytest.mark.parametrize('p',   [[2, 3, 1]])
@pytest.mark.parametrize('spl_kind', [[True, True, True], [False, False, False]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 0., 'r1': 2., 'l2': 0., 'r2': 3., 'l3': 0., 'r3': 4.}],
    ['HollowCylinder', {
        'a1': .1, 'a2': 2., 'R0': 0., 'Lz': 3.}]])
def test_mass_preconditioner(Nel, p, spl_kind, mapping):

    import numpy as np
    from mpi4py import MPI

    from struphy.geometry import domains
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.mass_psydac import WeightedMass
    from struphy.psydac_api.preconditioner import MassMatrixPreconditioner
    from struphy.psydac_api.linear_operators import InverseLinearOperator

    from psydac.linalg.stencil import StencilVector
    from psydac.linalg.block import BlockVector

    MPI_COMM = MPI.COMM_WORLD

    domain_class = getattr(domains, mapping[0])
    domain = domain_class(mapping[1])

    derham = Derham(Nel, p, spl_kind, comm=MPI_COMM)
    derham_spaces = [derham.V0, derham.V1, derham.V2, derham.V3, derham.V0vec]
    
    # assemble mass matrices in V0, V1, V2 and V3
    mass = WeightedMass(derham, domain)

    mass.assemble_M0()
    mass.assemble_M1()
    mass.assemble_M2()
    mass.assemble_M3()
    mass.assemble_Mv()
    
    derham_M = [mass.M0, mass.M1, mass.M2, mass.M3, mass.Mv]

    # create random vectors
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
    
    v += [BlockVector(derham.V0vec.vector_space)]
    for v1i in v[-1]:
        v1i._data = np.random.rand(*v1i._data.shape)

    # assemble preconditioners
    M_pre = []
    
    M_pre += [MassMatrixPreconditioner(derham, 'V0', mass._fun_M0)]
    M_pre += [MassMatrixPreconditioner(derham, 'V1', mass._fun_M1)]
    M_pre += [MassMatrixPreconditioner(derham, 'V2', mass._fun_M2)]
    M_pre += [MassMatrixPreconditioner(derham, 'V3', mass._fun_M3)]
    M_pre += [MassMatrixPreconditioner(derham, 'V0vec', mass._fun_Mv)]

    for n, (M, M_p, vn) in enumerate(zip(derham_M, M_pre, v)):
        
        if n == 4: n = 'v'

        if domain.kind_map == 10 or domain.kind_map == 11:
            assert np.allclose(M._operator.toarray(), M_p.matrix.toarray())
            print(f'Matrix assertion for space {n} case "Cuboid/HollowCylinder" passed.')

        inv_A = InverseLinearOperator(M, pc=M_p, tol=1e-8, maxiter=5000)
        wn = inv_A.dot(vn)

        if domain.kind_map == 10 or domain.kind_map == 11:
            assert inv_A.info['niter'] == 2
            print(f'Solver assertions for space {n} case "Cuboid/HollowCylinder" passed.')

        inv_A_nopc = InverseLinearOperator(M, pc=None, tol=1e-8, maxiter=30000)
        wn_nopc = inv_A_nopc.dot(vn)

        print(f'Inverse of M{n}: w/ pre {inv_A.info["niter"]} and w/o pre {inv_A_nopc.info["niter"]}')

        assert inv_A.info['success']
        assert inv_A.info["niter"] < inv_A_nopc.info["niter"]


if __name__ == '__main__':
    test_mass_preconditioner(
        [12, 16, 4], [2, 3, 2], [False, False, False], ['Cuboid', {
        'l1': 0., 'r1': 2., 'l2': 0., 'r2': 3., 'l3': 0., 'r3': 4.}])
    #test_mass_preconditioner(
    #    [12, 16, 4], [2, 3, 2], [False, True, False], ['HollowCylinder', {
    #    'a1': .1, 'a2': 2., 'R0': 0., 'Lz': 3.}])
    #test_mass_preconditioner(
    #    [12, 16, 4], [2, 3, 2], [False, True, True], ['Orthogonal', {
    #    'Lx': 1., 'Ly': 2., 'alpha': .1, 'Lz': 4.}])