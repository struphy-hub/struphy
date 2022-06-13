import pytest

@pytest.mark.parametrize('Nel', [[8, 12, 4]])
@pytest.mark.parametrize('p',   [[2, 3, 2]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True]])
@pytest.mark.parametrize('mapping', [
    ['cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}],
    ['colella', {
        'Lx' : 1., 'Ly' : 2., 'alpha' : .1, 'Lz' : 3.}],
    ['hollow_torus', {
        'a1': 1., 'a2': 2., 'R0': 3.}]])
def test_mass(Nel, p, spl_kind, mapping):
    
    import numpy as np

    from sympde.topology import Cube, Square, Line, Derham

    from psydac.api.discretization import discretize
    from psydac.linalg.stencil import StencilMatrix, StencilVector
    from psydac.linalg.block import BlockMatrix, BlockVector
    from psydac.fem.basic import FemSpace

    from struphy.feec.spline_space import Spline_space_1d
    from struphy.feec.spline_space import Tensor_spline_space
    from struphy.geometry.domain_3d import Domain
    from struphy.psydac_api.psydac_derham import DerhamBuild
    
    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    
    print('number of processes : ', mpi_size)
    
    # mapping
    domain = Domain(mapping[0], mapping[1])
    F = domain.Psydac_mapping('F', **mapping[1])

    # derham object
    derham = DerhamBuild(Nel, p, spl_kind, der_as_mat=True, F=F, comm=mpi_comm)
    
    # assemble mass matrices
    derham.assemble_M0_nonsymb(domain)
    derham.assemble_M1_nonsymb(domain)
    derham.assemble_M2_nonsymb(domain)
    derham.assemble_M3_nonsymb(domain)
    derham.assemble_Mv_nonsymb(domain)
    
    # compare to old STRUPHY
    spaces = [Spline_space_1d(Nel, p, spl, nq_el) for Nel, p, spl, nq_el in zip(Nel, p, spl_kind, [p[0] + 1, p[1] + 1, p[2] + 1])]

    space = Tensor_spline_space(spaces)

    space.assemble_Mk(domain, 'V0')
    space.assemble_Mk(domain, 'V1')
    space.assemble_Mk(domain, 'V2')
    space.assemble_Mk(domain, 'V3')
    space.assemble_Mk(domain, 'Vv')
    
    np.random.seed(1234)
    x0_str = np.random.rand(space.Ntot_0form)
    x1_str = np.random.rand(sum(space.Ntot_1form))
    x2_str = np.random.rand(sum(space.Ntot_2form))
    x3_str = np.random.rand(space.Ntot_3form)
    xv_str = np.random.rand(3*space.Ntot_0form)
    
    x0_psy = StencilVector(derham.V0.vector_space)
    x1_psy = BlockVector(derham.V1.vector_space)
    x2_psy = BlockVector(derham.V2.vector_space)
    x3_psy = StencilVector(derham.V3.vector_space)
    xv_psy = BlockVector(derham.V0vec.vector_space)
    
    x0_psy[x0_psy.starts[0]:x0_psy.ends[0] + 1, x0_psy.starts[1]:x0_psy.ends[1] + 1, x0_psy.starts[2]:x0_psy.ends[2] + 1] = space.extract_0(x0_str)[x0_psy.starts[0]:x0_psy.ends[0] + 1, x0_psy.starts[1]:x0_psy.ends[1] + 1, x0_psy.starts[2]:x0_psy.ends[2] + 1]
    
    x1_psy[0][x1_psy[0].starts[0]:x1_psy[0].ends[0] + 1, x1_psy[0].starts[1]:x1_psy[0].ends[1] + 1, x1_psy[0].starts[2]:x1_psy[0].ends[2] + 1] = space.extract_1(x1_str)[0][x1_psy[0].starts[0]:x1_psy[0].ends[0] + 1, x1_psy[0].starts[1]:x1_psy[0].ends[1] + 1, x1_psy[0].starts[2]:x1_psy[0].ends[2] + 1]
    
    x1_psy[1][x1_psy[1].starts[0]:x1_psy[1].ends[0] + 1, x1_psy[1].starts[1]:x1_psy[1].ends[1] + 1, x1_psy[1].starts[2]:x1_psy[1].ends[2] + 1] = space.extract_1(x1_str)[1][x1_psy[1].starts[0]:x1_psy[1].ends[0] + 1, x1_psy[1].starts[1]:x1_psy[1].ends[1] + 1, x1_psy[1].starts[2]:x1_psy[1].ends[2] + 1]
    
    x1_psy[2][x1_psy[2].starts[0]:x1_psy[2].ends[0] + 1, x1_psy[2].starts[1]:x1_psy[2].ends[1] + 1, x1_psy[2].starts[2]:x1_psy[1].ends[2] + 1] = space.extract_1(x1_str)[2][x1_psy[2].starts[0]:x1_psy[2].ends[0] + 1, x1_psy[2].starts[1]:x1_psy[2].ends[1] + 1, x1_psy[2].starts[2]:x1_psy[2].ends[2] + 1]
    
    x2_psy[0][x2_psy[0].starts[0]:x2_psy[0].ends[0] + 1, x2_psy[0].starts[1]:x2_psy[0].ends[1] + 1, x2_psy[0].starts[2]:x2_psy[0].ends[2] + 1] = space.extract_2(x2_str)[0][x2_psy[0].starts[0]:x2_psy[0].ends[0] + 1, x2_psy[0].starts[1]:x2_psy[0].ends[1] + 1, x2_psy[0].starts[2]:x2_psy[0].ends[2] + 1]
    
    x2_psy[1][x2_psy[1].starts[0]:x2_psy[1].ends[0] + 1, x2_psy[1].starts[1]:x2_psy[1].ends[1] + 1, x2_psy[1].starts[2]:x2_psy[1].ends[2] + 1] = space.extract_2(x2_str)[1][x2_psy[1].starts[0]:x2_psy[1].ends[0] + 1, x2_psy[1].starts[1]:x2_psy[1].ends[1] + 1, x2_psy[1].starts[2]:x2_psy[1].ends[2] + 1]
    
    x2_psy[2][x2_psy[2].starts[0]:x2_psy[2].ends[0] + 1, x2_psy[2].starts[1]:x2_psy[2].ends[1] + 1, x2_psy[2].starts[2]:x2_psy[1].ends[2] + 1] = space.extract_2(x2_str)[2][x2_psy[2].starts[0]:x2_psy[2].ends[0] + 1, x2_psy[2].starts[1]:x2_psy[2].ends[1] + 1, x2_psy[2].starts[2]:x2_psy[2].ends[2] + 1]
    
    x3_psy[x3_psy.starts[0]:x3_psy.ends[0] + 1, x3_psy.starts[1]:x3_psy.ends[1] + 1, x3_psy.starts[2]:x3_psy.ends[2] + 1] = space.extract_3(x3_str)[x3_psy.starts[0]:x3_psy.ends[0] + 1, x3_psy.starts[1]:x3_psy.ends[1] + 1, x3_psy.starts[2]:x3_psy.ends[2] + 1]
    
    xv_psy[0][xv_psy[0].starts[0]:xv_psy[0].ends[0] + 1, xv_psy[0].starts[1]:xv_psy[0].ends[1] + 1, xv_psy[0].starts[2]:xv_psy[0].ends[2] + 1] = space.extract_v(xv_str)[0][xv_psy[0].starts[0]:xv_psy[0].ends[0] + 1, xv_psy[0].starts[1]:xv_psy[0].ends[1] + 1, xv_psy[0].starts[2]:xv_psy[0].ends[2] + 1]
    
    xv_psy[1][xv_psy[1].starts[0]:xv_psy[1].ends[0] + 1, xv_psy[1].starts[1]:xv_psy[1].ends[1] + 1, xv_psy[1].starts[2]:xv_psy[1].ends[2] + 1] = space.extract_v(xv_str)[1][xv_psy[1].starts[0]:xv_psy[1].ends[0] + 1, xv_psy[1].starts[1]:xv_psy[1].ends[1] + 1, xv_psy[1].starts[2]:xv_psy[1].ends[2] + 1]
    
    xv_psy[2][xv_psy[2].starts[0]:xv_psy[2].ends[0] + 1, xv_psy[2].starts[1]:xv_psy[2].ends[1] + 1, xv_psy[2].starts[2]:xv_psy[1].ends[2] + 1] = space.extract_v(xv_str)[2][xv_psy[2].starts[0]:xv_psy[2].ends[0] + 1, xv_psy[2].starts[1]:xv_psy[2].ends[1] + 1, xv_psy[2].starts[2]:xv_psy[2].ends[2] + 1]
    
    r0_str = space.M0(x0_str)
    r1_str = space.M1(x1_str)
    r2_str = space.M2(x2_str)
    r3_str = space.M3(x3_str)
    rv_str = space.Mv(xv_str)
    
    r0_psy = derham.M0.dot(x0_psy)
    r1_psy = derham.M1.dot(x1_psy)
    r2_psy = derham.M2.dot(x2_psy)
    r3_psy = derham.M3.dot(x3_psy)
    rv_psy = derham.Mv.dot(xv_psy)
    
    assert np.allclose(space.extract_0(r0_str)[x0_psy.starts[0]:x0_psy.ends[0] + 1, x0_psy.starts[1]:x0_psy.ends[1] + 1, x0_psy.starts[2]:x0_psy.ends[2] + 1], r0_psy[x0_psy.starts[0]:x0_psy.ends[0] + 1, x0_psy.starts[1]:x0_psy.ends[1] + 1, x0_psy.starts[2]:x0_psy.ends[2] + 1])
    
    assert np.allclose(space.extract_1(r1_str)[0][x1_psy[0].starts[0]:x1_psy[0].ends[0] + 1, x1_psy[0].starts[1]:x1_psy[0].ends[1] + 1, x1_psy[0].starts[2]:x1_psy[0].ends[2] + 1], r1_psy[0][x1_psy[0].starts[0]:x1_psy[0].ends[0] + 1, x1_psy[0].starts[1]:x1_psy[0].ends[1] + 1, x1_psy[0].starts[2]:x1_psy[0].ends[2] + 1])
    
    assert np.allclose(space.extract_1(r1_str)[1][x1_psy[1].starts[0]:x1_psy[1].ends[0] + 1, x1_psy[1].starts[1]:x1_psy[1].ends[1] + 1, x1_psy[1].starts[2]:x1_psy[1].ends[2] + 1], r1_psy[1][x1_psy[1].starts[0]:x1_psy[1].ends[0] + 1, x1_psy[1].starts[1]:x1_psy[1].ends[1] + 1, x1_psy[1].starts[2]:x1_psy[1].ends[2] + 1])
    
    assert np.allclose(space.extract_1(r1_str)[2][x1_psy[2].starts[0]:x1_psy[2].ends[0] + 1, x1_psy[2].starts[1]:x1_psy[2].ends[1] + 1, x1_psy[2].starts[2]:x1_psy[1].ends[2] + 1], r1_psy[2][x1_psy[2].starts[0]:x1_psy[2].ends[0] + 1, x1_psy[2].starts[1]:x1_psy[2].ends[1] + 1, x1_psy[2].starts[2]:x1_psy[1].ends[2] + 1])
    
    assert np.allclose(space.extract_2(r2_str)[0][x2_psy[0].starts[0]:x2_psy[0].ends[0] + 1, x2_psy[0].starts[1]:x2_psy[0].ends[1] + 1, x2_psy[0].starts[2]:x2_psy[0].ends[2] + 1], r2_psy[0][x2_psy[0].starts[0]:x2_psy[0].ends[0] + 1, x2_psy[0].starts[1]:x2_psy[0].ends[1] + 1, x2_psy[0].starts[2]:x2_psy[0].ends[2] + 1])
    
    assert np.allclose(space.extract_2(r2_str)[1][x2_psy[1].starts[0]:x2_psy[1].ends[0] + 1, x2_psy[1].starts[1]:x2_psy[1].ends[1] + 1, x2_psy[1].starts[2]:x2_psy[1].ends[2] + 1], r2_psy[1][x2_psy[1].starts[0]:x2_psy[1].ends[0] + 1, x2_psy[1].starts[1]:x2_psy[1].ends[1] + 1, x2_psy[1].starts[2]:x2_psy[1].ends[2] + 1])
    
    assert np.allclose(space.extract_2(r2_str)[2][x2_psy[2].starts[0]:x2_psy[2].ends[0] + 1, x2_psy[2].starts[1]:x2_psy[2].ends[1] + 1, x2_psy[2].starts[2]:x2_psy[1].ends[2] + 1], r2_psy[2][x2_psy[2].starts[0]:x2_psy[2].ends[0] + 1, x2_psy[2].starts[1]:x2_psy[2].ends[1] + 1, x2_psy[2].starts[2]:x2_psy[1].ends[2] + 1])
    
    assert np.allclose(space.extract_3(r3_str)[x3_psy.starts[0]:x3_psy.ends[0] + 1, x3_psy.starts[1]:x3_psy.ends[1] + 1, x3_psy.starts[2]:x3_psy.ends[2] + 1], r3_psy[x3_psy.starts[0]:x3_psy.ends[0] + 1, x3_psy.starts[1]:x3_psy.ends[1] + 1, x3_psy.starts[2]:x3_psy.ends[2] + 1])
    
    assert np.allclose(space.extract_v(rv_str)[0][xv_psy[0].starts[0]:xv_psy[0].ends[0] + 1, xv_psy[0].starts[1]:xv_psy[0].ends[1] + 1, xv_psy[0].starts[2]:xv_psy[0].ends[2] + 1], rv_psy[0][xv_psy[0].starts[0]:xv_psy[0].ends[0] + 1, xv_psy[0].starts[1]:xv_psy[0].ends[1] + 1, xv_psy[0].starts[2]:xv_psy[0].ends[2] + 1])
    
    assert np.allclose(space.extract_v(rv_str)[1][xv_psy[1].starts[0]:xv_psy[1].ends[0] + 1, xv_psy[1].starts[1]:xv_psy[1].ends[1] + 1, xv_psy[1].starts[2]:xv_psy[1].ends[2] + 1], rv_psy[1][xv_psy[1].starts[0]:xv_psy[1].ends[0] + 1, xv_psy[1].starts[1]:xv_psy[1].ends[1] + 1, xv_psy[1].starts[2]:xv_psy[1].ends[2] + 1])
    
    assert np.allclose(space.extract_v(rv_str)[2][xv_psy[2].starts[0]:xv_psy[2].ends[0] + 1, xv_psy[2].starts[1]:xv_psy[2].ends[1] + 1, xv_psy[2].starts[2]:xv_psy[1].ends[2] + 1], rv_psy[2][xv_psy[2].starts[0]:xv_psy[2].ends[0] + 1, xv_psy[2].starts[1]:xv_psy[2].ends[1] + 1, xv_psy[2].starts[2]:xv_psy[1].ends[2] + 1])
    

    
if __name__ == '__main__':
    test_mass([8, 6, 4], [2, 2, 2], [False, True, True], ['colella', {'Lx' : 1., 'Ly' : 2., 'alpha' : .1, 'Lz' : 3.}])