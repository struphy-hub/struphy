import pytest


@pytest.mark.parametrize('Nel', [[8, 9, 6], [5, 6, 20]])
@pytest.mark.parametrize('p', [[3, 2, 1], [4, 3, 2]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [False, True, False]])
def test_spaces(Nel, p, spl_kind):

    from struphy.psydac_api.psydac_derham import Derham
    from struphy.polar.basic import PolarDerhamSpace, PolarVector 

    derham = Derham(Nel, p, spl_kind)

    print('polar V0:')
    V = PolarDerhamSpace(derham, 'H1')
    print('dimensions (parent, polar):', derham.V0.nbasis, V.dimension)
    print(V.dtype)
    print(V.zeros(), '\n')
    a = PolarVector(V)
    a.pol[0][:] = 1.
    a.tp[:] = 1.
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2*a).toarray())
    print((a*2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print() 

    print('polar V1:')
    V = PolarDerhamSpace(derham, 'Hcurl')
    print('dimensions (parent, polar):', derham.V1.nbasis, V.dimension)
    print(V.dtype)
    print(V.zeros(), '\n')
    a = PolarVector(V)
    a.pol[0][:] = 1.
    a.pol[1][:] = 2.
    a.pol[2][:] = 3.
    a.tp[0][:] = 1.
    a.tp[1][:] = 2.
    a.tp[2][:] = 3.
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2*a).toarray())
    print((a*2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print() 

    print('polar V2:')
    V = PolarDerhamSpace(derham, 'Hdiv')
    print('dimensions (parent, polar):', derham.V2.nbasis, V.dimension)
    print(V.dtype)
    print(V.zeros(), '\n')
    a = PolarVector(V)
    a.pol[0][:] = 1.
    a.pol[1][:] = 2.
    a.pol[2][:] = 3.
    a.tp[0][:] = 1.
    a.tp[1][:] = 2.
    a.tp[2][:] = 3.
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2*a).toarray())
    print((a*2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print() 

    print('polar V3:')
    V = PolarDerhamSpace(derham, 'L2')
    print('dimensions (parent, polar):', derham.V3.nbasis, V.dimension)
    print(V.dtype)
    print(V.zeros(), '\n')
    a = PolarVector(V)
    a.pol[0][:] = 1.
    a.tp[:] = 1.
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2*a).toarray())
    print((a*2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print() 

    print('polar V0vec:')
    V = PolarDerhamSpace(derham, 'H1vec')
    print('dimensions (parent, polar):', derham.V0vec.nbasis, V.dimension)
    print(V.dtype)
    print(V.zeros(), '\n')
    a = PolarVector(V)
    a.pol[0][:] = 1.
    a.pol[1][:] = 2.
    a.pol[2][:] = 3.
    a.tp[0][:] = 1.
    a.tp[1][:] = 2.
    a.tp[2][:] = 3.
    print(a.toarray())
    a.set_tp_coeffs_to_zero()
    b = a.copy()
    print(a.toarray())
    print(a.dot(b))
    print((-a).toarray())
    print((2*a).toarray())
    print((a*2).toarray())
    print((a + b).toarray())
    print((a - b).toarray())
    a *= 2
    print(a.toarray())
    a += b
    print(a.toarray())
    a -= b
    print(a.toarray())
    print(a.toarray_tp())

    print() 


@pytest.mark.parametrize('Nel', [[6, 9, 6], [8, 12, 7]])
@pytest.mark.parametrize('p', [[3, 2, 1], [4, 3, 2]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [False, True, False]])
def test_extraction_ops_and_derivatives(Nel, p, spl_kind):
    
    import numpy as np

    from struphy.geometry.domains import PoloidalSplineCylinder
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays

    from struphy.polar.extraction_operators import PolarExtractionBlocksC1
    from struphy.polar.basic import PolarDerhamSpace, PolarVector
    from struphy.polar.linear_operators import PolarExtractionOperator, PolarLinearOperator

    from struphy.feec.spline_space import Spline_space_1d, Tensor_spline_space

    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # create de Rham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # create control points
    params_map = {'Nel' : Nel[:2], 'p' : p[:2], 'spl_kind' : spl_kind[:2], 'a' : 1., 'R0' : 3.}
    domain = PoloidalSplineCylinder(params_map)

    # create legacy FEM spaces
    spaces = [Spline_space_1d(Nel, p, spl_kind) for Nel, p, spl_kind in zip(Nel, p, spl_kind)]

    for space_i in spaces:
        space_i.set_projectors()

    space = Tensor_spline_space(spaces, ck=1, cx=domain.cx[:, :, 0], cy=domain.cy[:, :, 0])
    space.set_projectors('general')

    if rank == 0:
        print()
        print('Domain decomposition : \n', derham.domain_array)
        print()

    comm.Barrier()

    # create polar FEM spaces
    V0_pol = PolarDerhamSpace(derham, 'H1')
    V1_pol = PolarDerhamSpace(derham, 'Hcurl')
    V2_pol = PolarDerhamSpace(derham, 'Hdiv')
    V3_pol = PolarDerhamSpace(derham, 'L2')

    f0_pol = PolarVector(V0_pol)
    e1_pol = PolarVector(V1_pol)
    b2_pol = PolarVector(V2_pol)
    p3_pol = PolarVector(V3_pol)

    # create pure tensor-product and polar vectors (legacy and distributed)
    f0_tp_leg, f0_tp = create_equal_random_arrays(derham.V0, flattened=True)
    e1_tp_leg, e1_tp = create_equal_random_arrays(derham.V1, flattened=True)
    b2_tp_leg, b2_tp = create_equal_random_arrays(derham.V2, flattened=True)
    p3_tp_leg, p3_tp = create_equal_random_arrays(derham.V3, flattened=True)

    f0_pol.tp = f0_tp
    e1_pol.tp = e1_tp
    b2_pol.tp = b2_tp
    p3_pol.tp = p3_tp

    np.random.seed(1607)
    f0_pol.pol = [np.random.rand(f0_pol.pol[0].shape[0], f0_pol.pol[0].shape[1])]
    e1_pol.pol = [np.random.rand(e1_pol.pol[n].shape[0], e1_pol.pol[n].shape[1]) for n in range(3)]
    b2_pol.pol = [np.random.rand(b2_pol.pol[n].shape[0], b2_pol.pol[n].shape[1]) for n in range(3)]
    p3_pol.pol = [np.random.rand(p3_pol.pol[0].shape[0], p3_pol.pol[0].shape[1])]

    f0_pol_leg = f0_pol.toarray(True)
    e1_pol_leg = e1_pol.toarray(True)
    b2_pol_leg = b2_pol.toarray(True)
    p3_pol_leg = p3_pol.toarray(True)

    # create polar extraction blocks
    c1_blocks = PolarExtractionBlocksC1(domain, derham)

    # ==================== test basis extraction operators ===================
    if rank == 0:
        print('----------- Test basis extraction operators ---------')
    
    # test basis extraction operator
    E0 = PolarExtractionOperator(derham.V0.vector_space, V0_pol, c1_blocks.e0_blocks_ten_to_pol, c1_blocks.e0_blocks_ten_to_ten)
    E1 = PolarExtractionOperator(derham.V1.vector_space, V1_pol, c1_blocks.e1_blocks_ten_to_pol, c1_blocks.e1_blocks_ten_to_ten)
    E2 = PolarExtractionOperator(derham.V2.vector_space, V2_pol, c1_blocks.e2_blocks_ten_to_pol, c1_blocks.e2_blocks_ten_to_ten)
    E3 = PolarExtractionOperator(derham.V3.vector_space, V3_pol, c1_blocks.e3_blocks_ten_to_pol, c1_blocks.e3_blocks_ten_to_ten)
    
    r0_pol = E0.dot(f0_tp)
    r1_pol = E1.dot(e1_tp)
    r2_pol = E2.dot(b2_tp)
    r3_pol = E3.dot(p3_tp)

    assert np.allclose(r0_pol.toarray(True), space.E0.dot(f0_tp_leg))
    assert np.allclose(r1_pol.toarray(True), space.E1.dot(e1_tp_leg))
    assert np.allclose(r2_pol.toarray(True), space.E2.dot(b2_tp_leg))
    assert np.allclose(r3_pol.toarray(True), space.E3.dot(p3_tp_leg))
    
    # test transposed extraction operators
    E0T = E0.transpose()
    E1T = E1.transpose()
    E2T = E2.transpose()
    E3T = E3.transpose()
    
    r0 = E0T.dot(f0_pol)
    r1 = E1T.dot(e1_pol)
    r2 = E2T.dot(b2_pol)
    r3 = E3T.dot(p3_pol)
    
    compare_arrays(r0, space.E0.T.dot(f0_pol_leg), rank)
    compare_arrays(r1, space.E1.T.dot(e1_pol_leg), rank)
    compare_arrays(r2, space.E2.T.dot(b2_pol_leg), rank)
    compare_arrays(r3, space.E3.T.dot(p3_pol_leg), rank)

    
    if rank == 0:
        print('------------- Test passed ---------------------------')
        print()
        
    # ==================== test discrete derivatives ======================
    if rank == 0:
        print('----------- Test discrete derivatives ---------')
        
    # test discrete derivatives
    G = PolarLinearOperator(V0_pol, V1_pol, derham.grad, c1_blocks.grad_blocks_pol_to_ten, c1_blocks.grad_blocks_pol_to_pol, c1_blocks.grad_blocks_e3)
    C = PolarLinearOperator(V1_pol, V2_pol, derham.curl, c1_blocks.curl_blocks_pol_to_ten, c1_blocks.curl_blocks_pol_to_pol, c1_blocks.curl_blocks_e3)
    D = PolarLinearOperator(V2_pol, V3_pol, derham.div , c1_blocks.div_blocks_pol_to_ten,  c1_blocks.div_blocks_pol_to_pol, c1_blocks.div_blocks_e3)
    
    r1_pol = G.dot(f0_pol)
    r2_pol = C.dot(e1_pol)
    r3_pol = D.dot(b2_pol)
    
    assert np.allclose(r1_pol.toarray(True), space.G.dot(f0_pol_leg))
    assert np.allclose(r2_pol.toarray(True), space.C.dot(e1_pol_leg))
    assert np.allclose(r3_pol.toarray(True), space.D.dot(b2_pol_leg))
    
    # test transposed derivatives
    GT = G.transpose()
    CT = C.transpose()
    DT = D.transpose()
    
    r0_pol = GT.dot(e1_pol)
    r1_pol = CT.dot(b2_pol)
    r2_pol = DT.dot(p3_pol)
    
    assert np.allclose(r0_pol.toarray(True), space.G.T.dot(e1_pol_leg))
    assert np.allclose(r1_pol.toarray(True), space.C.T.dot(b2_pol_leg))
    assert np.allclose(r2_pol.toarray(True), space.D.T.dot(p3_pol_leg))
    
    if rank == 0:
        print('------------- Test passed ---------------------------')
        
        
@pytest.mark.parametrize('Nel', [[8, 12, 4], [6, 15, 7]])
@pytest.mark.parametrize('p', [[2, 2, 3], [4, 3, 2]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [False, True, False]])
def test_projectors(Nel, p, spl_kind):
    
    import numpy as np

    from struphy.geometry.domains import PoloidalSplineCylinder
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays
    from struphy.psydac_api.linear_operators import InverseLinearOperator, CompositeLinearOperator
    
    from struphy.polar.extraction_operators import PolarExtractionBlocksC1
    from struphy.polar.basic import PolarDerhamSpace, PolarVector
    from struphy.polar.linear_operators import PolarExtractionOperator, PolarLinearOperator, PolarProjectionPreconditioner

    from struphy.feec.spline_space import Spline_space_1d, Tensor_spline_space
    
    from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
    from psydac.linalg.stencil import StencilVector, StencilMatrix
    from psydac.linalg.block import BlockVector, BlockMatrix

    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # create de Rham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm, nq_pr=[6, 6, 6])

    # create control points
    params_map = {'Nel' : Nel[:2], 'p' : p[:2], 'spl_kind' : spl_kind[:2], 'a' : 1., 'R0' : 3.}
    domain = PoloidalSplineCylinder(params_map)

    # create legacy FEM spaces
    spaces = [Spline_space_1d(Nel, p, spl_kind) for Nel, p, spl_kind in zip(Nel, p, spl_kind)]

    for space_i in spaces:
        space_i.set_projectors(nq=6)

    space = Tensor_spline_space(spaces, ck=1, cx=domain.cx[:, :, 0], cy=domain.cy[:, :, 0])
    space.set_projectors('general')

    if rank == 0:
        print()
        print('Domain decomposition : \n', derham.domain_array)
        print()

    comm.Barrier()

    # create polar FEM spaces
    V0_pol = PolarDerhamSpace(derham, 'H1')
    V1_pol = PolarDerhamSpace(derham, 'Hcurl')
    V2_pol = PolarDerhamSpace(derham, 'Hdiv')
    V3_pol = PolarDerhamSpace(derham, 'L2')

    f0_pol = PolarVector(V0_pol)
    e1_pol = PolarVector(V1_pol)
    b2_pol = PolarVector(V2_pol)
    p3_pol = PolarVector(V3_pol)

    # create pure tensor-product and polar vectors (legacy and distributed)
    f0_tp_leg, f0_tp = create_equal_random_arrays(derham.V0, flattened=True)
    e1_tp_leg, e1_tp = create_equal_random_arrays(derham.V1, flattened=True)
    b2_tp_leg, b2_tp = create_equal_random_arrays(derham.V2, flattened=True)
    p3_tp_leg, p3_tp = create_equal_random_arrays(derham.V3, flattened=True)

    f0_pol.tp = f0_tp
    e1_pol.tp = e1_tp
    b2_pol.tp = b2_tp
    p3_pol.tp = p3_tp

    np.random.seed(1607)
    f0_pol.pol = [np.random.rand(f0_pol.pol[0].shape[0], f0_pol.pol[0].shape[1])]
    e1_pol.pol = [np.random.rand(e1_pol.pol[n].shape[0], e1_pol.pol[n].shape[1]) for n in range(3)]
    b2_pol.pol = [np.random.rand(b2_pol.pol[n].shape[0], b2_pol.pol[n].shape[1]) for n in range(3)]
    p3_pol.pol = [np.random.rand(p3_pol.pol[0].shape[0], p3_pol.pol[0].shape[1])]

    f0_pol_leg = f0_pol.toarray(True)
    e1_pol_leg = e1_pol.toarray(True)
    b2_pol_leg = b2_pol.toarray(True)
    p3_pol_leg = p3_pol.toarray(True)

    # create polar extraction blocks
    c1_blocks = PolarExtractionBlocksC1(domain, derham)
    
    # basis extraction operators
    E0 = PolarExtractionOperator(derham.V0.vector_space, V0_pol, c1_blocks.e0_blocks_ten_to_pol, c1_blocks.e0_blocks_ten_to_ten)
    E1 = PolarExtractionOperator(derham.V1.vector_space, V1_pol, c1_blocks.e1_blocks_ten_to_pol, c1_blocks.e1_blocks_ten_to_ten)
    E2 = PolarExtractionOperator(derham.V2.vector_space, V2_pol, c1_blocks.e2_blocks_ten_to_pol, c1_blocks.e2_blocks_ten_to_ten)
    E3 = PolarExtractionOperator(derham.V3.vector_space, V3_pol, c1_blocks.e3_blocks_ten_to_pol, c1_blocks.e3_blocks_ten_to_ten)

    E0T = E0.transpose()
    E1T = E1.transpose()
    E2T = E2.transpose()
    E3T = E3.transpose()
    
    # DOF extraction operators
    P0 = PolarExtractionOperator(derham.V0.vector_space, V0_pol, c1_blocks.p0_blocks_ten_to_pol, c1_blocks.p0_blocks_ten_to_ten)
    P1 = PolarExtractionOperator(derham.V1.vector_space, V1_pol, c1_blocks.p1_blocks_ten_to_pol, c1_blocks.p1_blocks_ten_to_ten)
    P2 = PolarExtractionOperator(derham.V2.vector_space, V2_pol, c1_blocks.p2_blocks_ten_to_pol, c1_blocks.p2_blocks_ten_to_ten)
    P3 = PolarExtractionOperator(derham.V3.vector_space, V3_pol, c1_blocks.p3_blocks_ten_to_pol, c1_blocks.p3_blocks_ten_to_ten)
    
    # tensor-product inter-/histopolation matrices for projectors
    print('assemble tensor-product inter/-histopolation matrices')
    
    I0 = derham.P0.imat_stencil.tostencil()
    
    I1 = BlockMatrix(derham.V1.vector_space, derham.V1.vector_space,
                     [[derham.P1.imat_stencil[0, 0].tostencil(), None, None], 
                      [None, derham.P1.imat_stencil[1, 1].tostencil(), None], 
                      [None, None, derham.P1.imat_stencil[2, 2].tostencil()]])
    
    I2 = BlockMatrix(derham.V2.vector_space, derham.V2.vector_space, 
                     [[derham.P2.imat_stencil[0, 0].tostencil(), None, None], 
                      [None, derham.P2.imat_stencil[1, 1].tostencil(), None], 
                      [None, None, derham.P2.imat_stencil[2, 2].tostencil()]])
    
    I3 = derham.P3.imat_stencil.tostencil()
    print('done')
    print()

    I0.set_backend(PSYDAC_BACKEND_GPYCCEL)
    I1.set_backend(PSYDAC_BACKEND_GPYCCEL)
    I2.set_backend(PSYDAC_BACKEND_GPYCCEL)
    I3.set_backend(PSYDAC_BACKEND_GPYCCEL)

    # restriction I_pol = P *  I * ET to polar space/DOFs
    I0_pol = CompositeLinearOperator(P0, I0, E0T)
    I1_pol = CompositeLinearOperator(P1, I1, E1T)
    I2_pol = CompositeLinearOperator(P2, I2, E2T)
    I3_pol = CompositeLinearOperator(P3, I3, E3T)
    
    #I0_pol = CompositeLinearOperator(P0, derham.P0.imat_stencil, E0T)
    #I1_pol = CompositeLinearOperator(P1, derham.P1.imat_stencil, E1T)
    #I2_pol = CompositeLinearOperator(P2, derham.P2.imat_stencil, E2T)
    #I3_pol = CompositeLinearOperator(P3, derham.P3.imat_stencil, E3T)

    # preconditioners for projections
    PC0 = PolarProjectionPreconditioner(P0, derham.P0, E0T)
    PC1 = PolarProjectionPreconditioner(P1, derham.P1, E1T)
    PC2 = PolarProjectionPreconditioner(P2, derham.P2, E2T)
    PC3 = PolarProjectionPreconditioner(P3, derham.P3, E3T)

    # inverse polar inter-/histopolation matrices for projectors
    I0inv = InverseLinearOperator(I0_pol, pc=PC0, solver_name='pbicgstab', tol=1e-14)
    I1inv = InverseLinearOperator(I1_pol, pc=PC1, solver_name='pbicgstab', tol=1e-14)
    I2inv = InverseLinearOperator(I2_pol, pc=PC2, solver_name='pbicgstab', tol=1e-14)
    I3inv = InverseLinearOperator(I3_pol, pc=PC3, solver_name='pbicgstab', tol=1e-14)
    
    # function to project
    fun = lambda e1, e2, e3 : (1 - e1)**2*np.sin(2*np.pi*e1)*np.sin(e1*np.sin(4*np.pi*e2))*np.cos(2*np.pi*e3)
    
    print('start computation of DOFs')
    
    # legacy projections
    dofs0_pol_leg = space.projectors.dofs_0(fun)
    dofs1_pol_leg = space.projectors.dofs_1([fun, fun, fun], with_subs=False)
    dofs2_pol_leg = space.projectors.dofs_2([fun, fun, fun], with_subs=False)
    dofs3_pol_leg = space.projectors.dofs_3(fun, with_subs=False)

    r0_pol_leg = space.projectors.solve_V0(dofs0_pol_leg, include_bc=True)
    r1_pol_leg = space.projectors.solve_V1(dofs1_pol_leg, include_bc=True)
    r2_pol_leg = space.projectors.solve_V2(dofs2_pol_leg, include_bc=True)
    r3_pol_leg = space.projectors.solve_V3(dofs3_pol_leg, include_bc=True)
    
    # new polar DOFs
    dofs0_pol = P0.dot(derham.P0(fun, dofs_only=True))
    dofs1_pol = P1.dot(derham.P1([fun, fun, fun], dofs_only=True))
    dofs2_pol = P2.dot(derham.P2([fun, fun, fun], dofs_only=True))
    dofs3_pol = P3.dot(derham.P3(fun, dofs_only=True))
    
    
    # ============ project to V0 =========================
    if rank == 0:
        r0_pol = I0inv.dot(dofs0_pol, verbose=True)
    else:
        r0_pol = I0inv.dot(dofs0_pol, verbose=False)
    
    if rank == 0: 
        print('Test passed for PI_0 polar projector')
        print()
        
    # ============ project to V1 =========================
    if rank == 0:
        r1_pol = I1inv.dot(dofs1_pol, verbose=True)
    else:
        r1_pol = I1inv.dot(dofs1_pol, verbose=False)
        
    assert np.allclose(r1_pol.toarray(True), r1_pol_leg)
    
    if rank == 0: 
        print('Test passed for PI_1 polar projector')
        print()
        
    # ============ project to V2 =========================
    if rank == 0:
        r2_pol = I2inv.dot(dofs2_pol, verbose=True)
    else:
        r2_pol = I2inv.dot(dofs2_pol, verbose=False)
        
    assert np.allclose(r2_pol.toarray(True), r2_pol_leg)
    
    if rank == 0: 
        print('Test passed for PI_2 polar projector')
        print()
        
    # ============ project to V3 =========================
    if rank == 0:
        r3_pol = I3inv.dot(dofs3_pol, verbose=True)
    else:
        r3_pol = I3inv.dot(dofs3_pol, verbose=False)
        
    assert np.allclose(r3_pol.toarray(True), r3_pol_leg)
    
    if rank == 0: 
        print('Test passed for PI_3 polar projector')
        print()
    
    
    
if __name__ == '__main__':
    #test_spaces([6, 9, 4], [2, 2, 2], [False, True, False])
    #test_extraction_ops_and_derivatives([8, 12, 6], [2, 2, 3], [False, True, False])
    test_projectors([8, 15, 4], [2, 2, 3], [False, True, True])