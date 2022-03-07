def test_psydac_mapping(map=None, params_map=None):

    from struphy.geometry.domain_3d import Domain

    import numpy as np

    # Mappings to be tested
    if map==None:
        maps = [
            'cuboid',
            'orthogonal',
            'colella',
            'hollow_cyl',
            'hollow_torus',
            'ellipse',
            'rotated_ellipse',
            'soloviev_approx',
            'soloviev_sqrt',
            'soloviev_cf',
        ]
    else:
        maps = [map]

    # Mapping parameters to be tested
    if params_map==None:
        param_sets = [
            {'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.},
            {'Lx': 1., 'Ly': 2., 'alpha': .5, 'Lz': 3.},
            {'Lx': 1., 'Ly': 2., 'alpha': .5, 'Lz': 3.},
            {'a1': 1., 'a2': 2., 'R0': 3.},
            {'a1': 1., 'a2': 2., 'R0': 3.},
            {'x0': 1., 'y0': 2., 'z0': 3., 'rx': 4., 'ry': 5., 'Lz': 6.},
            {'x0': 1., 'y0': 2., 'z0': 3., 'r1': 4., 'r2': 5., 'Lz': 6., 'th': 7.},
            {'x0': 1., 'y0': 2., 'z0': 3., 'rx': 4., 'ry': 5., 'Lz': 6., 'delta': 7.},
            {'x0': 1., 'y0': 2., 'z0': 3., 'rx': 4., 'ry': 5., 'Lz': 6., 'delta': 7.},
            {'x0': 1., 'y0': 2., 'z0': 3., 'R0': 4., 'Lz': 5., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.,}
        ]
    else:
        param_sets = [params_map]

    ########################
    ### TEST EVALUATIONS ###
    ########################
    for map, params in zip(maps, param_sets):

        print(map)
        print(params)

        # Struphy domain object
        DOMAIN         = Domain(map, params)
        Mapping_psydac = DOMAIN.Psydac_mapping('F', **params) 

        print(Mapping_psydac._expressions, '\n')

        # Psydac mapping
        F_PSY = Mapping_psydac.get_callable_mapping()

        # Comparisons at random logical point
        eta = np.random.rand(3)

        # Mapping
        assert np.allclose(F_PSY(*eta)[0], DOMAIN.evaluate(*eta, 'x'))
        assert np.allclose(F_PSY(*eta)[1], DOMAIN.evaluate(*eta, 'y'))
        assert np.allclose(F_PSY(*eta)[2], DOMAIN.evaluate(*eta, 'z'))

        # Absolute value of Jacobian determinant
        assert np.allclose(np.sqrt(F_PSY.metric_det(*eta)), np.abs(DOMAIN.evaluate(*eta, 'det_df')))

        # Jacobian
        assert np.allclose(F_PSY.jacobian(*eta)[0, 0], DOMAIN.evaluate(*eta, 'df_11'))
        assert np.allclose(F_PSY.jacobian(*eta)[0, 1], DOMAIN.evaluate(*eta, 'df_12'))
        assert np.allclose(F_PSY.jacobian(*eta)[0, 2], DOMAIN.evaluate(*eta, 'df_13'))
        assert np.allclose(F_PSY.jacobian(*eta)[1, 0], DOMAIN.evaluate(*eta, 'df_21'))
        assert np.allclose(F_PSY.jacobian(*eta)[1, 1], DOMAIN.evaluate(*eta, 'df_22'))
        assert np.allclose(F_PSY.jacobian(*eta)[1, 2], DOMAIN.evaluate(*eta, 'df_23'))
        assert np.allclose(F_PSY.jacobian(*eta)[2, 0], DOMAIN.evaluate(*eta, 'df_31'))
        assert np.allclose(F_PSY.jacobian(*eta)[2, 1], DOMAIN.evaluate(*eta, 'df_32'))
        assert np.allclose(F_PSY.jacobian(*eta)[2, 2], DOMAIN.evaluate(*eta, 'df_33'))

        # Inverse Jacobian
        assert np.allclose(F_PSY.jacobian_inv(*eta)[0, 0], DOMAIN.evaluate(*eta, 'df_inv_11'))
        assert np.allclose(F_PSY.jacobian_inv(*eta)[0, 1], DOMAIN.evaluate(*eta, 'df_inv_12'))
        assert np.allclose(F_PSY.jacobian_inv(*eta)[0, 2], DOMAIN.evaluate(*eta, 'df_inv_13'))
        assert np.allclose(F_PSY.jacobian_inv(*eta)[1, 0], DOMAIN.evaluate(*eta, 'df_inv_21'))
        assert np.allclose(F_PSY.jacobian_inv(*eta)[1, 1], DOMAIN.evaluate(*eta, 'df_inv_22'))
        assert np.allclose(F_PSY.jacobian_inv(*eta)[1, 2], DOMAIN.evaluate(*eta, 'df_inv_23'))
        assert np.allclose(F_PSY.jacobian_inv(*eta)[2, 0], DOMAIN.evaluate(*eta, 'df_inv_31'))
        assert np.allclose(F_PSY.jacobian_inv(*eta)[2, 1], DOMAIN.evaluate(*eta, 'df_inv_32'))
        assert np.allclose(F_PSY.jacobian_inv(*eta)[2, 2], DOMAIN.evaluate(*eta, 'df_inv_33'))

        # Metric tensor
        assert np.allclose(F_PSY.metric(*eta)[0, 0], DOMAIN.evaluate(*eta, 'g_11'))
        assert np.allclose(F_PSY.metric(*eta)[0, 1], DOMAIN.evaluate(*eta, 'g_12'))
        assert np.allclose(F_PSY.metric(*eta)[0, 2], DOMAIN.evaluate(*eta, 'g_13'))
        assert np.allclose(F_PSY.metric(*eta)[1, 0], DOMAIN.evaluate(*eta, 'g_21'))
        assert np.allclose(F_PSY.metric(*eta)[1, 1], DOMAIN.evaluate(*eta, 'g_22'))
        assert np.allclose(F_PSY.metric(*eta)[1, 2], DOMAIN.evaluate(*eta, 'g_23'))
        assert np.allclose(F_PSY.metric(*eta)[2, 0], DOMAIN.evaluate(*eta, 'g_31'))
        assert np.allclose(F_PSY.metric(*eta)[2, 1], DOMAIN.evaluate(*eta, 'g_32'))
        assert np.allclose(F_PSY.metric(*eta)[2, 2], DOMAIN.evaluate(*eta, 'g_33'))

        # Inverse metric tensor
        metric_inv_PSY = np.matmul(F_PSY.jacobian_inv(*eta), F_PSY.jacobian_inv(*eta).T) # missing in psydac
        assert np.allclose(metric_inv_PSY[0, 0], DOMAIN.evaluate(*eta, 'g_inv_11'))
        assert np.allclose(metric_inv_PSY[0, 1], DOMAIN.evaluate(*eta, 'g_inv_12'))
        assert np.allclose(metric_inv_PSY[0, 2], DOMAIN.evaluate(*eta, 'g_inv_13'))
        assert np.allclose(metric_inv_PSY[1, 0], DOMAIN.evaluate(*eta, 'g_inv_21'))
        assert np.allclose(metric_inv_PSY[1, 1], DOMAIN.evaluate(*eta, 'g_inv_22'))
        assert np.allclose(metric_inv_PSY[1, 2], DOMAIN.evaluate(*eta, 'g_inv_23'))
        assert np.allclose(metric_inv_PSY[2, 0], DOMAIN.evaluate(*eta, 'g_inv_31'))
        assert np.allclose(metric_inv_PSY[2, 1], DOMAIN.evaluate(*eta, 'g_inv_32'))
        assert np.allclose(metric_inv_PSY[2, 2], DOMAIN.evaluate(*eta, 'g_inv_33'))

        print(map + ' done.\n')


def test_psydac_FemSpace():
    '''Requires pyccel==1.4.0, install manually (pyccel=0.10.1 installed by struphy).'''

    from struphy.geometry.domain_3d import Domain
    from struphy.feec.spline_space import Spline_space_1d, Tensor_spline_space 

    from sympde.topology import Cube, Derham

    from psydac.api.discretization import discretize
    from psydac.fem.tensor import TensorFemSpace 
    from psydac.fem.vector import ProductFemSpace
    from psydac.fem.basic import FemField
    from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
    from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockMatrix

    from mpi4py import MPI
    import numpy as np

    # Domain object
    map = 'cuboid'
    params_map = {'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}

    DOMAIN = Domain(map, params_map)

    # Psydac mapping
    Mapping_psydac = DOMAIN.Psydac_mapping('F', **params_map)
    
    # Psydac symbolic domain
    DOMAIN_PSYDAC_LOGICAL = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
    DOMAIN_symb = Mapping_psydac(DOMAIN_PSYDAC_LOGICAL)

    # Psydac symbolic Derham
    DERHAM_symb = Derham(DOMAIN_symb)

    # grid parameters
    Nel      = [8, 9, 10]
    p        = [3, 3, 3]
    spl_kind = [True, True, True] 
    nq_el    = [4, 4, 4]
    n_quad   = [4, 4, 4]

    # Struphy Derham (only for return statement)
    spaces = [Spline_space_1d(Nel_i, p_i, spl_kind_i, nq_el_i) for Nel_i, p_i, spl_kind_i, nq_el_i in zip(Nel, p, spl_kind, nq_el)] 

    spaces[0].set_projectors(n_quad[0]) 
    spaces[1].set_projectors(n_quad[1])
    spaces[2].set_projectors(n_quad[2])

    DERHAM_STR = Tensor_spline_space(spaces)
    DERHAM_STR.set_projectors('tensor')

    # Psydac discrete De Rham
    DOMAIN_PSY  = discretize(DOMAIN_symb, ncells=Nel, comm=MPI.COMM_WORLD) # The parallelism is initiated here.
    DERHAM_PSY  = discretize(DERHAM_symb, DOMAIN_PSY, degree=p, periodic=spl_kind)

    # Spline spaces
    V0 = DERHAM_PSY.V0
    V1 = DERHAM_PSY.V1
    V2 = DERHAM_PSY.V2
    V3 = DERHAM_PSY.V3

    print('\nV0.nbasis:', V0.nbasis)

    assert isinstance(V0, TensorFemSpace)
    assert isinstance(V1, ProductFemSpace)
    assert isinstance(V2, ProductFemSpace)
    assert isinstance(V3, TensorFemSpace)

    assert isinstance(V0.vector_space, StencilVectorSpace)
    assert isinstance(V1.vector_space, BlockVectorSpace)
    assert isinstance(V2.vector_space, BlockVectorSpace)
    assert isinstance(V3.vector_space, StencilVectorSpace)

    ################################
    ### FEM fields (distributed) ###
    ################################
    u0_fem = FemField(V0)
    print('\nV0 FemField:')
    print(u0_fem)
    print('\n.space:')
    print(u0_fem.space)
    print('\ntype(coeffs):')
    print(type(u0_fem.coeffs))
    print('\n.coeffs.shape:')
    print(u0_fem.coeffs.shape)
    print('\ntype(coeffs[:]):')
    print(type(u0_fem.coeffs[:]))
    print('\n.coeffs[:].shape:')
    print(u0_fem.coeffs[:].shape)
    print('\n.fields:')
    print(u0_fem.fields)

    # FemField can be avaluated
    assert u0_fem(0, 0, 0) == 0.

    #####################################
    ### Stencil objects (distributed) ###
    #####################################
    # Stencil vector
    u0 = StencilVector(V0.vector_space)
    print('\nStencilVector:')
    print(u0)
    print('.space:')
    print(u0.space)
    print('\n.shape:')
    print(u0.shape)
    print('\n[:].shape:')
    print(u0[:].shape)
    print('\n.toarray.shape:')
    print(u0.shape)

    # StencilVector can be assigned
    assert u0[0, 0, 0] == 0.

    # Overwrite id with new stencil vector
    v0 = StencilVector(V0.vector_space)
    old_id = id(u0)
    u0[:] = v0[:]
    assert id(u0) == old_id

    # Stencil matrix
    A00 = StencilMatrix(V0.vector_space, V0.vector_space)
    A03 = StencilMatrix(V0.vector_space, V3.vector_space)
    print('\nStencilMatrix:')
    print(A00)
    print('\n.domain:')
    print(A00.domain)
    print('\n.codomain:')
    print(A00.codomain)
    print('\n.shape:')
    print(A00.shape)
    print('\n[:, :].shape:')
    print(A00[:, :].shape)
    print('\n.toarray.shape:')
    print(A00.toarray().shape)
    print('\n._data.shape:')
    print(A00._data.shape)

    assert A00[0, 0, 0, 0, 0, 0] == 0.

    # Product spaces
    # Vector
    u1 = BlockVector(V1.vector_space)
    assert isinstance(u1[0], StencilVector)
    assert isinstance(u1[1], StencilVector)
    assert isinstance(u1[2], StencilVector)

    print('\nBlockVector: u1[0].shape =', u1[0].shape, 'u1[0][:].shape =', u1[0][:].shape)
    print('\nBlockVector: u1[1].shape =', u1[1].shape, 'u1[1][:].shape =', u1[1][:].shape)
    print('\nBlockVector: u1[2].shape =', u1[2].shape, 'u1[2][:].shape =', u1[2][:].shape)

    assert u1[0][0, 0, 0] == 0.

    # Matrix
    Bxx = StencilMatrix(V1.spaces[0].vector_space, V1.spaces[0].vector_space) # maps from 0 -> 0, thus first column x
    Byx = StencilMatrix(V1.spaces[0].vector_space, V1.spaces[1].vector_space) # maps from 0 -> 1, thus first column x
    Bzx = StencilMatrix(V1.spaces[0].vector_space, V1.spaces[2].vector_space) # maps from 0 -> 2, thus first column x

    Bxy = StencilMatrix(V1.spaces[1].vector_space, V1.spaces[0].vector_space)
    Byy = StencilMatrix(V1.spaces[1].vector_space, V1.spaces[1].vector_space)
    Bzy = StencilMatrix(V1.spaces[1].vector_space, V1.spaces[2].vector_space)

    Bxz = StencilMatrix(V1.spaces[2].vector_space, V1.spaces[0].vector_space)
    Byz = StencilMatrix(V1.spaces[2].vector_space, V1.spaces[1].vector_space)
    Bzz = StencilMatrix(V1.spaces[2].vector_space, V1.spaces[2].vector_space)

    blocks = [[Bxx, Bxy, Bxz], [Byx, Byy, Byz], [Bzx, Bzy, Bzz]]

    A11 = BlockMatrix(V1.vector_space, V1.vector_space, blocks)

    print('BlockMatrix: A11[0, 0].shape =', A11[0, 0].shape, 'A11[0, 0][:, :].shape =', A11[0, 0][:, :].shape)
    print('BlockMatrix: A11[0, 1].shape =', A11[0, 1].shape, 'A11[0, 1][:, :].shape =', A11[0, 1][:, :].shape)
    print('BlockMatrix: A11[0, 2].shape =', A11[0, 2].shape, 'A11[0, 2][:, :].shape =', A11[0, 2][:, :].shape)

    assert A11[0, 0][0, 0, 0, 0, 0, 0] == 0.
    
    #########################
    ### Access dimensions ###
    #########################
    print('\nPsydac dimensions:')
    N0_tot = V0.nbasis
    N1_tot = [space.nbasis for space in V1.spaces]
    N2_tot = [space.nbasis for space in V2.spaces]
    N3_tot = V3.nbasis
    N0     = [space.nbasis for space in V0.spaces]
    N1     = [[direction.nbasis for direction in space.spaces] for space in V1.spaces]
    N2     = [[direction.nbasis for direction in space.spaces] for space in V2.spaces]
    N3     = [space.nbasis for space in V3.spaces]
    print('V0.nbasis:', N0_tot)
    print('V0.spaces.nbasis:',    N0)
    print('V1.spaces.nbasis:', N1_tot)
    print('V1.spaces.spaces.nbasis:',    N1)
    print('V2.spaces.nbasis:', N2_tot)
    print('V2.spaces.spaces.nbasis:',    N2)
    print('V3.nbasis:', N3_tot)
    print('V3.spaces.nbasis:',    N3, '\n')

    return DERHAM_STR, DERHAM_PSY, DERHAM_symb, DOMAIN_symb, DOMAIN, DOMAIN_PSY


def test_psydac_derham():

    from psydac.linalg.stencil import StencilVector
    from psydac.linalg.block   import BlockVector
    from psydac.api.discretization import discretize
    from psydac.api.settings       import PSYDAC_BACKEND_PYTHON, PSYDAC_BACKEND_GPYCCEL

    from sympde.topology import elements_of
    from sympde.expr     import BilinearForm, integral
    from sympde.calculus import dot
    
    import numpy as np

    # Derham objects
    DERHAM_STR, DERHAM_PSY, DERHAM_symb, DOMAIN_symb, DOMAIN, DOMAIN_PSY = test_psydac_FemSpace()

    # Spline spaces
    V0 = DERHAM_PSY.V0
    V1 = DERHAM_PSY.V1
    V2 = DERHAM_PSY.V2
    V3 = DERHAM_PSY.V3

    # Space dimensions
    N0_tot = DERHAM_STR.Ntot_0form
    N1_tot = DERHAM_STR.Ntot_1form
    N2_tot = DERHAM_STR.Ntot_2form
    N3_tot = DERHAM_STR.Ntot_3form

    # Random vectors for testing
    x0 = np.random.rand(N0_tot)
    x1 = np.random.rand(np.sum(N1_tot))
    x2 = np.random.rand(np.sum(N2_tot))
    x3 = np.random.rand(N3_tot)

    ############################
    ### TEST STENCIL VECTORS ###
    ############################
    # Stencil vectors for Psydac:
    x0_PSY = StencilVector(V0.vector_space)
    print('0-form StencilVector:')
    print('starts:', x0_PSY.starts)
    print('ends  :', x0_PSY.ends)
    print('pads  :', x0_PSY.pads)
    print('shape (=dim):', x0_PSY.shape)
    print('[:].shape (=shape):', x0_PSY[:].shape)

    s1_V0, s2_V0, s3_V0 = x0_PSY.starts
    e1_V0, e2_V0, e3_V0 = x0_PSY.ends

    # Assign from start to end index + 1
    x0_PSY[s1_V0:e1_V0 + 1, s2_V0:e2_V0 + 1, s3_V0:e3_V0 + 1] = DERHAM_STR.extract_0(x0) 

    # toarray() eliminates the padding
    assert np.all(x0 == x0_PSY.toarray())

    # Block of StencilVecttors
    x1_PSY = BlockVector(V1.vector_space)
    print('\n1-form StencilVector:')
    print('starts:', [component.starts for component in x1_PSY])
    print('ends  :', [component.ends for component in x1_PSY])
    print('pads  :', [component.pads for component in x1_PSY])
    print('shape (=dim):', [component.shape for component in x1_PSY])
    print('[:].shape (=shape):', [component[:].shape for component in x1_PSY])

    s_V11,  s_V12,  s_V13 = [component.starts for component in x1_PSY]
    s1_V11, s2_V11, s3_V11 = s_V11
    s1_V12, s2_V12, s3_V12 = s_V12
    s1_V13, s2_V13, s3_V13 = s_V13

    e_V11,  e_V12,  e_V13 = [component.ends for component in x1_PSY]
    e1_V11, e2_V11, e3_V11 = e_V11
    e1_V12, e2_V12, e3_V12 = e_V12
    e1_V13, e2_V13, e3_V13 = e_V13

    x11, x12, x13 = DERHAM_STR.extract_1(x1)
    x1_PSY[0][s1_V11:e1_V11 + 1, s2_V11:e2_V11 + 1, s3_V11:e3_V11 + 1] = x11
    x1_PSY[1][s1_V12:e1_V12 + 1, s2_V12:e2_V12 + 1, s3_V12:e3_V12 + 1] = x12
    x1_PSY[2][s1_V13:e1_V13 + 1, s2_V13:e2_V13 + 1, s3_V13:e3_V13 + 1] = x13

    # toarray() is flattened and concatenated
    assert np.all(x1 == x1_PSY.toarray())

    x2_PSY = BlockVector(V2.vector_space)
    print('\n2-form StencilVector:')
    print('starts:', [component.starts for component in x2_PSY])
    print('ends  :', [component.ends for component in x2_PSY])
    print('pads  :', [component.pads for component in x2_PSY])
    print('shape (=dim):', [component.shape for component in x2_PSY])
    print('[:].shape (=shape):', [component[:].shape for component in x2_PSY])

    s_V21,  s_V22,  s_V23 = [component.starts for component in x2_PSY]
    s1_V21, s2_V21, s3_V21 = s_V21
    s1_V22, s2_V22, s3_V22 = s_V22
    s1_V23, s2_V23, s3_V23 = s_V23

    e_V21,  e_V22,  e_V23 = [component.ends for component in x2_PSY]
    e1_V21, e2_V21, e3_V21 = e_V21
    e1_V22, e2_V22, e3_V22 = e_V22
    e1_V23, e2_V23, e3_V23 = e_V23

    x21, x22, x23 = DERHAM_STR.extract_2(x2)
    x2_PSY[0][s1_V21:e1_V21 + 1, s2_V21:e2_V21 + 1, s3_V21:e3_V21 + 1] = x21
    x2_PSY[1][s1_V22:e1_V22 + 1, s2_V22:e2_V22 + 1, s3_V22:e3_V22 + 1] = x22
    x2_PSY[2][s1_V23:e1_V23 + 1, s2_V23:e2_V23 + 1, s3_V23:e3_V23 + 1] = x23

    assert np.all(x2 == x2_PSY.toarray())

    x3_PSY = StencilVector(V3.vector_space)
    print('\n3-form StencilVector:')
    print('starts:', x3_PSY.starts)
    print('ends  :', x3_PSY.ends)
    print('pads  :', x3_PSY.pads)
    print('shape (=dim):', x3_PSY.shape)
    print('[:].shape (=shape):', x3_PSY[:].shape)

    s1_V3, s2_V3, s3_V3 = x3_PSY.starts
    e1_V3, e2_V3, e3_V3 = x3_PSY.ends

    x3_PSY[s1_V3:e1_V3 + 1, s2_V3:e2_V3 + 1, s3_V3:e3_V3 + 1] = DERHAM_STR.extract_3(x3)

    assert np.all(x3 == x3_PSY.toarray())

    ########################
    ### TEST DERIVATIVES ###
    ########################
    # Struphy derivative operators
    grad = DERHAM_STR.G0
    curl = DERHAM_STR.C0
    div  = DERHAM_STR.D0
    print('')
    print('Struphy derivatives operators type:')
    print(type(grad), type(curl), type(div))

    # Psydac derivative operators
    grad_PSY, curl_PSY, div_PSY = DERHAM_PSY.derivatives_as_matrices
    print('Psydac derivatives operators type:')
    print(type(grad_PSY), type(curl_PSY), type(div_PSY))

    # compare derivatives
    d1_STR =  grad.dot(x0)
    d1_PSY =  grad_PSY.dot(x0_PSY)

    zero1_STR = curl.dot(d1_STR)
    zero1_PSY = curl_PSY.dot(d1_PSY)

    assert np.all(d1_STR == d1_PSY.toarray())
    assert np.allclose(zero1_STR, np.zeros_like(zero1_STR))
    assert np.allclose(zero1_PSY.toarray(), np.zeros_like(zero1_STR))

    d2_STR =  curl.dot(x1)
    d2_PSY =  curl_PSY.dot(x1_PSY)

    zero2_STR = div.dot(d2_STR)
    zero2_PSY = div_PSY.dot(d2_PSY)

    assert np.allclose(d2_STR, d2_PSY.toarray())
    assert np.allclose(zero2_STR, np.zeros_like(zero2_STR))
    assert np.allclose(zero2_PSY.toarray(), np.zeros_like(zero2_STR))

    d3_STR =  div.dot(x2)
    d3_PSY =  div_PSY.dot(x2_PSY)

    assert np.allclose(d3_STR, d3_PSY.toarray())

    ##########################
    ### TEST MASS MATRICES ###
    ##########################
    # Struphy mass matrices
    DERHAM_STR.assemble_M0(DOMAIN)
    DERHAM_STR.assemble_M1(DOMAIN)
    DERHAM_STR.assemble_M2(DOMAIN)
    DERHAM_STR.assemble_M3(DOMAIN)

    print('\nStruphy mass matrices type:')
    print(type(DERHAM_STR.M0), type(DERHAM_STR.M0_mat))

    # Psydac mass matrices
    u0, v0 = elements_of(DERHAM_symb.V0, names='u0, v0')
    u1, v1 = elements_of(DERHAM_symb.V1, names='u1, v1')
    u2, v2 = elements_of(DERHAM_symb.V2, names='u2, v2')
    u3, v3 = elements_of(DERHAM_symb.V3, names='u3, v3')

    a0 = BilinearForm((u0, v0), integral(DOMAIN_symb, u0*v0))
    a1 = BilinearForm((u1, v1), integral(DOMAIN_symb, dot(u1, v1)))
    a2 = BilinearForm((u2, v2), integral(DOMAIN_symb, dot(u2, v2)))
    a3 = BilinearForm((u3, v3), integral(DOMAIN_symb, u3*v3))

    a0_h = discretize(a0, DOMAIN_PSY, (V0, V0), backend=PSYDAC_BACKEND_GPYCCEL)
    a1_h = discretize(a1, DOMAIN_PSY, (V1, V1), backend=PSYDAC_BACKEND_GPYCCEL)
    a2_h = discretize(a2, DOMAIN_PSY, (V2, V2), backend=PSYDAC_BACKEND_GPYCCEL)
    a3_h = discretize(a3, DOMAIN_PSY, (V3, V3), backend=PSYDAC_BACKEND_GPYCCEL)

    M0_PSY = a0_h.assemble()
    M1_PSY = a1_h.assemble()
    M2_PSY = a2_h.assemble()
    M3_PSY = a3_h.assemble()

    print('Psydac mass matrices type:')
    print(type(M0_PSY), '\n')

    # compare mass matrices
    prod0     = DERHAM_STR.M0(x0)
    prod0_STR = DERHAM_STR.M0_mat.dot(x0)
    prod0_PSY = M0_PSY.dot(x0_PSY)

    print('rel 0 error STR-STR:', np.max( np.abs( prod0_STR - prod0 ) ) / np.max(np.abs(prod0_STR)) )
    print('rel 0 error STR-PSY:', np.max( np.abs( prod0_STR - prod0_PSY.toarray() ) ) / np.max(np.abs(prod0_STR)) )

    assert np.allclose(prod0_STR, prod0)
    assert np.allclose(prod0_STR, prod0_PSY.toarray())

    prod1     = DERHAM_STR.M1(x1)
    prod1_STR = DERHAM_STR.M1_mat.dot(x1)
    prod1_PSY = M1_PSY.dot(x1_PSY)

    print('rel 1 error STR-STR:', np.max( np.abs( prod1_STR - prod1 ) ) / np.max(np.abs(prod1_STR)) )
    print('rel 1 error STR-PSY:', np.max( np.abs( prod1_STR - prod1_PSY.toarray() ) ) / np.max(np.abs(prod1_STR)) )

    assert np.allclose(prod1_STR, prod1)
    assert np.allclose(prod1_STR, prod1_PSY.toarray())

    prod2     = DERHAM_STR.M2(x2)
    prod2_STR = DERHAM_STR.M2_mat.dot(x2)
    prod2_PSY = M2_PSY.dot(x2_PSY)

    print('rel 2 error STR-STR:', np.max( np.abs( prod2_STR - prod2 ) ) / np.max(np.abs(prod2_STR)) )
    print('rel 2 error STR-PSY:', np.max( np.abs( prod2_STR - prod2_PSY.toarray() ) ) / np.max(np.abs(prod2_STR)) )

    assert np.allclose(prod2_STR, prod2)
    assert np.allclose(prod2_STR, prod2_PSY.toarray())

    prod3     = DERHAM_STR.M3(x3)
    prod3_STR = DERHAM_STR.M3_mat.dot(x3)
    prod3_PSY = M3_PSY.dot(x3_PSY)

    print('rel 3 error STR-STR:', np.max( np.abs( prod3_STR - prod3 ) ) / np.max(np.abs(prod3_STR)) )
    print('rel 3 error STR-PSY:', np.max( np.abs( prod3_STR - prod3_PSY.toarray() ) ) / np.max(np.abs(prod3_STR)) )

    assert np.allclose(prod3_STR, prod3)
    assert np.allclose(prod3_STR, prod3_PSY.toarray())

    #######################
    ### TEST PROJECTORS ###
    #######################
    # Struphy projectors
    DERHAM_STR.set_projectors()
    PI     = DERHAM_STR.projectors.PI     # callable as input
    PI_mat = DERHAM_STR.projectors.PI_mat # dofs as input (as 3d array)
    print('\nStruphy projectors type:')
    print(type(PI), type(PI_mat))

    # Psydac projectors
    P0_PSY, P1_PSY, P2_PSY, P3_PSY  = DERHAM_PSY.projectors(nquads=DERHAM_STR.n_quad)
    print('Psydac projectors type:')
    print(type(P0_PSY))
    print(type(P1_PSY))
    print(type(P2_PSY))
    print(type(P3_PSY))

    # compare projectors
    f = lambda eta1, eta2, eta3 : np.sin(4*np.pi*eta1)*np.cos(2*np.pi*eta2) + np.exp(np.cos(2*np.pi*eta3))

    fh0_STR = PI('0', f)
    fh0_PSY = P0_PSY(f)
    print('\nfh0 shapes:')
    print('Struphy:', fh0_STR.shape, ', flattened:', fh0_STR.flatten().shape)
    print('Psydac:', fh0_PSY.coeffs.toarray().shape)

    print(fh0_STR.flatten()[:10])
    print(fh0_PSY.coeffs.toarray()[:10])

    assert np.allclose(fh0_STR.flatten(), fh0_PSY.coeffs.toarray())

    fh11_STR = PI('11', f)
    fh12_STR = PI('12', f)
    fh13_STR = PI('13', f)
    fh1_STR  = np.concatenate((fh11_STR.flatten(), fh12_STR.flatten(), fh13_STR.flatten()))
    fh1_PSY  = P1_PSY((f, f, f))

    print('\nfh1 shapes:')
    print('Struphy 1:', fh11_STR.shape, ', flattened:', fh11_STR.flatten().shape)
    print('Struphy 2:', fh12_STR.shape, ', flattened:', fh12_STR.flatten().shape)
    print('Struphy 3:', fh13_STR.shape, ', flattened:', fh13_STR.flatten().shape)
    print('Struphy:', fh1_STR.shape)
    print('Psydac:', fh1_PSY.coeffs.toarray().shape)

    assert np.allclose(fh1_STR, fh1_PSY.coeffs.toarray())

    fh21_STR = PI('21', f)
    fh22_STR = PI('22', f)
    fh23_STR = PI('23', f)
    fh2_STR  = np.concatenate((fh21_STR.flatten(), fh22_STR.flatten(), fh23_STR.flatten()))
    fh2_PSY  = P2_PSY((f, f, f))

    print('\nfh2 shapes:')
    print('Struphy 1:', fh21_STR.shape, ', flattened:', fh21_STR.flatten().shape)
    print('Struphy 2:', fh22_STR.shape, ', flattened:', fh22_STR.flatten().shape)
    print('Struphy 3:', fh23_STR.shape, ', flattened:', fh23_STR.flatten().shape)
    print('Struphy:', fh2_STR.shape)
    print('Psydac:', fh2_PSY.coeffs.toarray().shape)

    assert np.allclose(fh2_STR, fh2_PSY.coeffs.toarray())

    fh3_STR = PI('3', f)
    fh3_PSY = P3_PSY(f)
    print('\nfh3 shapes:')
    print('Struphy:', fh3_STR.shape, ', flattened:', fh3_STR.flatten().shape)
    print('Psydac:', fh3_PSY.coeffs.toarray().shape)

    assert np.allclose(fh3_STR.flatten(), fh3_PSY.coeffs.toarray())
    


if __name__ == '__main__':
    test_psydac_mapping()
    test_psydac_FemSpace()
    test_psydac_derham()