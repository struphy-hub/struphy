def test_lin_mhd_ops():
    '''Tests the MHD specific projection operators PI_ijk(fun*Lambda_mno).
    
    Here, PI_ijk is the commuting projector of the output space (codomain), 
    Lambda_mno are the basis functions of the input space (domain), 
    and fun is an arbitrary (matrix-valued) function.
    '''

    from struphy.psydac_linear_operators.mhd_ops import MHD_operator
    from struphy.feec.projectors.pro_global.mhd_operators_MF_for_tests import projectors_dot_x

    from struphy.geometry.domain_3d            import Domain
    from struphy.feec.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.mhd_equil.mhd_equil_physical  import Equilibrium_mhd_physical
    from struphy.mhd_equil.mhd_equil_logical   import Equilibrium_mhd_logical 

    from sympde.topology import Cube, Derham

    from psydac.api.discretization import discretize
    from psydac.linalg.stencil import StencilVector
    from psydac.linalg.block import BlockVector
    from psydac.fem.basic import FemField

    from mpi4py import MPI      
    import time
    import numpy as np

    # mpi communicator
    MPI_COMM = MPI.COMM_WORLD
    mpi_rank = MPI_COMM.Get_rank()
    MPI_COMM.Barrier()

    # Domain object
    map = 'cuboid'
    params_map = {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}

    DOMAIN = Domain(map, params_map)

    # Psydac mapping
    Mapping_psydac = DOMAIN.Psydac_mapping('F', **params_map)
    F = Mapping_psydac.get_callable_mapping()

    # Missing in Psydac: inverse metric tensor
    Ginv = lambda x1, x2, x3 : np.matmul(F.jacobian_inv(x1, x2, x3), F.jacobian_inv(x1, x2, x3).T)

    # Psydac symbolic domain
    DOMAIN_PSYDAC_LOGICAL = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
    DOMAIN_symb = Mapping_psydac(DOMAIN_PSYDAC_LOGICAL)

    # Psydac symbolic Derham
    DERHAM_symb = Derham(DOMAIN_symb)

    # grid parameters
    Nel      = [8, 8, 8]
    p        = [3, 3, 3]
    spl_kind = [True, True, True] 
    n_quad   = [4, 4, 4]
    bc       = [['f', 'f'], None, None]

    # Psydac discrete De Rham
    DOMAIN_PSY  = discretize(DOMAIN_symb, ncells=Nel, comm=MPI.COMM_WORLD) # The parallelism is initiated here.
    DERHAM_PSY  = discretize(DERHAM_symb, DOMAIN_PSY, degree=p, periodic=spl_kind)

    # Psydac spline spaces
    V0 = DERHAM_PSY.V0
    V1 = DERHAM_PSY.V1
    V2 = DERHAM_PSY.V2
    V3 = DERHAM_PSY.V3
    print(f'Rank {mpi_rank} | type(V0) {type(V0)}')
    print(f'Rank {mpi_rank} | type(V1) {type(V1)}')
    print(f'Rank {mpi_rank} | type(V2) {type(V2)}')
    print(f'Rank {mpi_rank} | type(V3) {type(V3)}')
    print(f'Rank {mpi_rank} | ')


    # Psydac projectors
    P0, P1, P2, P3  = DERHAM_PSY.projectors(nquads=n_quad)
    print(f'Rank {mpi_rank} | type(P0) {type(P0)}')
    print(f'Rank {mpi_rank} | type(P1) {type(P1)}')
    print(f'Rank {mpi_rank} | type(P2) {type(P2)}')
    print(f'Rank {mpi_rank} | type(P3) {type(P3)}')
    print(f'Rank {mpi_rank} | ')

    # Struphy spline spaces
    space_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0], n_quad[0]) 
    space_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1], n_quad[1])
    space_3 = Spline_space_1d(Nel[2], p[2], spl_kind[2], n_quad[2])

    space_1.set_projectors(n_quad[0]) 
    space_2.set_projectors(n_quad[1])
    space_3.set_projectors(n_quad[2])

    projectors_1d = (space_1.projectors, space_2.projectors, space_3.projectors)

    SPACES = Tensor_spline_space([space_1, space_2, space_3])
    SPACES.set_projectors('tensor')

    SPACES.assemble_M0(DOMAIN)
    SPACES.assemble_M1(DOMAIN) 
    SPACES.assemble_M2(DOMAIN)
    SPACES.assemble_M3(DOMAIN)

    # Mhd equilibirum
    mhd_equil_general = {'type': 'slab', 'mass_number' : 1 }
    mhd_equil_params = {'B0x': 0., 'B0y': 0., 'B0z': 1., 'rho0': 1., 'beta': 200.}

    EQ_MHD_P = Equilibrium_mhd_physical(mhd_equil_general['type'], mhd_equil_params)
    EQ_MHD_L = Equilibrium_mhd_logical(DOMAIN, EQ_MHD_P)

    # !!!Weird behaviour of lambda function!!!
    # fun_Q1 = []
    # for m in range(3):
    #     fun_Q1 += [[]]
    #     for n in range(3):
    #         fun_Q1[-1] += [lambda x1, x2, x3 : EQ_MHD_L.r3_eq(x1, x2, x3) * Ginv(x1, x2, x3)[m, n]]
            #print(m, n, fun_Q1[-1][-1])
            #print(fun_Q1[-1][-1](.5, .5, .5))
    #print('Ginv_mn')
    #for fun_line in fun_Q1:
        #print(fun_line)
        #for f in fun_line:
            #print(f)
            #print(f(.5, .5, .5))

    # Example for make m, n local variables:
    # print('For loop:')
    # _fun_Q1 = []
    # for m in range(3):
    #     _fun_Q1 += [[]]
    #     for n in range(3):
    #         _fun_Q1[-1] += [lambda x1, x2, x3, m=m, n=n : EQ_MHD_L.r3_eq(x1, x2, x3) * Ginv(x1, x2, x3)[m, n]]
    #         # _fun_Q1[-1] += [lambda x1, x2, x3, m=m, n=n : print(m, n)]
    # for m in range(3):
    #     for n in range(3):
    #         print(m, n, _fun_Q1[m][n])
    #         _fun_Q1[m][n](.5, .5, .5)

    # Multiplier functions in MHD operators:

    # Hard coding the indices m,n (lambda function does not work here, see above):
    fun_Q1 = []
    fun_Q1 += [[]]
    fun_Q1[-1] += [lambda x1, x2, x3 : EQ_MHD_L.r3_eq(x1, x2, x3) * Ginv(x1, x2, x3)[0, 0]]
    fun_Q1[-1] += [lambda x1, x2, x3 : EQ_MHD_L.r3_eq(x1, x2, x3) * Ginv(x1, x2, x3)[0, 1]]
    fun_Q1[-1] += [lambda x1, x2, x3 : EQ_MHD_L.r3_eq(x1, x2, x3) * Ginv(x1, x2, x3)[0, 2]]
    fun_Q1 += [[]]
    fun_Q1[-1] += [lambda x1, x2, x3 : EQ_MHD_L.r3_eq(x1, x2, x3) * Ginv(x1, x2, x3)[1, 0]]
    fun_Q1[-1] += [lambda x1, x2, x3 : EQ_MHD_L.r3_eq(x1, x2, x3) * Ginv(x1, x2, x3)[1, 1]]
    fun_Q1[-1] += [lambda x1, x2, x3 : EQ_MHD_L.r3_eq(x1, x2, x3) * Ginv(x1, x2, x3)[1, 2]]
    fun_Q1 += [[]]
    fun_Q1[-1] += [lambda x1, x2, x3 : EQ_MHD_L.r3_eq(x1, x2, x3) * Ginv(x1, x2, x3)[2, 0]]
    fun_Q1[-1] += [lambda x1, x2, x3 : EQ_MHD_L.r3_eq(x1, x2, x3) * Ginv(x1, x2, x3)[2, 1]]
    fun_Q1[-1] += [lambda x1, x2, x3 : EQ_MHD_L.r3_eq(x1, x2, x3) * Ginv(x1, x2, x3)[2, 2]]

    # lambda function without indices m,n works
    fun_W1 = []
    for m in range(3):
        fun_W1 += [[]]
        for n in range(3):
            if n == m:
                fun_W1[-1] += [lambda x1, x2, x3 : EQ_MHD_L.r3_eq(x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3))]
            else:
                fun_W1[-1] += [lambda x1, x2, x3 : 0.]

    fun_Q2 = fun_W1
    
    fun_K1  = lambda x1, x2, x3 : EQ_MHD_L.p3_eq(x1, x2, x3) / np.sqrt(F.metric_det(x1, x2, x3))
    fun_K10 = EQ_MHD_L.p0_eq

    # Psydac MHD operators
    Q1     = MHD_operator(V1, V2, P2, fun_Q1,      projectors_1d)
    W1     = MHD_operator(V1, V1, P1, fun_W1,      projectors_1d)
    Q2     = MHD_operator(V2, V2, P2, fun_Q2,      projectors_1d)
    K1     = MHD_operator(V3, V3, P3, [[fun_K1]],  projectors_1d)
    K10    = MHD_operator(V0, V0, P0, [[fun_K10]], projectors_1d)

    # Struphy matrix-free MHD operators
    OPS_STR = projectors_dot_x(SPACES, EQ_MHD_L, DOMAIN, 1, 0)

    # Test vectors
    #x0 = np.random.rand(*[space.nbasis for space in V0.spaces])
    x0 = np.reshape(np.arange(V0.nbasis), [space.nbasis for space in V0.spaces])
    x1 = [np.random.rand(*[space.nbasis for space in comp.spaces]) for comp in V1.spaces]
    #x1 = [np.ones(tuple([space.nbasis for space in comp.spaces])) for comp in V1.spaces]
    x2 = [np.random.rand(*[space.nbasis for space in comp.spaces]) for comp in V2.spaces]
    #x2 = [np.ones(tuple([space.nbasis for space in comp.spaces])) for comp in V2.spaces]
    #x3 = np.random.rand(*[space.nbasis for space in V3.spaces])
    x3 = np.reshape(np.arange(V3.nbasis), [space.nbasis for space in V3.spaces])
    print(x0.shape)
    print(x1[0].shape)
    print(x1[1].shape)
    print(x1[2].shape)
    print(x3.shape)

    x0_st = StencilVector(V0.vector_space)
    x1_st = BlockVector(  V1.vector_space, [StencilVector(comp) for comp in V1.vector_space])
    x2_st = BlockVector(  V2.vector_space, [StencilVector(comp) for comp in V2.vector_space])
    x3_st = StencilVector(V3.vector_space)

    # Use .copy() in case input will be overwritten (is not the case I guess)
    x0_st[
        x0_st.starts[0] : x0_st.ends[0] + 1,
        x0_st.starts[1] : x0_st.ends[1] + 1,
        x0_st.starts[2] : x0_st.ends[2] + 1,
    ] = x0[
        x0_st.starts[0] : x0_st.ends[0] + 1,
        x0_st.starts[1] : x0_st.ends[1] + 1,
        x0_st.starts[2] : x0_st.ends[2] + 1,
    ].copy()

    x1_st[0][
        x1_st[0].starts[0] : x1_st[0].ends[0] + 1,
        x1_st[0].starts[1] : x1_st[0].ends[1] + 1,
        x1_st[0].starts[2] : x1_st[0].ends[2] + 1,
    ] = x1[0][
        x1_st[0].starts[0] : x1_st[0].ends[0] + 1,
        x1_st[0].starts[1] : x1_st[0].ends[1] + 1,
        x1_st[0].starts[2] : x1_st[0].ends[2] + 1,
    ].copy()
    x1_st[1][
        x1_st[1].starts[0] : x1_st[1].ends[0] + 1,
        x1_st[1].starts[1] : x1_st[1].ends[1] + 1,
        x1_st[1].starts[2] : x1_st[1].ends[2] + 1,
    ] = x1[1][
        x1_st[1].starts[0] : x1_st[1].ends[0] + 1,
        x1_st[1].starts[1] : x1_st[1].ends[1] + 1,
        x1_st[1].starts[2] : x1_st[1].ends[2] + 1,
    ].copy()
    x1_st[2][
        x1_st[2].starts[0] : x1_st[2].ends[0] + 1,
        x1_st[2].starts[1] : x1_st[2].ends[1] + 1,
        x1_st[2].starts[2] : x1_st[2].ends[2] + 1,
    ] = x1[2][
        x1_st[2].starts[0] : x1_st[2].ends[0] + 1,
        x1_st[2].starts[1] : x1_st[2].ends[1] + 1,
        x1_st[2].starts[2] : x1_st[2].ends[2] + 1,
    ].copy()

    x2_st[0][
        x2_st[0].starts[0] : x2_st[0].ends[0] + 1,
        x2_st[0].starts[1] : x2_st[0].ends[1] + 1,
        x2_st[0].starts[2] : x2_st[0].ends[2] + 1,
    ] = x2[0][
        x2_st[0].starts[0] : x2_st[0].ends[0] + 1,
        x2_st[0].starts[1] : x2_st[0].ends[1] + 1,
        x2_st[0].starts[2] : x2_st[0].ends[2] + 1,
    ].copy()
    x2_st[1][
        x2_st[1].starts[0] : x2_st[1].ends[0] + 1,
        x2_st[1].starts[1] : x2_st[1].ends[1] + 1,
        x2_st[1].starts[2] : x2_st[1].ends[2] + 1,
    ] = x2[1][
        x2_st[1].starts[0] : x2_st[1].ends[0] + 1,
        x2_st[1].starts[1] : x2_st[1].ends[1] + 1,
        x2_st[1].starts[2] : x2_st[1].ends[2] + 1,
    ].copy()
    x2_st[2][
        x2_st[2].starts[0] : x2_st[2].ends[0] + 1,
        x2_st[2].starts[1] : x2_st[2].ends[1] + 1,
        x2_st[2].starts[2] : x2_st[2].ends[2] + 1,
    ] = x2[2][
        x2_st[2].starts[0] : x2_st[2].ends[0] + 1,
        x2_st[2].starts[1] : x2_st[2].ends[1] + 1,
        x2_st[2].starts[2] : x2_st[2].ends[2] + 1,
    ].copy()

    x3_st[
        x3_st.starts[0] : x3_st.ends[0] + 1,
        x3_st.starts[1] : x3_st.ends[1] + 1,
        x3_st.starts[2] : x3_st.ends[2] + 1,
    ] = x3[
        x3_st.starts[0] : x3_st.ends[0] + 1,
        x3_st.starts[1] : x3_st.ends[1] + 1,
        x3_st.starts[2] : x3_st.ends[2] + 1,
    ].copy()

    MPI_COMM.Barrier()

    x0_st.update_ghost_regions()
    x1_st.update_ghost_regions()
    x2_st.update_ghost_regions()
    x3_st.update_ghost_regions()

    # Compare to Struphy matrix-free operators 
    # See struphy.feec.projectors.pro_global.mhd_operators_MF.projectors_dot_x for the definition of these operators

    # operator K1
    print('\nK1 (=Identity operator in this case):')
    dof_PSY = K1._dofs_mat.dot(x3_st)
    print(type(dof_PSY))

    # Compare vectors of FE coeffs.
    # `x3_st.toarray()` returns only local values, the rest is zero.
    assert np.allclose(
        x3_st[
            x3_st.starts[0] : x3_st.ends[0] + 1,
            x3_st.starts[1] : x3_st.ends[1] + 1,
            x3_st.starts[2] : x3_st.ends[2] + 1,
        ],
        x3[
            x3_st.starts[0] : x3_st.ends[0] + 1,
            x3_st.starts[1] : x3_st.ends[1] + 1,
            x3_st.starts[2] : x3_st.ends[2] + 1,
        ]
    )

    res_PSY = K1.dot(x3_st)
    print(type(res_PSY))
    res_STR, dof_STR = OPS_STR.K1_dot(x3.flatten())

    # compare DOFs
    # assert np.allclose(dof_PSY.toarray(), dof_STR)
    assert np.allclose(
        dof_PSY[
            dof_PSY.starts[0] : dof_PSY.ends[0] + 1,
            dof_PSY.starts[1] : dof_PSY.ends[1] + 1,
            dof_PSY.starts[2] : dof_PSY.ends[2] + 1,
        ], SPACES.extract_3(dof_STR)[
            dof_PSY.starts[0] : dof_PSY.ends[0] + 1,
            dof_PSY.starts[1] : dof_PSY.ends[1] + 1,
            dof_PSY.starts[2] : dof_PSY.ends[2] + 1,
        ])

    # compare results (works only for Nel=[N, N, N] so far! TODO: find this bug!)
    # assert np.allclose(res_PSY.toarray(), res_STR)
    assert np.allclose(
        res_PSY[
            res_PSY.starts[0] : res_PSY.ends[0] + 1,
            res_PSY.starts[1] : res_PSY.ends[1] + 1,
            res_PSY.starts[2] : res_PSY.ends[2] + 1,
        ], SPACES.extract_3(res_STR)[
            res_PSY.starts[0] : res_PSY.ends[0] + 1,
            res_PSY.starts[1] : res_PSY.ends[1] + 1,
            res_PSY.starts[2] : res_PSY.ends[2] + 1,
        ])

    print('psydac result :', res_PSY.toarray()[:5])
    print('struphy result:', res_STR[:5])
    print('input vector  :', x3.flatten()[:5])

    print(f'Rank {mpi_rank} | ')
    print(f'Rank {mpi_rank} | res_STR starts & ends:')
    print([
        res_PSY.starts[0], res_PSY.ends[0] + 1,
        res_PSY.starts[1], res_PSY.ends[1] + 1,
        res_PSY.starts[2], res_PSY.ends[2] + 1,
    ])

    print(f'Rank {mpi_rank} | res_STR:')
    print(SPACES.extract_3(res_STR)[
        res_PSY.starts[0],
        res_PSY.starts[1] : res_PSY.ends[1] + 1,
        res_PSY.starts[2] : res_PSY.ends[2] + 1,
    ])

    print(f'Rank {mpi_rank} | res_PSY:')
    print(res_PSY[
        res_PSY.starts[0],
        res_PSY.starts[1] : res_PSY.ends[1] + 1,
        res_PSY.starts[2] : res_PSY.ends[2] + 1,
    ])
    print(f'Rank {mpi_rank} | ')

    MPI_COMM.Barrier()

    # operator K10
    print('\nK10 (=Identity operator in this case):')
    dof_PSY = K10._dofs_mat.dot(x0_st)
    print(type(dof_PSY))

    # Compare vectors of FE coeffs.
    # `x0_st.toarray()` returns only local values, the rest is zero.
    assert np.allclose(
        x0_st[
            x0_st.starts[0] : x0_st.ends[0] + 1,
            x0_st.starts[1] : x0_st.ends[1] + 1,
            x0_st.starts[2] : x0_st.ends[2] + 1,
        ],
        x0[
            x0_st.starts[0] : x0_st.ends[0] + 1,
            x0_st.starts[1] : x0_st.ends[1] + 1,
            x0_st.starts[2] : x0_st.ends[2] + 1,
        ]
    )

    res_PSY = K10.dot(x0_st)
    print(type(res_PSY))
    res_STR, dof_STR = OPS_STR.K10_dot(x0.flatten())

    # compare DOFs
    # assert np.allclose(dof_PSY.toarray(), dof_STR)
    assert np.allclose(
        dof_PSY[
            dof_PSY.starts[0] : dof_PSY.ends[0] + 1,
            dof_PSY.starts[1] : dof_PSY.ends[1] + 1,
            dof_PSY.starts[2] : dof_PSY.ends[2] + 1,
        ], SPACES.extract_0(dof_STR)[
            dof_PSY.starts[0] : dof_PSY.ends[0] + 1,
            dof_PSY.starts[1] : dof_PSY.ends[1] + 1,
            dof_PSY.starts[2] : dof_PSY.ends[2] + 1,
        ])

    # compare results (works only for Nel=[N, N, N] so far! TODO: find this bug!)
    # assert np.allclose(res_PSY.toarray(), res_STR)
    assert np.allclose(
        res_PSY[
            res_PSY.starts[0] : res_PSY.ends[0] + 1,
            res_PSY.starts[1] : res_PSY.ends[1] + 1,
            res_PSY.starts[2] : res_PSY.ends[2] + 1,
        ], SPACES.extract_0(res_STR)[
            res_PSY.starts[0] : res_PSY.ends[0] + 1,
            res_PSY.starts[1] : res_PSY.ends[1] + 1,
            res_PSY.starts[2] : res_PSY.ends[2] + 1,
        ])

    print('psydac result :', res_PSY.toarray()[:5])
    print('struphy result:', res_STR[:5])
    print('input vector  :', x0.flatten()[:5])

    print(f'Rank {mpi_rank} | ')
    print(f'Rank {mpi_rank} | res_STR starts & ends:')
    print([
        res_PSY.starts[0], res_PSY.ends[0] + 1,
        res_PSY.starts[1], res_PSY.ends[1] + 1,
        res_PSY.starts[2], res_PSY.ends[2] + 1,
    ])

    print(f'Rank {mpi_rank} | res_STR:')
    print(SPACES.extract_0(res_STR)[
        res_PSY.starts[0],
        res_PSY.starts[1] : res_PSY.ends[1] + 1,
        res_PSY.starts[2] : res_PSY.ends[2] + 1,
    ])

    print(f'Rank {mpi_rank} | res_PSY:')
    print(res_PSY[
        res_PSY.starts[0],
        res_PSY.starts[1] : res_PSY.ends[1] + 1,
        res_PSY.starts[2] : res_PSY.ends[2] + 1,
    ])
    print(f'Rank {mpi_rank} | ')

    MPI_COMM.Barrier()

    # operator Q1
    print('\nQ1:')
    dof_PSY = Q1._dofs_mat.dot(x1_st)
    print(type(dof_PSY))
    print(type(dof_PSY[0]))
    res_PSY = Q1.dot(x1_st)
    print(type(res_PSY))
    res_STR, dof_STR_0, dof_STR_1, dof_STR_2 = OPS_STR.Q1_dot( np.concatenate((x1[0].flatten(), x1[1].flatten(), x1[2].flatten())) )
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_2(res_STR)

    print('\nFirst component')
    print(res_PSY[0].toarray()[:5])
    print(res_STR_0.flatten()[:5])

    assert np.allclose(dof_PSY[0].toarray(), dof_STR_0)
    assert np.allclose(res_PSY[0].toarray(), res_STR_0.flatten())

    print('\nSecond component')
    print(res_PSY[1].toarray()[:5])
    print(res_STR_1.flatten()[:5])

    assert np.allclose(dof_PSY[1].toarray(), dof_STR_1)
    assert np.allclose(res_PSY[1].toarray(), res_STR_1.flatten())

    print('\nThird component')
    print(res_PSY[2].toarray()[:5])
    print(res_STR_2.flatten()[:5])

    assert np.allclose(dof_PSY[2].toarray(), dof_STR_2)
    assert np.allclose(res_PSY[2].toarray(), res_STR_2.flatten())

    # operator W1
    print('\nW1:')
    dof_PSY = W1._dofs_mat.dot(x1_st)
    print(type(dof_PSY))
    print(type(dof_PSY[0]))
    res_PSY = W1.dot(x1_st)
    print(type(res_PSY))
    res_STR, dof_STR_0, dof_STR_1, dof_STR_2 = OPS_STR.W1_dot( np.concatenate((x1[0].flatten(), x1[1].flatten(), x1[2].flatten())) )
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_1(res_STR)

    print('\nFirst component')
    print(res_PSY[0].toarray()[:5])
    print(res_STR_0.flatten()[:5])

    assert np.allclose(dof_PSY[0].toarray(), dof_STR_0)
    assert np.allclose(res_PSY[0].toarray(), res_STR_0.flatten())

    print('\nSecond component')
    print(res_PSY[1].toarray()[:5])
    print(res_STR_1.flatten()[:5])

    assert np.allclose(dof_PSY[1].toarray(), dof_STR_1)
    assert np.allclose(res_PSY[1].toarray(), res_STR_1.flatten())

    print('\nThird component')
    print(res_PSY[2].toarray()[:5])
    print(res_STR_2.flatten()[:5])

    assert np.allclose(dof_PSY[2].toarray(), dof_STR_2)
    assert np.allclose(res_PSY[2].toarray(), res_STR_2.flatten())

    # operator Q2
    print('\nQ2:')
    dof_PSY = Q2._dofs_mat.dot(x2_st)
    print(type(dof_PSY))
    print(type(dof_PSY[0]))
    res_PSY = Q2.dot(x2_st)
    print(type(res_PSY))
    res_STR, dof_STR_0, dof_STR_1, dof_STR_2 = OPS_STR.Q2_dot( np.concatenate((x2[0].flatten(), x2[1].flatten(), x2[2].flatten())) )
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_2(res_STR)

    print('\nFirst component')
    print(res_PSY[0].toarray()[:5])
    print(res_STR_0.flatten()[:5])

    assert np.allclose(dof_PSY[0].toarray(), dof_STR_0)
    assert np.allclose(res_PSY[0].toarray(), res_STR_0.flatten())

    print('\nSecond component')
    print(res_PSY[1].toarray()[:5])
    print(res_STR_1.flatten()[:5])

    assert np.allclose(dof_PSY[1].toarray(), dof_STR_1)
    assert np.allclose(res_PSY[1].toarray(), res_STR_1.flatten())

    print('\nThird component')
    print(res_PSY[2].toarray()[:5])
    print(res_STR_2.flatten()[:5])
    
    assert np.allclose(dof_PSY[2].toarray(), dof_STR_2)
    assert np.allclose(res_PSY[2].toarray(), res_STR_2.flatten())



if __name__ == '__main__':
    test_lin_mhd_ops()