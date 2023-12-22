import pytest


@pytest.mark.parametrize('Nel', [[8, 12, 4]])
@pytest.mark.parametrize('p',   [[2, 3, 2]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}]
])
def test_some_basis_ops(Nel, p, spl_kind, mapping):
    '''Tests the MHD specific projection operators PI_ijk(fun*Lambda_mno).

    Here, PI_ijk is the commuting projector of the output space (codomain), 
    Lambda_mno are the basis functions of the input space (domain), 
    and fun is an arbitrary (matrix-valued) function.
    '''
    from struphy.geometry import domains
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    
    from struphy.fields_background.mhd_equil.equils import HomogenSlab
    
    from struphy.eigenvalue_solvers.legacy.mhd_operators_MF import projectors_dot_x

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.basis_projection_ops import BasisProjectionOperators

    from psydac.linalg.stencil import StencilVector
    from psydac.linalg.block import BlockVector

    from mpi4py import MPI
    from time import time
    import numpy as np

    # mpi communicator
    MPI_COMM = MPI.COMM_WORLD
    mpi_rank = MPI_COMM.Get_rank()
    MPI_COMM.Barrier()

    # Domain object
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    # de Rham object
    n_quad_el = [5, 5, 5]
    n_quad_pr = [4, 4, 4]

    DERHAM_PSY = Derham(Nel, p, spl_kind, nq_pr=n_quad_pr,
                        nquads=n_quad_el, comm=MPI_COMM)

    # grid parameters
    if mpi_rank == 0:
        print(f'Rank {mpi_rank} | Nel: {Nel}')
        print(f'Rank {mpi_rank} | p: {p}')
        print(f'Rank {mpi_rank} | spl_kind: {spl_kind}')
        print(f'Rank {mpi_rank} | ')

    # Mhd equilibirum (slab)
    mhd_equil_params = {'B0x': 0., 'B0y': 0.,
                        'B0z': 1., 'beta': 2., 'n0': 1.}

    EQ_MHD = HomogenSlab(**mhd_equil_params)
    EQ_MHD.domain = domain

    # Psydac spline spaces
    V0 = DERHAM_PSY.Vh_fem['0']
    V1 = DERHAM_PSY.Vh_fem['1']
    V2 = DERHAM_PSY.Vh_fem['2']
    V3 = DERHAM_PSY.Vh_fem['3']
    V0vec = DERHAM_PSY.Vh_fem['v']

    if mpi_rank == 0:
        print(f'Rank {mpi_rank} | type(V0) {type(V0)}')
        print(f'Rank {mpi_rank} | type(V1) {type(V1)}')
        print(f'Rank {mpi_rank} | type(V2) {type(V2)}')
        print(f'Rank {mpi_rank} | type(V3) {type(V3)}')
        print(f'Rank {mpi_rank} | type(V0vec) {type(V0vec)}')
        print(f'Rank {mpi_rank} | ')

    # Psydac projectors
    P0 = DERHAM_PSY.P['0']
    P1 = DERHAM_PSY.P['1']
    P2 = DERHAM_PSY.P['2']
    P3 = DERHAM_PSY.P['3']
    P0vec = DERHAM_PSY.P['v']
    if mpi_rank == 0:
        print(f'Rank {mpi_rank} | type(P0) {type(P0)}')
        print(f'Rank {mpi_rank} | type(P1) {type(P1)}')
        print(f'Rank {mpi_rank} | type(P2) {type(P2)}')
        print(f'Rank {mpi_rank} | type(P3) {type(P3)}')
        print(f'Rank {mpi_rank} | type(P0vec) {type(P0vec)}')
        print(f'Rank {mpi_rank} | ')

    # Struphy spline spaces
    space_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0], n_quad_el[0] + 1)
    space_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1], n_quad_el[1] + 1)
    space_3 = Spline_space_1d(Nel[2], p[2], spl_kind[2], n_quad_el[2] + 1)

    space_1.set_projectors(n_quad_pr[0])
    space_2.set_projectors(n_quad_pr[1])
    space_3.set_projectors(n_quad_pr[2])

    # print('\nSTRUPHY point sets:')
    # print('\nDirection 1:')
    # print(f'x_int: {space_1.projectors.x_int}')
    # print(f'x_hisG: {space_1.projectors.x_hisG}')
    # print(f'x_his: {space_1.projectors.x_his}')
    # print('\nDirection 2:')
    # print(f'x_int: {space_2.projectors.x_int}')
    # print(f'x_hisG: {space_2.projectors.x_hisG}')
    # print(f'x_his: {space_2.projectors.x_his}')
    # print('\nDirection 3:')
    # print(f'x_int: {space_3.projectors.x_int}')
    # print(f'x_hisG: {space_3.projectors.x_hisG}')
    # print(f'x_his: {space_3.projectors.x_his}')

    SPACES = Tensor_spline_space([space_1, space_2, space_3])
    SPACES.set_projectors('tensor')

    # Psydac MHD operators
    OPS_PSY = BasisProjectionOperators(DERHAM_PSY, domain, eq_mhd=EQ_MHD)

    # Struphy matrix-free MHD operators
    print(f'Rank {mpi_rank} | Init STRUPHY `projectors_dot_x`...')
    elapsed = time()
    OPS_STR = projectors_dot_x(SPACES, EQ_MHD)
    print(
        f'Rank {mpi_rank} | Init `projectors_dot_x` done ({time()-elapsed:.4f}s).')

    # Test vectors
    x0 = np.reshape(np.arange(V0.nbasis), [
                    space.nbasis for space in V0.spaces])

    x1 = [np.reshape(np.arange(comp.nbasis), [
                     space.nbasis for space in comp.spaces]) for comp in V1.spaces]

    x2 = [np.reshape(np.arange(comp.nbasis), [
                     space.nbasis for space in comp.spaces]) for comp in V2.spaces]

    x3 = np.reshape(np.arange(V3.nbasis), [
                    space.nbasis for space in V3.spaces])

    x0_st = StencilVector(V0.vector_space)
    x1_st = BlockVector(
        V1.vector_space, [StencilVector(comp) for comp in V1.vector_space])
    x2_st = BlockVector(
        V2.vector_space, [StencilVector(comp) for comp in V2.vector_space])
    x3_st = StencilVector(V3.vector_space)

    # for testing X1T:
    x0vec_st = BlockVector(V0vec.vector_space, [
                           StencilVector(comp) for comp in V0vec.vector_space])

    MPI_COMM.Barrier()

    print(
        f'rank: {mpi_rank} | x3_starts[0]: {x3_st.starts[0]}, x3_ends[0]: {x3_st.ends[0]}')
    MPI_COMM.Barrier()
    print(
        f'rank: {mpi_rank} | x3_starts[1]: {x3_st.starts[1]}, x3_ends[1]: {x3_st.ends[1]}')
    MPI_COMM.Barrier()
    print(
        f'rank: {mpi_rank} | x3_starts[2]: {x3_st.starts[2]}, x3_ends[2]: {x3_st.ends[2]}')
    MPI_COMM.Barrier()

    # Use .copy() in case input will be overwritten (is not the case I guess)
    x0_st[
        x0_st.starts[0]: x0_st.ends[0] + 1,
        x0_st.starts[1]: x0_st.ends[1] + 1,
        x0_st.starts[2]: x0_st.ends[2] + 1,
    ] = x0[
        x0_st.starts[0]: x0_st.ends[0] + 1,
        x0_st.starts[1]: x0_st.ends[1] + 1,
        x0_st.starts[2]: x0_st.ends[2] + 1,
    ].copy()

    for n in range(3):
        x1_st[n][
            x1_st[n].starts[0]: x1_st[n].ends[0] + 1,
            x1_st[n].starts[1]: x1_st[n].ends[1] + 1,
            x1_st[n].starts[2]: x1_st[n].ends[2] + 1,
        ] = x1[n][
            x1_st[n].starts[0]: x1_st[n].ends[0] + 1,
            x1_st[n].starts[1]: x1_st[n].ends[1] + 1,
            x1_st[n].starts[2]: x1_st[n].ends[2] + 1,
        ].copy()

    for n in range(3):
        x2_st[n][
            x2_st[n].starts[0]: x2_st[n].ends[0] + 1,
            x2_st[n].starts[1]: x2_st[n].ends[1] + 1,
            x2_st[n].starts[2]: x2_st[n].ends[2] + 1,
        ] = x2[n][
            x2_st[n].starts[0]: x2_st[n].ends[0] + 1,
            x2_st[n].starts[1]: x2_st[n].ends[1] + 1,
            x2_st[n].starts[2]: x2_st[n].ends[2] + 1,
        ].copy()

    x3_st[
        x3_st.starts[0]: x3_st.ends[0] + 1,
        x3_st.starts[1]: x3_st.ends[1] + 1,
        x3_st.starts[2]: x3_st.ends[2] + 1,
    ] = x3[
        x3_st.starts[0]: x3_st.ends[0] + 1,
        x3_st.starts[1]: x3_st.ends[1] + 1,
        x3_st.starts[2]: x3_st.ends[2] + 1,
    ].copy()

    for n in range(3):
        x0vec_st[n][
            x0vec_st[n].starts[0]: x0vec_st[n].ends[0] + 1,
            x0vec_st[n].starts[1]: x0vec_st[n].ends[1] + 1,
            x0vec_st[n].starts[2]: x0vec_st[n].ends[2] + 1,
        ] = x0[
            x0vec_st[n].starts[0]: x0vec_st[n].ends[0] + 1,
            x0vec_st[n].starts[1]: x0vec_st[n].ends[1] + 1,
            x0vec_st[n].starts[2]: x0vec_st[n].ends[2] + 1,
        ].copy()

    MPI_COMM.Barrier()

    x0_st.update_ghost_regions()
    x1_st.update_ghost_regions()
    x2_st.update_ghost_regions()
    x3_st.update_ghost_regions()

    MPI_COMM.Barrier()

    # Compare to Struphy matrix-free operators
    # See struphy.feec.projectors.pro_global.mhd_operators_MF.projectors_dot_x for the definition of these operators

    # operator K3 (V3 --> V3)
    if mpi_rank == 0:
        print('\nK3 (V3 --> V3, Identity operator in this case):')

    res_PSY = OPS_PSY.K3.dot(x3_st)
    res_STR = OPS_STR.K1_dot(x3.flatten())
    res_STR = SPACES.extract_3(res_STR)

    print(f'Rank {mpi_rank} | Asserting MHD operator K3.')
    assert_ops(mpi_rank, res_PSY, res_STR, verbose=True)
    print(f'Rank {mpi_rank} | Assertion passed.')

    K3T = OPS_PSY.K3.transpose()
    res_PSY = K3T.dot(x3_st)
    res_STR = OPS_STR.transpose_K1_dot(x3.flatten())
    res_STR = SPACES.extract_3(res_STR)

    print(f'Rank {mpi_rank} | Asserting TRANSPOSE MHD operator K3T.')
    assert_ops(mpi_rank, res_PSY, res_STR, verbose=True)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    # operator K0 (V0 --> V0)
    if mpi_rank == 0:
        print('\nK0 (V0 --> V0, Identity operator in this case):')

    res_PSY = OPS_PSY.K0.dot(x0_st)
    res_STR = OPS_STR.K10_dot(x0.flatten())
    res_STR = SPACES.extract_0(res_STR)

    print(f'Rank {mpi_rank} | Asserting MHD operator K0.')
    assert_ops(mpi_rank, res_PSY, res_STR, verbose=True)
    print(f'Rank {mpi_rank} | Assertion passed.')

    K10T = OPS_PSY.K0.transpose()
    res_PSY = K10T.dot(x0_st)
    res_STR = OPS_STR.transpose_K10_dot(x0.flatten())
    res_STR = SPACES.extract_0(res_STR)

    print(f'Rank {mpi_rank} | Asserting TRANSPOSE MHD operator K10T.')
    assert_ops(mpi_rank, res_PSY, res_STR, verbose=True)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    # operator Q1 (V1 --> V2)
    if mpi_rank == 0:
        print('\nQ1 (V1 --> V2):')

    res_PSY = OPS_PSY.Q1.dot(x1_st)
    res_STR = OPS_STR.Q1_dot(
        np.concatenate((x1[0].flatten(), x1[1].flatten(), x1[2].flatten())))
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_2(res_STR)

    MPI_COMM.Barrier()

    print(f'Rank {mpi_rank} | Asserting MHD operator Q1, first component.')
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(f'Rank {mpi_rank} | Asserting MHD operator Q1, second component.')
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(f'Rank {mpi_rank} | Asserting MHD operator Q1, third component.')
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f'Rank {mpi_rank} | Assertion passed.')

    Q1T = OPS_PSY.Q1.transpose()
    res_PSY = Q1T.dot(x2_st)
    res_STR = OPS_STR.transpose_Q1_dot(
        np.concatenate((x2[0].flatten(), x2[1].flatten(), x2[2].flatten())))
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_1(res_STR)

    MPI_COMM.Barrier()

    print(
        f'Rank {mpi_rank} | Asserting TRANSPOSE MHD operator Q1T, first component.')
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(
        f'Rank {mpi_rank} | Asserting TRANSPOSE MHD operator Q1T, second component.')
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(
        f'Rank {mpi_rank} | Asserting TRANSPOSE MHD operator Q1T, third component.')
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # operator W1 (V1 --> V1)
    if mpi_rank == 0:
        print('\nW1 (V1 --> V1, Identity operator in this case):')

    res_PSY = OPS_PSY.W1.dot(x1_st)
    res_STR = OPS_STR.W1_dot(
        np.concatenate((x1[0].flatten(), x1[1].flatten(), x1[2].flatten())))
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_1(res_STR)

    MPI_COMM.barrier()

    print(f'Rank {mpi_rank} | Asserting MHD operator W1, first component.')
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(f'Rank {mpi_rank} | Asserting MHD operator W1, second component.')
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(f'Rank {mpi_rank} | Asserting MHD operator W1, third component.')
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f'Rank {mpi_rank} | Assertion passed.')

    W1T = OPS_PSY.W1.transpose()
    res_PSY = W1T.dot(x1_st)
    res_STR = OPS_STR.transpose_W1_dot(
        np.concatenate((x1[0].flatten(), x1[1].flatten(), x1[2].flatten())))
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_1(res_STR)

    MPI_COMM.barrier()

    print(
        f'Rank {mpi_rank} | Asserting TRANSPOSE MHD operator W1T, first component.')
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(
        f'Rank {mpi_rank} | Asserting TRANSPOSE MHD operator W1T, second component.')
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(
        f'Rank {mpi_rank} | Asserting TRANSPOSE MHD operator W1T, third component.')
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # operator Q2 (V2 --> V2)
    if mpi_rank == 0:
        print('\nQ2 (V2 --> V2, Identity operator in this case):')

    res_PSY = OPS_PSY.Q2.dot(x2_st)
    res_STR = OPS_STR.Q2_dot(
        np.concatenate((x2[0].flatten(), x2[1].flatten(), x2[2].flatten())))
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_2(res_STR)

    MPI_COMM.Barrier()

    print(f'Rank {mpi_rank} | Asserting MHD operator Q2, first component.')
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(f'Rank {mpi_rank} | Asserting MHD operator Q2, second component.')
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(f'Rank {mpi_rank} | Asserting MHD operator Q2, third component.')
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f'Rank {mpi_rank} | Assertion passed.')

    Q2T = OPS_PSY.Q2.transpose()
    res_PSY = Q2T.dot(x2_st)
    res_STR = OPS_STR.transpose_Q2_dot(
        np.concatenate((x2[0].flatten(), x2[1].flatten(), x2[2].flatten())))
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_2(res_STR)

    MPI_COMM.Barrier()

    print(
        f'Rank {mpi_rank} | Asserting TRANSPOSE MHD operator Q2T, first component.')
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(
        f'Rank {mpi_rank} | Asserting TRANSPOSE MHD operator Q2T, second component.')
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(
        f'Rank {mpi_rank} | Asserting TRANSPOSE MHD operator Q2T, third component.')
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # operator X1 (V1 --> V0 x V0 x V0)
    if mpi_rank == 0:
        print('\nX1 (V1 --> V0 x V0 x V0):')

    res_PSY = OPS_PSY.X1.dot(x1_st)
    res_STR = OPS_STR.X1_dot(
        np.concatenate((x1[0].flatten(), x1[1].flatten(), x1[2].flatten())))
    res_STR_0 = SPACES.extract_0(res_STR[0])
    res_STR_1 = SPACES.extract_0(res_STR[1])
    res_STR_2 = SPACES.extract_0(res_STR[2])

    MPI_COMM.Barrier()

    print(f'Rank {mpi_rank} | Asserting MHD operator X1, first component.')
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(f'Rank {mpi_rank} | Asserting MHD operator X1, second component.')
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(f'Rank {mpi_rank} | Asserting MHD operator X1, third component.')
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f'Rank {mpi_rank} | Assertion passed.')

    X1T = OPS_PSY.X1.transpose()
    res_PSY = X1T.dot(x0vec_st)
    res_STR = OPS_STR.transpose_X1_dot(
        [x0.flatten(), x0.flatten(), x0.flatten()])
    res_STR_0, res_STR_1, res_STR_2 = SPACES.extract_1(res_STR)

    MPI_COMM.Barrier()

    print(
        f'Rank {mpi_rank} | Asserting TRANSPOSE MHD operator X1T, first component.')
    assert_ops(mpi_rank, res_PSY[0], res_STR_0)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(
        f'Rank {mpi_rank} | Asserting TRANSPOSE MHD operator X1T, second component.')
    assert_ops(mpi_rank, res_PSY[1], res_STR_1)
    print(f'Rank {mpi_rank} | Assertion passed.')

    MPI_COMM.Barrier()

    print(
        f'Rank {mpi_rank} | Asserting TRANSPOSE MHD operator X1T, third component.')
    assert_ops(mpi_rank, res_PSY[2], res_STR_2)
    print(f'Rank {mpi_rank} | Assertion passed.')


def assert_ops(mpi_rank, res_PSY, res_STR, verbose=False, MPI_COMM=None):
    """
    TODO
    """

    import numpy as np

    if verbose:

        if MPI_COMM is not None:
            MPI_COMM.Barrier()

        # print(f'Rank {mpi_rank} | ')
        # print(f'Rank {mpi_rank} | res_PSY.shape   : {res_PSY.shape}')
        # print(f'Rank {mpi_rank} | res_PSY[:].shape: {res_PSY[:].shape}')
        # print(f'Rank {mpi_rank} | res_STR.shape   : {res_STR.shape}')

        # print(f'Rank {mpi_rank} | res_PSY starts & ends:')
        # print([
        #     res_PSY.starts[0], res_PSY.ends[0] + 1,
        #     res_PSY.starts[1], res_PSY.ends[1] + 1,
        #     res_PSY.starts[2], res_PSY.ends[2] + 1,
        # ])

        # print(f'Rank {mpi_rank} | res_PSY starts & ends:')
        # print([
        #     res_PSY.starts[0], res_PSY.ends[0] + 1,
        #     res_PSY.starts[1], res_PSY.ends[1] + 1,
        #     res_PSY.starts[2], res_PSY.ends[2] + 1,
        # ])

        # if MPI_COMM is not None: MPI_COMM.Barrier()

        # print(f'Rank {mpi_rank} | res_PSY (local slice at starts[0]):')
        # print(res_PSY[
        #     res_PSY.starts[0],
        #     res_PSY.starts[1] : res_PSY.ends[1] + 1,
        #     res_PSY.starts[2] : res_PSY.ends[2] + 1,
        # ])

        # print(f'Rank {mpi_rank} | res_STR (local slice at starts[0]):')
        # print(res_STR[
        #     res_PSY.starts[0],
        #     res_PSY.starts[1] : res_PSY.ends[1] + 1,
        #     res_PSY.starts[2] : res_PSY.ends[2] + 1,
        # ])
        # print(f'Rank {mpi_rank} | ')

        # for n in range(res_PSY.ends[0] + 1):

        #     print(f'Rank {mpi_rank} | dof_PSY (local slice at starts[0] + {n}):')
        #     print(dof_PSY[
        #         res_PSY.starts[0] + n,
        #         res_PSY.starts[1] : res_PSY.ends[1] + 1,
        #         res_PSY.starts[2] : res_PSY.ends[2] + 1,
        #     ])

        #     print(f'Rank {mpi_rank} | dof_STR (local slice at starts[0] + {n}):')
        #     print(dof_STR[
        #         res_PSY.starts[0] + n,
        #         res_PSY.starts[1] : res_PSY.ends[1] + 1,
        #         res_PSY.starts[2] : res_PSY.ends[2] + 1,
        #     ])
        #     print(f'Rank {mpi_rank} | ')

        # if MPI_COMM is not None: MPI_COMM.Barrier()

        print(f'Rank {mpi_rank} | Maximum absolute diference (result):\n', np.max(np.abs(
            res_PSY[
                res_PSY.starts[0]: res_PSY.ends[0] + 1,
                res_PSY.starts[1]: res_PSY.ends[1] + 1,
                res_PSY.starts[2]: res_PSY.ends[2] + 1,
            ] -
            res_STR[
                res_PSY.starts[0]: res_PSY.ends[0] + 1,
                res_PSY.starts[1]: res_PSY.ends[1] + 1,
                res_PSY.starts[2]: res_PSY.ends[2] + 1,
            ]
        )))

    if MPI_COMM is not None:
        MPI_COMM.Barrier()

    # Compare results. (Works only for Nel=[N, N, N] so far! TODO: Find this bug!)
    assert np.allclose(
        res_PSY[
            res_PSY.starts[0]: res_PSY.ends[0] + 1,
            res_PSY.starts[1]: res_PSY.ends[1] + 1,
            res_PSY.starts[2]: res_PSY.ends[2] + 1,
        ],
        res_STR[
            res_PSY.starts[0]: res_PSY.ends[0] + 1,
            res_PSY.starts[1]: res_PSY.ends[1] + 1,
            res_PSY.starts[2]: res_PSY.ends[2] + 1,
        ])

    if MPI_COMM is not None:
        MPI_COMM.Barrier()


if __name__ == '__main__':
    test_some_basis_ops(Nel=[8, 8, 8], p=[2, 2, 2], spl_kind=[False, True, True],
                        mapping=['Cuboid', {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}])
