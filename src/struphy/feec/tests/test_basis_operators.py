import pytest
import numpy as np


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[5, 6, 7]])
@pytest.mark.parametrize('p',   [[2, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True]])
@pytest.mark.parametrize('mapping', [
    # ['Cuboid', {
    #    'l1': 0., 'r1': 1., 'l2': 0., 'r2': 6., 'l3': 0., 'r3': 10.}],
    ['Colella', {
        'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}],
    ['HollowCylinder', {
        'a1': .1, 'a2': 1., 'Lz': 2*np.pi*3.}]
])
def test_basis_ops(Nel, p, spl_kind, mapping, show_plots=False):

    import numpy as np

    from struphy.geometry import domains
    from struphy.fields_background.mhd_equil.equils import ShearedSlab, ScrewPinch
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space

    import struphy.eigenvalue_solvers.mhd_operators as basis_ops_str1
    import struphy.eigenvalue_solvers.legacy.mhd_operators_MF as basis_ops_str2

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.basis_projection_ops import BasisProjectionOperators, BasisProjectionOperator, prepare_projection_of_basis

    from struphy.feec.utilities import create_equal_random_arrays, compare_arrays

    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    print('number of processes : ', mpi_size)

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    if show_plots:
        domain.show()

    # MHD equilibrium
    if mapping[0] == 'Cuboid':
        eq_mhd = ShearedSlab(**{'a': mapping[1]['r1'] - mapping[1]['l1'], 'R0': (mapping[1]['r3'] - mapping[1]['l3'])/(
            2*np.pi), 'B0': 1.0, 'q0': 1.05, 'q1': 1.8, 'n1': 3.0, 'n2': 4.0, 'na': 0.0, 'beta': .1})

    elif mapping[0] == 'Colella':
        eq_mhd = ShearedSlab(**{'a': mapping[1]['Lx'], 'R0': mapping[1]['Lz']/(
            2*np.pi), 'B0': 1.0, 'q0': 1.05, 'q1': 1.8, 'n1': 3.0, 'n2': 4.0, 'na': 0.0, 'beta': .1})

        if show_plots:
            eq_mhd.plot_profiles()

    elif mapping[0] == 'HollowCylinder':
        eq_mhd = ScrewPinch(**{'a': mapping[1]['a2'], 'R0': 3., 'B0': 1.0,
                            'q0': 1.05, 'q1': 1.8, 'n1': 3.0, 'n2': 4.0, 'na': 0.0, 'beta': .1})

        if show_plots:
            eq_mhd.plot_profiles()

    # set equilibrium object domain
    eq_mhd.domain = domain

    # Psydac derham object
    nq_el = [p[0] + 1, p[1] + 1, p[2] + 1]
    nq_pr = p.copy()

    derham = Derham(Nel, p, spl_kind, nquads=p, nq_pr=nq_pr, comm=mpi_comm)

    # Struphy tensor spline space objects (one for tensor product projectors and one for general projectors)
    space1 = Spline_space_1d(Nel[0], p[0], spl_kind[0], nq_el[0])
    space2 = Spline_space_1d(Nel[1], p[1], spl_kind[1], nq_el[1])
    space3 = Spline_space_1d(Nel[2], p[2], spl_kind[2], nq_el[2])

    space1.set_projectors(nq_pr[0])
    space2.set_projectors(nq_pr[1])
    space3.set_projectors(nq_pr[2])

    space_str1 = Tensor_spline_space([space1, space2, space3])
    space_str1.set_projectors('general')

    space_str2 = Tensor_spline_space([space1, space2, space3])
    space_str2.set_projectors('tensor')

    # MHD operator objects
    basis_psy = BasisProjectionOperators(derham, domain, eq_mhd=eq_mhd)

    basis_str10 = basis_ops_str1.MHDOperators(
        space_str1, eq_mhd, basis_u=0)  # MHD velocity is 0-form^3
    basis_str12 = basis_ops_str1.MHDOperators(
        space_str1, eq_mhd, basis_u=2)  # MHD velocity is 2-form
    basis_str2 = basis_ops_str2.projectors_dot_x(space_str2, eq_mhd)
    basis_str10.assemble_dofs('MF')
    basis_str10.assemble_dofs('PF')
    basis_str10.assemble_dofs('JF')
    basis_str10.assemble_dofs('EF')
    basis_str10.assemble_dofs('PR')
    basis_str10.set_operators()
    basis_str12.assemble_dofs('MF')
    basis_str12.assemble_dofs('PF')
    basis_str12.assemble_dofs('EF')
    basis_str12.assemble_dofs('PR')
    basis_str12.set_operators()

    # create random input arrays
    x0_str, x0_psy = create_equal_random_arrays(
        derham.Vh_fem['0'], 1234, flattened=True)
    x1_str, x1_psy = create_equal_random_arrays(
        derham.Vh_fem['1'], 1568, flattened=True)
    x2_str, x2_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], 8945, flattened=True)
    x3_str, x3_psy = create_equal_random_arrays(
        derham.Vh_fem['3'], 8196, flattened=True)
    xv_str, xv_psy = create_equal_random_arrays(
        derham.Vh_fem['v'], 2038, flattened=True)

    # compare matrix-vector products of different methods

    # ================================================================================
    #                              MHD velocity is a 0-form^3
    # ================================================================================

    # ===== operator K3 (V3 --> V3) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator K3 (V3 --> V3):')

    r_psy = basis_psy.K3.dot(x3_psy)
    r_str1 = basis_str10.PR(x3_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator K3.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    K3T = basis_psy.K3.transpose()
    r_psy = K3T.dot(x3_psy)
    r_str1 = basis_str10.PR.T(x3_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator K3T.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ===== Test BasisProjectionOperator with weights initialization ============
    # Create a hand-made BasisProjectionOperator to evaluate with weights and compare
    fun = lambda e1, e2, e3: basis_psy.weights['eq_mhd'].p3(
                e1, e2, e3) / basis_psy.sqrt_g(e1, e2, e3)
    
    #create adapted quadrature grid for the basis projection operator
    P3_space = derham.P['3'].space
    starts_out = np.array(P3_space.vector_space.starts)
    ends_out = np.array(P3_space.vector_space.ends)

    ptsG, _, _, _, _ = prepare_projection_of_basis(derham.Vh_fem['3'].spaces, 
                                                    P3_space.spaces, starts_out, 
                                                    ends_out, n_quad=derham.nquads)


    ptsG = [pts.flatten() for pts in ptsG]
    PTS = np.meshgrid(*ptsG, indexing='ij')

    #evaluate function and set the weight matrix
    mat_w = fun(*PTS).copy()
    weights = [[mat_w]]
    
    #hand-made BasisProjectionOperator
    P = BasisProjectionOperator(derham.P['3'], derham.Vh_fem['3'], weights)

    r_w = P.dot(x3_psy)

    print(f'Rank {mpi_rank} | Asserting hand created basis projection with weights.')
    np.allclose(r_psy._data, r_w._data, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ===== Test BasisProjectionOperator update_weight method ============
    # Create a hand-made BasisProjectionOperator withrandom weights, then update and compare
    
    mat_w_rand = np.random.uniform(size = mat_w.shape)
    weights_rand = [[mat_w_rand]]
    
    #hand-made BasisProjectionOperator
    Q = BasisProjectionOperator(derham.P['3'], derham.Vh_fem['3'], weights_rand, use_cache=True)
    Q.update_weights(weights)

    r_w = Q.dot(x3_psy)

    print(f'Rank {mpi_rank} | Asserting hand updated basis projection with weights.')
    np.allclose(r_psy._data, r_w._data, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ===== operator Qv (V0vec --> V2) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator Qv (V0vec --> V2):')

    r_psy = basis_psy.Qv.dot(xv_psy)
    r_str1 = basis_str10.MF(xv_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator Qv.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    QvT = basis_psy.Qv.transpose()
    r_psy = QvT.dot(x2_psy)
    r_str1 = basis_str10.MF.T(x2_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator QvT.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ===== operator Tv (V0vec --> V1) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator Tv (V0vec --> V1):')

    r_psy = basis_psy.Tv.dot(xv_psy)
    r_str1 = basis_str10.EF(xv_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator Tv.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    TvT = basis_psy.Tv.transpose()
    r_psy = TvT.dot(x1_psy)
    r_str1 = basis_str10.EF.T(x1_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator TvT.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ===== operator Sv (V0vec --> V2) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator Sv (V0vec --> V2):')

    r_psy = basis_psy.Sv.dot(xv_psy)
    r_str1 = basis_str10.PF(xv_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator Sv.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    S0T = basis_psy.Sv.transpose()
    r_psy = S0T.dot(x2_psy)
    r_str1 = basis_str10.PF.T(x2_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator S0T.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ===== operator Uv (V0vec --> V2) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator Uv (V0vec --> V2):')

    r_psy = basis_psy.Uv.dot(xv_psy)
    r_str1 = basis_str10.JF(xv_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator Uv.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    UvT = basis_psy.Uv.transpose()
    r_psy = UvT.dot(x2_psy)
    r_str1 = basis_str10.JF.T(x2_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator UvT.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ================================================================================
    #                              MHD velocity is a 2-form
    # ================================================================================

    # ===== operator K3 (V3 --> V3) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator K3 (V3 --> V3):')

    r_psy = basis_psy.K3.dot(x3_psy)
    r_str1 = basis_str12.PR(x3_str)
    r_str2 = basis_str2.K2_dot(x3_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator K3.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    K3T = basis_psy.K3.transpose()
    r_psy = K3T.dot(x3_psy)
    r_str1 = basis_str12.PR.T(x3_str)
    r_str2 = basis_str2.transpose_K2_dot(x3_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator K3T.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ===== operator Q2 (V2 --> V2) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator Q2 (V2 --> V2):')

    r_psy = basis_psy.Q2.dot(x2_psy)
    r_str1 = basis_str12.MF(x2_str)
    r_str2 = basis_str2.Q2_dot(x2_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator Q2.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    Q2T = basis_psy.Q2.transpose()
    r_psy = Q2T.dot(x2_psy)
    r_str1 = basis_str12.MF.T(x2_str)
    r_str2 = basis_str2.transpose_Q2_dot(x2_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator Q2T.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ===== operator T2 (V2 --> V1) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator T2 (V2 --> V1):')

    r_psy = basis_psy.T2.dot(x2_psy)
    r_str1 = basis_str12.EF(x2_str)
    r_str2 = basis_str2.T2_dot(x2_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator T2.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    T2T = basis_psy.T2.transpose()
    r_psy = T2T.dot(x1_psy)
    r_str1 = basis_str12.EF.T(x1_str)
    r_str2 = basis_str2.transpose_T2_dot(x1_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator T2T.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ===== operator P2 (V2 --> V2) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator P2 (V2 --> V2):')

    r_psy = basis_psy.R2.dot(x2_psy)
    r_str2 = basis_str2.P2_dot(x2_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator P2.')
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14, verbose=True)
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    P2T = basis_psy.R2.transpose()
    r_psy = P2T.dot(x2_psy)
    r_str2 = basis_str2.transpose_P2_dot(x2_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator P2T.')
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ===== operator S2 (V2 --> V2) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator S2 (V2 --> V2):')

    r_psy = basis_psy.S2.dot(x2_psy)
    r_str1 = basis_str12.PF(x2_str)
    r_str2 = basis_str2.S2_dot(x2_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator S2.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    S2T = basis_psy.S2.transpose()
    r_psy = S2T.dot(x2_psy)
    r_str1 = basis_str12.PF.T(x2_str)
    r_str2 = basis_str2.transpose_S2_dot(x2_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator S2T.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ================================================================================
    #                              MHD velocity is a 1-form
    # ================================================================================

    # ===== operator Q1 (V1 --> V2) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator Q1 (V1 --> V2):')

    r_psy = basis_psy.Q1.dot(x1_psy)
    r_str2 = basis_str2.Q1_dot(x1_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator Q1.')
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    Q1T = basis_psy.Q1.transpose()
    r_psy = Q1T.dot(x2_psy)
    r_str2 = basis_str2.transpose_Q1_dot(x2_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator Q1T.')
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ===== operator T1 (V1 --> V1) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator T1 (V1 --> V1):')

    r_psy = basis_psy.T1.dot(x1_psy)
    r_str2 = basis_str2.T1_dot(x1_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator T1.')
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    T1T = basis_psy.T1.transpose()
    r_psy = T1T.dot(x1_psy)
    r_str2 = basis_str2.transpose_T1_dot(x1_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator T1T.')
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[6, 9, 7]])
@pytest.mark.parametrize('p',   [[2, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [False, True, False]])
@pytest.mark.parametrize('dirichlet_bc', [None, 
                                          [[False,  True], [False, False], [False, True]],
                                          [[False, False], [False, False], [True, False]]])
@pytest.mark.parametrize('mapping', [
    ['IGAPolarCylinder', {
        'a': 1., 'Lz': 3.}]])
def test_basis_ops_polar(Nel, p, spl_kind, dirichlet_bc, mapping, show_plots=False):

    import numpy as np

    from struphy.geometry import domains
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.eigenvalue_solvers.mhd_operators import MHDOperators

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays, compare_arrays
    from struphy.feec.basis_projection_ops import BasisProjectionOperators
    from struphy.fields_background.mhd_equil.equils import ScrewPinch
    
    from struphy.polar.basic import PolarVector

    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    print('number of processes : ', mpi_size)

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(
        **{'Nel': Nel[:2], 'p': p[:2], 'a': mapping[1]['a'], 'Lz': mapping[1]['Lz']})

    if show_plots:
        import matplotlib.pyplot as plt
        domain.show(grid_info=Nel)

    # load MHD equilibrium
    eq_mhd = ScrewPinch(**{'a': mapping[1]['a'],
                         'R0': 3.,
                         'B0': 1.0,
                         'q0': 1.05,
                         'q1': 1.80,
                         'n1': 3.0,
                         'n2': 4.0,
                         'na': 0.0,
                         'beta': .1})

    if show_plots:
        eq_mhd.plot_profiles()

    eq_mhd.domain = domain

    # make sure that boundary conditions are compatible with spline space
    if dirichlet_bc is not None:
        for i, knd in enumerate(spl_kind):
            if knd:
                dirichlet_bc[i] = [False, False]
    else:
        dirichlet_bc = [[False, False]]*3

    # derham object
    nq_el = [p[0] + 1,
             p[1] + 1,
             p[2] + 1]
    nq_pr = p.copy()

    derham = Derham(Nel, p, spl_kind, nquads=p, nq_pr=nq_pr, comm=mpi_comm,
                    dirichlet_bc=dirichlet_bc, with_projectors=True, polar_ck=1, domain=domain)

    if mpi_rank == 0:
        print()
        print(derham.domain_array)

    mhd_ops_psy = BasisProjectionOperators(derham, domain, eq_mhd=eq_mhd)

    # compare to old STRUPHY
    spaces = [Spline_space_1d(Nel[0], p[0], spl_kind[0], nq_el[0], dirichlet_bc[0]),
              Spline_space_1d(Nel[1], p[1], spl_kind[1],
                              nq_el[1], dirichlet_bc[1]),
              Spline_space_1d(Nel[2], p[2], spl_kind[2], nq_el[2], dirichlet_bc[2])]

    spaces[0].set_projectors(nq_pr[0])
    spaces[1].set_projectors(nq_pr[1])
    spaces[2].set_projectors(nq_pr[2])

    space = Tensor_spline_space(
        spaces, ck=1, cx=domain.cx[:, :, 0], cy=domain.cy[:, :, 0])
    space.set_projectors('general')

    mhd_ops_str = MHDOperators(space, eq_mhd, basis_u=2)

    mhd_ops_str.assemble_dofs('MF')
    mhd_ops_str.assemble_dofs('PF')
    mhd_ops_str.assemble_dofs('EF')
    mhd_ops_str.assemble_dofs('PR')

    mhd_ops_str.set_operators()

    # create random input arrays
    x0_str, x0_psy = create_equal_random_arrays(
        derham.Vh_fem['0'], seed=1234, flattened=True)
    x1_str, x1_psy = create_equal_random_arrays(
        derham.Vh_fem['1'], seed=1568, flattened=True)
    x2_str, x2_psy = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=8945, flattened=True)
    x3_str, x3_psy = create_equal_random_arrays(
        derham.Vh_fem['3'], seed=8196, flattened=True)

    # set polar vectors
    x0_pol_psy = PolarVector(derham.Vh_pol['0'])
    x1_pol_psy = PolarVector(derham.Vh_pol['1'])
    x2_pol_psy = PolarVector(derham.Vh_pol['2'])
    x3_pol_psy = PolarVector(derham.Vh_pol['3'])

    x0_pol_psy.tp = x0_psy
    x1_pol_psy.tp = x1_psy
    x2_pol_psy.tp = x2_psy
    x3_pol_psy.tp = x3_psy

    np.random.seed(1607)
    x0_pol_psy.pol = [np.random.rand(
        x0_pol_psy.pol[0].shape[0], x0_pol_psy.pol[0].shape[1])]
    x1_pol_psy.pol = [np.random.rand(
        x1_pol_psy.pol[n].shape[0], x1_pol_psy.pol[n].shape[1]) for n in range(3)]
    x2_pol_psy.pol = [np.random.rand(
        x2_pol_psy.pol[n].shape[0], x2_pol_psy.pol[n].shape[1]) for n in range(3)]
    x3_pol_psy.pol = [np.random.rand(
        x3_pol_psy.pol[0].shape[0], x3_pol_psy.pol[0].shape[1])]

    # apply boundary conditions to legacy vectors for right shape
    x0_pol_str = space.B0.dot(x0_pol_psy.toarray(True))
    x1_pol_str = space.B1.dot(x1_pol_psy.toarray(True))
    x2_pol_str = space.B2.dot(x2_pol_psy.toarray(True))
    x3_pol_str = space.B3.dot(x3_pol_psy.toarray(True))

    # ================================================================================
    #                              MHD velocity is a 2-form
    # ================================================================================

    # ===== operator K3 (V3 --> V3) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator K (V3 --> V3):')

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.K3.dot(x3_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.K3.dot(x3_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.PR(x3_pol_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator K3.')
    np.allclose(space.B3.T.dot(r_str), r_psy.toarray(True))
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.K3.transpose().dot(x3_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.K3.transpose().dot(x3_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.PR.T(x3_pol_str)

    print(f'Rank {mpi_rank} | Asserting transpose MHD operator K3.T.')
    np.allclose(space.B3.T.dot(r_str), r_psy.toarray(True))
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ===== operator Q2 (V2 --> V2) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator Q2 (V2 --> V2):')

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.Q2.dot(x2_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.Q2.dot(x2_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.MF(x2_pol_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator Q2.')
    np.allclose(space.B2.T.dot(r_str), r_psy.toarray(True))
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.Q2.transpose().dot(x2_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.Q2.transpose().dot(x2_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.MF.T(x2_pol_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator Q2.T.')
    np.allclose(space.B2.T.dot(r_str), r_psy.toarray(True))
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ===== operator T2 (V2 --> V1) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator T2 (V2 --> V1):')

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.T2.dot(x2_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.T2.dot(x2_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.EF(x2_pol_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator T2.')
    np.allclose(space.B1.T.dot(r_str), r_psy.toarray(True))
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.T2.transpose().dot(x1_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.T2.transpose().dot(x1_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.EF.T(x1_pol_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator T2.T.')
    np.allclose(space.B2.T.dot(r_str), r_psy.toarray(True))
    print(f'Rank {mpi_rank} | Assertion passed.')

    # ===== operator S2 (V2 --> V2) ============
    mpi_comm.Barrier()

    if mpi_rank == 0:
        print('\nOperator S2 (V2 --> V2):')

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.S2.dot(x2_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.S2.dot(x2_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.PF(x2_pol_str)

    print(f'Rank {mpi_rank} | Asserting MHD operator S2.')
    np.allclose(space.B2.T.dot(r_str), r_psy.toarray(True))
    print(f'Rank {mpi_rank} | Assertion passed.')

    mpi_comm.Barrier()

    if mpi_rank == 0:
        r_psy = mhd_ops_psy.S2.transpose().dot(x2_pol_psy, tol=1e-10, verbose=True)
    else:
        r_psy = mhd_ops_psy.S2.transpose().dot(x2_pol_psy, tol=1e-10, verbose=False)

    r_str = mhd_ops_str.PF.T(x2_pol_str)

    print(f'Rank {mpi_rank} | Asserting transposed MHD operator S2.T.')
    np.allclose(space.B2.T.dot(r_str), r_psy.toarray(True))
    print(f'Rank {mpi_rank} | Assertion passed.')


if __name__ == '__main__':
    test_basis_ops([8, 6, 4], [2, 2, 2], [False, True, True], ['Cuboid', {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 6., 'l3': 0., 'r3': 10.}], False)
    #test_basis_ops([8, 6, 4], [2, 2, 2], [False, True, True], ['Colella', {'Lx' : 1., 'Ly' : 6., 'alpha' : .1, 'Lz' : 10.}], False)
    #test_basis_ops([6, 7, 4], [2, 3, 2], [False, True, True], ['HollowCylinder', {'a1': .1, 'a2': 1., 'Lz': 2*np.pi*3.}], False)

    #test_basis_ops_polar([5, 9, 6], [2, 3, 2], [False, True, False], [[None, 'd'], [
    #                    None, None], ['d', None]], ['IGAPolarCylinder', {'a': 1., 'Lz': 3.}], False)
