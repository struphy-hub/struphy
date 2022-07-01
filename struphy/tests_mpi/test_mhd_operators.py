import pytest
import numpy as np


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[6, 7, 4], [5, 6, 7]])
@pytest.mark.parametrize('p',   [[2, 3, 2], [4, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True]])
@pytest.mark.parametrize('mapping', [
    #['cuboid', {
    #    'l1': 0., 'r1': 1., 'l2': 0., 'r2': 6., 'l3': 0., 'r3': 10.}],
    ['colella', {
        'Lx' : 1., 'Ly' : 6., 'alpha' : .1, 'Lz' : 10.}],
    ['hollow_cyl', {
        'a1': .1, 'a2': 1., 'R0': 3., 'lz': 2*np.pi*3.}]
        ])
def test_mhd_ops(Nel, p, spl_kind, mapping, show_plots=False):
    
    import numpy as np
    
    from struphy.geometry.domain_3d import Domain
    from struphy.fields_equil.mhd_equil.analytical import EquilibriumMHDShearedSlab, EquilibriumMHDCylinder
    from struphy.feec.spline_space import Spline_space_1d, Tensor_spline_space
    
    import struphy.feec.projectors.pro_global.mhd_operators_cc_lin_6d as mhd_ops_str1
    import struphy.feec.projectors.pro_global.mhd_operators_MF as mhd_ops_str2
    
    from struphy.psydac_api.psydac_derham import Derham
    import struphy.psydac_api.mhd_ops_pure_psydac as mhd_ops_psy
    
    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays
    
    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    
    print('number of processes : ', mpi_size)
    
    # mapping
    domain = Domain(mapping[0], mapping[1])
    F = domain.Psydac_mapping('F', **mapping[1])
    
    if show_plots:
        import matplotlib.pyplot as plt
        domain.show()
    
    # MHD equilibrium
    if   mapping[0] == 'cuboid':
        eq_mhd = EquilibriumMHDShearedSlab({'a': mapping[1]['r1'] - mapping[1]['l1'], 'R0': (mapping[1]['r3'] - mapping[1]['l3'])/(2*np.pi), 'B0': 1.0, 'q0': 1.05, 'q1': 1.8, 'n1': 3.0, 'n2': 4.0, 'na': 0.0, 'beta': 10.0}, domain)
            
    elif mapping[0] == 'colella':
        eq_mhd = EquilibriumMHDShearedSlab({'a': mapping[1]['Lx'], 'R0': mapping[1]['Lz']/(2*np.pi), 'B0': 1.0, 'q0': 1.05, 'q1': 1.8, 'n1': 3.0, 'n2': 4.0, 'na': 0.0, 'beta': 10.0}, domain)
        
        if show_plots:
            eq_mhd.plot_profiles()
        
    elif mapping[0] == 'hollow_cyl':
        eq_mhd = EquilibriumMHDCylinder({'a': mapping[1]['a2'], 'R0': mapping[1]['R0'], 'B0': 1.0, 'q0': 1.05, 'q1': 1.8, 'n1': 3.0, 'n2': 4.0, 'na': 0.0, 'beta': 10.0}, domain)
        
        if show_plots:
            eq_mhd.plot_profiles()

    # Psydac derham object
    nq_el = [p[0] + 1, p[1] + 1, p[2] + 1]
    nq_pr = p.copy()
    
    derham = Derham(Nel, p, spl_kind, nq_pr, quad_order=p, der_as_mat=True, F=F, comm=mpi_comm)
    
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
    mhd_psy = mhd_ops_psy.MHDOperators(derham, F.get_callable_mapping(), eq_mhd)
    
    mhd_str10 = mhd_ops_str1.MHDOperators(space_str1, eq_mhd, basis_u=0) # MHD velocity is 0-form^3
    mhd_str12 = mhd_ops_str1.MHDOperators(space_str1, eq_mhd, basis_u=2) # MHD velocity is 2-form
    
    mhd_str2 = mhd_ops_str2.projectors_dot_x(space_str2, eq_mhd)
    
    # assemble matrix operators
    mhd_psy.assemble_K0()
    mhd_psy.assemble_Q0()
    mhd_psy.assemble_T0()
    mhd_psy.assemble_S0()
    mhd_psy.assemble_J0()
    
    mhd_psy.assemble_K2()
    mhd_psy.assemble_Q2()
    mhd_psy.assemble_T2()
    mhd_psy.assemble_P2()
    mhd_psy.assemble_S2()
    
    mhd_psy.assemble_Q1()
    mhd_psy.assemble_T1()
    
    mhd_str10.assemble_dofs('MF')
    mhd_str10.assemble_dofs('PF')
    mhd_str10.assemble_dofs('JF')
    mhd_str10.assemble_dofs('EF')
    mhd_str10.assemble_dofs('PR')
    mhd_str10.set_operators()
    
    mhd_str12.assemble_dofs('MF')
    mhd_str12.assemble_dofs('PF')
    mhd_str12.assemble_dofs('EF')
    mhd_str12.assemble_dofs('PR')
    mhd_str12.set_operators()
    
    
    # create random input arrays
    x0_str, x0_psy = create_equal_random_arrays(derham.V0, 1234, flattened=True)
    x1_str, x1_psy = create_equal_random_arrays(derham.V1, 1568, flattened=True)
    x2_str, x2_psy = create_equal_random_arrays(derham.V2, 8945, flattened=True)
    x3_str, x3_psy = create_equal_random_arrays(derham.V3, 8196, flattened=True)
    xv_str, xv_psy = create_equal_random_arrays(derham.V0vec, 2038, flattened=True)
    
    
    # compare matrix-vector products of different methods
    
    # ================================================================================
    #                              MHD velocity is a 0-form^3
    # ================================================================================
    
    # ===== operator K0 (V3 --> V3) ============
    mpi_comm.Barrier()
    
    if mpi_rank == 0:
        print('\nOperator K0 (V3 --> V3):')
    
    r_psy = mhd_psy.K0.dot(x3_psy)
    r_str1 = mhd_str10.PR(x3_str)
    
    print(f'Rank {mpi_rank} | Asserting MHD operator K0.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    mpi_comm.Barrier()
    
    r_psy = mhd_psy.K0T.dot(x3_psy)
    r_str1 = mhd_str10.PR.T(x3_str)
    
    print(f'Rank {mpi_rank} | Asserting transposed MHD operator K0T.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    # ===== operator Q0 (V0vec --> V2) ============
    mpi_comm.Barrier()
    
    if mpi_rank == 0:
        print('\nOperator Q0 (V0vec --> V2):')
    
    r_psy = mhd_psy.Q0.dot(xv_psy)
    r_str1 = mhd_str10.MF(xv_str)
    
    print(f'Rank {mpi_rank} | Asserting MHD operator Q0.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    mpi_comm.Barrier()
    
    r_psy = mhd_psy.Q0T.dot(x2_psy)
    r_str1 = mhd_str10.MF.T(x2_str)
    
    print(f'Rank {mpi_rank} | Asserting transposed MHD operator Q0T.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    # ===== operator T0 (V0vec --> V1) ============
    mpi_comm.Barrier()
    
    if mpi_rank == 0:
        print('\nOperator T0 (V0vec --> V1):')
    
    r_psy = mhd_psy.T0.dot(xv_psy)
    r_str1 = mhd_str10.EF(xv_str)
    
    print(f'Rank {mpi_rank} | Asserting MHD operator T0.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    mpi_comm.Barrier()
    
    r_psy = mhd_psy.T0T.dot(x1_psy)
    r_str1 = mhd_str10.EF.T(x1_str)
    
    print(f'Rank {mpi_rank} | Asserting transposed MHD operator T0T.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    # ===== operator S0 (V0vec --> V2) ============
    mpi_comm.Barrier()
    
    if mpi_rank == 0:
        print('\nOperator S0 (V0vec --> V2):')
    
    r_psy = mhd_psy.S0.dot(xv_psy)
    r_str1 = mhd_str10.PF(xv_str)
    
    print(f'Rank {mpi_rank} | Asserting MHD operator S0.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    mpi_comm.Barrier()
    
    r_psy = mhd_psy.S0T.dot(x2_psy)
    r_str1 = mhd_str10.PF.T(x2_str)
    
    print(f'Rank {mpi_rank} | Asserting transposed MHD operator S0T.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    # ===== operator J0 (V0vec --> V2) ============
    mpi_comm.Barrier()
    
    if mpi_rank == 0:
        print('\nOperator J0 (V0vec --> V2):')
    
    r_psy = mhd_psy.J0.dot(xv_psy)
    r_str1 = mhd_str10.JF(xv_str)
    
    print(f'Rank {mpi_rank} | Asserting MHD operator J0.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    mpi_comm.Barrier()
    
    r_psy = mhd_psy.J0T.dot(x2_psy)
    r_str1 = mhd_str10.JF.T(x2_str)
    
    print(f'Rank {mpi_rank} | Asserting transposed MHD operator J0T.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    # ================================================================================
    #                              MHD velocity is a 2-form
    # ================================================================================
    
    # ===== operator K2 (V3 --> V3) ============
    mpi_comm.Barrier()
    
    if mpi_rank == 0:
        print('\nOperator K2 (V3 --> V3):')
    
    r_psy = mhd_psy.K2.dot(x3_psy)
    r_str1 = mhd_str12.PR(x3_str)
    r_str2 = mhd_str2.K2_dot(x3_str)
    
    print(f'Rank {mpi_rank} | Asserting MHD operator K2.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    mpi_comm.Barrier()
    
    r_psy = mhd_psy.K2T.dot(x3_psy)
    r_str1 = mhd_str12.PR.T(x3_str)
    r_str2 = mhd_str2.transpose_K2_dot(x3_str)
    
    print(f'Rank {mpi_rank} | Asserting transposed MHD operator K2T.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    # ===== operator Q2 (V2 --> V2) ============
    mpi_comm.Barrier()
    
    if mpi_rank == 0:
        print('\nOperator Q2 (V2 --> V2):')
    
    r_psy = mhd_psy.Q2.dot(x2_psy)
    r_str1 = mhd_str12.MF(x2_str)
    r_str2 = mhd_str2.Q2_dot(x2_str)
    
    print(f'Rank {mpi_rank} | Asserting MHD operator Q2.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    mpi_comm.Barrier()
    
    r_psy = mhd_psy.Q2T.dot(x2_psy)
    r_str1 = mhd_str12.MF.T(x2_str)
    r_str2 = mhd_str2.transpose_Q2_dot(x2_str)
    
    print(f'Rank {mpi_rank} | Asserting transposed MHD operator Q2T.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    # ===== operator T2 (V2 --> V1) ============
    mpi_comm.Barrier()
    
    if mpi_rank == 0:
        print('\nOperator T2 (V2 --> V1):')
    
    r_psy = mhd_psy.T2.dot(x2_psy)
    r_str1 = mhd_str12.EF(x2_str)
    r_str2 = mhd_str2.T2_dot(x2_str)
    
    print(f'Rank {mpi_rank} | Asserting MHD operator T2.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    mpi_comm.Barrier()
    
    r_psy = mhd_psy.T2T.dot(x1_psy)
    r_str1 = mhd_str12.EF.T(x1_str)
    r_str2 = mhd_str2.transpose_T2_dot(x1_str)
    
    print(f'Rank {mpi_rank} | Asserting transposed MHD operator T2T.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    # ===== operator P2 (V2 --> V2) ============
    mpi_comm.Barrier()
    
    if mpi_rank == 0:
        print('\nOperator P2 (V2 --> V2):')
    
    r_psy = mhd_psy.P2.dot(x2_psy)
    r_str2 = mhd_str2.P2_dot(x2_str)
    
    print(f'Rank {mpi_rank} | Asserting MHD operator P2.')
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    mpi_comm.Barrier()
    
    r_psy = mhd_psy.P2T.dot(x2_psy)
    r_str2 = mhd_str2.transpose_P2_dot(x2_str)
    
    print(f'Rank {mpi_rank} | Asserting transposed MHD operator P2T.')
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    # ===== operator S2 (V2 --> V2) ============
    mpi_comm.Barrier()
    
    if mpi_rank == 0:
        print('\nOperator S2 (V2 --> V2):')
    
    r_psy = mhd_psy.S2.dot(x2_psy)
    r_str1 = mhd_str12.PF(x2_str)
    r_str2 = mhd_str2.S2_dot(x2_str)
    
    print(f'Rank {mpi_rank} | Asserting MHD operator S2.')
    compare_arrays(r_psy, r_str1, mpi_rank, atol=1e-14)
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    mpi_comm.Barrier()
    
    r_psy = mhd_psy.S2T.dot(x2_psy)
    r_str1 = mhd_str12.PF.T(x2_str)
    r_str2 = mhd_str2.transpose_S2_dot(x2_str)
    
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
    
    r_psy = mhd_psy.Q1.dot(x1_psy)
    r_str2 = mhd_str2.Q1_dot(x1_str)
    
    print(f'Rank {mpi_rank} | Asserting MHD operator Q1.')
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    mpi_comm.Barrier()
    
    r_psy = mhd_psy.Q1T.dot(x2_psy)
    r_str2 = mhd_str2.transpose_Q1_dot(x2_str)
    
    print(f'Rank {mpi_rank} | Asserting transposed MHD operator Q1T.')
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    # ===== operator T1 (V1 --> V1) ============
    mpi_comm.Barrier()
    
    if mpi_rank == 0:
        print('\nOperator T1 (V1 --> V1):')
    
    r_psy = mhd_psy.T1.dot(x1_psy)
    r_str2 = mhd_str2.T1_dot(x1_str)
    
    print(f'Rank {mpi_rank} | Asserting MHD operator T1.')
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
    mpi_comm.Barrier()
    
    r_psy = mhd_psy.T1T.dot(x1_psy)
    r_str2 = mhd_str2.transpose_T1_dot(x1_str)
    
    print(f'Rank {mpi_rank} | Asserting transposed MHD operator T1T.')
    compare_arrays(r_psy, r_str2, mpi_rank, atol=1e-14)
    print(f'Rank {mpi_rank} | Assertion passed.')
    
if __name__ == '__main__':
    #test_mhd_ops([8, 6, 4], [2, 2, 2], [False, True, True], ['cuboid', {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 6., 'l3': 0., 'r3': 10.}], False)
    #test_mhd_ops([8, 6, 4], [2, 2, 2], [False, True, True], ['colella', {'Lx' : 1., 'Ly' : 6., 'alpha' : .1, 'Lz' : 10.}], False)
    test_mhd_ops([6, 7, 4], [2, 3, 2], [False, True, True], ['hollow_cyl', {'a1': .1, 'a2': 1., 'R0': 3., 'lz': 2*np.pi*3.}], False)