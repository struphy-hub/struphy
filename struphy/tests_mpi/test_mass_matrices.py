import pytest


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
        'a1': .1, 'a2': 1., 'R0': 3., 'Lz': 10.}]])
def test_mass(Nel, p, spl_kind, mapping, show_plots=False):
    
    import numpy as np
    
    from struphy.geometry.domain_3d import Domain
    from struphy.feec.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.feec.projectors.pro_global.mhd_operators_cc_lin_6d import MHDOperators
    
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays
    from struphy.psydac_api.mass_psydac import WeightedMass
    from struphy.fields_background.mhd_equil.analytical import ShearedSlab, ScrewPinch
    
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
        eq_mhd = ShearedSlab({'a': mapping[1]['r1'] - mapping[1]['l1'], 'R0': (mapping[1]['r3'] - mapping[1]['l3'])/(2*np.pi), 'B0': 1.0, 'q0': 1.05, 'q1': 1.8, 'n1': 3.0, 'n2': 4.0, 'na': 0.0, 'beta': 10.0}, domain)
            
    elif mapping[0] == 'colella':
        eq_mhd = ShearedSlab({'a': mapping[1]['Lx'], 'R0': mapping[1]['Lz']/(2*np.pi), 'B0': 1.0, 'q0': 1.05, 'q1': 1.8, 'n1': 3.0, 'n2': 4.0, 'na': 0.0, 'beta': 10.0}, domain)
        
        if show_plots:
            eq_mhd.plot_profiles()
        
    elif mapping[0] == 'hollow_cyl':
        eq_mhd = ScrewPinch({'a': mapping[1]['a2'], 'R0': mapping[1]['R0'], 'B0': 1.0, 'q0': 1.05, 'q1': 1.8, 'n1': 3.0, 'n2': 4.0, 'na': 0.0, 'beta': 10.0}, domain)
        
        if show_plots:
            eq_mhd.plot_profiles()

    # derham object
    derham = Derham(Nel, p, spl_kind, der_as_mat=True, F=F, comm=mpi_comm)
    
    # mass object
    mass_mats = WeightedMass(derham, F.get_callable_mapping(), eq_mhd=eq_mhd)
    
    # assemble mass matrices
    mass_mats.assemble_M0()
    mass_mats.assemble_M1()
    mass_mats.assemble_M2()
    mass_mats.assemble_M3()
    mass_mats.assemble_Mv()
    
    mass_mats.assemble_M2n()
    mass_mats.assemble_M2J()
    #mass_mats.assemble_Mvn()
    #mass_mats.assemble_MvJ()
    
    # compare to old STRUPHY
    spaces = [Spline_space_1d(Nel, p, spl, nq_el) for Nel, p, spl, nq_el in zip(Nel, p, spl_kind, [p[0] + 1, p[1] + 1, p[2] + 1])]
    
    spaces[0].set_projectors()
    spaces[1].set_projectors()
    spaces[2].set_projectors()

    space = Tensor_spline_space(spaces)
    space.set_projectors('general')

    space.assemble_Mk(domain, 'V0')
    space.assemble_Mk(domain, 'V1')
    space.assemble_Mk(domain, 'V2')
    space.assemble_Mk(domain, 'V3')
    space.assemble_Mk(domain, 'Vv')
    
    mhd_ops_str = MHDOperators(space, eq_mhd, 2)
    
    mhd_ops_str.assemble_Mn()
    mhd_ops_str.assemble_MJ()
    
    # create random input arrays
    x0_str, x0_psy = create_equal_random_arrays(derham.V0, 1234)
    x1_str, x1_psy = create_equal_random_arrays(derham.V1, 1568)
    x2_str, x2_psy = create_equal_random_arrays(derham.V2, 8945)
    x3_str, x3_psy = create_equal_random_arrays(derham.V3, 8196)
    xv_str, xv_psy = create_equal_random_arrays(derham.V0vec, 2038)
    
    # perfrom matrix-vector products
    r0_str = space.M0(x0_str[0].flatten())
    r1_str = space.M1(np.concatenate((x1_str[0].flatten(), x1_str[1].flatten(), x1_str[2].flatten())))
    r2_str = space.M2(np.concatenate((x2_str[0].flatten(), x2_str[1].flatten(), x2_str[2].flatten())))
    r3_str = space.M3(x3_str[0].flatten())
    rv_str = space.Mv(np.concatenate((xv_str[0].flatten(), xv_str[1].flatten(), xv_str[2].flatten())))
    
    rn_str = mhd_ops_str.Mn_mat.dot(np.concatenate((x2_str[0].flatten(), x2_str[1].flatten(), x2_str[2].flatten())))
    rJ_str = mhd_ops_str.MJ_mat.dot(np.concatenate((x2_str[0].flatten(), x2_str[1].flatten(), x2_str[2].flatten())))
    
    r0_psy = mass_mats.M0.dot(x0_psy)
    r1_psy = mass_mats.M1.dot(x1_psy)
    r2_psy = mass_mats.M2.dot(x2_psy)
    r3_psy = mass_mats.M3.dot(x3_psy)
    rv_psy = mass_mats.Mv.dot(xv_psy)
    
    rn_psy = mass_mats.M2n.dot(x2_psy)
    rJ_psy = mass_mats.M2J.dot(x2_psy)
    
    # compare output arrays
    compare_arrays(r0_psy, r0_str, mpi_rank, atol=1e-14)
    compare_arrays(r1_psy, r1_str, mpi_rank, atol=1e-14)
    compare_arrays(r2_psy, r2_str, mpi_rank, atol=1e-14)
    compare_arrays(r3_psy, r3_str, mpi_rank, atol=1e-14)
    compare_arrays(rv_psy, rv_str, mpi_rank, atol=1e-14)
    
    compare_arrays(rn_psy, rn_str, mpi_rank, atol=1e-14)
    compare_arrays(rJ_psy, rJ_str, mpi_rank, atol=1e-14)
    
    
if __name__ == '__main__':
    #test_mass([8, 6, 4], [2, 2, 2], [False, True, True], ['cuboid', {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 6., 'l3': 0., 'r3': 10.}], False)
    #test_mass([8, 6, 4], [2, 2, 2], [False, True, True], ['colella', {'Lx' : 1., 'Ly' : 6., 'alpha' : .1, 'Lz' : 10.}], False)
    test_mass([8, 6, 4], [2, 2, 2], [False, True, True], ['hollow_cyl', {'a1': .1, 'a2': 1., 'R0': 3., 'Lz': 10.}], False)