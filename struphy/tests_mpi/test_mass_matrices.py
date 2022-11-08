import pytest

@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[6, 7, 4], [5, 6, 7]])
@pytest.mark.parametrize('p',   [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True]])
@pytest.mark.parametrize('bc', [[[None, None], [None, None], [None, None]],
                                [[None,  'd'], ['d' , None], [None, None]], 
                                [['d' , None], [None,  'd'], [None, None]]])
@pytest.mark.parametrize('mapping', [
    ['Colella', {
        'Lx' : 1., 'Ly' : 6., 'alpha' : .1, 'Lz' : 10.}]])
def test_mass(Nel, p, spl_kind, bc, mapping, show_plots=False):
    
    import numpy as np
    
    from struphy.geometry import domains
    from struphy.feec.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.feec.projectors.pro_global.mhd_operators_cc_lin_6d import MHDOperators
    
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays, apply_essential_bc_to_array
    from struphy.psydac_api.mass import WeightedMassOperators
    from struphy.fields_background.mhd_equil.analytical import ShearedSlab, ScrewPinch
    
    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    
    print('number of processes : ', mpi_size)
    
    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(mapping[1])
    
    if show_plots:
        import matplotlib.pyplot as plt
        domain.show()
    
    # load MHD equilibrium
    if   mapping[0] == 'Cuboid':
        eq_mhd = ShearedSlab({'a'   : (mapping[1]['r1'] - mapping[1]['l1']), 
                              'R0'  : (mapping[1]['r3'] - mapping[1]['l3'])/(2*np.pi), 
                              'B0'  : 1.0, 'q0': 1.05, 
                              'q1'  : 1.8, 'n1': 3.0, 
                              'n2'  : 4.0, 'na': 0.0, 
                              'beta': 10.0})
            
    elif mapping[0] == 'Colella':
        eq_mhd = ShearedSlab({'a'   : mapping[1]['Lx'], 
                              'R0'  : mapping[1]['Lz']/(2*np.pi), 
                              'B0'  : 1.0, 
                              'q0'  : 1.05, 
                              'q1'  : 1.8, 
                              'n1'  : 3.0, 
                              'n2'  : 4.0, 
                              'na'  : 0.0, 
                              'beta': 10.0})
        
        if show_plots:
            eq_mhd.plot_profiles()
        
    elif mapping[0] == 'HollowCylinder':
        eq_mhd = ScrewPinch({'a'   : mapping[1]['a2'], 
                             'R0'  : mapping[1]['R0'], 
                             'B0'  : 1.0, 
                             'q0'  : 1.05, 
                             'q1'  : 1.8, 
                             'n1'  : 3.0, 
                             'n2'  : 4.0, 
                             'na'  : 0.0, 
                             'beta': 10.0})
        
        if show_plots:
            eq_mhd.plot_profiles()
            
    eq_mhd.domain = domain
    
    # make sure that boundary conditions are compatible with spline space
    bc_compatible = []
    
    for spl_i, bc_i in zip(spl_kind, bc):
        if spl_i:
            bc_compatible += [[None, None]]
        else:
            bc_compatible += [bc_i]
        

    # derham object
    derham = Derham(Nel, p, spl_kind, der_as_mat=True, comm=mpi_comm, bc=bc_compatible)
    
    # mass matrices object
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)
    
    # compare to old STRUPHY
    spaces = [Spline_space_1d(Nel, p, spl, nq_el, bc) for Nel, p, spl, nq_el, bc in zip(Nel, p, spl_kind, [p[0] + 1, p[1] + 1, p[2] + 1], bc_compatible)]
    
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
    
    mhd_ops_str.set_operators()
    
    # create random input arrays and set boundary conditions
    x0_str, x0_psy = create_equal_random_arrays(derham.V0, seed=1234, flattened=True)
    x1_str, x1_psy = create_equal_random_arrays(derham.V1, seed=1568, flattened=True)
    x2_str, x2_psy = create_equal_random_arrays(derham.V2, seed=8945, flattened=True)
    x3_str, x3_psy = create_equal_random_arrays(derham.V3, seed=8196, flattened=True)
    xv_str, xv_psy = create_equal_random_arrays(derham.V0vec, seed=2038, flattened=True)
    
    apply_essential_bc_to_array('H1'   , x0_psy, derham.bc)
    apply_essential_bc_to_array('Hcurl', x1_psy, derham.bc)
    apply_essential_bc_to_array('Hdiv' , x2_psy, derham.bc)
    apply_essential_bc_to_array('L2'   , x3_psy, derham.bc)
    apply_essential_bc_to_array('H1vec', xv_psy, derham.bc)
    
    x0_str = space.B0.dot(x0_str)
    x1_str = space.B1.dot(x1_str)
    x2_str = space.B2.dot(x2_str)
    x3_str = space.B3.dot(x3_str)
    xv_str = space.Bv.dot(xv_str)
    
    # perfrom matrix-vector products
    r0_str = space.B0.T.dot(space.M0_0(x0_str))
    r1_str = space.B1.T.dot(space.M1_0(x1_str))
    r2_str = space.B2.T.dot(space.M2_0(x2_str))
    r3_str = space.B3.T.dot(space.M3_0(x3_str))
    rv_str = space.Bv.T.dot(space.Mv_0(xv_str))
    
    rn_str = space.B2.T.dot(mhd_ops_str.Mn(x2_str))
    rJ_str = space.B2.T.dot(mhd_ops_str.MJ(x2_str))
    
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
    
    
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 12, 6], [6, 15, 4]])
@pytest.mark.parametrize('p',   [[2, 2, 3], [4, 3, 2]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [False, True, False]])
@pytest.mark.parametrize('bc', [[[None,  'd'], [None, None], [None, ' d']],
                                [[None, None], [None, None], ['d', None]]])
@pytest.mark.parametrize('mapping', [
    ['PoloidalSplineCylinder', {
        'a' : 1., 'R0' : 3.}]])
def test_mass_polar(Nel, p, spl_kind, bc, mapping, show_plots=False):
    
    import numpy as np
    
    from struphy.geometry import domains
    from struphy.feec.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.feec.projectors.pro_global.mhd_operators_cc_lin_6d import MHDOperators
    
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays, apply_essential_bc_to_array, apply_essential_bc_to_pol
    from struphy.psydac_api.mass import WeightedMassOperators
    from struphy.fields_background.mhd_equil.analytical import ScrewPinch
    
    from struphy.polar.basic import PolarVector
    
    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    
    print('number of processes : ', mpi_size)
    
    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class({'Nel' : Nel[:2], 'p' : p[:2], 'spl_kind' : spl_kind[:2], 'a' : mapping[1]['a'], 'R0' : mapping[1]['R0']})
    
    if show_plots:
        import matplotlib.pyplot as plt
        domain.show(grid_info=Nel)
    
    # load MHD equilibrium
    eq_mhd = ScrewPinch({'a'   : mapping[1]['a'], 
                         'R0'  : mapping[1]['R0'], 
                         'B0'  : 1.0, 
                         'q0'  : 1.05, 
                         'q1'  : 1.8, 
                         'n1'  : 3.0, 
                         'n2'  : 4.0, 
                         'na'  : 0.0, 
                         'beta': 10.0})

    if show_plots:
        eq_mhd.plot_profiles()
            
    eq_mhd.domain = domain
    
    # make sure that boundary conditions are compatible with spline space
    bc_compatible = []
    
    for spl_i, bc_i in zip(spl_kind, bc):
        if spl_i:
            bc_compatible += [[None, None]]
        else:
            bc_compatible += [bc_i]
            
    
    # derham object
    derham = Derham(Nel, p, spl_kind, der_as_mat=True, comm=mpi_comm, bc=bc_compatible, polar_ck=1, domain=domain)
    
    # mass matrices object
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)
    
    # compare to old STRUPHY
    spaces = [Spline_space_1d(Nel, p, spl, nq_el, bc) for Nel, p, spl, nq_el, bc in zip(Nel, p, spl_kind, [p[0] + 1, p[1] + 1, p[2] + 1], bc_compatible)]
    
    spaces[0].set_projectors()
    spaces[1].set_projectors()
    spaces[2].set_projectors()

    space = Tensor_spline_space(spaces, ck=1, cx=domain.cx[:, :, 0], cy=domain.cy[:, :, 0])
    space.set_projectors('general')

    space.assemble_Mk(domain, 'V0')
    space.assemble_Mk(domain, 'V1')
    space.assemble_Mk(domain, 'V2')
    space.assemble_Mk(domain, 'V3')
    
    mhd_ops_str = MHDOperators(space, eq_mhd, 2)
    
    mhd_ops_str.assemble_Mn()
    mhd_ops_str.assemble_MJ()
    
    mhd_ops_str.set_operators()
    
    # create random input arrays and set boundary conditions
    x0_str, x0_psy = create_equal_random_arrays(derham.V0, seed=1234, flattened=True)
    x1_str, x1_psy = create_equal_random_arrays(derham.V1, seed=1568, flattened=True)
    x2_str, x2_psy = create_equal_random_arrays(derham.V2, seed=8945, flattened=True)
    x3_str, x3_psy = create_equal_random_arrays(derham.V3, seed=8196, flattened=True)
    
    apply_essential_bc_to_array('H1'   , x0_psy, derham.bc)
    apply_essential_bc_to_array('Hcurl', x1_psy, derham.bc)
    apply_essential_bc_to_array('Hdiv' , x2_psy, derham.bc)
    apply_essential_bc_to_array('L2'   , x3_psy, derham.bc)
    
    x0_pol_psy = PolarVector(derham.V0_pol)
    x1_pol_psy = PolarVector(derham.V1_pol)
    x2_pol_psy = PolarVector(derham.V2_pol)
    x3_pol_psy = PolarVector(derham.V3_pol)

    # set polar vectors
    x0_pol_psy.tp = x0_psy
    x1_pol_psy.tp = x1_psy
    x2_pol_psy.tp = x2_psy
    x3_pol_psy.tp = x3_psy

    np.random.seed(1607)
    x0_pol_psy.pol = [np.random.rand(x0_pol_psy.pol[0].shape[0], x0_pol_psy.pol[0].shape[1])]
    x1_pol_psy.pol = [np.random.rand(x1_pol_psy.pol[n].shape[0], x1_pol_psy.pol[n].shape[1]) for n in range(3)]
    x2_pol_psy.pol = [np.random.rand(x2_pol_psy.pol[n].shape[0], x2_pol_psy.pol[n].shape[1]) for n in range(3)]
    x3_pol_psy.pol = [np.random.rand(x3_pol_psy.pol[0].shape[0], x3_pol_psy.pol[0].shape[1])]
    
    apply_essential_bc_to_pol('H1'   , x0_pol_psy.pol, derham.bc[2])
    apply_essential_bc_to_pol('Hcurl', x1_pol_psy.pol, derham.bc[2])
    apply_essential_bc_to_pol('Hdiv' , x2_pol_psy.pol, derham.bc[2])
    apply_essential_bc_to_pol('L2'   , x3_pol_psy.pol, derham.bc[2])

    x0_pol_str = space.B0.dot(x0_pol_psy.toarray(True))
    x1_pol_str = space.B1.dot(x1_pol_psy.toarray(True))
    x2_pol_str = space.B2.dot(x2_pol_psy.toarray(True))
    x3_pol_str = space.B3.dot(x3_pol_psy.toarray(True))
    
    # perfrom matrix-vector products
    r0_pol_str = space.B0.T.dot(space.M0_0(x0_pol_str))
    r1_pol_str = space.B1.T.dot(space.M1_0(x1_pol_str))
    r2_pol_str = space.B2.T.dot(space.M2_0(x2_pol_str))
    r3_pol_str = space.B3.T.dot(space.M3_0(x3_pol_str))
    
    rn_pol_str = space.B2.T.dot(mhd_ops_str.Mn(x2_pol_str))
    rJ_pol_str = space.B2.T.dot(mhd_ops_str.MJ(x2_pol_str))
    
    r0_pol_psy = mass_mats.M0.dot(x0_pol_psy)
    r1_pol_psy = mass_mats.M1.dot(x1_pol_psy)
    r2_pol_psy = mass_mats.M2.dot(x2_pol_psy)
    r3_pol_psy = mass_mats.M3.dot(x3_pol_psy)
    
    rn_pol_psy = mass_mats.M2n.dot(x2_pol_psy)
    rJ_pol_psy = mass_mats.M2J.dot(x2_pol_psy)

    assert np.allclose(r0_pol_str, r0_pol_psy.toarray(True))
    assert np.allclose(r1_pol_str, r1_pol_psy.toarray(True))
    assert np.allclose(r2_pol_str, r2_pol_psy.toarray(True))
    assert np.allclose(r3_pol_str, r3_pol_psy.toarray(True))
    assert np.allclose(rn_pol_str, rn_pol_psy.toarray(True))
    assert np.allclose(rJ_pol_str, rJ_pol_psy.toarray(True))
    
if __name__ == '__main__':
    #test_mass([8, 6, 4], [2, 2, 2], [False, True, True], [['d', 'd'], [None, None], [None, None]], ['Cuboid', {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 6., 'l3': 0., 'r3': 10.}], True)
    #test_mass([8, 6, 4], [2, 2, 2], [False, True, True], [['d', 'd'], [None, None], [None, None]], ['Colella', {'Lx' : 1., 'Ly' : 6., 'alpha' : .1, 'Lz' : 10.}], True)
    #test_mass([8, 6, 4], [2, 2, 2], [False, True, True], [['d', 'd'], [None, None], [None, None]], ['HollowCylinder', {'a1': .1, 'a2': 1., 'R0': 3., 'Lz': 10.}], True)
    
    test_mass_polar([8, 12, 6], [4, 3, 2], [False, True, False], [[None, 'd'], [None, None], ['d', None]], ['PoloidalSplineCylinder', {'a': 1., 'R0': 3.}], False)