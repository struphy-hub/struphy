import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[5, 6, 7]])
@pytest.mark.parametrize('p',   [[2, 3, 1], [1, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 6., 'l3': 0., 'r3': 10.}],
    ['Colella', {
        'Lx' : 1., 'Ly' : 6., 'alpha' : .1, 'Lz' : 10.}]])
def test_solvers(Nel, p, spl_kind, mapping, show_plots=False, verbose=False):
    
    import numpy as np
    
    from struphy.geometry import domains
    
    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays
    from struphy.psydac_api.mass_psydac import WeightedMass
    from struphy.fields_background.mhd_equil.analytical import ShearedSlab, ScrewPinch
    
    from struphy.linear_algebra.iterative_solvers import bicgstab, pbicgstab
    from struphy.psydac_api.preconditioner import MassMatrixPreconditioner
    
    from psydac.linalg.iterative_solvers import pcg
    
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
    
    # MHD equilibrium
    if   mapping[0] == 'Cuboid':
        eq_mhd = ShearedSlab({'a': mapping[1]['r1'] - mapping[1]['l1'], 'R0': (mapping[1]['r3'] - mapping[1]['l3'])/(2*np.pi), 'B0': 1.0, 'q0': 1.05, 'q1': 1.8, 'n1': 3.0, 'n2': 4.0, 'na': 0.0, 'beta': 10.0}, domain)
            
    elif mapping[0] == 'Colella':
        eq_mhd = ShearedSlab({'a': mapping[1]['Lx'], 'R0': mapping[1]['Lz']/(2*np.pi), 'B0': 1.0, 'q0': 1.05, 'q1': 1.8, 'n1': 3.0, 'n2': 4.0, 'na': 0.0, 'beta': 10.0}, domain)
        
        if show_plots:
            eq_mhd.plot_profiles()

    # derham object
    derham = Derham(Nel, p, spl_kind, der_as_mat=True, comm=mpi_comm)
    
    # mass object
    mass_mats = WeightedMass(derham, domain, eq_mhd=eq_mhd)
    
    # assemble mass matrices
    mass_mats.assemble_M0()
    mass_mats.assemble_M1()
    
    # assemble preconditioners
    pc0 = MassMatrixPreconditioner(derham, 'V0', mass_mats._fun_M0)
    pc1 = MassMatrixPreconditioner(derham, 'V1', mass_mats._fun_M1)
    
    # create random right-hand side vectors
    b0_str, b0 = create_equal_random_arrays(derham.V0, 1234)
    b1_str, b1 = create_equal_random_arrays(derham.V1, 1607)
    
    x0_bicgstab, info0_bicgstab = bicgstab(mass_mats.M0, b0)
    x0_pbicgstab, info0_pbicgstab = pbicgstab(mass_mats.M0, b0, pc0)
    x0_pcg, info0_pcg = pcg(mass_mats.M0, b0, pc0)
    
    #x1_bicgstab, info1_bicgstab = bicgstab(mass_mats.M1, b1)
    x1_pbicgstab, info1_pbicgstab = pbicgstab(mass_mats.M1, b1, pc1)
    x1_pcg, info1_pcg = pcg(mass_mats.M1, b1, pc1)
    
    assert info0_bicgstab['success']
    assert info0_pbicgstab['success']
    assert info0_pcg['success']
    
    #assert info1_bicgstab['success']
    assert info1_pbicgstab['success']
    assert info1_pcg['success']
    
    if verbose and mpi_rank == 0:
        print('info for bicgstab  (M0) : ', info0_bicgstab)
        print('info for pbicgstab (M0) : ', info0_pbicgstab)
        print('info for pcg       (M0) : ', info0_pcg)
        
        #print('info for bicgstab  (M1) : ', info1_bicgstab)
        print('info for pbicgstab (M1) : ', info1_pbicgstab)
        print('info for pcg       (M1) : ', info1_pcg)
    
    
if __name__ == '__main__':
    test_solvers([5, 6, 7], [1, 2, 3], [False, True, True], ['Cuboid', {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 6., 'l3': 0., 'r3': 10.}], False, True)
    #test_solvers([8, 6, 4], [2, 2, 2], [False, True, True], ['Colella', {'Lx' : 1., 'Ly' : 6., 'alpha' : .1, 'Lz' : 10.}], False, True)