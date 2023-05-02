import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[5, 6, 7]])
@pytest.mark.parametrize('p',   [[2, 3, 2]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 0., 'r1': 1., 'l2': 0., 'r2': 6., 'l3': 0., 'r3': 10.}],
    ['Colella', {
        'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}]])
def test_solvers(Nel, p, spl_kind, mapping, show_plots=False, verbose=False):

    import numpy as np

    from struphy.geometry import domains

    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays
    from struphy.psydac_api.mass import WeightedMassOperators
    from struphy.fields_background.mhd_equil.equils import ShearedSlab
    
    from struphy.linear_algebra.iterative_solvers import ConjugateGradient, PConjugateGradient, BiConjugateGradientStab, PBiConjugateGradientStab
    from struphy.psydac_api.preconditioner import MassMatrixPreconditioner
    
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space

    from psydac.linalg.iterative_solvers import pcg, cg, bicg, lsmr, minres

    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    print()
    print('number of processes : ', mpi_size)

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    if show_plots:
        import matplotlib.pyplot as plt
        domain.show()

    # MHD equilibrium
    if mapping[0] == 'Cuboid':
        eq_mhd = ShearedSlab(**{'a': mapping[1]['r1'] - mapping[1]['l1'], 'R0': (mapping[1]['r3'] - mapping[1]['l3'])/(
            2*np.pi), 'B0': 1.0, 'q0': 1.05, 'q1': 1.8, 'n1': 3.0, 'n2': 4.0, 'na': 0.0, 'beta': 10.0})

    elif mapping[0] == 'Colella':
        eq_mhd = ShearedSlab(**{'a': mapping[1]['Lx'], 'R0': mapping[1]['Lz']/(
            2*np.pi), 'B0': 1.0, 'q0': 1.05, 'q1': 1.8, 'n1': 3.0, 'n2': 4.0, 'na': 0.0, 'beta': 10.0})

        if show_plots:
            eq_mhd.plot_profiles()

    # set equilibrium object domain
    eq_mhd.domain = domain

    # derham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm)

    # mass object and preconditioners
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)
    
    M0 = mass_mats.M0
    M1 = mass_mats.M1
    pc0 = MassMatrixPreconditioner(M0)
    pc1 = MassMatrixPreconditioner(M1)
    
    # compare to legacy STRUPHY
    spaces = [Spline_space_1d(Nel[0], p[0], spl_kind[0], p[0] + 1),
              Spline_space_1d(Nel[1], p[1], spl_kind[1], p[1] + 1),
              Spline_space_1d(Nel[2], p[2], spl_kind[2], p[2] + 1)]
    
    space = Tensor_spline_space(spaces)
    
    space.assemble_Mk(domain, 'V0')
    space.assemble_Mk(domain, 'V1')
    
    M0_arr = M0.matrix.toarray()
    M1_arr = M1.matrix.toarray()
    
    mpi_comm.Allreduce(MPI.IN_PLACE, M0_arr, op=MPI.SUM)
    mpi_comm.Allreduce(MPI.IN_PLACE, M1_arr, op=MPI.SUM)
    
    assert np.allclose(M0_arr, space.M0_mat.toarray(), atol=1e-14)
    assert np.allclose(M1_arr, space.M1_mat.toarray(), atol=1e-14)
    
    M0 = M0.matrix
    M1 = M1.matrix

    # create linear solvers
    cg_solver0 = ConjugateGradient(M0.domain)
    cg_solver1 = ConjugateGradient(M1.domain)
    
    pcg_solver0 = PConjugateGradient(M0.domain)
    pcg_solver1 = PConjugateGradient(M1.domain)
    
    bicgstab_solver0 = BiConjugateGradientStab(M0.domain)
    bicgstab_solver1 = BiConjugateGradientStab(M1.domain)
    
    pbicgstab_solver0 = PBiConjugateGradientStab(M0.domain)
    pbicgstab_solver1 = PBiConjugateGradientStab(M1.domain)

    # create random right-hand side vectors
    b0_str, b0 = create_equal_random_arrays(derham.Vh_fem['0'], 1234)
    b1_str, b1 = create_equal_random_arrays(derham.Vh_fem['1'], 1607)

    # ============ solve systems (M0) ==============
    res, info0_1 = cg(M0, b0)
    res, info0_2 = pcg(M0, b0, pc0)
    res, info0_3 = minres(M0, b0)
    
    res, info0_4 = cg_solver0.solve(M0, b0)
    res, info0_5 = pcg_solver0.solve(M0, b0, pc0)
    res, info0_6 = bicgstab_solver0.solve(M0, b0)
    res, info0_7 = pbicgstab_solver0.solve(M0, b0, pc0)
    
    # ============ solve systems (M1) (only ones with preconditioner) ============
    #res, info1_1 = cg(M1, b1)
    res, info1_2 = pcg(M1, b1, pc1)
    #res, info1_3 = minres(M1, b1)
    
    #res, info1_4 = cg_solver1.solve(M1, b1)
    res, info1_5 = pcg_solver1.solve(M1, b1, pc1)
    #res, info1_6 = bicgstab_solver1.solve(M1, b1)
    res, info1_7 = pbicgstab_solver1.solve(M1, b1, pc1)

    assert info0_1['success']
    assert info0_2['success']
    assert info0_3['success']
    assert info0_4['success']
    assert info0_5['success']
    assert info0_6['success']
    assert info0_7['success']

    #assert info1_1['success']
    assert info1_2['success']
    #assert info1_3['success']
    #assert info1_4['success']
    assert info1_5['success']
    #assert info1_6['success']
    assert info1_7['success']

    if verbose and mpi_rank == 0:
        print('info for cg                       (M0) : ', info0_1)
        print('info for pcg                      (M0) : ', info0_2)
        print('info for minres                   (M0) : ', info0_3)
        
        print()
        
        print('info for ConjugateGradient        (M0) : ', info0_4)
        print('info for PConjugateGradient       (M0) : ', info0_5)
        print('info for BiConjugateGradientStab  (M0) : ', info0_6)
        print('info for PBiConjugateGradientStab (M0) : ', info0_7)
        
        print('-----------------------------')
        
        #print('info for cg                       (M1) : ', info1_1)
        print('info for pcg                      (M1) : ', info1_2)
        #print('info for minres                   (M1) : ', info1_3)
        
        print()
        
        #print('info for ConjugateGradient        (M1) : ', info1_4)
        print('info for PConjugateGradient       (M1) : ', info1_5)
        #print('info for BiConjugateGradientStab  (M1) : ', info1_6)
        print('info for PBiConjugateGradientStab (M1) : ', info1_7)


if __name__ == '__main__':
    #test_solvers([8, 4, 4], [2, 2, 2], [False, True, True], ['Cuboid', {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 6., 'l3': 0., 'r3': 10.}], False, True)
    test_solvers([8, 6, 4], [2, 2, 2], [False, True, True], ['Colella', {'Lx' : 1., 'Ly' : 6., 'alpha' : .1, 'Lz' : 10.}], False, True)
