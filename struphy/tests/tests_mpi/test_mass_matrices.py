import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[5, 6, 7]])
@pytest.mark.parametrize('p',   [[2, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True]])
@pytest.mark.parametrize('bc', [[[None, None], [None, None], [None, None]],
                                [[None,  'd'], ['d', None], [None, None]],
                                [['d', None], [None,  'd'], [None, None]]])
@pytest.mark.parametrize('mapping', [
    ['Colella', {
        'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}]])
def test_mass(Nel, p, spl_kind, bc, mapping, show_plots=False):

    import numpy as np

    from struphy.geometry import domains
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.eigenvalue_solvers.mhd_operators import MHDOperators

    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays
    from struphy.psydac_api.mass import WeightedMassOperators
    from struphy.fields_background.mhd_equil.equils import ShearedSlab, ScrewPinch
    
    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank == 0:
        print()

    mpi_comm.Barrier()

    print(f'Rank {mpi_rank} | Start test_mass with ' +
          str(mpi_size) + ' MPI processes!')

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    if show_plots:
        import matplotlib.pyplot as plt
        domain.show()

    # load MHD equilibrium
    if mapping[0] == 'Cuboid':
        eq_mhd = ShearedSlab(**{'a': (mapping[1]['r1'] - mapping[1]['l1']),
                              'R0': (mapping[1]['r3'] - mapping[1]['l3'])/(2*np.pi),
                              'B0': 1.0, 'q0': 1.05,
                              'q1': 1.8, 'n1': 3.0,
                              'n2': 4.0, 'na': 0.0,
                              'beta': 10.0})

    elif mapping[0] == 'Colella':
        eq_mhd = ShearedSlab(**{'a': mapping[1]['Lx'],
                              'R0': mapping[1]['Lz']/(2*np.pi),
                              'B0': 1.0,
                              'q0': 1.05,
                              'q1': 1.8,
                              'n1': 3.0,
                              'n2': 4.0,
                              'na': 0.0,
                              'beta': 10.0})

        if show_plots:
            eq_mhd.plot_profiles()

    elif mapping[0] == 'HollowCylinder':
        eq_mhd = ScrewPinch(**{'a': mapping[1]['a2'],
                             'R0': 3.,
                             'B0': 1.0,
                             'q0': 1.05,
                             'q1': 1.8,
                             'n1': 3.0,
                             'n2': 4.0,
                             'na': 0.0,
                             'beta': 10.0})

        if show_plots:
            eq_mhd.plot_profiles()

    eq_mhd.domain = domain

    # make sure that boundary conditions are compatible with spline space (periodic only allows for None)
    bc_compatible = []

    for spl_i, bc_i in zip(spl_kind, bc):
        if spl_i:
            bc_compatible += [[None, None]]
        else:
            bc_compatible += [bc_i]

    # derham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm, bc=bc_compatible)

    print(f'Rank {mpi_rank} | Local domain : ' +
          str(derham.domain_array[mpi_rank]))

    fem_spaces = [derham.Vh_fem['0'],
                  derham.Vh_fem['1'],
                  derham.Vh_fem['2'],
                  derham.Vh_fem['3'],
                  derham.Vh_fem['v']]

    # mass matrices object
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)

    # compare to old STRUPHY
    spaces = [Spline_space_1d(Nel[0], p[0], spl_kind[0], p[0] + 1, bc_compatible[0]),
              Spline_space_1d(Nel[1], p[1], spl_kind[1],
                              p[1] + 1, bc_compatible[1]),
              Spline_space_1d(Nel[2], p[2], spl_kind[2], p[2] + 1, bc_compatible[2])]

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

    # create random input arrays
    x0_str, x0_psy = create_equal_random_arrays(
        fem_spaces[0], seed=1234, flattened=True)
    x1_str, x1_psy = create_equal_random_arrays(
        fem_spaces[1], seed=1568, flattened=True)
    x2_str, x2_psy = create_equal_random_arrays(
        fem_spaces[2], seed=8945, flattened=True)
    x3_str, x3_psy = create_equal_random_arrays(
        fem_spaces[3], seed=8196, flattened=True)
    xv_str, xv_psy = create_equal_random_arrays(
        fem_spaces[4], seed=2038, flattened=True)

    x0_str0 = space.B0.dot(x0_str)
    x1_str0 = space.B1.dot(x1_str)
    x2_str0 = space.B2.dot(x2_str)
    x3_str0 = space.B3.dot(x3_str)
    xv_str0 = space.Bv.dot(xv_str)

    # perfrom matrix-vector products (with boundary conditions)
    r0_str = space.B0.T.dot(space.M0_0(x0_str0))
    r1_str = space.B1.T.dot(space.M1_0(x1_str0))
    r2_str = space.B2.T.dot(space.M2_0(x2_str0))
    r3_str = space.B3.T.dot(space.M3_0(x3_str0))
    rv_str = space.Bv.T.dot(space.Mv_0(xv_str0))

    rn_str = space.B2.T.dot(mhd_ops_str.Mn(x2_str0))
    rJ_str = space.B2.T.dot(mhd_ops_str.MJ(x2_str0))

    r0_psy = mass_mats.M0.dot(x0_psy, apply_bc=True)
    r1_psy = mass_mats.M1.dot(x1_psy, apply_bc=True)
    r2_psy = mass_mats.M2.dot(x2_psy, apply_bc=True)
    r3_psy = mass_mats.M3.dot(x3_psy, apply_bc=True)
    rv_psy = mass_mats.Mv.dot(xv_psy, apply_bc=True)

    rn_psy = mass_mats.M2n.dot(x2_psy, apply_bc=True)
    rJ_psy = mass_mats.M2J.dot(x2_psy, apply_bc=True)

    # compare output arrays
    compare_arrays(r0_psy, r0_str, mpi_rank, atol=1e-14)
    compare_arrays(r1_psy, r1_str, mpi_rank, atol=1e-14)
    compare_arrays(r2_psy, r2_str, mpi_rank, atol=1e-14)
    compare_arrays(r3_psy, r3_str, mpi_rank, atol=1e-14)
    compare_arrays(rv_psy, rv_str, mpi_rank, atol=1e-14)

    compare_arrays(rn_psy, rn_str, mpi_rank, atol=1e-14)
    compare_arrays(rJ_psy, rJ_str, mpi_rank, atol=1e-14)

    # perfrom matrix-vector products (without boundary conditions)
    r0_str = space.M0(x0_str)
    r1_str = space.M1(x1_str)
    r2_str = space.M2(x2_str)
    r3_str = space.M3(x3_str)
    rv_str = space.Mv(xv_str)

    r0_psy = mass_mats.M0.dot(x0_psy, apply_bc=False)
    r1_psy = mass_mats.M1.dot(x1_psy, apply_bc=False)
    r2_psy = mass_mats.M2.dot(x2_psy, apply_bc=False)
    r3_psy = mass_mats.M3.dot(x3_psy, apply_bc=False)
    rv_psy = mass_mats.Mv.dot(xv_psy, apply_bc=False)

    # compare output arrays
    compare_arrays(r0_psy, r0_str, mpi_rank, atol=1e-14)
    compare_arrays(r1_psy, r1_str, mpi_rank, atol=1e-14)
    compare_arrays(r2_psy, r2_str, mpi_rank, atol=1e-14)
    compare_arrays(r3_psy, r3_str, mpi_rank, atol=1e-14)
    compare_arrays(rv_psy, rv_str, mpi_rank, atol=1e-14)

    print(f'Rank {mpi_rank} | All tests passed!')


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 12, 6]])
@pytest.mark.parametrize('p',   [[2, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [False, True, False]])
@pytest.mark.parametrize('bc', [[[None,  'd'], [None, None], [None, ' d']],
                                [[None, None], [None, None], ['d', None]]])
@pytest.mark.parametrize('mapping', [
    ['IGAPolarCylinder', {
        'a': 1., 'Lz': 3.}]])
def test_mass_polar(Nel, p, spl_kind, bc, mapping, show_plots=False):

    import numpy as np

    from struphy.geometry import domains
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.eigenvalue_solvers.mhd_operators import MHDOperators

    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays
    from struphy.psydac_api.mass import WeightedMassOperators
    from struphy.fields_background.mhd_equil.equils import ScrewPinch
    
    from struphy.polar.basic import PolarVector

    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank == 0:
        print()

    mpi_comm.Barrier()

    print(f'Rank {mpi_rank} | Start test_mass_polar with ' +
          str(mpi_size) + ' MPI processes!')

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(
        **{'Nel': Nel[:2], 'p': p[:2], 'a': mapping[1]['a'], 'Lz': mapping[1]['Lz']})

    if show_plots:
        import matplotlib.pyplot as plt
        domain.show(grid_info=Nel)

    # load MHD equilibrium
    eq_mhd = ScrewPinch(**{'a': mapping[1]['a'],
                         'R0': mapping[1]['Lz'],
                         'B0': 1.0,
                         'q0': 1.05,
                         'q1': 1.8,
                         'n1': 3.0,
                         'n2': 4.0,
                         'na': 0.0,
                         'beta': 10.0})

    if show_plots:
        eq_mhd.plot_profiles()

    eq_mhd.domain = domain

    # make sure that boundary conditions are compatible with spline space (periodic only allows for None)
    bc_compatible = []

    for spl_i, bc_i in zip(spl_kind, bc):
        if spl_i:
            bc_compatible += [[None, None]]
        else:
            bc_compatible += [bc_i]

    # derham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm, bc=bc_compatible,
                    with_projectors=False, polar_ck=1, domain=domain)

    print(f'Rank {mpi_rank} | Local domain : ' +
          str(derham.domain_array[mpi_rank]))

    # mass matrices object
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)

    # compare to old STRUPHY
    spaces = [Spline_space_1d(Nel[0], p[0], spl_kind[0], p[0] + 1, bc_compatible[0]),
              Spline_space_1d(Nel[1], p[1], spl_kind[1],
                              p[1] + 1, bc_compatible[1]),
              Spline_space_1d(Nel[2], p[2], spl_kind[2], p[2] + 1, bc_compatible[2])]

    spaces[0].set_projectors()
    spaces[1].set_projectors()
    spaces[2].set_projectors()

    space = Tensor_spline_space(
        spaces, ck=1, cx=domain.cx[:, :, 0], cy=domain.cy[:, :, 0])
    space.set_projectors('general')

    space.assemble_Mk(domain, 'V0')
    space.assemble_Mk(domain, 'V1')
    space.assemble_Mk(domain, 'V2')
    space.assemble_Mk(domain, 'V3')

    mhd_ops_str = MHDOperators(space, eq_mhd, 2)

    mhd_ops_str.assemble_Mn()
    mhd_ops_str.assemble_MJ()

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

    # apply boundary conditions to old STRUPHY
    x0_pol_str = x0_pol_psy.toarray(True)
    x1_pol_str = x1_pol_psy.toarray(True)
    x2_pol_str = x2_pol_psy.toarray(True)
    x3_pol_str = x3_pol_psy.toarray(True)

    x0_pol_str0 = space.B0.dot(x0_pol_str)
    x1_pol_str0 = space.B1.dot(x1_pol_str)
    x2_pol_str0 = space.B2.dot(x2_pol_str)
    x3_pol_str0 = space.B3.dot(x3_pol_str)

    # perfrom matrix-vector products (with boundary conditions)
    r0_pol_str = space.B0.T.dot(space.M0_0(x0_pol_str0))
    r1_pol_str = space.B1.T.dot(space.M1_0(x1_pol_str0))
    r2_pol_str = space.B2.T.dot(space.M2_0(x2_pol_str0))
    r3_pol_str = space.B3.T.dot(space.M3_0(x3_pol_str0))

    rn_pol_str = space.B2.T.dot(mhd_ops_str.Mn(x2_pol_str0))
    rJ_pol_str = space.B2.T.dot(mhd_ops_str.MJ(x2_pol_str0))

    r0_pol_psy = mass_mats.M0.dot(x0_pol_psy, apply_bc=True)
    r1_pol_psy = mass_mats.M1.dot(x1_pol_psy, apply_bc=True)
    r2_pol_psy = mass_mats.M2.dot(x2_pol_psy, apply_bc=True)
    r3_pol_psy = mass_mats.M3.dot(x3_pol_psy, apply_bc=True)

    rn_pol_psy = mass_mats.M2n.dot(x2_pol_psy, apply_bc=True)
    rJ_pol_psy = mass_mats.M2J.dot(x2_pol_psy, apply_bc=True)

    assert np.allclose(r0_pol_str, r0_pol_psy.toarray(True))
    assert np.allclose(r1_pol_str, r1_pol_psy.toarray(True))
    assert np.allclose(r2_pol_str, r2_pol_psy.toarray(True))
    assert np.allclose(r3_pol_str, r3_pol_psy.toarray(True))
    assert np.allclose(rn_pol_str, rn_pol_psy.toarray(True))
    assert np.allclose(rJ_pol_str, rJ_pol_psy.toarray(True))

    # perfrom matrix-vector products (without boundary conditions)
    r0_pol_str = space.M0(x0_pol_str)
    r1_pol_str = space.M1(x1_pol_str)
    r2_pol_str = space.M2(x2_pol_str)
    r3_pol_str = space.M3(x3_pol_str)

    r0_pol_psy = mass_mats.M0.dot(x0_pol_psy, apply_bc=False)
    r1_pol_psy = mass_mats.M1.dot(x1_pol_psy, apply_bc=False)
    r2_pol_psy = mass_mats.M2.dot(x2_pol_psy, apply_bc=False)
    r3_pol_psy = mass_mats.M3.dot(x3_pol_psy, apply_bc=False)

    assert np.allclose(r0_pol_str, r0_pol_psy.toarray(True))
    assert np.allclose(r1_pol_str, r1_pol_psy.toarray(True))
    assert np.allclose(r2_pol_str, r2_pol_psy.toarray(True))
    assert np.allclose(r3_pol_str, r3_pol_psy.toarray(True))
    assert np.allclose(rn_pol_str, rn_pol_psy.toarray(True))
    assert np.allclose(rJ_pol_str, rJ_pol_psy.toarray(True))

    print(f'Rank {mpi_rank} | All tests passed!')


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 12, 6]])
@pytest.mark.parametrize('p',   [[2, 3, 2]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [False, True, False]])
@pytest.mark.parametrize('bc', [[[None,  'd'], [None, None], [None, ' d']],
                                [[None, None], [None, None], ['d', None]]])
@pytest.mark.parametrize('mapping', [
    ['HollowCylinder', {
        'a1': .1, 'a2': 1., 'Lz': 18.84955592153876}]])
def test_mass_preconditioner(Nel, p, spl_kind, bc, mapping, show_plots=False):

    import numpy as np
    import time

    from struphy.geometry import domains
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.eigenvalue_solvers.mhd_operators import MHDOperators

    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays
    from struphy.psydac_api.mass import WeightedMassOperators
    from struphy.psydac_api.preconditioner import MassMatrixPreconditioner
    from struphy.psydac_api.linear_operators import InverseLinearOperator
    
    from struphy.fields_background.mhd_equil.equils import ShearedSlab, ScrewPinch
    
    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank == 0:
        print()

    mpi_comm.Barrier()

    print(f'Rank {mpi_rank} | Start test_mass_preconditioner with ' +
          str(mpi_size) + ' MPI processes!')

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(**mapping[1])

    if show_plots:
        import matplotlib.pyplot as plt
        domain.show()

    # load MHD equilibrium
    if mapping[0] == 'Cuboid':
        eq_mhd = ShearedSlab(**{'a': (mapping[1]['r1'] - mapping[1]['l1']),
                              'R0': (mapping[1]['r3'] - mapping[1]['l3'])/(2*np.pi),
                              'B0': 1.0, 'q0': 1.05,
                              'q1': 1.8, 'n1': 3.0,
                              'n2': 4.0, 'na': 0.0,
                              'beta': 10.0})

    elif mapping[0] == 'Colella':
        eq_mhd = ShearedSlab(**{'a': mapping[1]['Lx'],
                              'R0': mapping[1]['Lz']/(2*np.pi),
                              'B0': 1.0,
                              'q0': 1.05,
                              'q1': 1.8,
                              'n1': 3.0,
                              'n2': 4.0,
                              'na': 0.0,
                              'beta': 10.0})

        if show_plots:
            eq_mhd.plot_profiles()

    elif mapping[0] == 'HollowCylinder':
        eq_mhd = ScrewPinch(**{'a': mapping[1]['a2'],
                             'R0': 3.,
                             'B0': 1.0,
                             'q0': 1.05,
                             'q1': 1.8,
                             'n1': 3.0,
                             'n2': 4.0,
                             'na': 0.0,
                             'beta': 10.0})

        if show_plots:
            eq_mhd.plot_profiles()

    eq_mhd.domain = domain

    # make sure that boundary conditions are compatible with spline space (periodic only allows for None)
    bc_compatible = []

    for spl_i, bc_i in zip(spl_kind, bc):
        if spl_i:
            bc_compatible += [[None, None]]
        else:
            bc_compatible += [bc_i]

    # derham object
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm, bc=bc_compatible)

    fem_spaces = [derham.Vh_fem['0'],
                  derham.Vh_fem['1'],
                  derham.Vh_fem['2'],
                  derham.Vh_fem['3'],
                  derham.Vh_fem['v']]

    print(f'Rank {mpi_rank} | Local domain : ' +
          str(derham.domain_array[mpi_rank]))

    # exact mass matrices
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)

    # assemble preconditioners
    if mpi_rank == 0:
        print('Start assembling preconditioners')

    M0pre = MassMatrixPreconditioner(mass_mats.M0)
    M1pre = MassMatrixPreconditioner(mass_mats.M1)
    M2pre = MassMatrixPreconditioner(mass_mats.M2)
    M3pre = MassMatrixPreconditioner(mass_mats.M3)
    Mvpre = MassMatrixPreconditioner(mass_mats.Mv)

    M1npre = MassMatrixPreconditioner(mass_mats.M1n)
    M2npre = MassMatrixPreconditioner(mass_mats.M2n)
    Mvnpre = MassMatrixPreconditioner(mass_mats.Mvn)

    if mpi_rank == 0:
        print('Done')

    # create random input arrays
    x0 = create_equal_random_arrays(
        fem_spaces[0], seed=1234, flattened=True)[1]
    x1 = create_equal_random_arrays(
        fem_spaces[1], seed=1568, flattened=True)[1]
    x2 = create_equal_random_arrays(
        fem_spaces[2], seed=8945, flattened=True)[1]
    x3 = create_equal_random_arrays(
        fem_spaces[3], seed=8196, flattened=True)[1]
    xv = create_equal_random_arrays(
        fem_spaces[4], seed=2038, flattened=True)[1]

    # compare mass matrix-vector products with Kronecker products of preconditioner
    do_this_test = False

    if mapping[0] == 'Cuboid' or mapping[0] == 'HollowCylinder' and do_this_test:

        if mpi_rank == 0:
            print(
                'Start matrix-vector products in stencil format for mapping Cuboid/HollowCylinder')

        r0 = mass_mats.M0.dot(x0)
        r1 = mass_mats.M1.dot(x1)
        r2 = mass_mats.M2.dot(x2)
        r3 = mass_mats.M3.dot(x3)
        rv = mass_mats.Mv.dot(xv)

        r1n = mass_mats.M1n.dot(x1)
        r2n = mass_mats.M2n.dot(x2)
        rvn = mass_mats.Mvn.dot(xv)

        if mpi_rank == 0:
            print('Done')

        if mpi_rank == 0:
            print(
                'Start matrix-vector products in KroneckerStencil format for mapping Cuboid/HollowCylinder')

        r0_pre = M0pre.matrix.dot(x0)
        r1_pre = M1pre.matrix.dot(x1)
        r2_pre = M2pre.matrix.dot(x2)
        r3_pre = M3pre.matrix.dot(x3)
        rv_pre = Mvpre.matrix.dot(xv)

        r1n_pre = M1npre.matrix.dot(x1)
        r2n_pre = M2npre.matrix.dot(x2)
        rvn_pre = Mvnpre.matrix.dot(xv)

        if mpi_rank == 0:
            print('Done')

        # compare output arrays
        assert np.allclose(r0.toarray(), r0_pre.toarray())
        assert np.allclose(r1.toarray(), r1_pre.toarray())
        assert np.allclose(r2.toarray(), r2_pre.toarray())
        assert np.allclose(r3.toarray(), r3_pre.toarray())
        assert np.allclose(rv.toarray(), rv_pre.toarray())

        assert np.allclose(r1n.toarray(), r1n_pre.toarray())
        assert np.allclose(r2n.toarray(), r2n_pre.toarray())
        assert np.allclose(rvn.toarray(), rvn_pre.toarray())

    # test if preconditioner satisfies PC * M = Identity
    if mapping[0] == 'Cuboid' or mapping[0] == 'HollowCylinder':

        assert np.allclose(mass_mats.M0.dot(M0pre.solve(
            x0)).toarray(), derham.B['0'].dot(x0).toarray())
        assert np.allclose(mass_mats.M1.dot(M1pre.solve(
            x1)).toarray(), derham.B['1'].dot(x1).toarray())
        assert np.allclose(mass_mats.M2.dot(M2pre.solve(
            x2)).toarray(), derham.B['2'].dot(x2).toarray())
        assert np.allclose(mass_mats.M3.dot(M3pre.solve(
            x3)).toarray(), derham.B['3'].dot(x3).toarray())
        assert np.allclose(mass_mats.Mv.dot(Mvpre.solve(
            xv)).toarray(), derham.B['v'].dot(xv).toarray())

    # test preconditioner in iterative solver
    M0inv = InverseLinearOperator(
        mass_mats.M0, pc=M0pre, tol=1e-8, maxiter=1000)
    M1inv = InverseLinearOperator(
        mass_mats.M1, pc=M1pre, tol=1e-8, maxiter=1000)
    M2inv = InverseLinearOperator(
        mass_mats.M2, pc=M2pre, tol=1e-8, maxiter=1000)
    M3inv = InverseLinearOperator(
        mass_mats.M3, pc=M3pre, tol=1e-8, maxiter=1000)
    Mvinv = InverseLinearOperator(
        mass_mats.Mv, pc=Mvpre, tol=1e-8, maxiter=1000)

    M1ninv = InverseLinearOperator(
        mass_mats.M1n, pc=M1npre, tol=1e-8, maxiter=1000)
    M2ninv = InverseLinearOperator(
        mass_mats.M2n, pc=M2npre, tol=1e-8, maxiter=1000)
    Mvninv = InverseLinearOperator(
        mass_mats.Mvn, pc=Mvnpre, tol=1e-8, maxiter=1000)

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M0 with preconditioner')
        r0 = M0inv.dot(derham.B['0'].dot(x0), verbose=True)
    else:
        r0 = M0inv.dot(derham.B['0'].dot(x0), verbose=False)

    if mapping[0] == 'Cuboid' or mapping[0] == 'HollowCylinder':
        assert M0inv.info['niter'] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M1 with preconditioner')
        r1 = M1inv.dot(derham.B['1'].dot(x1), verbose=True)
    else:
        r1 = M1inv.dot(derham.B['1'].dot(x1), verbose=False)

    if mapping[0] == 'Cuboid' or mapping[0] == 'HollowCylinder':
        assert M1inv.info['niter'] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M2 with preconditioner')
        r2 = M2inv.dot(derham.B['2'].dot(x2), verbose=True)
    else:
        r2 = M2inv.dot(derham.B['2'].dot(x2), verbose=False)

    if mapping[0] == 'Cuboid' or mapping[0] == 'HollowCylinder':
        assert M2inv.info['niter'] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M3 with preconditioner')
        r3 = M3inv.dot(derham.B['3'].dot(x3), verbose=True)
    else:
        r3 = M3inv.dot(derham.B['3'].dot(x3), verbose=False)

    if mapping[0] == 'Cuboid' or mapping[0] == 'HollowCylinder':
        assert M3inv.info['niter'] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert Mv with preconditioner')
        rv = Mvinv.dot(derham.B['v'].dot(xv), verbose=True)
    else:
        rv = Mvinv.dot(derham.B['v'].dot(xv), verbose=False)

    if mapping[0] == 'Cuboid' or mapping[0] == 'HollowCylinder':
        assert Mvinv.info['niter'] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Apply M1n with preconditioner')
        r1n = M1ninv.dot(derham.B['1'].dot(x1), verbose=True)
    else:
        r1n = M1ninv.dot(derham.B['1'].dot(x1), verbose=False)

    if mapping[0] == 'Cuboid' or mapping[0] == 'HollowCylinder':
        assert M1ninv.info['niter'] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Apply M2n with preconditioner')
        r2n = M2ninv.dot(derham.B['2'].dot(x2), verbose=True)
    else:
        r2n = M2ninv.dot(derham.B['2'].dot(x2), verbose=False)

    if mapping[0] == 'Cuboid' or mapping[0] == 'HollowCylinder':
        assert M2ninv.info['niter'] == 2

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Apply Mvn with preconditioner')
        rvn = Mvninv.dot(derham.B['v'].dot(xv), verbose=True)
    else:
        rvn = Mvninv.dot(derham.B['v'].dot(xv), verbose=False)

    if mapping[0] == 'Cuboid' or mapping[0] == 'HollowCylinder':
        assert Mvninv.info['niter'] == 2

    time.sleep(2)
    print(f'Rank {mpi_rank} | All tests passed!')


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 6]])
@pytest.mark.parametrize('p',   [[2, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [False, True, False]])
@pytest.mark.parametrize('bc', [[[None,  'd'], [None, None], [None, ' d']],
                                [[None, None], [None, None], ['d', None]]])
@pytest.mark.parametrize('mapping', [
    ['IGAPolarCylinder', {
        'a': 1., 'Lz': 3.}]])
def test_mass_preconditioner_polar(Nel, p, spl_kind, bc, mapping, show_plots=False):

    import numpy as np
    import time

    from struphy.geometry import domains
    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d, Tensor_spline_space
    from struphy.eigenvalue_solvers.mhd_operators import MHDOperators

    from struphy.psydac_api.psydac_derham import Derham
    from struphy.psydac_api.utilities import create_equal_random_arrays, compare_arrays
    from struphy.psydac_api.mass import WeightedMassOperators
    from struphy.psydac_api.preconditioner import MassMatrixPreconditioner
    from struphy.psydac_api.linear_operators import InverseLinearOperator

    from struphy.polar.basic import PolarVector
    from struphy.fields_background.mhd_equil.equils import ScrewPinch
    
    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank == 0:
        print()

    mpi_comm.Barrier()

    print(f'Rank {mpi_rank} | Start test_mass_preconditioner_polar with ' +
          str(mpi_size) + ' MPI processes!')

    # mapping
    domain_class = getattr(domains, mapping[0])
    domain = domain_class(
        **{'Nel': Nel[:2], 'p': p[:2], 'a': mapping[1]['a'], 'Lz': mapping[1]['Lz']})

    if show_plots:
        import matplotlib.pyplot as plt
        domain.show()

    # load MHD equilibrium
    eq_mhd = ScrewPinch(**{'a': mapping[1]['a'],
                         'R0': mapping[1]['Lz'],
                         'B0': 1.0,
                         'q0': 1.05,
                         'q1': 1.8,
                         'n1': 3.0,
                         'n2': 4.0,
                         'na': 0.0,
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
    derham = Derham(Nel, p, spl_kind, comm=mpi_comm, bc=bc_compatible,
                    with_projectors=False, polar_ck=1, domain=domain)

    print(f'Rank {mpi_rank} | Local domain : ' +
          str(derham.domain_array[mpi_rank]))

    # exact mass matrices
    mass_mats = WeightedMassOperators(derham, domain, eq_mhd=eq_mhd)

    # preconditioners
    if mpi_rank == 0:
        print('Start assembling preconditioners')

    M0pre = MassMatrixPreconditioner(mass_mats.M0)
    M1pre = MassMatrixPreconditioner(mass_mats.M1)
    M2pre = MassMatrixPreconditioner(mass_mats.M2)
    M3pre = MassMatrixPreconditioner(mass_mats.M3)

    M1npre = MassMatrixPreconditioner(mass_mats.M1n)
    M2npre = MassMatrixPreconditioner(mass_mats.M2n)

    if mpi_rank == 0:
        print('Done')

    # create random input arrays
    x0 = create_equal_random_arrays(
        derham.Vh_fem['0'], seed=1234, flattened=True)[1]
    x1 = create_equal_random_arrays(
        derham.Vh_fem['1'], seed=1568, flattened=True)[1]
    x2 = create_equal_random_arrays(
        derham.Vh_fem['2'], seed=8945, flattened=True)[1]
    x3 = create_equal_random_arrays(
        derham.Vh_fem['3'], seed=8196, flattened=True)[1]

    # set polar vectors
    x0_pol = PolarVector(derham.Vh_pol['0'])
    x1_pol = PolarVector(derham.Vh_pol['1'])
    x2_pol = PolarVector(derham.Vh_pol['2'])
    x3_pol = PolarVector(derham.Vh_pol['3'])

    x0_pol.tp = x0
    x1_pol.tp = x1
    x2_pol.tp = x2
    x3_pol.tp = x3

    np.random.seed(1607)
    x0_pol.pol = [np.random.rand(
        x0_pol.pol[0].shape[0], x0_pol.pol[0].shape[1])]
    x1_pol.pol = [np.random.rand(
        x1_pol.pol[n].shape[0], x1_pol.pol[n].shape[1]) for n in range(3)]
    x2_pol.pol = [np.random.rand(
        x2_pol.pol[n].shape[0], x2_pol.pol[n].shape[1]) for n in range(3)]
    x3_pol.pol = [np.random.rand(
        x3_pol.pol[0].shape[0], x3_pol.pol[0].shape[1])]

    # test preconditioner in iterative solver and compare to case without preconditioner
    M0inv = InverseLinearOperator(
        mass_mats.M0, pc=M0pre, tol=1e-8, maxiter=500)
    M1inv = InverseLinearOperator(
        mass_mats.M1, pc=M1pre, tol=1e-8, maxiter=500)
    M2inv = InverseLinearOperator(
        mass_mats.M2, pc=M2pre, tol=1e-8, maxiter=500)
    M3inv = InverseLinearOperator(
        mass_mats.M3, pc=M3pre, tol=1e-8, maxiter=500)

    M1ninv = InverseLinearOperator(
        mass_mats.M1n, pc=M1npre, tol=1e-8, maxiter=500)
    M2ninv = InverseLinearOperator(
        mass_mats.M2n, pc=M2npre, tol=1e-8, maxiter=500)

    M0inv_nopc = InverseLinearOperator(
        mass_mats.M0, pc=None, tol=1e-8, maxiter=500)
    M1inv_nopc = InverseLinearOperator(
        mass_mats.M1, pc=None, tol=1e-8, maxiter=500)
    M2inv_nopc = InverseLinearOperator(
        mass_mats.M2, pc=None, tol=1e-8, maxiter=500)
    M3inv_nopc = InverseLinearOperator(
        mass_mats.M3, pc=None, tol=1e-8, maxiter=500)

    M1ninv_nopc = InverseLinearOperator(
        mass_mats.M1n, pc=None, tol=1e-8, maxiter=500)
    M2ninv_nopc = InverseLinearOperator(
        mass_mats.M2n, pc=None, tol=1e-8, maxiter=500)

    # =============== M0 ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M0 with preconditioner')
        r0 = M0inv.dot(derham.B['0'].dot(x0_pol), verbose=True)
        print('Number of iterations : ', M0inv.info['niter'])
    else:
        r0 = M0inv.dot(derham.B['0'].dot(x0_pol), verbose=False)

    assert M0inv.info['success']

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M0 without preconditioner')
        r0 = M0inv_nopc.dot(derham.B['0'].dot(x0_pol), verbose=False)
        print('Number of iterations : ', M0inv_nopc.info['niter'])
    else:
        r0 = M0inv_nopc.dot(derham.B['0'].dot(x0_pol), verbose=False)

    assert M0inv.info['niter'] < M0inv_nopc.info['niter']
    # =======================================================

    # =============== M1 ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M1 with preconditioner')
        r1 = M1inv.dot(derham.B['1'].dot(x1_pol), verbose=True)
        print('Number of iterations : ', M1inv.info['niter'])
    else:
        r1 = M1inv.dot(derham.B['1'].dot(x1_pol), verbose=False)

    assert M1inv.info['success']

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M1 without preconditioner')
        r1 = M1inv_nopc.dot(derham.B['1'].dot(x1_pol), verbose=False)
        print('Number of iterations : ', M1inv_nopc.info['niter'])
    else:
        r1 = M1inv_nopc.dot(derham.B['1'].dot(x1_pol), verbose=False)

    assert M1inv.info['niter'] < M1inv_nopc.info['niter']
    # =======================================================

    # =============== M2 ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M2 with preconditioner')
        r2 = M2inv.dot(derham.B['2'].dot(x2_pol), verbose=True)
        print('Number of iterations : ', M2inv.info['niter'])
    else:
        r2 = M2inv.dot(derham.B['2'].dot(x2_pol), verbose=False)

    assert M2inv.info['success']

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M2 without preconditioner')
        r2 = M2inv_nopc.dot(derham.B['2'].dot(x2_pol), verbose=False)
        print('Number of iterations : ', M2inv_nopc.info['niter'])
    else:
        r2 = M2inv_nopc.dot(derham.B['2'].dot(x2_pol), verbose=False)

    assert M2inv.info['niter'] < M2inv_nopc.info['niter']
    # =======================================================

    # =============== M3 ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M3 with preconditioner')
        r3 = M3inv.dot(derham.B['3'].dot(x3_pol), verbose=True)
        print('Number of iterations : ', M3inv.info['niter'])
    else:
        r3 = M3inv.dot(derham.B['3'].dot(x3_pol), verbose=False)

    assert M3inv.info['success']

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M3 without preconditioner')
        r3 = M3inv_nopc.dot(derham.B['3'].dot(x3_pol), verbose=False)
        print('Number of iterations : ', M3inv_nopc.info['niter'])
    else:
        r3 = M3inv_nopc.dot(derham.B['3'].dot(x3_pol), verbose=False)

    assert M3inv.info['niter'] < M3inv_nopc.info['niter']
    # =======================================================

    # =============== M1n ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M1n with preconditioner')
        r1 = M1ninv.dot(derham.B['1'].dot(x1_pol), verbose=True)
        print('Number of iterations : ', M1ninv.info['niter'])
    else:
        r1 = M1ninv.dot(derham.B['1'].dot(x1_pol), verbose=False)

    assert M1ninv.info['success']

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M1n without preconditioner')
        r1 = M1ninv_nopc.dot(derham.B['1'].dot(x1_pol), verbose=False)
        print('Number of iterations : ', M1ninv_nopc.info['niter'])
    else:
        r1 = M1ninv_nopc.dot(derham.B['1'].dot(x1_pol), verbose=False)

    assert M1ninv.info['niter'] < M1ninv_nopc.info['niter']
    # =======================================================

    # =============== M2n ===================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M2n with preconditioner')
        r2 = M2ninv.dot(derham.B['2'].dot(x2_pol), verbose=True)
        print('Number of iterations : ', M2ninv.info['niter'])
    else:
        r2 = M2ninv.dot(derham.B['2'].dot(x2_pol), verbose=False)

    assert M2ninv.info['success']

    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('Invert M2n without preconditioner')
        r2 = M2ninv_nopc.dot(derham.B['2'].dot(x2_pol), verbose=False)
        print('Number of iterations : ', M2ninv_nopc.info['niter'])
    else:
        r2 = M2ninv_nopc.dot(derham.B['2'].dot(x2_pol), verbose=False)

    assert M2ninv.info['niter'] < M2ninv_nopc.info['niter']
    # =======================================================

    time.sleep(2)
    print(f'Rank {mpi_rank} | All tests passed!')


if __name__ == '__main__':
    test_mass([6, 7, 4], [2, 3, 1], [False, True, False], [[None, None], [None, None], [
              None, None]], ['Colella', {'Lx': 1., 'Ly': 6., 'alpha': .1, 'Lz': 10.}], False)
    # test_mass([8, 6, 4], [2, 3, 2], [False, True, False], [['d', 'd'], [None, None], [None, 'd']], ['Colella', {'Lx' : 1., 'Ly' : 6., 'alpha' : .1, 'Lz' : 10.}], False)
    # test_mass([8, 6, 4], [2, 2, 2], [False, True, True], [['d', 'd'], [None, None], [None, None]], ['HollowCylinder', {'a1': .1, 'a2': 1., 'Lz': 10.}], False)

    #test_mass_polar([8, 12, 6], [4, 3, 2], [False, True, False], [[None, 'd'], [None, None], ['d', None]], ['IGAPolarCylinder', {'a': 1., 'Lz': 3.}], False)

    #test_mass_preconditioner([8, 6, 4], [2, 2, 2], [False, False, False], [['d', 'd'], [None, None], [None, None]], ['Cuboid', {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 6., 'l3': 0., 'r3': 10.}], False)
    #test_mass_preconditioner([8, 6, 4], [2, 2, 2], [False, False, False], [['d', 'd'], [None, None], [None, None]], ['Colella', {'Lx' : 1., 'Ly' : 6., 'alpha' : .05, 'Lz' : 10.}], False)
    #test_mass_preconditioner([6, 9, 4], [4, 3, 2], [False, True, False], [[None, 'd'], [None, None], ['d', None]], ['HollowCylinder', {'a1' : .1, 'a2' : 1., 'Lz' : 18.84955592153876}], False)

    #test_mass_preconditioner_polar([8, 12, 6], [4, 3, 2], [False, True, False], [[None, 'd'], [None, None], ['d', None]], ['IGAPolarCylinder', {'a': 1., 'Lz': 3.}], False)
