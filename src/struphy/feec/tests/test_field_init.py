import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 10, 12]])
@pytest.mark.parametrize('p', [[1, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [True, True, False]])
@pytest.mark.parametrize('spaces', [['H1', 'Hcurl', 'Hdiv'], ['Hdiv', 'L2'], ['H1vec']])
@pytest.mark.parametrize('vec_comps', [[True, True, False], [False, True, True]])
def test_bckgr_init_const(Nel, p, spl_kind, spaces, vec_comps):
    '''Test field background initialization of "LogicalConst" with multiple fields in params.'''

    from mpi4py import MPI
    import numpy as np

    from struphy.feec.psydac_derham import Derham

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence and field of space
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # background parameters
    bckgr_params = {'type': 'LogicalConst',
                    'LogicalConst': {'comps': {}}
                    }

    np.random.seed(1234)

    vals = []
    for i, space in enumerate(spaces):

        val = np.random.rand()
        # sometimes test integers
        if val > .5:
            val = 1
        vals += [val]

        if space in ('H1', 'L2'):
            bckgr_params['LogicalConst']['comps']['name_' + str(i)] = val
        else:
            li = [val if veci else None for veci in vec_comps]
            bckgr_params['LogicalConst']['comps']['name_' + str(i)] = li

    if rank == 0:
        print(f'{bckgr_params =}')

    # evaluation grids for comparisons
    e1 = np.linspace(0., 1., Nel[0])
    e2 = np.linspace(0., 1., Nel[1])
    e3 = np.linspace(0., 1., Nel[2])
    meshgrids = np.meshgrid(e1, e2, e3, indexing='ij')

    # test
    for i, (space, val) in enumerate(zip(spaces, vals)):
        field = derham.create_field(
            'name_' + str(i), space, bckgr_params=bckgr_params)
        field.initialize_coeffs()

        if space in ('H1', 'L2'):
            print(
                f'\n{rank = }, {space = }, after init:\n {np.max(np.abs(field(*meshgrids) - val)) = }')
            # print(f'{field(*meshgrids) = }')
            assert np.allclose(field(*meshgrids), val)
        else:
            for i in range(3):
                if vec_comps[i]:
                    print(
                        f'\n{rank = }, {space = }, after init:\n {i = }, {np.max(np.abs(field(*meshgrids)[i] - val)) = }')
                    # print(f'{field(*meshgrids)[i] = }')
                    assert np.allclose(field(*meshgrids)[i], val)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[18, 24, 12]])
@pytest.mark.parametrize('p', [[1, 2, 1]])
@pytest.mark.parametrize('spl_kind', [[False, True, True]])
def test_bckgr_init_mhd(Nel, p, spl_kind, with_desc, show_plot=False):
    '''Test field background initialization of "MHD" with multiple fields in params.'''

    from mpi4py import MPI
    import numpy as np
    import inspect
    from matplotlib import pyplot as plt

    from struphy.feec.psydac_derham import Derham
    from struphy.geometry import domains
    from struphy.fields_background.mhd_equil import equils

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence and field of space
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # background parameters
    bckgr_params = {'type': 'MHD',
                    'MHD': {'comps': {}}
                    }

    vals = []
    bckgr_params['MHD']['comps']['name_0'] = 'absB0'
    bckgr_params['MHD']['comps']['name_1'] = 'b1'
    bckgr_params['MHD']['comps']['name_2'] = 'b2'
    bckgr_params['MHD']['comps']['name_3'] = 'p3'
    bckgr_params['MHD']['comps']['name_4'] = 'bv'

    if rank == 0:
        print(f'{bckgr_params =}')

    # evaluation grids for comparisons
    e1 = np.linspace(0., 1., Nel[0])
    e2 = np.linspace(0., 1., Nel[1])
    e3 = np.linspace(0., 1., Nel[2])
    meshgrids = np.meshgrid(e1, e2, e3, indexing='ij')

    # test
    for key, val in inspect.getmembers(equils):
        if inspect.isclass(val) and 'MHDequilibrium' not in key:
            print(f'{key = }')
            if 'DESCequilibrium' in key and not with_desc:
                print(f'Attention: {with_desc = }, DESC not tested here !!')
                continue
            mhd_equil = val()
            print(f'{mhd_equil.params = }')
            if 'AdhocTorus' in key:
                mhd_equil.domain = domains.HollowTorus(
                    a1=1e-3, a2=mhd_equil.params['a'], R0=mhd_equil.params['R0'], tor_period=1)
            elif 'EQDSKequilibrium' in key:
                mhd_equil.domain = domains.Tokamak(equilibrium=mhd_equil)
            elif 'HomogenSlab' in key:
                mhd_equil.domain = domains.Cuboid()
            elif 'ShearedSlab' in key:
                mhd_equil.domain = domains.Cuboid(r1=mhd_equil.params['a'],
                                                  r2=mhd_equil.params['a'] *
                                                  2*np.pi,
                                                  r3=mhd_equil.params['R0']*2*np.pi)
            elif 'ShearFluid' in key:
                mhd_equil.domain = domains.Cuboid(r1=mhd_equil.params['a'],
                                                  r2=mhd_equil.params['b'],
                                                  r3=mhd_equil.params['c'])
            elif 'ScrewPinch' in key:
                mhd_equil.domain = domains.HollowCylinder(a1=1e-3,
                                                          a2=mhd_equil.params['a'],
                                                          Lz=mhd_equil.params['R0']*2*np.pi)

            field_0 = derham.create_field(
                'name_0', 'H1', bckgr_params=bckgr_params)
            field_1 = derham.create_field(
                'name_1', 'Hcurl', bckgr_params=bckgr_params)
            field_2 = derham.create_field(
                'name_2', 'Hdiv', bckgr_params=bckgr_params)
            field_3 = derham.create_field(
                'name_3', 'L2', bckgr_params=bckgr_params)
            field_4 = derham.create_field(
                'name_4', 'H1vec', bckgr_params=bckgr_params)

            field_0.initialize_coeffs(mhd_equil=mhd_equil)
            print('field_0 initialized.')
            field_1.initialize_coeffs(mhd_equil=mhd_equil)
            print('field_1 initialized.')
            field_2.initialize_coeffs(mhd_equil=mhd_equil)
            print('field_2 initialized.')
            field_3.initialize_coeffs(mhd_equil=mhd_equil)
            print('field_3 initialized.')
            field_4.initialize_coeffs(mhd_equil=mhd_equil)
            print('field_4 initialized.')

            # scalar spaces
            print(f'{np.max(np.abs(field_0(*meshgrids) - mhd_equil.absB0(*meshgrids))) / np.max(np.abs(mhd_equil.absB0(*meshgrids)))}')
            print(f'{np.max(np.abs(field_3(*meshgrids) - mhd_equil.p3(*meshgrids))) / np.max(np.abs(mhd_equil.p3(*meshgrids)))}')
            assert np.max(np.abs(field_0(*meshgrids) - mhd_equil.absB0(*meshgrids))
                          ) / np.max(np.abs(mhd_equil.absB0(*meshgrids))) < 0.057
            assert np.max(np.abs(field_3(*meshgrids) - mhd_equil.p3(*meshgrids))
                          ) / np.max(np.abs(mhd_equil.p3(*meshgrids))) < 0.54
            print('Scalar asserts passed.')

            # vector-valued spaces
            ref = mhd_equil.b1(*meshgrids)
            if np.max(np.abs(ref[0])) < 1e-11:
                denom = 1.
            else:
                denom = np.max(np.abs(ref[0]))
            print(
                f'{np.max(np.abs(field_1(*meshgrids)[0] - ref[0])) / denom = }')
            assert np.max(np.abs(field_1(*meshgrids)[0] - ref[0])) / denom < .28
            if np.max(np.abs(ref[1])) < 1e-11:
                denom = 1.
            else:
                denom = np.max(np.abs(ref[1]))
            print(
                f'{np.max(np.abs(field_1(*meshgrids)[1] - ref[1])) / denom = }')
            assert np.max(np.abs(field_1(*meshgrids)[1] - ref[1])) / denom < .33
            if np.max(np.abs(ref[2])) < 1e-11:
                denom = 1.
            else:
                denom = np.max(np.abs(ref[2]))
            print(
                f'{np.max(np.abs(field_1(*meshgrids)[2] - ref[2])) / denom = }')
            assert np.max(np.abs(field_1(*meshgrids)
                          [2] - ref[2])) / denom < 0.1
            print('b1 asserts passed.')

            ref = mhd_equil.b2(*meshgrids)
            if np.max(np.abs(ref[0])) < 1e-11:
                denom = 1.
            else:
                denom = np.max(np.abs(ref[0]))
            print(
                f'{np.max(np.abs(field_2(*meshgrids)[0] - ref[0])) / denom = }')
            assert np.max(np.abs(field_2(*meshgrids)[0] - ref[0])) / denom < .86
            if np.max(np.abs(ref[1])) < 1e-11:
                denom = 1.
            else:
                denom = np.max(np.abs(ref[1]))
            print(
                f'{np.max(np.abs(field_2(*meshgrids)[1] - ref[1])) / denom = }')
            assert np.max(np.abs(field_2(*meshgrids)
                          [1] - ref[1])) / denom < 0.4
            if np.max(np.abs(ref[2])) < 1e-11:
                denom = 1.
            else:
                denom = np.max(np.abs(ref[2]))
            print(
                f'{np.max(np.abs(field_2(*meshgrids)[2] - ref[2])) / denom = }')
            assert np.max(np.abs(field_2(*meshgrids)[2] - ref[2])) / denom < .18
            print('b2 asserts passed.')

            ref = mhd_equil.bv(*meshgrids)
            if np.max(np.abs(ref[0])) < 1e-11:
                denom = 1.
            else:
                denom = np.max(np.abs(ref[0]))
            print(
                f'{np.max(np.abs(field_4(*meshgrids)[0] - ref[0])) / denom = }')
            assert np.max(np.abs(field_4(*meshgrids)[0] - ref[0])) / denom < .55
            if np.max(np.abs(ref[1])) < 1e-11:
                denom = 1.
            else:
                denom = np.max(np.abs(ref[1]))
            print(
                f'{np.max(np.abs(field_4(*meshgrids)[1] - ref[1])) / denom = }')
            assert np.max(np.abs(field_4(*meshgrids)
                          [1] - ref[1])) / denom < .2
            if np.max(np.abs(ref[2])) < 1e-11:
                denom = 1.
            else:
                denom = np.max(np.abs(ref[2]))
            print(
                f'{np.max(np.abs(field_4(*meshgrids)[2] - ref[2])) / denom = }')
            assert np.max(np.abs(field_4(*meshgrids)
                          [2] - ref[2])) / denom < .04
            print('bv asserts passed.')

            # plotting fields with equilibrium
            if show_plot and rank == 0:
                plt.figure(f'0/3-forms top, {mhd_equil = }', figsize=(24, 16))
                plt.figure(
                    f'0/3-forms poloidal, {mhd_equil = }', figsize=(24, 16))
                plt.figure(f'1-forms top, {mhd_equil = }', figsize=(24, 16))
                plt.figure(
                    f'1-forms poloidal, {mhd_equil = }', figsize=(24, 16))
                plt.figure(f'2-forms top, {mhd_equil = }', figsize=(24, 16))
                plt.figure(
                    f'2-forms poloidal, {mhd_equil = }', figsize=(24, 16))
                plt.figure(
                    f'vector-fields top, {mhd_equil = }', figsize=(24, 16))
                plt.figure(
                    f'vector-fields poloidal, {mhd_equil = }', figsize=(24, 16))
                x, y, z = mhd_equil.domain(*meshgrids)

                # 0-form
                absB0_h = mhd_equil.domain.push(field_0, *meshgrids)
                absB0 = mhd_equil.domain.push(mhd_equil.absB0, *meshgrids)

                levels = np.linspace(np.min(absB0) - 1e-10, np.max(absB0), 20)

                plt.figure(f'0/3-forms top, {mhd_equil = }')
                plt.subplot(2, 3, 1)
                if 'Slab' in key or 'Pinch' in key:
                    plt.contourf(x[:, 0, :], z[:, 0, :],
                                 absB0_h[:, 0, :], levels=levels)
                    plt.contourf(x[:, Nel[1]//2, :], z[:, Nel[1] //
                                 2 - 1, :], absB0_h[:, Nel[1]//2, :], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('z')
                else:
                    plt.contourf(x[:, 0, :], y[:, 0, :],
                                 absB0_h[:, 0, :], levels=levels)
                    plt.contourf(x[:, Nel[1]//2, :], y[:, Nel[1] //
                                 2 - 1, :], absB0_h[:, Nel[1]//2, :], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('y')
                plt.axis('equal')
                plt.colorbar()
                plt.title('Equilibrium $|B_0|$, top view (e1-e3)')
                plt.subplot(2, 3, 3 + 1)
                if 'Slab' in key or 'Pinch' in key:
                    plt.contourf(x[:, 0, :], z[:, 0, :],
                                 absB0[:, 0, :], levels=levels)
                    plt.contourf(x[:, Nel[1]//2, :], z[:, Nel[1] //
                                 2 - 1, :], absB0[:, Nel[1]//2, :], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('z')
                else:
                    plt.contourf(x[:, 0, :], y[:, 0, :],
                                 absB0[:, 0, :], levels=levels)
                    plt.contourf(x[:, Nel[1]//2, :], y[:, Nel[1] //
                                 2 - 1, :], absB0[:, Nel[1]//2, :], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('y')
                plt.axis('equal')
                plt.colorbar()
                plt.title('reference, top view (e1-e3)')

                plt.figure(f'0/3-forms poloidal, {mhd_equil = }')
                plt.subplot(2, 3, 1)
                if 'Slab' in key or 'Pinch' in key:
                    plt.contourf(x[:, :, 0], y[:, :, 0],
                                 absB0_h[:, :, 0], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('y')
                else:
                    plt.contourf(x[:, :, 0], z[:, :, 0],
                                 absB0_h[:, :, 0], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('z')
                plt.axis('equal')
                plt.colorbar()
                plt.title('Equilibrium $|B_0|$, poloidal view (e1-e2)')
                plt.subplot(2, 3, 3 + 1)
                if 'Slab' in key or 'Pinch' in key:
                    plt.contourf(x[:, :, 0], y[:, :, 0],
                                 absB0[:, :, 0], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('y')
                else:
                    plt.contourf(x[:, :, 0], z[:, :, 0],
                                 absB0[:, :, 0], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('z')
                plt.axis('equal')
                plt.colorbar()
                plt.title('reference, poloidal view (e1-e2)')

                # 3-form
                p3_h = mhd_equil.domain.push(field_3, *meshgrids)
                p3 = mhd_equil.domain.push(mhd_equil.p3, *meshgrids)

                levels = np.linspace(np.min(p3) - 1e-10, np.max(p3), 20)

                plt.figure(f'0/3-forms top, {mhd_equil = }')
                plt.subplot(2, 3, 2)
                if 'Slab' in key or 'Pinch' in key:
                    plt.contourf(x[:, 0, :], z[:, 0, :],
                                 p3_h[:, 0, :], levels=levels)
                    plt.contourf(x[:, Nel[1]//2, :], z[:, Nel[1] //
                                 2 - 1, :], p3_h[:, Nel[1]//2, :], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('z')
                else:
                    plt.contourf(x[:, 0, :], y[:, 0, :],
                                 p3_h[:, 0, :], levels=levels)
                    plt.contourf(x[:, Nel[1]//2, :], y[:, Nel[1] //
                                 2 - 1, :], p3_h[:, Nel[1]//2, :], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('y')
                plt.axis('equal')
                plt.colorbar()
                plt.title('Equilibrium $p_0$, top view (e1-e3)')
                plt.subplot(2, 3, 3 + 2)
                if 'Slab' in key or 'Pinch' in key:
                    plt.contourf(x[:, 0, :], z[:, 0, :],
                                 p3[:, 0, :], levels=levels)
                    plt.contourf(x[:, Nel[1]//2, :], z[:, Nel[1] //
                                 2 - 1, :], p3[:, Nel[1]//2, :], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('z')
                else:
                    plt.contourf(x[:, 0, :], y[:, 0, :],
                                 p3[:, 0, :], levels=levels)
                    plt.contourf(x[:, Nel[1]//2, :], y[:, Nel[1] //
                                 2 - 1, :], p3[:, Nel[1]//2, :], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('y')
                plt.axis('equal')
                plt.colorbar()
                plt.title('reference, top view (e1-e3)')

                plt.figure(f'0/3-forms poloidal, {mhd_equil = }')
                plt.subplot(2, 3, 2)
                if 'Slab' in key or 'Pinch' in key:
                    plt.contourf(x[:, :, 0], y[:, :, 0],
                                 p3_h[:, :, 0], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('y')
                else:
                    plt.contourf(x[:, :, 0], z[:, :, 0],
                                 p3_h[:, :, 0], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('z')
                plt.axis('equal')
                plt.colorbar()
                plt.title('Equilibrium $p_0$, poloidal view (e1-e2)')
                plt.subplot(2, 3, 3 + 2)
                if 'Slab' in key or 'Pinch' in key:
                    plt.contourf(x[:, :, 0], y[:, :, 0],
                                 p3[:, :, 0], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('y')
                else:
                    plt.contourf(x[:, :, 0], z[:, :, 0],
                                 p3[:, :, 0], levels=levels)
                    plt.xlabel('x')
                    plt.ylabel('z')
                plt.axis('equal')
                plt.colorbar()
                plt.title('reference, poloidal view (e1-e2)')

                # 1-form magnetic field plots
                b1h = mhd_equil.domain.push(
                    field_1(*meshgrids), *meshgrids, kind='1')
                b1 = mhd_equil.domain.push(
                    [*mhd_equil.b1(*meshgrids)], *meshgrids, kind='1')

                for i, (bh, b) in enumerate(zip(b1h, b1)):

                    levels = np.linspace(np.min(b) - 1e-10, np.max(b), 20)

                    plt.figure(f'1-forms top, {mhd_equil = }')
                    plt.subplot(2, 3, 1 + i)
                    if 'Slab' in key or 'Pinch' in key:
                        plt.contourf(x[:, 0, :], z[:, 0, :],
                                     bh[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1]//2, :], z[:, Nel[1] //
                                                           2 - 1, :], bh[:, Nel[1]//2, :], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('z')
                    else:
                        plt.contourf(x[:, 0, :], y[:, 0, :],
                                     bh[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1]//2, :], y[:, Nel[1] //
                                                           2 - 1, :], bh[:, Nel[1]//2, :], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('y')
                    plt.axis('equal')
                    plt.colorbar()
                    plt.title(f'Equilibrium $B_{i + 1}$, top view (e1-e3)')
                    plt.subplot(2, 3, 3 + 1 + i)
                    if 'Slab' in key or 'Pinch' in key:
                        plt.contourf(x[:, 0, :], z[:, 0, :],
                                     b[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1]//2, :], z[:, Nel[1] //
                                                           2 - 1, :], b[:, Nel[1]//2, :], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('z')
                    else:
                        plt.contourf(x[:, 0, :], y[:, 0, :],
                                     b[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1]//2, :], y[:, Nel[1] //
                                                           2 - 1, :], b[:, Nel[1]//2, :], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('y')
                    plt.axis('equal')
                    plt.colorbar()
                    plt.title('reference, top view (e1-e3)')

                    plt.figure(f'1-forms poloidal, {mhd_equil = }')
                    plt.subplot(2, 3, 1 + i)
                    if 'Slab' in key or 'Pinch' in key:
                        plt.contourf(x[:, :, 0], y[:, :, 0],
                                     bh[:, :, 0], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('y')
                    else:
                        plt.contourf(x[:, :, 0], z[:, :, 0],
                                     bh[:, :, 0], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('z')
                    plt.axis('equal')
                    plt.colorbar()
                    plt.title(
                        f'Equilibrium $B_{i + 1}$, poloidal view (e1-e2)')
                    plt.subplot(2, 3, 3 + 1 + i)
                    if 'Slab' in key or 'Pinch' in key:
                        plt.contourf(x[:, :, 0], y[:, :, 0],
                                     b[:, :, 0], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('y')
                    else:
                        plt.contourf(x[:, :, 0], z[:, :, 0],
                                     b[:, :, 0], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('z')
                    plt.axis('equal')
                    plt.colorbar()
                    plt.title('reference, poloidal view (e1-e2)')

                # 2-form magnetic field plots
                b2h = mhd_equil.domain.push(
                    field_2(*meshgrids), *meshgrids, kind='2')
                b2 = mhd_equil.domain.push(
                    [*mhd_equil.b2(*meshgrids)], *meshgrids, kind='2')

                for i, (bh, b) in enumerate(zip(b2h, b2)):

                    levels = np.linspace(np.min(b) - 1e-10, np.max(b), 20)

                    plt.figure(f'2-forms top, {mhd_equil = }')
                    plt.subplot(2, 3, 1 + i)
                    if 'Slab' in key or 'Pinch' in key:
                        plt.contourf(x[:, 0, :], z[:, 0, :],
                                     bh[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1]//2, :], z[:, Nel[1] //
                                                           2 - 1, :], bh[:, Nel[1]//2, :], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('z')
                    else:
                        plt.contourf(x[:, 0, :], y[:, 0, :],
                                     bh[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1]//2, :], y[:, Nel[1] //
                                                           2 - 1, :], bh[:, Nel[1]//2, :], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('y')
                    plt.axis('equal')
                    plt.colorbar()
                    plt.title(f'Equilibrium $B_{i + 1}$, top view (e1-e3)')
                    plt.subplot(2, 3, 3 + 1 + i)
                    if 'Slab' in key or 'Pinch' in key:
                        plt.contourf(x[:, 0, :], z[:, 0, :],
                                     b[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1]//2, :], z[:, Nel[1] //
                                                           2 - 1, :], b[:, Nel[1]//2, :], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('z')
                    else:
                        plt.contourf(x[:, 0, :], y[:, 0, :],
                                     b[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1]//2, :], y[:, Nel[1] //
                                                           2 - 1, :], b[:, Nel[1]//2, :], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('y')
                    plt.axis('equal')
                    plt.colorbar()
                    plt.title('reference, top view (e1-e3)')

                    plt.figure(f'2-forms poloidal, {mhd_equil = }')
                    plt.subplot(2, 3, 1 + i)
                    if 'Slab' in key or 'Pinch' in key:
                        plt.contourf(x[:, :, 0], y[:, :, 0],
                                     bh[:, :, 0], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('y')
                    else:
                        plt.contourf(x[:, :, 0], z[:, :, 0],
                                     bh[:, :, 0], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('z')
                    plt.axis('equal')
                    plt.colorbar()
                    plt.title(
                        f'Equilibrium $B_{i + 1}$, poloidal view (e1-e2)')
                    plt.subplot(2, 3, 3 + 1 + i)
                    if 'Slab' in key or 'Pinch' in key:
                        plt.contourf(x[:, :, 0], y[:, :, 0],
                                     b[:, :, 0], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('y')
                    else:
                        plt.contourf(x[:, :, 0], z[:, :, 0],
                                     b[:, :, 0], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('z')
                    plt.axis('equal')
                    plt.colorbar()
                    plt.title('reference, poloidal view (e1-e2)')

                # vector-field magnetic field plots
                bvh = mhd_equil.domain.push(
                    field_4(*meshgrids), *meshgrids, kind='v')
                bv = mhd_equil.domain.push(
                    [*mhd_equil.bv(*meshgrids)], *meshgrids, kind='v')

                for i, (bh, b) in enumerate(zip(bvh, bv)):

                    levels = np.linspace(np.min(b) - 1e-10, np.max(b), 20)

                    plt.figure(f'vector-fields top, {mhd_equil = }')
                    plt.subplot(2, 3, 1 + i)
                    if 'Slab' in key or 'Pinch' in key:
                        plt.contourf(x[:, 0, :], z[:, 0, :],
                                     bh[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1]//2, :], z[:, Nel[1] //
                                                           2 - 1, :], bh[:, Nel[1]//2, :], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('z')
                    else:
                        plt.contourf(x[:, 0, :], y[:, 0, :],
                                     bh[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1]//2, :], y[:, Nel[1] //
                                                           2 - 1, :], bh[:, Nel[1]//2, :], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('y')
                    plt.axis('equal')
                    plt.colorbar()
                    plt.title(f'Equilibrium $B_{i + 1}$, top view (e1-e3)')
                    plt.subplot(2, 3, 3 + 1 + i)
                    if 'Slab' in key or 'Pinch' in key:
                        plt.contourf(x[:, 0, :], z[:, 0, :],
                                     b[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1]//2, :], z[:, Nel[1] //
                                                           2 - 1, :], b[:, Nel[1]//2, :], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('z')
                    else:
                        plt.contourf(x[:, 0, :], y[:, 0, :],
                                     b[:, 0, :], levels=levels)
                        plt.contourf(x[:, Nel[1]//2, :], y[:, Nel[1] //
                                                           2 - 1, :], b[:, Nel[1]//2, :], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('y')
                    plt.axis('equal')
                    plt.colorbar()
                    plt.title('reference, top view (e1-e3)')

                    plt.figure(f'vector-fields poloidal, {mhd_equil = }')
                    plt.subplot(2, 3, 1 + i)
                    if 'Slab' in key or 'Pinch' in key:
                        plt.contourf(x[:, :, 0], y[:, :, 0],
                                     bh[:, :, 0], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('y')
                    else:
                        plt.contourf(x[:, :, 0], z[:, :, 0],
                                     bh[:, :, 0], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('z')
                    plt.axis('equal')
                    plt.colorbar()
                    plt.title(
                        f'Equilibrium $B_{i + 1}$, poloidal view (e1-e2)')
                    plt.subplot(2, 3, 3 + 1 + i)
                    if 'Slab' in key or 'Pinch' in key:
                        plt.contourf(x[:, :, 0], y[:, :, 0],
                                     b[:, :, 0], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('y')
                    else:
                        plt.contourf(x[:, :, 0], z[:, :, 0],
                                     b[:, :, 0], levels=levels)
                        plt.xlabel('x')
                        plt.ylabel('z')
                    plt.axis('equal')
                    plt.colorbar()
                    plt.title('reference, poloidal view (e1-e2)')

                plt.show()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[1, 32, 32]])
@pytest.mark.parametrize('p', [[1, 3, 3]])
@pytest.mark.parametrize('spl_kind', [[True, True, True]])
def test_sincos_init_const(Nel, p, spl_kind, show_plot=False):
    '''Test field perturbation with ModesSin + ModesCos on top of of "LogicalConst" with multiple fields in params.'''

    from mpi4py import MPI
    import numpy as np
    from matplotlib import pyplot as plt

    from struphy.feec.psydac_derham import Derham
    from struphy.initial.perturbations import ModesSin, ModesCos

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # background parameters
    avg_0 = 1.2
    avg_1 = [None, 2.6, 3.7]
    avg_2 = [2, 3, 4.2]

    bckgr_params = {'type': 'LogicalConst',
                    'LogicalConst': {'comps': {'name_0': avg_0,
                                               'name_1': avg_1,
                                               'name_2': avg_2}}
                    }

    # perturbations
    ms_s = [0, 2]
    ns_s = [1, 1]
    amps = [.2]
    f_sin = ModesSin(ms=ms_s, ns=ns_s, amps=amps)

    ms_c = [1]
    ns_c = [0]
    f_cos = ModesCos(ms=ms_c, ns=ns_c, amps=amps)

    pert_params = {'type': ['ModesSin', 'ModesCos'],
                   'ModesSin':
                       {'comps':
                           {'name_0': '0',
                            'name_1': ['1', None, '1']
                            },
                        'ms':
                            {'name_0': ms_s,
                             'name_1': [ms_s, None, ms_s]
                             },
                        'ns':
                            {'name_0': ns_s,
                             'name_1': [ns_s, None, ns_s]
                             },
                        'amps':
                            {'name_0': amps,
                             'name_1': [amps, None, amps]
                             }
                        },
                   'ModesCos':
                   {'comps':
                            {'name_0': '0',
                             'name_1': ['1', '1', None],
                             'name_2': [None, '2', None]
                             },
                    'ms':
                            {'name_0': ms_c,
                             'name_1': [ms_c, ms_c, None],
                             'name_2': [None, ms_c, None]
                             },
                    'ns':
                            {'name_0': ns_c,
                             'name_1': [ns_c, ns_c, None],
                             'name_2': [None, ns_c, None]
                             },
                    'amps':
                            {'name_0': amps,
                             'name_1': [amps, amps, None],
                             'name_2': [None, amps, None]
                             }
                    }
                   }

    # Psydac discrete Derham sequence and fields
    derham = Derham(Nel, p, spl_kind, comm=comm)

    field_0 = derham.create_field(
        'name_0', 'H1', bckgr_params=bckgr_params, pert_params=pert_params)
    field_1 = derham.create_field(
        'name_1', 'Hcurl', bckgr_params=bckgr_params, pert_params=pert_params)
    field_2 = derham.create_field(
        'name_2', 'Hdiv', bckgr_params=bckgr_params, pert_params=pert_params)

    field_0.initialize_coeffs()
    field_1.initialize_coeffs()
    field_2.initialize_coeffs()

    # evaluation grids for comparisons
    e1 = np.linspace(0., 1., Nel[0])
    e2 = np.linspace(0., 1., Nel[1])
    e3 = np.linspace(0., 1., Nel[2])
    meshgrids = np.meshgrid(e1, e2, e3, indexing='ij')

    fun_0 = avg_0 + f_sin(*meshgrids) + f_cos(*meshgrids)

    for i, a in enumerate(avg_1):
        if a is None:
            avg_1[i] = 0.

    for i, a in enumerate(avg_2):
        if a is None:
            avg_2[i] = 0.

    fun_1 = [avg_1[0] + f_sin(*meshgrids) + + f_cos(*meshgrids),
             avg_1[1] + f_cos(*meshgrids),
             avg_1[2] + f_sin(*meshgrids)]
    fun_2 = [avg_2[0] + 0.*meshgrids[0],
             avg_2[1] + f_cos(*meshgrids),
             avg_2[2] + 0.*meshgrids[0]]

    f0_h = field_0(*meshgrids)
    f1_h = field_1(*meshgrids)
    f2_h = field_2(*meshgrids)

    print(f'{np.max(np.abs(fun_0 - f0_h)) = }')
    print(f'{np.max(np.abs(fun_1[0] - f1_h[0])) = }')
    print(f'{np.max(np.abs(fun_1[1] - f1_h[1])) = }')
    print(f'{np.max(np.abs(fun_1[2] - f1_h[2])) = }')
    print(f'{np.max(np.abs(fun_2[0] - f2_h[0])) = }')
    print(f'{np.max(np.abs(fun_2[1] - f2_h[1])) = }')
    print(f'{np.max(np.abs(fun_2[2] - f2_h[2])) = }')

    assert np.max(np.abs(fun_0 - f0_h)) < 3e-5
    assert np.max(np.abs(fun_1[0] - f1_h[0])) < 3e-5
    assert np.max(np.abs(fun_1[1] - f1_h[1])) < 3e-5
    assert np.max(np.abs(fun_1[2] - f1_h[2])) < 3e-5
    assert np.max(np.abs(fun_2[0] - f2_h[0])) < 3e-5
    assert np.max(np.abs(fun_2[1] - f2_h[1])) < 3e-5
    assert np.max(np.abs(fun_2[2] - f2_h[2])) < 3e-5

    if show_plot and rank == 0:

        levels = np.linspace(np.min(fun_0) - 1e-10, np.max(fun_0), 40)

        plt.figure('0-form', figsize=(10, 16))
        plt.subplot(2, 1, 1)
        plt.contourf(meshgrids[1][0, :, :], meshgrids[2]
                     [0, :, :], f0_h[0, :, :], levels=levels)
        plt.xlabel('$\eta_2$')
        plt.ylabel('$\eta_3$')
        plt.xlim([0, 1.])
        plt.title('field_0')
        plt.axis('equal')
        plt.colorbar()
        plt.subplot(2, 1, 2)
        plt.contourf(meshgrids[1][0, :, :], meshgrids[2]
                     [0, :, :], fun_0[0, :, :], levels=levels)
        plt.xlabel('$\eta_2$')
        plt.ylabel('$\eta_3$')
        plt.title('reference')
        # plt.figure('1-form', figsize=(24, 16))
        # plt.figure('2-form', figsize=(24, 16))
        plt.axis('equal')
        plt.colorbar()

        plt.figure('1-form', figsize=(30, 16))
        for i, (f_h, fun) in enumerate(zip(f1_h, fun_1)):

            levels = np.linspace(np.min(fun) - 1e-10, np.max(fun), 40)

            plt.subplot(2, 3, 1 + i)
            plt.contourf(meshgrids[1][0, :, :], meshgrids[2]
                         [0, :, :], f_h[0, :, :], levels=levels)
            plt.xlabel('$\eta_2$')
            plt.ylabel('$\eta_3$')
            plt.xlim([0, 1.])
            plt.title(f'field_1, component {i + 1}')
            plt.axis('equal')
            plt.colorbar()
            plt.subplot(2, 3, 4 + i)
            plt.contourf(meshgrids[1][0, :, :], meshgrids[2]
                         [0, :, :], fun[0, :, :], levels=levels)
            plt.xlabel('$\eta_2$')
            plt.ylabel('$\eta_3$')
            plt.title('reference')
            # plt.figure('1-form', figsize=(24, 16))
            # plt.figure('2-form', figsize=(24, 16))
            plt.axis('equal')
            plt.colorbar()

        plt.figure('2-form', figsize=(30, 16))
        for i, (f_h, fun) in enumerate(zip(f2_h, fun_2)):

            levels = np.linspace(np.min(fun) - 1e-10, np.max(fun), 40)

            plt.subplot(2, 3, 1 + i)
            plt.contourf(meshgrids[1][0, :, :], meshgrids[2]
                         [0, :, :], f_h[0, :, :], levels=levels)
            plt.xlabel('$\eta_2$')
            plt.ylabel('$\eta_3$')
            plt.xlim([0, 1.])
            plt.title(f'field_2, component {i + 1}')
            plt.axis('equal')
            plt.colorbar()
            plt.subplot(2, 3, 4 + i)
            plt.contourf(meshgrids[1][0, :, :], meshgrids[2]
                         [0, :, :], fun[0, :, :], levels=levels)
            plt.xlabel('$\eta_2$')
            plt.ylabel('$\eta_3$')
            plt.title('reference')
            # plt.figure('1-form', figsize=(24, 16))
            # plt.figure('2-form', figsize=(24, 16))
            plt.axis('equal')
            plt.colorbar()

        plt.show()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 10, 12]])
@pytest.mark.parametrize('p', [[1, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, True, True], [True, False, True]])
@pytest.mark.parametrize('space', ['Hcurl', 'Hdiv', 'H1vec'])
@pytest.mark.parametrize('direction', ['e1', 'e2', 'e3'])
def test_noise_init(Nel, p, spl_kind, space, direction):
    '''Only tests 1d noise ('e1', 'e2', 'e3') !!'''

    from mpi4py import MPI
    import numpy as np

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays

    comm = MPI.COMM_WORLD
    assert comm.size >= 2
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence and field of space
    derham = Derham(Nel, p, spl_kind, comm=comm)
    field = derham.create_field('field', space)

    derham_np = Derham(Nel, p, spl_kind, comm=None)
    field_np = derham_np.create_field('field', space)

    # initial conditions
    init_params = {
        'type': 'noise',
        'noise': {
            'comps':
                {'field': [True, False, False]},
            'direction': direction,
            'amp': 0.0001,
            'seed': 1234,
        }
    }
    field.initialize_coeffs(init_params)
    field_np.initialize_coeffs(init_params)

    # print('#'*80)
    # print(f'npts={field.vector[0].space.npts}, npts_np={field_np.vector[0].space.npts}')
    # print(f'rank={rank}: nprocs={derham.domain_array[rank]}')
    # print(f'rank={rank}, field={field.vector[0].toarray_local().shape}, field_np={field_np.vector[0].toarray_local().shape}')
    # print(f'rank={rank}: \ncomp{0}={field.vector[0].toarray_local()}, \ncomp{0}_np={field_np.vector[0].toarray_local()}')

    compare_arrays(
        field.vector, [field_np.vector[n].toarray_local() for n in range(3)], rank)


if __name__ == '__main__':
    # test_bckgr_init_const([8, 10, 12], [1, 2, 3], [False, False, True], [
    #     'H1', 'Hcurl', 'Hdiv'], [True, True, False])
    test_bckgr_init_mhd([18, 24, 12], [1, 2, 1], [
                        False, True, True], show_plot=False)
    # test_sincos_init_const([1, 32, 32], [1, 3, 3], [True]*3, show_plot=True)
    # test_noise_init([4, 8, 6], [1, 1, 1], [True, True, True], 'Hcurl', 'e1')
