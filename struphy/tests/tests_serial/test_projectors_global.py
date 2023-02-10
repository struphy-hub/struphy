def test_projectors_1d(plot=False, p_range=6, N_range=8):
    '''
    1) test order of convergence
    2) test projector property, i.e. Pi(basis_function) = basis_function
    '''

    import sys
    sys.path.append('..')

    import numpy as np
    import struphy.eigenvalue_solvers.spline_space as spl
    import matplotlib.pyplot as plt

    # test arbitrary function
    def f_per(eta): return np.cos(2*np.pi*eta/.5)
    def f_cla(eta): return np.exp(2.*eta) - 2.*np.cos(2*np.pi*eta/.5)

    # plot points
    eta_plot = np.linspace(0, 1, 200)
    fig, axs = plt.subplots(2, 2)

    print('1) convergence test:')
    # 1) convergence test
    for p in range(1, p_range):
        err0_per = [np.NaN]
        err0_cla = [np.NaN]
        err1_per = [np.NaN]
        err1_cla = [np.NaN]

        order0_per = []
        order0_cla = []
        order1_per = []
        order1_cla = []
        for Nel in [2**n for n in range(5, N_range)]:

            # spline spaces
            Vh_per = spl.Spline_space_1d(Nel, p, spl_kind=True)
            Vh_cla = spl.Spline_space_1d(Nel, p, spl_kind=False)

            # projectors
            Vh_per.set_projectors()
            Vh_cla.set_projectors()

            # callable as input
            c0_per = Vh_per.projectors.pi_0(f_per)
            c0_cla = Vh_cla.projectors.pi_0(f_cla)
            c1_per = Vh_per.projectors.pi_1(f_per)
            c1_cla = Vh_cla.projectors.pi_1(f_cla)

            # dofs as input
            dofs0_per = Vh_per.projectors.dofs_0(f_per)
            dofs0_cla = Vh_cla.projectors.dofs_0(f_cla)
            dofs1_per = Vh_per.projectors.dofs_1(f_per)
            dofs1_cla = Vh_cla.projectors.dofs_1(f_cla)

            c0_per_mat = Vh_per.projectors.pi_0_mat(dofs0_per)
            c0_cla_mat = Vh_cla.projectors.pi_0_mat(dofs0_cla)
            c1_per_mat = Vh_per.projectors.pi_1_mat(dofs1_per)
            c1_cla_mat = Vh_cla.projectors.pi_1_mat(dofs1_cla)

            # result must be the same
            assert np.allclose(c0_per, c0_per_mat, atol=1e-14)
            assert np.allclose(c0_cla, c0_cla_mat, atol=1e-14)
            assert np.allclose(c1_per, c1_per_mat, atol=1e-14)
            assert np.allclose(c1_cla, c1_cla_mat, atol=1e-14)

            # compute error:
            f0_h_per = Vh_per.evaluate_N(eta_plot, c0_per)
            f0_h_cla = Vh_cla.evaluate_N(eta_plot, c0_cla)
            f1_h_per = Vh_per.evaluate_D(eta_plot, c1_per)
            f1_h_cla = Vh_cla.evaluate_D(eta_plot, c1_cla)

            err0_per.append(np.max(np.abs(f0_h_per - f_per(eta_plot))))
            err0_cla.append(np.max(np.abs(f0_h_cla - f_cla(eta_plot))))
            err1_per.append(np.max(np.abs(f1_h_per - f_per(eta_plot))))
            err1_cla.append(np.max(np.abs(f1_h_cla - f_cla(eta_plot))))

            order0_per.append(np.log2(err0_per[-2]/err0_per[-1]))
            order0_cla.append(np.log2(err0_cla[-2]/err0_cla[-1]))
            order1_per.append(np.log2(err1_per[-2]/err1_per[-1]))
            order1_cla.append(np.log2(err1_cla[-2]/err1_cla[-1]))

            # print to screen
            if True:
                print('p: {0:2d}, Nel: {1:4d},   0_per: {2:8.6f} {3:4.2f},   0_cla: {4:8.6f} {5:4.2f},   1_per: {6:8.6f} {7:4.2f},   1_cla: {8:8.6f} {9:4.2f}'.format(
                    p, Nel,
                    err0_per[-1], order0_per[-1],
                    err0_cla[-1], order0_cla[-1],
                    err1_per[-1], order1_per[-1],
                    err1_cla[-1], order1_cla[-1])
                )

            # make plot
            if p < 3 and Nel == 2**5:
                axs.flatten()[2*p-2].plot(eta_plot,
                                          f_per(eta_plot), 'r', label='f')
                axs.flatten()[2*p-2].plot(eta_plot,
                                          f0_h_per, 'b--', label='f0h_per')
                axs.flatten()[2*p-2].plot(eta_plot,
                                          f1_h_per, 'k--', label='f1h_per')
                axs.flatten()[
                    2*p-2].set_title('p: {0:2d}, Nel: {1:4d}, periodic'.format(p, Nel))
                axs.flatten()[2*p-2].legend()
                axs.flatten()[2*p-2].autoscale(enable=True,
                                               axis='x', tight=True)

                axs.flatten()[2*p-1].plot(eta_plot,
                                          f_cla(eta_plot), 'r', label='f')
                axs.flatten()[2*p-1].plot(eta_plot,
                                          f0_h_cla, 'b--', label='f0h_cla')
                axs.flatten()[2*p-1].plot(eta_plot,
                                          f1_h_cla, 'k--', label='f1h_cla')
                axs.flatten()[
                    2*p-1].set_title('p: {0:2d}, Nel: {1:4d}, non-periodic'.format(p, Nel))
                axs.flatten()[2*p-1].legend()
                axs.flatten()[2*p-1].autoscale(enable=True,
                                               axis='x', tight=True)

        print()

        # check order of covergence
        assert order0_per[-1] > (p+1) - 0.6
        assert order0_cla[-1] > (p+1) - 0.6
        assert order1_per[-1] > (p) - 0.6
        assert order1_cla[-1] > (p) - 0.6

    if plot:
        plt.show()

    print('2) projector test:')
    # 2) projector test
    for p in range(1, p_range):

        # spline spaces
        Vh_per = spl.Spline_space_1d(32, p, spl_kind=True)
        Vh_cla = spl.Spline_space_1d(32, p, spl_kind=False)

        for nq in range(1, 6):

            # projectors
            Vh_per.set_projectors(nq)
            Vh_cla.set_projectors(nq)

            f0_per = np.zeros(Vh_per.NbaseN)
            f1_per = np.zeros(Vh_per.NbaseD)
            f0_cla = np.zeros(Vh_cla.NbaseN)
            f1_cla = np.zeros(Vh_cla.NbaseD)

            err0_per = [0., 0]
            err1_per = [0., 0]
            err0_cla = [0., 0]
            err1_cla = [0., 0]

            # basis functions V0 periodic
            for i in range(Vh_per.NbaseN):
                f0_per[i] = 1.

                def f(eta): return Vh_per.evaluate_N(eta, f0_per)

                # callable as input
                c0_per = Vh_per.projectors.pi_0(f)

                # dofs as input
                dofs0_per = Vh_per.projectors.dofs_0(f)
                c0_per_mat = Vh_per.projectors.pi_0_mat(dofs0_per)

                # result must be the same
                assert np.allclose(c0_per, c0_per_mat, atol=1e-14)

                if np.max(np.abs(f0_per - c0_per)) > err0_per[0]:
                    err0_per = [np.max(np.abs(f0_per - c0_per)), i]

                f0_per[i] = 0.

            # basis functions V1 periodic
            for i in range(Vh_per.NbaseD):
                f1_per[i] = 1.

                def f(eta): return Vh_per.evaluate_D(eta, f1_per)

                # callable as input
                c1_per = Vh_per.projectors.pi_1(f)

                # dofs as input
                dofs1_per = Vh_per.projectors.dofs_1(f)
                c1_per_mat = Vh_per.projectors.pi_1_mat(dofs1_per)

                # result must be the same
                assert np.allclose(c1_per, c1_per_mat, atol=1e-14)

                if np.max(np.abs(f1_per - c1_per)) > err1_per[0]:
                    err1_per = [np.max(np.abs(f1_per - c1_per)), i]

                f1_per[i] = 0.

            # basis functions V0 clamped
            for i in range(Vh_cla.NbaseN):
                f0_cla[i] = 1.

                def f(eta): return Vh_cla.evaluate_N(eta, f0_cla)

                # callable as input
                c0_cla = Vh_cla.projectors.pi_0(f)

                # dofs as input
                dofs0_cla = Vh_cla.projectors.dofs_0(f)
                c0_cla_mat = Vh_cla.projectors.pi_0_mat(dofs0_cla)

                # result must be the same
                assert np.allclose(c0_cla, c0_cla_mat, atol=1e-14)

                if np.max(np.abs(f0_cla - c0_cla)) > err0_cla[0]:
                    err0_cla = [np.max(np.abs(f0_cla - c0_cla)), i]

                f0_cla[i] = 0.

            # basis functions V1 clamped
            for i in range(Vh_cla.NbaseD):
                f1_cla[i] = 1.

                def f(eta): return Vh_cla.evaluate_D(eta, f1_cla)

                # callable as input
                c1_cla = Vh_cla.projectors.pi_1(f)

                # dofs as input
                dofs1_cla = Vh_cla.projectors.dofs_1(f)
                c1_cla_mat = Vh_cla.projectors.pi_1_mat(dofs1_cla)

                # result must be the same
                assert np.allclose(c1_cla, c1_cla_mat, atol=1e-14)

                if np.max(np.abs(f1_cla - c1_cla)) > err1_cla[0]:
                    err1_cla = [np.max(np.abs(f1_cla - c1_cla)), i]

                f1_cla[i] = 0.

            print('p: {0:2d}, nq: {1:2d},   maxerr V0_per: {2:4.2e} at i={3:2d},   maxerr V1_per: {4:4.2e} at i={5:2d},   maxerr V0_cla: {6:4.2e} at i={7:2d},   maxerr V1_cla: {8:4.2e} at i={9:2d}'.format(
                p, nq,
                err0_per[0], err0_per[1],
                err1_per[0], err1_per[1],
                err0_cla[0], err0_cla[1],
                err1_cla[0], err1_cla[1])
            )

        # check the projector property pi(pi) = pi
        assert err0_per[0] < 1e-13
        assert err1_per[0] < 1e-13
        assert err0_cla[0] < 1e-13
        assert err1_cla[0] < 1e-13

        print()

    print()


def test_projectors_2d(p_range=6, N_range=6):
    """
    TODO
    """

    import sys
    sys.path.append('..')

    import numpy as np
    import struphy.eigenvalue_solvers.spline_space as spl
    import matplotlib.pyplot as plt

    from struphy.eigenvalue_solvers import projectors_global as proj
    from struphy.b_splines import bspline_evaluation_2d as eval

    # test arbitrary function
    def f(eta1, eta2): return np.cos(2*np.pi*eta2/.5) * \
        (np.exp(2.*eta1) - 2.*np.cos(2*np.pi*eta1/.5))

    eta1_v = np.linspace(0, 1, 100)
    eta2_v = np.linspace(0, 1, 100)
    ee1, ee2 = np.meshgrid(eta1_v, eta2_v, indexing='ij')

    print('Convergence test 2d:')
    # convergence test
    for p in range(1, p_range):
        err0 = [1.]
        err11 = [1.]
        err12 = [1.]
        err2 = [1.]

        order0 = []
        order11 = []
        order12 = []
        order2 = []
        for Nel in [2**n for n in range(4, N_range)]:

            # spline spaces
            Vh_eta1 = spl.Spline_space_1d(Nel, p, spl_kind=False)
            Vh_eta2 = spl.Spline_space_1d(Nel, p, spl_kind=True)

            Vh_2d = spl.Tensor_spline_space([Vh_eta1, Vh_eta2], basis_tor='n')

            # projectors
            Vh_eta1.set_projectors()
            Vh_eta2.set_projectors()

            Vh_2d.set_projectors()

            # A) callable as input
            cij_0 = Vh_2d.projectors.PI_0(f)
            cij_11, cij_12 = Vh_2d.projectors.PI_1(f, f)
            cij_2 = Vh_2d.projectors.PI_2(f)

            # B) dofs as input
            # 1) values of f at point sets
            f_pts_0 = Vh_2d.projectors.eval_for_PI('0', f)
            f_pts_11 = Vh_2d.projectors.eval_for_PI('11', f)
            f_pts_12 = Vh_2d.projectors.eval_for_PI('12', f)
            f_pts_2 = Vh_2d.projectors.eval_for_PI('2', f)

            # 2) degrees of freedom
            dofs_0 = Vh_2d.projectors.dofs('0', f_pts_0)
            dofs_11 = Vh_2d.projectors.dofs('11', f_pts_11)
            dofs_12 = Vh_2d.projectors.dofs('12', f_pts_12)
            dofs_2 = Vh_2d.projectors.dofs('2', f_pts_2)

            # 3) fem coefficients obtained from projection
            cij_0_mat = Vh_2d.projectors.PI_mat('0', dofs_0)
            cij_11_mat = Vh_2d.projectors.PI_mat('11', dofs_11)
            cij_12_mat = Vh_2d.projectors.PI_mat('12', dofs_12)
            cij_2_mat = Vh_2d.projectors.PI_mat('2', dofs_2)

            # result from A) and B) must be the same
            assert np.allclose(cij_0, cij_0_mat, atol=1e-14)
            assert np.allclose(cij_11, cij_11_mat, atol=1e-14)
            assert np.allclose(cij_12, cij_12_mat, atol=1e-14)
            assert np.allclose(cij_2, cij_2_mat, atol=1e-14)

            # evaluation of splines:
            fh_0 = np.empty((eta1_v.size, eta2_v.size))
            fh_11 = fh_0.copy()
            fh_12 = fh_0.copy()
            fh_2 = fh_0.copy()
            f_mat = fh_0.copy()

            # print(cij_0.shape)

            fh_0 = Vh_2d.evaluate_NN(eta1_v, eta2_v, np.array(
                [0.]), cij_0.flatten(), 'V0')[:, :, 0]
            fh_11 = Vh_2d.evaluate_DN(eta1_v, eta2_v, np.array([0.]), np.concatenate(
                (cij_11.flatten(), cij_12.flatten(), cij_0.flatten())), 'V1')[:, :, 0]
            fh_12 = Vh_2d.evaluate_ND(eta1_v, eta2_v, np.array([0.]), np.concatenate(
                (cij_11.flatten(), cij_12.flatten(), cij_0.flatten())), 'V1')[:, :, 0]
            fh_2 = Vh_2d.evaluate_DD(eta1_v, eta2_v, np.array(
                [0.]), cij_2.flatten(), 'V3')[:, :, 0]

            # compute error:
            f_mat = f(ee1, ee2)

            err0.append(np.max(np.abs(fh_0 - f_mat)))
            err11.append(np.max(np.abs(fh_11 - f_mat)))
            err12.append(np.max(np.abs(fh_12 - f_mat)))
            err2.append(np.max(np.abs(fh_2 - f_mat)))

            order0.append(np.log2(err0[-2]/err0[-1]))
            order11.append(np.log2(err11[-2]/err11[-1]))
            order12.append(np.log2(err12[-2]/err12[-1]))
            order2.append(np.log2(err2[-2]/err2[-1]))

            #print('look  :', np.max(f_cla(eta_plot)), np.min(f_cla(eta_plot)))
            #print('look h:', np.max(f0_h_cla), np.min(f0_h_cla))

            if True:
                print('p: {0:2d}, Nel: {1:4d},   0: {2:8.6f} {3:4.2f},   11: {4:8.6f} {5:4.2f},   12: {6:8.6f} {7:4.2f},   2: {8:8.6f} {9:4.2f}'.format(
                    p, Nel,
                    err0[-1], order0[-1],
                    err11[-1], order11[-1],
                    err12[-1], order12[-1],
                    err2[-1], order2[-1])
                )

            if False:
                plt.figure()

        print()

        # check order of covergence
        assert order0[-1] > (p+1) - 0.6
        assert order11[-1] > (p) - 0.6
        assert order12[-1] > (p) - 0.6
        assert order2[-1] > (p) - 0.6

    # plt.show()


def test_projectors_3d(p_range=6, N_range=5):
    """
    TODO
    """

    import sys
    sys.path.append('..')

    import numpy as np
    import struphy.eigenvalue_solvers.spline_space as spl
    import matplotlib.pyplot as plt

    from struphy.eigenvalue_solvers import projectors_global as proj
    from struphy.b_splines import bspline_evaluation_3d as eval

    # test arbitrary function
    def f(eta1, eta2, eta3): return np.sin(2*np.pi*eta3/.5) * \
        np.cos(2*np.pi*eta2) * (np.exp(eta1) - 2.*np.cos(eta1/.5))

    eta1_v = np.linspace(0, 1, 80)
    eta2_v = np.linspace(0, 1, 80)
    eta3_v = np.linspace(0, 1, 80)

    ee1, ee2, ee3 = np.meshgrid(eta1_v, eta2_v, eta3_v, indexing='ij')

    print('Convergence test 3d:')
    # convergence test
    for p in range(1, p_range):
        err0 = [10.]
        err11 = [10.]
        err12 = [10.]
        err13 = [10.]
        err21 = [10.]
        err22 = [10.]
        err23 = [10.]
        err3 = [10.]

        order0 = []
        order11 = []
        order12 = []
        order13 = []
        order21 = []
        order22 = []
        order23 = []
        order3 = []
        for Nel in [2**n for n in range(4, N_range)]:

            # spline spaces
            Vh_eta1 = spl.Spline_space_1d(Nel, p, spl_kind=False)
            Vh_eta2 = spl.Spline_space_1d(Nel, p, spl_kind=True)
            Vh_eta3 = spl.Spline_space_1d(Nel, p, spl_kind=True)

            Vh_3d = spl.Tensor_spline_space([Vh_eta1, Vh_eta2, Vh_eta3])
            Vh_pol = spl.Tensor_spline_space([Vh_eta1, Vh_eta2, Vh_eta3])

            # projectors
            Vh_eta1.set_projectors()
            Vh_eta2.set_projectors()
            Vh_eta3.set_projectors()

            Vh_3d.set_projectors()
            Vh_pol.set_projectors('general')

            # A) callable as input
            cijk_0 = Vh_3d.projectors.PI_0(f)
            cijk_11, cijk_12, cijk_13 = Vh_3d.projectors.PI_1(f, f, f)
            cijk_21, cijk_22, cijk_23 = Vh_3d.projectors.PI_2(f, f, f)
            cijk_3 = Vh_3d.projectors.PI_3(f)

            # B) dofs as input
            # 1) values of f at point sets
            f_pts_0 = Vh_3d.projectors.eval_for_PI('0', f)
            f_pts_11 = Vh_3d.projectors.eval_for_PI('11', f)
            f_pts_12 = Vh_3d.projectors.eval_for_PI('12', f)
            f_pts_13 = Vh_3d.projectors.eval_for_PI('13', f)
            f_pts_21 = Vh_3d.projectors.eval_for_PI('21', f)
            f_pts_22 = Vh_3d.projectors.eval_for_PI('22', f)
            f_pts_23 = Vh_3d.projectors.eval_for_PI('23', f)
            f_pts_3 = Vh_3d.projectors.eval_for_PI('3', f)

            # 2) degrees of freedom
            dofs_0 = Vh_3d.projectors.dofs('0', f_pts_0)
            dofs_11 = Vh_3d.projectors.dofs('11', f_pts_11)
            dofs_12 = Vh_3d.projectors.dofs('12', f_pts_12)
            dofs_13 = Vh_3d.projectors.dofs('13', f_pts_13)
            dofs_21 = Vh_3d.projectors.dofs('21', f_pts_21)
            dofs_22 = Vh_3d.projectors.dofs('22', f_pts_22)
            dofs_23 = Vh_3d.projectors.dofs('23', f_pts_23)
            dofs_3 = Vh_3d.projectors.dofs('3', f_pts_3)

            # 3) fem coefficients obtained from projection
            cijk_0_mat = Vh_3d.projectors.PI_mat('0', dofs_0)
            cijk_11_mat = Vh_3d.projectors.PI_mat('11', dofs_11)
            cijk_12_mat = Vh_3d.projectors.PI_mat('12', dofs_12)
            cijk_13_mat = Vh_3d.projectors.PI_mat('13', dofs_13)
            cijk_21_mat = Vh_3d.projectors.PI_mat('21', dofs_21)
            cijk_22_mat = Vh_3d.projectors.PI_mat('22', dofs_22)
            cijk_23_mat = Vh_3d.projectors.PI_mat('23', dofs_23)
            cijk_3_mat = Vh_3d.projectors.PI_mat('3', dofs_3)

            # C) callable as input in 'general' peojectors
            temp_0 = Vh_pol.projectors.pi_0(f)
            temp_1 = Vh_pol.projectors.pi_1([f, f, f])
            temp_2 = Vh_pol.projectors.pi_2([f, f, f])
            temp_3 = Vh_pol.projectors.pi_3(f)

            cijk_0_pol = Vh_pol.extract_0(temp_0)
            cijk_11_pol, cijk_12_pol, cijk_13_pol = Vh_pol.extract_1(temp_1)
            cijk_21_pol, cijk_22_pol, cijk_23_pol = Vh_pol.extract_2(temp_2)
            cijk_3_pol = Vh_pol.extract_3(temp_3)

            # result from A) and B) must be the same
            assert np.allclose(cijk_0,   cijk_0_mat, atol=1e-14)
            assert np.allclose(cijk_11, cijk_11_mat, atol=1e-14)
            assert np.allclose(cijk_12, cijk_12_mat, atol=1e-14)
            assert np.allclose(cijk_13, cijk_13_mat, atol=1e-14)
            assert np.allclose(cijk_21, cijk_21_mat, atol=1e-14)
            assert np.allclose(cijk_22, cijk_22_mat, atol=1e-14)
            assert np.allclose(cijk_23, cijk_23_mat, atol=1e-14)
            assert np.allclose(cijk_3,   cijk_3_mat, atol=1e-14)

            # print('A) and B) yield same result.')

            # result from A) and C) must be the same
            assert np.allclose(cijk_0,   cijk_0_pol, atol=1e-13)
            assert np.allclose(cijk_11, cijk_11_pol, atol=1e-13)
            assert np.allclose(cijk_12, cijk_12_pol, atol=1e-13)
            assert np.allclose(cijk_13, cijk_13_pol, atol=1e-13)
            assert np.allclose(cijk_21, cijk_21_pol, atol=1e-13)
            assert np.allclose(cijk_22, cijk_22_pol, atol=1e-13)
            assert np.allclose(cijk_23, cijk_23_pol, atol=1e-13)
            assert np.allclose(cijk_3,   cijk_3_pol, atol=1e-13)

            # print('A) and C) yield same result.')

            # evaluation of splines:
            fh_0 = np.empty((eta1_v.size, eta2_v.size, eta3_v.size))
            fh_11 = fh_0.copy()
            fh_12 = fh_0.copy()
            fh_13 = fh_0.copy()
            fh_21 = fh_0.copy()
            fh_22 = fh_0.copy()
            fh_23 = fh_0.copy()
            fh_3 = fh_0.copy()
            f_mat = fh_0.copy()

            fh_0 = Vh_3d.evaluate_NNN(eta1_v, eta2_v, eta3_v, cijk_0)
            fh_11 = Vh_3d.evaluate_DNN(eta1_v, eta2_v, eta3_v, cijk_11)
            fh_12 = Vh_3d.evaluate_NDN(eta1_v, eta2_v, eta3_v, cijk_12)
            fh_13 = Vh_3d.evaluate_NND(eta1_v, eta2_v, eta3_v, cijk_13)
            fh_21 = Vh_3d.evaluate_NDD(eta1_v, eta2_v, eta3_v, cijk_21)
            fh_22 = Vh_3d.evaluate_DND(eta1_v, eta2_v, eta3_v, cijk_22)
            fh_23 = Vh_3d.evaluate_DDN(eta1_v, eta2_v, eta3_v, cijk_23)
            fh_3 = Vh_3d.evaluate_DDD(eta1_v, eta2_v, eta3_v, cijk_3)

            # compute error
            f_mat = f(ee1, ee2, ee3)

            err0.append(np.max(np.abs(fh_0 - f_mat)))
            err11.append(np.max(np.abs(fh_11 - f_mat)))
            err12.append(np.max(np.abs(fh_12 - f_mat)))
            err13.append(np.max(np.abs(fh_13 - f_mat)))
            err21.append(np.max(np.abs(fh_21 - f_mat)))
            err22.append(np.max(np.abs(fh_22 - f_mat)))
            err23.append(np.max(np.abs(fh_23 - f_mat)))
            err3.append(np.max(np.abs(fh_3 - f_mat)))

            order0.append(np.log2(err0[-2]/err0[-1]))
            order11.append(np.log2(err11[-2]/err11[-1]))
            order12.append(np.log2(err12[-2]/err12[-1]))
            order13.append(np.log2(err13[-2]/err13[-1]))
            order21.append(np.log2(err21[-2]/err21[-1]))
            order22.append(np.log2(err22[-2]/err22[-1]))
            order23.append(np.log2(err23[-2]/err23[-1]))
            order3.append(np.log2(err3[-2]/err3[-1]))

            #print('look  :', np.max(f_cla(eta_plot)), np.min(f_cla(eta_plot)))
            #print('look h:', np.max(f0_h_cla), np.min(f0_h_cla))

            if True:
                print('p: {0:2d}, Nel: {1:2d},   0: {2:6.4f} {3:4.2f},   11: {4:6.4f} {5:4.2f},   12: {6:6.4f} {7:4.2f},   13: {8:6.4f} {9:4.2f},   21: {10:6.4f} {11:4.2f},   22: {12:6.4f} {13:4.2f},   23: {14:6.4f} {15:4.2f},  3: {16:6.4f} {17:4.2f}'.format(
                    p, Nel,
                    err0[-1], order0[-1],
                    err11[-1], order11[-1],
                    err12[-1], order12[-1],
                    err13[-1], order13[-1],
                    err21[-1], order21[-1],
                    err22[-1], order22[-1],
                    err23[-1], order23[-1],
                    err3[-1], order3[-1])
                )

            if False:
                plt.figure()

        print()

        # check order of covergence
        assert order0[-1] > (p+1) - 0.6
        assert order11[-1] > (p) - 0.6
        assert order12[-1] > (p) - 0.6
        assert order13[-1] > (p) - 0.6
        assert order21[-1] > (p) - 0.6
        assert order22[-1] > (p) - 0.6
        assert order23[-1] > (p) - 0.6
        assert order3[-1] > (p) - 0.6

    # plt.show()


def test_project_splines():
    '''Project spline functions on various FEM spaces. 
    '''

    import os
    import sys
    sys.path.append('..')  # Because we are inside './test/' directory.

    import numpy as np

    import struphy.eigenvalue_solvers.spline_space as spl

    Nel = [16, 16, 8]
    p = [1, 1, 2]
    spl_kind = [False, True, True]
    n_quad = p.copy()
    bc = ['d', 'd']

    # 1d B-spline spline spaces for finite elements
    spaces_FEM = [spl.Spline_space_1d(Nel_i, p_i, spl_kind_i, n_quad_i, bc)
                  for Nel_i, p_i, spl_kind_i, n_quad_i in zip(Nel, p, spl_kind, n_quad)]

    # 3d tensor-product B-spline space for finite elements
    tensor_space_FEM = spl.Tensor_spline_space(spaces_FEM)

    # projectors
    spaces_FEM[0].set_projectors(n_quad[0])
    spaces_FEM[1].set_projectors(n_quad[1])
    spaces_FEM[2].set_projectors(n_quad[2])

    tensor_space_FEM.set_projectors()

    # Set extraction operators (is automatic now) and discrete derivatives
    #polar_splines = None
    # tensor_space_FEM.set_extraction_operators(bc, polar_splines) # why bc again passed?

    ########################
    # random spline in V0_h:
    ########################
    coeffs = np.random.rand(tensor_space_FEM.Ntot_0form)
    coeffs = tensor_space_FEM.extract_0(coeffs)

    def phi_0(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_NNN(eta1, eta2, eta3, coeffs)

    # project phi_0 on other spaces:
    print('projecting phi_0... ')
    c0 = tensor_space_FEM.projectors.PI_0(phi_0)
    c11, c12, c13 = tensor_space_FEM.projectors.PI_1(phi_0, phi_0, phi_0)
    c21, c22, c23 = tensor_space_FEM.projectors.PI_2(phi_0, phi_0, phi_0)
    c3 = tensor_space_FEM.projectors.PI_3(phi_0)

    assert np.allclose(c0, coeffs, atol=1e-14)

    ########################
    # random spline in V1_h:
    ########################
    coeffs = np.random.rand(tensor_space_FEM.Ntot_1form_cum[-1])
    coeffs_1, coeffs_2, coeffs_3 = tensor_space_FEM.extract_1(coeffs)

    def phi_11(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_DNN(eta1, eta2, eta3, coeffs_1)

    def phi_12(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_NDN(eta1, eta2, eta3, coeffs_2)

    def phi_13(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_NND(eta1, eta2, eta3, coeffs_3)

    # project phi_11 on other spaces:
    print('projecting phi_11... ')
    c0 = tensor_space_FEM.projectors.PI_0(phi_11)
    c11, c12, c13 = tensor_space_FEM.projectors.PI_1(phi_11, phi_11, phi_11)
    c21, c22, c23 = tensor_space_FEM.projectors.PI_2(phi_11, phi_11, phi_11)
    c3 = tensor_space_FEM.projectors.PI_3(phi_11)

    assert np.allclose(c11, coeffs_1, atol=1e-14)

    # project phi_12 on other spaces:
    print('projecting phi_12... ')
    c0 = tensor_space_FEM.projectors.PI_0(phi_12)
    c11, c12, c13 = tensor_space_FEM.projectors.PI_1(phi_12, phi_12, phi_12)
    c21, c22, c23 = tensor_space_FEM.projectors.PI_2(phi_12, phi_12, phi_12)
    c3 = tensor_space_FEM.projectors.PI_3(phi_12)

    assert np.allclose(c12, coeffs_2, atol=1e-14)

    # project phi_13 on other spaces:
    print('projecting phi_13... ')
    c0 = tensor_space_FEM.projectors.PI_0(phi_13)
    c11, c12, c13 = tensor_space_FEM.projectors.PI_1(phi_13, phi_13, phi_13)
    c21, c22, c23 = tensor_space_FEM.projectors.PI_2(phi_13, phi_13, phi_13)
    c3 = tensor_space_FEM.projectors.PI_3(phi_13)

    assert np.allclose(c13, coeffs_3, atol=1e-14)

    ########################
    # random spline in V2_h:
    ########################
    coeffs = np.random.rand(tensor_space_FEM.Ntot_2form_cum[-1])
    coeffs_1, coeffs_2, coeffs_3 = tensor_space_FEM.extract_2(coeffs)

    def phi_21(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_NDD(eta1, eta2, eta3, coeffs_1)

    def phi_22(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_DND(eta1, eta2, eta3, coeffs_2)

    def phi_23(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_DDN(eta1, eta2, eta3, coeffs_3)

    # project phi_21 on other spaces:
    print('projecting phi_21... ')
    c0 = tensor_space_FEM.projectors.PI_0(phi_21)
    c11, c12, c13 = tensor_space_FEM.projectors.PI_1(phi_21, phi_21, phi_21)
    c21, c22, c23 = tensor_space_FEM.projectors.PI_2(phi_21, phi_21, phi_21)
    c3 = tensor_space_FEM.projectors.PI_3(phi_21)

    assert np.allclose(c21, coeffs_1, atol=1e-14)

    # project phi_22 on other spaces:
    print('projecting phi_22... ')
    c0 = tensor_space_FEM.projectors.PI_0(phi_22)
    c11, c12, c13 = tensor_space_FEM.projectors.PI_1(phi_22, phi_22, phi_22)
    c21, c22, c23 = tensor_space_FEM.projectors.PI_2(phi_22, phi_22, phi_22)
    c3 = tensor_space_FEM.projectors.PI_3(phi_22)

    assert np.allclose(c22, coeffs_2, atol=1e-14)

    # project phi_23 on other spaces:
    print('projecting phi_23... ')
    c0 = tensor_space_FEM.projectors.PI_0(phi_23)
    c11, c12, c13 = tensor_space_FEM.projectors.PI_1(phi_23, phi_23, phi_23)
    c21, c22, c23 = tensor_space_FEM.projectors.PI_2(phi_23, phi_23, phi_23)
    c3 = tensor_space_FEM.projectors.PI_3(phi_23)

    assert np.allclose(c23, coeffs_3, atol=1e-14)

    ########################
    # random spline in V3_h:
    ########################
    coeffs = np.random.rand(tensor_space_FEM.Ntot_3form)
    coeffs = tensor_space_FEM.extract_3(coeffs)

    def phi_3(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_DDD(eta1, eta2, eta3, coeffs)

    # project phi_3 on other spaces:
    print('projecting phi_3... ')
    c0 = tensor_space_FEM.projectors.PI_0(phi_3)
    c11, c12, c13 = tensor_space_FEM.projectors.PI_1(phi_3, phi_3, phi_3)
    c21, c22, c23 = tensor_space_FEM.projectors.PI_2(phi_3, phi_3, phi_3)
    c3 = tensor_space_FEM.projectors.PI_3(phi_3)

    assert np.allclose(c3, coeffs, atol=1e-14)


if __name__ == '__main__':
    # test_projectors_1d(plot=True)
    test_projectors_2d()
    # test_projectors_3d()
    # test_project_splines()
