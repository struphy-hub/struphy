def test_projectors_1d():

    import sys
    sys.path.append('..')

    import numpy as np
    import hylife.utilitis_FEEC.spline_space as spl
    import matplotlib.pyplot as plt

    from hylife.utilitis_FEEC.projectors import projectors_global    as proj
    from hylife.utilitis_FEEC.basics     import spline_evaluation_1d as eval

    # test arbitrary function
    f_per = lambda eta : np.cos(2*np.pi*eta)
    f_cla = lambda eta : np.exp(eta) - 2.*np.cos(eta/.1)

    eta_v = np.linspace(0, 1, 200)

    # convergence test
    for p in range(1,4): 
        err0_per = [1.]
        err0_cla = [1.]
        err1_per = [1.]
        err1_cla = [1.]

        order0_per = []
        order0_cla = []
        order1_per = []
        order1_cla = []
        for Nel in [2**n for n in range(5,9)]:
    
            # spline spaces
            Vh_per = spl.spline_space_1d(Nel, p, spl_kind=True)
            Vh_cla = spl.spline_space_1d(Nel, p, spl_kind=False)

            # projectors
            proj_per = proj.projectors_global_1d(Vh_per, 6)
            proj_cla = proj.projectors_global_1d(Vh_cla, 6)

            dofs0_per = proj_per.rhs_0(f_per)
            dofs0_cla = proj_cla.rhs_0(f_cla)
            dofs1_per = proj_per.rhs_1(f_per)
            dofs1_cla = proj_cla.rhs_1(f_cla)

            #c0_per = proj_per.pi_0(f_per)
            #c0_cla = proj_cla.pi_0(f_cla)
            #c1_per = proj_per.pi_1(f_per)
            #c1_cla = proj_cla.pi_1(f_cla)

            c0_per = proj_per.pi_0_v2(dofs0_per)
            c0_cla = proj_cla.pi_0_v2(dofs0_cla)
            c1_per = proj_per.pi_1_v2(dofs1_per)
            c1_cla = proj_cla.pi_1_v2(dofs1_cla)

            # compute error:
            f0_h_per = []
            f0_h_cla = []
            f1_h_per = []
            f1_h_cla = []
            for eta in eta_v:
                f0_h_per.append(eval.evaluate_n(Vh_per.T, Vh_per.p,     Vh_per.NbaseN, c0_per, eta))
                f0_h_cla.append(eval.evaluate_n(Vh_cla.T, Vh_cla.p,     Vh_cla.NbaseN, c0_cla, eta))
                f1_h_per.append(eval.evaluate_d(Vh_per.t, Vh_per.p - 1, Vh_per.NbaseD, c1_per, eta))
                f1_h_cla.append(eval.evaluate_d(Vh_cla.t, Vh_cla.p - 1, Vh_cla.NbaseD, c1_cla, eta))

            err0_per.append(np.abs(np.max(f0_h_per - f_per(eta_v))))
            err0_cla.append(np.abs(np.max(f0_h_cla - f_cla(eta_v))))
            err1_per.append(np.abs(np.max(f1_h_per - f_per(eta_v))))
            err1_cla.append(np.abs(np.max(f1_h_cla - f_cla(eta_v))))

            order0_per.append(np.log2(err0_per[-2]/err0_per[-1]))
            order0_cla.append(np.log2(err0_cla[-2]/err0_cla[-1]))
            order1_per.append(np.log2(err1_per[-2]/err1_per[-1]))
            order1_cla.append(np.log2(err1_cla[-2]/err1_cla[-1]))

            #print('look  :', np.max(f_cla(eta_v)), np.min(f_cla(eta_v)))
            #print('look h:', np.max(f0_h_cla), np.min(f0_h_cla))

            print('p: {0:2d}, Nel: {1:4d},   0_per: {2:8.6f} {3:4.2f},   0_cla: {4:8.6f} {5:4.2f},   1_per: {6:8.6f} {7:4.2f},   1_cla: {8:8.6f} {9:4.2f}'.format(
                p, Nel,
                err0_per[-1], order0_per[-1],
                err0_cla[-1], order0_cla[-1],
                err1_per[-1], order1_per[-1],
                err1_cla[-1], order1_cla[-1])
                )

            if p<4 and Nel==32:
                plt.figure()
                plt.plot(eta_v, f_per(eta_v), 'r', label='per')
                plt.plot(eta_v, f_cla(eta_v), 'r', label='cla')
                plt.plot(eta_v, f0_h_per, 'b--', label='f0h_per')
                plt.plot(eta_v, f0_h_cla, 'm--', label='f0h_cla')
                plt.plot(eta_v, f1_h_per, 'b--', label='f1h_per')
                plt.plot(eta_v, f1_h_cla, 'g--', label='f1h_cla')
                plt.title('p: {0:2d}, Nel: {1:4d}'.format(p, Nel))
                plt.legend()
                plt.autoscale(enable=True, axis='x', tight=True)

        assert np.all(order0_per[1:] > (p+1)*np.ones(len(order0_per) - 1) - 0.15 )
        assert np.all(order0_cla[1:] > (p+1)*np.ones(len(order0_per) - 1) - 0.15 )
        assert np.all(order1_per[1:] >   (p)*np.ones(len(order0_per) - 1) - 0.15 )
        assert np.all(order1_cla[1:] >   (p)*np.ones(len(order0_per) - 1) - 0.15 )

    #plt.show()


if __name__ == '__main__':
    test_projectors_1d()