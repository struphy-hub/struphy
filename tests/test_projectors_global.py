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

            dofs0_per = proj_per.dofs_0(f_per)
            dofs0_cla = proj_cla.dofs_0(f_cla)
            dofs1_per = proj_per.dofs_1(f_per)
            dofs1_cla = proj_cla.dofs_1(f_cla)

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

            if False:
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

        print()

        assert np.all(order0_per[1:] > (p+1)*np.ones(len(order0_per) - 1) - 0.15 )
        assert np.all(order0_cla[1:] > (p+1)*np.ones(len(order0_per) - 1) - 0.15 )
        assert np.all(order1_per[1:] >   (p)*np.ones(len(order0_per) - 1) - 0.15 )
        assert np.all(order1_cla[1:] >   (p)*np.ones(len(order0_per) - 1) - 0.15 )

    #plt.show()



def test_projectors_2d():

    import sys
    sys.path.append('..')

    import numpy as np
    import hylife.utilitis_FEEC.spline_space as spl
    import matplotlib.pyplot as plt

    from hylife.utilitis_FEEC.projectors import projectors_global    as proj
    from hylife.utilitis_FEEC.basics     import spline_evaluation_2d as eval

    # test arbitrary function
    f = lambda eta1, eta2 : np.cos(2*np.pi*eta2) * ( np.exp(eta1) - 2.*np.cos(eta1/.1) )

    eta1_v = np.linspace(0, 1, 100)
    eta2_v = np.linspace(0, 1, 100)

    # convergence test
    for p in range(1,4): 
        err0  = [1.]
        err11 = [1.]
        err12 = [1.]
        err2  = [1.]

        order0  = []
        order11 = []
        order12 = []
        order2  = []
        for Nel in [2**n for n in range(3,9)]:
    
            # spline spaces
            Vh_eta1 = spl.spline_space_1d(Nel, p, spl_kind=False)
            Vh_eta2 = spl.spline_space_1d(Nel, p, spl_kind=True)

            # projectors
            proj_eta1 = proj.projectors_global_1d(Vh_eta1, 6)
            proj_eta2 = proj.projectors_global_1d(Vh_eta2, 6)

            proj_2d = proj.projectors_tensor_2d([proj_eta1, proj_eta2])

            # values of f at point sets
            f_pts_0  = proj_2d.eval_for_PI('0', f)  
            f_pts_11 = proj_2d.eval_for_PI('11', f)  
            f_pts_12 = proj_2d.eval_for_PI('12', f)  
            f_pts_2  = proj_2d.eval_for_PI('2', f)    
            
            # degrees of freedom 
            dofs_0  = proj_2d.dofs('0', f_pts_0)
            dofs_11 = proj_2d.dofs('11', f_pts_11)
            dofs_12 = proj_2d.dofs('12', f_pts_12)
            dofs_2  = proj_2d.dofs('2', f_pts_2)

            # fem coefficients obtained from projection
            cij_0  = proj_2d.PI_mat('0', dofs_0)
            cij_11 = proj_2d.PI_mat('11', dofs_11)
            cij_12 = proj_2d.PI_mat('12', dofs_12)
            cij_2  = proj_2d.PI_mat('2', dofs_2)

            # compute error:
            fh_0  = np.empty((eta1_v.size, eta2_v.size))
            fh_11 = fh_0.copy()
            fh_12 = fh_0.copy()
            fh_2  = fh_0.copy()
            f_mat = fh_0.copy()

            # slow loops:
            for i, eta1 in enumerate(eta1_v):
                for j, eta2 in enumerate(eta2_v):

                    fh_0[i, j]  = eval.evaluate_n_n(Vh_eta1.T, Vh_eta2.T, Vh_eta1.p, Vh_eta2.p,
                                             Vh_eta1.NbaseN, Vh_eta2.NbaseN, cij_0, eta1, eta2)

                    fh_11[i, j] = eval.evaluate_d_n(Vh_eta1.t, Vh_eta2.T, Vh_eta1.p - 1, Vh_eta2.p,
                                             Vh_eta1.NbaseD, Vh_eta2.NbaseN, cij_11, eta1, eta2)

                    fh_12[i, j] = eval.evaluate_n_d(Vh_eta1.T, Vh_eta2.t, Vh_eta1.p, Vh_eta2.p - 1,
                                             Vh_eta1.NbaseN, Vh_eta2.NbaseD, cij_12, eta1, eta2)

                    fh_2[i, j]  = eval.evaluate_d_d(Vh_eta1.t, Vh_eta2.t, Vh_eta1.p - 1, Vh_eta2.p - 1,
                                             Vh_eta1.NbaseD, Vh_eta2.NbaseD, cij_2, eta1, eta2)

                    f_mat[i, j] = f(eta1, eta2)

            err0.append(np.abs(np.max(fh_0 - f_mat)))
            err11.append(np.abs(np.max(fh_11 - f_mat)))
            err12.append(np.abs(np.max(fh_12 - f_mat)))
            err2.append(np.abs(np.max(fh_2 - f_mat)))

            order0.append(np.log2(err0[-2]/err0[-1]))
            order11.append(np.log2(err11[-2]/err11[-1]))
            order12.append(np.log2(err12[-2]/err12[-1]))
            order2.append(np.log2(err2[-2]/err2[-1]))

            #print('look  :', np.max(f_cla(eta_v)), np.min(f_cla(eta_v)))
            #print('look h:', np.max(f0_h_cla), np.min(f0_h_cla))

            print('p: {0:2d}, Nel: {1:4d},   0: {2:8.6f} {3:4.2f},   11: {4:8.6f} {5:4.2f},   12: {6:8.6f} {7:4.2f},   2: {8:8.6f} {9:4.2f}'.format(
                p, Nel,
                err0[-1], order0[-1],
                err11[-1], order11[-1],
                err12[-1], order12[-1],
                err2[-1], order2[-1])
                )

            if False:
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

        print()

        assert np.all(order0[1:]  > (p+1)*np.ones(len(order0) - 1) - 0.3 )
        assert np.all(order11[1:] > (p)*np.ones(len(order0) - 1) - 0.3 )
        assert np.all(order12[1:] > (p)*np.ones(len(order0) - 1) - 0.3 )
        assert np.all(order2[1:]  > (p)*np.ones(len(order0) - 1) - 0.3 )

    #plt.show()


if __name__ == '__main__':
    test_projectors_1d()
    test_projectors_2d()