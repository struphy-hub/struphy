def test_projectors_1d():
    '''
    1) test order of convergence
    2) test projector property, i.e. Pi(basis_function) = basis_function
    '''

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

    print('1) convergence test:')
    # 1) convergence test
    for p in range(1,2): 
        err0_per = [1.]
        err0_cla = [1.]
        err1_per = [1.]
        err1_cla = [1.]

        order0_per = []
        order0_cla = []
        order1_per = []
        order1_cla = []
        for Nel in [2**n for n in range(5,6)]:
    
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

            if True:
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

    print('2) projector test:')
    # 2) projector test
    for p in range(1,9):

        # spline spaces
        Vh_per = spl.spline_space_1d(32, p, spl_kind=True)
        Vh_cla = spl.spline_space_1d(32, p, spl_kind=False)

        for nq in range(1,6):

            # projectors
            proj_per = proj.projectors_global_1d(Vh_per, nq)
            proj_cla = proj.projectors_global_1d(Vh_cla, nq)

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
                def f(eta):
                    values = np.empty(eta.shape)
                    eval.evaluate_vector(Vh_per.T, Vh_per.p, Vh_per.NbaseN, f0_per, eta, values, 0)
                    return values
                dofs0_per = proj_per.dofs_0(f)
                c0_per = proj_per.pi_0_v2(dofs0_per)

                if np.max(np.abs(f0_per - c0_per)) > err0_per[0]:
                   err0_per = [np.max(np.abs(f0_per - c0_per)), i]
                #assert np.allclose(f0_per, c0_per, atol=1e-12), 'Basis function {0:2d} failed.'.format(i)
                f0_per[i] = 0.

            # basis functions V1 periodic
            for i in range(Vh_per.NbaseD): 
                f1_per[i] = 1.
                def f(eta):
                    values = np.empty(eta.shape)
                    eval.evaluate_vector(Vh_per.t, Vh_per.p - 1, Vh_per.NbaseD, f1_per, eta, values, 1)
                    return values
                dofs1_per = proj_per.dofs_1(f)
                c1_per = proj_per.pi_1_v2(dofs1_per)

                if np.max(np.abs(f1_per - c1_per)) > err1_per[0]:
                   err1_per = [np.max(np.abs(f1_per - c1_per)), i]
                #assert np.allclose(f1_per, c1_per, atol=1e-12), 'Basis function {0:2d} failed.'.format(i)
                f1_per[i] = 0.

            # basis functions V0 clamped
            for i in range(Vh_cla.NbaseN): 
                f0_cla[i] = 1.
                def f(eta):
                    values = np.empty(eta.shape)
                    eval.evaluate_vector(Vh_cla.T, Vh_cla.p, Vh_cla.NbaseN, f0_cla, eta, values, 0)
                    return values
                dofs0_cla = proj_cla.dofs_0(f)
                c0_cla = proj_cla.pi_0_v2(dofs0_cla)

                if np.max(np.abs(f0_cla - c0_cla)) > err0_cla[0]:
                   err0_cla = [np.max(np.abs(f0_cla - c0_cla)), i]
                #assert np.allclose(f0_cla, c0_cla, atol=1e-12), 'Basis function {0:2d} failed.'.format(i)
                f0_cla[i] = 0.

            # basis functions V1 clamped
            for i in range(Vh_cla.NbaseD): 
                f1_cla[i] = 1.
                def f(eta):
                    values = np.empty(eta.shape)
                    eval.evaluate_vector(Vh_cla.t, Vh_cla.p - 1, Vh_cla.NbaseD, f1_cla, eta, values, 1)
                    return values
                dofs1_cla = proj_cla.dofs_1(f)
                c1_cla = proj_cla.pi_1_v2(dofs1_cla)

                if np.max(np.abs(f1_cla - c1_cla)) > err1_cla[0]:
                   err1_cla = [np.max(np.abs(f1_cla - c1_cla)), i]
                #assert np.allclose(f1_cla, c1_cla, atol=1e-12), 'Basis function {0:2d} failed.'.format(i)
                f1_cla[i] = 0.

            print('p: {0:2d}, nq: {1:2d},   maxerr V0_per: {2:4.2e} at i={3:2d},   maxerr V1_per: {4:4.2e} at i={5:2d},   maxerr V0_cla: {4:4.2e} at i={5:2d},   maxerr V1_cla: {4:4.2e} at i={5:2d}'.format(
                p, nq,
                err0_per[0], err0_per[1],
                err1_per[0], err1_per[1],
                err0_cla[0], err0_cla[1],
                err1_cla[0], err1_cla[1])
                ) 

        print()

    print() 
            





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

            if False:
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

        assert np.all(order0[1:]  > (p+1)*np.ones(len(order0) - 1) - 0.3 )
        assert np.all(order11[1:] > (p)*np.ones(len(order0) - 1) - 0.3 )
        assert np.all(order12[1:] > (p)*np.ones(len(order0) - 1) - 0.3 )
        assert np.all(order2[1:]  > (p)*np.ones(len(order0) - 1) - 0.3 )

    #plt.show()



def test_projectors_3d():

    import sys
    sys.path.append('..')

    import numpy as np
    import hylife.utilitis_FEEC.spline_space as spl
    import matplotlib.pyplot as plt

    from hylife.utilitis_FEEC.projectors import projectors_global    as proj
    from hylife.utilitis_FEEC.basics     import spline_evaluation_3d as eval

    # test arbitrary function
    f = lambda eta1, eta2, eta3 : np.sin(2*np.pi*eta3/.5) * np.cos(2*np.pi*eta2) * ( np.exp(eta1) - 2.*np.cos(eta1/.1) )

    eta1_v = np.linspace(0, 1, 80)
    eta2_v = np.linspace(0, 1, 80)
    eta3_v = np.linspace(0, 1, 80)

    # convergence test
    for p in range(1,4): 
        err0  = [10.]
        err11 = [10.]
        err12 = [10.]
        err13 = [10.]
        err21 = [10.]
        err22 = [10.]
        err23 = [10.]
        err3  = [10.]

        order0  = []
        order11 = []
        order12 = []
        order13 = []
        order21 = []
        order22 = []
        order23 = []
        order3  = []
        for Nel in [10, 20, 40]:
    
            # spline spaces
            Vh_eta1 = spl.spline_space_1d(Nel, p, spl_kind=False)
            Vh_eta2 = spl.spline_space_1d(Nel, p, spl_kind=True)
            Vh_eta3 = spl.spline_space_1d(Nel, p, spl_kind=True)

            # projectors
            proj_eta1 = proj.projectors_global_1d(Vh_eta1, 6)
            proj_eta2 = proj.projectors_global_1d(Vh_eta2, 6)
            proj_eta3 = proj.projectors_global_1d(Vh_eta3, 6)

            proj_3d = proj.projectors_tensor_3d([proj_eta1, proj_eta2, proj_eta3])

            # values of f at point sets
            f_pts_0  = proj_3d.eval_for_PI('0', f)  
            f_pts_11 = proj_3d.eval_for_PI('11', f)  
            f_pts_12 = proj_3d.eval_for_PI('12', f)
            f_pts_13 = proj_3d.eval_for_PI('13', f)
            f_pts_21 = proj_3d.eval_for_PI('21', f)
            f_pts_22 = proj_3d.eval_for_PI('22', f)
            f_pts_23 = proj_3d.eval_for_PI('23', f)  
            f_pts_3  = proj_3d.eval_for_PI('3', f)    
            
            # degrees of freedom 
            dofs_0  = proj_3d.dofs('0', f_pts_0)
            dofs_11 = proj_3d.dofs('11', f_pts_11)
            dofs_12 = proj_3d.dofs('12', f_pts_12)
            dofs_13 = proj_3d.dofs('13', f_pts_13)
            dofs_21 = proj_3d.dofs('21', f_pts_21)
            dofs_22 = proj_3d.dofs('22', f_pts_22)
            dofs_23 = proj_3d.dofs('23', f_pts_23)
            dofs_3  = proj_3d.dofs('3', f_pts_3)

            # fem coefficients obtained from projection
            cijk_0  = proj_3d.PI_mat('0', dofs_0)
            cijk_11 = proj_3d.PI_mat('11', dofs_11)
            cijk_12 = proj_3d.PI_mat('12', dofs_12)
            cijk_13 = proj_3d.PI_mat('13', dofs_13)
            cijk_21 = proj_3d.PI_mat('21', dofs_21)
            cijk_22 = proj_3d.PI_mat('22', dofs_22)
            cijk_23 = proj_3d.PI_mat('23', dofs_23)
            cijk_3  = proj_3d.PI_mat('3', dofs_3)

            # compute error:
            fh_0  = np.empty((eta1_v.size, eta2_v.size, eta3_v.size))
            fh_11 = fh_0.copy()
            fh_12 = fh_0.copy()
            fh_13 = fh_0.copy()
            fh_21 = fh_0.copy()
            fh_22 = fh_0.copy()
            fh_23 = fh_0.copy()
            fh_3  = fh_0.copy()
            f_mat = fh_0.copy()

            # slow loops:
            for i, eta1 in enumerate(eta1_v):
                for j, eta2 in enumerate(eta2_v):
                    for k, eta3 in enumerate(eta3_v):

                        fh_0[i, j, k]   = eval.evaluate_n_n_n(Vh_eta1.T, Vh_eta2.T, Vh_eta3.T,
                                                              Vh_eta1.p, Vh_eta2.p, Vh_eta3.p,
                                                              Vh_eta1.NbaseN, Vh_eta2.NbaseN, Vh_eta3.NbaseN,
                                                              cijk_0, eta1, eta2, eta3)

                        fh_11[i, j, k]  = eval.evaluate_d_n_n(Vh_eta1.t, Vh_eta2.T, Vh_eta3.T,
                                                              Vh_eta1.p - 1, Vh_eta2.p, Vh_eta3.p,
                                                              Vh_eta1.NbaseD, Vh_eta2.NbaseN, Vh_eta3.NbaseN,
                                                              cijk_11, eta1, eta2, eta3)

                        fh_12[i, j, k]  = eval.evaluate_n_d_n(Vh_eta1.T, Vh_eta2.t, Vh_eta3.T,
                                                              Vh_eta1.p, Vh_eta2.p - 1, Vh_eta3.p,
                                                              Vh_eta1.NbaseN, Vh_eta2.NbaseD, Vh_eta3.NbaseN,
                                                              cijk_12, eta1, eta2, eta3)

                        fh_13[i, j, k]  = eval.evaluate_n_n_d(Vh_eta1.T, Vh_eta2.T, Vh_eta3.t,
                                                              Vh_eta1.p, Vh_eta2.p, Vh_eta3.p - 1,
                                                              Vh_eta1.NbaseN, Vh_eta2.NbaseN, Vh_eta3.NbaseD,
                                                              cijk_13, eta1, eta2, eta3)

                        fh_21[i, j, k]  = eval.evaluate_n_d_d(Vh_eta1.T, Vh_eta2.t, Vh_eta3.t,
                                                              Vh_eta1.p, Vh_eta2.p - 1, Vh_eta3.p - 1,
                                                              Vh_eta1.NbaseN, Vh_eta2.NbaseD, Vh_eta3.NbaseD,
                                                              cijk_21, eta1, eta2, eta3) 

                        fh_22[i, j, k]  = eval.evaluate_d_n_d(Vh_eta1.t, Vh_eta2.T, Vh_eta3.t,
                                                              Vh_eta1.p - 1, Vh_eta2.p, Vh_eta3.p - 1,
                                                              Vh_eta1.NbaseD, Vh_eta2.NbaseN, Vh_eta3.NbaseD,
                                                              cijk_22, eta1, eta2, eta3) 

                        fh_23[i, j, k]  = eval.evaluate_d_d_n(Vh_eta1.t, Vh_eta2.t, Vh_eta3.T,
                                                              Vh_eta1.p - 1, Vh_eta2.p - 1, Vh_eta3.p,
                                                              Vh_eta1.NbaseD, Vh_eta2.NbaseD, Vh_eta3.NbaseN,
                                                              cijk_23, eta1, eta2, eta3) 

                        fh_3[i, j, k]   = eval.evaluate_d_d_d(Vh_eta1.t, Vh_eta2.t, Vh_eta3.t,
                                                              Vh_eta1.p - 1, Vh_eta2.p - 1, Vh_eta3.p - 1,
                                                              Vh_eta1.NbaseD, Vh_eta2.NbaseD, Vh_eta3.NbaseD,
                                                              cijk_3, eta1, eta2, eta3)     

                        f_mat[i, j, k] = f(eta1, eta2, eta3)

            err0.append(np.abs(np.max(fh_0 - f_mat)))
            err11.append(np.abs(np.max(fh_11 - f_mat)))
            err12.append(np.abs(np.max(fh_12 - f_mat)))
            err13.append(np.abs(np.max(fh_13 - f_mat)))
            err21.append(np.abs(np.max(fh_21 - f_mat)))
            err22.append(np.abs(np.max(fh_22 - f_mat)))
            err23.append(np.abs(np.max(fh_23 - f_mat)))
            err3.append(np.abs(np.max(fh_3 - f_mat)))

            order0.append(np.log2(err0[-2]/err0[-1]))
            order11.append(np.log2(err11[-2]/err11[-1]))
            order12.append(np.log2(err12[-2]/err12[-1]))
            order13.append(np.log2(err13[-2]/err13[-1]))
            order21.append(np.log2(err21[-2]/err21[-1]))
            order22.append(np.log2(err22[-2]/err22[-1]))
            order23.append(np.log2(err23[-2]/err23[-1]))
            order3.append(np.log2(err3[-2]/err3[-1]))

            #print('look  :', np.max(f_cla(eta_v)), np.min(f_cla(eta_v)))
            #print('look h:', np.max(f0_h_cla), np.min(f0_h_cla))

            if False:
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

        assert np.all(order0[1:]  > (p+1)*np.ones(len(order0) - 1) - 0.3 )
        assert np.all(order11[1:] > (p)*np.ones(len(order0) - 1) - 0.3 )
        assert np.all(order12[1:] > (p)*np.ones(len(order0) - 1) - 0.3 )
        assert np.all(order13[1:] > (p)*np.ones(len(order0) - 1) - 0.3 )
        assert np.all(order21[1:] > (p)*np.ones(len(order0) - 1) - 0.3 )
        assert np.all(order22[1:] > (p)*np.ones(len(order0) - 1) - 0.3 )
        assert np.all(order23[1:] > (p)*np.ones(len(order0) - 1) - 0.3 )
        assert np.all(order3[1:]  > (p)*np.ones(len(order0) - 1) - 0.3 )

    #plt.show()



if __name__ == '__main__':
    test_projectors_1d()
    #test_projectors_2d()
    #test_projectors_3d()