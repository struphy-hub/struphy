from matplotlib.pyplot import subplot


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
    
    # test arbitrary function
    f_per = lambda eta : np.cos(2*np.pi*eta/.2)
    f_cla = lambda eta : np.exp(2.*eta) - 2.*np.cos(2*np.pi*eta/.2)

    eta_v = np.linspace(0, 1, 200)

    fig, axs = plt.subplots(2, 2)

    print('1) convergence test:')
    # 1) convergence test
    for p in range(1,7): 
        err0_per = [10.]
        err0_cla = [10.]
        err1_per = [10.]
        err1_cla = [10.]

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

            # callable as input
            c0_per = proj_per.pi_0(f_per)
            c0_cla = proj_cla.pi_0(f_cla)
            c1_per = proj_per.pi_1(f_per)
            c1_cla = proj_cla.pi_1(f_cla)

            # dofs as input
            dofs0_per = proj_per.dofs_0(f_per)
            dofs0_cla = proj_cla.dofs_0(f_cla)
            dofs1_per = proj_per.dofs_1(f_per)
            dofs1_cla = proj_cla.dofs_1(f_cla)

            c0_per_mat = proj_per.pi_0_mat(dofs0_per)
            c0_cla_mat = proj_cla.pi_0_mat(dofs0_cla)
            c1_per_mat = proj_per.pi_1_mat(dofs1_per)
            c1_cla_mat = proj_cla.pi_1_mat(dofs1_cla)

            # result must be the same
            assert np.allclose(c0_per, c0_per_mat, atol=1e-14)
            assert np.allclose(c0_cla, c0_cla_mat, atol=1e-14)
            assert np.allclose(c1_per, c1_per_mat, atol=1e-14)
            assert np.allclose(c1_cla, c1_cla_mat, atol=1e-14)

            # compute error:
            f0_h_per = Vh_per.evaluate_N(eta_v, c0_per)
            f0_h_cla = Vh_cla.evaluate_N(eta_v, c0_cla)
            f1_h_per = Vh_per.evaluate_D(eta_v, c1_per)
            f1_h_cla = Vh_cla.evaluate_D(eta_v, c1_cla)
            
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

            if p<3 and Nel==2**5:
                axs.flatten()[2*p-2].plot(eta_v, f_per(eta_v), 'r', label='f')
                axs.flatten()[2*p-2].plot(eta_v, f0_h_per, 'b--', label='f0h_per')
                axs.flatten()[2*p-2].plot(eta_v, f1_h_per, 'k--', label='f1h_per')
                axs.flatten()[2*p-2].set_title('p: {0:2d}, Nel: {1:4d}, periodic'.format(p, Nel))
                axs.flatten()[2*p-2].legend()
                axs.flatten()[2*p-2].autoscale(enable=True, axis='x', tight=True)

                axs.flatten()[2*p-1].plot(eta_v, f_cla(eta_v), 'r', label='f')
                axs.flatten()[2*p-1].plot(eta_v, f0_h_cla, 'b--', label='f0h_cla')
                axs.flatten()[2*p-1].plot(eta_v, f1_h_cla, 'k--', label='f1h_cla')
                axs.flatten()[2*p-1].set_title('p: {0:2d}, Nel: {1:4d}, non-periodic'.format(p, Nel))
                axs.flatten()[2*p-1].legend()
                axs.flatten()[2*p-1].autoscale(enable=True, axis='x', tight=True)

        print()

        assert np.all(order0_per[1:] > (p+1)*np.ones(len(order0_per) - 1) - 0.3 )
        assert np.all(order0_cla[1:] > (p+1)*np.ones(len(order0_per) - 1) - 0.3 )
        assert np.all(order1_per[1:] >   (p)*np.ones(len(order0_per) - 1) - 0.3 )
        assert np.all(order1_cla[1:] >   (p)*np.ones(len(order0_per) - 1) - 0.3 )


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
                
                f = lambda eta : Vh_per.evaluate_N(eta, f0_per)
                
                # callable as input
                c0_per = proj_per.pi_0(f)

                # dofs as input
                dofs0_per = proj_per.dofs_0(f)
                c0_per_mat = proj_per.pi_0_mat(dofs0_per)

                # result must be the same
                assert np.allclose(c0_per, c0_per_mat, atol=1e-14)

                if np.max(np.abs(f0_per - c0_per)) > err0_per[0]:
                    err0_per = [np.max(np.abs(f0_per - c0_per)), i]
                #assert np.allclose(f0_per, c0_per, atol=1e-12), 'Basis function {0:2d} failed.'.format(i)
                f0_per[i] = 0.

            # basis functions V1 periodic
            for i in range(Vh_per.NbaseD): 
                f1_per[i] = 1.
                
                f = lambda eta : Vh_per.evaluate_D(eta, f1_per)
                
                # callable as input
                c1_per = proj_per.pi_1(f)

                # dofs as input
                dofs1_per = proj_per.dofs_1(f)
                c1_per_mat = proj_per.pi_1_mat(dofs1_per)

                # result must be the same
                assert np.allclose(c1_per, c1_per_mat, atol=1e-14)

                if np.max(np.abs(f1_per - c1_per)) > err1_per[0]:
                    err1_per = [np.max(np.abs(f1_per - c1_per)), i]
                #assert np.allclose(f1_per, c1_per, atol=1e-12), 'Basis function {0:2d} failed.'.format(i)
                f1_per[i] = 0.

            # basis functions V0 clamped
            for i in range(Vh_cla.NbaseN): 
                f0_cla[i] = 1.
                
                f = lambda eta : Vh_cla.evaluate_N(eta, f0_cla)
                
                # callable as input
                c0_cla = proj_cla.pi_0(f)

                # dofs as input
                dofs0_cla = proj_cla.dofs_0(f)
                c0_cla_mat = proj_cla.pi_0_mat(dofs0_cla)

                # result must be the same
                assert np.allclose(c0_cla, c0_cla_mat, atol=1e-14)

                if np.max(np.abs(f0_cla - c0_cla)) > err0_cla[0]:
                    err0_cla = [np.max(np.abs(f0_cla - c0_cla)), i]
                #assert np.allclose(f0_cla, c0_cla, atol=1e-12), 'Basis function {0:2d} failed.'.format(i)
                f0_cla[i] = 0.

            # basis functions V1 clamped
            for i in range(Vh_cla.NbaseD): 
                f1_cla[i] = 1.
                
                f = lambda eta : Vh_cla.evaluate_D(eta, f1_cla)
                
                # callable as input
                c1_cla = proj_cla.pi_1(f)

                # dofs as input
                dofs1_cla = proj_cla.dofs_1(f)
                c1_cla_mat = proj_cla.pi_1_mat(dofs1_cla)

                # result must be the same
                assert np.allclose(c1_cla, c1_cla_mat, atol=1e-14)

                if np.max(np.abs(f1_cla - c1_cla)) > err1_cla[0]:
                    err1_cla = [np.max(np.abs(f1_cla - c1_cla)), i]
                #assert np.allclose(f1_cla, c1_cla, atol=1e-12), 'Basis function {0:2d} failed.'.format(i)
                f1_cla[i] = 0.

            print('p: {0:2d}, nq: {1:2d},   maxerr V0_per: {2:4.2e} at i={3:2d},   maxerr V1_per: {4:4.2e} at i={5:2d},   maxerr V0_cla: {6:4.2e} at i={7:2d},   maxerr V1_cla: {8:4.2e} at i={9:2d}'.format(
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
    f = lambda eta1, eta2 : np.cos(2*np.pi*eta2/.2) * ( np.exp(2.*eta1) - 2.*np.cos(2*np.pi*eta1/.2) )

    eta1_v   = np.linspace(0, 1, 100)
    eta2_v   = np.linspace(0, 1, 100)
    ee1, ee2 = np.meshgrid(eta1_v, eta2_v, indexing='ij')

    print('Convergence test 2d:')
    # convergence test
    for p in range(1,5): 
        err0  = [1.]
        err11 = [1.]
        err12 = [1.]
        err2  = [1.]

        order0  = []
        order11 = []
        order12 = []
        order2  = []
        for Nel in [2**n for n in range(4,9)]:
    
            # spline spaces
            Vh_eta1 = spl.spline_space_1d(Nel, p, spl_kind=False)
            Vh_eta2 = spl.spline_space_1d(Nel, p, spl_kind=True)

            Vh_tensor = spl.tensor_spline_space([Vh_eta1, Vh_eta2]) # needed only for evaluation of splines

            # projectors
            proj_eta1 = proj.projectors_global_1d(Vh_eta1, 6)
            proj_eta2 = proj.projectors_global_1d(Vh_eta2, 6)

            proj_2d = proj.projectors_tensor_2d([proj_eta1, proj_eta2])

            # A) callable as input
            cij_0          = proj_2d.PI_0(f)
            cij_11, cij_12 = proj_2d.PI_1(f, f)
            cij_2          = proj_2d.PI_2(f)

            # B) dofs as input
            # 1) values of f at point sets
            f_pts_0  = proj_2d.eval_for_PI('0', f)  
            f_pts_11 = proj_2d.eval_for_PI('11', f)  
            f_pts_12 = proj_2d.eval_for_PI('12', f)  
            f_pts_2  = proj_2d.eval_for_PI('2', f)    
            
            # 2) degrees of freedom 
            dofs_0  = proj_2d.dofs('0', f_pts_0)
            dofs_11 = proj_2d.dofs('11', f_pts_11)
            dofs_12 = proj_2d.dofs('12', f_pts_12)
            dofs_2  = proj_2d.dofs('2', f_pts_2)

            # 3) fem coefficients obtained from projection
            cij_0_mat  = proj_2d.PI_mat('0', dofs_0)
            cij_11_mat = proj_2d.PI_mat('11', dofs_11)
            cij_12_mat = proj_2d.PI_mat('12', dofs_12)
            cij_2_mat  = proj_2d.PI_mat('2', dofs_2)

            # result from A) and B) must be the same
            assert np.allclose(cij_0, cij_0_mat, atol=1e-14)
            assert np.allclose(cij_11, cij_11_mat, atol=1e-14)
            assert np.allclose(cij_12, cij_12_mat, atol=1e-14)
            assert np.allclose(cij_2, cij_2_mat, atol=1e-14)

            # evaluation of splines:
            fh_0  = np.empty((eta1_v.size, eta2_v.size))
            fh_11 = fh_0.copy()
            fh_12 = fh_0.copy()
            fh_2  = fh_0.copy()
            f_mat = fh_0.copy()

            fh_0  = Vh_tensor.evaluate_NN(eta1_v, eta2_v, cij_0)
            fh_11 = Vh_tensor.evaluate_DN(eta1_v, eta2_v, cij_11)
            fh_12 = Vh_tensor.evaluate_ND(eta1_v, eta2_v, cij_12)
            fh_2  = Vh_tensor.evaluate_DD(eta1_v, eta2_v, cij_2)

            # compute error:
            f_mat = f(ee1, ee2)

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

    ee1, ee2, ee3 = np.meshgrid(eta1_v, eta2_v, eta3_v, indexing='ij')

    print('Convergence test 3d:')
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

            Vh_tensor = spl.tensor_spline_space([Vh_eta1, Vh_eta2, Vh_eta3]) # needed only for evaluation of splines

            # projectors
            proj_eta1 = proj.projectors_global_1d(Vh_eta1, 6)
            proj_eta2 = proj.projectors_global_1d(Vh_eta2, 6)
            proj_eta3 = proj.projectors_global_1d(Vh_eta3, 6)

            proj_3d = proj.projectors_tensor_3d([proj_eta1, proj_eta2, proj_eta3])

            # A) callable as input
            cijk_0                    = proj_3d.PI_0(f)
            cijk_11, cijk_12, cijk_13 = proj_3d.PI_1(f, f, f)
            cijk_21, cijk_22, cijk_23 = proj_3d.PI_2(f, f, f)
            cijk_3                    = proj_3d.PI_3(f)

            # B) dofs as input
            # 1) values of f at point sets
            f_pts_0  = proj_3d.eval_for_PI('0', f)  
            f_pts_11 = proj_3d.eval_for_PI('11', f)  
            f_pts_12 = proj_3d.eval_for_PI('12', f)
            f_pts_13 = proj_3d.eval_for_PI('13', f)
            f_pts_21 = proj_3d.eval_for_PI('21', f)
            f_pts_22 = proj_3d.eval_for_PI('22', f)
            f_pts_23 = proj_3d.eval_for_PI('23', f)  
            f_pts_3  = proj_3d.eval_for_PI('3', f)    
            
            # 2) degrees of freedom 
            dofs_0  = proj_3d.dofs('0', f_pts_0)
            dofs_11 = proj_3d.dofs('11', f_pts_11)
            dofs_12 = proj_3d.dofs('12', f_pts_12)
            dofs_13 = proj_3d.dofs('13', f_pts_13)
            dofs_21 = proj_3d.dofs('21', f_pts_21)
            dofs_22 = proj_3d.dofs('22', f_pts_22)
            dofs_23 = proj_3d.dofs('23', f_pts_23)
            dofs_3  = proj_3d.dofs('3', f_pts_3)

            # 3) fem coefficients obtained from projection
            cijk_0_mat  = proj_3d.PI_mat('0', dofs_0)
            cijk_11_mat = proj_3d.PI_mat('11', dofs_11)
            cijk_12_mat = proj_3d.PI_mat('12', dofs_12)
            cijk_13_mat = proj_3d.PI_mat('13', dofs_13)
            cijk_21_mat = proj_3d.PI_mat('21', dofs_21)
            cijk_22_mat = proj_3d.PI_mat('22', dofs_22)
            cijk_23_mat = proj_3d.PI_mat('23', dofs_23)
            cijk_3_mat  = proj_3d.PI_mat('3', dofs_3)

            # result from A) and B) must be the same
            assert np.allclose(cijk_0, cijk_0_mat, atol=1e-14)
            assert np.allclose(cijk_11, cijk_11_mat, atol=1e-14)
            assert np.allclose(cijk_12, cijk_12_mat, atol=1e-14)
            assert np.allclose(cijk_13, cijk_13_mat, atol=1e-14)
            assert np.allclose(cijk_21, cijk_21_mat, atol=1e-14)
            assert np.allclose(cijk_22, cijk_22_mat, atol=1e-14)
            assert np.allclose(cijk_23, cijk_23_mat, atol=1e-14)
            assert np.allclose(cijk_3, cijk_3_mat, atol=1e-14)

            # evaluation of splines:
            fh_0  = np.empty((eta1_v.size, eta2_v.size, eta3_v.size))
            fh_11 = fh_0.copy()
            fh_12 = fh_0.copy()
            fh_13 = fh_0.copy()
            fh_21 = fh_0.copy()
            fh_22 = fh_0.copy()
            fh_23 = fh_0.copy()
            fh_3  = fh_0.copy()
            f_mat = fh_0.copy()

            fh_0  = Vh_tensor.evaluate_NNN(eta1_v, eta2_v, eta3_v, cijk_0)
            fh_11 = Vh_tensor.evaluate_DNN(eta1_v, eta2_v, eta3_v, cijk_11)
            fh_12 = Vh_tensor.evaluate_NDN(eta1_v, eta2_v, eta3_v, cijk_12)
            fh_13 = Vh_tensor.evaluate_NND(eta1_v, eta2_v, eta3_v, cijk_13)
            fh_21 = Vh_tensor.evaluate_NDD(eta1_v, eta2_v, eta3_v, cijk_21)
            fh_22 = Vh_tensor.evaluate_DND(eta1_v, eta2_v, eta3_v, cijk_22)
            fh_23 = Vh_tensor.evaluate_DDN(eta1_v, eta2_v, eta3_v, cijk_23)
            fh_3  = Vh_tensor.evaluate_DDD(eta1_v, eta2_v, eta3_v, cijk_3)

            # compute error
            f_mat = f(ee1, ee2, ee3)

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
    test_projectors_2d()
    test_projectors_3d()