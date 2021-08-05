def test_projectors_dot(plot=False):

        import sys
        sys.path.append('..')

        import numpy as np
        import hylife.utilitis_FEEC.bsplines as bsp
        import matplotlib.pyplot as plt
        import time

    
        import hylife.geometry.domain_3d as dom
        import hylife.utilitis_FEEC.spline_space as spl

        from hylife.utilitis_FEEC.projectors import mhd_operators_3d_global_V2 as mhd_op_V2

        # spline spaces
        Nel      = [7, 9, 8]
        p        = [4, 4, 4]
        spl_kind = [True, True, True]
        n_quad   = p.copy()
        bc       = ['d', 'd']
        kind_map = 'cuboid'     

        # 1d B-spline spline spaces for finite elements
        spaces_FEM = [spl.spline_space_1d(Nel_i, p_i, spl_kind_i, n_quad_i, bc) 
                    for Nel_i, p_i, spl_kind_i, n_quad_i in zip(Nel, p, spl_kind, n_quad)]

        # 3d tensor-product B-spline space for finite elements
        tensor_space_FEM = spl.tensor_spline_space(spaces_FEM)

        # domain
        domain = dom.domain(kind_map)
        #domain = dom.domain(kind_map, params_map, Nel, p, spl_kind)

        # 1D commuting projectors
        spaces_FEM[0].set_projectors(n_quad[0])
        spaces_FEM[1].set_projectors(n_quad[1])
        spaces_FEM[2].set_projectors(n_quad[2])

        tensor_space_FEM.set_projectors()

        # mhd projectors dot operator (automatically call "equilibrium_MHD.py")
        dot_ops = mhd_op_V2.projectors_dot_x([spaces_FEM[0].projectors, spaces_FEM[1].projectors, spaces_FEM[2].projectors], domain)

        # x which is going to product with projectors
        # V0_h
        x = np.ones(tensor_space_FEM.Ntot_0form)
        x_N0 = tensor_space_FEM.extract_0form(x)
        x_N0_flat = x_N0.flatten()

        x = np.random.rand(tensor_space_FEM.Ntot_0form)
        x_N0_rand = tensor_space_FEM.extract_0form(x)
        x_N0_rand_flat = x_N0_rand.flatten()

        # V1_h
        x_1 = np.ones(tensor_space_FEM.Ntot_1form_cum[-1])
        x_11, x_12, x_13 = tensor_space_FEM.extract_1form(x_1)
        x_N1 = [x_11, x_12, x_13]

        x_11_flat, x_12_flat, x_13_flat = x_11.flatten(), x_12.flatten(), x_13.flatten()
        x_N1_conc = np.concatenate((x_11_flat, x_12_flat, x_13_flat))

        x_1 = np.random.rand(tensor_space_FEM.Ntot_1form_cum[-1])
        x_11_rand, x_12_rand, x_13_rand = tensor_space_FEM.extract_1form(x_1)
        x_N1_rand = [x_11_rand, x_12_rand, x_13_rand]

        x_11_rand_flat, x_12_rand_flat, x_13_rand_flat = x_11_rand.flatten(), x_12_rand.flatten(), x_13_rand.flatten()
        x_N1_rand_conc = np.concatenate((x_11_rand_flat, x_12_rand_flat, x_13_rand_flat))

        # V2_h
        x_2 = np.ones(tensor_space_FEM.Ntot_2form_cum[-1])
        x_21, x_22, x_23 = tensor_space_FEM.extract_2form(x_2)
        x_N2 = [x_21, x_22, x_23]

        x_21_flat, x_22_flat, x_23_flat = x_21.flatten(), x_22.flatten(), x_23.flatten()
        x_N2_conc = np.concatenate((x_21_flat, x_22_flat, x_23_flat))

        x_2 = np.random.rand(tensor_space_FEM.Ntot_2form_cum[-1])
        x_21_rand, x_22_rand, x_23_rand = tensor_space_FEM.extract_2form(x_2)
        x_N2_rand = [x_21_rand, x_22_rand, x_23_rand]

        x_21_rand_flat, x_22_rand_flat, x_23_rand_flat = x_21_rand.flatten(), x_22_rand.flatten(), x_23_rand.flatten()
        x_N2_rand_conc = np.concatenate((x_21_rand_flat, x_22_rand_flat, x_23_rand_flat))

        # test conditions
        print()
        print('MHD_equilibrium :')
        print('p_eq = 1.0')
        print('r_eq = 1.0')
        print('b_eq_x = 0')
        print('b_eq_y = 0')
        print('b_eq_z = 1.0')
        print('j_eq_x = 0')
        print('j_eq_y = 0')
        print('j_eq_z = 0')

        print()
        print('maping  :')
        print('kind_map = ' + str(kind_map))
        print('params_map = ' + str(domain.params_map))

        print()

        # ========== Identity test ========== #
        print('Identity test for the projection K, K.T, S, S.T, W and W.T   :')
        print('Under the condition, p_eq = 1, projection K dot x = x')
        # projection K 
        K_dot_x = dot_ops.K_dot(x_N0_rand)
        assert np.allclose(K_dot_x, x_N0_rand, atol=1e-14)
        print('Done. K_dot(x_random) == x_random')

        # projection transpose K
        transpose_K_dot_x = dot_ops.transpose_K_dot(x_N0_rand)
        assert np.allclose(transpose_K_dot_x, x_N0_rand, atol=1e-14)
        print('Done. Transpose_K_dot(x_random) == x_random')

        # projection S
        print('Under the condition, p_eq = 1, projection S dot x = x')
        S_dot_x = dot_ops.S_dot(x_N1_rand) 
        assert np.allclose(S_dot_x[0], x_N1_rand[0], atol=1e-14)
        assert np.allclose(S_dot_x[1], x_N1_rand[1], atol=1e-14)
        assert np.allclose(S_dot_x[2], x_N1_rand[2], atol=1e-14)
        print('Done. S_dot(x_random) == x_random')

        # projection transpose S
        transpose_S_dot_x = dot_ops.transpose_S_dot(x_N1_rand)  
        assert np.allclose(transpose_S_dot_x[0], x_N1_rand[0], atol=1e-14)
        assert np.allclose(transpose_S_dot_x[1], x_N1_rand[1], atol=1e-14)
        assert np.allclose(transpose_S_dot_x[2], x_N1_rand[2], atol=1e-14)
        print('Done. Transpose_S_dot(x_random) == x_random')

        # projection W
        print('Under the conditions, rho_eq = 1 and G_inv = [Identity matrix], projection W dot x = x')
        W_dot_x = dot_ops.W_dot(x_N1_rand) 
        assert np.allclose(W_dot_x[0], x_N1_rand[0], atol=1e-14)
        assert np.allclose(W_dot_x[1], x_N1_rand[1], atol=1e-14)
        assert np.allclose(W_dot_x[2], x_N1_rand[2], atol=1e-14)
        print('Done. W_dot(x_random) == x_random')

        # projection transpose W
        transpose_W_dot_x = dot_ops.transpose_W_dot(x_N1_rand)  
        assert np.allclose(transpose_W_dot_x[0], x_N1_rand[0], atol=1e-14)
        assert np.allclose(transpose_W_dot_x[1], x_N1_rand[1], atol=1e-14)
        assert np.allclose(transpose_W_dot_x[2], x_N1_rand[2], atol=1e-14)
        print('Done. Transpose_W_dot(x_random) == x_random')
        print()

        # ========== convergence test ========== #
        print('Convergence test for the projection Q, P and T :')

        ################
        # dot operator #
        ################
        # Q
        start = time.time()
        Q_dot_x = dot_ops.Q_dot(x_N1_rand)

        # P
        P_dot_x = dot_ops.P_dot(x_N2_rand)

        # T
        T_dot_x = dot_ops.T_dot(x_N1_rand)

        end = time.time()
        print('calculating projection Q, P and T using dot operator is done!')
        print(end-start)

        # PP
        PP_dot_x = dot_ops.PP_dot(x_N1_rand)
        
        ########
        # plot #
        ########
        # ========== random splines ========== #
        # V2_h
        def phi_21(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDD(eta1, eta2, eta3, x_N2_rand[0])

        def phi_22(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DND(eta1, eta2, eta3, x_N2_rand[1])

        def phi_23(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DDN(eta1, eta2, eta3, x_N2_rand[2])
        
        # V1_h
        def phi_11(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DNN(eta1, eta2, eta3, x_N1_rand[0])

        def phi_12(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDN(eta1, eta2, eta3, x_N1_rand[1])

        def phi_13(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NND(eta1, eta2, eta3, x_N1_rand[2])

        # ========== splines constructed from the dot operator ========== #
        # construct splines in V2_h with the coeffs from the operator Q_dot    
        def Q_dot_x_1(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDD(eta1, eta2, eta3, Q_dot_x[0])

        def Q_dot_x_2(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DND(eta1, eta2, eta3, Q_dot_x[1])

        def Q_dot_x_3(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DDN(eta1, eta2, eta3, Q_dot_x[2])

        # construct splines in V1_h with the coeffs from the operator P_dot   
        def P_dot_x_1(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DNN(eta1, eta2, eta3, P_dot_x[0])

        def P_dot_x_2(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDN(eta1, eta2, eta3, P_dot_x[1])

        def P_dot_x_3(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NND(eta1, eta2, eta3, P_dot_x[2])
            
        # construct splines in V1_h with the coeffs from the operator T_dot    
        def T_dot_x_1(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DNN(eta1, eta2, eta3, T_dot_x[0])

        def T_dot_x_2(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDN(eta1, eta2, eta3, T_dot_x[1])

        def T_dot_x_3(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NND(eta1, eta2, eta3, T_dot_x[2])

        # construct splines in V0_h with the coeffs from the operator PP_dot    
        def PP_dot_x_1(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NNN(eta1, eta2, eta3, PP_dot_x[0])

        def PP_dot_x_2(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NNN(eta1, eta2, eta3, PP_dot_x[1])

        def PP_dot_x_3(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NNN(eta1, eta2, eta3, PP_dot_x[2])

        # ========== Evaluation for the plot ========== #
        eta1 = np.linspace(0., 1., 100)
        eta2 = np.linspace(0., 1., 100)
        eta3 = np.linspace(0., 1., 100)

        rhoeq_Ginv_lambda1_1_plot = phi_11(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_11') * dot_ops.rho_eq_fun(eta1, eta2, eta3) +\
                                    phi_12(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_12') * dot_ops.rho_eq_fun(eta1, eta2, eta3) +\
                                    phi_13(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_13') * dot_ops.rho_eq_fun(eta1, eta2, eta3)
        rhoeq_Ginv_lambda1_2_plot = phi_11(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_21') * dot_ops.rho_eq_fun(eta1, eta2, eta3) +\
                                    phi_12(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_22') * dot_ops.rho_eq_fun(eta1, eta2, eta3) +\
                                    phi_13(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_23') * dot_ops.rho_eq_fun(eta1, eta2, eta3)
        rhoeq_Ginv_lambda1_3_plot = phi_11(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_31') * dot_ops.rho_eq_fun(eta1, eta2, eta3) +\
                                    phi_12(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_32') * dot_ops.rho_eq_fun(eta1, eta2, eta3) +\
                                    phi_13(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_33') * dot_ops.rho_eq_fun(eta1, eta2, eta3)

        jeq_lambda2_1_plot = phi_22(eta1, eta2, eta3) * -dot_ops.j_eq_z_fun(eta1, eta2, eta3) + phi_23(eta1, eta2, eta3) * dot_ops.j_eq_y_fun(eta1, eta2, eta3)
        jeq_lambda2_2_plot = phi_21(eta1, eta2, eta3) * dot_ops.j_eq_z_fun(eta1, eta2, eta3) - phi_23(eta1, eta2, eta3) * dot_ops.j_eq_x_fun(eta1, eta2, eta3)
        jeq_lambda2_3_plot = phi_21(eta1, eta2, eta3) * -dot_ops.j_eq_y_fun(eta1, eta2, eta3) + phi_22(eta1, eta2, eta3) * dot_ops.j_eq_x_fun(eta1, eta2, eta3)

        Beq_Ginv_lambda1_1_plot = phi_11(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_31') * dot_ops.b_eq_y_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_21') * dot_ops.b_eq_z_fun(eta1, eta2, eta3)) +\
                                  phi_12(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_32') * dot_ops.b_eq_y_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_22') * dot_ops.b_eq_z_fun(eta1, eta2, eta3)) +\
                                  phi_13(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_33') * dot_ops.b_eq_y_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_23') * dot_ops.b_eq_z_fun(eta1, eta2, eta3))
        Beq_Ginv_lambda1_2_plot = phi_11(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_11') * dot_ops.b_eq_z_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_31') * dot_ops.b_eq_x_fun(eta1, eta2, eta3)) +\
                                  phi_12(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_12') * dot_ops.b_eq_z_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_32') * dot_ops.b_eq_x_fun(eta1, eta2, eta3)) +\
                                  phi_13(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_13') * dot_ops.b_eq_z_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_33') * dot_ops.b_eq_x_fun(eta1, eta2, eta3))
        Beq_Ginv_lambda1_3_plot = phi_11(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_21') * dot_ops.b_eq_x_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_11') * dot_ops.b_eq_y_fun(eta1, eta2, eta3)) +\
                                  phi_12(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_22') * dot_ops.b_eq_x_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_12') * dot_ops.b_eq_y_fun(eta1, eta2, eta3)) +\
                                  phi_13(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_23') * dot_ops.b_eq_x_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_13') * dot_ops.b_eq_y_fun(eta1, eta2, eta3))

        DFinv_T_lambda1_1_plot = phi_11(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'df_inv_11') +\
                                 phi_12(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'df_inv_21') +\
                                 phi_13(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'df_inv_31')
        DFinv_T_lambda1_2_plot = phi_11(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'df_inv_12') +\
                                 phi_12(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'df_inv_22') +\
                                 phi_13(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'df_inv_32')
        DFinv_T_lambda1_3_plot = phi_11(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'df_inv_13') +\
                                 phi_12(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'df_inv_23') +\
                                 phi_13(eta1, eta2, eta3) * dot_ops.domain.evaluate(eta1, eta2, eta3, 'df_inv_33')

        Q_dot_x_1_plot = Q_dot_x_1(eta1, eta2, eta3)
        Q_dot_x_2_plot = Q_dot_x_2(eta1, eta2, eta3)
        Q_dot_x_3_plot = Q_dot_x_3(eta1, eta2, eta3)
                
        P_dot_x_1_plot = P_dot_x_1(eta1, eta2, eta3)
        P_dot_x_2_plot = P_dot_x_2(eta1, eta2, eta3)
        P_dot_x_3_plot = P_dot_x_3(eta1, eta2, eta3)
                
        T_dot_x_1_plot = T_dot_x_1(eta1, eta2, eta3)
        T_dot_x_2_plot = T_dot_x_2(eta1, eta2, eta3)
        T_dot_x_3_plot = T_dot_x_3(eta1, eta2, eta3)

        PP_dot_x_1_plot = PP_dot_x_1(eta1, eta2, eta3)
        PP_dot_x_2_plot = PP_dot_x_2(eta1, eta2, eta3)
        PP_dot_x_3_plot = PP_dot_x_3(eta1, eta2, eta3)

        fig, axs = plt.subplots(4,3, figsize = (10, 12))
    
        axs[0,0].plot(eta1, rhoeq_Ginv_lambda1_1_plot[:, 50, 50], 'r', label = 'rhoeq_Ginv_lambda1[V1_h]')
        axs[0,1].plot(eta2, rhoeq_Ginv_lambda1_2_plot[50, :, 50], 'r', label = 'rhoeq_Ginv_lambda1[V1_h]')
        axs[0,2].plot(eta3, rhoeq_Ginv_lambda1_3_plot[50, 50, :], 'r', label = 'rhoeq_Ginv_lambda1[V1_h]')
                
        axs[0,0].plot(eta1, Q_dot_x_1_plot[:, 50, 50], 'b--', label = 'Q_dot[V2_h]')
        axs[0,1].plot(eta2, Q_dot_x_2_plot[50, :, 50], 'b--', label = 'Q_dot[V2_h]')
        axs[0,2].plot(eta3, Q_dot_x_3_plot[50, 50, :], 'b--', label = 'Q_dot[V2_h]')
                
        axs[1,0].plot(eta1, jeq_lambda2_1_plot[:, 50, 50], 'r', label = 'jeq_lambda2[V2_h]')
        axs[1,1].plot(eta2, jeq_lambda2_2_plot[50, :, 50], 'r', label = 'jeq_lambda2[V2_h]')
        axs[1,2].plot(eta3, jeq_lambda2_3_plot[50, 50, :], 'r', label = 'jeq_lambda2[V2_h]')
                
        axs[1,0].plot(eta1, P_dot_x_1_plot[:, 50, 50], 'b--', label = 'P_dot[V1_h]')
        axs[1,1].plot(eta2, P_dot_x_2_plot[50, :, 50], 'b--', label = 'P_dot[V1_h]')
        axs[1,2].plot(eta3, P_dot_x_3_plot[50, 50, :], 'b--', label = 'P_dot[V1_h]')
                
        axs[2,0].plot(eta1, Beq_Ginv_lambda1_1_plot[:, 50, 50], 'r', label = 'Beq_Ginv_lambda1[V1_h]')
        axs[2,1].plot(eta2, Beq_Ginv_lambda1_2_plot[50, :, 50], 'r', label = 'Beq_Ginv_lambda1[V1_h]')
        axs[2,2].plot(eta3, Beq_Ginv_lambda1_3_plot[50, 50, :], 'r', label = 'Beq_Ginv_lambda1[V1_h]')
                
        axs[2,0].plot(eta1, T_dot_x_1_plot[:, 50, 50], 'b--', label = 'T_dot[V1_h]')
        axs[2,1].plot(eta2, T_dot_x_2_plot[50, :, 50], 'b--', label = 'T_dot[V1_h]')
        axs[2,2].plot(eta3, T_dot_x_3_plot[50, 50, :], 'b--', label = 'T_dot[V1_h]')

        axs[3,0].plot(eta1, DFinv_T_lambda1_1_plot[:, 50, 50], 'r', label = 'DFinv_T_lambda1[V1_h]')
        axs[3,1].plot(eta2, DFinv_T_lambda1_2_plot[50, :, 50], 'r', label = 'DFinv_T_lambda1[V1_h]')
        axs[3,2].plot(eta3, DFinv_T_lambda1_3_plot[50, 50, :], 'r', label = 'DFinv_T_lambda1[V1_h]')
                
        axs[3,0].plot(eta1, PP_dot_x_1_plot[:, 50, 50], 'b--', label = 'PP_dot[V0_h]')
        axs[3,1].plot(eta2, PP_dot_x_2_plot[50, :, 50], 'b--', label = 'PP_dot[V0_h]')
        axs[3,2].plot(eta3, PP_dot_x_3_plot[50, 50, :], 'b--', label = 'PP_dot[V0_h]')
                
        axs[0,0].legend()
        axs[0,1].legend()
        axs[0,2].legend()
        axs[1,0].legend()
        axs[1,1].legend()
        axs[1,2].legend()
        axs[2,0].legend()
        axs[2,1].legend()
        axs[2,2].legend()
        axs[3,0].legend()
        axs[3,1].legend()
        axs[3,2].legend()

        axs[0,0].set_title('[eta1]   Nel: ' + str(Nel[0]) + '  p: ' + str(p[0]) + '  bc: ' + str(spl_kind[0]))
        axs[0,1].set_title('[eta2]   Nel: ' + str(Nel[1]) + '  p: ' + str(p[1]) + '  bc: ' + str(spl_kind[1]))
        axs[0,2].set_title('[eta3]   Nel: ' + str(Nel[2]) + '  p: ' + str(p[2]) + '  bc: ' + str(spl_kind[2]))
        axs[1,0].set_title('[eta1]   Nel: ' + str(Nel[0]) + '  p: ' + str(p[0]) + '  bc: ' + str(spl_kind[0]))
        axs[1,1].set_title('[eta2]   Nel: ' + str(Nel[1]) + '  p: ' + str(p[1]) + '  bc: ' + str(spl_kind[1]))
        axs[1,2].set_title('[eta3]   Nel: ' + str(Nel[2]) + '  p: ' + str(p[2]) + '  bc: ' + str(spl_kind[2]))
        axs[2,0].set_title('[eta1]   Nel: ' + str(Nel[0]) + '  p: ' + str(p[0]) + '  bc: ' + str(spl_kind[0]))
        axs[2,1].set_title('[eta2]   Nel: ' + str(Nel[1]) + '  p: ' + str(p[1]) + '  bc: ' + str(spl_kind[1]))
        axs[2,2].set_title('[eta3]   Nel: ' + str(Nel[2]) + '  p: ' + str(p[2]) + '  bc: ' + str(spl_kind[2]))
        axs[3,0].set_title('[eta1]   Nel: ' + str(Nel[0]) + '  p: ' + str(p[0]) + '  bc: ' + str(spl_kind[0]))
        axs[3,1].set_title('[eta2]   Nel: ' + str(Nel[1]) + '  p: ' + str(p[1]) + '  bc: ' + str(spl_kind[1]))
        axs[3,2].set_title('[eta3]   Nel: ' + str(Nel[2]) + '  p: ' + str(p[2]) + '  bc: ' + str(spl_kind[2]))

        fig.tight_layout()

        if plot:
                plt.show()
        

def test_comparison_with_basic_projector(plot=False):
        import sys
        sys.path.append('..')

        import numpy as np
        import matplotlib.pyplot as plt
        import hylife.utilitis_FEEC.bsplines as bsp

        import hylife.geometry.domain_3d as dom
        import hylife.utilitis_FEEC.spline_space as spl

        from hylife.utilitis_FEEC.projectors import mhd_operators_3d_global_V2 as mhd_op_V2

        Nel      = [22, 22, 22]
        p        = [4, 4, 4]
        spl_kind = [True, True, True]
        n_quad   = p.copy()
        bc       = ['d', 'd']
        kind_map = 'cuboid'

        # test! what cause differences
        params_map = [1., 1., 1.]

        # 1d B-spline spline spaces for finite elements
        spaces_FEM = [spl.spline_space_1d(Nel_i, p_i, spl_kind_i, n_quad_i, bc) 
                    for Nel_i, p_i, spl_kind_i, n_quad_i in zip(Nel, p, spl_kind, n_quad)]

        # 3d tensor-product B-spline space for finite elements
        tensor_space_FEM = spl.tensor_spline_space(spaces_FEM)

        # random coefficients
        # V0_h
        coeffs = np.random.rand(tensor_space_FEM.Ntot_0form)
        coeffs_0 = tensor_space_FEM.extract_0form(coeffs)

        # V1_h
        coeffs = np.random.rand(tensor_space_FEM.Ntot_1form_cum[-1])
        coeffs_11, coeffs_12, coeffs_13 = tensor_space_FEM.extract_1form(coeffs)

        # V2_h
        coeffs = np.random.rand(tensor_space_FEM.Ntot_2form_cum[-1])
        coeffs_21, coeffs_22, coeffs_23 = tensor_space_FEM.extract_2form(coeffs)
        
        #####################################################
        # splines projection from mhd_operator_3d_global_V2 #
        #####################################################

        # 1D commuting projectors
        spaces_FEM[0].set_projectors(n_quad[0])
        spaces_FEM[1].set_projectors(n_quad[1])
        spaces_FEM[2].set_projectors(n_quad[2])

        tensor_space_FEM.set_projectors()

        # domain
        domain = dom.domain(kind_map)
        #domain = dom.domain(kind_map, params_map, Nel, p, spl_kind)

        # mhd projectors (automatically call "equilibrium_MHD.py")
        dot_ops = mhd_op_V2.projectors_dot_x([spaces_FEM[0].projectors, spaces_FEM[1].projectors, spaces_FEM[2].projectors], domain)

        # pi_1[lambda^2]
        PI1_lambda2_dot_x = dot_ops.PI1_lambda2_dot([coeffs_21, coeffs_22, coeffs_23])

        # pi_2[lambda^1]
        #PI2_lambda1_dot_x = dot_ops.PI_2_lambda_1_dot([coeffs_11, coeffs_12, coeffs_13])
        PI2_lambda1_dot_x = dot_ops.PI2_lambda1_dot([coeffs_11, coeffs_12, coeffs_13])

        ##################################################################
        # splines projection from projectors_global.projectors_global_1d #
        ##################################################################
        # random spline in V1_h
        def phi_11(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DNN(eta1, eta2, eta3, coeffs_11)

        def phi_12(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDN(eta1, eta2, eta3, coeffs_12)

        def phi_13(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NND(eta1, eta2, eta3, coeffs_13)

        # project phi_11 on V2_h space:
        c21, c22, c23 = tensor_space_FEM.projectors.PI_2(phi_11, phi_12, phi_13)

        print()
        print('Random spline projection test :')

        assert np.allclose(c21, PI2_lambda1_dot_x[0], atol=1e-14)
        assert np.allclose(c22, PI2_lambda1_dot_x[1], atol=1e-14)
        assert np.allclose(c23, PI2_lambda1_dot_x[2], atol=1e-14)
        print('projection of Lambda^1 on V2_h space using both projectors are identical')


        # random spline in V2_h
        def phi_21(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDD(eta1, eta2, eta3, coeffs_21)

        def phi_22(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DND(eta1, eta2, eta3, coeffs_22)

        def phi_23(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DDN(eta1, eta2, eta3, coeffs_23)

        # project phi_21 on V1_h space:
        c11, c12, c13 = tensor_space_FEM.projectors.PI_1(phi_21, phi_22, phi_23)

        assert np.allclose(c11, PI1_lambda2_dot_x[0], atol=1e-14)
        assert np.allclose(c12, PI1_lambda2_dot_x[1], atol=1e-14)
        assert np.allclose(c13, PI1_lambda2_dot_x[2], atol=1e-14)
        print('projection of Lambda^2 on V1_h space using both projectors are identical')

        ##############################
        # plot the projected splines #
        ##############################
        
        # construct splines in V1_h space with the coeffs from pi_1[ lambda^2]
        def phi_h_11(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DNN(eta1, eta2, eta3, c11)

        def phi_h_12(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDN(eta1, eta2, eta3, c12)

        def phi_h_13(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NND(eta1, eta2, eta3, c13)

        # construct splines in V2_h space with the coeffs from pi_2[ lambda^1]
        def phi_h_21(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDD(eta1, eta2, eta3, c21)

        def phi_h_22(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DND(eta1, eta2, eta3, c22)

        def phi_h_23(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DDN(eta1, eta2, eta3, c23)
            
        # construct splines in V1_h space with the coeffs from the operator PI1_lambda2_dot    
        def PI1_lambda2_1(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DNN(eta1, eta2, eta3, PI1_lambda2_dot_x[0])

        def PI1_lambda2_2(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDN(eta1, eta2, eta3, PI1_lambda2_dot_x[1])

        def PI1_lambda2_3(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NND(eta1, eta2, eta3, PI1_lambda2_dot_x[2])

        # construct splines in V2_h space with the coeffs from the operator PI2_lambda1_dot   
        def PI2_lambda1_1(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDD(eta1, eta2, eta3, PI2_lambda1_dot_x[0])

        def PI2_lambda1_2(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DND(eta1, eta2, eta3, PI2_lambda1_dot_x[1])

        def PI2_lambda1_3(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DDN(eta1, eta2, eta3, PI2_lambda1_dot_x[2])

        
        if plot:
                eta1 = np.linspace(0., 1., 100)
                eta2 = np.linspace(0., 1., 100)
                eta3 = np.linspace(0., 1., 100)
                
                phi_11_plot = phi_11(eta1, eta2, eta3)
                phi_12_plot = phi_12(eta1, eta2, eta3)
                phi_13_plot = phi_13(eta1, eta2, eta3)
                
                phi_21_plot = phi_21(eta1, eta2, eta3)
                phi_22_plot = phi_22(eta1, eta2, eta3)
                phi_23_plot = phi_23(eta1, eta2, eta3)
                
                phi_h_11_plot = phi_h_11(eta1, eta2, eta3)
                phi_h_12_plot = phi_h_12(eta1, eta2, eta3)
                phi_h_13_plot = phi_h_13(eta1, eta2, eta3)
                
                phi_h_21_plot = phi_h_21(eta1, eta2, eta3)
                phi_h_22_plot = phi_h_22(eta1, eta2, eta3)
                phi_h_23_plot = phi_h_23(eta1, eta2, eta3)
                
                PI1_lambda2_1_plot = PI1_lambda2_1(eta1, eta2, eta3)
                PI1_lambda2_2_plot = PI1_lambda2_2(eta1, eta2, eta3)
                PI1_lambda2_3_plot = PI1_lambda2_3(eta1, eta2, eta3)

                PI2_lambda1_1_plot = PI2_lambda1_1(eta1, eta2, eta3)
                PI2_lambda1_2_plot = PI2_lambda1_2(eta1, eta2, eta3)
                PI2_lambda1_3_plot = PI2_lambda1_3(eta1, eta2, eta3)

                fig, axs = plt.subplots(2,3, figsize = (8, 8))
                axs[0,0].plot(eta1, phi_11_plot[:, 50, 50], 'r', label = 'random_spline[V1_h]')
                axs[0,1].plot(eta2, phi_12_plot[50, :, 50], 'r', label = 'random_spline[V1_h]')
                axs[0,2].plot(eta3, phi_13_plot[50, 50, :], 'r', label = 'random_spline[V1_h]')
                
                axs[0,0].plot(eta1, PI2_lambda1_1_plot[:, 50, 50], 'b--', label = 'PI2_lambda1_dot_operator[V2_h]')
                axs[0,1].plot(eta2, PI2_lambda1_2_plot[50, :, 50], 'b--', label = 'PI2_lambda1_dot_operator[V2_h]')
                axs[0,2].plot(eta3, PI2_lambda1_3_plot[50, 50, :], 'b--', label = 'PI2_lambda1_dot_operator[V2_h]')
                
                axs[0,0].plot(eta1, phi_h_21_plot[:, 50, 50], 'g--', label = 'PI2_lambda1_basic_projector[V2_h]')
                axs[0,1].plot(eta2, phi_h_22_plot[50, :, 50], 'g--', label = 'PI2_lambda1_basic_projector[V2_h]')
                axs[0,2].plot(eta3, phi_h_23_plot[50, 50, :], 'g--', label = 'PI2_lambda1_basic_projector[V2_h]')
                
                axs[1,0].plot(eta1, phi_21_plot[:, 50, 50], 'r', label = 'random_spline[V2_h]')
                axs[1,1].plot(eta2, phi_22_plot[50, :, 50], 'r', label = 'random_spline[V2_h]')
                axs[1,2].plot(eta3, phi_23_plot[50, 50, :], 'r', label = 'random_spline[V2_h]')
                
                axs[1,0].plot(eta1, PI1_lambda2_1_plot[:, 50, 50], 'b--', label = 'PI1_lambda2_dot_operator[V1_h]')
                axs[1,1].plot(eta2, PI1_lambda2_2_plot[50, :, 50], 'b--', label = 'PI1_lambda2_dot_operator[V1_h]')
                axs[1,2].plot(eta3, PI1_lambda2_3_plot[50, 50, :], 'b--', label = 'PI1_lambda2_dot_operator[V1_h]')
                
                axs[1,0].plot(eta1, phi_h_11_plot[:, 50, 50], 'g--', label = 'PI1_lambda2_basic_projector[V1_h]')
                axs[1,1].plot(eta2, phi_h_12_plot[50, :, 50], 'g--', label = 'PI1_lambda2_basic_projector[V1_h]')
                axs[1,2].plot(eta3, phi_h_13_plot[50, 50, :], 'g--', label = 'PI1_lambda2_basic_projector[V1_h]')

                axs[0,0].legend()
                axs[0,1].legend()
                axs[0,2].legend()
                axs[1,0].legend()
                axs[1,1].legend()
                axs[1,2].legend()

                axs[0,0].set_title('[eta1]   Nel: ' + str(Nel[0]) + '  p: ' + str(p[0]) + '  bc: ' + str(spl_kind[0]))
                axs[0,1].set_title('[eta2]   Nel: ' + str(Nel[1]) + '  p: ' + str(p[1]) + '  bc: ' + str(spl_kind[1]))
                axs[0,2].set_title('[eta3]   Nel: ' + str(Nel[2]) + '  p: ' + str(p[2]) + '  bc: ' + str(spl_kind[2]))
                axs[1,0].set_title('[eta1]   Nel: ' + str(Nel[0]) + '  p: ' + str(p[0]) + '  bc: ' + str(spl_kind[0]))
                axs[1,1].set_title('[eta2]   Nel: ' + str(Nel[1]) + '  p: ' + str(p[1]) + '  bc: ' + str(spl_kind[1]))
                axs[1,2].set_title('[eta3]   Nel: ' + str(Nel[2]) + '  p: ' + str(p[2]) + '  bc: ' + str(spl_kind[2]))

                if plot:
                        plt.show()


def test_new_projectors_dot(plot=False):

        import sys
        sys.path.append('..')

        import numpy as np
        import hylife.utilitis_FEEC.bsplines as bsp
        import matplotlib.pyplot as plt
        import time

    
        import hylife.geometry.domain_3d as dom
        import hylife.utilitis_FEEC.spline_space as spl

        from hylife.utilitis_FEEC.projectors import mhd_operators_3d_global_V2 as mhd_op_V2

        # spline spaces
        Nel      = [7, 8, 6]
        p        = [4, 5, 3]
        spl_kind = [True, True, True]
        n_quad   = p.copy()
        bc       = ['d', 'd']
        kind_map = 'cuboid'     

        # 1d B-spline spline spaces for finite elements
        spaces_FEM = [spl.spline_space_1d(Nel_i, p_i, spl_kind_i, n_quad_i, bc) 
                    for Nel_i, p_i, spl_kind_i, n_quad_i in zip(Nel, p, spl_kind, n_quad)]

        # 3d tensor-product B-spline space for finite elements
        tensor_space_FEM = spl.tensor_spline_space(spaces_FEM)

        # domain
        domain = dom.domain(kind_map)
        #domain = dom.domain(kind_map, params_map, Nel, p, spl_kind)

        # 1D commuting projectors
        spaces_FEM[0].set_projectors(n_quad[0])
        spaces_FEM[1].set_projectors(n_quad[1])
        spaces_FEM[2].set_projectors(n_quad[2])

        tensor_space_FEM.set_projectors()

        # mhd projectors dot operator (automatically call "equilibrium_MHD.py")
        dot_ops = mhd_op_V2.projectors_dot_x([spaces_FEM[0].projectors, spaces_FEM[1].projectors, spaces_FEM[2].projectors], domain)

        # x which is going to product with projectors
        # V0_h
        x = np.ones(tensor_space_FEM.Ntot_0form)
        x_N0 = tensor_space_FEM.extract_0form(x)
        x_N0_flat = x_N0.flatten()

        x = np.random.rand(tensor_space_FEM.Ntot_0form)
        x_N0_rand = tensor_space_FEM.extract_0form(x)
        x_N0_rand_flat = x_N0_rand.flatten()

        # V1_h
        x_1 = np.ones(tensor_space_FEM.Ntot_1form_cum[-1])
        x_11, x_12, x_13 = tensor_space_FEM.extract_1form(x_1)
        x_N1 = [x_11, x_12, x_13]

        x_11_flat, x_12_flat, x_13_flat = x_11.flatten(), x_12.flatten(), x_13.flatten()
        x_N1_conc = np.concatenate((x_11_flat, x_12_flat, x_13_flat))

        x_1 = np.random.rand(tensor_space_FEM.Ntot_1form_cum[-1])
        x_11_rand, x_12_rand, x_13_rand = tensor_space_FEM.extract_1form(x_1)
        x_N1_rand = [x_11_rand, x_12_rand, x_13_rand]

        x_11_rand_flat, x_12_rand_flat, x_13_rand_flat = x_11_rand.flatten(), x_12_rand.flatten(), x_13_rand.flatten()
        x_N1_rand_conc = np.concatenate((x_11_rand_flat, x_12_rand_flat, x_13_rand_flat))

        # V2_h
        x_2 = np.ones(tensor_space_FEM.Ntot_2form_cum[-1])
        x_21, x_22, x_23 = tensor_space_FEM.extract_2form(x_2)
        x_N2 = [x_21, x_22, x_23]

        x_21_flat, x_22_flat, x_23_flat = x_21.flatten(), x_22.flatten(), x_23.flatten()
        x_N2_conc = np.concatenate((x_21_flat, x_22_flat, x_23_flat))

        x_2 = np.random.rand(tensor_space_FEM.Ntot_2form_cum[-1])
        x_21_rand, x_22_rand, x_23_rand = tensor_space_FEM.extract_2form(x_2)
        x_N2_rand = [x_21_rand, x_22_rand, x_23_rand]

        x_21_rand_flat, x_22_rand_flat, x_23_rand_flat = x_21_rand.flatten(), x_22_rand.flatten(), x_23_rand.flatten()
        x_N2_rand_conc = np.concatenate((x_21_rand_flat, x_22_rand_flat, x_23_rand_flat))

        # V3_h
        x = np.ones(tensor_space_FEM.Ntot_3form)
        x_N3 = tensor_space_FEM.extract_3form(x)
        x_N3_flat = x_N3.flatten()

        x = np.random.rand(tensor_space_FEM.Ntot_3form)
        x_N3_rand = tensor_space_FEM.extract_0form(x)
        x_N3_rand_flat = x_N3_rand.flatten()

        # test conditions
        print()
        print('MHD_equilibrium :')
        print('p_eq = 1.0')
        print('r_eq = 1.0')
        print('b_eq_x = 0')
        print('b_eq_y = 0')
        print('b_eq_z = 1.0')
        print('j_eq_x = 0')
        print('j_eq_y = 0')
        print('j_eq_z = 1.0')

        print()
        print('maping  :')
        print('kind_map = ' + str(kind_map))
        print('params_map = ' + str(domain.params_map))

        print()

        # ========== Identity test ========== #
        print('Identity test for the projection Q2, Q2.T, P2, P2.T, M and M.T   :')
        # projection Q2
        print('Under the condition, rho_eq = 1 and g_sqrt = 1, projection Q2 dot x = x')
        Q2_dot_x = dot_ops.Q2_dot(x_N2_rand) 
        assert np.allclose(Q2_dot_x[0], x_N2_rand[0], atol=1e-14)
        assert np.allclose(Q2_dot_x[1], x_N2_rand[1], atol=1e-14)
        assert np.allclose(Q2_dot_x[2], x_N2_rand[2], atol=1e-14)
        print('Done. Q2_dot(x_random) == x_random')

        # projection transpose Q2
        transpose_Q2_dot_x = dot_ops.transpose_Q2_dot(x_N2_rand)  
        assert np.allclose(transpose_Q2_dot_x[0], x_N2_rand[0], atol=1e-14)
        assert np.allclose(transpose_Q2_dot_x[1], x_N2_rand[1], atol=1e-14)
        assert np.allclose(transpose_Q2_dot_x[2], x_N2_rand[2], atol=1e-14)
        print('Done. Transpose_Q2_dot(x_random) == x_random')
        print()

        # projection P2
        print('Under the condition, p_eq = 1 and g_sqrt = 1, projection P2 dot x = x')
        P2_dot_x = dot_ops.P2_dot(x_N2_rand) 
        assert np.allclose(P2_dot_x[0], x_N2_rand[0], atol=1e-14)
        assert np.allclose(P2_dot_x[1], x_N2_rand[1], atol=1e-14)
        assert np.allclose(P2_dot_x[2], x_N2_rand[2], atol=1e-14)
        print('Done. P2_dot(x_random) == x_random')

        # projection transpose P2
        transpose_P2_dot_x = dot_ops.transpose_P2_dot(x_N2_rand)  
        assert np.allclose(transpose_P2_dot_x[0], x_N2_rand[0], atol=1e-14)
        assert np.allclose(transpose_P2_dot_x[1], x_N2_rand[1], atol=1e-14)
        assert np.allclose(transpose_P2_dot_x[2], x_N2_rand[2], atol=1e-14)
        print('Done. Transpose_P2_dot(x_random) == x_random')

         # projection M 
        print('Under the condition, p_eq = 1 and g_sqrt = 1, projection M dot x = x')
        M_dot_x = dot_ops.M_dot(x_N3_rand)
        assert np.allclose(M_dot_x, x_N3_rand, atol=1e-14)
        print('Done. M_dot(x_random) == x_random')

        # projection transpose M
        transpose_M_dot_x = dot_ops.transpose_M_dot(x_N3_rand)
        assert np.allclose(transpose_M_dot_x, x_N3_rand, atol=1e-14)
        print('Done. Transpose_M_dot(x_random) == x_random')
        print()

        # ========== convergence test ========== #
        print('Convergence test for the projection T2 and S2 :')

        ################
        # dot operator #
        ################
        # T2
        start = time.time()
        T2_dot_x = dot_ops.T2_dot(x_N2_rand)

        # S2
        S2_dot_x = dot_ops.S2_dot(x_N2_rand)

        end = time.time()
        print('calculating projection T2 and S2 using dot operator is done!')
        print(end-start)
        
        ########
        # plot #
        ########
        # ========== random splines ========== #
        # V2_h
        def phi_21(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDD(eta1, eta2, eta3, x_N2_rand[0])

        def phi_22(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DND(eta1, eta2, eta3, x_N2_rand[1])

        def phi_23(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DDN(eta1, eta2, eta3, x_N2_rand[2])
        
        # V1_h
        def phi_11(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DNN(eta1, eta2, eta3, x_N1_rand[0])

        def phi_12(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDN(eta1, eta2, eta3, x_N1_rand[1])

        def phi_13(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NND(eta1, eta2, eta3, x_N1_rand[2])

        # ========== splines constructed from the dot operator ========== #
        # construct splines in V2_h with the coeffs from the operator S2_dot    
        def S2_dot_x_1(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDD(eta1, eta2, eta3, S2_dot_x[0])

        def S2_dot_x_2(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DND(eta1, eta2, eta3, S2_dot_x[1])

        def S2_dot_x_3(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DDN(eta1, eta2, eta3, S2_dot_x[2])

        # construct splines in V1_h with the coeffs from the operator T2_dot   
        def T2_dot_x_1(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_DNN(eta1, eta2, eta3, T2_dot_x[0])

        def T2_dot_x_2(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NDN(eta1, eta2, eta3, T2_dot_x[1])

        def T2_dot_x_3(eta1, eta2, eta3):
                return tensor_space_FEM.evaluate_NND(eta1, eta2, eta3, T2_dot_x[2])

        # ========== Evaluation for the plot ========== #
        eta1 = np.linspace(0., 1., 100)
        eta2 = np.linspace(0., 1., 100)
        eta3 = np.linspace(0., 1., 100)

        Ginv_jeq_lambda2_1_plot = phi_21(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_12') * dot_ops.j_eq_z_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_13') * dot_ops.j_eq_y_fun(eta1, eta2, eta3)) +\
                                  phi_22(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_13') * dot_ops.j_eq_x_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_11') * dot_ops.j_eq_z_fun(eta1, eta2, eta3)) +\
                                  phi_23(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_11') * dot_ops.j_eq_y_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_12') * dot_ops.j_eq_x_fun(eta1, eta2, eta3))
        Ginv_jeq_lambda2_2_plot = phi_21(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_22') * dot_ops.j_eq_z_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_23') * dot_ops.j_eq_y_fun(eta1, eta2, eta3)) +\
                                  phi_22(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_23') * dot_ops.j_eq_x_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_21') * dot_ops.j_eq_z_fun(eta1, eta2, eta3)) +\
                                  phi_23(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_21') * dot_ops.j_eq_y_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_22') * dot_ops.j_eq_x_fun(eta1, eta2, eta3))
        Ginv_jeq_lambda2_3_plot = phi_21(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_32') * dot_ops.j_eq_z_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_33') * dot_ops.j_eq_y_fun(eta1, eta2, eta3)) +\
                                  phi_22(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_33') * dot_ops.j_eq_x_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_31') * dot_ops.j_eq_z_fun(eta1, eta2, eta3)) +\
                                  phi_23(eta1, eta2, eta3) * (dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_31') * dot_ops.j_eq_y_fun(eta1, eta2, eta3) - dot_ops.domain.evaluate(eta1, eta2, eta3, 'g_inv_32') * dot_ops.j_eq_x_fun(eta1, eta2, eta3))

        beq_gsqrt_lambda2_1_plot = phi_22(eta1, eta2, eta3) / dot_ops.domain.evaluate(eta1, eta2, eta3, 'det_df') * -dot_ops.b_eq_z_fun(eta1, eta2, eta3) + phi_23(eta1, eta2, eta3) / dot_ops.domain.evaluate(eta1, eta2, eta3, 'det_df') * dot_ops.b_eq_y_fun(eta1, eta2, eta3)
        beq_gsqrt_lambda2_2_plot = phi_21(eta1, eta2, eta3) / dot_ops.domain.evaluate(eta1, eta2, eta3, 'det_df') *  dot_ops.b_eq_z_fun(eta1, eta2, eta3) - phi_23(eta1, eta2, eta3) / dot_ops.domain.evaluate(eta1, eta2, eta3, 'det_df') * dot_ops.b_eq_x_fun(eta1, eta2, eta3)
        beq_gsqrt_lambda2_3_plot = phi_21(eta1, eta2, eta3) / dot_ops.domain.evaluate(eta1, eta2, eta3, 'det_df') * -dot_ops.b_eq_y_fun(eta1, eta2, eta3) + phi_22(eta1, eta2, eta3) / dot_ops.domain.evaluate(eta1, eta2, eta3, 'det_df') * dot_ops.b_eq_x_fun(eta1, eta2, eta3)

        S2_dot_x_1_plot = S2_dot_x_1(eta1, eta2, eta3)
        S2_dot_x_2_plot = S2_dot_x_2(eta1, eta2, eta3)
        S2_dot_x_3_plot = S2_dot_x_3(eta1, eta2, eta3)
                
        T2_dot_x_1_plot = T2_dot_x_1(eta1, eta2, eta3)
        T2_dot_x_2_plot = T2_dot_x_2(eta1, eta2, eta3)
        T2_dot_x_3_plot = T2_dot_x_3(eta1, eta2, eta3)


        fig, axs = plt.subplots(2,3, figsize = (10, 12))
    
        axs[0,0].plot(eta1, Ginv_jeq_lambda2_1_plot[:, 50, 50], 'r', label = 'Ginv_jeq_lambda2[V2_h]')
        axs[0,1].plot(eta2, Ginv_jeq_lambda2_2_plot[50, :, 50], 'r', label = 'Ginv_jeq_lambda2[V2_h]')
        axs[0,2].plot(eta3, Ginv_jeq_lambda2_3_plot[50, 50, :], 'r', label = 'Ginv_jeq_lambda2[V2_h]')
                
        axs[0,0].plot(eta1, S2_dot_x_1_plot[:, 50, 50], 'b--', label = 'S2_dot[V2_h]')
        axs[0,1].plot(eta2, S2_dot_x_2_plot[50, :, 50], 'b--', label = 'S2_dot[V2_h]')
        axs[0,2].plot(eta3, S2_dot_x_3_plot[50, 50, :], 'b--', label = 'S2_dot[V2_h]')
                
        axs[1,0].plot(eta1, beq_gsqrt_lambda2_1_plot[:, 50, 50], 'r', label = 'beq_gsqrt_lambda2[V2_h]')
        axs[1,1].plot(eta2, beq_gsqrt_lambda2_2_plot[50, :, 50], 'r', label = 'beq_gsqrt_lambda2[V2_h]')
        axs[1,2].plot(eta3, beq_gsqrt_lambda2_3_plot[50, 50, :], 'r', label = 'beq_gsqrt_lambda2[V2_h]')
                
        axs[1,0].plot(eta1, T2_dot_x_1_plot[:, 50, 50], 'b--', label = 'T2_dot[V1_h]')
        axs[1,1].plot(eta2, T2_dot_x_2_plot[50, :, 50], 'b--', label = 'T2_dot[V1_h]')
        axs[1,2].plot(eta3, T2_dot_x_3_plot[50, 50, :], 'b--', label = 'T2_dot[V1_h]')
                
        axs[0,0].legend()
        axs[0,1].legend()
        axs[0,2].legend()
        axs[1,0].legend()
        axs[1,1].legend()
        axs[1,2].legend()

        axs[0,0].set_title('[eta1]   Nel: ' + str(Nel[0]) + '  p: ' + str(p[0]) + '  bc: ' + str(spl_kind[0]))
        axs[0,1].set_title('[eta2]   Nel: ' + str(Nel[1]) + '  p: ' + str(p[1]) + '  bc: ' + str(spl_kind[1]))
        axs[0,2].set_title('[eta3]   Nel: ' + str(Nel[2]) + '  p: ' + str(p[2]) + '  bc: ' + str(spl_kind[2]))
        axs[1,0].set_title('[eta1]   Nel: ' + str(Nel[0]) + '  p: ' + str(p[0]) + '  bc: ' + str(spl_kind[0]))
        axs[1,1].set_title('[eta2]   Nel: ' + str(Nel[1]) + '  p: ' + str(p[1]) + '  bc: ' + str(spl_kind[1]))
        axs[1,2].set_title('[eta3]   Nel: ' + str(Nel[2]) + '  p: ' + str(p[2]) + '  bc: ' + str(spl_kind[2]))

        fig.tight_layout()

        if plot:
                plt.show()


if __name__ == '__main__':
    #test_projectors_dot(plot=True)
    #test_comparison_with_basic_projector(plot=True)
    test_new_projectors_dot(plot=True)
