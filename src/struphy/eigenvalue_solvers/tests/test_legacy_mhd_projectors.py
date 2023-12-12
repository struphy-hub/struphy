def test_1form_projectors_dot():
    """
    TODO
    """

    import sys
    sys.path.append('..')

    import numpy as np
    import time

    from struphy.geometry import domains

    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d
    from struphy.eigenvalue_solvers.spline_space import Tensor_spline_space

    from struphy.fields_background.mhd_equil.equils import HomogenSlab

    from struphy.eigenvalue_solvers.legacy import mhd_operators_MF as mhd_op_V2

    # spline spaces
    Nel = [7, 7, 7]
    p = [4, 4, 4]
    spl_kind = [True, True, True]
    n_quad = p.copy()
    bc = ['d', 'd']
    dom_type = 'Cuboid'

    # 1d B-spline spline spaces for finite elements
    spaces_FEM = [Spline_space_1d(Nel_i, p_i, spl_kind_i, n_quad_i, bc)
                  for Nel_i, p_i, spl_kind_i, n_quad_i in zip(Nel, p, spl_kind, n_quad)]

    # 1D commuting projectors
    spaces_FEM[0].set_projectors(n_quad[0])
    spaces_FEM[1].set_projectors(n_quad[1])
    spaces_FEM[2].set_projectors(n_quad[2])

    # 3d tensor-product B-spline space for finite elements
    tensor_space_FEM = Tensor_spline_space(spaces_FEM)

    tensor_space_FEM.set_projectors()

    # domain
    domain_class = getattr(domains, dom_type)
    domain = domain_class()

    # assemble mass matrices
    tensor_space_FEM.assemble_Mk(domain, 'V0')
    tensor_space_FEM.assemble_Mk(domain, 'V1')
    tensor_space_FEM.assemble_Mk(domain, 'V2')
    tensor_space_FEM.assemble_Mk(domain, 'V3')
    print('Assembly of mass matrices done.')
    print()

    # mhd projectors dot operator
    eq_MHD = HomogenSlab(
        **{'B0x': 0., 'B0y': 0., 'B0z': 1., 'beta': 2., 'n0': 1.})
    eq_MHD.domain = domain

    dot_ops = mhd_op_V2.projectors_dot_x(tensor_space_FEM, eq_MHD)

    # random x which is going to product with projectors
    x_0 = np.random.rand(tensor_space_FEM.Ntot_0form)
    x_1 = np.random.rand(tensor_space_FEM.Ntot_1form_cum[-1])
    x_11, x_12, x_13 = tensor_space_FEM.extract_1(x_1)
    x_2 = np.random.rand(tensor_space_FEM.Ntot_2form_cum[-1])
    x_21, x_22, x_23 = tensor_space_FEM.extract_2(x_2)
    x_3 = np.random.rand(tensor_space_FEM.Ntot_3form)

    # test conditions
    print()
    print('MHD_equilibrium :')
    print('p_eq =', dot_ops.eq_MHD.p3(0., 0., 0.))
    print('n_eq =', dot_ops.eq_MHD.n3(0., 0., 0.))
    print('b_eq_x =', dot_ops.eq_MHD.b2_1(0., 0., 0.))
    print('b_eq_y =', dot_ops.eq_MHD.b2_2(0., 0., 0.))
    print('b_eq_z =', dot_ops.eq_MHD.b2_3(0., 0., 0.))
    print('j_eq_x =', dot_ops.eq_MHD.j2_1(0., 0., 0.))
    print('j_eq_y =', dot_ops.eq_MHD.j2_2(0., 0., 0.))
    print('j_eq_z =', dot_ops.eq_MHD.j2_3(0., 0., 0.))

    print()
    print('maping  :')
    print('dom_type = ' + str(dom_type))
    print('params_map = ' + str(domain.params_map))

    print()

    # ========== Identity test ========== #
    print('Identity test for the projection operator W1_dot and K1_dot :')
    print('Calculating projection operator W1_dot, K1_dot, K10_dot and S10_dot with random x is done!')
    print()
    # projection W1
    print('Under the condition, rho_eq = 1 and g_sqrt = 1, projection W1 dot x = x for any x')
    W1_dot_x = dot_ops.W1_dot(x_2)
    assert np.allclose(W1_dot_x, x_2, atol=1e-14)
    print('Done. W1_dot(x_random) == x_random')

    # projection K1
    print('Under the condition, p_eq = 1 and g_sqrt = 1, projection K1 dot x = x for any x')
    K1_dot_x = dot_ops.K1_dot(x_3)
    assert np.allclose(K1_dot_x, x_3, atol=1e-14)
    print('Done. K1_dot(x_random) == x_random')
    print()

    # projection K10
    print('Under the condition, p_eq = 1, projection K10 dot x = x for any x')
    K10_dot_x = dot_ops.K10_dot(x_0)
    assert np.allclose(K10_dot_x, x_0, atol=1e-14)
    print('Done. K10_dot(x_random) == x_random')
    print()

    # projection S10
    print('Under the condition, p_eq = 1, projection S10 dot x = x for any x')
    S10_dot_x = dot_ops.S10_dot(x_1)
    assert np.allclose(S10_dot_x, x_1, atol=1e-14)
    print('Done. S10_dot(x_random) == x_random')
    print()

    # ========== comparison test ========== #
    print('Comparison test with basic projectors for the projection operator Q1_dot, U1_dot, P1_dot, S1_dot ,T1_dot and X1_dot :')
    print()
    ################
    # dot operator #
    ################
    start = time.time()
    # Q1
    Q1_dot_x = dot_ops.Q1_dot(x_1)

    # U1
    U1_dot_x = dot_ops.U1_dot(x_1)

    # P1
    P1_dot_x = dot_ops.P1_dot(x_2)

    # S1
    S1_dot_x = dot_ops.S1_dot(x_1)

    # T1
    T1_dot_x = dot_ops.T1_dot(x_1)

    # X1
    X1_dot_x = dot_ops.X1_dot(x_1)

    ####################
    # Basic projectors #
    ####################
    # ========== construct random splines ========== #
    # V1_h
    def phi_11(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_DNN(eta1, eta2, eta3, x_11)

    def phi_12(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_NDN(eta1, eta2, eta3, x_12)

    def phi_13(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_NND(eta1, eta2, eta3, x_13)
    # V2_h

    def phi_21(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_NDD(eta1, eta2, eta3, x_21)

    def phi_22(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_DND(eta1, eta2, eta3, x_22)

    def phi_23(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_DDN(eta1, eta2, eta3, x_23)

    # Beq_Ginv_lambda1 for T1
    def Beq_Ginv_lambda1_1(eta1, eta2, eta3):
        return phi_11(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 0] * dot_ops.eq_MHD.b2_2(eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 0] * dot_ops.eq_MHD.b2_3(eta1, eta2, eta3)) +\
            phi_12(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 1] * dot_ops.eq_MHD.b2_2(eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 1] * dot_ops.eq_MHD.b2_3(eta1, eta2, eta3)) +\
            phi_13(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 2] * dot_ops.eq_MHD.b2_2(
                eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 2] * dot_ops.eq_MHD.b2_3(eta1, eta2, eta3))

    def Beq_Ginv_lambda1_2(eta1, eta2, eta3):
        return phi_11(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 0] * dot_ops.eq_MHD.b2_3(eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 0] * dot_ops.eq_MHD.b2_1(eta1, eta2, eta3)) +\
            phi_12(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 1] * dot_ops.eq_MHD.b2_3(eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 1] * dot_ops.eq_MHD.b2_1(eta1, eta2, eta3)) +\
            phi_13(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 2] * dot_ops.eq_MHD.b2_3(
                eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 2] * dot_ops.eq_MHD.b2_1(eta1, eta2, eta3))

    def Beq_Ginv_lambda1_3(eta1, eta2, eta3):
        return phi_11(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 0] * dot_ops.eq_MHD.b2_1(eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 0] * dot_ops.eq_MHD.b2_2(eta1, eta2, eta3)) +\
            phi_12(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 1] * dot_ops.eq_MHD.b2_1(eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 1] * dot_ops.eq_MHD.b2_2(eta1, eta2, eta3)) +\
            phi_13(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 2] * dot_ops.eq_MHD.b2_1(
                eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 2] * dot_ops.eq_MHD.b2_2(eta1, eta2, eta3))

    # rhoeq_Ginv_lambda1 for Q1
    def rhoeq_Ginv_lambda1_1(eta1, eta2, eta3):
        return phi_11(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 0] * dot_ops.eq_MHD.n3(eta1, eta2, eta3) +\
            phi_12(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 1] * dot_ops.eq_MHD.n3(eta1, eta2, eta3) +\
            phi_13(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1,
                                                                        eta2, eta3)[0, 2] * dot_ops.eq_MHD.n3(eta1, eta2, eta3)

    def rhoeq_Ginv_lambda1_2(eta1, eta2, eta3):
        return phi_11(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 0] * dot_ops.eq_MHD.n3(eta1, eta2, eta3) +\
            phi_12(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 1] * dot_ops.eq_MHD.n3(eta1, eta2, eta3) +\
            phi_13(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1,
                                                                        eta2, eta3)[1, 2] * dot_ops.eq_MHD.n3(eta1, eta2, eta3)

    def rhoeq_Ginv_lambda1_3(eta1, eta2, eta3):
        return phi_11(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 0] * dot_ops.eq_MHD.n3(eta1, eta2, eta3) +\
            phi_12(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 1] * dot_ops.eq_MHD.n3(eta1, eta2, eta3) +\
            phi_13(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1,
                                                                        eta2, eta3)[2, 2] * dot_ops.eq_MHD.n3(eta1, eta2, eta3)

    # gsqrt_Ginv_lambda1 for U1
    def gsqrt_Ginv_lambda1_1(eta1, eta2, eta3):
        return dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * (phi_11(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 0] +
                                                                       phi_12(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 1] +
                                                                       phi_13(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 2])

    def gsqrt_Ginv_lambda1_2(eta1, eta2, eta3):
        return dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * (phi_11(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 0] +
                                                                       phi_12(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 1] +
                                                                       phi_13(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 2])

    def gsqrt_Ginv_lambda1_3(eta1, eta2, eta3):
        return dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * (phi_11(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 0] +
                                                                       phi_12(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 1] +
                                                                       phi_13(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 2])

    # jeq_gsqrt_lambda2 for P1
    def jeq_gsqrt_lambda2_1(eta1, eta2, eta3):
        return phi_22(eta1, eta2, eta3) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * -dot_ops.eq_MHD.j2_3(eta1, eta2, eta3) + phi_23(eta1, eta2, eta3) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * dot_ops.eq_MHD.j2_2(eta1, eta2, eta3)

    def jeq_gsqrt_lambda2_2(eta1, eta2, eta3):
        return phi_21(eta1, eta2, eta3) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * dot_ops.eq_MHD.j2_3(eta1, eta2, eta3) - phi_23(eta1, eta2, eta3) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * dot_ops.eq_MHD.j2_1(eta1, eta2, eta3)

    def jeq_gsqrt_lambda2_3(eta1, eta2, eta3):
        return phi_21(eta1, eta2, eta3) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * -dot_ops.eq_MHD.j2_2(eta1, eta2, eta3) + phi_22(eta1, eta2, eta3) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * dot_ops.eq_MHD.j2_1(eta1, eta2, eta3)

    # peq_Ginv_lambda1 for S1
    def peq_Ginv_lambda1_1(eta1, eta2, eta3):
        return phi_11(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 0] * dot_ops.eq_MHD.p3(eta1, eta2, eta3) +\
            phi_12(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 1] * dot_ops.eq_MHD.p3(eta1, eta2, eta3) +\
            phi_13(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1,
                                                                        eta2, eta3)[0, 2] * dot_ops.eq_MHD.p3(eta1, eta2, eta3)

    def peq_Ginv_lambda1_2(eta1, eta2, eta3):
        return phi_11(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 0] * dot_ops.eq_MHD.p3(eta1, eta2, eta3) +\
            phi_12(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 1] * dot_ops.eq_MHD.p3(eta1, eta2, eta3) +\
            phi_13(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1,
                                                                        eta2, eta3)[1, 2] * dot_ops.eq_MHD.p3(eta1, eta2, eta3)

    def peq_Ginv_lambda1_3(eta1, eta2, eta3):
        return phi_11(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 0] * dot_ops.eq_MHD.p3(eta1, eta2, eta3) +\
            phi_12(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 1] * dot_ops.eq_MHD.p3(eta1, eta2, eta3) +\
            phi_13(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric_inv(eta1,
                                                                        eta2, eta3)[2, 2] * dot_ops.eq_MHD.p3(eta1, eta2, eta3)

    # DFinv_T_lambda1 for X1
    def DFinv_T_lambda1_1(eta1, eta2, eta3):
        return phi_11(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian_inv(eta1, eta2, eta3)[0, 0] +\
            phi_12(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian_inv(eta1, eta2, eta3)[1, 0] +\
            phi_13(eta1, eta2, eta3) * \
            dot_ops.eq_MHD.domain.jacobian_inv(eta1, eta2, eta3)[2, 0]

    def DFinv_T_lambda1_2(eta1, eta2, eta3):
        return phi_11(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian_inv(eta1, eta2, eta3)[0, 1] +\
            phi_12(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian_inv(eta1, eta2, eta3)[1, 1] +\
            phi_13(eta1, eta2, eta3) * \
            dot_ops.eq_MHD.domain.jacobian_inv(eta1, eta2, eta3)[2, 1]

    def DFinv_T_lambda1_3(eta1, eta2, eta3):
        return phi_11(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian_inv(eta1, eta2, eta3)[0, 2] +\
            phi_12(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian_inv(eta1, eta2, eta3)[1, 2] +\
            phi_13(eta1, eta2, eta3) * \
            dot_ops.eq_MHD.domain.jacobian_inv(eta1, eta2, eta3)[2, 2]

    # ========== apply basic projection operator ========== #
    # project Beq_Ginv_lambda1 on V1_h space [ T1 ]:
    c11, c12, c13 = tensor_space_FEM.projectors.PI_1(
        Beq_Ginv_lambda1_1, Beq_Ginv_lambda1_2, Beq_Ginv_lambda1_3)
    c = np.concatenate((c11.flatten(), c12.flatten(), c13.flatten()))

    assert np.allclose(c, T1_dot_x, atol=1e-14)
    print(
        'pi1 projection of Beq_Ginv_lambda1[ T1 ] using both projectors are identical')

    # project rhoeq_Ginv_lambda1 on V1_h space [ Q1 ]:
    c21, c22, c23 = tensor_space_FEM.projectors.PI_2(
        rhoeq_Ginv_lambda1_1, rhoeq_Ginv_lambda1_2, rhoeq_Ginv_lambda1_3)
    c = np.concatenate((c21.flatten(), c22.flatten(), c23.flatten()))

    assert np.allclose(c, Q1_dot_x, atol=1e-14)
    print(
        'pi2 projection of rhoeq_Ginv_lambda1[ Q1 ] using both projectors are identical')

    # project gsqrt_Ginv_lambda1 on V2_h space [ U1 ]:
    c21, c22, c23 = tensor_space_FEM.projectors.PI_2(
        gsqrt_Ginv_lambda1_1, gsqrt_Ginv_lambda1_2, gsqrt_Ginv_lambda1_3)
    c = np.concatenate((c21.flatten(), c22.flatten(), c23.flatten()))

    assert np.allclose(c, U1_dot_x, atol=1e-14)
    print(
        'pi2 projection of gsqrt_Ginv_lambda1[ U1 ] using both projectors are identical')

    # project jeq_gsqrt_lambda2 on V1_h space [ P1 ]:
    c11, c12, c13 = tensor_space_FEM.projectors.PI_1(
        jeq_gsqrt_lambda2_1, jeq_gsqrt_lambda2_2, jeq_gsqrt_lambda2_3)
    c = np.concatenate((c11.flatten(), c12.flatten(), c13.flatten()))

    assert np.allclose(c, P1_dot_x, atol=1e-14)
    print(
        'pi1 projection of jeq_gsqrt_lambda2[ P1 ] using both projectors are identical')

    # project peq_Ginv_lambda1 on V2_h space [ S1 ]:
    c21, c22, c23 = tensor_space_FEM.projectors.PI_2(
        peq_Ginv_lambda1_1, peq_Ginv_lambda1_2, peq_Ginv_lambda1_3)
    c = np.concatenate((c21.flatten(), c22.flatten(), c23.flatten()))

    assert np.allclose(c, S1_dot_x, atol=1e-14)
    print(
        'pi2 projection of peq_Ginv_lambda1[ S1 ] using both projectors are identical')

    # project DFinv_T_lambda1 on V0_h space [ X1 ]:
    c01 = tensor_space_FEM.projectors.PI_0(DFinv_T_lambda1_1)
    c02 = tensor_space_FEM.projectors.PI_0(DFinv_T_lambda1_2)
    c03 = tensor_space_FEM.projectors.PI_0(DFinv_T_lambda1_3)
    c = np.concatenate((c01.flatten(), c02.flatten(), c03.flatten()))

    assert np.allclose(c, np.concatenate(
        (X1_dot_x[0], X1_dot_x[1], X1_dot_x[2])), atol=1e-14)
    print(
        'pi0 projection of DFinv_T_lambda1[ X1 ] using both projectors are identical')
    print()


def test_2form_projectors_dot():
    """
    TODO
    """

    import sys
    sys.path.append('..')

    import numpy as np
    import time

    from struphy.geometry import domains

    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d
    from struphy.eigenvalue_solvers.spline_space import Tensor_spline_space

    from struphy.fields_background.mhd_equil.equils import HomogenSlab

    from struphy.eigenvalue_solvers.legacy import mhd_operators_MF as mhd_op_V2

    # spline spaces
    Nel = [7, 7, 7]
    p = [4, 4, 4]
    spl_kind = [True, True, True]
    n_quad = p.copy()
    bc = ['d', 'd']
    dom_type = 'Cuboid'

    # 1d B-spline spline spaces for finite elements
    spaces_FEM = [Spline_space_1d(Nel_i, p_i, spl_kind_i, n_quad_i, bc)
                  for Nel_i, p_i, spl_kind_i, n_quad_i in zip(Nel, p, spl_kind, n_quad)]

    # 1D commuting projectors
    spaces_FEM[0].set_projectors(n_quad[0])
    spaces_FEM[1].set_projectors(n_quad[1])
    spaces_FEM[2].set_projectors(n_quad[2])

    # 3d tensor-product B-spline space for finite elements
    tensor_space_FEM = Tensor_spline_space(spaces_FEM)

    tensor_space_FEM.set_projectors()

    # domain
    domain_class = getattr(domains, dom_type)
    domain = domain_class()

    # assemble mass matrices
    tensor_space_FEM.assemble_Mk(domain, 'V0')
    tensor_space_FEM.assemble_Mk(domain, 'V1')
    tensor_space_FEM.assemble_Mk(domain, 'V2')
    tensor_space_FEM.assemble_Mk(domain, 'V3')
    print('Assembly of mass matrices done.')
    print()

    # mhd projectors dot operator
    eq_MHD = HomogenSlab(
        **{'B0x': 0., 'B0y': 0., 'B0z': 1., 'beta': 2., 'n0': 1.})
    eq_MHD.domain = domain

    dot_ops = mhd_op_V2.projectors_dot_x(tensor_space_FEM, eq_MHD)

    # random x which is going to product with projectors
    x_0 = np.random.rand(tensor_space_FEM.Ntot_0form)
    x_1 = np.random.rand(tensor_space_FEM.Ntot_1form_cum[-1])
    x_11, x_12, x_13 = tensor_space_FEM.extract_1(x_1)
    x_2 = np.random.rand(tensor_space_FEM.Ntot_2form_cum[-1])
    x_21, x_22, x_23 = tensor_space_FEM.extract_2(x_2)
    x_3 = np.random.rand(tensor_space_FEM.Ntot_3form)

    # test conditions
    print()
    print('MHD_equilibrium :')
    print('p_eq =', dot_ops.eq_MHD.p3(0., 0., 0.))
    print('r_eq =', dot_ops.eq_MHD.n3(0., 0., 0.))
    print('b_eq_x =', dot_ops.eq_MHD.b2_1(0., 0., 0.))
    print('b_eq_y =', dot_ops.eq_MHD.b2_2(0., 0., 0.))
    print('b_eq_z =', dot_ops.eq_MHD.b2_3(0., 0., 0.))
    print('j_eq_x =', dot_ops.eq_MHD.j2_1(0., 0., 0.))
    print('j_eq_y =', dot_ops.eq_MHD.j2_2(0., 0., 0.))
    print('j_eq_z =', dot_ops.eq_MHD.j2_3(0., 0., 0.))

    print()
    print('maping  :')
    print('dom_type = ' + str(dom_type))
    print('params_map = ' + str(domain.params_map))
    print()

    # ========== Identity test ========== #
    print('Identity test for the projection Q2_dot, S2_dot and K2_dot :')
    print('Calculating projection operator Q2_dot, S2_dot and K2_dot with random x is done!')
    print()
    # projection Q2
    print('Under the condition, rho_eq = 1 and g_sqrt = 1, projection Q2 dot x = x for all x')
    Q2_dot_x = dot_ops.Q2_dot(x_2)
    assert np.allclose(Q2_dot_x, x_2, atol=1e-14)
    print('Done. Q2_dot(x_random) == x_random')

    # projection S2
    print('Under the condition, p_eq = 1 and g_sqrt = 1, projection S2 dot x = x for all x')
    S2_dot_x = dot_ops.S2_dot(x_2)
    assert np.allclose(S2_dot_x, x_2, atol=1e-14)
    print('Done. S2_dot(x_random) == x_random')

    # projection K2
    print('Under the condition, p_eq = 1 and g_sqrt = 1, projection K2 dot x = x for all x')
    K2_dot_x = dot_ops.K2_dot(x_3)
    assert np.allclose(K2_dot_x, x_3, atol=1e-14)
    print('Done. K2_dot(x_random) == x_random')
    print()

    # ========== comparison test ========== #
    print('Comparison test with basic projectors for the projection operator T2_dot, P2_dot  and X2_dot :')
    print()

    ################
    # dot operator #
    ################
    # T2
    start = time.time()
    T2_dot_x = dot_ops.T2_dot(x_2)

    # P2
    P2_dot_x = dot_ops.P2_dot(x_2)

    # X2
    X2_dot_x = dot_ops.X2_dot(x_2)

    # S20
    S20_dot_x = dot_ops.S20_dot(x_2)

    # Z20
    Z20_dot_x = dot_ops.Z20_dot(x_2)

    # Y20
    Y20_dot_x = dot_ops.Y20_dot(x_0)

    ####################
    # Basic projectors #
    ####################
    # ========== construct random splines ========== #
    # V0_h
    def phi_0(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_NNN(eta1, eta2, eta3, x_0)
    # V1_h

    def phi_11(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_DNN(eta1, eta2, eta3, x_1)

    def phi_12(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_NDN(eta1, eta2, eta3, x_1)

    def phi_13(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_NND(eta1, eta2, eta3, x_1)
    # V2_h

    def phi_21(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_NDD(eta1, eta2, eta3, x_2)

    def phi_22(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_DND(eta1, eta2, eta3, x_2)

    def phi_23(eta1, eta2, eta3):
        return tensor_space_FEM.evaluate_DDN(eta1, eta2, eta3, x_2)

    # Beq_gsqrt_lambda2 for T2
    def Beq_gsqrt_lambda2_1(eta1, eta2, eta3):
        return phi_22(eta1, eta2, eta3) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * -dot_ops.eq_MHD.b2_3(eta1, eta2, eta3) + phi_23(eta1, eta2, eta3) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * dot_ops.eq_MHD.b2_2(eta1, eta2, eta3)

    def Beq_gsqrt_lambda2_2(eta1, eta2, eta3):
        return phi_21(eta1, eta2, eta3) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * dot_ops.eq_MHD.b2_3(eta1, eta2, eta3) - phi_23(eta1, eta2, eta3) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * dot_ops.eq_MHD.b2_1(eta1, eta2, eta3)

    def Beq_gsqrt_lambda2_3(eta1, eta2, eta3):
        return phi_21(eta1, eta2, eta3) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * -dot_ops.eq_MHD.b2_2(eta1, eta2, eta3) + phi_22(eta1, eta2, eta3) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3) * dot_ops.eq_MHD.b2_1(eta1, eta2, eta3)

    # Ginv_jeq_lambda2 for P2
    def Ginv_jeq_lambda2_1(eta1, eta2, eta3):
        return phi_21(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 1] * dot_ops.eq_MHD.j2_3(eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 2] * dot_ops.eq_MHD.j2_2(eta1, eta2, eta3)) +\
            phi_22(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 2] * dot_ops.eq_MHD.j2_1(eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 0] * dot_ops.eq_MHD.j2_3(eta1, eta2, eta3)) +\
            phi_23(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 0] * dot_ops.eq_MHD.j2_2(
                eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[0, 1] * dot_ops.eq_MHD.j2_1(eta1, eta2, eta3))

    def Ginv_jeq_lambda2_2(eta1, eta2, eta3):
        return phi_21(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 1] * dot_ops.eq_MHD.j2_3(eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 2] * dot_ops.eq_MHD.j2_2(eta1, eta2, eta3)) +\
            phi_22(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 2] * dot_ops.eq_MHD.j2_1(eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 0] * dot_ops.eq_MHD.j2_3(eta1, eta2, eta3)) +\
            phi_23(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 0] * dot_ops.eq_MHD.j2_2(
                eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[1, 1] * dot_ops.eq_MHD.j2_1(eta1, eta2, eta3))

    def Ginv_jeq_lambda2_3(eta1, eta2, eta3):
        return phi_21(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 1] * dot_ops.eq_MHD.j2_3(eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 2] * dot_ops.eq_MHD.j2_2(eta1, eta2, eta3)) +\
            phi_22(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 2] * dot_ops.eq_MHD.j2_1(eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 0] * dot_ops.eq_MHD.j2_3(eta1, eta2, eta3)) +\
            phi_23(eta1, eta2, eta3) * (dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 0] * dot_ops.eq_MHD.j2_2(
                eta1, eta2, eta3) - dot_ops.eq_MHD.domain.metric_inv(eta1, eta2, eta3)[2, 1] * dot_ops.eq_MHD.j2_1(eta1, eta2, eta3))

    # DF_gsqrt_lambda2 for X2
    def DF_gsqrt_lambda2_1(eta1, eta2, eta3):
        return (phi_21(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian(eta1, eta2, eta3)[0, 0] +
                phi_22(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian(eta1, eta2, eta3)[0, 1] +
                phi_23(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian(eta1, eta2, eta3)[0, 2]) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3)

    def DF_gsqrt_lambda2_2(eta1, eta2, eta3):
        return (phi_21(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian(eta1, eta2, eta3)[1, 0] +
                phi_22(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian(eta1, eta2, eta3)[1, 1] +
                phi_23(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian(eta1, eta2, eta3)[1, 2]) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3)

    def DF_gsqrt_lambda2_3(eta1, eta2, eta3):
        return (phi_21(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian(eta1, eta2, eta3)[2, 0] +
                phi_22(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian(eta1, eta2, eta3)[2, 1] +
                phi_23(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian(eta1, eta2, eta3)[2, 2]) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3)

    # peq_G_gsqrt_lambda2 for S20
    def peq_G_lambda2_1(eta1, eta2, eta3):
        return (phi_21(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_11')[0, 0] * dot_ops.eq_MHD.p0(eta1, eta2, eta3) +
                phi_22(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_12')[0, 1] * dot_ops.eq_MHD.p0(eta1, eta2, eta3) +
                phi_23(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_13')[0, 2] * dot_ops.eq_MHD.p0(eta1, eta2, eta3)) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3)

    def peq_G_lambda2_2(eta1, eta2, eta3):
        return (phi_21(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_21')[1, 0] * dot_ops.eq_MHD.p0(eta1, eta2, eta3) +
                phi_22(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_22')[1, 1] * dot_ops.eq_MHD.p0(eta1, eta2, eta3) +
                phi_23(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_23')[1, 2] * dot_ops.eq_MHD.p0(eta1, eta2, eta3)) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3)

    def peq_G_lambda2_3(eta1, eta2, eta3):
        return (phi_21(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_31')[2, 0] * dot_ops.eq_MHD.p0(eta1, eta2, eta3) +
                phi_22(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_32')[2, 1] * dot_ops.eq_MHD.p0(eta1, eta2, eta3) +
                phi_23(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_33')[2, 2] * dot_ops.eq_MHD.p0(eta1, eta2, eta3)) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3)
    # G_gsqrt_lambda2 for Z20

    def G_lambda2_1(eta1, eta2, eta3):
        return (phi_21(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_11')[0, 0] +
                phi_22(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_12')[0, 1] +
                phi_23(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_13')[0, 2]) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3)

    def G_lambda2_2(eta1, eta2, eta3):
        return (phi_21(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_21')[1, 0] +
                phi_22(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_22')[1, 1] +
                phi_23(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_23')[1, 2]) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3)

    def G_lambda2_3(eta1, eta2, eta3):
        return (phi_21(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_31')[2, 0] +
                phi_22(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_32')[2, 1] +
                phi_23(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.metric(eta1, eta2, eta3, 'g_33')[2, 2]) / dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3)

    # g_sqrt * lambda0 for Y20
    def gsqrt_lambda0(eta1, eta2, eta3):
        return phi_0(eta1, eta2, eta3) * dot_ops.eq_MHD.domain.jacobian_det(eta1, eta2, eta3)

    # ========== apply basic projection operator ========== #
    # project Beq_gsqrt_lambda2 on V1_h space [ T2 ]:
    c11, c12, c13 = tensor_space_FEM.projectors.PI_1(
        Beq_gsqrt_lambda2_1, Beq_gsqrt_lambda2_2, Beq_gsqrt_lambda2_3)
    c = np.concatenate((c11.flatten(), c12.flatten(), c13.flatten()))

    assert np.allclose(c, T2_dot_x, atol=1e-14)
    print(
        'pi1 projection of Beq_gsqrt_lambda2[ T2 ] using both projectors are identical')

    # project Ginv_jeq_lambda2 on V2_h space [ P2 ]:
    c21, c22, c23 = tensor_space_FEM.projectors.PI_2(
        Ginv_jeq_lambda2_1, Ginv_jeq_lambda2_2, Ginv_jeq_lambda2_3)
    c = np.concatenate((c21.flatten(), c22.flatten(), c23.flatten()))

    assert np.allclose(c, P2_dot_x, atol=1e-14)
    print(
        'pi2 projection of Ginv_jeq_lambda2[ P2 ] using both projectors are identical')

    # project DF_gsqrt_lambda2 on V0_h space [ X2 ]:
    c01 = tensor_space_FEM.projectors.PI_0(DF_gsqrt_lambda2_1)
    c02 = tensor_space_FEM.projectors.PI_0(DF_gsqrt_lambda2_2)
    c03 = tensor_space_FEM.projectors.PI_0(DF_gsqrt_lambda2_3)
    c = np.concatenate((c01.flatten(), c02.flatten(), c03.flatten()))

    assert np.allclose(c, np.concatenate(
        (X2_dot_x[0], X2_dot_x[1], X2_dot_x[2])), atol=1e-14)
    print(
        'pi0 projection of DF_gsqrt_lambda2[ X2 ] using both projectors are identical')

    # project p_eq_G_lambda2 on V1_h space [ S20 ]:
    c11, c12, c13 = tensor_space_FEM.projectors.PI_1(
        peq_G_lambda2_1, peq_G_lambda2_2, peq_G_lambda2_3)
    c = np.concatenate((c11.flatten(), c12.flatten(), c13.flatten()))

    assert np.allclose(c, S20_dot_x, atol=1e-14)
    print(
        'pi1 projection of p_eq_G_lambda2[ S20 ] using both projectors are identical')

    # project G_lambda2 on V1_h space [ Z20 ]:
    c11, c12, c13 = tensor_space_FEM.projectors.PI_1(
        G_lambda2_1, G_lambda2_2, G_lambda2_3)
    c = np.concatenate((c11.flatten(), c12.flatten(), c13.flatten()))

    assert np.allclose(c, Z20_dot_x, atol=1e-14)
    print(
        'pi1 projection of G_lambda2[ Z20 ] using both projectors are identical')

    # project gsqrt_lambda0 on V3_h space [Y20]
    c3 = tensor_space_FEM.projectors.PI_3(gsqrt_lambda0)

    assert np.allclose(c3.flatten(), Y20_dot_x, atol=1e-14)
    print(
        'pi3 projection of gsqrt_lambda0 [ Y20 ] using both projectors are identical')
    print()


def test_1form_symmetric():
    """
    TODO
    """

    import sys
    sys.path.append('..')

    import numpy as np

    from struphy.geometry import domains

    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d
    from struphy.eigenvalue_solvers.spline_space import Tensor_spline_space

    from struphy.fields_background.mhd_equil.equils import HomogenSlab

    from struphy.eigenvalue_solvers.legacy import mhd_operators_MF as mhd_op_V2

    # spline spaces
    Nel = [7, 7, 7]
    p = [4, 4, 4]
    spl_kind = [True, True, True]
    n_quad = p.copy()
    bc = ['d', 'd']
    dom_type = 'Cuboid'

    # 1d B-spline spline spaces for finite elements
    spaces_FEM = [Spline_space_1d(Nel_i, p_i, spl_kind_i, n_quad_i, bc)
                  for Nel_i, p_i, spl_kind_i, n_quad_i in zip(Nel, p, spl_kind, n_quad)]

    # 1D commuting projectors
    spaces_FEM[0].set_projectors(n_quad[0])
    spaces_FEM[1].set_projectors(n_quad[1])
    spaces_FEM[2].set_projectors(n_quad[2])

    # 3d tensor-product B-spline space for finite elements
    tensor_space_FEM = Tensor_spline_space(spaces_FEM)

    tensor_space_FEM.set_projectors()

    # domain
    domain_class = getattr(domains, dom_type)
    domain = domain_class()

    # assemble mass matrices
    tensor_space_FEM.assemble_Mk(domain, 'V0')
    tensor_space_FEM.assemble_Mk(domain, 'V1')
    tensor_space_FEM.assemble_Mk(domain, 'V2')
    tensor_space_FEM.assemble_Mk(domain, 'V3')
    print('Assembly of mass matrices done.')
    print()

    # mhd projectors dot operator
    eq_MHD = HomogenSlab(
        **{'B0x': 0., 'B0y': 0., 'B0z': 1., 'beta': 2., 'n0': 1.})
    eq_MHD.domain = domain

    dot_ops = mhd_op_V2.projectors_dot_x(tensor_space_FEM, eq_MHD)

    # test conditions
    print()
    print('MHD_equilibrium :')
    print('p_eq =', dot_ops.eq_MHD.p3(0., 0., 0.))
    print('r_eq =', dot_ops.eq_MHD.n3(0., 0., 0.))
    print('b_eq_x =', dot_ops.eq_MHD.b2_1(0., 0., 0.))
    print('b_eq_y =', dot_ops.eq_MHD.b2_2(0., 0., 0.))
    print('b_eq_z =', dot_ops.eq_MHD.b2_3(0., 0., 0.))
    print('j_eq_x =', dot_ops.eq_MHD.j2_1(0., 0., 0.))
    print('j_eq_y =', dot_ops.eq_MHD.j2_2(0., 0., 0.))
    print('j_eq_z =', dot_ops.eq_MHD.j2_3(0., 0., 0.))

    print()
    print('maping  :')
    print('dom_type = ' + str(dom_type))
    print('params_map = ' + str(domain.params_map))

    print()

    # ========== Symmetric test ========== #
    print('With given random x and y, x.T M.T M y == y.T M.T M x should be always satisfied')

    num = 100

    # X1 is not yet implemented
    res_Q1_1 = np.zeros(num)
    res_Q1_2 = np.zeros(num)
    res_W1_1 = np.zeros(num)
    res_W1_2 = np.zeros(num)
    res_U1_1 = np.zeros(num)
    res_U1_2 = np.zeros(num)
    res_P1_1 = np.zeros(num)
    res_P1_2 = np.zeros(num)
    res_S1_1 = np.zeros(num)
    res_S1_2 = np.zeros(num)
    res_K1_1 = np.zeros(num)
    res_K1_2 = np.zeros(num)
    res_T1_1 = np.zeros(num)
    res_T1_2 = np.zeros(num)
    res_X1_1 = np.zeros(num)
    res_X1_2 = np.zeros(num)
    res_S10_1 = np.zeros(num)
    res_S10_2 = np.zeros(num)
    res_K10_1 = np.zeros(num)
    res_K10_2 = np.zeros(num)

    for i in range(num):
        # 0form random x and y
        x_0 = np.random.rand(tensor_space_FEM.Ntot_0form)
        y_0 = np.random.rand(tensor_space_FEM.Ntot_0form)

        # 1form random x and y
        x_1 = np.random.rand(tensor_space_FEM.Ntot_1form_cum[-1])
        y_1 = np.random.rand(tensor_space_FEM.Ntot_1form_cum[-1])

        # 2form random x and y
        x_2 = np.random.rand(tensor_space_FEM.Ntot_2form_cum[-1])
        y_2 = np.random.rand(tensor_space_FEM.Ntot_2form_cum[-1])

        # 3form random x and y
        x_3 = np.random.rand(tensor_space_FEM.Ntot_3form)
        y_3 = np.random.rand(tensor_space_FEM.Ntot_3form)

        res_Q1_1[i] = x_1.T.dot(dot_ops.transpose_Q1_dot(dot_ops.Q1_dot(y_1)))
        res_Q1_2[i] = y_1.T.dot(dot_ops.transpose_Q1_dot(dot_ops.Q1_dot(x_1)))
        res_W1_1[i] = x_1.T.dot(dot_ops.transpose_W1_dot(dot_ops.W1_dot(y_1)))
        res_W1_2[i] = y_1.T.dot(dot_ops.transpose_W1_dot(dot_ops.W1_dot(x_1)))
        res_U1_1[i] = x_1.T.dot(dot_ops.transpose_U1_dot(dot_ops.U1_dot(y_1)))
        res_U1_2[i] = y_1.T.dot(dot_ops.transpose_U1_dot(dot_ops.U1_dot(x_1)))
        res_P1_1[i] = x_2.T.dot(dot_ops.transpose_P1_dot(dot_ops.P1_dot(y_2)))
        res_P1_2[i] = y_2.T.dot(dot_ops.transpose_P1_dot(dot_ops.P1_dot(x_2)))
        res_S1_1[i] = x_1.T.dot(dot_ops.transpose_S1_dot(dot_ops.S1_dot(y_1)))
        res_S1_2[i] = y_1.T.dot(dot_ops.transpose_S1_dot(dot_ops.S1_dot(x_1)))
        res_K1_1[i] = x_3.T.dot(dot_ops.transpose_K1_dot(dot_ops.K1_dot(y_3)))
        res_K1_2[i] = y_3.T.dot(dot_ops.transpose_K1_dot(dot_ops.K1_dot(x_3)))
        res_T1_1[i] = x_1.T.dot(dot_ops.transpose_T1_dot(dot_ops.T1_dot(y_1)))
        res_T1_2[i] = y_1.T.dot(dot_ops.transpose_T1_dot(dot_ops.T1_dot(x_1)))
        res_X1_1[i] = x_1.T.dot(dot_ops.transpose_X1_dot(dot_ops.X1_dot(y_1)))
        res_X1_2[i] = y_1.T.dot(dot_ops.transpose_X1_dot(dot_ops.X1_dot(x_1)))
        res_S10_1[i] = x_1.T.dot(
            dot_ops.transpose_S10_dot(dot_ops.S10_dot(y_1)))
        res_S10_2[i] = y_1.T.dot(
            dot_ops.transpose_S10_dot(dot_ops.S10_dot(x_1)))
        res_K10_1[i] = x_0.T.dot(
            dot_ops.transpose_K10_dot(dot_ops.K10_dot(y_0)))
        res_K10_2[i] = y_0.T.dot(
            dot_ops.transpose_K10_dot(dot_ops.K10_dot(x_0)))

    tol = 1e-14

    assert np.allclose(res_Q1_1, res_Q1_2, atol=tol)
    print('(Q1.T Q1) is a symmetric operator')
    assert np.allclose(res_W1_1, res_W1_2, atol=tol)
    print('(W1.T W1) is a symmetric operator')
    assert np.allclose(res_U1_1, res_U1_2, atol=tol)
    print('(U1.T U1) is a symmetric operator')
    assert np.allclose(res_P1_1, res_P1_2, atol=tol)
    print('(P1.T P1) is a symmetric operator')
    assert np.allclose(res_S1_1, res_S1_2, atol=tol)
    print('(S1.T S1) is a symmetric operator')
    assert np.allclose(res_K1_1, res_K1_2, atol=tol)
    print('(K1.T K1) is a symmetric operator')
    assert np.allclose(res_T1_1, res_T1_2, atol=tol)
    print('(T1.T T1) is a symmetric operator')
    assert np.allclose(res_X1_1, res_X1_2, atol=tol)
    print('(X1.T X1) is a symmetric operator')
    assert np.allclose(res_S10_1, res_S10_2, atol=tol)
    print('(S10.T S10) is a symmetric operator')
    assert np.allclose(res_K10_1, res_K10_2, atol=tol)
    print('(K10.T K10) is a symmetric operator')
    print()


def test_2form_symmetric():
    """
    TODO
    """

    import sys
    sys.path.append('..')

    import numpy as np

    from struphy.geometry import domains

    from struphy.eigenvalue_solvers.spline_space import Spline_space_1d
    from struphy.eigenvalue_solvers.spline_space import Tensor_spline_space

    from struphy.fields_background.mhd_equil.equils import HomogenSlab

    from struphy.eigenvalue_solvers.legacy import mhd_operators_MF as mhd_op_V2

    # spline spaces
    Nel = [7, 7, 7]
    p = [4, 4, 4]
    spl_kind = [True, True, True]
    n_quad = p.copy()
    bc = ['d', 'd']
    dom_type = 'Cuboid'

    # 1d B-spline spline spaces for finite elements
    spaces_FEM = [Spline_space_1d(Nel_i, p_i, spl_kind_i, n_quad_i, bc)
                  for Nel_i, p_i, spl_kind_i, n_quad_i in zip(Nel, p, spl_kind, n_quad)]

    # 1D commuting projectors
    spaces_FEM[0].set_projectors(n_quad[0])
    spaces_FEM[1].set_projectors(n_quad[1])
    spaces_FEM[2].set_projectors(n_quad[2])

    # 3d tensor-product B-spline space for finite elements
    tensor_space_FEM = Tensor_spline_space(spaces_FEM)

    tensor_space_FEM.set_projectors()

    # domain
    domain_class = getattr(domains, dom_type)
    domain = domain_class()

    # assemble mass matrices
    tensor_space_FEM.assemble_Mk(domain, 'V0')
    tensor_space_FEM.assemble_Mk(domain, 'V1')
    tensor_space_FEM.assemble_Mk(domain, 'V2')
    tensor_space_FEM.assemble_Mk(domain, 'V3')
    print('Assembly of mass matrices done.')
    print()

    # mhd projectors dot operator
    eq_MHD = HomogenSlab(
        **{'B0x': 0., 'B0y': 0., 'B0z': 1., 'beta': 2., 'n0': 1.})
    eq_MHD.domain = domain

    dot_ops = mhd_op_V2.projectors_dot_x(tensor_space_FEM, eq_MHD)

    # test conditions
    print()
    print('MHD_equilibrium :')
    print('p_eq =', dot_ops.eq_MHD.p3(0., 0., 0.))
    print('r_eq =', dot_ops.eq_MHD.n3(0., 0., 0.))
    print('b_eq_x =', dot_ops.eq_MHD.b2_1(0., 0., 0.))
    print('b_eq_y =', dot_ops.eq_MHD.b2_2(0., 0., 0.))
    print('b_eq_z =', dot_ops.eq_MHD.b2_3(0., 0., 0.))
    print('j_eq_x =', dot_ops.eq_MHD.j2_1(0., 0., 0.))
    print('j_eq_y =', dot_ops.eq_MHD.j2_2(0., 0., 0.))
    print('j_eq_z =', dot_ops.eq_MHD.j2_3(0., 0., 0.))
    print()

    print()
    print('maping  :')
    print('dom_type = ' + str(dom_type))
    print('params_map = ' + str(domain.params_map))

    # print('G = ')
    # print(domain.metric(0., 0., 0.)[0, 0], domain.metric(0., 0., 0.)[0, 1], domain.metric(0., 0., 0.)[0, 2])
    # print(domain.metric(0., 0., 0.)[1, 0], domain.metric(0., 0., 0.)[1, 1], domain.metric(0., 0., 0.)[1, 2])
    # print(domain.metric(0., 0., 0.)[2, 0], domain.metric(0., 0., 0.)[2, 1], domain.metric(0., 0., 0.)[2, 2])

    # print('g_sqrt = ')
    # print(domain.jacobian_det(0., 0., 0.))
    # print()

    # ========== Symmetric test ========== #
    print('With given random x and y, x.T M.T M y == y.T M.T M x should be always satisfied')

    num = 100

    # X1 is not yet implemented
    res_Q2_1 = np.zeros(num)
    res_Q2_2 = np.zeros(num)
    res_P2_1 = np.zeros(num)
    res_P2_2 = np.zeros(num)
    res_S2_1 = np.zeros(num)
    res_S2_2 = np.zeros(num)
    res_K2_1 = np.zeros(num)
    res_K2_2 = np.zeros(num)
    res_T2_1 = np.zeros(num)
    res_T2_2 = np.zeros(num)
    res_X2_1 = np.zeros(num)
    res_X2_2 = np.zeros(num)
    res_Y20_1 = np.zeros(num)
    res_Y20_2 = np.zeros(num)
    res_S20_1 = np.zeros(num)
    res_S20_2 = np.zeros(num)
    res_Z20_1 = np.zeros(num)
    res_Z20_2 = np.zeros(num)

    for i in range(num):
        # 0form random x and y
        x_0 = np.random.rand(tensor_space_FEM.Ntot_0form)
        y_0 = np.random.rand(tensor_space_FEM.Ntot_0form)

        # 1form random x and y
        x_1 = np.random.rand(tensor_space_FEM.Ntot_1form_cum[-1])
        y_1 = np.random.rand(tensor_space_FEM.Ntot_1form_cum[-1])

        # 2form random x and y
        x_2 = np.random.rand(tensor_space_FEM.Ntot_2form_cum[-1])
        y_2 = np.random.rand(tensor_space_FEM.Ntot_2form_cum[-1])

        # 3form random x and y
        x_3 = np.random.rand(tensor_space_FEM.Ntot_3form)
        y_3 = np.random.rand(tensor_space_FEM.Ntot_3form)

        res_Q2_1[i] = x_2.T.dot(dot_ops.transpose_Q2_dot(dot_ops.Q2_dot(y_2)))
        res_Q2_2[i] = y_2.T.dot(dot_ops.transpose_Q2_dot(dot_ops.Q2_dot(x_2)))
        res_T2_1[i] = x_2.T.dot(dot_ops.transpose_T2_dot(dot_ops.T2_dot(y_2)))
        res_T2_2[i] = y_2.T.dot(dot_ops.transpose_T2_dot(dot_ops.T2_dot(x_2)))
        res_P2_1[i] = x_2.T.dot(dot_ops.transpose_P2_dot(dot_ops.P2_dot(y_2)))
        res_P2_2[i] = y_2.T.dot(dot_ops.transpose_P2_dot(dot_ops.P2_dot(x_2)))
        res_S2_1[i] = x_2.T.dot(dot_ops.transpose_S2_dot(dot_ops.S2_dot(y_2)))
        res_S2_2[i] = y_2.T.dot(dot_ops.transpose_S2_dot(dot_ops.S2_dot(x_2)))
        res_K2_1[i] = x_3.T.dot(dot_ops.transpose_K2_dot(dot_ops.K2_dot(y_3)))
        res_K2_2[i] = y_3.T.dot(dot_ops.transpose_K2_dot(dot_ops.K2_dot(x_3)))
        res_X2_1[i] = x_2.T.dot(dot_ops.transpose_X2_dot(dot_ops.X2_dot(y_2)))
        res_X2_2[i] = y_2.T.dot(dot_ops.transpose_X2_dot(dot_ops.X2_dot(x_2)))
        res_Y20_1[i] = x_0.T.dot(
            dot_ops.transpose_Y20_dot(dot_ops.Y20_dot(y_0)))
        res_Y20_2[i] = y_0.T.dot(
            dot_ops.transpose_Y20_dot(dot_ops.Y20_dot(x_0)))
        res_S20_1[i] = x_2.T.dot(
            dot_ops.transpose_S20_dot(dot_ops.S20_dot(y_2)))
        res_S20_2[i] = y_2.T.dot(
            dot_ops.transpose_S20_dot(dot_ops.S20_dot(x_2)))
        res_Z20_1[i] = x_2.T.dot(
            dot_ops.transpose_Z20_dot(dot_ops.Z20_dot(y_2)))
        res_Z20_2[i] = y_2.T.dot(
            dot_ops.transpose_Z20_dot(dot_ops.Z20_dot(x_2)))

    tol = 1e-14

    assert np.allclose(res_Q2_1, res_Q2_2, atol=tol)
    print('(Q2.T Q2) is a symmetric operator')
    assert np.allclose(res_T2_1, res_T2_2, atol=tol)
    print('(T2.T T2) is a symmetric operator')
    assert np.allclose(res_P2_1, res_P2_2, atol=tol)
    print('(P2.T P2) is a symmetric operator')
    assert np.allclose(res_S2_1, res_S2_2, atol=tol)
    print('(S2.T S2) is a symmetric operator')
    assert np.allclose(res_K2_1, res_K2_2, atol=tol)
    print('(K2.T K2) is a symmetric operator')
    assert np.allclose(res_X2_1, res_X2_2, atol=tol)
    print('(X2.T X2) is a symmetric operator')
    assert np.allclose(res_Y20_1, res_Y20_2, atol=tol)
    print('(Y20.T Y20) is a symmetric operator')
    assert np.allclose(res_S20_1, res_S20_2, atol=tol)
    print('(S20.T S20) is a symmetric operator')
    assert np.allclose(res_Z20_1, res_Z20_2, atol=tol)
    print('(Z20.T Z20) is a symmetric operator')
    print()


if __name__ == '__main__':
    test_1form_projectors_dot()
    test_2form_projectors_dot()
    test_1form_symmetric()
    test_2form_symmetric()
