import numpy as np
import scipy.sparse as spa

from struphy.eigenvalue_solvers.projectors_global import Projectors_tensor_3d
from struphy.eigenvalue_solvers.spline_space import Tensor_spline_space
from struphy.linear_algebra.linalg_kron import kron_matvec_3d, kron_solve_3d


# =================================================================================================
class projectors_dot_x:
    """
    Caclulate the product of vector 'x' with the several kinds of projection matrices.
    Global projectors based on tensor product are used.

    Parameters
    ----------
        space : obj
            Tensor_spline_space object.

        eq_MHD : obj
            Equilibrium_mhd object (pullbacks must be enabled).

    Notes
    -----
        Implemented operators (transpose also implemented) with

        * MHD with velocity (up) as 1-form:
            ===================================================== =========================== =====================
            operator                                              dim of matrix               verification method
            ===================================================== =========================== =====================
            Q1  = pi_2[rho_eq * G_inv * lambda^1]                  R^{N^2 x N^1}               comparison test with basic projector
            W1  = pi_1[rho_eq / g_sqrt * lambda^1]                 R^{N^1 x N^1}               identity test
            U1  = pi_2[g_sqrt * G_inv * lambda^1]                  R^{N^2 x N^1}               comparison test with basic projector
            P1  = pi_1[j_eq / g_sqrt * lambda^2]                   R^{N^1 x N^2}               comparison test with basic projector
            S1  = pi_2[p_eq * G_inv * lambda^1]                    R^{N^2 x N^1}               comparison test with basic projector
            S10 = pi_1[p_eq * lambda^1]                            R^{N^1 x N^1}               identity test
            K1  = pi_3[p_eq / g_sqrt * lambda^3]                   R^{N^3 x N^3}               identity test
            K10 = pi_0[p_eq * lambda^0]                            R^{N^0 x N^0}               identity test
            T1  = pi_1[B_eq * G_inv * lambda^1]                    R^{N^1 x N^1}               comparison test with basic projector
            X1  = pi_0[DF^-T * lambda^1]                           R^{N^0 x 3 x N^1}           comparison test with basic projector
            ===================================================== =========================== =====================

        * MHD with velocity (up) as 2-form:
            ===================================================== ================= =====================
            operator                                              dim of matrx      verification method
            ===================================================== ================= =====================
            Q2  = pi_2[rho_eq / g_sqrt * lambda^2]                R^{N^2 x N^2}      identity test
            T2  = pi_1[B_eq / g_sqrt * lambda^2]                  R^{N^1 x N^2}      comparison test with basic projector
            P2  = pi_2[G_inv * j_eq * lambda^2]                   R^{N^2 x N^2}      comparison test with basic projector
            S2  = pi_2[p_eq / g_sqrt * lambda^2]                  R^{N^2 x N^2}      identity test
            K2  = pi_3[p_eq / g_sqrt * lambda^3]                  R^{N^3 x N^3}      identity test
            X2  = pi_0[DF / g_sqrt * lambda^2]                    R^{N^0 x 3 x N^2}  comparison test with basic projector
            Z20 = pi_1[G / g_sqrt * lambda^2]                     R^{N^1 x N^2}      comparison test with basic projector
            Y20 = pi_3[g_sqrt * lambda^0]                         R^{N^3 x N^0}      comparison test with basic projector
            S20 = pi_1[p_eq * G / g_sqrt * lambda^2]              R^{N^1 x N^2}      comparison test with basic projector
            ===================================================== ================= =====================
    """

    def __init__(self, space, eq_MHD):
        self.space = space
        self.eq_MHD = eq_MHD

        self.dim_0 = self.space.Ntot_0form
        self.dim_1 = self.space.Ntot_1form_cum[-1]
        self.dim_2 = self.space.Ntot_2form_cum[-1]
        self.dim_3 = self.space.Ntot_3form
        # self.M1     = self.space.M1
        # self.M2     = self.space.M2
        self.NbaseN = self.space.NbaseN
        self.NbaseD = self.space.NbaseD

        # self.basis_u = basis_u
        # self.basis_p = basis_p

        # Interpolation matrices
        # self.N_1 = self.proj_eta1.N
        # self.N_2 = self.proj_eta2.N
        # self.N_3 = self.proj_eta3.N

        self.N_1 = self.space.spaces[0].projectors.I
        self.N_2 = self.space.spaces[1].projectors.I
        self.N_3 = self.space.spaces[2].projectors.I

        # Histopolation matrices
        # self.D_1 = self.proj_eta1.D
        # self.D_2 = self.proj_eta2.D
        # self.D_3 = self.proj_eta3.D

        self.D_1 = self.space.spaces[0].projectors.H
        self.D_2 = self.space.spaces[1].projectors.H
        self.D_3 = self.space.spaces[2].projectors.H

        # Collocation matrices for different point sets
        self.pts0_N_1 = self.space.spaces[0].projectors.N_int
        self.pts0_N_2 = self.space.spaces[1].projectors.N_int
        self.pts0_N_3 = self.space.spaces[2].projectors.N_int

        self.pts0_D_1 = self.space.spaces[0].projectors.D_int
        self.pts0_D_2 = self.space.spaces[1].projectors.D_int
        self.pts0_D_3 = self.space.spaces[2].projectors.D_int

        self.pts1_N_1 = self.space.spaces[0].projectors.N_pts
        self.pts1_N_2 = self.space.spaces[1].projectors.N_pts
        self.pts1_N_3 = self.space.spaces[2].projectors.N_pts

        self.pts1_D_1 = self.space.spaces[0].projectors.D_pts
        self.pts1_D_2 = self.space.spaces[1].projectors.D_pts
        self.pts1_D_3 = self.space.spaces[2].projectors.D_pts

        # assert np.allclose(self.N_1.toarray(), self.pts0_N_1.toarray(), atol=1e-14)
        # assert np.allclose(self.N_2.toarray(), self.pts0_N_2.toarray(), atol=1e-14)
        # assert np.allclose(self.N_3.toarray(), self.pts0_N_3.toarray(), atol=1e-14)

        # ===== call equilibrium_mhd values at the projection points =====
        # projection points
        self.pts_PI_0 = self.space.projectors.pts_PI["0"]
        self.pts_PI_11 = self.space.projectors.pts_PI["11"]
        self.pts_PI_12 = self.space.projectors.pts_PI["12"]
        self.pts_PI_13 = self.space.projectors.pts_PI["13"]
        self.pts_PI_21 = self.space.projectors.pts_PI["21"]
        self.pts_PI_22 = self.space.projectors.pts_PI["22"]
        self.pts_PI_23 = self.space.projectors.pts_PI["23"]
        self.pts_PI_3 = self.space.projectors.pts_PI["3"]

        # p0_eq
        self.p0_eq_0 = self.space.projectors.eval_for_PI("0", self.eq_MHD.p0)
        self.p0_eq_11 = self.space.projectors.eval_for_PI("11", self.eq_MHD.p0)
        self.p0_eq_12 = self.space.projectors.eval_for_PI("12", self.eq_MHD.p0)
        self.p0_eq_13 = self.space.projectors.eval_for_PI("13", self.eq_MHD.p0)

        # p3_eq
        self.p3_eq_21 = self.space.projectors.eval_for_PI("21", self.eq_MHD.p3)
        self.p3_eq_22 = self.space.projectors.eval_for_PI("22", self.eq_MHD.p3)
        self.p3_eq_23 = self.space.projectors.eval_for_PI("23", self.eq_MHD.p3)
        self.p3_eq_3 = self.space.projectors.eval_for_PI("3", self.eq_MHD.p3)

        # n3_eq
        self.n3_eq_11 = self.space.projectors.eval_for_PI("11", self.eq_MHD.n3)
        self.n3_eq_12 = self.space.projectors.eval_for_PI("12", self.eq_MHD.n3)
        self.n3_eq_13 = self.space.projectors.eval_for_PI("13", self.eq_MHD.n3)
        self.n3_eq_21 = self.space.projectors.eval_for_PI("21", self.eq_MHD.n3)
        self.n3_eq_22 = self.space.projectors.eval_for_PI("22", self.eq_MHD.n3)
        self.n3_eq_23 = self.space.projectors.eval_for_PI("23", self.eq_MHD.n3)

        # b2_eq
        self.b2_eq_11_1 = self.space.projectors.eval_for_PI("11", self.eq_MHD.b2_1)
        self.b2_eq_12_1 = self.space.projectors.eval_for_PI("12", self.eq_MHD.b2_1)
        self.b2_eq_13_1 = self.space.projectors.eval_for_PI("13", self.eq_MHD.b2_1)
        self.b2_eq_11_2 = self.space.projectors.eval_for_PI("11", self.eq_MHD.b2_2)
        self.b2_eq_12_2 = self.space.projectors.eval_for_PI("12", self.eq_MHD.b2_2)
        self.b2_eq_13_2 = self.space.projectors.eval_for_PI("13", self.eq_MHD.b2_2)
        self.b2_eq_11_3 = self.space.projectors.eval_for_PI("11", self.eq_MHD.b2_3)
        self.b2_eq_12_3 = self.space.projectors.eval_for_PI("12", self.eq_MHD.b2_3)
        self.b2_eq_13_3 = self.space.projectors.eval_for_PI("13", self.eq_MHD.b2_3)

        # j2_eq
        self.j2_eq_11_1 = self.space.projectors.eval_for_PI("11", self.eq_MHD.j2_1)
        self.j2_eq_12_1 = self.space.projectors.eval_for_PI("12", self.eq_MHD.j2_1)
        self.j2_eq_13_1 = self.space.projectors.eval_for_PI("13", self.eq_MHD.j2_1)
        self.j2_eq_11_2 = self.space.projectors.eval_for_PI("11", self.eq_MHD.j2_2)
        self.j2_eq_12_2 = self.space.projectors.eval_for_PI("12", self.eq_MHD.j2_2)
        self.j2_eq_13_2 = self.space.projectors.eval_for_PI("13", self.eq_MHD.j2_2)
        self.j2_eq_11_3 = self.space.projectors.eval_for_PI("11", self.eq_MHD.j2_3)
        self.j2_eq_12_3 = self.space.projectors.eval_for_PI("12", self.eq_MHD.j2_3)
        self.j2_eq_13_3 = self.space.projectors.eval_for_PI("13", self.eq_MHD.j2_3)
        self.j2_eq_21_1 = self.space.projectors.eval_for_PI("21", self.eq_MHD.j2_1)
        self.j2_eq_22_1 = self.space.projectors.eval_for_PI("22", self.eq_MHD.j2_1)
        self.j2_eq_23_1 = self.space.projectors.eval_for_PI("23", self.eq_MHD.j2_1)
        self.j2_eq_21_2 = self.space.projectors.eval_for_PI("21", self.eq_MHD.j2_2)
        self.j2_eq_22_2 = self.space.projectors.eval_for_PI("22", self.eq_MHD.j2_2)
        self.j2_eq_23_2 = self.space.projectors.eval_for_PI("23", self.eq_MHD.j2_2)
        self.j2_eq_21_3 = self.space.projectors.eval_for_PI("21", self.eq_MHD.j2_3)
        self.j2_eq_22_3 = self.space.projectors.eval_for_PI("22", self.eq_MHD.j2_3)
        self.j2_eq_23_3 = self.space.projectors.eval_for_PI("23", self.eq_MHD.j2_3)

        # g_sqrt
        self.det_df_0 = self.eq_MHD.domain.jacobian_det(self.pts_PI_0[0], self.pts_PI_0[1], self.pts_PI_0[2])
        self.det_df_11 = self.eq_MHD.domain.jacobian_det(self.pts_PI_11[0], self.pts_PI_11[1], self.pts_PI_11[2])
        self.det_df_12 = self.eq_MHD.domain.jacobian_det(self.pts_PI_12[0], self.pts_PI_12[1], self.pts_PI_12[2])
        self.det_df_13 = self.eq_MHD.domain.jacobian_det(self.pts_PI_13[0], self.pts_PI_13[1], self.pts_PI_13[2])
        self.det_df_21 = self.eq_MHD.domain.jacobian_det(self.pts_PI_21[0], self.pts_PI_21[1], self.pts_PI_21[2])
        self.det_df_22 = self.eq_MHD.domain.jacobian_det(self.pts_PI_22[0], self.pts_PI_22[1], self.pts_PI_22[2])
        self.det_df_23 = self.eq_MHD.domain.jacobian_det(self.pts_PI_23[0], self.pts_PI_23[1], self.pts_PI_23[2])
        self.det_df_3 = self.eq_MHD.domain.jacobian_det(self.pts_PI_3[0], self.pts_PI_3[1], self.pts_PI_3[2])

        # G
        self.g_11 = self.eq_MHD.domain.metric(self.pts_PI_11[0], self.pts_PI_11[1], self.pts_PI_11[2])
        self.g_12 = self.eq_MHD.domain.metric(self.pts_PI_12[0], self.pts_PI_12[1], self.pts_PI_12[2])
        self.g_13 = self.eq_MHD.domain.metric(self.pts_PI_13[0], self.pts_PI_13[1], self.pts_PI_13[2])

        # self.g_21 = self.eq_MHD.domain.metric(self.pts_PI_21[0], self.pts_PI_21[1], self.pts_PI_21[2])
        # self.g_22 = self.eq_MHD.domain.metric(self.pts_PI_22[0], self.pts_PI_22[1], self.pts_PI_22[2])
        # self.g_23 = self.eq_MHD.domain.metric(self.pts_PI_23[0], self.pts_PI_23[1], self.pts_PI_23[2])

        # G_inv
        self.g_inv_11 = self.eq_MHD.domain.metric_inv(self.pts_PI_11[0], self.pts_PI_11[1], self.pts_PI_11[2])
        self.g_inv_12 = self.eq_MHD.domain.metric_inv(self.pts_PI_12[0], self.pts_PI_12[1], self.pts_PI_12[2])
        self.g_inv_13 = self.eq_MHD.domain.metric_inv(self.pts_PI_13[0], self.pts_PI_13[1], self.pts_PI_13[2])

        self.g_inv_21 = self.eq_MHD.domain.metric_inv(self.pts_PI_21[0], self.pts_PI_21[1], self.pts_PI_21[2])
        self.g_inv_22 = self.eq_MHD.domain.metric_inv(self.pts_PI_22[0], self.pts_PI_22[1], self.pts_PI_22[2])
        self.g_inv_23 = self.eq_MHD.domain.metric_inv(self.pts_PI_23[0], self.pts_PI_23[1], self.pts_PI_23[2])

        # DF^-1
        self.df_inv_0 = self.eq_MHD.domain.jacobian_inv(self.pts_PI_0[0], self.pts_PI_0[1], self.pts_PI_0[2])

        # DF
        self.df_0 = self.eq_MHD.domain.jacobian(self.pts_PI_0[0], self.pts_PI_0[1], self.pts_PI_0[2])

        # # Operator A
        # if self.basis_u == 1:
        #     self.A = spa.linalg.LinearOperator((self.dim_1, self.dim_1), matvec = lambda x : (self.M1.dot(self.W1_dot(x)) + self.transpose_W1_dot(self.M1.dot(x))) / 2 )
        #     self.A_mat = spa.csc_matrix(self.A.dot(np.identity(self.dim_1)))

        # elif self.basis_u == 2:
        #     self.A = spa.linalg.LinearOperator((self.dim_2, self.dim_2), matvec = lambda x : (self.M2.dot(self.Q2_dot(x)) + self.transpose_Q2_dot(self.M2.dot(x))) / 2 )
        #     self.A_mat = spa.csc_matrix(self.A.dot(np.identity(self.dim_2)))

        # self.A_inv = spa.linalg.inv(self.A_mat)

    ########################################
    ########## 1-form formulation ##########
    ########################################
    # ==================================================================
    def Q1_dot(self, x):
        """
        Matrix-vector product Q1.x

        Parameters
        ----------
            x : np.array
                dim R^{N^1}

        Returns
        ----------
            res : np.array
                dim R^{N^2}

        Notes
        -----
            Q1    = pi_2[rho_eq * G_inv * lambda^1] in R^{N^2 x N^1}

            Q1.x = I_2( R_2 ( F_Q1.x))

            I_2 ... inverse inter/histopolation matrix (tensor product)

            R_2  ... compute DOFs from function values at point set pts_ijk

            F_Q1[ijk, mno] = rho_eq(pts_ijk) * G_inv(pts_ijk) * lambda^1_mno(pts_ijk)

            * spline evaluation (at V_2 point sets)
                * [int, his, his] points: {greville[0], quad_pts[1], quad_pts[2]}
                * [his, int, his] points: {quad_pts[0], greville[1], quad_pts[2]}
                * [his, his, int] points: {quad_pts[0], quad_pts[1], greville[2]}

            * Components of F_Q1:
                * evaluated at [int, his, his] : (D, N, N) * G_inv_11 * rho_eq + (N, D, N) * G_inv_12 * rho_eq + (N, N, D) * G_inv_13 * rho_eq
                * evaluated at [his, int, his] : (D, N, N) * G_inv_21 * rho_eq + (N, D, N) * G_inv_22 * rho_eq + (N, N, D) * G_inv_23 * rho_eq
                * evaluated at [his, his, int] : (D, N, N) * G_inv_31 * rho_eq + (N, D, N) * G_inv_32 * rho_eq + (N, N, D) * G_inv_33 * rho_eq
        """

        # x dim check
        # x should be R{N^1}
        # assert len(x) == self.tensor_spl.Ntot_1form_cum[-1]
        x_loc = list(self.space.extract_1(x))

        # assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        # assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        # assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_11 = kron_matvec_3d([self.pts0_D_1, self.pts1_N_2, self.pts1_N_3], x_loc[0])
        mat_f_12 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts1_N_3], x_loc[1])
        mat_f_13 = kron_matvec_3d([self.pts0_N_1, self.pts1_N_2, self.pts1_D_3], x_loc[2])

        # xi2
        mat_f_21 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts1_N_3], x_loc[0])
        mat_f_22 = kron_matvec_3d([self.pts1_N_1, self.pts0_D_2, self.pts1_N_3], x_loc[1])
        mat_f_23 = kron_matvec_3d([self.pts1_N_1, self.pts0_N_2, self.pts1_D_3], x_loc[2])

        # xi3
        mat_f_31 = kron_matvec_3d([self.pts1_D_1, self.pts1_N_2, self.pts0_N_3], x_loc[0])
        mat_f_32 = kron_matvec_3d([self.pts1_N_1, self.pts1_D_2, self.pts0_N_3], x_loc[1])
        mat_f_33 = kron_matvec_3d([self.pts1_N_1, self.pts1_N_2, self.pts0_D_3], x_loc[2])

        mat_f_11_c = mat_f_11 * self.n3_eq_21 * self.g_inv_21[0, 0]
        mat_f_12_c = mat_f_12 * self.n3_eq_21 * self.g_inv_21[0, 1]
        mat_f_13_c = mat_f_13 * self.n3_eq_21 * self.g_inv_21[0, 2]
        mat_f_21_c = mat_f_21 * self.n3_eq_22 * self.g_inv_22[1, 0]
        mat_f_22_c = mat_f_22 * self.n3_eq_22 * self.g_inv_22[1, 1]
        mat_f_23_c = mat_f_23 * self.n3_eq_22 * self.g_inv_22[1, 2]
        mat_f_31_c = mat_f_31 * self.n3_eq_23 * self.g_inv_23[2, 0]
        mat_f_32_c = mat_f_32 * self.n3_eq_23 * self.g_inv_23[2, 1]
        mat_f_33_c = mat_f_33 * self.n3_eq_23 * self.g_inv_23[2, 2]

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.space.projectors.dofs("21", mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.space.projectors.dofs("22", mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.space.projectors.dofs("23", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : inter(xi1)-histo(xi2)-histo(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("21", DOF_1)

        # xi2 : histo(xi1)-inter(xi2)-histo(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("22", DOF_2)

        # xi3 : histo(xi1)-histo(xi2)-inter(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("23", DOF_3)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ==================================================================
    def transpose_Q1_dot(self, x):
        """
        Matrix-vector product Q1.T.x

        Parameters
        ----------
            x : np.array
                dim R^{N^2}

        Returns
        ----------
            res : np.array
                dim R^{N^1}

        Notes
        -----
            Q1.x = I_2( R_2 ( F_Q1.x))

            Q1.T.x = F_Q1.T( R_2.T ( I_2.T.x))

            See Q1_dot for more info.
        """

        # x dim check
        # x should be R{N^2}
        # assert len(x) == self.space.Ntot_2form_cum[-1]
        x_loc = list(self.space.extract_2(x))

        # assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        # assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        # assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])

        # step1 : I.T(x)
        # xi1 : transpose of inter(xi1)-histo(xi2)-histo(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.N_1.T, self.D_2.T, self.D_3.T], x_loc[0])

        # xi2 : transpose of histo(xi1)-inter(xi2)-histo(xi3)-polati on.
        mat_dofs_2 = kron_solve_3d([self.D_1.T, self.N_2.T, self.D_3.T], x_loc[1])

        # xi3 : transpose of histo(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.D_1.T, self.D_2.T, self.N_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.space.projectors.dofs_T("21", mat_dofs_1)

        # xi2
        mat_f_2 = self.space.projectors.dofs_T("22", mat_dofs_2)

        # xi3
        mat_f_3 = self.space.projectors.dofs_T("23", mat_dofs_3)

        # step3 : F.T( R.T( I.T(x) ) )
        mat_f_11_c = mat_f_1 * self.n3_eq_21 * self.g_inv_21[0, 0]
        mat_f_12_c = mat_f_1 * self.n3_eq_21 * self.g_inv_21[0, 1]
        mat_f_13_c = mat_f_1 * self.n3_eq_21 * self.g_inv_21[0, 2]
        mat_f_21_c = mat_f_2 * self.n3_eq_22 * self.g_inv_22[1, 0]
        mat_f_22_c = mat_f_2 * self.n3_eq_22 * self.g_inv_22[1, 1]
        mat_f_23_c = mat_f_2 * self.n3_eq_22 * self.g_inv_22[1, 2]
        mat_f_31_c = mat_f_3 * self.n3_eq_23 * self.g_inv_23[2, 0]
        mat_f_32_c = mat_f_3 * self.n3_eq_23 * self.g_inv_23[2, 1]
        mat_f_33_c = mat_f_3 * self.n3_eq_23 * self.g_inv_23[2, 2]

        res_11 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts1_N_3.T], mat_f_11_c)
        res_12 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts1_N_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_N_2.T, self.pts1_D_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts1_N_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_22_c)
        res_23 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts1_D_1.T, self.pts1_N_2.T, self.pts0_N_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts1_N_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts1_N_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_33_c)

        res_1 = res_11 + res_21 + res_31
        res_2 = res_12 + res_22 + res_32
        res_3 = res_13 + res_23 + res_33

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ===================================================================
    def W1_dot(self, x):
        """
        Matrix-vector product W1.x

        Parameters
        ----------
            x : np.array
                dim R^{N^1}

        Returns
        ----------
            res : np.array
                dim R^{N^1}

        Notes
        -----
            W1 = pi_1[rho_eq / g_sqrt * lambda^1]  in   R{N^1 x N^1}

            W1.x = I_1( R_1 ( F_W1.x))

            I_1 ... inverse inter/histopolation matrix (tensor product)

            R_1  ... compute DOFs from function values at point set pts_ijk

            F_W1[ijk,mno] = rho_eq(pts_ijk) / g_sqrt(pts_ijk) * lambda^1_mno(pts_ijk)

            * spline evaluation (at V_1 point sets)
                * [his, int, int] points: {quad_pts[0], greville[1], greville[2]}
                * [int, his, int] points: {greville[0], quad_pts[1], greville[2]}
                * [int, int, his] points: {greville[0], greville[1], quad_pts[2]}

            * Components of F_W1:
                * evaluated at [his, int, int] : (D, N, N) * rho_eq / g_sqrt
                * evaluated at [int, his, int] : (N, D, N) * rho_eq / g_sqrt
                * evaluated at [int, int, his] : (N, N, D) * rho_eq / g_sqrt
        """
        # x dim check
        # x should be R{N^1}
        # assert len(x) == self.space.Ntot_1form_cum[-1]
        x_loc = list(self.space.extract_1(x))

        # assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        # assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        # assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        mat_f_1 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts0_N_3], x_loc[0])
        mat_f_2 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts0_N_3], x_loc[1])
        mat_f_3 = kron_matvec_3d([self.pts0_N_1, self.pts0_N_2, self.pts1_D_3], x_loc[2])

        mat_f_1_c = mat_f_1 * self.n3_eq_11 / self.det_df_11
        mat_f_2_c = mat_f_2 * self.n3_eq_12 / self.det_df_12
        mat_f_3_c = mat_f_3 * self.n3_eq_13 / self.det_df_13

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.space.projectors.dofs("11", mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.space.projectors.dofs("12", mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.space.projectors.dofs("13", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : histo(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("11", DOF_1)

        # xi2 : inter(xi1)-histo(xi2)-inter(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("12", DOF_2)

        # xi3 : inter(xi1)-inter(xi2)-histo(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("13", DOF_3)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ===================================================================
    def transpose_W1_dot(self, x):
        """
        Matrix-vector product W1.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^1}

        Returns
        ----------
            res : np.array
                dim R{N^1}

        Notes
        -----
            W1.x = I_1( R_1 ( F_W1.x))

            W1.T.x = F_W1.T( R_1.T ( I_1.T.x))

            See W1_dot for more details.
        """
        # x dim check
        # x should be R{N^1}
        # assert len(x) == self.space.Ntot_1form_cum[-1]
        x_loc = list(self.space.extract_1(x))

        # assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        # assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        # assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # step1 : I.T(x)
        # xi1 : transpose of histo(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.D_1.T, self.N_2.T, self.N_3.T], x_loc[0])

        # xi2 : transpose of inter(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.D_2.T, self.N_3.T], x_loc[1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-histo(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.D_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1 : mat_f_1_{i,m,j,k} = w_{i,m} * DOF_1_{i,j,k}
        mat_f_1 = self.space.projectors.dofs_T("11", mat_dofs_1)

        # xi2 : mat_f_2_{i,j,m,k} = w_{j,m} * DOF_2_{i,j,k}
        mat_f_2 = self.space.projectors.dofs_T("12", mat_dofs_2)

        # xi3 : mat_f_2_{i,j,k,m} = w_{k,m} * DOF_3_{i,j,k}
        mat_f_3 = self.space.projectors.dofs_T("13", mat_dofs_3)

        mat_f_1_c = mat_f_1 * self.n3_eq_11 / self.det_df_11
        mat_f_2_c = mat_f_2 * self.n3_eq_12 / self.det_df_12
        mat_f_3_c = mat_f_3 * self.n3_eq_13 / self.det_df_13

        res_1 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_1_c)
        res_2 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_2_c)
        res_3 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_3_c)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def U1_dot(self, x):
        """
        Matrix-vector product U1.x

        Parameters
        ----------
            x : np.array
                dim R^{N^1}

        Returns
        ----------
            res : np.array
                dim R^{N^2}

        Notes
        -----
            U1     = pi_2[g_sqrt * G_inv * lambda^1]   in  R^{N^2 x N^1}

            U1.x = I_2( R_2 ( F_U1.x))

            I_2 ... inverse inter/histopolation matrix (tensor product)

            R_2  ... compute DOFs from function values at point set pts_ijk

            F_U1[ijk, mno] = g_sqrt(pts_ijk) * G_inv(pts_ijk) * lambda^1_mno(pts_ijk)

            * spline evaluation (at V_2 point sets)
                * [int, his, his] points: {greville[0], quad_pts[1], quad_pts[2]}
                * [his, int, his] points: {quad_pts[0], greville[1], quad_pts[2]}
                * [his, his, int] points: {quad_pts[0], quad_pts[1], greville[2]}

            * Components of F_U1:
                * evaluated at [int, his, his] : (D, N, N) * G_inv_11 * g_sqrt + (N, D, N) * G_inv_12 * g_sqrt + (N, N, D) * G_inv_13 * g_sqrt
                * evaluated at [his, int, his] : (D, N, N) * G_inv_21 * g_sqrt + (N, D, N) * G_inv_22 * g_sqrt + (N, N, D) * G_inv_23 * g_sqrt
                * evaluated at [his, his, int] : (D, N, N) * G_inv_31 * g_sqrt + (N, D, N) * G_inv_32 * g_sqrt + (N, N, D) * G_inv_33 * g_sqrt
        """
        # x dim check
        # x should be R{N^1}
        # assert len(x) == self.space.Ntot_1form_cum[-1]
        x_loc = list(self.space.extract_1(x))

        # assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        # assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        # assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_11 = kron_matvec_3d([self.pts0_D_1, self.pts1_N_2, self.pts1_N_3], x_loc[0])
        mat_f_12 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts1_N_3], x_loc[1])
        mat_f_13 = kron_matvec_3d([self.pts0_N_1, self.pts1_N_2, self.pts1_D_3], x_loc[2])

        # xi2
        mat_f_21 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts1_N_3], x_loc[0])
        mat_f_22 = kron_matvec_3d([self.pts1_N_1, self.pts0_D_2, self.pts1_N_3], x_loc[1])
        mat_f_23 = kron_matvec_3d([self.pts1_N_1, self.pts0_N_2, self.pts1_D_3], x_loc[2])

        # xi3
        mat_f_31 = kron_matvec_3d([self.pts1_D_1, self.pts1_N_2, self.pts0_N_3], x_loc[0])
        mat_f_32 = kron_matvec_3d([self.pts1_N_1, self.pts1_D_2, self.pts0_N_3], x_loc[1])
        mat_f_33 = kron_matvec_3d([self.pts1_N_1, self.pts1_N_2, self.pts0_D_3], x_loc[2])

        mat_f_11_c = mat_f_11 * self.det_df_21 * self.g_inv_21[0, 0]
        mat_f_12_c = mat_f_12 * self.det_df_21 * self.g_inv_21[0, 1]
        mat_f_13_c = mat_f_13 * self.det_df_21 * self.g_inv_21[0, 2]
        mat_f_21_c = mat_f_21 * self.det_df_22 * self.g_inv_22[1, 0]
        mat_f_22_c = mat_f_22 * self.det_df_22 * self.g_inv_22[1, 1]
        mat_f_23_c = mat_f_23 * self.det_df_22 * self.g_inv_22[1, 2]
        mat_f_31_c = mat_f_31 * self.det_df_23 * self.g_inv_23[2, 0]
        mat_f_32_c = mat_f_32 * self.det_df_23 * self.g_inv_23[2, 1]
        mat_f_33_c = mat_f_33 * self.det_df_23 * self.g_inv_23[2, 2]

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.space.projectors.dofs("21", mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.space.projectors.dofs("22", mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.space.projectors.dofs("23", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : inter(xi1)-histo(xi2)-histo(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("21", DOF_1)

        # xi2 : histo(xi1)-inter(xi2)-histo(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("22", DOF_2)

        # xi3 : histo(xi1)-histo(xi2)-inter(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("23", DOF_3)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def transpose_U1_dot(self, x):
        """
        Matrix-vector product U1.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^2}

        Returns
        ----------
            res : np.array
                dim R{N^1}

        Notes
        -----
            U1.x = I_2( R_2 ( F_U1.x))

            U1.T.x = F_U1.T( R_2.T ( I_2.T.x))

            See U1_dot for more details.
        """

        # x dim check
        # x should be R{N^2}
        # assert len(x) == self.space.Ntot_2form_cum[-1]
        x_loc = list(self.space.extract_2(x))

        # assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        # assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        # assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])

        # step1 : I.T(x)
        # xi1 : transpose of inter(xi1)-histo(xi2)-histo(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.N_1.T, self.D_2.T, self.D_3.T], x_loc[0])

        # xi2 : transpose of histo(xi1)-inter(xi2)-histo(xi3)-polati on.
        mat_dofs_2 = kron_solve_3d([self.D_1.T, self.N_2.T, self.D_3.T], x_loc[1])

        # xi3 : transpose of histo(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.D_1.T, self.D_2.T, self.N_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.space.projectors.dofs_T("21", mat_dofs_1)

        # xi2
        mat_f_2 = self.space.projectors.dofs_T("22", mat_dofs_2)

        # xi3
        mat_f_3 = self.space.projectors.dofs_T("23", mat_dofs_3)

        # step3 : F.T( R.T( I.T(x) ) )
        mat_f_11_c = mat_f_1 * self.det_df_21 * self.g_inv_21[0, 0]
        mat_f_12_c = mat_f_1 * self.det_df_21 * self.g_inv_21[0, 1]
        mat_f_13_c = mat_f_1 * self.det_df_21 * self.g_inv_21[0, 2]
        mat_f_21_c = mat_f_2 * self.det_df_22 * self.g_inv_22[1, 0]
        mat_f_22_c = mat_f_2 * self.det_df_22 * self.g_inv_22[1, 1]
        mat_f_23_c = mat_f_2 * self.det_df_22 * self.g_inv_22[1, 2]
        mat_f_31_c = mat_f_3 * self.det_df_23 * self.g_inv_23[2, 0]
        mat_f_32_c = mat_f_3 * self.det_df_23 * self.g_inv_23[2, 1]
        mat_f_33_c = mat_f_3 * self.det_df_23 * self.g_inv_23[2, 2]

        res_11 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts1_N_3.T], mat_f_11_c)
        res_12 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts1_N_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_N_2.T, self.pts1_D_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts1_N_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_22_c)
        res_23 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts1_D_1.T, self.pts1_N_2.T, self.pts0_N_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts1_N_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts1_N_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_33_c)

        res_1 = res_11 + res_21 + res_31
        res_2 = res_12 + res_22 + res_32
        res_3 = res_13 + res_23 + res_33

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def P1_dot(self, x):
        """
        Matrix-vector product P1.x

        Parameters
        ----------
            x : np.array
                dim R^{N^2}

        Returns
        ----------
            res : np.array
                dim R^{N^1}

        Notes
        -----
            P1     = pi_1[jmat_eq / g_sqrt * lambda^2]  in   R{N^1 x N^2}

            jmat_eq  = [[0  , -j_eq_z,  j_eq_y], [j_eq_z,     0  , -j_eq_x], [-j_eq_y,  j_eq_x,     0]] in R^{3 x 3}

            P1.x = I_1( R_1 ( F_P1.x))

            I_1 ... inverse inter/histopolation matrix (tensor product)

            R_1  ... compute DOFs from function values at point set pts_ijk

            F_P1[ijk, mno] = jmat_eq(pts_ijk) / g_sqrt(pts_ijk) * lambda^2_mno(pts_ijk)

            * spline evaluation (at V_1 point sets)
                * [his, int, int] points: {quad_pts[0], greville[1], greville[2]}
                * [int, his, int] points: {greville[0], quad_pts[1], greville[2]}
                * [int, int, his] points: {greville[0], greville[1], quad_pts[2]}

            * Components of F_P1:
                * evaluated at [his, int, int] : (D, N, D) * (-j_eq_z) / g_sqrt + (D, D, N) *  j_eq_y / g_sqrt
                * evaluated at [int, his, int] : (N, D, D) *  j_eq_z / g_sqrt   + (D, D, N) * (-j_eq_x) / g_sqrt
                * evaluated at [int, int, his] : (N, D, D) * (-j_eq_y) / g_sqrt + (D, N, D) *  j_eq_x / g_sqrt,
        """

        # x dim check
        # x should be R{N^2}
        # assert len(x) == self.space.Ntot_2form_cum[-1]
        x_loc = list(self.space.extract_2(x))

        # assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        # assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        # assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_11 = kron_matvec_3d([self.pts1_N_1, self.pts0_D_2, self.pts0_D_3], x_loc[0])  # 0
        mat_f_12 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts0_D_3], x_loc[1])
        mat_f_13 = kron_matvec_3d([self.pts1_D_1, self.pts0_D_2, self.pts0_N_3], x_loc[2])

        # xi2
        mat_f_21 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts0_D_3], x_loc[0])
        mat_f_22 = kron_matvec_3d([self.pts0_D_1, self.pts1_N_2, self.pts0_D_3], x_loc[1])  # 0
        mat_f_23 = kron_matvec_3d([self.pts0_D_1, self.pts1_D_2, self.pts0_N_3], x_loc[2])

        # xi3
        mat_f_31 = kron_matvec_3d([self.pts0_N_1, self.pts0_D_2, self.pts1_D_3], x_loc[0])
        mat_f_32 = kron_matvec_3d([self.pts0_D_1, self.pts0_N_2, self.pts1_D_3], x_loc[1])
        mat_f_33 = kron_matvec_3d([self.pts0_D_1, self.pts0_D_2, self.pts1_N_3], x_loc[2])  # 0

        mat_f_11_c = mat_f_11 * 0 / self.det_df_11
        mat_f_12_c = mat_f_12 * (-self.j2_eq_11_3) / self.det_df_11
        mat_f_13_c = mat_f_13 * (self.j2_eq_11_2) / self.det_df_11
        mat_f_21_c = mat_f_21 * (self.j2_eq_12_3) / self.det_df_12
        mat_f_22_c = mat_f_22 * 0 / self.det_df_12
        mat_f_23_c = mat_f_23 * (-self.j2_eq_12_1) / self.det_df_12
        mat_f_31_c = mat_f_31 * (-self.j2_eq_13_2) / self.det_df_13
        mat_f_32_c = mat_f_32 * (self.j2_eq_13_1) / self.det_df_13
        mat_f_33_c = mat_f_33 * 0 / self.det_df_13

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.space.projectors.dofs("11", mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.space.projectors.dofs("12", mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.space.projectors.dofs("13", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : histo(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("11", DOF_1)

        # xi2 : inter(xi1)-histo(xi2)-inter(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("12", DOF_2)

        # xi3 : inter(xi1)-inter(xi2)-histo(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("13", DOF_3)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def transpose_P1_dot(self, x):
        """
        Matrix-vector product P1.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^1}

        Returns
        ----------
            res : np.array
                dim R{N^2}

        Notes
        -----
            P1.x = I_1( R_1 ( F_P1.x))

            P1.T.x = F_P1.T( R_1.T ( I_1.T.x))

            See P1_dot for more details.
        """

        # x dim check
        # x should be R{N^1}
        # assert len(x) == self.space.Ntot_1form_cum[-1]
        x_loc = list(self.space.extract_1(x))

        # assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        # assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        # assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # step1 : I.T(x)
        # xi1 : transpose of histo(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.D_1.T, self.N_2.T, self.N_3.T], x_loc[0])

        # xi2 : transpose of inter(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.D_2.T, self.N_3.T], x_loc[1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-histo(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.D_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.space.projectors.dofs_T("11", mat_dofs_1)

        # xi2
        mat_f_2 = self.space.projectors.dofs_T("12", mat_dofs_2)

        # xi3
        mat_f_3 = self.space.projectors.dofs_T("13", mat_dofs_3)

        # step3 : F.T( R.T( I.T(x) ) )
        mat_f_11_c = mat_f_1 * 0 / self.det_df_11
        mat_f_12_c = mat_f_1 * (-self.j2_eq_11_3) / self.det_df_11
        mat_f_13_c = mat_f_1 * (self.j2_eq_11_2) / self.det_df_11
        mat_f_21_c = mat_f_2 * (self.j2_eq_12_3) / self.det_df_12
        mat_f_22_c = mat_f_2 * 0 / self.det_df_12
        mat_f_23_c = mat_f_2 * (-self.j2_eq_12_1) / self.det_df_12
        mat_f_31_c = mat_f_3 * (-self.j2_eq_13_2) / self.det_df_13
        mat_f_32_c = mat_f_3 * (self.j2_eq_13_1) / self.det_df_13
        mat_f_33_c = mat_f_3 * 0 / self.det_df_13

        res_11 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts0_D_3.T], mat_f_11_c)  # 0
        res_12 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts0_D_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_D_2.T, self.pts0_N_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts0_D_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_22_c)  # 0
        res_23 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_D_2.T, self.pts1_D_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_33_c)  # 0

        res_1 = res_11 + res_21 + res_31
        res_2 = res_12 + res_22 + res_32
        res_3 = res_13 + res_23 + res_33

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def S1_dot(self, x):
        """
        Matrix-vector product S1.x

        Parameters
        ----------
            x : np.array
                dim R^{N^1}

        Returns
        ----------
            res : np.array
                dim R^{N^2}

        Notes
        -----
            S1     = pi_2[p_eq * G_inv * lambda^1]  in  R{N^2 x N^1}

            S1.x = I_2( R_2 ( F_S1.x))

            I_2 ... inverse inter/histopolation matrix (tensor product)

            R_2  ... compute DOFs from function values at point set pts_ijk

            F_S1[ijk, mno] = p_eq(pts_ijk) * G_inv(pts_ijk) * lambda^1_mno(pts_ijk)

            * spline evaluation (at V_2 point sets)
                * [int, his, his] points: {greville[0], quad_pts[1], quad_pts[2]}
                * [his, int, his] points: {quad_pts[0], greville[1], quad_pts[2]}
                * [his, his, int] points: {quad_pts[0], quad_pts[1], greville[2]}

            * Components of F_S1:
                * evaluated at [int, his, his] : (D, N, N) * G_inv_11 * p_eq + (N, D, N) * G_inv_12 * p_eq + (N, N, D) * G_inv_13 * p_eq
                * evaluated at [his, int, his] : (D, N, N) * G_inv_21 * p_eq + (N, D, N) * G_inv_22 * p_eq + (N, N, D) * G_inv_23 * p_eq
                * evaluated at [his, his, int] : (D, N, N) * G_inv_31 * p_eq + (N, D, N) * G_inv_32 * p_eq + (N, N, D) * G_inv_33 * p_eq
        """

        # x dim check
        # x should be R{N^1}
        # assert len(x) == self.space.Ntot_1form_cum[-1]
        x_loc = list(self.space.extract_1(x))

        # assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        # assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        # assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_11 = kron_matvec_3d([self.pts0_D_1, self.pts1_N_2, self.pts1_N_3], x_loc[0])
        mat_f_12 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts1_N_3], x_loc[1])
        mat_f_13 = kron_matvec_3d([self.pts0_N_1, self.pts1_N_2, self.pts1_D_3], x_loc[2])

        # xi2
        mat_f_21 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts1_N_3], x_loc[0])
        mat_f_22 = kron_matvec_3d([self.pts1_N_1, self.pts0_D_2, self.pts1_N_3], x_loc[1])
        mat_f_23 = kron_matvec_3d([self.pts1_N_1, self.pts0_N_2, self.pts1_D_3], x_loc[2])

        # xi3
        mat_f_31 = kron_matvec_3d([self.pts1_D_1, self.pts1_N_2, self.pts0_N_3], x_loc[0])
        mat_f_32 = kron_matvec_3d([self.pts1_N_1, self.pts1_D_2, self.pts0_N_3], x_loc[1])
        mat_f_33 = kron_matvec_3d([self.pts1_N_1, self.pts1_N_2, self.pts0_D_3], x_loc[2])

        mat_f_11_c = mat_f_11 * self.p3_eq_21 * self.g_inv_21[0, 0]
        mat_f_12_c = mat_f_12 * self.p3_eq_21 * self.g_inv_21[0, 1]
        mat_f_13_c = mat_f_13 * self.p3_eq_21 * self.g_inv_21[0, 2]
        mat_f_21_c = mat_f_21 * self.p3_eq_22 * self.g_inv_22[1, 0]
        mat_f_22_c = mat_f_22 * self.p3_eq_22 * self.g_inv_22[1, 1]
        mat_f_23_c = mat_f_23 * self.p3_eq_22 * self.g_inv_22[1, 2]
        mat_f_31_c = mat_f_31 * self.p3_eq_23 * self.g_inv_23[2, 0]
        mat_f_32_c = mat_f_32 * self.p3_eq_23 * self.g_inv_23[2, 1]
        mat_f_33_c = mat_f_33 * self.p3_eq_23 * self.g_inv_23[2, 2]

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.space.projectors.dofs("21", mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.space.projectors.dofs("22", mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.space.projectors.dofs("23", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : inter(xi1)-histo(xi2)-histo(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("21", DOF_1)

        # xi2 : histo(xi1)-inter(xi2)-histo(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("22", DOF_2)

        # xi3 : histo(xi1)-histo(xi2)-inter(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("23", DOF_3)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def transpose_S1_dot(self, x):
        """
        Matrix-vector product S1.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^2}

        Returns
        ----------
            res : np.array
                dim R{N^1}

        Notes
        -----
            S1.x = I_2( R_2 ( F_S1.x))

            S1.T.x = F_S1.T( R_2.T ( I_2.T.x))

            See S1_dot for more details.
        """

        # x dim check
        # x should be R{N^2}
        # assert len(x) == self.space.Ntot_2form_cum[-1]
        x_loc = list(self.space.extract_2(x))

        # assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        # assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        # assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])

        # step1 : I.T(x)
        # xi1 : transpose of inter(xi1)-histo(xi2)-histo(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.N_1.T, self.D_2.T, self.D_3.T], x_loc[0])

        # xi2 : transpose of histo(xi1)-inter(xi2)-histo(xi3)-polati on.
        mat_dofs_2 = kron_solve_3d([self.D_1.T, self.N_2.T, self.D_3.T], x_loc[1])

        # xi3 : transpose of histo(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.D_1.T, self.D_2.T, self.N_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.space.projectors.dofs_T("21", mat_dofs_1)

        # xi2
        mat_f_2 = self.space.projectors.dofs_T("22", mat_dofs_2)

        # xi3
        mat_f_3 = self.space.projectors.dofs_T("23", mat_dofs_3)

        # step3 : F.T( R.T( I.T(x) ) )
        mat_f_11_c = mat_f_1 * self.p3_eq_21 * self.g_inv_21[0, 0]
        mat_f_12_c = mat_f_1 * self.p3_eq_21 * self.g_inv_21[0, 1]
        mat_f_13_c = mat_f_1 * self.p3_eq_21 * self.g_inv_21[0, 2]
        mat_f_21_c = mat_f_2 * self.p3_eq_22 * self.g_inv_22[1, 0]
        mat_f_22_c = mat_f_2 * self.p3_eq_22 * self.g_inv_22[1, 1]
        mat_f_23_c = mat_f_2 * self.p3_eq_22 * self.g_inv_22[1, 2]
        mat_f_31_c = mat_f_3 * self.p3_eq_23 * self.g_inv_23[2, 0]
        mat_f_32_c = mat_f_3 * self.p3_eq_23 * self.g_inv_23[2, 1]
        mat_f_33_c = mat_f_3 * self.p3_eq_23 * self.g_inv_23[2, 2]

        res_11 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts1_N_3.T], mat_f_11_c)
        res_21 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts1_N_3.T], mat_f_12_c)
        res_31 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_N_2.T, self.pts1_D_3.T], mat_f_13_c)

        res_12 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts1_N_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_22_c)
        res_32 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_23_c)

        res_13 = kron_matvec_3d([self.pts1_D_1.T, self.pts1_N_2.T, self.pts0_N_3.T], mat_f_31_c)
        res_23 = kron_matvec_3d([self.pts1_N_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts1_N_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_33_c)

        res_1 = res_11 + res_21 + res_31
        res_2 = res_12 + res_22 + res_32
        res_3 = res_13 + res_23 + res_33

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ===================================================================
    def S10_dot(self, x):
        """
        Matrix-vector product S10.x

        Parameters
        ----------
            x : np.array
                dim R^{N^1}

        Returns
        ----------
            res : np.array
                dim R^{N^1}

        Notes
        -----
            S10     = pi_1[p_eq * lambda^1]  in   R{N^1 x N^1}

            S10.x = I_1( R_1 ( F_S10.x))

            I_1 ... inverse inter/histopolation matrix (tensor product)

            R_1  ... compute DOFs from function values at point set pts_ijk

            F_S10[ijk, mno] = p_eq(pts_ijk) * lambda^1_mno(pts_ijk)

            * spline evaluation (at V_1 point sets)
                * [his, int, int] points: {quad_pts[0], greville[1], greville[2]}
                * [int, his, int] points: {greville[0], quad_pts[1], greville[2]}
                * [int, int, his] points: {greville[0], greville[1], quad_pts[2]}

            * Components of F_W1:
                * evaluated at [his, int, int] : (D, N, N) * p_eq + (N, D, N) * p_eq + (N, N, D) * p_eq
                * evaluated at [int, his, int] : (D, N, N) * p_eq + (N, D, N) * p_eq + (N, N, D) * p_eq
                * evaluated at [int, int, his] : (D, N, N) * p_eq + (N, D, N) * p_eq + (N, N, D) * p_eq
        """

        # x dim check
        # x should be R{N^1}
        # assert len(x) == self.space.Ntot_1form_cum[-1]
        x_loc = list(self.space.extract_1(x))

        # assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        # assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        # assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        mat_f_1 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts0_N_3], x_loc[0])
        mat_f_2 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts0_N_3], x_loc[1])
        mat_f_3 = kron_matvec_3d([self.pts0_N_1, self.pts0_N_2, self.pts1_D_3], x_loc[2])

        mat_f_1_c = mat_f_1 * self.p0_eq_11
        mat_f_2_c = mat_f_2 * self.p0_eq_12
        mat_f_3_c = mat_f_3 * self.p0_eq_13

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.space.projectors.dofs("11", mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.space.projectors.dofs("12", mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.space.projectors.dofs("13", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : histo(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("11", DOF_1)

        # xi2 : inter(xi1)-histo(xi2)-inter(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("12", DOF_2)

        # xi3 : inter(xi1)-inter(xi2)-histo(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("13", DOF_3)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ===================================================================
    def transpose_S10_dot(self, x):
        """
        Matrix-vector product S10.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^1}

        Returns
        ----------
            res : np.array
                dim R{N^1}

        Notes
        -----
            S10.x = I_1( R_1 ( F_S10.x))

            S10.T.x = F_S10.T( R_1.T ( I_1.T.x))

            See S10_dot for more details.
        """
        # x dim check
        # x should be R{N^1}
        # assert len(x) == self.space.Ntot_1form_cum[-1]
        x_loc = list(self.space.extract_1(x))

        # assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        # assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        # assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # step1 : I.T(x)
        # xi1 : transpose of histo(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.D_1.T, self.N_2.T, self.N_3.T], x_loc[0])

        # xi2 : transpose of inter(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.D_2.T, self.N_3.T], x_loc[1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-histo(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.D_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1 : mat_f_1_{i,m,j,k} = w_{i,m} * DOF_1_{i,j,k}
        mat_f_1 = self.space.projectors.dofs_T("11", mat_dofs_1)

        # xi2 : mat_f_2_{i,j,m,k} = w_{j,m} * DOF_2_{i,j,k}
        mat_f_2 = self.space.projectors.dofs_T("12", mat_dofs_2)

        # xi3 : mat_f_2_{i,j,k,m} = w_{k,m} * DOF_3_{i,j,k}
        mat_f_3 = self.space.projectors.dofs_T("13", mat_dofs_3)

        mat_f_1_c = mat_f_1 * self.p0_eq_11
        mat_f_2_c = mat_f_2 * self.p0_eq_12
        mat_f_3_c = mat_f_3 * self.p0_eq_13

        res_1 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_1_c)
        res_2 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_2_c)
        res_3 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_3_c)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # =================================================================
    def K1_dot(self, x):
        """
        Matrix-vector product K1.x

        Parameters
        ----------
            x : np.array
                dim R^{N^3}

        Returns
        ----------
            res : np.array
                dim R^{N^3}

        Notes
        -----
            K1 = pi_3[p_eq / g_sqrt * lambda^3]   in  R{N^3 x N^3}

            K1.x = I_3( R_3 ( F_K1.x))

            I_3 ... inverse histopolation matrix (tensor product)

            R_3  ... compute DOFs from function values at point set pts_ijk

            F_K1[ijk,mno] = p_eq(pts_ijk) / g_sqrt(pts_ijk) * lambda^3_mno(pts_ijk)

            * spline evaluation (at V_3 point sets)
                * [his, his, his] points: {quad_pts[0], quad_pts[1], quad_pts[2]}

            * Components of F_K1:
                * evaluated at [his, his, his] : (D, D, D) * p_eq / sqrt g
        """

        # x dim check
        # assert len(x) == self.space.Ntot_3form
        x_loc = self.space.extract_3(x)

        # assert x_loc.shape == (self.NbaseD[0], self.NbaseD[1], self.NbaseD[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.

        mat_f = kron_matvec_3d([self.pts1_D_1, self.pts1_D_2, self.pts1_D_3], x_loc)

        mat_f_c = mat_f * self.p3_eq_3 / self.det_df_3

        # ========== Step 2 : R( F(x) ) ==========#
        # Linear operator : evaluation values at the projection points to the Degree of Freedom of the spline.
        DOF = self.space.projectors.dofs("3", mat_f_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # histo(xi1)-histo(xi2)-histo(xi3)-polation.
        res = self.space.projectors.PI_mat("3", DOF)

        return res.flatten()

    # =================================================================
    def transpose_K1_dot(self, x):
        """
        Matrix-vector product K1.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^3}

        Returns
        ----------
            res : 3d array
                dim R{N^3}

        Notes
        -----
            K1.x = I_3( R_3 ( F_K1.x))

            K1.T.x = F_K1.T( R_3.T ( I_3.T.x))

            See K1_dot for more details.
        """

        # x dim check
        # assert len(x) == self.space.Ntot_3form
        x_loc = self.space.extract_3(x)

        # assert x_loc.shape == (self.NbaseD[0], self.NbaseD[1], self.NbaseD[2])

        # step1 : I.T(x)
        mat_dofs = kron_solve_3d([self.D_1.T, self.D_2.T, self.D_3.T], x_loc)

        # step2 : R.T( I.T(x) )
        mat_f = self.space.projectors.dofs_T("3", mat_dofs)

        # step3 : F.T( R.T( I.T(x) ) )
        mat_f_c = self.p3_eq_3 * mat_f / self.det_df_3

        res = kron_matvec_3d([self.pts1_D_1.T, self.pts1_D_2.T, self.pts1_D_3.T], mat_f_c)

        return res.flatten()

    # =================================================================
    def K10_dot(self, x):
        """
        Matrix-vector product K10.x

        Parameters
        ----------
            x : np.array
                dim R^{N^0}

        Returns
        ----------
            res : np.array
                dim R^{N^0}

        Notes
        -----
            K10 = pi_0[p_eq * lambda^0]  in   R^{N^0 x N^0}

            K10.x = I_0( R_0 ( F_K10.x))

            I_0 ... inverse interpolation matrix (tensor product)

            R_0  ... compute DOFs from function values at point set pts_ijk

            F_K10[ijk,mno] = p_eq(pts_ijk) * lambda^0_mno(pts_ijk)

            * spline evaluation (at V_0 point sets)
                * [int, int, int] points: {greville[0], greville[1], greville[2]}

            * Components of F_K10:
                * evaluated at [int, int, int] : (N, N, N) * p_eq
        """

        # x dim check
        # assert len(x) == self.space.Ntot_0form
        x_loc = self.space.extract_0(x)

        # assert x_loc.shape == (self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        mat_f = kron_matvec_3d([self.pts0_N_1, self.pts0_N_2, self.pts0_N_3], x_loc)

        mat_f_c = mat_f * self.p0_eq_0

        # ========== Step 2 : R( F(x) ) ==========#
        # Linear operator : evaluation values at the projection points to the Degree of Freedom of the spline.
        DOF = self.space.projectors.dofs("0", mat_f_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # histo(xi1)-histo(xi2)-histo(xi3)-polation.
        res = self.space.projectors.PI_mat("0", DOF)

        return res.flatten()

    # =================================================================
    def transpose_K10_dot(self, x):
        """
        Matrix vecotr product K10.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^0}

        Returns
        ----------
            res : 3d array
                dim R{N^0}

        Notes
        -----
            K10.x = I_0( R_0 ( F_K10.x))

            K10.T.x = F_K10.T( R_0.T ( I_0.T.x))

            See K10_dot for more details.
        """

        # x dim check
        # assert len(x) == self.space.Ntot_0form
        x_loc = self.space.extract_0(x)

        # assert x_loc.shape == (self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])

        # step1 : I.T(x)
        mat_dofs = kron_solve_3d([self.N_1.T, self.N_2.T, self.N_3.T], x_loc)

        # step2 : R.T( I.T(x) )
        mat_f = self.space.projectors.dofs_T("0", mat_dofs)

        # step3 : F.T( R.T( I.T(x) ) )
        mat_f_c = self.p0_eq_0 * mat_f

        res = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_c)

        return res.flatten()

    # =================================================================
    def T1_dot(self, x):
        """
        Matrix-vector product T1.x

        Parameters
        ----------
            x : np.array
                dim R^{N^1}

        Returns
        ----------
            res : np.array
                dim R^{N^1}

        Notes
        -----
            T1    = pi_1[Bmat_eq * G_inv * lambda^1]   in  R^{N^1 x N^1}

            Bmat_eq  = [[0  , -b_eq_z,  b_eq_y], [b_eq_z,     0  , -b_eq_x], [-b_eq_y,  b_eq_x,     0]]

            T1.x = I_1( R_1 ( F_T1.x))

            I_1 ... inverse inter/histopolation matrix (tensor product)

            R_1  ... compute DOFs from function values at point set pts_ijk

            F_T1[ijk, mno] = B_eq(pts_ijk) * G_inv(pts_ijk) lambda^1_mno(pts_ijk)

            * spline evaluation (at V_1 point sets)
                * [his, int, int] points: {quad_pts[0], greville[1], greville[2]}
                * [int, his, int] points: {greville[0], quad_pts[1], greville[2]}
                * [int, int, his] points: {greville[0], greville[1], quad_pts[2]}

            * Components of F_T1:
                * evaluated at [his, int, int] : (D, N, N) * (G_inv_31 * b_eq_y - G_inv_21 * b_eq_z) + (N, D, N) * (G_inv_32 * b_eq_y - G_inv_22 * b_eq_z) + (N, N, D) * (G_inv_33 * b_eq_y - G_inv_23 * b_eq_z)
                * evaluated at [int, his, int] : (D, N, N) * (G_inv_11 * b_eq_z - G_inv_31 * b_eq_x) + (N, D, N) * (G_inv_12 * b_eq_z - G_inv_32 * b_eq_x) + (N, N, D) * (G_inv_13 * b_eq_z - G_inv_33 * b_eq_x)
                * evaluated at [int, int, his] : (D, N, N) * (G_inv_21 * b_eq_x - G_inv_11 * b_eq_y) + (N, D, N) * (G_inv_22 * b_eq_x - G_inv_12 * b_eq_y) + (N, N, D) * (G_inv_23 * b_eq_x - G_inv_13 * b_eq_y)
        """

        # x dim check
        # x should be R{N^1}
        # assert len(x) == self.space.Ntot_1form_cum[-1]
        x_loc = list(self.space.extract_1(x))

        # assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        # assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        # assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_11 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts0_N_3], x_loc[0])
        mat_f_12 = kron_matvec_3d([self.pts1_N_1, self.pts0_D_2, self.pts0_N_3], x_loc[1])
        mat_f_13 = kron_matvec_3d([self.pts1_N_1, self.pts0_N_2, self.pts0_D_3], x_loc[2])

        # xi2
        mat_f_21 = kron_matvec_3d([self.pts0_D_1, self.pts1_N_2, self.pts0_N_3], x_loc[0])
        mat_f_22 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts0_N_3], x_loc[1])
        mat_f_23 = kron_matvec_3d([self.pts0_N_1, self.pts1_N_2, self.pts0_D_3], x_loc[2])

        # xi3
        mat_f_31 = kron_matvec_3d([self.pts0_D_1, self.pts0_N_2, self.pts1_N_3], x_loc[0])
        mat_f_32 = kron_matvec_3d([self.pts0_N_1, self.pts0_D_2, self.pts1_N_3], x_loc[1])
        mat_f_33 = kron_matvec_3d([self.pts0_N_1, self.pts0_N_2, self.pts1_D_3], x_loc[2])

        mat_f_11_c = mat_f_11 * (self.g_inv_11[2, 0] * self.b2_eq_11_2 - self.g_inv_11[1, 0] * self.b2_eq_11_3)
        mat_f_12_c = mat_f_12 * (self.g_inv_11[2, 1] * self.b2_eq_11_2 - self.g_inv_11[1, 1] * self.b2_eq_11_3)
        mat_f_13_c = mat_f_13 * (self.g_inv_11[2, 2] * self.b2_eq_11_2 - self.g_inv_11[1, 2] * self.b2_eq_11_3)
        mat_f_21_c = mat_f_21 * (self.g_inv_12[0, 0] * self.b2_eq_12_3 - self.g_inv_12[2, 0] * self.b2_eq_12_1)
        mat_f_22_c = mat_f_22 * (self.g_inv_12[0, 1] * self.b2_eq_12_3 - self.g_inv_12[2, 1] * self.b2_eq_12_1)
        mat_f_23_c = mat_f_23 * (self.g_inv_12[0, 2] * self.b2_eq_12_3 - self.g_inv_12[2, 2] * self.b2_eq_12_1)
        mat_f_31_c = mat_f_31 * (self.g_inv_13[1, 0] * self.b2_eq_13_1 - self.g_inv_13[0, 0] * self.b2_eq_13_2)
        mat_f_32_c = mat_f_32 * (self.g_inv_13[1, 1] * self.b2_eq_13_1 - self.g_inv_13[0, 1] * self.b2_eq_13_2)
        mat_f_33_c = mat_f_33 * (self.g_inv_13[1, 2] * self.b2_eq_13_1 - self.g_inv_13[0, 2] * self.b2_eq_13_2)

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.space.projectors.dofs("11", mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.space.projectors.dofs("12", mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.space.projectors.dofs("13", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : histo(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("11", DOF_1)

        # xi2 : inter(xi1)-histo(xi2)-inter(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("12", DOF_2)

        # xi3 : inter(xi1)-inter(xi2)-histo(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("13", DOF_3)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # =================================================================
    def transpose_T1_dot(self, x):
        """
        Matrix-vector product T1.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^1}

        Returns
        ----------
            res : np.array
                dim R{N^1}

        Notes
        -----
            T1.x = I_1( R_1 ( F_T1.x))

            T1.T.x = F_T1.T( R_1.T ( I_1.T.x))

            See T1_dot for more details.
        """

        # x dim check
        # x should be R{N^1}
        # assert len(x) == self.space.Ntot_1form_cum[-1]
        x_loc = list(self.space.extract_1(x))

        # assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        # assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        # assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # step1 : I.T(x)
        # xi1 : transpose of histo(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.D_1.T, self.N_2.T, self.N_3.T], x_loc[0])

        # xi2 : transpose of inter(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.D_2.T, self.N_3.T], x_loc[1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-histo(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.D_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.space.projectors.dofs_T("11", mat_dofs_1)

        # xi2
        mat_f_2 = self.space.projectors.dofs_T("12", mat_dofs_2)

        # xi3
        mat_f_3 = self.space.projectors.dofs_T("13", mat_dofs_3)

        # step3 : F.T( R.T( I.T(x) ) )
        mat_f_11_c = mat_f_1 * (self.g_inv_11[2, 0] * self.b2_eq_11_2 - self.g_inv_11[1, 0] * self.b2_eq_11_3)
        mat_f_12_c = mat_f_1 * (self.g_inv_11[2, 1] * self.b2_eq_11_2 - self.g_inv_11[1, 1] * self.b2_eq_11_3)
        mat_f_13_c = mat_f_1 * (self.g_inv_11[2, 2] * self.b2_eq_11_2 - self.g_inv_11[1, 2] * self.b2_eq_11_3)
        mat_f_21_c = mat_f_2 * (self.g_inv_12[0, 0] * self.b2_eq_12_3 - self.g_inv_12[2, 0] * self.b2_eq_12_1)
        mat_f_22_c = mat_f_2 * (self.g_inv_12[0, 1] * self.b2_eq_12_3 - self.g_inv_12[2, 1] * self.b2_eq_12_1)
        mat_f_23_c = mat_f_2 * (self.g_inv_12[0, 2] * self.b2_eq_12_3 - self.g_inv_12[2, 2] * self.b2_eq_12_1)
        mat_f_31_c = mat_f_3 * (self.g_inv_13[1, 0] * self.b2_eq_13_1 - self.g_inv_13[0, 0] * self.b2_eq_13_2)
        mat_f_32_c = mat_f_3 * (self.g_inv_13[1, 1] * self.b2_eq_13_1 - self.g_inv_13[0, 1] * self.b2_eq_13_2)
        mat_f_33_c = mat_f_3 * (self.g_inv_13[1, 2] * self.b2_eq_13_1 - self.g_inv_13[0, 2] * self.b2_eq_13_2)

        res_11 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_11_c)
        res_12 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts0_N_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_N_2.T, self.pts0_D_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts0_N_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_22_c)
        res_23 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_N_2.T, self.pts1_N_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_33_c)

        res_1 = res_11 + res_21 + res_31
        res_2 = res_12 + res_22 + res_32
        res_3 = res_13 + res_23 + res_33

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # =================================================================
    def X1_dot(self, x):
        """
        Matrix-vector product X1.x

        Parameters
        ----------
            x : np.array
                dim R^{N^1}

        Returns
        ----------
            res : list
                3 np.arrays of dim R^{N^0}

        Notes
        -----
            X1 = pi_0[DF^-T * lambda^1]   in  R^{N^0 x 3 x N^1}

            X1.x = I_0( R_0 ( F_X1.x))

            I_0 ... inverse interpolation matrix (tensor product)

            R_0  ... compute DOFs from function values at point set pts_ijk

            F_X1[ijk, mno] = DF^-T(pts.ijk) * lambda^1_mno(pts_ijk)

            * spline evaluation (at V_0 point sets)
                * [int, int, int] points: {greville[0], greville[1], greville[2]}

            * Components of F_X1:
                * evaluated at [int, int, int] : (D, N, N) * DF^-T_11 + (N, D, N) * DF^-T_12 + (N, N, D) * DF^-T_13
                * evaluated at [int, int, int] : (D, N, N) * DF^-T_21 + (N, D, N) * DF^-T_22 + (N, N, D) * DF^-T_23
                * evaluated at [int, int, int] : (D, N, N) * DF^-T_31 + (N, D, N) * DF^-T_32 + (N, N, D) * DF^-T_33
        """

        # x dim check
        # x should be R{N^1}
        assert len(x) == self.space.Ntot_1form_cum[-1]
        x_loc = list(self.space.extract_1(x))

        assert x_loc[0].shape == (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        assert x_loc[1].shape == (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        assert x_loc[2].shape == (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_1 = kron_matvec_3d([self.pts0_D_1, self.pts0_N_2, self.pts0_N_3], x_loc[0])
        mat_f_2 = kron_matvec_3d([self.pts0_N_1, self.pts0_D_2, self.pts0_N_3], x_loc[1])
        mat_f_3 = kron_matvec_3d([self.pts0_N_1, self.pts0_N_2, self.pts0_D_3], x_loc[2])

        mat_f_11_c = mat_f_1 * self.df_inv_0[0, 0]
        mat_f_12_c = mat_f_2 * self.df_inv_0[1, 0]
        mat_f_13_c = mat_f_3 * self.df_inv_0[2, 0]
        mat_f_21_c = mat_f_1 * self.df_inv_0[0, 1]
        mat_f_22_c = mat_f_2 * self.df_inv_0[1, 1]
        mat_f_23_c = mat_f_3 * self.df_inv_0[2, 1]
        mat_f_31_c = mat_f_1 * self.df_inv_0[0, 2]
        mat_f_32_c = mat_f_2 * self.df_inv_0[1, 2]
        mat_f_33_c = mat_f_3 * self.df_inv_0[2, 2]

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c

        # ========== Step 2 : R( F(x) ) ==========#
        DOF_1 = self.space.projectors.dofs("0", mat_f_1_c)
        DOF_2 = self.space.projectors.dofs("0", mat_f_2_c)
        DOF_3 = self.space.projectors.dofs("0", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # inter(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("0", DOF_1)

        # inter(xi1)-inter(xi2)-inter(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("0", DOF_2)

        # inter(xi1)-inter(xi2)-inter(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("0", DOF_3)

        return [res_1.flatten(), res_2.flatten(), res_3.flatten()]

    # =================================================================
    def transpose_X1_dot(self, x):
        """
        Matrix-vector product X1.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^0 x 3}

        Returns
        ----------
            res : np.array
                dim R{N^1}

        Notes
        -----
            X1.x = I_0( R_0 ( F_X1.x))

            X1.T.x = F_X1.T( R_0.T ( I_0.T.x))

            See X1_dot for more details.
        """

        # x dim check
        # x should be R{N^0 * 3}
        # assert len(x) == self.space.Ntot_0form * 3
        # x_loc_1 = self.space.extract_0(np.split(x,3)[0])
        # x_loc_2 = self.space.extract_0(np.split(x,3)[1])
        # x_loc_3 = self.space.extract_0(np.split(x,3)[2])
        # x_loc = list((x_loc_1, x_loc_2, x_loc_3))

        x_loc_1 = self.space.extract_0(x[0])
        x_loc_2 = self.space.extract_0(x[1])
        x_loc_3 = self.space.extract_0(x[2])
        x_loc = list((x_loc_1, x_loc_2, x_loc_3))

        assert x_loc[0].shape == (self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])
        assert x_loc[1].shape == (self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])
        assert x_loc[2].shape == (self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])

        # step1 : I.T(x)
        # xi1 : transpose of inter(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.N_1.T, self.N_2.T, self.N_3.T], x_loc[0])

        # xi2 : transpose of inter(xi1)-inter(xi2)-inter(xi3)-polati on.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.N_2.T, self.N_3.T], x_loc[1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.N_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        mat_f_1 = self.space.projectors.dofs_T("0", mat_dofs_1)
        mat_f_2 = self.space.projectors.dofs_T("0", mat_dofs_2)
        mat_f_3 = self.space.projectors.dofs_T("0", mat_dofs_3)

        # step3 : F.T( R.T( I.T(x) ) )
        mat_f_11_c = mat_f_1 * self.df_inv_0[0, 0]
        mat_f_12_c = mat_f_1 * self.df_inv_0[1, 0]
        mat_f_13_c = mat_f_1 * self.df_inv_0[2, 0]
        mat_f_21_c = mat_f_2 * self.df_inv_0[0, 1]
        mat_f_22_c = mat_f_2 * self.df_inv_0[1, 1]
        mat_f_23_c = mat_f_2 * self.df_inv_0[2, 1]
        mat_f_31_c = mat_f_3 * self.df_inv_0[0, 2]
        mat_f_32_c = mat_f_3 * self.df_inv_0[1, 2]
        mat_f_33_c = mat_f_3 * self.df_inv_0[2, 2]

        res_11 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_11_c)
        res_12 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_D_2.T, self.pts0_N_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts0_D_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_D_2.T, self.pts0_N_3.T], mat_f_22_c)
        res_23 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts0_D_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_D_2.T, self.pts0_N_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts0_D_3.T], mat_f_33_c)

        res_1 = res_11 + res_21 + res_31
        res_2 = res_12 + res_22 + res_32
        res_3 = res_13 + res_23 + res_33

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    ########################################
    ########## 2-form formulation ##########
    ########################################
    # ====================================================================
    def Q2_dot(self, x):
        """
        Matrix-vector product Q2.x

        Parameters
        ----------
            x : np.array
                dim R^{N^2}

        Returns
        ----------
            res : np.array
                dim R^{N^2}

        Notes
        -----
            Q2 = pi_2[rho_eq / g_sqrt * lambda^2]  in  R^{N^2 x N^2}

            Q2.x = I_2( R_2 ( F_Q2.x))

            I_2 ... inverse inter/histopolation matrix (tensor product)

            R_2  ... compute DOFs from function values at point set pts_ijk

            F_Q2[ijk, mno] = rho_eq(pts_ijk) / g_sqrt(pts_ijk) * lambda^2_mno(pts_ijk)

            * spline evaluation (at V_2 point sets)
                * [int, his, his] points: {greville[0], quad_pts[1], quad_pts[2]}
                * [his, int, his] points: {quad_pts[0], greville[1], quad_pts[2]}
                * [his, his, int] points: {quad_pts[0], quad_pts[1], greville[2]}

            * Components of F_Q2:
                * evaluated at [int, his, his] : (N, D, D) * rho_eq / g_sqrt
                * evaluated at [his, int, his] : (D, N, D) * rho_eq / g_sqrt
                * evaluated at [his, his, int] : (D, D, N) * rho_eq / g_sqrt
        """

        # x dim check
        # x should be R{N^2}
        # assert len(x) == self.space.Ntot_2form_cum[-1]
        x_loc = list(self.space.extract_2(x))

        # assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        # assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        # assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_1 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts1_D_3], x_loc[0])

        # xi2
        mat_f_2 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts1_D_3], x_loc[1])

        # xi3
        mat_f_3 = kron_matvec_3d([self.pts1_D_1, self.pts1_D_2, self.pts0_N_3], x_loc[2])

        mat_f_1_c = mat_f_1 * self.n3_eq_21 / self.det_df_21
        mat_f_2_c = mat_f_2 * self.n3_eq_22 / self.det_df_22
        mat_f_3_c = mat_f_3 * self.n3_eq_23 / self.det_df_23

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.space.projectors.dofs("21", mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.space.projectors.dofs("22", mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.space.projectors.dofs("23", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : inter(xi1)-histo(xi2)-histo(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("21", DOF_1)

        # xi2 : histo(xi1)-inter(xi2)-histo(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("22", DOF_2)

        # xi3 : histo(xi1)-histo(xi2)-inter(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("23", DOF_3)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def transpose_Q2_dot(self, x):
        """
        Matrix-vector product Q2.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^2}

        Returns
        ----------
            res : np.array
                dim R{N^2}

        Notes
        -----
            Q2.x = I_2( R_2 ( F_Q2.x))

            Q2.T.x = F_Q2.T( R_2.T ( I2.T.x))

            See Q2_dot for more details.
        """

        # x dim check
        # x should be R{N^2}
        # assert len(x) == self.space.Ntot_2form_cum[-1]
        x_loc = list(self.space.extract_2(x))

        # assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        # assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        # assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])

        # step1 : I.T(x)
        # xi1 : transpose of inter(xi1)-histo(xi2)-histo(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.N_1.T, self.D_2.T, self.D_3.T], x_loc[0])

        # xi2 : transpose of histo(xi1)-inter(xi2)-histo(xi3)-polati on.
        mat_dofs_2 = kron_solve_3d([self.D_1.T, self.N_2.T, self.D_3.T], x_loc[1])

        # xi3 : transpose of histo(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.D_1.T, self.D_2.T, self.N_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.space.projectors.dofs_T("21", mat_dofs_1)

        # xi2
        mat_f_2 = self.space.projectors.dofs_T("22", mat_dofs_2)

        # xi3
        mat_f_3 = self.space.projectors.dofs_T("23", mat_dofs_3)

        mat_f_1_c = mat_f_1 * self.n3_eq_21 / self.det_df_21
        mat_f_2_c = mat_f_2 * self.n3_eq_22 / self.det_df_22
        mat_f_3_c = mat_f_3 * self.n3_eq_23 / self.det_df_23

        res_1 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts1_D_3.T], mat_f_1_c)
        res_2 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_2_c)
        res_3 = kron_matvec_3d([self.pts1_D_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_3_c)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def T2_dot(self, x):
        """
        Matrix-vector product T2.x

        Parameters
        ----------
            x : np.array
                dim R^{N^2}

        Returns
        ----------
            res : np.array
                dim R^{N^1}

        Notes
        -----
            T2     = pi_1[Bmat_eq / g_sqrt * lambda^2]  in   R^{N^1 * N^2}

            Bmat_eq  = [[0  , -b_eq_z,  b_eq_y], [b_eq_z,     0  , -b_eq_x], [-b_eq_y,  b_eq_x,     0]] in R^{3 x 3}

            T2.x = I_1( R_1 ( F_T2.x))

            I_1 ... inverse inter/histopolation matrix (tensor product)

            R_1  ... compute DOFs from function values at point set pts_ijk

            F_T2[ijk, mno] = B_eq(pts_ijk) / g_sqrt(pts_ijk) * lambda^2_mno(pts_ijk)

            * spline evaluation (at V_1 point sets)
                * [his, int, int] points: {quad_pts[0], greville[1], greville[2]}
                * [int, his, int] points: {greville[0], quad_pts[1], greville[2]}
                * [int, int, his] points: {greville[0], greville[1], quad_pts[2]}

            * Components of F_T2:
                * evaluated at [his, int, int] : (D, N, D) * (-b_eq_z) / g_sqrt + (D, D, N) *  b_eq_y / g_sqrt
                * evaluated at [int, his, int] : (N, D, D) *  b_eq_z / g_sqrt   + (D, D, N) * (-b_eq_x) / g_sqrt
                * evaluated at [int, int, his] : (N, D, D) * (-b_eq_y) / g_sqrt + (D, N, D) *  b_eq_x / g_sqrt,
        """

        # x dim check
        # x should be R{N^2}
        # assert len(x) == self.space.Ntot_2form_cum[-1]
        x_loc = list(self.space.extract_2(x))

        # assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        # assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        # assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_11 = kron_matvec_3d([self.pts1_N_1, self.pts0_D_2, self.pts0_D_3], x_loc[0])  # 0
        mat_f_12 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts0_D_3], x_loc[1])
        mat_f_13 = kron_matvec_3d([self.pts1_D_1, self.pts0_D_2, self.pts0_N_3], x_loc[2])

        # xi2
        mat_f_21 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts0_D_3], x_loc[0])
        mat_f_22 = kron_matvec_3d([self.pts0_D_1, self.pts1_N_2, self.pts0_D_3], x_loc[1])  # 0
        mat_f_23 = kron_matvec_3d([self.pts0_D_1, self.pts1_D_2, self.pts0_N_3], x_loc[2])

        # xi3
        mat_f_31 = kron_matvec_3d([self.pts0_N_1, self.pts0_D_2, self.pts1_D_3], x_loc[0])
        mat_f_32 = kron_matvec_3d([self.pts0_D_1, self.pts0_N_2, self.pts1_D_3], x_loc[1])
        mat_f_33 = kron_matvec_3d([self.pts0_D_1, self.pts0_D_2, self.pts1_N_3], x_loc[2])  # 0

        mat_f_11_c = mat_f_11 * 0 / self.det_df_11
        mat_f_12_c = mat_f_12 * (-self.b2_eq_11_3) / self.det_df_11
        mat_f_13_c = mat_f_13 * (self.b2_eq_11_2) / self.det_df_11
        mat_f_21_c = mat_f_21 * (self.b2_eq_12_3) / self.det_df_12
        mat_f_22_c = mat_f_22 * 0 / self.det_df_12
        mat_f_23_c = mat_f_23 * (-self.b2_eq_12_1) / self.det_df_12
        mat_f_31_c = mat_f_31 * (-self.b2_eq_13_2) / self.det_df_13
        mat_f_32_c = mat_f_32 * (self.b2_eq_13_1) / self.det_df_13
        mat_f_33_c = mat_f_33 * 0 / self.det_df_13

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.space.projectors.dofs("11", mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.space.projectors.dofs("12", mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.space.projectors.dofs("13", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : histo(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("11", DOF_1)

        # xi2 : inter(xi1)-histo(xi2)-inter(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("12", DOF_2)

        # xi3 : inter(xi1)-inter(xi2)-histo(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("13", DOF_3)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def transpose_T2_dot(self, x):
        """
        Matrix-vector product T2.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^1}

        Returns
        ----------
            res : np.array
                dim R{N^2}

        Notes
        -----
            T2.x = I_1( R_1 ( F_T2.x))

            T2.T.x = F_T2.T( R_1.T ( I_1.T.x))

            See T2_dot for more details.
        """

        # x dim check
        # x should be R{N^1}
        # assert len(x) == self.space.Ntot_1form_cum[-1]
        x_loc = list(self.space.extract_1(x))

        # assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        # assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        # assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # step1 : I.T(x)
        # xi1 : transpose of histo(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.D_1.T, self.N_2.T, self.N_3.T], x_loc[0])

        # xi2 : transpose of inter(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.D_2.T, self.N_3.T], x_loc[1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-histo(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.D_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.space.projectors.dofs_T("11", mat_dofs_1)

        # xi2
        mat_f_2 = self.space.projectors.dofs_T("12", mat_dofs_2)

        # xi3
        mat_f_3 = self.space.projectors.dofs_T("13", mat_dofs_3)

        # step3 : F.T( R.T( I.T(x) ) )
        mat_f_11_c = mat_f_1 * 0 / self.det_df_11
        mat_f_12_c = mat_f_1 * (-self.b2_eq_11_3) / self.det_df_11
        mat_f_13_c = mat_f_1 * (self.b2_eq_11_2) / self.det_df_11
        mat_f_21_c = mat_f_2 * (self.b2_eq_12_3) / self.det_df_12
        mat_f_22_c = mat_f_2 * 0 / self.det_df_12
        mat_f_23_c = mat_f_2 * (-self.b2_eq_12_1) / self.det_df_12
        mat_f_31_c = mat_f_3 * (-self.b2_eq_13_2) / self.det_df_13
        mat_f_32_c = mat_f_3 * (self.b2_eq_13_1) / self.det_df_13
        mat_f_33_c = mat_f_3 * 0 / self.det_df_13

        res_11 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts0_D_3.T], mat_f_11_c)  # 0
        res_12 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts0_D_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_D_2.T, self.pts0_N_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts0_D_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_22_c)  # 0
        res_23 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_D_2.T, self.pts1_D_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_33_c)  # 0

        res_1 = res_11 + res_21 + res_31
        res_2 = res_12 + res_22 + res_32
        res_3 = res_13 + res_23 + res_33

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def P2_dot(self, x):
        """
        Matrix-vector product P2.x

        Parameters
        ----------
            x : np.array
                dim R^{N^2}

        Returns
        ----------
            res : np.array
                dim R^{N^2}

        Notes
        -----
            P2    = pi_2[G_inv * jmat_eq * lambda^2]  in   R^{N^2 x N^2}

            jmat_eq  = [[0  , -j_eq_z,  j_eq_y], [ j_eq_z,     0  , -j_eq_x], [-j_eq_y,  j_eq_x,     0]]

            P2.x = I_2( R_2 ( F_P2.x))

            I_2 ... inverse inter/histopolation matrix (tensor product)

            R_2  ... compute DOFs from function values at point set pts_ijk

            F_P2[ijk, mno] = G_inv(pts_ijk) * jmat_eq(pts_ijk) lambda^2_mno(pts_ijk)

            * spline evaluation (at V_2 point sets)
                * [int, his, his] points: {greville[0], quad_pts[1], quad_pts[2]}
                * [his, int, his] points: {quad_pts[0], greville[1], quad_pts[2]}
                * [his, his, int] points: {quad_pts[0], quad_pts[1], greville[2]}

            * Components of F_P2:
                * evaluated at [int, his, his] : (N, D, D) * (G_inv_12 * j_eq_z - G_inv_13 * j_eq_y) + (D, N, D) * (G_inv_13 * j_eq_x - G_inv_11 * j_eq_z) + (D, D, N) * (G_inv_11 * j_eq_y - G_inv_12 * j_eq_x)
                * evaluated at [his, int, his] : (N, D, D) * (G_inv_22 * j_eq_z - G_inv_23 * j_eq_y) + (D, N, D) * (G_inv_23 * j_eq_x - G_inv_21 * j_eq_z) + (D, D, N) * (G_inv_21 * j_eq_y - G_inv_22 * j_eq_x)
                * evaluated at [his, his, int] : (N, D, D) * (G_inv_32 * j_eq_z - G_inv_33 * j_eq_y) + (D, N, D) * (G_inv_33 * j_eq_x - G_inv_31 * j_eq_z) + (D, D, N) * (G_inv_31 * j_eq_y - G_inv_32 * j_eq_x)
        """

        # x dim check
        # x should be R{N^2}
        # assert len(x) == self.space.Ntot_2form_cum[-1]
        x_loc = list(self.space.extract_2(x))

        # assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        # assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        # assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_11 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts1_D_3], x_loc[0])
        mat_f_12 = kron_matvec_3d([self.pts0_D_1, self.pts1_N_2, self.pts1_D_3], x_loc[1])
        mat_f_13 = kron_matvec_3d([self.pts0_D_1, self.pts1_D_2, self.pts1_N_3], x_loc[2])

        # xi2
        mat_f_21 = kron_matvec_3d([self.pts1_N_1, self.pts0_D_2, self.pts1_D_3], x_loc[0])
        mat_f_22 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts1_D_3], x_loc[1])
        mat_f_23 = kron_matvec_3d([self.pts1_D_1, self.pts0_D_2, self.pts1_N_3], x_loc[2])

        # xi3
        mat_f_31 = kron_matvec_3d([self.pts1_N_1, self.pts1_D_2, self.pts0_D_3], x_loc[0])
        mat_f_32 = kron_matvec_3d([self.pts1_D_1, self.pts1_N_2, self.pts0_D_3], x_loc[1])
        mat_f_33 = kron_matvec_3d([self.pts1_D_1, self.pts1_D_2, self.pts0_N_3], x_loc[2])

        mat_f_11_c = mat_f_11 * (self.g_inv_21[0, 1] * self.j2_eq_21_3 - self.g_inv_21[0, 2] * self.j2_eq_21_2)
        mat_f_12_c = mat_f_12 * (self.g_inv_21[0, 2] * self.j2_eq_21_1 - self.g_inv_21[0, 0] * self.j2_eq_21_3)
        mat_f_13_c = mat_f_13 * (self.g_inv_21[0, 0] * self.j2_eq_21_2 - self.g_inv_21[0, 1] * self.j2_eq_21_1)
        mat_f_21_c = mat_f_21 * (self.g_inv_22[1, 1] * self.j2_eq_22_3 - self.g_inv_22[1, 2] * self.j2_eq_22_2)
        mat_f_22_c = mat_f_22 * (self.g_inv_22[1, 2] * self.j2_eq_22_1 - self.g_inv_22[1, 0] * self.j2_eq_22_3)
        mat_f_23_c = mat_f_23 * (self.g_inv_22[1, 0] * self.j2_eq_22_2 - self.g_inv_22[1, 1] * self.j2_eq_22_1)
        mat_f_31_c = mat_f_31 * (self.g_inv_23[2, 1] * self.j2_eq_23_3 - self.g_inv_23[2, 2] * self.j2_eq_23_2)
        mat_f_32_c = mat_f_32 * (self.g_inv_23[2, 2] * self.j2_eq_23_1 - self.g_inv_23[2, 0] * self.j2_eq_23_3)
        mat_f_33_c = mat_f_33 * (self.g_inv_23[2, 0] * self.j2_eq_23_2 - self.g_inv_23[2, 1] * self.j2_eq_23_1)

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.space.projectors.dofs("21", mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.space.projectors.dofs("22", mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.space.projectors.dofs("23", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : inter(xi1)-histo(xi2)-histo(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("21", DOF_1)

        # xi2 : histo(xi1)-inter(xi2)-histo(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("22", DOF_2)

        # xi3 : histo(xi1)-histo(xi2)-inter(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("23", DOF_3)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def transpose_P2_dot(self, x):
        """
        Matrix-vector product P2.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^2}

        Returns
        ----------
            res : np.array
                dim R{N^2}

        Notes
        -----
            P2.x = I_2( R_2 ( F_P2.x))

            P2.T.x = F_P2.T( R_2.T ( I_2.T.x))

            See P2_dot for more details.
        """

        # x dim check
        # x should be R{N^2}
        # assert len(x) == self.space.Ntot_2form_cum[-1]
        x_loc = list(self.space.extract_2(x))

        # assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        # assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        # assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])

        # step1 : I.T(x)
        # xi1 : transpose of inter(xi1)-histo(xi2)-histo(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.N_1.T, self.D_2.T, self.D_3.T], x_loc[0])

        # xi2 : transpose of histo(xi1)-inter(xi2)-histo(xi3)-polati on.
        mat_dofs_2 = kron_solve_3d([self.D_1.T, self.N_2.T, self.D_3.T], x_loc[1])

        # xi3 : transpose of histo(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.D_1.T, self.D_2.T, self.N_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.space.projectors.dofs_T("21", mat_dofs_1)

        # xi2
        mat_f_2 = self.space.projectors.dofs_T("22", mat_dofs_2)

        # xi3
        mat_f_3 = self.space.projectors.dofs_T("23", mat_dofs_3)

        mat_f_11_c = mat_f_1 * (self.g_inv_21[0, 1] * self.j2_eq_21_3 - self.g_inv_21[0, 2] * self.j2_eq_21_2)
        mat_f_12_c = mat_f_1 * (self.g_inv_21[0, 2] * self.j2_eq_21_1 - self.g_inv_21[0, 0] * self.j2_eq_21_3)
        mat_f_13_c = mat_f_1 * (self.g_inv_21[0, 0] * self.j2_eq_21_2 - self.g_inv_21[0, 1] * self.j2_eq_21_1)
        mat_f_21_c = mat_f_2 * (self.g_inv_22[1, 1] * self.j2_eq_22_3 - self.g_inv_22[1, 2] * self.j2_eq_22_2)
        mat_f_22_c = mat_f_2 * (self.g_inv_22[1, 2] * self.j2_eq_22_1 - self.g_inv_22[1, 0] * self.j2_eq_22_3)
        mat_f_23_c = mat_f_2 * (self.g_inv_22[1, 0] * self.j2_eq_22_2 - self.g_inv_22[1, 1] * self.j2_eq_22_1)
        mat_f_31_c = mat_f_3 * (self.g_inv_23[2, 1] * self.j2_eq_23_3 - self.g_inv_23[2, 2] * self.j2_eq_23_2)
        mat_f_32_c = mat_f_3 * (self.g_inv_23[2, 2] * self.j2_eq_23_1 - self.g_inv_23[2, 0] * self.j2_eq_23_3)
        mat_f_33_c = mat_f_3 * (self.g_inv_23[2, 0] * self.j2_eq_23_2 - self.g_inv_23[2, 1] * self.j2_eq_23_1)

        res_11 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts1_D_3.T], mat_f_11_c)
        res_12 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts1_D_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_D_2.T, self.pts1_N_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts1_D_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_22_c)
        res_23 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts1_N_1.T, self.pts1_D_2.T, self.pts0_D_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts1_D_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts1_D_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_33_c)

        res_1 = res_11 + res_21 + res_31
        res_2 = res_12 + res_22 + res_32
        res_3 = res_13 + res_23 + res_33

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def S2_dot(self, x):
        """
        Matrix-vector product S2.x

        Parameters
        ----------
            x : np.array
                dim R^{N^2}

        Returns
        ----------
            res : np.array
                dim R^{N^2}

        Notes
        -----
            S2 = pi_2[p_eq / g_sqrt * lambda^2]   in  R^{N^2 x N^2}

            S2.x = I_2( R_2 ( F_S2.x))

            I_2 ... inverse inter/histopolation matrix (tensor product)

            R_2  ... compute DOFs from function values at point set pts_ijk

            F_S2[ijk, mno] = p_eq(pts_ijk) / g_sqrt(pts_ijk) * lambda^2_mno(pts_ijk)

            * spline evaluation (at V_2 point sets)
                * [int, his, his] points: {greville[0], quad_pts[1], quad_pts[2]}
                * [his, int, his] points: {quad_pts[0], greville[1], quad_pts[2]}
                * [his, his, int] points: {quad_pts[0], quad_pts[1], greville[2]}

            * Components of F_S2:
                * evaluated at [int, his, his] : (N, D, D) * p_eq / g_sqrt
                * evaluated at [his, int, his] : (D, N, D) * p_eq / g_sqrt
                * evaluated at [his, his, int] : (D, D, N) * p_eq / g_sqrt
        """

        # x dim check
        # x should be R{N^2}
        # assert len(x) == self.space.Ntot_2form_cum[-1]
        x_loc = list(self.space.extract_2(x))

        # assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        # assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        # assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_1 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts1_D_3], x_loc[0])

        # xi2
        mat_f_2 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts1_D_3], x_loc[1])

        # xi3
        mat_f_3 = kron_matvec_3d([self.pts1_D_1, self.pts1_D_2, self.pts0_N_3], x_loc[2])

        mat_f_1_c = mat_f_1 * self.p3_eq_21 / self.det_df_21
        mat_f_2_c = mat_f_2 * self.p3_eq_22 / self.det_df_22
        mat_f_3_c = mat_f_3 * self.p3_eq_23 / self.det_df_23

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.space.projectors.dofs("21", mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.space.projectors.dofs("22", mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.space.projectors.dofs("23", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : inter(xi1)-histo(xi2)-histo(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("21", DOF_1)

        # xi2 : histo(xi1)-inter(xi2)-histo(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("22", DOF_2)

        # xi3 : histo(xi1)-histo(xi2)-inter(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("23", DOF_3)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def transpose_S2_dot(self, x):
        """
        Matrix-vecotr product S2.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^2}

        Returns
        ----------
            res : np.array
                dim R{N^2}

        Notes
        -----
            S2.x = I_2( R_2 ( F_S2.x))

            S2.T.x = F_S2.T( R_2.T ( I_2.T.x))

            See S2_dot for more details.
        """

        # x dim check
        # x should be R{N^2}
        # assert len(x) == self.space.Ntot_2form_cum[-1]
        x_loc = list(self.space.extract_2(x))

        # assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        # assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        # assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])

        # step1 : I.T(x)
        # xi1 : transpose of inter(xi1)-histo(xi2)-histo(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.N_1.T, self.D_2.T, self.D_3.T], x_loc[0])

        # xi2 : transpose of histo(xi1)-inter(xi2)-histo(xi3)-polati on.
        mat_dofs_2 = kron_solve_3d([self.D_1.T, self.N_2.T, self.D_3.T], x_loc[1])

        # xi3 : transpose of histo(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.D_1.T, self.D_2.T, self.N_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.space.projectors.dofs_T("21", mat_dofs_1)

        # xi2
        mat_f_2 = self.space.projectors.dofs_T("22", mat_dofs_2)

        # xi3
        mat_f_3 = self.space.projectors.dofs_T("23", mat_dofs_3)

        mat_f_1_c = mat_f_1 * self.p3_eq_21 / self.det_df_21
        mat_f_2_c = mat_f_2 * self.p3_eq_22 / self.det_df_22
        mat_f_3_c = mat_f_3 * self.p3_eq_23 / self.det_df_23

        res_1 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts1_D_3.T], mat_f_1_c)
        res_2 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_2_c)
        res_3 = kron_matvec_3d([self.pts1_D_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_3_c)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def K2_dot(self, x):
        """
        Matrix-vector product K2.x

        Parameters
        ----------
            x : np.array
                dim R^{N^3}

        Returns
        ----------
            res : np.array
                dim R^{N^3}

        Notes
        -----
            K2 = pi_3[p_eq / g_sqrt * lambda^3]   in  R{N^3 x N^3}

            K2.x = I_3( R_3 ( F_K2.x))

            I_3 ... inverse histopolation matrix (tensor product)

            R_3  ... compute DOFs from function values at point set pts_ijk

            F_K2[ijk,mno] = p_eq(pts_ijk) / g_sqrt(pts_ijk) * lambda^3_mno(pts_ijk)

            * spline evaluation (at V_3 point sets)
                * [his, his, his] points: {quad_pts[0], quad_pts[1], quad_pts[2]}

            * Components of F_K2:
                * evaluated at [his, his, his] : (D, D, D) * p_eq / sqrt g
        """

        # x dim check
        # assert len(x) == self.space.Ntot_3form
        x_loc = self.space.extract_3(x)

        # assert x_loc.shape == (self.NbaseD[0], self.NbaseD[1], self.NbaseD[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        mat_f = kron_matvec_3d([self.pts1_D_1, self.pts1_D_2, self.pts1_D_3], x_loc)

        # Point-wise multiplication of mat_f and peq
        mat_f_c = mat_f * self.p3_eq_3 / self.det_df_3

        # ========== Step 2 : R( F(x) ) ==========#
        # Linear operator : evaluation values at the projection points to the Degree of Freedom of the spline.
        DOF = self.space.projectors.dofs("3", mat_f_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # histo(xi1)-histo(xi2)-histo(xi3)-polation.
        res = self.space.projectors.PI_mat("3", DOF)

        return res.flatten()

    # ====================================================================
    def transpose_K2_dot(self, x):
        """
        Matrix-vector product K2.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^3}

        Returns
        ----------
            res : 3d array
                dim R{N^3}

        Notes
        -----
            K2.x = I_3( R_3 ( F_K2.x))

            K2.T.x = F_K2.T( R_3.T ( I_3.T.x))

            See K2_dot for more details.
        """

        # x dim check
        # assert len(x) == self.space.Ntot_3form
        x_loc = self.space.extract_3(x)

        # assert x_loc.shape == (self.NbaseD[0], self.NbaseD[1], self.NbaseD[2])

        # step1 : I.T(x)
        mat_dofs = kron_solve_3d([self.D_1.T, self.D_2.T, self.D_3.T], x_loc)

        # step2 : R.T( I.T(x) )
        mat_f = self.space.projectors.dofs_T("3", mat_dofs)

        # step3 : F.T( R.T( I.T(x) ) )
        mat_f_c = self.p3_eq_3 * mat_f / self.det_df_3

        res = kron_matvec_3d([self.pts1_D_1.T, self.pts1_D_2.T, self.pts1_D_3.T], mat_f_c)

        return res.flatten()

    # ====================================================================
    def X2_dot(self, x):
        """
        Matrix-vector product X2.x

        Parameters
        ----------
            x : np.array
                dim R^{N^2}

        Returns
        ----------
            res : list
                3 np.arrays of dim R^{N^0}

        Notes
        -----
            X2 = pi_0[DF / g_sqrt *  lambda^2]  in   R^{N^0 x 3 x N^2}

            X2.x = I_0( R_0 ( F_X2.x))

            I_0 ... inverse interpolation matrix (tensor product)

            R_0  ... compute DOFs from function values at point set pts_ijk

            F_X2[ijk, mno] = G(pts_ijk) / g_sqrt(pts_ijk) * lambda^2_mno(pts_ijk)

            * spline evaluation (at V_0 point sets)
                * [int, int, int] points: {greville[0], greville[1], greville[2]}

            * Components of F_X2:
                * evaluated at [int, int, int] : (N, D, D) * df_11 / g_sqrt + (D, N, D) * df_21 / g_sqrt + (D, D, N) * df_31 / g_sqrt
                * evaluated at [int, int, int] : (N, D, D) * df_12 / g_sqrt + (D, N, D) * df_22 / g_sqrt + (D, D, N) * df_23 / g_sqrt
                * evaluated at [int, int, int] : (N, D, D) * df_13 / g_sqrt + (D, N, D) * df_23 / g_sqrt + (D, D, N) * df_33 / g_sqrt
        """

        # x dim check
        # x should be R{N^2}
        assert len(x) == self.space.Ntot_2form_cum[-1]
        x_loc = list(self.space.extract_2(x))

        assert x_loc[0].shape == (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        assert x_loc[1].shape == (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        assert x_loc[2].shape == (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_1 = kron_matvec_3d([self.pts0_N_1, self.pts0_D_2, self.pts0_D_3], x_loc[0])
        mat_f_2 = kron_matvec_3d([self.pts0_D_1, self.pts0_N_2, self.pts0_D_3], x_loc[1])
        mat_f_3 = kron_matvec_3d([self.pts0_D_1, self.pts0_D_2, self.pts0_N_3], x_loc[2])

        mat_f_11_c = mat_f_1 * self.df_0[0, 0] / self.det_df_0
        mat_f_12_c = mat_f_2 * self.df_0[0, 1] / self.det_df_0
        mat_f_13_c = mat_f_3 * self.df_0[0, 2] / self.det_df_0
        mat_f_21_c = mat_f_1 * self.df_0[1, 0] / self.det_df_0
        mat_f_22_c = mat_f_2 * self.df_0[1, 1] / self.det_df_0
        mat_f_23_c = mat_f_3 * self.df_0[1, 2] / self.det_df_0
        mat_f_31_c = mat_f_1 * self.df_0[2, 0] / self.det_df_0
        mat_f_32_c = mat_f_2 * self.df_0[2, 1] / self.det_df_0
        mat_f_33_c = mat_f_3 * self.df_0[2, 2] / self.det_df_0

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c

        # ========== Step 2 : R( F(x) ) ==========#
        DOF_1 = self.space.projectors.dofs("0", mat_f_1_c)
        DOF_2 = self.space.projectors.dofs("0", mat_f_2_c)
        DOF_3 = self.space.projectors.dofs("0", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # inter(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("0", DOF_1)

        # inter(xi1)-inter(xi2)-inter(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("0", DOF_2)

        # inter(xi1)-inter(xi2)-inter(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("0", DOF_3)

        return [res_1.flatten(), res_2.flatten(), res_3.flatten()]

    # ====================================================================
    def transpose_X2_dot(self, x):
        """
        Matrix-vector product X2.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^0 x 3}

        Returns
        ----------
            res : np.array
                dim R{N^2}

        Notes
        -----
            X2.x = I_0( R_0 ( F_X2.x))

            X2.T.x = F_X2.T( R_0.T ( I_0.T.x))

            See X2_dot for more details.
        """

        # x dim check
        # x should be R{N^0 * 3}
        # assert len(x) == self.space.Ntot_0form * 3
        # x_loc_1 = self.space.extract_0(np.split(x,3)[0])
        # x_loc_2 = self.space.extract_0(np.split(x,3)[1])
        # x_loc_3 = self.space.extract_0(np.split(x,3)[2])
        # x_loc = list((x_loc_1, x_loc_2, x_loc_3))

        x_loc_1 = self.space.extract_0(x[0])
        x_loc_2 = self.space.extract_0(x[1])
        x_loc_3 = self.space.extract_0(x[2])
        x_loc = list((x_loc_1, x_loc_2, x_loc_3))

        assert x_loc[0].shape == (self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])
        assert x_loc[1].shape == (self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])
        assert x_loc[2].shape == (self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])

        # step1 : I.T(x)
        # xi1 : transpose of inter(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.N_1.T, self.N_2.T, self.N_3.T], x_loc[0])

        # xi2 : transpose of inter(xi1)-inter(xi2)-inter(xi3)-polati on.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.N_2.T, self.N_3.T], x_loc[1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.N_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        mat_f_1 = self.space.projectors.dofs_T("0", mat_dofs_1)
        mat_f_2 = self.space.projectors.dofs_T("0", mat_dofs_2)
        mat_f_3 = self.space.projectors.dofs_T("0", mat_dofs_3)

        # step3 : F.T( R.T( I.T(x) ) )
        mat_f_11_c = mat_f_1 * self.df_0[0, 0] / self.det_df_0
        mat_f_12_c = mat_f_1 * self.df_0[0, 1] / self.det_df_0
        mat_f_13_c = mat_f_1 * self.df_0[0, 2] / self.det_df_0
        mat_f_21_c = mat_f_2 * self.df_0[1, 0] / self.det_df_0
        mat_f_22_c = mat_f_2 * self.df_0[1, 1] / self.det_df_0
        mat_f_23_c = mat_f_2 * self.df_0[1, 2] / self.det_df_0
        mat_f_31_c = mat_f_3 * self.df_0[2, 0] / self.det_df_0
        mat_f_32_c = mat_f_3 * self.df_0[2, 1] / self.det_df_0
        mat_f_33_c = mat_f_3 * self.df_0[2, 2] / self.det_df_0

        res_11 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_D_2.T, self.pts0_D_3.T], mat_f_11_c)
        res_12 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_N_2.T, self.pts0_D_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_D_2.T, self.pts0_N_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_D_2.T, self.pts0_D_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_N_2.T, self.pts0_D_3.T], mat_f_22_c)
        res_23 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_D_2.T, self.pts0_N_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_D_2.T, self.pts0_D_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_N_2.T, self.pts0_D_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_D_2.T, self.pts0_N_3.T], mat_f_33_c)

        res_1 = res_11 + res_21 + res_31
        res_2 = res_12 + res_22 + res_32
        res_3 = res_13 + res_23 + res_33

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def Z20_dot(self, x):
        """
        Matrix-vector product Z20.x

        Parameters
        ----------
            x : np.array
                dim R^{N^2}

        Returns
        ----------
            res : np.array
                dim R^{N^1}

        Notes
        -----
            Z20     = pi_1[G / g_sqrt * lambda^2]      in     R{N^1 x N^2}

            Z20.x = I_1( R_1 ( F_Z20.x))

            I_1 ... inverse inter/histopolation matrix (tensor product)

            R_1  ... compute DOFs from function values at point set pts_ijk

            F_Z20[ijk, mno] = G(pts_ijk) / g_sqrt(pts_ijk) * lambda^2_mno(pts_ijk)

            * spline evaluation (at V_1 point sets)
                * [his, int, int] points: {quad_pts[0], greville[1], greville[2]}
                * [int, his, int] points: {greville[0], quad_pts[1], greville[2]}
                * [int, int, his] points: {greville[0], greville[1], quad_pts[2]}

            * Components of F_Z20:
                * evaluated at [his, int, int] : (N, D, D) * G_11 / g_sqrt + (D, N, D) * G_12 / g_sqrt + (D, D, N) * G_13 / g_sqrt
                * evaluated at [int, his, int] : (N, D, D) * G_21 / g_sqrt + (D, N, D) * G_22 / g_sqrt + (D, D, N) * G_23 / g_sqrt
                * evaluated at [int, int, his] : (N, D, D) * G_31 / g_sqrt + (D, N, D) * G_32 / g_sqrt + (D, D, N) * G_33 / g_sqrt
        """

        # x dim check
        # x should be R{N^1}
        assert len(x) == self.space.Ntot_2form_cum[-1]
        x_loc = list(self.space.extract_2(x))

        assert x_loc[0].shape == (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        assert x_loc[1].shape == (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        assert x_loc[2].shape == (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_11 = kron_matvec_3d([self.pts1_N_1, self.pts0_D_2, self.pts0_D_3], x_loc[0])
        mat_f_12 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts0_D_3], x_loc[1])
        mat_f_13 = kron_matvec_3d([self.pts1_D_1, self.pts0_D_2, self.pts0_N_3], x_loc[2])

        # xi2
        mat_f_21 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts0_D_3], x_loc[0])
        mat_f_22 = kron_matvec_3d([self.pts0_D_1, self.pts1_N_2, self.pts0_D_3], x_loc[1])
        mat_f_23 = kron_matvec_3d([self.pts0_D_1, self.pts1_D_2, self.pts0_N_3], x_loc[2])

        # xi3
        mat_f_31 = kron_matvec_3d([self.pts0_N_1, self.pts0_D_2, self.pts1_D_3], x_loc[0])
        mat_f_32 = kron_matvec_3d([self.pts0_D_1, self.pts0_N_2, self.pts1_D_3], x_loc[1])
        mat_f_33 = kron_matvec_3d([self.pts0_D_1, self.pts0_D_2, self.pts1_N_3], x_loc[2])

        mat_f_11_c = mat_f_11 * self.g_11[0, 0] / self.det_df_11
        mat_f_12_c = mat_f_12 * self.g_11[0, 1] / self.det_df_11  # 0
        mat_f_13_c = mat_f_13 * self.g_11[0, 2] / self.det_df_11  # 0
        mat_f_21_c = mat_f_21 * self.g_12[1, 0] / self.det_df_12  # 0
        mat_f_22_c = mat_f_22 * self.g_12[1, 1] / self.det_df_12
        mat_f_23_c = mat_f_23 * self.g_12[1, 2] / self.det_df_12  # 0
        mat_f_31_c = mat_f_31 * self.g_13[2, 0] / self.det_df_13  # 0
        mat_f_32_c = mat_f_32 * self.g_13[2, 1] / self.det_df_13  # 0
        mat_f_33_c = mat_f_33 * self.g_13[2, 2] / self.det_df_13

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.space.projectors.dofs("11", mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.space.projectors.dofs("12", mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.space.projectors.dofs("13", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : histo(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("11", DOF_1)

        # xi2 : inter(xi1)-histo(xi2)-inter(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("12", DOF_2)

        # xi3 : inter(xi1)-inter(xi2)-histo(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("13", DOF_3)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def transpose_Z20_dot(self, x):
        """
        Matrix-vector product Z20.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^2}

        Returns
        ----------
            res : np.array
                dim R{N^1}

        Notes
        -----
            Z20.x = I_1( R_1 ( F_Z20.x))

            Z20.T.x = F_Z20.T( R_1.T ( I_1.T.x))

            See Z20_dot for more details.
        """

        # x dim check
        # x should be R{N^1}
        assert len(x) == self.space.Ntot_1form_cum[-1]
        x_loc = list(self.space.extract_1(x))

        assert x_loc[0].shape == (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        assert x_loc[1].shape == (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        assert x_loc[2].shape == (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # step1 : I.T(x)
        # xi1 : transpose of histo(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.D_1.T, self.N_2.T, self.N_3.T], x_loc[0])

        # xi2 : transpose of inter(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.D_2.T, self.N_3.T], x_loc[1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-histo(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.D_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.space.projectors.dofs_T("11", mat_dofs_1)

        # xi2
        mat_f_2 = self.space.projectors.dofs_T("12", mat_dofs_2)

        # xi3
        mat_f_3 = self.space.projectors.dofs_T("13", mat_dofs_3)

        # step3 : F.T( R.T( I.T(x) ) )
        mat_f_11_c = mat_f_1 * self.g_11[0, 0] / self.det_df_11
        mat_f_12_c = mat_f_1 * self.g_11[0, 1] / self.det_df_11
        mat_f_13_c = mat_f_1 * self.g_11[0, 2] / self.det_df_11
        mat_f_21_c = mat_f_2 * self.g_12[1, 0] / self.det_df_12
        mat_f_22_c = mat_f_2 * self.g_12[1, 1] / self.det_df_12
        mat_f_23_c = mat_f_2 * self.g_12[1, 2] / self.det_df_12
        mat_f_31_c = mat_f_3 * self.g_13[2, 0] / self.det_df_13
        mat_f_32_c = mat_f_3 * self.g_13[2, 1] / self.det_df_13
        mat_f_33_c = mat_f_3 * self.g_13[2, 2] / self.det_df_13

        res_11 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts0_D_3.T], mat_f_11_c)  # 0
        res_12 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts0_D_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_D_2.T, self.pts0_N_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts0_D_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_22_c)  # 0
        res_23 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_D_2.T, self.pts1_D_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_33_c)  # 0

        res_1 = res_11 + res_21 + res_31
        res_2 = res_12 + res_22 + res_32
        res_3 = res_13 + res_23 + res_33

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def Y20_dot(self, x):
        """
        Matrix-vector product Y20.x

        Parameters
        ----------
            x : np.array
                dim R^{N^0}

        Returns
        ----------
            res : np.array
                dim R^{N^3}

        Notes
        -----
            Y20 = pi_3[g_sqrt * lambda^0]  in   R^{N^3 x N^0}

            Y20.x = I_3( R_3 ( F_Y20.x))

            I_3 ... inverse histopolation matrix (tensor product)

            R_3  ... compute DOFs from function values at point set pts_ijk

            F_Y20[ijk,mno] = g_sqrt(pts_ijk) * lambda^0_mno(pts_ijk)

            * spline evaluation (at V_3 point sets)
                * [his, his, his] points: {quad_pts[0], quad_pts[1], quad_pts[2]}

            * Components of F_Y20:
                * evaluated at [his, his, his] : (N, N, N) * sqrt g
        """

        # x dim check
        # assert len(x) == self.space.Ntot_0form
        x_loc = self.space.extract_0(x)

        # assert x_loc.shape == (self.NbaseD[0], self.NbaseD[1], self.NbaseD[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        mat_f = kron_matvec_3d([self.pts1_N_1, self.pts1_N_2, self.pts1_N_3], x_loc)

        # Point-wise multiplication of mat_f and peq
        mat_f_c = mat_f * self.det_df_3

        # ========== Step 2 : R( F(x) ) ==========#
        # Linear operator : evaluation values at the projection points to the Degree of Freedom of the spline.
        DOF = self.space.projectors.dofs("3", mat_f_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # histo(xi1)-histo(xi2)-histo(xi3)-polation.
        res = self.space.projectors.PI_mat("3", DOF)

        return res.flatten()

    # ====================================================================
    def transpose_Y20_dot(self, x):
        """
        Matrix-vector product Y20.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^3}

        Returns
        ----------
            res : np.array
                dim R{N^0}

        Notes
        -----
            Y20.x = I_3( R_3 ( F_Y20.x))

            Y20.T.x = I_3.T( R_3.T ( F_Y20.T.x))

            See Y20_dot for more details.
        """

        # x dim check
        # assert len(x) == self.space.Ntot_3form
        x_loc = self.space.extract_3(x)

        # assert x_loc.shape == (self.NbaseD[0], self.NbaseD[1], self.NbaseD[2])

        # step1 : I.T(x)
        mat_dofs = kron_solve_3d([self.D_1.T, self.D_2.T, self.D_3.T], x_loc)

        # step2 : R.T( I.T(x) )
        mat_f = self.space.projectors.dofs_T("3", mat_dofs)

        # step3 : F.T( R.T( I.T(x) ) )
        mat_f_c = mat_f * self.det_df_3

        res = kron_matvec_3d([self.pts1_N_1.T, self.pts1_N_2.T, self.pts1_N_3.T], mat_f_c)

        return res.flatten()

    # ====================================================================
    def S20_dot(self, x):
        """
        Matrix-vector product S20.x

        Parameters
        ----------
            x : np.array
                dim R^{N^2}

        Returns
        ----------
            res : np.array
                dim R^{N^1}

        Notes
        -----
            S20     = pi_1[p_eq * G / g_sqrt lambda^2]      in     R^{N^1 x N^2}

            S20.x = I_1( R_1 ( F_S20.x))

            I_1 ... inverse inter/histopolation matrix (tensor product)

            R_1  ... compute DOFs from function values at point set pts_ijk

            F_S20[ijk, mno] = p_eq(pts_ijk) * G(pts_ijk) / g_sqrt(pts_ijk) * lambda^2_mno(pts_ijk)

            * spline evaluation (at V_1 point sets)
                * [his, int, int] points: {quad_pts[0], greville[1], greville[2]}
                * [int, his, int] points: {greville[0], quad_pts[1], greville[2]}
                * [int, int, his] points: {greville[0], greville[1], quad_pts[2]}

            * Components of F_S20:
                * evaluated at [his, int, int] : (N, D, D) * G_11 * p_eq / g_sqrt + (D, N, D) * G_12 * p_eq / g_sqrt + (D, D, N) * G_13 * p_eq / g_sqrt
                * evaluated at [int, his, int] : (N, D, D) * G_21 * p_eq / g_sqrt + (D, N, D) * G_22 * p_eq / g_sqrt + (D, D, N) * G_23 * p_eq / g_sqrt
                * evaluated at [int, int, his] : (N, D, D) * G_31 * p_eq / g_sqrt + (D, N, D) * G_32 * p_eq / g_sqrt + (D, D, N) * G_33 * p_eq / g_sqrt
        """

        # x dim check
        # x should be R{N^2}
        assert len(x) == self.space.Ntot_2form_cum[-1]
        x_loc = list(self.space.extract_2(x))

        assert x_loc[0].shape == (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        assert x_loc[1].shape == (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        assert x_loc[2].shape == (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_11 = kron_matvec_3d([self.pts1_N_1, self.pts0_D_2, self.pts0_D_3], x_loc[0])
        mat_f_12 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts0_D_3], x_loc[1])
        mat_f_13 = kron_matvec_3d([self.pts1_D_1, self.pts0_D_2, self.pts0_N_3], x_loc[2])

        # xi2
        mat_f_21 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts0_D_3], x_loc[0])
        mat_f_22 = kron_matvec_3d([self.pts0_D_1, self.pts1_N_2, self.pts0_D_3], x_loc[1])
        mat_f_23 = kron_matvec_3d([self.pts0_D_1, self.pts1_D_2, self.pts0_N_3], x_loc[2])

        # xi3
        mat_f_31 = kron_matvec_3d([self.pts0_N_1, self.pts0_D_2, self.pts1_D_3], x_loc[0])
        mat_f_32 = kron_matvec_3d([self.pts0_D_1, self.pts0_N_2, self.pts1_D_3], x_loc[1])
        mat_f_33 = kron_matvec_3d([self.pts0_D_1, self.pts0_D_2, self.pts1_N_3], x_loc[2])

        mat_f_11_c = mat_f_11 * self.p0_eq_11 * self.g_11[0, 0] / self.det_df_11
        mat_f_12_c = mat_f_12 * self.p0_eq_11 * self.g_11[0, 1] / self.det_df_11
        mat_f_13_c = mat_f_13 * self.p0_eq_11 * self.g_11[0, 2] / self.det_df_11
        mat_f_21_c = mat_f_21 * self.p0_eq_12 * self.g_12[1, 0] / self.det_df_12
        mat_f_22_c = mat_f_22 * self.p0_eq_12 * self.g_12[1, 1] / self.det_df_12
        mat_f_23_c = mat_f_23 * self.p0_eq_12 * self.g_12[1, 2] / self.det_df_12
        mat_f_31_c = mat_f_31 * self.p0_eq_13 * self.g_13[2, 0] / self.det_df_13
        mat_f_32_c = mat_f_32 * self.p0_eq_13 * self.g_13[2, 1] / self.det_df_13
        mat_f_33_c = mat_f_33 * self.p0_eq_13 * self.g_13[2, 2] / self.det_df_13

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.space.projectors.dofs("11", mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.space.projectors.dofs("12", mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.space.projectors.dofs("13", mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : histo(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.space.projectors.PI_mat("11", DOF_1)

        # xi2 : inter(xi1)-histo(xi2)-inter(xi3)-polation.
        res_2 = self.space.projectors.PI_mat("12", DOF_2)

        # xi3 : inter(xi1)-inter(xi2)-histo(xi3)-polation.
        res_3 = self.space.projectors.PI_mat("13", DOF_3)

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))

    # ====================================================================
    def transpose_S20_dot(self, x):
        """
        Matrix-vector product S20.T.x

        Parameters
        ----------
            x : np.array
                dim R{N^1}

        Returns
        ----------
            res : np.array
                dim R{N^2}

        Notes
        -----
            S20.x = I_1( R_1 ( F_S20.x))

            S20.T dot x = F_S20.T( R_1.T ( I_1.T(x)))

            See S20_dot for more details.
        """

        # x dim check
        # x should be R{N^1}
        assert len(x) == self.space.Ntot_1form_cum[-1]
        x_loc = list(self.space.extract_1(x))

        assert x_loc[0].shape == (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        assert x_loc[1].shape == (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        assert x_loc[2].shape == (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # step1 : I.T(x)
        # xi1 : transpose of histo(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.D_1.T, self.N_2.T, self.N_3.T], x_loc[0])

        # xi2 : transpose of inter(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.D_2.T, self.N_3.T], x_loc[1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-histo(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.D_3.T], x_loc[2])

        # step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.space.projectors.dofs_T("11", mat_dofs_1)

        # xi2
        mat_f_2 = self.space.projectors.dofs_T("12", mat_dofs_2)

        # xi3
        mat_f_3 = self.space.projectors.dofs_T("13", mat_dofs_3)

        # step3 : F.T( R.T( I.T(x) ) )
        mat_f_11_c = mat_f_1 * self.p0_eq_11 * self.g_11[0, 0] / self.det_df_11
        mat_f_12_c = mat_f_1 * self.p0_eq_11 * self.g_11[0, 1] / self.det_df_11
        mat_f_13_c = mat_f_1 * self.p0_eq_11 * self.g_11[0, 2] / self.det_df_11
        mat_f_21_c = mat_f_2 * self.p0_eq_12 * self.g_12[1, 0] / self.det_df_12
        mat_f_22_c = mat_f_2 * self.p0_eq_12 * self.g_12[1, 1] / self.det_df_12
        mat_f_23_c = mat_f_2 * self.p0_eq_12 * self.g_12[1, 2] / self.det_df_12
        mat_f_31_c = mat_f_3 * self.p0_eq_13 * self.g_13[2, 0] / self.det_df_13
        mat_f_32_c = mat_f_3 * self.p0_eq_13 * self.g_13[2, 1] / self.det_df_13
        mat_f_33_c = mat_f_3 * self.p0_eq_13 * self.g_13[2, 2] / self.det_df_13

        res_11 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts0_D_3.T], mat_f_11_c)  # 0
        res_12 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts0_D_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_D_2.T, self.pts0_N_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts0_D_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_22_c)  # 0
        res_23 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_D_2.T, self.pts1_D_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_33_c)  # 0

        res_1 = res_11 + res_21 + res_31
        res_2 = res_12 + res_22 + res_32
        res_3 = res_13 + res_23 + res_33

        return np.concatenate((res_1.flatten(), res_2.flatten(), res_3.flatten()))
