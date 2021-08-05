import numpy              as np

from hylife.utilitis_FEEC.projectors.projectors_global import projectors_tensor_3d
from hylife.linear_algebra.linalg_kron import kron_matvec_3d, kron_solve_3d

import simulations.template_slab.input_run.equilibrium_MHD    as eq_mhd

# =================================================================================================
class projectors_dot_x:
    '''
    Caclulate the product of vector 'x' with the several kinds of projection matrices.
    Global projectors based on tensor product are used.

    List of projection matrices :                      dim of matrices           verification method
    - PI_1_lambda_2 = pi_1[lambda^2]                    R{N^1 * N^2}               convergence test   
    - PI_2_lambda_1 = pi_2[lambda^1]                    R{N^2 * N^1}               convergence test
    - K             = pi_0[p_eq * lambda^0]             R{N^0 * N^0}                  identity test
    - S             = pi_1[p_eq * lambda^1]             R{N^1 * N^1}                  identity test
    - Q             = pi_2[rho_eq * G_inv * lambda^1]   R{N^2 * N^1}               convergence test
    - P             = pi_1[j_eq * lambda^2]             R{N^1 * N^2}               convergence test   
    - W             = pi_1[rho_eq / g_sqrt * lambda^1]  R{N^1 * N^1}                  identity test       
    - T             = pi_1[B_eq * G_inv * lambda^1]     R{N^1 * N^1}               convergence test               
    - PP            = pi_0[DF^{-T} * lambda^1]          R{N^0 * 3 * N^1}  for PC   convergence test

    2 form formulation
    - Q2            = pi_2[rho_eq / g_sqrt * lambda^2]  R{N^2 * N^2}                  identity test
    - T2            = pi_1[B_eq / q_sqrt * lambda^2]    R{N^1 * N^2}               convergence test
    - S2            = pi_2[G_inv * j_eq * lambda^2]     R{N^2 * N^2}               convergence test
    - P2            = pi_2[p_eq / g_sqrt * lambda^2]    R{N^2 * N^2}                  identity test
    - M             = pi_3[p_eq / g_sqrt * lambda^3]    R{N^3 * N^3}                  identity test 

    Parameters :
    ---------------------------
    proj_list : list of 1d projector objects
    domain :    domain object

    Methods :
    ---------------------------
    PI1_lambda2_dot(x)
    PI2_lambda1_dot(x)
    K_dot(x)
    transpose_K_dot(x)
    S_dot(x)
    transpose_S_dot(x)
    Q_dot(x)
    transpose_W_dot(x)
    P_dot(x)
    transpose_P_dot(x)
    W_dot(x)
    transpose_W_dot(x)
    T_dot(x)
    transpose_T_dot(x)
    PP_dot(x)
    transpose_PP_dot(x)
    Q2_dot(x)
    transpose_Q2_dot(x)
    T2_dot(x)
    transpose_T2_dot(x)
    S2_dot(x)
    transpose_S2_dot(x)
    P2_dot(x)
    transpose_P2_dot(x)
    M_dot(x)
    transpose_M_dot(x)

    '''

    def __init__(self, proj_list, domain):

        # 1d projectors
        self.proj_eta1 = proj_list[0]
        self.proj_eta2 = proj_list[1]
        self.proj_eta3 = proj_list[2]

        self.NbaseN = [proj_list[0].space.NbaseN, proj_list[1].space.NbaseN, proj_list[2].space.NbaseN]
        self.NbaseD = [proj_list[0].space.NbaseD, proj_list[1].space.NbaseD, proj_list[2].space.NbaseD]

        # 3d tensor projectors
        self.proj = projectors_tensor_3d([self.proj_eta1, self.proj_eta2, self.proj_eta3])

        # Interpolation matrices
        self.N_1= self.proj_eta1.N
        self.N_2= self.proj_eta2.N
        self.N_3= self.proj_eta3.N

        # Histopolation matrices
        self.D_1= self.proj_eta1.D
        self.D_2= self.proj_eta2.D
        self.D_3= self.proj_eta3.D

        # Collocation matrices for different point sets
        self.pts0_N_1, self.pts0_D_1, self.pts1_N_1, self.pts1_D_1 = self.proj_eta1.bases_at_pts()
        self.pts0_N_2, self.pts0_D_2, self.pts1_N_2, self.pts1_D_2 = self.proj_eta2.bases_at_pts()
        self.pts0_N_3, self.pts0_D_3, self.pts1_N_3, self.pts1_D_3 = self.proj_eta3.bases_at_pts()

        assert np.allclose(self.N_1.toarray(), self.pts0_N_1.toarray(), atol=1e-14)
        assert np.allclose(self.N_2.toarray(), self.pts0_N_2.toarray(), atol=1e-14)
        assert np.allclose(self.N_3.toarray(), self.pts0_N_3.toarray(), atol=1e-14)
        
        # domain
        self.domain    =  domain

        # call equilibrium_mhd function
        self.equilibrium_mhd = eq_mhd.equilibrium_mhd(domain)

        self.p_eq_fun = lambda xi1, xi2, xi3 : self.equilibrium_mhd.p_eq(xi1, xi2, xi3)
        self.b_eq_fun = lambda xi1, xi2, xi3 : self.equilibrium_mhd.b_eq(xi1, xi2, xi3)
        self.rho_eq_fun = lambda xi1, xi2, xi3 : self.equilibrium_mhd.r_eq(xi1, xi2, xi3)
        self.b_eq_x_fun = lambda xi1, xi2, xi3 : self.equilibrium_mhd.b_eq_x(xi1, xi2, xi3)
        self.b_eq_y_fun = lambda xi1, xi2, xi3 : self.equilibrium_mhd.b_eq_y(xi1, xi2, xi3)
        self.b_eq_z_fun = lambda xi1, xi2, xi3 : self.equilibrium_mhd.b_eq_z(xi1, xi2, xi3)
        self.j_eq_x_fun = lambda xi1, xi2, xi3 : self.equilibrium_mhd.j_eq_x(xi1, xi2, xi3)
        self.j_eq_y_fun = lambda xi1, xi2, xi3 : self.equilibrium_mhd.j_eq_y(xi1, xi2, xi3)
        self.j_eq_z_fun = lambda xi1, xi2, xi3 : self.equilibrium_mhd.j_eq_z(xi1, xi2, xi3)

    # ====================================================================
    def PI1_lambda2_dot(self, x):
        '''
        Calculate the dot product of projection matrix pi_1(Lambda^2) with x
        PI1_lambda2 = pi_1[lambda^2]     R{N^1 * N^2}
        
        PI1_lambda2 dot x = I_1( R_1 ( lambda^2(x)))

        lambda^2[ijk, mno] = lambda^2_mno(pts_ijk)
    
        lambda^2          : xi1 : {N, D, D}
                            xi2 : {D, N, D}
                            xi3 : {D, D, N} 
        Evaluation points : xi1 : {quad_pts[0], greville[1], greville[2]}
                            xi2 : {greville[0], quad_pts[1], quad_pts[2]}
                            xi3 : {greville[0], greville[1], quad_pts[2]}

        # The following blocks are needed to be computed:
        xi1: [his, int, int] : (N, D, D), 
                               (D, N, D), 
                               (D, D, N) 
        xi2: [int, his, int] : (N, D, D), 
                               (D, N, D), 
                               (D, D, N) 
        xi3: [int, int, his] : (N, D, D),
                               (D, N, D),  
                               (D, D, N)

        Parameters
        ----------
        x : list of three np.arrays
            list contains V1 finite element coefficients, either as 3d matrix or flattened 1d vector.
            x[0].size = dim V2_1
            x[1].size = dim V2_2
            x[2].size = dim V2_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^1}
        '''    
        # x dim check
        # x should be R{N^2}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]) 

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_1 = kron_matvec_3d([self.pts1_N_1, self.pts0_D_2, self.pts0_D_3], x_loc[0])

        # xi2
        mat_f_2 = kron_matvec_3d([self.pts0_D_1, self.pts1_N_2, self.pts0_D_3], x_loc[1])

        # xi3
        mat_f_3 = kron_matvec_3d([self.pts0_D_1, self.pts0_D_2, self.pts1_N_3], x_loc[2])


        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.proj.dofs('11', mat_f_1)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.proj.dofs('12', mat_f_2)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.proj.dofs('13', mat_f_3)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : histo(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.proj.PI_mat('11', DOF_1)

        # xi2 : inter(xi1)-histo(xi2)-inter(xi3)-polation.
        res_2 = self.proj.PI_mat('12', DOF_2)

        # xi3 : inter(xi1)-inter(xi2)-histo(xi3)-polation.
        res_3 = self.proj.PI_mat('13', DOF_3)

        return [res_1, res_2, res_3]

    # ====================================================================
    def PI2_lambda1_dot(self, x):
        '''
        Calculate the dot product of projection matrix pi_2(Lambda^1) with x
        PI2_lambda1 = pi_2[lambda^1]     R{N^2 * N^1}
        
        PI2_lambda1 dot x = I_2( R_2 ( lambda^1(x)))

        lambda^1[ijk, mno] = lambda^1_mno(pts_ijk)

        # spline evaluation
        lambda^1          : xi1 : {D, N, N}
                            xi2 : {N, D, N}
                            xi3 : {N, N, D} 
        Evaluation points : xi1 : {greville[0], quad_pts[1], quad_pts[2]}
                            xi2 : {quad_pts[0], greville[1], quad_pts[2]}
                            xi2 : {quad_pts[0], quad_pts[1], greville[2]}

        # The following blocks need to be computed:
        xi1: [int, his, his] : (D, N, N), 
                               (N, D, N), 
                               (N, N, D)
        xi2: [his, int, his] : (D, N, N), 
                               (N, D, N), 
                               (N, N, D)
        xi3: [his, his, int] : (D, N, N), 
                               (N, D, N), 
                               (N, N, D)

        Parameters
        ----------
        x : list of three np.arrays
            list contains V1 finite element coefficients, either as 3d matrix or flattened 1d vector.
            x[0].size = dim V1_1
            x[1].size = dim V1_2
            x[2].size = dim V1_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^2}
        '''
        # x dim check
        # x should be R{N^1}
        x_loc = [None, None, None]

        if x[0].shape[0] ==  (self.NbaseD[0] * self.NbaseN[1] * self.NbaseN[2]):
            x_loc[0] = x[0].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc[0] = x[0]
        assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])

        if x[1].shape[0] ==  self.NbaseN[0] * self.NbaseD[1] * self.NbaseN[2]:
            x_loc[1] = x[1].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[1] = x[1]
        assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])

        if x[2].shape[0] ==  self.NbaseN[0] * self.NbaseN[1] * self.NbaseD[2]:
            x_loc[2] = x[2].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[2] = x[2]
        assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])  

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_1 = kron_matvec_3d([self.pts0_D_1, self.pts1_N_2, self.pts1_N_3], x_loc[0])

        # xi2
        mat_f_2 = kron_matvec_3d([self.pts1_N_1, self.pts0_D_2, self.pts1_N_3], x_loc[1])

        # xi3
        mat_f_3 = kron_matvec_3d([self.pts1_N_1, self.pts1_N_2, self.pts0_D_3], x_loc[2])

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.proj.dofs('21', mat_f_1)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.proj.dofs('22', mat_f_2)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.proj.dofs('23', mat_f_3)


        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : inter(xi1)-histo(xi2)-histo(xi3)-polation.
        res_1 = self.proj.PI_mat('21', DOF_1)

        # xi2 : histo(xi1)-inter(xi2)-histo(xi3)-polation.
        res_2 = self.proj.PI_mat('22', DOF_2)

        # xi3 : histo(xi1)-histo(xi2)-inter(xi3)-polation.
        res_3 = self.proj.PI_mat('23', DOF_3)

        return [res_1, res_2, res_3]


    # ====================================================================
    def K_dot(self, x):
        '''
        Calculate the dot product of projection matrix K with x
        K = pi_0[p_eq * lambda^0]     R{N^0 * N^0}

        K dot x = I_0( R_0 ( F_K(x)))
        
        F_K[ijk,mno] = p_eq(pts_ijk) * lambda^0_mno(pts_ijk)

        # spline evaluation
        lambda^0          : {N, N, N} 
        Evaulation points : {greville[0], greville[1], greville[2]}

        Parameters
        ----------
        x : np.array
            V0 finite element coefficients, either 3d matrix or flattened 1d vector.
            x.size = dim V0

        Returns
        ----------
        res : 3d array     R{N^0}
        '''

        # x dim check
        if len(x[0].shape) == 1:
            x_loc = x.reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc = x

        assert x_loc.shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.

        mat_f = kron_matvec_3d([self.pts0_N_1, self.pts0_N_2, self.pts0_N_3], x_loc)
        
        # p_eq at the projeciton points
        p_eq = self.proj.eval_for_PI('0', self.p_eq_fun)

        # Point-wise multiplication of mat_f and peq
        mat_f_c = mat_f * p_eq

        # ========== Step 2 : R( F(x) ) ==========#
        # Linear operator : evaluation values at the projection points to the Degree of Freedom of the spline.
        DOF = self.proj.dofs('0', mat_f_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # inter(xi1)-inter(xi2)-inter(xi3)-polation.
        res = self.proj.PI_mat('0', DOF)

        return res

    # ====================================================================
    def transpose_K_dot(self, x):
        '''
        Calculate the dot product of projection matrix K with x
        K = pi_0[p_eq * lambda^0]     R{N^0 * N^0}

        K.T dot x = F_K.T( R_0.T ( I_0.T(x)))
        
        F_K[ijk,mno] = p_eq(pts_ijk) * lambda^0_mno(pts_ijk)

        Parameters
        ----------
        x : np.array
            3d matrix or flattened 1d vector.
            x.size = dim V0

        Returns
        ----------
        res : 3d array     R{N^0}
        '''

        # x dim check
        if len(x[0].shape) == 1:
            x_loc = x.reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc = x

        assert x_loc.shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])

        # step1 : I.T(x)
        mat_dofs = kron_solve_3d([self.N_1.T, self.N_2.T, self.N_3.T], x_loc)

        #step2 : R.T( I.T(x) )
        mat_f = self.proj.dofs_T('0', mat_dofs)

        #step3 : F.T( R.T( I.T(x) ) )
        # p_eq at the projeciton points
        p_eq = self.proj.eval_for_PI('0', self.p_eq_fun)

        mat_f_c = p_eq * mat_f

        res = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_c)

        return res

    # ====================================================================
    def S_dot(self, x):
        '''
        Calculate the dot product of projection matrix S with x
        S = pi_1[p_eq * lambda^1]     R{N^1 * N^1}

        S dot x = I_1( R_1 ( F_S(x))) 

        F_S[ijk, mno] = p_eq(pts_ijk) * lambda^1_mno(pts_ijk)

        # spline evaluation
        lambda^1          : xi1 : {D, N, N}
                            xi2 : {N, D, N}
                            xi3 : {N, N, D} 
        Evaluation points : xi1 : {quad_pts[0], greville[1], greville[2]}
                            xi2 : {greville[0], quad_pts[1], greville[2]}
                            xi2 : {greville[0], greville[1], quad_pts[2]}

        # The following blocks need to be computed:
        xi1: [his, int, int] : (D, N, N) * p_eq, 0, 0
        xi2: [int, his, int] : 0, (N, D, N) * p_eq, 0
        xi3: [int, int, his] : 0, 0, (N, N, D) * p_eq

        Parameters
        ----------
        x : list of three np.arrays
            list contains V1 finite element coefficients, either as 3d matrix or flattened 1d vector.
            x[0].size = dim V1_1
            x[1].size = dim V1_2
            x[2].size = dim V1_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^1}
        '''

        # x dim check
        # x should be R{N^1}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]) 

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        mat_f_1 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts0_N_3], x_loc[0])
        mat_f_2 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts0_N_3], x_loc[1])
        mat_f_3 = kron_matvec_3d([self.pts0_N_1, self.pts0_N_2, self.pts1_D_3], x_loc[2])
 
        # p_eq at the projeciton points
        p_eq_1 = self.proj.eval_for_PI('11', self.p_eq_fun)
        p_eq_2 = self.proj.eval_for_PI('12', self.p_eq_fun)
        p_eq_3 = self.proj.eval_for_PI('13', self.p_eq_fun)

        # function at the projection points
        mat_f_1_c = mat_f_1 * p_eq_1
        mat_f_2_c = mat_f_2 * p_eq_2
        mat_f_3_c = mat_f_3 * p_eq_3

        # ========== Step 2 : Computation of degrees of freedom R( F(x) ) ==========#
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_c{i,m,j,k}
        DOF_1 = self.proj.dofs('11', mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_c{i,j,m,k}
        DOF_2 = self.proj.dofs('12', mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_c{i,j,k,m}
        DOF_3 = self.proj.dofs('13', mat_f_3_c)


        # ========== Step 3 : Projection I( R( F(x) ) ) ==========#
        # xi1 : histo(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.proj.PI_mat('11', DOF_1)

        # xi2 : inter(xi1)-histo(xi2)-inter(xi3)-polation.
        res_2 = self.proj.PI_mat('12', DOF_2)

        # xi3 : inter(xi1)-inter(xi2)-histo(xi3)-polation.
        res_3 = self.proj.PI_mat('13', DOF_3)


        return [res_1, res_2, res_3]

    # ====================================================================
    def transpose_S_dot(self, x):
        '''
        Calculate the dot product of transpose of projection matrix S with x
        S = pi_1[p_eq * lambda^1]     R{N^1 * N^1}

        S.T dot x = F_S.T( R_1.T ( I_1.T(x)))

        Parameters
        ----------
        x : list of three np.arrays
            list contains 3d matrix or flattened 1d vector.
            x[0].size = dim V1_1
            x[1].size = dim V1_2
            x[2].size = dim V1_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^1}
        '''
        # x dim check
        # x should be R{N^1}
        x_loc = [None, None, None]

        if x[0].shape[0] ==  (self.NbaseD[0] * self.NbaseN[1] * self.NbaseN[2]):
            x_loc[0] = x[0].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc[0] = x[0]
        assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])

        if x[1].shape[0] ==  self.NbaseN[0] * self.NbaseD[1] * self.NbaseN[2]:
            x_loc[1] = x[1].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[1] = x[1]
        assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])

        if x[2].shape[0] ==  self.NbaseN[0] * self.NbaseN[1] * self.NbaseD[2]:
            x_loc[2] = x[2].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[2] = x[2]
        assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])

        # step1 : I.T(x)
        # xi1 : transpose of histo(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.D_1.T, self.N_2.T, self.N_3.T], x_loc [0])

        # xi2 : transpose of inter(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.D_2.T, self.N_3.T], x_loc [1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-histo(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.D_3.T], x_loc [2])


        #step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1 : mat_f_1_{i,m,j,k} = w_{i,m} * DOF_1_{i,j,k}
        mat_f_1 = self.proj.dofs_T('11', mat_dofs_1)

        # xi2 : mat_f_2_{i,j,m,k} = w_{j,m} * DOF_2_{i,j,k}
        mat_f_2 = self.proj.dofs_T('12', mat_dofs_2)

        # xi3 : mat_f_2_{i,j,k,m} = w_{k,m} * DOF_3_{i,j,k}
        mat_f_3 = self.proj.dofs_T('13', mat_dofs_3)


        #step3 : F.T( R.T( I.T(x) ) )
        # p_eq at the projeciton points
        p_eq_1 = self.proj.eval_for_PI('11', self.p_eq_fun)
        p_eq_2 = self.proj.eval_for_PI('12', self.p_eq_fun)
        p_eq_3 = self.proj.eval_for_PI('13', self.p_eq_fun)

        mat_f_1_c = p_eq_1 * mat_f_1
        mat_f_2_c = p_eq_2 * mat_f_2
        mat_f_3_c = p_eq_3 * mat_f_3

        res_1 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_1_c)
        res_2 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_2_c)
        res_3 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_3_c)

        return [res_1, res_2, res_3]

    # ====================================================================
    def Q_dot(self, x):
        '''
        Calculate the dot product of projection matrix Q with x
        Q     = pi_2[rho_eq * G_inv * lambda^1]     R{N^2 * N^1}
        G_inv = (DF)^(-1)*DF^(-T)

        Q dot x = I_2( R_2 ( F_Q(x)))

        F_Q[ijk, mno] = rho_eq(pts_ijk) * G_inv(pts_ijk) * lambda^1_mno(pts_ijk)   

        # spline evaluation
        lambda^1          : xi1 : {D, N, N}
                            xi2 : {N, D, N}
                            xi3 : {N, N, D} 
        Evaluation points : xi1 : {greville[0], quad_pts[1], quad_pts[2]}
                            xi2 : {quad_pts[0], greville[1], quad_pts[2]}
                            xi2 : {quad_pts[0], quad_pts[1], greville[2]}

        # The following blocks need to be computed:
        xi1: [int, his, his] : (D, N, N) * G_inv_11 * rho_eq, 
                               (N, D, N) * G_inv_12 * rho_eq, 
                               (N, N, D) * G_inv_13 * rho_eq
        xi2: [his, int, his] : (D, N, N) * G_inv_21 * rho_eq, 
                               (N, D, N) * G_inv_22 * rho_eq, 
                               (N, N, D) * G_inv_23 * rho_eq
        xi3: [his, his, int] : (D, N, N) * G_inv_31 * rho_eq, 
                               (N, D, N) * G_inv_32 * rho_eq, 
                               (N, N, D) * G_inv_33 * rho_eq

        Parameters
        ----------
        x : list of three np.arrays
            list contains V1 finite element coefficients, either as 3d matrix or flattened 1d vector.
            x[0].size = dim V1_1
            x[1].size = dim V1_2
            x[2].size = dim V1_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^2}
        '''
        # x dim check
        # x should be R{N^1}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])  

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

        # rho_eq at the projeciton points
        rho_eq_1 = self.proj.eval_for_PI('21', self.rho_eq_fun)
        rho_eq_2 = self.proj.eval_for_PI('22', self.rho_eq_fun)
        rho_eq_3 = self.proj.eval_for_PI('23', self.rho_eq_fun)

        # g_inv at the projection points
        pts_PI_1 = self.proj.pts_PI['21']
        pts_PI_2 = self.proj.pts_PI['22']
        pts_PI_3 = self.proj.pts_PI['23']

        g_inv_11 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_11')
        g_inv_12 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_12')
        g_inv_13 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_13')
        g_inv_21 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_21')
        g_inv_22 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_22')
        g_inv_23 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_23')
        g_inv_31 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_31')
        g_inv_32 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_32')
        g_inv_33 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_33')

        mat_f_11_c = mat_f_11 * rho_eq_1 * g_inv_11
        mat_f_12_c = mat_f_12 * rho_eq_1 * g_inv_12
        mat_f_13_c = mat_f_13 * rho_eq_1 * g_inv_13
        mat_f_21_c = mat_f_21 * rho_eq_2 * g_inv_21
        mat_f_22_c = mat_f_22 * rho_eq_2 * g_inv_22
        mat_f_23_c = mat_f_23 * rho_eq_2 * g_inv_23
        mat_f_31_c = mat_f_31 * rho_eq_3 * g_inv_31
        mat_f_32_c = mat_f_32 * rho_eq_3 * g_inv_32
        mat_f_33_c = mat_f_33 * rho_eq_3 * g_inv_33

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.proj.dofs('21', mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.proj.dofs('22', mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.proj.dofs('23', mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : inter(xi1)-histo(xi2)-histo(xi3)-polation.
        res_1 = self.proj.PI_mat('21', DOF_1)

        # xi2 : histo(xi1)-inter(xi2)-histo(xi3)-polation.
        res_2 = self.proj.PI_mat('22', DOF_2)

        # xi3 : histo(xi1)-histo(xi2)-inter(xi3)-polation.
        res_3 = self.proj.PI_mat('23', DOF_3)

        return [res_1, res_2, res_3]


    # ====================================================================
    def transpose_Q_dot(self, x):
        '''
        Calculate the dot product of projection matrix Q with x
        Q     = pi_2[rho_eq * G_inv * lambda^1]     R{N^2 * N^1}
        G_inv = (DF)^(-1)*DF^(-T)

        Q.T dot x = F_Q.T( R_2.T ( I_2.T(x)))

        Parameters
        ----------
        x : list of three np.arrays
            list contains 3d matrix or flattened 1d vector.
            x[0].size = dim V2_1
            x[1].size = dim V2_2
            x[2].size = dim V2_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^1}
        '''

        # x dim check
        # x should be R{N^2}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[2] = x[2]
            
        assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]) 
 
        # step1 : I.T(x)
        # xi1 : transpose of inter(xi1)-histo(xi2)-histo(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.N_1.T, self.D_2.T, self.D_3.T], x_loc[0])

        # xi2 : transpose of histo(xi1)-inter(xi2)-histo(xi3)-polati on.
        mat_dofs_2 = kron_solve_3d([self.D_1.T, self.N_2.T, self.D_3.T], x_loc[1])

        # xi3 : transpose of histo(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.D_1.T, self.D_2.T, self.N_3.T], x_loc[2])


        #step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.proj.dofs_T('21', mat_dofs_1)

        # xi2
        mat_f_2 = self.proj.dofs_T('22', mat_dofs_2)

        # xi3
        mat_f_3 = self.proj.dofs_T('23', mat_dofs_3)         

        #step3 : F.T( R.T( I.T(x) ) )
        # rho_eq at the projeciton points
        rho_eq_1 = self.proj.eval_for_PI('21', self.rho_eq_fun)
        rho_eq_2 = self.proj.eval_for_PI('22', self.rho_eq_fun)
        rho_eq_3 = self.proj.eval_for_PI('23', self.rho_eq_fun)

        # g_inv at the projection points
        pts_PI_1 = self.proj.pts_PI['21']
        pts_PI_2 = self.proj.pts_PI['22']
        pts_PI_3 = self.proj.pts_PI['23']

        g_inv_11 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_11')
        g_inv_12 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_12')
        g_inv_13 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_13')
        g_inv_21 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_21')
        g_inv_22 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_22')
        g_inv_23 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_23')
        g_inv_31 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_31')
        g_inv_32 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_32')
        g_inv_33 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_33')

        mat_f_11_c = mat_f_1 * rho_eq_1 * g_inv_11
        mat_f_12_c = mat_f_1 * rho_eq_1 * g_inv_12
        mat_f_13_c = mat_f_1 * rho_eq_1 * g_inv_13
        mat_f_21_c = mat_f_2 * rho_eq_2 * g_inv_21
        mat_f_22_c = mat_f_2 * rho_eq_2 * g_inv_22
        mat_f_23_c = mat_f_2 * rho_eq_2 * g_inv_23
        mat_f_31_c = mat_f_3 * rho_eq_3 * g_inv_31
        mat_f_32_c = mat_f_3 * rho_eq_3 * g_inv_32
        mat_f_33_c = mat_f_3 * rho_eq_3 * g_inv_33

        res_11 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts1_N_3.T], mat_f_11_c)
        res_12 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts1_N_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts1_N_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_22_c)
        res_23 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts1_N_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts1_N_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts1_N_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_33_c)

        return [res_11 + res_12 + res_13, res_21 + res_22 + res_23, res_31 + res_32 + res_33]


# ====================================================================
    def P_dot(self, x):
        '''
        Calculate the dot product of projection matrix P with x
        P     = pi_1[j_eq * lambda^2]     R{N^1 * N^2}
        j_eq  = (    0  , -j_eq_z,  j_eq_y)
                ( j_eq_z,     0  , -j_eq_x)
                (-j_eq_y,  j_eq_x,     0  )

        P dot x = I_1( R_1 ( F_P( x )))

        F_P[ijk, mno] = j_eq(pts_ijk) * lambda^2_mno(pts_ijk)  

        # spline evaluation
        lambda^2          : xi1 : {N, D, D}
                            xi2 : {D, N, D}
                            xi3 : {D, D, N} 
        Evaluation points : xi1 : {quad_pts[0], greville[1], greville[2]}
                            xi2 : {greville[0], quad_pts[1], quad_pts[2]}
                            xi3 : {greville[0], greville[1], quad_pts[2]}

        # The following blocks need to be computed:
        xi1: [his, int, int] : (N, D, D) *    0    ,
                               (D, N, D) * -j_eq_z ,
                               (D, D, N) *  j_eq_y
        xi2: [int, his, int] : (N, D, D) *  j_eq_z ,
                               (D, N, D) *    0    ,
                               (D, D, N) * -j_eq_x 
        xi3: [int, int, his] : (N, D, D) * -j_eq_y ,
                               (D, N, D) *  j_eq_x ,  
                               (D, D, N) *    0

        Parameters
        ----------
        x : list of three np.arrays
            list contains V1 finite element coefficients, either as 3d matrix or flattened 1d vector.
            x[0].size = dim V2_1
            x[1].size = dim V2_2
            x[2].size = dim V2_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^1}
        '''    
        # x dim check
        # x should be R{N^2}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[2] = x[2]
            
        assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]) 

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_11 = kron_matvec_3d([self.pts1_N_1, self.pts0_D_2, self.pts0_D_3], x_loc[0]) #0
        mat_f_12 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts0_D_3], x_loc[1])
        mat_f_13 = kron_matvec_3d([self.pts1_D_1, self.pts0_D_2, self.pts0_N_3], x_loc[2])

        # xi2
        mat_f_21 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts0_D_3], x_loc[0])
        mat_f_22 = kron_matvec_3d([self.pts0_D_1, self.pts1_N_2, self.pts0_D_3], x_loc[1]) #0
        mat_f_23 = kron_matvec_3d([self.pts0_D_1, self.pts1_D_2, self.pts0_N_3], x_loc[2])

        # xi3
        mat_f_31 = kron_matvec_3d([self.pts0_N_1, self.pts0_D_2, self.pts1_D_3], x_loc[0])
        mat_f_32 = kron_matvec_3d([self.pts0_D_1, self.pts0_N_2, self.pts1_D_3], x_loc[1])
        mat_f_33 = kron_matvec_3d([self.pts0_D_1, self.pts0_D_2, self.pts1_N_3], x_loc[2]) #0

        # j_eq at the projeciton points
        j_eq_x_1 = self.proj.eval_for_PI('11', self.j_eq_x_fun) # not used
        j_eq_x_2 = self.proj.eval_for_PI('12', self.j_eq_x_fun)
        j_eq_x_3 = self.proj.eval_for_PI('13', self.j_eq_x_fun)

        j_eq_y_1 = self.proj.eval_for_PI('11', self.j_eq_y_fun)
        j_eq_y_2 = self.proj.eval_for_PI('12', self.j_eq_y_fun) # not used
        j_eq_y_3 = self.proj.eval_for_PI('13', self.j_eq_y_fun)

        j_eq_z_1 = self.proj.eval_for_PI('11', self.j_eq_z_fun)
        j_eq_z_2 = self.proj.eval_for_PI('12', self.j_eq_z_fun)
        j_eq_z_3 = self.proj.eval_for_PI('13', self.j_eq_z_fun) # not used

        mat_f_11_c = mat_f_11 * 0
        mat_f_12_c = mat_f_12 * (-j_eq_z_1)
        mat_f_13_c = mat_f_13 * ( j_eq_y_1)
        mat_f_21_c = mat_f_21 * ( j_eq_z_2)
        mat_f_22_c = mat_f_22 * 0
        mat_f_23_c = mat_f_23 * (-j_eq_x_2)
        mat_f_31_c = mat_f_31 * (-j_eq_y_3)
        mat_f_32_c = mat_f_32 * ( j_eq_x_3)
        mat_f_33_c = mat_f_33 * 0

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.proj.dofs('11', mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.proj.dofs('12', mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.proj.dofs('13', mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : histo(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.proj.PI_mat('11', DOF_1)

        # xi2 : inter(xi1)-histo(xi2)-inter(xi3)-polation.
        res_2 = self.proj.PI_mat('12', DOF_2)

        # xi3 : inter(xi1)-inter(xi2)-histo(xi3)-polation.
        res_3 = self.proj.PI_mat('13', DOF_3)

        return [res_1, res_2, res_3]

    # ====================================================================
    def transpose_P_dot(self, x):
        '''
        Calculate the dot product of transpose of projection matrix P with x
        P     = pi_1[j_eq * lambda^2]     R{N^1 * N^2}
        j_eq  = (    0  , -j_eq_z,  j_eq_y)
                ( j_eq_z,     0  , -j_eq_x)
                (-j_eq_y,  j_eq_x,     0  )

        P.T dot x = F_P.T( R_1.T ( I_1.T (x)))

        Parameters
        ----------
        x : list of three np.arrays
            list contains 3d matrix or flattened 1d vector.
            x[0].size = dim V1_1
            x[1].size = dim V1_2
            x[2].size = dim V1_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^2}
        '''

        # x dim check
        # x should be R{N^1}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]) 

        # step1 : I.T(x)
        # xi1 : transpose of histo(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.D_1.T, self.N_2.T, self.N_3.T], x_loc[0])

        # xi2 : transpose of inter(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.D_2.T, self.N_3.T], x_loc[1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-histo(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.D_3.T], x_loc[2])


        #step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.proj.dofs_T('11', mat_dofs_1)

        # xi2
        mat_f_2 = self.proj.dofs_T('12', mat_dofs_2)

        # xi3
        mat_f_3 = self.proj.dofs_T('13', mat_dofs_3)         

        #step3 : F.T( R.T( I.T(x) ) )
        # j_eq at the projeciton points
        j_eq_x_1 = self.proj.eval_for_PI('11', self.j_eq_x_fun) # not used
        j_eq_x_2 = self.proj.eval_for_PI('12', self.j_eq_x_fun)
        j_eq_x_3 = self.proj.eval_for_PI('13', self.j_eq_x_fun)

        j_eq_y_1 = self.proj.eval_for_PI('11', self.j_eq_y_fun)
        j_eq_y_2 = self.proj.eval_for_PI('12', self.j_eq_y_fun) # not used
        j_eq_y_3 = self.proj.eval_for_PI('13', self.j_eq_y_fun)

        j_eq_z_1 = self.proj.eval_for_PI('11', self.j_eq_z_fun)
        j_eq_z_2 = self.proj.eval_for_PI('12', self.j_eq_z_fun)
        j_eq_z_3 = self.proj.eval_for_PI('13', self.j_eq_z_fun) # not used

        mat_f_11_c = mat_f_1 * 0
        mat_f_12_c = mat_f_1 * (-j_eq_z_1)
        mat_f_13_c = mat_f_1 * ( j_eq_y_1)
        mat_f_21_c = mat_f_2 * ( j_eq_z_2)
        mat_f_22_c = mat_f_2 * 0
        mat_f_23_c = mat_f_2 * (-j_eq_x_2)
        mat_f_31_c = mat_f_3 * (-j_eq_y_3)
        mat_f_32_c = mat_f_3 * ( j_eq_x_3)
        mat_f_33_c = mat_f_3 * 0

        res_11 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts0_D_3.T], mat_f_11_c) #0
        res_12 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts0_D_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts0_D_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_22_c) #0
        res_23 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_33_c) #0

        return [res_11 + res_12 + res_13, res_21 + res_22 + res_23, res_31 + res_32 + res_33]

    
    # ====================================================================
    def W_dot(self, x):
        '''
        Calculate the dot product of projection matrix W with x
        W = pi_1[rho_eq / g_sqrt * lambda^1]     R{N^1 * N^1}

        W dot x = I_1( R_1 ( F_W(x)))
        F_W[ijk,mno] = rho_eq(pts_ijk) / g_sqrt * lambda^1_mno(pts_ijk)

        #spline evaluation
        lambda^1          : xi1 : {D, N, N}
                            xi2 : {N, D, N}
                            xi3 : {N, N, D} 
        Evaluation points : xi1 : {quad_pts[0], greville[1], greville[2]}
                            xi2 : {greville[0], quad_pts[1], greville[2]}
                            xi2 : {greville[0], greville[1], quad_pts[2]}

        # The following blocks need to be computed:
        xi1: [his, int, int] : (D, N, N) * rho_eq / g_sqrt, 0, 0
        xi2: [int, his, int] : 0, (N, D, N) * rho_eq / g_sqrt, 0
        xi3: [int, int, his] : 0, 0, (N, N, D) * rho_eq / g_sqrt

        Parameters
        ----------
        x : list of three np.arrays
            list contains V1 finite element coefficients, either as 3d matrix or flattened 1d vector.
            x[0].size = dim V1_1
            x[1].size = dim V1_2
            x[2].size = dim V1_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^1}
        '''    
        # x dim check
        # x should be R{N^1}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]) 

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        mat_f_1 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts0_N_3], x_loc[0])
        mat_f_2 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts0_N_3], x_loc[1])
        mat_f_3 = kron_matvec_3d([self.pts0_N_1, self.pts0_N_2, self.pts1_D_3], x_loc[2])
 
        # rho_eq at the projeciton points
        rho_eq_1 = self.proj.eval_for_PI('11', self.rho_eq_fun)
        rho_eq_2 = self.proj.eval_for_PI('12', self.rho_eq_fun)
        rho_eq_3 = self.proj.eval_for_PI('13', self.rho_eq_fun)

        # g_inv at the projection points
        pts_PI_1 = self.proj.pts_PI['11']
        pts_PI_2 = self.proj.pts_PI['12']
        pts_PI_3 = self.proj.pts_PI['13']

        det_df_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'det_df')
        det_df_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'det_df')
        det_df_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'det_df')

        mat_f_1_c = mat_f_1 * rho_eq_1 / det_df_1
        mat_f_2_c = mat_f_2 * rho_eq_2 / det_df_2
        mat_f_3_c = mat_f_3 * rho_eq_3 / det_df_3


        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.proj.dofs('11', mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.proj.dofs('12', mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.proj.dofs('13', mat_f_3_c)


        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : histo(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.proj.PI_mat('11', DOF_1)

        # xi2 : inter(xi1)-histo(xi2)-inter(xi3)-polation.
        res_2 = self.proj.PI_mat('12', DOF_2)

        # xi3 : inter(xi1)-inter(xi2)-histo(xi3)-polation.
        res_3 = self.proj.PI_mat('13', DOF_3)


        return [res_1, res_2, res_3]

    # ====================================================================
    def transpose_W_dot(self, x):
        '''
        Calculate the dot product of transpose of projection matrix W with x
        W = pi_1[rho_eq / g_sqrt * lambda^1]     R{N^1 * N^1}

        W.T dot x = F_W.T( R_1.T ( I_1.T(x)))
        F_W[ijk,mno] = rho_eq(pts_ijk) / g_sqrt * lambda^1_mno(pts_ijk)

        Parameters
        ----------
    	x : list of three np.arrays
                  list contains 3d matrix or flattened 1d vector.
                  x[0].size = dim V1_1
                  x[1].size = dim V1_2
                  x[2].size = dim V1_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^1}
        '''
        # x dim check
        # x should be R{N^1}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]) 

        # step1 : I.T(x)
        # xi1 : transpose of histo(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.D_1.T, self.N_2.T, self.N_3.T], x_loc [0])

        # xi2 : transpose of inter(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.D_2.T, self.N_3.T], x_loc [1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-histo(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.D_3.T], x_loc [2])


        #step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1 : mat_f_1_{i,m,j,k} = w_{i,m} * DOF_1_{i,j,k}
        mat_f_1 = self.proj.dofs_T('11', mat_dofs_1)

        # xi2 : mat_f_2_{i,j,m,k} = w_{j,m} * DOF_2_{i,j,k}
        mat_f_2 = self.proj.dofs_T('12', mat_dofs_2)

        # xi3 : mat_f_2_{i,j,k,m} = w_{k,m} * DOF_3_{i,j,k}
        mat_f_3 = self.proj.dofs_T('13', mat_dofs_3)

        # rho_eq at the projeciton points
        rho_eq_1 = self.proj.eval_for_PI('11', self.rho_eq_fun)
        rho_eq_2 = self.proj.eval_for_PI('12', self.rho_eq_fun)
        rho_eq_3 = self.proj.eval_for_PI('13', self.rho_eq_fun)

        # g_inv at the projection points
        pts_PI_1 = self.proj.pts_PI['11']
        pts_PI_2 = self.proj.pts_PI['12']
        pts_PI_3 = self.proj.pts_PI['13']

        det_df_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'det_df')
        det_df_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'det_df')
        det_df_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'det_df')

        mat_f_1_c = mat_f_1 * rho_eq_1 / det_df_1
        mat_f_2_c = mat_f_2 * rho_eq_2 / det_df_2
        mat_f_3_c = mat_f_3 * rho_eq_3 / det_df_3

        res_1 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_1_c)
        res_2 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_2_c)
        res_3 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_3_c)

        return [res_1, res_2, res_3]

    
    # ====================================================================
    def T_dot(self, x):
        '''
        Calculate the dot product of projection matrix T with x
        T     = pi_1[B_eq * G_inv * lambda^1]     R{N^1 * N^1}
        G_inv = (DF)^(-1)*DF^(-T)
        B_eq  = (    0  , -b_eq_z,  b_eq_y)
                ( b_eq_z,     0  , -b_eq_x)
                (-b_eq_y,  b_eq_x,     0  )

        T dot x = I_1( R_1 ( F_T(x)))

        F_T[ijk, mno] = B_eq(pts_ijk) * G_inv(pts_ijk) lambda^1_mno(pts_ijk)

        # spline evaluation
        lambda^1          : xi1 : {D, N, N}
                            xi2 : {N, D, N}
                            xi3 : {N, N, D} 
        Evaluation points : xi1 : {quad_pts[0], greville[1], greville[2]}
                            xi2 : {greville[0], quad_pts[1], greville[2]}
                            xi2 : {greville[0], greville[1], quad_pts[2]}

        # The following blocks need to be computed:
        xi1: [his, int, int] : (D, N, N) * (G_inv_31 * b_eq_y - G_inv_21 * b_eq_z),
                               (N, D, N) * (G_inv_32 * b_eq_y - G_inv_22 * b_eq_z),
                               (N, N, D) * (G_inv_33 * b_eq_y - G_inv_23 * b_eq_z)
        xi2: [int, his, int] : (D, N, N) * (G_inv_11 * b_eq_z - G_inv_31 * b_eq_x),
                               (N, D, N) * (G_inv_12 * b_eq_z - G_inv_32 * b_eq_x),
                               (N, N, D) * (G_inv_13 * b_eq_z - G_inv_33 * b_eq_x)
        xi3: [int, int, his] : (D, N, N) * (G_inv_21 * b_eq_x - G_inv_11 * b_eq_y),
                               (N, D, N) * (G_inv_22 * b_eq_x - G_inv_12 * b_eq_y),  
                               (N, N, D) * (G_inv_23 * b_eq_x - G_inv_13 * b_eq_y)

        Parameters
        ----------
        x : list of three np.arrays
            list contains V1 finite element coefficients, either as 3d matrix or flattened 1d vector.
            x[0].size = dim V1_1
            x[1].size = dim V1_2
            x[2].size = dim V1_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^1}
        '''
        # x dim check
        # x should be R{N^1}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]) 

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

        # b_eq at the projeciton points
        b_eq_x_1 = self.proj.eval_for_PI('11', self.b_eq_x_fun)
        b_eq_x_2 = self.proj.eval_for_PI('12', self.b_eq_x_fun)
        b_eq_x_3 = self.proj.eval_for_PI('13', self.b_eq_x_fun)

        b_eq_y_1 = self.proj.eval_for_PI('11', self.b_eq_y_fun)
        b_eq_y_2 = self.proj.eval_for_PI('12', self.b_eq_y_fun) 
        b_eq_y_3 = self.proj.eval_for_PI('13', self.b_eq_y_fun)

        b_eq_z_1 = self.proj.eval_for_PI('11', self.b_eq_z_fun)
        b_eq_z_2 = self.proj.eval_for_PI('12', self.b_eq_z_fun)
        b_eq_z_3 = self.proj.eval_for_PI('13', self.b_eq_z_fun)

        # g_inv at the projection points
        pts_PI_1 = self.proj.pts_PI['11']
        pts_PI_2 = self.proj.pts_PI['12']
        pts_PI_3 = self.proj.pts_PI['13']

        g_inv_11_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_11')
        g_inv_11_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_11')
        g_inv_12_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_12')
        g_inv_12_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_12')
        g_inv_13_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_13')
        g_inv_13_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_13')
        g_inv_21_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_21')
        g_inv_21_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_21')
        g_inv_22_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_22')
        g_inv_22_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_22')
        g_inv_23_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_23')
        g_inv_23_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_23')
        g_inv_31_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_31')
        g_inv_31_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_31')
        g_inv_32_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_32')
        g_inv_32_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_32')
        g_inv_33_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_33')
        g_inv_33_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_33')

        mat_f_11_c = mat_f_11 * (g_inv_31_1 * b_eq_y_1 - g_inv_21_1 * b_eq_z_1)        
        mat_f_12_c = mat_f_12 * (g_inv_32_1 * b_eq_y_1 - g_inv_22_1 * b_eq_z_1) 
        mat_f_13_c = mat_f_13 * (g_inv_33_1 * b_eq_y_1 - g_inv_23_1 * b_eq_z_1) 

        mat_f_21_c = mat_f_21 * (g_inv_11_2 * b_eq_z_2 - g_inv_31_2 * b_eq_x_2) 
        mat_f_22_c = mat_f_22 * (g_inv_12_2 * b_eq_z_2 - g_inv_32_2 * b_eq_x_2) 
        mat_f_23_c = mat_f_23 * (g_inv_13_2 * b_eq_z_2 - g_inv_33_2 * b_eq_x_2) 

        mat_f_31_c = mat_f_31 * (g_inv_21_3 * b_eq_x_3 - g_inv_11_3 * b_eq_y_3) 
        mat_f_32_c = mat_f_32 * (g_inv_22_3 * b_eq_x_3 - g_inv_12_3 * b_eq_y_3) 
        mat_f_33_c = mat_f_33 * (g_inv_23_3 * b_eq_x_3 - g_inv_13_3 * b_eq_y_3) 

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c


        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.proj.dofs('11', mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.proj.dofs('12', mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.proj.dofs('13', mat_f_3_c)


        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : histo(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.proj.PI_mat('11', DOF_1)

        # xi2 : inter(xi1)-histo(xi2)-inter(xi3)-polation.
        res_2 = self.proj.PI_mat('12', DOF_2)

        # xi3 : inter(xi1)-inter(xi2)-histo(xi3)-polation.
        res_3 = self.proj.PI_mat('13', DOF_3)

        return [res_1, res_2, res_3]


    # ====================================================================
    def transpose_T_dot(self, x):
        '''
        Calculate the dot product of transpose of projection matrix T with x
        T     = pi_1[B_eq * G_inv * lambda^1]     R{N^1 * N^1}
        G_inv = (DF)^(-1)*DF^(-T)
        B_eq  = (    0  , -b_eq_z,  b_eq_y)
                ( b_eq_z,     0  , -b_eq_x)
                (-b_eq_y,  b_eq_x,     0  )

        T.T dot x = F_Q.T( R_2.T ( I_2.T (x)))

        F_T[ijk, mno] = B_eq(pts_ijk) * G_inv(pts_ijk) lambda^1_mno(pts_ijk)

        Parameters
        ----------
    	x : list of three of 3d array which will product with the projection matrices  R{N^1}

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^1}
        '''

        # x dim check
        # x should be R{N^1}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]) 
 
        # step1 : I.T(x)
        # xi1 : transpose of histo(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.D_1.T, self.N_2.T, self.N_3.T], x_loc[0])

        # xi2 : transpose of inter(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.D_2.T, self.N_3.T], x_loc[1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-histo(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.D_3.T], x_loc[2])


        #step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.proj.dofs_T('11', mat_dofs_1)

        # xi2
        mat_f_2 = self.proj.dofs_T('12', mat_dofs_2)

        # xi3
        mat_f_3 = self.proj.dofs_T('13', mat_dofs_3)         

        #step3 : F.T( R.T( I.T(x) ) )
        # b_eq at the projeciton points
        b_eq_x_1 = self.proj.eval_for_PI('11', self.b_eq_x_fun)
        b_eq_x_2 = self.proj.eval_for_PI('12', self.b_eq_x_fun)
        b_eq_x_3 = self.proj.eval_for_PI('13', self.b_eq_x_fun)

        b_eq_y_1 = self.proj.eval_for_PI('11', self.b_eq_y_fun)
        b_eq_y_2 = self.proj.eval_for_PI('12', self.b_eq_y_fun) 
        b_eq_y_3 = self.proj.eval_for_PI('13', self.b_eq_y_fun)

        b_eq_z_1 = self.proj.eval_for_PI('11', self.b_eq_z_fun)
        b_eq_z_2 = self.proj.eval_for_PI('12', self.b_eq_z_fun)
        b_eq_z_3 = self.proj.eval_for_PI('13', self.b_eq_z_fun)

        # g_inv at the projection points
        pts_PI_1 = self.proj.pts_PI['11']
        pts_PI_2 = self.proj.pts_PI['12']
        pts_PI_3 = self.proj.pts_PI['13']

        g_inv_11_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_11')
        g_inv_11_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_11')
        g_inv_12_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_12')
        g_inv_12_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_12')
        g_inv_13_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_13')
        g_inv_13_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_13')
        g_inv_21_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_21')
        g_inv_21_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_21')
        g_inv_22_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_22')
        g_inv_22_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_22')
        g_inv_23_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_23')
        g_inv_23_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_23')
        g_inv_31_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_31')
        g_inv_31_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_31')
        g_inv_32_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_32')
        g_inv_32_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_32')
        g_inv_33_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_33')
        g_inv_33_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_33')

        mat_f_11_c = mat_f_1 * (g_inv_31_1 * b_eq_y_1 - g_inv_21_1 * b_eq_z_1)        
        mat_f_12_c = mat_f_1 * (g_inv_32_1 * b_eq_y_1 - g_inv_22_1 * b_eq_z_1) 
        mat_f_13_c = mat_f_1 * (g_inv_33_1 * b_eq_y_1 - g_inv_23_1 * b_eq_z_1) 

        mat_f_21_c = mat_f_2 * (g_inv_11_2 * b_eq_z_2 - g_inv_31_2 * b_eq_x_2) 
        mat_f_22_c = mat_f_2 * (g_inv_12_2 * b_eq_z_2 - g_inv_32_2 * b_eq_x_2) 
        mat_f_23_c = mat_f_2 * (g_inv_13_2 * b_eq_z_2 - g_inv_33_2 * b_eq_x_2) 

        mat_f_31_c = mat_f_3 * (g_inv_21_3 * b_eq_x_3 - g_inv_11_3 * b_eq_y_3) 
        mat_f_32_c = mat_f_3 * (g_inv_22_3 * b_eq_x_3 - g_inv_12_3 * b_eq_y_3) 
        mat_f_33_c = mat_f_3 * (g_inv_23_3 * b_eq_x_3 - g_inv_13_3 * b_eq_y_3) 

        res_11 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_11_c)
        res_12 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_22_c)
        res_23 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_33_c)

        return [res_11 + res_12 + res_13, res_21 + res_22 + res_23, res_31 + res_32 + res_33]


    # ====================================================================
    def PP_dot(self, x):
        '''
        Calculate the dot product of projection matrix PP with x
        PP = pi_0[DF^{-T} * lambda^1]     R{N^0 * 3 * N^1}

        PP dot x = I_0( R_0 ( F_PP(x)))

        F_PP[ijk, mno] = DF^{-T}(pts_ijk) * lambda^1_mno(pts_ijk)

        # spline evaluation
        lambda^1          : xi1 : {D, N, N}
                            xi2 : {N, D, N}
                            xi3 : {N, N, D} 
        Evaulation points : {greville[0], greville[1], greville[2]}

        # The following blocks need to be computed:
        xi1: [int, int, int] : (D, N, N) * df_inv_11, 
                               (N, D, N) * df_inv_21, 
                               (N, N, D) * df_inv_31
        xi2: [int, int, int] : (D, N, N) * df_inv_12, 
                               (N, D, N) * df_inv_22, 
                               (N, N, D) * df_inv_23
        xi3: [int, int, int] : (D, N, N) * df_inv_13, 
                               (N, D, N) * df_inv_23, 
                               (N, N, D) * df_inv_33

        Parameters
        ----------
        x : list of three np.arrays
            list contains V1 finite element coefficients, either as 3d matrix or flattened 1d vector.
            x[0].size = dim V1_1
            x[1].size = dim V1_2
            x[2].size = dim V1_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^0 * 3}
        '''
        # x dim check
        # x should be R{N^1}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]) 

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_1 = kron_matvec_3d([self.pts0_D_1, self.pts0_N_2, self.pts0_N_3], x_loc[0])
        mat_f_2 = kron_matvec_3d([self.pts0_N_1, self.pts0_D_2, self.pts0_N_3], x_loc[1])
        mat_f_3 = kron_matvec_3d([self.pts0_N_1, self.pts0_N_2, self.pts0_D_3], x_loc[2])

        # df_inv at the projection points
        pts_PI = self.proj.pts_PI['0']

        df_inv_11 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_11')
        df_inv_12 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_12')
        df_inv_13 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_13')
        df_inv_21 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_21')
        df_inv_22 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_22')
        df_inv_23 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_23')
        df_inv_31 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_31')
        df_inv_32 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_32')
        df_inv_33 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_33')

        mat_f_11_c = mat_f_1 * df_inv_11
        mat_f_12_c = mat_f_2 * df_inv_21
        mat_f_13_c = mat_f_3 * df_inv_31
        mat_f_21_c = mat_f_1 * df_inv_12
        mat_f_22_c = mat_f_2 * df_inv_22
        mat_f_23_c = mat_f_3 * df_inv_32
        mat_f_31_c = mat_f_1 * df_inv_13
        mat_f_32_c = mat_f_2 * df_inv_23
        mat_f_33_c = mat_f_3 * df_inv_33

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c


        # ========== Step 2 : R( F(x) ) ==========#
        DOF_1 = self.proj.dofs('0', mat_f_1_c)
        DOF_2 = self.proj.dofs('0', mat_f_2_c)
        DOF_3 = self.proj.dofs('0', mat_f_3_c)


        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # inter(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.proj.PI_mat('0', DOF_1)

        # inter(xi1)-inter(xi2)-inter(xi3)-polation.
        res_2 = self.proj.PI_mat('0', DOF_2)

        # inter(xi1)-inter(xi2)-inter(xi3)-polation.
        res_3 = self.proj.PI_mat('0', DOF_3)

        return [res_1, res_2, res_3]


    # ====================================================================
    def transpose_PP_dot(self, x):
        '''
        Calculate the dot product of transpose of projection matrix PP with x
        PP = pi_0[DF^{-T} * lambda^1]     R{N^0 * 3 * N^1}

        transpose PP dot x = F_PP.T( R_0.T ( I_0.T(x)))
    
        F_PP[ijk, mno] = DF^{-T}(pts_ijk) * lambda^1_mno(pts_ijk)

        Parameters
        ----------
    	x : list of three of 3d array which will product with the projection matrices  R{N^0}

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^1}
        '''

        # x dim check
        # x should be R{3 * N^0}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])  

 
        # step1 : I.T(x)
        # xi1 : transpose of inter(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.N_1.T, self.N_2.T, self.N_3.T], x_loc[0])

        # xi2 : transpose of inter(xi1)-inter(xi2)-inter(xi3)-polati on.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.N_2.T, self.N_3.T], x_loc[1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.N_3.T], x_loc[2])


        #step2 : R.T( I.T(x) )
        mat_f_1 = self.proj.dofs_T('0', mat_dofs_1)
        mat_f_2 = self.proj.dofs_T('0', mat_dofs_2)
        mat_f_3 = self.proj.dofs_T('0', mat_dofs_3)         

        #step3 : F.T( R.T( I.T(x) ) )
        # rho_eq at the projeciton points
        pts_PI = self.proj.pts_PI['0']

        df_inv_11 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_11')
        df_inv_12 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_12')
        df_inv_13 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_13')
        df_inv_21 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_21')
        df_inv_22 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_22')
        df_inv_23 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_23')
        df_inv_31 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_31')
        df_inv_32 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_32')
        df_inv_33 = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'df_inv_33')

        mat_f_11_c = mat_f_1 * df_inv_11
        mat_f_12_c = mat_f_1 * df_inv_21
        mat_f_13_c = mat_f_1 * df_inv_31
        mat_f_21_c = mat_f_2 * df_inv_12
        mat_f_22_c = mat_f_2 * df_inv_22
        mat_f_23_c = mat_f_2 * df_inv_32
        mat_f_31_c = mat_f_3 * df_inv_13
        mat_f_32_c = mat_f_3 * df_inv_23
        mat_f_33_c = mat_f_3 * df_inv_33

        res_11 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_11_c)
        res_12 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_N_2.T, self.pts0_N_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_D_2.T, self.pts0_N_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_D_2.T, self.pts0_N_3.T], mat_f_22_c)
        res_23 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_D_2.T, self.pts0_N_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts0_D_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts0_D_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts0_N_1.T, self.pts0_N_2.T, self.pts0_D_3.T], mat_f_33_c)

        return [res_11 + res_12 + res_13, res_21 + res_22 + res_23, res_31 + res_32 + res_33]


    # ====================================================================
    def Q2_dot(self, x):
        '''
        Calculate the dot product of projection matrix Q2 with x
        Q2 = pi_2[rho_eq / g_sqrt * lambda^2]     R{N^2 * N^2}

        Q2 dot x = I_1( R_1 ( F_Q2(x)))
        F_Q2[ijk,mno] = rho_eq(pts_ijk) / g_sqrt * lambda^2_mno(pts_ijk)

        # Spline evaluation
        lambda^2          : xi1 : {N, D, D}
                            xi2 : {D, N, D}
                            xi3 : {D, D, N} 
        Evaluation points : xi1 : {greville[0], quad_pts[1], quad_pts[2]}
                            xi2 : {quad_pts[0], greville[1], quad_pts[2]}
                            xi2 : {quad_pts[0], quad_pts[1], greville[2]}

        # The following blocks need to be computed:
        xi1: [int, his, his] : (N, D, D) * rho_eq / g_sqrt, 0, 0
        xi2: [his, int, his] : 0, (D, N, D) * rho_eq / g_sqrt, 0
        xi3: [his, his, int] : 0, 0, (D, D, N) * rho_eq / g_sqrt

        Parameters
        ----------
        x : list of three np.arrays
            list contains V1 finite element coefficients, either as 3d matrix or flattened 1d vector.
            x[0].size = dim V2_1
            x[1].size = dim V2_2
            x[2].size = dim V2_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^2}
        '''    
        # x dim check
        # x should be R{N^2}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]) 

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_1 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts1_D_3], x_loc[0])

        # xi2
        mat_f_2 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts1_D_3], x_loc[1])

        # xi3
        mat_f_3 = kron_matvec_3d([self.pts1_D_1, self.pts1_D_2, self.pts0_N_3], x_loc[2])

 
        # rho_eq at the projeciton points
        rho_eq_1 = self.proj.eval_for_PI('21', self.rho_eq_fun)
        rho_eq_2 = self.proj.eval_for_PI('22', self.rho_eq_fun)
        rho_eq_3 = self.proj.eval_for_PI('23', self.rho_eq_fun)

        # g_inv at the projection points
        pts_PI_1 = self.proj.pts_PI['21']
        pts_PI_2 = self.proj.pts_PI['22']
        pts_PI_3 = self.proj.pts_PI['23']

        det_df_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'det_df')
        det_df_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'det_df')
        det_df_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'det_df')

        mat_f_1_c = mat_f_1 * rho_eq_1 / det_df_1
        mat_f_2_c = mat_f_2 * rho_eq_2 / det_df_2
        mat_f_3_c = mat_f_3 * rho_eq_3 / det_df_3


        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.proj.dofs('21', mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.proj.dofs('22', mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.proj.dofs('23', mat_f_3_c)


        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : inter(xi1)-histo(xi2)-histo(xi3)-polation.
        res_1 = self.proj.PI_mat('21', DOF_1)

        # xi2 : histo(xi1)-inter(xi2)-histo(xi3)-polation.
        res_2 = self.proj.PI_mat('22', DOF_2)

        # xi3 : histo(xi1)-histo(xi2)-inter(xi3)-polation.
        res_3 = self.proj.PI_mat('23', DOF_3)


        return [res_1, res_2, res_3]

    # ====================================================================
    def transpose_Q2_dot(self, x):
        '''
        Calculate the dot product of projection matrix Q2 with x
        Q2 = pi_2[rho_eq / g_sqrt * lambda^2]     R{N^2 * N^2}

        Q2 dot x = I_1( R_1 ( F_Q2(x)))
        F_Q2[ijk,mno] = rho_eq(pts_ijk) / g_sqrt * lambda^2_mno(pts_ijk)

        Parameters
        ----------
    	x : list of three np.arrays
                  list contains 3d matrix or flattened 1d vector.
                  x[0].size = dim V2_1
                  x[1].size = dim V2_2
                  x[2].size = dim V2_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^2}
        '''
        # x dim check
        # x should be R{N^2}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]) 

        # step1 : I.T(x)
        # xi1 : transpose of inter(xi1)-histo(xi2)-histo(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.N_1.T, self.D_2.T, self.D_3.T], x_loc[0])

        # xi2 : transpose of histo(xi1)-inter(xi2)-histo(xi3)-polati on.
        mat_dofs_2 = kron_solve_3d([self.D_1.T, self.N_2.T, self.D_3.T], x_loc[1])

        # xi3 : transpose of histo(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.D_1.T, self.D_2.T, self.N_3.T], x_loc[2])


        #step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.proj.dofs_T('21', mat_dofs_1)

        # xi2
        mat_f_2 = self.proj.dofs_T('22', mat_dofs_2)

        # xi3
        mat_f_3 = self.proj.dofs_T('23', mat_dofs_3)  

        # rho_eq at the projeciton points
        rho_eq_1 = self.proj.eval_for_PI('21', self.rho_eq_fun)
        rho_eq_2 = self.proj.eval_for_PI('22', self.rho_eq_fun)
        rho_eq_3 = self.proj.eval_for_PI('23', self.rho_eq_fun)

        # g_inv at the projection points
        pts_PI_1 = self.proj.pts_PI['21']
        pts_PI_2 = self.proj.pts_PI['22']
        pts_PI_3 = self.proj.pts_PI['23']

        det_df_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'det_df')
        det_df_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'det_df')
        det_df_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'det_df')

        mat_f_1_c = mat_f_1 * rho_eq_1 / det_df_1
        mat_f_2_c = mat_f_2 * rho_eq_2 / det_df_2
        mat_f_3_c = mat_f_3 * rho_eq_3 / det_df_3

        res_1 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts1_D_3.T], mat_f_1_c)
        res_2 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_2_c)
        res_3 = kron_matvec_3d([self.pts1_D_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_3_c)

        return [res_1, res_2, res_3]

    # ====================================================================
    def T2_dot(self, x):
        '''
        Calculate the dot product of projection matrix P with x
        T2     = pi_1[B_eq / g_sqrt * lambda^2]     R{N^1 * N^2}
        B_eq  = (    0  , -b_eq_z,  b_eq_y)
                ( b_eq_z,     0  , -b_eq_x)
                (-b_eq_y,  b_eq_x,     0  )

        T2 dot x = I_1( R_1 ( F_T2( x )))

        F_T2[ijk, mno] = B_eq(pts_ijk) * lambda^2_mno(pts_ijk)  

        # spline evaluation
        lambda^2          : xi1 : {N, D, D}
                            xi2 : {D, N, D}
                            xi3 : {D, D, N} 
        Evaluation points : xi1 : {quad_pts[0], greville[1], greville[2]}
                            xi2 : {greville[0], quad_pts[1], quad_pts[2]}
                            xi3 : {greville[0], greville[1], quad_pts[2]}

        # The following blocks need to be computed:
        xi1: [his, int, int] : (N, D, D) / g_sqrt *    0    ,
                               (D, N, D) / g_sqrt * -b_eq_z ,
                               (D, D, N) / g_sqrt *  b_eq_y
        xi2: [int, his, int] : (N, D, D) / g_sqrt *  b_eq_z ,
                               (D, N, D) / g_sqrt *    0    ,
                               (D, D, N) / g_sqrt * -b_eq_x 
        xi3: [int, int, his] : (N, D, D) / g_sqrt * -b_eq_y ,
                               (D, N, D) / g_sqrt *  b_eq_x ,  
                               (D, D, N) / g_sqrt *    0

        Parameters
        ----------
        x : list of three np.arrays
            list contains V1 finite element coefficients, either as 3d matrix or flattened 1d vector.
            x[0].size = dim V2_1
            x[1].size = dim V2_2
            x[2].size = dim V2_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^1}
        '''    
        # x dim check
        # x should be R{N^2}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[2] = x[2]
            
        assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]) 

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_11 = kron_matvec_3d([self.pts1_N_1, self.pts0_D_2, self.pts0_D_3], x_loc[0]) #0
        mat_f_12 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts0_D_3], x_loc[1])
        mat_f_13 = kron_matvec_3d([self.pts1_D_1, self.pts0_D_2, self.pts0_N_3], x_loc[2])

        # xi2
        mat_f_21 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts0_D_3], x_loc[0])
        mat_f_22 = kron_matvec_3d([self.pts0_D_1, self.pts1_N_2, self.pts0_D_3], x_loc[1]) #0
        mat_f_23 = kron_matvec_3d([self.pts0_D_1, self.pts1_D_2, self.pts0_N_3], x_loc[2])

        # xi3
        mat_f_31 = kron_matvec_3d([self.pts0_N_1, self.pts0_D_2, self.pts1_D_3], x_loc[0])
        mat_f_32 = kron_matvec_3d([self.pts0_D_1, self.pts0_N_2, self.pts1_D_3], x_loc[1])
        mat_f_33 = kron_matvec_3d([self.pts0_D_1, self.pts0_D_2, self.pts1_N_3], x_loc[2]) #0

        # b_eq at the projeciton points
        b_eq_x_1 = self.proj.eval_for_PI('11', self.b_eq_x_fun) # not used
        b_eq_x_2 = self.proj.eval_for_PI('12', self.b_eq_x_fun)
        b_eq_x_3 = self.proj.eval_for_PI('13', self.b_eq_x_fun)

        b_eq_y_1 = self.proj.eval_for_PI('11', self.b_eq_y_fun)
        b_eq_y_2 = self.proj.eval_for_PI('12', self.b_eq_y_fun) # not used
        b_eq_y_3 = self.proj.eval_for_PI('13', self.b_eq_y_fun)

        b_eq_z_1 = self.proj.eval_for_PI('11', self.b_eq_z_fun)
        b_eq_z_2 = self.proj.eval_for_PI('12', self.b_eq_z_fun)
        b_eq_z_3 = self.proj.eval_for_PI('13', self.b_eq_z_fun) # not used

        # g_inv at the projection points
        pts_PI_1 = self.proj.pts_PI['11']
        pts_PI_2 = self.proj.pts_PI['12']
        pts_PI_3 = self.proj.pts_PI['13']

        det_df_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'det_df')
        det_df_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'det_df')
        det_df_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'det_df')

        mat_f_11_c = mat_f_11 * 0           /det_df_1
        mat_f_12_c = mat_f_12 * (-b_eq_z_1) /det_df_1
        mat_f_13_c = mat_f_13 * ( b_eq_y_1) /det_df_1

        mat_f_21_c = mat_f_21 * ( b_eq_z_2) /det_df_2
        mat_f_22_c = mat_f_22 * 0           /det_df_2
        mat_f_23_c = mat_f_23 * (-b_eq_x_2) /det_df_2

        mat_f_31_c = mat_f_31 * (-b_eq_y_3) /det_df_3
        mat_f_32_c = mat_f_32 * ( b_eq_x_3) /det_df_3
        mat_f_33_c = mat_f_33 * 0           /det_df_3

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c

        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.proj.dofs('11', mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.proj.dofs('12', mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.proj.dofs('13', mat_f_3_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : histo(xi1)-inter(xi2)-inter(xi3)-polation.
        res_1 = self.proj.PI_mat('11', DOF_1)

        # xi2 : inter(xi1)-histo(xi2)-inter(xi3)-polation.
        res_2 = self.proj.PI_mat('12', DOF_2)

        # xi3 : inter(xi1)-inter(xi2)-histo(xi3)-polation.
        res_3 = self.proj.PI_mat('13', DOF_3)

        return [res_1, res_2, res_3]

    # ====================================================================
    def transpose_T2_dot(self, x):
        '''
        Calculate the dot product of transpose of projection matrix T2 with x
        T2     = pi_2[B_eq / g_sqrt * lambda^2]     R{N^1 * N^2}
        B_eq  = (    0  , -b_eq_z,  b_eq_y)
                ( b_eq_z,     0  , -b_eq_x)
                (-b_eq_y,  b_eq_x,     0  )

        T2.T dot x = F_T2.T( R_1.T ( I_1.T (x)))

        Parameters
        ----------
        x : list of three np.arrays
            list contains 3d matrix or flattened 1d vector.
            x[0].size = dim V1_1
            x[1].size = dim V1_2
            x[2].size = dim V1_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^2}
        '''

        # x dim check
        # x should be R{N^1}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseN[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseN[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]) 

        # step1 : I.T(x)
        # xi1 : transpose of histo(xi1)-inter(xi2)-inter(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.D_1.T, self.N_2.T, self.N_3.T], x_loc[0])

        # xi2 : transpose of inter(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_2 = kron_solve_3d([self.N_1.T, self.D_2.T, self.N_3.T], x_loc[1])

        # xi3 : transpose of inter(xi1)-inter(xi2)-histo(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.N_1.T, self.N_2.T, self.D_3.T], x_loc[2])


        #step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.proj.dofs_T('11', mat_dofs_1)

        # xi2
        mat_f_2 = self.proj.dofs_T('12', mat_dofs_2)

        # xi3
        mat_f_3 = self.proj.dofs_T('13', mat_dofs_3)         

        #step3 : F.T( R.T( I.T(x) ) )
        # b_eq at the projeciton points
        b_eq_x_1 = self.proj.eval_for_PI('11', self.b_eq_x_fun) # not used
        b_eq_x_2 = self.proj.eval_for_PI('12', self.b_eq_x_fun)
        b_eq_x_3 = self.proj.eval_for_PI('13', self.b_eq_x_fun)

        b_eq_y_1 = self.proj.eval_for_PI('11', self.b_eq_y_fun)
        b_eq_y_2 = self.proj.eval_for_PI('12', self.b_eq_y_fun) # not used
        b_eq_y_3 = self.proj.eval_for_PI('13', self.b_eq_y_fun)

        b_eq_z_1 = self.proj.eval_for_PI('11', self.b_eq_z_fun)
        b_eq_z_2 = self.proj.eval_for_PI('12', self.b_eq_z_fun)
        b_eq_z_3 = self.proj.eval_for_PI('13', self.b_eq_z_fun) # not used

        # g_inv at the projection points
        pts_PI_1 = self.proj.pts_PI['11']
        pts_PI_2 = self.proj.pts_PI['12']
        pts_PI_3 = self.proj.pts_PI['13']

        det_df_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'det_df')
        det_df_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'det_df')
        det_df_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'det_df')

        mat_f_11_c = mat_f_1 * 0           /det_df_1
        mat_f_12_c = mat_f_1 * (-b_eq_z_1) /det_df_1
        mat_f_13_c = mat_f_1 * ( b_eq_y_1) /det_df_1

        mat_f_21_c = mat_f_2 * ( b_eq_z_2) /det_df_2
        mat_f_22_c = mat_f_2 * 0           /det_df_2
        mat_f_23_c = mat_f_2 * (-b_eq_x_2) /det_df_2

        mat_f_31_c = mat_f_3 * (-b_eq_y_3) /det_df_3
        mat_f_32_c = mat_f_3 * ( b_eq_x_3) /det_df_3
        mat_f_33_c = mat_f_3 * 0           /det_df_3

        res_11 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts0_D_3.T], mat_f_11_c) #0
        res_12 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts0_D_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts1_N_1.T, self.pts0_D_2.T, self.pts0_D_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_22_c) #0
        res_23 = kron_matvec_3d([self.pts0_D_1.T, self.pts1_N_2.T, self.pts0_D_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts0_D_1.T, self.pts0_D_2.T, self.pts1_N_3.T], mat_f_33_c) #0

        return [res_11 + res_12 + res_13, res_21 + res_22 + res_23, res_31 + res_32 + res_33]

    # ====================================================================
    def S2_dot(self, x):
        '''
        Calculate the dot product of projection matrix S2 with x
        S2    = pi_2[G_inv * j_eq * lambda^2]     R{N^2 * N^2}
        G_inv = (DF)^(-1)*DF^(-T)
        j_eq  = (    0  , -j_eq_z,  j_eq_y)
                ( j_eq_z,     0  , -j_eq_x)
                (-j_eq_y,  j_eq_x,     0  )

        S2 dot x = I_1( R_1 ( F_S2(x)))

        F_S2[ijk, mno] = G_inv(pts_ijk) * j_eq(pts_ijk) lambda^2_mno(pts_ijk)

        # Spline evaluation
        lambda^2          : xi1 : {N, D, D}
                            xi2 : {D, N, D}
                            xi3 : {D, D, N} 
        Evaluation points : xi1 : {greville[0], quad_pts[1], quad_pts[2]}
                            xi2 : {quad_pts[0], greville[1], quad_pts[2]}
                            xi2 : {quad_pts[0], quad_pts[1], greville[2]}

        # The following blocks need to be computed:
        xi1: [int, his, his] : (N, D, D) * (G_inv_12 * j_eq_z - G_inv_13 * j_eq_y),
                               (D, N, D) * (G_inv_13 * j_eq_x - G_inv_11 * j_eq_z),
                               (D, D, N) * (G_inv_11 * j_eq_y - G_inv_12 * j_eq_x)
        xi2: [his, int, his] : (N, D, D) * (G_inv_22 * j_eq_z - G_inv_23 * j_eq_y),
                               (D, N, D) * (G_inv_23 * j_eq_x - G_inv_21 * j_eq_z),
                               (D, D, N) * (G_inv_21 * j_eq_y - G_inv_22 * j_eq_x)
        xi3: [his, his, int] : (N, D, D) * (G_inv_32 * j_eq_z - G_inv_33 * j_eq_y),
                               (D, N, D) * (G_inv_33 * j_eq_x - G_inv_31 * j_eq_z),  
                               (D, D, N) * (G_inv_31 * j_eq_y - G_inv_32 * j_eq_x)

        Parameters
        ----------
        x : list of three np.arrays
            list contains V1 finite element coefficients, either as 3d matrix or flattened 1d vector.
            x[0].size = dim V2_1
            x[1].size = dim V2_2
            x[2].size = dim V2_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^2}
        '''
        # x dim check
        # x should be R{N^2}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]) 

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

        # j_eq at the projeciton points
        j_eq_x_1 = self.proj.eval_for_PI('21', self.j_eq_x_fun)
        j_eq_x_2 = self.proj.eval_for_PI('22', self.j_eq_x_fun)
        j_eq_x_3 = self.proj.eval_for_PI('23', self.j_eq_x_fun)

        j_eq_y_1 = self.proj.eval_for_PI('21', self.j_eq_y_fun)
        j_eq_y_2 = self.proj.eval_for_PI('22', self.j_eq_y_fun) 
        j_eq_y_3 = self.proj.eval_for_PI('23', self.j_eq_y_fun)

        j_eq_z_1 = self.proj.eval_for_PI('21', self.j_eq_z_fun)
        j_eq_z_2 = self.proj.eval_for_PI('22', self.j_eq_z_fun)
        j_eq_z_3 = self.proj.eval_for_PI('23', self.j_eq_z_fun)

        # g_inv at the projection points
        pts_PI_1 = self.proj.pts_PI['21']
        pts_PI_2 = self.proj.pts_PI['22']
        pts_PI_3 = self.proj.pts_PI['23']

        g_inv_11 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_11')
        g_inv_12 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_12')
        g_inv_13 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_13')
        g_inv_21 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_21')
        g_inv_22 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_22')
        g_inv_23 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_23')
        g_inv_31 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_31')
        g_inv_32 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_32')
        g_inv_33 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_33')

        mat_f_11_c = mat_f_11 * (g_inv_12 * j_eq_z_1 - g_inv_13 * j_eq_y_1)        
        mat_f_12_c = mat_f_12 * (g_inv_13 * j_eq_x_1 - g_inv_11 * j_eq_z_1) 
        mat_f_13_c = mat_f_13 * (g_inv_11 * j_eq_y_1 - g_inv_12 * j_eq_x_1) 

        mat_f_21_c = mat_f_21 * (g_inv_22 * j_eq_z_2 - g_inv_23 * j_eq_y_2) 
        mat_f_22_c = mat_f_22 * (g_inv_23 * j_eq_x_2 - g_inv_21 * j_eq_z_2) 
        mat_f_23_c = mat_f_23 * (g_inv_21 * j_eq_y_2 - g_inv_22 * j_eq_x_2) 

        mat_f_31_c = mat_f_31 * (g_inv_32 * j_eq_z_3 - g_inv_33 * j_eq_y_3) 
        mat_f_32_c = mat_f_32 * (g_inv_33 * j_eq_x_3 - g_inv_31 * j_eq_z_3) 
        mat_f_33_c = mat_f_33 * (g_inv_31 * j_eq_y_3 - g_inv_32 * j_eq_x_3) 

        mat_f_1_c = mat_f_11_c + mat_f_12_c + mat_f_13_c
        mat_f_2_c = mat_f_21_c + mat_f_22_c + mat_f_23_c
        mat_f_3_c = mat_f_31_c + mat_f_32_c + mat_f_33_c


        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.proj.dofs('21', mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.proj.dofs('22', mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.proj.dofs('23', mat_f_3_c)


        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : inter(xi1)-histo(xi2)-histo(xi3)-polation.
        res_1 = self.proj.PI_mat('21', DOF_1)

        # xi2 : histo(xi1)-inter(xi2)-histo(xi3)-polation.
        res_2 = self.proj.PI_mat('22', DOF_2)

        # xi3 : histo(xi1)-histo(xi2)-inter(xi3)-polation.
        res_3 = self.proj.PI_mat('23', DOF_3)

        return [res_1, res_2, res_3]


    # ====================================================================
    def transpose_S2_dot(self, x):
        '''
        Calculate the dot product of projection matrix S2 with x
        S2    = pi_2[G_inv * j_eq * lambda^2]     R{N^2 * N^2}
        G_inv = (DF)^(-1)*DF^(-T)
        j_eq  = (    0  , -j_eq_z,  j_eq_y)
                ( j_eq_z,     0  , -j_eq_x)
                (-j_eq_y,  j_eq_x,     0  )

        S2.T dot x = F_S2.T( R_2.T ( I_2.T (x)))

        F_S2[ijk, mno] = G_inv(pts_ijk) * j_eq(pts_ijk) lambda^2_mno(pts_ijk)

        Parameters
        ----------
    	x : list of three of 3d array which will product with the projection matrices  R{N^2}

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^2   }
        '''

        # x dim check
        # x should be R{N^2}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]) 
 
        # step1 : I.T(x)
        # xi1 : transpose of inter(xi1)-histo(xi2)-histo(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.N_1.T, self.D_2.T, self.D_3.T], x_loc[0])

        # xi2 : transpose of histo(xi1)-inter(xi2)-histo(xi3)-polati on.
        mat_dofs_2 = kron_solve_3d([self.D_1.T, self.N_2.T, self.D_3.T], x_loc[1])

        # xi3 : transpose of histo(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.D_1.T, self.D_2.T, self.N_3.T], x_loc[2])


        #step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.proj.dofs_T('21', mat_dofs_1)

        # xi2
        mat_f_2 = self.proj.dofs_T('22', mat_dofs_2)

        # xi3
        mat_f_3 = self.proj.dofs_T('23', mat_dofs_3)         

        # j_eq at the projeciton points
        j_eq_x_1 = self.proj.eval_for_PI('21', self.j_eq_x_fun)
        j_eq_x_2 = self.proj.eval_for_PI('22', self.j_eq_x_fun)
        j_eq_x_3 = self.proj.eval_for_PI('23', self.j_eq_x_fun)

        j_eq_y_1 = self.proj.eval_for_PI('21', self.j_eq_y_fun)
        j_eq_y_2 = self.proj.eval_for_PI('22', self.j_eq_y_fun) 
        j_eq_y_3 = self.proj.eval_for_PI('23', self.j_eq_y_fun)

        j_eq_z_1 = self.proj.eval_for_PI('21', self.j_eq_z_fun)
        j_eq_z_2 = self.proj.eval_for_PI('22', self.j_eq_z_fun)
        j_eq_z_3 = self.proj.eval_for_PI('23', self.j_eq_z_fun)

        # g_inv at the projection points
        pts_PI_1 = self.proj.pts_PI['21']
        pts_PI_2 = self.proj.pts_PI['22']
        pts_PI_3 = self.proj.pts_PI['23']

        g_inv_11 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_11')
        g_inv_12 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_12')
        g_inv_13 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'g_inv_13')
        g_inv_21 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_21')
        g_inv_22 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_22')
        g_inv_23 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'g_inv_23')
        g_inv_31 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_31')
        g_inv_32 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_32')
        g_inv_33 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'g_inv_33')

        mat_f_11_c = mat_f_1 * (g_inv_12 * j_eq_z_1 - g_inv_13 * j_eq_y_1)        
        mat_f_12_c = mat_f_1 * (g_inv_13 * j_eq_x_1 - g_inv_11 * j_eq_z_1) 
        mat_f_13_c = mat_f_1 * (g_inv_11 * j_eq_y_1 - g_inv_12 * j_eq_x_1) 

        mat_f_21_c = mat_f_2 * (g_inv_22 * j_eq_z_2 - g_inv_23 * j_eq_y_2) 
        mat_f_22_c = mat_f_2 * (g_inv_23 * j_eq_x_2 - g_inv_21 * j_eq_z_2) 
        mat_f_23_c = mat_f_2 * (g_inv_21 * j_eq_y_2 - g_inv_22 * j_eq_x_2) 

        mat_f_31_c = mat_f_3 * (g_inv_32 * j_eq_z_3 - g_inv_33 * j_eq_y_3) 
        mat_f_32_c = mat_f_3 * (g_inv_33 * j_eq_x_3 - g_inv_31 * j_eq_z_3) 
        mat_f_33_c = mat_f_3 * (g_inv_31 * j_eq_y_3 - g_inv_32 * j_eq_x_3) 

        res_11 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts1_D_3.T], mat_f_11_c)
        res_12 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts1_D_3.T], mat_f_12_c)
        res_13 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts1_D_3.T], mat_f_13_c)

        res_21 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_21_c)
        res_22 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_22_c)
        res_23 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_23_c)

        res_31 = kron_matvec_3d([self.pts1_D_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_31_c)
        res_32 = kron_matvec_3d([self.pts1_D_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_32_c)
        res_33 = kron_matvec_3d([self.pts1_D_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_33_c)

        return [res_11 + res_12 + res_13, res_21 + res_22 + res_23, res_31 + res_32 + res_33]


        # ====================================================================
    def P2_dot(self, x):
        '''
        Calculate the dot product of projection matrix P2 with x
        P2 = pi_2[p_eq / g_sqrt * lambda^2]     R{N^2 * N^2}

        P2 dot x = I_1( R_1 ( F_P2(x)))
        F_P2[ijk,mno] = p_eq(pts_ijk) / g_sqrt * lambda^2_mno(pts_ijk)

        # Spline evaluation
        lambda^2          : xi1 : {N, D, D}
                            xi2 : {D, N, D}
                            xi3 : {D, D, N} 
        Evaluation points : xi1 : {greville[0], quad_pts[1], quad_pts[2]}
                            xi2 : {quad_pts[0], greville[1], quad_pts[2]}
                            xi2 : {quad_pts[0], quad_pts[1], greville[2]}

        # The following blocks need to be computed:
        xi1: [int, his, his] : (N, D, D) * p_eq / g_sqrt, 0, 0
        xi2: [his, int, his] : 0, (D, N, D) * p_eq / g_sqrt, 0
        xi3: [his, his, int] : 0, 0, (D, D, N) * p_eq / g_sqrt

        Parameters
        ----------
        x : list of three np.arrays
            list contains V1 finite element coefficients, either as 3d matrix or flattened 1d vector.
            x[0].size = dim V2_1
            x[1].size = dim V2_2
            x[2].size = dim V2_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^2}
        '''    
        # x dim check
        # x should be R{N^2}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]) 

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.
        # xi1
        mat_f_1 = kron_matvec_3d([self.pts0_N_1, self.pts1_D_2, self.pts1_D_3], x_loc[0])

        # xi2
        mat_f_2 = kron_matvec_3d([self.pts1_D_1, self.pts0_N_2, self.pts1_D_3], x_loc[1])

        # xi3
        mat_f_3 = kron_matvec_3d([self.pts1_D_1, self.pts1_D_2, self.pts0_N_3], x_loc[2])

 
        # rho_eq at the projeciton points
        p_eq_1 = self.proj.eval_for_PI('21', self.p_eq_fun)
        p_eq_2 = self.proj.eval_for_PI('22', self.p_eq_fun)
        p_eq_3 = self.proj.eval_for_PI('23', self.p_eq_fun)

        # g_inv at the projection points
        pts_PI_1 = self.proj.pts_PI['21']
        pts_PI_2 = self.proj.pts_PI['22']
        pts_PI_3 = self.proj.pts_PI['23']

        det_df_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'det_df')
        det_df_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'det_df')
        det_df_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'det_df')

        mat_f_1_c = mat_f_1 * p_eq_1 / det_df_1
        mat_f_2_c = mat_f_2 * p_eq_2 / det_df_2
        mat_f_3_c = mat_f_3 * p_eq_3 / det_df_3


        # ========== Step 2 : R( F(x) ) ==========#
        # integration over quadrature points
        # xi1 : DOF_1_{i,j,k} = sum_{m} w_{i,m} * mat_f_1_{i,m,j,k}
        DOF_1 = self.proj.dofs('21', mat_f_1_c)

        # xi2 : DOF_2_{i,j,k} = sum_{m} w_{j,m} * mat_f_2_{i,j,m,k}
        DOF_2 = self.proj.dofs('22', mat_f_2_c)

        # xi3 : DOF_3_{i,j,k} = sum_{m} w_{k,m} * mat_f_3_{i,j,k,m}
        DOF_3 = self.proj.dofs('23', mat_f_3_c)


        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # xi1 : inter(xi1)-histo(xi2)-histo(xi3)-polation.
        res_1 = self.proj.PI_mat('21', DOF_1)

        # xi2 : histo(xi1)-inter(xi2)-histo(xi3)-polation.
        res_2 = self.proj.PI_mat('22', DOF_2)

        # xi3 : histo(xi1)-histo(xi2)-inter(xi3)-polation.
        res_3 = self.proj.PI_mat('23', DOF_3)


        return [res_1, res_2, res_3]

    # ====================================================================
    def transpose_P2_dot(self, x):
        '''
        Calculate the dot product of projection matrix P2 with x
        P2 = pi_2[p_eq / g_sqrt * lambda^2]     R{N^2 * N^2}

        P2 dot x = I_1( R_1 ( F_P2(x)))
        F_P2[ijk,mno] = p_eq(pts_ijk) / g_sqrt * lambda^2_mno(pts_ijk)

        Parameters
        ----------
    	x : list of three np.arrays
                  list contains 3d matrix or flattened 1d vector.
                  x[0].size = dim V2_1
                  x[1].size = dim V2_2
                  x[2].size = dim V2_3

        Returns
        ----------
        res : list of three 3d arrays [res_1, res_2, res_3]     R{N^2}
        '''
        # x dim check
        # x should be R{N^2}
        x_loc = [None, None, None]

        if len(x[0].shape) == 1:
            x_loc[0] = x[0].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])
        else:
            x_loc[0] = x[0]

        assert x_loc[0].shape ==  (self.NbaseN[0], self.NbaseD[1], self.NbaseD[2])

        if len(x[1].shape) == 1:
            x_loc[1] = x[1].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
        else:
            x_loc[1] = x[1]

        assert x_loc[1].shape ==  (self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])

        if len(x[2].shape) == 1:
            x_loc[2] = x[2].reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])
        else:
            x_loc[2] = x[2]

        assert x_loc[2].shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]) 

        # step1 : I.T(x)
        # xi1 : transpose of inter(xi1)-histo(xi2)-histo(xi3)-polation.
        mat_dofs_1 = kron_solve_3d([self.N_1.T, self.D_2.T, self.D_3.T], x_loc[0])

        # xi2 : transpose of histo(xi1)-inter(xi2)-histo(xi3)-polati on.
        mat_dofs_2 = kron_solve_3d([self.D_1.T, self.N_2.T, self.D_3.T], x_loc[1])

        # xi3 : transpose of histo(xi1)-histo(xi2)-inter(xi3)-polation.
        mat_dofs_3 = kron_solve_3d([self.D_1.T, self.D_2.T, self.N_3.T], x_loc[2])


        #step2 : R.T( I.T(x) )
        # transpose of integration over quadrature points
        # xi1
        mat_f_1 = self.proj.dofs_T('21', mat_dofs_1)

        # xi2
        mat_f_2 = self.proj.dofs_T('22', mat_dofs_2)

        # xi3
        mat_f_3 = self.proj.dofs_T('23', mat_dofs_3)  

        # p_eq at the projeciton points
        p_eq_1 = self.proj.eval_for_PI('21', self.p_eq_fun)
        p_eq_2 = self.proj.eval_for_PI('22', self.p_eq_fun)
        p_eq_3 = self.proj.eval_for_PI('23', self.p_eq_fun)

        # g_inv at the projection points
        pts_PI_1 = self.proj.pts_PI['21']
        pts_PI_2 = self.proj.pts_PI['22']
        pts_PI_3 = self.proj.pts_PI['23']

        det_df_1 = self.domain.evaluate(pts_PI_1[0], pts_PI_1[1], pts_PI_1[2], 'det_df')
        det_df_2 = self.domain.evaluate(pts_PI_2[0], pts_PI_2[1], pts_PI_2[2], 'det_df')
        det_df_3 = self.domain.evaluate(pts_PI_3[0], pts_PI_3[1], pts_PI_3[2], 'det_df')

        mat_f_1_c = mat_f_1 * p_eq_1 / det_df_1
        mat_f_2_c = mat_f_2 * p_eq_2 / det_df_2
        mat_f_3_c = mat_f_3 * p_eq_3 / det_df_3

        res_1 = kron_matvec_3d([self.pts0_N_1.T, self.pts1_D_2.T, self.pts1_D_3.T], mat_f_1_c)
        res_2 = kron_matvec_3d([self.pts1_D_1.T, self.pts0_N_2.T, self.pts1_D_3.T], mat_f_2_c)
        res_3 = kron_matvec_3d([self.pts1_D_1.T, self.pts1_D_2.T, self.pts0_N_3.T], mat_f_3_c)

        return [res_1, res_2, res_3]

    # ====================================================================
    def M_dot(self, x):
        '''
        Calculate the dot product of projection matrix M with x
        M = pi_3[p_eq / g_sqrt * lambda^3]     R{N^3 * N^3}

        M dot x = I_3( R_3 ( F_M(x)))
        
        F_M[ijk,mno] = p_eq(pts_ijk) / g_sqrt * lambda^3_mno(pts_ijk)

        # spline evaluation
        lambda^3          : {D, D, D} 
        Evaulation points : {quad_pts[0], quad_pts[1], quad_pts[2]}

        Parameters
        ----------
        x : np.array
            V3 finite element coefficients, either 3d matrix or flattened 1d vector.
            x.size = dim V3

        Returns
        ----------
        res : 3d array     R{N^3}
        '''

        # x dim check
        if len(x[0].shape) == 1:
            x_loc = x.reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseD[2])
        else:
            x_loc = x

        assert x_loc.shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseD[2])

        # ========== Step 1 : F(x) ========== #
        # Splline evaulation at the projection points then dot product with the x.

        mat_f = kron_matvec_3d([self.pts1_D_1, self.pts1_D_2, self.pts1_D_3], x_loc)
        
        # p_eq at the projeciton points
        p_eq = self.proj.eval_for_PI('3', self.p_eq_fun)

        # g_inv at the projection points
        pts_PI = self.proj.pts_PI['3']

        det_df = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'det_df')

        # Point-wise multiplication of mat_f and peq
        mat_f_c = mat_f * p_eq / det_df

        # ========== Step 2 : R( F(x) ) ==========#
        # Linear operator : evaluation values at the projection points to the Degree of Freedom of the spline.
        DOF = self.proj.dofs('3', mat_f_c)

        # ========== Step 3 : I( R( F(x) ) ) ==========#
        # histo(xi1)-histo(xi2)-histo(xi3)-polation.
        res = self.proj.PI_mat('3', DOF)

        return res

    # ====================================================================
    def transpose_M_dot(self, x):
        '''
        Calculate the dot product of projection matrix M with x
        M = pi_3[p_eq / g_sqrt * lambda^3]     R{N^3 * N^3}
        
        M.T dot x = F_M.T( R_3.T ( I_3.T(x)))

        F_M[ijk,mno] = p_eq(pts_ijk) / g_sqrt * lambda^3_mno(pts_ijk)

        Parameters
        ----------
        x : np.array
            3d matrix or flattened 1d vector.
            x.size = dim V0

        Returns
        ----------
        res : 3d array     R{N^0}
        '''

        # x dim check
        if len(x[0].shape) == 1:
            x_loc = x.reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseD[2])
        else:
            x_loc = x

        assert x_loc.shape ==  (self.NbaseD[0], self.NbaseD[1], self.NbaseD[2])

        # step1 : I.T(x)
        mat_dofs = kron_solve_3d([self.D_1.T, self.D_2.T, self.D_3.T], x_loc)

        #step2 : R.T( I.T(x) )
        mat_f = self.proj.dofs_T('3', mat_dofs)

        #step3 : F.T( R.T( I.T(x) ) )
        # p_eq at the projeciton points
        p_eq = self.proj.eval_for_PI('3', self.p_eq_fun)

        # g_inv at the projection points
        pts_PI = self.proj.pts_PI['3']

        det_df = self.domain.evaluate(pts_PI[0], pts_PI[1], pts_PI[2], 'det_df')

        mat_f_c = p_eq * mat_f / det_df

        res = kron_matvec_3d([self.pts1_D_1.T, self.pts1_D_2.T, self.pts1_D_3.T], mat_f_c)

        return res

