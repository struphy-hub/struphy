# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Class for 2D linear MHD operators.
"""


import numpy as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.bsplines as bsp
import hylife.utilitis_FEEC.projectors.kernels_projectors_global_mhd as ker

import hylife.utilitis_FEEC.basics.mass_matrices_2d as mass

import hylife.utilitis_FEEC.projectors.projectors_global as pro


class operators_mhd:
    """
    Global linear MHD operators.
    
    Parameters
    ----------
    projectors_2d : projectors_global_2d
        2d global commuting projectors object
        
    basis_u : int
        formulation for MHD bulk velocity (0: vector field, 1: 1-form, 2: 2-form)
    """
    
    def __init__(self, projectors_2d, basis_u):
        
        # 2D projector
        self.pro = projectors_2d
        
        # parameters
        self.basis_u = basis_u
        
        # non-vanishing 1D indices of expressions R0(N), R0(D), R1(N) and R1(D)
        projectors_1d = [pro.projectors_global_1d(space, n_quad) for space, n_quad in zip(self.pro.space.spaces, self.pro.n_quad)]
        
        self.pi0_x_N_i, self.pi0_x_D_i, self.pi1_x_N_i, self.pi1_x_D_i = projectors_1d[0].dofs_1d_bases()
        self.pi0_y_N_i, self.pi0_y_D_i, self.pi1_y_N_i, self.pi1_y_D_i = projectors_1d[1].dofs_1d_bases()
        
        # non-vanishing 1D indices of expressions R0(NN), R0(DN), R0(ND), R0(DD), R1(NN), R1(DN), R1(ND) and R1(DD)
        #self.pi0_x_NN_i, self.pi0_x_DN_i, self.pi0_x_ND_i, self.pi0_x_DD_i, self.pi1_x_NN_i, self.pi1_x_DN_i, self.pi1_x_ND_i, self.pi1_x_DD_i = projectors_1d[0].dofs_1d_bases_products()
        #self.pi0_y_NN_i, self.pi0_y_DN_i, self.pi0_y_ND_i, self.pi0_y_DD_i, self.pi1_y_NN_i, self.pi1_y_DN_i, self.pi1_y_ND_i, self.pi1_y_DD_i = projectors_1d[1].dofs_1d_bases_products()
        
        kind_splines = [False, True]
        
        # 1D collocation matrices for interpolation in format (point, global basis function)
        x_int_1d = [np.copy(pro.x_int) for pro in projectors_1d]
        
        self.basis_int_N = [bsp.collocation_matrix(T, p    , x_int, bc, normalize=kind_splines[0]) for T, p, x_int, bc in zip(self.pro.space.T, self.pro.space.p, x_int_1d, self.pro.space.spl_kind)]
        self.basis_int_D = [bsp.collocation_matrix(t, p - 1, x_int, bc, normalize=kind_splines[1]) for t, p, x_int, bc in zip(self.pro.space.t, self.pro.space.p, x_int_1d, self.pro.space.spl_kind)]
        
        # remove small values resulting from round-off errors
        self.basis_int_N[0][self.basis_int_N[0] < 1e-12] = 0.
        self.basis_int_N[1][self.basis_int_N[1] < 1e-12] = 0.

        self.basis_int_D[0][self.basis_int_D[0] < 1e-12] = 0.
        self.basis_int_D[1][self.basis_int_D[1] < 1e-12] = 0.
                
        # evaluate basis functions at quadrature points in format (interval, local quad. point, global basis function)
        self.basis_his_N = [bsp.collocation_matrix(T, p    , pts.flatten(), bc, normalize=kind_splines[0]).reshape(pts.shape[0], pts.shape[1], NbaseN) for T, p, pts, bc, NbaseN in zip(self.pro.space.T, self.pro.space.p, self.pro.pts, self.pro.space.spl_kind, self.pro.space.NbaseN)]
        self.basis_his_D = [bsp.collocation_matrix(t, p - 1, pts.flatten(), bc, normalize=kind_splines[1]).reshape(pts.shape[0], pts.shape[1], NbaseD) for t, p, pts, bc, NbaseD in zip(self.pro.space.t, self.pro.space.p, self.pro.pts, self.pro.space.spl_kind, self.pro.space.NbaseD)]
            
            
    # =================================================================
    def assemble_rhs_EF(self, domain, b2_eq):
        
        eta3 = np.array([0.])
        
        if self.basis_u == 2:

            # ====================== 12 - block ([his, int] of DN) ===========================
            # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
            if callable(b2_eq[2]):
                B2_3_pts = b2_eq[2](self.pro.pts[0].flatten(), self.pro.x_int[1])
            else:
                B2_3_pts = self.pro.space.evaluate_DD(self.pro.pts[0].flatten(), self.pro.x_int[1], b2_eq, 'V2')
                
            B2_3_pts = B2_3_pts.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size)

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.pts[0].flatten(), self.pro.x_int[1], eta3, 'det_df'))
            det_dF = det_dF.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size)

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size, dtype=int)

            ker.rhs11_2d(self.pi1_x_D_i[0], self.pi0_y_N_i[0], self.pi1_x_D_i[1], self.pi0_y_N_i[1], self.pro.subs[0], self.pro.subs_cum[0], self.pro.wts[0], self.basis_his_D[0], self.basis_int_N[1], self.pro.space.NbaseN, self.pro.space.NbaseD, -B2_3_pts/det_dF, values, row_all, col_all)

            self.EF_12 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[0], self.pro.space.Ntot_2form[1]))
            self.EF_12.eliminate_zeros()

            
            # ====================== 13 - block ([his, int] of DD) ===========================
            # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
            if callable(b2_eq[1]):
                B2_2_pts = b2_eq[1](self.pro.pts[0].flatten(), self.pro.x_int[1])
            else:
                B2_2_pts = self.pro.space.evaluate_DN(self.pro.pts[0].flatten(), self.pro.x_int[1], b2_eq, 'V2')

            B2_2_pts = B2_2_pts.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size)
            
            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.pts[0].flatten(), self.pro.x_int[1], eta3, 'det_df'))
            det_dF = det_dF.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size)

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_D_i[0].size, dtype=int)

            ker.rhs11_2d(self.pi1_x_D_i[0], self.pi0_y_D_i[0], self.pi1_x_D_i[1], self.pi0_y_D_i[1], self.pro.subs[0], self.pro.subs_cum[0], self.pro.wts[0], self.basis_his_D[0], self.basis_int_D[1], self.pro.space.NbaseN, self.pro.space.NbaseD,  B2_2_pts/det_dF, values, row_all, col_all)

            self.EF_13 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[0], self.pro.space.Ntot_2form[2]))
            self.EF_13.eliminate_zeros()

            
            # ====================== 21 - block ([int, his] of ND) ===========================
            # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
            if callable(b2_eq[2]):
                B2_3_pts = b2_eq[2](self.pro.x_int[0], self.pro.pts[1].flatten())
            else:
                B2_3_pts = self.pro.space.evaluate_DD(self.pro.x_int[0], self.pro.pts[1].flatten(), b2_eq, 'V2')

            B2_3_pts = B2_3_pts.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1])
            
            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.x_int[0], self.pro.pts[1].flatten(), eta3, 'det_df'))
            det_dF = det_dF.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size, dtype=int)

            ker.rhs12_2d(self.pi0_x_N_i[0], self.pi1_y_D_i[0], self.pi0_x_N_i[1], self.pi1_y_D_i[1], self.pro.subs[1], self.pro.subs_cum[1], self.pro.wts[1], self.basis_int_N[0], self.basis_his_D[1], self.pro.space.NbaseN, self.pro.space.NbaseD,  B2_3_pts/det_dF, values, row_all, col_all)

            self.EF_21 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[1], self.pro.space.Ntot_2form[0]))
            self.EF_21.eliminate_zeros()


            # ====================== 23 - block ([int, his] of DD) ===========================
            # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
            if callable(b2_eq[0]):
                B2_1_pts = b2_eq[0](self.pro.x_int[0], self.pro.pts[1].flatten())
            else:
                B2_1_pts = self.pro.space.evaluate_ND(self.pro.x_int[0], self.pro.pts[1].flatten(), b2_eq, 'V2')

            B2_1_pts = B2_1_pts.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1])
            
            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.x_int[0], self.pro.pts[1].flatten(), eta3, 'det_df'))
            det_dF = det_dF.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_D_i[0].size*self.pi1_y_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_D_i[0].size*self.pi1_y_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_D_i[0].size*self.pi1_y_D_i[0].size, dtype=int)

            ker.rhs12_2d(self.pi0_x_D_i[0], self.pi1_y_D_i[0], self.pi0_x_D_i[1], self.pi1_y_D_i[1], self.pro.subs[1], self.pro.subs_cum[1], self.pro.wts[1], self.basis_int_D[0], self.basis_his_D[1], self.pro.space.NbaseN, self.pro.space.NbaseD, -B2_1_pts/det_dF, values, row_all, col_all)

            self.EF_23 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[1], self.pro.space.Ntot_2form[2]))
            self.EF_23.eliminate_zeros()


            # ====================== 31 - block ([int, int] of ND) ===========================
            # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
            if callable(b2_eq[1]):
                B2_2_pts = b2_eq[1](self.pro.x_int[0], self.pro.x_int[1])
            else:
                B2_2_pts = self.pro.space.evaluate_DN(self.pro.x_int[0], self.pro.x_int[1], b2_eq, 'V2')

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.x_int[0], self.pro.x_int[1], eta3, 'det_df'))[:, :, 0]

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_D_i[0].size, dtype=int)

            ker.rhs0_2d(self.pi0_x_N_i[0], self.pi0_y_D_i[0], self.pi0_x_N_i[1], self.pi0_y_D_i[1], self.basis_int_N[0], self.basis_int_D[1], -B2_2_pts/det_dF, values, row_all, col_all)

            self.EF_31 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[2], self.pro.space.Ntot_2form[0]))
            self.EF_31.eliminate_zeros()

            
            # ====================== 32 - block ([int, int] of DN) ===========================
            # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
            if callable(b2_eq[0]):
                B2_1_pts = b2_eq[0](self.pro.x_int[0], self.pro.x_int[1])
            else:
                B2_1_pts = self.pro.space.evaluate_ND(self.pro.x_int[0], self.pro.x_int[1], b2_eq, 'V2')
            
            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.x_int[0], self.pro.x_int[1], eta3, 'det_df'))[:, :, 0]

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_D_i[0].size*self.pi0_y_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_D_i[0].size*self.pi0_y_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_D_i[0].size*self.pi0_y_N_i[0].size, dtype=int)

            ker.rhs0_2d(self.pi0_x_D_i[0], self.pi0_y_N_i[0], self.pi0_x_D_i[1], self.pi0_y_N_i[1], self.basis_int_D[0], self.basis_int_N[1],  B2_1_pts/det_dF, values, row_all, col_all)

            self.EF_32 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[2], self.pro.space.Ntot_2form[1]))
            self.EF_32.eliminate_zeros()

            # ============================== full operator ==========================================
            self.rhs_EF = self.pro.P1.dot(spa.bmat([[None, self.EF_12, self.EF_13], [self.EF_21, None, self.EF_23], [self.EF_31, self.EF_32, None]], format='csr').dot(self.pro.space.E2.T))
            self.rhs_EF.eliminate_zeros()
            
        elif self.basis_u == 0:
            print('not yet implemented!')
        
        
            
    # =================================================================
    def assemble_rhs_F(self, domain, c3_eq, which):
        
        eta3 = np.array([0.])
        
        if self.basis_u == 2:

            # ====================== 11 - block ([int, his] of ND) ===========================
            # evaluate equilibrium density/pressure at interpolation and quadrature points
            if callable(c3_eq):
                EQ = c3_eq(self.pro.x_int[0], self.pro.pts[1].flatten())
            else:
                EQ = self.pro.space.evaluate_DD(self.pro.x_int[0], self.pro.pts[1].flatten(), c3_eq, 'V3')

            EQ = EQ.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1])
            
            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.x_int[0], self.pro.pts[1].flatten(), eta3, 'det_df'))
            det_dF = det_dF.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size, dtype=int)

            ker.rhs12_2d(self.pi0_x_N_i[0], self.pi1_y_D_i[0], self.pi0_x_N_i[1], self.pi1_y_D_i[1], self.pro.subs[1], self.pro.subs_cum[1], self.pro.wts[1], self.basis_int_N[0], self.basis_his_D[1], self.pro.space.NbaseN, self.pro.space.NbaseD, EQ/det_dF, values, row_all, col_all)

            self.F_11 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[0], self.pro.space.Ntot_2form[0]))
            self.F_11.eliminate_zeros()


            # ====================== 22 - block ([his, int] of DN) ===========================
            # evaluate equilibrium density/pressure at interpolation and quadrature points
            if callable(c3_eq):
                EQ = c3_eq(self.pro.pts[0].flatten(), self.pro.x_int[1])
            else:
                EQ = self.pro.space.evaluate_DD(self.pro.pts[0].flatten(), self.pro.x_int[1], c3_eq, 'V3')
                
            EQ = EQ.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size)

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.pts[0].flatten(), self.pro.x_int[1], eta3, 'det_df'))
            det_dF = det_dF.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size)

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size, dtype=int)

            ker.rhs11_2d(self.pi1_x_D_i[0], self.pi0_y_N_i[0], self.pi1_x_D_i[1], self.pi0_y_N_i[1], self.pro.subs[0], self.pro.subs_cum[0], self.pro.wts[0], self.basis_his_D[0], self.basis_int_N[1], self.pro.space.NbaseN, self.pro.space.NbaseD, EQ/det_dF, values, row_all, col_all)

            self.F_22 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[1], self.pro.space.Ntot_2form[1]))
            self.F_22.eliminate_zeros()


            # ====================== 33 - block ([his, his] of DD) ===========================
            # evaluate equilibrium density/pressure at interpolation and quadrature points
            if callable(c3_eq):
                EQ = c3_eq(self.pro.pts[0].flatten(), self.pro.pts[1].flatten())
            else:
                EQ = self.pro.space.evaluate_DD(self.pro.pts[0].flatten(), self.pro.pts[1].flatten(), c3_eq, 'V3')

            EQ = EQ.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.pts[1].shape[0], self.pro.pts[1].shape[1])
            
            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.pts[0].flatten(), self.pro.pts[1].flatten(), eta3, 'det_df'))
            det_dF = det_dF.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.pts[1].shape[0], self.pro.pts[1].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size, dtype=int)

            ker.rhs2_2d(self.pi1_x_D_i[0], self.pi1_y_D_i[0], self.pi1_x_D_i[1], self.pi1_y_D_i[1], self.pro.subs[0], self.pro.subs[1], self.pro.subs_cum[0], self.pro.subs_cum[1], self.pro.wts[0], self.pro.wts[1], self.basis_his_D[0], self.basis_his_D[1], self.pro.space.NbaseN, self.pro.space.NbaseD, EQ/det_dF, values, row_all, col_all)

            self.F_33 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[2], self.pro.space.Ntot_2form[2]))
            self.F_33.eliminate_zeros()
        
            # ==================================== full operator ===================================
            if   which == 'm':
                self.rhs_MF = self.pro.P2.dot(spa.bmat([[self.F_11, None, None], [None, self.F_22, None], [None, None, self.F_33]], format='csr').dot(self.pro.space.E2.T))
                self.rhs_MF.eliminate_zeros()
                
            elif which == 'p':
                self.rhs_PF = self.pro.P2.dot(spa.bmat([[self.F_11, None, None], [None, self.F_22, None], [None, None, self.F_33]], format='csr').dot(self.pro.space.E2.T))
                self.rhs_PF.eliminate_zeros()
                
        elif self.basis_u == 0:
            print('not yet implemented!')
            
    
    # =================================================================
    def assemble_rhs_PR(self, domain, p3_eq):
            
        eta3 = np.array([0.])
        
        # ====================== ([his, his] of DD) ===========================
        # evaluate equilibrium pressure at quadrature points
        if callable(p3_eq):
            PR_eq = p3_eq(self.pro.pts[0].flatten(), self.pro.pts[1].flatten())
        else:
            PR_eq = self.pro.space.evaluate_DD(self.pro.pts[0].flatten(), self.pro.pts[1].flatten(), p3_eq, 'V3')
            
        PR_eq = PR_eq.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.pts[1].shape[0], self.pro.pts[1].shape[1])
        
        # evaluate Jacobian determinant at at interpolation and quadrature points
        det_dF = abs(domain.evaluate(self.pro.pts[0].flatten(), self.pro.pts[1].flatten(), eta3, 'det_df'))
        det_dF = det_dF.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.pts[1].shape[0], self.pro.pts[1].shape[1])
        
        # assemble sparse matrix
        values  = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size, dtype=float)
        row_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size, dtype=int)
        col_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size, dtype=int)
        
        ker.rhs2_2d(self.pi1_x_D_i[0], self.pi1_y_D_i[0], self.pi1_x_D_i[1], self.pi1_y_D_i[1], self.pro.subs[0], self.pro.subs[1], self.pro.subs_cum[0], self.pro.subs_cum[1], self.pro.wts[0], self.pro.wts[1], self.basis_his_D[0], self.basis_his_D[1], self.pro.space.NbaseN, self.pro.space.NbaseD, PR_eq/det_dF, values, row_all, col_all)
        
        self.rhs_PR = self.pro.P3.dot(spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_3form, self.pro.space.Ntot_3form)).dot(self.pro.space.E3.T))
        self.rhs_PR.eliminate_zeros()
        
        
    
    # =================================================================
    def assemble_MR(self, domain, r3_eq):
        
        eta3 = np.array([0.])
        
        if callable(r3_eq):
            rho0_eq = lambda eta1, eta2: r3_eq(eta1, eta2)/abs(domain.evaluate(eta1, eta2, eta3, 'det_df')[:, :, 0])
        else:
            rho0_eq = lambda eta1, eta2: self.pro.space.evaluate_DD(eta1, eta2, r3_eq, 'V3')/abs(domain.evaluate(eta1, eta2, eta3, 'det_df')[:, :, 0])
        
        weight11 = lambda eta1, eta2 :rho0_eq(eta1, eta2)*domain.evaluate(eta1, eta2, eta3, 'g_11')[:, :, 0]
        weight12 = lambda eta1, eta2: rho0_eq(eta1, eta2)*domain.evaluate(eta1, eta2, eta3, 'g_12')[:, :, 0]
        weight13 = lambda eta1, eta2: rho0_eq(eta1, eta2)*domain.evaluate(eta1, eta2, eta3, 'g_13')[:, :, 0]

        weight21 = lambda eta1, eta2: rho0_eq(eta1, eta2)*domain.evaluate(eta1, eta2, eta3, 'g_21')[:, :, 0]
        weight22 = lambda eta1, eta2: rho0_eq(eta1, eta2)*domain.evaluate(eta1, eta2, eta3, 'g_22')[:, :, 0]
        weight23 = lambda eta1, eta2: rho0_eq(eta1, eta2)*domain.evaluate(eta1, eta2, eta3, 'g_23')[:, :, 0]

        weight31 = lambda eta1, eta2: rho0_eq(eta1, eta2)*domain.evaluate(eta1, eta2, eta3, 'g_31')[:, :, 0]
        weight32 = lambda eta1, eta2: rho0_eq(eta1, eta2)*domain.evaluate(eta1, eta2, eta3, 'g_32')[:, :, 0]
        weight33 = lambda eta1, eta2: rho0_eq(eta1, eta2)*domain.evaluate(eta1, eta2, eta3, 'g_33')[:, :, 0]
        
        weights = [[weight11, weight12, weight13], [weight21, weight22, weight23], [weight31, weight32, weight33]]
        
        self.MR = mass.get_M2(self.pro.space, domain, False, weights)
    
    
    # =================================================================
    def assemble_JB_weak(self, domain, b2_eq):
        
        if callable(b2_eq):
            raiseValueError('given equilibrium magnetic field must be given in terms of FEM coefficients')
        
        eta3 = np.array([0.])
        
        # compute C_wn.T.(M2.b2)
        f1 = self.pro.space.C_wn.T.dot(self.pro.space.M2.dot(b2_eq))

        # extract 1C-form and 0-form
        f1_1, f1_3 = np.split(f1, [self.pro.space.E1_pol.shape[0]])

        # apply inverse, transposed interpolation matrices
        f1_1 = self.pro.I1_pol_inv.T.dot(f1_1)
        f1_3 = self.pro.I0_pol_inv.T.dot(f1_3)

        # apply transposed projection extraction operators
        f1_1 = self.pro.P1_pol.T.dot(f1_1)
        f1_3 = self.pro.P0_pol.T.dot(f1_3)

        f1_1, f1_2 = np.split(f1_1, [self.pro.space.Ntot_1form[0]])
        
        # reshape to tensor-product structure
        f1_1 = f1_1.reshape(self.pro.space.NbaseD[0], self.pro.space.NbaseN[1])
        f1_2 = f1_2.reshape(self.pro.space.NbaseN[0], self.pro.space.NbaseD[1])
        f1_3 = f1_3.reshape(self.pro.space.NbaseN[0], self.pro.space.NbaseN[1])
        
        # evaluate Jacobian determinant at point sets
        det_DF_hi = domain.evaluate(self.pro.pts[0].flatten(), self.pro.x_int[1], eta3, 'det_df')[:, :, 0]
        det_DF_ih = domain.evaluate(self.pro.x_int[0], self.pro.pts[1].flatten(), eta3, 'det_df')[:, :, 0]
        det_DF_ii = domain.evaluate(self.pro.x_int[0], self.pro.x_int[1]        , eta3, 'det_df')[:, :, 0]
        
        det_DF_hi = det_DF_hi.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size)
        det_DF_ih = det_DF_ih.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1])
        
        
        # ====================== 12 - block ([int, int] of ND DN) ===========================
        values  = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1), dtype=int)
        
        ker.rhs0_f_2d(self.pi0_x_ND_i, self.pi0_y_DN_i, self.basis_int_N[0], self.basis_int_D[0], self.basis_int_D[1], self.basis_int_N[1], 1/det_DF_ii, f1_3, values, row_all, col_all)
        
        JB_12 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[0], self.pro.space.Ntot_2form[1]))
        
        
        # ====================== 13 - block ([int, his] of ND DD) ===========================
        values  = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1), dtype=int)
        
        ker.rhs12_f_2d(self.pi0_x_ND_i, self.pi1_y_DD_i, self.pro.subs[1], self.pro.subs_cum[1], self.pro.wts[1], self.basis_int_N[0], self.basis_int_D[0], self.basis_his_D[1], self.basis_his_D[1], 1/det_DF_ih, -f1_2, values, row_all, col_all)
        
        JB_13 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[0], self.pro.space.Ntot_2form[2]))
        
        
        # ====================== 21 - block ([int, int] of DN ND) ===========================
        values  = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1), dtype=int)
        
        ker.rhs0_f_2d(self.pi0_x_DN_i, self.pi0_y_ND_i, self.basis_int_D[0], self.basis_int_N[0], self.basis_int_N[1], self.basis_int_D[1], 1/det_DF_ii, -f1_3, values, row_all, col_all)
        
        JB_21 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[1], self.pro.space.Ntot_2form[0]))
        
        
        # ====================== 23 - block ([his, int] of DD ND) ===========================
        values  = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1), dtype=int)
        
        ker.rhs11_f_2d(self.pi1_x_DD_i, self.pi0_y_ND_i, self.pro.subs[0], self.pro.subs_cum[0], self.pro.wts[0], self.basis_his_D[0], self.basis_his_D[0], self.basis_int_N[1], self.basis_int_D[1], 1/det_DF_hi, f1_1, values, row_all, col_all)
        
        JB_23 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[1], self.pro.space.Ntot_2form[2]))
        
        
        # ====================== 31 - block ([int, his] of DN DD) ===========================
        values  = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1), dtype=int)
        
        ker.rhs12_f_2d(self.pi0_x_DN_i, self.pi1_y_DD_i, self.pro.subs[1], self.pro.subs_cum[1], self.pro.wts[1], self.basis_int_D[0], self.basis_int_N[0], self.basis_his_D[1], self.basis_his_D[1], 1/det_DF_ih, f1_2, values, row_all, col_all)
        
        JB_31 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[2], self.pro.space.Ntot_2form[0]))
        
        
        # ====================== 32 - block ([his, int] of DD DN) ===========================
        values  = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1), dtype=int)
        
        ker.rhs11_f_2d(self.pi1_x_DD_i, self.pi0_y_DN_i, self.pro.subs[0], self.pro.subs_cum[0], self.pro.wts[0], self.basis_his_D[0], self.basis_his_D[0], self.basis_int_D[1], self.basis_int_N[1], 1/det_DF_hi, -f1_1, values, row_all, col_all)
        
        JB_32 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[2], self.pro.space.Ntot_2form[1]))
        
        # ==================================== full operator ================================
        self.mat_JB = (self.pro.space.E2.dot(spa.bmat([[None, A_12, A_13], [A_21, None, A_23], [A_31, A_32, None]], format='csr').dot(self.pro.space.E2.T))).T
        self.mat_JB.eliminate_zeros()
        
        
    # =================================================================
    def assemble_JB_strong(self, domain, j2_eq):
        
        eta3 = np.array([0.])
        
        if callable(j2_eq[0]):
            
            weight11 = lambda eta1, eta2:  np.zeros((eta1.size, eta2.size), dtype=float)
            weight12 = lambda eta1, eta2: -j2_eq[2](eta1, eta2)
            weight13 = lambda eta1, eta2:  j2_eq[1](eta1, eta2)
            
            weight21 = lambda eta1, eta2:  j2_eq[2](eta1, eta2)
            weight22 = lambda eta1, eta2:  np.zeros((eta1.size, eta2.size), dtype=float)
            weight23 = lambda eta1, eta2: -j2_eq[0](eta1, eta2)
            
            weight31 = lambda eta1, eta2: -j2_eq[1](eta1, eta2)
            weight32 = lambda eta1, eta2:  j2_eq[0](eta1, eta2)
            weight33 = lambda eta1, eta2:  np.zeros((eta1.size, eta2.size), dtype=float)
        
        else:
            
            weight11 = lambda eta1, eta2:  np.zeros((eta1.size, eta2.size), dtype=float)
            weight12 = lambda eta1, eta2: -self.pro.space.evaluate_DD(eta1, eta2, j2_eq, 'V2')
            weight13 = lambda eta1, eta2:  self.pro.space.evaluate_DN(eta1, eta2, j2_eq, 'V2')
            
            weight21 = lambda eta1, eta2:  self.pro.space.evaluate_DD(eta1, eta2, j2_eq, 'V2')
            weight22 = lambda eta1, eta2:  np.zeros((eta1.size, eta2.size), dtype=float)
            weight23 = lambda eta1, eta2: -self.pro.space.evaluate_ND(eta1, eta2, j2_eq, 'V2')
            
            weight31 = lambda eta1, eta2: -self.pro.space.evaluate_DN(eta1, eta2, j2_eq, 'V2')
            weight32 = lambda eta1, eta2:  self.pro.space.evaluate_ND(eta1, eta2, j2_eq, 'V2')
            weight33 = lambda eta1, eta2:  np.zeros((eta1.size, eta2.size), dtype=float)

        weights = [[weight11, weight12, weight13], [weight21, weight22, weight23], [weight31, weight32, weight33]]

        self.mat_JB = mass.get_M2(self.pro.space, domain, False, weights)
        
        #self.mat_JB = mass.get_M2_a(self.pro.space, domain, j2_eq)
        #self.mat_JB.eliminate_zeros()