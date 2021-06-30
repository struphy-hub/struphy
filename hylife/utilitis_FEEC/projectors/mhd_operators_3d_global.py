# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Class for 3D linear MHD operators.
"""


import numpy as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.bsplines as bsp
import hylife.utilitis_FEEC.projectors.kernels_projectors_global_mhd as ker

import hylife.utilitis_FEEC.basics.mass_matrices_3d     as mass
import hylife.utilitis_FEEC.basics.mass_matrices_3d_pre as mass_pre

import hylife.utilitis_FEEC.projectors.projectors_global as pro

import hylife.geometry.domain_3d as dom


class operators_mhd:
    """
    Global linear MHD operators.
    
    Parameters
    ----------
    projectors_3d : projectors_global_3d
        3d global commuting projectors object
        
    dt : double
        time step
        
    gamma : double
        adiabatic gas index
        
    loc_jeq : string
        whether to solve j_eq x B term in second (step_2) or sixth (step_6) sub-step in time integration scheme
        
    basis_u : int
        formulation for MHD bulk velocity (0: vector field, 1: 1-form, 2: 2-form)
    """
    
    def __init__(self, projectors_3d, dt, gamma, loc_jeq, basis_u):
        
        # 3D projector
        self.pro = projectors_3d
        
        # parameters
        self.dt      = dt
        self.gamma   = gamma
        self.loc_jeq = loc_jeq
        self.basis_u = basis_u
        
        # non-vanishing 1D indices of expressions R0(N), R0(D), R1(N) and R1(D)
        projectors_1d = [pro.projectors_global_1d(space, n_quad) for space, n_quad in zip(self.pro.space.spaces, self.pro.n_quad)]
        
        self.pi0_x_N_i, self.pi0_x_D_i, self.pi1_x_N_i, self.pi1_x_D_i = projectors_1d[0].dofs_1d_bases()
        self.pi0_y_N_i, self.pi0_y_D_i, self.pi1_y_N_i, self.pi1_y_D_i = projectors_1d[1].dofs_1d_bases()
        self.pi0_z_N_i, self.pi0_z_D_i, self.pi1_z_N_i, self.pi1_z_D_i = projectors_1d[2].dofs_1d_bases()
        
        # non-vanishing 1D indices of expressions R0(NN), R0(DN), R0(ND), R0(DD), R1(NN), R1(DN), R1(ND) and R1(DD)
        #self.pi0_x_NN_i, self.pi0_x_DN_i, self.pi0_x_ND_i, self.pi0_x_DD_i, self.pi1_x_NN_i, self.pi1_x_DN_i, self.pi1_x_ND_i, self.pi1_x_DD_i = projectors_1d[0].dofs_1d_bases_products()
        #self.pi0_y_NN_i, self.pi0_y_DN_i, self.pi0_y_ND_i, self.pi0_y_DD_i, self.pi1_y_NN_i, self.pi1_y_DN_i, self.pi1_y_ND_i, self.pi1_y_DD_i = projectors_1d[1].dofs_1d_bases_products()
        #self.pi0_z_NN_i, self.pi0_z_DN_i, self.pi0_z_ND_i, self.pi0_z_DD_i, self.pi1_z_NN_i, self.pi1_z_DN_i, self.pi1_z_ND_i, self.pi1_z_DD_i = projectors_1d[2].dofs_1d_bases_products()
        
        kind_splines = [False, True]
        
        # 1D collocation matrices for interpolation in format (point, global basis function)
        x_int_1d = [np.copy(pro.x_int) for pro in projectors_1d]
        
        self.basis_int_N = [bsp.collocation_matrix(T, p    , x_int, bc, normalize=kind_splines[0]) for T, p, x_int, bc in zip(self.pro.space.T, self.pro.space.p, x_int_1d, self.pro.space.spl_kind)]
        self.basis_int_D = [bsp.collocation_matrix(t, p - 1, x_int, bc, normalize=kind_splines[1]) for t, p, x_int, bc in zip(self.pro.space.t, self.pro.space.p, x_int_1d, self.pro.space.spl_kind)]
        
        # remove small values resulting from round-off errors
        self.basis_int_N[0][self.basis_int_N[0] < 1e-12] = 0.
        self.basis_int_N[1][self.basis_int_N[1] < 1e-12] = 0.
        self.basis_int_N[2][self.basis_int_N[2] < 1e-12] = 0.

        self.basis_int_D[0][self.basis_int_D[0] < 1e-12] = 0.
        self.basis_int_D[1][self.basis_int_D[1] < 1e-12] = 0.
        self.basis_int_D[2][self.basis_int_D[2] < 1e-12] = 0.
                
        # evaluate basis functions at quadrature points in format (interval, local quad. point, global basis function)
        self.basis_his_N = [bsp.collocation_matrix(T, p    , pts.flatten(), bc, normalize=kind_splines[0]).reshape(pts.shape[0], pts.shape[1], NbaseN) for T, p, pts, bc, NbaseN in zip(self.pro.space.T, self.pro.space.p, self.pro.pts, self.pro.space.spl_kind, self.pro.space.NbaseN)]
        self.basis_his_D = [bsp.collocation_matrix(t, p - 1, pts.flatten(), bc, normalize=kind_splines[1]).reshape(pts.shape[0], pts.shape[1], NbaseD) for t, p, pts, bc, NbaseD in zip(self.pro.space.t, self.pro.space.p, self.pro.pts, self.pro.space.spl_kind, self.pro.space.NbaseD)]
    
    
    # =================================================================
    def assemble_rhs_EF(self, domain, b2_eq):
        
        if self.basis_u == 2:

            # ====================== 12 - block ([his, int, int] of DND) ===========================
            # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
            if callable(b2_eq[2]):
                B2_3_pts = b2_eq[2](self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.x_int[2])
            else:
                B2_3_pts = self.pro.space.evaluate_DDN(self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.x_int[2], b2_eq)
                
            B2_3_pts = B2_3_pts.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size, self.pro.x_int[2].size)

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.x_int[2], 'det_df'))
            det_dF = det_dF.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size, self.pro.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_D_i[0].size, dtype=int)

            ker.rhs11(self.pi1_x_D_i[0], self.pi0_y_N_i[0], self.pi0_z_D_i[0], self.pi1_x_D_i[1], self.pi0_y_N_i[1], self.pi0_z_D_i[1], self.pro.subs[0], np.append(0, np.cumsum(self.pro.subs[0] - 1)[:-1]), self.pro.wts[0], self.basis_his_D[0], self.basis_int_N[1], self.basis_int_D[2], self.pro.space.NbaseN, self.pro.space.NbaseD, -B2_3_pts/det_dF, values, row_all, col_all)

            EF_12 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[0], self.pro.space.Ntot_2form[1]))
            
            #self.EF_12 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[0], self.pro.space.Ntot_2form[1]))
            #self.EF_12.eliminate_zeros()

            
            # ====================== 13 - block ([his, int, int] of DDN) ===========================
            # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
            if callable(b2_eq[1]):
                B2_2_pts = b2_eq[1](self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.x_int[2])
            else:
                B2_2_pts = self.pro.space.evaluate_DND(self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.x_int[2], b2_eq)

            B2_2_pts = B2_2_pts.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size, self.pro.x_int[2].size)
            
            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.x_int[2], 'det_df'))
            det_dF = det_dF.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size, self.pro.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs11(self.pi1_x_D_i[0], self.pi0_y_D_i[0], self.pi0_z_N_i[0], self.pi1_x_D_i[1], self.pi0_y_D_i[1], self.pi0_z_N_i[1], self.pro.subs[0], np.append(0, np.cumsum(self.pro.subs[0] - 1)[:-1]), self.pro.wts[0], self.basis_his_D[0], self.basis_int_D[1], self.basis_int_N[2], self.pro.space.NbaseN, self.pro.space.NbaseD,  B2_2_pts/det_dF, values, row_all, col_all)

            EF_13 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[0], self.pro.space.Ntot_2form[2]))
            
            #self.EF_13 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[0], self.pro.space.Ntot_2form[2]))
            #self.EF_13.eliminate_zeros()

            
            # ====================== 21 - block ([int, his, int] of NDD) ===========================
            # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
            if callable(b2_eq[2]):
                B2_3_pts = b2_eq[2](self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.x_int[2])
            else:
                B2_3_pts = self.pro.space.evaluate_DDN(self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.x_int[2], b2_eq)

            B2_3_pts = B2_3_pts.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.x_int[2].size)
            
            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.x_int[2], 'det_df'))
            det_dF = det_dF.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_D_i[0].size, dtype=int)

            ker.rhs12(self.pi0_x_N_i[0], self.pi1_y_D_i[0], self.pi0_z_D_i[0], self.pi0_x_N_i[1], self.pi1_y_D_i[1], self.pi0_z_D_i[1], self.pro.subs[1], np.append(0, np.cumsum(self.pro.subs[1] - 1)[:-1]), self.pro.wts[1], self.basis_int_N[0], self.basis_his_D[1], self.basis_int_D[2], self.pro.space.NbaseN, self.pro.space.NbaseD,  B2_3_pts/det_dF, values, row_all, col_all)

            EF_21 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[1], self.pro.space.Ntot_2form[0]))
            
            #self.EF_21 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[1], self.pro.space.Ntot_2form[0]))
            #self.EF_21.eliminate_zeros()


            # ====================== 23 - block ([int, his, int] of DDN) ===========================
            # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
            if callable(b2_eq[0]):
                B2_1_pts = b2_eq[0](self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.x_int[2])
            else:
                B2_1_pts = self.pro.space.evaluate_NDD(self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.x_int[2], b2_eq)

            B2_1_pts = B2_1_pts.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.x_int[2].size)
            
            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.x_int[2], 'det_df'))
            det_dF = det_dF.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs12(self.pi0_x_D_i[0], self.pi1_y_D_i[0], self.pi0_z_N_i[0], self.pi0_x_D_i[1], self.pi1_y_D_i[1], self.pi0_z_N_i[1], self.pro.subs[1], np.append(0, np.cumsum(self.pro.subs[1] - 1)[:-1]), self.pro.wts[1], self.basis_int_D[0], self.basis_his_D[1], self.basis_int_N[2], self.pro.space.NbaseN, self.pro.space.NbaseD, -B2_1_pts/det_dF, values, row_all, col_all)

            EF_23 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[1], self.pro.space.Ntot_2form[2]))
            
            #self.EF_23 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[1], self.pro.space.Ntot_2form[2]))
            #self.EF_23.eliminate_zeros()


            # ====================== 31 - block ([int, int, his] of NDD) ===========================
            # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
            if callable(b2_eq[1]):
                B2_2_pts = b2_eq[1](self.pro.x_int[0], self.pro.x_int[1], self.pro.pts[2].flatten())
            else:
                B2_2_pts = self.pro.space.evaluate_DND(self.pro.x_int[0], self.pro.x_int[1], self.pro.pts[2].flatten(), b2_eq)

            B2_2_pts = B2_2_pts.reshape(self.pro.x_int[0].size, self.pro.x_int[1].size, self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])
            
            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.x_int[0], self.pro.x_int[1], self.pro.pts[2].flatten(), 'det_df'))
            det_dF = det_dF.reshape(self.pro.x_int[0].size, self.pro.x_int[1].size, self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)

            ker.rhs13(self.pi0_x_N_i[0], self.pi0_y_D_i[0], self.pi1_z_D_i[0], self.pi0_x_N_i[1], self.pi0_y_D_i[1], self.pi1_z_D_i[1], self.pro.subs[2], np.append(0, np.cumsum(self.pro.subs[2] - 1)[:-1]), self.pro.wts[2], self.basis_int_N[0], self.basis_int_D[1], self.basis_his_D[2], self.pro.space.NbaseN, self.pro.space.NbaseD, -B2_2_pts/det_dF, values, row_all, col_all)

            EF_31 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[2], self.pro.space.Ntot_2form[0]))
            
            #self.EF_31 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[2], self.pro.space.Ntot_2form[0]))
            #self.EF_31.eliminate_zeros()

            
            # ====================== 32 - block ([int, int, his] of DND) ===========================
            # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
            if callable(b2_eq[0]):
                B2_1_pts = b2_eq[0](self.pro.x_int[0], self.pro.x_int[1], self.pro.pts[2].flatten())
            else:
                B2_1_pts = self.pro.space.evaluate_NDD(self.pro.x_int[0], self.pro.x_int[1], self.pro.pts[2].flatten(), b2_eq)

            B2_1_pts = B2_1_pts.reshape(self.pro.x_int[0].size, self.pro.x_int[1].size, self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])
            
            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.x_int[0], self.pro.x_int[1], self.pro.pts[2].flatten(), 'det_df'))
            det_dF = det_dF.reshape(self.pro.x_int[0].size, self.pro.x_int[1].size, self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=int)

            ker.rhs13(self.pi0_x_D_i[0], self.pi0_y_N_i[0], self.pi1_z_D_i[0], self.pi0_x_D_i[1], self.pi0_y_N_i[1], self.pi1_z_D_i[1], self.pro.subs[2], np.append(0, np.cumsum(self.pro.subs[2] - 1)[:-1]), self.pro.wts[2], self.basis_int_D[0], self.basis_int_N[1], self.basis_his_D[2], self.pro.space.NbaseN, self.pro.space.NbaseD,  B2_1_pts/det_dF, values, row_all, col_all)

            EF_32 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[2], self.pro.space.Ntot_2form[1]))
            
            #self.EF_32 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[2], self.pro.space.Ntot_2form[1]))
            #self.EF_32.eliminate_zeros()

            # ============================== full operator ==========================================
            self.rhs_EF = self.pro.P1.dot(spa.bmat([[None, EF_12, EF_13], [EF_21, None, EF_23], [EF_31, EF_32, None]]).dot(self.pro.space.E2.T)).tocsr()
            self.rhs_EF.eliminate_zeros()
            
        elif self.basis_u == 0:
            
            # ====================== 12 - block ([his, int, int] of NNN) ===========================
            # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
            if callable(b2_eq[2]):
                B2_3_pts = b2_eq[2](self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.x_int[2])
            else:
                B2_3_pts = self.pro.space.evaluate_DDN(self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.x_int[2], b2_eq)
                
            B2_3_pts = B2_3_pts.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size, self.pro.x_int[2].size)

            # assemble sparse matrix
            values   = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all  = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all  = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs11(self.pi1_x_N_i[0], self.pi0_y_N_i[0], self.pi0_z_N_i[0], self.pi1_x_N_i[1], self.pi0_y_N_i[1], self.pi0_z_N_i[1], self.pro.subs[0], np.append(0, np.cumsum(self.pro.subs[0] - 1)[:-1]), self.pro.wts[0], self.basis_his_N[0], self.basis_int_N[1], self.basis_int_N[2], self.pro.space.NbaseN, self.pro.space.NbaseD, -B2_3_pts, values, row_all, col_all)

            EF_12 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[0], self.pro.space.Ntot_0form))

            
            # ====================== 13 - block ([his, int, int] of NNN) ===========================
            # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
            if callable(b2_eq[1]):
                B2_2_pts = b2_eq[1](self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.x_int[2])
            else:
                B2_2_pts = self.pro.space.evaluate_DND(self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.x_int[2], b2_eq)

            B2_2_pts = B2_2_pts.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size, self.pro.x_int[2].size)

            # assemble sparse matrix
            values   = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all  = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all  = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs11(self.pi1_x_N_i[0], self.pi0_y_N_i[0], self.pi0_z_N_i[0], self.pi1_x_N_i[1], self.pi0_y_N_i[1], self.pi0_z_N_i[1], self.pro.subs[0], np.append(0, np.cumsum(self.pro.subs[0] - 1)[:-1]), self.pro.wts[0], self.basis_his_N[0], self.basis_int_N[1], self.basis_int_N[2], self.pro.space.NbaseN, self.pro.space.NbaseD,  B2_2_pts, values, row_all, col_all)

            EF_13 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[0], self.pro.space.Ntot_0form))


            # ====================== 21 - block ([int, his, int] of NNN) ===========================
            # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
            if callable(b2_eq[2]):
                B2_3_pts = b2_eq[2](self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.x_int[2])
            else:
                B2_3_pts = self.pro.space.evaluate_DDN(self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.x_int[2], b2_eq)

            B2_3_pts = B2_3_pts.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.x_int[2].size)

            # assemble sparse matrix
            values   = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs12(self.pi0_x_N_i[0], self.pi1_y_N_i[0], self.pi0_z_N_i[0], self.pi0_x_N_i[1], self.pi1_y_N_i[1], self.pi0_z_N_i[1], self.pro.subs[1], np.append(0, np.cumsum(self.pro.subs[1] - 1)[:-1]), self.pro.wts[1], self.basis_int_N[0], self.basis_his_N[1], self.basis_int_N[2], self.pro.space.NbaseN, self.pro.space.NbaseD,  B2_3_pts, values, row_all, col_all)

            EF_21 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[1], self.pro.space.Ntot_0form))

            
            # ====================== 23 - block ([int, his, int] of NNN) ===========================
            # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
            if callable(b2_eq[0]):
                B2_1_pts = b2_eq[0](self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.x_int[2])
            else:
                B2_1_pts = self.pro.space.evaluate_NDD(self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.x_int[2], b2_eq)

            B2_1_pts = B2_1_pts.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.x_int[2].size)

            # assemble sparse matrix
            values   = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs12(self.pi0_x_N_i[0], self.pi1_y_N_i[0], self.pi0_z_N_i[0], self.pi0_x_N_i[1], self.pi1_y_N_i[1], self.pi0_z_N_i[1], self.pro.subs[1], np.append(0, np.cumsum(self.pro.subs[1] - 1)[:-1]), self.pro.wts[1], self.basis_int_N[0], self.basis_his_N[1], self.basis_int_N[2], self.pro.space.NbaseN, self.pro.space.NbaseD, -B2_1_pts, values, row_all, col_all)

            EF_23 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[1], self.pro.space.Ntot_0form))


            # ====================== 31 - block ([int, int, his] of NNN) ===========================
            # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
            if callable(b2_eq[1]):
                B2_2_pts = b2_eq[1](self.pro.x_int[0], self.pro.x_int[1], self.pro.pts[2].flatten())
            else:
                B2_2_pts = self.pro.space.evaluate_DND(self.pro.x_int[0], self.pro.x_int[1], self.pro.pts[2].flatten(), b2_eq)

            B2_2_pts = B2_2_pts.reshape(self.pro.x_int[0].size, self.pro.x_int[1].size, self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])

            # assemble sparse matrix
            values   = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=float)
            row_all  = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)
            col_all  = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)

            ker.rhs13(self.pi0_x_N_i[0], self.pi0_y_N_i[0], self.pi1_z_N_i[0], self.pi0_x_N_i[1], self.pi0_y_N_i[1], self.pi1_z_N_i[1], self.pro.subs[2], np.append(0, np.cumsum(self.pro.subs[2] - 1)[:-1]), self.pro.wts[2], self.basis_int_N[0], self.basis_int_N[1], self.basis_his_N[2], self.pro.space.NbaseN, self.pro.space.NbaseD, -B2_2_pts, values, row_all, col_all)

            EF_31 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[2], self.pro.space.Ntot_0form))

            
            # ====================== 32 - block ([int, int, his] of NNN) ===========================
            # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
            if callable(b2_eq[0]):
                B2_1_pts = b2_eq[0](self.pro.x_int[0], self.pro.x_int[1], self.pro.pts[2].flatten())
            else:
                B2_1_pts = self.pro.space.evaluate_NDD(self.pro.x_int[0], self.pro.x_int[1], self.pro.pts[2].flatten(), b2_eq)

            B2_1_pts = B2_1_pts.reshape(self.pro.x_int[0].size, self.pro.x_int[1].size, self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])

            # assemble sparse matrix
            values   = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=float)
            row_all  = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)
            col_all  = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)

            ker.rhs13(self.pi0_x_N_i[0], self.pi0_y_N_i[0], self.pi1_z_N_i[0], self.pi0_x_N_i[1], self.pi0_y_N_i[1], self.pi1_z_N_i[1], self.pro.subs[2], np.append(0, np.cumsum(self.pro.subs[2] - 1)[:-1]), self.pro.wts[2], self.basis_int_N[0], self.basis_int_N[1], self.basis_his_N[2], self.pro.space.NbaseN, self.pro.space.NbaseD,  B2_1_pts, values, row_all, col_all)

            EF_32 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_1form[2], self.pro.space.Ntot_0form))
            
            # ============================== full operator ==========================================
            E_temp = spa.bmat([[self.pro.space.E0, None, None], [None, self.pro.space.E0_all, None], [None, None, self.pro.space.E0_all]], format='csr')
            
            self.rhs_EF = self.pro.P1.dot(spa.bmat([[None, EF_12, EF_13], [EF_21, None, EF_23], [EF_31, EF_32, None]]).dot(E_temp.T)).tocsr()
            self.rhs_EF.eliminate_zeros()
            
    
    
    
    # =================================================================
    def assemble_rhs_F(self, domain, which, c3_eq=None):
        
        """
        which = m (mass)
        which = p (pressure)
        which = j (jacobian)
        
        which = m and basis_u = 2 --> EQ = rho3/|det_dF|
        which = p and basis_u = 2 --> EQ = p3/|det_dF|
        
        which = m and basis_u = 0 --> EQ = rho3
        which = p and basis_u = 0 --> EQ = p3
        which = j and basis_u = 0 --> EQ = |det_dF|
        """
        
        if self.basis_u == 2:

            # ====================== 11 - block ([int, his, his] of NDD) ===========================
            # evaluate equilibrium density/pressure at interpolation and quadrature points
            if callable(c3_eq):
                EQ = c3_eq(self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.pts[2].flatten())
            else:
                EQ = self.pro.space.evaluate_DDD(self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.pts[2].flatten(), c3_eq)

            EQ = EQ.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])
            
            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.pts[2].flatten(), 'det_df'))
            det_dF = det_dF.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)

            ker.rhs21(self.pi0_x_N_i[0], self.pi1_y_D_i[0], self.pi1_z_D_i[0], self.pi0_x_N_i[1], self.pi1_y_D_i[1], self.pi1_z_D_i[1], self.pro.subs[1], self.pro.subs[2], np.append(0, np.cumsum(self.pro.subs[1] - 1)[:-1]), np.append(0, np.cumsum(self.pro.subs[2] - 1)[:-1]), self.pro.wts[1], self.pro.wts[2], self.basis_int_N[0], self.basis_his_D[1], self.basis_his_D[2], self.pro.space.NbaseN, self.pro.space.NbaseD, EQ/det_dF, values, row_all, col_all)

            F_11 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[0], self.pro.space.Ntot_2form[0]))
            
            #self.F_11 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[0], self.pro.space.Ntot_2form[0]))
            #self.F_11.eliminate_zeros()


            # ====================== 22 - block ([his, int, his] of DND) ===========================
            # evaluate equilibrium density/pressure at interpolation and quadrature points
            if callable(c3_eq):
                EQ = c3_eq(self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.pts[2].flatten())
            else:
                EQ = self.pro.space.evaluate_DDD(self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.pts[2].flatten(), c3_eq)
                
            EQ = EQ.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size, self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.pts[2].flatten(), 'det_df'))
            det_dF = det_dF.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size, self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=int)

            ker.rhs22(self.pi1_x_D_i[0], self.pi0_y_N_i[0], self.pi1_z_D_i[0], self.pi1_x_D_i[1], self.pi0_y_N_i[1], self.pi1_z_D_i[1], self.pro.subs[0], self.pro.subs[2], np.append(0, np.cumsum(self.pro.subs[0] - 1)[:-1]), np.append(0, np.cumsum(self.pro.subs[2] - 1)[:-1]), self.pro.wts[0], self.pro.wts[2], self.basis_his_D[0], self.basis_int_N[1], self.basis_his_D[2], self.pro.space.NbaseN, self.pro.space.NbaseD, EQ/det_dF, values, row_all, col_all)

            F_22 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[1], self.pro.space.Ntot_2form[1]))
            
            #self.F_22 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[1], self.pro.space.Ntot_2form[1]))
            #self.F_22.eliminate_zeros()


            # ====================== 33 - block ([his, his, int] of DDN) ===========================
            # evaluate equilibrium density/pressure at interpolation and quadrature points
            if callable(c3_eq):
                EQ = c3_eq(self.pro.pts[0].flatten(), self.pro.pts[1].flatten(), self.pro.x_int[2])
            else:
                EQ = self.pro.space.evaluate_DDD(self.pro.pts[0].flatten(), self.pro.pts[1].flatten(), self.pro.x_int[2], c3_eq)

            EQ = EQ.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.x_int[2].size)
            
            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(domain.evaluate(self.pro.pts[0].flatten(), self.pro.pts[1].flatten(), self.pro.x_int[2], 'det_df'))
            det_dF = det_dF.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs23(self.pi1_x_D_i[0], self.pi1_y_D_i[0], self.pi0_z_N_i[0], self.pi1_x_D_i[1], self.pi1_y_D_i[1], self.pi0_z_N_i[1], self.pro.subs[0], self.pro.subs[1], np.append(0, np.cumsum(self.pro.subs[0] - 1)[:-1]), np.append(0, np.cumsum(self.pro.subs[1] - 1)[:-1]), self.pro.wts[0], self.pro.wts[1], self.basis_his_D[0], self.basis_his_D[1], self.basis_int_N[2], self.pro.space.NbaseN, self.pro.space.NbaseD, EQ/det_dF, values, row_all, col_all)

            F_33 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[2], self.pro.space.Ntot_2form[2]))
            
            #self.F_33 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[2], self.pro.space.Ntot_2form[2]))
            #self.F_33.eliminate_zeros()
        
            # ==================================== full operator ===================================
            if   which == 'm':
                self.rhs_MF = self.pro.P2.dot(spa.bmat([[F_11, None, None], [None, F_22, None], [None, None, F_33]]).dot(self.pro.space.E2.T)).tocsr()
                self.rhs_MF.eliminate_zeros()
                
            elif which == 'p':
                self.rhs_PF = self.pro.P2.dot(spa.bmat([[F_11, None, None], [None, F_22, None], [None, None, F_33]]).dot(self.pro.space.E2.T)).tocsr()
                self.rhs_PF.eliminate_zeros()
                
        elif self.basis_u == 0:
            
            # ====================== 11 - block ([int, his, his] of NNN) ===========================
            # evaluate equilibrium density/pressure at interpolation and quadrature points
            if which == 'j':
                EQ = domain.evaluate(self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.pts[2].flatten(), 'det_df')
            else:
                if callable(c3_eq):
                    EQ = c3_eq(self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.pts[2].flatten())
                else:
                    EQ = self.pro.space.evaluate_DDD(self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.pts[2].flatten(), c3_eq)

            EQ = EQ.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)

            ker.rhs21(self.pi0_x_N_i[0], self.pi1_y_N_i[0], self.pi1_z_N_i[0], self.pi0_x_N_i[1], self.pi1_y_N_i[1], self.pi1_z_N_i[1], self.pro.subs[1], self.pro.subs[2], np.append(0, np.cumsum(self.pro.subs[1] - 1)[:-1]), np.append(0, np.cumsum(self.pro.subs[2] - 1)[:-1]), self.pro.wts[1], self.pro.wts[2], self.basis_int_N[0], self.basis_his_N[1], self.basis_his_N[2], self.pro.space.NbaseN, self.pro.space.NbaseD, EQ, values, row_all, col_all)

            F_11 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[0], self.pro.space.Ntot_0form))


            # ====================== 22 - block ([his, int, his] of NNN) ===========================
            # evaluate equilibrium density/pressure at interpolation and quadrature points
            if which == 'j':
                EQ = domain.evaluate(self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.pts[2].flatten(), 'det_df')
            else:
                if callable(c3_eq):
                    EQ = c3_eq(self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.pts[2].flatten())
                else:
                    EQ = self.pro.space.evaluate_DDD(self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.pts[2].flatten(), c3_eq)
                
            EQ = EQ.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size, self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)

            ker.rhs22(self.pi1_x_N_i[0], self.pi0_y_N_i[0], self.pi1_z_N_i[0], self.pi1_x_N_i[1], self.pi0_y_N_i[1], self.pi1_z_N_i[1], self.pro.subs[0], self.pro.subs[2], np.append(0, np.cumsum(self.pro.subs[0] - 1)[:-1]), np.append(0, np.cumsum(self.pro.subs[2] - 1)[:-1]), self.pro.wts[0], self.pro.wts[2], self.basis_his_N[0], self.basis_int_N[1], self.basis_his_N[2], self.pro.space.NbaseN, self.pro.space.NbaseD, EQ, values, row_all, col_all)

            F_22 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[1], self.pro.space.Ntot_0form))


            # ====================== 33 - block ([his, his, int] of NNN) ===========================
            # evaluate equilibrium density/pressure at interpolation and quadrature points
            if which == 'j':
                EQ = domain.evaluate(self.pro.pts[0].flatten(), self.pro.pts[1].flatten(), self.pro.x_int[2], 'det_df')
            else:
                if callable(c3_eq):
                    EQ = c3_eq(self.pro.pts[0].flatten(), self.pro.pts[1].flatten(), self.pro.x_int[2])
                else:
                    EQ = self.pro.space.evaluate_DDD(self.pro.pts[0].flatten(), self.pro.pts[1].flatten(), self.pro.x_int[2], c3_eq)

            EQ = EQ.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs23(self.pi1_x_N_i[0], self.pi1_y_N_i[0], self.pi0_z_N_i[0], self.pi1_x_N_i[1], self.pi1_y_N_i[1], self.pi0_z_N_i[1], self.pro.subs[0], self.pro.subs[1], np.append(0, np.cumsum(self.pro.subs[0] - 1)[:-1]), np.append(0, np.cumsum(self.pro.subs[1] - 1)[:-1]), self.pro.wts[0], self.pro.wts[1], self.basis_his_N[0], self.basis_his_N[1], self.basis_int_N[2], self.pro.space.NbaseN, self.pro.space.NbaseD, EQ, values, row_all, col_all)

            F_33 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[2], self.pro.space.Ntot_0form))
        
            # ==================================== full operator ===================================
            E_temp = spa.bmat([[self.pro.space.E0, None, None], [None, self.pro.space.E0_all, None], [None, None, self.pro.space.E0_all]], format='csr')
            
            if   which == 'm':
                self.rhs_MF = self.pro.P2.dot(spa.bmat([[F_11, None, None], [None, F_22, None], [None, None, F_33]]).dot(E_temp.T)).tocsr()
                self.rhs_MF.eliminate_zeros()
                
            elif which == 'p':
                self.rhs_PF = self.pro.P2.dot(spa.bmat([[F_11, None, None], [None, F_22, None], [None, None, F_33]]).dot(E_temp.T)).tocsr()
                self.rhs_PF.eliminate_zeros()
                
            elif which == 'j':
                self.rhs_JF = self.pro.P2.dot(spa.bmat([[F_11, None, None], [None, F_22, None], [None, None, F_33]]).dot(E_temp.T)).tocsr()
                self.rhs_JF.eliminate_zeros()
            
    
    
    # =================================================================
    def assemble_rhs_W(self, domain, r3_eq):
        
        # ====================== ([int, int, int] of NNN) ===========================
        # evaluate equilibrium density at quadrature points
        if callable(r3_eq):
            EQ = r3_eq(self.pro.x_int[0], self.pro.x_int[1], self.pro.x_int[2])
        else:
            EQ = self.pro.space.evaluate_DDD(self.pro.x_int[0], self.pro.x_int[1], self.pro.x_int[2], r3_eq)
        
        # evaluate Jacobian determinant at interpolation points
        det_dF = abs(domain.evaluate(self.pro.x_int[0], self.pro.x_int[1], self.pro.x_int[2], 'det_df'))
        
        # assemble sparse matrix
        values  = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
        row_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
        col_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
        
        ker.rhs0(self.pi0_x_N_i[0], self.pi0_y_N_i[0], self.pi0_z_N_i[0], self.pi0_x_N_i[1], self.pi0_y_N_i[1], self.pi0_z_N_i[1], self.basis_int_N[0], self.basis_int_N[1], self.basis_int_N[2], EQ/det_dF, values, row_all, col_all)
        
        rhs_W = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_0form, self.pro.space.Ntot_0form))
        rhs_W.eliminate_zeros()
        
        self.rhs_W11 = self.pro.P0.dot(rhs_W.dot(self.pro.space.E0.T)).tocsr()
        self.rhs_W22 = self.pro.P0_all.dot(rhs_W.dot(self.pro.space.E0_all.T)).tocsr()
        self.rhs_W33 = self.pro.P0_all.dot(rhs_W.dot(self.pro.space.E0_all.T)).tocsr()
        
    
    # =================================================================
    def assemble_rhs_PR(self, domain, p3_eq):
            
        # ====================== ([his, his, his] of DDD) ===========================
        # evaluate equilibrium pressure at quadrature points
        if callable(p3_eq):
            PR_eq = p3_eq(self.pro.pts[0].flatten(), self.pro.pts[1].flatten(), self.pro.pts[2].flatten())
        else:
            PR_eq = self.pro.space.evaluate_DDD(self.pro.pts[0].flatten(), self.pro.pts[1].flatten(), self.pro.pts[2].flatten(), p3_eq)
            
        PR_eq = PR_eq.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])
        
        # evaluate Jacobian determinant at at interpolation and quadrature points
        det_dF = abs(domain.evaluate(self.pro.pts[0].flatten(), self.pro.pts[1].flatten(), self.pro.pts[2].flatten(), 'det_df'))
        det_dF = det_dF.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])
        
        # assemble sparse matrix
        values  = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
        row_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
        col_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
        
        ker.rhs3(self.pi1_x_D_i[0], self.pi1_y_D_i[0], self.pi1_z_D_i[0], self.pi1_x_D_i[1], self.pi1_y_D_i[1], self.pi1_z_D_i[1], self.pro.subs[0], self.pro.subs[1], self.pro.subs[2], np.append(0, np.cumsum(self.pro.subs[0] - 1)[:-1]), np.append(0, np.cumsum(self.pro.subs[1] - 1)[:-1]), np.append(0, np.cumsum(self.pro.subs[2] - 1)[:-1]), self.pro.wts[0], self.pro.wts[1], self.pro.wts[2], self.basis_his_D[0], self.basis_his_D[1], self.basis_his_D[2], self.pro.space.NbaseN, self.pro.space.NbaseD, PR_eq/det_dF, values, row_all, col_all)
        
        self.rhs_PR = self.pro.P3.dot(spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_3form, self.pro.space.Ntot_3form)).dot(self.pro.space.E3.T)).tocsr()
        self.rhs_PR.eliminate_zeros()
        
    
    # =================================================================
    def assemble_MR(self, domain, r3_eq):
        
        if callable(r3_eq):
            rho0_eq = lambda eta1, eta2, eta3: r3_eq(eta1, eta2, eta3)/abs(domain.evaluate(eta1, eta2, eta3, 'det_df'))
        else:
            rho0_eq = lambda eta1, eta2, eta3: self.pro.space.evaluate_DDD(eta1, eta2, eta3, r3_eq)/abs(domain.evaluate(eta1, eta2, eta3, 'det_df'))
                                                                                                        
        weight11 = lambda eta1, eta2, eta3 :rho0_eq(eta1, eta2, eta3)*domain.evaluate(eta1, eta2, eta3, 'g_11')
        weight12 = lambda eta1, eta2, eta3: rho0_eq(eta1, eta2, eta3)*domain.evaluate(eta1, eta2, eta3, 'g_12')
        weight13 = lambda eta1, eta2, eta3: rho0_eq(eta1, eta2, eta3)*domain.evaluate(eta1, eta2, eta3, 'g_13')

        weight21 = lambda eta1, eta2, eta3: rho0_eq(eta1, eta2, eta3)*domain.evaluate(eta1, eta2, eta3, 'g_21')
        weight22 = lambda eta1, eta2, eta3: rho0_eq(eta1, eta2, eta3)*domain.evaluate(eta1, eta2, eta3, 'g_22')
        weight23 = lambda eta1, eta2, eta3: rho0_eq(eta1, eta2, eta3)*domain.evaluate(eta1, eta2, eta3, 'g_23')

        weight31 = lambda eta1, eta2, eta3: rho0_eq(eta1, eta2, eta3)*domain.evaluate(eta1, eta2, eta3, 'g_31')
        weight32 = lambda eta1, eta2, eta3: rho0_eq(eta1, eta2, eta3)*domain.evaluate(eta1, eta2, eta3, 'g_32')
        weight33 = lambda eta1, eta2, eta3: rho0_eq(eta1, eta2, eta3)*domain.evaluate(eta1, eta2, eta3, 'g_33')
        
        weights = [[weight11, weight12, weight13], [weight21, weight22, weight23], [weight31, weight32, weight33]]
        
        if   self.basis_u == 2:
            self.MR = mass.get_M2(self.pro.space, domain   , weights)
        elif self.basis_u == 0:
            self.MR = mass.get_Mv(self.pro.space, domain, 0, weights)
    
    
    # =================================================================
    def assemble_JB_weak(self, domain, b2_eq):
        
        if callable(b2_eq[0]):
            raise ValueError('given equilibrium magnetic field must be 2-form coefficients and not a callable!')
        
        # compute C.T(M2(b2_eq)) in weak Lorentz force term
        f1 = self.pro.space.C.T.dot(self.pro.space.M2.dot(b2_eq))
        
        # apply transposed projection extraction operator
        f1_1, f1_2, f1_3 = np.split(self.pro.P1.T.dot(self.pro.apply_IinvT_V1(f1)), [self.pro.space.Ntot_1form_cum[0], self.pro.space.Ntot_1form_cum[1]])
        
        # evaluate Jacobian determinant at point sets
        det_DF_hii = domain.evaluate(self.pro.pts[0].flatten(), self.pro.x_int[1], self.pro.x_int[2], 'det_df')
        det_DF_ihi = domain.evaluate(self.pro.x_int[0], self.pro.pts[1].flatten(), self.pro.x_int[2], 'det_df')
        det_DF_iih = domain.evaluate(self.pro.x_int[0], self.pro.x_int[1], self.pro.pts[2].flatten(), 'det_df')
        
        det_DF_hii = det_DF_hii.reshape(self.pro.pts[0].shape[0], self.pro.pts[0].shape[1], self.pro.x_int[1].size, self.pro.x_int[2].size)
        det_DF_ihi = det_DF_ihi.reshape(self.pro.x_int[0].size, self.pro.pts[1].shape[0], self.pro.pts[1].shape[1], self.pro.x_int[2].size)
        det_DF_iih = det_DF_iih.reshape(self.pro.x_int[0].size, self.pro.x_int[1].size, self.pro.pts[2].shape[0], self.pro.pts[2].shape[1])
        
        
        # ====================== 12 - block ([int, int, his] of ND DN DD) ===========================
        
        # assemble sparse matrix
        values  = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1)*(self.pi1_z_DD_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1)*(self.pi1_z_DD_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1)*(self.pi1_z_DD_i[3].max() + 1), dtype=int)
        
        ker.rhs13_f(self.pi0_x_ND_i, self.pi0_y_DN_i, self.pi1_z_DD_i, self.pro.subs[2], np.append(0, np.cumsum(self.pro.subs[2] - 1)[:-1]), self.pro.wts[2], self.basis_int_N[0], self.basis_int_D[0], self.basis_int_D[1], self.basis_int_N[1], self.basis_his_D[2], self.basis_his_D[2], 1/det_DF_iih, f1_3, values, row_all, col_all)
        
        JB_12 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[0], self.pro.space.Ntot_2form[1]))
        
        
        # ====================== 13 - block ([int, his, int] of ND DD DN) ===========================
        
        # assemble sparse matrix
        values  = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1)*(self.pi0_z_DN_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1)*(self.pi0_z_DN_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1)*(self.pi0_z_DN_i[3].max() + 1), dtype=int)
        
        ker.rhs12_f(self.pi0_x_ND_i, self.pi1_y_DD_i, self.pi0_z_DN_i, self.pro.subs[1], np.append(0, np.cumsum(self.pro.subs[1] - 1)[:-1]), self.pro.wts[1], self.basis_int_N[0], self.basis_int_D[0], self.basis_his_D[1], self.basis_his_D[1], self.basis_int_D[2], self.basis_int_N[2], 1/det_DF_ihi, -f1_2, values, row_all, col_all)
        
        JB_13 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[0], self.pro.space.Ntot_2form[2]))
        
        # ====================== 21 - block ([int, int, his] of DN ND DD) ===========================
        
        # assemble sparse matrix
        values  = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1)*(self.pi1_z_DD_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1)*(self.pi1_z_DD_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1)*(self.pi1_z_DD_i[3].max() + 1), dtype=int)
        
        ker.rhs13_f(self.pi0_x_DN_i, self.pi0_y_ND_i, self.pi1_z_DD_i, self.pro.subs[2], np.append(0, np.cumsum(self.pro.subs[2] - 1)[:-1]), self.pro.wts[2], self.basis_int_D[0], self.basis_int_N[0], self.basis_int_N[1], self.basis_int_D[1], self.basis_his_D[2], self.basis_his_D[2], 1/det_DF_iih, -f1_3, values, row_all, col_all)
        
        JB_21 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[1], self.pro.space.Ntot_2form[0]))
        
        
        # ====================== 23 - block ([his, int, int] of DD ND DN) ===========================
        
        # assemble sparse matrix
        values  = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1)*(self.pi0_z_DN_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1)*(self.pi0_z_DN_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1)*(self.pi0_z_DN_i[3].max() + 1), dtype=int)
        
        ker.rhs11_f(self.pi1_x_DD_i, self.pi0_y_ND_i, self.pi0_z_DN_i, self.pro.subs[0], np.append(0, np.cumsum(self.pro.subs[0] - 1)[:-1]), self.pro.wts[0], self.basis_his_D[0], self.basis_his_D[0], self.basis_int_N[1], self.basis_int_D[1], self.basis_int_D[2], self.basis_int_N[2], 1/det_DF_hii, f1_1, values, row_all, col_all)
        
        JB_23 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[1], self.pro.space.Ntot_2form[2]))
        
        
        # ====================== 31 - block ([int, his, int] of DN DD ND) ===========================

        # assemble sparse matrix
        values  = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1)*(self.pi0_z_ND_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1)*(self.pi0_z_ND_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1)*(self.pi0_z_ND_i[3].max() + 1), dtype=int)
        
        ker.rhs12_f(self.pi0_x_DN_i, self.pi1_y_DD_i, self.pi0_z_ND_i, self.pro.subs[1], np.append(0, np.cumsum(self.pro.subs[1] - 1)[:-1]), self.pro.wts[1], self.basis_int_D[0], self.basis_int_N[0], self.basis_his_D[1], self.basis_his_D[1], self.basis_int_N[2], self.basis_int_D[2], 1/det_DF_ihi, f1_2, values, row_all, col_all)
        
        JB_31 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[2], self.pro.space.Ntot_2form[0]))
        
        
        # ====================== 32 - block ([his, int, int] of DD DN ND) ===========================
        
        # assemble sparse matrix
        values  = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1)*(self.pi0_z_ND_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1)*(self.pi0_z_ND_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1)*(self.pi0_z_ND_i[3].max() + 1), dtype=int)
        
        ker.rhs11_f(self.pi1_x_DD_i, self.pi0_y_DN_i, self.pi0_z_ND_i, self.pro.subs[0], np.append(0, np.cumsum(self.pro.subs[0] - 1)[:-1]), self.pro.wts[0], self.basis_his_D[0], self.basis_his_D[0], self.basis_int_D[1], self.basis_int_N[1], self.basis_int_N[2], self.basis_int_D[2], 1/det_DF_hii, -f1_1, values, row_all, col_all)
        
        JB_32 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.pro.space.Ntot_2form[2], self.pro.space.Ntot_2form[1]))
        
        # ==================================== full operator ========================================
        self.mat_JB = (self.pro.space.E2.dot(spa.bmat([[None, JB_12, JB_13], [JB_21, None, JB_23], [JB_31, JB_32, None]], format='csr').dot(self.pro.space.E2.T))).T
        self.mat_JB.eliminate_zeros()
    
    
    # =================================================================
    def assemble_JB_strong(self, domain, j2_eq):
        
        if callable(j2_eq[0]):
            
            weight11 = lambda eta1, eta2, eta3:  np.zeros((eta1.size, eta2.size, eta3.size), dtype=float)
            weight12 = lambda eta1, eta2, eta3: -j2_eq[2](eta1, eta2, eta3)
            weight13 = lambda eta1, eta2, eta3:  j2_eq[1](eta1, eta2, eta3)
            
            weight21 = lambda eta1, eta2, eta3:  j2_eq[2](eta1, eta2, eta3)
            weight22 = lambda eta1, eta2, eta3:  np.zeros((eta1.size, eta2.size, eta3.size), dtype=float)
            weight23 = lambda eta1, eta2, eta3: -j2_eq[0](eta1, eta2, eta3)
            
            weight31 = lambda eta1, eta2, eta3: -j2_eq[1](eta1, eta2, eta3)
            weight32 = lambda eta1, eta2, eta3:  j2_eq[0](eta1, eta2, eta3)
            weight33 = lambda eta1, eta2, eta3:  np.zeros((eta1.size, eta2.size, eta3.size), dtype=float)
        
        else:
            
            weight11 = lambda eta1, eta2, eta3:  np.zeros((eta1.size, eta2.size, eta3.size), dtype=float)
            weight12 = lambda eta1, eta2, eta3: -self.pro.space.evaluate_DDN(eta1, eta2, eta3, j2_eq)
            weight13 = lambda eta1, eta2, eta3:  self.pro.space.evaluate_DND(eta1, eta2, eta3, j2_eq)
            
            weight21 = lambda eta1, eta2, eta3:  self.pro.space.evaluate_DDN(eta1, eta2, eta3, j2_eq)
            weight22 = lambda eta1, eta2, eta3:  np.zeros((eta1.size, eta2.size, eta3.size), dtype=float)
            weight23 = lambda eta1, eta2, eta3: -self.pro.space.evaluate_NDD(eta1, eta2, eta3, j2_eq)
            
            weight31 = lambda eta1, eta2, eta3: -self.pro.space.evaluate_DND(eta1, eta2, eta3, j2_eq)
            weight32 = lambda eta1, eta2, eta3:  self.pro.space.evaluate_NDD(eta1, eta2, eta3, j2_eq)
            weight33 = lambda eta1, eta2, eta3:  np.zeros((eta1.size, eta2.size, eta3.size), dtype=float)

        weights = [[weight11, weight12, weight13], [weight21, weight22, weight23], [weight31, weight32, weight33]]
        
        if  self.basis_u == 2:
            self.mat_JB = mass.get_M2(self.pro.space, domain   , weights)
        elif self.basis_u == 0:
            self.mat_JB = mass.get_Mv(self.pro.space, domain, 0, weights)
        
        #self.mat_JB = mass.get_M2_a(self.pro.space, domain, j2_eq)
        #self.mat_JB.eliminate_zeros()

        
    # ======================================
    def __EF(self, u):
        return self.pro.solve_V1(False, self.rhs_EF.dot(u))
    
    # ======================================
    def __EF_transposed(self, e):
        return self.rhs_EF.T.dot(self.pro.apply_IinvT_V1(e))
    
    # ======================================
    def __MF(self, u):
        return self.pro.solve_V2(False, self.rhs_MF.dot(u))
    
    # ======================================
    def __MF_transposed(self, u):
        return self.rhs_MF.T.dot(self.pro.apply_IinvT_V2(u))
    
    # ======================================
    def __PF(self, u):
        return self.pro.solve_V2(False, self.rhs_PF.dot(u))
    
    # ======================================
    def __PF_transposed(self, u):
        return self.rhs_PF.T.dot(self.pro.apply_IinvT_V2(u))
    
    # ======================================
    def __JF(self, u):
        return self.pro.solve_V2(False, self.rhs_JF.dot(u))
    
    # ======================================
    def __JF_transposed(self, u):
        return self.rhs_JF.T.dot(self.pro.apply_IinvT_V2(u))
    
    # ======================================
    def __PR(self, f3):
        return self.pro.solve_V3(False, self.rhs_PR.dot(f3))
    
    # ======================================
    def __PR_transposed(self, f3):
        return self.rhs_PR.T.dot(self.pro.apply_IinvT_V3(f3))
    
    # ======================================
    def __JB(self, b2):
        return self.mat_JB.dot(b2)
    
    # ======================================
    def __JB_transposed(self, u2):
        return self.mat_JB.T.dot(u2)
    
    # ======================================
    def __W(self, u):
        
        u1 = u[:self.pro.space.E0.shape[0]]
        u2 = u[ self.pro.space.E0.shape[0]:self.pro.space.E0.shape[0] + self.pro.space.E0_all.shape[0]]
        u3 = u[ self.pro.space.E0.shape[0] + self.pro.space.E0_all.shape[0]:]
        
        a = self.pro.solve_V0(False, self.rhs_W11.dot(u1))
        b = self.pro.solve_V0(True , self.rhs_W22.dot(u2))
        c = self.pro.solve_V0(True , self.rhs_W33.dot(u3))
        
        return np.concatenate((a, b, c))
    
    # ======================================
    def __W_transposed(self, u):
        
        u1 = u[:self.pro.space.E0.shape[0]]
        u2 = u[ self.pro.space.E0.shape[0]:self.pro.space.E0.shape[0] + self.pro.space.E0_all.shape[0]]
        u3 = u[ self.pro.space.E0.shape[0] + self.pro.space.E0_all.shape[0]:]
        
        a = self.rhs_W11.T.dot(self.pro.apply_IinvT_V0(u1, False))
        b = self.rhs_W22.T.dot(self.pro.apply_IinvT_V0(u2, True))
        c = self.rhs_W33.T.dot(self.pro.apply_IinvT_V0(u3, True))
        
        return np.concatenate((a, b, c))
    
    # ======================================
    def __A(self, u):
        
        if self.basis_u == 2:
            #return 1/2*(self.__MF_transposed(self.pro.space.M2.dot(u)) + self.pro.space.M2.dot(self.__MF(u)))
            return self.MR.dot(u)
        
        elif self.basis_u == 0:
            #return 1/2*(self.__W_transposed(self.pro.space.Mv.dot(u)) + self.pro.space.Mv.dot(self.__W(u)))
            return self.MR.dot(u)
    
    # ======================================
    def __L(self, u):
        if self.basis_u == 2:
            return -self.pro.space.D.dot(self.__PF(u)) - (self.gamma - 1)*self.__PR(self.pro.space.D.dot(u))
        elif self.basis_u == 0:
            return -self.pro.space.D.dot(self.__PF(u)) - (self.gamma - 1)*self.__PR(self.pro.space.D.dot(self.__JF(u)))
    
    # ======================================
    def __S2(self, u):
        
        bu   = self.pro.space.C.dot(self.__EF(u))
        
        out  = self.__A(u)
        out += self.dt**2/4*self.__EF_transposed(self.pro.space.C.T.dot(self.pro.space.M2.dot(bu)))
        
        # with additional J_eq x B
        if self.loc_jeq == 'step_2':
            
            out += self.dt**2/4*self.__JB(bu)

        return out
    
    # ======================================
    def __S6(self, u):
        
        out = self.__A(u)
        
        if   self.basis_u == 2:
            out -= self.dt**2/4*self.pro.space.D.T.dot(self.pro.space.M3.dot(self.__L(u)))
        elif self.basis_u == 0:
            out -= self.dt**2/4*self.__JF_transposed(self.pro.space.D.T.dot(self.pro.space.M3.dot(self.__L(u))))

        return out
    
    # ======================================
    def setOperators(self):
        
        self.MF = spa.linalg.LinearOperator((self.pro.space.E2.shape[0], self.pro.space.E2.shape[0]), matvec=self.__MF, rmatvec=self.__MF_transposed)
        
        self.PF = spa.linalg.LinearOperator((self.pro.space.E2.shape[0], self.pro.space.E2.shape[0]), matvec=self.__PF, rmatvec=self.__PF_transposed)
        
        if self.basis_u == 0:
            self.JF = spa.linalg.LinearOperator((self.pro.space.E2.shape[0], self.pro.space.E0.shape[0] + 2*self.pro.space.E0_all.shape[0]), matvec=self.__JF, rmatvec=self.__JF_transposed)
        
        self.EF = spa.linalg.LinearOperator((self.pro.space.E1.shape[0], self.pro.space.E2.shape[0]), matvec=self.__EF, rmatvec=self.__EF_transposed)
        
        self.PR = spa.linalg.LinearOperator((self.pro.space.E3.shape[0], self.pro.space.E3.shape[0]), matvec=self.__PR, rmatvec=self.__PR_transposed)
        
        self.JB = spa.linalg.LinearOperator((self.pro.space.E2.shape[0], self.pro.space.E2.shape[0]), matvec=self.__JB, rmatvec=self.__JB_transposed)
        
        self.A  = spa.linalg.LinearOperator((self.pro.space.E2.shape[0], self.pro.space.E2.shape[0]), matvec=self.__A)
        
        self.L  = spa.linalg.LinearOperator((self.pro.space.E3.shape[0], self.pro.space.E2.shape[0]), matvec=self.__L)
        
        self.S2 = spa.linalg.LinearOperator((self.pro.space.E2.shape[0], self.pro.space.E2.shape[0]), matvec=self.__S2)
        
        self.S6 = spa.linalg.LinearOperator((self.pro.space.E2.shape[0], self.pro.space.E2.shape[0]), matvec=self.__S6)
        
    
    # ======================================
    def RHS2(self, u, b):
        
        bu   = self.pro.space.C.dot(self.EF(u))
        
        out  = self.A(u)
        out -= self.dt**2/4*self.EF.T(self.pro.space.C.T.dot(self.pro.space.M2.dot(bu)))
        out += self.dt*self.EF.T(self.pro.space.C.T.dot(self.pro.space.M2.dot(b)))
        
        # with additional J_eq x B
        if self.loc_jeq == 'step_2':

            out -= self.dt**2/4*self.JB(bu) 
            out += self.dt*self.JB(b)
        
        return out
    
    # ======================================
    def RHS6(self, u, p, b):
        
        out = self.A(u)
        
        # MHD bulk velocity is a 2-form
        if   self.basis_u == 2:
            
            out += self.dt**2/4*self.pro.space.D.T.dot(self.pro.space.M3.dot(self.L(u))) 
            out += self.dt*self.pro.space.D.T.dot(self.pro.space.M3.dot(p))

        # MHD bulk velocity is a 0-form
        elif self.basis_u == 0:
            
            out += self.dt**2/4*self.JF.T(self.pro.space.D.T.dot(self.pro.space.M3.dot(self.L(u))))
            out += self.dt*self.JF.T(self.pro.space.D.T.dot(self.pro.space.M3.dot(p)))
            
        # with additional J_eq x B
        if self.loc_jeq == 'step_6':
            
            out += self.dt*self.JB(b)
        
        return out
    
    # ======================================
    def setPreconditionerA(self, domain, which, drop_tol=1e-4, fill_fac=10.):
        
        # ILU preconditioner
        if which == 'ILU':
        
            ## MHD bulk velocity is a 2-form
            #if self.basis_u == 2:
            #    MF_local = self.pro.I2_inv_approx.dot(self.rhs_MF)
            #    A_local  = 1/2*(MF_local.T.dot(self.pro.space.M2) + self.pro.space.M2.dot(MF_local)).tolil()
            #
            ## MHD bulk velocity is a 0-form
            #elif self.basis_u == 0:
            #    MF_local_1 = self.pro.I0_inv_approx.dot(self.rhs_W11)
            #    MF_local_2 = self.pro.I0_all_inv_approx.dot(self.rhs_W22)
            #    MF_local_3 = self.pro.I0_all_inv_approx.dot(self.rhs_W33)
            #    MF_local = spa.bmat([[MF_local_1, None, None], [None, MF_local_2, None], [None, None, MF_local_3]], format='csr')
            #    A_local  = 1/2*(MF_local.T.dot(self.pro.space.Mv) + self.pro.space.Mv.dot(MF_local)).tolil()

            #A_ILU = spa.linalg.spilu(A_local.tocsc(), drop_tol=drop_tol, fill_factor=fill_fac)

            A_ILU = spa.linalg.spilu(self.MR.tocsc(), drop_tol=drop_tol, fill_factor=fill_fac)

            self.A_PRE = spa.linalg.LinearOperator(self.MR.shape, lambda x : A_ILU.solve(x))
            
        # FFT preconditioner
        elif which == 'FFT':
            
            if self.basis_u == 2:
                self.A_PRE = mass_pre.get_M2_PRE_3(self.pro.space, domain)
            elif self.basis_u == 0:
                self.A_PRE = mass_pre.get_Mv_PRE_3(self.pro.space, domain)
         
    
    # ======================================
    def setPreconditionerS2(self, domain, which, drop_tol=1e-4, fill_fac=10.):
        
        # ILU preconditioner
        if which == 'ILU':
        
            ## MHD bulk velocity is a 2-form
            #if self.basis_u == 2:
            #    MF_local = self.pro.I2_inv_approx.dot(self.rhs_MF)
            #    A_local  = 1/2*(MF_local.T.dot(self.pro.space.M2) + self.pro.space.M2.dot(MF_local)).tolil()
            #
            ## MHD bulk velocity is a 0-form
            #elif self.basis_u == 0:
            #    MF_local_1 = self.pro.I0_inv_approx.dot(self.rhs_W11)
            #    MF_local_2 = self.pro.I0_all_inv_approx.dot(self.rhs_W22)
            #    MF_local_3 = self.pro.I0_all_inv_approx.dot(self.rhs_W33)
            #    MF_local = spa.bmat([[MF_local_1, None, None], [None, MF_local_2, None], [None, None, MF_local_3]], format='csr')
            #    A_local  = 1/2*(MF_local.T.dot(self.pro.space.Mv) + self.pro.space.Mv.dot(MF_local)).tolil()

            EF_local = self.pro.I1_inv_approx.dot(self.rhs_EF)
            
            S2_local = self.MR + self.dt**2/4*EF_local.T.dot(self.pro.space.C.T.dot(self.pro.space.M2.dot(self.pro.space.C.dot(EF_local))))

            # with additional J_eq x B
            if self.loc_jeq == 'step_2':
                S2_local += self.dt**2/4*self.mat_JB.dot(self.pro.space.C.dot(EF_local))

            del EF_local

            S2_ILU = spa.linalg.spilu(S2_local.tocsc(), drop_tol=drop_tol , fill_factor=fill_fac)
            self.S2_PRE = spa.linalg.LinearOperator(S2_local.shape, lambda x : S2_ILU.solve(x))
            
        # FFT preconditioner
        elif which == 'FFT':
            
            if self.basis_u == 2:
                self.S2_PRE = mass_pre.get_M2_PRE_3(self.pro.space, domain)
            elif self.basis_u == 0:
                self.S2_PRE = mass_pre.get_Mv_PRE_3(self.pro.space, domain)

 
    # ======================================
    def setPreconditionerS6(self, domain, which, drop_tol=1e-4, fill_fac=10.):
        
        # ILU preconditioner
        if which == 'ILU':
        
            ## MHD bulk velocity is a 2-form
            #if self.basis_u == 2:
            #    MF_local = self.pro.I2_inv_approx.dot(self.rhs_MF)
            #    A_local  = 1/2*(MF_local.T.dot(self.pro.space.M2) + self.pro.space.M2.dot(MF_local)).tolil()
            #
            ## MHD bulk velocity is a 0-form
            #elif self.basis_u == 0:
            #    MF_local_1 = self.pro.I0_inv_approx.dot(self.rhs_W11)
            #    MF_local_2 = self.pro.I0_all_inv_approx.dot(self.rhs_W22)
            #    MF_local_3 = self.pro.I0_all_inv_approx.dot(self.rhs_W33)
            #    MF_local = spa.bmat([[MF_local_1, None, None], [None, MF_local_2, None], [None, None, MF_local_3]], format='csr')
            #    A_local  = 1/2*(MF_local.T.dot(self.pro.space.Mv) + self.pro.space.Mv.dot(MF_local)).tolil()

            # MHD bulk velocity is a 2-form
            if self.basis_u == 2:

                PF_local = self.pro.I2_inv_approx.dot(self.rhs_PF)
                PR_local = self.pro.I3_inv_approx.dot(self.rhs_PR)

                L_local  = -self.pro.space.D.dot(PF_local) - (self.gamma - 1)*PR_local.dot(self.pro.space.D)

                del PF_local, PR_local

                S6_local = self.MR - self.dt**2/4*self.pro.space.D.T.dot(self.pro.space.M3.dot(L_local))

            # MHD bulk velocity is a 0-form
            elif self.basis_u == 0:

                PF_local = self.pro.I2_inv_approx.dot(self.rhs_PF)
                JF_local = self.pro.I2_inv_approx.dot(self.rhs_JF)
                PR_local = self.pro.I3_inv_approx.dot(self.rhs_PR)

                L_local  = -self.pro.space.D.dot(PF_local) - (self.gamma - 1)*PR_local.dot(self.pro.space.D.dot(JF_local))

                del PF_local, PR_local

                S6_local = self.MR - self.dt**2/4*JF_local.T.dot(self.pro.space.D.T.dot(self.pro.space.M3.dot(L_local)))

                del JF_local

            S6_ILU = spa.linalg.spilu(S6_local.tocsc(), drop_tol=drop_tol , fill_factor=fill_fac)
            self.S6_PRE = spa.linalg.LinearOperator(S6_local.shape, lambda x : S6_ILU.solve(x))
            
        # FFT preconditioner
        elif which == 'FFT':
            
            if self.basis_u == 2:
                self.S6_PRE = mass_pre.get_M2_PRE_3(self.pro.space, domain)
            elif self.basis_u == 0:
                self.S6_PRE = mass_pre.get_Mv_PRE_3(self.pro.space, domain)