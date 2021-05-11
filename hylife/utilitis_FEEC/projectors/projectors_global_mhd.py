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

import hylife.utilitis_FEEC.projectors.projectors_global as pro


class operators_mhd:
    
    def __init__(self, projectors_3d, basis_u, bc_u1, bc_b1, dt, gamma, add_jeq_step2):
        
        self.projectors_3d = projectors_3d
        
        self.tensor_space  = self.projectors_3d.tensor_space
        
        self.T    = self.tensor_space.T
        self.p    = self.tensor_space.p
        self.t    = self.tensor_space.t
        self.bc   = self.tensor_space.bc
        self.el_b = self.tensor_space.el_b
        
        self.NbaseN  = self.tensor_space.NbaseN
        self.NbaseD  = self.tensor_space.NbaseD
        
        kind_splines = [False, True]
        
        self.basis_u = basis_u
        self.bc_u1   = bc_u1
        self.bc_b1   = bc_b1
        self.dt      = dt
        self.gamma   = gamma
        self.add_jeq_step2 = add_jeq_step2
        
        # non-vanishing 1D indices of expressions pi0(N), pi0(D), pi1(N) and pi1(D)
        projectors_1d = [pro.projectors_global_1d(space, n_quad) for space, n_quad in zip(self.tensor_space.spaces, self.projectors_3d.n_quad)]
        
        self.pi0_x_N_i, self.pi0_x_D_i, self.pi1_x_N_i, self.pi1_x_D_i = projectors_1d[0].projection_matrices_1d_reduced()
        self.pi0_y_N_i, self.pi0_y_D_i, self.pi1_y_N_i, self.pi1_y_D_i = projectors_1d[1].projection_matrices_1d_reduced()
        self.pi0_z_N_i, self.pi0_z_D_i, self.pi1_z_N_i, self.pi1_z_D_i = projectors_1d[2].projection_matrices_1d_reduced()
        
        # non-vanishing 1D indices of expressions pi0(NN), pi0(DN), pi0(ND), pi0(DD), pi1(NN), pi1(DN), pi1(ND) and pi0(DD)
        self.pi0_x_NN_i, self.pi0_x_DN_i, self.pi0_x_ND_i, self.pi0_x_DD_i, self.pi1_x_NN_i, self.pi1_x_DN_i, self.pi1_x_ND_i, self.pi1_x_DD_i = projectors_1d[0].projection_matrices_1d()
        self.pi0_y_NN_i, self.pi0_y_DN_i, self.pi0_y_ND_i, self.pi0_y_DD_i, self.pi1_y_NN_i, self.pi1_y_DN_i, self.pi1_y_ND_i, self.pi1_y_DD_i = projectors_1d[1].projection_matrices_1d()
        self.pi0_z_NN_i, self.pi0_z_DN_i, self.pi0_z_ND_i, self.pi0_z_DD_i, self.pi1_z_NN_i, self.pi1_z_DN_i, self.pi1_z_ND_i, self.pi1_z_DD_i = projectors_1d[2].projection_matrices_1d()
        
        # 1D collocation matrices for interpolation in format (point, global basis function)
        self.x_int = np.copy(self.projectors_3d.x_int)
        
        if self.projectors_3d.polar == True:
            self.x_int[0][0] = 0.
        
        self.basis_int_N = [bsp.collocation_matrix(T, p    , x_int, bc, normalize=kind_splines[0]) for T, p, x_int, bc in zip(self.T, self.p, self.x_int, self.bc)]
        self.basis_int_D = [bsp.collocation_matrix(t, p - 1, x_int, bc, normalize=kind_splines[1]) for t, p, x_int, bc in zip(self.t, self.p, self.x_int, self.bc)]
        
        # 1D integration sub-intervals, quadrature points and weights
        self.subs  = [0, 0, 0]
        self.x_his = [0, 0, 0]
        self.pts   = [0, 0, 0]
        self.wts   = [0, 0, 0]

        for dim in range(3):
            
            # compute quadrature grid for clamped splines
            if self.bc[dim] == False:
                self.x_his[dim] = np.union1d(self.x_int[dim], self.el_b[dim])
            
            # compute quadrature grid for periodic splines
            else:
                
                # even spline degree
                if self.p[dim]%2 == 0:
                    self.x_his[dim] = np.union1d(self.x_int[dim], self.el_b[dim][1:])
                    self.x_his[dim] = np.append(self.x_his[dim], self.el_b[dim][-1] + self.x_his[dim][0])
                
                # odd spline degree
                else:
                    self.x_his[dim] = np.append(self.x_int[dim], self.el_b[dim][-1])

            self.pts[dim], self.wts[dim] = bsp.quadrature_grid(self.x_his[dim], self.projectors_3d.pts_loc[dim], self.projectors_3d.wts_loc[dim])
            self.pts[dim]                = self.pts[dim]%1.

            # compute number of sub-intervals for integrations (even degree)
            if self.p[dim]%2 == 0:
                self.subs[dim] = 2*np.ones(projectors_3d.pts[dim].shape[0], dtype=int)

                if self.bc[dim] == False:
                    self.subs[dim][:self.p[dim]//2 ] = 1
                    self.subs[dim][-self.p[dim]//2:] = 1

            # compute number of sub-intervals for integrations (odd degree)
            else:
                self.subs[dim] = np.ones(projectors_3d.pts[dim].shape[0], dtype=int)
                
        # evaluate basis functions on quadrature points in format (interval, local quad. point, global basis function)
        self.basis_his_N = [bsp.collocation_matrix(T, p    , pts.flatten(), bc, normalize=kind_splines[0]).reshape(pts.shape[0], pts.shape[1], NbaseN) for T, p, pts, bc, NbaseN in zip(self.T, self.p, self.pts, self.bc, self.NbaseN)]
        self.basis_his_D = [bsp.collocation_matrix(t, p - 1, pts.flatten(), bc, normalize=kind_splines[1]).reshape(pts.shape[0], pts.shape[1], NbaseD) for t, p, pts, bc, NbaseD in zip(self.t, self.p, self.pts, self.bc, self.NbaseD)]
        
        if self.projectors_3d.polar == True:
            self.x_int[0][0] += 0.00001
    
    # =================================================================
    def assemble_rhs_EF(self, domain, b2_eq):
        
        b2_1_eq, b2_2_eq, b2_3_eq = self.tensor_space.unravel_2form(self.tensor_space.E2.T.dot(b2_eq))
        
        if self.basis_u == 2:

            # ====================== 12 - block ([his, int, int] of DND) ===========================

            # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
            B3_eq   = self.tensor_space.evaluate_DDN(self.pts[0].flatten(), self.x_int[1], self.x_int[2], b2_3_eq).reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.x_int[2].size)

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF  = abs(domain.evaluate(self.pts[0].flatten(), self.x_int[1], self.x_int[2], 'det_df'))
            det_dF  = det_dF.reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_D_i[0].size, dtype=int)

            ker.rhs11(self.pi1_x_D_i[0], self.pi0_y_N_i[0], self.pi0_z_D_i[0], self.pi1_x_D_i[1], self.pi0_y_N_i[1], self.pi0_z_D_i[1], self.subs[0], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), self.wts[0], self.basis_his_D[0], self.basis_int_N[1], self.basis_int_D[2], self.NbaseN, self.NbaseD, -B3_eq/det_dF, values, row_all, col_all)

            EF_12 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[0], self.tensor_space.Ntot_2form[1]))

            # ====================== 13 - block ([his, int, int] of DDN) ===========================

            # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
            B2_eq   = self.tensor_space.evaluate_DND(self.pts[0].flatten(), self.x_int[1], self.x_int[2], b2_2_eq).reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.x_int[2].size)

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF  = abs(domain.evaluate(self.pts[0].flatten(), self.x_int[1], self.x_int[2], 'det_df'))
            det_dF  = det_dF.reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs11(self.pi1_x_D_i[0], self.pi0_y_D_i[0], self.pi0_z_N_i[0], self.pi1_x_D_i[1], self.pi0_y_D_i[1], self.pi0_z_N_i[1], self.subs[0], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), self.wts[0], self.basis_his_D[0], self.basis_int_D[1], self.basis_int_N[2], self.NbaseN, self.NbaseD,  B2_eq/det_dF, values, row_all, col_all)

            EF_13 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[0], self.tensor_space.Ntot_2form[2]))


            # ====================== 21 - block ([int, his, int] of NDD) ===========================

            # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
            B3_eq   = self.tensor_space.evaluate_DDN(self.x_int[0], self.pts[1].flatten(), self.x_int[2], b2_3_eq).reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF  = abs(domain.evaluate(self.x_int[0], self.pts[1].flatten(), self.x_int[2], 'det_df'))
            det_dF  = det_dF.reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_D_i[0].size, dtype=int)

            ker.rhs12(self.pi0_x_N_i[0], self.pi1_y_D_i[0], self.pi0_z_D_i[0], self.pi0_x_N_i[1], self.pi1_y_D_i[1], self.pi0_z_D_i[1], self.subs[1], np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), self.wts[1], self.basis_int_N[0], self.basis_his_D[1], self.basis_int_D[2], self.NbaseN, self.NbaseD,  B3_eq/det_dF, values, row_all, col_all)

            EF_21 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[1], self.tensor_space.Ntot_2form[0]))


            # ====================== 23 - block ([int, his, int] of DDN) ===========================

            # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
            B1_eq   = self.tensor_space.evaluate_NDD(self.x_int[0], self.pts[1].flatten(), self.x_int[2], b2_1_eq).reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF  = abs(domain.evaluate(self.x_int[0], self.pts[1].flatten(), self.x_int[2], 'det_df'))
            det_dF  = det_dF.reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs12(self.pi0_x_D_i[0], self.pi1_y_D_i[0], self.pi0_z_N_i[0], self.pi0_x_D_i[1], self.pi1_y_D_i[1], self.pi0_z_N_i[1], self.subs[1], np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), self.wts[1], self.basis_int_D[0], self.basis_his_D[1], self.basis_int_N[2], self.NbaseN, self.NbaseD,  -B1_eq/det_dF, values, row_all, col_all)

            EF_23 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[1], self.tensor_space.Ntot_2form[2]))


            # ====================== 31 - block ([int, int, his] of NDD) ===========================

            # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
            B2_eq   = self.tensor_space.evaluate_DND(self.x_int[0], self.x_int[1], self.pts[2].flatten(), b2_2_eq).reshape(self.x_int[0].size, self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF  = abs(domain.evaluate(self.x_int[0], self.x_int[1], self.pts[2].flatten(), 'det_df'))
            det_dF  = det_dF.reshape(self.x_int[0].size, self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)

            ker.rhs13(self.pi0_x_N_i[0], self.pi0_y_D_i[0], self.pi1_z_D_i[0], self.pi0_x_N_i[1], self.pi0_y_D_i[1], self.pi1_z_D_i[1], self.subs[2], np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[2], self.basis_int_N[0], self.basis_int_D[1], self.basis_his_D[2], self.NbaseN, self.NbaseD,  -B2_eq/det_dF, values, row_all, col_all)

            EF_31 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[2], self.tensor_space.Ntot_2form[0]))

            # ====================== 32 - block ([int, int, his] of DND) ===========================

            # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
            B1_eq   = self.tensor_space.evaluate_NDD(self.x_int[0], self.x_int[1], self.pts[2].flatten(), b2_1_eq).reshape(self.x_int[0].size, self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF  = abs(domain.evaluate(self.x_int[0], self.x_int[1], self.pts[2].flatten(), 'det_df'))
            det_dF  = det_dF.reshape(self.x_int[0].size, self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=int)

            ker.rhs13(self.pi0_x_D_i[0], self.pi0_y_N_i[0], self.pi1_z_D_i[0], self.pi0_x_D_i[1], self.pi0_y_N_i[1], self.pi1_z_D_i[1], self.subs[2], np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[2], self.basis_int_D[0], self.basis_int_N[1], self.basis_his_D[2], self.NbaseN, self.NbaseD,  B1_eq/det_dF, values, row_all, col_all)

            EF_32 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[2], self.tensor_space.Ntot_2form[1]))

            self.rhs_EF = self.projectors_3d.P1.dot(spa.bmat([[None, EF_12, EF_13], [EF_21, None, EF_23], [EF_31, EF_32, None]], format='csr').dot(self.tensor_space.E2.T))
            self.rhs_EF.eliminate_zeros()
            
        elif self.basis_u == 0:
            
            # ====================== 12 - block ([his, int, int] of NNN) ===========================

            # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
            B3_eq   = self.tensor_space.evaluate_DDN(self.pts[0].flatten(), self.x_int[1], self.x_int[2], b2_3_eq).reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs11(self.pi1_x_N_i[0], self.pi0_y_N_i[0], self.pi0_z_N_i[0], self.pi1_x_N_i[1], self.pi0_y_N_i[1], self.pi0_z_N_i[1], self.subs[0], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), self.wts[0], self.basis_his_N[0], self.basis_int_N[1], self.basis_int_N[2], self.NbaseN, self.NbaseD, -B3_eq, values, row_all, col_all)

            EF_12 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[0], self.tensor_space.Ntot_0form))

            # ====================== 13 - block ([his, int, int] of NNN) ===========================

            # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
            B2_eq   = self.tensor_space.evaluate_DND(self.pts[0].flatten(), self.x_int[1], self.x_int[2], b2_2_eq).reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs11(self.pi1_x_N_i[0], self.pi0_y_N_i[0], self.pi0_z_N_i[0], self.pi1_x_N_i[1], self.pi0_y_N_i[1], self.pi0_z_N_i[1], self.subs[0], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), self.wts[0], self.basis_his_N[0], self.basis_int_N[1], self.basis_int_N[2], self.NbaseN, self.NbaseD,  B2_eq, values, row_all, col_all)

            EF_13 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[0], self.tensor_space.Ntot_0form))


            # ====================== 21 - block ([int, his, int] of NNN) ===========================

            # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
            B3_eq   = self.tensor_space.evaluate_DDN(self.x_int[0], self.pts[1].flatten(), self.x_int[2], b2_3_eq).reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs12(self.pi0_x_N_i[0], self.pi1_y_N_i[0], self.pi0_z_N_i[0], self.pi0_x_N_i[1], self.pi1_y_N_i[1], self.pi0_z_N_i[1], self.subs[1], np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), self.wts[1], self.basis_int_N[0], self.basis_his_N[1], self.basis_int_N[2], self.NbaseN, self.NbaseD,  B3_eq, values, row_all, col_all)

            EF_21 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[1], self.tensor_space.Ntot_0form))


            # ====================== 23 - block ([int, his, int] of NNN) ===========================

            # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
            B1_eq   = self.tensor_space.evaluate_NDD(self.x_int[0], self.pts[1].flatten(), self.x_int[2], b2_1_eq).reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs12(self.pi0_x_N_i[0], self.pi1_y_N_i[0], self.pi0_z_N_i[0], self.pi0_x_N_i[1], self.pi1_y_N_i[1], self.pi0_z_N_i[1], self.subs[1], np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), self.wts[1], self.basis_int_N[0], self.basis_his_N[1], self.basis_int_N[2], self.NbaseN, self.NbaseD,  -B1_eq, values, row_all, col_all)

            EF_23 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[1], self.tensor_space.Ntot_0form))


            # ====================== 31 - block ([int, int, his] of NNN) ===========================

            # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
            B2_eq   = self.tensor_space.evaluate_DND(self.x_int[0], self.x_int[1], self.pts[2].flatten(), b2_2_eq).reshape(self.x_int[0].size, self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)

            ker.rhs13(self.pi0_x_N_i[0], self.pi0_y_N_i[0], self.pi1_z_N_i[0], self.pi0_x_N_i[1], self.pi0_y_N_i[1], self.pi1_z_N_i[1], self.subs[2], np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[2], self.basis_int_N[0], self.basis_int_N[1], self.basis_his_N[2], self.NbaseN, self.NbaseD,  -B2_eq, values, row_all, col_all)

            EF_31 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[2], self.tensor_space.Ntot_0form))

            # ====================== 32 - block ([int, int, his] of NNN) ===========================

            # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
            B1_eq   = self.tensor_space.evaluate_NDD(self.x_int[0], self.x_int[1], self.pts[2].flatten(), b2_1_eq).reshape(self.x_int[0].size, self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)

            ker.rhs13(self.pi0_x_N_i[0], self.pi0_y_N_i[0], self.pi1_z_N_i[0], self.pi0_x_N_i[1], self.pi0_y_N_i[1], self.pi1_z_N_i[1], self.subs[2], np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[2], self.basis_int_N[0], self.basis_int_N[1], self.basis_his_N[2], self.NbaseN, self.NbaseD,  B1_eq, values, row_all, col_all)

            EF_32 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[2], self.tensor_space.Ntot_0form))

            self.rhs_EF = spa.bmat([[None, EF_12, EF_13], [EF_21, None, EF_23], [EF_31, EF_32, None]], format='csr')
            self.rhs_EF.eliminate_zeros()
            
            
    
    
    # =================================================================
    def assemble_rhs_F(self, domain, which, coeff3_eq=None):
        
        """
        which = mass and basis_u = 2 --> EQ = rho3/|det_dF|
        which = pressure and basis_u = 2 --> EQ = p3/|det_dF|
        
        which = mass and basis_u = 0 --> EQ = rho3
        which = pressure and basis_u = 0 --> EQ = p3
        which = jacobian and basis_u = 0 --> EQ = |det_dF|
        """
        
        if self.basis_u == 2:
            
            coeff3_eq = self.tensor_space.E3.T.dot(coeff3_eq).reshape(self.tensor_space.Nbase_3form)

            # ====================== 11 - block ([int, his, his] of NDD) ===========================

            # evaluate equilibrium density/pressure at interpolation and quadrature points
            EQ = self.tensor_space.evaluate_DDD(self.x_int[0], self.pts[1].flatten(), self.pts[2].flatten(), coeff3_eq).reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.pts[2].shape[0], self.pts[2].shape[1])

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF  = abs(domain.evaluate(self.x_int[0], self.pts[1].flatten(), self.pts[2].flatten(), 'det_df'))
            det_dF  = det_dF.reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.pts[2].shape[0], self.pts[2].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)

            ker.rhs21(self.pi0_x_N_i[0], self.pi1_y_D_i[0], self.pi1_z_D_i[0], self.pi0_x_N_i[1], self.pi1_y_D_i[1], self.pi1_z_D_i[1], self.subs[1], self.subs[2], np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[1], self.wts[2], self.basis_int_N[0], self.basis_his_D[1], self.basis_his_D[2], self.NbaseN, self.NbaseD, EQ/det_dF, values, row_all, col_all)

            F_11 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[0], self.tensor_space.Ntot_2form[0]))


            # ====================== 22 - block ([his, int, his] of DND) ===========================

            # evaluate equilibrium density/pressure at interpolation and quadrature points
            EQ = self.tensor_space.evaluate_DDD(self.pts[0].flatten(), self.x_int[1], self.pts[2].flatten(), coeff3_eq).reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF  = abs(domain.evaluate(self.pts[0].flatten(), self.x_int[1], self.pts[2].flatten(), 'det_df'))
            det_dF  = det_dF.reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=int)

            ker.rhs22(self.pi1_x_D_i[0], self.pi0_y_N_i[0], self.pi1_z_D_i[0], self.pi1_x_D_i[1], self.pi0_y_N_i[1], self.pi1_z_D_i[1], self.subs[0], self.subs[2], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[0], self.wts[2], self.basis_his_D[0], self.basis_int_N[1], self.basis_his_D[2], self.NbaseN, self.NbaseD, EQ/det_dF, values, row_all, col_all)

            F_22 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[1], self.tensor_space.Ntot_2form[1]))


            # ====================== 33 - block ([his, his, int] of DDN) ===========================

            # evaluate equilibrium density/pressure at interpolation and quadrature points
            EQ = self.tensor_space.evaluate_DDD(self.pts[0].flatten(), self.pts[1].flatten(), self.x_int[2], coeff3_eq).reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF  = abs(domain.evaluate(self.pts[0].flatten(), self.pts[1].flatten(), self.x_int[2], 'det_df'))
            det_dF  = det_dF.reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs23(self.pi1_x_D_i[0], self.pi1_y_D_i[0], self.pi0_z_N_i[0], self.pi1_x_D_i[1], self.pi1_y_D_i[1], self.pi0_z_N_i[1], self.subs[0], self.subs[1], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), self.wts[0], self.wts[1], self.basis_his_D[0], self.basis_his_D[1], self.basis_int_N[2], self.NbaseN, self.NbaseD, EQ/det_dF, values, row_all, col_all)

            F_33 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[2], self.tensor_space.Ntot_2form[2]))
        
            if   which == 'mass':
                self.rhs_FM = self.projectors_3d.P2.dot(spa.bmat([[F_11, None, None], [None, F_22, None], [None, None, F_33]], format='csr').dot(self.tensor_space.E2.T))
                self.rhs_FM.eliminate_zeros()
            elif which == 'pressure':
                self.rhs_FP = self.projectors_3d.P2.dot(spa.bmat([[F_11, None, None], [None, F_22, None], [None, None, F_33]], format='csr').dot(self.tensor_space.E2.T))
                self.rhs_FP.eliminate_zeros()
                
        elif self.basis_u == 0:
            
            # ====================== 11 - block ([int, his, his] of NNN) ===========================
            if which == 'jacobian':
                # evaluate Jacobian determinant at at interpolation and quadrature points
                EQ = abs(domain.evaluate(self.x_int[0], self.pts[1].flatten(), self.pts[2].flatten(), 'det_df'))
                EQ = EQ.reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.pts[2].shape[0], self.pts[2].shape[1])
                
                print(which, EQ.min(), EQ.max())
            
            else:
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                coeff3_eq = self.tensor_space.E3.T.dot(coeff3_eq).reshape(self.tensor_space.Nbase_3form)
                EQ = self.tensor_space.evaluate_DDD(self.x_int[0], self.pts[1].flatten(), self.pts[2].flatten(), coeff3_eq).reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.pts[2].shape[0], self.pts[2].shape[1])
                
                print(which, EQ.min(), EQ.max())

            # assemble sparse matrix
            values  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)

            ker.rhs21(self.pi0_x_N_i[0], self.pi1_y_N_i[0], self.pi1_z_N_i[0], self.pi0_x_N_i[1], self.pi1_y_N_i[1], self.pi1_z_N_i[1], self.subs[1], self.subs[2], np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[1], self.wts[2], self.basis_int_N[0], self.basis_his_N[1], self.basis_his_N[2], self.NbaseN, self.NbaseD, EQ, values, row_all, col_all)

            F_11 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[0], self.tensor_space.Ntot_0form))


            # ====================== 22 - block ([his, int, his] of NNN) ===========================
            if which == 'jacobian':
                # evaluate Jacobian determinant at at interpolation and quadrature points
                EQ = abs(domain.evaluate(self.pts[0].flatten(), self.x_int[1], self.pts[2].flatten(), 'det_df'))
                EQ = EQ.reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])
                
                print(which, EQ.min(), EQ.max())
                
            else:
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                EQ = self.tensor_space.evaluate_DDD(self.pts[0].flatten(), self.x_int[1], self.pts[2].flatten(), coeff3_eq).reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])
                
                print(which, EQ.min(), EQ.max())

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_N_i[0].size, dtype=int)

            ker.rhs22(self.pi1_x_N_i[0], self.pi0_y_N_i[0], self.pi1_z_N_i[0], self.pi1_x_N_i[1], self.pi0_y_N_i[1], self.pi1_z_N_i[1], self.subs[0], self.subs[2], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[0], self.wts[2], self.basis_his_N[0], self.basis_int_N[1], self.basis_his_N[2], self.NbaseN, self.NbaseD, EQ, values, row_all, col_all)

            F_22 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[1], self.tensor_space.Ntot_0form))


            # ====================== 33 - block ([his, his, int] of NNN) ===========================
            if which == 'jacobian':
                # evaluate Jacobian determinant at at interpolation and quadrature points
                EQ = abs(domain.evaluate(self.pts[0].flatten(), self.pts[1].flatten(), self.x_int[2], 'det_df'))
                EQ = EQ.reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)
                
                print(which, EQ.min(), EQ.max())
                
            else:
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                EQ = self.tensor_space.evaluate_DDD(self.pts[0].flatten(), self.pts[1].flatten(), self.x_int[2], coeff3_eq).reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size) 
                
                print(which, EQ.min(), EQ.max())

            # assemble sparse matrix
            values  = np.empty(self.pi1_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
            row_all = np.empty(self.pi1_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
            col_all = np.empty(self.pi1_x_N_i[0].size*self.pi1_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)

            ker.rhs23(self.pi1_x_N_i[0], self.pi1_y_N_i[0], self.pi0_z_N_i[0], self.pi1_x_N_i[1], self.pi1_y_N_i[1], self.pi0_z_N_i[1], self.subs[0], self.subs[1], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), self.wts[0], self.wts[1], self.basis_his_N[0], self.basis_his_N[1], self.basis_int_N[2], self.NbaseN, self.NbaseD, EQ, values, row_all, col_all)

            F_33 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[2], self.tensor_space.Ntot_0form))
        
            if   which == 'mass':
                self.rhs_FM = self.projectors_3d.P2.dot(spa.bmat([[F_11, None, None], [None, F_22, None], [None, None, F_33]], format='csr'))
                self.rhs_FM.eliminate_zeros()
            elif which == 'pressure':
                self.rhs_FP = self.projectors_3d.P2.dot(spa.bmat([[F_11, None, None], [None, F_22, None], [None, None, F_33]], format='csr'))
                self.rhs_FP.eliminate_zeros()
            elif which == 'jacobian':
                self.rhs_FG = self.projectors_3d.P2.dot(spa.bmat([[F_11, None, None], [None, F_22, None], [None, None, F_33]], format='csr'))
                self.rhs_FG.eliminate_zeros()
            
    # =================================================================
    def assemble_rhs_W(self, domain, rho3_eq):
        
        rho3_eq = self.tensor_space.E3.T.dot(rho3_eq).reshape(self.tensor_space.Nbase_3form)
        
        # ====================== ([int, int, int] of NNN) ===========================
        # evaluate equilibrium density at quadrature points
        EQ = self.tensor_space.evaluate_DDD(self.x_int[0], self.x_int[1], self.x_int[2], rho3_eq)
        
        # evaluate Jacobian determinant at at interpolation points
        det_dF = abs(domain.evaluate(self.x_int[0], self.x_int[1], self.x_int[2], 'det_df'))
        
        # assemble sparse matrix
        values  = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
        row_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
        col_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
        
        ker.rhs0(self.pi0_x_N_i[0], self.pi0_y_N_i[0], self.pi0_z_N_i[0], self.pi0_x_N_i[1], self.pi0_y_N_i[1], self.pi0_z_N_i[1], self.basis_int_N[0], self.basis_int_N[1], self.basis_int_N[2], EQ/det_dF, values, row_all, col_all)
        
        self.rhs_W = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_0form, self.tensor_space.Ntot_0form))
        self.rhs_W.eliminate_zeros()
    
    
    # =================================================================
    def assemble_rhs_PR(self, domain, p3_eq):
        
        p3_eq  = self.tensor_space.E3.T.dot(p3_eq).reshape(self.tensor_space.Nbase_3form)
            
        # ====================== ([his, his, his] of DDD) ===========================
        # evaluate equilibrium pressure at quadrature points
        PR_eq  = self.tensor_space.evaluate_DDD(self.pts[0].flatten(), self.pts[1].flatten(), self.pts[2].flatten(), p3_eq).reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.pts[1].shape[0], self.pts[1].shape[1], self.pts[2].shape[0], self.pts[2].shape[1])
        
        # evaluate Jacobian determinant at at interpolation and quadrature points
        det_dF = abs(domain.evaluate(self.pts[0].flatten(), self.pts[1].flatten(), self.pts[2].flatten(), 'det_df'))
        det_dF = det_dF.reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.pts[1].shape[0], self.pts[1].shape[1], self.pts[2].shape[0], self.pts[2].shape[1])
        
        # assemble sparse matrix
        values  = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
        row_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
        col_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
        
        ker.rhs3(self.pi1_x_D_i[0], self.pi1_y_D_i[0], self.pi1_z_D_i[0], self.pi1_x_D_i[1], self.pi1_y_D_i[1], self.pi1_z_D_i[1], self.subs[0], self.subs[1], self.subs[2], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[0], self.wts[1], self.wts[2], self.basis_his_D[0], self.basis_his_D[1], self.basis_his_D[2], self.NbaseN, self.NbaseD, PR_eq/det_dF, values, row_all, col_all)
        
        self.rhs_PR = self.projectors_3d.P3.dot(spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_3form, self.tensor_space.Ntot_3form)).dot(self.tensor_space.E3.T))
        self.rhs_PR.eliminate_zeros()
        
    
    
    # =================================================================
    def assemble_TF(self, domain, b2_eq):
        
        # compute C.T(M2(b2_eq)) in weak Lorentz force term
        f1 = self.tensor_space.CURL.T.dot(self.tensor_space.M2.dot(b2_eq))
        
        # apply transposed projection extraction operator
        f1_1, f1_2, f1_3 = self.tensor_space.unravel_1form(self.projectors_3d.P1.T.dot(self.projectors_3d.apply_IinvT_V1(f1)))
        
        # evaluate Jacobian determinant at point sets
        det_DF_hii = domain.evaluate(self.pts[0].flatten(), self.x_int[1], self.x_int[2], 'det_df')
        det_DF_ihi = domain.evaluate(self.x_int[0], self.pts[1].flatten(), self.x_int[2], 'det_df')
        det_DF_iih = domain.evaluate(self.x_int[0], self.x_int[1], self.pts[2].flatten(), 'det_df')
        
        det_DF_hii = det_DF_hii.reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.x_int[2].size)
        det_DF_ihi = det_DF_ihi.reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)
        det_DF_iih = det_DF_iih.reshape(self.x_int[0].size, self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])
        
        
        # ====================== 12 - block ([int, int, his] of ND DN DD) ===========================
        
        # assemble sparse matrix
        values  = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1)*(self.pi1_z_DD_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1)*(self.pi1_z_DD_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1)*(self.pi1_z_DD_i[3].max() + 1), dtype=int)
        
        ker.rhs13_f(self.pi0_x_ND_i, self.pi0_y_DN_i, self.pi1_z_DD_i, self.subs[2], np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[2], self.basis_int_N[0], self.basis_int_D[0], self.basis_int_D[1], self.basis_int_N[1], self.basis_his_D[2], self.basis_his_D[2], 1/det_DF_iih, f1_3, values, row_all, col_all)
        
        A_12 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[0], self.tensor_space.Ntot_2form[1]))
        
        
        # ====================== 13 - block ([int, his, int] of ND DD DN) ===========================
        
        # assemble sparse matrix
        values  = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1)*(self.pi0_z_DN_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1)*(self.pi0_z_DN_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi0_x_ND_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1)*(self.pi0_z_DN_i[3].max() + 1), dtype=int)
        
        ker.rhs12_f(self.pi0_x_ND_i, self.pi1_y_DD_i, self.pi0_z_DN_i, self.subs[1], np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), self.wts[1], self.basis_int_N[0], self.basis_int_D[0], self.basis_his_D[1], self.basis_his_D[1], self.basis_int_D[2], self.basis_int_N[2], 1/det_DF_ihi, -f1_2, values, row_all, col_all)
        
        A_13 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[0], self.tensor_space.Ntot_2form[2]))
        
        # ====================== 21 - block ([int, int, his] of DN ND DD) ===========================
        
        # assemble sparse matrix
        values  = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1)*(self.pi1_z_DD_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1)*(self.pi1_z_DD_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1)*(self.pi1_z_DD_i[3].max() + 1), dtype=int)
        
        ker.rhs13_f(self.pi0_x_DN_i, self.pi0_y_ND_i, self.pi1_z_DD_i, self.subs[2], np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[2], self.basis_int_D[0], self.basis_int_N[0], self.basis_int_N[1], self.basis_int_D[1], self.basis_his_D[2], self.basis_his_D[2], 1/det_DF_iih, -f1_3, values, row_all, col_all)
        
        A_21 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[1], self.tensor_space.Ntot_2form[0]))
        
        
        # ====================== 23 - block ([his, int, int] of DD ND DN) ===========================
        
        # assemble sparse matrix
        values  = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1)*(self.pi0_z_DN_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1)*(self.pi0_z_DN_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_ND_i[3].max() + 1)*(self.pi0_z_DN_i[3].max() + 1), dtype=int)
        
        ker.rhs11_f(self.pi1_x_DD_i, self.pi0_y_ND_i, self.pi0_z_DN_i, self.subs[0], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), self.wts[0], self.basis_his_D[0], self.basis_his_D[0], self.basis_int_N[1], self.basis_int_D[1], self.basis_int_D[2], self.basis_int_N[2], 1/det_DF_hii, f1_1, values, row_all, col_all)
        
        A_23 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[1], self.tensor_space.Ntot_2form[2]))
        
        
        # ====================== 31 - block ([int, his, int] of DN DD ND) ===========================

        # assemble sparse matrix
        values  = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1)*(self.pi0_z_ND_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1)*(self.pi0_z_ND_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi0_x_DN_i[3].max() + 1)*(self.pi1_y_DD_i[3].max() + 1)*(self.pi0_z_ND_i[3].max() + 1), dtype=int)
        
        ker.rhs12_f(self.pi0_x_DN_i, self.pi1_y_DD_i, self.pi0_z_ND_i, self.subs[1], np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), self.wts[1], self.basis_int_D[0], self.basis_int_N[0], self.basis_his_D[1], self.basis_his_D[1], self.basis_int_N[2], self.basis_int_D[2], 1/det_DF_ihi, f1_2, values, row_all, col_all)
        
        A_31 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[2], self.tensor_space.Ntot_2form[0]))
        
        
        # ====================== 32 - block ([his, int, int] of DD DN ND) ===========================
        
        # assemble sparse matrix
        values  = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1)*(self.pi0_z_ND_i[3].max() + 1), dtype=float)
        row_all = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1)*(self.pi0_z_ND_i[3].max() + 1), dtype=int)
        col_all = np.empty((self.pi1_x_DD_i[3].max() + 1)*(self.pi0_y_DN_i[3].max() + 1)*(self.pi0_z_ND_i[3].max() + 1), dtype=int)
        
        ker.rhs11_f(self.pi1_x_DD_i, self.pi0_y_DN_i, self.pi0_z_ND_i, self.subs[0], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), self.wts[0], self.basis_his_D[0], self.basis_his_D[0], self.basis_int_D[1], self.basis_int_N[1], self.basis_int_N[2], self.basis_int_D[2], 1/det_DF_hii, -f1_1, values, row_all, col_all)
        
        A_32 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[2], self.tensor_space.Ntot_2form[1]))
        
        self.mat_TF = (self.tensor_space.E2.dot(spa.bmat([[None, A_12, A_13], [A_21, None, A_23], [A_31, A_32, None]], format='csr').dot(self.tensor_space.E2.T))).T
        self.mat_TF.eliminate_zeros()
    
    
    # ======================================
    def __EF(self, u):
        return self.projectors_3d.solve_V1(self.rhs_EF.dot(u))
    
    # ======================================
    def __EF_transposed(self, e):
        return self.rhs_EF.T.dot(self.projectors_3d.apply_IinvT_V1(e))
    
    # ======================================
    def __FM(self, u):
        return self.projectors_3d.solve_V2(self.rhs_FM.dot(u))
    
    # ======================================
    def __FM_transposed(self, u):
        return self.rhs_FM.T.dot(self.projectors_3d.apply_IinvT_V2(u))
    
    # ======================================
    def __FP(self, u):
        return self.projectors_3d.solve_V2(self.rhs_FP.dot(u))
    
    # ======================================
    def __FP_transposed(self, u):
        return self.rhs_FP.T.dot(self.projectors_3d.apply_IinvT_V2(u))
    
    # ======================================
    def __FG(self, u):
        return self.projectors_3d.solve_V2(self.rhs_FG.dot(u))
    
    # ======================================
    def __FG_transposed(self, u):
        return self.rhs_FG.T.dot(self.projectors_3d.apply_IinvT_V2(u))
    
    # ======================================
    def __PR(self, f3):
        return self.projectors_3d.solve_V3(self.rhs_PR.dot(f3))
    
    # ======================================
    def __PR_transposed(self, f3):
        return self.rhs_PR.T.dot(self.projectors_3d.apply_IinvT_V3(f3))
    
    # ======================================
    def __TF(self, b2):
        return self.mat_TF.dot(b2)
    
    # ======================================
    def __TF_transposed(self, u2):
        return self.mat_TF.T.dot(u2)
    
    # ======================================
    def __W(self, u):
        
        u1, u2, u3 = self.tensor_space.unravel_0form(u)
        
        a = self.projectors_3d.solve_V0(self.rhs_W.dot(u1.flatten()))
        b = self.projectors_3d.solve_V0(self.rhs_W.dot(u2.flatten()))
        c = self.projectors_3d.solve_V0(self.rhs_W.dot(u3.flatten()))
        
        return np.concatenate((a, b, c))
    
    # ======================================
    def __W_transposed(self, u):
        
        u1, u2, u3 = self.tensor_space.unravel_0form(u)
        
        a = self.rhs_W.T.dot(self.projectors_3d.apply_IinvT_V0(u1.flatten()))
        b = self.rhs_W.T.dot(self.projectors_3d.apply_IinvT_V0(u2.flatten()))
        c = self.rhs_W.T.dot(self.projectors_3d.apply_IinvT_V0(u3.flatten()))
        
        return np.concatenate((a, b, c))
    
    # ======================================
    def __A(self, u):
        if self.basis_u == 2:
            return 1/2*(self.__FM_transposed(self.tensor_space.M2.dot(u)) + self.tensor_space.M2.dot(self.__FM(u)))
        elif self.basis_u == 0:
            return 1/2*(self.__W_transposed(self.tensor_space.Mv.dot(u)) + self.tensor_space.Mv.dot(self.__W(u)))
    
    # ======================================
    def __A_transposed(self, u):
        if self.basis_u == 2:
            return 1/2*(self.__FM_transposed(self.tensor_space.M2.dot(u)) + self.tensor_space.M2.dot(self.__FM(u)))
        elif self.basis_u == 0:
            return 1/2*(self.__W_transposed(self.tensor_space.Mv.dot(u)) + self.tensor_space.Mv.dot(self.__W(u)))
    
    # ======================================
    def __L(self, u):
        if self.basis_u == 2:
            return -self.tensor_space.DIV.dot(self.__FP(u)) - (self.gamma - 1)*self.__PR(self.tensor_space.DIV.dot(u))
        elif self.basis_u == 0:
            return -self.tensor_space.DIV.dot(self.__FP(u)) - (self.gamma - 1)*self.__PR(self.tensor_space.DIV.dot(self.__FG(u)))
    
    
    # ======================================
    def __S2(self, u):
        
        # without J_eq x B
        if self.add_jeq_step2 == False:
            out = self.__A(u) + self.dt**2/4*self.__EF_transposed(self.tensor_space.CURL.T.dot(self.tensor_space.M2.dot(self.tensor_space.CURL.dot(self.__EF(u)))))
            
        # with J_eq x B
        else:
            temp = self.tensor_space.CURL.dot(self.__EF(u))
            
            out = self.__A(u) + self.dt**2/4*self.__EF_transposed(self.tensor_space.CURL.T.dot(self.tensor_space.M2.dot(temp)))
            
            + self.dt**2/4*self.__TF(temp)

        # apply boundary conditions
        if self.bc[0] == False:
            
            if self.tensor_space.polar == False:
                # eta1 = 0
                if self.bc_u1[0] == 'dirichlet':
                    out[:self.NbaseD[1]*self.NbaseD[2]] = u[:self.NbaseD[1]*self.NbaseD[2]]
                # eta1 = 1
                if self.bc_u1[1] == 'dirichlet':
                    out[(self.NbaseN[0] - 1)*self.NbaseD[1]*self.NbaseD[2]:self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2]] = u[(self.NbaseN[0] - 1)*self.NbaseD[1]*self.NbaseD[2]:self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2]]

            # for polar splines only b.c. at eta1 = 1 possible
            else:
                if self.bc_u1[1] == 'dirichlet':
                    out[2*self.NbaseD[2] + (self.NbaseN[0] - 3)*self.NbaseD[1]*self.NbaseD[2]:2*self.NbaseD[2] + (self.NbaseN[0] - 2)*self.NbaseD[1]*self.NbaseD[2]] = u[2*self.NbaseD[2] + (self.NbaseN[0] - 3)*self.NbaseD[1]*self.NbaseD[2]:2*self.NbaseD[2] + (self.NbaseN[0] - 2)*self.NbaseD[1]*self.NbaseD[2]]

        return out
    
    # ======================================
    def __S6(self, u):
        
        if self.basis_u == 2:
            out = self.__A(u) - self.dt**2/4*self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(self.__L(u)))
        elif self.basis_u == 0:
            out = self.__A(u) - self.dt**2/4*self.__FG_transposed(self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(self.__L(u))))

        # apply boundary conditions
        if self.bc[0] == False:
            
            # without polar splines b.c. at eta1 = 0 and eta1 = 1 possible
            if self.tensor_space.polar == False:
                # eta1 = 0
                if self.bc_u1[0] == 'dirichlet':
                    out[:self.NbaseD[1]*self.NbaseD[2]] = u[:self.NbaseD[1]*self.NbaseD[2]]
                # eta1 = 1
                if self.bc_u1[1] == 'dirichlet':
                    out[(self.NbaseN[0] - 1)*self.NbaseD[1]*self.NbaseD[2]:self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2]] = u[(self.NbaseN[0] - 1)*self.NbaseD[1]*self.NbaseD[2]:self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2]]

            # for polar splines only b.c. at eta1 = 1 possible
            else:
                if self.bc_u1[1] == 'dirichlet':
                    out[2*self.NbaseD[2] + (self.NbaseN[0] - 3)*self.NbaseD[1]*self.NbaseD[2]:2*self.NbaseD[2] + (self.NbaseN[0] - 2)*self.NbaseD[1]*self.NbaseD[2]] = u[2*self.NbaseD[2] + (self.NbaseN[0] - 3)*self.NbaseD[1]*self.NbaseD[2]:2*self.NbaseD[2] + (self.NbaseN[0] - 2)*self.NbaseD[1]*self.NbaseD[2]]

        return out
    
    # ======================================
    def setOperators(self):
        
        self.FM = spa.linalg.LinearOperator((self.tensor_space.E2.shape[0], self.tensor_space.E2.shape[0]), matvec=self.__FM, rmatvec=self.__FM_transposed)
        
        self.FP = spa.linalg.LinearOperator((self.tensor_space.E2.shape[0], self.tensor_space.E2.shape[0]), matvec=self.__FP, rmatvec=self.__FP_transposed)
        
        if self.basis_u == 0:
            self.FG = spa.linalg.LinearOperator((self.tensor_space.E2.shape[0], 3*self.tensor_space.E0.shape[0]), matvec=self.__FG, rmatvec=self.__FG_transposed)
        
        self.EF = spa.linalg.LinearOperator((self.tensor_space.E1.shape[0], self.tensor_space.E2.shape[0]), matvec=self.__EF, rmatvec=self.__EF_transposed)
        
        self.PR = spa.linalg.LinearOperator((self.tensor_space.E3.shape[0], self.tensor_space.E3.shape[0]), matvec=self.__PR, rmatvec=self.__PR_transposed)
        
        self.TF = spa.linalg.LinearOperator((self.tensor_space.E2.shape[0], self.tensor_space.E2.shape[0]), matvec=self.__TF, rmatvec=self.__TF_transposed)
        
        self.A  = spa.linalg.LinearOperator((self.tensor_space.E2.shape[0], self.tensor_space.E2.shape[0]), matvec=self.__A, rmatvec=self.__A_transposed)
        
        self.L  = spa.linalg.LinearOperator((self.tensor_space.E3.shape[0], self.tensor_space.E2.shape[0]), matvec=self.__L)
        
        self.S2 = spa.linalg.LinearOperator((self.tensor_space.E2.shape[0], self.tensor_space.E2.shape[0]), matvec=self.__S2)
        
        self.S6 = spa.linalg.LinearOperator((self.tensor_space.E2.shape[0], self.tensor_space.E2.shape[0]), matvec=self.__S6)
        
    # ======================================
    def RHS2(self, u, b):
        
        # without J_eq x B
        if self.add_jeq_step2 == False:
            out = self.A(u) - self.dt**2/4*self.EF.T(self.tensor_space.CURL.T.dot(self.tensor_space.M2.dot(self.tensor_space.CURL.dot(self.EF(u))))) + self.dt*self.EF.T(self.tensor_space.CURL.T.dot(self.tensor_space.M2.dot(b)))
        
        # with J_eq x B
        else:
            
            temp = self.tensor_space.CURL.dot(self.EF(u))

            out = self.A(u) - self.dt**2/4*self.EF.T(self.tensor_space.CURL.T.dot(self.tensor_space.M2.dot(temp))) - self.dt**2/4*self.TF(temp) + self.dt*self.EF.T(self.tensor_space.CURL.T.dot(self.tensor_space.M2.dot(b))) + self.dt*self.TF(b)
        
        # apply boundary conditions
        self.tensor_space.apply_bc_2form(out, self.bc_u1)
        
        return out
    
    # ======================================
    def RHS6(self, u, p, b):
        
        # with J_eq x B
        if self.add_jeq_step2 == False: 
        
            # MHD bulk velocity is a 2-form
            if   self.basis_u == 2:
                out = self.A(u) + self.dt**2/4*self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(self.L(u))) + self.dt*self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(p)) + self.dt*self.TF(b)
            # MHD bulk velocity is a 0-form
            elif self.basis_u == 0:
                out = self.A(u) + self.dt**2/4*self.FG.T(self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(self.L(u)))) + self.dt*self.FG.T(self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(p)))
                
        # without J_eq x B
        else:
            
            # MHD bulk velocity is a 2-form
            if   self.basis_u == 2:
                out = self.A(u) + self.dt**2/4*self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(self.L(u))) + self.dt*self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(p))
            # MHD bulk velocity is a 0-form
            elif self.basis_u == 0:
                out = self.A(u) + self.dt**2/4*self.FG.T(self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(self.L(u)))) + self.dt*self.FG.T(self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(p)))
            
        self.tensor_space.apply_bc_2form(out, self.bc_u1)
        
        return out
    
    # ======================================
    def setPreconditionerA(self, drop_tol, fill_fac):
        
        if self.basis_u == 2:
            FM_local = self.projectors_3d.I2_inv_approx.dot(self.rhs_FM)
            A_local  = 1/2*(FM_local.T.dot(self.tensor_space.M2) + self.tensor_space.M2.dot(FM_local)).tolil()
        elif self.basis_u == 0:
            FM_local = self.projectors_3d.I0_inv_approx.dot(self.rhs_W)
            FM_local = spa.bmat([[FM_local, None, None], [None, FM_local, None], [None, None, FM_local]], format='csr')
            A_local  = 1/2*(FM_local.T.dot(self.tensor_space.Mv) + self.tensor_space.Mv.dot(FM_local)).tolil()

        del FM_local

        # apply boundary conditions to A_local
        if self.bc[0] == False:
            
            # without polar splines b.c. at eta1 = 0 and eta1 = 1 possible
            if self.tensor_space.polar == False:

                # eta1 = 0
                if self.bc_u1[0] == 'dirichlet':
                    lower = 0
                    upper = self.NbaseD[1]*self.NbaseD[2]

                    A_local[lower:upper,      :     ] = 0.
                    A_local[     :     , lower:upper] = 0.
                    A_local[lower:upper, lower:upper] = np.identity(self.NbaseD[1]*self.NbaseD[2])
                    
                # eta1 = 1
                if self.bc_u1[1] == 'dirichlet':
                    lower = (self.NbaseN[0] - 1)*self.NbaseD[1]*self.NbaseD[2]
                    upper =  self.NbaseN[0]     *self.NbaseD[1]*self.NbaseD[2]
                    
                    A_local[lower:upper,      :     ] = 0.
                    A_local[     :     , lower:upper] = 0.
                    A_local[lower:upper, lower:upper] = np.identity(self.NbaseD[1]*self.NbaseD[2])

            # for polar splines only b.c. at eta1 = 1 possible
            else:
                if self.bc_u1[1] == 'dirichlet':
                    lower = 2*self.NbaseD[2] + (self.NbaseN[0] - 3)*self.NbaseD[1]*self.NbaseD[2]
                    upper = 2*self.NbaseD[2] + (self.NbaseN[0] - 2)*self.NbaseD[1]*self.NbaseD[2]

                    A_local[lower:upper,      :     ] = 0.
                    A_local[     :     , lower:upper] = 0.
                    A_local[lower:upper, lower:upper] = np.identity(self.NbaseD[1]*self.NbaseD[2])


        A_ILU = spa.linalg.spilu(A_local.tocsc(), drop_tol=drop_tol , fill_factor=fill_fac)
        self.A_PRE = spa.linalg.LinearOperator(A_local.shape, lambda x : A_ILU.solve(x))
        
        
    # ======================================
    def setPreconditionerS2(self, drop_tol, fill_fac):
        
        if self.basis_u == 2:
            FM_local = self.projectors_3d.I2_inv_approx.dot(self.rhs_FM)
            A_local  = 1/2*(FM_local.T.dot(self.tensor_space.M2) + self.tensor_space.M2.dot(FM_local)).tolil()
        elif self.basis_u == 0:
            FM_local = self.projectors_3d.I0_inv_approx.dot(self.rhs_W)
            FM_local = spa.bmat([[FM_local, None, None], [None, FM_local, None], [None, None, FM_local]], format='csr')
            A_local  = 1/2*(FM_local.T.dot(self.tensor_space.Mv) + self.tensor_space.Mv.dot(FM_local)).tolil()

        del FM_local
        
        EF_local = self.projectors_3d.I1_inv_approx.dot(self.rhs_EF)
        
        # without J_eq x B
        if self.add_jeq_step2 == False:
            S2_local = (A_local + self.dt**2/4*EF_local.T.dot(self.tensor_space.CURL.T.dot(self.tensor_space.M2.dot(self.tensor_space.CURL.dot(EF_local))))).tolil()
            
        # with J_eq x B
        else:
            S2_local = (A_local + self.dt**2/4*EF_local.T.dot(self.tensor_space.CURL.T.dot(self.tensor_space.M2.dot(self.tensor_space.CURL.dot(EF_local)))) + self.dt**2/4*self.mat_TF.dot(self.tensor_space.CURL.dot(EF_local))).tolil()
            
        
        del A_local, EF_local

        # apply boundary conditions to S2_local
        if self.bc[0] == False:
            
            # without polar splines b.c. at eta1 = 0 and eta1 = 1 possible
            if self.tensor_space.polar == False:

                # eta1 = 0
                if self.bc_u1[0] == 'dirichlet':
                    lower = 0
                    upper = self.NbaseD[1]*self.NbaseD[2]

                    S2_local[lower:upper,      :     ] = 0.
                    S2_local[     :     , lower:upper] = 0.
                    S2_local[lower:upper, lower:upper] = np.identity(self.NbaseD[1]*self.NbaseD[2])
                    
                # eta1 = 1
                if self.bc_u1[1] == 'dirichlet':
                    lower = (self.NbaseN[0] - 1)*self.NbaseD[1]*self.NbaseD[2]
                    upper =  self.NbaseN[0]     *self.NbaseD[1]*self.NbaseD[2]
                    
                    S2_local[lower:upper,      :     ] = 0.
                    S2_local[     :     , lower:upper] = 0.
                    S2_local[lower:upper, lower:upper] = np.identity(self.NbaseD[1]*self.NbaseD[2])

            # for polar splines only b.c. at eta1 = 1 possible
            else:
                if self.bc_u1[1] == 'dirichlet':
                    lower = 2*self.NbaseD[2] + (self.NbaseN[0] - 3)*self.NbaseD[1]*self.NbaseD[2]
                    upper = 2*self.NbaseD[2] + (self.NbaseN[0] - 2)*self.NbaseD[1]*self.NbaseD[2]

                    S2_local[lower:upper,      :     ] = 0.
                    S2_local[     :     , lower:upper] = 0.
                    S2_local[lower:upper, lower:upper] = np.identity(self.NbaseD[1]*self.NbaseD[2])
                    

        S2_ILU = spa.linalg.spilu(S2_local.tocsc(), drop_tol=drop_tol , fill_factor=fill_fac)
        self.S2_PRE = spa.linalg.LinearOperator(S2_local.shape, lambda x : S2_ILU.solve(x))
        
        
    # ======================================
    def setPreconditionerS6(self, drop_tol, fill_fac):
        
        if self.basis_u == 2:
            FM_local = self.projectors_3d.I2_inv_approx.dot(self.rhs_FM)
            A_local  = 1/2*(FM_local.T.dot(self.tensor_space.M2) + self.tensor_space.M2.dot(FM_local)).tolil()
        elif self.basis_u == 0:
            FM_local = self.projectors_3d.I0_inv_approx.dot(self.rhs_W)
            FM_local = spa.bmat([[FM_local, None, None], [None, FM_local, None], [None, None, FM_local]], format='csr')
            A_local  = 1/2*(FM_local.T.dot(self.tensor_space.Mv) + self.tensor_space.Mv.dot(FM_local)).tolil()

        del FM_local
        
        
        if self.basis_u == 2:
            FP_local = self.projectors_3d.I2_inv_approx.dot(self.rhs_FP)
            PR_local = self.projectors_3d.I3_inv_approx.dot(self.rhs_PR)

            L_local  = -self.tensor_space.DIV.dot(FP_local) - (self.gamma - 1)*PR_local.dot(self.tensor_space.DIV)

            del FP_local, PR_local

            S6_local = (A_local - self.dt**2/4*self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(L_local))).tolil()

            del A_local
        elif self.basis_u == 0:
            FP_local = self.projectors_3d.I2_inv_approx.dot(self.rhs_FP)
            FG_local = self.projectors_3d.I2_inv_approx.dot(self.rhs_FG)
            PR_local = self.projectors_3d.I3_inv_approx.dot(self.rhs_PR)
            
            L_local  = -self.tensor_space.DIV.dot(FP_local) - (self.gamma - 1)*PR_local.dot(self.tensor_space.DIV.dot(FG_local))

            del FP_local, PR_local

            S6_local = (A_local - self.dt**2/4*FG_local.T.dot(self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(L_local)))).tolil()

            del A_local, FG_local


        # apply boundary conditions to S6_local
        if self.bc[0] == False:
            
            # without polar splines b.c. at eta1 = 0 and eta1 = 1 possible
            if self.tensor_space.polar == False:

                # eta1 = 0
                if self.bc_u1[0] == 'dirichlet':
                    lower = 0
                    upper = self.NbaseD[1]*self.NbaseD[2]

                    S6_local[lower:upper,      :     ] = 0.
                    S6_local[     :     , lower:upper] = 0.
                    S6_local[lower:upper, lower:upper] = np.identity(self.NbaseD[1]*self.NbaseD[2])
                    
                # eta1 = 1
                if self.bc_u1[1] == 'dirichlet':
                    lower = (self.NbaseN[0] - 1)*self.NbaseD[1]*self.NbaseD[2]
                    upper =  self.NbaseN[0]     *self.NbaseD[1]*self.NbaseD[2]
                    
                    S6_local[lower:upper,      :     ] = 0.
                    S6_local[     :     , lower:upper] = 0.
                    S6_local[lower:upper, lower:upper] = np.identity(self.NbaseD[1]*self.NbaseD[2])

            # for polar splines only b.c. at eta1 = 1 possible
            else:
                if self.bc_u1[1] == 'dirichlet':
                    lower = 2*self.NbaseD[2] + (self.NbaseN[0] - 3)*self.NbaseD[1]*self.NbaseD[2]
                    upper = 2*self.NbaseD[2] + (self.NbaseN[0] - 2)*self.NbaseD[1]*self.NbaseD[2]

                    S6_local[lower:upper,      :     ] = 0.
                    S6_local[     :     , lower:upper] = 0.
                    S6_local[lower:upper, lower:upper] = np.identity(self.NbaseD[1]*self.NbaseD[2])


        S6_ILU = spa.linalg.spilu(S6_local.tocsc(), drop_tol=drop_tol , fill_factor=fill_fac)
        self.S6_PRE = spa.linalg.LinearOperator(S6_local.shape, lambda x : S6_ILU.solve(x))