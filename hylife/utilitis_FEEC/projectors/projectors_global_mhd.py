import numpy as np
import scipy.sparse as spa


import hylife.utilitis_FEEC.bsplines as bsp
import hylife.utilitis_FEEC.projectors.kernels_projectors_global_mhd as ker

import source_run.projectors_global as pro
import source_run.kernels_projectors_evaluation as ker_eva



class operators_mhd:
    
    def __init__(self, projectors_3d):
        
        self.projectors_3d = projectors_3d
        
        self.tensor_space  = self.projectors_3d.tensor_space
        
        self.T    = self.tensor_space.T
        self.p    = self.tensor_space.p
        self.t    = self.tensor_space.t
        self.bc   = self.tensor_space.bc
        self.el_b = self.tensor_space.el_b
        
        self.NbaseN = self.tensor_space.NbaseN
        self.NbaseD = self.tensor_space.NbaseD
        
        kind_splines = [False, True]
        
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
    def assemble_rhs_TAU(self, b2_eq, kind_map, params_map=None, tensor_space_F=None, cx=None, cy=None, cz=None):
        
        b2_1_eq, b2_2_eq, b2_3_eq = self.tensor_space.unravel_2form(self.tensor_space.E2.T.dot(b2_eq))

        # create dummy variables
        if kind_map == 0:
            T_F        =  tensor_space_F.T
            p_F        =  tensor_space_F.p
            NbaseN_F   =  tensor_space_F.NbaseN
            params_map =  np.zeros((1,     ), dtype=float)
        else:
            T_F        = [np.zeros((1,     ), dtype=float), np.zeros(1, dtype=float), np.zeros(1, dtype=float)]
            p_F        =  np.zeros((1,     ), dtype=int)
            NbaseN_F   =  np.zeros((1,     ), dtype=int)
            cx         =  np.zeros((1, 1, 1), dtype=float)
            cy         =  np.zeros((1, 1, 1), dtype=float)
            cz         =  np.zeros((1, 1, 1), dtype=float)

        # ====================== 12 - block ([his, int, int] of DND) ===========================

        # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
        B3_eq   = self.tensor_space.evaluate_DDN(self.pts[0].flatten(), self.x_int[1], self.x_int[2], b2_3_eq).reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.x_int[2].size)

        # evaluate Jacobian determinant at at interpolation and quadrature points
        det_dF  = np.empty((self.pts[0].flatten().size, self.x_int[1].size, self.x_int[2].size), dtype=float)

        ker_eva.kernel_eva(self.pts[0].flatten(), self.x_int[1], self.x_int[2], det_dF, 51, kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)

        det_dF  = det_dF.reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.x_int[2].size)
        
        # assemble sparse matrix
        values  = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_D_i[0].size, dtype=float)
        row_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_D_i[0].size, dtype=int)
        col_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi0_z_D_i[0].size, dtype=int)
        
        ker.rhs11(self.pi1_x_D_i[0], self.pi0_y_N_i[0], self.pi0_z_D_i[0], self.pi1_x_D_i[1], self.pi0_y_N_i[1], self.pi0_z_D_i[1], self.subs[0], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), self.wts[0], self.basis_his_D[0], self.basis_int_N[1], self.basis_int_D[2], self.NbaseN, self.NbaseD, -B3_eq/det_dF, values, row_all, col_all)
        
        T_12 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[0], self.tensor_space.Ntot_2form[1]))
        
        # ====================== 13 - block ([his, int, int] of DDN) ===========================
        
        # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
        B2_eq   = self.tensor_space.evaluate_DND(self.pts[0].flatten(), self.x_int[1], self.x_int[2], b2_2_eq).reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.x_int[2].size)
        
        # evaluate Jacobian determinant at at interpolation and quadrature points
        det_dF  = np.empty((self.pts[0].flatten().size, self.x_int[1].size, self.x_int[2].size), dtype=float)

        ker_eva.kernel_eva(self.pts[0].flatten(), self.x_int[1], self.x_int[2], det_dF, 51, kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)

        det_dF  = det_dF.reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.x_int[2].size)
        
        # assemble sparse matrix
        values  = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
        row_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
        col_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
        
        ker.rhs11(self.pi1_x_D_i[0], self.pi0_y_D_i[0], self.pi0_z_N_i[0], self.pi1_x_D_i[1], self.pi0_y_D_i[1], self.pi0_z_N_i[1], self.subs[0], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), self.wts[0], self.basis_his_D[0], self.basis_int_D[1], self.basis_int_N[2], self.NbaseN, self.NbaseD,  B2_eq/det_dF, values, row_all, col_all)
        
        T_13 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[0], self.tensor_space.Ntot_2form[2]))
        
        
        # ====================== 21 - block ([int, his, int] of NDD) ===========================
        
        # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
        B3_eq   = self.tensor_space.evaluate_DDN(self.x_int[0], self.pts[1].flatten(), self.x_int[2], b2_3_eq).reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)
        
        # evaluate Jacobian determinant at at interpolation and quadrature points
        det_dF  = np.empty((self.x_int[0].size, self.pts[1].flatten().size, self.x_int[2].size), dtype=float)

        ker_eva.kernel_eva(self.x_int[0], self.pts[1].flatten(), self.x_int[2], det_dF, 51, kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)

        det_dF  = det_dF.reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)
        
        # assemble sparse matrix
        values  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_D_i[0].size, dtype=float)
        row_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_D_i[0].size, dtype=int)
        col_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_D_i[0].size, dtype=int)
        
        ker.rhs12(self.pi0_x_N_i[0], self.pi1_y_D_i[0], self.pi0_z_D_i[0], self.pi0_x_N_i[1], self.pi1_y_D_i[1], self.pi0_z_D_i[1], self.subs[1], np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), self.wts[1], self.basis_int_N[0], self.basis_his_D[1], self.basis_int_D[2], self.NbaseN, self.NbaseD,  B3_eq/det_dF, values, row_all, col_all)
        
        T_21 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[1], self.tensor_space.Ntot_2form[0]))
        
        
        # ====================== 23 - block ([int, his, int] of DDN) ===========================
        
        # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
        B1_eq   = self.tensor_space.evaluate_NDD(self.x_int[0], self.pts[1].flatten(), self.x_int[2], b2_1_eq).reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)
        
        # evaluate Jacobian determinant at at interpolation and quadrature points
        det_dF  = np.empty((self.x_int[0].size, self.pts[1].flatten().size, self.x_int[2].size), dtype=float)

        ker_eva.kernel_eva(self.x_int[0], self.pts[1].flatten(), self.x_int[2], det_dF, 51, kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)

        det_dF  = det_dF.reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)
        
        # assemble sparse matrix
        values  = np.empty(self.pi0_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
        row_all = np.empty(self.pi0_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
        col_all = np.empty(self.pi0_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
        
        ker.rhs12(self.pi0_x_D_i[0], self.pi1_y_D_i[0], self.pi0_z_N_i[0], self.pi0_x_D_i[1], self.pi1_y_D_i[1], self.pi0_z_N_i[1], self.subs[1], np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), self.wts[1], self.basis_int_D[0], self.basis_his_D[1], self.basis_int_N[2], self.NbaseN, self.NbaseD,  -B1_eq/det_dF, values, row_all, col_all)
        
        T_23 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[1], self.tensor_space.Ntot_2form[2]))
        
        
        # ====================== 31 - block ([int, int, his] of NDD) ===========================
        
        # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
        B2_eq   = self.tensor_space.evaluate_DND(self.x_int[0], self.x_int[1], self.pts[2].flatten(), b2_2_eq).reshape(self.x_int[0].size, self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])
        
        # evaluate Jacobian determinant at at interpolation and quadrature points
        det_dF  = np.empty((self.x_int[0].size, self.x_int[1].size, self.pts[2].flatten().size), dtype=float)

        ker_eva.kernel_eva(self.x_int[0], self.x_int[1], self.pts[2].flatten(), det_dF, 51, kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)

        det_dF  = det_dF.reshape(self.x_int[0].size, self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])
        
        # assemble sparse matrix
        values  = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
        row_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
        col_all = np.empty(self.pi0_x_N_i[0].size*self.pi0_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
        
        ker.rhs13(self.pi0_x_N_i[0], self.pi0_y_D_i[0], self.pi1_z_D_i[0], self.pi0_x_N_i[1], self.pi0_y_D_i[1], self.pi1_z_D_i[1], self.subs[2], np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[2], self.basis_int_N[0], self.basis_int_D[1], self.basis_his_D[2], self.NbaseN, self.NbaseD,  -B2_eq/det_dF, values, row_all, col_all)
        
        T_31 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[2], self.tensor_space.Ntot_2form[0]))
        
        # ====================== 32 - block ([int, int, his] of DND) ===========================
        
        # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
        B1_eq   = self.tensor_space.evaluate_NDD(self.x_int[0], self.x_int[1], self.pts[2].flatten(), b2_1_eq).reshape(self.x_int[0].size, self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])
        
        # evaluate Jacobian determinant at at interpolation and quadrature points
        det_dF  = np.empty((self.x_int[0].size, self.x_int[1].size, self.pts[2].flatten().size), dtype=float)

        ker_eva.kernel_eva(self.x_int[0], self.x_int[1], self.pts[2].flatten(), det_dF, 51, kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)

        det_dF  = det_dF.reshape(self.x_int[0].size, self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])
        
        # assemble sparse matrix
        values  = np.empty(self.pi0_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
        row_all = np.empty(self.pi0_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
        col_all = np.empty(self.pi0_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
        
        ker.rhs13(self.pi0_x_D_i[0], self.pi0_y_N_i[0], self.pi1_z_D_i[0], self.pi0_x_D_i[1], self.pi0_y_N_i[1], self.pi1_z_D_i[1], self.subs[2], np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[2], self.basis_int_D[0], self.basis_int_N[1], self.basis_his_D[2], self.NbaseN, self.NbaseD,  B1_eq/det_dF, values, row_all, col_all)
        
        T_32 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_1form[2], self.tensor_space.Ntot_2form[1]))

        self.rhs_TAU = self.projectors_3d.P1.dot(spa.bmat([[None, T_12, T_13], [T_21, None, T_23], [T_31, T_32, None]], format='csr').dot(self.tensor_space.E2.T))
        self.rhs_TAU.eliminate_zeros()
    
    
    # =================================================================
    def assemble_rhs_WS(self, coeff3_eq, which, kind_map, params_map=None, tensor_space_F=None, cx=None, cy=None, cz=None):
        
        coeff3_eq = self.tensor_space.E3.T.dot(coeff3_eq).reshape(self.tensor_space.Nbase_3form)

        # create dummy variables
        if kind_map == 0:
            T_F        =  tensor_space_F.T
            p_F        =  tensor_space_F.p
            NbaseN_F   =  tensor_space_F.NbaseN
            params_map =  np.zeros((1,     ), dtype=float)
        else:
            T_F        = [np.zeros((1,     ), dtype=float), np.zeros(1, dtype=float), np.zeros(1, dtype=float)]
            p_F        =  np.zeros((1,     ), dtype=int)
            NbaseN_F   =  np.zeros((1,     ), dtype=int)
            cx         =  np.zeros((1, 1, 1), dtype=float)
            cy         =  np.zeros((1, 1, 1), dtype=float)
            cz         =  np.zeros((1, 1, 1), dtype=float)
            
            
        # ====================== 11 - block ([int, his, his] of NDD) ===========================

        # evaluate equilibrium density at interpolation and quadrature points
        EQ = self.tensor_space.evaluate_DDD(self.x_int[0], self.pts[1].flatten(), self.pts[2].flatten(), coeff3_eq).reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.pts[2].shape[0], self.pts[2].shape[1])

        # evaluate Jacobian determinant at at interpolation and quadrature points
        det_dF  = np.empty((self.x_int[0].size, self.pts[1].flatten().size, self.pts[2].flatten().size), dtype=float)

        ker_eva.kernel_eva(self.x_int[0], self.pts[1].flatten(), self.pts[2].flatten(), det_dF, 51, kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)

        det_dF  = det_dF.reshape(self.x_int[0].size, self.pts[1].shape[0], self.pts[1].shape[1], self.pts[2].shape[0], self.pts[2].shape[1])
        
        # assemble sparse matrix
        values  = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
        row_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
        col_all = np.empty(self.pi0_x_N_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
        
        ker.rhs21(self.pi0_x_N_i[0], self.pi1_y_D_i[0], self.pi1_z_D_i[0], self.pi0_x_N_i[1], self.pi1_y_D_i[1], self.pi1_z_D_i[1], self.subs[1], self.subs[2], np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[1], self.wts[2], self.basis_int_N[0], self.basis_his_D[1], self.basis_his_D[2], self.NbaseN, self.NbaseD, EQ/det_dF, values, row_all, col_all)
        
        W_11 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[0], self.tensor_space.Ntot_2form[0]))
        
        
        # ====================== 22 - block ([his, int, his] of DND) ===========================

        # evaluate equilibrium density at interpolation and quadrature points
        EQ = self.tensor_space.evaluate_DDD(self.pts[0].flatten(), self.x_int[1], self.pts[2].flatten(), coeff3_eq).reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])

        # evaluate Jacobian determinant at at interpolation and quadrature points
        det_dF  = np.empty((self.pts[0].flatten().size, self.x_int[1].size, self.pts[2].flatten().size), dtype=float)

        ker_eva.kernel_eva(self.pts[0].flatten(), self.x_int[1], self.pts[2].flatten(), det_dF, 51, kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)

        det_dF  = det_dF.reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.x_int[1].size, self.pts[2].shape[0], self.pts[2].shape[1])
        
        # assemble sparse matrix
        values  = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
        row_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
        col_all = np.empty(self.pi1_x_D_i[0].size*self.pi0_y_N_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
        
        ker.rhs22(self.pi1_x_D_i[0], self.pi0_y_N_i[0], self.pi1_z_D_i[0], self.pi1_x_D_i[1], self.pi0_y_N_i[1], self.pi1_z_D_i[1], self.subs[0], self.subs[2], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[0], self.wts[2], self.basis_his_D[0], self.basis_int_N[1], self.basis_his_D[2], self.NbaseN, self.NbaseD, EQ/det_dF, values, row_all, col_all)
        
        W_22 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[1], self.tensor_space.Ntot_2form[1]))
        
        
        # ====================== 33 - block ([his, his, int] of DDN) ===========================

        # evaluate equilibrium density at interpolation and quadrature points
        EQ = self.tensor_space.evaluate_DDD(self.pts[0].flatten(), self.pts[1].flatten(), self.x_int[2], coeff3_eq).reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)

        # evaluate Jacobian determinant at at interpolation and quadrature points
        det_dF  = np.empty((self.pts[0].flatten().size, self.pts[1].flatten().size, self.x_int[2].size), dtype=float)

        ker_eva.kernel_eva(self.pts[0].flatten(), self.pts[1].flatten(), self.x_int[2], det_dF, 51, kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)

        det_dF  = det_dF.reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.pts[1].shape[0], self.pts[1].shape[1], self.x_int[2].size)
        
        # assemble sparse matrix
        values  = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=float)
        row_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
        col_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi0_z_N_i[0].size, dtype=int)
        
        ker.rhs23(self.pi1_x_D_i[0], self.pi1_y_D_i[0], self.pi0_z_N_i[0], self.pi1_x_D_i[1], self.pi1_y_D_i[1], self.pi0_z_N_i[1], self.subs[0], self.subs[1], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), self.wts[0], self.wts[1], self.basis_his_D[0], self.basis_his_D[1], self.basis_int_N[2], self.NbaseN, self.NbaseD, EQ/det_dF, values, row_all, col_all)
        
        W_33 = spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_2form[2], self.tensor_space.Ntot_2form[2]))
        
        if which == 'W':
            self.rhs_W = self.projectors_3d.P2.dot(spa.bmat([[W_11, None, None], [None, W_22, None], [None, None, W_33]], format='csr').dot(self.tensor_space.E2.T))
            self.rhs_W.eliminate_zeros()
        elif which == 'S':
            self.rhs_S = self.projectors_3d.P2.dot(spa.bmat([[W_11, None, None], [None, W_22, None], [None, None, W_33]], format='csr').dot(self.tensor_space.E2.T))
            self.rhs_S.eliminate_zeros()
    
    
    # =================================================================
    def assemble_rhs_K(self, p3_eq, kind_map, params_map=None, tensor_space_F=None, cx=None, cy=None, cz=None):
        
        p3_eq = self.tensor_space.E3.T.dot(p3_eq).reshape(self.tensor_space.Nbase_3form)

        # create dummy variables
        if kind_map == 0:
            T_F        =  tensor_space_F.T
            p_F        =  tensor_space_F.p
            NbaseN_F   =  tensor_space_F.NbaseN
            params_map =  np.zeros((1,     ), dtype=float)
        else:
            T_F        = [np.zeros((1,     ), dtype=float), np.zeros(1, dtype=float), np.zeros(1, dtype=float)]
            p_F        =  np.zeros((1,     ), dtype=int)
            NbaseN_F   =  np.zeros((1,     ), dtype=int)
            cx         =  np.zeros((1, 1, 1), dtype=float)
            cy         =  np.zeros((1, 1, 1), dtype=float)
            cz         =  np.zeros((1, 1, 1), dtype=float)
            
            
        # ====================== ([his, his, his] of DDD) ===========================
        # evaluate equilibrium pressure at quadrature points
        PR_eq = self.tensor_space.evaluate_DDD(self.pts[0].flatten(), self.pts[1].flatten(), self.pts[2].flatten(), p3_eq).reshape(self.pts[0].shape[0], self.pts[0].shape[1], self.pts[1].shape[0], self.pts[1].shape[1], self.pts[2].shape[0], self.pts[2].shape[1])
        
        # assemble sparse matrix
        values  = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=float)
        row_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
        col_all = np.empty(self.pi1_x_D_i[0].size*self.pi1_y_D_i[0].size*self.pi1_z_D_i[0].size, dtype=int)
        
        ker.rhs3(self.pi1_x_D_i[0], self.pi1_y_D_i[0], self.pi1_z_D_i[0], self.pi1_x_D_i[1], self.pi1_y_D_i[1], self.pi1_z_D_i[1], self.subs[0], self.subs[1], self.subs[2], np.append(0, np.cumsum(self.subs[0] - 1)[:-1]), np.append(0, np.cumsum(self.subs[1] - 1)[:-1]), np.append(0, np.cumsum(self.subs[2] - 1)[:-1]), self.wts[0], self.wts[1], self.wts[2], self.basis_his_D[0], self.basis_his_D[1], self.basis_his_D[2], self.NbaseN, self.NbaseD, PR_eq, values, row_all, col_all)
        
        self.rhs_K = self.projectors_3d.P3.dot(spa.csr_matrix((values, (row_all, col_all)), shape=(self.tensor_space.Ntot_3form, self.tensor_space.Ntot_3form)).dot(self.tensor_space.E3.T))
        self.rhs_K.eliminate_zeros()
        
    
    
    # =================================================================
    def assemble_TF(self, f1, kind_map, params_map=None, tensor_space_F=None, cx=None, cy=None, cz=None):
        
        # apply transposed projection extraction operator
        f1_1, f1_2, f1_3 = self.tensor_space.unravel_1form(self.projectors_3d.P1.T.dot(self.projectors_3d.apply_IinvT_V1(f1)))
        
        # create dummy variables
        if kind_map == 0:
            T_F        =  tensor_space_F.T
            p_F        =  tensor_space_F.p
            NbaseN_F   =  tensor_space_F.NbaseN
            params_map =  np.zeros((1,     ), dtype=float)
        else:
            T_F        = [np.zeros((1,     ), dtype=float), np.zeros(1, dtype=float), np.zeros(1, dtype=float)]
            p_F        =  np.zeros((1,     ), dtype=int)
            NbaseN_F   =  np.zeros((1,     ), dtype=int)
            cx         =  np.zeros((1, 1, 1), dtype=float)
            cy         =  np.zeros((1, 1, 1), dtype=float)
            cz         =  np.zeros((1, 1, 1), dtype=float)
            
            
        # evaluate Jacobian determinant at point sets
        det_DF_hii = np.empty((self.pts[0].flatten().size, self.x_int[1].size, self.x_int[2].size), dtype=float)
        det_DF_ihi = np.empty((self.x_int[0].size, self.pts[1].flatten().size, self.x_int[2].size), dtype=float)
        det_DF_iih = np.empty((self.x_int[0].size, self.x_int[1].size, self.pts[2].flatten().size), dtype=float)
        
        ker_eva.kernel_eva(self.pts[0].flatten(), self.x_int[1], self.x_int[2], det_DF_hii, 51, kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)
        
        ker_eva.kernel_eva(self.x_int[0], self.pts[1].flatten(), self.x_int[2], det_DF_ihi, 51, kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)
        
        ker_eva.kernel_eva(self.x_int[0], self.x_int[1], self.pts[2].flatten(), det_DF_iih, 51, kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)
        
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
    def __TAU(self, u):
        return self.projectors_3d.solve_V1(self.rhs_TAU.dot(u))
    
    # ======================================
    def __TAU_transposed(self, e):
        return self.rhs_TAU.T.dot(self.projectors_3d.apply_IinvT_V1(e))
    
    # ======================================
    def __W(self, u):
        return self.projectors_3d.solve_V2(self.rhs_W.dot(u))
    
    # ======================================
    def __W_transposed(self, u):
        return self.rhs_W.T.dot(self.projectors_3d.apply_IinvT_V2(u))
    
    # ======================================
    def __S(self, u):
        return self.projectors_3d.solve_V2(self.rhs_S.dot(u))
    
    # ======================================
    def __S_transposed(self, u):
        return self.rhs_S.T.dot(self.projectors_3d.apply_IinvT_V2(u))
    
    # ======================================
    def __K(self, f3):
        return self.projectors_3d.solve_V3(self.rhs_K.dot(f3))
    
    # ======================================
    def __K_transposed(self, f3):
        return self.rhs_K.T.dot(self.projectors_3d.apply_IinvT_V3(f3))
    
    # ======================================
    def __TF(self, b2):
        return self.mat_TF.dot(b2)
    
    # ======================================
    def __TF_transposed(self, u2):
        return self._mat_TF.T.dot(u2)
    
    # ======================================
    def setOperators(self):
        
        self.W   = spa.linalg.LinearOperator((self.tensor_space.E2.shape[0], self.tensor_space.E2.shape[0]), matvec=self.__W, rmatvec=self.__W_transposed)
        
        self.S   = spa.linalg.LinearOperator((self.tensor_space.E2.shape[0], self.tensor_space.E2.shape[0]), matvec=self.__S, rmatvec=self.__S_transposed)
        
        self.TAU = spa.linalg.LinearOperator((self.tensor_space.E1.shape[0], self.tensor_space.E2.shape[0]), matvec=self.__TAU, rmatvec=self.__TAU_transposed)
        
        self.K   = spa.linalg.LinearOperator((self.tensor_space.E3.shape[0], self.tensor_space.E3.shape[0]), matvec=self.__K, rmatvec=self.__K_transposed)
        
        self.TF  = spa.linalg.LinearOperator((self.tensor_space.E2.shape[0], self.tensor_space.E2.shape[0]), matvec=self.__TF, rmatvec=self.__TF_transposed)