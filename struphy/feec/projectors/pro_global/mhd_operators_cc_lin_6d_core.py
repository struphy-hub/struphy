# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Class for 2D/3D linear MHD projection operators.
"""


import numpy as np
import scipy.sparse as spa

import struphy.feec.projectors.pro_global.kernels_projectors_global_mhd as ker

import struphy.feec.basics.mass_matrices_2d as mass_2d
import struphy.feec.basics.mass_matrices_3d as mass_3d


class MHD_operators:
    """
    TODO
    """
    
    def __init__(self, space, equilibrium, basis_u):
        
        # tensor-product spline space (either 3D or 2D x Fourier)
        self.space = space
        
        # MHD equilibrium object for evaluation of equilibrium fields
        self.equilibrium = equilibrium
        
        # bulk veloctiy formulation (either vector field, 1-form or 2-form)
        self.basis_u = basis_u
        
        # get 1D interpolation points (copies) and shift first point in eta_1 direction for polar domains
        self.eta_int = [space.projectors.x_int.copy() for space in self.space.spaces]
        
        if self.space.ck == 0 or self.space.ck == 1:
            self.eta_int[0][0] += 0.00001
        
        self.nint = [eta_int.size for eta_int in self.eta_int]
        
        # get 1D quadrature points and weights
        self.eta_his = [space.projectors.pts for space in self.space.spaces]
        self.wts     = [space.projectors.wts for space in self.space.spaces]
        
        self.nhis    = [eta_his.shape[0] for eta_his in self.eta_his]
        self.nq      = [eta_his.shape[1] for eta_his in self.eta_his]
       
        # get 1D number of sub-integration intervals
        self.subs     = [space.projectors.subs     for space in self.space.spaces]
        self.subs_cum = [space.projectors.subs_cum for space in self.space.spaces]
        
        # get 1D indices of non-vanishing values of expressions dofs_0(N), dofs_0(D), dofs_1(N) and dofs_1(D)
        self.dofs_0_N_i = [np.nonzero(space.projectors.I.toarray()) for space in self.space.spaces]
        self.dofs_1_D_i = [np.nonzero(space.projectors.H.toarray()) for space in self.space.spaces]
        
        self.dofs_0_D_i = [np.nonzero(space.projectors.ID.toarray()) for space in self.space.spaces]
        self.dofs_1_N_i = [np.nonzero(space.projectors.HN.toarray()) for space in self.space.spaces]
        
        # get 1D collocation matrices for interpolation and histopolation
        self.basis_int_N = [space.projectors.N_int.toarray() for space in self.space.spaces]
        self.basis_int_D = [space.projectors.D_int.toarray() for space in self.space.spaces]
        
        self.basis_his_N = [space.projectors.N_pts.toarray().reshape(nhis, nq, space.NbaseN) for space, nhis, nq in zip(self.space.spaces, self.nhis, self.nq)]
        self.basis_his_D = [space.projectors.D_pts.toarray().reshape(nhis, nq, space.NbaseD) for space, nhis, nq in zip(self.space.spaces, self.nhis, self.nq)]
        
        # number of basis functions in third dimension
        self.N3 = self.space.NbaseN[2]
        self.D3 = self.space.NbaseD[2]
    
    
    # =================================================================
    def get_blocks_EF(self, pol=True):
        """
        TODO
        """

        if self.basis_u == 0:

            if pol:

                # evaluation in third direction at eta_3 = 0
                eta3 = np.array([0.])

                # ---------- 12 - block ([his, int] of NN) -----------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_eq_3(self.eta_his[0].flatten(), self.eta_int[1], eta3)[:, :, 0]
                B2_3_pts = B2_3_pts.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs11_2d(self.dofs_1_N_i[0][0], self.dofs_0_N_i[1][0], self.dofs_1_N_i[0][1], self.dofs_0_N_i[1][1], self.subs[0], self.subs_cum[0], self.wts[0], self.basis_his_N[0], self.basis_int_N[1], self.space.NbaseN, self.space.NbaseD, -B2_3_pts, val, row, col)

                EF_12 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[0]//self.N3, self.space.Ntot_0form//self.N3))
                EF_12.eliminate_zeros()
                # ----------------------------------------------------


                # ---------- 13 - block ([his, int] of NN) -----------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_eq_2(self.eta_his[0].flatten(), self.eta_int[1], eta3)[:, :, 0]
                B2_2_pts = B2_2_pts.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs11_2d(self.dofs_1_N_i[0][0], self.dofs_0_N_i[1][0], self.dofs_1_N_i[0][1], self.dofs_0_N_i[1][1], self.subs[0], self.subs_cum[0], self.wts[0], self.basis_his_N[0], self.basis_int_N[1], self.space.NbaseN, self.space.NbaseD,  B2_2_pts, val, row, col)

                EF_13 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[0]//self.N3, self.space.Ntot_0form//self.N3))
                EF_13.eliminate_zeros()
                # ----------------------------------------------------


                # ---------- 21 - block ([int, his] of NN) ----------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_eq_3(self.eta_int[0], self.eta_his[1].flatten(), eta3)[:, :, 0]
                B2_3_pts = B2_3_pts.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size, dtype=int)

                ker.rhs12_2d(self.dofs_0_N_i[0][0], self.dofs_1_N_i[1][0], self.dofs_0_N_i[0][1], self.dofs_1_N_i[1][1], self.subs[1], self.subs_cum[1], self.wts[1], self.basis_int_N[0], self.basis_his_N[1], self.space.NbaseN, self.space.NbaseD,  B2_3_pts, val, row, col)

                EF_21 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[1]//self.N3, self.space.Ntot_0form//self.N3))
                EF_21.eliminate_zeros()
                # ----------------------------------------------------


                # ---------- 23 - block ([int, his] of NN) ----------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_eq_1(self.eta_int[0], self.eta_his[1].flatten(), eta3)[:, :, 0]
                B2_1_pts = B2_1_pts.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size, dtype=int)

                ker.rhs12_2d(self.dofs_0_N_i[0][0], self.dofs_1_N_i[1][0], self.dofs_0_N_i[0][1], self.dofs_1_N_i[1][1], self.subs[1], self.subs_cum[1], self.wts[1], self.basis_int_N[0], self.basis_his_N[1], self.space.NbaseN, self.space.NbaseD, -B2_1_pts, val, row, col)

                EF_23 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[1]//self.N3, self.space.Ntot_0form//self.N3))
                EF_23.eliminate_zeros()
                # ----------------------------------------------------


                # ---------- 31 - block ([int, int] of NN) -----------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_eq_2(self.eta_int[0], self.eta_int[1], eta3)[:, :, 0]

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs0_2d(self.dofs_0_N_i[0][0], self.dofs_0_N_i[1][0], self.dofs_0_N_i[0][1], self.dofs_0_N_i[1][1], self.basis_int_N[0], self.basis_int_N[1], -B2_2_pts, val, row, col)

                EF_31 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[2]//self.D3, self.space.Ntot_0form//self.N3))
                EF_31.eliminate_zeros()
                # ----------------------------------------------------


                # ---------- 32 - block ([int, int] of NN) ----------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_eq_1(self.eta_int[0], self.eta_int[1], eta3)[:, :, 0]

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs0_2d(self.dofs_0_N_i[0][0], self.dofs_0_N_i[1][0], self.dofs_0_N_i[0][1], self.dofs_0_N_i[1][1], self.basis_int_N[0], self.basis_int_N[1],  B2_1_pts, val, row, col)

                EF_32 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[2]//self.D3, self.space.Ntot_0form//self.N3))
                EF_32.eliminate_zeros()
                # ----------------------------------------------------

            else:

                # ------- 12 - block ([his, int, int] of NNN) --------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_eq_3(self.eta_his[0].flatten(), self.eta_int[1], self.eta_int[2])  
                B2_3_pts = B2_3_pts.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nint[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)

                ker.rhs11(self.dofs_1_N_i[0][0], self.dofs_0_N_i[1][0], self.dofs_0_N_i[2][0], self.dofs_1_N_i[0][1], self.dofs_0_N_i[1][1], self.dofs_0_N_i[2][1], self.subs[0], self.subs_cum[0], self.wts[0], self.basis_his_N[0], self.basis_int_N[1], self.basis_int_N[2], self.space.NbaseN, self.space.NbaseD, -B2_3_pts, val, row, col)

                EF_12 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[0], self.space.Ntot_0form))
                EF_12.eliminate_zeros()
                # ----------------------------------------------------


                # ------- 13 - block ([his, int, int] of NNN) --------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_eq_2(self.eta_his[0].flatten(), self.eta_int[1], self.eta_int[2])
                B2_2_pts = B2_2_pts.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nint[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)

                ker.rhs11(self.dofs_1_N_i[0][0], self.dofs_0_N_i[1][0], self.dofs_0_N_i[2][0], self.dofs_1_N_i[0][1], self.dofs_0_N_i[1][1], self.dofs_0_N_i[2][1], self.subs[0], self.subs_cum[0], self.wts[0], self.basis_his_N[0], self.basis_int_N[1], self.basis_int_N[2], self.space.NbaseN, self.space.NbaseD,  B2_2_pts, val, row, col)

                EF_13 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[0], self.space.Ntot_0form))
                EF_13.eliminate_zeros()
                # ----------------------------------------------------


                # ------- 21 - block ([int, his, int] of NNN) --------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_eq_3(self.eta_int[0], self.eta_his[1].flatten(), self.eta_int[2])
                B2_3_pts = B2_3_pts.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nint[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)

                ker.rhs12(self.dofs_0_N_i[0][0], self.dofs_1_N_i[1][0], self.dofs_0_N_i[2][0], self.dofs_0_N_i[0][1], self.dofs_1_N_i[1][1], self.dofs_0_N_i[2][1], self.subs[1], self.subs_cum[1], self.wts[1], self.basis_int_N[0], self.basis_his_N[1], self.basis_int_N[2], self.space.NbaseN, self.space.NbaseD,  B2_3_pts, val, row, col)

                EF_21 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[1], self.space.Ntot_0form))
                EF_21.eliminate_zeros()
                # ----------------------------------------------------


                # ------- 23 - block ([int, his, int] of NNN) --------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_eq_1(self.eta_int[0], self.eta_his[1].flatten(), self.eta_int[2])
                B2_1_pts = B2_1_pts.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nint[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)

                ker.rhs12(self.dofs_0_N_i[0][0], self.dofs_1_N_i[1][0], self.dofs_0_N_i[2][0], self.dofs_0_N_i[0][1], self.dofs_1_N_i[1][1], self.dofs_0_N_i[2][1], self.subs[1], self.subs_cum[1], self.wts[1], self.basis_int_N[0], self.basis_his_N[1], self.basis_int_N[2], self.space.NbaseN, self.space.NbaseD, -B2_1_pts, val, row, col)

                EF_23 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[1], self.space.Ntot_0form))
                EF_23.eliminate_zeros()
                # ----------------------------------------------------


                # ------- 31 - block ([int, int, his] of NNN) --------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_eq_2(self.eta_int[0], self.eta_int[1], self.eta_his[2].flatten())
                B2_2_pts = B2_2_pts.reshape(self.nint[0], self.nint[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_N_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_N_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_N_i[2][0].size, dtype=int)

                ker.rhs13(self.dofs_0_N_i[0][0], self.dofs_0_N_i[1][0], self.dofs_1_N_i[2][0], self.dofs_0_N_i[0][1], self.dofs_0_N_i[1][1], self.dofs_1_N_i[2][1], self.subs[2], self.subs_cum[2], self.wts[2], self.basis_int_N[0], self.basis_int_N[1], self.basis_his_N[2], self.space.NbaseN, self.space.NbaseD, -B2_2_pts, val, row, col)

                EF_31 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[2], self.space.Ntot_0form))
                EF_31.eliminate_zeros()
                # ----------------------------------------------------


                # ------- 32 - block ([int, int, his] of NNN) --------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_eq_1(self.eta_int[0], self.eta_int[1], self.eta_his[2].flatten())
                B2_1_pts = B2_1_pts.reshape(self.nint[0], self.nint[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_N_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_N_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_N_i[2][0].size, dtype=int)

                ker.rhs13(self.dofs_0_N_i[0][0], self.dofs_0_N_i[1][0], self.dofs_1_N_i[2][0], self.dofs_0_N_i[0][1], self.dofs_0_N_i[1][1], self.dofs_1_N_i[2][1], self.subs[2], self.subs_cum[2], self.wts[2], self.basis_int_N[0], self.basis_int_N[1], self.basis_his_N[2], self.space.NbaseN, self.space.NbaseD,  B2_1_pts, val, row, col)

                EF_32 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[2], self.space.Ntot_0form))
                EF_32.eliminate_zeros()
                # ----------------------------------------------------


        elif self.basis_u == 1:
            print('1-form MHD is not yet implemented')

        elif self.basis_u == 2:

            if pol:

                # evaluation in third direction at eta_3 = 0
                eta3 = np.array([0.])

                # ---------- 12 - block ([his, int] of DN) -----------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_eq_3(self.eta_his[0].flatten(), self.eta_int[1], eta3)[:, :, 0]
                B2_3_pts = B2_3_pts.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_his[0].flatten(), self.eta_int[1], eta3, 'det_df'))[:, :, 0]
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs11_2d(self.dofs_1_D_i[0][0], self.dofs_0_N_i[1][0], self.dofs_1_D_i[0][1], self.dofs_0_N_i[1][1], self.subs[0], self.subs_cum[0], self.wts[0], self.basis_his_D[0], self.basis_int_N[1], self.space.NbaseN, self.space.NbaseD, -B2_3_pts/det_dF, val, row, col)

                EF_12 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[0]//self.N3, self.space.Ntot_2form[1]//self.D3))
                EF_12.eliminate_zeros()
                # ----------------------------------------------------


                # ---------- 13 - block ([his, int] of DD) -----------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_eq_2(self.eta_his[0].flatten(), self.eta_int[1], eta3)[:, :, 0]
                B2_2_pts = B2_2_pts.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_his[0].flatten(), self.eta_int[1], eta3, 'det_df'))[:, :, 0]
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_D_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_D_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_D_i[1][0].size, dtype=int)

                ker.rhs11_2d(self.dofs_1_D_i[0][0], self.dofs_0_D_i[1][0], self.dofs_1_D_i[0][1], self.dofs_0_D_i[1][1], self.subs[0], self.subs_cum[0], self.wts[0], self.basis_his_D[0], self.basis_int_D[1], self.space.NbaseN, self.space.NbaseD,  B2_2_pts/det_dF, val, row, col)

                EF_13 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[0]//self.N3, self.space.Ntot_2form[2]//self.N3))
                EF_13.eliminate_zeros()
                # ----------------------------------------------------


                # ---------- 21 - block ([int, his] of ND) -----------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_eq_3(self.eta_int[0], self.eta_his[1].flatten(), eta3)[:, :, 0]
                B2_3_pts = B2_3_pts.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_int[0], self.eta_his[1].flatten(), eta3, 'det_df'))[:, :, 0]
                det_dF = det_dF.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=int)

                ker.rhs12_2d(self.dofs_0_N_i[0][0], self.dofs_1_D_i[1][0], self.dofs_0_N_i[0][1], self.dofs_1_D_i[1][1], self.subs[1], self.subs_cum[1], self.wts[1], self.basis_int_N[0], self.basis_his_D[1], self.space.NbaseN, self.space.NbaseD,  B2_3_pts/det_dF, val, row, col)

                EF_21 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[1]//self.N3, self.space.Ntot_2form[0]//self.D3))
                EF_21.eliminate_zeros()
                # ----------------------------------------------------


                # ---------- 23 - block ([int, his] of DD) -----------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_eq_1(self.eta_int[0], self.eta_his[1].flatten(), eta3)[:, :, 0]
                B2_1_pts = B2_1_pts.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_int[0], self.eta_his[1].flatten(), eta3, 'det_df'))[:, :, 0]
                det_dF = det_dF.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_D_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_0_D_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_0_D_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=int)

                ker.rhs12_2d(self.dofs_0_D_i[0][0], self.dofs_1_D_i[1][0], self.dofs_0_D_i[0][1], self.dofs_1_D_i[1][1], self.subs[1], self.subs_cum[1], self.wts[1], self.basis_int_D[0], self.basis_his_D[1], self.space.NbaseN, self.space.NbaseD, -B2_1_pts/det_dF, val, row, col)

                EF_23 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[1]//self.N3, self.space.Ntot_2form[2]//self.N3))
                EF_23.eliminate_zeros()
                # ----------------------------------------------------


                # ---------- 31 - block ([int, int] of ND) -----------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_eq_2(self.eta_int[0], self.eta_int[1], eta3)[:, :, 0]

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_int[0], self.eta_int[1], eta3, 'det_df'))[:, :, 0]

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_D_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_D_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_D_i[1][0].size, dtype=int)

                ker.rhs0_2d(self.dofs_0_N_i[0][0], self.dofs_0_D_i[1][0], self.dofs_0_N_i[0][1], self.dofs_0_D_i[1][1], self.basis_int_N[0], self.basis_int_D[1], -B2_2_pts/det_dF, val, row, col)

                EF_31 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[2]//self.D3, self.space.Ntot_2form[0]//self.D3))
                EF_31.eliminate_zeros()
                # ----------------------------------------------------


                # ---------- 32 - block ([int, int] of DN) -----------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_eq_1(self.eta_int[0], self.eta_int[1], eta3)[:, :, 0]

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_int[0], self.eta_int[1], eta3, 'det_df'))[:, :, 0]

                # assemble sparse matrix
                val = np.empty(self.dofs_0_D_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_0_D_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_0_D_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs0_2d(self.dofs_0_D_i[0][0], self.dofs_0_N_i[1][0], self.dofs_0_D_i[0][1], self.dofs_0_N_i[1][1], self.basis_int_D[0], self.basis_int_N[1],  B2_1_pts/det_dF, val, row, col)

                EF_32 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[2]//self.D3, self.space.Ntot_2form[1]//self.D3))
                EF_32.eliminate_zeros()
                # ----------------------------------------------------

            else:

                # ------- 12 - block ([his, int, int] of DND) --------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_eq_3(self.eta_his[0].flatten(), self.eta_int[1], self.eta_int[2])
                B2_3_pts = B2_3_pts.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nint[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_his[0].flatten(), self.eta_int[1], self.eta_int[2], 'det_df'))
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nint[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_0_D_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_0_D_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_0_D_i[2][0].size, dtype=int)

                ker.rhs11(self.dofs_1_D_i[0][0], self.dofs_0_N_i[1][0], self.dofs_0_D_i[2][0], self.dofs_1_D_i[0][1], self.dofs_0_N_i[1][1], self.dofs_0_D_i[2][1], self.subs[0], self.subs_cum[0], self.wts[0], self.basis_his_D[0], self.basis_int_N[1], self.basis_int_D[2], self.space.NbaseN, self.space.NbaseD, -B2_3_pts/det_dF, val, row, col)

                EF_12 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[0], self.space.Ntot_2form[1]))
                EF_12.eliminate_zeros()
                # ----------------------------------------------------


                # ------- 13 - block ([his, int, int] of DDN) --------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_eq_2(self.eta_his[0].flatten(), self.eta_int[1], self.eta_int[2])
                B2_2_pts = B2_2_pts.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nint[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_his[0].flatten(), self.eta_int[1], self.eta_int[2], 'det_df'))
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nint[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_D_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_D_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_D_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)

                ker.rhs11(self.dofs_1_D_i[0][0], self.dofs_0_D_i[1][0], self.dofs_0_N_i[2][0], self.dofs_1_D_i[0][1], self.dofs_0_D_i[1][1], self.dofs_0_N_i[2][1], self.subs[0], self.subs_cum[0], self.wts[0], self.basis_his_D[0], self.basis_int_D[1], self.basis_int_N[2], self.space.NbaseN, self.space.NbaseD,  B2_2_pts/det_dF, val, row, col)

                EF_13 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[0], self.space.Ntot_2form[2]))
                EF_13.eliminate_zeros()
                # ----------------------------------------------------


                # ------- 21 - block ([int, his, int] of NDD) --------
                # evaluate equilibrium magnetic field (3-component) at interpolation and quadrature points
                B2_3_pts = self.equilibrium.b2_eq_3(self.eta_int[0], self.eta_his[1].flatten(), self.eta_int[2])
                B2_3_pts = B2_3_pts.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nint[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_int[0], self.eta_his[1].flatten(), self.eta_int[2], 'det_df'))
                det_dF = det_dF.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nint[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_0_D_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_0_D_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_0_D_i[2][0].size, dtype=int)

                ker.rhs12(self.dofs_0_N_i[0][0], self.dofs_1_D_i[1][0], self.dofs_0_D_i[2][0], self.dofs_0_N_i[0][1], self.dofs_1_D_i[1][1], self.dofs_0_D_i[2][1], self.subs[1], self.subs_cum[1], self.wts[1], self.basis_int_N[0], self.basis_his_D[1], self.basis_int_D[2], self.space.NbaseN, self.space.NbaseD,  B2_3_pts/det_dF, val, row, col)

                EF_21 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[1], self.space.Ntot_2form[0]))
                EF_21.eliminate_zeros()
                # ----------------------------------------------------


                # ------- 23 - block ([int, his, int] of DDN) --------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_eq_1(self.eta_int[0], self.eta_his[1].flatten(), self.eta_int[2])
                B2_1_pts = B2_1_pts.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nint[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_int[0], self.eta_his[1].flatten(), self.eta_int[2], 'det_df'))
                det_dF = det_dF.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nint[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_D_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_0_D_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_0_D_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)

                ker.rhs12(self.dofs_0_D_i[0][0], self.dofs_1_D_i[1][0], self.dofs_0_N_i[2][0], self.dofs_0_D_i[0][1], self.dofs_1_D_i[1][1], self.dofs_0_N_i[2][1], self.subs[1], self.subs_cum[1], self.wts[1], self.basis_int_D[0], self.basis_his_D[1], self.basis_int_N[2], self.space.NbaseN, self.space.NbaseD, -B2_1_pts/det_dF, val, row, col)

                EF_23 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[1], self.space.Ntot_2form[2]))
                EF_23.eliminate_zeros()
                # ----------------------------------------------------


                # ------- 31 - block ([int, int, his] of NDD) --------
                # evaluate equilibrium magnetic field (2-component) at interpolation and quadrature points
                B2_2_pts = self.equilibrium.b2_eq_2(self.eta_int[0], self.eta_int[1], self.eta_his[2].flatten())
                B2_2_pts = B2_2_pts.reshape(self.nint[0], self.nint[1], self.nhis[2], self.nq[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_int[0], self.eta_int[1], self.eta_his[2].flatten(), 'det_df'))
                det_dF = det_dF.reshape(self.nint[0], self.nint[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_D_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_D_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_0_D_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=int)

                ker.rhs13(self.dofs_0_N_i[0][0], self.dofs_0_D_i[1][0], self.dofs_1_D_i[2][0], self.dofs_0_N_i[0][1], self.dofs_0_D_i[1][1], self.dofs_1_D_i[2][1], self.subs[2], self.subs_cum[2], self.wts[2], self.basis_int_N[0], self.basis_int_D[1], self.basis_his_D[2], self.space.NbaseN, self.space.NbaseD, -B2_2_pts/det_dF, val, row, col)

                EF_31 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[2], self.space.Ntot_2form[0]))
                EF_31.eliminate_zeros()
                # ----------------------------------------------------


                # ------- 32 - block ([int, int, his] of DND) --------
                # evaluate equilibrium magnetic field (1-component) at interpolation and quadrature points
                B2_1_pts = self.equilibrium.b2_eq_1(self.eta_int[0], self.eta_int[1], self.eta_his[2].flatten())
                B2_1_pts = B2_1_pts.reshape(self.nint[0], self.nint[1], self.nhis[2], self.nq[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_int[0], self.eta_int[1], self.eta_his[2].flatten(), 'det_df'))
                det_dF = det_dF.reshape(self.nint[0], self.nint[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_D_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_0_D_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_0_D_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=int)

                ker.rhs13(self.dofs_0_D_i[0][0], self.dofs_0_N_i[1][0], self.dofs_1_D_i[2][0], self.dofs_0_D_i[0][1], self.dofs_0_N_i[1][1], self.dofs_1_D_i[2][1], self.subs[2], self.subs_cum[2], self.wts[2], self.basis_int_D[0], self.basis_int_N[1], self.basis_his_D[2], self.space.NbaseN, self.space.NbaseD,  B2_1_pts/det_dF, val, row, col)

                EF_32 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_1form[2], self.space.Ntot_2form[1]))
                EF_32.eliminate_zeros()
                # ----------------------------------------------------

        return EF_12, EF_13, EF_21, EF_23, EF_31, EF_32




    # =================================================================
    def get_blocks_FL(self, which, pol=True):
        """
        TODO
        """
        
        if self.basis_u == 2:
            assert which == 'm' or which == 'p'
        else:
            assert which == 'm' or which == 'p' or which == 'j'


        if self.basis_u == 0:

            if pol:

                # evaluation in third direction at eta_3 = 0
                eta3 = np.array([0.])

                # ------------- 11 - block ([int, his] of NN) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if   which == 'm':
                    EQ = self.equilibrium.r3_eq(self.eta_int[0], self.eta_his[1].flatten(), eta3)[:, :, 0]
                elif which == 'p':
                    EQ = self.equilibrium.p3_eq(self.eta_int[0], self.eta_his[1].flatten(), eta3)[:, :, 0]
                else:
                    EQ = self.equilibrium.domain.evaluate(self.eta_int[0], self.eta_his[1].flatten(), eta3, 'det_df')[:, :, 0]

                EQ = EQ.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size, dtype=int)

                ker.rhs12_2d(self.dofs_0_N_i[0][0], self.dofs_1_N_i[1][0], self.dofs_0_N_i[0][1], self.dofs_1_N_i[1][1], self.subs[1], self.subs_cum[1], self.wts[1], self.basis_int_N[0], self.basis_his_N[1], self.space.NbaseN, self.space.NbaseD, EQ, val, row, col)

                F_11 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[0]//self.D3, self.space.Ntot_0form//self.N3))
                F_11.eliminate_zeros()
                # ------------------------------------------------------------


                # ------------- 22 - block ([his, int] of NN) ----------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if   which == 'm':
                    EQ = self.equilibrium.r3_eq(self.eta_his[0].flatten(), self.eta_int[1], eta3)[:, :, 0]
                elif which == 'p':
                    EQ = self.equilibrium.p3_eq(self.eta_his[0].flatten(), self.eta_int[1], eta3)[:, :, 0]
                else:
                    EQ = self.equilibrium.domain.evaluate(self.eta_his[0].flatten(), self.eta_int[1], eta3, 'det_df')[:, :, 0]

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs11_2d(self.dofs_1_N_i[0][0], self.dofs_0_N_i[1][0], self.dofs_1_N_i[0][1], self.dofs_0_N_i[1][1], self.subs[0], self.subs_cum[0], self.wts[0], self.basis_his_N[0], self.basis_int_N[1], self.space.NbaseN, self.space.NbaseD, EQ, val, row, col)

                F_22 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[1]//self.D3, self.space.Ntot_0form//self.N3))
                F_22.eliminate_zeros()
                # ------------------------------------------------------------


                # ------------- 33 - block ([his, his] of NN) ----------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if   which == 'm':
                    EQ = self.equilibrium.r3_eq(self.eta_his[0].flatten(), self.eta_his[1].flatten(), eta3)[:, :, 0]
                elif which == 'p':
                    EQ = self.equilibrium.p3_eq(self.eta_his[0].flatten(), self.eta_his[1].flatten(), eta3)[:, :, 0]
                else:
                    EQ = self.equilibrium.domain.evaluate(self.eta_his[0].flatten(), self.eta_his[1].flatten(), eta3, 'det_df')[:, :, 0]

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_1_N_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_1_N_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_1_N_i[1][0].size, dtype=int)

                ker.rhs2_2d(self.dofs_1_N_i[0][0], self.dofs_1_N_i[1][0], self.dofs_1_N_i[0][1], self.dofs_1_N_i[1][1], self.subs[0], self.subs[1], self.subs_cum[0], self.subs_cum[1], self.wts[0], self.wts[1], self.basis_his_N[0], self.basis_his_N[1], self.space.NbaseN, self.space.NbaseD, EQ, val, row, col)

                F_33 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[2]//self.N3, self.space.Ntot_0form//self.N3))
                F_33.eliminate_zeros()
                # ------------------------------------------------------------

            else:

                # -------- 11 - block ([int, his, his] of NNN) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if   which == 'm':
                    EQ = self.equilibrium.r3_eq(self.eta_int[0], self.eta_his[1].flatten(), self.eta_his[2].flatten())
                elif which == 'p':
                    EQ = self.equilibrium.p3_eq(self.eta_int[0], self.eta_his[1].flatten(), self.eta_his[2].flatten())
                else:
                    EQ = self.equilibrium.domain.evaluate(self.eta_int[0], self.eta_his[1].flatten(), self.eta_his[2].flatten(), 'det_df')

                EQ = EQ.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size*self.dofs_1_N_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size*self.dofs_1_N_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_N_i[1][0].size*self.dofs_1_N_i[2][0].size, dtype=int)

                ker.rhs21(self.dofs_0_N_i[0][0], self.dofs_1_N_i[1][0], self.dofs_1_N_i[2][0], self.dofs_0_N_i[0][1], self.dofs_1_N_i[1][1], self.dofs_1_N_i[2][1], self.subs[1], self.subs[2], self.subs_cum[1], self.subs_cum[2], self.wts[1], self.wts[2], self.basis_int_N[0], self.basis_his_N[1], self.basis_his_N[2], self.space.NbaseN, self.space.NbaseD, EQ, val, row, col)

                F_11 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[0], self.space.Ntot_0form))
                F_11.eliminate_zeros()
                # ------------------------------------------------------------


                # -------- 22 - block ([his, int, his] of NNN) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if   which == 'm':
                    EQ = self.equilibrium.r3_eq(self.eta_his[0].flatten(), self.eta_int[1], self.eta_his[2].flatten())
                elif which == 'p':
                    EQ = self.equilibrium.p3_eq(self.eta_his[0].flatten(), self.eta_int[1], self.eta_his[2].flatten())
                else:
                    EQ = self.equilibrium.domain.evaluate(self.eta_his[0].flatten(), self.eta_int[1], self.eta_his[2].flatten(), 'det_df')

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_N_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_N_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_N_i[2][0].size, dtype=int)

                ker.rhs22(self.dofs_1_N_i[0][0], self.dofs_0_N_i[1][0], self.dofs_1_N_i[2][0], self.dofs_1_N_i[0][1], self.dofs_0_N_i[1][1], self.dofs_1_N_i[2][1], self.subs[0], self.subs[2], self.subs_cum[0], self.subs_cum[2], self.wts[0], self.wts[2], self.basis_his_N[0], self.basis_int_N[1], self.basis_his_N[2], self.space.NbaseN, self.space.NbaseD, EQ, val, row, col)

                F_22 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[1], self.space.Ntot_0form))
                F_22.eliminate_zeros()
                # ------------------------------------------------------------


                # -------- 33 - block ([his, his, int] of NNN) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if   which == 'm':
                    EQ = self.equilibrium.r3_eq(self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_int[2])
                elif which == 'p':
                    EQ = self.equilibrium.p3_eq(self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_int[2])
                else:
                    EQ = self.equilibrium.domain.evaluate(self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_int[2], 'det_df')

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1], self.nint[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_1_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_1_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_1_N_i[0][0].size*self.dofs_1_N_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)

                ker.rhs23(self.dofs_1_N_i[0][0], self.dofs_1_N_i[1][0], self.dofs_0_N_i[2][0], self.dofs_1_N_i[0][1], self.dofs_1_N_i[1][1], self.dofs_0_N_i[2][1], self.subs[0], self.subs[1], self.subs_cum[0], self.subs_cum[1], self.wts[0], self.wts[1], self.basis_his_N[0], self.basis_his_N[1], self.basis_int_N[2], self.space.NbaseN, self.space.NbaseD, EQ, val, row, col)

                F_33 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[2], self.space.Ntot_0form))
                F_33.eliminate_zeros()  
                # ------------------------------------------------------------

        elif self.basis_u == 1:
            print('1-form MHD is not yet implemented')

        elif self.basis_u == 2:

            if pol:

                # evaluation in third direction at eta_3 = 0
                eta3 = np.array([0.])

                # ------------- 11 - block ([int, his] of ND) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == 'm':
                    EQ = self.equilibrium.r3_eq(self.eta_int[0], self.eta_his[1].flatten(), eta3)[:, :, 0]
                else:
                    EQ = self.equilibrium.p3_eq(self.eta_int[0], self.eta_his[1].flatten(), eta3)[:, :, 0]

                EQ = EQ.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # evaluate Jacobian determinant at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_int[0], self.eta_his[1].flatten(), eta3, 'det_df'))[:, :, 0]
                det_dF = det_dF.reshape(self.nint[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=int)

                ker.rhs12_2d(self.dofs_0_N_i[0][0], self.dofs_1_D_i[1][0], self.dofs_0_N_i[0][1], self.dofs_1_D_i[1][1], self.subs[1], self.subs_cum[1], self.wts[1], self.basis_int_N[0], self.basis_his_D[1], self.space.NbaseN, self.space.NbaseD, EQ/det_dF, val, row, col)

                F_11 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[0]//self.D3, self.space.Ntot_2form[0]//self.D3))
                F_11.eliminate_zeros()
                # ------------------------------------------------------------


                # ------------- 22 - block ([his, int] of DN) ----------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == 'm':
                    EQ = self.equilibrium.r3_eq(self.eta_his[0].flatten(), self.eta_int[1], eta3)[:, :, 0]
                else:
                    EQ = self.equilibrium.p3_eq(self.eta_his[0].flatten(), self.eta_int[1], eta3)[:, :, 0]

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_his[0].flatten(), self.eta_int[1], eta3, 'det_df'))[:, :, 0]
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nint[1])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_N_i[1][0].size, dtype=int)

                ker.rhs11_2d(self.dofs_1_D_i[0][0], self.dofs_0_N_i[1][0], self.dofs_1_D_i[0][1], self.dofs_0_N_i[1][1], self.subs[0], self.subs_cum[0], self.wts[0], self.basis_his_D[0], self.basis_int_N[1], self.space.NbaseN, self.space.NbaseD, EQ/det_dF, val, row, col)

                F_22 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[1]//self.D3, self.space.Ntot_2form[1]//self.D3))
                F_22.eliminate_zeros()
                # ------------------------------------------------------------


                # ------------- 33 - block ([his, his] of DD) ----------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == 'm':
                    EQ = self.equilibrium.r3_eq(self.eta_his[0].flatten(), self.eta_his[1].flatten(), eta3)[:, :, 0]
                else:
                    EQ = self.equilibrium.p3_eq(self.eta_his[0].flatten(), self.eta_his[1].flatten(), eta3)[:, :, 0]

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_his[0].flatten(), self.eta_his[1].flatten(), eta3, 'det_df'))[:, :, 0]
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=float)
                row = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=int)
                col = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=int)

                ker.rhs2_2d(self.dofs_1_D_i[0][0], self.dofs_1_D_i[1][0], self.dofs_1_D_i[0][1], self.dofs_1_D_i[1][1], self.subs[0], self.subs[1], self.subs_cum[0], self.subs_cum[1], self.wts[0], self.wts[1], self.basis_his_D[0], self.basis_his_D[1], self.space.NbaseN, self.space.NbaseD, EQ/det_dF, val, row, col)

                F_33 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[2]//self.N3, self.space.Ntot_2form[2]//self.N3))
                F_33.eliminate_zeros()
                # ------------------------------------------------------------

            else:

                # -------- 11 - block ([int, his, his] of NDD) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == 'm':
                    EQ = self.equilibrium.r3_eq(self.eta_int[0], self.eta_his[1].flatten(), self.eta_his[2].flatten())
                else:
                    EQ = self.equilibrium.p3_eq(self.eta_int[0], self.eta_his[1].flatten(), self.eta_his[2].flatten())

                EQ = EQ.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nhis[2], self.nq[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_int[0], self.eta_his[1].flatten(), self.eta_his[2].flatten(), 'det_df'))
                det_dF = det_dF.reshape(self.nint[0], self.nhis[1], self.nq[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_0_N_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=int)

                ker.rhs21(self.dofs_0_N_i[0][0], self.dofs_1_D_i[1][0], self.dofs_1_D_i[2][0], self.dofs_0_N_i[0][1], self.dofs_1_D_i[1][1], self.dofs_1_D_i[2][1], self.subs[1], self.subs[2], self.subs_cum[1], self.subs_cum[2], self.wts[1], self.wts[2], self.basis_int_N[0], self.basis_his_D[1], self.basis_his_D[2], self.space.NbaseN, self.space.NbaseD, EQ/det_dF, val, row, col)

                F_11 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[0], self.space.Ntot_2form[0]))
                F_11.eliminate_zeros()
                # ------------------------------------------------------------


                # -------- 22 - block ([his, int, his] of DND) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == 'm':
                    EQ = self.equilibrium.r3_eq(self.eta_his[0].flatten(), self.eta_int[1], self.eta_his[2].flatten())
                else:
                    EQ = self.equilibrium.p3_eq(self.eta_his[0].flatten(), self.eta_int[1], self.eta_his[2].flatten())

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nhis[2], self.nq[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_his[0].flatten(), self.eta_int[1], self.eta_his[2].flatten(), 'det_df'))
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nint[1], self.nhis[2], self.nq[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_0_N_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=int)

                ker.rhs22(self.dofs_1_D_i[0][0], self.dofs_0_N_i[1][0], self.dofs_1_D_i[2][0], self.dofs_1_D_i[0][1], self.dofs_0_N_i[1][1], self.dofs_1_D_i[2][1], self.subs[0], self.subs[2], self.subs_cum[0], self.subs_cum[2], self.wts[0], self.wts[2], self.basis_his_D[0], self.basis_int_N[1], self.basis_his_D[2], self.space.NbaseN, self.space.NbaseD, EQ/det_dF, val, row, col)

                F_22 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[1], self.space.Ntot_2form[1]))
                F_22.eliminate_zeros()
                # ------------------------------------------------------------


                # -------- 33 - block ([his, his, int] of DDN) ---------------
                # evaluate equilibrium density/pressure at interpolation and quadrature points
                if which == 'm':
                    EQ = self.equilibrium.r3_eq(self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_int[2])
                else:
                    EQ = self.equilibrium.p3_eq(self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_int[2])

                EQ = EQ.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1], self.nint[2])

                # evaluate Jacobian determinant at at interpolation and quadrature points
                det_dF = abs(self.equilibrium.domain.evaluate(self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_int[2], 'det_df'))
                det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1], self.nint[2])

                # assemble sparse matrix
                val = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=float)
                row = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)
                col = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_0_N_i[2][0].size, dtype=int)

                ker.rhs23(self.dofs_1_D_i[0][0], self.dofs_1_D_i[1][0], self.dofs_0_N_i[2][0], self.dofs_1_D_i[0][1], self.dofs_1_D_i[1][1], self.dofs_0_N_i[2][1], self.subs[0], self.subs[1], self.subs_cum[0], self.subs_cum[1], self.wts[0], self.wts[1], self.basis_his_D[0], self.basis_his_D[1], self.basis_int_N[2], self.space.NbaseN, self.space.NbaseD, EQ/det_dF, val, row, col)

                F_33 = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_2form[2], self.space.Ntot_2form[2]))
                F_33.eliminate_zeros()
                # ------------------------------------------------------------

        return F_11, F_22, F_33



    # =================================================================
    def get_blocks_PR(self, pol=True):
        """
        TODO
        """

        if pol:

            # evalutation in third direction at eta_3 = 0
            eta3 = np.array([0.])

            # ------------ ([his, his] of DD) --------------------
            # evaluate equilibrium pressure at quadrature points
            P3_pts = self.equilibrium.p3_eq(self.eta_his[0].flatten(), self.eta_his[1].flatten(), eta3)[:, :, 0]
            P3_pts = P3_pts.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1])

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(self.equilibrium.domain.evaluate(self.eta_his[0].flatten(), self.eta_his[1].flatten(), eta3, 'det_df'))[:, :, 0]
            det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1])

            # assemble sparse matrix
            val = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=float)
            row = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=int)
            col = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_1_D_i[1][0].size, dtype=int)

            ker.rhs2_2d(self.dofs_1_D_i[0][0], self.dofs_1_D_i[1][0], self.dofs_1_D_i[0][1], self.dofs_1_D_i[1][1], self.subs[0], self.subs[1], self.subs_cum[0], self.subs_cum[1], self.wts[0], self.wts[1], self.basis_his_D[0], self.basis_his_D[1], self.space.NbaseN, self.space.NbaseD, P3_pts/det_dF, val, row, col)

            PR = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_3form//self.D3, self.space.Ntot_3form//self.D3))
            PR.eliminate_zeros()
            # -----------------------------------------------------

        else:

            # --------------- ([his, his, his] of DDD) ------------
            # evaluate equilibrium pressure at quadrature points
            P3_pts = self.equilibrium.p3_eq(self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_his[2].flatten())

            P3_pts = P3_pts.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1], self.nhis[2], self.nq[2])

            # evaluate Jacobian determinant at at interpolation and quadrature points
            det_dF = abs(self.equilibrium.domain.evaluate(self.eta_his[0].flatten(), self.eta_his[1].flatten(), self.eta_his[2].flatten(), 'det_df'))
            det_dF = det_dF.reshape(self.nhis[0], self.nq[0], self.nhis[1], self.nq[1], self.nhis[2], self.nq[2])

            # assemble sparse matrix
            val = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=float)
            row = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=int)
            col = np.empty(self.dofs_1_D_i[0][0].size*self.dofs_1_D_i[1][0].size*self.dofs_1_D_i[2][0].size, dtype=int)

            ker.rhs3(self.dofs_1_D_i[0][0], self.dofs_1_D_i[1][0], self.dofs_1_D_i[2][0], self.dofs_1_D_i[0][1], self.dofs_1_D_i[1][1], self.dofs_1_D_i[2][1], self.subs[0], self.subs[1], self.subs[2], self.subs_cum[0], self.subs_cum[1], self.subs_cum[2], self.wts[0], self.wts[1], self.wts[2], self.basis_his_D[0], self.basis_his_D[1], self.basis_his_D[2], self.space.NbaseN, self.space.NbaseD, P3_pts/det_dF, val, row, col)

            PR = spa.csr_matrix((val, (row, col)), shape=(self.space.Ntot_3form, self.space.Ntot_3form))
            PR.eliminate_zeros()
            # ----------------------------------------------------

        return PR
    
    
    # ====================================================================
    def get_blocks_MR(self, pol=True):
        """
        TODO
        """
        
        weight11 = lambda eta1, eta2, eta3 : self.equilibrium.r0_eq(eta1, eta2, eta3)*self.equilibrium.domain.evaluate(eta1, eta2, eta3, 'g_11')
        weight12 = lambda eta1, eta2, eta3 : self.equilibrium.r0_eq(eta1, eta2, eta3)*self.equilibrium.domain.evaluate(eta1, eta2, eta3, 'g_12')
        weight13 = lambda eta1, eta2, eta3 : self.equilibrium.r0_eq(eta1, eta2, eta3)*self.equilibrium.domain.evaluate(eta1, eta2, eta3, 'g_13')

        weight21 = lambda eta1, eta2, eta3 : self.equilibrium.r0_eq(eta1, eta2, eta3)*self.equilibrium.domain.evaluate(eta1, eta2, eta3, 'g_21')
        weight22 = lambda eta1, eta2, eta3 : self.equilibrium.r0_eq(eta1, eta2, eta3)*self.equilibrium.domain.evaluate(eta1, eta2, eta3, 'g_22')
        weight23 = lambda eta1, eta2, eta3 : self.equilibrium.r0_eq(eta1, eta2, eta3)*self.equilibrium.domain.evaluate(eta1, eta2, eta3, 'g_23')

        weight31 = lambda eta1, eta2, eta3 : self.equilibrium.r0_eq(eta1, eta2, eta3)*self.equilibrium.domain.evaluate(eta1, eta2, eta3, 'g_31')
        weight32 = lambda eta1, eta2, eta3 : self.equilibrium.r0_eq(eta1, eta2, eta3)*self.equilibrium.domain.evaluate(eta1, eta2, eta3, 'g_32')
        weight33 = lambda eta1, eta2, eta3 : self.equilibrium.r0_eq(eta1, eta2, eta3)*self.equilibrium.domain.evaluate(eta1, eta2, eta3, 'g_33')
        
        self.weights_MR = [[weight11, weight12, weight13], 
                           [weight21, weight22, weight23], 
                           [weight31, weight32, weight33]]

        
        # ----------- 0-form ----------------------
        if self.basis_u == 0:
            if pol:
                MR = mass_2d.get_Mv(self.space, self.equilibrium.domain, self.weights_MR)
            else:
                MR = mass_3d.get_Mv(self.space, self.equilibrium.domain, self.weights_MR)
        # -----------------------------------------
        
        
        # ----------- 1-form ----------------------
        elif self.basis_u == 1:
            print('1-form MHD is not yet implemented')
        # -----------------------------------------
        
        
        # ----------- 2-form ----------------------
        elif self.basis_u == 2:
            if pol:
                MR = mass_2d.get_M2(self.space, self.equilibrium.domain, self.weights_MR)
            else:
                MR = mass_3d.get_M2(self.space, self.equilibrium.domain, self.weights_MR)
        # -----------------------------------------
        
        return MR
                
    
    
    # =================================================================
    def get_blocks_MJ(self, pol=True):
        """
        TODO
        """
            
        weight11 = lambda eta1, eta2, eta3:  np.zeros((eta1.size, eta2.size, eta3.size), dtype=float)
        weight12 = lambda eta1, eta2, eta3: -self.equilibrium.j2_eq_3(eta1, eta2, eta3)
        weight13 = lambda eta1, eta2, eta3:  self.equilibrium.j2_eq_2(eta1, eta2, eta3)

        weight21 = lambda eta1, eta2, eta3:  self.equilibrium.j2_eq_3(eta1, eta2, eta3)
        weight22 = lambda eta1, eta2, eta3:  np.zeros((eta1.size, eta2.size, eta3.size), dtype=float)
        weight23 = lambda eta1, eta2, eta3: -self.equilibrium.j2_eq_1(eta1, eta2, eta3)

        weight31 = lambda eta1, eta2, eta3: -self.equilibrium.j2_eq_2(eta1, eta2, eta3)
        weight32 = lambda eta1, eta2, eta3:  self.equilibrium.j2_eq_1(eta1, eta2, eta3)
        weight33 = lambda eta1, eta2, eta3:  np.zeros((eta1.size, eta2.size, eta3.size), dtype=float)

        self.weights_MJ = [[weight11, weight12, weight13], 
                           [weight21, weight22, weight23], 
                           [weight31, weight32, weight33]]
        
        
        # ----------- 0-form ----------------------
        if self.basis_u == 0:
            if pol:
                MJ = mass_2d.get_Mv(self.space, self.equilibrium.domain, self.weights_MJ)
            else:
                MJ = mass_3d.get_Mv(self.space, self.equilibrium.domain, self.weights_MJ)
        # -----------------------------------------
        
        
        # ----------- 1-form ----------------------
        elif self.basis_u == 1:
            print('1-form MHD is not yet implemented')
        # -----------------------------------------
        
        
        # ----------- 2-form ----------------------
        elif self.basis_u == 2:
            if pol:
                MJ = mass_2d.get_M2(self.space, self.equilibrium.domain, self.weights_MJ)
            else:
                MJ = mass_3d.get_M2(self.space, self.equilibrium.domain, self.weights_MJ)
        # -----------------------------------------
        
        return MJ