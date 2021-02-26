# coding: utf-8
#
# Copyright 2020 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Class for control variates in delta-f method for current coupling scheme.
"""


import numpy         as np
import scipy.sparse  as spa

import hylife.utilitis_FEEC.basics.kernels_3d as ker

import source_run.kernels_control_variate     as ker_cv



class terms_control_variate:
    """
    Contains method for computing the terms (B x jh_eq) and (rhoh_eq * (B x U)).
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        3D tensor product B-spline space
        
    pic_accumulation : accumulation
        object created from class "accumulation" from hylife/utilitis_PIC/accumulation.py
        
    kind_map : int
        type of mapping
        
    params_map : list of doubles
        parameters for the mapping
    """
    
    def __init__(self, tensor_space, pic_accumulator, kind_map, params_map=None, tensor_space_F=None, cx=None, cy=None, cz=None):
        
        self.p      = tensor_space.p       # spline degrees
        self.Nel    = tensor_space.Nel     # number of elements
        self.NbaseN = tensor_space.NbaseN  # total number of basis functions (N)
        self.NbaseD = tensor_space.NbaseD  # total number of basis functions (D)
        
        self.n_quad = tensor_space.n_quad  # number of quadrature points per element
        self.wts    = tensor_space.wts     # quadrature weights in format (element, local point)
        self.pts    = tensor_space.pts     # quadrature points in format (element, local point)
        self.n_pts  = tensor_space.n_pts   # total number of quadrature points
        
        # basis functions evaluated at quadrature points in format (element, local basis function, derivative, local point)
        self.basisN = tensor_space.basisN
        self.basisD = tensor_space.basisD
        
        # particle accumulator and representation of bulk velocity
        self.pic    = pic_accumulator
        
        if   self.pic.basis_u == 0:
            kind_fun_eq = [ 1,  2,  3,  4]
            
        elif self.pic.basis_u == 2:
            kind_fun_eq = [11, 12, 13, 14]
        
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
        
        
        # ========= evaluation of DF^(-1) * jh_eq_phys * |det(DF)| at quadrature points =========
        self.mat_jh1 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.mat_jh2 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.mat_jh3 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)

        ker_cv.kernel_evaluation_quad(self.Nel, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.mat_jh1, kind_fun_eq[0], kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)
        ker_cv.kernel_evaluation_quad(self.Nel, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.mat_jh2, kind_fun_eq[1], kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)
        ker_cv.kernel_evaluation_quad(self.Nel, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.mat_jh3, kind_fun_eq[2], kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)
            
        
        
        # ========= evaluation of nh_eq_phys * |det(DF)| at quadrature points ===================
        self.mat_nh = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        
        ker_cv.kernel_evaluation_quad(self.Nel, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.mat_nh, kind_fun_eq[3], kind_map, params_map, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)
        
        
        # =========== 2-form magnetic field at quadrature points =================================
        self.B2_1 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.B2_2 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.B2_3 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        
        
        # ================== correction matrices in step 1 ========================
        if self.pic.basis_u == 0:
            self.M12 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.M13 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.M23 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            
        elif self.pic.basis_u == 2:
            self.M12 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.M13 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.M23 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            
        
        
        # ==================== correction vectors in step 3 =======================
        if self.pic.basis_u == 0:
            self.F1 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
            self.F2 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
            self.F3 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
            
        elif self.pic.basis_u == 2:
            self.F1 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]), dtype=float)
            self.F2 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]), dtype=float)
            self.F3 = np.empty((self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]), dtype=float)
        
            
        
    
    # ===== inner product in V0^3 of (B x jh_eq) - term ==========
    def inner_prod_jh_eq(self, b1, b2, b3):
        """
        Computes the inner product of the term (B x (DF^(-1) * jh_eq_phys) * |det(DF)|) with each basis function in V0^3.
        
        Parameters
        ----------
        b1 : array_like
            the B-field FEM coefficients (1-component)
            
        b2 : array_like
            the B-field FEM coefficients (2-component)
            
        b3 : array_like
            the B-field FEM coefficients (3-component)
            
        Returns
        -------
        F : array_like
            inner products with each basis function in V0^3
        """
        
        
        # evaluation of magnetic field at quadrature points
        ker.kernel_evaluate_2form(self.Nel, self.p, [0, 1, 1], self.n_quad, b1, [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], self.basisN[0], self.basisD[1], self.basisD[2], self.B2_1)
        ker.kernel_evaluate_2form(self.Nel, self.p, [1, 0, 1], self.n_quad, b2, [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], self.basisD[0], self.basisN[1], self.basisD[2], self.B2_2)
        ker.kernel_evaluate_2form(self.Nel, self.p, [1, 1, 0], self.n_quad, b3, [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]], self.basisD[0], self.basisD[1], self.basisN[2], self.B2_3)
 
        
        if self.pic.basis_u == 0:
            # assembly of F (1-component)
            ker.kernel_inner_2(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisN[1], self.basisN[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.F1, self.B2_2*self.mat_jh3 - self.B2_3*self.mat_jh2)

            # assembly of F (2-component)
            ker.kernel_inner_2(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisN[1], self.basisN[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.F2, self.B2_3*self.mat_jh1 - self.B2_1*self.mat_jh3)

            # assembly of F (3-component)
            ker.kernel_inner_2(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisN[1], self.basisN[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.F3, self.B2_1*self.mat_jh2 - self.B2_2*self.mat_jh1)
            
        elif self.pic.basis_u == 2:
            # assembly of F (1-component)
            ker.kernel_inner_2(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 1, 1, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisD[1], self.basisD[2], self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], self.F1, self.B2_2*self.mat_jh3 - self.B2_3*self.mat_jh2)

            # assembly of F (2-component)
            ker.kernel_inner_2(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 1, 0, 1, self.wts[0], self.wts[1], self.wts[2], self.basisD[0], self.basisN[1], self.basisD[2], self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], self.F2, self.B2_3*self.mat_jh1 - self.B2_1*self.mat_jh3)

            # assembly of F (3-component)
            ker.kernel_inner_2(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 1, 1, 0, self.wts[0], self.wts[1], self.wts[2], self.basisD[0], self.basisD[1], self.basisN[2], self.NbaseD[0], self.NbaseD[1], self.NbaseN[2], self.F3, self.B2_1*self.mat_jh2 - self.B2_2*self.mat_jh1)
        
        return np.concatenate((self.F1.flatten(), self.F2.flatten(), self.F3.flatten()))
    
    
   
    # ===== mass matrix in V0^3 (3d) of (rhoh_eq * (B x U)) - term =======
    def mass_nh_eq(self, b1, b2, b3):
        """
        Computes the mass matrix in V0^3 weighted with the term (rhoh_eq_phys * B * |det(DF)|).
        
        Parameters
        ----------
        b1 : array_like
            the B-field FEM coefficients (1-component)
            
        b2 : array_like
            the B-field FEM coefficients (2-component)
            
        b3 : array_like
            the B-field FEM coefficients (3-component)
            
        Returns
        -------
        M : sparse matrix in csr-format
            weighted mass matrix in V0^3
        """
        
        
        # evaluation of magnetic field at quadrature points
        ker.kernel_evaluate_2form(self.Nel, self.p, [0, 1, 1], self.n_quad, b1, [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], self.basisN[0], self.basisD[1], self.basisD[2], self.B2_1)
        ker.kernel_evaluate_2form(self.Nel, self.p, [1, 0, 1], self.n_quad, b2, [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], self.basisD[0], self.basisN[1], self.basisD[2], self.B2_2)
        ker.kernel_evaluate_2form(self.Nel, self.p, [1, 1, 0], self.n_quad, b3, [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]], self.basisD[0], self.basisD[1], self.basisN[2], self.B2_3)
        
        
        if self.pic.basis_u == 0:
            # assembly of M12
            ker.kernel_mass(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 0, 0, 0, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisN[1], self.basisN[2], self.basisN[0], self.basisN[1], self.basisN[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.M12, -self.mat_nh*self.B2_3)

            # assembly of M13
            ker.kernel_mass(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 0, 0, 0, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisN[1], self.basisN[2], self.basisN[0], self.basisN[1], self.basisN[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.M13, +self.mat_nh*self.B2_2)

            # assembly of M23
            ker.kernel_mass(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 0, 0, 0, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisN[1], self.basisN[2], self.basisN[0], self.basisN[1], self.basisN[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.M23, -self.mat_nh*self.B2_1)
            
        elif self.pic.basis_u == 2:
            # assembly of M12
            ker.kernel_mass(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 1, 1, 1, 0, 1, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisD[1], self.basisD[2], self.basisD[0], self.basisN[1], self.basisD[2], self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], self.M12, -self.mat_nh*self.B2_3)

            # assembly of M13
            ker.kernel_mass(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 1, 1, 1, 1, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisD[1], self.basisD[2], self.basisD[0], self.basisD[1], self.basisN[2], self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], self.M13, +self.mat_nh*self.B2_2)

            # assembly of M23
            ker.kernel_mass(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 1, 0, 1, 1, 1, 0, self.wts[0], self.wts[1], self.wts[2], self.basisD[0], self.basisN[1], self.basisD[2], self.basisD[0], self.basisD[1], self.basisN[2], self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], self.M23, -self.mat_nh*self.B2_1)
        
        # conversion to sparse matrix and return
        return self.pic.to_sparse_step1(self.M12, self.M13, self.M23)