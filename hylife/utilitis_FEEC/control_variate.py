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
    Contains method for computing the terms (B x jh_eq) and (rhoh_eq * (U x B)).
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        tensor product B-spline space
        
    pic_accumulation : accumulation
        object created from class "accumulation" from hylife/utilitis_PIC/accumulation.py
        
    kind_map : int
        type of mapping
        
    params_map : list of doubles
        parameters for the mapping
    """
    
    def __init__(self, tensor_space, pic_accumulation, kind_map, params_map):
        
        self.kind_map   = kind_map         # type of mapping
        self.params_map = params_map       # parameters for mapping
        
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
        
        # particle accumulator
        self.pic    = pic_accumulation
        
        # evaluation of DF^T * jh_eq_phys at quadrature points
        self.mat_jh1 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.mat_jh2 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.mat_jh3 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        
        ker_cv.kernel_evaluation(self.Nel, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.mat_jh1, 1, kind_map, params_map)
        ker_cv.kernel_evaluation(self.Nel, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.mat_jh2, 2, kind_map, params_map)
        ker_cv.kernel_evaluation(self.Nel, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.mat_jh3, 3, kind_map, params_map)
        
        # evaluation of nh_eq_phys at quadrature points
        self.mat_nh = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        
        ker_cv.kernel_evaluation(self.Nel, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.mat_nh, 4, kind_map, params_map)
        
        # evaluation of G / sqrt(g) at quadrature points
        self.mat_g11 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.mat_g21 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.mat_g22 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.mat_g31 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.mat_g32 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.mat_g33 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        
        
        ker_cv.kernel_evaluation(self.Nel, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.mat_g11, 21, kind_map, params_map)
        ker_cv.kernel_evaluation(self.Nel, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.mat_g21, 22, kind_map, params_map)
        ker_cv.kernel_evaluation(self.Nel, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.mat_g22, 23, kind_map, params_map)
        ker_cv.kernel_evaluation(self.Nel, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.mat_g31, 24, kind_map, params_map)
        ker_cv.kernel_evaluation(self.Nel, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.mat_g32, 25, kind_map, params_map)
        ker_cv.kernel_evaluation(self.Nel, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.mat_g33, 26, kind_map, params_map)
        
        
        # total magnetic field at quadrature points (perturbed + equilibrium)
        self.B1  = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.B2  = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.B3  = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        
        # correction vectors in step 3
        self.F1  = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
        self.F2  = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]), dtype=float)
        self.F3  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]), dtype=float)
        
        # correction matrices in step 1
        self.M12 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
        self.M13 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
        self.M23 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
        
        
    
    # ===== inner product in V1 (3d) of (B x jh_eq) - term =======
    def inner_prod_V1_jh_eq(self, b1, b2, b3):
        """
        Computes the inner product of the term ((G * b) x (DF^T * jh_eq_phys)/g_sqrt) with each basis function in V1.
        
        Parameters
        ----------
        b1 : list of array_like
            the FEM coefficients of the perturbed B-field (1 - component)
            
        b2 : list of array_like
            the FEM coefficients of the perturbed B-field (2 - component)
            
        b3 : list of array_like
            the FEM coefficients of the perturbed B-field (3 - component)
            
        Returns
        -------
        F : array_like
            inner products with each basis function in V1
        """
        
        # evaluation of total magnetic field at quadrature points (perturbed + equilibrium)
        ker_cv.kernel_evaluate_2form(self.Nel, self.p, [0, 1, 1], self.n_quad, self.pts[0], self.pts[1], self.pts[2], b1, [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], self.basisN[0], self.basisD[1], self.basisD[2], self.B1, 11, self.kind_map, self.params_map)
        ker_cv.kernel_evaluate_2form(self.Nel, self.p, [1, 0, 1], self.n_quad, self.pts[0], self.pts[1], self.pts[2], b2, [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], self.basisD[0], self.basisN[1], self.basisD[2], self.B2, 12, self.kind_map, self.params_map)
        ker_cv.kernel_evaluate_2form(self.Nel, self.p, [1, 1, 0], self.n_quad, self.pts[0], self.pts[1], self.pts[2], b3, [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]], self.basisD[0], self.basisD[1], self.basisN[2], self.B3, 13, self.kind_map, self.params_map)
        
        # assembly of F (1 - component)
        ker_cv.kernel_inner(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 1, 0, 0, self.wts[0], self.wts[1], self.wts[2], self.basisD[0], self.basisN[1], self.basisN[2], self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.F1, self.mat_jh3*(self.mat_g21*self.B1 + self.mat_g22*self.B2 + self.mat_g32*self.B3) - self.mat_jh2*(self.mat_g31*self.B1 + self.mat_g32*self.B2 + self.mat_g33*self.B3))
        
        # assembly of F (2 - component)
        ker_cv.kernel_inner(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 1, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisD[1], self.basisN[2], self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.F2, self.mat_jh1*(self.mat_g31*self.B1 + self.mat_g32*self.B2 + self.mat_g33*self.B3) - self.mat_jh3*(self.mat_g11*self.B1 + self.mat_g21*self.B2 + self.mat_g31*self.B3))
        
        # assembly of F (3 - component)
        ker_cv.kernel_inner(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 1, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisN[1], self.basisD[2], self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], self.F3, self.mat_jh2*(self.mat_g11*self.B1 + self.mat_g21*self.B2 + self.mat_g31*self.B3) - self.mat_jh1*(self.mat_g21*self.B1 + self.mat_g22*self.B2 + self.mat_g32*self.B3))
        
        # convert to 1d array and return
        return np.concatenate((self.F1.flatten(), self.F2.flatten(), self.F3.flatten()))
    
    
   
    # ===== mass matrix in V1 (3d) of (rhoh_eq * (U x B)) - term =======
    def mass_V1_nh_eq(self, b1, b2, b3):
        """
        Computes the mass matrix in V1 weighted with the term (rhoh_eq_phys * (G * B)/g_sqrt).
        
        Parameters
        ----------
        b1 : list of array_like
            the FEM coefficients of the perturbed B-field (1 - component)
            
        b2 : list of array_like
            the FEM coefficients of the perturbed B-field (2 - component)
            
        b3 : list of array_like
            the FEM coefficients of the perturbed B-field (3 - component)
            
        Returns
        -------
        M : sparse matrix in csc-format
            weighted mass matrix in V1
        """
        
        
        # evaluation of total magnetic field at quadrature points (perturbed + equilibrium)   
        ker_cv.kernel_evaluate_2form(self.Nel, self.p, [0, 1, 1], self.n_quad, self.pts[0], self.pts[1], self.pts[2], b1, [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], self.basisN[0], self.basisD[1], self.basisD[2], self.B1, 11, self.kind_map, self.params_map)
        ker_cv.kernel_evaluate_2form(self.Nel, self.p, [1, 0, 1], self.n_quad, self.pts[0], self.pts[1], self.pts[2], b2, [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], self.basisD[0], self.basisN[1], self.basisD[2], self.B2, 12, self.kind_map, self.params_map)
        ker_cv.kernel_evaluate_2form(self.Nel, self.p, [1, 1, 0], self.n_quad, self.pts[0], self.pts[1], self.pts[2], b3, [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]], self.basisD[0], self.basisD[1], self.basisN[2], self.B3, 13, self.kind_map, self.params_map)
        
        
        # assembly of M12
        ker.kernel_mass(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 1, 0, 0, 0, 1, 0, self.wts[0], self.wts[1], self.wts[2], self.basisD[0], self.basisN[1], self.basisN[2], self.basisN[0], self.basisD[1], self.basisN[2], self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.M12, -self.mat_nh*(self.mat_g31*self.B1 + self.mat_g32*self.B2 + self.mat_g33*self.B3))
        
        # assembly of M13
        ker.kernel_mass(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 1, 0, 0, 0, 0, 1, self.wts[0], self.wts[1], self.wts[2], self.basisD[0], self.basisN[1], self.basisN[2], self.basisN[0], self.basisN[1], self.basisD[2], self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], self.M13,  self.mat_nh*(self.mat_g21*self.B1 + self.mat_g22*self.B2 + self.mat_g32*self.B3))
        
        # assembly of M23
        ker.kernel_mass(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 1, 0, 0, 0, 1, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisD[1], self.basisN[2], self.basisN[0], self.basisN[1], self.basisD[2], self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], self.M23, -self.mat_nh*(self.mat_g11*self.B1 + self.mat_g21*self.B2 + self.mat_g31*self.B3))
        
        # conversion to sparse matrix and return
        return self.pic.to_sparse_step1(self.M12, self.M13, self.M23)