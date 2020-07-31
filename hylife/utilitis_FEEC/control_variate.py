# coding: utf-8
#
# Copyright 2020 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Class for control variates in delta-f method for current coupling scheme.
"""


import numpy         as np
import scipy.sparse  as spa

import hylife.utilitis_FEEC.basics.kernels_3d       as ker
import hylife.utilitis_FEEC.kernels_control_variate as ker_cv


class terms_control_variate:
    """
    Contains method for computing the terms (B x jh_eq) and (rhoh_eq * (U x B)).
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        tensor product B-spline space
        
    kind_map : int
        type of mapping
        
    params_map : list of doubles
        parameters for the mapping
    """
    
    def __init__(self, tensor_space, kind_map, params_map):
        
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
    
        
    # ===== inner product in V1 (3d) of (B x jh_eq) - term =======
    def inner_prod_V1_jh_eq(self, b_coeff):
        """
        Computes the inner product of the term ((G * b) x (DF^T * jh_eq_phys)/g_sqrt) with each basis function in V1.
        
        Parameters
        ----------
        b_coeff : list of array_like
            the FEM coefficients of the perturbed B-field
            
        Returns
        -------
        F1, F2, F3 : array_like
            inner products with each basis function in V1
        """
        
        # evaluation of total magnetic field at quadrature points (perturbed + equilibrium)
        B1 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        B2 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        B3 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        
        ker_cv.kernel_evaluate_2form(self.Nel, self.p, [0, 1, 1], self.n_quad, self.pts[0], self.pts[1], self.pts[2], b_coeff[0], [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], self.basisN[0], self.basisD[1], self.basisD[2], B1, 11, self.kind_map, self.params_map)
        ker_cv.kernel_evaluate_2form(self.Nel, self.p, [1, 0, 1], self.n_quad, self.pts[0], self.pts[1], self.pts[2], b_coeff[1], [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], self.basisD[0], self.basisN[1], self.basisD[2], B2, 12, self.kind_map, self.params_map)
        ker_cv.kernel_evaluate_2form(self.Nel, self.p, [1, 1, 0], self.n_quad, self.pts[0], self.pts[1], self.pts[2], b_coeff[2], [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]], self.basisD[0], self.basisD[1], self.basisN[2], B3, 13, self.kind_map, self.params_map)
        
        
        # computation of 1 - component
        F1 = np.zeros((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]))
        
        ker_cv.kernel_inner(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 1, 0, 0, self.wts[0], self.wts[1], self.wts[2], self.basisD[0], self.basisN[1], self.basisN[2], self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], F1, self.mat_jh3*(self.mat_g21*B1 + self.mat_g22*B2 + self.mat_g32*B3) - self.mat_jh2*(self.mat_g31*B1 + self.mat_g32*B2 + self.mat_g33*B3))
        
        # computation of 2 - component
        F2 = np.zeros((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]))
        
        ker_cv.kernel_inner(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 1, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisD[1], self.basisN[2], self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], F2, self.mat_jh1*(self.mat_g31*B1 + self.mat_g32*B2 + self.mat_g33*B3) - self.mat_jh3*(self.mat_g11*B1 + self.mat_g21*B2 + self.mat_g31*B3))
        
        # computation of 3 - component
        F3 = np.zeros((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]))
        
        ker_cv.kernel_inner(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 1, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisN[1], self.basisD[2], self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], F3, self.mat_jh2*(self.mat_g11*B1 + self.mat_g21*B2 + self.mat_g31*B3) - self.mat_jh1*(self.mat_g21*B1 + self.mat_g22*B2 + self.mat_g32*B3))
        
        
        return F1, F2, F3
    
    
    # ===== mass matrix in V1 (3d) of (rhoh_eq * (U x B)) - term =======
    def mass_V1_nh_eq(self, b_coeff):
        """
        Computes the mass matrix in V1 weighted with the term (rhoh_eq_phys * (G * B)/g_sqrt).
        
        Parameters
        ----------
        b_coeff : list of array_like
            the FEM coefficients of the perturbed B-field
            
        Returns
        -------
        M : sparse matrix in csc-format
            weighted mass matrix in V1
        """
        
        M21 = np.zeros((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        M31 = np.zeros((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        M32 = np.zeros((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        
        # evaluation of total magnetic field at quadrature points (perturbed + equilibrium)
        B1 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        B2 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        B3 = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        
        ker_cv.kernel_evaluate_2form(self.Nel, self.p, [0, 1, 1], self.n_quad, self.pts[0], self.pts[1], self.pts[2], b_coeff[0], [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], self.basisN[0], self.basisD[1], self.basisD[2], B1, 11, self.kind_map, self.params_map)
        ker_cv.kernel_evaluate_2form(self.Nel, self.p, [1, 0, 1], self.n_quad, self.pts[0], self.pts[1], self.pts[2], b_coeff[1], [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], self.basisD[0], self.basisN[1], self.basisD[2], B2, 12, self.kind_map, self.params_map)
        ker_cv.kernel_evaluate_2form(self.Nel, self.p, [1, 1, 0], self.n_quad, self.pts[0], self.pts[1], self.pts[2], b_coeff[2], [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]], self.basisD[0], self.basisD[1], self.basisN[2], B3, 13, self.kind_map, self.params_map)
        
        
        # assembly
        ker.kernel_mass(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 1, 0, 1, 0, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisD[1], self.basisN[2], self.basisD[0], self.basisN[1], self.basisN[2], self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], M21, self.mat_nh*(self.mat_g31*B1 + self.mat_g32*B2 + self.mat_g33*B3))
        
        ker.kernel_mass(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 1, 1, 0, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisN[1], self.basisD[2], self.basisD[0], self.basisN[1], self.basisN[2], self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], M31, -self.mat_nh*(self.mat_g21*B1 + self.mat_g22*B2 + self.mat_g32*B3))
        
        ker.kernel_mass(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 1, 0, 1, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisN[1], self.basisD[2], self.basisN[0], self.basisD[1], self.basisN[2], self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], M32, self.mat_nh*(self.mat_g11*B1 + self.mat_g21*B2 + self.mat_g31*B3))
        
        # conversion to sparse matrices
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseN[0]) - self.p[0]
        shift2  = np.arange(self.NbaseD[1]) - self.p[1]
        shift3  = np.arange(self.NbaseN[2]) - self.p[2]
        
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseD[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

        col     = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        M21     = spa.csc_matrix((M21.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2]))
        M21.eliminate_zeros()
        
        indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseN[0]) - self.p[0]
        shift2  = np.arange(self.NbaseN[1]) - self.p[1]
        shift3  = np.arange(self.NbaseD[2]) - self.p[2]
        
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseD[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

        col     = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        M31     = spa.csc_matrix((M31.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2]))
        M31.eliminate_zeros()
        
        indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseN[0]) - self.p[0]
        shift2  = np.arange(self.NbaseN[1]) - self.p[1]
        shift3  = np.arange(self.NbaseD[2]) - self.p[2]
        
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseD[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

        col     = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        M32     = spa.csc_matrix((M32.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]))
        M32.eliminate_zeros()
        
        M = spa.bmat([[None, -M21.T, -M31.T], [M21, None, -M32.T], [M31, M32, None]], format='csc')
        
        return M