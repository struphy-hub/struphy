# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic modules to create tensor-product finite element spaces of univariate B-splines.
"""


import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.bsplines         as bsp
import hylife.utilitis_FEEC.bsplines_kernels as bsp_kernels

import hylife.utilitis_FEEC.basics.spline_evaluation_1d as eva_1d
import hylife.utilitis_FEEC.basics.spline_evaluation_2d as eva_2d
import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva_3d



# =============== 1d B-spline space ======================
class spline_space_1d:
    """
    Defines a 1d space of B-splines.
    
    Parameters
    ----------
    T : array_like
        spline knot vector
        
    p : int
        spline degree
        
    bc : boolean
        boundary conditions (True = periodic, False = clamped)
        
    n_quad : int
        optional: number of Gauss-Legendre quadrature points per element for integrations
    """
    
    def __init__(self, T, p, bc, n_quad=None):
        
        self.T       = T                                               # knot vector
        self.p       = p                                               # spline degree
        self.bc      = bc                                              # boundary conditions
        self.n_quad  = n_quad                                          # number of Gauss-Legendre quadrature points per element
        
        self.el_b    = bsp.breakpoints(self.T, self.p)                 # element boundaries
        self.Nel     = len(self.el_b) - 1                              # number of elements
        self.delta   = 1/self.Nel                                      # element length
        self.t       = self.T[1:-1]                                    # reduced knot vector for M-splines (D)
        
        self.NbaseN  = len(self.T) - self.p - 1 - self.bc*self.p       # total number of basis functions (N)
        self.NbaseD  = self.NbaseN - 1 + self.bc                       # total number of basis functions (D)
        
        
        if n_quad != None:
            
            self.pts_loc = np.polynomial.legendre.leggauss(self.n_quad)[0] # Gauss-Legendre quadrature points  (GLQP) in (-1, 1)
            self.wts_loc = np.polynomial.legendre.leggauss(self.n_quad)[1] # Gauss-Legendre quadrature weights (GLQW) in (-1, 1)
        
            # global GLQP in format (element, local point) and total number of GLQP
            self.pts     = bsp.quadrature_grid(self.el_b, self.pts_loc, self.wts_loc)[0]
            self.n_pts   = self.pts.flatten().size

            # global GLQW in format (element, local point)
            self.wts     = bsp.quadrature_grid(self.el_b, self.pts_loc, self.wts_loc)[1]

            # basis functions evaluated at quadrature points in format (element, local basis function, derivative, local point)
            self.basisN  = bsp.basis_ders_on_quad_grid(self.T, self.p    , self.pts, 0, normalize=False)
            self.basisD  = bsp.basis_ders_on_quad_grid(self.t, self.p - 1, self.pts, 0, normalize=True)
        
            
    def evaluate_N(self, eta, coeff):
        """
        Evaluates the spline space (N) at the point eta for the coefficients coeff.

        Parameters
        ----------
        eta : double
            evaluation point
        
        coeff : array_like
            FEM coefficients

        Returns
        -------
        value : double
            evaluated FEM field at the point eta
        """
        
        return eva_1d.evaluate_n(self.T, self.p, self.NbaseN, coeff, eta)
    
    
    def evaluate_D(self, eta, coeff):
        """
        Evaluates the spline space (D) at the point eta for the coefficients coeff.

        Parameters
        ----------
        eta : double
            evaluation point
        
        coeff : array_like
            FEM coefficients

        Returns
        -------
        value : double
            evaluated FEM field at the point eta
        """
        
        return eva_1d.evaluate_d(self.t, self.p - 1, self.NbaseD, coeff, eta)
        
        
# =============== multi-d B-spline tensor product space ======================        
class tensor_spline_space:
    """
    Defines a tensor product space of 1d B-spline spaces in higher dimensions.
    
    Parameters
    ----------
    spline_spaces : list of spline_space_1d
        1d B-spline spaces 
    """
    
    def __init__(self, spline_spaces):
        
        self.spaces  = spline_spaces
        
        self.T       = [spl.T       for spl in self.spaces]    # knot vectors
        self.p       = [spl.p       for spl in self.spaces]    # spline degrees
        self.bc      = [spl.bc      for spl in self.spaces]    # boundary conditions
        
        
        self.el_b    = [spl.el_b    for spl in self.spaces]    # element boundaries
        self.Nel     = [spl.Nel     for spl in self.spaces]    # number of elements
        self.delta   = [spl.delta   for spl in self.spaces]    # element length
        self.t       = [spl.t       for spl in self.spaces]    # reduced knot vectors for M-splines (D)
        
        self.NbaseN  = [spl.NbaseN  for spl in self.spaces]    # total number of basis functions (N)
        self.NbaseD  = [spl.NbaseD  for spl in self.spaces]    # total number of basis functions (D)
        
        if self.spaces[0].n_quad != None:
            
            self.n_quad  = [spl.n_quad  for spl in self.spaces]    # number of Gauss-Legendre quadrature points per element

            self.n_pts   = [spl.n_pts   for spl in self.spaces]    # total number of quadrature points

            self.pts_loc = [spl.pts_loc for spl in self.spaces]    # Gauss-Legendre quadrature points  (GLQP) in (-1, 1)
            self.wts_loc = [spl.wts_loc for spl in self.spaces]    # Gauss-Legendre quadrature weights (GLQW) in (-1, 1)

            self.pts     = [spl.pts     for spl in self.spaces]    # global GLQP in format (element, local point)
            self.wts     = [spl.wts     for spl in self.spaces]    # global GLQW in format (element, local point)

            # basis functions evaluated at quadrature points in format (element, local basis function, derivative, local point)
            self.basisN  = [spl.basisN  for spl in self.spaces] 
            self.basisD  = [spl.basisD  for spl in self.spaces]
        
        
    # =================================================
    def evaluate_NN(self, eta1, eta2, coeff):
        """
        Evaluates the spline space (NN) with coefficients 'coeff' at the point eta = (eta1, eta2).

        Parameters
        ----------
        eta1 : double
            1st component of logical evaluation point
            
        eta2 : double
            2nd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2)
        """
        
        return eva_2d.evaluate_n_n(self.T[0], self.T[1], self.p[0], self.p[1], self.NbaseN[0], self.NbaseN[1], coeff, eta1, eta2)
    
    
    # =================================================
    def evaluate_DN(self, eta1, eta2, coeff):
        """
        Evaluates the spline space (DN) with coefficients 'coeff' at the point eta = (eta1, eta2).

        Parameters
        ----------
        eta1 : double
            1st component of logical evaluation point
            
        eta2 : double
            2nd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2)
        """
        
        return eva_2d.evaluate_d_n(self.t[0], self.T[1], self.p[0] - 1, self.p[1], self.NbaseD[0], self.NbaseN[1], coeff, eta1, eta2)
    
    
    # =================================================
    def evaluate_ND(self, eta1, eta2, coeff):
        """
        Evaluates the spline space (ND) with coefficients 'coeff' at the point eta = (eta1, eta2).

        Parameters
        ----------
        eta1 : double
            1st component of logical evaluation point
            
        eta2 : double
            2nd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2)
        """
        
        return eva_2d.evaluate_n_d(self.T[0], self.t[1], self.p[0], self.p[1] - 1, self.NbaseN[0], self.NbaseD[1], coeff, eta1, eta2)
    
    
    # =================================================
    def evaluate_DD(self, eta1, eta2, coeff):
        """
        Evaluates the spline space (DD) with coefficients 'coeff' at the point eta = (eta1, eta2).

        Parameters
        ----------
        eta1 : double
            1st component of logical evaluation point
            
        eta2 : double
            2nd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2)
        """
        
        return eva_2d.evaluate_d_d(self.t[0], self.t[1], self.p[0] - 1, self.p[1] - 1, self.NbaseD[0], self.NbaseD[1], coeff, eta1, eta2)
    
    
    # =================================================
    def evaluate_NNN(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (NNN) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double
            1st component of logical evaluation point
            
        eta2 : double
            2nd component of logical evaluation point
            
        eta3 : double
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        return eva_3d.evaluate_n_n_n(self.T[0], self.T[1], self.T[2], self.p[0], self.p[1], self.p[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], coeff, eta1, eta2, eta3)
    
    
    # =================================================
    def evaluate_DNN(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (DNN) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double
            1st component of logical evaluation point
            
        eta2 : double
            2nd component of logical evaluation point
            
        eta3 : double
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        return eva_3d.evaluate_d_n_n(self.t[0], self.T[1], self.T[2], self.p[0] - 1, self.p[1], self.p[2], self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], coeff, eta1, eta2, eta3)
    
    
    # =================================================
    def evaluate_NDN(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (NDN) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double
            1st component of logical evaluation point
            
        eta2 : double
            2nd component of logical evaluation point
            
        eta3 : double
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        return eva_3d.evaluate_n_d_n(self.T[0], self.t[1], self.T[2], self.p[0], self.p[1] - 1, self.p[2], self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], coeff, eta1, eta2, eta3)
    
    
    # =================================================
    def evaluate_NND(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (NND) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double
            1st component of logical evaluation point
            
        eta2 : double
            2nd component of logical evaluation point
            
        eta3 : double
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        return eva_3d.evaluate_n_n_d(self.T[0], self.T[1], self.t[2], self.p[0], self.p[1], self.p[2] - 1, self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], coeff, eta1, eta2, eta3)
    
    
    # =================================================
    def evaluate_NDD(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (NDD) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double
            1st component of logical evaluation point
            
        eta2 : double
            2nd component of logical evaluation point
            
        eta3 : double
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        return eva_3d.evaluate_n_d_d(self.T[0], self.t[1], self.t[2], self.p[0], self.p[1] - 1, self.p[2] - 1, self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], coeff, eta1, eta2, eta3)
    
    
    # =================================================
    def evaluate_DND(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (DND) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double
            1st component of logical evaluation point
            
        eta2 : double
            2nd component of logical evaluation point
            
        eta3 : double
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        return eva_3d.evaluate_d_n_d(self.t[0], self.T[1], self.t[2], self.p[0] - 1, self.p[1], self.p[2] - 1, self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], coeff, eta1, eta2, eta3)
    
    
    # =================================================
    def evaluate_DDN(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (DDN) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double
            1st component of logical evaluation point
            
        eta2 : double
            2nd component of logical evaluation point
            
        eta3 : double
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        return eva_3d.evaluate_d_d_n(self.t[0], self.t[1], self.T[2], self.p[0] - 1, self.p[1] - 1, self.p[2], self.NbaseD[0], self.NbaseD[1], self.NbaseN[2], coeff, eta1, eta2, eta3)
    
    
    # =================================================
    def evaluate_DDD(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (DDD) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double
            1st component of logical evaluation point
            
        eta2 : double
            2nd component of logical evaluation point
            
        eta3 : double
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        return eva_3d.evaluate_d_d_d(self.t[0], self.t[1], self.t[2], self.p[0] - 1, self.p[1] - 1, self.p[2] - 1, self.NbaseD[0], self.NbaseD[1], self.NbaseD[2], coeff, eta1, eta2, eta3)