# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic modules to create a finite element spaces of B-splines.
"""


import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.bsplines as bsp


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
        number of quadrature points per element for integrations
    """
    
    def __init__(self, T, p, bc, n_quad):
        
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
        
        self.pts_loc = np.polynomial.legendre.leggauss(self.n_quad)[0] # Gauss-Legendre quadrature points (GLQP) in (-1, 1)
        self.wts_loc = np.polynomial.legendre.leggauss(self.n_quad)[1] # Gauss-Legendre quadrature weights (GLQW) in (-1, 1)
        
        
        # global GLQP in format (element, local point) and total number of GLQP
        self.pts     = np.asfortranarray(bsp.quadrature_grid(self.el_b, self.pts_loc, self.wts_loc)[0])
        self.n_pts   = self.pts.flatten().size
        
        # global GLQW in format (element, local point)
        self.wts     = np.asfortranarray(bsp.quadrature_grid(self.el_b, self.pts_loc, self.wts_loc)[1])
        
        # basis functions evaluated at quadrature points in format (element, local basis function, derivative, local point)
        self.basisN  = np.asfortranarray(bsp.basis_ders_on_quad_grid(self.T, self.p    , self.pts, 0, normalize=False))
        self.basisD  = np.asfortranarray(bsp.basis_ders_on_quad_grid(self.t, self.p - 1, self.pts, 0, normalize=True))
        
        
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
        
        self.T       = [spl.p       for spl in self.spaces]    # knot vectors
        self.p       = [spl.p       for spl in self.spaces]    # spline degrees
        self.bc      = [spl.bc      for spl in self.spaces]    # boundary conditions
        self.n_quad  = [spl.n_quad  for spl in self.spaces]    # number of Gauss-Legendre quadrature points per element
        
        self.n_pts   = [spl.n_pts   for spl in self.spaces]    # total number of quadrature points
        
        self.el_b    = [spl.el_b    for spl in self.spaces]    # element boundaries
        self.Nel     = [spl.Nel     for spl in self.spaces]    # number of elements
        self.delta   = [spl.delta   for spl in self.spaces]    # element length
        self.t       = [spl.t       for spl in self.spaces]    # reduced knot vectors for M-splines (D)
        
        self.NbaseN  = [spl.NbaseN  for spl in self.spaces]    # total number of basis functions (N)
        self.NbaseD  = [spl.NbaseD  for spl in self.spaces]    # total number of basis functions (D)
        
        self.pts_loc = [spl.pts_loc for spl in self.spaces]    # Gauss-Legendre quadrature points (GLQP) in (-1, 1)
        self.wts_loc = [spl.wts_loc for spl in self.spaces]    # Gauss-Legendre quadrature weights (GLQW) in (-1, 1)
        
        self.pts     = [spl.pts     for spl in self.spaces]    # global GLQP in format (element, local point)
        self.wts     = [spl.wts     for spl in self.spaces]    # global GLQW in format (element, local point)
        
        # basis functions evaluated at quadrature points in format (element, local basis function, derivative, local point)
        self.basisN  = [spl.basisN  for spl in self.spaces] 
        self.basisD  = [spl.basisD  for spl in self.spaces] 