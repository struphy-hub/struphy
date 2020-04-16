# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic modules to create a finite element spaces of B-splines.
"""


import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.bsplines as bsp

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
        
        
    def evaluate_0form(self, coeff, xi):
        """
        Evaluates the spline space (N) at the points xi for the coefficients coeff.

        Parameters
        ----------
        coeff : array_like
            FEM coefficients

        xi : array_like
            evaluation points
            
            
        Returns
        -------
        values : array_like
            evaluated FEM field at the points xi
        """
        
        values = bsp.collocation_matrix(self.T, self.p, xi, self.bc, normalize=False).dot(coeff)
        
        return values
    
    
    def evaluate_1form(self, coeff, xi):
        """
        Evaluates the spline space (D) at the points xi for the coefficients coeff.

        Parameters
        ----------
        coeff : array_like
            FEM coefficients

        xi : array_like
            evaluation points
            
            
        Returns
        -------
        values : array_like
            evaluated FEM field at the points xi
        """
        
        values = bsp.collocation_matrix(self.t, self.p - 1, xi, self.bc, normalize=True).dot(coeff)
        
        return values
        
        
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
        
        
    # =================================================
    def evaluate_0form_2d(self, coeff, xi):
        """
        Evaluates the spline space (NN) with coefficient 'coeff' on a tensor product grid given by (xi[0], xi[1]).

        Parameters
        ----------
        coeff : array_like
            FEM coefficients

        xi : list of array_like
            evaluation points
            
        Returns
        -------
        values : array_like
            evaluated FEM field at the tensor product grid given by (xi[0], xi[1])
        """
        
        N = [spa.csr_matrix(bsp.collocation_matrix(T, p, xi, bc)) for T, p, xi, bc in zip(self.T, self.p, xi, self.bc)]
        
        values = spa.kron(N[0], N[1]).dot(coeff.flatten()).reshape(len(xi[0]), len(xi[1]))
        
        return values
    
    
    # =================================================
    def evaluate_1form_2d(self, coeff, xi, kind, component):
        """
        Evaluates the spline space (DN, ND) or (ND, DN) with coefficients (coeff[0], coeff[1]) on a tensor product grid given by (xi[0], xi[1]).

        Parameters
        ----------
        coeff : list of array_like
            FEM coefficients

        xi : list of array_like
            evaluation points
            
        kind : string
            'Hcurl' corresponds to the sequence grad --> curl
            'Hdiv'  correpsonds to the sequence curl --> div
            
        component : int
            component of 1-form to be evaluated (1, 2)
            
        Returns
        -------
        values : array_like
            evaluated FEM field at the tensor product grid given by (xi[0], xi[1])
        """
        
        if   kind == 'Hcurl':
            
            if   component == 1:
                
                D1 = spa.csr_matrix(bsp.collocation_matrix(self.t[0], self.p[0] - 1, xi[0], self.bc[0], normalize=True))
                N2 = spa.csr_matrix(bsp.collocation_matrix(self.T[1], self.p[1],     xi[1], self.bc[1], normalize=False))
                
                values = spa.kron(D1, N2).dot(coeff[0].flatten()).reshape(len(xi[0]), len(xi[1]))
                
            elif component == 2:
                
                N1 = spa.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0],     xi[0], self.bc[0], normalize=False))
                D2 = spa.csr_matrix(bsp.collocation_matrix(self.t[1], self.p[1] - 1, xi[1], self.bc[1], normalize=True))
                
                values = spa.kron(N1, D2).dot(coeff[1].flatten()).reshape(len(xi[0]), len(xi[1]))
                
                
        if   kind == 'Hdiv':
            
            if   component == 1:
                
                N1 = spa.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0],     xi[0], self.bc[0], normalize=False))
                D2 = spa.csr_matrix(bsp.collocation_matrix(self.t[1], self.p[1] - 1, xi[1], self.bc[1], normalize=True))
                
                values = spa.kron(N1, D2).dot(coeff[0].flatten()).reshape(len(xi[0]), len(xi[1]))
                
            elif component == 2:
                
                D1 = spa.csr_matrix(bsp.collocation_matrix(self.t[0], self.p[0] - 1, xi[0], self.bc[0], normalize=True))
                N2 = spa.csr_matrix(bsp.collocation_matrix(self.T[1], self.p[1],     xi[1], self.bc[1], normalize=False))
                
                values = spa.kron(D1, N2).dot(coeff[1].flatten()).reshape(len(xi[0]), len(xi[1]))
        
        return values
    
    
    # =================================================
    def evaluate_2form_2d(self, coeff, xi):
        """
        Evaluates the spline space (DD) with coefficients 'coeff' on a tensor product grid given by (xi[0], xi[1]).

        Parameters
        ----------
        coeff : array_like
            FEM coefficients

        xi : list of array_like
            evaluation points
                
        Returns
        -------
        values : array_like
            evaluated FEM field at the tensor product grid given by (xi[0], xi[1])
        """
        
        D = [spa.csr_matrix(bsp.collocation_matrix(t, p - 1, xi, bc, normalize=True)) for T, p, xi, bc in zip(self.t, self.p, xi, self.bc)]
        
        values = spa.kron(D[0], D[1]).dot(coeff.flatten()).reshape(len(xi[0]), len(xi[1]))
        
        return values
    
    
    # =================================================
    def evaluate_0form_3d(self, coeff, xi):
        """
        Evaluates the spline space (NNN) with coefficients 'coeff' on a tensor product grid given by (xi[0], xi[1], xi[2]).

        Parameters
        ----------
        coeff : array_like
            FEM coefficients

        xi : list of array_like
            evaluation points
            
        Returns
        -------
        values : array_like
            evaluated FEM field at the tensor product grid given by (xi[0], xi[1], xi[2])
        """
        
        N = [spa.csr_matrix(bsp.collocation_matrix(T, p, xi, bc)) for T, p, xi, bc in zip(self.T, self.p, xi, self.bc)]
        
        values = spa.kron(spa.kron(N[0], N[1]), N[2]).dot(coeff.flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
        
        return values
    
    
    # =================================================
    def evaluate_1form_3d(self, coeff, xi, component):
        """
        Evaluates the spline space (DNN, NDN, NND) with coefficients (coeff[0], coeff[1], coeff[2]) on a tensor product grid given by (xi[0], xi[1], xi[2]).

        Parameters
        ----------
        coeff : list of array_like
            FEM coefficients

        xi : list of array_like
            evaluation points
            
        component : int
            component of 1-form to be evaluated (1, 2, 3)
                
        Returns
        -------
        values : array_like
            evaluated FEM field at the tensor product grid given by (xi[0], xi[1], xi[2])
        """
        
        if   component == 1:
        
            D1 = spa.csr_matrix(bsp.collocation_matrix(self.t[0], self.p[0] - 1, xi[0], self.bc[0], normalize=True))
            N2 = spa.csr_matrix(bsp.collocation_matrix(self.T[1], self.p[1],     xi[1], self.bc[1], normalize=False))
            N3 = spa.csr_matrix(bsp.collocation_matrix(self.T[2], self.p[2],     xi[2], self.bc[2], normalize=False))
        
            values = spa.kron(spa.kron(D1, N2), N3).dot(coeff[0].flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
            
        elif component == 2:
            
            N1 = spa.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0],     xi[0], self.bc[0], normalize=False))
            D2 = spa.csr_matrix(bsp.collocation_matrix(self.t[1], self.p[1] - 1, xi[1], self.bc[1], normalize=True))
            N3 = spa.csr_matrix(bsp.collocation_matrix(self.T[2], self.p[2],     xi[2], self.bc[2], normalize=False))
            
            values = spa.kron(spa.kron(N1, D2), N3).dot(coeff[1].flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
            
        elif component == 3:
            
            N1 = spa.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0],     xi[0], self.bc[0], normalize=False))
            N2 = spa.csr_matrix(bsp.collocation_matrix(self.T[1], self.p[1],     xi[1], self.bc[1], normalize=False))
            D3 = spa.csr_matrix(bsp.collocation_matrix(self.t[2], self.p[2] - 1, xi[2], self.bc[2], normalize=True))
            
            values = spa.kron(spa.kron(N1, N2), D3).dot(coeff[2].flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
            
        return values
    
    
    # =================================================
    def evaluate_2form_3d(self, coeff, xi, component):
        """
        Evaluates the spline space (NDD, DND, DDN) with coefficients (coeff[0], coeff[1], coeff[2]) on a tensor product grid given by (xi[0], xi[1], xi[2]).

        Parameters
        ----------
        coeff : list of array_like
            FEM coefficients

        xi : list of array_like
            evaluation points
            
        component : int
            component of 1-form to be evaluated (1, 2, 3)
            
        Returns
        -------
        values : array_like
            evaluated FEM field at the tensor product grid given by (xi[0], xi[1], xi[2])
        """
        
        if   component == 1:
        
            N1 = spa.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0],     xi[0], self.bc[0], normalize=False))
            D2 = spa.csr_matrix(bsp.collocation_matrix(self.t[1], self.p[1] - 1, xi[1], self.bc[1], normalize=True))
            D3 = spa.csr_matrix(bsp.collocation_matrix(self.t[2], self.p[2] - 1, xi[2], self.bc[2], normalize=True))
            
            values = spa.kron(spa.kron(N1, D2), D3).dot(coeff[0].flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
            
        elif component == 2:
            
            D1 = spa.csr_matrix(bsp.collocation_matrix(self.t[0], self.p[0] - 1, xi[0], self.bc[0], normalize=True))
            N2 = spa.csr_matrix(bsp.collocation_matrix(self.T[1], self.p[1],     xi[1], self.bc[1], normalize=False))
            D3 = spa.csr_matrix(bsp.collocation_matrix(self.t[2], self.p[2] - 1, xi[2], self.bc[2], normalize=True))
            
            values = spa.kron(spa.kron(D1, N2), D3).dot(coeff[1].flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
            
        elif component == 3:
            
            D1 = spa.csr_matrix(bsp.collocation_matrix(self.t[0], self.p[0] - 1, xi[0], self.bc[0], normalize=True))
            D2 = spa.csr_matrix(bsp.collocation_matrix(self.t[1], self.p[1] - 1, xi[1], self.bc[1], normalize=True))
            N3 = spa.csr_matrix(bsp.collocation_matrix(self.T[2], self.p[2],     xi[2], self.bc[2], normalize=False))
            
            values = spa.kron(spa.kron(D1, D2), N3).dot(coeff[2].flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
            
            
        return values
    
    
    # =================================================
    def evaluate_3form_3d(self, coeff, xi):
        """
        Evaluates the spline space (DDD) with coefficients 'coeff' on a tensor product grid given by (xi[0], xi[1], xi[2]).

        Parameters
        ----------
        coeff : array_like
            FEM coefficients

        xi : list of array_like
            evaluation points
            
        Returns
        -------
        values : array_like
            evaluated FEM field at the tensor product grid given by (xi[0], xi[1], xi[2])
        """
        
        D = [spa.csr_matrix(bsp.collocation_matrix(t, p - 1, xi, bc, normalize=True)) for T, p, xi, bc in zip(self.t, self.p, xi, self.bc)]
        
        values = spa.kron(spa.kron(D[0], D[1]), D[2]).dot(coeff.flatten()).reshape(len(xi[0]), len(xi[1]), len(xi[2]))
        
        return values