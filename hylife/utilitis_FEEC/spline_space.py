# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic modules to create tensor-product finite element spaces of univariate B-splines.
"""


import numpy        as np
import scipy.sparse as spa

import matplotlib.pyplot as plt

import hylife.utilitis_FEEC.bsplines         as bsp
import hylife.utilitis_FEEC.bsplines_kernels as bsp_kernels

import hylife.utilitis_FEEC.basics.spline_evaluation_1d as eva_1d
import hylife.utilitis_FEEC.basics.spline_evaluation_2d as eva_2d
import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva_3d

import hylife.utilitis_FEEC.basics.mass_matrices_1d as mass_1d
import hylife.utilitis_FEEC.basics.mass_matrices_2d as mass_2d
import hylife.utilitis_FEEC.basics.mass_matrices_3d as mass_3d

import hylife.utilitis_FEEC.derivatives.derivatives as der


# =============== 1d B-spline space ======================
class spline_space_1d:
    """
    Defines a 1d space of B-splines.
    
    Parameters
    ----------
    Nel : int
        number of elements of discretized 1D domain [0, 1]
        
    p : int
        spline degree
        
    spl_kind : boolean
        kind of spline space (True = periodic, False = clamped)
        
    n_quad : int
        optional: number of Gauss-Legendre quadrature points per element for integrations
    """
    
    def __init__(self, Nel, p, spl_kind, n_quad=None):
        
        self.Nel      = Nel                                              # number of elements
        self.p        = p                                                # spline degree
        self.spl_kind = spl_kind                                         # kind of spline space (periodic or clamped)
        self.n_quad   = n_quad                                           # number of Gauss-Legendre points per element
        
        self.el_b     = np.linspace(0., 1., Nel + 1)                     # element boundaries
        self.delta    = 1/self.Nel                                       # element length
         
        self.T        = bsp.make_knots(self.el_b, self.p, self.spl_kind) # spline knot vector for B-splines (N)
        self.t        = self.T[1:-1]                                     # reduced knot vector for M-splines (D)
        
        self.NbaseN   = len(self.T) - self.p - 1 - self.spl_kind*self.p  # total number of B-splines (N)
        self.NbaseD   = self.NbaseN - 1 + self.spl_kind                  # total number of M-splines (D)
        
        
        if n_quad != None:
            
            self.pts_loc = np.polynomial.legendre.leggauss(self.n_quad)[0] # Gauss-Legendre points  (GLQP) in (-1, 1)
            self.wts_loc = np.polynomial.legendre.leggauss(self.n_quad)[1] # Gauss-Legendre weights (GLQW) in (-1, 1)
        
            # global GLQP in format (element, local point) and total number of GLQP
            self.pts     = bsp.quadrature_grid(self.el_b, self.pts_loc, self.wts_loc)[0]
            self.n_pts   = self.pts.flatten().size

            # global GLQW in format (element, local point)
            self.wts     = bsp.quadrature_grid(self.el_b, self.pts_loc, self.wts_loc)[1]

            # basis functions evaluated at quadrature points in format (element, local basis function, derivative, local point)
            self.basisN  = bsp.basis_ders_on_quad_grid(self.T, self.p    , self.pts, 0, normalize=False)
            self.basisD  = bsp.basis_ders_on_quad_grid(self.t, self.p - 1, self.pts, 0, normalize=True)
        
    
    # =================================================
    def assemble_M0(self, mapping=None):
        self.M0 = mass_1d.get_V0(self, mapping)
        
    # =================================================
    def assemble_M1(self, mapping=None):
        self.M1 = mass_1d.get_V1(self, mapping)
    
    
    # =================================================
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
        
        if isinstance(eta, np.ndarray):
            
            if eta.ndim == 1:
                values = np.empty(eta.size, dtype=float)

                for i in range(eta.size):
                    values[i] = eva_1d.evaluate_n(self.T, self.p, self.NbaseN, coeff, eta[i])
                    
            elif eta.ndim == 2:
                values = np.empty(eta.shape, dtype=float)
                
                for i in range(eta.shape[0]):
                    for j in range(eta.shape[1]):
                        values[i, j] = eva_1d.evaluate_n(self.T, self.p, self.NbaseN, coeff, eta[i, j])
                    
            return values
        
        else:
            return eva_1d.evaluate_n(self.T, self.p, self.NbaseN, coeff, eta)
        
        
    # =================================================
    def evaluate_dN(self, eta, coeff):
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
        
        if isinstance(eta, np.ndarray):
            
            if eta.ndim == 1:
                values = np.empty(eta.size, dtype=float)

                for i in range(eta.size):
                    values[i] = eva_1d.evaluate_diffn(self.T, self.p, self.NbaseN, coeff, eta[i])
                    
            elif eta.ndim == 2:
                values = np.empty(eta.shape, dtype=float)
                
                for i in range(eta.shape[0]):
                    for j in range(eta.shape[1]):
                        values[i, j] = eva_1d.evaluate_diffn(self.T, self.p, self.NbaseN, coeff, eta[i, j])
                    
            return values
        
        else:
            return eva_1d.evaluate_diffn(self.T, self.p, self.NbaseN, coeff, eta)
    
    
    
    # =================================================
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
        
        if isinstance(eta, np.ndarray):
            
            if eta.ndim == 1:
                values = np.empty(eta.size, dtype=float)

                for i in range(eta.size):
                    values[i] = eva_1d.evaluate_d(self.t, self.p - 1, self.NbaseD, coeff, eta[i])
                    
            elif eta.ndim == 2:
                values = np.empty(eta.shape, dtype=float)
                
                for i in range(eta.shape[0]):
                    for j in range(eta.shape[1]):
                        values[i, j] = eva_1d.evaluate_d(self.t, self.p - 1, self.NbaseD, coeff, eta[i, j])
                    
            return values
        
        else:
            return eva_1d.evaluate_d(self.t, self.p - 1, self.NbaseD, coeff, eta)
        
        
    # =================================================
    def plot_splines(self, n_pts, which='B-splines'):

        etaplot = np.linspace(0., 1., n_pts)

        if which == 'B-splines':

            coeff = np.zeros(self.NbaseN, dtype=float)

            for i in range(self.NbaseN):
                coeff[:] = 0.
                coeff[i] = 1.
                plt.plot(etaplot, self.evaluate_N(etaplot, coeff))

        elif which == 'M-splines':

            coeff = np.zeros(self.NbaseD, dtype=float)

            for i in range(self.NbaseD):
                coeff[:] = 0.
                coeff[i] = 1.
                plt.plot(etaplot, self.evaluate_D(etaplot, coeff))

        else:
            print('only B-splines and M-splines available')
            
        
        
# =============== multi-d B-spline tensor product space ======================        
class tensor_spline_space:
    """
    Defines a tensor product space of 1d B-spline spaces in higher dimensions (2d and 3d).
    
    Parameters
    ----------
    spline_spaces : list of spline_space_1d
        1d B-spline spaces from which the tensor_product B-spline space is built 
    """
    
    def __init__(self, spline_spaces, ):
        
        self.spaces   = spline_spaces                            # 1D B-spline spaces
        self.dim      = len(self.spaces)                         # number of 1D B-spline spaces (= dimension)
        
        self.T        = [spl.T        for spl in self.spaces]    # knot vectors
        self.p        = [spl.p        for spl in self.spaces]    # spline degrees
        self.spl_kind = [spl.spl_kind for spl in self.spaces]    # kind of spline space (periodic or clamped)
        
        
        self.el_b     = [spl.el_b     for spl in self.spaces]    # element boundaries
        self.Nel      = [spl.Nel      for spl in self.spaces]    # number of elements
        self.delta    = [spl.delta    for spl in self.spaces]    # element length
        self.t        = [spl.t        for spl in self.spaces]    # reduced knot vectors for M-splines (D)
        
        self.NbaseN   = [spl.NbaseN   for spl in self.spaces]    # total number of basis functions (N)
        self.NbaseD   = [spl.NbaseD   for spl in self.spaces]    # total number of basis functions (D)
        
        if self.spaces[0].n_quad != None:
            
            self.n_quad  = [spl.n_quad  for spl in self.spaces]  # number of Gauss-Legendre quadrature points per element

            self.n_pts   = [spl.n_pts   for spl in self.spaces]  # total number of quadrature points

            self.pts_loc = [spl.pts_loc for spl in self.spaces]  # Gauss-Legendre quadrature points  (GLQP) in (-1, 1)
            self.wts_loc = [spl.wts_loc for spl in self.spaces]  # Gauss-Legendre quadrature weights (GLQW) in (-1, 1)

            self.pts     = [spl.pts     for spl in self.spaces]  # global GLQP in format (element, local point)
            self.wts     = [spl.wts     for spl in self.spaces]  # global GLQW in format (element, local point)

            # basis functions evaluated at quadrature points in format (element, local basis function, derivative, local point)
            self.basisN  = [spl.basisN  for spl in self.spaces] 
            self.basisD  = [spl.basisD  for spl in self.spaces]
        
        
        # number of basis functions of discrete tensor-product p-forms in 2D
        if self.dim == 2:
            
            # number of basis functions in each direction
            self.Nbase_0form =  [self.NbaseN[0], self.NbaseN[1]]

            self.Nbase_1form = [[self.NbaseD[0], self.NbaseN[1]], 
                                [self.NbaseN[0], self.NbaseD[1]], 
                                [self.NbaseN[0], self.NbaseN[1]]]

            self.Nbase_2form = [[self.NbaseN[0], self.NbaseD[1]], 
                                [self.NbaseD[0], self.NbaseN[1]], 
                                [self.NbaseD[0], self.NbaseD[1]]]

            self.Nbase_3form =  [self.NbaseD[0], self.NbaseD[1]]

            # total number of basis functions
            self.Ntot_0form  =  self.NbaseN[0]*self.NbaseN[1]

            self.Ntot_1form  = [self.NbaseD[0]*self.NbaseN[1], 
                                self.NbaseN[0]*self.NbaseD[1], 
                                self.NbaseN[0]*self.NbaseN[1]]

            self.Ntot_2form  = [self.NbaseN[0]*self.NbaseD[1], 
                                self.NbaseD[0]*self.NbaseN[1], 
                                self.NbaseD[0]*self.NbaseD[1]]

            self.Ntot_3form  =  self.NbaseD[0]*self.NbaseD[1]
            
            # cumulative number of basis functions for vector-valued spaces
            self.Ntot_1form_cum = [self.Ntot_1form[0],
                                   self.Ntot_1form[0] + self.Ntot_1form[1], 
                                   self.Ntot_1form[0] + self.Ntot_1form[1] + self.Ntot_1form[2]]
            
            self.Ntot_2form_cum = [self.Ntot_2form[0],
                                   self.Ntot_2form[0] + self.Ntot_2form[1],
                                   self.Ntot_2form[0] + self.Ntot_2form[1] + self.Ntot_2form[2]]
        
        
        # number of basis functions of discrete tensor-product p-forms in 3D
        if self.dim == 3:
            
            # number of basis functions in each direction
            self.Nbase_0form =  [self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]]

            self.Nbase_1form = [[self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]], 
                                [self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]], 
                                [self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]]]

            self.Nbase_2form = [[self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], 
                                [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], 
                                [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]]]

            self.Nbase_3form =  [self.NbaseD[0], self.NbaseD[1], self.NbaseD[2]]

            # total number of basis functions
            self.Ntot_0form  =  self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2] 

            self.Ntot_1form  = [self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], 
                                self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], 
                                self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]]

            self.Ntot_2form  = [self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2], 
                                self.NbaseD[0]*self.NbaseN[1]*self.NbaseD[2], 
                                self.NbaseD[0]*self.NbaseD[1]*self.NbaseN[2]]

            self.Ntot_3form  =  self.NbaseD[0]*self.NbaseD[1]*self.NbaseD[2]
            
            # cumulative number of basis functions for vector-valued spaces
            self.Ntot_1form_cum = [self.Ntot_1form[0],
                                   self.Ntot_1form[0] + self.Ntot_1form[1], 
                                   self.Ntot_1form[0] + self.Ntot_1form[1] + self.Ntot_1form[2]]
            
            self.Ntot_2form_cum = [self.Ntot_2form[0],
                                   self.Ntot_2form[0] + self.Ntot_2form[1],
                                   self.Ntot_2form[0] + self.Ntot_2form[1] + self.Ntot_2form[2]]
        
    
    # ====== spline extraction operators ===============
    def set_extraction_operators(self, bc=['f', 'f'], polar_splines=None):
        
        # set boundary conditions in first logical direction
        self.bc = bc
        
        # set polar splines
        self.polar_splines = polar_splines
        
        # 2D extraction operators
        if self.dim == 2:
            
            n1, n2 = self.NbaseN
            d1, d2 = self.NbaseD

            # standard domain
            if polar_splines == None:

                self.polar = False

                # including boundary splines
                self.E0_pol_all = spa.identity(n1*n2        , dtype=float, format='csr')
                self.E1_pol_all = spa.identity(d1*n2 + n1*d2, dtype=float, format='csr')
                self.E2_pol_all = spa.identity(n1*d2 + d1*n2, dtype=float, format='csr')
                self.E3_pol_all = spa.identity(d1*d2        , dtype=float, format='csr')
                
                self.E0_all     =            self.E0_pol_all.copy()
                self.E1_all     = spa.bmat([[self.E1_pol_all, None], [None, self.E0_pol_all]], format='csr')
                self.E2_all     = spa.bmat([[self.E2_pol_all, None], [None, self.E3_pol_all]], format='csr')
                self.E3_all     =            self.E3_pol_all.copy()
                
                # without boundary splines
                E_NN = spa.identity(n1*n2, format='csr')
                E_DN = spa.identity(d1*n2, format='csr')
                E_ND = spa.identity(n1*d2, format='csr')
                E_DD = spa.identity(d1*d2, format='csr')
                
                # remove contributions from N-splines at eta1 = 0
                if bc[0] == 'd' and self.spl_kind[0] == False:
                    E_NN = E_NN[n2:, :]
                    E_ND = E_ND[d2:, :]
                elif bc[0] == 'd' and self.spl_kind[0] == True:
                    raise ValueError('dirichlet boundary conditions can only be set with clamped splines')
                    
                # remove contributions from N-splines at eta1 = 1
                if bc[1] == 'd' and self.spl_kind[0] == False:
                    E_NN = E_NN[:-n2, :]
                    E_ND = E_ND[:-d2, :]
                elif bc[0] == 'd' and self.spl_kind[0] == True:
                    raise ValueError('dirichlet boundary conditions can only be set with clamped splines')
                    
                self.E0_pol = E_NN.tocsr().copy()
                self.E1_pol = spa.bmat([[E_DN, None], [None, E_ND]], format='csr')
                self.E2_pol = spa.bmat([[E_ND, None], [None, E_DN]], format='csr')
                self.E3_pol = E_DD.tocsr().copy()
                
                self.E0     =            self.E0_pol.copy()
                self.E1     = spa.bmat([[self.E1_pol, None], [None, self.E0_pol]], format='csr')
                self.E2     = spa.bmat([[self.E2_pol, None], [None, self.E3_pol]], format='csr')
                self.E3     =            self.E3_pol.copy()
                    
            # polar domain        
            else:

                self.polar = True
                
                # including boundary splines
                self.E0_pol_all = polar_splines.E0.copy()
                self.E1_pol_all = polar_splines.E1C.copy()
                self.E2_pol_all = polar_splines.E1D.copy()
                self.E3_pol_all = polar_splines.E2.copy()
                
                self.E0_all     =            self.E0_pol_all.copy()
                self.E1_all     = spa.bmat([[self.E1_pol_all, None], [None, self.E0_pol_all]], format='csr')
                self.E2_all     = spa.bmat([[self.E2_pol_all, None], [None, self.E3_pol_all]], format='csr')
                self.E3_all     =            self.E3_pol_all.copy()
                
                # without boundary splines
                E0_NN = polar_splines.E0.copy()
                
                E1_DN = polar_splines.E1C.copy()[:(0 + (d1 - 1)*d2) , :]
                E1_ND = polar_splines.E1C.copy()[ (0 + (d1 - 1)*d2):, :]
                
                E2_ND = polar_splines.E1D.copy()[:(2 + (n1 - 2)*d2) , :]
                E2_DN = polar_splines.E1D.copy()[ (2 + (n1 - 2)*d2):, :]
                
                E3_DD = polar_splines.E2.copy()
                
                # remove contributions from N-splines at eta1 = 1
                if bc[1] == 'd' and self.spl_kind[0] == False:
                    E0_NN = E0_NN[:-n2, :]
                    E1_ND = E1_ND[:-d2, :]
                    E2_ND = E2_ND[:-d2, :]
                elif bc[0] == 'd' and self.spl_kind[0] == True:
                    raise ValueError('dirichlet boundary conditions can only be set with clamped splines')
                    
                self.E0_pol = E0_NN.tocsr().copy()
                self.E1_pol = spa.bmat([[E1_DN], [E1_ND]], format='csr')
                self.E2_pol = spa.bmat([[E2_ND], [E2_DN]], format='csr')
                self.E3_pol = E3_DD.tocsr().copy()
                    
                self.E0     =            self.E0_pol.copy()
                self.E1     = spa.bmat([[self.E1_pol, None], [None, self.E0_pol]], format='csr')
                self.E2     = spa.bmat([[self.E2_pol, None], [None, self.E3_pol]], format='csr')
                self.E3     =            self.E3_pol.copy()
                    
    
        # 3D extraction operators    
        if self.dim == 3:
            
            n1, n2, n3 = self.NbaseN
            d1, d2, d3 = self.NbaseD
            
            # standard domain
            if polar_splines == None:

                self.polar = False
                
                # including boundary splines
                self.E0_pol_all = spa.identity(n1*n2        , dtype=float, format='csr')
                self.E1_pol_all = spa.identity(d1*n2 + n1*d2, dtype=float, format='csr')
                self.E2_pol_all = spa.identity(n1*d2 + d1*n2, dtype=float, format='csr')
                self.E3_pol_all = spa.identity(d1*d2        , dtype=float, format='csr')
                
                self.E0_all     = spa.identity(self.Ntot_0form       , dtype=float, format='csr')
                self.E1_all     = spa.identity(self.Ntot_1form_cum[2], dtype=float, format='csr')
                self.E2_all     = spa.identity(self.Ntot_2form_cum[2], dtype=float, format='csr')
                self.E3_all     = spa.identity(self.Ntot_3form       , dtype=float, format='csr')
                
                # without boundary splines
                E_NN = spa.identity(n1*n2, format='csr')
                E_DN = spa.identity(d1*n2, format='csr')
                E_ND = spa.identity(n1*d2, format='csr')
                E_DD = spa.identity(d1*d2, format='csr')
                
                # remove contributions from N-splines at eta1 = 0
                if bc[0] == 'd' and self.spl_kind[0] == False:
                    E_NN = E_NN[n2:, :]
                    E_ND = E_ND[d2:, :]
                elif bc[0] == 'd' and self.spl_kind[0] == True:
                    raise ValueError('dirichlet boundary conditions can only be set with clamped splines')
                    
                # remove contributions from N-splines at eta1 = 1
                if bc[1] == 'd' and self.spl_kind[0] == False:
                    E_NN = E_NN[:-n2, :]
                    E_ND = E_ND[:-d2, :]
                elif bc[0] == 'd' and self.spl_kind[0] == True:
                    raise ValueError('dirichlet boundary conditions can only be set with clamped splines')
                    
                self.E0_pol = E_NN.tocsr().copy()
                self.E1_pol = spa.bmat([[E_DN, None], [None, E_ND]], format='csr')
                self.E2_pol = spa.bmat([[E_ND, None], [None, E_DN]], format='csr')
                self.E3_pol = E_DD.tocsr().copy()
                
                self.E0     = spa.kron(self.E0_pol, spa.identity(n3), format='csr')  
                E1_1        = spa.kron(self.E1_pol, spa.identity(n3), format='csr')
                E1_3        = spa.kron(self.E0_pol, spa.identity(d3), format='csr')
                
                E2_1        = spa.kron(self.E2_pol, spa.identity(d3), format='csr')
                E2_3        = spa.kron(self.E3_pol, spa.identity(n3), format='csr')
                self.E3     = spa.kron(self.E3_pol, spa.identity(d3), format='csr')

                self.E1     = spa.bmat([[E1_1, None], [None, E1_3]], format='csr')
                self.E2     = spa.bmat([[E2_1, None], [None, E2_3]], format='csr')

            # polar domain
            else:

                self.polar = True
                
                # including boundary splines
                self.E0_pol_all = polar_splines.E0_pol.copy()
                self.E1_pol_all = polar_splines.E1_pol.copy()
                self.E2_pol_all = polar_splines.E2_pol.copy()
                self.E3_pol_all = polar_splines.E3_pol.copy()                                                        
                                               
                self.E0_all     = polar_splines.E0.copy()
                self.E1_all     = polar_splines.E1.copy()
                self.E2_all     = polar_splines.E2.copy()
                self.E3_all     = polar_splines.E3.copy()
                
                # without boundary splines
                E0_NN = polar_splines.E0_pol.copy()
                
                E1_DN = polar_splines.E1_pol.copy()[:(0 + (d1 - 1)*d2) , :]
                E1_ND = polar_splines.E1_pol.copy()[ (0 + (d1 - 1)*d2):, :]
                
                E2_ND = polar_splines.E2_pol.copy()[:(2 + (n1 - 2)*d2) , :]
                E2_DN = polar_splines.E2_pol.copy()[ (2 + (n1 - 2)*d2):, :]
                
                E3_DD = polar_splines.E3_pol.copy()
                
                # remove contributions from N-splines at eta1 = 1
                if bc[1] == 'd' and self.spl_kind[0] == False:
                    E0_NN = E0_NN[:-n2, :]
                    E1_ND = E1_ND[:-d2, :]
                    E2_ND = E2_ND[:-d2, :]
                elif bc[0] == 'd' and self.spl_kind[0] == True:
                    raise ValueError('dirichlet boundary conditions can only be set with clamped splines')
                    
                self.E0_pol = E0_NN.tocsr().copy()
                self.E1_pol = spa.bmat([[E1_DN], [E1_ND]], format='csr')
                self.E2_pol = spa.bmat([[E2_ND], [E2_DN]], format='csr')
                self.E3_pol = E3_DD.tocsr().copy()
                
                self.E0     = spa.kron(self.E0_pol, spa.identity(n3), format='csr')  
                E1_1        = spa.kron(self.E1_pol, spa.identity(n3), format='csr')
                E1_3        = spa.kron(self.E0_pol, spa.identity(d3), format='csr')
                
                E2_1        = spa.kron(self.E2_pol, spa.identity(d3), format='csr')
                E2_3        = spa.kron(self.E3_pol, spa.identity(n3), format='csr')
                self.E3     = spa.kron(self.E3_pol, spa.identity(d3), format='csr')

                self.E1     = spa.bmat([[E1_1, None], [None, E1_3]], format='csr')
                self.E2     = spa.bmat([[E2_1, None], [None, E2_3]], format='csr')
    
    
    
    # ========= discrete derivatives (2D) =============
    def set_derivatives_2D(self, mode_n):

        derivatives = der.discrete_derivatives_2D(self, mode_n)
           
        # discrete gradient
        self.G = derivatives.G

        # discrete curl
        self.C    = derivatives.C
        self.C_wn = derivatives.C_wn
        self.C_all = derivatives.C_all

        # discrete div
        self.D = derivatives.D 
        
    # ========= discrete derivatives (3D) =============
    def set_derivatives_3D(self):

        derivatives = der.discrete_derivatives_3D(self)
           
        # discrete gradient
        self.G = derivatives.G

        # discrete curl
        self.C = derivatives.C

        # discrete div
        self.D = derivatives.D 
    
    # ============== mass matrices (2D) ===============
    def assemble_M0_2D(self, domain):
        self.M0 = mass_2d.get_M0(self, domain)

    def assemble_M1_2D(self, domain):
        self.M1 = mass_2d.get_M1(self, domain)

    def assemble_M2_2D(self, domain):
        self.M2 = mass_2d.get_M2(self, domain)

    def assemble_M3_2D(self, domain):
        self.M3 = mass_2d.get_M3(self, domain)
    
    # ============== mass matrices (3D) ===============
    def assemble_M0( self, domain):
        self.M0 = mass_3d.get_M0( self, domain)

    def assemble_M1( self, domain):
        self.M1 = mass_3d.get_M1( self, domain)

    def assemble_M2( self, domain):
        self.M2 = mass_3d.get_M2( self, domain)

    def assemble_M3( self, domain):
        self.M3 = mass_3d.get_M3( self, domain)

    def assemble_Mv0(self, domain):
        self.Mv = mass_3d.get_Mv0(self, domain)

    def assemble_Mv2(self, domain):
        self.Mv = mass_3d.get_Mv2(self, domain)
    
    # ========= extraction of coefficients =========
    def extract_0form(self, coeff):
        
        if coeff.size == self.E0_all.shape[0]:
            coeff0 = self.E0_all.T.dot(coeff)
        
        elif coeff.size == self.E0.shape[0]:
            coeff0 = self.E0.T.dot(coeff)
            
        else:
            print('number of coefficients is not correct')
            
        coeff0 = coeff0.reshape(self.Nbase_0form)
            
        return coeff0
    
    def extract_1form(self, coeff):
        
        if coeff.size == self.E1_all.shape[0]:
            coeff1 = self.E1_all.T.dot(coeff)
        
        elif coeff.size == self.E1.shape[0]:
            coeff1 = self.E1.T.dot(coeff)
            
        else:
            print('number of coefficients is not correct')
        
        coeff1_1, coeff1_2, coeff1_3 = np.split(coeff1, [self.Ntot_1form_cum[0], self.Ntot_1form_cum[1]])
        
        coeff1_1 = coeff1_1.reshape(self.Nbase_1form[0])
        coeff1_2 = coeff1_2.reshape(self.Nbase_1form[1])
        coeff1_3 = coeff1_3.reshape(self.Nbase_1form[2])
        
        return coeff1_1, coeff1_2, coeff1_3
        
    def extract_2form(self, coeff):
        
        if coeff.size == self.E2_all.shape[0]:
            coeff2 = self.E2_all.T.dot(coeff)
        
        elif coeff.size == self.E2.shape[0]:
            coeff2 = self.E2.T.dot(coeff)
            
        else:
            print('number of coefficients is not correct')
        
        coeff2_1, coeff2_2, coeff2_3 = np.split(coeff2, [self.Ntot_2form_cum[0], self.Ntot_2form_cum[1]])
        
        coeff2_1 = coeff2_1.reshape(self.Nbase_2form[0])
        coeff2_2 = coeff2_2.reshape(self.Nbase_2form[1])
        coeff2_3 = coeff2_3.reshape(self.Nbase_2form[2])
        
        return coeff2_1, coeff2_2, coeff2_3
        
    def extract_3form(self, coeff):
        
        coeff3 = self.E3_all.T.dot(coeff)
        coeff3 = coeff3.reshape(self.Nbase_3form)
        
        return coeff3
    
    def extract_0form_vec(self, coeff):
        
        if coeff.size == 3*self.E0_all.shape[0]:
            E = spa.bmat([[self.E0_all, None, None], [None, self.E0_all, None], [None, None, self.E0_all]], format='csr')
            coeff0 = E.T.dot(coeff)
        
        elif coeff.size == self.E0.shape[0] + 2*self.E0_all.shape[0]:
            E = spa.bmat([[self.E0, None, None], [None, self.E0_all, None], [None, None, self.E0_all]], format='csr')
            coeff0 = E.T.dot(coeff)
            
        else:
            print('number of coefficients is not correct')
            
        coeff0_1, coeff0_2, coeff0_3 = np.split(coeff0, [self.Ntot_0form, 2*self.Ntot_0form])
        
        coeff0_1 = coeff0_1.reshape(self.Nbase_0form)
        coeff0_2 = coeff0_2.reshape(self.Nbase_0form)
        coeff0_3 = coeff0_3.reshape(self.Nbase_0form)
        
        return coeff0_1, coeff0_2, coeff0_3
    

    # =================================================
    def evaluate_NN(self, eta1, eta2, coeff, which):
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
            
        which : string
            which space (V0 or V1)
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2)
        """
        
        # extract coefficients
        if which == 'V0':
            coeff = self.extract_0form(coeff)
        else:
            coeff = self.extract_1form(coeff)[2]
        
        # evaluate FEM field at given points
        if isinstance(eta1, np.ndarray):
            
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)
                
                eva_2d.evaluate_tensor_product(self.T[0], self.T[1], self.p[0], self.p[1], self.NbaseN[0], self.NbaseN[1], coeff, eta1, eta2, values, 0)
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)
                
                eva_2d.evaluate_matrix(self.T[0], self.T[1], self.p[0], self.p[1], self.NbaseN[0], self.NbaseN[1], coeff, eta1, eta2, eta1.shape[0], eta2.shape[1], values, 0)
                
            return values
        
        else:
            return eva_2d.evaluate_n_n(self.T[0], self.T[1], self.p[0], self.p[1], self.NbaseN[0], self.NbaseN[1], coeff, eta1, eta2)
    
    
    # =================================================
    def evaluate_DN(self, eta1, eta2, coeff, which):
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
            
        which : string
            which space (V1 or V2)
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2)
        """
        
        # extract coefficients
        if which == 'V1':
            coeff = self.extract_1form(coeff)[0]
        else:
            coeff = self.extract_2form(coeff)[1]
        
        # evaluate FEM field at given points
        if isinstance(eta1, np.ndarray):
            
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)
                
                eva_2d.evaluate_tensor_product(self.t[0], self.T[1], self.p[0] - 1, self.p[1], self.NbaseD[0], self.NbaseN[1], coeff, eta1, eta2, values, 11)
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)
                
                eva_2d.evaluate_matrix(self.t[0], self.T[1], self.p[0] - 1, self.p[1], self.NbaseD[0], self.NbaseN[1], coeff, eta1, eta2, eta1.shape[0], eta2.shape[1], values, 11)
                
            return values
        
        else:
            return eva_2d.evaluate_d_n(self.t[0], self.T[1], self.p[0] - 1, self.p[1], self.NbaseD[0], self.NbaseN[1], coeff, eta1, eta2)
    
    
    # =================================================
    def evaluate_ND(self, eta1, eta2, coeff, which):
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
              
        which : string
            which space (V1 or V2)
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2)
        """
        
        # extract coefficients
        if which == 'V1':
            coeff = self.extract_1form(coeff)[1]
        else:
            coeff = self.extract_2form(coeff)[0]
        
        # evaluate FEM field at given points
        if isinstance(eta1, np.ndarray):
            
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)
                
                eva_2d.evaluate_tensor_product(self.T[0], self.t[1], self.p[0], self.p[1] - 1, self.NbaseN[0], self.NbaseD[1], coeff, eta1, eta2, values, 12)
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)
                
                eva_2d.evaluate_matrix(self.T[0], self.t[1], self.p[0], self.p[1] - 1, self.NbaseN[0], self.NbaseD[1], coeff, eta1, eta2, eta1.shape[0], eta2.shape[1], values, 12)
                
            return values
            
        else:
            return eva_2d.evaluate_n_d(self.T[0], self.t[1], self.p[0], self.p[1] - 1, self.NbaseN[0], self.NbaseD[1], coeff, eta1, eta2)
    
    
    # =================================================
    def evaluate_DD(self, eta1, eta2, coeff, which):
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

        which : string
            which space (V2 or V3)
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2)
        """
        
        # extract coefficients
        if which == 'V2':
            coeff = self.extract_2form(coeff)[2]
        else:
            coeff = self.extract_3form(coeff)
            
        
        # evaluate FEM field at given points
        if isinstance(eta1, np.ndarray):
            
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)
                
                eva_2d.evaluate_tensor_product(self.t[0], self.t[1], self.p[0] - 1, self.p[1] - 1, self.NbaseD[0], self.NbaseD[1], coeff, eta1, eta2, values, 2)
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)
                
                eva_2d.evaluate_matrix(self.t[0], self.t[1], self.p[0] - 1, self.p[1] - 1, self.NbaseD[0], self.NbaseD[1], coeff, eta1, eta2, eta1.shape[0], eta2.shape[1], values, 2)
                
            return values
            
        else:
            return eva_2d.evaluate_d_d(self.t[0], self.t[1], self.p[0] - 1, self.p[1] - 1, self.NbaseD[0], self.NbaseD[1], coeff, eta1, eta2)
    
        
        
    # =================================================
    def evaluate_NNN(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (NNN) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point
            
        eta2 : double or np.ndarray
            2nd component of logical evaluation point
            
        eta3 : double or np.ndarray
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        if coeff.ndim == 1:
            coeff = self.extract_0form(coeff)
        
        if isinstance(eta1, np.ndarray):
            
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0], eta3.shape[0]), dtype=float)
                
                eva_3d.evaluate_tensor_product(self.T[0], self.T[1], self.T[2], self.p[0], self.p[1], self.p[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], coeff, eta1, eta2, eta3, values, 0)
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                
                eva_3d.evaluate_matrix(self.T[0], self.T[1], self.T[2], self.p[0], self.p[1], self.p[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 0)
            
            return values
        
        else:
            return eva_3d.evaluate_n_n_n(self.T[0], self.T[1], self.T[2], self.p[0], self.p[1], self.p[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], coeff, eta1, eta2, eta3)
    
    
    # =================================================
    def evaluate_DNN(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (DNN) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point
            
        eta2 : double or np.ndarray
            2nd component of logical evaluation point
            
        eta3 : double or np.ndarray
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        if coeff.ndim == 1:
            coeff = self.extract_1form(coeff)[0]
        
        if isinstance(eta1, np.ndarray):
            
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0], eta3.shape[0]), dtype=float)
                
                eva_3d.evaluate_tensor_product(self.t[0], self.T[1], self.T[2], self.p[0] - 1, self.p[1], self.p[2], self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], coeff, eta1, eta2, eta3, values, 11)
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                
                eva_3d.evaluate_matrix(self.t[0], self.T[1], self.T[2], self.p[0] - 1, self.p[1], self.p[2], self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 11)
            
            return values
        
        else:
            return eva_3d.evaluate_d_n_n(self.t[0], self.T[1], self.T[2], self.p[0] - 1, self.p[1], self.p[2], self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], coeff, eta1, eta2, eta3)
    
    
    # =================================================
    def evaluate_NDN(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (NDN) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point
            
        eta2 : double or np.ndarray
            2nd component of logical evaluation point
            
        eta3 : double or np.ndarray
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        if coeff.ndim == 1:
            coeff = self.extract_1form(coeff)[1]
        
        if isinstance(eta1, np.ndarray):
            
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0], eta3.shape[0]), dtype=float)
                
                eva_3d.evaluate_tensor_product(self.T[0], self.t[1], self.T[2], self.p[0], self.p[1] - 1, self.p[2], self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], coeff, eta1, eta2, eta3, values, 12)
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                
                eva_3d.evaluate_matrix(self.T[0], self.t[1], self.T[2], self.p[0], self.p[1] - 1, self.p[2], self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 12)
            
            return values
        
        else:
            return eva_3d.evaluate_n_d_n(self.T[0], self.t[1], self.T[2], self.p[0], self.p[1] - 1, self.p[2], self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], coeff, eta1, eta2, eta3)
    
    
    # =================================================
    def evaluate_NND(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (NND) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point
            
        eta2 : double or np.ndarray
            2nd component of logical evaluation point
            
        eta3 : double or np.ndarray
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        if coeff.ndim == 1:
            coeff = self.extract_1form(coeff)[2]
        
        if isinstance(eta1, np.ndarray):
            
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0], eta3.shape[0]), dtype=float)
                
                eva_3d.evaluate_tensor_product(self.T[0], self.T[1], self.t[2], self.p[0], self.p[1], self.p[2] - 1, self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], coeff, eta1, eta2, eta3, values, 13)
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                
                eva_3d.evaluate_matrix(self.T[0], self.T[1], self.t[2], self.p[0], self.p[1], self.p[2] - 1, self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 13)
            
            return values
        
        else:
            return eva_3d.evaluate_n_n_d(self.T[0], self.T[1], self.t[2], self.p[0], self.p[1], self.p[2] - 1, self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], coeff, eta1, eta2, eta3)
    
    
    # =================================================
    def evaluate_NDD(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (NDD) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point
            
        eta2 : double or np.ndarray
            2nd component of logical evaluation point
            
        eta3 : double or np.ndarray
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        if coeff.ndim == 1:
            coeff = self.extract_2form(coeff)[0]
        
        if isinstance(eta1, np.ndarray):
            
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0], eta3.shape[0]), dtype=float)
                
                eva_3d.evaluate_tensor_product(self.T[0], self.t[1], self.t[2], self.p[0], self.p[1] - 1, self.p[2] - 1, self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], coeff, eta1, eta2, eta3, values, 21)
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                
                eva_3d.evaluate_matrix(self.T[0], self.t[1], self.t[2], self.p[0], self.p[1] - 1, self.p[2] - 1, self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 21)
            
            return values
        
        else:
            return eva_3d.evaluate_n_d_d(self.T[0], self.t[1], self.t[2], self.p[0], self.p[1] - 1, self.p[2] - 1, self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], coeff, eta1, eta2, eta3)
    
    
    # =================================================
    def evaluate_DND(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (DND) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point
            
        eta2 : double or np.ndarray
            2nd component of logical evaluation point
            
        eta3 : double or np.ndarray
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        if coeff.ndim == 1:
            coeff = self.extract_2form(coeff)[1]
        
        if isinstance(eta1, np.ndarray):
            
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0], eta3.shape[0]), dtype=float)
                
                eva_3d.evaluate_tensor_product(self.t[0], self.T[1], self.t[2], self.p[0] - 1, self.p[1], self.p[2] - 1, self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], coeff, eta1, eta2, eta3, values, 22)
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                
                eva_3d.evaluate_matrix(self.t[0], self.T[1], self.t[2], self.p[0] - 1, self.p[1], self.p[2] - 1, self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 22)
            
            return values
        
        else:
            return eva_3d.evaluate_d_n_d(self.t[0], self.T[1], self.t[2], self.p[0] - 1, self.p[1], self.p[2] - 1, self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], coeff, eta1, eta2, eta3)
    
    
    # =================================================
    def evaluate_DDN(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (DDN) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point
            
        eta2 : double or np.ndarray
            2nd component of logical evaluation point
            
        eta3 : double or np.ndarray
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        if coeff.ndim == 1:
            coeff = self.extract_2form(coeff)[2]
        
        if isinstance(eta1, np.ndarray):
            
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0], eta3.shape[0]), dtype=float)
                
                eva_3d.evaluate_tensor_product(self.t[0], self.t[1], self.T[2], self.p[0] - 1, self.p[1] - 1, self.p[2], self.NbaseD[0], self.NbaseD[1], self.NbaseN[2], coeff, eta1, eta2, eta3, values, 23)
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                
                eva_3d.evaluate_matrix(self.t[0], self.t[1], self.T[2], self.p[0] - 1, self.p[1] - 1, self.p[2], self.NbaseD[0], self.NbaseD[1], self.NbaseN[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 23)
            
            return values
        
        else:
            return eva_3d.evaluate_d_d_n(self.t[0], self.t[1], self.T[2], self.p[0] - 1, self.p[1] - 1, self.p[2], self.NbaseD[0], self.NbaseD[1], self.NbaseN[2], coeff, eta1, eta2, eta3)
    
    
    # =================================================
    def evaluate_DDD(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (DDD) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point
            
        eta2 : double or np.ndarray
            2nd component of logical evaluation point
            
        eta3 : double or np.ndarray
            3rd component of logical evaluation point
        
        coeff : array_like
            FEM coefficients
            
        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """
        
        if coeff.ndim == 1:
            coeff = self.extract_3form(coeff)
        
        if isinstance(eta1, np.ndarray):
            
            if eta1.ndim == 1:
                values = np.empty((eta1.shape[0], eta2.shape[0], eta3.shape[0]), dtype=float)
                
                eva_3d.evaluate_tensor_product(self.t[0], self.t[1], self.t[2], self.p[0] - 1, self.p[1] - 1, self.p[2] - 1, self.NbaseD[0], self.NbaseD[1], self.NbaseD[2], coeff, eta1, eta2, eta3, values, 3)
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                
                eva_3d.evaluate_matrix(self.t[0], self.t[1], self.t[2], self.p[0] - 1, self.p[1] - 1, self.p[2] - 1, self.NbaseD[0], self.NbaseD[1], self.NbaseD[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 3)
            
            return values
        
        else:
            return eva_3d.evaluate_d_d_d(self.t[0], self.t[1], self.t[2], self.p[0] - 1, self.p[1] - 1, self.p[2] - 1, self.NbaseD[0], self.NbaseD[1], self.NbaseD[2], coeff, eta1, eta2, eta3)