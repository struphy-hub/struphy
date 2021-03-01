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
        
    bc : boolean
        type of splines (True = periodic, False = clamped)
        
    n_quad : int
        optional: number of Gauss-Legendre quadrature points per element for integrations
    """
    
    def __init__(self, Nel, p, bc, n_quad=None):
        
        self.Nel     = Nel                                             # number of elements
        self.p       = p                                               # spline degree
        self.bc      = bc                                              # boundary conditions
        self.n_quad  = n_quad                                          # number of Gauss-Legendre quadrature points per element
        
        self.el_b    = np.linspace(0., 1., Nel + 1)                    # element boundaries
        self.delta   = 1/self.Nel                                      # element length
        
        self.T       = bsp.make_knots(self.el_b, self.p, self.bc)      # spline knot vector for B-splines (N)
        self.t       = self.T[1:-1]                                    # reduced knot vector for M-splines (D)
        
        self.NbaseN  = len(self.T) - self.p - 1 - self.bc*self.p       # total number of B-splines (N)
        self.NbaseD  = self.NbaseN - 1 + self.bc                       # total number of M-splines (D)
        
        
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
    def plot_splines(self, n_pts, kind='B-splines'):

        etaplot = np.linspace(0., 1., n_pts)

        if kind == 'B-splines':

            coeff = np.zeros(self.NbaseN, dtype=float)

            for i in range(self.NbaseN):
                coeff[:] = 0.
                coeff[i] = 1.
                plt.plot(etaplot, self.evaluate_N(etaplot, coeff))

        elif kind == 'M-splines':

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
        
        
        # number of basis functions of discrete tensor-product p-forms
        if len(self.spaces) == 3:
            
            self.Nbase_0form =  [self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]]

            self.Nbase_1form = [[self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]], 
                                [self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]], 
                                [self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]]]

            self.Nbase_2form = [[self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], 
                                [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], 
                                [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]]]

            self.Nbase_3form =  [self.NbaseD[0], self.NbaseD[1], self.NbaseD[2]]

            self.Ntot_0form  =  self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2] 

            self.Ntot_1form  = [self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], 
                                self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], 
                                self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]]

            self.Ntot_2form  = [self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2], 
                                self.NbaseD[0]*self.NbaseN[1]*self.NbaseD[2], 
                                self.NbaseD[0]*self.NbaseD[1]*self.NbaseN[2]]

            self.Ntot_3form  =  self.NbaseD[0]*self.NbaseD[1]*self.NbaseD[2]
            
            self.Ntot_1form_cum = [self.Ntot_1form[0], self.Ntot_1form[0] + self.Ntot_1form[1], self.Ntot_1form[0] + self.Ntot_1form[1] + self.Ntot_1form[2]]
            
            self.Ntot_2form_cum = [self.Ntot_2form[0], self.Ntot_2form[0] + self.Ntot_2form[1], self.Ntot_2form[0] + self.Ntot_2form[1] + self.Ntot_2form[2]]
        
        
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
    def assemble_M0(self, domain):
        self.M0 = mass_3d.get_M0(self, domain)
        
    # =================================================
    def assemble_M1(self, domain):
        self.M1 = mass_3d.get_M1(self, domain)
        
    # =================================================
    def assemble_M2(self, domain):
        self.M2 = mass_3d.get_M2(self, domain)
        
    # =================================================
    def assemble_M3(self, domain):
        self.M3 = mass_3d.get_M3(self, domain)
        
    # =================================================
    def assemble_Mv0(self, domain):
        self.Mv = mass_3d.get_Mv0(self, domain)
        
    # =================================================
    def assemble_Mv2(self, domain):
        self.Mv = mass_3d.get_Mv2(self, domain)
    
    
    
    # ================================================
    def ravel_pform(self, x1, x2, x3):
        return np.concatenate((x1.flatten(), x2.flatten(), x3.flatten()))
    
    
    # ================================================
    def unravel_0form(self, x):
        
        x1, x2, x3 = np.split(x, [self.Ntot_0form, 2*self.Ntot_0form])
        
        x1 = x1.reshape(self.Nbase_0form)
        x2 = x2.reshape(self.Nbase_0form)
        x3 = x3.reshape(self.Nbase_0form)
        
        return x1, x2, x3
    
    
    # ================================================
    def unravel_1form(self, x):
        
        x1, x2, x3 = np.split(x, [self.Ntot_1form_cum[0], self.Ntot_1form_cum[1]])
        
        x1 = x1.reshape(self.Nbase_1form[0])
        x2 = x2.reshape(self.Nbase_1form[1])
        x3 = x3.reshape(self.Nbase_1form[2])
        
        return x1, x2, x3
    
    
    # ================================================
    def unravel_2form(self, x):
        
        x1, x2, x3 = np.split(x, [self.Ntot_2form_cum[0], self.Ntot_2form_cum[1]])
        
        x1 = x1.reshape(self.Nbase_2form[0])
        x2 = x2.reshape(self.Nbase_2form[1])
        x3 = x3.reshape(self.Nbase_2form[2])
        
        return x1, x2, x3
        
    
    
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
    def set_derivatives(self, polar_splines=None):
        
        derivatives = der.discrete_derivatives_3D(self, polar_splines)
        
        self.GRAD   = derivatives.GRAD
        self.CURL   = derivatives.CURL
        self.DIV    = derivatives.DIV
    
    # =================================================
    def set_extraction_operators(self, polar_splines=None):
        
        if polar_splines == None:
            
            self.polar = False
            
            # 2D number of basis functions
            self.Nbase0_pol = self.NbaseN[0]*self.NbaseN[1]
            self.Nbase1_pol = self.NbaseD[0]*self.NbaseN[1] + self.NbaseN[0]*self.NbaseD[1]
            self.Nbase2_pol = self.NbaseN[0]*self.NbaseD[1] + self.NbaseD[0]*self.NbaseN[1]
            self.Nbase3_pol = self.NbaseD[0]*self.NbaseD[1]
            
            # 2D operators
            self.E0_pol = spa.identity(self.Nbase0_pol, dtype=float, format='csr')
            self.E1_pol = spa.identity(self.Nbase1_pol, dtype=float, format='csr')
            self.E2_pol = spa.identity(self.Nbase2_pol, dtype=float, format='csr')
            self.E3_pol = spa.identity(self.Nbase3_pol, dtype=float, format='csr')
            
            # 3D operators
            self.E0     = spa.identity(    self.Ntot_0form , dtype=float, format='csr')
            self.E1     = spa.identity(sum(self.Ntot_1form), dtype=float, format='csr')
            self.E2     = spa.identity(sum(self.Ntot_2form), dtype=float, format='csr')
            self.E3     = spa.identity(    self.Ntot_3form , dtype=float, format='csr')
            
        else:
            
            self.polar = False
            
            # 2D number of basis functions
            self.Nbase0_pol = polar_splines.Nbase0_pol
            self.Nbase1_pol = polar_splines.Nbase1_pol
            self.Nbase2_pol = polar_splines.Nbase2_pol
            self.Nbase3_pol = polar_splines.Nbase3_pol
            
            # 2D operators
            self.E0_pol = polar_splines.E0_pol
            self.E1_pol = polar_splines.E1_pol
            self.E2_pol = polar_splines.E2_pol
            self.E3_pol = polar_splines.E3_pol
            
            # 3D operators
            self.E0 = polar_splines.E0
            self.E1 = polar_splines.E1
            self.E2 = polar_splines.E2
            self.E3 = polar_splines.E3
            
    
    # =================================================
    def apply_bc_2form(self, coeff, bc):
        
        if self.bc[0] == False:
        
            # eta1 = 0
            if bc[0] == 'dirichlet':
                coeff[:self.NbaseD[1]*self.NbaseD[2]] = 0.
            
            # eta1 = 1
            if bc[1] == 'dirichlet':
                if self.polar == False:
                    coeff[(self.NbaseN[0] - 1)*self.NbaseD[1]*self.NbaseD[2]:self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2]] = 0.
                else:
                    coeff[2*self.NbaseD[2] + (self.NbaseN[0] - 3)*self.NbaseD[1]*self.NbaseD[2]:2*self.NbaseD[2] + (self.NbaseN[0] - 2)*self.NbaseD[1]*self.NbaseD[2]] = 0.
        
        

    
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