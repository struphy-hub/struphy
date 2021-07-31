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

import hylife.utilitis_FEEC.projectors.projectors_global as pro

import hylife.utilitis_FEEC.derivatives.derivatives as der

import hylife.geometry.polar_splines as pol


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
        number of Gauss-Legendre quadrature points per grid cell (defined by break points)

    bc : [str, str]
        boundary conditions at eta=0.0 and eta=1.0, 'f' free, 'd' dirichlet (remove boundary spline)
    """
    
    def __init__(self, Nel, p, spl_kind, n_quad=6, bc=['f', 'f']):
        
        self.Nel      = Nel                                              # number of elements
        self.p        = p                                                # spline degree
        self.spl_kind = spl_kind                                         # kind of spline space (periodic or clamped)
        
        # boundary conditions at eta=0. and eta=1. in case of clamped splines
        if spl_kind:
            self.bc   = [None, None]                                     
        else:
            self.bc   = bc                                               

        self.el_b     = np.linspace(0., 1., Nel + 1)                     # element boundaries
        self.delta    = 1/self.Nel                                       # element length
         
        self.T        = bsp.make_knots(self.el_b, self.p, self.spl_kind) # spline knot vector for B-splines (N)
        self.t        = self.T[1:-1]                                     # spline knot vector for M-splines (D)

        self.greville = bsp.greville(self.T, self.p, self.spl_kind)      # greville points
        
        self.NbaseN   = len(self.T) - self.p - 1 - self.spl_kind*self.p  # total number of B-splines (N)
        self.NbaseD   = self.NbaseN - 1 + self.spl_kind                  # total number of M-splines (D)
        
        # global indices of non-vanishing splines in each element in format (Nel, global index)
        self.indN     = (np.indices((self.Nel, self.p + 1 - 0))[1] + np.arange(self.Nel)[:, None])%self.NbaseN
        self.indD     = (np.indices((self.Nel, self.p + 1 - 1))[1] + np.arange(self.Nel)[:, None])%self.NbaseD
            
        self.n_quad   = n_quad  # number of Gauss-Legendre points per grid cell (defined by break points)
        
        self.pts_loc  = np.polynomial.legendre.leggauss(self.n_quad)[0] # Gauss-Legendre points  (GLQP) in (-1, 1)
        self.wts_loc  = np.polynomial.legendre.leggauss(self.n_quad)[1] # Gauss-Legendre weights (GLQW) in (-1, 1)
    
        # global GLQP in format (element, local point) and total number of GLQP
        self.pts      = bsp.quadrature_grid(self.el_b, self.pts_loc, self.wts_loc)[0]
        self.n_pts    = self.pts.flatten().size

        # global GLQW in format (element, local point)
        self.wts      = bsp.quadrature_grid(self.el_b, self.pts_loc, self.wts_loc)[1]

        # basis functions evaluated at quadrature points in format (element, local basis function, derivative, local point)
        self.basisN   = bsp.basis_ders_on_quad_grid(self.T, self.p    , self.pts, 0, normalize=False)
        self.basisD   = bsp.basis_ders_on_quad_grid(self.t, self.p - 1, self.pts, 0, normalize=True)
        
        # -------------------------------------------------
        # Set extraction operators for boundary conditions:
        # -------------------------------------------------
        n1 = self.NbaseN
        d1 = self.NbaseD

        # bc = 'f': including boundary splines
        self.E0_all = spa.identity(n1, dtype=float, format='csr')
        self.E1_all = spa.identity(d1, dtype=float, format='csr')

        # bc = 'd': without boundary splines
        E_NN = self.E0_all.copy()
        E_DD = self.E1_all.copy()

        # remove contributions from N-splines at eta = 0
        if self.bc[0] == 'd':
            E_NN = E_NN[1:, :]

        # remove contributions from N-splines at eta = 1
        if self.bc[1] == 'd': 
            E_NN = E_NN[:-1, :]

        self.E0 = E_NN.tocsr().copy()
        self.E1 = E_DD.tocsr().copy()
        
        
        # -------------------------------------------------
        # Set discrete derivatives:
        # -------------------------------------------------
        self.G = der.discrete_derivatives_1d(self)

        print('Spline space set up (1d) done.')
                    

    # functions for setting mass matrices:        
    # =================================================
    def assemble_M0(self, weight=None):
        self.M0  = self.E0.dot(mass_1d.get_M(self, 0, 0, weight).dot(self.E0.T))
        print('Assembly of M0 (1d) done.')
        
    # =================================================
    def assemble_M1(self, weight=None):
        self.M1  = self.E1.dot(mass_1d.get_M(self, 1, 1, weight).dot(self.E1.T))
        print('Assembly of M1 (1d) done.')
        
    # =================================================
    def assemble_M01(self, weight=None):
        self.M01 = self.E0.dot(mass_1d.get_M(self, 0, 1, weight).dot(self.E1.T))
        print('Assembly of M01 (1d) done.')
        
    # =================================================
    def assemble_M10(self, weight=None):
        self.M10 = self.E1.dot(mass_1d.get_M(self, 1, 0, weight).dot(self.E0.T))
        print('Assembly of M10 (1d) done.')

    
    # functions for setting projectors:        
    # =================================================
    def set_projectors(self, nq=6):
        self.projectors = pro.projectors_global_1d(self, nq)
        print('Set projectors (1d) done.')

    
    # spline evaluation and plotting:
    # =================================================
    def evaluate_N(self, eta, coeff, kind=0):
        """
        Evaluates the spline space (N) at the point(s) eta for given coefficients coeff.

        Parameters
        ----------
        eta : double or array_like
            evaluation point(s)
        
        coeff : array_like
            FEM coefficients
            
        kind : int
            kind of evaluation (0 : spline space, 2 : derivative of spline space)

        Returns
        -------
        value : double or array_like
            evaluated FEM field at the point(s) eta
        """
        
        assert (coeff.size == self.E0.shape[0]) or (coeff.size == self.E0.shape[1])
        assert (kind == 0) or (kind == 2)
        
        if coeff.size == self.E0.shape[0]:
            coeff = self.E0.T.dot(coeff)
            
        if isinstance(eta, float):
            pts = np.array([eta])
        elif isinstance(eta, np.ndarray):
            pts = eta.flatten()
            
        values = np.empty(pts.size, dtype=float)
        eva_1d.evaluate_vector(self.T, self.p, self.NbaseN, coeff, pts, values, kind)
        
        if isinstance(eta, float):
            values = values[0]
        elif isinstance(eta, np.ndarray):
            values = values.reshape(eta.shape)
            
        return values
        
        
        
    # =================================================
    def evaluate_D(self, eta, coeff):
        """
        Evaluates the spline space (D) at the point(s) eta for given coefficients coeff.

        Parameters
        ----------
        eta : double or array_like
            evaluation point(s)
        
        coeff : array_like
            FEM coefficients

        Returns
        -------
        value : double or array_like
            evaluated FEM field at the point(s) eta
        """
        
        assert (coeff.size == self.E1.shape[0])
        
        if isinstance(eta, float):
            pts = np.array([eta])
        elif isinstance(eta, np.ndarray):
            pts = eta.flatten()
            
        values = np.empty(pts.size, dtype=float)
        eva_1d.evaluate_vector(self.t, self.p - 1, self.NbaseD, coeff, pts, values, 1)
        
        if isinstance(eta, float):
            values = values[0]
        elif isinstance(eta, np.ndarray):
            values = values.reshape(eta.shape)
            
        return values
    
    
    
    # =================================================
    def plot_splines(self, n_pts=500, which='B-splines'):
        """
        Plots all basis functions.
        
        Parameters
        ----------
        n_pts : int
            number of points for plotting (optinal, default=500)
            
        which : string
            which basis to plot. B-splines (N) or M-splines (D) (optional, default='B-splines')
        """

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
            print('Only B-splines and M-splines available')

        plt.plot(self.greville, np.zeros(self.greville.shape), 'ro', label='greville')
        plt.plot(self.el_b, np.zeros(self.el_b.shape), 'k+', label='breaks')
        plt.title(which + ', spl_kind=' + str(self.spl_kind) + ', p={0:2d}, Nel={1:4d}'.format(self.p, self.Nel))
        plt.legend()
            
        
        
# =============== multi-d B-spline tensor product space ======================        
class tensor_spline_space:
    """
    Defines a tensor product space of 1d B-spline spaces in higher dimensions (2d and 3d).
    
    Parameters
    ----------
    spline_spaces : list of spline_space_1d
        1d B-spline spaces from which the tensor_product B-spline space is built
        
    n_tor : int
        mode number in third dimension if two spline spaces are passed
    """
    

    def __init__(self, spline_spaces, n_tor=0):
        
        self.spaces   = spline_spaces                            # 1D B-spline spaces
        self.dim      = len(self.spaces)                         # number of 1D B-spline spaces

        # set mode number in third dimension (only for 2D space)
        if self.dim == 2:
            self.n_tor = n_tor
        
        # polar splines can be set below
        self.polar    = False
        
        # input from 1d spaces
        # ====================
        self.Nel      = [spl.Nel      for spl in self.spaces]    # number of elements
        self.p        = [spl.p        for spl in self.spaces]    # spline degree
        self.spl_kind = [spl.spl_kind for spl in self.spaces]    # kind of spline space (periodic or clamped)
        
        self.bc       = [spl.bc       for spl in self.spaces]    # boundary conditions at eta = 0 and eta = 1
        
        self.el_b     = [spl.el_b     for spl in self.spaces]    # element boundaries
        self.delta    = [spl.delta    for spl in self.spaces]    # element lengths
        
        self.T        = [spl.T        for spl in self.spaces]    # spline knot vector for B-splines (N)
        self.t        = [spl.t        for spl in self.spaces]    # spline knot vector for M-splines (D)
        
        self.NbaseN   = [spl.NbaseN   for spl in self.spaces]    # total number of B-splines (N)
        self.NbaseD   = [spl.NbaseD   for spl in self.spaces]    # total number of M-splines (D)
        
        self.indN     = [spl.indN     for spl in self.spaces]    # global indices of non-vanishing B-splines (N) per element
        self.indD     = [spl.indD     for spl in self.spaces]    # global indices of non-vanishing M-splines (D) per element
            
        self.n_quad   = [spl.n_quad   for spl in self.spaces]    # number of Gauss-Legendre quadrature points per element
        
        self.pts_loc  = [spl.pts_loc  for spl in self.spaces]    # Gauss-Legendre quadrature points  (GLQP) in (-1, 1)
        self.wts_loc  = [spl.wts_loc  for spl in self.spaces]    # Gauss-Legendre quadrature weights (GLQW) in (-1, 1)

        self.pts      = [spl.pts      for spl in self.spaces]    # global GLQP in format (element, local point)
        self.wts      = [spl.wts      for spl in self.spaces]    # global GLQW in format (element, local weight)
        
        self.n_pts    = [spl.n_pts    for spl in self.spaces]    # total number of quadrature points

        # basis functions evaluated at quadrature points in format (element, local basis function, derivative, local point)
        self.basisN   = [spl.basisN   for spl in self.spaces] 
        self.basisD   = [spl.basisD   for spl in self.spaces]
        
        # number of basis functions of discrete tensor-product p-forms in 2D x analytical third dimension
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
        
        # extension to 3D
        if self.dim == 3:
            
            self.Nbase_0form    += [self.NbaseN[2]]

            self.Nbase_1form[0] += [self.NbaseN[2]]
            self.Nbase_1form[1] += [self.NbaseN[2]]
            self.Nbase_1form[2] += [self.NbaseD[2]]
            
            self.Nbase_2form[0] += [self.NbaseD[2]]
            self.Nbase_2form[1] += [self.NbaseD[2]]
            self.Nbase_2form[2] += [self.NbaseN[2]]

            self.Nbase_3form    += [self.NbaseD[2]]

            # total number of basis functions
            self.Ntot_0form     *= self.NbaseN[2] 

            self.Ntot_1form[0]  *= self.NbaseN[2]
            self.Ntot_1form[1]  *= self.NbaseN[2]
            self.Ntot_1form[2]  *= self.NbaseD[2]

            self.Ntot_2form[0]  *= self.NbaseD[2]
            self.Ntot_2form[1]  *= self.NbaseD[2]
            self.Ntot_2form[2]  *= self.NbaseN[2]

            self.Ntot_3form     *= self.NbaseD[2]
            
        # cumulative number of basis functions for vector-valued spaces
        self.Ntot_1form_cum = [self.Ntot_1form[0],
                               self.Ntot_1form[0] + self.Ntot_1form[1], 
                               self.Ntot_1form[0] + self.Ntot_1form[1] + self.Ntot_1form[2]]

        self.Ntot_2form_cum = [self.Ntot_2form[0],
                               self.Ntot_2form[0] + self.Ntot_2form[1],
                               self.Ntot_2form[0] + self.Ntot_2form[1] + self.Ntot_2form[2]]

        #print('Tensor space set up ({}d) done.'.format(self.dim))


        # -------------------------------------------------
        # Set extraction operators for boundary conditions:
        # -------------------------------------------------
        n1, n2 = self.NbaseN[:2]
        d1, d2 = self.NbaseD[:2]
        
        # including boundary splines
        self.E0_pol_all = spa.identity(n1*n2        , dtype=float, format='csr')
        self.E1_pol_all = spa.identity(d1*n2 + n1*d2, dtype=float, format='csr')
        self.E2_pol_all = spa.identity(n1*d2 + d1*n2, dtype=float, format='csr')
        self.E3_pol_all = spa.identity(d1*d2        , dtype=float, format='csr')
        
        # without boundary splines
        E_NN = spa.identity(n1*n2, format='csr')
        E_DN = spa.identity(d1*n2, format='csr')
        E_ND = spa.identity(n1*d2, format='csr')
        E_DD = spa.identity(d1*d2, format='csr')
        
        # remove contributions from N-splines at eta_1 = 0
        if self.bc[0][0] == 'd':
            E_NN = E_NN[n2:, :]
            E_ND = E_ND[d2:, :]

        # remove contributions from N-splines at eta_1 = 1
        if self.bc[0][1] == 'd':
            E_NN = E_NN[:-n2, :]
            E_ND = E_ND[:-d2, :]
            
        self.E0_pol = E_NN.tocsr().copy()
        self.E1_pol = spa.bmat([[E_DN, None], [None, E_ND]], format='csr')
        self.E2_pol = spa.bmat([[E_ND, None], [None, E_DN]], format='csr')
        self.E3_pol = E_DD.tocsr().copy()
        
        
        self.E0_all =            self.E0_pol_all.copy()
        self.E1_all = spa.bmat([[self.E1_pol_all, None], [None, self.E0_pol_all]], format='csr')
        self.E2_all = spa.bmat([[self.E2_pol_all, None], [None, self.E3_pol_all]], format='csr')
        self.E3_all =            self.E3_pol_all.copy()

        self.E0     =            self.E0_pol.copy()
        self.E1     = spa.bmat([[self.E1_pol, None], [None, self.E0_pol]], format='csr')
        self.E2     = spa.bmat([[self.E2_pol, None], [None, self.E3_pol]], format='csr')
        self.E3     =            self.E3_pol.copy()
 
        # 3D extraction operators    
        if self.dim == 3:
            
            n3 = self.NbaseN[2]
            d3 = self.NbaseD[2]
            
            # Kronecker product with third dimension (including boundary splines)
            self.E0_all = spa.kron(self.E0_pol_all, spa.identity(n3), format='csr')
            
            E1_1_all    = spa.kron(self.E1_pol_all, spa.identity(n3), format='csr')
            E1_3_all    = spa.kron(self.E0_pol_all, spa.identity(d3), format='csr')
            self.E1_all = spa.bmat([[E1_1_all, None], [None, E1_3_all]], format='csr')
            
            E2_1_all    = spa.kron(self.E2_pol_all, spa.identity(d3), format='csr')
            E2_3_all    = spa.kron(self.E3_pol_all, spa.identity(n3), format='csr')
            self.E2_all = spa.bmat([[E2_1_all, None], [None, E2_3_all]], format='csr')   
            
            self.E3_all = spa.kron(self.E3_pol_all, spa.identity(d3), format='csr')
            
            # Kronecker product with third dimension (without boundary splines)
            self.E0     = spa.kron(self.E0_pol, spa.identity(n3), format='csr')
            
            E1_1        = spa.kron(self.E1_pol, spa.identity(n3), format='csr')
            E1_3        = spa.kron(self.E0_pol, spa.identity(d3), format='csr')
            self.E1     = spa.bmat([[E1_1, None], [None, E1_3]], format='csr')
            
            E2_1        = spa.kron(self.E2_pol, spa.identity(d3), format='csr')
            E2_3        = spa.kron(self.E3_pol, spa.identity(n3), format='csr')
            self.E2     = spa.bmat([[E2_1, None], [None, E2_3]], format='csr')   
            
            self.E3     = spa.kron(self.E3_pol, spa.identity(d3), format='csr')

        elif self.dim > 3:     
            raise NotImplementedError('Only 2d and 3d supported.')
            
            
        # -------------------------------------------------
        # Set discrete derivatives:
        # -------------------------------------------------
        self.G, self.C, self.D = der.discrete_derivatives_3d(self)
                

        print('Set extraction operators for boundary conditions ({}d) done.'.format(self.dim))

    
    # function for setting projectors:        
    # =================================================
    def set_projectors(self, which='tensor', nq=[6, 6]):
        
        if   which == 'tensor':
        
            if self.dim==2:
                self.projectors = pro.projectors_tensor_2d([space.projectors for space in self.spaces])
            elif self.dim==3:
                self.projectors = pro.projectors_tensor_3d([space.projectors for space in self.spaces])
            else:
                raise NotImplementedError('Only 2d and 3d supported.')
                
        elif which == 'general':
            
            if self.dim==2:
                self.projectors = pro.projectors_global_2d(self, nq)
            elif self.dim==3:
                self.projectors = pro.projectors_global_3d(self, nq)
            else:
                raise NotImplementedError('Only 2d and 3d supported.')

        print('Set projectors ({}d) done.'.format(self.dim))


    # function for setting polar splines:
    # ===================================
    def set_polar_splines(self, cx, cy):
        
        self.polar_splines = pol.polar_splines_2D(cx, cy)
        
        n1, n2 = self.NbaseN[:2]
        d1, d2 = self.NbaseD[:2]
        
        # including boundary splines
        self.E0_pol_all = self.polar_splines.E0.copy()
        self.E1_pol_all = self.polar_splines.E1C.copy()
        self.E2_pol_all = self.polar_splines.E1D.copy()
        self.E3_pol_all = self.polar_splines.E2.copy()
        
        # without boundary splines
        E0_NN = self.polar_splines.E0.copy()
            
        E1_DN = self.polar_splines.E1C.copy()[:(0 + (d1 - 1)*d2) , :]
        E1_ND = self.polar_splines.E1C.copy()[ (0 + (d1 - 1)*d2):, :]

        E2_ND = self.polar_splines.E1D.copy()[:(2 + (n1 - 2)*d2) , :]
        E2_DN = self.polar_splines.E1D.copy()[ (2 + (n1 - 2)*d2):, :]

        E3_DD = self.polar_splines.E2.copy()
        
        # remove contributions from N-splines at eta_1 = 1
        if self.bc[0][1] == 'd':
            E0_NN = E0_NN[:-n2, :]
            E1_ND = E1_ND[:-d2, :]
            E2_ND = E2_ND[:-d2, :]
            
        self.E0_pol = E0_NN.tocsr().copy()
        self.E1_pol = spa.bmat([[E1_DN], [E1_ND]], format='csr')
        self.E2_pol = spa.bmat([[E2_ND], [E2_DN]], format='csr')
        self.E3_pol = E3_DD.tocsr().copy()

        self.E0_all =            self.E0_pol_all.copy()
        self.E1_all = spa.bmat([[self.E1_pol_all, None], [None, self.E0_pol_all]], format='csr')
        self.E2_all = spa.bmat([[self.E2_pol_all, None], [None, self.E3_pol_all]], format='csr')
        self.E3_all =            self.E3_pol_all.copy()

        self.E0     =            self.E0_pol.copy()
        self.E1     = spa.bmat([[self.E1_pol, None], [None, self.E0_pol]], format='csr')
        self.E2     = spa.bmat([[self.E2_pol, None], [None, self.E3_pol]], format='csr')
        self.E3     =            self.E3_pol.copy()

        # 3D extraction operators
        if self.dim == 3:

            n3 = self.NbaseN[2]
            d3 = self.NbaseD[2]
            
            # Kronecker product with third dimension (including boundary splines)
            self.E0_all = spa.kron(self.E0_pol_all, spa.identity(n3), format='csr')
            
            E1_all_1    = spa.kron(self.E1_pol_all, spa.identity(n3), format='csr')
            E1_all_3    = spa.kron(self.E0_pol_all, spa.identity(d3), format='csr')
            self.E1_all = spa.bmat([[E1_all_1, None], [None, E1_all_3]], format='csr')
            
            E2_all_1    = spa.kron(self.E2_pol_all, spa.identity(d3), format='csr')
            E2_all_3    = spa.kron(self.E3_pol_all, spa.identity(n3), format='csr')
            self.E2_all = spa.bmat([[E2_all_1, None], [None, E2_all_3]], format='csr')
            
            self.E3_all = spa.kron(self.E3_pol_all, spa.identity(d3), format='csr')

            # Kronecker product with third dimension (without boundary splines)
            self.E0     = spa.kron(self.E0_pol, spa.identity(n3), format='csr') 
            
            E1_1        = spa.kron(self.E1_pol, spa.identity(n3), format='csr')
            E1_3        = spa.kron(self.E0_pol, spa.identity(d3), format='csr')
            self.E1     = spa.bmat([[E1_1, None], [None, E1_3]], format='csr')
            
            E2_1        = spa.kron(self.E2_pol, spa.identity(d3), format='csr')
            E2_3        = spa.kron(self.E3_pol, spa.identity(n3), format='csr')
            self.E2     = spa.bmat([[E2_1, None], [None, E2_3]], format='csr')
            
            self.E3     = spa.kron(self.E3_pol, spa.identity(d3), format='csr')

        elif self.dim > 3:     
            raise NotImplementedError('Only 2d and 3d supported.')

        self.polar = True
        
        self.G, self.C, self.D = der.discrete_derivatives_3d(self)

        print('Set polar splines ({}d) done.'.format(self.dim))

    
    # ============== mass matrices (2D) ===============
    def assemble_M0_2D(self, domain, weight=None):
        self.M0 = mass_2d.get_M0(self, domain, weight)

    def assemble_M1_2D(self, domain, weight=None):
        self.M1 = mass_2d.get_M1(self, domain, weight)

    def assemble_M2_2D(self, domain, weight=None):
        self.M2 = mass_2d.get_M2(self, domain, weight)

    def assemble_M3_2D(self, domain, weight=None):
        self.M3 = mass_2d.get_M3(self, domain, weight)
        
    def assemble_Mv_2D(self, domain, weight=None):
        self.Mv = mass_2d.get_Mv(self, domain, weight)
        
    def assemble_M1_2D_blocks(self, domain, weight=None):
        self.M1_12, self.M1_33 = mass_2d.get_M1(self, domain, weight, True)
        
    def assemble_M2_2D_blocks(self, domain, weight=None):
        self.M2_12, self.M2_33 = mass_2d.get_M2(self, domain, weight, True)
        
    def assemble_Mv_2D_blocks(self, domain, weight=None):
        self.Mv_12, self.Mv_33 = mass_2d.get_Mv(self, domain, weight, True)
    
    # ============== mass matrices (3D) ===============
    def assemble_M0(self, domain, weight=None):
        self.M0 = mass_3d.get_M0(self, domain, weight)

    def assemble_M1(self, domain, weight=None):
        self.M1 = mass_3d.get_M1(self, domain, weight)

    def assemble_M2(self, domain, weight=None):
        self.M2 = mass_3d.get_M2(self, domain, weight)

    def assemble_M3(self, domain, weight=None):
        self.M3 = mass_3d.get_M3(self, domain, weight)

    def assemble_Mv(self, domain, weight=None):
        self.Mv = mass_3d.get_Mv(self, domain, weight)
    

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
    def evaluate_NN(self, eta1, eta2, coeff, which=None):
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
        
        # extract coefficients if flattened
        if coeff.ndim == 1:
            if   which == 'V0':
                coeff = self.extract_0form(coeff)
            elif which == 'V1':
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
    def evaluate_DN(self, eta1, eta2, coeff, which=None):
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
        
        # extract coefficients if flattened
        if coeff.ndim == 1:
            if   which == 'V1':
                coeff = self.extract_1form(coeff)[0]
            elif which == 'V2':
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
    def evaluate_ND(self, eta1, eta2, coeff, which=None):
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
        
        # extract coefficients if flattened
        if coeff.ndim == 1:
            if   which == 'V1':
                coeff = self.extract_1form(coeff)[1]
            elif which == 'V2':
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
    def evaluate_DD(self, eta1, eta2, coeff, which=None):
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
        
        # extract coefficients if flattened
        if coeff.ndim == 1:
            if   which == 'V2':
                coeff = self.extract_2form(coeff)[2]
            elif which == 'V3':
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

            # tensor-product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(self.T[0], self.T[1], self.T[2], self.p[0], self.p[1], self.p[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], coeff, eta1, eta2, eta3, values, 0)

            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(self.T[0], self.T[1], self.T[2], self.p[0], self.p[1], self.p[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 0)

                # `eta1` is a dense meshgrid. Process each point as default.
                else:
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
            
            # tensor product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(self.t[0], self.T[1], self.T[2], self.p[0] - 1, self.p[1], self.p[2], self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], coeff, eta1, eta2, eta3, values, 11)
            
            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(self.t[0], self.T[1], self.T[2], self.p[0] - 1, self.p[1], self.p[2], self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 11)
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
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
            
            # tensor product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(self.T[0], self.t[1], self.T[2], self.p[0], self.p[1] - 1, self.p[2], self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], coeff, eta1, eta2, eta3, values, 12)
            
            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(self.T[0], self.t[1], self.T[2], self.p[0], self.p[1] - 1, self.p[2], self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 12)
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
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
            
            # tensor product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(self.T[0], self.T[1], self.t[2], self.p[0], self.p[1], self.p[2] - 1, self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], coeff, eta1, eta2, eta3, values, 13)
            
            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(self.T[0], self.T[1], self.t[2], self.p[0], self.p[1], self.p[2] - 1, self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 13)
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
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
            
            # tensor product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(self.T[0], self.t[1], self.t[2], self.p[0], self.p[1] - 1, self.p[2] - 1, self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], coeff, eta1, eta2, eta3, values, 21)
            
            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(self.T[0], self.t[1], self.t[2], self.p[0], self.p[1] - 1, self.p[2] - 1, self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 21)
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
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
            
            # tensor product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(self.t[0], self.T[1], self.t[2], self.p[0] - 1, self.p[1], self.p[2] - 1, self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], coeff, eta1, eta2, eta3, values, 22)
            
            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(self.t[0], self.T[1], self.t[2], self.p[0] - 1, self.p[1], self.p[2] - 1, self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 22)
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
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
            
            # tensor product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(self.t[0], self.t[1], self.T[2], self.p[0] - 1, self.p[1] - 1, self.p[2], self.NbaseD[0], self.NbaseD[1], self.NbaseN[2], coeff, eta1, eta2, eta3, values, 23)
            
            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(self.t[0], self.t[1], self.T[2], self.p[0] - 1, self.p[1] - 1, self.p[2], self.NbaseD[0], self.NbaseD[1], self.NbaseN[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 23)
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
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
            
            # tensor product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(self.t[0], self.t[1], self.t[2], self.p[0] - 1, self.p[1] - 1, self.p[2] - 1, self.NbaseD[0], self.NbaseD[1], self.NbaseD[2], coeff, eta1, eta2, eta3, values, 3)
            
            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(self.t[0], self.t[1], self.t[2], self.p[0] - 1, self.p[1] - 1, self.p[2] - 1, self.NbaseD[0], self.NbaseD[1], self.NbaseD[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 3)
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
                    eva_3d.evaluate_matrix(self.t[0], self.t[1], self.t[2], self.p[0] - 1, self.p[1] - 1, self.p[2] - 1, self.NbaseD[0], self.NbaseD[1], self.NbaseD[2], coeff, eta1, eta2, eta3, eta1.shape[0], eta2.shape[1], eta3.shape[2], values, 3)
            
            return values
        
        else:
            return eva_3d.evaluate_d_d_d(self.t[0], self.t[1], self.t[2], self.p[0] - 1, self.p[1] - 1, self.p[2] - 1, self.NbaseD[0], self.NbaseD[1], self.NbaseD[2], coeff, eta1, eta2, eta3)