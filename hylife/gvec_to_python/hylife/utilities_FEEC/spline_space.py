# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic modules to create tensor-product finite element spaces of univariate B-splines.
"""


import numpy        as np
import scipy.sparse as spa

import matplotlib.pyplot as plt

import gvec_to_python.hylife.utilities_FEEC.bsplines         as bsp
import gvec_to_python.hylife.utilities_FEEC.bsplines_kernels as bsp_kernels

import gvec_to_python.hylife.utilities_FEEC.basics.spline_evaluation_1d as eva_1d
# import gvec_to_python.hylife.utilities_FEEC.basics.spline_evaluation_2d as eva_2d
# import gvec_to_python.hylife.utilities_FEEC.basics.spline_evaluation_3d as eva_3d

# import gvec_to_python.hylife.utilities_FEEC.basics.mass_matrices_1d as mass_1d
# import gvec_to_python.hylife.utilities_FEEC.basics.mass_matrices_2d as mass_2d
# import gvec_to_python.hylife.utilities_FEEC.basics.mass_matrices_3d as mass_3d

import gvec_to_python.hylife.utilities_FEEC.derivatives.derivatives as der



# from enum import Enum, unique
# @unique # Require Enum values to be unique.
# class Kind(Enum):
#     N  = 1
#     D  = 2
#     dN = 3



# =============== 1d B-spline space ======================
class spline_space_1d:
    """
    Defines a 1d space of B-splines.

    Parameters
    ----------
    T : array_like
        Spline knot vector.

    p : int
        Spline degree.

    spl_kind : boolean
        Kind of spline space (True = periodic, False = clamped).
        Formerly `bc`.

    n_quad : int
        Number of Gauss-Legendre quadrature points per grid cell (defined by break points).

    bc : [str, str]
        Boundary conditions at eta=0.0 and eta=1.0, 'f' free, 'd' dirichlet (remove boundary spline).
    """

    def __init__(self, T, p, spl_kind, n_quad=6, bc=['f', 'f']):

        self.T        = T                                                # spline knot vector for B-splines (N)
        self.p        = p                                                # spline degree
        self.spl_kind = spl_kind                                         # kind of spline space (periodic or clamped)

        # boundary conditions at eta=0. and eta=1. in case of clamped splines
        if spl_kind:
            self.bc   = [None, None]
        else:
            self.bc   = bc

        self.el_b     = bsp.breakpoints(self.T, self.p)                  # element boundaries
        self.Nel      = len(self.el_b) - 1                               # number of elements
        self.delta    = 1/self.Nel                                       # element length
        self.t        = self.T[1:-1]                                     # reduced knot vector for M-splines (D)

        # New code that limits spline construction from `np.linspace()`. 
        # Disabled.
        # self.el_b   = np.linspace(0., 1., Nel + 1)                     # element boundaries
        # self.T      = bsp.make_knots(self.el_b, self.p, self.spl_kind) # spline knot vector for B-splines (N)
        # self.t      = self.T[1:-1]                                     # spline knot vector for M-splines (D)

        self.greville = bsp.greville(self.T, self.p, self.spl_kind)      # greville points

        self.NbaseN   = len(self.T) - self.p - 1 - self.spl_kind*self.p  # total number of B-splines basis functions (N)
        self.NbaseD   = self.NbaseN - 1 + self.spl_kind                  # total number of M-splines basis functions (D)
        # print('spline_space_1d.__init__()')
        # print('type(self.T), type(self.t):', type(self.T), type(self.t))
        # print('type(self.NbaseN), type(self.NbaseD):', type(self.NbaseN), type(self.NbaseD))

        # global indices of non-vanishing splines in each element in format (Nel, global index)
        self.indN     = (np.indices((self.Nel, self.p + 1 - 0))[1] + np.arange(self.Nel)[:, None])%self.NbaseN
        self.indD     = (np.indices((self.Nel, self.p + 1 - 1))[1] + np.arange(self.Nel)[:, None])%self.NbaseD

        self.n_quad  = n_quad  # number of Gauss-Legendre points per grid cell (defined by break points)

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

        # print('Spline space set up (1d) done.')



    # =================================================
    def evaluate_N(self, eta, coeff, kind=0):
        """
        Evaluates the spline space (N) at the point(s) eta for given coefficients coeff.

        Parameters
        ----------
        eta : double or np.ndarray
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

        # if coeff.size == self.E0.shape[0]:
        #     coeff = self.E0.T.dot(coeff)

        # `eva_1d.evaluate_n()`: Evaluate a single point `eta`.
        # `eva_1d.evaluate_vector(kind=0)`: Just a for-loop of `eva_1d.evaluate_n()`!
        if isinstance(eta, np.ndarray):

            if eta.ndim == 1:

                values = np.empty(eta.size, dtype=float)
                eva_1d.evaluate_vector(self.T, self.p, self.NbaseN, coeff, eta, values, kind=kind)

            else:

                raise NotImplementedError('`eta` must be a 1D numpy array! Sparse/Dense meshgrids not supported. Flatten them yourself.')

            return values

        else:

            return eva_1d.evaluate_n(self.T, self.p, self.NbaseN, coeff, eta)



    # =================================================
    def evaluate_D(self, eta, coeff):
        """
        Evaluates the spline space (D) at the point(s) eta for given coefficients coeff.

        Parameters
        ----------
        eta : double or np.ndarray
            evaluation point(s)
        
        coeff : array_like
            FEM coefficients

        Returns
        -------
        value : double or array_like
            evaluated FEM field at the point(s) eta
        """

        assert (coeff.size == self.E1.shape[0])

        # `eva_1d.evaluate_d()`: Evaluate a single point `eta`.
        # `eva_1d.evaluate_vector(kind=1)`: Just a for-loop of `eva_1d.evaluate_d()`!
        if isinstance(eta, np.ndarray):

            if eta.ndim == 1:

                values = np.empty(eta.size, dtype=float)
                eva_1d.evaluate_vector(self.t, self.p - 1, self.NbaseD, coeff, eta, values, kind=1)

            else:

                raise NotImplementedError('`eta` must be a 1D numpy array! Sparse/Dense meshgrids not supported. Flatten them yourself.')

            return values

        else:

            return eva_1d.evaluate_d(self.t, self.p - 1, self.NbaseD, coeff, eta)



    # =================================================
    def evaluate_dN(self, eta, coeff):
        """
        Evaluates the dervivative of the spline space (N) at the point(s) eta for the coefficients coeff.

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

        # `eva_1d.evaluate_diffn()`: Evaluate a single point `eta`.
        # `eva_1d.evaluate_vector(kind=2)`: Just a for-loop of `eva_1d.evaluate_diffn()`!
        if isinstance(eta, np.ndarray):

            if eta.ndim == 1:

                values = np.empty(eta.size, dtype=float)
                eva_1d.evaluate_vector(self.T, self.p, self.NbaseN, coeff, eta, values, kind=2)

            else:

                raise NotImplementedError('`eta` must be a 1D numpy array! Sparse/Dense meshgrids not supported. Flatten them yourself.')

            return values

        else:

            return eva_1d.evaluate_diffn(self.T, self.p, self.NbaseN, coeff, eta)



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
            print('only B-splines and M-splines available')

        plt.plot(self.greville, np.zeros(self.greville.shape), 'ro', label='greville')
        plt.plot(self.el_b, np.zeros(self.el_b.shape), 'k+', label='breaks')
        plt.title(which + ', spl_kind=' + str(self.spl_kind) + ', p={0:2d}, Nel={1:4d}'.format(self.p, self.Nel))
        plt.legend()



# =============== multi-d B-spline tensor product space ======================        
# Entire class deleted, because GVEC has only one 1D spline space.
