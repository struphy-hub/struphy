# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Classes for projectors in 1D, 2D and 3D based on global spline interpolation and histopolation.
"""

import numpy as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.bsplines as bsp

import hylife.utilitis_FEEC.projectors.kernels_projectors_global as ker_glob

from   hylife.linear_algebra.linalg_kron import kron_lusolve_2d
from   hylife.linear_algebra.linalg_kron import kron_lusolve_3d



# ======================= 1d ====================================
class projectors_global_1d:
    """
    Global commuting projectors pi_0 and pi_1 in 1d.
    
    Parameters:
    -----------
    spline_space : spline_space_1d
        A 1d space of B-splines and corresponding D-splines.
        
    n_quad : int
        Number of Gauss-Legendre quadrature points per integration interval for histopolation.

    Attributes:
    -----------
    space : spline_space_1d
        The input space.

    n_quad : int
        The input number of quadrature points.

    kind : str
        Kind of projector = 'global'.

    pts_loc : 1d array
        Gauss-Legendre quadrature points in (-1, 1).

    wts_loc : 1d array
        Gauss-Legendre quadrature weights in (-1, 1).

    x_int : 1d array
        Interpolation points = Greville points of space.

    x_his : 1d array
        Integration cell boundaries for histolpolation.

    subs : 1d array
        Number of integration intervals per cell to achieve exact integration of splines:
        subs[:]=1 for odd spline degree
        subs[:]=2 for even splines degree and periodic bc (some boundary changes are made for clamped)

    subs_cum : list
        Cumulative sum of subs, starting with 0.

    pts : 2d array
        Quadrature points in format (element, quad point) 

    wts : 2d array
        Quadrature weights in format (element, quad weight)

    N : sparse csr matrix
        Collocation matrix N_j(x_i).

    D : sparse csr matrix
        Histopolation matrix int_(x_i)^(x_i+1) D_j dx.

    N_LU : sparce csc matrix
        LU decompositions of N.

    D_LU : sparce csc matrix
        LU decompositions of D.

    N_T_LU : sparse csc matrix
        LU decompositions of transpose N.

    D_T_LU : sparse csc matrix
        LU decompositions of transpose D.

    Methods:
    --------
    dofs_0
    dofs_1
    pi_0 
    pi_1 
    pi_0_mat
    pi_1_mat
    bases_at_pts
    dofs_1d_bases
    dofs_1d_bases_products
    """
    
    def __init__(self, spline_space, n_quad=6):
        
        self.space  = spline_space     # 1D B-splines space
        self.n_quad = n_quad           # number of quadrature point per integration interval
        self.kind   = 'global'         # kind of projector (global vs. local)
        
        # Gauss - Legendre quadrature points and weights in (-1, 1)
        self.pts_loc = np.polynomial.legendre.leggauss(self.n_quad)[0]  
        self.wts_loc = np.polynomial.legendre.leggauss(self.n_quad)[1]
        
        # set interpolation points (Greville points)
        self.x_int = self.space.greville.copy()
        
        # set number of sub-intervals per integration interval between Greville points and integration boundaries
        self.subs = np.ones(self.space.NbaseD, dtype=int)
        self.x_his = np.array([self.x_int[0]])
            
        for i in range(self.space.NbaseD):
            for br in self.space.el_b:
                
                # left and right integration boundaries
                if self.space.spl_kind == False:
                    xl = self.x_int[i]
                    xr = self.x_int[i + 1]
                else:  
                    xl = self.x_int[i]
                    xr = self.x_int[(i + 1)%self.space.NbaseD]
                    if i == self.space.NbaseD - 1:
                        xr += self.space.el_b[-1]

                # compute subs and x_his
                if (br > xl + 1e-10) and (br < xr - 1e-10):
                    self.subs[i] += 1
                    self.x_his = np.append(self.x_his, br)
                elif br >= xr - 1e-10:
                    self.x_his = np.append(self.x_his, xr)
                    break
        
        if self.space.spl_kind == True and self.space.p%2 == 0:
            self.x_his = np.append(self.x_his, self.space.el_b[-1] + self.x_his[0])            
        
        # cumulative number of sub-intervals for conversion local interval --> global interval
        self.subs_cum = np.append(0, np.cumsum(self.subs - 1)[:-1])
        
        # quadrature points and weights
        self.pts, self.wts = bsp.quadrature_grid(self.x_his, self.pts_loc, self.wts_loc)
        self.pts           = self.pts%self.space.el_b[-1]
        
        # interpolation and histopolation_matrix
        self.N = bsp.collocation_matrix(  self.space.T, self.space.p, self.x_int, self.space.spl_kind)
        self.D = bsp.histopolation_matrix(self.space.T, self.space.p, self.x_int, self.space.spl_kind)
        
        self.N[self.N < 1e-12] = 0.
        self.D[self.D < 1e-12] = 0.
        
        self.N = spa.csr_matrix(self.N)
        self.D = spa.csr_matrix(self.D)
        
        # LU decompositions
        self.N_LU = spa.linalg.splu(self.N.tocsc())
        self.D_LU = spa.linalg.splu(self.D.tocsc())
        
        # LU decompositions of transposed
        self.N_T_LU = spa.linalg.splu(self.N.T.tocsc())
        self.D_T_LU = spa.linalg.splu(self.D.T.tocsc())
        
    
    # evaluate function at interpolation points    
    def dofs_0(self, fun):
        '''
        Evaluate the callable fun at interpolation points, f[i] = fun(x_i).
        '''
        return fun(self.x_int)
    
    # evaluate integrals between interpolation points
    def dofs_1(self, fun):
        '''
        Integrate the callable fun between interpolation points, f[i] = int_(x_i)^(x_i+1) fun(x) dx.
        Gauss-Legendre integration with n_quad points is used.
        '''
        
        # evaluate function at quadrature points
        mat_f  = fun(self.pts.flatten()).reshape(self.pts.shape[0], self.pts.shape[1])
        values = np.zeros(self.space.NbaseD, dtype=float)
        
        # compute integrals
        for i in range(self.space.NbaseD):
            value = 0.
            for j in range(self.subs[i]):
                value += ker_glob.kernel_int_1d(self.n_quad, self.wts[i + j + self.subs_cum[i]], mat_f[i + j + self.subs_cum[i]])
                
            values[i] = value
                
        return values
    
    # pi_0 projector
    def pi_0(self, fun):
        return self.N_LU.solve(self.dofs_0(fun)) 
    # pi_1 projector
    def pi_1(self, fun):
        return self.D_LU.solve(self.dofs_1(fun)) 

    # pi_0 projector with discrete input
    def pi_0_mat(self, dofs_0):
        '''
        Returns the solution of the interpolation problem N.x = dofs_0 .
        '''
        return self.N_LU.solve(dofs_0)
    
    # pi_1 projector with discrete input
    def pi_1_mat(self, dofs_1):
        '''
        Returns the solution of the histopolation problem D.x = dofs_1 .
        '''
        return self.D_LU.solve(dofs_1)
    
    
    def bases_at_pts(self):
        """
        Basis functions evaluated at point sets for projectors: 

        N_j[ x_int[i] ]
        N_j[ pts.flatten[i] ]

        D_j[ x_int[i] ]
        D_j[ pts.flatten[i] ]
        
        Recall pts : Quadrature points in format (element, quad point) 
        
        Returns 4 scipy.sparse csr matrices.
        """
        
        kind_splines = [False, True]
        
        pts0_N = spa.csr_matrix(bsp.collocation_matrix(self.space.T, self.space.p    , self.x_int, self.space.spl_kind, kind_splines[0]))
        pts0_D = spa.csr_matrix(bsp.collocation_matrix(self.space.t, self.space.p - 1, self.x_int, self.space.spl_kind, kind_splines[1]))
        
        pts1_N = spa.csr_matrix(bsp.collocation_matrix(self.space.T, self.space.p    , self.pts.flatten(), self.space.spl_kind, kind_splines[0]))
        pts1_D = spa.csr_matrix(bsp.collocation_matrix(self.space.t, self.space.p - 1, self.pts.flatten(), self.space.spl_kind, kind_splines[1]))
        
        return pts0_N, pts0_D, pts1_N, pts1_D
    
    
    def dofs_1d_bases(self):
        """
        Computes degrees of freedom of basis functions.
        
        dofs_0_i(N_j)
        dofs_1_i(N_j)
        
        dofs_0_i(D_j)
        dofs_1_i(D_j)
        
        dofs_0 : evaluation at greville points
        dofs_1 : integral between greville points

        Returns 4 numpy arrays.
        """
        
        R0_N = np.empty((self.space.NbaseN, self.space.NbaseN), dtype=float)
        R1_N = np.empty((self.space.NbaseD, self.space.NbaseN), dtype=float)
        
        R0_D = np.empty((self.space.NbaseN, self.space.NbaseD), dtype=float)
        R1_D = np.empty((self.space.NbaseD, self.space.NbaseD), dtype=float)


        # ========= R0_N and R1_N =============
        cj = np.zeros(self.space.NbaseN, dtype=float)

        for j in range(self.space.NbaseN):

            cj[:] = 0.
            cj[j] = 1.

            N_j = lambda eta : self.space.evaluate_N(eta, cj)

            R0_N[:, j] = self.dofs_0(N_j)
            R1_N[:, j] = self.dofs_1(N_j)
            
        # ========= R0_D and R1_D =============
        cj = np.zeros(self.space.NbaseD, dtype=float)

        for j in range(self.space.NbaseD):

            cj[:] = 0.
            cj[j] = 1.

            D_j = lambda eta : self.space.evaluate_D(eta, cj)

            R0_D[:, j] = self.dofs_0(D_j)
            R1_D[:, j] = self.dofs_1(D_j)
            
        R0_N_indices = np.nonzero(R0_N)
        R0_D_indices = np.nonzero(R0_D)
        R1_N_indices = np.nonzero(R1_N)
        R1_D_indices = np.nonzero(R1_D)
        
        return R0_N_indices, R0_D_indices, R1_N_indices, R1_D_indices
    
    
    def dofs_1d_bases_products(self):
        """
        DISCLAIMER: this routine is not finished and should not be used.

        Computes degrees of freedom of products of basis functions.
        
        dofs_0_i(N_j*N_k)
        dofs_0_i(D_j*N_k)
        dofs_0_i(N_j*D_k)
        dofs_0_i(D_j*D_k)
        
        dofs_1_i(N_j*N_k)
        dofs_1_i(D_j*N_k)
        dofs_1_i(N_j*D_k)
        dofs_1_i(D_j*D_k)
        
        dofs_0 : evaluation at greville points
        dofs_1 : integral between greville points

        Returns 8 numpy arrays of the form ().
        """
    
        R0_NN = np.empty((self.space.NbaseN, self.space.NbaseN, self.space.NbaseN), dtype=float)
        R0_DN = np.empty((self.space.NbaseN, self.space.NbaseD, self.space.NbaseN), dtype=float)
        R0_DD = np.empty((self.space.NbaseN, self.space.NbaseD, self.space.NbaseD), dtype=float)

        R1_NN = np.empty((self.space.NbaseD, self.space.NbaseN, self.space.NbaseN), dtype=float)
        R1_DN = np.empty((self.space.NbaseD, self.space.NbaseD, self.space.NbaseN), dtype=float)
        R1_DD = np.empty((self.space.NbaseD, self.space.NbaseD, self.space.NbaseD), dtype=float)


        # ========= R0_NN and R1_NN ==============
        cj = np.zeros(self.space.NbaseN, dtype=float)
        ck = np.zeros(self.space.NbaseN, dtype=float)

        for j in range(self.space.NbaseN):
            for k in range(self.space.NbaseN):

                cj[:] = 0.
                ck[:] = 0.

                cj[j] = 1.
                ck[k] = 1.
                
                N_jN_k = lambda eta : self.space.evaluate_N(eta, cj)*self.space.evaluate_N(eta, ck)
                # There are two evaluation routines at the moment: spline_evaluation_1d (pyccel) and spline_space_1d (slow).
                # The slow one is used here.

                R0_NN[:, j, k] = self.dofs_0(N_jN_k) # These matrices are full and should not be assembled.
                R1_NN[:, j, k] = self.dofs_1(N_jN_k)


        # ========= R0_DN and R1_DN ==============
        cj = np.zeros(self.space.NbaseD, dtype=float)
        ck = np.zeros(self.space.NbaseN, dtype=float)

        for j in range(self.space.NbaseD):
            for k in range(self.space.NbaseN):

                cj[:] = 0.
                ck[:] = 0.

                cj[j] = 1.
                ck[k] = 1.
                
                D_jN_k = lambda eta : self.space.evaluate_D(eta, cj)*self.space.evaluate_N(eta, ck)

                R0_DN[:, j, k] = self.dofs_0(D_jN_k)
                R1_DN[:, j, k] = self.dofs_1(D_jN_k)


        # ========= R0_DD and R1_DD =============
        cj = np.zeros(self.space.NbaseD, dtype=float)
        ck = np.zeros(self.space.NbaseD, dtype=float)

        for j in range(self.space.NbaseD):
            for k in range(self.space.NbaseD):

                cj[:] = 0.
                ck[:] = 0.

                cj[j] = 1.
                ck[k] = 1.
                
                D_jD_k = lambda eta : self.space.evaluate_D(eta, cj)*self.space.evaluate_D(eta, ck)
                
                R0_DD[:, j, k] = self.dofs_0(D_jD_k)
                R1_DD[:, j, k] = self.dofs_1(D_jD_k)


        R0_ND = np.transpose(R0_DN, (0, 2, 1))
        R1_ND = np.transpose(R1_DN, (0, 2, 1))


        # find non-zero entries
        R0_NN_indices = np.nonzero(R0_NN)
        R0_DN_indices = np.nonzero(R0_DN)
        R0_ND_indices = np.nonzero(R0_ND)
        R0_DD_indices = np.nonzero(R0_DD)

        R1_NN_indices = np.nonzero(R1_NN)
        R1_DN_indices = np.nonzero(R1_DN)
        R1_ND_indices = np.nonzero(R1_ND)
        R1_DD_indices = np.nonzero(R1_DD)
        
        R0_NN_i_red = np.empty(R0_NN_indices[0].size, dtype=int)
        R0_DN_i_red = np.empty(R0_DN_indices[0].size, dtype=int)
        R0_ND_i_red = np.empty(R0_ND_indices[0].size, dtype=int)
        R0_DD_i_red = np.empty(R0_DD_indices[0].size, dtype=int)

        R1_NN_i_red = np.empty(R1_NN_indices[0].size, dtype=int)
        R1_DN_i_red = np.empty(R1_DN_indices[0].size, dtype=int)
        R1_ND_i_red = np.empty(R1_ND_indices[0].size, dtype=int)
        R1_DD_i_red = np.empty(R1_DD_indices[0].size, dtype=int)
        
        # ================================
        nv = self.space.NbaseN*R0_NN_indices[1] + R0_NN_indices[2]
        un = np.unique(nv)
        
        for i in range(R0_NN_indices[0].size):
            R0_NN_i_red[i] = np.nonzero(un == nv[i])[0]
            
        # ================================
        nv = self.space.NbaseN*R0_DN_indices[1] + R0_DN_indices[2]
        un = np.unique(nv)
        
        for i in range(R0_DN_indices[0].size):
            R0_DN_i_red[i] = np.nonzero(un == nv[i])[0]
            
        # ================================
        nv = self.space.NbaseD*R0_ND_indices[1] + R0_ND_indices[2]
        un = np.unique(nv)
        
        for i in range(R0_ND_indices[0].size):
            R0_ND_i_red[i] = np.nonzero(un == nv[i])[0]
            
        # ================================
        nv = self.space.NbaseD*R0_DD_indices[1] + R0_DD_indices[2]
        un = np.unique(nv)
        
        for i in range(R0_DD_indices[0].size):
            R0_DD_i_red[i] = np.nonzero(un == nv[i])[0]
            
        # ================================
        nv = self.space.NbaseN*R1_NN_indices[1] + R1_NN_indices[2]
        un = np.unique(nv)
        
        for i in range(R1_NN_indices[0].size):
            R1_NN_i_red[i] = np.nonzero(un == nv[i])[0]
            
        # ================================
        nv = self.space.NbaseN*R1_DN_indices[1] + R1_DN_indices[2]
        un = np.unique(nv)
        
        for i in range(R1_DN_indices[0].size):
            R1_DN_i_red[i] = np.nonzero(un == nv[i])[0]
            
        # ================================
        nv = self.space.NbaseD*R1_ND_indices[1] + R1_ND_indices[2]
        un = np.unique(nv)
        
        for i in range(R1_ND_indices[0].size):
            R1_ND_i_red[i] = np.nonzero(un == nv[i])[0]
            
        # ================================
        nv = self.space.NbaseD*R1_DD_indices[1] + R1_DD_indices[2]
        un = np.unique(nv)
        
        for i in range(R1_DD_indices[0].size):
            R1_DD_i_red[i] = np.nonzero(un == nv[i])[0] 
            
            
        R0_NN_indices = np.vstack((R0_NN_indices[0], R0_NN_indices[1], R0_NN_indices[2], R0_NN_i_red))
        R0_DN_indices = np.vstack((R0_DN_indices[0], R0_DN_indices[1], R0_DN_indices[2], R0_DN_i_red))
        R0_ND_indices = np.vstack((R0_ND_indices[0], R0_ND_indices[1], R0_ND_indices[2], R0_ND_i_red))
        R0_DD_indices = np.vstack((R0_DD_indices[0], R0_DD_indices[1], R0_DD_indices[2], R0_DD_i_red))

        R1_NN_indices = np.vstack((R1_NN_indices[0], R1_NN_indices[1], R1_NN_indices[2], R1_NN_i_red))
        R1_DN_indices = np.vstack((R1_DN_indices[0], R1_DN_indices[1], R1_DN_indices[2], R1_DN_i_red))
        R1_ND_indices = np.vstack((R1_ND_indices[0], R1_ND_indices[1], R1_ND_indices[2], R1_ND_i_red))
        R1_DD_indices = np.vstack((R1_DD_indices[0], R1_DD_indices[1], R1_DD_indices[2], R1_DD_i_red))
        

        return R0_NN_indices, R0_DN_indices, R0_ND_indices, R0_DD_indices, R1_NN_indices, R1_DN_indices, R1_ND_indices, R1_DD_indices 
        
        #return R0_NN, R0_DN, R0_ND, R0_DD, R1_NN, R1_DN, R1_ND, R1_DD, R0_NN_indices, R0_DN_indices, R0_ND_indices, R0_DD_indices, R1_NN_indices, R1_DN_indices, R1_ND_indices, R1_DD_indices



# ======================= 2d for tensor products ====================================
class projectors_tensor_2d:
    """
    Global commuting projectors pi_0, pi_1, pi_2 in 2d.
    
    Parameters:
    -----------
    proj_1d : list of two "projectors_global_1d" objects

    Methods:
    --------
    eval_for_PI:    evaluation at point sets.
    dofs:           degrees of freedom sigma.
    PI_mat:         Kronecker solve of projection problem, dofs input.
    PI:             De Rham commuting projectors.
    PI_0:           projects callable from V_0
    PI_1:           projects callable from V_1
    PI_2:           projects callable from V_2
    """

    def __init__(self, proj_1d):

        self.pts_PI = {'0': None, '11': None, '12': None, '2': None}

        # collection of the point sets for different 2D projectors
        self.pts_PI['0']  = [proj_1d[0].x_int,
                             proj_1d[1].x_int
                             ]
        self.pts_PI['11'] = [proj_1d[0].pts.flatten(),
                             proj_1d[1].x_int
                             ]
        self.pts_PI['12'] = [proj_1d[0].x_int,
                             proj_1d[1].pts.flatten()
                             ]
        self.pts_PI['2']  = [proj_1d[0].pts.flatten(),
                             proj_1d[1].pts.flatten()
                             ] 

        self.ne1, self.nq1 = proj_1d[0].pts.shape
        self.ne2, self.nq2 = proj_1d[1].pts.shape

        self.wts1 = proj_1d[0].wts
        self.wts2 = proj_1d[1].wts

        self.subs1     = proj_1d[0].subs
        self.subs_cum1 = proj_1d[0].subs_cum
        self.subs2     = proj_1d[1].subs
        self.subs_cum2 = proj_1d[1].subs_cum

        self.n1 = proj_1d[0].space.NbaseN
        self.d1 = proj_1d[0].space.NbaseD
        self.n2 = proj_1d[1].space.NbaseN
        self.d2 = proj_1d[1].space.NbaseD

        self.N_LU1 = proj_1d[0].N_LU
        self.D_LU1 = proj_1d[0].D_LU
        self.N_LU2 = proj_1d[1].N_LU
        self.D_LU2 = proj_1d[1].D_LU


    # ======================================        
    def eval_for_PI(self, comp, fun):
        '''
        Evaluate the callable fun at the points corresponding to the projector comp.
            
        Parameters
        ----------
        comp: str
            Which projector: '0', '11', '12' or '2'.

        fun : callable
            fun(eta1, eta2)

        Returns the 2d numpy array f(eta1_i, eta2_j).
        '''
        
        pts_PI = self.pts_PI[comp]
            
        pts1, pts2 = np.meshgrid(pts_PI[0], pts_PI[1], indexing='ij', sparse=True) # numpy >1.7

        #mat_f = np.empty( (pts_PI[0].size, pts_PI[1].size) )
        #mat_f[:, :] = fun(pts1, pts2)

        return fun(pts1, pts2)


    # ======================================        
    def dofs(self, comp, mat_f):
        '''
        Compute the degrees of freedom (rhs) for the projector comp.
            
        Parameters
        ----------
        comp: str
            Which projector: '0', '11', '12' or '2'.

        mat_f : 2d numpy array
            Function values f(eta1_i, eta2_j) at the points set of the projector (from eval_for_PI).

        Returns
        -------
        rhs : 2d numpy array 
            The degrees of freedom sigma_ij.
        '''

        assert mat_f.shape==(self.pts_PI[comp][0].size, 
                             self.pts_PI[comp][1].size
                             )

        if comp=='0':
            rhs = mat_f

        elif comp=='11':
            rhs = np.empty( (self.d1, self.n2) )

            ker_glob.kernel_int_2d_eta1(self.subs1, self.subs_cum1, self.wts1,
                                        mat_f.reshape(self.ne1, self.nq1, self.n2), rhs
                                        )
        elif comp=='12':
            rhs = np.empty( (self.n1, self.d2) )
            
            ker_glob.kernel_int_2d_eta2(self.subs2, self.subs_cum2, self.wts2,
                                        mat_f.reshape(self.n1, self.ne2, self.nq2), rhs
                                        )
        elif comp=='2':
            rhs = np.empty( (self.d1, self.d2) )
            
            ker_glob.kernel_int_2d_eta1_eta2(self.subs1, self.subs2, self.subs_cum1, self.subs_cum2,
                                             self.wts1, self.wts2, 
                                             mat_f.reshape(self.ne1, self.nq1, self.ne2, self.nq2), rhs
                                             )
        else:
            raise ValueError ("wrong projector specified")

        return rhs


    # ======================================        
    def PI_mat(self, comp, rhs):
        '''
        Kronecker solve of the projection problem I.coeffs = rhs

        Parameters:
        -----------
        comp : str
            Which projector: '0', '11', '12' or '2'.

        rhs : 2d numpy array 
            The degrees of freedom sigma_ij.

        Returns:
        --------
        coeffs : 2d numpy array
            The spline coefficients c_ij obtained by projection.
        '''

        if comp=='0':
            assert rhs.shape==(self.n1, self.n2) 
            coeffs = kron_lusolve_2d([self.N_LU1, self.N_LU2], rhs)
        elif comp=='11':
            assert rhs.shape==(self.d1, self.n2) 
            coeffs = kron_lusolve_2d([self.D_LU1, self.N_LU2], rhs)
        elif comp=='12':
            assert rhs.shape==(self.n1, self.d2) 
            coeffs = kron_lusolve_2d([self.N_LU1, self.D_LU2], rhs)
        elif comp=='2':
            assert rhs.shape==(self.d1, self.d2)  
            coeffs = kron_lusolve_2d([self.D_LU1, self.D_LU2], rhs)
        else:
            raise ValueError ("wrong projector specified")
            
        return coeffs


    # ======================================        
    def PI(self, comp, fun):
        '''
        De Rham commuting projectors.

        Parameters:
        -----------
        comp : str
            Which projector: '0', '11', '12' or '2'.

        fun : callable 
            fun(eta1, eta2).

        Returns:
        --------
        coeffs : 2d numpy array
            The spline coefficients c_ij obtained by projection.
        '''

        mat_f = self.eval_for_PI(comp, fun)
        rhs   = self.dofs(comp, mat_f) 

        if comp=='0':
            assert rhs.shape==(self.n1, self.n2) 
            coeffs = kron_lusolve_2d([self.N_LU1, self.N_LU2], rhs)
        elif comp=='11':
            assert rhs.shape==(self.d1, self.n2) 
            coeffs = kron_lusolve_2d([self.D_LU1, self.N_LU2], rhs)
        elif comp=='12':
            assert rhs.shape==(self.n1, self.d2) 
            coeffs = kron_lusolve_2d([self.N_LU1, self.D_LU2], rhs)
        elif comp=='2':
            assert rhs.shape==(self.d1, self.d2)  
            coeffs = kron_lusolve_2d([self.D_LU1, self.D_LU2], rhs)
        else:
            raise ValueError ("wrong projector specified")
            
        return coeffs


    # ======================================        
    def PI_0(self, fun):
        '''
        De Rham commuting projector Pi_0.

        Parameters:
        -----------
        fun : callable 
            Element in V_0 continuous space, fun(eta1, eta2).

        Returns:
        --------
        coeffs : 2d numpy array
            The spline coefficients c_ij obtained by projection.
        '''

        coeffs = self.PI('0', fun)
        
        return coeffs


    # ======================================        
    def PI_1(self, fun1, fun2):
        '''
        De Rham commuting projector Pi_1 acting on fun = (fun1, fun2) in V_1.

        Parameters:
        -----------
        fun1 : callable 
            First component of element in V_1 continuous space, fun1(eta1, eta2).
        fun2 : callable 
            Second component of element in V_1 continuous space, fun2(eta1, eta2).

        Returns:
        --------
        coeffs1 : 2d numpy array
            The spline coefficients c_ij obtained by projection of fun1 on DN.
        coeffs2 : 2d numpy array
            The spline coefficients c_ij obtained by projection of fun2 on ND.
        '''

        coeffs1 = self.PI('11', fun1)
        coeffs2 = self.PI('12', fun2)
            
        return coeffs1, coeffs2


    # ======================================        
    def PI_2(self, fun):
        '''
        De Rham commuting projector Pi_2.

        Parameters:
        -----------
        fun : callable 
            Element in V_2 continuous space, fun(eta1, eta2).

        Returns:
        --------
        coeffs : 2d numpy array
            The spline coefficients c_ij obtained by projection.
        '''

        coeffs = self.PI('2', fun)
        
        return coeffs



# ======================= 3d for tensor products ====================================
class projectors_tensor_3d:
    """
    Global commuting projectors pi_0, pi_1, pi_2, pi_3 in 3d.
    
    Parameters:
    -----------
    proj_1d : list of three "projectors_global_1d" objects

    Methods:
    --------
    eval_for_PI:    evaluation at point sets.
    dofs:           degrees of freedom sigma.
    PI_mat:         Kronecker solve of projection problem, dofs input.
    PI:             De Rham commuting projectors.
    PI_0:           projects callable from V_0
    PI_1:           projects callable from V_1
    PI_2:           projects callable from V_2
    PI_3:           projects callable from V_3
    """

    def __init__(self, proj_1d):

        self.pts_PI = {'0': None, '11': None, '12': None, '13': None
                                , '21': None, '22': None, '23': None, '3': None}

        # collection of the point sets for different 2D projectors
        self.pts_PI['0']  = [proj_1d[0].x_int,
                             proj_1d[1].x_int,
                             proj_1d[2].x_int
                             ]
        self.pts_PI['11'] = [proj_1d[0].pts.flatten(),
                             proj_1d[1].x_int,
                             proj_1d[2].x_int
                             ]
        self.pts_PI['12'] = [proj_1d[0].x_int,
                             proj_1d[1].pts.flatten(),
                             proj_1d[2].x_int
                             ]
        self.pts_PI['13'] = [proj_1d[0].x_int,
                             proj_1d[1].x_int,
                             proj_1d[2].pts.flatten()
                             ]
        self.pts_PI['21'] = [proj_1d[0].x_int,
                             proj_1d[1].pts.flatten(),
                             proj_1d[2].pts.flatten()
                             ]
        self.pts_PI['22'] = [proj_1d[0].pts.flatten(),
                             proj_1d[1].x_int,
                             proj_1d[2].pts.flatten()
                             ]
        self.pts_PI['23'] = [proj_1d[0].pts.flatten(),
                             proj_1d[1].pts.flatten(),
                             proj_1d[2].x_int
                             ]
        self.pts_PI['3']  = [proj_1d[0].pts.flatten(),
                             proj_1d[1].pts.flatten(),
                             proj_1d[2].pts.flatten()
                             ] 

        self.ne1, self.nq1 = proj_1d[0].pts.shape
        self.ne2, self.nq2 = proj_1d[1].pts.shape
        self.ne3, self.nq3 = proj_1d[2].pts.shape

        self.wts1 = proj_1d[0].wts
        self.wts2 = proj_1d[1].wts
        self.wts3 = proj_1d[2].wts

        self.subs1     = proj_1d[0].subs
        self.subs_cum1 = proj_1d[0].subs_cum
        self.subs2     = proj_1d[1].subs
        self.subs_cum2 = proj_1d[1].subs_cum
        self.subs3     = proj_1d[2].subs
        self.subs_cum3 = proj_1d[2].subs_cum

        self.n1 = proj_1d[0].space.NbaseN
        self.d1 = proj_1d[0].space.NbaseD
        self.n2 = proj_1d[1].space.NbaseN
        self.d2 = proj_1d[1].space.NbaseD
        self.n3 = proj_1d[2].space.NbaseN
        self.d3 = proj_1d[2].space.NbaseD

        self.N_LU1 = proj_1d[0].N_LU
        self.D_LU1 = proj_1d[0].D_LU
        self.N_LU2 = proj_1d[1].N_LU
        self.D_LU2 = proj_1d[1].D_LU
        self.N_LU3 = proj_1d[2].N_LU
        self.D_LU3 = proj_1d[2].D_LU


    # ======================================        
    def eval_for_PI(self, comp, fun):
        '''
        Evaluate the callable fun at the points corresponding to the projector comp.
            
        Parameters
        ----------
        comp: str
            Which projector: '0', '11', '12', '13', '21', '22', '23' or '3'.

        fun : callable
            fun(eta1, eta2, eta3)

        Returns the 3d numpy array f(eta1_i, eta2_j, eta3_k).
        '''
        
        pts_PI = self.pts_PI[comp]
            
        pts1, pts2, pts3 = np.meshgrid(pts_PI[0], pts_PI[1], pts_PI[2], indexing='ij', sparse=True) # numpy >1.7

        #mat_f = np.empty( (pts_PI[0].size, pts_PI[1].size) )
        #mat_f[:, :] = fun(pts1, pts2)

        return fun(pts1, pts2, pts3)


    # ======================================        
    def dofs(self, comp, mat_f):
        '''
        Compute the degrees of freedom (rhs) for the projector comp.
            
        Parameters
        ----------
        comp: str
            Which projector: '0', '11', '12', '13', '21', '22', '23' or '3'.

        mat_f : 3d numpy array
            Function values f(eta1_i, eta2_j, eta3_k) at the points set of the projector (from eval_for_PI).

        Returns
        -------
        rhs : 3d numpy array 
            The degrees of freedom sigma_ijk.
        '''

        assert mat_f.shape==(self.pts_PI[comp][0].size, 
                             self.pts_PI[comp][1].size,
                             self.pts_PI[comp][2].size
                             )

        if comp=='0':
            rhs = mat_f

        elif comp=='11':
            rhs = np.empty( (self.d1, self.n2, self.n3) )

            ker_glob.kernel_int_3d_eta1(self.subs1, self.subs_cum1, self.wts1,
                                        mat_f.reshape(self.ne1, self.nq1, self.n2, self.n3), rhs
                                        )
        elif comp=='12':
            rhs = np.empty( (self.n1, self.d2, self.n3) )
            
            ker_glob.kernel_int_3d_eta2(self.subs2, self.subs_cum2, self.wts2,
                                        mat_f.reshape(self.n1, self.ne2, self.nq2, self.n3), rhs
                                        )
        elif comp=='13':
            rhs = np.empty( (self.n1, self.n2, self.d3) )
            
            ker_glob.kernel_int_3d_eta3(self.subs3, self.subs_cum3, self.wts3,
                                        mat_f.reshape(self.n1, self.n2, self.ne3, self.nq3), rhs
                                        )
        elif comp=='21':
            rhs = np.empty( (self.n1, self.d2, self.d3) )
            
            ker_glob.kernel_int_3d_eta2_eta3(self.subs2, self.subs3,
                                             self.subs_cum2, self.subs_cum3,
                                             self.wts2, self.wts3, 
                  mat_f.reshape(self.n1, self.ne2, self.nq2, self.ne3, self.nq3), rhs
                                                 )
        elif comp=='22':
            rhs = np.empty( (self.d1, self.n2, self.d3) )
            
            ker_glob.kernel_int_3d_eta1_eta3(self.subs1, self.subs3,
                                             self.subs_cum1, self.subs_cum3,
                                             self.wts1, self.wts3, 
                  mat_f.reshape(self.ne1, self.nq1, self.n2, self.ne3, self.nq3), rhs
                                                 )
        elif comp=='23':
            rhs = np.empty( (self.d1, self.d2, self.n3) )
            
            ker_glob.kernel_int_3d_eta1_eta2(self.subs1, self.subs2,
                                             self.subs_cum1, self.subs_cum2,
                                             self.wts1, self.wts2, 
                  mat_f.reshape(self.ne1, self.nq1, self.ne2, self.nq2, self.n3), rhs
                                                 )
        elif comp=='3':
            rhs = np.empty( (self.d1, self.d2, self.d3) )
            
            ker_glob.kernel_int_3d_eta1_eta2_eta3(self.subs1, self.subs2, self.subs3,
                                                  self.subs_cum1, self.subs_cum2, self.subs_cum3,
                                                  self.wts1, self.wts2, self.wts3, 
                  mat_f.reshape(self.ne1, self.nq1, self.ne2, self.nq2, self.ne3, self.nq3), rhs
                                                 )
        else:
            raise ValueError ("wrong projector specified")

        return rhs


    # ======================================        
    def dofs_T(self, comp, mat_dofs):
        '''
        Transpose of dofs
            
        Parameters
        ----------
        comp: str
            Which projector: '0', '11', '12', '13', '21', '22', '23' or '3'.

        mat_dofs : 3d numpy array
            Degrees of freedom.

        Returns
        -------
        mat_pts : numpy array 
            comp == '0' 3d of the form(n1, n2, n3)
            comp == '11' 4d of the form(ne1, nq1, n2, n3)
            comp == '12' 4d of the form(n1, n2, nq2, n3)
            comp == '13' 4d of the form(n1, n2, n3, nq3)
            comp == '21' 5d of the form(n1, ne2, nq2, ne3, nq3)
            comp == '22' 5d of the form(ne1, nq1, n2, ne3, nq3)
            comp == '23' 5d of the form(ne1, nq1, ne2, nq2, n3)
            comp == '3' 6d of the form(ne1, nq1, ne2, nq2, n3, nq3)

        '''
        # d1 could be different from n1 in case of non-periodic bc
        # assert mat_f.shape==(self.pts_PI[comp][0].size, 
        #                     self.pts_PI[comp][1].size,
        #                     self.pts_PI[comp][2].size
        #                     )
        #                        )

        if comp=='0':
            rhs = mat_dofs

        elif comp=='11':
            assert mat_dofs.shape == (self.d1, self.n2, self.n3)
            
            #print('f_int shape is', mat_dofs.shape)
            #print('ne1 = ', self.ne1)
            #print('nq1 = ', self.nq1)
            #print('n2 = ', self.n2)
            #print('n3 = ', self.n3)
            #print('subs1.shape is ',self.subs1.shape)
            #print('subs1 is ',self.subs1)
            #print('subs_cum1.shape is ',self.subs_cum1.shape)
            #print('subs_cum1 is ',self.subs_cum1)
            #print('wts1.shape is ', self.wts1.shape)
            rhs = np.empty( (self.ne1, self.nq1, self.n2, self.n3) )

            ker_glob.kernel_int_3d_eta1_transpose(self.subs1, self.subs_cum1, self.wts1,
                                                  mat_dofs, rhs)

            rhs = rhs.reshape(self.ne1 * self.nq1, self.n2, self.n3)
            #print('rhs.shape is', rhs.shape)

        elif comp=='12':
            assert mat_dofs.shape == (self.n1, self.d2, self.n3)

            rhs = np.empty( (self.n1, self.ne2, self.nq2, self.n3) )
            
            ker_glob.kernel_int_3d_eta2_transpose(self.subs2, self.subs_cum2, self.wts2,
                                                  mat_dofs, rhs)

            rhs = rhs.reshape(self.n1, self.ne2 * self.nq2, self.n3)

        elif comp=='13':
            assert mat_dofs.shape == (self.n1, self.n2, self.d3)

            rhs = np.empty( (self.n1, self.n2, self.ne3, self.nq3) )
            
            ker_glob.kernel_int_3d_eta3_transpose(self.subs3, self.subs_cum3, self.wts3,
                                                  mat_dofs, rhs)
                                        
            rhs = rhs.reshape(self.n1, self.n2, self.ne3 * self.nq3)

        elif comp=='21':
            assert mat_dofs.shape == (self.n1, self.d2, self.d3)

            rhs = np.empty( (self.n1, self.ne2, self.nq2, self.ne3, self.nq3) )
            
            ker_glob.kernel_int_3d_eta2_eta3_transpose(self.subs2, self.subs3,
                                             self.subs_cum2, self.subs_cum3,
                                             self.wts2, self.wts3,
                                             mat_dofs, rhs)
            rhs = rhs.reshape(self.n1, self.ne2 * self.nq2, self.ne3 * self.nq3)

        elif comp=='22':
            assert mat_dofs.shape == (self.d1, self.n2, self.d3)

            rhs = np.empty( (self.ne1, self.nq1, self.n2, self.ne3, self.nq3) )
            
            ker_glob.kernel_int_3d_eta1_eta3_transpose(self.subs1, self.subs3,
                                             self.subs_cum1, self.subs_cum3,
                                             self.wts1, self.wts3,
                                             mat_dofs, rhs)
            rhs = rhs.reshape(self.ne1 * self.nq1, self.n2, self.ne3 * self.nq3)

        elif comp=='23':
            assert mat_dofs.shape == (self.d1, self.d2, self.n3)

            rhs = np.empty( (self.ne1, self.nq1, self.ne2, self.nq2, self.n3) )
            
            ker_glob.kernel_int_3d_eta1_eta2_transpose(self.subs1, self.subs2,
                                             self.subs_cum1, self.subs_cum2,
                                             self.wts1, self.wts2,
                                             mat_dofs, rhs)
            rhs = rhs.reshape(self.ne1 * self.nq1, self.ne2 * self.nq2, self.n3)
            
        elif comp=='3':
            assert mat_dofs.shape == (self.d1, self.d2, self.d3)

            rhs = np.empty( (self.ne1, self.nq1, self.ne2, self.nq2, self.ne3, self.nq3) )
            
            ker_glob.kernel_int_3d_eta1_eta2_eta3_transpose(self.subs1, self.subs2, self.subs3,
                                                  self.subs_cum1, self.subs_cum2, self.subs_cum3,
                                                  self.wts1, self.wts2, self.wts3,
                                                  mat_dofs, rhs)
            rhs = rhs.reshape(self.ne1 * self.nq1, self.ne2 * self.nq2, self.ne3 * self.nq3)
        else:
            raise ValueError ("wrong projector specified")

        return rhs

    # ======================================        
    def PI_mat(self, comp, rhs):
        '''
        Kronecker solve of the projection problem I.coeffs = rhs

        Parameters:
        -----------
        comp : str
            Which projector: '0', '11', '12', '13', '21', '22', '23' or '3'.

        rhs : 3d numpy array 
            The degrees of freedom sigma_ijk.

        Returns:
        --------
        coeffs : 3d numpy array
            The spline coefficients c_ijk obtained by projection.
        '''

        if comp=='0':
            assert rhs.shape==(self.n1, self.n2, self.n3) 
            coeffs = kron_lusolve_3d([self.N_LU1, self.N_LU2, self.N_LU3], rhs)
        elif comp=='11':
            assert rhs.shape==(self.d1, self.n2, self.n3) 
            coeffs = kron_lusolve_3d([self.D_LU1, self.N_LU2, self.N_LU3], rhs)
        elif comp=='12':
            assert rhs.shape==(self.n1, self.d2, self.n3) 
            coeffs = kron_lusolve_3d([self.N_LU1, self.D_LU2, self.N_LU3], rhs)
        elif comp=='13':
            assert rhs.shape==(self.n1, self.n2, self.d3) 
            coeffs = kron_lusolve_3d([self.N_LU1, self.N_LU2, self.D_LU3], rhs)
        elif comp=='21':
            assert rhs.shape==(self.n1, self.d2, self.d3)  
            coeffs = kron_lusolve_3d([self.N_LU1, self.D_LU2, self.D_LU3], rhs)
        elif comp=='22':
            assert rhs.shape==(self.d1, self.n2, self.d3)  
            coeffs = kron_lusolve_3d([self.D_LU1, self.N_LU2, self.D_LU3], rhs)
        elif comp=='23':
            assert rhs.shape==(self.d1, self.d2, self.n3)  
            coeffs = kron_lusolve_3d([self.D_LU1, self.D_LU2, self.N_LU3], rhs)
        elif comp=='3':
            assert rhs.shape==(self.d1, self.d2, self.d3)  
            coeffs = kron_lusolve_3d([self.D_LU1, self.D_LU2, self.D_LU3], rhs)
        else:
            raise ValueError ("wrong projector specified")
            
        return coeffs


    # ======================================        
    def PI(self, comp, fun):
        '''
        De Rham commuting projectors.

        Parameters:
        -----------
        comp : str
            Which projector: '0', '11', '12', '13', '21', '22', '23' or '3'.

        fun : callable 
            f(eta1, eta2, eta3)

        Returns:
        --------
        coeffs : 3d numpy array
            The spline coefficients c_ijk obtained by projection.
        '''

        mat_f = self.eval_for_PI(comp, fun)
        rhs   = self.dofs(comp, mat_f) 

        if comp=='0':
            assert rhs.shape==(self.n1, self.n2, self.n3) 
            coeffs = kron_lusolve_3d([self.N_LU1, self.N_LU2, self.N_LU3], rhs)
        elif comp=='11':
            assert rhs.shape==(self.d1, self.n2, self.n3) 
            coeffs = kron_lusolve_3d([self.D_LU1, self.N_LU2, self.N_LU3], rhs)
        elif comp=='12':
            assert rhs.shape==(self.n1, self.d2, self.n3) 
            coeffs = kron_lusolve_3d([self.N_LU1, self.D_LU2, self.N_LU3], rhs)
        elif comp=='13':
            assert rhs.shape==(self.n1, self.n2, self.d3) 
            coeffs = kron_lusolve_3d([self.N_LU1, self.N_LU2, self.D_LU3], rhs)
        elif comp=='21':
            assert rhs.shape==(self.n1, self.d2, self.d3)  
            coeffs = kron_lusolve_3d([self.N_LU1, self.D_LU2, self.D_LU3], rhs)
        elif comp=='22':
            assert rhs.shape==(self.d1, self.n2, self.d3)  
            coeffs = kron_lusolve_3d([self.D_LU1, self.N_LU2, self.D_LU3], rhs)
        elif comp=='23':
            assert rhs.shape==(self.d1, self.d2, self.n3)  
            coeffs = kron_lusolve_3d([self.D_LU1, self.D_LU2, self.N_LU3], rhs)
        elif comp=='3':
            assert rhs.shape==(self.d1, self.d2, self.d3)  
            coeffs = kron_lusolve_3d([self.D_LU1, self.D_LU2, self.D_LU3], rhs)
        else:
            raise ValueError ("wrong projector specified")
            
        return coeffs


    # ======================================        
    def PI_0(self, fun):
        '''
        De Rham commuting projector Pi_0.

        Parameters:
        -----------
        fun : callable 
            Element in V_0 continuous space, f(eta1, eta2, eta3)

        Returns:
        --------
        coeffs : 3d numpy array
            The spline coefficients c_ijk obtained by projection.
        '''

        coeffs = self.PI('0', fun)

        return coeffs


    # ======================================        
    def PI_1(self, fun1, fun2, fun3):
        '''
        De Rham commuting projector Pi_1 acting on fun = (fun1, fun2, fun3) in V_1.

        Parameters:
        -----------
        fun1 : callable 
            First component of element in V_1 continuous space, fun1(eta1, eta2, eta3).
        fun2 : callable 
            Second component of element in V_1 continuous space, fun2(eta1, eta2, eta3).
        fun3 : callable 
            Thirs component of element in V_1 continuous space, fun3(eta1, eta2, eta3).

        Returns:
        --------
        coeffs1 : 3d numpy array
            The spline coefficients c_ijk obtained by projection of fun1 on DNN.
        coeffs2 : 3d numpy array
            The spline coefficients c_ijk obtained by projection of fun2 on NDN.
        coeffs3 : 3d numpy array
            The spline coefficients c_ijk obtained by projection of fun3 on NND.
        '''

        coeffs1 = self.PI('11', fun1)
        coeffs2 = self.PI('12', fun2)
        coeffs3 = self.PI('13', fun3)

        return coeffs1, coeffs2, coeffs3


    # ======================================        
    def PI_2(self, fun1, fun2, fun3):
        '''
        De Rham commuting projector Pi_2 acting on fun = (fun1, fun2, fun3) in V_2.

        Parameters:
        -----------
        fun1 : callable 
            First component of element in V_2 continuous space, fun1(eta1, eta2, eta3).
        fun2 : callable 
            Second component of element in V_2 continuous space, fun2(eta1, eta2, eta3).
        fun3 : callable 
            Thirs component of element in V_2 continuous space, fun3(eta1, eta2, eta3).

        Returns:
        --------
        coeffs1 : 3d numpy array
            The spline coefficients c_ijk obtained by projection of fun1 on NDD.
        coeffs2 : 3d numpy array
            The spline coefficients c_ijk obtained by projection of fun2 on DND.
        coeffs3 : 3d numpy array
            The spline coefficients c_ijk obtained by projection of fun3 on DDN.
        '''

        coeffs1 = self.PI('21', fun1)
        coeffs2 = self.PI('22', fun2)
        coeffs3 = self.PI('23', fun3)

        return coeffs1, coeffs2, coeffs3


    # ======================================        
    def PI_3(self, fun):
        '''
        De Rham commuting projector Pi_3.

        Parameters:
        -----------
        fun : callable 
            Element in V_3 continuous space, f(eta1, eta2, eta3)

        Returns:
        --------
        coeffs : 3d numpy array
            The spline coefficients c_ijk obtained by projection.
        '''

        coeffs = self.PI('3', fun)

        return coeffs



# ======================= 2d ====================================
class projectors_global_2d:
    """
    Global commuting projectors pi_0, pi_1, pi_2 and pi_3 in 2d.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        a 2d tensor-product space of B-splines (set_extraction_operators must have been called)
        
    n_quad : int
        number of Gauss-Legendre quadrature points per integration interval for histopolations
    """
    
    def __init__(self, tensor_space, n_quad):
        
        self.space  = tensor_space     # 2D tensor-product B-splines space
        self.n_quad = n_quad           # number of quadrature point per integration interval
        self.kind   = 'global'         # kind of projector (global vs. local)
        
        n1, n2 = self.space.NbaseN
        d1, d2 = self.space.NbaseD
        
        # standard domain
        if self.space.polar == False:
            
            # including boundary splines
            self.P0_pol_all = spa.identity(n1*n2        , dtype=float, format='csr')
            self.P1_pol_all = spa.identity(d1*n2 + n1*d2, dtype=float, format='csr')
            self.P2_pol_all = spa.identity(n1*d2 + d1*n2, dtype=float, format='csr')
            self.P3_pol_all = spa.identity(d1*d2        , dtype=float, format='csr')

            self.P0_all     =            self.P0_pol_all.copy()
            self.P1_all     = spa.bmat([[self.P1_pol_all, None], [None, self.P0_pol_all]], format='csr')
            self.P2_all     = spa.bmat([[self.P2_pol_all, None], [None, self.P3_pol_all]], format='csr')
            self.P3_all     =            self.P3_pol_all.copy()

            # without boundary splines
            P_NN = spa.identity(n1*n2, format='csr')
            P_DN = spa.identity(d1*n2, format='csr')
            P_ND = spa.identity(n1*d2, format='csr')
            P_DD = spa.identity(d1*d2, format='csr')

            # remove contributions from N-splines at eta1 = 0
            if   self.space.bc[0][0] == 'd' and self.space.spl_kind[0] == False:
                P_NN = P_NN[n2:, :]
                P_ND = P_ND[d2:, :]
            elif self.space.bc[0][0] == 'd' and self.space.spl_kind[0] == True:
                raise ValueError('dirichlet boundary conditions can only be set with clamped splines')

            # remove contributions from N-splines at eta1 = 1
            if   self.space.bc[0][1] == 'd' and self.space.spl_kind[0] == False:
                P_NN = P_NN[:-n2, :]
                P_ND = P_ND[:-d2, :]
            elif self.space.bc[0][1] == 'd' and self.space.spl_kind[0] == True:
                raise ValueError('dirichlet boundary conditions can only be set with clamped splines')

            self.P0_pol = P_NN.tocsr().copy()
            self.P1_pol = spa.bmat([[P_DN, None], [None, P_ND]], format='csr')
            self.P2_pol = spa.bmat([[P_ND, None], [None, P_DN]], format='csr')
            self.P3_pol = P_DD.tocsr().copy()

            self.P0     =            self.P0_pol.copy()
            self.P1     = spa.bmat([[self.P1_pol, None], [None, self.P0_pol]], format='csr')
            self.P2     = spa.bmat([[self.P2_pol, None], [None, self.P3_pol]], format='csr')
            self.P3     =            self.P3_pol.copy()
        
        # polar domain
        else:
            
            # including boundary splines
            self.P0_pol_all = self.space.polar_splines.P0.copy()
            self.P1_pol_all = self.space.polar_splines.P1C.copy()
            self.P2_pol_all = self.space.polar_splines.P1D.copy()
            self.P3_pol_all = self.space.polar_splines.P2.copy()

            self.P0_all     =            self.P0_pol_all.copy()
            self.P1_all     = spa.bmat([[self.P1_pol_all, None], [None, self.P0_pol_all]], format='csr')
            self.P2_all     = spa.bmat([[self.P2_pol_all, None], [None, self.P3_pol_all]], format='csr')
            self.P3_all     =            self.P3_pol_all.copy()

            # without boundary splines
            P0_NN = self.space.polar_splines.P0.copy()

            P1_DN = self.space.polar_splines.P1C.copy()[:(0 + (d1 - 1)*d2) , :]
            P1_ND = self.space.polar_splines.P1C.copy()[ (0 + (d1 - 1)*d2):, :]

            P2_ND = self.space.polar_splines.P1D.copy()[:(2 + (n1 - 2)*d2) , :]
            P2_DN = self.space.polar_splines.P1D.copy()[ (2 + (n1 - 2)*d2):, :]

            P3_DD = self.space.polar_splines.P2.copy()

            # remove contributions from N-splines at eta1 = 1
            if   self.space.bc[0][1] == 'd' and self.space.spl_kind[0] == False:
                P0_NN = P0_NN[:-n2, :]
                P1_ND = P1_ND[:-d2, :]
                P2_ND = P2_ND[:-d2, :]
            elif self.space.bc[0][1] == 'd' and self.space.spl_kind[0] == True:
                raise ValueError('dirichlet boundary conditions can only be set with clamped splines')

            self.P0_pol = P0_NN.tocsr().copy()
            self.P1_pol = spa.bmat([[P1_DN], [P1_ND]], format='csr')
            self.P2_pol = spa.bmat([[P2_ND], [P2_DN]], format='csr')
            self.P3_pol = P3_DD.tocsr().copy()

            self.P0     =            self.P0_pol.copy()
            self.P1     = spa.bmat([[self.P1_pol, None], [None, self.P0_pol]], format='csr')
            self.P2     = spa.bmat([[self.P2_pol, None], [None, self.P3_pol]], format='csr')
            self.P3     =            self.P3_pol.copy()
        
            
        # Gauss - Legendre quadrature points and weights in (-1, 1)
        self.pts_loc = [np.polynomial.legendre.leggauss(n_quad)[0] for n_quad in self.n_quad]
        self.wts_loc = [np.polynomial.legendre.leggauss(n_quad)[1] for n_quad in self.n_quad]
        
        # set interpolation points (Greville points)
        self.x_int = [bsp.greville(T, p, spl_kind) for T, p, spl_kind in zip(self.space.T, self.space.p, self.space.spl_kind)]
        
        # set histopolation grid and number of sub-intervals
        self.x_his    = [0, 0]
        self.subs     = [0, 0]
        self.subs_cum = [0, 0]
        self.pts      = [0, 0]
        self.wts      = [0, 0]
        
        for dim in range(2):
        
            # clamped splines
            if self.space.spl_kind[dim] == False:

                # even spline degree
                if self.space.p[dim]%2 == 0:
                    self.x_his[dim] = np.union1d(self.x_int[dim], self.space.el_b[dim])
                    self.subs[dim]  = 2*np.ones(self.x_int[dim].size - 1, dtype=int)

                    self.subs[dim][:self.space.p[dim]//2 ] = 1
                    self.subs[dim][-self.space.p[dim]//2:] = 1

                # odd spline degree
                else:
                    self.x_his[dim] = np.copy(self.x_int[dim])
                    self.subs[dim]  = 1*np.ones(self.x_int[dim].size - 1, dtype=int)

            # periodic splines
            else:

                # even spline degree
                if self.space.p[dim]%2 == 0:
                    self.x_his[dim] = np.union1d(self.x_int[dim], self.space.el_b[dim][1:])
                    self.x_his[dim] = np.append(self.x_his[dim], self.space.el_b[dim][-1] + self.x_his[dim][0])
                    self.subs[dim]  = 2*np.ones(self.x_int[dim].size, dtype=int)

                # odd spline degree
                else:
                    self.x_his[dim] = np.append(self.x_int[dim], self.space.el_b[dim][-1])
                    self.subs[dim]  = 1*np.ones(self.x_int[dim].size, dtype=int)

            self.subs_cum[dim] = np.append(0, np.cumsum(self.subs[dim] - 1)[:-1])

            # quadrature points and weights
            self.pts[dim], self.wts[dim] = bsp.quadrature_grid(self.x_his[dim], self.pts_loc[dim], self.wts_loc[dim])
            self.pts[dim]                = self.pts[dim]%self.space.el_b[dim][-1]
        
        
        # interpolation and histopolation_matrix
        self.N = [bsp.collocation_matrix(  T, p, x, spl_kind) for T, p, x, spl_kind in zip(self.space.T, self.space.p, self.x_int, self.space.spl_kind)]
        self.D = [bsp.histopolation_matrix(T, p, x, spl_kind) for T, p, x, spl_kind in zip(self.space.T, self.space.p, self.x_int, self.space.spl_kind)]
        
        self.N[0][self.N[0] < 1e-12] = 0.
        self.N[1][self.N[1] < 1e-12] = 0.
        
        self.D[0][self.D[0] < 1e-12] = 0.
        self.D[1][self.D[1] < 1e-12] = 0.
        
        self.N = [spa.csr_matrix(N) for N in self.N]
        self.D = [spa.csr_matrix(D) for D in self.D]
        
        # LU decompositions of 1D (transposed) interpolation and histopolation matrices
        self.N_LU = [spa.linalg.splu(N.tocsc()) for N in self.N]
        self.D_LU = [spa.linalg.splu(D.tocsc()) for D in self.D]
        
        self.N_T_LU = [spa.linalg.splu(N.T.tocsc()) for N in self.N]
        self.D_T_LU = [spa.linalg.splu(D.T.tocsc()) for D in self.D]
        
        self.N_inv = [np.linalg.inv(N.toarray()) for N in self.N]
        self.D_inv = [np.linalg.inv(D.toarray()) for D in self.D]
        
        # 2D interpolation/histopolation matrices in poloidal plane
        NN = spa.kron(self.N[0], self.N[1], format='csr')
        DN = spa.kron(self.D[0], self.N[1], format='csr')
        ND = spa.kron(self.N[0], self.D[1], format='csr')
        DD = spa.kron(self.D[0], self.D[1], format='csr')
        
        DN_ND = spa.bmat([[DN, None], [None, ND]], format='csr')
        ND_DN = spa.bmat([[ND, None], [None, DN]], format='csr')
        
        DN_ND_NN = spa.bmat([[DN, None, None], [None, ND, None], [None, None, NN]], format='csr')
        ND_DN_DD = spa.bmat([[ND, None, None], [None, DN, None], [None, None, DD]], format='csr')
        
        self.I0_pol     = self.P0_pol.dot(NN.dot(self.space.E0_pol.T)).tocsr()
        self.I0_pol_all = self.P0_pol_all.dot(NN.dot(self.space.E0_pol_all.T)).tocsr()
        
        self.I1_pol     = self.P1_pol.dot(DN_ND.dot(self.space.E1_pol.T)).tocsr()
        self.I1_pol_all = self.P1_pol_all.dot(DN_ND.dot(self.space.E1_pol_all.T)).tocsr()
        
        self.I2_pol     = self.P2_pol.dot(ND_DN.dot(self.space.E2_pol.T)).tocsr()
        self.I2_pol_all = self.P2_pol_all.dot(ND_DN.dot(self.space.E2_pol_all.T)).tocsr()
       
        self.I3_pol     = self.P3_pol.dot(DD.dot(self.space.E3_pol.T)).tocsr()
        self.I3_pol_all = self.P3_pol_all.dot(DD.dot(self.space.E3_pol_all.T)).tocsr()
        
        self.I0 = self.P0.dot(NN.dot(self.space.E0.T)).tocsr()
        self.I1 = self.P1.dot(DN_ND_NN.dot(self.space.E1.T)).tocsr()
        self.I2 = self.P2.dot(ND_DN_DD.dot(self.space.E2.T)).tocsr()
        self.I3 = self.P3.dot(DD.dot(self.space.E3.T)).tocsr()
        
        # LU decompositions in poloidal plane
        self.I0_pol_LU = spa.linalg.splu(self.I0_pol.tocsc())
        self.I1_pol_LU = spa.linalg.splu(self.I1_pol.tocsc())
        self.I2_pol_LU = spa.linalg.splu(self.I2_pol.tocsc())
        self.I3_pol_LU = spa.linalg.splu(self.I3_pol.tocsc())
        
        self.I0_pol_all_LU = spa.linalg.splu(self.I0_pol_all.tocsc())
        self.I1_pol_all_LU = spa.linalg.splu(self.I1_pol_all.tocsc())
        self.I2_pol_all_LU = spa.linalg.splu(self.I2_pol_all.tocsc())
        self.I3_pol_all_LU = spa.linalg.splu(self.I3_pol_all.tocsc())
        
        self.I0_pol_T_LU = spa.linalg.splu(self.I0_pol.T.tocsc())
        self.I1_pol_T_LU = spa.linalg.splu(self.I1_pol.T.tocsc())
        self.I2_pol_T_LU = spa.linalg.splu(self.I2_pol.T.tocsc())
        self.I3_pol_T_LU = spa.linalg.splu(self.I3_pol.T.tocsc())

        # shift first radial interpolation point away from pole
        if self.space.polar == True:
            self.x_int[0][0] += 0.00001
        
        # collection of the point sets for different 2D projectors
        self.pts_PI_0   = [self.x_int[0]        , self.x_int[1]        ]
        
        self.pts_PI_1_1 = [self.pts[0].flatten(), self.x_int[1]        ]
        self.pts_PI_1_2 = [self.x_int[0]        , self.pts[1].flatten()]
        self.pts_PI_1_3 = [self.x_int[0]        , self.x_int[1]        ]
        
        self.pts_PI_2_1 = [self.x_int[0]        , self.pts[1].flatten()]
        self.pts_PI_2_2 = [self.pts[0].flatten(), self.x_int[1]        ]
        self.pts_PI_2_3 = [self.pts[0].flatten(), self.pts[1].flatten()]
        
        self.pts_PI_3   = [self.pts[0].flatten(), self.pts[1].flatten()]
        
        
    # ========================================    
    def getpts_for_PI(self, comp):
        """
        Get the needed point sets for a given projector.
        
        Parameters
        ----------
        comp: int
            which projector, one of (0, 11, 12, 13, 21, 22, 23, 3).
        
        Returns
        -------
        pts_PI : list of 1d arrays 
            the 1D point sets.
        """
        
        if   comp == 0:
            pts_PI = self.pts_PI_0
        
        elif comp == 11:
            pts_PI = self.pts_PI_1_1 
        elif comp == 12:
            pts_PI = self.pts_PI_1_2
        elif comp == 13:
            pts_PI = self.pts_PI_1_3
            
        elif comp == 21:
            pts_PI = self.pts_PI_2_1 
        elif comp == 22:
            pts_PI = self.pts_PI_2_2
        elif comp == 23:
            pts_PI = self.pts_PI_2_3
        
        elif comp == 3:
            pts_PI = self.pts_PI_3
            
        else:
            raise ValueError ("wrong projector specified")

        return pts_PI
    
    # ======================================        
    def eval_for_PI(self, comp, fun, eval_kind):
        """
        Evaluates the callable "fun" at the points corresponding to the projector, and returns the result as 2d array "mat_f".
            
        Parameters
        ----------
        comp: int
            which projector, one of (0, 11, 12, 13, 21, 22, 23, 3).
            
        fun : callable
            the function fun(eta1, eta2) to project
               
        Returns
        -------
        mat_f : 2d array
            function evaluated on a 2d meshgrid contstructed from the 1d point sets.
        """
        
        # get intepolation and quadrature points
        pts_PI = self.getpts_for_PI(comp)
        
        # number of evaluation points in each direction
        n_pts  = [pts_PI[0].size, pts_PI[1].size]

        # array of evaluated function
        mat_f  = np.empty((n_pts[0], n_pts[1]), dtype=float)
        
        # external function call if a callable is passed
        if callable(fun):
            
            # create a meshgrid and evaluate function on point set
            if eval_kind == 'meshgrid':
                pts1, pts2  = np.meshgrid(pts_PI[0], pts_PI[1], indexing='ij')
                mat_f[:, :] = fun(pts1, pts2)
                
            # tensor-product evaluation is done by input function
            elif eval_kind == 'tensor_product':
                mat_f[:, :] = fun(pts_PI[0], pts_PI[1])
                
            # point-wise evaluation
            else:
                for i1 in range(pts_PI[0].size):
                    for i2 in range(pts_PI[1].size):
                        mat_f[i1, i2] = fun(pts_PI[0][i1], pts_PI[1][i2])
        
        # internal function call
        else:
            print('no internal 2D function implemented!')
       
        return mat_f
    
    
    # ======================================
    def solve_V0(self, include_bc, rhs):
        
        # solve system
        if include_bc == True:
            coeffs = self.I0_pol_all_LU.solve(rhs)
        else:
            coeffs = self.I0_pol_LU.solve(rhs)
            
        return coeffs
    
    
    # ======================================
    def solve_V1(self, include_bc, rhs):

        # solve system
        if include_bc == True:
            coeffs1 = self.I1_pol_all_LU.solve(rhs[:self.P1_pol_all.shape[0] ])
            coeffs3 = self.I0_pol_all_LU.solve(rhs[ self.P1_pol_all.shape[0]:])
        else:
            coeffs1 = self.I1_pol_LU.solve(rhs[:self.P1_pol.shape[0] ])
            coeffs3 = self.I0_pol_LU.solve(rhs[ self.P1_pol.shape[0]:])
            
        return np.concatenate((coeffs1, coeffs3))
    
    # ======================================
    def solve_V2(self, include_bc, rhs):
        
        # solve system
        if include_bc == True:
            coeffs1 = self.I2_pol_all_LU.solve(rhs[:self.P2_pol_all.shape[0] ])
            coeffs3 = self.I3_pol_all_LU.solve(rhs[ self.P2_pol_all.shape[0]:])
        else:
            coeffs1 = self.I2_pol_LU.solve(rhs[:self.P2_pol.shape[0] ])
            coeffs3 = self.I3_pol_LU.solve(rhs[ self.P2_pol.shape[0]:])
            
        return np.concatenate((coeffs1, coeffs3))
    
    # ======================================
    def solve_V3(self, include_bc, rhs):
        
        # solve system
        if include_bc == True:
            coeffs = self.I3_pol_all_LU.solve(rhs)
        else:
            coeffs = self.I3_pol_LU.solve(rhs)
            
        return coeffs
        
    
    # ======================================        
    def pi_0(self, fun, include_bc=True, eval_kind='meshgrid', interp=True):
        
        #  ==== evaluate on tensor-product grid ====
        rhs = self.eval_for_PI(0, fun, eval_kind)
        
        # ====== apply extraction operator =========
        if include_bc == True:
            rhs = self.P0_pol_all.dot(rhs.flatten())
        else:
            rhs = self.P0_pol.dot(rhs.flatten())
        
        # ====== solve for coefficients ============
        if interp == True:
            return self.solve_V0(include_bc, rhs)
        else:
            return rhs
    
    # ======================================        
    def pi_1(self, fun, include_bc=True, eval_kind='meshgrid', interp=True):
        
        # ====== integrate along 1-direction =======
        n1   = self.pts_PI_1_1[0].size//self.n_quad[0] 
        n2   = self.pts_PI_1_1[1].size 
        
        pts  = self.eval_for_PI(11, fun[0], eval_kind)
        
        rhs1 = np.empty((self.subs[0].size, n2), dtype=float)
        ker_glob.kernel_int_2d_eta1(self.subs[0], self.subs_cum[0], self.wts[0], pts.reshape(n1, self.n_quad[0], n2), rhs1)
            
        # ====== integrate along 2-direction =======
        n1   = self.pts_PI_1_2[0].size
        n2   = self.pts_PI_1_2[1].size//self.n_quad[1]  
        
        pts  = self.eval_for_PI(12, fun[1], eval_kind)
        
        rhs2 = np.empty((n1, self.subs[1].size), dtype=float)
        ker_glob.kernel_int_2d_eta2(self.subs[1], self.subs_cum[1], self.wts[1], pts.reshape(n1, n2, self.n_quad[1]), rhs2)
        
        # ====== interpolation of third component ==
        rhs3 = self.eval_for_PI(0, fun[2], eval_kind)
        
        # ====== apply extraction operator =========
        if include_bc == True:
            rhs = self.P1_all.dot(np.concatenate((rhs1.flatten(), rhs2.flatten(), rhs3.flatten())))
        else:
            rhs = self.P1.dot(np.concatenate((rhs1.flatten(), rhs2.flatten(), rhs3.flatten())))
            
        # ====== solve for coefficients ============
        if interp == True:
            return self.solve_V1(include_bc, rhs)
        else:
            return rhs
        
    # ======================================        
    def pi_2(self, fun, include_bc=True, eval_kind='meshgrid', interp=True):
        
        # ====== integrate along 2-direction =======
        n1   = self.pts_PI_2_1[0].size
        n2   = self.pts_PI_2_1[1].size//self.n_quad[1]  
        
        pts  = self.eval_for_PI(21, fun[0], eval_kind)
        
        rhs1 = np.empty((n1, self.subs[1].size), dtype=float)
        ker_glob.kernel_int_2d_eta2(self.subs[1], self.subs_cum[1], self.wts[1], pts.reshape(n1, n2, self.n_quad[1]), rhs1)
        
        # ====== integrate along 1-direction =======
        n1   = self.pts_PI_2_2[0].size//self.n_quad[0] 
        n2   = self.pts_PI_2_2[1].size 
        
        pts  = self.eval_for_PI(22, fun[1], eval_kind)
        
        rhs2 = np.empty((self.subs[0].size, n2), dtype=float)
        ker_glob.kernel_int_2d_eta1(self.subs[0], self.subs_cum[0], self.wts[0], pts.reshape(n1, self.n_quad[0], n2), rhs2)
        
        # ====== integrate in 1-2-plane ============
        n1   = self.pts_PI_2_3[0].size//self.n_quad[0] 
        n2   = self.pts_PI_2_3[1].size//self.n_quad[1]
        
        pts  = self.eval_for_PI(23, fun[2], eval_kind)
        
        rhs3 = np.empty((self.subs[0].size, self.subs[1].size), dtype=float)
        ker_glob.kernel_int_2d_eta1_eta2(self.subs[0], self.subs[1], self.subs_cum[0], self.subs_cum[1], self.wts[0], self.wts[1], pts.reshape(n1, self.n_quad[0], n2, self.n_quad[1]), rhs3)
            
        # ====== apply extraction operator =========
        if include_bc == True:
            rhs = self.P2_all.dot(np.concatenate((rhs1.flatten(), rhs2.flatten(), rhs3.flatten())))
        else:
            rhs = self.P2.dot(np.concatenate((rhs1.flatten(), rhs2.flatten(), rhs3.flatten())))
            
        # ====== solve for coefficients ============
        if interp == True:
            return self.solve_V2(include_bc, rhs)
        else:
            return rhs
        
    # ======================================        
    def pi_3(self, fun, include_bc=True, eval_kind='meshgrid', interp=True):
        
        n1  = self.pts_PI_3[0].size//self.n_quad[0] 
        n2  = self.pts_PI_3[1].size//self.n_quad[1]
        
        pts = self.eval_for_PI(3, fun, eval_kind)
        
        rhs = np.empty((self.subs[0].size, self.subs[1].size), dtype=float)
        ker_glob.kernel_int_2d_eta1_eta2(self.subs[0], self.subs[1], self.subs_cum[0], self.subs_cum[1], self.wts[0], self.wts[1], pts.reshape(n1, self.n_quad[0], n2, self.n_quad[1]), rhs)
        
        # ====== apply extraction operator =========
        if include_bc == True:
            rhs = self.P3_pol_all.dot(rhs.flatten())
        else:
            rhs = self.P3_pol.dot(rhs.flatten())
            
        # ====== solve for coefficients ============
        if interp == True:
            return self.solve_V3(include_bc, rhs)
        else:
            return rhs
    
    
    

    
# ======================= 3d ====================================
class projectors_global_3d:
    """
    Global commuting projectors pi_0, pi_1, pi_2 and pi_3 in 3d.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        a 3d tensor-product space of B-splines (set_extraction_operators must have been called)
        
    n_quad : int
        number of Gauss-Legendre quadrature points per integration interval for histopolations
    """
    
    def __init__(self, tensor_space, n_quad):
        
        self.space  = tensor_space     # 3D tensor-product B-splines space
        self.n_quad = n_quad           # number of quadrature point per integration interval
        self.kind   = 'global'         # kind of projector (global vs. local)
        
        n1, n2, n3 = self.space.NbaseN
        d1, d2, d3 = self.space.NbaseD
        
        # standard domain
        if self.space.polar == False:

            # including boundary splines
            self.P0_pol_all = spa.identity(n1*n2        , dtype=float, format='csr')
            self.P1_pol_all = spa.identity(d1*n2 + n1*d2, dtype=float, format='csr')
            self.P2_pol_all = spa.identity(n1*d2 + d1*n2, dtype=float, format='csr')
            self.P3_pol_all = spa.identity(d1*d2        , dtype=float, format='csr')

            self.P0_all     = spa.identity(self.space.Ntot_0form       , dtype=float, format='csr')
            self.P1_all     = spa.identity(self.space.Ntot_1form_cum[2], dtype=float, format='csr')
            self.P2_all     = spa.identity(self.space.Ntot_2form_cum[2], dtype=float, format='csr')
            self.P3_all     = spa.identity(self.space.Ntot_3form       , dtype=float, format='csr')

            # without boundary splines
            P_NN = spa.identity(n1*n2, format='csr')
            P_DN = spa.identity(d1*n2, format='csr')
            P_ND = spa.identity(n1*d2, format='csr')
            P_DD = spa.identity(d1*d2, format='csr')

            # remove contributions from N-splines at eta1 = 0
            if   self.space.bc[0][0] == 'd' and self.space.spl_kind[0] == False:
                P_NN = P_NN[n2:, :]
                P_ND = P_ND[d2:, :]
            elif self.space.bc[0][0] == 'd' and self.space.spl_kind[0] == True:
                raise ValueError('dirichlet boundary conditions can only be set with clamped splines')

            # remove contributions from N-splines at eta1 = 1
            if   self.space.bc[0][1] == 'd' and self.space.spl_kind[0] == False:
                P_NN = P_NN[:-n2, :]
                P_ND = P_ND[:-d2, :]
            elif self.space.bc[0][1] == 'd' and self.space.spl_kind[0] == True:
                raise ValueError('dirichlet boundary conditions can only be set with clamped splines')

            self.P0_pol = P_NN.tocsr().copy()
            self.P1_pol = spa.bmat([[P_DN, None], [None, P_ND]], format='csr')
            self.P2_pol = spa.bmat([[P_ND, None], [None, P_DN]], format='csr')
            self.P3_pol = P_DD.tocsr().copy()

            self.P0     = spa.kron(self.P0_pol, spa.identity(n3), format='csr')  
            P1_1        = spa.kron(self.P1_pol, spa.identity(n3), format='csr')
            P1_3        = spa.kron(self.P0_pol, spa.identity(d3), format='csr')

            P2_1        = spa.kron(self.P2_pol, spa.identity(d3), format='csr')
            P2_3        = spa.kron(self.P3_pol, spa.identity(n3), format='csr')
            self.P3     = spa.kron(self.P3_pol, spa.identity(d3), format='csr')

            self.P1     = spa.bmat([[P1_1, None], [None, P1_3]], format='csr')
            self.P2     = spa.bmat([[P2_1, None], [None, P2_3]], format='csr')

        # polar domain
        else:
            
            # including boundary splines
            self.P0_pol_all = self.space.polar_splines.P0.copy()
            self.P1_pol_all = self.space.polar_splines.P1C.copy()
            self.P2_pol_all = self.space.polar_splines.P1D.copy()
            self.P3_pol_all = self.space.polar_splines.P2.copy()
            
            # expansion in third dimension
            self.P0_all     = spa.kron(self.P0_pol_all, spa.identity(n3), format='csr')  
            P1_all_1        = spa.kron(self.P1_pol_all, spa.identity(n3), format='csr')
            P1_all_3        = spa.kron(self.P0_pol_all, spa.identity(d3), format='csr')

            P2_all_1        = spa.kron(self.P2_pol_all, spa.identity(d3), format='csr')
            P2_all_3        = spa.kron(self.P3_pol_all, spa.identity(n3), format='csr')
            self.P3_all     = spa.kron(self.P3_pol_all, spa.identity(d3), format='csr')

            self.P1_all     = spa.bmat([[P1_all_1, None], [None, P1_all_3]], format='csr')
            self.P2_all     = spa.bmat([[P2_all_1, None], [None, P2_all_3]], format='csr')
            
            # without boundary splines
            P0_NN = self.space.polar_splines.P0.copy()

            P1_DN = self.space.polar_splines.P1C.copy()[:(0 + (d1 - 1)*d2) , :]
            P1_ND = self.space.polar_splines.P1C.copy()[ (0 + (d1 - 1)*d2):, :]

            P2_ND = self.space.polar_splines.P1D.copy()[:(2 + (n1 - 2)*d2) , :]
            P2_DN = self.space.polar_splines.P1D.copy()[ (2 + (n1 - 2)*d2):, :]

            P3_DD = self.space.polar_splines.P2.copy()

            # remove contributions from N-splines at eta1 = 1
            if   self.space.bc[0][1] == 'd' and self.space.spl_kind[0] == False:
                P0_NN = P0_NN[:-n2, :]
                P1_ND = P1_ND[:-d2, :]
                P2_ND = P2_ND[:-d2, :]
            elif self.space.bc[0][1] == 'd' and self.space.spl_kind[0] == True:
                raise ValueError('dirichlet boundary conditions can only be set with clamped splines')

            self.P0_pol = P0_NN.tocsr().copy()
            self.P1_pol = spa.bmat([[P1_DN], [P1_ND]], format='csr')
            self.P2_pol = spa.bmat([[P2_ND], [P2_DN]], format='csr')
            self.P3_pol = P3_DD.tocsr().copy()

            self.P0     = spa.kron(self.P0_pol, spa.identity(n3), format='csr')  
            P1_1        = spa.kron(self.P1_pol, spa.identity(n3), format='csr')
            P1_3        = spa.kron(self.P0_pol, spa.identity(d3), format='csr')

            P2_1        = spa.kron(self.P2_pol, spa.identity(d3), format='csr')
            P2_3        = spa.kron(self.P3_pol, spa.identity(n3), format='csr')
            self.P3     = spa.kron(self.P3_pol, spa.identity(d3), format='csr')

            self.P1     = spa.bmat([[P1_1, None], [None, P1_3]], format='csr')
            self.P2     = spa.bmat([[P2_1, None], [None, P2_3]], format='csr')
            

        # Gauss - Legendre quadrature points and weights in (-1, 1)
        self.pts_loc = [np.polynomial.legendre.leggauss(n_quad)[0] for n_quad in self.n_quad]
        self.wts_loc = [np.polynomial.legendre.leggauss(n_quad)[1] for n_quad in self.n_quad]
        
        # set interpolation points (Greville points)
        self.x_int = [bsp.greville(T, p, spl_kind) for T, p, spl_kind in zip(self.space.T, self.space.p, self.space.spl_kind)]
        
        # set histopolation grid and number of sub-intervals
        self.x_his    = [0, 0, 0]
        self.subs     = [0, 0, 0]
        self.subs_cum = [0, 0, 0]
        self.pts      = [0, 0, 0]
        self.wts      = [0, 0, 0]
        
        for dim in range(3):
        
            # clamped splines
            if self.space.spl_kind[dim] == False:

                # even spline degree
                if self.space.p[dim]%2 == 0:
                    self.x_his[dim] = np.union1d(self.x_int[dim], self.space.el_b[dim])
                    self.subs[dim]  = 2*np.ones(self.x_int[dim].size - 1, dtype=int)

                    self.subs[dim][:self.space.p[dim]//2 ] = 1
                    self.subs[dim][-self.space.p[dim]//2:] = 1

                # odd spline degree
                else:
                    self.x_his[dim] = np.copy(self.x_int[dim])
                    self.subs[dim]  = 1*np.ones(self.x_int[dim].size - 1, dtype=int)

            # periodic splines
            else:

                # even spline degree
                if self.space.p[dim]%2 == 0:
                    self.x_his[dim] = np.union1d(self.x_int[dim], self.space.el_b[dim][1:])
                    self.x_his[dim] = np.append(self.x_his[dim], self.space.el_b[dim][-1] + self.x_his[dim][0])
                    self.subs[dim]  = 2*np.ones(self.x_int[dim].size, dtype=int)

                # odd spline degree
                else:
                    self.x_his[dim] = np.append(self.x_int[dim], self.space.el_b[dim][-1])
                    self.subs[dim]  = 1*np.ones(self.x_int[dim].size, dtype=int)

            self.subs_cum[dim] = np.append(0, np.cumsum(self.subs[dim] - 1)[:-1])

            # quadrature points and weights
            self.pts[dim], self.wts[dim] = bsp.quadrature_grid(self.x_his[dim], self.pts_loc[dim], self.wts_loc[dim])
            self.pts[dim]                = self.pts[dim]%self.space.el_b[dim][-1]

        
        # 1D interpolation and histopolation matrices
        self.N = [bsp.collocation_matrix(  T, p, x, spl_kind) for T, p, x, spl_kind in zip(self.space.T, self.space.p, self.x_int, self.space.spl_kind)]
        self.D = [bsp.histopolation_matrix(T, p, x, spl_kind) for T, p, x, spl_kind in zip(self.space.T, self.space.p, self.x_int, self.space.spl_kind)]
        
        self.N[0][self.N[0] < 1e-12] = 0.
        self.N[1][self.N[1] < 1e-12] = 0.
        self.N[2][self.N[2] < 1e-12] = 0.

        self.D[0][self.D[0] < 1e-12] = 0.
        self.D[1][self.D[1] < 1e-12] = 0.
        self.D[2][self.D[2] < 1e-12] = 0.
        
        self.N = [spa.csr_matrix(N) for N in self.N]
        self.D = [spa.csr_matrix(D) for D in self.D]
        
        # LU decompositions of 1D (transposed) interpolation and histopolation matrices
        self.N_LU = [spa.linalg.splu(N.tocsc()) for N in self.N]
        self.D_LU = [spa.linalg.splu(D.tocsc()) for D in self.D]
        
        self.N_T_LU = [spa.linalg.splu(N.T.tocsc()) for N in self.N]
        self.D_T_LU = [spa.linalg.splu(D.T.tocsc()) for D in self.D]
        
        self.N_inv = [np.linalg.inv(N.toarray()) for N in self.N]
        self.D_inv = [np.linalg.inv(D.toarray()) for D in self.D]
        
        # 2D interpolation/histopolation matrices in poloidal plane
        NN = spa.kron(self.N[0], self.N[1], format='csr')
        DN = spa.kron(self.D[0], self.N[1], format='csr')
        ND = spa.kron(self.N[0], self.D[1], format='csr')
        DD = spa.kron(self.D[0], self.D[1], format='csr')
        
        DN_ND = spa.bmat([[DN, None], [None, ND]], format='csr')
        ND_DN = spa.bmat([[ND, None], [None, DN]], format='csr')
        
        DN_ND_NN = spa.bmat([[DN, None, None], [None, ND, None], [None, None, NN]], format='csr')
        ND_DN_DD = spa.bmat([[ND, None, None], [None, DN, None], [None, None, DD]], format='csr')
        
        self.I0_pol     = self.P0_pol.dot(NN.dot(self.space.E0_pol.T)).tocsr()
        self.I0_pol_all = self.P0_pol_all.dot(NN.dot(self.space.E0_pol_all.T)).tocsr()
        
        self.I1_pol     = self.P1_pol.dot(DN_ND.dot(self.space.E1_pol.T)).tocsr()
        self.I1_pol_all = self.P1_pol_all.dot(DN_ND.dot(self.space.E1_pol_all.T)).tocsr()
        
        self.I2_pol     = self.P2_pol.dot(ND_DN.dot(self.space.E2_pol.T)).tocsr()
        self.I2_pol_all = self.P2_pol_all.dot(ND_DN.dot(self.space.E2_pol_all.T)).tocsr()
       
        self.I3_pol     = self.P3_pol.dot(DD.dot(self.space.E3_pol.T)).tocsr()
        self.I3_pol_all = self.P3_pol_all.dot(DD.dot(self.space.E3_pol_all.T)).tocsr()

        # LU decompositions in poloidal plane
        self.I0_pol_LU = spa.linalg.splu(self.I0_pol.tocsc())
        self.I1_pol_LU = spa.linalg.splu(self.I1_pol.tocsc())
        self.I2_pol_LU = spa.linalg.splu(self.I2_pol.tocsc())
        self.I3_pol_LU = spa.linalg.splu(self.I3_pol.tocsc())
        
        self.I0_pol_all_LU = spa.linalg.splu(self.I0_pol_all.tocsc())
        self.I1_pol_all_LU = spa.linalg.splu(self.I1_pol_all.tocsc())
        self.I2_pol_all_LU = spa.linalg.splu(self.I2_pol_all.tocsc())
        self.I3_pol_all_LU = spa.linalg.splu(self.I3_pol_all.tocsc())
        
        self.I0_pol_T_LU = spa.linalg.splu(self.I0_pol.T.tocsc())
        self.I1_pol_T_LU = spa.linalg.splu(self.I1_pol.T.tocsc())
        self.I2_pol_T_LU = spa.linalg.splu(self.I2_pol.T.tocsc())
        self.I3_pol_T_LU = spa.linalg.splu(self.I3_pol.T.tocsc())
        
        self.I0_pol_all_T_LU = spa.linalg.splu(self.I0_pol_all.T.tocsc())
        
        # shift first radial interpolation point away from pole
        if self.space.polar == True:
            self.x_int[0][0] += 0.00001
        
        # collection of the point sets for different 3D projectors
        self.pts_PI_0  = [self.x_int[0]        , self.x_int[1]        , self.x_int[2]        ]
        
        self.pts_PI_11 = [self.pts[0].flatten(), self.x_int[1]        , self.x_int[2]        ]
        self.pts_PI_12 = [self.x_int[0]        , self.pts[1].flatten(), self.x_int[2]        ]
        self.pts_PI_13 = [self.x_int[0]        , self.x_int[1]        , self.pts[2].flatten()]
        
        self.pts_PI_21 = [self.x_int[0]        , self.pts[1].flatten(), self.pts[2].flatten()]
        self.pts_PI_22 = [self.pts[0].flatten(), self.x_int[1]        , self.pts[2].flatten()]
        self.pts_PI_23 = [self.pts[0].flatten(), self.pts[1].flatten(), self.x_int[2]        ]
        
        self.pts_PI_3  = [self.pts[0].flatten(), self.pts[1].flatten(), self.pts[2].flatten()]    
    
    
    # ========================================
    def assemble_approx_inv_V2(self):
        
        self.N_approx = [0, 0, 0]
        self.D_approx = [0, 0, 0]
        
        for a in range(3):
            self.N_approx[a] = np.copy(self.N[a].toarray())
            self.D_approx[a] = np.copy(self.D[a].toarray())
            
            if self.bc[a] == False:
                diagN = self.N[a].diagonal()
                diagD = self.D[a].diagonal()
                
                self.N_approx[a][:, :] = 0.
                self.D_approx[a][:, :] = 0.
                
                np.fill_diagonal(self.N_approx[a], diagN)
                np.fill_diagonal(self.D_approx[a], diagD)
                
            else:
                self.N_approx[a][np.abs(self.N_approx[a]) < (self.N_approx[a].max() - 0.01)] = 0.
                self.D_approx[a][np.abs(self.D_approx[a]) < (self.D_approx[a].max() - 0.01)] = 0.
                
            self.N_approx[a] = spa.csr_matrix(self.N_approx[a])
            self.D_approx[a] = spa.csr_matrix(self.D_approx[a])
                
        self.I0_approx = self.P0.dot(spa.kron(self.N_approx[0], spa.kron(self.N_approx[1], self.N_approx[2])).dot(self.tensor_space.E0.T)).tocsr()
        
        a = spa.kron(self.D_approx[0], spa.kron(self.N_approx[1], self.N_approx[2]))
        b = spa.kron(self.N_approx[0], spa.kron(self.D_approx[1], self.N_approx[2]))
        c = spa.kron(self.N_approx[0], spa.kron(self.N_approx[1], self.D_approx[2]))
        
        d = spa.bmat([[a, None, None], [None, b, None], [None, None, c]])
        
        self.I1_approx = self.P1.dot(d.dot(self.tensor_space.E1.T)).tocsr()
        
        a = spa.kron(self.N_approx[0], spa.kron(self.D_approx[1], self.D_approx[2]))
        b = spa.kron(self.D_approx[0], spa.kron(self.N_approx[1], self.D_approx[2]))
        c = spa.kron(self.D_approx[0], spa.kron(self.D_approx[1], self.N_approx[2]))
        
        d = spa.bmat([[a, None, None], [None, b, None], [None, None, c]])
        
        self.I2_approx = self.P2.dot(d.dot(self.tensor_space.E2.T)).tocsr()
        
        self.I3_approx = self.P3.dot(spa.kron(self.D_approx[0], spa.kron(self.D_approx[1], self.D_approx[2])).dot(self.tensor_space.E3.T))
        
        self.I1_inv_approx = spa.linalg.inv(self.I1_approx.tocsc()).tocsr()
        self.I2_inv_approx = spa.linalg.inv(self.I2_approx.tocsc()).tocsr()
        self.I3_inv_approx = spa.linalg.inv(self.I3_approx.tocsc()).tocsr()
    
    
    
    
    # ========================================
    def assemble_approx_inv(self, tol):
        
        # poloidal direction
        I0_pol_inv_approx = np.linalg.inv(self.I0_pol.toarray())
        I0_pol_inv_approx[np.abs(I0_pol_inv_approx) < tol] = 0.
        
        I1_pol_inv_approx = np.linalg.inv(self.I1_pol.toarray())
        I1_pol_inv_approx[np.abs(I1_pol_inv_approx) < tol] = 0.
        
        I2_pol_inv_approx = np.linalg.inv(self.I2_pol.toarray())
        I2_pol_inv_approx[np.abs(I2_pol_inv_approx) < tol] = 0.
        
        I3_pol_inv_approx = np.linalg.inv(self.I3_pol.toarray())
        I3_pol_inv_approx[np.abs(I3_pol_inv_approx) < tol] = 0.
        
        I0_pol_all_inv_approx = np.linalg.inv(self.I0_pol_all.toarray())
        I0_pol_all_inv_approx[np.abs(I0_pol_all_inv_approx) < tol] = 0.
        
        I0_pol_inv_approx = spa.csr_matrix(I0_pol_inv_approx)
        I1_pol_inv_approx = spa.csr_matrix(I1_pol_inv_approx)
        I2_pol_inv_approx = spa.csr_matrix(I2_pol_inv_approx)
        I3_pol_inv_approx = spa.csr_matrix(I3_pol_inv_approx)
        
        I0_pol_all_inv_approx = spa.csr_matrix(I0_pol_all_inv_approx)
        
        # toroidal direction
        N_inv_z_approx = np.copy(self.N_inv[2])
        D_inv_z_approx = np.copy(self.D_inv[2])

        N_inv_z_approx[np.abs(N_inv_z_approx) < tol] = 0.
        D_inv_z_approx[np.abs(D_inv_z_approx) < tol] = 0.
        
        N_inv_z_approx = spa.csr_matrix(N_inv_z_approx)
        D_inv_z_approx = spa.csr_matrix(D_inv_z_approx)

        # tensor-product poloidal x toroidal
        self.I0_inv_approx = spa.kron(I0_pol_inv_approx, N_inv_z_approx, format='csr')

        self.I1_inv_approx = spa.bmat([[spa.kron(I1_pol_inv_approx, N_inv_z_approx), None], [None, spa.kron(I0_pol_inv_approx, D_inv_z_approx)]], format='csr') 
        self.I2_inv_approx = spa.bmat([[spa.kron(I2_pol_inv_approx, D_inv_z_approx), None], [None, spa.kron(I3_pol_inv_approx, N_inv_z_approx)]], format='csr')
        
        self.I3_inv_approx = spa.kron(I3_pol_inv_approx, D_inv_z_approx, format='csr')
        
        self.I0_all_inv_approx = spa.kron(I0_pol_all_inv_approx, N_inv_z_approx, format='csr')
    
    
        
    # ========================================    
    def getpts_for_PI(self, comp):
        """
        Get the needed point sets for a given projector.
        
        Parameters
        ----------
        comp: int
            which projector, one of (0, 11, 12, 13, 21, 22, 23, 3).
        
        Returns
        -------
        pts_PI : list of 1d arrays 
            the 1D point sets.
        """
        
        if   comp == 0:
            pts_PI = self.pts_PI_0
        
        elif comp == 11:
            pts_PI = self.pts_PI_11 
        elif comp == 12:
            pts_PI = self.pts_PI_12
        elif comp == 13:
            pts_PI = self.pts_PI_13
            
        elif comp == 21:
            pts_PI = self.pts_PI_21 
        elif comp == 22:
            pts_PI = self.pts_PI_22
        elif comp == 23:
            pts_PI = self.pts_PI_23
        
        elif comp == 3:
            pts_PI = self.pts_PI_3
            
        else:
            raise ValueError ("wrong projector specified")

        return pts_PI
    
    
    # ======================================        
    def eval_for_PI(self, comp, fun, eval_kind):
        """
        Evaluates the callable "fun" at the points corresponding to the projector, and returns the result as 3d array "mat_f".
            
        Parameters
        ----------
        comp: int
            which projector, one of (0, 11, 12, 13, 21, 22, 23, 3).
            
        fun : callable
            the function fun(eta1, eta2, eta3) to project
            
        eval_kind : string
            function evaluation at interpolation/quadrature points ('meshgrid', 'tensor_product' or point-wise)
               
        Returns
        -------
        mat_f : 3d array
            function evaluated on a 3d meshgrid contstructed from the 1d point sets.
        """
        
        # get intepolation and quadrature points
        pts_PI = self.getpts_for_PI(comp)
        
        # number of evaluation points in each direction
        n_pts  = [pts_PI[0].size, pts_PI[1].size, pts_PI[2].size]

        # array of evaluated function
        mat_f  = np.empty((n_pts[0], n_pts[1], n_pts[2]), dtype=float)
        
        # external function call if a callable is passed
        if callable(fun):
            
            # create a meshgrid and evaluate function on point set
            if eval_kind == 'meshgrid':
                pts1, pts2, pts3 = np.meshgrid(pts_PI[0], pts_PI[1], pts_PI[2], indexing='ij')
                mat_f[:, :, :]   = fun(pts1, pts2, pts3)
                
            # tensor-product evaluation is done by input function
            elif eval_kind == 'tensor_product':
                mat_f[:, :, :] = fun(pts_PI[0], pts_PI[1], pts_PI[2])
                
            # point-wise evaluation
            else:
                for i1 in range(pts_PI[0].size):
                    for i2 in range(pts_PI[1].size):
                        for i3 in range(pts_PI[2].size):
                            mat_f[i1, i2, i3] = fun(pts_PI[0][i1], pts_PI[1][i2], pts_PI[2][i3])
            
        
        # internal function call
        else:
            print('no internal 3D function implemented!')
       
        return mat_f
    
    
    # ======================================
    def solve_V0(self, include_bc, rhs):
        
        # solve system
        if include_bc == True:
            rhs = rhs.reshape(self.P0_pol_all.shape[0], self.space.NbaseN[2])
            coeffs = self.N_LU[2].solve(self.I0_pol_all_LU.solve(rhs).T).T
        else:
            rhs = rhs.reshape(self.P0_pol.shape[0], self.space.NbaseN[2])
            coeffs = self.N_LU[2].solve(self.I0_pol_LU.solve(rhs).T).T
            
        return coeffs.flatten()
    
    # ======================================
    def solve_V1(self, include_bc, rhs):
        
        # solve system
        if include_bc == True:
            rhs1 = rhs[:self.P1_pol_all.shape[0]*self.space.NbaseN[2] ].reshape(self.P1_pol_all.shape[0], self.space.NbaseN[2])
            rhs3 = rhs[ self.P1_pol_all.shape[0]*self.space.NbaseN[2]:].reshape(self.P0_pol_all.shape[0], self.space.NbaseD[2])

            coeffs1 = self.N_LU[2].solve(self.I1_pol_all_LU.solve(rhs1).T).T
            coeffs3 = self.D_LU[2].solve(self.I0_pol_all_LU.solve(rhs3).T).T
        else:
            rhs1 = rhs[:self.P1_pol.shape[0]*self.space.NbaseN[2] ].reshape(self.P1_pol.shape[0], self.space.NbaseN[2])
            rhs3 = rhs[ self.P1_pol.shape[0]*self.space.NbaseN[2]:].reshape(self.P0_pol.shape[0], self.space.NbaseD[2])

            coeffs1 = self.N_LU[2].solve(self.I1_pol_LU.solve(rhs1).T).T
            coeffs3 = self.D_LU[2].solve(self.I0_pol_LU.solve(rhs3).T).T
        
        return np.concatenate((coeffs1.flatten(), coeffs3.flatten()))
    
    # ======================================
    def solve_V2(self, include_bc, rhs):
        
        # solve system
        if include_bc == True:
            rhs1 = rhs[:self.P2_pol_all.shape[0]*self.space.NbaseD[2] ].reshape(self.P2_pol_all.shape[0], self.space.NbaseD[2])
            rhs3 = rhs[ self.P2_pol_all.shape[0]*self.space.NbaseD[2]:].reshape(self.P3_pol_all.shape[0], self.space.NbaseN[2])

            coeffs1 = self.D_LU[2].solve(self.I2_pol_all_LU.solve(rhs1).T).T
            coeffs3 = self.N_LU[2].solve(self.I3_pol_all_LU.solve(rhs3).T).T
        else:
            rhs1 = rhs[:self.P2_pol.shape[0]*self.space.NbaseN[2] ].reshape(self.P2_pol.shape[0], self.space.NbaseD[2])
            rhs3 = rhs[ self.P2_pol.shape[0]*self.space.NbaseN[2]:].reshape(self.P3_pol.shape[0], self.space.NbaseN[2])

            coeffs1 = self.D_LU[2].solve(self.I2_pol_LU.solve(rhs1).T).T
            coeffs3 = self.N_LU[2].solve(self.I3_pol_LU.solve(rhs3).T).T
        
        return np.concatenate((coeffs1.flatten(), coeffs3.flatten()))
        
    # ======================================
    def solve_V3(self, include_bc, rhs):
        
        # solve system
        if include_bc == True:
            rhs = rhs.reshape(self.P3_pol_all.shape[0], self.space.NbaseD[2])
            coeffs = self.D_LU[2].solve(self.I3_pol_all_LU.solve(rhs).T).T
        else:
            rhs = rhs.reshape(self.P3_pol.shape[0], self.space.NbaseD[2])
            coeffs = self.D_LU[2].solve(self.I3_pol_LU.solve(rhs).T).T
            
        return coeffs.flatten()
        
    # ======================================
    def apply_IinvT_V0(self, rhs, include_bc=False):
        
        if include_bc == False:
            rhs = rhs.reshape(self.P0_pol.shape[0], self.space.NbaseN[2])
            return self.I0_pol_T_LU.solve(self.N_T_LU[2].solve(rhs.T).T).flatten()
        else:
            rhs = rhs.reshape(self.P0_pol_all.shape[0], self.space.NbaseN[2])
            return self.I0_pol_all_T_LU.solve(self.N_T_LU[2].solve(rhs.T).T).flatten()
        
    
    # ======================================
    def apply_IinvT_V1(self, rhs):
        
        rhs1 = rhs[:self.P1_pol.shape[0]*self.space.NbaseN[2] ].reshape(self.P1_pol.shape[0], self.space.NbaseN[2])
        rhs3 = rhs[ self.P1_pol.shape[0]*self.space.NbaseN[2]:].reshape(self.P0_pol.shape[0], self.space.NbaseD[2])
        
        rhs1 = self.I1_pol_T_LU.solve(self.N_T_LU[2].solve(rhs1.T).T)
        rhs3 = self.I0_pol_T_LU.solve(self.D_T_LU[2].solve(rhs3.T).T)
        
        return np.concatenate((rhs1.flatten(), rhs3.flatten()))
    
    # ======================================
    def apply_IinvT_V2(self, rhs):
                
        rhs1 = rhs[:self.P2_pol.shape[0]*self.space.NbaseD[2] ].reshape(self.P2_pol.shape[0], self.space.NbaseD[2])
        rhs3 = rhs[ self.P2_pol.shape[0]*self.space.NbaseD[2]:].reshape(self.P3_pol.shape[0], self.space.NbaseN[2])
        
        rhs1 = self.I2_pol_T_LU.solve(self.D_T_LU[2].solve(rhs1.T).T)
        rhs3 = self.I3_pol_T_LU.solve(self.N_T_LU[2].solve(rhs3.T).T)
        
        return np.concatenate((rhs1.flatten(), rhs3.flatten()))
    
    # ======================================
    def apply_IinvT_V3(self, rhs):
        
        rhs = rhs.reshape(self.P3_pol.shape[0], self.space.NbaseD[2])
        
        return self.I3_pol_T_LU.solve(self.D_T_LU[2].solve(rhs.T).T).flatten()  
    
    
    # ======================================        
    def pi_0(self, fun, include_bc=True, eval_kind='meshgrid', interp=True):
        
        #  ==== evaluate on tensor-product grid ====
        rhs = self.eval_for_PI(0, fun, eval_kind)
        
        # ====== apply extraction operator =========
        if include_bc == True:
            rhs = self.P0_all.dot(rhs.flatten())
        else:
            rhs = self.P0.dot(rhs.flatten())
        
        # ====== solve for coefficients ============
        if interp == True:
            return self.solve_V0(include_bc, rhs)
        else:
            return rhs
    
    
    # ======================================        
    def pi_1(self, fun, include_bc=True, eval_kind='meshgrid', interp=True):
        
        # ====== integrate along 1-direction =======
        n1   = self.pts_PI_11[0].size//self.n_quad[0] 
        n2   = self.pts_PI_11[1].size 
        n3   = self.pts_PI_11[2].size
        
        pts  = self.eval_for_PI(11, fun[0], eval_kind)
        
        rhs1 = np.empty((self.subs[0].size, n2, n3), dtype=float)
        ker_glob.kernel_int_3d_eta1(self.subs[0], self.subs_cum[0], self.wts[0], pts.reshape(n1, self.n_quad[0], n2, n3), rhs1)
            
        # ====== integrate along 2-direction =======
        n1   = self.pts_PI_12[0].size
        n2   = self.pts_PI_12[1].size//self.n_quad[1]  
        n3   = self.pts_PI_12[2].size
        
        pts  = self.eval_for_PI(12, fun[1], eval_kind)
        
        rhs2 = np.empty((n1, self.subs[1].size, n3), dtype=float)
        ker_glob.kernel_int_3d_eta2(self.subs[1], self.subs_cum[1], self.wts[1], pts.reshape(n1, n2, self.n_quad[1], n3), rhs2)
        
        # ====== integrate along 3-direction =======
        n1   = self.pts_PI_13[0].size
        n2   = self.pts_PI_13[1].size  
        n3   = self.pts_PI_13[2].size//self.n_quad[2]
        
        pts  = self.eval_for_PI(13, fun[2], eval_kind)
        
        rhs3 = np.empty((n1, n2, self.subs[2].size), dtype=float)
        ker_glob.kernel_int_3d_eta3(self.subs[2], self.subs_cum[2], self.wts[2], pts.reshape(n1, n2, n3, self.n_quad[2]), rhs3)
        
        # ====== apply extraction operator =========
        if include_bc == True:
            rhs = self.P1_all.dot(np.concatenate((rhs1.flatten(), rhs2.flatten(), rhs3.flatten())))
        else:
            rhs = self.P1.dot(np.concatenate((rhs1.flatten(), rhs2.flatten(), rhs3.flatten())))
            
        # ====== solve for coefficients ============
        if interp == True:
            return self.solve_V1(include_bc, rhs)
        else:
            return rhs
            

    # ======================================        
    def pi_2(self, fun, include_bc=True, eval_kind='meshgrid', interp=True):
        
        # ====== integrate in 2-3-plane =======
        n1   = self.pts_PI_21[0].size
        n2   = self.pts_PI_21[1].size//self.n_quad[1] 
        n3   = self.pts_PI_21[2].size//self.n_quad[2]
        
        pts  = self.eval_for_PI(21, fun[0], eval_kind)
        
        rhs1 = np.empty((n1, self.subs[1].size, self.subs[2].size), dtype=float)
        ker_glob.kernel_int_3d_eta2_eta3(self.subs[1], self.subs[2], self.subs_cum[1], self.subs_cum[2], self.wts[1], self.wts[2], pts.reshape(n1, n2, self.n_quad[1], n3, self.n_quad[2]), rhs1)   
        # ====== integrate in 1-3-plane =======
        n1   = self.pts_PI_22[0].size//self.n_quad[0] 
        n2   = self.pts_PI_22[1].size
        n3   = self.pts_PI_22[2].size//self.n_quad[2]
        
        pts  = self.eval_for_PI(22, fun[1], eval_kind)
        
        rhs2 = np.empty((self.subs[0].size, n2, self.subs[2].size), dtype=float)
        ker_glob.kernel_int_3d_eta1_eta3(self.subs[0], self.subs[2], self.subs_cum[0], self.subs_cum[2], self.wts[0], self.wts[2], pts.reshape(n1, self.n_quad[0], n2, n3, self.n_quad[2]), rhs2)  
        # ====== integrate in 1-2-plane =======
        n1   = self.pts_PI_23[0].size//self.n_quad[0] 
        n2   = self.pts_PI_23[1].size//self.n_quad[1]
        n3   = self.pts_PI_23[2].size
        
        pts  = self.eval_for_PI(23, fun[2], eval_kind)
        
        rhs3 = np.empty((self.subs[0].size, self.subs[1].size, n3), dtype=float)
        ker_glob.kernel_int_3d_eta1_eta2(self.subs[0], self.subs[1], self.subs_cum[0], self.subs_cum[1], self.wts[0], self.wts[1], pts.reshape(n1, self.n_quad[0], n2, self.n_quad[1], n3), rhs3)
        
        # ====== apply extraction operator =========
        if include_bc == True:
            rhs = self.P2_all.dot(np.concatenate((rhs1.flatten(), rhs2.flatten(), rhs3.flatten())))
        else:
            rhs = self.P2.dot(np.concatenate((rhs1.flatten(), rhs2.flatten(), rhs3.flatten())))
        
        # ====== solve for coefficients ============
        if interp == True:
            return self.solve_V2(include_bc, rhs)
        else:
            return rhs
    
    
    # ======================================        
    def pi_3(self, fun, include_bc=True, eval_kind='meshgrid', interp=True):
        
        n1  = self.pts_PI_3[0].size//self.n_quad[0] 
        n2  = self.pts_PI_3[1].size//self.n_quad[1]
        n3  = self.pts_PI_3[2].size//self.n_quad[2]
        
        pts = self.eval_for_PI(3, fun, eval_kind)
        
        rhs = np.empty((self.subs[0].size, self.subs[1].size, self.subs[2].size), dtype=float)
        ker_glob.kernel_int_3d_eta1_eta2_eta3(self.subs[0], self.subs[1], self.subs[2], self.subs_cum[0], self.subs_cum[1], self.subs_cum[2], self.wts[0], self.wts[1], self.wts[2], pts.reshape(n1, self.n_quad[0], n2, self.n_quad[1], n3, self.n_quad[2]), rhs)
            
        # ====== apply extraction operator =========
        if include_bc == True:
            rhs = self.P3_all.dot(rhs.flatten())
        else:
            rhs = self.P3.dot(rhs.flatten())
        
        # ====== solve for coefficients ============
        if interp == True:
            return self.solve_V3(include_bc, rhs)
        else:
            return rhs