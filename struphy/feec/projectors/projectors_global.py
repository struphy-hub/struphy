# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Classes for projectors in 1D, 2D and 3D based on global spline interpolation and histopolation.
"""

import numpy as np
import scipy.sparse as spa

import struphy.feec.bsplines as bsp

import struphy.feec.projectors.kernels_projectors_global as ker_glob

from   struphy.linear_algebra.linalg_kron import kron_lusolve_2d
from   struphy.linear_algebra.linalg_kron import kron_lusolve_3d
from   struphy.linear_algebra.linalg_kron import kron_matvec_3d
from   struphy.linear_algebra.linalg_kron import kron_matvec_3d_1, kron_matvec_3d_2, kron_matvec_3d_3
from   struphy.linear_algebra.linalg_kron import kron_matvec_3d_23, kron_matvec_3d_13, kron_matvec_3d_12




# ======================= 1d ====================================
class projectors_global_1d:
    """
    Global commuting projectors pi_0 and pi_1 in 1d.
    
    Parameters:
    -----------
    spline_space : Spline_space_1d
        A 1d space of B-splines and corresponding D-splines.
        
    n_quad : int
        Number of Gauss-Legendre quadrature points per integration interval for histopolation.

    Attributes:
    -----------
    space : Spline_space_1d
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
        
        # Q-matrix for integration of dofs_1
        self.Q = np.zeros((self.space.NbaseD, self.wts.size), dtype=float)
        accul = 0
        for i in range(self.Q.shape[0]):
            self.Q[i, accul*n_quad : (accul + self.subs[i])*n_quad] = self.wts[accul : accul + self.subs[i], :].flatten()
            accul += self.subs[i]
        self.Q = spa.csr_matrix(self.Q)

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

    def dofs_1_mat(self, fun):
        '''
        Same as dofs_1, but using matrix instead of kernel.
        '''
        
        # evaluate function at quadrature points
        mat_f  = fun(self.pts.flatten())
                
        return self.Q.dot(mat_f)
    
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
                # There are two evaluation routines at the moment: spline_evaluation_1d (pyccel) and Spline_space_1d (slow).
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

        self.Q1 = proj_1d[0].Q
        self.Q2 = proj_1d[1].Q
        self.Q3 = proj_1d[2].Q


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
    def dofs_mat(self, comp, mat_f):
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
            
            rhs = kron_matvec_3d_1(self.Q1, mat_f)

        elif comp=='12':
            rhs = np.empty( (self.n1, self.d2, self.n3) )
            
            rhs = kron_matvec_3d_2(self.Q2, mat_f)

        elif comp=='13':
            rhs = np.empty( (self.n1, self.n2, self.d3) )
            
            rhs = kron_matvec_3d_3(self.Q3, mat_f)

        elif comp=='21':
            rhs = np.empty( (self.n1, self.d2, self.d3) )
            
            rhs = kron_matvec_3d_23([self.Q2, self.Q3], mat_f)

        elif comp=='22':
            rhs = np.empty( (self.d1, self.n2, self.d3) )
            
            rhs = kron_matvec_3d_13([self.Q1, self.Q3], mat_f)

        elif comp=='23':
            rhs = np.empty( (self.d1, self.d2, self.n3) )
            
            rhs = kron_matvec_3d_12([self.Q1, self.Q2], mat_f)

        elif comp=='3':
            rhs = np.empty( (self.d1, self.d2, self.d3) )
            
            rhs = kron_matvec_3d([self.Q1, self.Q2, self.Q3], mat_f)

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

        if comp=='0':
            rhs = mat_dofs

        elif comp=='11':
            assert mat_dofs.shape == (self.d1, self.n2, self.n3)
            rhs = np.empty( (self.ne1, self.nq1, self.n2, self.n3) )

            ker_glob.kernel_int_3d_eta1_transpose(self.subs1, self.subs_cum1, self.wts1,
                                                  mat_dofs, rhs)

            rhs = rhs.reshape(self.ne1 * self.nq1, self.n2, self.n3)

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
    def dofs_T_mat(self, comp, mat_dofs):
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
        rhs : 3d numpy array 
            The degrees of freedom sigma_ijk.
        '''

        if comp=='0':
            rhs = mat_dofs

        elif comp=='11':
            rhs = np.empty( (self.d1, self.n2, self.n3) )
            
            rhs = kron_matvec_3d_1(self.Q1.T, mat_dofs)

        elif comp=='12':
            rhs = np.empty( (self.n1, self.d2, self.n3) )
            
            rhs = kron_matvec_3d_2(self.Q2.T, mat_dofs)

        elif comp=='13':
            rhs = np.empty( (self.n1, self.n2, self.d3) )
            
            rhs = kron_matvec_3d_3(self.Q3.T, mat_dofs)

        elif comp=='21':
            rhs = np.empty( (self.n1, self.d2, self.d3) )
            
            rhs = kron_matvec_3d_23([self.Q2.T, self.Q3.T], mat_dofs)

        elif comp=='22':
            rhs = np.empty( (self.d1, self.n2, self.d3) )
            
            rhs = kron_matvec_3d_13([self.Q1.T, self.Q3.T], mat_dofs)

        elif comp=='23':
            rhs = np.empty( (self.d1, self.d2, self.n3) )
            
            rhs = kron_matvec_3d_12([self.Q1.T, self.Q2.T], mat_dofs)

        elif comp=='3':
            rhs = np.empty( (self.d1, self.d2, self.d3) )
            
            rhs = kron_matvec_3d([self.Q1.T, self.Q2.T, self.Q3.T], mat_dofs)

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



# ===============================================================
class projectors_global_3d:
    
    def __init__(self, tensor_space):
        
        # assemble extraction operators P^k for degrees of freedom
        
        
        # ----------- standard tensor-product splines in eta_1 x eta_2 plane -----------
        if tensor_space.ck == -1:
            
            n1, n2 = tensor_space.NbaseN[:2]
            d1, d2 = tensor_space.NbaseD[:2]
            
            # with boundary dofs
            self.P0_pol = spa.identity(n1*n2        , dtype=float, format='csr')
            self.P1_pol = spa.identity(d1*n2 + n1*d2, dtype=float, format='csr')
            self.P2_pol = spa.identity(n1*d2 + d1*n2, dtype=float, format='csr')
            self.P3_pol = spa.identity(d1*d2        , dtype=float, format='csr')
            
            # without boundary dofs
            self.P0_pol_0 = tensor_space.B0_pol.dot(self.P0_pol).tocsr()
            self.P1_pol_0 = tensor_space.B1_pol.dot(self.P1_pol).tocsr()
            self.P2_pol_0 = tensor_space.B2_pol.dot(self.P2_pol).tocsr()
            self.P3_pol_0 = tensor_space.B3_pol.dot(self.P3_pol).tocsr()
        # ---------------------------------------------------------------------------------
        
        
        # ----------------- C^k polar splines in eta_1 x eta_2 plane ----------------------
        else:
            
            # with boundary dofs
            self.P0_pol = tensor_space.polar_splines.P0.copy()
            self.P1_pol = tensor_space.polar_splines.P1C.copy()
            self.P2_pol = tensor_space.polar_splines.P1D.copy()
            self.P3_pol = tensor_space.polar_splines.P2.copy()
            
            # without boundary dofs
            self.P0_pol_0 = tensor_space.B0_pol.dot(self.P0_pol).tocsr()
            self.P1_pol_0 = tensor_space.B1_pol.dot(self.P1_pol).tocsr()
            self.P2_pol_0 = tensor_space.B2_pol.dot(self.P2_pol).tocsr()
            self.P3_pol_0 = tensor_space.B3_pol.dot(self.P3_pol).tocsr()
        # ---------------------------------------------------------------------------------     
            

        # 3D operators: with boundary dofs (3rd dimension MUST be periodic)
        self.P0 =            self.P0_pol.copy()
        self.P1 = spa.bmat([[self.P1_pol, None], [None, self.P0_pol]], format='csr')
        self.P2 = spa.bmat([[self.P2_pol, None], [None, self.P3_pol]], format='csr')
        self.P3 =            self.P3_pol.copy()
        
        self.P0 = spa.kron(self.P0, spa.identity(tensor_space.NbaseN[2]), format='csr')
        self.P1 = spa.kron(self.P1, spa.identity(tensor_space.NbaseN[2]), format='csr')
        self.P2 = spa.kron(self.P2, spa.identity(tensor_space.NbaseN[2]), format='csr')
        self.P3 = spa.kron(self.P3, spa.identity(tensor_space.NbaseN[2]), format='csr')
        
        # 3D operators: without boundary dofs (3rd dimension MUST be periodic)
        self.P0_0 = tensor_space.B0.dot(self.P0).tocsr()
        self.P1_0 = tensor_space.B1.dot(self.P1).tocsr()
        self.P2_0 = tensor_space.B2.dot(self.P2).tocsr()
        self.P3_0 = tensor_space.B3.dot(self.P3).tocsr()
            
        #if tensor_space.ck == 1:
        #    
        #    # blocks of I0 matrix
        #    self.I0_11 = spa.kron(self.projectors_1d[0].N[:2, :2], self.projectors_1d[1].N)
        #    self.I0_11 = tensor_space.polar_splines.P0_11.dot(self.I0_11.dot(tensor_space.polar_splines.E0_11.T)).tocsr()
#
        #    self.I0_12 = spa.kron(self.projectors_1d[0].N[:2, 2:], self.projectors_1d[1].N)
        #    self.I0_12 = tensor_space.polar_splines.P0_11.dot(self.I0_12).tocsr()
#
        #    self.I0_21 = spa.kron(self.projectors_1d[0].N[2:, :2], self.projectors_1d[1].N)
        #    self.I0_21 = self.I0_21.dot(tensor_space.polar_splines.E0_11.T).tocsr()
#
        #    self.I0_22 = spa.kron(self.projectors_1d[0].N[2:, 2:], self.projectors_1d[1].N, format='csr')
        #    
        #    self.I0_22_LUs = [spa.linalg.splu(self.projectors_1d[0].N[2:, 2:].tocsc()), self.projectors_1d[1].N_LU]
    
        # 2D interpolation/histopolation matrices in poloidal plane
        II = spa.kron(tensor_space.spaces[0].projectors.I, tensor_space.spaces[1].projectors.I, format='csr')
        HI = spa.kron(tensor_space.spaces[0].projectors.H, tensor_space.spaces[1].projectors.I, format='csr')
        IH = spa.kron(tensor_space.spaces[0].projectors.I, tensor_space.spaces[1].projectors.H, format='csr')
        HH = spa.kron(tensor_space.spaces[0].projectors.H, tensor_space.spaces[1].projectors.H, format='csr')
        
        HI_IH = spa.bmat([[HI, None], [None, IH]], format='csr')
        IH_HI = spa.bmat([[IH, None], [None, HI]], format='csr')
        
        # including boundary splines
        self.I0_pol = self.P0_pol.dot(   II.dot(tensor_space.E0_pol.T)).tocsr()
        self.I1_pol = self.P1_pol.dot(HI_IH.dot(tensor_space.E1_pol.T)).tocsr()
        self.I2_pol = self.P2_pol.dot(IH_HI.dot(tensor_space.E2_pol.T)).tocsr()
        self.I3_pol = self.P3_pol.dot(   HH.dot(tensor_space.E3_pol.T)).tocsr()
        
        # without boundary splines
        self.I0_pol_0 = self.P0_pol_0.dot(   II.dot(tensor_space.E0_pol_0.T)).tocsr()
        self.I1_pol_0 = self.P1_pol_0.dot(HI_IH.dot(tensor_space.E1_pol_0.T)).tocsr()
        self.I2_pol_0 = self.P2_pol_0.dot(IH_HI.dot(tensor_space.E2_pol_0.T)).tocsr()
        self.I3_pol_0 = self.P3_pol_0.dot(   HH.dot(tensor_space.E3_pol_0.T)).tocsr()
        
        # LU decompositions in poloidal plane (including boundary splines)
        self.I0_pol_LU = spa.linalg.splu(self.I0_pol.tocsc())
        self.I1_pol_LU = spa.linalg.splu(self.I1_pol.tocsc())
        self.I2_pol_LU = spa.linalg.splu(self.I2_pol.tocsc())
        self.I3_pol_LU = spa.linalg.splu(self.I3_pol.tocsc())
        
        # LU decompositions in poloidal plane (without boundary splines)
        self.I0_pol_0_LU = spa.linalg.splu(self.I0_pol_0.tocsc())
        self.I1_pol_0_LU = spa.linalg.splu(self.I1_pol_0.tocsc())
        self.I2_pol_0_LU = spa.linalg.splu(self.I2_pol_0.tocsc())
        self.I3_pol_0_LU = spa.linalg.splu(self.I3_pol_0.tocsc())
        
        self.I0_pol_0_T_LU = spa.linalg.splu(self.I0_pol_0.T.tocsc())
        self.I1_pol_0_T_LU = spa.linalg.splu(self.I1_pol_0.T.tocsc())
        self.I2_pol_0_T_LU = spa.linalg.splu(self.I2_pol_0.T.tocsc())
        self.I3_pol_0_T_LU = spa.linalg.splu(self.I3_pol_0.T.tocsc())
        
        # get 1D interpolation points
        x_i1 = tensor_space.spaces[0].projectors.x_int.copy()
        x_i2 = tensor_space.spaces[1].projectors.x_int.copy()
            
        # get 1D quadrature points
        x_q1 = tensor_space.spaces[0].projectors.pts.flatten()
        x_q2 = tensor_space.spaces[1].projectors.pts.flatten()
        
        # get 1D quadrature weight matrices
        self.Q1 = tensor_space.spaces[0].projectors.Q
        self.Q2 = tensor_space.spaces[1].projectors.Q
        
        # 1D interpolation/histopolation points and matrices in third direction
        if tensor_space.dim == 3:
            
            x_i3 = tensor_space.spaces[2].projectors.x_int
            x_q3 = tensor_space.spaces[2].projectors.pts.flatten()
            
            self.Q3 = tensor_space.spaces[2].projectors.Q
            
            self.I_tor = tensor_space.spaces[2].projectors.I
            self.H_tor = tensor_space.spaces[2].projectors.H
            
            self.I_tor_LU = tensor_space.spaces[2].projectors.I_LU
            self.H_tor_LU = tensor_space.spaces[2].projectors.H_LU
            
            self.I_tor_T_LU = tensor_space.spaces[2].projectors.I_T_LU
            self.H_tor_T_LU = tensor_space.spaces[2].projectors.H_T_LU
            
        else:
            
            if tensor_space.basis_tor == 'r':
                
                if   tensor_space.n_tor == 0:
                    x_i3 = np.array([0., 0.])
                    x_q3 = np.array([0., 0.])
                    
                elif tensor_space.n_tor > 0:
                    x_i3 = np.array([0.25/tensor_space.n_tor, 1.])
                    x_q3 = np.array([0.25/tensor_space.n_tor, 1.])
                    
                else:
                    x_i3 = np.array([0.75/(-tensor_space.n_tor), 1.])
                    x_q3 = np.array([0.75/(-tensor_space.n_tor), 1.])
                    
            else:
                x_i3 = np.array([0.])
                x_q3 = np.array([0.])
            
            self.Q3 = spa.identity(tensor_space.NbaseN[2], format='csr')
            
            self.I_tor = spa.identity(tensor_space.NbaseN[2], format='csr')
            self.H_tor = spa.identity(tensor_space.NbaseN[2], format='csr')
            
            self.I_tor_LU = spa.linalg.splu(self.I_tor.tocsc())
            self.H_tor_LU = spa.linalg.splu(self.H_tor.tocsc())
            
            self.I_tor_T_LU = spa.linalg.splu(self.I_tor.T.tocsc())
            self.H_tor_T_LU = spa.linalg.splu(self.H_tor.T.tocsc())
            
        
        # collection of the point sets for different projectors in poloidal plane
        self.pts_PI_0  = [x_i1, x_i2, x_i3]
        
        self.pts_PI_11 = [x_q1, x_i2, x_i3]
        self.pts_PI_12 = [x_i1, x_q2, x_i3]
        self.pts_PI_13 = [x_i1, x_i2, x_q3]
        
        self.pts_PI_21 = [x_i1, x_q2, x_q3]
        self.pts_PI_22 = [x_q1, x_i2, x_q3]
        self.pts_PI_23 = [x_q1, x_q2, x_i3]
        
        self.pts_PI_3  = [x_q1, x_q2, x_q3]
        
    
    
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
            kind of function evaluation at interpolation/quadrature points ('meshgrid' or 'tensor_product', point-wise else)
               
        Returns
        -------
        mat_f : 3d array
            function evaluated on a 3d meshgrid contstructed from the 1d point sets.
        """
        
        assert callable(fun)
        
        # get intepolation and quadrature points
        pts_PI = self.getpts_for_PI(comp)
        
        # array of evaluated function
        mat_f = np.empty((pts_PI[0].size, pts_PI[1].size, pts_PI[2].size), dtype=float)
        
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
       
        return mat_f
    
    
    ## ======================================
    #def assemble_Schur0_inv(self):
    #    
    #    n1 = self.pts_PI_0[0].size
    #    n2 = self.pts_PI_0[1].size
    #    
    #    # apply (I0_22) to each column
    #    self.S0 = np.zeros(((n1 - 2)*n2, 3), dtype=float)
    #    
    #    for i in range(3):
    #        self.S0[:, i] = kron_lusolve_2d(self.I0_22_LUs, self.I0_21[:, i].toarray().reshape(n1 - 2, n2)).flatten()
    #        
    #    # 3 x 3 matrix
    #    self.S0 = np.linalg.inv(self.I0_11.toarray() - self.I0_12.toarray().dot(self.S0))
    #    
    #    
    ## ======================================
    #def I0_inv(self, rhs, include_bc):
    #    
    #    n1 = self.pts_PI_0[0].size
    #    n2 = self.pts_PI_0[1].size
    #    
    #    if include_bc:
    #        rhs1 = rhs[:3]
    #        rhs2 = rhs[3:].reshape(n1 - 2, n2)
    #        
    #        # solve pure 3x3 polar contribution
    #        out1 = self.S0.dot(rhs1)
    #        
    #        # solve pure tensor-product contribution I0_22^(-1)*rhs2
    #        out2 = kron_lusolve_2d(self.I0_22_LUs, rhs2)
    #        
    #        # solve for polar coefficients
    #        out1 -= self.S0.dot(self.I0_12.dot(out2.flatten()))
    #        
    #        # solve for tensor-product coefficients
    #        out2  = out2 - kron_lusolve_2d(self.I0_22_LUs, self.I0_21.dot(self.S0.dot(rhs1)).reshape(n1 - 2, n2)) + kron_lusolve_2d(self.I0_22_LUs, self.I0_21.dot(self.S0.dot(self.I0_12.dot(out2.flatten()))).reshape(n1 - 2, n2)) 
    #        
    #    return np.concatenate((out1, out2.flatten()))
    
    
    
    # ======================================
    def solve_V0(self, dofs_0, include_bc):
        
        # with boundary splines
        if include_bc:
            dofs_0 = dofs_0.reshape(self.P0_pol.shape[0], self.I_tor.shape[0])
            coeffs = self.I_tor_LU.solve(self.I0_pol_LU.solve(dofs_0).T).T
        
        # without boundary splines
        else:
            dofs_0 = dofs_0.reshape(self.P0_pol_0.shape[0], self.I_tor.shape[0])
            coeffs = self.I_tor_LU.solve(self.I0_pol_0_LU.solve(dofs_0).T).T
            
        return coeffs.flatten()
    
    # ======================================
    def solve_V1(self, dofs_1, include_bc):
        
        # with boundary splines
        if include_bc:
            dofs_11 = dofs_1[:self.P1_pol.shape[0]*self.I_tor.shape[0] ].reshape(self.P1_pol.shape[0], self.I_tor.shape[0])
            dofs_12 = dofs_1[ self.P1_pol.shape[0]*self.I_tor.shape[0]:].reshape(self.P0_pol.shape[0], self.H_tor.shape[0])

            coeffs1 = self.I_tor_LU.solve(self.I1_pol_LU.solve(dofs_11).T).T
            coeffs2 = self.H_tor_LU.solve(self.I0_pol_LU.solve(dofs_12).T).T
        
        # without boundary splines
        else:
            dofs_11 = dofs_1[:self.P1_pol_0.shape[0]*self.I_tor.shape[0] ].reshape(self.P1_pol_0.shape[0], self.I_tor.shape[0])
            dofs_12 = dofs_1[ self.P1_pol_0.shape[0]*self.I_tor.shape[0]:].reshape(self.P0_pol_0.shape[0], self.H_tor.shape[0])

            coeffs1 = self.I_tor_LU.solve(self.I1_pol_0_LU.solve(dofs_11).T).T
            coeffs2 = self.H_tor_LU.solve(self.I0_pol_0_LU.solve(dofs_12).T).T
        
        return np.concatenate((coeffs1.flatten(), coeffs2.flatten()))
    
    # ======================================
    def solve_V2(self, dofs_2, include_bc):
        
        # with boundary splines
        if include_bc:
            dofs_21 = dofs_2[:self.P2_pol.shape[0]*self.H_tor.shape[0] ].reshape(self.P2_pol.shape[0], self.H_tor.shape[0])
            dofs_22 = dofs_2[ self.P2_pol.shape[0]*self.H_tor.shape[0]:].reshape(self.P3_pol.shape[0], self.I_tor.shape[0])

            coeffs1 = self.H_tor_LU.solve(self.I2_pol_LU.solve(dofs_21).T).T
            coeffs2 = self.I_tor_LU.solve(self.I3_pol_LU.solve(dofs_22).T).T
        
        # without boundary splines
        else:
            dofs_21 = dofs_2[:self.P2_pol_0.shape[0]*self.H_tor.shape[0] ].reshape(self.P2_pol_0.shape[0], self.H_tor.shape[0])
            dofs_22 = dofs_2[ self.P2_pol_0.shape[0]*self.H_tor.shape[0]:].reshape(self.P3_pol_0.shape[0], self.I_tor.shape[0])

            coeffs1 = self.H_tor_LU.solve(self.I2_pol_0_LU.solve(dofs_21).T).T
            coeffs2 = self.I_tor_LU.solve(self.I3_pol_0_LU.solve(dofs_22).T).T
        
        return np.concatenate((coeffs1.flatten(), coeffs2.flatten()))
        
    # ======================================
    def solve_V3(self, dofs_3, include_bc):
        
        # with boundary splines
        if include_bc:
            dofs_3 = dofs_3.reshape(self.P3_pol.shape[0], self.H_tor.shape[0])
            coeffs = self.H_tor_LU.solve(self.I3_pol_LU.solve(dofs_3).T).T
        
        # without boundary splines
        else:
            dofs_3 = dofs_3.reshape(self.P3_pol_0.shape[0], self.H_tor.shape[0])
            coeffs = self.H_tor_LU.solve(self.I3_pol_0_LU.solve(dofs_3).T).T
            
        return coeffs.flatten()
    
    
    
    # ======================================
    def apply_IinvT_V0(self, rhs, include_bc=False):
        
        # with boundary splines
        if include_bc:
            rhs = rhs.reshape(self.P0_pol.shape[0], self.I_tor.shape[0])
            rhs = self.I0_pol_T_LU.solve(self.I_tor_T_LU.solve(rhs.T).T)
        
        # without boundary splines
        else:
            rhs = rhs.reshape(self.P0_pol_0.shape[0], self.I_tor.shape[0])
            rhs = self.I0_pol_0_T_LU.solve(self.I_tor_T_LU.solve(rhs.T).T)
          
        return rhs.flatten()
        
    # ======================================
    def apply_IinvT_V1(self, rhs, include_bc=False):
        
        rhs1 = rhs[:self.P1_pol_0.shape[0]*self.I_tor.shape[0] ].reshape(self.P1_pol_0.shape[0], self.I_tor.shape[0])
        rhs2 = rhs[ self.P1_pol_0.shape[0]*self.I_tor.shape[0]:].reshape(self.P0_pol_0.shape[0], self.H_tor.shape[0])
        
        rhs1 = self.I1_pol_0_T_LU.solve(self.I_tor_T_LU.solve(rhs1.T).T)
        rhs2 = self.I0_pol_0_T_LU.solve(self.H_tor_T_LU.solve(rhs2.T).T)
        
        return np.concatenate((rhs1.flatten(), rhs2.flatten()))
    
    # ======================================
    def apply_IinvT_V2(self, rhs, include_bc=False):
                
        rhs1 = rhs[:self.P2_pol_0.shape[0]*self.H_tor.shape[0] ].reshape(self.P2_pol_0.shape[0], self.H_tor.shape[0])
        rhs2 = rhs[ self.P2_pol_0.shape[0]*self.H_tor.shape[0]:].reshape(self.P3_pol_0.shape[0], self.I_tor.shape[0])
        
        rhs1 = self.I2_pol_0_T_LU.solve(self.H_tor_T_LU.solve(rhs1.T).T)
        rhs2 = self.I3_pol_0_T_LU.solve(self.I_tor_T_LU.solve(rhs2.T).T)
        
        return np.concatenate((rhs1.flatten(), rhs2.flatten()))
    
    # ======================================
    def apply_IinvT_V3(self, rhs, include_bc=False):
        
        rhs = rhs.reshape(self.P3_pol_0.shape[0], self.H_tor.shape[0])
        rhs = self.I3_pol_0_T_LU.solve(self.H_tor_T_LU.solve(rhs.T).T)
        
        return rhs.flatten()
    
    
    
    # ======================================        
    def pi_0(self, fun, include_bc=True, eval_kind='meshgrid', interp=True):
        
        # get function values at point sets
        dofs_0 = self.eval_for_PI(0, fun, eval_kind)
        
        # get dofs_0 on tensor-product grid
        dofs_0 = kron_matvec_3d([spa.identity(dofs_0.shape[0]), spa.identity(dofs_0.shape[1]), spa.identity(dofs_0.shape[2])], dofs_0)
        
        # apply extraction operator for dofs
        if include_bc:
            dofs_0 = self.P0.dot(dofs_0.flatten())
        else:
            dofs_0 = self.P0_0.dot(dofs_0.flatten())
        
        # solve for FE coefficients
        if interp:
            coeffs = self.solve_V0(dofs_0, include_bc)
        else:
            coeffs = dofs_0
            
        return coeffs
    
    # ======================================        
    def pi_1(self, fun, include_bc=True, eval_kind='meshgrid', interp=True):
        
        # get function values at point sets
        dofs_11 = self.eval_for_PI(11, fun[0], eval_kind)
        dofs_12 = self.eval_for_PI(12, fun[1], eval_kind)
        dofs_13 = self.eval_for_PI(13, fun[2], eval_kind)
        
        # get dofs_11 on tensor-product grid: integrate along 1-direction
        dofs_11 = kron_matvec_3d([self.Q1, spa.identity(dofs_11.shape[1]), spa.identity(dofs_11.shape[2])], dofs_11)

        # get dofs_12 on tensor-product grid: integrate along 2-direction
        dofs_12 = kron_matvec_3d([spa.identity(dofs_12.shape[0]), self.Q2, spa.identity(dofs_12.shape[2])], dofs_12)
        
        # get dofs_13 on tensor-product grid: integrate along 3-direction
        dofs_13 = kron_matvec_3d([spa.identity(dofs_13.shape[0]), spa.identity(dofs_13.shape[1]), self.Q3], dofs_13)
        
        # apply extraction operator for dofs
        if include_bc:
            dofs_1 = self.P1.dot(np.concatenate((dofs_11.flatten(), dofs_12.flatten(), dofs_13.flatten())))
        else:
            dofs_1 = self.P1_0.dot(np.concatenate((dofs_11.flatten(), dofs_12.flatten(), dofs_13.flatten())))
            
        # solve for FE coefficients
        if interp:
            coeffs = self.solve_V1(dofs_1, include_bc)
        else:
            coeffs = dofs_1
            
        return coeffs
            
    # ======================================        
    def pi_2(self, fun, include_bc=True, eval_kind='meshgrid', interp=True):
        
        # get function values at point sets
        dofs_21 = self.eval_for_PI(21, fun[0], eval_kind)
        dofs_22 = self.eval_for_PI(22, fun[1], eval_kind)
        dofs_23 = self.eval_for_PI(23, fun[2], eval_kind)
        
        # get dofs_21 on tensor-product grid: integrate in 2-3-plane
        dofs_21 = kron_matvec_3d([spa.identity(dofs_21.shape[0]), self.Q2, self.Q3], dofs_21)

        # get dofs_22 on tensor-product grid: integrate in 1-3-plane
        dofs_22 = kron_matvec_3d([self.Q1, spa.identity(dofs_22.shape[1]), self.Q3], dofs_22)
            
        # get dofs_23 on tensor-product grid: integrate in 1-2-plane
        dofs_23 = kron_matvec_3d([self.Q1, self.Q2, spa.identity(dofs_23.shape[2])], dofs_23)
        
        # apply extraction operator for dofs
        if include_bc:
            dofs_2 = self.P2.dot(np.concatenate((dofs_21.flatten(), dofs_22.flatten(), dofs_23.flatten())))
        else:
            dofs_2 = self.P2_0.dot(np.concatenate((dofs_21.flatten(), dofs_22.flatten(), dofs_23.flatten())))
        
        # solve for FE coefficients
        if interp:
            coeffs = self.solve_V2(dofs_2, include_bc)
        else:
            coeffs = dofs_2
            
        return coeffs
    
    # ======================================        
    def pi_3(self, fun, include_bc=True, eval_kind='meshgrid', interp=True):
        
        # get function values at point sets
        dofs_3 = self.eval_for_PI(3, fun, eval_kind)
        
        # get dofs_3 on tensor-product grid: integrate in 1-2-3-cell
        dofs_3 = kron_matvec_3d([self.Q1, self.Q2, self.Q3], dofs_3)
            
        # apply extraction operator for dofs
        if include_bc:
            dofs_3 = self.P3.dot(dofs_3.flatten())
        else:
            dofs_3 = self.P3_0.dot(dofs_3.flatten())
        
        # solve for FE coefficients
        if interp:
            coeffs = self.solve_V3(dofs_3, include_bc)
        else:
            coeffs = dofs_3
            
        return coeffs
    
    
    
    # ========================================
    def assemble_approx_inv(self, tol):
        
        # poloidal plane
        I0_pol_0_inv_approx = np.linalg.inv(self.I0_pol_0.toarray())
        I1_pol_0_inv_approx = np.linalg.inv(self.I1_pol_0.toarray())
        I2_pol_0_inv_approx = np.linalg.inv(self.I2_pol_0.toarray())
        I3_pol_0_inv_approx = np.linalg.inv(self.I3_pol_0.toarray())
        I0_pol_inv_approx = np.linalg.inv(self.I0_pol.toarray())
        
        if tol > 1e-14:
            I0_pol_0_inv_approx[np.abs(I0_pol_0_inv_approx) < tol] = 0.
            I1_pol_0_inv_approx[np.abs(I1_pol_0_inv_approx) < tol] = 0.
            I2_pol_0_inv_approx[np.abs(I2_pol_0_inv_approx) < tol] = 0.
            I3_pol_0_inv_approx[np.abs(I3_pol_0_inv_approx) < tol] = 0.
            I0_pol_inv_approx[np.abs(I0_pol_inv_approx) < tol] = 0.
        
        I0_pol_0_inv_approx = spa.csr_matrix(I0_pol_0_inv_approx)
        I1_pol_0_inv_approx = spa.csr_matrix(I1_pol_0_inv_approx)
        I2_pol_0_inv_approx = spa.csr_matrix(I2_pol_0_inv_approx)
        I3_pol_0_inv_approx = spa.csr_matrix(I3_pol_0_inv_approx)
        I0_pol_inv_approx = spa.csr_matrix(I0_pol_inv_approx)
        
        # toroidal direction
        I_inv_tor_approx = np.linalg.inv(self.I_tor.toarray())
        H_inv_tor_approx = np.linalg.inv(self.H_tor.toarray())
        
        if tol > 1e-14:
            I_inv_tor_approx[np.abs(I_inv_tor_approx) < tol] = 0.
            H_inv_tor_approx[np.abs(H_inv_tor_approx) < tol] = 0.
        
        I_inv_tor_approx = spa.csr_matrix(I_inv_tor_approx)
        H_inv_tor_approx = spa.csr_matrix(H_inv_tor_approx)

        # tensor-product poloidal x toroidal
        self.I0_0_inv_approx = spa.kron(I0_pol_0_inv_approx, I_inv_tor_approx, format='csr')

        self.I1_0_inv_approx = spa.bmat([[spa.kron(I1_pol_0_inv_approx, I_inv_tor_approx), None], [None, spa.kron(I0_pol_0_inv_approx, H_inv_tor_approx)]], format='csr')
        
        self.I2_0_inv_approx = spa.bmat([[spa.kron(I2_pol_0_inv_approx, H_inv_tor_approx), None], [None, spa.kron(I3_pol_0_inv_approx, I_inv_tor_approx)]], format='csr')
        
        self.I3_0_inv_approx = spa.kron(I3_pol_0_inv_approx, H_inv_tor_approx, format='csr')
        
        self.I0_inv_approx = spa.kron(I0_pol_inv_approx, I_inv_tor_approx, format='csr')