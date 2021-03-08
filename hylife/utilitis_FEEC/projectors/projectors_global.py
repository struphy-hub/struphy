# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Classes for global projectors in 1D and 3D based on spline interpolation and histopolation.
"""

import numpy as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.bsplines as bsp

import hylife.utilitis_FEEC.projectors.kernels_projectors_global     as ker_glob
import source_run.kernels_projectors_evaluation as ker_eva

from   hylife.linear_algebra.linalg_kron import kron_lusolve_3d



# ======================= 1d ====================================
class projectors_global_1d:
    """
    Global commuting projectors pi_0 and pi_1 in 1d.
    
    Parameters
    ----------
    spline_space : spline_space_1d
        a 1d space of B-splines
        
    n_quad : int
        number of quadrature points per integration interval for histopolations
        
    polar : boolean
        whether there is a polar singularity in the mapping
    """
    
    def __init__(self, spline_space, n_quad, polar=False):
        
        self.kind     = 'global'
        
        self.space    = spline_space         # 1D spline space
        self.T        = spline_space.T       # knot vector
        self.p        = spline_space.p       # spline degree
        self.bc       = spline_space.bc      # boundary conditions
        
        self.NbaseN   = spline_space.NbaseN  # number of basis functions (N)
        self.NbaseD   = spline_space.NbaseD  # number of basis functions (D)
        
        self.el_b     = spline_space.el_b    # element boundaries
        
        self.n_quad   = n_quad               # number of quadrature point per integration interval
        
        self.polar    = polar                # whether polar splines are used in the poloidal plane
        
        # Gauss - Legendre quadrature points and weights in (-1, 1)
        self.pts_loc  = np.polynomial.legendre.leggauss(self.n_quad)[0]  
        self.wts_loc  = np.polynomial.legendre.leggauss(self.n_quad)[1]
        
        # set interpolation points (Greville points)
        self.x_int    = bsp.greville(self.T, self.p, self.bc)
        
        # set histopolation grid
        if self.bc == False:
            self.x_his = np.copy(self.x_int)
        else:
            self.x_his = np.append(self.x_int, self.el_b[-1] + self.x_int[0])
              
        # quadrature grid and weights
        self.pts, self.wts = bsp.quadrature_grid(self.x_his, self.pts_loc, self.wts_loc)
        self.pts           = self.pts%self.el_b[-1]
        
        # intepolation and histopolation_matrix
        self.N     = spa.csr_matrix(bsp.collocation_matrix(self.T, self.p, self.x_int, self.bc))
        self.D     = spa.csr_matrix(bsp.histopolation_matrix(self.T, self.p, self.x_int, self.bc))
                       
        # shift first interplation point in radial direction away from pole
        if self.bc == False and self.polar == True:
            self.x_int[0] += 0.000001
        
        # LU decompositions
        self.N_LU  = spa.linalg.splu(self.N.tocsc())
        self.D_LU  = spa.linalg.splu(self.D.tocsc())
        
        # inverse intepolation and histopolation_matrix
        self.N_inv = np.linalg.inv(self.N.toarray())
        self.D_inv = np.linalg.inv(self.D.toarray())
        
    
    # evaluate function at interpolation points    
    def rhs_0(self, fun):
        return fun(self.x_int)
    
    # evaluate integrals betwenn interpolation points
    def rhs_1(self, fun):
        
        # evaluate function at quadrature points
        mat_f  = fun(self.pts)
        values = np.zeros(self.NbaseD, dtype=float)
        
        for i in range(self.NbaseD):
            values[i] = ker_glob.kernel_int_1d(self.n_quad, self.wts[i], mat_f[i])
                
        return values
    
    # pi_0 projector
    def pi_0(self, fun):
        return self.N_LU.solve(self.rhs_0(fun))
    
    # pi_1 projector
    def pi_1(self, fun):
        return self.D_LU.solve(self.rhs_1(fun))
    
    
    # projection matrices of products of basis functions: pi0_i(A_j*B_k) and pi1_i(A_j*B_k)
    def projection_matrices_1d(self, params_map=None, kind='rhs', bc_kind=['free', 'free']):
    
        PI0_NN = np.empty((self.NbaseN, self.NbaseN, self.NbaseN), dtype=float)
        PI0_DN = np.empty((self.NbaseN, self.NbaseD, self.NbaseN), dtype=float)
        PI0_DD = np.empty((self.NbaseN, self.NbaseD, self.NbaseD), dtype=float)

        PI1_NN = np.empty((self.NbaseD, self.NbaseN, self.NbaseN), dtype=float)
        PI1_DN = np.empty((self.NbaseD, self.NbaseD, self.NbaseN), dtype=float)
        PI1_DD = np.empty((self.NbaseD, self.NbaseD, self.NbaseD), dtype=float)


        # ========= PI0__NN and PI1_NN =============
        ci = np.zeros(self.NbaseN, dtype=float)
        cj = np.zeros(self.NbaseN, dtype=float)

        for i in range(self.NbaseN):
            for j in range(self.NbaseN):

                ci[:] = 0.
                cj[:] = 0.

                ci[i] = 1.
                cj[j] = 1.
                
                if self.bc == False and self.polar == True:
                    fun = lambda eta : self.space.evaluate_N(eta, ci)*self.space.evaluate_N(eta, cj)/(eta*2*np.pi*params_map[2]*params_map[1]**2)
                else:
                    fun = lambda eta : self.space.evaluate_N(eta, ci)*self.space.evaluate_N(eta, cj)

                if kind == 'rhs':
                    PI0_NN[:, i, j] = self.rhs_0(fun)
                    PI1_NN[:, i, j] = self.rhs_1(fun)
                else:
                    PI0_NN[:, i, j] = self.pi_0(fun)
                    PI1_NN[:, i, j] = self.pi_1(fun)



        # ========= PI0__DN and PI1_DN =============
        ci = np.zeros(self.NbaseD, dtype=float)
        cj = np.zeros(self.NbaseN, dtype=float)

        for i in range(self.NbaseD):
            for j in range(self.NbaseN):

                ci[:] = 0.
                cj[:] = 0.

                ci[i] = 1.
                cj[j] = 1.
                
                if self.bc == False and self.polar == True:
                    fun = lambda eta : self.space.evaluate_D(eta, ci)*self.space.evaluate_N(eta, cj)/(eta*2*np.pi*params_map[2]*params_map[1]**2)
                else:
                    fun = lambda eta : self.space.evaluate_D(eta, ci)*self.space.evaluate_N(eta, cj)

                if kind == 'rhs':
                    PI0_DN[:, i, j] = self.rhs_0(fun)
                    PI1_DN[:, i, j] = self.rhs_1(fun)
                else:
                    PI0_DN[:, i, j] = self.pi_0(fun)
                    PI1_DN[:, i, j] = self.pi_1(fun)



        # ========= PI0__DD and PI1_DD =============
        ci = np.zeros(self.NbaseD, dtype=float)
        cj = np.zeros(self.NbaseD, dtype=float)

        for i in range(self.NbaseD):
            for j in range(self.NbaseD):

                ci[:] = 0.
                cj[:] = 0.

                ci[i] = 1.
                cj[j] = 1.
                
                if self.bc == False and self.polar == True:
                    fun = lambda eta : self.space.evaluate_D(eta, ci)*self.space.evaluate_D(eta, cj)/(eta*2*np.pi*params_map[2]*params_map[1]**2)
                else:
                    fun = lambda eta : self.space.evaluate_D(eta, ci)*self.space.evaluate_D(eta, cj)

                if kind == 'rhs':
                    PI0_DD[:, i, j] = self.rhs_0(fun)
                    PI1_DD[:, i, j] = self.rhs_1(fun)
                else:
                    PI0_DD[:, i, j] = self.pi_0(fun)
                    PI1_DD[:, i, j] = self.pi_1(fun)


        PI0_ND = np.transpose(PI0_DN, (0, 2, 1))
        PI1_ND = np.transpose(PI1_DN, (0, 2, 1))


        # remove contributions from first and last N-splines
        if bc_kind[0] == 'dirichlet':
            PI0_NN[:,  :,  0] = 0.
            PI0_NN[:,  0,  :] = 0.
            PI0_DN[:,  :,  0] = 0.
            PI0_ND[:,  0,  :] = 0.

            PI1_NN[:,  :,  0] = 0.
            PI1_NN[:,  0,  :] = 0.
            PI1_DN[:,  :,  0] = 0.
            PI1_ND[:,  0,  :] = 0.

        if bc_kind[1] == 'dirichlet':    
            PI0_NN[:,  :, -1] = 0.
            PI0_NN[:, -1,  :] = 0.
            PI0_DN[:,  :, -1] = 0.
            PI0_ND[:, -1,  :] = 0.

            PI1_NN[:,  :, -1] = 0.
            PI1_NN[:, -1,  :] = 0.
            PI1_DN[:,  :, -1] = 0.
            PI1_ND[:, -1,  :] = 0.


        PI0_NN_indices = np.nonzero(PI0_NN)
        PI0_DN_indices = np.nonzero(PI0_DN)
        PI0_ND_indices = np.nonzero(PI0_ND)
        PI0_DD_indices = np.nonzero(PI0_DD)

        PI1_NN_indices = np.nonzero(PI1_NN)
        PI1_DN_indices = np.nonzero(PI1_DN)
        PI1_ND_indices = np.nonzero(PI1_ND)
        PI1_DD_indices = np.nonzero(PI1_DD)
        
        PI0_NN_i_red = np.empty(PI0_NN_indices[0].size, dtype=int)
        PI0_DN_i_red = np.empty(PI0_DN_indices[0].size, dtype=int)
        PI0_ND_i_red = np.empty(PI0_ND_indices[0].size, dtype=int)
        PI0_DD_i_red = np.empty(PI0_DD_indices[0].size, dtype=int)

        PI1_NN_i_red = np.empty(PI1_NN_indices[0].size, dtype=int)
        PI1_DN_i_red = np.empty(PI1_DN_indices[0].size, dtype=int)
        PI1_ND_i_red = np.empty(PI1_ND_indices[0].size, dtype=int)
        PI1_DD_i_red = np.empty(PI1_DD_indices[0].size, dtype=int)
        
        # ================================
        nv = self.NbaseN*PI0_NN_indices[1] + PI0_NN_indices[2]
        un = np.unique(nv)
        
        for i in range(PI0_NN_indices[0].size):
            PI0_NN_i_red[i] = np.nonzero(un == nv[i])[0]
            
        # ================================
        nv = self.NbaseN*PI0_DN_indices[1] + PI0_DN_indices[2]
        un = np.unique(nv)
        
        for i in range(PI0_DN_indices[0].size):
            PI0_DN_i_red[i] = np.nonzero(un == nv[i])[0]
            
        # ================================
        nv = self.NbaseD*PI0_ND_indices[1] + PI0_ND_indices[2]
        un = np.unique(nv)
        
        for i in range(PI0_ND_indices[0].size):
            PI0_ND_i_red[i] = np.nonzero(un == nv[i])[0]
            
        # ================================
        nv = self.NbaseD*PI0_DD_indices[1] + PI0_DD_indices[2]
        un = np.unique(nv)
        
        for i in range(PI0_DD_indices[0].size):
            PI0_DD_i_red[i] = np.nonzero(un == nv[i])[0]
            
        # ================================
        nv = self.NbaseN*PI1_NN_indices[1] + PI1_NN_indices[2]
        un = np.unique(nv)
        
        for i in range(PI1_NN_indices[0].size):
            PI1_NN_i_red[i] = np.nonzero(un == nv[i])[0]
            
        # ================================
        nv = self.NbaseN*PI1_DN_indices[1] + PI1_DN_indices[2]
        un = np.unique(nv)
        
        for i in range(PI1_DN_indices[0].size):
            PI1_DN_i_red[i] = np.nonzero(un == nv[i])[0]
            
        # ================================
        nv = self.NbaseD*PI1_ND_indices[1] + PI1_ND_indices[2]
        un = np.unique(nv)
        
        for i in range(PI1_ND_indices[0].size):
            PI1_ND_i_red[i] = np.nonzero(un == nv[i])[0]
            
        # ================================
        nv = self.NbaseD*PI1_DD_indices[1] + PI1_DD_indices[2]
        un = np.unique(nv)
        
        for i in range(PI1_DD_indices[0].size):
            PI1_DD_i_red[i] = np.nonzero(un == nv[i])[0] 
            
            
        PI0_NN_indices = np.vstack((PI0_NN_indices[0], PI0_NN_indices[1], PI0_NN_indices[2], PI0_NN_i_red))
        PI0_DN_indices = np.vstack((PI0_DN_indices[0], PI0_DN_indices[1], PI0_DN_indices[2], PI0_DN_i_red))
        PI0_ND_indices = np.vstack((PI0_ND_indices[0], PI0_ND_indices[1], PI0_ND_indices[2], PI0_ND_i_red))
        PI0_DD_indices = np.vstack((PI0_DD_indices[0], PI0_DD_indices[1], PI0_DD_indices[2], PI0_DD_i_red))

        PI1_NN_indices = np.vstack((PI1_NN_indices[0], PI1_NN_indices[1], PI1_NN_indices[2], PI1_NN_i_red))
        PI1_DN_indices = np.vstack((PI1_DN_indices[0], PI1_DN_indices[1], PI1_DN_indices[2], PI1_DN_i_red))
        PI1_ND_indices = np.vstack((PI1_ND_indices[0], PI1_ND_indices[1], PI1_ND_indices[2], PI1_ND_i_red))
        PI1_DD_indices = np.vstack((PI1_DD_indices[0], PI1_DD_indices[1], PI1_DD_indices[2], PI1_DD_i_red))
        

        #return PI0_NN, PI0_DN, PI0_ND, PI0_DD, PI1_NN, PI1_DN, PI1_ND, PI1_DD, PI0_NN_indices, PI0_DN_indices, PI0_ND_indices, PI0_DD_indices, PI1_NN_indices, PI1_DN_indices, PI1_ND_indices, PI1_DD_indices
        
        return PI0_NN_indices, PI0_DN_indices, PI0_ND_indices, PI0_DD_indices, PI1_NN_indices, PI1_DN_indices, PI1_ND_indices, PI1_DD_indices 
    
    
    
    # projection matrices of products basis functions: pi0_i(A_j) and pi1_i(A_j) for A = N or A = D
    def projection_matrices_1d_reduced(self):
        
        PI0_N = np.empty((self.NbaseN, self.NbaseN), dtype=float)
        PI0_D = np.empty((self.NbaseN, self.NbaseD), dtype=float)

        PI1_N = np.empty((self.NbaseD, self.NbaseN), dtype=float)
        PI1_D = np.empty((self.NbaseD, self.NbaseD), dtype=float)


        # ========= PI0_N and PI1_N =============
        ci = np.zeros(self.NbaseN, dtype=float)

        for i in range(self.NbaseN):

            ci[:] = 0.
            ci[i] = 1.

            fun = lambda eta : self.space.evaluate_N(eta, ci)

            PI0_N[:, i] = self.rhs_0(fun)
            PI1_N[:, i] = self.rhs_1(fun)
            
        # ========= PI0_D and PI1_D =============
        ci = np.zeros(self.NbaseD, dtype=float)

        for i in range(self.NbaseD):

            ci[:] = 0.
            ci[i] = 1.

            fun = lambda eta : self.space.evaluate_D(eta, ci)

            PI0_D[:, i] = self.rhs_0(fun)
            PI1_D[:, i] = self.rhs_1(fun)
            
        PI0_N_indices = np.nonzero(PI0_N)
        PI0_D_indices = np.nonzero(PI0_D)
        PI1_N_indices = np.nonzero(PI1_N)
        PI1_D_indices = np.nonzero(PI1_D)
        
        return PI0_N_indices, PI0_D_indices, PI1_N_indices, PI1_D_indices

    
    
    

    
# ======================= 3d ====================================
class projectors_global_3d:
    """
    Global commuting projectors pi_0, pi_1, pi_2 and pi_3 in 3d.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        a 3d tensor-product space of B-splines
        
    n_quad : int
        number of Gauss-Legendre quadrature points per integration interval for histopolations
        
    polar_splines : polar_splines_3d
        class for polar splines in the poloidal plane (optional)
    """
    
    def __init__(self, tensor_space, n_quad, polar_splines=None):
        
        self.tensor_space = tensor_space     # 3D tensor-product B-splines space
        
        self.kind     = 'global'             # kind of projector
        
        self.T        = tensor_space.T       # knot vector
        self.p        = tensor_space.p       # spline degree
        self.bc       = tensor_space.bc      # boundary conditions
        self.el_b     = tensor_space.el_b    # element boundaries
        
        self.Nel      = tensor_space.Nel     # number of elements
        self.NbaseN   = tensor_space.NbaseN  # number of basis functions (N)
        self.NbaseD   = tensor_space.NbaseD  # number of basis functions (D)
        
        self.n_quad   = n_quad               # number of quadrature point per integration interval
        
        # polar extraction operators
        if polar_splines != None:
            
            self.polar  = True
            
            self.P0_pol = polar_splines.P0_pol # 2D extraction operator for V0 interpolation
            self.P1_pol = polar_splines.P1_pol # 2D extraction operator for V1 histo-/inter- and inter-/histoplation
            self.P2_pol = polar_splines.P2_pol # 2D extraction operator for V2 inter-/histo- and histo-/interpolation
            self.P3_pol = polar_splines.P3_pol # 2D extraction operator for V3 histoplation
            
            self.P0     = polar_splines.P0     # 3D extraction operator for V0 interpolation
            self.P1     = polar_splines.P1     # 3D extraction operator for V1 histo-/inter- and inter-/histoplation
            self.P2     = polar_splines.P2     # 3D extraction operator for V2 inter-/histo- and histo-/interpolation
            self.P3     = polar_splines.P3     # 3D extraction operator for V3 histoplation
        
        else:
            
            self.polar  = False
            
            self.P0_pol = spa.identity(tensor_space.Nbase0_pol, dtype=float, format='csr')
            self.P1_pol = spa.identity(tensor_space.Nbase1_pol, dtype=float, format='csr')
            self.P2_pol = spa.identity(tensor_space.Nbase2_pol, dtype=float, format='csr')
            self.P3_pol = spa.identity(tensor_space.Nbase3_pol, dtype=float, format='csr')
            
            self.P0     = spa.identity(tensor_space.Nbase0_pol*self.NbaseN[2], dtype=float, format='csr')  
            self.P1     = spa.identity(tensor_space.Nbase1_pol*self.NbaseN[2] + tensor_space.Nbase0_pol*self.NbaseD[2], dtype=float, format='csr')     
            self.P2     = spa.identity(tensor_space.Nbase2_pol*self.NbaseD[2] + tensor_space.Nbase3_pol*self.NbaseN[2], dtype=float, format='csr')    
            self.P3     = spa.identity(tensor_space.Nbase3_pol*self.NbaseD[2], dtype=float, format='csr')   
            
        
        # Gauss - Legendre quadrature points and weights in (-1, 1)
        self.pts_loc = [np.polynomial.legendre.leggauss(n_quad)[0] for n_quad in self.n_quad]
        self.wts_loc = [np.polynomial.legendre.leggauss(n_quad)[1] for n_quad in self.n_quad]
        
        # set interpolation points (Greville points)
        self.x_int = [bsp.greville(T, p, bc) for T, p, bc in zip(self.T, self.p, self.bc)]
        
        # set histopolation grid
        self.x_his = [0, 0, 0]
        self.pts   = [0, 0, 0]
        self.wts   = [0, 0, 0]
        
        for a in range(3):
            if self.bc[a] == False:
                self.x_his[a] = np.copy(self.x_int[a])
            else:
                self.x_his[a] = np.append(self.x_int[a], self.el_b[a][-1] + self.x_int[a][0])
            
            # quadrature grid and weights
            self.pts[a], self.wts[a] = bsp.quadrature_grid(self.x_his[a], self.pts_loc[a], self.wts_loc[a])
            self.pts[a]              = self.pts[a]%self.el_b[a][-1]
        
        
        # shift first radial interpolation point away from pole
        if self.polar == True:
            self.x_int[0][0] += 0.00001
        
        # 1D interpolation and histopolation matrices
        self.N = [spa.csr_matrix(bsp.collocation_matrix(  T, p, x, bc)) for T, p, x, bc in zip(self.T, self.p, self.x_int, self.bc)]
        self.D = [spa.csr_matrix(bsp.histopolation_matrix(T, p, x, bc)) for T, p, x, bc in zip(self.T, self.p, self.x_int, self.bc)]
        
        # LU decompositions and inverses of 1D interpolation and histopolation matrices
        self.N_LU  = [spa.linalg.splu(N.tocsc()) for N in self.N]
        self.D_LU  = [spa.linalg.splu(D.tocsc()) for D in self.D]
        
        self.N_inv = [np.linalg.inv(N.toarray()) for N in self.N]
        self.D_inv = [np.linalg.inv(D.toarray()) for D in self.D]
        
        # collection of the point sets for different 3D projectors
        self.pts_PI_0  = [self.x_int[0]        , self.x_int[1]        , self.x_int[2]        ]
        
        self.pts_PI_11 = [self.pts[0].flatten(), self.x_int[1]        , self.x_int[2]        ]
        self.pts_PI_12 = [self.x_int[0]        , self.pts[1].flatten(), self.x_int[2]        ]
        self.pts_PI_13 = [self.x_int[0]        , self.x_int[1]        , self.pts[2].flatten()]
        
        self.pts_PI_21 = [self.x_int[0]        , self.pts[1].flatten(), self.pts[2].flatten()]
        self.pts_PI_22 = [self.pts[0].flatten(), self.x_int[1]        , self.pts[2].flatten()]
        self.pts_PI_23 = [self.pts[0].flatten(), self.pts[1].flatten(), self.x_int[2]        ]
        
        self.pts_PI_3  = [self.pts[0].flatten(), self.pts[1].flatten(), self.pts[2].flatten()]
        
        # 3D interpolation/histopolation matrices
        self.I0 = spa.kron(self.N[0], spa.kron(self.N[1], self.N[2]), format='csr')
        self.I0 = self.P0.dot(self.I0.dot(self.tensor_space.E0.T)).tocsr()
        
        I1_1    = spa.kron(self.D[0], spa.kron(self.N[1], self.N[2]), format='csr')
        I1_2    = spa.kron(self.N[0], spa.kron(self.D[1], self.N[2]), format='csr')
        I1_3    = spa.kron(self.N[0], spa.kron(self.N[1], self.D[2]), format='csr')
        
        self.I1 = spa.bmat([[I1_1, None, None], [None, I1_2, None], [None, None, I1_3]])
        self.I1 = self.P1.dot(self.I1.dot(self.tensor_space.E1.T)).tocsr()
        
        I2_1    = spa.kron(self.N[0], spa.kron(self.D[1], self.D[2]), format='csr')
        I2_2    = spa.kron(self.D[0], spa.kron(self.N[1], self.D[2]), format='csr')
        I2_3    = spa.kron(self.D[0], spa.kron(self.D[1], self.N[2]), format='csr')
        
        self.I2 = spa.bmat([[I2_1, None, None], [None, I2_2, None], [None, None, I2_3]])
        self.I2 = self.P2.dot(self.I2.dot(self.tensor_space.E2.T)).tocsr()
        
        self.I3 = spa.kron(self.D[0], spa.kron(self.D[1], self.D[2]), format='csr')
        self.I3 = self.P3.dot(self.I3.dot(self.tensor_space.E3.T)).tocsr()
        
        
        # 2D interpolation matrix in poloidal plane
        self.I0_pol = self.P0_pol.dot(spa.kron(self.N[0], self.N[1])).dot(self.tensor_space.E0_pol.T).tocsr()

        # 2D histo-/interpolation and inter-/histopolation matrix in poloidal plane (and vice versa)
        DN = spa.kron(self.D[0], self.N[1], format='csr')
        ND = spa.kron(self.N[0], self.D[1], format='csr')

        self.I1_pol = self.P1_pol.dot(spa.bmat([[DN, None],[None, ND]]).dot(self.tensor_space.E1_pol.T)).tocsr()
        self.I2_pol = self.P2_pol.dot(spa.bmat([[ND, None],[None, DN]]).dot(self.tensor_space.E2_pol.T)).tocsr()

        # 2D histopolation matrix in poloidal plane
        self.I3_pol = self.P3_pol.dot(spa.kron(self.D[0], self.D[1])).dot(self.tensor_space.E3_pol.T).tocsr()

        # LU decompositions and inverses in poloidal plane
        self.I0_pol_LU  = spa.linalg.splu(self.I0_pol.tocsc())
        self.I1_pol_LU  = spa.linalg.splu(self.I1_pol.tocsc())
        self.I2_pol_LU  = spa.linalg.splu(self.I2_pol.tocsc())
        self.I3_pol_LU  = spa.linalg.splu(self.I3_pol.tocsc())

        self.I0_pol_inv = np.linalg.inv(self.I0_pol.toarray())
        self.I1_pol_inv = np.linalg.inv(self.I1_pol.toarray())
        self.I2_pol_inv = np.linalg.inv(self.I2_pol.toarray())
        self.I3_pol_inv = np.linalg.inv(self.I3_pol.toarray())
            
    
    
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
        
        I0_pol_inv_approx = np.copy(self.I0_pol_inv)
        I1_pol_inv_approx = np.copy(self.I1_pol_inv)
        I2_pol_inv_approx = np.copy(self.I2_pol_inv)
        I3_pol_inv_approx = np.copy(self.I3_pol_inv)

        I0_pol_inv_approx[np.abs(I0_pol_inv_approx) < tol] = 0.
        I1_pol_inv_approx[np.abs(I1_pol_inv_approx) < tol] = 0.
        I2_pol_inv_approx[np.abs(I2_pol_inv_approx) < tol] = 0.
        I3_pol_inv_approx[np.abs(I3_pol_inv_approx) < tol] = 0.
        
        I0_pol_inv_approx = spa.csr_matrix(I0_pol_inv_approx)
        I1_pol_inv_approx = spa.csr_matrix(I1_pol_inv_approx)
        I2_pol_inv_approx = spa.csr_matrix(I2_pol_inv_approx)
        I3_pol_inv_approx = spa.csr_matrix(I3_pol_inv_approx)

        N_inv_z_approx = np.copy(self.N_inv[2])
        D_inv_z_approx = np.copy(self.D_inv[2])

        N_inv_z_approx[np.abs(N_inv_z_approx) < tol] = 0.
        D_inv_z_approx[np.abs(D_inv_z_approx) < tol] = 0.
        
        N_inv_z_approx = spa.csr_matrix(N_inv_z_approx)
        D_inv_z_approx = spa.csr_matrix(D_inv_z_approx)
        
        self.I0_inv_approx = spa.kron(I0_pol_inv_approx, N_inv_z_approx, format='csr')

        self.I1_inv_approx = spa.bmat([[spa.kron(I1_pol_inv_approx, N_inv_z_approx), None], [None, spa.kron(I0_pol_inv_approx, D_inv_z_approx)]], format='csr') 
        self.I2_inv_approx = spa.bmat([[spa.kron(I2_pol_inv_approx, D_inv_z_approx), None], [None, spa.kron(I3_pol_inv_approx, N_inv_z_approx)]], format='csr')
        
        self.I3_inv_approx = spa.kron(I3_pol_inv_approx, D_inv_z_approx, format='csr')
    
    
        
    # ========================================    
    def getpts_for_PI(self, comp):
        """
        Get the needed point sets for a given projector.
        
        Parameters
        ----------
        comp: int
            which projector, one of (0, 1, 2, 3).
        
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
    def eval_for_PI(self, comp, fun, domain=None):
        """
        Evaluates the callable "fun" at the points corresponding to the projector, and returns the result as 3d array "mat_f".
            
        Parameters
        ----------
        comp: int
            which projector, one of (0, 11, 12, 13, 21, 22, 23, 3).
            
        fun : callable
            the function fun(eta1, eta2, eta3) to project
               
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
            pts1, pts2, pts3 = np.meshgrid(pts_PI[0], pts_PI[1], pts_PI[2], indexing='ij')
            mat_f[:, :, :]   = fun(pts1, pts2, pts3)
        
        # internal function call
        else:
            # evaluate function on point set
            ker_eva.kernel_eva(pts_PI[0], pts_PI[1], pts_PI[2], mat_f, fun, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
       
        return mat_f
    
    
    # ======================================
    def solve_V0(self, rhs):
        
        rhs    = rhs.reshape(self.tensor_space.Nbase0_pol, self.NbaseN[2])
        
        # solve system
        coeffs = self.N_LU[2].solve(self.I0_pol_LU.solve(rhs).T).T
            
        return coeffs.flatten()
    
    
    # ======================================
    def solve_V1(self, rhs):
        
        rhs12    = rhs[:self.tensor_space.Nbase1_pol*self.NbaseN[2] ].reshape(self.tensor_space.Nbase1_pol, self.NbaseN[2])
        rhs3     = rhs[ self.tensor_space.Nbase1_pol*self.NbaseN[2]:].reshape(self.tensor_space.Nbase0_pol, self.NbaseD[2])

        # solve systems
        coeffs12 = self.N_LU[2].solve(self.I1_pol_LU.solve(rhs12).T).T
        coeffs3  = self.D_LU[2].solve(self.I0_pol_LU.solve(rhs3 ).T).T
        
        return np.concatenate((coeffs12.flatten(), coeffs3.flatten()))
        
    
    # ======================================
    def solve_V2(self, rhs):
        
        rhs12    = rhs[:self.tensor_space.Nbase2_pol*self.NbaseD[2] ].reshape(self.tensor_space.Nbase2_pol, self.NbaseD[2])
        rhs3     = rhs[ self.tensor_space.Nbase2_pol*self.NbaseD[2]:].reshape(self.tensor_space.Nbase3_pol, self.NbaseN[2])
        
        # solve systems
        coeffs12 = self.D_LU[2].solve(self.I2_pol_LU.solve(rhs12).T).T
        coeffs3  = self.N_LU[2].solve(self.I3_pol_LU.solve(rhs3 ).T).T
        
        return np.concatenate((coeffs12.flatten(), coeffs3.flatten()))
        
    # ======================================
    def solve_V3(self, rhs):
        
        rhs    = rhs.reshape(self.tensor_space.Nbase3_pol, self.NbaseD[2])
        
        # solve system
        coeffs = self.D_LU[2].solve(self.I3_pol_LU.solve(rhs).T).T
            
        return coeffs.flatten()
        
    # ======================================
    def apply_IinvT_V0(self, rhs):
        
        rhs = rhs.reshape(self.tensor_space.Nbase0_pol, self.NbaseN[2])
        
        return self.I0_pol_inv.T.dot(self.N_inv[2].T.dot(rhs.T).T).flatten()
    
    # ======================================
    def apply_IinvT_V1(self, rhs):
                
        rhs_12 = rhs[:self.tensor_space.Nbase1_pol*self.NbaseN[2] ].reshape(self.tensor_space.Nbase1_pol, self.NbaseN[2])
        rhs_3  = rhs[ self.tensor_space.Nbase1_pol*self.NbaseN[2]:].reshape(self.tensor_space.Nbase0_pol, self.NbaseD[2])

        rhs_12 = self.I1_pol_inv.T.dot(self.N_inv[2].T.dot(rhs_12.T).T)
        rhs_3  = self.I0_pol_inv.T.dot(self.D_inv[2].T.dot(rhs_3.T).T)
        
        return np.concatenate((rhs_12.flatten(), rhs_3.flatten()))
    
    # ======================================
    def apply_IinvT_V2(self, rhs):
                
        rhs_12 = rhs[:self.tensor_space.Nbase2_pol*self.NbaseD[2] ].reshape(self.tensor_space.Nbase2_pol, self.NbaseD[2])
        rhs_3  = rhs[ self.tensor_space.Nbase2_pol*self.NbaseD[2]:].reshape(self.tensor_space.Nbase3_pol, self.NbaseN[2])

        rhs_12 = self.I2_pol_inv.T.dot(self.D_inv[2].T.dot(rhs_12.T).T)
        rhs_3  = self.I3_pol_inv.T.dot(self.N_inv[2].T.dot(rhs_3.T).T)
        
        return np.concatenate((rhs_12.flatten(), rhs_3.flatten()))
    
    # ======================================
    def apply_IinvT_V3(self, rhs):
        
        rhs = rhs.reshape(self.tensor_space.Nbase3_pol, self.NbaseD[2])
        
        return self.I3_pol_inv.T.dot(self.D_inv[2].T.dot(rhs.T).T).flatten()  
    
    # ======================================        
    def pi_0(self, fun, domain=None):
        
        #  ==== evaluate on tensor-product grid ====
        rhs = self.eval_for_PI(0, fun, domain)
        
        # ====== solve for coefficients ============
        return self.solve_V0(self.P0.dot(rhs.flatten()))
    
    
    # ======================================        
    def pi_1(self, fun, domain=None):
        
        # ====== integrate along 1-direction =======
        n1   = self.pts_PI_11[0].size//self.n_quad[0] 
        n2   = self.pts_PI_11[1].size 
        n3   = self.pts_PI_11[2].size
        
        pts  = self.eval_for_PI(11, fun[0], domain)
        
        rhs1 = np.empty((n1, n2, n3), dtype=float)
        ker_glob.kernel_int_1d_ext_eta1(self.wts[0], pts.reshape(n1, self.n_quad[0], n2, n3), rhs1)
            
        # ====== integrate along 2-direction =======
        n1   = self.pts_PI_12[0].size
        n2   = self.pts_PI_12[1].size//self.n_quad[1]  
        n3   = self.pts_PI_12[2].size
        
        pts  = self.eval_for_PI(12, fun[1], domain)
        
        rhs2 = np.empty((n1, n2, n3), dtype=float)
        ker_glob.kernel_int_1d_ext_eta2(self.wts[1], pts.reshape(n1, n2, self.n_quad[1], n3), rhs2)
        
        # ====== integrate along 3-direction =======
        n1   = self.pts_PI_13[0].size
        n2   = self.pts_PI_13[1].size  
        n3   = self.pts_PI_13[2].size//self.n_quad[2]
        
        pts  = self.eval_for_PI(13, fun[2], domain)
        
        rhs3 = np.empty((n1, n2, n3), dtype=float)
        ker_glob.kernel_int_1d_ext_eta3(self.wts[2], pts.reshape(n1, n2, n3, self.n_quad[2]), rhs3)
        
        # total tensor-product right-hand side
        rhs  = np.concatenate((rhs1.flatten(), rhs2.flatten(), rhs3.flatten()))
            
        # ====== solve for coefficients ============
        return self.solve_V1(self.P1.dot(rhs))
            

    # ======================================        
    def pi_2(self, fun, domain=None):
        
        # ====== integrate in 2-3-plane =======
        n1   = self.pts_PI_21[0].size
        n2   = self.pts_PI_21[1].size//self.n_quad[1] 
        n3   = self.pts_PI_21[2].size//self.n_quad[2]
        
        pts  = self.eval_for_PI(21, fun[0], domain)
        
        rhs1 = np.empty((n1, n2, n3), dtype=float)
        ker_glob.kernel_int_2d_ext_eta2_eta3(self.wts[1], self.wts[2], pts.reshape(n1, n2, self.n_quad[1], n3, self.n_quad[2]), rhs1)   
        # ====== integrate in 1-3-plane =======
        n1   = self.pts_PI_22[0].size//self.n_quad[0] 
        n2   = self.pts_PI_22[1].size
        n3   = self.pts_PI_22[2].size//self.n_quad[2]
        
        pts  = self.eval_for_PI(22, fun[1], domain)
        
        rhs2 = np.empty((n1, n2, n3), dtype=float)
        ker_glob.kernel_int_2d_ext_eta1_eta3(self.wts[0], self.wts[2], pts.reshape(n1, self.n_quad[0], n2, n3, self.n_quad[2]), rhs2)  
        # ====== integrate in 1-2-plane =======
        n1   = self.pts_PI_23[0].size//self.n_quad[0] 
        n2   = self.pts_PI_23[1].size//self.n_quad[1]
        n3   = self.pts_PI_23[2].size
        
        pts  = self.eval_for_PI(23, fun[2], domain)
        
        rhs3 = np.empty((n1, n2, n3), dtype=float)
        ker_glob.kernel_int_2d_ext_eta1_eta2(self.wts[0], self.wts[1], pts.reshape(n1, self.n_quad[0], n2, self.n_quad[1], n3), rhs3)
        
        # total tensor-product right-hand side
        rhs  = np.concatenate((rhs1.flatten(), rhs2.flatten(), rhs3.flatten()))
        
        # ====== solve for coefficients ============
        return self.solve_V2(self.P2.dot(rhs))
    
    
    # ======================================        
    def pi_3(self, fun, domain=None):
        
        n1  = self.pts_PI_3[0].size//self.n_quad[0] 
        n2  = self.pts_PI_3[1].size//self.n_quad[1]
        n3  = self.pts_PI_3[2].size//self.n_quad[2]
        
        pts = self.eval_for_PI(3, fun, domain)
        
        rhs = np.empty((n1, n2, n3), dtype=float)
        ker_glob.kernel_int_3d_ext(self.wts[0], self.wts[1], self.wts[2], pts.reshape(n1, self.n_quad[0], n2, self.n_quad[1], n3, self.n_quad[2]), rhs)
            
        # ====== solve for coefficients ============
        return self.solve_V3(self.P3.dot(rhs.flatten()))