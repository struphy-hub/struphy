'''
 Module that provides projectors for commuting diagrams in the de Rham sequence, based on
 inter-/histopolation of b-splines at Greville points. 
 Periodic and Dirichlet boundary conditions are available.
 
 Written 2019/20 by Florian Holderied and Stefan Possanner
'''

import numpy              as np
import scipy.sparse       as sparse
from scipy.sparse.linalg import splu

import hylife.utilitis_FEEC.bsplines as bsp
import hylife.utilitis_FEEC.kernels_projectors_global as kernels


# ===================================================================
class projectors_3d:
    
    '''
    Commuting projectors for the 3D de Rham complex, based on inter-/histopolation of b-splines
    at Greville points. 

    Parameters
    ----------
    T : list of 1D array_like
        Knot vectors defining the spline basis.
    
    p : list of int
        Spline degrees.
        
    bc : list of boolean
        Boundary conditions (True = periodic, False = clamped)
        
    Nq : list of int
        Number of quadratute points per element in each direction.
        
    Returns
    -------
    self.T : list of 1D array_like
        Knot vectors defining the spline basis in each direction.
             
    self.p : list of int
        Spline degrees in each direction.
             
    self.bc : list boolean
        Boundary conditions (True = periodic, False = clamped) in each direction.
        
    self.Nq : list of int
        Number of quadratute points per element in each direction.
              
    self.el_b : list of 1D array_like
        Element boundaries in each direction.
        
    self.greville : list of 1D array_like
        Greville points in each direction.
                
    self.Nel : list of int
        Number of elements.
              
    self.NbaseN : list of int
        Number of N-spline basis functions.
                  
    self.NbaseD : list of int
        Number of D-spline basis functions. 
                
    self.quad_loc : list of array_like
        Local Gauss-Legendre quadrature points and weights
                    
    self.delta : list of float
        Grid spacings.
                 
    self.grid : list of array_like
        Grid points.
                
    self.pts : list of array_like
        Gauss-Legendre quadrature points between Greville points.
               
    self.wts : list of array_like
        Gauss-Legendre quadrature weights between Greville points.
               
    self.N : list of array_like (sparse)
        Collocation matrices for N-splines in each direction.
             
    self.D : list of array_like (sparse)
        Collocation matrices for D-splines in each direction.
             
    Returns methods (described in detail below)
    ---------------------------------
    self.NNN_LU
    self.DNN_LU
    self.NDN_LU
    self.NND_LU
    self.NDD_LU
    self.DND_LU
    self.DDN_LU
    self.DDD_LU
    self.PI_0  : Projection on the space V0 via inter-inter-inter-polation in x1-x2-x3.
    self.PI_11 : FIRST  component of projection on the space V1 via histo-inter-inter-polation.
    self.PI_12 : SECOND component of projection on the space V1 via inter-histo-inter-polation.
    self.PI_13 : THIRD  component of projection on the space V1 via inter-inter-histo-polation.
    self.PI_21 : FIRST  component of projection on the space V2 via inter-histo-histo-polation.
    self.PI_22 : SECOND component of projection on the space V2 via histo-inter-histo-polation.
    self.PI_23 : THIRD  component of projection on the space V2 via histo-histo-inter-polation.
    self.PI_3  : Projection on the space V3 via histo-histo-histo-polation in x1-x2-x3.
    '''
    
    def __init__(self, T, p, bc, Nq):
        
        self.T         = T
        self.p         = p
        self.bc        = bc
        self.Nq        = Nq
        self.el_b      = [bsp.breakpoints(T_i, p_i) for T_i, p_i in zip(T, p)]
        self.greville  = [bsp.greville(T_i, p_i, bc_i) for T_i, p_i, bc_i in zip(T, p, bc)]
        self.Nel       = [len(el_b_i) - 1 for el_b_i in self.el_b]
        self.NbaseN    = [Nel_i + p_i - bc_i*p_i for Nel_i, p_i, bc_i in zip(self.Nel, p, bc)]
        self.NbaseD    = [NbaseN_i - (1 - bc_i) for NbaseN_i, bc_i in zip(self.NbaseN, bc)]
        self.quad_loc  = [np.polynomial.legendre.leggauss(Nq_i) for Nq_i in Nq] 
        self.delta     = [el_b_i[1] - el_b_i[0] for el_b_i in self.el_b] 

        # Quadrature grids in cells defined by consecutive Greville points in each direction
        self.pts = []
        self.wts = []
        for a in range(3):
            
            if self.bc[a]:
                xgrid = np.append( self.greville[a], self.el_b[a][-1] + self.greville[a][0] )
            else:
                xgrid = self.greville[a]
            
            pts_a, wts_a = bsp.quadrature_grid( xgrid , self.quad_loc[a][0],
                                                        self.quad_loc[a][1] )
            
            self.pts.append(pts_a)
            self.wts.append(wts_a)
        
        # Collocation matrices for N-splines in each direction
        self.N = [sparse.csc_matrix(bsp.collocation_matrix(T_i, p_i, greville_i, bc_i)) 
                  for T_i, p_i, greville_i, bc_i in zip(self.T, self.p, self.greville, self.bc)]
        
        # evaluation of basis at quadrature points pts
        #self.Nq = [sparse.csc_matrix(bsp.collocation_matrix(T_i, p_i, pts_i.flatten(), bc_i)) 
        #          for T_i, p_i, greville_i, bc_i in zip(self.T, self.p, self.pts, self.bc)]
        
        # Histopolation matrices for D-splines in each direction
        self.D = [sparse.csc_matrix(bsp.histopolation_matrix(T_i[1:-1], p_i-1, greville_i, bc_i, True)) 
                  for T_i, p_i, greville_i, bc_i in zip(self.T, self.p, self.greville, self.bc)]
        
        # Histopolation basis evaluated at quadrature points pts
        #self.Dq = [sparse.csc_matrix(bsp.histopolation_matrix(T_i[1:-1], p_i-1, pts_i.flatten(), bc_i, True))  #
        #          for T_i, p_i, greville_i, bc_i in zip(self.T, self.p, self.pts, self.bc)]
        
         
    # ======================================
    def NNN_LU(self):
        
        '''
        LU decompostion of Kronecker product of collocation matrices in x1, x2, x3 in sparse format.
        '''

        self.NNN_LU = splu( sparse.kron( sparse.kron( self.N[0], self.N[1] ), self.N[2], format='csc' ) )
        
    # ======================================
    def DNN_LU(self):
        
        '''
        LU decompostion of Kronecker product of collocation matrices in x2, x3 and
        histopolation matrix in x1 in sparse format.
        '''
        
        self.DNN_LU = splu( sparse.kron( sparse.kron( self.D[0], self.N[1] ), self.N[2], format='csc' ) )
        
    # ======================================
    def NDN_LU(self):
        
        '''
        LU decompostion of Kronecker product of collocation matrices in x1, x3 and 
        histopolation matrix in x2 in sparse format.
        '''

        self.NDN_LU = splu( sparse.kron( sparse.kron( self.N[0], self.D[1] ), self.N[2], format='csc' ) )
        
    # ======================================
    def NND_LU(self):
        
        '''
        LU decompostion of Kronecker product of collocation matrices in x1, x2 and
        histopolation matrix in x3 in sparse format.
        '''

        self.NND_LU = splu( sparse.kron( sparse.kron( self.N[0], self.N[1] ), self.D[2], format='csc' ) )
        
    # ======================================
    def NDD_LU(self):
        
        '''
        LU decompostion of Kronecker product of collocation matrix in x1 and 
        histopolation matrices in x2, x3 in sparse format.
        '''

        self.NDD_LU = splu( sparse.kron( sparse.kron( self.N[0], self.D[1] ), self.D[2], format='csc' ) )
        
    # ======================================
    def DND_LU(self):
        
        '''
        LU decompostion of Kronecker product of collocation matrix in x2 and 
        histopolation matrices in x1, x3 in sparse format.
        '''

        self.DND_LU = splu( sparse.kron( sparse.kron( self.D[0], self.N[1] ), self.D[2], format='csc' ) )
        
    # ======================================
    def DDN_LU(self):
        
        '''
        LU decompostion of Kronecker product of collocation matrix in x3 and
        histopolation matrices in x1, x2 in sparse format.
        '''
        
        self.DDN_LU = splu( sparse.kron( sparse.kron( self.D[0], self.D[1] ), self.N[2], format='csc' ) )
        
    # ======================================
    def DDD_LU(self):
        
        '''
        LU decompostion of Kronecker product of histopolation matrices in x1, x2, x3 in sparse format.
        '''

        self.DDD_LU = splu( sparse.kron( sparse.kron( self.D[0], self.D[1] ), self.D[2], format='csc' ) )

# ======================================
    def assemble_V0(self):
        
        self.NNN_LU()
        
    # ======================================
    def assemble_V1(self):
        
        self.DNN_LU()
        self.NDN_LU()
        self.NND_LU()
        
    
    # ======================================    
    def assemble_V2(self):
                
        self.NDD_LU()
        self.DND_LU()
        self.DDN_LU()
        
    
    # ======================================
    def assemble_V3(self):
        
        self.DDD_LU()

    # ======================================        
    def PI_0(self, fun):
        
        '''
        Projection on the space V0 via inter-inter-inter-polation in x1-x2-x3.
        
        Parameters
        ----------
        fun : callable
            fun(x1,x2,x3) \in R is the 0-form to be projected.

        Returns
        -------
        coeffs : 3D array_like
            Finite element coefficients obtained by projection.
        '''
        
        n = [greville.size for greville in self.greville]
        
        rhs = np.empty( (n[0], n[1], n[2]) )
        
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    rhs[i, j, k] = fun( self.greville[0][i], self.greville[1][j], self.greville[2][k] )
                               
        coeffs = self.NNN_LU.solve(rhs.flatten())
        
        return coeffs.reshape( self.NbaseN[0], self.NbaseN[1], self.NbaseN[2] )
    
    # ======================================
    def PI_11(self, fun):
        
        '''
        FIRST component of projection on the space V1 via histo-inter-inter-polation in x1-x2-x3.
        
        Parameters
        ----------
        fun : callable
            fun(x1,x2,x3) \in R is the FIRST component of the 1-form to be projected.

        Returns
        -------
        coeffs : 3D array_like
            Finite element coefficients obtained by projection.
        '''
        
        n = [greville.size for greville in self.greville]
            
        rhs = np.empty( (n[0] - 1 + self.bc[0], n[1], n[2]) )
                    
        for j in range(n[1]):
            for k in range(n[2]):

                integrand = lambda xi1 : fun( xi1, self.greville[1][j], self.greville[2][k] )

                rhs[:, j, k] = integrate_1d(self.pts[0], self.wts[0], integrand)


        ## alternative, using a 3D kernel
        #
        #integrand = np.empty((self.pts[0].shape[0], n[1], n[2],self.pts[0].shape[1]))
        #for i in range(self.pts[0].shape[0]):
        #    for j in range(n[1]):
        #        for k in range(n[2]):
        #            for p in range(self.pts[0].shape[1]):
        #               integrand[i,j,k,p]=fun[0](self.pts[0,i,p],self.greville[1][j], self.greville[2][k])
        #
        #kernel_int_1d_ext_xi1(self.wts[0], integrand, rhs[0][:, :, :]  )
                
        coeffs = self.DNN_LU.solve(rhs.flatten()) 
        
        return coeffs.reshape( self.NbaseD[0], self.NbaseN[1], self.NbaseN[2] )
    
    # ======================================
    def PI_12(self, fun):
        
        '''
        SECOND component of projection on the space V1 via inter-histo-inter-polation in x1-x2-x3.
        
        Parameters
        ----------
        fun : callable
            fun(x1,x2,x3) \in R is the SECOND component of the 1-form to be projected.

        Returns
        -------
        coeffs : 3D array_like
            Finite element coefficients obtained by projection.
        '''
        
        n = [greville.size for greville in self.greville]
            
        rhs = np.empty( (n[0], n[1] - 1 + self.bc[1], n[2]) )

        for i in range(n[0]):
            for k in range(n[2]):

                integrand = lambda xi2 : fun( self.greville[0][i], xi2, self.greville[2][k] )

                rhs[i, :, k] = integrate_1d(self.pts[1], self.wts[1], integrand)
                
        coeffs = self.NDN_LU.solve(rhs.flatten()) 
        
        return  coeffs.reshape( self.NbaseN[0], self.NbaseD[1], self.NbaseN[2] )
    
    # ======================================
    def PI_13(self, fun):
        
        '''
        THIRD component of projection on the space V1 via inter-inter-histo-polation in x1-x2-x3.
        
        Parameters
        ----------
        fun : callable
            fun(x1,x2,x3) \in R is the THIRD component of the 1-form to be projected.

        Returns
        -------
        coeffs : 3D array_like
            Finite element coefficients obtained by projection.
        '''
        
        n = [greville.size for greville in self.greville]
            
        rhs = np.empty( (n[0], n[1], n[2] - 1 + self.bc[2]) )
                    
        for i in range(n[0]):
            for j in range(n[1]):

                integrand = lambda xi3 : fun( self.greville[0][i], self.greville[1][j], xi3 )

                rhs[i, j, :] = integrate_1d(self.pts[2], self.wts[2], integrand)
                
        coeffs = self.NND_LU.solve(rhs.flatten()) 
        
        return coeffs.reshape( self.NbaseN[0], self.NbaseN[1], self.NbaseD[2] )
    
    # ======================================
    def PI_21(self, fun):
        
        '''
        FIRST component of projection on the space V2 via inter-histo-histo-polation in x1-x2-x3.
        
        Parameters
        ----------
        fun : callable
            fun(x1,x2,x3) \in R is the FIRST component of the 2-form to be projected.

        Returns
        -------
        coeffs : 3D array_like
            Finite element coefficients obtained by projection.
        '''
        
        n = [greville.size for greville in self.greville]
        
        rhs = np.empty( (n[0], n[1] - 1 + self.bc[1], n[2] - 1 + self.bc[2]) )
        
        for i in range(n[0]):
            
            integrand = lambda xi2, xi3 : fun( self.greville[0][i], xi2, xi3 )
            
            rhs[i, :, :] = integrate_2d([self.pts[1], self.pts[2]], 
                                        [self.wts[1], self.wts[2]], integrand )
            
        coeffs = self.NDD_LU.solve(rhs.flatten()) 
        
        return coeffs.reshape( self.NbaseN[0], self.NbaseD[1], self.NbaseD[2] )
    
    # ======================================
    def PI_22(self, fun):
        
        '''
        SECOND component of projection on the space V2 via histo-inter-histo-polation in x1-x2-x3.
        
        Parameters
        ----------
        fun : callable
            fun(x1,x2,x3) \in R is the SECOND component of the 2-form to be projected.

        Returns
        -------
        coeffs : 3D array_like
            Finite element coefficients obtained by projection.
        '''
        
        n = [greville.size for greville in self.greville]
        
        rhs = np.empty( (n[0] - 1 + self.bc[0], n[1], n[2] - 1 + self.bc[2]) )
            
        for j in range(n[1]):
            
            integrand = lambda xi1, xi3 : fun(xi1, self.greville[1][j], xi3)
            
            rhs[:, j, :] = integrate_2d([self.pts[0], self.pts[2]], 
                                        [self.wts[0], self.wts[2]], integrand)
        
        coeffs = self.DND_LU.solve(rhs.flatten())
        
        return coeffs.reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseD[2])
    
    # ======================================
    def PI_23(self, fun):
        
        '''
        THIRD component of projection on the space V2 via histo-histo-inter-polation in x1-x2-x3.
        
        Parameters
        ----------
        fun : callable
            fun(x1,x2,x3) \in R is the THIRD component of the 2-form to be projected.

        Returns
        -------
        coeffs : 3D array_like
            Finite element coefficients obtained by projection.
        '''
        
        n = [greville.size for greville in self.greville]
        
        rhs = np.empty((n[0] - 1 + self.bc[0], n[1] - 1 + self.bc[1], n[2]))
            
        for k in range(n[2]):
            
            integrand = lambda xi1, xi2 : fun(xi1, xi2, self.greville[2][k])
            
            rhs[:, :, k] = integrate_2d([self.pts[0], self.pts[1]], 
                                        [self.wts[0], self.wts[1]], integrand)
            
        coeffs = self.DDN_LU.solve(rhs.flatten())
        
        return coeffs.reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])
    
    # ======================================
    def PI_3(self, fun):
        
        '''
        Projection on the space V3 via histo-histo-histo-polation in x1-x2-x3.
        
        Parameters
        ----------
        fun : callable
            fun(x1,x2,x3) \in R is the 3-form to be projected.

        Returns
        -------
        coeffs : 3D array_like
            Finite element coefficients obtained by projection.
        '''

        n = [greville.size for greville in self.greville]

        rhs = np.empty((n[0] - 1 + self.bc[0], n[1] - 1 + self.bc[1], n[2] - 1 + self.bc[2]))

        rhs[:, :, :] = integrate_3d([self.pts[0], self.pts[1], self.pts[2]],
                                    [self.wts[0], self.wts[1], self.wts[2]], fun)

        coeffs = self.DDD_LU.solve(rhs.flatten())

        return coeffs.reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseD[2])
# ===================================================================



# ===================================================================
class projectors_1d:
    
    '''
    Projectors for the 1D commuting diagram, based on inter-/histopolation of b-splines
    at Greville points. 
    
    Written 2019/20 by Florian Holderied, Stefan Possanner 
    
    Parameters
    ----------
    T : 1D array_like
        Knot vectors defining the spline basis.
    
    p : int
        Spline degrees.
        
    bc : boolean
        Boundary conditions (True = periodic, False = clamped)
        
    Nq : int
        Number of quadratute points per element.
        
    Returns
    -------
    self.T : 1D array_like
        Knot vectors defining the spline basis.
             
    self.p : int
        Spline degree.
             
    self.bc : boolean
        Boundary conditions (True = periodic, False = clamped).
        
    self.Nq : int
        Number of quadrature points per element.
              
    self.el_b : 1D array_like
        Element boundaries.
        
    self.greville : 1D array_like
        Greville points.
                
    self.Nel : int
        Number of elements.
              
    self.NbaseN : int
        Number of N-spline basis functions.
                  
    self.NbaseD : int
        Number of D-spline basis functions. 
                
    self.quad_loc : array_like
        Local Gauss-Legendre quadrature points and weights
                    
    self.delta : float
        Grid spacings.
                 
    self.grid : array_like
        Grid points.
                
    self.pts : array_like
        Gauss-Legendre quadrature points between Greville points.
               
    self.wts : array_like
        Gauss-Legendre quadrature weights between Greville points.
               
    self.N : array_like (sparse)
        Collocation matrix for N-splines.
             
    self.D : array_like (sparse)
        Collocation matrix for D-splines.
             
    Returns methods (described in detail below)
    ---------------------------------
    self.N_LU
    self.D_LU
    self.PI_0  : Projection on the space V0 via interpolation.
    self.PI_1  : Projection on the space V1 via histopolation.
    '''
    
    def __init__(self, T, p, bc, Nq):
        
        self.T         = T
        self.p         = p
        self.bc        = bc
        self.Nq        = Nq
        self.el_b      = bsp.breakpoints(self.T, self.p)
        self.greville  = bsp.greville(self.T, self.p, self.bc)
        self.Nel       = len(self.el_b) - 1
        self.NbaseN    = self.Nel + self.p - self.bc*self.p
        self.NbaseD    = self.NbaseN - (1 - self.bc)
        self.quad_loc  = np.polynomial.legendre.leggauss(self.Nq)
        self.delta     = self.el_b[1] - self.el_b[0]
         
        # Quadrature grids in cells defined by consecutive Greville points   
        if bc:
            xgrid = np.append( self.greville, self.el_b[-1] + self.greville[0] )
        else:
            xgrid = self.greville

        self.pts, self.wts = bsp.quadrature_grid( xgrid , self.quad_loc[0], self.quad_loc[1] )
        
        # Collocation matrix for N-splines and its LU decomposition in sparse format 
        self.N    = sparse.csc_matrix(bsp.collocation_matrix(T, p, self.greville, bc)) 
        self.N_LU = splu(self.N)
        # Histopolation matrix for D-splines and its LU decomposition in sparse format
        self.D = sparse.csc_matrix(bsp.histopolation_matrix(T[1:-1], p-1, self.greville, bc, True)) 
        self.D_LU = splu(self.D)          
    
    # ======================================        
    def PI_0(self, fun):
        
        '''
        Projection on the space V0 via interpolation.
        
        Parameters
        ----------
        fun : callable
            fun(x) \in R is the 0-form to be projected.

        Returns
        -------
        coeffs : 1D array_like
            Finite element coefficients obtained by projection.
        '''
        
        n = self.greville.size
        
        rhs = np.empty(n)
        
        for i in range(n):
            rhs[i] = fun(self.greville[i])
                              
        return self.N_LU.solve(rhs)
    
    
    # ======================================
    def PI_1(self, fun):
        
        '''
        Projection on the space V1 via histopolation.
        
        Parameters
        ----------
        fun : callable
            fun(x) \in R is the 1-form to be projected.

        Returns
        -------
        coeffs : 1D array_like
            Finite element coefficients obtained by projection.
        '''

        n = self.greville.size

        rhs = np.empty(n - 1 + self.bc)

        rhs[:] = integrate_1d(self.pts, self.wts, fun)

        return self.D_LU.solve(rhs)
# ===================================================================



# ===================================================================
def integrate_1d(points, weights, fun):
    """
    Integrates the function 'fun' over the quadrature grid defined by (points, weights) in 1d.
    
    Parameters
    ----------
    points : 2d np.array
        quadrature points in format (element, local point)
        
    weights : 2d np.array
        quadrature weights in format (element, local point)
    
    fun : callable
        1d function to be integrated
        
    Returns
    -------
    f_int : np.array
        the value of the integration in each element
    """
    
    n1  = points.shape[0]
    nq1 = points.shape[1]
    
    f_int = np.empty(n1)
    mat_f = np.empty(nq1)
    
    for ie1 in range(n1):
        
        w1   = weights[ie1, :]
        #Pts1 = points[ie1, :]
        #mat_f[:] = fun(Pts)
        mat_f[:] = fun(points[ie1, :])
        
        #for i in range(nq1):
        #    mat_f[i] = fun(points[ie1,i])
        
        f_int[ie1] = kernels.kernel_int_1d(nq1, w1, mat_f)
            
    return f_int
# ===================================================================



# ===================================================================
def integrate_2d(points, weights, fun):
    """
    Integrates the function 'fun' over the quadrature grid defined by (points, weights) in 2d.
    
    Parameters
    ----------
    points : list of 2d np.arrays
        quadrature points in format (element, local point)
        
    weights : list of 2d np.arrays
        quadrature weights in format (element, local point)
    
    fun : callable
        2d function to be integrated
        
    Returns
    -------
    f_int : np.array
        the value of the integration in each element
    """
    
    pts1, pts2 = points
    wts1, wts2 = weights
    
    n1 = pts1.shape[0]
    n2 = pts2.shape[0]
    
    nq1 = pts1.shape[1]
    nq2 = pts2.shape[1]
        
    f_int = np.empty((n1, n2))
    mat_f = np.empty((nq1, nq2))
    f_loc = np.array([0.])
    
    for ie1 in range(n1):
        for ie2 in range(n2):
            
            w1 = wts1[ie1, :]
            w2 = wts2[ie2, :]
            
            Pts1, Pts2 = np.meshgrid(pts1[ie1, :], pts2[ie2, :], indexing='ij')
            mat_f[:, :] = fun(Pts1, Pts2)
            #mat_f[:, :] = fun(pts1[ie1], pts2[ie2])
            #for i, xi in enumerate(pts1[ie1, :]):
                #for j, yj in enumerate(pts2[ie2, :]):
                    #mat_f[i,j] = fun(xi,yj)

            #kernels.kernel_int_2d_new(nq1, nq2, w1, w2, mat_f, f_loc)
            #f_int[ie1, ie2] = f_loc
            f_int[ie1, ie2] = kernels.kernel_int_2d(nq1, nq2, w1, w2, mat_f)
            
    return f_int
# ===================================================================



# ===================================================================
def integrate_3d(points, weights, fun):
    """
    Integrates the function 'fun' over the quadrature grid defined by (points, weights) in 3d.
    
    Parameters
    ----------
    points : list of 2d np.arrays
        quadrature points in format (element, local point)
        
    weights : list of 2d np.arrays
        quadrature weights in format (element, local point)
    
    fun : callable
        3d function to be integrated
        
    Returns
    -------
    f_int : np.array
        the value of the integration in each element
    """
    
    pts1, pts2, pts3 = points
    wts1, wts2, wts3 = weights
    
    n1 = pts1.shape[0]
    n2 = pts2.shape[0]
    n3 = pts3.shape[0]
    
    nq1 = pts1.shape[1]
    nq2 = pts2.shape[1]
    nq3 = pts3.shape[1]
      
    f_int = np.empty((n1, n2, n3))
    mat_f = np.empty((nq1, nq2, nq3))
    
    for ie1 in range(n1):
        for ie2 in range(n2):
            for ie3 in range(n3):
            
                w1 = wts1[ie1, :]
                w2 = wts2[ie2, :]
                w3 = wts3[ie3, :]

                Pts1, Pts2, Pts3 = np.meshgrid(pts1[ie1, :], pts2[ie2, :], pts3[ie3, :], indexing='ij')
                mat_f[:, :, :]   = fun(Pts1, Pts2, Pts3)
                #mat_f[:, :, :] = fun(pts1[ie1], pts2[ie2], pts3[ie3])
                
                #for i, xi in enumerate(pts1[ie1, :]):
                #    for j, yj in enumerate(pts2[ie2, :]):
                #        for k, zj in enumerate(pts3[ie3, :]):
                #            mat_f[i,j,k] = fun(xi,yj,zj)

                f_int[ie1, ie2, ie3] = kernels.kernel_int_3d(nq1, nq2, nq3, w1, w2, w3, mat_f)

    return f_int
# ===================================================================
