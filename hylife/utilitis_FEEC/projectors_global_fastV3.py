'''
 Module that provides projectors for commuting diagrams in the de Rham sequence, based on
 inter-/histopolation of B-splines at Greville points. 
 Periodic and Dirichlet boundary conditions are available.
 
 Written 2019/20 by Florian Holderied and Stefan Possanner
'''

import numpy              as np
import scipy.sparse       as sparse
from scipy.sparse.linalg import splu

import hylife.utilitis_FEEC.bsplines as bsp
from  hylife.utilitis_FEEC.linalg_kron import kron_lusolve_3d
import hylife.utilitis_FEEC.kernels_projectors_global_V2 as kernels


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

    self.N_LU : list of array_like (sparse)
        LU decomposition of collocation matrices for N-splines in each direction.
             
    self.D_LU : list of array_like (sparse)
        LU decomposition of collocation matrices for D-splines in each direction.
             
    Returns methods (described in detail below)
    ---------------------------------
    self.getpts_for_PI    : get points used for a projection onto a given space
    self.eval_for_PI      : evaluate a callable function at the points used for a projection onto a given space
    self.PI_mat           : Projection onto a given space, calling the routines below
    self.PI_0_mat         : Projection on the space V0 via inter-inter-inter-polation in x1-x2-x3.
    self.PI_11_mat        : FIRST  component of projection on the space V1 via histo-inter-inter-polation.
    self.PI_12_mat        : SECOND component of projection on the space V1 via inter-histo-inter-polation.
    self.PI_13_mat        : THIRD  component of projection on the space V1 via inter-inter-histo-polation.
    self.PI_21_mat        : FIRST  component of projection on the space V2 via inter-histo-histo-polation.
    self.PI_22_mat        : SECOND component of projection on the space V2 via histo-inter-histo-polation.
    self.PI_23_mat        : THIRD  component of projection on the space V2 via histo-histo-inter-polation.
    self.PI_3_mat         : Projection on the space V3 via histo-histo-histo-polation in x1-x2-x3.
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
        
        # collection of the point sets for different 3D projectors
        self.pts_PI_0  = [self.greville[0],
                          self.greville[1],
                          self.greville[2]           ]
        self.pts_PI_11 = [     self.pts[0].flatten(),
                          self.greville[1],
                          self.greville[2]           ]
        self.pts_PI_12 = [self.greville[0],
                               self.pts[1].flatten(),
                          self.greville[2]           ]
        self.pts_PI_13 = [self.greville[0],
                          self.greville[1],
                               self.pts[2].flatten() ]
        self.pts_PI_21 = [self.greville[0],
                               self.pts[1].flatten(),
                               self.pts[2].flatten() ]
        self.pts_PI_22 = [     self.pts[0].flatten(),
                          self.greville[1],
                               self.pts[2].flatten() ]
        self.pts_PI_23 = [     self.pts[0].flatten(),
                               self.pts[1].flatten(),
                          self.greville[2]           ]
        self.pts_PI_3  = [     self.pts[0].flatten(),
                               self.pts[1].flatten(),
                               self.pts[2].flatten() ]


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

        self.N_LU= [splu(self.N[0]),splu(self.N[1]),splu(self.N[2])]
        self.D_LU= [splu(self.D[0]),splu(self.D[1]),splu(self.D[2])]
        
         
    def getpts_for_PI(self,comp):
        '''
        Get the point set for a given projector
        Parameters
        ----------
        comp: which projector, one of (0,11,12,13,21,22,23,3)
        Returns
        -------
        pts_PI : result as list of 1d point sets
        ''' 
        if comp==0:
            pts_PI = self.pts_PI_0
        elif comp==11:
            pts_PI = self.pts_PI_11
        elif comp==12:
            pts_PI = self.pts_PI_12
        elif comp==13:
            pts_PI = self.pts_PI_13
        elif comp==21:
            pts_PI = self.pts_PI_21
        elif comp==22:
            pts_PI = self.pts_PI_22
        elif comp==23:
            pts_PI = self.pts_PI_23
        elif comp==3:
            pts_PI = self.pts_PI_3
        else:
            raise ValueError ("wrong projector specified")

        return pts_PI

    # ======================================        
    def eval_for_PI(self,comp, fun):
        '''
        Evaluates the callable "fun" at the points corresponding to the projector,
        and returns the result as 3d nparray "mat_f".
            
        Parameters
        ----------
        comp: which projector, one of (0,11,12,13,21,22,23,3)
        fun : callable
              fun(x1,x2,x3)
        Returns
        -------
        mat_f : result as 3d nparray 
        '''
        pts_PI = self.getpts_for_PI(comp)
            
        #n = [p_i.size for p_i in pts_PI]
        #
        #mat_f = np.empty( (n[0], n[1], n[2]) )
        # 
        #for i in range(n[0]):
        #    for j in range(n[1]):
        #        for k in range(n[2]):
        #            mat_f[i, j, k] = fun( pts_PI[0][i], pts_PI[1][j], pts_PI[2][k] )
        
        # with meshgrid its much faster!
        pts1,pts2,pts3 = np.meshgrid(pts_PI[0],pts_PI[1],pts_PI[2], indexing='ij',sparse=True ) # numpy >1.7
        #pts1,pts2,pts3 = np.meshgrid(pts_PI[0],pts_PI[1],pts_PI[2], indexing='ij')

        mat_f = np.empty( (pts_PI[0].size, pts_PI[1].size, pts_PI[2].size) )
        mat_f[:, :, :]   = fun(pts1,pts2,pts3)

        return mat_f

    # ======================================        
    def PI_mat(self,comp,mat_f):
        '''
        Call the projector specified with the corresponding point set
        Parameters
        ----------
        comp  : which projector, one of (0,11,12,13,21,22,23,3)
        mat_f : 3d array_like
                values inside the projector, shape(mat_f)=(n1,n2,n3) with n_i=size(pointset_i). 

        Returns
        -------
        coeffs : 3d array_like
            Finite element coefficients obtained by projection.
        '''
        pts_PI = self.getpts_for_PI(comp)
        assert (mat_f.shape == (len(pts_PI[0]), len(pts_PI[1]), len(pts_PI[2])) ) 
        if comp==0:
            coeffs =  self.PI_0_mat(  mat_f)
        elif comp==11:
            coeffs =  self.PI_11_mat( mat_f)
        elif comp==12:
            coeffs =  self.PI_12_mat( mat_f)
        elif comp==13:
            coeffs =  self.PI_13_mat( mat_f)
        elif comp==21:
            coeffs =  self.PI_21_mat( mat_f)
        elif comp==22:
            coeffs =  self.PI_22_mat( mat_f)
        elif comp==23:
            coeffs =  self.PI_23_mat( mat_f)
        elif comp==3:
            coeffs =  self.PI_3_mat(  mat_f)
        else:
            raise ValueError ("wrong projector specified")
            

        return coeffs

    # ======================================        
    def PI_0_mat(self, mat_f):
        
        '''
        Projection on the space V0 via inter(xi1)-inter(xi2)-inter(xi3)-polation.
        
        Parameters
        ----------
        mat_f : 3d array_like
            Right-hand side of the linear system. shape(mat_f)=(n1,n2,n3) with n_i=size(greville_i). 

        Returns
        -------
        coeffs : 3d array_like
            Finite element coefficients obtained by projection.
        '''
                   
        coeffs = kron_lusolve_3d(self.N_LU,mat_f)
        
        return coeffs
        

    # ======================================
    def PI_11_mat(self, mat_f):
        
        '''
        FIRST component of projection on the space V1 via histo(xi1)-inter(xi2)-inter(xi3)-polation.
        
        Parameters
        ----------
        mat_f : 3d array_like, must be reshaped to 4d array
            Right-hand side of the linear system. shape(mat_f)=(ne1*nq1,n2,n3) with
            ne1=number of elements in xi1-direction 
            nq1=quadrature point per element in xi1-direction
            n2=size(greville_xi2)
            n3=size(greville_xi3)
            
            

        Returns
        -------
        coeffs : 3d array_like
            Finite element coefficients obtained by projection.
        '''
        
        ne1 , nq1 = self.wts[0].shape
        n2  = self.pts_PI_11[1].size 
        n3  = self.pts_PI_11[2].size 
        rhs = np.empty( (ne1, n2, n3) )
        
        kernels.kernel_int_1d_ext_xi1(self.wts[0], mat_f.reshape(ne1,nq1,n2,n3), rhs[:, :, :]  )
            
        coeffs = kron_lusolve_3d([self.D_LU[0],self.N_LU[1],self.N_LU[2]],rhs)
        
        return coeffs
    
    # ======================================
    def PI_12_mat(self, mat_f):
        
        '''
        SECOND component of projection on the space V1 via inter(xi1)-histo(xi2)-inter(xi3)-polation.
        
        Parameters
        ----------
        mat_f : 3d array_like, must be reshaped to 4d array
            Right-hand side of the linear system. shape(mat_f)=(n1,ne2*nq2,n3) with
            n1=size(greville_xi1)
            ne2=number of elements in xi2-direction 
            nq2=quadrature point per element in xi2-direction
            n3=size(greville_xi3)
            
            

        Returns
        -------
        coeffs : 3d array_like
            Finite element coefficients obtained by projection.
        '''
        
        n1  = self.pts_PI_12[0].size 
        ne2 , nq2 = self.wts[1].shape
        n3  = self.pts_PI_12[2].size  
        rhs = np.empty( (n1, ne2, n3) )
        
        kernels.kernel_int_1d_ext_xi2(self.wts[1], mat_f.reshape(n1,ne2,nq2,n3), rhs[:, :, :]  )
            
        coeffs = kron_lusolve_3d([self.N_LU[0],self.D_LU[1],self.N_LU[2]],rhs)
        
        return coeffs
    
    # ======================================
    def PI_13_mat(self, mat_f):
        
        '''
        THIRD component of projection on the space V1 via inter(xi1)-inter(xi2)-histo(xi3)-polation.
        
        Parameters
        ----------
        mat_f : 3d array_like, must be reshaped to 4d array
            Right-hand side of the linear system. shape(mat_f)=(n1,n2,ne3*nq3) with
            n1=size(greville_xi1)
            n2=size(greville_xi2)
            ne3=number of elements in xi3-direction 
            nq3=quadrature point per element in xi3-direction
            

        Returns
        -------
        coeffs : 3d array_like
            Finite element coefficients obtained by projection.
        '''
        
        n1  = self.pts_PI_13[0].size 
        n2  = self.pts_PI_13[1].size  
        ne3 , nq3 = self.wts[2].shape
        rhs = np.empty( (n1, n2, ne3 ) )
        
        kernels.kernel_int_1d_ext_xi3(self.wts[2], mat_f.reshape(n1,n2,ne3,nq3), rhs[:, :, :]  )
            
        coeffs = kron_lusolve_3d([self.N_LU[0],self.N_LU[1],self.D_LU[2]],rhs)
        
        return coeffs
        
    
    # ======================================
    def PI_21_mat(self, mat_f):
        
        '''
        FIRST component of projection on the space V2 via inter(xi1)-histo(xi2)-histo(xi3)-polation.
        
        Parameters
        ----------
        mat_f : 3d array_like must be reshaped to 5d array
            Right-hand side of the linear system. shape(mat_f)=(n1,ne2*nq2,ne3*nq3) with
            n1=size(greville_1)
            ne2=number of elements in 2-direction 
            nq2=quadrature point per element in 2-direction
            ne3=number of elements in 3-direction 
            nq3=quadrature point per element in 3-direction
            
        Returns
        -------
        coeffs : 3d array_like
            Finite element coefficients obtained by projection.
        '''
        
        n1  = self.pts_PI_21[0].size
        ne2,nq2   = self.wts[1].shape
        ne3,nq3   = self.wts[2].shape
        rhs = np.empty( (n1, ne2,ne3) )
        
        kernels.kernel_int_2d_ext_xi2_xi3( self.wts[1], self.wts[2], mat_f.reshape(n1,ne2,nq2,ne3,nq3), rhs[:, :, :]  )
            
        coeffs = kron_lusolve_3d([self.N_LU[0],self.D_LU[1],self.D_LU[2]],rhs)
        
        return coeffs
        
    
    # ======================================
    def PI_22_mat(self, mat_f):
        
        '''
        SECOND component of projection on the space V2 via histo(xi1)-inter(xi2)-histo(xi3)-polation.
        
        Parameters
        ----------
        mat_f : 3d array_like must be reshaped to 5d array
            Right-hand side of the linear system. shape(mat_f)=(ne1*nq1,n2,ne3*nq3) with
            ne1=number of elements in 1-direction 
            nq1=quadrature point per element in 1-direction
            n2=size(greville_2)
            ne3=number of elements in 3-direction 
            nq3=quadrature point per element in 3-direction
            
        Returns
        -------
        coeffs : 3d array_like
            Finite element coefficients obtained by projection.
        '''
        
        ne1,nq1   = self.wts[0].shape
        n2  = self.pts_PI_22[1].size
        ne3,nq3   = self.wts[2].shape
        rhs = np.empty( (ne1, n2, ne3) )
        
        kernels.kernel_int_2d_ext_xi1_xi3( self.wts[0], self.wts[2], mat_f.reshape(ne1,nq1,n2,ne3,nq3), rhs[:, :, :]  )
            
        coeffs = kron_lusolve_3d([self.D_LU[0],self.N_LU[1],self.D_LU[2]],rhs)
        
        return coeffs
        
    
    # ======================================
    def PI_23_mat(self, mat_f):
        
        '''
        THIRD component of projection on the space V2 via histo(xi1)-histo(xi2)-inter(xi3)-polation.
        
        Parameters
        ----------
        mat_f : 3d array_like must be reshaped to 5d array
            Right-hand side of the linear system. shape(mat_f)=(ne1*nq1,ne2*nq2,n3) with
            ne1=number of elements in 1-direction 
            nq1=quadrature point per element in 1-direction
            ne2=number of elements in 2-direction 
            nq2=quadrature point per element in 2-direction
            n3=size(greville_3)
            
        Returns
        -------
        coeffs : 3d array_like
            Finite element coefficients obtained by projection.
        '''
        
        ne1,nq1   = self.wts[0].shape
        ne2,nq2   = self.wts[1].shape
        n3  = self.pts_PI_23[2].size
        rhs = np.empty( (ne1,ne2,n3) )
        
        kernels.kernel_int_2d_ext_xi1_xi2( self.wts[0], self.wts[1], mat_f.reshape(ne1,nq1,ne2,nq2,n3), rhs[:, :, :]  )
            
        coeffs = kron_lusolve_3d([self.D_LU[0],self.D_LU[1],self.N_LU[2]],rhs)
        
        return coeffs
        
    
    # ======================================
    def PI_3_mat(self, mat_f):
        
        '''
        Projection on the space V3 via histo(xi1)-histo(xi2)-histo(xi3)-polation.
        
        Parameters
        ----------
        mat_f : 3d arra_like must be reshaped to 6d array
            shape(mat_f)=(ne1*nq1,ne2*nq2,ne3*nq3) with
            ne1=number of elements in 1-direction 
            nq1=quadrature point per element in 1-direction
            ne2=number of elements in 2-direction 
            nq2=quadrature point per element in 2-direction
            ne3=number of elements in 3-direction
            nq3=quadrature point per element in 3-direction
            
        Returns
        -------
        coeffs : 3d array_like
            Finite element coefficients obtained by projection.
        '''
        
        ne1,nq1   = self.wts[0].shape
        ne2,nq2   = self.wts[1].shape
        ne3,nq3   = self.wts[2].shape
        rhs = np.empty((ne1,ne2,ne3))
        
        kernels.kernel_int_3d_ext( self.wts[0], self.wts[1], self.wts[2], mat_f.reshape(ne1,nq1,ne2,nq2,ne3,nq3), rhs[:, :, :]  )
            
        coeffs = kron_lusolve_3d(self.D_LU,rhs)
        
        return coeffs



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
