import numpy as np
import scipy.sparse as sparse

from scipy.sparse.linalg import splu

import utilitis_FEEC.bsplines           as bsp
import utilitis_FEEC.kernels_projectors as kernels

from pyccel import epyccel
kernels = epyccel(kernels)


#============================================================================================================================== 
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
    f_loc = np.array([0.])
    
    for ie1 in range(n1):
        
        w1   = weights[ie1, :]
        Pts1 = points[ie1, :]
        
        mat_f[:] = fun(Pts1)
        
        kernels.kernel_int_1d(nq1, w1, mat_f, f_loc)
        
        f_int[ie1] = f_loc
        
    return f_int
#==============================================================================================================================



#==============================================================================================================================
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
    
    
    f_int = np.empty((n1, n2), order='F')
    mat_f = np.empty((nq1, nq2), order='F')
    f_loc = np.array([0.])
    
    for ie1 in range(n1):
        for ie2 in range(n2):
            
            w1 = wts1[ie1, :]
            w2 = wts2[ie2, :]
            
            Pts1, Pts2 = np.meshgrid(pts1[ie1, :], pts2[ie2, :], indexing='ij')
            mat_f[:, :] = fun(Pts1, Pts2)
            
            kernels.kernel_int_2d(nq1, nq2, w1, w2, mat_f, f_loc)
            
            f_int[ie1, ie2] = f_loc
                     
    return f_int
#==============================================================================================================================



#==============================================================================================================================
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
    
    
    f_int = np.empty((n1, n2, n3), order='F')
    mat_f = np.empty((nq1, nq2, nq3), order='F')
    f_loc = np.array([0.])
    
    for ie1 in range(n1):
        for ie2 in range(n2):
            for ie3 in range(n3):
            
                w1 = wts1[ie1, :]
                w2 = wts2[ie2, :]
                w3 = wts3[ie3, :]

                Pts1, Pts2, Pts3 = np.meshgrid(pts1[ie1, :], pts2[ie2, :], pts3[ie3, :], indexing='ij')
                mat_f[:, :, :] = fun(Pts1, Pts2, Pts3)

                kernels.kernel_int_3d(nq1, nq2, nq3, w1, w2, w3, mat_f, f_loc)

                f_int[ie1, ie2, ie3] = f_loc
                     
    return f_int
#==============================================================================================================================



#==============================================================================================================================
def histopolation_matrix_1d(p, Nbase, T, grev, bc):
    """
    Computest the 1d histopolation matrix.
    
    Parameters
    ----------
    p : int
        spline degree
        
    Nbase : int
        number of spline functions
    
    T : np.array 
        knot vector
    
    grev : np.array
        greville points
        
    bc : boolean
        boundary conditions (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions)
        
    Returns
    -------
    D : 2d np.array
        histopolation matrix
    """
    
    el_b = bsp.breakpoints(T, p)
    Nel = len(el_b) - 1
    t = T[1:-1]
    
    if bc == True:
        
        pts_loc, wts_loc = np.polynomial.legendre.leggauss(p - 1)
        
        ne = Nbase - p
        D = np.zeros((ne, ne))
        
        if p%2 != 0:
            
            grid = el_b
            
            pts, wts = bsp.quadrature_grid(grid, pts_loc, wts_loc)
            col_quad = bsp.collocation_matrix(t, p - 1, pts.flatten(), bc, normalize=True)
            
            for ie in range(Nel):
                for il in range(p):
                    
                    i = (ie + il)%ne
                    
                    for k in range(p - 1):
                        D[ie, i] += wts[ie, k]*col_quad[ie*(p - 1) + k, i]

            return D
        
        else:
            
            grid = np.linspace(0., el_b[-1], 2*Nel + 1)
            
            pts, wts = bsp.quadrature_grid(grid, pts_loc, wts_loc)
            col_quad = bsp.collocation_matrix(t, p - 1, pts.flatten(), bc, normalize=True)
            
            for iee in range(2*Nel):
                for il in range(p):
                    
                    ie = int(iee/2)
                    ie_grev = int(np.ceil(iee/2) - 1)
                    
                    i = (ie + il)%ne
                    
                    for k in range(p - 1):
                        D[ie_grev, i] += wts[iee, k]*col_quad[iee*(p - 1) + k, i]

            return D
        
    else:
        
        ng = len(grev)
        
        col_quad = bsp.collocation_matrix(T, p, grev, bc)
        
        D = np.zeros((ng - 1, Nbase - 1))
        
        for i in range(ng - 1):
            for j in range(max(i - p + 1, 1), min(i + p + 3, Nbase)):
                s = 0.
                for k in range(j):
                    s += col_quad[i, k] - col_quad[i + 1, k]
                    
                D[i, j - 1] = s
                
        return D
#==============================================================================================================================



#==============================================================================================================================
class projectors_3d:
    
    def __init__(self, p, Nbase, T, bc):
        
        self.p  = p
        self.Nbase = Nbase
        self.T  = T
        self.bc = bc
        
        p1, p2, p3 = p
        Nbase_1, Nbase_2, Nbase_3 = Nbase
        T1, T2, T3 = T
        bc_1, bc_2, bc_3 = bc
        
        
        self.greville = [bsp.greville(T1, p1, bc_1), bsp.greville(T2, p2, bc_2), bsp.greville(T3, p3, bc_3)]
        self.breakpoints = [bsp.breakpoints(T1, p1), bsp.breakpoints(T2, p2), bsp.breakpoints(T3, p3)]
        
        pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
        pts_2_loc, wts_2_loc = np.polynomial.legendre.leggauss(p2 + 1)
        pts_3_loc, wts_3_loc = np.polynomial.legendre.leggauss(p3 + 1)
        
        self.pts_loc = [pts_1_loc, pts_2_loc, pts_3_loc]
        self.wts_loc = [wts_1_loc, wts_2_loc, wts_3_loc]
        
        
        
    def assemble_V0(self):
        
        N1 = sparse.csc_matrix(bsp.collocation_matrix(self.T[0], self.p[0], self.greville[0], self.bc[0])) 
        N2 = sparse.csc_matrix(bsp.collocation_matrix(self.T[1], self.p[1], self.greville[1], self.bc[1])) 
        N3 = sparse.csc_matrix(bsp.collocation_matrix(self.T[2], self.p[2], self.greville[2], self.bc[2])) 
        
        self.interhistopolation_V0 = splu(sparse.kron(sparse.kron(N1, N2), N3, format='csc'))
     
    
    def assemble_V1(self):
        
        N1 = sparse.csc_matrix(bsp.collocation_matrix(self.T[0], self.p[0], self.greville[0], self.bc[0])) 
        N2 = sparse.csc_matrix(bsp.collocation_matrix(self.T[1], self.p[1], self.greville[1], self.bc[1])) 
        N3 = sparse.csc_matrix(bsp.collocation_matrix(self.T[2], self.p[2], self.greville[2], self.bc[2]))
        
        D1 = sparse.csc_matrix(histopolation_matrix_1d(self.p[0], self.Nbase[0], self.T[0], self.greville[0], self.bc[0]))
        D2 = sparse.csc_matrix(histopolation_matrix_1d(self.p[1], self.Nbase[1], self.T[1], self.greville[1], self.bc[1]))
        D3 = sparse.csc_matrix(histopolation_matrix_1d(self.p[2], self.Nbase[2], self.T[2], self.greville[2], self.bc[2]))
        
        self.interhistopolation_V1_1 = splu(sparse.kron(sparse.kron(D1, N2), N3, format='csc'))
        self.interhistopolation_V1_2 = splu(sparse.kron(sparse.kron(N1, D2), N3, format='csc'))
        self.interhistopolation_V1_3 = splu(sparse.kron(sparse.kron(N1, N2), D3, format='csc'))
        
    
    def assemble_V2(self):
        
        N1 = sparse.csc_matrix(bsp.collocation_matrix(self.T[0], self.p[0], self.greville[0], self.bc[0])) 
        N2 = sparse.csc_matrix(bsp.collocation_matrix(self.T[1], self.p[1], self.greville[1], self.bc[1])) 
        N3 = sparse.csc_matrix(bsp.collocation_matrix(self.T[2], self.p[2], self.greville[2], self.bc[2]))
        
        D1 = sparse.csc_matrix(histopolation_matrix_1d(self.p[0], self.Nbase[0], self.T[0], self.greville[0], self.bc[0]))
        D2 = sparse.csc_matrix(histopolation_matrix_1d(self.p[1], self.Nbase[1], self.T[1], self.greville[1], self.bc[1]))
        D3 = sparse.csc_matrix(histopolation_matrix_1d(self.p[2], self.Nbase[2], self.T[2], self.greville[2], self.bc[2]))
        
        self.interhistopolation_V2_1 = splu(sparse.kron(sparse.kron(N1, D2), D3, format='csc'))
        self.interhistopolation_V2_2 = splu(sparse.kron(sparse.kron(D1, N2), D3, format='csc'))
        self.interhistopolation_V2_3 = splu(sparse.kron(sparse.kron(D1, D2), N3, format='csc'))
        
        
    def assemble_V3(self):
        
        D1 = sparse.csc_matrix(histopolation_matrix_1d(self.p[0], self.Nbase[0], self.T[0], self.greville[0], self.bc[0]))
        D2 = sparse.csc_matrix(histopolation_matrix_1d(self.p[1], self.Nbase[1], self.T[1], self.greville[1], self.bc[1]))
        D3 = sparse.csc_matrix(histopolation_matrix_1d(self.p[2], self.Nbase[2], self.T[2], self.greville[2], self.bc[2]))
        
        self.interhistopolation_V3 = splu(sparse.kron(sparse.kron(D1, D2), D3, format='csc'))
        
    
    def PI_0(self, fun):
        
        n1 = self.greville[0].size
        n2 = self.greville[1].size
        n3 = self.greville[2].size
        
        rhs = np.empty((n1, n2, n3))
        
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    rhs[i, j, k] = fun(self.greville[0][i], self.greville[1][j], self.greville[2][k])
                    
        vec0 = self.interhistopolation_V0.solve(np.reshape(rhs, n1*n2*n3))
        
        return vec0
    
    
    def PI_1(self, fun):
        
        n1 = self.greville[0].size
        n2 = self.greville[1].size
        n3 = self.greville[2].size
        
        
        # Quadrature grid in 1-direction
        if self.bc[0] == True:
            
            if self.p[0]%2 != 0:
                
                grid_1 = self.breakpoints[0]
                pts_1, wts_1 = bsp.quadrature_grid(grid_1, self.pts_loc[0], self.wts_loc[0])
                
            else:
                
                grid_1 = np.append(self.greville[0], self.greville[0][-1] + (self.greville[0][-1] - self.greville[0][-2]))
                pts_1, wts_1 = bsp.quadrature_grid(grid_1, self.pts_loc[0], self.wts_loc[0])%self.breakpoints[0][-1]
                
        else:
            
            pts_1, wts_1 = bsp.quadrature_grid(self.greville[0], self.pts_loc[0], self.wts_loc[0])
            
            
        # Quadrature grid in 2-direction
        if self.bc[1] == True:
            
            if self.p[1]%2 != 0:
                
                grid_2 = self.breakpoints[1]
                pts_2, wts_2 = bsp.quadrature_grid(grid_2, self.pts_loc[1], self.wts_loc[1])
                
            else:
                
                grid_2 = np.append(self.greville[1], self.greville[1][-1] + (self.greville[1][-1] - self.greville[1][-2]))
                pts_2, wts_2 = bsp.quadrature_grid(grid_2, self.pts_loc[1], self.wts_loc[1])%self.breakpoints[1][-1]
                
        else:
            
            pts_2, wts_2 = bsp.quadrature_grid(self.greville[1], self.pts_loc[1], self.wts_loc[1])
            
            
        # Quadrature grid in 3-direction
        if self.bc[2] == True:
            
            if self.p[2]%2 != 0:
                
                grid_3 = self.breakpoints[2]
                pts_3, wts_3 = bsp.quadrature_grid(grid_3, self.pts_loc[2], self.wts_loc[2])
                
            else:
                
                grid_3 = np.append(self.greville[2], self.greville[2][-1] + (self.greville[2][-1] - self.greville[2][-2]))
                pts_3, wts_3 = bsp.quadrature_grid(grid_3, self.pts_loc[2], self.wts_loc[2])%self.breakpoints[2][-1]
                
        else:
            
            pts_3, wts_3 = bsp.quadrature_grid(self.greville[2], self.pts_loc[2], self.wts_loc[2])
            
            
            
        rhs_1 = np.empty((n1 - 1 + self.bc[0], n2, n3))
        rhs_2 = np.empty((n1, n2 - 1 + self.bc[1], n3))
        rhs_3 = np.empty((n1, n2, n3 - 1 + self.bc[2]))
        
        
        for j in range(n2):
            for k in range(n3):
                
                integrand = lambda q1 : fun[0](q1, self.greville[1][j], self.greville[2][k])
                
                rhs_1[:, j, k] = integrate_1d(pts_1, wts_1, integrand)
        
        
        for i in range(n1):
            for k in range(n3):
                
                integrand = lambda q2 : fun[1](self.greville[0][i], q2, self.greville[2][k])
                
                rhs_2[i, :, k] = integrate_1d(pts_2, wts_2, integrand)
                
        for i in range(n1):
            for j in range(n2):
                
                integrand = lambda q3 : fun[2](self.greville[0][i], self.greville[1][j], q3)
                
                rhs_3[i, j, :] = integrate_1d(pts_3, wts_3, integrand)
                
              
        vec1_1 = self.interhistopolation_V1_1.solve(np.reshape(rhs_1, (n1 - 1 + self.bc[0])*n2*n3))
        vec1_2 = self.interhistopolation_V1_2.solve(np.reshape(rhs_2, n1*(n2 - 1 + self.bc[1])*n3))
        vec1_3 = self.interhistopolation_V1_3.solve(np.reshape(rhs_3, n1*n2*(n3 - 1 + self.bc[2])))
        
        return [vec1_1, vec1_2, vec1_3]
    
    
    def PI_2(self, fun):
        
        n1 = self.greville[0].size
        n2 = self.greville[1].size
        n3 = self.greville[2].size
        
        
        # Quadrature grid in 1-direction
        if self.bc[0] == True:
            
            if self.p[0]%2 != 0:
                
                grid_1 = self.breakpoints[0]
                pts_1, wts_1 = bsp.quadrature_grid(grid_1, self.pts_loc[0], self.wts_loc[0])
                
            else:
                
                grid_1 = np.append(self.greville[0], self.greville[0][-1] + (self.greville[0][-1] - self.greville[0][-2]))
                pts_1, wts_1 = bsp.quadrature_grid(grid_1, self.pts_loc[0], self.wts_loc[0])%self.breakpoints[0][-1]
                
        else:
            
            pts_1, wts_1 = bsp.quadrature_grid(self.greville[0], self.pts_loc[0], self.wts_loc[0])
            
            
        # Quadrature grid in 2-direction
        if self.bc[1] == True:
            
            if self.p[1]%2 != 0:
                
                grid_2 = self.breakpoints[1]
                pts_2, wts_2 = bsp.quadrature_grid(grid_2, self.pts_loc[1], self.wts_loc[1])
                
            else:
                
                grid_2 = np.append(self.greville[1], self.greville[1][-1] + (self.greville[1][-1] - self.greville[1][-2]))
                pts_2, wts_2 = bsp.quadrature_grid(grid_2, self.pts_loc[1], self.wts_loc[1])%self.breakpoints[1][-1]
                
        else:
            
            pts_2, wts_2 = bsp.quadrature_grid(self.greville[1], self.pts_loc[1], self.wts_loc[1])
            
            
        # Quadrature grid in 3-direction
        if self.bc[2] == True:
            
            if self.p[2]%2 != 0:
                
                grid_3 = self.breakpoints[2]
                pts_3, wts_3 = bsp.quadrature_grid(grid_3, self.pts_loc[2], self.wts_loc[2])
                
            else:
                
                grid_3 = np.append(self.greville[2], self.greville[2][-1] + (self.greville[2][-1] - self.greville[2][-2]))
                pts_3, wts_3 = bsp.quadrature_grid(grid_3, self.pts_loc[2], self.wts_loc[2])%self.breakpoints[2][-1]
                
        else:
            
            pts_3, wts_3 = bsp.quadrature_grid(self.greville[2], self.pts_loc[2], self.wts_loc[2])
            
            
        rhs_1 = np.empty((n1, n2 - 1 + self.bc[1], n3 - 1 + self.bc[2]))
        rhs_2 = np.empty((n1 - 1 + self.bc[0], n2, n3 - 1 + self.bc[2]))
        rhs_3 = np.empty((n1 - 1 + self.bc[0], n2 - 1 + self.bc[1], n3))
        
        
        for i in range(n1):
            
            integrand = lambda q2, q3 : fun[0](self.greville[0][i], q2, q3)
            
            rhs_1[i, :, :] = integrate_2d([pts_2, pts_3], [wts_2, wts_3], integrand)
            
        for j in range(n2):
            
            integrand = lambda q1, q3 : fun[1](q1, self.greville[1][j], q3)
            
            rhs_2[:, j, :] = integrate_2d([pts_1, pts_3], [wts_1, wts_3], integrand)
            
        for k in range(n3):
            
            integrand = lambda q1, q2 : fun[2](q1, q2, self.greville[2][k])
            
            rhs_3[:, :, k] = integrate_2d([pts_1, pts_2], [wts_1, wts_2], integrand)
            
        
        vec2_1 = self.interhistopolation_V2_1.solve(np.reshape(rhs_1, n1*(n2 - 1 + self.bc[1])*(n3 - 1 + self.bc[2])))
        vec2_2 = self.interhistopolation_V2_2.solve(np.reshape(rhs_2, (n1 - 1 + self.bc[0])*n2*(n3 - 1 + self.bc[2])))
        vec2_3 = self.interhistopolation_V2_3.solve(np.reshape(rhs_3, (n1 - 1 + self.bc[0])*(n2 - 1 + self.bc[1])*n3))
        
        return [vec2_1, vec2_2, vec2_3]
    
            
    def PI_3(self, fun):

        n1 = self.greville[0].size
        n2 = self.greville[1].size
        n3 = self.greville[2].size
        
        # Quadrature grid in 1-direction
        if self.bc[0] == True:
            
            if self.p[0]%2 != 0:
                
                grid_1 = self.breakpoints[0]
                pts_1, wts_1 = bsp.quadrature_grid(grid_1, self.pts_loc[0], self.wts_loc[0])
                
            else:
                
                grid_1 = np.append(self.greville[0], self.greville[0][-1] + (self.greville[0][-1] - self.greville[0][-2]))
                pts_1, wts_1 = bsp.quadrature_grid(grid_1, self.pts_loc[0], self.wts_loc[0])%self.breakpoints[0][-1]
                
        else:
            
            pts_1, wts_1 = bsp.quadrature_grid(self.greville[0], self.pts_loc[0], self.wts_loc[0])
            
            
        # Quadrature grid in 2-direction
        if self.bc[1] == True:
            
            if self.p[1]%2 != 0:
                
                grid_2 = self.breakpoints[1]
                pts_2, wts_2 = bsp.quadrature_grid(grid_2, self.pts_loc[1], self.wts_loc[1])
                
            else:
                
                grid_2 = np.append(self.greville[1], self.greville[1][-1] + (self.greville[1][-1] - self.greville[1][-2]))
                pts_2, wts_2 = bsp.quadrature_grid(grid_2, self.pts_loc[1], self.wts_loc[1])%self.breakpoints[1][-1]
                
        else:
            
            pts_2, wts_2 = bsp.quadrature_grid(self.greville[1], self.pts_loc[1], self.wts_loc[1])
            
            
        # Quadrature grid in 3-direction
        if self.bc[2] == True:
            
            if self.p[2]%2 != 0:
                
                grid_3 = self.breakpoints[2]
                pts_3, wts_3 = bsp.quadrature_grid(grid_3, self.pts_loc[2], self.wts_loc[2])
                
            else:
                
                grid_3 = np.append(self.greville[2], self.greville[2][-1] + (self.greville[2][-1] - self.greville[2][-2]))
                pts_3, wts_3 = bsp.quadrature_grid(grid_3, self.pts_loc[2], self.wts_loc[2])%self.breakpoints[2][-1]
                
        else:
            
            pts_3, wts_3 = bsp.quadrature_grid(self.greville[2], self.pts_loc[2], self.wts_loc[2])

        rhs = np.empty((n1 - 1 + self.bc[0], n2 - 1 + self.bc[1], n3 - 1 + self.bc[2]))

        rhs[:, :, :] = integrate_3d([pts_1, pts_2, pts_3], [wts_1, wts_2, wts_3], fun)

        vec3 = self.interhistopolation_V3.solve(np.reshape(rhs, (n1 - 1 + self.bc[0])*(n2 - 1 + self.bc[1])*(n3 - 1 + self.bc[2])))

        return vec3
#==============================================================================================================================