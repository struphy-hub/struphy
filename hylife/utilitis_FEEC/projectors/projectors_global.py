import numpy              as np
import scipy.sparse       as sparse
import hylife.utilitis_FEEC.bsplines as bsp
import hylife.utilitis_FEEC.projectors.kernels_projectors_global as kernels

from scipy.sparse.linalg import splu



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
    f_loc = np.array([0.])
    
    for ie1 in range(n1):
        
        w1   = weights[ie1, :]
        Pts1 = points[ie1, :]
        
        mat_f[:] = fun(Pts1)
        
        kernels.kernel_int_1d(nq1, w1, mat_f, f_loc)
        
        f_int[ie1] = f_loc
        
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
# ===================================================================



# ===================================================================
def histopolation_matrix_1d(T, p, greville, bc):
    """
    Computest the 1d histopolation matrix of the M-splines at the greville points.
    
    Parameters
    ----------
    
    T : np.array 
        knot vector
    
    p : int
        spline degree
    
    greville : np.array
        greville points
    
    bc : boolean
        boundary conditions (True = periodic, False = else)
      
    Returns
    -------
    D : 2d np.array
        histopolation matrix
    """
    
    el_b = bsp.breakpoints(T, p)
    Nel  = len(el_b) - 1
    t    = T[1:-1]
    
    if bc == True:
        
        pts_loc, wts_loc = np.polynomial.legendre.leggauss(p - 1)
        D = np.zeros((Nel, Nel))
        
        if p%2 != 0:
            
            grid = el_b
            
            pts, wts = bsp.quadrature_grid(grid, pts_loc, wts_loc)
            col_quad = bsp.collocation_matrix(t, p - 1, pts.flatten(), bc, normalize=True)
            
            for ie in range(Nel):
                for il in range(p):
                    
                    i = (ie + il)%Nel
                    
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
                    
                    i = (ie + il)%Nel
                    
                    for k in range(p - 1):
                        D[ie_grev, i] += wts[iee, k]*col_quad[iee*(p - 1) + k, i]

            return D
        
    else:
        
        ng = len(greville)
        
        col_quad = bsp.collocation_matrix(T, p, greville, bc)
        
        Nbase = Nel + p
        D = np.zeros((ng - 1, Nbase - 1))
        
        for i in range(ng - 1):
            for j in range(max(i - p + 1, 1), min(i + p + 3, Nbase)):
                s = 0.
                for k in range(j):
                    s += col_quad[i, k] - col_quad[i + 1, k]
                    
                if np.abs(s) > 1e-15:
                    
                    D[i, j - 1] = s
                
        return D
# ===================================================================




# ===================================================================
class projectors_3d:
    
    def __init__(self, T, p, bc):
        
        self.T         = T
        self.p         = p
        self.bc        = bc
        self.el_b      = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
        self.Nel       = [len(el_b) - 1 for el_b in self.el_b]
        self.NbaseN    = [Nel + p - bc*p for Nel, p, bc in zip(self.Nel, p, bc)]
        self.NbaseD    = [NbaseN - (1 - bc) for NbaseN, bc in zip(self.NbaseN, bc)]
        self.quad_loc  = [np.polynomial.legendre.leggauss(p + 1) for p in p]
        self.greville  = [bsp.greville(T, p, bc) for T, p, bc in zip(T, p, bc)]
        self.delta     = [1/Nel for Nel in self.Nel] 
        
        
        # Quadrature grids
        pts = []
        wts = []
        
        for a in range(3):
            
            if self.bc[a] == True:
                
                if self.p[a]%2 != 0:
                    grid = self.el_b[a]
                    pts_a, wts_a = bsp.quadrature_grid(grid, self.quad_loc[a][0], self.quad_loc[a][1])
                    
                else:
                    grid = self.el_b[a] + self.delta[a]/2
                    pts_a, wts_a = bsp.quadrature_grid(grid, self.quad_loc[a][0], self.quad_loc[a][1])
                    pts_a = pts_a%1.


            else:
                
                grid = self.greville[a]
                pts_a, wts_a = bsp.quadrature_grid(grid, self.quad_loc[a][0], self.quad_loc[a][1])
                
            pts.append(pts_a)
            wts.append(wts_a)
            
        self.pts = pts
        self.wts = wts
        
    
    # ======================================
    def assemble_V0(self):
        
        N = [sparse.csc_matrix(bsp.collocation_matrix(T, p, greville, bc)) for T, p, greville, bc in zip(self.T, self.p, self.greville, self.bc)]
        self.interhistopolation_V0    = sparse.kron(sparse.kron(N[0], N[1]), N[2], format='csc')
        self.interhistopolation_V0_LU = splu(self.interhistopolation_V0)
        
    # ======================================
    def assemble_V1(self):
        
        N = [sparse.csc_matrix(bsp.collocation_matrix(T, p, greville, bc)) for T, p, greville, bc in zip(self.T, self.p, self.greville, self.bc)]
        D = [sparse.csc_matrix(histopolation_matrix_1d(T, p, greville, bc)) for T, p, greville, bc in zip(self.T, self.p, self.greville, self.bc)]
        
        self.interhistopolation_V1    = [sparse.kron(sparse.kron(D[0], N[1]), N[2], format='csc'), sparse.kron(sparse.kron(N[0], D[1]), N[2], format='csc'), sparse.kron(sparse.kron(N[0], N[1]), D[2], format='csc')] 
        
        self.interhistopolation_V1_LU = [splu(interhistopolation_V1) for interhistopolation_V1 in self.interhistopolation_V1]
    
    # ======================================    
    def assemble_V2(self):
        
        N = [sparse.csc_matrix(bsp.collocation_matrix(T, p, greville, bc)) for T, p, greville, bc in zip(self.T, self.p, self.greville, self.bc)]
        D = [sparse.csc_matrix(histopolation_matrix_1d(T, p, greville, bc)) for T, p, greville, bc in zip(self.T, self.p, self.greville, self.bc)]
        
        self.interhistopolation_V2    = [sparse.kron(sparse.kron(N[0], D[1]), D[2], format='csc'), sparse.kron(sparse.kron(D[0], N[1]), D[2], format='csc'), sparse.kron(sparse.kron(D[0], D[1]), N[2], format='csc')] 
        
        self.interhistopolation_V2_LU = [splu(interhistopolation_V2) for interhistopolation_V2 in self.interhistopolation_V2]
    
    # ======================================
    def assemble_V3(self):
        
        D = [sparse.csc_matrix(histopolation_matrix_1d(T, p, greville, bc)) for T, p, greville, bc in zip(self.T, self.p, self.greville, self.bc)]
        
        self.interhistopolation_V3    = sparse.kron(sparse.kron(D[0], D[1]), D[2], format='csc') 
        self.interhistopolation_V3_LU = splu(self.interhistopolation_V3)
    
    
    # ======================================        
    def PI_0(self, fun):
        
        n = [greville.size for greville in self.greville]
        
        rhs = np.empty((n[0], n[1], n[2]))
        
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    rhs[i, j, k] = fun(self.greville[0][i], self.greville[1][j], self.greville[2][k])
                    
                    
        vec0 = self.interhistopolation_V0_LU.solve(rhs.flatten())
        
        return vec0.reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseN[2])
    
    # ======================================
    def PI_1(self, fun):
        
        n = [greville.size for greville in self.greville]
            
        rhs = [np.empty((n[0] - 1 + self.bc[0], n[1], n[2])), np.empty((n[0], n[1] - 1 + self.bc[1], n[2])), np.empty((n[0], n[1], n[2] - 1 + self.bc[2]))]
                    
        for j in range(n[1]):
            for k in range(n[2]):

                integrand = lambda xi1 : fun[0](xi1, self.greville[1][j], self.greville[2][k])

                rhs[0][:, j, k] = integrate_1d(self.pts[0], self.wts[0], integrand)


        for i in range(n[0]):
            for k in range(n[2]):

                integrand = lambda xi2 : fun[1](self.greville[0][i], xi2, self.greville[2][k])

                rhs[1][i, :, k] = integrate_1d(self.pts[1], self.wts[1], integrand)

        
        for i in range(n[0]):
            for j in range(n[1]):

                integrand = lambda xi3 : fun[2](self.greville[0][i], self.greville[1][j], xi3)

                rhs[2][i, j, :] = integrate_1d(self.pts[2], self.wts[2], integrand)
                
        vec1 = [interhistopolation_V1_LU.solve(rhs.flatten()) for interhistopolation_V1_LU, rhs in zip(self.interhistopolation_V1_LU, rhs)]
        
        return [vec1[0].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]), vec1[1].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]), vec1[2].reshape(self.NbaseN[0], self.NbaseN[1], self.NbaseD[2])]
    
    # ======================================
    def PI_2(self, fun):
        
        n = [greville.size for greville in self.greville]
        
        rhs = [np.empty((n[0], n[1] - 1 + self.bc[1], n[2] - 1 + self.bc[2])), np.empty((n[0] - 1 + self.bc[0], n[1], n[2] - 1 + self.bc[2])), np.empty((n[0] - 1 + self.bc[0], n[1] - 1 + self.bc[1], n[2]))]
        
        
        for i in range(n[0]):
            
            integrand = lambda xi2, xi3 : fun[0](self.greville[0][i], xi2, xi3)
            
            rhs[0][i, :, :] = integrate_2d([self.pts[1], self.pts[2]], [self.wts[1], self.wts[2]], integrand)
            
        for j in range(n[1]):
            
            integrand = lambda xi1, xi3 : fun[1](xi1, self.greville[1][j], xi3)
            
            rhs[1][:, j, :] = integrate_2d([self.pts[0], self.pts[2]], [self.wts[0], self.wts[2]], integrand)
            
        for k in range(n[2]):
            
            integrand = lambda xi1, xi2 : fun[2](xi1, xi2, self.greville[2][k])
            
            rhs[2][:, :, k] = integrate_2d([self.pts[0], self.pts[1]], [self.wts[0], self.wts[1]], integrand)
            
        
        vec2 = [interhistopolation_V2_LU.solve(rhs.flatten()) for interhistopolation_V2_LU, rhs in zip(self.interhistopolation_V2_LU, rhs)]
        
        return [vec2[0].reshape(self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]), vec2[1].reshape(self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]), vec2[2].reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseN[2])]
    
    # ======================================
    def PI_3(self, fun):

        n = [greville.size for greville in self.greville]

        rhs = np.empty((n[0] - 1 + self.bc[0], n[1] - 1 + self.bc[1], n[2] - 1 + self.bc[2]))

        rhs[:, :, :] = integrate_3d([self.pts[0], self.pts[1], self.pts[2]], [self.wts[0], self.wts[1], self.wts[2]], fun)

        vec3 = self.interhistopolation_V3_LU.solve(rhs.flatten())

        return vec3.reshape(self.NbaseD[0], self.NbaseD[1], self.NbaseD[2])
# ===================================================================


# ===================================================================
class projectors_1d:
    
    def __init__(self, T, p, bc):
        
        self.T         = T
        self.p         = p
        self.bc        = bc
        self.el_b      = bsp.breakpoints(self.T, self.p)
        self.Nel       = len(self.el_b) - 1
        self.NbaseN    = self.Nel + self.p - self.bc*self.p
        self.NbaseD    = self.NbaseN - (1 - self.bc)
        self.quad_loc  = np.polynomial.legendre.leggauss(self.p + 1)
        self.greville  = bsp.greville(self.T, self.p, self.bc)
        self.delta     = 1/self.Nel
        
        
        # Quadrature grid
        if self.bc == True:

            if self.p%2 != 0:
                grid = self.el_b
                self.pts, self.wts = bsp.quadrature_grid(grid, self.quad_loc[0], self.quad_loc[1])

            else:
                grid = self.el_b + self.delta/2
                self.pts, self.wts = bsp.quadrature_grid(grid, self.quad_loc[0], self.quad_loc[1])
                self.pts = self.pts%1.


        else:

            grid = self.greville
            self.pts, self.wts = bsp.quadrature_grid(grid, self.quad_loc[0], self.quad_loc[1])

        
    
    # ======================================
    def assemble_V0(self):
        
        self.interhistopolation_V0    = sparse.csc_matrix(bsp.collocation_matrix(self.T, self.p, self.greville, self.bc))
        self.interhistopolation_V0_LU = splu(self.interhistopolation_V0)
        
    # ======================================
    def assemble_V1(self):
        
        self.interhistopolation_V1    = sparse.csc_matrix(histopolation_matrix_1d(self.T, self.p, self.greville, self.bc))
        self.interhistopolation_V1_LU = splu(self.interhistopolation_V1)
    
    
    # ======================================        
    def PI_0(self, fun):
        
        n = self.greville.size
        
        rhs = np.empty(n)
        
        for i in range(n):
            rhs[i] = fun(self.greville[i])
                              
        return self.interhistopolation_V0_LU.solve(rhs)
    
    
    # ======================================
    def PI_1(self, fun):

        n = self.greville.size

        rhs = np.empty(n - 1 + self.bc)

        rhs[:] = integrate_1d(self.pts, self.wts, fun)

        return self.interhistopolation_V1_LU.solve(rhs)
# ===================================================================