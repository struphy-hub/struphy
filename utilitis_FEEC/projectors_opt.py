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
def collocation_matrix_1d_sparse(p, Nbase, T, greville, bc):
    
    n_greville = len(greville)
    
    pts = np.reshape(greville, (n_greville, 1))
        
    basis = bsp.basis_ders_on_quad_grid(T, p, pts, 0, normalize=False)
    
    D = np.zeros((n_greville, p + 1))
    D[:, :] = basis[:, :, 0, 0]
            
    indices = np.indices((n_greville, p + 1))
    
    dat = D.flatten()
    row = indices[0].flatten()
    
    
    if bc == True:
        col = ((indices[1] + np.arange(n_greville)[:, None])%n_greville).flatten()
        
    else:
        
        if p%2 != 0:
            
            ind = list(np.arange(n_greville - 5) + 1)
            col = (indices[1] + np.array([0] * 2 + ind + [ind[-1] + 1] * 3)[:, None]).flatten() 
            
        else:
            
            boundaries = int(p/2) + 1
            
            ind = list(np.arange(n_greville - 2*boundaries) + 1)
            col = (indices[1] + np.array([0] * boundaries + ind + [ind[-1] + 1] * boundaries)[:, None]).flatten()
            
            
    
    D = sparse.csr_matrix((dat, (row, col)), shape=(n_greville, n_greville))
    
    return D
#==============================================================================================================================
    
    

#==============================================================================================================================
def histopolation_matrix_1d_sparse(p, Nbase, T, greville, bc):
    
    el_b = bsp.breakpoints(T, p)
    Nel = len(el_b) - 1
    t = T[1:-1]
    
    if bc == True:
        
        if p%2 != 0:
            
            pts_loc, wts_loc = np.polynomial.legendre.leggauss(p - 1)
            dx = 1./Nel
        
            ne = Nel
            
            grid = el_b
            
            pts, wts = bsp.quadrature_grid(grid, pts_loc, wts_loc)
            basis = bsp.basis_ders_on_quad_grid(t, p - 1, pts, 0, normalize=True)
            
            D = np.zeros((ne, p))
            
            for ie in range(Nel):
                for il in range(p):
                    for k in range(p - 1):
                        D[ie, il] += wts[ie, k] * basis[ie, il, 0, k]
            
            indices = np.indices((ne, p))
            
            dat = D.flatten()
            row = indices[0].flatten()
            col = ((indices[1] + np.arange(ne)[:, None])%ne).flatten()
            
            D = sparse.csr_matrix((dat, (row, col)), shape=(ne, ne))

            return D
        
        else:
            
            pts_loc, wts_loc = np.polynomial.legendre.leggauss(p - 1)
            dx = 1./Nel
            
            ne = Nel
            
            grid = np.append(greville, greville + dx/2)
            grid = np.append(grid, grid[-1] + dx/2)
            grid.sort()
            
            pts, wts = bsp.quadrature_grid(grid, pts_loc, wts_loc)
            basis = bsp.basis_ders_on_quad_grid(t, p - 1, pts%1., 0, normalize=True)
            
            D   = np.zeros((ne, p + 1))
            ies = np.floor_divide(np.arange(2*Nel), 2)
            
            for iee in range(2*Nel):
                for il in range(p):
                    for k in range(p - 1):
                        D[ies[iee], il + iee%2] += wts[iee, k] * basis[iee, il, 0, k]
                        
            indices = np.indices((ne, p + 1))
            
            dat = D.flatten()
            row = indices[0].flatten()
            col = ((indices[1] + np.arange(ne)[:, None])%ne).flatten()
            
            D = sparse.csr_matrix((dat, (row, col)), shape=(ne, ne))

            return D
        
    else:
        
        if p%2 != 0:
        
            pts_loc, wts_loc = np.polynomial.legendre.leggauss(p - 1)
            
            ng = len(greville)
            
            grid = greville

            pts, wts = bsp.quadrature_grid(grid, pts_loc, wts_loc)
            basis = bsp.basis_ders_on_quad_grid(t, p - 1, pts, 0, normalize=True)

            D = np.zeros((ng - 1, p))

            for ie in range(ng - 1):
                for il in range(p):
                    for k in range(p - 1):
                        D[ie, il] += wts[ie, k] * basis[ie, il, 0, k]

            indices = np.indices((ng - 1, p)) 

            ind = list(np.arange(ng - 1 - 2*(p - 1)) + 1)

            dat = D.flatten()
            row = indices[0].flatten()
            col = (indices[1] + np.array([0] * 2 + ind + [ind[-1] + 1] * 2)[:, None]).flatten()

            D = sparse.csr_matrix((dat, (row, col)), shape=(ng - 1, ng - 1))  
            
            
        else:
            
            pts_loc, wts_loc = np.polynomial.legendre.leggauss(p - 1)
            
            ng = len(greville)
            ne = 2*Nel + p - 2
            
            grid = np.union1d(greville, el_b)
            
            pts, wts = bsp.quadrature_grid(grid, pts_loc, wts_loc)
            basis = bsp.basis_ders_on_quad_grid(t, p - 1, pts, 0, normalize=True)
            
            D = np.zeros((ng - 1, p + 1))
            
            boundaries = int(p/2)
            ies = np.floor_divide(np.arange(ne - p), 2) + boundaries
            ies = np.array(list(np.arange(boundaries)) + list(ies) + list(np.arange(boundaries) + ies[-1] + 1))
            
            il_add = np.array([0] * boundaries + [0, 1] * int((ne - p)/2) + [1] * boundaries)
            
            for iee in range(ne):
                for il in range(p):
                    for k in range(p - 1):
                        D[ies[iee], il + il_add[iee]] += wts[iee, k] * basis[iee, il, 0, k]
                        
            indices = np.indices((ng - 1, p + 1))
            
            dat = D.flatten()
            row = indices[0].flatten()
            
            ind = list(np.arange(ng - 1 - 2*(boundaries + 1)) + 1)
            col = (indices[1] + np.array([0] * (boundaries + 1) + ind + [ind[-1] + 1] * (boundaries + 1))[:, None]).flatten()
            
            D = sparse.csr_matrix((dat, (row, col)), shape=(ng - 1, ng - 1))
                
        return D
#============================================================================================================================== 



#==============================================================================================================================
def histopolation_V1_1(p, Nbase, T, bc):
    
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc
    
    t1 = T1[1:-1]
    t2 = T2[1:-1]
    t3 = T3[1:-1]
    
    grev_1 = bsp.greville(T1, p1, bc_1)
    grev_2 = bsp.greville(T2, p2, bc_2)
    grev_3 = bsp.greville(T3, p3, bc_3)
    
    el_b_1 = bsp.breakpoints(T1, p1)
    el_b_2 = bsp.breakpoints(T2, p2)
    el_b_3 = bsp.breakpoints(T3, p3)
    
    pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 - 1)
    
    
    # Quadrature grid in 1-direction
    if bc_1 == True:

        if p1%2 != 0:

            grid_1 = el_b_1
            pts_1, wts_1 = bsp.quadrature_grid(grid_1, pts_1_loc, wts_1_loc)

        else:

            print('not implemented yet!')

    else:

        pts_1, wts_1 = bsp.quadrature_grid(grev_1, pts_1_loc, wts_1_loc)
        
    
    n1 = grev_1.size - 1 + bc_1
    n2 = grev_2.size
    n3 = grev_3.size
    
    pts_2 = np.reshape(grev_2 + 1e-15, (n2, 1))
    pts_3 = np.reshape(grev_3 + 1e-15, (n3, 1))
        
    basis_1 = bsp.basis_ders_on_quad_grid(t1, p1 - 1, pts_1, 0, normalize=True)
    basis_2 = bsp.basis_ders_on_quad_grid(T2, p2, pts_2, 0, normalize=False)
    basis_3 = bsp.basis_ders_on_quad_grid(T3, p3, pts_3, 0, normalize=False)
    
            
    D = np.zeros((n1, n2, n3, p1, p2 + 1, p3 + 1))
    
    
    for ie1 in range(n1):
        for ie2 in range(n2):
            for ie3 in range(n3):
                
                for il1 in range(p1):
                    for il2 in range(p2 + 1):
                        for il3 in range(p3 + 1):
                            
                            for q1 in range(p1 - 1):
                                D[ie1, ie2, ie3, il1, il2, il3] += wts_1[ie1, q1] * basis_1[ie1, il1, 0, q1] * basis_2[ie2, il2, 0, 0] * basis_3[ie3, il3, 0, 0]
     
    
    
    # Grid indices
    indices = np.indices((n1, n2, n3, p1, p2 + 1, p3 + 1))
    
    
    # Row indices and matrix entries
    row = (n2*n3*indices[0] + n3*indices[1] + indices[2]).flatten()
    dat = D.flatten()
    
    
    # Column indices in 1-direction (histopolation) 
    if bc_1 == True:
        
        col1 = (indices[3] + np.arange(n1)[:, None, None, None, None, None])%n1
        
    else:
    
        ind1 = list(np.arange(n1 - 2*(p1 - 2)))
        col1 = indices[3] + np.array([0] + ind1 + [ind1[-1]])[:, None, None, None, None, None]
        
    
    # Column indices in 2-direction (interpolation)
    if bc_2 == True:
        
        col2 = (indices[4] + np.arange(n2)[None, :, None, None, None, None])%n2
        
    else:
        
        ind2 = list(np.arange(n2 - 1 - 2*(p2 - 2)))
        col2 = indices[4] + np.array([0] + ind2 + [ind2[-1]] * 2)[None, :, None, None, None, None]
    
    
    # Column indices in 3-direction (interpolation)
    if bc_3 == True:
        
        col3 = (indices[5] + np.arange(n3)[None, None, :, None, None, None])%n3
        
    else:
        
        ind3 = list(np.arange(n3 - 1 - 2*(p3 - 2)))
        col3 = indices[5] + np.array([0] + ind3 + [ind3[-1]] * 2)[None, None, :, None, None, None]
        
    col = (n2*n3*col1 + n3*col2 + col3).flatten()
    
    
    # Create sparse marix
    D = sparse.csr_matrix((dat, (row, col)), shape=(n1*n2*n3, n1*n2*n3))
    
    D.eliminate_zeros()
    
                                                    
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
        
        
        self.greville = [bsp.greville(T1, p1, bc_1) + p1%2*1e-15, bsp.greville(T2, p2, bc_2) + p2%2*1e-15, bsp.greville(T3, p3, bc_3) + p3%2*1e-15]
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
        
        self.interhistopolation_V0    = sparse.kron(sparse.kron(N1, N2), N3, format='csc')
        self.interhistopolation_V0_LU = splu(self.interhistopolation_V0)
     
    
    def assemble_V1(self):
        
        N1 = sparse.csc_matrix(bsp.collocation_matrix(self.T[0], self.p[0], self.greville[0], self.bc[0])) 
        N2 = sparse.csc_matrix(bsp.collocation_matrix(self.T[1], self.p[1], self.greville[1], self.bc[1])) 
        N3 = sparse.csc_matrix(bsp.collocation_matrix(self.T[2], self.p[2], self.greville[2], self.bc[2]))
        
        D1 = sparse.csc_matrix(histopolation_matrix_1d(self.p[0], self.Nbase[0], self.T[0], self.greville[0], self.bc[0]))
        D2 = sparse.csc_matrix(histopolation_matrix_1d(self.p[1], self.Nbase[1], self.T[1], self.greville[1], self.bc[1]))
        D3 = sparse.csc_matrix(histopolation_matrix_1d(self.p[2], self.Nbase[2], self.T[2], self.greville[2], self.bc[2]))
        
        self.interhistopolation_V1_1    = sparse.kron(sparse.kron(D1, N2), N3, format='csc')
        self.interhistopolation_V1_2    = sparse.kron(sparse.kron(N1, D2), N3, format='csc')
        self.interhistopolation_V1_3    = sparse.kron(sparse.kron(N1, N2), D3, format='csc')
        
        self.interhistopolation_V1_1_LU = splu(self.interhistopolation_V1_1)
        self.interhistopolation_V1_2_LU = splu(self.interhistopolation_V1_2)
        self.interhistopolation_V1_3_LU = splu(self.interhistopolation_V1_3)
        
    
    def assemble_V2(self):
        
        N1 = sparse.csc_matrix(bsp.collocation_matrix(self.T[0], self.p[0], self.greville[0], self.bc[0])) 
        N2 = sparse.csc_matrix(bsp.collocation_matrix(self.T[1], self.p[1], self.greville[1], self.bc[1])) 
        N3 = sparse.csc_matrix(bsp.collocation_matrix(self.T[2], self.p[2], self.greville[2], self.bc[2]))
        
        D1 = sparse.csc_matrix(histopolation_matrix_1d(self.p[0], self.Nbase[0], self.T[0], self.greville[0], self.bc[0]))
        D2 = sparse.csc_matrix(histopolation_matrix_1d(self.p[1], self.Nbase[1], self.T[1], self.greville[1], self.bc[1]))
        D3 = sparse.csc_matrix(histopolation_matrix_1d(self.p[2], self.Nbase[2], self.T[2], self.greville[2], self.bc[2]))
        
        self.interhistopolation_V2_1 = sparse.kron(sparse.kron(N1, D2), D3, format='csc')
        self.interhistopolation_V2_2 = sparse.kron(sparse.kron(D1, N2), D3, format='csc')
        self.interhistopolation_V2_3 = sparse.kron(sparse.kron(D1, D2), N3, format='csc')
        
        self.interhistopolation_V2_1_LU = splu(self.interhistopolation_V2_1)
        self.interhistopolation_V2_2_LU = splu(self.interhistopolation_V2_2)
        self.interhistopolation_V2_3_LU = splu(self.interhistopolation_V2_3)
        
        
    def assemble_V3(self):
        
        D1 = sparse.csc_matrix(histopolation_matrix_1d(self.p[0], self.Nbase[0], self.T[0], self.greville[0], self.bc[0]))
        D2 = sparse.csc_matrix(histopolation_matrix_1d(self.p[1], self.Nbase[1], self.T[1], self.greville[1], self.bc[1]))
        D3 = sparse.csc_matrix(histopolation_matrix_1d(self.p[2], self.Nbase[2], self.T[2], self.greville[2], self.bc[2]))
        
        self.interhistopolation_V3    = sparse.kron(sparse.kron(D1, D2), D3, format='csc')
        self.interhistopolation_V3_LU = splu(self.interhistopolation_V3)
        
    
    def PI_0(self, fun):
        
        n1 = self.greville[0].size
        n2 = self.greville[1].size
        n3 = self.greville[2].size
        
        rhs = np.empty((n1, n2, n3))
        
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    rhs[i, j, k] = fun(self.greville[0][i], self.greville[1][j], self.greville[2][k])
                    
                    
        vec0 = self.interhistopolation_V0_LU.solve(np.reshape(rhs, n1*n2*n3))
        
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

              
        vec1_1 = self.interhistopolation_V1_1_LU.solve(np.reshape(rhs_1, (n1 - 1 + self.bc[0])*n2*n3))
        vec1_2 = self.interhistopolation_V1_2_LU.solve(np.reshape(rhs_2, n1*(n2 - 1 + self.bc[1])*n3))
        vec1_3 = self.interhistopolation_V1_3_LU.solve(np.reshape(rhs_3, n1*n2*(n3 - 1 + self.bc[2])))
        
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
            
        
        vec2_1 = self.interhistopolation_V2_1_LU.solve(np.reshape(rhs_1, n1*(n2 - 1 + self.bc[1])*(n3 - 1 + self.bc[2])))
        vec2_2 = self.interhistopolation_V2_2_LU.solve(np.reshape(rhs_2, (n1 - 1 + self.bc[0])*n2*(n3 - 1 + self.bc[2])))
        vec2_3 = self.interhistopolation_V2_3_LU.solve(np.reshape(rhs_3, (n1 - 1 + self.bc[0])*(n2 - 1 + self.bc[1])*n3))
        
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

        vec3 = self.interhistopolation_V3_LU.solve(np.reshape(rhs, (n1 - 1 + self.bc[0])*(n2 - 1 + self.bc[1])*(n3 - 1 + self.bc[2])))

        return vec3
#==============================================================================================================================