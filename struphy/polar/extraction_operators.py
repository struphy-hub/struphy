import numpy as np
import scipy.sparse as spa

import struphy.feec.derivatives.derivatives as der


# ============================= 2D polar splines (C1) ===================================
class PolarExtractionBlocksC1:
    """
    Class for C1 global extraction operators.
    
    Parameters
    ----------
        cx : array-like
            Control points defining the x-component of a 2D B-spline mapping.
            
        cy : array-like
            Control points defining the y-component of a 2D B-spline mapping.
    """
    
    def __init__(self, cx, cy):
        
        self._cx = cx
        self._cy = cy
        
        self._pole = (cx[0, 0], cy[0, 0])
        
        assert np.all(cx[0] == self.pole[0])
        assert np.all(cy[0] == self.pole[1])
        
        self._n0 = cx.shape[0]
        self._n1 = cx.shape[1]
        
        self._d0 = self.n0 - 1
        self._d1 = self.n1 - 0
        
        self._n_rings = [(2,), (1, 2), (2, 1), (1,)]
        self._n_polar = [(3,), (0, 2), (2, 0), (0,)]
        
        # number of polar basis functions in V0 (NN)
        self._nbasis0 = (self.n0 - self.n_rings[0][0])*self.n1 + self.n_polar[0][0]
        
        # number of polar basis functions in V1_curl (DN ND) (1st and 2nd component)
        self._nbasis1c_1 = (self.d0 - self.n_rings[1][0])*self.n1 + self.n_polar[1][0]
        self._nbasis1c_2 = (self.n0 - self.n_rings[1][1])*self.d1 + self.n_polar[1][1]
        
        self._nbasis1c = self.nbasis1c_1 + self.nbasis1c_2
        
        # number of polar basis functions in V1_div (ND DN) (1st and 2nd component)
        self._nbasis1d_1 = (self.n0 - self.n_rings[2][0])*self.d1 + self.n_polar[2][0]
        self._nbasis1d_2 = (self.d0 - self.n_rings[2][1])*self.n1 + self.n_polar[2][1]
        
        self._nbasis1d = self.nbasis1d_1 + self.nbasis1d_2
        
        # number of polar basis functions in V2 (DD)
        self._nbasis2 = (self.d0 - self.n_rings[3][0])*self.d1 + self.n_polar[3][0]
        
        # size of control triangle
        self._tau  = max([((self.cx[1] - self.pole[0])*(-2)).max(), 
                          ((self.cx[1] - self.pole[0]) - np.sqrt(3)*(self.cy[1] - self.pole[1])).max(), 
                          ((self.cx[1] - self.pole[0]) + np.sqrt(3)*(self.cy[1] - self.pole[1])).max()])

        
        # barycentric coordinates
        self._xi_0 = np.zeros((3, self.n1), dtype=float)
        self._xi_1 = np.zeros((3, self.n1), dtype=float)

        self._xi_0[:, :] = 1/3

        self._xi_1[0, :] = 1/3 + 2/(3*self.tau)*(self.cx[1] - self.pole[0])
        self._xi_1[1, :] = 1/3 - 1/(3*self.tau)*(self.cx[1] - self.pole[0]) + np.sqrt(3)/(3*self.tau)*(self.cy[1] - self.pole[1])
        self._xi_1[2, :] = 1/3 - 1/(3*self.tau)*(self.cx[1] - self.pole[0]) - np.sqrt(3)/(3*self.tau)*(self.cy[1] - self.pole[1])
        
        # remove small values
        self._xi_1[abs(self._xi_1) < 1e-14] = 0.
        
        # basis extraction operator for discrete 0-forms
        self._e0 = [[np.hstack((self.xi_0, self.xi_1))]]
        
        # basis extraction operator for discrete 1-forms (Hcurl)
        e1_11 = np.zeros((self.n_polar[1][0], 2*self.n1), dtype=float)
        e1_12 = np.zeros((self.n_polar[1][0], 2*self.d1), dtype=float)
        
        e1_21 = np.zeros((self.n_polar[1][1], 2*self.n1), dtype=float)
        e1_22 = np.zeros((self.n_polar[1][1], 2*self.d1), dtype=float)
        
        e1_33 = np.zeros((self.n_polar[0][0], 2*self.n1), dtype=float)
        
        # 1st component
        for l in range(2):
            for j in range(self.n1):
                e1_21[l, j] = self.xi_1[l + 1, j] - self.xi_0[l + 1, j]
        
        # 2nd component
        for l in range(2):
            for j in range(1*self.d1, 2*self.d1):
                e1_22[l, j] = self.xi_1[l + 1, (j - self.d1 + 1)%self.d1] - self.xi_1[l + 1, j - self.d1]
                
        # 3rd component
        e1_33[:, :] = self.e0[0][0]
        
        self._e1 = [[e1_11, e1_12, None ],
                    [e1_21, e1_22, None ],
                    [None , None , e1_33]]
        
        # basis extraction operator for discrete 1-forms (Hdiv)
        e2_33 = np.zeros((self.n_polar[3][0], 2*self.d1), dtype=float)
        
        self._e2 = [[e1_21, -e1_22, None ],
                    [e1_11,  e1_12, None ],
                    [None ,  None , e2_33]]
        
        # basis extraction operator for discrete 2-forms
        self._e3 = [[e2_33.copy()]]
        
        # projection extraction operator for discrete 0-forms
        p0 = np.zeros((self.n_polar[0][0], 2*self.n1), dtype=float)
        
        p0[0, self.n1 + 0*self.n1//3] = 1.
        p0[1, self.n1 + 1*self.n1//3] = 1.
        p0[2, self.n1 + 2*self.n1//3] = 1.
        
        self._p0 = [[p0]]
        
        
    @property
    def cx(self):
        return self._cx
    
    @property
    def cy(self):
        return self._cy
    
    @property
    def pole(self):
        return self._pole
    
    @property
    def n0(self):
        return self._n0
    
    @property
    def n1(self):
        return self._n1
    
    @property
    def d0(self):
        return self._d0
    
    @property
    def d1(self):
        return self._d1
    
    @property
    def n_rings(self):
        return self._n_rings
    
    @property
    def n_polar(self):
        return self._n_polar
    
    @property
    def nbasis0(self):
        return self._nbasis0
    
    @property
    def nbasis1c_1(self):
        return self._nbasis1c_1
    
    @property
    def nbasis1c_2(self):
        return self._nbasis1c_2
    
    @property
    def nbasis1c(self):
        return self._nbasis1c
    
    @property
    def nbasis1d_1(self):
        return self._nbasis1d_1
    
    @property
    def nbasis1d_2(self):
        return self._nbasis1d_2
    
    @property
    def nbasis1d(self):
        return self._nbasis1d
    
    @property
    def nbasis2(self):
        return self._nbasis2
    
    @property
    def tau(self):
        return self._tau
    
    @property
    def xi_0(self):
        return self._xi_0
    
    @property
    def xi_1(self):
        return self._xi_1
    
    @property
    def e0(self):
        return self._e0 
    
    @property
    def e1(self):
        return self._e1 
    
    @property
    def e2(self):
        return self._e2 
    
    @property
    def e3(self):
        return self._e3
    
    @property
    def p0(self):
        return self._p0
    


# ============================= 2D polar splines (C0) ===================================
class polar_splines_C0_2D:
    
    def __init__(self, n0, n1):
        
        d0 = n0 - 1
        d1 = n1 - 0
        
        # number of polar basis functions in V0   (NN)
        self.Nbase0  = (n0 - 1)*n1 + 1
        
        # number of polar basis functions in V1_C (DN ND) (1st and 2nd component)
        self.Nbase1C_1 = d0*n1
        self.Nbase1C_2 = (n0 - 1)*d1
        
        self.Nbase1C = self.Nbase1C_1 + self.Nbase1C_2 
        
        # number of polar basis functions in V1_D (ND DN) (1st and 2nd component)
        self.Nbase1D_1 = (n0 - 1)*d1
        self.Nbase1D_2 = d0*n1
        
        self.Nbase1D = self.Nbase1D_1 + self.Nbase1D_2
        
        # number of polar basis functions in V2   (DD)
        self.Nbase2  = d0*d1
        
        # =========== extraction operators for discrete 0-forms ==================
        # extraction operator for basis functions
        self.E0_11 = spa.csr_matrix(np.ones((1, n1), dtype=float))
        self.E0_22 = spa.identity((n0 - 1)*n1, format='csr')
        
        self.E0 = spa.bmat([[self.E0_11, None], [None, self.E0_22]], format='csr')
        
        # global projection extraction operator for interpolation points
        self.P0_11 = np.zeros((1, n1), dtype=float)
        
        self.P0_11[0, 0] = 1.
        
        self.P0_11 = spa.csr_matrix(self.P0_11)
        
        self.P0_22 = spa.identity((n0 - 1)*n1, format='csr')
        
        self.P0 = spa.bmat([[self.P0_11, None], [None, self.P0_22]], format='csr')
        # =======================================================================
        
        
        # =========== extraction operators for discrete 1-forms (H_curl) ========
        self.E1C_1 = spa.identity(d0*n1, format='csr')
        self.E1C_2 = spa.identity(n0*d1, format='lil')[d1:, :].tocsr()
        
        # combined first and second component
        self.E1C = spa.bmat([[self.E1C_1, None], [None, self.E1C_2]], format='csr')

        # extraction operator for interpolation/histopolation in global projector
        self.P1C_1 = spa.identity(d0*n1, format='csr')
        self.P1C_2 = spa.identity(n0*d1, format='lil')[d1:, :].tocsr()
        
        # combined first and second component
        self.P1C = spa.bmat([[self.P1C_1, None], [None, self.P1C_2]], format='csr')
        # ========================================================================
        
        
        # =========== extraction operators for discrete 1-forms (H_div) ==========
        self.E1D_1 = spa.identity(n0*d1, format='lil')[d1:, :].tocsr()
        self.E1D_2 = spa.identity(d0*n1, format='csr')
        
        # combined first and second component
        self.E1D = spa.bmat([[self.E1D_1, None], [None, self.E1D_2]], format='csr')

        # extraction operator for interpolation/histopolation in global projector
        self.P1D_1 = spa.identity(n0*d1, format='lil')[d1:, :].tocsr()
        self.P1D_2 = spa.identity(d0*n1, format='csr')
        
        # combined first and second component
        self.P1D = spa.bmat([[self.P1D_1, None], [None, self.P1D_2]], format='csr')
        # ========================================================================
        
        
        # =========== extraction operators for discrete 2-forms ==================
        self.E2 = spa.identity(d0*d1, format='csr')
        self.P2 = spa.identity(d0*d1, format='csr')
        # ========================================================================
        
        
        # ========================= 1D discrete derivatives ======================
        grad_1d_1 = spa.csc_matrix(der.grad_1d_matrix(False, n0))
        grad_1d_2 = spa.csc_matrix(der.grad_1d_matrix(True , n1))
        # ========================================================================
        
        
        # ========= discrete polar gradient matrix ===============================
        # radial dofs (DN)
        G11 = np.zeros(((d0 - 0)*n1, 1), dtype=float)
        G11[:n1, 0] = -1.
        
        G12 = spa.kron(grad_1d_1[:, 1:], spa.identity(n1))
        
        self.G1 = spa.bmat([[G11, G12]], format='csr')
        
        # angular dofs (ND)
        G21 = np.zeros(((n0 - 1)*d1, 1), dtype=float)
        G22 = spa.kron(spa.identity(n0 - 1), grad_1d_2, format='csr')
        
        self.G2 = spa.bmat([[G21, G22]], format='csr')
        
        # combined 1st and 2nd component
        self.G = spa.bmat([[self.G1], [self.G2]], format='csr')
        # ========================================================================
        
        
        # ========= discrete polar curl matrix ===================================
        # 2D vector curl (NN --> ND DN)
        
        # angular dofs (ND)
        VC11 = np.zeros(((n0 - 1)*d1, 1), dtype=float)
        VC12 = spa.kron(spa.identity(n0 - 1), grad_1d_2, format='csr')
        
        self.VC1 = spa.bmat([[VC11, VC12]], format='csr')
        
        # radial dofs (DN)
        VC21 = np.zeros(((d0 - 0)*n1, 1), dtype=float)
        VC21[:n1, 0] = 1.
        
        VC22 = -spa.kron(grad_1d_1[:, 1:], spa.identity(n1))
        
        self.VC2 = spa.bmat([[VC21, VC22]], format='csr')

        # combined 1st and 2nd component
        self.VC = spa.bmat([[self.VC1], [self.VC2]], format='csr')
        
        # 2D scalar curl (DN ND --> DD)
        self.SC1 = -spa.kron(spa.identity(d0), grad_1d_2, format='csr')
        self.SC2 =  spa.kron(grad_1d_1[:, 1:], spa.identity(d1), format='csr')
        
        # combined 1st and 2nd component
        self.SC = spa.bmat([[self.SC1, self.SC2]], format='csr')
        # ========================================================================
        
        
        # =============== discrete polar div matrix ==============================
        self.D1 = spa.kron(grad_1d_1[:, 1:], spa.identity(d1), format='csr')
        self.D2 = spa.kron(spa.identity(d0), grad_1d_2, format='csr')
        
        # combined 1st and 2nd component
        self.D = spa.bmat([[self.D1, self.D2]], format='csr')
        # ========================================================================
        
        



# ============================= 2D polar splines (C1) ===================================
class polar_splines_C1_2D:
    
    def __init__(self, cx, cy):
        
        n0, n1 = cx.shape
        
        d0 = n0 - 1
        d1 = n1 - 0
        
        # location of pole
        self.x0 = cx[0, 0]
        self.y0 = cy[0, 0]

        # number of polar basis functions in V0   (NN)
        self.Nbase0  = (n0 - 2)*n1 + 3
        
        # number of polar basis functions in V1_C (DN ND) (1st and 2nd component)
        self.Nbase1C_1 = (d0 - 1)*n1
        self.Nbase1C_2 = (n0 - 2)*d1 + 2
        
        self.Nbase1C = self.Nbase1C_1 + self.Nbase1C_2
        
        # number of polar basis functions in V1_D (ND DN) (1st and 2nd component)
        self.Nbase1D_1 = (n0 - 2)*d1 + 2
        self.Nbase1D_2 = (d0 - 1)*n1
        
        self.Nbase1D = self.Nbase1D_1 + self.Nbase1D_2
        
        # number of polar basis functions in V2   (DD)
        self.Nbase2  = (d0 - 1)*d1
        
        # size of control triangle
        self.tau  = np.array([(-2*(cx[1] - self.x0)).max(), ((cx[1] - self.x0) - np.sqrt(3)*(cy[1] - self.y0)).max(), ((cx[1] - self.x0) + np.sqrt(3)*(cy[1] - self.y0)).max()]).max()

        self.Xi_0 = np.zeros((3, n1), dtype=float)
        self.Xi_1 = np.zeros((3, n1), dtype=float)

        # barycentric coordinates
        self.Xi_0[:, :] = 1/3

        self.Xi_1[0, :] = 1/3 + 2/(3*self.tau)*(cx[1] - self.x0)
        self.Xi_1[1, :] = 1/3 - 1/(3*self.tau)*(cx[1] - self.x0) + np.sqrt(3)/(3*self.tau)*(cy[1] - self.y0)
        self.Xi_1[2, :] = 1/3 - 1/(3*self.tau)*(cx[1] - self.x0) - np.sqrt(3)/(3*self.tau)*(cy[1] - self.y0)
        
        # remove small values
        self.Xi_1[abs(self.Xi_1) < 1e-15] = 0.
        
        
        # =========== extraction operators for discrete 0-forms ==================
        # extraction operator for basis functions
        self.E0_11 = spa.csr_matrix(np.hstack((self.Xi_0, self.Xi_1)))
        self.E0_22 = spa.identity((n0 - 2)*n1, format='csr')
        
        self.E0 = spa.bmat([[self.E0_11, None], [None, self.E0_22]], format='csr')
        
        # global projection extraction operator for interpolation points
        self.P0_11 = np.zeros((3, 2*n1), dtype=float)
        
        self.P0_11[0, n1 + 0*n1//3] = 1.
        self.P0_11[1, n1 + 1*n1//3] = 1.
        self.P0_11[2, n1 + 2*n1//3] = 1.
        
        self.P0_11 = spa.csr_matrix(self.P0_11)
        
        self.P0_22 = spa.identity((n0 - 2)*n1)
        
        self.P0 = spa.bmat([[self.P0_11, None], [None, self.P0_22]], format='csr')
        # =======================================================================
        
        
        
        # =========== extraction operators for discrete 1-forms (H_curl) ========
        self.E1C_12 = spa.identity((d0 - 1)*n1)
        self.E1C_34 = spa.identity((n0 - 2)*d1)
        
        self.E1C_21 = np.zeros((2, 1*n1), dtype=float)
        self.E1C_23 = np.zeros((2, 2*d1), dtype=float)
        
        # 1st component
        for s in range(2):
            for j in range(n1):
                self.E1C_21[s, j] = self.Xi_1[s + 1, j] - self.Xi_0[s + 1, j]
        
        # 2nd component
        for s in range(2):
            for j in range(d1):
                self.E1C_23[s,      j] = 0.
                self.E1C_23[s, n1 + j] = self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j]
                
        # combined first and second component
        self.E1C = spa.bmat([[None, self.E1C_12, None, None], [self.E1C_21, None, self.E1C_23, None], [None, None, None, self.E1C_34]], format='csr')
        
        
        # extraction operator for interpolation/histopolation in global projector

        # 1st component
        self.P1C_11 = np.zeros((n1, n1), dtype=float)
        self.P1C_12 = spa.identity(n1)
        self.P1C_23 = spa.identity((d0 - 2)*n1)
        
        self.P1C_11[:, 0*n1//3]  = -self.Xi_1[0]
        self.P1C_11[:, 1*n1//3]  = -self.Xi_1[1]
        self.P1C_11[:, 2*n1//3]  = -self.Xi_1[2]
        self.P1C_11             += np.identity(n1)
        
        # 2nd component
        self.P1C_34 = np.zeros((2, 2*d1), dtype=float)
        self.P1C_45 = spa.identity((n0 - 2)*d1)
        
        self.P1C_34[0, (d1 + 0*d1//3):(d1 + 1*d1//3)] = np.ones(d1//3, dtype=float)
        self.P1C_34[1, (d1 + 0*d1//3):(d1 + 1*d1//3)] = np.ones(d1//3, dtype=float)
        self.P1C_34[1, (d1 + 1*d1//3):(d1 + 2*d1//3)] = np.ones(d1//3, dtype=float)
        
        # combined first and second component
        self.P1C = spa.bmat([[self.P1C_11, self.P1C_12, None       , None       , None       ], 
                             [None       , None       , self.P1C_23, None       , None       ], 
                             [None       , None       , None       , self.P1C_34, None       ], 
                             [None       , None       , None       , None       , self.P1C_45]], format='csr')
        # =========================================================================
        
        
        
        # ========= extraction operators for discrete 1-forms (H_div) =============
        self.E1D_11 = np.zeros((2, 2*d1), dtype=float)
        self.E1D_13 = np.zeros((2, 1*n1), dtype=float)
        
        self.E1D_22 = spa.identity((n0 - 2)*d1)
        self.E1D_34 = spa.identity((d0 - 1)*n1)
        
        # 1st component
        for s in range(2):
            for j in range(d1):
                self.E1D_11[s,      j] = 0.
                self.E1D_11[s, n1 + j] = (self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j])
                
        # 2nd component
        for s in range(2):
            for j in range(n1):
                self.E1D_13[s, j] = -(self.Xi_1[s + 1, j] - self.Xi_0[s + 1, j])
                
        # combined first and second component
        self.E1D = spa.bmat([[self.E1D_11, None, self.E1D_13, None], [None, self.E1D_22, None, None], [None, None, None, self.E1C_34]], format='csr')
        
        
        # extraction operator for interpolation/histopolation in global projector
        self.P1D_11 = self.P1C_34.copy()
        self.P1D_22 = self.P1C_45.copy()
        
        self.P1D_33 = self.P1C_11.copy()
        self.P1D_34 = self.P1C_12.copy()
        self.P1D_45 = self.P1C_23.copy()
        
        # combined first and second component
        self.P1D = spa.bmat([[self.P1D_11, None       , None        , None       , None       ], 
                             [None       , self.P1D_22, None        , None       , None       ], 
                             [None       , None       , self.P1D_33 , self.P1D_34, None       ], 
                             [None       , None       , None        , None       , self.P1D_45]], format='csr')
        # =========================================================================
        
        
        
        # =========== extraction operators for discrete 2-forms ===================
        self.E2_1 = np.zeros(((d0 - 1)*d1, d1), dtype=float)
        self.E2_2 = spa.identity((d0 - 1)*d1)
        
        self.E2 = spa.bmat([[self.E2_1, self.E2_2]], format='csr')
        
        # extraction operator for histopolation in global projector
        self.P2_11 = np.zeros((d1, d1), dtype=float)
        self.P2_12 = spa.identity(d1)
        self.P2_23 = spa.identity((d0 - 2)*d1)
        
        for i in range(d1):
            
            # block A
            self.P2_11[i, 0*n1//3:1*n1//3] = -(self.Xi_1[1, (i + 1)%n1] - self.Xi_1[1, i]) - (self.Xi_1[2, (i + 1)%n1] - self.Xi_1[2, i])
            
            # block B
            self.P2_11[i, 1*n1//3:2*n1//3] = -(self.Xi_1[2, (i + 1)%n1] - self.Xi_1[2, i])
            
        self.P2_11 += np.identity(d1)
        
        self.P2 = spa.bmat([[self.P2_11, self.P2_12, None], [None, None, self.P2_23]], format='csr')
        # =========================================================================
        
        
        # ========================= 1D discrete derivatives =======================
        grad_1d_1 = spa.csc_matrix(der.grad_1d_matrix(False, n0))
        grad_1d_2 = spa.csc_matrix(der.grad_1d_matrix(True , n1))
        # =========================================================================
        
        
        # ========= discrete polar gradient matrix ================================
        self.G1_1 = np.zeros(((d0 - 1)*n1, 3), dtype=float)
        self.G1_1[:n1, :] = -self.Xi_1.T
        
        self.G1_2 = spa.kron(grad_1d_1[1:, 2:], spa.identity(n1))
        
        self.G1 = spa.bmat([[self.G1_1, self.G1_2]], format='csr')
        
        self.G2_11 = np.zeros((2, 3), dtype=float)
        
        self.G2_11[0, 0] = -1.
        self.G2_11[0, 1] =  1.
        
        self.G2_11[1, 0] = -1.
        self.G2_11[1, 2] =  1.
        
        self.G2_22 = spa.kron(spa.identity(n0 - 2), grad_1d_2)
        
        self.G2 = spa.bmat([[self.G2_11, None], [None, self.G2_22]], format='csr')
        
        self.G = spa.bmat([[self.G1], [self.G2]], format='csr')
        # =======================================================================
        
        
        
        # ========= discrete polar curl matrix ===================================
        # 2D vector curl
        self.VC1_11 = np.zeros((2, 3), dtype=float)
        
        self.VC1_11[0, 0] = -1.
        self.VC1_11[0, 1] =  1.
        
        self.VC1_11[1, 0] = -1.
        self.VC1_11[1, 2] =  1.
        
        self.VC1_22 = spa.kron(spa.identity(n0 - 2), grad_1d_2)
        
        self.VC1 = spa.bmat([[self.VC1_11, None], [None, self.VC1_22]], format='csr')
        
        
        self.VC2_11 = np.zeros(((d0 - 1)*n1, 3), dtype=float)
        self.VC2_11[:n1, :] = -self.Xi_1.T
        
        self.VC2_22 = spa.kron(grad_1d_1[1:, 2:], spa.identity(n1))
        
        self.VC2 = -spa.bmat([[self.VC2_11, self.VC2_22]], format='csr')
        
        self.VC = spa.bmat([[self.VC1], [self.VC2]], format='csr')
        
        # 2D scalar curl
        self.SC1 = -spa.kron(spa.identity(d0 - 1), grad_1d_2)
        
        self.SC2_1 = np.zeros(((d0 - 1)*d1, 2), dtype=float)
        
        for s in range(2):
            for j in range(d1):
                self.SC2_1[j, s] = -(self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j])
                
        self.SC2_2 = spa.kron(grad_1d_1[1:, 2:], spa.identity(d1))
        
        self.SC2 = spa.bmat([[self.SC2_1, self.SC2_2]], format='csr')
        
        self.SC = spa.bmat([[self.SC1, self.SC2]], format='csr')
        # =========================================================================
        
        
        
        # ========= discrete polar div matrix =====================================
        self.D1_1 = np.zeros(((d0 - 1)*d1, 2), dtype=float)
        
        for s in range(2):
            for j in range(d1):
                self.D1_1[j, s]  = -(self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j])
                
        self.D1_2 = spa.kron(grad_1d_1[1:, 2:], spa.identity(d1))
        
        self.D1 = spa.bmat([[self.D1_1, self.D1_2]], format='csr')
        
        self.D2 = spa.kron(spa.identity(d0 - 1), grad_1d_2)
        
        self.D = spa.bmat([[self.D1, self.D2]], format='csr')
        # =========================================================================
        
        


# ============================= 3D polar splines ===================================
class polar_splines:
    
    def __init__(self, tensor_space, cx, cy):
        
        n0, n1, n2 = tensor_space.NbaseN
        d0, d1, d2 = tensor_space.NbaseD

        # number of polar basis functions in V0 (NN)
        self.Nbase0_pol = (n0 - 2)*n1 + 3
        
        # number of polar basis functions in V1 (DN ND) (1st and 2nd component)
        self.Nbase1_pol = (d0 - 1)*n1 + (n0 - 2)*d1 + 2
        
        # number of polar basis functions in V2 (ND DN) (1st and 2nd component)
        self.Nbase2_pol = (d0 - 1)*n1 + (n0 - 2)*d1 + 2
        
        # number of polar basis functions in V3 (DD)
        self.Nbase3_pol = (d0 - 1)*d1
        
        # size of control triangle
        self.tau  = np.array([(-2*cx[1]).max(), (cx[1] - np.sqrt(3)*cy[1]).max(), (cx[1] + np.sqrt(3)*cy[1]).max()]).max()

        self.Xi_0 = np.zeros((3, n1), dtype=float)
        self.Xi_1 = np.zeros((3, n1), dtype=float)

        # barycentric coordinates
        self.Xi_0[:, :] = 1/3

        self.Xi_1[0, :] = 1/3 + 2/(3*self.tau)*cx[1, :, 0]
        self.Xi_1[1, :] = 1/3 - 1/(3*self.tau)*cx[1, :, 0] + np.sqrt(3)/(3*self.tau)*cy[1, :, 0]
        self.Xi_1[2, :] = 1/3 - 1/(3*self.tau)*cx[1, :, 0] - np.sqrt(3)/(3*self.tau)*cy[1, :, 0]

        
        # =========== extraction operators for discrete 0-forms ==================
        # extraction operator for basis functions
        self.E0_pol = spa.bmat([[np.hstack((self.Xi_0, self.Xi_1)), None], [None, spa.identity((n0 - 2)*n1)]], format='csr')
        self.E0     = spa.kron(self.E0_pol, spa.identity(n2), format='csr')
        
        # global projection extraction operator for interpolation points
        self.P0_pol                   = spa.lil_matrix((self.Nbase0_pol, n0*n1), dtype=float)
        self.P0_pol[0 , n1 + 0*n1//3] = 1.
        self.P0_pol[1 , n1 + 1*n1//3] = 1.
        self.P0_pol[2 , n1 + 2*n1//3] = 1.
        self.P0_pol[3:, 2*n1:]        = spa.identity((n0 - 2)*n1)
        self.P0_pol                   = self.P0_pol.tocsr()
        self.P0                       = spa.kron(self.P0_pol, spa.identity(n2), format='csr')
        # =======================================================================
        
        
        
        # =========== extraction operators for discrete 1-forms =================
        self.E1_1_pol = spa.lil_matrix((self.Nbase1_pol, d0*n1), dtype=float)
        self.E1_2_pol = spa.lil_matrix((self.Nbase1_pol, n0*d1), dtype=float)

        # 1st component
        for s in range(2):
            for j in range(n1):
                self.E1_1_pol[(d0 - 1)*n1 + s, j] = self.Xi_1[s + 1, j] - self.Xi_0[s + 1, j]
                
        self.E1_1_pol[:(d0 - 1)*n1, n1:] = np.identity((d0 - 1)*n1)
        self.E1_1_pol = self.E1_1_pol.tocsr()

        # 2nd component
        for s in range(2):
            for j in range(n1):
                self.E1_2_pol[(d0 - 1)*n1 + s,      j] = 0.
                self.E1_2_pol[(d0 - 1)*n1 + s, n1 + j] = self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j]

        self.E1_2_pol[((d0 - 1)*n1 + 2):, 2*d1:] = np.identity((n0 - 2)*d1)
        self.E1_2_pol = self.E1_2_pol.tocsr()
        
        # 3rd component
        self.E1_3_pol = self.E0_pol
        
        # combined first and second component
        self.E1_pol = spa.bmat([[self.E1_1_pol, self.E1_2_pol]], format='csr')

        # expansion in third dimension
        self.E1 = spa.bmat([[spa.kron(self.E1_pol, spa.identity(n2)), None], [None, spa.kron(self.E1_3_pol, spa.identity(d2))]], format='csr')
        
        # extraction operator for interpolation/histopolation in global projector
        self.P1_1_pol = spa.lil_matrix(((d0 - 1)*n1    , d0*n1), dtype=float)
        self.P1_2_pol = spa.lil_matrix(((n0 - 2)*d1 + 2, n0*d1), dtype=float)
        
        # 1st component
        self.P1_1_pol[:n1, 0*n1//3]  = -self.Xi_1[0].reshape(n1, 1)
        self.P1_1_pol[:n1, 1*n1//3]  = -self.Xi_1[1].reshape(n1, 1)
        self.P1_1_pol[:n1, 2*n1//3]  = -self.Xi_1[2].reshape(n1, 1)
        self.P1_1_pol[:n1,   :1*n1] += spa.identity(n1)
        self.P1_1_pol[:n1, n1:2*n1]  = spa.identity(n1)
        self.P1_1_pol[n1:,   2*n1:]  = spa.identity((d0 - 2)*n1)
        self.P1_1_pol                = self.P1_1_pol.tocsr()
        
        # 2nd component
        self.P1_2_pol[0, (n1 + 0*n1//3):(n1 + 1*n1//3)] = np.ones((1, n1//3), dtype=float)
        self.P1_2_pol[1, (n1 + 0*n1//3):(n1 + 1*n1//3)] = np.ones((1, n1//3), dtype=float)
        self.P1_2_pol[1, (n1 + 1*n1//3):(n1 + 2*n1//3)] = np.ones((1, n1//3), dtype=float)
        self.P1_2_pol[2:, 2*n1:]                        = spa.identity((n0 - 2)*d1)
        self.P1_2_pol                                   = self.P1_2_pol.tocsr()
        
        # 3rd component
        self.P1_3_pol = self.P0_pol
        
        # combined first and second component
        self.P1_pol = spa.bmat([[self.P1_1_pol, None], [None, self.P1_2_pol]], format='csr')
        
        # expansion in third dimension
        self.P1 = spa.bmat([[spa.kron(self.P1_pol, spa.identity(n2)), None], [None, spa.kron(self.P1_3_pol, spa.identity(d2))]], format='csr')
        # =========================================================================
        
        
        
        # =========== extraction operators for discrete 2-forms ===================
        self.E2_1_pol = spa.lil_matrix((self.Nbase2_pol, n0*d1), dtype=float)
        self.E2_2_pol = spa.lil_matrix((self.Nbase2_pol, d0*n1), dtype=float)
        self.E2_3_pol = spa.lil_matrix((self.Nbase3_pol, d0*d1), dtype=float)

        # 1st component
        for s in range(2):
            for j in range(n1):
                self.E2_1_pol[s,      j] = 0.
                self.E2_1_pol[s, n1 + j] = (self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j])
                
        self.E2_1_pol[2:(2 + (n0 - 2)*d1), 2*n1:] = np.identity((n0 - 2)*d1)
        self.E2_1_pol = self.E2_1_pol.tocsr()

        # 2nd component
        for s in range(2):
            for j in range(n1):
                self.E2_2_pol[s, j] = -(self.Xi_1[s + 1, j] - self.Xi_0[s + 1, j])
                
        self.E2_2_pol[(2 + (n0 - 2)*d1):, 1*n1:] = np.identity((d0 - 1)*n1)
        self.E2_2_pol = self.E2_2_pol.tocsr()
        
        # 3rd component
        self.E2_3_pol[:, 1*d1:] = np.identity((d0 - 1)*d1)
        self.E2_3_pol = self.E2_3_pol.tocsr()

        # combined first and second component
        self.E2_pol = spa.bmat([[self.E2_1_pol, self.E2_2_pol]], format='csr')
   
        # expansion in third dimension
        self.E2 = spa.bmat([[spa.kron(self.E2_pol, spa.identity(d2)), None], [None, spa.kron(self.E2_3_pol, spa.identity(n2))]], format='csr')
        
        # extraction operator for interpolation/histopolation in global projector
        
        # 1st component
        self.P2_1_pol = self.P1_2_pol
        
        # 2nd component
        self.P2_2_pol = self.P1_1_pol
        
        # 3rd component
        self.P2_3_pol = spa.lil_matrix(((d0 - 1)*d1, d0*d1), dtype=float)
        
        for i2 in range(d1):
            
            # block A
            self.P2_3_pol[i2, 0*n1//3:1*n1//3] = -(self.Xi_1[1, (i2 + 1)%n1] - self.Xi_1[1, i2]) - (self.Xi_1[2, (i2 + 1)%n1] - self.Xi_1[2, i2])
            
            # block B
            self.P2_3_pol[i2, 1*n1//3:2*n1//3] = -(self.Xi_1[2, (i2 + 1)%n1] - self.Xi_1[2, i2])
            
        self.P2_3_pol[:d1,   :1*d1] += spa.identity(d1)
        self.P2_3_pol[:d1, d1:2*d1]  = spa.identity(d1)
        
        self.P2_3_pol[d1:, 2*d1:]    = spa.identity((d0 - 2)*d1)
        self.P2_3_pol                = self.P2_3_pol.tocsr()
        
        # combined first and second component
        self.P2_pol = spa.bmat([[self.P2_1_pol, None], [None, self.P2_2_pol]], format='csr')
        
        # expansion in third dimension
        self.P2 = spa.bmat([[spa.kron(self.P2_pol, spa.identity(d2)), None], [None, spa.kron(self.P2_3_pol, spa.identity(n2))]], format='csr')
        # =========================================================================
        
        
        # =========== extraction operators for discrete 3-forms ===================
        self.E3_pol = self.E2_3_pol
        
        self.E3     = spa.kron(self.E3_pol, spa.identity(d2), format='csr')
        
        self.P3_pol = self.P2_3_pol
        self.P3     = spa.kron(self.P3_pol, spa.identity(d2), format='csr')
        # =========================================================================
        
        
        # ========================= 1D discrete derivatives =======================
        grad_1d_1 = spa.csc_matrix(der.grad_1d_matrix(tensor_space.spaces[0]))
        grad_1d_2 = spa.csc_matrix(der.grad_1d_matrix(tensor_space.spaces[1]))
        grad_1d_3 = spa.csc_matrix(der.grad_1d_matrix(tensor_space.spaces[2]))
        # =========================================================================
        
        
        
        # ========= discrete polar gradient matrix ================================
        grad_1 = spa.lil_matrix(((d0 - 1)*n1    , self.Nbase0_pol), dtype=float)
        grad_2 = spa.lil_matrix(((n0 - 2)*d1 + 2, self.Nbase0_pol), dtype=float)

        # radial dofs (D N)
        grad_1[:  , 3:] = spa.kron(grad_1d_1[1:, 2:], spa.identity(n1))
        grad_1[:n1, :3] = -self.Xi_1.T

        # angular dofs (N D)
        grad_2[0, 0] = -1.
        grad_2[0, 1] =  1.

        grad_2[1, 0] = -1.
        grad_2[1, 2] =  1.
        
        grad_2[2:, 3:] = spa.kron(spa.identity(n0 - 2), grad_1d_2)
        
        # combined 1st and 2nd component
        self.grad_pol = spa.bmat([[grad_1], [grad_2]], format='csr')

        # expansion in 3rd dimension
        self.GRAD = spa.bmat([[spa.kron(self.grad_pol, spa.identity(n2))], [spa.kron(spa.identity(self.Nbase0_pol), grad_1d_3)]], format='csr')
        # =======================================================================
        
        
        
        # ========= discrete polar curl matrix ===================================
        # 2D vector curl
        vector_curl_1 = spa.lil_matrix(((n0 - 2)*d1 + 2, self.Nbase0_pol), dtype=float)
        vector_curl_2 = spa.lil_matrix(((d0 - 1)*n1    , self.Nbase0_pol), dtype=float)
        
        # angular dofs (N D)
        vector_curl_1[0, 0] = -1.
        vector_curl_1[0, 1] =  1.

        vector_curl_1[1, 0] = -1.
        vector_curl_1[1, 2] =  1.
        
        vector_curl_1[2:, 3:] = spa.kron(spa.identity(n0 - 2), grad_1d_2)
        
        # radial dofs (D N)
        vector_curl_2[:  , 3:] = -spa.kron(grad_1d_1[1:, 2:], spa.identity(n1))
        vector_curl_2[:n1, :3] =  self.Xi_1.T
        
        # combined 1st and 2nd component
        self.vector_curl_pol = spa.bmat([[vector_curl_1], [vector_curl_2]], format='csr')
        
        # 2D scalar curl
        self.scalar_curl_pol = spa.lil_matrix((self.Nbase3_pol, self.Nbase2_pol), dtype=float)
        
        # radial dofs (D N)
        self.scalar_curl_pol[:, :(d0 - 1)*n1] = -spa.kron(spa.identity(d0 - 1), grad_1d_2)
        
        # angular dofs (N D)
        for s in range(2):
            for j in range(n1):
                self.scalar_curl_pol[j, (d0 - 1)*n1 + s] = -(self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j])
                
        self.scalar_curl_pol[:, ((d0 - 1)*n1 + 2):] = spa.kron(grad_1d_1[1:, 2:], spa.identity(d1))
        self.scalar_curl_pol = self.scalar_curl_pol.tocsr()
        
        # derivatives along 3rd dimension
        DZ = spa.bmat([[None, -spa.kron(spa.identity((n0 - 2) * d1 + 2), grad_1d_3)],
                       [spa.kron(spa.identity((d0 - 1)*n1), grad_1d_3), None       ]])
        
        # total polar curl
        self.CURL = spa.bmat([[DZ, spa.kron(self.vector_curl_pol, spa.identity(d2))  ], 
                              [spa.kron(self.scalar_curl_pol, spa.identity(n2)), None]], format='csr')
        # =========================================================================
        
        
        
        # ========= discrete polar div matrix =====================================
        
        self.div_pol = spa.lil_matrix((self.Nbase3_pol, self.Nbase2_pol), dtype=float)
        
        # angular dofs (N D)
        for s in range(2):
            for j in range(d1):
                self.div_pol[j, s]  = -(self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j])
                
        self.div_pol[:, 2:((d0 - 1)*n1 + 2)] = spa.kron(grad_1d_1[1:, 2:], spa.identity(d1))
        
        # radial dofs (D N)
        self.div_pol[:, ((d0 - 1)*n1 + 2):] = spa.kron(spa.identity(d0 - 1), grad_1d_2)
        self.div_pol = self.div_pol.tocsr()
        
        # expansion along 3rd dimension
        self.DIV = spa.bmat([[spa.kron(self.div_pol, spa.identity(d2)), spa.kron(spa.identity(self.Nbase3_pol), grad_1d_3)]], format='csr')
        
        # =========================================================================