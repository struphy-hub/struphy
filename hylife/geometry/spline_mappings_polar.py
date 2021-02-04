import numpy as np
import scipy.sparse as sparse

import hylife.utilitis_FEEC.bsplines    as bsp
import hylife.utilitis_FEEC.derivatives as der
import hylife.utilitis_FEEC.Bspline     as bspline

from scipy.sparse.linalg import splu




class discrete_mapping_2d:
    '''
    Defines a discrete spline mapping from a logical domain [0, 1] x [0, 1] to a physical cartesian domain
    '''
    
    def __init__(self, tensor_space, kind):
        
        self.p        = tensor_space.p
        self.T        = tensor_space.T
        self.bc       = tensor_space.bc
        
        self.el_b     = [bsp.breakpoints(T, p) for T, p in zip(self.T, self.p)]
        self.Nel      = [len(el_b) - 1 for el_b in self.el_b]
        
        self.NbaseN   = [Nel + p - bc*p for Nel, p, bc in zip(self.Nel, self.p, self.bc)]
        self.NbaseD   = [NbaseN - (1 - bc) for NbaseN, bc in zip(self.NbaseN, self.bc)]
        
        self.grad_1d  = [sparse.csr_matrix(der.grad_1d(space)) for space in tensor_space.spaces]
        self.greville = [bsp.greville(T, p, bc) for T, p, bc in zip(self.T, self.p, self.bc)]
        
        N1 = sparse.csc_matrix(bsp.collocation_matrix(self.T[0], self.p[0], self.greville[0], self.bc[0])) 
        N2 = sparse.csc_matrix(bsp.collocation_matrix(self.T[1], self.p[1], self.greville[1], self.bc[1]))  
        
        self.interpolation = splu(sparse.kron(N1, N2, format='csc'))
        
        
        if   kind == 'standard polar':
            
            x0 = 0.
            y0 = 0.
            
            self.x = lambda r, theta : r*np.cos(2*np.pi*theta) + x0
            self.y = lambda r, theta : r*np.sin(2*np.pi*theta) + y0
            
        elif kind == 'square root polar':
            
            x0 = 0.
            y0 = 0.
            
            self.x = lambda r, theta : np.sqrt(r)*np.cos(2*np.pi*theta) + x0
            self.y = lambda r, theta : np.sqrt(r)*np.sin(2*np.pi*theta) + y0
            
        elif kind == 'tokamak 1':
            
            x0    = 0.
            y0    = 0.
            kappa = 0.3
            delta = 0.2
            
            self.x = lambda s, phi : x0 + (1 - kappa)*s*np.cos(2*np.pi*phi) - delta*s**2
            self.y = lambda s, phi : y0 + (1 + kappa)*s*np.sin(2*np.pi*phi)
            
        elif kind == 'tokamak 2':
            
            y0  = 0. 
            eps = 0.3
            e   = 1.4
            xi  = 1/np.sqrt(1 - eps**2/4)
            x0  = (1 - np.sqrt(1 + eps**2))/eps
            
            self.x = lambda s, phi : 1/eps*(1 - np.sqrt(1 + eps*(eps + 2*s*np.cos(2*np.pi*phi)))) - x0
            self.y = lambda s, phi : y0 + e*xi*s*np.sin(2*np.pi*phi)/(1 + eps*self.x(s, phi))
            
        elif kind == 'slab':
            
            Lx = 2.
            Ly = 2.
            
            self.x = lambda q1, q2 : Lx*q1
            self.y = lambda q1, q2 : Ly*q2
            
        elif kind == 'colella':
            
            Lx    = 2.
            Ly    = 2.
            alpha = 0.1
            
            self.x = lambda q1, q2 : Lx*(q1 + alpha*np.sin(2*np.pi*q1)*np.sin(2*np.pi*q2))
            self.y = lambda q1, q2 : Ly*(q2 + alpha*np.sin(2*np.pi*q1)*np.sin(2*np.pi*q2))
            
        else:
            print('mapping not implemented!')
            
        
        g1, g2 = np.meshgrid(self.greville[0], self.greville[1], indexing='ij')

        cx = self.interpolation.solve(self.x(g1, g2).flatten()).reshape(self.NbaseN[0], self.NbaseN[1])
        cy = self.interpolation.solve(self.y(g1, g2).flatten()).reshape(self.NbaseN[0], self.NbaseN[1])
        
        
        self.c = [cx, cy]
    
    
    
    def mapping(self, q):
        '''
        Evaluates the spline mapping at the points q[0] x q[1] (tensor product)
        '''
           
        N1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0], q[0], self.bc[0]))
        N2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1], self.p[1], q[1], self.bc[1]))
        
        x = sparse.kron(N1, N2).dot(self.c[0].flatten()).reshape(len(q[0]), len(q[1]))
        y = sparse.kron(N1, N2).dot(self.c[1].flatten()).reshape(len(q[0]), len(q[1]))
        
        return x, y
     
    
    
    def jacobian(self, q, component):
        '''
        Evaluates the jacobian DF = dF_i/dF_j of the spline mapping at the points q[0] x q[1]
        '''
        
        grad1 = sparse.kron(self.grad_1d[0], np.identity(self.NbaseN[1]))
        grad2 = sparse.kron(np.identity(self.NbaseN[0]), self.grad_1d[1])
        
        N1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0], q[0], self.bc[0]))
        N2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1], self.p[1], q[1], self.bc[1]))
        
        D1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0][1:-1], self.p[0] - 1, q[0], self.bc[0], normalize=True))
        D2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1][1:-1], self.p[1] - 1, q[1], self.bc[1], normalize=True))
        
        
        DF_00 = sparse.kron(D1, N2).dot(grad1.dot(self.c[0].flatten())).reshape(len(q[0]), len(q[1]))
        DF_01 = sparse.kron(N1, D2).dot(grad2.dot(self.c[0].flatten())).reshape(len(q[0]), len(q[1]))
        
        DF_10 = sparse.kron(D1, N2).dot(grad1.dot(self.c[1].flatten())).reshape(len(q[0]), len(q[1]))
        DF_11 = sparse.kron(N1, D2).dot(grad2.dot(self.c[1].flatten())).reshape(len(q[0]), len(q[1]))
        
        if   component == '00':
            return DF_00
        
        elif component == '01':
            return DF_01
        
        elif component == '10':
            return DF_10
        
        elif component == '11':
            return DF_11
        
        
    def jacobian_inverse(self, q, component):
        '''
        Evaluates the inverse jacobian DF^(-1) of the spline mapping at the points q[0] x q[1]
        '''
        
        grad1 = sparse.kron(self.grad_1d[0], np.identity(self.NbaseN[1]))
        grad2 = sparse.kron(np.identity(self.NbaseN[0]), self.grad_1d[1])
        
        N1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0], q[0], self.bc[0]))
        N2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1], self.p[1], q[1], self.bc[1]))
        
        D1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0][1:-1], self.p[0] - 1, q[0], self.bc[0], normalize=True))
        D2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1][1:-1], self.p[1] - 1, q[1], self.bc[1], normalize=True))
        
        
        DF_00 = sparse.kron(D1, N2).dot(grad1.dot(self.c[0].flatten())).reshape(len(q[0]), len(q[1]))
        DF_01 = sparse.kron(N1, D2).dot(grad2.dot(self.c[0].flatten())).reshape(len(q[0]), len(q[1]))
        
        DF_10 = sparse.kron(D1, N2).dot(grad1.dot(self.c[1].flatten())).reshape(len(q[0]), len(q[1]))
        DF_11 = sparse.kron(N1, D2).dot(grad2.dot(self.c[1].flatten())).reshape(len(q[0]), len(q[1]))
        
        det_DF = DF_00*DF_11 - DF_01*DF_10
        
        if   component == '00':
            return  DF_11/det_DF
        
        elif component == '01':
            return -DF_01/det_DF
        
        elif component == '10':
            return -DF_10/det_DF
        
        elif component == '11':
            return  DF_00/det_DF
        
        
    def jacobian_determinant(self, q):
        '''
        Evaluates the square root of the determinant of the metric tensor at the points q[0] x q[1]
        '''
        
        grad1 = sparse.kron(self.grad_1d[0], np.identity(self.NbaseN[1]))
        grad2 = sparse.kron(np.identity(self.NbaseN[0]), self.grad_1d[1])
        
        N1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0], q[0], self.bc[0]))
        N2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1], self.p[1], q[1], self.bc[1]))
        
        D1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0][1:-1], self.p[0] - 1, q[0], self.bc[0], normalize=True))
        D2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1][1:-1], self.p[1] - 1, q[1], self.bc[1], normalize=True))
        
        
        DF_00 = sparse.kron(D1, N2).dot(grad1.dot(self.c[0].flatten())).reshape(len(q[0]), len(q[1]))
        DF_01 = sparse.kron(N1, D2).dot(grad2.dot(self.c[0].flatten())).reshape(len(q[0]), len(q[1]))
        
        DF_10 = sparse.kron(D1, N2).dot(grad1.dot(self.c[1].flatten())).reshape(len(q[0]), len(q[1]))
        DF_11 = sparse.kron(N1, D2).dot(grad2.dot(self.c[1].flatten())).reshape(len(q[0]), len(q[1]))
        
        return np.abs(DF_00*DF_11 - DF_01*DF_10)
    
    
    
    def metric_tensor(self, q, component):
        '''
        Evaluates the metric tensor G = DF^T * DF of the spline mapping at the points q[0] x q[1]
        '''
        
        grad1 = sparse.kron(self.grad_1d[0], np.identity(self.NbaseN[1]))
        grad2 = sparse.kron(np.identity(self.NbaseN[0]), self.grad_1d[1])
        
        N1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0], q[0], self.bc[0]))
        N2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1], self.p[1], q[1], self.bc[1]))
        
        D1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0][1:-1], self.p[0] - 1, q[0], self.bc[0], normalize=True))
        D2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1][1:-1], self.p[1] - 1, q[1], self.bc[1], normalize=True))
        
        
        DF_00 = sparse.kron(D1, N2).dot(grad1.dot(self.c[0].flatten())).reshape(len(q[0]), len(q[1]))
        DF_01 = sparse.kron(N1, D2).dot(grad2.dot(self.c[0].flatten())).reshape(len(q[0]), len(q[1]))
        
        DF_10 = sparse.kron(D1, N2).dot(grad1.dot(self.c[1].flatten())).reshape(len(q[0]), len(q[1]))
        DF_11 = sparse.kron(N1, D2).dot(grad2.dot(self.c[1].flatten())).reshape(len(q[0]), len(q[1]))
        
        if   component == '00':
            return DF_00**2 + DF_10**2
        
        elif component == '01':
            return DF_00*DF_01 + DF_10*DF_11
        
        elif component == '10':
            return DF_00*DF_01 + DF_10*DF_11
        
        elif component == '11':
            return DF_01**2 + DF_11**2
    
    
    
    def metric_tensor_inverse(self, q, component):
        '''
        Evaluates the inverse metric tensor G^(-1) = DF^(-1) * DF^(-T) of the spline mapping at the points q[0] x q[1]
        '''
        
        grad1 = sparse.kron(self.grad_1d[0], np.identity(self.NbaseN[1]))
        grad2 = sparse.kron(np.identity(self.NbaseN[0]), self.grad_1d[1])
        
        N1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0], q[0], self.bc[0]))
        N2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1], self.p[1], q[1], self.bc[1]))
        
        D1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0][1:-1], self.p[0] - 1, q[0], self.bc[0], normalize=True))
        D2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1][1:-1], self.p[1] - 1, q[1], self.bc[1], normalize=True))
        
        
        DF_00 = sparse.kron(D1, N2).dot(grad1.dot(self.c[0].flatten())).reshape(len(q[0]), len(q[1]))
        DF_01 = sparse.kron(N1, D2).dot(grad2.dot(self.c[0].flatten())).reshape(len(q[0]), len(q[1]))
        
        DF_10 = sparse.kron(D1, N2).dot(grad1.dot(self.c[1].flatten())).reshape(len(q[0]), len(q[1]))
        DF_11 = sparse.kron(N1, D2).dot(grad2.dot(self.c[1].flatten())).reshape(len(q[0]), len(q[1]))
        
        det_DF = DF_00*DF_11 - DF_01*DF_10
        
        if   component == '00':
            return  (DF_11**2 + DF_01**2)/det_DF**2
        
        elif component == '01':
            return -(DF_11*DF_10 + DF_01*DF_00)/det_DF**2
        
        elif component == '10':
            return -(DF_10*DF_11 + DF_00*DF_01)/det_DF**2
        
        elif component == '11':
            return  (DF_10**2 + DF_00**2)/det_DF**2
            
        
    def barycentric_coordinates(self, i):
        '''
        Returns the barycentric coordinates with respect to a triangle that encloses the i-th ring of control points
        '''
        
        tau0 = (-2*self.c[0][i]).max()
        tau1 = (self.c[0][i] - np.sqrt(3)*self.c[1][i]).max()
        tau2 = (self.c[0][i] + np.sqrt(3)*self.c[1][i]).max()
        
        tau  = np.array([tau0, tau1, tau2]).max()
        
        lambda0 = lambda x, y : 1/3 + 2/(3*tau)*x
        lambda1 = lambda x, y : 1/3 - 1/(3*tau)*x + 1/(np.sqrt(3)*tau)*y
        lambda2 = lambda x, y : 1/3 - 1/(3*tau)*x - 1/(np.sqrt(3)*tau)*y
        
        return tau, [lambda0, lambda1, lambda2]
    
    
    
    def C1_coefficients(self):
        '''
        Return the coeffcients e0, e1 for the three C1 continuous polar spline basis functions
        '''
        
        # get tau (enclose first ring of control points)
        tau  = self.barycentric_coordinates(1)[0]
        
        # 3 Bernstein basis functions 
        indices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        Tl      = np.empty(3, dtype=float)
        Tl_dx   = np.empty(3, dtype=float)
        Tl_dy   = np.empty(3, dtype=float)
        
        for l in range(3):
            Tl   [l] = 1/3
            Tl_dx[l] = (2*indices[l, 0] - indices[l, 1] - indices[l, 2])/(3*tau)
            Tl_dy[l] = (indices[l, 1] - indices[l, 2])/(np.sqrt(3)*tau)
            
        # compute coefficients e0, e1
        e0 = np.empty((3, self.NbaseN[1]))
        e1 = np.empty((3, self.NbaseN[1]))
        
        for l in range(3):
            e0[l] = Tl[l]
            e1[l] = e0[l] + self.c[0][1]*Tl_dx[l] + self.c[1][1]*Tl_dy[l]
            
        # compute extraction operator
        Ebar = np.hstack((e0, e1))
        Ntot = self.NbaseN[0]*self.NbaseN[1] - 2*self.NbaseN[1] + 3 
        I    = sparse.identity(Ntot - 3)
        E    = sparse.bmat([[Ebar, None], [None, I]], format='csc')
            
        return E, tau
            
            
    
    def C2_coefficients(self):
        '''
        Returns the coeffcients e0, e1, e2 for the six C2 continuous polar spline basis functions
        '''
        
        from math import factorial
        
        # knot vectors, degrees and number of basis functions in polar region (first 3 rings of control points)
        T_pole      = [self.T[0], np.repeat(self.T[1], 4)[9:-9]]
        p_pole      = [self.p[0], 2*self.p[1]]
        NbaseN_pole = [self.NbaseN[0], len(T_pole[1]) - p_pole[1] - 1 - self.p[1]]
        
        
        # needed mass-like matrices for transformations of control points
        B3 = bspline.Bspline(self.T[1], self.p[1])
        B6 = bspline.Bspline(T_pole[1], p_pole[1])
        
        M66_pole  = np.zeros((NbaseN_pole[1], NbaseN_pole[1])                 , dtype=float)   
        M63_pole  = np.zeros((NbaseN_pole[1], self.NbaseN[1])                 , dtype=float)
        M633_pole = np.zeros((NbaseN_pole[1], self.NbaseN[1] , self.NbaseN[1]), dtype=float)
        
        pts_loc, wts_loc = np.polynomial.legendre.leggauss(p_pole[1] + 1)
        pts,     wts     = bsp.quadrature_grid(self.el_b[1], pts_loc, wts_loc)
        
        #-------------------------------------------------------------------------------------
        for ie in range(self.Nel[1]):
    
            for il in range(p_pole[1]+ 1):
                for jl in range(p_pole[1] + 1):

                    i = 4*ie + il
                    j = 4*ie + jl

                    value = 0.
                    for q in range(p_pole[1] + 1):
                        value += wts[ie, q] * B6(pts[ie, q], i) * B6(pts[ie, q], j)
                        
                    M66_pole[i%NbaseN_pole[1], j%NbaseN_pole[1]] += value
        #-------------------------------------------------------------------------------------
        for ie in range(self.Nel[1]):
    
            for il in range(p_pole[1] + 1):
                for jl in range(self.p[1] + 1):

                    i = 4*ie + il
                    j = 1*ie + jl

                    value = 0.
                    for q in range(p_pole[1] + 1):
                        value += wts[ie, q] * B6(pts[ie, q], i) * B3(pts[ie, q], j)

                    M63_pole[i%NbaseN_pole[1], j%self.NbaseN[1]] += value
        #-------------------------------------------------------------------------------------
        for ie in range(self.Nel[1]):
    
            for il in range(p_pole[1] + 1):
                for jl in range(self.p[1] + 1):
                    for kl in range(self.p[1] + 1):

                        i = 4*ie + il
                        j = 1*ie + jl
                        k = 1*ie + kl

                        value = 0.
                        for q in range(p_pole[1] + 1):
                            value += wts[ie, q] * B6(pts[ie, q], i) * B3(pts[ie, q], j) * B3(pts[ie, q], k)

                        M633_pole[i%NbaseN_pole[1], j%self.NbaseN[1], k%self.NbaseN[1]] += value
        #-------------------------------------------------------------------------------------
        
        
        # transform control points from degree 3 space to degree 6 space
        cx_hat_1 = np.linalg.solve(M66_pole, M63_pole.dot(self.c[0][1]))
        cy_hat_1 = np.linalg.solve(M66_pole, M63_pole.dot(self.c[1][1]))

        cx_hat_2 = np.linalg.solve(M66_pole, M63_pole.dot(self.c[0][2]))
        cy_hat_2 = np.linalg.solve(M66_pole, M63_pole.dot(self.c[1][2]))
        
        # transfrom control points from degree (3 x 3) product space to degree 6 space
        rhs_xx_11 = np.zeros(NbaseN_pole[1], dtype=float)
        rhs_xy_11 = np.zeros(NbaseN_pole[1], dtype=float)
        rhs_yy_11 = np.zeros(NbaseN_pole[1], dtype=float)

        for i in range(NbaseN_pole[1]):
            for j in range(self.NbaseN[1]):
                for k in range(self.NbaseN[1]):
                    rhs_xx_11[i] += M633_pole[i, j, k] * self.c[0][1, j] * self.c[0][1, k]
                    rhs_xy_11[i] += M633_pole[i, j, k] * self.c[0][1, j] * self.c[1][1, k]
                    rhs_yy_11[i] += M633_pole[i, j, k] * self.c[1][1, j] * self.c[1][1, k]

        cxx_hat_11 = np.linalg.solve(M66_pole, rhs_xx_11)
        cxy_hat_11 = np.linalg.solve(M66_pole, rhs_xy_11)
        cyy_hat_11 = np.linalg.solve(M66_pole, rhs_yy_11)
        
        # first and second derivatives of first three splines in radial direction at r=0
        ders = bsp.basis_funs_all_ders(self.T[0], self.p[0], 0., self.p[0], 2)

        d0,   d1,  d2 = ders[1, 0:3]
        dd0, dd1, dd2 = ders[2, 0:3]
        
        # starting tau for algorithm (enclose second ring of control points)
        ring = 2
        tau  = self.barycentric_coordinates(ring)[0]
        
        # 6 Bernstein basis functions 
        indices = np.array([[2, 0, 0], [1, 1, 0], [0, 2, 0], [1, 0, 1], [0, 1, 1], [0, 0, 2]])

        Tl      = np.empty(6, dtype=float)
        Tl_dx   = np.empty(6, dtype=float)
        Tl_dy   = np.empty(6, dtype=float)
        Tl_dxdx = np.empty(6, dtype=float)
        Tl_dydy = np.empty(6, dtype=float)
        Tl_dxdy = np.empty(6, dtype=float)
        
        while True:
            
            e0 = np.empty((6, NbaseN_pole[1]))
            e1 = np.empty((6, NbaseN_pole[1]))
            e2 = np.empty((6, NbaseN_pole[1]))
            
            for l in range(6):
                prefactor  = 2/(factorial(indices[l, 0])*factorial(indices[l, 1])*factorial(indices[l, 2]))

                Tl     [l] = prefactor/9
                Tl_dx  [l] = prefactor*(2*indices[l, 0] - indices[l, 1] - indices[l, 2])/(9*tau)
                Tl_dy  [l] = prefactor*(indices[l, 1] - indices[l, 2])/(3*np.sqrt(3)*tau)
                Tl_dxdx[l] = prefactor*(2 + 3*indices[l, 0]*(3*indices[l, 0] - 5))/(9*tau**2)
                Tl_dydy[l] = prefactor*((indices[l, 2] - indices[l, 1])**2 - (indices[l, 1] + indices[l, 2]))/(3*tau**2)
                Tl_dxdy[l] = prefactor*((1 - 3*indices[l, 0])*(indices[l, 2] - indices[l, 1]))/(3*np.sqrt(3)*tau**2)
            
            
            # compute coefficients e0, e1, e2
            for l in range(6):
                e0[l] = Tl[l]
                e1[l] = e0[l] + cx_hat_1*Tl_dx[l] + cy_hat_1*Tl_dy[l]
                e2[l] = (dd1 + dd2)*e0[l] - dd1*e1[l] + d1**2*cxx_hat_11*Tl_dxdx[l] + 2*d1**2*cxy_hat_11*Tl_dxdy[l] + d1**2*cyy_hat_11*Tl_dydy[l] + (dd1*cx_hat_1 + dd2*cx_hat_2)*Tl_dx[l] + (dd1*cy_hat_1 + dd2*cy_hat_2)*Tl_dy[l]

            
            e2 = e2/dd2
            
            # check if all coefficients are > 1 (if not increase tau such that the control triangle encloses one more ring)
            if np.all(e0 > 0) and np.all(e1 > 0) and np.all(e2 > 0):
                break
                
            else:
                ring += 1
                tau   = self.barycentric_coordinates(ring)[0]
                
        
        # compute extraction operator 
        Ebar   = np.hstack((e0, e1, e2))
        Ntot   = NbaseN_pole[0]*NbaseN_pole[1] - 3*NbaseN_pole[1] + 6
        I      = sparse.identity(Ntot - 6)
        E      = sparse.bmat([[Ebar, None], [None, I]], format='csc')
        
        return E, tau