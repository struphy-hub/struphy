import numpy as np
import scipy.sparse as sparse

from scipy.sparse.linalg import splu

import utilitis_FEEC.bsplines as bsp
import utilitis_FEEC.derivatives as der







class discrete_mapping_2d:
    
    def __init__(self, p, T, b):
        
        self.p = p
        self.T = T
        self.b = b
        
        p1, p2 = p
        T1, T2 = T
        b1, b2 = b
        
        self.greville = (bsp.greville(T1, p1, b1), bsp.greville(T2, p2, b2))
        
        N1 = sparse.csc_matrix(bsp.collocation_matrix(T1, p1, self.greville[0], b1)) 
        N2 = sparse.csc_matrix(bsp.collocation_matrix(T2, p2, self.greville[1], b2))  
        
        self.interpolation = splu(sparse.kron(N1, N2, format='csc'))
        
        self.grad = (sparse.csr_matrix(der.GRAD_1d(p1, self.greville[0].size, None)), sparse.csr_matrix(der.GRAD_1d(p2, self.greville[1].size + p2, True)))
        
            
        self.x = lambda r, theta : r*np.cos(2*np.pi*theta)
        self.y = lambda r, theta : r*np.sin(2*np.pi*theta)
        
        
    def get_controlpoints(self):
        
            
        n1 = self.greville[0].size
        n2 = self.greville[1].size

        g1, g2 = np.meshgrid(self.greville[0], self.greville[1], indexing='ij')

        cx = self.interpolation.solve(self.x(g1, g2).flatten())
        cy = self.interpolation.solve(self.y(g1, g2).flatten())
        
        cx = np.reshape(cx, (self.greville[0].size, self.greville[1].size))
        cy = np.reshape(cy, (self.greville[0].size, self.greville[1].size))
        
        return cx, cy
    
    
    def mapping(self, c, q, component):
        
        cx, cy = c
        q1, q2 = q
           
        N1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0], q1, self.b[0]))
        N2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1], self.p[1], q2, self.b[1]))
        
        if component == 'x':
            return sparse.kron(N1, N2).dot(cx.flatten())
        
        if component == 'y':
            return sparse.kron(N1, N2).dot(cy.flatten())
     
    
    def jacobian_inverse_trans(self, c, q):
        
        cx, cy = c
        q1, q2 = q
        
        grad1 = sparse.kron(self.grad[0], np.identity(self.greville[1].size))
        grad2 = sparse.kron(np.identity(self.greville[0].size), self.grad[1])
        
        N1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0], q1, self.b[0]))
        N2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1], self.p[1], q2, self.b[1]))
        
        D1 = sparse.csr_matrix(bsp.collocation_matrix(self.T[0][1:-1], self.p[0] - 1, q1, self.b[0], normalize=True))
        D2 = sparse.csr_matrix(bsp.collocation_matrix(self.T[1][1:-1], self.p[1] - 1, q2, self.b[1], normalize=True))
        
        # jacobian
        xx = sparse.kron(D1, N2).dot(grad1.dot(cx.flatten()))
        xy = sparse.kron(N1, D2).dot(grad2.dot(cx.flatten()))
        
        yx = sparse.kron(D1, N2).dot(grad1.dot(cy.flatten()))
        yy = sparse.kron(N1, D2).dot(grad2.dot(cy.flatten()))
        
        return [[xx, xy], [yx, yy]]
        
        
    def barycentric_coordinates(self, c):
        
        cx, cy = c
        
        tau0 = (-2*cx[1, :]).max()
        tau1 = (cx[1, :] - np.sqrt(3)*cy[1, :]).max()
        tau2 = (cx[1, :] + np.sqrt(3)*cy[1, :]).max()
        
        tau = np.array([tau0, tau1, tau2]).max()
        
        lambda0 = lambda x, y : 1/3 + 2/(2*tau)*x
        lambda1 = lambda x, y : 1/3 - 1/(3*tau)*x + np.sqrt(3)/(2*tau)*y
        lambda2 = lambda x, y : 1/3 - 1/(3*tau)*x - np.sqrt(3)/(2*tau)*y
        
        
        return tau, [lambda0, lambda1, lambda2]
    
    

    

    
    
class C2_construction_2d:
    
    def __init__(self, p, T, b):
        
        self.p = p
        self.T = T
        self.b = b
        
        p1, p2 = p
        T1, T2_inner, T2_outer = T
        b1, b2 = b
        
        self.greville = (bsp.greville(T1, p1, b1), bsp.greville(T2_inner, p2 - 1, b2), bsp.greville(T2_outer, p2, b2))
        
        N1_inner = sparse.csc_matrix(bsp.collocation_matrix(T1, p1, self.greville[0], b1)[:, :3])
        N1_outer = sparse.csc_matrix(bsp.collocation_matrix(T1, p1, self.greville[0], b1)[:, 3:])
        
        N2_inner = sparse.csc_matrix(bsp.collocation_matrix(T2_inner, p2 - 1, self.greville[1], b2))
        N2_outer = sparse.csc_matrix(bsp.collocation_matrix(T2_outer, p2,     self.greville[2], b2))
        
        self.interpolation = splu(sparse.bmat([[sparse.kron(N1_inner, N2_inner), sparse.kron(N1_outer, N2_outer)]], format='csc'))
        
        self.x = lambda r, theta : r*np.cos(2*np.pi*theta)
        self.y = lambda r, theta : r*np.sin(2*np.pi*theta)
        
        
    def get_controlpoints(self):
            
        g1_inner, g2_inner = np.meshgrid(self.greville[0][:3], self.greville[1], indexing='ij')
        g1_outer, g2_outer = np.meshgrid(self.greville[0][3:], self.greville[2], indexing='ij')
        
        rhs_x = np.concatenate((self.x(g1_inner, g2_inner).flatten(), self.x(g1_outer, g2_outer).flatten()))
        rhs_y = np.concatenate((self.y(g1_inner, g2_inner).flatten(), self.y(g1_outer, g2_outer).flatten()))

        cx = self.interpolation.solve(rhs_x)
        cy = self.interpolation.solve(rhs_y)
        
        return cx, cy
    
    
    def mapping(self, c, q, component):
        
        cx, cy = c
        q1, q2 = q
           
        N1_inner = sparse.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0], q1, self.b[0])[:, :3])
        N1_outer = sparse.csr_matrix(bsp.collocation_matrix(self.T[0], self.p[0], q1, self.b[0])[:, 3:])
        
        N2_inner = sparse.csr_matrix(bsp.collocation_matrix(self.T[1], self.p[1] - 1, q2, self.b[1]))
        N2_outer = sparse.csr_matrix(bsp.collocation_matrix(self.T[2], self.p[1], q2, self.b[1]))
        
        if component == 'x':
            return sparse.bmat([[sparse.kron(N1_inner, N2_inner), sparse.kron(N1_outer, N2_outer)]]).dot(cx.flatten())
        
        if component == 'y':
            return sparse.bmat([[sparse.kron(N1_inner, N2_inner), sparse.kron(N1_outer, N2_outer)]]).dot(cy.flatten())
        
    
    def get_coefficients(self, c, T):
        
        from math import factorial
        
        cx, cy = c
        T1, T2 = T
        
        # number of basis functions of mapping and FEM basis
        Nbase_2_F = len(self.T[1]) - self.p[1] - 1
        Nbase_2   = len(T2) - self.p[1] - 2
        
        
        # control points of first three circles
        cx_inner = np.reshape(cx[:3*Nbase_2_F], (3, Nbase_2_F))
        cy_inner = np.reshape(cy[:3*Nbase_2_F], (3, Nbase_2_F))
        
        
        # transformation of control points
        new_coeff = np.zeros((Nbase_2, Nbase_2_F))
        new_coeff[0, 0]  = 1.
        new_coeff[1, 0]  = 1/2
        new_coeff[-1, 0] = 1/2

        for i in range(1, Nbase_2_F):
            new_coeff[:, i] = np.roll(new_coeff[:, i - 1], 2)

        cx_1_hat = new_coeff.dot(cx_inner[1])
        cy_1_hat = new_coeff.dot(cy_inner[1])
        
        cx_2_hat = new_coeff.dot(cx_inner[2])
        cy_2_hat = new_coeff.dot(cy_inner[2])
        
        
        c_11xx = np.empty(Nbase_2)
        c_12xx = np.empty(Nbase_2)
        c_21xx = np.empty(Nbase_2)
        c_22xx = np.empty(Nbase_2)

        c_11yy = np.empty(Nbase_2)
        c_12yy = np.empty(Nbase_2)
        c_21yy = np.empty(Nbase_2)
        c_22yy = np.empty(Nbase_2)

        c_11xy = np.empty(Nbase_2)


        for j in range(Nbase_2_F):
            for k in range(2):
                
                if k == 0:
                    factor = 1.
                elif k == 1:
                    factor = 1/2

                c_11xx[2*j + k] = factor*cx_inner[1, j]*cx_inner[1, (j + k)%Nbase_2_F]
                c_12xx[2*j + k] = factor*cx_inner[1, j]*cx_inner[2, (j + k)%Nbase_2_F]
                c_21xx[2*j + k] = factor*cx_inner[2, j]*cx_inner[1, (j + k)%Nbase_2_F]
                c_22xx[2*j + k] = factor*cx_inner[2, j]*cx_inner[2, (j + k)%Nbase_2_F]

                c_11yy[2*j + k] = factor*cy_inner[1, j]*cy_inner[1, (j + k)%Nbase_2_F]
                c_12yy[2*j + k] = factor*cy_inner[1, j]*cy_inner[2, (j + k)%Nbase_2_F]
                c_21yy[2*j + k] = factor*cy_inner[2, j]*cy_inner[1, (j + k)%Nbase_2_F]
                c_22yy[2*j + k] = factor*cy_inner[2, j]*cy_inner[2, (j + k)%Nbase_2_F]

                c_11xy[2*j + k] = factor*cx_inner[1, j]*cy_inner[1, (j + k)%Nbase_2_F]
                
        
        # derivatives of first three splines
        ders = bsp.basis_funs_all_ders(self.T[0], self.p[0], 0., self.p[0], 2)
            
        d0,   d1,  d2 = ders[1, 0:3]
        dd0, dd1, dd2 = ders[2, 0:3]

        
        # starting tau
        tau = 4*self.greville[0][1]
        
        while True:
            
            # Bernstein basis functions 
            indices = np.array([[2, 0, 0], [1, 1, 0], [0, 2, 0], [1, 0, 1], [0, 1, 1], [0, 0, 2]])

            Tl      = np.empty(6)
            Tl_dx   = np.empty(6)
            Tl_dy   = np.empty(6)
            Tl_dxdx = np.empty(6)
            Tl_dydy = np.empty(6)
            Tl_dxdy = np.empty(6)


            for l in range(6):

                prefactor = 2/(factorial(indices[l, 0])*factorial(indices[l, 1])*factorial(indices[l, 2]))

                Tl[l]      = prefactor*1/9
                Tl_dx[l]   = prefactor*(2*indices[l, 0] - indices[l, 1] - indices[l, 2])/(9*tau)
                Tl_dy[l]   = prefactor*(indices[l, 1] - indices[l, 2])/(3*np.sqrt(3)*tau)
                Tl_dxdx[l] = prefactor*(2 + 3*indices[l, 0]*(3*indices[l, 0] - 5))/(9*tau**2)
                Tl_dydy[l] = prefactor*((indices[l, 2] - indices[l, 1])**2 - (indices[l, 1] + indices[l, 2]))/(3*tau**2)
                Tl_dxdy[l] = prefactor*((1 - 3*indices[l, 0])*(indices[l, 2] - indices[l, 1]))/(3*np.sqrt(3)*tau**2)
 
            
            # compute coefficients E0
            E0 = np.empty((6, Nbase_2))

            for l in range(6):
                E0[l] = np.ones(Nbase_2)*Tl[l]


            # compute coefficients E1
            E1 = np.empty((6, Nbase_2))

            for l in range(6):
                E1[l] = E0[l] + cx_1_hat*Tl_dx[l] + cy_1_hat*Tl_dy[l]


            # compute coefficients E2
            E2 = np.empty((6, Nbase_2))

            for l in range(6):
                E2[l] = -dd0*E0[l] - dd1*E1[l] + (dd1**2*c_11xx + dd1*dd2*c_12xx + dd1*dd2*c_21xx + dd2**2*c_22xx)*Tl_dxdx[l] + (dd1**2*c_11yy + dd1*dd2*c_12yy + dd1*dd2*c_21yy + dd2**2*c_22yy)*Tl_dydy[l] + d1**2*c_11xy*2*Tl_dxdy[l] + (dd1*cx_1_hat + dd2*cx_2_hat)*Tl_dx[l] + (dd1*cy_1_hat + dd2*cy_2_hat)*Tl_dy[l]

            E2 = E2/dd2
            
            # check if all coefficients are > 1 (if not increase tau)
            if np.all(E0 > 0) and np.all(E1 > 0) and np.all(E2 > 0):
                break
                
            else:
                tau += 2*self.greville[0][1]
            
            
        return E0, E1, E2, tau