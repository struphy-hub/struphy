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