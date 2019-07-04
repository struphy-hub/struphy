import numpy as np
import scipy.sparse as sparse
import utilitis_FEEC.bsplines as bsp

import utilitis_FEEC.kernels_mhd as kernels

from pyccel import epyccel
kernels = epyccel(kernels)

class projections_mhd:
    
    def __init__(self, p, Nbase, T, bc):
        
        self.p  = p
        self.Nbase = Nbase
        self.T  = T
        self.bc = bc
        
        p1, p2, p3 = p
        Nbase_1, Nbase_2, Nbase_3 = Nbase
        T1, T2, T3 = T
        bc_1, bc_2, bc_3 = bc
        
        t1 = T1[1:-1]
        t2 = T2[1:-1]
        t3 = T3[1:-1]
        
        self.t = (T1[1:-1], T2[1:-1], T3[1:-1])
        
        self.greville    = (bsp.greville(T1, p1, bc_1), bsp.greville(T2, p2, bc_2), bsp.greville(T3, p3, bc_3))
        self.breakpoints = (bsp.breakpoints(T1, p1),    bsp.breakpoints(T2, p2),    bsp.breakpoints(T3, p3))
        
        pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
        pts_2_loc, wts_2_loc = np.polynomial.legendre.leggauss(p2 + 1)
        pts_3_loc, wts_3_loc = np.polynomial.legendre.leggauss(p3 + 1)
        
        self.nq = (p1 + 1, p2 + 1, p3 + 1)
        
        self.pts_loc = (pts_1_loc, pts_2_loc, pts_3_loc)
        self.wts_loc = (wts_1_loc, wts_2_loc, wts_3_loc)
        
        
        self.n_grev = (self.greville[0].size, self.greville[1].size, self.greville[2].size)
        
        
        # Quadrature grid in 1-direction
        if bc_1 == True:
            
            if p1%2 != 0:
                
                grid_1 = self.breakpoints[0]
                pts_1, wts_1 = bsp.quadrature_grid(grid_1, pts_1_loc, wts_1_loc)
                
            else:
                
                grid_1 = np.append(self.greville[0], self.greville[0][-1] + (self.greville[0][-1] - self.greville[0][-2]))
                pts_1, wts_1 = bsp.quadrature_grid(grid_1, pts_1_loc, wts_1_loc)%self.breakpoints[0][-1]
                
        else:
            
            pts_1, wts_1 = bsp.quadrature_grid(self.greville[0], pts_1_loc, wts_1_loc)
            
            
        # Quadrature grid in 2-direction
        if bc_2 == True:
            
            if p2%2 != 0:
                
                grid_2 = self.breakpoints[1]
                pts_2, wts_2 = bsp.quadrature_grid(grid_2, pts_2_loc, wts_2_loc)
                
            else:
                
                grid_2 = np.append(self.greville[1], self.greville[1][-1] + (self.greville[1][-1] - self.greville[1][-2]))
                pts_2, wts_2 = bsp.quadrature_grid(grid_2, pts_2_loc, self.wts_2_loc)%self.breakpoints[1][-1]
                
        else:
            
            pts_2, wts_2 = bsp.quadrature_grid(self.greville[1], pts_2_loc, wts_2_loc)
            
            
        # Quadrature grid in 3-direction
        if bc_3 == True:
            
            if p3%2 != 0:
                
                grid_3 = self.breakpoints[2]
                pts_3, wts_3 = bsp.quadrature_grid(grid_3, pts_3_loc, wts_3_loc)
                
            else:
                
                grid_3 = np.append(self.greville[2], self.greville[2][-1] + (self.greville[2][-1] - self.greville[2][-2]))
                pts_3, wts_3 = bsp.quadrature_grid(grid_3, pts_3_loc, wts_3_loc)%self.breakpoints[2][-1]
                
        else:
            
            pts_3, wts_3 = bsp.quadrature_grid(self.greville[2], pts_3_loc, wts_3_loc)
            
        self.quad_grid = (pts_1, pts_2, pts_3)
        self.weights   = (np.asfortranarray(wts_1), np.asfortranarray(wts_2), np.asfortranarray(wts_3))
        
        
        # N- and D- basis functions at greville points
        N1_grev = sparse.csr_matrix(bsp.collocation_matrix(T1, p1, self.greville[0], bc_1)) 
        N2_grev = sparse.csr_matrix(bsp.collocation_matrix(T2, p2, self.greville[1], bc_2)) 
        N3_grev = sparse.csr_matrix(bsp.collocation_matrix(T3, p3, self.greville[2], bc_3)) 
        
        D1_grev = sparse.csr_matrix(bsp.collocation_matrix(t1, p1 - 1, self.greville[0], bc_1, normalize=True)) 
        D2_grev = sparse.csr_matrix(bsp.collocation_matrix(t2, p2 - 1, self.greville[1], bc_2, normalize=True)) 
        D3_grev = sparse.csr_matrix(bsp.collocation_matrix(t3, p3 - 1, self.greville[2], bc_3, normalize=True)) 
        
        
        # N- and D- basis functions at quadrature points for integrations between greville points
        N1_quad = sparse.csr_matrix(bsp.collocation_matrix(T1, p1, pts_1.flatten(), bc_1))
        N2_quad = sparse.csr_matrix(bsp.collocation_matrix(T2, p1, pts_2.flatten(), bc_2))
        N3_quad = sparse.csr_matrix(bsp.collocation_matrix(T3, p1, pts_3.flatten(), bc_3))
        
        D1_quad = sparse.csr_matrix(bsp.collocation_matrix(t1, p1 - 1, pts_1.flatten(), bc_1, normalize=True))
        D2_quad = sparse.csr_matrix(bsp.collocation_matrix(t2, p1 - 1, pts_2.flatten(), bc_2, normalize=True))
        D3_quad = sparse.csr_matrix(bsp.collocation_matrix(t3, p1 - 1, pts_3.flatten(), bc_3, normalize=True))
        
        
        # Evaluation matrices (mixed interpolation-histopolation points)
        self.eva1_1 = sparse.kron(sparse.kron(D1_grev, N2_quad), N3_quad)
        self.eva1_2 = sparse.kron(sparse.kron(N1_grev, D2_quad), N3_quad)
        self.eva1_3 = sparse.kron(sparse.kron(N1_grev, N2_quad), D3_quad)
        
        self.eva2_1 = sparse.kron(sparse.kron(D1_quad, N2_grev), N3_quad)
        self.eva2_2 = sparse.kron(sparse.kron(N1_quad, D2_grev), N3_quad)
        self.eva2_3 = sparse.kron(sparse.kron(N1_quad, N2_grev), D3_quad)
        
        self.eva3_1 = sparse.kron(sparse.kron(D1_quad, N2_quad), N3_grev)
        self.eva3_2 = sparse.kron(sparse.kron(N1_quad, D2_quad), N3_grev)
        self.eva3_3 = sparse.kron(sparse.kron(N1_quad, N2_quad), D3_grev)
        
        
    
    def assemble_equilibrium(self, rho_0, Ginv):
        
        Q11, Q12, Q13 = np.meshgrid(self.greville[0], self.quad_grid[1].flatten(), self.quad_grid[2].flatten(), indexing='ij')
        Q21, Q22, Q23 = np.meshgrid(self.quad_grid[0].flatten(), self.greville[1], self.quad_grid[2].flatten(), indexing='ij')
        Q31, Q32, Q33 = np.meshgrid(self.quad_grid[0].flatten(), self.quad_grid[1].flatten(), self.greville[2], indexing='ij')
        
        self.rho_0 = (rho_0(Q11, Q12, Q13), rho_0(Q21, Q22, Q23), rho_0(Q31, Q32, Q33))
        
        self.Ginv_1 = (Ginv[0][0](Q11, Q12, Q13), Ginv[0][1](Q11, Q12, Q13), Ginv[0][2](Q11, Q12, Q13))
        self.Ginv_2 = (Ginv[1][0](Q21, Q22, Q23), Ginv[1][1](Q21, Q22, Q23), Ginv[1][2](Q21, Q22, Q23))
        self.Ginv_3 = (Ginv[2][0](Q31, Q32, Q33), Ginv[2][1](Q31, Q32, Q33), Ginv[2][2](Q31, Q32, Q33))
        
     
    
    def rhs_A(self, u_vec):
        
        # Evaluation (1-component)
        u1_1 = np.reshape(self.eva1_1.dot(u_vec[0]), (self.n_grev[0], (self.n_grev[1] - 1 + self.bc[1])*(self.p[1] + 1), (self.n_grev[2] - 1 + self.bc[2])*(self.p[2] + 1)))
        
        u1_2 = np.reshape(self.eva1_2.dot(u_vec[1]), (self.n_grev[0], (self.n_grev[1] - 1 + self.bc[1])*(self.p[1] + 1), (self.n_grev[2] - 1 + self.bc[2])*(self.p[2] + 1)))
        
        u1_3 = np.reshape(self.eva1_3.dot(u_vec[2]), (self.n_grev[0], (self.n_grev[1] - 1 + self.bc[1])*(self.p[1] + 1), (self.n_grev[2] - 1 + self.bc[2])*(self.p[2] + 1)))
        
        
        # Evaluation (2-component)
        u2_1 = np.reshape(self.eva2_1.dot(u_vec[0]), ((self.n_grev[0] - 1 + self.bc[0])*(self.p[0] + 1), self.n_grev[1], (self.n_grev[2] - 1 + self.bc[2])*(self.p[2] + 1)))
        
        u2_2 = np.reshape(self.eva2_2.dot(u_vec[1]), ((self.n_grev[0] - 1 + self.bc[0])*(self.p[0] + 1), self.n_grev[1], (self.n_grev[2] - 1 + self.bc[2])*(self.p[2] + 1)))
        
        u2_3 = np.reshape(self.eva2_3.dot(u_vec[2]), ((self.n_grev[0] - 1 + self.bc[0])*(self.p[0] + 1), self.n_grev[1], (self.n_grev[2] - 1 + self.bc[2])*(self.p[2] + 1)))
        
        
        # Evaluation (3-component)
        u3_1 = np.reshape(self.eva3_1.dot(u_vec[0]), ((self.n_grev[0] - 1 + self.bc[0])*(self.p[0] + 1), (self.n_grev[1] - 1 + self.bc[1])*(self.p[1] + 1), self.n_grev[2]))
        
        u3_2 = np.reshape(self.eva3_2.dot(u_vec[1]), ((self.n_grev[0] - 1 + self.bc[0])*(self.p[0] + 1), (self.n_grev[1] - 1 + self.bc[1])*(self.p[1] + 1), self.n_grev[2]))
        
        u3_3 = np.reshape(self.eva3_3.dot(u_vec[2]), ((self.n_grev[0] - 1 + self.bc[0])*(self.p[0] + 1), (self.n_grev[1] - 1 + self.bc[1])*(self.p[1] + 1), self.n_grev[2]))
        
        
        # Final data for projection
        mat_f1 = np.asfortranarray(self.rho_0[0] * (self.Ginv_1[0]*u1_1 + self.Ginv_1[1]*u1_2 + self.Ginv_1[2]*u1_3))
        mat_f2 = np.asfortranarray(self.rho_0[1] * (self.Ginv_2[0]*u2_1 + self.Ginv_2[1]*u2_2 + self.Ginv_2[2]*u2_3))
        mat_f3 = np.asfortranarray(self.rho_0[2] * (self.Ginv_3[0]*u3_1 + self.Ginv_3[1]*u3_2 + self.Ginv_3[2]*u3_3))
        
        
        # Right-hand sides
        rhs_1 = np.empty((self.n_grev[0], self.n_grev[1] - 1 + self.bc[1], self.n_grev[2] - 1 + self.bc[2]), order='F')
        rhs_2 = np.empty((self.n_grev[0] - 1 + self.bc[0], self.n_grev[1], self.n_grev[2] - 1 + self.bc[2]), order='F')
        rhs_3 = np.empty((self.n_grev[0] - 1 + self.bc[0], self.n_grev[1] - 1 + self.bc[1], self.n_grev[2]), order='F')
        
        
        # Assembly
        kernels.kernel_A_1(self.n_grev[0], self.n_grev[1] - 1 + self.bc[1], self.n_grev[2] - 1 + self.bc[2], self.nq[1], self.nq[2], self.weights[1], self.weights[2], mat_f1, rhs_1)
        
        kernels.kernel_A_2(self.n_grev[0] - 1 + self.bc[0], self.n_grev[1], self.n_grev[2] - 1 + self.bc[2], self.nq[0], self.nq[2], self.weights[0], self.weights[2], mat_f2, rhs_2)
        
        kernels.kernel_A_3(self.n_grev[0] - 1 + self.bc[0], self.n_grev[1] - 1 + self.bc[1], self.n_grev[2], self.nq[0], self.nq[1], self.weights[0], self.weights[1], mat_f3, rhs_3)
        
        
        

        return rhs_1, rhs_2, rhs_3