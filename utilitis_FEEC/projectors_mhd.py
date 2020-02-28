import numpy                                as np
import scipy.sparse                         as sparse
import utilitis_FEEC.bsplines               as bsp
import utilitis_FEEC.kernels_projectors_mhd as kernels



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
        
        self.t  = (T1[1:-1], T2[1:-1], T3[1:-1])
        self.pr = (p1 - 1, p2 - 1, p3 - 1)
        
        self.greville    = (bsp.greville(T1, p1, bc_1) + p1%2*1e-15, bsp.greville(T2, p2, bc_2) + p2%2*1e-15, bsp.greville(T3, p3, bc_3) + p3%2*1e-15)
        self.breakpoints = (bsp.breakpoints(T1, p1),      bsp.breakpoints(T2, p2),    bsp.breakpoints(T3, p3))
        self.Nel         = (len(self.breakpoints[0]) - 1, len(self.breakpoints[1]) - 1, len(self.breakpoints[2]) - 1)
        
        pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
        pts_2_loc, wts_2_loc = np.polynomial.legendre.leggauss(p2 + 1)
        pts_3_loc, wts_3_loc = np.polynomial.legendre.leggauss(p3 + 1)
        
        self.nq = (p1 + 1, p2 + 1, p3 + 1)
        
        self.pts_loc = (pts_1_loc, pts_2_loc, pts_3_loc)
        self.wts_loc = (wts_1_loc, wts_2_loc, wts_3_loc)
        
        
        self.n_grev = (self.greville[0].size, self.greville[1].size, self.greville[2].size)
        
        
        #========== Quadrature grid for histopolations in 1-direction ==========
        if bc_1 == True:
            
            if p1%2 != 0:
                
                grid = self.breakpoints[0]
                pts_1, wts_1 = bsp.quadrature_grid(grid, pts_1_loc, wts_1_loc)
                
                ne_1 = self.Nel[0]
                
                ies_1 = np.arange(ne_1)
                
            else:
                
                dd = 1./self.Nel[0]
                
                grid = np.append(self.greville[0], self.greville[0] + dd/2)
                grid = np.append(grid, grid[-1] + dd/2)
                grid.sort()
                
                pts_1, wts_1 = bsp.quadrature_grid(grid, pts_1_loc, wts_1_loc)
                pts_1 = pts_1%1.
                
                ne_1 = 2*self.Nel[0]
                
                ies_1 = np.floor_divide(np.arange(ne_1), 2)
                
        else:
            
            if p1%2 != 0:
                
                grid = self.greville[0]
                pts_1, wts_1 = bsp.quadrature_grid(grid, pts_1_loc, wts_1_loc)
                                        
                ne_1 = len(self.greville[0]) - 1
                
                ies_1 = np.arange(ne_1)
                
            else:
                
                grid = np.union1d(self.greville[0], self.breakpoints[0])
                pts_1, wts_1 = bsp.quadrature_grid(grid, pts_1_loc, wts_1_loc)
                
                ne_1 = 2*self.Nel[0] + p1 - 2
                
                boundaries = int(p1/2)
                ies_1 = np.floor_divide(np.arange(ne_1 - p1), 2) + boundaries
                ies_1 = np.array(list(np.arange(boundaries)) + list(ies_1) + list(np.arange(boundaries) + ies_1[-1] + 1))
        #=======================================================================
            
            
        #========== Quadrature grid for histopolations in 2-direction ==========
        if bc_2 == True:
            
            if p2%2 != 0:
                
                grid = self.breakpoints[1]
                pts_2, wts_2 = bsp.quadrature_grid(grid, pts_2_loc, wts_2_loc)
                                        
                ne_2 = self.Nel[1]
                
                ies_2 = np.arange(ne_2)
                
            else:
                
                dd = 1./self.Nel[1]
                
                grid = np.append(self.greville[1], self.greville[1] + dd/2)
                grid = np.append(grid, grid[-1] + dd/2)
                grid.sort()
                
                pts_2, wts_2 = bsp.quadrature_grid(grid, pts_2_loc, wts_2_loc)
                pts_2 = pts_2%1.
                                        
                ne_2 = 2*self.Nel[1]
                
                ies_2 = np.floor_divide(np.arange(ne_2), 2)
                
        else:
            
            if p2%2 != 0:
                
                grid = self.greville[1]
                pts_2, wts_2 = bsp.quadrature_grid(grid, pts_2_loc, wts_2_loc)
                                        
                ne_2 = len(self.greville[1]) - 1
                
                ies_2 = np.arange(ne_2)
                
            else:
                
                grid = np.union1d(self.greville[1], self.breakpoints[1])
                pts_2, wts_2 = bsp.quadrature_grid(grid, pts_2_loc, wts_2_loc)
                
                ne_2 = 2*self.Nel[1] + p2 - 2
                
                boundaries = int(p2/2)
                ies_2 = np.floor_divide(np.arange(ne_2 - p2), 2) + boundaries
                ies_2 = np.array(list(np.arange(boundaries)) + list(ies_2) + list(np.arange(boundaries) + ies_2[-1] + 1))
        #=======================================================================
            
            
        #========== Quadrature grid for histopolations in 3-direction ==========
        if bc_3 == True:
            
            if p3%2 != 0:
                
                grid = self.breakpoints[2]
                pts_3, wts_3 = bsp.quadrature_grid(grid, pts_3_loc, wts_3_loc)
                                        
                ne_3 = self.Nel[2]
                
                ies_3 = np.arange(ne_3)
                
            else:
                
                dd = 1./self.Nel[2]
                
                grid = np.append(self.greville[2], self.greville[2] + dd/2)
                grid = np.append(grid, grid[-1] + dd/2)
                grid.sort()
                
                pts_3, wts_3 = bsp.quadrature_grid(grid, pts_3_loc, wts_3_loc)
                pts_3 = pts_3%1.
                                        
                ne_3 = 2*self.Nel[2]
                
                ies_3 = np.floor_divide(np.arange(ne_3), 2)
                
        else:
            
            if p3%2 != 0:
                
                grid = self.greville[2]
                pts_3, wts_3 = bsp.quadrature_grid(grid, pts_3_loc, wts_3_loc)
                                        
                ne_3 = len(self.greville[2]) - 1
                
                ies_3 = np.arange(ne_3)
                
            else:
                
                grid = np.union1d(self.greville[2], self.breakpoints[2])
                pts_3, wts_3 = bsp.quadrature_grid(grid, pts_3_loc, wts_3_loc)
                
                ne_3 = 2*self.Nel[2] + p3 - 2
                
                boundaries = int(p3/2)
                ies_3 = np.floor_divide(np.arange(ne_3 - p3), 2) + boundaries
                ies_3 = np.array(list(np.arange(boundaries)) + list(ies_3) + list(np.arange(boundaries) + ies_3[-1] + 1))
        #=======================================================================
        
        self.ies       = (ies_1, ies_2, ies_3)
        self.ne        = (ne_1, ne_2, ne_3)
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
        N2_quad = sparse.csr_matrix(bsp.collocation_matrix(T2, p2, pts_2.flatten(), bc_2))
        N3_quad = sparse.csr_matrix(bsp.collocation_matrix(T3, p3, pts_3.flatten(), bc_3))
        
        D1_quad = sparse.csr_matrix(bsp.collocation_matrix(t1, p1 - 1, pts_1.flatten(), bc_1, normalize=True))
        D2_quad = sparse.csr_matrix(bsp.collocation_matrix(t2, p2 - 1, pts_2.flatten(), bc_2, normalize=True))
        D3_quad = sparse.csr_matrix(bsp.collocation_matrix(t3, p3 - 1, pts_3.flatten(), bc_3, normalize=True))
        
        
        # Evaluation matrices for projection A (PI_2: mixed interpolation-histopolation points)
        eva11 = sparse.kron(sparse.kron(D1_grev, N2_quad), N3_quad)
        eva12 = sparse.kron(sparse.kron(N1_grev, D2_quad), N3_quad)
        eva13 = sparse.kron(sparse.kron(N1_grev, N2_quad), D3_quad)
        
        eva21 = sparse.kron(sparse.kron(D1_quad, N2_grev), N3_quad)
        eva22 = sparse.kron(sparse.kron(N1_quad, D2_grev), N3_quad)
        eva23 = sparse.kron(sparse.kron(N1_quad, N2_grev), D3_quad)
        
        eva31 = sparse.kron(sparse.kron(D1_quad, N2_quad), N3_grev)
        eva32 = sparse.kron(sparse.kron(N1_quad, D2_quad), N3_grev)
        eva33 = sparse.kron(sparse.kron(N1_quad, N2_quad), D3_grev)
        
        self.evaA = ((eva11, eva12, eva13), (eva21, eva22, eva23), (eva31, eva32, eva33))
        
        
        # Evaluation matrices for projection B (PI_1: mixed interpolation-histopolation points)
        eva11 = sparse.kron(sparse.kron(D1_quad, N2_grev), N3_grev)
        eva12 = sparse.kron(sparse.kron(N1_quad, D2_grev), N3_grev)
        eva13 = sparse.kron(sparse.kron(N1_quad, N2_grev), D3_grev)
        
        eva21 = sparse.kron(sparse.kron(D1_grev, N2_quad), N3_grev)
        eva22 = sparse.kron(sparse.kron(N1_grev, D2_quad), N3_grev)
        eva23 = sparse.kron(sparse.kron(N1_grev, N2_quad), D3_grev)
        
        eva31 = sparse.kron(sparse.kron(D1_grev, N2_grev), N3_quad)
        eva32 = sparse.kron(sparse.kron(N1_grev, D2_grev), N3_quad)
        eva33 = sparse.kron(sparse.kron(N1_grev, N2_grev), D3_quad)
        
        self.evaB = ((eva11, eva12, eva13), (eva21, eva22, eva23), (eva31, eva32, eva33))
        
        
    
    def assemble_equilibrium_A(self, rho_0, Ginv):
        
        Q11, Q12, Q13 = np.meshgrid(self.greville[0], self.quad_grid[1].flatten(), self.quad_grid[2].flatten(), indexing='ij')
        Q21, Q22, Q23 = np.meshgrid(self.quad_grid[0].flatten(), self.greville[1], self.quad_grid[2].flatten(), indexing='ij')
        Q31, Q32, Q33 = np.meshgrid(self.quad_grid[0].flatten(), self.quad_grid[1].flatten(), self.greville[2], indexing='ij')
        
        
        EQ_1 = ( rho_0(Q11, Q12, Q13) * Ginv[0][0](Q11, Q12, Q13), rho_0(Q11, Q12, Q13) * Ginv[0][1](Q11, Q12, Q13), rho_0(Q11, Q12, Q13) * Ginv[0][2](Q11, Q12, Q13) )
        
        EQ_2 = ( rho_0(Q21, Q22, Q23) * Ginv[1][0](Q21, Q22, Q23), rho_0(Q21, Q22, Q23) * Ginv[1][1](Q21, Q22, Q23), rho_0(Q21, Q22, Q23) * Ginv[1][2](Q21, Q22, Q23) )
        
        EQ_3 = ( rho_0(Q31, Q32, Q33) * Ginv[2][0](Q31, Q32, Q33), rho_0(Q31, Q32, Q33) * Ginv[2][1](Q31, Q32, Q33), rho_0(Q31, Q32, Q33) * Ginv[2][2](Q31, Q32, Q33) )
        
        self.A = (EQ_1, EQ_2, EQ_3)
        
    
    def assemble_equilibrium_B(self, Ginv, B0):
        
        Q11, Q12, Q13 = np.meshgrid(self.quad_grid[0].flatten(), self.greville[1], self.greville[2], indexing='ij')
        Q21, Q22, Q23 = np.meshgrid(self.greville[0], self.quad_grid[1].flatten(), self.greville[2], indexing='ij')
        Q31, Q32, Q33 = np.meshgrid(self.greville[0], self.greville[1], self.quad_grid[2].flatten(), indexing='ij')
        
        
        EQ_1 = ( (B0[1](Q11, Q12, Q13) * Ginv[2][0](Q11, Q12, Q13) - B0[2](Q11, Q12, Q13) * Ginv[1][0](Q11, Q12, Q13)), (B0[1](Q11, Q12, Q13) * Ginv[2][1](Q11, Q12, Q13) - B0[2](Q11, Q12, Q13) * Ginv[1][1](Q11, Q12, Q13)), (B0[1](Q11, Q12, Q13) * Ginv[2][2](Q11, Q12, Q13) - B0[2](Q11, Q12, Q13) * Ginv[1][2](Q11, Q12, Q13)) )
        
        EQ_2 = ( (B0[2](Q21, Q22, Q23) * Ginv[0][0](Q21, Q22, Q23) - B0[0](Q21, Q22, Q23) * Ginv[2][0](Q21, Q22, Q23)), (B0[2](Q21, Q22, Q23) * Ginv[0][1](Q21, Q22, Q23) - B0[0](Q21, Q22, Q23) * Ginv[2][1](Q21, Q22, Q23)), (B0[2](Q21, Q22, Q23) * Ginv[0][2](Q21, Q22, Q23) - B0[0](Q21, Q22, Q23) * Ginv[2][2](Q21, Q22, Q23)) )
        
        EQ_3 = ( (B0[0](Q31, Q32, Q33) * Ginv[1][0](Q31, Q32, Q33) - B0[1](Q31, Q32, Q33) * Ginv[0][0](Q31, Q32, Q33)), (B0[0](Q31, Q32, Q33) * Ginv[1][1](Q31, Q32, Q33) - B0[1](Q31, Q32, Q33) * Ginv[0][1](Q31, Q32, Q33)), (B0[0](Q31, Q32, Q33) * Ginv[1][2](Q31, Q32, Q33) - B0[1](Q31, Q32, Q33) * Ginv[0][2](Q31, Q32, Q33)) )
        
        self.B = (EQ_1, EQ_2, EQ_3)
        
    
    
    def projection_Q(self, rho0, Ginv):
        '''
        Computes the right-hand sides for each basis function of the expression Pi_2(rho0 * Ginv * lambda^1)
        '''
        
        p1,    p2,   p3 = self.p         # spline degrees
        bc1,  bc2,  bc3 = self.bc        # boundary conditions
        nq1,  nq2,  nq3 = self.nq        # number of quadrature points per element
        w1,    w2,   w3 = self.weights   # quadrature weights
        
        
        
        # number of intervals for three components of the projector Pi2 (cyclic permutation of interpolation in i and integration in j and k)
        n1 = [self.n_grev[0], self.n_grev[1] - 1 + bc2, self.n_grev[2] - 1 + bc3]
        n2 = [self.n_grev[0] - 1 + bc1, self.n_grev[1], self.n_grev[2] - 1 + bc3]
        n3 = [self.n_grev[0] - 1 + bc1, self.n_grev[1] - 1 + bc2, self.n_grev[2]]
        
        
        # number of non-vanishing splines per interval (for even degree one additional spline per integration interval!)
        nv1 = [[p1, p2 + 2 - p2%2, p3 + 2 - p3%2], [p1 + 1, p2 + 1 - p2%2, p3 + 2 - p3%2], [p1 + 1, p2 + 2 - p2%2, p3 + 1 - p3%2]]
        nv2 = [[p1 + 1 - p1%2, p2 + 1, p3 + 2 - p3%2], [p1 + 2 - p1%2, p2, p3 + 2 - p3%2], [p1 + 2 - p1%2, p2 + 1, p3 + 1 - p3%2]]
        nv3 = [[p1 + 1 - p1%2, p2 + 2 - p2%2, p3 + 1], [p1 + 2 - p1%2, p2 + 1 - p2%2, p3 + 1], [p1 + 2 - p1%2, p2 + 2 - p2%2, p3]]
        
        
        # reshape greville points (n x 1)
        PP = [np.reshape(greville, (n_greville, 1)) for greville, n_greville in zip(self.greville, self.n_grev)]
        
        # Evaluate N - functions on interpolation and quadrature points
        N_int = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, PP, 0, normalize=False)) for T, p, PP in zip(self.T, self.p, PP)]
        
        N_his = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad, 0, normalize=False)) for T, p, quad in zip(self.T, self.p, self.quad_grid)]
        
        
        # Evaluate D - functions on interpolation and quadrature points
        D_int = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p, PP, 0, normalize=True)) for t, p, PP in zip(self.t, self.pr, PP)] 
        
        D_his = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p, quad, 0, normalize=True)) for t, p, quad in zip(self.t, self.pr, self.quad_grid)]
        
        
        # Evaluate equilibrium quantities on interpolation and quadrature points
        P11, P12, P13 = np.meshgrid(self.greville[0], self.quad_grid[1].flatten(), self.quad_grid[2].flatten(), indexing='ij')
        P21, P22, P23 = np.meshgrid(self.quad_grid[0].flatten(), self.greville[1], self.quad_grid[2].flatten(), indexing='ij')
        P31, P32, P33 = np.meshgrid(self.quad_grid[0].flatten(), self.quad_grid[1].flatten(), self.greville[2], indexing='ij')
        
        EQ1 = [np.asfortranarray(rho0(P11, P12, P13)*Ginv(P11, P12, P13)) for Ginv in Ginv[0]]
        EQ2 = [np.asfortranarray(rho0(P21, P22, P23)*Ginv(P21, P22, P23)) for Ginv in Ginv[1]]
        EQ3 = [np.asfortranarray(rho0(P31, P32, P33)*Ginv(P31, P32, P33)) for Ginv in Ginv[2]]
        
        
        # Local storage of right-hand-sides (for even degree one additional spline per integration interval!)
        RHS1 = [np.zeros((n1[0], n1[1], n1[2], nv1[0], nv1[1], nv1[2]), order='F') for nv1 in nv1]
        RHS2 = [np.zeros((n2[0], n2[1], n2[2], nv2[0], nv2[1], nv2[2]), order='F') for nv2 in nv2]
        RHS3 = [np.zeros((n3[0], n3[1], n3[2], nv3[0], nv3[1], nv3[2]), order='F') for nv3 in nv3]
         
        
        #==================shift in local indices for integrations============================================================
        il_add = []
               
        for mu in range(3):
               
            if self.bc[mu] == True:
                   
                if self.p[mu]%2 != 0:
                    il_add.append(np.array([0] * self.ne[mu]))
               
                else:
                    il_add.append(np.array([0, 1] * self.Nel[mu]))
                                     
            else:
                                     
                if self.p[mu]%2 != 0:
                    il_add.append(np.array([0] * self.ne[mu]))
                                     
                else:
                    boundaries = int(self.p[mu]/2)
                    il_add.append(np.array([0] * boundaries + [0, 1] * int((self.ne[0] - self.p[mu])/2) + [1] * boundaries))
        #=====================================================================================================================
                        
        
        
        # Assembly of the first line
        kernels.kernel_pi2_1(n1[0], self.ne[1], self.ne[2], p1, p2 + 1, p3 + 1, self.ies[1], self.ies[2], il_add[1], il_add[2], nq2, nq3, w2, w3, D_int[0], N_his[1], N_his[2], EQ1[0], RHS1[0])
                                     
        kernels.kernel_pi2_1(n1[0], self.ne[1], self.ne[2], p1 + 1, p2, p3 + 1, self.ies[1], self.ies[2], il_add[1], il_add[2], nq2, nq3, w2, w3, N_int[0], D_his[1], N_his[2], EQ1[1], RHS1[1])
                                     
        kernels.kernel_pi2_1(n1[0], self.ne[1], self.ne[2], p1 + 1, p2 + 1, p3, self.ies[1], self.ies[2], il_add[1], il_add[2], nq2, nq3, w2, w3, N_int[0], N_his[1], D_his[2], EQ1[2], RHS1[2])
        
        
        # Assembly of the second line
        kernels.kernel_pi2_2(self.ne[0], n2[1], self.ne[2], p1, p2 + 1, p3 + 1, self.ies[0], self.ies[2], il_add[0], il_add[2], nq1, nq3, w1, w3, D_his[0], N_int[1], N_his[2], EQ2[0], RHS2[0])
                                     
        kernels.kernel_pi2_2(self.ne[0], n2[1], self.ne[2], p1 + 1, p2, p3 + 1, self.ies[0], self.ies[2], il_add[0], il_add[2], nq1, nq3, w1, w3, N_his[0], D_int[1], N_his[2], EQ2[1], RHS2[1])
                                     
        kernels.kernel_pi2_2(self.ne[0], n2[1], self.ne[2], p1 + 1, p2 + 1, p3, self.ies[0], self.ies[2], il_add[0], il_add[2], nq1, nq3, w1, w3, N_his[0], N_int[1], D_his[2], EQ2[2], RHS2[2])
        
        
        # Assembly of the third line
        kernels.kernel_pi2_3(self.ne[0], self.ne[1], n3[2], p1, p2 + 1, p3 + 1, self.ies[0], self.ies[1], il_add[0], il_add[1], nq1, nq2, w1, w2, D_his[0], N_his[1], N_int[2], EQ3[0], RHS3[0])
                                     
        kernels.kernel_pi2_3(self.ne[0], self.ne[1], n3[2], p1 + 1, p2, p3 + 1, self.ies[0], self.ies[1], il_add[0], il_add[1], nq1, nq2, w1, w2, N_his[0], D_his[1], N_int[2], EQ3[1], RHS3[1])
                                     
        kernels.kernel_pi2_3(self.ne[0], self.ne[1], n3[2], p1 + 1, p2 + 1, p3, self.ies[0], self.ies[1], il_add[0], il_add[1], nq1, nq2, w1, w2, N_his[0], N_his[1], D_int[2], EQ3[2], RHS3[2])
        
        
        # Grid indices               
        indices1 = [np.indices((n1[0], n1[1], n1[2], nv1[0], nv1[1], nv1[2])) for nv1 in nv1]
        indices2 = [np.indices((n2[0], n2[1], n2[2], nv2[0], nv2[1], nv2[2])) for nv2 in nv2]
        indices3 = [np.indices((n3[0], n3[1], n3[2], nv3[0], nv3[1], nv3[2])) for nv3 in nv3]                             
        
        
        
        # Row indices of global matrix
        row1 = [(n1[1]*n1[2]*indices1[0] + n1[2]*indices1[1] + indices1[2]).flatten() for indices1 in indices1]
        row2 = [(n2[1]*n2[2]*indices2[0] + n2[2]*indices2[1] + indices2[2]).flatten() for indices2 in indices2]
        row3 = [(n3[1]*n3[2]*indices3[0] + n3[2]*indices3[1] + indices3[2]).flatten() for indices3 in indices3]
        
        
        
        # Column indices of global matrix in 1-direction
        if bc1 == True:
            
            col1_1 = [(indices1[3] + np.arange(n1[0])[:, None, None, None, None, None])%n1[0] for indices1 in indices1]
            col1_2 = [(indices2[3] + np.arange(n2[0])[:, None, None, None, None, None])%n2[0] for indices2 in indices2]
            col1_3 = [(indices3[3] + np.arange(n3[0])[:, None, None, None, None, None])%n3[0] for indices3 in indices3]
                                        
        else:
            
            print('not yet implemented!')
            
        
        # Column indices of global matrix in 2-direction
        if bc2 == True:
                                     
            col2_1 = [(indices1[4] + np.arange(n1[1])[None, :, None, None, None, None])%n1[1] for indices1 in indices1]
            col2_2 = [(indices2[4] + np.arange(n2[1])[None, :, None, None, None, None])%n2[1] for indices2 in indices2]
            col2_3 = [(indices3[4] + np.arange(n3[1])[None, :, None, None, None, None])%n3[1] for indices3 in indices3]
               
        else:
            
            print('not yet implemented!')
            
            
        # Column indices of global matrix in 3-direction
        if bc3 == True:
                                     
            col3_1 = [(indices1[5] + np.arange(n1[2])[None, None, :, None, None, None])%n1[2] for indices1 in indices1]
            col3_2 = [(indices2[5] + np.arange(n2[2])[None, None, :, None, None, None])%n2[2] for indices2 in indices2]
            col3_3 = [(indices3[5] + np.arange(n3[2])[None, None, :, None, None, None])%n3[2] for indices3 in indices3]
               
        else:
            
            print('not yet implemented!')
        
        col1 = [(n1[1]*n1[2]*col1_1 + n1[2]*col2_1 + col3_1).flatten() for col1_1, col2_1, col3_1 in zip(col1_1, col2_1, col3_1)]
        col2 = [(n2[1]*n2[2]*col1_2 + n2[2]*col2_2 + col3_2).flatten() for col1_2, col2_2, col3_2 in zip(col1_2, col2_2, col3_2)]
        col3 = [(n3[1]*n3[2]*col1_3 + n3[2]*col2_3 + col3_3).flatten() for col1_3, col2_3, col3_3 in zip(col1_3, col2_3, col3_3)]
        

        # Create sparse matrices (1 - component)
        R1 = [sparse.csc_matrix((RHS1.flatten(), (row1, col1)), shape=(n1[0]*n1[1]*n1[2], n1[0]*n1[1]*n1[2])) for RHS1, row1, col1 in zip(RHS1, row1, col1)]
                  
        R1[0].eliminate_zeros()
        R1[1].eliminate_zeros()
        R1[2].eliminate_zeros()
        
        R1 = sparse.bmat([R1], format='csc')
        
        
        # Create sparse matrices (2 - component)
        R2 = [sparse.csc_matrix((RHS2.flatten(), (row2, col2)), shape=(n2[0]*n2[1]*n2[2], n2[0]*n2[1]*n2[2])) for RHS2, row2, col2 in zip(RHS2, row2, col2)]
                  
        R2[0].eliminate_zeros()
        R2[1].eliminate_zeros()
        R2[2].eliminate_zeros()
        
        R2 = sparse.bmat([R2], format='csc')
        
        
        # Create sparse matrices (3 - component)
        R3 = [sparse.csc_matrix((RHS3.flatten(), (row3, col3)), shape=(n3[0]*n3[1]*n3[2], n3[0]*n3[1]*n3[2])) for RHS3, row3, col3 in zip(RHS3, row3, col3)]
                  
        R3[0].eliminate_zeros()
        R3[1].eliminate_zeros()
        R3[2].eliminate_zeros()
        
        R3 = sparse.bmat([R3], format='csc')
        
        return R1, R2, R3
    
    
    
    
    
    def projection_W(self, rho0, g_sqrt):
        '''
        Computes the right-hand sides for each basis function of the expression Pi_1(rho0/g_sqrt * lambda^1)
        '''
        
        p1,    p2,   p3 = self.p         # spline degrees
        bc1,  bc2,  bc3 = self.bc        # boundary conditions
        nq1,  nq2,  nq3 = self.nq        # number of quadrature points per element
        w1,    w2,   w3 = self.weights   # quadrature weights
        
        
        
        # number of intervals for three components of the projector Pi1 (cyclic permutation of integration in i and interpolation in j and k)
        n1 = [self.n_grev[0] - 1 + bc1, self.n_grev[1], self.n_grev[2]]
        n2 = [self.n_grev[0], self.n_grev[1] - 1 + bc2, self.n_grev[2]]
        n3 = [self.n_grev[0], self.n_grev[1], self.n_grev[2] - 1 + bc3]
        
        
        # number of non-vanishing splines per interval (for even degree one additional spline per integration interval!)
        nv1 = [p1 + 1 - p1%2, p2 + 1, p3 + 1]
        nv2 = [p1 + 1, p2 + 1 - p2%2, p3 + 1]
        nv3 = [p1 + 1, p2 + 1, p3 + 1 - p3%2]
        
        
        # reshape greville points (n x 1)
        PP = [np.reshape(greville, (n_greville, 1)) for greville, n_greville in zip(self.greville, self.n_grev)]
        
        
        # Evaluate N - functions on interpolation points
        N_int = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, PP, 0, normalize=False)) for T, p, PP in zip(self.T, self.p, PP)]
        
        
        # Evaluate D - functions on quadrature points
        D_his = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p, quad, 0, normalize=True)) for t, p, quad in zip(self.t, self.pr, self.quad_grid)]
        
        
        # Evaluate equilibrium quantities on interpolation and quadrature points
        P11, P12, P13 = np.meshgrid(self.quad_grid[0].flatten(), self.greville[1], self.greville[2], indexing='ij')
        P21, P22, P23 = np.meshgrid(self.greville[0], self.quad_grid[1].flatten(), self.greville[2], indexing='ij')
        P31, P32, P33 = np.meshgrid(self.greville[0], self.greville[1], self.quad_grid[2].flatten(), indexing='ij')
        
        EQ1 = np.asfortranarray(rho0(P11, P12, P13) / g_sqrt(P11, P12, P13))
        EQ2 = np.asfortranarray(rho0(P21, P22, P23) / g_sqrt(P21, P22, P23))
        EQ3 = np.asfortranarray(rho0(P31, P32, P33) / g_sqrt(P31, P32, P33))
        
        
        # Local storage of right-hand-sides (for even degree one additional spline per integration interval!)
        RHS1 = np.zeros((n1[0], n1[1], n1[2], nv1[0], nv1[1], nv1[2]), order='F')
        RHS2 = np.zeros((n2[0], n2[1], n2[2], nv2[0], nv2[1], nv2[2]), order='F')
        RHS3 = np.zeros((n3[0], n3[1], n3[2], nv3[0], nv3[1], nv3[2]), order='F') 
        
        
        #==================shift in local indices for integrations============================================================
        il_add = []
               
        for mu in range(3):
               
            if self.bc[mu] == True:
                   
                if self.p[mu]%2 != 0:
                    il_add.append(np.array([0] * self.ne[mu]))
               
                else:
                    il_add.append(np.array([0, 1] * self.Nel[mu]))
                                     
            else:
                                     
                if self.p[mu]%2 != 0:
                    il_add.append(np.array([0] * self.ne[mu]))
                                     
                else:
                    boundaries = int(self.p[mu]/2)
                    il_add.append(np.array([0] * boundaries + [0, 1] * int((self.ne[0] - self.p[mu])/2) + [1] * boundaries))
        #=====================================================================================================================
        
        
        
        # Assembly of the first line
        kernels.kernel_pi1_1(self.ne[0], n1[1], n1[2], p1, p2 + 1, p3 + 1, self.ies[0], il_add[0], nq1, w1, D_his[0], N_int[1], N_int[2], EQ1, RHS1)
        
        
        # Assembly of the second line
        kernels.kernel_pi1_2(n2[0], self.ne[1], n2[2], p1 + 1, p2, p3 + 1, self.ies[1], il_add[1], nq2, w2, N_int[0], D_his[1], N_int[2], EQ2, RHS2)
        
        
        # Assembly of the third line
        kernels.kernel_pi1_3(n3[0], n3[1], self.ne[2], p1 + 1, p2 + 1, p3, self.ies[2], il_add[2], nq3, w3, N_int[0], N_int[1], D_his[2], EQ3, RHS3)
        
        
         
        # Grid indices               
        indices1 = np.indices((n1[0], n1[1], n1[2], nv1[0], nv1[1], nv1[2]))
        indices2 = np.indices((n2[0], n2[1], n2[2], nv2[0], nv2[1], nv2[2]))
        indices3 = np.indices((n3[0], n3[1], n3[2], nv3[0], nv3[1], nv3[2]))   
        
        
        # Row indices of global matrix
        row1 = (n1[1]*n1[2]*indices1[0] + n1[2]*indices1[1] + indices1[2]).flatten()
        row2 = (n2[1]*n2[2]*indices2[0] + n2[2]*indices2[1] + indices2[2]).flatten()
        row3 = (n3[1]*n3[2]*indices3[0] + n3[2]*indices3[1] + indices3[2]).flatten()
        
        
        # Column indices of global matrix in 1-direction
        if bc1 == True:
            
            col1_1 = (indices1[3] + np.arange(n1[0])[:, None, None, None, None, None])%n1[0]
            col1_2 = (indices2[3] + np.arange(n2[0])[:, None, None, None, None, None])%n2[0] 
            col1_3 = (indices3[3] + np.arange(n3[0])[:, None, None, None, None, None])%n3[0]
                                        
        else:
            
            print('not yet implemented!')
            
        
        # Column indices of global matrix in 2-direction
        if bc2 == True:
                                     
            col2_1 = (indices1[4] + np.arange(n1[1])[None, :, None, None, None, None])%n1[1]
            col2_2 = (indices2[4] + np.arange(n2[1])[None, :, None, None, None, None])%n2[1]
            col2_3 = (indices3[4] + np.arange(n3[1])[None, :, None, None, None, None])%n3[1]
               
        else:
            
            print('not yet implemented!')
            
            
        # Column indices of global matrix in 3-direction
        if bc3 == True:
                                     
            col3_1 = (indices1[5] + np.arange(n1[2])[None, None, :, None, None, None])%n1[2]
            col3_2 = (indices2[5] + np.arange(n2[2])[None, None, :, None, None, None])%n2[2]
            col3_3 = (indices3[5] + np.arange(n3[2])[None, None, :, None, None, None])%n3[2]
               
        else:
            
            print('not yet implemented!')
        
        col1 = (n1[1]*n1[2]*col1_1 + n1[2]*col2_1 + col3_1).flatten()
        col2 = (n2[1]*n2[2]*col1_2 + n2[2]*col2_2 + col3_2).flatten()
        col3 = (n3[1]*n3[2]*col1_3 + n3[2]*col2_3 + col3_3).flatten()
        
        
        # Create sparse matrix (1 - component)
        R1 = sparse.csc_matrix((RHS1.flatten(), (row1, col1)), shape=(n1[0]*n1[1]*n1[2], n1[0]*n1[1]*n1[2]))         
        R1.eliminate_zeros()
        
        # Create sparse matrix (2 - component)
        R2 = sparse.csc_matrix((RHS2.flatten(), (row2, col2)), shape=(n2[0]*n2[1]*n2[2], n2[0]*n2[1]*n2[2]))
        R2.eliminate_zeros()
        
        # Create sparse matrix (3 - component)
        R3 = sparse.csc_matrix((RHS3.flatten(), (row3, col3)), shape=(n3[0]*n3[1]*n3[2], n3[0]*n3[1]*n3[2]))
        R3.eliminate_zeros()
        
        
        return R1, R2, R3 
    
    
    
    
    
    
    def projection_T(self, B0, Ginv):
        '''
        Computes the right-hand sides for each basis function of the expression Pi_1(B_eq * Ginv * lambda^1)
        '''
        
        p1,    p2,   p3 = self.p         # spline degrees
        bc1,  bc2,  bc3 = self.bc        # boundary conditions
        nq1,  nq2,  nq3 = self.nq        # number of quadrature points per element
        w1,    w2,   w3 = self.weights   # quadrature weights
        
        
        
        # number of intervals for three components of the projector Pi1 (cyclic permutation of integration in i and interpolation in j and k)
        n1 = [self.n_grev[0] - 1 + bc1, self.n_grev[1], self.n_grev[2]]
        n2 = [self.n_grev[0], self.n_grev[1] - 1 + bc2, self.n_grev[2]]
        n3 = [self.n_grev[0], self.n_grev[1], self.n_grev[2] - 1 + bc3]
        
        
        
        
        
        # number of non-vanishing splines per interval (for even degree one additional spline per integration interval!)
        nv1 = [[p1 + 1 - p1%2, p2 + 1, p3 + 1], [p1 + 2 - p1%2, p2, p3 + 1], [p1 + 2 - p1%2, p2 + 1, p3]]
        nv2 = [[p1, p2 + 2 - p2%2, p3 + 1], [p1 + 1, p2 + 1 - p2%2, p3 + 1], [p1 + 1, p2 + 2 - p2%2, p3]]
        nv3 = [[p1, p2 + 1, p3 + 2 - p3%2], [p1 + 1, p2, p3 + 2 - p3%2], [p1 + 1, p2 + 1, p3 + 1 - p3%2]]
        
        
        # reshape greville points (n x 1)
        PP = [np.reshape(greville, (n_greville, 1)) for greville, n_greville in zip(self.greville, self.n_grev)]
        
        
        # Evaluate N - functions on interpolation and quadrature points
        N_int = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, PP, 0, normalize=False)) for T, p, PP in zip(self.T, self.p, PP)]
        
        N_his = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad, 0, normalize=False)) for T, p, quad in zip(self.T, self.p, self.quad_grid)]
        
        
        # Evaluate D - functions on interpolation and quadrature points
        D_int = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p, PP, 0, normalize=True)) for t, p, PP in zip(self.t, self.pr, PP)] 
        
        D_his = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p, quad, 0, normalize=True)) for t, p, quad in zip(self.t, self.pr, self.quad_grid)]
        
        
        # Evaluate equilibrium quantities on interpolation and quadrature points
        P11, P12, P13 = np.meshgrid(self.quad_grid[0].flatten(), self.greville[1], self.greville[2], indexing='ij')
        P21, P22, P23 = np.meshgrid(self.greville[0], self.quad_grid[1].flatten(), self.greville[2], indexing='ij')
        P31, P32, P33 = np.meshgrid(self.greville[0], self.greville[1], self.quad_grid[2].flatten(), indexing='ij')
        
        EQ11 = np.asfortranarray(B0[1](P11, P12, P13)*Ginv[2][0](P11, P12, P13) - B0[2](P11, P12, P13)*Ginv[1][0](P11, P12, P13))
        EQ12 = np.asfortranarray(B0[1](P11, P12, P13)*Ginv[2][1](P11, P12, P13) - B0[2](P11, P12, P13)*Ginv[1][1](P11, P12, P13))
        EQ13 = np.asfortranarray(B0[1](P11, P12, P13)*Ginv[2][2](P11, P12, P13) - B0[2](P11, P12, P13)*Ginv[1][2](P11, P12, P13))
        EQ1 = [EQ11, EQ12, EQ13]
        
        EQ21 = np.asfortranarray(B0[2](P21, P22, P23)*Ginv[0][0](P21, P22, P23) - B0[0](P21, P22, P23)*Ginv[2][0](P21, P22, P23))
        EQ22 = np.asfortranarray(B0[2](P21, P22, P23)*Ginv[0][1](P21, P22, P23) - B0[0](P21, P22, P23)*Ginv[2][1](P21, P22, P23))
        EQ23 = np.asfortranarray(B0[2](P21, P22, P23)*Ginv[0][2](P21, P22, P23) - B0[0](P21, P22, P23)*Ginv[2][2](P21, P22, P23))
        EQ2 = [EQ21, EQ22, EQ23]
                                 
        
        EQ31 = np.asfortranarray(B0[0](P31, P32, P33)*Ginv[1][0](P31, P32, P33) - B0[1](P31, P32, P33)*Ginv[0][0](P31, P32, P33))
        EQ32 = np.asfortranarray(B0[0](P31, P32, P33)*Ginv[1][1](P31, P32, P33) - B0[1](P31, P32, P33)*Ginv[0][1](P31, P32, P33))
        EQ33 = np.asfortranarray(B0[0](P31, P32, P33)*Ginv[1][2](P31, P32, P33) - B0[1](P31, P32, P33)*Ginv[0][2](P31, P32, P33))
        EQ3 = [EQ31, EQ32, EQ33]
        
        
        # Local storage of right-hand-sides (for even degree one additional spline per integration interval!)
        RHS1 = [np.zeros((n1[0], n1[1], n1[2], nv1[0], nv1[1], nv1[2]), order='F') for nv1 in nv1]
        RHS2 = [np.zeros((n2[0], n2[1], n2[2], nv2[0], nv2[1], nv2[2]), order='F') for nv2 in nv2]
        RHS3 = [np.zeros((n3[0], n3[1], n3[2], nv3[0], nv3[1], nv3[2]), order='F') for nv3 in nv3]
        
        
        
        #==================shift in local indices for integrations============================================================
        il_add = []
               
        for mu in range(3):
               
            if self.bc[mu] == True:
                   
                if self.p[mu]%2 != 0:
                    il_add.append(np.array([0] * self.ne[mu]))
               
                else:
                    il_add.append(np.array([0, 1] * self.Nel[mu]))
                                     
            else:
                                     
                if self.p[mu]%2 != 0:
                    il_add.append(np.array([0] * self.ne[mu]))
                                     
                else:
                    boundaries = int(self.p[mu]/2)
                    il_add.append(np.array([0] * boundaries + [0, 1] * int((self.ne[0] - self.p[mu])/2) + [1] * boundaries))
        #=====================================================================================================================
                       
        
        
        # Assembly of the first line
        kernels.kernel_pi1_1(self.ne[0], n1[1], n1[2], p1, p2 + 1, p3 + 1, self.ies[0], il_add[0], nq1, w1, D_his[0], N_int[1], N_int[2], EQ1[0], RHS1[0])
                                     
        kernels.kernel_pi1_1(self.ne[0], n1[1], n1[2], p1 + 1, p2, p3 + 1, self.ies[0], il_add[0], nq1, w1, N_his[0], D_int[1], N_int[2], EQ1[1], RHS1[1])
                                     
        kernels.kernel_pi1_1(self.ne[0], n1[1], n1[2], p1 + 1, p2 + 1, p3, self.ies[0], il_add[0], nq1, w1, N_his[0], N_int[1], D_int[2], EQ1[2], RHS1[2])
        
        
        # Assembly of the second line
        kernels.kernel_pi1_2(n2[0], self.ne[1], n2[2], p1, p2 + 1, p3 + 1, self.ies[1], il_add[1], nq2, w2, D_int[0], N_his[1], N_int[2], EQ2[0], RHS2[0])
                                     
        kernels.kernel_pi1_2(n2[0], self.ne[1], n2[2], p1 + 1, p2, p3 + 1, self.ies[1], il_add[1], nq2, w2, N_int[0], D_his[1], N_int[2], EQ2[1], RHS2[1])
                                     
        kernels.kernel_pi1_2(n2[0], self.ne[1], n2[2], p1 + 1, p2 + 1, p3, self.ies[1], il_add[1], nq2, w2, N_int[0], N_his[1], D_int[2], EQ2[2], RHS2[2])
        
        
        # Assembly of the third line
        kernels.kernel_pi1_3(n3[0], n3[1], self.ne[2], p1, p2 + 1, p3 + 1, self.ies[2], il_add[2], nq3, w3, D_int[0], N_int[1], N_his[2], EQ3[0], RHS3[0])
                                     
        kernels.kernel_pi1_3(n3[0], n3[1], self.ne[2], p1 + 1, p2, p3 + 1, self.ies[2], il_add[2], nq3, w3, N_int[0], D_int[1], N_his[2], EQ3[1], RHS3[1])
                                     
        kernels.kernel_pi1_3(n3[0], n3[1], self.ne[2], p1 + 1, p2 + 1, p3, self.ies[2], il_add[2], nq3, w3, N_int[0], N_int[1], D_his[2], EQ3[2], RHS3[2])
        
        
        # Grid indices               
        indices1 = [np.indices((n1[0], n1[1], n1[2], nv1[0], nv1[1], nv1[2])) for nv1 in nv1]
        indices2 = [np.indices((n2[0], n2[1], n2[2], nv2[0], nv2[1], nv2[2])) for nv2 in nv2]
        indices3 = [np.indices((n3[0], n3[1], n3[2], nv3[0], nv3[1], nv3[2])) for nv3 in nv3]                             
        
        
        
        # Row indices of global matrix
        row1 = [(n1[1]*n1[2]*indices1[0] + n1[2]*indices1[1] + indices1[2]).flatten() for indices1 in indices1]
        row2 = [(n2[1]*n2[2]*indices2[0] + n2[2]*indices2[1] + indices2[2]).flatten() for indices2 in indices2]
        row3 = [(n3[1]*n3[2]*indices3[0] + n3[2]*indices3[1] + indices3[2]).flatten() for indices3 in indices3]
        
        
        # Column indices of global matrix in 1-direction
        if bc1 == True:
            
            col1_1 = [(indices1[3] + np.arange(n1[0])[:, None, None, None, None, None])%n1[0] for indices1 in indices1]
            col1_2 = [(indices2[3] + np.arange(n2[0])[:, None, None, None, None, None])%n2[0] for indices2 in indices2]
            col1_3 = [(indices3[3] + np.arange(n3[0])[:, None, None, None, None, None])%n3[0] for indices3 in indices3]
                                        
        else:
            
            print('not yet implemented!')
            
        
        # Column indices of global matrix in 2-direction
        if bc2 == True:
                                     
            col2_1 = [(indices1[4] + np.arange(n1[1])[None, :, None, None, None, None])%n1[1] for indices1 in indices1]
            col2_2 = [(indices2[4] + np.arange(n2[1])[None, :, None, None, None, None])%n2[1] for indices2 in indices2]
            col2_3 = [(indices3[4] + np.arange(n3[1])[None, :, None, None, None, None])%n3[1] for indices3 in indices3]
               
        else:
            
            print('not yet implemented!')
            
            
        # Column indices of global matrix in 3-direction
        if bc3 == True:
                                     
            col3_1 = [(indices1[5] + np.arange(n1[2])[None, None, :, None, None, None])%n1[2] for indices1 in indices1]
            col3_2 = [(indices2[5] + np.arange(n2[2])[None, None, :, None, None, None])%n2[2] for indices2 in indices2]
            col3_3 = [(indices3[5] + np.arange(n3[2])[None, None, :, None, None, None])%n3[2] for indices3 in indices3]
               
        else:
            
            print('not yet implemented!')
        
        col1 = [(n1[1]*n1[2]*col1_1 + n1[2]*col2_1 + col3_1).flatten() for col1_1, col2_1, col3_1 in zip(col1_1, col2_1, col3_1)]
        col2 = [(n2[1]*n2[2]*col1_2 + n2[2]*col2_2 + col3_2).flatten() for col1_2, col2_2, col3_2 in zip(col1_2, col2_2, col3_2)]
        col3 = [(n3[1]*n3[2]*col1_3 + n3[2]*col2_3 + col3_3).flatten() for col1_3, col2_3, col3_3 in zip(col1_3, col2_3, col3_3)]
        

        # Create sparse matrices (1 - component)
        R1 = [sparse.csc_matrix((RHS1.flatten(), (row1, col1)), shape=(n1[0]*n1[1]*n1[2], n1[0]*n1[1]*n1[2])) for RHS1, row1, col1 in zip(RHS1, row1, col1)]
                  
        R1[0].eliminate_zeros()
        R1[1].eliminate_zeros()
        R1[2].eliminate_zeros()
        
        R1 = sparse.bmat([R1], format='csc')
        
        
        # Create sparse matrices (2 - component)
        R2 = [sparse.csc_matrix((RHS2.flatten(), (row2, col2)), shape=(n2[0]*n2[1]*n2[2], n2[0]*n2[1]*n2[2])) for RHS2, row2, col2 in zip(RHS2, row2, col2)]
                  
        R2[0].eliminate_zeros()
        R2[1].eliminate_zeros()
        R2[2].eliminate_zeros()
        
        R2 = sparse.bmat([R2], format='csc')
        
        
        # Create sparse matrices (3 - component)
        R3 = [sparse.csc_matrix((RHS3.flatten(), (row3, col3)), shape=(n3[0]*n3[1]*n3[2], n3[0]*n3[1]*n3[2])) for RHS3, row3, col3 in zip(RHS3, row3, col3)]
                  
        R3[0].eliminate_zeros()
        R3[1].eliminate_zeros()
        R3[2].eliminate_zeros()
        
        R3 = sparse.bmat([R3], format='csc')
        
        return R1, R2, R3
      
    
    
    
    def projection_S(self, p0):
        '''
        Computes the right-hand sides for each basis function of the expression Pi_1(p0 * lambda^1)
        '''
        
        p1,    p2,   p3 = self.p         # spline degrees
        bc1,  bc2,  bc3 = self.bc        # boundary conditions
        nq1,  nq2,  nq3 = self.nq        # number of quadrature points per element
        w1,    w2,   w3 = self.weights   # quadrature weights
        
        
        
        # number of intervals for three components of the projector Pi1 (cyclic permutation of integration in i and interpolation in j and k)
        n1 = [self.n_grev[0] - 1 + bc1, self.n_grev[1], self.n_grev[2]]
        n2 = [self.n_grev[0], self.n_grev[1] - 1 + bc2, self.n_grev[2]]
        n3 = [self.n_grev[0], self.n_grev[1], self.n_grev[2] - 1 + bc3]
        
        
        # number of non-vanishing splines per interval (for even degree one additional spline per integration interval!)
        nv1 = [p1 + 1 - p1%2, p2 + 1, p3 + 1]
        nv2 = [p1 + 1, p2 + 1 - p2%2, p3 + 1]
        nv3 = [p1 + 1, p2 + 1, p3 + 1 - p3%2]
        
        
        # reshape greville points (n x 1)
        PP = [np.reshape(greville, (n_greville, 1)) for greville, n_greville in zip(self.greville, self.n_grev)]
        
        
        # Evaluate N - functions on interpolation points
        N_int = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, PP, 0, normalize=False)) for T, p, PP in zip(self.T, self.p, PP)]
        
        
        # Evaluate D - functions on quadrature points
        D_his = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p, quad, 0, normalize=True)) for t, p, quad in zip(self.t, self.pr, self.quad_grid)]
        
        
        # Evaluate equilibrium quantities on interpolation and quadrature points
        P11, P12, P13 = np.meshgrid(self.quad_grid[0].flatten(), self.greville[1], self.greville[2], indexing='ij')
        P21, P22, P23 = np.meshgrid(self.greville[0], self.quad_grid[1].flatten(), self.greville[2], indexing='ij')
        P31, P32, P33 = np.meshgrid(self.greville[0], self.greville[1], self.quad_grid[2].flatten(), indexing='ij')
        
        EQ1 = np.asfortranarray(p0(P11, P12, P13))
        EQ2 = np.asfortranarray(p0(P21, P22, P23))
        EQ3 = np.asfortranarray(p0(P31, P32, P33))
        
        
        # Local storage of right-hand-sides (for even degree one additional spline per integration interval!)
        RHS1 = np.zeros((n1[0], n1[1], n1[2], nv1[0], nv1[1], nv1[2]), order='F')
        RHS2 = np.zeros((n2[0], n2[1], n2[2], nv2[0], nv2[1], nv2[2]), order='F')
        RHS3 = np.zeros((n3[0], n3[1], n3[2], nv3[0], nv3[1], nv3[2]), order='F') 
        
        
        #==================shift in local indices for integrations============================================================
        il_add = []
               
        for mu in range(3):
               
            if self.bc[mu] == True:
                   
                if self.p[mu]%2 != 0:
                    il_add.append(np.array([0] * self.ne[mu]))
               
                else:
                    il_add.append(np.array([0, 1] * self.Nel[mu]))
                                     
            else:
                                     
                if self.p[mu]%2 != 0:
                    il_add.append(np.array([0] * self.ne[mu]))
                                     
                else:
                    boundaries = int(self.p[mu]/2)
                    il_add.append(np.array([0] * boundaries + [0, 1] * int((self.ne[0] - self.p[mu])/2) + [1] * boundaries))
        #=====================================================================================================================
        
        
        
        # Assembly of the first line
        kernels.kernel_pi1_1(self.ne[0], n1[1], n1[2], p1, p2 + 1, p3 + 1, self.ies[0], il_add[0], nq1, w1, D_his[0], N_int[1], N_int[2], EQ1, RHS1)
        
        
        # Assembly of the second line
        kernels.kernel_pi1_2(n2[0], self.ne[1], n2[2], p1 + 1, p2, p3 + 1, self.ies[1], il_add[1], nq2, w2, N_int[0], D_his[1], N_int[2], EQ2, RHS2)
        
        
        # Assembly of the third line
        kernels.kernel_pi1_3(n3[0], n3[1], self.ne[2], p1 + 1, p2 + 1, p3, self.ies[2], il_add[2], nq3, w3, N_int[0], N_int[1], D_his[2], EQ3, RHS3)
        
        
         
        # Grid indices               
        indices1 = np.indices((n1[0], n1[1], n1[2], nv1[0], nv1[1], nv1[2]))
        indices2 = np.indices((n2[0], n2[1], n2[2], nv2[0], nv2[1], nv2[2]))
        indices3 = np.indices((n3[0], n3[1], n3[2], nv3[0], nv3[1], nv3[2]))   
        
        
        # Row indices of global matrix
        row1 = (n1[1]*n1[2]*indices1[0] + n1[2]*indices1[1] + indices1[2]).flatten()
        row2 = (n2[1]*n2[2]*indices2[0] + n2[2]*indices2[1] + indices2[2]).flatten()
        row3 = (n3[1]*n3[2]*indices3[0] + n3[2]*indices3[1] + indices3[2]).flatten()
        
        
        # Column indices of global matrix in 1-direction
        if bc1 == True:
            
            col1_1 = (indices1[3] + np.arange(n1[0])[:, None, None, None, None, None])%n1[0]
            col1_2 = (indices2[3] + np.arange(n2[0])[:, None, None, None, None, None])%n2[0] 
            col1_3 = (indices3[3] + np.arange(n3[0])[:, None, None, None, None, None])%n3[0]
                                        
        else:
            
            print('not yet implemented!')
            
        
        # Column indices of global matrix in 2-direction
        if bc2 == True:
                                     
            col2_1 = (indices1[4] + np.arange(n1[1])[None, :, None, None, None, None])%n1[1]
            col2_2 = (indices2[4] + np.arange(n2[1])[None, :, None, None, None, None])%n2[1]
            col2_3 = (indices3[4] + np.arange(n3[1])[None, :, None, None, None, None])%n3[1]
               
        else:
            
            print('not yet implemented!')
            
            
        # Column indices of global matrix in 3-direction
        if bc3 == True:
                                     
            col3_1 = (indices1[5] + np.arange(n1[2])[None, None, :, None, None, None])%n1[2]
            col3_2 = (indices2[5] + np.arange(n2[2])[None, None, :, None, None, None])%n2[2]
            col3_3 = (indices3[5] + np.arange(n3[2])[None, None, :, None, None, None])%n3[2]
               
        else:
            
            print('not yet implemented!')
        
        col1 = (n1[1]*n1[2]*col1_1 + n1[2]*col2_1 + col3_1).flatten()
        col2 = (n2[1]*n2[2]*col1_2 + n2[2]*col2_2 + col3_2).flatten()
        col3 = (n3[1]*n3[2]*col1_3 + n3[2]*col2_3 + col3_3).flatten()
        
        
        # Create sparse matrix (1 - component)
        R1 = sparse.csc_matrix((RHS1.flatten(), (row1, col1)), shape=(n1[0]*n1[1]*n1[2], n1[0]*n1[1]*n1[2]))         
        R1.eliminate_zeros()
        
        # Create sparse matrix (2 - component)
        R2 = sparse.csc_matrix((RHS2.flatten(), (row2, col2)), shape=(n2[0]*n2[1]*n2[2], n2[0]*n2[1]*n2[2]))
        R2.eliminate_zeros()
        
        # Create sparse matrix (3 - component)
        R3 = sparse.csc_matrix((RHS3.flatten(), (row3, col3)), shape=(n3[0]*n3[1]*n3[2], n3[0]*n3[1]*n3[2]))
        R3.eliminate_zeros()
        
        
        return R1, R2, R3
    
    
    
    def projection_K(self, p0):
        '''
        Computes the right-hand sides for each basis function of the expression Pi_0(p0 * lambda^0)
        '''
        
        p1,    p2,   p3 = self.p         # spline degrees
        bc1,  bc2,  bc3 = self.bc        # boundary conditions
        nq1,  nq2,  nq3 = self.nq        # number of quadrature points per element
        w1,    w2,   w3 = self.weights   # quadrature weights
        
        
        
        # number of intervals for the projector Pi0 (pure interpolation)
        n = [self.n_grev[0], self.n_grev[1], self.n_grev[2]]
        
        # number of non-vanishing splines per interval
        nv = [p1 + 1, p2 + 1, p3 + 1]
        
        # reshape greville points (n x 1)
        PP = [np.reshape(greville, (n_greville, 1)) for greville, n_greville in zip(self.greville, self.n_grev)]
        
        
        # Evaluate N - functions on interpolation points
        N_int = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, PP, 0, normalize=False)) for T, p, PP in zip(self.T, self.p, PP)]
        
        # Evaluate equilibrium quantities on interpolation points
        P1, P2, P3 = np.meshgrid(self.greville[0].flatten(), self.greville[1], self.greville[2], indexing='ij')
        EQ = np.asfortranarray(p0(P1, P2, P3))
        
        # Local storage of right-hand-sides
        RHS = np.zeros((n[0], n[1], n[2], nv[0], nv[1], nv[2]), order='F')
        
        # Assembly
        kernels.kernel_pi0(n[0], n[1], n[2], p1 + 1, p2 + 1, p3 + 1, N_int[0], N_int[1], N_int[2], EQ, RHS)
        
         
        # Grid indices               
        indices = np.indices((n[0], n[1], n[2], nv[0], nv[1], nv[2])) 
        
        
        # Row indices of global matrix
        row = (n[1]*n[2]*indices[0] + n[2]*indices[1] + indices[2]).flatten()
        
        
        # Column indices of global matrix in 1-direction
        if bc1 == True:
            
            col1 = (indices[3] + np.arange(n[0])[:, None, None, None, None, None])%n[0]
                                        
        else:
            
            print('not yet implemented!')
            
        
        # Column indices of global matrix in 2-direction
        if bc2 == True:
                                     
            col2 = (indices[4] + np.arange(n[1])[None, :, None, None, None, None])%n[1]
               
        else:
            
            print('not yet implemented!')
            
            
        # Column indices of global matrix in 3-direction
        if bc3 == True:
                                     
            col3 = (indices[5] + np.arange(n[2])[None, None, :, None, None, None])%n[2]
               
        else:
            
            print('not yet implemented!')
        
        col = (n[1]*n[2]*col1 + n[2]*col2 + col3).flatten()
        
        
        # Create sparse matrix 
        R = sparse.csc_matrix((RHS.flatten(), (row, col)), shape=(n[0]*n[1]*n[2], n[0]*n[1]*n[2]))         
        R.eliminate_zeros()
        
        return R
    
    
    
    
    
    def projection_L(self, g_sqrt, G):
        
        p1,    p2,   p3 = self.p         # spline degrees
        bc1,  bc2,  bc3 = self.bc        # boundary conditions
        nq1,  nq2,  nq3 = self.nq        # number of quadrature points per element
        w1,    w2,   w3 = self.weights   # quadrature weights
        
        
        
        # number of intervals for three components of the projector Pi1 (cyclic permutation of integration in i and interpolation in j and k)
        n1 = [self.n_grev[0] - 1 + bc1, self.n_grev[1], self.n_grev[2]]
        n2 = [self.n_grev[0], self.n_grev[1] - 1 + bc2, self.n_grev[2]]
        n3 = [self.n_grev[0], self.n_grev[1], self.n_grev[2] - 1 + bc3]
        
         
        
        # number of non-vanishing splines per interval (for even degree one additional spline per integration interval!)
        nv1 = [[p1 + 2 - p1%2, p2, p3], [p1 + 1 - p1%2, p2 + 1, p3], [p1 + 1 - p1%2, p2, p3 + 1]]
        nv2 = [[p1 + 1, p2 + 1 - p2%2, p3], [p1, p2 + 2 - p2%2, p3], [p1, p2 + 1 - p2%2, p3 + 1]]
        nv3 = [[p1 + 1, p2, p3 + 1 - p3%2], [p1, p2 + 1, p3 + 1 - p3%2], [p1, p2, p3 + 2 - p3%2]]
        
        
        # reshape greville points (n x 1)
        PP = [np.reshape(greville, (n_greville, 1)) for greville, n_greville in zip(self.greville, self.n_grev)]
        
        
        # Evaluate N - functions on interpolation and quadrature points
        N_int = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, PP, 0, normalize=False)) for T, p, PP in zip(self.T, self.p, PP)]
        
        N_his = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad, 0, normalize=False)) for T, p, quad in zip(self.T, self.p, self.quad_grid)]
        
        
        # Evaluate D - functions on interpolation and quadrature points
        D_int = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p, PP, 0, normalize=True)) for t, p, PP in zip(self.t, self.pr, PP)] 
        
        D_his = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p, quad, 0, normalize=True)) for t, p, quad in zip(self.t, self.pr, self.quad_grid)]
        
        
        # Evaluate equilibrium quantities on interpolation and quadrature points
        P11, P12, P13 = np.meshgrid(self.quad_grid[0].flatten(), self.greville[1], self.greville[2], indexing='ij')
        P21, P22, P23 = np.meshgrid(self.greville[0], self.quad_grid[1].flatten(), self.greville[2], indexing='ij')
        P31, P32, P33 = np.meshgrid(self.greville[0], self.greville[1], self.quad_grid[2].flatten(), indexing='ij')
        
        EQ11 = np.asfortranarray(G[0][0](P11, P12, P13) / g_sqrt(P11, P12, P13))
        EQ12 = np.asfortranarray(G[0][1](P11, P12, P13) / g_sqrt(P11, P12, P13))
        EQ13 = np.asfortranarray(G[0][2](P11, P12, P13) / g_sqrt(P11, P12, P13))
        EQ1 = [EQ11, EQ12, EQ13]
        
        EQ21 = np.asfortranarray(G[1][0](P21, P22, P23) / g_sqrt(P21, P22, P23))
        EQ22 = np.asfortranarray(G[1][1](P21, P22, P23) / g_sqrt(P21, P22, P23))
        EQ23 = np.asfortranarray(G[1][2](P21, P22, P23) / g_sqrt(P21, P22, P23))
        EQ2 = [EQ21, EQ22, EQ23]
                                 
        
        EQ31 = np.asfortranarray(G[2][0](P31, P32, P33) / g_sqrt(P31, P32, P33))
        EQ32 = np.asfortranarray(G[2][1](P31, P32, P33) / g_sqrt(P31, P32, P33))
        EQ33 = np.asfortranarray(G[2][2](P31, P32, P33) / g_sqrt(P31, P32, P33))
        EQ3 = [EQ31, EQ32, EQ33]
        
        
        # Local storage of right-hand-sides (for even degree one additional spline per integration interval!)
        RHS1 = [np.zeros((n1[0], n1[1], n1[2], nv1[0], nv1[1], nv1[2]), order='F') for nv1 in nv1]
        RHS2 = [np.zeros((n2[0], n2[1], n2[2], nv2[0], nv2[1], nv2[2]), order='F') for nv2 in nv2]
        RHS3 = [np.zeros((n3[0], n3[1], n3[2], nv3[0], nv3[1], nv3[2]), order='F') for nv3 in nv3]
        
        
        
        #==================shift in local indices for integrations============================================================
        il_add = []
               
        for mu in range(3):
               
            if self.bc[mu] == True:
                   
                if self.p[mu]%2 != 0:
                    il_add.append(np.array([0] * self.ne[mu]))
               
                else:
                    il_add.append(np.array([0, 1] * self.Nel[mu]))
                                     
            else:
                                     
                if self.p[mu]%2 != 0:
                    il_add.append(np.array([0] * self.ne[mu]))
                                     
                else:
                    boundaries = int(self.p[mu]/2)
                    il_add.append(np.array([0] * boundaries + [0, 1] * int((self.ne[0] - self.p[mu])/2) + [1] * boundaries))
        #=====================================================================================================================
                       
        
        
        # Assembly of the first line
        kernels.kernel_pi1_1(self.ne[0], n1[1], n1[2], p1 + 1, p2, p3, self.ies[0], il_add[0], nq1, w1, N_his[0], D_int[1], D_int[2], EQ1[0], RHS1[0])
                                     
        kernels.kernel_pi1_1(self.ne[0], n1[1], n1[2], p1, p2 + 1, p3, self.ies[0], il_add[0], nq1, w1, D_his[0], N_int[1], D_int[2], EQ1[1], RHS1[1])
                                     
        kernels.kernel_pi1_1(self.ne[0], n1[1], n1[2], p1, p2, p3 + 1, self.ies[0], il_add[0], nq1, w1, D_his[0], D_int[1], N_int[2], EQ1[2], RHS1[2])
        
        
        # Assembly of the second line
        kernels.kernel_pi1_2(n2[0], self.ne[1], n2[2], p1 + 1, p2, p3, self.ies[1], il_add[1], nq2, w2, N_int[0], D_his[1], D_int[2], EQ2[0], RHS2[0])
                                     
        kernels.kernel_pi1_2(n2[0], self.ne[1], n2[2], p1, p2 + 1, p3, self.ies[1], il_add[1], nq2, w2, D_int[0], N_his[1], D_int[2], EQ2[1], RHS2[1])
                                     
        kernels.kernel_pi1_2(n2[0], self.ne[1], n2[2], p1, p2, p3 + 1, self.ies[1], il_add[1], nq2, w2, D_int[0], D_his[1], N_int[2], EQ2[2], RHS2[2])
        
        
        # Assembly of the third line
        kernels.kernel_pi1_3(n3[0], n3[1], self.ne[2], p1 + 1, p2, p3, self.ies[2], il_add[2], nq3, w3, N_int[0], D_int[1], D_his[2], EQ3[0], RHS3[0])
                                     
        kernels.kernel_pi1_3(n3[0], n3[1], self.ne[2], p1, p2 + 1, p3, self.ies[2], il_add[2], nq3, w3, D_int[0], N_int[1], D_his[2], EQ3[1], RHS3[1])
                                     
        kernels.kernel_pi1_3(n3[0], n3[1], self.ne[2], p1, p2, p3 + 1, self.ies[2], il_add[2], nq3, w3, D_int[0], D_int[1], N_his[2], EQ3[2], RHS3[2])
        
        
        # Grid indices               
        indices1 = [np.indices((n1[0], n1[1], n1[2], nv1[0], nv1[1], nv1[2])) for nv1 in nv1]
        indices2 = [np.indices((n2[0], n2[1], n2[2], nv2[0], nv2[1], nv2[2])) for nv2 in nv2]
        indices3 = [np.indices((n3[0], n3[1], n3[2], nv3[0], nv3[1], nv3[2])) for nv3 in nv3]                             
        
        
        
        # Row indices of global matrix
        row1 = [(n1[1]*n1[2]*indices1[0] + n1[2]*indices1[1] + indices1[2]).flatten() for indices1 in indices1]
        row2 = [(n2[1]*n2[2]*indices2[0] + n2[2]*indices2[1] + indices2[2]).flatten() for indices2 in indices2]
        row3 = [(n3[1]*n3[2]*indices3[0] + n3[2]*indices3[1] + indices3[2]).flatten() for indices3 in indices3]
        
        
        # Column indices of global matrix in 1-direction
        if bc1 == True:
            
            col1_1 = [(indices1[3] + np.arange(n1[0])[:, None, None, None, None, None])%n1[0] for indices1 in indices1]
            col1_2 = [(indices2[3] + np.arange(n2[0])[:, None, None, None, None, None])%n2[0] for indices2 in indices2]
            col1_3 = [(indices3[3] + np.arange(n3[0])[:, None, None, None, None, None])%n3[0] for indices3 in indices3]
                                        
        else:
            
            print('not yet implemented!')
            
        
        # Column indices of global matrix in 2-direction
        if bc2 == True:
                                     
            col2_1 = [(indices1[4] + np.arange(n1[1])[None, :, None, None, None, None])%n1[1] for indices1 in indices1]
            col2_2 = [(indices2[4] + np.arange(n2[1])[None, :, None, None, None, None])%n2[1] for indices2 in indices2]
            col2_3 = [(indices3[4] + np.arange(n3[1])[None, :, None, None, None, None])%n3[1] for indices3 in indices3]
               
        else:
            
            print('not yet implemented!')
            
            
        # Column indices of global matrix in 3-direction
        if bc3 == True:
                                     
            col3_1 = [(indices1[5] + np.arange(n1[2])[None, None, :, None, None, None])%n1[2] for indices1 in indices1]
            col3_2 = [(indices2[5] + np.arange(n2[2])[None, None, :, None, None, None])%n2[2] for indices2 in indices2]
            col3_3 = [(indices3[5] + np.arange(n3[2])[None, None, :, None, None, None])%n3[2] for indices3 in indices3]
               
        else:
            
            print('not yet implemented!')
        
        col1 = [(n1[1]*n1[2]*col1_1 + n1[2]*col2_1 + col3_1).flatten() for col1_1, col2_1, col3_1 in zip(col1_1, col2_1, col3_1)]
        col2 = [(n2[1]*n2[2]*col1_2 + n2[2]*col2_2 + col3_2).flatten() for col1_2, col2_2, col3_2 in zip(col1_2, col2_2, col3_2)]
        col3 = [(n3[1]*n3[2]*col1_3 + n3[2]*col2_3 + col3_3).flatten() for col1_3, col2_3, col3_3 in zip(col1_3, col2_3, col3_3)]
        

        # Create sparse matrices (1 - component)
        R1 = [sparse.csc_matrix((RHS1.flatten(), (row1, col1)), shape=(n1[0]*n1[1]*n1[2], n1[0]*n1[1]*n1[2])) for RHS1, row1, col1 in zip(RHS1, row1, col1)]
                  
        R1[0].eliminate_zeros()
        R1[1].eliminate_zeros()
        R1[2].eliminate_zeros()
        
        R1 = sparse.bmat([R1], format='csc')
        
        
        # Create sparse matrices (2 - component)
        R2 = [sparse.csc_matrix((RHS2.flatten(), (row2, col2)), shape=(n2[0]*n2[1]*n2[2], n2[0]*n2[1]*n2[2])) for RHS2, row2, col2 in zip(RHS2, row2, col2)]
                  
        R2[0].eliminate_zeros()
        R2[1].eliminate_zeros()
        R2[2].eliminate_zeros()
        
        R2 = sparse.bmat([R2], format='csc')
        
        
        # Create sparse matrices (3 - component)
        R3 = [sparse.csc_matrix((RHS3.flatten(), (row3, col3)), shape=(n3[0]*n3[1]*n3[2], n3[0]*n3[1]*n3[2])) for RHS3, row3, col3 in zip(RHS3, row3, col3)]
                  
        R3[0].eliminate_zeros()
        R3[1].eliminate_zeros()
        R3[2].eliminate_zeros()
        
        R3 = sparse.bmat([R3], format='csc')
        
        return R1, R2, R3
    
    
    
    def projection_Y(self, g_sqrt, Ginv):
        
        p1,    p2,   p3 = self.p         # spline degrees
        bc1,  bc2,  bc3 = self.bc        # boundary conditions
        nq1,  nq2,  nq3 = self.nq        # number of quadrature points per element
        w1,    w2,   w3 = self.weights   # quadrature weights
        
        
        
        # number of intervals for three components of the projector Pi1 (cyclic permutation of integration in i and interpolation in j and k)
        n1 = [self.n_grev[0] - 1 + bc1, self.n_grev[1], self.n_grev[2]]
        n2 = [self.n_grev[0], self.n_grev[1] - 1 + bc2, self.n_grev[2]]
        n3 = [self.n_grev[0], self.n_grev[1], self.n_grev[2] - 1 + bc3]
        
        
        
        
        
        # number of non-vanishing splines per interval (for even degree one additional spline per integration interval!)
        nv1 = [[p1 + 1 - p1%2, p2 + 1, p3 + 1], [p1 + 2 - p1%2, p2, p3 + 1], [p1 + 2 - p1%2, p2 + 1, p3]]
        nv2 = [[p1, p2 + 2 - p2%2, p3 + 1], [p1 + 1, p2 + 1 - p2%2, p3 + 1], [p1 + 1, p2 + 2 - p2%2, p3]]
        nv3 = [[p1, p2 + 1, p3 + 2 - p3%2], [p1 + 1, p2, p3 + 2 - p3%2], [p1 + 1, p2 + 1, p3 + 1 - p3%2]]
        
        
        # reshape greville points (n x 1)
        PP = [np.reshape(greville, (n_greville, 1)) for greville, n_greville in zip(self.greville, self.n_grev)]
        
        
        # Evaluate N - functions on interpolation and quadrature points
        N_int = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, PP, 0, normalize=False)) for T, p, PP in zip(self.T, self.p, PP)]
        
        N_his = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad, 0, normalize=False)) for T, p, quad in zip(self.T, self.p, self.quad_grid)]
        
        
        # Evaluate D - functions on interpolation and quadrature points
        D_int = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p, PP, 0, normalize=True)) for t, p, PP in zip(self.t, self.pr, PP)] 
        
        D_his = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p, quad, 0, normalize=True)) for t, p, quad in zip(self.t, self.pr, self.quad_grid)]
        
        
        # Evaluate equilibrium quantities on interpolation and quadrature points
        P11, P12, P13 = np.meshgrid(self.quad_grid[0].flatten(), self.greville[1], self.greville[2], indexing='ij')
        P21, P22, P23 = np.meshgrid(self.greville[0], self.quad_grid[1].flatten(), self.greville[2], indexing='ij')
        P31, P32, P33 = np.meshgrid(self.greville[0], self.greville[1], self.quad_grid[2].flatten(), indexing='ij')
        
        EQ11 = np.asfortranarray(Ginv[0][0](P11, P12, P13)*g_sqrt(P11, P12, P13))
        EQ12 = np.asfortranarray(Ginv[0][1](P11, P12, P13)*g_sqrt(P11, P12, P13))
        EQ13 = np.asfortranarray(Ginv[0][2](P11, P12, P13)*g_sqrt(P11, P12, P13))
        EQ1 = [EQ11, EQ12, EQ13]
        
        EQ21 = np.asfortranarray(Ginv[1][0](P21, P22, P23)*g_sqrt(P21, P22, P23))
        EQ22 = np.asfortranarray(Ginv[1][1](P21, P22, P23)*g_sqrt(P21, P22, P23))
        EQ23 = np.asfortranarray(Ginv[1][2](P21, P22, P23)*g_sqrt(P21, P22, P23))
        EQ2 = [EQ21, EQ22, EQ23]
                                 
        
        EQ31 = np.asfortranarray(Ginv[2][0](P31, P32, P33)*g_sqrt(P31, P32, P33))
        EQ32 = np.asfortranarray(Ginv[2][1](P31, P32, P33)*g_sqrt(P31, P32, P33))
        EQ33 = np.asfortranarray(Ginv[2][2](P31, P32, P33)*g_sqrt(P31, P32, P33))
        EQ3 = [EQ31, EQ32, EQ33]
        
        
        # Local storage of right-hand-sides (for even degree one additional spline per integration interval!)
        RHS1 = [np.zeros((n1[0], n1[1], n1[2], nv1[0], nv1[1], nv1[2]), order='F') for nv1 in nv1]
        RHS2 = [np.zeros((n2[0], n2[1], n2[2], nv2[0], nv2[1], nv2[2]), order='F') for nv2 in nv2]
        RHS3 = [np.zeros((n3[0], n3[1], n3[2], nv3[0], nv3[1], nv3[2]), order='F') for nv3 in nv3]
        
        
        
        #==================shift in local indices for integrations============================================================
        il_add = []
               
        for mu in range(3):
               
            if self.bc[mu] == True:
                   
                if self.p[mu]%2 != 0:
                    il_add.append(np.array([0] * self.ne[mu]))
               
                else:
                    il_add.append(np.array([0, 1] * self.Nel[mu]))
                                     
            else:
                                     
                if self.p[mu]%2 != 0:
                    il_add.append(np.array([0] * self.ne[mu]))
                                     
                else:
                    boundaries = int(self.p[mu]/2)
                    il_add.append(np.array([0] * boundaries + [0, 1] * int((self.ne[0] - self.p[mu])/2) + [1] * boundaries))
        #=====================================================================================================================
                       
        
        
        # Assembly of the first line
        kernels.kernel_pi1_1(self.ne[0], n1[1], n1[2], p1, p2 + 1, p3 + 1, self.ies[0], il_add[0], nq1, w1, D_his[0], N_int[1], N_int[2], EQ1[0], RHS1[0])
                                     
        kernels.kernel_pi1_1(self.ne[0], n1[1], n1[2], p1 + 1, p2, p3 + 1, self.ies[0], il_add[0], nq1, w1, N_his[0], D_int[1], N_int[2], EQ1[1], RHS1[1])
                                     
        kernels.kernel_pi1_1(self.ne[0], n1[1], n1[2], p1 + 1, p2 + 1, p3, self.ies[0], il_add[0], nq1, w1, N_his[0], N_int[1], D_int[2], EQ1[2], RHS1[2])
        
        
        # Assembly of the second line
        kernels.kernel_pi1_2(n2[0], self.ne[1], n2[2], p1, p2 + 1, p3 + 1, self.ies[1], il_add[1], nq2, w2, D_int[0], N_his[1], N_int[2], EQ2[0], RHS2[0])
                                     
        kernels.kernel_pi1_2(n2[0], self.ne[1], n2[2], p1 + 1, p2, p3 + 1, self.ies[1], il_add[1], nq2, w2, N_int[0], D_his[1], N_int[2], EQ2[1], RHS2[1])
                                     
        kernels.kernel_pi1_2(n2[0], self.ne[1], n2[2], p1 + 1, p2 + 1, p3, self.ies[1], il_add[1], nq2, w2, N_int[0], N_his[1], D_int[2], EQ2[2], RHS2[2])
        
        
        # Assembly of the third line
        kernels.kernel_pi1_3(n3[0], n3[1], self.ne[2], p1, p2 + 1, p3 + 1, self.ies[2], il_add[2], nq3, w3, D_int[0], N_int[1], N_his[2], EQ3[0], RHS3[0])
                                     
        kernels.kernel_pi1_3(n3[0], n3[1], self.ne[2], p1 + 1, p2, p3 + 1, self.ies[2], il_add[2], nq3, w3, N_int[0], D_int[1], N_his[2], EQ3[1], RHS3[1])
                                     
        kernels.kernel_pi1_3(n3[0], n3[1], self.ne[2], p1 + 1, p2 + 1, p3, self.ies[2], il_add[2], nq3, w3, N_int[0], N_int[1], D_his[2], EQ3[2], RHS3[2])
        
        
        # Grid indices               
        indices1 = [np.indices((n1[0], n1[1], n1[2], nv1[0], nv1[1], nv1[2])) for nv1 in nv1]
        indices2 = [np.indices((n2[0], n2[1], n2[2], nv2[0], nv2[1], nv2[2])) for nv2 in nv2]
        indices3 = [np.indices((n3[0], n3[1], n3[2], nv3[0], nv3[1], nv3[2])) for nv3 in nv3]                             
        
        
        
        # Row indices of global matrix
        row1 = [(n1[1]*n1[2]*indices1[0] + n1[2]*indices1[1] + indices1[2]).flatten() for indices1 in indices1]
        row2 = [(n2[1]*n2[2]*indices2[0] + n2[2]*indices2[1] + indices2[2]).flatten() for indices2 in indices2]
        row3 = [(n3[1]*n3[2]*indices3[0] + n3[2]*indices3[1] + indices3[2]).flatten() for indices3 in indices3]
        
        
        # Column indices of global matrix in 1-direction
        if bc1 == True:
            
            col1_1 = [(indices1[3] + np.arange(n1[0])[:, None, None, None, None, None])%n1[0] for indices1 in indices1]
            col1_2 = [(indices2[3] + np.arange(n2[0])[:, None, None, None, None, None])%n2[0] for indices2 in indices2]
            col1_3 = [(indices3[3] + np.arange(n3[0])[:, None, None, None, None, None])%n3[0] for indices3 in indices3]
                                        
        else:
            
            print('not yet implemented!')
            
        
        # Column indices of global matrix in 2-direction
        if bc2 == True:
                                     
            col2_1 = [(indices1[4] + np.arange(n1[1])[None, :, None, None, None, None])%n1[1] for indices1 in indices1]
            col2_2 = [(indices2[4] + np.arange(n2[1])[None, :, None, None, None, None])%n2[1] for indices2 in indices2]
            col2_3 = [(indices3[4] + np.arange(n3[1])[None, :, None, None, None, None])%n3[1] for indices3 in indices3]
               
        else:
            
            print('not yet implemented!')
            
            
        # Column indices of global matrix in 3-direction
        if bc3 == True:
                                     
            col3_1 = [(indices1[5] + np.arange(n1[2])[None, None, :, None, None, None])%n1[2] for indices1 in indices1]
            col3_2 = [(indices2[5] + np.arange(n2[2])[None, None, :, None, None, None])%n2[2] for indices2 in indices2]
            col3_3 = [(indices3[5] + np.arange(n3[2])[None, None, :, None, None, None])%n3[2] for indices3 in indices3]
               
        else:
            
            print('not yet implemented!')
        
        col1 = [(n1[1]*n1[2]*col1_1 + n1[2]*col2_1 + col3_1).flatten() for col1_1, col2_1, col3_1 in zip(col1_1, col2_1, col3_1)]
        col2 = [(n2[1]*n2[2]*col1_2 + n2[2]*col2_2 + col3_2).flatten() for col1_2, col2_2, col3_2 in zip(col1_2, col2_2, col3_2)]
        col3 = [(n3[1]*n3[2]*col1_3 + n3[2]*col2_3 + col3_3).flatten() for col1_3, col2_3, col3_3 in zip(col1_3, col2_3, col3_3)]
        

        # Create sparse matrices (1 - component)
        R1 = [sparse.csc_matrix((RHS1.flatten(), (row1, col1)), shape=(n1[0]*n1[1]*n1[2], n1[0]*n1[1]*n1[2])) for RHS1, row1, col1 in zip(RHS1, row1, col1)]
                  
        R1[0].eliminate_zeros()
        R1[1].eliminate_zeros()
        R1[2].eliminate_zeros()
        
        R1 = sparse.bmat([R1], format='csc')
        
        
        # Create sparse matrices (2 - component)
        R2 = [sparse.csc_matrix((RHS2.flatten(), (row2, col2)), shape=(n2[0]*n2[1]*n2[2], n2[0]*n2[1]*n2[2])) for RHS2, row2, col2 in zip(RHS2, row2, col2)]
                  
        R2[0].eliminate_zeros()
        R2[1].eliminate_zeros()
        R2[2].eliminate_zeros()
        
        R2 = sparse.bmat([R2], format='csc')
        
        
        # Create sparse matrices (3 - component)
        R3 = [sparse.csc_matrix((RHS3.flatten(), (row3, col3)), shape=(n3[0]*n3[1]*n3[2], n3[0]*n3[1]*n3[2])) for RHS3, row3, col3 in zip(RHS3, row3, col3)]
                  
        R3[0].eliminate_zeros()
        R3[1].eliminate_zeros()
        R3[2].eliminate_zeros()
        
        R3 = sparse.bmat([R3], format='csc')
        
        return R1, R2, R3 
    
    
    
    
    def projection_T_old(self, B0, Ginv):
        
        p1, p2, p3 = self.p
        bc_1, bc_2, bc_3 = self.bc
        
        nq1, nq2, nq3 = self.nq
        w1, w2, w3 = self.weights
        
        
        n11, n12, n13 = self.n_grev[0] - 1 + bc_1, self.n_grev[1], self.n_grev[2]
        n21, n22, n23 = self.n_grev[0], self.n_grev[1] - 1 + bc_2, self.n_grev[2]
        n31, n32, n33 = self.n_grev[0], self.n_grev[1], self.n_grev[2] - 1 + bc_3
        
        
        PP1 = np.reshape(self.greville[0], (self.n_grev[0], 1))
        PP2 = np.reshape(self.greville[1], (self.n_grev[1], 1))
        PP3 = np.reshape(self.greville[2], (self.n_grev[2], 1))
        
        
        # Evaluate N - functions at interpolation and quadrature points
        bs0_1_int = np.asfortranarray(bsp.basis_ders_on_quad_grid(self.T[0], p1, PP1, 0, normalize=False))
        bs0_2_int = np.asfortranarray(bsp.basis_ders_on_quad_grid(self.T[1], p2, PP2, 0, normalize=False))
        bs0_3_int = np.asfortranarray(bsp.basis_ders_on_quad_grid(self.T[2], p3, PP3, 0, normalize=False))
        
        bs0_1_his = np.asfortranarray(bsp.basis_ders_on_quad_grid(self.T[0], p1, self.quad_grid[0], 0, normalize=False))
        bs0_2_his = np.asfortranarray(bsp.basis_ders_on_quad_grid(self.T[1], p2, self.quad_grid[1], 0, normalize=False))
        bs0_3_his = np.asfortranarray(bsp.basis_ders_on_quad_grid(self.T[2], p3, self.quad_grid[2], 0, normalize=False))
        
        
        # Evaluate D - functions at interpolation and quadrature points
        bs1_1_int = np.asfortranarray(bsp.basis_ders_on_quad_grid(self.t[0], p1 - 1, PP1, 0, normalize=True))
        bs1_2_int = np.asfortranarray(bsp.basis_ders_on_quad_grid(self.t[1], p2 - 1, PP2, 0, normalize=True))
        bs1_3_int = np.asfortranarray(bsp.basis_ders_on_quad_grid(self.t[2], p3 - 1, PP3, 0, normalize=True))
        
        bs1_1_his = np.asfortranarray(bsp.basis_ders_on_quad_grid(self.t[0], p1 - 1, self.quad_grid[0], 0, normalize=True))
        bs1_2_his = np.asfortranarray(bsp.basis_ders_on_quad_grid(self.t[1], p2 - 1, self.quad_grid[1], 0, normalize=True))
        bs1_3_his = np.asfortranarray(bsp.basis_ders_on_quad_grid(self.t[2], p3 - 1, self.quad_grid[2], 0, normalize=True))
        
        
        RHS_11 = np.zeros((n11, n12, n13, p1 - (p1%2 - 1), p2 + 1, p3 + 1), order='F')
        RHS_12 = np.zeros((n11, n12, n13, p1 + 1 - (p1%2 - 1), p2, p3 + 1), order='F')
        RHS_13 = np.zeros((n11, n12, n13, p1 + 1 - (p1%2 - 1), p2 + 1, p3), order='F')
        
        RHS_21 = np.zeros((n21, n22, n23, p1, p2 + 1 - (p2%2 - 1), p3 + 1), order='F')
        RHS_22 = np.zeros((n21, n22, n23, p1 + 1, p2 - (p2%2 - 1), p3 + 1), order='F')
        RHS_23 = np.zeros((n21, n22, n23, p1 + 1, p2 + 1 - (p2%2 - 1), p3), order='F')
        
        RHS_31 = np.zeros((n31, n32, n33, p1, p2 + 1, p3 + 1 - (p3%2 - 1)), order='F')
        RHS_32 = np.zeros((n31, n32, n33, p1 + 1, p2, p3 + 1 - (p3%2 - 1)), order='F')
        RHS_33 = np.zeros((n31, n32, n33, p1 + 1, p2 + 1, p3 - (p3%2 - 1)), order='F')
         
        
        EQ_11, EQ_12, EQ_13 = np.asfortranarray(self.B[0][0]), np.asfortranarray(self.B[0][1]), np.asfortranarray(self.B[0][2])
        EQ_21, EQ_22, EQ_23 = np.asfortranarray(self.B[1][0]), np.asfortranarray(self.B[1][1]), np.asfortranarray(self.B[1][2])
        EQ_31, EQ_32, EQ_33 = np.asfortranarray(self.B[2][0]), np.asfortranarray(self.B[2][1]), np.asfortranarray(self.B[2][2])
        
        
        # Shift in local indices in 1-direction for histopolation
        if bc_1 == True:
            
            if p1%2 != 0:
                
                il_add_1 = np.array([0] * self.ne[0])
                
            else:
                
                il_add_1 = np.array([0, 1] * self.Nel[0])
                
        else:
            
            if p1%2 != 0:
                
                il_add_1 = np.array([0] * self.ne[0])
                
            else:
                
                boundaries = int(p1/2)
                il_add_1 = np.array([0] * boundaries + [0, 1] * int((self.ne[0] - p1)/2) + [1] * boundaries)
                
        # Shift in local indices in 2-direction
        if bc_2 == True:
            
            if p2%2 != 0:
                
                il_add_2 = np.array([0] * self.ne[1])
                
            else:
                
                il_add_2 = np.array([0, 1] * self.Nel[1])
                
        else:
            
            if p2%2 != 0:
                
                il_add_2 = np.array([0] * self.ne[1])
                
            else:
                
                boundaries = int(p2/2)
                il_add_2 = np.array([0] * boundaries + [0, 1] * int((self.ne[1] - p2)/2) + [1] * boundaries)
                
        # Shift in local indices in 3-direction
        if bc_3 == True:
            
            if p3%2 != 0:
                
                il_add_3 = np.array([0] * self.ne[2])
                
            else:
                
                il_add_3 = np.array([0, 1] * self.Nel[2])
                
        else:
            
            if p3%2 != 0:
                
                il_add_3 = np.array([0] * self.ne[2])
                
            else:
                
                boundaries = int(p3/2)
                il_add_3 = np.array([0] * boundaries + [0, 1] * int((self.ne[2] - p3)/2) + [1] * boundaries)
                       
        
        
        # Assembly
        kernels.kernel_pi1_1(self.ne[0], n12, n13, p1, p2 + 1, p3 + 1, self.ies[0], il_add_1, nq1, w1, bs1_1_his, bs0_2_int, bs0_3_int, EQ_11, RHS_11)
        kernels.kernel_pi1_1(self.ne[0], n12, n13, p1 + 1, p2, p3 + 1, self.ies[0], il_add_1, nq1, w1, bs0_1_his, bs1_2_int, bs0_3_int, EQ_12, RHS_12)
        kernels.kernel_pi1_1(self.ne[0], n12, n13, p1 + 1, p2 + 1, p3, self.ies[0], il_add_1, nq1, w1, bs0_1_his, bs0_2_int, bs1_3_int, EQ_13, RHS_13)
        
        
        kernels.kernel_pi1_2(n21, self.ne[1], n23, p1, p2 + 1, p3 + 1, self.ies[1], il_add_2, nq2, w2, bs1_1_int, bs0_2_his, bs0_3_int, EQ_21, RHS_21)
        kernels.kernel_pi1_2(n21, self.ne[1], n23, p1 + 1, p2, p3 + 1, self.ies[1], il_add_2, nq2, w2, bs0_1_int, bs1_2_his, bs0_3_int, EQ_22, RHS_22)
        kernels.kernel_pi1_2(n21, self.ne[1], n23, p1 + 1, p2 + 1, p3, self.ies[1], il_add_2, nq2, w2, bs0_1_int, bs0_2_his, bs1_3_int, EQ_23, RHS_23)
        
        
        kernels.kernel_pi1_3(n31, n32, self.ne[2], p1, p2 + 1, p3 + 1, self.ies[2], il_add_3, nq3, w3, bs1_1_int, bs0_2_int, bs0_3_his, EQ_31, RHS_31)
        kernels.kernel_pi1_3(n31, n32, self.ne[2], p1 + 1, p2, p3 + 1, self.ies[2], il_add_3, nq3, w3, bs0_1_int, bs1_2_int, bs0_3_his, EQ_32, RHS_32)
        kernels.kernel_pi1_3(n31, n32, self.ne[2], p1 + 1, p2 + 1, p3, self.ies[2], il_add_3, nq3, w3, bs0_1_int, bs0_2_int, bs1_3_his, EQ_33, RHS_33)
        
        
        # Grid indices
        indices_11 = np.indices((n11, n12, n13, p1 - (p1%2 - 1), p2 + 1, p3 + 1))
        indices_12 = np.indices((n11, n12, n13, p1 + 1 - (p1%2 - 1), p2, p3 + 1))
        indices_13 = np.indices((n11, n12, n13, p1 + 1 - (p1%2 - 1), p2 + 1, p3))
        
        indices_21 = np.indices((n21, n22, n23, p1, p2 + 1 - (p2%2 - 1), p3 + 1))
        indices_22 = np.indices((n21, n22, n23, p1 + 1, p2 - (p2%2 - 1), p3 + 1))
        indices_23 = np.indices((n21, n22, n23, p1 + 1, p2 + 1 - (p2%2 - 1), p3))
        
        indices_31 = np.indices((n31, n32, n33, p1, p2 + 1, p3 + 1 - (p3%2 - 1)))
        indices_32 = np.indices((n31, n32, n33, p1 + 1, p2, p3 + 1 - (p3%2 - 1)))
        indices_33 = np.indices((n31, n32, n33, p1 + 1, p2 + 1, p3 - (p3%2 - 1)))
        
        
        # Row indices
        row_11 = (n12*n13*indices_11[0] + n13*indices_11[1] + indices_11[2]).flatten()
        row_12 = (n12*n13*indices_12[0] + n13*indices_12[1] + indices_12[2]).flatten()
        row_13 = (n12*n13*indices_13[0] + n13*indices_13[1] + indices_13[2]).flatten()
        
        row_21 = (n22*n23*indices_21[0] + n23*indices_21[1] + indices_21[2]).flatten()
        row_22 = (n22*n23*indices_22[0] + n23*indices_22[1] + indices_22[2]).flatten()
        row_23 = (n22*n23*indices_23[0] + n23*indices_23[1] + indices_23[2]).flatten()
        
        row_31 = (n32*n33*indices_31[0] + n33*indices_31[1] + indices_31[2]).flatten()
        row_32 = (n32*n33*indices_32[0] + n33*indices_32[1] + indices_32[2]).flatten()
        row_33 = (n32*n33*indices_33[0] + n33*indices_33[1] + indices_33[2]).flatten()
        
        
        
        # Column indices in 1-direction
        if bc_1 == True:
            
            col_1_11 = (indices_11[3] + np.arange(n11)[:, None, None, None, None, None])%n11
            col_1_12 = (indices_12[3] + np.arange(n11)[:, None, None, None, None, None])%n11
            col_1_13 = (indices_13[3] + np.arange(n11)[:, None, None, None, None, None])%n11
            
            col_1_21 = (indices_21[3] + np.arange(n21)[:, None, None, None, None, None])%n21
            col_1_22 = (indices_22[3] + np.arange(n21)[:, None, None, None, None, None])%n21
            col_1_23 = (indices_23[3] + np.arange(n21)[:, None, None, None, None, None])%n21
            
            col_1_31 = (indices_31[3] + np.arange(n31)[:, None, None, None, None, None])%n31
            col_1_32 = (indices_32[3] + np.arange(n31)[:, None, None, None, None, None])%n31
            col_1_33 = (indices_33[3] + np.arange(n31)[:, None, None, None, None, None])%n31
            
        else:
            
            if p1%2 != 0:
            
                ind_11 = list(np.arange(n11 - 2*(p1 - 1)) + 1)
                ind_21 = list(np.arange(n21 - 1 - 2*(p1 - 1)) + 1)
                ind_31 = list(np.arange(n31 - 1 - 2*(p1 - 1)) + 1)

                col_11 = indices_11[3] + np.array([0] * 2 + ind_11 + [ind_11[-1] + 1] * 2)[:, None, None, None, None, None]
                col_21 = indices_21[3] + np.array([0] * 2 + ind_21 + [ind_21[-1] + 1] * 3)[:, None, None, None, None, None]
                col_31 = indices_31[3] + np.array([0] * 2 + ind_31 + [ind_31[-1] + 1] * 3)[:, None, None, None, None, None]
                
            else:
                
                ind_11 = list(np.arange(n11 - 2*(int(p1/2) + 1)) + 1)
                ind_21 = list(np.arange(n21 - 2*(int(p1/2) + 1)) + 1)
                ind_31 = list(np.arange(n31 - 2*(int(p1/2) + 1)) + 1)
                
                col_11 = indices_11[3] + np.array([0] * (int(p1/2) + 1) + ind_11 + [ind_11[-1] + 1] * (int(p1/2) + 1))[:, None, None, None, None, None]
                col_21 = indices_21[3] + np.array([0] * (int(p1/2) + 1) + ind_21 + [ind_21[-1] + 1] * (int(p1/2) + 1))[:, None, None, None, None, None]
                col_31 = indices_31[3] + np.array([0] * (int(p1/2) + 1) + ind_31 + [ind_31[-1] + 1] * (int(p1/2) + 1))[:, None, None, None, None, None]
            
        
        
        
        # Column indices in 2-direction
        if bc_2 == True:
            
            col_2_11 = (indices_11[4] + np.arange(n12)[None, :, None, None, None, None])%n12
            col_2_12 = (indices_12[4] + np.arange(n12)[None, :, None, None, None, None])%n12
            col_2_13 = (indices_13[4] + np.arange(n12)[None, :, None, None, None, None])%n12
            
            col_2_21 = (indices_21[4] + np.arange(n22)[None, :, None, None, None, None])%n22
            col_2_22 = (indices_22[4] + np.arange(n22)[None, :, None, None, None, None])%n22
            col_2_23 = (indices_23[4] + np.arange(n22)[None, :, None, None, None, None])%n22
            
            col_2_31 = (indices_31[4] + np.arange(n32)[None, :, None, None, None, None])%n32
            col_2_32 = (indices_32[4] + np.arange(n32)[None, :, None, None, None, None])%n32
            col_2_33 = (indices_33[4] + np.arange(n32)[None, :, None, None, None, None])%n32
            
        else:
            
            if p2%2 != 0:
            
                ind_12 = list(np.arange(n12 - 1 - 2*(p2 - 1)) + 1)
                ind_22 = list(np.arange(n22 - 2*(p2 - 1)) + 1)
                ind_32 = list(np.arange(n32 - 1 - 2*(p2 - 1)) + 1)

                col_12 = indices_22[4] + np.array([0] * 2 + ind_12 + [ind_12[-1] + 1] * 3)[None, :, None, None, None, None]
                col_22 = indices_22[4] + np.array([0] * 2 + ind_22 + [ind_22[-1] + 1] * 2)[None, :, None, None, None, None]
                col_32 = indices_32[4] + np.array([0] * 2 + ind_32 + [ind_32[-1] + 1] * 3)[None, :, None, None, None, None]
                
            else:
                
                ind_12 = list(np.arange(n12 - 2*(int(p2/2) + 1)) + 1)
                ind_22 = list(np.arange(n22 - 2*(int(p2/2) + 1)) + 1)
                ind_32 = list(np.arange(n32 - 2*(int(p2/2) + 1)) + 1)
                
                col_12 = indices_12[4] + np.array([0] * (int(p2/2) + 1) + ind_12 + [ind_12[-1] + 1] * (int(p2/2) + 1))[None, :, None, None, None, None]
                col_22 = indices_22[4] + np.array([0] * (int(p2/2) + 1) + ind_22 + [ind_22[-1] + 1] * (int(p2/2) + 1))[None, :, None, None, None, None]
                col_32 = indices_32[4] + np.array([0] * (int(p2/2) + 1) + ind_32 + [ind_32[-1] + 1] * (int(p2/2) + 1))[None, :, None, None, None, None]
            
            
        # Column indices in 3-direction
        if bc_3 == True:
            
            col_3_11 = (indices_11[5] + np.arange(n13)[None, None, :, None, None, None])%n13
            col_3_12 = (indices_12[5] + np.arange(n13)[None, None, :, None, None, None])%n13
            col_3_13 = (indices_13[5] + np.arange(n13)[None, None, :, None, None, None])%n13
            
            col_3_21 = (indices_21[5] + np.arange(n23)[None, None, :, None, None, None])%n23
            col_3_22 = (indices_22[5] + np.arange(n23)[None, None, :, None, None, None])%n23
            col_3_23 = (indices_23[5] + np.arange(n23)[None, None, :, None, None, None])%n23
            
            col_3_31 = (indices_31[5] + np.arange(n33)[None, None, :, None, None, None])%n33
            col_3_32 = (indices_32[5] + np.arange(n33)[None, None, :, None, None, None])%n33
            col_3_33 = (indices_33[5] + np.arange(n33)[None, None, :, None, None, None])%n33
            
        else:
            
            if p3%2 != 0:
            
                ind_13 = list(np.arange(n13 - 1 - 2*(p3 - 1)) + 1)
                ind_23 = list(np.arange(n23 - 1 - 2*(p3 - 1)) + 1)
                ind_33 = list(np.arange(n33 - 2*(p3 - 1)) + 1)

                col_13 = indices_13[5] + np.array([0] * 2 + ind_13 + [ind_13[-1] + 1] * 3)[None, None, :, None, None, None]
                col_23 = indices_23[5] + np.array([0] * 2 + ind_23 + [ind_23[-1] + 1] * 3)[None, None, :, None, None, None]
                col_33 = indices_33[5] + np.array([0] * 2 + ind_33 + [ind_33[-1] + 1] * 2)[None, None, :, None, None, None]
                
            else:
                
                ind_13 = list(np.arange(n13 - 2*(int(p3/2) + 1)) + 1)
                ind_23 = list(np.arange(n23 - 2*(int(p3/2) + 1)) + 1)
                ind_33 = list(np.arange(n33 - 2*(int(p3/2) + 1)) + 1)
                
                col_13 = indices_13[5] + np.array([0] * (int(p3/2) + 1) + ind_13 + [ind_13[-1] + 1] * (int(p3/2) + 1))[None, None, :, None, None, None]
                col_23 = indices_23[5] + np.array([0] * (int(p3/2) + 1) + ind_23 + [ind_23[-1] + 1] * (int(p3/2) + 1))[None, None, :, None, None, None]
                col_33 = indices_33[5] + np.array([0] * (int(p3/2) + 1) + ind_33 + [ind_33[-1] + 1] * (int(p3/2) + 1))[None, None, :, None, None, None]
        
        col_11 = (n12*n13*col_1_11 + n13*col_2_11 + col_3_11).flatten()
        col_12 = (n12*n13*col_1_12 + n13*col_2_12 + col_3_12).flatten()
        col_13 = (n12*n13*col_1_13 + n13*col_2_13 + col_3_13).flatten()
        
        col_21 = (n22*n23*col_1_21 + n23*col_2_21 + col_3_21).flatten()
        col_22 = (n22*n23*col_1_22 + n23*col_2_22 + col_3_22).flatten()
        col_23 = (n22*n23*col_1_23 + n23*col_2_23 + col_3_23).flatten()
        
        col_31 = (n32*n33*col_1_31 + n33*col_2_31 + col_3_31).flatten()
        col_32 = (n32*n33*col_1_32 + n33*col_2_32 + col_3_32).flatten()
        col_33 = (n32*n33*col_1_33 + n33*col_2_33 + col_3_33).flatten()
        
        
        
        # Create sparse matrices (1 - component)
        D11 = sparse.csc_matrix((RHS_11.flatten(), (row_11, col_11)), shape=(n11*n12*n13, n11*n12*n13))
        D12 = sparse.csc_matrix((RHS_12.flatten(), (row_12, col_12)), shape=(n11*n12*n13, n11*n12*n13))
        D13 = sparse.csc_matrix((RHS_13.flatten(), (row_13, col_13)), shape=(n11*n12*n13, n11*n12*n13))
        
        D11.eliminate_zeros()
        D12.eliminate_zeros()
        D13.eliminate_zeros()
        
        D1 = sparse.bmat([[D11, D12, D13]], format='csc')
        
        
        # Create sparse matrices (2 - component)
        D21 = sparse.csc_matrix((RHS_21.flatten(), (row_21, col_21)), shape=(n21*n22*n23, n21*n22*n23))
        D22 = sparse.csc_matrix((RHS_22.flatten(), (row_22, col_22)), shape=(n21*n22*n23, n21*n22*n23))
        D23 = sparse.csc_matrix((RHS_23.flatten(), (row_23, col_23)), shape=(n21*n22*n23, n21*n22*n23))
        
        D21.eliminate_zeros()
        D22.eliminate_zeros()
        D23.eliminate_zeros()
        
        D2 = sparse.bmat([[D21, D22, D23]], format='csc')
        
        
        # Create sparse matrices (3 - component)
        D31 = sparse.csc_matrix((RHS_31.flatten(), (row_31, col_31)), shape=(n31*n32*n33, n31*n32*n33))
        D32 = sparse.csc_matrix((RHS_32.flatten(), (row_32, col_32)), shape=(n31*n32*n33, n31*n32*n33))
        D33 = sparse.csc_matrix((RHS_33.flatten(), (row_33, col_33)), shape=(n31*n32*n33, n31*n32*n33))
        
        D31.eliminate_zeros()
        D32.eliminate_zeros()
        D33.eliminate_zeros()
        
        D3 = sparse.bmat([[D31, D32, D33]], format='csc')
        
        return D1, D2, D3