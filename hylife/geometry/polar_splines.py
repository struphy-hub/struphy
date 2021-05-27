import numpy as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.derivatives.derivatives as der


# ============================= 2D polar splines ===================================
class polar_splines_2D:
    
    def __init__(self, tensor_space, cx, cy):
        
        n0, n1 = tensor_space.NbaseN
        d0, d1 = tensor_space.NbaseD
        
        # location of pole
        self.x0 = cx[0, 0]
        self.y0 = cy[0, 0]

        # number of polar basis functions in V0   (NN)
        self.Nbase0  = (n0 - 2)*n1 + 3
        
        # number of polar basis functions in V1_C (DN ND) (1st and 2nd component)
        self.Nbase1C = (d0 - 1)*n1 + (n0 - 2)*d1 + 2
        
        # number of polar basis functions in V1_D (ND DN) (1st and 2nd component)
        self.Nbase1D = (d0 - 1)*n1 + (n0 - 2)*d1 + 2
        
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
        self.E0 = spa.bmat([[np.hstack((self.Xi_0, self.Xi_1)), None], [None, spa.identity((n0 - 2)*n1)]], format='csr')
        
        # global projection extraction operator for interpolation points
        self.P0                   = spa.lil_matrix((self.Nbase0, n0*n1), dtype=float)
        self.P0[0 , n1 + 0*n1//3] = 1.
        self.P0[1 , n1 + 1*n1//3] = 1.
        self.P0[2 , n1 + 2*n1//3] = 1.
        self.P0[3:, 2*n1:]        = spa.identity((n0 - 2)*n1)
        self.P0                   = self.P0.tocsr()
        # =======================================================================
        
        
        # =========== extraction operators for discrete 1-forms (H_curl) ========
        self.E1C_1 = spa.lil_matrix((self.Nbase1C, d0*n1), dtype=float)
        self.E1C_2 = spa.lil_matrix((self.Nbase1C, n0*d1), dtype=float)

        # 1st component
        for s in range(2):
            for j in range(n1):
                self.E1C_1[(d0 - 1)*n1 + s, j] = self.Xi_1[s + 1, j] - self.Xi_0[s + 1, j]
                
        self.E1C_1[:(d0 - 1)*n1, n1:] = np.identity((d0 - 1)*n1)
        self.E1C_1 = self.E1C_1.tocsr()

        # 2nd component
        for s in range(2):
            for j in range(n1):
                self.E1C_2[(d0 - 1)*n1 + s,      j] = 0.
                self.E1C_2[(d0 - 1)*n1 + s, n1 + j] = self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j]

        self.E1C_2[((d0 - 1)*n1 + 2):, 2*d1:] = np.identity((n0 - 2)*d1)
        self.E1C_2 = self.E1C_2.tocsr()
        
        # combined first and second component
        self.E1C = spa.bmat([[self.E1C_1, self.E1C_2]], format='csr')

        # extraction operator for interpolation/histopolation in global projector
        self.P1C_1 = spa.lil_matrix(((d0 - 1)*n1    , d0*n1), dtype=float)
        self.P1C_2 = spa.lil_matrix(((n0 - 2)*d1 + 2, n0*d1), dtype=float)
        
        # 1st component
        self.P1C_1[:n1, 0*n1//3]  = -self.Xi_1[0].reshape(n1, 1)
        self.P1C_1[:n1, 1*n1//3]  = -self.Xi_1[1].reshape(n1, 1)
        self.P1C_1[:n1, 2*n1//3]  = -self.Xi_1[2].reshape(n1, 1)
        self.P1C_1[:n1,   :1*n1] += spa.identity(n1)
        self.P1C_1[:n1, n1:2*n1]  = spa.identity(n1)
        self.P1C_1[n1:,   2*n1:]  = spa.identity((d0 - 2)*n1)
        self.P1C_1                = self.P1C_1.tocsr()
        
        # 2nd component
        self.P1C_2[0, (n1 + 0*n1//3):(n1 + 1*n1//3)] = np.ones((1, n1//3), dtype=float)
        self.P1C_2[1, (n1 + 0*n1//3):(n1 + 1*n1//3)] = np.ones((1, n1//3), dtype=float)
        self.P1C_2[1, (n1 + 1*n1//3):(n1 + 2*n1//3)] = np.ones((1, n1//3), dtype=float)
        self.P1C_2[2:, 2*n1:]                        = spa.identity((n0 - 2)*d1)
        self.P1C_2                                   = self.P1C_2.tocsr()
        
        # combined first and second component
        self.P1C = spa.bmat([[self.P1C_1, None], [None, self.P1C_2]], format='csr')
        # =========================================================================
        
        
        # ========= extraction operators for discrete 1-forms (H_div) =============
        self.E1D_1 = spa.lil_matrix((self.Nbase1D, n0*d1), dtype=float)
        self.E1D_2 = spa.lil_matrix((self.Nbase1D, d0*n1), dtype=float)

        # 1st component
        for s in range(2):
            for j in range(n1):
                self.E1D_1[s,      j] = 0.
                self.E1D_1[s, n1 + j] = (self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j])
                
        self.E1D_1[2:(2 + (n0 - 2)*d1), 2*n1:] = np.identity((n0 - 2)*d1)
        self.E1D_1 = self.E1D_1.tocsr()

        # 2nd component
        for s in range(2):
            for j in range(n1):
                self.E1D_2[s, j] = -(self.Xi_1[s + 1, j] - self.Xi_0[s + 1, j])
                
        self.E1D_2[(2 + (n0 - 2)*d1):, 1*n1:] = np.identity((d0 - 1)*n1)
        self.E1D_2 = self.E1D_2.tocsr()

        # combined first and second component
        self.E1D = spa.bmat([[self.E1D_1, self.E1D_2]], format='csr')
        
        # extraction operator for interpolation/histopolation in global projector
        
        # 1st component
        self.P1D_1 = self.P1C_2
        
        # 2nd component
        self.P1D_2 = self.P1C_1
        
        # combined first and second component
        self.P1D = spa.bmat([[self.P1D_1, None], [None, self.P1D_2]], format='csr')
        # =========================================================================
        
        
        # =========== extraction operators for discrete 2-forms ===================
        self.E2 = spa.lil_matrix((self.Nbase2, d0*d1), dtype=float)
        
        self.E2[:, 1*d1:] = np.identity((d0 - 1)*d1)
        self.E2 = self.E2.tocsr()
        
        # 3rd component
        self.P2 = spa.lil_matrix(((d0 - 1)*d1, d0*d1), dtype=float)
        
        for i2 in range(d1):
            
            # block A
            self.P2[i2, 0*n1//3:1*n1//3] = -(self.Xi_1[1, (i2 + 1)%n1] - self.Xi_1[1, i2]) - (self.Xi_1[2, (i2 + 1)%n1] - self.Xi_1[2, i2])
            
            # block B
            self.P2[i2, 1*n1//3:2*n1//3] = -(self.Xi_1[2, (i2 + 1)%n1] - self.Xi_1[2, i2])
            
        self.P2[:d1,   :1*d1] += spa.identity(d1)
        self.P2[:d1, d1:2*d1]  = spa.identity(d1)
        
        self.P2[d1:, 2*d1:]    = spa.identity((d0 - 2)*d1)
        self.P2                = self.P2.tocsr()
        # =========================================================================
        
        
        # ========================= 1D discrete derivatives =======================
        grad_1d_1 = spa.csc_matrix(der.grad_1d_matrix(tensor_space.spaces[0]))
        grad_1d_2 = spa.csc_matrix(der.grad_1d_matrix(tensor_space.spaces[1]))
        # =========================================================================
        
        
        # ========= discrete polar gradient matrix ================================
        grad_1 = spa.lil_matrix(((d0 - 1)*n1    , self.Nbase0), dtype=float)
        grad_2 = spa.lil_matrix(((n0 - 2)*d1 + 2, self.Nbase0), dtype=float)

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
        self.G = spa.bmat([[grad_1], [grad_2]], format='csr')
        # =======================================================================
        
        
        # ========= discrete polar curl matrix ===================================
        # 2D vector curl
        vector_curl_1 = spa.lil_matrix(((n0 - 2)*d1 + 2, self.Nbase0), dtype=float)
        vector_curl_2 = spa.lil_matrix(((d0 - 1)*n1    , self.Nbase0), dtype=float)
        
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
        self.VC = spa.bmat([[vector_curl_1], [vector_curl_2]], format='csr')
        
        # 2D scalar curl
        self.SC = spa.lil_matrix((self.Nbase2, self.Nbase1C), dtype=float)
        
        # radial dofs (D N)
        self.SC[:, :(d0 - 1)*n1] = -spa.kron(spa.identity(d0 - 1), grad_1d_2)
        
        # angular dofs (N D)
        for s in range(2):
            for j in range(n1):
                self.SC[j, (d0 - 1)*n1 + s] = -(self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j])
                
        self.SC[:, ((d0 - 1)*n1 + 2):] = spa.kron(grad_1d_1[1:, 2:], spa.identity(d1))
        self.SC = self.SC.tocsr()
        # =========================================================================
        
        
        # ========= discrete polar div matrix =====================================
        self.D = spa.lil_matrix((self.Nbase2, self.Nbase1D), dtype=float)
        
        # angular dofs (N D)
        for s in range(2):
            for j in range(d1):
                self.D[j, s]  = -(self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j])
                
        self.D[:, 2:((d0 - 1)*n1 + 2)] = spa.kron(grad_1d_1[1:, 2:], spa.identity(d1))
        
        # radial dofs (D N)
        self.D[:, ((d0 - 1)*n1 + 2):] = spa.kron(spa.identity(d0 - 1), grad_1d_2)
        self.D = self.D.tocsr()
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