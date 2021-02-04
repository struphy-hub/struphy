import numpy as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.derivatives.derivatives as der



class polar_splines:
    
    def __init__(self, tensor_space, cx, cy):
        
        # =========== extraction operator for discrete 0-forms ==================
        NbaseN = tensor_space.NbaseN
        NbaseD = tensor_space.NbaseD
        
        n0     = NbaseN[0]
        n1     = NbaseN[1]

        n00    = NbaseN[0]*NbaseN[1]
        n10    = NbaseD[0]*NbaseN[1]
        n01    = NbaseN[0]*NbaseD[1]
        n11    = NbaseD[0]*NbaseD[1]

        n0_pol = n00       - 2*n1 + 3
        n1_pol = n10 + n01 - 3*n1 + 2
        n2_pol = n11       - 1*n1

        # size of control triangle
        self.tau  = np.array([(-2*cx[1]).max(), (cx[1] - np.sqrt(3)*cy[1]).max(), (cx[1] + np.sqrt(3)*cy[1]).max()]).max()

        self.Xi_0 = np.zeros((3, NbaseN[1]), dtype=float)
        self.Xi_1 = np.zeros((3, NbaseN[1]), dtype=float)

        self.Xi_0[:, :] = 1/3

        self.Xi_1[0, :] = 1/3 + 2/(3*self.tau)*cx[1, :, 0]
        self.Xi_1[1, :] = 1/3 - 1/(3*self.tau)*cx[1, :, 0] + np.sqrt(3)/(3*self.tau)*cy[1, :, 0]
        self.Xi_1[2, :] = 1/3 - 1/(3*self.tau)*cx[1, :, 0] - np.sqrt(3)/(3*self.tau)*cy[1, :, 0]

        # compute extraction operator
        E0_pol         = np.hstack((self.Xi_0, self.Xi_1))
        E0_pol         = spa.bmat([[E0_pol, None], [None, spa.identity(n0_pol - 3)]], format='csr')
        self.E0_pol_2D = E0_pol
        self.E0_pol_3D = spa.kron(E0_pol, spa.identity(NbaseN[2]), format='csr')
        
        # extraction operator for interpolation points in global projector
        self.I0_pol_2D                  = spa.lil_matrix((n0_pol, n00), dtype=float)
        self.I0_pol_2D[0, n1 + 0*n1//3] = 1.
        self.I0_pol_2D[1, n1 + 1*n1//3] = 1.
        self.I0_pol_2D[2, n1 + 2*n1//3] = 1.
        self.I0_pol_2D[3:, 2*n1:]       = spa.identity(n0_pol - 3)
        self.I0_pol_2D                  = self.I0_pol_2D.tocsr()
        self.I0_pol_3D                  = spa.kron(self.I0_pol_2D, spa.identity(NbaseN[2]), format='csr')
        # =======================================================================
        
        
        
        # =========== extraction operator for discrete 1-forms ==================
        E100_pol = np.zeros((n1_pol, n10), dtype=float)
        E010_pol = np.zeros((n1_pol, n01), dtype=float)
        E001_pol = np.zeros((n0_pol, n00), dtype=float)

        # 1st component
        for s in range(2):
            for j in range(n1):
                E100_pol[s, j] = self.Xi_1[s + 1, j] - self.Xi_0[s + 1, j]

        # 2nd component
        for s in range(2):
            for j in range(n1):
                E010_pol[s,      j] = 0.
                E010_pol[s, n1 + j] = self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j]

        # 3rd component
        for s in range(3):
            for j in range(n1):
                E001_pol[s,      j] = self.Xi_0[s, j]
                E001_pol[s, n1 + j] = self.Xi_1[s, j]

        # tensor product contributions
        E100_pol[2:(2 + n10 - n1)     , 1*n1:] = np.identity(n10 - 1*n1)
        E010_pol[(2 + n10 - n1):n1_pol, 2*n1:] = np.identity(n01 - 2*n1)
        E001_pol[3:, 2*n1:]                    = np.identity(n00 - 2*n1)
        
        self.E1_pol_2D = spa.bmat([[spa.csr_matrix(E100_pol), spa.csr_matrix(E010_pol)]], format='csr')

        # expansion in third dimension
        self.E100_pol  = spa.kron(spa.csr_matrix(E100_pol), spa.identity(NbaseN[2]), format='csr')
        self.E010_pol  = spa.kron(spa.csr_matrix(E010_pol), spa.identity(NbaseN[2]), format='csr')
        self.E001_pol  = spa.kron(spa.csr_matrix(E001_pol), spa.identity(NbaseD[2]), format='csr')

        self.E1_pol    = spa.bmat([[self.E100_pol, self.E010_pol, None], [None, None, self.E001_pol]], format='csr')
        
        # extraction operator for interpolation/histopolation in global projector
        self.I1_1_pol                = spa.lil_matrix((n10 - n1, n10), dtype=float)
        self.I1_1_pol[:n1, 0*n1//3]  = -self.Xi_1[0].reshape(n1, 1)
        self.I1_1_pol[:n1, 1*n1//3]  = -self.Xi_1[1].reshape(n1, 1)
        self.I1_1_pol[:n1, 2*n1//3]  = -self.Xi_1[2].reshape(n1, 1)
        self.I1_1_pol[:n1,   :1*n1] += spa.identity(n1)
        self.I1_1_pol[:n1, n1:2*n1]  = spa.identity(n1)
        self.I1_1_pol[n1:,   2*n1:]  = spa.identity(n10 - 2*n1)
        self.I1_1_pol                = self.I1_1_pol.tocsr()
        
        self.I1_2_pol                                   = spa.lil_matrix((2 + n01 - 2*n1, n01), dtype=float)
        self.I1_2_pol[0, (n1 + 0*n1//3):(n1 + 1*n1//3)] = np.ones((1, n1//3), dtype=float)
        self.I1_2_pol[1, (n1 + 0*n1//3):(n1 + 1*n1//3)] = np.ones((1, n1//3), dtype=float)
        self.I1_2_pol[1, (n1 + 1*n1//3):(n1 + 2*n1//3)] = np.ones((1, n1//3), dtype=float)
        self.I1_2_pol[2:, 2*n1:]                        = spa.identity(n01 - 2*n1)
        self.I1_2_pol                                   = self.I1_2_pol.tocsr()
        
        self.I1_pol_2D = spa.bmat([[self.I1_1_pol, None], [None, self.I1_2_pol]], format='csr')
        # =========================================================================
        
        
        
        # =========== extraction operator for discrete 2-forms ====================
        E011_pol = np.zeros((n1_pol, n01), dtype=float)
        E101_pol = np.zeros((n1_pol, n10), dtype=float)
        E110_pol = np.zeros((n2_pol, n11), dtype=float)

        # 1st component
        for s in range(2):
            for j in range(n1):
                E011_pol[s,      j] = 0.
                E011_pol[s, n1 + j] = (self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j])

        # 2nd component
        for s in range(2):
            for j in range(n1):
                E101_pol[s, j] = -(self.Xi_1[s + 1, j] - self.Xi_0[s + 1, j]) 

        # tensor product contributions
        E011_pol[2:(2 + n01 - 2*n1), 2*n1:] = np.identity(n01 - 2*n1)
        E101_pol[(2 + n01 - 2*n1): , 1*n1:] = np.identity(n10 - 1*n1)
        E110_pol[:, 1*n1:]                  = np.identity(n11 - 1*n1)
        
        self.E2_pol_2D = spa.bmat([[spa.csr_matrix(E011_pol), spa.csr_matrix(E101_pol)]], format='csr')
   
        # expansion in third dimension
        self.E011_pol = spa.kron(spa.csr_matrix(E011_pol), spa.identity(NbaseD[2]), format='csr')
        self.E101_pol = spa.kron(spa.csr_matrix(E101_pol), spa.identity(NbaseD[2]), format='csr')
        self.E110_pol = spa.kron(spa.csr_matrix(E110_pol), spa.identity(NbaseN[2]), format='csr')
        self.E2_pol   = spa.bmat([[self.E011_pol, self.E101_pol, None], [None, None, self.E110_pol]], format='csr')
        
        # extraction operator for 2D histopolation in global projector
        self.I2_pol_2D = spa.bmat([[self.I1_2_pol, None], [None, self.I1_1_pol]], format='csr')
        # =========================================================================
        
        
        # =========== extraction operator for discrete 3-forms ====================
        E3_pol = np.zeros((n2_pol, n11), dtype=float)
        
        # tensor product contributions
        E3_pol[:, 1*n1:] = np.identity(n11 - 1*n1)
        
        self.E3_pol_2D = spa.csr_matrix(E3_pol)
        
        # expansion in third dimension
        self.E3_pol = spa.kron(spa.csr_matrix(E3_pol), spa.identity(NbaseD[2]), format='csr')
        
        # extraction operator for 2D histopolation in global projector
        self.I3_pol_2D = spa.lil_matrix((n11 - n1, n11), dtype=float)
        
        for i2 in range(n1):
            self.I3_pol_2D[i2, 0*n1//3:1*n1//3] = -(self.Xi_1[1, (i2 + 1)%n1] - self.Xi_1[1, i2]) - (self.Xi_1[2, (i2 + 1)%n1] - self.Xi_1[2, i2])
            self.I3_pol_2D[i2, 1*n1//3:2*n1//3] = -(self.Xi_1[2, (i2 + 1)%n1] - self.Xi_1[2, i2])
            
        self.I3_pol_2D[:n1,   :1*n1] += spa.identity(n1)
        self.I3_pol_2D[:n1, n1:2*n1]  = spa.identity(n1)
        
        self.I3_pol_2D[n1:, 2*n1:]    = spa.identity(n11 - 2*n1)
        # =========================================================================
        
        
        
        # ========= discrete gradient matrix ====================================
        derivatives = der.discrete_derivatives(tensor_space)
        
        grad_new    = np.zeros((2, n0_pol), dtype=float)

        grad1       = np.zeros((n10 - 1*n1, n0_pol), dtype=float)
        grad2       = np.zeros((n01 - 2*n1, n0_pol), dtype=float)

        grad1_ten   = spa.kron(derivatives.grad_1d[0], spa.identity(n1), format='csr').toarray()
        grad2_ten   = spa.kron(spa.identity(n0), derivatives.grad_1d[1], format='csr').toarray()


        # two new basis functions
        grad_new[0, 0] = -1.
        grad_new[0, 1] =  1.

        grad_new[1, 0] = -1.
        grad_new[1, 2] =  1.

        # radial dofs (D N)
        grad1[:, 3:] = grad1_ten[1*n1:, 2*n1:]

        for j in range(n1):
            for s in range(3):
                grad1[j, s] = - self.Xi_1[s, j]

        # angular dofs (N D)
        grad2[:, 3:] = grad2_ten[2*n1:, 2*n1:]

        self.grad_pol = spa.bmat([[spa.kron(grad_new            , spa.identity(NbaseN[2]), format='csr')], 
                                  [spa.kron(grad1               , spa.identity(NbaseN[2]), format='csr')], 
                                  [spa.kron(grad2               , spa.identity(NbaseN[2]), format='csr')], 
                                  [spa.kron(spa.identity(n0_pol), derivatives.grad_1d[2] , format='csr')]], format='csr')
        # =======================================================================
        
        
        
        # ========= discrete curl matrix ==========================================
        div_new  = np.zeros((n2_pol, 2), dtype=float)

        div1_ten = -spa.kron(spa.identity(NbaseD[0]), derivatives.grad_1d[1], format='csr').toarray()
        div2_ten =  spa.kron(derivatives.grad_1d[0], spa.identity(NbaseD[1]), format='csr').toarray()

        for s in range(2):
            for j in range(n1):
                div_new[j, s] = -(self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j])
                
        curl_pol_11 = -spa.kron(spa.identity(2), derivatives.grad_1d[2], format='csr')
        curl_pol_14 =  spa.kron(grad_new, np.identity(NbaseD[2]), format='csr')
        
        curl_pol_23 = -spa.kron(spa.identity(NbaseN[0] - 2), spa.kron(spa.identity(NbaseD[1]), derivatives.grad_1d[2]), format='csr')
        curl_pol_24 =  spa.kron(grad2, spa.identity(NbaseD[2]), format='csr')
        
        curl_pol_32 =  spa.kron(spa.identity(NbaseD[0] - 1), spa.kron(spa.identity(NbaseN[1]), derivatives.grad_1d[2]), format='csr')
        curl_pol_34 = -spa.kron(grad1, spa.identity(NbaseD[2]), format='csr')
        
        curl_pol_41 =  spa.kron(div_new, spa.identity(NbaseN[2]), format='csr')
        curl_pol_42 =  spa.kron(spa.csr_matrix(div1_ten[n1:, 1*n1:]), spa.identity(NbaseN[2]), format='csr')
        curl_pol_43 =  spa.kron(spa.csr_matrix(div2_ten[n1:, 2*n1:]), spa.identity(NbaseN[2]), format='csr')

        self.curl_pol = spa.bmat([[curl_pol_11, None       , None       , curl_pol_14], 
                                  [None       , None       , curl_pol_23, curl_pol_24], 
                                  [None       , curl_pol_32, None       , curl_pol_34], 
                                  [curl_pol_41, curl_pol_42, curl_pol_43, None       ]])
        # =========================================================================
        
        
        
        # ========= discrete div matrix ===========================================
        div_new  = np.zeros((n2_pol, 2), dtype=float)

        div1_ten = spa.kron(derivatives.grad_1d[0] , spa.identity(NbaseD[1]), format='csr').toarray()
        div2_ten = spa.kron(spa.identity(NbaseD[0]), derivatives.grad_1d[1] , format='csr').toarray()

        for s in range(2):
            for j in range(n1):
                div_new[j, s] = -(self.Xi_1[s + 1, (j + 1)%n1] - self.Xi_1[s + 1, j])

        self.div_pol = spa.bmat([[spa.kron(div_new, spa.identity(NbaseD[2])                             , format='csr'),
                             spa.kron(spa.csr_matrix(div1_ten[n1:, 2*n1:]), spa.identity(NbaseD[2]), format='csr'), 
                             spa.kron(spa.csr_matrix(div2_ten[n1:, 1*n1:]), spa.identity(NbaseD[2]), format='csr'), 
                             spa.kron(spa.identity(NbaseD[0] - 1), spa.kron(spa.identity(NbaseD[1]), derivatives.grad_1d[2]), format='csr')]], format='csr')
        # =========================================================================