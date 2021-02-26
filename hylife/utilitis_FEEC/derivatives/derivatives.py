# coding: utf-8
#
# Copyright 2021 Florian Holderied

"""
Modules to assemble discrete derivatives.
"""

import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.derivatives.kernels_derivatives as ker


# ============== discrete gradient matrix (1d) ==============
def grad_1d_matrix(spline_space):
    """
    Returns the 1d strong discrete gradient matrix corresponding to the given B-spline space of degree p.
    
    Parameters
    ----------
    spline_space : spline_space_1d
        
    Returns
    -------
    grad : array_like
        strong discrete gradient matrix
    """
    
    NbaseN = spline_space.NbaseN      # total number of basis functions (N)
    bc     = spline_space.bc          # boundary conditions (True : periodic, False : clamped)
    
    
    # periodic splines
    if bc == True:
        
        grad = np.zeros((NbaseN, NbaseN), dtype=float)
        
        for i in range(NbaseN):
            grad[i, i] = -1.
            if i < NbaseN - 1:
                grad[i, i + 1] = 1.
        grad[-1, 0] = 1.
        
        return grad
    
    # non-periodic splines
    else:
        
        grad = np.zeros((NbaseN - 1, NbaseN))
    
        for i in range(NbaseN - 1):        
            grad[i, i] = -1.
            grad[i, i  + 1] = 1.
            
        return grad

    
    
    
# ===== discrete derivatives in two dimensions ================
class discrete_derivatives_2D:
    """
    Class for discrete derivatives for a 2d tensor product B-spline space.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
    """
    
    def __init__(self, tensor_space):
        
        self.NbaseN  = tensor_space.NbaseN
        self.NbaseD  = tensor_space.NbaseD
        self.bc      = tensor_space.bc
        
        self.grad_1d = [grad_1d_matrix(spl) for spl in tensor_space.spaces]
        
        # transform to sparse matrix
        self.grad_1d = [spa.csc_matrix(grad_1d) for grad_1d in self.grad_1d]
    
    
    # ================== grad ======================
    def grad_2d_matrix(self):
        
        G1 = spa.kron(self.grad_1d[0], spa.identity(self.NbaseN[1]))
        G2 = spa.kron(spa.identity(self.NbaseN[0]), self.grad_1d[1])
        
        G  = spa.bmat([[G1], [G2]], format='csc')
        
        return G
    
    
    # ================== curl ======================
    def curl_2d_matrix(self, kind):
        
        # 'Hcurl' : corresponds to sequence grad --> curl
        if   kind == 'Hcurl':
            
            C1 = spa.kron(spa.identity(self.NbaseD[0]), self.grad_1d[1])
            C2 = spa.kron(self.grad_1d[0], spa.identity(self.NbaseD[1]))

            C  = spa.bmat([[-C1, C2]], format='csc')
            
        # 'Hdiv' : corresponds to sequence curl --> div
        elif kind == 'Hdiv':
            
            C1 = spa.kron(spa.identity(self.NbaseN[0]), self.grad_1d[1])
            C2 = spa.kron(self.grad_1d[0], spa.identity(self.NbaseN[1]))

            C  = spa.bmat([[C1], [-C2]], format='csc')
            
        return C
            
    # ================== div ========================
    def div_2d(self):
        
        D1 = spa.kron(self.grad_1d[0], spa.identity(self.NbaseD[1]))
        D2 = spa.kron(spa.identity(self.NbaseD[0]), self.grad_1d[1])
        
        D  = sparse.bmat([[D1, D2]], format='csc')
        
        return D
    
    
    
    
    
# ===== discrete derivatives in three dimensions ================
class discrete_derivatives_3D:
    """
    Class for discrete derivatives for a 3d tensor product B-spline space.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
    """
    
    def __init__(self, tensor_space, polar_splines=None, bc1=['free', 'free'], bc2=['free', 'free'], bc3=['free', 'free']):
        
        self.NbaseN       = tensor_space.NbaseN
        self.NbaseD       = tensor_space.NbaseD
        
        self.tensor_space = tensor_space
        
        self.bc1 = bc1
        self.bc2 = bc2
        self.bc3 = bc3
        
        self.grad_1d = [grad_1d_matrix(spl) for spl in tensor_space.spaces]
        
        # apply boundary conditions in 1-direction
        if tensor_space.bc[0] == False:
        
            if bc1[0] == 'dirichlet':
                self.grad_1d[0][:,  0] = 0.
            if bc1[1] == 'dirichlet':
                self.grad_1d[0][:, -1] = 0.
            
        # apply boundary conditions in 2-direction
        if tensor_space.bc[1] == False:
        
            if bc2[0] == 'dirichlet':
                self.grad_1d[1][:,  0] = 0.
            if bc2[1] == 'dirichlet':
                self.grad_1d[1][:, -1] = 0.
            
        # apply boundary conditions in 3-direction
        if tensor_space.bc[2] == False:
        
            if bc3[0] == 'dirichlet':
                self.grad_1d[2][:,  0] = 0.
            if bc3[1] == 'dirichlet':
                self.grad_1d[2][:, -1] = 0.
            
        # transform to sparse matrix
        self.grad_1d = [spa.csc_matrix(grad_1d) for grad_1d in self.grad_1d]
        
        
        if polar_splines == None:
            
            # discrete grad
            G1 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseN[1])), spa.identity(self.NbaseN[2]))
            G2 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), self.grad_1d[1]), spa.identity(self.NbaseN[2]))
            G3 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), spa.identity(self.NbaseN[1])), self.grad_1d[2])

            self.GRAD = spa.bmat([[G1], [G2], [G3]], format='csr')
            
            # discrete curl
            C12 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), spa.identity(self.NbaseD[1])), self.grad_1d[2])
            C13 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), self.grad_1d[1]), spa.identity(self.NbaseD[2]))

            C21 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), spa.identity(self.NbaseN[1])), self.grad_1d[2])
            C23 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseN[1])), spa.identity(self.NbaseD[2]))

            C31 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), self.grad_1d[1]), spa.identity(self.NbaseN[2]))
            C32 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseD[1])), spa.identity(self.NbaseN[2]))

            self.CURL = spa.bmat([[None, -C12, C13], [C21, None, -C23], [-C31, C32, None]], format='csr')
            
            # discrete div
            D1 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseD[1])), spa.identity(self.NbaseD[2]))
            D2 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), self.grad_1d[1]), spa.identity(self.NbaseD[2]))
            D3 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), spa.identity(self.NbaseD[1])), self.grad_1d[2])

            self.DIV = spa.bmat([[D1, D2, D3]], format='csr')
            
        else:
            
            # discrete polar grad
            self.GRAD = polar_splines.GRAD
            
            # discrete polar curl
            self.CURL = polar_splines.CURL
            
            # discrete polar div
            self.DIV  = polar_splines.DIV
        
    
    # ================== grad ==================
    def grad_3d_matrix(self):
        
        G1 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseN[1])), spa.identity(self.NbaseN[2]))
        G2 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), self.grad_1d[1]), spa.identity(self.NbaseN[2]))
        G3 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), spa.identity(self.NbaseN[1])), self.grad_1d[2])

        G  = spa.bmat([[G1], [G2], [G3]], format='csc')

        return G
    
    # ================== curl ==================
    def curl_3d_matrix(self):
        
        C12 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), spa.identity(self.NbaseD[1])), self.grad_1d[2])
        C13 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), self.grad_1d[1]), spa.identity(self.NbaseD[2]))
        
        C21 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), spa.identity(self.NbaseN[1])), self.grad_1d[2])
        C23 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseN[1])), spa.identity(self.NbaseD[2]))
        
        C31 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), self.grad_1d[1]), spa.identity(self.NbaseN[2]))
        C32 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseD[1])), spa.identity(self.NbaseN[2]))
        
        C   = spa.bmat([[None, -C12, C13], [C21, None, -C23], [-C31, C32, None]], format='csc')
        
        return C
    
    # ================== div ==================
    def div_3d_matrix(self):
        
        D1 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseD[1])), spa.identity(self.NbaseD[2]))
        D2 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), self.grad_1d[1]), spa.identity(self.NbaseD[2]))
        D3 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), spa.identity(self.NbaseD[1])), self.grad_1d[2])

        D  = spa.bmat([[D1, D2, D3]], format='csc')

        return D     

    
    # ========================================
    def apply_GRAD_3d_kron(self, f0):
        """
        apply the disrete gradient operator in tensor-product fashion with 3d vectors

        f1^1_mjk = G^1_mi f0_ijk 
        f1^2_ink = G^2_nj f0_ijk
        f1^3_ijo = G^3_ok f0_ijk

        Parameters
        ----------
        self.grad_1d : 3 sparse matrices representing the 1D gradient matrix in each direction, 
                       of size (d0, n0), (d1, n1), (d2, n2)

        f0 : 3d array of size (n0, n1, n2)
        
        Returns
        -------
        f1_1 : 3d array of 1st component of size (d0, n1, n2)   
        f1_2 : 3d array of 2nd component of size (n0, d1, n2)   
        f1_3 : 3d array of 3rd component of size (n0, n1, d2)   
        """
        
        d0, n0 = self.grad_1d[0].shape
        d1, n1 = self.grad_1d[1].shape
        d2, n2 = self.grad_1d[2].shape

        assert (f0.shape == (n0, n1, n2))
        
        f1_1 = self.grad_1d[0].dot(f0.reshape(n0, n1*n2))
        f1_2 = self.grad_1d[1].dot(f0.reshape(n0, n1*n2).T.reshape(n1, n2*n0)).reshape(d1*n2, n0).T
        f1_3 = self.grad_1d[2].dot(f0.reshape(n0*n1, n2).T).T

        return f1_1.reshape(d0, n1, n2), f1_2.reshape(n0, d1, n2), f1_3.reshape(n0, n1, d2)     


    # ==========================================
    def apply_CURL_3d_kron(self, f1_1, f1_2, f1_3):
        """
        apply the discrete curl operator in tensor-product fashion with 3d vectors

        f2^1_ino = G^2_nj f1^3_ijo - G^3_ok f1^2_ink
        f2^2_mjo = G^3_ok f1^1_mjk - G^1_mi f1^3_ijo
        f2^3_mnk = G^1_mi f1^2_ink - G^2_nj f1^1_mjk

        Parameters
        ----------
        self.grad_1d : 3 sparse matrices representing the 1D gradient matrix in each direction, 
                       of size (d0, n0), (d1, n1), (d2, n2)
            
        f1_1 : 3d array of 1st component of size (d0, n1, n2)   
        f1_2 : 3d array of 2nd component of size (n0, d1, n2)   
        f1_3 : 3d array of 3rd component of size (n0, n1, d2)   

        Returns
        -------
        f2_1 : 3d array of 1st component of size (n0, d1, d2)   
        f2_2 : 3d array of 2nd component of size (d0, n1, d2)   
        f2_3 : 3d array of 3rd component of size (d0, d1, n2)   
        """
        
        d0, n0 = self.grad_1d[0].shape
        d1, n1 = self.grad_1d[1].shape
        d2, n2 = self.grad_1d[2].shape

        assert (f1_1.shape == (d0, n1, n2)) 
        assert (f1_2.shape == (n0, d1, n2)) 
        assert (f1_3.shape == (n0, n1, d2))
        
        f2_1  = np.zeros((n0, d1, d2), dtype=float)
        f2_2  = np.zeros((d0, n1, d2), dtype=float)
        f2_3  = np.zeros((d0, d1, n2), dtype=float)
        
        
        # 1st component
        f2_1 += self.grad_1d[1].dot(f1_3.reshape(n0, n1*d2).T.reshape(n1, d2*n0)).reshape(d1*d2, n0).T.reshape(n0, d1, d2)
        
        f2_1 -= self.grad_1d[2].dot(f1_2.reshape(n0*d1, n2).T).T.reshape(n0, d1, d2)
        
        # 2nd component
        f2_2 += self.grad_1d[2].dot(f1_1.reshape(d0*n1, n2).T).T.reshape(d0, n1, d2)
        
        f2_2 -= self.grad_1d[0].dot(f1_3.reshape(n0, n1*d2)).reshape(d0, n1, d2)

        # 3rd omponent
        f2_3 += self.grad_1d[0].dot(f1_2.reshape(n0, d1*n2)).reshape(d0, d1, n2)
        
        f2_3 -= self.grad_1d[1].dot(f1_1.reshape(d0, n1*n2).T.reshape(n1, n2*d0)).reshape(d1*n2, d0).T.reshape(d0, d1, n2)
        
        return f2_1, f2_2, f2_3     

    
    # ===========================================
    def apply_DIV_3d_kron(self, f2_1, f2_2, f2_3):
        """
        apply the discrete divergence operator in tensor-product fashion with 3d vectors

        f3_mno = G_mi f2^1_ino + G_nj f2^2_mjo + G_ok f2^3_mnk

        Parameters
        ----------
        self.grad_1d : 3 sparse matrices representing the 1D gradient matrix in each direction, 
                       of size (d0, n0), (d1, n1), (d2, n2)
            
        f2_1 : 3d array of 1st component of size (n0, d1, d2)   
        f2_2 : 3d array of 2nd component of size (d0, n1, d2)   
        f2_3 : 3d array of 3rd component of size (d0, d1, n2)   

        Returns
        -------
        f3 : 3d array of size (d0, d1, d2)
        """

        d0, n0 = self.grad_1d[0].shape
        d1, n1 = self.grad_1d[1].shape
        d2, n2 = self.grad_1d[2].shape

        assert (f2_1.shape == (n0, d1, d2)) 
        assert (f2_2.shape == (d0, n1, d2)) 
        assert (f2_3.shape == (d0, d1, n2))
                    
        f3 = np.zeros((d0, d1, d2), dtype=float)
                    
        f3 += self.grad_1d[0].dot(f2_1.reshape(n0, d1*d2)).reshape(d0, d1, d2)
        f3 += self.grad_1d[1].dot(f2_2.reshape(d0, n1*d2).T.reshape(n1, d2*d0)).reshape(d1*d2, d0).T.reshape(d0, d1, d2)
        f3 += self.grad_1d[2].dot(f2_3.reshape(d0*d1, n2).T).T.reshape(d0, d1, d2)

        return f3
    
    
    # ================================================
    def grad_strong(self, f0):
        
        # final coefficients of the operation GRAD(f0)
        f1_1 = np.empty(self.tensor_space.Nbase_1form[0], dtype=float)
        f1_2 = np.empty(self.tensor_space.Nbase_1form[1], dtype=float)
        f1_3 = np.empty(self.tensor_space.Nbase_1form[2], dtype=float)
        
        ker.grad_strong(f0, f1_1, f1_2, f1_3)
        
        return f1_1, f1_2, f1_3
    
    # ================================================
    def curl_strong(self, f1_1, f1_2, f1_3):
        
        # final coefficients of the operation CURL(f1)
        f2_1 = np.empty(self.tensor_space.Nbase_2form[0], dtype=float)
        f2_2 = np.empty(self.tensor_space.Nbase_2form[1], dtype=float)
        f2_3 = np.empty(self.tensor_space.Nbase_2form[2], dtype=float)
        
        ker.curl_strong(f1_1, f1_2, f1_3, f2_1, f2_2, f2_3)
        
        return f2_1, f2_2, f2_3
    
    # ================================================
    def div_strong(self, f2_1, f2_2, f2_3):
        
        # final coefficients of the operation DIV(f2)
        f3 = np.empty(self.tensor_space.Nbase_3form, dtype=float)
        
        ker.div_strong(f2_1, f2_2, f2_3, f3)
        
        return f3
    
    # ================================================
    def grad_weak(self, f1_1, f1_2, f1_3):
        
        # final 0-form coefficients of the operation GRAD.T(f1)
        f0 = np.zeros(self.tensor_space.Nbase_0form, dtype=float)
        
        # spline boundary conditions
        bc_splines = [1*self.tensor_space.bc[0], 1*self.tensor_space.bc[1], 1*self.tensor_space.bc[2]]
        
        ker.grad_weak(f1_1, f1_2, f1_3, f0, bc_splines)
        
        # boundary conditions (1-direction)
        if self.tensor_space.bc[0] == False:
            
            # f0(0, y, z)
            if self.bc1[0] == 'free':
                f0[ 0, :, :] -= f1_1[ 0, :, :]
            # f0(1, y, z)
            if self.bc1[1] == 'free':
                f0[-1, :, :] += f1_1[-1, :, :]
            
        # boundary conditions (2-direction)
        if self.tensor_space.bc[1] == False:
            
            # f0(x, 0, z)
            if self.bc2[0] == 'free':
                f0[:,  0, :] -= f1_2[:,  0, :]
            # f0(x, 1, z)
            if self.bc2[1] == 'free':
                f0[:, -1, :] += f1_2[:, -1, :]
            
        # boundary conditions (3-direction)
        if self.tensor_space.bc[2] == False:
            
            # f0(x, y, 0)
            if self.bc3[0] == 'free':
                f0[:, :,  0] -= f1_3[:, :,  0]
            # f0(x, y, 1)
            if self.bc3[1] == 'free':
                f0[:, :, -1] += f1_3[:, :, -1]
        
        return f0
    
    # ================================================
    def curl_weak(self, f2_1, f2_2, f2_3):
        
        # final 1-form coefficients of the operation CURL.T(f2)
        f1_1 = np.zeros(self.tensor_space.Nbase_1form[0], dtype=float)
        f1_2 = np.zeros(self.tensor_space.Nbase_1form[1], dtype=float)
        f1_3 = np.zeros(self.tensor_space.Nbase_1form[2], dtype=float)
        
        # spline boundary conditions
        bc_splines = [1*self.tensor_space.bc[0], 1*self.tensor_space.bc[1], 1*self.tensor_space.bc[2]]
        
        ker.curl_weak(f2_1, f2_2, f2_3, f1_1, f1_2, f1_3, bc_splines)
        
        # boundary conditions (1-direction)
        if self.tensor_space.bc[0] == False:
            
            # f1_2(0, y, z)
            if self.bc1[0] == 'free':
                f1_2[ 0, :, :] -= f2_3[ 0, :, :]
            # f1_2(1, y, z)
            if self.bc1[1] == 'free':
                f1_2[-1, :, :] += f2_3[-1, :, :]
                
            # f1_3(0, y, z)
            if self.bc1[0] == 'free':
                f1_3[ 0, :, :] += f2_2[ 0, :, :]
            # f1_3(1, y, z)
            if self.bc1[1] == 'free':
                f1_3[-1, :, :] -= f2_2[-1, :, :]
            
        # boundary conditions (2-direction)
        if self.tensor_space.bc[1] == False:
            
            # f1_3(x, 0, z)
            if self.bc2[0] == 'free':
                f1_3[:,  0, :] -= f2_1[:,  0, :]
            # f1_3(x, 1, z)
            if self.bc2[1] == 'free':
                f1_3[:, -1, :] += f2_1[:, -1, :]
                
            # f1_1(x, 0, z)
            if self.bc2[0] == 'free':
                f1_1[:,  0, :] += f2_3[:,  0, :]
            # f1_1(x, 1, z)
            if self.bc2[1] == 'free':
                f1_1[:, -1, :] -= f2_3[:, -1, :]
                
        # boundary conditions (3-direction)
        if self.tensor_space.bc[2] == False:
            
            # f1_1(x, y, 0)
            if self.bc3[0] == 'free':
                f1_1[:, :,  0] -= f2_2[:, :,  0]
            # f1_1(x, y, 1)
            if self.bc3[1] == 'free':
                f1_1[:, :, -1] += f2_2[:, :, -1]
                
            # f1_2(x, y, 0)
            if self.bc3[0] == 'free':
                f1_2[:, :,  0] += f2_1[:, :,  0]
            # f1_2(x, y, 1)
            if self.bc3[1] == 'free':
                f1_2[:, :, -1] -= f2_1[:, :, -1]
        
        return f1_1, f1_2, f1_3
    
    # ================================================
    def div_weak(self, f3):
        
        # final 2-form coefficients of the operation DIV.T(f3)
        f2_1 = np.zeros(self.tensor_space.Nbase_2form[0], dtype=float)
        f2_2 = np.zeros(self.tensor_space.Nbase_2form[1], dtype=float)
        f2_3 = np.zeros(self.tensor_space.Nbase_2form[2], dtype=float)
        
        # spline boundary conditions
        bc_splines = [1*self.tensor_space.bc[0], 1*self.tensor_space.bc[1], 1*self.tensor_space.bc[2]]
        
        ker.div_weak(f3, f2_1, f2_2, f2_3, bc_splines)
        
        # boundary conditions (1-direction)
        if self.tensor_space.bc[0] == False:
            
            # f2_1(0, y, z)
            if self.bc1[0] == 'free':
                f2_1[ 0, :, :] -= f3[ 0, :, :]
            # f2_1(1, y, z)
            if self.bc1[1] == 'free':
                f2_1[-1, :, :] += f3[-1, :, :]
            
        # boundary conditions (2-direction)
        if self.tensor_space.bc[1] == False:
            
            # f2_2(x, 0, z)
            if self.bc2[0] == 'free':
                f2_2[:,  0, :] -= f3[:,  0, :]
            # f2_2(x, 1, z)
            if self.bc2[1] == 'free':
                f2_2[:, -1, :] += f3[:, -1, :]
            
        # boundary conditions (3-direction)
        if self.tensor_space.bc[2] == False:
            
            # f2_3(x, y, 0)
            if self.bc3[0] == 'free':
                f2_3[:, :,  0] -= f3[:, :,  0]
            # f2_3(x, y, 1)
            if self.bc3[1] == 'free':
                f2_3[:, :, -1] += f3[:, :, -1]
                
        return f2_1, f2_2, f2_3