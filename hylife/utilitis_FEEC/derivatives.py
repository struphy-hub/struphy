# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to assemble discrete derivatives.
"""

import numpy        as np
import scipy.sparse as spa




# ============== discrete gradient matrix (1d) ==============
def grad_1d(spline_space):
    """
    Returns the 1d discrete gradient matrix corresponding to the given B-spline space of degree p.
    
    Parameters
    ----------
    spline_space : spline_space_1d
        
    Returns
    -------
    grad : array_like
        discrete gradient matrix
    """
    
    NbaseN = spline_space.NbaseN      # total number of basis functions (N)
    bc     = spline_space.bc          # boundary conditions (True : periodic, False : clamped)
    
    
    if bc == True:
        
        grad = np.zeros((NbaseN, NbaseN), dtype=float)
        
        for i in range(NbaseN):
            grad[i, i] = -1.
            if i < NbaseN - 1:
                grad[i, i + 1] = 1.
        grad[-1, 0] = 1.
        
        return grad
    
    else:
        
        grad = np.zeros((NbaseN - 1, NbaseN))
    
        for i in range(NbaseN - 1):        
            grad[i, i] = -1.
            grad[i, i  + 1] = 1.
            
        return grad
    
    
# ===== discrete derivatives in higher dimensions ============
class discrete_derivatives:
    """
    Class for discrete derivatives for 2d and 3d tensor product B-spline spaces.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
    """
    
    def __init__(self, tensor_space):
        
        self.NbaseN  = tensor_space.NbaseN
        self.NbaseD  = tensor_space.NbaseD
        self.bc      = tensor_space.bc
        
        self.grad_1d = [spa.csc_matrix(grad_1d(spl)) for spl in tensor_space.spaces]
        
    
    # ================== 2d ==================
    def grad_2d(self):
        
        G1 = spa.kron(self.grad_1d[0], spa.identity(self.NbaseN[1]))
        G2 = spa.kron(spa.identity(self.NbaseN[0]), self.grad_1d[1])
        
        G  = spa.bmat([[G1], [G2]], format='csc')
        
        return G
    
    
    def curl_2d(self, kind):
        """
        Parameters
        ----------
        kind : string
            'Hcurl' corresponds to the sequence grad --> curl
            'Hdiv'  correpsonds to the sequence curl --> div
        """
        
        if kind == 'Hcurl':
            
            C1 = spa.kron(spa.identity(self.NbaseD[0]), self.grad_1d[1])
            C2 = spa.kron(self.grad_1d[0], spa.identity(self.NbaseD[1]))

            C  = spa.bmat([[-C1, C2]], format='csc')
            
        elif kind == 'Hdiv':
            
            C1 = spa.kron(spa.identity(self.NbaseN[0]), self.grad_1d[1])
            C2 = spa.kron(self.grad_1d[0], spa.identity(self.NbaseN[1]))

            C  = spa.bmat([[C1], [-C2]], format='csc')
            
        return C
            
    
    def div_2d(self):
        
        D1 = spa.kron(self.grad_1d[0], spa.identity(self.NbaseD[1]))
        D2 = spa.kron(spa.identity(self.NbaseD[0]), self.grad_1d[1])
        
        D  = sparse.bmat([[D1, D2]], format='csc')
        
        return D
    
    
    # ================== 3d ==================
    def grad_3d(self, bc_kind=None):
        
        G1 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseN[1])), spa.identity(self.NbaseN[2]))
        G2 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), self.grad_1d[1]), spa.identity(self.NbaseN[2]))
        G3 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), spa.identity(self.NbaseN[1])), self.grad_1d[2])

        G  = spa.bmat([[G1], [G2], [G3]], format='csc')

        return G
    
    
    def curl_3d(self, bc_kind=None):
        
        # apply Dirichlet boundary conditions
        if self.bc[0] == False:
            g1        = self.grad_1d[0].copy().tolil()
            id1       = spa.identity(self.NbaseN[0], format='lil')
            
            if bc_kind[0][0] == 'dirichlet':
                g1[:,  0] = 0.
                id1[0, 0] = 0.
                
            if bc_kind[0][1] == 'dirichlet':
                g1[:, -1] = 0.
                id1[-1, -1] = 0.
        
        C12 = spa.kron(spa.kron(id1, spa.identity(self.NbaseD[1])), self.grad_1d[2])
        C13 = spa.kron(spa.kron(id1, self.grad_1d[1]), spa.identity(self.NbaseD[2]))
        
        C21 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), spa.identity(self.NbaseN[1])), self.grad_1d[2])
        C23 = spa.kron(spa.kron(g1             , spa.identity(self.NbaseN[1])), spa.identity(self.NbaseD[2]))
        
        C31 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), self.grad_1d[1]), spa.identity(self.NbaseN[2]))
        C32 = spa.kron(spa.kron(g1             , spa.identity(self.NbaseD[1])), spa.identity(self.NbaseN[2]))
        
        C   = spa.bmat([[None, -C12, C13], [C21, None, -C23], [-C31, C32, None]], format='csc')
        
        return C
    
    
    def div_3d(self, bc_kind=None):
        
        D1 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseD[1])), spa.identity(self.NbaseD[2]))
        D2 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), self.grad_1d[1]), spa.identity(self.NbaseD[2]))
        D3 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), spa.identity(self.NbaseD[1])), self.grad_1d[2])

        D  = spa.bmat([[D1, D2, D3]], format='csc')

        return D     

    
    def apply_GRAD_3d_kron(self,vec3d):
        '''
        apply the divergence operator in tensor-product fashion with 3d vectors

        res^1_mjk = G^1_mi vec3d_ijk 
        res^2_ink = G^2_nj vec3d_ijk
        res^3_ijo = G^3_ok vec3d_ijk

        Parameters
        ----------
        self.grad_1d  : 3 sparse matrices representing the 1D gradient matrix in each direction, 
                        of size (d0,n0),(d1,n1),(d2,n2)

        vec3d : 3d array of size (n0,n1,n2)
        Returns
        -------
        grad_1 : 3d array of 1st component of size (d0,n1,n2)   
        grad_2 : 3d array of 2nd component of size (n0,d1,n2)   
        grad_3 : 3d array of 3rd component of size (n0,n1,d2)   
        '''
        d0 , n0 = self.grad_1d[0].shape
        d1 , n1 = self.grad_1d[1].shape
        d2 , n2 = self.grad_1d[2].shape

        assert ( vec3d.shape == (n0, n1, n2) ) 
        
        grad_1 =   (self.grad_1d[0].dot(  vec3d.reshape(n0,n1*n2))                                            ).reshape(d0,n1,n2)
        grad_2 = (((self.grad_1d[1].dot(((vec3d.reshape(n0,n1*n2)).T).reshape(n1,n2*n0))).reshape(d1*n2,n0)).T).reshape(n0,d1,n2)
        grad_3 = ( (self.grad_1d[2].dot(( vec3d.reshape(n0*n1,n2)).T)).T                                      ).reshape(n0,n1,d2)

        return grad_1,grad_2,grad_3     


    def apply_CURL_3d_kron(self,vec3d_1,vec3d_2,vec3d_3):
        '''
        apply the divergence operator in tensor-product fashion with 3d vectors

        curl3d^1_ino = G^2_nj vec3d^3_ijo - G^3_ok vec3d^2_ink
        curl3d^2_mjo = G^3_ok vec3d^1_mjk - G^1_mi vec3d^3_ijo
        curl3d^3_mnk = G^1_mi vec3d^2_ink - G^2_nj vec3d^1_mjk

        Parameters
        ----------
        self.grad_1d  : 3 sparse matrices representing the 1D gradient matrix in each direction, 
                  of size (d0,n0),(d1,n1),(d2,n2)
            
        vec3d_1 : 3d array of 1st component of size (d0,n1,n2)   
        vec3d_2 : 3d array of 2nd component of size (n0,d1,n2)   
        vec3d_3 : 3d array of 3rd component of size (n0,n1,d2)   

        Returns
        -------
        curl3d_1 : 3d array of 1st component of size (n0,d1,d2)   
        curl3d_2 : 3d array of 2nd component of size (d0,n1,d2)   
        curl3d_3 : 3d array of 3rd component of size (d0,d1,n2)   
        '''
        d0 , n0 = self.grad_1d[0].shape
        d1 , n1 = self.grad_1d[1].shape
        d2 , n2 = self.grad_1d[2].shape

        assert ( vec3d_1.shape == (d0, n1, n2) ) 
        assert ( vec3d_2.shape == (n0, d1, n2) ) 
        assert ( vec3d_3.shape == (n0, n1, d2) ) 
        
        curl3d_1 =( (((self.grad_1d[1].dot(((vec3d_3.reshape(n0,n1*d2)).T).reshape(n1,d2*n0))).reshape(d1*d2,n0)).T).reshape(n0,d1,d2) \
                   -( (self.grad_1d[2].dot(( vec3d_2.reshape(n0*d1,n2)).T)).T                                      ).reshape(n0,d1,d2) )

        curl3d_2 =( ( (self.grad_1d[2].dot(( vec3d_1.reshape(d0*n1,n2)).T)).T                                      ).reshape(d0,n1,d2) \
                   -  (self.grad_1d[0].dot(  vec3d_3.reshape(n0,n1*d2))                                            ).reshape(d0,n1,d2) )

        curl3d_3 =(   (self.grad_1d[0].dot(  vec3d_2.reshape(n0,d1*n2))                                            ).reshape(d0,d1,n2) \
                   -(((self.grad_1d[1].dot(((vec3d_1.reshape(d0,n1*n2)).T).reshape(n1,n2*d0))).reshape(d1*n2,d0)).T).reshape(d0,d1,n2) )

        return curl3d_1,curl3d_2,curl3d_3     

    
    def apply_DIV_3d_kron(self,vec3d_1,vec3d_2,vec3d_3):
        '''
        apply the divergence operator in tensor-product fashion with 3d vectors

        div3d_mno = G_mi vec3d^1_ino + G_nj vec3d^2_mjo + G_ok vec3d^3_mnk

        Parameters
        ----------
        self.grad_1d  : 3 sparse matrices representing the 1D gradient matrix in each direction, 
                  of size (d0,n0),(d1,n1),(d2,n2)
            
        vec3d_1 : 3d array of 1st component of size (n0,d1,d2)   
        vec3d_2 : 3d array of 2nd component of size (d0,n1,d2)   
        vec3d_3 : 3d array of 3rd component of size (d0,d1,n2)   

        Returns
        -------
        div3d : 3d array of size (d0,d1,d2)
        '''

        d0 , n0 = self.grad_1d[0].shape
        d1 , n1 = self.grad_1d[1].shape
        d2 , n2 = self.grad_1d[2].shape

        assert ( vec3d_1.shape == (n0, d1, d2) ) 
        assert ( vec3d_2.shape == (d0, n1, d2) ) 
        assert ( vec3d_3.shape == (d0, d1, n2) ) 
        
        div3d=(   (self.grad_1d[0].dot(  vec3d_1.reshape(n0,d1*d2))                                            ).reshape(d0,d1,d2) \
               +(((self.grad_1d[1].dot(((vec3d_2.reshape(d0,n1*d2)).T).reshape(n1,d2*d0))).reshape(d1*d2,d0)).T).reshape(d0,d1,d2) \
               +( (self.grad_1d[2].dot(( vec3d_3.reshape(d0*d1,n2)).T)).T                                      ).reshape(d0,d1,d2) )

        return div3d
    
    
    
# ============== discrete gradient matrix (1d) for arbitrary number of basis functions ==============
def grad_1d_ar(NbaseN, bc):
    """
    Returns the 1d discrete gradient matrix corresponding to the given B-spline space of degree p.
    
    Parameters
    ----------
    NbaseN : int 
        number of basis functions in first space
    
    bc : boolean
        True : periodic, False : clamped
        
    Returns
    -------
    grad : array_like
        discrete gradient matrix
    """
    
    
    if bc == True:
        
        grad = np.zeros((NbaseN, NbaseN), dtype=float)
        
        for i in range(NbaseN):
            grad[i, i] = -1.
            if i < NbaseN - 1:
                grad[i, i + 1] = 1.
        grad[-1, 0] = 1.
        
        return grad
    
    else:
        
        grad = np.zeros((NbaseN - 1, NbaseN))
    
        for i in range(NbaseN - 1):        
            grad[i, i] = -1.
            grad[i, i  + 1] = 1.
            
        return grad
