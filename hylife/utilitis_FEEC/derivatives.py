import numpy as np
import hylife.utilitis_FEEC.bsplines as bsp

from scipy import sparse



#=============================================== discrete gradient matrix (1d) ================================================
def GRAD_1d(T, p, bc):
    """
    Returns the 1d discrete gradient matrix.
    
    Parameters
    ----------
    p : int
        spline degree
        
    Nbase : int
        number of spline functions
        
    bc : boolean
        boundary conditions (True = periodic, False = else)
        
    Returns
    -------
    G: 2d np.array
        discrete gradient matrix
    """
    
    el_b      = bsp.breakpoints(T, p)
    Nel       = len(el_b) - 1
    NbaseN    = Nel + p - bc*p
    
    
    if bc == True:
        
        G = np.zeros((NbaseN, NbaseN))
        
        for i in range(NbaseN):
            
            G[i, i] = -1.
            
            if i < NbaseN - 1:
                G[i, i + 1] = 1.
                
        G[-1, 0] = 1.
        
        return G
    
    
    else:
        
        G = np.zeros((NbaseN - 1, NbaseN))
    
        for i in range(NbaseN - 1):
            
            G[i, i] = -1.
            G[i, i  + 1] = 1.
            
        return G
#==============================================================================================================================


    
    
    

#=============================================== discrete derivatives in higher dimensions ====================================
class discrete_derivatives:
    
    def __init__(self, T, p, bc):
        
        self.el_b    = [bsp.breakpoints(T_i, p_i) for T_i, p_i in zip(T, p)]
        self.Nel     = [len(el_b) - 1 for el_b in self.el_b]
        
        self.NbaseN  = [Nel + p_i - bc_i*p_i for Nel, p_i, bc_i in zip(self.Nel, p, bc)]
        self.NbaseD  = [NbaseN - (1 - bc_i) for NbaseN, bc_i in zip(self.NbaseN, bc)]
        
        self.grad_1d = [sparse.csr_matrix(GRAD_1d(T_i, p_i, bc_i)) for T_i, p_i, bc_i in zip(T, p, bc)]
        
        
    
    def GRAD_2d(self):
        '''
        corresponds to diagram grad --> curl
        '''
        
        G1 = sparse.kron(self.grad_1d[0], sparse.identity(self.NbaseN[1]))
        G2 = sparse.kron(sparse.identity(self.NbaseN[0]), self.grad_1d[1])
        
        G  = sparse.bmat([[G1], [G2]], format='csr')
        
        return G
        
    
    
    def CURL_2d_vector(self):
        '''
        corresponds to diagram grad --> curl
        '''
        
        C1 = sparse.kron(sparse.identity(self.NbaseD[0]), self.grad_1d[1])
        C2 = sparse.kron(self.grad_1d[0], sparse.identity(self.NbaseD[1]))
        
        C  = sparse.bmat([[-C1, C2]], format='csr')
        
        return C
    
    
    
    def CURL_2d_scalar(self):
        '''
        corresponds to diagram curl --> div
        '''
        
        C1 = sparse.kron(sparse.identity(self.NbaseN[0]), self.grad_1d[1])
        C2 = sparse.kron(self.grad_1d[0], sparse.identity(self.NbaseN[1]))
        
        C  = sparse.bmat([[C1], [-C2]], format='csr')
        
        return C
        
        
    
    def DIV_2d(self):
        '''
        corresponds to diagram curl --> div
        '''
        
        D1 = sparse.kron(self.grad_1d[0], sparse.identity(self.NbaseD[1]))
        D2 = sparse.kron(sparse.identity(self.NbaseD[0]), self.grad_1d[1])
        
        D  = sparse.bmat([[D1, D2]], format='csr')
        
        return D
        
    
    
    def GRAD_3d(self):
        '''
        corresponds to diagram grad --> curl --> div
        '''
        
        G1 = sparse.kron(sparse.kron(self.grad_1d[0], sparse.identity(self.NbaseN[1])), sparse.identity(self.NbaseN[2]))
        G2 = sparse.kron(sparse.kron(sparse.identity(self.NbaseN[0]), self.grad_1d[1]), sparse.identity(self.NbaseN[2]))
        G3 = sparse.kron(sparse.kron(sparse.identity(self.NbaseN[0]), sparse.identity(self.NbaseN[1])), self.grad_1d[2])

        G  = sparse.bmat([[G1], [G2], [G3]], format='csr')

        return G
    
    
    
    def CURL_3d(self):
        '''
        corresponds to diagram grad --> curl --> div
        '''
        
        C12 = sparse.kron(sparse.kron(sparse.identity(self.NbaseN[0]), sparse.identity(self.NbaseD[1])), self.grad_1d[2])
        C13 = sparse.kron(sparse.kron(sparse.identity(self.NbaseN[0]), self.grad_1d[1]), sparse.identity(self.NbaseD[2]))
        
        C21 = sparse.kron(sparse.kron(sparse.identity(self.NbaseD[0]), sparse.identity(self.NbaseN[1])), self.grad_1d[2])
        C23 = sparse.kron(sparse.kron(self.grad_1d[0], sparse.identity(self.NbaseN[1])), sparse.identity(self.NbaseD[2]))
        
        C31 = sparse.kron(sparse.kron(sparse.identity(self.NbaseD[0]), self.grad_1d[1]), sparse.identity(self.NbaseN[2]))
        C32 = sparse.kron(sparse.kron(self.grad_1d[0], sparse.identity(self.NbaseD[1])), sparse.identity(self.NbaseN[2]))
        
        C   = sparse.bmat([[None, -C12, C13], [C21, None, -C23], [-C31, C32, None]], format='csr')
        
        return C
        
    
    
    def DIV_3d(self):
        '''
        corresponds to diagram grad --> curl --> div
        '''
        
        D1 = sparse.kron(sparse.kron(self.grad_1d[0], sparse.identity(self.NbaseD[1])), sparse.identity(self.NbaseD[2]))
        D2 = sparse.kron(sparse.kron(sparse.identity(self.NbaseD[0]), self.grad_1d[1]), sparse.identity(self.NbaseD[2]))
        D3 = sparse.kron(sparse.kron(sparse.identity(self.NbaseD[0]), sparse.identity(self.NbaseD[1])), self.grad_1d[2])

        D  = sparse.bmat([[D1, D2, D3]], format='csr')

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
#==============================================================================================================================
