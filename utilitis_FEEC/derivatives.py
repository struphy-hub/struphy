import numpy                  as np
import utilitis_FEEC.bsplines as bsp

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
        
        self.el_b    = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
        self.Nel     = [len(el_b) - 1 for el_b in self.el_b]
        
        self.NbaseN  = [Nel + p - bc*p for Nel, p, bc in zip(self.Nel, p, bc)]
        self.NbaseD  = [NbaseN - (1 - bc) for NbaseN, bc in zip(self.NbaseN, bc)]
        
        self.grad_1d = [sparse.csr_matrix(GRAD_1d(T, p, bc)) for T, p, bc in zip(T, p, bc)]
        
        
    
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
        
    
    
    def GRAD_3d(self, T, p, bc):
        '''
        corresponds to diagram grad --> curl --> div
        '''
        
        G1 = sparse.kron(sparse.kron(self.grad_1d[0], sparse.identity(self.NbaseN[1])), sparse.identity(self.NbaseN[2]))
        G2 = sparse.kron(sparse.kron(sparse.identity(self.NbaseN[0]), self.grad_1d[1]), sparse.identity(self.NbaseN[2]))
        G3 = sparse.kron(sparse.kron(sparse.identity(self.NbaseN[0]), sparse.identity(self.NbaseN[1])), self.grad_1d[2])

        G  = sparse.bmat([[G1], [G2], [G3]], format='csr')

        return G
    
    
    
    def CURL_3d(self, T, p, bc):
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
        
    
    
    def DIV_3d(self, T, p, bc):
        '''
        corresponds to diagram grad --> curl --> div
        '''
        
        D1 = sparse.kron(sparse.kron(self.grad_1d[0], sparse.identity(self.NbaseD[1])), sparse.identity(self.NbaseD[2]))
        D2 = sparse.kron(sparse.kron(sparse.identity(self.NbaseD[0]), self.grad_1d[1]), sparse.identity(self.NbaseD[2]))
        D3 = sparse.kron(sparse.kron(sparse.identity(self.NbaseD[0]), sparse.identity(self.NbaseD[1])), self.grad_1d[2])

        D  = sparse.bmat([[D1, D2, D3]], format='csr')

        return D     
#==============================================================================================================================