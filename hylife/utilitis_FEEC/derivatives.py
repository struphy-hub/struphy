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
    def grad_3d(self):
        
        G1 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseN[1])), spa.identity(self.NbaseN[2]))
        G2 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), self.grad_1d[1]), spa.identity(self.NbaseN[2]))
        G3 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), spa.identity(self.NbaseN[1])), self.grad_1d[2])

        G  = spa.bmat([[G1], [G2], [G3]], format='csc')

        return G
    
    
    def curl_3d(self):
        
        C12 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), spa.identity(self.NbaseD[1])), self.grad_1d[2])
        C13 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), self.grad_1d[1]), spa.identity(self.NbaseD[2]))
        
        C21 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), spa.identity(self.NbaseN[1])), self.grad_1d[2])
        C23 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseN[1])), spa.identity(self.NbaseD[2]))
        
        C31 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), self.grad_1d[1]), spa.identity(self.NbaseN[2]))
        C32 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseD[1])), spa.identity(self.NbaseN[2]))
        
        C   = spa.bmat([[None, -C12, C13], [C21, None, -C23], [-C31, C32, None]], format='csc')
        
        return C
    
    
    def div_3d(self):
        
        D1 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseD[1])), spa.identity(self.NbaseD[2]))
        D2 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), self.grad_1d[1]), spa.identity(self.NbaseD[2]))
        D3 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), spa.identity(self.NbaseD[1])), self.grad_1d[2])

        D  = spa.bmat([[D1, D2, D3]], format='csc')

        return D
    
    
    
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