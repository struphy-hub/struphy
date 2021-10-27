# coding: utf-8
#
# Copyright 2021 Florian Holderied

"""
Modules to assemble discrete derivatives.
"""

import numpy        as np
import scipy.sparse as spa

import gvec_to_python.hylife.utilities_FEEC.derivatives.kernels_derivatives as ker


# ================== 1d incident matrix =======================
def grad_1d_matrix(spl_kind, NbaseN):
    """
    Returns the 1d strong discrete gradient matrix (V0 --> V1) corresponding either periodic or clamped B-splines (without boundary conditions).
    
    Parameters
    ----------
    spl_kind : boolean
        periodic (True) or clamped (False) B-splines
        
    NbaseN : int
        total number of basis functions in V0
        
    Returns
    -------
    grad : array_like
        strong discrete gradient matrix
    """
    
    NbaseD = NbaseN - 1 + spl_kind
    
    grad = np.zeros((NbaseD, NbaseN), dtype=float)
    
    for i in range(NbaseD):
        grad[i, i] = -1.
        grad[i, (i + 1)%NbaseN] =  1.
            
    return grad



# ===== discrete derivatives in one dimension =================
def discrete_derivatives_1d(space):
    """
    Returns discrete derivatives for a 1d B-spline space.
    
    Parameters
    ----------
    space : spline_space_1d
        1d B-splines space
    """
        
    # 1D discrete derivative
    G = grad_1d_matrix(space.spl_kind, space.NbaseN)

    # boundary conditions in first logical direction
    if space.bc[0] == 'd':
        G = G[:, 1:]
    if space.bc[1] == 'd':
        G = G[:, :-1]
        
    return spa.csr_matrix(G)



# ===== discrete derivatives in three dimensions ==============
def discrete_derivatives_3d(space):
    """
    Returns discrete derivatives for 2d and 3d B-spline spaces.
    
    Parameters
    ----------
    space : tensor_spline_space
        2d or 3d tensor-product B-splines space
    """
        
    # 1d derivatives and number of degrees of freedom in each direction
    grad_1d_1 = space.spaces[0].G.copy()
    grad_1d_2 = space.spaces[1].G.copy()
    
    if space.dim == 3:
        grad_1d_3 = space.spaces[2].G.copy()
    else:    
        grad_1d_3 = 1j*2*np.pi*space.n_tor*spa.identity(1, format='csr')
    
    n1 = grad_1d_1.shape[1]
    n2 = grad_1d_2.shape[1]  
    n3 = grad_1d_3.shape[1]
    
    d1 = grad_1d_1.shape[0]
    d2 = grad_1d_2.shape[0]  
    d3 = grad_1d_3.shape[0]
    
    # standard derivatives
    if space.polar == False:
        
        # discrete grad
        G1  = spa.kron(spa.kron(grad_1d_1, spa.identity(n2)), spa.identity(n3))
        G2  = spa.kron(spa.kron(spa.identity(n1), grad_1d_2), spa.identity(n3))
        G3  = spa.kron(spa.kron(spa.identity(n1), spa.identity(n2)), grad_1d_3)

        G   = [[G1], [G2], [G3]]
        G   = spa.bmat(G, format='csr')
        
        # discrete curl
        C12 = spa.kron(spa.kron(spa.identity(n1), spa.identity(d2)), grad_1d_3)
        C13 = spa.kron(spa.kron(spa.identity(n1), grad_1d_2), spa.identity(d3))

        C21 = spa.kron(spa.kron(spa.identity(d1), spa.identity(n2)), grad_1d_3)
        C23 = spa.kron(spa.kron(grad_1d_1, spa.identity(n2)), spa.identity(d3))

        C31 = spa.kron(spa.kron(spa.identity(d1), grad_1d_2), spa.identity(n3))
        C32 = spa.kron(spa.kron(grad_1d_1, spa.identity(d2)), spa.identity(n3))

        C   = [[None, -C12, C13], [C21, None, -C23], [-C31, C32, None]]
        C   = spa.bmat(C, format='csr')

        # discrete div
        D1  = spa.kron(spa.kron(grad_1d_1, spa.identity(d2)), spa.identity(d3))
        D2  = spa.kron(spa.kron(spa.identity(d1), grad_1d_2), spa.identity(d3))
        D3  = spa.kron(spa.kron(spa.identity(d1), spa.identity(d2)), grad_1d_3)

        D   = [[D1, D2, D3]]
        D   = spa.bmat(D, format='csr')
        
    # polar derivatives
    else:
        
        NbaseN = space.NbaseN
        NbaseD = space.NbaseD
        
        # discrete polar grad ([DN ND NN] x NN)
        G_pol   = space.polar_splines.G.copy()
        G_pol_3 = spa.identity(G_pol.shape[1], format='csr')

        # discrete polar vector curl ([ND DN] x NN)
        VC_pol  = space.polar_splines.VC.copy()

        # discrete polar scalar curl (DD x [DN ND])
        SC_pol  = space.polar_splines.SC.copy()

        C12 = spa.identity(2 + (NbaseN[0] - 2)*NbaseD[1], format='csr')
        C21 = spa.identity(0 + (NbaseD[0] - 1)*NbaseN[1], format='csr')

        # discrete polar div (DD x [ND DN])
        D_pol   = space.polar_splines.D.copy()
        D_pol_3 = spa.identity(D_pol.shape[0], format='csr')

        # boundary condition at eta_1 = 1
        if space.bc[0][1] == 'd':
            G_pol    = G_pol[:-NbaseD[1], :-NbaseN[1]].tocsr()
            G_pol_3  = G_pol_3[:-NbaseN[1], :-NbaseN[1]].tocsr()

            VC_pol_1 = VC_pol[:(2 + (NbaseN[0] - 3)*NbaseD[1]) , :-NbaseN[1]]
            VC_pol_2 = VC_pol[ (2 + (NbaseN[0] - 2)*NbaseD[1]):, :-NbaseN[1]]

            VC_pol   = spa.bmat([[VC_pol_1], [VC_pol_2]], format='csr')

            SC_pol   = SC_pol[:, :-NbaseD[1]].tocsr()

            C12      = C12[:-NbaseD[1], :-NbaseD[1]]

            D_pol_1  = D_pol[:, :(2 + (NbaseN[0] - 3)*NbaseD[1]) ]
            D_pol_2  = D_pol[:,  (2 + (NbaseN[0] - 2)*NbaseD[1]):]

            D_pol    = spa.bmat([[D_pol_1, D_pol_2]], format='csr')

        # final operators
        G = spa.bmat([[spa.kron(G_pol, spa.identity(n3))], [spa.kron(G_pol_3, grad_1d_3)]], format='csr')

        C11 = spa.bmat([[None, -spa.kron(C12, grad_1d_3)], [spa.kron(C21, grad_1d_3), None]])

        C = spa.bmat([[C11 , spa.kron(VC_pol, spa.identity(d3))], [spa.kron(SC_pol, spa.identity(n3)), None]], format='csr')

        D = spa.bmat([[spa.kron(D_pol, spa.identity(d3)), spa.kron(D_pol_3, grad_1d_3)]], format='csr')
        
        
    return G, C, D
