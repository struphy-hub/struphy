# coding: utf-8
#
# Copyright 2021 Florian Holderied

"""
Modules to assemble discrete derivatives.
"""

import numpy        as np
import scipy.sparse as spa

import struphy.feec.derivatives.kernels_derivatives as ker


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
    space : Spline_space_1d
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
    space : Tensor_spline_space
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




    
## ===== discrete derivatives in one dimension =================
#class discrete_derivatives_1D:
#    """
#    Class for discrete derivatives for a 1d B-spline space.
#    
#    Parameters
#    ----------
#    space : Spline_space_1d
#        1d B-splines space
#    """
#    
#    def __init__(self, space):
#        
#        self.space = space
#        
#        # 1D discrete derivative
#        self.grad_1d = spa.csr_matrix(grad_1d_matrix(self.space.spl_kind, self.space.NbaseN))
#        
#        # boundary conditions in first logical direction
#        if self.space.bc[0] == 'd':
#            self.grad_1d = self.grad_1d[:, 1:]
#        if self.space.bc[1] == 'd':
#            self.grad_1d = self.grad_1d[:, :-1]
#
#
#    
## ===== discrete derivatives in two dimensions ================
#class discrete_derivatives_2D:
#    """
#    Class for discrete derivatives for a 2d tensor product B-spline space with Fourier decomposition in third dimension.
#    
#    Parameters
#    ----------
#    tensor_space : Tensor_spline_space
#        2d tensor-product B-splines space
#    
#    n_tor : int
#        toroidal mode number
#    """
#    
#    def __init__(self, tensor_space, n_tor):
#        
#        self.space = tensor_space
#        
#        # 1D discrete derivatives in each logical direction
#        self.grad_1d    = [0, 0]
#        self.grad_1d[0] = spa.csr_matrix(grad_1d_matrix(self.space.spl_kind[0], self.space.NbaseN[0]))
#        self.grad_1d[1] = spa.csr_matrix(grad_1d_matrix(self.space.spl_kind[1], self.space.NbaseN[1]))
#        
#        # number of degrees of freedom in each direction
#        n1, n2 = self.space.NbaseN
#        
#        # boundary conditions in first logical direction
#        if self.space.bc[0][0] == 'd':
#            self.grad_1d[0] = self.grad_1d[0][:, 1:]
#            n1 -= 1
#        if self.space.bc[0][1] == 'd':
#            self.grad_1d[0] = self.grad_1d[0][:, :-1]
#            n1 -= 1
#        
#        NbaseN = self.space.NbaseN
#        NbaseD = self.space.NbaseD
#        
#        # standard derivatives
#        if self.space.polar == False:
#            
#            # discrete grad
#            G1 = spa.kron(self.grad_1d[0], spa.identity(n2))
#            G2 = spa.kron(spa.identity(n1), self.grad_1d[1])
#
#            self.G = [[G1], [G2]]
#            self.G = spa.bmat(self.G, format='csr')
#            
#            self.G3 = 1j*2*np.pi*n_tor*spa.identity(n1*n2, format='csr')
#            
#            self.G = spa.bmat([[self.G], [self.G3]], format='csr')
#            
#            # discrete vector curl
#            C1 = spa.kron(spa.identity(n1), self.grad_1d[1])
#            C2 = spa.kron(self.grad_1d[0], spa.identity(n2))
#
#            self.VC = [[C1], [-C2]]
#            self.VC = spa.bmat(self.VC, format='csr')
#            
#            # discrete scalar curl
#            C1 = spa.kron(spa.identity(NbaseD[0]), self.grad_1d[1])
#            C2 = spa.kron(self.grad_1d[0], spa.identity(NbaseD[1]))
#
#            self.SC = [[-C1, C2]]
#            self.SC = spa.bmat(self.SC, format='csr')
#            
#            self.C12 = 1j*2*np.pi*n_tor*spa.identity(n1*NbaseD[1], format='csr')
#            self.C21 = 1j*2*np.pi*n_tor*spa.identity(NbaseD[0]*n2, format='csr')
#            
#            self.C11 = spa.bmat([[None, -self.C12], [self.C21, None]], format='csr')
#            
#            self.C   = spa.bmat([[self.C11, self.VC], [self.SC, None]], format='csr')
#            
#            # discrete div
#            D1 = spa.kron(self.grad_1d[0], spa.identity(NbaseD[1]))
#            D2 = spa.kron(spa.identity(NbaseD[0]), self.grad_1d[1])
#
#            self.D = [[D1, D2]]
#            self.D = spa.bmat(self.D, format='csr')
#            
#            self.D3 = 1j*2*np.pi*n_tor*spa.identity(NbaseD[0]*NbaseD[1], format='csr')
#            
#            self.D = spa.bmat([[self.D, self.D3]], format='csr')
#            
#        # polar derivatives
#        else:
#            
#            # discrete polar grad ([DN ND NN] x NN)
#            self.G   = self.space.polar_splines.G.copy()
#            self.G3  = 1j*2*np.pi*n_tor*spa.identity(self.G.shape[1], format='csr')
#            
#            # discrete polar vector curl ([ND DN] x NN)
#            self.VC  = self.space.polar_splines.VC.copy()
#            
#            # discrete polar scalar curl (DD x [DN ND])
#            self.SC  = self.space.polar_splines.SC.copy()
#            
#            self.C12 = 1j*2*np.pi*n_tor*spa.identity(2 + (NbaseN[0] - 2)*NbaseD[1], format='csr')
#            self.C21 = 1j*2*np.pi*n_tor*spa.identity(0 + (NbaseD[0] - 1)*NbaseN[1], format='csr')
#            
#            # discrete polar div (DD x [ND DN DD])
#            self.D   = self.space.polar_splines.D.copy()
#            self.D3  = 1j*2*np.pi*n_tor*spa.identity(self.SC.shape[0], format='csr')
#            
#            # boundary condition at eta1 = 1
#            if self.space.bc[0][1] == 'd':
#                self.G  = self.G[:-NbaseD[1], :-NbaseN[1]].tocsr()
#                self.G3 = self.G3[:-NbaseN[1], :-NbaseN[1]].tocsr()
#                
#                VC1     = self.VC[:(2 + (NbaseN[0] - 3)*NbaseD[1]) , :-NbaseN[1]]
#                VC2     = self.VC[ (2 + (NbaseN[0] - 2)*NbaseD[1]):, :-NbaseN[1]]
#
#                self.VC = spa.bmat([[VC1], [VC2]], format='csr')
#                
#                self.SC = self.SC[:, :-NbaseD[1]].tocsr()
#                
#                self.C12 = self.C12[:-NbaseD[1], :-NbaseD[1]]
#                
#                D1      = self.D[:, :(2 + (NbaseN[0] - 3)*NbaseD[1]) ]
#                D2      = self.D[:,  (2 + (NbaseN[0] - 2)*NbaseD[1]):]
#
#                self.D  = spa.bmat([[D1, D2]], format='csr')
#            
#            # final operators
#            self.G   = spa.bmat([[self.G], [self.G3]], format='csr')
#            
#            self.C11 = spa.bmat([[None, -self.C12], [self.C21, None]], format='csr')
#            self.C   = spa.bmat([[self.C11, self.VC], [self.SC, None]], format='csr')
#            
#            self.D   = spa.bmat([[self.D ,  self.D3]], format='csr')
#                
#            
#            
## ===== discrete derivatives in three dimensions ================
#class discrete_derivatives_3D:
#    """
#    Class for discrete derivatives for a 3d tensor product B-spline space.
#    
#    Parameters
#    ----------
#    tensor_space : Tensor_spline_space
#        3d tensor-product B-splines space
#    """
#    
#    def __init__(self, tensor_space):
#        
#        self.space = tensor_space
#        
#        # 1D discrete derivatives in each logical direction
#        self.grad_1d    = [0, 0, 0]
#        self.grad_1d[0] = spa.csr_matrix(grad_1d_matrix(self.space.spl_kind[0], self.space.NbaseN[0]))
#        self.grad_1d[1] = spa.csr_matrix(grad_1d_matrix(self.space.spl_kind[1], self.space.NbaseN[1]))
#        self.grad_1d[2] = spa.csr_matrix(grad_1d_matrix(self.space.spl_kind[2], self.space.NbaseN[2]))
#        
#        # number of degrees of freedom in each direction
#        n1, n2, n3 = self.space.NbaseN
#        
#        # boundary conditions in first logical direction
#        if self.space.bc[0][0] == 'd':
#            self.grad_1d[0] = self.grad_1d[0][:, 1:]
#            n1 -= 1
#        if self.space.bc[0][1] == 'd':
#            self.grad_1d[0] = self.grad_1d[0][:, :-1]
#            n1 -= 1
#        
#        NbaseN = self.space.NbaseN
#        NbaseD = self.space.NbaseD
#        
#        # standard derivatives
#        if self.space.polar == False:
#            
#            # discrete grad
#            G1 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(n2)), spa.identity(n3))
#            G2 = spa.kron(spa.kron(spa.identity(n1), self.grad_1d[1]), spa.identity(n3))
#            G3 = spa.kron(spa.kron(spa.identity(n1), spa.identity(n2)), self.grad_1d[2])
#
#            self.G = [[G1], [G2], [G3]]
#            self.G = spa.bmat(self.G, format='csr')
#            
#            # discrete curl
#            C12 = spa.kron(spa.kron(spa.identity(n1), spa.identity(NbaseD[1])), self.grad_1d[2])
#            C13 = spa.kron(spa.kron(spa.identity(n1), self.grad_1d[1]), spa.identity(NbaseD[2]))
#
#            C21 = spa.kron(spa.kron(spa.identity(NbaseD[0]), spa.identity(n2)), self.grad_1d[2])
#            C23 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(n2)), spa.identity(NbaseD[2]))
#
#            C31 = spa.kron(spa.kron(spa.identity(NbaseD[0]), self.grad_1d[1]), spa.identity(n3))
#            C32 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(NbaseD[1])), spa.identity(n3))
#
#            self.C = [[None, -C12, C13], [C21, None, -C23], [-C31, C32, None]]
#            self.C = spa.bmat(self.C, format='csr')
#            
#            # discrete div
#            D1 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(NbaseD[1])), spa.identity(NbaseD[2]))
#            D2 = spa.kron(spa.kron(spa.identity(NbaseD[0]), self.grad_1d[1]), spa.identity(NbaseD[2]))
#            D3 = spa.kron(spa.kron(spa.identity(NbaseD[0]), spa.identity(NbaseD[1])), self.grad_1d[2])
#
#            self.D = [[D1, D2, D3]]
#            self.D = spa.bmat(self.D, format='csr')
#            
#        # polar derivatives
#        else:
#            
#            # discrete polar grad ([DN ND NN] x NN)
#            G_pol   = self.space.polar_splines.G.copy()
#            G_pol_3 = spa.identity(G_pol.shape[1], format='csr')
#            
#            # discrete polar vector curl ([ND DN] x NN)
#            VC_pol  = self.space.polar_splines.VC.copy()
#            
#            # discrete polar scalar curl (DD x [DN ND])
#            SC_pol  = self.space.polar_splines.SC.copy()
#            
#            C12 = spa.identity(2 + (NbaseN[0] - 2)*NbaseD[1], format='csr')
#            C21 = spa.identity(0 + (NbaseD[0] - 1)*NbaseN[1], format='csr')
#            
#            # discrete polar div (DD x [ND DN])
#            D_pol   = self.space.polar_splines.D.copy()
#            D_pol_3 = spa.identity(D_pol.shape[0], format='csr')
#            
#            # boundary condition at eta1 = 1
#            if self.space.bc[0][1] == 'd':
#                G_pol    = G_pol[:-NbaseD[1], :-NbaseN[1]].tocsr()
#                G_pol_3  = G_pol_3[:-NbaseN[1], :-NbaseN[1]].tocsr()
#                
#                VC_pol_1 = VC_pol[:(2 + (NbaseN[0] - 3)*NbaseD[1]) , :-NbaseN[1]]
#                VC_pol_2 = VC_pol[ (2 + (NbaseN[0] - 2)*NbaseD[1]):, :-NbaseN[1]]
#
#                VC_pol   = spa.bmat([[VC_pol_1], [VC_pol_2]], format='csr')
#                
#                SC_pol   = SC_pol[:, :-NbaseD[1]].tocsr()
#                
#                C12      = C12[:-NbaseD[1], :-NbaseD[1]]
#                
#                D_pol_1  = D_pol[:, :(2 + (NbaseN[0] - 3)*NbaseD[1]) ]
#                D_pol_2  = D_pol[:,  (2 + (NbaseN[0] - 2)*NbaseD[1]):]
# 
#                D_pol    = spa.bmat([[D_pol_1, D_pol_2]], format='csr')
#            
#            # final operators
#            self.G = spa.bmat([[spa.kron(G_pol, spa.identity(NbaseN[2]))], [spa.kron(G_pol_3, self.grad_1d[2])]], format='csr')
#            
#            C11    = spa.bmat([[None, -spa.kron(C12, self.grad_1d[2])], [spa.kron(C21, self.grad_1d[2]), None]])
#            
#            self.C = spa.bmat([[C11, spa.kron(VC_pol, spa.identity(NbaseD[2]))], 
#                               [spa.kron(SC_pol, spa.identity(NbaseN[2])), None]], format='csr')
#            
#            self.D = spa.bmat([[spa.kron(D_pol, spa.identity(NbaseD[2])), spa.kron(D_pol_3, self.grad_1d[2])]], format='csr')
#        
#    
#    # ================== grad ==================
#    def get_G(self):
#        
#        G1 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseN[1])), spa.identity(self.NbaseN[2]))
#        G2 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), self.grad_1d[1]), spa.identity(self.NbaseN[2]))
#        G3 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), spa.identity(self.NbaseN[1])), self.grad_1d[2])
#
#        G = [[G1], [G2], [G3]]
#        G = spa.bmat(G, format='csr')
#
#        return G
#    
#    # ================== curl ==================
#    def get_C(self):
#        
#        C12 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), spa.identity(self.NbaseD[1])), self.grad_1d[2])
#        C13 = spa.kron(spa.kron(spa.identity(self.NbaseN[0]), self.grad_1d[1]), spa.identity(self.NbaseD[2]))
#
#        C21 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), spa.identity(self.NbaseN[1])), self.grad_1d[2])
#        C23 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseN[1])), spa.identity(self.NbaseD[2]))
#
#        C31 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), self.grad_1d[1]), spa.identity(self.NbaseN[2]))
#        C32 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseD[1])), spa.identity(self.NbaseN[2]))
#
#        C = [[None, -C12, C13], [C21, None, -C23], [-C31, C32, None]]
#        C = spa.bmat(C, format='csr')
#        
#        return C
#    
#    # ================== div ==================
#    def get_D(self):
#        
#        D1 = spa.kron(spa.kron(self.grad_1d[0], spa.identity(self.NbaseD[1])), spa.identity(self.NbaseD[2]))
#        D2 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), self.grad_1d[1]), spa.identity(self.NbaseD[2]))
#        D3 = spa.kron(spa.kron(spa.identity(self.NbaseD[0]), spa.identity(self.NbaseD[1])), self.grad_1d[2])
#
#        D = [[D1, D2, D3]]
#        D = spa.bmat(D, format='csr')
#
#        return D     
#
#    
#    # ========================================
#    def apply_G_kron(self, f0):
#        """
#        apply the disrete gradient operator in tensor-product fashion with 3d vectors
#
#        f1^1_mjk = G^1_mi f0_ijk 
#        f1^2_ink = G^2_nj f0_ijk
#        f1^3_ijo = G^3_ok f0_ijk
#
#        Parameters
#        ----------
#        self.grad_1d : 3 sparse matrices representing the 1D gradient matrix in each direction, 
#                       of size (d0, n0), (d1, n1), (d2, n2)
#
#        f0 : 3d array of size (n0, n1, n2)
#        
#        Returns
#        -------
#        f1_1 : 3d array of 1st component of size (d0, n1, n2)   
#        f1_2 : 3d array of 2nd component of size (n0, d1, n2)   
#        f1_3 : 3d array of 3rd component of size (n0, n1, d2)   
#        """
#        
#        d0, n0 = self.grad_1d[0].shape
#        d1, n1 = self.grad_1d[1].shape
#        d2, n2 = self.grad_1d[2].shape
#
#        assert (f0.shape == (n0, n1, n2))
#        
#        f1_1 = self.grad_1d[0].dot(f0.reshape(n0, n1*n2))
#        f1_2 = self.grad_1d[1].dot(f0.reshape(n0, n1*n2).T.reshape(n1, n2*n0)).reshape(d1*n2, n0).T
#        f1_3 = self.grad_1d[2].dot(f0.reshape(n0*n1, n2).T).T
#
#        return f1_1.reshape(d0, n1, n2), f1_2.reshape(n0, d1, n2), f1_3.reshape(n0, n1, d2)     
#
#
#    # ==========================================
#    def apply_C_kron(self, f1_1, f1_2, f1_3):
#        """
#        apply the discrete curl operator in tensor-product fashion with 3d vectors
#
#        f2^1_ino = G^2_nj f1^3_ijo - G^3_ok f1^2_ink
#        f2^2_mjo = G^3_ok f1^1_mjk - G^1_mi f1^3_ijo
#        f2^3_mnk = G^1_mi f1^2_ink - G^2_nj f1^1_mjk
#
#        Parameters
#        ----------
#        self.grad_1d : 3 sparse matrices representing the 1D gradient matrix in each direction, 
#                       of size (d0, n0), (d1, n1), (d2, n2)
#            
#        f1_1 : 3d array of 1st component of size (d0, n1, n2)   
#        f1_2 : 3d array of 2nd component of size (n0, d1, n2)   
#        f1_3 : 3d array of 3rd component of size (n0, n1, d2)   
#
#        Returns
#        -------
#        f2_1 : 3d array of 1st component of size (n0, d1, d2)   
#        f2_2 : 3d array of 2nd component of size (d0, n1, d2)   
#        f2_3 : 3d array of 3rd component of size (d0, d1, n2)   
#        """
#        
#        d0, n0 = self.grad_1d[0].shape
#        d1, n1 = self.grad_1d[1].shape
#        d2, n2 = self.grad_1d[2].shape
#
#        assert (f1_1.shape == (d0, n1, n2)) 
#        assert (f1_2.shape == (n0, d1, n2)) 
#        assert (f1_3.shape == (n0, n1, d2))
#        
#        f2_1  = np.zeros((n0, d1, d2), dtype=float)
#        f2_2  = np.zeros((d0, n1, d2), dtype=float)
#        f2_3  = np.zeros((d0, d1, n2), dtype=float)
#        
#        
#        # 1st component
#        f2_1 += self.grad_1d[1].dot(f1_3.reshape(n0, n1*d2).T.reshape(n1, d2*n0)).reshape(d1*d2, n0).T.reshape(n0, d1, d2)
#        
#        f2_1 -= self.grad_1d[2].dot(f1_2.reshape(n0*d1, n2).T).T.reshape(n0, d1, d2)
#        
#        # 2nd component
#        f2_2 += self.grad_1d[2].dot(f1_1.reshape(d0*n1, n2).T).T.reshape(d0, n1, d2)
#        
#        f2_2 -= self.grad_1d[0].dot(f1_3.reshape(n0, n1*d2)).reshape(d0, n1, d2)
#
#        # 3rd omponent
#        f2_3 += self.grad_1d[0].dot(f1_2.reshape(n0, d1*n2)).reshape(d0, d1, n2)
#        
#        f2_3 -= self.grad_1d[1].dot(f1_1.reshape(d0, n1*n2).T.reshape(n1, n2*d0)).reshape(d1*n2, d0).T.reshape(d0, d1, n2)
#        
#        return f2_1, f2_2, f2_3     
#
#    
#    # ===========================================
#    def apply_D_kron(self, f2_1, f2_2, f2_3):
#        """
#        apply the discrete divergence operator in tensor-product fashion with 3d vectors
#
#        f3_mno = G_mi f2^1_ino + G_nj f2^2_mjo + G_ok f2^3_mnk
#
#        Parameters
#        ----------
#        self.grad_1d : 3 sparse matrices representing the 1D gradient matrix in each direction, 
#                       of size (d0, n0), (d1, n1), (d2, n2)
#            
#        f2_1 : 3d array of 1st component of size (n0, d1, d2)   
#        f2_2 : 3d array of 2nd component of size (d0, n1, d2)   
#        f2_3 : 3d array of 3rd component of size (d0, d1, n2)   
#
#        Returns
#        -------
#        f3 : 3d array of size (d0, d1, d2)
#        """
#
#        d0, n0 = self.grad_1d[0].shape
#        d1, n1 = self.grad_1d[1].shape
#        d2, n2 = self.grad_1d[2].shape
#
#        assert (f2_1.shape == (n0, d1, d2)) 
#        assert (f2_2.shape == (d0, n1, d2)) 
#        assert (f2_3.shape == (d0, d1, n2))
#                    
#        f3 = np.zeros((d0, d1, d2), dtype=float)
#                    
#        f3 += self.grad_1d[0].dot(f2_1.reshape(n0, d1*d2)).reshape(d0, d1, d2)
#        f3 += self.grad_1d[1].dot(f2_2.reshape(d0, n1*d2).T.reshape(n1, d2*d0)).reshape(d1*d2, d0).T.reshape(d0, d1, d2)
#        f3 += self.grad_1d[2].dot(f2_3.reshape(d0*d1, n2).T).T.reshape(d0, d1, d2)
#
#        return f3
#    
#    
#    # ================================================
#    def apply_G_strong(self, f0):
#        
#        # final coefficients of the operation G(f0)
#        f1_1 = np.empty(self.tensor_space.Nbase_1form[0], dtype=float)
#        f1_2 = np.empty(self.tensor_space.Nbase_1form[1], dtype=float)
#        f1_3 = np.empty(self.tensor_space.Nbase_1form[2], dtype=float)
#        
#        ker.g_strong(f0, f1_1, f1_2, f1_3)
#        
#        return f1_1, f1_2, f1_3
#    
#    # ================================================
#    def apply_C_strong(self, f1_1, f1_2, f1_3):
#        
#        # final coefficients of the operation C(f1)
#        f2_1 = np.empty(self.tensor_space.Nbase_2form[0], dtype=float)
#        f2_2 = np.empty(self.tensor_space.Nbase_2form[1], dtype=float)
#        f2_3 = np.empty(self.tensor_space.Nbase_2form[2], dtype=float)
#        
#        ker.c_strong(f1_1, f1_2, f1_3, f2_1, f2_2, f2_3)
#        
#        return f2_1, f2_2, f2_3
#    
#    # ================================================
#    def apply_D_strong(self, f2_1, f2_2, f2_3):
#        
#        # final coefficients of the operation D(f2)
#        f3 = np.empty(self.tensor_space.Nbase_3form, dtype=float)
#        
#        ker.d_strong(f2_1, f2_2, f2_3, f3)
#        
#        return f3
#    
#    # ================================================
#    def apply_G_weak(self, f1_1, f1_2, f1_3):
#        
#        # final 0-form coefficients of the operation G.T(f1)
#        f0 = np.zeros(self.tensor_space.Nbase_0form, dtype=float)
#        
#        # spline boundary conditions
#        bc_splines = [1*self.tensor_space.spl_kind[0], 1*self.tensor_space.spl_kind[1], 1*self.tensor_space.spl_kind[2]]
#        
#        ker.g_weak(f1_1, f1_2, f1_3, f0, bc_splines)
#        
#        # boundary conditions (1-direction)
#        if self.tensor_space.bc[0] == False:
#            
#            # f0(0, y, z)
#            if self.bc1[0] == 'free':
#                f0[ 0, :, :] -= f1_1[ 0, :, :]
#            # f0(1, y, z)
#            if self.bc1[1] == 'free':
#                f0[-1, :, :] += f1_1[-1, :, :]
#            
#        # boundary conditions (2-direction)
#        if self.tensor_space.bc[1] == False:
#            
#            # f0(x, 0, z)
#            if self.bc2[0] == 'free':
#                f0[:,  0, :] -= f1_2[:,  0, :]
#            # f0(x, 1, z)
#            if self.bc2[1] == 'free':
#                f0[:, -1, :] += f1_2[:, -1, :]
#            
#        # boundary conditions (3-direction)
#        if self.tensor_space.bc[2] == False:
#            
#            # f0(x, y, 0)
#            if self.bc3[0] == 'free':
#                f0[:, :,  0] -= f1_3[:, :,  0]
#            # f0(x, y, 1)
#            if self.bc3[1] == 'free':
#                f0[:, :, -1] += f1_3[:, :, -1]
#        
#        return f0
#    
#    # ================================================
#    def apply_C_weak(self, f2_1, f2_2, f2_3):
#        
#        # final 1-form coefficients of the operation C.T(f2)
#        f1_1 = np.zeros(self.tensor_space.Nbase_1form[0], dtype=float)
#        f1_2 = np.zeros(self.tensor_space.Nbase_1form[1], dtype=float)
#        f1_3 = np.zeros(self.tensor_space.Nbase_1form[2], dtype=float)
#        
#        # spline boundary conditions
#        bc_splines = [1*self.tensor_space.spl_kind[0], 1*self.tensor_space.spl_kind[1], 1*self.tensor_space.spl_kind[2]]
#        
#        ker.c_weak(f2_1, f2_2, f2_3, f1_1, f1_2, f1_3, bc_splines)
#        
#        # boundary conditions (1-direction)
#        if self.tensor_space.bc[0] == False:
#            
#            # f1_2(0, y, z)
#            if self.bc1[0] == 'free':
#                f1_2[ 0, :, :] -= f2_3[ 0, :, :]
#            # f1_2(1, y, z)
#            if self.bc1[1] == 'free':
#                f1_2[-1, :, :] += f2_3[-1, :, :]
#                
#            # f1_3(0, y, z)
#            if self.bc1[0] == 'free':
#                f1_3[ 0, :, :] += f2_2[ 0, :, :]
#            # f1_3(1, y, z)
#            if self.bc1[1] == 'free':
#                f1_3[-1, :, :] -= f2_2[-1, :, :]
#            
#        # boundary conditions (2-direction)
#        if self.tensor_space.bc[1] == False:
#            
#            # f1_3(x, 0, z)
#            if self.bc2[0] == 'free':
#                f1_3[:,  0, :] -= f2_1[:,  0, :]
#            # f1_3(x, 1, z)
#            if self.bc2[1] == 'free':
#                f1_3[:, -1, :] += f2_1[:, -1, :]
#                
#            # f1_1(x, 0, z)
#            if self.bc2[0] == 'free':
#                f1_1[:,  0, :] += f2_3[:,  0, :]
#            # f1_1(x, 1, z)
#            if self.bc2[1] == 'free':
#                f1_1[:, -1, :] -= f2_3[:, -1, :]
#                
#        # boundary conditions (3-direction)
#        if self.tensor_space.bc[2] == False:
#            
#            # f1_1(x, y, 0)
#            if self.bc3[0] == 'free':
#                f1_1[:, :,  0] -= f2_2[:, :,  0]
#            # f1_1(x, y, 1)
#            if self.bc3[1] == 'free':
#                f1_1[:, :, -1] += f2_2[:, :, -1]
#                
#            # f1_2(x, y, 0)
#            if self.bc3[0] == 'free':
#                f1_2[:, :,  0] += f2_1[:, :,  0]
#            # f1_2(x, y, 1)
#            if self.bc3[1] == 'free':
#                f1_2[:, :, -1] -= f2_1[:, :, -1]
#        
#        return f1_1, f1_2, f1_3
#    
#    # ================================================
#    def apply_D_weak(self, f3):
#        
#        # final 2-form coefficients of the operation D.T(f3)
#        f2_1 = np.zeros(self.tensor_space.Nbase_2form[0], dtype=float)
#        f2_2 = np.zeros(self.tensor_space.Nbase_2form[1], dtype=float)
#        f2_3 = np.zeros(self.tensor_space.Nbase_2form[2], dtype=float)
#        
#        # spline boundary conditions
#        bc_splines = [1*self.tensor_space.spl_kind[0], 1*self.tensor_space.spl_kind[1], 1*self.tensor_space.spl_kind[2]]
#        
#        ker.d_weak(f3, f2_1, f2_2, f2_3, bc_splines)
#        
#        # boundary conditions (1-direction)
#        if self.tensor_space.bc[0] == False:
#            
#            # f2_1(0, y, z)
#            if self.bc1[0] == 'free':
#                f2_1[ 0, :, :] -= f3[ 0, :, :]
#            # f2_1(1, y, z)
#            if self.bc1[1] == 'free':
#                f2_1[-1, :, :] += f3[-1, :, :]
#            
#        # boundary conditions (2-direction)
#        if self.tensor_space.bc[1] == False:
#            
#            # f2_2(x, 0, z)
#            if self.bc2[0] == 'free':
#                f2_2[:,  0, :] -= f3[:,  0, :]
#            # f2_2(x, 1, z)
#            if self.bc2[1] == 'free':
#                f2_2[:, -1, :] += f3[:, -1, :]
#            
#        # boundary conditions (3-direction)
#        if self.tensor_space.bc[2] == False:
#            
#            # f2_3(x, y, 0)
#            if self.bc3[0] == 'free':
#                f2_3[:, :,  0] -= f3[:, :,  0]
#            # f2_3(x, y, 1)
#            if self.bc3[1] == 'free':
#                f2_3[:, :, -1] += f3[:, :, -1]
#                
#        return f2_1, f2_2, f2_3