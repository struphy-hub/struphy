# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to compute mass matrices in 1d.
"""

import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.bsplines as bsp


# ======= mass matrix in V0 ====================
def get_M0(spline_space, mapping=None):
    """
    Assembles the 1d mass matrix (N) of the given B-spline space of degree p.
    
    Parameters
    ----------
    spline_space : spline_space_1d
        a 1d B-spline space
        
    mapping : callable
        derivative of mapping df/dxi
    """
      
    p      = spline_space.p       # spline degrees
    Nel    = spline_space.Nel     # number of elements
    NbaseN = spline_space.NbaseN  # total number of basis functions (N)
    
    n_quad = spline_space.n_quad  # number of quadrature points per element
    pts    = spline_space.pts     # global quadrature points in format (element, local quad_point)
    wts    = spline_space.wts     # global quadrature weights in format (element, local weight)
    
    basisN = spline_space.basisN  # evaluated basis functions at quadrature points
    
    
    # evaluation of mapping at quadrature points
    if mapping == None:
        mat_map = np.ones(pts.shape, dtype=float)
    else:
        mat_map = mapping(pts)
    
    # assembly
    M      = np.zeros((NbaseN, 2*p + 1), dtype=float)

    for ie in range(Nel):
        for il in range(p + 1):
            for jl in range(p + 1):

                value = 0.

                for q in range(n_quad):
                    value += wts[ie, q] * basisN[ie, il, 0, q] * basisN[ie, jl, 0, q] * mat_map[ie, q]

                M[(ie + il)%NbaseN, p + jl - il] += value
                
    indices = np.indices((NbaseN, 2*p + 1))
    shift   = np.arange(NbaseN) - p
    
    row     = indices[0].flatten()
    col     = (indices[1] + shift[:, None])%NbaseN
    
    M       = spa.csc_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseN, NbaseN))
    M.eliminate_zeros()
                
    return M


# ======= mass matrix in V1 ====================
def get_M1(spline_space, mapping=None):
    """
    Assembles the 1d mass matrix (D) of the given B-spline space of degree p.
    
    Parameters
    ----------
    spline_space : spline_space_1d
        a 1d B-spline space
        
    mapping : callable
        derivative of mapping df/dxi
    """
      
    p      = spline_space.p       # spline degrees
    Nel    = spline_space.Nel     # number of elements
    NbaseD = spline_space.NbaseD  # total number of basis functions (N)
    
    n_quad = spline_space.n_quad  # number of quadrature points per element
    pts    = spline_space.pts     # global quadrature points in format (element, local quad_point)
    wts    = spline_space.wts     # global quadrature weights in format (element, local weight)
    
    basisD = spline_space.basisD  # evaluated basis functions at quadrature points
    
    
    # evaluation of mapping at quadrature points
    if mapping == None:
        mat_map = np.ones(pts.shape, dtype=float)
    else:
        mat_map = 1/mapping(pts)
    
    # assembly
    M      = np.zeros((NbaseD, 2*p + 1), dtype=float)

    for ie in range(Nel):
        for il in range(p):
            for jl in range(p):

                value = 0.

                for q in range(n_quad):
                    value += wts[ie, q] * basisD[ie, il, 0, q] * basisD[ie, jl, 0, q] * mat_map[ie, q]

                M[(ie + il)%NbaseD, p + jl - il] += value
                
    indices = np.indices((NbaseD, 2*p + 1))
    shift   = np.arange(NbaseD) - p
    
    row     = indices[0].flatten()
    col     = (indices[1] + shift[:, None])%NbaseD
    
    M       = spa.csc_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseD, NbaseD))
    M.eliminate_zeros()
                
    return M



# ======= general mass matrix  ====================
def get_M(spline_space, phi_i=0, phi_j=0, fun=None, jac=None):
    """
    General assembly of matrices of the form M_ij = phi_i(eta) * phi_j(eta) * fun(eta) under the jacobian jac = dF/deta.
    
    Parameters
    ----------
    spline_space : spline_space_1d
        a 1d B-spline space
        
    phi_i : int
        kind of basis function phi_i (0 : spline of degree p, 1 : derivative of splines of degree p, 2 : spline of degree p - 1)
        
    phi_j : int
        kind of basis function phi_j (0 : spline of degree p, 1 : derivative of splines of degree p, 2 : spline of degree p - 1)
        
    fun : callable
        weight function
        
    jac : callable
        derivative of the mapping x = F(eta)
    """
      
    p      = spline_space.p       # spline degree
    Nel    = spline_space.Nel     # number of elements
    
    NbaseN = spline_space.NbaseN  # total number of basis functions (p)
    NbaseD = spline_space.NbaseD  # total number of basis functions (p-1)
    
    n_quad = spline_space.n_quad  # number of quadrature points per element
    pts    = spline_space.pts     # global quadrature points in format (element, local quad_point)
    wts    = spline_space.wts     # global quadrature weights in format (element, local weight)
    
    # evaluation of basis functions at quadrature points in format (element, local function, derivative, local quad_point)
    basis_T = bsp.basis_ders_on_quad_grid(spline_space.T, p    , spline_space.pts, 1, normalize=False)
    basis_t = bsp.basis_ders_on_quad_grid(spline_space.t, p - 1, spline_space.pts, 0, normalize=False)
    
    # evaluation of weight function at quadrature points (optional)
    if fun == None:
        mat_fun = np.ones(pts.shape, dtype=float)
    else:
        mat_fun = fun(pts.flatten()).reshape(Nel, n_quad)
    
    # evaluation of jacobian at quadrature points
    if jac == None:
        mat_jac = np.ones(pts.shape, dtype=float)
    else:
        mat_jac = jac(pts.flatten()).reshape(Nel, n_quad) 
    
    # selection of phi_i basis functions
    if   phi_i == 0:
        Ni = NbaseN
        ni = 0
        bi = basis_T[:, :, 0, :]
        
    elif phi_i == 1:
        Ni = NbaseN
        ni = 0
        bi = basis_T[:, :, 1, :]/mat_jac[:, None, :]
        
    elif phi_i == 2:
        Ni = NbaseD
        ni = 1
        bi = basis_t[:, :, 0, :]
        
    # selection of phi_j basis functions
    if   phi_j == 0:
        Nj = NbaseN
        nj = 0
        bj = basis_T[:, :, 0, :]
        
    elif phi_j == 1:
        Nj = NbaseN
        nj = 0
        bj = basis_T[:, :, 1, :]/mat_jac[:, None, :]
        
    elif phi_j == 2:
        Nj = NbaseD
        nj = 1
        bj = basis_t[:, :, 0, :]
    
    # matrix assembly
    M = np.zeros((Ni, 2*p + 1), dtype=float)

    for ie in range(Nel):
        for il in range(p + 1 - ni):
            for jl in range(p + 1 - nj):

                value = 0.

                for q in range(n_quad):
                    value += wts[ie, q] * bi[ie, il, q] * bj[ie, jl, q] * mat_fun[ie, q] * mat_jac[ie, q]

                M[(ie + il)%Ni, p + jl - il] += value
                
    indices = np.indices((Ni, 2*p + 1))
    shift   = np.arange(Ni) - p
    
    row     = indices[0].flatten()
    col     = (indices[1] + shift[:, None])%Nj
    
    M       = spa.csc_matrix((M.flatten(), (row, col.flatten())), shape=(Ni, Nj))
    M.eliminate_zeros()
                
    return M


# ======= test for general mass matrix  ====================
def test_M(spline_space, phi_i=0, phi_j=0, fun=lambda eta : 1., jac=lambda eta : 1.):
    
    from scipy.integrate import quad

    # selection of phi_i basis functions
    if   phi_i == 0:
        Ni = spline_space.NbaseN
        bi = lambda eta : spline_space.evaluate_N(eta, ci)
        
    elif phi_i == 1:
        Ni = spline_space.NbaseN
        bi = lambda eta : spline_space.evaluate_dN(eta, ci)/jac(eta)
        
    elif phi_i == 2:
        Ni = spline_space.NbaseD
        bi = lambda eta : spline_space.evaluate_D(eta, ci)/spline_space.Nel
        
    # selection of phi_j basis functions
    if   phi_j == 0:
        Nj = spline_space.NbaseN
        bj = lambda eta : spline_space.evaluate_N(eta, cj)
        
    elif phi_j == 1:
        Nj = spline_space.NbaseN
        bj = lambda eta : spline_space.evaluate_dN(eta, cj)/jac(eta)
        
    elif phi_j == 2:
        Nj = spline_space.NbaseD
        bj = lambda eta : spline_space.evaluate_D(eta, cj)/spline_space.Nel
    
    # coefficients
    ci = np.zeros(Ni, dtype=float)
    cj = np.zeros(Nj, dtype=float)
    
    # integration
    M = np.zeros((Ni, Nj), dtype=float)
    
    for i in range(Ni):
        for j in range(Nj):
        
            ci[:] = 0.
            cj[:] = 0.

            ci[i] = 1.
            cj[j] = 1.
            
            integrand = lambda eta : bi(eta) * bj(eta) * fun(eta) * jac(eta)
            
            M[i, j] = quad(integrand, 0., 1.)[0]
            
    return M