# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to compute mass matrices in 1d.
"""

import numpy        as np
import scipy.sparse as spa


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