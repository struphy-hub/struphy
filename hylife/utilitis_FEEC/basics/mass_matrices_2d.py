# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to compute mass matrices 2d.
"""


import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.basics.kernels_2d as ker


# ================ mass matrix in V0 ===========================
def mass_V0(tensor_space_FEM, mapping, kind_map=None, params_map=None, tensor_space_F=None, cx=None, cy=None):
    """
    ----------------------------------------------------------------------------------------------------------
    Assembles the 2d mass matrix (NN NN) of the given tensor product B-spline spaces of multi-degree (p1, p2).
    
    In case of an analytical mapping, all mapping related quantities are called from hylife.geometry.mappings_analytical_2d.
    One must then pass kind_map and params_map.
    
    In case of a discrete mapping, one must pass a tensor product B-spline space together with control points cx and cy.
    -----------------------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    mapping : int
        0 : analytical mapping
        1 : discrete   mapping
        
    kind_map : int
        type of mapping in case of analytical mapping
        
    params_map : list of doubles
        parameters for the mapping in case of analytical mapping
        
    tensor_space_F : tensor_spline_space
        tensor product B-spline space for discrete mapping in case of discrete mapping
        
    cx : array_like
        x control points in case of discrete mapping
        
    cy : array_like
        y control points in case of discrete mapping
    """
    
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    bc     = tensor_space_FEM.bc      # boundary conditions (periodic vs. clamped)
    NbaseN = tensor_space_FEM.NbaseN  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points
    
    
    # evaluation of Jacobian determinant at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], n_quad[0], n_quad[1]), dtype=float)
    
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], mat_map, 1, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, Nel, n_quad, pts[0], pts[1], mat_map, 1)
    
    # assembly of global mass matrix
    M = np.zeros((NbaseN[0], NbaseN[1], 2*p[0] + 1, 2*p[1] + 1), dtype=float)
    
    ker.kernel_mass(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], 0, 0, 0, 0, wts[0], wts[1], basisN[0], basisN[1], basisN[0], basisN[1], NbaseN[0], NbaseN[1], M, mat_map)
              
    # conversion to sparse matrix
    indices = np.indices((NbaseN[0], NbaseN[1], 2*p[0] + 1, 2*p[1] + 1))
    
    shift   = [np.arange(NbaseN) - p for NbaseN, p in zip(NbaseN, p)]
    
    row     = (NbaseN[1]*indices[0] + indices[1]).flatten()
    
    col1    = (indices[2] + shift[0][:, None, None, None])%NbaseN[0]
    col2    = (indices[3] + shift[1][None, :, None, None])%NbaseN[1]

    col     = NbaseN[1]*col1 + col2
                
    M       = spa.csc_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseN[0]*NbaseN[1], NbaseN[0]*NbaseN[1]))
    M.eliminate_zeros()
    
    return M