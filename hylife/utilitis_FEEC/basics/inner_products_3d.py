# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to compute inner products with given functions in 3d.
"""


import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.basics.kernels_3d as ker


# ================ inner product in V0 ===========================
def inner_prod_V0(tensor_space, kind_map, params_map, fun):
    """
    Assembles the 3d inner product (NNN) of the given tensor product B-spline spaces of multi-degree (p1, p2, p3) with the function fun.
    The mapping is called from hylife.geometry.mappings_analytical which contains a collection of analytical mappings.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        tensor product B-spline space
        
    kind_map : int
        type of mapping
        
    params_map : list of doubles
        parameters for the mapping
        
    fun : callable
        analytical function for which the inner product with all basis function in V0 shall be computed
    """
    
    p      = tensor_space.p       # spline degrees
    Nel    = tensor_space.Nel     # number of elements
    NbaseN = tensor_space.NbaseN  # total number of basis functions (N)
    
    n_quad = tensor_space.n_quad  # number of quadrature points per element
    pts    = tensor_space.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space.wts     # global quadrature weights in format (element, local weight)
    
    basisN = tensor_space.basisN  # evaluated basis functions at quadrature points
    
    
    # evaluation of Jacobian determinant at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float, order='F')
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 1, kind_map, params_map)
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    mat_f     = np.asfortranarray(fun(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    
    # assembly
    F = np.zeros((NbaseN[0], NbaseN[1], NbaseN[2]), dtype=float, order='F')
    
    ker.kernel_inner(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], 0, 0, 0, wts[0], wts[1], wts[2], basisN[0], basisN[1], basisN[2], NbaseN[0], NbaseN[1], NbaseN[2], F, mat_f, mat_map)
                
    return F


# ================ inner product in V1 ===========================
def inner_prod_V1(tensor_space, kind_map, params_map, fun):
    """
    Assembles the 3d inner product (DNN, NDN, NND) of the given tensor product B-spline spaces of multi-degree (p1, p2, p3) with the function fun.
    The mapping is called from hylife.geometry.mappings_analytical which contains a collection of analytical mappings.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        tensor product B-spline space
        
    kind_map : int
        type of mapping
        
    params_map : list of doubles
        parameters for the mapping
        
    fun : list of callables
        components of the analytical function for which the inner product with all basis function in V1 shall be computed
    """
    
    p      = tensor_space.p       # spline degrees
    Nel    = tensor_space.Nel     # number of elements
    NbaseN = tensor_space.NbaseN  # total number of basis functions (N)
    NbaseD = tensor_space.NbaseD  # total number of basis functions (D)
    
    n_quad = tensor_space.n_quad  # number of quadrature points per element
    pts    = tensor_space.pts     # global quadrature points
    wts    = tensor_space.wts     # global quadrature weights
    
    basisN = tensor_space.basisN  # evaluated basis functions at quadrature points (N)
    basisD = tensor_space.basisD  # evaluated basis functions at quadrature points (D)
    
    
    # basis functions of components of a 1 - form
    basis = [[basisD[0], basisN[1], basisN[2]], 
             [basisN[0], basisD[1], basisN[2]], 
             [basisN[0], basisN[1], basisD[2]]]
    
    Nbase = [[NbaseD[0], NbaseN[1], NbaseN[2]], 
             [NbaseN[0], NbaseD[1], NbaseN[2]], 
             [NbaseN[0], NbaseN[1], NbaseD[2]]]
    
    ns    = [[1, 0, 0], 
             [0, 1, 0], 
             [0, 0, 1]]
    
    # mappings at quadrature points
    mat_map   = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float, order='F')
    kind_funs = [11, 12, 14, 12, 13, 15, 14, 15, 16]
    
    # function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    
    # components of global inner product
    F = [np.zeros((Nbase[0], Nbase[1], Nbase[2]), order='F') for Nbase in Nbase]
    
    # assembly
    counter = 0
    
    for a in range(3):
        
        ni1,    ni2,    ni3    = ns[a]
        bi1,    bi2,    bi3    = basis[a]
        Nbase1, Nbase2, Nbase3 = Nbase[a]
        
        for b in range(3):
            
            ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, kind_funs[counter], kind_map, params_map)
            
            mat_f = np.asfortranarray(fun[b](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
            
            ker.kernel_inner(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ni1, ni2, ni3, wts[0], wts[1], wts[2], bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, F[a], mat_f, mat_map)
            
            counter += 1
            
    return F


# ================ inner product in V2 ===========================
def inner_prod_V2(tensor_space, kind_map, params_map, fun):
    """
    Assembles the 3d inner product (NDD, DND, DDN) of the given tensor product B-spline spaces of multi-degree (p1, p2, p3) with the function fun.
    The mapping is called from hylife.geometry.mappings_analytical which contains a collection of analytical mappings.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        tensor product B-spline space
        
    kind_map : int
        type of mapping
        
    params_map : list of doubles
        parameters for the mapping
        
    fun : list of callables
        components of the analytical function for which the inner product with all basis function in V2 shall be computed
    """
    
    p      = tensor_space.p       # spline degrees
    Nel    = tensor_space.Nel     # number of elements
    NbaseN = tensor_space.NbaseN  # total number of basis functions (N)
    NbaseD = tensor_space.NbaseD  # total number of basis functions (D)
    
    n_quad = tensor_space.n_quad  # number of quadrature points per element
    pts    = tensor_space.pts     # global quadrature points
    wts    = tensor_space.wts     # global quadrature weights
    
    basisN = tensor_space.basisN  # evaluated basis functions at quadrature points (N)
    basisD = tensor_space.basisD  # evaluated basis functions at quadrature points (D)
    
    
    # basis functions of components of a 2 - form
    basis = [[basisN[0], basisD[1], basisD[2]], 
             [basisD[0], basisN[1], basisD[2]], 
             [basisD[0], basisD[1], basisN[2]]]
    
    Nbase = [[NbaseN[0], NbaseD[1], NbaseD[2]], 
             [NbaseD[0], NbaseN[1], NbaseD[2]], 
             [NbaseD[0], NbaseD[1], NbaseN[2]]]
    
    ns    = [[0, 1, 1], 
             [1, 0, 1], 
             [1, 1, 0]]
    
    # mappings at quadrature points
    mat_map   = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float, order='F')
    kind_funs = [21, 22, 24, 22, 23, 25, 24, 25, 26]
    
    # function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    
    # components of global inner product
    F = [np.zeros((Nbase[0], Nbase[1], Nbase[2]), order='F') for Nbase in Nbase]
    
    # assembly
    counter = 0
    
    for a in range(3):
        
        ni1,    ni2,    ni3    = ns[a]
        bi1,    bi2,    bi3    = basis[a]
        Nbase1, Nbase2, Nbase3 = Nbase[a]
        
        for b in range(3):
            
            ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, kind_funs[counter], kind_map, params_map)
            
            mat_f = np.asfortranarray(fun[b](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
            
            ker.kernel_inner(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ni1, ni2, ni3, wts[0], wts[1], wts[2], bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, F[a], mat_f, mat_map)
            
            counter += 1
            
    return F


# ================ inner product in V3 ===========================
def inner_prod_V3(tensor_space, kind_map, params_map, fun):
    """
    Assembles the 3d inner product (DDD) of the given tensor product B-spline spaces of multi-degree (p1, p2, p3) with the function fun.
    The mapping is called from hylife.geometry.mappings_analytical which contains a collection of analytical mappings.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        tensor product B-spline space
        
    kind_map : int
        type of mapping
        
    params_map : list of doubles
        parameters for the mapping
        
    fun : callable
        analytical function for which the inner product with all basis function in V3 shall be computed
    """
    
    p      = tensor_space.p       # spline degrees
    Nel    = tensor_space.Nel     # number of elements
    NbaseD = tensor_space.NbaseD  # total number of basis functions (N)
    
    n_quad = tensor_space.n_quad  # number of quadrature points per element
    pts    = tensor_space.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space.wts     # global quadrature weights in format (element, local weight)
    
    basisD = tensor_space.basisD  # evaluated basis functions at quadrature points
    
    
    # evaluation of 1 / Jacobian determinant at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float, order='F')
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 1, kind_map, params_map)
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    mat_f     = np.asfortranarray(fun(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    
    # assembly
    F = np.zeros((NbaseD[0], NbaseD[1], NbaseD[2]), dtype=float, order='F')
    
    ker.kernel_inner(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], 1, 1, 1, wts[0], wts[1], wts[2], basisD[0], basisD[1], basisD[2], NbaseD[0], NbaseD[1], NbaseD[2], F, mat_f, mat_map)
                
    return F