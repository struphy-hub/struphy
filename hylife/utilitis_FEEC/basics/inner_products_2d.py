# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to compute inner products with given functions in 2D.
"""


import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.basics.kernels_2d as ker


# ================ inner product in V0 ===========================
def inner_prod_V0(tensor_space_FEM, fun, kind_map, params_map=None, tensor_space_F=None, cx=None, cy=None):
    """
    ----------------------------------------------------------------------------------------------------------
    Assembles the 2D inner prodcut (NN, fun) of the given tensor product B-spline spaces of bi-degree (p1, p2).
    
    In case of an analytical mapping (kind_map >= 1), all mapping related quantities are called from hylife.geometry.mappings_2d. One must then pass the parameter list params_map.
    
    In case of a discrete mapping (kind_map = 0), one must pass a 2D tensor product B-spline space tensor_space_F together with control points cx and cy which together define the mapping.
    -----------------------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    fun : callable
        the 0-form with which the inner products shall be computed
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
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
    NbaseN = tensor_space_FEM.NbaseN  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points
    
    # create dummy variables
    if kind_map == 0:
        T_F        =  tensor_space_F.T
        p_F        =  tensor_space_F.p
        NbaseN_F   =  tensor_space_F.NbaseN
        params_map =  np.zeros((1,  ), dtype=float)
    else:
        T_F        = [np.zeros((1,  ), dtype=float), np.zeros(1, dtype=float)]
        p_F        =  np.zeros((1,  ), dtype=int)
        NbaseN_F   =  np.zeros((1,  ), dtype=int)
        cx         =  np.zeros((1, 1), dtype=float)
        cy         =  np.zeros((1, 1), dtype=float)
    
    
    # evaluation of |det(DF)| at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], n_quad[0], n_quad[1]), dtype=float)
    
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], mat_map, 1, kind_map, params_map, T_F[0], T_F[1], p_F, NbaseN_F, cx, cy)
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing='ij') 
    mat_f     = fun(quad_mesh[0], quad_mesh[1])
    
    # assembly
    F = np.zeros((NbaseN[0], NbaseN[1]), dtype=float)
    
    ker.kernel_inner_1(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], 0, 0, wts[0], wts[1], basisN[0], basisN[1], NbaseN[0], NbaseN[1], F, mat_f, mat_map)
                
    return F


# ================ inner product in V1 (H curl) ===========================
def inner_prod_V1_curl(tensor_space_FEM, fun, kind_map, params_map=None, tensor_space_F=None, cx=None, cy=None):
    """
    ----------------------------------------------------------------------------------------------------------
    Assembles the 2D inner prodcut ([DN, ND], [fun_1, fun_2]) of the given tensor product B-spline spaces of bi-degree (p1, p2).
    
    In case of an analytical mapping (kind_map >= 1), all mapping related quantities are called from hylife.geometry.mappings_2d. One must then pass the parameter list params_map.
    
    In case of a discrete mapping (kind_map = 0), one must pass a 2D tensor product B-spline space tensor_space_F together with control points cx and cy which together define the mapping.
    -----------------------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    fun : list of callables
        the two 1-form components with which the inner products shall be computed
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
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
    NbaseN = tensor_space_FEM.NbaseN  # total number of basis functions (N)
    NbaseD = tensor_space_FEM.NbaseD  # total number of basis functions (D)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points
    wts    = tensor_space_FEM.wts     # global quadrature weights
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points (D)
    
    # create dummy variables
    if kind_map == 0:
        T_F        =  tensor_space_F.T
        p_F        =  tensor_space_F.p
        NbaseN_F   =  tensor_space_F.NbaseN
        params_map =  np.zeros((1,  ), dtype=float)
    else:
        T_F        = [np.zeros((1,  ), dtype=float), np.zeros(1, dtype=float)]
        p_F        =  np.zeros((1,  ), dtype=int)
        NbaseN_F   =  np.zeros((1,  ), dtype=int)
        cx         =  np.zeros((1, 1), dtype=float)
        cy         =  np.zeros((1, 1), dtype=float)
    
    # basis functions of components of a 1-form
    basis = [[basisD[0], basisN[1]], 
             [basisN[0], basisD[1]]]
    
    Nbase = [[NbaseD[0], NbaseN[1]], 
             [NbaseN[0], NbaseD[1]]]
    
    ns    = [[1, 0], 
             [0, 1]]
    
    # G^(-1)|det(DF)| at quadrature points
    mat_map   = np.empty((Nel[0], Nel[1], n_quad[0], n_quad[1]), dtype=float)
    kind_funs = [11, 12, 12, 13]
    
    # function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing='ij') 
    
    # components of global inner product
    F = [np.zeros((Nbase[0], Nbase[1])) for Nbase in Nbase]
    
    # assembly
    counter = 0
    
    for a in range(2):
        
        ni1,    ni2    = ns[a]
        bi1,    bi2    = basis[a]
        Nbase1, Nbase2 = Nbase[a]
        
        for b in range(2):
            
            # evaluate G^(-1)|det(DF)| at quadrature points
            ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], mat_map, kind_funs[counter], kind_map, params_map, T_F[0], T_F[1], p_F, NbaseN_F, cx, cy)
            
            # evaluate function at quadrature points
            mat_f = fun[b](quad_mesh[0], quad_mesh[1])
            
            ker.kernel_inner_1(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], ni1, ni2, wts[0], wts[1], bi1, bi2, Nbase1, Nbase2, F[a], mat_f, mat_map)
            
            counter += 1
            
    return F


# ================ inner product in V1 (H div) ===========================
def inner_prod_V1_div(tensor_space_FEM, fun, kind_map, params_map=None, tensor_space_F=None, cx=None, cy=None):
    """
    ----------------------------------------------------------------------------------------------------------
    Assembles the 2D inner prodcut ([ND, DN], [fun_1, fun_2]) of the given tensor product B-spline spaces of bi-degree (p1, p2).
    
    In case of an analytical mapping (kind_map >= 1), all mapping related quantities are called from hylife.geometry.mappings_2d. One must then pass the parameter list params_map.
    
    In case of a discrete mapping (kind_map = 0), one must pass a 2D tensor product B-spline space tensor_space_F together with control points cx and cy which together define the mapping.
    -----------------------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    fun : list of callables
        the two 1-form components with which the inner products shall be computed
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
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
    NbaseN = tensor_space_FEM.NbaseN  # total number of basis functions (N)
    NbaseD = tensor_space_FEM.NbaseD  # total number of basis functions (D)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points
    wts    = tensor_space_FEM.wts     # global quadrature weights
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points (D)
    
    # create dummy variables
    if kind_map == 0:
        T_F        =  tensor_space_F.T
        p_F        =  tensor_space_F.p
        NbaseN_F   =  tensor_space_F.NbaseN
        params_map =  np.zeros((1,  ), dtype=float)
    else:
        T_F        = [np.zeros((1,  ), dtype=float), np.zeros(1, dtype=float)]
        p_F        =  np.zeros((1,  ), dtype=int)
        NbaseN_F   =  np.zeros((1,  ), dtype=int)
        cx         =  np.zeros((1, 1), dtype=float)
        cy         =  np.zeros((1, 1), dtype=float)
    
    # basis functions of components of a 1-form
    basis = [[basisN[0], basisD[1]], 
             [basisD[0], basisN[1]]]
    
    Nbase = [[NbaseN[0], NbaseD[1]], 
             [NbaseD[0], NbaseN[1]]]
    
    ns    = [[0, 1], 
             [1, 0]]
    
    # G/|det(DF)| at quadrature points
    mat_map   = np.empty((Nel[0], Nel[1], n_quad[0], n_quad[1]), dtype=float)
    kind_funs = [21, 22, 22, 23]
    
    # function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing='ij') 
    
    # components of global inner product
    F = [np.zeros((Nbase[0], Nbase[1])) for Nbase in Nbase]
    
    # assembly
    counter = 0
    
    for a in range(2):
        
        ni1,    ni2    = ns[a]
        bi1,    bi2    = basis[a]
        Nbase1, Nbase2 = Nbase[a]
        
        for b in range(2):
            
            # evaluate G/|det(DF)| at quadrature points
            ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], mat_map, kind_funs[counter], kind_map, params_map, T_F[0], T_F[1], p_F, NbaseN_F, cx, cy)
            
            # evaluate function at quadrature points
            mat_f = fun[b](quad_mesh[0], quad_mesh[1])
            
            ker.kernel_inner_1(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], ni1, ni2, wts[0], wts[1], bi1, bi2, Nbase1, Nbase2, F[a], mat_f, mat_map)
            
            counter += 1
            
    return F


# ================ inner product in V2 ===========================
def inner_prod_V2(tensor_space_FEM, fun, kind_map, params_map=None, tensor_space_F=None, cx=None, cy=None):
    """
    ----------------------------------------------------------------------------------------------------------
    Assembles the 2D inner prodcut (DD, fun) of the given tensor product B-spline spaces of bi-degree (p1, p2).
    
    In case of an analytical mapping (kind_map >= 1), all mapping related quantities are called from hylife.geometry.mappings_2d. One must then pass the parameter list params_map.
    
    In case of a discrete mapping (kind_map = 0), one must pass a 2D tensor product B-spline space tensor_space_F together with control points cx and cy which together define the mapping.
    -----------------------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    fun : callable
        the two 2-form components with which the inner products shall be computed
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
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
    NbaseD = tensor_space_FEM.NbaseD  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points
    
    
    # evaluation of 1/|det(DF)| at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], n_quad[0], n_quad[1]), dtype=float)
    
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], mat_map, 2, kind_map, params_map, T_F[0], T_F[1], p_F, NbaseN_F, cx, cy)
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing='ij') 
    mat_f     = fun(quad_mesh[0], quad_mesh[1])
    
    # assembly
    F = np.zeros((NbaseD[0], NbaseD[1]), dtype=float)
    
    ker.kernel_inner_1(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], 1, 1, wts[0], wts[1], basisD[0], basisD[1], NbaseD[0], NbaseD[1], F, mat_f, mat_map)
                
    return F