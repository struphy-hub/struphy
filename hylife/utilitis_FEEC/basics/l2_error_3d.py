# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to compute L2 - errors in 3d.
"""

import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.basics.kernels_3d as ker


# ======= error in V0 ====================
def l2_error_V0(tensor_space_FEM, fun, coeff, mapping, kind_map=None, params_map=None, tensor_space_F=None, cx=None, cy=None, cz=None):
    """
    Computes the 3d L2-error (NNN) of the given tensor product B-spline spaces of multi-degree (p1, p2, p3) with coefficients coeff with the function fun.
    
    In case of an analytical mapping, all quantities related to the mapping are called from hylife.geometry.mappings_analytical which contains a collection of analytical mappings. One must pass the parameters kind_map and params_map.
    
    In case of a discrete mapping, one must pass an additional tensor product B-spline space together with control points cx, cy and cz which together define the mapping.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    fun : callable
        0-form for which the L2-error in V0 shall be computed
        
    coeff : array_like
        coefficients of the B-spline space
        
    mapping : int
        0 : analytical mapping
        1 : discrete mapping
        
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
        
    cz : array_like
        z control points in case of discrete mapping
    """
      
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    NbaseN = tensor_space_FEM.NbaseN  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points
    
    # evaluation of Jacobian determinant at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 1, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.T[2], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, cz, Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 1)
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    mat_f     = fun(quad_mesh[0], quad_mesh[1], quad_mesh[2])
    
    # compute error
    error = np.zeros(Nel, dtype=float)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 0, 0], [0, 0, 0], basisN[0], basisN[1], basisN[2], basisN[0], basisN[1], basisN[2], [NbaseN[0], NbaseN[1], NbaseN[2]], [NbaseN[0], NbaseN[1], NbaseN[2]], error, mat_f, mat_f, coeff, coeff, mat_map)
                
    return np.sqrt(error.sum())


# ======= error in V1 ====================
def l2_error_V1(tensor_space_FEM, fun, coeff, mapping, kind_map=None, params_map=None, tensor_space_F=None, cx=None, cy=None, cz=None):
    """
    Computes the 3d L2-error (DNN, NDN, NND) of the given tensor product B-spline spaces of multi-degree (p1, p2, p3) with coefficients coeff with the function fun.
    
    In case of an analytical mapping, all quantities related to the mapping are called from hylife.geometry.mappings_analytical which contains a collection of analytical mappings. One must pass the parameters kind_map and params_map.
    
    In case of a discrete mapping, one must pass an additional tensor product B-spline space together with control points cx, cy and cz which together define the mapping.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    fun : list of callables
        components of 1-form for which the L2-error in V1 shall be computed
        
    coeff : list of array_like
        coefficients of the B-spline space
        
    mapping : int
        0 : analytical mapping
        1 : discrete mapping
        
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
        
    cz : array_like
        z control points in case of discrete mapping
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
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    mat_f1    = fun[0](quad_mesh[0], quad_mesh[1], quad_mesh[2])
    mat_f2    = fun[1](quad_mesh[0], quad_mesh[1], quad_mesh[2])
    mat_f3    = fun[2](quad_mesh[0], quad_mesh[1], quad_mesh[2])
    
    # evaluation of mapping at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    
    # compute error
    error = np.zeros(Nel, dtype=float)
    
    # 1 * f1 * G^11 * sqrt(g) * f1
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 11, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.T[2], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, cz, Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 11)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 0], [1, 0, 0], basisD[0], basisN[1], basisN[2], basisD[0], basisN[1], basisN[2], [NbaseD[0], NbaseN[1], NbaseN[2]], [NbaseD[0], NbaseN[1], NbaseN[2]], error, mat_f1, mat_f1, coeff[0], coeff[0], mat_map)
    
    # 2 * f1 * G^12 * sqrt(g) * f2
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 12, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.T[2], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, cz, Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 12)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 0], [0, 1, 0], basisD[0], basisN[1], basisN[2], basisN[0], basisD[1], basisN[2], [NbaseD[0], NbaseN[1], NbaseN[2]], [NbaseN[0], NbaseD[1], NbaseN[2]], error, mat_f1, mat_f2, coeff[0], coeff[1], 2*mat_map)
    
    # 2 * f1 * G^13 * sqrt(g) * f3
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 14, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.T[2], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, cz, Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 14)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 0], [0, 0, 1], basisD[0], basisN[1], basisN[2], basisN[0], basisN[1], basisD[2], [NbaseD[0], NbaseN[1], NbaseN[2]], [NbaseN[0], NbaseN[1], NbaseD[2]], error, mat_f1, mat_f3, coeff[0], coeff[2], 2*mat_map)
    
    # 1 * f2 * G^22 * sqrt(g) * f2
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 13, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.T[2], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, cz, Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 13)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 0], [0, 1, 0], basisN[0], basisD[1], basisN[2], basisN[0], basisD[1], basisN[2], [NbaseN[0], NbaseD[1], NbaseN[2]], [NbaseN[0], NbaseD[1], NbaseN[2]], error, mat_f2, mat_f2, coeff[1], coeff[1], mat_map)
    
    # 2 * f2 * G^23 * sqrt(g) * f3
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 15, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.T[2], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, cz, Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 15)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 0], [0, 0, 1], basisN[0], basisD[1], basisN[2], basisN[0], basisN[1], basisD[2], [NbaseN[0], NbaseD[1], NbaseN[2]], [NbaseN[0], NbaseN[1], NbaseD[2]], error, mat_f2, mat_f3, coeff[1], coeff[2], 2*mat_map)
    
    # 1 * f3 * G^33 * sqrt(g) * f3
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 16, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.T[2], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, cz, Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 16)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 0, 1], [0, 0, 1], basisN[0], basisN[1], basisD[2], basisN[0], basisN[1], basisD[2], [NbaseN[0], NbaseN[1], NbaseD[2]], [NbaseN[0], NbaseN[1], NbaseD[2]], error, mat_f3, mat_f3, coeff[2], coeff[2], mat_map)
                
    return np.sqrt(error.sum())


# ======= error in V2 ====================
def l2_error_V2(tensor_space_FEM, fun, coeff, mapping, kind_map=None, params_map=None, tensor_space_F=None, cx=None, cy=None, cz=None):
    """
    Computes the 3d L2-error (NDD, DND, DDN) of the given tensor product B-spline spaces of multi-degree (p1, p2, p3) with coefficients coeff with the function fun.
    
    In case of an analytical mapping, all quantities related to the mapping are called from hylife.geometry.mappings_analytical which contains a collection of analytical mappings. One must pass the parameters kind_map and params_map.
    
    In case of a discrete mapping, one must pass an additional tensor product B-spline space together with control points cx, cy and cz which together define the mapping.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    fun : list of callables
        components of 2-form for which the L2-error in V2 shall be computed
        
    coeff : list of array_like
        coefficients of the B-spline space
        
    mapping : int
        0 : analytical mapping
        1 : discrete mapping
        
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
        
    cz : array_like
        z control points in case of discrete mapping
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
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    mat_f1    = fun[0](quad_mesh[0], quad_mesh[1], quad_mesh[2])
    mat_f2    = fun[1](quad_mesh[0], quad_mesh[1], quad_mesh[2])
    mat_f3    = fun[2](quad_mesh[0], quad_mesh[1], quad_mesh[2])
    
    # evaluation of mapping at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    
    # compute error
    error = np.zeros(Nel, dtype=float)
    
    # 1 * f1 * G_11 / sqrt(g) * f1
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 21, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.T[2], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, cz, Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 21)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 1], [0, 1, 1], basisN[0], basisD[1], basisD[2], basisN[0], basisD[1], basisD[2], [NbaseN[0], NbaseD[1], NbaseD[2]], [NbaseN[0], NbaseD[1], NbaseD[2]], error, mat_f1, mat_f1, coeff[0], coeff[0], mat_map)
    
    # 2 * f1 * G_12 / sqrt(g) * f2
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 22, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.T[2], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, cz, Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 22)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 1], [1, 0, 1], basisN[0], basisD[1], basisD[2], basisD[0], basisN[1], basisD[2], [NbaseN[0], NbaseD[1], NbaseD[2]], [NbaseD[0], NbaseN[1], NbaseD[2]], error, mat_f1, mat_f2, coeff[0], coeff[1], 2*mat_map)
    
    # 2 * f1 * G_13 / sqrt(g) * f3
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 24, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.T[2], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, cz, Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 24)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 1], [1, 1, 0], basisN[0], basisD[1], basisD[2], basisD[0], basisD[1], basisN[2], [NbaseN[0], NbaseD[1], NbaseD[2]], [NbaseD[0], NbaseD[1], NbaseN[2]], error, mat_f1, mat_f3, coeff[0], coeff[2], 2*mat_map)
    
    # 1 * f2 * G_22 / sqrt(g) * f2
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 23, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.T[2], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, cz, Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 23)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 1], [1, 0, 1], basisD[0], basisN[1], basisD[2], basisD[0], basisN[1], basisD[2], [NbaseD[0], NbaseN[1], NbaseD[2]], [NbaseD[0], NbaseN[1], NbaseD[2]], error, mat_f2, mat_f2, coeff[1], coeff[1], mat_map)
    
    # 2 * f2 * G_23 / sqrt(g) * f3
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 25, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.T[2], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, cz, Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 25)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 1], [1, 1, 0], basisD[0], basisN[1], basisD[2], basisD[0], basisD[1], basisN[2], [NbaseD[0], NbaseN[1], NbaseD[2]], [NbaseD[0], NbaseD[1], NbaseN[2]], error, mat_f2, mat_f3, coeff[1], coeff[2], 2*mat_map)
    
    # 1 * f3 * G_33 / sqrt(g) * f3
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 26, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.T[2], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, cz, Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 26)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 1, 0], [1, 1, 0], basisD[0], basisD[1], basisN[2], basisD[0], basisD[1], basisN[2], [NbaseD[0], NbaseD[1], NbaseN[2]], [NbaseD[0], NbaseD[1], NbaseN[2]], error, mat_f3, mat_f3, coeff[2], coeff[2], mat_map)
                
    return np.sqrt(error.sum())


# ======= error in V3 ====================
def l2_error_V3(tensor_space_FEM, fun, coeff, mapping, kind_map=None, params_map=None, tensor_space_F=None, cx=None, cy=None, cz=None):
    """
    Computes the 3d L2-error (DDD) of the given tensor product B-spline spaces of multi-degree (p1, p2, p3) with coefficients coeff with the function fun.
    
    In case of an analytical mapping, all quantities related to the mapping are called from hylife.geometry.mappings_analytical which contains a collection of analytical mappings. One must pass the parameters kind_map and params_map.
    
    In case of a discrete mapping, one must pass an additional tensor product B-spline space together with control points cx, cy and cz which together define the mapping.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    fun : callable
        component of 3-form for which the L2-error in V0 shall be computed
        
    coeff : array_like
        coefficients of the B-spline space
        
    mapping : int
        0 : analytical mapping
        1 : discrete mapping
        
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
        
    cz : array_like
        z control points in case of discrete mapping
    """
      
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    NbaseD = tensor_space_FEM.NbaseD  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points
    
    # evaluation of Jacobian determinant at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    
    if   mapping == 0:
        ker.kernel_evaluation_ana(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 2, kind_map, params_map)
    elif mapping == 1:
        ker.kernel_evaluation_dis(tensor_space_F.T[0], tensor_space_F.T[1], tensor_space_F.T[2], tensor_space_F.p, tensor_space_F.NbaseN, cx, cy, cz, Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 2)
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    mat_f     = fun(quad_mesh[0], quad_mesh[1], quad_mesh[2])
    
    # compute error
    error = np.zeros(Nel, dtype=float)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 1, 1], [1, 1, 1], basisD[0], basisD[1], basisD[2], basisD[0], basisD[1], basisD[2], [NbaseD[0], NbaseD[1], NbaseD[2]], [NbaseD[0], NbaseD[1], NbaseD[2]], error, mat_f, mat_f, coeff, coeff, mat_map)
                
    return np.sqrt(error.sum())