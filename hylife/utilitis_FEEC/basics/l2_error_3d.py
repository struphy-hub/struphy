# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to compute L2 - error in 3d.
"""


import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.basics.kernels_3d as ker


# ======= error in V0 ====================
def l2_error_V0(tensor_space, kind_map, params_map, coeff, fun):
    """
    Computes the 3d L2 - error (NNN) of the given B-spline space of degree p with coefficients coeff with the function fun.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        a 1d B-spline space
        
    kind_map : int
        type of mapping
        
    params_map : list of doubles
        parameters for the mapping
        
    coeff : array_like
        coefficients of the spline space
        
    fun : callable
        function for which the error shall be computed
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
    
    # compute error
    error = np.zeros(Nel, dtype=float, order='F')

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 0, 0], [0, 0, 0], basisN[0], basisN[1], basisN[2], basisN[0], basisN[1], basisN[2], [NbaseN[0], NbaseN[1], NbaseN[2]], [NbaseN[0], NbaseN[1], NbaseN[2]], error, mat_f, mat_f, np.asfortranarray(coeff), np.asfortranarray(coeff), mat_map)
                
    return np.sqrt(error.sum())


# ======= error in V1 ====================
def l2_error_V1(tensor_space, kind_map, params_map, coeff, fun):
    """
    Computes the 3d L2 - error (DNN, NDN, NND) of the given B-spline space of degree p with coefficients coeff with the function fun.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        a 1d B-spline space
        
    kind_map : int
        type of mapping
        
    params_map : list of doubles
        parameters for the mapping
        
    coeff : list of array_like
        coefficients of the spline space
        
    fun : list of callables
        components of function for which the error shall be computed
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
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    mat_f1    = np.asfortranarray(fun[0](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    mat_f2    = np.asfortranarray(fun[1](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    mat_f3    = np.asfortranarray(fun[2](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    
    # evaluation of mapping at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float, order='F')
    
    # compute error
    error = np.zeros(Nel, dtype=float, order='F')
    
    # 1 * f1 * G^11 * sqrt(g) * f1
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 11, kind_map, params_map)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 0], [1, 0, 0], basisD[0], basisN[1], basisN[2], basisD[0], basisN[1], basisN[2], [NbaseD[0], NbaseN[1], NbaseN[2]], [NbaseD[0], NbaseN[1], NbaseN[2]], error, mat_f1, mat_f1, np.asfortranarray(coeff[0]), np.asfortranarray(coeff[0]), mat_map)
    
    # 2 * f1 * G^12 * sqrt(g) * f2
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 12, kind_map, params_map)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 0], [0, 1, 0], basisD[0], basisN[1], basisN[2], basisN[0], basisD[1], basisN[2], [NbaseD[0], NbaseN[1], NbaseN[2]], [NbaseN[0], NbaseD[1], NbaseN[2]], error, mat_f1, mat_f2, np.asfortranarray(coeff[0]), np.asfortranarray(coeff[1]), 2*mat_map)
    
    # 2 * f1 * G^13 * sqrt(g) * f3
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 14, kind_map, params_map)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 0], [0, 0, 1], basisD[0], basisN[1], basisN[2], basisN[0], basisN[1], basisD[2], [NbaseD[0], NbaseN[1], NbaseN[2]], [NbaseN[0], NbaseN[1], NbaseD[2]], error, mat_f1, mat_f3, np.asfortranarray(coeff[0]), np.asfortranarray(coeff[2]), 2*mat_map)
    
    # 1 * f2 * G^22 * sqrt(g) * f2
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 13, kind_map, params_map)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 0], [0, 1, 0], basisN[0], basisD[1], basisN[2], basisN[0], basisD[1], basisN[2], [NbaseN[0], NbaseD[1], NbaseN[2]], [NbaseN[0], NbaseD[1], NbaseN[2]], error, mat_f2, mat_f2, np.asfortranarray(coeff[1]), np.asfortranarray(coeff[1]), mat_map)
    
    # 2 * f2 * G^23 * sqrt(g) * f3
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 15, kind_map, params_map)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 0], [0, 0, 1], basisN[0], basisD[1], basisN[2], basisN[0], basisN[1], basisD[2], [NbaseN[0], NbaseD[1], NbaseN[2]], [NbaseN[0], NbaseN[1], NbaseD[2]], error, mat_f2, mat_f3, np.asfortranarray(coeff[1]), np.asfortranarray(coeff[2]), 2*mat_map)
    
    # 1 * f3 * G^33 * sqrt(g) * f3
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 16, kind_map, params_map)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 0, 1], [0, 0, 1], basisN[0], basisN[1], basisD[2], basisN[0], basisN[1], basisD[2], [NbaseN[0], NbaseN[1], NbaseD[2]], [NbaseN[0], NbaseN[1], NbaseD[2]], error, mat_f3, mat_f3, np.asfortranarray(coeff[2]), np.asfortranarray(coeff[2]), mat_map)
                
    return np.sqrt(error.sum())


# ======= error in V2 ====================
def l2_error_V2(tensor_space, kind_map, params_map, coeff, fun):
    """
    Computes the 3d L2 - error (NDD, DND, DDN) of the given B-spline space of degree p with coefficients coeff with the function fun.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        a 1d B-spline space
        
    kind_map : int
        type of mapping
        
    params_map : list of doubles
        parameters for the mapping
        
    coeff : list of array_like
        coefficients of the spline space
        
    fun : list of callables
        components of function for which the error shall be computed
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
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    mat_f1    = np.asfortranarray(fun[0](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    mat_f2    = np.asfortranarray(fun[1](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    mat_f3    = np.asfortranarray(fun[2](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    
    # evaluation of mapping at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float, order='F')
    
    # compute error
    error = np.zeros(Nel, dtype=float, order='F')
    
    # 1 * f1 * G_11 / sqrt(g) * f1
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 21, kind_map, params_map)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 1], [0, 1, 1], basisN[0], basisD[1], basisD[2], basisN[0], basisD[1], basisD[2], [NbaseN[0], NbaseD[1], NbaseD[2]], [NbaseN[0], NbaseD[1], NbaseD[2]], error, mat_f1, mat_f1, np.asfortranarray(coeff[0]), np.asfortranarray(coeff[0]), mat_map)
    
    # 2 * f1 * G_12 / sqrt(g) * f2
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 22, kind_map, params_map)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 1], [1, 0, 1], basisN[0], basisD[1], basisD[2], basisD[0], basisN[1], basisD[2], [NbaseN[0], NbaseD[1], NbaseD[2]], [NbaseD[0], NbaseN[1], NbaseD[2]], error, mat_f1, mat_f2, np.asfortranarray(coeff[0]), np.asfortranarray(coeff[1]), 2*mat_map)
    
    # 2 * f1 * G_13 / sqrt(g) * f3
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 24, kind_map, params_map)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 1], [1, 1, 0], basisN[0], basisD[1], basisD[2], basisD[0], basisD[1], basisN[2], [NbaseN[0], NbaseD[1], NbaseD[2]], [NbaseD[0], NbaseD[1], NbaseN[2]], error, mat_f1, mat_f3, np.asfortranarray(coeff[0]), np.asfortranarray(coeff[2]), 2*mat_map)
    
    # 1 * f2 * G_22 / sqrt(g) * f2
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 23, kind_map, params_map)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 1], [1, 0, 1], basisD[0], basisN[1], basisD[2], basisD[0], basisN[1], basisD[2], [NbaseD[0], NbaseN[1], NbaseD[2]], [NbaseD[0], NbaseN[1], NbaseD[2]], error, mat_f2, mat_f2, np.asfortranarray(coeff[1]), np.asfortranarray(coeff[1]), mat_map)
    
    # 2 * f2 * G_23 / sqrt(g) * f3
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 25, kind_map, params_map)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 1], [1, 1, 0], basisD[0], basisN[1], basisD[2], basisD[0], basisD[1], basisN[2], [NbaseD[0], NbaseN[1], NbaseD[2]], [NbaseD[0], NbaseD[1], NbaseN[2]], error, mat_f2, mat_f3, np.asfortranarray(coeff[1]), np.asfortranarray(coeff[2]), 2*mat_map)
    
    # 1 * f3 * G_33 / sqrt(g) * f3
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 26, kind_map, params_map)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 1, 0], [1, 1, 0], basisD[0], basisD[1], basisN[2], basisD[0], basisD[1], basisN[2], [NbaseD[0], NbaseD[1], NbaseN[2]], [NbaseD[0], NbaseD[1], NbaseN[2]], error, mat_f3, mat_f3, np.asfortranarray(coeff[2]), np.asfortranarray(coeff[2]), mat_map)
                
    return np.sqrt(error.sum())


# ======= error in V3 ====================
def l2_error_V3(tensor_space, kind_map, params_map, coeff, fun):
    """
    Computes the 3d L2 - error (DDD) of the given B-spline space of degree p with coefficients coeff with the function fun.
    
    Parameters
    ----------
    tensor_space : tensor_spline_space
        a 1d B-spline space
        
    kind_map : int
        type of mapping
        
    params_map : list of doubles
        parameters for the mapping
        
    coeff : array_like
        coefficients of the spline space
        
    fun : callable
        function for which the error shall be computed
    """
      
    p      = tensor_space.p       # spline degrees
    Nel    = tensor_space.Nel     # number of elements
    NbaseD = tensor_space.NbaseD  # total number of basis functions (N)
    
    n_quad = tensor_space.n_quad  # number of quadrature points per element
    pts    = tensor_space.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space.wts     # global quadrature weights in format (element, local weight)
    
    basisD = tensor_space.basisD  # evaluated basis functions at quadrature points
    
    # evaluation of Jacobian determinant at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float, order='F')
    ker.kernel_evaluation(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 2, kind_map, params_map)
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    mat_f     = np.asfortranarray(fun(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    
    # compute error
    error = np.zeros(Nel, dtype=float, order='F')

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 1, 1], [1, 1, 1], basisD[0], basisD[1], basisD[2], basisD[0], basisD[1], basisD[2], [NbaseD[0], NbaseD[1], NbaseD[2]], [NbaseD[0], NbaseD[1], NbaseD[2]], error, mat_f, mat_f, np.asfortranarray(coeff), np.asfortranarray(coeff), mat_map)
                
    return np.sqrt(error.sum())