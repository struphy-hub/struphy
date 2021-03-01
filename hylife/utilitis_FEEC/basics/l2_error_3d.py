# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to compute L2-errors of discrete p-forms with analytical forms in 3D.
"""

import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.basics.kernels_3d as ker


# ======= error in V0 ====================
def l2_error_V0(tensor_space_FEM, domain, fun, coeff):
    """
    Computes the 3D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 3D tensor product B-spline space of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    fun : callable
        the 0-form with which the error shall be computed
        
    coeff : array_like
        the FEM coefficients of the discrete function
    """
      
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    NbaseN = tensor_space_FEM.NbaseN  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points
    
    # evaluation of |det(DF)| at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 1, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    mat_f     = fun(quad_mesh[0], quad_mesh[1], quad_mesh[2])
    
    # compute error
    error = np.zeros(Nel, dtype=float)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 0, 0], [0, 0, 0], basisN[0], basisN[1], basisN[2], basisN[0], basisN[1], basisN[2], [NbaseN[0], NbaseN[1], NbaseN[2]], [NbaseN[0], NbaseN[1], NbaseN[2]], error, mat_f, mat_f, coeff, coeff, mat_map)
                
    return np.sqrt(error.sum())


# ======= error in V1 ====================
def l2_error_V1(tensor_space_FEM, domain, fun, coeff):
    """
    Computes the 3D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 3D tensor product B-spline space of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    fun : list of callables
        the three 1-form components with which the error shall be computed
        
    coeff : list of array_like
        the FEM coefficients of the discrete components
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
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 11, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 0], [1, 0, 0], basisD[0], basisN[1], basisN[2], basisD[0], basisN[1], basisN[2], [NbaseD[0], NbaseN[1], NbaseN[2]], [NbaseD[0], NbaseN[1], NbaseN[2]], error, mat_f1, mat_f1, coeff[0], coeff[0], mat_map)
    
    # 2 * f1 * G^12 * sqrt(g) * f2
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 12, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 0], [0, 1, 0], basisD[0], basisN[1], basisN[2], basisN[0], basisD[1], basisN[2], [NbaseD[0], NbaseN[1], NbaseN[2]], [NbaseN[0], NbaseD[1], NbaseN[2]], error, mat_f1, mat_f2, coeff[0], coeff[1], 2*mat_map)
    
    # 2 * f1 * G^13 * sqrt(g) * f3
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 14, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 0], [0, 0, 1], basisD[0], basisN[1], basisN[2], basisN[0], basisN[1], basisD[2], [NbaseD[0], NbaseN[1], NbaseN[2]], [NbaseN[0], NbaseN[1], NbaseD[2]], error, mat_f1, mat_f3, coeff[0], coeff[2], 2*mat_map)
    
    # 1 * f2 * G^22 * sqrt(g) * f2
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 13, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 0], [0, 1, 0], basisN[0], basisD[1], basisN[2], basisN[0], basisD[1], basisN[2], [NbaseN[0], NbaseD[1], NbaseN[2]], [NbaseN[0], NbaseD[1], NbaseN[2]], error, mat_f2, mat_f2, coeff[1], coeff[1], mat_map)
    
    # 2 * f2 * G^23 * sqrt(g) * f3
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 15, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 0], [0, 0, 1], basisN[0], basisD[1], basisN[2], basisN[0], basisN[1], basisD[2], [NbaseN[0], NbaseD[1], NbaseN[2]], [NbaseN[0], NbaseN[1], NbaseD[2]], error, mat_f2, mat_f3, coeff[1], coeff[2], 2*mat_map)
    
    # 1 * f3 * G^33 * sqrt(g) * f3
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 16, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 0, 1], [0, 0, 1], basisN[0], basisN[1], basisD[2], basisN[0], basisN[1], basisD[2], [NbaseN[0], NbaseN[1], NbaseD[2]], [NbaseN[0], NbaseN[1], NbaseD[2]], error, mat_f3, mat_f3, coeff[2], coeff[2], mat_map)
                
    return np.sqrt(error.sum())


# ======= error in V2 ====================
def l2_error_V2(tensor_space_FEM, domain, fun, coeff):
    """
    Computes the 3D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 3D tensor product B-spline space of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    fun : list of callables
        the three 2-form components with which the error shall be computed
        
    coeff : list of array_like
        the FEM coefficients of the discrete components
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
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 21, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 1], [0, 1, 1], basisN[0], basisD[1], basisD[2], basisN[0], basisD[1], basisD[2], [NbaseN[0], NbaseD[1], NbaseD[2]], [NbaseN[0], NbaseD[1], NbaseD[2]], error, mat_f1, mat_f1, coeff[0], coeff[0], mat_map)
    
    # 2 * f1 * G_12 / sqrt(g) * f2
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 22, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 1], [1, 0, 1], basisN[0], basisD[1], basisD[2], basisD[0], basisN[1], basisD[2], [NbaseN[0], NbaseD[1], NbaseD[2]], [NbaseD[0], NbaseN[1], NbaseD[2]], error, mat_f1, mat_f2, coeff[0], coeff[1], 2*mat_map)
    
    # 2 * f1 * G_13 / sqrt(g) * f3
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 24, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [0, 1, 1], [1, 1, 0], basisN[0], basisD[1], basisD[2], basisD[0], basisD[1], basisN[2], [NbaseN[0], NbaseD[1], NbaseD[2]], [NbaseD[0], NbaseD[1], NbaseN[2]], error, mat_f1, mat_f3, coeff[0], coeff[2], 2*mat_map)
    
    # 1 * f2 * G_22 / sqrt(g) * f2
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 23, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 1], [1, 0, 1], basisD[0], basisN[1], basisD[2], basisD[0], basisN[1], basisD[2], [NbaseD[0], NbaseN[1], NbaseD[2]], [NbaseD[0], NbaseN[1], NbaseD[2]], error, mat_f2, mat_f2, coeff[1], coeff[1], mat_map)
    
    # 2 * f2 * G_23 / sqrt(g) * f3
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 25, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 0, 1], [1, 1, 0], basisD[0], basisN[1], basisD[2], basisD[0], basisD[1], basisN[2], [NbaseD[0], NbaseN[1], NbaseD[2]], [NbaseD[0], NbaseD[1], NbaseN[2]], error, mat_f2, mat_f3, coeff[1], coeff[2], 2*mat_map)
    
    # 1 * f3 * G_33 / sqrt(g) * f3
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 26, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 1, 0], [1, 1, 0], basisD[0], basisD[1], basisN[2], basisD[0], basisD[1], basisN[2], [NbaseD[0], NbaseD[1], NbaseN[2]], [NbaseD[0], NbaseD[1], NbaseN[2]], error, mat_f3, mat_f3, coeff[2], coeff[2], mat_map)
                
    return np.sqrt(error.sum())


# ======= error in V3 ====================
def l2_error_V3(tensor_space_FEM, domain, fun, coeff):
    """
    Computes the 3D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 3D tensor product B-spline space of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    fun : callable
        the 3-form with which the error shall be computed
        
    coeff : array_like
        the FEM coefficients of the discrete function
    """
      
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    NbaseD = tensor_space_FEM.NbaseD  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points
    
    # evaluation of 1(|det(DF)| at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 2, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    mat_f     = fun(quad_mesh[0], quad_mesh[1], quad_mesh[2])
    
    # compute error
    error = np.zeros(Nel, dtype=float)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], wts[2], [1, 1, 1], [1, 1, 1], basisD[0], basisD[1], basisD[2], basisD[0], basisD[1], basisD[2], [NbaseD[0], NbaseD[1], NbaseD[2]], [NbaseD[0], NbaseD[1], NbaseD[2]], error, mat_f, mat_f, coeff, coeff, mat_map)
                
    return np.sqrt(error.sum())