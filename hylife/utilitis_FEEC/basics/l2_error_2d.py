# coding: utf-8
#
# Copyright 2021 Florian Holderied

"""
Modules to compute L2-errors of discrete p-forms with analytical forms in 2D.
"""

import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.basics.kernels_2d as ker


# ======= error in V0 ====================
def l2_error_V0(tensor_space_FEM, domain, fun, coeff):
    """
    Computes the 2D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 2D tensor product B-spline space of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    fun : callable or np.ndarray
        the 0-form with which the error shall be computed
        
    coeff : array_like
        the FEM coefficients of the discrete 0-form
    """
      
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    indN   = tensor_space_FEM.indN    # global indices of local non-vanishing basis functions in format (element, global index)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points
    
    # extract coefficients to tensor-product space
    coeff  = tensor_space_FEM.extract_0form(coeff)
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1*nq1, Nel2*nq2)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'det_df'))[:, :, 0]
    
    # evaluation of given 0-form at quadrature points
    mat_f  = np.empty((pts[0].size, pts[1].size), dtype=float)
    
    if callable(fun):
        quad_mesh   = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing='ij')
        mat_f[:, :] = fun(quad_mesh[0], quad_mesh[1])
    else:
        mat_f[:, :] = fun
    
    # compute error
    error = np.zeros(Nel, dtype=float)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], [0, 0], [0, 0], basisN[0], basisN[1], basisN[0], basisN[1], indN[0], indN[1], indN[0], indN[1], error, mat_f, mat_f, coeff, coeff, det_df)
                
    return np.sqrt(error.sum())


# ======= error in V1 ====================
def l2_error_V1(tensor_space_FEM, domain, fun, coeff):
    """
    Computes the 2D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 2D tensor product B-spline space of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    fun : list of callables or np.ndarrays
        the three 1-form components with which the error shall be computed
        
    coeff : list of array_like
        the FEM coefficients of the discrete components
    """
      
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    indN   = tensor_space_FEM.indN    # global indices of non-vanishing basis functions (N) in format (element, global index) 
    indD   = tensor_space_FEM.indD    # global indices of non-vanishing basis functions (D) in format (element, global index)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points
    wts    = tensor_space_FEM.wts     # global quadrature weights
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points (D)
    
    # extract coefficients to tensor-product space
    coeff1, coeff2, coeff3 = tensor_space_FEM.extract_1form(coeff)
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1*nq1, Nel2*nq2)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'det_df'))[:, :, 0]
    
    # evaluation of given 1-form components at quadrature points
    mat_f1 = np.empty((pts[0].size, pts[1].size), dtype=float)
    mat_f2 = np.empty((pts[0].size, pts[1].size), dtype=float)
    mat_f3 = np.empty((pts[0].size, pts[1].size), dtype=float)
    
    if callable(fun[0]):
        quad_mesh    = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing='ij') 
        mat_f1[:, :] = fun[0](quad_mesh[0], quad_mesh[1])
        mat_f2[:, :] = fun[1](quad_mesh[0], quad_mesh[1])
        mat_f3[:, :] = fun[2](quad_mesh[0], quad_mesh[1])
    else:
        mat_f1[:, :] = fun[0]
        mat_f2[:, :] = fun[1]
        mat_f3[:, :] = fun[2]
    
    # compute error
    error = np.zeros(Nel, dtype=float)
    
    # 1 * f1 * G^11 * |det(DF)| * f1
    g_inv = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'g_inv_11')[:, :, 0]

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], [1, 0], [1, 0], basisD[0], basisN[1], basisD[0], basisN[1], indD[0], indN[1], indD[0], indN[1], error, mat_f1, mat_f1, coeff1, coeff1, 1*g_inv*det_df)
    
    # 2 * f1 * G^12 * |det(DF)| * f2
    g_inv = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'g_inv_12')[:, :, 0]

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], [1, 0], [0, 1], basisD[0], basisN[1], basisN[0], basisD[1], indD[0], indN[1], indN[0], indD[1], error, mat_f1, mat_f2, coeff1, coeff2, 2*g_inv*det_df)
    
    # 2 * f1 * G^13 * |det(DF)| * f3
    g_inv = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'g_inv_13')[:, :, 0]

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], [1, 0], [0, 0], basisD[0], basisN[1], basisN[0], basisN[1], indD[0], indN[1], indN[0], indN[1], error, mat_f1, mat_f3, coeff1, coeff3, 2*g_inv*det_df)
    
    # 1 * f2 * G^22 * |det(DF)| * f2
    g_inv = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'g_inv_22')[:, :, 0]

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], [0, 1], [0, 1], basisN[0], basisD[1], basisN[0], basisD[1], indN[0], indD[1], indN[0], indD[1], error, mat_f2, mat_f2, coeff2, coeff2, 1*g_inv*det_df)
    
    # 2 * f2 * G^23 * |det(DF)| * f3
    g_inv = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'g_inv_23')[:, :, 0]

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], [0, 1], [0, 0], basisN[0], basisD[1], basisN[0], basisN[1], indN[0], indD[1], indN[0], indN[1], error, mat_f2, mat_f3, coeff2, coeff3, 2*g_inv*det_df)
    
    # 1 * f3 * G^33 * |det(DF)| * f3
    g_inv = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'g_inv_33')[:, :, 0]

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], [0, 0], [0, 0], basisN[0], basisN[1], basisN[0], basisN[1], indN[0], indN[1], indN[0], indN[1], error, mat_f3, mat_f3, coeff3, coeff3, 1*g_inv*det_df)
                
    return np.sqrt(error.sum())


# ======= error in V2 ====================
def l2_error_V2(tensor_space_FEM, domain, fun, coeff):
    """
    Computes the 2D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 2D tensor product B-spline space of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    fun : list of callables or np.ndarrays
        the three 2-form components with which the error shall be computed
        
    coeff : list of array_like
        the FEM coefficients of the discrete components
    """
      
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    indN   = tensor_space_FEM.indN    # global indices of non-vanishing basis functions (N) in format (element, global index) 
    indD   = tensor_space_FEM.indD    # global indices of non-vanishing basis functions (D) in format (element, global index)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points
    wts    = tensor_space_FEM.wts     # global quadrature weights
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points (D)
    
    # extract coefficients to tensor-product space
    coeff1, coeff2, coeff3 = tensor_space_FEM.extract_2form(coeff)
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1*nq1, Nel2*nq2)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'det_df'))[:, :, 0]
    
    # evaluation of given 2-form components at quadrature points
    mat_f1 = np.empty((pts[0].size, pts[1].size), dtype=float)
    mat_f2 = np.empty((pts[0].size, pts[1].size), dtype=float)
    mat_f3 = np.empty((pts[0].size, pts[1].size), dtype=float)
    
    if callable(fun[0]):
        quad_mesh    = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing='ij') 
        mat_f1[:, :] = fun[0](quad_mesh[0], quad_mesh[1])
        mat_f2[:, :] = fun[1](quad_mesh[0], quad_mesh[1])
        mat_f3[:, :] = fun[2](quad_mesh[0], quad_mesh[1])
    else:
        mat_f1[:, :] = fun[0]
        mat_f2[:, :] = fun[1]
        mat_f3[:, :] = fun[2]
    
    # compute error
    error = np.zeros(Nel, dtype=float)
    
    # 1 * f1 * G_11 / |det(DF)| * f1
    g = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'g_11')[:, :, 0]

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], [0, 1], [0, 1], basisN[0], basisD[1], basisN[0], basisD[1], indN[0], indD[1], indN[0], indD[1], error, mat_f1, mat_f1, coeff1, coeff1, 1*g/det_df)
    
    # 2 * f1 * G_12 / |det(DF)| * f2
    g = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'g_12')[:, :, 0]

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], [0, 1], [1, 0], basisN[0], basisD[1], basisD[0], basisN[1], indN[0], indD[1], indD[0], indN[1], error, mat_f1, mat_f2, coeff1, coeff2, 2*g/det_df)
    
    # 2 * f1 * G_13 / |det(DF)| * f3
    g = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'g_13')[:, :, 0]

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], [0, 1], [1, 1], basisN[0], basisD[1], basisD[0], basisD[1], indN[0], indD[1], indD[0], indD[1], error, mat_f1, mat_f3, coeff1, coeff3, 2*g/det_df)
    
    # 1 * f2 * G_22 / |det(DF)| * f2
    g = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'g_22')[:, :, 0]

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], [1, 0], [1, 0], basisD[0], basisN[1], basisD[0], basisN[1], indD[0], indN[1], indD[0], indN[1], error, mat_f2, mat_f2, coeff2, coeff2, 1*g/det_df)
    
    # 2 * f2 * G_23 / |det(DF)| * f3
    g = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'g_23')[:, :, 0]

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], [1, 0], [1, 1], basisD[0], basisN[1], basisD[0], basisD[1], indD[0], indN[1], indD[0], indD[1], error, mat_f2, mat_f3, coeff2, coeff3, 2*g/det_df)
    
    # 1 * f3 * G_33 / |det(DF)| * f3
    g = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'g_33')[:, :, 0]

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], [1, 1], [1, 1], basisD[0], basisD[1], basisD[0], basisD[1], indD[0], indD[1], indD[0], indD[1], error, mat_f3, mat_f3, coeff3, coeff3, 1*g/det_df)
                
    return np.sqrt(error.sum())


# ======= error in V3 ====================
def l2_error_V3(tensor_space_FEM, domain, fun, coeff):
    """
    Computes the 2D L2-error of (fun - fun_h) of the analytical function fun with the discrete function fun_h living in a 2D tensor product B-spline space of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    fun : callable or np.ndarray
        the 3-form component with which the error shall be computed
        
    coeff : array_like
        the FEM coefficients of the discrete function
    """
      
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    indD   = tensor_space_FEM.indD    # global indices of local non-vanishing basis functions in format (element, global index)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points
    
    # extract coefficients to tensor-product space
    coeff  = tensor_space_FEM.extract_3form(coeff)
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1*nq1, Nel2*nq2)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'det_df'))[:, :, 0]
    
    # evaluation of given 3-form at quadrature points
    mat_f  = np.empty((pts[0].size, pts[1].size), dtype=float)
    
    if callable(fun):
        quad_mesh   = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing='ij')
        mat_f[:, :] = fun(quad_mesh[0], quad_mesh[1])
    else:
        mat_f[:, :] = fun
    
    # compute error
    error = np.zeros(Nel, dtype=float)

    ker.kernel_l2error(Nel, p, n_quad, wts[0], wts[1], [1, 1], [1, 1], basisD[0], basisD[1], basisD[0], basisD[1], indD[0], indD[1], indD[0], indD[1], error, mat_f, mat_f, coeff, coeff, 1/det_df)
                
    return np.sqrt(error.sum())