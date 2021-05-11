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
def inner_prod_V0(tensor_space_FEM, domain, fun):
    """
    Assembles the 2D inner prodcut (NN, fun) of the given tensor product B-spline space of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
    
    fun : callable
        the 0-form with which the inner products shall be computed
    """
    
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    NbaseN = tensor_space_FEM.NbaseN  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points
    
    # evaluation of |det(DF)| at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], n_quad[0], n_quad[1]), dtype=float)
    
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], mat_map, 1, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.p, domain.NbaseN, domain.cx, domain.cy)
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing='ij') 
    mat_f     = fun(quad_mesh[0], quad_mesh[1])
    
    # assembly
    F = np.zeros((NbaseN[0], NbaseN[1]), dtype=float)
    
    ker.kernel_inner_1(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], 0, 0, wts[0], wts[1], basisN[0], basisN[1], NbaseN[0], NbaseN[1], F, mat_f, mat_map)
                
    return tensor_space_FEM.E0.dot(F.flatten())


# ================ inner product in V1 (H curl) ===========================
def inner_prod_V1C(tensor_space_FEM, domain, fun):
    """
    Assembles the 2D inner prodcut ([DN, ND], [fun_1, fun_2]) of the given tensor product B-spline space of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
    
    fun : list of callables
        the three 1-form components (H curl) with which the inner products shall be computed
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
            ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], mat_map, kind_funs[counter], domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.p, domain.NbaseN, domain.cx, domain.cy)
            
            # evaluate function at quadrature points
            mat_f = fun[b](quad_mesh[0], quad_mesh[1])
            
            ker.kernel_inner_1(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], ni1, ni2, wts[0], wts[1], bi1, bi2, Nbase1, Nbase2, F[a], mat_f, mat_map)
            
            counter += 1
            
    return tensor_space_FEM.E1C.dot(np.concatenate((F[0].flatten(), F[1].flatten())))


# ================ inner product in V1 (H div) ===========================
def inner_prod_V1D(tensor_space_FEM, domain, fun):
    """
    Assembles the 2D inner prodcut ([ND, DN], [fun_1, fun_2]) of the given tensor product B-spline space of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
    
    fun : list of callables
        the three 1-form components (H div) with which the inner products shall be computed
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
            ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], mat_map, kind_funs[counter], domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.p, domain.NbaseN, domain.cx, domain.cy)
            
            # evaluate function at quadrature points
            mat_f = fun[b](quad_mesh[0], quad_mesh[1])
            
            ker.kernel_inner_1(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], ni1, ni2, wts[0], wts[1], bi1, bi2, Nbase1, Nbase2, F[a], mat_f, mat_map)
            
            counter += 1
            
    return tensor_space_FEM.E1D.dot(np.concatenate((F[0].flatten(), F[1].flatten())))


# ================ inner product in V2 ===========================
def inner_prod_V2(tensor_space_FEM, domain, fun):
    """
    Assembles the 2D inner prodcut (DD, fun) of the given tensor product B-spline space of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
    
    fun : callable
        the 2-form component with which the inner products shall be computed
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
    
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], mat_map, 2, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.p, domain.NbaseN, domain.cx, domain.cy)
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing='ij') 
    mat_f     = fun(quad_mesh[0], quad_mesh[1])
    
    # assembly
    F = np.zeros((NbaseD[0], NbaseD[1]), dtype=float)
    
    ker.kernel_inner_1(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], 1, 1, wts[0], wts[1], basisD[0], basisD[1], NbaseD[0], NbaseD[1], F, mat_f, mat_map)
                
    return tensor_space_FEM.E2.dot(F.flatten())