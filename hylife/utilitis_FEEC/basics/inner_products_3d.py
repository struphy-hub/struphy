# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to compute inner products with given functions in 3D.
"""


import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.basics.kernels_3d as ker


# ================ inner product in V0 ===========================
def inner_prod_V0(tensor_space_FEM, domain, fun):
    """
    Assembles the 3D inner prodcut (NNN, fun) of the given tensor product B-spline space of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
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
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 1, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    mat_f     = fun(quad_mesh[0], quad_mesh[1], quad_mesh[2])
    
    # assembly
    F = np.zeros((NbaseN[0], NbaseN[1], NbaseN[2]), dtype=float)
    
    ker.kernel_inner_1(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], 0, 0, 0, wts[0], wts[1], wts[2], basisN[0], basisN[1], basisN[2], NbaseN[0], NbaseN[1], NbaseN[2], F, mat_f, mat_map)
                
    return F


# ================ inner product in V1 ===========================
def inner_prod_V1(tensor_space_FEM, domain, fun):
    """
    Assembles the 3D inner prodcut ([DNN, NDN, NND], [fun_1, fun_2, fun_3]) of the given tensor product B-spline space of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
    
    fun : list of callables
        the three 1-form components with which the inner products shall be computed
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
    basis = [[basisD[0], basisN[1], basisN[2]], 
             [basisN[0], basisD[1], basisN[2]], 
             [basisN[0], basisN[1], basisD[2]]]
    
    Nbase = [[NbaseD[0], NbaseN[1], NbaseN[2]], 
             [NbaseN[0], NbaseD[1], NbaseN[2]], 
             [NbaseN[0], NbaseN[1], NbaseD[2]]]
    
    ns    = [[1, 0, 0], 
             [0, 1, 0], 
             [0, 0, 1]]
    
    # G^(-1)|det(DF)| at quadrature points
    mat_map   = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    kind_funs = [11, 12, 14, 12, 13, 15, 14, 15, 16]
    
    # function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    
    # components of global inner product
    F = [np.zeros((Nbase[0], Nbase[1], Nbase[2])) for Nbase in Nbase]
    
    # assembly
    counter = 0
    
    for a in range(3):
        
        ni1,    ni2,    ni3    = ns[a]
        bi1,    bi2,    bi3    = basis[a]
        Nbase1, Nbase2, Nbase3 = Nbase[a]
        
        for b in range(3):
            
            # evaluate G^(-1)|det(DF)| at quadrature points
            ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, kind_funs[counter], domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
            
            # evaluate function at quadrature points
            mat_f = fun[b](quad_mesh[0], quad_mesh[1], quad_mesh[2])
            
            ker.kernel_inner_1(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ni1, ni2, ni3, wts[0], wts[1], wts[2], bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, F[a], mat_f, mat_map)
            
            counter += 1
            
    return F


# ================ inner product in V2 ===========================
def inner_prod_V2(tensor_space_FEM, domain, fun):
    """
    Assembles the 3D inner prodcut ([NDD, DND, DDN], [fun_1, fun_2, fun_3]) of the given tensor product B-spline space of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
    
    fun : list of callables
        the three 2-form components with which the inner products shall be computed
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
    
    # basis functions of components of a 2-form
    basis = [[basisN[0], basisD[1], basisD[2]], 
             [basisD[0], basisN[1], basisD[2]], 
             [basisD[0], basisD[1], basisN[2]]]
    
    Nbase = [[NbaseN[0], NbaseD[1], NbaseD[2]], 
             [NbaseD[0], NbaseN[1], NbaseD[2]], 
             [NbaseD[0], NbaseD[1], NbaseN[2]]]
    
    ns    = [[0, 1, 1], 
             [1, 0, 1], 
             [1, 1, 0]]
    
    # G/|det(DF)| at quadrature points
    mat_map   = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    kind_funs = [21, 22, 24, 22, 23, 25, 24, 25, 26]
    
    # function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    
    # components of global inner product
    F = [np.zeros((Nbase[0], Nbase[1], Nbase[2])) for Nbase in Nbase]
    
    # assembly
    counter = 0
    
    for a in range(3):
        
        ni1,    ni2,    ni3    = ns[a]
        bi1,    bi2,    bi3    = basis[a]
        Nbase1, Nbase2, Nbase3 = Nbase[a]
        
        for b in range(3):
            
            # evaluate G/|det(DF)| at quadrature points
            ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, kind_funs[counter], domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
            
            # evaluate function at quadrature points
            mat_f = fun[b](quad_mesh[0], quad_mesh[1], quad_mesh[2])
            
            ker.kernel_inner_1(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ni1, ni2, ni3, wts[0], wts[1], wts[2], bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, F[a], mat_f, mat_map)
            
            counter += 1
            
    return F


# ================ inner product in V3 ===========================
def inner_prod_V3(tensor_space_FEM, domain, fun):
    """
    Assembles the 3D inner prodcut (DDD, fun) of the given tensor product B-spline space of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
    
    fun : callable
        the 3-form component with which the inner products shall be computed
    """
    
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    NbaseD = tensor_space_FEM.NbaseD  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points
    
    # evaluation of 1/|det(DF)| at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 2, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
    
    # evaluation of function at quadrature points
    quad_mesh = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij') 
    mat_f     = fun(quad_mesh[0], quad_mesh[1], quad_mesh[2])
    
    # assembly
    F = np.zeros((NbaseD[0], NbaseD[1], NbaseD[2]), dtype=float)
    
    ker.kernel_inner_1(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], 1, 1, 1, wts[0], wts[1], wts[2], basisD[0], basisD[1], basisD[2], NbaseD[0], NbaseD[1], NbaseD[2], F, mat_f, mat_map)
                
    return F