# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to compute mass matrices in 2D.
"""


import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.basics.kernels_2d as ker


# ================ mass matrix in V0 ===========================
def mass_V0(tensor_space_FEM, kind_map, params_map=None, tensor_space_F=None, cx=None, cy=None):
    """
    ----------------------------------------------------------------------------------------------------------
    Assembles the 2D mass matrix [[NN NN]] of the given tensor product B-spline spaces of bi-degree (p1, p2).
    
    In case of an analytical mapping (kind_map >= 1), all mapping related quantities are called from hylife.geometry.mappings_2d. One must then pass the parameter list params_map.
    
    In case of a discrete mapping (kind_map = 0), one must pass a 2D tensor product B-spline space tensor_space_F together with control points cx and cy which together define the mapping.
    -----------------------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
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
    bc     = tensor_space_FEM.bc      # boundary conditions (periodic vs. clamped)
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


# ================ mass matrix in V1 (H curl) ===========================
def mass_V1_curl(tensor_space_FEM, kind_map, params_map=None, tensor_space_F=None, cx=None, cy=None):
    """
    ----------------------------------------------------------------------------------------------------------
    Assembles the 2D mass matrix [[DN DN, DN ND], [ND DN, ND ND]] of the given tensor product B-spline space of bi-degree (p1, p2).
    
    In case of an analytical mapping (kind_map >= 1), all mapping related quantities are called from hylife.geometry.mappings_2d. One must then pass the parameter list params_map.
    
    In case of a discrete mapping (kind_map = 0), one must pass a 2D tensor product B-spline space tensor_space_F together with control points cx and cy which together define the mapping.
    -----------------------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
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
    bc     = tensor_space_FEM.bc      # boundary conditions (periodic vs. clamped)
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
    
    # blocks   11         21         22
    Nbi1 = [NbaseD[0], NbaseN[0], NbaseN[0]]
    Nbi2 = [NbaseN[1], NbaseD[1], NbaseD[1]]
    
    Nbj1 = [NbaseD[0], NbaseD[0], NbaseN[0]]
    Nbj2 = [NbaseN[1], NbaseN[1], NbaseD[1]]
    
    # basis functions of components of a 1-form
    basis = [[basisD[0], basisN[1]], 
             [basisN[0], basisD[1]]]
    
    ns    = [[1, 0], 
             [0, 1]]
    
    # G^(-1)|det(DF)| at quadrature points
    mat_map   = np.empty((Nel[0], Nel[1], n_quad[0], n_quad[1]), dtype=float)
    kind_funs = [11, 12, 13]
    
    # blocks of global mass matrix
    M = [np.zeros((Nbi1, Nbi2, 2*p[0] + 1, 2*p[1] + 1), dtype=float) for Nbi1, Nbi2 in zip(Nbi1, Nbi2)]
    
    # assembly of blocks 11, 21, 22
    counter = 0
    
    for a in range(2):
        for b in range(a + 1):
            
            # evaluate G^(-1)|det(DF)| at quadrature points
            ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], mat_map, kind_funs[counter], kind_map, params_map, T_F[0], T_F[1], p_F, NbaseN_F, cx, cy)
            
            ni1, ni2 = ns[a]
            nj1, nj2 = ns[b]
            
            bi1, bi2 = basis[a]
            bj1, bj2 = basis[b]
            
            ker.kernel_mass(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], ni1, ni2, nj1, nj2, wts[0], wts[1], bi1, bi2, bj1, bj2, Nbi1[counter], Nbi2[counter], M[counter], mat_map)
            
            indices = np.indices((Nbi1[counter], Nbi2[counter], 2*p[0] + 1, 2*p[1] + 1))
            
            shift1  = np.arange(Nbi1[counter]) - p[0]
            shift2  = np.arange(Nbi2[counter]) - p[1]
            
            row     = (Nbi2[counter]*indices[0] + indices[1]).flatten()
            
            col1    = (indices[2] + shift1[:, None, None, None])%Nbj1[counter]
            col2    = (indices[3] + shift2[None, :, None, None])%Nbj2[counter]
            
            col     = Nbj2[counter]*col1 + col2
            
            M[counter] = spa.csc_matrix((M[counter].flatten(), (row, col.flatten())), shape=(Nbi1[counter]*Nbi2[counter], Nbj1[counter]*Nbj2[counter]))
            M[counter].eliminate_zeros()
            
            counter += 1
                       
    M = spa.bmat([[M[0], M[1].T], [M[1], M[2]]], format='csc')
                
    return M


# ================ mass matrix in V1 (H div) ===========================
def mass_V1_div(tensor_space_FEM, kind_map, params_map=None, tensor_space_F=None, cx=None, cy=None):
    """
    ----------------------------------------------------------------------------------------------------------
    Assembles the 2D mass matrix [[ND ND, ND DN], [DN ND, DN DN]] of the given tensor product B-spline space of bi-degree (p1, p2).
    
    In case of an analytical mapping (kind_map >= 1), all mapping related quantities are called from hylife.geometry.mappings_2d. One must then pass the parameter list params_map.
    
    In case of a discrete mapping (kind_map = 0), one must pass a 2D tensor product B-spline space tensor_space_F together with control points cx and cy which together define the mapping.
    -----------------------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
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
    bc     = tensor_space_FEM.bc      # boundary conditions (periodic vs. clamped)
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
    
    # blocks   11         21         22
    Nbi1   = [NbaseN[0], NbaseD[0], NbaseD[0]]
    Nbi2   = [NbaseD[1], NbaseN[1], NbaseN[1]]
    
    Nbj1   = [NbaseN[0], NbaseN[0], NbaseD[0]]
    Nbj2   = [NbaseD[1], NbaseD[1], NbaseN[1]]
    
    # basis functions of components of a 1-form
    basis = [[basisN[0], basisD[1]], 
             [basisD[0], basisN[1]]]
    
    ns    = [[0, 1], 
             [1, 0]]
    
    # G/|det(DF)| at quadrature points
    mat_map   = np.empty((Nel[0], Nel[1], n_quad[0], n_quad[1]), dtype=float)
    kind_funs = [21, 22, 23]
    
    # blocks of global mass matrix
    M = [np.zeros((Nbi1, Nbi2, 2*p[0] + 1, 2*p[1] + 1), dtype=float) for Nbi1, Nbi2 in zip(Nbi1, Nbi2)]
    
    # assembly of blocks 11, 21, 22
    counter = 0
    
    for a in range(2):
        for b in range(a + 1):
            
            # evaluate G/|det(DF)| at quadrature points
            ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], mat_map, kind_funs[counter], kind_map, params_map, T_F[0], T_F[1], p_F, NbaseN_F, cx, cy)
            
            ni1, ni2 = ns[a]
            nj1, nj2 = ns[b]
            
            bi1, bi2 = basis[a]
            bj1, bj2 = basis[b]
            
            ker.kernel_mass(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], ni1, ni2, nj1, nj2, wts[0], wts[1], bi1, bi2, bj1, bj2, Nbi1[counter], Nbi2[counter], M[counter], mat_map)
            
            indices = np.indices((Nbi1[counter], Nbi2[counter], 2*p[0] + 1, 2*p[1] + 1))
            
            shift1  = np.arange(Nbi1[counter]) - p[0]
            shift2  = np.arange(Nbi2[counter]) - p[1]
            
            row     = (Nbi2[counter]*indices[0] + indices[1]).flatten()
            
            col1    = (indices[2] + shift1[:, None, None, None])%Nbj1[counter]
            col2    = (indices[3] + shift2[None, :, None, None])%Nbj2[counter]
            
            col     = Nbj2[counter]*col1 + col2
            
            M[counter] = spa.csc_matrix((M[counter].flatten(), (row, col.flatten())), shape=(Nbi1[counter]*Nbi2[counter], Nbj1[counter]*Nbj2[counter]))
            M[counter].eliminate_zeros()
            
            counter += 1
                       
    M = spa.bmat([[M[0], M[1].T], [M[1], M[2]]], format='csc')
                
    return M


# ================ mass matrix in V2 ===========================
def mass_V2(tensor_space_FEM, kind_map, params_map=None, tensor_space_F=None, cx=None, cy=None):
    """
    ----------------------------------------------------------------------------------------------------------
    Assembles the 2D mass matrix [[DD DD]] of the given tensor product B-spline space of bi-degree (p1, p2).
    
    In case of an analytical mapping (kind_map >= 1), all mapping related quantities are called from hylife.geometry.mappings_2d. One must then pass the parameter list params_map.
    
    In case of a discrete mapping (kind_map = 0), one must pass a 2D tensor product B-spline space tensor_space_F together with control points cx and cy which together define the mapping.
    -----------------------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
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
    bc     = tensor_space_FEM.bc      # boundary conditions (periodic vs. clamped)
    NbaseD = tensor_space_FEM.NbaseD  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points
    
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
    
    
    # evaluation of 1/|det(DF)| at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], n_quad[0], n_quad[1]), dtype=float)
    
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], mat_map, 2, kind_map, params_map, T_F[0], T_F[1], p_F, NbaseN_F, cx, cy)
    
    # assembly of global mass matrix
    M = np.zeros((NbaseD[0], NbaseD[1], 2*p[0] + 1, 2*p[1] + 1), dtype=float)
    
    ker.kernel_mass(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], 1, 1, 1, 1, wts[0], wts[1], basisD[0], basisD[1], basisD[0], basisD[1], NbaseD[0], NbaseD[1], M, mat_map)
              
    # conversion to sparse matrix
    indices = np.indices((NbaseD[0], NbaseD[1], 2*p[0] + 1, 2*p[1] + 1))
    
    shift   = [np.arange(NbaseD) - p for NbaseD, p in zip(NbaseD, p)]
    
    row     = (NbaseD[1]*indices[0] + indices[1]).flatten()
    
    col1    = (indices[2] + shift[0][:, None, None, None])%NbaseD[0]
    col2    = (indices[3] + shift[1][None, :, None, None])%NbaseD[1]

    col     = NbaseD[1]*col1 + col2
                
    M       = spa.csc_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseD[0]*NbaseD[1], NbaseD[0]*NbaseD[1]))
    M.eliminate_zeros()
    
    return M