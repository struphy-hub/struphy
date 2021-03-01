# coding: utf-8
#
# Copyright 2020 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Modules to compute mass matrices in 3D.
"""


import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.basics.kernels_3d as ker



# ================ mass matrix in V0 ===========================
def get_M0(tensor_space_FEM, domain):
    """
    Assembles the 3D mass matrix [[NNN NNN]] of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
    """
    
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    bc     = tensor_space_FEM.bc      # boundary conditions (periodic vs. clamped)
    NbaseN = tensor_space_FEM.NbaseN  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1, nq1, Nel2, nq2, Nel3, nq3)
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 1, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
    
    # assembly of global mass matrix
    M = np.zeros((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    
    ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], 0, 0, 0, 0, 0, 0, wts[0], wts[1], wts[2], basisN[0], basisN[1], basisN[2], basisN[0], basisN[1], basisN[2], NbaseN[0], NbaseN[1], NbaseN[2], M, mat_map)
              
    # conversion to sparse matrix
    indices = np.indices((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
    
    shift   = [np.arange(NbaseN) - p for NbaseN, p in zip(NbaseN, p)]
    
    row     = (NbaseN[1]*NbaseN[2]*indices[0] + NbaseN[2]*indices[1] + indices[2]).flatten()
    
    col1    = (indices[3] + shift[0][:, None, None, None, None, None])%NbaseN[0]
    col2    = (indices[4] + shift[1][None, :, None, None, None, None])%NbaseN[1]
    col3    = (indices[5] + shift[2][None, None, :, None, None, None])%NbaseN[2]

    col     = NbaseN[1]*NbaseN[2]*col1 + NbaseN[2]*col2 + col3
                
    M       = spa.csc_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseN[0]*NbaseN[1]*NbaseN[2], NbaseN[0]*NbaseN[1]*NbaseN[2]))
    M.eliminate_zeros()
    
    ## apply boundary conditions
    #M       = M.tolil()
    #
    #if bc[0] == False:
    #    
    #    # first and last spline in rows (Ni) 
    #    if bc_kind[0][0] == 'dirichlet':
    #        M[:NbaseN[1]*NbaseN[2] , :] = 0.
    #    if bc_kind[0][1] == 'dirichlet':
    #        M[-NbaseN[1]*NbaseN[2]:, :] = 0.
    #        
    #    # first and last spline in columns (Nj)
    #    if bc_kind[0][0] == 'dirichlet':
    #        M[:, :NbaseN[1]*NbaseN[2] ] = 0.
    #    if bc_kind[0][1] == 'dirichlet':
    #        M[:, -NbaseN[1]*NbaseN[2]:] = 0.
    #
    #M = M.tocsr()
    
    # apply spline extraction operator and return
    return tensor_space_FEM.E0.dot(M.dot(tensor_space_FEM.E0.T))



# ================ mass matrix in V1 ===========================
def get_M1(tensor_space_FEM, domain):
    """
    Assembles the 3D mass matrix [[DNN DNN, DNN NDN, DNN NND], [NDN DNN, NDN NDN, NDN NND], [NND DNN, NND NDN, NND NND]] of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
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
    
    
    # blocks   11         21         22         31         32          33
    Nbi1 = [NbaseD[0], NbaseN[0], NbaseN[0], NbaseN[0], NbaseN[0], NbaseN[0]]
    Nbi2 = [NbaseN[1], NbaseD[1], NbaseD[1], NbaseN[1], NbaseN[1], NbaseN[1]]
    Nbi3 = [NbaseN[2], NbaseN[2], NbaseN[2], NbaseD[2], NbaseD[2], NbaseD[2]]
    
    Nbj1 = [NbaseD[0], NbaseD[0], NbaseN[0], NbaseD[0], NbaseN[0], NbaseN[0]]
    Nbj2 = [NbaseN[1], NbaseN[1], NbaseD[1], NbaseN[1], NbaseD[1], NbaseN[1]]
    Nbj3 = [NbaseN[2], NbaseN[2], NbaseN[2], NbaseN[2], NbaseN[2], NbaseD[2]]
    
    # basis functions of components of a 1-form
    basis = [[basisD[0], basisN[1], basisN[2]], 
             [basisN[0], basisD[1], basisN[2]], 
             [basisN[0], basisN[1], basisD[2]]]
    
    ns    = [[1, 0, 0], 
             [0, 1, 0], 
             [0, 0, 1]]
    
    # G^(-1)|det(DF)| at quadrature points
    mat_map   = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    kind_funs = [11, 12, 13, 14, 15, 16]
    
    # blocks of global mass matrix
    M = [np.zeros((Nbi1, Nbi2, Nbi3, 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float) for Nbi1, Nbi2, Nbi3 in zip(Nbi1, Nbi2, Nbi3)]
    
    # assembly of blocks 11, 21, 22, 31, 32, 33
    counter = 0
    
    for a in range(3):
        for b in range(a + 1):
            
            # evaluate G^(-1)|det(DF)| at quadrature points
            ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, kind_funs[counter], domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
            
            ni1, ni2, ni3 = ns[a]
            nj1, nj2, nj3 = ns[b]
            
            bi1, bi2, bi3 = basis[a]
            bj1, bj2, bj3 = basis[b]
            
            ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ni1, ni2, ni3, nj1, nj2, nj3, wts[0], wts[1], wts[2], bi1, bi2, bi3, bj1, bj2, bj3, Nbi1[counter], Nbi2[counter], Nbi3[counter], M[counter], mat_map)
            
            
            # convert to sparse matrix
            indices = np.indices((Nbi1[counter], Nbi2[counter], Nbi3[counter], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
            
            shift1  = np.arange(Nbi1[counter]) - p[0]
            shift2  = np.arange(Nbi2[counter]) - p[1]
            shift3  = np.arange(Nbi3[counter]) - p[2]
            
            row     = (Nbi2[counter]*Nbi3[counter]*indices[0] + Nbi3[counter]*indices[1] + indices[2]).flatten()
            
            col1    = (indices[3] + shift1[:, None, None, None, None, None])%Nbj1[counter]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%Nbj2[counter]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%Nbj3[counter]
            
            col     = Nbj2[counter]*Nbj3[counter]*col1 + Nbj3[counter]*col2 + col3
            
            M[counter] = spa.csc_matrix((M[counter].flatten(), (row, col.flatten())), shape=(Nbi1[counter]*Nbi2[counter]*Nbi3[counter], Nbj1[counter]*Nbj2[counter]*Nbj3[counter]))
            M[counter].eliminate_zeros()
            
            counter += 1
                       
    M = spa.bmat([[M[0], M[1].T, M[3].T], [M[1], M[2], M[4].T], [M[3], M[4], M[5]]], format='csc')
                
    # apply spline extraction operator and return
    return tensor_space_FEM.E1.dot(M.dot(tensor_space_FEM.E1.T))



# ================ mass matrix in V2 ===========================
def get_M2(tensor_space_FEM, domain, bc_kind=['free', 'free']):
    """
    Assembles the 3D mass matrix [[NDD NDD, NDD DND, NDD DDN], [DND NDD, DND DND, DND DDN], [DDN NDD, DDN DND, DDN DDN]] of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
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
    
    
    # blocks   11         21         22         31         32          33
    Nbi1   = [NbaseN[0], NbaseD[0], NbaseD[0], NbaseD[0], NbaseD[0], NbaseD[0]]
    Nbi2   = [NbaseD[1], NbaseN[1], NbaseN[1], NbaseD[1], NbaseD[1], NbaseD[1]]
    Nbi3   = [NbaseD[2], NbaseD[2], NbaseD[2], NbaseN[2], NbaseN[2], NbaseN[2]]
    
    Nbj1   = [NbaseN[0], NbaseN[0], NbaseD[0], NbaseN[0], NbaseD[0], NbaseD[0]]
    Nbj2   = [NbaseD[1], NbaseD[1], NbaseN[1], NbaseD[1], NbaseN[1], NbaseD[1]]
    Nbj3   = [NbaseD[2], NbaseD[2], NbaseD[2], NbaseD[2], NbaseD[2], NbaseN[2]]
    
    # basis functions of components of a 2-form
    basis = [[basisN[0], basisD[1], basisD[2]], 
             [basisD[0], basisN[1], basisD[2]], 
             [basisD[0], basisD[1], basisN[2]]]
    
    ns    = [[0, 1, 1], 
             [1, 0, 1], 
             [1, 1, 0]]
    
    # G/|det(DF)| at quadrature points
    mat_map   = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    kind_funs = [21, 22, 23, 24, 25, 26]
    
    # blocks of global mass matrix
    M = [np.zeros((Nbi1, Nbi2, Nbi3, 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float) for Nbi1, Nbi2, Nbi3 in zip(Nbi1, Nbi2, Nbi3)]
    
    # assembly of blocks 11, 21, 22, 31, 32, 33
    counter = 0
    
    for a in range(3):
        for b in range(a + 1):
            
            # evaluate G/|det(DF)| at quadrature points
            ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, kind_funs[counter], domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
            
            ni1, ni2, ni3 = ns[a]
            nj1, nj2, nj3 = ns[b]
            
            bi1, bi2, bi3 = basis[a]
            bj1, bj2, bj3 = basis[b]
            
            ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ni1, ni2, ni3, nj1, nj2, nj3, wts[0], wts[1], wts[2], bi1, bi2, bi3, bj1, bj2, bj3, Nbi1[counter], Nbi2[counter], Nbi3[counter], M[counter], mat_map)
                    
            # convert to sparse matrix
            indices = np.indices((Nbi1[counter], Nbi2[counter], Nbi3[counter], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
            
            shift1  = np.arange(Nbi1[counter]) - p[0]
            shift2  = np.arange(Nbi2[counter]) - p[1]
            shift3  = np.arange(Nbi3[counter]) - p[2]
            
            row     = (Nbi2[counter]*Nbi3[counter]*indices[0] + Nbi3[counter]*indices[1] + indices[2]).flatten()
            
            col1    = (indices[3] + shift1[:, None, None, None, None, None])%Nbj1[counter]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%Nbj2[counter]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%Nbj3[counter]
            
            col     = Nbj2[counter]*Nbj3[counter]*col1 + Nbj3[counter]*col2 + col3
            
            M[counter] = spa.csc_matrix((M[counter].flatten(), (row, col.flatten())), shape=(Nbi1[counter]*Nbi2[counter]*Nbi3[counter], Nbj1[counter]*Nbj2[counter]*Nbj3[counter]))
            M[counter].eliminate_zeros()
            M[counter] = M[counter].tolil()
            
            counter += 1
                       
    
    # apply boundary conditions to first block
    if bc[0] == False:
        
        # first and last spline in rows (Ni) 
        if bc_kind[0] == 'dirichlet':
            M[0][:NbaseD[1]*NbaseD[2] , :] = 0.
        if bc_kind[1] == 'dirichlet':
            M[0][-NbaseD[1]*NbaseD[2]:, :] = 0.
        
        # first and last spline in columns (Nj)
        if bc_kind[0] == 'dirichlet':
            M[0][:, :NbaseD[1]*NbaseD[2] ] = 0.
        if bc_kind[1] == 'dirichlet':
            M[0][:, -NbaseD[1]*NbaseD[2]:] = 0.
        
    M = spa.bmat([[M[0], M[1].T, M[3].T], [M[1], M[2], M[4].T], [M[3], M[4], M[5]]], format='csc')
                
    # apply spline extraction operator and return
    return tensor_space_FEM.E2.dot(M.dot(tensor_space_FEM.E2.T))



# ================ mass matrix in V3 ===========================
def get_M3(tensor_space_FEM, domain):
    """
    Assembles the 3D mass matrix [[DDD DDD]] of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
    """
    
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    bc     = tensor_space_FEM.bc      # boundary conditions (periodic vs. clamped)
    NbaseD = tensor_space_FEM.NbaseD  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points
    
    # evaluation of 1/|det(DF)| at quadrature points
    mat_map = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    
    ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, 2, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
    
    # assembly of global mass matrix
    M = np.zeros((NbaseD[0], NbaseD[1], NbaseD[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    
    ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], 1, 1, 1, 1, 1, 1, wts[0], wts[1], wts[2], basisD[0], basisD[1], basisD[2], basisD[0], basisD[1], basisD[2], NbaseD[0], NbaseD[1], NbaseD[2], M, mat_map)
              
    # conversion to sparse matrix
    indices   = np.indices((NbaseD[0], NbaseD[1], NbaseD[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
    
    shift     = [np.arange(NbaseD) - p for NbaseD, p in zip(NbaseD, p)]
    
    row       = (NbaseD[1]*NbaseD[2]*indices[0] + NbaseD[2]*indices[1] + indices[2]).flatten()
    
    col1      = (indices[3] + shift[0][:, None, None, None, None, None])%NbaseD[0]
    col2      = (indices[4] + shift[1][None, :, None, None, None, None])%NbaseD[1]
    col3      = (indices[5] + shift[2][None, None, :, None, None, None])%NbaseD[2]

    col       = NbaseD[1]*NbaseD[2]*col1 + NbaseD[2]*col2 + col3
                
    M         = spa.csc_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseD[0]*NbaseD[1]*NbaseD[2], NbaseD[0]*NbaseD[1]*NbaseD[2]))
    M.eliminate_zeros()
                
    # apply spline extraction operator and return
    return tensor_space_FEM.E3.dot(M.dot(tensor_space_FEM.E3.T))



# ================ mass matrix of vector field in V0 ===========================
def get_Mv0(tensor_space_FEM, domain, bc_kind=['free', 'free']):
    """
    Assembles the 3D mass matrix [[NNN NNN, NNN NNN, NNN NNN], [NNN NNN, NNN NNN, NNN NNN], [NNN NNN, NNN NNN, NNN NNN]] of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
    """
    
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    bc     = tensor_space_FEM.bc      # boundary conditions (periodic vs. clamped)
    NbaseN = tensor_space_FEM.NbaseN  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points
    wts    = tensor_space_FEM.wts     # global quadrature weights
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)
    
    # G|det(DF)| at quadrature points
    mat_map   = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    kind_funs = [31, 32, 33, 34, 35, 36]
    
    # blocks of global mass matrix
    M = [np.zeros((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float) for i in range(6)]
    
    # assembly of blocks 11, 21, 22, 31, 32, 33
    counter = 0
    
    for a in range(3):
        for b in range(a + 1):
            
            # evaluate G/|det(DF)| at quadrature points
            ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, kind_funs[counter], domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

            ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], 0, 0, 0, 0, 0, 0, wts[0], wts[1], wts[2], basisN[0], basisN[1], basisN[2], basisN[0], basisN[1], basisN[2], NbaseN[0], NbaseN[1], NbaseN[2], M[counter], mat_map)
            
            # convert to sparse matrix
            indices = np.indices((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
    
            shift   = [np.arange(NbaseN) - p for NbaseN, p in zip(NbaseN, p)]

            row     = (NbaseN[1]*NbaseN[2]*indices[0] + NbaseN[2]*indices[1] + indices[2]).flatten()

            col1    = (indices[3] + shift[0][:, None, None, None, None, None])%NbaseN[0]
            col2    = (indices[4] + shift[1][None, :, None, None, None, None])%NbaseN[1]
            col3    = (indices[5] + shift[2][None, None, :, None, None, None])%NbaseN[2]

            col     = NbaseN[1]*NbaseN[2]*col1 + NbaseN[2]*col2 + col3
            
            M[counter] = spa.csc_matrix((M[counter].flatten(), (row, col.flatten())), shape=(NbaseN[0]*NbaseN[1]*NbaseN[2], NbaseN[0]*NbaseN[1]*NbaseN[2]))
            M[counter].eliminate_zeros()
            M[counter] = M[counter].tolil()
            
            counter += 1
    
    # apply boundary conditions to first block
    if bc[0] == False:
        
        # first and last spline in rows (Ni) 
        if bc_kind[0] == 'dirichlet':
            M[0][:NbaseN[1]*NbaseN[2] , :] = 0.
        if bc_kind[1] == 'dirichlet':
            M[0][-NbaseN[1]*NbaseN[2]:, :] = 0.
        
        # first and last spline in columns (Nj)
        if bc_kind[0] == 'dirichlet':
            M[0][:, :NbaseN[1]*NbaseN[2] ] = 0.
        if bc_kind[1] == 'dirichlet':
            M[0][:, -NbaseN[1]*NbaseN[2]:] = 0.
    
    
    M = spa.bmat([[M[0], M[1].T, M[3].T], [M[1], M[2], M[4].T], [M[3], M[4], M[5]]], format='csc')
                
    return M



# ================ mass matrix for vector fields in V2 ===========================
def get_Mv2(tensor_space_FEM, domain):
    """
    Assembles the 3D mass matrix [[NDD NDD, NDD DND, NDD DDN], [DND NDD, DND DND, DND DDN], [DDN NDD, DDN DND, DDN DDN]] of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
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
    
    
    # blocks   11         21         22         31         32          33
    Nbi1   = [NbaseN[0], NbaseD[0], NbaseD[0], NbaseD[0], NbaseD[0], NbaseD[0]]
    Nbi2   = [NbaseD[1], NbaseN[1], NbaseN[1], NbaseD[1], NbaseD[1], NbaseD[1]]
    Nbi3   = [NbaseD[2], NbaseD[2], NbaseD[2], NbaseN[2], NbaseN[2], NbaseN[2]]
    
    Nbj1   = [NbaseN[0], NbaseN[0], NbaseD[0], NbaseN[0], NbaseD[0], NbaseD[0]]
    Nbj2   = [NbaseD[1], NbaseD[1], NbaseN[1], NbaseD[1], NbaseN[1], NbaseD[1]]
    Nbj3   = [NbaseD[2], NbaseD[2], NbaseD[2], NbaseD[2], NbaseD[2], NbaseN[2]]
    
    # basis functions of components of a 2-form
    basis = [[basisN[0], basisD[1], basisD[2]], 
             [basisD[0], basisN[1], basisD[2]], 
             [basisD[0], basisD[1], basisN[2]]]
    
    ns    = [[0, 1, 1], 
             [1, 0, 1], 
             [1, 1, 0]]
    
    # G|det(DF)| at quadrature points
    mat_map   = np.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    kind_funs = [31, 32, 33, 34, 35, 36]
    
    # blocks of global mass matrix
    M = [np.zeros((Nbi1, Nbi2, Nbi3, 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float) for Nbi1, Nbi2, Nbi3 in zip(Nbi1, Nbi2, Nbi3)]
    
    # assembly of blocks 11, 21, 22, 31, 32, 33
    counter = 0
    
    for a in range(3):
        for b in range(a + 1):
            
            # evaluate G|det(DF)| at quadrature points
            ker.kernel_evaluate_quadrature(Nel, n_quad, pts[0], pts[1], pts[2], mat_map, kind_funs[counter], domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
            
            ni1, ni2, ni3 = ns[a]
            nj1, nj2, nj3 = ns[b]
            
            bi1, bi2, bi3 = basis[a]
            bj1, bj2, bj3 = basis[b]
            
            ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ni1, ni2, ni3, nj1, nj2, nj3, wts[0], wts[1], wts[2], bi1, bi2, bi3, bj1, bj2, bj3, Nbi1[counter], Nbi2[counter], Nbi3[counter], M[counter], mat_map)
                    
            # convert to sparse matrix
            indices = np.indices((Nbi1[counter], Nbi2[counter], Nbi3[counter], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
            
            shift1  = np.arange(Nbi1[counter]) - p[0]
            shift2  = np.arange(Nbi2[counter]) - p[1]
            shift3  = np.arange(Nbi3[counter]) - p[2]
            
            row     = (Nbi2[counter]*Nbi3[counter]*indices[0] + Nbi3[counter]*indices[1] + indices[2]).flatten()
            
            col1    = (indices[3] + shift1[:, None, None, None, None, None])%Nbj1[counter]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%Nbj2[counter]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%Nbj3[counter]
            
            col     = Nbj2[counter]*Nbj3[counter]*col1 + Nbj3[counter]*col2 + col3
            
            M[counter] = spa.csc_matrix((M[counter].flatten(), (row, col.flatten())), shape=(Nbi1[counter]*Nbi2[counter]*Nbi3[counter], Nbj1[counter]*Nbj2[counter]*Nbj3[counter]))
            M[counter].eliminate_zeros()
            M[counter] = M[counter].tolil()
            
            counter += 1
    
    M = spa.bmat([[M[0], M[1].T, M[3].T], [M[1], M[2], M[4].T], [M[3], M[4], M[5]]], format='csc')
                
    return M