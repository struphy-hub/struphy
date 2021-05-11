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
def get_M0(tensor_space_FEM, domain, weight=None):
    """
    Assembles the 2D mass matrix [[NN NN]] * |det(DF)| of the given tensor product B-spline spaces of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    weight : callable
        optional additional weight function
    """
    
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    NbaseN = tensor_space_FEM.NbaseN  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points
    
    # evaluation of |det(DF)| at eta3 = 0 and quadrature points in format (Nel1, nq1, Nel2, nq2)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'det_df'))[:, :, 0]
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
    
    # evaluation of weight function at quadrature points
    if weight != None:
        mat_w = weight(pts[0].flatten(), pts[1].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
    else:
        mat_w = np.ones(det_df.shape, dtype=float)
    
    # assembly of global mass matrix
    M = np.zeros((NbaseN[0], NbaseN[1], 2*p[0] + 1, 2*p[1] + 1), dtype=float)
    
    ker.kernel_mass(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], 0, 0, 0, 0, wts[0], wts[1], basisN[0], basisN[1], basisN[0], basisN[1], NbaseN[0], NbaseN[1], M, mat_w*det_df)
              
    # conversion to sparse matrix
    indices = np.indices((NbaseN[0], NbaseN[1], 2*p[0] + 1, 2*p[1] + 1))
    
    shift   = [np.arange(NbaseN) - p for NbaseN, p in zip(NbaseN, p)]
    
    row     = (NbaseN[1]*indices[0] + indices[1]).flatten()
    
    col1    = (indices[2] + shift[0][:, None, None, None])%NbaseN[0]
    col2    = (indices[3] + shift[1][None, :, None, None])%NbaseN[1]

    col     = NbaseN[1]*col1 + col2
                
    M       = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseN[0]*NbaseN[1], NbaseN[0]*NbaseN[1]))
    M.eliminate_zeros()
    
    # apply spline extraction operator and return
    return tensor_space_FEM.E0.dot(M.dot(tensor_space_FEM.E0.T))


# ================ mass matrix in V1 ===========================
def get_M1(tensor_space_FEM, domain, weight=None):
    """
    Assembles the 2D mass matrix [[DN DN, DN ND, DN NN], [ND DN, ND ND, ND NN], [NN DN, NN ND, NN NN]] * G^(-1) * |det(DF)| of the given tensor product B-spline spaces of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    weight : callable
        optional additional weight function
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
    
    
    # blocks   11         21         22         31         32          33
    Nbi1 = [NbaseD[0], NbaseN[0], NbaseN[0], NbaseN[0], NbaseN[0], NbaseN[0]]
    Nbi2 = [NbaseN[1], NbaseD[1], NbaseD[1], NbaseN[1], NbaseN[1], NbaseN[1]]
    
    Nbj1 = [NbaseD[0], NbaseD[0], NbaseN[0], NbaseD[0], NbaseN[0], NbaseN[0]]
    Nbj2 = [NbaseN[1], NbaseN[1], NbaseD[1], NbaseN[1], NbaseD[1], NbaseN[1]]
    
    # basis functions of components of a 1-form
    basis = [[basisD[0], basisN[1]], [basisN[0], basisD[1]], [basisN[0], basisN[1]]]
    
    ns    = [[1, 0], [0, 1], [0, 0]]
    
    # evaluation of |det(DF)| at eta3 = 0 and quadrature points in format (Nel1, nq1, Nel2, nq2)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'det_df'))[:, :, 0]
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
    
    # evaluation of weight function at quadrature points
    if weight != None:
        mat_w = weight(pts[0].flatten(), pts[1].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
    else:
        mat_w = np.ones(det_df.shape, dtype=float)
    
    # keys for components of inverse metric tensor
    kind_funs = ['g_inv_11', 'g_inv_21', 'g_inv_22', 'g_inv_31', 'g_inv_32', 'g_inv_33']
    
    # blocks of global mass matrix
    M = [np.zeros((Nbi1, Nbi2, 2*p[0] + 1, 2*p[1] + 1), dtype=float) for Nbi1, Nbi2 in zip(Nbi1, Nbi2)]
    
    # assembly of blocks 11, 21, 22, 31, 32, 33
    counter = 0
    
    for a in range(3):
        for b in range(a + 1):
            
            # evaluate inverse metric tensor at quadrature points
            g_inv = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), kind_funs[counter])[:, :, 0]
            g_inv = g_inv.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
            
            ni1, ni2 = ns[a]
            nj1, nj2 = ns[b]
            
            bi1, bi2 = basis[a]
            bj1, bj2 = basis[b]
            
            ker.kernel_mass(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], ni1, ni2, nj1, nj2, wts[0], wts[1], bi1, bi2, bj1, bj2, Nbi1[counter], Nbi2[counter], M[counter], mat_w*g_inv*det_df)
            
            # convert to sparse matrix
            indices = np.indices((Nbi1[counter], Nbi2[counter], 2*p[0] + 1, 2*p[1] + 1))
            
            shift1  = np.arange(Nbi1[counter]) - p[0]
            shift2  = np.arange(Nbi2[counter]) - p[1]
            
            row     = (Nbi2[counter]*indices[0] + indices[1]).flatten()
            
            col1    = (indices[2] + shift1[:, None, None, None])%Nbj1[counter]
            col2    = (indices[3] + shift2[None, :, None, None])%Nbj2[counter]
            
            col     = Nbj2[counter]*col1 + col2
            
            M[counter] = spa.csr_matrix((M[counter].flatten(), (row, col.flatten())), shape=(Nbi1[counter]*Nbi2[counter], Nbj1[counter]*Nbj2[counter]))
            M[counter].eliminate_zeros()
            
            counter += 1
                       
    M = spa.bmat([[M[0], M[1].T, M[3].T], [M[1], M[2], M[4].T], [M[3], M[4], M[5]]], format='csr')
                
    # apply spline extraction operator and return
    return tensor_space_FEM.E1.dot(M.dot(tensor_space_FEM.E1.T))


# ================ mass matrix in V2 ===========================
def get_M2(tensor_space_FEM, domain, weight=None):
    """
    Assembles the 2D mass matrix [[ND ND, ND DN, ND DD], [DN ND, DN DN, DN DD], [DD ND, DD DN, DD DD]] * G / |det(DF)| of the given tensor product B-spline spaces of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    weight : callable
        optional additional weight function
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
    
    
    # blocks   11         21         22         31         32          33
    Nbi1   = [NbaseN[0], NbaseD[0], NbaseD[0], NbaseD[0], NbaseD[0], NbaseD[0]]
    Nbi2   = [NbaseD[1], NbaseN[1], NbaseN[1], NbaseD[1], NbaseD[1], NbaseD[1]]
    
    Nbj1   = [NbaseN[0], NbaseN[0], NbaseD[0], NbaseN[0], NbaseD[0], NbaseD[0]]
    Nbj2   = [NbaseD[1], NbaseD[1], NbaseN[1], NbaseD[1], NbaseN[1], NbaseD[1]]
    
    # basis functions of components of a 2-form
    basis = [[basisN[0], basisD[1]], [basisD[0], basisN[1]], [basisD[0], basisD[1]]]
    
    ns    = [[0, 1], [1, 0], [1, 1]]
    
    # evaluation of |det(DF)| at eta3 = 0 and quadrature points in format (Nel1, nq1, Nel2, nq2)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'det_df'))[:, :, 0]
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
    
    # evaluation of weight function at quadrature points
    if weight != None:
        mat_w = weight(pts[0].flatten(), pts[1].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
    else:
        mat_w = np.ones(det_df.shape, dtype=float)
    
    # keys for components of metric tensor
    kind_funs = ['g_11', 'g_21', 'g_22', 'g_31', 'g_32', 'g_33']
    
    # blocks of global mass matrix
    M = [np.zeros((Nbi1, Nbi2, 2*p[0] + 1, 2*p[1] + 1), dtype=float) for Nbi1, Nbi2 in zip(Nbi1, Nbi2)]
    
    # assembly of blocks 11, 21, 22, 31, 32, 33
    counter = 0
    
    for a in range(3):
        for b in range(a + 1):
            
            # evaluate metric tensor at quadrature points
            g = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), kind_funs[counter])[:, :, 0]
            g = g.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
            
            ni1, ni2 = ns[a]
            nj1, nj2 = ns[b]
            
            bi1, bi2 = basis[a]
            bj1, bj2 = basis[b]
            
            ker.kernel_mass(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], ni1, ni2, nj1, nj2, wts[0], wts[1], bi1, bi2, bj1, bj2, Nbi1[counter], Nbi2[counter], M[counter], mat_w*g/det_df)
                    
            # convert to sparse matrix
            indices = np.indices((Nbi1[counter], Nbi2[counter], 2*p[0] + 1, 2*p[1] + 1))
            
            shift1  = np.arange(Nbi1[counter]) - p[0]
            shift2  = np.arange(Nbi2[counter]) - p[1]
            
            row     = (Nbi2[counter]*indices[0] + indices[1]).flatten()
            
            col1    = (indices[2] + shift1[:, None, None, None])%Nbj1[counter]
            col2    = (indices[3] + shift2[None, :, None, None])%Nbj2[counter]
            
            col     = Nbj2[counter]*col1 + col2
            
            M[counter] = spa.csr_matrix((M[counter].flatten(), (row, col.flatten())), shape=(Nbi1[counter]*Nbi2[counter], Nbj1[counter]*Nbj2[counter]))
            M[counter].eliminate_zeros()
            
            counter += 1
        
    M = spa.bmat([[M[0], M[1].T, M[3].T], [M[1], M[2], M[4].T], [M[3], M[4], M[5]]], format='csr')
                
    # apply spline extraction operator and return
    return tensor_space_FEM.E2.dot(M.dot(tensor_space_FEM.E2.T))


# ================ mass matrix in V3 ===========================
def get_M3(tensor_space_FEM, domain, weight=None):
    """
    Assembles the 3D mass matrix [[DD DD]] / |det(DF)| of the given tensor product B-spline spaces of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    weight : callable
        optional additional weight function
    """
    
    p      = tensor_space_FEM.p       # spline degrees
    Nel    = tensor_space_FEM.Nel     # number of elements
    NbaseD = tensor_space_FEM.NbaseD  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points
    
    # evaluation of |det(DF)| at eta3 = 0 and quadrature points in format (Nel1, nq1, Nel2, nq2)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'det_df'))[:, :, 0]
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
    
    # evaluation of weight function at quadrature points
    if weight != None:
        mat_w = weight(pts[0].flatten(), pts[1].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
    else:
        mat_w = np.ones(det_df.shape, dtype=float)
    
    # assembly of global mass matrix
    M = np.zeros((NbaseD[0], NbaseD[1], 2*p[0] + 1, 2*p[1] + 1), dtype=float)
    
    ker.kernel_mass(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], 1, 1, 1, 1, wts[0], wts[1], basisD[0], basisD[1], basisD[0], basisD[1], NbaseD[0], NbaseD[1], M, mat_w/det_df)
              
    # conversion to sparse matrix
    indices   = np.indices((NbaseD[0], NbaseD[1], 2*p[0] + 1, 2*p[1] + 1))
    
    shift     = [np.arange(NbaseD) - p for NbaseD, p in zip(NbaseD, p)]
    
    row       = (NbaseD[1]*indices[0] + indices[1]).flatten()
    
    col1      = (indices[2] + shift[0][:, None, None, None])%NbaseD[0]
    col2      = (indices[3] + shift[1][None, :, None, None])%NbaseD[1]

    col       = NbaseD[1]*col1 + col2
                
    M         = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseD[0]*NbaseD[1], NbaseD[0]*NbaseD[1]))
    M.eliminate_zeros()
                
    # apply spline extraction operator and return
    return tensor_space_FEM.E3.dot(M.dot(tensor_space_FEM.E3.T))


# ================ antisymmetric weighted mass matrix in V2 ===========================
def get_M2_a(tensor_space_FEM, domain, funs):
    """
    Assembles the weighted 2D mass matrix [[0, -ND DN fun_3, ND DD fun_2], [DN ND fun_3, 0, -DN DD fun_1], [-DD ND fun_2, DD DN fun_1, 0]] / |det(DF)| of the given tensor product B-spline spaces of bi-degree (p1, p2) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    funs : list of callables or coefficients in V2
        weight functions
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
    
    # evaluation of |det(DF)| at eta3 = 0 and quadrature points in format (Nel1, nq1, Nel2, nq2)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), 'det_df'))[:, :, 0]
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
    
    # evaluation of weight functions at quadrature points
    if isinstance(funs, np.ndarray):
        fun_1 = tensor_space_FEM.evaluate_ND(pts[0].flatten(), pts[1].flatten(), funs, 'V2').reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
        fun_2 = tensor_space_FEM.evaluate_DN(pts[0].flatten(), pts[1].flatten(), funs, 'V2').reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
        fun_3 = tensor_space_FEM.evaluate_DD(pts[0].flatten(), pts[1].flatten(), funs, 'V2').reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
        
    else:
        fun_1 = funs[0](pts[0].flatten(), pts[1].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
        fun_2 = funs[1](pts[0].flatten(), pts[1].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
        fun_3 = funs[2](pts[0].flatten(), pts[1].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
    
    funs_all = [-fun_3, fun_2, -fun_1]
    
    # blocks    12         13         23
    Nbi1    = [NbaseN[0], NbaseN[0], NbaseD[0]]
    Nbi2    = [NbaseD[1], NbaseD[1], NbaseN[1]]
    
    Nbj1    = [NbaseD[0], NbaseD[0], NbaseD[0]]
    Nbj2    = [NbaseN[1], NbaseD[1], NbaseD[1]]
    
    # basis functions of components of a 2-form
    basis_i = [[basisN[0], basisD[1]], [basisN[0], basisD[1]], [basisD[0], basisN[1]]]
    basis_j = [[basisD[0], basisN[1]], [basisD[0], basisD[1]], [basisD[0], basisD[1]]]
    
    ns_i    = [[0, 1], [0, 1], [1, 0]]
    ns_j    = [[1, 0], [1, 1], [1, 1]]
    
    # blocks of global mass matrix
    M = [np.zeros((Nbi1, Nbi2, 2*p[0] + 1, 2*p[1] + 1), dtype=float) for Nbi1, Nbi2 in zip(Nbi1, Nbi2)]
    
    # assembly of blocks 12, 13, 23
    for a in range(3):
            
        ni1, ni2 = ns_i[a]
        nj1, nj2 = ns_j[a]

        bi1, bi2 = basis_i[a]
        bj1, bj2 = basis_j[a]

        ker.kernel_mass(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], ni1, ni2, nj1, nj2, wts[0], wts[1], bi1, bi2, bj1, bj2, Nbi1[a], Nbi2[a], M[a], funs_all[a]/det_df)

        indices = np.indices((Nbi1[a], Nbi2[a], 2*p[0] + 1, 2*p[1] + 1))

        shift1  = np.arange(Nbi1[a]) - p[0]
        shift2  = np.arange(Nbi2[a]) - p[1]

        row     = (Nbi2[a]*indices[0] + indices[1]).flatten()

        col1    = (indices[2] + shift1[:, None, None, None])%Nbj1[a]
        col2    = (indices[3] + shift2[None, :, None, None])%Nbj2[a]

        col     = Nbj2[a]*col1 + col2

        M[a] = spa.csr_matrix((M[a].flatten(), (row, col.flatten())), shape=(Nbi1[a]*Nbi2[a], Nbj1[a]*Nbj2[a]))
        M[a].eliminate_zeros()           
    
    
    M = spa.bmat([[None, M[0], M[1]], [-M[0].T, None, M[2]], [-M[1].T, -M[2].T, None]], format='csr')
    
    # apply spline extraction operator and return
    return tensor_space_FEM.E2.dot(M.dot(tensor_space_FEM.E2.T))