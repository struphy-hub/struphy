# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to compute mass matrices in 2D.
"""
import time

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
    indN   = tensor_space_FEM.indN    # global indices of local non-vanishing basis functions in format (element, global index)
    
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
    Ni = tensor_space_FEM.Nbase_0form
    Nj = tensor_space_FEM.Nbase_0form
    
    M  = np.zeros((Ni[0], Ni[1], 2*p[0] + 1, 2*p[1] + 1), dtype=float)
    
    ker.kernel_mass(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], 0, 0, 0, 0, wts[0], wts[1], basisN[0], basisN[1], basisN[0], basisN[1], indN[0], indN[1], M, mat_w*det_df)
              
    # conversion to sparse matrix
    indices = np.indices((Ni[0], Ni[1], 2*p[0] + 1, 2*p[1] + 1))
    
    shift   = [np.arange(Ni) - p for Ni, p in zip(Ni, p)]
    
    row     = (Ni[1]*indices[0] + indices[1]).flatten()
    
    col1    = (indices[2] + shift[0][:, None, None, None])%Nj[0]
    col2    = (indices[3] + shift[1][None, :, None, None])%Nj[1]

    col     = Nj[1]*col1 + col2
                
    M       = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1], Nj[0]*Nj[1]))
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
    indN   = tensor_space_FEM.indN    # global indices of non-vanishing basis functions (N) in format (element, global index) 
    indD   = tensor_space_FEM.indD    # global indices of non-vanishing basis functions (D) in format (element, global index)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points
    wts    = tensor_space_FEM.wts     # global quadrature weights
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points (D)
    
    # indices and basis functions of components of a 1-form
    ind   = [[indD[0], indN[1]], [indN[0], indD[1]], [indN[0], indN[1]]] 
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
    kind_funs = [['g_inv_11'], ['g_inv_21', 'g_inv_22'], ['g_inv_31', 'g_inv_32', 'g_inv_33']]
    
    # blocks of global mass matrix
    M = [[0], [0, 0], [0, 0, 0]]
    
    # assembly of blocks 11, 21, 22, 31, 32, 33
    for a in range(3):
        for b in range(a + 1):
            
            # evaluate inverse metric tensor at quadrature points
            g_inv = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), kind_funs[a][b])[:, :, 0]
            g_inv = g_inv.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
            
            Ni = tensor_space_FEM.Nbase_1form[a]
            Nj = tensor_space_FEM.Nbase_1form[b]
            
            M[a][b] = np.zeros((Ni[0], Ni[1], 2*p[0] + 1, 2*p[1] + 1), dtype=float)
            
            ker.kernel_mass(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], ns[a][0], ns[a][1], ns[b][0], ns[b][1], wts[0], wts[1], basis[a][0], basis[a][1], basis[b][0], basis[b][1], ind[a][0], ind[a][1], M[a][b], mat_w*g_inv*det_df)
            
            # convert to sparse matrix
            indices = np.indices((Ni[0], Ni[1], 2*p[0] + 1, 2*p[1] + 1))
            
            shift   = [np.arange(Ni) - p for Ni, p in zip(Ni, p)]
            
            row     = (Ni[1]*indices[0] + indices[1]).flatten()
            
            col1    = (indices[2] + shift[0][:, None, None, None])%Nj[0]
            col2    = (indices[3] + shift[1][None, :, None, None])%Nj[1]
            
            col     = Nj[1]*col1 + col2
            
            M[a][b] = spa.csr_matrix((M[a][b].flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1], Nj[0]*Nj[1]))
            M[a][b].eliminate_zeros()
                       
    M = spa.bmat([[M[0][0], M[1][0].T, M[2][0].T], [M[1][0], M[1][1], M[2][1].T], [M[2][0], M[2][1], M[2][2]]], format='csr')
                
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
    indN   = tensor_space_FEM.indN    # global indices of non-vanishing basis functions (N) in format (element, global index) 
    indD   = tensor_space_FEM.indD    # global indices of non-vanishing basis functions (D) in format (element, global index)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points
    wts    = tensor_space_FEM.wts     # global quadrature weights
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points (D)
    
    # indices and basis functions of components of a 2-form
    ind   = [[indN[0], indD[1]], [indD[0], indN[1]], [indD[0], indD[1]]] 
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
    kind_funs = [['g_11'], ['g_21', 'g_22'], ['g_31', 'g_32', 'g_33']]
    
    # blocks of global mass matrix
    M = [[0], [0, 0], [0, 0, 0]]
    
    # assembly of blocks 11, 21, 22, 31, 32, 33
    for a in range(3):
        for b in range(a + 1):
            
            # evaluate metric tensor at quadrature points
            g = domain.evaluate(pts[0].flatten(), pts[1].flatten(), np.array([0.]), kind_funs[a][b])[:, :, 0]
            g = g.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1])
            
            Ni = tensor_space_FEM.Nbase_2form[a]
            Nj = tensor_space_FEM.Nbase_2form[b]
            
            M[a][b] = np.zeros((Ni[0], Ni[1], 2*p[0] + 1, 2*p[1] + 1), dtype=float)
            
            ker.kernel_mass(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], ns[a][0], ns[a][1], ns[b][0], ns[b][1], wts[0], wts[1], basis[a][0], basis[a][1], basis[b][0], basis[b][1], ind[a][0], ind[a][1], M[a][b], mat_w*g/det_df)
                    
            # convert to sparse matrix
            indices = np.indices((Ni[0], Ni[1], 2*p[0] + 1, 2*p[1] + 1))
            
            shift   = [np.arange(Ni) - p for Ni, p in zip(Ni, p)]
            
            row     = (Ni[1]*indices[0] + indices[1]).flatten()
            
            col1    = (indices[2] + shift[0][:, None, None, None])%Nj[0]
            col2    = (indices[3] + shift[1][None, :, None, None])%Nj[1]
            
            col     = Nj[1]*col1 + col2
            
            M[a][b] = spa.csr_matrix((M[a][b].flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1], Nj[0]*Nj[1]))
            M[a][b].eliminate_zeros()
        
    M = spa.bmat([[M[0][0], M[1][0].T, M[2][0].T], [M[1][0], M[1][1], M[2][1].T], [M[2][0], M[2][1], M[2][2]]], format='csr')
                
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
    indD   = tensor_space_FEM.indD    # global indices of local non-vanishing basis functions in format (element, global index)
    
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
    Ni = tensor_space_FEM.Nbase_3form
    Nj = tensor_space_FEM.Nbase_3form
    
    M  = np.zeros((Ni[0], Ni[1], 2*p[0] + 1, 2*p[1] + 1), dtype=float)
    
    ker.kernel_mass(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], 1, 1, 1, 1, wts[0], wts[1], basisD[0], basisD[1], basisD[0], basisD[1], indD[0], indD[1], M, mat_w/det_df)
              
    # conversion to sparse matrix
    indices = np.indices((Ni[0], Ni[1], 2*p[0] + 1, 2*p[1] + 1))
    
    shift   = [np.arange(Ni) - p for Ni, p in zip(Ni, p)]
    
    row     = (Ni[1]*indices[0] + indices[1]).flatten()
    
    col1    = (indices[2] + shift[0][:, None, None, None])%Nj[0]
    col2    = (indices[3] + shift[1][None, :, None, None])%Nj[1]

    col     = Nj[1]*col1 + col2
                
    M       = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1], Nj[0]*Nj[1]))
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
    indN   = tensor_space_FEM.indN    # global indices of non-vanishing basis functions (N) in format (element, global index) 
    indD   = tensor_space_FEM.indD    # global indices of non-vanishing basis functions (D) in format (element, global index)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points
    wts    = tensor_space_FEM.wts     # global quadrature weights
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points (D)
    
    # indices and basis functions of components of a 2-form
    ind   = [[indN[0], indD[1]], [indD[0], indN[1]], [indD[0], indD[1]]] 
    basis = [[basisN[0], basisD[1]], [basisD[0], basisN[1]], [basisD[0], basisD[1]]]
    ns    = [[0, 1], [1, 0], [1, 1]]
    
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
    
    funs_all = [fun_3, -fun_2, fun_1]
    
    # blocks of global mass matrix
    M = [[0], [0, 0], [0, 0, 0]]
    
    # assembly of blocks 21, 31, 32
    blocks = [[1, 0], [2, 0], [2, 1]]
    
    for c in range(3):
        
        a, b = blocks[c]
        
        Ni = tensor_space_FEM.Nbase_2form[a]
        Nj = tensor_space_FEM.Nbase_2form[b]

        M[a][b] = np.zeros((Ni[0], Ni[1], 2*p[0] + 1, 2*p[1] + 1), dtype=float)

        ker.kernel_mass(Nel[0], Nel[1], p[0], p[1], n_quad[0], n_quad[1], ns[a][0], ns[a][1], ns[b][0], ns[b][1], wts[0], wts[1], basis[a][0], basis[a][1], basis[b][0], basis[b][1], ind[a][0], ind[a][1], M[a][b], funs_all[c]/det_df)
            
        
        # convert to sparse matrix
        indices = np.indices((Ni[0], Ni[1], 2*p[0] + 1, 2*p[1] + 1))

        shift   = [np.arange(Ni) - p for Ni, p in zip(Ni, p)]

        row     = (Ni[1]*indices[0] + indices[1]).flatten()

        col1    = (indices[2] + shift[0][:, None, None, None])%Nj[0]
        col2    = (indices[3] + shift[1][None, :, None, None])%Nj[1]

        col     = Nj[1]*col1 + col2

        M[a][b] = spa.csr_matrix((M[a][b].flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1], Nj[0]*Nj[1]))
        M[a][b].eliminate_zeros()         
    
    M = spa.bmat([[None, -M[1][0].T, -M[2][0].T], [M[1][0], None, -M[2][1].T], [M[2][0], M[2][1], None]], format='csr')
    
    # apply spline extraction operator and return
    return tensor_space_FEM.E2.dot(M.dot(tensor_space_FEM.E2.T))