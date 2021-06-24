# coding: utf-8
#
# Copyright 2020 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Modules to compute mass matrices in 3D.
"""


import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.spline_space as spl

import hylife.utilitis_FEEC.basics.kernels_3d as ker

import hylife.linear_algebra.linalg_kron as linkron



# ================ mass matrix in V0 ===========================
def get_M0(tensor_space_FEM, domain, weight=None):
    """
    Assembles the 3D mass matrix [[NNN NNN]] * |det(DF)| of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
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
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1, nq1, Nel2, nq2, Nel3, nq3)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), 'det_df'))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    # evaluation of weight function at quadrature points
    if weight == None:
        mat_w = np.ones(det_df.shape, dtype=float)
    else:
        mat_w = weight(pts[0].flatten(), pts[1].flatten(), pts[2].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    # assembly of global mass matrix
    Ni = tensor_space_FEM.Nbase_0form
    Nj = tensor_space_FEM.Nbase_0form
    
    M  = np.zeros((Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    
    ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], 0, 0, 0, 0, 0, 0, wts[0], wts[1], wts[2], basisN[0], basisN[1], basisN[2], basisN[0], basisN[1], basisN[2], indN[0], indN[1], indN[2], M, mat_w*det_df)
              
    # conversion to sparse matrix
    indices = np.indices((Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
    
    shift   = [np.arange(Ni) - p for Ni, p in zip(Ni, p)]
    
    row     = (Ni[1]*Ni[2]*indices[0] + Ni[2]*indices[1] + indices[2]).flatten()
    
    col1    = (indices[3] + shift[0][:, None, None, None, None, None])%Nj[0]
    col2    = (indices[4] + shift[1][None, :, None, None, None, None])%Nj[1]
    col3    = (indices[5] + shift[2][None, None, :, None, None, None])%Nj[2]

    col     = Nj[1]*Ni[2]*col1 + Ni[2]*col2 + col3
                
    M       = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1]*Ni[2], Nj[0]*Nj[1]*Nj[2]))
    M.eliminate_zeros()
    
    # apply spline extraction operator and return
    return tensor_space_FEM.E0.dot(M.dot(tensor_space_FEM.E0.T)).tocsr()



# ================ mass matrix in V1 ===========================
def get_M1(tensor_space_FEM, domain, weights=None):
    """
    Assembles the 3D mass matrix [[DNN DNN, DNN NDN, DNN NND], [NDN DNN, NDN NDN, NDN NND], [NND DNN, NND NDN, NND NND]] * G^(-1) * |det(DF)| of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    weights : callable
        optional additional weight functions
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
    ind   = [[indD[0], indN[1], indN[2]], [indN[0], indD[1], indN[2]], [indN[0], indN[1], indD[2]]] 
    basis = [[basisD[0], basisN[1], basisN[2]], [basisN[0], basisD[1], basisN[2]], [basisN[0], basisN[1], basisD[2]]]
    ns    = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1, nq1, Nel2, nq2, Nel3, nq3)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), 'det_df'))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    # keys for components of inverse metric tensor
    kind_funs = [['g_inv_11', 'g_inv_12', 'g_inv_13'], ['g_inv_21', 'g_inv_22', 'g_inv_23'], ['g_inv_31', 'g_inv_32', 'g_inv_33']]
    
    # blocks of global mass matrix
    M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    for a in range(3):
        for b in range(3):
            
            Ni = tensor_space_FEM.Nbase_1form[a]
            Nj = tensor_space_FEM.Nbase_1form[b]
            
            M[a][b] = np.zeros((Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
            
            # evaluate metric tensor at quadrature points
            if weights == None:
                mat_w = domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), kind_funs[a][b]) 
            else:
                mat_w = weights[a][b](pts[0].flatten(), pts[1].flatten(), pts[2].flatten())
                
            mat_w = mat_w.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
            
            # assemble block if weight is not zero
            if np.any(mat_w):
                ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ns[a][0], ns[a][1], ns[a][2], ns[b][0], ns[b][1], ns[b][2], wts[0], wts[1], wts[2], basis[a][0], basis[a][1], basis[a][2], basis[b][0], basis[b][1], basis[b][2], ind[a][0], ind[a][1], ind[a][2], M[a][b], mat_w*det_df)
            
            # convert to sparse matrix
            indices = np.indices((Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
            
            shift   = [np.arange(Ni) - p for Ni, p in zip(Ni, p)]
            
            row     = (Ni[1]*Ni[2]*indices[0] + Ni[2]*indices[1] + indices[2]).flatten()
            
            col1    = (indices[3] + shift[0][:, None, None, None, None, None])%Nj[0]
            col2    = (indices[4] + shift[1][None, :, None, None, None, None])%Nj[1]
            col3    = (indices[5] + shift[2][None, None, :, None, None, None])%Nj[2]
            
            col     = Nj[1]*Nj[2]*col1 + Nj[2]*col2 + col3
            
            M[a][b] = spa.csr_matrix((M[a][b].flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1]*Ni[2], Nj[0]*Nj[1]*Nj[2]))
            M[a][b].eliminate_zeros()
        
    M = spa.bmat([[M[0][0], M[0][1], M[0][2]], [M[1][0], M[1][1], M[1][2]], [M[2][0], M[2][1], M[2][2]]], format='csr')
                
    # apply spline extraction operator and return
    return tensor_space_FEM.E1.dot(M.dot(tensor_space_FEM.E1.T)).tocsr()



# ================ mass matrix in V2 ===========================
def get_M2(tensor_space_FEM, domain, weights=None):
    """
    Assembles the 3D mass matrix [[NDD NDD, NDD DND, NDD DDN], [DND NDD, DND DND, DND DDN], [DDN NDD, DDN DND, DDN DDN]] * G / |det(DF)| of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    weights : callable
        optional additional weight functions
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
    ind   = [[indN[0], indD[1], indD[2]], [indD[0], indN[1], indD[2]], [indD[0], indD[1], indN[2]]] 
    basis = [[basisN[0], basisD[1], basisD[2]], [basisD[0], basisN[1], basisD[2]], [basisD[0], basisD[1], basisN[2]]]
    ns    = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1, nq1, Nel2, nq2, Nel3, nq3)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), 'det_df'))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    # keys for components of metric tensor
    kind_funs = [['g_11', 'g_12', 'g_13'], ['g_21', 'g_22', 'g_23'], ['g_31', 'g_32', 'g_33']]
    
    # blocks of global mass matrix
    M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    for a in range(3):
        for b in range(3):
            
            Ni = tensor_space_FEM.Nbase_2form[a]
            Nj = tensor_space_FEM.Nbase_2form[b]
            
            M[a][b] = np.zeros((Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
            
            # evaluate metric tensor at quadrature points
            if weights == None:
                mat_w = domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), kind_funs[a][b]) 
            else:
                mat_w = weights[a][b](pts[0].flatten(), pts[1].flatten(), pts[2].flatten())
                
            mat_w = mat_w.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
            
            # assemble block if weight is not zero
            if np.any(mat_w):
                ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ns[a][0], ns[a][1], ns[a][2], ns[b][0], ns[b][1], ns[b][2], wts[0], wts[1], wts[2], basis[a][0], basis[a][1], basis[a][2], basis[b][0], basis[b][1], basis[b][2], ind[a][0], ind[a][1], ind[a][2], M[a][b], mat_w/det_df)
            
            # convert to sparse matrix
            indices = np.indices((Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
            
            shift   = [np.arange(Ni) - p for Ni, p in zip(Ni, p)]
            
            row     = (Ni[1]*Ni[2]*indices[0] + Ni[2]*indices[1] + indices[2]).flatten()
            
            col1    = (indices[3] + shift[0][:, None, None, None, None, None])%Nj[0]
            col2    = (indices[4] + shift[1][None, :, None, None, None, None])%Nj[1]
            col3    = (indices[5] + shift[2][None, None, :, None, None, None])%Nj[2]
            
            col     = Nj[1]*Nj[2]*col1 + Nj[2]*col2 + col3
            
            M[a][b] = spa.csr_matrix((M[a][b].flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1]*Ni[2], Nj[0]*Nj[1]*Nj[2]))
            M[a][b].eliminate_zeros()
        
    M = spa.bmat([[M[0][0], M[0][1], M[0][2]], [M[1][0], M[1][1], M[1][2]], [M[2][0], M[2][1], M[2][2]]], format='csr')
                
    # apply spline extraction operator and return
    return tensor_space_FEM.E2.dot(M.dot(tensor_space_FEM.E2.T)).tocsr()



# ================ mass matrix in V3 ===========================
def get_M3(tensor_space_FEM, domain, weight=None):
    """
    Assembles the 3D mass matrix [[DDD DDD]] of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
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
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1, nq1, Nel2, nq2, Nel3, nq3)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), 'det_df'))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    # evaluation of weight function at quadrature points
    if weight == None:
        mat_w = np.ones(det_df.shape, dtype=float)
    else:
        mat_w = weight(pts[0].flatten(), pts[1].flatten(), pts[2].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    # assembly of global mass matrix
    Ni = tensor_space_FEM.Nbase_3form
    Nj = tensor_space_FEM.Nbase_3form
    
    M  = np.zeros((Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    
    ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], 1, 1, 1, 1, 1, 1, wts[0], wts[1], wts[2], basisD[0], basisD[1], basisD[2], basisD[0], basisD[1], basisD[2], indD[0], indD[1], indD[2], M, mat_w/det_df)
              
    # conversion to sparse matrix
    indices = np.indices((Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
    
    shift   = [np.arange(Ni) - p for Ni, p in zip(Ni, p)]
    
    row     = (Ni[1]*Ni[2]*indices[0] + Ni[2]*indices[1] + indices[2]).flatten()
    
    col1    = (indices[3] + shift[0][:, None, None, None, None, None])%Nj[0]
    col2    = (indices[4] + shift[1][None, :, None, None, None, None])%Nj[1]
    col3    = (indices[5] + shift[2][None, None, :, None, None, None])%Nj[2]

    col     = Nj[1]*Ni[2]*col1 + Ni[2]*col2 + col3
                
    M       = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1]*Ni[2], Nj[0]*Nj[1]*Nj[2]))
    M.eliminate_zeros()
                
    # apply spline extraction operator and return
    return tensor_space_FEM.E3.dot(M.dot(tensor_space_FEM.E3.T)).tocsr()




# ================ mass matrix for vector fields in V2 ===========================
def get_Mv(tensor_space_FEM, domain, bs, weights=None):
    """
    Assembles the 3D mass matrix [[NNN NNN, NNN NNN, NNN NNN], [NNN NNN, NNN NNN, NNN NNN], [NNN NNN, NNN NNN, NNN NNN]] * G * |det(DF)| of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    weights : callable
        optional additional weight functions
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
    if bs == 0:
        ind   = [[indN[0], indN[1], indN[2]], [indN[0], indN[1], indN[2]], [indN[0], indN[1], indN[2]]] 
        basis = [[basisN[0], basisN[1], basisN[2]], [basisN[0], basisN[1], basisN[2]], [basisN[0], basisN[1], basisN[2]]]
        ns    = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        
    elif bs == 2:
        ind   = [[indN[0], indD[1], indD[2]], [indD[0], indN[1], indD[2]], [indD[0], indD[1], indN[2]]] 
        basis = [[basisN[0], basisD[1], basisD[2]], [basisD[0], basisN[1], basisD[2]], [basisD[0], basisD[1], basisN[2]]]
        ns    = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1, nq1, Nel2, nq2, Nel3, nq3)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), 'det_df'))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    # keys for components of metric tensor
    kind_funs = [['g_11', 'g_12', 'g_13'], ['g_21', 'g_22', 'g_23'], ['g_31', 'g_32', 'g_33']]
    
    # blocks of global mass matrix
    M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    for a in range(3):
        for b in range(3):
            
            if bs == 0:
                Ni = tensor_space_FEM.Nbase_0form
                Nj = tensor_space_FEM.Nbase_0form
            elif bs == 2:
                Ni = tensor_space_FEM.Nbase_2form[a]
                Nj = tensor_space_FEM.Nbase_2form[b]
            
            M[a][b] = np.zeros((Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
            
            # evaluate metric tensor at quadrature points
            if weights == None:
                mat_w = domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), kind_funs[a][b]) 
            else:
                mat_w = weights[a][b](pts[0].flatten(), pts[1].flatten(), pts[2].flatten())
                
            mat_w = mat_w.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
            
            # assemble block if weight is not zero
            if np.any(mat_w):
                ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ns[a][0], ns[a][1], ns[a][2], ns[b][0], ns[b][1], ns[b][2], wts[0], wts[1], wts[2], basis[a][0], basis[a][1], basis[a][2], basis[b][0], basis[b][1], basis[b][2], ind[a][0], ind[a][1], ind[a][2], M[a][b], mat_w*det_df)
                
            # convert to sparse matrix
            indices = np.indices((Ni[0], Ni[1], Ni[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
            
            shift   = [np.arange(Ni) - p for Ni, p in zip(Ni, p)]
            
            row     = (Ni[1]*Ni[2]*indices[0] + Ni[2]*indices[1] + indices[2]).flatten()
            
            col1    = (indices[3] + shift[0][:, None, None, None, None, None])%Nj[0]
            col2    = (indices[4] + shift[1][None, :, None, None, None, None])%Nj[1]
            col3    = (indices[5] + shift[2][None, None, :, None, None, None])%Nj[2]
            
            col     = Nj[1]*Nj[2]*col1 + Nj[2]*col2 + col3
            
            M[a][b] = spa.csr_matrix((M[a][b].flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1]*Ni[2], Nj[0]*Nj[1]*Nj[2]))
            M[a][b].eliminate_zeros()
                    
    # apply spline extraction operator and return
    E = spa.bmat([[tensor_space_FEM.E0, None, None], [None, tensor_space_FEM.E0_all, None], [None, None, tensor_space_FEM.E0_all]], format='csr')
    
    M = spa.bmat([[M[0][0], M[0][1], M[0][2]], [M[1][0], M[1][1], M[1][2]], [M[2][0], M[2][1], M[2][2]]], format='csr')
                
    # apply spline extraction operator and return
    return E.dot(M.dot(E.T))



# ================ antisymmetric weighted mass matrix in V2 ===========================
def get_M2_a(tensor_space_FEM, domain, funs):
    """
    Assembles the weighted §D mass matrix [[0, -NDD DND fun_3, NDD DDN fun_2], [DND NDD fun_3, 0, -DND DDN fun_1], [-DDN NDD fun_2, DDN DND fun_1, 0]] / |det(DF)| of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
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
    indN   = tensor_space_FEM.indN    # global indices of non-vanishing basis functions (N) in format (element, global index) 
    indD   = tensor_space_FEM.indD    # global indices of non-vanishing basis functions (D) in format (element, global index)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points
    wts    = tensor_space_FEM.wts     # global quadrature weights
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points (D)
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1, nq1, Nel2, nq2, Nel3, nq3)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), 'det_df'))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    # evaluation of weight functions at quadrature points
    if isinstance(funs, np.ndarray):
        fun_1 = tensor_space_FEM.evaluate_NDD(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), funs).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
        fun_2 = tensor_space_FEM.evaluate_DND(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), funs).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
        fun_3 = tensor_space_FEM.evaluate_DDN(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), funs).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
        
    else:
        fun_1 = funs[0](pts[0].flatten(), pts[1].flatten(), pts[2].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
        fun_2 = funs[1](pts[0].flatten(), pts[1].flatten(), pts[2].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
        fun_3 = funs[2](pts[0].flatten(), pts[1].flatten(), pts[2].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    funs_all = [-fun_3, fun_2, -fun_1]
    
    # blocks    12         13         23
    Nbi1    = [NbaseN[0], NbaseN[0], NbaseD[0]]
    Nbi2    = [NbaseD[1], NbaseD[1], NbaseN[1]]
    Nbi3    = [NbaseD[2], NbaseD[2], NbaseD[2]]
    
    Nbj1    = [NbaseD[0], NbaseD[0], NbaseD[0]]
    Nbj2    = [NbaseN[1], NbaseD[1], NbaseD[1]]
    Nbj3    = [NbaseD[2], NbaseN[2], NbaseN[2]]
    
    ind1    = [indN[0], indN[0], indD[0]]
    ind2    = [indD[1], indD[1], indN[1]]
    ind3    = [indD[2], indD[2], indD[2]]
    
    # basis functions of components of a 2-form
    basis_i = [[basisN[0], basisD[1], basisD[2]], [basisN[0], basisD[1], basisD[2]], [basisD[0], basisN[1], basisD[2]]]
    basis_j = [[basisD[0], basisN[1], basisD[2]], [basisD[0], basisD[1], basisN[2]], [basisD[0], basisD[1], basisN[2]]]
    
    ns_i    = [[0, 1, 1], [0, 1, 1], [1, 0, 1]]
    ns_j    = [[1, 0, 1], [1, 1, 0], [1, 1, 0]]
    
    # blocks of global mass matrix
    M = [np.zeros((Nbi1, Nbi2, Nbi3, 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float) for Nbi1, Nbi2, Nbi3 in zip(Nbi1, Nbi2, Nbi3)]
    
    # assembly of blocks 12, 13, 23
    counter = 0
    
    for a in range(3):
            
        ni1, ni2, ni3 = ns_i[a]
        nj1, nj2, nj3 = ns_j[a]

        bi1, bi2, bi3 = basis_i[a]
        bj1, bj2, bj3 = basis_j[a]

        ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ni1, ni2, ni3, nj1, nj2, nj3, wts[0], wts[1], wts[2], bi1, bi2, bi3, bj1, bj2, bj3, ind1[a], ind2[a], ind3[a], M[a], funs_all[a]/det_df)

        # convert to sparse matrix
        indices = np.indices((Nbi1[a], Nbi2[a], Nbi3[a], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))

        shift1  = np.arange(Nbi1[a]) - p[0]
        shift2  = np.arange(Nbi2[a]) - p[1]
        shift3  = np.arange(Nbi3[a]) - p[2]

        row     = (Nbi2[a]*Nbi3[a]*indices[0] + Nbi3[a]*indices[1] + indices[2]).flatten()

        col1    = (indices[3] + shift1[:, None, None, None, None, None])%Nbj1[a]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%Nbj2[a]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%Nbj3[a]

        col     = Nbj2[a]*Nbj3[a]*col1 + Nbj3[a]*col2 + col3

        M[a] = spa.csr_matrix((M[a].flatten(), (row, col.flatten())), shape=(Nbi1[a]*Nbi2[a]*Nbi3[a], Nbj1[a]*Nbj2[a]*Nbj3[a]))
        M[a].eliminate_zeros()
        
        
    M = spa.bmat([[None, M[0], M[1]], [-M[0].T, None, M[2]], [-M[1].T, -M[2].T, None]], format='csr')
                
    # apply spline extraction operator and return
    return tensor_space_FEM.E2.dot(M.dot(tensor_space_FEM.E2.T)).tocsr()



# ================ inverse mass matrix in V0 ===========================
def get_M0_PRE(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 1d spaces for pre-conditioning with fft:
    Nel_pre      = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN] 
    spl_kind_pre = [True, True, True]
    spaces_pre   = [spl.spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)]
    
    # tensor product mass matrices for pre-conditioning
    spaces_pre[0].set_extraction_operators()
    spaces_pre[1].set_extraction_operators()
    spaces_pre[2].set_extraction_operators()
    
    spaces_pre[0].assemble_M0(lambda eta : domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta : domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta : domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    c_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M0.shape, matvec=lambda x: (linkron.kron_fftsolve_3d(c_pre, x.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2]))).flatten())



# ================ inverse mass matrix in V1 ===========================
def get_M1_PRE(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 1d spaces for pre-conditioning with fft:
    Nel_pre      = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN] 
    spl_kind_pre = [True, True, True]
    spaces_pre   = [spl.spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)]
    
    # tensor product mass matrices for pre-conditioning of the three diagonal blocks
    spaces_pre[0].set_extraction_operators()
    spaces_pre[1].set_extraction_operators()
    spaces_pre[2].set_extraction_operators()
    
    spaces_pre[0].assemble_M0(lambda eta : domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta : domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta : domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    spaces_pre[0].assemble_M1(lambda eta : 1/domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M1(lambda eta : 1/domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M1(lambda eta : 1/domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    c11_pre = [spaces_pre[0].M1.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]
    c22_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M1.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]
    c33_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M1.toarray()[:, 0]]
    
    def solve(x):
        
        x1, x2, x3 = np.split(x, 3)
        
        x1 = x1.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x2 = x2.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x3 = x3.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        
        r1 = linkron.kron_fftsolve_3d(c11_pre, x1).flatten()
        r2 = linkron.kron_fftsolve_3d(c22_pre, x2).flatten()
        r3 = linkron.kron_fftsolve_3d(c33_pre, x3).flatten()
        
        return np.concatenate((r1, r2, r3))
            
    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M1.shape, matvec=solve)


# ================ inverse mass matrix in V2 ===========================
def get_M2_PRE(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 1d spaces for pre-conditioning with fft:
    Nel_pre      = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN] 
    spl_kind_pre = [True, True, True]
    spaces_pre   = [spl.spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)]
    
    # tensor product mass matrices for pre-conditioning of the three diagonal blocks
    spaces_pre[0].set_extraction_operators()
    spaces_pre[1].set_extraction_operators()
    spaces_pre[2].set_extraction_operators()
    
    spaces_pre[0].assemble_M0(lambda eta : domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta : domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta : domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    spaces_pre[0].assemble_M1(lambda eta : 1/domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M1(lambda eta : 1/domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M1(lambda eta : 1/domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    c11_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M1.toarray()[:, 0], spaces_pre[2].M1.toarray()[:, 0]]
    c22_pre = [spaces_pre[0].M1.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M1.toarray()[:, 0]]
    c33_pre = [spaces_pre[0].M1.toarray()[:, 0], spaces_pre[1].M1.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]
    
    def solve(x):
        
        x1, x2, x3 = np.split(x, 3)
        
        x1 = x1.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x2 = x2.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x3 = x3.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        
        r1 = linkron.kron_fftsolve_3d(c11_pre, x1).flatten()
        r2 = linkron.kron_fftsolve_3d(c22_pre, x2).flatten()
        r3 = linkron.kron_fftsolve_3d(c33_pre, x3).flatten()
        
        return np.concatenate((r1, r2, r3))
            
    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M2.shape, matvec=solve)


# ================ inverse mass matrix in V3 ===========================
def get_M3_PRE(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 1d spaces for pre-conditioning with fft:
    Nel_pre      = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN] 
    spl_kind_pre = [True, True, True]
    spaces_pre   = [spl.spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)]
    
    # tensor product mass matrices for pre-conditioning
    spaces_pre[0].set_extraction_operators()
    spaces_pre[1].set_extraction_operators()
    spaces_pre[2].set_extraction_operators()
    
    spaces_pre[0].assemble_M1(lambda eta : 1/domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M1(lambda eta : 1/domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M1(lambda eta : 1/domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    c_pre = [spaces_pre[0].M1.toarray()[:, 0], spaces_pre[1].M1.toarray()[:, 0], spaces_pre[2].M1.toarray()[:, 0]]

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M3.shape, matvec=lambda x: (linkron.kron_fftsolve_3d(c_pre, x.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2]))).flatten())



# ================ inverse mass matrix in V0^3 ===========================
def get_Mv_PRE(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 1d spaces for pre-conditioning with fft:
    Nel_pre      = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN] 
    spl_kind_pre = [True, True, True]
    spaces_pre   = [spl.spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)]
    
    # tensor product mass matrices for pre-conditioning of the three diagonal blocks
    spaces_pre[0].set_extraction_operators()
    spaces_pre[1].set_extraction_operators()
    spaces_pre[2].set_extraction_operators()
    
    spaces_pre[0].assemble_M0(lambda eta : domain.params_map[0]**3*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta : domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta : domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    c11_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]
    
    spaces_pre[0].assemble_M0(lambda eta : domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta : domain.params_map[1]**3*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta : domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    c22_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]
    
    spaces_pre[0].assemble_M0(lambda eta : domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta : domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta : domain.params_map[2]**3*np.ones(eta.shape, dtype=float))
    
    c33_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]
    
    def solve(x):
        
        x1, x2, x3 = np.split(x, 3)
        
        x1 = x1.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x2 = x2.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x3 = x3.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        
        r1 = linkron.kron_fftsolve_3d(c11_pre, x1).flatten()
        r2 = linkron.kron_fftsolve_3d(c22_pre, x2).flatten()
        r3 = linkron.kron_fftsolve_3d(c33_pre, x3).flatten()
        
        return np.concatenate((r1, r2, r3))
            
    return spa.linalg.LinearOperator(shape=tensor_space_FEM.Mv.shape, matvec=solve)



# ========== inverse mass matrix in V0 with decomposition poloidal x toroidal ==============
def get_M0_PRE_3(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 2D tensor product space in poloidal plane
    space_pol = spl.tensor_spline_space([tensor_space_FEM.spaces[0], tensor_space_FEM.spaces[1]])
    
    # 1D space in toroidal direction
    space_tor = tensor_space_FEM.spaces[2]
    
    # set extraction operators
    space_pol.set_extraction_operators(tensor_space_FEM.bc, tensor_space_FEM.polar_splines)
    space_tor.set_extraction_operators()
    
    # mass matrices
    space_pol.assemble_M0_2D(domain)
    space_tor.assemble_M0()
    
    # LU decomposition of poloidal mass matrix
    M0_pol_LU = spa.linalg.splu(space_pol.M0.tocsc())
    
    # vector defining the circulant mass matrix in toroidal direction
    tor_vec = space_tor.M0.toarray()[:, 0]

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M0.shape, matvec=lambda x: linkron.kron_fftsolve_2d(M0_pol_LU, tor_vec, x.reshape(tensor_space_FEM.E0_pol.shape[0], tensor_space_FEM.NbaseN[2])).flatten())


# ========== inverse mass matrix in V1 with decomposition poloidal x toroidal ==============
def get_M1_PRE_3(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 2D tensor product space in poloidal plane
    space_pol = spl.tensor_spline_space([tensor_space_FEM.spaces[0], tensor_space_FEM.spaces[1]])
    
    # 1D space in toroidal direction
    space_tor = tensor_space_FEM.spaces[2]
    
    # set extraction operators
    space_pol.set_extraction_operators(tensor_space_FEM.bc, tensor_space_FEM.polar_splines)
    space_tor.set_extraction_operators()
    
    # mass matrices
    space_pol.assemble_M1_2D_blocks(domain)
    space_tor.assemble_M0()
    space_tor.assemble_M1()
    
    # LU decomposition of poloidal mass matrix
    M1_pol_12_LU = spa.linalg.splu(space_pol.M1_12.tocsc())
    M1_pol_33_LU = spa.linalg.splu(space_pol.M1_33.tocsc())
    
    # vectors defining the circulant mass matrices in toroidal direction
    tor_vec0 = space_tor.M0.toarray()[:, 0]
    tor_vec1 = space_tor.M1.toarray()[:, 0]
    
    def solve(x):
        
        x1 = x[:tensor_space_FEM.E1_pol.shape[0]*tensor_space_FEM.NbaseN[2] ].reshape(tensor_space_FEM.E1_pol.shape[0], tensor_space_FEM.NbaseN[2])
        x3 = x[ tensor_space_FEM.E1_pol.shape[0]*tensor_space_FEM.NbaseN[2]:].reshape(tensor_space_FEM.E0_pol.shape[0], tensor_space_FEM.NbaseD[2])
        
        r1 = linkron.kron_fftsolve_2d(M1_pol_12_LU, tor_vec0, x1).flatten()
        r3 = linkron.kron_fftsolve_2d(M1_pol_33_LU, tor_vec1, x3).flatten()
        
        return np.concatenate((r1, r3))

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M1.shape, matvec=solve)


# ========== inverse mass matrix in V2 with decomposition poloidal x toroidal ==============
def get_M2_PRE_3(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 2D tensor product space in poloidal plane
    space_pol = spl.tensor_spline_space([tensor_space_FEM.spaces[0], tensor_space_FEM.spaces[1]])
    
    # 1D space in toroidal direction
    space_tor = tensor_space_FEM.spaces[2]
    
    # set extraction operators
    space_pol.set_extraction_operators(tensor_space_FEM.bc, tensor_space_FEM.polar_splines)
    space_tor.set_extraction_operators()
    
    # mass matrices
    space_pol.assemble_M2_2D_blocks(domain)
    space_tor.assemble_M0()
    space_tor.assemble_M1()
    
    # LU decomposition of poloidal mass matrix
    M2_pol_12_LU = spa.linalg.splu(space_pol.M2_12.tocsc())
    M2_pol_33_LU = spa.linalg.splu(space_pol.M2_33.tocsc())
    
    # vectors defining the circulant mass matrices in toroidal direction
    tor_vec0 = space_tor.M0.toarray()[:, 0]
    tor_vec1 = space_tor.M1.toarray()[:, 0]
    
    def solve(x):
        
        x1 = x[:tensor_space_FEM.E2_pol.shape[0]*tensor_space_FEM.NbaseD[2] ].reshape(tensor_space_FEM.E2_pol.shape[0], tensor_space_FEM.NbaseD[2])
        x3 = x[ tensor_space_FEM.E2_pol.shape[0]*tensor_space_FEM.NbaseD[2]:].reshape(tensor_space_FEM.E3_pol.shape[0], tensor_space_FEM.NbaseN[2])
        
        r1 = linkron.kron_fftsolve_2d(M2_pol_12_LU, tor_vec1, x1).flatten()
        r3 = linkron.kron_fftsolve_2d(M2_pol_33_LU, tor_vec0, x3).flatten()
        
        return np.concatenate((r1, r3))

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M2.shape, matvec=solve)


# ========== inverse mass matrix in V3 with decomposition poloidal x toroidal ==============
def get_M3_PRE_3(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 2D tensor product space in poloidal plane
    space_pol = spl.tensor_spline_space([tensor_space_FEM.spaces[0], tensor_space_FEM.spaces[1]])
    
    # 1D space in toroidal direction
    space_tor = tensor_space_FEM.spaces[2]
    
    # set extraction operators
    space_pol.set_extraction_operators(tensor_space_FEM.bc, tensor_space_FEM.polar_splines)
    space_tor.set_extraction_operators()
    
    # mass matrices
    space_pol.assemble_M3_2D(domain)
    space_tor.assemble_M1()
    
    # LU decomposition of poloidal mass matrix
    M3_pol_LU = spa.linalg.splu(space_pol.M3.tocsc())
    
    # vector defining the circulant mass matrix in toroidal direction
    tor_vec = space_tor.M1.toarray()[:, 0]

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M3.shape, matvec=lambda x: linkron.kron_fftsolve_2d(M3_pol_LU, tor_vec, x.reshape(tensor_space_FEM.E3_pol.shape[0], tensor_space_FEM.NbaseD[2])).flatten())



# ========== inverse mass matrix in V0^3 with decomposition poloidal x toroidal ==============
def get_Mv_PRE_3(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 2D tensor product space in poloidal plane
    space_pol = spl.tensor_spline_space([tensor_space_FEM.spaces[0], tensor_space_FEM.spaces[1]])
    
    # 1D space in toroidal direction
    space_tor = tensor_space_FEM.spaces[2]
    
    # set extraction operators
    space_pol.set_extraction_operators(tensor_space_FEM.bc, tensor_space_FEM.polar_splines)
    space_tor.set_extraction_operators()
    
    # mass matrices
    space_pol.assemble_Mv_2D_blocks(domain)
    space_tor.assemble_M0()
    
    # LU decomposition of poloidal mass matrix
    Mv_pol_12_LU = spa.linalg.splu(space_pol.Mv_12.tocsc())
    Mv_pol_33_LU = spa.linalg.splu(space_pol.Mv_33.tocsc())
    
    # vectors defining the circulant mass matrices in toroidal direction
    tor_vec0 = space_tor.M0.toarray()[:, 0]
    
    def solve(x):
        
        x1 = x[:(tensor_space_FEM.E0_pol.shape[0] + tensor_space_FEM.E0_pol_all.shape[0])*tensor_space_FEM.NbaseN[2] ].reshape(tensor_space_FEM.E0_pol.shape[0] + tensor_space_FEM.E0_pol_all.shape[0], tensor_space_FEM.NbaseN[2])
        x3 = x[ (tensor_space_FEM.E0_pol.shape[0] + tensor_space_FEM.E0_pol_all.shape[0])*tensor_space_FEM.NbaseN[2]:].reshape(tensor_space_FEM.E0_pol_all.shape[0], tensor_space_FEM.NbaseN[2])
        
        r1 = linkron.kron_fftsolve_2d(Mv_pol_12_LU, tor_vec0, x1).flatten()
        r3 = linkron.kron_fftsolve_2d(Mv_pol_33_LU, tor_vec0, x3).flatten()
        
        return np.concatenate((r1, r3))

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.Mv.shape, matvec=solve)