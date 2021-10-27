# coding: utf-8
#
# Copyright 2020 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Modules to compute mass matrices in 3D.
"""


import numpy        as np
import scipy.sparse as spa

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
        mat_w = weight(pts[0].flatten(), pts[1].flatten(), pts[2].flatten())
        mat_w = mat_w.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
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
def get_M1(tensor_space_FEM, domain, weight=None):
    """
    Assembles the 3D mass matrix [[DNN DNN, DNN NDN, DNN NND], [NDN DNN, NDN NDN, NDN NND], [NND DNN, NND NDN, NND NND]] * G^(-1) * |det(DF)| of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    weight : callable
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
            if weight == None:
                mat_w = domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), kind_funs[a][b]) 
            else:
                mat_w = weight[a][b](pts[0].flatten(), pts[1].flatten(), pts[2].flatten())
                
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
def get_M2(tensor_space_FEM, domain, weight=None):
    """
    Assembles the 3D mass matrix [[NDD NDD, NDD DND, NDD DDN], [DND NDD, DND DND, DND DDN], [DDN NDD, DDN DND, DDN DDN]] * G / |det(DF)| of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    weight : callable
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
            if weight == None:
                mat_w = domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), kind_funs[a][b]) 
            else:
                mat_w = weight[a][b](pts[0].flatten(), pts[1].flatten(), pts[2].flatten())
                
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
        mat_w = weight(pts[0].flatten(), pts[1].flatten(), pts[2].flatten())
        mat_w = mat_w.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
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
def get_Mv(tensor_space_FEM, domain, weight=None):
    """
    Assembles the 3D mass matrix [[NNN NNN, NNN NNN, NNN NNN], [NNN NNN, NNN NNN, NNN NNN], [NNN NNN, NNN NNN, NNN NNN]] * G * |det(DF)| of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space for finite element spaces
        
    domain : domain
        domain object defining the geometry
        
    weight : callable
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
    
    bs = 0
    
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
            if weight == None:
                mat_w = domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), kind_funs[a][b]) 
            else:
                mat_w = weight[a][b](pts[0].flatten(), pts[1].flatten(), pts[2].flatten())
                
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