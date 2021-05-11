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
    NbaseN = tensor_space_FEM.NbaseN  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1, nq1, Nel2, nq2, Nel3, nq3)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), 'det_df'))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    # evaluation of weight function at quadrature points
    if weight != None:
        mat_w = weight(pts[0].flatten(), pts[1].flatten(), pts[2].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    else:
        mat_w = np.ones(det_df.shape, dtype=float)
    
    # assembly of global mass matrix
    M = np.zeros((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    
    ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], 0, 0, 0, 0, 0, 0, wts[0], wts[1], wts[2], basisN[0], basisN[1], basisN[2], basisN[0], basisN[1], basisN[2], NbaseN[0], NbaseN[1], NbaseN[2], M, mat_w*det_df)
              
    # conversion to sparse matrix
    indices = np.indices((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
    
    shift   = [np.arange(NbaseN) - p for NbaseN, p in zip(NbaseN, p)]
    
    row     = (NbaseN[1]*NbaseN[2]*indices[0] + NbaseN[2]*indices[1] + indices[2]).flatten()
    
    col1    = (indices[3] + shift[0][:, None, None, None, None, None])%NbaseN[0]
    col2    = (indices[4] + shift[1][None, :, None, None, None, None])%NbaseN[1]
    col3    = (indices[5] + shift[2][None, None, :, None, None, None])%NbaseN[2]

    col     = NbaseN[1]*NbaseN[2]*col1 + NbaseN[2]*col2 + col3
                
    M       = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseN[0]*NbaseN[1]*NbaseN[2], NbaseN[0]*NbaseN[1]*NbaseN[2]))
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
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1, nq1, Nel2, nq2, Nel3, nq3)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), 'det_df'))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    # evaluation of weight function at quadrature points
    if weight != None:
        mat_w = weight(pts[0].flatten(), pts[1].flatten(), pts[2].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    else:
        mat_w = np.ones(det_df.shape, dtype=float)
    
    # keys for components of inverse metric tensor
    kind_funs = ['g_inv_11', 'g_inv_21', 'g_inv_22', 'g_inv_31', 'g_inv_32', 'g_inv_33']
    
    # blocks of global mass matrix
    M = [np.zeros((Nbi1, Nbi2, Nbi3, 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float) for Nbi1, Nbi2, Nbi3 in zip(Nbi1, Nbi2, Nbi3)]
    
    # assembly of blocks 11, 21, 22, 31, 32, 33
    counter = 0
    
    for a in range(3):
        for b in range(a + 1):
            
            # evaluate inverse metric tensor at quadrature points
            g_inv = domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), kind_funs[counter])
            g_inv = g_inv.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
            
            ni1, ni2, ni3 = ns[a]
            nj1, nj2, nj3 = ns[b]
            
            bi1, bi2, bi3 = basis[a]
            bj1, bj2, bj3 = basis[b]
            
            ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ni1, ni2, ni3, nj1, nj2, nj3, wts[0], wts[1], wts[2], bi1, bi2, bi3, bj1, bj2, bj3, Nbi1[counter], Nbi2[counter], Nbi3[counter], M[counter], mat_w*g_inv*det_df)
            
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
            
            M[counter] = spa.csr_matrix((M[counter].flatten(), (row, col.flatten())), shape=(Nbi1[counter]*Nbi2[counter]*Nbi3[counter], Nbj1[counter]*Nbj2[counter]*Nbj3[counter]))
            M[counter].eliminate_zeros()
            
            counter += 1
                       
    M = spa.bmat([[M[0], M[1].T, M[3].T], [M[1], M[2], M[4].T], [M[3], M[4], M[5]]], format='csr')
                
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
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1, nq1, Nel2, nq2, Nel3, nq3)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), 'det_df'))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    # evaluation of weight function at quadrature points
    if weight != None:
        mat_w = weight(pts[0].flatten(), pts[1].flatten(), pts[2].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    else:
        mat_w = np.ones(det_df.shape, dtype=float)
    
    # keys for components of metric tensor
    kind_funs = ['g_11', 'g_21', 'g_22', 'g_31', 'g_32', 'g_33']
    
    # blocks of global mass matrix
    M = [np.zeros((Nbi1, Nbi2, Nbi3, 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float) for Nbi1, Nbi2, Nbi3 in zip(Nbi1, Nbi2, Nbi3)]
    
    # assembly of blocks 11, 21, 22, 31, 32, 33
    counter = 0
    
    for a in range(3):
        for b in range(a + 1):
            
            # evaluate metric tensor at quadrature points
            g = domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), kind_funs[counter])
            g = g.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
            
            ni1, ni2, ni3 = ns[a]
            nj1, nj2, nj3 = ns[b]
            
            bi1, bi2, bi3 = basis[a]
            bj1, bj2, bj3 = basis[b]
            
            ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ni1, ni2, ni3, nj1, nj2, nj3, wts[0], wts[1], wts[2], bi1, bi2, bi3, bj1, bj2, bj3, Nbi1[counter], Nbi2[counter], Nbi3[counter], M[counter], mat_w*g/det_df)
                    
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
            
            M[counter] = spa.csr_matrix((M[counter].flatten(), (row, col.flatten())), shape=(Nbi1[counter]*Nbi2[counter]*Nbi3[counter], Nbj1[counter]*Nbj2[counter]*Nbj3[counter]))
            M[counter].eliminate_zeros()
            M[counter] = M[counter].tolil()
            
            counter += 1
        
    M = spa.bmat([[M[0], M[1].T, M[3].T], [M[1], M[2], M[4].T], [M[3], M[4], M[5]]], format='csr')
                
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
    NbaseD = tensor_space_FEM.NbaseD  # total number of basis functions (N)
    
    n_quad = tensor_space_FEM.n_quad  # number of quadrature points per element
    pts    = tensor_space_FEM.pts     # global quadrature points in format (element, local quad_point)
    wts    = tensor_space_FEM.wts     # global quadrature weights in format (element, local weight)
    
    basisD = tensor_space_FEM.basisD  # evaluated basis functions at quadrature points
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1, nq1, Nel2, nq2, Nel3, nq3)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), 'det_df'))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    # evaluation of weight function at quadrature points
    if weight != None:
        mat_w = weight(pts[0].flatten(), pts[1].flatten(), pts[2].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    else:
        mat_w = np.ones(det_df.shape, dtype=float)
    
    # assembly of global mass matrix
    M = np.zeros((NbaseD[0], NbaseD[1], NbaseD[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    
    ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], 1, 1, 1, 1, 1, 1, wts[0], wts[1], wts[2], basisD[0], basisD[1], basisD[2], basisD[0], basisD[1], basisD[2], NbaseD[0], NbaseD[1], NbaseD[2], M, mat_w/det_df)
              
    # conversion to sparse matrix
    indices   = np.indices((NbaseD[0], NbaseD[1], NbaseD[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
    
    shift     = [np.arange(NbaseD) - p for NbaseD, p in zip(NbaseD, p)]
    
    row       = (NbaseD[1]*NbaseD[2]*indices[0] + NbaseD[2]*indices[1] + indices[2]).flatten()
    
    col1      = (indices[3] + shift[0][:, None, None, None, None, None])%NbaseD[0]
    col2      = (indices[4] + shift[1][None, :, None, None, None, None])%NbaseD[1]
    col3      = (indices[5] + shift[2][None, None, :, None, None, None])%NbaseD[2]

    col       = NbaseD[1]*NbaseD[2]*col1 + NbaseD[2]*col2 + col3
                
    M         = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseD[0]*NbaseD[1]*NbaseD[2], NbaseD[0]*NbaseD[1]*NbaseD[2]))
    M.eliminate_zeros()
                
    # apply spline extraction operator and return
    return tensor_space_FEM.E3.dot(M.dot(tensor_space_FEM.E3.T)).tocsr()



# ================ mass matrix of vector field in V0 ===========================
def get_Mv0(tensor_space_FEM, domain, weight=None):
    """
    Assembles the 3D mass matrix [[NNN NNN, NNN NNN, NNN NNN], [NNN NNN, NNN NNN, NNN NNN], [NNN NNN, NNN NNN, NNN NNN]] * G * |det(DF)| of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
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
    pts    = tensor_space_FEM.pts     # global quadrature points
    wts    = tensor_space_FEM.wts     # global quadrature weights
    
    basisN = tensor_space_FEM.basisN  # evaluated basis functions at quadrature points (N)
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1, nq1, Nel2, nq2, Nel3, nq3)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), 'det_df'))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    # evaluation of weight function at quadrature points
    if weight != None:
        mat_w = weight(pts[0].flatten(), pts[1].flatten(), pts[2].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    else:
        mat_w = np.ones(det_df.shape, dtype=float)
    
    # keys for components of metric tensor
    kind_funs = ['g_11', 'g_21', 'g_22', 'g_31', 'g_32', 'g_33']
    
    # blocks of global mass matrix
    M = [np.zeros((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float) for i in range(6)]
    
    # assembly of blocks 11, 21, 22, 31, 32, 33
    counter = 0
    
    for a in range(3):
        for b in range(a + 1):
            
            # evaluate metric tensor at quadrature points
            g = domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), kind_funs[counter])
            g = g.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])

            ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], 0, 0, 0, 0, 0, 0, wts[0], wts[1], wts[2], basisN[0], basisN[1], basisN[2], basisN[0], basisN[1], basisN[2], NbaseN[0], NbaseN[1], NbaseN[2], M[counter], mat_w*g*det_df)
            
            # convert to sparse matrix
            indices = np.indices((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
    
            shift   = [np.arange(NbaseN) - p for NbaseN, p in zip(NbaseN, p)]

            row     = (NbaseN[1]*NbaseN[2]*indices[0] + NbaseN[2]*indices[1] + indices[2]).flatten()

            col1    = (indices[3] + shift[0][:, None, None, None, None, None])%NbaseN[0]
            col2    = (indices[4] + shift[1][None, :, None, None, None, None])%NbaseN[1]
            col3    = (indices[5] + shift[2][None, None, :, None, None, None])%NbaseN[2]

            col     = NbaseN[1]*NbaseN[2]*col1 + NbaseN[2]*col2 + col3
            
            M[counter] = spa.csr_matrix((M[counter].flatten(), (row, col.flatten())), shape=(NbaseN[0]*NbaseN[1]*NbaseN[2], NbaseN[0]*NbaseN[1]*NbaseN[2]))
            M[counter].eliminate_zeros()
            
            counter += 1
    
    # apply spline extraction operator and return
    E = spa.bmat([[tensor_space_FEM.E0, None, None], [None, tensor_space_FEM.E0_all, None], [None, None, tensor_space_FEM.E0_all]], format='csr')
    
    M = spa.bmat([[M[0], M[1].T, M[3].T], [M[1], M[2], M[4].T], [M[3], M[4], M[5]]], format='csr')
                
    return E.dot(M.dot(E.T))



# ================ mass matrix for vector fields in V2 ===========================
def get_Mv2(tensor_space_FEM, domain, weight=None):
    """
    Assembles the 3D mass matrix [[NDD NDD, NDD DND, NDD DDN], [DND NDD, DND DND, DND DDN], [DDN NDD, DDN DND, DDN DDN]] * G * |det(DF)| of the given tensor product B-spline spaces of tri-degree (p1, p2, p3) within a computational domain defined by the given object "domain" from hylife.geometry.domain.
    
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
    
    # evaluation of |det(DF)| at quadrature points in format (Nel1, nq1, Nel2, nq2, Nel3, nq3)
    det_df = abs(domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), 'det_df'))
    det_df = det_df.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    
    # evaluation of weight function at quadrature points
    if weight != None:
        mat_w = weight(pts[0].flatten(), pts[1].flatten(), pts[2].flatten()).reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
    else:
        mat_w = np.ones(det_df.shape, dtype=float)
    
    # keys for components of metric tensor
    kind_funs = ['g_11', 'g_21', 'g_22', 'g_31', 'g_32', 'g_33']
    
    # blocks of global mass matrix
    M = [np.zeros((Nbi1, Nbi2, Nbi3, 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float) for Nbi1, Nbi2, Nbi3 in zip(Nbi1, Nbi2, Nbi3)]
    
    # assembly of blocks 11, 21, 22, 31, 32, 33
    counter = 0
    
    for a in range(3):
        for b in range(a + 1):
            
            # evaluate metric tensor at quadrature points
            g = domain.evaluate(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), kind_funs[counter])
            g = g.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
            
            ni1, ni2, ni3 = ns[a]
            nj1, nj2, nj3 = ns[b]
            
            bi1, bi2, bi3 = basis[a]
            bj1, bj2, bj3 = basis[b]
            
            ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ni1, ni2, ni3, nj1, nj2, nj3, wts[0], wts[1], wts[2], bi1, bi2, bi3, bj1, bj2, bj3, Nbi1[counter], Nbi2[counter], Nbi3[counter], M[counter], mat_w*g*det_df)
                    
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
            
            M[counter] = spa.csr_matrix((M[counter].flatten(), (row, col.flatten())), shape=(Nbi1[counter]*Nbi2[counter]*Nbi3[counter], Nbj1[counter]*Nbj2[counter]*Nbj3[counter]))
            M[counter].eliminate_zeros()
            
            counter += 1
    
    M = spa.bmat([[M[0], M[1].T, M[3].T], [M[1], M[2], M[4].T], [M[3], M[4], M[5]]], format='csr')
                
    # apply spline extraction operator and return
    return tensor_space_FEM.E2.dot(M.dot(tensor_space_FEM.E2.T)).tocsr()



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

        ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], n_quad[0], n_quad[1], n_quad[2], ni1, ni2, ni3, nj1, nj2, nj3, wts[0], wts[1], wts[2], bi1, bi2, bi3, bj1, bj2, bj3, Nbi1[a], Nbi2[a], Nbi3[a], M[a], funs_all[a]/det_df)

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