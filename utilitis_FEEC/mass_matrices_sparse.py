import numpy as np
import scipy.sparse as sparse
import psydac.core.interface as inter
from pyccel import epyccel

import utilitis_FEEC.kernels_sparse as kernels


kernels = epyccel(kernels)


def mass_matrix_V0(p, Nbase, T, g_sqrt, bc):
    """
    Computes the mass matrix in the space V0 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    g : callable
        square root of the Jacobi determinant of the metric tensor g
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions)
        
        
    Returns
    -------
    M: sparse matrix
        mass matrix in V0
    """
    
    
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc
    
    if bc_1 == True:
        N1 = Nbase_1 - p1
        
    else:
        N1 = Nbase_1
        
    if bc_2 == True:
        N2 = Nbase_2 - p2
        
    else:
        N2 = Nbase_2
        
    if bc_3 == True:
        N3 = Nbase_3 - p3
        
    else:
        N3 = Nbase_3
        
    el_b_1 = inter.construct_grid_from_knots(p1, Nbase_1, T1)
    el_b_2 = inter.construct_grid_from_knots(p2, Nbase_2, T2)
    el_b_3 = inter.construct_grid_from_knots(p3, Nbase_3, T3)
    
    Nel_1 = len(el_b_1) - 1
    Nel_2 = len(el_b_2) - 1
    Nel_3 = len(el_b_3) - 1
    
    pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
    pts_2_loc, wts_2_loc = np.polynomial.legendre.leggauss(p2 + 1)
    pts_3_loc, wts_3_loc = np.polynomial.legendre.leggauss(p3 + 1)

    pts_1, wts_1 = inter.construct_quadrature_grid(Nel_1, p1 + 1, pts_1_loc, wts_1_loc, el_b_1)
    pts_2, wts_2 = inter.construct_quadrature_grid(Nel_2, p2 + 1, pts_2_loc, wts_2_loc, el_b_2)
    pts_3, wts_3 = inter.construct_quadrature_grid(Nel_3, p3 + 1, pts_3_loc, wts_3_loc, el_b_3)

    basis0_1 = inter.eval_on_grid_splines_ders(p1, Nbase_1, p1 + 1, 0, T1, pts_1) 
    basis0_2 = inter.eval_on_grid_splines_ders(p2, Nbase_2, p2 + 1, 0, T2, pts_2) 
    basis0_3 = inter.eval_on_grid_splines_ders(p3, Nbase_3, p3 + 1, 0, T3, pts_3)
    
    
    # Create element matrices
    b_loc = (p1 + 1)*(p2 + 1)*(p3 + 1)
    
    mat_m = np.empty(b_loc**2)
    mat_g = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F')                         
    
    
    # Create lists for local indices
    i_loc = np.empty(b_loc**2)
    j_loc = np.empty(b_loc**2)
    
    i_glob = np.array([])
    j_glob = np.array([])
    m_glob = np.array([])
    
    
    # Assembly
    for n1 in range(Nel_1):
        for n2 in range(Nel_2):
            for n3 in range(Nel_3):
                
                w1 = wts_1[:, n1]
                w2 = wts_2[:, n2]
                w3 = wts_3[:, n3]
                
                bs0_1 = basis0_1[:, 0, :, n1]
                bs0_2 = basis0_2[:, 0, :, n2]
                bs0_3 = basis0_3[:, 0, :, n3]
                
                Pts1, Pts2, Pts3 = np.meshgrid(pts_1[:, n1], pts_2[:, n2], pts_3[:, n3], indexing='ij')
                mat_g[:, :, :] = g_sqrt(Pts1, Pts2, Pts3)
                
                kernels.kernel0(N1, N2, N3, n1, n2, n3, p1, p2, p3, bs0_1, bs0_2, bs0_3, w1, w2, w3, mat_g, mat_m, i_loc, j_loc)
                
                i_glob = np.append(i_glob, i_loc)
                j_glob = np.append(j_glob, j_loc)
                m_glob = np.append(m_glob, mat_m)
     
    print('assembly done!')
          
    M = sparse.coo_matrix((m_glob, (i_glob, j_glob)), shape=(N1*N2*N3, N1*N2*N3)).tolil()
    
    if bc_1 == False:
        
        M[:N2*N3, :]  = 0.
        M[-N2*N3:, :] = 0.
        
        M[:, :N2*N3]  = 0.
        M[:, -N2*N3:] = 0.
        
    if bc_2 == False:
        
        left  = np.mod(np.arange(m_glob.size), N2*N3) <  N3
        right = np.mod(np.arange(m_glob.size), N2*N3) >= (N2 - 1)*N3
        
        #m_glob[left]  = 0.
        #m_glob[right] = 0.
        
    if bc_3 == False:
        
        m_glob[0::N3]  = 0.
        m_glob[N3::N3] = 0.
        
                
    
    # ...   
    
                    
    return M.tocsc()