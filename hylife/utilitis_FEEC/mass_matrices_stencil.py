from psydac.linalg.stencil import StencilMatrix, StencilVector, StencilVectorSpace
from psydac.linalg.block   import ProductSpace, BlockVector, BlockLinearOperator, BlockMatrix

import numpy                              as np
import utilitis_FEEC.bsplines             as bsp
import utilitis_FEEC.kernels_mass_stencil as kernels





#==================================================calling epyccel for acceleration===========================================
from pyccel import epyccel
kernels = epyccel(kernels)
#=============================================================================================================================






#============== mass matrix in V0 in 1d ======================================================================================
def mass_matrix_V0_1d(p, Nbase, T, g_sqrt, bc):
    """
    Computes the mass matrix in the space V0 in 1d.
    
    Parameters
    ----------
    p : int
        spline degree
    
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    g_sqrt : callable
        square root of the Jacobi determinant of the metric tensor G
        
    bc : boolean
        boundary condition (True = periodic, False = else)
        
        
    Returns
    -------
    M0 : StencilMatrix
        mass matrix in V0
    """
    
    # Unpack degrees, total number of basis functions, knot vectors and boundary conditions in each direction
    p1 = p
    Nbase_1 = Nbase
    T1 = T
    bc_1 = bc
    
    # Changes for periodic case 1 (reduction of number of basis functions and additional cycles in assembly)
    ca_1 = bc_1*p1
    
    # Element boundaries and number of elements
    el_b_1 = bsp.breakpoints(T1, p1)
    Nel_1 = len(el_b_1) - 1
     
    # Changes for periodic case 2 (periodicity in element cycles)
    cb_1 = (Nel_1 - p1)*bc_1
    
    # Local and global quadrature points and weights (Gauss-Legendre)
    pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
    pts_1, wts_1 = bsp.quadrature_grid(el_b_1, pts_1_loc, wts_1_loc)
    
    # Evaluation of basis functions at global quadrature points 
    d = 0
    basis0_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T1, p1, pts_1, d))
    
    # Create stencil vector spaces for trial and test space
    V0_i = StencilVectorSpace([Nbase_1 - ca_1], [p1], [bc_1])
    V0_j = StencilVectorSpace([Nbase_1 - ca_1], [p1], [bc_1])
    
    # Create global stencil matrix (columns, lines)
    M0 = StencilMatrix(V0_j, V0_i)

    # Create element matrices
    mat_m = np.zeros((p1 + 1, 2*p1 + 1), order='F') # local mass matrix
    mat_g = np.empty(p1 + 1)                        # Jacobi determinant at local quadrature points
    
    # Build global matrices: cycle over elements
    for ie1 in range(Nel_1 + ca_1):
                
        ie1_p = (ie1 + cb_1)%Nel_1

        # Get B-spline values and quadrature weights
        bs0_1 = basis0_1[ie1_p, :, 0, :]
        w1  = wts_1[ie1_p, :]

        # Evaluate Jacobi determinant at all quadrature points
        mat_g[:] = g_sqrt(pts_1[ie1_p, :])

        # Compute element matrix
        kernels.kernel_V0_1d(p1, p1 + 1, bs0_1, bs0_1, w1, mat_g, mat_m)

        # Update global matrix
        is1 = ie1 + p1 - ca_1

        M0[is1 - p1:is1 + 1, :] += mat_m[:, :]

        # Make sure that periodic corners are zero in non-periodic case
        M0.remove_spurious_entries()
                
    return M0
#=============================================================================================================================



#============== mass matrix in V1 in 1d ======================================================================================
def mass_matrix_V1_1d(p, Nbase, T, g_sqrt, bc):
    """
    Computes the mass matrix in the space V1 in 1d.
    
    Parameters
    ----------
    p : int
        spline degree
    
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    g_sqrt : callable
        square root of the Jacobi determinant of the metric tensor G
        
    bc : boolean
        boundary condition (True = periodic, False = else)
        
        
    Returns
    -------
    M1 : StencilMatrix
        mass matrix in V1
    """
    
    # Unpack degrees, total number of basis functions, knot vectors and boundary conditions in each direction
    p1 = p
    Nbase_1 = Nbase
    T1 = T
    bc_1 = bc
    
    t1 = T1[1:-1]
    
    # Changes for periodic case 1 (reduction of number of basis functions and additional cycles in assembly)
    ca_1 = bc_1*(p1 - 1)
    
    # Element boundaries and number of elements
    el_b_1 = bsp.breakpoints(T1, p1)
    Nel_1 = len(el_b_1) - 1
     
    # Changes for periodic case 2 (periodicity in element cycles)
    cb_1 = (Nel_1 - (p1 - 1))*bc_1
    
    # Local and global quadrature points and weights (Gauss-Legendre)
    pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
    pts_1, wts_1 = bsp.quadrature_grid(el_b_1, pts_1_loc, wts_1_loc)
    
    # Evaluation of basis functions at global quadrature points 
    d = 0
    basis1_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t1, p1 - 1, pts_1, d, normalize=True))
    
    # Create stencil vector spaces for trial and test space
    V0_i = StencilVectorSpace([Nbase_1 - ca_1 - 1], [p1], [bc_1])
    V0_j = StencilVectorSpace([Nbase_1 - ca_1 - 1], [p1], [bc_1])
    
    # Create global stencil matrix (columns, lines)
    M1 = StencilMatrix(V0_j, V0_i)

    # Create element matrices
    mat_m = np.zeros((p1, 2*p1 + 1), order='F') # local mass matrix
    mat_g = np.empty(p1 + 1)                    # Jacobi determinant at local quadrature points
    
    # Build global matrices: cycle over elements
    for ie1 in range(Nel_1 + ca_1):
                
        ie1_p = (ie1 + cb_1)%Nel_1

        # Get B-spline values and quadrature weights
        bs1_1 = basis1_1[ie1_p, :, 0, :]
        w1  = wts_1[ie1_p, :]

        # Evaluate Jacobi determinant and inverse metric tensor at all quadrature points
        mat_g[:] = g_sqrt(pts_1[ie1_p, :])

        # Compute element matrix
        kernels.kernel_V1_1d(p1, p1 + 1, bs1_1, bs1_1, w1, mat_g, mat_m)

        # Update global matrix
        is1 = ie1 + p1 - ca_1

        M1[is1 - p1:is1, :] += mat_m[:, :]

        # Make sure that periodic corners are zero in non-periodic case
        M1.remove_spurious_entries()
                
    return M1
#=============================================================================================================================



#============== mass matrix in V0 in 3d ======================================================================================
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
        
    g_sqrt : callable
        square root of the Jacobi determinant of the metric tensor G
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = else)
        
        
    Returns
    -------
    M0 : StencilMatrix
        mass matrix in V0
    """
    
    # Unpack degrees, total number of basis functions, knot vectors and boundary conditions in each direction
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc
    
    # Changes for periodic case 1 (reduction of number of basis functions and additional cycles in assembly)
    ca_1 = bc_1*p1
    ca_2 = bc_2*p2
    ca_3 = bc_3*p3
    
    # Element boundaries and number of elements
    el_b_1 = bsp.breakpoints(T1, p1)
    el_b_2 = bsp.breakpoints(T2, p2)
    el_b_3 = bsp.breakpoints(T3, p3)
    
    Nel_1 = len(el_b_1) - 1
    Nel_2 = len(el_b_2) - 1
    Nel_3 = len(el_b_3) - 1
    
    # Changes for periodic case 2 (periodicity in element cycles)
    cb_1 = (Nel_1 - p1)*bc_1
    cb_2 = (Nel_2 - p2)*bc_2
    cb_3 = (Nel_3 - p3)*bc_3
    
    # Local and global quadrature points and weights (Gauss-Legendre)
    pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
    pts_2_loc, wts_2_loc = np.polynomial.legendre.leggauss(p2 + 1)
    pts_3_loc, wts_3_loc = np.polynomial.legendre.leggauss(p3 + 1)

    pts_1, wts_1 = bsp.quadrature_grid(el_b_1, pts_1_loc, wts_1_loc)
    pts_2, wts_2 = bsp.quadrature_grid(el_b_2, pts_2_loc, wts_2_loc)
    pts_3, wts_3 = bsp.quadrature_grid(el_b_3, pts_3_loc, wts_3_loc)
    
    # Evaluation of basis functions at global quadrature points 
    d = 0
    basis0_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T1, p1, pts_1, d))
    basis0_2 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T2, p2, pts_2, d))
    basis0_3 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T3, p3, pts_3, d))
    
    # Create stencil vector space of basis
    V0 = StencilVectorSpace([Nbase_1 - ca_1, Nbase_2 - ca_2, Nbase_3 - ca_3], [p1, p2, p3], [bc_1, bc_2, bc_3])
    
    # Create global stencil matrix (columns, lines)
    M0 = StencilMatrix(V0, V0)

    # Create element matrices
    mat_m = np.zeros((p1 + 1, p2 + 1, p3 + 1, 2*p1 + 1, 2*p2 + 1, 2*p3 + 1), order='F') # local mass matrix
    mat_g = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F')                               # Jacobi determinant
    
    # Build global matrices: cycle over elements
    for ie1 in range(Nel_1 + ca_1):
        for ie2 in range(Nel_2 + ca_2):
            for ie3 in range(Nel_3 + ca_3):
                
                ie1_p = (ie1 + cb_1)%Nel_1
                ie2_p = (ie2 + cb_2)%Nel_2
                ie3_p = (ie3 + cb_3)%Nel_3
                
                # Get B-spline values and quadrature weights
                bs0_1 = basis0_1[ie1_p, :, 0, :]
                bs0_2 = basis0_2[ie2_p, :, 0, :]
                bs0_3 = basis0_3[ie3_p, :, 0, :]
                
                w1  = wts_1[ie1_p, :]
                w2  = wts_2[ie2_p, :]
                w3  = wts_3[ie3_p, :]
                
                # Evaluate Jacobi determinant at all quadrature points
                Pts1, Pts2, Pts3 = np.meshgrid(pts_1[ie1_p, :], pts_2[ie2_p, :], pts_3[ie3_p, :], indexing='ij')
                mat_g[:, :, :] = g_sqrt(Pts1, Pts2, Pts3)
                
                # Compute element matrix
                kernels.kernel_V0(p1, p2, p3, p1 + 1, p2 + 1, p3 + 1, bs0_1, bs0_2, bs0_3, bs0_1, bs0_2, bs0_3, w1, w2, w3, mat_g, mat_m)
                
                # Update global matrix
                is1 = ie1 + p1 - ca_1
                is2 = ie2 + p2 - ca_2
                is3 = ie3 + p3 - ca_3
                
                M0[is1 - p1:is1 + 1, is2 - p2:is2 + 1, is3 - p3:is3 + 1, :, :, :] += mat_m[:, :, :, :, :, :]
                
                # Make sure that periodic corners are zero in non-periodic case
                M0.remove_spurious_entries()
                
    return M0
#=============================================================================================================================



#============== mass matrix in V1 in 3d ======================================================================================
def mass_matrix_V1(p, Nbase, T, Ginv, g_sqrt, bc):
    """
    Computes the mass matrix in the space V1 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    Ginv : callable
        the inverse of the metric tensor G
        
    g_sqrt : callable
        square root of the Jacobi determinant of the metric tensor G
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = else)
        
        
    Returns
    -------
    M1 : list of StencilMatrix
        blocks of the mass matrix in V1
        
    V1 : ProductSpace
        product space composed of the three components in V1
    """
    
    # Unpack degrees, total number of basis functions, knot vectors and boundary conditions in each direction
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc
    
    t1 = T1[1:-1]
    t2 = T2[1:-1]
    t3 = T3[1:-1]
    
    ns = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    # Changes for periodic case 1 (reduction of number of basis functions and additional cycles in assembly)
    ca0_1 = bc_1*p1
    ca0_2 = bc_2*p2
    ca0_3 = bc_3*p3
    
    ca1_1 = bc_1*(p1 - 1)
    ca1_2 = bc_2*(p2 - 1)
    ca1_3 = bc_3*(p3 - 1)
    
    ca = [[ca1_1, ca0_2, ca0_3], [ca0_1, ca1_2, ca0_3], [ca0_1, ca0_2, ca1_3]]
    
    # Element boundaries and number of elements
    el_b_1 = bsp.breakpoints(T1, p1)
    el_b_2 = bsp.breakpoints(T2, p2)
    el_b_3 = bsp.breakpoints(T3, p3)
    
    Nel_1 = len(el_b_1) - 1
    Nel_2 = len(el_b_2) - 1
    Nel_3 = len(el_b_3) - 1
    
    # Changes for periodic case 2 (periodicity in element cycles)
    cb0_1 = (Nel_1 - p1)*bc_1
    cb0_2 = (Nel_2 - p2)*bc_2
    cb0_3 = (Nel_3 - p3)*bc_3
    
    cb1_1 = (Nel_1 - (p1 - 1))*bc_1
    cb1_2 = (Nel_2 - (p2 - 1))*bc_2
    cb1_3 = (Nel_3 - (p3 - 1))*bc_3
    
    cb = [[cb1_1, cb0_2, cb0_3], [cb0_1, cb1_2, cb0_3], [cb0_1, cb0_2, cb1_3]]
    
    # Local and global quadrature points and weights (Gauss-Legendre)
    pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
    pts_2_loc, wts_2_loc = np.polynomial.legendre.leggauss(p2 + 1)
    pts_3_loc, wts_3_loc = np.polynomial.legendre.leggauss(p3 + 1)

    pts_1, wts_1 = bsp.quadrature_grid(el_b_1, pts_1_loc, wts_1_loc)
    pts_2, wts_2 = bsp.quadrature_grid(el_b_2, pts_2_loc, wts_2_loc)
    pts_3, wts_3 = bsp.quadrature_grid(el_b_3, pts_3_loc, wts_3_loc)
    
    # Evaluation of basis functions at global quadrature points 
    d = 0
    basis0_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T1, p1, pts_1, d))
    basis0_2 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T2, p2, pts_2, d))
    basis0_3 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T3, p3, pts_3, d))
    
    basis1_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t1, p1 - 1, pts_1, d, normalize=True))
    basis1_2 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t2, p2 - 1, pts_2, d, normalize=True))
    basis1_3 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t3, p3 - 1, pts_3, d, normalize=True))
    
    basis = [[basis1_1, basis0_2, basis0_3], [basis0_1, basis1_2, basis0_3], [basis0_1, basis0_2, basis1_3]]
    
    # Create stencil vector spaces for three components and product space
    V1_1 = StencilVectorSpace([Nbase_1 - ca1_1 - 1, Nbase_2 - ca0_2, Nbase_3 - ca0_3], [p1, p2, p3], [bc_1, bc_2, bc_3])
    V1_2 = StencilVectorSpace([Nbase_1 - ca0_1, Nbase_2 - ca1_2 - 1, Nbase_3 - ca0_3], [p1, p2, p3], [bc_1, bc_2, bc_3])
    V1_3 = StencilVectorSpace([Nbase_1 - ca0_1, Nbase_2 - ca0_2, Nbase_3 - ca1_3 - 1], [p1, p2, p3], [bc_1, bc_2, bc_3])
    
    V1   = ProductSpace(V1_1, V1_2, V1_3)
    
    
    # Create global stencil matrices (columns, lines)
    M1_11 = StencilMatrix(V1_1, V1_1)
    M1_12 = StencilMatrix(V1_2, V1_1)
    M1_13 = StencilMatrix(V1_3, V1_1)
    
    M1_21 = StencilMatrix(V1_1, V1_2)
    M1_22 = StencilMatrix(V1_2, V1_2)
    M1_23 = StencilMatrix(V1_3, V1_2)
    
    M1_31 = StencilMatrix(V1_1, V1_3)
    M1_32 = StencilMatrix(V1_2, V1_3)
    M1_33 = StencilMatrix(V1_3, V1_3)
    
    M1 = [[M1_11, M1_12, M1_13], [M1_21, M1_22, M1_23], [M1_31, M1_32, M1_33]]

    # Create element matrices for metric
    mat_g    = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') 
    mat_Ginv = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') 
    
    
    # Cycle over components
    for a in range(3):
        
        ca_1, ca_2, ca_3 = ca[a]
        cb_1, cb_2, cb_3 = cb[a]
        
        for b in range(3):
            
            # Get basis functions
            basis_i_1, basis_i_2, basis_i_3 = basis[a]
            basis_j_1, basis_j_2, basis_j_3 = basis[b]
            
            # Get spline degrees
            n1_i, n2_i, n3_i = ns[a]
            n1_j, n2_j, n3_j = ns[b]
            
            # Create element matrix
            mat_m = np.zeros((p1 + 1 - n1_i, p2 + 1 - n2_i, p3 + 1 - n3_i, 2*p1 + 1, 2*p2 + 1, 2*p3 + 1), order='F')
            
            # Build global matrices: cycle over elements
            for ie1 in range(Nel_1 + ca_1):
                for ie2 in range(Nel_2 + ca_2):
                    for ie3 in range(Nel_3 + ca_3):

                        ie1_p = (ie1 + cb_1)%Nel_1
                        ie2_p = (ie2 + cb_2)%Nel_2
                        ie3_p = (ie3 + cb_3)%Nel_3

                        # Get B-spline values and quadrature weights
                        bsi_1 = basis_i_1[ie1_p, :, 0, :]
                        bsi_2 = basis_i_2[ie2_p, :, 0, :]
                        bsi_3 = basis_i_3[ie3_p, :, 0, :]
                        
                        bsj_1 = basis_j_1[ie1_p, :, 0, :]
                        bsj_2 = basis_j_2[ie2_p, :, 0, :]
                        bsj_3 = basis_j_3[ie3_p, :, 0, :]

                        w1  = wts_1[ie1_p, :]
                        w2  = wts_2[ie2_p, :]
                        w3  = wts_3[ie3_p, :]

                        # Evaluate Jacobi determinant at all quadrature points
                        Pts1, Pts2, Pts3 = np.meshgrid(pts_1[ie1_p, :], pts_2[ie2_p, :], pts_3[ie3_p, :], indexing='ij')
                        
                        mat_g[:, :, :]    = g_sqrt(Pts1, Pts2, Pts3)
                        mat_Ginv[:, :, :] = Ginv[a][b](Pts1, Pts2, Pts3)

                        # Compute element matrix
                        kernels.kernel_V1(p1, p2, p3, n1_i, n2_i, n3_i, n1_j, n2_j, n3_j, p1 + 1, p2 + 1, p3 + 1, bsi_1, bsi_2, bsi_3, bsj_1, bsj_2, bsj_3, w1, w2, w3, mat_Ginv, mat_g, mat_m)

                        # Update global matrix
                        is1 = ie1 + p1 - ca_1
                        is2 = ie2 + p2 - ca_2
                        is3 = ie3 + p3 - ca_3

                        M1[a][b][is1 - p1:is1 + 1 - n1_i, is2 - p2:is2 + 1 - n2_i, is3 - p3:is3 + 1 - n3_i, :, :, :] += mat_m[:, :, :, :, :, :]

                        # Make sure that periodic corners are zero in non-periodic case
                        M1[a][b].remove_spurious_entries()
    
    return M1, V1
#=============================================================================================================================



#============== mass matrix in V2 in 3d ======================================================================================
def mass_matrix_V2(p, Nbase, T, G, g_sqrt, bc):
    """
    Computes the mass matrix in the space V2 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    G : callable
        the metric tensor G
        
    g_sqrt : callable
        square root of the Jacobi determinant of the metric tensor G
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = else)
        
        
    Returns
    -------
    M2 : list of StencilMatrix
        blocks of the mass matrix in V2
        
    V2 : ProductSpace
        product space composed of the three components in V2
    """
    
    # Unpack degrees, total number of basis functions, knot vectors and boundary conditions in each direction
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc
    
    t1 = T1[1:-1]
    t2 = T2[1:-1]
    t3 = T3[1:-1]
    
    ns = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    
    # Changes for periodic case 1 (reduction of number of basis functions and additional cycles in assembly)
    ca0_1 = bc_1*p1
    ca0_2 = bc_2*p2
    ca0_3 = bc_3*p3
    
    ca1_1 = bc_1*(p1 - 1)
    ca1_2 = bc_2*(p2 - 1)
    ca1_3 = bc_3*(p3 - 1)
    
    ca = [[ca0_1, ca1_2, ca1_3], [ca1_1, ca0_2, ca1_3], [ca1_1, ca1_2, ca0_3]]
    
    # Element boundaries and number of elements
    el_b_1 = bsp.breakpoints(T1, p1)
    el_b_2 = bsp.breakpoints(T2, p2)
    el_b_3 = bsp.breakpoints(T3, p3)
    
    Nel_1 = len(el_b_1) - 1
    Nel_2 = len(el_b_2) - 1
    Nel_3 = len(el_b_3) - 1
    
    # Changes for periodic case 2 (periodicity in element cycles)
    cb0_1 = (Nel_1 - p1)*bc_1
    cb0_2 = (Nel_2 - p2)*bc_2
    cb0_3 = (Nel_3 - p3)*bc_3
    
    cb1_1 = (Nel_1 - (p1 - 1))*bc_1
    cb1_2 = (Nel_2 - (p2 - 1))*bc_2
    cb1_3 = (Nel_3 - (p3 - 1))*bc_3
    
    cb = [[cb0_1, cb1_2, cb1_3], [cb1_1, cb0_2, cb1_3], [cb1_1, cb1_2, cb0_3]]
    
    # Local and global quadrature points and weights (Gauss-Legendre)
    pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
    pts_2_loc, wts_2_loc = np.polynomial.legendre.leggauss(p2 + 1)
    pts_3_loc, wts_3_loc = np.polynomial.legendre.leggauss(p3 + 1)

    pts_1, wts_1 = bsp.quadrature_grid(el_b_1, pts_1_loc, wts_1_loc)
    pts_2, wts_2 = bsp.quadrature_grid(el_b_2, pts_2_loc, wts_2_loc)
    pts_3, wts_3 = bsp.quadrature_grid(el_b_3, pts_3_loc, wts_3_loc)
    
    # Evaluation of basis functions at global quadrature points 
    d = 0
    basis0_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T1, p1, pts_1, d))
    basis0_2 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T2, p2, pts_2, d))
    basis0_3 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T3, p3, pts_3, d))
    
    basis1_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t1, p1 - 1, pts_1, d, normalize=True))
    basis1_2 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t2, p2 - 1, pts_2, d, normalize=True))
    basis1_3 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t3, p3 - 1, pts_3, d, normalize=True))
    
    basis = [[basis0_1, basis1_2, basis1_3], [basis1_1, basis0_2, basis1_3], [basis1_1, basis1_2, basis0_3]]
    
    # Create stencil vector spaces for three components and product space
    V2_1 = StencilVectorSpace([Nbase_1 - ca0_1, Nbase_2 - ca1_2 - 1, Nbase_3 - ca1_3 - 1], [p1, p2, p3], [bc_1, bc_2, bc_3])
    V2_2 = StencilVectorSpace([Nbase_1 - ca1_1 - 1, Nbase_2 - ca0_2, Nbase_3 - ca1_3 - 1], [p1, p2, p3], [bc_1, bc_2, bc_3])
    V2_3 = StencilVectorSpace([Nbase_1 - ca1_1 - 1, Nbase_2 - ca1_2 - 1, Nbase_3 - ca0_3], [p1, p2, p3], [bc_1, bc_2, bc_3])
    
    V2   = ProductSpace(V2_1, V2_2, V2_3)
    
    
    # Create global stencil matrices (columns, lines)
    M2_11 = StencilMatrix(V2_1, V2_1)
    M2_12 = StencilMatrix(V2_2, V2_1)
    M2_13 = StencilMatrix(V2_3, V2_1)
    
    M2_21 = StencilMatrix(V2_1, V2_2)
    M2_22 = StencilMatrix(V2_2, V2_2)
    M2_23 = StencilMatrix(V2_3, V2_2)
    
    M2_31 = StencilMatrix(V2_1, V2_3)
    M2_32 = StencilMatrix(V2_2, V2_3)
    M2_33 = StencilMatrix(V2_3, V2_3)
    
    M2 = [[M2_11, M2_12, M2_13], [M2_21, M2_22, M2_23], [M2_31, M2_32, M2_33]]

    # Create element matrices for metric
    mat_g  = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') 
    mat_GG = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') 
    
    
    # Cycle over components
    for a in range(3):
        
        ca_1, ca_2, ca_3 = ca[a]
        cb_1, cb_2, cb_3 = cb[a]
        
        for b in range(3):
            
            # Get basis functions
            basis_i_1, basis_i_2, basis_i_3 = basis[a]
            basis_j_1, basis_j_2, basis_j_3 = basis[b]
            
            # Get spline degrees
            n1_i, n2_i, n3_i = ns[a]
            n1_j, n2_j, n3_j = ns[b]
            
            # Create element matrix
            mat_m = np.zeros((p1 + 1 - n1_i, p2 + 1 - n2_i, p3 + 1 - n3_i, 2*p1 + 1, 2*p2 + 1, 2*p3 + 1), order='F')
            
            # Build global matrices: cycle over elements
            for ie1 in range(Nel_1 + ca_1):
                for ie2 in range(Nel_2 + ca_2):
                    for ie3 in range(Nel_3 + ca_3):

                        ie1_p = (ie1 + cb_1)%Nel_1
                        ie2_p = (ie2 + cb_2)%Nel_2
                        ie3_p = (ie3 + cb_3)%Nel_3

                        # Get B-spline values and quadrature weights
                        bsi_1 = basis_i_1[ie1_p, :, 0, :]
                        bsi_2 = basis_i_2[ie2_p, :, 0, :]
                        bsi_3 = basis_i_3[ie3_p, :, 0, :]
                        
                        bsj_1 = basis_j_1[ie1_p, :, 0, :]
                        bsj_2 = basis_j_2[ie2_p, :, 0, :]
                        bsj_3 = basis_j_3[ie3_p, :, 0, :]

                        w1  = wts_1[ie1_p, :]
                        w2  = wts_2[ie2_p, :]
                        w3  = wts_3[ie3_p, :]

                        # Evaluate Jacobi determinant at all quadrature points
                        Pts1, Pts2, Pts3 = np.meshgrid(pts_1[ie1_p, :], pts_2[ie2_p, :], pts_3[ie3_p, :], indexing='ij')
                        
                        mat_g[:, :, :]  = g_sqrt(Pts1, Pts2, Pts3)
                        mat_GG[:, :, :] = G[a][b](Pts1, Pts2, Pts3)

                        # Compute element matrix
                        kernels.kernel_V2(p1, p2, p3, n1_i, n2_i, n3_i, n1_j, n2_j, n3_j, p1 + 1, p2 + 1, p3 + 1, bsi_1, bsi_2, bsi_3, bsj_1, bsj_2, bsj_3, w1, w2, w3, mat_GG, mat_g, mat_m)

                        # Update global matrix
                        is1 = ie1 + p1 - ca_1
                        is2 = ie2 + p2 - ca_2
                        is3 = ie3 + p3 - ca_3

                        M2[a][b][is1 - p1:is1 + 1 - n1_i, is2 - p2:is2 + 1 - n2_i, is3 - p3:is3 + 1 - n3_i, :, :, :] += mat_m[:, :, :, :, :, :]

                        # Make sure that periodic corners are zero in non-periodic case
                        M2[a][b].remove_spurious_entries()
    
    return M2, V2
#=============================================================================================================================



#============== mass matrix in V3 in 3d ======================================================================================
def mass_matrix_V3(p, Nbase, T, g_sqrt, bc):
    """
    Computes the mass matrix in the space V3 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    g_sqrt : callable
        square root of the Jacobi determinant of the metric tensor G
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = else)
        
        
    Returns
    -------
    M3 : StencilMatrix
        mass matrix in V3
    """
    
    # Unpack degrees, total number of basis functions, knot vectors and boundary conditions in each direction
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc
    
    t1 = T1[1:-1]
    t2 = T2[1:-1]
    t3 = T3[1:-1]
    
    # Changes for periodic case 1 (reduction of number of basis functions and additional cycles in assembly)
    ca_1 = bc_1*(p1 - 1)
    ca_2 = bc_2*(p2 - 1)
    ca_3 = bc_3*(p3 - 1)
    
    # Element boundaries and number of elements
    el_b_1 = bsp.breakpoints(T1, p1)
    el_b_2 = bsp.breakpoints(T2, p2)
    el_b_3 = bsp.breakpoints(T3, p3)
    
    Nel_1 = len(el_b_1) - 1
    Nel_2 = len(el_b_2) - 1
    Nel_3 = len(el_b_3) - 1
    
    # Changes for periodic case 2 (periodicity in element cycles)
    cb_1 = (Nel_1 - (p1 - 1))*bc_1
    cb_2 = (Nel_2 - (p2 - 1))*bc_2
    cb_3 = (Nel_3 - (p3 - 1))*bc_3
    
    # Local and global quadrature points and weights (Gauss-Legendre)
    pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
    pts_2_loc, wts_2_loc = np.polynomial.legendre.leggauss(p2 + 1)
    pts_3_loc, wts_3_loc = np.polynomial.legendre.leggauss(p3 + 1)

    pts_1, wts_1 = bsp.quadrature_grid(el_b_1, pts_1_loc, wts_1_loc)
    pts_2, wts_2 = bsp.quadrature_grid(el_b_2, pts_2_loc, wts_2_loc)
    pts_3, wts_3 = bsp.quadrature_grid(el_b_3, pts_3_loc, wts_3_loc)
    
    # Evaluation of basis functions at global quadrature points 
    d = 0
    basis1_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t1, p1 - 1, pts_1, d, normalize=True))
    basis1_2 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t2, p2 - 1, pts_2, d, normalize=True))
    basis1_3 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t3, p3 - 1, pts_3, d, normalize=True))
    
    # Create stencil vector spaces for trial and test space
    V3 = StencilVectorSpace([Nbase_1 - ca_1 - 1, Nbase_2 - ca_2 - 1, Nbase_3 - ca_3 - 1], [p1, p2, p3], [bc_1, bc_2, bc_3])
    
    # Create global stencil matrix (columns, lines)
    M3 = StencilMatrix(V3, V3)

    # Create element matrices
    mat_m = np.zeros((p1, p2, p3, 2*p1 + 1, 2*p2 + 1, 2*p3 + 1), order='F') # local mass matrix
    mat_g = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F')                   # Jacobi determinant
    
    # Build global matrices: cycle over elements
    for ie1 in range(Nel_1 + ca_1):
        for ie2 in range(Nel_2 + ca_2):
            for ie3 in range(Nel_3 + ca_3):
                
                ie1_p = (ie1 + cb_1)%Nel_1
                ie2_p = (ie2 + cb_2)%Nel_2
                ie3_p = (ie3 + cb_3)%Nel_3
                
                # Get B-spline values and quadrature weights
                bs1_1 = basis1_1[ie1_p, :, 0, :]
                bs1_2 = basis1_2[ie2_p, :, 0, :]
                bs1_3 = basis1_3[ie3_p, :, 0, :]
                
                w1  = wts_1[ie1_p, :]
                w2  = wts_2[ie2_p, :]
                w3  = wts_3[ie3_p, :]
                
                # Evaluate Jacobi determinant at all quadrature points
                Pts1, Pts2, Pts3 = np.meshgrid(pts_1[ie1_p, :], pts_2[ie2_p, :], pts_3[ie3_p, :], indexing='ij')
                mat_g[:, :, :] = g_sqrt(Pts1, Pts2, Pts3)
                
                # Compute element matrix
                kernels.kernel_V3(p1, p2, p3, p1 + 1, p2 + 1, p3 + 1, bs1_1, bs1_2, bs1_3, bs1_1, bs1_2, bs1_3, w1, w2, w3, mat_g, mat_m)
                
                # Update global matrix
                is1 = ie1 + p1 - ca_1
                is2 = ie2 + p2 - ca_2
                is3 = ie3 + p3 - ca_3
                
                M3[is1 - p1:is1, is2 - p2:is2, is3 - p3:is3, :, :, :] += mat_m[:, :, :, :, :, :]
                
                # Make sure that periodic corners are zero in non-periodic case
                M3.remove_spurious_entries()
                
    return M3
#=============================================================================================================================



#============== inner product in V0 in 3d ====================================================================================
def inner_prod_V0(fun, p, Nbase, T, g_sqrt, bc):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V0 in general curvilinear coordinates
    q = (q1, q2, q3) with metric tensor G using a quadrature rule of order p + 1.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    g_sqrt : callable
        square root of the Jacobi determinant
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = else)
        
    Returns
    -------
    F0 : StencilVector
        the result of the integration with each basis function
    """
    
    # Unpack degrees, total number of basis functions, knot vectors and boundary conditions in each direction
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc
    
    # Changes for periodic case 1 (reduction of number of basis functions and additional cycles in assembly)
    ca_1 = bc_1*p1
    ca_2 = bc_2*p2
    ca_3 = bc_3*p3
    
    # Element boundaries and number of elements
    el_b_1 = bsp.breakpoints(T1, p1)
    el_b_2 = bsp.breakpoints(T2, p2)
    el_b_3 = bsp.breakpoints(T3, p3)
    
    Nel_1 = len(el_b_1) - 1
    Nel_2 = len(el_b_2) - 1
    Nel_3 = len(el_b_3) - 1
    
    # Changes for periodic case 2 (periodicity in element cycles)
    cb_1 = (Nel_1 - p1)*bc_1
    cb_2 = (Nel_2 - p2)*bc_2
    cb_3 = (Nel_3 - p3)*bc_3
    
    # Local and global quadrature points and weights (Gauss-Legendre)
    pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
    pts_2_loc, wts_2_loc = np.polynomial.legendre.leggauss(p2 + 1)
    pts_3_loc, wts_3_loc = np.polynomial.legendre.leggauss(p3 + 1)

    pts_1, wts_1 = bsp.quadrature_grid(el_b_1, pts_1_loc, wts_1_loc)
    pts_2, wts_2 = bsp.quadrature_grid(el_b_2, pts_2_loc, wts_2_loc)
    pts_3, wts_3 = bsp.quadrature_grid(el_b_3, pts_3_loc, wts_3_loc)
    
    # Evaluation of basis functions at global quadrature points 
    d = 0
    basis0_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T1, p1, pts_1, d))
    basis0_2 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T2, p2, pts_2, d))
    basis0_3 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T3, p3, pts_3, d))
    
    # Create stencil vector space for test space
    V0 = StencilVectorSpace([Nbase_1 - ca_1, Nbase_2 - ca_2, Nbase_3 - ca_3], [p1, p2, p3], [bc_1, bc_2, bc_3])
    
    # Create global stencil vector
    F0 = StencilVector(V0)

    # Create element matrices
    mat_m = np.zeros((p1 + 1, p2 + 1, p3 + 1), order='F') # local vector
    mat_g = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') # Jacobi determinant
    mat_f = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') # function at quadrature points
    
    # Build global matrices: cycle over elements
    for ie1 in range(Nel_1 + ca_1):
        for ie2 in range(Nel_2 + ca_2):
            for ie3 in range(Nel_3 + ca_3):
                
                ie1_p = (ie1 + cb_1)%Nel_1
                ie2_p = (ie2 + cb_2)%Nel_2
                ie3_p = (ie3 + cb_3)%Nel_3
                
                # Get B-spline values and quadrature weights
                bs0_1 = basis0_1[ie1_p, :, 0, :]
                bs0_2 = basis0_2[ie2_p, :, 0, :]
                bs0_3 = basis0_3[ie3_p, :, 0, :]
                
                w1  = wts_1[ie1_p, :]
                w2  = wts_2[ie2_p, :]
                w3  = wts_3[ie3_p, :]
                
                # Evaluate Jacobi determinant at all quadrature points
                Pts1, Pts2, Pts3 = np.meshgrid(pts_1[ie1_p, :], pts_2[ie2_p, :], pts_3[ie3_p, :], indexing='ij')
                
                mat_g[:, :, :] = g_sqrt(Pts1, Pts2, Pts3)
                mat_f[:, :, :] = fun(Pts1, Pts2, Pts3)
                
                # Compute element vector
                kernels.kernel_L0(p1, p2, p3, p1 + 1, p2 + 1, p3 + 1, bs0_1, bs0_2, bs0_3, w1, w2, w3, mat_f, mat_g, mat_m)
                
                # Update global vector
                is1 = ie1 + p1 - ca_1
                is2 = ie2 + p2 - ca_2
                is3 = ie3 + p3 - ca_3
                
                F0[is1 - p1:is1 + 1, is2 - p2:is2 + 1, is3 - p3:is3 + 1] += mat_m[:, :, :]
                
                # IMPORTANT: ghost regions must be up-to-date
                F0.update_ghost_regions()
                
    return F0
#=============================================================================================================================



#============== inner product in V1 in 3d ====================================================================================
def inner_prod_V1(fun, p, Nbase, T, Ginv, g_sqrt, bc):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V1 in general curvilinear coordinates
    q = (q1, q2, q3) with metric tensor G using a quadrature rule of order p + 1.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    Ginv : callable
        the inverse of the metric tensor G
        
    g_sqrt : callable
        square root of the Jacobi determinant
        
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = else)
        
    Returns
    -------
    F1 : list of StencilVectors
        the result of the integration with each basis function
    """
    
    # Unpack degrees, total number of basis functions, knot vectors and boundary conditions in each direction
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc
    
    t1 = T1[1:-1]
    t2 = T2[1:-1]
    t3 = T3[1:-1]
    
    ns = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    # Changes for periodic case 1 (reduction of number of basis functions and additional cycles in assembly)
    ca0_1 = bc_1*p1
    ca0_2 = bc_2*p2
    ca0_3 = bc_3*p3
    
    ca1_1 = bc_1*(p1 - 1)
    ca1_2 = bc_2*(p2 - 1)
    ca1_3 = bc_3*(p3 - 1)
    
    ca = [[ca1_1, ca0_2, ca0_3], [ca0_1, ca1_2, ca0_3], [ca0_1, ca0_2, ca1_3]]
    
    # Element boundaries and number of elements
    el_b_1 = bsp.breakpoints(T1, p1)
    el_b_2 = bsp.breakpoints(T2, p2)
    el_b_3 = bsp.breakpoints(T3, p3)
    
    Nel_1 = len(el_b_1) - 1
    Nel_2 = len(el_b_2) - 1
    Nel_3 = len(el_b_3) - 1
    
    # Changes for periodic case 2 (periodicity in element cycles)
    cb0_1 = (Nel_1 - p1)*bc_1
    cb0_2 = (Nel_2 - p2)*bc_2
    cb0_3 = (Nel_3 - p3)*bc_3
    
    cb1_1 = (Nel_1 - (p1 - 1))*bc_1
    cb1_2 = (Nel_2 - (p2 - 1))*bc_2
    cb1_3 = (Nel_3 - (p3 - 1))*bc_3
    
    cb = [[cb1_1, cb0_2, cb0_3], [cb0_1, cb1_2, cb0_3], [cb0_1, cb0_2, cb1_3]]
    
    # Local and global quadrature points and weights (Gauss-Legendre)
    pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
    pts_2_loc, wts_2_loc = np.polynomial.legendre.leggauss(p2 + 1)
    pts_3_loc, wts_3_loc = np.polynomial.legendre.leggauss(p3 + 1)

    pts_1, wts_1 = bsp.quadrature_grid(el_b_1, pts_1_loc, wts_1_loc)
    pts_2, wts_2 = bsp.quadrature_grid(el_b_2, pts_2_loc, wts_2_loc)
    pts_3, wts_3 = bsp.quadrature_grid(el_b_3, pts_3_loc, wts_3_loc)
    
    # Evaluation of basis functions at global quadrature points 
    d = 0
    basis0_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T1, p1, pts_1, d))
    basis0_2 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T2, p2, pts_2, d))
    basis0_3 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T3, p3, pts_3, d))
    
    basis1_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t1, p1 - 1, pts_1, d, normalize=True))
    basis1_2 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t2, p2 - 1, pts_2, d, normalize=True))
    basis1_3 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t3, p3 - 1, pts_3, d, normalize=True))
    
    basis = [[basis1_1, basis0_2, basis0_3], [basis0_1, basis1_2, basis0_3], [basis0_1, basis0_2, basis1_3]]
    
    # Create stencil vector spaces for three components and product space
    V1_1 = StencilVectorSpace([Nbase_1 - ca1_1 - 1, Nbase_2 - ca0_2, Nbase_3 - ca0_3], [p1, p2, p3], [bc_1, bc_2, bc_3])
    V1_2 = StencilVectorSpace([Nbase_1 - ca0_1, Nbase_2 - ca1_2 - 1, Nbase_3 - ca0_3], [p1, p2, p3], [bc_1, bc_2, bc_3])
    V1_3 = StencilVectorSpace([Nbase_1 - ca0_1, Nbase_2 - ca0_2, Nbase_3 - ca1_3 - 1], [p1, p2, p3], [bc_1, bc_2, bc_3])
    
    V1 = ProductSpace(V1_1, V1_2, V1_3)
    
    
    # Create global stencil vectors
    F1_11 = StencilVector(V1_1)
    F1_12 = StencilVector(V1_2)
    F1_13 = StencilVector(V1_3)
    
    F1_21 = StencilVector(V1_1)
    F1_22 = StencilVector(V1_2)
    F1_23 = StencilVector(V1_3)
    
    F1_31 = StencilVector(V1_1)
    F1_32 = StencilVector(V1_2)
    F1_33 = StencilVector(V1_3)
    
    F1 = [[F1_11, F1_12, F1_13], [F1_21, F1_22, F1_23], [F1_31, F1_32, F1_33]]

    # Create element matrices for metric
    mat_g    = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') 
    mat_Ginv = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') 
    mat_f    = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') 
    
    
    # Cycle over components
    for a in range(3):
        for b in range(3):
            
            ca_1, ca_2, ca_3 = ca[b]
            cb_1, cb_2, cb_3 = cb[b]
            
            # Get basis functions
            basis_i_1, basis_i_2, basis_i_3 = basis[b]
            
            # Get spline degrees
            n1_i, n2_i, n3_i = ns[b]
            
            # Create element matrix
            mat_m = np.zeros((p1 + 1 - n1_i, p2 + 1 - n2_i, p3 + 1 - n3_i), order='F')
            
            # Build global matrices: cycle over elements
            for ie1 in range(Nel_1 + ca_1):
                for ie2 in range(Nel_2 + ca_2):
                    for ie3 in range(Nel_3 + ca_3):

                        ie1_p = (ie1 + cb_1)%Nel_1
                        ie2_p = (ie2 + cb_2)%Nel_2
                        ie3_p = (ie3 + cb_3)%Nel_3

                        # Get B-spline values and quadrature weights
                        bsi_1 = basis_i_1[ie1_p, :, 0, :]
                        bsi_2 = basis_i_2[ie2_p, :, 0, :]
                        bsi_3 = basis_i_3[ie3_p, :, 0, :]

                        w1  = wts_1[ie1_p, :]
                        w2  = wts_2[ie2_p, :]
                        w3  = wts_3[ie3_p, :]

                        # Evaluate Jacobi determinant at all quadrature points
                        Pts1, Pts2, Pts3 = np.meshgrid(pts_1[ie1_p, :], pts_2[ie2_p, :], pts_3[ie3_p, :], indexing='ij')
                        
                        mat_g[:, :, :]    = g_sqrt(Pts1, Pts2, Pts3)
                        mat_Ginv[:, :, :] = Ginv[a][b](Pts1, Pts2, Pts3)
                        mat_f[:, :, :]    = fun[a](Pts1, Pts2, Pts3)

                        # Compute element matrix
                        kernels.kernel_L1(p1, p2, p3, n1_i, n2_i, n3_i, p1 + 1, p2 + 1, p3 + 1, bsi_1, bsi_2, bsi_3, w1, w2, w3, mat_f, mat_Ginv, mat_g, mat_m)

                        # Update global matrix
                        is1 = ie1 + p1 - ca_1
                        is2 = ie2 + p2 - ca_2
                        is3 = ie3 + p3 - ca_3

                        F1[a][b][is1 - p1:is1 + 1 - n1_i, is2 - p2:is2 + 1 - n2_i, is3 - p3:is3 + 1 - n3_i] += mat_m[:, :, :]
                        
                        
    F_1 = F1_11 + F1_21 + F1_31
    F_2 = F1_12 + F1_22 + F1_32
    F_3 = F1_13 + F1_23 + F1_33
    
    # IMPORTANT: ghost regions must be up-to-date
    F_1.update_ghost_regions()
    F_2.update_ghost_regions()
    F_3.update_ghost_regions()
    
    return [F_1, F_2, F_3], V1
#=============================================================================================================================



#============== inner product in V2 in 3d ====================================================================================
def inner_prod_V2(fun, p, Nbase, T, G, g_sqrt, bc):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V2 in general curvilinear coordinates
    q = (q1, q2, q3) with metric tensor G using a quadrature rule of order p + 1.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    g_sqrt : callable
        square root of the Jacobi determinant
        
    G : callable
        the metric tensor G
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = else)
        
    Returns
    -------
    F2 : list of StencilVectors
        the result of the integration with each basis function
    """
    
    # Unpack degrees, total number of basis functions, knot vectors and boundary conditions in each direction
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc
    
    t1 = T1[1:-1]
    t2 = T2[1:-1]
    t3 = T3[1:-1]
    
    ns = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    
    # Changes for periodic case 1 (reduction of number of basis functions and additional cycles in assembly)
    ca0_1 = bc_1*p1
    ca0_2 = bc_2*p2
    ca0_3 = bc_3*p3
    
    ca1_1 = bc_1*(p1 - 1)
    ca1_2 = bc_2*(p2 - 1)
    ca1_3 = bc_3*(p3 - 1)
    
    ca = [[ca0_1, ca1_2, ca1_3], [ca1_1, ca0_2, ca1_3], [ca1_1, ca1_2, ca0_3]]
    
    # Element boundaries and number of elements
    el_b_1 = bsp.breakpoints(T1, p1)
    el_b_2 = bsp.breakpoints(T2, p2)
    el_b_3 = bsp.breakpoints(T3, p3)
    
    Nel_1 = len(el_b_1) - 1
    Nel_2 = len(el_b_2) - 1
    Nel_3 = len(el_b_3) - 1
    
    # Changes for periodic case 2 (periodicity in element cycles)
    cb0_1 = (Nel_1 - p1)*bc_1
    cb0_2 = (Nel_2 - p2)*bc_2
    cb0_3 = (Nel_3 - p3)*bc_3
    
    cb1_1 = (Nel_1 - (p1 - 1))*bc_1
    cb1_2 = (Nel_2 - (p2 - 1))*bc_2
    cb1_3 = (Nel_3 - (p3 - 1))*bc_3
    
    cb = [[cb0_1, cb1_2, cb1_3], [cb1_1, cb0_2, cb1_3], [cb1_1, cb1_2, cb0_3]]
    
    # Local and global quadrature points and weights (Gauss-Legendre)
    pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
    pts_2_loc, wts_2_loc = np.polynomial.legendre.leggauss(p2 + 1)
    pts_3_loc, wts_3_loc = np.polynomial.legendre.leggauss(p3 + 1)

    pts_1, wts_1 = bsp.quadrature_grid(el_b_1, pts_1_loc, wts_1_loc)
    pts_2, wts_2 = bsp.quadrature_grid(el_b_2, pts_2_loc, wts_2_loc)
    pts_3, wts_3 = bsp.quadrature_grid(el_b_3, pts_3_loc, wts_3_loc)
    
    # Evaluation of basis functions at global quadrature points 
    d = 0
    basis0_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T1, p1, pts_1, d))
    basis0_2 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T2, p2, pts_2, d))
    basis0_3 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T3, p3, pts_3, d))
    
    basis1_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t1, p1 - 1, pts_1, d, normalize=True))
    basis1_2 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t2, p2 - 1, pts_2, d, normalize=True))
    basis1_3 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t3, p3 - 1, pts_3, d, normalize=True))
    
    basis = [[basis0_1, basis1_2, basis1_3], [basis1_1, basis0_2, basis1_3], [basis1_1, basis1_2, basis0_3]]
    
    # Create stencil vector spaces for three components and product space
    V2_1 = StencilVectorSpace([Nbase_1 - ca0_1, Nbase_2 - ca1_2 - 1, Nbase_3 - ca1_3 - 1], [p1, p2, p3], [bc_1, bc_2, bc_3])
    V2_2 = StencilVectorSpace([Nbase_1 - ca1_1 - 1, Nbase_2 - ca0_2, Nbase_3 - ca1_3 - 1], [p1, p2, p3], [bc_1, bc_2, bc_3])
    V2_3 = StencilVectorSpace([Nbase_1 - ca1_1 - 1, Nbase_2 - ca1_2 - 1, Nbase_3 - ca0_3], [p1, p2, p3], [bc_1, bc_2, bc_3])
    
    V2 = ProductSpace(V2_1, V2_2, V2_3)
    
    
    # Create global stencil vectors
    F2_11 = StencilVector(V2_1)
    F2_12 = StencilVector(V2_2)
    F2_13 = StencilVector(V2_3)
    
    F2_21 = StencilVector(V2_1)
    F2_22 = StencilVector(V2_2)
    F2_23 = StencilVector(V2_3)
    
    F2_31 = StencilVector(V2_1)
    F2_32 = StencilVector(V2_2)
    F2_33 = StencilVector(V2_3)
    
    F2 = [[F2_11, F2_12, F2_13], [F2_21, F2_22, F2_23], [F2_31, F2_32, F2_33]]

    # Create element matrices for metric
    mat_g  = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') 
    mat_GG = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') 
    mat_f  = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') 
    
    
    # Cycle over components
    for a in range(3):
        for b in range(3):
            
            ca_1, ca_2, ca_3 = ca[b]
            cb_1, cb_2, cb_3 = cb[b]
            
            # Get basis functions
            basis_i_1, basis_i_2, basis_i_3 = basis[b]
            
            # Get spline degrees
            n1_i, n2_i, n3_i = ns[b]
            
            # Create element matrix
            mat_m = np.zeros((p1 + 1 - n1_i, p2 + 1 - n2_i, p3 + 1 - n3_i), order='F')
            
            # Build global matrices: cycle over elements
            for ie1 in range(Nel_1 + ca_1):
                for ie2 in range(Nel_2 + ca_2):
                    for ie3 in range(Nel_3 + ca_3):

                        ie1_p = (ie1 + cb_1)%Nel_1
                        ie2_p = (ie2 + cb_2)%Nel_2
                        ie3_p = (ie3 + cb_3)%Nel_3

                        # Get B-spline values and quadrature weights
                        bsi_1 = basis_i_1[ie1_p, :, 0, :]
                        bsi_2 = basis_i_2[ie2_p, :, 0, :]
                        bsi_3 = basis_i_3[ie3_p, :, 0, :]

                        w1  = wts_1[ie1_p, :]
                        w2  = wts_2[ie2_p, :]
                        w3  = wts_3[ie3_p, :]

                        # Evaluate Jacobi determinant at all quadrature points
                        Pts1, Pts2, Pts3 = np.meshgrid(pts_1[ie1_p, :], pts_2[ie2_p, :], pts_3[ie3_p, :], indexing='ij')
                        
                        mat_g[:, :, :]  = g_sqrt(Pts1, Pts2, Pts3)
                        mat_GG[:, :, :] = G[a][b](Pts1, Pts2, Pts3)
                        mat_f[:, :, :]  = fun[a](Pts1, Pts2, Pts3)

                        # Compute element matrix
                        kernels.kernel_L2(p1, p2, p3, n1_i, n2_i, n3_i, p1 + 1, p2 + 1, p3 + 1, bsi_1, bsi_2, bsi_3, w1, w2, w3, mat_f, mat_GG, mat_g, mat_m)

                        # Update global matrix
                        is1 = ie1 + p1 - ca_1
                        is2 = ie2 + p2 - ca_2
                        is3 = ie3 + p3 - ca_3

                        F2[a][b][is1 - p1:is1 + 1 - n1_i, is2 - p2:is2 + 1 - n2_i, is3 - p3:is3 + 1 - n3_i] += mat_m[:, :, :]
                        
                        
    F_1 = F2_11 + F2_21 + F2_31
    F_2 = F2_12 + F2_22 + F2_32
    F_3 = F2_13 + F2_23 + F2_33
    
    # IMPORTANT: ghost regions must be up-to-date
    F_1.update_ghost_regions()
    F_2.update_ghost_regions()
    F_3.update_ghost_regions()
    
    return [F_1, F_2, F_3], V2
#=============================================================================================================================



#============== inner product in V0 in 3d ====================================================================================
def inner_prod_V3(fun, p, Nbase, T, g_sqrt, bc):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V3 in general curvilinear coordinates
    q = (q1, q2, q3) with metric tensor G using a quadrature rule of order p + 1.
    
    Parameters
    ----------
    fun : callable
        function for scalar product
    
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    g_sqrt : callable
        square root of the Jacobi determinant
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = else)
        
    Returns
    -------
    F0 : StencilVector
        the result of the integration with each basis function
    """
    
    # Unpack degrees, total number of basis functions, knot vectors and boundary conditions in each direction
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc
    
    t1 = T1[1:-1]
    t2 = T2[1:-1]
    t3 = T3[1:-1]
    
    # Changes for periodic case 1 (reduction of number of basis functions and additional cycles in assembly)
    ca_1 = bc_1*(p1 - 1)
    ca_2 = bc_2*(p2 - 1)
    ca_3 = bc_3*(p3 - 1)
    
    # Element boundaries and number of elements
    el_b_1 = bsp.breakpoints(T1, p1)
    el_b_2 = bsp.breakpoints(T2, p2)
    el_b_3 = bsp.breakpoints(T3, p3)
    
    Nel_1 = len(el_b_1) - 1
    Nel_2 = len(el_b_2) - 1
    Nel_3 = len(el_b_3) - 1
    
    # Changes for periodic case 2 (periodicity in element cycles)
    cb_1 = (Nel_1 - (p1 - 1))*bc_1
    cb_2 = (Nel_2 - (p2 - 1))*bc_2
    cb_3 = (Nel_3 - (p3 - 1))*bc_3
    
    # Local and global quadrature points and weights (Gauss-Legendre)
    pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
    pts_2_loc, wts_2_loc = np.polynomial.legendre.leggauss(p2 + 1)
    pts_3_loc, wts_3_loc = np.polynomial.legendre.leggauss(p3 + 1)

    pts_1, wts_1 = bsp.quadrature_grid(el_b_1, pts_1_loc, wts_1_loc)
    pts_2, wts_2 = bsp.quadrature_grid(el_b_2, pts_2_loc, wts_2_loc)
    pts_3, wts_3 = bsp.quadrature_grid(el_b_3, pts_3_loc, wts_3_loc)
    
    # Evaluation of basis functions at global quadrature points 
    d = 0
    basis1_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t1, p1 - 1, pts_1, d, normalize=True))
    basis1_2 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t2, p2 - 1, pts_2, d, normalize=True))
    basis1_3 = np.asfortranarray(bsp.basis_ders_on_quad_grid(t3, p3 - 1, pts_3, d, normalize=True))
    
    # Create stencil vector space
    V3 = StencilVectorSpace([Nbase_1 - ca_1 - 1, Nbase_2 - ca_2 - 1, Nbase_3 - ca_3 - 1], [p1, p2, p3], [bc_1, bc_2, bc_3])
    
    # Create global stencil vector
    F3 = StencilVector(V3)

    # Create element matrices
    mat_m = np.zeros((p1, p2, p3), order='F')             # local vector
    mat_g = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') # Jacobi determinant
    mat_f = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') # function at quadrature points
    
    # Build global matrices: cycle over elements
    for ie1 in range(Nel_1 + ca_1):
        for ie2 in range(Nel_2 + ca_2):
            for ie3 in range(Nel_3 + ca_3):
                
                ie1_p = (ie1 + cb_1)%Nel_1
                ie2_p = (ie2 + cb_2)%Nel_2
                ie3_p = (ie3 + cb_3)%Nel_3
                
                # Get B-spline values and quadrature weights
                bs1_1 = basis1_1[ie1_p, :, 0, :]
                bs1_2 = basis1_2[ie2_p, :, 0, :]
                bs1_3 = basis1_3[ie3_p, :, 0, :]
                
                w1  = wts_1[ie1_p, :]
                w2  = wts_2[ie2_p, :]
                w3  = wts_3[ie3_p, :]
                
                # Evaluate Jacobi determinant at all quadrature points
                Pts1, Pts2, Pts3 = np.meshgrid(pts_1[ie1_p, :], pts_2[ie2_p, :], pts_3[ie3_p, :], indexing='ij')
                
                mat_g[:, :, :] = g_sqrt(Pts1, Pts2, Pts3)
                mat_f[:, :, :] = fun(Pts1, Pts2, Pts3)
                
                # Compute element vector
                kernels.kernel_L3(p1, p2, p3, p1 + 1, p2 + 1, p3 + 1, bs1_1, bs1_2, bs1_3, w1, w2, w3, mat_f, mat_g, mat_m)
                
                # Update global vector
                is1 = ie1 + p1 - ca_1
                is2 = ie2 + p2 - ca_2
                is3 = ie3 + p3 - ca_3
                
                F3[is1 - p1:is1, is2 - p2:is2, is3 - p3:is3] += mat_m[:, :, :]
                
                # IMPORTANT: ghost regions must be up-to-date
                F3.update_ghost_regions()
                
    return F3
#=============================================================================================================================



#============== L2-error in V0 in 3d =========================================================================================
def L2_error_V0(coeff, fun, p, Nbase, T, g_sqrt, bc):
    """
    Computes the L2 scalar product of the function 'fun' with the B-splines of the space V0 in general curvilinear coordinates
    q = (q1, q2, q3) with metric tensor G using a quadrature rule of order p + 1.
    
    Parameters
    ----------
    coeff : StencilVector of np.array
        the finite element coefficients of approximate field
    
    fun : callable
        the analytical function to which the error is computed
    
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    g_sqrt : callable
        square root of the Jacobi determinant
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = else)
        
    Returns
    -------
    error : double
        the L2-error
    """
    
    # Unpack degrees, total number of basis functions, knot vectors and boundary conditions in each direction
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc
    
    ca_1 = bc_1*p1
    ca_2 = bc_2*p2
    ca_3 = bc_3*p3
    
    # Bring coefficients in convenient order if not in stencil format
    mat_c = np.empty((Nbase_1 - ca_1, Nbase_2 - ca_2, Nbase_3 - ca_3), order='F')
    
    if isinstance(coeff, np.ndarray):
        mat_c[:, :, :] = np.reshape(coeff, (Nbase_1 - ca_1, Nbase_2 - ca_2, Nbase_3 - ca_3))
        
    else:
        mat_c[:, :, :] = coeff[:, :, :]
    
    
    # Element boundaries and number of elements
    el_b_1 = bsp.breakpoints(T1, p1)
    el_b_2 = bsp.breakpoints(T2, p2)
    el_b_3 = bsp.breakpoints(T3, p3)
    
    Nel_1 = len(el_b_1) - 1
    Nel_2 = len(el_b_2) - 1
    Nel_3 = len(el_b_3) - 1
    
    # Local and global quadrature points and weights (Gauss-Legendre)
    pts_1_loc, wts_1_loc = np.polynomial.legendre.leggauss(p1 + 1)
    pts_2_loc, wts_2_loc = np.polynomial.legendre.leggauss(p2 + 1)
    pts_3_loc, wts_3_loc = np.polynomial.legendre.leggauss(p3 + 1)

    pts_1, wts_1 = bsp.quadrature_grid(el_b_1, pts_1_loc, wts_1_loc)
    pts_2, wts_2 = bsp.quadrature_grid(el_b_2, pts_2_loc, wts_2_loc)
    pts_3, wts_3 = bsp.quadrature_grid(el_b_3, pts_3_loc, wts_3_loc)
    
    # Evaluation of basis functions at global quadrature points 
    d = 0
    basis0_1 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T1, p1, pts_1, d))
    basis0_2 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T2, p2, pts_2, d))
    basis0_3 = np.asfortranarray(bsp.basis_ders_on_quad_grid(T3, p3, pts_3, d))

    # Create element matrices
    mat_g = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') # Jacobi determinant at quadrature points
    mat_f = np.empty((p1 + 1, p2 + 1, p3 + 1), order='F') # function at quadrature points
    mat_c
    
    # global error
    error = np.array([0.])
    
    # Build global integral: cycle over elements
    for ie1 in range(Nel_1):
        for ie2 in range(Nel_2):
            for ie3 in range(Nel_3):
                
                # Get B-spline values and quadrature weights
                bs0_1 = basis0_1[ie1, :, 0, :]
                bs0_2 = basis0_2[ie2, :, 0, :]
                bs0_3 = basis0_3[ie3, :, 0, :]
                
                w1 = wts_1[ie1, :]
                w2 = wts_2[ie2, :]
                w3 = wts_3[ie3, :]
                  
                # Evaluate Jacobi determinant and function at all quadrature points
                Pts1, Pts2, Pts3 = np.meshgrid(pts_1[ie1, :], pts_2[ie2, :], pts_3[ie3, :], indexing='ij')
                
                mat_g[:, :, :] = g_sqrt(Pts1, Pts2, Pts3)
                mat_f[:, :, :] = fun(Pts1, Pts2, Pts3)
                
                # Compute element contribution and add to global error
                kernels.kernel_L2error_V0(ie1, ie2, ie3, Nbase_1 - ca_1, Nbase_2 - ca_2, Nbase_3 - ca_3, p1, p2, p3, p1 + 1, p2 + 1, p3 + 1, bs0_1, bs0_2, bs0_3, w1, w2, w3, mat_f, mat_g, mat_c, error)
                
                
    return error[0]
#=============================================================================================================================