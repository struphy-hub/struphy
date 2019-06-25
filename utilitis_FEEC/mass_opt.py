import numpy as np

from psydac.linalg.stencil import StencilMatrix, StencilVector
from psydac.core.interface import make_open_knots, make_periodic_knots

from pyccel import epyccel
import utilitis_FEEC.kernels_opt as kernels


kernels = epyccel(kernels)


def mass_matrix_V0_1d(V):
    
    # Sizes
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
    [p1] = V.vector_space.pads

    # Quadrature data
    [      nk1] = [g.num_elements for g in V.quad_grids]
    [      nq1] = [g.num_quad_pts for g in V.quad_grids]
    [  spans_1] = [g.spans        for g in V.quad_grids]
    [  basis_1] = [g.basis        for g in V.quad_grids]
    [ points_1] = [g.points       for g in V.quad_grids]
    [weights_1] = [g.weights      for g in V.quad_grids]
    
    
    # Create global matrices
    M0 = StencilMatrix(V.vector_space, V.vector_space)

    # Create element matrices
    mat_m = np.zeros((p1 + 1, 2*p1 + 1), order='F') # mass
    
    # Build global matrices: cycle over elements
    for k1 in range(nk1):
                
        # Get spline index, B-splines' values and quadrature weights
        is1 =   spans_1[k1]
        bs1 =   basis_1[k1, :, 0, :]
        w1  = weights_1[k1, :]

        # Compute element matrix
        kernels.kernel_V0_1d(p1, nq1, bs1, w1, mat_m)

        # Update global matrix
        M0[is1 - p1:is1 + 1, :] += mat_m[:, :]

        # Make sure that periodic corners are zero in non-periodic case
        M0.remove_spurious_entries()
                
    return M0


def mass_matrix_V1_1d(V):
    
    # Sizes
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
    [p1] = V.vector_space.pads

    # Quadrature data
    [      nk1] = [g.num_elements for g in V.quad_grids]
    [      nq1] = [g.num_quad_pts for g in V.quad_grids]
    [  spans_1] = [g.spans        for g in V.quad_grids]
    [  basis_1] = [g.basis        for g in V.quad_grids]
    [ points_1] = [g.points       for g in V.quad_grids]
    [weights_1] = [g.weights      for g in V.quad_grids]
    
    # Periodicity
    [per1] = V.periodic
    
    # Knot vector
    if per1 == True:
        t1 = make_periodic_knots(p1 + 1, e1 + 1 + p1 + 1)
        
    else:
        t1 = make_open_knots(p1 + 1, nk1 + p1 + 1)[1:-1]
    
    
    # Create global matrices
    M1 = StencilMatrix(V.vector_space, V.vector_space)

    # Create element matrices
    mat_m = np.zeros((p1 + 1, 2*p1 + 1), order='F') # mass
    
    # Build global matrices: cycle over elements
    for k1 in range(nk1):
                
        # Get spline index, B-splines' values and quadrature weights
        is1 =   spans_1[k1]
        bs1 =   basis_1[k1, :, 0, :]
        w1  = weights_1[k1, :]

        # Compute element matrix
        kernels.kernel_V1_1d(k1, p1, nq1, bs1, t1, w1, mat_m)

        # Update global matrix
        M1[is1 - p1:is1 + 1, :] += mat_m[:, :]

        # Make sure that periodic corners are zero in non-periodic case
        M1.remove_spurious_entries()
                
    return M1



def mass_matrix_V1_1d_opt(V):
    
    # Sizes
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
    [p1] = V.vector_space.pads
    

    # Quadrature data
    [      nk1] = [g.num_elements for g in V.quad_grids]
    [      nq1] = [g.num_quad_pts for g in V.quad_grids]
    [  spans_1] = [g.spans        for g in V.quad_grids]
    [  basis_1] = [g.basis        for g in V.quad_grids]
    [ points_1] = [g.points       for g in V.quad_grids]
    [weights_1] = [g.weights      for g in V.quad_grids]
    
    
    # Create global matrices
    M1 = StencilMatrix(V.vector_space, V.vector_space)

    # Create element matrices
    mat_m = np.zeros((p1 + 1, 2*p1 + 1), order='F') # mass
    
    # Build global matrices: cycle over elements
    for k1 in range(nk1):
                
        # Get spline index, B-splines' values and quadrature weights
        is1 =   spans_1[k1]
        bs1 =   basis_1[k1, :, 0, :]
        w1  = weights_1[k1, :]

        # Compute element matrix
        kernels.kernel_V0_1d_test(p1, nq1, bs1, w1, mat_m)

        # Update global matrix
        M1[is1 - p1:is1 + 1, :] += mat_m[:, :]

        # Make sure that periodic corners are zero in non-periodic case
        M1.remove_spurious_entries()
                
    return M1




def mass_matrix_V0(V, g_sqrt):
    
    # Sizes
    [s1, s2, s3] = V.vector_space.starts
    [e1, e2, e3] = V.vector_space.ends
    [p1, p2, p3] = V.vector_space.pads

    # Quadrature data
    [      nk1,       nk2,       nk3] = [g.num_elements for g in V.quad_grids]
    [      nq1,       nq2,       nq3] = [g.num_quad_pts for g in V.quad_grids]
    [  spans_1,   spans_2,   spans_3] = [g.spans        for g in V.quad_grids]
    [  basis_1,   basis_2,   basis_3] = [g.basis        for g in V.quad_grids]
    [ points_1,  points_2,  points_3] = [g.points       for g in V.quad_grids]
    [weights_1, weights_2, weights_3] = [g.weights      for g in V.quad_grids]
    
    
    # Create global matrices
    M0 = StencilMatrix(V.vector_space, V.vector_space)

    # Create element matrices
    mat_m = np.zeros((p1 + 1, p2 + 1, p3 + 1, 2*p1 + 1, 2*p2 + 1, 2*p3 + 1), order='F') # mass
    mat_g = np.empty((nq1, nq2, nq3), order='F')                                        # Jacobi determinant
    
    # Build global matrices: cycle over elements
    for k1 in range(nk1):
        for k2 in range(nk2):
            for k3 in range(nk3):
                
                # Get spline index, B-splines' values and quadrature weights
                is1 =   spans_1[k1]
                bs1 =   basis_1[k1, :, 0, :]
                w1  = weights_1[k1, :]

                is2 =   spans_2[k2]
                bs2 =   basis_2[k2, :, 0, :]
                w2  = weights_2[k2, :]
                
                is3 =   spans_3[k3]
                bs3 =   basis_3[k3, :, 0, :]
                w3  = weights_3[k3, :]
                
                # Evaluate Jacobi determinant at all quadrature points
                Pts1, Pts2, Pts3 = np.meshgrid(points_1[k1, :], points_2[k2, :], points_3[k3, :], indexing='ij')
                mat_g[:, :, :] = g_sqrt(Pts1, Pts2, Pts3)
                
                # Compute element matrix
                kernels.kernel_V0(p1, p2, p3, nq1, nq2, nq3, bs1, bs2, bs3, w1, w2, w3, mat_g, mat_m)
                
                # Update global matrix
                M0[is1 - p1:is1 + 1, is2 - p2:is2 + 1, is3 - p3:is3 + 1, :, :, :] += mat_m[:, :, :, :, :, :]
                
                # Make sure that periodic corners are zero in non-periodic case
                M0.remove_spurious_entries()
                
    return M0

























def mass_matrix_V0_opt(V, g_sqrt):
    
    # Sizes
    [s1, s2, s3] = V.vector_space.starts
    [e1, e2, e3] = V.vector_space.ends
    [p1, p2, p3] = V.vector_space.pads

    # Quadrature data
    [      nk1,       nk2,       nk3] = [g.num_elements for g in V.quad_grids]
    [      nq1,       nq2,       nq3] = [g.num_quad_pts for g in V.quad_grids]
    [  spans_1,   spans_2,   spans_3] = [g.spans        for g in V.quad_grids]
    [  basis_1,   basis_2,   basis_3] = [g.basis        for g in V.quad_grids]
    [ points_1,  points_2,  points_3] = [g.points       for g in V.quad_grids]
    [weights_1, weights_2, weights_3] = [g.weights      for g in V.quad_grids]
    
    
    # Create global matrices
    M0 = StencilMatrix(V.vector_space, V.vector_space)

    # Create element matrices
    mat_m = np.zeros((p1 + 1, p2 + 1, p3 + 1, 2*p1 + 1, 2*p2 + 1, 2*p3 + 1), order='F') # mass
    mat_g = np.empty((nq1, nq2, nq3), order='F')                                        # Jacobi determinant
    
    # Build global matrices: cycle over elements
    for k1 in range(s1, e1 + 1 - p1):
        for k2 in range(s2, e2 + 1 - p2):
            for k3 in range(s3, e3 + 1 - p3):
                
                # Get spline index, B-splines' values and quadrature weights
                is1 =   spans_1[k1]
                bs1 =   basis_1[k1, :, 0, :]
                w1  = weights_1[k1, :]

                is2 =   spans_2[k2]
                bs2 =   basis_2[k2, :, 0, :]
                w2  = weights_2[k2, :]
                
                is3 =   spans_3[k3]
                bs3 =   basis_3[k3, :, 0, :]
                w3  = weights_3[k3, :]
                
                # Evaluate Jacobi determinant at all quadrature points
                Pts1, Pts2, Pts3 = np.meshgrid(points_1[k1, :], points_2[k2, :], points_3[k3, :], indexing='ij')
                mat_g[:, :, :] = g_sqrt(Pts1, Pts2, Pts3)
                
                # Compute element matrix
                kernels.kernel_V0(p1, p2, p3, nq1, nq2, nq3, bs1, bs2, bs3, w1, w2, w3, mat_g, mat_m)
                
                # Update global matrix
                s1 = is1 - p1 - 1
                s2 = is2 - p2 - 1
                s3 = is3 - p3 - 1
                M0._data[s1:s1 + p1 + 1, s2:s2 + p2 + 1, s3:s3 + p3 + 1, :, :, :] += mat_m[:,:,:,:,:,:]
                
                # Make sure that periodic corners are zero in non-periodic case
                M0.remove_spurious_entries()
                
    return M0


def assembly_v3(V, g_sqrt):

    # ... sizes
    [s1, s2, s3] = V.vector_space.starts
    [e1, e2, e3] = V.vector_space.ends
    [p1, p2, p3] = V.vector_space.pads
    # ...

    # Quadrature data
    [      nk1,       nk2,       nk3] = [g.num_elements for g in V.quad_grids]
    [       k1,        k2,        k3] = [g.num_quad_pts for g in V.quad_grids]
    [  spans_1,   spans_2,   spans_3] = [g.spans        for g in V.quad_grids]
    [  basis_1,   basis_2,   basis_3] = [g.basis        for g in V.quad_grids]
    [ points_1,  points_2,  points_3] = [g.points       for g in V.quad_grids]
    [weights_1, weights_2, weights_3] = [g.weights      for g in V.quad_grids]

    # ... data structure
    M = StencilMatrix(V.vector_space, V.vector_space)
    # ...

    # ... element matrix
    mat = np.zeros((p1+1, p2+1, p3+1, 2*p1+1, 2*p2+1, 2*p3+1), order='F')
    mat_g = np.empty((k1, k2, k3), order='F')
    # ...

    # ... build matrices
    for ie1 in range(s1, e1+1-p1):
        for ie2 in range(s2, e2+1-p2):
            for ie3 in range(s3, e3+1-p3):
                is1 = spans_1[ie1]
                is2 = spans_2[ie2]
                is3 = spans_3[ie3]

                bs1 = basis_1[ie1, :, 0, :]
                bs2 = basis_2[ie2, :, 0, :]
                bs3 = basis_3[ie3, :, 0, :]
                w1 = weights_1[ie1, :]
                w2 = weights_2[ie2, :]
                w3 = weights_3[ie3, :]
                
                # Evaluate Jacobi determinant at all quadrature points
                Pts1, Pts2, Pts3 = np.meshgrid(points_1[ie1, :], points_2[ie2, :], points_3[ie3, :], indexing='ij')
                mat_g[:, :, :] = g_sqrt(Pts1, Pts2, Pts3)
                
                kernels.kernel_V0(p1, p2, p3, k1, k2, k3, bs1, bs2, bs3, w1, w2, w3, mat_g, mat)
                

                s1 = is1 - p1 - 1
                s2 = is2 - p2 - 1
                s3 = is3 - p3 - 1
                
                M._data[s1:s1+p1+1,s2:s2+p2+1,s3:s3+p3+1,:,:,:] += mat[:,:,:,:,:,:]
    # ...
    
    return M



def inner_prod_V0(V, g_sqrt, f):
    
    # Sizes
    [s1, s2, s3] = V.vector_space.starts
    [e1, e2, e3] = V.vector_space.ends
    [p1, p2, p3] = V.vector_space.pads

    # Quadrature data
    [      nk1,       nk2,       nk3] = [g.num_elements for g in V.quad_grids]
    [      nq1,       nq2,       nq3] = [g.num_quad_pts for g in V.quad_grids]
    [  spans_1,   spans_2,   spans_3] = [g.spans        for g in V.quad_grids]
    [  basis_1,   basis_2,   basis_3] = [g.basis        for g in V.quad_grids]
    [ points_1,  points_2,  points_3] = [g.points       for g in V.quad_grids]
    [weights_1, weights_2, weights_3] = [g.weights      for g in V.quad_grids]
    
    # Create global vector
    L2_0 = StencilVector(V.vector_space)
    
    # Create element matrices
    mat_m = np.zeros((p1 + 1, p2 + 1, p3 + 1), order='F') # mass
    mat_g = np.empty((nq1, nq2, nq3), order='F')          # Jacobi determinant at local quadrature points
    mat_f = np.empty((nq1, nq2, nq3), order='F')          # function at local quadrature points
    
    
    # Build global matrices: cycle over elements
    for k1 in range(nk1):
        for k2 in range(nk2):
            for k3 in range(nk3):
                
                # Get spline index, B-splines' values and quadrature weights
                is1 =   spans_1[k1]
                bs1 =   basis_1[k1, :, 0, :]
                w1  = weights_1[k1, :]

                is2 =   spans_2[k2]
                bs2 =   basis_2[k2, :, 0, :]
                w2  = weights_2[k2, :]
                
                is3 =   spans_3[k3]
                bs3 =   basis_3[k3, :, 0, :]
                w3  = weights_3[k3, :]
                
                # Evaluate Jacobi determinant and function at all quadrature points
                pts1, pts2, pts3 = np.meshgrid(points_1[k1, :], points_2[k2, :], points_3[k3, :], indexing='ij')
                
                mat_g[:, :, :] = g_sqrt(pts1, pts2, pts3)
                mat_f[:, :, :] = f(pts1, pts2, pts3)
                
                # Compute element matrix
                kernels.kernel_L0(p1, p2, p3, nq1, nq2, nq3, bs1, bs2, bs3, w1, w2, w3, mat_f, mat_g, mat_m)
                
                # Update global matrix
                L2_0[is1 - p1:is1 + 1, is2 - p2:is2 + 1, is3 - p3:is3 + 1] += mat_m[:, :, :]
                
                # IMPORTANT: ghost regions must be up-to-date
                L2_0.update_ghost_regions()
                
    return L2_0