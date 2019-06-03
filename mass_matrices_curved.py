import numpy as np
import scipy.sparse as sparse
import psydac.core.interface as inter
from pyccel import epyccel

import kernels


kernels = epyccel(kernels)



def mass_matrix_V1_xx(p, Nbase, T, Ginv, g, bc, pts, wts, basis0, basis1):
    """
    Computes the xx-block of the mass matrix of the space V1 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors in each direction
        
    Ginv : callable
        the xx-component of the inverse of the metric tensor G
        
    g : callable
        square root of the Jacobi determinant
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions) 
        
    pts : list of np.arrays
        quadrature points in each direction
        
    wts : list of np.arrays
        quadrature weights in each direction
        
    basis0 : list of np.arrays
        basis functions in H1 evaluated at quadrature points in each direction
        
    basis1 : list of np.arrays
        basis functions in L2 evaluated at quadrature points in each direction
    
    Returns
    -------
    M1_xx : sparse matrix
        xx-block of the mass matrix in V1
    """
    
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    pts_x, pts_y, pts_z = pts
    wts_x, wts_y, wts_z = wts
    basis0_x, basis0_y, basis0_z = basis0
    basis1_x, basis1_y, basis1_z = basis1
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    tz = Tz[1:-1]
        
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    Nel_x = len(el_b_x) - 1
    Nel_y = len(el_b_y) - 1
    Nel_z = len(el_b_z) - 1
    
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    GG = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                for gx in range(px + 1):
                    for gy in range(py + 1):
                        for gz in range(pz + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            GG[gx, gy, gz, nx, ny, nz] = Ginv(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
                            
    
    # ... global matrix and local element matrix
    M1_xx = np.zeros((Nbase_x  - 1, Nbase_y, Nbase_z, Nbase_x - 1, Nbase_y, Nbase_z))
    mat = np.zeros((px, py + 1, pz + 1, px, py + 1, pz + 1), order='F')
    # ...
     
    
    # ... assembly
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                
                wx = wts_x[:, nx]
                wy = wts_y[:, ny]
                wz = wts_z[:, nz]
                
                bs0_x = basis0_x[:, 0, :, nx]
                bs0_y = basis0_y[:, 0, :, ny]
                bs0_z = basis0_z[:, 0, :, nz]
                
                bs1_x = basis1_x[:, 0, :, nx]
                bs1_y = basis1_y[:, 0, :, ny]
                bs1_z = basis1_z[:, 0, :, nz]
                
                ggs = gg[:, :, :, nx, ny, nz]
                GGs = GG[:, :, :, nx, ny, nz]
                
                kernels.kernel1(nx, ny, nz, px, py, pz, 1, 0, 0, 1, 0, 0, bs1_x, bs0_y, bs0_z, bs1_x, bs0_y, bs0_z, px, px, tx, tx, wx, wy, wz, ggs, GGs, mat)
                
                M1_xx[nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1, nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1] += mat[:, :, :, :, :, :]
    # ...            
    
      
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        M1_xx[:px - 1, :, :, :, :, :] += M1_xx[-px + 1:, :, :, :, :, :]
        M1_xx[:, :, :, :px - 1, :, :] += M1_xx[:, :, :, -px + 1:, :, :]
        M1_xx = M1_xx[:M1_xx.shape[0] - px + 1, :, :, :M1_xx.shape[3] - px + 1, :, :]
        
        Nbase_x_0_i = Nbase_x - px
        Nbase_x_0_j = Nbase_x - px
        
    elif bc_x == False:
        Nbase_x_0_i = Nbase_x - 1
        Nbase_x_0_j = Nbase_x - 1
        
    else:
        Nbase_x_0_i = Nbase_x - 1
        Nbase_x_0_j = Nbase_x - 1
    # ...
      
        
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)
    if bc_y == True:
        M1_xx[:, :py, :, :, :, :] += M1_xx[:, -py:, :, :, :, :]
        M1_xx[:, :, :, :, :py, :] += M1_xx[:, :, :, :, -py:, :]
        M1_xx = M1_xx[:, :M1_xx.shape[1] - py, :, :, :M1_xx.shape[4] - py, :]
        
        Nbase_y_0_i = Nbase_y - py
        Nbase_y_0_j = Nbase_y - py
        
    elif bc_y == False:
        M1_xx = M1_xx[:, 1:-1, :, :, 1:-1, :]
        
        Nbase_y_0_i = Nbase_y - 2
        Nbase_y_0_j = Nbase_y - 2
        
    else:
        Nbase_y_0_i = Nbase_y
        Nbase_y_0_j = Nbase_y
    # ...
    
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)   
    if bc_z == True:
        M1_xx[:, :, :pz, :, :, :] += M1_xx[:, :, -pz:, :, :, :]
        M1_xx[:, :, :, :, :, :pz] += M1_xx[:, :, :, :, :, -pz:]
        M1_xx = M1_xx[:, :, :M1_xx.shape[2] - pz, :, :, :M1_xx.shape[5] - pz]
        
        Nbase_z_0_i = Nbase_z - pz
        Nbase_z_0_j = Nbase_z - pz
        
    elif bc_z == False:
        M1_xx = M1_xx[:, :, 1:-1, :, :, 1:-1]
        
        Nbase_z_0_i = Nbase_z - 2
        Nbase_z_0_j = Nbase_z - 2
        
    else:
        Nbase_z_0_i = Nbase_z
        Nbase_z_0_j = Nbase_z
    # ...
        
                    
    return sparse.csr_matrix(np.reshape(M1_xx, (Nbase_x_0_i*Nbase_y_0_i*Nbase_z_0_i, Nbase_x_0_j*Nbase_y_0_j*Nbase_z_0_j)))





def mass_matrix_V1_xy(p, Nbase, T, Ginv, g, bc, pts, wts, basis0, basis1):
    """
    Computes the xy-block of the mass matrix of the space V1 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors in each direction
        
    Ginv : callable
        the xx-component of the inverse of the metric tensor G
        
    g : callable
        square root of the Jacobi determinant
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions) 
        
    pts : list of np.arrays
        quadrature points in each direction
        
    wts : list of np.arrays
        quadrature weights in each direction
        
    basis0 : list of np.arrays
        basis functions in H1 evaluated at quadrature points in each direction
        
    basis1 : list of np.arrays
        basis functions in L2 evaluated at quadrature points in each direction
    
    Returns
    -------
    M1_xy : sparse matrix
        xy-block of the mass matrix in V1
    """
    
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    pts_x, pts_y, pts_z = pts
    wts_x, wts_y, wts_z = wts
    basis0_x, basis0_y, basis0_z = basis0
    basis1_x, basis1_y, basis1_z = basis1
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    tz = Tz[1:-1]
        
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    Nel_x = len(el_b_x) - 1
    Nel_y = len(el_b_y) - 1
    Nel_z = len(el_b_z) - 1
    
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    GG = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                for gx in range(px + 1):
                    for gy in range(py + 1):
                        for gz in range(pz + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            GG[gx, gy, gz, nx, ny, nz] = Ginv(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
                            
    
     
    # ... global matrix and local element matrix
    M1_xy = np.zeros((Nbase_x  - 1, Nbase_y, Nbase_z, Nbase_x, Nbase_y - 1, Nbase_z))
    mat = np.zeros((px, py + 1, pz + 1, px + 1, py, pz + 1), order='F')
    # ...
         
        
    # ... assembly
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                
                wx = wts_x[:, nx]
                wy = wts_y[:, ny]
                wz = wts_z[:, nz]
                
                bs0_x = basis0_x[:, 0, :, nx]
                bs0_y = basis0_y[:, 0, :, ny]
                bs0_z = basis0_z[:, 0, :, nz]
                
                bs1_x = basis1_x[:, 0, :, nx]
                bs1_y = basis1_y[:, 0, :, ny]
                bs1_z = basis1_z[:, 0, :, nz]
                
                ggs = gg[:, :, :, nx, ny, nz]
                GGs = GG[:, :, :, nx, ny, nz]
                
                kernels.kernel1(nx, ny, nz, px, py, pz, 1, 0, 0, 0, 1, 0, bs1_x, bs0_y, bs0_z, bs0_x, bs1_y, bs0_z, px, py, tx, ty, wx, wy, wz, ggs, GGs, mat)
                
                M1_xy[nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1, nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1] += mat[:, :, :, :, :, :]
    # ...
        
        
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        M1_xy[:px - 1, :, :, :, :, :] += M1_xy[-px + 1:, :, :, :, :, :]
        M1_xy[:, :, :, :px, :, :] += M1_xy[:, :, :, -px:, :, :]
        M1_xy = M1_xy[:M1_xy.shape[0] - px + 1, :, :, :M1_xy.shape[3] - px, :, :]
        
        Nbase_x_0_i = Nbase_x - px
        Nbase_x_0_j = Nbase_x - px
        
    elif bc_x == False:
        M1_xy = M1_xy[:, :, :, 1:-1, :, :]
        
        Nbase_x_0_i = Nbase_x - 1
        Nbase_x_0_j = Nbase_x - 2
        
    else:
        Nbase_x_0_i = Nbase_x - 1
        Nbase_x_0_j = Nbase_x
    # ... 
        
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)    
    if bc_y == True:
        M1_xy[:, :py, :, :, :, :] += M1_xy[:, -py:, :, :, :, :]
        M1_xy[:, :, :, :, :py - 1, :] += M1_xy[:, :, :, :, -py + 1:, :]
        M1_xy = M1_xy[:, :M1_xy.shape[1] - py, :, :, :M1_xy.shape[4] - py + 1, :]
        
        Nbase_y_0_i = Nbase_y - py
        Nbase_y_0_j = Nbase_y - py
        
    elif bc_y == False:
        M1_xy = M1_xy[:, 1:-1, :, :, :, :]
        
        Nbase_y_0_i = Nbase_y - 2
        Nbase_y_0_j = Nbase_y - 1
        
    else:
        Nbase_y_0_i = Nbase_y
        Nbase_y_0_j = Nbase_y - 1
    # ...
    
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)        
    if bc_z == True:
        M1_xy[:, :, :pz, :, :, :] += M1_xy[:, :, -pz:, :, :, :]
        M1_xy[:, :, :, :, :, :pz] += M1_xy[:, :, :, :, :, -pz:]
        M1_xy = M1_xy[:, :, :M1_xy.shape[2] - pz, :, :, :M1_xy.shape[5] - pz]
        
        Nbase_z_0_i = Nbase_z - pz
        Nbase_z_0_j = Nbase_z - pz
        
    elif bc_z == False:
        M1_xy = M1_xy[:, :, 1:-1, :, :, 1:-1]
        
        Nbase_z_0_i = Nbase_z - 2
        Nbase_z_0_j = Nbase_z - 2
        
    else:
        Nbase_z_0_i = Nbase_z
        Nbase_z_0_j = Nbase_z
        
                    
    return sparse.csr_matrix(np.reshape(M1_xy, (Nbase_x_0_i*Nbase_y_0_i*Nbase_z_0_i, Nbase_x_0_j*Nbase_y_0_j*Nbase_z_0_j)))





def mass_matrix_V1_xz(p, Nbase, T, Ginv, g, bc, pts, wts, basis0, basis1):
    """
    Computes the xz-block of the mass matrix of the space V1 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors in each direction
        
    Ginv : callable
        the xx-component of the inverse of the metric tensor G
        
    g : callable
        square root of the Jacobi determinant
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions) 
        
    pts : list of np.arrays
        quadrature points in each direction
        
    wts : list of np.arrays
        quadrature weights in each direction
        
    basis0 : list of np.arrays
        basis functions in H1 evaluated at quadrature points in each direction
        
    basis1 : list of np.arrays
        basis functions in L2 evaluated at quadrature points in each direction
    
    Returns
    -------
    M1_xz : sparse matrix
        xz-block of the mass matrix in V1
    """
    
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    pts_x, pts_y, pts_z = pts
    wts_x, wts_y, wts_z = wts
    basis0_x, basis0_y, basis0_z = basis0
    basis1_x, basis1_y, basis1_z = basis1
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    tz = Tz[1:-1]
        
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    Nel_x = len(el_b_x) - 1
    Nel_y = len(el_b_y) - 1
    Nel_z = len(el_b_z) - 1
    
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    GG = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                for gx in range(px + 1):
                    for gy in range(py + 1):
                        for gz in range(pz + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            GG[gx, gy, gz, nx, ny, nz] = Ginv(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
                            
    
     
    # ... global matrix and local element matrix
    M1_xz = np.zeros((Nbase_x  - 1, Nbase_y, Nbase_z, Nbase_x, Nbase_y, Nbase_z - 1))
    mat = np.zeros((px, py + 1, pz + 1, px + 1, py + 1, pz), order='F')
    # ...
         
        
    # ... assembly
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                
                wx = wts_x[:, nx]
                wy = wts_y[:, ny]
                wz = wts_z[:, nz]
                
                bs0_x = basis0_x[:, 0, :, nx]
                bs0_y = basis0_y[:, 0, :, ny]
                bs0_z = basis0_z[:, 0, :, nz]
                
                bs1_x = basis1_x[:, 0, :, nx]
                bs1_y = basis1_y[:, 0, :, ny]
                bs1_z = basis1_z[:, 0, :, nz]
                
                ggs = gg[:, :, :, nx, ny, nz]
                GGs = GG[:, :, :, nx, ny, nz]
                
                kernels.kernel1(nx, ny, nz, px, py, pz, 1, 0, 0, 0, 0, 1, bs1_x, bs0_y, bs0_z, bs0_x, bs0_y, bs1_z, px, pz, tx, tz, wx, wy, wz, ggs, GGs, mat)
                
                M1_xz[nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1, nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz] += mat[:, :, :, :, :, :]
    # ...
        
        
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        M1_xz[:px - 1, :, :, :, :, :] += M1_xz[-px + 1:, :, :, :, :, :]
        M1_xz[:, :, :, :px, :, :] += M1_xz[:, :, :, -px:, :, :]
        M1_xz = M1_xz[:M1_xz.shape[0] - px + 1, :, :, :M1_xz.shape[3] - px, :, :]
        
        Nbase_x_0_i = Nbase_x - px
        Nbase_x_0_j = Nbase_x - px
        
    elif bc_x == False:
        M1_xz = M1_xz[:, :, :, 1:-1, :, :]
        
        Nbase_x_0_i = Nbase_x - 1
        Nbase_x_0_j = Nbase_x - 2
        
    else:
        Nbase_x_0_i = Nbase_x - 1
        Nbase_x_0_j = Nbase_x
    # ... 
        
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)    
    if bc_y == True:
        M1_xz[:, :py, :, :, :, :] += M1_xz[:, -py:, :, :, :, :]
        M1_xz[:, :, :, :, :py, :] += M1_xz[:, :, :, :, -py:, :]
        M1_xz = M1_xz[:, :M1_xz.shape[1] - py, :, :, :M1_xz.shape[4] - py, :]
        
        Nbase_y_0_i = Nbase_y - py
        Nbase_y_0_j = Nbase_y - py
        
    elif bc_y == False:
        M1_xz = M1_xz[:, 1:-1, :, :, 1:-1, :]
        
        Nbase_y_0_i = Nbase_y - 2
        Nbase_y_0_j = Nbase_y - 2
        
    else:
        Nbase_y_0_i = Nbase_y
        Nbase_y_0_j = Nbase_y 
    # ...
    
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)        
    if bc_z == True:
        M1_xz[:, :, :pz, :, :, :] += M1_xz[:, :, -pz:, :, :, :]
        M1_xz[:, :, :, :, :, :pz - 1] += M1_xz[:, :, :, :, :, -pz + 1:]
        M1_xz = M1_xz[:, :, :M1_xz.shape[2] - pz, :, :, :M1_xz.shape[5] - pz + 1]
        
        Nbase_z_0_i = Nbase_z - pz
        Nbase_z_0_j = Nbase_z - pz
        
    elif bc_z == False:
        M1_xz = M1_xz[:, :, 1:-1, :, :, :]
        
        Nbase_z_0_i = Nbase_z - 2
        Nbase_z_0_j = Nbase_z - 1
        
    else:
        Nbase_z_0_i = Nbase_z
        Nbase_z_0_j = Nbase_z - 1
        
                    
    return sparse.csr_matrix(np.reshape(M1_xz, (Nbase_x_0_i*Nbase_y_0_i*Nbase_z_0_i, Nbase_x_0_j*Nbase_y_0_j*Nbase_z_0_j)))





def mass_matrix_V1_yx(p, Nbase, T, Ginv, g, bc, pts, wts, basis0, basis1):
    """
    Computes the yx-block of the mass matrix of the space V1 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors in each direction
        
    Ginv : callable
        the xx-component of the inverse of the metric tensor G
        
    g : callable
        square root of the Jacobi determinant
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions) 
        
    pts : list of np.arrays
        quadrature points in each direction
        
    wts : list of np.arrays
        quadrature weights in each direction
        
    basis0 : list of np.arrays
        basis functions in H1 evaluated at quadrature points in each direction
        
    basis1 : list of np.arrays
        basis functions in L2 evaluated at quadrature points in each direction
    
    Returns
    -------
    M1_yx : sparse matrix
        yx-block of the mass matrix in V1
    """
    
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    pts_x, pts_y, pts_z = pts
    wts_x, wts_y, wts_z = wts
    basis0_x, basis0_y, basis0_z = basis0
    basis1_x, basis1_y, basis1_z = basis1
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    tz = Tz[1:-1]
        
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    Nel_x = len(el_b_x) - 1
    Nel_y = len(el_b_y) - 1
    Nel_z = len(el_b_z) - 1
    
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    GG = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                for gx in range(px + 1):
                    for gy in range(py + 1):
                        for gz in range(pz + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            GG[gx, gy, gz, nx, ny, nz] = Ginv(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
                            
    
     
    # ... global matrix and local element matrix
    M1_yx = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z, Nbase_x - 1, Nbase_y, Nbase_z))
    mat = np.zeros((px + 1, py, pz + 1, px, py + 1, pz + 1), order='F')
    # ...
         
        
    # ... assembly
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                
                wx = wts_x[:, nx]
                wy = wts_y[:, ny]
                wz = wts_z[:, nz]
                
                bs0_x = basis0_x[:, 0, :, nx]
                bs0_y = basis0_y[:, 0, :, ny]
                bs0_z = basis0_z[:, 0, :, nz]
                
                bs1_x = basis1_x[:, 0, :, nx]
                bs1_y = basis1_y[:, 0, :, ny]
                bs1_z = basis1_z[:, 0, :, nz]
                
                ggs = gg[:, :, :, nx, ny, nz]
                GGs = GG[:, :, :, nx, ny, nz]
                
                kernels.kernel1(nx, ny, nz, px, py, pz, 0, 1, 0, 1, 0, 0, bs0_x, bs1_y, bs0_z, bs1_x, bs0_y, bs0_z, py, px, ty, tx, wx, wy, wz, ggs, GGs, mat)
                
                M1_yx[nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1, nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1] += mat[:, :, :, :, :, :]
    # ...
        
        
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        M1_yx[:px, :, :, :, :, :] += M1_yx[-px:, :, :, :, :, :]
        M1_yx[:, :, :, :px - 1, :, :] += M1_yx[:, :, :, -px + 1:, :, :]
        M1_yx = M1_yx[:M1_yx.shape[0] - px, :, :, :M1_yx.shape[3] - px + 1, :, :]
        
        Nbase_x_0_i = Nbase_x - px
        Nbase_x_0_j = Nbase_x - px
        
    elif bc_x == False:
        M1_yx = M1_yx[1:-1, :, :, :, :, :]
        
        Nbase_x_0_i = Nbase_x - 2
        Nbase_x_0_j = Nbase_x - 1
        
    else:
        Nbase_x_0_i = Nbase_x
        Nbase_x_0_j = Nbase_x - 1
    # ... 
        
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)    
    if bc_y == True:
        M1_yx[:, :py - 1, :, :, :, :] += M1_yx[:, -py + 1:, :, :, :, :]
        M1_yx[:, :, :, :, :py, :] += M1_yx[:, :, :, :, -py:, :]
        M1_yx = M1_yx[:, :M1_yx.shape[1] - py + 1, :, :, :M1_yx.shape[4] - py, :]
        
        Nbase_y_0_i = Nbase_y - py
        Nbase_y_0_j = Nbase_y - py
        
    elif bc_y == False:
        M1_yx = M1_yx[:, :, :, :, 1:-1, :]
        
        Nbase_y_0_i = Nbase_y - 1
        Nbase_y_0_j = Nbase_y - 2
        
    else:
        Nbase_y_0_i = Nbase_y - 1
        Nbase_y_0_j = Nbase_y 
    # ...
    
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)        
    if bc_z == True:
        M1_yx[:, :, :pz, :, :, :] += M1_yx[:, :, -pz:, :, :, :]
        M1_yx[:, :, :, :, :, :pz] += M1_yx[:, :, :, :, :, -pz:]
        M1_yx = M1_yx[:, :, :M1_yx.shape[2] - pz, :, :, :M1_yx.shape[5] - pz]
        
        Nbase_z_0_i = Nbase_z - pz
        Nbase_z_0_j = Nbase_z - pz
        
    elif bc_z == False:
        M1_yx = M1_yx[:, :, 1:-1, :, :, 1:-1]
        
        Nbase_z_0_i = Nbase_z - 2
        Nbase_z_0_j = Nbase_z - 2
        
    else:
        Nbase_z_0_i = Nbase_z
        Nbase_z_0_j = Nbase_z
        
                    
    return sparse.csr_matrix(np.reshape(M1_yx, (Nbase_x_0_i*Nbase_y_0_i*Nbase_z_0_i, Nbase_x_0_j*Nbase_y_0_j*Nbase_z_0_j)))




def mass_matrix_V1_yy(p, Nbase, T, Ginv, g, bc, pts, wts, basis0, basis1):
    """
    Computes the yy-block of the mass matrix of the space V1 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors in each direction
        
    Ginv : callable
        the xx-component of the inverse of the metric tensor G
        
    g : callable
        square root of the Jacobi determinant
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions) 
        
    pts : list of np.arrays
        quadrature points in each direction
        
    wts : list of np.arrays
        quadrature weights in each direction
        
    basis0 : list of np.arrays
        basis functions in H1 evaluated at quadrature points in each direction
        
    basis1 : list of np.arrays
        basis functions in L2 evaluated at quadrature points in each direction
    
    Returns
    -------
    M1_yy : sparse matrix
        yy-block of the mass matrix in V1
    """
    
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    pts_x, pts_y, pts_z = pts
    wts_x, wts_y, wts_z = wts
    basis0_x, basis0_y, basis0_z = basis0
    basis1_x, basis1_y, basis1_z = basis1
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    tz = Tz[1:-1]
        
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    Nel_x = len(el_b_x) - 1
    Nel_y = len(el_b_y) - 1
    Nel_z = len(el_b_z) - 1
    
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    GG = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                for gx in range(px + 1):
                    for gy in range(py + 1):
                        for gz in range(pz + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            GG[gx, gy, gz, nx, ny, nz] = Ginv(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
                            
    
     
    # ... global matrix and local element matrix
    M1_yy = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z, Nbase_x, Nbase_y - 1, Nbase_z))
    mat = np.zeros((px + 1, py, pz + 1, px + 1, py, pz + 1), order='F')
    # ...
         
        
    # ... assembly
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                
                wx = wts_x[:, nx]
                wy = wts_y[:, ny]
                wz = wts_z[:, nz]
                
                bs0_x = basis0_x[:, 0, :, nx]
                bs0_y = basis0_y[:, 0, :, ny]
                bs0_z = basis0_z[:, 0, :, nz]
                
                bs1_x = basis1_x[:, 0, :, nx]
                bs1_y = basis1_y[:, 0, :, ny]
                bs1_z = basis1_z[:, 0, :, nz]
                
                ggs = gg[:, :, :, nx, ny, nz]
                GGs = GG[:, :, :, nx, ny, nz]
                
                kernels.kernel1(nx, ny, nz, px, py, pz, 0, 1, 0, 0, 1, 0, bs0_x, bs1_y, bs0_z, bs0_x, bs1_y, bs0_z, py, py, ty, ty, wx, wy, wz, ggs, GGs, mat)
                
                M1_yy[nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1, nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1] += mat[:, :, :, :, :, :]
    # ...
        
        
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        M1_yy[:px, :, :, :, :, :] += M1_yy[-px:, :, :, :, :, :]
        M1_yy[:, :, :, :px, :, :] += M1_yy[:, :, :, -px:, :, :]
        M1_yy = M1_yy[:M1_yy.shape[0] - px, :, :, :M1_yy.shape[3] - px, :, :]
        
        Nbase_x_0_i = Nbase_x - px
        Nbase_x_0_j = Nbase_x - px
        
    elif bc_x == False:
        M1_yy = M1_yy[1:-1, :, :, 1:-1, :, :]
        
        Nbase_x_0_i = Nbase_x - 2
        Nbase_x_0_j = Nbase_x - 2
        
    else:
        Nbase_x_0_i = Nbase_x
        Nbase_x_0_j = Nbase_x
    # ... 
        
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)    
    if bc_y == True:
        M1_yy[:, :py - 1, :, :, :, :] += M1_yy[:, -py + 1:, :, :, :, :]
        M1_yy[:, :, :, :, :py - 1, :] += M1_yy[:, :, :, :, -py + 1:, :]
        M1_yy = M1_yy[:, :M1_yy.shape[1] - py + 1, :, :, :M1_yy.shape[4] - py + 1, :]
        
        Nbase_y_0_i = Nbase_y - py
        Nbase_y_0_j = Nbase_y - py
        
    elif bc_y == False:
        Nbase_y_0_i = Nbase_y - 1
        Nbase_y_0_j = Nbase_y - 1
        
    else:
        Nbase_y_0_i = Nbase_y - 1
        Nbase_y_0_j = Nbase_y - 1
    # ...
    
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)        
    if bc_z == True:
        M1_yy[:, :, :pz, :, :, :] += M1_yy[:, :, -pz:, :, :, :]
        M1_yy[:, :, :, :, :, :pz] += M1_yy[:, :, :, :, :, -pz:]
        M1_yy = M1_yy[:, :, :M1_yy.shape[2] - pz, :, :, :M1_yy.shape[5] - pz]
        
        Nbase_z_0_i = Nbase_z - pz
        Nbase_z_0_j = Nbase_z - pz
        
    elif bc_z == False:
        M1_yy = M1_yy[:, :, 1:-1, :, :, 1:-1]
        
        Nbase_z_0_i = Nbase_z - 2
        Nbase_z_0_j = Nbase_z - 2
        
    else:
        Nbase_z_0_i = Nbase_z
        Nbase_z_0_j = Nbase_z
        
                    
    return sparse.csr_matrix(np.reshape(M1_yy, (Nbase_x_0_i*Nbase_y_0_i*Nbase_z_0_i, Nbase_x_0_j*Nbase_y_0_j*Nbase_z_0_j)))





def mass_matrix_V1_yz(p, Nbase, T, Ginv, g, bc, pts, wts, basis0, basis1):
    """
    Computes the yx-block of the mass matrix of the space V1 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors in each direction
        
    Ginv : callable
        the xx-component of the inverse of the metric tensor G
        
    g : callable
        square root of the Jacobi determinant
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions) 
        
    pts : list of np.arrays
        quadrature points in each direction
        
    wts : list of np.arrays
        quadrature weights in each direction
        
    basis0 : list of np.arrays
        basis functions in H1 evaluated at quadrature points in each direction
        
    basis1 : list of np.arrays
        basis functions in L2 evaluated at quadrature points in each direction
    
    Returns
    -------
    M1_yz : sparse matrix
        yz-block of the mass matrix in V1
    """
    
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    pts_x, pts_y, pts_z = pts
    wts_x, wts_y, wts_z = wts
    basis0_x, basis0_y, basis0_z = basis0
    basis1_x, basis1_y, basis1_z = basis1
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    tz = Tz[1:-1]
        
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    Nel_x = len(el_b_x) - 1
    Nel_y = len(el_b_y) - 1
    Nel_z = len(el_b_z) - 1
    
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    GG = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                for gx in range(px + 1):
                    for gy in range(py + 1):
                        for gz in range(pz + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            GG[gx, gy, gz, nx, ny, nz] = Ginv(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
                            
    
     
    # ... global matrix and local element matrix
    M1_yz = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z, Nbase_x, Nbase_y, Nbase_z - 1))
    mat = np.zeros((px + 1, py, pz + 1, px + 1, py + 1, pz), order='F')
    # ...
         
        
    # ... assembly
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                
                wx = wts_x[:, nx]
                wy = wts_y[:, ny]
                wz = wts_z[:, nz]
                
                bs0_x = basis0_x[:, 0, :, nx]
                bs0_y = basis0_y[:, 0, :, ny]
                bs0_z = basis0_z[:, 0, :, nz]
                
                bs1_x = basis1_x[:, 0, :, nx]
                bs1_y = basis1_y[:, 0, :, ny]
                bs1_z = basis1_z[:, 0, :, nz]
                
                ggs = gg[:, :, :, nx, ny, nz]
                GGs = GG[:, :, :, nx, ny, nz]
                
                kernels.kernel1(nx, ny, nz, px, py, pz, 0, 1, 0, 0, 0, 1, bs0_x, bs1_y, bs0_z, bs0_x, bs0_y, bs1_z, py, pz, ty, tz, wx, wy, wz, ggs, GGs, mat)
                
                M1_yz[nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1, nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz] += mat[:, :, :, :, :, :]
    # ...
        
        
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        M1_yz[:px, :, :, :, :, :] += M1_yz[-px:, :, :, :, :, :]
        M1_yz[:, :, :, :px, :, :] += M1_yz[:, :, :, -px:, :, :]
        M1_yz = M1_yz[:M1_yz.shape[0] - px, :, :, :M1_yz.shape[3] - px, :, :]
        
        Nbase_x_0_i = Nbase_x - px
        Nbase_x_0_j = Nbase_x - px
        
    elif bc_x == False:
        M1_yz = M1_yz[1:-1, :, :, 1:-1, :, :]
        
        Nbase_x_0_i = Nbase_x - 2
        Nbase_x_0_j = Nbase_x - 2
        
    else:
        Nbase_x_0_i = Nbase_x
        Nbase_x_0_j = Nbase_x
    # ... 
        
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)    
    if bc_y == True:
        M1_yz[:, :py - 1, :, :, :, :] += M1_yz[:, -py + 1:, :, :, :, :]
        M1_yz[:, :, :, :, :py, :] += M1_yz[:, :, :, :, -py:, :]
        M1_yz = M1_yz[:, :M1_yz.shape[1] - py + 1, :, :, :M1_yz.shape[4] - py, :]
        
        Nbase_y_0_i = Nbase_y - py
        Nbase_y_0_j = Nbase_y - py
        
    elif bc_y == False:
        M1_yz = M1_yz[:, :, :, :, 1:-1, :]
        
        Nbase_y_0_i = Nbase_y - 1
        Nbase_y_0_j = Nbase_y - 2
        
    else:
        Nbase_y_0_i = Nbase_y - 1
        Nbase_y_0_j = Nbase_y
    # ...
    
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)        
    if bc_z == True:
        M1_yz[:, :, :pz, :, :, :] += M1_yz[:, :, -pz:, :, :, :]
        M1_yz[:, :, :, :, :, :pz - 1] += M1_yz[:, :, :, :, :, -pz + 1:]
        M1_yz = M1_yz[:, :, :M1_yz.shape[2] - pz, :, :, :M1_yz.shape[5] - pz + 1]
        
        Nbase_z_0_i = Nbase_z - pz
        Nbase_z_0_j = Nbase_z - pz
        
    elif bc_z == False:
        M1_yz = M1_yz[:, :, 1:-1, :, :, :]
        
        Nbase_z_0_i = Nbase_z - 2
        Nbase_z_0_j = Nbase_z - 1
        
    else:
        Nbase_z_0_i = Nbase_z
        Nbase_z_0_j = Nbase_z - 1
        
                    
    return sparse.csr_matrix(np.reshape(M1_yz, (Nbase_x_0_i*Nbase_y_0_i*Nbase_z_0_i, Nbase_x_0_j*Nbase_y_0_j*Nbase_z_0_j)))





def mass_matrix_V1_zx(p, Nbase, T, Ginv, g, bc, pts, wts, basis0, basis1):
    """
    Computes the zx-block of the mass matrix of the space V1 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors in each direction
        
    Ginv : callable
        the xx-component of the inverse of the metric tensor G
        
    g : callable
        square root of the Jacobi determinant
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions) 
        
    pts : list of np.arrays
        quadrature points in each direction
        
    wts : list of np.arrays
        quadrature weights in each direction
        
    basis0 : list of np.arrays
        basis functions in H1 evaluated at quadrature points in each direction
        
    basis1 : list of np.arrays
        basis functions in L2 evaluated at quadrature points in each direction
    
    Returns
    -------
    M1_zx : sparse matrix
        zx-block of the mass matrix in V1
    """
    
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    pts_x, pts_y, pts_z = pts
    wts_x, wts_y, wts_z = wts
    basis0_x, basis0_y, basis0_z = basis0
    basis1_x, basis1_y, basis1_z = basis1
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    tz = Tz[1:-1]
        
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    Nel_x = len(el_b_x) - 1
    Nel_y = len(el_b_y) - 1
    Nel_z = len(el_b_z) - 1
    
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    GG = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                for gx in range(px + 1):
                    for gy in range(py + 1):
                        for gz in range(pz + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            GG[gx, gy, gz, nx, ny, nz] = Ginv(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
                            
    
     
    # ... global matrix and local element matrix
    M1_zx = np.zeros((Nbase_x, Nbase_y, Nbase_z - 1, Nbase_x - 1, Nbase_y, Nbase_z))
    mat = np.zeros((px + 1, py + 1, pz, px, py + 1, pz + 1), order='F')
    # ...
         
        
    # ... assembly
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                
                wx = wts_x[:, nx]
                wy = wts_y[:, ny]
                wz = wts_z[:, nz]
                
                bs0_x = basis0_x[:, 0, :, nx]
                bs0_y = basis0_y[:, 0, :, ny]
                bs0_z = basis0_z[:, 0, :, nz]
                
                bs1_x = basis1_x[:, 0, :, nx]
                bs1_y = basis1_y[:, 0, :, ny]
                bs1_z = basis1_z[:, 0, :, nz]
                
                ggs = gg[:, :, :, nx, ny, nz]
                GGs = GG[:, :, :, nx, ny, nz]
                
                kernels.kernel1(nx, ny, nz, px, py, pz, 0, 0, 1, 1, 0, 0, bs0_x, bs0_y, bs1_z, bs1_x, bs0_y, bs0_z, pz, px, tz, tx, wx, wy, wz, ggs, GGs, mat)
                
                M1_zx[nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz, nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1] += mat[:, :, :, :, :, :]
    # ...
        
        
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        M1_zx[:px, :, :, :, :, :] += M1_zx[-px:, :, :, :, :, :]
        M1_zx[:, :, :, :px - 1, :, :] += M1_zx[:, :, :, -px + 1:, :, :]
        M1_zx = M1_zx[:M1_zx.shape[0] - px, :, :, :M1_zx.shape[3] - px + 1, :, :]
        
        Nbase_x_0_i = Nbase_x - px
        Nbase_x_0_j = Nbase_x - px
        
    elif bc_x == False:
        M1_zx = M1_zx[1:-1, :, :, :, :, :]
        
        Nbase_x_0_i = Nbase_x - 2
        Nbase_x_0_j = Nbase_x - 1
        
    else:
        Nbase_x_0_i = Nbase_x
        Nbase_x_0_j = Nbase_x - 1
    # ... 
        
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)    
    if bc_y == True:
        M1_zx[:, :py, :, :, :, :] += M1_zx[:, -py:, :, :, :, :]
        M1_zx[:, :, :, :, :py, :] += M1_zx[:, :, :, :, -py:, :]
        M1_zx = M1_zx[:, :M1_zx.shape[1] - py, :, :, :M1_zx.shape[4] - py, :]
        
        Nbase_y_0_i = Nbase_y - py
        Nbase_y_0_j = Nbase_y - py
        
    elif bc_y == False:
        M1_zx = M1_zx[:, 1:-1, :, :, 1:-1, :]
        
        Nbase_y_0_i = Nbase_y - 2
        Nbase_y_0_j = Nbase_y - 2
        
    else:
        Nbase_y_0_i = Nbase_y
        Nbase_y_0_j = Nbase_y
    # ...
    
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)        
    if bc_z == True:
        M1_zx[:, :, :pz - 1, :, :, :] += M1_zx[:, :, -pz + 1:, :, :, :]
        M1_zx[:, :, :, :, :, :pz] += M1_zx[:, :, :, :, :, -pz:]
        M1_zx = M1_zx[:, :, :M1_zx.shape[2] - pz + 1, :, :, :M1_zx.shape[5] - pz]
        
        Nbase_z_0_i = Nbase_z - pz
        Nbase_z_0_j = Nbase_z - pz
        
    elif bc_z == False:
        M1_zx = M1_zx[:, :, :, :, :, 1:-1]
        
        Nbase_z_0_i = Nbase_z - 1
        Nbase_z_0_j = Nbase_z - 2
        
    else:
        Nbase_z_0_i = Nbase_z - 1
        Nbase_z_0_j = Nbase_z
        
                    
    return sparse.csr_matrix(np.reshape(M1_zx, (Nbase_x_0_i*Nbase_y_0_i*Nbase_z_0_i, Nbase_x_0_j*Nbase_y_0_j*Nbase_z_0_j)))





def mass_matrix_V1_zy(p, Nbase, T, Ginv, g, bc, pts, wts, basis0, basis1):
    """
    Computes the zy-block of the mass matrix of the space V1 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors in each direction
        
    Ginv : callable
        the xx-component of the inverse of the metric tensor G
        
    g : callable
        square root of the Jacobi determinant
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions) 
        
    pts : list of np.arrays
        quadrature points in each direction
        
    wts : list of np.arrays
        quadrature weights in each direction
        
    basis0 : list of np.arrays
        basis functions in H1 evaluated at quadrature points in each direction
        
    basis1 : list of np.arrays
        basis functions in L2 evaluated at quadrature points in each direction
    
    Returns
    -------
    M1_zy : sparse matrix
        zy-block of the mass matrix in V1
    """
    
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    pts_x, pts_y, pts_z = pts
    wts_x, wts_y, wts_z = wts
    basis0_x, basis0_y, basis0_z = basis0
    basis1_x, basis1_y, basis1_z = basis1
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    tz = Tz[1:-1]
        
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    Nel_x = len(el_b_x) - 1
    Nel_y = len(el_b_y) - 1
    Nel_z = len(el_b_z) - 1
    
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    GG = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                for gx in range(px + 1):
                    for gy in range(py + 1):
                        for gz in range(pz + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            GG[gx, gy, gz, nx, ny, nz] = Ginv(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
                            
    
     
    # ... global matrix and local element matrix
    M1_zy = np.zeros((Nbase_x, Nbase_y, Nbase_z - 1, Nbase_x, Nbase_y - 1, Nbase_z))
    mat = np.zeros((px + 1, py + 1, pz, px + 1, py, pz + 1), order='F')
    # ...
         
        
    # ... assembly
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                
                wx = wts_x[:, nx]
                wy = wts_y[:, ny]
                wz = wts_z[:, nz]
                
                bs0_x = basis0_x[:, 0, :, nx]
                bs0_y = basis0_y[:, 0, :, ny]
                bs0_z = basis0_z[:, 0, :, nz]
                
                bs1_x = basis1_x[:, 0, :, nx]
                bs1_y = basis1_y[:, 0, :, ny]
                bs1_z = basis1_z[:, 0, :, nz]
                
                ggs = gg[:, :, :, nx, ny, nz]
                GGs = GG[:, :, :, nx, ny, nz]
                
                kernels.kernel1(nx, ny, nz, px, py, pz, 0, 0, 1, 0, 1, 0, bs0_x, bs0_y, bs1_z, bs0_x, bs1_y, bs0_z, pz, py, tz, ty, wx, wy, wz, ggs, GGs, mat)
                
                M1_zy[nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz, nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1] += mat[:, :, :, :, :, :]
    # ...
        
        
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        M1_zy[:px, :, :, :, :, :] += M1_zy[-px:, :, :, :, :, :]
        M1_zy[:, :, :, :px, :, :] += M1_zy[:, :, :, -px:, :, :]
        M1_zy = M1_zy[:M1_zy.shape[0] - px, :, :, :M1_zy.shape[3] - px, :, :]
        
        Nbase_x_0_i = Nbase_x - px
        Nbase_x_0_j = Nbase_x - px
        
    elif bc_x == False:
        M1_zy = M1_zy[1:-1, :, :, 1:-1, :, :]
        
        Nbase_x_0_i = Nbase_x - 2
        Nbase_x_0_j = Nbase_x - 2
        
    else:
        Nbase_x_0_i = Nbase_x
        Nbase_x_0_j = Nbase_x
    # ... 
        
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)    
    if bc_y == True:
        M1_zy[:, :py, :, :, :, :] += M1_zy[:, -py:, :, :, :, :]
        M1_zy[:, :, :, :, :py - 1, :] += M1_zy[:, :, :, :, -py + 1:, :]
        M1_zy = M1_zy[:, :M1_zy.shape[1] - py, :, :, :M1_zy.shape[4] - py + 1, :]
        
        Nbase_y_0_i = Nbase_y - py
        Nbase_y_0_j = Nbase_y - py
        
    elif bc_y == False:
        M1_zy = M1_zy[:, 1:-1, :, :, :, :]
        
        Nbase_y_0_i = Nbase_y - 2
        Nbase_y_0_j = Nbase_y - 1
        
    else:
        Nbase_y_0_i = Nbase_y
        Nbase_y_0_j = Nbase_y - 1
    # ...
    
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)        
    if bc_z == True:
        M1_zy[:, :, :pz - 1, :, :, :] += M1_zy[:, :, -pz + 1:, :, :, :]
        M1_zy[:, :, :, :, :, :pz] += M1_zy[:, :, :, :, :, -pz:]
        M1_zy = M1_zy[:, :, :M1_zy.shape[2] - pz + 1, :, :, :M1_zy.shape[5] - pz]
        
        Nbase_z_0_i = Nbase_z - pz
        Nbase_z_0_j = Nbase_z - pz
        
    elif bc_z == False:
        M1_zy = M1_zy[:, :, :, :, :, 1:-1]
        
        Nbase_z_0_i = Nbase_z - 1
        Nbase_z_0_j = Nbase_z - 2
        
    else:
        Nbase_z_0_i = Nbase_z - 1
        Nbase_z_0_j = Nbase_z
        
                    
    return sparse.csr_matrix(np.reshape(M1_zy, (Nbase_x_0_i*Nbase_y_0_i*Nbase_z_0_i, Nbase_x_0_j*Nbase_y_0_j*Nbase_z_0_j)))





def mass_matrix_V1_zz(p, Nbase, T, Ginv, g, bc, pts, wts, basis0, basis1):
    """
    Computes the zz-block of the mass matrix of the space V1 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors in each direction
        
    Ginv : callable
        the xx-component of the inverse of the metric tensor G
        
    g : callable
        square root of the Jacobi determinant
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions) 
        
    pts : list of np.arrays
        quadrature points in each direction
        
    wts : list of np.arrays
        quadrature weights in each direction
        
    basis0 : list of np.arrays
        basis functions in H1 evaluated at quadrature points in each direction
        
    basis1 : list of np.arrays
        basis functions in L2 evaluated at quadrature points in each direction
    
    Returns
    -------
    M1_zz : sparse matrix
        zz-block of the mass matrix in V1
    """
    
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    pts_x, pts_y, pts_z = pts
    wts_x, wts_y, wts_z = wts
    basis0_x, basis0_y, basis0_z = basis0
    basis1_x, basis1_y, basis1_z = basis1
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    tz = Tz[1:-1]
        
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    Nel_x = len(el_b_x) - 1
    Nel_y = len(el_b_y) - 1
    Nel_z = len(el_b_z) - 1
    
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    GG = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                for gx in range(px + 1):
                    for gy in range(py + 1):
                        for gz in range(pz + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            GG[gx, gy, gz, nx, ny, nz] = Ginv(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
                            
    
     
    # ... global matrix and local element matrix
    M1_zz = np.zeros((Nbase_x, Nbase_y, Nbase_z - 1, Nbase_x, Nbase_y, Nbase_z - 1))
    mat = np.zeros((px + 1, py + 1, pz, px + 1, py + 1, pz), order='F')
    # ...
         
        
    # ... assembly
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                
                wx = wts_x[:, nx]
                wy = wts_y[:, ny]
                wz = wts_z[:, nz]
                
                bs0_x = basis0_x[:, 0, :, nx]
                bs0_y = basis0_y[:, 0, :, ny]
                bs0_z = basis0_z[:, 0, :, nz]
                
                bs1_x = basis1_x[:, 0, :, nx]
                bs1_y = basis1_y[:, 0, :, ny]
                bs1_z = basis1_z[:, 0, :, nz]
                
                ggs = gg[:, :, :, nx, ny, nz]
                GGs = GG[:, :, :, nx, ny, nz]
                
                kernels.kernel1(nx, ny, nz, px, py, pz, 0, 0, 1, 0, 0, 1, bs0_x, bs0_y, bs1_z, bs0_x, bs0_y, bs1_z, pz, pz, tz, tz, wx, wy, wz, ggs, GGs, mat)
                
                M1_zz[nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz, nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz] += mat[:, :, :, :, :, :]
    # ...
        
        
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        M1_zz[:px, :, :, :, :, :] += M1_zz[-px:, :, :, :, :, :]
        M1_zz[:, :, :, :px, :, :] += M1_zz[:, :, :, -px:, :, :]
        M1_zz = M1_zz[:M1_zz.shape[0] - px, :, :, :M1_zz.shape[3] - px, :, :]
        
        Nbase_x_0_i = Nbase_x - px
        Nbase_x_0_j = Nbase_x - px
        
    elif bc_x == False:
        M1_zz = M1_zz[1:-1, :, :, 1:-1, :, :]
        
        Nbase_x_0_i = Nbase_x - 2
        Nbase_x_0_j = Nbase_x - 2
        
    else:
        Nbase_x_0_i = Nbase_x
        Nbase_x_0_j = Nbase_x
    # ... 
        
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)    
    if bc_y == True:
        M1_zz[:, :py, :, :, :, :] += M1_zz[:, -py:, :, :, :, :]
        M1_zz[:, :, :, :, :py, :] += M1_zz[:, :, :, :, -py:, :]
        M1_zz = M1_zz[:, :M1_zz.shape[1] - py, :, :, :M1_zz.shape[4] - py, :]
        
        Nbase_y_0_i = Nbase_y - py
        Nbase_y_0_j = Nbase_y - py
        
    elif bc_y == False:
        M1_zz = M1_zz[:, 1:-1, :, :, 1:-1, :]
        
        Nbase_y_0_i = Nbase_y - 2
        Nbase_y_0_j = Nbase_y - 2
        
    else:
        Nbase_y_0_i = Nbase_y
        Nbase_y_0_j = Nbase_y
    # ...
    
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)        
    if bc_z == True:
        M1_zz[:, :, :pz - 1, :, :, :] += M1_zz[:, :, -pz + 1:, :, :, :]
        M1_zz[:, :, :, :, :, :pz - 1] += M1_zz[:, :, :, :, :, -pz + 1:]
        M1_zz = M1_zz[:, :, :M1_zz.shape[2] - pz + 1, :, :, :M1_zz.shape[5] - pz + 1]
        
        Nbase_z_0_i = Nbase_z - pz
        Nbase_z_0_j = Nbase_z - pz
        
    elif bc_z == False:
        Nbase_z_0_i = Nbase_z - 1
        Nbase_z_0_j = Nbase_z - 1
        
    else:
        Nbase_z_0_i = Nbase_z - 1
        Nbase_z_0_j = Nbase_z - 1
        
                    
    return sparse.csr_matrix(np.reshape(M1_zz, (Nbase_x_0_i*Nbase_y_0_i*Nbase_z_0_i, Nbase_x_0_j*Nbase_y_0_j*Nbase_z_0_j)))





def mass_matrix_V1(p, Nbase, T, Ginv, g, bc):
    """
    Computes the mass matrix of the space V1 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
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
        
    g : callable
        square root of the Jacobi determinant
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions)
        
        
    Returns
    -------
    M1: sparse matrix
        mass matrix in V1
    """
    
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    tz = Tz[1:-1]
        
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    Nel_x = len(el_b_x) - 1
    Nel_y = len(el_b_y) - 1
    Nel_z = len(el_b_z) - 1
    
    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px + 1)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py + 1)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz + 1)

    pts_x, wts_x = inter.construct_quadrature_grid(Nel_x, px + 1, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(Nel_y, py + 1, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(Nel_z, pz + 1, pts_z_loc, wts_z_loc, el_b_z)
    
    pts = [pts_x, pts_y, pts_z]
    wts = [wts_x, wts_y, wts_z]

    basis0_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px + 1, 0, Tx, pts_x) 
    basis0_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py + 1, 0, Ty, pts_y) 
    basis0_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz + 1, 0, Tz, pts_z)
    
    basis0 = [basis0_x, basis0_y, basis0_z]
    
    basis1_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px + 1, 0, tx, pts_x) 
    basis1_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py + 1, 0, ty, pts_y) 
    basis1_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz + 1, 0, tz, pts_z)
    
    basis1 = [basis1_x, basis1_y, basis1_z]
    
    M1_xx = mass_matrix_V1_xx(p, Nbase, T, Ginv[0][0], g, bc, pts, wts, basis0, basis1)
    M1_xy = mass_matrix_V1_xy(p, Nbase, T, Ginv[0][1], g, bc, pts, wts, basis0, basis1)
    M1_xz = mass_matrix_V1_xz(p, Nbase, T, Ginv[0][2], g, bc, pts, wts, basis0, basis1)
    M1_yx = mass_matrix_V1_yx(p, Nbase, T, Ginv[1][0], g, bc, pts, wts, basis0, basis1)
    M1_yy = mass_matrix_V1_yy(p, Nbase, T, Ginv[1][1], g, bc, pts, wts, basis0, basis1)
    M1_yz = mass_matrix_V1_yz(p, Nbase, T, Ginv[1][2], g, bc, pts, wts, basis0, basis1)
    M1_zx = mass_matrix_V1_zx(p, Nbase, T, Ginv[2][0], g, bc, pts, wts, basis0, basis1)
    M1_zy = mass_matrix_V1_zy(p, Nbase, T, Ginv[2][1], g, bc, pts, wts, basis0, basis1)
    M1_zz = mass_matrix_V1_zz(p, Nbase, T, Ginv[2][2], g, bc, pts, wts, basis0, basis1)
        
                    
    return sparse.bmat([[M1_xx, M1_xy, M1_xz], [M1_yx, M1_yy, M1_yz], [M1_zx, M1_zy, M1_zz]], format='csr'), M1_xx, M1_xy, M1_xz, M1_yx, M1_yy, M1_yz, M1_zx, M1_zy, M1_zz 




def L2_prod_V0_curved(fun, p, Nbase, T, g):
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
        
    g : callable
        square root of the Jacobi determinant
        
    Returns
    -------
    f_int : np.array
        the result of the integration with each basis function
    """
        
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    Nel_x = len(el_b_x) - 1
    Nel_y = len(el_b_y) - 1
    Nel_z = len(el_b_z) - 1
    
    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px + 1)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py + 1)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz + 1)
    
    pts_x, wts_x = inter.construct_quadrature_grid(Nel_x, px + 1, pts_x_loc, wts_x_loc, el_b_x)
    pts_y, wts_y = inter.construct_quadrature_grid(Nel_y, py + 1, pts_y_loc, wts_y_loc, el_b_y)
    pts_z, wts_z = inter.construct_quadrature_grid(Nel_z, pz + 1, pts_z_loc, wts_z_loc, el_b_z)
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z))
    ff = np.zeros((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z))
    
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                for gx in range(px + 1):
                    for gy in range(py + 1):
                        for gz in range(pz + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            ff[gx, gy, gz, nx, ny, nz] = fun(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
    
    d = 0
    basis0_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px + 1, d, Tx, pts_x)
    basis0_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py + 1, d, Ty, pts_y)
    basis0_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz + 1, d, Tz, pts_z)
    
    # ... global vector and local element vector
    f_int = np.zeros((Nbase_x, Nbase_y, Nbase_z))
    mat = np.zeros((px + 1, py + 1, pz + 1), order='F')
    # ...
    
    
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                
                
                wx = wts_x[:, nx]
                wy = wts_y[:, ny]
                wz = wts_z[:, nz]
                
                bs0_x = basis0_x[:, 0, :, nx]
                bs0_y = basis0_y[:, 0, :, ny]
                bs0_z = basis0_z[:, 0, :, nz]
                
                ggs = gg[:, :, :, nx, ny, nz]
                ffs = ff[:, :, :, nx, ny, nz]
                
                kernels.kernelL0(px, py, pz, bs0_x, bs0_y, bs0_z, wx, wy, wz, ggs, ffs, mat)
                
                f_int[nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz + 1] += mat[:, :, :]
    
    return f_int