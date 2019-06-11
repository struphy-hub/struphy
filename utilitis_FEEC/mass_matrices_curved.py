import numpy as np
import scipy.sparse as sparse
import psydac.core.interface as inter
from pyccel import epyccel

import utilitis_FEEC.kernels as kernels


kernels = epyccel(kernels)


def mass_matrix_V0(p, Nbase, T, g, bc):
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
        
    g : callable
        square root of the Jacobi determinant of the metric tensor g
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions)
        
        
    Returns
    -------
    M: sparse matrix
        mass matrix in V0
    """
    
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
        
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

    basis0_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px + 1, 0, Tx, pts_x) 
    basis0_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py + 1, 0, Ty, pts_y) 
    basis0_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz + 1, 0, Tz, pts_z)
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.empty((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nz in range(Nel_z):
        for ny in range(Nel_y):
            for nx in range(Nel_x):
                for gz in range(pz + 1):
                    for gy in range(py + 1):
                        for gx in range(px + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
    
    
    # ... global matrix and local element matrix
    M = np.zeros((Nbase_x, Nbase_y, Nbase_z, Nbase_x, Nbase_y, Nbase_z))
    mat = np.zeros((px + 1, py + 1, pz + 1, px + 1, py + 1, pz + 1), order='F')
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
                
                ggs = gg[:, :, :, nx, ny, nz]
                
                kernels.kernel0(px, py, pz, bs0_x, bs0_y, bs0_z, wx, wy, wz, ggs, mat)
                
                M[nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz + 1, nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz + 1] += mat[:, :, :, :, :, :]
    # ...   
    
    
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        M[:px, :, :, :, :, :] += M[-px:, :, :, :, :, :]
        M[:, :, :, :px, :, :] += M[:, :, :, -px:, :, :]
        M = M[:M.shape[0] - px, :, :, :M.shape[3] - px, :, :]

        Nbase_x_0_i = Nbase_x - px
        Nbase_x_0_j = Nbase_x - px

    elif bc_x == False:
        M = M[1:-1, :, :, 1:-1, :, :]
        
        Nbase_x_0_i = Nbase_x - 2
        Nbase_x_0_j = Nbase_x - 2

    else:
        Nbase_x_0_i = Nbase_x
        Nbase_x_0_j = Nbase_x
    # ...
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)
    if bc_y == True:
        M[:, :py, :, :, :, :] += M[:, -py:, :, :, :, :]
        M[:, :, :, :, :py, :] += M[:, :, :, :, -py:, :]
        M = M[:, :M.shape[1] - py, :, :, :M.shape[4] - py, :]
        
        Nbase_y_0_i = Nbase_y - py
        Nbase_y_0_j = Nbase_y - py

    elif bc_y == False:
        M = M[:, 1:-1, :, :, 1:-1, :]
        
        Nbase_y_0_i = Nbase_y - 2
        Nbase_y_0_j = Nbase_y - 2
        
    else:
        Nbase_y_0_i = Nbase_y
        Nbase_y_0_j = Nbase_y
    # ...
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)   
    if bc_z == True:
        M[:, :, :pz, :, :, :] += M[:, :, -pz:, :, :, :]
        M[:, :, :, :, :, :pz] += M[:, :, :, :, :, -pz:]
        M = M[:, :, :M.shape[2] - pz, :, :, :M.shape[5] - pz]
        
        Nbase_z_0_i = Nbase_z - pz
        Nbase_z_0_j = Nbase_z - pz

    elif bc_z == False:
        M = M[:, :, 1:-1, :, :, 1:-1]
        
        Nbase_z_0_i = Nbase_z - 2
        Nbase_z_0_j = Nbase_z - 2
        
    else:
        Nbase_z_0_i = Nbase_z
        Nbase_z_0_j = Nbase_z
    # ...
    
    M = sparse.csr_matrix(np.reshape(M, (Nbase_x_0_i*Nbase_y_0_i*Nbase_z_0_i, Nbase_x_0_j*Nbase_y_0_j*Nbase_z_0_j)))
                    
    return M




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
    M: sparse matrix
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

    basis0_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px + 1, 0, Tx, pts_x) 
    basis0_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py + 1, 0, Ty, pts_y) 
    basis0_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz + 1, 0, Tz, pts_z)
    
    basis1_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px + 1, 0, tx, pts_x) 
    basis1_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py + 1, 0, ty, pts_y) 
    basis1_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz + 1, 0, tz, pts_z)
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.empty((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    GG = np.empty((3, 3, px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nz in range(Nel_z):
        for ny in range(Nel_y):
            for nx in range(Nel_x):
                for gz in range(pz + 1):
                    for gy in range(py + 1):
                        for gx in range(px + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            
                            for d2 in range(3):
                                for d1 in range(3):
                                    GG[d1, d2, gx, gy, gz, nx, ny, nz] = Ginv[d1][d2](pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
    
    
    # ... global matrices and local element matrices
    M_xx = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z, Nbase_x - 1, Nbase_y, Nbase_z))
    M_xy = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z, Nbase_x, Nbase_y - 1, Nbase_z))
    M_xz = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z, Nbase_x, Nbase_y, Nbase_z - 1))
    M_yx = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z, Nbase_x - 1, Nbase_y, Nbase_z))
    M_yy = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z, Nbase_x, Nbase_y - 1, Nbase_z))
    M_yz = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z, Nbase_x, Nbase_y, Nbase_z - 1))
    M_zx = np.zeros((Nbase_x, Nbase_y, Nbase_z - 1, Nbase_x - 1, Nbase_y, Nbase_z))
    M_zy = np.zeros((Nbase_x, Nbase_y, Nbase_z - 1, Nbase_x, Nbase_y - 1, Nbase_z))
    M_zz = np.zeros((Nbase_x, Nbase_y, Nbase_z - 1, Nbase_x, Nbase_y, Nbase_z - 1))
    
    mat_xx = np.zeros((px, py + 1, pz + 1, px, py + 1, pz + 1), order='F')
    mat_xy = np.zeros((px, py + 1, pz + 1, px + 1, py, pz + 1), order='F')
    mat_xz = np.zeros((px, py + 1, pz + 1, px + 1, py + 1, pz), order='F')
    mat_yx = np.zeros((px + 1, py, pz + 1, px, py + 1, pz + 1), order='F')
    mat_yy = np.zeros((px + 1, py, pz + 1, px + 1, py, pz + 1), order='F')
    mat_yz = np.zeros((px + 1, py, pz + 1, px + 1, py + 1, pz), order='F')
    mat_zx = np.zeros((px + 1, py + 1, pz, px, py + 1, pz + 1), order='F')
    mat_zy = np.zeros((px + 1, py + 1, pz, px + 1, py, pz + 1), order='F')
    mat_zz = np.zeros((px + 1, py + 1, pz, px + 1, py + 1, pz), order='F')
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
                
                kernels.kernel1(nx, ny, nz, px, py, pz, 1, 0, 0, 1, 0, 0, bs1_x, bs0_y, bs0_z, bs1_x, bs0_y, bs0_z, px, px, tx, tx, wx, wy, wz, ggs, GG[0, 0, :, :, :, nx, ny, nz], mat_xx)
                kernels.kernel1(nx, ny, nz, px, py, pz, 1, 0, 0, 0, 1, 0, bs1_x, bs0_y, bs0_z, bs0_x, bs1_y, bs0_z, px, py, tx, ty, wx, wy, wz, ggs, GG[0, 1, :, :, :, nx, ny, nz], mat_xy)
                kernels.kernel1(nx, ny, nz, px, py, pz, 1, 0, 0, 0, 0, 1, bs1_x, bs0_y, bs0_z, bs0_x, bs0_y, bs1_z, px, pz, tx, tz, wx, wy, wz, ggs, GG[0, 2, :, :, :, nx, ny, nz], mat_xz)
                kernels.kernel1(nx, ny, nz, px, py, pz, 0, 1, 0, 1, 0, 0, bs0_x, bs1_y, bs0_z, bs1_x, bs0_y, bs0_z, py, px, ty, tx, wx, wy, wz, ggs, GG[1, 0, :, :, :, nx, ny, nz], mat_yx)
                kernels.kernel1(nx, ny, nz, px, py, pz, 0, 1, 0, 0, 1, 0, bs0_x, bs1_y, bs0_z, bs0_x, bs1_y, bs0_z, py, py, ty, ty, wx, wy, wz, ggs, GG[1, 1, :, :, :, nx, ny, nz], mat_yy)
                kernels.kernel1(nx, ny, nz, px, py, pz, 0, 1, 0, 0, 0, 1, bs0_x, bs1_y, bs0_z, bs0_x, bs0_y, bs1_z, py, pz, ty, tz, wx, wy, wz, ggs, GG[1, 2, :, :, :, nx, ny, nz], mat_yz)
                kernels.kernel1(nx, ny, nz, px, py, pz, 0, 0, 1, 1, 0, 0, bs0_x, bs0_y, bs1_z, bs1_x, bs0_y, bs0_z, pz, px, tz, tx, wx, wy, wz, ggs, GG[2, 0, :, :, :, nx, ny, nz], mat_zx)
                kernels.kernel1(nx, ny, nz, px, py, pz, 0, 0, 1, 0, 1, 0, bs0_x, bs0_y, bs1_z, bs0_x, bs1_y, bs0_z, pz, py, tz, ty, wx, wy, wz, ggs, GG[2, 1, :, :, :, nx, ny, nz], mat_zy)
                kernels.kernel1(nx, ny, nz, px, py, pz, 0, 0, 1, 0, 0, 1, bs0_x, bs0_y, bs1_z, bs0_x, bs0_y, bs1_z, pz, pz, tz, tz, wx, wy, wz, ggs, GG[2, 2, :, :, :, nx, ny, nz], mat_zz)
                
                M_xx[nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1, nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1] += mat_xx[:, :, :, :, :, :]
                M_xy[nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1, nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1] += mat_xy[:, :, :, :, :, :]
                M_xz[nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1, nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz] += mat_xz[:, :, :, :, :, :]
                M_yx[nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1, nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1] += mat_yx[:, :, :, :, :, :]
                M_yy[nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1, nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1] += mat_yy[:, :, :, :, :, :]
                M_yz[nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1, nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz] += mat_yz[:, :, :, :, :, :]
                M_zx[nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz, nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1] += mat_zx[:, :, :, :, :, :]
                M_zy[nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz, nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1] += mat_zy[:, :, :, :, :, :]
                M_zz[nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz, nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz] += mat_zz[:, :, :, :, :, :]
    # ...   
    
    
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        M_xx[:px - 1, :, :, :, :, :] += M_xx[-px + 1:, :, :, :, :, :]
        M_xx[:, :, :, :px - 1, :, :] += M_xx[:, :, :, -px + 1:, :, :]
        M_xx = M_xx[:M_xx.shape[0] - px + 1, :, :, :M_xx.shape[3] - px + 1, :, :]

        M_xy[:px - 1, :, :, :, :, :] += M_xy[-px + 1:, :, :, :, :, :]
        M_xy[:, :, :, :px, :, :] += M_xy[:, :, :, -px:, :, :]
        M_xy = M_xy[:M_xy.shape[0] - px + 1, :, :, :M_xy.shape[3] - px, :, :]

        M_xz[:px - 1, :, :, :, :, :] += M_xz[-px + 1:, :, :, :, :, :]
        M_xz[:, :, :, :px, :, :] += M_xz[:, :, :, -px:, :, :]
        M_xz = M_xz[:M_xz.shape[0] - px + 1, :, :, :M_xz.shape[3] - px, :, :]

        M_yx[:px, :, :, :, :, :] += M_yx[-px:, :, :, :, :, :]
        M_yx[:, :, :, :px - 1, :, :] += M_yx[:, :, :, -px + 1:, :, :]
        M_yx = M_yx[:M_yx.shape[0] - px, :, :, :M_yx.shape[3] - px + 1, :, :]

        M_yy[:px, :, :, :, :, :] += M_yy[-px:, :, :, :, :, :]
        M_yy[:, :, :, :px, :, :] += M_yy[:, :, :, -px:, :, :]
        M_yy = M_yy[:M_yy.shape[0] - px, :, :, :M_yy.shape[3] - px, :, :]

        M_yz[:px, :, :, :, :, :] += M_yz[-px:, :, :, :, :, :]
        M_yz[:, :, :, :px, :, :] += M_yz[:, :, :, -px:, :, :]
        M_yz = M_yz[:M_yz.shape[0] - px, :, :, :M_yz.shape[3] - px, :, :]

        M_zx[:px, :, :, :, :, :] += M_zx[-px:, :, :, :, :, :]
        M_zx[:, :, :, :px - 1, :, :] += M_zx[:, :, :, -px + 1:, :, :]
        M_zx = M_zx[:M_zx.shape[0] - px, :, :, :M_zx.shape[3] - px + 1, :, :]

        M_zy[:px, :, :, :, :, :] += M_zy[-px:, :, :, :, :, :]
        M_zy[:, :, :, :px, :, :] += M_zy[:, :, :, -px:, :, :]
        M_zy = M_zy[:M_zy.shape[0] - px, :, :, :M_zy.shape[3] - px, :, :]

        M_zz[:px, :, :, :, :, :] += M_zz[-px:, :, :, :, :, :]
        M_zz[:, :, :, :px, :, :] += M_zz[:, :, :, -px:, :, :]
        M_zz = M_zz[:M_zz.shape[0] - px, :, :, :M_zz.shape[3] - px, :, :]

        Nbase_x_0_xx_i = Nbase_x - px
        Nbase_x_0_xx_j = Nbase_x - px

        Nbase_x_0_xy_i = Nbase_x - px
        Nbase_x_0_xy_j = Nbase_x - px

        Nbase_x_0_xz_i = Nbase_x - px
        Nbase_x_0_xz_j = Nbase_x - px

        Nbase_x_0_yx_i = Nbase_x - px
        Nbase_x_0_yx_j = Nbase_x - px

        Nbase_x_0_yy_i = Nbase_x - px
        Nbase_x_0_yy_j = Nbase_x - px

        Nbase_x_0_yz_i = Nbase_x - px
        Nbase_x_0_yz_j = Nbase_x - px

        Nbase_x_0_zx_i = Nbase_x - px
        Nbase_x_0_zx_j = Nbase_x - px

        Nbase_x_0_zy_i = Nbase_x - px
        Nbase_x_0_zy_j = Nbase_x - px

        Nbase_x_0_zz_i = Nbase_x - px
        Nbase_x_0_zz_j = Nbase_x - px

    elif bc_x == False:
        M_xy = M_xy[:, :, :, 1:-1, :, :]
        M_xz = M_xz[:, :, :, 1:-1, :, :]
        M_yx = M_yx[1:-1, :, :, :, :, :]
        M_yy = M_yy[1:-1, :, :, 1:-1, :, :]
        M_yz = M_yz[1:-1, :, :, 1:-1, :, :]
        M_zx = M_zx[1:-1, :, :, :, :, :]
        M_zy = M_zy[1:-1, :, :, 1:-1, :, :]
        M_zz = M_zz[1:-1, :, :, 1:-1, :, :]
        
        Nbase_x_0_xx_i = Nbase_x - 1
        Nbase_x_0_xx_j = Nbase_x - 1

        Nbase_x_0_xy_i = Nbase_x - 1
        Nbase_x_0_xy_j = Nbase_x - 2

        Nbase_x_0_xz_i = Nbase_x - 1
        Nbase_x_0_xz_j = Nbase_x - 2

        Nbase_x_0_yx_i = Nbase_x - 2
        Nbase_x_0_yx_j = Nbase_x - 1

        Nbase_x_0_yy_i = Nbase_x - 2
        Nbase_x_0_yy_j = Nbase_x - 2

        Nbase_x_0_yz_i = Nbase_x - 2
        Nbase_x_0_yz_j = Nbase_x - 2

        Nbase_x_0_zx_i = Nbase_x - 2
        Nbase_x_0_zx_j = Nbase_x - 1

        Nbase_x_0_zy_i = Nbase_x - 2
        Nbase_x_0_zy_j = Nbase_x - 2

        Nbase_x_0_zz_i = Nbase_x - 2
        Nbase_x_0_zz_j = Nbase_x - 2

    else:
        Nbase_x_0_xx_i = Nbase_x - 1
        Nbase_x_0_xx_j = Nbase_x - 1

        Nbase_x_0_xy_i = Nbase_x - 1
        Nbase_x_0_xy_j = Nbase_x

        Nbase_x_0_xz_i = Nbase_x - 1
        Nbase_x_0_xz_j = Nbase_x

        Nbase_x_0_yx_i = Nbase_x
        Nbase_x_0_yx_j = Nbase_x - 1

        Nbase_x_0_yy_i = Nbase_x
        Nbase_x_0_yy_j = Nbase_x

        Nbase_x_0_yz_i = Nbase_x
        Nbase_x_0_yz_j = Nbase_x

        Nbase_x_0_zx_i = Nbase_x
        Nbase_x_0_zx_j = Nbase_x - 1

        Nbase_x_0_zy_i = Nbase_x
        Nbase_x_0_zy_j = Nbase_x

        Nbase_x_0_zz_i = Nbase_x
        Nbase_x_0_zz_j = Nbase_x
    # ...
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)
    if bc_y == True:
        M_xx[:, :py, :, :, :, :] += M_xx[:, -py:, :, :, :, :]
        M_xx[:, :, :, :, :py, :] += M_xx[:, :, :, :, -py:, :]
        M_xx = M_xx[:, :M_xx.shape[1] - py, :, :, :M_xx.shape[4] - py, :]
        
        M_xy[:, :py, :, :, :, :] += M_xy[:, -py:, :, :, :, :]
        M_xy[:, :, :, :, :py - 1, :] += M_xy[:, :, :, :, -py + 1:, :]
        M_xy = M_xy[:, :M_xy.shape[1] - py, :, :, :M_xy.shape[4] - py + 1, :]
        
        M_xz[:, :py, :, :, :, :] += M_xz[:, -py:, :, :, :, :]
        M_xz[:, :, :, :, :py, :] += M_xz[:, :, :, :, -py:, :]
        M_xz = M_xz[:, :M_xz.shape[1] - py, :, :, :M_xz.shape[4] - py, :]
        
        M_yx[:, :py - 1, :, :, :, :] += M_yx[:, -py + 1:, :, :, :, :]
        M_yx[:, :, :, :, :py, :] += M_yx[:, :, :, :, -py:, :]
        M_yx = M_yx[:, :M_yx.shape[1] - py + 1, :, :, :M_yx.shape[4] - py, :]
        
        M_yy[:, :py - 1, :, :, :, :] += M_yy[:, -py + 1:, :, :, :, :]
        M_yy[:, :, :, :, :py - 1, :] += M_yy[:, :, :, :, -py + 1:, :]
        M_yy = M_yy[:, :M_yy.shape[1] - py + 1, :, :, :M_yy.shape[4] - py + 1, :]
        
        M_yz[:, :py - 1, :, :, :, :] += M_yz[:, -py + 1:, :, :, :, :]
        M_yz[:, :, :, :, :py, :] += M_yz[:, :, :, :, -py:, :]
        M_yz = M_yz[:, :M_yz.shape[1] - py + 1, :, :, :M_yz.shape[4] - py, :]
        
        M_zx[:, :py, :, :, :, :] += M_zx[:, -py:, :, :, :, :]
        M_zx[:, :, :, :, :py, :] += M_zx[:, :, :, :, -py:, :]
        M_zx = M_zx[:, :M_zx.shape[1] - py, :, :, :M_zx.shape[4] - py, :]
        
        M_zy[:, :py, :, :, :, :] += M_zy[:, -py:, :, :, :, :]
        M_zy[:, :, :, :, :py - 1, :] += M_zy[:, :, :, :, -py + 1:, :]
        M_zy = M_zy[:, :M_zy.shape[1] - py, :, :, :M_zy.shape[4] - py + 1, :]
        
        M_zz[:, :py, :, :, :, :] += M_zz[:, -py:, :, :, :, :]
        M_zz[:, :, :, :, :py, :] += M_zz[:, :, :, :, -py:, :]
        M_zz = M_zz[:, :M_zz.shape[1] - py, :, :, :M_zz.shape[4] - py, :]
        
        Nbase_y_0_xx_i = Nbase_y - py
        Nbase_y_0_xx_j = Nbase_y - py

        Nbase_y_0_xy_i = Nbase_y - py
        Nbase_y_0_xy_j = Nbase_y - py

        Nbase_y_0_xz_i = Nbase_y - py
        Nbase_y_0_xz_j = Nbase_y - py

        Nbase_y_0_yx_i = Nbase_y - py
        Nbase_y_0_yx_j = Nbase_y - py
        
        Nbase_y_0_yy_i = Nbase_y - py
        Nbase_y_0_yy_j = Nbase_y - py

        Nbase_y_0_yz_i = Nbase_y - py
        Nbase_y_0_yz_j = Nbase_y - py

        Nbase_y_0_zx_i = Nbase_y - py
        Nbase_y_0_zx_j = Nbase_y - py

        Nbase_y_0_zy_i = Nbase_y - py
        Nbase_y_0_zy_j = Nbase_y - py

        Nbase_y_0_zz_i = Nbase_y - py
        Nbase_y_0_zz_j = Nbase_y - py
        
    elif bc_y == False:
        M_xx = M_xx[:, 1:-1, :, :, 1:-1, :]
        M_xy = M_xy[:, 1:-1, :, :, :, :]
        M_xz = M_xz[:, 1:-1, :, :, 1:-1, :]
        M_yx = M_yx[:, :, :, :, 1:-1, :]
        M_yz = M_yz[:, :, :, :, 1:-1, :]
        M_zx = M_zx[:, 1:-1, :, :, 1:-1, :]
        M_zy = M_zy[:, 1:-1, :, :, :, :]
        M_zz = M_zz[:, 1:-1, :, :, 1:-1, :]
        
        Nbase_y_0_xx_i = Nbase_y - 2
        Nbase_y_0_xx_j = Nbase_y - 2

        Nbase_y_0_xy_i = Nbase_y - 2
        Nbase_y_0_xy_j = Nbase_y - 1

        Nbase_y_0_xz_i = Nbase_y - 2
        Nbase_y_0_xz_j = Nbase_y - 2

        Nbase_y_0_yx_i = Nbase_y - 1
        Nbase_y_0_yx_j = Nbase_y - 2
        
        Nbase_y_0_yy_i = Nbase_y - 1
        Nbase_y_0_yy_j = Nbase_y - 1

        Nbase_y_0_yz_i = Nbase_y - 1
        Nbase_y_0_yz_j = Nbase_y - 2

        Nbase_y_0_zx_i = Nbase_y - 2
        Nbase_y_0_zx_j = Nbase_y - 2

        Nbase_y_0_zy_i = Nbase_y - 2
        Nbase_y_0_zy_j = Nbase_y - 1

        Nbase_y_0_zz_i = Nbase_y - 2
        Nbase_y_0_zz_j = Nbase_y - 2
        
    else:
        Nbase_y_0_xx_i = Nbase_y
        Nbase_y_0_xx_j = Nbase_y

        Nbase_y_0_xy_i = Nbase_y
        Nbase_y_0_xy_j = Nbase_y - 1

        Nbase_y_0_xz_i = Nbase_y
        Nbase_y_0_xz_j = Nbase_y

        Nbase_y_0_yx_i = Nbase_y - 1
        Nbase_y_0_yx_j = Nbase_y
        
        Nbase_y_0_yy_i = Nbase_y - 1
        Nbase_y_0_yy_j = Nbase_y - 1

        Nbase_y_0_yz_i = Nbase_y - 1
        Nbase_y_0_yz_j = Nbase_y

        Nbase_y_0_zx_i = Nbase_y
        Nbase_y_0_zx_j = Nbase_y

        Nbase_y_0_zy_i = Nbase_y
        Nbase_y_0_zy_j = Nbase_y - 1

        Nbase_y_0_zz_i = Nbase_y
        Nbase_y_0_zz_j = Nbase_y
    # ...
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)   
    if bc_z == True:
        M_xx[:, :, :pz, :, :, :] += M_xx[:, :, -pz:, :, :, :]
        M_xx[:, :, :, :, :, :pz] += M_xx[:, :, :, :, :, -pz:]
        M_xx = M_xx[:, :, :M_xx.shape[2] - pz, :, :, :M_xx.shape[5] - pz]
        
        M_xy[:, :, :pz, :, :, :] += M_xy[:, :, -pz:, :, :, :]
        M_xy[:, :, :, :, :, :pz] += M_xy[:, :, :, :, :, -pz:]
        M_xy = M_xy[:, :, :M_xy.shape[2] - pz, :, :, :M_xy.shape[5] - pz]
        
        M_xz[:, :, :pz, :, :, :] += M_xz[:, :, -pz:, :, :, :]
        M_xz[:, :, :, :, :, :pz - 1] += M_xz[:, :, :, :, :, -pz + 1:]
        M_xz = M_xz[:, :, :M_xz.shape[2] - pz, :, :, :M_xz.shape[5] - pz + 1]
        
        M_yx[:, :, :pz, :, :, :] += M_yx[:, :, -pz:, :, :, :]
        M_yx[:, :, :, :, :, :pz] += M_yx[:, :, :, :, :, -pz:]
        M_yx = M_yx[:, :, :M_yx.shape[2] - pz, :, :, :M_yx.shape[5] - pz]
        
        M_yy[:, :, :pz, :, :, :] += M_yy[:, :, -pz:, :, :, :]
        M_yy[:, :, :, :, :, :pz] += M_yy[:, :, :, :, :, -pz:]
        M_yy = M_yy[:, :, :M_yy.shape[2] - pz, :, :, :M_yy.shape[5] - pz]
        
        M_yz[:, :, :pz, :, :, :] += M_yz[:, :, -pz:, :, :, :]
        M_yz[:, :, :, :, :, :pz - 1] += M_yz[:, :, :, :, :, -pz + 1:]
        M_yz = M_yz[:, :, :M_yz.shape[2] - pz, :, :, :M_yz.shape[5] - pz + 1]
        
        M_zx[:, :, :pz - 1, :, :, :] += M_zx[:, :, -pz + 1:, :, :, :]
        M_zx[:, :, :, :, :, :pz] += M_zx[:, :, :, :, :, -pz:]
        M_zx = M_zx[:, :, :M_zx.shape[2] - pz + 1, :, :, :M_zx.shape[5] - pz]
        
        M_zy[:, :, :pz - 1, :, :, :] += M_zy[:, :, -pz + 1:, :, :, :]
        M_zy[:, :, :, :, :, :pz] += M_zy[:, :, :, :, :, -pz:]
        M_zy = M_zy[:, :, :M_zy.shape[2] - pz + 1, :, :, :M_zy.shape[5] - pz]
        
        M_zz[:, :, :pz - 1, :, :, :] += M_zz[:, :, -pz + 1:, :, :, :]
        M_zz[:, :, :, :, :, :pz - 1] += M_zz[:, :, :, :, :, -pz + 1:]
        M_zz = M_zz[:, :, :M_zz.shape[2] - pz + 1, :, :, :M_zz.shape[5] - pz + 1]
        
        Nbase_z_0_xx_i = Nbase_z - pz
        Nbase_z_0_xx_j = Nbase_z - pz

        Nbase_z_0_xy_i = Nbase_z - pz
        Nbase_z_0_xy_j = Nbase_z - pz

        Nbase_z_0_xz_i = Nbase_z - pz
        Nbase_z_0_xz_j = Nbase_z - pz
        
        Nbase_z_0_yx_i = Nbase_z - pz
        Nbase_z_0_yx_j = Nbase_z - pz
        
        Nbase_z_0_yy_i = Nbase_z - pz
        Nbase_z_0_yy_j = Nbase_z - pz

        Nbase_z_0_yz_i = Nbase_z - pz
        Nbase_z_0_yz_j = Nbase_z - pz

        Nbase_z_0_zx_i = Nbase_z - pz
        Nbase_z_0_zx_j = Nbase_z - pz

        Nbase_z_0_zy_i = Nbase_z - pz
        Nbase_z_0_zy_j = Nbase_z - pz

        Nbase_z_0_zz_i = Nbase_z - pz
        Nbase_z_0_zz_j = Nbase_z - pz
        
    elif bc_z == False:
        M_xx = M_xx[:, :, 1:-1, :, :, 1:-1]
        M_xy = M_xy[:, :, 1:-1, :, :, 1:-1]
        M_xz = M_xz[:, :, 1:-1, :, :, :]
        M_yx = M_yx[:, :, 1:-1, :, :, 1:-1]
        M_yy = M_yy[:, :, 1:-1, :, :, 1:-1]
        M_yz = M_yz[:, :, 1:-1, :, :, :]
        M_zx = M_zx[:, :, :, :, :, 1:-1]
        M_zy = M_zy[:, :, :, :, :, 1:-1]
        
        Nbase_z_0_xx_i = Nbase_z - 2
        Nbase_z_0_xx_j = Nbase_z - 2

        Nbase_z_0_xy_i = Nbase_z - 2
        Nbase_z_0_xy_j = Nbase_z - 2

        Nbase_z_0_xz_i = Nbase_z - 2
        Nbase_z_0_xz_j = Nbase_z - 1
        
        Nbase_z_0_yx_i = Nbase_z - 2
        Nbase_z_0_yx_j = Nbase_z - 2
        
        Nbase_z_0_yy_i = Nbase_z - 2
        Nbase_z_0_yy_j = Nbase_z - 2

        Nbase_z_0_yz_i = Nbase_z - 1
        Nbase_z_0_yz_j = Nbase_z - 2

        Nbase_z_0_zx_i = Nbase_z - 1
        Nbase_z_0_zx_j = Nbase_z - 2

        Nbase_z_0_zy_i = Nbase_z - 1
        Nbase_z_0_zy_j = Nbase_z - 2

        Nbase_z_0_zz_i = Nbase_z - 1
        Nbase_z_0_zz_j = Nbase_z - 1
        
    else:
        Nbase_z_0_xx_i = Nbase_z
        Nbase_z_0_xx_j = Nbase_z

        Nbase_z_0_xy_i = Nbase_z
        Nbase_z_0_xy_j = Nbase_z

        Nbase_z_0_xz_i = Nbase_z
        Nbase_z_0_xz_j = Nbase_z - 1
        
        Nbase_z_0_yx_i = Nbase_z
        Nbase_z_0_yx_j = Nbase_z
        
        Nbase_z_0_yy_i = Nbase_z
        Nbase_z_0_yy_j = Nbase_z

        Nbase_z_0_yz_i = Nbase_z
        Nbase_z_0_yz_j = Nbase_z - 1

        Nbase_z_0_zx_i = Nbase_z - 1
        Nbase_z_0_zx_j = Nbase_z

        Nbase_z_0_zy_i = Nbase_z - 1
        Nbase_z_0_zy_j = Nbase_z

        Nbase_z_0_zz_i = Nbase_z - 1
        Nbase_z_0_zz_j = Nbase_z - 1
    # ...
    
    M_xx = sparse.csr_matrix(np.reshape(M_xx, (Nbase_x_0_xx_i*Nbase_y_0_xx_i*Nbase_z_0_xx_i, Nbase_x_0_xx_j*Nbase_y_0_xx_j*Nbase_z_0_xx_j)))
    M_xy = sparse.csr_matrix(np.reshape(M_xy, (Nbase_x_0_xy_i*Nbase_y_0_xy_i*Nbase_z_0_xy_i, Nbase_x_0_xy_j*Nbase_y_0_xy_j*Nbase_z_0_xy_j)))
    M_xz = sparse.csr_matrix(np.reshape(M_xz, (Nbase_x_0_xz_i*Nbase_y_0_xz_i*Nbase_z_0_xz_i, Nbase_x_0_xz_j*Nbase_y_0_xz_j*Nbase_z_0_xz_j)))
    M_yx = sparse.csr_matrix(np.reshape(M_yx, (Nbase_x_0_yx_i*Nbase_y_0_yx_i*Nbase_z_0_yx_i, Nbase_x_0_yx_j*Nbase_y_0_yx_j*Nbase_z_0_yx_j)))
    M_yy = sparse.csr_matrix(np.reshape(M_yy, (Nbase_x_0_yy_i*Nbase_y_0_yy_i*Nbase_z_0_yy_i, Nbase_x_0_yy_j*Nbase_y_0_yy_j*Nbase_z_0_yy_j)))
    M_yz = sparse.csr_matrix(np.reshape(M_yz, (Nbase_x_0_yz_i*Nbase_y_0_yz_i*Nbase_z_0_yz_i, Nbase_x_0_yz_j*Nbase_y_0_yz_j*Nbase_z_0_yz_j)))
    M_zx = sparse.csr_matrix(np.reshape(M_zx, (Nbase_x_0_zx_i*Nbase_y_0_zx_i*Nbase_z_0_zx_i, Nbase_x_0_zx_j*Nbase_y_0_zx_j*Nbase_z_0_zx_j)))
    M_zy = sparse.csr_matrix(np.reshape(M_zy, (Nbase_x_0_zy_i*Nbase_y_0_zy_i*Nbase_z_0_zy_i, Nbase_x_0_zy_j*Nbase_y_0_zy_j*Nbase_z_0_zy_j)))
    M_zz = sparse.csr_matrix(np.reshape(M_zz, (Nbase_x_0_zz_i*Nbase_y_0_zz_i*Nbase_z_0_zz_i, Nbase_x_0_zz_j*Nbase_y_0_zz_j*Nbase_z_0_zz_j)))
                    
    return sparse.bmat([[M_xx, M_xy, M_xz], [M_yx, M_yy, M_yz], [M_zx, M_zy, M_zz]], format='csr')




def mass_matrix_V2(p, Nbase, T, G, g, bc):
    """
    Computes the mass matrix of the space V2 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
    
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    G : callable
        the metric tensor
        
    g : callable
        square root of the Jacobi determinant
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions)
        
        
    Returns
    -------
    M: sparse matrix
        mass matrix in V2
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

    basis0_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px + 1, 0, Tx, pts_x) 
    basis0_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py + 1, 0, Ty, pts_y) 
    basis0_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz + 1, 0, Tz, pts_z)
    
    basis1_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px + 1, 0, tx, pts_x) 
    basis1_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py + 1, 0, ty, pts_y) 
    basis1_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz + 1, 0, tz, pts_z)
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.empty((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    GG = np.empty((3, 3, px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nz in range(Nel_z):
        for ny in range(Nel_y):
            for nx in range(Nel_x):
                for gz in range(pz + 1):
                    for gy in range(py + 1):
                        for gx in range(px + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            
                            for d2 in range(3):
                                for d1 in range(3):
                                    GG[d1, d2, gx, gy, gz, nx, ny, nz] = G[d1][d2](pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
    
    
    
    # ... global matrices and local element matrices
    M_xx = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z - 1, Nbase_x, Nbase_y - 1, Nbase_z - 1))
    M_xy = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z - 1, Nbase_x - 1, Nbase_y, Nbase_z - 1))
    M_xz = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z - 1, Nbase_x - 1, Nbase_y - 1, Nbase_z))
    
    M_yx = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z - 1, Nbase_x, Nbase_y - 1, Nbase_z - 1))
    M_yy = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z - 1, Nbase_x - 1, Nbase_y, Nbase_z - 1))
    M_yz = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z - 1, Nbase_x - 1, Nbase_y - 1, Nbase_z))
    
    M_zx = np.zeros((Nbase_x - 1, Nbase_y - 1, Nbase_z, Nbase_x, Nbase_y - 1, Nbase_z - 1))
    M_zy = np.zeros((Nbase_x - 1, Nbase_y - 1, Nbase_z, Nbase_x - 1, Nbase_y, Nbase_z - 1))
    M_zz = np.zeros((Nbase_x - 1, Nbase_y - 1, Nbase_z, Nbase_x - 1, Nbase_y - 1, Nbase_z))
    
    mat_xx = np.zeros((px + 1, py, pz, px + 1, py, pz), order='F')
    mat_xy = np.zeros((px + 1, py, pz, px, py + 1, pz), order='F')
    mat_xz = np.zeros((px + 1, py, pz, px, py, pz + 1), order='F')
    
    mat_yx = np.zeros((px, py + 1, pz, px + 1, py, pz), order='F')
    mat_yy = np.zeros((px, py + 1, pz, px, py + 1, pz), order='F')
    mat_yz = np.zeros((px, py + 1, pz, px, py, pz + 1), order='F')
    
    mat_zx = np.zeros((px, py, pz + 1, px + 1, py, pz), order='F')
    mat_zy = np.zeros((px, py, pz + 1, px, py + 1, pz), order='F')
    mat_zz = np.zeros((px, py, pz + 1, px, py, pz + 1), order='F')
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
                
                kernels.kernel2(nx, ny, nz, px, py, pz, 0, 1, 1, 0, 1, 1, bs0_x, bs1_y, bs1_z, bs0_x, bs1_y, bs1_z, py, pz, py, pz, ty, tz, ty, tz, wx, wy, wz, ggs, GG[0, 0, :, :, :, nx, ny, nz], mat_xx)
                kernels.kernel2(nx, ny, nz, px, py, pz, 0, 1, 1, 1, 0, 1, bs0_x, bs1_y, bs1_z, bs1_x, bs0_y, bs1_z, py, pz, px, pz, ty, tz, tx, tz, wx, wy, wz, ggs, GG[0, 1, :, :, :, nx, ny, nz], mat_xy)
                kernels.kernel2(nx, ny, nz, px, py, pz, 0, 1, 1, 1, 1, 0, bs0_x, bs1_y, bs1_z, bs1_x, bs1_y, bs0_z, py, pz, px, py, ty, tz, tx, ty, wx, wy, wz, ggs, GG[0, 2, :, :, :, nx, ny, nz], mat_xz)
                kernels.kernel2(nx, ny, nz, px, py, pz, 1, 0, 1, 0, 1, 1, bs1_x, bs0_y, bs1_z, bs0_x, bs1_y, bs1_z, px, pz, py, pz, tx, tz, ty, tz, wx, wy, wz, ggs, GG[1, 0, :, :, :, nx, ny, nz], mat_yx)
                kernels.kernel2(nx, ny, nz, px, py, pz, 1, 0, 1, 1, 0, 1, bs1_x, bs0_y, bs1_z, bs1_x, bs0_y, bs1_z, px, pz, px, pz, tx, tz, tx, tz, wx, wy, wz, ggs, GG[1, 1, :, :, :, nx, ny, nz], mat_yy)
                kernels.kernel2(nx, ny, nz, px, py, pz, 1, 0, 1, 1, 1, 0, bs1_x, bs0_y, bs1_z, bs1_x, bs1_y, bs0_z, px, pz, px, py, tx, tz, tx, ty, wx, wy, wz, ggs, GG[1, 2, :, :, :, nx, ny, nz], mat_yz)
                kernels.kernel2(nx, ny, nz, px, py, pz, 1, 1, 0, 0, 1, 1, bs1_x, bs1_y, bs0_z, bs0_x, bs1_y, bs1_z, px, py, py, pz, tx, ty, ty, tz, wx, wy, wz, ggs, GG[2, 0, :, :, :, nx, ny, nz], mat_zx)
                kernels.kernel2(nx, ny, nz, px, py, pz, 1, 1, 0, 1, 0, 1, bs1_x, bs1_y, bs0_z, bs1_x, bs0_y, bs1_z, px, py, px, pz, tx, ty, tx, tz, wx, wy, wz, ggs, GG[2, 1, :, :, :, nx, ny, nz], mat_zy)
                kernels.kernel2(nx, ny, nz, px, py, pz, 1, 1, 0, 1, 1, 0, bs1_x, bs1_y, bs0_z, bs1_x, bs1_y, bs0_z, px, py, px, py, tx, ty, tx, ty, wx, wy, wz, ggs, GG[2, 2, :, :, :, nx, ny, nz], mat_zz)
                
                M_xx[nx:nx + px + 1, ny:ny + py, nz:nz + pz, nx:nx + px + 1, ny:ny + py, nz:nz + pz] += mat_xx[:, :, :, :, :, :]
                M_xy[nx:nx + px + 1, ny:ny + py, nz:nz + pz, nx:nx + px, ny:ny + py + 1, nz:nz + pz] += mat_xy[:, :, :, :, :, :]
                M_xz[nx:nx + px + 1, ny:ny + py, nz:nz + pz, nx:nx + px, ny:ny + py, nz:nz + pz + 1] += mat_xz[:, :, :, :, :, :]
                M_yx[nx:nx + px, ny:ny + py + 1, nz:nz + pz, nx:nx + px + 1, ny:ny + py, nz:nz + pz] += mat_yx[:, :, :, :, :, :]
                M_yy[nx:nx + px, ny:ny + py + 1, nz:nz + pz, nx:nx + px, ny:ny + py + 1, nz:nz + pz] += mat_yy[:, :, :, :, :, :]
                M_yz[nx:nx + px, ny:ny + py + 1, nz:nz + pz, nx:nx + px, ny:ny + py, nz:nz + pz + 1] += mat_yz[:, :, :, :, :, :]
                M_zx[nx:nx + px, ny:ny + py, nz:nz + pz + 1, nx:nx + px + 1, ny:ny + py, nz:nz + pz] += mat_zx[:, :, :, :, :, :]
                M_zy[nx:nx + px, ny:ny + py, nz:nz + pz + 1, nx:nx + px, ny:ny + py + 1, nz:nz + pz] += mat_zy[:, :, :, :, :, :]
                M_zz[nx:nx + px, ny:ny + py, nz:nz + pz + 1, nx:nx + px, ny:ny + py, nz:nz + pz + 1] += mat_zz[:, :, :, :, :, :]
    # ...   
    
    
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        M_xx[:px, :, :, :, :, :] += M_xx[-px:, :, :, :, :, :]
        M_xx[:, :, :, :px, :, :] += M_xx[:, :, :, -px:, :, :]
        M_xx = M_xx[:M_xx.shape[0] - px, :, :, :M_xx.shape[3] - px, :, :]

        M_xy[:px, :, :, :, :, :] += M_xy[-px:, :, :, :, :, :]
        M_xy[:, :, :, :px - 1, :, :] += M_xy[:, :, :, -px + 1:, :, :]
        M_xy = M_xy[:M_xy.shape[0] - px, :, :, :M_xy.shape[3] - px + 1, :, :]

        M_xz[:px, :, :, :, :, :] += M_xz[-px:, :, :, :, :, :]
        M_xz[:, :, :, :px - 1, :, :] += M_xz[:, :, :, -px + 1:, :, :]
        M_xz = M_xz[:M_xz.shape[0] - px, :, :, :M_xz.shape[3] - px + 1, :, :]

        M_yx[:px - 1, :, :, :, :, :] += M_yx[-px + 1:, :, :, :, :, :]
        M_yx[:, :, :, :px, :, :] += M_yx[:, :, :, -px:, :, :]
        M_yx = M_yx[:M_yx.shape[0] - px + 1, :, :, :M_yx.shape[3] - px, :, :]

        M_yy[:px - 1, :, :, :, :, :] += M_yy[-px + 1:, :, :, :, :, :]
        M_yy[:, :, :, :px - 1, :, :] += M_yy[:, :, :, -px + 1:, :, :]
        M_yy = M_yy[:M_yy.shape[0] - px + 1, :, :, :M_yy.shape[3] - px + 1, :, :]

        M_yz[:px - 1, :, :, :, :, :] += M_yz[-px + 1:, :, :, :, :, :]
        M_yz[:, :, :, :px - 1, :, :] += M_yz[:, :, :, -px + 1:, :, :]
        M_yz = M_yz[:M_yz.shape[0] - px + 1, :, :, :M_yz.shape[3] - px + 1, :, :]

        M_zx[:px - 1, :, :, :, :, :] += M_zx[-px + 1:, :, :, :, :, :]
        M_zx[:, :, :, :px, :, :] += M_zx[:, :, :, -px:, :, :]
        M_zx = M_zx[:M_zx.shape[0] - px + 1, :, :, :M_zx.shape[3] - px, :, :]

        M_zy[:px - 1, :, :, :, :, :] += M_zy[-px + 1:, :, :, :, :, :]
        M_zy[:, :, :, :px - 1, :, :] += M_zy[:, :, :, -px + 1:, :, :]
        M_zy = M_zy[:M_zy.shape[0] - px + 1, :, :, :M_zy.shape[3] - px + 1, :, :]

        M_zz[:px - 1, :, :, :, :, :] += M_zz[-px + 1:, :, :, :, :, :]
        M_zz[:, :, :, :px - 1, :, :] += M_zz[:, :, :, -px + 1:, :, :]
        M_zz = M_zz[:M_zz.shape[0] - px + 1, :, :, :M_zz.shape[3] - px + 1, :, :]

        Nbase_x_0_xx_i = Nbase_x - px
        Nbase_x_0_xx_j = Nbase_x - px

        Nbase_x_0_xy_i = Nbase_x - px
        Nbase_x_0_xy_j = Nbase_x - px

        Nbase_x_0_xz_i = Nbase_x - px
        Nbase_x_0_xz_j = Nbase_x - px

        Nbase_x_0_yx_i = Nbase_x - px
        Nbase_x_0_yx_j = Nbase_x - px

        Nbase_x_0_yy_i = Nbase_x - px
        Nbase_x_0_yy_j = Nbase_x - px

        Nbase_x_0_yz_i = Nbase_x - px
        Nbase_x_0_yz_j = Nbase_x - px

        Nbase_x_0_zx_i = Nbase_x - px
        Nbase_x_0_zx_j = Nbase_x - px

        Nbase_x_0_zy_i = Nbase_x - px
        Nbase_x_0_zy_j = Nbase_x - px

        Nbase_x_0_zz_i = Nbase_x - px
        Nbase_x_0_zz_j = Nbase_x - px

    elif bc_x == False:
        M_xx = M_xx[1:-1, :, :, 1:-1, :, :]
        M_xy = M_xy[1:-1, :, :, :, :, :]
        M_xz = M_xz[1:-1, :, :, :, :, :]
        M_yx = M_yx[:, :, :, 1:-1, :, :]
        M_zx = M_zx[:, :, :, 1:-1, :, :]
        
        Nbase_x_0_xx_i = Nbase_x - 2
        Nbase_x_0_xx_j = Nbase_x - 2

        Nbase_x_0_xy_i = Nbase_x - 2
        Nbase_x_0_xy_j = Nbase_x - 1

        Nbase_x_0_xz_i = Nbase_x - 2
        Nbase_x_0_xz_j = Nbase_x - 1

        Nbase_x_0_yx_i = Nbase_x - 1
        Nbase_x_0_yx_j = Nbase_x - 2

        Nbase_x_0_yy_i = Nbase_x - 1
        Nbase_x_0_yy_j = Nbase_x - 1

        Nbase_x_0_yz_i = Nbase_x - 1
        Nbase_x_0_yz_j = Nbase_x - 1

        Nbase_x_0_zx_i = Nbase_x - 1
        Nbase_x_0_zx_j = Nbase_x - 2

        Nbase_x_0_zy_i = Nbase_x - 1
        Nbase_x_0_zy_j = Nbase_x - 1

        Nbase_x_0_zz_i = Nbase_x - 1
        Nbase_x_0_zz_j = Nbase_x - 1

    else:
        Nbase_x_0_xx_i = Nbase_x 
        Nbase_x_0_xx_j = Nbase_x

        Nbase_x_0_xy_i = Nbase_x
        Nbase_x_0_xy_j = Nbase_x - 1

        Nbase_x_0_xz_i = Nbase_x
        Nbase_x_0_xz_j = Nbase_x - 1

        Nbase_x_0_yx_i = Nbase_x - 1
        Nbase_x_0_yx_j = Nbase_x

        Nbase_x_0_yy_i = Nbase_x - 1
        Nbase_x_0_yy_j = Nbase_x - 1

        Nbase_x_0_yz_i = Nbase_x - 1
        Nbase_x_0_yz_j = Nbase_x - 1

        Nbase_x_0_zx_i = Nbase_x - 1
        Nbase_x_0_zx_j = Nbase_x

        Nbase_x_0_zy_i = Nbase_x - 1
        Nbase_x_0_zy_j = Nbase_x - 1

        Nbase_x_0_zz_i = Nbase_x - 1
        Nbase_x_0_zz_j = Nbase_x - 1
    # ...
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)
    if bc_y == True:
        M_xx[:, :py - 1, :, :, :, :] += M_xx[:, -py + 1:, :, :, :, :]
        M_xx[:, :, :, :, :py - 1, :] += M_xx[:, :, :, :, -py + 1:, :]
        M_xx = M_xx[:, :M_xx.shape[1] - py + 1, :, :, :M_xx.shape[4] - py + 1, :]
        
        M_xy[:, :py - 1, :, :, :, :] += M_xy[:, -py + 1:, :, :, :, :]
        M_xy[:, :, :, :, :py, :] += M_xy[:, :, :, :, -py:, :]
        M_xy = M_xy[:, :M_xy.shape[1] - py + 1, :, :, :M_xy.shape[4] - py, :]
        
        M_xz[:, :py - 1, :, :, :, :] += M_xz[:, -py + 1:, :, :, :, :]
        M_xz[:, :, :, :, :py - 1, :] += M_xz[:, :, :, :, -py + 1:, :]
        M_xz = M_xz[:, :M_xz.shape[1] - py + 1, :, :, :M_xz.shape[4] - py + 1, :]
        
        M_yx[:, :py, :, :, :, :] += M_yx[:, -py:, :, :, :, :]
        M_yx[:, :, :, :, :py - 1, :] += M_yx[:, :, :, :, -py + 1:, :]
        M_yx = M_yx[:, :M_yx.shape[1] - py, :, :, :M_yx.shape[4] - py + 1, :]
        
        M_yy[:, :py, :, :, :, :] += M_yy[:, -py:, :, :, :, :]
        M_yy[:, :, :, :, :py, :] += M_yy[:, :, :, :, -py:, :]
        M_yy = M_yy[:, :M_yy.shape[1] - py, :, :, :M_yy.shape[4] - py, :]
        
        M_yz[:, :py, :, :, :, :] += M_yz[:, -py:, :, :, :, :]
        M_yz[:, :, :, :, :py - 1, :] += M_yz[:, :, :, :, -py + 1:, :]
        M_yz = M_yz[:, :M_yz.shape[1] - py, :, :, :M_yz.shape[4] - py + 1, :]
        
        M_zx[:, :py - 1, :, :, :, :] += M_zx[:, -py + 1:, :, :, :, :]
        M_zx[:, :, :, :, :py - 1, :] += M_zx[:, :, :, :, -py + 1:, :]
        M_zx = M_zx[:, :M_zx.shape[1] - py + 1, :, :, :M_zx.shape[4] - py + 1, :]
        
        M_zy[:, :py - 1, :, :, :, :] += M_zy[:, -py + 1:, :, :, :, :]
        M_zy[:, :, :, :, :py, :] += M_zy[:, :, :, :, -py:, :]
        M_zy = M_zy[:, :M_zy.shape[1] - py + 1, :, :, :M_zy.shape[4] - py, :]
        
        M_zz[:, :py - 1, :, :, :, :] += M_zz[:, -py + 1:, :, :, :, :]
        M_zz[:, :, :, :, :py - 1, :] += M_zz[:, :, :, :, -py + 1:, :]
        M_zz = M_zz[:, :M_zz.shape[1] - py + 1, :, :, :M_zz.shape[4] - py + 1, :]
        
        Nbase_y_0_xx_i = Nbase_y - py
        Nbase_y_0_xx_j = Nbase_y - py

        Nbase_y_0_xy_i = Nbase_y - py
        Nbase_y_0_xy_j = Nbase_y - py

        Nbase_y_0_xz_i = Nbase_y - py
        Nbase_y_0_xz_j = Nbase_y - py

        Nbase_y_0_yx_i = Nbase_y - py
        Nbase_y_0_yx_j = Nbase_y - py
        
        Nbase_y_0_yy_i = Nbase_y - py
        Nbase_y_0_yy_j = Nbase_y - py

        Nbase_y_0_yz_i = Nbase_y - py
        Nbase_y_0_yz_j = Nbase_y - py

        Nbase_y_0_zx_i = Nbase_y - py
        Nbase_y_0_zx_j = Nbase_y - py

        Nbase_y_0_zy_i = Nbase_y - py
        Nbase_y_0_zy_j = Nbase_y - py

        Nbase_y_0_zz_i = Nbase_y - py
        Nbase_y_0_zz_j = Nbase_y - py
        
    elif bc_y == False:
        M_xy = M_xy[:, :, :, :, 1:-1, :]
        M_yx = M_yx[:, 1:-1, :, :, :, :]
        M_yy = M_yy[:, 1:-1, :, :, 1:-1, :]
        M_yz = M_yz[:, 1:-1, :, :, :, :]
        M_zy = M_zy[:, :, :, :, 1:-1, :]
        
        Nbase_y_0_xx_i = Nbase_y - 1
        Nbase_y_0_xx_j = Nbase_y - 1

        Nbase_y_0_xy_i = Nbase_y - 1
        Nbase_y_0_xy_j = Nbase_y - 2

        Nbase_y_0_xz_i = Nbase_y - 1
        Nbase_y_0_xz_j = Nbase_y - 1

        Nbase_y_0_yx_i = Nbase_y - 2
        Nbase_y_0_yx_j = Nbase_y - 1
        
        Nbase_y_0_yy_i = Nbase_y - 2
        Nbase_y_0_yy_j = Nbase_y - 2

        Nbase_y_0_yz_i = Nbase_y - 2
        Nbase_y_0_yz_j = Nbase_y - 1

        Nbase_y_0_zx_i = Nbase_y - 1
        Nbase_y_0_zx_j = Nbase_y - 1

        Nbase_y_0_zy_i = Nbase_y - 1
        Nbase_y_0_zy_j = Nbase_y - 2

        Nbase_y_0_zz_i = Nbase_y - 1
        Nbase_y_0_zz_j = Nbase_y - 1
        
    else:
        Nbase_y_0_xx_i = Nbase_y - 1
        Nbase_y_0_xx_j = Nbase_y - 1

        Nbase_y_0_xy_i = Nbase_y - 1
        Nbase_y_0_xy_j = Nbase_y

        Nbase_y_0_xz_i = Nbase_y - 1
        Nbase_y_0_xz_j = Nbase_y - 1

        Nbase_y_0_yx_i = Nbase_y
        Nbase_y_0_yx_j = Nbase_y - 1
        
        Nbase_y_0_yy_i = Nbase_y
        Nbase_y_0_yy_j = Nbase_y

        Nbase_y_0_yz_i = Nbase_y 
        Nbase_y_0_yz_j = Nbase_y - 1

        Nbase_y_0_zx_i = Nbase_y - 1
        Nbase_y_0_zx_j = Nbase_y - 1

        Nbase_y_0_zy_i = Nbase_y - 1
        Nbase_y_0_zy_j = Nbase_y
 
        Nbase_y_0_zz_i = Nbase_y - 1
        Nbase_y_0_zz_j = Nbase_y - 1
    # ...
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)   
    if bc_z == True:
        M_xx[:, :, :pz - 1, :, :, :] += M_xx[:, :, -pz + 1:, :, :, :]
        M_xx[:, :, :, :, :, :pz - 1] += M_xx[:, :, :, :, :, -pz + 1:]
        M_xx = M_xx[:, :, :M_xx.shape[2] - pz + 1, :, :, :M_xx.shape[5] - pz + 1]
        
        M_xy[:, :, :pz - 1, :, :, :] += M_xy[:, :, -pz + 1:, :, :, :]
        M_xy[:, :, :, :, :, :pz - 1] += M_xy[:, :, :, :, :, -pz + 1:]
        M_xy = M_xy[:, :, :M_xy.shape[2] - pz + 1, :, :, :M_xy.shape[5] - pz + 1]
        
        M_xz[:, :, :pz - 1, :, :, :] += M_xz[:, :, -pz + 1:, :, :, :]
        M_xz[:, :, :, :, :, :pz] += M_xz[:, :, :, :, :, -pz:]
        M_xz = M_xz[:, :, :M_xz.shape[2] - pz + 1, :, :, :M_xz.shape[5] - pz]
        
        M_yx[:, :, :pz - 1, :, :, :] += M_yx[:, :, -pz + 1:, :, :, :]
        M_yx[:, :, :, :, :, :pz - 1] += M_yx[:, :, :, :, :, -pz + 1:]
        M_yx = M_yx[:, :, :M_yx.shape[2] - pz + 1, :, :, :M_yx.shape[5] - pz + 1]
        
        M_yy[:, :, :pz - 1, :, :, :] += M_yy[:, :, -pz + 1:, :, :, :]
        M_yy[:, :, :, :, :, :pz - 1] += M_yy[:, :, :, :, :, -pz + 1:]
        M_yy = M_yy[:, :, :M_yy.shape[2] - pz + 1, :, :, :M_yy.shape[5] - pz + 1]
        
        M_yz[:, :, :pz - 1, :, :, :] += M_yz[:, :, -pz + 1:, :, :, :]
        M_yz[:, :, :, :, :, :pz] += M_yz[:, :, :, :, :, -pz:]
        M_yz = M_yz[:, :, :M_yz.shape[2] - pz + 1, :, :, :M_yz.shape[5] - pz]
        
        M_zx[:, :, :pz, :, :, :] += M_zx[:, :, -pz:, :, :, :]
        M_zx[:, :, :, :, :, :pz - 1] += M_zx[:, :, :, :, :, -pz + 1:]
        M_zx = M_zx[:, :, :M_zx.shape[2] - pz, :, :, :M_zx.shape[5] - pz + 1]
        
        M_zy[:, :, :pz, :, :, :] += M_zy[:, :, -pz:, :, :, :]
        M_zy[:, :, :, :, :, :pz - 1] += M_zy[:, :, :, :, :, -pz + 1:]
        M_zy = M_zy[:, :, :M_zy.shape[2] - pz, :, :, :M_zy.shape[5] - pz + 1]
        
        M_zz[:, :, :pz, :, :, :] += M_zz[:, :, -pz:, :, :, :]
        M_zz[:, :, :, :, :, :pz] += M_zz[:, :, :, :, :, -pz:]
        M_zz = M_zz[:, :, :M_zz.shape[2] - pz, :, :, :M_zz.shape[5] - pz]
        
        Nbase_z_0_xx_i = Nbase_z - pz
        Nbase_z_0_xx_j = Nbase_z - pz

        Nbase_z_0_xy_i = Nbase_z - pz
        Nbase_z_0_xy_j = Nbase_z - pz

        Nbase_z_0_xz_i = Nbase_z - pz
        Nbase_z_0_xz_j = Nbase_z - pz
        
        Nbase_z_0_yx_i = Nbase_z - pz
        Nbase_z_0_yx_j = Nbase_z - pz
        
        Nbase_z_0_yy_i = Nbase_z - pz
        Nbase_z_0_yy_j = Nbase_z - pz

        Nbase_z_0_yz_i = Nbase_z - pz
        Nbase_z_0_yz_j = Nbase_z - pz

        Nbase_z_0_zx_i = Nbase_z - pz
        Nbase_z_0_zx_j = Nbase_z - pz

        Nbase_z_0_zy_i = Nbase_z - pz
        Nbase_z_0_zy_j = Nbase_z - pz

        Nbase_z_0_zz_i = Nbase_z - pz
        Nbase_z_0_zz_j = Nbase_z - pz
        
    elif bc_z == False:
        M_xz = M_xz[:, :, :, :, :, 1:-1]
        M_yz = M_yz[:, :, :, :, :, 1:-1]
        M_zx = M_zx[:, :, 1:-1, :, :, :]
        M_zy = M_zy[:, :, 1:-1, :, :, :]
        M_zz = M_zz[:, :, 1:-1, :, :, 1:-1]
       
        Nbase_z_0_xx_i = Nbase_z - 1
        Nbase_z_0_xx_j = Nbase_z - 1

        Nbase_z_0_xy_i = Nbase_z - 1
        Nbase_z_0_xy_j = Nbase_z - 1

        Nbase_z_0_xz_i = Nbase_z - 1
        Nbase_z_0_xz_j = Nbase_z - 2
        
        Nbase_z_0_yx_i = Nbase_z - 1
        Nbase_z_0_yx_j = Nbase_z - 1
        
        Nbase_z_0_yy_i = Nbase_z - 1
        Nbase_z_0_yy_j = Nbase_z - 1

        Nbase_z_0_yz_i = Nbase_z - 2
        Nbase_z_0_yz_j = Nbase_z - 1

        Nbase_z_0_zx_i = Nbase_z - 2
        Nbase_z_0_zx_j = Nbase_z - 1

        Nbase_z_0_zy_i = Nbase_z - 2
        Nbase_z_0_zy_j = Nbase_z - 1

        Nbase_z_0_zz_i = Nbase_z - 2
        Nbase_z_0_zz_j = Nbase_z - 2
        
    else:
        Nbase_z_0_xx_i = Nbase_z - 1
        Nbase_z_0_xx_j = Nbase_z - 1

        Nbase_z_0_xy_i = Nbase_z - 1
        Nbase_z_0_xy_j = Nbase_z - 1

        Nbase_z_0_xz_i = Nbase_z - 1
        Nbase_z_0_xz_j = Nbase_z
        
        Nbase_z_0_yx_i = Nbase_z - 1
        Nbase_z_0_yx_j = Nbase_z - 1
        
        Nbase_z_0_yy_i = Nbase_z - 1
        Nbase_z_0_yy_j = Nbase_z - 1

        Nbase_z_0_yz_i = Nbase_z - 1
        Nbase_z_0_yz_j = Nbase_z

        Nbase_z_0_zx_i = Nbase_z
        Nbase_z_0_zx_j = Nbase_z - 1

        Nbase_z_0_zy_i = Nbase_z 
        Nbase_z_0_zy_j = Nbase_z - 1

        Nbase_z_0_zz_i = Nbase_z
        Nbase_z_0_zz_j = Nbase_z
    # ...
    
    M_xx = sparse.csr_matrix(np.reshape(M_xx, (Nbase_x_0_xx_i*Nbase_y_0_xx_i*Nbase_z_0_xx_i, Nbase_x_0_xx_j*Nbase_y_0_xx_j*Nbase_z_0_xx_j)))
    M_xy = sparse.csr_matrix(np.reshape(M_xy, (Nbase_x_0_xy_i*Nbase_y_0_xy_i*Nbase_z_0_xy_i, Nbase_x_0_xy_j*Nbase_y_0_xy_j*Nbase_z_0_xy_j)))
    M_xz = sparse.csr_matrix(np.reshape(M_xz, (Nbase_x_0_xz_i*Nbase_y_0_xz_i*Nbase_z_0_xz_i, Nbase_x_0_xz_j*Nbase_y_0_xz_j*Nbase_z_0_xz_j)))

    M_yx = sparse.csr_matrix(np.reshape(M_yx, (Nbase_x_0_yx_i*Nbase_y_0_yx_i*Nbase_z_0_yx_i, Nbase_x_0_yx_j*Nbase_y_0_yx_j*Nbase_z_0_yx_j)))
    M_yy = sparse.csr_matrix(np.reshape(M_yy, (Nbase_x_0_yy_i*Nbase_y_0_yy_i*Nbase_z_0_yy_i, Nbase_x_0_yy_j*Nbase_y_0_yy_j*Nbase_z_0_yy_j)))
    M_yz = sparse.csr_matrix(np.reshape(M_yz, (Nbase_x_0_yz_i*Nbase_y_0_yz_i*Nbase_z_0_yz_i, Nbase_x_0_yz_j*Nbase_y_0_yz_j*Nbase_z_0_yz_j)))

    M_zx = sparse.csr_matrix(np.reshape(M_zx, (Nbase_x_0_zx_i*Nbase_y_0_zx_i*Nbase_z_0_zx_i, Nbase_x_0_zx_j*Nbase_y_0_zx_j*Nbase_z_0_zx_j)))
    M_zy = sparse.csr_matrix(np.reshape(M_zy, (Nbase_x_0_zy_i*Nbase_y_0_zy_i*Nbase_z_0_zy_i, Nbase_x_0_zy_j*Nbase_y_0_zy_j*Nbase_z_0_zy_j)))
    M_zz = sparse.csr_matrix(np.reshape(M_zz, (Nbase_x_0_zz_i*Nbase_y_0_zz_i*Nbase_z_0_zz_i, Nbase_x_0_zz_j*Nbase_y_0_zz_j*Nbase_z_0_zz_j)))
                    
    return sparse.bmat([[M_xx, M_xy, M_xz], [M_yx, M_yy, M_yz], [M_zx, M_zy, M_zz]], format='csr')



def mass_matrix_V3(p, Nbase, T, g, bc):
    """
    Computes the mass matrix of the space V3 in general curvilinear coordinates q = (q1, q2, q3) with metric tensor G.
    
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
        mass matrix in V3
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
    
    basis1_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px + 1, 0, tx, pts_x) 
    basis1_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py + 1, 0, ty, pts_y) 
    basis1_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz + 1, 0, tz, pts_z)
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.empty((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nz in range(Nel_z):
        for ny in range(Nel_y):
            for nx in range(Nel_x):
                for gz in range(pz + 1):
                    for gy in range(py + 1):
                        for gx in range(px + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
    
    
    # ... global matrix and local element matrix
    M = np.zeros((Nbase_x - 1, Nbase_y - 1, Nbase_z - 1, Nbase_x - 1, Nbase_y - 1, Nbase_z - 1))
    mat = np.zeros((px, py, pz, px, py, pz), order='F')
    # ...
    
    
    # ... assembly
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                
                wx = wts_x[:, nx]
                wy = wts_y[:, ny]
                wz = wts_z[:, nz]
                
                bs1_x = basis1_x[:, 0, :, nx]
                bs1_y = basis1_y[:, 0, :, ny]
                bs1_z = basis1_z[:, 0, :, nz]
                
                ggs = gg[:, :, :, nx, ny, nz]
                
                kernels.kernel3(nx, ny, nz, px, py, pz, bs1_x, bs1_y, bs1_z, tx, ty, tz, wx, wy, wz, ggs, mat)
                
                M[nx:nx + px, ny:ny + py, nz:nz + pz, nx:nx + px, ny:ny + py, nz:nz + pz] += mat[:, :, :, :, :, :]
    # ...   
    
    
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        M[:px - 1, :, :, :, :, :] += M[-px + 1:, :, :, :, :, :]
        M[:, :, :, :px - 1, :, :] += M[:, :, :, -px + 1:, :, :]
        M = M[:M.shape[0] - px + 1, :, :, :M.shape[3] - px + 1, :, :]

        Nbase_x_0_i = Nbase_x - px
        Nbase_x_0_j = Nbase_x - px

    else:
        Nbase_x_0_i = Nbase_x - 1
        Nbase_x_0_j = Nbase_x - 1
    # ...
    
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)
    if bc_y == True:
        M[:, :py - 1, :, :, :, :] += M[:, -py + 1:, :, :, :, :]
        M[:, :, :, :, :py - 1, :] += M[:, :, :, :, -py + 1:, :]
        M = M[:, :M.shape[1] - py + 1, :, :, :M.shape[4] - py + 1, :]
        
        Nbase_y_0_i = Nbase_y - py
        Nbase_y_0_j = Nbase_y - py

    else:
        Nbase_y_0_i = Nbase_y - 1
        Nbase_y_0_j = Nbase_y - 1
    # ...
    
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)   
    if bc_z == True:
        M[:, :, :pz - 1, :, :, :] += M[:, :, -pz + 1:, :, :, :]
        M[:, :, :, :, :, :pz - 1] += M[:, :, :, :, :, -pz + 1:]
        M = M[:, :, :M.shape[2] - pz + 1, :, :, :M.shape[5] - pz + 1]
        
        Nbase_z_0_i = Nbase_z - pz
        Nbase_z_0_j = Nbase_z - pz
        
    else:
        Nbase_z_0_i = Nbase_z - 1
        Nbase_z_0_j = Nbase_z - 1
    # ...
    
    M = sparse.csr_matrix(np.reshape(M, (Nbase_x_0_i*Nbase_y_0_i*Nbase_z_0_i, Nbase_x_0_j*Nbase_y_0_j*Nbase_z_0_j)))
                    
    return M




def L2_prod_V0(fun, p, Nbase, T, g, bc):
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
    bc_x, bc_y, bc_z = bc
    
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
    gg = np.empty((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    ff = np.empty((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                for gx in range(px + 1):
                    for gy in range(py + 1):
                        for gz in range(pz + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            ff[gx, gy, gz, nx, ny, nz] = fun(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
    
    basis0_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px + 1, 0, Tx, pts_x)
    basis0_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py + 1, 0, Ty, pts_y)
    basis0_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz + 1, 0, Tz, pts_z)
    
    # ... global vector and local element vector
    f_int = np.zeros((Nbase_x, Nbase_y, Nbase_z))
    mat = np.zeros((px + 1, py + 1, pz + 1), order='F')
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
                
                ggs = gg[:, :, :, nx, ny, nz]
                ffs = ff[:, :, :, nx, ny, nz]
                
                kernels.kernelL0(px, py, pz, bs0_x, bs0_y, bs0_z, wx, wy, wz, ggs, ffs, mat)
                
                f_int[nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz + 1] += mat[:, :, :]
    # ...
    
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        f_int[:px, :, :] += f_int[-px:, :, :]
        f_int = f_int[:f_int.shape[0] - px, :, :]

        Nbase_x_0_i = Nbase_x - px

    elif bc_x == False:
        f_int = f_int[1:-1, :, :]
        
        Nbase_x_0_i = Nbase_x - 2

    else:
        Nbase_x_0_i = Nbase_x
    # ...
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)
    if bc_y == True:
        f_int[:, :py, :] += f_int[:, -py:, :]
        f_int = f_int[:, :f_int.shape[1] - py, :]
        
        Nbase_y_0_i = Nbase_y - py
     
    elif bc_y == False:
        f_int = f_int[:, 1:-1, :]
        
        Nbase_y_0_i = Nbase_y - 2
        
    else:
        Nbase_y_0_i = Nbase_y 
    # ...
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)   
    if bc_z == True:
        f_int[:, :, :pz] += f_int[:, :, -pz:]
        f_int = f_int[:, :, :f_int.shape[2] - pz]
        
        Nbase_z_0_i = Nbase_z - pz

    elif bc_z == False:
        f_int = f_int[:, :, 1:-1]
        
        Nbase_z_0_i = Nbase_z - 2
        
    else:
        Nbase_z_0_i = Nbase_z
    # ...
    
    return np.reshape(f_int, Nbase_x_0_i*Nbase_y_0_i*Nbase_z_0_i)



def L2_prod_V1(fun, p, Nbase, T, Ginv, g, bc):
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
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.empty((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    ff = np.empty((3, px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    GG = np.empty((3, 3, px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nz in range(Nel_z):
        for ny in range(Nel_y):
            for nx in range(Nel_x):
                for gz in range(pz + 1):
                    for gy in range(py + 1):
                        for gx in range(px + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            
                            for d2 in range(3):
                                ff[d2, gx, gy, gz, nx, ny, nz] = fun[d2](pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                                
                                for d1 in range(3):
                                    GG[d1, d2, gx, gy, gz, nx, ny, nz] = Ginv[d1][d2](pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
    
    basis0_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px + 1, 0, Tx, pts_x)
    basis0_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py + 1, 0, Ty, pts_y)
    basis0_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz + 1, 0, Tz, pts_z)
    
    basis1_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px + 1, 0, tx, pts_x) 
    basis1_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py + 1, 0, ty, pts_y) 
    basis1_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz + 1, 0, tz, pts_z)
    
    # ... global vectors and local element vectors
    f_int_xx = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z))
    f_int_xy = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z))
    f_int_xz = np.zeros((Nbase_x, Nbase_y, Nbase_z - 1))
    
    f_int_yx = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z))
    f_int_yy = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z))
    f_int_yz = np.zeros((Nbase_x, Nbase_y, Nbase_z - 1))
    
    f_int_zx = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z))
    f_int_zy = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z))
    f_int_zz = np.zeros((Nbase_x, Nbase_y, Nbase_z - 1))
    
    mat_xx = np.zeros((px, py + 1, pz + 1), order='F')
    mat_xy = np.zeros((px + 1, py, pz + 1), order='F')
    mat_xz = np.zeros((px + 1, py + 1, pz), order='F')
    
    mat_yx = np.zeros((px, py + 1, pz + 1), order='F')
    mat_yy = np.zeros((px + 1, py, pz + 1), order='F')
    mat_yz = np.zeros((px + 1, py + 1, pz), order='F')
    
    mat_zx = np.zeros((px, py + 1, pz + 1), order='F')
    mat_zy = np.zeros((px + 1, py, pz + 1), order='F')
    mat_zz = np.zeros((px + 1, py + 1, pz), order='F')
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
                
                kernels.kernelL1(nx, ny, nz, px, py, pz, 1, 0, 0, bs1_x, bs0_y, bs0_z, px, tx, wx, wy, wz, ggs, GG[0, 0, :, :, :, nx, ny, nz], ff[0, :, :, :, nx, ny, nz], mat_xx)
                kernels.kernelL1(nx, ny, nz, px, py, pz, 0, 1, 0, bs0_x, bs1_y, bs0_z, py, ty, wx, wy, wz, ggs, GG[0, 1, :, :, :, nx, ny, nz], ff[0, :, :, :, nx, ny, nz], mat_xy)
                kernels.kernelL1(nx, ny, nz, px, py, pz, 0, 0, 1, bs0_x, bs0_y, bs1_z, pz, tz, wx, wy, wz, ggs, GG[0, 2, :, :, :, nx, ny, nz], ff[0, :, :, :, nx, ny, nz], mat_xz)
                
                kernels.kernelL1(nx, ny, nz, px, py, pz, 1, 0, 0, bs1_x, bs0_y, bs0_z, px, tx, wx, wy, wz, ggs, GG[1, 0, :, :, :, nx, ny, nz], ff[1, :, :, :, nx, ny, nz], mat_yx)
                kernels.kernelL1(nx, ny, nz, px, py, pz, 0, 1, 0, bs0_x, bs1_y, bs0_z, py, ty, wx, wy, wz, ggs, GG[1, 1, :, :, :, nx, ny, nz], ff[1, :, :, :, nx, ny, nz], mat_yy)
                kernels.kernelL1(nx, ny, nz, px, py, pz, 0, 0, 1, bs0_x, bs0_y, bs1_z, pz, tz, wx, wy, wz, ggs, GG[1, 2, :, :, :, nx, ny, nz], ff[1, :, :, :, nx, ny, nz], mat_yz)
                
                kernels.kernelL1(nx, ny, nz, px, py, pz, 1, 0, 0, bs1_x, bs0_y, bs0_z, px, tx, wx, wy, wz, ggs, GG[2, 0, :, :, :, nx, ny, nz], ff[2, :, :, :, nx, ny, nz], mat_zx)
                kernels.kernelL1(nx, ny, nz, px, py, pz, 0, 1, 0, bs0_x, bs1_y, bs0_z, py, ty, wx, wy, wz, ggs, GG[2, 1, :, :, :, nx, ny, nz], ff[2, :, :, :, nx, ny, nz], mat_zy)
                kernels.kernelL1(nx, ny, nz, px, py, pz, 0, 0, 1, bs0_x, bs0_y, bs1_z, pz, tz, wx, wy, wz, ggs, GG[2, 2, :, :, :, nx, ny, nz], ff[2, :, :, :, nx, ny, nz], mat_zz)
                
                f_int_xx[nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1] += mat_xx[:, :, :]
                f_int_xy[nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1] += mat_xy[:, :, :]
                f_int_xz[nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz] += mat_xz[:, :, :]
                
                f_int_yx[nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1] += mat_yx[:, :, :]
                f_int_yy[nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1] += mat_yy[:, :, :]
                f_int_yz[nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz] += mat_yz[:, :, :]
                
                f_int_zx[nx:nx + px, ny:ny + py + 1, nz:nz + pz + 1] += mat_zx[:, :, :]
                f_int_zy[nx:nx + px + 1, ny:ny + py, nz:nz + pz + 1] += mat_zy[:, :, :]
                f_int_zz[nx:nx + px + 1, ny:ny + py + 1, nz:nz + pz] += mat_zz[:, :, :]
    # ...
                
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        f_int_xx[:px - 1, :, :] += f_int_xx[-px + 1:, :, :]
        f_int_xx = f_int_xx[:f_int_xx.shape[0] - px + 1, :, :]

        f_int_xy[:px, :, :] += f_int_xy[-px:, :, :]
        f_int_xy = f_int_xy[:f_int_xy.shape[0] - px, :, :]

        f_int_xz[:px, :, :] += f_int_xz[-px:, :, :]
        f_int_xz = f_int_xz[:f_int_xz.shape[0] - px, :, :]

        f_int_yx[:px - 1, :, :] += f_int_yx[-px + 1:, :, :]
        f_int_yx = f_int_yx[:f_int_yx.shape[0] - px + 1, :, :]

        f_int_yy[:px, :, :] += f_int_yy[-px:, :, :]
        f_int_yy = f_int_yy[:f_int_yy.shape[0] - px, :, :]

        f_int_yz[:px, :, :] += f_int_yz[-px:, :, :]
        f_int_yz = f_int_yz[:f_int_yz.shape[0] - px, :, :]

        f_int_zx[:px - 1, :, :] += f_int_zx[-px + 1:, :, :]
        f_int_zx = f_int_zx[:f_int_zx.shape[0] - px + 1, :, :]

        f_int_zy[:px, :, :] += f_int_zy[-px:, :, :]
        f_int_zy = f_int_zy[:f_int_zy.shape[0] - px, :, :]

        f_int_zz[:px, :, :] += f_int_zz[-px:, :, :]
        f_int_zz = f_int_zz[:f_int_zz.shape[0] - px, :, :]

        Nbase_x_0_xx_i = Nbase_x - px
        Nbase_x_0_xy_i = Nbase_x - px
        Nbase_x_0_xz_i = Nbase_x - px
        
        Nbase_x_0_yx_i = Nbase_x - px
        Nbase_x_0_yy_i = Nbase_x - px
        Nbase_x_0_yz_i = Nbase_x - px
        
        Nbase_x_0_zx_i = Nbase_x - px
        Nbase_x_0_zy_i = Nbase_x - px
        Nbase_x_0_zz_i = Nbase_x - px

    elif bc_x == False:
        f_int_xy = f_int_xy[1:-1, :, :]
        f_int_xz = f_int_xz[1:-1, :, :]
        
        f_int_yy = f_int_yy[1:-1, :, :]
        f_int_yz = f_int_yz[1:-1, :, :]
        
        f_int_zy = f_int_zy[1:-1, :, :]
        f_int_zz  =f_int_zz[1:-1, :, :]
        
        Nbase_x_0_xx_i = Nbase_x - 1
        Nbase_x_0_xy_i = Nbase_x - 2
        Nbase_x_0_xz_i = Nbase_x - 2

        Nbase_x_0_yx_i = Nbase_x - 1
        Nbase_x_0_yy_i = Nbase_x - 2
        Nbase_x_0_yz_i = Nbase_x - 2

        Nbase_x_0_zx_i = Nbase_x - 1
        Nbase_x_0_zy_i = Nbase_x - 2
        Nbase_x_0_zz_i = Nbase_x - 2

    else:
        Nbase_x_0_xx_i = Nbase_x - 1
        Nbase_x_0_xy_i = Nbase_x
        Nbase_x_0_xz_i = Nbase_x

        Nbase_x_0_yx_i = Nbase_x - 1
        Nbase_x_0_yy_i = Nbase_x
        Nbase_x_0_yz_i = Nbase_x

        Nbase_x_0_zx_i = Nbase_x - 1
        Nbase_x_0_zy_i = Nbase_x
        Nbase_x_0_zz_i = Nbase_x
    # ...
    
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)
    if bc_y == True:
        f_int_xx[:, :py, :] += f_int_xx[:, -py:, :]
        f_int_xx = f_int_xx[:, :f_int_xx.shape[1] - py, :]

        f_int_xy[:, :py - 1, :] += f_int_xy[:, -py + 1:, :]
        f_int_xy = f_int_xy[:, :f_int_xy.shape[1] - py + 1, :]

        f_int_xz[:, :py, :] += f_int_xz[:, -py:, :]
        f_int_xz = f_int_xz[:, :f_int_xz.shape[1] - py, :]

        f_int_yx[:, :py, :] += f_int_yx[:, -py:, :]
        f_int_yx = f_int_yx[:, :f_int_yx.shape[1] - py, :]

        f_int_yy[:, :py - 1, :] += f_int_yy[:, -py + 1:, :]
        f_int_yy = f_int_yy[:, :f_int_yy.shape[1] - py + 1, :]

        f_int_yz[:, :py, :] += f_int_yz[:, -py:, :]
        f_int_yz = f_int_yz[:, :f_int_yz.shape[1] - py, :]

        f_int_zx[:, :py, :] += f_int_zx[:, -py:, :]
        f_int_zx = f_int_zx[:, :f_int_zx.shape[1] - py, :]

        f_int_zy[:, :py - 1, :] += f_int_zy[:, -py + 1:, :]
        f_int_zy = f_int_zy[:, :f_int_zy.shape[1] - py + 1, :]

        f_int_zz[:, :py, :] += f_int_zz[:, -py:, :]
        f_int_zz = f_int_zz[:, :f_int_zz.shape[1] - py, :]

        Nbase_y_0_xx_i = Nbase_y - py
        Nbase_y_0_xy_i = Nbase_y - py
        Nbase_y_0_xz_i = Nbase_y - py
    
        Nbase_y_0_yx_i = Nbase_y - py
        Nbase_y_0_yy_i = Nbase_y - py
        Nbase_y_0_yz_i = Nbase_y - py
        
        Nbase_y_0_zx_i = Nbase_y - py
        Nbase_y_0_zy_i = Nbase_y - py
        Nbase_y_0_zz_i = Nbase_y - py

    elif bc_y == False:
        f_int_xx = f_int_xx[:, 1:-1, :]
        f_int_xz = f_int_xz[:, 1:-1, :]
        
        f_int_yx = f_int_yx[:, 1:-1, :]
        f_int_yz = f_int_yz[:, 1:-1, :]
        
        f_int_zx = f_int_zx[:, 1:-1, :]
        f_int_zz = f_int_zz[:, 1:-1, :]
        
        Nbase_y_0_xx_i = Nbase_y - 2
        Nbase_y_0_xy_i = Nbase_y - 1
        Nbase_y_0_xz_i = Nbase_y - 2

        Nbase_y_0_yx_i = Nbase_y - 2
        Nbase_y_0_yy_i = Nbase_y - 1
        Nbase_y_0_yz_i = Nbase_y - 2

        Nbase_y_0_zx_i = Nbase_y - 2
        Nbase_y_0_zy_i = Nbase_y - 1
        Nbase_y_0_zz_i = Nbase_y - 2

    else:
        Nbase_y_0_xx_i = Nbase_y
        Nbase_y_0_xy_i = Nbase_y - 1
        Nbase_y_0_xz_i = Nbase_y
        
        Nbase_y_0_yx_i = Nbase_y
        Nbase_y_0_yy_i = Nbase_y - 1
        Nbase_y_0_yz_i = Nbase_y

        Nbase_y_0_zx_i = Nbase_y
        Nbase_y_0_zy_i = Nbase_y - 1
        Nbase_y_0_zz_i = Nbase_y
    # ...
    
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)
    if bc_z == True:
        f_int_xx[:, :, :pz] += f_int_xx[:, :, -pz:]
        f_int_xx = f_int_xx[:, :, :f_int_xx.shape[2] - pz]

        f_int_xy[:, :, :pz] += f_int_xy[:, :, -pz:]
        f_int_xy = f_int_xy[:, :, :f_int_xy.shape[2] - pz]

        f_int_xz[:, :, :pz - 1] += f_int_xz[:, :, -pz + 1:]
        f_int_xz = f_int_xz[:, :, :f_int_xz.shape[2] - pz + 1]

        f_int_yx[:, :, :pz] += f_int_yx[:, :, -pz:]
        f_int_yx = f_int_yx[:, :, :f_int_yx.shape[2] - pz]

        f_int_yy[:, :, :pz] += f_int_yy[:, :, -pz:]
        f_int_yy = f_int_yy[:, :, :f_int_yy.shape[2] - pz]

        f_int_yz[:, :, :pz - 1] += f_int_yz[:, :, -pz + 1:]
        f_int_yz = f_int_yz[:, :, :f_int_yz.shape[2] - pz + 1]

        f_int_zx[:, :, :pz] += f_int_zx[:, :, -pz:]
        f_int_zx = f_int_zx[:, :, :f_int_zx.shape[2] - pz]

        f_int_zy[:, :, :pz] += f_int_zy[:, :, -pz:]
        f_int_zy = f_int_zy[:, :, :f_int_zy.shape[2] - pz]

        f_int_zz[:, :, :pz - 1] += f_int_zz[:, :, -pz + 1:]
        f_int_zz = f_int_zz[:, :, :f_int_zz.shape[2] - pz + 1]

        Nbase_z_0_xx_i = Nbase_z - pz
        Nbase_z_0_xy_i = Nbase_z - pz
        Nbase_z_0_xz_i = Nbase_z - pz
    
        Nbase_z_0_yx_i = Nbase_z - pz
        Nbase_z_0_yy_i = Nbase_z - pz
        Nbase_z_0_yz_i = Nbase_z - pz
        
        Nbase_z_0_zx_i = Nbase_z - pz
        Nbase_z_0_zy_i = Nbase_z - pz
        Nbase_z_0_zz_i = Nbase_z - pz

    elif bc_z == False:
        f_int_xx = f_int_xx[:, :, 1:-1]
        f_int_xy = f_int_xy[:, :, 1:-1]
        
        f_int_yx = f_int_yx[:, :, 1:-1]
        f_int_yy = f_int_yy[:, :, 1:-1]
        
        f_int_zx = f_int_zx[:, :, 1:-1]
        f_int_zy = f_int_zy[:, :, 1:-1]
        
        Nbase_z_0_xx_i = Nbase_z - 2
        Nbase_z_0_xy_i = Nbase_z - 2
        Nbase_z_0_xz_i = Nbase_z - 1
    
        Nbase_z_0_yx_i = Nbase_z - 2
        Nbase_z_0_yy_i = Nbase_z - 2
        Nbase_z_0_yz_i = Nbase_z - 1
        
        Nbase_z_0_zx_i = Nbase_z - 2
        Nbase_z_0_zy_i = Nbase_z - 2
        Nbase_z_0_zz_i = Nbase_z - 1

    else:
        Nbase_z_0_xx_i = Nbase_z
        Nbase_z_0_xy_i = Nbase_z
        Nbase_z_0_xz_i = Nbase_z - 1
    
        Nbase_z_0_yx_i = Nbase_z
        Nbase_z_0_yy_i = Nbase_z
        Nbase_z_0_yz_i = Nbase_z - 1
        
        Nbase_z_0_zx_i = Nbase_z
        Nbase_z_0_zy_i = Nbase_z
        Nbase_z_0_zz_i = Nbase_z - 1
    # ...
    
    
    f_int_xx = np.reshape(f_int_xx, Nbase_x_0_xx_i*Nbase_y_0_xx_i*Nbase_z_0_xx_i)
    f_int_xy = np.reshape(f_int_xy, Nbase_x_0_xy_i*Nbase_y_0_xy_i*Nbase_z_0_xy_i)
    f_int_xz = np.reshape(f_int_xz, Nbase_x_0_xz_i*Nbase_y_0_xz_i*Nbase_z_0_xz_i)
    f_int_yx = np.reshape(f_int_yx, Nbase_x_0_yx_i*Nbase_y_0_yx_i*Nbase_z_0_yx_i)
    f_int_yy = np.reshape(f_int_yy, Nbase_x_0_yy_i*Nbase_y_0_yy_i*Nbase_z_0_yy_i)
    f_int_yz = np.reshape(f_int_yz, Nbase_x_0_yz_i*Nbase_y_0_yz_i*Nbase_z_0_yz_i)
    f_int_zx = np.reshape(f_int_zx, Nbase_x_0_zx_i*Nbase_y_0_zx_i*Nbase_z_0_zx_i)
    f_int_zy = np.reshape(f_int_zy, Nbase_x_0_zy_i*Nbase_y_0_zy_i*Nbase_z_0_zy_i)
    f_int_zz = np.reshape(f_int_zz, Nbase_x_0_zz_i*Nbase_y_0_zz_i*Nbase_z_0_zz_i)
    
    return np.concatenate((f_int_xx + f_int_yx + f_int_zx, f_int_xy + f_int_yy + f_int_zy, f_int_xz + f_int_yz + f_int_zz))




def L2_prod_V2(fun, p, Nbase, T, G, g, bc):
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
        
    G : callable
        the inverse of the metric tensor G
        
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
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.empty((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    ff = np.empty((3, px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    GG = np.empty((3, 3, px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nz in range(Nel_z):
        for ny in range(Nel_y):
            for nx in range(Nel_x):
                for gz in range(pz + 1):
                    for gy in range(py + 1):
                        for gx in range(px + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            
                            for d2 in range(3):
                                ff[d2, gx, gy, gz, nx, ny, nz] = fun[d2](pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                                
                                for d1 in range(3):
                                    GG[d1, d2, gx, gy, gz, nx, ny, nz] = G[d1][d2](pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
    
    basis0_x = inter.eval_on_grid_splines_ders(px, Nbase_x, px + 1, 0, Tx, pts_x)
    basis0_y = inter.eval_on_grid_splines_ders(py, Nbase_y, py + 1, 0, Ty, pts_y)
    basis0_z = inter.eval_on_grid_splines_ders(pz, Nbase_z, pz + 1, 0, Tz, pts_z)
    
    basis1_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px + 1, 0, tx, pts_x) 
    basis1_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py + 1, 0, ty, pts_y) 
    basis1_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz + 1, 0, tz, pts_z)
    
    # ... global vectors and local element vectors
    f_int_xx = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z - 1))
    f_int_xy = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z - 1))
    f_int_xz = np.zeros((Nbase_x - 1, Nbase_y - 1, Nbase_z))
    
    f_int_yx = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z - 1))
    f_int_yy = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z - 1))
    f_int_yz = np.zeros((Nbase_x - 1, Nbase_y - 1, Nbase_z))
    
    f_int_zx = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z - 1))
    f_int_zy = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z - 1))
    f_int_zz = np.zeros((Nbase_x - 1, Nbase_y - 1, Nbase_z))
    
    mat_xx = np.zeros((px + 1, py, pz), order='F')
    mat_xy = np.zeros((px, py + 1, pz), order='F')
    mat_xz = np.zeros((px, py, pz + 1), order='F')
    
    mat_yx = np.zeros((px + 1, py, pz), order='F')
    mat_yy = np.zeros((px, py + 1, pz), order='F')
    mat_yz = np.zeros((px, py, pz + 1), order='F')
    
    mat_zx = np.zeros((px + 1, py, pz), order='F')
    mat_zy = np.zeros((px, py + 1, pz), order='F')
    mat_zz = np.zeros((px, py, pz + 1), order='F')
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
                
                kernels.kernelL2(nx, ny, nz, px, py, pz, 0, 1, 1, bs0_x, bs1_y, bs1_z, py, pz, ty, tz, wx, wy, wz, ggs, GG[0, 0, :, :, :, nx, ny, nz], ff[0, :, :, :, nx, ny, nz], mat_xx)
                kernels.kernelL2(nx, ny, nz, px, py, pz, 1, 0, 1, bs1_x, bs0_y, bs1_z, px, pz, tx, tz, wx, wy, wz, ggs, GG[0, 1, :, :, :, nx, ny, nz], ff[0, :, :, :, nx, ny, nz], mat_xy)
                kernels.kernelL2(nx, ny, nz, px, py, pz, 1, 1, 0, bs1_x, bs1_y, bs0_z, px, py, tx, ty, wx, wy, wz, ggs, GG[0, 2, :, :, :, nx, ny, nz], ff[0, :, :, :, nx, ny, nz], mat_xz)
                
                kernels.kernelL2(nx, ny, nz, px, py, pz, 0, 1, 1, bs0_x, bs1_y, bs1_z, py, pz, ty, tz, wx, wy, wz, ggs, GG[1, 0, :, :, :, nx, ny, nz], ff[1, :, :, :, nx, ny, nz], mat_yx)
                kernels.kernelL2(nx, ny, nz, px, py, pz, 1, 0, 1, bs1_x, bs0_y, bs1_z, px, pz, tx, tz, wx, wy, wz, ggs, GG[1, 1, :, :, :, nx, ny, nz], ff[1, :, :, :, nx, ny, nz], mat_yy)
                kernels.kernelL2(nx, ny, nz, px, py, pz, 1, 1, 0, bs1_x, bs1_y, bs0_z, px, py, tx, ty, wx, wy, wz, ggs, GG[1, 2, :, :, :, nx, ny, nz], ff[1, :, :, :, nx, ny, nz], mat_yz)
                
                kernels.kernelL2(nx, ny, nz, px, py, pz, 0, 1, 1, bs0_x, bs1_y, bs1_z, py, pz, ty, tz, wx, wy, wz, ggs, GG[2, 0, :, :, :, nx, ny, nz], ff[2, :, :, :, nx, ny, nz], mat_zx)
                kernels.kernelL2(nx, ny, nz, px, py, pz, 1, 0, 1, bs1_x, bs0_y, bs1_z, px, pz, tx, tz, wx, wy, wz, ggs, GG[2, 1, :, :, :, nx, ny, nz], ff[2, :, :, :, nx, ny, nz], mat_zy)
                kernels.kernelL2(nx, ny, nz, px, py, pz, 1, 1, 0, bs1_x, bs1_y, bs0_z, px, py, tx, ty, wx, wy, wz, ggs, GG[2, 2, :, :, :, nx, ny, nz], ff[2, :, :, :, nx, ny, nz], mat_zz)
                
                f_int_xx[nx:nx + px + 1, ny:ny + py, nz:nz + pz] += mat_xx[:, :, :]
                f_int_xy[nx:nx + px, ny:ny + py + 1, nz:nz + pz] += mat_xy[:, :, :]
                f_int_xz[nx:nx + px, ny:ny + py, nz:nz + pz + 1] += mat_xz[:, :, :]
                
                f_int_yx[nx:nx + px + 1, ny:ny + py, nz:nz + pz] += mat_yx[:, :, :]
                f_int_yy[nx:nx + px, ny:ny + py + 1, nz:nz + pz] += mat_yy[:, :, :]
                f_int_yz[nx:nx + px, ny:ny + py, nz:nz + pz + 1] += mat_yz[:, :, :]
                
                f_int_zx[nx:nx + px + 1, ny:ny + py, nz:nz + pz] += mat_zx[:, :, :]
                f_int_zy[nx:nx + px, ny:ny + py + 1, nz:nz + pz] += mat_zy[:, :, :]
                f_int_zz[nx:nx + px, ny:ny + py, nz:nz + pz + 1] += mat_zz[:, :, :]
    # ...
                
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        f_int_xx[:px, :, :] += f_int_xx[-px:, :, :]
        f_int_xx = f_int_xx[:f_int_xx.shape[0] - px, :, :]

        f_int_xy[:px - 1, :, :] += f_int_xy[-px + 1:, :, :]
        f_int_xy = f_int_xy[:f_int_xy.shape[0] - px + 1, :, :]

        f_int_xz[:px - 1, :, :] += f_int_xz[-px + 1:, :, :]
        f_int_xz = f_int_xz[:f_int_xz.shape[0] - px + 1, :, :]

        f_int_yx[:px, :, :] += f_int_yx[-px:, :, :]
        f_int_yx = f_int_yx[:f_int_yx.shape[0] - px, :, :]

        f_int_yy[:px - 1, :, :] += f_int_yy[-px + 1:, :, :]
        f_int_yy = f_int_yy[:f_int_yy.shape[0] - px + 1, :, :]

        f_int_yz[:px - 1, :, :] += f_int_yz[-px + 1:, :, :]
        f_int_yz = f_int_yz[:f_int_yz.shape[0] - px + 1, :, :]

        f_int_zx[:px, :, :] += f_int_zx[-px:, :, :]
        f_int_zx = f_int_zx[:f_int_zx.shape[0] - px, :, :]

        f_int_zy[:px - 1, :, :] += f_int_zy[-px + 1:, :, :]
        f_int_zy = f_int_zy[:f_int_zy.shape[0] - px + 1, :, :]

        f_int_zz[:px - 1, :, :] += f_int_zz[-px + 1:, :, :]
        f_int_zz = f_int_zz[:f_int_zz.shape[0] - px + 1, :, :]

        Nbase_x_0_xx_i = Nbase_x - px
        Nbase_x_0_xy_i = Nbase_x - px
        Nbase_x_0_xz_i = Nbase_x - px
        
        Nbase_x_0_yx_i = Nbase_x - px
        Nbase_x_0_yy_i = Nbase_x - px
        Nbase_x_0_yz_i = Nbase_x - px
        
        Nbase_x_0_zx_i = Nbase_x - px
        Nbase_x_0_zy_i = Nbase_x - px
        Nbase_x_0_zz_i = Nbase_x - px

    elif bc_x == False:
        f_int_xx = f_int_xx[1:-1, :, :]
        f_int_yx = f_int_yx[1:-1, :, :]
        f_int_zx = f_int_zx[1:-1, :, :]
        
        Nbase_x_0_xx_i = Nbase_x - 2
        Nbase_x_0_xy_i = Nbase_x - 1
        Nbase_x_0_xz_i = Nbase_x - 1

        Nbase_x_0_yx_i = Nbase_x - 2
        Nbase_x_0_yy_i = Nbase_x - 1
        Nbase_x_0_yz_i = Nbase_x - 1

        Nbase_x_0_zx_i = Nbase_x - 2
        Nbase_x_0_zy_i = Nbase_x - 1
        Nbase_x_0_zz_i = Nbase_x - 1

    else:
        Nbase_x_0_xx_i = Nbase_x
        Nbase_x_0_xy_i = Nbase_x - 1
        Nbase_x_0_xz_i = Nbase_x - 1

        Nbase_x_0_yx_i = Nbase_x
        Nbase_x_0_yy_i = Nbase_x - 1
        Nbase_x_0_yz_i = Nbase_x - 1

        Nbase_x_0_zx_i = Nbase_x
        Nbase_x_0_zy_i = Nbase_x - 1
        Nbase_x_0_zz_i = Nbase_x - 1
    # ...
    
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)
    if bc_y == True:
        f_int_xx[:, :py - 1, :] += f_int_xx[:, -py + 1:, :]
        f_int_xx = f_int_xx[:, :f_int_xx.shape[1] - py + 1, :]

        f_int_xy[:, :py, :] += f_int_xy[:, -py:, :]
        f_int_xy = f_int_xy[:, :f_int_xy.shape[1] - py, :]

        f_int_xz[:, :py - 1, :] += f_int_xz[:, -py + 1:, :]
        f_int_xz = f_int_xz[:, :f_int_xz.shape[1] - py + 1, :]

        f_int_yx[:, :py - 1, :] += f_int_yx[:, -py + 1:, :]
        f_int_yx = f_int_yx[:, :f_int_yx.shape[1] - py + 1, :]

        f_int_yy[:, :py, :] += f_int_yy[:, -py:, :]
        f_int_yy = f_int_yy[:, :f_int_yy.shape[1] - py, :]

        f_int_yz[:, :py - 1, :] += f_int_yz[:, -py + 1:, :]
        f_int_yz = f_int_yz[:, :f_int_yz.shape[1] - py + 1, :]

        f_int_zx[:, :py - 1, :] += f_int_zx[:, -py + 1:, :]
        f_int_zx = f_int_zx[:, :f_int_zx.shape[1] - py + 1, :]

        f_int_zy[:, :py, :] += f_int_zy[:, -py:, :]
        f_int_zy = f_int_zy[:, :f_int_zy.shape[1] - py, :]

        f_int_zz[:, :py - 1, :] += f_int_zz[:, -py + 1:, :]
        f_int_zz = f_int_zz[:, :f_int_zz.shape[1] - py + 1, :]

        Nbase_y_0_xx_i = Nbase_y - py
        Nbase_y_0_xy_i = Nbase_y - py
        Nbase_y_0_xz_i = Nbase_y - py
    
        Nbase_y_0_yx_i = Nbase_y - py
        Nbase_y_0_yy_i = Nbase_y - py
        Nbase_y_0_yz_i = Nbase_y - py
        
        Nbase_y_0_zx_i = Nbase_y - py
        Nbase_y_0_zy_i = Nbase_y - py
        Nbase_y_0_zz_i = Nbase_y - py

    elif bc_y == False:
        f_int_xy = f_int_xy[:, 1:-1, :]
        f_int_yy = f_int_yy[:, 1:-1, :]
        f_int_zy = f_int_zy[:, 1:-1, :]
        
        Nbase_y_0_xx_i = Nbase_y - 1
        Nbase_y_0_xy_i = Nbase_y - 2
        Nbase_y_0_xz_i = Nbase_y - 1

        Nbase_y_0_yx_i = Nbase_y - 1
        Nbase_y_0_yy_i = Nbase_y - 2
        Nbase_y_0_yz_i = Nbase_y - 1

        Nbase_y_0_zx_i = Nbase_y - 1
        Nbase_y_0_zy_i = Nbase_y - 2
        Nbase_y_0_zz_i = Nbase_y - 1

    else:
        Nbase_y_0_xx_i = Nbase_y - 1
        Nbase_y_0_xy_i = Nbase_y
        Nbase_y_0_xz_i = Nbase_y - 1
        
        Nbase_y_0_yx_i = Nbase_y - 1
        Nbase_y_0_yy_i = Nbase_y
        Nbase_y_0_yz_i = Nbase_y - 1

        Nbase_y_0_zx_i = Nbase_y - 1
        Nbase_y_0_zy_i = Nbase_y
        Nbase_y_0_zz_i = Nbase_y - 1
    # ...
    
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)
    if bc_z == True:
        f_int_xx[:, :, :pz - 1] += f_int_xx[:, :, -pz + 1:]
        f_int_xx = f_int_xx[:, :, :f_int_xx.shape[2] - pz + 1]

        f_int_xy[:, :, :pz - 1] += f_int_xy[:, :, -pz + 1:]
        f_int_xy = f_int_xy[:, :, :f_int_xy.shape[2] - pz + 1]

        f_int_xz[:, :, :pz] += f_int_xz[:, :, -pz:]
        f_int_xz = f_int_xz[:, :, :f_int_xz.shape[2] - pz]

        f_int_yx[:, :, :pz - 1] += f_int_yx[:, :, -pz + 1:]
        f_int_yx = f_int_yx[:, :, :f_int_yx.shape[2] - pz + 1]

        f_int_yy[:, :, :pz - 1] += f_int_yy[:, :, -pz + 1:]
        f_int_yy = f_int_yy[:, :, :f_int_yy.shape[2] - pz + 1]

        f_int_yz[:, :, :pz] += f_int_yz[:, :, -pz:]
        f_int_yz = f_int_yz[:, :, :f_int_yz.shape[2] - pz]

        f_int_zx[:, :, :pz - 1] += f_int_zx[:, :, -pz + 1:]
        f_int_zx = f_int_zx[:, :, :f_int_zx.shape[2] - pz + 1]

        f_int_zy[:, :, :pz - 1] += f_int_zy[:, :, -pz + 1:]
        f_int_zy = f_int_zy[:, :, :f_int_zy.shape[2] - pz + 1]

        f_int_zz[:, :, :pz] += f_int_zz[:, :, -pz:]
        f_int_zz = f_int_zz[:, :, :f_int_zz.shape[2] - pz]

        Nbase_z_0_xx_i = Nbase_z - pz
        Nbase_z_0_xy_i = Nbase_z - pz
        Nbase_z_0_xz_i = Nbase_z - pz
    
        Nbase_z_0_yx_i = Nbase_z - pz
        Nbase_z_0_yy_i = Nbase_z - pz
        Nbase_z_0_yz_i = Nbase_z - pz
        
        Nbase_z_0_zx_i = Nbase_z - pz
        Nbase_z_0_zy_i = Nbase_z - pz
        Nbase_z_0_zz_i = Nbase_z - pz

    elif bc_z == False:
        f_int_xz = f_int_xz[:, :, 1:-1]
        f_int_yz = f_int_yz[:, :, 1:-1]
        f_int_zz = f_int_zz[:, :, 1:-1]
        
        Nbase_z_0_xx_i = Nbase_z - 1
        Nbase_z_0_xy_i = Nbase_z - 1
        Nbase_z_0_xz_i = Nbase_z - 2
    
        Nbase_z_0_yx_i = Nbase_z - 1
        Nbase_z_0_yy_i = Nbase_z - 1
        Nbase_z_0_yz_i = Nbase_z - 2
        
        Nbase_z_0_zx_i = Nbase_z - 1
        Nbase_z_0_zy_i = Nbase_z - 1
        Nbase_z_0_zz_i = Nbase_z - 2

    else:
        Nbase_z_0_xx_i = Nbase_z - 1
        Nbase_z_0_xy_i = Nbase_z - 1
        Nbase_z_0_xz_i = Nbase_z
    
        Nbase_z_0_yx_i = Nbase_z - 1
        Nbase_z_0_yy_i = Nbase_z - 1
        Nbase_z_0_yz_i = Nbase_z
        
        Nbase_z_0_zx_i = Nbase_z - 1
        Nbase_z_0_zy_i = Nbase_z - 1
        Nbase_z_0_zz_i = Nbase_z
    # ...
    
    
    f_int_xx = np.reshape(f_int_xx, Nbase_x_0_xx_i*Nbase_y_0_xx_i*Nbase_z_0_xx_i)
    f_int_xy = np.reshape(f_int_xy, Nbase_x_0_xy_i*Nbase_y_0_xy_i*Nbase_z_0_xy_i)
    f_int_xz = np.reshape(f_int_xz, Nbase_x_0_xz_i*Nbase_y_0_xz_i*Nbase_z_0_xz_i)
    f_int_yx = np.reshape(f_int_yx, Nbase_x_0_yx_i*Nbase_y_0_yx_i*Nbase_z_0_yx_i)
    f_int_yy = np.reshape(f_int_yy, Nbase_x_0_yy_i*Nbase_y_0_yy_i*Nbase_z_0_yy_i)
    f_int_yz = np.reshape(f_int_yz, Nbase_x_0_yz_i*Nbase_y_0_yz_i*Nbase_z_0_yz_i)
    f_int_zx = np.reshape(f_int_zx, Nbase_x_0_zx_i*Nbase_y_0_zx_i*Nbase_z_0_zx_i)
    f_int_zy = np.reshape(f_int_zy, Nbase_x_0_zy_i*Nbase_y_0_zy_i*Nbase_z_0_zy_i)
    f_int_zz = np.reshape(f_int_zz, Nbase_x_0_zz_i*Nbase_y_0_zz_i*Nbase_z_0_zz_i)
    
    return np.concatenate((f_int_xx + f_int_yx + f_int_zx, f_int_xy + f_int_yy + f_int_zy, f_int_xz + f_int_yz + f_int_zz))



def L2_prod_V3(fun, p, Nbase, T, g, bc):
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
    
    # ... evaluation of the mapping functions at the quadrature points
    gg = np.empty((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    ff = np.empty((px + 1, py + 1, pz + 1, Nel_x, Nel_y, Nel_z), order='F')
    
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                for gx in range(px + 1):
                    for gy in range(py + 1):
                        for gz in range(pz + 1):
                            gg[gx, gy, gz, nx, ny, nz] = g(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
                            ff[gx, gy, gz, nx, ny, nz] = fun(pts_x[gx, nx], pts_y[gy, ny], pts_z[gz, nz])
    # ...
    
    basis1_x = inter.eval_on_grid_splines_ders(px - 1, Nbase_x - 1, px + 1, 0, tx, pts_x) 
    basis1_y = inter.eval_on_grid_splines_ders(py - 1, Nbase_y - 1, py + 1, 0, ty, pts_y) 
    basis1_z = inter.eval_on_grid_splines_ders(pz - 1, Nbase_z - 1, pz + 1, 0, tz, pts_z)
    
    # ... global vector and local element vector
    f_int = np.zeros((Nbase_x - 1, Nbase_y - 1, Nbase_z - 1))
    mat = np.zeros((px, py, pz), order='F')
    # ...
    
    # ... assembly
    for nx in range(Nel_x):
        for ny in range(Nel_y):
            for nz in range(Nel_z):
                
                wx = wts_x[:, nx]
                wy = wts_y[:, ny]
                wz = wts_z[:, nz]
                
                bs1_x = basis1_x[:, 0, :, nx]
                bs1_y = basis1_y[:, 0, :, ny]
                bs1_z = basis1_z[:, 0, :, nz]
                
                ggs = gg[:, :, :, nx, ny, nz]
                ffs = ff[:, :, :, nx, ny, nz]
                
                kernels.kernelL3(nx, ny, nz, px, py, pz, bs1_x, bs1_y, bs1_z, tx, ty, tz, wx, wy, wz, ggs, ffs, mat)
                
                f_int[nx:nx + px, ny:ny + py, nz:nz + pz] += mat[:, :, :]
    # ...
    
    # ... boundary conditions in x-direction (periodic, hom. Dirichlet, none)
    if bc_x == True:
        f_int[:px - 1, :, :] += f_int[-px + 1:, :, :]
        f_int = f_int[:f_int.shape[0] - px + 1, :, :]

        Nbase_x_0_i = Nbase_x - px

    else:
        Nbase_x_0_i = Nbase_x - 1
    # ...
    
    
    # ... boundary conditions in y-direction (periodic, hom. Dirichlet, none)
    if bc_y == True:
        f_int[:, :py - 1, :] += f_int[:, -py + 1:, :]
        f_int = f_int[:, :f_int.shape[1] - py + 1, :]
        
        Nbase_y_0_i = Nbase_y - py
        
    else:
        Nbase_y_0_i = Nbase_y - 1
    # ...
    
    
    # ... boundary conditions in z-direction (periodic, hom. Dirichlet, none)   
    if bc_z == True:
        f_int[:, :, :pz - 1] += f_int[:, :, -pz + 1:]
        f_int = f_int[:, :, :f_int.shape[2] - pz + 1]
        
        Nbase_z_0_i = Nbase_z - pz
        
    else:
        Nbase_z_0_i = Nbase_z - 1
    # ...
    
    return np.reshape(f_int, Nbase_x_0_i*Nbase_y_0_i*Nbase_z_0_i)