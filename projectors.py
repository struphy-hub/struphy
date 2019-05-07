import numpy as np
import psydac.core.interface as inter
import integrate as intgr
import scipy.sparse as sparse



def PI_0_1d(fun, p, Nbase, T, bc):
    """
    Computes the FEM coefficient of the function 'fun' projected on the space V0.
    
    Parameters
    ----------
    fun : callable
        the function to be projected
        
    p : int
        spline degree
        
    Nbase: int
        number of spline functions
        
    T : np.array
        knot vector
        
    Returns
    -------
    
    vec : np.array
        the FEM coefficients
    """
    
    # compute greville points
    grev = inter.compute_greville(p, Nbase, T)
    
    # assemble vector of interpolation problem at greville points
    rhs = fun(grev)
    
    # assemble interpolation matrix
    N = inter.collocation_matrix(p, Nbase, T, grev)       
    
    # apply boundary conditions
    if bc == True:
        lower = int(np.floor(p/2))
        upper = -int(np.ceil(p/2))
            
        N[:, :p] += N[:, -p:]
        N = N[lower:upper, :N.shape[1] - p]
        
    else: 
        lower = 1
        upper = -1
        
        N = N[lower:upper, lower:upper]
        
    rhs = rhs[lower:upper]
        
        
    # solve interpolation problem
    vec = np.linalg.solve(N, rhs)
    
    return vec



def PI_1_1d(fun, p, Nbase, T, bc):
    """
    Computes the FEM coefficient of the function 'fun' projected on the space V1.
    
    Parameters
    ----------
    fun : callable
        the function to be projected
        
    p : int
        spline degree
        
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    Returns
    -------
    
    vec : np.array
        the FEM coefficients
    """
    
    # compute greville points
    grev = inter.compute_greville(p, Nbase, T)
    
    # compute quadrature grid
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p)
    pts, wts = inter.construct_quadrature_grid(Nbase - 1, p, pts_loc, wts_loc, grev)
    
    # compute element boundaries to get length of domain for periodic boundary conditions
    el_b = inter.construct_grid_from_knots(p, Nbase, T)
    
    # assemble vector of histopolation problem at greville points
    rhs = intgr.integrate_1d(pts%el_b[-1], wts, fun)
    
    # assemble histopolation matrix
    D = intgr.histopolation_matrix_1d(p, Nbase, T, grev, bc)
    
    # apply boundary conditions
    if bc == True:
        lower = int(np.ceil(p/2) - 1)
        upper = -int(np.floor(p/2))
        
    else:
        lower = 0
        upper = Nbase - 1
                          
    rhs = rhs[lower:upper]
    
    # solve histopolation problem
    vec = np.linalg.solve(D, rhs)
    
    return vec



def PI_0(fun, p, Nbase, T, bc):
    """
    Computes the FEM coefficient of the function 'fun' projected on the space V0.
    
    Parameters
    ----------
    fun : callable
        the function to be projected
        
    p : list of ints
        spline degrees in each direction
        
    Nbase: list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    Returns
    -------
    
    vec : np.array
        the FEM coefficients
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    
    Nbase_x_0 = Nbase_x - bc_x*px - (1 - bc_x)*2
    Nbase_y_0 = Nbase_y - bc_y*py - (1 - bc_y)*2
    Nbase_z_0 = Nbase_z - bc_z*pz - (1 - bc_z)*2
    
    # compute greville points
    grev_x = inter.compute_greville(px, Nbase_x, Tx)
    grev_y = inter.compute_greville(py, Nbase_y, Ty)
    grev_z = inter.compute_greville(pz, Nbase_z, Tz)
    
    # assemble vector of interpolation problem at greville points
    rhs = np.zeros((Nbase_x, Nbase_y, Nbase_z))

    for i in range(Nbase_x):
        for j in range(Nbase_y):
            for k in range(Nbase_z):
                rhs[i, j, k] = fun(grev_x[i], grev_y[j], grev_z[k])
                
    # assemble interpolation matrices
    N_x = inter.collocation_matrix(Tx, px, grev_x)
    N_y = inter.collocation_matrix(Ty, py, grev_y)
    N_z = inter.collocation_matrix(Tz, pz, grev_z)
    
    # apply boundary conditions
    if bc_x == True:
        lower_x = int(np.floor(px/2))
        upper_x = -int(np.ceil(px/2))
            
        N_x[:, :px] += N_x[:, -px:]
        N_x = sparse.csr_matrix(N_x[lower_x:upper_x, :N_x.shape[1] - px])
        
    else: 
        lower_x = 1
        upper_x = -1
        
        N_x = sparse.csr_matrix(N_x[lower_x:upper_x, lower_x:upper_x])
    
    if bc_y == True:
        lower_y = int(np.floor(py/2))
        upper_y = -int(np.ceil(py/2))
            
        N_y[:, :py] += N_y[:, -py:]
        N_y = sparse.csr_matrix(N_y[lower_x:upper_x, :N_y.shape[1] - py])
        
    else: 
        lower_y = 1
        upper_y = -1
        
        N_y = sparse.csr_matrix(N_y[lower_y:upper_y, lower_y:upper_y])
    
    if bc_z == True:
        lower_z = int(np.floor(pz/2))
        upper_z = -int(np.ceil(pz/2))
            
        N_z[:, :pz] += N_z[:, -pz:]
        N_z = sparse.csr_matrix(N_z[lower_z:upper_z, :N_z.shape[1] - pz])
        
    else: 
        lower_z = 1
        upper_z = -1
        
        N_z = sparse(N_z[lower_z:upper_z, lower_z:upper_z])
        
    
    I = sparse.kron(sparse.kron(N_x, N_y), N_z, format='csr')
    rhs = np.reshape(rhs[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z], Nbase_x_0*Nbase_y_0*Nbase_z_0)
    
    # solve interpolation problem
    vec = sparse.linalg.spsolve(I, rhs)
    
    return vec



def PI_1_x(fun, p, Nbase, T, bc):
    """
    Computes the FEM coefficient of the function 'fun' projected on the space V1 (x-component).
    
    Parameters
    ----------
    fun : callable
        the function to be projected
        
    p : list of ints
        spline degrees in each direction
        
    Nbase: list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    Returns
    -------
    
    vec : np.array
        the FEM coefficients
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    
    Nbase_x_0 = Nbase_x - bc_x*px - (1 - bc_x)*2
    Nbase_y_0 = Nbase_y - bc_y*py - (1 - bc_y)*2
    Nbase_z_0 = Nbase_z - bc_z*pz - (1 - bc_z)*2
    
    # compute greville points
    grev_x = inter.compute_greville(px, Nbase_x, Tx)
    grev_y = inter.compute_greville(py, Nbase_y, Ty)
    grev_z = inter.compute_greville(pz, Nbase_z, Tz)
    
    # compute quadrature grid
    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px)
    pts_x, wts_x = inter.construct_quadrature_grid(Nbase_x - 1, px, pts_x_loc, wts_x_loc, grev_x)
    
    # compute element boundaries to get length of domain for periodic boundary conditions
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    
    # assemble vector of interpolation-histopolation problem at greville points
    rhs = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z))
                                
    for j in range(Nbase_y):
        for k in range(Nbase_z):
                                
            integrand = lambda x : fun(x, grev_y[j], grev_z[k])
            
            rhs[:, j, k] = intgr.integrate_1d(pts_x%el_b_x[-1], wts_x, integrand)
            
    # assemble interpolation-histopolation matrices
    D_x = sparse.csr_matrix(intgr.histopolation_matrix_1d(px, Nbase_x, Tx, grev_x, bc_x))
    N_y = inter.collocation_matrix(Ty, py, grev_y)
    N_z = inter.collocation_matrix(Tz, pz, grev_z)
                                
    # apply boundary conditions
    if bc_x == True:
        lower_x = int(np.ceil(px/2) - 1)
        upper_x = -int(np.floor(px/2))
        
    else:
        lower_x = 0
        upper_x = Nbase_x - 1
        
    if bc_y == True:
        lower_y = int(np.floor(py/2))
        upper_y = -int(np.ceil(py/2))
            
        N_y[:, :py] += N_y[:, -py:]
        N_y = sparse.csr_matrix(N_y[lower_x:upper_x, :N_y.shape[1] - py])
        
    else: 
        lower_y = 1
        upper_y = -1
        
        N_y = sparse.csr_matrix(N_y[lower_y:upper_y, lower_y:upper_y])
    
    if bc_z == True:
        lower_z = int(np.floor(pz/2))
        upper_z = -int(np.ceil(pz/2))
            
        N_z[:, :pz] += N_z[:, -pz:]
        N_z = sparse.csr_matrix(N_z[lower_z:upper_z, :N_z.shape[1] - pz])
        
    else: 
        lower_z = 1
        upper_z = -1
        
        N_z = sparse(N_z[lower_z:upper_z, lower_z:upper_z])
        
    I = sparse.kron(sparse.kron(D_x, N_y), N_z, format='csr') 
    rhs = np.reshape(rhs[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z], ((Nbase_x_0 + (1 - bc_x))*Nbase_y_0*Nbase_z_0))
                     
    # solve interpolation-histopolation problem
    vec = sparse.linalg.spsolve(I, rhs)
        
    return vec



def PI_1_y(fun, Nbase_x, Nbase_y, Nbase_z, px, py, pz, el_b_x, el_b_y, el_b_z, bc):
    
    if bc == True:
        bcon = 0
        
        Nbase_x_0 = Nbase_x
        Nbase_y_0 = Nbase_y
        Nbase_z_0 = Nbase_z
    else:
        bcon = 1
        
        Nbase_x_0 = Nbase_x - 2
        Nbase_y_0 = Nbase_y - 2
        Nbase_z_0 = Nbase_z - 2
    
    
    Tx = bsp.make_knots(el_b_x, px, bc)
    Ty = bsp.make_knots(el_b_y, py, bc)
    Tz = bsp.make_knots(el_b_z, pz, bc)
    
    greville_x = bsp.greville(Tx, px, bc)
    greville_y = bsp.greville(Ty, py, bc)
    greville_z = bsp.greville(Tz, pz, bc)
    
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    
    # assemble vector of interpolation-histopolation problem at greville points
    if bc == True:
        dy = el_b_y[-1]/(len(el_b_y) - 1)
        rhs = np.zeros((Nbase_x_0, Nbase_y_0 + bcon, Nbase_z_0))
        pts_y, wts_y = bsp.quadrature_grid(np.append(greville_y, el_b_y[-1] + (1 - py%2)*dy/2), pts_y_loc, wts_y_loc)

        for i in range(Nbase_x_0):
            for k in range(Nbase_z_0):

                integrand = lambda y : fun(greville_x[i + bcon], y, greville_z[k + bcon])

                rhs[i, :, k] = intgr.integrate_1d(pts_y%el_b_y[-1], wts_y, integrand)
                    
        
    else:
        rhs = np.zeros((Nbase_x_0, Nbase_y_0 + bcon, Nbase_z_0))
        pts_y, wts_y = bsp.quadrature_grid(greville_y, pts_y_loc, wts_y_loc)
        
        for i in range(Nbase_x_0):
            for k in range(Nbase_z_0):

                integrand = lambda y : fun(greville_x[i + bcon], y, greville_z[k + bcon])

                rhs[i, :, k] = intgr.integrate_1d(pts_y, wts_y, integrand)
                
    
    rhs = np.reshape(rhs, Nbase_x_0*(Nbase_y_0 + bcon)*Nbase_z_0)
                
        
    # assemble interpolation-histopolation matrix
    N_x = sparse.csr_matrix(bsp.collocation_matrix(Tx, px, greville_x, bc)[bcon:Nbase_x_0 + bcon, bcon:Nbase_x_0 + bcon])
    D_y = sparse.csr_matrix(intgr.histopolation_matrix(Ty, py, greville_y, bc))
    N_z = sparse.csr_matrix(bsp.collocation_matrix(Tz, pz, greville_z, bc)[bcon:Nbase_z_0 + bcon, bcon:Nbase_z_0 + bcon])
    
    I = sparse.kron(sparse.kron(N_x, D_y), N_z, format='csr')
    
    # solve interpolation-histopolation problem
    return sparse.linalg.spsolve(I, rhs)



def PI_1_z(fun, Nbase_x, Nbase_y, Nbase_z, px, py, pz, el_b_x, el_b_y, el_b_z, bc):
    
    if bc == True:
        bcon = 0
        
        Nbase_x_0 = Nbase_x
        Nbase_y_0 = Nbase_y
        Nbase_z_0 = Nbase_z
    else:
        bcon = 1
        
        Nbase_x_0 = Nbase_x - 2
        Nbase_y_0 = Nbase_y - 2
        Nbase_z_0 = Nbase_z - 2
    
    
    Tx = bsp.make_knots(el_b_x, px, bc)
    Ty = bsp.make_knots(el_b_y, py, bc)
    Tz = bsp.make_knots(el_b_z, pz, bc)
    
    greville_x = bsp.greville(Tx, px, bc)
    greville_y = bsp.greville(Ty, py, bc)
    greville_z = bsp.greville(Tz, pz, bc)
    
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    
    # assemble vector of interpolation-histopolation problem at greville points
    if bc == True:
        dz = el_b_z[-1]/(len(el_b_z) - 1)
        rhs = np.zeros((Nbase_x_0, Nbase_y_0, Nbase_z_0 + bcon))
        pts_z, wts_z = bsp.quadrature_grid(np.append(greville_z, el_b_z[-1] + (1 - pz%2)*dz/2), pts_z_loc, wts_z_loc)

        for i in range(Nbase_x_0):
            for j in range(Nbase_y_0):

                integrand = lambda z : fun(greville_x[i + bcon], greville_y[j + bcon], z)

                rhs[i, j, :] = intgr.integrate_1d(pts_z, wts_z, integrand)
        
        
    else:
        rhs = np.zeros((Nbase_x_0, Nbase_y_0, Nbase_z_0 + bcon))
        pts_z, wts_z = bsp.quadrature_grid(greville_z, pts_z_loc, wts_z_loc)
        
        for i in range(Nbase_x_0):
            for j in range(Nbase_y_0):

                integrand = lambda z : fun(greville_x[i + bcon], greville_y[j + bcon], z)

                rhs[i, j, :] = intgr.integrate_1d(pts_z, wts_z, integrand)
                
    
    rhs = np.reshape(rhs, Nbase_x_0*Nbase_y_0*(Nbase_z_0 + bcon))
                
        
    # assemble interpolation-histopolation matrix
    N_x = sparse.csr_matrix(bsp.collocation_matrix(Tx, px, greville_x, bc)[bcon:Nbase_x_0 + bcon, bcon:Nbase_x_0 + bcon])
    N_y = sparse.csr_matrix(bsp.collocation_matrix(Ty, py, greville_y, bc)[bcon:Nbase_y_0 + bcon, bcon:Nbase_y_0 + bcon])
    D_z = sparse.csr_matrix(intgr.histopolation_matrix(Tz, pz, greville_z, bc))
    
    I = sparse.kron(sparse.kron(N_x, N_y), D_z, format='csr')
    
    # solve interpolation-histopolation problem
    return sparse.linalg.spsolve(I, rhs)


def PI_2_x(fun, Nbase_x, Nbase_y, Nbase_z, px, py, pz, el_b_x, el_b_y, el_b_z, bc):
    
    if bc == True:
        bcon = 0
        
        Nbase_x_0 = Nbase_x
        Nbase_y_0 = Nbase_y
        Nbase_z_0 = Nbase_z
    else:
        bcon = 1
        
        Nbase_x_0 = Nbase_x - 2
        Nbase_y_0 = Nbase_y - 2
        Nbase_z_0 = Nbase_z - 2
    
    
    Tx = bsp.make_knots(el_b_x, px, bc)
    Ty = bsp.make_knots(el_b_y, py, bc)
    Tz = bsp.make_knots(el_b_z, pz, bc)
    
    greville_x = bsp.greville(Tx, px, bc)
    greville_y = bsp.greville(Ty, py, bc)
    greville_z = bsp.greville(Tz, pz, bc)
    
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)

    
    # assemble vector of interpolation-histopolation problem at greville points
    if bc == True:
        dy = el_b_y[-1]/(len(el_b_y) - 1)
        dz = el_b_z[-1]/(len(el_b_z) - 1)
        rhs = np.zeros((Nbase_x_0, Nbase_y_0 + bcon, Nbase_z_0 + bcon))

        pts_y, wts_y = bsp.quadrature_grid(np.append(greville_y, el_b_y[-1] + (1 - py%2)*dy/2), pts_y_loc, wts_y_loc)
        pts_z, wts_z = bsp.quadrature_grid(np.append(greville_z, el_b_z[-1] + (1 - pz%2)*dz/2), pts_z_loc, wts_z_loc)

        pts_yz = [pts_y%el_b_y[-1], pts_z%el_b_z[-1]]
        wts_yz = [wts_y, wts_z]

        for i in range(Nbase_x_0):

            integrand = lambda y, z : fun(greville_x[i + bcon], y, z)

            rhs[i, :, :] = intgr.integrate_2d(pts_yz, wts_yz, integrand)
                
        
                
    else:
        rhs = np.zeros((Nbase_x_0, Nbase_y_0 + bcon, Nbase_z_0 + bcon))
        
        pts_y, wts_y = bsp.quadrature_grid(greville_y, pts_y_loc, wts_y_loc)
        pts_z, wts_z = bsp.quadrature_grid(greville_z, pts_z_loc, wts_z_loc)

        pts_yz = [pts_y, pts_z]
        wts_yz = [wts_y, wts_z]
            
        for i in range(Nbase_x_0):

            integrand = lambda y, z : fun(greville_x[i + bcon], y, z)

            rhs[i, :, :] = intgr.integrate_2d(pts_yz, wts_yz, integrand)
     
    
    rhs = np.reshape(rhs, Nbase_x_0*(Nbase_y_0 + bcon)*(Nbase_z_0 + bcon))
                
        
    # assemble interpolation-histopolation matrix
    N_x = sparse.csr_matrix(bsp.collocation_matrix(Tx, px, greville_x, bc)[bcon:Nbase_x_0 + bcon, bcon:Nbase_x_0 + bcon])
    D_y = sparse.csr_matrix(intgr.histopolation_matrix(Ty, py, greville_y, bc))
    D_z = sparse.csr_matrix(intgr.histopolation_matrix(Tz, pz, greville_z, bc))
    
    I = sparse.kron(sparse.kron(N_x, D_y), D_z, format='csr')
    
    # solve interpolation-histopolation problem
    return sparse.linalg.spsolve(I, rhs)



def PI_2_y(fun, Nbase_x, Nbase_y, Nbase_z, px, py, pz, el_b_x, el_b_y, el_b_z, bc):
    
    if bc == True:
        bcon = 0
        
        Nbase_x_0 = Nbase_x
        Nbase_y_0 = Nbase_y
        Nbase_z_0 = Nbase_z
    else:
        bcon = 1
        
        Nbase_x_0 = Nbase_x - 2
        Nbase_y_0 = Nbase_y - 2
        Nbase_z_0 = Nbase_z - 2
    
    
    Tx = bsp.make_knots(el_b_x, px, bc)
    Ty = bsp.make_knots(el_b_y, py, bc)
    Tz = bsp.make_knots(el_b_z, pz, bc)
    
    greville_x = bsp.greville(Tx, px, bc)
    greville_y = bsp.greville(Ty, py, bc)
    greville_z = bsp.greville(Tz, pz, bc)
    
    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    
    
    # assemble vector of interpolation-histopolation problem at greville points
    if bc == True:
        dx = el_b_x[-1]/(len(el_b_x) - 1)
        dz = el_b_z[-1]/(len(el_b_z) - 1)
        rhs = np.zeros((Nbase_x_0 + bcon, Nbase_y_0, Nbase_z_0 + bcon))

        pts_x, wts_x = bsp.quadrature_grid(np.append(greville_x, el_b_x[-1] + (1 - px%2)*dx/2), pts_x_loc, wts_x_loc)
        pts_z, wts_z = bsp.quadrature_grid(np.append(greville_z, el_b_z[-1] + (1 - pz%2)*dz/2), pts_z_loc, wts_z_loc)

        pts_xz = [pts_x%el_b_x[-1], pts_z%el_b_z[-1]]
        wts_xz = [wts_x, wts_z]

        for j in range(Nbase_y_0):

            integrand = lambda x, z : fun(x, greville_y[j + bcon], z)

            rhs[:, j, :] = intgr.integrate_2d(pts_xz, wts_xz, integrand)   
                
    else:
        rhs = np.zeros((Nbase_x_0 + bcon, Nbase_y_0, Nbase_z_0 + bcon))
        
        pts_x, wts_x = bsp.quadrature_grid(greville_x, pts_x_loc, wts_x_loc)
        pts_z, wts_z = bsp.quadrature_grid(greville_z, pts_z_loc, wts_z_loc)

        pts_xz = [pts_x, pts_z]
        wts_xz = [wts_x, wts_z]
            
        for j in range(Nbase_y_0):

            integrand = lambda x, z : fun(x, greville_y[j + bcon], z)

            rhs[:, j, :] = intgr.integrate_2d(pts_xz, wts_xz, integrand)
     
    
    rhs = np.reshape(rhs, (Nbase_x_0 + bcon)*Nbase_y_0*(Nbase_z_0 + bcon))
                
        
    # assemble interpolation-histopolation matrix
    D_x = sparse.csr_matrix(intgr.histopolation_matrix(Tx, px, greville_x, bc))
    N_y = sparse.csr_matrix(bsp.collocation_matrix(Ty, py, greville_y, bc)[bcon:Nbase_y_0 + bcon, bcon:Nbase_y_0 + bcon])
    D_z = sparse.csr_matrix(intgr.histopolation_matrix(Tz, pz, greville_z, bc))
    
    I = sparse.kron(sparse.kron(D_x, N_y), D_z, format='csr')
    
    # solve interpolation-histopolation problem
    return sparse.linalg.spsolve(I, rhs)


def PI_2_z(fun, Nbase_x, Nbase_y, Nbase_z, px, py, pz, el_b_x, el_b_y, el_b_z, bc):
    
    if bc == True:
        bcon = 0
        
        Nbase_x_0 = Nbase_x
        Nbase_y_0 = Nbase_y
        Nbase_z_0 = Nbase_z
    else:
        bcon = 1
        
        Nbase_x_0 = Nbase_x - 2
        Nbase_y_0 = Nbase_y - 2
        Nbase_z_0 = Nbase_z - 2
    
    
    Tx = bsp.make_knots(el_b_x, px, bc)
    Ty = bsp.make_knots(el_b_y, py, bc)
    Tz = bsp.make_knots(el_b_z, pz, bc)
    
    greville_x = bsp.greville(Tx, px, bc)
    greville_y = bsp.greville(Ty, py, bc)
    greville_z = bsp.greville(Tz, pz, bc)
    
    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    
    # assemble vector of interpolation-histopolation problem at greville points
    if bc == True:
        dx = el_b_x[-1]/(len(el_b_x) - 1)
        dy = el_b_y[-1]/(len(el_b_y) - 1)
        rhs = np.zeros((Nbase_x_0 + bcon, Nbase_y_0 + bcon, Nbase_z_0))

        pts_x, wts_x = bsp.quadrature_grid(np.append(greville_x, el_b_x[-1] + (1 - px%2)*dx/2), pts_x_loc, wts_x_loc)
        pts_y, wts_y = bsp.quadrature_grid(np.append(greville_y, el_b_y[-1] + (1 - py%2)*dy/2), pts_y_loc, wts_y_loc)

        pts_xy = [pts_x%el_b_x[-1], pts_y%el_b_y[-1]]
        wts_xy = [wts_x, wts_y]

        for k in range(Nbase_z_0):

            integrand = lambda x, y : fun(x, y, greville_z[k + bcon])

            rhs[:, :, k] = intgr.integrate_2d(pts_xy, wts_xy, integrand)
                
    else:
        rhs = np.zeros((Nbase_x_0 + bcon, Nbase_y_0 + bcon, Nbase_z_0))
        
        pts_x, wts_x = bsp.quadrature_grid(greville_x, pts_x_loc, wts_x_loc)
        pts_y, wts_y = bsp.quadrature_grid(greville_y, pts_y_loc, wts_y_loc)

        pts_xy = [pts_x, pts_y]
        wts_xy = [wts_x, wts_y]
            
        for k in range(Nbase_z_0):

            integrand = lambda x, y : fun(x, y, greville_z[k + bcon])

            rhs[:, :, k] = intgr.integrate_2d(pts_xy, wts_xy, integrand)
     
    
    rhs = np.reshape(rhs, (Nbase_x_0 + bcon)*(Nbase_y_0 + bcon)*Nbase_z_0)
                
        
    # assemble interpolation-histopolation matrix
    D_x = sparse.csr_matrix(intgr.histopolation_matrix(Tx, px, greville_x, bc))
    D_y = sparse.csr_matrix(intgr.histopolation_matrix(Ty, py, greville_y, bc))
    N_z = sparse.csr_matrix(bsp.collocation_matrix(Tz, pz, greville_z, bc)[bcon:Nbase_z_0 + bcon, bcon:Nbase_z_0 + bcon])
    
    I = sparse.kron(sparse.kron(D_x, D_y), N_z, format='csr')
    
    # solve interpolation-histopolation problem
    return sparse.linalg.spsolve(I, rhs)



def PI_3(fun, Nbase_x, Nbase_y, Nbase_z, px, py, pz, el_b_x, el_b_y, el_b_z, bc):
    
    if bc == True:
        bcon = 0
        
        Nbase_x_0 = Nbase_x
        Nbase_y_0 = Nbase_y
        Nbase_z_0 = Nbase_z
    else:
        bcon = 1
        
        Nbase_x_0 = Nbase_x - 2
        Nbase_y_0 = Nbase_y - 2
        Nbase_z_0 = Nbase_z - 2
    
    
    Tx = bsp.make_knots(el_b_x, px, bc)
    Ty = bsp.make_knots(el_b_y, py, bc)
    Tz = bsp.make_knots(el_b_z, pz, bc)
    
    greville_x = bsp.greville(Tx, px, bc)
    greville_y = bsp.greville(Ty, py, bc)
    greville_z = bsp.greville(Tz, pz, bc)
    
    pts_x_loc, wts_x_loc = np.polynomial.legendre.leggauss(px)
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    
    
    # assemble vector of histopolation problem at greville points
    if bc == True:
        dx = el_b_x[-1]/(len(el_b_x) - 1)
        dy = el_b_y[-1]/(len(el_b_y) - 1)
        dz = el_b_z[-1]/(len(el_b_z) - 1)
        rhs = np.zeros((Nbase_x_0 + bcon, Nbase_y_0 + bcon, Nbase_z_0 + bcon))

        pts_x, wts_x = bsp.quadrature_grid(np.append(greville_x, el_b_x[-1] + (1 - px%2)*dx/2), pts_x_loc, wts_x_loc)
        pts_y, wts_y = bsp.quadrature_grid(np.append(greville_y, el_b_y[-1] + (1 - py%2)*dy/2), pts_y_loc, wts_y_loc)
        pts_z, wts_z = bsp.quadrature_grid(np.append(greville_z, el_b_z[-1] + (1 - pz%2)*dz/2), pts_z_loc, wts_z_loc)

        pts_xyz = [pts_x%el_b_x[-1], pts_y%el_b_y[-1], pts_z%el_b_z[-1]]
        wts_xyz = [wts_x, wts_y, wts_z]

        rhs[:, :, :] = intgr.integrate_3d(pts_xyz, wts_xyz, fun)
        
            
    else:
        rhs = np.zeros((Nbase_x_0 + bcon, Nbase_y_0 + bcon, Nbase_z_0 + bcon))
        
        pts_x, wts_x = bsp.quadrature_grid(greville_x, pts_x_loc, wts_x_loc)
        pts_y, wts_y = bsp.quadrature_grid(greville_y, pts_y_loc, wts_y_loc)
        pts_z, wts_z = bsp.quadrature_grid(greville_z, pts_z_loc, wts_z_loc)

        pts_xyz = [pts_x, pts_y, pts_z]
        wts_xyz = [wts_x, wts_y, wts_z]
        
        rhs[:, :, :] = intgr.integrate_3d(pts_xyz, wts_xyz, fun)
        
    
    rhs = np.reshape(rhs, (Nbase_x_0 + bcon)*(Nbase_y_0 + bcon)*(Nbase_z_0 + bcon))
                
        
    # assemble interpolation-histopolation matrix
    D_x = sparse.csr_matrix(intgr.histopolation_matrix(Tx, px, greville_x, bc))
    D_y = sparse.csr_matrix(intgr.histopolation_matrix(Ty, py, greville_y, bc))
    D_z = sparse.csr_matrix(intgr.histopolation_matrix(Tz, pz, greville_z, bc))
    
    I = sparse.kron(sparse.kron(D_x, D_y), D_z, format='csr')
    
    # solve interpolation-histopolation problem
    return sparse.linalg.spsolve(I, rhs)





