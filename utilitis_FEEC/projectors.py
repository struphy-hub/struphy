import numpy as np
import psydac.core.interface as inter
import utilitis_FEEC.integrate as intgr
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
    N_x = inter.collocation_matrix(px, Nbase_x, Tx, grev_x)   
    N_y = inter.collocation_matrix(py, Nbase_y, Ty, grev_y)   
    N_z = inter.collocation_matrix(pz, Nbase_z, Tz, grev_z)   
    
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
        N_y = sparse.csr_matrix(N_y[lower_y:upper_y, :N_y.shape[1] - py])
        
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
        
        N_z = sparse.csr_matrix(N_z[lower_z:upper_z, lower_z:upper_z])
        
    
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
    N_y = inter.collocation_matrix(py, Nbase_y, Ty, grev_y)   
    N_z = inter.collocation_matrix(pz, Nbase_z, Tz, grev_z)   
                                
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
        N_y = sparse.csr_matrix(N_y[lower_y:upper_y, :N_y.shape[1] - py])
        
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
        
        N_z = sparse.csr_matrix(N_z[lower_z:upper_z, lower_z:upper_z])
        
        
    I = sparse.kron(sparse.kron(D_x, N_y), N_z, format='csr') 
    rhs = np.reshape(rhs[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z], ((Nbase_x_0 + (1 - bc_x))*Nbase_y_0*Nbase_z_0))
                     
    # solve interpolation-histopolation problem
    vec = sparse.linalg.spsolve(I, rhs)
        
    return vec



def PI_1_y(fun, p, Nbase, T, bc):
    """
    Computes the FEM coefficient of the function 'fun' projected on the space V1 (y-component).
    
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
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    pts_y, wts_y = inter.construct_quadrature_grid(Nbase_y - 1, py, pts_y_loc, wts_y_loc, grev_y)
    
    # compute element boundaries to get length of domain for periodic boundary conditions
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    
    # assemble vector of interpolation-histopolation problem at greville points
    rhs = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z))
                                
    for i in range(Nbase_x):
        for k in range(Nbase_z):
                                
            integrand = lambda y : fun(grev_x[i], y, grev_z[k])
            
            rhs[i, :, k] = intgr.integrate_1d(pts_y%el_b_y[-1], wts_y, integrand)
            
    # assemble interpolation-histopolation matrices
    N_x = inter.collocation_matrix(px, Nbase_x, Tx, grev_x) 
    D_y = sparse.csr_matrix(intgr.histopolation_matrix_1d(py, Nbase_y, Ty, grev_y, bc_y))
    N_z = inter.collocation_matrix(pz, Nbase_z, Tz, grev_z) 
                
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
        lower_y = int(np.ceil(py/2) - 1)
        upper_y = -int(np.floor(py/2))
        
    else:
        lower_y = 0
        upper_y = Nbase_y - 1
        
    if bc_z == True:
        lower_z = int(np.floor(pz/2))
        upper_z = -int(np.ceil(pz/2))
            
        N_z[:, :pz] += N_z[:, -pz:]
        N_z = sparse.csr_matrix(N_z[lower_z:upper_z, :N_z.shape[1] - pz])
        
    else: 
        lower_z = 1
        upper_z = -1
        
        N_z = sparse.csr_matrix(N_z[lower_z:upper_z, lower_z:upper_z])
    
    I = sparse.kron(sparse.kron(N_x, D_y), N_z, format='csr') 
    rhs = np.reshape(rhs[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z], (Nbase_x_0*(Nbase_y_0 + (1 - bc_y))*Nbase_z_0))
                     
    # solve interpolation-histopolation problem
    vec = sparse.linalg.spsolve(I, rhs)
        
    return vec



def PI_1_z(fun, p, Nbase, T, bc):
    """
    Computes the FEM coefficient of the function 'fun' projected on the space V1 (z-component).
    
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
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    pts_z, wts_z = inter.construct_quadrature_grid(Nbase_z - 1, pz, pts_z_loc, wts_z_loc, grev_z)
    
    # compute element boundaries to get length of domain for periodic boundary conditions
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)
    
    # assemble vector of interpolation-histopolation problem at greville points
    rhs = np.zeros((Nbase_x, Nbase_y, Nbase_z - 1))
                                
    for i in range(Nbase_x):
        for j in range(Nbase_y):
                                
            integrand = lambda z : fun(grev_x[i], grev_y[j], z)
            
            rhs[i, j, :] = intgr.integrate_1d(pts_z%el_b_z[-1], wts_z, integrand)
                
    # assemble interpolation-histopolation matrices
    N_x = inter.collocation_matrix(px, Nbase_x, Tx, grev_x)
    N_y = inter.collocation_matrix(py, Nbase_y, Ty, grev_y)
    D_z = sparse.csr_matrix(intgr.histopolation_matrix_1d(pz, Nbase_z, Tz, grev_z, bc_z))
    
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
        N_y = sparse.csr_matrix(N_y[lower_y:upper_y, :N_y.shape[1] - py])
        
    else: 
        lower_y = 1
        upper_y = -1
        
        N_y = sparse.csr_matrix(N_y[lower_y:upper_y, lower_y:upper_y])    
    
    if bc_z == True:
        lower_z = int(np.ceil(pz/2) - 1)
        upper_z = -int(np.floor(pz/2))
        
    else:
        lower_z = 0
        upper_z = Nbase_z - 1
        
    
    I = sparse.kron(sparse.kron(N_x, N_y), D_z, format='csr') 
    rhs = np.reshape(rhs[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z], (Nbase_x_0*Nbase_y_0*(Nbase_z_0 + (1 - bc_z))))
                     
    # solve interpolation-histopolation problem
    vec = sparse.linalg.spsolve(I, rhs)
        
    return vec


def PI_2_x(fun, p, Nbase, T, bc):
    """
    Computes the FEM coefficient of the function 'fun' projected on the space V2 (x-component).
    
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
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    
    pts_y, wts_y = inter.construct_quadrature_grid(Nbase_y - 1, py, pts_y_loc, wts_y_loc, grev_y)
    pts_z, wts_z = inter.construct_quadrature_grid(Nbase_z - 1, pz, pts_z_loc, wts_z_loc, grev_z)
    
    # compute element boundaries to get length of domain for periodic boundary conditions
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)

    # assemble vector of interpolation-histopolation problem at greville points
    rhs = np.zeros((Nbase_x, Nbase_y - 1, Nbase_z - 1))
                                
    for i in range(Nbase_x):
                                
        integrand = lambda y, z : fun(grev_x[i], y, z)

        rhs[i, :, :] = intgr.integrate_2d([pts_y%el_b_y[-1], pts_z%el_b_z[-1]], [wts_y, wts_z], integrand)
                
        
    # assemble interpolation-histopolation matrices
    N_x = inter.collocation_matrix(px, Nbase_x, Tx, grev_x)
    D_y = sparse.csr_matrix(intgr.histopolation_matrix_1d(py, Nbase_y, Ty, grev_y, bc_y))
    D_z = sparse.csr_matrix(intgr.histopolation_matrix_1d(pz, Nbase_z, Tz, grev_z, bc_z))
    
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
        lower_y = int(np.ceil(py/2) - 1)
        upper_y = -int(np.floor(py/2))
        
    else:
        lower_y = 0
        upper_y = Nbase_y - 1 
    
    if bc_z == True:
        lower_z = int(np.ceil(pz/2) - 1)
        upper_z = -int(np.floor(pz/2))
        
    else:
        lower_z = 0
        upper_z = Nbase_z - 1
        
    
    I = sparse.kron(sparse.kron(N_x, D_y), D_z, format='csr') 
    rhs = np.reshape(rhs[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z], (Nbase_x_0*(Nbase_y_0 + (1 - bc_y))*(Nbase_z_0 + (1 - bc_z))))
                     
    # solve interpolation-histopolation problem
    vec = sparse.linalg.spsolve(I, rhs)
        
    return vec



def PI_2_y(fun, p, Nbase, T, bc):
    """
    Computes the FEM coefficient of the function 'fun' projected on the space V2 (y-component).
    
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
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    
    pts_x, wts_x = inter.construct_quadrature_grid(Nbase_x - 1, px, pts_x_loc, wts_x_loc, grev_x)
    pts_z, wts_z = inter.construct_quadrature_grid(Nbase_z - 1, pz, pts_z_loc, wts_z_loc, grev_z)
    
    # compute element boundaries to get length of domain for periodic boundary conditions
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_z, Tz)

    # assemble vector of interpolation-histopolation problem at greville points
    rhs = np.zeros((Nbase_x - 1, Nbase_y, Nbase_z - 1))
                                
    for j in range(Nbase_y):
                                
        integrand = lambda x, z : fun(x, grev_y[j], z)

        rhs[:, j, :] = intgr.integrate_2d([pts_x%el_b_x[-1], pts_z%el_b_z[-1]], [wts_x, wts_z], integrand)
                
        
    # assemble interpolation-histopolation matrices
    D_x = sparse.csr_matrix(intgr.histopolation_matrix_1d(px, Nbase_x, Tx, grev_x, bc_x))
    N_y = inter.collocation_matrix(py, Nbase_y, Ty, grev_y)
    D_z = sparse.csr_matrix(intgr.histopolation_matrix_1d(pz, Nbase_z, Tz, grev_z, bc_z))
    
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
        N_y = sparse.csr_matrix(N_y[lower_y:upper_y, :N_y.shape[1] - py])
        
    else: 
        lower_y = 1
        upper_y = -1
        
        N_y = sparse.csr_matrix(N_y[lower_y:upper_y, lower_y:upper_y])
        
    if bc_z == True:
        lower_z = int(np.ceil(pz/2) - 1)
        upper_z = -int(np.floor(pz/2))
        
    else:
        lower_z = 0
        upper_z = Nbase_z - 1
        
    
    I = sparse.kron(sparse.kron(D_x, N_y), D_z, format='csr') 
    rhs = np.reshape(rhs[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z], ((Nbase_x_0 + (1 - bc_x))*Nbase_y_0*(Nbase_z_0 + (1 - bc_z))))
                     
    # solve interpolation-histopolation problem
    vec = sparse.linalg.spsolve(I, rhs)
        
    return vec



def PI_2_z(fun, p, Nbase, T, bc):
    """
    Computes the FEM coefficient of the function 'fun' projected on the space V2 (z-component).
    
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
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    
    pts_x, wts_x = inter.construct_quadrature_grid(Nbase_x - 1, px, pts_x_loc, wts_x_loc, grev_x)
    pts_y, wts_y = inter.construct_quadrature_grid(Nbase_y - 1, py, pts_y_loc, wts_y_loc, grev_y)
    
    # compute element boundaries to get length of domain for periodic boundary conditions
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)

    # assemble vector of interpolation-histopolation problem at greville points
    rhs = np.zeros((Nbase_x - 1, Nbase_y - 1, Nbase_z))
                                
    for k in range(Nbase_z):
                                
        integrand = lambda x, y : fun(x, y, grev_z[k])

        rhs[:, :, k] = intgr.integrate_2d([pts_x%el_b_x[-1], pts_y%el_b_y[-1]], [wts_x, wts_y], integrand)
                
        
    # assemble interpolation-histopolation matrices
    D_x = sparse.csr_matrix(intgr.histopolation_matrix_1d(px, Nbase_x, Tx, grev_x, bc_x))
    D_y = sparse.csr_matrix(intgr.histopolation_matrix_1d(py, Nbase_y, Ty, grev_y, bc_y))
    N_z = inter.collocation_matrix(pz, Nbase_z, Tz, grev_z)
    
    # apply boundary conditions
    if bc_x == True:
        lower_x = int(np.ceil(px/2) - 1)
        upper_x = -int(np.floor(px/2))
        
    else:
        lower_x = 0
        upper_x = Nbase_x - 1 
        
    if bc_y == True:
        lower_y = int(np.ceil(py/2) - 1)
        upper_y = -int(np.floor(py/2))
        
    else:
        lower_y = 0
        upper_y = Nbase_y - 1    
    
    if bc_z == True:
        lower_z = int(np.floor(pz/2))
        upper_z = -int(np.ceil(pz/2))
            
        N_z[:, :pz] += N_z[:, -pz:]
        N_z = sparse.csr_matrix(N_z[lower_z:upper_z, :N_z.shape[1] - pz])
        
    else: 
        lower_z = 1
        upper_z = -1
        
        N_z = sparse.csr_matrix(N_z[lower_z:upper_z, lower_z:upper_z])
         
    
    I = sparse.kron(sparse.kron(D_x, D_y), N_z, format='csr') 
    rhs = np.reshape(rhs[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z], ((Nbase_x_0 + (1 - bc_x))*(Nbase_y_0 + (1 - bc_y))*Nbase_z_0))
                     
    # solve interpolation-histopolation problem
    vec = sparse.linalg.spsolve(I, rhs)
        
    return vec



def PI_3(fun, p, Nbase, T, bc):
    """
    Computes the FEM coefficient of the function 'fun' projected on the space V3.
    
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
    pts_y_loc, wts_y_loc = np.polynomial.legendre.leggauss(py)
    pts_z_loc, wts_z_loc = np.polynomial.legendre.leggauss(pz)
    
    pts_x, wts_x = inter.construct_quadrature_grid(Nbase_x - 1, px, pts_x_loc, wts_x_loc, grev_x)
    pts_y, wts_y = inter.construct_quadrature_grid(Nbase_y - 1, py, pts_y_loc, wts_y_loc, grev_y)
    pts_z, wts_z = inter.construct_quadrature_grid(Nbase_z - 1, pz, pts_z_loc, wts_z_loc, grev_z)
    
    # compute element boundaries to get length of domain for periodic boundary conditions
    el_b_x = inter.construct_grid_from_knots(px, Nbase_x, Tx)
    el_b_y = inter.construct_grid_from_knots(py, Nbase_y, Ty)
    el_b_z = inter.construct_grid_from_knots(pz, Nbase_y, Tz)
    
    # assemble vector of interpolation-histopolation problem at greville points
    rhs = intgr.integrate_3d([pts_x%el_b_x[-1], pts_y%el_b_y[-1], pts_z%el_b_z[-1]], [wts_x, wts_y, wts_z], fun)
    
    # assemble interpolation-histopolation matrices
    D_x = sparse.csr_matrix(intgr.histopolation_matrix_1d(px, Nbase_x, Tx, grev_x, bc_x))
    D_y = sparse.csr_matrix(intgr.histopolation_matrix_1d(py, Nbase_y, Ty, grev_y, bc_y))
    D_z = sparse.csr_matrix(intgr.histopolation_matrix_1d(pz, Nbase_z, Tz, grev_z, bc_z))
    
    # apply boundary conditions
    if bc_x == True:
        lower_x = int(np.ceil(px/2) - 1)
        upper_x = -int(np.floor(px/2))
        
    else:
        lower_x = 0
        upper_x = Nbase_x - 1 
        
    if bc_y == True:
        lower_y = int(np.ceil(py/2) - 1)
        upper_y = -int(np.floor(py/2))
        
    else:
        lower_y = 0
        upper_y = Nbase_y - 1    
    
    if bc_z == True:
        lower_z = int(np.ceil(pz/2) - 1)
        upper_z = -int(np.floor(pz/2))
        
    else:
        lower_z = 0
        upper_z = Nbase_z - 1    
         
    
    I = sparse.kron(sparse.kron(D_x, D_y), D_z, format='csr') 
    rhs = np.reshape(rhs[lower_x:upper_x, lower_y:upper_y, lower_z:upper_z], ((Nbase_x_0 + (1 - bc_x))*(Nbase_y_0 + (1 - bc_y))*(Nbase_z_0 + (1 - bc_z))))
                     
    # solve interpolation-histopolation problem
    vec = sparse.linalg.spsolve(I, rhs)
        
    return vec





