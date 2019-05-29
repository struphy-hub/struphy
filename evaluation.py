import numpy as np
import psydac.core.interface as inter
import scipy.sparse as sparse


def evaluate_field_V0_1d(vec, x, p, Nbase, T, bc, full=False):
    """
    Evaluates the 1d FEM field of the space V0 at the points x.
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    x : np.array
        evaluation points
        
    p : int
        spline degree
        
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    bc : boolean
        boundary conditions (True = periodic, False = homogeneous Dirichlet)
        
    Returns
    -------
    eva: np.arrray
        the function values at the points x
    """
    
    if bc == True:
        N = inter.collocation_matrix(p, Nbase, T, x)
        N[:, :p] += N[:, -p:]
        N = sparse.csr_matrix(N[:, :N.shape[1] - p])
        
    else:
        
        if full == False:
            N = sparse.csr_matrix(inter.collocation_matrix(p, Nbase, T, x)[:, 1:-1])
            
        else:
            N = sparse.csr_matrix(inter.collocation_matrix(p, Nbase, T, x))
        
    eva = N.dot(vec) 
        
    return eva



def evaluate_field_V1_1d(vec, x, p, Nbase, T, bc):
    """
    Evaluates the 1d FEM field of the space V1 at the points x.
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    x : np.array
        evaluation points
        
    p : int
        spline degree
        
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    bc : boolean
        boundary conditions (True = periodic, False = homogeneous Dirichlet)
        
    Returns
    -------
    eval: np.arrray
        the function values at the points x
    """
    
    t = T[1:-1]
    
    if bc == True:
        D = inter.collocation_matrix(p - 1, Nbase - 1, t, x)
        
        D[:, :(p - 1)] += D[:, -(p - 1):]
        D = D[:, :D.shape[1] - (p - 1)]
        
        for j in range(Nbase - p):
            D[:, j] = p*D[:, j]/(t[j + p] - t[j])
            
        D = sparse.csr_matrix(D)
        
    else:
        D = inter.collocation_matrix(p - 1, Nbase - 1, t, x)
        
        for j in range(Nbase - 1):
            D[:, j] = p*D[:, j]/(t[j + p] - t[j])
            
        D = sparse.csr_matrix(D)
    
    eva = D.dot(vec)
    
    return eva



def evaluate_field_V0(vec, x, p, Nbase, T, bc):
    """
    Evaluates the 3d FEM field of the space V0 at the tensor grid given by x = (x, y, z).
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    x : list of np.arrays
        evaluation points in each direction
        
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    Returns
    -------
    EVAL : np.array
        the function values at the points x.
    """
    
    x, y, z = x
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    
    if bc_x == True:
        Nx = inter.collocation_matrix(px, Nbase_x, Tx, x)
        Nx[:, :px] += Nx[:, -px:]
        Nx = sparse.csr_matrix(Nx[:, :Nx.shape[1] - px])
        
    else:
        Nx = sparse.csr_matrix(inter.collocation_matrix(px, Nbase_x, Tx, x)[:, 1:-1])
        
    if bc_y == True:
        Ny = inter.collocation_matrix(py, Nbase_y, Ty, y)
        Ny[:, :py] += Ny[:, -py:]
        Ny = sparse.csr_matrix(Ny[:, :Ny.shape[1] - py])
        
    else:
        Ny = sparse.csr_matrix(inter.collocation_matrix(py, Nbase_y, Ty, y)[:, 1:-1])
        
    if bc_z == True:
        Nz = inter.collocation_matrix(pz, Nbase_z, Tz, z)
        Nz[:, :pz] += Nz[:, -pz:]
        Nz = sparse.csr_matrix(Nz[:, :Nz.shape[1] - pz])
        
    else:
        Nz = sparse.csr_matrix(inter.collocation_matrix(pz, Nbase_z, Tz, z)[:, 1:-1])
    
    EVAL = (sparse.kron(sparse.kron(Nx, Ny), Nz)).dot(vec)
    
    return EVAL



def evaluate_field_V1_x(vec, x, p, Nbase, T, bc):
    """
    Evaluates the x-component of the 3d FEM field of the space V1 at the tensor grid given by x = (x, y, z).
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    x : list of np.arrays
        evaluation points in each direction
        
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    Returns
    -------
    EVAL : np.array
        the function values at the points x.
    """
    
    x, y, z = x
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc

    tx = Tx[1:-1]
    
    if bc_x == True:
        Dx = inter.collocation_matrix(px - 1, Nbase_x - 1, tx, x)
        
        Dx[:, :(px - 1)] += Dx[:, -(px - 1):]
        Dx = Dx[:, :Dx.shape[1] - (px - 1)]
        
        for j in range(Nbase_x - px):
            Dx[:, j] = px*Dx[:, j]/(tx[j + px] - tx[j])
            
        Dx = sparse.csr_matrix(Dx)
        
    else:
        Dx = inter.collocation_matrix(px - 1, Nbase_x - 1, tx, x)
        
        for j in range(Nbase_x - 1):
            Dx[:, j] = px*Dx[:, j]/(tx[j + px] - tx[j])
            
        Dx = sparse.csr_matrix(Dx)
    
    if bc_y == True:
        Ny = inter.collocation_matrix(py, Nbase_y, Ty, y)
        Ny[:, :py] += Ny[:, -py:]
        Ny = sparse.csr_matrix(Ny[:, :Ny.shape[1] - py])
        
    else:
        Ny = sparse.csr_matrix(inter.collocation_matrix(py, Nbase_y, Ty, y)[:, 1:-1])
        
    if bc_z == True:
        Nz = inter.collocation_matrix(pz, Nbase_z, Tz, z)
        Nz[:, :pz] += Nz[:, -pz:]
        Nz = sparse.csr_matrix(Nz[:, :Nz.shape[1] - pz])
        
    else:
        Nz = sparse.csr_matrix(inter.collocation_matrix(pz, Nbase_z, Tz, z)[:, 1:-1])
    
    EVAL = (sparse.kron(sparse.kron(Dx, Ny), Nz)).dot(vec)
    
    return EVAL



def evaluate_field_V1_y(vec, x, p, Nbase, T, bc):
    """
    Evaluates the y-component of the 3d FEM field of the space V1 at the tensor grid given by x = (x, y, z).
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    x : list of np.arrays
        evaluation points in each direction
        
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    Returns
    -------
    EVAL : np.array
        the function values at the points x.
    """
    
    x, y, z = x
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    
    ty = Ty[1:-1]
    
    if bc_x == True:
        Nx = inter.collocation_matrix(px, Nbase_x, Tx, x)
        Nx[:, :px] += Nx[:, -px:]
        Nx = sparse.csr_matrix(Nx[:, :Nx.shape[1] - px])
        
    else:
        Nx = sparse.csr_matrix(inter.collocation_matrix(px, Nbase_x, Tx, x)[:, 1:-1])
        
    if bc_y == True:
        Dy = inter.collocation_matrix(py - 1, Nbase_y - 1, ty, y)
        
        Dy[:, :(py - 1)] += Dy[:, -(py - 1):]
        Dy = Dy[:, :Dy.shape[1] - (py - 1)]
        
        for j in range(Nbase_y - py):
            Dy[:, j] = py*Dy[:, j]/(ty[j + py] - ty[j])
            
        Dy = sparse.csr_matrix(Dy)
        
    else:
        Dy = inter.collocation_matrix(py - 1, Nbase_y - 1, ty, y)
        
        for j in range(Nbase_y - 1):
            Dy[:, j] = py*Dy[:, j]/(ty[j + py] - ty[j])
            
        Dy = sparse.csr_matrix(Dy)
        
    if bc_z == True:
        Nz = inter.collocation_matrix(pz, Nbase_z, Tz, z)
        Nz[:, :pz] += Nz[:, -pz:]
        Nz = sparse.csr_matrix(Nz[:, :Nz.shape[1] - pz])
        
    else:
        Nz = sparse.csr_matrix(inter.collocation_matrix(pz, Nbase_z, Tz, z)[:, 1:-1])
        
    EVAL = (sparse.kron(sparse.kron(Nx, Dy), Nz)).dot(vec)
    
    return EVAL



def evaluate_field_V1_z(vec, x, p, Nbase, T, bc):
    """
    Evaluates the z-component of the 3d FEM field of the space V1 at the tensor grid given by x = (x, y, z).
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    x : list of np.arrays
        evaluation points in each direction
        
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    Returns
    -------
    EVAL : np.array
        the function values at the points x.
    """
    
    x, y, z = x
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    
    tz = Tz[1:-1]
    
    if bc_x == True:
        Nx = inter.collocation_matrix(px, Nbase_x, Tx, x)
        Nx[:, :px] += Nx[:, -px:]
        Nx = sparse.csr_matrix(Nx[:, :Nx.shape[1] - px])
        
    else:
        Nx = sparse.csr_matrix(inter.collocation_matrix(px, Nbase_x, Tx, x)[:, 1:-1])
        
    if bc_y == True:
        Ny = inter.collocation_matrix(py, Nbase_y, Ty, y)
        Ny[:, :py] += Ny[:, -py:]
        Ny = sparse.csr_matrix(Ny[:, :Ny.shape[1] - py])
        
    else:
        Ny = sparse.csr_matrix(inter.collocation_matrix(py, Nbase_y, Ty, y)[:, 1:-1])
        
    if bc_z == True:
        Dz = inter.collocation_matrix(pz - 1, Nbase_z - 1, tz, z)
        
        Dz[:, :(pz - 1)] += Dz[:, -(pz - 1):]
        Dz = Dz[:, :Dz.shape[1] - (pz - 1)]
        
        for j in range(Nbase_z - pz):
            Dz[:, j] = pz*Dz[:, j]/(tz[j + pz] - tz[j])
            
        Dz = sparse.csr_matrix(Dz)
        
    else:
        Dz = inter.collocation_matrix(pz - 1, Nbase_z - 1, tz, z)
        
        for j in range(Nbase_z - 1):
            Dz[:, j] = pz*Dz[:, j]/(tz[j + pz] - tz[j])
            
        Dz = sparse.csr_matrix(Dz)
    
    EVAL = (sparse.kron(sparse.kron(Nx, Ny), Dz)).dot(vec)
    
    return EVAL



def evaluate_field_V2_x(vec, x, p, Nbase, T, bc):
    """
    Evaluates the x-component of the 3d FEM field of the space V2 at the tensor grid given by x = (x, y, z).
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    x : list of np.arrays
        evaluation points in each direction
        
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    Returns
    -------
    EVAL : np.array
        the function values at the points x.
    """
    
    x, y, z = x
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    
    ty = Ty[1:-1]
    tz = Tz[1:-1]
    
    if bc_x == True:
        Nx = inter.collocation_matrix(px, Nbase_x, Tx, x)
        Nx[:, :px] += Nx[:, -px:]
        Nx = sparse.csr_matrix(Nx[:, :Nx.shape[1] - px])
        
    else:
        Nx = sparse.csr_matrix(inter.collocation_matrix(px, Nbase_x, Tx, x)[:, 1:-1])
        
    if bc_y == True:
        Dy = inter.collocation_matrix(py - 1, Nbase_y - 1, ty, y)
        
        Dy[:, :(py - 1)] += Dy[:, -(py - 1):]
        Dy = Dy[:, :Dy.shape[1] - (py - 1)]
        
        for j in range(Nbase_y - py):
            Dy[:, j] = py*Dy[:, j]/(ty[j + py] - ty[j])
            
        Dy = sparse.csr_matrix(Dy)
        
    else:
        Dy = inter.collocation_matrix(py - 1, Nbase_y - 1, ty, y)
        
        for j in range(Nbase_y - 1):
            Dy[:, j] = py*Dy[:, j]/(ty[j + py] - ty[j])
            
        Dy = sparse.csr_matrix(Dy)
        
    if bc_z == True:
        Dz = inter.collocation_matrix(pz - 1, Nbase_z - 1, tz, z)
        
        Dz[:, :(pz - 1)] += Dz[:, -(pz - 1):]
        Dz = Dz[:, :Dz.shape[1] - (pz - 1)]
        
        for j in range(Nbase_z - pz):
            Dz[:, j] = pz*Dz[:, j]/(tz[j + pz] - tz[j])
            
        Dz = sparse.csr_matrix(Dz)
        
    else:
        Dz = inter.collocation_matrix(pz - 1, Nbase_z - 1, tz, z)
        
        for j in range(Nbase_z - 1):
            Dz[:, j] = pz*Dz[:, j]/(tz[j + pz] - tz[j])
            
        Dz = sparse.csr_matrix(Dz)
        
    EVAL = (sparse.kron(sparse.kron(Nx, Dy), Dz)).dot(vec)
    
    return EVAL



def evaluate_field_V2_y(vec, x, p, Nbase, T, bc):
    """
    Evaluates the y-component of the 3d FEM field of the space V2 at the tensor grid given by x = (x, y, z).
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    x : list of np.arrays
        evaluation points in each direction
        
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    Returns
    -------
    EVAL : np.array
        the function values at the points x.
    """
    
    x, y, z = x
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    
    tx = Tx[1:-1]
    tz = Tz[1:-1]
    
    if bc_x == True:
        Dx = inter.collocation_matrix(px - 1, Nbase_x - 1, tx, x)
        
        Dx[:, :(px - 1)] += Dx[:, -(px - 1):]
        Dx = Dx[:, :Dx.shape[1] - (px - 1)]
        
        for j in range(Nbase_x - px):
            Dx[:, j] = px*Dx[:, j]/(tx[j + px] - tx[j])
            
        Dx = sparse.csr_matrix(Dx)
        
    else:
        Dx = inter.collocation_matrix(px - 1, Nbase_x - 1, tx, x)
        
        for j in range(Nbase_x - 1):
            Dx[:, j] = px*Dx[:, j]/(tx[j + px] - tx[j])
            
        Dx = sparse.csr_matrix(Dx)
        
    if bc_y == True:
        Ny = inter.collocation_matrix(py, Nbase_y, Ty, y)
        Ny[:, :py] += Ny[:, -py:]
        Ny = sparse.csr_matrix(Ny[:, :Ny.shape[1] - py])
        
    else:
        Ny = sparse.csr_matrix(inter.collocation_matrix(py, Nbase_y, Ty, y)[:, 1:-1])
        
    if bc_z == True:
        Dz = inter.collocation_matrix(pz - 1, Nbase_z - 1, tz, z)
        
        Dz[:, :(pz - 1)] += Dz[:, -(pz - 1):]
        Dz = Dz[:, :Dz.shape[1] - (pz - 1)]
        
        for j in range(Nbase_z - pz):
            Dz[:, j] = pz*Dz[:, j]/(tz[j + pz] - tz[j])
            
        Dz = sparse.csr_matrix(Dz)
        
    else:
        Dz = inter.collocation_matrix(pz - 1, Nbase_z - 1, tz, z)
        
        for j in range(Nbase_z - 1):
            Dz[:, j] = pz*Dz[:, j]/(tz[j + pz] - tz[j])
            
        Dz = sparse.csr_matrix(Dz)
    
    EVAL = (sparse.kron(sparse.kron(Dx, Ny), Dz)).dot(vec)
    
    return EVAL



def evaluate_field_V2_z(vec, x, p, Nbase, T, bc):
    """
    Evaluates the z-component of the 3d FEM field of the space V2 at the tensor grid given by x = (x, y, z).
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    x : list of np.arrays
        evaluation points in each direction
        
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    Returns
    -------
    EVAL : np.array
        the function values at the points x.
    """
    
    x, y, z = x
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    
    if bc_x == True:
        Dx = inter.collocation_matrix(px - 1, Nbase_x - 1, tx, x)
        
        Dx[:, :(px - 1)] += Dx[:, -(px - 1):]
        Dx = Dx[:, :Dx.shape[1] - (px - 1)]
        
        for j in range(Nbase_x - px):
            Dx[:, j] = px*Dx[:, j]/(tx[j + px] - tx[j])
            
        Dx = sparse.csr_matrix(Dx)
        
    else:
        Dx = inter.collocation_matrix(px - 1, Nbase_x - 1, tx, x)
        
        for j in range(Nbase_x - 1):
            Dx[:, j] = px*Dx[:, j]/(tx[j + px] - tx[j])
            
        Dx = sparse.csr_matrix(Dx)
        
    if bc_y == True:
        Dy = inter.collocation_matrix(py - 1, Nbase_y - 1, ty, y)
        
        Dy[:, :(py - 1)] += Dy[:, -(py - 1):]
        Dy = Dy[:, :Dy.shape[1] - (py - 1)]
        
        for j in range(Nbase_y - py):
            Dy[:, j] = py*Dy[:, j]/(ty[j + py] - ty[j])
            
        Dy = sparse.csr_matrix(Dy)
        
    else:
        Dy = inter.collocation_matrix(py - 1, Nbase_y - 1, ty, y)
        
        for j in range(Nbase_y - 1):
            Dy[:, j] = py*Dy[:, j]/(ty[j + py] - ty[j])
            
        Dy = sparse.csr_matrix(Dy)
        
    if bc_z == True:
        Nz = inter.collocation_matrix(pz, Nbase_z, Tz, z)
        Nz[:, :pz] += Nz[:, -pz:]
        Nz = sparse.csr_matrix(Nz[:, :Nz.shape[1] - pz])
        
    else:
        Nz = sparse.csr_matrix(inter.collocation_matrix(pz, Nbase_z, Tz, z)[:, 1:-1])
    
    EVAL = (sparse.kron(sparse.kron(Dx, Dy), Nz)).dot(vec)
    
    return EVAL



def evaluate_field_V3(vec, x, p, Nbase, T, bc):
    """
    Evaluates the 3d FEM field of the space V3 at the tensor grid given by x = (x, y, z).
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    x : list of np.arrays
        evaluation points in each direction
        
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    Returns
    -------
    EVAL : np.array
        the function values at the points x.
    """
    
    x, y, z = x
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    Tx, Ty, Tz = T
    bc_x, bc_y, bc_z = bc
    
    tx = Tx[1:-1]
    ty = Ty[1:-1]
    tz = Tz[1:-1]
    
    if bc_x == True:
        Dx = inter.collocation_matrix(px - 1, Nbase_x - 1, tx, x)
        
        Dx[:, :(px - 1)] += Dx[:, -(px - 1):]
        Dx = Dx[:, :Dx.shape[1] - (px - 1)]
        
        for j in range(Nbase_x - px):
            Dx[:, j] = px*Dx[:, j]/(tx[j + px] - tx[j])
            
        Dx = sparse.csr_matrix(Dx)
        
    else:
        Dx = inter.collocation_matrix(px - 1, Nbase_x - 1, tx, x)
        
        for j in range(Nbase_x - 1):
            Dx[:, j] = px*Dx[:, j]/(tx[j + px] - tx[j])
            
        Dx = sparse.csr_matrix(Dx)
        
    if bc_y == True:
        Dy = inter.collocation_matrix(py - 1, Nbase_y - 1, ty, y)
        
        Dy[:, :(py - 1)] += Dy[:, -(py - 1):]
        Dy = Dy[:, :Dy.shape[1] - (py - 1)]
        
        for j in range(Nbase_y - py):
            Dy[:, j] = py*Dy[:, j]/(ty[j + py] - ty[j])
            
        Dy = sparse.csr_matrix(Dy)
        
    else:
        Dy = inter.collocation_matrix(py - 1, Nbase_y - 1, ty, y)
        
        for j in range(Nbase_y - 1):
            Dy[:, j] = py*Dy[:, j]/(ty[j + py] - ty[j])
            
        Dy = sparse.csr_matrix(Dy)
        
    if bc_z == True:
        Dz = inter.collocation_matrix(pz - 1, Nbase_z - 1, tz, z)
        
        Dz[:, :(pz - 1)] += Dz[:, -(pz - 1):]
        Dz = Dz[:, :Dz.shape[1] - (pz - 1)]
        
        for j in range(Nbase_z - pz):
            Dz[:, j] = pz*Dz[:, j]/(tz[j + pz] - tz[j])
            
        Dz = sparse.csr_matrix(Dz)
        
    else:
        Dz = inter.collocation_matrix(pz - 1, Nbase_z - 1, tz, z)
        
        for j in range(Nbase_z - 1):
            Dz[:, j] = pz*Dz[:, j]/(tz[j + pz] - tz[j])
            
        Dz = sparse.csr_matrix(Dz)
    
    EVAL = (sparse.kron(sparse.kron(Dx, Dy), Dz)).dot(vec)
    
    return EVAL