import numpy as np
from scipy import sparse


def GRAD_1d(p, Nbase, bc, full=False):
    """
    Returns the 1d discrete gradient matrix.
    
    Parameters
    ----------
    p : int
        spline degree
        
    Nbase : int
        number of spline functions
        
    bc : boolean
        boundary conditions (True = periodic, False = homogeneous Dirichlet)
        
    Returns
    -------
    G: 2d np.array
        discrete gradient matrix
    """
    
    if bc == True:
        Nbase_0 = Nbase - p
        
        G = np.zeros((Nbase_0, Nbase_0))
        
        for i in range(Nbase_0):
            G[i, i] = -1.
            if i < Nbase_0 - 1:
                G[i, i + 1] = 1.
                
        G[-1, 0] = 1.
        
        return G
    
    else:
        
        G = np.zeros((Nbase - 1, Nbase))
    
        for i in range(Nbase - 1):
            G[i, i] = -1.
            G[i, i  + 1] = 1.
            
        if full == False:
            G = G[:, 1:-1]

        return G
    

    
def GRAD_3d(p, Nbase, bc, full=[False, False, False]):
    """
    Returns the 3d discrete gradient matrix.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    full : boolean
        if 'True' return full matrix without applying boundary conditions (in case of Dirichlet)
        
    Returns
    -------
    G : sparse matrix
        3d discrete gradient matrix
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    bc_x, bc_y, bc_z = bc
    full_x, full_y, full_z = full
    
    grad_x_1d = sparse.csr_matrix(GRAD_1d(px, Nbase_x, bc_x, full_x))
    grad_y_1d = sparse.csr_matrix(GRAD_1d(py, Nbase_y, bc_y, full_y))
    grad_z_1d = sparse.csr_matrix(GRAD_1d(pz, Nbase_z, bc_z, full_z))
    
    Nbase_x_0 = Nbase_x - bc_x*px - (1 - bc_x)*2 + full_x*2
    Nbase_y_0 = Nbase_y - bc_y*py - (1 - bc_y)*2 + full_y*2
    Nbase_z_0 = Nbase_z - bc_z*pz - (1 - bc_z)*2 + full_z*2
    
    grad_x = sparse.kron(sparse.kron(grad_x_1d, sparse.identity(Nbase_y_0)), sparse.identity(Nbase_z_0))
    grad_y = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0), grad_y_1d), sparse.identity(Nbase_z_0))
    grad_z = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0), sparse.identity(Nbase_y_0)), grad_z_1d)
    
    G = sparse.bmat([[grad_x], [grad_y], [grad_z]], format='csr')
    
    return G



def CURL_3d(p, Nbase, bc, full=[False, False, False]):
    """
    Returns the 3d discrete curl matrix.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    full : boolean
        if 'True' return full matrix without applying boundary conditions (in case of Dirichlet)
        
    Returns
    -------
    C : sparse matrix
        discrete curl matrix    
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    bc_x, bc_y, bc_z = bc
    full_x, full_y, full_z = full
    
    grad_x_1d = sparse.csr_matrix(GRAD_1d(px, Nbase_x, bc_x, full_x))
    grad_y_1d = sparse.csr_matrix(GRAD_1d(py, Nbase_y, bc_y, full_y))
    grad_z_1d = sparse.csr_matrix(GRAD_1d(pz, Nbase_z, bc_z, full_z))
    
    Nbase_x_0 = Nbase_x - bc_x*px - (1 - bc_x)*2 + full_x*2
    Nbase_y_0 = Nbase_y - bc_y*py - (1 - bc_y)*2 + full_y*2
    Nbase_z_0 = Nbase_z - bc_z*pz - (1 - bc_z)*2 + full_z*2

    grad_xy = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0), sparse.identity(Nbase_y_0 + (1 - bc_y) - 2*full_y)), grad_z_1d)
    grad_xz = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0), grad_y_1d), sparse.identity(Nbase_z_0 + (1 - bc_z) - 2*full_z))

    grad_yx = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0 + (1 - bc_x) - 2*full_x), sparse.identity(Nbase_y_0)), grad_z_1d)
    grad_yz = sparse.kron(sparse.kron(grad_x_1d, sparse.identity(Nbase_y_0)), sparse.identity(Nbase_z_0 + (1 - bc_z) - 2*full_z))

    grad_zx = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0 + (1 - bc_x) - 2*full_x), grad_y_1d), sparse.identity(Nbase_z_0))
    grad_zy = sparse.kron(sparse.kron(grad_x_1d, sparse.identity(Nbase_y_0 + (1 - bc_y) - 2*full_y)), sparse.identity(Nbase_z_0))

    ZERO_xx = sparse.csr_matrix((Nbase_x_0*(Nbase_y_0 + (1 - bc_y) - 2*full_y)*(Nbase_z_0 + (1 - bc_z) - 2*full_z), (Nbase_x_0 + (1 - bc_x) - 2*full_x)*Nbase_y_0*Nbase_z_0))
    ZERO_yy = sparse.csr_matrix(((Nbase_x_0 + (1 - bc_x) - 2*full_x)*Nbase_y_0*(Nbase_z_0 + (1 - bc_z) - 2*full_z), Nbase_x_0*(Nbase_y_0 + (1 - bc_y) - 2*full_y)*Nbase_z_0))
    ZERO_zz = sparse.csr_matrix(((Nbase_x_0 + (1 - bc_x) - 2*full_x)*(Nbase_y_0 + (1 - bc_y) - 2*full_y)*Nbase_z_0, Nbase_x_0*Nbase_y_0*(Nbase_z_0 + (1 - bc_z) - 2*full_z)))
    
    C = sparse.bmat([[ZERO_xx, -grad_xy, grad_xz], [grad_yx, ZERO_yy, -grad_yz], [-grad_zx, grad_zy, ZERO_zz]], format='csr')

    return C
    
    
    
def DIV_3d(p, Nbase, bc, full=[False, False, False]):
    """
    Returns the 3d discrete divergence matrix.
    
    Parameters
    ----------
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet)
        
    full : boolean
        if 'True' return full matrix without applying boundary conditions (in case of Dirichlet)
        
    Returns
    -------
    D : sparse matrix
        discrete divergence matrix    
    """
    
    px, py, pz = p
    Nbase_x, Nbase_y, Nbase_z = Nbase
    bc_x, bc_y, bc_z = bc
    full_x, full_y, full_z = full
    
    grad_x_1d = sparse.csr_matrix(GRAD_1d(px, Nbase_x, bc_x, full_x))
    grad_y_1d = sparse.csr_matrix(GRAD_1d(py, Nbase_y, bc_y, full_y))
    grad_z_1d = sparse.csr_matrix(GRAD_1d(pz, Nbase_z, bc_z, full_z))
    
    Nbase_x_0 = Nbase_x - bc_x*px - (1 - bc_x)*2 + 2*full_x
    Nbase_y_0 = Nbase_y - bc_y*py - (1 - bc_y)*2 + 2*full_y
    Nbase_z_0 = Nbase_z - bc_z*pz - (1 - bc_z)*2 + 2*full_z

    grad_x = sparse.kron(sparse.kron(grad_x_1d, sparse.identity(Nbase_y_0 + (1 - bc_y) - 2*full_y)), sparse.identity(Nbase_z_0 + (1 - bc_z) - 2*full_z))
    grad_y = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0 + (1 - bc_x) - 2*full_x), grad_y_1d), sparse.identity(Nbase_z_0 + (1 - bc_z) - 2*full_z))
    grad_z = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0 + (1 - bc_x) - 2*full_x), sparse.identity(Nbase_y_0 + (1 - bc_y) - 2*full_y)), grad_z_1d)
    
    D = sparse.bmat([[grad_x, grad_y, grad_z]], format='csr')

    return D