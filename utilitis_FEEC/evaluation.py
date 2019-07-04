import numpy as np
import scipy.sparse as sparse
import utilitis_FEEC.bsplines as bsp


#==============================================================================================================================
def evaluate_field_V0(vec, q, p, Nbase, T, bc):
    """
    Evaluates the 1d FEM field in the space V0 at the points q.
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    q : np.array
        evaluation points
        
    p : int
        spline degree
        
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    bc : boolean
        boundary conditions (True = periodic, False = else)
        
    Returns
    -------
    eva : np.arrray
        the function values at the points q
    """
    
    N = bsp.collocation_matrix(T, p, q, bc)
              
    eva = np.dot(N, vec) 
        
    return eva
#==============================================================================================================================



#==============================================================================================================================
def evaluate_field_V1(vec, q, p, Nbase, T, bc):
    """
    Evaluates the 1d FEM field in the space V1 at the points q.
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    q : np.array
        evaluation points
        
    p : int
        spline degree
        
    Nbase : int
        number of spline functions
        
    T : np.array
        knot vector
        
    bc : boolean
        boundary conditions (True = periodic, False = else)
        
    Returns
    -------
    eva : np.arrray
        the function values at the points q
    """
    
    t = T[1:-1]
    
    D = bsp.collocation_matrix(t, p - 1, q, bc, normalize=True)
    
    eva = np.dot(D, vec)
    
    return eva
#==============================================================================================================================



#==============================================================================================================================
def evaluate_field_V0(vec, q, p, Nbase, T, bc):
    """
    Evaluates the 3d FEM field of the space V0 at the tensor grid given by q = (q1, q2, q2).
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    q : list of np.arrays
        evaluation points in each direction
        
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions)
        
    Returns
    -------
    eva : np.array
        the function values at the points q.
    """
    
    q1, q2, q3 = q
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc
    

    N1 = sparse.csr_matrix(bsp.collocation_matrix(T1, p1, q1, bc_1))
    N2 = sparse.csr_matrix(bsp.collocation_matrix(T2, p2, q2, bc_2))
    N3 = sparse.csr_matrix(bsp.collocation_matrix(T3, p3, q3, bc_3))
    
    eva = (sparse.kron(sparse.kron(N1, N2), N3)).dot(vec)
    
    return eva
#==============================================================================================================================



#==============================================================================================================================
def evaluate_field_V1(vec, q, p, Nbase, T, bc):
    """
    Evaluates the components of the 3d FEM field of the space V1 at the tensor grid given by q = (q1, q2, q3).
    
    Parameters
    ----------
    vec : list of np.arrays
        coefficient vectors in each direction
        
    q : list of np.arrays
        evaluation points in each direction
        
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = else)
        
    Returns
    -------
    eva : np.array
        the function values at the points q.
    """
    
    q1, q2, q3 = q
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc

    t1 = T1[1:-1]
    t2 = T2[1:-1]
    t3 = T3[1:-1]
    
    N1 = sparse.csr_matrix(bsp.collocation_matrix(T1, p1, q1, bc_1))
    N2 = sparse.csr_matrix(bsp.collocation_matrix(T2, p2, q2, bc_2))
    N3 = sparse.csr_matrix(bsp.collocation_matrix(T3, p3, q3, bc_3))
    
    D1 = sparse.csr_matrix(bsp.collocation_matrix(t1, p1 - 1, q1, bc_1, normalize=True))
    D2 = sparse.csr_matrix(bsp.collocation_matrix(t2, p2 - 1, q2, bc_2, normalize=True))
    D3 = sparse.csr_matrix(bsp.collocation_matrix(t3, p3 - 1, q3, bc_3, normalize=True))
    
    eva1 = (sparse.kron(sparse.kron(D1, N2), N3)).dot(vec[0])
    eva2 = (sparse.kron(sparse.kron(N1, D2), N3)).dot(vec[1])
    eva3 = (sparse.kron(sparse.kron(N1, N2), D3)).dot(vec[2])
    
    return [eva1, eva2, eva3]
#==============================================================================================================================



#==============================================================================================================================
def evaluate_field_V2(vec, q, p, Nbase, T, bc):
    """
    Evaluates the components of the 3d FEM field of the space V2 at the tensor grid given by q = (q1, q2, q3).
    
    Parameters
    ----------
    vec : list of np.arrays
        coefficient vectors in each direction
        
    q : list of np.arrays
        evaluation points in each direction
        
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = else)
        
    Returns
    -------
    eva : np.array
        the function values at the points q.
    """
    
    q1, q2, q3 = q
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc

    t1 = T1[1:-1]
    t2 = T2[1:-1]
    t3 = T3[1:-1]
    
    N1 = sparse.csr_matrix(bsp.collocation_matrix(T1, p1, q1, bc_1))
    N2 = sparse.csr_matrix(bsp.collocation_matrix(T2, p2, q2, bc_2))
    N3 = sparse.csr_matrix(bsp.collocation_matrix(T3, p3, q3, bc_3))
    
    D1 = sparse.csr_matrix(bsp.collocation_matrix(t1, p1 - 1, q1, bc_1, normalize=True))
    D2 = sparse.csr_matrix(bsp.collocation_matrix(t2, p2 - 1, q2, bc_2, normalize=True))
    D3 = sparse.csr_matrix(bsp.collocation_matrix(t3, p3 - 1, q3, bc_3, normalize=True))
    
    eva1 = (sparse.kron(sparse.kron(N1, D2), D3)).dot(vec[0])
    eva2 = (sparse.kron(sparse.kron(D1, N2), D3)).dot(vec[1])
    eva3 = (sparse.kron(sparse.kron(D1, D2), N3)).dot(vec[2])
    
    return [eva1, eva2, eva3]
#==============================================================================================================================



#==============================================================================================================================
def evaluate_field_V3(vec, q, p, Nbase, T, bc):
    """
    Evaluates the 3d FEM field of the space V3 at the tensor grid given by q = (q1, q2, q2).
    
    Parameters
    ----------
    vec : np.array
        coefficient vector
        
    q : list of np.arrays
        evaluation points in each direction
        
    p : list of ints
        spline degrees in each direction
        
    Nbase : list of ints
        number of spline functions in each direction
        
    T : list of np.arrays
        knot vectors
        
    bc : list of booleans
        boundary conditions in each direction (True = periodic, False = homogeneous Dirichlet, None = no boundary conditions)
        
    Returns
    -------
    eva : np.array
        the function values at the points q.
    """
    
    q1, q2, q3 = q
    p1, p2, p3 = p
    Nbase_1, Nbase_2, Nbase_3 = Nbase
    T1, T2, T3 = T
    bc_1, bc_2, bc_3 = bc
    
    t1 = T1[1:-1]
    t2 = T2[1:-1]
    t3 = T3[1:-1]
    
    
    D1 = sparse.csr_matrix(bsp.collocation_matrix(t1, p1 - 1, q1, bc_1, normalize=True))
    D2 = sparse.csr_matrix(bsp.collocation_matrix(t2, p2 - 1, q2, bc_2, normalize=True))
    D3 = sparse.csr_matrix(bsp.collocation_matrix(t3, p3 - 1, q3, bc_3, normalize=True))
    
    eva = (sparse.kron(sparse.kron(D1, D2), D3)).dot(vec)
    
    return eva
#==============================================================================================================================