import numpy as np
import scipy.sparse as sparse
import utilitis_FEEC.bsplines as bsp




#==============================================================================================================================
def FEM_field_V0_1d(coeff, q, T, p, bc):
    """
    Evaluates the 1d FEM field in the space V0 at the points q.
    
    Parameters
    ----------
    coeff : ndarray
        1d coefficient vector
        
    q : ndarray
        1d evaluation points
        
    T : ndarray
        1d knot vector defining the spline basis
    
    p : int
        spline degree 
        
    bc : boolean
        boundary conditions (True = periodic, False = else)
        
    Returns
    -------
    eva : ndarray
        the function values at the points q
    """
    
    N = bsp.collocation_matrix(T, p, q, bc)
              
    eva = np.dot(N, coeff) 
        
    return eva
#==============================================================================================================================



#==============================================================================================================================
def FEM_field_V1_1d(coeff, q, T, p, bc):
    """
    Evaluates the 1d FEM field in the space V1 at the points q.
    
    Parameters
    ----------
    coeff : ndarray
        1d coefficient vector
        
    q : ndarray
        1d evaluation points
        
    T : ndarray
        1d knot vector defining the spline basis
    
    p : int
        spline degree 
        
    bc : boolean
        boundary conditions (True = periodic, False = else)
        
    Returns
    -------
    eva : ndarray
        the function values at the points q
    """
    
    t = T[1:-1]
    
    D = bsp.collocation_matrix(t, p - 1, q, bc, normalize=True)
    
    eva = np.dot(D, coeff)
    
    return eva
#==============================================================================================================================




#==============================================================================================================================
def FEM_field_V0_2d(coeff, q, T, p, bc):
    """
    Evaluates the 2d FEM field in the space V2 at the tensor-grid given by q = [q1, q2].
    
    Parameters
    ----------
    coeff : ndarray
        1d coefficient vector (flattened)
        
    q : list of ndarrays
        1d evaluation points in each direction
        
    T : list of ndarrays
        1d knot vectors defining the spline basis
    
    p : list of ints
        spline degrees 
        
    bc : list of booleans
        boundary conditions (True = periodic, False = else)
        
    Returns
    -------
    eva : ndarray
        the function values at the points q (flattened)
    """

    N = [sparse.csr_matrix(bsp.collocation_matrix(T, p, q, bc)) for T, p, q, bc in zip(T, p, q, bc)]
    
    eva = sparse.kron(N[0], N[1]).dot(coeff)
    
    return eva
#==============================================================================================================================




#==============================================================================================================================
def FEM_field_V1curl_2d(coeff, q, T, p, bc):
    """
    Evaluates the 2d FEM field in the space V1 (curl) at the tensor-grid given by q = [q1, q2].
    
    Parameters
    ----------
    coeff : list of ndarrays
        1d coefficient vector for each component (flattened)
        
    q : list of ndarrays
        1d evaluation points in each direction
        
    T : list of ndarrays
        1d knot vectors defining the spline basis
    
    p : list of ints
        spline degrees 
        
    bc : list of booleans
        boundary conditions (True = periodic, False = else)
        
    Returns
    -------
    eva : list of ndarrays
        evaluated values at the points q for each component (flattened)
    """
    
    t = [T[1:-1] for T in T]
    
    N = [sparse.csr_matrix(bsp.collocation_matrix(T, p, q, bc)) for T, p, q, bc in zip(T, p, q, bc)]
    D = [sparse.csr_matrix(bsp.collocation_matrix(t, p - 1, q, bc, normalize=True)) for t, p, q, bc in zip(t, p, q, bc)]
    
    eva1 = sparse.kron(D[0], N[1]).dot(coeff[0])
    eva2 = sparse.kron(N[0], D[1]).dot(coeff[1])
    
    return [eva1, eva2]
#==============================================================================================================================




#==============================================================================================================================
def FEM_field_V1div_2d(coeff, q, T, p, bc):
    """
    Evaluates the 2d FEM field in the space V1 (div) at the tensor-grid given by q = [q1, q2].
    
    Parameters
    ----------
    coeff : list of ndarrays
        1d coefficient vector for each component (flattened)
        
    q : list of ndarrays
        1d evaluation points in each direction
        
    T : list of ndarrays
        1d knot vectors defining the spline basis
    
    p : list of ints
        spline degrees 
        
    bc : list of booleans
        boundary conditions (True = periodic, False = else)
        
    Returns
    -------
    eva : list of ndarrays
        evaluated values at the points q for each component (flattened)
    """
    
    t = [T[1:-1] for T in T]
    
    N = [sparse.csr_matrix(bsp.collocation_matrix(T, p, q, bc)) for T, p, q, bc in zip(T, p, q, bc)]
    D = [sparse.csr_matrix(bsp.collocation_matrix(t, p - 1, q, bc, normalize=True)) for t, p, q, bc in zip(t, p, q, bc)]
    
    eva1 = sparse.kron(N[0], D[1]).dot(coeff[0])
    eva2 = sparse.kron(D[0], N[1]).dot(coeff[1])
    
    return [eva1, eva2]
#==============================================================================================================================





#==============================================================================================================================
def FEM_field_V2_2d(coeff, q, T, p, bc):
    """
    Evaluates the 2d FEM field in the space V2 at the tensor-grid given by q = [q1, q2].
    
    Parameters
    ----------
    coeff : ndarray
        1d coefficient vector (flattened)
        
    q : list of ndarrays
        1d evaluation points in each direction
        
    T : list of ndarrays
        1d knot vectors defining the spline basis
    
    p : list of ints
        spline degrees 
        
    bc : list of booleans
        boundary conditions (True = periodic, False = else)
        
    Returns
    -------
    eva : ndarray
        the function values at the points q (flattened)
    """
    
    t = [T[1:-1] for T in T]
    
    D = [sparse.csr_matrix(bsp.collocation_matrix(t, p - 1, q, bc, normalize=True)) for t, p, q, bc in zip(t, p, q, bc)]
    
    eva = sparse.kron(D[0], D[1]).dot(coeff)
    
    return eva
#==============================================================================================================================




#==============================================================================================================================
def FEM_field_V0_3d(coeff, q, T, p, bc):
    """
    Evaluates the 3d FEM field in the space V0 at the tensor-grid given by q = [q1, q2, q3].
    
    Parameters
    ----------
    coeff : ndarray
        1d coefficient vector (flattened)
        
    q : list of ndarrays
        1d evaluation points in each direction
        
    T : list of ndarrays
        1d knot vectors defining the spline basis
    
    p : list of ints
        spline degrees 
        
    bc : list of booleans
        boundary conditions (True = periodic, False = else)
        
    Returns
    -------
    eva : ndarray
        the function values at the points q (flattened)
    """

    N = [sparse.csr_matrix(bsp.collocation_matrix(T, p, q, bc)) for T, p, q, bc in zip(T, p, q, bc)]
    
    eva = sparse.kron(sparse.kron(N[0], N[1]), N[2]).dot(coeff)
    
    return eva
#==============================================================================================================================



#==============================================================================================================================
def FEM_field_V1_3d(coeff, q, T, p, bc):
    """
    Evaluates the 3d FEM field in the space V1 at the tensor-grid given by q = [q1, q2, q3].
    
    Parameters
    ----------
    coeff : list of ndarrays
        1d coefficient vector for each component (flattened)
        
    q : list of ndarrays
        1d evaluation points in each direction
        
    T : list of ndarrays
        1d knot vectors defining the spline basis
    
    p : list of ints
        spline degrees 
        
    bc : list of booleans
        boundary conditions (True = periodic, False = else)
        
    Returns
    -------
    eva : list of ndarrays
        evaluated values at the points q for each component (flattened)
    """
    
    t = [T[1:-1] for T in T]
    
    N = [sparse.csr_matrix(bsp.collocation_matrix(T, p, q, bc)) for T, p, q, bc in zip(T, p, q, bc)]
    D = [sparse.csr_matrix(bsp.collocation_matrix(t, p - 1, q, bc, normalize=True)) for t, p, q, bc in zip(t, p, q, bc)]
    
    eva1 = sparse.kron(sparse.kron(D[0], N[1]), N[2]).dot(coeff[0])
    eva2 = sparse.kron(sparse.kron(N[0], D[1]), N[2]).dot(coeff[1])
    eva3 = sparse.kron(sparse.kron(N[0], N[1]), D[2]).dot(coeff[2])
    
    return [eva1, eva2, eva3]
#==============================================================================================================================



#==============================================================================================================================
def FEM_field_V2_3d(coeff, q, T, p, bc):
    """
    Evaluates the 3d FEM field in the space V2 at the tensor-grid given by q = [q1, q2, q3].
    
    Parameters
    ----------
    coeff : list of ndarrays
        1d coefficient vector for each component (flattened)
        
    q : list of ndarrays
        1d evaluation points in each direction
        
    T : list of ndarrays
        1d knot vectors defining the spline basis
    
    p : list of ints
        spline degrees 
        
    bc : list of booleans
        boundary conditions (True = periodic, False = else)
        
    Returns
    -------
    eva : list of ndarrays
        evaluated values at the points q for each component (flattened)
    """
    
    t = [T[1:-1] for T in T]
    
    N = [sparse.csr_matrix(bsp.collocation_matrix(T, p, q, bc)) for T, p, q, bc in zip(T, p, q, bc)]
    D = [sparse.csr_matrix(bsp.collocation_matrix(t, p - 1, q, bc, normalize=True)) for t, p, q, bc in zip(t, p, q, bc)]
    
    eva1 = sparse.kron(sparse.kron(N[0], D[1]), D[2]).dot(coeff[0])
    eva2 = sparse.kron(sparse.kron(D[0], N[1]), D[2]).dot(coeff[1])
    eva3 = sparse.kron(sparse.kron(D[0], D[1]), N[2]).dot(coeff[2])
    
    return [eva1, eva2, eva3]
#==============================================================================================================================



#==============================================================================================================================
def FEM_field_V3_3d(coeff, q, T, p, bc):
    """
    Evaluates the 3d FEM field in the space V3 at the tensor-grid given by q = [q1, q2, q3].
    
    Parameters
    ----------
    coeff : ndarray
        1d coefficient vector (flattened)
        
    q : list of ndarrays
        1d evaluation points in each direction
        
    T : list of ndarrays
        1d knot vectors defining the spline basis
    
    p : list of ints
        spline degrees 
        
    bc : list of booleans
        boundary conditions (True = periodic, False = else)
        
    Returns
    -------
    eva : ndarray
        the function values at the points q (flattened)
    """
    
    t = [T[1:-1] for T in T]

    D = [sparse.csr_matrix(bsp.collocation_matrix(t, p - 1, q, bc, normalize=True)) for t, p, q, bc in zip(t, p, q, bc)]
    
    eva = sparse.kron(sparse.kron(D[0], D[1]), D[2]).dot(coeff)
    
    return eva
#==============================================================================================================================