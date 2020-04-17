'''
 Module for evaluating functions living in the discrete subspaces of the de Rham complex. 
 Periodic and Dirichlet boundary conditions are available.
 
 Written 2019/20 by Florian Holderied and Stefan Possanner
'''

import numpy        as np
import scipy.sparse as sparse
import hylife.utilitis_FEEC.bsplines as bsp


__all__ = ['FEM_field_1d',
           'FEM_field_2d',
           'FEM_field_3d']


#==============================================================================================================================
def FEM_field_1d(coeff, basis, q, T, p, bc):
    """
    Evaluates the 1d FEM field in the space V0 at the points q.
    
    Parameters
    ----------
    coeff : ndarray
        1d coefficient vector
        
    basis : int
        The basis in which the FEM field is expandend.
             0: V0
             1: V1
        
    q : ndarray
        1d evaluation points
        
    T : ndarray
        1d knot vector defining the spline basis
    
    p : int
        spline degree 
        
    bc : boolean
        boundary conditions (True = periodic, False = clamped)
        
    Returns
    -------
    eva : ndarray
        the function values at the points q
    """

    if basis==0:
        eva = np.dot( bsp.collocation_matrix(T, p, q, bc), coeff ) 
    elif basis==1:
        eva = np.dot( bsp.collocation_matrix(T[1:-1], p - 1, q, bc, normalize=True), coeff )
    else:
        print('WARNING: no basis matched.')
        
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
def FEM_field_3d(coeff, basis, q, T, p, bc):
    """
    Evaluates the 3d FEM field in the suitable basis at the tensor-grid given by q = [q1, q2, q3].
    
    Parameters
    ----------
    coeff : ndarray
        FEM coefficient vectors (either 3d or 1d flattened)
        
    basis : int
        The basis in which the FEM field is expandend.
             0: V0
            11: first  component of V1
            12: second component of V1
            13: third  component of V1
            21: first  component of V2
            22: second component of V2
            23: third  component of V2
             3: V3
        
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
    
    N = [sparse.csr_matrix(bsp.collocation_matrix(T, p, q, bc))
         for T, p, q, bc in zip(T, p, q, bc)]
    D = [sparse.csr_matrix(bsp.collocation_matrix(T[1:-1], p - 1, q, bc, normalize=True)) 
         for T, p, q, bc in zip(T, p, q, bc)]
    
    if basis==0:
        eva = sparse.kron(sparse.kron(N[0], N[1]), N[2]).dot(coeff.flatten()).reshape(
                                                                    len(q[0]), len(q[1]), len(q[2]))
    elif basis==11:
        eva = sparse.kron(sparse.kron(D[0], N[1]), N[2]).dot(coeff.flatten()).reshape(
                                                                    len(q[0]), len(q[1]), len(q[2]))
    elif basis==12:
        eva = sparse.kron(sparse.kron(N[0], D[1]), N[2]).dot(coeff.flatten()).reshape(
                                                                    len(q[0]), len(q[1]), len(q[2]))
    elif basis==13:
        eva = sparse.kron(sparse.kron(N[0], N[1]), D[2]).dot(coeff.flatten()).reshape(
                                                                    len(q[0]), len(q[1]), len(q[2]))
    elif basis==21:
        eva = sparse.kron(sparse.kron(N[0], D[1]), D[2]).dot(coeff.flatten()).reshape(
                                                                    len(q[0]), len(q[1]), len(q[2]))
    elif basis==22:
        eva = sparse.kron(sparse.kron(D[0], N[1]), D[2]).dot(coeff.flatten()).reshape(
                                                                    len(q[0]), len(q[1]), len(q[2]))
    elif basis==23:
        eva = sparse.kron(sparse.kron(D[0], D[1]), N[2]).dot(coeff.flatten()).reshape(
                                                                    len(q[0]), len(q[1]), len(q[2]))
    elif basis==3:
        eva = sparse.kron(sparse.kron(D[0], D[1]), D[2]).dot(coeff.flatten()).reshape(
                                                                    len(q[0]), len(q[1]), len(q[2]))
    else:
        print('WARNING: no basis matched.')
    
    return eva
#==============================================================================================================================