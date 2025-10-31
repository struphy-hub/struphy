import time

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from struphy.bsplines.bsplines import basis_funs, find_span
from struphy.bsplines.evaluation_kernels_1d import evaluation_kernel_1d
from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.local_projectors_kernels import fill_matrix_column
from struphy.feec.psydac_derham import Derham
from struphy.feec.utilities_local_projectors import get_one_spline, get_span_and_basis, get_values_and_indices_splines

from psydac.linalg.solvers import inverse
from struphy.feec import preconditioner
from psydac.linalg.basic  import VectorSpace, Vector, LinearOperator
from psydac.fem.projectors import knot_insertion_projection_operator
from struphy.feec.linear_operators import LinOpWithTransp
from psydac.fem.basic import FemSpace
from psydac.fem.tensor import TensorFemSpace
from struphy.feec.mass import WeightedMassOperators
from struphy.geometry.domains import Tokamak, Cuboid, HollowCylinder
from struphy.fields_background.equils import AdhocTorusQPsi
from math import comb, log2
import random


def jacobi(A, b, x_init=None, tol=1e-6, max_iter=1000, verbose = False):
    """
    Solves the linear system Ax = b using the Jacobi iterative method.
    
    Parameters:
    A : numpy.ndarray
        Coefficient matrix (2D array)
    b : numpy.ndarray
        Right-hand side vector (1D array)
    x_init : numpy.ndarray, optional
        Initial guess for the solution (1D array)
    tol : float, optional
        Convergence tolerance (default: 1e-10)
    max_iter : int, optional
        Maximum number of iterations (default: 1000)
    
    Returns:
    x : numpy.ndarray
        Solution vector
    """
    n = A.shape[0]
    x = x_init if x_init is not None else np.zeros(n)
    x_new = np.zeros(n)
    converged = False
    for itterations in range(max_iter):
        for i in range(n):
            sum_except_i = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - sum_except_i) / A[i, i]
        
        
        error = np.linalg.norm(b - np.dot(A,x_new), ord=np.inf)
        if error < tol:
            converged = True
            if verbose:
                print(f'{converged = }')
                print(f'{itterations = }')
                print(f'{error = }')
            return x_new, itterations
        
        x[:] = x_new  # Update x
    
    
    if verbose:
        print(f'{converged = }')
        print(f'{itterations = }')
        print(f'{error = }')
    return x_new, itterations
    #raise ValueError("Jacobi method did not converge within the maximum number of iterations")


def will_jacobi_converge(A, verbose = False):
    converges = False
    #We get the diagonal, lower triangular and uper tireangular parts of A
    Darr = np.diag(np.diag(A))
    Darr_inv = np.diag(1.0 / np.diag(Darr)) 
    Larr = np.tril(A, k=-1)
    Uarr = np.triu(A, k=1)
    #Then the Jacobi iteration matrix is
    Jarr = np.matmul(Darr_inv,(Larr+Uarr))
    #Now we get its eigenvalues
    eigenvalues = np.linalg.eigvals(Jarr)
    max_eigen_value = abs(eigenvalues[np.argmax(np.abs(eigenvalues))])
    
    if max_eigen_value < 1.0:
        converges = True
    
    if(verbose):
        print(f'{max_eigen_value = }')
    
    return converges
    

def will_gauss_seidel_converge(A, verbose = False):
    converges = False
    #We get the diagonal, lower triangular and uper triangular parts of A
    Darr = np.diag(np.diag(A))
    Larr = np.tril(A, k=-1)
    Uarr = np.triu(A, k=1)
    Aux =  np.linalg.inv(Darr - Larr)
    #Then the Gauss-Seidel iteration matrix is
    Garr = np.matmul(Aux, Uarr)
    #Now we get its eigenvalues
    eigenvalues = np.linalg.eigvals(Garr)
    max_eigen_value = abs(eigenvalues[np.argmax(np.abs(eigenvalues))])
    
    if max_eigen_value < 1.0:
        converges = True
    
    if(verbose):
        print(f'{max_eigen_value = }')
    
    return converges


def from_array_to_psydac(x_vector, fem_space):
    #fem_space = derham.Vh_fem[sp_key]
    symbolic_name = fem_space.symbolic_space.name
    
    if(symbolic_name == 'H1' or symbolic_name == "L2"):
        spaces = [fem_space.spaces]
        N = [spaces[0][i].nbasis for i in range(3)]
        starts = np.array(fem_space.vector_space.starts)
        x = fem_space.vector_space.zeros()
        
        cont= 0
        for i0 in range(N[0]):
            for i1 in range(N[1]):
                for i2 in range(N[2]):
                    x[starts[0]+i0,starts[1]+i1,starts[2]+i2] = x_vector[cont]
                    cont += 1
            
    else:
        spaces = [comp.spaces for comp in fem_space.spaces]
        N = [[spaces[h][i].nbasis for i in range(3)] for h in range(3)]
        starts = np.array([vi.starts for vi in fem_space.vector_space.spaces])
        x = fem_space.vector_space.zeros()
        
        cont = 0
        for h in range(3):
            for i0 in range(N[h][0]):
                for i1 in range(N[h][1]):
                    for i2 in range(N[h][2]):
                        x[h][starts[h][0]+i0,starts[h][1]+i1,starts[h][2]+i2] = x_vector[cont]
                        cont += 1
    return x
    
    
def direct_solver(A_inv,b, fem_space):
    # A_inv is already the inverse matrix of A
    #fem_space = derham.Vh_fem[sp_key]
    symbolic_name = fem_space.symbolic_space.name
    
    if(symbolic_name == 'H1' or symbolic_name == "L2"):
        spaces = [fem_space.spaces]
        N = [spaces[0][i].nbasis for i in range(3)]
        starts = np.array(fem_space.vector_space.starts)
        
        b_vector = remove_padding(fem_space, b)
        x_vector = np.dot(A_inv, b_vector)
        x = fem_space.vector_space.zeros()
        
        cont= 0
        for i0 in range(N[0]):
            for i1 in range(N[1]):
                for i2 in range(N[2]):
                    x[starts[0]+i0,starts[1]+i1,starts[2]+i2] = x_vector[cont]
                    cont += 1
            
    else:
        spaces = [comp.spaces for comp in fem_space.spaces]
        N = [[spaces[h][i].nbasis for i in range(3)] for h in range(3)]
        starts = np.array([vi.starts for vi in fem_space.vector_space.spaces])
        
        b_vector = remove_padding(fem_space, b)
        x_vector = np.dot(A_inv, b_vector)
        x = fem_space.vector_space.zeros()
        
        cont = 0
        for h in range(3):
            for i0 in range(N[h][0]):
                for i1 in range(N[h][1]):
                    for i2 in range(N[h][2]):
                        x[h][starts[h][0]+i0,starts[h][1]+i1,starts[h][2]+i2] = x_vector[cont]
                        cont += 1
    return x
    

def get_b_spline_degree(V):
    """
    Determines the degree of the B-splines.

    Parameters
    ----------
    V : psydac.fem.basic.FemSpace
        Finite element spline space (domain, input space).
        
    Returns
    -------
    p : numpy array
        numpy array of 3 ints containing the B-spline degrees for each spatial direction.
    """
    
    assert isinstance(V, FemSpace)
    p = np.zeros(3, dtype=int)
    
    if hasattr(V.symbolic_space, "name"):
        V_name = V.symbolic_space.name
    
    if(V_name == "H1"):
        for i, space in enumerate(V.spaces):
            p[i] = space.degree
    elif(V_name == "L2"):
        for i, space in enumerate(V.spaces):
            p[i] = space.degree+1
    elif(V_name == "Hcurl"):
        V1ds = [comp.spaces for comp in V.spaces]
        for i in range(3):
            p[i] = V1ds[i][i].degree+1
    elif(V_name == "Hdiv"):
        V1ds = [comp.spaces for comp in V.spaces]
        for i in range(3):
            p[i] = V1ds[i][i].degree
    elif(V_name == "H1H1H1"):
        V1ds = [comp.spaces for comp in V.spaces]
        for i in range(3):
            p[i] = V1ds[i][i].degree
    else:
        raise Exception("Invalid symbolic name.")
    
    return p


def remove_padding(fem_space, v):
    
    #fem_space = derham.Vh_fem[sp_key]
    symbolic_name = fem_space.symbolic_space.name
    
    if(symbolic_name == 'H1' or symbolic_name == "L2"):
    
        spaces = [fem_space.spaces]
        N = [spaces[0][i].nbasis for i in range(3)]
        
        starts = np.array(fem_space.vector_space.starts)
        
        #To make it easier to read I will extract the data out of out disregarding all the padding it come with
        v_array = np.zeros(N[0]*N[1]*N[2], dtype=float)
        cont = 0
        for i0 in range(N[0]):
            for i1 in range(N[1]):
                for i2 in range(N[2]):
                    v_array[cont] = v[starts[0]+i0,starts[1]+i1,starts[2]+i2]
                    cont += 1
            
    else:
        spaces = [comp.spaces for comp in fem_space.spaces]
        N = [[spaces[h][i].nbasis for i in range(3)] for h in range(3)]
        
        starts = np.array([vi.starts for vi in fem_space.vector_space.spaces])
        
        #To make it easier to read I will extract the data out of out disregarding all the padding it come with
        v_array = np.zeros(N[0][0]*N[0][1]*N[0][2]+N[1][0]*N[1][1]*N[1][2]+N[2][0]*N[2][1]*N[2][2], dtype=float)
        cont = 0
        for h in range(3):
            for i0 in range(N[h][0]):
                for i1 in range(N[h][1]):
                    for i2 in range(N[h][2]):
                        v_array[cont] = v[h][starts[h][0]+i0,starts[h][1]+i1,starts[h][2]+i2]
                        cont += 1
        
        
    return v_array


class RestrictionOperator(LinOpWithTransp):
    """
    Linear operator which operates between vector spaces of the same kind but different resolutions.

    We assume that the vectors in the domain belong to a De-rham space with n elements, while the codomain
    belong to the coarser De-rham space with n/2 elements.
    
    At the moment we also assume that we are halving only the first spatial direction.

    Parameters
    ----------
    V : psydac.fem.basic.FemSpace
        Finite element spline space (domain, input space).

    W : psydac.fem.basic.FemSpace
        Finite element spline space (codomain, output space).

    """
    def __init__(self, V, W):

        # Check domain and codomain
        assert isinstance(V, FemSpace)
        assert isinstance(W, FemSpace)

        self._V = V
        self._W = W
        
        # Store info in object
        self._domain   = V.vector_space
        self._codomain = W.vector_space
        self._dtype = V.vector_space.dtype
        
        #Can be "H1", "L2", "Hcurl", "Hdiv", "H1H1H1"
        self._V_name = V.symbolic_space.name
        self._W_name = W.symbolic_space.name
        assert(self._V_name == self._W_name)
        
        #This list will tell us in which spatial direction we are halving the problem.
        self._halving_directions = [False,False,False]
        
        # input space: 3d StencilVectorSpaces and 1d SplineSpaces of each component
        if isinstance(V, TensorFemSpace):
            self._V1ds = [V.spaces]
            self._VNbasis = np.array([self._V1ds[0][0].nbasis, self._V1ds[0][1].nbasis, self._V1ds[0][2].nbasis])
            
            # We get the start and endpoint for each sublist in input
            self._in_starts = np.array(V.vector_space.starts)
            self._in_ends = np.array(V.vector_space.ends)
        else:
            self._V1ds = [comp.spaces for comp in V.spaces]
            self._VNbasis = np.array(
                [
                    [self._V1ds[0][0].nbasis, self._V1ds[0][1].nbasis, self._V1ds[0][2].nbasis],
                    [
                        self._V1ds[1][0].nbasis,
                        self._V1ds[1][1].nbasis,
                        self._V1ds[1][2].nbasis,
                    ],
                    [self._V1ds[2][0].nbasis, self._V1ds[2][1].nbasis, self._V1ds[2][2].nbasis],
                ]
            )
            
            # We get the start and endpoint for each sublist in input
            self._in_starts = np.array([vi.starts for vi in V.vector_space.spaces])
            self._in_ends = np.array([vi.ends for vi in V.vector_space.spaces])
            
        # output space: 3d StencilVectorSpaces and 1d SplineSpaces of each component
        if isinstance(W, TensorFemSpace):
            self._W1ds = [W.spaces]
            self._WNbasis = np.array([self._W1ds[0][0].nbasis, self._W1ds[0][1].nbasis, self._W1ds[0][2].nbasis])
            
            for i in range(3):
                if(self._WNbasis[i] < self._VNbasis[i]):
                    #If this breaks for clamped splines it means .nbasis gives you the number of basis functions, not the number of elements
                    assert self._VNbasis[i] == self._WNbasis[i]*2
                    self._halving_directions[i] = True
            
            # We get the start and endpoint for each sublist in out
            self._out_starts = np.array(W.vector_space.starts)
            self._out_ends = np.array(W.vector_space.ends)
            
        else:
            self._W1ds = [comp.spaces for comp in W.spaces]
            self._WNbasis = np.array(
                [
                    [self._W1ds[0][0].nbasis, self._W1ds[0][1].nbasis, self._W1ds[0][2].nbasis],
                    [
                        self._W1ds[1][0].nbasis,
                        self._W1ds[1][1].nbasis,
                        self._W1ds[1][2].nbasis,
                    ],
                    [self._W1ds[2][0].nbasis, self._W1ds[2][1].nbasis, self._W1ds[2][2].nbasis],
                ]
            )
            for i in range(3):
                if(self._WNbasis[1][i] < self._VNbasis[1][i]):
                    #If this breaks for clamped splines it means .nbasis gives you the number of basis functions, not the number of elements
                    assert self._VNbasis[0][i] == self._WNbasis[0][i]*2 and self._VNbasis[1][i] == self._WNbasis[1][i]*2 and self._VNbasis[2][i] == self._WNbasis[2][i]*2
                    self._halving_directions[i] = True
                    
            # We get the start and endpoint for each sublist in out
            self._out_starts = np.array([vi.starts for vi in W.vector_space.spaces])
            self._out_ends = np.array([vi.ends for vi in W.vector_space.spaces])
            
        
        
        # Degree of the B-spline space, not to be confused with the degrees given by fem_space.spaces.degree since depending on the situation 
        # it will give the D-spline degree instead
        self._p = get_b_spline_degree(V)
        
        #We also get the D-spline degree
        self._pD = self._p - 1
        
        #Now we compute the weights that define this linear operator
        
        #We begin by defining a list that will contain the 3 numpy arrays, each one with the weights for one spatial direction.
        #In the case there are direction over which we do not halve the resolution we shall have an array with only one 1.0
        self._all_weights = []
        self._all_weightsD = []
        for i in range(3):
            if self._halving_directions[i]:
                #Here we store the weights needed for B-splines
                weights = np.zeros(self._p[i]+2, dtype=float)
                #Here we store the weights needed for D-splines
                weightsD = np.zeros(self._pD[i]+2, dtype=float)
                for j in range(self._p[i]+2):
                    weights[j] = 2.0**(-self._p[i])*comb(self._p[i]+1,j)
                for j in range(self._pD[i]+2):
                    weightsD[j] = 2.0**-(self._pD[i]+1)*comb(self._pD[i]+1,j)
                self._all_weights.append(weights)
                self._all_weightsD.append(weightsD)
            else:
                self._all_weights.append(np.array([1.0],dtype=float))
                self._all_weightsD.append(np.array([1.0],dtype=float))
        
        
        
    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype
    
    def _dot_helper(self, v, out, p, weights, h=None):
        """Helper function to perform dot product computation."""
        #First we get the number of weights in each direction
        weights_len = []
        for i in range(3):
            if self._halving_directions[i]:
                weights_len.append(p[i] + 2)
            else:
                weights_len.append(1)
        if h is None:  # Scalar case (H1, L2)
            for i0 in range(self._out_starts[0], self._out_ends[0] + 1):
                for i1 in range(self._out_starts[1], self._out_ends[1] + 1):
                    for i2 in range(self._out_starts[2], self._out_ends[2] + 1):
                        for j0 in range(weights_len[0]):
                            if self._halving_directions[0]:
                                pos0 = (2 * i0 - p[0] + j0) % self._VNbasis[0]
                            else:
                                pos0 = i0
                            for j1 in range(weights_len[1]):
                                if self._halving_directions[1]:
                                    pos1 = (2 * i1 - p[1] + j1) % self._VNbasis[1]
                                else:
                                    pos1 = i1
                                for j2 in range(weights_len[2]):
                                    if self._halving_directions[2]:
                                        pos2 = (2 * i2 - p[2] + j2) % self._VNbasis[2]
                                    else:
                                        pos2 = i2
                                    out[i0, i1, i2] += weights[0][j0]* weights[1][j1] *weights[2][j2] * v[pos0, pos1, pos2]
        else:  # Vector case (Hcurl, Hdiv, H1H1H1)
            for i0 in range(self._out_starts[h][0], self._out_ends[h][0] + 1):
                for i1 in range(self._out_starts[h][1], self._out_ends[h][1] + 1):
                    for i2 in range(self._out_starts[h][2], self._out_ends[h][2] + 1):
                        for j0 in range(weights_len[0]):
                            if self._halving_directions[0]:
                                pos0 = (2 * i0 - p[0] + j0) % self._VNbasis[h][0]
                            else:
                                pos0 = i0
                            for j1 in range(weights_len[1]):
                                if self._halving_directions[1]:
                                    pos1 = (2 * i1 - p[1] + j1) % self._VNbasis[h][1]
                                else:
                                    pos1 = i1
                                for j2 in range(weights_len[2]):
                                    if self._halving_directions[2]:
                                        pos2 = (2 * i2 - p[2] + j2) % self._VNbasis[h][2]
                                    else:
                                        pos2 = i2
                                    out[h][i0, i1, i2] += weights[0][j0]* weights[1][j1] *weights[2][j2] * v[h][pos0, pos1, pos2]
            
        return out

    def dot_H1(self, v, out):
        return self._dot_helper(v, out, self._p, self._all_weights)

    def dot_L2(self, v, out):
        return self._dot_helper(v, out, self._pD, self._all_weightsD)

    def dot_Hcurl(self, v, out):
        out = self._dot_helper(v, out, [self._pD[0],self._p[1],self._p[2]], [self._all_weightsD[0], self._all_weights[1], self._all_weights[2]], h = 0)
        self._dot_helper(v, out, [self._p[0],self._pD[1],self._p[2]], [self._all_weights[0], self._all_weightsD[1], self._all_weights[2]], h = 1)
        return self._dot_helper(v, out, [self._p[0],self._p[1],self._pD[2]], [self._all_weights[0], self._all_weights[1], self._all_weightsD[2]], h = 2)

    def dot_Hdiv(self, v, out):
        out = self._dot_helper(v, out, [self._p[0],self._pD[1],self._pD[2]], [self._all_weights[0], self._all_weightsD[1], self._all_weightsD[2]], h = 0)
        self._dot_helper(v, out, [self._pD[0],self._p[1],self._pD[2]], [self._all_weightsD[0], self._all_weights[1], self._all_weightsD[2]], h = 1)
        return self._dot_helper(v, out, [self._pD[0],self._pD[1],self._p[2]], [self._all_weightsD[0], self._all_weightsD[1], self._all_weights[2]], h = 2)

    def dot_H1H1H1(self, v, out):
        out = self._dot_helper(v, out, self._p, self._all_weights, h=0)
        self._dot_helper(v, out, self._p, self._all_weights, h=1)
        return self._dot_helper(v, out, self._p, self._all_weights, h=2)

    def dot(self, v, out=None):

        assert isinstance(v, Vector) and v.space == self.domain
 
        if out is None:
            out = self.codomain.zeros()   
        else:
            assert isinstance(out, Vector) and out.space == self.codomain
            
            if self._V_name == 'H1' or self._V_name == 'L2':
                for i0 in range(self._out_starts[0], self._out_ends[0]+1):
                    for i1 in range(self._out_starts[1], self._out_ends[1]+1):
                        for i2 in range(self._out_starts[2], self._out_ends[2]+1):
                            out[i0,i1,i2] = 0.0
            else:
                for h in range(3):
                    for i0 in range(self._out_starts[h][0], self._out_ends[h][0]+1):
                        for i1 in range(self._out_starts[h][1], self._out_ends[h][1]+1):
                            for i2 in range(self._out_starts[h][2], self._out_ends[h][2]+1):
                                out[h][i0,i1,i2] = 0.0
            
        dot_methods = {
            "H1": self.dot_H1,
            "L2": self.dot_L2,
            "Hcurl": self.dot_Hcurl,
            "Hdiv": self.dot_Hdiv,
            "H1H1H1": self.dot_H1H1H1,
        }
        
        return dot_methods.get(self._V_name)(v, out)
    
    def transpose(self, *, out = None):
        if out is None:
            out = ExtensionOperator(self._W, self._V)
        else:
            assert isinstance(out, ExtensionOperator)
            assert out.domain is self.codomain
            assert out.codomain is self.domain
            
        return out
    

class ExtensionOperator(LinOpWithTransp):
    """
    Linear operator which operates between vector spaces of the same kind but different resolutions.

    We assume that the vectors in the domain belong to a De-rham space with n/2 elements, while the codomain
    belong to the coarser De-rham space with n elements.
    
    At the moment we also assume that we are halving only the first spatial direction.

    Parameters
    ----------
    V : psydac.fem.basic.FemSpace
        Finite element spline space (domain, input space).

    W : psydac.fem.basic.FemSpace
        Finite element spline space (codomain, output space).

    """
    def __init__(self, V, W):

        # Check domain and codomain
        assert isinstance(V, FemSpace)
        assert isinstance(W, FemSpace)

        self._V = V
        self._W = W
        
        # Store info in object
        self._domain   = V.vector_space
        self._codomain = W.vector_space
        self._dtype = V.vector_space.dtype
        
        #Can be "H1", "L2", "Hcurl", "Hdiv", "H1H1H1"
        self._V_name = V.symbolic_space.name
        self._W_name = W.symbolic_space.name
        assert(self._V_name == self._W_name)
        
        #This list will tell us in which spatial direction we are halving the problem.
        self._halving_directions = [False,False,False]
        
        # input space: 3d StencilVectorSpaces and 1d SplineSpaces of each component
        if isinstance(V, TensorFemSpace):
            self._V1ds = [V.spaces]
            self._VNbasis = np.array([self._V1ds[0][0].nbasis, self._V1ds[0][1].nbasis, self._V1ds[0][2].nbasis])
            
            # We get the start and endpoint for each sublist in input
            self._in_starts = np.array(V.vector_space.starts)
            self._in_ends = np.array(V.vector_space.ends)
        else:
            self._V1ds = [comp.spaces for comp in V.spaces]
            self._VNbasis = np.array(
                [
                    [self._V1ds[0][0].nbasis, self._V1ds[0][1].nbasis, self._V1ds[0][2].nbasis],
                    [
                        self._V1ds[1][0].nbasis,
                        self._V1ds[1][1].nbasis,
                        self._V1ds[1][2].nbasis,
                    ],
                    [self._V1ds[2][0].nbasis, self._V1ds[2][1].nbasis, self._V1ds[2][2].nbasis],
                ]
            )
            
            # We get the start and endpoint for each sublist in input
            self._in_starts = np.array([vi.starts for vi in V.vector_space.spaces])
            self._in_ends = np.array([vi.ends for vi in V.vector_space.spaces])
            
        # output space: 3d StencilVectorSpaces and 1d SplineSpaces of each component
        if isinstance(W, TensorFemSpace):
            self._W1ds = [W.spaces]
            self._WNbasis = np.array([self._W1ds[0][0].nbasis, self._W1ds[0][1].nbasis, self._W1ds[0][2].nbasis])
            
            for i in range(3):
                if(self._VNbasis[i] < self._WNbasis[i]):
                    #If this breaks for clamped splines it means .nbasis gives you the number of basis functions, not the number of elements
                    assert self._WNbasis[i] == self._VNbasis[i]*2
                    self._halving_directions[i] = True
            
            # We get the start and endpoint for each sublist in out
            self._out_starts = np.array(W.vector_space.starts)
            self._out_ends = np.array(W.vector_space.ends)
            
        else:
            self._W1ds = [comp.spaces for comp in W.spaces]
            self._WNbasis = np.array(
                [
                    [self._W1ds[0][0].nbasis, self._W1ds[0][1].nbasis, self._W1ds[0][2].nbasis],
                    [
                        self._W1ds[1][0].nbasis,
                        self._W1ds[1][1].nbasis,
                        self._W1ds[1][2].nbasis,
                    ],
                    [self._W1ds[2][0].nbasis, self._W1ds[2][1].nbasis, self._W1ds[2][2].nbasis],
                ]
            )
            
            for i in range(3):
                if(self._VNbasis[1][i] < self._WNbasis[1][i]):
                    assert self._WNbasis[0][i] == self._VNbasis[0][i]*2 and self._WNbasis[1][i] == self._VNbasis[1][i]*2 and self._WNbasis[2][i] == self._VNbasis[2][i]*2
                    self._halving_directions[i] = True
                    
            # We get the start and endpoint for each sublist in out
            self._out_starts = np.array([vi.starts for vi in W.vector_space.spaces])
            self._out_ends = np.array([vi.ends for vi in W.vector_space.spaces])
            
        
        
        # Degree of the B-spline space, not to be confused with the degrees given by fem_space.spaces.degree since depending on the situation 
        # it will give the D-spline degree instead
        self._p = get_b_spline_degree(V)
        #We also get the D-splines degree
        self._pD = self._p-1
        
        #Now we compute the weights that define this linear operator
        
        #We begin by defining a list that will contain the 3 numpy arrays, each one with the weights for one spatial direction.
        #In the case there are a direction over which we do not halve the resolution we shall have an array with only one 1.0
        self._all_weights_even = []
        self._all_weights_evenD = []
        self._all_weights_odd = []
        self._all_weights_oddD = []
        
        #Each list has 3 integers, each one denoting the number of weights in the corresponding weights array.
        self._all_size_even = []
        self._all_size_evenD = []
        self._all_size_odd = []
        self._all_size_oddD = []
        
        for i in range(3):
            if self._halving_directions[i]:
                #First for B-splines
                if(self._p[i]%2 == 0):
                    size_even = self._p[i]//2 +1
                    size_odd = self._p[i]//2 +1
                    weights_even = np.zeros(size_even, dtype=float)
                    weights_odd = np.zeros(size_odd, dtype=float)
                    for j in range(size_even):
                        weights_even[j] = 2.0**(-self._p[i])*comb(self._p[i]+1,2*j)
                        weights_odd[j] = 2.0**(-self._p[i])*comb(self._p[i]+1,2*j+1)
                else:
                    size_even = (self._p[i]+1)//2 +1
                    size_odd = (self._p[i]-1)//2 +1
                    weights_even = np.zeros(size_even, dtype=float)
                    weights_odd = np.zeros(size_odd, dtype=float)
                    for j in range(size_even):
                        weights_even[j] = 2.0**(-self._p[i])*comb(self._p[i]+1,2*j)
                    for j in range(size_odd):
                        weights_odd[j] = 2.0**(-self._p[i])*comb(self._p[i]+1,2*j+1)
                        
                #Second for D-splines
                if(self._pD[i]%2 == 0):
                    size_evenD = self._pD[i]//2 +1
                    size_oddD = self._pD[i]//2 +1
                    weights_evenD = np.zeros(size_evenD, dtype=float)
                    weights_oddD = np.zeros(size_oddD, dtype=float)
                    for j in range(size_evenD):
                        weights_evenD[j] = 2.0**-(self._pD[i]+1)*comb(self._pD[i]+1,2*j)
                        weights_oddD[j] = 2.0**-(self._pD[i]+1)*comb(self._pD[i]+1,2*j+1)
                else:
                    size_evenD = (self._pD[i]+1)//2 +1
                    size_oddD = (self._pD[i]-1)//2 +1
                    weights_evenD = np.zeros(size_evenD, dtype=float)
                    weights_oddD = np.zeros(size_oddD, dtype=float)
                    for j in range(size_evenD):
                        weights_evenD[j] = 2.0**-(self._pD[i]+1)*comb(self._pD[i]+1,2*j)
                    for j in range(size_oddD):
                        weights_oddD[j] = 2.0**-(self._pD[i]+1)*comb(self._pD[i]+1,2*j+1)
                
                self._all_weights_even.append(weights_even)
                self._all_weights_evenD.append(weights_evenD)
                self._all_weights_odd.append(weights_odd)
                self._all_weights_oddD.append(weights_oddD)
                self._all_size_even.append(size_even)
                self._all_size_evenD.append(size_evenD)
                self._all_size_odd.append(size_odd)
                self._all_size_oddD.append(size_oddD)
                
            else:
                self._all_weights_even.append(np.array([1.0],dtype=float))
                self._all_weights_evenD.append(np.array([1.0],dtype=float))
                self._all_weights_odd.append(np.array([1.0],dtype=float))
                self._all_weights_oddD.append(np.array([1.0],dtype=float))
                self._all_size_even.append(1)
                self._all_size_evenD.append(1)
                self._all_size_odd.append(1)
                self._all_size_oddD.append(1)
        
           
    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    def tosparse(self):
        pass

    def toarray(self):
        pass
    
    def _dot_helper(self, v, out, p, weights_even, weights_odd, size_even, size_odd, h=None):
        """Helper function to perform dot product computation."""
        parity_match = []
        for i in range(3):
            parity_match.append(p[i] % 2)
            
        if h is None:  # Scalar case (H1, L2)
            for j0 in range(self._out_starts[0], self._out_ends[0] + 1):
                parity_j0 = j0 % 2
                weights0, size0, offset0 = ((weights_even[0], size_even[0], 0) if parity_j0 == parity_match[0] else (weights_odd[0], size_odd[0], 1))
                for j1 in range(self._out_starts[1], self._out_ends[1] + 1):
                    parity_j1 = j1 % 2
                    weights1, size1, offset1 = ((weights_even[1], size_even[1], 0) if parity_j1 == parity_match[1] else (weights_odd[1], size_odd[1], 1))
                    for j2 in range(self._out_starts[2], self._out_ends[2] + 1):
                        parity_j2 = j2 % 2
                        weights2, size2, offset2 = ((weights_even[2], size_even[2], 0) if parity_j2 == parity_match[2] else (weights_odd[2], size_odd[2], 1))
                        for i0 in range(size0):
                            if self._halving_directions[0]:
                                pos0 = ((j0 + p[0] - 2 * i0 - offset0) // 2) % self._VNbasis[0]
                            else:
                                pos0 = j0
                            for i1 in range(size1):
                                if self._halving_directions[1]:
                                    pos1 = ((j1 + p[1] - 2 * i1 - offset1) // 2) % self._VNbasis[1]
                                else:
                                    pos1 = j1
                                for i2 in range(size2):
                                    if self._halving_directions[2]:
                                        pos2 = ((j2 + p[2] - 2 * i2 - offset2) // 2) % self._VNbasis[2]
                                    else:
                                        pos2 = j2
                                    out[j0, j1, j2] += weights0[i0] * weights1[i1] * weights2[i2] * v[pos0, pos1, pos2]
            
        else:  # Vector case (Hcurl, Hdiv, H1H1H1)
            for j0 in range(self._out_starts[h][0], self._out_ends[h][0] + 1):
                parity_j0 = j0 % 2
                weights0, size0, offset0 = ((weights_even[0], size_even[0], 0) if parity_j0 == parity_match[0] else (weights_odd[0], size_odd[0], 1))
                for j1 in range(self._out_starts[h][1], self._out_ends[h][1] + 1):
                    parity_j1 = j1 % 2
                    weights1, size1, offset1 = ((weights_even[1], size_even[1], 0) if parity_j1 == parity_match[1] else (weights_odd[1], size_odd[1], 1))
                    for j2 in range(self._out_starts[h][2], self._out_ends[h][2] + 1):
                        parity_j2 = j2 % 2
                        weights2, size2, offset2 = ((weights_even[2], size_even[2], 0) if parity_j2 == parity_match[2] else (weights_odd[2], size_odd[2], 1))
                        for i0 in range(size0):
                            if self._halving_directions[0]:
                                pos0 = ((j0 + p[0] - 2 * i0 - offset0) // 2) % self._VNbasis[h][0]
                            else:
                                pos0 = j0
                            for i1 in range(size1):
                                if self._halving_directions[1]:
                                    pos1 = ((j1 + p[1] - 2 * i1 - offset1) // 2) % self._VNbasis[h][1]
                                else:
                                    pos1 = j1
                                for i2 in range(size2):
                                    if self._halving_directions[2]:
                                        pos2 = ((j2 + p[2] - 2 * i2 - offset2) // 2) % self._VNbasis[h][2]
                                    else:
                                        pos2 = j2
                                    out[h][j0, j1, j2] += weights0[i0] * weights1[i1] * weights2[i2] * v[h][pos0, pos1, pos2]
                
        return out

    def dot_H1(self, v, out):
        return self._dot_helper(v, out, self._p, self._all_weights_even, self._all_weights_odd, self._all_size_even, self._all_size_odd)

    def dot_L2(self, v, out):
        return self._dot_helper(v, out, self._pD, self._all_weights_evenD, self._all_weights_oddD, self._all_size_evenD, self._all_size_oddD)

    def dot_Hcurl(self, v, out):
        out = self._dot_helper(v, out, [self._pD[0], self._p[1],self._p[2]], [self._all_weights_evenD[0],self._all_weights_even[1],self._all_weights_even[2]], [self._all_weights_oddD[0],self._all_weights_odd[1],self._all_weights_odd[2]], [self._all_size_evenD[0],self._all_size_even[1],self._all_size_even[2]], [self._all_size_oddD[0],self._all_size_odd[1],self._all_size_odd[2]], h=0)
        out = self._dot_helper(v, out, [self._p[0], self._pD[1],self._p[2]], [self._all_weights_even[0],self._all_weights_evenD[1],self._all_weights_even[2]], [self._all_weights_odd[0],self._all_weights_oddD[1],self._all_weights_odd[2]], [self._all_size_even[0],self._all_size_evenD[1],self._all_size_even[2]], [self._all_size_odd[0],self._all_size_oddD[1],self._all_size_odd[2]], h=1)
        return self._dot_helper(v, out, [self._p[0], self._p[1],self._pD[2]], [self._all_weights_even[0],self._all_weights_even[1],self._all_weights_evenD[2]], [self._all_weights_odd[0],self._all_weights_odd[1],self._all_weights_oddD[2]], [self._all_size_even[0],self._all_size_even[1],self._all_size_evenD[2]], [self._all_size_odd[0],self._all_size_odd[1],self._all_size_oddD[2]], h=2)


    def dot_Hdiv(self, v, out):
        out = self._dot_helper(v, out, [self._p[0], self._pD[1],self._pD[2]], [self._all_weights_even[0],self._all_weights_evenD[1],self._all_weights_evenD[2]], [self._all_weights_odd[0],self._all_weights_oddD[1],self._all_weights_oddD[2]], [self._all_size_even[0],self._all_size_evenD[1],self._all_size_evenD[2]], [self._all_size_odd[0],self._all_size_oddD[1],self._all_size_oddD[2]], h=0)
        out = self._dot_helper(v, out, [self._pD[0], self._p[1],self._pD[2]], [self._all_weights_evenD[0],self._all_weights_even[1],self._all_weights_evenD[2]], [self._all_weights_oddD[0],self._all_weights_odd[1],self._all_weights_oddD[2]], [self._all_size_evenD[0],self._all_size_even[1],self._all_size_evenD[2]], [self._all_size_oddD[0],self._all_size_odd[1],self._all_size_oddD[2]], h=1)
        return self._dot_helper(v, out, [self._pD[0], self._pD[1],self._p[2]], [self._all_weights_evenD[0],self._all_weights_evenD[1],self._all_weights_even[2]], [self._all_weights_oddD[0],self._all_weights_oddD[1],self._all_weights_odd[2]], [self._all_size_evenD[0],self._all_size_evenD[1],self._all_size_even[2]], [self._all_size_oddD[0],self._all_size_oddD[1],self._all_size_odd[2]], h=2)


    def dot_H1H1H1(self, v, out):
        out = self._dot_helper(v, out, self._p, self._all_weights_even, self._all_weights_odd, self._all_size_even, self._all_size_odd, h = 0)
        self._dot_helper(v, out, self._p, self._all_weights_even, self._all_weights_odd, self._all_size_even, self._all_size_odd, h = 1)
        return self._dot_helper(v, out, self._p, self._all_weights_even, self._all_weights_odd, self._all_size_even, self._all_size_odd, h = 2)

    
    
    def dot(self, v, out=None):

        assert isinstance(v, Vector) and v.space == self.domain
 
        if out is None:
            out = self.codomain.zeros()   
        else:
            assert isinstance(out, Vector) and out.space == self.codomain
            
            if self._V_name == 'H1' or self._V_name == 'L2':
                for i0 in range(self._out_starts[0], self._out_ends[0]+1):
                    for i1 in range(self._out_starts[1], self._out_ends[1]+1):
                        for i2 in range(self._out_starts[2], self._out_ends[2]+1):
                            out[i0,i1,i2] = 0.0
            else:
                for h in range(3):
                    for i0 in range(self._out_starts[h][0], self._out_ends[h][0]+1):
                        for i1 in range(self._out_starts[h][1], self._out_ends[h][1]+1):
                            for i2 in range(self._out_starts[h][2], self._out_ends[h][2]+1):
                                out[h][i0,i1,i2] = 0.0
                
        dot_methods = {
            "H1": self.dot_H1,
            "L2": self.dot_L2,
            "Hcurl": self.dot_Hcurl,
            "Hdiv": self.dot_Hdiv,
            "H1H1H1": self.dot_H1H1H1,
        }
        
        return dot_methods.get(self._V_name)(v, out)
        
    def transpose(self, *, out = None):
        if out is None:
            out = RestrictionOperator(self._W, self._V)
        else:
            assert isinstance(out, RestrictionOperator)
            assert out.domain is self.codomain
            assert out.codomain is self.domain
            
        return out
        

def Compute_rate_of_smoothing(Nel, plist, spl_kind):
    
    from struphy.feec.utilities import create_equal_random_arrays
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    a1= 0.2
    #domain = HollowCylinder(a1 = a1)
    domain = Cuboid()
    sp_key = '1'
    derham = []
    mass_ops = []
    A = []
    
    epsilon = 0.0002
    derham.append(Derham([Nel[0],Nel[1],Nel[2]], plist, spl_kind, comm=comm, local_projectors=False))
    mass_ops.append(WeightedMassOperators(derham[0], domain))
    A.append(epsilon*derham[0].curl.T @ mass_ops[0].M2 @ derham[0].curl + mass_ops[0].M1)
       
    
    field_star = derham[0].create_field('fh', 'Hcurl')
    field_aprox = derham[0].create_field('fh', 'Hcurl')
    
    #We are gonna use the method of manufacture solutions to determine the behaviour of our error
    #Our exact solution is u_star
    u_stararr, u_star = create_equal_random_arrays(derham[0].Vh_fem[sp_key], seed=45)
    #We compute the rhs
    b = A[0].dot(u_star)
    field_star.vector = u_star
    
    #We store the number of itterations
    N_itter = []
    #We define a set of point to evaluate the exact solution and the aproximated one
    pointsx = np.linspace(0.0,1.0,100)
    pointsy = np.linspace(0.0,1.0,100)
    pointsz = np.array([0.0,0.5])
    dx = pointsx[1] - pointsx[0]  # Spacing between points
    dy = pointsy[1] - pointsy[0]  # Spacing between points
    dz = pointsz[1] - pointsz[0]  # Spacing between points
    X, Y, Z = np.meshgrid(pointsx, pointsy, pointsz, indexing="ij")
    
    
    
    #We define a list where to store the arrays with the values of the errors for each aproximation and vector component
    errorsx = []
    errorsy = []
    errorsz = []
    
    #Fourier Transform coefficients of the error function for each vector component
    fourier_coeffsx = []
    fourier_coeffs_shiftedx = []
    fourier_coeffsy = []
    fourier_coeffs_shiftedy = []
    fourier_coeffsz = []
    fourier_coeffs_shiftedz = []
    Max_iter = 10
    for i in range(Max_iter):
        solver = inverse(A[0],'cg', maxiter = int(i+2))
        u = solver.dot(b)
        field_aprox.vector = u
        errorsx.append(field_star(X,Y,Z)[0]-field_aprox(X,Y,Z)[0])
        errorsy.append(field_star(X,Y,Z)[1]-field_aprox(X,Y,Z)[1])
        errorsz.append(field_star(X,Y,Z)[2]-field_aprox(X,Y,Z)[2])
        N_itter.append(solver._info['niter'])
        
        # Compute Fourier Transform coefficients
        fourier_coeffsx.append(np.fft.fftn(errorsx[-1]))
        fourier_coeffs_shiftedx.append(np.fft.fftshift(fourier_coeffsx[-1]))
        fourier_coeffsy.append(np.fft.fftn(errorsy[-1]))
        fourier_coeffs_shiftedy.append(np.fft.fftshift(fourier_coeffsy[-1]))
        fourier_coeffsz.append(np.fft.fftn(errorsz[-1]))
        fourier_coeffs_shiftedz.append(np.fft.fftshift(fourier_coeffsz[-1]))
    
    # Compute corresponding frequencies
    freqsx = np.fft.fftshift(np.fft.fftfreq(len(pointsx), d=dx))
    freqsy = np.fft.fftshift(np.fft.fftfreq(len(pointsy), d=dy))
    freqsz = np.fft.fftshift(np.fft.fftfreq(len(pointsz), d=dz))
    # Create a 3D meshgrid of frequencies
    #Fx, Fy, Fz = np.meshgrid(freq_x[-1], freq_y[-1], freq_z[-1], indexing="ij")
        
    #Now we compute the magnitude of the high frequencies between two interations
    def get_smoothing_rate(freqsx,freqsy,freqsz, coeff_old, coeff_new, hx,hy,hz):
        smoothing_rate = -1.0
        freq = np.zeros(3,dtype=float)
        for ix in range(len(freqsx)):
            for iy in range(len(freqsy)):
                for iz in range(len(freqsz)):
                    if((abs(freqsx[ix])> 1.0 / (4.0*hx) or abs(freqsy[iy])> 1.0 / (4.0*hy) or abs(freqsz[iz])> 1.0 / (4.0*hz) ) and abs(coeff_new[ix,iy,iz])>10.0**-6):
                        smoothing_rate = max(smoothing_rate,abs(coeff_new[ix,iy,iz]/coeff_old[ix,iy,iz])) 
                        freq[0] = freqsx[ix]
                        freq[1] = freqsx[iy]
                        freq[2] = freqsx[iz]
                
        return smoothing_rate, freq
                
    
    for i in range(Max_iter-1):
    
        smoothing_rate, freq = get_smoothing_rate(freqsx,freqsy,freqsz,fourier_coeffs_shiftedx[i],fourier_coeffs_shiftedx[i+1],1.0/Nel[0],1.0/Nel[1],0.00000001)  
        
        print("#######################")
        print(f'{i =}')
        print(f'{smoothing_rate =}')
        print(f'{freq =}')
        print("#######################")
    
    #for i in range(Max_iter-1):
        #smoothing_rate_1d(freqsx, coeff_old, coeff_new, h)


def Visualized_high_frequency_dampening(Nel, plist, spl_kind):
    
    from struphy.feec.utilities import create_equal_random_arrays
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    a1= 0.2
    #domain = HollowCylinder(a1 = a1)
    domain = Cuboid()
    sp_key = '2'
    sp_id = 'Hdiv'
    derham = []
    mass_ops = []
    A = []
    
    epsilon = 1.0
    derham.append(Derham([Nel[0],Nel[1],Nel[2]], plist, spl_kind, comm=comm, local_projectors=False))
    mass_ops.append(WeightedMassOperators(derham[0], domain))
    #Poisson
    #A.append(derham[0].grad.T @ mass_ops[0].M1 @ derham[0].grad)
    #Hall-ish
    #A.append(epsilon*derham[0].curl.T @ mass_ops[0].M2 @ derham[0].curl + mass_ops[0].M1)
    #Hall
    #A.append(mass_ops[0].M1 -epsilon*derham[0].curl.T @ mass_ops[0].M2 @ derham[0].curl)
    #Shear-Alfven-ish
    #A.append(mass_ops[0].M2 -epsilon* mass_ops[0].M2@ derham[0].curl @ mass_ops[0].M1 @ derham[0].curl.T @ mass_ops[0].M2)
    #Shear-Alfven
    pc_class = getattr(preconditioner,"MassMatrixPreconditioner")
    pc = pc_class(mass_ops[0].M1)
    M1_inv = inverse(
        mass_ops[0].M1,
        "pcg",
        pc=pc,
        maxiter=3000,
        verbose=False,
    )
    A.append(mass_ops[0].M2 -epsilon* mass_ops[0].M2@ derham[0].curl @ M1_inv @ derham[0].curl.T @ mass_ops[0].M2)
    #Shear-Alfven-v2
    #A.append(mass_ops[0].M2 -epsilon* derham[0].curl @ M1_inv @ derham[0].curl.T)
    
    
       
    
    field_star = derham[0].create_field('fh', sp_id)
    field_aprox = derham[0].create_field('fh', sp_id)
    
    #We are gonna use the method of manufacture solutions to determine the behaviour of our error
    #Our exact solution is u_star
    u_stararr, u_star = create_equal_random_arrays(derham[0].Vh_fem[sp_key], seed=45)
    #We compute the rhs
    b = A[0].dot(u_star)
    field_star.vector = u_star
    
    #Turn b into an array
    #barr = remove_padding(derham[0].Vh_fem[sp_key],b)
    #Turn A[0] into an array
    #Aarr = A[0].toarray()
    
    #Gauss = will_gauss_seidel_converge(Aarr, verbose=True)
    #print(f"{Gauss = }")
    
    
    #We store the number of itterations
    N_itter = []
    #We define a set of point to evaluate the exact solution and the aproximated one
    pointsx = np.linspace(0.0,1.0,Nel[0])
    #pointsx = np.array([0.0,0.5])
    #pointsy = np.linspace(0.0,1.0,Nel[1])
    pointsy = np.array([0.0,0.5])
    pointsz = np.array([0.0,0.5])
    dx = pointsx[1] - pointsx[0]  # Spacing between points
    dy = pointsy[1] - pointsy[0]  # Spacing between points
    dz = pointsz[1] - pointsz[0]  # Spacing between points
    X, Y, Z = np.meshgrid(pointsx, pointsy, pointsz, indexing="ij")
    
    
    
    #We define a list where to store the arrays with the values of the errors for each aproximation and vector component
    errorsx = []
    errorsy = []
    errorsz = []
    
    #Fourier Transform coefficients of the error function for each vector component
    fourier_coeffsx = []
    fourier_coeffs_shiftedx = []
    fourier_coeffsy = []
    fourier_coeffs_shiftedy = []
    fourier_coeffsz = []
    fourier_coeffs_shiftedz = []
    max_iter_list = [3]
    #max_iter_list = [0,2,3,4,5,6,7,8,9,10]
    Number_of_iter = len(max_iter_list)
    for i in range(Number_of_iter):
        
        if(max_iter_list[i]>1):
        
            solver = inverse(A[0],'cg', maxiter = max_iter_list[i], tol = 10.0**-6)
            u = solver.dot(b)
            #uarr, itter = jacobi(Aarr,barr,max_iter=max_iter_list[i])
            #u = from_array_to_psydac(uarr, derham[0].Vh_fem[sp_key])
            N_itter.append(solver._info['niter'])
            #N_itter.append(itter)
            
        else:
            u = derham[0].Vh[derham[0].space_to_form[sp_id]].zeros()
            N_itter.append(0)
        
        
        field_aprox.vector = u
        #errorsx.append(field_star(X,Y,Z)-field_aprox(X,Y,Z))
        errorsx.append(field_star(X,Y,Z)[0]-field_aprox(X,Y,Z)[0])
        errorsy.append(field_star(X,Y,Z)[1]-field_aprox(X,Y,Z)[1])
        errorsz.append(field_star(X,Y,Z)[2]-field_aprox(X,Y,Z)[2])
        
        
        # Compute Fourier Transform coefficients
        fourier_coeffsx.append(np.fft.fftn(errorsx[-1]))
        fourier_coeffs_shiftedx.append(np.fft.fftshift(fourier_coeffsx[-1]))
        fourier_coeffsy.append(np.fft.fftn(errorsy[-1]))
        fourier_coeffs_shiftedy.append(np.fft.fftshift(fourier_coeffsy[-1]))
        fourier_coeffsz.append(np.fft.fftn(errorsz[-1]))
        fourier_coeffs_shiftedz.append(np.fft.fftshift(fourier_coeffsz[-1]))
    
    # Compute corresponding frequencies
    freqsx = np.fft.fftshift(np.fft.fftfreq(len(pointsx), d=dx))
    freqsy = np.fft.fftshift(np.fft.fftfreq(len(pointsy), d=dy))
    freqsz = np.fft.fftshift(np.fft.fftfreq(len(pointsz), d=dz))
    # Create a 3D meshgrid of frequencies
    #Fx, Fy, Fz = np.meshgrid(freq_x[-1], freq_y[-1], freq_z[-1], indexing="ij")
        
    #Now we compute the magnitude of the high frequencies between two interations
    def get_magnitude_maximum_nasty_frequency_scalar(freqsx,freqsy,freqsz, coeff_new,hx,hy,hz):
        value= 0.0
        freq = np.zeros(3,dtype=float)
        for ix in range(len(freqsx)):
            for iy in range(len(freqsy)):
                for iz in range(len(freqsz)):
                    if((abs(freqsx[ix])> 1.0/(4.0*hx) or abs(freqsy[iy])> 1.0/(4.0*hy) or abs(freqsz[iz])> 1.0/(4.0*hz) ) and abs(coeff_new[ix,iy,iz])>10.0**-6):
                        if(abs(coeff_new[ix,iy,iz]) > value ):
                            value = abs(coeff_new[ix,iy,iz])
                            freq[0] = freqsx[ix]
                            freq[1] = freqsy[iy]
                            freq[2] = freqsz[iz]
                                
        return value, freq
    
    
    
    
    def get_magnitude_maximum_nasty_frequency(freqsx,freqsy,freqsz, coeff_newx, coeff_newy, coeff_newz,hx,hy,hz):
        value= 0.0
        freq = np.zeros(3,dtype=float)
        for ix in range(len(freqsx)):
            for iy in range(len(freqsy)):
                for iz in range(len(freqsz)):
                    if((abs(freqsx[ix])> 1.0/(4.0*hx) or abs(freqsy[iy])> 1.0/(4.0*hy) or abs(freqsz[iz])> 1.0/(4.0*hz) ) and (abs(coeff_newx[ix,iy,iz])>10.0**-6  or abs(coeff_newy[ix,iy,iz])>10.0**-6 or abs(coeff_newz[ix,iy,iz])>10.0**-6)):
                        if(abs(coeff_newx[ix,iy,iz]) > value and abs(coeff_newx[ix,iy,iz]) >= abs(coeff_newy[ix,iy,iz]) and abs(coeff_newx[ix,iy,iz])>= abs(coeff_newz[ix,iy,iz])):
                            value = abs(coeff_newx[ix,iy,iz])
                            freq[0] = freqsx[ix]
                            freq[1] = freqsy[iy]
                            freq[2] = freqsz[iz]
                            
                        elif(abs(coeff_newy[ix,iy,iz]) > value and abs(coeff_newy[ix,iy,iz]) >= abs(coeff_newx[ix,iy,iz]) and abs(coeff_newy[ix,iy,iz])>= abs(coeff_newz[ix,iy,iz])):
                            value = abs(coeff_newy[ix,iy,iz])
                            freq[0] = freqsx[ix]
                            freq[1] = freqsy[iy]
                            freq[2] = freqsz[iz]
                            
                        elif(abs(coeff_newz[ix,iy,iz]) > value and abs(coeff_newz[ix,iy,iz]) >= abs(coeff_newx[ix,iy,iz]) and abs(coeff_newz[ix,iy,iz])>= abs(coeff_newy[ix,iy,iz])):
                            value = abs(coeff_newz[ix,iy,iz])
                            freq[0] = freqsx[ix]
                            freq[1] = freqsy[iy]
                            freq[2] = freqsz[iz]
                
        return value, freq
                
    
    magnitudes = []
    bad_frequencies = []
    
    for i in range(Number_of_iter):
        #magnitude, freq = get_magnitude_maximum_nasty_frequency_scalar(freqsx,freqsy,freqsz,fourier_coeffs_shiftedx[i],1.0/Nel[0],1.0/Nel[1],0.000001)  
        magnitude, freq = get_magnitude_maximum_nasty_frequency(freqsx,freqsy,freqsz,fourier_coeffs_shiftedx[i],fourier_coeffs_shiftedy[i], fourier_coeffs_shiftedz[i],1.0/Nel[0],0.000001,0.000001)  
        magnitudes.append(magnitude)
        bad_frequencies.append(freq)
    
    
    print(f'{Nel[0] = }')
    print(f'{Nel[1] = }')
    print("magnitudes")
    for i in magnitudes:
        print(i)
    print("N_itter")
    for i in N_itter:
        print(i)
    print("bad_frequencies")
    for i in bad_frequencies:
        print(i[0:2])

    
    
    plt.figure()
    plt.scatter(N_itter,magnitudes)
    #plt.yscale("log")
    plt.show()


def Visualized_all_frequencies_dampening(Nel, plist, spl_kind, Is):
    
    from struphy.feec.utilities import create_equal_random_arrays
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    a1= 0.2
    #domain = HollowCylinder(a1 = a1)
    domain = Cuboid()
    model = 'Shear-Alfven'
    smoother = 'cg'
    derham = []
    mass_ops = []
    A = []
    
    epsilon = 10.0**-6.0
    derham.append(Derham([Nel[0],Nel[1],Nel[2]], plist, spl_kind, comm=comm, local_projectors=False))
    mass_ops.append(WeightedMassOperators(derham[0], domain))
    if(model == "Poisson"):
        sp_key = '0'
        sp_id = 'H1'
        #Poisson
        A.append(derham[0].grad.T @ mass_ops[0].M1 @ derham[0].grad)
    #Hall-ish
    #A.append(epsilon*derham[0].curl.T @ mass_ops[0].M2 @ derham[0].curl + mass_ops[0].M1)
    elif(model == "Hall"):
        sp_key = '1'
        sp_id = 'Hcurl'
        #Hall
        A.append(mass_ops[0].M1 -epsilon*derham[0].curl.T @ mass_ops[0].M2 @ derham[0].curl)
    #Shear-Alfven-ish
    #A.append(mass_ops[0].M2 -epsilon* mass_ops[0].M2@ derham[0].curl @ mass_ops[0].M1 @ derham[0].curl.T @ mass_ops[0].M2)
    elif(model == 'Shear-Alfven' or model == 'Shear-Alfven-v2'):
        sp_key = '2'
        sp_id = 'Hdiv'
        pc_class = getattr(preconditioner,"MassMatrixPreconditioner")
        pc = pc_class(mass_ops[0].M1)
        M1_inv = inverse(
            mass_ops[0].M1,
            "pcg",
            pc=pc,
            maxiter=3000,
            verbose=False,
        )
        #Shear-Alfven
        if (model == "Shear-Alfven"):
            A.append(mass_ops[0].M2 -epsilon* mass_ops[0].M2@ derham[0].curl @ M1_inv @ derham[0].curl.T @ mass_ops[0].M2)
        else:
            #Shear-Alfven-v2
            A.append(mass_ops[0].M2 -epsilon* derham[0].curl @ M1_inv @ derham[0].curl.T)
    
    field_star = derham[0].create_field('fh', sp_id)
    field_aprox = derham[0].create_field('fh', sp_id)
    
    #We are gonna use the method of manufacture solutions to determine the behaviour of our error
    #Our exact solution is u_star
    u_stararr, u_star = create_equal_random_arrays(derham[0].Vh_fem[sp_key], seed=45)
    #We compute the rhs
    b = A[0].dot(u_star)
    field_star.vector = u_star
    
    #Turn b into an array
    #barr = remove_padding(derham[0].Vh_fem[sp_key],b)
    #Turn A[0] into an array
    #Aarr = A[0].toarray()
    
    #Gauss = will_gauss_seidel_converge(Aarr, verbose=True)
    #print(f"{Gauss = }")
    
    
    #We store the number of itterations
    N_itter = []
    #We define a set of point to evaluate the exact solution and the aproximated one
    pointsx = np.linspace(0.0,1.0,Nel[0])
    #pointsx = np.array([0.0,0.5])
    pointsy = np.linspace(0.0,1.0,Nel[1])
    #pointsy = np.array([0.0,0.5])
    pointsz = np.array([0.0,0.5])
    dx = pointsx[1] - pointsx[0]  # Spacing between points
    dy = pointsy[1] - pointsy[0]  # Spacing between points
    dz = pointsz[1] - pointsz[0]  # Spacing between points
    X, Y, Z = np.meshgrid(pointsx, pointsy, pointsz, indexing="ij")
    
    
    
    #We define a list where to store the arrays with the values of the errors for each aproximation and vector component
    errorsx = []
    errorsy = []
    errorsz = []
    
    #Fourier Transform coefficients of the error function for each vector component
    fourier_coeffsx = []
    fourier_coeffs_shiftedx = []
    fourier_coeffsy = []
    fourier_coeffs_shiftedy = []
    fourier_coeffsz = []
    fourier_coeffs_shiftedz = []
    max_iter_list = [0,Is, int(10*Is)]
    #max_iter_list = [0,2,3,4,5,6,7,8,9,10]
    Number_of_iter = len(max_iter_list)
    for i in range(Number_of_iter):
        
        if(max_iter_list[i]>1):
        
            solver = inverse(A[0],smoother, maxiter = max_iter_list[i], tol = 10.0**-6)
            u = solver.dot(b)
            #uarr, itter = jacobi(Aarr,barr,max_iter=max_iter_list[i])
            #u = from_array_to_psydac(uarr, derham[0].Vh_fem[sp_key])
            N_itter.append(solver._info['niter'])
            #N_itter.append(itter)
            
        else:
            u = derham[0].Vh[derham[0].space_to_form[sp_id]].zeros()
            N_itter.append(0)
        
        
        field_aprox.vector = u
        if(model == "Poisson"):
            errorsx.append(field_star(X,Y,Z)-field_aprox(X,Y,Z))
        else:
            errorsx.append(field_star(X,Y,Z)[0]-field_aprox(X,Y,Z)[0])
            errorsy.append(field_star(X,Y,Z)[1]-field_aprox(X,Y,Z)[1])
            errorsz.append(field_star(X,Y,Z)[2]-field_aprox(X,Y,Z)[2])
        
        # Compute Fourier Transform coefficients
        fourier_coeffsx.append(np.fft.fftn(errorsx[-1]))
        fourier_coeffs_shiftedx.append(np.fft.fftshift(fourier_coeffsx[-1]))
        if(model != "Poisson"):
            fourier_coeffsy.append(np.fft.fftn(errorsy[-1]))
            fourier_coeffs_shiftedy.append(np.fft.fftshift(fourier_coeffsy[-1]))
            fourier_coeffsz.append(np.fft.fftn(errorsz[-1]))
            fourier_coeffs_shiftedz.append(np.fft.fftshift(fourier_coeffsz[-1]))
    
    # Compute corresponding frequencies
    freqsx = np.fft.fftshift(np.fft.fftfreq(len(pointsx), d=dx))
    freqsy = np.fft.fftshift(np.fft.fftfreq(len(pointsy), d=dy))
    freqsz = np.fft.fftshift(np.fft.fftfreq(len(pointsz), d=dz))
    # Create a 3D meshgrid of frequencies
    #Fx, Fy, Fz = np.meshgrid(freq_x[-1], freq_y[-1], freq_z[-1], indexing="ij")
    
    #i also want to visiualize the maximum freqeuncies the next 3 coarser grids can handle
    hx = 1/Nel[0]
    next_max_frequenciesx = [1.0/(4.0*hx),1.0/(8.0*hx),1.0/(16.0*hx)]
    hy = 1/Nel[1]
    next_max_frequenciesy = [1.0/(4.0*hy),1.0/(8.0*hy),1.0/(16.0*hy)]
    #I get the amplitude of the maximum initial error wave
    max_amplitude = np.max(np.abs(fourier_coeffs_shiftedx[0][:,1,1]))
    
    if(model == "Poisson"):
        plt.figure(figsize=(10,10))
        plt.title("Amplitude of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",size = 10)
        plt.scatter(freqsx,np.abs(fourier_coeffs_shiftedx[0][:,1,1]), label = "Initial")
        plt.scatter(freqsx,np.abs(fourier_coeffs_shiftedx[1][:,1,1]), label = "After "+str(max_iter_list[1])+" iterations ")
        plt.scatter(freqsx,np.abs(fourier_coeffs_shiftedx[2][:,1,1]), label = "After "+str(max_iter_list[2])+" iterations ")
        plt.plot(next_max_frequenciesx[0]*np.ones(100),np.linspace(0,max_amplitude,100),linestyle = 'dashed', color ='blue',label = "Max freuency level - 1.")
        plt.plot(-next_max_frequenciesx[0]*np.ones(100),np.linspace(0,max_amplitude,100),linestyle = 'dashed', color ='blue')
        plt.plot(next_max_frequenciesx[1]*np.ones(100),np.linspace(0,max_amplitude,100),linestyle = 'dashed', color ='purple',label = "Max freuency level - 2.")
        plt.plot(-next_max_frequenciesx[1]*np.ones(100),np.linspace(0,max_amplitude,100),linestyle = 'dashed', color ='purple')
        plt.plot(next_max_frequenciesx[2]*np.ones(100),np.linspace(0,max_amplitude,100),linestyle = 'dashed', color ='red',label = "Max freuency level - 3.")
        plt.plot(-next_max_frequenciesx[2]*np.ones(100),np.linspace(0,max_amplitude,100),linestyle = 'dashed', color ='red')
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.legend()
        #plt.ylim(0,30)
        plt.yscale("log")
        plt.savefig("./Wave-Amplitude-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+".pdf")
        plt.close()
        
    
        #We also want to see the relative difference between the initial amplitude and the amplitude after smoothing.
        #We compute abs(A_0)-abs(A_i)/abs(A_0), this tell us which percentage of the initial amplitude has been eliminated
        #A values of 0 means it all still remains, a value of 1 means we eliminated all the amplitude. And a negative value 
        #means that the amplitude increased.
        eliminated_portion = []
        for i in range(2):
            aux = (np.abs(fourier_coeffs_shiftedx[0][:,1,1]) - np.abs(fourier_coeffs_shiftedx[i+1][:,1,1]))/np.abs(fourier_coeffs_shiftedx[0][:,1,1])
            eliminated_portion.append(aux)
        
        mp = np.min(eliminated_portion[0])
        aux = np.min(eliminated_portion[1])
        mp = min(mp,aux)
        
        plt.figure(figsize=(10,10))
        plt.title("Eliminated portion of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",size = 10)
        plt.scatter(freqsx,eliminated_portion[0], label = "Portion of amplitude eliminated after " + str(max_iter_list[1]))
        plt.scatter(freqsx,eliminated_portion[1], label = "Portion of amplitude eliminated after " + str(max_iter_list[2]))
        plt.plot(next_max_frequenciesx[0]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='blue',label = "Max freuency level - 1.")
        plt.plot(-next_max_frequenciesx[0]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='blue')
        plt.plot(next_max_frequenciesx[1]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='purple',label = "Max freuency level - 2.")
        plt.plot(-next_max_frequenciesx[1]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='purple')
        plt.plot(next_max_frequenciesx[2]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='red',label = "Max freuency level - 3.")
        plt.plot(-next_max_frequenciesx[2]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='red')
        plt.xlabel("Frequency")
        plt.ylabel("Eliminated portion of the amplitude.")
        plt.legend()
        #plt.ylim(0,30)
        #plt.yscale("log")
        plt.savefig("./Eliminated-Portion-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+".pdf")
        plt.close()
        
    else:
        #I get the amplitude of the maximum initial error wave
        max_amplitudey = np.max(np.abs(fourier_coeffs_shiftedy[0][:,1,1]))
        #I get the amplitude of the maximum initial error wave
        max_amplitudez = np.max(np.abs(fourier_coeffs_shiftedz[0][:,1,1]))
        
        plt.figure(figsize=(10,10))
        plt.title("X-component. Epsilon 10^-6 Amplitude of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",size = 10)
        plt.scatter(freqsx,np.abs(fourier_coeffs_shiftedx[0][:,1,1]), label = "Initial")
        plt.scatter(freqsx,np.abs(fourier_coeffs_shiftedx[1][:,1,1]), label = "After "+str(max_iter_list[1])+" iterations ")
        plt.scatter(freqsx,np.abs(fourier_coeffs_shiftedx[2][:,1,1]), label = "After "+str(max_iter_list[2])+" iterations ")
        plt.plot(next_max_frequenciesx[0]*np.ones(100),np.linspace(0,max_amplitude,100),linestyle = 'dashed', color ='blue',label = "Max freuency level - 1.")
        plt.plot(-next_max_frequenciesx[0]*np.ones(100),np.linspace(0,max_amplitude,100),linestyle = 'dashed', color ='blue')
        plt.plot(next_max_frequenciesx[1]*np.ones(100),np.linspace(0,max_amplitude,100),linestyle = 'dashed', color ='purple',label = "Max freuency level - 2.")
        plt.plot(-next_max_frequenciesx[1]*np.ones(100),np.linspace(0,max_amplitude,100),linestyle = 'dashed', color ='purple')
        plt.plot(next_max_frequenciesx[2]*np.ones(100),np.linspace(0,max_amplitude,100),linestyle = 'dashed', color ='red',label = "Max freuency level - 3.")
        plt.plot(-next_max_frequenciesx[2]*np.ones(100),np.linspace(0,max_amplitude,100),linestyle = 'dashed', color ='red')
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.legend()
        #plt.ylim(0,30)
        plt.yscale("log")
        plt.savefig("./Wave-Amplitude-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+"X-component-Epsilon-10-6.pdf")
        plt.close()
        
        plt.figure(figsize=(10,10))
        plt.title("Y-component. Epsilon 10^-6 Amplitude of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",size = 10)
        plt.scatter(freqsx,np.abs(fourier_coeffs_shiftedy[0][:,1,1]), label = "Initial")
        plt.scatter(freqsx,np.abs(fourier_coeffs_shiftedy[1][:,1,1]), label = "After "+str(max_iter_list[1])+" iterations ")
        plt.scatter(freqsx,np.abs(fourier_coeffs_shiftedy[2][:,1,1]), label = "After "+str(max_iter_list[2])+" iterations ")
        plt.plot(next_max_frequenciesx[0]*np.ones(100),np.linspace(0,max_amplitudey,100),linestyle = 'dashed', color ='blue',label = "Max freuency level - 1.")
        plt.plot(-next_max_frequenciesx[0]*np.ones(100),np.linspace(0,max_amplitudey,100),linestyle = 'dashed', color ='blue')
        plt.plot(next_max_frequenciesx[1]*np.ones(100),np.linspace(0,max_amplitudey,100),linestyle = 'dashed', color ='purple',label = "Max freuency level - 2.")
        plt.plot(-next_max_frequenciesx[1]*np.ones(100),np.linspace(0,max_amplitudey,100),linestyle = 'dashed', color ='purple')
        plt.plot(next_max_frequenciesx[2]*np.ones(100),np.linspace(0,max_amplitudey,100),linestyle = 'dashed', color ='red',label = "Max freuency level - 3.")
        plt.plot(-next_max_frequenciesx[2]*np.ones(100),np.linspace(0,max_amplitudey,100),linestyle = 'dashed', color ='red')
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.legend()
        #plt.ylim(0,30)
        plt.yscale("log")
        plt.savefig("./Wave-Amplitude-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+"Y-component-Epsilon-10-6.pdf")
        plt.close()
        
        
        plt.figure(figsize=(10,10))
        plt.title("Z-component. Epsilon 10^-6 Amplitude of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",size = 10)
        plt.scatter(freqsx,np.abs(fourier_coeffs_shiftedz[0][:,1,1]), label = "Initial")
        plt.scatter(freqsx,np.abs(fourier_coeffs_shiftedz[1][:,1,1]), label = "After "+str(max_iter_list[1])+" iterations ")
        plt.scatter(freqsx,np.abs(fourier_coeffs_shiftedz[2][:,1,1]), label = "After "+str(max_iter_list[2])+" iterations ")
        plt.plot(next_max_frequenciesx[0]*np.ones(100),np.linspace(0,max_amplitudez,100),linestyle = 'dashed', color ='blue',label = "Max freuency level - 1.")
        plt.plot(-next_max_frequenciesx[0]*np.ones(100),np.linspace(0,max_amplitudez,100),linestyle = 'dashed', color ='blue')
        plt.plot(next_max_frequenciesx[1]*np.ones(100),np.linspace(0,max_amplitudez,100),linestyle = 'dashed', color ='purple',label = "Max freuency level - 2.")
        plt.plot(-next_max_frequenciesx[1]*np.ones(100),np.linspace(0,max_amplitudez,100),linestyle = 'dashed', color ='purple')
        plt.plot(next_max_frequenciesx[2]*np.ones(100),np.linspace(0,max_amplitudez,100),linestyle = 'dashed', color ='red',label = "Max freuency level - 3.")
        plt.plot(-next_max_frequenciesx[2]*np.ones(100),np.linspace(0,max_amplitudez,100),linestyle = 'dashed', color ='red')
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.legend()
        #plt.ylim(0,30)
        plt.yscale("log")
        plt.savefig("./Wave-Amplitude-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+"Z-component-Epsilon-10-6.pdf")
        plt.close()
        
    
        #We also want to see the relative difference between the initial amplitude and the amplitude after smoothing.
        #We compute abs(A_0)-abs(A_i)/abs(A_0), this tell us which percentage of the initial amplitude has been eliminated
        #A values of 0 means it all still remains, a value of 1 means we eliminated all the amplitude. And a negative value 
        #means that the amplitude increased.
        eliminated_portionx = []
        for i in range(2):
            aux = (np.abs(fourier_coeffs_shiftedx[0][:,1,1]) - np.abs(fourier_coeffs_shiftedx[i+1][:,1,1]))/np.abs(fourier_coeffs_shiftedx[0][:,1,1])
            eliminated_portionx.append(aux)
        
        mp = np.min(eliminated_portionx[0])
        aux = np.min(eliminated_portionx[1])
        mp = min(mp,aux)
        
        plt.figure(figsize=(10,10))
        plt.title("X-component. Epsilon 10^-6 Eliminated portion of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",size = 10)
        plt.scatter(freqsx,eliminated_portionx[0], label = "Portion of amplitude eliminated after " + str(max_iter_list[1]))
        plt.scatter(freqsx,eliminated_portionx[1], label = "Portion of amplitude eliminated after " + str(max_iter_list[2]))
        plt.plot(next_max_frequenciesx[0]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='blue',label = "Max freuency level - 1.")
        plt.plot(-next_max_frequenciesx[0]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='blue')
        plt.plot(next_max_frequenciesx[1]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='purple',label = "Max freuency level - 2.")
        plt.plot(-next_max_frequenciesx[1]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='purple')
        plt.plot(next_max_frequenciesx[2]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='red',label = "Max freuency level - 3.")
        plt.plot(-next_max_frequenciesx[2]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='red')
        plt.xlabel("Frequency")
        plt.ylabel("Eliminated portion of the amplitude.")
        plt.legend()
        #plt.ylim(0,30)
        #plt.yscale("log")
        plt.savefig("./Eliminated-Portion-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+"X-component-Epsilon-10-6.pdf")
        plt.close()
        
        
        eliminated_portiony = []
        for i in range(2):
            aux = (np.abs(fourier_coeffs_shiftedy[0][:,1,1]) - np.abs(fourier_coeffs_shiftedy[i+1][:,1,1]))/np.abs(fourier_coeffs_shiftedy[0][:,1,1])
            eliminated_portiony.append(aux)
        
        mp = np.min(eliminated_portiony[0])
        aux = np.min(eliminated_portiony[1])
        mp = min(mp,aux)
        
        plt.figure(figsize=(10,10))
        plt.title("Y-component. Epsilon 10^-6 Eliminated portion of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",size = 10)
        plt.scatter(freqsx,eliminated_portiony[0], label = "Portion of amplitude eliminated after " + str(max_iter_list[1]))
        plt.scatter(freqsx,eliminated_portiony[1], label = "Portion of amplitude eliminated after " + str(max_iter_list[2]))
        plt.plot(next_max_frequenciesx[0]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='blue',label = "Max freuency level - 1.")
        plt.plot(-next_max_frequenciesx[0]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='blue')
        plt.plot(next_max_frequenciesx[1]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='purple',label = "Max freuency level - 2.")
        plt.plot(-next_max_frequenciesx[1]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='purple')
        plt.plot(next_max_frequenciesx[2]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='red',label = "Max freuency level - 3.")
        plt.plot(-next_max_frequenciesx[2]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='red')
        plt.xlabel("Frequency")
        plt.ylabel("Eliminated portion of the amplitude.")
        plt.legend()
        #plt.ylim(0,30)
        #plt.yscale("log")
        plt.savefig("./Eliminated-Portion-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+"Y-component-Epsilon-10-6.pdf")
        plt.close()
        
        
        eliminated_portionz = []
        for i in range(2):
            aux = (np.abs(fourier_coeffs_shiftedz[0][:,1,1]) - np.abs(fourier_coeffs_shiftedz[i+1][:,1,1]))/np.abs(fourier_coeffs_shiftedz[0][:,1,1])
            eliminated_portionz.append(aux)
        
        mp = np.min(eliminated_portionz[0])
        aux = np.min(eliminated_portionz[1])
        mp = min(mp,aux)
        
        plt.figure(figsize=(10,10))
        plt.title("Z-component. Epsilon 10^-6 Eliminated portion of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",size = 10)
        plt.scatter(freqsx,eliminated_portionz[0], label = "Portion of amplitude eliminated after " + str(max_iter_list[1]))
        plt.scatter(freqsx,eliminated_portionz[1], label = "Portion of amplitude eliminated after " + str(max_iter_list[2]))
        plt.plot(next_max_frequenciesx[0]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='blue',label = "Max freuency level - 1.")
        plt.plot(-next_max_frequenciesx[0]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='blue')
        plt.plot(next_max_frequenciesx[1]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='purple',label = "Max freuency level - 2.")
        plt.plot(-next_max_frequenciesx[1]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='purple')
        plt.plot(next_max_frequenciesx[2]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='red',label = "Max freuency level - 3.")
        plt.plot(-next_max_frequenciesx[2]*np.ones(100),np.linspace(mp,1,100),linestyle = 'dashed', color ='red')
        plt.xlabel("Frequency")
        plt.ylabel("Eliminated portion of the amplitude.")
        plt.legend()
        #plt.ylim(0,30)
        #plt.yscale("log")
        plt.savefig("./Eliminated-Portion-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+"Z-component-Epsilon-10-6.pdf")
        plt.close()
    

def Visualized_all_frequencies_dampening_2D(Nel, plist, spl_kind, Is):
    
    from struphy.feec.utilities import create_equal_random_arrays
    import plotly.graph_objects as go

    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    a1= 0.2
    #domain = HollowCylinder(a1 = a1)
    domain = Cuboid()
    model = 'Shear-Alfven'
    smoother = 'gmres'
    derham = []
    mass_ops = []
    A = []
    
    epsilon = 1.0
    derham.append(Derham([Nel[0],Nel[1],Nel[2]], plist, spl_kind, comm=comm, local_projectors=False))
    mass_ops.append(WeightedMassOperators(derham[0], domain))
    if(model == "Poisson"):
        sp_key = '0'
        sp_id = 'H1'
        #Poisson
        A.append(derham[0].grad.T @ mass_ops[0].M1 @ derham[0].grad)
    #Hall-ish
    #A.append(epsilon*derham[0].curl.T @ mass_ops[0].M2 @ derham[0].curl + mass_ops[0].M1)
    elif(model == "Hall"):
        sp_key = '1'
        sp_id = 'Hcurl'
        #Hall
        A.append(mass_ops[0].M1 -epsilon*derham[0].curl.T @ mass_ops[0].M2 @ derham[0].curl)
    #Shear-Alfven-ish
    #A.append(mass_ops[0].M2 -epsilon* mass_ops[0].M2@ derham[0].curl @ mass_ops[0].M1 @ derham[0].curl.T @ mass_ops[0].M2)
    elif(model == 'Shear-Alfven' or model == 'Shear-Alfven-v2'):
        sp_key = '2'
        sp_id = 'Hdiv'
        pc_class = getattr(preconditioner,"MassMatrixPreconditioner")
        pc = pc_class(mass_ops[0].M1)
        M1_inv = inverse(
            mass_ops[0].M1,
            "pcg",
            pc=pc,
            maxiter=3000,
            verbose=False,
        )
        #Shear-Alfven
        if (model == "Shear-Alfven"):
            A.append(mass_ops[0].M2 -epsilon* mass_ops[0].M2@ derham[0].curl @ M1_inv @ derham[0].curl.T @ mass_ops[0].M2)
        else:
            #Shear-Alfven-v2
            A.append(mass_ops[0].M2 -epsilon* derham[0].curl @ M1_inv @ derham[0].curl.T)
    
    field_star = derham[0].create_field('fh', sp_id)
    field_aprox = derham[0].create_field('fh', sp_id)
    
    #We are gonna use the method of manufacture solutions to determine the behaviour of our error
    #Our exact solution is u_star
    u_stararr, u_star = create_equal_random_arrays(derham[0].Vh_fem[sp_key], seed=45)
    #We compute the rhs
    b = A[0].dot(u_star)
    field_star.vector = u_star
    
    #Turn b into an array
    #barr = remove_padding(derham[0].Vh_fem[sp_key],b)
    #Turn A[0] into an array
    #Aarr = A[0].toarray()
    
    #Gauss = will_gauss_seidel_converge(Aarr, verbose=True)
    #print(f"{Gauss = }")
    
    
    #We store the number of itterations
    N_itter = []
    #We define a set of point to evaluate the exact solution and the aproximated one
    pointsx = np.linspace(0.0,1.0,Nel[0])
    #pointsx = np.array([0.0,0.5])
    pointsy = np.linspace(0.0,1.0,Nel[1])
    #pointsy = np.array([0.0,0.5])
    pointsz = np.array([0.0,0.5])
    dx = pointsx[1] - pointsx[0]  # Spacing between points
    dy = pointsy[1] - pointsy[0]  # Spacing between points
    dz = pointsz[1] - pointsz[0]  # Spacing between points
    X, Y, Z = np.meshgrid(pointsx, pointsy, pointsz, indexing="ij")
    
    
    
    #We define a list where to store the arrays with the values of the errors for each aproximation and vector component
    errorsx = []
    errorsy = []
    errorsz = []
    
    #Fourier Transform coefficients of the error function for each vector component
    fourier_coeffsx = []
    fourier_coeffs_shiftedx = []
    fourier_coeffsy = []
    fourier_coeffs_shiftedy = []
    fourier_coeffsz = []
    fourier_coeffs_shiftedz = []
    max_iter_list = [0,Is, int(10*Is)]
    #max_iter_list = [0,2,3,4,5,6,7,8,9,10]
    Number_of_iter = len(max_iter_list)
    for i in range(Number_of_iter):
        
        if(max_iter_list[i]>1):
        
            solver = inverse(A[0],smoother, maxiter = max_iter_list[i], tol = 10.0**-6)
            u = solver.dot(b)
            #uarr, itter = jacobi(Aarr,barr,max_iter=max_iter_list[i])
            #u = from_array_to_psydac(uarr, derham[0].Vh_fem[sp_key])
            N_itter.append(solver._info['niter'])
            #N_itter.append(itter)
            
        else:
            u = derham[0].Vh[derham[0].space_to_form[sp_id]].zeros()
            N_itter.append(0)
        
        
        field_aprox.vector = u
        if(model == "Poisson"):
            errorsx.append(field_star(X,Y,Z)-field_aprox(X,Y,Z))
        else:
            errorsx.append(field_star(X,Y,Z)[0]-field_aprox(X,Y,Z)[0])
            errorsy.append(field_star(X,Y,Z)[1]-field_aprox(X,Y,Z)[1])
            errorsz.append(field_star(X,Y,Z)[2]-field_aprox(X,Y,Z)[2])
        
        # Compute Fourier Transform coefficients
        fourier_coeffsx.append(np.fft.fftn(errorsx[-1]))
        fourier_coeffs_shiftedx.append(np.fft.fftshift(fourier_coeffsx[-1]))
        if(model != "Poisson"):
            fourier_coeffsy.append(np.fft.fftn(errorsy[-1]))
            fourier_coeffs_shiftedy.append(np.fft.fftshift(fourier_coeffsy[-1]))
            fourier_coeffsz.append(np.fft.fftn(errorsz[-1]))
            fourier_coeffs_shiftedz.append(np.fft.fftshift(fourier_coeffsz[-1]))
    
    # Compute corresponding frequencies
    freqsx = np.fft.fftshift(np.fft.fftfreq(len(pointsx), d=dx))
    freqsy = np.fft.fftshift(np.fft.fftfreq(len(pointsy), d=dy))
    freqsz = np.fft.fftshift(np.fft.fftfreq(len(pointsz), d=dz))
    # Create a 3D meshgrid of frequencies
    Fx, Fy= np.meshgrid(freqsx, freqsy, indexing="ij")
    
    #i also want to visiualize the maximum freqeuncies the next 3 coarser grids can handle
    hx = 1/Nel[0]
    next_max_frequenciesx = [1.0/(4.0*hx),1.0/(8.0*hx),1.0/(16.0*hx)]
    hy = 1/Nel[1]
    next_max_frequenciesy = [1.0/(4.0*hy),1.0/(8.0*hy),1.0/(16.0*hy)]
    #I get the amplitude of the maximum initial error wave
    #max_amplitude = np.max(np.abs(fourier_coeffs_shiftedx[0][:,1,1]))
    
    if(model == "Poisson"):
        magnitude_spectrum = []
        log_magnitude_spectrum = [] 
        #We also want to see the relative difference between the initial amplitude and the amplitude after smoothing.
        #We compute abs(A_0)-abs(A_i)/abs(A_0), this tell us which percentage of the initial amplitude has been eliminated
        #A values of 0 means it all still remains, a value of 1 means we eliminated all the amplitude. And a negative value 
        #means that the amplitude increased.
        eliminated_portion = []
        for i in range(2):
            aux = (np.abs(fourier_coeffs_shiftedx[0][:,:,1]) - np.abs(fourier_coeffs_shiftedx[i+1][:,:,1]))/np.abs(fourier_coeffs_shiftedx[0][:,:,1])
            eliminated_portion.append(aux)
             # Create interactive 3D surface plot with Plotly
            fig = go.Figure()

            fig.add_trace(go.Surface(z=eliminated_portion[i], x=Fx, y=Fy, colorscale="Viridis"))

            # Set log scale for Z-axis
            fig.update_layout(
                title="Iteration "+str(max_iter_list[i+1])+". Eliminated portion of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",
                scene=dict(
                    xaxis_title="Frequency X",
                    yaxis_title="Frequency Y",
                    zaxis_title="Eliminated portion of initial amplitude",
                    #zaxis_type="log",  # Apply log scale to Z-axis
                )
            )

            # Save as an interactive HTML file
            fig.write_html("Eliminated-Portion-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+"-iteration-"+str(max_iter_list[i+1])+".html")

            # Show in browser
            #fig.show()
            
            
        for i in range(3):
            magnitude_spectrum.append(np.abs(fourier_coeffs_shiftedx[i][:,:,1]))
            log_magnitude_spectrum.append(np.log1p(magnitude_spectrum[i]))
            
            # Plot 3D surface of Fourier magnitude spectrum
            #fig = plt.figure(figsize=(10, 7))
            #ax = fig.add_subplot(111, projection="3d")
            #ax.plot_surface(Fx, Fy, log_magnitude_spectrum[i], cmap="viridis")
            
            # Add contour plot at the bottom (Z = 0)
            #contour = ax.contourf(Fx, Fy, log_magnitude_spectrum[i], zdir="z", offset=log_magnitude_spectrum[i].min(), cmap="viridis")
            
            # Labels
            #ax.set_xlabel("Frequency X")
            #ax.set_ylabel("Frequency Y")
            #ax.set_zlabel("Log(Amplitude + 1)")
            #ax.set_title("Iteration "+str(max_iter_list[i])+". Amplitude of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".")
            # Adjust view angle for better visibility
            #ax.view_init(elev=30, azim=135)
            #plt.show()
            #plt.close()
            
            # Create interactive 3D surface plot with Plotly
            fig = go.Figure()

            fig.add_trace(go.Surface(z=log_magnitude_spectrum[i], x=Fx, y=Fy, colorscale="Viridis"))

            # Set log scale for Z-axis
            fig.update_layout(
                title="Iteration "+str(max_iter_list[i])+". Amplitude of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",
                scene=dict(
                    xaxis_title="Frequency X",
                    yaxis_title="Frequency Y",
                    zaxis_title="Log(Amplitude + 1)",
                    #zaxis_type="log",  # Apply log scale to Z-axis
                )
            )

            # Save as an interactive HTML file
            fig.write_html("Wave-Amplitude-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+"-iteration-"+str(max_iter_list[i])+".html")

            # Show in browser
            #fig.show()
               
        
        
    else:
        
        magnitude_spectrumx = []
        log_magnitude_spectrumx = [] 
        #We also want to see the relative difference between the initial amplitude and the amplitude after smoothing.
        #We compute abs(A_0)-abs(A_i)/abs(A_0), this tell us which percentage of the initial amplitude has been eliminated
        #A values of 0 means it all still remains, a value of 1 means we eliminated all the amplitude. And a negative value 
        #means that the amplitude increased.
        eliminated_portionx = []
        for i in range(2):
            aux = (np.abs(fourier_coeffs_shiftedx[0][:,:,1]) - np.abs(fourier_coeffs_shiftedx[i+1][:,:,1]))/np.abs(fourier_coeffs_shiftedx[0][:,:,1])
            eliminated_portionx.append(aux)
             # Create interactive 3D surface plot with Plotly
            fig = go.Figure()

            fig.add_trace(go.Surface(z=eliminated_portionx[i], x=Fx, y=Fy, colorscale="Viridis"))

            # Set log scale for Z-axis
            fig.update_layout(
                title="X-component. Iteration "+str(max_iter_list[i+1])+". Eliminated portion of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",
                scene=dict(
                    xaxis_title="Frequency X",
                    yaxis_title="Frequency Y",
                    zaxis_title="Eliminated portion of initial amplitude",
                    #zaxis_type="log",  # Apply log scale to Z-axis
                )
            )

            # Save as an interactive HTML file
            fig.write_html("Eliminated-Portion-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+"-iteration-"+str(max_iter_list[i+1])+"X-component.html")

            # Show in browser
            #fig.show()
            
            
        for i in range(3):
            magnitude_spectrumx.append(np.abs(fourier_coeffs_shiftedx[i][:,:,1]))
            log_magnitude_spectrumx.append(np.log1p(magnitude_spectrumx[i]))
            
            # Create interactive 3D surface plot with Plotly
            fig = go.Figure()

            fig.add_trace(go.Surface(z=log_magnitude_spectrumx[i], x=Fx, y=Fy, colorscale="Viridis"))

            # Set log scale for Z-axis
            fig.update_layout(
                title="X-component. Iteration "+str(max_iter_list[i])+". Amplitude of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",
                scene=dict(
                    xaxis_title="Frequency X",
                    yaxis_title="Frequency Y",
                    zaxis_title="Log(Amplitude + 1)",
                    #zaxis_type="log",  # Apply log scale to Z-axis
                )
            )

            # Save as an interactive HTML file
            fig.write_html("Wave-Amplitude-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+"-iteration-"+str(max_iter_list[i])+"X-component.html")

            # Show in browser
            #fig.show()
            
            
        magnitude_spectrumy = []
        log_magnitude_spectrumy = [] 
        #We also want to see the relative difference between the initial amplitude and the amplitude after smoothing.
        #We compute abs(A_0)-abs(A_i)/abs(A_0), this tell us which percentage of the initial amplitude has been eliminated
        #A values of 0 means it all still remains, a value of 1 means we eliminated all the amplitude. And a negative value 
        #means that the amplitude increased.
        eliminated_portiony = []
        for i in range(2):
            aux = (np.abs(fourier_coeffs_shiftedy[0][:,:,1]) - np.abs(fourier_coeffs_shiftedy[i+1][:,:,1]))/np.abs(fourier_coeffs_shiftedy[0][:,:,1])
            eliminated_portiony.append(aux)
             # Create interactive 3D surface plot with Plotly
            fig = go.Figure()

            fig.add_trace(go.Surface(z=eliminated_portiony[i], x=Fx, y=Fy, colorscale="Viridis"))

            # Set log scale for Z-axis
            fig.update_layout(
                title="Y-component. Iteration "+str(max_iter_list[i+1])+". Eliminated portion of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",
                scene=dict(
                    xaxis_title="Frequency X",
                    yaxis_title="Frequency Y",
                    zaxis_title="Eliminated portion of initial amplitude",
                    #zaxis_type="log",  # Apply log scale to Z-axis
                )
            )

            # Save as an interactive HTML file
            fig.write_html("Eliminated-Portion-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+"-iteration-"+str(max_iter_list[i+1])+"Y-component.html")

            # Show in browser
            #fig.show()
            
            
        for i in range(3):
            magnitude_spectrumy.append(np.abs(fourier_coeffs_shiftedy[i][:,:,1]))
            log_magnitude_spectrumy.append(np.log1p(magnitude_spectrumy[i]))
            
            # Create interactive 3D surface plot with Plotly
            fig = go.Figure()

            fig.add_trace(go.Surface(z=log_magnitude_spectrumy[i], x=Fx, y=Fy, colorscale="Viridis"))

            # Set log scale for Z-axis
            fig.update_layout(
                title="Y-component. Iteration "+str(max_iter_list[i])+". Amplitude of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",
                scene=dict(
                    xaxis_title="Frequency X",
                    yaxis_title="Frequency Y",
                    zaxis_title="Log(Amplitude + 1)",
                    #zaxis_type="log",  # Apply log scale to Z-axis
                )
            )

            # Save as an interactive HTML file
            fig.write_html("Wave-Amplitude-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+"-iteration-"+str(max_iter_list[i])+"Y-component.html")

            # Show in browser
            #fig.show()
            
        magnitude_spectrumz = []
        log_magnitude_spectrumz = [] 
        #We also want to see the relative difference between the initial amplitude and the amplitude after smoothing.
        #We compute abs(A_0)-abs(A_i)/abs(A_0), this tell us which percentage of the initial amplitude has been eliminated
        #A values of 0 means it all still remains, a value of 1 means we eliminated all the amplitude. And a negative value 
        #means that the amplitude increased.
        eliminated_portionz = []
        for i in range(2):
            aux = (np.abs(fourier_coeffs_shiftedz[0][:,:,1]) - np.abs(fourier_coeffs_shiftedz[i+1][:,:,1]))/np.abs(fourier_coeffs_shiftedz[0][:,:,1])
            eliminated_portionz.append(aux)
             # Create interactive 3D surface plot with Plotly
            fig = go.Figure()

            fig.add_trace(go.Surface(z=eliminated_portionz[i], x=Fx, y=Fy, colorscale="Viridis"))

            # Set log scale for Z-axis
            fig.update_layout(
                title="Z-component. Iteration "+str(max_iter_list[i+1])+". Eliminated portion of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",
                scene=dict(
                    xaxis_title="Frequency X",
                    yaxis_title="Frequency Y",
                    zaxis_title="Eliminated portion of initial amplitude",
                    #zaxis_type="log",  # Apply log scale to Z-axis
                )
            )

            # Save as an interactive HTML file
            fig.write_html("Eliminated-Portion-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+"-iteration-"+str(max_iter_list[i+1])+"Z-component.html")

            # Show in browser
            #fig.show()
            
            
        for i in range(3):
            magnitude_spectrumz.append(np.abs(fourier_coeffs_shiftedz[i][:,:,1]))
            log_magnitude_spectrumz.append(np.log1p(magnitude_spectrumz[i]))
            
            # Create interactive 3D surface plot with Plotly
            fig = go.Figure()

            fig.add_trace(go.Surface(z=log_magnitude_spectrumz[i], x=Fx, y=Fy, colorscale="Viridis"))

            # Set log scale for Z-axis
            fig.update_layout(
                title="Z-component. Iteration "+str(max_iter_list[i])+". Amplitude of error waves for " + model + " with " +smoother+" solver, with resolution " + str(Nel)+".",
                scene=dict(
                    xaxis_title="Frequency X",
                    yaxis_title="Frequency Y",
                    zaxis_title="Log(Amplitude + 1)",
                    #zaxis_type="log",  # Apply log scale to Z-axis
                )
            )

            # Save as an interactive HTML file
            fig.write_html("Wave-Amplitude-Cuboid-"+model+"-"+smoother+"-"+str(Nel)+"-iteration-"+str(max_iter_list[i])+"Z-component.html")

            # Show in browser
            #fig.show()
        

def make_plot_smoothing():
    from numpy import array
    
    #######
    #Example 1
    #Shear-Alfven
    #Cuboid
    #2D
    #CG
    ######
    
    #Nel = [128,128,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #CG
    
    #Magnitude of the high frequency (period smaller than 4h) with highest magnitude
    
    
    ############################
    #Nel = [64,64,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #CG
    

    ############################
    #Nel = [32,32,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #CG
    

    ############################
    #Nel = [16,16,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #CG
    

    
    ############################
    #Nel = [8,8,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #CG
    
    
    ################################################################
    ################################################################
    ################################################################
    ################################################################
    #######
    #Example 2
    #Shear-Alfven
    #Cuboid
    #2D
    #biCG
    ######
    
    ############################
    #Nel = [128,128,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #biCG
    


    
    ############################
    #Nel = [64,64,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #biCG
    

    ############################
    #Nel = [32,32,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #biCG
    
    

    ############################
    #Nel = [16,16,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #biCG
    
    
    
    
    ################################################################
    ################################################################
    ################################################################
    ################################################################
    #######
    #Example 3
    #Shear-Alfven
    #Cuboid
    #2D
    #bicgstab
    ######
    #Terrible the high frequency errors increase with the number of itterations
    
    ############################
    #Nel = [128,128,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #bicgstab
    
    

    
    ############################
    #Nel = [64,64,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #bicgstab
    
    
    

    ################################################################
    ################################################################
    ################################################################
    ################################################################
    
    
    #######
    #Example 4
    #Shear-Alfven
    #Cuboid
    #2D
    #minres
    
    ############################
    #Nel = [128,128,1]
    #p = [1,1,1]
    #Hall
    #Cuboid
    #minres
    
    magnitudes = [23475.547406074205, 23507.57071761296, 23401.98167905894, 8975.398716277663, 5547.855656375274, 2215.014445147343, 1060.3501137658902, 829.7441809688149, 220.71658497349233, 257.8922866406492]
    bad_frequencies = [array([ 44.6484375, -14.8828125, -62.5078125]), array([-44.6484375,  14.8828125, -62.5078125]), array([-44.6484375,  14.8828125, -62.5078125]), array([ 62.5078125, -31.75     , -62.5078125]), array([-63.5      , -43.65625  , -62.5078125]), array([-59.53125  ,  62.5078125, -62.5078125]), array([-59.53125  ,  62.5078125, -62.5078125]), array([ 59.53125  , -62.5078125, -62.5078125]), array([-59.53125  ,  62.5078125, -62.5078125]), array([-59.53125  ,  62.5078125, -62.5078125])]
    N_itter = [2, 10, 20, 100, 200, 300, 400, 500, 600, 700]


    ############################
    #Nel = [64,64,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #minres
    
    magnitudes = [4816.563672246145, 4820.683874727215, 4750.075275409996, 1435.7294516177942, 230.5875341851168, 45.396511954402975, 6.7027103058152555, 6.6240853651862395, 5.862590252251258, 0.753372104197576]
    bad_frequencies = [array([-24.609375, -11.8125  , -30.515625]), array([-24.609375, -11.8125  , -30.515625]), array([-24.609375, -11.8125  , -30.515625]), array([-30.515625,  25.59375 , -30.515625]), array([-30.515625,  25.59375 , -30.515625]), array([ 29.53125 ,  30.515625, -30.515625]), array([-30.515625,  25.59375 , -30.515625]), array([-29.53125 , -30.515625, -30.515625]), array([-29.53125 , -30.515625, -30.515625]), array([-24.609375, -12.796875, -30.515625])]
    N_itter = [2, 10, 20, 100, 200, 300, 400, 500, 600, 700]

    
    ############################
    #Nel = [32,32,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #minres
    
    magnitudes = [1319.382611683582, 1271.7302885073973, 1096.5001351152466, 66.25984056212276, 2.305135582316737, 0.5738656559575724, 0.13428796602415516, 0.0666372580855523, 0.045512532749170435, 0.030227161820733185]
    bad_frequencies = [array([-13.5625 ,   0.     , -14.53125]), array([-13.5625 ,   0.     , -14.53125]), array([-13.5625 ,   0.     , -14.53125]), array([-12.59375, -13.5625 , -14.53125]), array([-12.59375, -13.5625 , -14.53125]), array([-13.5625 , -14.53125, -14.53125]), array([-13.5625 , -14.53125, -14.53125]), array([-12.59375, -15.5    , -14.53125]), array([-12.59375, -13.5625 , -14.53125]), array([ 14.53125,  14.53125, -14.53125])]
    N_itter = [2, 10, 20, 100, 200, 300, 400, 500, 600, 700]

    
    ################################################################
    ################################################################
    ################################################################
    ################################################################
    
    #######
    #Example 5
    #Shear-Alfven
    #Cuboid
    #2D
    #lsmr
    
    ############################
    #Nel = [128,128,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #lsmr
    magnitudes = [23625.08035709719, 23546.07810652249, 23555.35959665252, 23553.74183488159, 23553.11191281097, 23552.06535579401, 23550.62355697153, 23548.584106844555, 23546.582190989895, 23543.658419538646]
    bad_frequencies = [array([-40.6796875,  -0.9921875, -62.5078125]), array([-44.6484375,  14.8828125, -62.5078125]), array([-44.6484375,  14.8828125, -62.5078125]), array([-44.6484375,  14.8828125, -62.5078125]), array([-44.6484375,  14.8828125, -62.5078125]), array([-44.6484375,  14.8828125, -62.5078125]), array([-44.6484375,  14.8828125, -62.5078125]), array([-44.6484375,  14.8828125, -62.5078125]), array([-44.6484375,  14.8828125, -62.5078125]), array([-44.6484375,  14.8828125, -62.5078125])]
    N_itter = [2, 10, 20, 100, 200, 300, 400, 500, 600, 700]

    ############################
    #Nel = [64,64,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #lsmr
    
    magnitudes = [4902.303542373743, 4846.903947432333, 4849.383867406714, 4849.313905985293, 4847.890279297882, 4845.825151419513, 4842.427718581035, 4839.574277539976, 4832.651039101285, 4824.670040396377]
    bad_frequencies = [array([-24.609375, -11.8125  , -30.515625]), array([-24.609375, -11.8125  , -30.515625]), array([-24.609375, -11.8125  , -30.515625]), array([ 24.609375,  11.8125  , -30.515625]), array([-24.609375, -11.8125  , -30.515625]), array([-24.609375, -11.8125  , -30.515625]), array([-24.609375, -11.8125  , -30.515625]), array([-24.609375, -11.8125  , -30.515625]), array([-24.609375, -11.8125  , -30.515625]), array([-24.609375, -11.8125  , -30.515625])]
    N_itter = [2, 10, 20, 100, 200, 300, 400, 500, 600, 700]

    
    ################################################################
    ################################################################
    ################################################################
    ################################################################
    
    #######
    #Example 6
    #Shear-Alfven
    #Cuboid
    #2D
    #gmres
    
    ############################
    #Nel = [128,128,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #gmres
    
    magnitudes = [23631.325965428965, 23517.072741223346, 23415.519704485465, 9143.805698693457, 5620.759347312477, 2224.5465588280504, 1062.7363141542924, 813.4419688746405, 239.7543262098424, 257.9845124527011]
    bad_frequencies = [array([-40.6796875,  -0.9921875, -62.5078125]), array([-44.6484375,  14.8828125, -62.5078125]), array([ 44.6484375, -14.8828125, -62.5078125]), array([-62.5078125,  31.75     , -62.5078125]), array([-63.5      , -43.65625  , -62.5078125]), array([ 59.53125  , -62.5078125, -62.5078125]), array([-59.53125  ,  62.5078125, -62.5078125]), array([-59.53125  ,  62.5078125, -62.5078125]), array([-59.53125  ,  62.5078125, -62.5078125]), array([-59.53125  ,  62.5078125, -62.5078125])]
    N_itter = [2, 10, 20, 100, 200, 300, 400, 500, 600, 700]

    
    
    
    
    
    
    
    
    
    
    
    #######
    #Example 
    #Shear-Alfven
    #Cuboid
    #1D
    #CG
    ######
    
    
    ############################
    #Nel = [1024,1,1]
    #p = [1,1,1]
    #Shear-Alfven
    #Cuboid
    #CG
    
    
    #As you can see in 1D the CG reduces the magnitude of all problematic high frequencies to 2 e-05. In 1D the CG for the shear-alfven matrix is a good smoother. Thus the Multigrid method works
    #with it. But in 2D the CG method is a terrible smoother for this matrix so the multigrid method becomes almost useless with it.


def multigrid_Alfven(Nel, plist, spl_kind, u_space):
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    derham = Derham(Nel, plist, spl_kind, comm=comm, local_projectors=False)
    domain = Tokamak(p = [3,3], psi_shifts = [2.,2.])
    
    mhd_equil = AdhocTorusQPsi()

    # must set domain of Cartesian MHD equilibirum
    mhd_equil.domain = domain
    

    input = derham.Vh[derham.space_to_form[u_space]].zeros()
    
    
    mass_ops = WeightedMassOperators(derham, domain)
    basis_ops = BasisProjectionOperators(derham, domain,eq_mhd=mhd_equil)

    id_T = "T" + derham.space_to_form[u_space]

    _T = getattr(basis_ops, id_T)

    _B = -1 / 2 * _T.T @ derham.curl.T @ mass_ops.M2
    _C = 1 / 2 * derham.curl @ _T
    
    _BC = _B @ _C

    solver = inverse(_BC,'cg')
    x = solver.dot(input)
    
    print(solver._info)
   

def multigrid(Nel, plist, spl_kind, N_levels):
    from struphy.feec.utilities import create_equal_random_arrays
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    domain = Cuboid()
    #a1 = 0.002
    #domain = HollowCylinder(a1= a1)
    sp_key = '0'
    
    derham = []
    mass_ops = []
    A = []
    
    epsilon = 0.0002
    for level in range(N_levels):
        derham.append(Derham([Nel[0]//(2**level),Nel[1]//(2**level),Nel[2]], plist, spl_kind, comm=comm, local_projectors=False))
        mass_ops.append(WeightedMassOperators(derham[level], domain))
        A.append(derham[level].grad.T @ mass_ops[level].M1 @ derham[level].grad)
        #A.append(epsilon*derham[level].curl.T @ mass_ops[level].M2 @ derham[level].curl + mass_ops[level].M1)
    
    #We get the inverse of the coarsest system matrix to solve directly the problem in the smaller space
    A_inv = np.linalg.inv(A[-1].toarray())
    
    R = []
    E = []
    
    for level in range(N_levels-1):
        R.append(RestrictionOperator(derham[level].Vh_fem[sp_key],derham[level+1].Vh_fem[sp_key]))
        E.append(R[level].transpose())
        
    method = 'cg'
    
    #800
    max_iter_list = [14,14,14,18,12,10]
    #40
    N_cycles = 2
    
    u_stararr, u_star = create_equal_random_arrays(derham[0].Vh_fem[sp_key], seed=45)
    
    #We compute the rhs
    b = A[0].dot(u_star)
    
    timei = time.time()
    
    solver_no = inverse(A[0],method, maxiter = 1000000, tol = 10**(-6))
    
    u = solver_no.dot(b)
    
    timef = time.time()
    
    #We get the total number of itterations
    No_Multigrid_itterations = solver_no._info['niter']
    No_Multigrid_error = solver_no._info['res_norm']
    No_Multigrid_time = timef-timei
    
    #No_Multigrid_itterations = 35088
    #No_Multigrid_error = 9.22E-07
    #No_Multigrid_time = 451.228641271591
    
    print("################")
    print(f'{No_Multigrid_itterations = }')
    print(f'{No_Multigrid_error = }')
    print(f'{No_Multigrid_time = }')
    print("################")
    
    
    def call_multigrid(max_iter, N_cycles):
        u_stararr, u_star = create_equal_random_arrays(derham[0].Vh_fem[sp_key], seed=45)
        #We compute the rhs
        b = A[0].dot(u_star)
        
        #We define a list where to store the number of itteration it takes at each multigrid level
        #Change N_levels for 1D case
        Multigrid_itterations = np.zeros(N_levels, dtype=int)
        converged = np.zeros(N_levels, dtype=bool)
        
        def V_cycle(l, r_l):
            #Change for N_levels-1 for 1D case
            if (l < N_levels-1):
                solver_ini = inverse(A[l],method, maxiter= max_iter[l])
                x_l = solver_ini.dot(r_l)
                
                #We count the number of itterations
                Multigrid_itterations[l] += solver_ini._info['niter']
                
                #We determine if the itterative solver converged in the maximum number of itterations
                converged[l] = solver_ini._info['success']
                if converged[l] == True:
                    return x_l
                
                r_l = r_l - A[l].dot(x_l)
                
                r_l_plus_1 = R[l].dot(r_l)
                x_l_plus_1 = V_cycle(l+1, r_l_plus_1)
                #New
                #x_l_aux = E[l].dot(x_l_plus_1)
                #x_l = x_l + x_l_aux
                #r_l = r_l  - A[l].dot(x_l_aux)
                #solver_end = inverse(A[l].T,method, maxiter= max_iter, x0 =x_l)
                #x_l = solver_end.dot(r_l)
                #Multigrid_itterations[l] += solver_end._info['niter']
                ####
                #old
                x_l = x_l + E[l].dot(x_l_plus_1)
                ###
                
            else:
                #Solve directly
                x_l = direct_solver(A_inv,r_l, derham[l].Vh_fem[sp_key])
                
            return x_l 
        
        #N_cycles = 6
        x_0 = derham[0].Vh_fem[sp_key].vector_space.zeros()
        
        timei = time.time()
        for cycle in range(N_cycles):
            solver = inverse(A[0],method, maxiter= max_iter[0], x0 = x_0)
            x_0 = solver.dot(b)
            
            Multigrid_itterations[0] += solver._info['niter']
            
            #We determine if the itterative solver converged in the maximum number of itterations
            converged[0] = solver._info['success']
            if converged[0] == True:
                print("Hello")
                x = x_0
                break
            
            r_0 = b - A[0].dot(x_0)
            r_1 = R[0].dot(r_0)
            
            x_0 = x_0 + E[0].dot(V_cycle(1,r_1))
        
        if converged[0] == False:
            solver = inverse(A[0],method,x0 = x_0, tol = 10**(-6))
            x = solver.dot(b)
            Multigrid_itterations[0] += solver._info['niter']
        
        timef = time.time()
        
        
        
        #We get the final error
        Multigrid_error = solver._info['res_norm']
        
        Multigrid_time = timef- timei
        
        speed_up = No_Multigrid_time / Multigrid_time
        #speed_up = 27.3452200889587 / Multigrid_time
        
        print("################")
        print("################")
        print("################")
        #print(f'{a1 = }')
        print(f'{max_iter = }')
        print(f'{N_cycles = }')
        print("################")
        print("################")
        print(f'{Multigrid_itterations = }')
        print(f'{Multigrid_error = }')
        print(f'{Multigrid_time = }')
        print("################")
        print("################")
        print(f'{speed_up = }')   
        
    call_multigrid(max_iter_list, N_cycles) 
          
        
def Error_analysis(Nel, plist, spl_kind, N_levels):
    
    def determine_error(exact,aprox,x):
        errors = exact(x,0.0,0.0)-aprox(x,0.0,0.0)
        errors = errors.flatten()
        return errors
        
    
    
    
    from struphy.feec.utilities import create_equal_random_arrays
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    domain = Cuboid()
    
    sp_key = '0'
    derham = []
    mass_ops = []
    A = []
    
    
    for level in range(N_levels):
    
        derham.append(Derham([Nel[0]//(2**level),Nel[1],Nel[2]], plist, spl_kind, comm=comm, local_projectors=False))
        mass_ops.append(WeightedMassOperators(derham[level], domain))
        A.append(derham[level].grad.T @ mass_ops[level].M1 @ derham[level].grad)
       
    
    field_star = derham[0].create_field('fh', 'H1')
    field_aprox = derham[0].create_field('fh', 'H1')
    
    #We are gonna use the method of manufacture solutions to determine the behaviour of our error
    #Our exact solution is u_star
    u_stararr, u_star = create_equal_random_arrays(derham[0].Vh_fem[sp_key], seed=45)
    #We compute the rhs
    b = A[0].dot(u_star)
    field_star.vector = u_star
    
    #We store the number of itterations
    N_itter = []
    #We define a set of point to evaluate the exact solution and the aproximated one
    points = np.linspace(0.0,1.0,1000)
    dx = points[1] - points[0]  # Spacing between points
    #We define a list where to store the arrays with the values of the errors for each aproximation
    errors = []
    
    #Fourier Transform coefficients of the error function
    fourier_coeffs = []
    #Corresponding frequencies
    freqs = []
    for i in range(1, 10):
        solver = inverse(A[0],'cg', maxiter = int(i*50) )
        u = solver.dot(b)
        field_aprox.vector = u
        errors.append(determine_error(field_star,field_aprox,points))
        N_itter.append(solver._info['niter'])
        
        # Compute Fourier Transform coefficients
        fourier_coeffs.append(np.fft.fft(errors[-1]))
        # Compute corresponding frequencies
        freqs.append(np.fft.fftfreq(len(errors[-1]), d=dx))
        
        
    
    plt.figure()
    plt.plot(points,errors[0], label = str(N_itter[0]))
    plt.plot(points,errors[-1], label = str(N_itter[-1]))
    #plt.scatter(freqs[0],fourier_coeffs[0], label = str(N_itter[0]))
    #plt.scatter(freqs[3],fourier_coeffs[3], label = str(N_itter[3]))
    #plt.xlim(-50,50)
    plt.legend()
    plt.show()
    plt.close()
    
    
def Gather_data_V_cycle_parameter_study(Nel, plist, spl_kind, N_levels):
    from struphy.feec.utilities import create_equal_random_arrays
    # get global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    domain = Cuboid()
    a1 = 0.002
    #domain = HollowCylinder(a1= a1)
    sp_key = '2'
    
    derham = []
    mass_ops = []
    A = []
    
    epsilon = 1.0
    for level in range(N_levels):
        derham.append(Derham([Nel[0]//(2**level),Nel[1]//(2**level),Nel[2]], plist, spl_kind, comm=comm, local_projectors=False))
        mass_ops.append(WeightedMassOperators(derham[level], domain))
        #Poisson
        #A.append(derham[level].grad.T @ mass_ops[level].M1 @ derham[level].grad)
        #Hall-ish
        #A.append(epsilon*derham[level].curl.T @ mass_ops[level].M2 @ derham[level].curl + mass_ops[level].M1)
        #Hall
        #A.append(mass_ops[level].M1 -epsilon*derham[level].curl.T @ mass_ops[level].M2 @ derham[level].curl)
        #Shear-Alfven-ish
        #A.append(mass_ops[level].M2 -epsilon* mass_ops[level].M2@ derham[level].curl @ mass_ops[level].M1 @ derham[level].curl.T @ mass_ops[level].M2)
        #Shear-Alfven-ish-v2
        #A.append(mass_ops[level].M2 -epsilon* derham[level].curl @ mass_ops[level].M1 @ derham[level].curl.T)
        #Shear-Alfven
        pc_class = getattr(preconditioner,"MassMatrixPreconditioner")
        pc = pc_class(mass_ops[level].M1)
        M1_inv = inverse(
            mass_ops[level].M1,
            "pcg",
            pc=pc,
            maxiter=3000,
            verbose=False,
        )
        A.append(mass_ops[level].M2 -epsilon* mass_ops[level].M2@ derham[level].curl @ M1_inv @ derham[level].curl.T @ mass_ops[level].M2)
        #Shear-Alfven-v2
        #A.append(mass_ops[level].M2 -epsilon* derham[level].curl @ M1_inv @ derham[level].curl.T)
    
    #We get the inverse of the coarsest system matrix to solve directly the problem in the smaller space
    A_inv = np.linalg.inv(A[-1].toarray())
    
    R = []
    E = []
    
    for level in range(N_levels-1):
        R.append(RestrictionOperator(derham[level].Vh_fem[sp_key],derham[level+1].Vh_fem[sp_key]))
        E.append(R[level].transpose())
        
    method = 'cg'
    
    #800
    max_iter_list = [93]
    #40
    N_cycles_list = [5]
    
    u_stararr, u_star = create_equal_random_arrays(derham[0].Vh_fem[sp_key], seed=45)
    #We compute the rhs
    b = A[0].dot(u_star)
    
    timei = time.time()
    
    solver_no = inverse(A[0],method, maxiter = 10000)
    
    u = solver_no.dot(b)
    
    timef = time.time()
    
    #We get the total number of itterations
    No_Multigrid_itterations = solver_no._info['niter']
    No_Multigrid_error = solver_no._info['res_norm']
    No_Multigrid_time = timef-timei
    
    #No_Multigrid_itterations = 9485
    #No_Multigrid_error = 9.72E-07
    #No_Multigrid_time = 867.623549938202
    print("################")
    print("################")
    print(f'{Nel[0] = }')
    print(f'{Nel[1] = }')
    print("################")
    print("################")
    
    
    print("################")
    print(f'{No_Multigrid_itterations = }')
    print(f'{No_Multigrid_error = }')
    print(f'{No_Multigrid_time = }')
    print("################")
    
    
    def call_multigrid(max_iter, N_cycles):
        u_stararr, u_star = create_equal_random_arrays(derham[0].Vh_fem[sp_key], seed=45)
        #We compute the rhs
        b = A[0].dot(u_star)
        
        #We define a list where to store the number of itteration it takes at each multigrid level
        #Change N_levels for 1D case
        Multigrid_itterations = np.zeros(N_levels, dtype=int)
        converged = np.zeros(N_levels, dtype=bool)
        
        def V_cycle(l, r_l):
            #Change for N_levels-1 for 1D case
            if (l < N_levels-1):
                solver_ini = inverse(A[l],method, maxiter= max_iter)
                x_l = solver_ini.dot(r_l)
                
                #We count the number of itterations
                Multigrid_itterations[l] += solver_ini._info['niter']
                
                #We determine if the itterative solver converged in the maximum number of itterations
                converged[l] = solver_ini._info['success']
                if converged[l] == True:
                    return x_l
                
                r_l = r_l - A[l].dot(x_l)
                
                r_l_plus_1 = R[l].dot(r_l)
                x_l_plus_1 = V_cycle(l+1, r_l_plus_1)
                #New
                #x_l_aux = E[l].dot(x_l_plus_1)
                #x_l = x_l + x_l_aux
                #r_l = r_l  - A[l].dot(x_l_aux)
                #solver_end = inverse(A[l].T,method, maxiter= max_iter, x0 =x_l)
                #x_l = solver_end.dot(r_l)
                #Multigrid_itterations[l] += solver_end._info['niter']
                ####
                #old
                x_l = x_l + E[l].dot(x_l_plus_1)
                ###
                
            else:
                #Solve directly
                x_l = direct_solver(A_inv,r_l, derham[l].Vh_fem[sp_key])
                
            return x_l 
        
        #N_cycles = 6
        x_0 = derham[0].Vh_fem[sp_key].vector_space.zeros()
        
        timei = time.time()
        for cycle in range(N_cycles):
            solver = inverse(A[0],method, maxiter= max_iter, x0 = x_0)
            x_0 = solver.dot(b)
            
            Multigrid_itterations[0] += solver._info['niter']
            
            #We determine if the itterative solver converged in the maximum number of itterations
            converged[0] = solver._info['success']
            if converged[0] == True:
                print("Hello")
                x = x_0
                break
            
            r_0 = b - A[0].dot(x_0)
            r_1 = R[0].dot(r_0)
            
            x_0 = x_0 + E[0].dot(V_cycle(1,r_1))
        
        if converged[0] == False:
            solver = inverse(A[0],method, maxiter = 1000,x0 = x_0,)
            x = solver.dot(b)
            Multigrid_itterations[0] += solver._info['niter']
        
        timef = time.time()
        
        
        
        #We get the final error
        Multigrid_error = solver._info['res_norm']
        
        Multigrid_time = timef- timei
        
        speed_up = No_Multigrid_time / Multigrid_time
        
        print("################")
        print("################")
        print("################")
        #print(f'{a1 = }')
        print(f'{max_iter = }')
        print(f'{N_cycles = }')
        print("################")
        print("################")
        print(f'{Multigrid_itterations = }')
        print(f'{Multigrid_error = }')
        print(f'{Multigrid_time = }')
        print("################")
        print("################")
        print(f'{speed_up = }')   
        
    for max_iter in max_iter_list:
        for N_cycles in N_cycles_list:
            call_multigrid(max_iter, N_cycles)  


def Gather_data_V_cycle_scalability(Nellist, plist, spl_kind):
    for Nel in Nellist:
        #First we compute the number of levels for each Nel
        N_levels = int(log2(Nel[0])-1)
        Gather_data_V_cycle_parameter_study(Nel, plist, spl_kind, N_levels)

          
def make_plot_scalability():  
    
    model = 'Hall'
    
    Nel = [int(2**i * 2**i) for i in range(4,8)]
    Multi_time = [0.933451652526856,
                    4.96578407287598,
                    21.1356129646301,
                    121.159014463425
                    ]    
    No_Multi_time = [0.853443145751953,
                        6.89243221282959,
                        75.1050372123718,
                        1334.72232866287
                        ]
    speed_up = []
    
    for i in range(len(Multi_time)):
        speed_up.append(No_Multi_time[i]/Multi_time[i])
    
    plt.figure()
    plt.title('2D ' +model+' run times.')
    plt.plot(Nel, Multi_time, label = 'Multigrid.')
    plt.scatter(Nel, No_Multi_time, label = 'CG run time.')
    plt.xlabel("Nel[0] x Nel[1]")
    plt.ylabel('Time (s)')
    plt.legend()
    plt.savefig("2D-"+model+"-runtime.pdf")
    #plt.show()
    plt.close()
    
    plt.figure()
    plt.title('2D ' +model+ ' speed up.')
    plt.scatter(Nel, speed_up, label = 'Speed_up.')
    plt.xlabel("Nel[0] x Nel[1]")
    plt.ylabel('Speed_up')
    plt.legend()
    #plt.show()
    plt.savefig("2D-"+model+"-speed-up.pdf")
    plt.close()

    
def verify_formula(Nel, plist, spl_kind):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    derham = Derham(Nel, plist, spl_kind, comm=comm)

    # For B-splines
    sp_key = "0"
    spaces = derham.Vh_fem[sp_key].spaces
    space = spaces[0]
    N = space.nbasis
    ncells = space.ncells
    p = space.degree
    T = space.knots
    periodic = space.periodic
    basis = space.basis
    normalize = basis == "M"
    
    def make_basis_fun(i):
        def fun(etas, eta2, eta3):
            if isinstance(etas, float) or isinstance(etas, int):
                etas = np.array([etas])
            out = np.zeros_like(etas)
            for j, eta in enumerate(etas):
                span = find_span(T, p, eta)
                inds = np.arange(span - p, span + 1) % N
                pos = np.argwhere(inds == i)
                # print(f'{pos = }')
                if pos.size > 0:
                    pos = pos[0, 0]
                    out[j] = basis_funs(T, p, eta, span, normalize=normalize)[pos]
                else:
                    out[j] = 0.0
            return out

        return fun

    i = random.randint(0,N-1)
    fun = make_basis_fun(i)
    points = np.linspace(0.0,1.0,100)
    values = fun(points,0.0,0.0)
    
    
    #Now we build the fine grid
    derham_fine = Derham([Nel[0]*2,Nel[1],Nel[2]], plist, spl_kind, comm=comm)

    # For B-splines
    spaces_fine = derham_fine.Vh_fem[sp_key].spaces
    space_fine = spaces_fine[0]
    N_fine = space_fine.nbasis
    ncells_fine = space_fine.ncells
    p_fine = space_fine.degree
    T_fine = space_fine.knots
    periodic_fine = space_fine.periodic
    basis_fine = space_fine.basis
    normalize_fine = basis_fine == "M"
    
    def make_basis_fun_fine(i):
        def fun(etas, eta2, eta3):
            if isinstance(etas, float) or isinstance(etas, int):
                etas = np.array([etas])
            out = np.zeros_like(etas)
            for j, eta in enumerate(etas):
                span = find_span(T_fine, p_fine, eta)
                inds = np.arange(span - p_fine, span + 1) % N_fine
                pos = np.argwhere(inds == i)
                # print(f'{pos = }')
                if pos.size > 0:
                    pos = pos[0, 0]
                    out[j] = basis_funs(T_fine, p_fine, eta, span, normalize=normalize_fine)[pos]
                else:
                    out[j] = 0.0
            return out

        return fun
    
    fun_fine = []
    weights_fine = []
    for j in range(p_fine+2):
        fun_fine.append(make_basis_fun_fine((2*i-p_fine+j)%N_fine))
        weights_fine.append(2.0**(-p_fine)*comb(p_fine+1,j))
    
    
    def fun_combine(etas, eta2, eta3):
        if isinstance(etas, float) or isinstance(etas, int):
            etas = np.array([etas])
        out = np.zeros_like(etas)
        for j in range(p_fine+2):
            out += weights_fine[j]*fun_fine[j](etas,0.0,0.0)
        return out
        
    
    
    values_fine = fun_combine(points,0.0,0.0) 
    
    Equal = True
    where = -1
    for j in range(len(values)):
        if(abs(values[j] -values_fine[j])>10**-5):
            Equal = False
            where = j
    
    print(i)
    print(Equal)
    print(where)
    if Equal == False:
        print(values[where]) 
        print(values_fine[where])    
    
   
def verify_Restriction_Operator(Nel, plist, spl_kind):
    from struphy.feec.utilities import create_equal_random_arrays, compare_arrays
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    derham = Derham(Nel, plist, spl_kind, comm=comm)

    # For B-splines
    sp_key = "0"
    spaces = derham.Vh_fem[sp_key].spaces
    space = spaces[0]
    N = space.nbasis
    ncells = space.ncells
    p = space.degree
    T = space.knots
    periodic = space.periodic
    basis = space.basis
    normalize = basis == "M"
    
    #Now we build the fine grid
    derham_fine = Derham([Nel[0]*2,Nel[1],Nel[2]], plist, spl_kind, comm=comm)

    # For B-splines
    spaces_fine = derham_fine.Vh_fem[sp_key].spaces
    space_fine = spaces_fine[0]
    N_fine = space_fine.nbasis
    ncells_fine = space_fine.ncells
    p_fine = space_fine.degree
    T_fine = space_fine.knots
    periodic_fine = space_fine.periodic
    basis_fine = space_fine.basis
    normalize_fine = basis_fine == "M"
    
    #We intialize the restriction operator
    R = RestrictionOperator(derham_fine.Vh_fem[sp_key],derham.Vh_fem[sp_key])
    
    varr, v = create_equal_random_arrays(derham_fine.Vh_fem[sp_key], seed=4568)
    varr = varr[0].flatten()
    
    
    out = R.dot(v)
    
    #To make it easier to read I will extract the data out of out disregarding all the padding it come with
    out_array = np.zeros(N, dtype=float)
    for i in range(N):
        out_array[i] = out[R._out_starts[0]+i,0,0]
    
    #print(out._data)
   
    
    #####
    #Now we build the Restriction matrix directly to verify our RestrictionOperator is working properly.
    R_matrix = np.zeros((N,N_fine),dtype=float)
    for i in range(N):
        start = 2*i-p
        for j in range(p+2):
            R_matrix[i,(start+j)%N_fine] = 2.0**(-p)*comb(p+1,j)
            
    out2 = np.matmul(R_matrix, varr)
    
    
    Equal =True
    where = -1
    for i in range(len(out2)):
        if(abs(out2[i]-out_array[i])>10.0**(-6)):
            Equal = False
            where = i
    
    
    
    print(f'{Equal = }')
    print(f'{out_array = }')
    print(f'{out2 = }')
    if( Equal == False):
        print(f'{where = }')
    
        
def verify_Extension_Operator(Nel, plist, spl_kind):
    from struphy.feec.utilities import create_equal_random_arrays, compare_arrays
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    derham = Derham(Nel, plist, spl_kind, comm=comm)

    # For B-splines
    sp_key = "0"
    spaces = derham.Vh_fem[sp_key].spaces
    space = spaces[0]
    N = space.nbasis
    ncells = space.ncells
    p = space.degree
    T = space.knots
    periodic = space.periodic
    basis = space.basis
    normalize = basis == "M"
    
    #Now we build the fine grid
    derham_fine = Derham([Nel[0]*2,Nel[1],Nel[2]], plist, spl_kind, comm=comm)

    # For B-splines
    spaces_fine = derham_fine.Vh_fem[sp_key].spaces
    space_fine = spaces_fine[0]
    N_fine = space_fine.nbasis
    ncells_fine = space_fine.ncells
    p_fine = space_fine.degree
    T_fine = space_fine.knots
    periodic_fine = space_fine.periodic
    basis_fine = space_fine.basis
    normalize_fine = basis_fine == "M"
    
    #We intialize the restriction operator
    E = ExtensionOperator(derham.Vh_fem[sp_key],derham_fine.Vh_fem[sp_key])
    
    v = derham.Vh[sp_key].zeros()
    varr, v = create_equal_random_arrays(derham.Vh_fem[sp_key], seed=4568)
    varr = varr[0].flatten()
    
    
    out = E.dot(v)
    
    #To make it easier to read I will extract the data out of out disregarding all the padding it come with
    out_array = np.zeros(N_fine, dtype=float)
    for i in range(N_fine):
        out_array[i] = out[E._out_starts[0]+i,0,0]
    
    #print(out._data)
   
    
    #####
    #Now we build the Restriction matrix directly to verify our RestrictionOperator is working properly.
    R_matrix = np.zeros((N,N_fine),dtype=float)
    for i in range(N):
        start = 2*i-p
        for j in range(p+2):
            R_matrix[i,(start+j)%N_fine] = 2.0**(-p)*comb(p+1,j)
            
    E_matrix = R_matrix.T
            
    out2 = np.matmul(E_matrix, varr)
    
    
    Equal =True
    where = -1
    for i in range(len(out2)):
        if(abs(out2[i]-out_array[i])>10.0**(-6)):
            Equal = False
            where = i
    
    
    
    print(f'{Equal = }')
    print(f'{out_array = }')
    print(f'{out2 = }')
    if( Equal == False):
        print(f'{where = }')

    
def testing_random_stuff(Nel,plist,spl_kind):
    epsilon = 10.0**-6.0
    Nel = 512
    jarray = []
    for j in range(Nel):
        jarray.append(float(j))
    
    jarray = np.array(jarray)
    
    compute_directly = True
    
    #matrix = 'Shear-Alfven'
    matrix = 'Hall'
    if matrix == 'Shear-Alfven':
    
        def funl(j):
            return 9.0*Nel - epsilon * (243.0/2.0) * (Nel**3.0) *  (1.0-np.cos(2.0*np.pi *j / Nel)) /(2.0+np.cos(2.0*np.pi *j / Nel))
        
        max_eigen_value_x = 4.0/Nel
        
        
        
        
    elif( matrix == 'Hall'):
        def funl(j):
            return 8.0 /(3.0*Nel) - 18.0*epsilon*Nel + (4.0/(3.0*Nel)+18.0*epsilon*Nel)*np.cos(2.0*np.pi*j/Nel)
        
        max_eigen_value_x = 9.0*Nel
        
        
        
    eigen_values_y_z = funl(jarray)
    max_eigen_value_y_z=abs(eigen_values_y_z[np.argmax(np.abs(eigen_values_y_z))])
    
    
    order_of_magnitude_disparity = np.log10(max_eigen_value_y_z/max_eigen_value_x)
    print(f'{max_eigen_value_y_z = }')
    print(f'{max_eigen_value_x = }')
    print(f'{order_of_magnitude_disparity = }')
    
    plt.figure()
    plt.scatter(jarray,eigen_values_y_z)
    plt.show()
    
    
    if compute_directly:
        # get global communicator
        comm = MPI.COMM_WORLD
        
        domain = Cuboid()
        derham = []
        mass_ops = []
        A = []
        
        derham.append(Derham([Nel,1,1], [1,1,1], [True,True,True], comm=comm, local_projectors=False))
        mass_ops.append(WeightedMassOperators(derham[0], domain))
        if(matrix == "Poisson"):
            sp_key = '0'
            sp_id = 'H1'
            #Poisson
            A.append(derham[0].grad.T @ mass_ops[0].M1 @ derham[0].grad)
        #Hall-ish
        #A.append(epsilon*derham[0].curl.T @ mass_ops[0].M2 @ derham[0].curl + mass_ops[0].M1)
        elif(matrix == "Hall"):
            sp_key = '1'
            sp_id = 'Hcurl'
            #Hall
            A.append(mass_ops[0].M1 -epsilon*derham[0].curl.T @ mass_ops[0].M2 @ derham[0].curl)
        #Shear-Alfven-ish
        #A.append(mass_ops[0].M2 -epsilon* mass_ops[0].M2@ derham[0].curl @ mass_ops[0].M1 @ derham[0].curl.T @ mass_ops[0].M2)
        elif(matrix == 'Shear-Alfven' or matrix == 'Shear-Alfven-v2'):
            sp_key = '2'
            sp_id = 'Hdiv'
            pc_class = getattr(preconditioner,"MassMatrixPreconditioner")
            pc = pc_class(mass_ops[0].M1)
            M1_inv = inverse(
                mass_ops[0].M1,
                "pcg",
                pc=pc,
                maxiter=3000,
                verbose=False,
            )
            #Shear-Alfven
            if (matrix == "Shear-Alfven"):
                A.append(mass_ops[0].M2 -epsilon* mass_ops[0].M2@ derham[0].curl @ M1_inv @ derham[0].curl.T @ mass_ops[0].M2)

        Aarr = A[0].toarray()
        Aarrx = Aarr[0:Nel,0:Nel]
        Aarry = Aarr[Nel:2*Nel,Nel:2*Nel]
        eigenvaluesx = np.linalg.eigvals(Aarrx)
        max_x = abs(eigenvaluesx[np.argmax(np.abs(eigenvaluesx))])
        eigenvaluesy = np.linalg.eigvals(Aarry)
        
        plt.figure()
        plt.plot(jarray,eigenvaluesy)
        plt.show()
        
        max_y = abs(eigenvaluesy[np.argmax(np.abs(eigenvaluesy))])
        print(f'{max_x =}')
        
        
        
        
    
    
    
    #from struphy.feec.utilities import create_equal_random_arrays
    # get global communicator
    #comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()
    #world_size = comm.Get_size()
    
    #domain = Cuboid()
    #a1 = 0.002
    #domain = HollowCylinder(a1= a1)
    #sp_key = '0'
    
    #derham =Derham([Nel[0],Nel[1],Nel[2]], plist, spl_kind, comm=comm, local_projectors=False)
    #mass_ops=WeightedMassOperators(derham, domain)
    
    #u_stararr, u_star = create_equal_random_arrays(derham.Vh_fem[sp_key], seed=45)
    #u_stararr = remove_padding(derham.Vh_fem[sp_key], u_star)
    
    #G = derham.grad
    #Garr =G.toarray()
    
    #GT = derham.grad.T
    #GTarr = GT.toarray()
    
    #C = derham.curl
    #Carr = C.toarray()
    
    #M1 = mass_ops.M1
    #M1arr = M1.toarray()
    
    #Po = GT @ M1 @ G
    #Poarr = Po.toarray()
    
    
    #eigenvalues = np.linalg.eigvals(Poarr)
    #max_eigen_value = abs(eigenvalues[np.argmax(np.abs(eigenvalues))])
    #min_eigen_value = abs(eigenvalues[np.argmin(np.abs(eigenvalues))])
    #b = G.dot(u_star)
    #barr = np.matmul(Garr, u_stararr)
    
    #b = remove_padding(derham.Vh_fem['1'], b)
    
    #same = True
    #for i in range(np.size(b)):
        #if(barr[i]!= b[i]):
            #same = False
            #break
        
    #print(same)
    #print(f'{M1arr[32,32:64] =}')
    #print(f'{max_eigen_value =}')
    #print(f'{min_eigen_value =}')
    #for i in range(0,3):
        #print(f'{Garr[i,:] =}')
    
    
    
    

if __name__ == '__main__':
    from struphy.feec.utilities import create_equal_random_arrays
    comm = MPI.COMM_WORLD
    derham = Derham([10,1,1], [1,1,1], [True,True,True], comm=comm, local_projectors=False)
    u_stararr, u_star = create_equal_random_arrays(derham.Vh_fem['0'], seed=45)
    print("################")
    print(vars(u_star._space))
    print("################")
    #for i in range(7,8):
        #Nel = [int(2**i),int(2**i), 1]
        #p = [1, 1, 1]
        #spl_kind = [True, True, True]
        #Gather_data_V_cycle_parameter_study(Nel, p, spl_kind, i)
        #Visualized_high_frequency_dampening(Nel, p, spl_kind)
        #Visualized_all_frequencies_dampening(Nel, p, spl_kind,7)
        #Visualized_all_frequencies_dampening_2D(Nel, p, spl_kind,100)

    #testing_random_stuff([32,1,1],[1,1,1],[True,True,True])
    #128,64,32,16, 8
    # h, 2h,4h,8h,16h
    
    #p=1, Nel= 8192, level = 12. Coarsest one is 4x4 matrix
    #p=2, Nel= 8192, level = 11. Coarsest one is 8x8 matrix
    #p=4, Nel= 8192, level = 10. Coarsest one is 16x16 matrix
    
    #multigrid(Nel, p, spl_kind,6)
    #Gather_data_V_cycle_parameter_study(Nel, p, spl_kind, 11)
    #Gather_data_V_cycle_scalability([[int(2**i),1,1] for i in range(4,10)], p, spl_kind)
    #make_plot_scalability()
    #verify_formula(Nel, p, spl_kind)
    #verify_Restriction_Operator(Nel, p, spl_kind)
    #verify_Extension_Operator(Nel, p, spl_kind)
    #Error_analysis(Nel, p, spl_kind, 1)
    #trying_New_Restriction(Nel, p, spl_kind, 4)
    #Compute_rate_of_smoothing(Nel, p, spl_kind)
    #Visualized_high_frequency_dampening(Nel, p, spl_kind)
    