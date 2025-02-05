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
from psydac.linalg.basic  import VectorSpace, Vector, LinearOperator
from struphy.feec.linear_operators import LinOpWithTransp
from psydac.fem.basic import FemSpace
from psydac.fem.tensor import TensorFemSpace
from struphy.feec.mass import WeightedMassOperators
from struphy.geometry.domains import Tokamak, Cuboid
from struphy.fields_background.mhd_equil.equils import AdhocTorusQPsi
from math import comb, log2
import random


def jacobi(A, b, x_init=None, tol=1e-10, max_iter=1000, verbose = False):
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
        
        
        error = np.linalg.norm(x_new - x, ord=np.inf)
        if error < tol:
            converged = True
            if verbose:
                print(f'{converged = }')
                print(f'{itterations = }')
                print(f'{error = }')
            return x_new
        
        x[:] = x_new  # Update x
    
    
    if verbose:
        print(f'{converged = }')
        print(f'{itterations = }')
        print(f'{error = }')
    return x_new
    #raise ValueError("Jacobi method did not converge within the maximum number of iterations")


def direct_solver(A_inv,b, fem_space):
    # A_inv is already the inverse matrix of A
    #fem_space = derham.Vh_fem[sp_key]
    spaces = fem_space.spaces
    space = spaces[0]
    N = space.nbasis
    starts = np.array(fem_space.vector_space.starts)
    
    b_vector = remove_padding(fem_space, b)
    x_vector = np.dot(A_inv, b_vector)
    x = fem_space.vector_space.zeros()
    
    for i in range(N):
        x[starts[0]+i,0,0] = x_vector[i]
        
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
    spaces = fem_space.spaces
    space = spaces[0]
    N = space.nbasis
    
    starts = np.array(fem_space.vector_space.starts)
    
    #To make it easier to read I will extract the data out of out disregarding all the padding it come with
    v_array = np.zeros(N, dtype=float)
    for i in range(N):
        v_array[i] = v[starts[0]+i,0,0]
        
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
            # We get the start and endpoint for each sublist in out
            self._out_starts = np.array([vi.starts for vi in W.vector_space.spaces])
            self._out_ends = np.array([vi.ends for vi in W.vector_space.spaces])
            
        assert(self._VNbasis[0] == self._WNbasis[0]*2)
        
        # Degree of the B-spline space, not to be confused with the degrees given by fem_space.spaces.degree since depending on the situation 
        # it will give the D-spline degree instead
        self._p = get_b_spline_degree(V)
        
        #We also get the D-spline degree
        self._pD = self._p - 1
        
        #Now we compute the weights that define this linear operator
        #Here we store the weights needed for B-splines
        self._weights = np.zeros(self._p[0]+2, dtype=float)
        #Here we store the weights needed for D-splines
        self._weightsD = np.zeros(self._pD[0]+2, dtype=float)
        for j in range(self._p[0]+2):
            self._weights[j] = 2.0**(-self._p[0])*comb(self._p[0]+1,j)
        for j in range(self._pD[0]+2):
            self._weightsD[j] = 2.0**-(self._pD[0]+1)*comb(self._pD[0]+1,j)
        
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
    
    def dot_H1(self, v, out):
        p = self._p[0]
        for i in range(self._out_starts[0], self._out_ends[0]+1):
            for j in range(p+2):
                out[i,0,0] += self._weights[j]*v[(2*i-p+j)%self._VNbasis[0],0,0]
        return out
        
    def dot_L2(self, v, out):
        p = self._pD[0]
        for i in range(self._out_starts[0], self._out_ends[0]+1):
            for j in range(p+2):
                out[i,0,0] += self._weightsD[j]*v[(2*i-p+j)%self._VNbasis[0],0,0]
        return out
        
    def dot_Hcurl(self, v, out):
        for h in range(3):
            if h == 0:
                p = self._pD[0]
                weights = self._weightsD
            else:
                p = self._p[0]
                weights = self._weights
            for i in range(self._out_starts[h][0], self._out_ends[h][0]+1):
                for j in range(p+2):
                    out[h][i,0,0] += weights[j]*v[(2*i-p+j)%self._VNbasis[0],0,0]           
        return out
        
    def dot_Hdiv(self, v, out):
        for h in range(3):
            if h == 0:
                p = self._p[0]
                weights = self._weights
            else:
                p = self._pD[0]
                weights = self._weightsD
            for i in range(self._out_starts[h][0], self._out_ends[h][0]+1):
                for j in range(p+2):
                    out[h][i,0,0] += weights[j]*v[(2*i-p+j)%self._VNbasis[0],0,0]           
        return out
        
    def dot_H1H1H1(self, v, out):
        p = self._p[0]
        weights = self._weights
        for h in range(3): 
            for i in range(self._out_starts[h][0], self._out_ends[h][0]+1):
                for j in range(p+2):
                    out[h][i,0,0] += weights[j]*v[(2*i-p+j)%self._VNbasis[0],0,0]           
        return out
    
    
    def dot(self, v, out=None):

        assert isinstance(v, Vector) and v.space == self.domain
 
        if out is None:
            out = self.codomain.zeros()   
        else:
            assert isinstance(out, Vector) and out.space == self.codomain
            
            for i in range(self._out_starts[0], self._out_ends[0]+1):
                out[i,0,0] = 0.0
        
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
            # We get the start and endpoint for each sublist in out
            self._out_starts = np.array([vi.starts for vi in W.vector_space.spaces])
            self._out_ends = np.array([vi.ends for vi in W.vector_space.spaces])
            
        assert(self._WNbasis[0] == self._VNbasis[0]*2)
        
        # Degree of the B-spline space, not to be confused with the degrees given by fem_space.spaces.degree since depending on the situation 
        # it will give the D-spline degree instead
        self._p = get_b_spline_degree(V)
        
        #Now we compute the weights that define this linear operator
        if(self._p[0]%2 == 0):
            self._size_even = self._p[0]//2 +1
            self._size_odd = self._p[0]//2 +1
            self._weights_even = np.zeros(self._size_even, dtype=float)
            self._weights_odd = np.zeros(self._size_odd, dtype=float)
            for j in range(self._size_even):
                self._weights_even[j] = 2.0**(-self._p[0])*comb(self._p[0]+1,2*j)
                self._weights_odd[j] = 2.0**(-self._p[0])*comb(self._p[0]+1,2*j+1)
        else:
            self._size_even = (self._p[0]+1)//2 +1
            self._size_odd = (self._p[0]-1)//2 +1
            self._weights_even = np.zeros(self._size_even, dtype=float)
            self._weights_odd = np.zeros(self._size_odd, dtype=float)
            for j in range(self._size_even):
                self._weights_even[j] = 2.0**(-self._p[0])*comb(self._p[0]+1,2*j)
            for j in range(self._size_odd):
                self._weights_odd[j] = 2.0**(-self._p[0])*comb(self._p[0]+1,2*j+1)
           
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

    def dot(self, v, out=None):

        assert isinstance(v, Vector)
        assert v.space == self.domain

        p = self._p[0]
        if out is None:
            out = self.codomain.zeros()
            
        else:
            assert isinstance(out, Vector)
            assert out.space == self.codomain
            
            for j in range(self._out_starts[0], self._out_ends[0] + 1):
                out[j, 0, 0] = 0.0
                          
        parity_match = p % 2
        for j in range(self._out_starts[0], self._out_ends[0] + 1):
            parity_j = j % 2
            weights, size, offset = ((self._weights_even, self._size_even, 0) if parity_j == parity_match else (self._weights_odd, self._size_odd, 1))
            for i in range(size):
                out[j, 0, 0] += weights[i] * v[((j + p - 2 * i - offset) // 2) % self._VNbasis[0], 0, 0]
            
        return out
    
    def transpose(self, *, out = None):
        if out is None:
            out = RestrictionOperator(self._W, self._V)
        else:
            assert isinstance(out, RestrictionOperator)
            assert out.domain is self.codomain
            assert out.codomain is self.domain
            
        return out
        


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
    
    
    #domain = Tokamak(p = [3,3], psi_shifts = [2.,2.])
    domain = Cuboid()
    
    sp_key = '0'
    
    #mhd_equil = AdhocTorusQPsi()
    # must set domain of Cartesian MHD equilibirum
    #mhd_equil.domain = domain
    
    
    derham = []
    mass_ops = []
    A = []
    
    
    for level in range(N_levels):
    
        derham.append(Derham([Nel[0]//(2**level),Nel[1],Nel[2]], plist, spl_kind, comm=comm, local_projectors=False))
        mass_ops.append(WeightedMassOperators(derham[level], domain))
        A.append(derham[level].grad.T @ mass_ops[level].M1 @ derham[level].grad)
        #print(f'{level = }')
        #print(f'{ Nel[0]//(2**level)= }')
        #A.append(mass_ops[level].M0.toarray_struphy())
    
    A_inv = np.linalg.inv(A[-1].toarray())
    R = []
    E = []
    
    for level in range(N_levels-1):
        R.append(RestrictionOperator(derham[level].Vh_fem[sp_key],derham[level+1].Vh_fem[sp_key]))
        E.append(R[level].transpose())
        #R.append(RestrictionOperator(derham[level].Vh_fem[sp_key],derham[level+1].Vh_fem[sp_key]).toarray_struphy())
        #E.append(ExtensionOperator(derham[level+1].Vh_fem[sp_key],derham[level].Vh_fem[sp_key]).toarray_struphy())
    
    
    
    #_B = -1 / 2 * _T.T @ derham.curl.T @ mass_ops.M2
    #_C = 1 / 2 * derham.curl @ _T
    #_BC = _B @ _C

    Multigrid = 'V-cycle'
    method = 'cg'
    max_iter = 5
    N_cycles = 6
    
    if(Multigrid == 'V-cycle'):
        
        
        #barr, b = create_equal_random_arrays(derham[0].Vh_fem[sp_key], seed=4568)
        #barr = barr[0].flatten()
        u_stararr, u_star = create_equal_random_arrays(derham[0].Vh_fem[sp_key], seed=45)
        #We compute the rhs
        b = A[0].dot(u_star)
        
        timei = time.time()
        
        solver_no = inverse(A[0],method, maxiter = 10000)
        
        #barr, b = create_equal_random_arrays(derham[level].Vh_fem[sp_key], seed=4568)
        #barr = remove_padding(derham[level].Vh_fem[sp_key], b)
        #barr = barr[level].flatten()
        
        
        
        u = solver_no.dot(b)
        
        timef = time.time()
        
        #We get the total number of itterations
        No_Multigrid_itterations = solver_no._info['niter']
        No_Multigrid_error = solver_no._info['res_norm']
        No_Multigrid_time = timef-timei
        
        #We define a list where to store the number of itteration it takes at each multigrid level
        Multigrid_itterations = np.zeros(N_levels, dtype=int)
        
        
        def V_cycle(l, r_l):
            if (l < N_levels-1):
                solver_ini = inverse(A[l],method, maxiter= max_iter*2**(l))
                x_l = solver_ini.dot(r_l)
                
                #We count the number of itterations
                Multigrid_itterations[l] += solver_ini._info['niter']
                
                #print(f'{l = }')
                #print(f'{solver_ini._info = }')
                
                r_l = r_l - A[l].dot(x_l)
                
                r_l_plus_1 = R[l].dot(r_l)
                x_l_plus_1 = V_cycle(l+1, r_l_plus_1)
                x_l = x_l + E[l].dot(x_l_plus_1)
                
                
                #r_l = r_l - A[l].dot(E[l].dot(x_l_plus_1))
                #solver = inverse(A[l].T,method, maxiter= max_iter*2**(l), x0 = x_l)
                #x_l = solver.dot(r_l)  
                
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
            
            r_0 = b - A[0].dot(x_0)
            r_1 = R[0].dot(r_0)
            
            x_0 = x_0 + E[0].dot(V_cycle(1,r_1))
        
        solver = inverse(A[0],method,x0 = x_0)
        x = solver.dot(b)
        
        timef = time.time()
        
        Multigrid_itterations[0] += solver._info['niter']
        
        #We get the final error
        Multigrid_error = solver._info['res_norm']
        
        Multigrid_time = timef- timei
        
        speed_up = No_Multigrid_time / Multigrid_time
        
        print("################")
        print(f'{No_Multigrid_itterations = }')
        print(f'{No_Multigrid_error = }')
        print(f'{No_Multigrid_time = }')
        print("################")
        print("################")
        print(f'{Multigrid_itterations = }')
        print(f'{Multigrid_error = }')
        print(f'{Multigrid_time = }')
        print("################")
        print("################")
        print(f'{speed_up = }')
        
        #print(f'{solver._info = }')
        #print(f'{answer._data = }')
        #print(f'{b._data = }')
        
        #solver = inverse(A,'cg', maxiter= max_iter)
        #u = solver.dot(b)
        
        #r = b - A.dot(u)
        
        #r_2 = R.dot(r)
        
        #solver_2 = inverse(A_2, 'cg')
        #e_2 = solver_2.dot(r_2)
        
        #e = E.dot(e_2)
        
        #u = u + e
        
        #solver = inverse(A,'cg', x0 = u)
        
        #u = solver.dot(b)
        
        #print(f'{solver._info = }')
        #print(f'{solver_2._info = }')
        
    elif((Multigrid == 'v-cycle')):
        
        level = 0   
        
        solver_no = inverse(A[level],'cg')
        
        
        u_stararr, u_star = create_equal_random_arrays(derham[level].Vh_fem[sp_key], seed=45)
        #We compute the rhs
        b = A[level].dot(u_star)
        
        #barr, b = create_equal_random_arrays(derham[level].Vh_fem[sp_key], seed=4568)
        #barr = remove_padding(derham[level].Vh_fem[sp_key], b)
        #barr = barr[level].flatten()
        
        u = solver_no.dot(b)
        
        #answer = A[level].dot(u)
        print(f'{ solver_no._info = }')
        #print(f'{answer._data =}')
        #print(f'{b._data =}')
        
        
        fieldr = derham[level].create_field('fh', 'H1')
        fielde = derham[level+1].create_field('fh', 'H1')
        use_projectors = False
        
        
        #N_cycles = 5
        
        u = derham[level].Vh_fem[sp_key].vector_space.zeros()
        #uarr = remove_padding(derham[level].Vh_fem[sp_key], u)
        
        for i in range(N_cycles):
            #print(f'{i =}')
        
        

            solver = inverse(A[level],'cg', x0 = u, maxiter= 5)
            u = solver.dot(b)
            #uarr = jacobi(A[level], barr, x_init=uarr, max_iter=5)
            
            
            
            
        
            #print(f'{solver._info =}')
            #b_arr = remove_padding(derham[level].Vh_fem[sp_key], b)
            #print( f'{ b_arr = }')
            
            
            
            
            r = b - A[level].dot(u)
            #rarr = barr - np.matmul(A[level],uarr)
            
            
            
            
            #r_arr = remove_padding(derham[level].Vh_fem[sp_key], r)
            #print( f'{ r_arr = }')
            
            
            
            if use_projectors:
                fieldr.vector = r
                r_2 = derham[level+1].P[sp_key](fieldr)            
            else:
                r_2 = R[level].dot(r)
                
            #r_2arr = np.matmul(R[level],rarr)   
            
                
            
            #r_2arr = remove_padding(derham[level+1].Vh_fem[sp_key], r_2)
            #print(f'{ r_2arr =}')
            
            
            
            
            
            solver_2 = inverse(A[level+1], 'cg')
            e_2 = solver_2.dot(r_2)
            
            #e_2arr = jacobi(A[level+1], r_2arr, max_iter=1000)
            #e_2arr = np.linalg.solve(A[level+1], r_2arr)
            #print(f'{e_2arr =}')
            
            
            
            
            #e_2arr = remove_padding(derham[level+1].Vh_fem[sp_key], e_2)
            #print(f'{e_2arr = }')
            
            
            
            if use_projectors:
                fielde.vector = e_2
                e = derham[level].P[sp_key](fielde)
            else:
                e = E[level].dot(e_2)
            #e_arr = np.matmul(E[level],e_2arr)
            
            
            
            
            #e_arr = remove_padding(derham[level].Vh_fem[sp_key], e)
            
            #print(f'{e_arr = }')
            
            
            #u_arr = remove_padding(derham[level].Vh_fem[sp_key], u)
            #print(f'{u_arr = }')
            
            
            
            
            
            u = u + e
            #uarr = uarr + e_arr
            
            
            
            #u_arr = remove_padding(derham[level].Vh_fem[sp_key], u)
            #print(f'{u_arr = }')
        
        
        
        solver_final = inverse(A[level],'cg', x0 = u)
        u = solver_final.dot(b)
        #uarr = jacobi(A[level],barr, x_init=uarr, verbose=True)
        
        
        
        print(f'{solver_final._info = }')
        print(f'{solver_2._info = }')
    
    elif(Multigrid == 'Full'):
        #We will use the method of the manufactured solutions
        #u_star is the solution of the system
        u_stararr, u_star = create_equal_random_arrays(derham[0].Vh_fem[sp_key], seed=45)
        #For this method we need to compute a projection of the rhs for each level
        b = []
        for level in range(N_levels):
            if level == 0:
                b.append(A[0].dot(u_star))
            else:
                b.append(R[level-1].dot(b[-1]))
         
        #We compute the number of iterations it tkaes to solve the system without multigrid.
        solver_no = inverse(A[0],method)
        u = solver_no.dot(b[0])
        print(f'{ solver_no._info = }')
        
        def V_cycle(l, r_l):
            if (l < N_levels-1):
                solver_ini = inverse(A[l],method, maxiter= max_iter*2**(l))
                x_l = solver_ini.dot(r_l)
                
                r_l = r_l - A[l].dot(x_l)
                
                r_l_plus_1 = R[l].dot(r_l)
                x_l_plus_1 = V_cycle(l+1, r_l_plus_1)
                x_l = x_l + E[l].dot(x_l_plus_1)
            else:
                #Solve directly
                x_l = direct_solver(A_inv,r_l, derham[l].Vh_fem[sp_key])
               
            return x_l 
        
        def Full():
            #l determines our current level
            l = N_levels-1
            #First we solve the system directly in the coarsest level
            #If the coarsest level is still to coarse to solve directly then you can solve it itteratively instead.
            x_l = direct_solver(A_inv,b[l], derham[l].Vh_fem[sp_key])
            
            while(l > 0):
                #We extend the solution of the coarser level to the finner level to use as a first guess for the itterative solver
                x_l = E[l-1].dot(x_l)
                
                for cycle in range(N_cycles):
                
                    #We smooth the error on the finner grid
                    solver = inverse(A[l-1],method, maxiter= max_iter*2**(l-1), x0 = x_l)
                    x_l  = solver.dot(b[l-1])
                    r_l_minus_1 = b[l-1] - A[l-1].dot(x_l)
                    r_l = R[l-1].dot(r_l_minus_1)
                    
                    x_l = x_l + E[l-1].dot(V_cycle(l,r_l))
            
                l= l -1
            
            return x_l
        
        x_0 = Full()
        solver = inverse(A[0],method,x0 = x_0)
        x = solver.dot(b[0])
        
        print(f'{solver._info = }')
          
        
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
    sp_key = '0'
    
    derham = []
    mass_ops = []
    A = []
    
    for level in range(N_levels):
    
        derham.append(Derham([Nel[0]//(2**level),Nel[1],Nel[2]], plist, spl_kind, comm=comm, local_projectors=False))
        mass_ops.append(WeightedMassOperators(derham[level], domain))
        A.append(derham[level].grad.T @ mass_ops[level].M1 @ derham[level].grad)
    
    #We get the inverse of the coarsest system matrix to solve directly the problem in the smaller space
    A_inv = np.linalg.inv(A[-1].toarray())
    
    R = []
    E = []
    
    for level in range(N_levels-1):
        R.append(RestrictionOperator(derham[level].Vh_fem[sp_key],derham[level+1].Vh_fem[sp_key]))
        E.append(R[level].transpose())
        
    method = 'cg'
    
    max_iter_list = [7,8,9]
    N_cycles_list = [2,3,4,5,6,7]
    
    u_stararr, u_star = create_equal_random_arrays(derham[0].Vh_fem[sp_key], seed=45)
    #We compute the rhs
    b = A[0].dot(u_star)
    
    timei = time.time()
    
    solver_no = inverse(A[0],method, maxiter = 10000, tol = 10**(-6))
    
    u = solver_no.dot(b)
    
    timef = time.time()
    
    #We get the total number of itterations
    No_Multigrid_itterations = solver_no._info['niter']
    No_Multigrid_error = solver_no._info['res_norm']
    No_Multigrid_time = timef-timei
    
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
        Multigrid_itterations = np.zeros(N_levels, dtype=int)
        
        
        def V_cycle(l, r_l):
            if (l < N_levels-1):
                solver_ini = inverse(A[l],method, maxiter= max_iter)
                x_l = solver_ini.dot(r_l)
                
                #We count the number of itterations
                Multigrid_itterations[l] += solver_ini._info['niter']
                
                r_l = r_l - A[l].dot(x_l)
                
                r_l_plus_1 = R[l].dot(r_l)
                x_l_plus_1 = V_cycle(l+1, r_l_plus_1)
                x_l = x_l + E[l].dot(x_l_plus_1)
                
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
            
            r_0 = b - A[0].dot(x_0)
            r_1 = R[0].dot(r_0)
            
            x_0 = x_0 + E[0].dot(V_cycle(1,r_1))
        
        solver = inverse(A[0],method,x0 = x_0, tol = 10**(-6))
        x = solver.dot(b)
        
        timef = time.time()
        
        Multigrid_itterations[0] += solver._info['niter']
        
        #We get the final error
        Multigrid_error = solver._info['res_norm']
        
        Multigrid_time = timef- timei
        
        speed_up = No_Multigrid_time / Multigrid_time
        #speed_up = 27.3452200889587 / Multigrid_time
        

        
        print("################")
        print("################")
        print("################")
        print(f'{Nel[0] = }')
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
    Nel = [int(2**i) for i in range(4,14)]
    Multi_time = [0.0167179107666016, 0.0275583267211914, 0.0449323654174805, 0.0735518932342529, 0.188658952713013, 0.173357963562012, 0.291205167770386, 0.464969396591187, 0.960138320922852, 2.02050709724426]    
    No_Multi_time = [0.00164151191711426, 0.00258350372314453, 0.00557708740234375, 0.0137369632720947, 0.0344779491424561, 0.101414680480957, 0.479604005813599, 1.46064519882202, 7.63734912872314, 33.2413830757141]
    speed_up = [0.098188819167142, 0.0937467557185867, 0.124121829565956, 0.186765597295291, 0.182752785630607, 0.585001567837869, 1.64696255044404, 3.14137921663317, 7.95442590124139, 16.4520001543432]
    
    plt.figure()
    plt.plot(Nel, Multi_time, label = 'Multigrid solver run time.')
    plt.scatter(Nel, No_Multi_time, label = 'CG run time.')
    plt.xlabel("Nel[0]")
    plt.ylabel('Time (s)')
    plt.legend()
    plt.show()
    plt.close()
    
    plt.figure()
    plt.scatter(Nel, speed_up, label = 'Speed_up.')
    plt.xlabel("Nel[0]")
    plt.ylabel('Speed_up')
    plt.legend()
    plt.show()
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
     
            

    
    
    
    

if __name__ == '__main__':
    Nel = [8192, 1, 1]
    p = [1, 1, 1]
    spl_kind = [True, True, True]

    #128,64,32,16, 8
    # h, 2h,4h,8h,16h
    
    #p=1, Nel= 8192, level = 12. Coarsest one is 4x4 matrix
    #p=2, Nel= 8192, level = 11. Coarsest one is 8x8 matrix
    #p=4, Nel= 8192, level = 10. Coarsest one is 16x16 matrix
    
    #multigrid(Nel, p, spl_kind,12)
    Gather_data_V_cycle_parameter_study(Nel, p, spl_kind, 12)
    #Gather_data_V_cycle_scalability([[int(2**i),1,1] for i in range(4,10)], p, spl_kind)
    #make_plot_scalability()
    #verify_formula(Nel, p, spl_kind)
    #verify_Restriction_Operator(Nel, p, spl_kind)
    #verify_Extension_Operator(Nel, p, spl_kind)
    #Error_analysis(Nel, p, spl_kind, 1)