import time

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import re
from struphy.feec.psydac_derham import Derham

from psydac.linalg.solvers import inverse
from psydac.linalg.basic  import Vector
from psydac.fem.projectors import knot_insertion_projection_operator
from struphy.feec.linear_operators import LinOpWithTransp
from psydac.fem.basic import FemSpace
from psydac.fem.tensor import TensorFemSpace
from struphy.feec.mass import WeightedMassOperators
from struphy.geometry.domains import Cuboid
from math import comb
from struphy.feec.utilities import create_equal_random_arrays

def build_operator(expr: str, namespace: dict):
    """
    Build a composite LinOpWithTransp (or scalar * LinOpWithTransp expression) 
    from a string expression.

    Parameters
    ----------
    expr : str
        String expression defining the operator composition. 
        Example:
            "epsilon * curl.T @ M2 @ curl + mu * M1"
        
        The expression can contain:
            - scalars (e.g., epsilon, mu, ...)
            - LinOpWithTransp objects (already supporting +, -, *, @)
            - parentheses to control precedence

    namespace : dict
        Dictionary mapping symbol names in `expr` to actual Python objects 
        (scalars, operators, lists, etc.).
        Example:
            {
                "epsilon": 1.3,
                "mu": 0.7,
                "curl.T": derham.curl.T,
                "curl": derham.curl,
                "M1": WeightedMassOperators(derham, domain).M1,
                "M2": WeightedMassOperators(derham, domain).M2,
            }

    Returns
    -------
    LinOpWithTransp
        A composite operator built according to the expression. The returned 
        object supports `.dot(v)` and any other functionality of LinOpWithTransp.

    Raises
    ------
    ValueError
        If some symbols in the expression are not found in `namespace`.
    """

    # Find all potential variable-like tokens (ignore numbers)
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", expr))
    
    missing = tokens - set(namespace.keys())

    if missing:
        raise ValueError(
            f"The following symbols are missing from the namespace: {', '.join(sorted(missing))}"
        )

    # Let Python evaluate the expression with operator overloads
    return eval(expr, {}, namespace)

def remove_padding(fem_space, v):
    
    #fem_space = derham.Vh_fem[sp_key]
    symbolic_name = fem_space.symbolic_space
    
    if(symbolic_name == 'H1' or symbolic_name == "L2"):
    
        spaces = [fem_space.spaces]
        N = [spaces[0][i].nbasis for i in range(3)]
        
        starts = np.array(fem_space.coeff_space.starts)
        
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
        
        starts = np.array([vi.starts for vi in fem_space.coeff_space.spaces])
        
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


def direct_solver(A_inv,b, fem_space):
    # A_inv is already the inverse matrix of A
    #fem_space = derham.Vh_fem[sp_key]
    symbolic_name = fem_space.symbolic_space
    
    if(symbolic_name == 'H1' or symbolic_name == "L2"):
        spaces = [fem_space.spaces]
        N = [spaces[0][i].nbasis for i in range(3)]
        starts = np.array(fem_space.coeff_space.starts)
        
        b_vector = remove_padding(fem_space, b)
        x_vector = np.dot(A_inv, b_vector)
        x = fem_space.coeff_space.zeros()
        
        cont= 0
        for i0 in range(N[0]):
            for i1 in range(N[1]):
                for i2 in range(N[2]):
                    x[starts[0]+i0,starts[1]+i1,starts[2]+i2] = x_vector[cont]
                    cont += 1
            
    else:
        spaces = [comp.spaces for comp in fem_space.spaces]
        N = [[spaces[h][i].nbasis for i in range(3)] for h in range(3)]
        starts = np.array([vi.starts for vi in fem_space.coeff_space.spaces])
        
        b_vector = remove_padding(fem_space, b)
        x_vector = np.dot(A_inv, b_vector)
        x = fem_space.coeff_space.zeros()
        
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
    
    if hasattr(V, "symbolic_space"):
        V_name = V.symbolic_space
    
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



class MultiGridSolver:

    def __init__(self, derham, sp_key, N_levels, domain, expr: str, namespace: dict, max_iter_list, N_cycles, method='cg'):
        """
        
        Initialize the multigrid solver with the given parameters.
        Parameters
        ----------
        derham : Derham
            The Derham object representing the finest grid.
        sp_key : str
            Symbolic space key, e.g., "0", "1", "2", "3".
        N_levels : int
            Number of multigrid levels.
        domain : struphy.geometry.base.Domain
            The Geometric domain of the simulation.
            
        expr : str
            String expression defining the operator composition. 
            Example:
                "epsilon * curl.T @ M2 @ curl + mu * M1"
            
            The expression can contain:
                - scalars (e.g., epsilon, mu, ...)
                - LinOpWithTransp objects (already supporting +, -, *, @)
                - parentheses to control precedence

        namespace : dict
            A dictionary that serves as a **string template** or "recipe book" for
            building the operators and scalars used in the `expr` string.

            It maps symbol names to **strings of Python code** that describe how to
            create the corresponding object. This allows the solver to dynamically
            generate the correct operators for each grid level.

            The string recipes can use variables like `derham`, `WeightedMassOperators` and `domain`, which
            the solver makes available during the object creation at each level.

            Example:
                {
                    "epsilon": "1.3",
                    "mu": "0.7",
                    "curl.T": "derham.curl.T",
                    "curl": "derham.curl",
                    "M1": "WeightedMassOperators(derham, domain).M1",
                    "M2": "WeightedMassOperators(derham, domain).M2",
                }
        max_iter_list : list
            List of maximum iterations for each multigrid level. The first element corresponds to the finest level.
        N_cycles : int
            Number of multigrid cycles.
        method : str
            Solution method to use (e.g., 'cg' for conjugate gradient).

        """
        self.sp_key = sp_key
        self.Nel = derham.Nel
        self.plist = derham.p
        self.spl_kind = derham._spl_kind
        self.N_levels = N_levels
        self.domain = domain
        self.expr = expr
        self.namespace = namespace
        self.derham = derham
        self.method = method
        self.max_iter_list = max_iter_list
        self.N_cycles = N_cycles

        # get global communicator
        self.comm = MPI.COMM_WORLD
        self.derhamlist = []
        self.A = []
        
        for level in range(self.N_levels):
            if level == 0:
                self.derhamlist.append(self.derham)
            else:
                self.derhamlist.append(Derham([self.Nel[0]//(2**level),self.Nel[1]//(2**level),self.Nel[2]], self.plist, self.spl_kind, comm=self.comm, local_projectors=False))
            #It might look like we are not using derham but it is being used by updated_namespace 
            #for rebounding the expressions in self.expr to the correct derham level.
            derhamaux = self.derhamlist[level]

            # 1. Create a local context for the eval. It maps string names
            #    to the actual objects for the CURRENT level.
            eval_context = {
                "derham": derhamaux,
                "domain": self.domain,
                "WeightedMassOperators": WeightedMassOperators 
            }

            # 2. Build the namespace by evaluating each string from the template
            #    within the context of the current level.
            updated_namespace = {
                key: eval(recipe_string, {}, eval_context)
                for key, recipe_string in self.namespace.items()
            }
            
            self.A.append(build_operator(self.expr, updated_namespace))
            
        #We get the inverse of the coarsest system matrix to solve directly the problem in the smaller space
        self.A_inv = np.linalg.inv(self.A[-1].toarray())
        
        self.R = []
        self.E = []
        
        for level in range(self.N_levels-1):
            self.R.append(self.RestrictionOperator(self.derhamlist[level].Vh_fem[self.sp_key],self.derhamlist[level+1].Vh_fem[self.sp_key]))
            self.E.append(self.R[level].transpose())
            
        
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
            
            #print(vars(V))
            
            # Store info in object
            self._domain   = V.coeff_space
            self._codomain = W.coeff_space
            self._dtype = V.coeff_space.dtype
            
            #Can be "H1", "L2", "Hcurl", "Hdiv", "H1H1H1"
            self._V_name = V.symbolic_space
            self._W_name = W.symbolic_space
            assert(self._V_name == self._W_name)
            
            #This list will tell us in which spatial direction we are halving the problem.
            self._halving_directions = [False,False,False]
            
            # input space: 3d StencilVectorSpaces and 1d SplineSpaces of each component
            if isinstance(V, TensorFemSpace):
                self._V1ds = [V.spaces]
                self._VNbasis = np.array([self._V1ds[0][0].nbasis, self._V1ds[0][1].nbasis, self._V1ds[0][2].nbasis])
                
                # We get the start and endpoint for each sublist in input
                self._in_starts = np.array(V.coeff_space.starts)
                self._in_ends = np.array(V.coeff_space.ends)
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
                self._in_starts = np.array([vi.starts for vi in V.coeff_space.spaces])
                self._in_ends = np.array([vi.ends for vi in V.coeff_space.spaces])
                
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
                self._out_starts = np.array(W.coeff_space.starts)
                self._out_ends = np.array(W.coeff_space.ends)
                
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
                self._out_starts = np.array([vi.starts for vi in W.coeff_space.spaces])
                self._out_ends = np.array([vi.ends for vi in W.coeff_space.spaces])
                
            
            
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
                out = MultiGridSolver.ExtensionOperator(self._W, self._V)
            else:
                assert isinstance(out, MultiGridSolver.ExtensionOperator)
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
            self._domain   = V.coeff_space
            self._codomain = W.coeff_space
            self._dtype = V.coeff_space.dtype
            
            #Can be "H1", "L2", "Hcurl", "Hdiv", "H1H1H1"
            self._V_name = V.symbolic_space
            self._W_name = W.symbolic_space
            assert(self._V_name == self._W_name)
            
            #This list will tell us in which spatial direction we are halving the problem.
            self._halving_directions = [False,False,False]
            
            # input space: 3d StencilVectorSpaces and 1d SplineSpaces of each component
            if isinstance(V, TensorFemSpace):
                self._V1ds = [V.spaces]
                self._VNbasis = np.array([self._V1ds[0][0].nbasis, self._V1ds[0][1].nbasis, self._V1ds[0][2].nbasis])
                
                # We get the start and endpoint for each sublist in input
                self._in_starts = np.array(V.coeff_space.starts)
                self._in_ends = np.array(V.coeff_space.ends)
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
                self._in_starts = np.array([vi.starts for vi in V.coeff_space.spaces])
                self._in_ends = np.array([vi.ends for vi in V.coeff_space.spaces])
                
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
                self._out_starts = np.array(W.coeff_space.starts)
                self._out_ends = np.array(W.coeff_space.ends)
                
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
                self._out_starts = np.array([vi.starts for vi in W.coeff_space.spaces])
                self._out_ends = np.array([vi.ends for vi in W.coeff_space.spaces])
                
            
            
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
                out = MultiGridSolver.RestrictionOperator(self._W, self._V)
            else:
                assert isinstance(out, MultiGridSolver.RestrictionOperator)
                assert out.domain is self.codomain
                assert out.codomain is self.domain
                
            return out


    def solve(self, b, verbose=False): 
        max_iter = self.max_iter_list
        N_cycles = self.N_cycles
        
        #We define a list where to store the number of itteration it takes at each multigrid level
        #Change N_levels for 1D case
        Multigrid_itterations = np.zeros(self.N_levels, dtype=int)
        converged = np.ones(self.N_levels, dtype=bool)
        
        def V_cycle(l, r_l):
            #Change for N_levels-1 for 1D case
            if (l < self.N_levels-1):
                solver_ini = inverse(self.A[l],self.method, maxiter= max_iter[l])
                x_l = solver_ini.dot(r_l)
                
                #We count the number of itterations
                Multigrid_itterations[l] += solver_ini._info['niter']
                
                #We determine if the itterative solver converged in the maximum number of itterations
                converged[l] = solver_ini._info['success']
                if converged[l] == True:
                    return x_l
                
                r_l = r_l - self.A[l].dot(x_l)
                
                r_l_plus_1 = self.R[l].dot(r_l)
                x_l_plus_1 = V_cycle(l+1, r_l_plus_1)
                #New
                #x_l_aux = self.E[l].dot(x_l_plus_1)
                #x_l = x_l + x_l_aux
                #r_l = r_l  - self.A[l].dot(x_l_aux)
                #solver_end = inverse(self.A[l].T,self.method, maxiter= max_iter, x0 =x_l)
                #x_l = solver_end.dot(r_l)
                #Multigrid_itterations[l] += solver_end._info['niter']
                ####
                #old
                x_l = x_l + self.E[l].dot(x_l_plus_1)
                ###
                
            else:
                #Solve directly
                x_l = direct_solver(self.A_inv,r_l, self.derhamlist[l].Vh_fem[self.sp_key])  
            return x_l 
        
        timei = time.time()
        for cycle in range(N_cycles):
            if cycle == 0:
                solver = inverse(self.A[0],self.method, maxiter= max_iter[0], tol=1e-8)
            else:
                solver = inverse(self.A[0],self.method, maxiter= max_iter[0], x0 = x_0, tol=1e-8)
            x_0 = solver.dot(b)

            Multigrid_itterations[0] += solver._info['niter']
            
            #We determine if the itterative solver converged in the maximum number of itterations
            converged[0] = solver._info['success']
            if converged[0] == True:
                x = x_0
                break
            
            r_0 = b - self.A[0].dot(x_0)
            r_1 = self.R[0].dot(r_0)
            
            x_0 = x_0 + self.E[0].dot(V_cycle(1,r_1))
        
        if converged[0] == False:
            solver = inverse(self.A[0],self.method,x0 = x_0, tol = 10**(-10))
            x = solver.dot(b)
            Multigrid_itterations[0] += solver._info['niter']
        
        timef = time.time()
        
        #We get the final error
        Multigrid_error = solver._info['res_norm']
        
        Multigrid_time = timef- timei
        
        if verbose:
            print("################")
            print("################")
            print("################")
            print(f'{max_iter = }')
            print(f'{N_cycles = }')
            print("################")
            print("################")
            print(f'{Multigrid_itterations = }')
            print(f'{Multigrid_error = }')
            print(f'{Multigrid_time = }')
            print(f'{converged = }')
            print("################")
            print("################")  
        return x 
     
       
class MultiGridPoissonSolver(MultiGridSolver):
    def __init__(self, derham: Derham, sp_key: str, N_levels: int, domain, max_iter_list: list, N_cycles: int, method: str = 'cg'):
        expr = " grad.T @ M1 @ grad"
        namespace = {"grad.T": "derham.grad.T",
        "grad":   "derham.grad",
        "M1":     "WeightedMassOperators(derham, domain).M1"}
        super().__init__(derham, sp_key, N_levels, domain, expr, namespace, max_iter_list, N_cycles, method)
    
    
    
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    Nel = [16,16,1]
    sp_key = '0'
    plist = [2,2,1]
    spl_kind = [True,True,True]
    N_levels = 3
    domain = Cuboid()
    derham = Derham([Nel[0],Nel[1],Nel[2]], plist, spl_kind, comm=comm, local_projectors=False)
    max_iter = [10, 25, 25]
    N_cycles = 5
    method = 'cg'
    
    expr = " grad.T @ M1 @ grad"
    namespace = {"grad.T": "derham.grad.T",
    "grad":   "derham.grad",
    "M1":     "WeightedMassOperators(derham, domain).M1"}
    
    
     # 1. Create a local context for the eval. It maps string names
    #    to the actual objects for the CURRENT level.
    eval_context = {
        "derham": derham,
        "domain": domain,
        "WeightedMassOperators": WeightedMassOperators 
    }

    # 2. Build the namespace by evaluating each string from the template
    #    within the context of the current level.
    updated_namespace = {
        key: eval(recipe_string, {}, eval_context)
        for key, recipe_string in namespace.items()
    }

    A = build_operator(expr, updated_namespace)

    multigrid = MultiGridPoissonSolver(derham, sp_key, N_levels, domain, max_iter, N_cycles, method)
    
    u_stararr, u_star = create_equal_random_arrays(derham.Vh_fem[sp_key], seed=8765)
    #We compute the rhs
    b = A.dot(u_star)
    b_arr = b.toarray()
    u = multigrid.solve(b, verbose=True) 
    b_ans_arr = A.dot(u).toarray()
    if np.allclose(b_ans_arr, b_arr, atol=1e-6):
        print("The multigrid solver computed the correct solution.") 
    else:
        print("The multigrid solver did not compute the correct solution.") 
        print(f"{b_ans_arr = }")
        print(f"{b_arr = }")