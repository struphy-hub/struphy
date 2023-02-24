from psydac.linalg.basic import Vector, LinearSolver
from psydac.linalg.direct_solvers import DirectSolver, SparseSolver
from psydac.linalg.stencil import StencilMatrix, StencilVector, StencilVectorSpace
from psydac.linalg.block import BlockLinearOperator, BlockDiagonalSolver
from psydac.linalg.kron import KroneckerLinearSolver, KroneckerStencilMatrix

from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace

from psydac.ddm.cart import DomainDecomposition, CartDecomposition
from psydac.api.essential_bc import apply_essential_bc_stencil

from struphy.psydac_api.linear_operators import CompositeLinearOperator, BoundaryOperator, IdentityOperator
from struphy.psydac_api.mass import WeightedMassOperator

from scipy.linalg import solve_circulant
import numpy as np


class MassMatrixPreconditioner(LinearSolver):
    """
    Preconditioner for inverting 3d weighted mass matrices. 
    
    The mass matrix is approximated by a Kronecker product of 1d mass matrices in each direction with correct boundary conditions (block diagonal in case of vector-valued spaces). In this process, the 3d weight function is appoximated by a 1d counterpart in the FIRST (eta_1) direction at the fixed point (eta_2=0.5, eta_3=0.5). The inversion is then performed with a Kronecker solver.
    
    Parameters
    ----------
    mass_operator : struphy.psydac_api.mass.WeightedMassOperator
        The weighted mass operator for which the approximate inverse is needed.

    apply_bc : bool
        Whether to include boundary operators.
    """
    
    def __init__(self, mass_operator, apply_bc=True):
        
        assert isinstance(mass_operator, WeightedMassOperator)
        assert mass_operator.domain == mass_operator.codomain, 'Only square mass matrices can be inverted!'
        
        self._femspace = mass_operator.domain_femspace
        self._space = mass_operator.domain
        
        # 3d Kronecker stencil matrices and solvers
        solverblocks = []
        matrixblocks = []
        
        # collect TensorFemSpaces in a tuple
        if isinstance(self._femspace, TensorFemSpace):
            femspaces = (self._femspace,)
        else:
            femspaces = self._femspace.spaces
            
        n_comps = len(femspaces)
        n_dims = self._femspace.ldim

        assert n_dims == 3 # other dims not yet implemented
        
        # get boundary conditions list from BoundaryOperator in ComposedLinearOperator M0 of mass operator
        if isinstance(mass_operator.M0.operators[0], BoundaryOperator):
            bc = mass_operator.M0.operators[0].bc
        else:
            bc = [[None, None], [None, None], [None, None]]

        # loop over components
        for c in range(n_comps):
            
            # 1d mass matrices and solvers
            solvercells = []
            matrixcells = []
            
            # loop over spatial directions
            for d in range(n_dims):
                
                # weight function only along in first direction
                if d == 0:
                    #pts = [0.5] * (n_dims - 1)
                    fun = [[lambda e1 : mass_operator.weights[c][c](e1, np.array([.5]), np.array([.5])).squeeze()]]
                else:
                    fun = None
                    
                # get 1D FEM space (serial, not distributed) and quadrature order
                femspace_1d = femspaces[c].spaces[d]
                qu_order_1d = femspaces[c].quad_order[d]
                    
                # assemble 1d weighted mass matrix
                domain_decompos_1d = DomainDecomposition([femspace_1d.ncells], [femspace_1d.periodic])
                femspace_1d_tensor = TensorFemSpace(domain_decompos_1d, femspace_1d, quad_order=[qu_order_1d])
                
                M = WeightedMassOperator(femspace_1d_tensor, femspace_1d_tensor, weights=fun)
                M.assemble(verbose=False)
                M.matrix.exchange_assembly_data()
                M = M.matrix
                
                # apply boundary conditions
                if apply_bc:
                    
                    if mass_operator._domain_symbolic_name != 'H1vec':
                        if femspace_1d.basis == 'B':
                            if bc[d][0] == 'd': apply_essential_bc_stencil(M, axis=0, ext=-1, order=0, identity=True)
                            if bc[d][1] == 'd': apply_essential_bc_stencil(M, axis=0, ext=+1, order=0, identity=True)
                    else:
                        if c == d:
                            if bc[d][0] == 'd': apply_essential_bc_stencil(M, axis=0, ext=-1, order=0, identity=True)
                            if bc[d][1] == 'd': apply_essential_bc_stencil(M, axis=0, ext=+1, order=0, identity=True)
                        
                M_arr = M.toarray()
                
                # create 1d solver for mass matrix
                if is_circulant(M_arr):
                    solvercells += [FFTSolver(M_arr)]
                else:
                    solvercells += [SparseSolver(M.tosparse())]
                    
                # === NOTE: for KroneckerStencilMatrix being built correctly, 1d matrices must be local to process! ===
                periodic = femspaces[c].vector_space.periods[d]
                
                n = femspaces[c].vector_space.npts[d]
                p = femspaces[c].vector_space.pads[d]
                s = femspaces[c].vector_space.starts[d]
                e = femspaces[c].vector_space.ends[d]
                
                cart_decomp_1d = CartDecomposition(domain_decompos_1d, [n], [[s]], [[e]], [p], [1])
                
                V_local = StencilVectorSpace(cart_decomp_1d)
                
                M_local = StencilMatrix(V_local, V_local)
                
                row_indices, col_indices = np.nonzero(M_arr)

                for row_i, col_i in zip(row_indices, col_indices):

                    # only consider row indices on process
                    if row_i in range(V_local.starts[0], V_local.ends[0] + 1):
                        row_i_loc = row_i - s

                        M_local._data[row_i_loc + p, (col_i + p - row_i)%M_arr.shape[1]] = M_arr[row_i, col_i]

                # check if stencil matrix was built correctly
                assert np.allclose(M_local.toarray()[s:e + 1], M_arr[s:e + 1])
                
                matrixcells += [M_local.copy()]
                # =======================================================================================================
            
            if isinstance(self._femspace, TensorFemSpace):
                matrixblocks += [KroneckerStencilMatrix(self._femspace.vector_space, self._femspace.vector_space, *matrixcells)]
                solverblocks += [KroneckerLinearSolver(self._femspace.vector_space, solvercells)]
            else:
                matrixblocks += [KroneckerStencilMatrix(self._femspace.vector_space[c], self._femspace.vector_space[c], *matrixcells)]
                solverblocks += [KroneckerLinearSolver(self._femspace.vector_space[c], solvercells)]
                
        # build final matrix and solver
        if isinstance(self._femspace, TensorFemSpace):
            self._matrix = matrixblocks[0]
            self._solver = solverblocks[0]
        else:
            
            blocks = [[matrixblocks[0], None, None],
                      [None, matrixblocks[1], None],
                      [None, None, matrixblocks[2]]]
            
            self._matrix = BlockLinearOperator(self._femspace.vector_space, self._femspace.vector_space, blocks=blocks)
            self._solver = BlockDiagonalSolver(self._femspace.vector_space, solverblocks)
            
        # save mass operator to be inverted (needed in solve method)
        if apply_bc:
            self._M = mass_operator.M0
        else:
            self._M = mass_operator.M
            
        # temporary vectors for dot product
        tmp_vectors = []
        for op in self._M._operators[:-1]:
            tmp_vectors.append(op.codomain.zeros())
            
        self._tmp_vectors = tuple(tmp_vectors)
        
    @property
    def space(self):
        """ Stencil-/BlockVectorSpace or PolarDerhamSpace.
        """
        return self._space

    @property
    def matrix(self):
        """ Approximation of the input mass matrix as KroneckerStencilMatrix.
        """
        return self._matrix
    
    @property
    def solver(self):
        """ KroneckerLinearSolver or BlockDiagonalSolver for exactly inverting the approximate mass matrix self.matrix.
        """
        return self._solver
    
    def solve(self, rhs, out=None):
        """
        Computes (B * E * M^(-1) * E^T * B^T) * rhs as an approximation for an inverse mass matrix.
        
        Parameters
        ----------
        rhs : psydac.linalg.basic.Vector
            The right-hand side vector.
            
        out : psydac.linalg.basic.Vector, optional
            If given, the output vector will be written into this vector in-place.
        
        Returns
        -------
        out : psydac.linalg.basic.Vector
            The result of (B * E * M^(-1) * E^T * B^T) * rhs.
        """
        
        assert isinstance(rhs, Vector)
        assert rhs.space == self._space
            
        # successive dot products with all but last operator
        x = rhs
        for i in range(len(self._tmp_vectors)):
            y = self._tmp_vectors[i]
            A = self._M._operators[i]
            if isinstance(A, (StencilMatrix, BlockLinearOperator)):
                self.solver.solve(x, out=y)
            else:
                A.dot(x, out=y)
            x = y

        # last operator
        A = self._M.operators[-1]
        if out is None:
            out = A.dot(x)
        else:
            assert isinstance(out, Vector)
            assert out.space == self._space
            A.dot(x, out=out)
            
        return out
    
    
class ProjectorPreconditioner(LinearSolver):
    """
    Preconditioner for approximately inverting a (polar) 3d inter-/histopolation matrix via
    
        (B * P * I * E^T * B^T)^(-1) approx. B * P * I^(-1) * E^T * B^T.
        
    In case that P and E are identity operators, the solution is exact (pure tensor product case).
    
    Parameters
    ----------
    projector : struphy.psydac_api.projectors.Projector
        The global commuting projector for which the inter-/histopolation matrix shall be inverted. 

    transposed : bool, optional
        Whether to invert the transposed inter-/histopolation matrix.

    apply_bc : bool, optional
        Whether to include the boundary operators.
    """
    
    def __init__(self, projector, transposed=False, apply_bc=False):
        
        # vector space in tensor product case/polar case
        self._space = projector.I.domain
        
        # save Kronecker solver (needed in solve method)
        self._solver = projector.projector_tensor.solver
        
        self._transposed = transposed
        
        # save inter-/histopolation matrix to be inverted
        if transposed:
            if apply_bc:
                self._I = projector.I0T
            else:
                self._I = projector.IT
        else:
            if apply_bc:
                self._I = projector.I0
            else:
                self._I = projector.I
                
        # temporary vectors for dot product
        tmp_vectors = []
        for op in self._I._operators[:-1]:
            tmp_vectors.append(op.codomain.zeros())
            
        self._tmp_vectors = tuple(tmp_vectors)
        
    @property
    def space(self):
        """ Stencil-/BlockVectorSpace or PolarDerhamSpace.
        """
        return self._space
    
    @property
    def solver(self):
        """ KroneckerLinearSolver for exactly inverting tensor product inter-histopolation matrix.
        """
        return self._solver
    
    @property
    def transposed(self):
        """ Whether to invert the transposed inter-/histopolation matrix.
        """
        return self._transposed
    
    def solve(self, rhs, out=None):
        """
        Computes (B * P * I^(-1) * E^T * B^T) * rhs, resp. (B * P * I^(-T) * E^T * B^T) * rhs (transposed=True) as an approximation for an inverse inter-/histopolation matrix.
        
        Parameters
        ----------
        rhs : psydac.linalg.basic.Vector
            The right-hand side vector.
            
        out : psydac.linalg.basic.Vector, optional
            If given, the output vector will be written into this vector in-place.
        
        Returns
        -------
        out : psydac.linalg.basic.Vector
            The result of (B * E * M^(-1) * E^T * B^T) * rhs, resp. (B * P * I^(-T) * E^T * B^T) * rhs (transposed=True).
        """
        
        assert isinstance(rhs, Vector)
        assert rhs.space == self._space
            
        # successive dot products with all but last operator
        x = rhs
        for i in range(len(self._tmp_vectors)):
            y = self._tmp_vectors[i]
            A = self._I._operators[i]
            if isinstance(A, (StencilMatrix, KroneckerStencilMatrix, BlockLinearOperator)):
                self.solver.solve(x, out=y, transposed=self._transposed)
            else:
                A.dot(x, out=y)
            x = y

        # last operator
        A = self._I.operators[-1]
        if out is None:
            out = A.dot(x)
        else:
            assert isinstance(out, Vector)
            assert out.space == self._space
            A.dot(x, out=out)
        
        return out
        
    
class FFTSolver(DirectSolver):
    """
    Solve the equation Ax = b for x, assuming A is a circulant matrix.
    b can contain multiple right-hand sides (RHS) and is of shape (#RHS, N).

    Parameters
    ----------
    circmat : np.ndarray
        Generic circulant matrix.
    """
    
    def __init__(self, circmat):

        assert isinstance(circmat, np.ndarray)
        assert is_circulant(circmat)

        self._space = np.ndarray
        self._column = circmat[:, 0]

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space(self):
        return self._space

    #...
    def solve(self, rhs, out=None, transposed=False):
        """
        Solves for the given right-hand side.

        Parameters
        ----------
        rhs : np.ndarray
            The right-hand sides to solve for. The vectors are assumed to be given in C-contiguous order, 
            i.e. if multiple right-hand sides are given, then rhs is a two-dimensional array with the 0-th 
            index denoting the number of the right-hand side, and the 1-st index denoting the element inside 
            a right-hand side.

        out : np.ndarray, optional
            Output vector. If given, it has to have the same shape and datatype as rhs.

        transposed : bool
            If and only if set to true, we solve against the transposed matrix. (supported by the underlying solver)
        """
        
        assert rhs.T.shape[0] == self._column.size

        if out is None:
            out = solve_circulant(self._column, rhs.T).T

        else:
            assert out.shape == rhs.shape
            assert out.dtype == rhs.dtype

            out[:] = solve_circulant(self._column, rhs.T).T

        return out

def is_circulant(mat):
    """ 
    Returns true if a matrix is circulant.
    
    Parameters
    ----------
    mat : array[float]
        The matrix that is checked to be circulant.
            
    Returns
    -------
    circulant : bool
        Whether the matrix is circulant (=True) or not (=False).
    """
    
    assert isinstance(mat, np.ndarray)
    assert len(mat.shape) == 2

    for i in range(mat.shape[0] - 1):
        circulant = np.allclose(mat[i, :], np.roll(mat[i + 1, :], -1))
        if not circulant:
            return circulant

    return circulant