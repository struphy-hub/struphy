from sympde.topology import Line, Derham

from psydac.linalg.basic import LinearSolver
from psydac.linalg.direct_solvers import DirectSolver, SparseSolver
from psydac.linalg.stencil import StencilMatrix, StencilVector, StencilVectorSpace
from psydac.linalg.block import BlockVector, BlockMatrix, BlockDiagonalSolver
from psydac.linalg.kron import KroneckerLinearSolver, KroneckerStencilMatrix

from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace

from psydac.api.essential_bc import apply_essential_bc_stencil

from struphy.psydac_api.linear_operators import CompositeLinearOperator
from struphy.psydac_api.mass import WeightedMassOperator

from struphy.polar.basic import PolarVector

from scipy.linalg import solve_circulant

import numpy as np


class MassMatrixPreconditioner(LinearSolver):
    """
    Preconditioner for inverting 3d weighted mass matrices. 
    
    The mass matrix is approximated by a Kronecker product of 1d mass matrices in each direction with correct boundary conditions (block diagonal in case of vector-valued spaces). In this process, the 3d weight function is appoximated by a 1d counterpart in the FIRST (eta_1) direction at the fixed point (eta_2=0.5, eta_3=0.5). The inversion is then performed with a Kronecker solver.
    
    Parameters
    ----------
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete de Rham sequence on the logical unit cube.
            
        space : str
            The space corresponding to the mass matrix (V0, V1, V2, V3 or V0vec).
            
        weight : list[callables]
            A 2d list containing optional weight functions. Usually obtained from struphy.psydac_api.mass.WeightedMassOperators.
    """
    
    def __init__(self, mass_operator):
        
        assert isinstance(mass_operator, WeightedMassOperator)
        assert mass_operator.domain == mass_operator.codomain, 'Only square mass marices can be inverted!'
        
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

        # loop over components
        for c in range(n_comps):
            
            # 1d mass matrices and solvers
            solvercells = []
            matrixcells = []
            
            # loop over spatial directions
            for d in range(n_dims):
                
                # weight function only along in first direction
                if d == 0 and mass_operator._weight is not None:
                    pts = [0.5] * (n_dims - 1)
                    fun = [[lambda e1 : mass_operator._weight[c][c](e1, *pts).squeeze()]]
                else:
                    fun = None
                    
                # get 1D FEM space (serial, not distributed)
                femspace_1d = femspaces[c].spaces[d]
                quad_order_1d = femspaces[c].quad_order[d]
                    
                # assemble 1d mass matrix
                M = WeightedMassOperator.assemble_mat(TensorFemSpace(femspace_1d, quad_order=[quad_order_1d]), TensorFemSpace(femspace_1d, quad_order=[quad_order_1d]), fun)
                
                # apply boundary conditions
                if mass_operator._domain_symbolic_name != 'H1vec':
                    if femspace_1d.basis == 'B':
                        if mass_operator.bc[d][0] == 'd': apply_essential_bc_stencil(M, axis=0, ext=-1, order=0, identity=True)
                        if mass_operator.bc[d][1] == 'd': apply_essential_bc_stencil(M, axis=0, ext=+1, order=0, identity=True)
                else:
                    if c == d:
                        if mass_operator.bc[d][0] == 'd': apply_essential_bc_stencil(M, axis=0, ext=-1, order=0, identity=True)
                        if mass_operator.bc[d][1] == 'd': apply_essential_bc_stencil(M, axis=0, ext=+1, order=0, identity=True)
                        
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
                
                V_local = StencilVectorSpace([n], [p], [periodic], starts=[s], ends=[e])
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
            
            self._matrix = BlockMatrix(self._femspace.vector_space, self._femspace.vector_space, blocks=blocks)
            self._solver = BlockDiagonalSolver(self._femspace.vector_space, solverblocks)
            
        # save mass operator (needed in solve method)
        self._mass_operator = mass_operator
        
    @property
    def space(self):
        return self._space

    @property
    def matrix(self):
        return self._matrix
    
    @property
    def solver(self):
        return self._solver
    
    def solve(self, rhs, out=None):
        """
        TODO
        """
        
        assert isinstance(rhs, (StencilVector, BlockVector, PolarVector))
        assert rhs.space == self.space
        
        out = rhs.copy()
        
        # apply operators in mass_operator.operator.operators and replace StencilMatrix/BlockMatrix dot with self.solver.solve
        for op in self._mass_operator.operator.operators:
            if isinstance(op, (StencilMatrix, BlockMatrix)):
                out = self.solver.solve(out)
            else:
                out = op.dot(out)
        
        return out
    
    
class ProjectorPreconditioner(LinearSolver):
    """
    Preconditioner for inverting (polar) 3d inter-/histopolation matrices of the form (B * P * I * E^T * B^T)^(-1) via the approximation B * P * I^(-1) * E^T * B^T. In case that P and E are identity operators, the solution is exact (pure tensor product case).
    
    Parameters
    ----------
        projector : Projector
            The global commuting projector for which the inter-/histopolation matrix shall be inverted. 
            
        transposed : bool
            Whether to invert the transposed inter-/histopolation matrix.
    """
    
    def __init__(self, projector, transposed=False):
        
        # vector space in tensor product case/polar case
        if projector.I is None:
            self._space = projector.space.vector_space
        else:
            self._space = projector.dofs_extraction_op.codomain
        
        # save Kronecker solver (needed in solve method)
        self._solver = projector.projector_tensor.solver
        
        self._transposed = transposed
        
        # save inter-/histopolation matrix to be inverted (None in tensor product case, CompositeLinearOperator in polar case)
        self._I = projector.I
        
        # save boundary operator (either None or BoundaryOperator)
        self._B = projector.boundary_op
        
        if self._B is None:
            self._BT = None
        else:
            self._BT = projector.boundary_op.transpose()
        
    @property
    def space(self):
        return self._space
    
    @property
    def solver(self):
        return self._solver
    
    @property
    def transposed(self):
        return self._transposed
    
    def solve(self, rhs, out=None):
        """
        Solves approximately the system (B * P * I * ET * BT).dot(x) = rhs or (B * P * I * ET * BT)^T.dot(x) = rhs for x.
        """
        
        assert isinstance(rhs, (StencilVector, BlockVector, PolarVector))
        assert rhs.space == self.space
        
        if self._BT is not None:
            out = self._BT.dot(rhs)
        else:
            out = rhs.copy()
        
        # polar case
        if isinstance(self._I, CompositeLinearOperator):

            # apply operators in self._I.operators and replace StencilMatrix/BlockMatrix dot with tensor product self.solver
            for op in self._I.operators:
                if isinstance(op, (StencilMatrix, BlockMatrix)):
                    out = self.solver.solve(out, transposed=self.transposed)
                else:
                    out = op.dot(out)
                    
        # tensor product case            
        else:
            out = self.solver.solve(out, transposed=self.transposed)
            
        if self._B is not None:
            out = self._B.dot(out)
        else:
            out = out.copy()
        
        return out
    
    
class FFTSolver(DirectSolver):
    """
    Solve the equation Ax = b for x, assuming A is a circulant matrix.
    b can contain multiple right-hand sides (RHS) and is of shape (#RHS, N).

    Parameters
    ----------
        circmat : array[float]
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
            rhs : array[float]
                The right-hand sides to solve for. The vectors are assumed to be given in C-contiguous order, 
                i.e. if multiple right-hand sides are given, then rhs is a two-dimensional array with the 0-th 
                index denoting the number of the right-hand side, and the 1-st index denoting the element inside 
                a right-hand side.
        
            out : array[float] | NoneType
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

            # currently no in-place solve exposed
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