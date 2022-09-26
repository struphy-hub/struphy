from sympde.topology import Line, Derham

from psydac.linalg.basic import LinearSolver
from psydac.linalg.direct_solvers import DirectSolver, SparseSolver
from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector, BlockMatrix, BlockDiagonalSolver
from psydac.linalg.kron import KroneckerLinearSolver, KroneckerStencilMatrix

from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace

from psydac.api.discretization import discretize
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
from psydac.api.essential_bc import apply_essential_bc_stencil

from struphy.psydac_api.mass_psydac import get_mass

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
            A 2d list containing optional weight functions. Usually obtained from struphy.psydac_api.mass_psydac.WeightedMass.
    """
    
    def __init__(self, derham, space, weight=None):
        
        self._femspace = getattr(derham, space)

        # get 1d space infos:
        # number of elements and quadrature order
        Nel = derham.Nel
        nqs = derham.quad_order
        
        # degrees and periodicity
        degs = self._femspace.degree
        pers = self._femspace.periodic
        
        if isinstance(self._femspace, TensorFemSpace):
            degs = [degs]
            pers = [pers]
            
        # type of basis
        if isinstance(self._femspace, TensorFemSpace):
            bases = [[space.basis for space in self._femspace.spaces]]
        else:
            bases = [[space.basis for space in space.spaces] for space in self._femspace.spaces]
        
        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'

        # Psydac symbolic logical domain
        domain_log = Line('L', bounds=(0, 1))
        
        # Psydac symbolic Derham
        derham_symb = Derham(domain_log)
        
        # assemble 1d mass matrices and solvers
        self._matrices = []
        self._solvers = []
        
        # loop over components
        for c, (ps_comp, periodics_comp, bases_comp) in enumerate(zip(degs, pers, bases)):
            self._solvers += [[]]
            self._matrices += [[]]
            
            # loop over directions
            for d, (p, periodic, basis) in enumerate(zip(ps_comp, periodics_comp, bases_comp)):

                # discrete logical domain, no mpi communicator passed, hence not distributed
                domain_log_h = discretize(domain_log, ncells=[Nel[d]])
 
                # discrete De Rham sequence
                if basis == 'M': p += 1
                derham_1d = discretize(derham_symb, domain_log_h, degree=[p], periodic=[periodic], quad_order=[nqs[d]])
                
                # 1d mass matrix with weight in first direction
                if d == 0 and weight is not None:
                    fun = [[lambda e1 : weight[c][c](e1[:, None, None], np.array([0.5])[:, None, None], np.array([0.5])[:, None, None])]]
                else:
                    fun = None
                
                if basis == 'B':
                    M = get_mass(derham_1d.V0, derham_1d.V0, fun)
                    
                    # apply boundary conditions!
                    if derham.bc[d][0] == 'd': apply_essential_bc_stencil(M, axis=d, ext=-1, order=0, identity=True)
                    if derham.bc[d][1] == 'd': apply_essential_bc_stencil(M, axis=d, ext=+1, order=0, identity=True)
                    
                else:
                    M = get_mass(derham_1d.V1, derham_1d.V1, fun)

                self._matrices[-1] += [M]
                
                if is_circulant(M.toarray()):
                    self._solvers[-1] += [FFTSolver(M.toarray())]
                else:
                    self._solvers[-1] += [SparseSolver(M.tosparse())]
                    
        if len(self._solvers) == 1:

            self._matrix = KroneckerStencilMatrix(self.space, self.space, *self._matrices[0])
            self._solver = KroneckerLinearSolver(self.space, self._solvers[0])

        else:

            blocks_list = [KroneckerStencilMatrix(space, space, *self._matrices[n]) for n, space in enumerate(self.space.spaces)]
            blocks = {}
            
            for n in range(3):
                blocks[(n, n)] = blocks_list[n]
                
            self._matrix = BlockMatrix(self.space, self.space, blocks)

            solver_blocks = [KroneckerLinearSolver(space, self._solvers[n]) for n, space in enumerate(self.space.spaces)]
            self._solver = BlockDiagonalSolver(self.space, solver_blocks)
        
    @property
    def space(self):
        return self._femspace.vector_space

    @property
    def matrix(self):
        return self._matrix
    
    def solve(self, rhs, out=None, transposed=False):
        assert isinstance(rhs, (StencilVector, BlockVector))
        assert rhs.space == self.space
        return self._solver.solve(rhs)


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