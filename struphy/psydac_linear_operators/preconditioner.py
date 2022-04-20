from psydac.linalg.basic import LinearSolver
from psydac.linalg.direct_solvers import DirectSolver
from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector, BlockMatrix, BlockDiagonalSolver
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace
from psydac.api.discretization import discretize
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
from psydac.linalg.kron import KroneckerLinearSolver

from sympde.topology import elements_of
from sympde.expr import BilinearForm, integral
from sympde.calculus import dot
from sympde.topology import Line, Derham

from struphy.feec.psydac_derham import Derham_build
#from struphy.linear_algebra.linalg_kron import kron_fftsolve_3d

from scipy.linalg import solve_circulant, circulant
from numpy import ndarray, zeros_like, allclose, roll


class MassMatrixPreConditioner(LinearSolver):

    def __init__(self, femspace):

        # Get dimensions (=number of elements for periodic boundary conditions) and degrees
        if isinstance(femspace, TensorFemSpace):
            _Nel = [[space.nbasis for space in femspace.spaces]]
            _p   = [[space.degree for space in femspace.spaces]]
            _basis = [[space.basis for space in femspace.spaces]]
        elif isinstance(femspace, ProductFemSpace):
            _Nel = [[direction.nbasis for direction in space.spaces] for space in femspace.spaces]
            _p   = [[direction.degree for direction in space.spaces] for space in femspace.spaces]
            _basis = [[direction.basis for direction in space.spaces] for space in femspace.spaces]
        else:
            raise AssertionError('Argument "femspace" not of correct type.')

        self._femspace = femspace

        # Psydac symbolic logical domain
        _domain_log = Line('L', bounds=(0, 1))
        # Psydac symbolic Derham
        _derham_symb = Derham(_domain_log)

        # Obtain first column of 1d mass matrices 
        self._solvers = []
        for Nel_dirs, p_dirs, basis_dirs in zip(_Nel, _p, _basis):
            self._solvers += [[]]
            for Nel, p, basis in zip(Nel_dirs, p_dirs, basis_dirs):

                # Discrete logical domain, no mpi communicator passed (!), hence not distributed
                _domain_log_h = discretize(_domain_log, ncells=[Nel])

                # Discrete De Rham with periodic boundary conditions
                if basis == 'M': p += 1
                _derham = discretize(_derham_symb, _domain_log_h, degree=[p], periodic=[True])

                # 1d FemSpace
                if basis == 'B':
                    _space_symb = _derham_symb.V0
                    _space = _derham.V0
                elif basis == 'M':
                    _space_symb = _derham_symb.V1
                    _space = _derham.V1

                _u, _v = elements_of(_space_symb, names='u, v')

                _a = BilinearForm((_u, _v), integral(_domain_log, _u * _v))

                self._a_h = discretize(
                    _a, _domain_log_h, (_space, _space), backend=PSYDAC_BACKEND_GPYCCEL)

                _M = self._a_h.assemble()

                self._solvers[-1] += [FFTSolver(_M.toarray())]
                
        if len(self._solvers) == 1:
            self._solver = KroneckerLinearSolver(self.space, self._solvers[0])
        else:
            _blocks = [KroneckerLinearSolver(space, self._solvers[n]) for n, space in enumerate(self.space.spaces)]
            self._solver = BlockDiagonalSolver(self.space, _blocks)

    @property
    def space(self):
        return self._femspace.vector_space

    def solve( self, rhs, out=None, transposed=False ):
        assert isinstance(rhs, (StencilVector, BlockVector))
        assert rhs.space == self.space
        return self._solver.solve(rhs)


class FFTSolver(DirectSolver):
    """
    Solve the equation Ax = b for x, assuming A is a circulant matrix.
    b can contain mutliple RHS and is of shape (NRHS, N).

    Parameters
    ----------
    circmat : np.array
        Generic circulant matrix.

    """
    def __init__( self, circmat ):

        assert isinstance( circmat, ndarray )
        assert is_circulant(circmat)

        self._space = ndarray
        self._column = circmat[:, 0]

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space( self ):
        return self._space

    #...
    def solve( self, rhs, out=None, transposed=False ):
        """
        Solves for the given right-hand side.

        Parameters
        ----------
        rhs : ndarray
            The right-hand sides to solve for. The vectors are assumed to be given in C-contiguous order,
            i.e. if multiple right-hand sides are given, then rhs is a two-dimensional array with the 0-th
            index denoting the number of the right-hand side, and the 1-st index denoting the element inside
            a right-hand side.
        
        out : ndarray | NoneType
            Output vector. If given, it has to have the same shape and datatype as rhs.
        
        transposed : bool
            If and only if set to true, we solve against the transposed matrix. (supported by the underlying solver)
        """
        
        assert rhs.T.shape[0] == self._column.size

        if out is None:
            out = solve_circulant( self._column, rhs.T ).T

        else:
            assert out.shape == rhs.shape
            assert out.dtype == rhs.dtype

            # currently no in-place solve exposed
            out[:] = solve_circulant( self._column, rhs.T ).T

        return out


def is_circulant(mat):
    
    assert isinstance(mat, ndarray)
    assert len(mat.shape) == 2

    for i in range(mat.shape[0] - 1):
        row = allclose( mat[i, :], roll(mat[i + 1, :], -1) )
        if not row:
            return row

    return row