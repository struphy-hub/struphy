from psydac.linalg.iterative_solvers import pcg

from struphy.psydac_linear_operators.linear_operators import Sum_matrix as Sum
from struphy.psydac_linear_operators.linear_operators import Difference_matrix as Diff


class Schur_solver:
    '''Solves for x in the block system

    [[A B], [C Id]] [x, y] = [[A -B], [-C Id]] [xn, yn] ,

    using the Schur complement S=A-BC, where Id is the identity matrix and [a, b] is given.
    The solution is given by

    x = S^{-1}[(A + BC)*xn - 2B*yn] .'''

    def __init__(self, A, BC, pc=None, tol=1e-6, maxiter=1000, verbose=False):
        '''
        Parameters
        ----------
            A: LinearOperator
                Upper left block from [[A B], [C Id]].

            BC: LinearOperator
                Product from [[A B], [C Id]].

            pc: NoneType | str | psydac.linalg.basic.LinearSolver | Callable
                Preconditioner for S=A-BC, it should approximate the inverse of S.
                Can either be:
                * None, i.e. not pre-conditioning (this calls the standard `cg` method)
                * The strings 'jacobi' or 'weighted_jacobi'. (rather obsolete, supply a callable instead, if possible)
                * A LinearSolver object (in which case the out parameter is used)
                * A callable with two parameters (S, r), where S is the LinearOperator from above, and r is the residual.

            tol : float
                Absolute tolerance for L2-norm of residual r = A*x - b.

            maxiter: int
                Maximum number of iterations.

            verbose : bool
                If True, L2-norm of residual r is printed at each iteration. 

        Arguments
        ---------
            xn: StencilVector
                Solution from previous time step.

            Byn: StencilVector
                The product B*yn.

        Returns
        -------
            x: StencilVector
                Converged solution.

            info: dict
                Convergence information.
        '''

        self._pc = pc
        self._tol = tol
        self._maxiter = maxiter
        self._verbose = verbose

        self._schur = Diff(A, BC)
        self._rhs_mat = Sum(A, BC)

    def __call__(self, xn, Byn):

        _rhs = self._rhs_mat.dot(xn) - 2.*Byn

        x, info = pcg(self._schur, _rhs, self._pc, x0=xn, tol=self._tol,
                      maxiter=self._maxiter, verbose=self._verbose)

        return x, info
