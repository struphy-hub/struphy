from psydac.linalg.iterative_solvers import pcg

from struphy.psydac_api.linear_operators import SumLinearOperator as Sum
from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply


class Schur_solver:
    '''Solves for x in the block system

    [[A B], [C Id]] [x, y] = [[A -B], [-C Id]] [xn, yn] ,

    using the Schur complement S=A-BC, where Id is the identity matrix and [xn, yn] is given.
    The solution is given by

    x = S^{-1}[(A + BC)*xn - 2B*yn] .'''

    def __init__(self, A, BC, pc, tol, maxiter, verbose):
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
            xn : StencilVector
                Solution from previous time step.

            Byn : StencilVector
                The product B*yn.

            dt: float
                Time step size.

        Returns
        -------
            x : StencilVector
                Converged solution.

            info : dict
                Convergence information.
        '''

        self._A = A
        self._BC = BC
        self._domain = A.domain
        self._codomain = A.codomain
        self._dtype = A.dtype

        assert A.domain == BC.domain
        assert A.codomain == BC.codomain

        self._pc = pc
        self._tol = tol
        self._maxiter = maxiter
        self._verbose = verbose

    @property
    def A(self):
        """Upper left block from [[A B], [C Id]]."""
        return self._A
    
    @property
    def BC(self):
        """Product from [[A B], [C Id]]."""
        return self._BC
    
    @A.setter
    def A(self, a):
        """Upper left block from [[A B], [C Id]]."""
        self._A = a
    
    @BC.setter
    def BC(self, bc):
        """Product from [[A B], [C Id]]."""
        self._BC = bc
    

    def __call__(self, xn, Byn, dt):

        self._schur   = Sum(self._A, Multiply(-dt**2, self._BC) )
        self._rhs_mat = Sum(self._A, Multiply(dt**2, self._BC))

        assert xn.space == self._rhs_mat.domain
        assert Byn.space == self._rhs_mat.codomain

        _rhs = self._rhs_mat.dot(xn) - dt*2.*Byn

        x, info = pcg(self._schur, _rhs, self._pc, x0=xn, tol=self._tol,
                      maxiter=self._maxiter, verbose=self._verbose)

        return x, info
