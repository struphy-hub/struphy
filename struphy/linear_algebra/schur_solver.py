from psydac.linalg.basic import Vector, LinearOperator, LinearSolver

from struphy.psydac_api.linear_operators import SumLinearOperator as Sum
from struphy.psydac_api.linear_operators import ScalarTimesLinearOperator as Multiply

import struphy.linear_algebra.iterative_solvers as it_solvers


class SchurSolver:
    '''Solves for :math:`x^{n+1}` in the block system

    .. math::

        \left( \matrix{
            A & \Delta t B \cr
            \Delta t C & \\text{Id}
        } \\right)
        \left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)
        =
        \left( \matrix{
            A & - \Delta t B \cr
            - \Delta t C & \\text{Id}
        } \\right)
        \left( \matrix{
            x^n \cr y^n
        } \\right)

    using the Schur complement :math:`S = A - \Delta t^2 BC`, where Id is the identity matrix
    and :math:`(x^n, y^n)^T` is given. The solution is given by

    .. math::

        x^{n+1} = S^{-1} \left[ (A + \Delta t^2 BC) \, x^n - 2 \Delta t B \, y^n \\right] \,.
        
    Parameters
    ----------
    A : psydac.linalg.basic.LinearOperator
        Upper left block from [[A B], [C Id]].

    BC : psydac.linalg.basic.LinearOperator
        Product from [[A B], [C Id]].

    pc : NoneType | psydac.linalg.basic.LinearSolver
         Preconditioner for "operator", it should approximate the inverse of "operator". Must have a "solve(rhs, out)" method.

    solver_name : str
        The name of the iterative solver used for inverting S. Currently available:
            * ConjugateGradient (for positive-definite S, pc=None possible in this case)
            * PConjugateGradient (for positive-definite S, recommended in this case)
            * BiConjugateGradientStab (for general S, pc=None possible in this case)
            * PBiConjugateGradientStab (for general S)

    tol : float
        Absolute tolerance for L2-norm of residual r = A*x - b.

    maxiter : int
        Maximum number of iterations.

    verbose : bool
        If True, L2-norm of residual r is printed at each iteration. 
    '''

    def __init__(self, A, BC, pc, solver_name, tol, maxiter, verbose):
        
        assert isinstance(A, LinearOperator)
        assert isinstance(BC, LinearOperator)

        assert A.domain == BC.domain
        assert A.codomain == BC.codomain
        
        # linear operators
        self._A = A
        self._BC = BC
        
        # preconditioner
        if pc is not None:
            assert isinstance(pc, LinearSolver)
        self._pc = pc
        
        # stop tolerance, maximum number of iterations and printing
        self._tol = tol
        self._maxiter = maxiter
        self._verbose = verbose
        
        # load linear solver
        self._solver_name = solver_name
        self._solver = getattr(it_solvers, solver_name)(A.domain)
        
        # right-hand side vector (avoids temporary memory allocation!)
        self._rhs = A.codomain.zeros()

    @property
    def A(self):
        """ Upper left block from [[A B], [C Id]].
        """
        return self._A

    @property
    def BC(self):
        """ Product from [[A B], [C Id]].
        """
        return self._BC
    
    @A.setter
    def A(self, a):
        """ Upper left block from [[A B], [C Id]].
        """
        self._A = a

    @BC.setter
    def BC(self, bc):
        """ Product from [[A B], [C Id]].
        """
        self._BC = bc

    def __call__(self, xn, Byn, dt, out=None):
        """
        Solves the 2x2 block matrix linear system.
        
        Parameters
        ----------
        xn : psydac.linalg.basic.Vector
            Solution from previous time step.

        Byn : psydac.linalg.basic.Vector
            The product B*yn.

        dt : float
            Time step size.
            
        out : psydac.linalg.basic.Vector, optional
            If given, the converged solution will be written into this vector (in-place).

        Returns
        -------
        out : psydac.linalg.basic.Vector
            Converged solution.

        info : dict
            Convergence information.
        """
        
        assert isinstance(xn, Vector)
        assert isinstance(Byn, Vector)
        assert xn.space == self._A.domain
        assert Byn.space == self._A.codomain

        # left- and right-hand side operators
        schur = Sum(self._A, Multiply(-dt**2, self._BC))
        rhs_m = Sum(self._A, Multiply( dt**2, self._BC))
        
        # right-hand side vector (in-place!)
        Byn *= 2*dt
        rhs_m.dot(xn, out=self._rhs)
        self._rhs -= Byn
        
        # solve linear system (in-place if out is not None)
        
        # solvers with preconditioner (must start with a 'P')
        if self._solver_name[0] == 'P':
            x, info = self._solver.solve(schur, self._rhs, self._pc, 
                                         x0=xn, tol=self._tol, 
                                         maxiter=self._maxiter, 
                                         verbose=self._verbose, out=out)
        else:
            x, info = self._solver.solve(schur, self._rhs, 
                                         x0=xn, tol=self._tol, 
                                         maxiter=self._maxiter, 
                                         verbose=self._verbose, out=out)
            
        return x, info