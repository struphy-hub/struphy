from psydac.linalg.basic import Vector, LinearOperator
from psydac.linalg.solvers import inverse


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

    solver_name : str
        See [psydac.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params : 
        Must correspond to the chosen solver.
    '''

    def __init__(self, A, BC, solver_name, **solver_params):

        assert isinstance(A, LinearOperator)
        assert isinstance(BC, LinearOperator)

        assert A.domain == BC.domain
        assert A.codomain == BC.codomain

        # linear operators
        self._A = A
        self._BC = BC

        # initialize solver with dummy matrix A
        self._solver_name = solver_name
        self._solver = inverse(A, solver_name, **solver_params)

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
        schur = self._A - dt**2 * self._BC
        rhs_m = self._A + dt**2 * self._BC
        
        # use setter to update lhs matrix
        self._solver.linop = schur

        # right-hand side vector rhs = 2*dt*[ rhs_m/(2*dt) @ xn - Byn ] (in-place!)
        rhs = rhs_m.dot(xn, out=self._rhs)
        rhs /= 2*dt
        rhs -= Byn
        rhs *= 2*dt

        # solve linear system (in-place if out is not None)
        x = self._solver.dot(rhs, out=out)

        return x, self._solver._info
