import logging

from feectools.linalg.basic import IdentityOperator, LinearOperator
from feectools.linalg.block import BlockLinearOperator, BlockVectorSpace
from feectools.linalg.solvers import inverse

from struphy.feec.preconditioner import MassMatrixPreconditioner
from struphy.feec.variational_utilities import dot_normalized
from struphy.linear_algebra.schur_solver import SchurSolverFull
from struphy.propagators.base import Propagator

logger = logging.getLogger("struphy")


class VariationalLinearEvolve(Propagator):
    r"""Base class of the propagators of the linear variational MHD models that advance the velocity
    :math:`\mathbf u` and one other variable :math:`x` (magnetic field, :math:`q` or pressure) with a single linear
    solve per time step, without Picard or Newton iteration.

    The linear models are discretized as

    .. math::

        \mathbb M^v[\hat\rho_0] \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t}
        + c\, \mathbb A_0^\top \mathbb M_x \mathbf x^{n+\frac12} + \mathbf f(\mathbf x^n) = 0 \,,
        \qquad
        \frac{\mathbf x^{n+1} - \mathbf x^n}{\Delta t} + \mathbb A_0\, \mathbf u^{n+\frac12} = 0 \,,

    with the transport operator :math:`\mathbb A_0` weighted by the equilibrium, constant in time. The term
    :math:`\mathbf f(\mathbf x^n)` comes from the transport operator weighted by the perturbation; it is taken at
    :math:`t^n`, the same weight that advances the auxiliary second-order variable of the energy, which makes the step
    energy conserving (first order in time). The system is linear in :math:`(\mathbf u^{n+1}, \mathbf x^{n+1})` and
    solved in increment form,
    :math:`(\mathbf u^{n+1}, \mathbf x^{n+1}) = (\mathbf u^n, \mathbf x^n) - \mathbb J^{-1} \mathbf F(\mathbf u^n,
    \mathbf x^n)` with the residual :math:`\mathbf F` and the constant matrix

    .. math::

        \mathbb J = \begin{pmatrix} \mathbb M^v[\hat\rho_0] & \frac{c \Delta t}{2} \mathbb A_0^\top \mathbb M_x
        \\ \frac{\Delta t}{2} \mathbb A_0 & \mathbb I \end{pmatrix} \,,

    by :class:`~struphy.linear_algebra.schur_solver.SchurSolverFull`. The subclasses define the variables, the options
    and the operators.
    """

    def _allocate_mass(self, precond: str | None):
        """The density-weighted mass matrix (assembled in VariationalDensityEvolve) and its preconditioner."""
        self._Mrho = self.mass_ops.WMMnew
        self._Mrho_pc = None if precond is None else MassMatrixPreconditioner(self._Mrho, dim_reduce=None)

    def _refresh_mass_pc(self):
        """Update the preconditioner if WMMnew was re-assembled since the last step."""
        if self._Mrho_pc is not None:
            self._Mrho_pc.refresh()

    def _pc_kwargs(self) -> dict:
        return {} if self._Mrho_pc is None else {"pc": self._Mrho_pc}

    def _allocate_schur_solver(
        self,
        A0: LinearOperator,
        A0T_Mx: LinearOperator,
        c: float,
        space_x,
        solver: str,
        solver_params,
    ):
        r"""The matrix :math:`\mathbb J` and its Schur solver; the dt-dependent scalars are set in
        :meth:`_set_dt`."""
        self._c = c
        self._dt_J = None
        # scalars overwritten in _set_dt
        self._J01 = 2 * A0T_Mx
        self._J10 = 2 * A0

        Jacs = BlockVectorSpace(self.derham.Vvpol, space_x)
        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()
        self._J = BlockLinearOperator(Jacs, Jacs)
        self._J[0, 0] = self._Mrho
        self._J[0, 1] = self._J01
        self._J[1, 0] = self._J10
        self._J[1, 1] = IdentityOperator(space_x)
        self._inv_J = SchurSolverFull(
            self._J,
            solver,
            **self._pc_kwargs(),
            tol=solver_params.tol,
            maxiter=solver_params.maxiter,
            verbose=solver_params.verbose,
            recycle=True,
        )

    def _allocate_mass_solver(self, solver: str, solver_params):
        r"""The inverse of :math:`\mathbb M^v[\hat\rho_0]`, for a step without implicit coupling."""
        self._inv_Mrho = inverse(
            self._Mrho,
            solver,
            **self._pc_kwargs(),
            tol=solver_params.tol,
            maxiter=solver_params.maxiter,
            verbose=solver_params.verbose,
            recycle=True,
        )

    def _set_dt(self, dt: float):
        if dt != self._dt_J:
            self._J01._scalar = self._c * dt / 2
            self._J10._scalar = dt / 2
            self._dt_J = dt

    def _solve_increment(self, dt: float, f_u, f_x):
        r"""The increment :math:`\mathbb J^{-1} (\mathbf f_u, \mathbf f_x)`, solved for the normalized right-hand side
        (the tolerance of the linear solver is relative to it)."""
        self._set_dt(dt)
        self._tmp_f[0] = f_u
        self._tmp_f[1] = f_x
        incr = dot_normalized(self._inv_J, self._tmp_f, self._tmp_incr)
        if self._info:
            logger.info(f"{self.__class__.__name__}, linear solver: {self._inv_J._solver._info}")
        return incr
