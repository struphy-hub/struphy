import logging
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI
from scope_profiler import ProfileManager

from struphy.feec import preconditioner
from struphy.feec.preconditioner import MassMatrixPreconditioner
from struphy.feec.variational_utilities import (
    BracketOperator,
    HdivBracketMomentumDerivative,
    HdivBracketOperator,
    kinetic_metric,
)
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.utils import VelocityMass
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VariationalMomentumAdvection(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf u \in (H^1)^3` such that

    .. math::

        \int_{\Omega} \partial_t ( \rho  \mathbf{u}) \cdot \mathbf{v} \,\textrm d \mathbf x -
        \int_{\Omega} \rho \mathbf{u} \cdot  [\mathbf{u}, \mathbf{v}] \, \textrm d \mathbf x = 0 \,.

    On the logical domain:

    .. math::

        \int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \,\textrm d \boldsymbol \eta -
        \int_{\hat{\Omega}} \hat{\rho}^3 \hat{\mathbf{u}} \cdot G [\hat{\mathbf{u}}, \hat{\mathbf{v}}] \, \textrm d \boldsymbol \eta = 0 \,,

    which is discretized as

    .. math::

        \mathbb M^v[\hat{\rho}_h^n] \frac{\mathbf u^{n+1}- \mathbf u^n}{\Delta t} - \left(\sum_{\mu} (\hat{\Pi}^{0}[\hat{\mathbf u}_h^{n+1/2} \cdot \vec{\boldsymbol \Lambda}^1] \mathbb G P_{\mu} - \hat{\Pi}^0[\hat{\mathbf A}^1_{\mu,h} \cdot \vec{\boldsymbol \Lambda}^v])^\top P_i \right) \mathbb M^v[\hat{\rho}_h^n] \mathbf u^{n} = 0 ~ .

    where :math:`P_\mu` stand for the :class:`~struphy.feec.basis_projection_ops.CoordinateProjector` and the weights
    in the the two :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{u}}_h^{n+1/2} = (\mathbf{u}^{n+1/2})^\top \vec{\boldsymbol \Lambda}^v \in (V_h^0)^3 \,, \qquad \hat{\mathbf A}^1_{\mu,h} = \nabla P_\mu((\mathbf u^{n+1/2})^\top \vec{\boldsymbol \Lambda}^v)] \in V_h^1\,, \qquad \hat{\rho}_h^{n} = (\rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,.

    With a div-div regularization (``divdiv_alpha`` :math:`> 0`), the mass matrix :math:`\mathbb M^v[\hat\rho]` is
    replaced by the metric :math:`\mathbb A[\hat\rho]` of the regularized kinetic energy, see
    :func:`~struphy.feec.variational_utilities.kinetic_metric`.

    **Velocity in** :math:`H(\textnormal{div})` (velocity variable in ``"Hdiv"`` space): find
    :math:`\mathbf u \in H_0(\textnormal{div})` (vanishing normal component at the boundary) such that, for all
    :math:`\mathbf v \in H_0(\textnormal{div})`,

    .. math::

        \int_{\Omega} \partial_t ( \rho  \mathbf{u}) \cdot \mathbf{v} \,\textrm d \mathbf x -
        \int_{\Omega} \rho \mathbf{u} \cdot  [\mathbf{u}, \mathbf{v}] \, \textrm d \mathbf x = 0 \,, \qquad
        [\mathbf u, \mathbf v] = \mathbf u \cdot \nabla \mathbf v - \mathbf v \cdot \nabla \mathbf u\,.

    The velocity is a 2-form, :math:`\hat{\mathbf u}^2_h = (\mathbf u^n)^\top \vec{\boldsymbol\Lambda}^2`, and the bracket is
    discretized with the vector fields as derivations of the auxiliary spline space :math:`W` of double knots
    (:class:`~struphy.feec.variational_utilities.HdivBracketOperator`, :math:`\mathbb B[\hat{\mathbf y}]` with
    :math:`\mathbf v^\top \mathbb B[\hat{\mathbf y}] \mathbf u = \langle \mathbf m, \widehat{[A_{\mathbf u}, A_{\mathbf v}]}\rangle`
    and :math:`\hat{\mathbf y} = (\mathbb M^2_{\textnormal{log}})^{-1} \mathbf m` the representer of the momentum
    :math:`\mathbf m` with the unweighted logical mass matrix of the 2-forms). The semi-discrete problem is

    .. math::

        \mathbb M^2_\rho\, \dot{\mathbf u} = \mathbb B[\hat{\mathbf y}(\mathbb M^2_\rho \mathbf u)]\, \mathbf u \,,

    with the density-weighted mass matrix :math:`\mathbb M^2_\rho` (:class:`~struphy.propagators.utils.VelocityMass`:
    :math:`\mathbb M^2_{\rho_0}` without the constructor argument ``rho``, else the total density
    :math:`\rho_0 + \tilde\rho` with the density perturbation ``rho``, frozen during the step). The time discretization
    is the implicit midpoint rule with the momentum of the bracket at :math:`t^n` (``bracket_momentum="n"``, default),

    .. math::

        \left(\mathbb M^2_\rho - \frac{\Delta t}{2} \mathbb B[\hat{\mathbf y}^n]\right) \mathbf u^{n+1}
        = \left(\mathbb M^2_\rho + \frac{\Delta t}{2} \mathbb B[\hat{\mathbf y}^n]\right) \mathbf u^n \,, \qquad
        \hat{\mathbf y}^n = \hat{\mathbf y}(\mathbb M^2_\rho \mathbf u^n)\,,

    one linear solve (GMRES, left-preconditioned with :math:`(\mathbb M^2_\rho)^{-1}`), or at :math:`t^{n+1/2}`
    (``bracket_momentum="n+1/2"``), :math:`\hat{\mathbf y}^{n+1/2} = \hat{\mathbf y}(\mathbb M^2_\rho \mathbf u^{n+1/2})`,
    a nonlinear system solved by Picard iteration (one such linear solve per iteration) or by Newton iteration
    (``nonlin_solver.type``; Jacobian :math:`\mathbb M^2_\rho - \frac{\Delta t}{2}\mathbb B[\hat{\mathbf y}^{n+1/2}]
    - \frac{\Delta t}{2} \mathbb D[\mathbf u^{n+1/2}] \mathbb M^2_\rho` with the derivative :math:`\mathbb D` of the
    bracket with respect to the momentum,
    :class:`~struphy.feec.variational_utilities.HdivBracketMomentumDerivative`), followed by one linear solve with the
    momentum of the last iterate. Since :math:`\mathbb B` is skew-symmetric for any momentum, both conserve the kinetic energy
    :math:`\frac 12 \mathbf u^\top \mathbb M^2_\rho \mathbf u` exactly (up to the solver tolerances).
    Not available with polar splines or ``divdiv_alpha > 0``.
    """

    class Variables:
        """Container for variables advanced by :class:`VariationalMomentumAdvection`.

        Attributes
        ----------
        u : FEECVariable
            Velocity variable in ``"H1vec"`` or ``"Hdiv"`` space.
        """

        def __init__(self):
            self._u: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("H1vec", "Hdiv")
            self._u = new

    def __init__(self, divdiv_alpha: float = 0.0, rho: FEECVariable = None):
        """
        Parameters
        ----------
        divdiv_alpha : float, default=0.0
            Coefficient of the div-div regularization of the kinetic energy (a parameter of the model, the same in
            all its variational propagators), see :func:`~struphy.feec.variational_utilities.kinetic_metric`.

        rho : FEECVariable, optional
            Density perturbation :math:`\\tilde\\rho` (``"L2"`` space, not updated) of the mass matrix
            :math:`\\mathbb M^2_\\rho` of a velocity in ``"Hdiv"``; None for :math:`\\mathbb M^2_{\\rho_0}`.
            Not used for a velocity in ``"H1vec"``.
        """
        assert divdiv_alpha >= 0.0, f"divdiv_alpha must be non-negative, got {divdiv_alpha}."
        assert rho is None or (isinstance(rho, FEECVariable) and rho.space == "L2")
        self.variables = self.Variables()
        self.divdiv_alpha = divdiv_alpha
        self._rho = rho

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`VariationalMomentumAdvection`.

        Parameters
        ----------
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Linear solver for mass-matrix related solves
            (the non-symmetric Jacobian is always solved with "gmres").
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner used in linear solves.
        solver_params : SolverParameters, default=None
            Linear-solver controls.
        nonlin_solver : NonlinearSolverParameters, default=None
            Nonlinear iteration controls (Picard/Newton). For a velocity in ``"Hdiv"``, only ``type``, ``tol``,
            ``maxiter`` and ``info`` of the iteration with ``bracket_momentum="n+1/2"``.
        bracket_momentum : {"n", "n+1/2"}, default="n"
            Velocity in ``"Hdiv"``: time of the momentum in the bracket, :math:`t^n` (one linear solve) or
            :math:`t^{n+1/2}` (Picard or Newton iteration, ``nonlin_solver.type``). Must be ``"n"`` for a velocity in
            ``"H1vec"``.
        """

        OptsBracketMomentum = Literal["n", "n+1/2"]

        # propagator options
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        bracket_momentum: OptsBracketMomentum = "n"

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
            check_option(self.bracket_momentum, self.OptsBracketMomentum)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver

        self._info = self._nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        self._u_space = self.variables.u.space
        if self._u_space == "Hdiv":
            self._allocate_hdiv()
            return
        assert self.options.bracket_momentum == "n", (
            f'bracket_momentum="{self.options.bracket_momentum}" needs a velocity in "Hdiv".'
        )

        # assembly of WMMnew happens in VariationalDensityEvolve
        self._Mrho = kinetic_metric(self.derham, self.mass_ops, self.basis_ops, self.divdiv_alpha)
        if self.divdiv_alpha > 0.0 and self._nonlin_solver.type == "Picard":
            logger.warning(
                "VariationalMomentumAdvection: with divdiv_alpha > 0 the Picard iteration converges only for small "
                'time steps; use a small dt or NonlinearSolverParameters(type="Newton").'
            )
        # "cg" takes no preconditioner
        self._Mrho_pc = (
            MassMatrixPreconditioner(self.mass_ops.WMMnew, dim_reduce=None)
            if self.options.solver in ("pcg", "pcg_sr")
            else None
        )
        self._Mrho_inv = inverse(
            self._Mrho,
            self.options.solver,
            **({"pc": self._Mrho_pc} if self._Mrho_pc is not None else {}),
            tol=1e-10,
            maxiter=500,
            recycle=True,
        )

        self._initialize_mass()

        # bunch of temporaries to avoid allocating in the loop
        u = self.variables.u.spline.vector

        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_diff = u.space.zeros()
        self._tmp__pc_diff = u.space.zeros()
        self._tmp_update = u.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_advection = u.space.zeros()

        self.brack = BracketOperator(self.derham, self._tmp_mn)
        self._dt2_brack = 2.0 * self.brack
        self.derivative = self._Mrho + self._dt2_brack
        self.inv_derivative = inverse(
            self._Mrho_inv @ self.derivative,
            "gmres",
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=self._lin_solver.verbose,
            recycle=True,
        )

    def _allocate_hdiv(self):
        """Velocity in Hdiv: density-weighted 2-form mass matrix, bracket and linear solvers."""
        assert self.divdiv_alpha == 0.0, 'A velocity in "Hdiv" needs divdiv_alpha=0.'
        assert not self.derham.polar_splines, 'A velocity in "Hdiv" is not available with polar splines.'

        self._velocity_mass = VelocityMass(
            self.derham, self.domain, self.mass_ops, self.projected_equil, rho_field=self._rho
        )
        M = self._velocity_mass.M_rho
        # "cg" takes no preconditioner; the diagonal scaling of the Kronecker preconditioner is refreshed in
        # __call__ when the density changes
        self._Mrho_pc = (
            MassMatrixPreconditioner(M, dim_reduce=None) if self.options.solver in ("pcg", "pcg_sr") else None
        )
        self._Mrho_inv = inverse(
            M,
            self.options.solver,
            **({"pc": self._Mrho_pc} if self._Mrho_pc is not None else {}),
            tol=1e-14,
            maxiter=1000,
            recycle=True,
        )

        self.brack = HdivBracketOperator(self.derham, self.domain, self.mass_ops, tol=1e-14)
        # M - dt/2 B, with the scalar set in each step
        self._dt2_brack = -0.5 * self.brack
        self._lhs = M + self._dt2_brack
        self.inv_derivative = inverse(
            self._Mrho_inv @ self._lhs,
            "gmres",
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=self._lin_solver.verbose,
            recycle=True,
        )

        # Newton with the momentum at t^{n+1/2}: Jacobian M - dt/2 B - dt/2 D M, with D the derivative of the bracket
        # with respect to the momentum (at the velocity u^{n+1/2})
        if self.options.bracket_momentum == "n+1/2" and self._nonlin_solver.type == "Newton":
            self._brack_dm = HdivBracketMomentumDerivative(self.brack)
            self._dt2_dm = -0.5 * self._brack_dm
            self._jac = M + self._dt2_brack + self._dt2_dm @ M
            self.inv_jac = inverse(
                self._Mrho_inv @ self._jac,
                "gmres",
                tol=self._lin_solver.tol,
                maxiter=self._lin_solver.maxiter,
                verbose=self._lin_solver.verbose,
                recycle=True,
            )

        u = self.variables.u.spline.vector
        self._tmp_un1 = u.space.zeros()
        self._tmp_un1_old = u.space.zeros()
        self._tmp_F = u.space.zeros()
        self._tmp_pc_F = u.space.zeros()
        self._tmp_update = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_rhs = u.space.zeros()
        self._tmp_Bu = u.space.zeros()
        self._tmp_pc_rhs = u.space.zeros()
        self._tmp_diff = u.space.zeros()
        self._tmp_Mdiff = u.space.zeros()

    def _call_hdiv(self, dt):
        r"""Velocity in Hdiv: :math:`(\mathbb M - \frac{\Delta t}{2}\mathbb B) \mathbf u^{n+1} = (\mathbb M + \frac{\Delta t}{2}\mathbb B) \mathbf u^n`
        with the momentum at :math:`t^n`, or Picard iterations for the momentum at :math:`t^{n+1/2}`."""
        self._velocity_mass.update()
        if self._velocity_mass.weighted and self._Mrho_pc is not None:
            self._Mrho_pc.refresh()
        M = self._velocity_mass.M_rho

        un = self.variables.u.spline.vector
        self._dt2_brack._scalar = -dt / 2

        mn = M.dot(un, out=self._tmp_mn)
        self.brack.update_momentum(mn)
        un1 = self._solve_hdiv(un, mn, dt)

        if self.options.bracket_momentum == "n+1/2" and self._nonlin_solver.type == "Newton":
            un1 = self._newton_hdiv(un, un1, M, dt)
        elif self.options.bracket_momentum == "n+1/2":
            tol = self._nonlin_solver.tol
            if self._info:
                logger.info("")
                logger.info("Picard iteration in VariationalMomentumAdvection (Hdiv, momentum at t^{n+1/2})")
            for it in range(self._nonlin_solver.maxiter):
                un1_old = un1.copy(out=self._tmp_un1_old)
                un12 = un.copy(out=self._tmp_un12)
                un12 += un1
                un12 *= 0.5
                self.brack.update_momentum(M.dot(un12, out=self._tmp_mn))
                un1 = self._solve_hdiv(un, M.dot(un, out=self._tmp_mn), dt)
                diff = un1.copy(out=self._tmp_diff)
                diff -= un1_old
                err = diff.inner(M.dot(diff, out=self._tmp_Mdiff)) ** 0.5
                if self._info:
                    logger.info(f"iteration : {it} error : {err}")
                if err < tol or xp.isnan(err):
                    break
            if it == self._nonlin_solver.maxiter - 1 or xp.isnan(err):
                logger.warning(
                    f"!!!WARNING: Maximum iteration in VariationalMomentumAdvection reached - not converged \n {err =} \n {tol =}",
                )

        self.update_feec_variables(u=un1)

    def _newton_hdiv(self, un, un1, M, dt):
        r"""Newton iteration for :math:`F(\mathbf u^{n+1}) = \mathbb M (\mathbf u^{n+1} - \mathbf u^n) - \Delta t\,
        \mathbb B[\hat{\mathbf y}(\mathbb M \mathbf u^{n+1/2})] \mathbf u^{n+1/2} = 0`, from the initial guess ``un1``,
        followed by one linear solve with the momentum of the last iterate, which conserves the energy exactly."""
        tol = self._nonlin_solver.tol
        self._dt2_brack._scalar = -dt / 2
        self._dt2_dm._scalar = -dt / 2
        if self._info:
            logger.info("")
            logger.info("Newton iteration in VariationalMomentumAdvection (Hdiv, momentum at t^{n+1/2})")
        err = tol + 1.0
        for it in range(self._nonlin_solver.maxiter):
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5
            self.brack.update_momentum(M.dot(un12, out=self._tmp_mn))

            # residual F = M (u^{n+1} - u^n) - dt B u^{n+1/2}
            diff = un1.copy(out=self._tmp_diff)
            diff -= un
            F = M.dot(diff, out=self._tmp_F)
            Bu = self.brack.dot(un12, out=self._tmp_Bu)
            Bu *= dt
            F -= Bu
            pc_F = self._Mrho_inv.dot(F, out=self._tmp_pc_F)
            err = pc_F.inner(F) ** 0.5
            if self._info:
                logger.info(f"iteration : {it} error : {err}")
            if err < tol or xp.isnan(err):
                break

            self._brack_dm.set_velocity(un12)
            update = self.inv_jac.dot(pc_F, out=self._tmp_update)
            if self._info:
                logger.info(f"information on the linear solver : {self.inv_jac._info}")
            un1 -= update

        if it == self._nonlin_solver.maxiter - 1 or xp.isnan(err):
            logger.warning(
                f"!!!WARNING: Maximum iteration in VariationalMomentumAdvection reached - not converged \n {err =} \n {tol =}",
            )

        # final linear solve with the momentum of the last iterate (exact energy conservation)
        un12 = un.copy(out=self._tmp_un12)
        un12 += un1
        un12 *= 0.5
        self.brack.update_momentum(M.dot(un12, out=self._tmp_mn))
        return self._solve_hdiv(un, M.dot(un, out=self._tmp_mn), dt)

    def _solve_hdiv(self, un, mn, dt):
        """Solve (M - dt/2 B) u^{n+1} = M u^n + dt/2 B u^n for the current momentum of the bracket."""
        rhs = mn.copy(out=self._tmp_rhs)
        Bu = self.brack.dot(un, out=self._tmp_Bu)
        Bu *= dt / 2
        rhs += Bu
        with ProfileManager.profile_region(self._solve_region, functions=[self._Mrho_inv.dot]):
            pc_rhs = self._Mrho_inv.dot(rhs, out=self._tmp_pc_rhs)
        un1 = self.inv_derivative.dot(pc_rhs, out=self._tmp_un1)
        if self._info:
            logger.info(f"information on the linear solver : {self.inv_derivative._info}")
        return un1

    def __call__(self, dt):
        if self._u_space == "Hdiv":
            self._call_hdiv(dt)
            return

        # WMMnew is re-assembled in VariationalDensityEvolve: update the diagonal scaling of its preconditioner
        if self._Mrho_pc is not None:
            self._Mrho_pc.refresh()

        if self._nonlin_solver.type == "Newton":
            self.__call_newton(dt)
        elif self._nonlin_solver.type == "Picard":
            self.__call_picard(dt)

    def __call_newton(self, dt):
        # Initialize variable for Newton iteration
        un = self.variables.u.spline.vector
        mn = self._Mrho.dot(un, out=self._tmp_mn)
        mn1 = mn.copy(out=self._tmp_mn1)
        un1 = un.copy(out=self._tmp_un1)
        tol = self.options.nonlin_solver.tol
        err = tol + 1
        # Jacobian matrix for Newton solve
        self._dt2_brack._scalar = dt / 2
        if self._info:
            logger.info("")
            logger.info("Newton iteration in VariationalMomentumAdvection")

        for it in range(self.options.nonlin_solver.maxiter):
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Compute the advection term
            advection = self.brack.dot(un12, out=self._tmp_advection)
            advection *= dt

            # Difference with the previous approximation :
            # diff = m^{n+1,r}-m^{n+1,r+1} = m^{n+1,r}-m^{n}+advection
            diff = mn1.copy(out=self._tmp_diff)
            diff -= mn
            diff += advection

            # Get error and stop if small enough
            err = self._get_error_newton(diff)

            if self._info:
                logger.info(f"iteration : {it} error : {err}")
            if err < tol**2 or xp.isnan(err):
                break

            # Newton step
            with ProfileManager.profile_region(self._solve_region, functions=[self._Mrho_inv.dot]):
                pc_diff = self._Mrho_inv.dot(diff, out=self._tmp__pc_diff)
            update = self.inv_derivative.dot(pc_diff, out=self._tmp_update)
            if self._info:
                logger.info(f"information on the linear solver : {self.inv_derivative._info}")
            un1 -= update
            mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)

        if it == self.options.nonlin_solver.maxiter - 1 or xp.isnan(err):
            logger.warning(
                f"!!!WARNING: Maximum iteration in VariationalMomentumAdvection reached - not converged \n {err =} \n {tol**2 =}",
            )

        self.update_feec_variables(u=un1)

    def __call_picard(self, dt):
        # Initialize variable for Picard iteration
        un = self.variables.u.spline.vector
        mn = self._Mrho.dot(un, out=self._tmp_mn)
        mn1 = mn.copy(out=self._tmp_mn1)
        un1 = un.copy(out=self._tmp_un1)
        tol = self.options.nonlin_solver.tol
        err = tol + 1
        # Jacobian matrix for Newton solve

        for it in range(self.options.nonlin_solver.maxiter):
            # Picard iteration
            if err < tol**2 or xp.isnan(err):
                break
            # half time step approximation
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Compute the advection term
            advection = self.brack.dot(un12, out=self._tmp_advection)
            advection *= dt

            # Difference with the previous approximation :
            # diff = m^{n+1,r}-m^{n+1,r+1} = m^{n+1,r}-m^{n}+advection
            diff = mn1.copy(out=self._tmp_diff)
            diff -= mn
            diff += advection

            # Compute the norm of the difference
            err = self._Mrho_inv.dot_inner(self._tmp_diff, self._tmp_diff)

            # Update : m^{n+1,r+1} = m^n-advection
            mn1 = mn.copy(out=self._tmp_mn1)
            mn1 -= advection

            # Inverse the mass matrix to get the velocity
            with ProfileManager.profile_region(self._solve_region, functions=[self._Mrho_inv.dot]):
                un1 = self._Mrho_inv.dot(mn1, out=self._tmp_un1)

        if it == self.options.nonlin_solver.maxiter - 1 or xp.isnan(err):
            logger.warning(
                f"!!!WARNING: Maximum iteration in VariationalMomentumAdvection reached - not converged \n {err =} \n {tol**2 =}",
            )

        self.update_feec_variables(u=un1)

    def _initialize_mass(self):
        """Initialization of the mass matrix solver"""
        # Inverse mass matrix needed to compute the error
        self.pc_Mv = preconditioner.MassMatrixPreconditioner(
            self.mass_ops.Mv,
            dim_reduce=None,
        )
        self._inv_Mv = inverse(
            self.mass_ops.Mv,
            self.options.solver,
            # "cg" takes no preconditioner
            **({"pc": self.pc_Mv} if self.options.solver in ("pcg", "pcg_sr") else {}),
            tol=1e-10,
            maxiter=1000,
            verbose=False,
        )

    def _get_error_newton(self, mn_diff):
        err_u = self._inv_Mv.dot_inner(self.derham.boundary_ops["v"].dot(mn_diff), mn_diff)
        return err_u
