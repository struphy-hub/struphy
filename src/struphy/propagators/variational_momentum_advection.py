import logging
from dataclasses import dataclass

import cunumpy as xp
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI
from scope_profiler import ProfileManager

from struphy.feec import preconditioner
from struphy.feec.preconditioner import MassMatrixPreconditioner
from struphy.feec.variational_utilities import BracketOperator, kinetic_metric
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
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
    """

    class Variables:
        """Container for variables advanced by :class:`VariationalMomentumAdvection`.

        Attributes
        ----------
        u : FEECVariable
            Velocity variable in ``"H1vec"`` space.
        """

        def __init__(self):
            self._u: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1vec"
            self._u = new

    def __init__(self, divdiv_alpha: float = 0.0):
        """
        Parameters
        ----------
        divdiv_alpha : float, default=0.0
            Coefficient of the div-div regularization of the kinetic energy (a parameter of the model, the same in
            all its variational propagators), see :func:`~struphy.feec.variational_utilities.kinetic_metric`.
        """
        assert divdiv_alpha >= 0.0, f"divdiv_alpha must be non-negative, got {divdiv_alpha}."
        self.variables = self.Variables()
        self.divdiv_alpha = divdiv_alpha

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
            Nonlinear iteration controls (Picard/Newton).
        """

        # propagator options
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

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

        # assembly of WMMnew happens in VariationalDensityEvolve
        self._Mrho = kinetic_metric(self.derham, self.mass_ops, self.basis_ops, self.divdiv_alpha)
        if self.divdiv_alpha > 0.0 and self._nonlin_solver.type == "Picard":
            logger.warning(
                "VariationalMomentumAdvection: with divdiv_alpha > 0 the Picard iteration converges only for small "
                'time steps; use a small dt or NonlinearSolverParameters(type="Newton").'
            )
        # "cg" takes no preconditioner
        self._Mrho_pc = (
            MassMatrixPreconditioner(self.mass_ops.WMMnew, dim_reduce=None) if self.options.solver == "pcg" else None
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

    def __call__(self, dt):
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
            **({"pc": self.pc_Mv} if self.options.solver == "pcg" else {}),
            tol=1e-10,
            maxiter=1000,
            verbose=False,
        )

    def _get_error_newton(self, mn_diff):
        err_u = self._inv_Mv.dot_inner(self.derham.boundary_ops["v"].dot(mn_diff), mn_diff)
        return err_u
