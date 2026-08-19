import logging
from dataclasses import dataclass

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.feec.preconditioner import (
    H1vecKineticMetricPreconditioner,
    H1vecKineticMetricWoodburyPreconditioner,
    MassMatrixDiagonalPreconditioner,
)
from struphy.feec.variational_utilities import BracketOperator, H1vecKineticMetric
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

    def __init__(self, rho: FEECVariable = None):
        self.variables = self.Variables()
        ##Needed for metric regularization
        self.rho = rho

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`VariationalMomentumAdvection`.

        Parameters
        ----------
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Linear solver for mass-matrix related solves.
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
        with_regularization: bool = False
        alpha_divdiv: float = 1.0

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

        self._with_regularization = self.options.with_regularization
        self._alpha_divdiv = self.options.alpha_divdiv
        self._metric_alpha = 2.0 * self._alpha_divdiv

        assert self.rho is not None
        assert isinstance(self.rho, FEECVariable)
        assert self.rho.space == "L2"

        if self._with_regularization and not self.domain.has_exact_mapping_hessian:
            raise NotImplementedError(
                "VariationalMomentumAdvection regularization requires an "
                "analytical mapping Hessian. "
                f"Mapping {type(self.domain).__name__} with "
                f"kind_map={self.domain.kind_map} is not currently supported."
            )

        # assembly of WMMnew happens in VariationalDensityEvolve
        self._Mrho = self.mass_ops.WMMnew
        rho = self.rho.spline.vector

        if self._with_regularization:
            self._kinetic_metric = self.mass_ops.get_h1vec_kinetic_metric(
                self._metric_alpha,
            )

            self._Kdivrho = self._kinetic_metric.divdiv_operator

            self._kinetic_metric.update_weight_if_needed(rho, self.rho.generation)

            self._momentum_operator = self._kinetic_metric
            # pc = H1vecKineticMetricWoodburyPreconditioner(
            #                     self._kinetic_metric,
            #                     auxiliary_nsteps=5,
            #                     spectral_iterations=8,
            #                     spectral_safety=1.5,
            #     )
            pc = H1vecKineticMetricPreconditioner(self._kinetic_metric)
        else:
            self._Kdivrho = None
            self._kinetic_metric = None

            self._Mrho.spline_functions["l2_field"].vector = rho
            self._Mrho.assemble()

            self._momentum_operator = self._Mrho
            pc = MassMatrixDiagonalPreconditioner(self._momentum_operator)

        self._momentum_pc = pc

        self._momentum_solve = inverse(
            self._momentum_operator,
            "pcg",
            pc=self._momentum_pc,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=False,
            recycle=False,
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
        self.derivative = self._momentum_operator + self._dt2_brack

        self.inv_derivative = inverse(
            self._momentum_pc @ self.derivative,
            "gmres",
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=self._lin_solver.verbose,
            recycle=False,
        )

    @profile
    def __call__(self, dt):
        rho = self.rho.spline.vector
        self._update_momentum_operator(rho)

        if self._nonlin_solver.type == "Newton":
            self.__call_newton(dt)
        elif self._nonlin_solver.type == "Picard":
            self.__call_picard(dt)

    def __call_newton(self, dt):
        # Initialize variable for Newton iteration
        un = self.variables.u.spline.vector
        mn = self._momentum_operator.dot(un, out=self._tmp_mn)
        self.brack.update_u(mn)
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

            if xp.isnan(err):
                raise FloatingPointError("NaN residual in VariationalDensityEvolve.")
            if it == 0:
                err0 = max(float(err), xp.finfo(float).tiny)

            relative_err = err / err0

            if err <= tol**2 or relative_err <= tol**2:
                break

            # Newton step
            pc_diff = self._momentum_pc.dot(diff, out=self._tmp__pc_diff)
            update = self.inv_derivative.dot(pc_diff, out=self._tmp_update)
            if self._info:
                logger.info(
                    "information on the linear solver : ",
                    self.inv_derivative._info,
                )
            un1 -= update
            mn1 = self._momentum_operator.dot(un1, out=self._tmp_mn1)

        if it == self.options.nonlin_solver.maxiter - 1 or xp.isnan(err):
            logger.info(
                f"!!!WARNING: Maximum iteration in VariationalMomentumAdvection reached - not converged \n {err =} \n {tol**2 =}",
            )

        self.update_feec_variables(u=un1)

    def __call_picard(self, dt):
        """Advance momentum using Picard iteration."""

        un = self.variables.u.spline.vector

        mn = self._momentum_operator.dot(
            un,
            out=self._tmp_mn,
        )
        self.brack.update_u(mn)

        mn1 = mn.copy(out=self._tmp_mn1)
        un1 = un.copy(out=self._tmp_un1)

        tol = float(self._nonlin_solver.tol)
        tol_sq = tol * tol

        # _momentum_solve.dot_inner() returns a squared norm.
        absolute_threshold = 4.0 * tol_sq
        stagnation_threshold = 10.0 * tol_sq
        stagnation_relative_change = 1.0e-3
        stagnation_iterations = 3

        tiny = float(xp.finfo(float).tiny)

        err = float("inf")
        err0 = None
        previous_err = None
        stagnation_count = 0
        converged = False
        accepted_by_stagnation = False

        if self._info:
            logger.info("")
            logger.info(
                "Picard iteration in VariationalMomentumAdvection",
            )

        for it in range(self._nonlin_solver.maxiter):
            # Midpoint velocity.
            un.copy(out=self._tmp_un12)
            self._tmp_un12 += un1
            self._tmp_un12 *= 0.5
            un12 = self._tmp_un12

            # Advection term.
            self.brack.dot(
                un12,
                out=self._tmp_advection,
            )
            self._tmp_advection *= dt
            advection = self._tmp_advection

            # Residual:
            #
            #     F = m^{n+1,r} - m^n + dt * advection.
            mn1.copy(out=self._tmp_diff)
            self._tmp_diff -= mn
            self._tmp_diff += advection

            err = float(
                self._momentum_solve.dot_inner(
                    self._tmp_diff,
                    self._tmp_diff,
                )
            )

            if not bool(xp.isfinite(err)):
                raise FloatingPointError(
                    f"Non-finite residual in VariationalMomentumAdvection: iteration={it + 1}, err={err}."
                )

            if err0 is None:
                err0 = max(err, tiny)

            relative_err = err / err0

            if self._info:
                logger.info(
                    "Momentum Picard iteration: %d, error: %.16e, relative error: %.16e",
                    it + 1,
                    err,
                    relative_err,
                )

            # Since err is a squared norm, compare it against tol**2.
            # The factor four prevents pointless iterations at roundoff level.
            if err <= absolute_threshold or relative_err <= tol_sq:
                converged = True
                break

            # Accept stagnation only when already close to the requested
            # absolute tolerance.
            if previous_err is not None:
                relative_change = abs(previous_err - err) / max(
                    previous_err,
                    err,
                    tiny,
                )

                if relative_change <= stagnation_relative_change:
                    stagnation_count += 1
                else:
                    stagnation_count = 0

                if stagnation_count >= stagnation_iterations and err <= stagnation_threshold:
                    converged = True
                    accepted_by_stagnation = True
                    break

            previous_err = err

            # Picard update:
            #
            #     m^{n+1,r+1} = m^n - dt * advection.
            mn.copy(out=self._tmp_mn1)
            self._tmp_mn1 -= advection
            mn1 = self._tmp_mn1

            # Recover velocity from momentum.
            self._momentum_solve.dot(
                mn1,
                out=self._tmp_un1,
            )
            un1 = self._tmp_un1

        iterations = it + 1

        if not converged:
            logger.warning(
                "Maximum iteration count in "
                "VariationalMomentumAdvection reached without "
                "convergence:\n"
                "  iterations = %d\n"
                "  err        = %.16e\n"
                "  target     = %.16e",
                iterations,
                err,
                tol_sq,
            )
        elif accepted_by_stagnation and self._info:
            logger.info(
                "Accepted stagnated momentum Picard residual: iterations=%d, err=%.16e, target=%.16e.",
                iterations,
                err,
                tol_sq,
            )

        if self._info:
            logger.info(
                "Momentum Picard iterations: %d",
                iterations,
            )

        self.update_feec_variables(u=un1)

    def _initialize_mass(self):
        """Initialization of the mass matrix solver"""
        # Inverse mass matrix needed to compute the error
        self.pc_Mv = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.Mv,
        )
        self._inv_Mv = inverse(
            self.mass_ops.Mv,
            "pcg",
            pc=self.pc_Mv,
            tol=1e-10,
            maxiter=1000,
            verbose=False,
        )

    def _get_error_newton(self, mn_diff):
        err_u = self._inv_Mv.dot_inner(self.derham.boundary_ops["v"].dot(mn_diff), mn_diff)
        return err_u

    def _update_momentum_operator(self, rho):
        """Update the density-weighted momentum metric and preconditioner."""

        if self._with_regularization:
            metric_changed = self._kinetic_metric.update_weight_if_needed(
                rho,
                self.rho.generation,
            )
        else:
            self._Mrho.spline_functions["l2_field"].vector = rho
            self._Mrho.assemble()
            metric_changed = True

        if not metric_changed:
            return

        pc = self._momentum_pc

        if isinstance(pc, H1vecKineticMetricPreconditioner):
            pc.update_metric(self._kinetic_metric)

        elif isinstance(pc, H1vecKineticMetricWoodburyPreconditioner):
            pc.update_metric(self._kinetic_metric)

        elif isinstance(pc, MassMatrixDiagonalPreconditioner):
            pc.update_mass_operator(self._Mrho)
