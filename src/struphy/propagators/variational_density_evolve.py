import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.basic import IdentityOperator
from feectools.linalg.block import BlockLinearOperator, BlockVectorSpace
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.feec.preconditioner import (
    H1vecKineticMetricPreconditioner,
    H1vecKineticMetricWoodburyPreconditioner,
    MassMatrixPreconditioner,
    MassMatrixDiagonalPreconditioner,
)
from struphy.feec.variational_utilities import (
    H1vecKineticMetric,
    InternalEnergyEvaluator,
    KineticEnergyEvaluator,
)
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VariationalDensityEvolve(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\rho \in L^2` and  :math:`\mathbf u \in (H^1)^3` such that

    .. math::

        &\partial_t \rho + \nabla \cdot ( \tilde{\rho} \mathbf u ) = 0 \,,
        \\[4mm]
        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x + \int_\Omega \left(\frac{|\mathbf u|^2}{2} - \frac{\partial(\rho \mathcal U(\rho))}{\partial \rho}\right) \nabla \cdot (\tilde{\rho} \mathbf v) \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3\,.

    Where :math:`\tilde{\rho}` is either :math:`\rho` for full-f models, :math:`\rho_0` for linear models or :math:`\rho_0+\rho` for :math:`\delta f` models.

    In the case of linear model, the second equation is not updated.

    On the logical domain:

    .. math::

        \begin{align}
        &\partial_t \hat{\rho}^3 + \nabla \cdot ( \hat{\rho}^3 \hat{\mathbf{u}} ) = 0 ~ ,
        \\[4mm]
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta
        + \int_{\hat{\Omega}} \left( \frac{| DF \hat{\mathbf{u}} |^2}{2} - \frac{\partial (\hat{\rho}^3 \mathcal U)}{\partial \hat{\rho}^3} \right) \nabla \cdot (\hat{\rho}^3 \hat{\mathbf{v}}) \, \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        \end{align}

    where :math:`\mathcal U` depends on the chosen model. It is discretized as

    .. math::

        \begin{align}
        &\frac{\mathbb M^v[\hat{\rho}_h^{n+1}] \mathbf u^{n+1}- \mathbb M^v[\hat{\rho}_h^n] \mathbf u^n}{\Delta t}
        + (\mathbb D \hat{\Pi}^{2}[\hat{\rho_h^{n}} \vec{\boldsymbol \Lambda}^v])^\top \hat{l}^3\left(\frac{DF \hat{\mathbf{u}}_h^{n+1} \cdot DF \hat{\mathbf{u}}_h^{n}}{2}
        - \frac{\hat{\rho}_h^{n+1}\mathcal U(\hat{\rho}_h^{n+1})-\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n})}{\hat{\rho}_h^{n+1}-\hat{\rho}_h^n} \right) = 0 ~ ,
        \\[2mm]
        &\frac{\boldsymbol \rho^{n+1}- \boldsymbol \rho^n}{\Delta t} + \mathbb D \hat{\Pi}^{2}[\hat{\rho_h^{n}} \vec{\boldsymbol \Lambda}^v] \mathbf u^{n+1/2} = 0 ~ ,
        \\[2mm]
        \end{align}

    where :math:`\hat{l}^3(f)` denotes the vector representing the linear form :math:`v_h \mapsto \int_{\hat{\Omega}} f(\boldsymbol \eta) v_h(\boldsymbol \eta) d \boldsymbol \eta`, that is the vector with components

    .. math::
        \hat{l}^3(f)_{ijk}=\int_{\hat{\Omega}} f \Lambda^3_{ijk} \textrm d \boldsymbol \eta

    and the weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{u}}_h^{k} = (\mathbf{u}^{k})^\top \vec{\boldsymbol \Lambda}^v \in (V_h^0)^3 \, \text{for k in} \{n, n+1/2, n+1\}, \qquad \hat{\rho}_h^{k} = (\rho^{k})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \, \text{for k in} \{n, n+1/2, n+1\} .
    """

    class Variables:
        """Container for variables advanced by :class:`VariationalDensityEvolve`.

        Attributes
        ----------
        rho : FEECVariable
            Density variable in ``"L2"`` space.
        u : FEECVariable
            Velocity variable in ``"H1vec"`` space.
        """

        def __init__(self):
            self._rho: FEECVariable = None
            self._u: FEECVariable = None

        @property
        def rho(self) -> FEECVariable:
            return self._rho

        @rho.setter
        def rho(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._rho = new

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1vec"
            self._u = new

    def __init__(self, s: FEECVariable = None):
        """
        Parameters
        ----------
        s : FEECVariable, default=None
            Entropy density 3-form (``"L2"`` space) evolved alongside the mass density.
            If ``None``, only the density equation is solved.
        """
        self.variables = self.Variables()
        self.s = s

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`VariationalDensityEvolve`.

        Parameters
        ----------
        model : {"pressureless", "barotropic", "full", "full_p", "full_q", "linear", "deltaf", "linear_q", "deltaf_q"}, default="barotropic"
            Density/thermodynamic model variant.
        gamma : float, default=5/3
            Adiabatic index.
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Linear solver for implicit substeps.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner used in linear solves.
        solver_params : SolverParameters, default=None
            Linear-solver controls.
        nonlin_solver : NonlinearSolverParameters, default=None
            Nonlinear iteration controls.
        with_regularization : bool, default=False
            Whether to use the metric-based H1vec div-div regularization.
        alpha_divdiv : float, default=1.0
            Coefficient in

            alpha_divdiv * int rho (div u)^2 dx.

            The corresponding coefficient in the momentum metric is
            2 * alpha_divdiv.
        """

        # specific literals
        OptsModel = Literal[
            "pressureless",
            "barotropic",
            "full",
            "full_p",
            "full_q",
            "linear",
            "deltaf",
            "linear_q",
            "deltaf_q",
        ]
        # propagator options
        model: OptsModel = "barotropic"
        gamma: float = 5.0 / 3.0
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        with_regularization: bool = False
        alpha_divdiv: float = 1.0

        def __post_init__(self):
            # checks
            check_option(self.model, self.OptsModel)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters()
            if not isinstance(self.with_regularization, bool):
                raise TypeError(f"with_regularization must be a bool, got {type(self.with_regularization)}.")

            if self.alpha_divdiv < 0.0:
                raise ValueError(f"alpha_divdiv must be non-negative, got {self.alpha_divdiv}.")

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
        if self.options.model == "full":
            assert self.s is not None
        self._model = self.options.model
        self._gamma = self.options.gamma
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize = self.options.nonlin_solver.linearize
        self._with_regularization = self.options.with_regularization
        self._alpha_divdiv = self.options.alpha_divdiv
        self._metric_alpha = 2.0 * self._alpha_divdiv

        if self._with_regularization and not self.domain.has_exact_mapping_hessian:
            raise NotImplementedError(
                "VariationalDensityEvolve regularization requires an "
                "analytical mapping Hessian. "
                f"Mapping {type(self.domain).__name__} with "
                f"kind_map={self.domain.kind_map} is not currently supported."
            )

        self._info = self.options.nonlin_solver.info and MPI.COMM_WORLD.Get_rank() == 0

        # Obtain variables before assembling the density-weighted matrix.
        rho = self.variables.rho.spline.vector
        u = self.variables.u.spline.vector
        self._Mrho = self.mass_ops.WMMnew

        if self._model in ("linear", "linear_q"):
            rhotmp = self.projected_equil.n3

        elif self._model in ("deltaf", "deltaf_q"):
            self._tmp_rho_deltaf = rho.space.zeros()
            rhotmp = rho.copy(out=self._tmp_rho_deltaf)
            rhotmp += self.projected_equil.n3

        else:
            rhotmp = rho

        if self._with_regularization:
            # WMMnew is shared with the committed metric cache. Density Newton
            # is about to overwrite it with a private iterate, so invalidate the
            # committed cache first.
            self.mass_ops.invalidate_committed_h1vec_metric()

            self._Kdivrho = self.mass_ops.create_h1vec_div_div(
                name="DensityNewtonH1vecDivDiv",
            )

            self._kinetic_metric = H1vecKineticMetric(
                self._Mrho,
                self._Kdivrho,
                alpha=self._metric_alpha,
            )

            # Assemble before constructing the preconditioner.
            self._kinetic_metric.update_weight(rhotmp)

            # The initial FEEC density has already been committed, and the private
            # metric now corresponds to it.
            self.mass_ops.publish_committed_h1vec_metric(
                self._Kdivrho,
                self.variables.rho,
            )

            self._momentum_operator = self._kinetic_metric
            self._momentum_pc = H1vecKineticMetricPreconditioner(
                self._kinetic_metric,
            )
        else:
            self._Mrho.spline_functions["l2_field"].vector = rhotmp
            self._Mrho.assemble()

            self._Kdivrho = None
            self._kinetic_metric = None
            self._momentum_operator = self._Mrho
            self._momentum_pc = MassMatrixDiagonalPreconditioner(
                self._Mrho,
            )
        self._momentum_inv = inverse(
            self._momentum_operator,
            "pcg",
            pc=self._momentum_pc,
            tol=1.0e-10,
            maxiter=500,
            verbose=False,
            recycle=False,
        )

        # FEM fields used by the projector.
        self.rhof = self.derham.create_spline_function(
            "rhof",
            "L2",
        )
        self.rhof1 = self.derham.create_spline_function(
            "rhof1",
            "L2",
        )

        # Projectors/evaluators.
        self._energy_evaluator = InternalEnergyEvaluator(
            self.derham,
            self._gamma,
        )
        self._kinetic_evaluator = KineticEnergyEvaluator(
            self.derham,
            self.domain,
            self.mass_ops,
            with_regularization=self._with_regularization,
        )
        self._initialize_projectors_and_mass()

        # if self._model in ["linear", "linear_q"]:
        #     rhotmp = self.projected_equil.n3
        # elif self._model in ["deltaf", "deltaf_q"]:
        #     self._tmp_rho_deltaf = rho.space.zeros()
        #     rhotmp = rho.copy(out=self._tmp_rho_deltaf)
        #     rhotmp += self.projected_equil.n3
        # else:
        #     rhotmp = rho
        # self._update_weighted_MM(rhotmp)

        # bunch of temporaries to avoid allocating in the loop
        self._tmp_un1 = u.space.zeros()
        self._tmp_un2 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_rhon1 = rho.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_rhon12 = rho.space.zeros()
        self._tmp_rhon_diff = rho.space.zeros()
        self._tmp_un_weak_diff = u.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_rho_advection = rho.space.zeros()
        self._linear_form_dl_drho = rho.space.zeros()

        # Compute the initial force in case we want to 'linearize' around a given equilibrium
        if self._linearize:
            self._compute_init_linear_form()

        if self._model in ["linear", "linear_q"]:
            self._update_Pirho(self.projected_equil.n3)

    @profile
    def __call__(self, dt):
        self.__call_newton(dt)

    @profile
    def __call_newton(self, dt):
        """Advance density and velocity using Newton iteration."""

        if self._info:
            logger.info("")
            logger.info("Newton iteration in VariationalDensityEvolve")

        # Current solution.
        rhon = self.variables.rho.spline.vector
        un = self.variables.u.spline.vector

        # ---------------------------------------------------------------
        # Linear models
        # ---------------------------------------------------------------
        if self._model in ("linear", "linear_q"):
            self.divPirho.dot(
                un,
                out=self._tmp_rho_advection,
            )
            self._tmp_rho_advection *= dt

            rhon.copy(out=self._tmp_rhon1)
            self._tmp_rhon1 -= self._tmp_rho_advection

            self.update_feec_variables(
                rho=self._tmp_rhon1,
                u=un,
            )
            if self._with_regularization:
                self._kinetic_metric.mark_weight_generation(
                    self.variables.rho.generation,
                )
            return

        # ---------------------------------------------------------------
        # Momentum at the beginning of the time step
        # ---------------------------------------------------------------
        if self._model in ("deltaf", "deltaf_q"):
            rhon.copy(out=self._tmp_rho_deltaf)
            self._tmp_rho_deltaf += self.projected_equil.n3
            rho = self._tmp_rho_deltaf
        else:
            rho = rhon

        self._update_momentum_operator(rho, update_preconditioner=False)

        self._momentum_operator.dot(
            un,
            out=self._tmp_mn,
        )
        mn = self._tmp_mn

        # Entropy is only needed for the full model.
        if self._model == "full":
            s = self.s.spline.vector
        else:
            s = None

        # The transport operator uses the density at the beginning of the
        # time step.
        self._update_Pirho(rho)

        # ---------------------------------------------------------------
        # Initial Newton iterate
        # ---------------------------------------------------------------
        rhon.copy(out=self._tmp_rhon1)
        self._tmp_rhon1 += self._tmp_rhon_diff
        rhon1 = self._tmp_rhon1

        un.copy(out=self._tmp_un1)
        self._tmp_un1 += self._tmp_un_diff
        un1 = self._tmp_un1

        if self._model in ("deltaf", "deltaf_q"):
            rhon1.copy(out=self._tmp_rho_deltaf)
            self._tmp_rho_deltaf += self.projected_equil.n3
            rho1 = self._tmp_rho_deltaf
        else:
            rho1 = rhon1

        self._update_momentum_operator(rho1)

        self._momentum_operator.dot(
            un1,
            out=self._tmp_mn1,
        )
        mn1 = self._tmp_mn1

        # ---------------------------------------------------------------
        # Nonlinear convergence controls
        # ---------------------------------------------------------------
        tol = float(self._nonlin_solver.tol)
        tol_sq = tol * tol

        # The nonlinear error returned by _get_error_newton is a squared
        # norm. Allow a small factor around tol**2 to account for roundoff
        # and the inexact inner linear solve.
        acceptance_factor = 2.0
        absolute_threshold = acceptance_factor * tol_sq

        # If the residual stagnates within this factor of tol**2, accept it.
        stagnation_acceptance_factor = 10.0
        stagnation_threshold = stagnation_acceptance_factor * tol_sq

        stagnation_relative_change = 1.0e-3
        stagnation_iterations = 3

        tiny = float(xp.finfo(float).tiny)

        err = float("inf")
        err0 = None
        previous_err = None
        stagnation_count = 0
        converged = False
        accepted_by_stagnation = False
        metric_matches_rho1 = True

        # ---------------------------------------------------------------
        # Newton iteration
        # ---------------------------------------------------------------
        for it in range(self._nonlin_solver.maxiter):
            # Midpoint velocity.
            un.copy(out=self._tmp_un12)
            self._tmp_un12 += un1
            self._tmp_un12 *= 0.5
            un12 = self._tmp_un12

            # Update the discrete variational derivative.
            self._update_linear_form_dl_drho(
                rhon,
                rhon1,
                un,
                un1,
                s,
            )

            # Momentum advection contribution.
            self.divPirhoT.dot(
                self._linear_form_dl_drho,
                out=self._tmp_advection,
            )
            self._tmp_advection *= dt

            # Density advection contribution.
            self.divPirho.dot(
                un12,
                out=self._tmp_rho_advection,
            )
            self._tmp_rho_advection *= dt

            # Density-equation residual.
            rhon1.copy(out=self._tmp_rhon_diff)
            self._tmp_rhon_diff -= rhon
            self._tmp_rhon_diff += self._tmp_rho_advection
            rhon_diff = self._tmp_rhon_diff

            # Momentum-equation residual.
            mn1.copy(out=self._tmp_mn_diff)
            self._tmp_mn_diff -= mn
            self._tmp_mn_diff += self._tmp_advection
            mn_diff = self._tmp_mn_diff

            # Squared nonlinear residual norm.
            err = float(
                self._get_error_newton(
                    mn_diff,
                    rhon_diff,
                )
            )

            if not bool(xp.isfinite(err)):
                raise FloatingPointError(
                    f"Non-finite residual in VariationalDensityEvolve: iteration={it + 1}, err={err}."
                )

            if err0 is None:
                err0 = max(err, tiny)

            relative_err = err / err0

            if self._info:
                logger.info(
                    "iteration: %d, error: %.16e, relative error: %.16e",
                    it + 1,
                    err,
                    relative_err,
                )

            # Primary convergence criterion.
            #
            # _get_error_newton returns a squared norm, hence both the
            # absolute and relative targets are tol**2.
            if err <= absolute_threshold or relative_err <= tol_sq:
                converged = True
                break

            # Detect stagnation near the requested tolerance. This handles
            # cases such as err=1.12e-16 with tol**2=1.00e-16, where further
            # Newton iterations cannot improve the result because of
            # roundoff and inexact inner solves.
            if previous_err is not None:
                relative_change = abs(previous_err - err) / max(previous_err, err, tiny)

                if relative_change <= stagnation_relative_change:
                    stagnation_count += 1
                else:
                    stagnation_count = 0

                if stagnation_count >= stagnation_iterations and err <= stagnation_threshold:
                    converged = True
                    accepted_by_stagnation = True

                    if self._info:
                        logger.info(
                            "Accepting stagnated Newton residual near tolerance: err=%.16e, target=%.16e.",
                            err,
                            tol_sq,
                        )

                    break

            previous_err = err

            # -----------------------------------------------------------
            # Newton correction
            # -----------------------------------------------------------
            self._get_jacobian(
                dt,
                rhon,
                rhon1,
                un,
                un1,
                s,
            )

            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = rhon_diff

            self._inv_Jacobian.dot(
                self._tmp_f,
                out=self._tmp_incr,
            )
            incr = self._tmp_incr
            linear_info = self._inv_Jacobian._solver._info
            linear_niter = int(linear_info.get("niter", -1))
            linear_success = bool(linear_info.get("success", False))
            if linear_success and linear_niter == 0:
                converged = True
                accepted_by_stagnation = True

                if self._info:
                    logger.info(
                        "Stopping Newton because the Jacobian solve accepted the current residual without an iteration."
                    )

                break

            if self._info:
                logger.info(
                    "Information on the linear solver: %s",
                    self._inv_Jacobian._solver._info,
                )

            un1 -= incr[0]
            rhon1 -= incr[1]
            metric_matches_rho1 = False

            # Reassemble the density-dependent momentum operator and update
            # the momentum of the current Newton iterate.
            if self._model in ("deltaf", "deltaf_q"):
                rhon1.copy(out=self._tmp_rho_deltaf)
                self._tmp_rho_deltaf += self.projected_equil.n3
                rho1 = self._tmp_rho_deltaf
            else:
                rho1 = rhon1

            self._update_momentum_operator(rho1)
            metric_matches_rho1 = True

            self._momentum_operator.dot(
                un1,
                out=self._tmp_mn1,
            )
            mn1 = self._tmp_mn1

        # ---------------------------------------------------------------
        # Convergence report
        # ---------------------------------------------------------------
        newton_iterations = it + 1

        if not converged:
            logger.warning(
                "Maximum iteration count in VariationalDensityEvolve "
                "reached without convergence:\n"
                "  iterations = %d\n"
                "  err        = %.16e\n"
                "  target     = %.16e",
                newton_iterations,
                err,
                tol_sq,
            )
        elif accepted_by_stagnation:
            logger.debug(
                "Newton iteration accepted after stagnation near the "
                "requested tolerance: iterations=%d, err=%.16e, "
                "target=%.16e.",
                newton_iterations,
                err,
                tol_sq,
            )

        logger.info(
            "Newton iterations: %d",
            newton_iterations,
        )

        # ---------------------------------------------------------------
        # Store the converged increment for the next initial guess
        # ---------------------------------------------------------------
        un1.copy(out=self._tmp_un_diff)
        self._tmp_un_diff -= un

        rhon1.copy(out=self._tmp_rhon_diff)
        self._tmp_rhon_diff -= rhon

        if self._model in ("deltaf", "deltaf_q"):
            rhon1.copy(out=self._tmp_rho_deltaf)
            self._tmp_rho_deltaf += self.projected_equil.n3
            final_metric_rho = self._tmp_rho_deltaf
        else:
            final_metric_rho = rhon1

        if self._with_regularization and not metric_matches_rho1:
            self.mass_ops.invalidate_committed_h1vec_metric()

            self._kinetic_metric.update_weight(
                final_metric_rho,
            )
            metric_matches_rho1 = True

        self.update_feec_variables(rho=rhon1, u=un1)

        if self._with_regularization:
            self.mass_ops.publish_committed_h1vec_metric(
                self._Kdivrho,
                self.variables.rho,
            )
        else:
            self.mass_ops.publish_committed_WMMnew()

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and `CoordinateProjector` needed to compute the bracket term"""

        from struphy.feec.mass import L2Projector
        from struphy.feec.variational_utilities import L2_transport_operator

        # Initialize the transport operator and transposed
        self.divPirho = L2_transport_operator(self.derham)
        self.divPirhoT = self.divPirho.T

        # Inverse mass matrix needed to compute the error
        self.pc_Mv = MassMatrixPreconditioner(
            self.mass_ops.Mv,
        )
        self._inv_Mv = inverse(
            self.mass_ops.Mv,
            "pcg",
            pc=self.pc_Mv,
            tol=1e-10,
            maxiter=1000,
            verbose=False,
            recycle=True,
        )

        integration_grid = [grid_1d.flatten() for grid_1d in self.derham.V0splines.quad_grid_pts[0]]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = (
            self.derham.prepare_eval_tp_fixed(
                integration_grid,
            )
        )

        # tmps
        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])
        self._rhof_values = xp.zeros(grid_shape, dtype=float)

        # Other mass matrices for newton solve
        self._M_drho = self.mass_ops.create_weighted_mass("L2", "L2")

        Jacs = BlockVectorSpace(
            self.derham.Vvpol,
            self.derham.V3pol,
        )

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)

        # local version to avoid creating new version of LinearOperator every time
        self._I3 = IdentityOperator(self.derham.V3pol)

        self._dt_pc_divPirhoT = 2 * (self.divPirhoT)
        self._dt2_pc_divPirhoT = 2 * (self.divPirhoT)
        self._dt2_divPirho = 2 * self.divPirho

        if self._with_regularization:
            # dt * alpha_divdiv * divPirhoT @ B_div_un
            self._dt_alpha_div_term = 2 * (self.divPirhoT @ self._kinetic_evaluator.M_div_un)

            # 2 * alpha_divdiv * B_div_un1.T
            self._alpha_div_rho_term = 2 * (self._kinetic_evaluator.M_div_un1)

        if self._with_regularization:
            self._Jacobian[0, 0] = (
                self._kinetic_metric + self._dt2_pc_divPirhoT @ self._kinetic_evaluator.M_un + self._dt_alpha_div_term
            )

            self._Jacobian[0, 1] = (
                self._kinetic_evaluator.M_un1 + self._alpha_div_rho_term + self._dt_pc_divPirhoT @ self._M_drho
            )
        else:
            self._Jacobian[0, 0] = self._Mrho + self._dt2_pc_divPirhoT @ self._kinetic_evaluator.M_un
            self._Jacobian[0, 1] = self._kinetic_evaluator.M_un1 + self._dt_pc_divPirhoT @ self._M_drho

        self._Jacobian[1, 0] = self._dt2_divPirho
        self._Jacobian[1, 1] = self._I3

        from struphy.linear_algebra.schur_solver import SchurSolverFull

        self._inv_Jacobian = SchurSolverFull(
            self._Jacobian,
            "pbicgstab",
            # pc=self._momentum_inv,
            pc=None,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=self._lin_solver.verbose,
            recycle=False,
        )

        # self._inv_Jacobian = inverse(self._Jacobian,
        #                          'gmres',
        #                          tol=self._lin_solver.tol,
        #                          maxiter=self._lin_solver.maxiter,
        #                          verbose=self._lin_solver.verbose,
        #                          recycle=True)

        # L2-projector for V3
        self._get_L2dofs_V3 = L2Projector("L2", self.mass_ops).get_dofs

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])

        # tmps
        self._eval_dl_drho = xp.zeros(grid_shape, dtype=float)

        self._uf_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._uf1_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]

        self._tmp_int_grid = xp.zeros(grid_shape, dtype=float)
        self._tmp_int_grid2 = xp.zeros(grid_shape, dtype=float)
        self._rhof_values = xp.zeros(grid_shape, dtype=float)
        self._rhof1_values = xp.zeros(grid_shape, dtype=float)

        if self._model == "full":
            self._tmp_de_drho = xp.zeros(grid_shape, dtype=float)
            gam = self._gamma
            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                2 - gam,
            )
            self._proj_rho2_metric_term = deepcopy(metric)

            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                1 - gam,
            )
            self._proj_drho_metric_term = deepcopy(metric)

            if self._linearize:
                self._init_dener_drho = xp.zeros(grid_shape, dtype=float)

        if self._with_regularization:
            self._eval_div_dl_drho = xp.zeros(
                grid_shape,
                dtype=float,
            )

    def _update_Pirho(self, rho):
        """Update the weights of the `BasisProjectionOperator` Pirho"""

        self.divPirho.update_coeffs(rho)
        self.divPirhoT.update_coeffs(rho)

    @profile
    def _update_momentum_operator(
        self,
        rho,
        *,
        update_preconditioner=True,
    ):
        # WMMnew is about to represent a temporary Newton state.
        self.mass_ops.invalidate_committed_WMMnew()

        if self._with_regularization:
            self.mass_ops.invalidate_committed_h1vec_metric()
            self._kinetic_metric.update_weight(rho)
        else:
            self._Mrho.spline_functions["l2_field"].vector = rho
            self._Mrho.assemble()

        if not update_preconditioner:
            return

        if not hasattr(self, "_momentum_inv"):
            return

        pc = self._momentum_inv._options.get("pc")

        if isinstance(pc, H1vecKineticMetricPreconditioner):
            pc.update_metric(self._kinetic_metric)

        elif isinstance(pc, MassMatrixDiagonalPreconditioner):
            pc.update_mass_operator(self._Mrho)

    def _update_linear_form_dl_drho(self, rhon, rhon1, un, un1, sn):
        """Update the linearform representing integration in V3 against kinetic energy"""

        self._kinetic_evaluator.get_u2_grid(un, un1, self._eval_dl_drho)

        if self._with_regularization:
            self._kinetic_evaluator.get_div_u_product_grid(
                un,
                un1,
                self._eval_div_dl_drho,
            )

            self._eval_div_dl_drho *= self._alpha_divdiv
            self._eval_dl_drho += self._eval_div_dl_drho

        self.rhof.vector = rhon
        self.rhof1.vector = rhon1
        if self._model == "barotropic":
            rhof_values = self.rhof.eval_tp_fixed_loc(
                self.integration_grid_spans,
                self.integration_grid_bd,
                out=self._rhof_values,
            )
            rhof1_values = self.rhof1.eval_tp_fixed_loc(
                self.integration_grid_spans,
                self.integration_grid_bd,
                out=self._rhof1_values,
            )

            # self._eval_dl_drho -= (rhof_values + rhof1_values)/2
            rhof_values /= 2
            rhof1_values /= 2

            self._eval_dl_drho -= rhof_values
            self._eval_dl_drho -= rhof1_values

        if self._model == "full":
            self._energy_evaluator.evaluate_discrete_de_drho_grid(rhon, rhon1, sn, out=self._tmp_de_drho)

            self._tmp_int_grid *= 0
            self._tmp_int_grid += self._tmp_de_drho

            if self._linearize:
                self._tmp_int_grid -= self._init_dener_drho
            self._tmp_int_grid *= self._proj_rho2_metric_term

            # self._eval_dl_drho -= self._proj_rho2_metric_term * (self._energy_evaluator._DG_values + de_rhom_s)
            self._eval_dl_drho -= self._tmp_int_grid

        self._get_L2dofs_V3(self._eval_dl_drho, dofs=self._linear_form_dl_drho)

    def _compute_init_linear_form(self):
        if abs(self._gamma - 5 / 3) < 1e-3:
            self._energy_evaluator.evaluate_exact_de_drho_grid(
                self.projected_equil.n3,
                self.projected_equil.s3_monoatomic,
                out=self._init_dener_drho,
            )
        elif abs(self._gamma - 7 / 5) < 1e-3:
            self._energy_evaluator.evaluate_exact_de_drho_grid(
                self.projected_equil.n3,
                self.projected_equil.s3_diatomic,
                out=self._init_dener_drho,
            )
        else:
            raise ValueError("Gamma should be 7/5 or 5/3 for if you want to linearize")

    @profile
    def _get_jacobian(self, dt, rhon, rhon1, un, un1, sn):
        self._kinetic_evaluator.assemble_M_un(un)
        self._kinetic_evaluator.assemble_M_un1(un1)

        if self._with_regularization:
            self._kinetic_evaluator.assemble_M_div_un(un)
            self._kinetic_evaluator.assemble_M_div_un1(un1)

        if self._model == "barotropic":
            self._M_drho = -self.mass_ops.M3 / 2.0

        elif self._model == "full":
            self._energy_evaluator.evaluate_discrete_d2e_drho2_grid(rhon, rhon1, sn, out=self._tmp_int_grid)
            self._tmp_int_grid *= self._proj_drho_metric_term

            self._M_drho.assemble([[self._tmp_int_grid]])

        else:
            self._M_drho.assemble([[0.0 * self._tmp_int_grid]])

        # This way we can update only the scalar multiplying the operator and avoid creating multiple operators
        self._dt_pc_divPirhoT._scalar = dt
        self._dt2_pc_divPirhoT._scalar = dt / 2
        self._dt2_divPirho._scalar = dt / 2

        if self._with_regularization:
            self._dt_alpha_div_term._scalar = dt * self._alpha_divdiv

            self._alpha_div_rho_term._scalar = 2.0 * self._alpha_divdiv

    def _get_error_newton(self, mn_diff, rhon_diff):
        """Error for the newton method : max(|f(0)|,|f(1)|) where f is the function we're trying to nullify"""
        err_u = self._inv_Mv.dot_inner(
            self.derham.boundary_ops["v"].dot(mn_diff),
            mn_diff,
        )
        err_rho = self.mass_ops.M3.dot_inner(rhon_diff, rhon_diff)
        return max(err_rho, err_u)
