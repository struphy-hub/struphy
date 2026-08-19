import logging
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
    MassMatrixDiagonalPreconditioner,
)
from struphy.feec.variational_utilities import H1vecKineticMetric, Hdiv0_transport_operator
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolverFull
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VariationalMagFieldEvolve(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf u \in (H^1)^3` and :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x - \int_\Omega \mathbf B \cdot \nabla \times (\mathbf \tilde{B} \times \mathbf v) \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3\,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \cdot ( \mathbf \tilde{B} \times \mathbf u ) = 0 \,.


    Where :math:`\tilde{\mathbf B}` is either :math:`\mathbf B` for full-f models, :math:`\mathbf B_0` for linear models or :math:`\mathbf B_0+\mathbf B` for :math:`\delta f` models.

    On the logical domain:

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta
        - \int_{\hat{\Omega}} \hat{\mathbf{B}}^2 \cdot G \,\nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{v}}) \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{\mathbf{B}}^2 + \nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{u}}) = 0 ~ .
        \end{align}

    It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t}
        - (\mathbb C \hat{\Pi}^{1}[B_h^{n+1}} \cdot \vec{\boldsymbol \Lambda}^v])^\top \mathbb M^2 B^{n+\frac{1}{2}} \big) = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf b^{n+1}- \mathbf b^n}{\Delta t} + \mathbb C \hat{\Pi}^{1}[\hat{B_h^{n}} \cdot \vec{\boldsymbol \Lambda}^v]] \mathbf u^{n+1/2} = 0 ~ ,
        \end{align}

    where weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{B}}_h^{n+1/2} = (\mathbf{b}^{n+1/2})^\top \vec{\boldsymbol \Lambda}^2 \in V_h^2 \, \qquad \hat{\rho}_h^{n} = (\boldsymbol \rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,.

    """

    class Variables:
        """Container for variables advanced by :class:`VariationalMagFieldEvolve`.

        Attributes
        ----------
        u : FEECVariable
            Velocity variable in ``"H1vec"`` space.
        b : FEECVariable
            Magnetic-field variable in ``"Hdiv"`` space.
        """

        def __init__(self):
            self._u: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1vec"
            self._u = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(self, rho: FEECVariable = None):
        """
        Parameters
        ----------
        rho : FEECVariable, default=None
            Density variable used to construct the density-weighted momentum
            metric.
        """
        self.variables = self.Variables()
        self.rho = rho

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`VariationalMagFieldEvolve`.

        Parameters
        ----------
        model : {"full", "full_p", "linear"}, default="full"
            Magnetic-field evolution model variant.
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Linear solver for implicit substeps.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner used in linear solves.
        solver_params : SolverParameters, default=None
            Linear-solver controls.
        nonlin_solver : NonlinearSolverParameters, default=None
            Nonlinear iteration controls.
        with_regularization : bool, default=False
            Whether to use the density-weighted H1vec div-div kinetic
            regularization.
        alpha_divdiv : float, default=1.0
            Coefficient of

                alpha_divdiv * int rho * (div u)^2 dx

            in the kinetic energy. The corresponding coefficient in the
            momentum metric is ``2 * alpha_divdiv``.
        """

        OptsModel = Literal["full", "full_p", "linear"]
        # propagator options
        model: OptsModel = "full"
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
                self.nonlin_solver = NonlinearSolverParameters(type="Newton")
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
        self._model = self.options.model
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize = self._nonlin_solver.linearize
        self._with_regularization = self.options.with_regularization
        self._alpha_divdiv = self.options.alpha_divdiv

        # alpha_divdiv is the coefficient in the kinetic energy. Differentiating
        # with respect to velocity produces a factor of two in the momentum metric.
        self._metric_alpha = 2.0 * self._alpha_divdiv

        if self._with_regularization and not self.domain.has_exact_mapping_hessian:
            raise NotImplementedError(
                "VariationalMagFieldEvolve regularization requires an "
                "analytical mapping Hessian. "
                f"Mapping {type(self.domain).__name__} with "
                f"kind_map={self.domain.kind_map} is not currently supported."
            )

        self._info = self._nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        # Density-weighted H1vec mass operator. Its assembly normally happens in
        # VariationalDensityEvolve, but the magnetic propagator explicitly updates
        # it below so that it does not depend on propagator ordering.
        self._Mrho = self.mass_ops.WMMnew

        if self._with_regularization:
            if self.rho is None:
                raise ValueError("VariationalMagFieldEvolve requires rho when with_regularization=True.")

            self._kinetic_metric = self.mass_ops.get_h1vec_kinetic_metric(
                self._metric_alpha,
            )

            self._Kdivrho = self._kinetic_metric.divdiv_operator

            # Assemble M_rho and K_div,rho with the current density.
            self._kinetic_metric.update_weight(
                self.rho.spline.vector,
            )

            self._kinetic_metric.update_weight_if_needed(
                self.rho.spline.vector,
                self.rho.generation,
            )

            self._momentum_operator = self._kinetic_metric
            # pc = H1vecKineticMetricWoodburyPreconditioner(
            #                     self._kinetic_metric,
            #                     auxiliary_nsteps=1,
            #                     spectral_iterations=4,
            #                     spectral_safety=1.5,
            #     )
            pc = H1vecKineticMetricPreconditioner(self._kinetic_metric)
        else:
            self._Kdivrho = None
            self._kinetic_metric = None
            self._momentum_operator = self._Mrho

            # Preserve compatibility with models which do not pass rho explicitly.
            if self.rho is not None:
                self._Mrho.spline_functions["l2_field"].vector = self.rho.spline.vector
                self._Mrho.assemble()

            pc = MassMatrixDiagonalPreconditioner(self._Mrho)

        self._momentum_pc = pc

        # Projector
        self._initialize_projectors_and_mass()

        # bunch of temporaries to avoid allocating in the loop
        u = self.variables.u.spline.vector
        b = self.variables.b.spline.vector

        self._tmp_un1 = u.space.zeros()
        self._tmp_un2 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_bn1 = b.space.zeros()
        self._tmp_bn12 = b.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_bn_diff = b.space.zeros()
        self._tmp_un_weak_diff = u.space.zeros()
        self._tmp_bn_weak_diff = b.space.zeros()

        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_advection2 = u.space.zeros()
        self._tmp_b_advection = b.space.zeros()
        self._linear_form_dl_db = b.space.zeros()

        if self._linearize:
            self._extracted_b2 = self.derham.extraction_ops["2"].dot(self.projected_equil.b2)

    def __call__(self, dt):
        self._update_momentum_operator()
        self.__call_newton(dt)

    def _update_momentum_operator(self):
        """Update the fixed-density momentum metric and preconditioner."""

        if self.rho is None:
            raise ValueError("VariationalMagFieldEvolve requires a density variable.")

        if self._with_regularization:
            changed = self._kinetic_metric.update_weight_if_needed(
                self.rho.spline.vector,
                self.rho.generation,
            )
        else:
            generation = self.rho.generation
            changed = getattr(self, "_rho_generation", None) != generation

            if changed:
                self._Mrho.spline_functions["l2_field"].vector = self.rho.spline.vector
                self._Mrho.assemble()
                self._rho_generation = generation

        if not changed:
            return

        pc = self._momentum_pc

        if isinstance(
            pc,
            H1vecKineticMetricPreconditioner,
        ):
            if not self._with_regularization:
                raise TypeError("H1vecKineticMetricPreconditioner requires the regularized kinetic metric.")

            pc.update_metric(self._kinetic_metric)

        elif isinstance(
            pc,
            H1vecKineticMetricWoodburyPreconditioner,
        ):
            if not self._with_regularization:
                raise TypeError("H1vecKineticMetricWoodburyPreconditioner requires the regularized kinetic metric.")

            pc.update_metric(self._kinetic_metric)

        elif isinstance(
            pc,
            MassMatrixDiagonalPreconditioner,
        ):
            if self._with_regularization:
                raise TypeError("MassMatrixDiagonalPreconditioner cannot represent the regularized kinetic metric.")

            pc.update_mass_operator(self._Mrho)

    def __call_newton(self, dt):
        """Advance magnetic field and velocity using Newton iteration."""

        if self._info:
            logger.info("")
            logger.info(
                "Newton iteration in VariationalMagFieldEvolve",
            )

        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        # The transport operator uses the magnetic field at the beginning
        # of the substep.
        self._update_Pib(bn)

        self._momentum_operator.dot(
            un,
            out=self._tmp_mn,
        )
        mn = self._tmp_mn

        # Recycle the previous converged increment as initial guess.
        bn.copy(out=self._tmp_bn1)
        self._tmp_bn1 += self._tmp_bn_diff
        bn1 = self._tmp_bn1

        un.copy(out=self._tmp_un1)
        self._tmp_un1 += self._tmp_un_diff
        un1 = self._tmp_un1

        self._momentum_operator.dot(
            un1,
            out=self._tmp_mn1,
        )
        mn1 = self._tmp_mn1

        tol = float(self._nonlin_solver.tol)
        tol_sq = tol * tol

        # _get_error_newton returns a squared norm. Account for roundoff
        # and for the inexact inner linear solves.
        absolute_threshold = 4.0 * tol_sq

        # Optional stagnation acceptance close to the requested tolerance.
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
        schur_calls = 0

        for it in range(self._nonlin_solver.maxiter):
            # Midpoint magnetic field.
            bn.copy(out=self._tmp_bn12)
            self._tmp_bn12 += bn1
            self._tmp_bn12 *= 0.5

            # Midpoint velocity.
            un.copy(out=self._tmp_un12)
            self._tmp_un12 += un1
            self._tmp_un12 *= 0.5

            # Update M2 * B^(n+1/2).
            self._update_linear_form_dl_db()

            # Compute the coupled advection terms.
            if self._model == "linear":
                self.curlPibT0.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                self.curlPibT.dot(
                    self._linear_form_dl_db0,
                    out=self._tmp_advection2,
                )

                self._tmp_advection += self._tmp_advection2

                self.curlPib0.dot(
                    self._tmp_un12,
                    out=self._tmp_b_advection,
                )
            else:
                self.curlPibT.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                self.curlPib.dot(
                    self._tmp_un12,
                    out=self._tmp_b_advection,
                )

            self._tmp_advection *= dt
            self._tmp_b_advection *= dt

            # Magnetic residual.
            bn1.copy(out=self._tmp_bn_diff)
            self._tmp_bn_diff -= bn
            self._tmp_bn_diff += self._tmp_b_advection
            bn_diff = self._tmp_bn_diff

            # Momentum residual.
            mn1.copy(out=self._tmp_mn_diff)
            self._tmp_mn_diff -= mn
            self._tmp_mn_diff += self._tmp_advection
            mn_diff = self._tmp_mn_diff

            err = float(
                self._get_error_newton(
                    mn_diff,
                    bn_diff,
                )
            )

            if not bool(xp.isfinite(err)):
                raise FloatingPointError(
                    f"Non-finite residual in VariationalMagFieldEvolve: iteration={it + 1}, err={err}."
                )

            if err0 is None:
                err0 = max(err, tiny)

            relative_err = err / err0

            if self._info:
                logger.info(
                    "Magnetic Newton iteration: %d, error: %.16e, relative error: %.16e",
                    it + 1,
                    err,
                    relative_err,
                )

            # The error is a squared norm, so the targets are tol**2.
            if err <= absolute_threshold or relative_err <= tol_sq:
                converged = True
                break

            # Accept roundoff/inexact-solve stagnation only when the
            # residual is already close to the absolute target.
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

            # Update scalar factors in the Jacobian.
            self._get_jacobian(dt)

            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = bn_diff

            self._inv_Jacobian.dot(
                self._tmp_f,
                out=self._tmp_incr,
            )
            schur_calls += 1

            if self._info:
                logger.info(
                    "Magnetic Schur linear solver info: %r",
                    getattr(
                        self._inv_Jacobian._solver,
                        "_info",
                        None,
                    ),
                )

            un1 -= self._tmp_incr[0]
            bn1 -= self._tmp_incr[1]

            # The density is fixed in this substep, so no momentum-metric
            # reassembly is necessary inside Newton.
            self._momentum_operator.dot(
                un1,
                out=self._tmp_mn1,
            )
            mn1 = self._tmp_mn1

        newton_iterations = it + 1

        if not converged:
            logger.warning(
                "Maximum iteration count in "
                "VariationalMagFieldEvolve reached without convergence:\n"
                "  iterations  = %d\n"
                "  Schur calls = %d\n"
                "  err         = %.16e\n"
                "  target      = %.16e",
                newton_iterations,
                schur_calls,
                err,
                tol_sq,
            )
        elif accepted_by_stagnation and self._info:
            logger.info(
                "Accepted stagnated magnetic Newton residual: iterations=%d, Schur calls=%d, err=%.16e, target=%.16e.",
                newton_iterations,
                schur_calls,
                err,
                tol_sq,
            )

        if self._info:
            logger.info(
                "Magnetic Newton iterations: %d, Schur calls: %d",
                newton_iterations,
                schur_calls,
            )

        # Save the converged increments without allocating new vectors or
        # replacing the temporary-vector references.
        un1.copy(out=self._tmp_un_diff)
        self._tmp_un_diff -= un

        bn1.copy(out=self._tmp_bn_diff)
        self._tmp_bn_diff -= bn

        self.update_feec_variables(
            b=bn1,
            u=un1,
        )

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        self.curlPib = Hdiv0_transport_operator(self.derham)
        self.curlPibT = self.curlPib.T

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

        Jacs = BlockVectorSpace(
            self.derham.Vvpol,
            self.derham.V2pol,
        )

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)

        self._I2 = IdentityOperator(self.derham.V2pol)

        if self._model == "linear":
            # initialize the jacobian differently if linear model
            self._create_Pib0()

            self._linear_form_dl_db0 = self.mass_ops.M2.dot(self.projected_equil.b2)

            self._mdt2_pc_curlPibT_M = 2 * (self.curlPibT0 @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self.curlPib0

        else:
            self._mdt2_pc_curlPibT_M = 2 * (self.curlPibT @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self.curlPib

        # local version to avoid creating new version of LinearOperator every time

        self._Jacobian[0, 0] = self._momentum_operator
        self._Jacobian[0, 1] = self._mdt2_pc_curlPibT_M
        self._Jacobian[1, 0] = self._dt2_curlPib
        self._Jacobian[1, 1] = self._I2

        self._inv_Jacobian = SchurSolverFull(
            self._Jacobian,
            self.options.solver,
            pc=self._momentum_pc,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=self._lin_solver.verbose,
            recycle=True,
        )

        # self._inv_Jacobian = inverse(self._Jacobian,
        #                          'gmres',
        #                          tol=self._lin_solver['tol'],
        #                          maxiter=self._lin_solver['maxiter'],
        #                          verbose=self._lin_solver['verbose'],
        #                          recycle=True)

    def _update_Pib(self, b):
        """Update the weights of the `BasisProjectionOperator`"""

        self.curlPib.update_coeffs(b)
        self.curlPibT.update_coeffs(b)

    def _create_Pib0(self):
        self.curlPib0 = Hdiv0_transport_operator(self.derham)
        self.curlPibT0 = self.curlPib0.T

        self.curlPib0.update_coeffs(self.projected_equil.b2)
        self.curlPibT0.update_coeffs(self.projected_equil.b2)

    def _update_linear_form_dl_db(self):
        """Update the linearform representing integration in V2 derivative of the lagrangian"""
        if self._linearize:
            wb = self.mass_ops.M2.dot(self._tmp_bn12 - self._extracted_b2, out=self._linear_form_dl_db)
        else:
            wb = self.mass_ops.M2.dot(self._tmp_bn12, out=self._linear_form_dl_db)
        wb *= -1

    def _get_error_newton(self, mn_diff, bn_diff):
        self.derham.boundary_ops["v"].dot(
            mn_diff,
            out=self._tmp_un_weak_diff,
        )

        err_u = self._inv_Mv.dot_inner(
            self._tmp_un_weak_diff,
            mn_diff,
        )

        err_b = self.mass_ops.M2.dot_inner(
            bn_diff,
            bn_diff,
        )

        return max(err_b, err_u)

    def _get_jacobian(self, dt):
        self._mdt2_pc_curlPibT_M._scalar = -dt / 2
        self._dt2_curlPib._scalar = dt / 2
