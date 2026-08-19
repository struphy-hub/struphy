import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.feec.basis_projection_ops import CoordinateProjector
from struphy.feec.mass import L2Projector
from struphy.feec.preconditioner import H1vecKineticMetricPreconditioner, H1vecKineticMetricWoodburyPreconditioner, MassMatrixDiagonalPreconditioner
from struphy.feec.variational_utilities import InternalEnergyEvaluator
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VariationalViscosity(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`s \in L^2` and  :math:`\mathbf u \in (H^1)^3` such that

    .. math::

        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x + \int_\Omega (\mu + \mu_a(\mathbf x)) \nabla \mathbf u : \nabla \mathbf v \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3 \,,
        \\[4mm]
        &\int_\Omega \frac{\partial \mathcal U}{\partial s} \partial_t s \, q \,\textrm d \mathbf x - \mu \int_\Omega |\nabla \mathbf u|^2 \, q \,\textrm d \mathbf x = 0 \qquad \forall \, q \in L^2\,\text{if using } s,
        \\[4mm]
        &\int_\Omega \frac{1}{\gamma - 1} \partial_t p \, q\,\textrm d \mathbf x - \mu \int_\Omega |\nabla \mathbf u|^2 \, q \,\textrm d \mathbf x = 0 \qquad \forall \, q \in L^2\, \text{if using } p.

    With :math:`\mu_a(\mathbf x) = \mu_a |\nabla \mathbf u(\mathbf x)|`

    On the logical domain:

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta
        + \mu \int_{\hat{\Omega}} \nabla (DF \hat{\mathbf{u}}) : \nabla (DF \hat{\mathbf{v}}) \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\int_{\hat{\Omega}} \partial_t (\hat{\rho} \hat{e}(\hat{\rho}, \hat{s})) \hat{w} \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta -  \int_{\hat{\Omega}} (\mu + \mu_a(\boldsymbol \eta)) \nabla (DF \hat{\mathbf{u}}) : \nabla (DF \hat{\mathbf{u}}) \hat{w} \, \textrm d \boldsymbol \eta = 0 ~ , \text{if using } s,
        \\[2mm]
        &\int_{\hat{\Omega}} \partial_t (\frac{1}{\gamma -1} \hat{p} ) \hat{w} \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta - \int_{\hat{\Omega}} (\mu + \mu_a(\boldsymbol \eta)) \nabla (DF \hat{\mathbf{u}}) : \nabla (DF \hat{\mathbf{u}}) \hat{w} \, \textrm d \boldsymbol \eta = 0 ~, \text{if using } p.
        \end{align}

    It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t}
        +  \sum_\nu (\mathbb G \mathcal{X}^v_\nu)^T (\mu \mathbb M_0 + \mu_a \mathbb M_0[|\nabla u|] \mathbb G \mathcal{X}^v_\nu \mathbf u^{n+1} = 0 ~ ,
        \\[2mm]
        &\frac{P^{3}(\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n},\hat{s}_h^{n}))- P^{3}(\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n},\hat{s}_h^{n+1}))}{\Delta t} - \mu P^3(\sum_\nu DF \mathcal{X}^v_\nu \frac{ \mathbf u^{n+1}+\mathbf u^n}{2} \cdot DF \mathcal{X}^v_\nu \mathbf u^{n+1}) = 0 ~ , \text{if using } s,
        \\[2mm]
        &\frac{1}{\gamma -1}\frac{p^{n+1}- p^{n}}{\Delta t} - \mu P^3(\sum_\nu DF \mathcal{X}^v_\nu \frac{ \mathbf u^{n+1}+\mathbf u^n}{2} \cdot DF \mathcal{X}^v_\nu \mathbf u^{n+1}) = 0 ~ , \text{if using } p.
        \end{align}

    where $P^3$ denotes the $L^2$ projection in the last space of the de Rham sequence and the weights in :math:`\mathbb M_0[|\nabla u|]` are given by

    .. math::
        P^0(g \sqrt{\sum_\nu |(\mathbb G \mathcal{X}^v_\nu \mathbb u)^\top \vec{\boldsymbol \Lambda}^0 |^2]})^\top \vec{\boldsymbol \Lambda}^0 ~.

    """

    class Variables:
        """Container for variables advanced by :class:`VariationalViscosity`.

        Attributes
        ----------
        s : FEECVariable
            Thermodynamic scalar variable in ``"L2"`` space.
        u : FEECVariable
            Velocity variable in ``"H1vec"`` space.
        """

        def __init__(self):
            self._s: FEECVariable = None
            self._u: FEECVariable = None

        @property
        def s(self) -> FEECVariable:
            return self._s

        @s.setter
        def s(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._s = new

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1vec"
            self._u = new

    def __init__(self, rho: FEECVariable = None, pt3: FEECVariable = None):
        """
        Parameters
        ----------
        rho : FEECVariable, default=None
            Mass density 3-form (``"L2"`` space) weighting the velocity mass matrix.
        pt3 : FEECVariable, default=None
            Pressure or entropy 3-form (``"L2"`` space) evolved alongside the velocity.
            If ``None``, the thermodynamic equation is skipped.
        """
        self.variables = self.Variables()
        self.rho = rho
        self.pt3 = pt3

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`VariationalViscosity`.

        Parameters
        ----------
        model : {"full", "full_p", "full_q", "linear_p", "linear_q", "deltaf_q"}, default="full"
            Thermodynamic model variant.
        gamma : float, default=5/3
            Adiabatic index.
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Linear solver for implicit subproblems.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixDiagonalPreconditioner"
            Preconditioner used in linear solves.
        solver_params : SolverParameters, default=None
            Linear-solver controls.
        nonlin_solver : NonlinearSolverParameters, default=None
            Nonlinear iteration controls.
        mu : float, default=0.0
            Physical viscosity coefficient.
        mu_a : float, default=0.0
            Artificial-viscosity coefficient.
        alpha : float, default=0.0
            Optional linear damping/regularization parameter.
                with_regularization : bool, default=False
            Whether to use the density-weighted H1vec div-div kinetic
            regularization.
        alpha_divdiv : float, default=1.0
            Coefficient of the regularization term

                alpha_divdiv * int rho (div u)^2 dx

            in the kinetic Hamiltonian. The corresponding coefficient in the
            momentum metric is ``2 * alpha_divdiv``.
        """

        # specific literals
        OptsModel = Literal["full", "full_p", "full_q", "linear_p", "linear_q", "deltaf_q"]
        # propagator options
        model: OptsModel = "full"
        gamma: float = 5.0 / 3.0
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixDiagonalPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        mu: float = 0.0
        mu_a: float = 0.0
        alpha: float = 0.0
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
                raise TypeError(
                    "with_regularization must be a bool, "
                    f"got {type(self.with_regularization)}."
                )

            if self.alpha_divdiv < 0.0:
                raise ValueError(
                    "alpha_divdiv must be non-negative, "
                    f"got {self.alpha_divdiv}."
                )

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
        self._gamma = self.options.gamma
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._mu_a = self.options.mu_a
        self._alpha = self.options.alpha
        self._mu = self.options.mu
        self._with_regularization = self.options.with_regularization
        self._alpha_divdiv = self.options.alpha_divdiv

        # alpha_divdiv is the coefficient in the kinetic Hamiltonian.
        # Its Hessian with respect to velocity contains the factor two.
        self._metric_alpha = 2.0 * self._alpha_divdiv

        if self._model == "full" and self.rho is None:
            raise ValueError(
                "VariationalViscosity with model='full' requires rho."
            )

        if self._with_regularization and not self.domain.has_exact_mapping_hessian:
            raise NotImplementedError(
                "VariationalViscosity regularization requires an "
                "analytical mapping Hessian. "
                f"Mapping {type(self.domain).__name__} with "
                f"kind_map={self.domain.kind_map} is not currently supported."
            )

        self._info = self._nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        # Density-weighted momentum metric. Do not rely on another propagator
        # having assembled WMMnew with the current density.
        self._Mrho = self.mass_ops.WMMnew
        rho = self.rho.spline.vector

        if self._with_regularization:
            self._kinetic_metric = self.mass_ops.get_h1vec_kinetic_metric(
                self._metric_alpha,
            )
            self._Kdivrho = self._kinetic_metric.divdiv_operator

            # Assemble both M_rho and K_div,rho with the current density.
            self._kinetic_metric.update_weight(rho)

            self._momentum_operator = self._kinetic_metric
            self._momentum_pc = H1vecKineticMetricPreconditioner(
                self._kinetic_metric,
            )
        else:
            self._kinetic_metric = None
            self._Kdivrho = None

            self._Mrho.spline_functions["l2_field"].vector = rho
            self._Mrho.assemble()

            self._momentum_operator = self._Mrho
            self._momentum_pc = MassMatrixDiagonalPreconditioner(
                self._Mrho,
            )

        # Femfields for the projector
        self.sf = self.derham.create_spline_function("sf", "L2")
        self.sf1 = self.derham.create_spline_function("sf1", "L2")
        self.uf1 = self.derham.create_spline_function("uf", "H1vec")
        self.uf12 = self.derham.create_spline_function("uf1", "H1vec")
        self.gu0f = self.derham.create_spline_function("gu0", "Hcurl")
        self.gu1f = self.derham.create_spline_function("gu1", "Hcurl")
        self.gu2f = self.derham.create_spline_function("gu2", "Hcurl")
        self.gu120f = self.derham.create_spline_function("gu120", "Hcurl")
        self.gu121f = self.derham.create_spline_function("gu121", "Hcurl")
        self.gu122f = self.derham.create_spline_function("gu122", "Hcurl")

        # Projector
        self._energy_evaluator = InternalEnergyEvaluator(self.derham, self._gamma)
        self._initialize_projectors_and_mass()

        # bunch of temporaries to avoid allocating in the loop
        u = self.variables.u.spline.vector
        s = self.variables.s.spline.vector

        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_sn1 = s.space.zeros()
        self._tmp_sn_incr = s.space.zeros()
        self._tmp_sn_weak_diff = s.space.zeros()
        self._tmp_gu0 = self.derham.V1pol.zeros()
        self._tmp_gu1 = self.derham.V1pol.zeros()
        self._tmp_gu2 = self.derham.V1pol.zeros()
        self._tmp_gu120 = self.derham.V1pol.zeros()
        self._tmp_gu121 = self.derham.V1pol.zeros()
        self._tmp_gu122 = self.derham.V1pol.zeros()
        self._linear_form_tot_e = s.space.zeros()
        self._linear_form_en1 = s.space.zeros()
        self.tot_rhs = s.space.zeros()
    @profile
    def __call__(self, dt):
        rho = self.rho.spline.vector
        self._update_momentum_operator(rho)
        if self._nonlin_solver.type == "Newton":
            self.__call_newton(dt)
        else:
            raise ValueError(
                "wrong value for solver type in VariationalViscosity",
            )

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        # Compute dissipation implicitely
        sn = self.variables.s.spline.vector
        un = self.variables.u.spline.vector

        if self._mu < 1.0e-15 and self._mu_a < 1.0e-15 and self._alpha < 1.0e-15:
            self.update_feec_variables(s=sn, u=un)
            return

        if self._info:
            logger.info("")
            logger.info("Computing the dissipation in VariationalViscosity")

        # Update artificial viscosity weighted mass matrix
        total_viscosity = self._update_artificial_viscosity(un, dt)

        self._scaled_stiffness._scalar = dt * self._mu  # /2.
        self._scaled_Mv._scalar = dt * self._alpha
        # self.evol_op._multiplicants[1]._addends[0]._scalar = - dt*self._mu/2.
        un1 = self.evol_op.dot(un, out=self._tmp_un1)
        if self._info:
            logger.info(f"information on the linear solver : {self.inv_lop._info}")

        if self._model == "linear_p" or (self._model == "linear_q" and self._nonlin_solver["fast"]):
            self.update_feec_variables(s=sn, u=un1)
            return

        # Energy balance term
        # 1) Pointwize energy change
        energy_change = self._get_energy_change(un, un1, dt, total_viscosity)
        # 2) Initial energy and linear form
        rho = self.rho.spline.vector
        if self._model in ["deltaf_q", "linear_q"]:
            self.sf.vector = self.pt3.spline.vector
        else:
            self.sf.vector = sn

        sf_values = self.sf.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_bd,
            out=self._sf_values,
        )

        if self._model == "full":
            
            rhof_values = self._energy_evaluator.eval_3form(rho, out=self._rhof_values)

            e_n = self._energy_evaluator.ener(
                rhof_values,
                sf_values,
                out=self._e_n,
            )

            e_n *= self._energy_metric

        elif self._model == "full_p":
            e_n = self._e_n
            e_n *= 0.0
            e_n += sf_values
            e_n *= 1.0 / (self._gamma - 1.0)
            e_n *= self._energy_metric

        elif self._model in ["full_q"]:
            e_n = self._e_n
            e_n *= 0.0
            e_n += sf_values
            e_n **= 2
            e_n *= 1.0 / (self._gamma - 1.0)
            e_n *= self._energy_metric

        elif self._model in ["linear_q", "deltaf_q"]:
            e_n = self._e_n
            e_n *= 0.0
            e_n += sf_values
            e_n *= self._q0_values
            e_n *= 2.0 / (self._gamma - 1.0)
            e_n *= self._energy_metric

        energy_change += e_n

        self._get_L2dofs_V3(energy_change, dofs=self._linear_form_tot_e)

        # 3) Newton iteration
        sn1 = sn.copy(out=self._tmp_sn1)

        tol = float(self._nonlin_solver.tol)
        tol_sq = tol * tol
        
        acceptance_factor = 4.0
        absolute_threshold = acceptance_factor * tol_sq
        
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

        for it in range(self._nonlin_solver.maxiter):
            if self._model in ["deltaf_q", "linear_q"]:
                self.sf1.vector = self.pt3.spline.vector
            else:
                self.sf1.vector = sn1

            sf1_values = self.sf1.eval_tp_fixed_loc(
                self.integration_grid_spans,
                self.integration_grid_bd,
                out=self._sf1_values,
            )

            if self._model == "full":
                e_n1 = self._energy_evaluator.ener(
                    rhof_values,
                    sf1_values,
                    out=self._e_n1,
                )
                e_n1 *= self._energy_metric

            elif self._model == "full_p":
                e_n1 = self._e_n1
                e_n1 *= 0.0
                e_n1 += sf1_values
                e_n1 *= 1.0 / (self._gamma - 1.0)
                e_n1 *= self._energy_metric

            elif self._model in ["full_q"]:
                e_n1 = self._e_n1
                e_n1 *= 0.0
                e_n1 += sf1_values
                e_n1 **= 2
                e_n1 *= 1.0 / (self._gamma - 1.0)
                e_n1 *= self._energy_metric

            elif self._model in ["linear_q", "deltaf_q"]:
                e_n1 = self._e_n1
                e_n1 *= 0.0
                e_n1 += sf1_values
                e_n1 *= self._q0_values
                e_n1 *= 2.0 / (self._gamma - 1.0)
                e_n1 *= self._energy_metric

            self._get_L2dofs_V3(e_n1, dofs=self._linear_form_en1)

            self.tot_rhs *= 0.0
            self.tot_rhs -= self._linear_form_en1
            self.tot_rhs += self._linear_form_tot_e

            err = float(self._get_error_newton(self.tot_rhs))

            if not bool(xp.isfinite(err)):
                raise FloatingPointError(
                    "Non-finite residual in VariationalViscosity: "
                    f"iteration={it + 1}, err={err}."
                )
        
            if err0 is None:
                err0 = max(err, tiny)
        
            relative_err = err / err0
        
            if self._info:
                logger.info(
                    "Viscosity iteration: %d, error: %.16e, relative error: %.16e",
                    it + 1,
                    err,
                    relative_err,
                )
        
            # _get_error_newton returns a squared norm.
            if err <= absolute_threshold or relative_err <= tol_sq:
                converged = True
                break
        
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
        
                if (
                    stagnation_count >= stagnation_iterations
                    and err <= stagnation_threshold
                ):
                    converged = True
                    accepted_by_stagnation = True
                    break
        
            previous_err = err

            if self._model == "full":
                deds = self._energy_evaluator.dener_ds(
                    rhof_values,
                    sf1_values,
                    out=self._de_s1_values,
                )
                deds *= self._mass_metric_term

                self.M_de_ds.assemble([[deds]])
                self.pc_jac.update_mass_operator(self.M_de_ds)

            elif self._model in ["full_q", "linear_q", "deltaf_q"]:
                if self._model in ["deltaf_q", "linear_q"]:
                    sf1_values = self._q0_values

                deds = self._de_s1_values
                deds *= 0.0
                deds += sf1_values
                deds *= 2 / (self._gamma - 1.0)
                deds *= self._mass_metric_term

                self.M_de_ds.assemble([[deds]])
                self.pc_jac.update_mass_operator(self.M_de_ds)

            incr = self.inv_jac.dot(self.tot_rhs, out=self._tmp_sn_incr)

            if self._info:
                logger.info(f"information on the linear solver : {self.inv_jac._info}")

            if self._model in ["deltaf_q", "linear_q"]:
                self.pt3 += incr
            else:
                sn1 += incr

        if it == self._nonlin_solver.maxiter - 1 or xp.isnan(err):
            logger.info(
                f"!!!Warning: Maximum iteration in VariationalViscosity reached - not converged:\n {err =} \n {tol**2 =}",
            )

        self.update_feec_variables(s=sn1, u=un1)

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        Xv = getattr(self.basis_ops, "Xv")
        Pcoord0 = CoordinateProjector(
            0,
            self.derham.Vvpol,
            self.derham.V0pol,
        )
        Pcoord1 = CoordinateProjector(
            1,
            self.derham.Vvpol,
            self.derham.V0pol,
        )
        Pcoord2 = CoordinateProjector(
            2,
            self.derham.Vvpol,
            self.derham.V0pol,
        )

        M1 = self.mass_ops.M1
        self.M1_du = self.mass_ops.create_weighted_mass("Hcurl", "Hcurl")

        self.pc_M3 = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.M3,
        )
        self._inv_M3 = inverse(
            self.mass_ops.M3,
            "pcg",
            pc=self.pc_M3,
            tol=1e-10,
            maxiter=1000,
            verbose=False,
        )

        self.M_de_ds = self.mass_ops.create_weighted_mass("L2", "L2")

        if self.options.precond is None:
            self.pc_jac = None
        else:
            pc_class = getattr(
                preconditioner,
                self.options.precond,
            )
            self.pc_jac = pc_class(self.M_de_ds)

        self.inv_jac = inverse(
            self.M_de_ds,
            "pcg",
            pc=self.pc_jac,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=False,
            recycle=True,
        )

        grad = self.derham.grad_bcfree
        self.scalar_stiffness = grad.T @ M1 @ grad
        self.log_stiffness = (
            Pcoord0.T @ self.scalar_stiffness @ Pcoord0
            + Pcoord1.T @ self.scalar_stiffness @ Pcoord1
            + Pcoord2.T @ self.scalar_stiffness @ Pcoord2
        )

        self.phy_stiffness = Xv.T @ self.log_stiffness @ Xv

        self._scaled_stiffness = 0.00001 * self.phy_stiffness

        self.du_stiffness = grad.T @ self.M1_du @ grad
        self.du_log_stiffness = (
            Pcoord0.T @ self.du_stiffness @ Pcoord0
            + Pcoord1.T @ self.du_stiffness @ Pcoord1
            + Pcoord2.T @ self.du_stiffness @ Pcoord2
        )

        self.du_phy_stiffness = Xv.T @ self.du_log_stiffness @ Xv

        self._scaled_Mv = 0.1 * self.mass_ops.Mv

        # The regularization is part of the conservative kinetic metric, so it
        # must appear on both sides of the velocity update.
        self.r_op = self._momentum_operator
        self.l_op = self._momentum_operator+ self._scaled_Mv+ self._scaled_stiffness+ self.du_phy_stiffness

        self.grad_0 = grad @ Pcoord0 @ Xv
        self.grad_1 = grad @ Pcoord1 @ Xv
        self.grad_2 = grad @ Pcoord2 @ Xv

        self.inv_lop = inverse(
            self.l_op,
            "pcg",
            pc=self._momentum_pc,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=False,
            recycle=True,
        )

        self.evol_op = self.inv_lop @ self.r_op
        # self.evol_op = IdentityOperator(self.derham.Vvpol)
        integration_grid = [grid_1d.flatten() for grid_1d in self.derham.V3splines.quad_grid_pts[0]]
        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = (
            self.derham.prepare_eval_tp_fixed(
                integration_grid,
            )
        )

        self.integration_grid_gradient = [
            [self.integration_grid_bd[0], self.integration_grid_bn[1], self.integration_grid_bn[2]],
            [
                self.integration_grid_bn[0],
                self.integration_grid_bd[1],
                self.integration_grid_bn[2],
            ],
            [self.integration_grid_bn[0], self.integration_grid_bn[1], self.integration_grid_bd[2]],
        ]

        self.integration_grid_u = [
            [self.integration_grid_bn[0], self.integration_grid_bn[1], self.integration_grid_bn[2]],
            [
                self.integration_grid_bn[0],
                self.integration_grid_bn[1],
                self.integration_grid_bn[2],
            ],
            [self.integration_grid_bn[0], self.integration_grid_bn[1], self.integration_grid_bn[2]],
        ]

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])

        self._guf0_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._guf1_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._guf2_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]

        self._guf120_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._guf121_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._guf122_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]

        self._uf1_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._uf12_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]

        self._gu_sq_values = xp.zeros(grid_shape, dtype=float)
        self._u_sq_values = xp.zeros(grid_shape, dtype=float)
        self._gu_init_values = xp.zeros(grid_shape, dtype=float)

        self._sf_values = xp.zeros(grid_shape, dtype=float)
        self._sf1_values = xp.zeros(grid_shape, dtype=float)
        self._rhof_values = xp.zeros(grid_shape, dtype=float)

        self._e_n1 = xp.zeros(grid_shape, dtype=float)
        self._e_n = xp.zeros(grid_shape, dtype=float)

        self._de_s1_values = xp.zeros(grid_shape, dtype=float)

        self._tmp_int_grid = xp.zeros(grid_shape, dtype=float)

        gam = self._gamma
        if self._model == "full":
            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                -gam,
            )
            self._mass_metric_term = deepcopy(metric)

            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                1 - gam,
            )
            self._energy_metric = deepcopy(metric)

        elif self._model == "full_p":
            metric = 1.0 / self.domain.jacobian_det(
                *integration_grid,
            )
            self._mass_metric_term = deepcopy(metric)

            metric = (
                0
                * self.domain.jacobian_det(
                    *integration_grid,
                )
                + 1.0
            )
            self._energy_metric = deepcopy(metric)

            # no need to compute this every time step
            deds = self._de_s1_values
            deds *= 0.0
            deds += 1 / (self._gamma - 1.0)
            deds *= self._mass_metric_term

            self.M_de_ds.assemble([[deds]])
            self.pc_jac.update_mass_operator(self.M_de_ds)

        elif self._model in ["full_q", "linear_q", "deltaf_q"]:
            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                -2,
            )
            self._mass_metric_term = deepcopy(metric)

            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                -1,
            )
            self._energy_metric = deepcopy(metric)

        metric = xp.power(
            self.domain.jacobian_det(
                *integration_grid,
            ),
            1,
        )
        self._sq_term_metric = deepcopy(metric)

        self._jacobian_det = deepcopy(
            self.domain.jacobian_det(*integration_grid),
        )
        
        self._metric_inv = deepcopy(
            self.domain.metric_inv(*integration_grid),
        )
        
        # Integral metric:
        #
        #   |det(DF)| * (DF^T DF)^(-1).
        #
        # This is used for the physical gradient contraction inside integrals.
        self._mass_M1_metric = deepcopy(self._metric_inv)
        self._mass_M1_metric *= self._jacobian_det
        
        

        if self._model in ["linear_q", "deltaf_q"]:
            self.sf1.vector = self.projected_equil.q3

            self._q0_values = self.sf1.eval_tp_fixed_loc(self.integration_grid_spans, self.integration_grid_bd)

        metric = self.domain.metric(
            *integration_grid,
        ) * self.domain.jacobian_det(*integration_grid)
        self._mass_Mv_metric = deepcopy(metric)

        self._get_L2dofs_V3 = L2Projector("L2", self.mass_ops).get_dofs

    def _get_error_newton(self, sn_diff):
        err_s = self._inv_M3.dot_inner(sn_diff, sn_diff)
        return err_s

    def _update_artificial_viscosity(self, un, dt):
        r"""Update the frozen artificial-viscosity coefficient.
    
        The artificial viscosity is evaluated from the old velocity,
    
            mu_a(x) = mu_a * |grad_x u^n(x)|,
    
        and is used consistently in both the momentum equation and the
        internal-energy update.
    
        Returns
        -------
        total_viscosity : xp.ndarray
            Quadrature values of
    
                dt * (mu + mu_a * |grad_x u^n|).
        """
        gu0 = self.grad_0.dot(un, out=self._tmp_gu0)
        gu1 = self.grad_1.dot(un, out=self._tmp_gu1)
        gu2 = self.grad_2.dot(un, out=self._tmp_gu2)
    
        self.gu0f.vector = gu0
        self.gu1f.vector = gu1
        self.gu2f.vector = gu2
    
        # gua_v[i] is the logical derivative d_{eta_i} u_a, where `a`
        # denotes the physical Cartesian velocity component.
        gu0_v = self.gu0f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf0_values,
        )
        gu1_v = self.gu1f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf1_values,
        )
        gu2_v = self.gu2f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf2_values,
        )
    
        # Compute
        #
        #   |grad_x u|^2
        #       = sum_a sum_{i,j}
        #           d_{eta_i} u_a * g^{ij} * d_{eta_j} u_a.
        #
        # Do not include |det(DF)| here: this is a pointwise physical norm,
        # not an integral.
        grad_u_norm = self._gu_init_values
        grad_u_norm *= 0.0
    
        gradients = (gu0_v, gu1_v, gu2_v)
    
        for gradient in gradients:
            for i in range(3):
                for j in range(3):
                    grad_u_norm += (
                        gradient[i]
                        * self._metric_inv[i, j]
                        * gradient[j]
                    )
    
        # Roundoff can produce very small negative values, particularly
        # near coordinate singularities.
        xp.maximum(grad_u_norm, 0.0, out=grad_u_norm)
        xp.sqrt(grad_u_norm, out=grad_u_norm)
    
        # At this point:
        #
        #   grad_u_norm = |grad_x u^n|.
        #
        # Convert it into the time-scaled artificial viscosity used in the
        # implicit velocity operator.
        grad_u_norm *= dt * self._mu_a
    
        # Assemble
        #
        #   dt * mu_a * |grad_x u^n| * |J| * g^{-1}.
        #
        # The physical viscosity dt*mu is handled separately through
        # `_scaled_stiffness`.
        self.M1_du.assemble(
            [
                [
                    grad_u_norm * self._mass_M1_metric[0, 0],
                    grad_u_norm * self._mass_M1_metric[0, 1],
                    grad_u_norm * self._mass_M1_metric[0, 2],
                ],
                [
                    grad_u_norm * self._mass_M1_metric[1, 0],
                    grad_u_norm * self._mass_M1_metric[1, 1],
                    grad_u_norm * self._mass_M1_metric[1, 2],
                ],
                [
                    grad_u_norm * self._mass_M1_metric[2, 0],
                    grad_u_norm * self._mass_M1_metric[2, 1],
                    grad_u_norm * self._mass_M1_metric[2, 2],
                ],
            ],
        )
    
        # Reuse the same quadrature array for the heating update. This is
        # essential: the coefficient in the momentum equation and the
        # coefficient in the internal-energy equation must be identical.
        #
        # After this operation:
        #
        #   grad_u_norm = dt * (mu + mu_a * |grad_x u^n|).
        grad_u_norm += dt * self._mu
    
        return grad_u_norm

    def _update_momentum_operator(self, rho):
        """Update the fixed-density momentum metric and its preconditioner.

        Without regularization, the momentum metric is M_rho. With
        regularization, it is

            M_rho + 2 * alpha_divdiv * K_div,rho.

        The operator is fixed during one viscosity substep.
        """
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

    def _get_energy_change(self, un, un1, dt, total_viscosity):
        """Return the total energy change caused by the viscosity"""
        un12 = un.copy(out=self._tmp_un12)
        un12 += un1
        un12 /= 2.0
        gu0 = self.grad_0.dot(un1, out=self._tmp_gu0)
        gu1 = self.grad_1.dot(un1, out=self._tmp_gu1)
        gu2 = self.grad_2.dot(un1, out=self._tmp_gu2)

        gu012 = self.grad_0.dot(un12, out=self._tmp_gu120)
        gu112 = self.grad_1.dot(un12, out=self._tmp_gu121)
        gu212 = self.grad_2.dot(un12, out=self._tmp_gu122)

        self.gu0f.vector = gu0
        self.gu1f.vector = gu1
        self.gu2f.vector = gu2

        self.gu120f.vector = gu012
        self.gu121f.vector = gu112
        self.gu122f.vector = gu212

        self.uf1.vector = un1
        self.uf12.vector = un12

        gu0_v = self.gu0f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf0_values,
        )
        gu1_v = self.gu1f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf1_values,
        )
        gu2_v = self.gu2f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf2_values,
        )

        gu120_v = self.gu120f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf120_values,
        )
        gu121_v = self.gu121f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf121_values,
        )
        gu122_v = self.gu122f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf122_values,
        )

        u1_v = self.uf1.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_u,
            out=self._uf1_values,
        )
        u12_v = self.uf12.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_u,
            out=self._uf12_values,
        )

        gu_sq_v = self._gu_sq_values
        u_sq_v = self._u_sq_values
        gu_sq_v *= 0.0
        u_sq_v *= 0.0
        for i in range(3):
            for j in range(3):
                gu_sq_v += gu0_v[i] * self._mass_M1_metric[i, j] * gu120_v[j]
                gu_sq_v += gu1_v[i] * self._mass_M1_metric[i, j] * gu121_v[j]
                gu_sq_v += gu2_v[i] * self._mass_M1_metric[i, j] * gu122_v[j]
                u_sq_v += u1_v[i] * self._mass_Mv_metric[i, j] * u12_v[j]

        gu_sq_v *= total_viscosity
        u_sq_v *= dt * self._alpha
        gu_sq_v += u_sq_v

        return gu_sq_v
