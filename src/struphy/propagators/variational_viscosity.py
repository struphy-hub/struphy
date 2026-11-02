import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.feec.basis_projection_ops import CoordinateProjector
from struphy.feec.mass import L2Projector
from struphy.feec.preconditioner import MassMatrixPreconditioner
from struphy.feec.variational_utilities import (
    InternalEnergyEvaluator,
    LinearPressureHeating,
    NewtonConvergence,
    dot_normalized,
    inv_norm2,
    kinetic_metric,
)
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

    With a div-div regularization (``divdiv_alpha`` :math:`> 0`), the mass matrix :math:`\mathbb M^v[\hat\rho]` is
    replaced by the metric :math:`\mathbb A[\hat\rho]` of the regularized kinetic energy, see
    :func:`~struphy.feec.variational_utilities.kinetic_metric`.

    For ``model="linear_p"`` the viscous heating, which is quadratic in the perturbation, goes into the auxiliary
    pressure :math:`\hat p` (``pt3``) with the energy density :math:`\hat p/(\gamma - 1)`: the thermal energy increases
    by exactly the dissipated kinetic energy :math:`\Delta t\, \mu\, (\mathbf u^{n+1})^\top \mathbb K\, \mathbf u^{n+1/2}`,
    where :math:`\mathbb K` is the stiffness matrix (see
    :class:`~struphy.feec.variational_utilities.LinearPressureHeating`).

    **Velocity in** :math:`H(\textnormal{div})` (``model="linear_p"`` only): :math:`\mathbf u` is a 2-form, which has no
    gradient. The viscous term is discretized with its divergence and curl,

    .. math::

        \int_\Omega \mu \left( \nabla \cdot \mathbf u\, \nabla \cdot \mathbf v + \nabla \times \mathbf u \cdot \nabla \times \mathbf v \right) \textrm d \mathbf x \,,
        \qquad
        \mathbb K = \mathbb D^\top \mathbb M_3 \mathbb D + (\mathbb C \mathcal S^{21})^\top \mathbb M_2\, \mathbb C \mathcal S^{21} \,,

    which equals :math:`\int \mu \nabla \mathbf u : \nabla \mathbf v` up to boundary terms. The curl is taken of the
    1-form :math:`\mathcal S^{21} \mathbf u = \hat\Pi^1[G \hat{\mathbf u}^2 / \sqrt g]`. With Dirichlet boundary
    conditions its tangential trace is zero, which acts as a weak no-slip penalty (of order :math:`\mu/h`) on the
    tangential velocity at the wall. The velocity mass matrix is :math:`\mathbb M^2_n` (as in the ideal propagators of
    the linear model):

    .. math::

        (\mathbb M^2_n + \Delta t\, \mu\, \mathbb K)\, \mathbf u^{n+1} = \mathbb M^2_n \mathbf u^n \,.

    The artificial viscosity :math:`\mu_a`, the damping ``alpha`` and the div-div regularization are not available
    in this case.
    """

    class Variables:
        """Container for variables advanced by :class:`VariationalViscosity`.

        Attributes
        ----------
        s : FEECVariable
            Thermodynamic scalar variable in ``"L2"`` space.
        u : FEECVariable
            Velocity variable in ``"H1vec"`` space, or in ``"Hdiv"`` space (only for ``model="linear_p"``).
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
            assert new.space in ("H1vec", "Hdiv")
            self._u = new

    def __init__(self, rho: FEECVariable = None, pt3: FEECVariable = None, divdiv_alpha: float = 0.0):
        r"""
        Parameters
        ----------
        rho : FEECVariable, default=None
            Mass density 3-form (``"L2"`` space) weighting the velocity mass matrix.
        pt3 : FEECVariable, default=None
            For ``model="linear_q"`` and ``"perturbative_q"``: the auxiliary variable :math:`\hat q` (``"L2"`` space),
            which receives the viscous heating with the energy density :math:`2 q_0 \hat q / (\gamma - 1)`, i.e.
            :math:`\partial_t \hat q = (\gamma - 1)/(2 q_0)\, \mu |\nabla \mathbf u|^2`. For ``model="linear_p"``: the
            auxiliary pressure :math:`\hat p` (``"L2"`` space), with the energy density :math:`\hat p / (\gamma - 1)`,
            heated by :class:`~struphy.feec.variational_utilities.LinearPressureHeating`. If ``None``, the thermodynamic
            equation is skipped. For the other models the heating goes into the variable ``s``.
        divdiv_alpha : float, default=0.0
            Coefficient of the div-div regularization of the kinetic energy (a parameter of the model, the same in
            all its variational propagators), see :func:`~struphy.feec.variational_utilities.kinetic_metric`.
        """
        assert divdiv_alpha >= 0.0, f"divdiv_alpha must be non-negative, got {divdiv_alpha}."
        self.variables = self.Variables()
        self.rho = rho
        self.pt3 = pt3
        self.divdiv_alpha = divdiv_alpha

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`VariationalViscosity`.

        Parameters
        ----------
        model : {"full", "full_p", "full_q", "linear_p", "linear_q", "perturbative_q"}, default="full"
            Thermodynamic model variant.
        gamma : float, default=5/3
            Adiabatic index.
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Linear solver for implicit subproblems.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner used in linear solves.
        solver_params : SolverParameters, default=None
            Linear-solver controls. The solves are done for normalized right-hand sides, so ``tol`` is relative.
        nonlin_solver : NonlinearSolverParameters, default=None
            Nonlinear iteration controls. The iteration stops when the norm of the residual is below
            ``tol`` times the norm of the state at :math:`t^n` if that is smaller than one (absolute otherwise), or
            when it stagnates (with a warning).
        mu : float, default=0.0
            Physical viscosity coefficient.
        mu_a : float, default=0.0
            Artificial-viscosity coefficient.
        alpha : float, default=0.0
            Optional linear damping/regularization parameter.
        fast : bool, default=False
            If ``True`` and ``model="linear_q"``, skip the thermodynamic (energy balance) update.
        """

        # specific literals
        OptsModel = Literal["full", "full_p", "full_q", "linear_p", "linear_q", "perturbative_q"]
        # propagator options
        model: OptsModel = "full"
        gamma: float = 5.0 / 3.0
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        mu: float = 0.0
        mu_a: float = 0.0
        alpha: float = 0.0
        fast: bool = False

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
        self._fast = self.options.fast
        self._mu_a = self.options.mu_a
        self._alpha = self.options.alpha
        self._mu = self.options.mu

        self._info = self._nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        self._u_space = self.variables.u.space
        if self._u_space == "Hdiv":
            assert self._model == "linear_p", f'u in "Hdiv" needs model="linear_p", got {self._model!r}.'
            assert self._mu_a == 0.0 and self._alpha == 0.0 and self.divdiv_alpha == 0.0, (
                'u in "Hdiv": mu_a, alpha and divdiv_alpha are not available.'
            )
            # the fixed velocity mass matrix of the linear model (as in its ideal propagators)
            self._Mrho = self.mass_ops.M2n
            self._Mrho_pc = MassMatrixPreconditioner(self._Mrho)
        else:
            # assembly of WMMnew happens in VariationalDensityEvolve
            self._Mrho = kinetic_metric(self.derham, self.mass_ops, self.basis_ops, self.divdiv_alpha)
            self._Mrho_pc = MassMatrixPreconditioner(self.mass_ops.WMMnew, dim_reduce=None)

        # linear p model: the heating goes into the auxiliary pressure, whose energy is linear
        self._heating = None
        if self._model == "linear_p" and self.pt3 is not None:
            self._heating = LinearPressureHeating(
                self.derham, self.mass_ops, self.domain, self._gamma, tol=self._lin_solver.tol
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
        self._tmp_rhs = u.space.zeros()
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

    def __call__(self, dt):
        if self._u_space == "H1vec":
            # WMMnew is re-assembled in VariationalDensityEvolve: update the diagonal scaling of its preconditioner
            self._Mrho_pc.refresh()

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

        self._scaled_stiffness._scalar = dt * self._mu  # /2.
        if self._u_space == "H1vec":
            # Update artificial viscosity weighted mass matrix
            total_viscosity = self._update_artificial_viscosity(un, dt)
            self._scaled_Mv._scalar = dt * self._alpha
        # self.evol_op._multiplicands[1]._addends[0]._scalar = - dt*self._mu/2.
        # normalized: the tolerance of the implicit solve is relative to the size of its right-hand side M u^n
        rhs = self.r_op.dot(un, out=self._tmp_rhs)
        un1 = dot_normalized(self.inv_lop, rhs, self._tmp_un1)
        if self._info:
            logger.info(f"information on the linear solver : {self.inv_lop._info}")

        if self._model == "linear_p":
            # the dissipated kinetic energy dt mu u^{n+1} . K u^{n+1/2} goes into the auxiliary pressure
            if self._heating is not None:
                if self._u_space == "Hdiv":
                    energy_change = self._get_energy_change_hdiv(un, un1, dt)
                else:
                    energy_change = self._get_energy_change(un, un1, dt, total_viscosity)
                self._heating(energy_change, self.pt3.spline.vector)
                self.pt3.spline.vector.update_ghost_regions()
            self.update_feec_variables(s=sn, u=un1)
            return

        if self._model == "linear_q" and self._fast:
            self.update_feec_variables(s=sn, u=un1)
            return

        # Energy balance term
        # 1) Pointwize energy change
        energy_change = self._get_energy_change(un, un1, dt, total_viscosity)
        # 2) Initial energy and linear form
        rho = self.rho.spline.vector
        if self._model in ["perturbative_q", "linear_q"]:
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

        elif self._model in ["linear_q", "perturbative_q"]:
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

        tol = self._nonlin_solver.tol
        # stopping criterion relative to the state at t^n, with a check for stagnation
        conv = NewtonConvergence("VariationalViscosity", tol, self._get_error_newton(self._linear_form_tot_e))

        for it in range(self._nonlin_solver.maxiter):
            if self._model in ["perturbative_q", "linear_q"]:
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

            elif self._model in ["linear_q", "perturbative_q"]:
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

            # at least one iteration
            stop = conv.stop(self._get_error_newton(self.tot_rhs), it, min_iter=1)

            if self._info:
                logger.info(f"iteration : {it} relative error : {conv.err}")

            if stop:
                break

            if self._model == "full":
                deds = self._energy_evaluator.dener_ds(
                    rhof_values,
                    sf1_values,
                    out=self._de_s1_values,
                )
                deds *= self._mass_metric_term

                self.M_de_ds.assemble([[deds]])
                if self.pc_jac is not None:
                    self.pc_jac.update_mass_operator(self.M_de_ds)

            elif self._model in ["full_q", "linear_q", "perturbative_q"]:
                if self._model in ["perturbative_q", "linear_q"]:
                    sf1_values = self._q0_values

                deds = self._de_s1_values
                deds *= 0.0
                deds += sf1_values
                deds *= 2 / (self._gamma - 1.0)
                deds *= self._mass_metric_term

                self.M_de_ds.assemble([[deds]])
                if self.pc_jac is not None:
                    self.pc_jac.update_mass_operator(self.M_de_ds)

            # normalized right-hand side: the tolerance of the linear solver is relative to the residual
            incr = dot_normalized(self.inv_jac, self.tot_rhs, self._tmp_sn_incr)

            if self._info:
                logger.info(f"information on the linear solver : {self.inv_jac._info}")

            if self._model in ["perturbative_q", "linear_q"]:
                pt3_vec = self.pt3.spline.vector
                pt3_vec += incr
            else:
                sn1 += incr

        conv.warn_if_not_converged(it, self._nonlin_solver.maxiter)

        self.update_feec_variables(s=sn1, u=un1)

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        self.M1_du = self.mass_ops.create_weighted_mass("Hcurl", "Hcurl")

        self.pc_M3 = preconditioner.MassMatrixPreconditioner(
            self.mass_ops.M3,
            dim_reduce=None,
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
            # dim_reduce=None: the Kronecker factors do not depend on the weights, cheap to update
            self.pc_jac = pc_class(self.M_de_ds, dim_reduce=None)

        self.inv_jac = inverse(
            self.M_de_ds,
            "pcg",
            pc=self.pc_jac,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=False,
            recycle=True,
        )

        if self._u_space == "Hdiv":
            self._init_hdiv_operators()
        else:
            self._init_h1vec_operators()

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
            if self.pc_jac is not None:
                self.pc_jac.update_mass_operator(self.M_de_ds)

        elif self._model in ["full_q", "linear_q", "perturbative_q"]:
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

        metric = self.domain.metric_inv(
            *integration_grid,
        ) * self.domain.jacobian_det(*integration_grid)
        self._mass_M1_metric = deepcopy(metric)

        if self._model in ["linear_q", "perturbative_q"]:
            self.sf1.vector = self.projected_equil.q3

            self._q0_values = self.sf1.eval_tp_fixed_loc(self.integration_grid_spans, self.integration_grid_bd)

        metric = self.domain.metric(
            *integration_grid,
        ) * self.domain.jacobian_det(*integration_grid)
        self._mass_Mv_metric = deepcopy(metric)

        self._get_L2dofs_V3 = L2Projector("L2", self.mass_ops).get_dofs

        if self._u_space == "Hdiv":
            bn, bd = self.integration_grid_bn, self.integration_grid_bd
            self.integration_grid_2form = [[bn[0], bd[1], bd[2]], [bd[0], bn[1], bd[2]], [bd[0], bd[1], bn[2]]]
            sqrt_g = self.domain.jacobian_det(*integration_grid)
            self._inv_sqrt_g = 1.0 / sqrt_g
            self._G_over_sqrt_g = self.domain.metric(*integration_grid) / sqrt_g
            self.du1f = self.derham.create_spline_function("du1", "L2")
            self.du12f = self.derham.create_spline_function("du12", "L2")
            self.cu1f = self.derham.create_spline_function("cu1", "Hdiv")
            self.cu12f = self.derham.create_spline_function("cu12", "Hdiv")
            self._tmp_du1 = self.derham.V3pol.zeros()
            self._tmp_du12 = self.derham.V3pol.zeros()
            self._tmp_cu1 = self.derham.V2pol.zeros()
            self._tmp_cu12 = self.derham.V2pol.zeros()
            self._du1_values = xp.zeros(grid_shape, dtype=float)
            self._du12_values = xp.zeros(grid_shape, dtype=float)
            self._cu1_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
            self._cu12_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]

    def _init_h1vec_operators(self):
        """Velocity in H1vec: the stiffness of the components of the gradient, and the implicit step."""
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

        self._scaled_stiffness = 0.00001 * self.phy_stiffness

        self._scaled_Mv = 0.1 * self.mass_ops.Mv

        self.r_op = self._Mrho  # - self._scaled_stiffness - self.du_phy_stiffness
        self.l_op = self._Mrho + self._scaled_Mv + self._scaled_stiffness + self.du_phy_stiffness

        # with the boundary conditions of M1 (which acts only on their range): the pointwise energy change of the heating
        # is then the same quadratic form as the stiffness
        B1 = self.derham.boundary_ops["1"]
        self.grad_0 = B1 @ grad @ Pcoord0 @ Xv
        self.grad_1 = B1 @ grad @ Pcoord1 @ Xv
        self.grad_2 = B1 @ grad @ Pcoord2 @ Xv

        # a fixed preconditioner: an inner iterative solve (of fixed tolerance) would vary from one application to the
        # next, and the PCG then stagnates at tight tolerances
        self.inv_lop = inverse(
            self.l_op,
            "pcg",
            pc=self._Mrho_pc,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=False,
            recycle=True,
        )

        self.evol_op = self.inv_lop @ self.r_op

    def _init_hdiv_operators(self):
        r"""Velocity in Hdiv: the stiffness :math:`\mathbb D^\top \mathbb M^3 \mathbb D + (\mathbb C \mathcal S^{21})^\top
        \mathbb M^2 \,\mathbb C \mathcal S^{21}` (div and curl of :math:`\mathbf u`, the curl of the 1-form
        :math:`\mathcal S^{21} \mathbf u = \hat \Pi^1[G \hat{\mathbf u}^2/\sqrt g]`), with the velocity mass matrix
        :math:`\mathbb M^2_n` of the linear model, and the implicit step."""
        self._div = self.derham.div
        self._curl_S21 = self.derham.curl @ self.basis_ops.S21
        self.phy_stiffness = (
            self._div.T @ self.mass_ops.M3 @ self._div + self._curl_S21.T @ self.mass_ops.M2 @ self._curl_S21
        )
        self._scaled_stiffness = 0.00001 * self.phy_stiffness

        self.r_op = self._Mrho
        self.l_op = self._Mrho + self._scaled_stiffness

        self.inv_lop = inverse(
            self.l_op,
            "pcg",
            pc=self._Mrho_pc,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=False,
            recycle=True,
        )

        self.evol_op = self.inv_lop @ self.r_op

    def _get_error_newton(self, sn_diff):
        # the inverse mass matrix is applied to the normalized residual (relative solver tolerance)
        err_s = inv_norm2(self._inv_M3, sn_diff)
        return err_s

    def _update_artificial_viscosity(self, un, dt):
        """Update the artificial viscosity as the norm of the gradient of un.
        Update the associated mass matrix and return the total viscosity for later computation"""
        gu0 = self.grad_0.dot(un, out=self._tmp_gu0)
        gu1 = self.grad_1.dot(un, out=self._tmp_gu1)
        gu2 = self.grad_2.dot(un, out=self._tmp_gu2)

        self.gu0f.vector = gu0
        self.gu1f.vector = gu1
        self.gu2f.vector = gu2

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

        gu_sq_v = self._gu_init_values
        gu_sq_v *= 0.0
        for i in range(3):
            gu0_v[i] **= 2
            gu1_v[i] **= 2
            gu2_v[i] **= 2
            gu_sq_v += gu0_v[i]
            gu_sq_v += gu1_v[i]
            gu_sq_v += gu2_v[i]

        xp.sqrt(gu_sq_v, out=gu_sq_v)

        gu_sq_v *= dt * self._mu_a  # /2

        self.M1_du.assemble(
            [
                [
                    gu_sq_v * self._mass_M1_metric[0, 0],
                    gu_sq_v * self._mass_M1_metric[0, 1],
                    gu_sq_v * self._mass_M1_metric[0, 2],
                ],
                [
                    gu_sq_v * self._mass_M1_metric[1, 0],
                    gu_sq_v * self._mass_M1_metric[1, 1],
                    gu_sq_v * self._mass_M1_metric[1, 2],
                ],
                [
                    gu_sq_v * self._mass_M1_metric[2, 0],
                    gu_sq_v * self._mass_M1_metric[2, 1],
                    gu_sq_v * self._mass_M1_metric[2, 2],
                ],
            ],
        )

        # gu_sq_v *= 2.
        gu_sq_v += dt * self._mu

        return gu_sq_v

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

    def _get_energy_change_hdiv(self, un, un1, dt):
        """Velocity in Hdiv: the dissipated energy density dt mu (div u^{n+1} div u^{n+1/2} + curl S21 u^{n+1} . G curl
        S21 u^{n+1/2}) / sqrt(g) at the quadrature points, whose integral is dt mu u^{n+1} . K u^{n+1/2}."""
        un12 = un.copy(out=self._tmp_un12)
        un12 += un1
        un12 /= 2.0

        self.du1f.vector = self._div.dot(un1, out=self._tmp_du1)
        self.du12f.vector = self._div.dot(un12, out=self._tmp_du12)
        self.cu1f.vector = self._curl_S21.dot(un1, out=self._tmp_cu1)
        self.cu12f.vector = self._curl_S21.dot(un12, out=self._tmp_cu12)

        du1_v = self.du1f.eval_tp_fixed_loc(self.integration_grid_spans, self.integration_grid_bd, out=self._du1_values)
        du12_v = self.du12f.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_bd, out=self._du12_values
        )
        cu1_v = self.cu1f.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_2form, out=self._cu1_values
        )
        cu12_v = self.cu12f.eval_tp_fixed_loc(
            self.integration_grid_spans, self.integration_grid_2form, out=self._cu12_values
        )

        e = self._gu_sq_values
        e *= 0.0
        e += du1_v * du12_v * self._inv_sqrt_g
        for i in range(3):
            for j in range(3):
                e += cu1_v[i] * self._G_over_sqrt_g[i, j] * cu12_v[j]
        e *= dt * self._mu
        return e
