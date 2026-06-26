import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.feec.mass import L2Projector
from struphy.feec.variational_utilities import InternalEnergyEvaluator
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VariationalResistivity(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`s \in L^2` and  :math:`\mathbf B \in H(\textrm{div})` such that

    .. math::

        &\int_\Omega \partial_t \mathbf B \cdot \mathbf v \,\textrm d \mathbf x + \int_\Omega (\eta + \eta_a(\mathbf x)) \nabla \times \mathbf B \cdot \nabla \times \mathbf v \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in H(\textrm{div}) \,,
        \\[4mm]
        &\int_\Omega \frac{\partial \mathcal U}{\partial s} \partial_t s \, q\,\textrm d \mathbf x - \int_\Omega (\eta + \eta_a(\mathbf x)) |\nabla \times \mathbf B|^2 \, q \,\textrm d \mathbf x = 0 \qquad \forall \, q \in L^2\, \text{if using } s,
        \\[4mm]
        &\int_\Omega \frac{1}{\gamma - 1} \partial_t p \, q\,\textrm d \mathbf x - \int_\Omega (\eta + \eta_a(\mathbf x)) |\nabla \times \mathbf B|^2 \, q \,\textrm d \mathbf x = 0 \qquad \forall \, q \in L^2\, \text{if using } p.

    With :math:`\eta_a(\mathbf x) = \eta_a |\nabla \times \mathbf B(\mathbf x)|`

    On the logical domain:

    .. math::

        \begin{align}
        &\partial_t \hat{\boldsymbol B} - \eta \Delta \hat{\boldsymbol B} = 0 ~ ,
        \\[2mm]
        &\int_{\hat{\Omega}} \partial_t (\hat{\rho} \mathcal U(\hat{\rho}, \hat{s})) \hat{w} \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta - \int_{\hat{\Omega}} (\eta + \eta_a(\mathbf \eta)) |DF^{-T}\tilde{\nabla} \times \hat{\boldsymbol B}|^2  \hat{w} \, \textrm d \boldsymbol \eta = 0 ~ , \text{if using } s,
        \\[2mm]
        &\int_{\hat{\Omega}} \partial_t (\frac{1}{\gamma -1} \hat{p} ) \hat{w} \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta - \int_{\hat{\Omega}} (\eta + \eta_a(\mathbf \eta)) |DF^{-T}\tilde{\nabla} \times \hat{\boldsymbol B}|^2  \hat{w} \, \textrm d \boldsymbol \eta = 0 ~, \text{if using } p.
        \end{align}

    It is discretized as

    .. math::

        \begin{align}
        &\frac{\mathbf B^{n+1}-\mathbf B^n}{\Delta t}
        + \, \mathbb C \mathbb M_1^{-1} (\eta M_1 + \eta_a M_1[|\nabla \times \mathbf B|]) M_1^{-1} \mathbb C^T \mathbb M_2  \mathbf B^{n+1} = 0 ~ ,
        \\[2mm]
        &\frac{P^{3}(\rho e(s^{n+1})- P^{3}(\rho e(s^{n}))}{\Delta t} - P^3((\eta + \eta_a(\mathbf x)) DF^{-T} \tilde{\mathbb C} \frac{ \mathbf B^{n+1}+\mathbf B^n}{2} \cdot DF^{-T} \tilde{\mathbb C} \mathbf B^{n+1}) = 0 ~ , \text{if using } s,
        \\[2mm]
        &\frac{1}{\gamma -1}\frac{p^{n+1}- p^{n}}{\Delta t} - P^3((\eta + \eta_a(\mathbf x)) DF^{-T} \tilde{\mathbb C} \frac{ \mathbf B^{n+1}+\mathbf B^n}{2} \cdot DF^{-T} \tilde{\mathbb C} \mathbf B^{n+1}) = 0 ~ , \text{if using } p.
        \end{align}

    where $P^3$ denotes the $L^2$ projection in the last space of the de Rham sequence.

    """

    class Variables:
        """Container for variables advanced by :class:`VariationalResistivity`.

        Attributes
        ----------
        s : FEECVariable
            Thermodynamic scalar variable in ``"L2"`` space.
        b : FEECVariable
            Magnetic-field variable in ``"Hdiv"`` space.
        """

        def __init__(self):
            self._s: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def s(self) -> FEECVariable:
            return self._s

        @s.setter
        def s(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._s = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(self, rho: FEECVariable = None, pt3: FEECVariable = None):
        self.variables = self.Variables()
        self.rho = rho
        self.pt3 = pt3

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`VariationalResistivity`.

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
        linearize_current : bool, default=False
            If ``True``, linearize current terms around background fields.
        eta : float, default=0.0
            Physical resistivity coefficient.
        eta_a : float, default=0.0
            Artificial-resistivity coefficient.
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
        linearize_current: bool = False
        eta: float = 0.0
        eta_a: float = 0.0

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
        self._eta = self.options.eta
        self._eta_a = self.options.eta_a
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize_current = self.options.linearize_current

        self._info = self._nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        # Femfields for the projector
        self.rhof = self.derham.create_spline_function("rhof", "L2")
        self.sf = self.derham.create_spline_function("sf", "L2")
        self.sf1 = self.derham.create_spline_function("sf1", "L2")
        self.bf = self.derham.create_spline_function("Bf", "Hdiv")
        self.bf1 = self.derham.create_spline_function("Bf1", "Hdiv")
        self.cbf1 = self.derham.create_spline_function("cBf", "Hcurl")
        self.cbf12 = self.derham.create_spline_function("cBf", "Hcurl")

        # Projector
        self._energy_evaluator = InternalEnergyEvaluator(self.derham, self._gamma)
        self._initialize_projectors_and_mass()

        # bunch of temporaries to avoid allocating in the loop
        s = self.variables.s.spline.vector
        b = self.variables.b.spline.vector

        self._tmp_bn1 = b.space.zeros()
        self._tmp_bn12 = b.space.zeros()
        self._tmp_sn1 = s.space.zeros()
        self._tmp_sn_incr = s.space.zeros()
        self._tmp_sn_weak_diff = s.space.zeros()
        self._tmp_cb12 = self.derham.V1pol.zeros()
        self._tmp_cb1 = self.derham.V1pol.zeros()
        self._linear_form_tot_e = s.space.zeros()
        self._linear_form_en1 = s.space.zeros()
        self.tot_rhs = s.space.zeros()
        if True:  # self._linearize_current:
            self._extracted_b2 = self.derham.boundary_ops["2"].dot(
                self.derham.extraction_ops["2"].dot(self.projected_equil.b2),
            )

    def __call__(self, dt):
        if self._nonlin_solver.type == "Newton":
            self.__call_newton(dt)
        else:
            raise ValueError(
                "wrong value for solver type in VariationalResistivity",
            )

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        # Compute dissipation implicitely
        sn = self.variables.s.spline.vector
        bn = self.variables.b.spline.vector

        if self._eta < 1.0e-15 and self._eta_a < 1.0e-15:
            self.update_feec_variables(s=sn, b=bn)
            return

        if self._info:
            logger.info("")
            logger.info("Computing the dissipation in VariationalResistivity")

        total_resistivity = self._update_artificial_resistivity(bn, dt)

        self._scaled_stiffness._scalar = dt * self._eta
        # self.evol_op._multiplicants[1]._addends[0]._scalar = -dt*self._eta/2.
        if self._linearize_current:
            bn1 = self.evol_op.dot(
                bn
                + dt
                * self._eta
                * self.curl.dot(
                    self.Tcurl.dot(self._extracted_b2),
                ),
                out=self._tmp_bn1,
            )
        else:
            bn1 = self.evol_op.dot(bn, out=self._tmp_bn1)
        if self._info:
            logger.info(f"information on the linear solver : {self.inv_lop._info}")

        if self._model == "linear_p" or (self._model == "linear_q" and self._nonlin_solver["fast"]):
            self.update_feec_variables(s=sn, b=bn1)
            return

        # Energy balance term
        # 1) Pointwize energy change
        energy_change = self._get_energy_change(bn, bn1, total_resistivity)
        # 2) Initial energy and linear form
        rho = self.rho.spline.vector
        self.rhof.vector = rho
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
            rhof_values = self.rhof.eval_tp_fixed_loc(
                self.integration_grid_spans,
                self.integration_grid_bd,
                out=self._rhof_values,
            )

            e_n = self._energy_evaluator.ener(
                rhof_values,
                sf_values,
                out=self._e_n,
            )

            e_n *= self._energy_metric

        elif self._model in ["full_p", "linear_p", "delta_p"]:
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

        tol = self._nonlin_solver["tol"]
        err = tol + 1

        for it in range(self._nonlin_solver["maxiter"]):
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

            elif self._model in ["full_p", "linear_p", "delta_p"]:
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

            err = self._get_error_newton(self.tot_rhs)

            if self._info:
                logger.info(f"iteration : {it} error : {err}")

            if (err < tol**2 and it > 0) or xp.isnan(err):
                break

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

        if it == self._nonlin_solver["maxiter"] - 1 or xp.isnan(err):
            logger.info(
                f"!!!Warning: Maximum iteration in VariationalResistivity reached - not converged:\n {err =} \n {tol**2 =}",
            )

        self.update_feec_variables(s=sn1, b=bn1)

        # if self.pt3 is not None:
        #     bn12 = bn.copy(out=self._tmp_bn12)
        #     bn12 += bn1
        #     bn12 /= 2.0
        #     cb1 = self.Tcurl.dot(bn1, out=self._tmp_cb1)
        #     cb12 = self.Tcurl.dot(bn12, out=self._tmp_cb12)

        #     self.cbf12.vector = cb12
        #     self.cbf1.vector = cb1

        #     cb12_v = self.cbf12.eval_tp_fixed_loc(
        #         self.integration_grid_spans,
        #         self.integration_grid_curl,
        #         out=self._cb12_values,
        #     )
        #     cb1_v = self.cbf1.eval_tp_fixed_loc(
        #         self.integration_grid_spans,
        #         self.integration_grid_curl,
        #         out=self._cb1_values,
        #     )

        #     cb_sq_v = self._cb_sq_values
        #     cb_sq_v *= 0.0
        #     for i in range(3):
        #         for j in range(3):
        #             cb_sq_v += cb12_v[i] * self._sq_term_metric[i, j] * cb1_v[j]

        #     cb_sq_v *= self._cb_sq_values_init
        #     # 2) Initial energy and linear form
        #     self.sf.vector = self.pt3

        #     sf_values = self.sf.eval_tp_fixed_loc(
        #         self.integration_grid_spans,
        #         self.integration_grid_bd,
        #         out=self._sf_values,
        #     )

        #     e_n = self._e_n
        #     e_n *= 0.0
        #     e_n += sf_values
        #     e_n *= 1.0 / (self._gamma - 1.0)
        #     e_n *= self._energy_metric

        #     cb_sq_v += e_n

        #     self._get_L2dofs_V3(cb_sq_v, dofs=self._linear_form_tot_e)

        #     tol = self._nonlin_solver["tol"]
        #     err = tol + 1

        #     for it in range(self._nonlin_solver["maxiter"]):
        #         self.sf1.vector = self.pt3

        #         sf1_values = self.sf1.eval_tp_fixed_loc(
        #             self.integration_grid_spans,
        #             self.integration_grid_bd,
        #             out=self._sf1_values,
        #         )

        #         e_n1 = self._e_n1
        #         e_n1 *= 0.0
        #         e_n1 += sf1_values
        #         e_n1 *= 1.0 / (self._gamma - 1.0)
        #         e_n1 *= self._energy_metric

        #         self._get_L2dofs_V3(e_n1, dofs=self._linear_form_en1)

        #         self.tot_rhs *= 0.0
        #         self.tot_rhs -= self._linear_form_en1
        #         self.tot_rhs += self._linear_form_tot_e

        #         err = self._get_error_newton(self.tot_rhs)

        #         if self._info:
        #             logger.info("iteration : ", it, " error : ", err)

        #         if (err < tol**2 and it > 0) or xp.isnan(err):
        #             break

        #         incr = self.inv_jac.dot(self.tot_rhs, out=self._tmp_sn_incr)

        #         if self._info:
        #             logger.info("information on the linear solver : ", self.inv_jac._info)

        #         self.pt3 += incr

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        pc_M1 = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.M1,
        )
        inv_M1 = inverse(
            self.mass_ops.M1,
            "pcg",
            pc=pc_M1,
            tol=1e-16,
            maxiter=1000,
            verbose=False,
        )

        pc_M3 = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.M3,
        )
        self._inv_M3 = inverse(
            self.mass_ops.M3,
            "pcg",
            pc=pc_M3,
            tol=1e-16,
            maxiter=1000,
            verbose=False,
        )

        M2 = self.mass_ops.M2
        self.M_de_ds = self.mass_ops.create_weighted_mass("L2", "L2")

        D = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.M1_cb = self.mass_ops.create_weighted_mass("Hcurl", "Hcurl", weights=(D, "sqrt_g"))

        if self.options.precond is None:
            self.pc = None
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

        self.curl = self.derham.curl
        self.Tcurl = inv_M1 @ self.curl.T @ M2

        self.phy_stiffness = M2 @ self.curl @ inv_M1 @ self.curl.T @ M2
        self.phy_cb_stiffness = self.Tcurl.T @ self.M1_cb @ self.Tcurl

        self._scaled_stiffness = 0.00001 * self.phy_stiffness

        self.r_op = M2  # - self._scaled_stiffness
        self.l_op = M2 + self._scaled_stiffness + self.phy_cb_stiffness

        if self.options.precond is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner,
                self.options.precond,
            )
            self.pc = pc_class(M2)

        self.inv_lop = inverse(
            self.l_op,
            "pcg",
            pc=self.pc,
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

        self.integration_grid_curl = [
            [self.integration_grid_bd[0], self.integration_grid_bn[1], self.integration_grid_bn[2]],
            [
                self.integration_grid_bn[0],
                self.integration_grid_bd[1],
                self.integration_grid_bn[2],
            ],
            [self.integration_grid_bn[0], self.integration_grid_bn[1], self.integration_grid_bd[2]],
        ]

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])

        self._cb12_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._cb1_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]

        self._cb_sq_values = xp.zeros(grid_shape, dtype=float)
        self._cb_sq_values_init = xp.zeros(grid_shape, dtype=float)

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

        elif self._model in ["full_p", "linear_p", "delta_p"]:
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

            # No need to compute this every iteration
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

        if self._model in ["linear_q", "deltaf_q"]:
            self.sf1.vector = self.projected_equil.q3

            self._q0_values = self.sf1.eval_tp_fixed_loc(self.integration_grid_spans, self.integration_grid_bd)

        metric = self.domain.metric_inv(
            *integration_grid,
        ) * self.domain.jacobian_det(*integration_grid)
        self._sq_term_metric = deepcopy(metric)

        metric = self.domain.metric_inv(
            *integration_grid,
        )
        self._sq_term_metric_no_jac = deepcopy(metric)

        self._get_L2dofs_V3 = L2Projector("L2", self.mass_ops).get_dofs

    def _get_error_newton(self, sn_diff):
        err_s = self._inv_M3.dot_inner(sn_diff, sn_diff)
        return err_s

    def _update_artificial_resistivity(self, bn, dt):
        """Update the artificial resistivity as the norm of the gradient of un.
        Update the associated mass matrix and return the total resistivity for later computation"""
        if self._eta_a > 1e-15:
            cb = self.Tcurl.dot(bn, out=self._tmp_cb1)
            self.cbf1.vector = cb
            cb_v = self.cbf1.eval_tp_fixed_loc(
                self.integration_grid_spans,
                self.integration_grid_curl,
                out=self._cb1_values,
            )

            cb_sq_v = self._cb_sq_values_init
            cb_sq_v *= 0.0
            for i in range(3):
                for j in range(3):
                    cb_sq_v += cb_v[i] * self._sq_term_metric_no_jac[i, j] * cb_v[j]

            xp.sqrt(cb_sq_v, out=cb_sq_v)

            cb_sq_v *= dt * self._eta_a

            self.M1_cb.assemble(
                [
                    [
                        cb_sq_v * self._sq_term_metric[0, 0],
                        cb_sq_v * self._sq_term_metric[0, 1],
                        cb_sq_v * self._sq_term_metric[0, 2],
                    ],
                    [
                        cb_sq_v * self._sq_term_metric[1, 0],
                        cb_sq_v * self._sq_term_metric[1, 1],
                        cb_sq_v * self._sq_term_metric[1, 2],
                    ],
                    [
                        cb_sq_v * self._sq_term_metric[2, 0],
                        cb_sq_v * self._sq_term_metric[2, 1],
                        cb_sq_v * self._sq_term_metric[2, 2],
                    ],
                ],
            )

            cb_sq_v += dt * self._eta

        else:
            cb_sq_v = self._cb_sq_values_init
            cb_sq_v *= 0.0
            cb_sq_v += dt * self._eta

        return cb_sq_v

    def _get_energy_change(self, bn, bn1, total_resistivity):
        """Return the total energy change caused by the resistivity"""
        bn12 = bn.copy(out=self._tmp_bn12)
        bn12 += bn1
        bn12 /= 2.0
        if self._linearize_current:
            cb1 = self.Tcurl.dot(
                bn1 - self._extracted_b2,
                out=self._tmp_cb1,
            )
        else:
            cb1 = self.Tcurl.dot(bn1, out=self._tmp_cb1)

        if self._model in ["full", "full_p", "full_q"]:
            cb12 = self.Tcurl.dot(bn12, out=self._tmp_cb12)

        # elif self._model in ["linear_p", "linear_q"]:
        #     cb12 = self.Tcurl.dot(self._extracted_b2, out=self._tmp_cb12)

        elif self._model in ["delta_p", "deltaf_q", "linear_p", "linear_q"]:
            # bn12 += self._extracted_b2
            cb12 = self.Tcurl.dot(bn12, out=self._tmp_cb12)

        self.cbf12.vector = cb12
        self.cbf1.vector = cb1

        cb12_v = self.cbf12.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_curl,
            out=self._cb12_values,
        )
        cb1_v = self.cbf1.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_curl,
            out=self._cb1_values,
        )

        cb_sq_v = self._cb_sq_values
        cb_sq_v *= 0.0
        for i in range(3):
            for j in range(3):
                cb_sq_v += cb12_v[i] * self._sq_term_metric[i, j] * cb1_v[j]

        cb_sq_v *= total_resistivity

        return cb_sq_v
