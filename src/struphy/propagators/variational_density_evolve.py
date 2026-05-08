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
from struphy.feec.preconditioner import MassMatrixDiagonalPreconditioner
from struphy.feec.variational_utilities import (
    InternalEnergyEvaluator,
    KineticEnergyEvaluator,
)
from struphy.io.options import LiteralOptions
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

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
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
        s : FEECVariable, default=None
            Entropy-like variable required by ``model="full"``.
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
        s: FEECVariable = None

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

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new

    @profile
    def allocate(self, verbose: bool = False):
        if self.options.model == "full":
            assert self.options.s is not None

        self._model = self.options.model
        self._gamma = self.options.gamma
        self._s = self.options.s
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize = self.options.nonlin_solver.linearize

        self._info = self.options.nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        self._Mrho = self.mass_ops.WMMnew
        pc = MassMatrixDiagonalPreconditioner(self._Mrho)
        self._Mrho_inv = inverse(
            self._Mrho,
            "pcg",
            pc=pc,
            tol=1e-16,
            maxiter=500,
            recycle=True,
        )

        # Femfields for the projector
        self.rhof = self.derham.create_spline_function("rhof", "L2")
        self.rhof1 = self.derham.create_spline_function("rhof1", "L2")

        rho = self.variables.rho.spline.vector
        u = self.variables.u.spline.vector

        # Projector
        self._energy_evaluator = InternalEnergyEvaluator(self.derham, self._gamma)
        self._kinetic_evaluator = KineticEnergyEvaluator(self.derham, self.domain, self.mass_ops)
        self._initialize_projectors_and_mass()
        if self._model in ["linear", "linear_q"]:
            rhotmp = self.projected_equil.n3
        elif self._model in ["deltaf", "deltaf_q"]:
            self._tmp_rho_deltaf = rho.space.zeros()
            rhotmp = rho.copy(out=self._tmp_rho_deltaf)
            rhotmp += self.projected_equil.n3
        else:
            rhotmp = rho
        self._update_weighted_MM(rhotmp)

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

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""

        if self._info:
            logger.info("")
            logger.info("Newton iteration in VariationalDensityEvolve")

        # Initial variables
        rhon = self.variables.rho.spline.vector
        un = self.variables.u.spline.vector

        if self._model in ["linear", "linear_q"]:
            advection = self.divPirho.dot(un, out=self._tmp_rho_advection)
            advection *= dt
            rhon1 = rhon.copy(out=self._tmp_rhon1)
            rhon1 -= advection
            self.update_feec_variables(rho=rhon1, u=un)
            return

        if self._model in ["deltaf", "deltaf_q"]:
            rho = rhon.copy(out=self._tmp_rho_deltaf)
            rho += self.projected_equil.n3
        else:
            rho = rhon
        self._update_weighted_MM(rho)
        mn = self._Mrho.dot(un, out=self._tmp_mn)

        # Initialize variable for Newton iteration
        if self._model == "full":
            s = self._s.spline.vector
        else:
            s = None

        if self._model in ["deltaf", "deltaf_q"]:
            rho = rhon.copy(out=self._tmp_rho_deltaf)
            rho += self.projected_equil.n3
        else:
            rho = rhon
        self._update_Pirho(rho)

        rhon1 = rhon.copy(out=self._tmp_rhon1)
        rhon1 += self._tmp_rhon_diff
        if self._model in ["deltaf", "deltaf_q"]:
            rho = rhon1.copy(out=self._tmp_rho_deltaf)
            rho += self.projected_equil.n3
        else:
            rho = rhon1
        self._update_weighted_MM(rho)
        un1 = un.copy(out=self._tmp_un1)
        un1 += self._tmp_un_diff
        mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)
        tol = self._nonlin_solver.tol
        err = tol + 1

        for it in range(self._nonlin_solver.maxiter):
            # Newton iteration

            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # rhon12 = rhon.copy(out=self._tmp_rhon12)
            # rhon12 += rhon1
            # rhon12 *= 0.5
            # if self._model == "deltaf":
            #     rhon12 += self.projected_equil.n3

            # self._update_Pirho(rhon12)

            # Update the linear form
            self._update_linear_form_dl_drho(rhon, rhon1, un, un1, s)

            # Compute the advection terms
            advection = self.divPirhoT.dot(
                self._linear_form_dl_drho,
                out=self._tmp_advection,
            )
            advection *= dt

            rho_advection = self.divPirho.dot(
                un12,
                out=self._tmp_rho_advection,
            )
            rho_advection *= dt

            # Get diff
            rhon_diff = rhon1.copy(out=self._tmp_rhon_diff)
            rhon_diff -= rhon
            rhon_diff += rho_advection

            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            # Get error
            err = self._get_error_newton(mn_diff, rhon_diff)

            if self._info:
                logger.info(f"iteration : {it} error : {err}")

            if err < tol**2 or xp.isnan(err):
                break

            # Derivative for Newton
            self._get_jacobian(dt, rhon, rhon1, un, un1, s)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = rhon_diff

            incr = self._inv_Jacobian.dot(self._tmp_f, out=self._tmp_incr)
            if self._info:
                logger.info(
                    "information on the linear solver : ",
                    self._inv_Jacobian._solver._info,
                )
            un1 -= incr[0]
            rhon1 -= incr[1]

            # Multiply by the mass matrix to get the momentum

            if self._model in ["deltaf", "deltaf_q"]:
                rho = rhon1.copy(out=self._tmp_rho_deltaf)
                rho += self.projected_equil.n3
                self._update_weighted_MM(rho)
            else:
                self._update_weighted_MM(rhon1)

            mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)

        if it == self._nonlin_solver.maxiter - 1 or xp.isnan(err):
            logger.info(
                f"!!!Warning: Maximum iteration in VariationalDensityEvolve reached - not converged:\n {err =} \n {tol**2 =}",
            )

        self._tmp_un_diff = un1 - un
        self._tmp_rhon_diff = rhon1 - rhon
        self.update_feec_variables(rho=rhon1, u=un1)

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and `CoordinateProjector` needed to compute the bracket term"""

        from struphy.feec.mass import L2Projector
        from struphy.feec.variational_utilities import L2_transport_operator

        # Initialize the transport operator and transposed
        self.divPirho = L2_transport_operator(self.derham)
        self.divPirhoT = self.divPirho.T

        # Inverse mass matrix needed to compute the error
        self.pc_Mv = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.Mv,
        )
        self._inv_Mv = inverse(
            self.mass_ops.Mv,
            "pcg",
            pc=self.pc_Mv,
            tol=1e-16,
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

        self._Jacobian[0, 0] = self._Mrho + self._dt2_pc_divPirhoT @ self._kinetic_evaluator.M_un
        self._Jacobian[0, 1] = self._kinetic_evaluator.M_un1 + self._dt_pc_divPirhoT @ self._M_drho
        self._Jacobian[1, 0] = self._dt2_divPirho
        self._Jacobian[1, 1] = self._I3

        from struphy.linear_algebra.schur_solver import SchurSolverFull

        self._inv_Jacobian = SchurSolverFull(
            self._Jacobian,
            "pbicgstab",
            pc=self._Mrho_inv,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=self._lin_solver.verbose,
            recycle=True,
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

    def _update_Pirho(self, rho):
        """Update the weights of the `BasisProjectionOperator` Pirho"""

        self.divPirho.update_coeffs(rho)
        self.divPirhoT.update_coeffs(rho)

    def _update_weighted_MM(self, rho):
        """update the weighted mass matrix operator"""
        self._Mrho.spline_functions["l2_field"].vector = rho
        self._Mrho.assemble()

        logger.debug(f"In VariationalDensityEvolve: {self._Mrho_inv._options['pc'] = }")
        if hasattr(self, "_Mrho_inv") and isinstance(self._Mrho_inv._options["pc"], MassMatrixDiagonalPreconditioner):
            self._Mrho_inv._options["pc"].update_mass_operator(self._Mrho)

    def _update_linear_form_dl_drho(self, rhon, rhon1, un, un1, sn):
        """Update the linearform representing integration in V3 against kinetic energy"""

        self._kinetic_evaluator.get_u2_grid(un, un1, self._eval_dl_drho)

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

    def _get_jacobian(self, dt, rhon, rhon1, un, un1, sn):
        self._kinetic_evaluator.assemble_M_un(un)
        self._kinetic_evaluator.assemble_M_un1(un1)

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

    def _get_error_newton(self, mn_diff, rhon_diff):
        """Error for the newton method : max(|f(0)|,|f(1)|) where f is the function we're trying to nullify"""
        err_u = self._inv_Mv.dot_inner(
            self.derham.boundary_ops["v"].dot(mn_diff),
            mn_diff,
        )
        err_rho = self.mass_ops.M3.dot_inner(rhon_diff, rhon_diff)
        return max(err_rho, err_u)
