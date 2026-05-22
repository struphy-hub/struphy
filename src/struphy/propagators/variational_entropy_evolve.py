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
from struphy.feec.variational_utilities import InternalEnergyEvaluator
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VariationalEntropyEvolve(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf u \in (H^1)^3` and :math:`s \in L^2` such that

    .. math::

        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x - \int_\Omega \frac{\partial(\rho \mathcal U)}{\partial s} \nabla \cdot (s \mathbf v) \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3\,,
        \\[4mm]
        &\partial_t s + \nabla \cdot ( s \mathbf u ) = 0 \,.

    On the logical domain:

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta
        - \int_{\hat{\Omega}} \left(\frac{\partial \hat{\rho}^3 \mathcal U}{\partial \hat{s}} \right) \nabla \cdot (\hat{s} \hat{\mathbf{v}}) \, \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{s} + \nabla \cdot ( \hat{s} \hat{\mathbf{u}} ) = 0 ~ ,
        \end{align}

    where :math:`\mathcal U` depends on the chosen model. It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t} -
        (\mathbb D \hat{\Pi}^{2}[\hat{s_h^{n}} \vec{\boldsymbol \Lambda}^v])^\top \hat{l}^3\left( \frac{\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n},\hat{s}_h^{n+1})-\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n},\hat{s}_h^{n})}{\hat{s}_h^{n+1}-\hat{s}_h^n} \right) = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf s^{n+1}- \mathbf s^n}{\Delta t} + \mathbb D \hat{\Pi}^{2}[\hat{s_h^{n}} \vec{\boldsymbol \Lambda}^v] \mathbf u^{n+1/2} = 0 ~ ,
        \\[2mm]
        \end{align}

    where :math:`\hat{l}^3(f)` denotes the vector representing the linear form :math:`v_h \mapsto \int_{\hat{\Omega}} f(\boldsymbol \eta) v_h(\boldsymbol \eta) d \boldsymbol \eta`, that is the vector with components

    .. math::
        \hat{l}^3(f)_{ijk}=\int_{\hat{\Omega}} f \Lambda^3_{ijk} \textrm d \boldsymbol \eta

    and the weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{u}}_h^{k} = (\mathbf{u}^{k})^\top \vec{\boldsymbol \Lambda}^v \in (V_h^0)^3 \, \text{for k in} \{n, n+1/2, n+1\}, \qquad \hat{s}_h^{k} = (s^{k})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \, \text{for k in} \{n, n+1/2, n+1\} \qquad \hat{\rho}_h^{n} = (\rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \.
    """

    class Variables:
        """Container for variables advanced by :class:`VariationalEntropyEvolve`.

        Attributes
        ----------
        s : FEECVariable
            Entropy variable in ``"L2"`` space.
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

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`VariationalEntropyEvolve`.

        Parameters
        ----------
        model : {"full"}, default="full"
            Entropy-evolution model selection.
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
        rho : FEECVariable, default=None
            Density variable used in variational forms.
        """

        # specific literals
        OptsModel = Literal["full"]
        # propagator options
        model: OptsModel = "full"
        gamma: float = 5.0 / 3.0
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        rho: FEECVariable = None

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
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self, verbose: bool = False):
        if self.options.model == "full":
            assert self.options.rho is not None

        self._model = self.options.model
        self._gamma = self.options.gamma
        self._rho = self.options.rho
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize = self.options.nonlin_solver.linearize

        self._info = self._nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        # assembly of WMMnew happens in VariationalDensityEvolve
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

        # Projector
        self._energy_evaluator = InternalEnergyEvaluator(self.derham, self._gamma)
        self._initialize_projectors_and_mass()

        # bunch of temporaries to avoid allocating in the loop
        s = self.variables.s.spline.vector
        u = self.variables.u.spline.vector

        self._tmp_un1 = u.space.zeros()
        self._tmp_un2 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_sn1 = s.space.zeros()
        self._tmp_sn12 = s.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_sn_diff = s.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_un_weak_diff = u.space.zeros()
        self._tmp_sn_weak_diff = s.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_mn12 = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_s_advection = s.space.zeros()
        self._linear_form_dl_ds = s.space.zeros()
        if self._linearize:
            self._compute_init_linear_form()

    def __call__(self, dt):
        self.__call_newton(dt)

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        if self._info:
            logger.info("")
            logger.info("Newton iteration in VariationalEntropyEvolve")
        sn = self.variables.s.spline.vector
        un = self.variables.u.spline.vector

        sn1 = sn.copy(out=self._tmp_sn1)
        # Initialize variable for Newton iteration
        rho = self._rho.spline.vector
        self._update_Pis(sn)

        mn = self._Mrho.dot(un, out=self._tmp_mn)
        sn1 = sn.copy(out=self._tmp_sn1)
        sn1 += self._tmp_sn_diff
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

            # Update the linear form
            self._update_linear_form_dl_ds(rho, sn, sn1)

            # Compute the advection terms
            advection = self.divPisT.dot(
                self._linear_form_dl_ds,
                out=self._tmp_advection,
            )
            advection *= dt

            s_advection = self.divPis.dot(
                un12,
                out=self._tmp_s_advection,
            )
            s_advection *= dt

            # Get diff
            sn_diff = sn1.copy(out=self._tmp_sn_diff)
            sn_diff -= sn
            sn_diff += s_advection

            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            # Get error
            err = self._get_error_newton(mn_diff, sn_diff)

            if self._info:
                logger.info(f"iteration : {it} error : {err}")

            if err < tol**2 or xp.isnan(err):
                break

            # Derivative for Newton
            self._get_jacobian(dt, rho, sn, sn1)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = sn_diff

            incr = self._inv_Jacobian.dot(self._tmp_f, out=self._tmp_incr)
            if self._info:
                logger.info(
                    "information on the linear solver : ",
                    self._inv_Jacobian._solver._info,
                )
            un1 -= incr[0]
            sn1 -= incr[1]

            # Multiply by the mass matrix to get the momentum
            mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)

        if it == self._nonlin_solver.maxiter - 1 or xp.isnan(err):
            logger.info(
                f"!!!Warning: Maximum iteration in VariationalEntropyEvolve reached - not converged:\n {err =} \n {tol**2 =}",
            )
        self._tmp_sn_diff = sn1 - sn
        self._tmp_un_diff = un1 - un
        self.update_feec_variables(s=sn1, u=un1)

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and `CoordinateProjector` needed to compute the bracket term"""

        from struphy.feec.mass import L2Projector
        from struphy.feec.variational_utilities import L2_transport_operator

        # Initialize the transport operator and transposed
        self.divPis = L2_transport_operator(self.derham)
        self.divPisT = self.divPis.T

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
        )

        # For Newton solve
        self._M_ds = self.mass_ops.create_weighted_mass("L2", "L2")

        Jacs = BlockVectorSpace(
            self.derham.Vvpol,
            self.derham.V3pol,
        )

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)

        self._I3 = IdentityOperator(self.derham.V3pol)

        # local version to avoid creating new version of LinearOperator every time
        self._dt_pc_divPisT = 2 * (self.divPisT)
        self._dt2_divPis = 2 * self.divPis

        self._Jacobian[0, 0] = self._Mrho
        self._Jacobian[0, 1] = self._dt_pc_divPisT @ self._M_ds
        self._Jacobian[1, 0] = self._dt2_divPis
        self._Jacobian[1, 1] = self._I3

        from struphy.linear_algebra.schur_solver import SchurSolverFull

        self._inv_Jacobian = SchurSolverFull(
            self._Jacobian,
            self.options.solver,
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

        # prepare for integration of linear form
        # L2-projector for V3
        self._get_L2dofs_V3 = L2Projector("L2", self.mass_ops).get_dofs

        integration_grid = [grid_1d.flatten() for grid_1d in self.derham.V3splines.quad_grid_pts[0]]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = (
            self.derham.prepare_eval_tp_fixed(
                integration_grid,
            )
        )

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])
        self._tmp_int_grid = xp.zeros(grid_shape, dtype=float)

        if self._model == "full":
            self._tmp_de_ds = xp.zeros(grid_shape, dtype=float)
            if self._linearize:
                self._init_dener_ds = xp.zeros(grid_shape, dtype=float)

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
            self._proj_ds_metric_term = deepcopy(metric)

    def _update_Pis(self, s):
        """Update the weights of the `BasisProjectionOperator`"""

        self.divPis.update_coeffs(s)
        self.divPisT.update_coeffs(s)

    def _update_linear_form_dl_ds(self, rhon, sn, sn1):
        """Update the linear form representing integration in V3 against the derivative of the lagrangian"""

        if self._model == "full":
            self._energy_evaluator.evaluate_discrete_de_ds_grid(rhon, sn, sn1, out=self._tmp_de_ds)

            self._tmp_int_grid *= 0
            self._tmp_int_grid += self._tmp_de_ds

            if self._linearize:
                self._tmp_int_grid -= self._init_dener_ds
            self._tmp_int_grid *= self._proj_rho2_metric_term
            self._tmp_int_grid *= -1.0

        self._get_L2dofs_V3(self._tmp_int_grid, dofs=self._linear_form_dl_ds)

    def _compute_init_linear_form(self):
        if abs(self._gamma - 5 / 3) < 1e-3:
            self._energy_evaluator.evaluate_exact_de_ds_grid(
                self.projected_equil.n3,
                self.projected_equil.s3_monoatomic,
                out=self._init_dener_ds,
            )
        elif abs(self._gamma - 7 / 5) < 1e-3:
            self._energy_evaluator.evaluate_exact_de_ds_grid(
                self.projected_equil.n3,
                self.projected_equil.s3_diatomic,
                out=self._init_dener_ds,
            )
        else:
            raise ValueError("Gamma should be 7/5 or 5/3 for if you want to linearize")

    def _get_jacobian(self, dt, rhon, sn, sn1):
        if self._model == "full":
            self._energy_evaluator.evaluate_discrete_d2e_ds2_grid(rhon, sn, sn1, out=self._tmp_int_grid)
            self._tmp_int_grid *= self._proj_ds_metric_term

            self._M_ds.assemble([[self._tmp_int_grid]])

        # This way we can update only the scalar multiplying the operator and avoid creating multiple operators
        self._dt_pc_divPisT._scalar = dt
        self._dt2_divPis._scalar = dt / 2

    def _get_error_newton(self, mn_diff, sn_diff):
        err_u = self._inv_Mv.dot_inner(self.derham.boundary_ops["v"].dot(mn_diff), mn_diff)
        err_rho = self.mass_ops.M3.dot_inner(sn_diff, sn_diff)
        return max(err_rho, err_u)
