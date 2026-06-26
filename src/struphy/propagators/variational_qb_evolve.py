import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from feectools.linalg.basic import IdentityOperator
from feectools.linalg.block import BlockLinearOperator, BlockVectorSpace
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.feec.mass import L2Projector
from struphy.feec.preconditioner import MassMatrixDiagonalPreconditioner
from struphy.feec.variational_utilities import (
    Hdiv0_transport_operator,
    Pressure_transport_operator,
)
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VariationalQBEvolve(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf u \in (H^1)^3`, :math:`p \in L^2` and :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x - \int_\Omega \mathbf B \cdot \nabla \times (\tilde{\mathbf B} \times \mathbf v) - \int_\Omega \frac{2 q}{\gamma -1} (\nabla \cdot (\tilde{q} \mathbf v))\,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3\,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \cdot ( \tilde{\mathbf B} \times \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t q + \nabla \cdot(\tilde{q} \mathbf u) + (\gamma/2 - 1) \tilde{q} \nabla \cdot u = 0 \,.

    Where :math:`\tilde{\mathbf B}` (resp. :math:`\tilde{q}`) is either :math:`\mathbf B` (resp. :math:`q`) for full-f models, :math:`\mathbf B_0` (resp. :math:`q_0`) for linear models or :math:`\mathbf B_0+\mathbf B` (resp. :math:`q_0+q`) for :math:`\delta f` models.

    On the logical domain:

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta
        - \int_{\hat{\Omega}} \hat{\mathbf{B}}^2 \cdot G \,\nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{v}}) \,\frac{1}{\sqrt g}\,
        - \frac{q}{\gamma -1} \nabla \cdot (\hat{q} \hat{v})  \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{\mathbf{B}}^2 + \nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{u}}) = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{q} + \nabla \cdot (\hat{q} \hat{u}) + (\gamma/2 - 1 ) \hat{p} \nabla \cdot G \hat{u} = 0 \,
        \end{align}

    It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t}
        - (\mathbb C \hat{\Pi}^{1}[\hat{\mathbf B_h^{n}} \cdot \vec{\boldsymbol \Lambda}^v])^\top \mathbb M^2 \mathbf B^{n+\frac{1}{2}}-
        \Big(\big(\mathbb D \hat{\Pi}^{2}[\hat{q_h^n} \cdot \vec{\boldsymbol \Lambda}^v]]
        + (\gamma/2 - 1)\hat{\Pi}^{3}[\hat{q_h^n} \cdot \vec{\boldsymbol \Lambda}^3] \mathbb D \mathcal{U}^v \big) \mathbf v \Big)^\top \hat{l}^3(\frac{2q^{n+\frac{1}{2}}}{\gamma-1}) = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf b^{n+1}- \mathbf b^n}{\Delta t} + \mathbb C \hat{\Pi}^{1}[\hat{\mathbf B_h^{n+\frac{1}{2}}} \cdot \vec{\boldsymbol \Lambda}^v]] \mathbf u^{n+1/2} = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf q^{n+1}- \mathbf q^n}{\Delta t} + \big(\mathbb D \hat{\Pi}^{2}[\hat{q_h^n} \cdot \vec{\boldsymbol \Lambda}^v]]
        + (\gamma/2 - 1)\hat{\Pi}^{3}[\hat{q_h^n} \cdot \vec{\boldsymbol \Lambda}^3] \mathbb D \mathcal{U}^v \big) \mathbf u^{n+1/2}= 0 ~ ,
        \\[2mm]
        \end{align}

    with

    .. math::

        \hat{l}^3(f)_{ijk}=\int_{\hat{\Omega}} f \Lambda^3_{ijk} \textrm d \boldsymbol \eta

    where weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{B}}_h^{n+1/2} = (\mathbf{b}^{n+\frac{1}{2}})^\top \vec{\boldsymbol \Lambda}^2 \in V_h^2 \,
        \qquad \hat{\rho}_h^{n} = (\boldsymbol \rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,
        \qquad \hat{q}_h^{n+1/2} = (\boldsymbol q^{n+1/2})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,.

    and :math:`\mathcal{U}^v` is :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperators`.
    """

    class Variables:
        """Container for variables advanced by :class:`VariationalQBEvolve`.

        Attributes
        ----------
        q : FEECVariable
            Thermal-pressure-like scalar in ``"L2"`` space.
        u : FEECVariable
            Velocity variable in ``"H1vec"`` space.
        b : FEECVariable
            Magnetic-field variable in ``"Hdiv"`` space.
        """

        def __init__(self):
            self._q: FEECVariable = None
            self._u: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def q(self) -> FEECVariable:
            return self._q

        @q.setter
        def q(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._q = new

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

    def __init__(
        self,
        div_u: FEECVariable = None,
        u2: FEECVariable = None,
        qt3: FEECVariable = None,
        bt2: FEECVariable = None,
    ):
        self.variables = self.Variables()
        self.div_u = div_u
        self.u2 = u2
        self.qt3 = qt3
        self.bt2 = bt2

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`VariationalQBEvolve`.

        Parameters
        ----------
        model : {"full_q", "linear_q", "deltaf_q"}, default="full_q"
            Coupled q-B evolution model variant.
        gamma : float, default=5/3
            Adiabatic index.
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Linear solver for implicit substeps.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner used in linear solves.
        solver_params : SolverParameters, default=None
            Linear-solver controls.
        nonlin_solver : NonlinearSolverParameters, default=None
            Nonlinear/Picard iteration controls.
        """

        # specific literals
        OptsModel = Literal["full_q", "linear_q", "deltaf_q"]
        # propagator options
        model: OptsModel = "full_q"
        gamma: float = 5.0 / 3.0
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None

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
    def allocate(self):
        self._model = self.options.model
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize = self.options.nonlin_solver.linearize
        self._gamma = self.options.gamma

        if self.div_u is None:
            self._divu = None
        else:
            self._divu = self.div_u.spline.vector

        if self.u2 is None:
            self._u2 = None
        else:
            self._u2 = self.u2.spline.vector

        if self.qt3 is None:
            self._qt3 = None
        else:
            self._qt3 = self.qt3.spline.vector

        if self.bt2 is None:
            self._bt2 = None
        else:
            self._bt2 = self.bt2.spline.vector

        self._info = self._nonlin_solver.info and (self.rank == 0)

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
        self._initialize_projectors_and_mass()

        # bunch of temporaries to avoid allocating in the loop
        u = self.variables.u.spline.vector
        q = self.variables.q.spline.vector
        b = self.variables.b.spline.vector

        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_bn1 = b.space.zeros()
        self._tmp_bn12 = b.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_bn_diff = b.space.zeros()
        self._tmp_un_weak_diff = u.space.zeros()
        self._tmp_bn_weak_diff = b.space.zeros()
        self._tmp_qn1 = q.space.zeros()
        self._tmp_qn2 = q.space.zeros()
        self._tmp_qn_diff = q.space.zeros()
        self._tmp_qn_weak_diff = q.space.zeros()
        self._linear_form_dl_dq = q.space.zeros()
        self._tmp_qn12 = q.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_advection2 = u.space.zeros()
        self._tmp_b_advection = b.space.zeros()
        self._tmp_b_advection2 = b.space.zeros()
        self._linear_form_dl_db = b.space.zeros()

        self._tmp_q_advection = q.space.zeros()
        self._tmp_q_advection2 = q.space.zeros()

        if self._linearize:
            self._extracted_b2 = self.derham.extraction_ops["2"].dot(self.projected_equil.b2)
            self._extracted_q3 = self.derham.extraction_ops["3"].dot(self.projected_equil.q3)

    def __call__(self, dt):
        if self._nonlin_solver.type == "Picard":
            self.__call_picard(dt)
        else:
            raise ValueError("Only Picard solver is implemented for VariationalQBEvolve")

    def __call_picard(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        # In fact it is linear due to the explicit update, only one iteration will be done at each time step
        if self._info:
            logger.info("")
            logger.info("Newton iteration in VariationalQBEvolve")

        un = self.variables.u.spline.vector
        qn = self.variables.q.spline.vector
        bn = self.variables.b.spline.vector

        self._update_Pib(bn)
        self._update_Projq(qn)

        mn = self._Mrho.dot(un, out=self._tmp_mn)
        bn1 = bn.copy(out=self._tmp_bn1)
        bn1 += self._tmp_bn_diff
        qn1 = qn.copy(out=self._tmp_qn1)
        qn1 += self._tmp_qn_diff
        un1 = un.copy(out=self._tmp_un1)
        un1 += self._tmp_un_diff
        mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)
        tol = self._nonlin_solver.tol
        err = tol + 1

        for it in range(self._nonlin_solver.maxiter):
            # Picard iteration
            # half time step approximation

            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Update the linear form
            self._update_linear_form_dl_db(bn, bn1)
            self._update_linear_form_dl_dq(qn, qn1)

            # Compute the advection terms
            if self._model == "linear_q":
                advection = self.curlPibT0.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection += self.curlPibT.dot(
                    self._linear_form_dl_db0,
                    out=self._tmp_advection2,
                )

                advection += self._transop_q0T.dot(self._linear_form_dl_dq, out=self._tmp_advection2)
                advection += self._transop_qT.dot(self._linear_form_dl_dq0, out=self._tmp_advection2)

                b_advection = self.curlPib0.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                q_advection = self._transop_q0.dot(
                    un12,
                    out=self._tmp_q_advection,
                )

            elif self._model == "deltaf_q":
                advection = self.curlPibT0.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection += self.curlPibT.dot(
                    self._linear_form_dl_db0,
                    out=self._tmp_advection2,
                )

                advection += self.curlPibT.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection2,
                )

                advection += self._transop_qT.dot(self._linear_form_dl_dq, out=self._tmp_advection2)
                advection += self._transop_q0T.dot(self._linear_form_dl_dq, out=self._tmp_advection2)
                advection += self._transop_qT.dot(self._linear_form_dl_dq0, out=self._tmp_advection2)

                b_advection = self.curlPib.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                b_advection += self.curlPib0.dot(
                    un12,
                    out=self._tmp_b_advection2,
                )

                q_advection = self._transop_q0.dot(
                    un12,
                    out=self._tmp_q_advection,
                )

                q_advection += self._transop_q.dot(
                    un12,
                    out=self._tmp_q_advection2,
                )

            else:
                advection = self.curlPibT.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection += self._transop_qT.dot(self._linear_form_dl_dq, out=self._tmp_advection2)

                b_advection = self.curlPib.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                q_advection = self._transop_q.dot(
                    un12,
                    out=self._tmp_q_advection,
                )

            advection *= dt
            b_advection *= dt
            q_advection *= dt

            # Get diff
            bn_diff = bn1.copy(out=self._tmp_bn_diff)
            bn_diff -= bn
            bn_diff += b_advection

            qn_diff = qn1.copy(out=self._tmp_qn_diff)
            qn_diff -= qn
            qn_diff += q_advection

            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            # Get error
            err = self._get_error(mn_diff, bn_diff, qn_diff)

            if self._info:
                logger.info(f"iteration : {it} error : {err}")

            if err < tol**2 or xp.isnan(err):
                break

            # Derivative for Newton
            self._get_jacobian(dt)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = bn_diff
            self._tmp_f[2] = qn_diff

            incr = self._inv_Jacobian.dot(self._tmp_f, out=self._tmp_incr)
            if self._info:
                logger.info(
                    "information on the linear solver : ",
                    self._inv_Jacobian._solver._info,
                )
            un1 -= incr[0]
            bn1 -= incr[1]
            qn1 -= incr[2]

            # Multiply by the mass matrix to get the momentum
            mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)

        if it == self._nonlin_solver.maxiter - 1 or xp.isnan(err):
            logger.info(
                f"!!!Warning: Maximum iteration in VariationalPBEvolve reached - not converged:\n {err =} \n {tol**2 =}",
            )

        self._tmp_un_diff = un1 - un
        self._tmp_bn_diff = bn1 - bn
        self._tmp_qn_diff = qn1 - qn
        self.update_feec_variables(q=qn1, b=bn1, u=un1)

        self._transop_q.div.dot(un12, out=self._divu)
        self._transop_q._Uv.dot(un1, out=self._u2)

        # Update the 2nd order variables

        if self._qt3 is not None:
            q_advection = self._transop_q.dot(
                un12,
                out=self._tmp_q_advection,
            )
            q_advection *= dt
            self._qt3 -= q_advection

        if self._bt2 is not None:
            b_advection = self.curlPib.dot(
                un12,
                out=self._tmp_b_advection,
            )
            b_advection *= dt
            self._bt2 -= b_advection

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        self.curlPib = Hdiv0_transport_operator(self.derham)
        self.curlPibT = self.curlPib.T
        self._transop_q = Pressure_transport_operator(self.derham, self.domain, self.basis_ops.Uv, self._gamma / 2.0)
        self._transop_qT = self._transop_q.T

        integration_grid = [grid_1d.flatten() for grid_1d in self.derham.V3splines.quad_grid_pts[0]]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = (
            self.derham.prepare_eval_tp_fixed(
                integration_grid,
            )
        )

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])

        self._tmp_int_grid = xp.zeros(grid_shape, dtype=float)

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

        self._I2 = IdentityOperator(self.derham.V2pol)
        self._I3 = IdentityOperator(self.derham.V3pol)

        Jacs = BlockVectorSpace(
            self.derham.Vvpol,
            self.derham.V2pol,
            self.derham.V3pol,
        )

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)

        if self._model == "linear_q":
            # initialize the jacobian differently if linear model
            self._create_Pib0()
            self._create_transop0()

            self._linear_form_dl_db0 = -self.mass_ops.M2.dot(self.projected_equil.b2)
            self._linear_form_dl_dq0 = -2 / (self._gamma - 1.0) * self.mass_ops.M3.dot(self.projected_equil.q3)

            self._mdt2_pc_curlPibT_M = 2 * (self.curlPibT0 @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self.curlPib0

            self._mdt2_pc_transopT_M = 2 * (self._transop_q0T @ self.mass_ops.M3)
            self._dt2_transop = 2 * self._transop_q0

        elif self._model == "deltaf_q":
            # initialize the jacobian differently if linear model
            self._create_Pib0()
            self._create_transop0()

            self._full_curlPib = self.curlPib0 + self.curlPib
            self._full_curlPibT = self.curlPibT0 + self.curlPibT

            self._full_transop = self._transop_q0 + self._transop_q
            self._full_transopT = self._transop_q0T + self._transop_qT

            self._linear_form_dl_db0 = -self.mass_ops.M2.dot(self.projected_equil.b2)
            self._linear_form_dl_dq0 = -2 / (self._gamma - 1.0) * self.mass_ops.M3.dot(self.projected_equil.q3)

            self._mdt2_pc_curlPibT_M = 2 * (self._full_curlPibT @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self._full_curlPib

            self._mdt2_pc_transopT_M = 2 * (self._full_transopT @ self.mass_ops.M3)
            self._dt2_transop = 2 * self._full_transop

        else:
            self._mdt2_pc_curlPibT_M = 2 * (self.curlPibT @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self.curlPib

            self._mdt2_pc_transopT_M = 2 * (self._transop_qT @ self.mass_ops.M3)
            self._dt2_transop = 2 * self._transop_q

        self._get_L2dofs_V3 = L2Projector("L2", self.mass_ops).get_dofs
        metric = self.domain.jacobian_det(
            *integration_grid,
        )

        self._energy_metric_term = deepcopy(metric)

        # local version to avoid creating new version of LinearOperator every time

        self._Jacobian[0, 0] = self._Mrho
        self._Jacobian[0, 1] = self._mdt2_pc_curlPibT_M
        self._Jacobian[0, 2] = self._mdt2_pc_transopT_M
        self._Jacobian[1, 0] = self._dt2_curlPib
        self._Jacobian[1, 1] = self._I2
        self._Jacobian[2, 0] = self._dt2_transop
        self._Jacobian[2, 2] = self._I3

        from struphy.linear_algebra.schur_solver import SchurSolverFull3

        self._inv_Jacobian = SchurSolverFull3(
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
        #                          tol=self._lin_solver['tol'],
        #                          maxiter=self._lin_solver['maxiter'],
        #                          verbose=self._lin_solver['verbose'],

        #                     recycle=True)

    def _update_Pib(self, b):
        """Update the weights of the `BasisProjectionOperator`"""

        self.curlPib.update_coeffs(b)
        self.curlPibT.update_coeffs(b)

    def _create_Pib0(self):
        self.curlPib0 = Hdiv0_transport_operator(self.derham)
        self.curlPibT0 = self.curlPib.T
        self.curlPib0.update_coeffs(self.projected_equil.b2)
        self.curlPibT0.update_coeffs(self.projected_equil.b2)

    def _update_Projq(self, q):
        """Update the weights of the `BasisProjectionOperator`"""
        self._transop_q.update_coeffs(q)
        self._transop_qT.update_coeffs(q)

    def _create_transop0(self):
        """Update the weights of the `BasisProjectionOperator`"""

        self._transop_q0 = Pressure_transport_operator(self.derham, self.domain, self.basis_ops.Uv, self._gamma / 2.0)
        self._transop_q0T = self._transop_q0.T
        self._transop_q0.update_coeffs(self.projected_equil.q3)
        self._transop_q0T.update_coeffs(self.projected_equil.q3)

    def _update_linear_form_dl_db(self, bn, bn1):
        """Update the linearform representing integration in V2 derivative of the lagrangian"""
        bn12 = bn.copy(out=self._tmp_bn12)
        bn12 += bn1
        bn12 *= 0.5
        bn12 = bn.copy(out=self._tmp_bn12)
        bn12 += bn1
        bn12 *= 0.5
        if self._linearize:
            wb = self.mass_ops.M2.dot(bn12 - self._extracted_b2, out=self._linear_form_dl_db)
        else:
            wb = self.mass_ops.M2.dot(bn12, out=self._linear_form_dl_db)
        wb *= -1

    def _update_linear_form_dl_dq(self, qn, qn1):
        """Update the linearform representing integration in V2 derivative of the lagrangian"""
        qn12 = qn.copy(out=self._tmp_qn12)
        qn12 += qn1
        qn12 *= 0.5
        if self._linearize:
            wq = self.mass_ops.M3.dot(qn12 - self._extracted_q2, out=self._linear_form_dl_dq)
        else:
            wq = self.mass_ops.M3.dot(qn12, out=self._linear_form_dl_dq)
        wq *= -2 / (self._gamma - 1)

    def _get_error(self, mn_diff, bn_diff, qn_diff):
        err_u = self._inv_Mv.dot_inner(
            self.derham.boundary_ops["v"].dot(mn_diff),
            self.derham.boundary_ops["v"].dot(mn_diff),
        )
        err_b = self.mass_ops.M2.dot_inner(
            bn_diff,
            bn_diff,
        )
        err_q = self.mass_ops.M3.dot_inner(
            qn_diff,
            qn_diff,
        )
        # logger.info("err_b :"+str(err_b))
        # logger.info("err_p :"+str(err_p))
        # logger.info("err_u :"+str(err_u))
        # return max(err_b, err_u)
        return max(max(err_b, err_u), err_q)

    def _get_jacobian(self, dt):
        self._mdt2_pc_curlPibT_M._scalar = -dt / 2
        self._dt2_curlPib._scalar = dt / 2
        self._mdt2_pc_transopT_M._scalar = -dt / (self._gamma - 1.0)
        self._dt2_transop._scalar = dt / 2
