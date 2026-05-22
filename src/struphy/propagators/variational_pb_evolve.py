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


class VariationalPBEvolve(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf u \in (H^1)^3`, :math:`p \in L^2` and :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x - \int_\Omega \mathbf B \cdot \nabla \times (\tilde{\mathbf B} \times \mathbf v) - \int_\Omega \frac{1}{\gamma -1} (\nabla \cdot (\tilde{p} \mathbf v))\,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3\,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \cdot ( \tilde{\mathbf B} \times \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t p + \nabla \cdot(\tilde{p} \mathbf u) + (\gamma - 1) \tilde{p} \nabla \cdot u = 0 \,.

    Where :math:`\tilde{\mathbf B}` (resp. :math:`\tilde{p}`) is either :math:`\mathbf B` (resp. :math:`p`) for full-f models, :math:`\mathbf B_0` (resp. :math:`p_0`) for linear models or :math:`\mathbf B_0+\mathbf B` (resp. :math:`p_0+p`) for :math:`\delta f` models.

    On the logical domain:

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta
        - \int_{\hat{\Omega}} \hat{\mathbf{B}}^2 \cdot G \,\nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{v}}) \,\frac{1}{\sqrt g}\,
        - \frac{g}{\gamma -1} \nabla \cdot (\hat{p} \hat{v})  \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{\mathbf{B}}^2 + \nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{u}}) = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{p} + \nabla \cdot (\hat{p} \hat{u}) + (\gamma - 1 ) \hat{p} \nabla \cdot G \hat{u} = 0 \,
        \end{align}

    It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t}
        - (\mathbb C \hat{\Pi}^{1}[\hat{\mathbf B_h^{n+\frac{1}{2}}} \cdot \vec{\boldsymbol \Lambda}^v])^\top \mathbb M^2 \mathbf B^{n+\frac{1}{2}} = 0 ~ ,
        - (\mathbb D \hat{\Pi}^{2}[\hat{p_h^{n+\frac{1}{2}}} \cdot \vec{\boldsymbol \Lambda}^v])^\top \hat{l}^3(\frac{g}{\gamma-1})
        \\[2mm]
        &\frac{\mathbf b^{n+1}- \mathbf b^n}{\Delta t} + \mathbb C \hat{\Pi}^{1}[\hat{\mathbf B_h^{n+\frac{1}{2}}} \cdot \vec{\boldsymbol \Lambda}^v]] \mathbf u^{n+1/2} = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf p^{n+1}- \mathbf p^n}{\Delta t} + \big(\mathbb D \hat{\Pi}^{2}[\hat{p_h^{n+\frac{1}{2}}} \cdot \vec{\boldsymbol \Lambda}^v]]
        + (\gamma - 1)\hat{\Pi}^{3}[\hat{p_h^{n+\frac{1}{2}}} \cdot \vec{\boldsymbol \Lambda}^3] \mathbb D \mathcal{U}^v \big) \mathbf u^{n+1/2}= 0 ~ ,
        \\[2mm]
        \end{align}

    with

    .. math::

        \hat{l}^3(f)_{ijk}=\int_{\hat{\Omega}} f \Lambda^3_{ijk} \textrm d \boldsymbol \eta

    where weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{B}}_h^{n+1/2} = (\mathbf{b}^{n+\frac{1}{2}})^\top \vec{\boldsymbol \Lambda}^2 \in V_h^2 \,
        \qquad \hat{\rho}_h^{n} = (\boldsymbol \rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,
        \qquad \hat{p}_h^{n+1/2} = (\boldsymbol p^{n+1/2})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,.

    and :math:`\mathcal{U}^v` is :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperators`.
    """

    class Variables:
        """Container for variables advanced by :class:`VariationalPBEvolve`.

        Attributes
        ----------
        p : FEECVariable
            Pressure variable in ``"L2"`` space.
        u : FEECVariable
            Velocity variable in ``"H1vec"`` space.
        b : FEECVariable
            Magnetic-field variable in ``"Hdiv"`` space.
        """

        def __init__(self):
            self._p: FEECVariable = None
            self._u: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def p(self) -> FEECVariable:
            return self._p

        @p.setter
        def p(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._p = new

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

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`VariationalPBEvolve`.

        Parameters
        ----------
        model : {"full_p", "linear", "deltaf"}, default="full_p"
            Pressure-magnetic coupled model variant.
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
        div_u : FEECVariable, default=None
            Optional external divergence-of-velocity field.
        u2 : FEECVariable, default=None
            Optional external velocity in 2-form representation.
        pt3 : FEECVariable, default=None
            Optional pressure background field.
        bt2 : FEECVariable, default=None
            Optional magnetic background field.
        """

        # specific literals
        OptsModel = Literal["full_p", "linear", "deltaf"]
        # propagator options
        model: OptsModel = "full_p"
        gamma: float = 5.0 / 3.0
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        div_u: FEECVariable = None
        u2: FEECVariable = None
        pt3: FEECVariable = None
        bt2: FEECVariable = None

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
        self._model = self.options.model
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize = self.options.nonlin_solver.linearize
        self._gamma = self.options.gamma

        if self.options.div_u is None:
            self._divu = None
        else:
            self._divu = self.options.div_u.spline.vector

        if self.options.u2 is None:
            self._u2 = None
        else:
            self._u2 = self.options.u2.spline.vector

        if self.options.pt3 is None:
            self._pt3 = None
        else:
            self._pt3 = self.options.pt3.spline.vector

        if self.options.bt2 is None:
            self._bt2 = None
        else:
            self._bt2 = self.options.bt2.spline.vector

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
        self._initialize_projectors_and_mass()

        # bunch of temporaries to avoid allocating in the loop
        u = self.variables.u.spline.vector
        p = self.variables.p.spline.vector
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
        self._tmp_pn1 = p.space.zeros()
        self._tmp_pn2 = p.space.zeros()
        self._tmp_pn_diff = p.space.zeros()
        self._tmp_pn_weak_diff = p.space.zeros()
        self._linear_form_dl_dp = p.space.zeros()
        self._tmp_pn12 = p.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_advection2 = u.space.zeros()
        self._tmp_b_advection = b.space.zeros()
        self._tmp_b_advection2 = b.space.zeros()
        self._linear_form_dl_db = b.space.zeros()

        self._tmp_p_advection = p.space.zeros()
        self._tmp_p_advection2 = p.space.zeros()

        self._create_linear_form_p()

        if self._linearize:
            self._extracted_b2 = self.derham.extraction_ops["2"].dot(self.projected_equil.b2)

    def __call__(self, dt):
        if self._nonlin_solver.type == "Picard":
            self.__call_picard(dt)
        else:
            raise ValueError("Only Picard solver is implemented for VariationalPBEvolve")

    def __call_picard(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        # In fact it is linear due to the explicit update, only one iteration will be done at each time step
        if self._info:
            logger.info("")
            logger.info("Newton iteration in VariationalPBEvolve")

        un = self.variables.u.spline.vector
        pn = self.variables.p.spline.vector
        bn = self.variables.b.spline.vector

        self._update_Pib(bn)
        self._update_Projp(pn)

        mn = self._Mrho.dot(un, out=self._tmp_mn)
        bn1 = bn.copy(out=self._tmp_bn1)
        bn1 += self._tmp_bn_diff
        pn1 = pn.copy(out=self._tmp_pn1)
        pn1 += self._tmp_pn_diff
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

            # pn12 = pn.copy(out=self._tmp_pn12)
            # pn12 += pn1
            # pn12 *= 0.5

            # self._update_Pib()
            # self._update_Projp()
            # Update the linear form
            self._update_linear_form_dl_db(bn, bn1)

            # Compute the advection terms
            if self._model == "linear":
                advection = self.curlPibT0.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection += self.curlPibT.dot(
                    self._linear_form_dl_db0,
                    out=self._tmp_advection2,
                )

                advection += self._transop_pT.dot(self._linear_form_dl_dp, out=self._tmp_advection2)

                b_advection = self.curlPib0.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                p_advection = self._transop_p0.dot(
                    un12,
                    out=self._tmp_p_advection,
                )

            elif self._model == "deltaf":
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

                advection += self._transop_pT.dot(self._linear_form_dl_dp, out=self._tmp_advection2)

                b_advection = self.curlPib.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                b_advection += self.curlPib0.dot(
                    un12,
                    out=self._tmp_b_advection2,
                )

                p_advection = self._transop_p0.dot(
                    un12,
                    out=self._tmp_p_advection,
                )

                p_advection += self._transop_p.dot(
                    un12,
                    out=self._tmp_p_advection2,
                )

            else:
                advection = self.curlPibT.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection2 = self._transop_pT.dot(self._linear_form_dl_dp, out=self._tmp_advection2)

                advection += advection2

                b_advection = self.curlPib.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                p_advection = self._transop_p.dot(
                    un12,
                    out=self._tmp_p_advection,
                )

            advection *= dt
            b_advection *= dt
            p_advection *= dt

            # Get diff
            bn_diff = bn1.copy(out=self._tmp_bn_diff)
            bn_diff -= bn
            bn_diff += b_advection

            # pn_diff = pn1.copy(out= self._tmp_pn_diff)
            # pn_diff -= pn
            # pn_diff += p_advection

            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            pn1 = pn.copy(out=self._tmp_pn1)
            pn1 -= p_advection

            # Get error
            err = self._get_error(mn_diff, bn_diff)  # , pn_diff)

            if self._info:
                logger.info(f"iteration : {it} error : {err}")

            if err < tol**2 or xp.isnan(err):
                break

            # Derivative for Newton
            self._get_jacobian(dt)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = bn_diff

            incr = self._inv_Jacobian.dot(self._tmp_f, out=self._tmp_incr)
            if self._info:
                logger.info(
                    "information on the linear solver : ",
                    self._inv_Jacobian._solver._info,
                )
            un1 -= incr[0]
            bn1 -= incr[1]

            # Multiply by the mass matrix to get the momentum
            mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)

        if it == self._nonlin_solver.maxiter - 1 or xp.isnan(err):
            logger.info(
                f"!!!Warning: Maximum iteration in VariationalPBEvolve reached - not converged:\n {err =} \n {tol**2 =}",
            )

        self._tmp_un_diff = un1 - un
        self._tmp_bn_diff = bn1 - bn
        self._tmp_pn_diff = pn1 - pn
        self.update_feec_variables(p=pn1, b=bn1, u=un1)

        self._transop_p.div.dot(un12, out=self._divu)
        self._transop_p._Uv.dot(un1, out=self._u2)

        # Update the 2nd order variables

        if self._pt3 is not None:
            p_advection = self._transop_p.dot(
                un12,
                out=self._tmp_p_advection,
            )
            p_advection *= dt
            self._pt3 -= p_advection

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
        self._transop_p = Pressure_transport_operator(self.derham, self.domain, self.basis_ops.Uv, self._gamma)
        self._transop_pT = self._transop_p.T

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

        Jacs = BlockVectorSpace(
            self.derham.Vvpol,
            self.derham.V2pol,
        )

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)

        if self._model == "linear":
            # initialize the jacobian differently if linear model
            self._create_Pib0()
            self._create_transop0()

            self._linear_form_dl_db0 = -self.mass_ops.M2.dot(self.projected_equil.b2)

            self._mdt2_pc_curlPibT_M = 2 * (self.curlPibT0 @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self.curlPib0

        elif self._model == "deltaf":
            # initialize the jacobian differently if linear model
            self._create_Pib0()
            self._create_transop0()

            self._full_curlPib = self.curlPib0 + self.curlPib
            self._full_curlPibT = self.curlPibT0 + self.curlPibT

            self._linear_form_dl_db0 = -self.mass_ops.M2.dot(self.projected_equil.b2)

            self._mdt2_pc_curlPibT_M = 2 * (self._full_curlPibT @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self._full_curlPib

        else:
            self._mdt2_pc_curlPibT_M = 2 * (self.curlPibT @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self.curlPib

        self._get_L2dofs_V3 = L2Projector("L2", self.mass_ops).get_dofs
        metric = self.domain.jacobian_det(
            *integration_grid,
        )

        self._energy_metric_term = deepcopy(metric)

        # local version to avoid creating new version of LinearOperator every time

        self._Jacobian[0, 0] = self._Mrho
        self._Jacobian[0, 1] = self._mdt2_pc_curlPibT_M
        self._Jacobian[1, 0] = self._dt2_curlPib
        self._Jacobian[1, 1] = self._I2

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
        #                          tol=self._lin_solver['tol'],
        #                          maxiter=self._lin_solver['maxiter'],
        #                          verbose=self._lin_solver['verbose'],
        #
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

    def _update_Projp(self, p):
        """Update the weights of the `BasisProjectionOperator`"""
        self._transop_p.update_coeffs(p)
        self._transop_pT.update_coeffs(p)

    def _create_transop0(self):
        """Update the weights of the `BasisProjectionOperator`"""

        self._transop_p0 = Pressure_transport_operator(self.derham, self.domain, self.basis_ops.Uv, self._gamma)
        self._transop_p0T = self._transop_p0.T
        self._transop_p0.update_coeffs(self.projected_equil.p3)
        self._transop_p0T.update_coeffs(self.projected_equil.p3)

    def _update_linear_form_dl_db(self, bn, bn1):
        """Update the linearform representing integration in V2 derivative of the lagrangian"""
        bn12 = bn.copy(out=self._tmp_bn12)
        bn12 += bn1
        bn12 *= 0.5
        if self._linearize:
            wb = self.mass_ops.M2.dot(bn12 - self._extracted_b2, out=self._linear_form_dl_db)
        else:
            wb = self.mass_ops.M2.dot(bn12, out=self._linear_form_dl_db)
        wb *= -1

    def _create_linear_form_p(self):
        """Update the linearform representing integration in V3 against pressure energy"""

        if self._model in ["full_p", "linear", "deltaf"]:
            self._tmp_int_grid *= 0.0
            self._tmp_int_grid -= 1.0 / (self._gamma - 1.0)
            self._tmp_int_grid *= self._energy_metric_term

        self._get_L2dofs_V3(self._tmp_int_grid, dofs=self._linear_form_dl_dp)

    def _get_error(self, mn_diff, bn_diff):  # , pn_diff):
        err_u = self._inv_Mv.dot_inner(self.derham.boundary_ops["v"].dot(mn_diff), mn_diff)
        err_b = self.mass_ops.M2.dot_inner(bn_diff, bn_diff)
        # weak_pn_diff = self.mass_ops.M3.dot(
        #     pn_diff,
        #     out=self._tmp_pn_weak_diff,
        # )
        # err_p = weak_pn_diff.dot(pn_diff)
        # logger.info("err_b :"+str(err_b))
        # logger.info("err_p :"+str(err_p))
        # logger.info("err_u :"+str(err_u))
        return max(err_b, err_u)
        # return max(max(err_b, err_u),err_p)

    def _get_jacobian(self, dt):
        self._mdt2_pc_curlPibT_M._scalar = -dt / 2
        self._dt2_curlPib._scalar = dt / 2
