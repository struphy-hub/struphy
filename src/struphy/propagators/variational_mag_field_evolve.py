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
from struphy.feec.preconditioner import MassMatrixDiagonalPreconditioner
from struphy.feec.variational_utilities import Hdiv0_transport_operator
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

    def __init__(self):
        self.variables = self.Variables()

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
        """

        OptsModel = Literal["full", "full_p", "linear"]
        # propagator options
        model: OptsModel = "full"
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
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize = self._nonlin_solver.linearize

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
        self.__call_newton(dt)

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        if self._info:
            logger.info("")
            logger.info("Newton iteration in VariationalMagFieldEvolve")
        # Compute implicit approximation of s^{n+1}
        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        bn1 = bn.copy(out=self._tmp_bn1)
        # Initialize variable for Newton iteration

        self._update_Pib(bn)

        mn = self._Mrho.dot(un, out=self._tmp_mn)
        bn1 = bn.copy(out=self._tmp_bn1)
        bn1 += self._tmp_bn_diff
        un1 = un.copy(out=self._tmp_un1)
        un1 += self._tmp_un_diff
        mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)
        tol = self._nonlin_solver.tol
        err = tol + 1

        for it in range(self._nonlin_solver.maxiter):
            # Newton iteration
            # half time step approximation
            bn12 = bn.copy(out=self._tmp_bn12)
            bn12 += bn1
            bn12 *= 0.5

            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Update the linear form
            self._update_linear_form_dl_db()

            # Compute the advection terms
            if self._model == "linear":
                advection = self.curlPibT0.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection2 = self.curlPibT.dot(
                    self._linear_form_dl_db0,
                    out=self._tmp_advection2,
                )

                advection += advection2

                b_advection = self.curlPib0.dot(
                    un12,
                    out=self._tmp_b_advection,
                )
            else:
                advection = self.curlPibT.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                b_advection = self.curlPib.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

            advection *= dt
            b_advection *= dt

            # Get diff
            bn_diff = bn1.copy(out=self._tmp_bn_diff)
            bn_diff -= bn
            bn_diff += b_advection

            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            # Get error
            err = self._get_error_newton(mn_diff, bn_diff)

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
                f"!!!Warning: Maximum iteration in VariationalMagFieldEvolve reached - not converged:\n {err =} \n {tol**2 =}",
            )

        self._tmp_un_diff = un1 - un
        self._tmp_bn_diff = bn1 - bn
        self.update_feec_variables(b=bn1, u=un1)

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
            tol=1e-16,
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

        self._Jacobian[0, 0] = self._Mrho
        self._Jacobian[0, 1] = self._mdt2_pc_curlPibT_M
        self._Jacobian[1, 0] = self._dt2_curlPib
        self._Jacobian[1, 1] = self._I2

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
        err_u = self._inv_Mv.dot_inner(self.derham.boundary_ops["v"].dot(mn_diff), mn_diff)
        err_b = self.mass_ops.M2.dot_inner(bn_diff, bn_diff)
        return max(err_b, err_u)

    def _get_jacobian(self, dt):
        self._mdt2_pc_curlPibT_M._scalar = -dt / 2
        self._dt2_curlPib._scalar = dt / 2
