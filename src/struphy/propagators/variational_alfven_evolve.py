import logging
from dataclasses import dataclass
from typing import Literal

from feectools.linalg.basic import IdentityOperator
from feectools.linalg.block import BlockLinearOperator, BlockVectorSpace
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.feec.preconditioner import MassMatrixPreconditioner
from struphy.feec.variational_utilities import (
    Hdiv0_transport_operator,
    NewtonConvergence,
    alfven_schur_preconditioner,
    dot_normalized,
    inv_norm2,
)
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolverFull
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VariationalAlfvenEvolve(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the coupling of the velocity and the magnetic field (shear-Alfvén and
    fast magnetosonic waves) of the variational MHD models: find :math:`\mathbf u \in (H^1)^3` and
    :math:`\mathbf B \in H(\textnormal{div})` such that, with the transport operator
    :math:`\mathcal T_{\bar{\mathbf B}}(\mathbf v) = \nabla \times (\bar{\mathbf B} \times \mathbf v)`, depending on the
    option ``model``:

    * ``"full"`` (full MHD):

      .. math::

          \int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x
          - \int_\Omega \mathbf B \cdot \mathcal T_{\mathbf B}(\mathbf v)\,\textrm d \mathbf x = 0
          \qquad \forall \, \mathbf v \in (H^1)^3\,,
          \qquad
          \partial_t \mathbf B + \mathcal T_{\mathbf B}(\mathbf u) = 0 \,.

    * ``"linear"`` (:math:`\mathbf B` is the perturbation of the equilibrium :math:`\mathbf B_0`):

      .. math::

          \int_\Omega \partial_t (\rho_0 \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x
          - \int_\Omega \left(\mathbf B \cdot \mathcal T_{\mathbf B_0}(\mathbf v)
          + \mathbf B_0 \cdot \mathcal T_{\mathbf B}(\mathbf v)\right)\textrm d \mathbf x = 0\,,
          \qquad
          \partial_t \mathbf B + \mathcal T_{\mathbf B_0}(\mathbf u) = 0 \,.

    * ``"perturbative"`` (:math:`\mathbf B` is the perturbation, all nonlinear terms are kept):

      .. math::

          \int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x
          - \int_\Omega \left(\mathbf B \cdot \mathcal T_{\mathbf B_0 + \mathbf B}(\mathbf v)
          + \mathbf B_0 \cdot \mathcal T_{\mathbf B}(\mathbf v)\right)\textrm d \mathbf x = 0\,,
          \qquad
          \partial_t \mathbf B + \mathcal T_{\mathbf B_0 + \mathbf B}(\mathbf u) = 0 \,.

    The thermodynamic variable is not changed, so this propagator serves the models in :math:`p` and in
    :math:`q = \sqrt p` alike; together with :class:`~struphy.propagators.variational_sound_evolve.VariationalSoundEvolve`
    it splits the coupled step of :class:`~struphy.propagators.variational_qb_evolve.VariationalQBEvolve` into the
    shear-Alfvén/fast and the sound waves. The second-order term of the linear and perturbative models comes from the
    auxiliary variable :math:`\hat{\mathbf B}` (``bt2``), which, if given, is advanced with
    :math:`\partial_t \hat{\mathbf B} + \mathcal T_{\mathbf B}(\mathbf u) = 0`; it enters the conserved energy as
    :math:`\int \mathbf B_0 \cdot \hat{\mathbf B}`. For ``"perturbative"``, :math:`\rho` is :math:`\rho_0` plus the
    variable of :class:`~struphy.propagators.variational_density_evolve.VariationalDensityEvolve`.

    Discretization: with :math:`\mathbb T[\bar{\mathbf B}] = \mathbb C \hat\Pi^1[\hat{\bar{\mathbf B}} \times
    \vec{\boldsymbol\Lambda}^v]` (:class:`~struphy.feec.variational_utilities.Hdiv0_transport_operator`) and, for
    ``"linear"``,

    .. math::

        \begin{align}
        &\mathbb M^v[\hat\rho_0] \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t}
        - \mathbb T[\mathbf B_0]^\top \mathbb M^2 \mathbf b^{n+\frac12}
        - \mathbb T[\bar{\mathbf b}]^\top \mathbb M^2 \mathbf b_0 = 0 \,,
        \\[2mm]
        &\frac{\mathbf b^{n+1} - \mathbf b^n}{\Delta t} + \mathbb T[\mathbf B_0]\, \mathbf u^{n+\frac12} = 0 \,,
        \qquad
        \frac{\hat{\mathbf b}^{n+1} - \hat{\mathbf b}^n}{\Delta t} + \mathbb T[\bar{\mathbf b}]\, \mathbf u^{n+\frac12} = 0 \,,
        \end{align}

    and correspondingly for the other models. The weight :math:`\bar{\mathbf b}` of the transport operators is taken
    at the same time in all equations, which makes the step energy conserving: at :math:`t^n` (first order in time,
    one Newton step solves the linear system), or at the midpoint :math:`t^{n+1/2}` with the option
    ``implicit_weights`` (second order in time, a Picard iteration). The Newton system is solved with
    :class:`~struphy.linear_algebra.schur_solver.SchurSolverFull`; its Schur complement for ``"linear"`` is the
    shear-Alfvén operator :math:`\mathbb M^v[\hat\rho_0] + \frac{\Delta t^2}{4} \mathbb T[\mathbf B_0]^\top \mathbb M^2
    \mathbb T[\mathbf B_0]`, for which the option ``schur_precond`` provides a Kronecker preconditioner.
    """

    class Variables:
        """Container for variables advanced by :class:`VariationalAlfvenEvolve`.

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

    def __init__(self, bt2: FEECVariable = None):
        r"""
        Parameters
        ----------
        bt2 : FEECVariable, default=None
            Auxiliary second-order variable :math:`\hat{\mathbf B}` (``"Hdiv"`` space) of the linear and perturbative
            models, advanced with :math:`\partial_t \hat{\mathbf B} + \mathcal T_{\mathbf B}(\mathbf u) = 0`. It enters
            only the energy, not the momentum equation.
        """
        self.variables = self.Variables()
        self.bt2 = bt2

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`VariationalAlfvenEvolve`.

        Parameters
        ----------
        model : {"full", "linear", "perturbative"}, default="full"
            Model variant.
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Linear solver of the Schur system.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner used in linear solves.
        solver_params : SolverParameters, default=None
            Linear-solver controls. The Newton systems are solved for a normalized right-hand side, so ``tol`` is
            relative to the norm of the residual.
        nonlin_solver : NonlinearSolverParameters, default=None
            Nonlinear/Picard iteration controls. The iteration stops when the norm of the residual is below ``tol``
            times the norm of the state at :math:`t^n` if that is smaller than one (absolute otherwise), or when it
            stagnates (with a warning).
        implicit_weights : bool, default=None
            If ``True``, the weight of the transport operators is taken at the midpoint :math:`t^{n+1/2}` and updated
            in every Picard iteration (second order in time, several iterations per step); if ``False``, it is taken
            at :math:`t^n` (first order in time, one linear solve). Both conserve the energy. ``None`` means ``True``
            for ``"linear"`` and ``"perturbative"`` and ``False`` for ``"full"``.
        schur_precond : {"MassMatrixPreconditioner", "StiffnessPreconditioner"}, default="MassMatrixPreconditioner"
            Preconditioner of the Schur system: the mass-matrix preconditioner of the velocity block, or the Kronecker
            (fast diagonalization) preconditioner of the whole shear-Alfvén Schur operator
            (:func:`~struphy.feec.variational_utilities.alfven_schur_preconditioner`, built for each time step in
            :meth:`prepare_time_step`). The latter needs ``model="linear"``, ``solver="pcg"``, tensor-product splines
            and the global commuting projectors; otherwise the mass-matrix preconditioner is used (with a warning).
        exb_correction : bool, default=True
            For ``schur_precond="StiffnessPreconditioner"``: add the auxiliary space correction of the
            :math:`\mathbf E \times \mathbf B` flows :math:`\mathbf B_0 \times \nabla \phi`.
        """

        # specific literals
        OptsModel = Literal["full", "linear", "perturbative"]
        OptsSchurPrecond = Literal["MassMatrixPreconditioner", "StiffnessPreconditioner"]
        # propagator options
        model: OptsModel = "full"
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        implicit_weights: bool | None = None
        schur_precond: OptsSchurPrecond = "MassMatrixPreconditioner"
        exb_correction: bool = True

        def __post_init__(self):
            # checks
            check_option(self.model, self.OptsModel)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
            check_option(self.schur_precond, self.OptsSchurPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()
            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters()
            if self.implicit_weights is None:
                self.implicit_weights = self.model in ("linear", "perturbative")

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
        self._implicit_weights = self.options.implicit_weights
        self._bt2 = None if self.bt2 is None else self.bt2.spline.vector
        self._rank = self.derham.comm.Get_rank() if self.derham.comm is not None else 0
        self._info = self._nonlin_solver.info and (self.rank == 0)

        # assembly of WMMnew happens in VariationalDensityEvolve
        self._Mrho = self.mass_ops.WMMnew
        self._Mrho_pc = MassMatrixPreconditioner(self._Mrho, dim_reduce=None)

        self._initialize_operators()

        # Kronecker preconditioner of the Schur operator, built for the time step in prepare_time_step
        self._schur_pc = alfven_schur_preconditioner(self, [] if self._model == "linear" else ['model="linear"'])
        self._schur_pc_dt = None

        # temporaries
        u = self.variables.u.spline.vector
        b = self.variables.b.spline.vector
        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_advection2 = u.space.zeros()
        self._tmp_bn1 = b.space.zeros()
        self._tmp_bn12 = b.space.zeros()
        self._tmp_bn_diff = b.space.zeros()
        self._tmp_b_advection = b.space.zeros()
        self._tmp_b_advection2 = b.space.zeros()
        self._linear_form_dl_db = b.space.zeros()

        if self._linearize:
            self._extracted_b2 = self.derham.extraction_ops["2"].dot(self.projected_equil.b2)

    def prepare_time_step(self, dt: float):
        """Kronecker preconditioner of the Schur operator for the time step ``dt`` (with
        ``schur_precond="StiffnessPreconditioner"``): built in the setup, rebuilt when dt changes."""
        if self._schur_pc is not None and dt != self._schur_pc_dt:
            self._inv_Jacobian.precond = self._schur_pc.build(dt)
            self._schur_pc_dt = dt

    @profile
    def __call__(self, dt):
        # preconditioner for the current time step (no-op unless dt changed since the setup)
        self.prepare_time_step(dt)
        # WMMnew is re-assembled in VariationalDensityEvolve: update the diagonal scaling of its preconditioner
        self._Mrho_pc.refresh()

        if self._nonlin_solver.type != "Picard":
            raise ValueError("Only the Picard solver is implemented for VariationalAlfvenEvolve")

        if self._info:
            logger.info("")
            logger.info("Newton iteration in VariationalAlfvenEvolve")

        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        # with implicit weights the operators are updated at the midpoint inside the iteration
        if not self._implicit_weights:
            self._update_weights(bn)

        mn = self._Mrho.dot(un, out=self._tmp_mn)
        bn1 = bn.copy(out=self._tmp_bn1)
        bn1 += self._tmp_bn_diff
        un1 = un.copy(out=self._tmp_un1)
        un1 += self._tmp_un_diff
        mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)
        tol = self._nonlin_solver.tol
        # stopping criterion relative to the state at t^n, with a check for stagnation
        conv = NewtonConvergence("VariationalAlfvenEvolve", tol, self._get_error(mn, bn))

        for it in range(self._nonlin_solver.maxiter):
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5
            self._update_linear_form_dl_db(bn, bn1)
            if self._implicit_weights:
                self._update_weights(self._tmp_bn12)

            advection, b_advection = self._advection(un12)
            advection *= dt
            b_advection *= dt

            bn_diff = bn1.copy(out=self._tmp_bn_diff)
            bn_diff -= bn
            bn_diff += b_advection
            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            stop = conv.stop(self._get_error(mn_diff, bn_diff), it)
            if self._info:
                logger.info(f"iteration : {it} relative error : {conv.err}")
            if stop:
                break

            # Newton step for the normalized residual (the tolerance of the linear solver is relative to it)
            self._get_jacobian(dt)
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = bn_diff
            incr = dot_normalized(self._inv_Jacobian, self._tmp_f, self._tmp_incr)
            if self._info:
                logger.info(f"information on the linear solver : {self._inv_Jacobian._solver._info}")
            un1 -= incr[0]
            bn1 -= incr[1]
            mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)

        conv.warn_if_not_converged(it, self._nonlin_solver.maxiter)

        if not conv.converged:
            # the last iteration applied an update after the residual: midpoint velocity (and weights) of the final
            # iterate, such that the update of the auxiliary variable below is consistent with it
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5
            if self._implicit_weights:
                self._update_linear_form_dl_db(bn, bn1)
                self._update_weights(self._tmp_bn12)

        self._tmp_un_diff = un1 - un
        self._tmp_bn_diff = bn1 - bn
        self.update_feec_variables(b=bn1, u=un1)

        # auxiliary second-order variable, with the same weight
        if self._bt2 is not None:
            b_advection = self.curlPib.dot(un12, out=self._tmp_b_advection)
            b_advection *= dt
            self._bt2 -= b_advection

    def _advection(self, un12):
        """The force of the magnetic field on the velocity (without dt) and the induction term."""
        advection = self._tmp_advection
        advection *= 0.0
        b_advection = self._tmp_b_advection
        b_advection *= 0.0
        if self._model == "linear":
            advection += self.curlPibT0.dot(self._linear_form_dl_db, out=self._tmp_advection2)
            advection += self.curlPibT.dot(self._linear_form_dl_db0, out=self._tmp_advection2)
            b_advection += self.curlPib0.dot(un12, out=self._tmp_b_advection2)
        elif self._model == "perturbative":
            advection += self.curlPibT0.dot(self._linear_form_dl_db, out=self._tmp_advection2)
            advection += self.curlPibT.dot(self._linear_form_dl_db0, out=self._tmp_advection2)
            advection += self.curlPibT.dot(self._linear_form_dl_db, out=self._tmp_advection2)
            b_advection += self.curlPib.dot(un12, out=self._tmp_b_advection2)
            b_advection += self.curlPib0.dot(un12, out=self._tmp_b_advection2)
        else:
            advection += self.curlPibT.dot(self._linear_form_dl_db, out=self._tmp_advection2)
            b_advection += self.curlPib.dot(un12, out=self._tmp_b_advection2)
        return advection, b_advection

    def _initialize_operators(self):
        """Transport operators, the Jacobian of the Newton step and its Schur solver."""
        self.curlPib = Hdiv0_transport_operator(self.derham)
        self.curlPibT = self.curlPib.T

        # inverse mass matrix for the error norm
        self.pc_Mv = preconditioner.MassMatrixPreconditioner(self.mass_ops.Mv, dim_reduce=None)
        self._inv_Mv = inverse(self.mass_ops.Mv, "pcg", pc=self.pc_Mv, tol=1e-10, maxiter=1000, verbose=False)

        if self._model in ("linear", "perturbative"):
            self.curlPib0 = Hdiv0_transport_operator(self.derham)
            self.curlPibT0 = self.curlPib0.T
            self.curlPib0.update_coeffs(self.projected_equil.b2)
            self.curlPibT0.update_coeffs(self.projected_equil.b2)
            self._linear_form_dl_db0 = -self.mass_ops.M2.dot(self.projected_equil.b2)

        if self._model == "linear":
            T, TT = self.curlPib0, self.curlPibT0
        elif self._model == "perturbative":
            T, TT = self.curlPib0 + self.curlPib, self.curlPibT0 + self.curlPibT
        else:
            T, TT = self.curlPib, self.curlPibT
        # dt-dependent scalars set in _get_jacobian
        self._mdt2_TT_M2 = 2 * (TT @ self.mass_ops.M2)
        self._dt2_T = 2 * T

        Jacs = BlockVectorSpace(self.derham.Vvpol, self.derham.V2pol)
        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()
        self._Jacobian = BlockLinearOperator(Jacs, Jacs)
        self._Jacobian[0, 0] = self._Mrho
        self._Jacobian[0, 1] = self._mdt2_TT_M2
        self._Jacobian[1, 0] = self._dt2_T
        self._Jacobian[1, 1] = IdentityOperator(self.derham.V2pol)
        self._inv_Jacobian = SchurSolverFull(
            self._Jacobian,
            self.options.solver,
            # fixed (linear) preconditioner: an inner iterative solve is expensive and not a linear operator
            pc=self._Mrho_pc,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=self._lin_solver.verbose,
            recycle=True,
        )

    def _update_weights(self, b):
        """Weights of the transport operator with the (perturbed) magnetic field."""
        self.curlPib.update_coeffs(b)
        self.curlPibT.update_coeffs(b)

    def _update_linear_form_dl_db(self, bn, bn1):
        """The linear form -M2 b^{n+1/2} (derivative of the magnetic energy)."""
        bn12 = bn.copy(out=self._tmp_bn12)
        bn12 += bn1
        bn12 *= 0.5
        if self._linearize:
            wb = self.mass_ops.M2.dot(bn12 - self._extracted_b2, out=self._linear_form_dl_db)
        else:
            wb = self.mass_ops.M2.dot(bn12, out=self._linear_form_dl_db)
        wb *= -1

    def _get_error(self, mn_diff, bn_diff):
        # the inverse mass matrix is applied to the normalized residual (relative solver tolerance)
        err_u = inv_norm2(self._inv_Mv, self.derham.boundary_ops["v"].dot(mn_diff))
        err_b = self.mass_ops.M2.dot_inner(bn_diff, bn_diff)
        return max(err_b, err_u)

    def _get_jacobian(self, dt):
        self._mdt2_TT_M2._scalar = -dt / 2
        self._dt2_T._scalar = dt / 2
