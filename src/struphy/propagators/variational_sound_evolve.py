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
    NewtonConvergence,
    Pressure_transport_operator,
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


class VariationalSoundEvolve(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the coupling of the velocity and :math:`q = \sqrt p` (sound waves) of the
    variational MHD models in :math:`q`: find :math:`\mathbf u \in (H^1)^3` and :math:`q \in L^2` such that, with the
    transport operator :math:`\mathcal S_{\bar q}(\mathbf v) = \nabla \cdot (\bar q \mathbf v) + \left(\frac \gamma 2 -
    1\right) \bar q\, \nabla \cdot \mathbf v`, depending on the option ``model``:

    * ``"full_q"`` (full MHD):

      .. math::

          \int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x
          - \int_\Omega \frac{2 q}{\gamma -1} \mathcal S_q(\mathbf v)\,\textrm d \mathbf x = 0
          \qquad \forall \, \mathbf v \in (H^1)^3\,,
          \qquad
          \partial_t q + \mathcal S_q(\mathbf u) = 0 \,.

    * ``"linear_q"`` (:math:`q` is the perturbation of the equilibrium :math:`q_0`):

      .. math::

          \int_\Omega \partial_t (\rho_0 \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x
          - \int_\Omega \frac{2}{\gamma -1} \left(q\, \mathcal S_{q_0}(\mathbf v)
          + q_0\, \mathcal S_q(\mathbf v)\right)\textrm d \mathbf x = 0\,,
          \qquad
          \partial_t q + \mathcal S_{q_0}(\mathbf u) = 0 \,.

    * ``"perturbative_q"`` (:math:`q` is the perturbation, all nonlinear terms are kept):

      .. math::

          \int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x
          - \int_\Omega \frac{2}{\gamma -1} \left(q\, \mathcal S_{q_0 + q}(\mathbf v)
          + q_0\, \mathcal S_q(\mathbf v)\right)\textrm d \mathbf x = 0\,,
          \qquad
          \partial_t q + \mathcal S_{q_0 + q}(\mathbf u) = 0 \,.

    The magnetic field is not changed; together with
    :class:`~struphy.propagators.variational_alfven_evolve.VariationalAlfvenEvolve` this splits the coupled step of
    :class:`~struphy.propagators.variational_qb_evolve.VariationalQBEvolve` into the shear-Alfvén/fast and the sound
    waves. The second-order term of the linear and perturbative models comes from the auxiliary variable
    :math:`\hat q` (``qt3``), which, if given, is advanced with :math:`\partial_t \hat q + \mathcal S_q(\mathbf u) = 0`;
    it enters the conserved energy as :math:`\int 2 q_0 \hat q / (\gamma - 1)`. For ``"perturbative_q"``, :math:`\rho`
    is :math:`\rho_0` plus the variable of
    :class:`~struphy.propagators.variational_density_evolve.VariationalDensityEvolve`.

    Discretization: with :math:`\mathbb S[\bar q] = \mathbb D \hat\Pi^2[\hat{\bar q} \vec{\boldsymbol\Lambda}^v]
    + (\gamma/2 - 1) \hat\Pi^3[\hat{\bar q}/\sqrt g\, \vec{\boldsymbol\Lambda}^3] \mathbb D \mathcal U^v`
    (:class:`~struphy.feec.variational_utilities.Pressure_transport_operator`) and, for ``"linear_q"``,

    .. math::

        \begin{align}
        &\mathbb M^v[\hat\rho_0] \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t}
        - \frac{2}{\gamma - 1}\left(\mathbb S[q_0]^\top \mathbb M^3 \mathbf q^{n+\frac12}
        + \mathbb S[\bar{\mathbf q}]^\top \mathbb M^3 \mathbf q_0\right) = 0 \,,
        \\[2mm]
        &\frac{\mathbf q^{n+1} - \mathbf q^n}{\Delta t} + \mathbb S[q_0]\, \mathbf u^{n+\frac12} = 0 \,,
        \qquad
        \frac{\hat{\mathbf q}^{n+1} - \hat{\mathbf q}^n}{\Delta t} + \mathbb S[\bar{\mathbf q}]\, \mathbf u^{n+\frac12} = 0 \,,
        \end{align}

    and correspondingly for the other models. The weight :math:`\bar{\mathbf q}` is taken at the same time in all
    equations, which makes the step energy conserving: at :math:`t^n` (first order in time, one Newton step solves the
    linear system), or at the midpoint :math:`t^{n+1/2}` with the option ``implicit_weights`` (second order in time, a
    Picard iteration). The Newton system is solved with
    :class:`~struphy.linear_algebra.schur_solver.SchurSolverFull`; for ``"linear_q"`` its Schur complement is
    :math:`\mathbb M^v[\hat\rho_0] + \frac{\Delta t^2}{2(\gamma - 1)} \mathbb S[q_0]^\top \mathbb M^3 \mathbb S[q_0]`.
    """

    class Variables:
        """Container for variables advanced by :class:`VariationalSoundEvolve`.

        Attributes
        ----------
        u : FEECVariable
            Velocity variable in ``"H1vec"`` space.
        q : FEECVariable
            Square root of the pressure in ``"L2"`` space.
        """

        def __init__(self):
            self._u: FEECVariable = None
            self._q: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1vec"
            self._u = new

        @property
        def q(self) -> FEECVariable:
            return self._q

        @q.setter
        def q(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._q = new

    def __init__(self, div_u: FEECVariable = None, u2: FEECVariable = None, qt3: FEECVariable = None):
        r"""
        Parameters
        ----------
        div_u : FEECVariable, default=None
            Divergence diagnostic (``"L2"`` space) of the velocity at the midpoint.
        u2 : FEECVariable, default=None
            Velocity as 2-form, :math:`\mathcal U^v \mathbf u` (``"Hdiv"`` space), diagnostic.
        qt3 : FEECVariable, default=None
            Auxiliary second-order variable :math:`\hat q` (``"L2"`` space) of the linear and perturbative models,
            advanced with :math:`\partial_t \hat q + \mathcal S_q(\mathbf u) = 0`. It enters only the energy, not the
            momentum equation.
        """
        self.variables = self.Variables()
        self.div_u = div_u
        self.u2 = u2
        self.qt3 = qt3

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`VariationalSoundEvolve`.

        Parameters
        ----------
        model : {"full_q", "linear_q", "perturbative_q"}, default="full_q"
            Model variant.
        gamma : float, default=5/3
            Adiabatic index.
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
            for ``"linear_q"`` and ``"perturbative_q"`` and ``False`` for ``"full_q"``.
        """

        # specific literals
        OptsModel = Literal["full_q", "linear_q", "perturbative_q"]
        # propagator options
        model: OptsModel = "full_q"
        gamma: float = 5.0 / 3.0
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        implicit_weights: bool | None = None

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
            if self.implicit_weights is None:
                self.implicit_weights = self.model in ("linear_q", "perturbative_q")

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
        self._implicit_weights = self.options.implicit_weights
        self._divu = None if self.div_u is None else self.div_u.spline.vector
        self._u2 = None if self.u2 is None else self.u2.spline.vector
        self._qt3 = None if self.qt3 is None else self.qt3.spline.vector
        self._rank = self.derham.comm.Get_rank() if self.derham.comm is not None else 0
        self._info = self._nonlin_solver.info and (self.rank == 0)

        # assembly of WMMnew happens in VariationalDensityEvolve
        self._Mrho = self.mass_ops.WMMnew
        self._Mrho_pc = MassMatrixPreconditioner(self._Mrho, dim_reduce=None)

        self._initialize_operators()

        # temporaries
        u = self.variables.u.spline.vector
        q = self.variables.q.spline.vector
        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_advection2 = u.space.zeros()
        self._tmp_qn1 = q.space.zeros()
        self._tmp_qn12 = q.space.zeros()
        self._tmp_qn_diff = q.space.zeros()
        self._tmp_q_advection = q.space.zeros()
        self._tmp_q_advection2 = q.space.zeros()
        self._linear_form_dl_dq = q.space.zeros()

        if self._linearize:
            self._extracted_q3 = self.derham.extraction_ops["3"].dot(self.projected_equil.q3)

    @profile
    def __call__(self, dt):
        # WMMnew is re-assembled in VariationalDensityEvolve: update the diagonal scaling of its preconditioner
        self._Mrho_pc.refresh()

        if self._nonlin_solver.type != "Picard":
            raise ValueError("Only the Picard solver is implemented for VariationalSoundEvolve")

        if self._info:
            logger.info("")
            logger.info("Newton iteration in VariationalSoundEvolve")

        un = self.variables.u.spline.vector
        qn = self.variables.q.spline.vector

        # with implicit weights the operators are updated at the midpoint inside the iteration
        if not self._implicit_weights:
            self._update_weights(qn)

        mn = self._Mrho.dot(un, out=self._tmp_mn)
        qn1 = qn.copy(out=self._tmp_qn1)
        qn1 += self._tmp_qn_diff
        un1 = un.copy(out=self._tmp_un1)
        un1 += self._tmp_un_diff
        mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)
        tol = self._nonlin_solver.tol
        # stopping criterion relative to the state at t^n, with a check for stagnation
        conv = NewtonConvergence("VariationalSoundEvolve", tol, self._get_error(mn, qn))

        for it in range(self._nonlin_solver.maxiter):
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5
            self._update_linear_form_dl_dq(qn, qn1)
            if self._implicit_weights:
                self._update_weights(self._tmp_qn12)

            advection, q_advection = self._advection(un12)
            advection *= dt
            q_advection *= dt

            qn_diff = qn1.copy(out=self._tmp_qn_diff)
            qn_diff -= qn
            qn_diff += q_advection
            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            stop = conv.stop(self._get_error(mn_diff, qn_diff), it)
            if self._info:
                logger.info(f"iteration : {it} relative error : {conv.err}")
            if stop:
                break

            # Newton step for the normalized residual (the tolerance of the linear solver is relative to it)
            self._get_jacobian(dt)
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = qn_diff
            incr = dot_normalized(self._inv_Jacobian, self._tmp_f, self._tmp_incr)
            if self._info:
                logger.info(f"information on the linear solver : {self._inv_Jacobian._solver._info}")
            un1 -= incr[0]
            qn1 -= incr[1]
            mn1 = self._Mrho.dot(un1, out=self._tmp_mn1)

        conv.warn_if_not_converged(it, self._nonlin_solver.maxiter)

        if not conv.converged:
            # the last iteration applied an update after the residual: midpoint velocity (and weights) of the final
            # iterate, such that the update of the auxiliary variable below is consistent with it
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5
            if self._implicit_weights:
                self._update_linear_form_dl_dq(qn, qn1)
                self._update_weights(self._tmp_qn12)

        self._tmp_un_diff = un1 - un
        self._tmp_qn_diff = qn1 - qn
        self.update_feec_variables(q=qn1, u=un1)

        # diagnostics
        if self._divu is not None:
            self._transop_q.div.dot(un12, out=self._divu)
        if self._u2 is not None:
            self._transop_q._Uv.dot(un1, out=self._u2)

        # auxiliary second-order variable, with the same weight
        if self._qt3 is not None:
            q_advection = self._transop_q.dot(un12, out=self._tmp_q_advection)
            q_advection *= dt
            self._qt3 -= q_advection

    def _advection(self, un12):
        """The force of q on the velocity (without dt) and the transport of q."""
        advection = self._tmp_advection
        advection *= 0.0
        q_advection = self._tmp_q_advection
        q_advection *= 0.0
        if self._model == "linear_q":
            advection += self._transop_q0T.dot(self._linear_form_dl_dq, out=self._tmp_advection2)
            advection += self._transop_qT.dot(self._linear_form_dl_dq0, out=self._tmp_advection2)
            q_advection += self._transop_q0.dot(un12, out=self._tmp_q_advection2)
        elif self._model == "perturbative_q":
            advection += self._transop_qT.dot(self._linear_form_dl_dq, out=self._tmp_advection2)
            advection += self._transop_q0T.dot(self._linear_form_dl_dq, out=self._tmp_advection2)
            advection += self._transop_qT.dot(self._linear_form_dl_dq0, out=self._tmp_advection2)
            q_advection += self._transop_q0.dot(un12, out=self._tmp_q_advection2)
            q_advection += self._transop_q.dot(un12, out=self._tmp_q_advection2)
        else:
            advection += self._transop_qT.dot(self._linear_form_dl_dq, out=self._tmp_advection2)
            q_advection += self._transop_q.dot(un12, out=self._tmp_q_advection2)
        return advection, q_advection

    def _initialize_operators(self):
        """Transport operators, the Jacobian of the Newton step and its Schur solver."""
        self._transop_q = Pressure_transport_operator(self.derham, self.domain, self.basis_ops.Uv, self._gamma / 2.0)
        self._transop_qT = self._transop_q.T

        # inverse mass matrix for the error norm
        self.pc_Mv = preconditioner.MassMatrixPreconditioner(self.mass_ops.Mv, dim_reduce=None)
        self._inv_Mv = inverse(self.mass_ops.Mv, "pcg", pc=self.pc_Mv, tol=1e-10, maxiter=1000, verbose=False)

        if self._model in ("linear_q", "perturbative_q"):
            self._transop_q0 = Pressure_transport_operator(
                self.derham, self.domain, self.basis_ops.Uv, self._gamma / 2.0
            )
            self._transop_q0T = self._transop_q0.T
            self._transop_q0.update_coeffs(self.projected_equil.q3)
            self._transop_q0T.update_coeffs(self.projected_equil.q3)
            self._linear_form_dl_dq0 = -2 / (self._gamma - 1.0) * self.mass_ops.M3.dot(self.projected_equil.q3)

        if self._model == "linear_q":
            S, ST = self._transop_q0, self._transop_q0T
        elif self._model == "perturbative_q":
            S, ST = self._transop_q0 + self._transop_q, self._transop_q0T + self._transop_qT
        else:
            S, ST = self._transop_q, self._transop_qT
        # dt-dependent scalars set in _get_jacobian
        self._mdt_ST_M3 = 2 * (ST @ self.mass_ops.M3)
        self._dt2_S = 2 * S

        Jacs = BlockVectorSpace(self.derham.Vvpol, self.derham.V3pol)
        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()
        self._Jacobian = BlockLinearOperator(Jacs, Jacs)
        self._Jacobian[0, 0] = self._Mrho
        self._Jacobian[0, 1] = self._mdt_ST_M3
        self._Jacobian[1, 0] = self._dt2_S
        self._Jacobian[1, 1] = IdentityOperator(self.derham.V3pol)
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

    def _update_weights(self, q):
        """Weights of the transport operator with the (perturbed) q."""
        self._transop_q.update_coeffs(q)
        self._transop_qT.update_coeffs(q)

    def _update_linear_form_dl_dq(self, qn, qn1):
        """The linear form -2/(gamma-1) M3 q^{n+1/2} (derivative of the thermal energy)."""
        qn12 = qn.copy(out=self._tmp_qn12)
        qn12 += qn1
        qn12 *= 0.5
        if self._linearize:
            wq = self.mass_ops.M3.dot(qn12 - self._extracted_q3, out=self._linear_form_dl_dq)
        else:
            wq = self.mass_ops.M3.dot(qn12, out=self._linear_form_dl_dq)
        wq *= -2 / (self._gamma - 1)

    def _get_error(self, mn_diff, qn_diff):
        # the inverse mass matrix is applied to the normalized residual (relative solver tolerance)
        err_u = inv_norm2(self._inv_Mv, self.derham.boundary_ops["v"].dot(mn_diff))
        err_q = self.mass_ops.M3.dot_inner(qn_diff, qn_diff)
        return max(err_q, err_u)

    def _get_jacobian(self, dt):
        self._mdt_ST_M3._scalar = -dt / (self._gamma - 1.0)
        self._dt2_S._scalar = dt / 2
