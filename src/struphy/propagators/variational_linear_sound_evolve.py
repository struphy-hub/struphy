import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from line_profiler import profile

from struphy.feec.mass import L2Projector
from struphy.feec.variational_utilities import (
    Pressure_transport_operator,
    PressureTransportAdjointMatrix,
    dot_normalized,
)
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.variational_linear_base import VariationalLinearEvolve
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VariationalLinearSoundEvolve(VariationalLinearEvolve):
    r""":ref:`FEEC <gempic>` discretization of the coupling of the velocity and the thermodynamic variable (sound
    waves) of the linear variational MHD models, with one linear solve per time step: find
    :math:`\mathbf u \in (H^1)^3` and :math:`s \in L^2` (the perturbation of :math:`q = \sqrt p` or of :math:`p`)
    such that, with the transport operator :math:`\mathcal S^\kappa_{\bar s}(\mathbf v) = \nabla \cdot (\bar s
    \mathbf v) + (\kappa - 1) \bar s\, \nabla \cdot \mathbf v`, depending on the option ``model``:

    * ``"linear_q"`` (:math:`s = q`, :math:`\kappa = \gamma/2`):

      .. math::

          \int_\Omega \partial_t (\rho_0 \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x
          - \int_\Omega \frac{2}{\gamma -1} \left(q\, \mathcal S^\kappa_{q_0}(\mathbf v)
          + q_0\, \mathcal S^\kappa_q(\mathbf v)\right)\textrm d \mathbf x = 0
          \qquad \forall \, \mathbf v \in (H^1)^3\,,
          \qquad
          \partial_t q + \mathcal S^\kappa_{q_0}(\mathbf u) = 0 \,.

    * ``"linear_p"`` (:math:`s = p`, :math:`\kappa = \gamma`):

      .. math::

          \int_\Omega \partial_t (\rho_0 \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x
          - \int_\Omega \frac{1}{\gamma -1} \mathcal S^\kappa_p(\mathbf v)\,\textrm d \mathbf x = 0
          \qquad \forall \, \mathbf v \in (H^1)^3\,,
          \qquad
          \partial_t p + \mathcal S^\kappa_{p_0}(\mathbf u) = 0 \,.

    The magnetic field is not changed; together with
    :class:`~struphy.propagators.variational_linear_alfven_evolve.VariationalLinearAlfvenEvolve` this splits the
    linear MHD step into the shear-Alfvén/fast and the sound waves. The second-order term of the energy comes from
    the second-order variable :math:`\hat s` (``pt3``), which, if given, is advanced with
    :math:`\partial_t \hat s + \mathcal S^\kappa_s(\mathbf u) = 0`; it enters the conserved energy as
    :math:`\int 2 q_0 \hat q / (\gamma - 1)`, respectively :math:`\int \hat p / (\gamma - 1)`.

    Discretization: with :math:`\mathbb S[\bar s] = \mathbb D \hat\Pi^2[\hat{\bar s} \vec{\boldsymbol\Lambda}^v]
    + (\kappa - 1) \hat\Pi^3[\hat{\bar s}/\sqrt g\, \vec{\boldsymbol\Lambda}^3] \mathbb D \mathcal U^v`
    (:class:`~struphy.feec.variational_utilities.Pressure_transport_operator`),

    * ``"linear_q"``:

      .. math::

          \begin{align}
          &\mathbb M^v[\hat\rho_0] \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t}
          - \frac{2}{\gamma - 1}\left(\mathbb S[q_0]^\top \mathbb M^3 \mathbf q^{n+\frac12}
          + \mathbb S[\mathbf q^n]^\top \mathbb M^3 \mathbf q_0\right) = 0 \,,
          \\[2mm]
          &\frac{\mathbf q^{n+1} - \mathbf q^n}{\Delta t} + \mathbb S[q_0]\, \mathbf u^{n+\frac12} = 0 \,,
          \qquad
          \frac{\hat{\mathbf q}^{n+1} - \hat{\mathbf q}^n}{\Delta t} + \mathbb S[\mathbf q^n]\, \mathbf u^{n+\frac12}
          = 0 \,,
          \end{align}

      solved with one linear solve
      (:class:`~struphy.propagators.variational_linear_base.VariationalLinearEvolve`); the Schur complement
      :math:`\mathbb M^v[\hat\rho_0] + \frac{\Delta t^2}{2(\gamma - 1)} \mathbb S[q_0]^\top \mathbb M^3 \mathbb S[q_0]`
      is constant in time and the sound waves are implicit.

    * ``"linear_p"``: only the integral of :math:`\hat p` enters the energy, :math:`\int \hat p / (\gamma - 1) =
      -\hat{\mathbf l}^\top \hat{\mathbf p}` with :math:`\hat{\mathbf l} = -\hat l^3(\sqrt g) / (\gamma - 1)`,
      :math:`\hat l^3(\sqrt g)_{ijk} = \int_{\hat\Omega} \Lambda^3_{ijk} \sqrt g\, \textrm d \boldsymbol\eta`. The force is
      the variation of this term, the adjoint of the transport of :math:`\hat p` (as in the perturbative model,
      :class:`~struphy.propagators.variational_pb_evolve.VariationalPBEvolve`):

      .. math::

          \begin{align}
          &\mathbb M^v_{\rho_0} \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t}
          + \mathbb S[\mathbf p^w]^\top \hat{\mathbf l} = 0 \,,
          \qquad
          \frac{\mathbf p^{n+1} - \mathbf p^n}{\Delta t} + \mathbb S[p_0]\, \mathbf u^{n+\frac12} = 0 \,,
          \\[2mm]
          &\frac{\hat{\mathbf p}^{n+1} - \hat{\mathbf p}^n}{\Delta t} + \mathbb S[\mathbf p^w]\, \mathbf u^{n+\frac12}
          = 0 \,,
          \end{align}

      which conserves the energy exactly for any weight :math:`\mathbf p^w`. The option ``weights`` sets it. With
      ``"midpoint"`` (default), :math:`\mathbf p^w = \mathbf p^{n+\frac12}`, the implicit midpoint rule, second order in
      time with implicit sound waves. The force :math:`\mathbb S[\mathbf p^w]^\top \hat{\mathbf l} = \mathbb A
      \mathbf p^w` is linear in :math:`\mathbf p^w`, with the matrix :math:`\mathbb A` assembled once
      (:class:`~struphy.feec.variational_utilities.PressureTransportAdjointMatrix`), so the step is one linear solve
      (:class:`~struphy.propagators.variational_linear_base.VariationalLinearEvolve`, Schur complement
      :math:`\mathbb M^v_{\rho_0} - \frac{\Delta t^2}{4} \mathbb A\, \mathbb S[p_0]`, not symmetric). With ``"frozen"``,
      :math:`\mathbf p^w = \mathbf p^n`: the velocity is advanced with one mass-matrix solve, then the pressure. The
      sound waves are then explicit and not stable for any :math:`\Delta t`: the amplitude of a sound wave of
      frequency :math:`\omega_s` grows by :math:`|\lambda|^2 = 1 + (\Delta t\, \omega_s)^2 / 2` per step (first order
      in time).

    **Total pressure as transport weight** (constructor argument ``total_weight=True``, ``"linear_p"``; used by the
    perturbative model :class:`~struphy.models.VariationalPerturbativeMHD`): the pressure
    perturbation is transported with the total pressure :math:`p_0 + \tilde p` as weight,

    .. math::

        \frac{\mathbf p^{n+1} - \mathbf p^n}{\Delta t} + \left(\mathbb S[p_0] + \mathbb S[\mathbf p^{n+\frac12}]\right)
        \mathbf u^{n+\frac12} = 0 \,,

    with the momentum equation and the update of :math:`\hat{\mathbf p}` as above. The term
    :math:`\mathbb S[\mathbf p^{n+\frac12}]\, \mathbf u^{n+\frac12}`, quadratic in the perturbations, is then in the
    same implicit step as the force :math:`-\nabla \tilde p` that it drives: the sound waves of the total pressure are
    implicit. If it were in a separate substep (without force), the splitting of this coupling would be explicit
    and unstable for the grid-scale sound waves (:math:`\omega_s \Delta t \gtrsim 2`) as soon as
    :math:`\tilde p` is a finite fraction of :math:`p_0`. The energy is conserved as above (the weight of the
    transport of :math:`\mathbf p` does not enter it). The step is solved by a Picard iteration on the weight
    :math:`\mathbf w = \mathbf p^{n+\frac12}` (``Options.nonlin_solver``), with one linear solve per iteration (the
    Schur complement with :math:`\mathbb S[p_0] + \mathbb S[\mathbf w]`); the step depends only on the total pressure
    :math:`p_0 + \tilde p`, not on how it is split into equilibrium and perturbation. With ``weights="frozen"`` the
    transport weight is :math:`p_0 + \mathbf p^n` (explicit, first order, as the force).

    For ``"linear_q"`` the weight :math:`\mathbf q^n` at :math:`t^n` is the same in the momentum equation and in the
    update of :math:`\hat{\mathbf q}`, which makes the step energy conserving; it is first order in time, stable for
    :math:`\gamma \leq 2`, with a numerical damping of order :math:`\Delta t\, \omega_s^2` for :math:`\gamma < 2`
    (only ``weights="frozen"``).

    The velocity can also be a 2-form, :math:`\mathbf u \in H(\textnormal{div})` (velocity variable in ``"Hdiv"``
    space). The scheme is the same, with :math:`\mathbb M^v_{\rho_0} \to \mathbb M^2_{\rho_0}`,
    :math:`\mathcal U^v \to \mathbb I` and
    :math:`\mathbb S[\bar s] = \mathbb D \hat\Pi^2[\hat{\bar s} \vec{\boldsymbol\Lambda}^2/\sqrt g]
    + (\kappa - 1) \hat\Pi^3[\hat{\bar s}/\sqrt g\, \vec{\boldsymbol\Lambda}^3] \mathbb D`. The energy is conserved in
    the same way.
    """

    class Variables:
        """Container for variables advanced by :class:`VariationalLinearSoundEvolve`.

        Attributes
        ----------
        u : FEECVariable
            Velocity variable in ``"H1vec"`` or ``"Hdiv"`` space.
        s : FEECVariable
            Perturbation of :math:`q = \\sqrt p` (``"linear_q"``) or of the pressure (``"linear_p"``) in ``"L2"``
            space.
        """

        def __init__(self):
            self._u: FEECVariable = None
            self._s: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("H1vec", "Hdiv")
            self._u = new

        @property
        def s(self) -> FEECVariable:
            return self._s

        @s.setter
        def s(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._s = new

    def __init__(
        self,
        div_u: FEECVariable = None,
        u2: FEECVariable = None,
        pt3: FEECVariable = None,
        rho: FEECVariable = None,
        total_weight: bool = False,
    ):
        r"""
        Parameters
        ----------
        div_u : FEECVariable, default=None
            Diagnostic :math:`\nabla \cdot \mathbf u^{n+1/2}` (``"L2"`` space).
        u2 : FEECVariable, default=None
            Diagnostic velocity 2-form (``"Hdiv"`` space).
        pt3 : FEECVariable, default=None
            Auxiliary second-order variable :math:`\hat s` (``"L2"`` space), advanced with
            :math:`\partial_t \hat s + \mathcal S^\kappa_s(\mathbf u) = 0`. It enters only the energy.
        rho : FEECVariable, default=None
            Density perturbation :math:`\tilde\rho` (``"L2"`` space, not updated) of the perturbative model: with a
            velocity in ``"Hdiv"`` the velocity mass matrix is then :math:`\mathbb M^2_\rho` of the total density
            :math:`\rho_0 + \tilde\rho` (re-assembled in every step) instead of :math:`\mathbb M^2_{\rho_0}`.
        total_weight : bool, default=False
            Transport the pressure perturbation with the total pressure :math:`p_0 + \tilde p` as weight (perturbative
            model, ``"linear_p"``), instead of :math:`p_0` (linear model).
        """
        assert rho is None or (isinstance(rho, FEECVariable) and rho.space == "L2")
        self.total_weight = total_weight
        self.variables = self.Variables()
        self.div_u = div_u
        self.u2 = u2
        self.pt3 = pt3
        self._rho_field = rho

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`VariationalLinearSoundEvolve`.

        Parameters
        ----------
        model : {"linear_q", "linear_p"}, default="linear_q"
            Thermodynamic variable: :math:`q = \sqrt p` or :math:`p`.
        gamma : float, default=5/3
            Adiabatic index.
        weights : {"frozen", "midpoint"}, default=None
            Weight of the second-order term: :math:`\mathbf s^n` (first order in time) or :math:`\mathbf s^{n+1/2}`
            (second order, only for ``"linear_p"``). None gives ``"midpoint"`` for ``"linear_p"`` and ``"frozen"`` for
            ``"linear_q"``.
        solver : {"pcg", "pcg_sr", "cg", "pbicgstab", "rpbicgstab"}, default=None
            Linear solver of the Schur system (``"linear_q"``, or ``"linear_p"`` with midpoint weights) or of the
            mass matrix (``"linear_p"``, frozen weight, not ``"rpbicgstab"``); None gives ``"pcg"`` for
            ``weights="frozen"`` and ``"rpbicgstab"`` (BiCGStab with right preconditioning, see
            :class:`~struphy.linear_algebra.schur_solver.SchurSolverFull`) for ``weights="midpoint"``, whose Schur
            complement is not symmetric (only ``"rpbicgstab"`` or ``"pbicgstab"``).
        precond : {"MassMatrixPreconditioner", None}, default="MassMatrixPreconditioner"
            Preconditioner of the linear solve (of the velocity block).
        solver_params : SolverParameters, default=None
            Linear-solver controls. The system is solved for a normalized right-hand side, so ``tol`` is relative to
            its norm.
        info : bool, default=False
            Log the information of the linear solver in every step.
        nonlin_solver : NonlinearSolverParameters, default=None
            Controls of the Picard iteration on the weight with ``total_weight=True``: it stops when the change of the
            weight is below ``tol`` times the change of the pressure in the step (:math:`L^2` norms); at most
            ``maxiter`` iterations. None gives ``tol=1e-10``, ``maxiter=100``.
        relative_tol : bool, default=True
            Whether the tolerance of ``solver_params`` is relative to the norm of the right-hand side (the residual at
            :math:`t^n`, normalized before the solve), or absolute, as in the non-variational propagators (e.g.
            :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator`). The energy is conserved up
            to the residual of the linear solve.
        """

        # specific literals
        OptsModel = Literal["linear_q", "linear_p"]
        OptsPrecond = Literal["MassMatrixPreconditioner", None]
        OptsWeights = Literal["frozen", "midpoint"]
        OptsSolver = Literal["pcg", "pcg_sr", "cg", "pbicgstab", "rpbicgstab"]
        # propagator options
        model: OptsModel = "linear_q"
        gamma: float = 5.0 / 3.0
        weights: OptsWeights = None
        solver: OptsSolver = None
        precond: OptsPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        info: bool = False
        nonlin_solver: NonlinearSolverParameters = None
        relative_tol: bool = True

        def __post_init__(self):
            # checks
            check_option(self.model, self.OptsModel)
            if self.weights is None:
                self.weights = "midpoint" if self.model == "linear_p" else "frozen"
            check_option(self.weights, self.OptsWeights)
            assert self.model == "linear_p" or self.weights == "frozen", (
                'weights="midpoint" is only available for model="linear_p".'
            )
            if self.solver is None:
                self.solver = "pcg" if self.weights == "frozen" else "rpbicgstab"
            check_option(self.solver, self.OptsSolver)
            assert self.weights == "frozen" or self.solver in ("pbicgstab", "rpbicgstab"), (
                f'weights="midpoint" needs solver="rpbicgstab" or "pbicgstab" (not symmetric), got {self.solver!r}.'
            )
            assert not (self.model == "linear_p" and self.weights == "frozen" and self.solver == "rpbicgstab"), (
                '"rpbicgstab" is a solver of the Schur system, not of the mass matrix ("linear_p", frozen weight).'
            )
            check_option(self.precond, self.OptsPrecond)
            assert self.gamma > 1.0, f"gamma must be larger than 1, got {self.gamma}."

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()
            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters(tol=1e-10, maxiter=100)

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
        self._divu = None if self.div_u is None else self.div_u.spline.vector
        self._u2 = None if self.u2 is None else self.u2.spline.vector
        self._pt3 = None if self.pt3 is None else self.pt3.spline.vector
        self._info = self.options.info and (self.rank == 0)

        # assembly of WMMnew happens in VariationalDensityEvolve
        self._allocate_mass(self.options.precond)

        # transport operators with the equilibrium (constant) and with the perturbation (weight s^n)
        if self._model == "linear_q":
            kappa = self._gamma / 2.0
            s0 = self.projected_equil.q3
        else:
            kappa = self._gamma
            s0 = self.projected_equil.p3
        # equilibrium transport (the same operator as in the perturbative model, VariationalPBEvolve)
        self._S0 = Pressure_transport_operator(
            self.derham, self.domain, self.basis_ops.Uv, kappa, u_space=self._u_space
        )
        self._S0.update_coeffs(s0)

        u = self.variables.u.spline.vector
        s = self.variables.s.spline.vector
        # transport operator with the perturbation as weight
        self._Ss = Pressure_transport_operator(
            self.derham, self.domain, self.basis_ops.Uv, kappa, u_space=self._u_space
        )
        self._SsT = self._Ss.T
        self._div_u = self._Ss.div
        if self._model == "linear_q":
            # linear form of the equilibrium, -2/(gamma-1) M3 q0
            self._lf0 = -2.0 / (self._gamma - 1.0) * self.mass_ops.M3.dot(s0)
        else:
            # linear form of the energy int p-hat / (gamma - 1) of the second-order pressure (with the sign of the force
            # term), as in VariationalPBEvolve: the force is the adjoint S[p^w]^T of the transport of p-hat
            self._lf0 = L2Projector("L2", self.mass_ops).get_dofs(self.domain.jacobian_det)
            self._lf0 *= -1.0 / (self._gamma - 1.0)
            # the matrix A of p -> S[p]^T l: the force is linear in the weight, A p^w (assembled once, exact)
            self._A = PressureTransportAdjointMatrix(
                self.derham, self.domain, self.mass_ops, self.basis_ops, self._u_space, kappa, self._lf0
            )

        self._midpoint = self.options.weights == "midpoint"
        if self.total_weight:
            assert self._model == "linear_p", 'total_weight=True needs model="linear_p".'
            # transport with the perturbation as weight, added to the equilibrium transport in the Schur system
            self._Sw = Pressure_transport_operator(
                self.derham, self.domain, self.basis_ops.Uv, kappa, u_space=self._u_space
            )
            self._S_total = self._S0 + self._Sw

        if self._model == "linear_q":
            S0T = self._S0.T
            S0T.update_coeffs(s0)
            self._S0T = S0T
            self._allocate_schur_solver(
                self._S0,
                S0T @ self.mass_ops.M3,
                -2.0 / (self._gamma - 1.0),
                self.derham.V3pol,
                self.options.solver,
                self.options.solver_params,
            )
        elif self._midpoint:
            # the pressure enters the momentum equation only through the second-order term
            self._allocate_schur_solver(
                self._S_total if self.total_weight else self._S0,
                None,
                1.0,
                self.derham.V3pol,
                self.options.solver,
                self.options.solver_params,
                R=self._A,
            )
        else:
            self._allocate_mass_solver(self.options.solver, self.options.solver_params)

        # temporaries
        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_fu = u.space.zeros()
        self._tmp_fu2 = u.space.zeros()
        self._tmp_sn1 = s.space.zeros()
        self._tmp_sn12 = s.space.zeros()
        self._tmp_fs = s.space.zeros()
        self._tmp_M3s = s.space.zeros()
        self._tmp_du = u.space.zeros()
        if self.total_weight:
            self._tmp_w, self._tmp_dw, self._tmp_M3dw, self._tmp_fs2 = (s.space.zeros() for _ in range(4))

    @profile
    def __call__(self, dt):
        # WMMnew is assembled in VariationalDensityEvolve: update the diagonal scaling of its preconditioner
        self._refresh_mass_pc()

        un = self.variables.u.spline.vector
        sn = self.variables.s.spline.vector

        if self._model == "linear_q":
            # weight of the second-order term at t^n; residual at (u^n, q^n): the force
            # -2/(gamma-1) (S0^T M3 q^n + S[q^n]^T M3 q0) and S0 u^n (times dt)
            self._Ss.update_coeffs(sn)
            self._SsT.update_coeffs(sn)
            M3s = self.mass_ops.M3.dot(sn, out=self._tmp_M3s)
            M3s *= -2.0 / (self._gamma - 1.0)
            f_u = self._S0T.dot(M3s, out=self._tmp_fu)
            f_u += self._SsT.dot(self._lf0, out=self._tmp_fu2)
            f_u *= dt
            f_s = self._S0.dot(un, out=self._tmp_fs)
            f_s *= dt
            incr = self._solve_increment(dt, f_u, f_s)
            un1 = un.copy(out=self._tmp_un1)
            un1 -= incr[0]
            sn1 = sn.copy(out=self._tmp_sn1)
            sn1 -= incr[1]
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5
        elif self.total_weight and self._midpoint:
            un1, sn1, un12 = self._solve_total_weight(dt, un, sn)
        elif self._midpoint:
            # residual at (u^n, p^n): the force A p^n = S[p^n]^T l and S0 u^n (times dt); with A in the Schur system the
            # midpoint system is solved exactly by one linear solve
            f_u = self._A.dot(sn, out=self._tmp_fu)
            f_u *= dt
            f_s = self._S0.dot(un, out=self._tmp_fs)
            f_s *= dt
            incr = self._solve_increment(dt, f_u, f_s)
            un1 = un.copy(out=self._tmp_un1)
            un1 -= incr[0]
            sn1 = sn.copy(out=self._tmp_sn1)
            sn1 -= incr[1]
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5
        else:
            # velocity: M (u^{n+1} - u^n) = -dt S[p^n]^T l, with the pressure at t^n only
            f_u = self._A.dot(sn, out=self._tmp_fu)
            f_u *= dt
            # normalized right-hand side: the tolerance of the linear solver is relative to it
            if self.options.relative_tol:
                incr = dot_normalized(self._inv_Mrho, f_u, self._tmp_fu2)
            else:
                incr = self._inv_Mrho.dot(f_u, out=self._tmp_fu2)
            if self._info:
                logger.info(f"{self.__class__.__name__}, linear solver: {self._inv_Mrho._info}")
            un1 = un.copy(out=self._tmp_un1)
            un1 -= incr
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5
            # pressure with the midpoint velocity (and the weight p0 + p^n with total_weight)
            if self.total_weight:
                self._Sw.update_coeffs(sn)
            s_advection = (self._S_total if self.total_weight else self._S0).dot(un12, out=self._tmp_fs)
            s_advection *= dt
            sn1 = sn.copy(out=self._tmp_sn1)
            sn1 -= s_advection

        # diagnostics
        if self._divu is not None:
            self._div_u.dot(un12, out=self._divu)
        if self._u2 is not None:
            if self._u_space == "Hdiv":
                un1.copy(out=self._u2)
            else:
                self.basis_ops.Uv.dot(un1, out=self._u2)

        # second-order variable, with the same weight as the momentum equation: s-hat += -dt S[s^w] u^{n+1/2}
        if self._pt3 is not None:
            if self._midpoint:
                sw = sn.copy(out=self._tmp_sn12)
                sw += sn1
                sw *= 0.5
            else:
                sw = sn
            self._Ss.update_coeffs(sw)
            s_advection = self._Ss.dot(un12, out=self._tmp_fs)
            s_advection *= dt
            self._pt3 -= s_advection
            self._pt3.update_ghost_regions()

        self.update_feec_variables(u=un1, s=sn1)

    def _solve_total_weight(self, dt, un, sn):
        """The midpoint step with the transport weight p0 + p^{n+1/2}: Picard iteration on the weight w, one linear
        solve per iteration (the system of the midpoint step with S0 + S[w] instead of S0)."""
        M3 = self.mass_ops.M3
        tol = self.options.nonlin_solver.tol
        maxiter = self.options.nonlin_solver.maxiter
        w = sn.copy(out=self._tmp_w)
        err = np.inf
        for it in range(maxiter):
            self._Sw.update_coeffs(w)
            # residual at (u^n, p^n): the force A p^n and (S0 + S[w]) u^n (times dt)
            f_u = self._A.dot(sn, out=self._tmp_fu)
            f_u *= dt
            f_s = self._S_total.dot(un, out=self._tmp_fs)
            f_s *= dt
            incr = self._solve_increment(dt, f_u, f_s)
            un1 = un.copy(out=self._tmp_un1)
            un1 -= incr[0]
            sn1 = sn.copy(out=self._tmp_sn1)
            sn1 -= incr[1]
            # new weight p^{n+1/2} and its change, relative to the change of p in the step
            dw = w.copy(out=self._tmp_dw)
            w = sn.copy(out=self._tmp_w)
            w += sn1
            w *= 0.5
            dw -= w
            dp = sn1.copy(out=self._tmp_fs2)
            dp -= sn
            step = dp.inner(M3.dot(dp, out=self._tmp_M3dw))
            change = dw.inner(M3.dot(dw, out=self._tmp_M3dw))
            err = np.sqrt(change / step) if step > 0.0 else 0.0
            if err < tol:
                break
        else:
            if self.rank == 0:
                logger.warning(
                    f"{self.__class__.__name__}: Picard iteration on the weight not converged in {maxiter} "
                    f"iterations (relative change {err:.2e})."
                )
        if self._info:
            logger.info(f"{self.__class__.__name__}: {it + 1} Picard iterations, relative change {err:.2e}")
        un12 = un.copy(out=self._tmp_un12)
        un12 += un1
        un12 *= 0.5
        return un1, sn1, un12
