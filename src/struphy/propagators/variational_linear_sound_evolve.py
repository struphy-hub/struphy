import logging
from dataclasses import dataclass
from typing import Literal

from line_profiler import profile

from struphy.feec.mass import L2Projector
from struphy.feec.variational_utilities import Pressure_transport_operator, dot_normalized
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
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
    the auxiliary variable :math:`\hat s` (``pt3``), which, if given, is advanced with
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

    * ``"linear_p"``:

      .. math::

          \begin{align}
          &\mathbb M^v[\hat\rho_0] \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t}
          - \frac{1}{\gamma - 1} \mathbb S[\mathbf p^n]^\top \hat l^3(1) = 0 \,,
          \\[2mm]
          &\frac{\mathbf p^{n+1} - \mathbf p^n}{\Delta t} + \mathbb S[p_0]\, \mathbf u^{n+\frac12} = 0 \,,
          \qquad
          \frac{\hat{\mathbf p}^{n+1} - \hat{\mathbf p}^n}{\Delta t} + \mathbb S[\mathbf p^n]\, \mathbf u^{n+\frac12}
          = 0 \,,
          \end{align}

      with :math:`\hat l^3(1)_{ijk} = \int_{\hat\Omega} \Lambda^3_{ijk} \sqrt g \,\textrm d \boldsymbol\eta`. The
      thermal energy :math:`p/(\gamma - 1)` is linear in :math:`p`, so the pressure enters the momentum equation only
      through the weight :math:`\mathbf p^n`: the velocity is advanced with one mass-matrix solve, then the pressure.
      The sound waves are explicit and, with the weight :math:`\mathbf p^n` at :math:`t^n`, not stable for any
      :math:`\Delta t`: the amplitude of a sound wave of frequency :math:`\omega_s` grows by
      :math:`|\lambda|^2 = 1 + (\Delta t\, \omega_s)^2 / 2` per step, so :math:`\Delta t\, \omega_s` must be small
      over the simulated time. (``"linear_q"`` is stable for :math:`\gamma \leq 2`, with a numerical damping of
      order :math:`\Delta t\, \omega_s^2` for :math:`\gamma < 2`.)

    In both cases the weight at :math:`t^n` is the same in the momentum equation and in the update of :math:`\hat s`,
    which makes the step energy conserving; it is first order in time.

    With ``weights="midpoint"``, the weight is :math:`\mathbf s^{n+\frac12}` instead of :math:`\mathbf s^n`, in the
    momentum equation and in the update of :math:`\hat s`. The map :math:`\mathbb R \mathbf s = \mathbb S[\mathbf s]^\top
    \mathbf l` (:math:`\mathbf l = \mathbb M^3 \mathbf q_0`, respectively :math:`\hat l^3(1)`, with the factors above)
    is linear in the weight and is assembled once
    (:meth:`~struphy.feec.variational_utilities.Pressure_transport_operator.weight_map`), so the step is still one
    linear solve, with the (not symmetric) Schur complement
    :math:`\mathbb M^v[\hat\rho_0] + \frac{\Delta t^2}{2(\gamma - 1)} (\mathbb S[q_0]^\top \mathbb M^3 +
    \mathbb S[\cdot]^\top \mathbb M^3 \mathbf q_0)\, \mathbb S[q_0]`, respectively
    :math:`\mathbb M^v[\hat\rho_0] + \frac{\Delta t^2}{4 (\gamma - 1)} \mathbb S[\cdot]^\top \hat l^3(1)\,
    \mathbb S[p_0]`; the sound waves are then implicit for ``"linear_p"`` too. The step is energy conserving and
    second order in time (implicit midpoint rule), without the numerical damping (``"linear_q"``) or growth
    (``"linear_p"``) of the frozen weight.

    The velocity can also be a 2-form, :math:`\mathbf u \in H(\textnormal{div})` (velocity variable in ``"Hdiv"``
    space). The scheme is the same, with :math:`\mathbb M^v[\hat\rho_0] \to \mathbb M^{2,n}` and
    :math:`\mathbb S[\bar s] = \mathbb D \hat\Pi^2[\hat{\bar s} \vec{\boldsymbol\Lambda}^2/\sqrt g]
    + (\kappa - 1) \hat\Pi^3[\hat{\bar s}/\sqrt g\, \vec{\boldsymbol\Lambda}^3] \mathbb D` (no :math:`\mathcal U^v`);
    for ``"linear_p"`` the equilibrium operator is :math:`\mathbb D \mathcal S^2 + (\gamma - 1) \mathcal K^3 \mathbb D`
    of :class:`~struphy.propagators.magnetosonic.Magnetosonic`. The energy is conserved in the same way.
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

    def __init__(self, div_u: FEECVariable = None, u2: FEECVariable = None, pt3: FEECVariable = None):
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
        """
        self.variables = self.Variables()
        self.div_u = div_u
        self.u2 = u2
        self.pt3 = pt3

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`VariationalLinearSoundEvolve`.

        Parameters
        ----------
        model : {"linear_q", "linear_p"}, default="linear_q"
            Thermodynamic variable: :math:`q = \sqrt p` or :math:`p`.
        gamma : float, default=5/3
            Adiabatic index.
        weights : {"frozen", "midpoint"}, default="midpoint"
            Weight of the second-order term: :math:`\mathbf s^n` (first order in time) or :math:`\mathbf s^{n+1/2}`
            (second order, with the assembled map :math:`\mathbb R`).
        solver : {"pcg", "pcg_sr", "cg", "pbicgstab"}, default=None
            Linear solver of the Schur system (``"linear_q"``, or ``"linear_p"`` with midpoint weights) or of the
            mass matrix (``"linear_p"``, frozen weight); None gives ``"pcg"`` for ``weights="frozen"`` and
            ``"pbicgstab"`` for ``weights="midpoint"``, whose Schur complement is not symmetric (only
            ``"pbicgstab"``).
        precond : {"MassMatrixPreconditioner", None}, default="MassMatrixPreconditioner"
            Preconditioner of the linear solve (of the velocity block).
        solver_params : SolverParameters, default=None
            Linear-solver controls. The system is solved for a normalized right-hand side, so ``tol`` is relative to
            its norm.
        info : bool, default=False
            Log the information of the linear solver in every step.
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
        OptsSolver = Literal["pcg", "pcg_sr", "cg", "pbicgstab"]
        # propagator options
        model: OptsModel = "linear_q"
        gamma: float = 5.0 / 3.0
        weights: OptsWeights = "midpoint"
        solver: OptsSolver = None
        precond: OptsPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        info: bool = False
        relative_tol: bool = True

        def __post_init__(self):
            # checks
            check_option(self.model, self.OptsModel)
            check_option(self.weights, self.OptsWeights)
            if self.solver is None:
                self.solver = "pcg" if self.weights == "frozen" else "pbicgstab"
            check_option(self.solver, self.OptsSolver)
            assert self.weights == "frozen" or self.solver == "pbicgstab", (
                f'weights="midpoint" needs solver="pbicgstab" (not symmetric), got {self.solver!r}.'
            )
            check_option(self.precond, self.OptsPrecond)
            assert self.gamma > 1.0, f"gamma must be larger than 1, got {self.gamma}."

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

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
        if self._u_space == "Hdiv" and self._model == "linear_p":
            # equilibrium term as in Magnetosonic
            div = self.derham.div
            self._S0 = div @ self.basis_ops.S2 + (kappa - 1.0) * (self.basis_ops.K3 @ div)
        else:
            self._S0 = Pressure_transport_operator(
                self.derham, self.domain, self.basis_ops.Uv, kappa, u_space=self._u_space
            )
            self._S0.update_coeffs(s0)
        self._Ss = Pressure_transport_operator(
            self.derham, self.domain, self.basis_ops.Uv, kappa, u_space=self._u_space
        )
        self._SsT = self._Ss.T

        u = self.variables.u.spline.vector
        s = self.variables.s.spline.vector
        if self._model == "linear_q":
            # linear form of the equilibrium, -2/(gamma-1) M3 q0
            self._lf0 = -2.0 / (self._gamma - 1.0) * self.mass_ops.M3.dot(s0)
        else:
            # linear form of the thermal energy, -1/(gamma-1) l3(1)
            self._lf0 = L2Projector("L2", self.mass_ops).get_dofs(self.domain.jacobian_det)
            self._lf0 *= -1.0 / (self._gamma - 1.0)

        # midpoint weights: the second-order term R s = S[s]^T lf0, assembled once
        self._midpoint = self.options.weights == "midpoint"
        self._R = self._SsT.weight_map(self._lf0) if self._midpoint else None

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
                R=self._R,
            )
        elif self._midpoint:
            # the pressure enters the momentum equation only through the second-order term
            self._allocate_schur_solver(
                self._S0,
                None,
                1.0,
                self.derham.V3pol,
                self.options.solver,
                self.options.solver_params,
                R=self._R,
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

    @profile
    def __call__(self, dt):
        # WMMnew is assembled in VariationalDensityEvolve: update the diagonal scaling of its preconditioner
        self._refresh_mass_pc()

        un = self.variables.u.spline.vector
        sn = self.variables.s.spline.vector

        if not self._midpoint:
            # weight of the second-order term at t^n
            self._Ss.update_coeffs(sn)
            self._SsT.update_coeffs(sn)

        if self._model == "linear_q" or self._midpoint:
            # residual at (u^n, s^n): the force -2/(gamma-1) (S0^T M3 q^n + S[q^n]^T M3 q0), respectively
            # -1/(gamma-1) S[p^n]^T l3(1), and S0 u^n (times dt)
            if self._model == "linear_q":
                M3s = self.mass_ops.M3.dot(sn, out=self._tmp_M3s)
                M3s *= -2.0 / (self._gamma - 1.0)
                f_u = self._S0T.dot(M3s, out=self._tmp_fu)
                if self._midpoint:
                    f_u += self._R.dot(sn, out=self._tmp_fu2)
                else:
                    f_u += self._SsT.dot(self._lf0, out=self._tmp_fu2)
            else:
                f_u = self._R.dot(sn, out=self._tmp_fu)
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
            f_u = self._SsT.dot(self._lf0, out=self._tmp_fu)
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
            # pressure with the midpoint velocity
            s_advection = self._S0.dot(un12, out=self._tmp_fs)
            s_advection *= dt
            sn1 = sn.copy(out=self._tmp_sn1)
            sn1 -= s_advection

        # diagnostics
        if self._divu is not None:
            self._Ss.div.dot(un12, out=self._divu)
        if self._u2 is not None:
            self._Ss._Uv.dot(un1, out=self._u2)

        # auxiliary second-order variable, with the same weight as the momentum equation (s^n or s^{n+1/2})
        if self._pt3 is not None:
            if self._midpoint:
                sn12 = sn.copy(out=self._tmp_sn12)
                sn12 += sn1
                sn12 *= 0.5
                self._Ss.update_coeffs(sn12)
            s_advection = self._Ss.dot(un12, out=self._tmp_fs)
            s_advection *= dt
            self._pt3 -= s_advection

        self.update_feec_variables(u=un1, s=sn1)
