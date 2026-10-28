import logging
from dataclasses import dataclass
from typing import Literal

from line_profiler import profile

from struphy.feec.variational_utilities import Hdiv0_transport_operator, alfven_schur_preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.variational_linear_base import VariationalLinearEvolve
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VariationalLinearAlfvenEvolve(VariationalLinearEvolve):
    r""":ref:`FEEC <gempic>` discretization of the coupling of the velocity and the magnetic field (shear-Alfvén and
    fast magnetosonic waves) of the linear variational MHD models, with one linear solve per time step: find
    :math:`\mathbf u \in (H^1)^3` and :math:`\mathbf B \in H(\textnormal{div})` (the perturbation of the equilibrium
    :math:`\mathbf B_0`) such that, with the transport operator
    :math:`\mathcal T_{\bar{\mathbf B}}(\mathbf v) = \nabla \times (\bar{\mathbf B} \times \mathbf v)`,

    .. math::

        \int_\Omega \partial_t (\rho_0 \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x
        - \int_\Omega \left(\mathbf B \cdot \mathcal T_{\mathbf B_0}(\mathbf v)
        + \mathbf B_0 \cdot \mathcal T_{\mathbf B}(\mathbf v)\right)\textrm d \mathbf x = 0
        \qquad \forall \, \mathbf v \in (H^1)^3\,,
        \qquad
        \partial_t \mathbf B + \mathcal T_{\mathbf B_0}(\mathbf u) = 0 \,.

    The thermodynamic variable is not changed, so this propagator serves the linear models in :math:`p` and in
    :math:`q = \sqrt p` alike; together with
    :class:`~struphy.propagators.variational_linear_sound_evolve.VariationalLinearSoundEvolve` it splits the linear
    MHD step into the shear-Alfvén/fast and the sound waves. The second-order term comes from the auxiliary variable
    :math:`\hat{\mathbf B}` (``bt2``), which, if given, is advanced with
    :math:`\partial_t \hat{\mathbf B} + \mathcal T_{\mathbf B}(\mathbf u) = 0`; it enters the conserved energy as
    :math:`\int \mathbf B_0 \cdot \hat{\mathbf B}`.

    Discretization: with :math:`\mathbb T[\bar{\mathbf B}] = \mathbb C \hat\Pi^1[\hat{\bar{\mathbf B}} \times
    \vec{\boldsymbol\Lambda}^v]` (:class:`~struphy.feec.variational_utilities.Hdiv0_transport_operator`),

    .. math::

        \begin{align}
        &\mathbb M^v[\hat\rho_0] \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t}
        - \mathbb T[\mathbf B_0]^\top \mathbb M^2 \mathbf b^{n+\frac12}
        - \mathbb T[\mathbf b^n]^\top \mathbb M^2 \mathbf b_0 = 0 \,,
        \\[2mm]
        &\frac{\mathbf b^{n+1} - \mathbf b^n}{\Delta t} + \mathbb T[\mathbf B_0]\, \mathbf u^{n+\frac12} = 0 \,,
        \qquad
        \frac{\hat{\mathbf b}^{n+1} - \hat{\mathbf b}^n}{\Delta t} + \mathbb T[\mathbf b^n]\, \mathbf u^{n+\frac12} = 0
        \,.
        \end{align}

    The weight :math:`\mathbf b^n` of the second-order term is the same in the momentum equation and in the update of
    :math:`\hat{\mathbf b}`, which makes the step energy conserving; it is first order in time. These are the
    equations of :class:`~struphy.propagators.variational_qb_evolve.VariationalQBEvolve` without the sound waves,
    with ``model="linear_q"`` and ``implicit_weights=False``, but solved with one linear solve
    (:class:`~struphy.propagators.variational_linear_base.VariationalLinearEvolve`) instead of a Picard iteration.
    The Schur complement is the shear-Alfvén operator
    :math:`\mathbb M^v[\hat\rho_0] + \frac{\Delta t^2}{4} \mathbb T[\mathbf B_0]^\top \mathbb M^2
    \mathbb T[\mathbf B_0]`, constant in time; the option ``schur_precond`` provides a Kronecker preconditioner of it
    (:func:`~struphy.feec.variational_utilities.alfven_schur_preconditioner`).

    With ``weights="midpoint"``, the weight of the second-order term is :math:`\mathbf b^{n+\frac12}` instead of
    :math:`\mathbf b^n`, in the momentum equation and in the update of :math:`\hat{\mathbf b}`. The map
    :math:`\mathbb R \mathbf b = \mathbb T[\mathbf b]^\top \mathbb M^2 \mathbf b_0` is linear in the weight and is
    assembled once
    (:meth:`~struphy.feec.variational_utilities.Hdiv0_transport_operator.weight_map`): still one linear solve, with
    the (not symmetric) Schur complement :math:`\mathbb M^v[\hat\rho_0] + \frac{\Delta t^2}{4}
    (\mathbb T[\mathbf B_0]^\top \mathbb M^2 + \mathbb R)\, \mathbb T[\mathbf B_0]`. The step is energy
    conserving and second order in time (implicit midpoint rule), without the numerical damping or growth of the
    frozen weight.

    The velocity can also be a 2-form, :math:`\mathbf u \in H(\textnormal{div})` (velocity variable in ``"Hdiv"``
    space). The scheme is the same, with the operators of
    :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator` for the equilibrium terms,
    :math:`\mathbb M^v[\hat\rho_0] \to \mathbb M^{2,n}` and :math:`\mathbb T[\mathbf B_0] \to \mathbb C\, \mathcal T^2`
    (:attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.T2`), and with
    :math:`\mathbb T[\mathbf b^n] = \mathbb C \hat\Pi^1[(\hat{\mathbf B}^n \times \vec{\boldsymbol\Lambda}^2)/\sqrt g]`
    for the second-order term. The energy is conserved in the same way.
    """

    class Variables:
        """Container for variables advanced by :class:`VariationalLinearAlfvenEvolve`.

        Attributes
        ----------
        u : FEECVariable
            Velocity variable in ``"H1vec"`` or ``"Hdiv"`` space.
        b : FEECVariable
            Magnetic-field perturbation in ``"Hdiv"`` space.
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
            assert new.space in ("H1vec", "Hdiv")
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
            Auxiliary second-order variable :math:`\hat{\mathbf B}` (``"Hdiv"`` space), advanced with
            :math:`\partial_t \hat{\mathbf B} + \mathcal T_{\mathbf B}(\mathbf u) = 0`. It enters only the energy.
        """
        self.variables = self.Variables()
        self.bt2 = bt2

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`VariationalLinearAlfvenEvolve`.

        Parameters
        ----------
        weights : {"frozen", "midpoint"}, default="midpoint"
            Weight of the second-order term: :math:`\mathbf b^n` (first order in time) or :math:`\mathbf b^{n+1/2}`
            (second order, with the assembled map :math:`\mathbb R`).
        solver : {"pcg", "pcg_sr", "cg", "pbicgstab"}, default=None
            Linear solver of the Schur system; None gives ``"pcg"`` for ``weights="frozen"`` and ``"pbicgstab"`` for
            ``weights="midpoint"``, whose Schur complement is not symmetric (only ``"pbicgstab"``).
        precond : {"MassMatrixPreconditioner", None}, default="MassMatrixPreconditioner"
            Preconditioner of the Schur system (of its velocity block).
        solver_params : SolverParameters, default=None
            Linear-solver controls. The system is solved for a normalized right-hand side, so ``tol`` is relative to
            the norm of the residual at :math:`t^n`.
        info : bool, default=False
            Log the information of the linear solver in every step.
        relative_tol : bool, default=True
            Whether the tolerance of ``solver_params`` is relative to the norm of the right-hand side (the residual at
            :math:`t^n`, normalized before the solve), or absolute, as in the non-variational propagators (e.g.
            :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator`). The energy is conserved up
            to the residual of the linear solve.
        schur_precond : {"MassMatrixPreconditioner", "StiffnessPreconditioner"}, default="MassMatrixPreconditioner"
            Preconditioner of the Schur system: the one of ``precond`` (velocity block), or the Kronecker (fast
            diagonalization) preconditioner of the whole shear-Alfven Schur operator
            (:func:`~struphy.feec.variational_utilities.alfven_schur_preconditioner`, rebuilt when the time step
            changes). The latter needs ``solver="pcg"`` (or ``"pbicgstab"``), tensor-product splines and the global commuting projectors;
            otherwise ``precond`` is used (with a warning).
        exb_correction : bool, default=True
            For ``schur_precond="StiffnessPreconditioner"``: add the auxiliary space correction of the
            :math:`\mathbf E \times \mathbf B` flows :math:`\mathbf B_0 \times \nabla \phi`.
        """

        # specific literals
        OptsPrecond = Literal["MassMatrixPreconditioner", None]
        OptsSchurPrecond = Literal["MassMatrixPreconditioner", "StiffnessPreconditioner"]
        OptsWeights = Literal["frozen", "midpoint"]
        OptsSolver = Literal["pcg", "pcg_sr", "cg", "pbicgstab"]
        # propagator options
        weights: OptsWeights = "midpoint"
        solver: OptsSolver = None
        precond: OptsPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        info: bool = False
        relative_tol: bool = True
        schur_precond: OptsSchurPrecond = "MassMatrixPreconditioner"
        exb_correction: bool = True

        def __post_init__(self):
            # checks
            check_option(self.weights, self.OptsWeights)
            if self.solver is None:
                self.solver = "pcg" if self.weights == "frozen" else "pbicgstab"
            check_option(self.solver, self.OptsSolver)
            assert self.weights == "frozen" or self.solver == "pbicgstab", (
                f'weights="midpoint" needs solver="pbicgstab" (not symmetric), got {self.solver!r}.'
            )
            check_option(self.precond, self.OptsPrecond)
            check_option(self.schur_precond, self.OptsSchurPrecond)

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
        self._bt2 = None if self.bt2 is None else self.bt2.spline.vector
        self._info = self.options.info and (self.rank == 0)

        # assembly of WMMnew happens in VariationalDensityEvolve
        self._allocate_mass(self.options.precond)

        # transport operators with the equilibrium (constant) and with the perturbation (weight b^n)
        if self._u_space == "Hdiv":
            # equilibrium term as in ShearAlfvenPropagator
            self._T0 = self.derham.curl @ self.basis_ops.T2
            self._T0T = self._T0.T
        else:
            self._T0 = Hdiv0_transport_operator(self.derham)
            self._T0T = self._T0.T
            self._T0.update_coeffs(self.projected_equil.b2)
            self._T0T.update_coeffs(self.projected_equil.b2)
        self._Tb = Hdiv0_transport_operator(self.derham, u_space=self._u_space, phys_domain=self.domain)
        self._TbT = self._Tb.T

        # linear form of the equilibrium field, -M2 b0
        self._M2b0 = -self.mass_ops.M2.dot(self.projected_equil.b2)

        # midpoint weights: the second-order term R b = T[b]^T (-M2 b0), assembled once
        self._midpoint = self.options.weights == "midpoint"
        self._R = self._TbT.weight_map(self._M2b0) if self._midpoint else None

        self._allocate_schur_solver(
            self._T0,
            self._T0T @ self.mass_ops.M2,
            -1.0,
            self.derham.V2pol,
            self.options.solver,
            self.options.solver_params,
            R=self._R,
        )

        # Kronecker preconditioner of the Schur operator, built for the time step in prepare_time_step
        self._schur_pc = alfven_schur_preconditioner(self, [], u_space=self._u_space)
        self._schur_pc_dt = None

        # temporaries
        u = self.variables.u.spline.vector
        b = self.variables.b.spline.vector
        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_fu = u.space.zeros()
        self._tmp_fu2 = u.space.zeros()
        self._tmp_bn1 = b.space.zeros()
        self._tmp_bn12 = b.space.zeros()
        self._tmp_fb = b.space.zeros()
        self._tmp_M2b = b.space.zeros()

    def prepare_time_step(self, dt: float):
        """Kronecker preconditioner of the Schur operator for the time step ``dt`` (with
        ``schur_precond="StiffnessPreconditioner"``): built in the first step, rebuilt when dt changes."""
        if self._schur_pc is not None and dt != self._schur_pc_dt:
            self._inv_J.precond = self._schur_pc.build(dt)
            self._schur_pc_dt = dt

    @profile
    def __call__(self, dt):
        # preconditioner for the current time step (no-op unless dt changed)
        self.prepare_time_step(dt)
        # WMMnew is assembled in VariationalDensityEvolve: update the diagonal scaling of its preconditioner
        self._refresh_mass_pc()

        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        # residual at (u^n, b^n): the force -T0^T M2 b^n - T[b^n]^T M2 b0 and the induction T0 u^n (times dt)
        M2b = self.mass_ops.M2.dot(bn, out=self._tmp_M2b)
        M2b *= -1.0
        f_u = self._T0T.dot(M2b, out=self._tmp_fu)
        if self._midpoint:
            f_u += self._R.dot(bn, out=self._tmp_fu2)
        else:
            # weight of the second-order term at t^n
            self._Tb.update_coeffs(bn)
            self._TbT.update_coeffs(bn)
            f_u += self._TbT.dot(self._M2b0, out=self._tmp_fu2)
        f_u *= dt
        f_b = self._T0.dot(un, out=self._tmp_fb)
        f_b *= dt

        incr = self._solve_increment(dt, f_u, f_b)
        un1 = un.copy(out=self._tmp_un1)
        un1 -= incr[0]
        bn1 = bn.copy(out=self._tmp_bn1)
        bn1 -= incr[1]

        # auxiliary second-order variable, with the same weight as the momentum equation (b^n or b^{n+1/2})
        if self._bt2 is not None:
            if self._midpoint:
                bn12 = bn.copy(out=self._tmp_bn12)
                bn12 += bn1
                bn12 *= 0.5
                self._Tb.update_coeffs(bn12)
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5
            b_advection = self._Tb.dot(un12, out=self._tmp_fb)
            b_advection *= dt
            self._bt2 -= b_advection

        self.update_feec_variables(u=un1, b=bn1)
