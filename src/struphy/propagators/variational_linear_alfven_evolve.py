import logging
from dataclasses import dataclass
from typing import Literal

from line_profiler import profile

from struphy.feec.preconditioner import MassMatrixPreconditioner
from struphy.feec.variational_utilities import Hdiv0_transport_operator, alfven_schur_preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.utils import EquilibriumCurrentTerm
from struphy.propagators.variational_linear_base import VariationalLinearEvolve
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VariationalLinearAlfvenEvolve(VariationalLinearEvolve):
    r""":ref:`FEEC <gempic>` discretization of the coupling of the velocity and the magnetic field (shear-Alfvén and
    fast magnetosonic waves) of the linear variational MHD models.

    **Continuous problem.** Find :math:`\mathbf u \in (H^1)^3` (or :math:`\mathbf u \in H_0(\textnormal{div})`) and
    :math:`\mathbf B \in H_0(\textnormal{div})`, the perturbation of the equilibrium :math:`\mathbf B_0`, such that,
    with the transport operator :math:`\mathcal T_{\bar{\mathbf B}}(\mathbf v) = \nabla \times (\bar{\mathbf B} \times
    \mathbf v)`,

    .. math::

        \int_\Omega \partial_t (\rho_0 \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x
        - \int_\Omega \left(\mathbf B \cdot \mathcal T_{\mathbf B_0}(\mathbf v)
        + \mathbf B_0 \cdot \mathcal T_{\mathbf B}(\mathbf v)\right)\textrm d \mathbf x = 0
        \qquad \forall \, \mathbf v\,,
        \qquad
        \partial_t \mathbf B + \mathcal T_{\mathbf B_0}(\mathbf u) = 0 \,,
        \qquad
        \partial_t \hat{\mathbf B} + \mathcal T_{\mathbf B}(\mathbf u) = 0 \,.

    The equations are the Euler--Poincaré equations of the Lagrangian with the magnetic energy
    :math:`\frac12 \int |\mathbf B|^2 + \int \mathbf B_0 \cdot \hat{\mathbf B}` (in forms:
    :math:`\frac12 \int \beta \wedge \star \beta + \int \beta_0 \wedge \star \hat\beta`), where the second-order field
    :math:`\hat{\mathbf B}` (``bt2``) is advected with the perturbation as weight; the force of the second term is the
    adjoint of that transport. The thermodynamic variable is not changed, so this propagator serves the linear models
    in :math:`p` and in :math:`q = \sqrt p` alike; together with
    :class:`~struphy.propagators.variational_linear_sound_evolve.VariationalLinearSoundEvolve` it splits the linear
    MHD step into the shear-Alfvén/fast and the sound waves.

    **Semi-discrete problem.** The interior product is the commuting projection
    :math:`\hat\Pi^1` of the pointwise product, :math:`\mathbb T[\bar{\mathbf b}] = \mathbb C \hat\Pi^1[\hat{\bar{\mathbf
    b}} \times \vec{\boldsymbol\Lambda}^v]` (:class:`~struphy.feec.variational_utilities.Hdiv0_transport_operator`;
    for a velocity 2-form the product is divided by :math:`\sqrt g`), used for the equilibrium
    (:math:`\mathbb T_0 = \mathbb T[\mathbf b_0]`) and for the perturbation. The Hodge star is the Galerkin one,
    :math:`\mathbb M^2`, and the force is the variation of the discrete Lagrangian, without integration by parts:

    .. math::

        \mathbb M^v_{\rho_0} \dot{\mathbf u} - \mathbb T_0^\top \mathbb M^2 \mathbf b
        - \mathbb T[\mathbf b]^\top \mathbb M^2 \mathbf b_0 = 0 \,,
        \qquad
        \dot{\mathbf b} + \mathbb T_0\, \mathbf u = 0 \,,
        \qquad
        \dot{\hat{\mathbf b}} + \mathbb T[\mathbf b]\, \mathbf u = 0 \,,

    with :math:`\mathbb M^v_{\rho_0}` = ``WMMnew`` for :math:`\mathbf u \in (H^1)^3` and :math:`\mathbb M^2_{\rho_0}` =
    ``M2n`` for a velocity 2-form. The energy :math:`\frac12 \mathbf u^\top \mathbb M^v_{\rho_0} \mathbf u + \frac12
    \mathbf b^\top \mathbb M^2 \mathbf b + \mathbf b_0^\top \mathbb M^2 \hat{\mathbf b}` is conserved. These are the
    operators of the perturbative model
    (:class:`~struphy.propagators.variational_pb_evolve.VariationalPBEvolve`, ``model="perturbative"``), which
    reduces to this step at small amplitude.

    **Time discretization.** Implicit midpoint rule for :math:`(\mathbf u, \mathbf b)`,
    :math:`\hat{\mathbf b}` advanced with the same weight:

    .. math::

        \begin{align}
        &\mathbb M^v_{\rho_0} \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t}
        - \mathbb T_0^\top \mathbb M^2 \mathbf b^{n+\frac12} - \mathbb T[\mathbf b^w]^\top \mathbb M^2 \mathbf b_0 = 0 \,,
        \qquad
        \frac{\mathbf b^{n+1} - \mathbf b^n}{\Delta t} + \mathbb T_0\, \mathbf u^{n+\frac12} = 0 \,,
        \\[2mm]
        &\frac{\hat{\mathbf b}^{n+1} - \hat{\mathbf b}^n}{\Delta t} + \mathbb T[\mathbf b^w]\, \mathbf u^{n+\frac12} = 0 \,,
        \end{align}

    which conserves the energy exactly for any weight :math:`\mathbf b^w`. The option ``weights`` sets it:

    - ``"midpoint"`` (default): :math:`\mathbf b^w = \mathbf b^{n+\frac12}`, second order in time. The force
      :math:`\mathbb T[\mathbf b^w]^\top \mathbb M^2 \mathbf b_0` is linear in :math:`\mathbf b^w` but not available as a
      matrix; the system is solved by a Newton iteration on :math:`(\mathbf u^{n+1}, \mathbf b^{n+1})` with the exact
      residual and the approximate matrix :math:`\mathbb W` of :math:`\int \mathbf v \cdot (\mathbf j_0 \times
      \mathbf B)` (:class:`~struphy.propagators.utils.EquilibriumCurrentTerm`) in place of its derivative
      (:class:`~struphy.propagators.variational_linear_base.VariationalLinearEvolve`, Schur complement
      :math:`\mathbb M^v_{\rho_0} + \frac{\Delta t^2}{4} (\mathbb T_0^\top \mathbb M^2 + \mathbb W) \mathbb T_0`, not
      symmetric, ``solver="rpbicgstab"``). :math:`\mathbb W` only affects the convergence, not the solution
      (relative tolerance 1e-12).
    - ``"frozen"``: :math:`\mathbf b^w = \mathbf b^n`, first order in time: the force is explicit and one linear solve
      with the symmetric, constant shear-Alfvén Schur complement :math:`\mathbb M^v_{\rho_0} + \frac{\Delta t^2}{4}
      \mathbb T_0^\top \mathbb M^2 \mathbb T_0` gives the exact solution; ``schur_precond`` provides a Kronecker
      preconditioner of it (:func:`~struphy.feec.variational_utilities.alfven_schur_preconditioner`).
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
            Second-order field :math:`\hat{\mathbf B}` (``"Hdiv"`` space), advanced with
            :math:`\partial_t \hat{\mathbf B} + \mathcal T_{\mathbf B}(\mathbf u) = 0`. It enters only the energy,
            :math:`\mathbf b_0^\top \mathbb M^2 \hat{\mathbf b}`.
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
            (second order).
        solver : {"pcg", "pcg_sr", "cg", "pbicgstab", "rpbicgstab"}, default=None
            Linear solver of the Schur system; None gives ``"pcg"`` for ``weights="frozen"`` and ``"rpbicgstab"``
            (BiCGStab with right preconditioning, see :class:`~struphy.linear_algebra.schur_solver.SchurSolverFull`)
            for ``weights="midpoint"``, whose Schur complement is not symmetric (only ``"rpbicgstab"`` or
            ``"pbicgstab"``).
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
            changes). The latter needs ``solver="pcg"`` (or ``"rpbicgstab"``, ``"pbicgstab"``), tensor-product splines and the global commuting projectors;
            otherwise ``precond`` is used (with a warning).
        exb_correction : bool, default=True
            For ``schur_precond="StiffnessPreconditioner"``: add the auxiliary space correction of the
            :math:`\mathbf E \times \mathbf B` flows :math:`\mathbf B_0 \times \nabla \phi`.
        """

        # specific literals
        OptsPrecond = Literal["MassMatrixPreconditioner", None]
        OptsSchurPrecond = Literal["MassMatrixPreconditioner", "StiffnessPreconditioner"]
        OptsWeights = Literal["frozen", "midpoint"]
        OptsSolver = Literal["pcg", "pcg_sr", "cg", "pbicgstab", "rpbicgstab"]
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
                self.solver = "pcg" if self.weights == "frozen" else "rpbicgstab"
            check_option(self.solver, self.OptsSolver)
            assert self.weights == "frozen" or self.solver in ("pbicgstab", "rpbicgstab"), (
                f'weights="midpoint" needs solver="rpbicgstab" or "pbicgstab" (not symmetric), got {self.solver!r}.'
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

        # transport operator with the equilibrium, T0 = C Pi^1[b0 x .] (the same as in the perturbative model)
        self._T0 = Hdiv0_transport_operator(self.derham, u_space=self._u_space, phys_domain=self.domain)
        self._T0T = self._T0.T
        self._T0.update_coeffs(self.projected_equil.b2)
        self._T0T.update_coeffs(self.projected_equil.b2)

        self._midpoint = self.options.weights == "midpoint"
        self._newton_tol = 1e-12
        self._newton_maxiter = 30
        self._init_second_order_term()

        self._allocate_schur_solver(
            self._T0,
            self._T0T @ self.mass_ops.M2,
            -1.0,
            self.derham.V2pol,
            self.options.solver,
            self.options.solver_params,
            R=self._R if self._midpoint else None,
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
        self._tmp_du = u.space.zeros()
        self._tmp_fu3 = u.space.zeros()

    def _init_second_order_term(self):
        r"""Second-order term :math:`\mathbf b_0^\top \mathbb M^2 \hat{\mathbf b}` (Galerkin Hodge star): the transport
        :math:`\mathbb T[\mathbf b^w]` of :math:`\hat{\mathbf b}`, whose adjoint applied to :math:`\mathbb M^2 \mathbf
        b_0` is the force, and the matrix :math:`\mathbb W` (``-self._R``) of :math:`\int \mathbf v \cdot (\mathbf
        j_0 \times \mathbf B)`, used only in the Newton matrix as an approximation of the derivative of that force."""
        self._R = EquilibriumCurrentTerm(self.derham, self.domain, self.mass_ops, self.projected_equil, self._u_space).R
        self._Tw = Hdiv0_transport_operator(self.derham, u_space=self._u_space, phys_domain=self.domain)
        self._TwT = self._Tw.T
        # linear form of the second-order energy b0^T M2 b-hat (with the sign of the force term)
        self._lf0 = self.mass_ops.M2.dot(self.projected_equil.b2)
        self._lf0 *= -1.0

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

        # Newton iteration on (u^{n+1}, b^{n+1}): the residual has the exact force T[b^w]^T M2 b0 of the second-order
        # term, the Newton matrix W instead of its derivative; with weights b^n the force is explicit and one solve
        # is exact
        un1 = un.copy(out=self._tmp_un1)
        bn1 = bn.copy(out=self._tmp_bn1)
        if not self._midpoint:
            self._Tw.update_coeffs(bn)
            self._TwT.update_coeffs(bn)
        err0 = None
        for it in range(self._newton_maxiter):
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5
            bn12 = bn.copy(out=self._tmp_bn12)
            bn12 += bn1
            bn12 *= 0.5
            if self._midpoint:
                self._Tw.update_coeffs(bn12)
                self._TwT.update_coeffs(bn12)
            # momentum: M (u1 - un) + dt (-T0^T M2 b^{n+1/2} - T[b^w]^T M2 b0)
            du = un1.copy(out=self._tmp_du)
            du -= un
            f_u = self._Mrho.dot(du, out=self._tmp_fu)
            M2b = self.mass_ops.M2.dot(bn12, out=self._tmp_M2b)
            M2b *= -1.0
            g = self._T0T.dot(M2b, out=self._tmp_fu2)
            g += self._TwT.dot(self._lf0, out=self._tmp_fu3)
            g *= dt
            f_u += g
            # induction: b1 - bn + dt T0 u^{n+1/2}
            f_b = self._T0.dot(un12, out=self._tmp_fb)
            f_b *= dt
            f_b += bn1
            f_b -= bn
            if self._midpoint:
                err = (f_u.inner(f_u) + f_b.inner(f_b)) ** 0.5
                if err0 is None:
                    err0 = err
                if err <= self._newton_tol * err0 or err == 0.0:
                    break
            elif it == 1:
                break
            incr = self._solve_increment(dt, f_u, f_b)
            un1 -= incr[0]
            bn1 -= incr[1]
        else:
            logger.warning(f"{self.__class__.__name__}: Newton iteration not converged ({err / err0:.2e}).")
        if self._info:
            logger.info(f"{self.__class__.__name__}: {it} Newton iterations")

        # second-order field with the same weight: b-hat += -dt T[b^w] u^{n+1/2} (T[b^w] is set to b^w above)
        if self._bt2 is not None:
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5
            if self._midpoint:
                bn12 = bn.copy(out=self._tmp_bn12)
                bn12 += bn1
                bn12 *= 0.5
                self._Tw.update_coeffs(bn12)
            db = self._Tw.dot(un12, out=self._tmp_fb)
            db *= dt
            self._bt2 -= db
            self._bt2.update_ghost_regions()

        self.update_feec_variables(u=un1, b=bn1)
