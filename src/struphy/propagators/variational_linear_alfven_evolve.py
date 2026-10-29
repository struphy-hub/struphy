import logging
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec.mass import L2Projector
from struphy.feec.preconditioner import MassMatrixPreconditioner
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

    Discretization: the equilibrium term with :math:`\mathbb T[\bar{\mathbf B}] = \mathbb C \hat\Pi^1[\hat{\bar{\mathbf
    B}} \times \vec{\boldsymbol\Lambda}^v]` (:class:`~struphy.feec.variational_utilities.Hdiv0_transport_operator`),
    and the second-order term in weak form, :math:`\int \mathbf B_0 \cdot \mathcal T_{\mathbf B}(\mathbf v) = \int
    \mathbf v \cdot (\mathbf J_0 \times \mathbf B)` (for :math:`\mathbf B_0 \times \mathbf v = 0` on the boundary), with
    the discrete equilibrium current :math:`\mathbf j_0 = \mathbb M_1^{-1} \mathbb C^\top \mathbb M_2 \mathbf b_0` and
    the weighted mass matrix :math:`\mathbb W` of :math:`\int \mathbf v \cdot (\mathbf j_0 \times \mathbf B)`:

    .. math::

        \begin{align}
        &\mathbb M^v[\hat\rho_0] \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t}
        - \mathbb T[\mathbf B_0]^\top \mathbb M^2 \mathbf b^{n+\frac12} - \mathbb W \mathbf b^w = 0 \,,
        \qquad
        \frac{\mathbf b^{n+1} - \mathbf b^n}{\Delta t} + \mathbb T[\mathbf B_0]\, \mathbf u^{n+\frac12} = 0 \,,
        \\[2mm]
        &\frac{\hat{\mathbf b}^{n+1} - \hat{\mathbf b}^n}{\Delta t}
        + \mathbb C \mathbb M_1^{-1} \mathbf y(\mathbf b^w \times \mathbf u^{n+\frac12}) = 0 \,,
        \qquad
        \mathbf y(\mathbf w)_k = \int_\Omega \mathbf w \cdot \vec{\boldsymbol\Lambda}^1_k \,\textrm d \mathbf x \,.
        \end{align}

    The auxiliary field is :math:`\hat{\mathbf B} = \nabla \times \mathbf A` with :math:`\partial_t \mathbf A = -
    \mathbf B \times \mathbf u` projected into :math:`H(\textnormal{curl})`; its energy :math:`\mathbf b_0^\top
    \mathbb M_2 \hat{\mathbf b}` changes by exactly :math:`-\Delta t\, \mathbf j_0 \cdot \mathbf y = -\Delta t\,
    (\mathbf u^{n+\frac12})^\top \mathbb W \mathbf b^w`, the work of the force, which makes the step energy conserving
    (both forms are evaluated at the same quadrature points).

    The option ``weights`` sets :math:`\mathbf b^w`. With ``"midpoint"`` (default), :math:`\mathbf b^w =
    \mathbf b^{n+\frac12}`: one linear solve
    (:class:`~struphy.propagators.variational_linear_base.VariationalLinearEvolve`) with the (not symmetric) Schur
    complement :math:`\mathbb M^v[\hat\rho_0] + \frac{\Delta t^2}{4} (\mathbb T[\mathbf B_0]^\top \mathbb M^2 +
    \mathbb W)\, \mathbb T[\mathbf B_0]`; the step is the implicit midpoint rule, energy conserving and second order in
    time. With ``"frozen"``, :math:`\mathbf b^w = \mathbf b^n` (first order in time): the Schur complement is the
    shear-Alfvén operator :math:`\mathbb M^v[\hat\rho_0] + \frac{\Delta t^2}{4} \mathbb T[\mathbf B_0]^\top
    \mathbb M^2 \mathbb T[\mathbf B_0]`, symmetric and constant in time; the option ``schur_precond`` provides a
    Kronecker preconditioner of it (:func:`~struphy.feec.variational_utilities.alfven_schur_preconditioner`).

    The velocity can also be a 2-form, :math:`\mathbf u \in H(\textnormal{div})` (velocity variable in ``"Hdiv"``
    space). The scheme is the same, with the operators of
    :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator` for the equilibrium terms,
    :math:`\mathbb M^v[\hat\rho_0] \to \mathbb M^{2,n}` and :math:`\mathbb T[\mathbf B_0] \to \mathbb C\, \mathcal T^2`
    (:attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.T2`). The energy is conserved in the same way.
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

        # second-order term in weak form: the force int v . (j0 x b) of the discrete equilibrium current
        # j0 = M1^{-1} C^T M2 b0 (R = -W below), and the auxiliary field b-hat = curl a, a the L2 projection of b x u
        self._midpoint = self.options.weights == "midpoint"
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

    def _init_second_order_term(self):
        r"""The matrix :math:`\mathbb W` of :math:`\int \mathbf v \cdot (\mathbf j_0 \times \mathbf B)` (``-self._R``) and
        the evaluation of :math:`\mathbf B \times \mathbf u` at the quadrature points for the auxiliary field.

        With :math:`\mathbf u = DF \tilde{\mathbf u}`, :math:`\tilde{\mathbf u} = \hat{\mathbf u}^2 / \sqrt g` (Hdiv)
        or :math:`\hat{\mathbf u}` (H1vec), both forms are integrals of :math:`\hat{\mathbf j}_0^\top G^{-1}
        (\hat{\mathbf B}^2 \times \tilde{\mathbf u}) \sqrt g` at the same quadrature points, so that
        :math:`\mathbf j_0 \cdot \mathbf y(\mathbf b \times \mathbf u) = \mathbf u^\top \mathbb W \mathbf b` exactly."""
        tol, maxiter = 1e-14, 3000
        M1 = self.mass_ops.M1
        self._inv_M1 = inverse(M1, "pcg", pc=MassMatrixPreconditioner(M1), tol=tol, maxiter=maxiter, recycle=True)
        self._M2b0 = self.mass_ops.M2.dot(self.projected_equil.b2)
        j0 = self._inv_M1.dot(self.derham.curl.T.dot(self._M2b0))

        # spline evaluation at the quadrature points
        grid = [pts.flatten() for pts in self.derham.V3splines.quad_grid_pts[0]]
        self._spans, bn, bd = self.derham.prepare_eval_tp_fixed(grid)
        self._bases_1form = [[bd[0], bn[1], bn[2]], [bn[0], bd[1], bn[2]], [bn[0], bn[1], bd[2]]]
        self._bases_2form = [[bn[0], bd[1], bd[2]], [bd[0], bn[1], bd[2]], [bd[0], bd[1], bn[2]]]
        self._bases_u = self._bases_2form if self._u_space == "Hdiv" else [[bn[0], bn[1], bn[2]]] * 3
        sqrt_g = self.domain.jacobian_det(*grid)
        # u-tilde / u-hat
        self._u_scale = 1.0 / sqrt_g if self._u_space == "Hdiv" else None

        j0f = self.derham.create_spline_function("j0", "Hcurl")
        j0f.vector = j0
        j0_v = j0f.eval_tp_fixed_loc(self._spans, self._bases_1form)
        # c = G^{-1} j0 sqrt(g) times the factor of u-tilde: weights of W, u . (c x b) = u^T [c]_x b
        g_inv = self.domain.metric_inv(*grid)
        c = [sum(g_inv[a, e] * j0_v[e] for e in range(3)) for a in range(3)]
        if self._u_space == "H1vec":
            c = [ca * sqrt_g for ca in c]
        # R = -W, with the weights -[c]_x
        self._R = self.mass_ops.create_weighted_mass("Hdiv", self._u_space)
        self._R.assemble([[None, c[2], -c[1]], [-c[2], None, c[0]], [c[1], -c[0], None]])

        if self._bt2 is not None:
            self._bf = self.derham.create_spline_function("b_aux", "Hdiv")
            self._uf = self.derham.create_spline_function("u_aux", self._u_space)
            self._b_values = [xp.zeros_like(sqrt_g) for _ in range(3)]
            self._u_values = [xp.zeros_like(sqrt_g) for _ in range(3)]
            self._bxu_values = [xp.zeros_like(sqrt_g) for _ in range(3)]
            self._get_dofs_1 = L2Projector("Hcurl", self.mass_ops).get_dofs
            self._tmp_y = self.derham.V1pol.zeros()
            self._tmp_a = self.derham.V1pol.zeros()

    def _advance_aux(self, dt, un12, bw):
        r""":math:`\hat{\mathbf b} \leftarrow \hat{\mathbf b} - \Delta t\, \mathbb C \mathbb M_1^{-1}
        \mathbf y(\mathbf b^w \times \mathbf u^{n+1/2})`, with :math:`\mathbf y(\mathbf w)_k = \int \mathbf w \cdot
        \vec{\boldsymbol\Lambda}^1_k`: the energy :math:`\mathbf b_0^\top \mathbb M_2 \hat{\mathbf b}` changes by
        :math:`-\Delta t\, \mathbf j_0 \cdot \mathbf y = -\Delta t\, (\mathbf u^{n+1/2})^\top \mathbb W \mathbf b^w`."""
        self._bf.vector = bw
        self._uf.vector = un12
        b_v = self._bf.eval_tp_fixed_loc(self._spans, self._bases_2form, out=self._b_values)
        u_v = self._uf.eval_tp_fixed_loc(self._spans, self._bases_u, out=self._u_values)
        if self._u_scale is not None:
            for ui in u_v:
                ui *= self._u_scale
        bxu = self._bxu_values
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            bxu[i][:] = b_v[j] * u_v[k]
            bxu[i] -= b_v[k] * u_v[j]
        y = self._get_dofs_1(bxu, dofs=self._tmp_y, apply_bc=True)
        a = self._inv_M1.dot(y, out=self._tmp_a)
        ca = self.derham.curl.dot(a, out=self._tmp_fb)
        ca *= dt
        self._bt2 -= ca
        self._bt2.update_ghost_regions()

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
        f_u += self._R.dot(bn, out=self._tmp_fu2)
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
                bw = bn.copy(out=self._tmp_bn12)
                bw += bn1
                bw *= 0.5
            else:
                bw = bn
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5
            self._advance_aux(dt, un12, bw)

        self.update_feec_variables(u=un1, b=bn1)
