"Particle and FEEC variables are updated."

import logging
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation.kernels.dk_maxwell import dk_maxwell
from struphy.pic.accumulation.kernels.dk_maxwell_mf import dk_maxwell_mf
from struphy.pic.accumulation.kernels.marker_operator_asym_dot import marker_operator_asym_dot
from struphy.pic.accumulation.kernels.marker_operator_rank1_dot import marker_operator_rank1_dot
from struphy.pic.accumulation.particles_to_grid import Accumulator, AccumulatorVector, MarkerOperator
from struphy.pic.pushing.kernels.push_vpar_with_efield import push_vpar_with_efield
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class DriftKineticMaxwellCoupling(Propagator):
    r"""PIC-FEEC discretization of the following equations:
    find :math:`\mathbf E \in H(\textnormal{curl})`, :math:`\tilde{\mathbf B} \in H(\textnormal{div})` and the
    guiding-center distribution :math:`f(\mathbf X, v_\parallel, \mu)` (positions fixed in this step) such that

    .. math::

        \kappa \int_{\Omega} \frac{\partial \mathbf E}{\partial t} \cdot \mathbf F \,\textrm d \mathbf x
        + \chi \int_{\Omega} \frac{n_0}{|B_0|^2} \frac{\partial \mathbf E_\perp}{\partial t} \cdot \mathbf F_\perp
        \,\textrm d \mathbf x
        &= c_1 \int_{\Omega} \tilde{\mathbf B} \cdot \nabla \times \mathbf F \,\textrm d \mathbf x
        - c_2 \int_{\Omega} \iint f^\textnormal{vol} \left( v_\parallel \frac{\mathbf B^*}{B^*_\parallel}
        + \frac{\mathbf E \times \mathbf b_0}{B^*_\parallel} \right) \cdot \mathbf F
        \,\textrm d v_\parallel \textrm d \mu \,\textrm d \mathbf x
        \qquad \forall \, \mathbf F \in H(\textnormal{curl})\,,
        \\[2mm]
        \frac{\partial \tilde{\mathbf B}}{\partial t} + \nabla\times\mathbf E &= 0\,,
        \\[2mm]
        \frac{\partial f}{\partial t} + c_3\, \frac{\mathbf B^*}{B^*_\parallel} \cdot \mathbf E\,
        \frac{\partial f}{\partial v_\parallel} &= 0\,,

    with :math:`\mathbf E_\perp = \mathbf E - (\mathbf b_0 \cdot \mathbf E) \mathbf b_0`, the phase-space
    density :math:`f^\textnormal{vol} = B^*_\parallel f` (the marker weights) and, as in
    :class:`~struphy.propagators.push_guiding_center_parallel.PushGuidingCenterParallel`,

    .. math::

        \mathbf B^* = \mathbf B_0 + \varepsilon v_\parallel \nabla \times \mathbf b_0\,,\qquad
        B^*_\parallel = \mathbf b_0 \cdot (\mathbf B_0 + \tilde{\mathbf B}) + \varepsilon v_\parallel
        (\nabla \times \mathbf b_0) \cdot \mathbf b_0\,.

    The current is the guiding-center current :math:`\iint f^\textnormal{vol} \dot{\mathbf X}` without the
    magnetic drifts (which move the positions): parallel streaming, including the curvature drift in
    :math:`\mathbf B^*`, and (option ``exb_current``) the :math:`E \times B` drift. The coefficients are
    :math:`\kappa \geq 0`, :math:`\chi \geq 0` with :math:`\kappa + \chi > 0`, :math:`c_1 > 0` and
    :math:`c_2 c_3 > 0`; the defaults :math:`\kappa = 1`, :math:`\chi = \alpha^2`, :math:`c_1 = 1`,
    :math:`c_2 = \alpha^2/\varepsilon`, :math:`c_3 = 1/\varepsilon` are the units of
    :class:`~struphy.models.vlasov_maxwell_one_species.VlasovMaxwellOneSpecies`. The :math:`\chi` term is the
    (linearized, long-wavelength) polarization current of the guiding centers, which the drift-kinetic ordering
    removes from :math:`\dot{\mathbf X}`: without it, the perpendicular electrostatic field is undetermined for
    :math:`\kappa = 0` (flute modes), and for :math:`\kappa = 1` the shear Alfvén wave travels at the speed of light.

    With the FE coefficients :math:`\mathbf e \in \mathbb R^{N_1}` of :math:`\mathbf E \in V^1_h` and
    :math:`\mathbf b \in \mathbb R^{N_2}` of :math:`\tilde{\mathbf B} \in V^2_h`, and :math:`N` markers with logical
    positions :math:`\boldsymbol \eta_p` (fixed), parallel velocities :math:`v_{\parallel,p}` and weights :math:`w_p`,
    the semi-discrete system reads

    .. math::

        \mathbb M_E \dot{\mathbf e} = c_1\, \mathbb C^\top \mathbb M_2 \mathbf b
        - c_2 \left( \mathbf j(\mathbf V) + \mathbb S\, \mathbf e \right) \,,
        \qquad
        \dot{\mathbf b} = - \mathbb C \mathbf e\,,
        \qquad
        \dot v_{\parallel,p} = c_3\, \mathbf a_p \cdot \mathbf \Lambda^1_p \mathbf e\,,

    with :math:`\mathbb M_E = \kappa\, \mathbb M_1 + \chi\, \mathbb M_1^\perp` (:math:`\mathbb M_1^\perp` =
    :attr:`~struphy.feec.mass.WeightedMassOperators.M1gyro`), :math:`\mathbf a_p =
    \hat{\mathbf B}^{*2} / \hat B^{*3}_\parallel (\boldsymbol \eta_p)`, :math:`\mathbf \Lambda^1_p` the basis of
    :math:`V^1_h` at :math:`\boldsymbol \eta_p`, and

    .. math::

        \mathbf j(\mathbf V) = \sum_p w_p v_{\parallel,p}\, (\mathbf \Lambda^1_p)^\top \mathbf a_p\,,
        \qquad
        \mathbb A_\parallel = \sum_p w_p\, (\mathbf \Lambda^1_p)^\top \mathbf a_p \mathbf a_p^\top \mathbf \Lambda^1_p\,,
        \qquad
        \mathbf f^\top \mathbb S\, \mathbf e = \sum_p \frac{w_p}{\hat B^{*3}_\parallel(\boldsymbol \eta_p)}\,
        (\mathbf \Lambda^1_p \mathbf f) \cdot \big( (\mathbf \Lambda^1_p \mathbf e) \times \hat{\mathbf b}^1_0
        (\boldsymbol \eta_p) \big)\,.

    :math:`\mathbb A_\parallel` is symmetric positive semi-definite (the parallel response), :math:`\mathbb S` is
    antisymmetric (the :math:`E \times B` current does no work). The system conserves the energy

    .. math::

        H = \frac{1}{2}\, \mathbf e^\top \mathbb M_E \mathbf e + \frac{c_1}{2}\, \mathbf b^\top \mathbb M_2 \mathbf b
        + \frac{c_2}{c_3} \sum_p w_p \left( \frac{v_{\parallel,p}^2}{2} + \mu_p |B_0|(\boldsymbol \eta_p) \right)
        \qquad (:meth:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling.energy`)\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point), with :math:`\mathbf a_p` and :math:`\mathbb S`
    evaluated at the beginning of the step, which conserves :math:`H` exactly (up to the solver tolerance; for
    constant weights). Eliminating :math:`\mathbf b^\textnormal{mid}` and :math:`\mathbf V^\textnormal{mid}` gives the
    Schur system for the mid-point field

    .. math::

        \left( \mathbb M_E + \frac{\Delta t^2}{4} \left( c_1\, \mathbb C^\top \mathbb M_2 \mathbb C
        + c_2 c_3\, \mathbb A_\parallel \right) + \frac{\Delta t}{2}\, c_2\, \mathbb S \right) \mathbf e^\textnormal{mid}
        = \mathbb M_E \mathbf e^n + \frac{\Delta t}{2} \left( c_1\, \mathbb C^\top \mathbb M_2 \mathbf b^n
        - c_2\, \mathbf j(\mathbf V^n) \right)\,,

    accumulated by :func:`~struphy.pic.accumulation.kernels.dk_maxwell.dk_maxwell_kernels.dk_maxwell` (or
    matrix-free) and solved with ``solver`` and ``precond`` (non-symmetric with the :math:`E \times B` current).
    Then

    .. math::

        \mathbf b^{n+1} = \mathbf b^n - \Delta t\, \mathbb C \mathbf e^\textnormal{mid}\,,
        \qquad
        v_{\parallel,p}^{n+1} = v_{\parallel,p}^n + \Delta t\, c_3\, \mathbf a_p \cdot \mathbf \Lambda^1_p
        \mathbf e^\textnormal{mid}\,,
        \qquad
        \mathbf e^{n+1} = 2 \mathbf e^\textnormal{mid} - \mathbf e^n\,,

    with the velocity update by
    :func:`~struphy.pic.pushing.kernels.push_vpar_with_efield.push_vpar_with_efield_kernels.push_vpar_with_efield`.

    **Quasi-neutral limit** :math:`\kappa = 0`. The parallel part of :math:`\mathbf e` is then determined by
    :math:`\mathbb A_\parallel` alone (which needs positive weights, i.e. full-f, and markers everywhere), the
    perpendicular part by :math:`\chi\, \mathbb M_1^\perp` and the curl-curl operator. The perpendicular field
    still carries the energy :math:`\frac 12 \mathbf e^\top \mathbb M_E \mathbf e`, so the Crank-Nicolson update is
    kept (storing :math:`\mathbf e^\textnormal{mid}` would dissipate it). The parallel part of the stored
    :math:`\mathbf e^{n+1}` is not determined by the scheme and may oscillate from step to step; it does not feed
    back, since the steps see :math:`\mathbf e^n` only through :math:`\mathbb M_E \mathbf e^n` and the
    :math:`E \times B` push only through :math:`\mathbf E_\perp`. With ``initial_projection``
    (default), the setup replaces :math:`(\mathbf b^0, \mathbf V^0)` by the mid-point state of one step, which
    removes the undamped period-2 oscillation of the constraint residual.

    **Preconditioner.** With :math:`\mathbb A_\parallel \approx \bar n\, \mathbb M_1^\parallel`
    (:attr:`~struphy.feec.mass.WeightedMassOperators.M1para`), the symmetric part of the Schur operator is
    approximately :math:`\mathbb C^\top \left(\frac{\Delta t^2 c_1}{4} \mathbb M_2\right) \mathbb C + \mathbb M_E
    + \frac{\Delta t^2 c_2 c_3 \bar n}{4} \mathbb M_1^\parallel`, preconditioned (default) by
    :class:`~struphy.feec.preconditioner.StiffnessPreconditioner`, built for the current time step
    (:meth:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling.prepare_time_step`).
    """

    class Variables:
        """Container for variables advanced by :class:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling`.

        Attributes
        ----------
        e : FEECVariable
            Electric-field variable in ``"Hcurl"`` space.
        b : FEECVariable
            Magnetic-field perturbation in ``"Hdiv"`` space.
        ions : PICVariable
            Guiding-center particle variable in ``"Particles5D"`` space.
        """

        def __init__(self):
            self._e: FEECVariable = None
            self._b: FEECVariable = None
            self._ions: PICVariable = None

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

        @property
        def ions(self) -> PICVariable:
            return self._ions

        @ions.setter
        def ions(self, new):
            assert isinstance(new, PICVariable)
            assert new.space == "Particles5D"
            self._ions = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling`.

        Parameters
        ----------
        kappa : float, default=1.0
            Coefficient :math:`\kappa \geq 0` of the displacement current :math:`\partial_t \mathbf E`;
            ``kappa=0.0`` is the quasi-neutral limit (needs ``chi > 0`` and full-f weights, no control variate).

        chi : float, default=None
            Coefficient :math:`\chi \geq 0` of the polarization current
            :math:`\frac{n_0}{|B_0|^2} \partial_t \mathbf E_\perp`. If ``None``, :math:`\alpha^2` of the species
            (the polarization of the guiding centers themselves); a larger value mimics the polarization of a heavier
            species. ``kappa + chi`` must be positive.

        c1 : float, default=1.0
            Coefficient :math:`c_1 > 0` of :math:`\nabla \times \tilde{\mathbf B}` in Ampère's law.

        c2 : float, default=None
            Coefficient of the current density in Ampère's law. If ``None``, :math:`\alpha^2/\varepsilon` of the
            species.

        c3 : float, default=None
            Coefficient of the parallel electric force (:math:`c_2 c_3 > 0`). If ``None``, :math:`1/\varepsilon` of
            the species.

        exb_current : bool, default=True
            Include the :math:`E \times B` current (antisymmetric operator :math:`\mathbb S`). The Schur system is then
            non-symmetric and needs a general solver. Switch it off if the :math:`E \times B` current is compensated
            (e.g. by other species).

        solver : OptsSolver, default="pbicgstab"
            Iterative solver of the Schur system, passed to ``feectools.linalg.solvers.inverse``. The symmetric
            solvers ``"pcg"`` and ``"cg"`` need ``exb_current=False``.

        precond : LiteralOptions.OptsMaxwellPrecond, default="StiffnessPreconditioner"
            Preconditioner of the Schur system (only for ``"pcg"`` and ``"pbicgstab"``; set to ``None`` otherwise).

            - ``"StiffnessPreconditioner"``: :class:`~struphy.feec.preconditioner.StiffnessPreconditioner` with
              ``derivative="curl"`` for the stabilized curl-curl approximation of the Schur operator (see the class
              docstring), rebuilt when the time step changes.
            - ``"MassMatrixPreconditioner"``, ``"PolarMassMatrixPreconditioner"``: preconditioner of
              :math:`\mathbb M_1`.
            - ``None``: no preconditioner.

        solver_params : SolverParameters, default=None
            Solver controls. If ``None``, defaults to ``SolverParameters()``.

        n_ref : float, default=None
            Reference density :math:`\bar n` in the stiffness preconditioner. If ``None``, the Rayleigh quotient
            :math:`\mathbf 1^\top \mathbb A_\parallel \mathbf 1 / \mathbf 1^\top \mathbb M_1^\parallel \mathbf 1` at
            the setup (a density average).

        matrix_free : bool, default=False
            Apply the particle matrices :math:`\mathbb A_\parallel` and :math:`\mathbb S` matrix-free, with their
            fillings stored per marker (:class:`~struphy.pic.accumulation.particles_to_grid.MarkerOperator`), instead
            of assembling them. Needs 7 auxiliary marker columns. Ignored on the CuPy backend.

        initial_projection : bool, default=True
            Only for ``kappa=0``: in the setup (first :meth:`prepare_time_step`), replace
            :math:`(\mathbf b, \mathbf V)` by the mid-point state of one step, which satisfies the constraint of the
            quasi-neutral limit exactly. Not applied to a restored (restart) state.
        """

        # specific literals
        OptsSolver = Literal["pbicgstab", "bicgstab", "gmres", "pcg", "cg"]

        kappa: float = 1.0
        chi: float = None
        c1: float = 1.0
        c2: float = None
        c3: float = None
        exb_current: bool = True
        solver: OptsSolver = "pbicgstab"
        precond: LiteralOptions.OptsMaxwellPrecond = "StiffnessPreconditioner"
        solver_params: SolverParameters = None
        n_ref: float = None
        matrix_free: bool = False
        initial_projection: bool = True

        def __post_init__(self):
            # checks
            check_option(self.solver, self.OptsSolver)
            check_option(self.precond, LiteralOptions.OptsMaxwellPrecond)
            assert self.kappa >= 0.0, f"kappa must be >= 0, got {self.kappa}."
            if self.chi is not None:
                assert self.chi >= 0.0, f"chi must be >= 0, got {self.chi}."
                assert self.kappa + self.chi > 0.0, (
                    "kappa + chi must be > 0: without displacement and polarization current, the perpendicular "
                    "electric field is undetermined."
                )
            assert self.c1 > 0.0, f"c1 must be > 0, got {self.c1}."
            if self.c2 is not None and self.c3 is not None:
                assert self.c2 * self.c3 > 0.0, f"c2 * c3 must be > 0, got c2={self.c2}, c3={self.c3}."
            if self.n_ref is not None:
                assert self.n_ref > 0.0, f"n_ref must be > 0, got {self.n_ref}."
            assert isinstance(self.exb_current, bool)
            assert isinstance(self.matrix_free, bool)
            assert isinstance(self.initial_projection, bool)
            if self.exb_current:
                assert self.solver not in ("pcg", "cg"), (
                    f'solver="{self.solver}" needs a symmetric system: use a general solver or exb_current=False.'
                )
            if self.solver not in ("pcg", "pbicgstab") and self.precond is not None:
                logger.warning(f'solver="{self.solver}" takes no preconditioner; ignoring precond="{self.precond}".')
                self.precond = None

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
        # coefficients
        equation_params = self.variables.ions.species.equation_params
        alpha = equation_params.alpha
        epsilon = equation_params.epsilon
        self._epsilon = epsilon
        self._kappa = self.options.kappa
        self._chi = alpha**2 if self.options.chi is None else self.options.chi
        self._c1 = self.options.c1
        self._c2 = alpha**2 / epsilon if self.options.c2 is None else self.options.c2
        self._c3 = 1.0 / epsilon if self.options.c3 is None else self.options.c3
        self._exb = self.options.exb_current
        assert self._c2 * self._c3 > 0.0, f"c2 * c3 must be > 0, got c2={self._c2}, c3={self._c3}."
        assert self._kappa + self._chi > 0.0, f"kappa + chi must be > 0, got kappa={self._kappa}, chi={self._chi}."

        assert not self.derham.polar_splines, "DriftKineticMaxwellCoupling does not support polar splines yet."

        particles = self.variables.ions.particles
        if particles.control_variate:
            assert self._kappa > 0.0, (
                "The quasi-neutral limit kappa=0 needs full-f weights: with the control variate the accumulated "
                "matrix is indefinite and the Schur system is not well-posed."
            )
            logger.warning(
                "DriftKineticMaxwellCoupling with control variate: the energy is not conserved exactly "
                "(the weights change)."
            )

        self._info = self.options.solver_params.info

        # magnetic background: B* (2-form, equilibrium) and B*_parallel with the perturbation b (frozen in a step)
        self._b2 = self.projected_equil.b2
        self._curl_unit_b2 = self.projected_equil.curl_unit_b2
        self._unit_b1 = self.projected_equil.unit_b1
        self._absB0 = self.projected_equil.absB0
        self._curl_unit_b_dot_b0 = self.projected_equil.curl_unit_b_dot_b0

        self._E0T = self.derham.extraction_ops["0"].transpose()
        self._PB = getattr(self.basis_ops, "PB")
        self._B_dot_b = self.derham.V0.zeros()
        self._B_dot_b_pol = self._PB.codomain.zeros()
        self._update_b_dot_b()

        args_fields = (
            self._b2[0]._data,
            self._b2[1]._data,
            self._b2[2]._data,
            self._curl_unit_b2[0]._data,
            self._curl_unit_b2[1]._data,
            self._curl_unit_b2[2]._data,
        )
        args_unit_b = (
            self._unit_b1[0]._data,
            self._unit_b1[1]._data,
            self._unit_b1[2]._data,
        )
        args_b_par = (
            self._B_dot_b._data,
            self._curl_unit_b_dot_b0._data,
        )

        # accumulation of A_par and S (assembled, or matrix-free) and the vector j;
        # the matrix-free kernels have no CUDA version
        self._matrix_free = self.options.matrix_free and xp.get_backend() != "cupy"
        if self._matrix_free:
            # the vector, and the fillings of the matrices per marker (in the first free columns of the marker array)
            column = particles.args_markers.first_free_idx
            assert particles.n_cols_aux >= 7, "matrix_free needs 7 free (auxiliary) marker columns."
            self._args_accum_pre = (epsilon,)
            self._args_accum_post = args_fields + args_unit_b + args_b_par + (column,)
            self._accum = AccumulatorVector(
                particles,
                "Hcurl",
                dk_maxwell_mf,
                self.mass_ops,
                self.domain.args_domain,
            )
            self._A_par = MarkerOperator(
                particles,
                "Hcurl",
                marker_operator_rank1_dot,
                self.mass_ops,
                self.domain.args_domain,
                (1, column),
                symmetry="symm",
            )
            self._S = MarkerOperator(
                particles,
                "Hcurl",
                marker_operator_asym_dot,
                self.mass_ops,
                self.domain.args_domain,
                (1, column + 4),
                symmetry="asym",
            )
        else:
            # one matrix A_par + lam * S, lam = 2 / (dt * c3) (set in prepare_time_step)
            self._args_accum_pre = (epsilon,)
            self._args_accum_post = args_fields + args_unit_b + args_b_par
            self._accum = Accumulator(
                particles,
                "Hcurl",
                dk_maxwell,
                self.mass_ops,
                self.domain.args_domain,
                add_vector=True,
                symmetry=None,
            )
            self._A_lam = self._accum.operators[0]
        self._lam = 0.0

        # operators
        self._M1 = self.mass_ops.M1
        self._M2 = self.mass_ops.M2
        self._curl = self.derham.curl
        self._curl_curl = self._curl.T @ self._M2 @ self._curl
        if self._kappa > 0.0 and self._chi > 0.0:
            self._ME = self._kappa * self._M1 + self._chi * self.mass_ops.M1gyro
        elif self._kappa > 0.0:
            self._ME = self._kappa * self._M1
        else:
            self._ME = self._chi * self.mass_ops.M1gyro

        # reference density for the stiffness preconditioner: density average of the parallel particle matrix
        if self.options.n_ref is None:
            self._accumulate()
            ones = self.derham.V1.zeros()
            for block in ones.blocks:
                block._data[:] = 1.0
            A_par = self._A_par if self._matrix_free else self._A_lam
            ones_A = ones.inner(A_par.dot(ones))
            ones_M = ones.inner(self.mass_ops.M1para.dot(ones))
            if ones_A > 0.0 and ones_M > 0.0:
                self._n_ref = ones_A / ones_M
            else:
                self._n_ref = 1.0
                logger.warning(
                    "DriftKineticMaxwellCoupling: no positive density average of the markers, using n_ref=1."
                )
        else:
            self._n_ref = self.options.n_ref

        # preconditioner (the stiffness preconditioner depends on dt and is built in prepare_time_step)
        self._precond = self.options.precond
        if self._precond in (None, "StiffnessPreconditioner"):
            self._mass_pc = None
        else:
            pc_class = getattr(preconditioner, self._precond)
            self._mass_pc = pc_class(self._M1)

        # solver kwargs (the Schur operator and its solver are built in prepare_time_step)
        kwargs = self.options.solver_params.__dict__.copy()
        kwargs.pop("info")
        self._solver_kwargs = kwargs
        self._solver = None
        self._schur_dt = None
        self._projected = False

        # buffers (the pusher reads e_mid)
        self._e_mid = self.derham.V1.zeros()
        self._e_new = self.derham.V1.zeros()
        self._e_rhs = self.derham.V1.zeros()
        self._e_tmp = self.derham.V1.zeros()
        self._b_new = self.derham.V2.zeros()
        self._b_tmp = self.derham.V2.zeros()

        # particle pusher: v^{n+1} = v^n + dt * c3 * a_p . Lambda^1 e_mid
        args_kernel = (
            (
                self.derham.args_derham,
                epsilon,
                self._e_mid.blocks[0]._data,
                self._e_mid.blocks[1]._data,
                self._e_mid.blocks[2]._data,
            )
            + args_fields
            + args_b_par
            + (self._c3,)
        )

        self._pusher = Pusher(
            particles,
            push_vpar_with_efield,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
            pushes_eta=False,
        )

    def _update_b_dot_b(self):
        """Parallel magnetic field b0 . (B0 + b) in the tensor-product space V0 (used in B*_parallel)."""
        B_dot_b_pol = self._PB.dot(self.variables.b.spline.vector, out=self._B_dot_b_pol)
        B_dot_b = self._E0T.dot(B_dot_b_pol, out=self._B_dot_b)
        B_dot_b += self._absB0
        B_dot_b.update_ghost_regions()

    def _accumulate(self):
        """Accumulate the particle matrices (or their fillings per marker) and the vector j at the current markers."""
        if self._matrix_free:
            self._accum(*self._args_accum_pre, *self._args_accum_post)
        else:
            self._accum(*self._args_accum_pre, self._lam, *self._args_accum_post)

    def prepare_time_step(self, dt: float):
        """Schur operator, its preconditioner and solver for the time step ``dt``: built in the setup, rebuilt when dt
        changes. The particle matrices enter by reference (assembled, or matrix-free reading the fillings stored in
        the markers), so they are updated by the accumulation in every step. For ``kappa=0`` (and
        ``initial_projection``), the first call also projects the state onto the constraint."""
        if dt == self._schur_dt:
            return

        schur = self._ME + (dt**2 * self._c1 / 4) * self._curl_curl
        if self._matrix_free:
            schur = schur + (dt**2 * self._c2 * self._c3 / 4) * self._A_par
            if self._exb:
                schur = schur + (dt * self._c2 / 2) * self._S
        else:
            # A_lam = A_par + lam S with lam = 2 / (dt c3): dt^2 c2 c3 / 4 A_lam = dt^2 c2 c3 / 4 A_par + dt c2 / 2 S
            self._lam = 2.0 / (dt * self._c3) if self._exb else 0.0
            schur = schur + (dt**2 * self._c2 * self._c3 / 4) * self._A_lam

        if self._precond == "StiffnessPreconditioner":
            stab = self._ME + (dt**2 * self._c2 * self._c3 * self._n_ref / 4) * self.mass_ops.M1para
            pc = preconditioner.StiffnessPreconditioner(
                self.mass_ops,
                "curl",
                sigma=1.0,
                mid=(dt**2 * self._c1 / 4) * self._M2,
                stab=stab,
            )
        else:
            pc = self._mass_pc

        kwargs = dict(self._solver_kwargs)
        if pc is not None:
            kwargs["pc"] = pc
        self._solver = inverse(schur, self.options.solver, **kwargs)
        self._schur_dt = dt

        # kappa = 0: consistent initial state, once (in the setup, before the initial data are saved)
        if self._kappa == 0.0 and self.options.initial_projection and not self._projected:
            self._project_onto_constraint(dt)
            self._projected = True

    def energy(self) -> float:
        r"""The energy :math:`H = \frac{1}{2} \mathbf e^\top \mathbb M_E \mathbf e + \frac{c_1}{2} \mathbf b^\top
        \mathbb M_2 \mathbf b + \frac{c_2}{c_3} \sum_p w_p (v_{\parallel,p}^2/2 + \mu_p |B_0|(\boldsymbol \eta_p))`
        conserved by the propagator (summed over all MPI ranks; a collective operation)."""
        e = self.variables.e.spline.vector
        b = self.variables.b.spline.vector
        particles = self.variables.ions.particles

        particles.save_magnetic_background_energy()
        valid = particles.valid_mks
        markers = particles.markers
        kinetic = xp.array(
            [
                xp.sum(
                    markers[5, valid] * (0.5 * markers[3, valid] ** 2 + markers[particles.first_diagnostics_idx, valid])
                )
            ]
        )
        if self.derham.comm is not None:
            self.derham.comm.Allreduce(MPI.IN_PLACE, kinetic, op=MPI.SUM)
        if particles.clone_config is not None:
            particles.clone_config.inter_comm.Allreduce(MPI.IN_PLACE, kinetic, op=MPI.SUM)

        energy = (
            0.5 * e.inner(self._ME.dot(e, out=self._e_tmp))
            + self._c1 / 2 * b.inner(self._M2.dot(b, out=self._b_tmp))
            + self._c2 / self._c3 * float(kinetic[0])
        )
        return energy

    def _mid_point_field(self, dt: float):
        """Accumulate at the current markers and solve the Schur system for the mid-point field (into ``_e_mid``,
        with updated ghost regions, which the pusher kernel reads)."""
        en = self.variables.e.spline.vector
        bn = self.variables.b.spline.vector

        # B*_parallel with b^n, then A_par, S (or their fillings per marker) and j
        self._update_b_dot_b()
        self._accumulate()

        # rhs = M_E e^n + dt/2 (c1 C^T M2 b^n - c2 j)
        rhs = self._curl.T.dot(self._M2.dot(bn, out=self._b_tmp), out=self._e_rhs)
        rhs *= self._c1
        j = self._accum.vectors[0]
        j.copy(out=self._e_tmp)
        self._e_tmp *= self._c2
        rhs -= self._e_tmp
        rhs *= dt / 2
        rhs += self._ME.dot(en, out=self._e_tmp)

        e_mid = self._solver.dot(rhs, out=self._e_mid)
        e_mid.update_ghost_regions()
        return e_mid, self._solver._info

    def _project_onto_constraint(self, dt: float):
        """For kappa = 0: replace (b, V) by the mid-point state of one step, which satisfies the constraint of the
        quasi-neutral limit exactly. The electric field is kept."""
        bn = self.variables.b.spline.vector
        e_mid, _ = self._mid_point_field(dt)

        # b_mid = b^n - dt/2 C e_mid, v_mid = v^n + dt/2 c3 a_p . Lambda^1 e_mid
        b_new = self._curl.dot(e_mid, out=self._b_new)
        b_new *= -dt / 2
        b_new += bn
        self._pusher(dt / 2)
        self.update_feec_variables(b=b_new)
        logger.info("DriftKineticMaxwellCoupling: projected the initial state onto the constraint (kappa = 0).")

    @profile
    def __call__(self, dt):
        en = self.variables.e.spline.vector
        bn = self.variables.b.spline.vector

        # Schur operator for the current time step (no-op unless dt changed since the setup)
        self.prepare_time_step(dt)

        # mid-point field
        e_mid, info = self._mid_point_field(dt)

        # new b coeffs: b^{n+1} = b^n - dt C e_mid
        b_new = self._curl.dot(e_mid, out=self._b_new)
        b_new *= -dt
        b_new += bn

        # new e coeffs: Crank-Nicolson (M_E = kappa M1 + chi M1gyro is nonzero, so e carries energy also for kappa = 0)
        e_new = e_mid.copy(out=self._e_new)
        e_new *= 2.0
        e_new -= en

        # update parallel velocities with e_mid
        self._pusher(dt)

        # update weights
        if self.variables.ions.species.weights_params.control_variate:
            self.variables.ions.particles.update_weights()

        # write new coeffs into self.variables
        diffs = self.update_feec_variables(e=e_new, b=b_new)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for DriftKineticMaxwellCoupling: {info['success']}")
            logger.info(f"Iterations for DriftKineticMaxwellCoupling: {info['niter']}")
            logger.info(f"Maxdiff e  for DriftKineticMaxwellCoupling: {diffs['e']}")
            logger.info(f"Maxdiff b  for DriftKineticMaxwellCoupling: {diffs['b']}")
            logger.info("")
