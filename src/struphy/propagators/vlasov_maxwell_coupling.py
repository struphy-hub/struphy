"Particle and FEEC variables are updated."

import logging
from dataclasses import dataclass

import cunumpy as xp
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation.kernels.marker_operator_symm_dot import marker_operator_symm_dot
from struphy.pic.accumulation.kernels.vlasov_maxwell import vlasov_maxwell
from struphy.pic.accumulation.kernels.vlasov_maxwell_mf import vlasov_maxwell_mf
from struphy.pic.accumulation.particles_to_grid import Accumulator, AccumulatorVector, MarkerOperator
from struphy.pic.pushing.kernels.push_v_with_efield import push_v_with_efield
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class VlasovMaxwellCoupling(Propagator):
    r"""PIC-FEEC discretization of the following equations:
    find :math:`\mathbf E \in H(\textnormal{curl})`, :math:`\mathbf B \in H(\textnormal{div})` and :math:`f` such that

    .. math::

        \kappa \int_{\Omega} \frac{\partial \mathbf E}{\partial t} \cdot \mathbf F \,\textrm d \mathbf x
        &= c_1 \int_{\Omega} \mathbf B \cdot \nabla \times \mathbf F \,\textrm d \mathbf x
        - c_2 \int_{\Omega} \int_{\mathbb{R}^3} f \mathbf{v} \cdot \mathbf{F} \, \text{d}^3 \mathbf{v} \,\textrm d \mathbf x
        \qquad \forall \, \mathbf F \in H(\textnormal{curl})\,,
        \\[2mm]
        \frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E &= 0\,,
        \\[2mm]
        \frac{\partial f}{\partial t} + c_3\, \mathbf{E} \cdot \frac{\partial f}{\partial \mathbf{v}} &= 0\,,

    with :math:`\kappa \geq 0`, :math:`c_1 > 0` and :math:`c_2 c_3 > 0`. It solves the steps of
    :class:`~struphy.propagators.maxwell_weak_ampere.MaxwellWeakAmpere` and
    :class:`~struphy.propagators.vlasov_ampere_coupling.VlasovAmpereCoupling` in one implicit step (for
    :math:`\kappa = 1`, :math:`c_1 = 1`, :math:`c_2 = \alpha^2/\varepsilon`, :math:`c_3 = 1/\varepsilon`).
    For :math:`\kappa = 0` the displacement current vanishes, :math:`c_1 \nabla \times \mathbf B = c_2 \mathbf J`
    (quasi-neutral limit).

    With the FE coefficients :math:`\mathbf e \in \mathbb R^{N_1}` of :math:`\mathbf E \in V^1_h` and
    :math:`\mathbf b \in \mathbb R^{N_2}` of :math:`\mathbf B \in V^2_h`, and :math:`f` approximated by :math:`N`
    markers with positions :math:`\boldsymbol \eta_p` (logical, fixed in this step), velocities
    :math:`\mathbf V = (\mathbf v_p)_p \in \mathbb R^{3N}` and weights :math:`\mathbb W = \textnormal{diag}(w_p)`,
    the semi-discrete system reads

    .. math::

        \kappa\, \mathbb M_1 \dot{\mathbf e} = c_1\, \mathbb C^\top \mathbb M_2 \mathbf b
        - c_2\, (\mathbf \Lambda^1)^\top \, \overline{DF^{-1}} \, \mathbb W \, \mathbf V \,,
        \qquad
        \dot{\mathbf b} = - \mathbb C \mathbf e\,,
        \qquad
        \dot{\mathbf V} = c_3\, \overline{DF^{-\top}} \, \mathbf \Lambda^1 \, \mathbf e\,,

    with the notation of :class:`~struphy.propagators.vlasov_ampere_coupling.VlasovAmpereCoupling`. It conserves the
    energy

    .. math::

        H = \frac{\kappa}{2}\, \mathbf e^\top \mathbb M_1 \mathbf e + \frac{c_1}{2}\, \mathbf b^\top \mathbb M_2 \mathbf b
        + \frac{c_2}{2 c_3}\, \mathbf V^\top \mathbb W \mathbf V
        \qquad (:meth:`~struphy.propagators.vlasov_maxwell_coupling.VlasovMaxwellCoupling.energy`)\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point), with
    :math:`\mathbf e^\textnormal{mid} = \frac 12 (\mathbf e^{n+1} + \mathbf e^n)` etc.,

    .. math::

        \kappa\, \mathbb M_1 \frac{\mathbf e^{n+1} - \mathbf e^n}{\Delta t}
        = c_1\, \mathbb C^\top \mathbb M_2 \mathbf b^\textnormal{mid}
        - c_2\, (\mathbf \Lambda^1)^\top \, \overline{DF^{-1}} \, \mathbb W \, \mathbf V^\textnormal{mid}\,,
        \qquad
        \frac{\mathbf b^{n+1} - \mathbf b^n}{\Delta t} = - \mathbb C \mathbf e^\textnormal{mid}\,,
        \qquad
        \frac{\mathbf V^{n+1} - \mathbf V^n}{\Delta t} = c_3\, \overline{DF^{-\top}} \, \mathbf \Lambda^1 \, \mathbf e^\textnormal{mid}\,,

    which conserves :math:`H` exactly (up to the solver tolerance; for constant weights, i.e. without control
    variate). Eliminating :math:`\mathbf b^\textnormal{mid}` and :math:`\mathbf V^\textnormal{mid}` gives the Schur
    system for the mid-point field

    .. math::

        \left(\kappa\, \mathbb M_1 + \frac{\Delta t^2}{4}\, \mathbb K\right) \mathbf e^\textnormal{mid}
        = \kappa\, \mathbb M_1 \mathbf e^n
        + \frac{\Delta t}{2} \left( c_1\, \mathbb C^\top \mathbb M_2 \mathbf b^n - c_2\, \mathbf j \right)\,,
        \qquad
        \mathbb K = c_1\, \mathbb C^\top \mathbb M_2 \mathbb C + c_2 c_3\, \mathbb A\,,

    with the accumulated matrix :math:`\mathbb A` and vector :math:`\mathbf j` of
    :class:`~struphy.propagators.vlasov_ampere_coupling.VlasovAmpereCoupling` (kernel
    :func:`~struphy.pic.accumulation.kernels.vlasov_maxwell.vlasov_maxwell_kernels.vlasov_maxwell`). It is solved
    with ``solver`` and ``precond``. Then

    .. math::

        \mathbf b^{n+1} = \mathbf b^n - \Delta t\, \mathbb C \mathbf e^\textnormal{mid}\,,
        \qquad
        \mathbf e^{n+1} = 2 \mathbf e^\textnormal{mid} - \mathbf e^n \quad (\kappa > 0)\,,

    and the velocities are updated by
    :func:`~struphy.pic.pushing.kernels.push_v_with_efield.push_v_with_efield_kernels.push_v_with_efield` with
    :math:`\mathbf e^\textnormal{mid}`.

    **Quasi-neutral limit** :math:`\kappa = 0`. The curl-curl operator :math:`\mathbb C^\top \mathbb M_2 \mathbb C`
    is singular on the discrete gradients (and harmonic fields); this kernel is absorbed by the plasma term
    :math:`c_2 c_3 \mathbb A`, a particle approximation of the density-weighted mass matrix :math:`\mathbb M_1`.
    The Schur operator is therefore positive definite for positive weights (full-f, no control variate) and plasma
    everywhere in the domain. The scheme then imposes
    :math:`c_1\, \mathbb C^\top \mathbb M_2 \mathbf b^\textnormal{mid} = c_2\, (\mathbf \Lambda^1)^\top
    \overline{DF^{-1}} \mathbb W \mathbf V^\textnormal{mid}`, hence a weakly divergence-free mid-point current
    (multiply by :math:`\mathbb G^\top`); the gradient part of :math:`\mathbf E` is its Lagrange multiplier.
    :math:`\mathbf e^n` drops out of the system and only :math:`\mathbf e^\textnormal{mid}` is determined: the
    propagator stores :math:`\mathbf e^{n+1} = \mathbf e^\textnormal{mid}` (the Crank-Nicolson update
    :math:`2 \mathbf e^\textnormal{mid} - \mathbf e^n` would carry any error of :math:`\mathbf e^n` as an undamped
    even-odd oscillation). The energy :math:`H` (without the electric part) is still conserved exactly.

    The constraint holds at the mid-point only: its residual
    :math:`\mathbf r^n = c_1 \mathbb C^\top \mathbb M_2 \mathbf b^n - c_2 \mathbf j(\mathbf V^n)` (linear in
    :math:`\mathbf V` at fixed positions) satisfies :math:`\mathbf r^{n+1} = - \mathbf r^n`. Initial data that violate
    the constraint (e.g. by marker noise in :math:`\mathbf J`) would thus oscillate with period 2 without damping. With
    ``initial_projection`` (default), the setup replaces :math:`(\mathbf b^0, \mathbf V^0)` by the mid-point state of
    one step, :math:`\mathbf b^0 - \frac{\Delta t}{2} \mathbb C \mathbf e^\textnormal{mid}` and
    :math:`\mathbf V^0 + \frac{\Delta t}{2} c_3 \overline{DF^{-\top}} \mathbf \Lambda^1 \mathbf e^\textnormal{mid}`
    (an implicit Euler step of :math:`\Delta t / 2`), which satisfies the constraint exactly and does not depend on
    :math:`\Delta t`: a projection onto the constraint. The particle pushers of the splitting change
    :math:`\mathbf r` again by :math:`O(\Delta t)` in every step.

    **Preconditioner.** With :math:`\mathbb A \approx \bar n\, \mathbb M_1`, the Schur operator is approximately the
    stabilized curl-curl operator
    :math:`\mathbb C^\top \left(\frac{\Delta t^2 c_1}{4} \mathbb M_2\right) \mathbb C
    + \left(\kappa + \frac{\Delta t^2 c_2 c_3 \bar n}{4}\right) \mathbb M_1`, preconditioned (default) by
    :class:`~struphy.feec.preconditioner.StiffnessPreconditioner`, built for the current time step
    (:meth:`~struphy.propagators.vlasov_maxwell_coupling.VlasovMaxwellCoupling.prepare_time_step`). Its
    stabilization stays positive for :math:`\kappa = 0`.
    """

    class Variables:
        """Container for variables advanced by :class:`~struphy.propagators.vlasov_maxwell_coupling.VlasovMaxwellCoupling`.

        Attributes
        ----------
        e : FEECVariable
            Electric-field variable in ``"Hcurl"`` space.
        b : FEECVariable
            Magnetic-field variable in ``"Hdiv"`` space.
        ions : PICVariable
            Particle variable in ``"Particles6D"`` space.
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
            assert new.space == "Particles6D"
            self._ions = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`~struphy.propagators.vlasov_maxwell_coupling.VlasovMaxwellCoupling`.

        Parameters
        ----------
        kappa : float, default=1.0
            Coefficient :math:`\kappa \geq 0` of the displacement current :math:`\partial_t \mathbf E`;
            ``kappa=0.0`` is the quasi-neutral limit (needs full-f weights, no control variate).

        c1 : float, default=1.0
            Coefficient :math:`c_1 > 0` of :math:`\nabla \times \mathbf B` in Ampère's law.

        c2 : float, default=None
            Coefficient of the current density in Ampère's law. If ``None``, :math:`\alpha^2/\varepsilon` of the
            species.

        c3 : float, default=None
            Coefficient of the electric force in the Vlasov equation (:math:`c_2 c_3 > 0`). If ``None``,
            :math:`1/\varepsilon` of the species.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver of the Schur system, passed to ``feectools.linalg.solvers.inverse``.

        precond : LiteralOptions.OptsMaxwellPrecond, default="StiffnessPreconditioner"
            Preconditioner of the Schur system.

            - ``"StiffnessPreconditioner"``: :class:`~struphy.feec.preconditioner.StiffnessPreconditioner` with
              ``derivative="curl"`` for the stabilized curl-curl approximation of the Schur operator (see the class
              docstring), rebuilt when the time step changes.
            - ``"MassMatrixPreconditioner"``, ``"PolarMassMatrixPreconditioner"``: preconditioner of
              :math:`\mathbb M_1`, sufficient for :math:`\Delta t \lesssim h` (polar splines are not supported by
              this propagator yet).
            - ``None``: no preconditioner (set automatically, with a warning, for ``solver="cg"``).

        solver_params : SolverParameters, default=None
            Solver controls. If ``None``, defaults to ``SolverParameters()``.

        n_ref : float, default=None
            Reference density :math:`\bar n` in the stiffness preconditioner. If ``None``, the Rayleigh quotient
            :math:`\mathbf 1^\top \mathbb A \mathbf 1 / \mathbf 1^\top \mathbb M_1 \mathbf 1` of the accumulated
            matrix at the setup (a density average).

        matrix_free : bool, default=False
            Apply the particle matrix :math:`\mathbb A` matrix-free, with the filling
            :math:`w_p G^{-1}(\boldsymbol \eta_p)` stored per marker
            (:class:`~struphy.pic.accumulation.particles_to_grid.MarkerOperator`), instead of assembling it: one spline
            evaluation and one deposition per marker and solver iteration instead of the products of all pairs of
            basis functions per marker. Needs 6 auxiliary marker columns. Ignored on the CuPy backend.

        initial_projection : bool, default=True
            Only for ``kappa=0``: in the setup (first :meth:`prepare_time_step`), replace :math:`(\mathbf b, \mathbf V)`
            by the mid-point state of one step, which satisfies :math:`c_1 \nabla \times \mathbf B = c_2 \mathbf J`
            exactly (see the class docstring). Without it, initial data that violate the constraint lead to an undamped
            period-2 oscillation of :math:`\mathbf b` and :math:`\mathbf V`. Not applied to a restored (restart) state.
        """

        kappa: float = 1.0
        c1: float = 1.0
        c2: float = None
        c3: float = None
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMaxwellPrecond = "StiffnessPreconditioner"
        solver_params: SolverParameters = None
        n_ref: float = None
        matrix_free: bool = False
        initial_projection: bool = True

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMaxwellPrecond)
            assert self.kappa >= 0.0, f"kappa must be >= 0, got {self.kappa}."
            assert self.c1 > 0.0, f"c1 must be > 0, got {self.c1}."
            if self.c2 is not None and self.c3 is not None:
                assert self.c2 * self.c3 > 0.0, f"c2 * c3 must be > 0, got c2={self.c2}, c3={self.c3}."
            if self.n_ref is not None:
                assert self.n_ref > 0.0, f"n_ref must be > 0, got {self.n_ref}."
            assert isinstance(self.matrix_free, bool)
            assert isinstance(self.initial_projection, bool)
            if self.solver == "cg" and self.precond is not None:
                logger.warning(f'solver="cg" takes no preconditioner; ignoring precond="{self.precond}".')
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
        self._kappa = self.options.kappa
        self._c1 = self.options.c1
        self._c2 = alpha**2 / epsilon if self.options.c2 is None else self.options.c2
        self._c3 = 1.0 / epsilon if self.options.c3 is None else self.options.c3
        assert self._c2 * self._c3 > 0.0, f"c2 * c3 must be > 0, got c2={self._c2}, c3={self._c3}."

        assert not self.derham.polar_splines, "VlasovMaxwellCoupling does not support polar splines yet."

        particles = self.variables.ions.particles
        if particles.control_variate:
            assert self._kappa > 0.0, (
                "The quasi-neutral limit kappa=0 needs full-f weights: with the control variate the accumulated "
                "matrix is indefinite and the Schur system is not well-posed."
            )
            logger.warning(
                "VlasovMaxwellCoupling with control variate: the energy is not conserved exactly (the weights change)."
            )

        self._info = self.options.solver_params.info

        # accumulation of the matrix A (assembled, or matrix-free) and the vector j;
        # the matrix-free kernels have no CUDA version
        self._matrix_free = self.options.matrix_free and xp.get_backend() != "cupy"
        if self._matrix_free:
            # the vector, and the filling of the matrix per marker (in the first free columns of the marker array)
            column = particles.args_markers.first_free_idx
            assert particles.n_cols_aux >= 6, "matrix_free needs 6 free (auxiliary) marker columns."
            self._args_accum = (column,)
            self._accum = AccumulatorVector(
                particles,
                "Hcurl",
                vlasov_maxwell_mf,
                self.mass_ops,
                self.domain.args_domain,
            )
            self._A_particles = MarkerOperator(
                particles,
                "Hcurl",
                marker_operator_symm_dot,
                self.mass_ops,
                self.domain.args_domain,
                (1, column),
                symmetry="symm",
            )
        else:
            self._args_accum = ()
            self._accum = Accumulator(
                particles,
                "Hcurl",
                vlasov_maxwell,
                self.mass_ops,
                self.domain.args_domain,
                add_vector=True,
                symmetry="symm",
            )
            self._A_particles = self._accum.operators[0]

        # operators
        self._M1 = self.mass_ops.M1
        self._M2 = self.mass_ops.M2
        self._curl = self.derham.curl
        self._curl_curl = self._curl.T @ self._M2 @ self._curl

        # reference density for the stiffness preconditioner: density average of the accumulated matrix
        if self.options.n_ref is None:
            self._accum(*self._args_accum)
            ones = self.derham.V1.zeros()
            for block in ones.blocks:
                block._data[:] = 1.0
            ones_A = ones.inner(self._A_particles.dot(ones))
            ones_M1 = ones.inner(self._M1.dot(ones))
            if ones_A > 0.0:
                self._n_ref = ones_A / ones_M1
            else:
                self._n_ref = 1.0
                logger.warning("VlasovMaxwellCoupling: no positive density average of the markers, using n_ref=1.")
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

        # particle pusher: v^{n+1} = v^n + dt * c3 * DF^{-T} Lambda^1 e_mid
        args_kernel = (
            self.derham.args_derham,
            self._e_mid.blocks[0]._data,
            self._e_mid.blocks[1]._data,
            self._e_mid.blocks[2]._data,
            self._c3,
        )

        self._pusher = Pusher(
            particles,
            push_v_with_efield,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
            pushes_eta=False,
        )

    def prepare_time_step(self, dt: float):
        """Schur operator :math:`\\kappa \\mathbb M_1 + \\frac{\\Delta t^2}{4} \\mathbb K`, its preconditioner and
        solver for the time step ``dt``: built in the setup, rebuilt when dt changes. The particle matrix enters by
        reference (assembled, or matrix-free reading the fillings stored in the markers), so it is updated by the
        accumulation in every step. For ``kappa=0`` (and ``initial_projection``), the first call also projects the
        state onto the constraint :math:`c_1 \\mathbb C^\\top \\mathbb M_2 \\mathbf b = c_2 \\mathbf j`."""
        if dt == self._schur_dt:
            return

        schur = (dt**2 * self._c1 / 4) * self._curl_curl + (dt**2 * self._c2 * self._c3 / 4) * self._A_particles
        if self._kappa > 0.0:
            schur = self._kappa * self._M1 + schur

        if self._precond == "StiffnessPreconditioner":
            pc = preconditioner.StiffnessPreconditioner(
                self.mass_ops,
                "curl",
                sigma=self._kappa + dt**2 * self._c2 * self._c3 * self._n_ref / 4,
                mid=(dt**2 * self._c1 / 4) * self._M2,
                stab="M1",
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
        r"""The energy :math:`H = \frac{\kappa}{2} \mathbf e^\top \mathbb M_1 \mathbf e + \frac{c_1}{2} \mathbf b^\top
        \mathbb M_2 \mathbf b + \frac{c_2}{2 c_3} \mathbf V^\top \mathbb W \mathbf V` conserved by the propagator
        (summed over all MPI ranks; a collective operation)."""
        e = self.variables.e.spline.vector
        b = self.variables.b.spline.vector
        particles = self.variables.ions.particles

        kinetic = xp.array([0.5 * xp.sum(particles.weights * xp.sum(particles.velocities**2, axis=0))])
        if self.derham.comm is not None:
            self.derham.comm.Allreduce(MPI.IN_PLACE, kinetic, op=MPI.SUM)
        if particles.clone_config is not None:
            particles.clone_config.inter_comm.Allreduce(MPI.IN_PLACE, kinetic, op=MPI.SUM)

        energy = self._c1 / 2 * b.inner(self._M2.dot(b, out=self._b_tmp)) + self._c2 / self._c3 * float(kinetic[0])
        if self._kappa > 0.0:
            energy += self._kappa / 2 * e.inner(self._M1.dot(e, out=self._e_tmp))
        return energy

    def _mid_point_field(self, dt: float):
        """Accumulate at the current markers and solve the Schur system for the mid-point field (into ``_e_mid``,
        with updated ghost regions, which the pusher kernel reads)."""
        en = self.variables.e.spline.vector
        bn = self.variables.b.spline.vector

        # accumulate A (or its fillings per marker) and j at the current marker positions and velocities
        self._accum(*self._args_accum)

        # rhs = kappa M1 e^n + dt/2 (c1 C^T M2 b^n - c2 j)
        rhs = self._curl.T.dot(self._M2.dot(bn, out=self._b_tmp), out=self._e_rhs)
        rhs *= self._c1
        j = self._accum.vectors[0]
        j.copy(out=self._e_tmp)
        self._e_tmp *= self._c2
        rhs -= self._e_tmp
        rhs *= dt / 2
        if self._kappa > 0.0:
            self._M1.dot(en, out=self._e_tmp)
            self._e_tmp *= self._kappa
            rhs += self._e_tmp

        e_mid = self._solver.dot(rhs, out=self._e_mid)
        e_mid.update_ghost_regions()
        return e_mid, self._solver._info

    def _project_onto_constraint(self, dt: float):
        """For kappa = 0: replace (b, V) by the mid-point state of one step, which satisfies
        c1 C^T M2 b = c2 j(V) exactly (independent of dt), and e by the mid-point field."""
        bn = self.variables.b.spline.vector
        e_mid, _ = self._mid_point_field(dt)

        # b_mid = b^n - dt/2 C e_mid, V_mid = V^n + dt/2 c3 DF^{-T} Lambda^1 e_mid
        b_new = self._curl.dot(e_mid, out=self._b_new)
        b_new *= -dt / 2
        b_new += bn
        self._pusher(dt / 2)
        self.update_feec_variables(e=e_mid.copy(out=self._e_new), b=b_new)
        logger.info("VlasovMaxwellCoupling: projected the initial state onto c1 curl B = c2 J (kappa = 0).")

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

        # new e coeffs: Crank-Nicolson for kappa > 0, the mid-point field for kappa = 0
        e_new = e_mid.copy(out=self._e_new)
        if self._kappa > 0.0:
            e_new *= 2.0
            e_new -= en

        # update velocities with e_mid
        self._pusher(dt)

        # update weights
        if self.variables.ions.species.weights_params.control_variate:
            self.variables.ions.particles.update_weights()

        # write new coeffs into self.variables
        diffs = self.update_feec_variables(e=e_new, b=b_new)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for VlasovMaxwellCoupling: {info['success']}")
            logger.info(f"Iterations for VlasovMaxwellCoupling: {info['niter']}")
            logger.info(f"Maxdiff e  for VlasovMaxwellCoupling: {diffs['e']}")
            logger.info(f"Maxdiff b  for VlasovMaxwellCoupling: {diffs['b']}")
            logger.info("")
