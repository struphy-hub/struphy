"""Per-ray adaptive time stepping for steady-state ion-optics ray tracing.

A steady-state ray trace follows every ray on its own; only the path integral of its current matters,
so rays do not have to share a clock. With a global time step, the slow ions in the plasma (Bohm
speed) and the fast ions in the accelerating gap share one step, which is set by the fastest and most
strongly accelerated part of the path. Here every ray takes the step its own state requires:

.. math::

    \\Delta t_i = \\min\\left(\\Delta t_\\mathrm{max},\\;
        C_x \\frac{h}{|\\mathbf v|},\\;
        C_v \\frac{|\\mathbf v|}{|\\mathbf a|},\\;
        C_a \\sqrt{\\frac{h}{|\\mathbf a|}}\\right),

with ``h`` the local mesh cell size (the field is only resolved down to ``h``), :math:`|\\mathbf a|` the
acceleration of the field at the ray's current position (one cheap kernel evaluation per step), and the constants of
:class:`StepControl`. Each step is a Strang step (half kick, drift, half kick) with that ray's step, using the
kernels of :mod:`struphy.pic.pushing.ray_kernels`, so it stays second order.

**Deposit.** With variable steps the charge of a ray is the trapezoid rule of its current over its own path:
each node gets ``I * (dt_n + dt_(n+1)) / 2``, with half of the first and the partial last step at the ends
(see :class:`~struphy.models.ion_optics_steady_state.SteadyStateIteration`).

This is for steady-state ray tracing only: time-dependent PIC needs all markers at the same time.
"""

from dataclasses import dataclass

import numpy as np
from cunumpy import PyccelKernel

from struphy.pic.pushing import ray_kernels
from struphy.pic.pushing.pusher import Pusher


@dataclass(frozen=True)
class StepControl:
    """Constants of the local step-size rule of :class:`AdaptiveRayPusher`.

    Parameters
    ----------
    courant : float
        At most ``courant`` mesh cells per step: ``dt <= courant * h / |v|``.

    velocity_change : float
        The velocity changes by at most this fraction of its magnitude per step: ``dt <= velocity_change * |v| / |a|``.
        This is the binding rule for slow ions that start to accelerate (meniscus, sheath, emission from rest).

    acceleration : float
        ``dt <= acceleration * sqrt(h / |a|)``: the displacement from acceleration alone stays below a cell.

    min_scale : float
        Lower bound of the step as a fraction of ``dt_max``.
    """

    courant: float = 0.5
    velocity_change: float = 0.3
    acceleration: float = 0.5
    min_scale: float = 1e-3

    def __post_init__(self):
        if min(self.courant, self.velocity_change, self.acceleration) <= 0.0:
            raise ValueError("courant, velocity_change and acceleration must be positive.")
        if not 0.0 < self.min_scale <= 1.0:
            raise ValueError("min_scale must lie in (0, 1].")


class CellSizeMap:
    """Physical size of the mesh cell at a logical position.

    The size is the smallest extent of the cell over the directions that carry structure (directions with a single
    element of degree one, such as the invariant direction of a slit or the angle of a wedge, are skipped). It is
    sampled once on a grid of logical points and looked up by nearest neighbour.

    Parameters
    ----------
    domain : Domain
        Mapping from the logical cube to physical space.

    num_elements, degree : tuple[int, int, int]
        Mesh elements and spline degrees of the Derham sequence.

    samples_per_element : int
        Sampling resolution of the map.
    """

    def __init__(self, domain, num_elements, degree, samples_per_element=2):
        structured = [not (n == 1 and p == 1) for n, p in zip(num_elements, degree)]
        if not any(structured):
            raise ValueError("The mesh has no direction with structure.")
        shape = tuple(samples_per_element * n if s else 1 for n, s in zip(num_elements, structured))
        axes = [(np.arange(m) + 0.5) / m for m in shape]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        jacobian = np.asarray(domain.jacobian(grid, change_out_order=True, remove_outside=False))
        lengths = np.linalg.norm(jacobian, axis=1)  # (N, 3): physical length per unit eta_d
        sizes = np.where(structured, lengths / np.asarray(num_elements, dtype=float), np.inf)
        self._shape = shape
        self._size = sizes.min(axis=1).reshape(shape)

    def lookup(self, eta):
        """Cell sizes at logical positions ``eta`` of shape ``(n, 3)``."""
        index = tuple(np.clip((np.nan_to_num(eta[:, d]) * m).astype(int), 0, m - 1) for d, m in enumerate(self._shape))
        return self._size[index]


class AdaptiveRayPusher:
    """Strang steps with a per-marker time step for the rays of a steady-state trace.

    Parameters
    ----------
    particles : Particles
        Marker array of the rays (serial; rows must not be reordered, so no sorting).

    domain, derham : Domain, Derham
        Mapping and FEEC sequence.

    e_field : BlockVector
        Coefficients of the electric field 1-form (``E = -grad phi``); the arrays are shared, so updating the vector
        in place updates the push.

    epsilon : float
        Species scaling: ``dv/dt = E / epsilon``.

    butcher : ButcherTableau
        Runge–Kutta tableau of the drift.

    dt_max : float
        Largest step.

    control : StepControl
        Constants of the step rule.

    cells : CellSizeMap
        Local cell sizes.
    """

    def __init__(self, particles, domain, derham, e_field, epsilon, butcher, dt_max, control, cells):
        self.particles = particles
        self.dt_max = float(dt_max)
        self.control = control
        self.cells = cells
        n_rows = particles.markers.shape[0]
        self.scale = np.ones(n_rows)
        self._acceleration = np.zeros(n_rows)
        self._acceleration_args = (
            domain.args_domain,
            derham.args_derham,
            e_field[0]._data,
            e_field[1]._data,
            e_field[2]._data,
            1.0 / epsilon,
            self._acceleration,
        )
        self._acceleration_kernel = PyccelKernel(ray_kernels.acceleration_magnitude)
        self._push_v = Pusher(
            particles,
            PyccelKernel(ray_kernels.push_v_with_efield_scaled),
            (derham.args_derham, e_field[0]._data, e_field[1]._data, e_field[2]._data, 1.0 / epsilon, self.scale),
            domain.args_domain,
            alpha_in_kernel=1.0,
        )
        self._push_eta = Pusher(
            particles,
            PyccelKernel(ray_kernels.push_eta_stage_scaled),
            (butcher.a_stage, butcher.b, butcher.c, self.scale),
            domain.args_domain,
            alpha_in_kernel=1.0,
            n_stages=butcher.n_stages,
            mpi_sort="each",
        )

    def choose(self):
        """Set the step of every ray for its next step from its current state; returns the steps ``dt_i`` (all rows).

        Call once after launching and once after every push. The acceleration is the field's at the ray's current
        position, so a ray about to enter a strong field is already slowed down before it does.
        """
        markers, control = self.particles.markers, self.control
        speed = np.linalg.norm(markers[:, 3:6], axis=1)
        h = self.cells.lookup(markers[:, :3])
        self._acceleration_kernel(self.particles.args_markers, *self._acceleration_args)
        accel = self._acceleration
        tiny = 1e-300
        with np.errstate(divide="ignore", invalid="ignore"):
            limits = [
                control.courant * h / np.maximum(speed, tiny),
                control.velocity_change * speed / np.maximum(accel, tiny),
                control.acceleration * np.sqrt(h / np.maximum(accel, tiny)),
            ]
            dt = np.minimum.reduce(limits)
        dt = np.where(np.isfinite(dt), dt, self.dt_max)
        dt = np.clip(dt, control.min_scale * self.dt_max, self.dt_max)
        self.scale[:] = dt / self.dt_max
        return dt

    def step(self):
        """One Strang step (half kick, drift, half kick) of every ray with its own step."""
        self._push_v(0.5 * self.dt_max)
        self._push_eta(self.dt_max)
        self._push_v(0.5 * self.dt_max)


class FixedStepRayTracer:
    """Fused fixed-step Strang stepping and charge deposit of all rays (serial).

    One call of :func:`~struphy.pic.pushing.ray_kernels.trace_step` per step replaces the two ``PushVinForceField``
    half kicks, the four ``PushEta`` stage calls with their boundary checks, and the per-step ``AccumulatorVector``
    deposit. The results are those of the separate propagators (the same kernels' arithmetic, and the field of the
    second half kick is cached for the first half kick of the next step), see the regression test in
    ``struphy.models.tests.test_ion_optics_electrostatic``. Measured on the slit extraction (900 rays, 48 x 18
    elements): about 4x less time per round.

    Parameters
    ----------
    particles : Particles
        Marker array of the rays (serial, rows are not reordered).

    domain, derham : Domain, Derham
        Mapping and FEEC sequence.

    e_field : BlockVector
        Coefficients of the electric field 1-form (``E = -grad phi``); the arrays are shared, so updating the vector
        in place updates the push.

    epsilon : float
        Species scaling: ``dv/dt = E / epsilon``.

    butcher : ButcherTableau
        Explicit Runge–Kutta tableau of the drift.

    dt : float
        Time step.
    """

    def __init__(self, particles, domain, derham, e_field, epsilon, butcher, dt):
        self.particles = particles
        self.dt = float(dt)
        n_rows = particles.markers.shape[0]
        self.e_cache = np.zeros((n_rows, 3))
        self.state = np.zeros(n_rows, dtype=int)
        periodic = np.array([1 if bc == "periodic" else 0 for bc in particles.bc], dtype=int)
        self._kernel = PyccelKernel(ray_kernels.trace_step)
        self._args = (
            domain.args_domain,
            derham.args_derham,
            e_field[0]._data,
            e_field[1]._data,
            e_field[2]._data,
            1.0 / epsilon,
            np.asarray(butcher.a_stage, dtype=float),
            np.asarray(butcher.b, dtype=float),
            periodic,
            self.e_cache,
            self.state,
        )
        self._no_vec = np.zeros((1, 1, 1))

    def reset(self):
        """Forget the cached fields; call after (re)launching the rays."""
        self.state[:] = 0

    def step(self, charge=None):
        """One Strang step of every ray, with the kinetic boundary conditions applied.

        If ``charge`` (a StencilVector of V0) is given, every ray that is still inside after the step deposits
        ``weight * dt`` at its new position into ``charge._data`` (raw, without the assembly exchange: call
        ``charge.exchange_assembly_data()`` and ``charge.update_ghost_regions()`` once at the end of the trace).
        """
        vec = self._no_vec if charge is None else charge._data
        deposit = charge is not None
        particles = self.particles
        self._kernel(self.dt, particles.args_markers, *self._args, vec, deposit, 0)
        particles.apply_kinetic_bc()
        particles.update_holes()
        if np.any((self.state == 2) & particles.valid_mks):
            # rays reflected by the boundary conditions: second half kick and deposit at the reflected position
            self._kernel(self.dt, particles.args_markers, *self._args, vec, deposit, 1)
