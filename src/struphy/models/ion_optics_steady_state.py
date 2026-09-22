"""Steady-state Vlasov–Poisson iteration for :class:`IonOpticsElectrostatic`.

This is the IBSimu iteration cycle (Kalvas 2013, §5.1 and §5.6–5.8) built from the
model's pieces:

1. Start from the vacuum potential of the model's electrode solve.
2. Trace a fixed set of rays, each carrying a current ``I``, through the frozen field
   until all have left the domain. After each step, deposit every ray's charge
   ``I * dt`` at its position (``charge_density_0form``, a Galerkin deposit that
   needs no boundary-node correction). This is the trajectory deposition
   ``Q = I Δt`` of the thesis.
3. Under-relax the charge, ``rho_k = alpha rho* + (1 - alpha) rho_{k-1}``, and
   solve Poisson's equation with the electrode constraints.
4. Repeat until the rms emittance at the exit satisfies the stopping rule
   :func:`~struphy.diagnostics.beam_diagnostics.emittance_converged`.

Unlike time-dependent PIC (``space_charge=True``), this cannot model emission above
the space-charge limit: reflected beams make the iteration oscillate.
"""

from dataclasses import dataclass, field

import numpy as np
from cunumpy import PyccelKernel
from feectools.ddm.mpi import mpi as MPI
from scipy.special import ndtri
from scipy.stats import qmc

from feectools.linalg.utilities import array_to_psydac
from scope_profiler.profile_manager import ProfileManager

from struphy.diagnostics.beam_diagnostics import emittance_converged, rms_moments
from struphy.pic.accumulation import accum_kernels
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.pic.ion_beams import CurrentLedger, PlaneSource
from struphy.pic.ray_tracing import AdaptiveRayPusher, CellSizeMap, StepControl
from struphy.propagators.base import Propagator


@dataclass(frozen=True)
class SteadyStateOptions:
    """Configuration for the steady-state ion-optics solve run by ``Simulation.run``.

    Provide either ``source`` and ``n_rays`` (sampled once into a deterministic
    Sobol ray bundle), or an explicit ``rays`` bundle. The same rays are retraced
    in every Vlasov--Poisson round, so this is a steady-state solve rather than
    time-dependent particle injection.
    """

    dt: float
    source: PlaneSource | None = None
    n_rays: int | None = None
    rays: "RayBundle | None" = None
    alpha: float = 1.0
    loss_tags: tuple = ()
    exit_tag: str = "outlet"
    n_average: int = 5
    tol: float = 1e-3
    max_rounds: int = 50
    max_steps: int = 100000
    n_tracked: int = 0
    planes: tuple = None
    criterion: str = "emittance"
    anderson: int = 0
    verbose: bool = False
    seed: int = 0
    relaxation: str = "constant"
    step_control: StepControl | None = None

    def __post_init__(self):
        if (self.source is None) == (self.rays is None):
            raise ValueError("Provide exactly one of source or rays.")
        if self.source is not None:
            if not isinstance(self.source, PlaneSource):
                raise TypeError("steady-state source must be a PlaneSource.")
            if self.n_rays is None or self.n_rays <= 0:
                raise ValueError("source-based steady-state options need positive n_rays.")
        elif not isinstance(self.rays, RayBundle):
            raise TypeError("steady-state rays must be a RayBundle.")
        elif self.n_rays is not None and self.n_rays != len(self.rays):
            raise ValueError("n_rays must equal len(rays) when an explicit ray bundle is used.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")


def build_steady_state_simulation(output_dir, name, model, n_rays, domain, num_elements, degree, boundary_conditions):
    """Construct a simulation prepared for a steady ray-tracing iteration.

    The particle allocation is deliberately separate from
    :class:`SteadyStateOptions`: it describes storage and boundary handling,
    whereas the options describe the physical ray-tracing solve.  This helper
    lets examples and user scripts create an independent steady-state setup
    without importing implementation details from another example.
    """
    from struphy import (
        BoundaryParameters,
        DerhamOptions,
        EnvironmentOptions,
        LoadingParameters,
        SavingParameters,
        ProfilingOptions,
        Simulation,
        Time,
        WeightsParameters,
        grids,
        maxwellians,
    )

    model.ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    model.ions.set_markers(
        loading_params=LoadingParameters(Np=n_rays),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(bc=boundary_conditions),
        saving_params=SavingParameters(n_markers=1),
    )
    for propagator in model.prop_list:
        propagator.options = propagator.Options()
    return Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(output_dir), sim_folder=name),
        time_opts=Time(dt=0.01, Tend=0.01),  # the steady iteration drives the propagators itself
        domain=domain,
        grid=grids.TensorProductGrid(num_elements=num_elements),
        derham_opts=DerhamOptions(degree=degree, bcs=(("free", "free"),) * 3),
        profiling_opts=ProfilingOptions(),
    )


@dataclass
class RayBundle:
    """Rays launched every iteration round: logical positions, physical velocities, current per ray."""

    eta: np.ndarray
    v: np.ndarray
    current: np.ndarray

    def __post_init__(self):
        self.eta = np.atleast_2d(np.asarray(self.eta, dtype=float))
        self.v = np.atleast_2d(np.asarray(self.v, dtype=float))
        self.current = np.broadcast_to(np.asarray(self.current, dtype=float), (len(self.eta),)).copy()
        if self.eta.shape != self.v.shape or self.eta.shape[1] != 3:
            raise ValueError("eta and v must both have shape (n_rays, 3).")

    def __len__(self):
        return len(self.eta)

    @classmethod
    def from_plane_source(cls, source: PlaneSource, n_rays: int, seed: int = 0):
        """Quasi-random (scrambled Sobol) rays with the distribution of a ``PlaneSource``.

        The total current ``source.current`` is shared equally by the rays.
        """
        sampler = qmc.Sobol(d=5, scramble=True, seed=seed)
        u = sampler.random(n_rays)
        eta = np.empty((n_rays, 3))
        eta[:, source.axis] = source.eta_plane
        for k, ((low, high), other) in enumerate(zip(source.eta_ranges, [a for a in range(3) if a != source.axis])):
            eta[:, other] = low + (high - low) * u[:, k]
        normal = ndtri(np.clip(u[:, 2:], 1e-12, 1 - 1e-12))
        v = source.velocity + source.velocity_spread * normal
        return cls(eta=eta, v=v, current=source.current / n_rays)

    @classmethod
    def axisymmetric_disk(
        cls,
        domain,
        n_rays: int,
        z0: float,
        radius: float,
        current_density: float,
        axial_speed: float,
        transverse_spread: float = 0.0,
        seed: int = 0,
    ):
        """Quasi-random rays of a uniform disk beam in an :class:`~struphy.geometry.axisymmetric.AxisymmetricElectrodeChannel`.

        Radii follow ``r = sqrt(u) * radius`` (uniform current density), so every ray carries
        the same current ``current_density * pi * radius**2 / (n_rays * tor_period)``; only the
        wedge fraction of each ring is traced. Transverse velocities are isotropic Gaussians
        with standard deviation ``transverse_spread`` per Cartesian component.
        """
        sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
        u = sampler.random(n_rays)
        r = np.maximum(np.sqrt(u[:, 0]) * radius, 1.001 * domain.axis_radius)
        wall = float(domain.wall_radius(z0))
        eta = np.column_stack(
            [
                np.full(n_rays, z0 / domain.length),
                (r - domain.axis_radius) / (wall - domain.axis_radius),
                u[:, 1],
            ]
        )
        # Cartesian velocity: axial along z, transverse isotropic in (x, y)
        normal = ndtri(np.clip(u[:, 2:], 1e-12, 1 - 1e-12))
        v = np.column_stack(
            [transverse_spread * normal[:, 0], transverse_spread * normal[:, 1], np.full(n_rays, axial_speed)]
        )
        current = current_density * np.pi * radius**2 / (n_rays * domain.tor_period)
        return cls(eta=eta, v=v, current=current)


class AdaptiveRelaxation:
    r"""Adaptive under-relaxation ``rho <- (1 - alpha) rho + alpha F(rho)`` of a fixed-point iteration.

    A constant ``alpha`` must be small enough for the fastest, alternating mode of the map (the ion
    charge and the potential respond with opposite sign, so ``alpha = 0.5`` gives a limit cycle in the
    extraction problems here), but that makes slow monotone modes crawl. The factor is therefore adapted from
    the residuals :math:`r_k = \rho^*_k - \rho_{k-1}`, without further ray tracing:

    * **Direction rule.** If successive residuals agree (cosine above ``agree``), the iteration
      creeps along a slow mode: ``alpha`` grows by ``grow``. If they alternate (cosine below zero),
      it overshoots: ``alpha`` shrinks by ``shrink``. In the smooth regime ``alpha`` climbs to its cap
      of one, and the undamped iteration converges quickly near the solution.
    * **Noise safeguard.** With few rays the charge map is not smooth (a ray flips between
      "extracted" and "intercepted"), the residual stops decreasing, and damping is what averages
      the noise. If the relative residual sets no new minimum for ``patience`` rounds, the cap on
      ``alpha`` is halved (down to ``alpha_min``).

    Measured on the slit extraction (48 x 18 elements): at 900 rays the residual falls to 1e-6; at 300
    rays, where a constant ``alpha = 0.5`` never converges, the potential change reaches 1.5e-4 after
    about 50 rounds.
    """

    def __init__(self, alpha=0.2, alpha_min=0.05, grow=1.25, shrink=0.5, agree=0.5, patience=10):
        self.alpha = float(alpha)
        self.cap = 1.0
        self.alpha_min, self.grow, self.shrink, self.agree, self.patience = alpha_min, grow, shrink, agree, patience
        self.residual = np.nan
        self._previous = None
        self._best = np.inf
        self._since_best = 0

    def update(self, residual, scale):
        """New ``alpha`` from the residual vector ``rho* - rho`` and the norm ``scale`` of ``rho*``."""
        norm = float(np.linalg.norm(residual))
        self.residual = norm / max(scale, 1e-300)
        if self._previous is not None:
            cosine = float(residual @ self._previous) / max(norm * float(np.linalg.norm(self._previous)), 1e-300)
            if cosine > self.agree:
                self.alpha = min(self.cap, self.grow * self.alpha)
            elif cosine < 0.0:
                self.alpha = max(self.alpha_min, self.shrink * self.alpha)
        self._previous = np.array(residual, copy=True)
        if self.residual < self._best:
            self._best, self._since_best = self.residual, 0
        else:
            self._since_best += 1
            if self._since_best >= self.patience:
                self.cap = max(self.alpha_min, 0.5 * self.cap)
                self.alpha = min(self.alpha, self.cap)
                self._since_best = 0
        return self.alpha


@dataclass
class IterationRecord:
    """Diagnostics of one iteration round."""

    round: int
    exit_emittance: float
    exit_size: float
    exit_current: float
    lost_current: dict
    steps: int
    exit_records: np.ndarray = field(repr=False)
    plane_crossings: np.ndarray = field(default=None, repr=False)
    potential_change: float = np.nan
    alpha: float = np.nan  # relaxation factor used when mixing this round's charge
    residual: float = np.nan  # fixed-point residual ||rho* - rho|| / ||rho*|| of this round


class SteadyStateIteration:
    """Vlasov–Poisson iteration of an allocated :class:`IonOpticsElectrostatic` simulation.

    Parameters
    ----------
    sim : Simulation
        Simulation of an ``IonOpticsElectrostatic`` model with electrodes, not run.
        ``set_markers`` must provide room for all rays (``Np >= len(rays)``).

    rays : RayBundle
        Rays launched in every round.

    dt : float
        Time step of the ray tracing (normalized). With ``step_control`` it is the largest step.

    alpha : float
        Space-charge under-relaxation factor, 0 < alpha <= 1.

    loss_tags : tuple[LossTag]
        Boundary parts for current accounting; ``exit_tag`` must be one of them.

    exit_tag : str
        Boundary part whose crossing rays define the exit emittance.

    n_average, tol : int, float
        Parameters of :func:`~struphy.diagnostics.beam_diagnostics.emittance_converged`.

    anderson : int
        Depth of Anderson acceleration of the fixed-point map ``rho -> rho*`` (0: plain
        under-relaxation with ``alpha``). Anderson mixing uses the last ``anderson``
        residuals ``rho* - rho`` to extrapolate the charge, and ``alpha`` then acts as the
        mixing parameter. Experimental: in the Bohm-sheath test it reaches 1e-2 in half
        the rounds, but then stalls around 1e-4, while plain α = 1 continues to converge.

    step_control : StepControl, optional
        If given, every ray takes its own step (:class:`~struphy.pic.ray_tracing.AdaptiveRayPusher`), chosen from
        its speed, acceleration and the local cell size, and ``dt`` is the largest step. The charge deposit then uses
        the trapezoid rule of each ray's own path. Default ``None``: the same fixed step ``dt`` for all rays.

    relaxation : str
        ``"constant"`` (``alpha`` throughout) or ``"adaptive"`` (:class:`AdaptiveRelaxation`, with ``alpha``
        as the initial value). Adaptive damping is the robust choice for plasma extraction.

    criterion : str
        ``"emittance"`` (Kalvas 2013, §5.8.2), ``"potential"`` or ``"residual"``. ``"potential"``: the relative
        change of the potential coefficients, ``||phi_k - phi_{k-1}|| / ||phi_k||``, is below ``tol``
        for two consecutive rounds (it scales with ``alpha``, so a tiny ``alpha`` can look converged).
        ``"residual"``: the undamped fixed-point residual ``||rho* - rho|| / ||rho*||`` is below ``tol`` for two
        rounds (independent of ``alpha``; with few rays it stays at a noise floor). Use ``"potential"`` or
        ``"residual"`` when the exit emittance carries no information, e.g. in 1D.

    max_rounds, max_steps : int
        Limits on iteration rounds and on time steps per round.

    n_tracked : int
        Number of rays (the first ones) whose trajectories are stored in the last round.

    verbose : bool
        Print a summary of every round.

    planes : tuple[int, sequence[float]], optional
        Diagnostic planes ``(axis, eta_values)``: logical planes ``eta[axis] = const``.
        In every round, the first crossing of each ray is stored in
        ``IterationRecord.plane_crossings``, an array of shape ``(n_planes, n_rays, 6)``
        holding the logical position and the velocity (NaN if a ray never crosses).
    """

    def __init__(
        self,
        sim,
        rays: RayBundle,
        dt: float,
        alpha: float = 1.0,
        loss_tags: tuple = (),
        exit_tag: str = "outlet",
        n_average: int = 5,
        tol: float = 1e-3,
        max_rounds: int = 50,
        max_steps: int = 100000,
        n_tracked: int = 0,
        planes: tuple = None,
        criterion: str = "emittance",
        anderson: int = 0,
        verbose: bool = False,
        relaxation: str = "constant",
        step_control: StepControl | None = None,
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1].")
        if exit_tag not in [tag.name for tag in loss_tags]:
            raise ValueError("exit_tag must be the name of one of the loss_tags.")
        if MPI.COMM_WORLD.Get_size() != 1:
            raise NotImplementedError("SteadyStateIteration currently supports serial runs only.")
        self.sim = sim
        self.rays = rays
        self.dt = dt
        self.alpha = alpha
        self.loss_tags = tuple(loss_tags)
        self.exit_tag = exit_tag
        self.n_average = n_average
        self.tol = tol
        self.max_rounds = max_rounds
        self.max_steps = max_steps
        self.n_tracked = n_tracked
        if criterion not in ("emittance", "potential", "residual"):
            raise ValueError("criterion must be 'emittance', 'potential' or 'residual'.")
        self.criterion = criterion
        if relaxation not in ("constant", "adaptive"):
            raise ValueError("relaxation must be 'constant' or 'adaptive'.")
        self.relaxation = relaxation
        if step_control is not None and not isinstance(step_control, StepControl):
            raise TypeError("step_control must be a StepControl.")
        self.step_control = step_control
        self._ray_pusher = None
        self._adaptive = AdaptiveRelaxation(alpha) if relaxation == "adaptive" else None
        self._last_alpha, self._last_residual = 1.0, np.nan
        self.anderson = int(anderson)
        self.verbose = verbose
        self._mixing_history = []  # (rho_in, residual) pairs for Anderson acceleration
        self.history: list[IterationRecord] = []
        self.converged = False
        self.charge = None
        self.trajectories = None
        if planes is None:
            self.planes = None
        else:
            axis, values = planes
            self.planes = (int(axis), np.asarray(values, dtype=float))

    @property
    def model(self):
        return self.sim.model

    def run(self):
        """Iterate until converged or ``max_rounds``; returns the list of round records."""
        model = self.model
        if not hasattr(model, "_poisson"):
            self.sim.allocate()
        particles = model.ions.var.particles
        if particles.n_rows < len(self.rays):
            raise ValueError(f"The marker array has {particles.n_rows} rows for {len(self.rays)} rays.")
        self._accumulator = AccumulatorVector(
            particles,
            "H1",
            PyccelKernel(accum_kernels.charge_density_0form),
            Propagator.mass_ops,
            Propagator.domain.args_domain,
        )
        self._build_ray_pusher()
        for k in range(self.max_rounds):
            with ProfileManager.profile_region("ion optics: trace round"):
                record, charge = self._trace_round(k)
            self.history.append(record)
            with ProfileManager.profile_region("ion optics: mix charge"):
                self.charge = self._mix(charge)
            record.alpha, record.residual = self._last_alpha, self._last_residual
            if self._converged():
                self.converged = True
                break
            previous = model.em_fields.phi.spline.vector.toarray()
            with ProfileManager.profile_region("ion optics: field solve"):
                model.solve_potential(self.charge)
            current = model.em_fields.phi.spline.vector.toarray()
            record.potential_change = float(np.linalg.norm(current - previous) / max(np.linalg.norm(current), 1e-300))
            if self.verbose:
                print(
                    f"round {k:3d}: {record.steps} steps, exit current {record.exit_current:.4f}, "
                    f"exit emittance {record.exit_emittance:.4e}, |dphi|/|phi| {record.potential_change:.2e}, "
                    f"alpha {record.alpha:.2f}, residual {record.residual:.2e}",
                    flush=True,
                )
            Propagator.derham.grad.dot(-model.em_fields.phi.spline.vector, out=model.em_fields.e_field.spline.vector)
            model.em_fields.e_field.spline.vector.update_ghost_regions()
        # final round with trajectories stored, in the converged (or last) field
        if self.n_tracked > 0:
            self._trace_round(len(self.history), track=True)
        return self.history

    def _mix(self, charge):
        """Next input charge: under-relaxation (constant or adaptive), or Anderson acceleration of depth ``anderson``."""
        if self.charge is None:
            self._last_alpha, self._last_residual = 1.0, np.nan
            return charge
        new, old = charge.toarray(), self.charge.toarray()
        if self._adaptive is not None:
            alpha = self._adaptive.update(new - old, float(np.linalg.norm(new)))
            self._last_alpha, self._last_residual = alpha, self._adaptive.residual
            return alpha * charge + (1.0 - alpha) * self.charge
        self._last_alpha = self.alpha
        self._last_residual = float(np.linalg.norm(new - old) / max(np.linalg.norm(new), 1e-300))
        if self.anderson <= 0:
            return self.alpha * charge + (1.0 - self.alpha) * self.charge
        rho_in = self.charge.toarray()
        residual = charge.toarray() - rho_in
        self._mixing_history = (self._mixing_history + [(rho_in, residual)])[-(self.anderson + 1) :]
        mixed = rho_in + self.alpha * residual
        if len(self._mixing_history) > 1:
            inputs = np.array([h[0] for h in self._mixing_history])
            residuals = np.array([h[1] for h in self._mixing_history])
            d_inputs = np.diff(inputs, axis=0).T
            d_residuals = np.diff(residuals, axis=0).T
            gamma = np.linalg.lstsq(d_residuals, residual, rcond=None)[0]
            mixed = mixed - (d_inputs + self.alpha * d_residuals) @ gamma
        return array_to_psydac(mixed, self.charge.space)

    def averaged(self, rounds=10):
        """Mean and standard deviation over the last ``rounds`` rounds of the key results.

        With few rays the charge map is noisy and the iteration only settles to a scatter around the
        solution (see :class:`AdaptiveRelaxation`); the mean over the last rounds is then the estimate and the
        standard deviation its uncertainty. Returns a dict ``name -> (mean, std)`` with the current to each
        boundary part (fractions of the emitted current, ``lost_current``), ``exit_current``, and
        ``exit_emittance``.
        """
        recent = self.history[-rounds:]
        if not recent:
            raise ValueError("No rounds have been run.")
        series = {f"lost_current[{name}]": [r.lost_current[name] for r in recent] for name in recent[0].lost_current}
        series["exit_current"] = [r.exit_current for r in recent]
        series["exit_emittance"] = [r.exit_emittance for r in recent]
        return {
            name: (float(np.nanmean(values)), float(np.nanstd(values)))
            if np.any(np.isfinite(values))
            else (np.nan, np.nan)
            for name, values in series.items()
        }

    def _converged(self):
        if self.criterion == "emittance":
            return emittance_converged([r.exit_emittance for r in self.history], self.n_average, self.tol)
        if self.criterion == "residual":
            residuals = [r.residual for r in self.history[1:]]  # the first round has no residual
            return len(residuals) >= 2 and residuals[-1] < self.tol and residuals[-2] < self.tol
        # potential change of the solves after the last two traced rounds
        changes = [r.potential_change for r in self.history[:-1]]
        return len(changes) >= 2 and changes[-1] < self.tol and changes[-2] < self.tol

    def _build_ray_pusher(self):
        """Create the per-ray adaptive pusher if ``step_control`` is set (no-op otherwise)."""
        self._ray_pusher = None
        if self.step_control is None:
            return
        model, derham = self.model, Propagator.derham
        self._ray_pusher = AdaptiveRayPusher(
            model.ions.var.particles,
            Propagator.domain,
            derham,
            model.em_fields.e_field.spline.vector,
            model.ions.equation_params.epsilon,
            model.propagators.push_eta.options.butcher,
            self.dt,
            self.step_control,
            CellSizeMap(Propagator.domain, derham.grid.num_elements, derham.degree),
        )

    def _launch(self, particles):
        n = len(self.rays)
        particles.markers[:, :] = -1.0
        index = particles.index
        particles.markers[:n, :] = 0.0
        particles.markers[:n, index["pos"]] = self.rays.eta
        particles.markers[:n, index["vel"]] = self.rays.v
        particles.markers[:n, index["weights"]] = self.rays.current
        particles.markers[:n, index["s0"]] = 1.0
        particles.markers[:n, index["w0"]] = self.rays.current
        particles.markers[:n, -1] = np.arange(n)
        particles.update_holes()
        particles.pop_lost_markers()

    @ProfileManager.profile("ion optics: deposit")
    def _deposit(self, charge, weight):
        self._accumulator()
        charge += self._accumulator.vectors[0] * weight

    @ProfileManager.profile("ion optics: deposit")
    def _deposit_weighted(self, charge, step):
        """Deposit the charge ``I * step[i]`` of every ray at its current position (per-ray steps)."""
        particles = self.model.ions.var.particles
        column, n = particles.index["weights"], len(self.rays)
        saved = particles.markers[:n, column].copy()
        particles.markers[:n, column] = self.rays.current * step[:n]
        self._accumulator()
        charge += self._accumulator.vectors[0]
        particles.markers[:n, column] = saved

    def _trace_round(self, k, track=False):
        model = self.model
        particles = model.ions.var.particles
        push_v, push_eta = model.propagators.push_v, model.propagators.push_eta
        dt = self.dt
        pusher = self._ray_pusher
        self._launch(particles)
        ledger = CurrentLedger(self.loss_tags, keep_records=(self.exit_tag,))
        charge = model.em_fields.phi.spline.vector.space.zeros()
        # trapezoidal rule in time: half weight at launch
        if pusher is None:
            self._deposit(charge, 0.5 * dt)
        else:
            step_dt = pusher.choose()
            self._deposit_weighted(charge, 0.5 * step_dt)
        paths = [] if track else None
        crossings = None if self.planes is None else np.full((len(self.planes[1]), len(self.rays), 6), np.nan)
        previous = particles.markers[: len(self.rays), :6].copy()
        steps = 0
        while particles.n_mks_loc > 0:
            if steps >= self.max_steps:
                raise RuntimeError(f"{particles.n_mks_loc} rays still inside after {steps} steps; increase max_steps.")
            if pusher is None:
                push_v(0.5 * dt)
                push_eta(dt)
                push_v(0.5 * dt)
            else:
                pusher.step()
            steps += 1
            records = particles.pop_lost_markers()
            if len(records):
                self._finish_exits(charge, records, particles.lost_index, previous, None if pusher is None else step_dt)
            ledger.book(records, particles.lost_index, Propagator.domain, time=steps * dt)
            if crossings is not None:
                self._record_crossings(particles, previous, crossings)
            else:
                valid = particles.valid_mks
                previous[particles.markers[valid, -1].astype(int)] = particles.markers[valid, :6]
            if pusher is None:
                self._deposit(charge, dt)
            else:
                # trapezoid rule over each ray's own path: the node between two steps gets half of each
                next_dt = pusher.choose()
                self._deposit_weighted(charge, 0.5 * (step_dt + next_dt))
                step_dt = next_dt
            if track:
                paths.append(self._tracked_positions(particles))
        if track:
            self.trajectories = np.stack(paths)

        exits = ledger.records[self.exit_tag]
        moments = {}
        if len(exits) > 2:
            # Exit records hold Cartesian x, y, z, vx, vy, vz.  Ion-optics
            # examples use either physical x (straight channels) or physical z
            # (axisymmetric wedges) as the beam direction.  Select it from the
            # current-weighted mean velocity rather than assuming x, then use
            # the next Cartesian coordinate as a transverse projection.
            weights = np.abs(exits[:, 6])
            longitudinal = int(np.argmax(np.average(np.abs(exits[:, 3:6]), axis=0, weights=weights)))
            transverse = (longitudinal + 1) % 3
            denominator = exits[:, 3 + longitudinal]
            valid = np.abs(denominator) > 1e-14
            if np.count_nonzero(valid) > 2:
                moments = rms_moments(
                    exits[valid, transverse],
                    exits[valid, 3 + transverse] / denominator[valid],
                    exits[valid, 6],
                )
        total = self.rays.current.sum()
        record = IterationRecord(
            round=k,
            exit_emittance=moments.get("emittance", np.nan),
            exit_size=moments.get("size", np.nan),
            exit_current=ledger.lost_charge[self.exit_tag] / total,
            lost_current={name: ledger.lost_charge[name] / total for name in ledger.names},
            steps=steps,
            exit_records=exits,
            plane_crossings=crossings,
        )
        return record, charge

    @ProfileManager.profile("ion optics: finish exits")
    def _finish_exits(self, charge, records, index, previous, step=None):
        """Trapezoidal charge of the last, partial step of rays that left during this step.

        A ray that crosses the boundary at the fraction ``f`` of the step was inside for
        ``f * dt`` only. The rule ``(f dt / 2) [q(x_prev) + q(x_exit)]`` replaces the
        ``dt / 2`` already deposited at ``x_prev``. This keeps the deposit continuous in the
        field; otherwise the exit step count jumps and the fixed-point map becomes discontinuous.
        The records are moved to the exit point, which also sharpens the exit diagnostics.
        With per-ray steps, ``step`` holds the step each ray has just taken (else the global ``dt``).
        """
        ids = records[:, index["ids"]].astype(int)
        axis = records[:, index["axis"]].astype(int)
        side = records[:, index["side"]]
        rows = np.arange(len(records))
        eta_prev = previous[ids, :3]
        eta_out = records[:, index["pos"]]
        before, after = eta_prev[rows, axis], eta_out[rows, axis]
        fraction = np.clip((side - before) / np.where(after != before, after - before, 1.0), 0.0, 1.0)
        eta_exit = eta_prev + fraction[:, None] * (eta_out - eta_prev)
        eta_exit[rows, axis] = side
        records[:, index["pos"]] = eta_exit
        current = self.rays.current[ids]
        dt = self.dt if step is None else step[ids]
        self._deposit_points(
            charge,
            np.concatenate([eta_prev, eta_exit]),
            np.concatenate([-(1.0 - fraction) * 0.5 * dt * current, fraction * 0.5 * dt * current]),
        )

    def _deposit_points(self, charge, eta, weights):
        """Add point charges at logical positions ``eta`` to ``charge``, staged in holes of the marker array."""
        particles = self.model.ions.var.particles
        index = particles.index
        valid = particles.valid_mks.copy()
        saved = particles.markers[valid, index["weights"]].copy()
        holes = np.nonzero(particles.holes)[0][: len(eta)]
        if len(holes) < len(eta):
            raise RuntimeError("Not enough free rows in the marker array to stage the exit deposit.")
        particles.markers[valid, index["weights"]] = 0.0
        particles.markers[holes, :] = 0.0
        particles.markers[holes, index["pos"]] = np.clip(eta, 0.0, 1.0 - 1e-14)
        particles.markers[holes, index["weights"]] = weights
        particles.markers[holes, -1] = -3.0
        self._accumulator()
        charge += self._accumulator.vectors[0]
        particles.markers[holes, :] = -1.0
        particles.markers[valid, index["weights"]] = saved
        particles.update_holes()

    @ProfileManager.profile("ion optics: plane crossings")
    def _record_crossings(self, particles, previous, crossings):
        """Store first plane crossings between the previous and the current step (linear interpolation)."""
        axis, values = self.planes
        valid = particles.valid_mks
        ids = particles.markers[valid, -1].astype(int)
        current = particles.markers[valid, :6]
        before = previous[ids]
        for k, value in enumerate(values):
            a, b = before[:, axis] - value, current[:, axis] - value
            hit = (a < 0.0) & (b >= 0.0) | (a > 0.0) & (b <= 0.0)
            new = hit & np.isnan(crossings[k, ids, 0])
            if np.any(new):
                fraction = (a[new] / (a[new] - b[new]))[:, None]
                crossings[k, ids[new]] = before[new] + fraction * (current[new] - before[new])
        previous[ids] = current
        return previous

    def _tracked_positions(self, particles):
        """Logical positions of the first ``n_tracked`` rays (NaN once removed)."""
        out = np.full((self.n_tracked, 3), np.nan)
        valid = particles.valid_mks
        ids = particles.markers[valid, -1].astype(int)
        keep = ids < self.n_tracked
        out[ids[keep]] = particles.markers[valid][keep, :3]
        return out
