"""Electrostatic ion-optics test-particle model.

This is the first building block for self-consistent ion optics.  It advances
an injected ion population in a prescribed electrostatic field; space-charge
deposition and the nonlinear Poisson iteration are intentionally not part of
this first milestone.
"""

import copy

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from cunumpy import PyccelKernel
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.basic import IdentityOperator
from feectools.linalg.solvers import inverse
from feectools.linalg.utilities import array_to_psydac

from scope_profiler.profile_manager import ProfileManager

from struphy.feec.banded_assembly import assemble_banded_operator
from struphy.feec.h1_quadrature import BandedSPDSolver, H1QuadratureAssembler
from struphy.feec.mass import L2Projector
from struphy.feec.linear_operators import BoundaryOperator, SegmentedChannelBoundaryOperator

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.ion_optics_steady_state import RayBundle, SteadyStateIteration, SteadyStateOptions
from struphy.physics.plasma_models import BoltzmannElectrons
from struphy.models.scalars import FunctionScalarPIC, KineticEnergyPIC, Scalars
from struphy.models.species import FieldSpecies, ParticleSpecies
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation import accum_kernels
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.pic.ion_beams import BeamSource, CurrentLedger
from struphy.propagators.base import Propagator
from struphy.propagators.inject_markers import InjectMarkers
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_v_in_force_field import PushVinForceField


class IonOpticsElectrostatic(StruphyModel):
    r"""Ions moving in a prescribed electrostatic field.

    The model advances

    .. math::

        \dot{\mathbf x} = \mathbf v, \qquad
        \dot{\mathbf v} = \mathbf E(\mathbf x)/\varepsilon,

    with :math:`\mathbf E=-\nabla\phi`.  Initialize ``phi`` through
    Struphy's normal initial-condition path for whole-face electrodes, or pass
    ``electrode_segments`` for a channel whose segment voltages define the
    potential trace directly. The model constructs the FEEC electric field
    after allocation.

    Pass ``steady_state=SteadyStateOptions(...)`` to solve the self-consistent
    ray-traced Vlasov--Poisson problem through :meth:`Simulation.run`.  Without
    it, this is the prescribed-field/time-dependent particle model.
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class Ions(ParticleSpecies):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            alpha: float = None,
            epsilon: float = None,
        ):
            self.var = PICVariable(space="Particles6D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                alpha=alpha,
                epsilon=epsilon,
            )

    class Propagators:
        def __init__(self, phi: FEECVariable, source: BeamSource = None, ledger: CurrentLedger = None):
            if source is not None:
                self.inject = InjectMarkers(source, ledger=ledger)
            self.push_v = PushVinForceField(potential=phi)
            self.push_eta = PushEta()

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        alpha: float = None,
        epsilon: float = None,
        electrode_faces: tuple = None,
        electrode_segments: tuple = None,
        electrode_length: float = None,
        source: BeamSource = None,
        steady_state: SteadyStateOptions = None,
        loss_tags: tuple = (),
        keep_loss_records: tuple = (),
        space_charge: bool = False,
        poisson_direct: bool = True,
        plasma: BoltzmannElectrons = None,
    ):
        self.params = copy.deepcopy(locals())
        if electrode_faces is not None and electrode_segments is not None:
            raise ValueError("Use electrode_faces or electrode_segments, not both.")
        if electrode_faces is not None:
            if len(electrode_faces) != 3 or any(len(pair) != 2 for pair in electrode_faces):
                raise ValueError("electrode_faces must contain three (lower, upper) boolean pairs.")
            if not all(type(value) is bool for pair in electrode_faces for value in pair):
                raise ValueError("electrode_faces entries must be booleans.")
            if not any(value for pair in electrode_faces for value in pair):
                raise ValueError("At least one electrode face is required to fix the potential gauge.")
        if electrode_segments is not None:
            if electrode_length is None or electrode_length <= 0.0:
                raise ValueError("electrode_segments require a positive electrode_length.")
            if not tuple(electrode_segments):
                raise ValueError("electrode_segments must not be empty.")
        self.electrode_faces = electrode_faces
        self.electrode_segments = None if electrode_segments is None else tuple(electrode_segments)
        self.electrode_length = electrode_length
        if steady_state is not None and source is not None:
            raise ValueError("Use source for time-dependent injection or steady_state for ray tracing, not both.")
        if steady_state is not None and not isinstance(steady_state, SteadyStateOptions):
            raise TypeError("steady_state must be a SteadyStateOptions instance.")
        self.steady_state = steady_state
        self.steady_state_iteration = None
        if space_charge and electrode_faces is None and electrode_segments is None:
            raise ValueError(
                "space_charge needs electrodes (electrode_faces or electrode_segments) to fix the potential."
            )
        self.space_charge = space_charge
        # serial runs: factorize the constrained stiffness matrix once instead of CG in every solve
        self.poisson_direct = poisson_direct
        if plasma is not None and not poisson_direct:
            raise ValueError("The nonlinear plasma Poisson solve needs poisson_direct=True.")
        self.plasma = plasma
        self.em_fields = self.EMFields()
        self.ions = self.Ions(
            charge_number=charge_number,
            mass_number=mass_number,
            alpha=alpha,
            epsilon=epsilon,
        )
        self.setup_equation_params(base_units=base_units)

        # injected/lost charge bookkeeping, see struphy.pic.ion_beams
        self.ledger = CurrentLedger(loss_tags, keep_records=keep_loss_records)
        self.propagators = self.Propagators(phi=self.em_fields.phi, source=source, ledger=self.ledger)
        if source is not None:
            self.propagators.inject.variables.var = self.ions.var
        self.propagators.push_v.variables.var = self.ions.var
        self.propagators.push_eta.variables.var = self.ions.var

        scalars = {"kinetic_energy": KineticEnergyPIC(self.ions.var)}
        if source is not None or loss_tags:
            scalars["injected_charge"] = FunctionScalarPIC(lambda: self.ledger.injected_charge, self.ions.var)
            scalars["live_charge"] = FunctionScalarPIC(lambda: self.ions.var.particles.weights.sum(), self.ions.var)
            for name in self.ledger.names:
                scalars[f"lost_charge_{name}"] = FunctionScalarPIC(
                    lambda name=name: self._lost_charge(name),
                    self.ions.var,
                )
        self.scalars = Scalars(**scalars)

    def update_ledger(self):
        """Book the markers removed since the last update in :attr:`ledger`."""
        self.ledger.update(self.ions.var.particles, Propagator.domain, time=self.elapsed_time)

    def _lost_charge(self, name):
        self.update_ledger()
        return self.ledger.lost_charge[name]

    @property
    def bulk_species(self):
        return self.ions

    @property
    def velocity_scale(self):
        return "cyclotron"

    def run_steady_state(self, sim):
        """Run the configured steady Vlasov--Poisson iteration for ``sim``.

        This is called by :meth:`struphy.simulation.sim.Simulation.run`; the
        resulting iteration and its diagnostics are retained in
        :attr:`steady_state_iteration`.
        """
        if self.steady_state is None:
            raise RuntimeError("No steady-state options were configured.")
        options = self.steady_state
        rays = options.rays
        if rays is None:
            rays = RayBundle.from_plane_source(options.source, options.n_rays, seed=options.seed)
        iteration = SteadyStateIteration(
            sim,
            rays,
            dt=options.dt,
            alpha=options.alpha,
            loss_tags=options.loss_tags,
            exit_tag=options.exit_tag,
            n_average=options.n_average,
            tol=options.tol,
            max_rounds=options.max_rounds,
            max_steps=options.max_steps,
            n_tracked=options.n_tracked,
            planes=options.planes,
            criterion=options.criterion,
            anderson=options.anderson,
            verbose=options.verbose,
            relaxation=options.relaxation,
            step_control=options.step_control,
        )
        self.steady_state_iteration = iteration
        iteration.run()
        return iteration

    def post_allocate(self):
        if self.electrode_faces is not None or self.electrode_segments is not None:
            self.solve_vacuum_potential()
        if self.space_charge:
            self._charge_accumulator = AccumulatorVector(
                self.ions.var.particles,
                "H1",
                PyccelKernel(accum_kernels.charge_density_0form),
                Propagator.mass_ops,
                Propagator.domain.args_domain,
            )
            self.update_space_charge()
        Propagator.derham.grad.dot(
            -self.em_fields.phi.spline.vector,
            out=self.em_fields.e_field.spline.vector,
        )

    def solve_vacuum_potential(self):
        """Solve Laplace with the configured electrode potential trace.

        Use unconstrained FEEC spaces: electrode constraints apply only to the
        scalar solve, not to the electric field or particle interpolation.
        Unselected faces have the natural zero-normal-field condition.
        The initialized interior coefficients are discarded. Whole-face
        constraints obtain their trace from initialized ``phi``; segmented
        channel constraints obtain it from ``ElectrodeSegment.voltage``.
        """
        derham = Propagator.derham
        if any(any(pair) for pair in derham.dirichlet_bc):
            raise ValueError("Use free FEEC boundaries; select potential constraints with electrode_faces.")
        if self.electrode_faces is not None:
            for axis, pair in enumerate(self.electrode_faces):
                if any(pair) and derham.bcs[axis] is None:
                    raise ValueError("An electrode face cannot lie in a periodic direction.")
        phi = self.em_fields.phi.spline.vector
        if self.electrode_segments is None:
            interior = BoundaryOperator(phi.space, "H1", self.electrode_faces)
            boundary = IdentityOperator(phi.space) - interior
            lift = boundary.dot(phi)
        else:
            interior = SegmentedChannelBoundaryOperator(phi.space, self.electrode_segments, self.electrode_length)
            lift = interior.lifting()
            boundary = IdentityOperator(phi.space) - interior
        grad = derham.grad
        stiffness = grad.T @ Propagator.mass_ops.M1 @ grad
        lhs = interior @ stiffness @ interior + boundary
        # the electrode constraint and lifting are reused by every space-charge solve
        self._poisson = {
            "interior": interior,
            "lift": lift,
            "stiffness": stiffness,
            "vacuum_rhs": -interior.dot(stiffness.dot(lift)),
            "solver": inverse(lhs, "cg", x0=phi.space.zeros(), tol=1e-12, maxiter=10000, recycle=True),
        }
        if self.poisson_direct:
            self._setup_direct_poisson(phi.space)
        self.vacuum_solver_info = self.solve_potential()

    def _setup_direct_poisson(self, space):
        """Factorize the constrained stiffness matrix once (serial runs with a diagonal constraint)."""
        if MPI.COMM_WORLD.Get_size() != 1:
            return
        poisson = self._poisson
        interior = poisson["interior"]
        mask = interior.dot(array_to_psydac(np.ones(space.dimension), space)).toarray()
        probe = np.random.default_rng(0).standard_normal(space.dimension)
        if not np.allclose(interior.dot(array_to_psydac(probe, space)).toarray(), mask * probe):
            return  # not a diagonal projection; keep the iterative solver
        stiffness = assemble_banded_operator(poisson["stiffness"], space)
        constrained = sps.diags(mask) @ stiffness @ sps.diags(mask) + sps.diags(1.0 - mask)
        lift = poisson["lift"].toarray()
        poisson["direct"] = {
            "lu": spsl.splu(constrained.tocsc()),
            "mask": mask,
            "stiffness": stiffness,
            "vacuum_rhs": -mask * (stiffness @ lift),
            "lift": lift,
        }

    def solve_potential(self, charge=None):
        """Solve :math:`-\\nabla^2 \\phi = \\rho` with the electrode constraints of the vacuum solve.

        Parameters
        ----------
        charge : StencilVector, optional
            Deposited charge :math:`\\sum_p q_p \\Lambda^0_i(\\boldsymbol\\eta_p)` in normalized
            units (charge unit ``IonOpticsUnits.charge``); ``None`` gives the vacuum potential.

        Returns
        -------
        dict
            Info of the iterative solver.
        """
        poisson = self._poisson
        if self.plasma is not None and charge is not None:
            return self._solve_poisson_boltzmann(charge)
        if "direct" in poisson:
            direct = poisson["direct"]
            rhs = direct["vacuum_rhs"] if charge is None else direct["vacuum_rhs"] + direct["mask"] * charge.toarray()
            solution = direct["lift"] + direct["mask"] * direct["lu"].solve(rhs)
            self.em_fields.phi.spline.vector = array_to_psydac(solution, self.em_fields.phi.spline.vector.space)
            self.em_fields.phi.spline.vector.update_ghost_regions()
            # interior equations: mask * (S @ phi) = mask * charge
            residual = direct["mask"] * (direct["stiffness"] @ solution) - rhs + direct["vacuum_rhs"]
            return {"solver": "splu", "success": True, "res_norm": float(np.linalg.norm(residual))}
        rhs = poisson["vacuum_rhs"]
        if charge is not None:
            rhs = rhs + poisson["interior"].dot(charge)
        solver = poisson["solver"]
        correction = solver.solve(rhs)
        info = dict(solver._info)
        if not info["success"]:
            raise RuntimeError(f"Electrode Poisson solve failed: {info}")
        self.em_fields.phi.spline.vector = poisson["lift"] + poisson["interior"].dot(correction)
        self.em_fields.phi.spline.vector.update_ghost_regions()
        return info

    def _setup_plasma_quadrature(self):
        """Quadrature data for the Galerkin electron term (L2 projector of H1)."""
        projector = L2Projector("H1", Propagator.mass_ops)
        return {
            "projector": projector,
            "points": [pts.flatten() for pts in projector.quad_grid_pts[0]],
            "geom_weights": projector.geom_weights,
        }

    def _setup_fast_poisson(self, space):
        """Reduced-space data for the fast Poisson–Boltzmann Newton solve, or ``None`` if not applicable.

        Applies to serial, non-periodic spaces whose electrode constraint and lifting do not depend on
        the collapsed (invariant) directions, see :mod:`struphy.feec.h1_quadrature`.
        """
        direct = self._poisson.get("direct")
        if direct is None:
            return None
        try:
            assembler = H1QuadratureAssembler(Propagator.derham, space)
        except NotImplementedError:
            return None
        mask = assembler.reduce_coefficients(direct["mask"])
        lift = assembler.reduce_coefficients(direct["lift"])
        if not (
            np.allclose(assembler.expand(mask), direct["mask"]) and np.allclose(assembler.expand(lift), direct["lift"])
        ):
            return None
        quad = self._poisson.setdefault("plasma", self._setup_plasma_quadrature())
        return {
            "assembler": assembler,
            "mask": mask,
            "lift": lift,
            "stiffness": assembler.reduce_matrix(direct["stiffness"]),
            "geom_weights": np.asarray(quad["geom_weights"]),
        }

    def _solve_poisson_boltzmann(self, charge, tol=1e-12, max_iter=50):
        r"""Poisson–Boltzmann solve by Newton's method on a convex energy.

        For fixed ion charge :math:`q_i = \sum_p q_p \Lambda^0_i(\boldsymbol\eta_p)` the potential
        minimizes, over the coefficients with the electrode values fixed,

        .. math::

            E(\phi) = \tfrac12 \int |\nabla \phi|^2 - \sum_i q_i \phi_i
                + \int \rho_{e0} T_e\, e^{(\phi - \phi_P)/T_e}\,\mathrm d\mathbf x\,,

        whose Euler–Lagrange equation is the Galerkin form of
        :math:`-\nabla^2\phi = \rho_i + \rho_e(\phi)`. The electron integrals are evaluated
        by Gauss quadrature of :math:`\phi_h`, and the Hessian is the stiffness matrix plus the
        weighted mass matrix :math:`\mathbb M^0[\rho_{e0}/T_e\, e^{(\phi_h-\phi_P)/T_e}]`. E is
        strictly convex, so Newton with a backtracking line search on E converges
        globally to the unique solution; no initial plasma-region guess is needed.

        This is the fast path: quadrature transfers and the weighted mass matrix come from
        :class:`~struphy.feec.h1_quadrature.H1QuadratureAssembler` (element-wise sum-factorization,
        validated against Struphy's own operators to roundoff), invariant directions are collapsed
        (half the unknowns for a slit or wedge), and the symmetric positive definite banded Hessian is
        factorized with LAPACK's banded Cholesky. One Newton step costs about 40 ms on a 128 x 48 mesh,
        against about 2.4 s with the generic operators (:meth:`_solve_poisson_boltzmann_generic`,
        used as a fallback for periodic or parallel cases). The results agree to solver tolerance.
        """
        if "fast" not in self._poisson:
            space = self.em_fields.phi.spline.vector.space
            self._poisson["fast"] = self._setup_fast_poisson(space)
        fast = self._poisson["fast"]
        if fast is None:
            return self._solve_poisson_boltzmann_generic(charge, tol, max_iter)
        asm, geom, mask, stiffness, lift = (
            fast["assembler"],
            fast["geom_weights"],
            fast["mask"],
            fast["stiffness"],
            fast["lift"],
        )
        plasma = self.plasma
        space = self.em_fields.phi.spline.vector.space
        q = asm.reduce_load(charge.toarray())
        constraint = sps.diags(mask)
        identity_on_electrodes = sps.diags(1.0 - mask)

        def energy_and_gradient(c):
            with ProfileManager.profile_region("ion optics: electron terms"):
                electron_charge = plasma.charge_density(asm.values(c)) * geom  # negative
                electron_energy = -plasma.temperature * asm.integrate(electron_charge)
                electron_gradient = -asm.project(electron_charge)
                hessian_weight = -plasma.charge_density_derivative(asm.values(c)) * geom
            sc = stiffness @ c
            return 0.5 * c @ sc - q @ c + electron_energy, mask * (sc - q + electron_gradient), hessian_weight

        c = lift + mask * asm.reduce_coefficients(self.em_fields.phi.spline.vector.toarray())
        energy, gradient, hessian_weight = energy_and_gradient(c)
        for iteration in range(1, max_iter + 1):
            with ProfileManager.profile_region("ion optics: newton hessian"):
                hessian = constraint @ (stiffness + asm.matrix(hessian_weight)) @ constraint + identity_on_electrodes
            with ProfileManager.profile_region("ion optics: newton solve"):
                step = -BandedSPDSolver(hessian).solve(gradient)
            decrement = -gradient @ step
            if decrement < 0.0:
                raise RuntimeError("The Newton direction is not a descent direction.")
            # backtracking (Armijo) line search on the convex energy
            t = 1.0
            while True:
                trial = c + t * step
                trial_energy, trial_gradient, trial_weight = energy_and_gradient(trial)
                if trial_energy <= energy - 1e-4 * t * decrement or t < 1e-8:
                    break
                t *= 0.5
            c, energy, gradient, hessian_weight = trial, trial_energy, trial_gradient, trial_weight
            if decrement < tol * max(1.0, abs(energy)):
                break
        else:
            raise RuntimeError(f"Poisson–Boltzmann Newton did not converge in {max_iter} iterations.")
        self.em_fields.phi.spline.vector = array_to_psydac(asm.expand(c), space)
        self.em_fields.phi.spline.vector.update_ghost_regions()
        return {"solver": "newton", "success": True, "iterations": iteration, "decrement": float(decrement)}

    def _solve_poisson_boltzmann_generic(self, charge, tol=1e-12, max_iter=50):
        """Generic (slow) Poisson–Boltzmann Newton solve with Struphy's operators; see the fast path.

        Used for periodic or parallel cases. The Hessian mass matrix is assembled with
        ``create_weighted_mass`` and converted by probing; nothing here assumes invariant directions.
        """
        if "direct" not in self._poisson:
            raise NotImplementedError("The Poisson–Boltzmann solve needs the serial direct Poisson solver.")
        direct = self._poisson["direct"]
        quad = self._poisson.setdefault("plasma", self._setup_plasma_quadrature())
        mask, stiffness = direct["mask"], direct["stiffness"]
        plasma = self.plasma
        space = self.em_fields.phi.spline.vector.space
        q = charge.toarray()

        def electron_terms(c):
            """Energy integral, gradient vector and Hessian weights of the electron term at coefficients c."""
            with ProfileManager.profile_region("ion optics: electron terms"):
                return _electron_terms(c)

        def _electron_terms(c):
            self.em_fields.phi.spline.vector = array_to_psydac(c, space)
            self.em_fields.phi.spline.vector.update_ghost_regions()
            phi_q = np.asarray(self.em_fields.phi.spline(*quad["points"]))
            energy_density = -plasma.temperature * plasma.charge_density(phi_q)
            gradient_density = -plasma.charge_density(phi_q)
            projector = quad["projector"]
            energy = float(np.sum(projector.get_dofs(energy_density).toarray()))
            gradient = projector.get_dofs(gradient_density).toarray()
            return energy, gradient, -plasma.charge_density_derivative(phi_q)

        def energy_and_gradient(c):
            electron_energy, electron_gradient, hessian_weight = electron_terms(c)
            sc = stiffness @ c
            energy = 0.5 * c @ sc - q @ c + electron_energy
            return energy, mask * (sc - q + electron_gradient), hessian_weight

        c = self.em_fields.phi.spline.vector.toarray()
        c = direct["lift"] + mask * c
        energy, gradient, hessian_weight = energy_and_gradient(c)
        for iteration in range(1, max_iter + 1):
            with ProfileManager.profile_region("ion optics: newton hessian"):
                with ProfileManager.profile_region("ion optics: hessian assemble"):
                    weighted_mass = Propagator.mass_ops.create_weighted_mass(
                        "H1",
                        "H1",
                        name="ion_optics_boltzmann_hessian",
                        weights=[[hessian_weight * quad["geom_weights"]]],
                        assemble=True,
                    )
                with ProfileManager.profile_region("ion optics: hessian to sparse"):
                    hessian_mass = weighted_mass.tosparse()
                hessian = sps.diags(mask) @ (stiffness + hessian_mass) @ sps.diags(mask) + sps.diags(1.0 - mask)
            with ProfileManager.profile_region("ion optics: newton solve"):
                step = -spsl.spsolve(hessian.tocsc(), gradient)
            decrement = -gradient @ step
            if decrement < 0.0:
                raise RuntimeError("The Newton direction is not a descent direction.")
            # backtracking (Armijo) line search on the convex energy
            t = 1.0
            while True:
                trial = c + t * step
                trial_energy, trial_gradient, trial_weight = energy_and_gradient(trial)
                if trial_energy <= energy - 1e-4 * t * decrement or t < 1e-8:
                    break
                t *= 0.5
            c, energy, gradient, hessian_weight = trial, trial_energy, trial_gradient, trial_weight
            if decrement < tol * max(1.0, abs(energy)):
                break
        else:
            raise RuntimeError(f"Poisson–Boltzmann Newton did not converge in {max_iter} iterations.")
        # the last energy evaluation left phi at c
        return {"solver": "newton", "success": True, "iterations": iteration, "decrement": float(decrement)}

    @ProfileManager.profile("ion optics: solve potential")
    def solve_potential_profiled(self, charge=None):
        """``solve_potential`` under a profiling region (used by the steady-state iteration)."""
        return self.solve_potential(charge)

    def update_space_charge(self):
        """Deposit the ion charge, solve for the potential and update ``e_field``."""
        self._charge_accumulator()
        self.space_charge_solver_info = self.solve_potential(self._charge_accumulator.vectors[0])
        Propagator.derham.grad.dot(-self.em_fields.phi.spline.vector, out=self.em_fields.e_field.spline.vector)

    @property
    def elapsed_time(self) -> float:
        """Time advanced by :meth:`integrate` (normalized units)."""
        return getattr(self, "_elapsed_time", 0.0)

    def integrate(self, dt, split_algo="LieTrotter"):
        # The field is quasi-static: solve Poisson for the current charge, then advance the markers.
        if self.space_charge:
            self.update_space_charge()
        super().integrate(dt, split_algo)
        self._elapsed_time = self.elapsed_time + dt

    @classmethod
    def doc_pde(cls):
        return cls.__doc__

    @classmethod
    def doc_normalization(cls):
        return """The model uses the ion cyclotron normalization associated with ``BaseUnits``."""

    @classmethod
    def doc_scalar_quantities(cls):
        return """**The following scalars are tracked:**

        - Particle kinetic energy."""

    @classmethod
    def doc_discretization(cls):
        return """Time integration applies ``PushVinForceField`` followed by ``PushEta``."""

    @classmethod
    def doc_long_description(cls):
        return """A prescribed-field ion-optics baseline. Space charge is deliberately deferred to a later model milestone."""

    @classmethod
    def doc_examples(cls):
        return """Create the model with ``IonOpticsElectrostatic(base_units=IonOpticsUnits(...).base_units(), electrode_faces=...)``,
        initialize the electrode potential on ``em_fields.phi`` (e.g. with ``PiecewiseLinearPotential``) and load ``ions.var``.
        See ``examples/IonOpticsElectrostatic/slit_immersion_lens``."""

    @classmethod
    def doc_use_cases(cls):
        return """Prescribed-field extraction, acceleration and focusing tests."""

    @classmethod
    def doc_cannot_be_used_for(cls):
        return """Self-consistent space charge and internal electrode surfaces are not yet included."""
