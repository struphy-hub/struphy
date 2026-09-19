"""Electrostatic ion-optics test-particle model.

This is the first building block for self-consistent ion optics.  It advances
an injected ion population in a prescribed electrostatic field; space-charge
deposition and the nonlinear Poisson iteration are intentionally not part of
this first milestone.
"""

import copy

from feectools.linalg.basic import IdentityOperator
from feectools.linalg.solvers import inverse

from struphy.feec.linear_operators import BoundaryOperator

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import KineticEnergyPIC, Scalars
from struphy.models.species import FieldSpecies, ParticleSpecies
from struphy.models.variables import FEECVariable, PICVariable
from struphy.propagators.base import Propagator
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_v_in_force_field import PushVinForceField


class IonOpticsElectrostatic(StruphyModel):
    r"""Ions moving in a prescribed electrostatic field.

    The model advances

    .. math::

        \dot{\mathbf x} = \mathbf v, \qquad
        \dot{\mathbf v} = \mathbf E(\mathbf x)/\varepsilon,

    with :math:`\mathbf E=-\nabla\phi`.  Initialize ``phi`` through
    Struphy's normal initial-condition path; the model constructs the FEEC
    electric field from it after allocation.

    This intentionally contains no field evolution and no particle charge
    deposition.  It is the verification baseline for the subsequent
    self-consistent Poisson/space-charge model.
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
        def __init__(self, phi: FEECVariable):
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
    ):
        self.params = copy.deepcopy(locals())
        if electrode_faces is not None:
            if len(electrode_faces) != 3 or any(len(pair) != 2 for pair in electrode_faces):
                raise ValueError("electrode_faces must contain three (lower, upper) boolean pairs.")
            if not all(type(value) is bool for pair in electrode_faces for value in pair):
                raise ValueError("electrode_faces entries must be booleans.")
            if not any(value for pair in electrode_faces for value in pair):
                raise ValueError("At least one electrode face is required to fix the potential gauge.")
        self.electrode_faces = electrode_faces
        self.em_fields = self.EMFields()
        self.ions = self.Ions(
            charge_number=charge_number,
            mass_number=mass_number,
            alpha=alpha,
            epsilon=epsilon,
        )
        self.setup_equation_params(base_units=base_units)

        self.propagators = self.Propagators(phi=self.em_fields.phi)
        self.propagators.push_v.variables.var = self.ions.var
        self.propagators.push_eta.variables.var = self.ions.var

        self.scalars = Scalars(kinetic_energy=KineticEnergyPIC(self.ions.var))

    @property
    def bulk_species(self):
        return self.ions

    @property
    def velocity_scale(self):
        return "cyclotron"

    def post_allocate(self):
        if self.electrode_faces is not None:
            self.solve_vacuum_potential()
        Propagator.derham.grad.dot(
            -self.em_fields.phi.spline.vector,
            out=self.em_fields.e_field.spline.vector,
        )

    def solve_vacuum_potential(self):
        """Solve Laplace with boundary traces taken from the initialized phi.

        Use unconstrained FEEC spaces: electrode constraints apply only to the
        scalar solve, not to the electric field or particle interpolation.
        Unselected faces have the natural zero-normal-field condition.
        The initialized interior coefficients are discarded.
        """
        derham = Propagator.derham
        if any(any(pair) for pair in derham.dirichlet_bc):
            raise ValueError("Use free FEEC boundaries; select potential constraints with electrode_faces.")
        for axis, pair in enumerate(self.electrode_faces):
            if any(pair) and derham.bcs[axis] is None:
                raise ValueError("An electrode face cannot lie in a periodic direction.")
        phi = self.em_fields.phi.spline.vector
        interior = BoundaryOperator(phi.space, "H1", self.electrode_faces)
        boundary = IdentityOperator(phi.space) - interior
        lift = boundary.dot(phi)
        grad = derham.grad
        stiffness = grad.T @ Propagator.mass_ops.M1 @ grad
        lhs = interior @ stiffness @ interior + boundary
        rhs = -interior.dot(stiffness.dot(lift))
        solver = inverse(lhs, "cg", tol=1e-12, maxiter=10000)
        correction = solver.solve(rhs)
        self.vacuum_solver_info = dict(solver._info)
        if not self.vacuum_solver_info["success"]:
            raise RuntimeError(f"Vacuum Laplace solve failed: {self.vacuum_solver_info}")
        self.em_fields.phi.spline.vector = lift + interior.dot(correction)
        self.em_fields.phi.spline.vector.update_ghost_regions()

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
        return """Create the model with ``IonOpticsElectrostatic()`` and initialize ``em_fields.phi`` and ``ions.var``."""

    @classmethod
    def doc_use_cases(cls):
        return """Prescribed-field extraction, acceleration and focusing tests."""

    @classmethod
    def doc_cannot_be_used_for(cls):
        return """Self-consistent space charge and internal electrode surfaces are not yet included."""
