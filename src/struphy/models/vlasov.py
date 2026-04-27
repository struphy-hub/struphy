from feectools.ddm.mpi import mpi as MPI

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import KineticEnergyPIC, Scalars
from struphy.models.species import (
    ParticleSpecies,
)
from struphy.models.variables import PICVariable
from struphy.propagators import (
    propagators_markers,
)

rank = MPI.COMM_WORLD.Get_rank()


class Vlasov(StruphyModel):
    r"""Vlasov equation in static background magnetic field.

    Kinetic plasma description using particle distribution function in fixed external magnetic field.
    Evolves the particle distribution function f(x, v, t) under Lorentz force from externally applied fields.
    No self-consistent electromagnetic fields or particle feedback on fields.
    Suitable for investigating kinetic dynamics in imposed magnetic field configurations.

    **Physics Description:**
    This model solves the collisionless Vlasov equation for individual particle species.
    Particles move under the influence of a prescribed (static) background magnetic field B0.
    The distribution function f describes the density of particles at position x with velocity v.
    Useful for studying kinetic effects, gyro-motion, phase space dynamics, and particle trajectories
    in externally controlled magnetic field environments without plasma self-interactions.

    **Keywords:** kinetic, Vlasov equation, particle distribution, collisionless plasma, gyro-motion,
    cyclotron motion, magnetic field, Lorentz force, phase space, distribution function, kinetic dynamics,
    gyrokinetics, charged particle, background field, external field, drift motion, cyclotron radius,
    Larmor radius, particle tracing, kinetic transport, phase space evolution, kinetic instability,
    particle dynamics, magnetic confinement, drift orbits, gyroscale

    **Physics type:** Kinetic (particle-based, not fluid)
    **Particle dynamics:** Full kinetic evolution - gyro-motion, drifts, and advection in velocity space
    **Field coupling:** None - uses imposed external magnetic field only (no self-consistency)
    **Current source:** None (fields are given, not self-generated)
    **Collision effects:** None (collisionless kinetic model)
    **Temperature effects:** Implicit in initial distribution function (not enforced separately)

    **Use for:**
    - Kinetic dynamics in fixed magnetic field geometry
    - Gyro-motion and cyclotron dynamics studies
    - Particle distribution evolution in external fields
    - Phase space advection and flow
    - Kinetic effects without plasma feedback
    - Testing particle propagators and methods

    :ref:`normalization`:

    .. math::

        \hat v = \hat \Omega_\textnormal{c} \hat x\,.,,

    :ref:`Equations <gempic>`:

    .. math::

        \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \left(\mathbf{v}\times\mathbf{B}_0 \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushVxB`
    2. :class:`~struphy.propagators.propagators_markers.PushEta`
    """

    __exclusion__ = """
    This model should NOT be used for:
    - Self-consistent electromagnetic coupling
    - Plasma instabilities with particle feedback
    - Two-stream instability
    - Magnetic field evolution
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

    ## species

    class KineticIons(ParticleSpecies):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
        ):
            self.var = PICVariable(space="Particles6D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
            )

    ## propagators

    class Propagators:
        def __init__(self):
            self.push_vxb = propagators_markers.PushVxB()
            self.push_eta = propagators_markers.PushEta()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
    ):

        # 1. instantiate all species
        self.kinetic_ions = self.KineticIons(
            charge_number,
            mass_number,
        )

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.push_vxb.variables.ions = self.kinetic_ions.var
        self.propagators.push_eta.variables.var = self.kinetic_ions.var

        # 5. define scalars to be tracked during simulation
        kinetic_energy = KineticEnergyPIC(self.kinetic_ions.var)
        self.scalars = Scalars(kinetic_energy=kinetic_energy)

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "cyclotron"

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Vlasov equation:

        .. math::

            \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \left( \mathbf{v} \times \mathbf{B}_0 \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""The characteristic speed is the cyclotron scale

        .. math::

            \hat v = \hat\Omega_c \hat x.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Particle kinetic energy: ``kinetic_energy``"""

    @classmethod
    def doc_discretization(cls):
        doc = rf"""**1. propagators_markers.PushVxB:**

{propagators_markers.PushVxB.__doc__}

**2. propagators_markers.PushEta:**

{propagators_markers.PushEta.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""Vlasov is the simplest kinetic test-particle model in the 6D hierarchy.
        It evolves particles in a static magnetic background without electric or
        magnetic self-consistency."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a Vlasov test-particle model:

        .. code-block:: python

            from struphy.models import Vlasov

            model = Vlasov()
            model.kinetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - test-particle motion in prescribed magnetic fields
        - verification of the Boris-like VxB and PushEta splitting
        - reduced kinetic transport studies without field feedback"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - self-consistent electrostatic or electromagnetic coupling
        - collisional kinetic dynamics
        - guiding-center reduction studies
        - fluid or MHD-scale closures"""

    def allocate_helpers(self, verbose: bool = False):
        pass
