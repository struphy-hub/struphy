import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
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
        def __init__(self):
            self.var = PICVariable(space="Particles6D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.push_vxb = propagators_markers.PushVxB()
            self.push_eta = propagators_markers.PushEta()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}' ***")

        # 1. instantiate all species
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.push_vxb.variables.ions = self.kinetic_ions.var
        self.propagators.push_eta.variables.var = self.kinetic_ions.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_f", compute="from_particles", variable=self.kinetic_ions.var)

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "cyclotron"

    def allocate_helpers(self):
        self._tmp = xp.empty(1, dtype=float)

    def update_scalar_quantities(self):
        particles = self.kinetic_ions.var.particles
        self._tmp[0] = particles.markers_wo_holes[:, 6].dot(
            particles.markers_wo_holes[:, 3] ** 2
            + particles.markers_wo_holes[:, 4] ** 2
            + particles.markers_wo_holes[:, 5] ** 2,
        ) / (2 * particles.Np)

        self.update_scalar("en_f", self._tmp[0])
