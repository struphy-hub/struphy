from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators import (
    propagators_fields,
)

rank = MPI.COMM_WORLD.Get_rank()


class Maxwell(StruphyModel):
    r"""Maxwell's equations in vacuum.

    Pure electromagnetic field evolution without kinetic effects or particles.
    Solves the coupled system of electric and magnetic field evolution in vacuum or homogeneous media.
    Models electromagnetic wave propagation, field oscillations, and Poynting flux.

    **Physics Description:**
    This model tracks the evolution of electric (E) and magnetic (B) fields only.
    No particle distribution functions. No plasma response or current from particles.
    Fields evolve according to the fundamental Maxwell equations with no sources.
    Useful for understanding pure electromagnetic wave dynamics and field behavior.

    **Keywords:** electromagnetic radiation, light waves, EM waves, Faraday's law, Ampere's law,
    electric field, magnetic field, wave equation, wave propagation, Maxwell equations, vacuum,
    electromagnetic field solver, curl operator, Poynting vector, field dynamics, field evolution,
    electromagnetic coupling, transverse waves, field oscillation

    **Physics type:** Fluid (not kinetic)
    **Particle dynamics:** None (not a kinetic model)
    **Field coupling:** Self-consistent electromagnetic fields only (no particle feedback)
    **Current source:** None (no plasma or particles)
    **Collision effects:** None (vacuum)
    **Temperature effects:** None (no particles or thermodynamics)

    **Use for:**
    - Electromagnetic wave propagation in free space
    - Field dynamics in cavities or domains without plasma
    - Testing electromagnetic field solvers
    - Understanding pure wave behavior

    :ref:`normalization`:

    .. math::

        \hat E = c \hat B\,.,

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial \mathbf E}{\partial t} - \nabla\times\mathbf B = 0\,,

        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.Maxwell`
    """

    __exclusion__ = """
    This model should NOT be used for:
    - Kinetic plasma simulations
    - Particle distribution function evolution
    - Plasma response and coupling effects
    - Beam dynamics studies
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.maxwell = propagators_fields.Maxwell()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.em_fields = self.EMFields()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field

        # define scalars for update_scalar_quantities
        self.add_scalar("electric energy")
        self.add_scalar("magnetic energy")
        self.add_scalar("total energy")

    @property
    def bulk_species(self):
        return None

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self):
        pass

    def update_scalar_quantities(self):
        en_E = 0.5 * self.mass_ops.M1.dot_inner(
            self.em_fields.e_field.spline.vector,
            self.em_fields.e_field.spline.vector,
        )
        en_B = 0.5 * self.mass_ops.M2.dot_inner(
            self.em_fields.b_field.spline.vector,
            self.em_fields.b_field.spline.vector,
        )

        self.update_scalar("electric energy", en_E)
        self.update_scalar("magnetic energy", en_B)
        self.update_scalar("total energy", en_E + en_B)
