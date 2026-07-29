import copy

from feectools.ddm.mpi import mpi as MPI

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import KineticEnergyPIC, Scalars
from struphy.models.species import (
    ParticleSpecies,
)
from struphy.models.variables import PICVariable
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_vxb import PushVxB




class Vlasov(StruphyModel):
    """Vlasov equation for a single species in a static background magnetic field.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    charge_number: int
        Charge number (in units of the positive elementary charge) of the species (default: 1)
    mass_number: float
        Mass number (in units of Proton mass) of the species (default: 1.0)
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
            self.push_vxb = PushVxB()
            self.push_eta = PushEta()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
    ):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

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

    def allocate_helpers(self):
        pass

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
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.push_vxb.PushVxB`
        2. :class:`~struphy.propagators.push_eta.PushEta`
        """
        doc = rf"""**1. push_vxb.PushVxB:**

    {PushVxB.__doc__}

    **2. push_eta.PushEta:**

    {PushEta.__doc__}
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
