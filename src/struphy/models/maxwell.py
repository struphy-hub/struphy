from feectools.ddm.mpi import mpi as MPI

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, Scalars
from struphy.models.species import (
    FieldSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators import (
    propagators_fields,
)
from struphy.propagators.base import Propagator
from struphy.utils.docstring_converter import auto_convert_docstring

rank = MPI.COMM_WORLD.Get_rank()


@auto_convert_docstring
class Maxwell(StruphyModel):
    """Maxwell's equations in vacuum for electromagnetic field evolution.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Toy"

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

    def __init__(self, base_units: BaseUnits = BaseUnits()):

        # 1. instantiate all species
        self.em_fields = self.EMFields()

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field

        # 5. define scalars to be tracked during simulation
        electric_energy = BilinearEnergyFEEC(self.em_fields.e_field)
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        total_energy = electric_energy + magnetic_energy

        self.scalars = Scalars(
            electric_energy=electric_energy,
            magnetic_energy=magnetic_energy,
            total_energy=total_energy,
        )

    @property
    def bulk_species(self):
        return None

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self, verbose: bool = False):
        pass

    __doc_rst__ = r"""
Maxwell's equations in vacuum for electromagnetic field evolution.

This model simulates the propagation of electromagnetic waves in vacuum using Maxwell's equations
without sources. It uses a finite element exterior calculus (FEEC) formulation with the electric field
in H(curl) and the magnetic field in H(div) spaces.

**Governing Equations**

Ampère's law (no current):

.. math::

    \frac{\partial \mathbf E}{\partial t} - \nabla\times\mathbf B = 0

Faraday's law:

.. math::

    \frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0

**Normalization**

Fields are normalized such that:

.. math::

    \hat E = c \hat B

where :math:`c` is the speed of light.

**Species**

- ``em_fields.e_field`` - Electric field (H(curl) space)
- ``em_fields.b_field`` - Magnetic field (H(div) space)

**Propagators**

1. :class:`~struphy.propagators.propagators_fields.Maxwell` - Time integration scheme

**Scalar Quantities**

The following quantities are tracked during simulation:

- Electric energy: :math:`E_E = \frac{1}{2} \int |\mathbf E|^2 \, dV`
- Magnetic energy: :math:`E_B = \frac{1}{2} \int |\mathbf B|^2 \, dV`
- Total energy: :math:`E_{total} = E_E + E_B`

**Model Properties**

- **Model type:** Toy
- **Velocity scale:** Speed of light
- **Bulk species:** None

**See Also**

- :class:`~struphy.models.base.StruphyModel` - Base class for all Struphy models
- :class:`~struphy.propagators.propagators_fields.Maxwell` - Maxwell propagator implementation

**Examples**

Create and initialize a Maxwell model:

.. code-block:: python

    from struphy.models.maxwell import Maxwell
    
    model = Maxwell()
    # Fields are accessible via:
    # model.em_fields.e_field
    # model.em_fields.b_field
"""
