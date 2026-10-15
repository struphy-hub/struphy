import copy

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, Scalars
from struphy.models.species import FieldSpecies
from struphy.models.variables import FEECVariable
from struphy.propagators.scalar_wave_leapfrog import ScalarWaveLeapfrog


class ScalarWave(StruphyModel):
    """Scalar (0-form) wave equation, the model problem of the polar-spline wave studies.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.phi = FEECVariable(space="H1")
            self.psi = FEECVariable(space="H1")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.wave = ScalarWaveLeapfrog()

    ## abstract methods

    def __init__(self, base_units: BaseUnits = BaseUnits()):
        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()

        # 2. derive units
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.wave.variables.phi = self.em_fields.phi
        self.propagators.wave.variables.psi = self.em_fields.psi

        # 5. scalars: kinetic energy 1/2 |psi|^2 (the potential energy 1/2 |grad phi|^2 is computed in the pproc)
        self.scalars = Scalars(kinetic_energy=BilinearEnergyFEEC(self.em_fields.psi))

    @property
    def bulk_species(self):
        return None

    @property
    def velocity_scale(self):
        return None

    def post_allocate(self):
        pass

    ## abstract methods for documentation

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Toy"

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        .. math::

            \frac{\partial^2 \phi}{\partial t^2} - c^2 \Delta \phi = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""All quantities are dimensionless; the wave speed :math:`c` is an option of the propagator."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy: :math:`\frac{1}{2} \int |\partial_t \phi|^2 \, dV`"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.scalar_wave_leapfrog.ScalarWaveLeapfrog`
        """
        return rf"""**1. propagators.wave.ScalarWaveLeapfrog:**

{ScalarWaveLeapfrog.__doc__}
"""

    @classmethod
    def doc_long_description(cls):
        r"""The 0-form wave equation in H1 with the field :math:`\phi` and its time derivative :math:`\psi`, integrated
        by the symplectic leapfrog. It is the model problem of the high-order polar-spline studies (smoothness of the
        waves crossing the polar axis, see ``DerhamOptions(polar_method="homotopy")``)."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a ScalarWave model:

        .. code-block:: python

            from struphy.models import ScalarWave

            model = ScalarWave()
            model.em_fields.phi  # wave field
            model.em_fields.psi  # its time derivative
        """

    @classmethod
    def doc_use_cases(cls):
        """Verification of FEEC spaces (e.g. polar splines) with a simple hyperbolic problem."""

    @classmethod
    def doc_cannot_be_used_for(cls):
        """Plasma dynamics; the model has no particles and no coupling to other fields."""
