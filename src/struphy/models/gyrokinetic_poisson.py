import copy
import logging

from scope_profiler import ProfileManager

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators.gyrokinetic_poisson_solve import GyrokineticPoissonSolve
from struphy.propagators.time_dependent_source import TimeDependentSource

logger = logging.getLogger("struphy")


class GyrokineticPoisson(StruphyModel):
    """Weak discretization of the gyrokinetic Poisson equation with adiabatic electrons and an optional time-dependent right-hand side.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    epsilon: float
        Gyrokinetic parameter (default: 1.0)
    Z: int
        Charge number of the ions (default: 1)
    with_t_dep_source: bool
        Whether the right-hand side source term is time-dependent (default: False)
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.phi = FEECVariable(space="H1")
            self.source = FEECVariable(space="H1")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self, rho: FEECVariable = None, epsilon: float = 1.0, Z: int = 1, with_t_dep_source=False):
            if with_t_dep_source:
                self.source = TimeDependentSource()
            self.gyrokinetic_poisson = GyrokineticPoissonSolve(rho=rho, epsilon=epsilon, Z=Z)

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        epsilon: float = 1.0,
        Z: int = 1,
        with_t_dep_source=False,
    ):

        self.with_t_dep_source = with_t_dep_source

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(
            rho=self.em_fields.source,
            epsilon=epsilon,
            Z=Z,
            with_t_dep_source=with_t_dep_source,
        )

        # 4. assign variables to propagators
        if with_t_dep_source:
            self.propagators.source.variables.source = self.em_fields.source
        self.propagators.gyrokinetic_poisson.variables.phi = self.em_fields.phi

        # 5. define scalars to be tracked during simulation

    @property
    def bulk_species(self):
        return None

    @property
    def velocity_scale(self):
        return None

    def post_allocate(self):
        """Solve initial gyrokinetic Poisson equation.

        :meta private:
        """
        if self.with_t_dep_source:
            # Solve to get initial potential (before time stepping)
            logger.info("\nSolving initial gyrokinetic Poisson problem (before time stepping)...")

            with ProfileManager.profile_region(
                "initial gyrokinetic Poisson solve", functions=[self.propagators.gyrokinetic_poisson]
            ):
                self.propagators.gyrokinetic_poisson(1.0)

            logger.info("... Done.")

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Find :math:`\phi \in H^1` such that

        .. math::

            \frac{1}{Z\epsilon^2} \frac{n_0}{T_0} \left( \phi - \langle \phi \rangle \right) - \nabla \cdot \left( \frac{n_0}{|B_0|^2} \nabla_\perp \phi \right) = \frac{1}{\epsilon} \rho(t, \mathbf{x})\,,

        where :math:`n_0, T_0, |B_0| : \Omega \to \mathbb{R}` are the equilibrium density, temperature and magnetic field strength,
        :math:`\nabla_\perp = (\mathbb{1} - \mathbf b_0 \mathbf b_0^\top) \nabla` is the gradient perpendicular to the unit vector
        :math:`\mathbf b_0 = \mathbf B_0 / |B_0|`, and :math:`\langle \phi \rangle` is the :math:`n_0/T_0`-weighted average
        over the toroidal (cylindrical geometry) or the poloidal and toroidal (toroidal geometry) angle.
        :math:`\epsilon` is the gyrokinetic parameter, :math:`Z` the charge number of the ions, and the right-hand side
        :math:`\rho(t)` is parametrized by time :math:`t`. Boundary terms from integration by parts are assumed to vanish.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The coefficient scaling is

        .. math::

            \hat \rho = \hat n.

        No dedicated velocity normalization is used."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - No default scalar diagnostics are defined by this model."""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.time_dependent_source.TimeDependentSource` (if :attr:`with_t_dep_source` is True)
        2. :class:`~struphy.propagators.gyrokinetic_poisson_solve.GyrokineticPoissonSolve`
        """
        doc = rf"""**1. TimeDependentSource:**

{TimeDependentSource.__doc__}

**2. GyrokineticPoissonSolve:**

{GyrokineticPoissonSolve.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""GyrokineticPoisson is the standalone elliptic field-solve model for the
        gyrokinetic Poisson equation with adiabatic electrons, the field equation of
        :class:`~struphy.models.DriftKineticElectrostaticAdiabatic`. It solves the field
        equation with a prescribed (possibly time-dependent) source, e.g. for verification
        and profiling of the solver in a given MHD equilibrium."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a GyrokineticPoisson model:

        .. code-block:: python

            from struphy.models import GyrokineticPoisson

            model = GyrokineticPoisson(epsilon=1.0, Z=1)
            model.em_fields.phi
            model.em_fields.source
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - verification and profiling of the gyrokinetic Poisson solver
        - electrostatic potentials of a prescribed gyrokinetic charge density in an MHD equilibrium"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - self-consistent kinetic plasma evolution on its own
        - electromagnetic or magnetic-field dynamics"""
