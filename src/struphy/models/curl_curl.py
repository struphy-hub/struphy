import copy
import logging

from scope_profiler import ProfileManager

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators.curl_curl_solve import CurlCurlSolve
from struphy.propagators.time_dependent_source import TimeDependentSource

logger = logging.getLogger("struphy")


class CurlCurl(StruphyModel):
    """Weak discretization of the curl-curl problem with a mass (stabilization) term and an optional time-dependent right-hand side.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    with_t_dep_source: bool
        Whether the right-hand side source term is time-dependent (default: False)
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.source = FEECVariable(space="Hcurl")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self, j: FEECVariable = None, with_t_dep_source=False):
            if with_t_dep_source:
                self.source = TimeDependentSource()
            self.curl_curl = CurlCurlSolve(j=j)

    ## abstract methods

    def __init__(self, base_units: BaseUnits = BaseUnits(), with_t_dep_source=False):

        self.with_t_dep_source = with_t_dep_source

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(j=self.em_fields.source, with_t_dep_source=with_t_dep_source)

        # 4. assign variables to propagators
        if with_t_dep_source:
            self.propagators.source.variables.source = self.em_fields.source
        self.propagators.curl_curl.variables.e = self.em_fields.e_field

        # 5. define scalars to be tracked during simulation

    @property
    def bulk_species(self):
        return None

    @property
    def velocity_scale(self):
        return None

    def post_allocate(self):
        """Solve initial curl-curl problem.

        :meta private:
        """
        if self.with_t_dep_source:
            # Solve to get initial field (before time stepping)
            logger.info("\nSolving initial curl-curl problem (before time stepping)...")

            with ProfileManager.profile_region("initial curl-curl solve", functions=[self.propagators.curl_curl]):
                self.propagators.curl_curl(1.0)

            logger.info("... Done.")

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Find :math:`\mathbf E \in H(\textnormal{curl})` such that

        .. math::

            \nabla \times \nabla \times \mathbf E + \sigma\, \mathbf E = \mathbf J(t, \mathbf{x})

        where :math:`\sigma > 0` is a scalar and :math:`\mathbf J(t) : \Omega \to \mathbb{R}^3` is a
        source parametrized by time :math:`t`. Boundary terms from integration by parts are assumed to vanish.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The coefficient scaling is

        .. math::

            \hat \sigma = 1 / \hat x^2,\qquad \hat J = \hat E / \hat x^2.

        No dedicated velocity normalization is used."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - No default scalar diagnostics are defined by this model."""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.time_dependent_source.TimeDependentSource` (if :attr:`with_t_dep_source` is True)
        2. :class:`~struphy.propagators.curl_curl_solve.CurlCurlSolve`
        """
        doc = rf"""**1. TimeDependentSource:**

{TimeDependentSource.__doc__}

**2. CurlCurlSolve:**

{CurlCurlSolve.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""CurlCurl is the standalone elliptic field-solve model for the stabilized curl-curl problem
        in :math:`H(\textnormal{curl})`, e.g. from implicit time stepping of Maxwell's equations or
        from magnetostatics. It is used to test and profile solvers and preconditioners for this
        operator, whose large kernel (the gradients) makes standard preconditioners ineffective."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a CurlCurl model:

        .. code-block:: python

            from struphy.models import CurlCurl

            model = CurlCurl()
            model.em_fields.e_field
            model.em_fields.source
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - elliptic benchmark problems in :math:`H(\textnormal{curl})`
        - testing and profiling curl-curl solvers and preconditioners
        - field solves with prescribed current sources"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - hyperbolic time-dependent wave propagation
        - self-consistent kinetic plasma evolution on its own
        - the singular case :math:`\sigma = 0` (it is regularized with :math:`\sigma = 10^{-14}`)"""
