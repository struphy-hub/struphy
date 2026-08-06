import copy

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    ParticleSpecies,
)
from struphy.models.variables import PICVariable
from struphy.propagators.push_deterministic_diffusion import PushDeterministicDiffusion


class DeterministicParticleDiffusion(StruphyModel):
    """Diffusion equation discretized with a deterministic particle method.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

    ## species

    class Hydrogen(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles3D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.det_diff = PushDeterministicDiffusion()

    ## abstract methods

    def __init__(self, base_units: BaseUnits = BaseUnits()):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.hydrogen = self.Hydrogen()

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.det_diff.variables.var = self.hydrogen.var

        # 5. define scalars to be tracked during simulation

    @property
    def bulk_species(self):
        return self.hydrogen

    @property
    def velocity_scale(self):
        return None

    def allocate_helpers(self):
        pass

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Find :math:`u : \mathbb{R} \times \Omega \to \mathbb{R}^+` such that

        .. math::

            \frac{\partial u}{\partial t} + \nabla \cdot \left( \mathbf{F}(u) u \right) = 0, \qquad \mathbf{F}(u) = -\mathbb{D} \frac{\nabla u}{u}

        where :math:`\mathbb{D} : \Omega \to \mathbb{R}^{3 \times 3}` is a positive diffusion matrix.
        At the moment only matrices of the form :math:`D * Id` are implemented, where :math:`D > 0`
        is a positive diffusion coefficient.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The diffusion coefficient defines the normalization,

        .. math::

            \hat D = \hat x^2 / \hat t.

        No separate plasma velocity scale is used in this model."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - No default scalar diagnostics are defined by this model."""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.push_deterministic_diffusion.PushDeterministicDiffusion`
        """
        doc = rf"""**1. push_deterministic_diffusion.PushDeterministicDiffusion:**

    {PushDeterministicDiffusion.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""This is a particle discretization of diffusion where particles follow a
        deterministic drift derived from the current estimate of the density.
        It is intended mainly for diffusion-method development and verification
        rather than plasma dynamics."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a deterministic diffusion model:

        .. code-block:: python

            from struphy.models import DeterministicParticleDiffusion

            model = DeterministicParticleDiffusion()
            model.hydrogen.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - deterministic particle discretizations of diffusion
        - transport benchmarks with positive scalar densities
        - numerical comparison against stochastic diffusion methods"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - electromagnetic plasma dynamics
        - kinetic Vlasov problems in phase space
        - nonlinear fluid systems with pressure or momentum evolution
        - diffusion tensors outside the currently supported simplified forms"""
