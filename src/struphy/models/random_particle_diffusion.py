from feectools.ddm.mpi import mpi as MPI

from struphy import BaseUnits
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


class RandomParticleDiffusion(StruphyModel):
    r"""Diffusion equation discretized with a (random) particle method;
    the diffusion is computed through a Wiener process.

    :ref:`normalization`:

    .. math::

        \hat D := \frac{\hat x^2}{\hat t } \,.

    :ref:`Equations <gempic>`: Find :math:`u:\mathbb R\times \Omega\to \mathbb R^+` such that

    .. math::

        \frac{\partial u}{\partial t} -  D \, \Delta u = 0\,,

    where :math:`D > 0` is a positive diffusion coefficient.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushRandomDiffusion`

    :ref:`Model info <add_model>`:
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
            self.rand_diff = propagators_markers.PushRandomDiffusion()

    ## abstract methods

    def __init__(self, base_units: BaseUnits = BaseUnits()):

        # 1. instantiate all species
        self.hydrogen = self.Hydrogen()

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.rand_diff.variables.var = self.hydrogen.var

        # 5. define scalars to be tracked during simulation

    @property
    def bulk_species(self):
        return self.hydrogen

    @property
    def velocity_scale(self):
        return None

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Find :math:`u : \mathbb{R} \times \Omega \to \mathbb{R}^+` such that

        .. math::

            \frac{\partial u}{\partial t} - D \, \Delta u = 0

        where :math:`D > 0` is a positive diffusion coefficient.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The natural scaling is set by the diffusion coefficient:

        .. math::

            \hat D = \hat x^2 / \hat t.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - No default scalar diagnostics are defined by this model."""

    @classmethod
    def doc_discretization(cls):
        doc = rf"""**1. propagators_markers.PushRandomDiffusion:**

{propagators_markers.PushRandomDiffusion.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""This model is the stochastic counterpart of the deterministic particle
        diffusion model. It is intended for diffusion-method development and
        comparisons between random-walk and deterministic transport strategies."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a random diffusion model:

        .. code-block:: python

            from struphy.models import RandomParticleDiffusion

            model = RandomParticleDiffusion()
            model.hydrogen.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - stochastic particle diffusion benchmarks
        - Monte-Carlo transport verification
        - comparison against deterministic diffusion solvers"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - electromagnetic or fluid plasma dynamics
        - deterministic advection-dominated transport
        - anisotropic plasma kinetics in phase space"""

    def allocate_helpers(self, verbose: bool = False):
        pass
