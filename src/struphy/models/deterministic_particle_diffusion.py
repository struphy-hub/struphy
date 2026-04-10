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


class DeterministicParticleDiffusion(StruphyModel):
    r"""Diffusion equation discretized with a deterministic particle method;
    the solution is :math:`L^2`-projected onto :math:`V^0 \subset H^1` to compute the flux.

    :ref:`normalization`:

    .. math::

        \hat D := \frac{\hat x^2}{\hat t } \,.

    :ref:`Equations <gempic>`: Find :math:`u:\mathbb R\times \Omega\to \mathbb R^+` such that

    .. math::

        \frac{\partial u}{\partial t} +  \nabla \cdot\left(\mathbf F(u) u\right) = 0\,, \qquad \mathbf F(u) = -\mathbb D\,\frac{\nabla u}{u}\,,

    where :math:`\mathbb D: \Omega\to \mathbb R^{3\times 3 }` is a positive diffusion matrix.
    At the moment only matrices of the form :math:`D*Id` are implemented, where :math:`D > 0` is a positive diffusion coefficient.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushDeterministicDiffusion`

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
            self.det_diff = propagators_markers.PushDeterministicDiffusion()

    ## abstract methods

    def __init__(self, base_units: BaseUnits = BaseUnits()):

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

    def allocate_helpers(self, verbose: bool = False):
        pass