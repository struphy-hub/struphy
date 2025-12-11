
import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.block import BlockVector
from feectools.linalg.stencil import StencilVector

from struphy.feec.projectors import L2Projector
from struphy.feec.variational_utilities import (
    H1vecMassMatrix_density,
    InternalEnergyEvaluator,
)
from struphy.kinetic_background.base import KineticBackground
from struphy.kinetic_background.maxwellians import Maxwellian3D
from struphy.models.base import StruphyModel
from struphy.models.species import (
    DiagnosticSpecies,
    FieldSpecies,
    FluidSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.polar.basic import PolarVector
from struphy.propagators import (
    propagators_coupling,
    propagators_fields,
    propagators_markers,
)
from struphy.utils.pyccel import Pyccelkernel

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

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.hydrogen = self.Hydrogen()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.rand_diff.variables.var = self.hydrogen.var

        # define scalars for update_scalar_quantities
        # self.add_scalar("electric energy")
        # self.add_scalar("magnetic energy")
        # self.add_scalar("total energy")

    @property
    def bulk_species(self):
        return self.hydrogen

    @property
    def velocity_scale(self):
        return None

    def allocate_helpers(self):
        pass

    def update_scalar_quantities(self):
        pass
