import logging

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import FunctionScalarPIC, Scalars, KineticEnergyPIC
from struphy.models.species import (
    ParticleSpecies,
)
from struphy.models.variables import PICVariable
from struphy.propagators import (
    propagators_markers,
)
from struphy.propagators.base import Propagator

logger = logging.getLogger("struphy")
rank = MPI.COMM_WORLD.Get_rank()


class GuidingCenter(StruphyModel):
    r"""Guiding-center equation in static background magnetic field.

    :ref:`normalization`:

    .. math::

        \hat v = \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        \frac{\partial f}{\partial t} + \left[ v_\parallel \frac{\mathbf{B}^*}{B^*_\parallel} + \frac{\mathbf{E}^* \times \mathbf{b}_0}{B^*_\parallel}\right] \cdot \frac{\partial f}{\partial \mathbf{X}} + \left[\frac{1}{\epsilon} \frac{\mathbf{B}^*}{B^*_\parallel} \cdot \mathbf{E}^*\right] \cdot \frac{\partial f}{\partial v_\parallel} = 0\,.

    where :math:`f(\mathbf{X}, v_\parallel, \mu, t)` is the guiding center distribution and

    .. math::

        \mathbf{E}^* = -\epsilon \mu \nabla |B_0| \,,  \qquad \mathbf{B}^* = \mathbf{B}_0 + \epsilon v_\parallel \nabla \times \mathbf{b}_0 \,,\qquad B^*_\parallel = \mathbf B^* \cdot \mathbf b_0  \,.

    Moreover,

    .. math::

        \epsilon = \frac{1 }{ \hat \Omega_{\textnormal{c}} \hat t}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushGuidingCenterBxEstar`
    2. :class:`~struphy.propagators.propagators_markers.PushGuidingCenterParallel`
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
            epsilon: float = None,
        ):
            self.var = PICVariable(space="Particles5D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                epsilon=epsilon,
            )

    ## propagators

    class Propagators:
        def __init__(self):
            self.push_bxe = propagators_markers.PushGuidingCenterBxEstar()
            self.push_parallel = propagators_markers.PushGuidingCenterParallel()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        epsilon: float = None,
    ):

        # 1. instantiate all species
        self.kinetic_ions = self.KineticIons(
            charge_number,
            mass_number,
            epsilon,
        )

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.push_bxe.variables.ions = self.kinetic_ions.var
        self.propagators.push_parallel.variables.ions = self.kinetic_ions.var

        # 5. define scalars to be tracked during simulation
        kinetic_energy = KineticEnergyPIC(self.kinetic_ions.var)
        magnetic_energy = FunctionScalarPIC(self._compute_en_fB, self.kinetic_ions.var)
        self.scalars = Scalars(
            en_fv=kinetic_energy,
            en_fB=magnetic_energy,
            en_tot=kinetic_energy + magnetic_energy,
        )

        if rank == 0:
            logger.info("Done.")

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self, verbose: bool = False):
        self._n_lost_particles = xp.empty(1, dtype=float)

    def _compute_en_fB(self):
        particles = self.kinetic_ions.var.particles
        particles.save_magnetic_background_energy()
        energy = (
            particles.markers[~particles.holes, 5].dot(
                particles.markers[~particles.holes, 8],
            )
            / particles.Np
        )
        return energy

    def update_scalar_quantities(self):
        self.scalars.update()
        particles = self.kinetic_ions.var.particles
        self._n_lost_particles[0] = particles.n_lost_markers
        Propagator.derham.comm.Allreduce(
            MPI.IN_PLACE,
            self._n_lost_particles,
            op=MPI.SUM,
        )
