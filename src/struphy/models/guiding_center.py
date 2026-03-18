import logging


import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
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
        def __init__(self):
            self.var = PICVariable(space="Particles5D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.push_bxe = propagators_markers.PushGuidingCenterBxEstar()
            self.push_parallel = propagators_markers.PushGuidingCenterParallel()

    ## abstract methods

    def __init__(self):

        # 1. instantiate all species
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.push_bxe.variables.ions = self.kinetic_ions.var
        self.propagators.push_parallel.variables.ions = self.kinetic_ions.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_fv", compute="from_particles", variable=self.kinetic_ions.var)
        self.add_scalar("en_fB", compute="from_particles", variable=self.kinetic_ions.var)
        self.add_scalar("en_tot", compute="from_particles", variable=self.kinetic_ions.var)

        if rank == 0:
            logger.info("Done.")

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self, verbose: bool = False):
        self._en_fv = xp.empty(1, dtype=float)
        self._en_fB = xp.empty(1, dtype=float)
        self._en_tot = xp.empty(1, dtype=float)
        self._n_lost_particles = xp.empty(1, dtype=float)

    def update_scalar_quantities(self):
        particles = self.kinetic_ions.var.particles

        # particles' kinetic energy
        self._en_fv[0] = particles.markers[~particles.holes, 5].dot(
            particles.markers[~particles.holes, 3] ** 2,
        ) / (2.0 * particles.Np)

        particles.save_magnetic_background_energy()
        self._en_tot[0] = (
            particles.markers[~particles.holes, 5].dot(
                particles.markers[~particles.holes, 8],
            )
            / particles.Np
        )

        self._en_fB[0] = self._en_tot[0] - self._en_fv[0]

        self.update_scalar("en_fv", self._en_fv[0])
        self.update_scalar("en_fB", self._en_fB[0])
        self.update_scalar("en_tot", self._en_tot[0])

        self._n_lost_particles[0] = particles.n_lost_markers
        Propagator.derham.comm.Allreduce(
            MPI.IN_PLACE,
            self._n_lost_particles,
            op=MPI.SUM,
        )
