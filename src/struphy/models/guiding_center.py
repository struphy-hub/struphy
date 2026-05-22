import copy
import logging

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import FunctionScalarPIC, KineticEnergyPIC, LostMarkersPIC, Scalars
from struphy.models.species import (
    ParticleSpecies,
)
from struphy.models.variables import PICVariable
from struphy.propagators.base import Propagator
from struphy.propagators.push_guiding_center_bx_estar import PushGuidingCenterBxEstar
from struphy.propagators.push_guiding_center_parallel import PushGuidingCenterParallel

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

    1. :class:`~struphy.propagators.push_guiding_center_bx_estar.PushGuidingCenterBxEstar`
    2. :class:`~struphy.propagators.push_guiding_center_parallel.PushGuidingCenterParallel`
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
            self.push_bxe = PushGuidingCenterBxEstar()
            self.push_parallel = PushGuidingCenterParallel()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        epsilon: float = None,
    ):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

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
            n_lost_particles=LostMarkersPIC(self.kinetic_ions.var),
        )

        if rank == 0:
            logger.info("Done.")

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "alfvén"

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Guiding-center Vlasov equation:

        .. math::

            \frac{\partial f}{\partial t} + \left[ v_\parallel \frac{\mathbf{B}^*}{B^*_\parallel} + \frac{\mathbf{E}^* \times \mathbf{b}_0}{B^*_\parallel} \right] \cdot \frac{\partial f}{\partial \mathbf{X}} + \left[ \frac{1}{\epsilon} \frac{\mathbf{B}^*}{B^*_\parallel} \cdot \mathbf{E}^* \right] \cdot \frac{\partial f}{\partial v_\parallel} = 0

        where :math:`f(\mathbf{X}, v_\parallel, \mu, t)` is the guiding center distribution and

        .. math::

            \mathbf{E}^* = -\epsilon \mu \nabla |B_0|, \qquad \mathbf{B}^* = \mathbf{B}_0 + \epsilon v_\parallel \nabla \times \mathbf{b}_0, \qquad B^*_\parallel = \mathbf{B}^* \cdot \mathbf{b}_0

        """

    @classmethod
    def doc_normalization(cls):
        r"""The reference speed is Alfvénic and the key small parameter is the
        normalized cyclotron time:

        .. math::

            \hat v = \hat v_A,\qquad \varepsilon = 1/(\hat\Omega_c \hat t).
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Parallel kinetic energy: ``en_fv``
        - Magnetic-moment energy contribution: ``en_fB``
        - Total particle energy: ``en_tot``
        - Lost markers: ``n_lost_particles``"""

    @classmethod
    def doc_discretization(cls):
        doc = rf"""**1. push_guiding_center_bx_estar.PushGuidingCenterBxEstar:**

    {PushGuidingCenterBxEstar.__doc__}

    **2. push_guiding_center_parallel.PushGuidingCenterParallel:**

    {PushGuidingCenterParallel.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""GuidingCenter is the reduced kinetic model used when fast gyromotion is
        averaged out and only guiding-center motion needs to be resolved. It
        evolves particles in a fixed magnetic background without any
        self-consistent field solve."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a guiding-center model:

        .. code-block:: python

            from struphy.models import GuidingCenter

            model = GuidingCenter()
            model.kinetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - test-particle guiding-center orbit calculations
        - strongly magnetized plasmas with scale separation from gyromotion
        - verification of 5D guiding-center pushers and diagnostics"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - self-consistent electric or magnetic field evolution
        - full-orbit particle dynamics with resolved gyrophase
        - collisional transport or source terms not present in the equation
        - fluid closures or MHD force balance"""

    def allocate_helpers(self, verbose: bool = False):
        pass

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
