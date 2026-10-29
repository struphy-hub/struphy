import logging

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import FunctionScalarPIC, KineticEnergyPIC, LostMarkersPIC
from struphy.models.variables import FEECVariable
from struphy.models.vlasov import Vlasov
from struphy.propagators.push_guiding_center_bx_estar import PushGuidingCenterBxEstar
from struphy.propagators.push_guiding_center_parallel import PushGuidingCenterParallel

logger = logging.getLogger("struphy")


class GuidingCenter(StruphyModel):
    """Guiding-center equation for a single species in a static background magnetic field.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    charge_number: int
        Charge number (in units of the positive elementary charge) of the species (default: 1)
    mass_number: float
        Mass number (in units of Proton mass) of the species (default: 1.0)
    epsilon: float, optional
        Normalized cyclotron period: 1 / (cyclotron frequency × time unit). If None, computed from units and charge/mass numbers.
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

    ## species

    class KineticIons(Vlasov.KineticIons):
        space = "Particles5D"

    ## propagators

    class Propagators:
        """Propagators of the guiding-center equation: ``push_bxe`` (E* x b and grad-B drifts) and ``push_parallel``
        (parallel streaming and mirror force).

        Parameters
        ----------
        phi : FEECVariable, optional
            Electric potential, -grad phi in E* of both pushers (default: None).
        b_tilde : FEECVariable, optional
            Magnetic field (2-form) added to the equilibrium field in both pushers (default: None).
        e_field : FEECVariable, optional
            Electric field (1-form) in the E* x b drift of ``push_bxe`` only; the parallel electric force is then
            applied elsewhere (default: None).
        """

        def __init__(self, phi: FEECVariable = None, b_tilde: FEECVariable = None, e_field: FEECVariable = None):
            self.push_bxe = PushGuidingCenterBxEstar(phi=phi, b_tilde=b_tilde, e_field=e_field)
            self.push_parallel = PushGuidingCenterParallel(phi=phi, b_tilde=b_tilde)

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    kinetic_ions: KineticIons
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        epsilon: float = None,
    ):
        # 0. store input parameters (a model inheriting from GuidingCenter stores its own before super().__init__)
        self._store_params(locals())

        # 1.-5. species, units, propagators, variable links and scalars, see StruphyModel._setup
        self._setup(base_units)

    def _create_species(self):
        """:meta private:"""
        self.kinetic_ions = self.KineticIons(**self._particle_species_args())

    def _link_propagator_variables(self):
        """:meta private:"""
        self.propagators.push_bxe.variables.ions = self.kinetic_ions.var
        self.propagators.push_parallel.variables.ions = self.kinetic_ions.var

    def _create_scalars(self):
        """:meta private:"""
        en_f = KineticEnergyPIC(self.kinetic_ions.var)
        en_mu = FunctionScalarPIC(self._compute_en_fB, self.kinetic_ions.var)
        return {
            "en_f": en_f,
            "en_mu": en_mu,
            "en_tot": en_f + en_mu,
            "n_lost_particles": LostMarkersPIC(self.kinetic_ions.var),
        }

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "alfvén"

    def post_allocate(self):
        pass

    def _compute_en_fB(self):
        particles = self.kinetic_ions.var.particles
        particles.save_magnetic_background_energy()
        # The weights (column 5) already carry the 1/N_p of the Monte-Carlo estimate, see KineticEnergyPIC.
        energy = particles.markers[5, particles.valid_mks].dot(
            particles.markers[8, particles.valid_mks],
        )
        return energy

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

        - Parallel kinetic energy: ``en_f``
        - Magnetic-moment energy: ``en_mu``
        - Total particle energy: ``en_tot``
        - Lost markers: ``n_lost_particles``"""

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
