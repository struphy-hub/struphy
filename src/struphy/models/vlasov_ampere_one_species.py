import copy
import logging

from struphy import BaseUnits
from struphy.models.initial_poisson import InitialPoissonMixin
from struphy.models.scalars import BilinearEnergyFEEC, KineticEnergyPIC, Scalars
from struphy.models.species import FieldSpecies
from struphy.models.variables import FEECVariable, PICVariable
from struphy.models.vlasov import Vlasov
from struphy.propagators.vlasov_ampere_coupling import VlasovAmpereCoupling

logger = logging.getLogger("struphy")


class VlasovAmpereOneSpecies(InitialPoissonMixin, Vlasov):
    """Vlasov-Ampère system for a single kinetic species in an electric field.

    Extends :class:`~struphy.models.Vlasov` by the self-consistent electric field (Ampère's law).

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    charge_number: int
        Charge number (in units of the positive elementary charge) of the ion species (default: 1)
    mass_number: float
        Mass number (in units or Proton mass) of the ion species (default: 1.0)
    alpha : float, optional
        Dimensionless parameter: plasma frequency / cyclotron frequency.
        If None, computed from units and charge/mass numbers.
    epsilon : float, optional
        Normalized cyclotron period: 1 / (cyclotron frequency × time unit).
        If None, computed from units and charge/mass numbers.
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class KineticIons(Vlasov.KineticIons):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            alpha: float = None,
            epsilon: float = None,
        ):
            self.var = PICVariable(space="Particles6D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                alpha=alpha,
                epsilon=epsilon,
            )

    ## propagators

    class Propagators(Vlasov.Propagators):
        splitting_order = ("push_eta", "push_vxb", "coupling_va")

        def __init__(self):
            super().__init__()
            self.coupling_va = VlasovAmpereCoupling()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        alpha: float = None,
        epsilon: float = None,
    ):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons(
            charge_number,
            mass_number,
            alpha,
            epsilon,
        )

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self._link_propagator_variables()

        # 5. define scalars to be tracked during simulation
        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon

        electric_energy = BilinearEnergyFEEC(self.em_fields.e_field)
        kinetic_energy = KineticEnergyPIC(
            self.kinetic_ions.var,
            normalization=alpha**2,
        )
        total_energy = electric_energy + kinetic_energy

        self.scalars = Scalars(
            electric_energy=electric_energy,
            kinetic_energy=kinetic_energy,
            total_energy=total_energy,
        )

        # initial Poisson (not a propagator used in time stepping)
        self._create_initial_poisson(rho_coeffs=alpha**2 / epsilon)

    def _link_propagator_variables(self):
        """Assign the species variables to the propagators.

        :meta private:
        """
        super()._link_propagator_variables()
        self.propagators.coupling_va.variables.e = self.em_fields.e_field
        self.propagators.coupling_va.variables.ions = self.kinetic_ions.var

    @property
    def velocity_scale(self):
        return "light"

    def post_allocate(self):
        """Solve initial Poisson equation.

        :meta private:
        """
        # The right-hand side is the charge of f - f0, deposited with the delta-f weights also for full-f markers
        # (delta_f_source=True, the full-f weights are restored after the deposit): much less noise than the full-f
        # weights minus the background n0.
        self._solve_initial_poisson()

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Vlasov equation:

        .. math::

            \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \frac{1}{\varepsilon} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B}_0 \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0

        Ampère's law:

        .. math::

            -\frac{\partial \mathbf{E}}{\partial t} = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f \, \mathrm{d}^3 \mathbf{v}

        Initial Poisson equation: At :math:`t=0`, solve weakly for the electric potential :math:`\phi`:

        .. math::

            \int_{\Omega} \nabla \psi^{\top} \cdot \nabla \phi \, \mathrm{d} \mathbf{x} = \frac{\alpha^2}{\varepsilon} \int_{\Omega} \int_{\mathbb{R}^3} \psi \, (f - f_0) \, \mathrm{d}^3 \mathbf{v} \, \mathrm{d} \mathbf{x} \qquad \forall \ \psi \in H^1
            \\[2mm]
            \mathbf{E}(t=0) = -\nabla \phi(t=0)
        """

    @classmethod
    def doc_normalization(cls):
        r"""Velocity and field normalizations:

        .. math::

            \hat{v} = c, \qquad \hat{E} = \hat{B} \hat{v}, \qquad \hat{\phi} = \hat{E} \hat{x}

        Dimensionless parameters:

        .. math::

            \alpha = \frac{\hat{\omega}_\mathrm{p}}{\hat{\omega}_\mathrm{c}}, \qquad \varepsilon = \frac{1}{\hat{\omega}_\mathrm{c} \hat{t}}

        where

        .. math::

            \hat{\omega}_\mathrm{p} = \sqrt{\frac{\hat{n} (Ze)^2}{\epsilon_0 (A m_\mathrm{H})}}, \qquad \hat{\omega}_\mathrm{c} = \frac{(Ze) \hat{B}}{(A m_\mathrm{H})}

        For electrons: :math:`Z = -1`, :math:`A = 1/1836`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Electric field energy: :math:`E_E = \frac{1}{2} \int |\mathbf{E}|^2 \, \mathrm{d}V`
        - Kinetic energy: :math:`E_f = \frac{\alpha^2}{2N} \sum_p w_p |\mathbf{v}_p|^2`
        - Total energy: :math:`E_\mathrm{tot} = E_E + E_f`"""

    @classmethod
    def doc_long_description(cls):
        r"""This model couples the kinetic Vlasov equation for the particle distribution function
        with Ampère's law for the electric field evolution. It includes the effect of a static background
        magnetic field :math:`\mathbf{B}_0` and solves the initial Poisson equation to satisfy Gauss's law
        at :math:`t=0`. The model uses a particle-in-cell (PIC) method with finite element exterior
        calculus (FEEC) for the electromagnetic fields.

        All field variables are perturbations around a reference equilibrium distribution. The model enables
        studies of kinetic instabilities, particle-wave interactions, and nonlinear plasma physics without
        assuming a fluid approximation."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a Vlasov-Ampère model:

        .. code-block:: python

            from struphy.models import VlasovAmpereOneSpecies

            model = VlasovAmpereOneSpecies()

            # Access fields
            model.em_fields.e_field
            model.em_fields.phi
            model.kinetic_ions.var

            # Access tracked scalar quantities
            model.scalars["electric_energy"]
            model.scalars["kinetic_energy"]
            model.scalars["total_energy"]
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - studying kinetic instabilities in collisionless plasmas
        - particle-wave interactions and nonlinear plasma physics
        - initial value problems with prescribed equilibrium distributions
        - benchmark problems for kinetic-field coupling schemes
        - verification of PIC methods in simplified geometries"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - multi-species systems (requires extension)
        - collision operator effects
        - electromagnetic wave propagation (no magnetic field evolution)
        - drift-reduced or gyrokinetic approximations (full 6D Vlasov equation)"""
