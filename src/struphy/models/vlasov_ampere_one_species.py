import logging

from struphy import BaseUnits
from struphy.models.scalars import BilinearEnergyFEEC, KineticEnergyPIC
from struphy.models.species import FieldSpecies
from struphy.models.variables import FEECVariable
from struphy.models.vlasov import Vlasov
from struphy.propagators.vlasov_ampere_coupling import VlasovAmpereCoupling

logger = logging.getLogger("struphy")


class VlasovAmpereOneSpecies(Vlasov):
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

    ## additional species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    ## additional propagators

    class Propagators(Vlasov.Propagators):
        """Propagators of :class:`~struphy.models.Vlasov` plus ``coupling_va`` (electric force and Ampère's law).

        Parameters
        ----------
        **kwargs
            Passed to :class:`Vlasov.Propagators <struphy.models.Vlasov.Propagators>`.
        """

        splitting_order = ("push_eta", "push_vxb", "coupling_va")

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.coupling_va = VlasovAmpereCoupling()

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    em_fields: EMFields
    propagators: Propagators

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
        self._store_params(locals())

        # 1.-5. see Vlasov
        super().__init__(base_units, charge_number, mass_number)

        # initial Poisson (not a propagator used in time stepping)
        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon
        self._create_initial_poisson(rho_coeffs=alpha**2 / epsilon)

    def _create_species(self):
        """:meta private:"""
        self.em_fields = self.EMFields()
        super()._create_species()

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.coupling_va.variables.e = self.em_fields.e_field
        self.propagators.coupling_va.variables.ions = self.kinetic_ions.var

    def _create_scalars(self):
        """:meta private:"""
        alpha = self.kinetic_ions.equation_params.alpha
        en_E = BilinearEnergyFEEC(self.em_fields.e_field)
        en_f = KineticEnergyPIC(self.kinetic_ions.var, normalization=alpha**2)
        return {"en_E": en_E, "en_f": en_f, "en_tot": en_E + en_f}

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

        - Electric field energy ``en_E``: :math:`\frac{1}{2} \int |\mathbf{E}|^2 \, \mathrm{d}V`
        - Kinetic energy ``en_f``: :math:`\frac{\alpha^2}{2N} \sum_p w_p |\mathbf{v}_p|^2`
        - Total energy ``en_tot``: ``en_E + en_f``"""

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
            model.scalars["en_E"]
            model.scalars["en_f"]
            model.scalars["en_tot"]
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
