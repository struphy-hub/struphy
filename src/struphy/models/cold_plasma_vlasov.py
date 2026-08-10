import copy
import logging

from cunumpy import PyccelKernel

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, KineticEnergyPIC, Scalars
from struphy.models.species import (
    FieldSpecies,
    FluidSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation import accum_kernels
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.propagators.base import Propagator
from struphy.propagators.jxb_cold import JxBCold
from struphy.propagators.maxwell_weak_ampere import MaxwellWeakAmpere
from struphy.propagators.ohm_cold import OhmCold
from struphy.propagators.poisson_solve import PoissonSolve
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_vxb import PushVxB
from struphy.propagators.vlasov_ampere_coupling import VlasovAmpereCoupling

logger = logging.getLogger("struphy")


class ColdPlasmaVlasov(StruphyModel):
    """Hybrid cold-plasma model: a cold electron fluid coupled with a hot kinetic species and Maxwell's equations.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    thermal_charge_number: int
        Charge number of the cold (fluid) electron species (default: -1)
    thermal_mass_number: float
        Mass number of the cold (fluid) electron species (default: 1/1836)
    hot_charge_number: int
        Charge number of the hot (kinetic) electron species (default: -1)
    hot_mass_number: float
        Mass number of the hot (kinetic) electron species (default: 1/1836)
    thermal_alpha: float, optional
        Dimensionless parameter: cold-species plasma frequency / cyclotron frequency. If None, computed from units and charge/mass numbers.
    thermal_epsilon: float, optional
        Normalized cyclotron period of the cold species. If None, computed from units and charge/mass numbers.
    hot_epsilon: float, optional
        Normalized cyclotron period of the hot species. If None, computed from units and charge/mass numbers.
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Hybrid"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class ThermalElectrons(FluidSpecies):
        def __init__(
            self,
            charge_number: int,
            mass_number: float,
            alpha: float,
            epsilon: float,
        ):
            self.current = FEECVariable(space="Hcurl")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                alpha=alpha,
                epsilon=epsilon,
            )

    class HotElectrons(ParticleSpecies):
        def __init__(
            self,
            charge_number: int,
            mass_number: float,
            epsilon: float,
        ):
            self.var = PICVariable(space="Particles6D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                epsilon=epsilon,
            )

    ## propagators

    class Propagators:
        def __init__(self):
            self.maxwell = MaxwellWeakAmpere()
            self.ohm = OhmCold()
            self.jxb = JxBCold()
            self.push_eta = PushEta()
            self.push_vxb = PushVxB()
            self.coupling_va = VlasovAmpereCoupling()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        thermal_charge_number: int = -1,
        thermal_mass_number: float = 1 / 1836,
        hot_charge_number: int = -1,
        hot_mass_number: float = 1 / 1836,
        thermal_alpha: float = None,
        thermal_epsilon: float = None,
        hot_epsilon: float = None,
    ):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.thermal_elec = self.ThermalElectrons(
            thermal_charge_number,
            thermal_mass_number,
            thermal_alpha,
            thermal_epsilon,
        )
        self.hot_elec = self.HotElectrons(
            hot_charge_number,
            hot_mass_number,
            hot_epsilon,
        )

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field

        self.propagators.ohm.variables.j = self.thermal_elec.current
        self.propagators.ohm.variables.e = self.em_fields.e_field

        self.propagators.jxb.variables.j = self.thermal_elec.current

        self.propagators.push_eta.variables.var = self.hot_elec.var
        self.propagators.push_vxb.variables.ions = self.hot_elec.var

        self.propagators.coupling_va.variables.e = self.em_fields.e_field
        self.propagators.coupling_va.variables.ions = self.hot_elec.var

        # 5. define scalars to be tracked during simulation
        electric_energy = BilinearEnergyFEEC(self.em_fields.e_field)
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        current_energy = BilinearEnergyFEEC(
            self.thermal_elec.current,
            bilinear_form_name="M1ninv",
            normalization=self.thermal_elec.equation_params.alpha**2,
        )
        particle_energy = KineticEnergyPIC(
            self.hot_elec.var,
            normalization=self.hot_elec.equation_params.alpha**2,
        )
        self.scalars = Scalars(
            en_E=electric_energy,
            en_B=magnetic_energy,
            en_J=current_energy,
            en_f=particle_energy,
            en_tot=electric_energy + magnetic_energy + current_energy + particle_energy,
        )

        # initial Poisson (not a propagator used in time stepping)
        hot_alpha = self.hot_elec.equation_params.alpha
        hot_epsilon = self.hot_elec.equation_params.epsilon
        particles_to_grid = ParticlesToGrid(
            self.hot_elec.var,
            "H1",
            PyccelKernel(accum_kernels.charge_density_0form),
        )

        self.initial_poisson = PoissonSolve(
            rho=particles_to_grid,
            rho_coeffs=hot_alpha**2 / hot_epsilon,
        )
        self.initial_poisson.variables.phi = self.em_fields.phi

    @property
    def bulk_species(self):
        return self.thermal_elec

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self):
        """Solve initial Poisson equation.

        :meta private:
        """
        logger.info("\nINITIAL POISSON SOLVE:")

        # use control variate method
        particles = self.hot_elec.var.particles
        particles.update_weights()

        self.initial_poisson.allocate()

        # Solve with dt=1. and compute electric field
        logger.info("\nSolving initial Poisson problem...")
        self.initial_poisson(1.0)

        phi = self.initial_poisson.variables.phi.spline.vector
        Propagator.derham.grad.dot(-phi, out=self.em_fields.e_field.spline.vector)
        logger.info("... Done.")

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "coupling_va.Options" in line:
                    new_file += [line]
                    new_file += ["model.initial_poisson.options = model.initial_poisson.Options()\n"]
                elif "saving_params = " in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["saving_params = SavingParameters(binning_plots=(binplot,))\n\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Hot Vlasov species:

        .. math::

            \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \frac{1}{\varepsilon_\textnormal{h}} \Big[ \mathbf{E} + \mathbf{v} \times \left( \mathbf{B} + \mathbf{B}_0 \right) \Big] \cdot \frac{\partial f}{\partial \mathbf{v}} = 0

        Cold-plasma current:

        .. math::

            \frac{1}{n_0} \frac{\partial \mathbf{j}_\textnormal{c}}{\partial t} = \frac{1}{\varepsilon_\textnormal{c}} \mathbf{E} + \frac{1}{\varepsilon_\textnormal{c} n_0} \mathbf{j}_\textnormal{c} \times \mathbf{B}_0

        Faraday's law:

        .. math::

            \frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0

        Ampère's law:

        .. math::

            -\frac{\partial \mathbf{E}}{\partial t} + \nabla \times \mathbf{B} = \frac{\alpha^2}{\varepsilon_\textnormal{h}} \left( \mathbf{j}_\textnormal{c} + \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v} \right)

        where :math:`(n_0, \mathbf{B}_0)` denotes an inhomogeneous background.

        At initial time the Poisson equation is solved once to weakly satisfy the Gauss law:

        .. math::

            \nabla \cdot \mathbf{E} = \nu \frac{\alpha^2}{\varepsilon_\textnormal{h}} \int_{\mathbb{R}^3} f \, \text{d}^3 \mathbf{v}
        """

    @classmethod
    def doc_normalization(cls):
        r"""Velocities are normalized with the speed of light and the kinetic
        background is scaled as

        .. math::

            \hat v = c,\qquad \hat E = c \hat B,\qquad \hat f = \hat n / c^3.

        The model distinguishes cold and hot cyclotron scales through
        :math:`\varepsilon_\mathrm{c}` and :math:`\varepsilon_\mathrm{h}`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Electric field energy: ``en_E``
        - Magnetic field energy: ``en_B``
        - Cold-current energy: ``en_J``
        - Hot-particle kinetic energy: ``en_f``
        - Total energy: ``en_tot``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.maxwell_weak_ampere.MaxwellWeakAmpere`
        2. :class:`~struphy.propagators.ohm_cold.OhmCold`
        3. :class:`~struphy.propagators.jxb_cold.JxBCold`
        4. :class:`~struphy.propagators.push_eta.PushEta`
        5. :class:`~struphy.propagators.push_vxb.PushVxB`
        6. :class:`~struphy.propagators.vlasov_ampere_coupling.VlasovAmpereCoupling`
        """
        doc = rf"""**1. MaxwellWeakAmpere:**

{MaxwellWeakAmpere.__doc__}

**2. OhmCold:**

{OhmCold.__doc__}

**3. JxBCold:**

{JxBCold.__doc__}

**4. PushEta:**

{PushEta.__doc__}

**5. PushVxB:**

{PushVxB.__doc__}

**6. VlasovAmpereCoupling:**

{VlasovAmpereCoupling.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""ColdPlasmaVlasov is an electromagnetic hybrid model that keeps a cold
        fluid response for the thermal electrons and a kinetic PIC description
        for a hot species. It is designed for problems where the energetic
        population matters kinetically but the bulk response can still be
        treated as cold."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the cold-plasma plus Vlasov model:

        .. code-block:: python

            from struphy.models import ColdPlasmaVlasov

            model = ColdPlasmaVlasov()
            model.em_fields.e_field
            model.em_fields.b_field
            model.thermal_elec.current
            model.hot_elec.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - hybrid electromagnetic problems with one hot kinetic species
        - energetic-particle interaction with a cold background plasma
        - reduced-cost alternatives to fully kinetic multi-population models
        - benchmarks of fluid-kinetic current coupling"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - fully warm-fluid or pressure-anisotropic background dynamics
        - all-species kinetic simulations
        - collision operators or detailed dissipative closures
        - electrostatic-only reductions where magnetic evolution is irrelevant"""
