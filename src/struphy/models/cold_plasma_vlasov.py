import logging

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.cold_plasma import ColdPlasma
from struphy.models.scalars import KineticEnergyPIC
from struphy.models.species import ParticleSpecies
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation.kernels.charge_density_0form import charge_density_0form
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.propagators.base import Propagator
from struphy.propagators.poisson_solve import PoissonSolve
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_vxb import PushVxB
from struphy.propagators.vlasov_ampere_coupling import VlasovAmpereCoupling

logger = logging.getLogger("struphy")


class ColdPlasmaVlasov(ColdPlasma):
    """Hybrid cold-plasma model: a cold electron fluid coupled with a hot kinetic species and Maxwell's equations.

    Extends :class:`~struphy.models.ColdPlasma` by a hot kinetic electron species (Vlasov equation) and its current
    in Ampère's law.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    charge_number: int
        Charge number of the cold (fluid) electron species (default: -1)
    mass_number: float
        Mass number of the cold (fluid) electron species (default: 1/1836)
    alpha: float, optional
        Dimensionless parameter: cold-species plasma frequency / cyclotron frequency. If None, computed from units and charge/mass numbers.
    epsilon: float, optional
        Normalized cyclotron period of the cold species. If None, computed from units and charge/mass numbers.
    hot_charge_number: int
        Charge number of the hot (kinetic) electron species (default: -1)
    hot_mass_number: float
        Mass number of the hot (kinetic) electron species (default: 1/1836)
    hot_epsilon: float, optional
        Normalized cyclotron period of the hot species. If None, computed from units and charge/mass numbers.
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Hybrid"

    ## additional species

    class EMFields(ColdPlasma.EMFields):
        """Fields of :class:`~struphy.models.Maxwell` plus the potential ``phi`` of the initial Poisson solve."""

        def __init__(self):
            self.phi = FEECVariable(space="H1")
            super().__init__()

    class HotElectrons(ParticleSpecies):
        def __init__(
            self,
            charge_number: int = -1,
            mass_number: float = 1 / 1836,
            epsilon: float = None,
        ):
            self.var = PICVariable(space="Particles6D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                epsilon=epsilon,
            )

    ## additional propagators

    class Propagators(ColdPlasma.Propagators):
        """Propagators of :class:`~struphy.models.ColdPlasma` plus ``push_eta`` (streaming), ``push_vxb`` (Lorentz
        force of the background field) and ``coupling_va`` (electric force and current of the hot species).

        Parameters
        ----------
        **kwargs
            Passed to :class:`ColdPlasma.Propagators <struphy.models.ColdPlasma.Propagators>`.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.push_eta = PushEta()
            self.push_vxb = PushVxB()
            self.coupling_va = VlasovAmpereCoupling()

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    em_fields: EMFields
    hot_electrons: HotElectrons
    propagators: Propagators

    # names before the Maxwell family (old run metadata)
    _renamed_species = {"thermal_elec": "electrons", "hot_elec": "hot_electrons"}
    _renamed_params = {
        "thermal_charge_number": "charge_number",
        "thermal_mass_number": "mass_number",
        "thermal_alpha": "alpha",
        "thermal_epsilon": "epsilon",
    }

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = -1,
        mass_number: float = 1 / 1836,
        alpha: float = None,
        epsilon: float = None,
        hot_charge_number: int = -1,
        hot_mass_number: float = 1 / 1836,
        hot_epsilon: float = None,
    ):
        # 0. store input parameters
        self._store_params(locals())
        # for the hooks: the arguments of this constructor
        self._hot_args = {
            "charge_number": hot_charge_number,
            "mass_number": hot_mass_number,
            "epsilon": hot_epsilon,
        }

        # 1.-5. see Maxwell
        super().__init__(base_units, charge_number, mass_number, alpha, epsilon)

        # initial Poisson (not a propagator used in time stepping)
        hot_alpha = self.hot_electrons.equation_params.alpha
        hot_epsilon = self.hot_electrons.equation_params.epsilon
        particles_to_grid = ParticlesToGrid(
            self.hot_electrons.var,
            "H1",
            charge_density_0form,
        )

        self.initial_poisson = PoissonSolve(
            rho=particles_to_grid,
            rho_coeffs=hot_alpha**2 / hot_epsilon,
            delta_f_source=True,
        )
        self.initial_poisson.variables.phi = self.em_fields.phi

    def _create_species(self):
        """:meta private:"""
        super()._create_species()
        self.hot_electrons = self.HotElectrons(**self._hot_args)

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.push_eta.variables.var = self.hot_electrons.var
        self.propagators.push_vxb.variables.ions = self.hot_electrons.var

        self.propagators.coupling_va.variables.e = self.em_fields.e_field
        self.propagators.coupling_va.variables.ions = self.hot_electrons.var

    def _create_scalars(self):
        """:meta private:"""
        scalars = super()._create_scalars()
        # en_tot of ColdPlasma is replaced by the one including the hot-particle energy (and kept last)
        en_tot = scalars.pop("en_tot")
        en_f = KineticEnergyPIC(
            self.hot_electrons.var,
            normalization=self.hot_electrons.equation_params.alpha**2,
        )
        scalars["en_f"] = en_f
        scalars["en_tot"] = en_tot + en_f
        return scalars

    @property
    def velocity_scale(self):
        return "light"

    def post_allocate(self):
        """Solve initial Poisson equation.

        :meta private:
        """
        logger.info("\nINITIAL POISSON SOLVE:")

        # The right-hand side is the charge of f - f0, deposited with the delta-f weights also for full-f markers
        # (delta_f_source=True, the full-f weights are restored after the deposit): much less noise than the full-f
        # weights minus the background n0.
        self.initial_poisson.allocate()

        # Solve with dt=1. and compute electric field
        logger.info("\nSolving initial Poisson problem...")
        self.initial_poisson(1.0)

        phi = self.initial_poisson.variables.phi.spline.vector
        Propagator.derham.grad.dot(-phi, out=self.em_fields.e_field.spline.vector)
        # The pushers evaluate e at the markers, including in the ghost cells: grad.dot does not fill them,
        # so without this sync the first time step pushes with a wrong field (and breaks energy conservation).
        self.em_fields.e_field.spline.vector.update_ghost_regions()
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
        return params_path

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

            -\frac{\partial \mathbf{E}}{\partial t} + \nabla \times \mathbf{B} = \frac{\alpha_\textnormal{c}^2}{\varepsilon_\textnormal{c}} \mathbf{j}_\textnormal{c} + \frac{\alpha_\textnormal{h}^2}{\varepsilon_\textnormal{h}} \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v}

        where :math:`(n_0, \mathbf{B}_0)` denotes an inhomogeneous background, and :math:`\alpha_\textnormal{c},
        \varepsilon_\textnormal{c}` and :math:`\alpha_\textnormal{h}, \varepsilon_\textnormal{h}` are the
        parameters of the cold and the hot species.

        At initial time the Poisson equation is solved once to weakly satisfy the Gauss law:

        .. math::

            \nabla \cdot \mathbf{E} = \frac{\alpha_\textnormal{h}^2}{\varepsilon_\textnormal{h}} \int_{\mathbb{R}^3} (f - f_0) \, \text{d}^3 \mathbf{v}

        **Conserved quantities** (for :math:`\mathbf E \times \mathbf n = 0` on the boundary, or periodic
        boundaries):

        Energy:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \left[ \frac 12 \int \left( |\mathbf E|^2 + |\mathbf B|^2
            + \frac{\alpha_\textnormal{c}^2}{n_0} |\mathbf j_\textnormal{c}|^2 \right) \textnormal d \mathbf x
            + \frac{\alpha_\textnormal{h}^2}{2} \int \int_{\mathbb{R}^3} |\mathbf v|^2 f \, \text{d}^3 \mathbf{v}
            \, \textnormal d \mathbf x \right] = 0
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
        - Total energy: ``en_tot`` (sum of the above)"""

    @classmethod
    def doc_long_description(cls):
        r"""ColdPlasmaVlasov is an electromagnetic hybrid model that keeps a cold
        fluid response for the thermal electrons and a kinetic PIC description
        for a hot species. It is designed for problems where the energetic
        population matters kinetically but the bulk response can still be
        treated as cold.

        In the Maxwell family it extends :class:`~struphy.models.ColdPlasma` by the hot species and inherits its
        species (``em_fields``, ``electrons``), propagators and scalars."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the cold-plasma plus Vlasov model:

        .. code-block:: python

            from struphy.models import ColdPlasmaVlasov

            model = ColdPlasmaVlasov()
            model.em_fields.e_field
            model.em_fields.b_field
            model.electrons.current
            model.hot_electrons.var
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
