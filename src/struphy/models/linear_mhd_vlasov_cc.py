import logging

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.linear_mhd import LinearMHD
from struphy.models.scalars import KineticEnergyPIC, LostMarkersPIC
from struphy.models.species import ParticleSpecies
from struphy.models.variables import FEECVariable, PICVariable
from struphy.propagators.base import Propagator
from struphy.propagators.current_coupling_6d_current import CurrentCoupling6DCurrent
from struphy.propagators.current_coupling_6d_density import CurrentCoupling6DDensity
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_vxb import PushVxB

logger = logging.getLogger("struphy")


class LinearMHDVlasovCC(LinearMHD):
    """Hybrid linear MHD coupled with energetic ions (6D Vlasov) via the current-coupling scheme.

    Extends :class:`~struphy.models.LinearMHD` (magnetic field in H(div)) by energetic ions, whose density and current
    enter the MHD momentum equation.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mhd_mass_number: float
        Mass number (in units of Proton mass) of the MHD bulk species (default: 1.0)
    hot_charge_number: int
        Charge number (in units of the positive elementary charge) of the energetic ion species (default: 1)
    hot_mass_number: float
        Mass number (in units of Proton mass) of the energetic ion species (default: 1.0)
    hot_epsilon: float, optional
        Normalized cyclotron period of the energetic ion species. If None, computed from units and charge/mass numbers.
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Hybrid"

    ## additional species

    class EnergeticIons(ParticleSpecies):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            epsilon: float = None,
        ):
            self.var = PICVariable(space="Particles6D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                epsilon=epsilon,
            )

    ## additional propagators

    class Propagators(LinearMHD.Propagators):
        """Propagators of :class:`~struphy.models.LinearMHD` plus the current coupling (``couple_dens``,
        ``couple_curr``) and the energetic-ion pushers (``push_eta``, ``push_vxb``).

        Parameters
        ----------
        b_field : FEECVariable, optional
            Magnetic field perturbation (default: None).
        energetic_ions_var : PICVariable, optional
            Markers of the energetic ions, deposited by ``couple_dens`` (default: None).
        **kwargs
            Passed to :class:`LinearMHD.Propagators <struphy.models.LinearMHD.Propagators>`.
        """

        splitting_order = (
            "couple_dens",
            "shear_alf",
            "couple_curr",
            "push_eta",
            "push_vxb",
            "bg_current",
            "sound",
            "bg_pressure",
        )

        def __init__(
            self,
            b_field: FEECVariable = None,
            energetic_ions_var: PICVariable = None,
            **kwargs,
        ):
            super().__init__(b_field=b_field, **kwargs)
            self.couple_dens = CurrentCoupling6DDensity(energetic_ions=energetic_ions_var, b_tilde=b_field)
            self.couple_curr = CurrentCoupling6DCurrent(b_tilde=b_field)
            self.push_eta = PushEta()
            self.push_vxb = PushVxB()

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    energetic_ions: EnergeticIons
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mhd_mass_number: float = 1.0,
        hot_charge_number: int = 1,
        hot_mass_number: float = 1.0,
        hot_epsilon: float = None,
    ):
        # 0. store input parameters
        self._store_params(locals())

        # 1.-5. see LinearMHD (magnetic field in H(div))
        super().__init__(base_units, mass_number=mhd_mass_number)

    def _create_species(self):
        """:meta private:"""
        super()._create_species()
        self.energetic_ions = self.EnergeticIons(
            self.params["hot_charge_number"],
            self.params["hot_mass_number"],
            self.params["hot_epsilon"],
        )

    def _propagator_args(self):
        """:meta private:"""
        return {**super()._propagator_args(), "energetic_ions_var": self.energetic_ions.var}

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.couple_dens.variables.u = self.mhd.velocity

        self.propagators.couple_curr.variables.ions = self.energetic_ions.var
        self.propagators.couple_curr.variables.u = self.mhd.velocity

        self.propagators.push_eta.variables.var = self.energetic_ions.var
        self.propagators.push_vxb.variables.ions = self.energetic_ions.var

    def _create_scalars(self):
        """:meta private:"""
        scalars = super()._create_scalars()
        en_tot = scalars.pop("en_tot")
        Ab = self.mhd.mass_number
        Ah = self.energetic_ions.var.species.mass_number
        en_f = KineticEnergyPIC(self.energetic_ions.var, normalization=Ah / Ab)
        scalars["en_f"] = en_f
        scalars["en_tot"] = en_tot + en_f
        scalars["n_lost_particles"] = LostMarkersPIC(self.energetic_ions.var)
        return scalars

    def post_allocate(self):
        """:meta private:"""
        super().post_allocate()

        # add control variate to mass_ops object
        if self.energetic_ions.var.particles.control_variate:
            Propagator.mass_ops.weights["f0"] = self.energetic_ions.var.particles.f0

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "saving_params = " in line:
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

        MHD continuity:

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot (\rho_0 \tilde{\mathbf{U}}) = 0

        MHD momentum:

        .. math::

            \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde{p} = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B}_0 + \mathbf{J}_0 \times \tilde{\mathbf{B}} \color{blue} + \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon} \left( n_\textnormal{h} \tilde{\mathbf{U}} - n_\textnormal{h} \mathbf{u}_\textnormal{h} \right) \times (\mathbf{B}_0 + \tilde{\mathbf{B}}) \color{black}

        MHD pressure:

        .. math::

            \frac{\partial \tilde{p}}{\partial t} + \nabla \cdot (p_0 \tilde{\mathbf{U}}) + (\gamma - 1) p_0 \nabla \cdot \tilde{\mathbf{U}} = 0

        MHD induction:

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} = \nabla \times (\tilde{\mathbf{U}} \times \mathbf{B}_0), \qquad \nabla \cdot \tilde{\mathbf{B}} = 0

        Energetic-particle Vlasov equation:

        .. math::

            \frac{\partial f_\textnormal{h}}{\partial t} + \mathbf{v} \cdot \nabla f_\textnormal{h} + \frac{1}{\varepsilon} \left[ \color{blue} (\mathbf{B}_0 + \tilde{\mathbf{B}}) \times \tilde{\mathbf{U}} \color{black} + \mathbf{v} \times (\mathbf{B}_0 + \tilde{\mathbf{B}}) \right] \cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} = 0

        Energetic-particle moments:

        .. math::

            n_\textnormal{h} = \int_{\mathbb{R}^3} f_\textnormal{h} \, \textnormal{d}^3 \mathbf{v}, \qquad n_\textnormal{h} \mathbf{u}_\textnormal{h} = \int_{\mathbb{R}^3} f_\textnormal{h} \mathbf{v} \, \textnormal{d}^3 \mathbf{v}

        where :math:`\mathbf{J}_0 = \nabla\times\mathbf{B}_0`, :math:`\gamma = 5/3`, the equations of
        :class:`~struphy.models.LinearMHD` are extended by the energetic ions; the terms in blue couple them to the MHD
        equations.
        """

    @classmethod
    def doc_normalization(cls):
        r"""Both fluid and particle velocities are normalized with the Alfvén
        speed,

        .. math::

            \hat U = \hat v = \hat v_A,\qquad \hat f_h = \hat n / \hat v_A^3.

        The hot-species cyclotron parameter is :math:`\varepsilon =
        1/(\hat\Omega_{c,\mathrm{hot}}\hat t)`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - The scalars of :class:`~struphy.models.LinearMHD`: ``en_U``, ``en_p``, ``en_thermal``, ``en_p_eq``,
          ``en_B``, ``en_B_eq``, ``en_B_tot``
        - Energetic-particle kinetic energy: ``en_f``
        - Total energy: ``en_tot = en_U + en_thermal + en_B + en_f``
        - Lost particles: ``n_lost_particles``"""

    @classmethod
    def doc_long_description(cls):
        r"""LinearMHDVlasovCC is a current-coupling hybrid model for small
        perturbations around an MHD equilibrium. It is intended for energetic
        particle effects on linear MHD waves and instabilities without the cost
        of a fully kinetic background plasma."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the linear MHD-Vlasov current-coupling model:

        .. code-block:: python

            from struphy.models import LinearMHDVlasovCC

            model = LinearMHDVlasovCC()
            model.em_fields.b_field
            model.mhd.velocity
            model.energetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - linear energetic-particle coupling to MHD modes
        - current-coupling hybrid verification
        - reduced-cost studies of fast-ion effects on wave propagation"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - strongly nonlinear dynamics far from equilibrium
        - full multi-species kinetic plasma evolution
        - pressure-coupling closures
        - collisional or dissipative MHD effects not present in the equations"""
