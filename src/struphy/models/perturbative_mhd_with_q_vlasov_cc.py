import logging

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.perturbative_mhd_with_q import PerturbativeMHDWithQ
from struphy.models.species import ParticleSpecies
from struphy.models.utils import add_binning_plot, add_energetic_ion_scalars
from struphy.models.variables import FEECVariable, PICVariable
from struphy.propagators.base import Propagator
from struphy.propagators.current_coupling_6d_current import CurrentCoupling6DCurrent
from struphy.propagators.current_coupling_6d_density import CurrentCoupling6DDensity
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_vxb import PushVxB

logger = logging.getLogger("struphy")


class PerturbativeMHDWithQVlasovCC(PerturbativeMHDWithQ):
    """Hybrid nonlinear MHD (perturbations of a static equilibrium, :math:`q = \\sqrt p`) coupled with energetic ions
    (6D Vlasov) via the current-coupling scheme.

    Extends :class:`~struphy.models.PerturbativeMHDWithQ` by energetic ions, whose density and current enter the MHD
    momentum equation, as in :class:`~struphy.models.LinearMHDVlasovCC`, with the inertia of the total density in the
    coupling terms. The magnetic field is in :math:`H(\\textnormal{div})`.

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

    class Propagators(PerturbativeMHDWithQ.Propagators):
        """Propagators of :class:`~struphy.models.PerturbativeMHDWithQ` plus the current coupling (``couple_dens``,
        ``couple_curr``, with the inertia of the total density) and the energetic-ion pushers (``push_eta``,
        ``push_vxb``).

        Parameters
        ----------
        rho_field : FEECVariable, optional
            Density perturbation, for the inertia of the total density (default: None).
        b_field : FEECVariable, optional
            Magnetic field perturbation, in the current coupling and added to the equilibrium field in ``push_vxb``
            (default: None).
        energetic_ions_var : PICVariable, optional
            Markers of the energetic ions, deposited by ``couple_dens`` (default: None).
        **kwargs
            Passed to :class:`PerturbativeMHDWithQ.Propagators <struphy.models.PerturbativeMHDWithQ.Propagators>`.
        """

        splitting_order = (
            "couple_dens",
            "shear_alf",
            "nonlin_alf",
            "couple_curr",
            "push_eta",
            "push_vxb",
            "bg_current",
            "bg_q",
            "sound",
            "nonlin_sound",
            "mom_adv",
        )

        def __init__(
            self,
            rho_field: FEECVariable = None,
            b_field: FEECVariable = None,
            energetic_ions_var: PICVariable = None,
            **kwargs,
        ):
            super().__init__(rho_field=rho_field, b_field=b_field, **kwargs)
            self.couple_dens = CurrentCoupling6DDensity(
                energetic_ions=energetic_ions_var, b_tilde=b_field, rho_field=rho_field
            )
            self.couple_curr = CurrentCoupling6DCurrent(b_tilde=b_field, rho_field=rho_field)
            self.push_eta = PushEta()
            # Lorentz force of the total field B0 + b_tilde
            self.push_vxb = PushVxB(b2_var=b_field)

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

        # 1.-5. see PerturbativeMHDWithQ
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
        add_energetic_ion_scalars(scalars, self.mhd, self.energetic_ions)
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
        add_binning_plot(params_path)
        return params_path

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        MHD continuity (mass conservation):

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot \left( (\rho_0 + \tilde \rho) \tilde{\mathbf{U}} \right) = 0

        MHD momentum:

        .. math::

            (\rho_0 + \tilde \rho) \left( \frac{\partial \tilde{\mathbf{U}}}{\partial t}
            + \tilde{\mathbf U} \cdot \nabla \tilde{\mathbf U} \right) + \nabla (2 q_0 \tilde q + \tilde q^2)
            = (\nabla \times \tilde{\mathbf{B}}) \times (\mathbf{B}_0 + \tilde{\mathbf B}) + (\nabla \times \mathbf{B}_0) \times \tilde{\mathbf{B}}
            \color{blue} + \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon} \left( n_\textnormal{h} \tilde{\mathbf{U}} - n_\textnormal{h} \mathbf{u}_\textnormal{h} \right) \times (\mathbf{B}_0 + \tilde{\mathbf{B}}) \color{black}

        Square root of the pressure (adiabatic process):

        .. math::

            \frac{\partial \tilde{q}}{\partial t} + \tilde{\mathbf{U}} \cdot \nabla (q_0 + \tilde q)
            + \frac{\gamma}{2} (q_0 + \tilde q) \nabla \cdot \tilde{\mathbf{U}} = 0

        MHD induction:

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times \left(\tilde{\mathbf{U}} \times (\mathbf{B}_0 + \tilde{\mathbf B})\right) = 0

        Energetic-particle Vlasov equation:

        .. math::

            \frac{\partial f_\textnormal{h}}{\partial t} + \mathbf{v} \cdot \nabla f_\textnormal{h} + \frac{1}{\varepsilon} \left[ \color{blue} (\mathbf{B}_0 + \tilde{\mathbf{B}}) \times \tilde{\mathbf{U}} \color{black} + \mathbf{v} \times (\mathbf{B}_0 + \tilde{\mathbf{B}}) \right] \cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} = 0

        Energetic-particle moments:

        .. math::

            n_\textnormal{h} = \int_{\mathbb{R}^3} f_\textnormal{h} \, \textnormal{d}^3 \mathbf{v}, \qquad n_\textnormal{h} \mathbf{u}_\textnormal{h} = \int_{\mathbb{R}^3} f_\textnormal{h} \mathbf{v} \, \textnormal{d}^3 \mathbf{v}

        Here, :math:`\tilde \rho`, :math:`\tilde{\mathbf U}`, :math:`\tilde q`, :math:`\tilde{\mathbf B}` are the
        perturbed density, velocity, square root of the pressure and magnetic field (not necessarily small), and
        :math:`\rho_0`, :math:`q_0 = \sqrt{p_0}`, :math:`\mathbf B_0` those of a static MHD equilibrium;
        :math:`\gamma = 5/3`. The equations of :class:`~struphy.models.PerturbativeMHDWithQ` are extended by the
        energetic ions (distribution function :math:`f_\textnormal{h}`, mass number :math:`A_\textnormal{h}`); the
        terms in blue couple them to the MHD equations. Dropping the terms quadratic in the MHD perturbations
        (:math:`\rho_0 + \tilde \rho \to \rho_0` etc., keeping :math:`\mathbf B_0 + \tilde{\mathbf B}` in the coupling)
        gives the equations of :class:`~struphy.models.LinearMHDVlasovCC` (with :math:`\tilde p = 2 q_0 \tilde q`;
        that model pushes the energetic ions with :math:`\mathbf B_0` only, unless ``push_vxb.b2_var`` is set).

        **Conserved quantities** (for :math:`\tilde{\mathbf U} \cdot \mathbf n = 0`, :math:`\mathbf B \cdot \mathbf n = 0`
        and reflecting or periodic boundaries of the energetic ions):

        Mass perturbation and divergence of the magnetic field:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \tilde \rho \, \textnormal d \mathbf x = 0\,, \qquad
            \frac{\partial}{\partial t} \nabla \cdot \tilde{\mathbf B} = 0

        Energy, for an equilibrium with :math:`\nabla q_0 = 0` and :math:`\nabla \times \mathbf B_0 = 0`:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \left[ \int \left( \frac{\rho_0 + \tilde \rho}{2} |\tilde{\mathbf U}|^2
            + \frac{2 q_0 \tilde q + \tilde q^2}{\gamma - 1} + \mathbf B_0 \cdot \tilde{\mathbf B} + \frac 12 |\tilde{\mathbf B}|^2
            \right) \textnormal d \mathbf x + \frac{A_\textnormal{h}}{A_\textnormal{b}} \int \int \frac 12 |\mathbf v|^2
            f_\textnormal{h} \, \textnormal d^3 \mathbf v \, \textnormal d \mathbf x \right] = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""Both fluid and particle velocities are normalized with the Alfvén speed,

        .. math::

            \hat U = \hat v = \hat v_A = \frac{\hat{B}_0}{\sqrt{\mu_0 \rho_0}}\,, \qquad
            \hat{q} = \sqrt{\hat p} = \frac{\hat B_0}{\sqrt{\mu_0}}\,, \qquad \hat f_h = \hat n / \hat v_A^3.

        The hot-species cyclotron parameter is :math:`\varepsilon = 1/(\hat\Omega_{c,\mathrm{hot}}\hat t)`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - The scalars of :class:`~struphy.models.PerturbativeMHDWithQ`: ``en_U`` (with the total density),
          ``en_thermal``, ``en_q2``, ``en_internal``, ``en_B``, ``en_B_eq``, ``en_B_tot``, ``en_p_eq``
        - Energetic-particle kinetic energy: :math:`\frac{A_\textnormal{h}}{A_\textnormal{b}} \int \int \frac 12
          |\mathbf v|^2 f_\textnormal{h}` (``en_f``)
        - Total energy: ``en_tot`` = energy of the perturbation of :class:`~struphy.models.PerturbativeMHDWithQ` +
          ``en_f``, conserved for :math:`\nabla q_0 = 0`, :math:`\nabla \times \mathbf B_0 = 0` up to the splitting
          error
        - Lost particles: ``n_lost_particles``"""

    @classmethod
    def doc_long_description(cls):
        r"""PerturbativeMHDWithQVlasovCC is the nonlinear counterpart of :class:`~struphy.models.LinearMHDVlasovCC`:
        energetic ions, described by a 6D Vlasov equation, are coupled through their density and current to the full
        (nonlinear) ideal MHD equations for the perturbations of a static equilibrium, written in the square root of
        the pressure, :math:`q = \sqrt p`.

        The time step reuses the steps of :class:`~struphy.models.PerturbativeMHDWithQ` and adds the current-coupling
        steps of :class:`~struphy.models.LinearMHDVlasovCC`, where the inertia of the bulk plasma is that of the total
        density :math:`\rho_0 + \tilde \rho^n` (``rho_field`` of the coupling propagators). Each coupling step
        conserves the sum of the kinetic energies of the bulk and of the energetic ions exactly. The magnetic field is
        in :math:`H(\textnormal{div})`, and no numerical dissipation is added, so the model is meant for smooth
        solutions."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the perturbative MHD-Vlasov current-coupling model:

        .. code-block:: python

            from struphy.models import PerturbativeMHDWithQVlasovCC

            model = PerturbativeMHDWithQVlasovCC()
            model.em_fields.b_field
            model.mhd.density
            model.mhd.velocity
            model.mhd.sqrt_p
            model.energetic_ions.var

            # Access tracked scalar quantities
            model.scalars["en_f"]
            model.scalars["en_tot"]
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - nonlinear saturation of energetic-particle driven MHD modes (e.g. Alfvén eigenmodes) including the nonlinear
          MHD terms
        - finite-amplitude Alfvén and magnetosonic waves interacting with energetic ions
        - comparison with :class:`~struphy.models.LinearMHDVlasovCC` in the small-amplitude limit"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - shocks and other discontinuities (no numerical dissipation)
        - equilibria with non-zero background flow
        - the magnetic field in :math:`H(\textnormal{curl})`
        - pressure-coupling closures (see :class:`~struphy.models.PerturbativeMHDWithQVlasovPC`)
        - collisional, resistive or viscous effects"""
