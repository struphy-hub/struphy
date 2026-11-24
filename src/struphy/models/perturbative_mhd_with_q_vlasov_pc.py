import logging

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.perturbative_mhd_with_q import PerturbativeMHDWithQ
from struphy.models.species import ParticleSpecies
from struphy.models.utils import add_energetic_ion_scalars, check_pressure_coupling_perp
from struphy.models.variables import FEECVariable, PICVariable
from struphy.propagators.pressure_coupling_6d import PressureCoupling6D
from struphy.propagators.push_eta_pc import PushEtaPC
from struphy.propagators.push_vxb import PushVxB

logger = logging.getLogger("struphy")


class PerturbativeMHDWithQVlasovPC(PerturbativeMHDWithQ):
    """Hybrid nonlinear MHD (perturbations of a static equilibrium, :math:`q = \\sqrt p`) coupled with energetic ions
    (6D Vlasov) via the pressure-coupling scheme.

    Extends :class:`~struphy.models.PerturbativeMHDWithQ` by energetic ions, whose pressure force perpendicular to the
    equilibrium magnetic field enters the MHD momentum equation, as in :class:`~struphy.models.LinearMHDVlasovPC`,
    with the inertia of the total density in the coupling term. The magnetic field is in :math:`H(\\textnormal{div})`.

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
        """Propagators of :class:`~struphy.models.PerturbativeMHDWithQ` plus the energetic-ion pushers
        (``push_eta_pc``, ``push_vxb``) and the pressure coupling (``pc6d``, with the inertia of the total density).

        Parameters
        ----------
        rho_field : FEECVariable, optional
            Density perturbation, for the inertia of the total density (default: None).
        b_field : FEECVariable, optional
            Magnetic field perturbation, added to the equilibrium field in ``push_vxb`` (default: None).
        u_field : FEECVariable, optional
            Velocity perturbation, which ``push_eta_pc`` adds to the marker velocities (default: None).
        **kwargs
            Passed to :class:`PerturbativeMHDWithQ.Propagators <struphy.models.PerturbativeMHDWithQ.Propagators>`.
        """

        splitting_order = (
            "push_eta_pc",
            "push_vxb",
            "pc6d",
            "shear_alf",
            "nonlin_alf",
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
            u_field: FEECVariable = None,
            **kwargs,
        ):
            super().__init__(rho_field=rho_field, b_field=b_field, u_field=u_field, **kwargs)
            self.push_eta_pc = PushEtaPC(u_tilde=u_field)
            self.push_vxb = PushVxB(b2_var=b_field)
            self.pc6d = PressureCoupling6D(rho_field=rho_field)

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

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.pc6d.variables.u = self.mhd.velocity
        self.propagators.pc6d.variables.energetic_ions = self.energetic_ions.var
        self.propagators.push_eta_pc.variables.var = self.energetic_ions.var
        self.propagators.push_vxb.variables.ions = self.energetic_ions.var

    def post_allocate(self):
        """:meta private:"""
        super().post_allocate()
        check_pressure_coupling_perp(self)

    def _create_scalars(self):
        """:meta private:"""
        scalars = super()._create_scalars()
        add_energetic_ion_scalars(scalars, self.mhd, self.energetic_ions)
        return scalars

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
            \color{blue} + \frac{A_\textnormal{h}}{A_\textnormal{b}} \left( \nabla \cdot \mathbb{P}_\textnormal{h} \right)_\perp \color{black}
            = (\nabla \times \tilde{\mathbf{B}}) \times (\mathbf{B}_0 + \tilde{\mathbf B}) + (\nabla \times \mathbf{B}_0) \times \tilde{\mathbf{B}}

        Square root of the pressure (adiabatic process):

        .. math::

            \frac{\partial \tilde{q}}{\partial t} + \tilde{\mathbf{U}} \cdot \nabla (q_0 + \tilde q)
            + \frac{\gamma}{2} (q_0 + \tilde q) \nabla \cdot \tilde{\mathbf{U}} = 0

        MHD induction:

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times \left(\tilde{\mathbf{U}} \times (\mathbf{B}_0 + \tilde{\mathbf B})\right) = 0

        Energetic-particle Vlasov equation:

        .. math::

            \frac{\partial f_\textnormal{h}}{\partial t} + (\mathbf{v} \color{blue} + \tilde{\mathbf{U}}_\perp \color{black}) \cdot \nabla f_\textnormal{h} + \left[ \frac{1}{\epsilon} \mathbf{v} \times (\mathbf{B}_0 + \tilde{\mathbf{B}}) \color{blue} - \nabla \tilde{\mathbf{U}}_\perp \cdot \mathbf{v} \color{black} \right] \cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} = 0

        Energetic-particle pressure tensor:

        .. math::

            \mathbb{P}_\textnormal{h} = \int_{\mathbb{R}^3} \mathbf{v} \mathbf{v}^\top f_\textnormal{h} \, \textnormal{d}^3 \mathbf{v}

        with :math:`(\nabla \cdot \mathbb{P}_\textnormal{h})_i = \sum_k \partial_k (\mathbb{P}_\textnormal{h})_{ki}` and
        :math:`(\nabla \tilde{\mathbf U}_\perp \cdot \mathbf v)_k = \sum_i \partial_k \tilde U_{\perp,i} \, v_i`.

        Here, :math:`\tilde \rho`, :math:`\tilde{\mathbf U}`, :math:`\tilde q`, :math:`\tilde{\mathbf B}` are the
        perturbed density, velocity, square root of the pressure and magnetic field (not necessarily small), and
        :math:`\rho_0`, :math:`q_0 = \sqrt{p_0}`, :math:`\mathbf B_0` those of a static MHD equilibrium;
        :math:`\gamma = 5/3`. The subscript :math:`\perp` denotes the projection perpendicular to the equilibrium
        field, :math:`\mathbf a_\perp = \mathbf a - (\mathbf b_0 \cdot \mathbf a) \mathbf b_0` with
        :math:`\mathbf b_0 = \mathbf B_0 / |\mathbf B_0|` (option ``use_perp_model=True`` of ``push_eta_pc`` and
        ``pc6d``, the default; with ``use_perp_model=False`` the projection is dropped,
        :math:`\mathbf a_\perp = \mathbf a`). The equations of :class:`~struphy.models.PerturbativeMHDWithQ` are extended by the
        energetic ions (distribution function :math:`f_\textnormal{h}` in the frame moving with
        :math:`\tilde{\mathbf U}_\perp`, mass number :math:`A_\textnormal{h}`); the terms in blue couple them to the
        MHD equations. Dropping the terms quadratic in the MHD perturbations gives
        :class:`~struphy.models.LinearMHDVlasovPC` (with :math:`\tilde p = 2 q_0 \tilde q`).

        **Conserved quantities** (for :math:`\tilde{\mathbf U} \cdot \mathbf n = 0`, :math:`\mathbf B \cdot \mathbf n = 0`
        and reflecting or periodic boundaries of the energetic ions):

        Mass perturbation, number of energetic ions and divergence of the magnetic field:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \tilde \rho \, \textnormal d \mathbf x = 0\,, \qquad
            \frac{\textnormal d}{\textnormal d t} \int \int_{\mathbb{R}^3} f_\textnormal{h} \, \textnormal{d}^3 \mathbf{v}
            \, \textnormal d \mathbf x = 0\,, \qquad
            \frac{\partial}{\partial t} \nabla \cdot \tilde{\mathbf B} = 0

        Energy, for an equilibrium with :math:`\nabla q_0 = 0` and :math:`\nabla \times \mathbf B_0 = 0`. The pressure
        coupling exchanges energy between the MHD flow and the energetic ions: the work of
        :math:`(\nabla \cdot \mathbb{P}_\textnormal{h})_\perp` on the flow and of
        :math:`-\nabla \tilde{\mathbf U}_\perp \cdot \mathbf v` on the particles are
        :math:`\pm \int \mathbb{P}_\textnormal{h} : \nabla \tilde{\mathbf U}_\perp \, \textnormal d \mathbf x` (adjoint
        pair, also in :class:`~struphy.propagators.pressure_coupling_6d.PressureCoupling6D`). The transport with
        :math:`\tilde{\mathbf U}_\perp` does no work, because the phase-space flow
        :math:`(\mathbf v + \tilde{\mathbf U}_\perp, \epsilon^{-1} \mathbf v \times (\mathbf B_0 + \tilde{\mathbf B})
        - \nabla \tilde{\mathbf U}_\perp \cdot \mathbf v)` is divergence free:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \left[ \int \left( \frac{\rho_0 + \tilde \rho}{2} |\tilde{\mathbf U}|^2
            + \frac{2 q_0 \tilde q + \tilde q^2}{\gamma - 1} + \mathbf B_0 \cdot \tilde{\mathbf B} + \frac 12 |\tilde{\mathbf B}|^2
            \right) \textnormal d \mathbf x + \frac{A_\textnormal{h}}{A_\textnormal{b}} \int \int \frac 12 |\mathbf v|^2
            f_\textnormal{h} \, \textnormal d^3 \mathbf v \, \textnormal d \mathbf x \right] = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""Fluid and hot-particle velocities are normalized with the bulk Alfvén speed,

        .. math::

            \hat U = \hat v = \hat v_A = \frac{\hat{B}_0}{\sqrt{\mu_0 \rho_0}}\,, \qquad
            \hat{q} = \sqrt{\hat p} = \frac{\hat B_0}{\sqrt{\mu_0}}\,.

        The kinetic pressure tensor is scaled consistently with :math:`A_h m_H \hat n \hat v_A^2`, and the hot
        cyclotron parameter is :math:`\varepsilon`."""

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
        r"""PerturbativeMHDWithQVlasovPC is the nonlinear counterpart of :class:`~struphy.models.LinearMHDVlasovPC`:
        the pressure force (perpendicular to the equilibrium field) of energetic ions, described by a 6D Vlasov
        equation, enters the full (nonlinear) ideal MHD equations for the perturbations of a static equilibrium,
        written in the square root of the pressure, :math:`q = \sqrt p`.

        The time step reuses the steps of :class:`~struphy.models.PerturbativeMHDWithQ` and adds the pressure-coupling
        steps of :class:`~struphy.models.LinearMHDVlasovPC`, unchanged except for the inertia of the bulk plasma in
        ``pc6d``, which is that of the total density :math:`\rho_0 + \tilde \rho^n` (``rho_field`` of
        :class:`~struphy.propagators.pressure_coupling_6d.PressureCoupling6D`). The magnetic field is in
        :math:`H(\textnormal{div})`, and no numerical dissipation is added, so the model is meant for smooth
        solutions."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the perturbative MHD-Vlasov pressure-coupling model:

        .. code-block:: python

            from struphy.models import PerturbativeMHDWithQVlasovPC

            model = PerturbativeMHDWithQVlasovPC()
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

        - nonlinear pressure-coupling hybrid studies (energetic-particle pressure feedback on finite-amplitude MHD
          modes)
        - comparison with :class:`~struphy.models.LinearMHDVlasovPC` in the small-amplitude limit"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - shocks and other discontinuities (no numerical dissipation)
        - equilibria with non-zero background flow
        - the magnetic field in :math:`H(\textnormal{curl})`
        - current-coupling closures (see :class:`~struphy.models.PerturbativeMHDWithQVlasovCC`)
        - collisional, resistive or viscous effects"""
