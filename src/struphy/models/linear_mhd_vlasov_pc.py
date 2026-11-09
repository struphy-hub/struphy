from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.linear_mhd import LinearMHD
from struphy.models.species import ParticleSpecies
from struphy.models.utils import add_energetic_ion_scalars, check_pressure_coupling_perp
from struphy.models.variables import FEECVariable, PICVariable
from struphy.propagators.pressure_coupling_6d import PressureCoupling6D
from struphy.propagators.push_eta_pc import PushEtaPC
from struphy.propagators.push_vxb import PushVxB
from struphy.propagators.push_vxb_1form import PushVxB1form


class LinearMHDVlasovPC(LinearMHD):
    """Hybrid linear MHD coupled with energetic ions (6D Vlasov) via the pressure-coupling scheme.

    Extends :class:`~struphy.models.LinearMHD` by energetic ions, whose pressure force perpendicular to the equilibrium
    magnetic field enters the MHD momentum equation.

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
    b_space: {"Hdiv", "Hcurl"}
        Space of the magnetic field, as in :class:`~struphy.models.LinearMHD` (default: "Hdiv"). With ``"Hcurl"``, the
        energetic ions are pushed with the 1-form field
        (:class:`~struphy.propagators.push_vxb_1form.PushVxB1form` instead of
        :class:`~struphy.propagators.push_vxb.PushVxB`).
    projected: bool
        Projected curl-curl shear-Alfvén step, only with ``b_space="Hcurl"``, as in :class:`~struphy.models.LinearMHD`
        (default: False).
    """

    # propagators renamed when the model joined the LinearEuler family (options of old run metadata)
    _renamed_propagators = {"shearalfven": "shear_alf"}

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
        """Propagators of :class:`~struphy.models.LinearMHD` plus the energetic-ion pushers (``push_eta_pc``,
        ``push_vxb``) and the pressure coupling (``pc6d``).

        Parameters
        ----------
        b_field : FEECVariable, optional
            Magnetic field perturbation, added to the equilibrium field in ``push_vxb`` (default: None).
        u_field : FEECVariable, optional
            Velocity perturbation, which ``push_eta_pc`` adds to the marker velocities (default: None).
        b_space : str
            Space of the magnetic field, ``"Hdiv"`` (:class:`~struphy.propagators.push_vxb.PushVxB`) or ``"Hcurl"``
            (:class:`~struphy.propagators.push_vxb_1form.PushVxB1form`) (default: ``"Hdiv"``).
        **kwargs
            Passed to :class:`LinearMHD.Propagators <struphy.models.LinearMHD.Propagators>`.
        """

        splitting_order = ("push_eta_pc", "push_vxb", "pc6d", "shear_alf", "bg_current", "bg_pressure", "sound")

        def __init__(
            self,
            b_field: FEECVariable = None,
            u_field: FEECVariable = None,
            b_space: str = "Hdiv",
            **kwargs,
        ):
            super().__init__(b_field=b_field, u_field=u_field, b_space=b_space, **kwargs)
            self.push_eta_pc = PushEtaPC(u_tilde=u_field)
            if b_space == "Hdiv":
                self.push_vxb = PushVxB(b2_var=b_field)
            else:
                self.push_vxb = PushVxB1form(b1_var=b_field)
            self.pc6d = PressureCoupling6D()

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
        b_space: LinearMHD.OptsBSpace = "Hdiv",
        projected: bool = False,
    ):
        # 0. store input parameters
        self._store_params(locals())

        # 1.-5. see LinearMHD
        super().__init__(base_units, mass_number=mhd_mass_number, b_space=b_space, projected=projected)

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

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        MHD continuity:

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot (\rho_0 \tilde{\mathbf{U}}) = 0

        MHD momentum:

        .. math::

            \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde{p} + \frac{A_\textnormal{h}}{A_\textnormal{b}} \left( \nabla \cdot \mathbb{P}_\textnormal{h} \right)_\perp = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B}_0 + \mathbf{J}_0 \times \tilde{\mathbf{B}}

        .. math::

            \mathbf{J}_0 = \nabla \times \mathbf{B}_0

        MHD pressure:

        .. math::

            \frac{\partial \tilde{p}}{\partial t} + \nabla \cdot (p_0 \tilde{\mathbf{U}}) + \frac{2}{3} p_0 \nabla \cdot \tilde{\mathbf{U}} = 0

        MHD induction:

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times (\tilde{\mathbf{U}} \times \mathbf{B}_0) = 0

        Energetic-particle Vlasov equation:

        .. math::

            \frac{\partial f_\textnormal{h}}{\partial t} + (\mathbf{v} + \tilde{\mathbf{U}}_\perp) \cdot \nabla f_\textnormal{h} + \left[ \frac{1}{\epsilon} \mathbf{v} \times (\mathbf{B}_0 + \tilde{\mathbf{B}}) - \nabla \tilde{\mathbf{U}}_\perp \cdot \mathbf{v} \right] \cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} = 0

        Energetic-particle pressure tensor:

        .. math::

            \mathbb{P}_\textnormal{h} = \int_{\mathbb{R}^3} \mathbf{v} \mathbf{v}^\top f_\textnormal{h} \, \textnormal{d}^3 \mathbf{v}

        with :math:`(\nabla \cdot \mathbb{P}_\textnormal{h})_i = \sum_k \partial_k (\mathbb{P}_\textnormal{h})_{ki}` and
        :math:`(\nabla \tilde{\mathbf U}_\perp \cdot \mathbf v)_k = \sum_i \partial_k \tilde U_{\perp,i} \, v_i`.
        The index :math:`\perp` denotes the projection perpendicular to the equilibrium magnetic field,
        :math:`\mathbf a_\perp = \mathbf a - (\mathbf b_0 \cdot \mathbf a) \mathbf b_0` with
        :math:`\mathbf b_0 = \mathbf B_0 / |\mathbf B_0|` (option ``use_perp_model=True`` of ``push_eta_pc`` and
        ``pc6d``, the default; with ``use_perp_model=False`` the projection is dropped,
        :math:`\mathbf a_\perp = \mathbf a`).

        **Conserved quantities** (for :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` and :math:`\mathbf B_0 \cdot \mathbf n = 0`
        on the boundary, or periodic boundaries, and periodic or reflecting boundaries for the energetic ions;
        otherwise, lost markers carry energy out of the domain, see ``n_lost_particles``):

        Mass perturbation, number of energetic ions and divergence of the magnetic field:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \tilde \rho \, \textnormal d \mathbf x = 0\,, \qquad
            \frac{\textnormal d}{\textnormal d t} \int \int_{\mathbb{R}^3} f_\textnormal{h} \, \textnormal{d}^3 \mathbf{v}
            \, \textnormal d \mathbf x = 0\,, \qquad
            \frac{\partial}{\partial t} \nabla \cdot \tilde{\mathbf B} = 0

        Total perturbed energy, for an equilibrium with :math:`\nabla \times \mathbf B_0 = 0` and
        :math:`\nabla p_0 = 0`, with :math:`\gamma = 5/3`. The pressure coupling exchanges energy between the MHD flow
        and the energetic ions: the work of :math:`(\nabla \cdot \mathbb{P}_\textnormal{h})_\perp` on the flow and of
        :math:`-\nabla \tilde{\mathbf U}_\perp \cdot \mathbf v` on the particles are
        :math:`\pm \int \mathbb{P}_\textnormal{h} : \nabla \tilde{\mathbf U}_\perp \, \textnormal d \mathbf x` (adjoint
        pair, also in :class:`~struphy.propagators.pressure_coupling_6d.PressureCoupling6D`). The transport with
        :math:`\tilde{\mathbf U}_\perp` does no work, because the phase-space flow
        :math:`(\mathbf v + \tilde{\mathbf U}_\perp, \epsilon^{-1} \mathbf v \times (\mathbf B_0 + \tilde{\mathbf B})
        - \nabla \tilde{\mathbf U}_\perp \cdot \mathbf v)` is divergence free:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \left[ \int \left( \frac{\rho_0}{2} |\tilde{\mathbf U}|^2
            + \frac 12 |\tilde{\mathbf B}|^2 + \frac{\tilde p^2}{2 \gamma p_0} \right) \textnormal d \mathbf x
            + \frac{A_\textnormal{h}}{2 A_\textnormal{b}} \int \int_{\mathbb{R}^3} |\mathbf v|^2 f_\textnormal{h} \,
            \textnormal{d}^3 \mathbf{v} \, \textnormal d \mathbf x \right] = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""Fluid and hot-particle velocities are normalized with the bulk Alfvén
        speed. The kinetic pressure tensor is scaled consistently with
        :math:`A_h m_H \hat n \hat v_A^2`, and the hot cyclotron parameter is
        :math:`\varepsilon`."""

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
        r"""LinearMHDVlasovPC is the pressure-coupling counterpart to the
        current-coupling hybrid model. It is targeted at linear problems where
        energetic-particle pressure anisotropy is the relevant feedback channel
        on the bulk MHD dynamics."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the linear MHD-Vlasov pressure-coupling model:

        .. code-block:: python

            from struphy.models import LinearMHDVlasovPC

            model = LinearMHDVlasovPC()
            model.em_fields.b_field
            model.mhd.velocity
            model.energetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - linear pressure-coupling hybrid studies
        - energetic-particle pressure feedback on MHD modes
        - verification of PushEtaPC and pressure-coupling operators"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - current-coupling closure studies
        - nonlinear hybrid turbulence
        - dissipative/resistive MHD
        - fully kinetic treatment of the bulk plasma"""
