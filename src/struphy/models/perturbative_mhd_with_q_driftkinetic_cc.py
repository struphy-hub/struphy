import logging

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.perturbative_mhd_with_q import PerturbativeMHDWithQ
from struphy.models.scalars import FunctionScalarPIC
from struphy.models.species import ParticleSpecies
from struphy.models.utils import (
    add_energetic_ion_scalars,
    energetic_ion_mu_energy,
    set_guiding_center_push_algos,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.propagators.base import Propagator
from struphy.propagators.current_coupling_5d_curlb import CurrentCoupling5DCurlb
from struphy.propagators.current_coupling_5d_density import CurrentCoupling5DDensity
from struphy.propagators.current_coupling_5d_gradb import CurrentCoupling5DGradB
from struphy.propagators.push_guiding_center_bx_estar import PushGuidingCenterBxEstar
from struphy.propagators.push_guiding_center_parallel import PushGuidingCenterParallel
from struphy.propagators.shear_alfven_current_coupling_5d import ShearAlfvenCurrentCoupling5D

logger = logging.getLogger("struphy")


class PerturbativeMHDWithQDriftkineticCC(PerturbativeMHDWithQ):
    """Hybrid nonlinear MHD (perturbations of a static equilibrium, :math:`q = \\sqrt p`) coupled with energetic ions
    (5D drift-kinetic) via the current-coupling scheme.

    Extends :class:`~struphy.models.PerturbativeMHDWithQ` by energetic ions in the guiding-center approximation, whose
    current and magnetization enter the MHD momentum equation, as in :class:`~struphy.models.LinearMHDDriftkineticCC`,
    with the inertia of the total density in the coupling terms. The shear-Alfvén steps ``shear_alf`` and
    ``nonlin_alf`` of PerturbativeMHDWithQ are replaced by one step ``shear_alf``
    (:class:`~struphy.propagators.shear_alfven_current_coupling_5d.ShearAlfvenCurrentCoupling5D` with the total
    magnetic field, ``nonlinear=True``), which also contains the coupling to the magnetization of the guiding centers.
    The magnetic field is in :math:`H(\\textnormal{div})`.

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
            self.var = PICVariable(space="Particles5D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                epsilon=epsilon,
            )

    ## additional propagators

    class Propagators(PerturbativeMHDWithQ.Propagators):
        """Propagators of :class:`~struphy.models.PerturbativeMHDWithQ`, with ``shear_alf`` and ``nonlin_alf`` replaced
        by the shear-Alfvén step with the total magnetic field coupled to the guiding-center magnetization
        (``shear_alf``), plus the guiding-center pushers (``push_bxe``, ``push_parallel``) and the current coupling
        (``cc5d_density``, ``cc5d_gradb``, ``cc5d_curlb``), all with the inertia of the total density.

        Parameters
        ----------
        rho_field : FEECVariable, optional
            Density perturbation, for the inertia of the total density (default: None).
        b_field : FEECVariable, optional
            Magnetic field perturbation, added to the equilibrium field in the pushers and couplings (default: None).
        energetic_ions_var : PICVariable, optional
            Markers of the energetic ions, deposited by ``shear_alf`` and ``cc5d_density`` (default: None).
        **kwargs
            Passed to :class:`PerturbativeMHDWithQ.Propagators <struphy.models.PerturbativeMHDWithQ.Propagators>`.
        """

        splitting_order = (
            "push_bxe",
            "push_parallel",
            "shear_alf",
            "bg_current",
            "bg_q",
            "sound",
            "nonlin_sound",
            "mom_adv",
            "cc5d_density",
            "cc5d_gradb",
            "cc5d_curlb",
        )
        # the guiding-center pushers as one symmetric (second-order) block, as in LinearMHDDriftkineticCC (issue #882)
        strang_groups = (("push_bxe", "push_parallel"),) + PerturbativeMHDWithQ.Propagators.strang_groups

        def __init__(
            self,
            rho_field: FEECVariable = None,
            b_field: FEECVariable = None,
            energetic_ions_var: PICVariable = None,
            **kwargs,
        ):
            super().__init__(rho_field=rho_field, b_field=b_field, **kwargs)
            self.push_bxe = PushGuidingCenterBxEstar(b_tilde=b_field)
            self.push_parallel = PushGuidingCenterParallel(b_tilde=b_field)
            # replaces shear_alf and nonlin_alf of PerturbativeMHDWithQ: with the total field B0 + b_tilde
            # (nonlinear=True, fixed), the step contains the nonlinear shear-Alfven terms of nonlin_alf
            del self.nonlin_alf
            self.shear_alf = ShearAlfvenCurrentCoupling5D(
                energetic_ions=energetic_ions_var, rho_field=rho_field, nonlinear=True
            )
            self.cc5d_density = CurrentCoupling5DDensity(
                energetic_ions=energetic_ions_var, b_tilde=b_field, rho_field=rho_field
            )
            self.cc5d_gradb = CurrentCoupling5DGradB(b_tilde=b_field, rho_field=rho_field)
            self.cc5d_curlb = CurrentCoupling5DCurlb(b_tilde=b_field, rho_field=rho_field)

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
        # shear_alf (u, b) is linked by LinearMHDWithQ
        super()._link_propagator_variables()
        self.propagators.cc5d_density.variables.u = self.mhd.velocity
        self.propagators.cc5d_gradb.variables.u = self.mhd.velocity
        self.propagators.cc5d_gradb.variables.energetic_ions = self.energetic_ions.var
        self.propagators.cc5d_curlb.variables.u = self.mhd.velocity
        self.propagators.cc5d_curlb.variables.energetic_ions = self.energetic_ions.var
        self.propagators.push_bxe.variables.ions = self.energetic_ions.var
        self.propagators.push_parallel.variables.ions = self.energetic_ions.var

    def _create_scalars(self):
        """:meta private:"""
        scalars = super()._create_scalars()
        en_fB = FunctionScalarPIC(self._compute_en_fB, self.energetic_ions.var)
        add_energetic_ion_scalars(scalars, self.mhd, self.energetic_ions, extra={"en_fB": en_fB}, kinetic_name="en_fv")
        return scalars

    def generate_default_parameter_file(self, path=None, prompt=True):
        """The default parameter file, with the algorithms of the reference discretization for the guiding-center
        pushers (:data:`~struphy.models.utils.GC_PUSH_ALGOS`)."""
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        set_guiding_center_push_algos(params_path)
        return params_path

    def post_allocate(self):
        """:meta private:"""
        super().post_allocate()

        # parallel component of the magnetic field perturbation (2-form), a 0-form
        self._PB = Propagator.basis_ops.PB

    def _compute_en_fB(self):
        return energetic_ion_mu_energy(self.mhd, self.energetic_ions, self._PB, self.em_fields.b_field)

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
            = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B} + (\nabla \times \mathbf{B}_0) \times \tilde{\mathbf{B}}
            \color{blue} + \frac{A_\textnormal{h}}{A_\textnormal{b}} \left[ \frac{1}{\epsilon} n_\textnormal{gc} \tilde{\mathbf{U}} - \frac{1}{\epsilon} \mathbf{J}_\textnormal{gc} - \nabla \times \mathbf{M}_\textnormal{gc} \right] \times \mathbf{B} \color{black}

        Square root of the pressure (adiabatic process):

        .. math::

            \frac{\partial \tilde{q}}{\partial t} + \tilde{\mathbf{U}} \cdot \nabla (q_0 + \tilde q)
            + \frac{\gamma}{2} (q_0 + \tilde q) \nabla \cdot \tilde{\mathbf{U}} = 0

        MHD induction:

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times (\tilde{\mathbf{U}} \times \mathbf{B}) = 0

        Energetic-particle drift-kinetic equation:

        .. math::

            \frac{\partial f_\textnormal{h}}{\partial t} + \frac{1}{B_\parallel^*} \left( v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^* \right) \cdot \nabla f_\textnormal{h} + \frac{1}{\epsilon} \frac{1}{B_\parallel^*} (\mathbf{B}^* \cdot \mathbf{E}^*) \frac{\partial f_\textnormal{h}}{\partial v_\parallel} = 0

        Energetic-particle moments:

        .. math::

            n_\textnormal{gc} = \int f_\textnormal{h} B_\parallel^* \, \textnormal{d} v_\parallel \textnormal{d} \mu\,, \qquad
            \mathbf{J}_\textnormal{gc} = \int f_\textnormal{h} \left( v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^* \right) \, \textnormal{d} v_\parallel \textnormal{d} \mu\,, \qquad
            \mathbf{M}_\textnormal{gc} = -\int f_\textnormal{h} B_\parallel^* \mu \mathbf{b}_0 \, \textnormal{d} v_\parallel \textnormal{d} \mu

        where :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}` is the total magnetic field and

        .. math::

            B^*_\parallel = \mathbf{b}_0 \cdot \mathbf{B}^*\,, \qquad
            \mathbf{B}^* = \mathbf{B} + \epsilon v_\parallel \nabla \times \mathbf{b}_0\,, \qquad
            \mathbf{E}^* = -\tilde{\mathbf{U}} \times \mathbf{B} - \epsilon \mu \nabla (\mathbf{b}_0 \cdot \mathbf{B})\,.

        Here, :math:`\tilde \rho`, :math:`\tilde{\mathbf U}`, :math:`\tilde q`, :math:`\tilde{\mathbf B}` are the
        perturbed density, velocity, square root of the pressure and magnetic field (not necessarily small), and
        :math:`\rho_0`, :math:`q_0 = \sqrt{p_0}`, :math:`\mathbf B_0` (unit vector :math:`\mathbf b_0`) those of a
        static MHD equilibrium; :math:`\gamma = 5/3`. The equations of :class:`~struphy.models.PerturbativeMHDWithQ`
        are extended by the energetic ions (guiding-center distribution function :math:`f_\textnormal{h}(\mathbf x,
        v_\parallel, \mu)`, mass number :math:`A_\textnormal{h}`); the terms in blue couple them to the MHD equations.
        Dropping the terms quadratic in the MHD perturbations in the fluid equations
        (:math:`\rho_0 + \tilde \rho \to \rho_0`, :math:`\tilde{\mathbf U} \cdot \nabla \tilde{\mathbf U} \to 0`, the
        terms quadratic in :math:`\tilde q`) gives the equations of :class:`~struphy.models.LinearMHDDriftkineticCC`
        (with :math:`\tilde p = 2 q_0 \tilde q`), which already has the total field :math:`\mathbf B` in the
        Lorentz force and the induction equation.

        **Conserved quantities** (for :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` and
        :math:`\mathbf B_0 \cdot \mathbf n = \tilde{\mathbf B} \cdot \mathbf n = 0` on the boundary, or periodic
        boundaries, and periodic or reflecting boundaries for the energetic ions; otherwise, lost markers carry energy
        out of the domain, see ``n_lost_particles``):

        Mass perturbation, number of energetic ions (the guiding-center phase-space volume
        :math:`B_\parallel^* \, \textnormal{d} \mathbf x \, \textnormal{d} v_\parallel \textnormal{d} \mu` is preserved by
        the flow) and divergence of the magnetic field:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \tilde \rho \, \textnormal d \mathbf x = 0\,, \qquad
            \frac{\textnormal d}{\textnormal d t} \int \int f_\textnormal{h} B_\parallel^* \, \textnormal{d} v_\parallel
            \textnormal{d} \mu \, \textnormal d \mathbf x = 0\,, \qquad
            \frac{\partial}{\partial t} \nabla \cdot \tilde{\mathbf B} = 0

        Energy, for an equilibrium with :math:`\nabla q_0 = 0` and :math:`\nabla \times \mathbf B_0 = 0`. The
        guiding-center current :math:`\mathbf{J}_\textnormal{gc}` and magnetization :math:`\mathbf{M}_\textnormal{gc}`
        exchange energy between the MHD flow, the magnetic field and the energetic ions:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \left[ \int \left( \frac{\rho_0 + \tilde \rho}{2} |\tilde{\mathbf U}|^2
            + \frac{2 q_0 \tilde q + \tilde q^2}{\gamma - 1} + \mathbf B_0 \cdot \tilde{\mathbf B}
            + \frac 12 |\tilde{\mathbf B}|^2 \right) \textnormal d \mathbf x
            + \frac{A_\textnormal{h}}{A_\textnormal{b}} \int \int f_\textnormal{h} B_\parallel^* \left( \frac 12
            v_\parallel^2 + \mu \, \mathbf b_0 \cdot \mathbf B \right) \textnormal{d} v_\parallel \textnormal{d} \mu
            \, \textnormal d \mathbf x \right] = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""The bulk and energetic-particle flow scales are normalized with the bulk Alfvén speed, while the magnetic
        moment carries its own unit:

        .. math::

            \hat U = \hat v = \hat v_{A,\mathrm{bulk}} = \frac{\hat{B}_0}{\sqrt{\mu_0 \rho_0}}\,,\qquad
            \hat{q} = \sqrt{\hat p} = \frac{\hat B_0}{\sqrt{\mu_0}}\,,\qquad
            \hat\mu = A_h m_H \hat v_h^2 / \hat B.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - The scalars of :class:`~struphy.models.PerturbativeMHDWithQ`: ``en_U`` (with the total density),
          ``en_thermal``, ``en_q2``, ``en_internal``, ``en_B``, ``en_B_eq``, ``en_B_tot``, ``en_p_eq``
        - Parallel energetic-particle energy: ``en_fv``
        - Magnetic-moment energetic-particle energy: ``en_fB``
        - Total energy: ``en_tot`` = energy of the perturbation of :class:`~struphy.models.PerturbativeMHDWithQ` +
          ``en_fv`` + ``en_fB``
        - Lost particles: ``n_lost_particles``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence; the magnetic field is in
        :math:`H(\\textnormal{div})`). The default parameter file of this model
        (``struphy params PerturbativeMHDWithQDriftkineticCC``) sets the energy-conserving second-order discrete
        gradient method ``algo="discrete_gradient_2nd_order"`` (implicit mid-point) for both guiding-center pushers,
        ``push_bxe`` and ``push_parallel``, which form one symmetric (second-order) block of the splitting
        (``strang_groups``), as in :class:`~struphy.models.LinearMHDDriftkineticCC`. The shear-Alfvén step
        ``shear_alf`` has the total magnetic field in its basis projection operator (``nonlinear=True``, fixed by the
        model), with the weight at :math:`t^n`:"""
        return cls.propagators_doc()

    @classmethod
    def doc_long_description(cls):
        r"""PerturbativeMHDWithQDriftkineticCC is the nonlinear counterpart of
        :class:`~struphy.models.LinearMHDDriftkineticCC`: energetic ions in the guiding-center (drift-kinetic)
        approximation are coupled through their current and magnetization to the full (nonlinear) ideal MHD equations
        for the perturbations of a static equilibrium, written in the square root of the pressure, :math:`q = \sqrt p`.

        The time step reuses the steps of :class:`~struphy.models.PerturbativeMHDWithQ` and adds the guiding-center
        pushers and the current-coupling steps of :class:`~struphy.models.LinearMHDDriftkineticCC`, where the inertia
        of the bulk plasma is that of the total density :math:`\rho_0 + \tilde \rho^n` (``rho_field`` of the coupling
        propagators). The two shear-Alfvén steps of PerturbativeMHDWithQ (``shear_alf`` with :math:`\mathbf B_0`,
        ``nonlin_alf`` with :math:`\tilde{\mathbf B}`) are one step here: the coupled step of LinearMHDDriftkineticCC
        with the total field :math:`\mathbf B_0 + \tilde{\mathbf B}` in its basis projection operator, which also
        contains the coupling to the magnetization of the guiding centers (the weight is at :math:`t^n`, so these
        nonlinear terms are first order in time). The magnetic field is in :math:`H(\textnormal{div})`, and no
        numerical dissipation is added, so the model is meant for smooth solutions."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the perturbative MHD plus drift-kinetic current-coupling model:

        .. code-block:: python

            from struphy.models import PerturbativeMHDWithQDriftkineticCC

            model = PerturbativeMHDWithQDriftkineticCC()
            model.em_fields.b_field
            model.mhd.density
            model.mhd.velocity
            model.mhd.sqrt_p
            model.energetic_ions.var

            # Access tracked scalar quantities
            model.scalars["en_fv"]
            model.scalars["en_fB"]
            model.scalars["en_tot"]
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - nonlinear saturation of energetic-particle driven Alfvén eigenmodes including the nonlinear MHD terms, with
          guiding-center energetic ions
        - comparison with :class:`~struphy.models.LinearMHDDriftkineticCC` in the small-amplitude limit"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - shocks and other discontinuities (no numerical dissipation)
        - equilibria with non-zero background flow
        - the magnetic field in :math:`H(\textnormal{curl})`
        - regimes where the gyromotion of the energetic ions must be resolved (see
          :class:`~struphy.models.PerturbativeMHDWithQVlasovCC`)
        - collisional, resistive or viscous effects"""
