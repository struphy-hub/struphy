from struphy.io.options import BaseUnits
from struphy.models.linear_euler_mixins import PerturbativeQMixin
from struphy.models.linear_mhd_with_q import LinearMHDWithQ
from struphy.models.variables import FEECVariable
from struphy.propagators.momentum_advection import MomentumAdvection
from struphy.propagators.shear_alfven_nonlinear import ShearAlfvenNonlinear
from struphy.propagators.sound_wave_nonlinear_with_q import SoundWaveNonlinearWithQ


class PerturbativeMHDWithQ(PerturbativeQMixin, LinearMHDWithQ):
    """Ideal MHD equations for the perturbations of a static (zero-flow) equilibrium, with all nonlinear terms, and the
    square root of the pressure, :math:`q = \\sqrt p`, as thermodynamic variable.

    Extends :class:`~struphy.models.LinearMHDWithQ` by the terms that are nonlinear in the perturbations: the inertia of
    the density perturbation, the advection of mass and momentum, the nonlinear sound wave terms and the nonlinear
    terms of the Lorentz force and of the induction equation. The magnetic field is in :math:`H(\\textnormal{div})`.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mass_number: float
        Mass number (in units of Proton mass) of the ion species (default: 1.0)
    """

    ## additional propagators

    class Propagators(LinearMHDWithQ.Propagators):
        """Propagators of :class:`~struphy.models.LinearMHDWithQ` (with the inertia of the total density) plus
        ``nonlin_alf`` (nonlinear shear-Alfvén terms), ``nonlin_sound`` (nonlinear sound wave terms) and ``mom_adv``
        (advection of mass and momentum).

        Parameters
        ----------
        rho_field : FEECVariable, optional
            Density perturbation, for the inertia of the total density (default: None).
        **kwargs
            Passed to :class:`LinearMHDWithQ.Propagators <struphy.models.LinearMHDWithQ.Propagators>`.
        """

        splitting_order = ("shear_alf", "nonlin_alf", "bg_current", "bg_q", "sound", "nonlin_sound", "mom_adv")

        def __init__(self, rho_field: FEECVariable = None, **kwargs):
            super().__init__(rho_field=rho_field, **kwargs)
            self.nonlin_alf = ShearAlfvenNonlinear(rho_field=rho_field)
            self.nonlin_sound = SoundWaveNonlinearWithQ(rho_field=rho_field)
            self.mom_adv = MomentumAdvection()

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mass_number: float = 1.0,
    ):
        # 0. store input parameters
        self._store_params(locals())

        # 1.-5. see LinearMHDWithQ (magnetic field in H(div), as needed by the nonlinear steps)
        super().__init__(base_units, mass_number, b_space="Hdiv", projected=False)

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.nonlin_alf.variables.u = self.mhd.velocity
        self.propagators.nonlin_alf.variables.b = self.em_fields.b_field

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity (mass conservation):

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot \left( (\rho_0 + \tilde \rho) \tilde{\mathbf{U}} \right) = 0

        Momentum (Lorentz force):

        .. math::

            (\rho_0 + \tilde \rho) \left( \frac{\partial \tilde{\mathbf{U}}}{\partial t}
            + \tilde{\mathbf U} \cdot \nabla \tilde{\mathbf U} \right) + \nabla (2 q_0 \tilde q + \tilde q^2)
            = (\nabla \times \tilde{\mathbf{B}}) \times (\mathbf{B}_0 + \tilde{\mathbf B}) + (\nabla \times \mathbf{B}_0) \times \tilde{\mathbf{B}}

        Square root of the pressure (adiabatic process):

        .. math::

            \frac{\partial \tilde{q}}{\partial t} + \tilde{\mathbf{U}} \cdot \nabla (q_0 + \tilde q)
            + \frac{\gamma}{2} (q_0 + \tilde q) \nabla \cdot \tilde{\mathbf{U}} = 0

        Induction (Faraday's law):

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times \left(\tilde{\mathbf{U}} \times (\mathbf{B}_0 + \tilde{\mathbf B})\right) = 0.

        Here, :math:`\tilde \rho`, :math:`\tilde{\mathbf U}`, :math:`\tilde q`, :math:`\tilde{\mathbf B}` are the perturbed
        density, velocity, square root of the pressure and magnetic field (not necessarily small), and :math:`\rho_0`,
        :math:`q_0 = \sqrt{p_0}`, :math:`\mathbf B_0` those of a static MHD equilibrium,
        :math:`\nabla q_0^2 = (\nabla \times \mathbf B_0) \times \mathbf B_0`; :math:`\gamma = 5/3`. These are the ideal
        MHD equations for :math:`\rho = \rho_0 + \tilde \rho`, :math:`\mathbf U = \tilde{\mathbf U}`,
        :math:`q = q_0 + \tilde q`, :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}`, with the equilibrium force
        balance subtracted. Dropping the terms quadratic in the perturbations gives :class:`~struphy.models.LinearMHDWithQ`.

        **Conserved quantities** (for :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` and :math:`\mathbf B \cdot \mathbf n = 0`
        on the boundary, or periodic boundaries):

        Mass perturbation and divergence of the magnetic field:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \tilde \rho \, \textnormal d \mathbf x = 0\,, \qquad
            \frac{\partial}{\partial t} \nabla \cdot \tilde{\mathbf B} = 0

        Energy of the perturbation, for an equilibrium with :math:`\nabla q_0 = 0` and :math:`\nabla \times \mathbf B_0 = 0`:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \left( \frac{\rho_0 + \tilde \rho}{2} |\tilde{\mathbf U}|^2
            + \frac{2 q_0 \tilde q + \tilde q^2}{\gamma - 1} + \mathbf B_0 \cdot \tilde{\mathbf B} + \frac 12 |\tilde{\mathbf B}|^2
            \right) \textnormal d \mathbf x = 0
        """

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence; the magnetic field is in
        :math:`H(\textnormal{div})`):"""
        return cls.propagators_doc()

    @classmethod
    def doc_normalization(cls):
        r"""All velocities are normalized by the Alfvén velocity:

        .. math::

            \hat{U} = \hat{v}_\mathrm{A} = \frac{\hat{B}_0}{\sqrt{\mu_0 \rho_0}}\,, \qquad
            \hat{q} = \sqrt{\hat p} = \frac{\hat B_0}{\sqrt{\mu_0}}

        The model therefore uses the Alfvén velocity as its characteristic speed scale."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy: :math:`E_U = \frac{1}{2} \int (\rho_0 + \tilde \rho) |\tilde{\mathbf{U}}|^2 \, \mathrm{d}V`
          (``en_U``), with the total density (projected :math:`\rho_0`)
        - Quadratic thermal energy of the linear sound wave step: :math:`\frac{2}{\gamma}\int \tilde{q}^2 \, \mathrm{d}V`
          (``en_thermal``); of the nonlinear sound wave step: :math:`\frac{1}{\gamma - 1}\int \tilde{q}^2 \, \mathrm{d}V`
          (``en_q2``)
        - Internal energy of the perturbation: :math:`E_\mathrm{int} = \int \frac{2 q_0 \tilde q + \tilde q^2}{\gamma - 1}
          \, \mathrm{d}V` (``en_internal``)
        - Magnetic energy (perturbation): :math:`E_B = \frac{1}{2} \int |\tilde{\mathbf{B}}|^2 \, \mathrm{d}V` (``en_B``),
          equilibrium magnetic energy (``en_B_eq``), total magnetic energy
          :math:`\frac 12 \int |\mathbf B_0 + \tilde{\mathbf B}|^2 \, \mathrm{d}V` (``en_B_tot``)
        - Equilibrium internal energy: :math:`E_{p0} = \int \frac{p_0}{\gamma - 1} \, \mathrm{d}V` (``en_p_eq``)
        - Energy of the perturbation: :math:`E_{\mathrm{tot}} = E_U + E_\mathrm{int} + \frac 12 \int (|\mathbf B_0 +
          \tilde{\mathbf B}|^2 - |\mathbf B_0|^2) \, \mathrm{d}V` (``en_tot``), conserved by the equations for
          :math:`\nabla q_0 = 0`, :math:`\nabla \times \mathbf B_0 = 0`. Each step of the splitting conserves its own
          energy exactly, so :math:`E_{\mathrm{tot}}` changes by the splitting error only."""

    @classmethod
    def doc_long_description(cls):
        r"""This model simulates finite-amplitude perturbations in a magnetized plasma around a static MHD equilibrium.
        It solves the ideal MHD equations for the perturbations of the density, the velocity, the square root of the
        pressure, :math:`q = \sqrt p`, and the magnetic field, written as the linear system of
        :class:`~struphy.models.LinearMHDWithQ` plus the nonlinear terms.

        The time step reuses the steps of the linear model (with the inertia of the total density
        :math:`\rho_0 + \tilde \rho^n` and without the continuity equation) and adds the steps of the nonlinear terms,
        each energy conserving for its own energy:

        - ``nonlin_alf``: the shear-Alfvén step with the weight :math:`\mathbf B_0 \to \tilde{\mathbf B}`, i.e. the force
          :math:`(\nabla \times \tilde{\mathbf B}) \times \tilde{\mathbf B}` and the induction term
          :math:`\nabla \times (\tilde{\mathbf U} \times \tilde{\mathbf B})`;
        - ``nonlin_sound``: the terms of the sound waves quadratic in :math:`\tilde q`;
        - ``mom_adv``: the continuity equation and the advection of the momentum (discrete gradient, Picard iteration).

        The weights of the nonlinear terms are at :math:`t^n` by default (``Options(weights="frozen")``) or at
        :math:`t^{n+1/2}` (``weights="midpoint"``, Picard iteration, second order in time). The magnetic field is in
        :math:`H(\textnormal{div})` (the shear-Alfvén step of :class:`~struphy.models.LinearMHD` with
        ``b_space="Hdiv"``). No numerical dissipation is added, so the model is meant for smooth solutions."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a perturbative MHD model in :math:`q`:

        .. code-block:: python

            from struphy.models import PerturbativeMHDWithQ

            model = PerturbativeMHDWithQ()

            # Access fields
            model.em_fields.b_field
            model.mhd.density
            model.mhd.velocity
            model.mhd.sqrt_p

            # Access tracked scalar quantities
            model.scalars["en_U"]
            model.scalars["en_B"]
            model.scalars["en_tot"]
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for finite-amplitude perturbations of a static magnetized plasma.

        Typical use cases include:

        - nonlinear Alfvén and magnetosonic waves
        - nonlinear saturation of instabilities of a static equilibrium (in the absence of shocks)
        - comparison with :class:`~struphy.models.LinearMHDWithQ` in the small-amplitude limit"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - shocks and other discontinuities (no numerical dissipation)
        - equilibria with non-zero background flow
        - dissipative effects such as resistivity or viscosity (reconnection)
        - the magnetic field in :math:`H(\textnormal{curl})`
        - kinetic or particle-field effects beyond the fluid MHD description"""
