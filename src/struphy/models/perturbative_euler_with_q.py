from struphy.io.options import BaseUnits
from struphy.models.linear_euler_mixins import PerturbativeQMixin
from struphy.models.linear_euler_with_q import LinearEulerWithQ
from struphy.models.variables import FEECVariable
from struphy.propagators.momentum_advection import MomentumAdvection
from struphy.propagators.sound_wave_nonlinear_with_q import SoundWaveNonlinearWithQ


class PerturbativeEulerWithQ(PerturbativeQMixin, LinearEulerWithQ):
    """Compressible Euler equations for the perturbations of a static (zero-flow) equilibrium, with all nonlinear
    terms, and the square root of the pressure, :math:`q = \\sqrt p`, as thermodynamic variable.

    Extends :class:`~struphy.models.LinearEulerWithQ` by the terms that are nonlinear in the perturbations: the
    inertia of the density perturbation, the advection of mass and momentum and the nonlinear sound wave terms.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: ``BaseUnits(kBT=1.0)``).
    mass_number: float
        Mass number (in units of Proton mass) of the fluid species (default: 1.0)
    """

    ## additional propagators

    class Propagators(LinearEulerWithQ.Propagators):
        """Propagators of :class:`~struphy.models.LinearEulerWithQ` (with the inertia of the total density) plus
        ``nonlin_sound`` (nonlinear sound wave terms) and ``mom_adv`` (advection of mass and momentum).

        Parameters
        ----------
        rho_field : FEECVariable, optional
            Density perturbation, for the inertia of the total density (default: None).
        **kwargs
            Passed to :class:`LinearEulerWithQ.Propagators <struphy.models.LinearEulerWithQ.Propagators>`.
        """

        def __init__(self, rho_field: FEECVariable = None, **kwargs):
            super().__init__(rho_field=rho_field, **kwargs)
            self.nonlin_sound = SoundWaveNonlinearWithQ(rho_field=rho_field)
            self.mom_adv = MomentumAdvection()

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        mass_number: float = 1.0,
    ):
        # 0. store input parameters
        self._store_params(locals())

        # 1.-5. see LinearEulerWithQ
        super().__init__(base_units, mass_number)

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity (mass conservation):

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot \left( (\rho_0 + \tilde \rho) \tilde{\mathbf{U}} \right) = 0

        Momentum:

        .. math::

            (\rho_0 + \tilde \rho) \left( \frac{\partial \tilde{\mathbf{U}}}{\partial t}
            + \tilde{\mathbf U} \cdot \nabla \tilde{\mathbf U} \right) + \nabla (2 q_0 \tilde q + \tilde q^2) = 0

        Square root of the pressure (adiabatic process):

        .. math::

            \frac{\partial \tilde{q}}{\partial t} + \tilde{\mathbf{U}} \cdot \nabla (q_0 + \tilde q)
            + \frac{\gamma}{2} (q_0 + \tilde q) \nabla \cdot \tilde{\mathbf{U}} = 0\,.

        Here, :math:`\tilde \rho`, :math:`\tilde{\mathbf U}`, :math:`\tilde q` are the perturbed density, velocity and
        square root of the pressure (not necessarily small), :math:`\rho_0`, :math:`q_0 = \sqrt{p_0}` those of a static
        equilibrium (zero flow) with :math:`\nabla p_0 = 0`, and :math:`\gamma = 5/3`. These are the compressible Euler
        equations :math:`\partial_t \rho + \nabla \cdot (\rho \mathbf U) = 0`,
        :math:`\rho (\partial_t \mathbf U + \mathbf U \cdot \nabla \mathbf U) + \nabla q^2 = 0`,
        :math:`\partial_t q + \mathbf U \cdot \nabla q + \frac{\gamma}{2} q \nabla \cdot \mathbf U = 0` for
        :math:`\rho = \rho_0 + \tilde \rho`, :math:`\mathbf U = \tilde{\mathbf U}`, :math:`q = q_0 + \tilde q`, with the
        equilibrium subtracted. Dropping the terms quadratic in the perturbations gives
        :class:`~struphy.models.LinearEulerWithQ`.

        **Conserved quantities** (for :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` on the boundary, or periodic
        boundaries):

        Mass perturbation:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \tilde \rho \, \textnormal d \mathbf x = 0

        Energy of the perturbation (kinetic plus internal energy, the internal energy of the equilibrium subtracted),
        for an equilibrium with :math:`\nabla q_0 = 0`:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \left( \frac{\rho_0 + \tilde \rho}{2} |\tilde{\mathbf U}|^2
            + \frac{2 q_0 \tilde q + \tilde q^2}{\gamma - 1} \right) \textnormal d \mathbf x = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""All velocities are normalized by the thermal velocity of the bulk species, defined by the unit of the
        temperature :math:`\hat{k_B T}` (``BaseUnits.kBT``):

        .. math::

            \hat{U} = \hat{v}_\mathrm{th} = \sqrt{\frac{\hat{k_B T}}{A m_\mathrm{H}}}\,, \qquad
            \hat{\rho} = A m_\mathrm{H} \hat{n}\,, \qquad \hat{q} = \sqrt{\hat{p}} = \sqrt{\hat{\rho}}\, \hat{v}_\mathrm{th}\,,
            \qquad \hat{t} = \frac{\hat{x}}{\hat{v}_\mathrm{th}}\,.

        The equations then have no dimensionless parameters."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy: :math:`E_U = \frac{1}{2} \int (\rho_0 + \tilde \rho) |\tilde{\mathbf{U}}|^2 \, \mathrm{d}V`
          (``en_U``), with the total density (projected :math:`\rho_0`)
        - Quadratic thermal energy of the linear sound wave step: :math:`\frac{2}{\gamma}\int \tilde{q}^2 \, \mathrm{d}V`
          (``en_thermal``)
        - Quadratic thermal energy of the nonlinear sound wave step: :math:`\frac{1}{\gamma - 1}\int \tilde{q}^2 \, \mathrm{d}V`
          (``en_q2``)
        - Internal energy of the perturbation: :math:`E_\mathrm{int} = \int \frac{2 q_0 \tilde q + \tilde q^2}{\gamma - 1}
          \, \mathrm{d}V = \int \frac{q^2 - q_0^2}{\gamma - 1} \, \mathrm{d}V` (``en_internal``)
        - Equilibrium internal energy: :math:`E_{p0} = \int \frac{p_0}{\gamma - 1} \, \mathrm{d}V` (``en_p_eq``)
        - Energy of the perturbation: :math:`E_{\mathrm{tot}} = E_U + E_\mathrm{int}` (``en_tot``), conserved by the
          equations for :math:`\nabla q_0 = 0`. Each step of the splitting conserves its own energy exactly (the linear
          sound wave step :math:`E_U` + ``en_thermal``, the nonlinear one :math:`E_U` + ``en_q2``, the advection step
          :math:`E_U`), so :math:`E_{\mathrm{tot}}` changes by the splitting error only."""

    @classmethod
    def doc_long_description(cls):
        r"""This model simulates finite-amplitude perturbations of a compressible, inviscid fluid around a static
        equilibrium (e.g. nonlinear sound waves up to the formation of shocks). It solves the full compressible Euler
        equations for the perturbations of the density, the velocity and the square root of the pressure,
        :math:`q = \sqrt p`, written as the linear system of :class:`~struphy.models.LinearEulerWithQ` plus the
        nonlinear terms.

        The time step reuses the steps of the linear model and adds the steps of the nonlinear terms:

        - the steps of :class:`~struphy.models.LinearEulerWithQ` (background :math:`q` step and linear sound wave step,
          one symmetric block), with the inertia of the total density :math:`\rho_0 + \tilde \rho^n` instead of
          :math:`\rho_0` and without the continuity equation;
        - ``nonlin_sound``: the terms of the sound waves quadratic in :math:`\tilde q`, with the transport operator
          :math:`\mathcal S_{\tilde q}` and its transpose, so that it conserves
          :math:`\frac 12 \int \rho |\tilde{\mathbf U}|^2 + \frac{1}{\gamma - 1} \int \tilde q^2` exactly;
        - ``mom_adv``: the continuity equation and the advection of the momentum, which conserves the mass and
          :math:`\frac 12 \int \rho |\tilde{\mathbf U}|^2` exactly (discrete gradient, Picard iteration).

        All steps are energy conserving, each for its own energy; the nonlinear steps have the weights of the nonlinear
        terms at :math:`t^n` by default (``Options(weights="frozen")``) or at :math:`t^{n+1/2}`
        (``weights="midpoint"``, Picard iteration, second order in time). The inertia of every step is that of the total
        density at :math:`t^n`, which is not changed except in ``mom_adv``. No numerical dissipation is added, so
        shocks are not captured; the model is meant for smooth solutions."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a perturbative Euler model in :math:`q`:

        .. code-block:: python

            from struphy import BaseUnits
            from struphy.models import PerturbativeEulerWithQ

            model = PerturbativeEulerWithQ(base_units=BaseUnits(kBT=1.0))

            # Access fields
            model.mhd.density
            model.mhd.velocity
            model.mhd.sqrt_p

            # second order in time: midpoint weights of the nonlinear steps
            props = model.propagators
            props.nonlin_sound.options = props.nonlin_sound.Options(weights="midpoint")
            props.mom_adv.options = props.mom_adv.Options(weights="midpoint")

            # Access tracked scalar quantities
            model.scalars["en_U"]
            model.scalars["en_internal"]
            model.scalars["en_tot"]
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for finite-amplitude perturbations of a static compressible fluid.

        Typical use cases include:

        - nonlinear sound waves (steepening, before the formation of shocks)
        - pressureless and compressible advection of density and momentum
        - the nonlinear fluid part of :class:`~struphy.models.PerturbativeMHDWithQ`"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - shocks and other discontinuities (no numerical dissipation)
        - equilibria with non-zero background flow, or with a pressure gradient (no force balance without a magnetic
          field)
        - dissipative effects such as viscosity or heat conduction
        - magnetized plasmas (see :class:`~struphy.models.PerturbativeMHDWithQ`)"""
