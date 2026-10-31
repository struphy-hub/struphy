from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.linear_euler_mixins import EquilibriumPressureMixin
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC
from struphy.models.species import FluidSpecies
from struphy.models.variables import FEECVariable
from struphy.propagators.background_q_gradient import BackgroundQGradient
from struphy.propagators.sound_wave_linear_with_q import SoundWaveLinearWithQ


class LinearEulerWithQ(EquilibriumPressureMixin, StruphyModel):
    """Linearized compressible Euler equations around a static (zero-flow) equilibrium, for sound waves, with the
    square root of the pressure, :math:`q = \\sqrt p`, as thermodynamic variable.

    The models of the LinearEulerWithQ family inherit from this model: they call ``super().__init__()`` and extend the
    hooks of :meth:`~struphy.models.base.StruphyModel._setup` (species, propagators, variable links, scalars).

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: ``BaseUnits(kBT=1.0)``).
    mass_number: float
        Mass number (in units of Proton mass) of the fluid species (default: 1.0)
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class MHD(FluidSpecies):
        """Fluid species of the LinearEulerWithQ family: density, velocity and :math:`q = \\sqrt p` perturbations
        (named ``mhd`` in all models of the family)."""

        def __init__(self, mass_number: float = 1.0):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="Hdiv")
            self.sqrt_p = FEECVariable(space="L2")
            self.init_variables(mass_number=mass_number)

    ## propagators

    class Propagators:
        """Propagators of the linearized Euler equations in :math:`q`: ``bg_q`` (advection of the background
        :math:`q_0`) and ``sound`` (sound waves, energy conserving), integrated as one symmetric (Strang) block with
        ``sound`` in the middle, second order in time.

        Parameters
        ----------
        u_field : FEECVariable, optional
            Velocity perturbation, which ``bg_q`` advects the background :math:`q_0` with (default: None).
        rho_field : FEECVariable, optional
            Density perturbation, for the inertia of the total density in ``sound`` (perturbative models; default:
            None, the inertia of the equilibrium density).
        """

        strang_groups = (("bg_q", "sound"),)

        def __init__(self, u_field: FEECVariable = None, rho_field: FEECVariable = None):
            self.bg_q = BackgroundQGradient(u_field=u_field)
            self.sound = SoundWaveLinearWithQ(rho_field=rho_field)

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    mhd: MHD
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        mass_number: float = 1.0,
    ):
        # 0. store input parameters (a model inheriting from LinearEulerWithQ stores its own before calling
        # super().__init__)
        self._store_params(locals())
        # for the hooks: the arguments of this constructor
        self._mass_number = mass_number

        # 1.-5. species, units, propagators, variable links and scalars, see StruphyModel._setup
        self._setup(base_units)

    def _create_species(self):
        """:meta private:"""
        self.mhd = self.MHD(mass_number=self._mass_number)

    def _propagator_args(self):
        """:meta private:"""
        return {"u_field": self.mhd.velocity}

    def _link_propagator_variables(self):
        """:meta private:"""
        self.propagators.sound.variables.n = self.mhd.density
        self.propagators.sound.variables.u = self.mhd.velocity
        self.propagators.sound.variables.q = self.mhd.sqrt_p

        self.propagators.bg_q.variables.q = self.mhd.sqrt_p

    def _create_scalars(self):
        """:meta private:"""
        en_U = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="M2n")
        # (2 / gamma) q^T M3 q, the energy of the sound wave step
        en_thermal = BilinearEnergyFEEC(self.mhd.sqrt_p, bilinear_form_name="M3", normalization=4.0 / (5 / 3))
        en_p_eq = FunctionScalarFEEC(self._compute_en_p_eq)
        return {"en_U": en_U, "en_thermal": en_thermal, "en_p_eq": en_p_eq, "en_tot": en_U + en_thermal}

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "thermal"

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity (mass conservation):

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot (\rho_0 \tilde{\mathbf{U}}) = 0

        Momentum:

        .. math::

            \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla (2 q_0 \tilde q) = 0

        Square root of the pressure (adiabatic process):

        .. math::

            \frac{\partial \tilde{q}}{\partial t} + \tilde{\mathbf{U}} \cdot \nabla q_0
            + \frac{\gamma}{2} q_0 \nabla \cdot \tilde{\mathbf{U}} = 0\,.

        Here, :math:`\tilde \rho`, :math:`\tilde{\mathbf U}`, :math:`\tilde q` are the perturbed density, velocity and
        square root of the pressure, :math:`\rho_0`, :math:`q_0 = \sqrt{p_0}` those of a static equilibrium (zero
        flow, :math:`\mathbf U_0 = 0`), and :math:`\gamma = 5/3`. The equation of :math:`q` is the linearization of
        :math:`\partial_t q + \mathbf U \cdot \nabla q + \frac{\gamma}{2} q \nabla \cdot \mathbf U = 0`, which follows
        from the adiabatic pressure equation with :math:`p = q^2`; the pressure perturbation is
        :math:`\tilde p = 2 q_0 \tilde q`, and the equations are those of :class:`~struphy.models.LinearEuler` written
        in :math:`\tilde q`. Force balance of the equilibrium requires :math:`\nabla q_0 = 0`; the term
        :math:`\tilde{\mathbf U} \cdot \nabla q_0` is nevertheless kept, since it is part of the models of the family
        whose equilibria carry a pressure gradient (e.g. :class:`~struphy.models.LinearMHDWithQ`).
        For constant :math:`\rho_0`, :math:`q_0` the perturbations are sound waves with the speed
        :math:`c_\textnormal{s} = \sqrt{\gamma q_0^2 / \rho_0}`.

        **Conserved quantities** (for :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` on the boundary, or periodic
        boundaries):

        Mass perturbation:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \tilde \rho \, \textnormal d \mathbf x = 0

        Perturbed energy, for an equilibrium with :math:`\nabla q_0 = 0`:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \left( \frac{\rho_0}{2} |\tilde{\mathbf U}|^2
            + \frac{2}{\gamma} \tilde q^2 \right) \textnormal d \mathbf x = 0
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

        - Kinetic energy (perturbation): :math:`E_U = \frac{1}{2} \int \rho_0 |\tilde{\mathbf{U}}|^2 \, \mathrm{d}V` (``en_U``)
        - Thermal perturbation energy: :math:`E_{\mathrm{thermal}} = \frac{2}{\gamma}\int \tilde{q}^2 \, \mathrm{d}V`
          (``en_thermal``), with :math:`\gamma = 5/3`; it equals :math:`\frac 12 \int \tilde p^2 / (\gamma p_0)` for
          :math:`\tilde p = 2 q_0 \tilde q`
        - Equilibrium internal energy: :math:`E_{p0} = \int \frac{p_0}{\gamma - 1} \, \mathrm{d}V` (``en_p_eq``)
        - Total perturbed energy: :math:`E_{\mathrm{tot}} = E_U + E_{\mathrm{thermal}}` (``en_tot``). The sound wave
          step conserves it up to the solver tolerances; the background step changes it only for
          :math:`\nabla q_0 \neq 0`."""

    @classmethod
    def doc_long_description(cls):
        r"""This model simulates small-amplitude perturbations of a compressible, inviscid fluid around a static
        equilibrium. It solves the linearized Euler equations for the perturbations of the density, the velocity and
        the square root of the pressure, :math:`q = \sqrt p`. Its only waves are sound waves; with a non-uniform
        equilibrium density they are refracted. It solves the same equations as :class:`~struphy.models.LinearEuler`,
        written in :math:`\tilde q = \tilde p / (2 q_0)`; it is the parent of the family with the variable :math:`q`,
        whose nonlinear (perturbative) models keep the terms quadratic in :math:`\tilde q` in the energy.

        :class:`~struphy.models.LinearMHDWithQ` adds the magnetic field (shear-Alfvén waves and the background
        current), and inherits the species, propagators and scalars of this model.

        The time step is split into the sound wave step, which conserves the perturbed energy
        :math:`\frac 12 \int \rho_0 |\tilde{\mathbf U}|^2 \,\textnormal d \mathbf x
        + \frac{2}{\gamma} \int \tilde q^2 \,\textnormal d \mathbf x` exactly (Crank-Nicolson; the force is the exact
        transpose of the operator of the :math:`\tilde q` equation, so the discrete spectrum is purely imaginary and
        the step does not damp or amplify the waves), and the exact-in-time step of the term
        :math:`- \tilde{\mathbf U} \cdot \nabla q_0`, which vanishes for an equilibrium in force balance. The two
        steps do not commute; they are integrated with symmetric (Strang) splitting, the sound wave step in the middle,
        such that the time step is second order in time (also with ``Time(split_algo="LieTrotter")``).

        The equilibrium is given by an MHD or fluid equilibrium (``equil``) providing :math:`n_0` and :math:`p_0`; a
        magnetic field of the equilibrium is not used. With free boundary conditions the pressure term imposes
        :math:`\tilde q = 0` weakly at the boundary; with essential boundary conditions on the velocity (Dirichlet in
        the de Rham sequence), :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` holds strongly (rigid wall)."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a linear Euler model in :math:`q`:

        .. code-block:: python

            from struphy import BaseUnits
            from struphy.models import LinearEulerWithQ

            model = LinearEulerWithQ(base_units=BaseUnits(kBT=1.0))

            # Access fields
            model.mhd.density
            model.mhd.velocity
            model.mhd.sqrt_p

            # Access tracked scalar quantities
            model.scalars["en_U"]
            model.scalars["en_thermal"]
            model.scalars["en_tot"]
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for studying linear sound waves in a static compressible fluid with the
        variable :math:`q = \sqrt p`.

        Typical use cases include:

        - verification of the sound wave dispersion relation and of eigenmodes in curvilinear geometries
        - sound waves in non-uniform equilibrium densities (refraction)
        - the linear limit of the perturbative models in :math:`q` and testing the fluid part of
          :class:`~struphy.models.LinearMHDWithQ`"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - nonlinear fluid dynamics with finite-amplitude perturbations (shocks)
        - equilibria with non-zero background flow
        - dissipative effects such as viscosity or heat conduction
        - magnetized plasmas (see :class:`~struphy.models.LinearMHDWithQ`)"""
