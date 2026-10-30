from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC, VolumeFormEnergyFEEC
from struphy.models.species import FluidSpecies
from struphy.models.variables import FEECVariable
from struphy.polar.basic import PolarVector
from struphy.propagators.background_pressure_gradient import BackgroundPressureGradient
from struphy.propagators.base import Propagator
from struphy.propagators.sound_wave_linear import SoundWaveLinear


class LinearEuler(StruphyModel):
    """Linearized compressible Euler equations around a static (zero-flow) equilibrium, for sound waves.

    The models of the LinearEuler family inherit from this model: they call ``super().__init__()`` and extend the hooks
    of :meth:`~struphy.models.base.StruphyModel._setup` (species, propagators, variable links, scalars).

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
        """Fluid species of the LinearEuler family: density, velocity and pressure perturbations (named ``mhd`` in
        all models of the family)."""

        def __init__(self, mass_number: float = 1.0):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="Hdiv")
            self.pressure = FEECVariable(space="L2")
            self.init_variables(mass_number=mass_number)

    ## propagators

    class Propagators:
        """Propagators of the linearized Euler equations: ``sound`` (sound waves, energy conserving) and
        ``bg_pressure`` (advection of the background pressure).

        Parameters
        ----------
        u_field : FEECVariable, optional
            Velocity perturbation, which ``bg_pressure`` advects the background pressure with (default: None).
        """

        def __init__(self, u_field: FEECVariable = None):
            self.sound = SoundWaveLinear()
            self.bg_pressure = BackgroundPressureGradient(u_field=u_field)

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    mhd: MHD
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        mass_number: float = 1.0,
    ):
        # 0. store input parameters (a model inheriting from LinearEuler stores its own before calling super().__init__)
        self._store_params(locals())
        # for the hooks: the arguments of this constructor (self.params holds those of the most derived model, whose
        # names can differ, e.g. mhd_mass_number of the hybrid models)
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
        self.propagators.sound.variables.p = self.mhd.pressure

        self.propagators.bg_pressure.variables.p = self.mhd.pressure

    def _create_scalars(self):
        """:meta private:"""
        en_U = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="M2n")
        # The volume integral is retained as a diagnostic of the mean pressure perturbation.
        # It is linear and is not part of the linearized energy.
        en_p = VolumeFormEnergyFEEC(self.mhd.pressure, normalization=1.0 / (5 / 3 - 1))
        en_thermal = BilinearEnergyFEEC(self.mhd.pressure, bilinear_form_name="M3p_inv", normalization=1.0 / (5 / 3))
        en_p_eq = FunctionScalarFEEC(self._compute_en_p_eq)
        return {"en_U": en_U, "en_p": en_p, "en_thermal": en_thermal, "en_p_eq": en_p_eq, "en_tot": en_U + en_thermal}

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "thermal"

    def post_allocate(self):
        """:meta private:"""
        if Propagator.projected_equil is None:
            raise ValueError(
                f"{type(self).__name__} needs an equilibrium (Simulation(equil=...)) providing n0 and p0, e.g. "
                "equils.HomogenSlab."
            )
        self._ones = Propagator.projected_equil.p3.space.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

    def _compute_en_p_eq(self):
        return Propagator.projected_equil.p3.inner(self._ones) / (5 / 3 - 1)

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity (mass conservation):

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot (\rho_0 \tilde{\mathbf{U}}) = 0

        Momentum:

        .. math::

            \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde{p} = 0

        Energy (adiabatic process):

        .. math::

            \frac{\partial \tilde{p}}{\partial t} + \nabla \cdot (p_0 \tilde{\mathbf{U}}) + (\gamma - 1) p_0 \nabla \cdot \tilde{\mathbf{U}} = 0\,.

        Here, :math:`\tilde \rho`, :math:`\tilde{\mathbf U}`, :math:`\tilde p` are the perturbed density, velocity and
        pressure, :math:`\rho_0`, :math:`p_0` the density and pressure of a static equilibrium (zero flow,
        :math:`\mathbf U_0 = 0`), and :math:`\gamma = 5/3`. Force balance of the equilibrium requires
        :math:`\nabla p_0 = 0` (the density :math:`\rho_0` can vary); the term
        :math:`\tilde{\mathbf U} \cdot \nabla p_0` of the energy equation is nevertheless kept, since it is part of
        the models of the family whose equilibria carry a pressure gradient (e.g.
        :math:`\nabla p_0 = \mathbf J_0 \times \mathbf B_0` in :class:`~struphy.models.LinearMHD`).
        For constant :math:`\rho_0`, :math:`p_0` the perturbations are sound waves with the speed
        :math:`c_\textnormal{s} = \sqrt{\gamma p_0 / \rho_0}`.

        **Conserved quantities** (for :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` on the boundary, or periodic
        boundaries):

        Mass perturbation:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \tilde \rho \, \textnormal d \mathbf x = 0

        Perturbed energy, for an equilibrium with :math:`\nabla p_0 = 0`:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \left( \frac{\rho_0}{2} |\tilde{\mathbf U}|^2
            + \frac{\tilde p^2}{2 \gamma p_0} \right) \textnormal d \mathbf x = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""All velocities are normalized by the thermal velocity of the bulk species, defined by the unit of the
        temperature :math:`\hat{k_B T}` (``BaseUnits.kBT``):

        .. math::

            \hat{U} = \hat{v}_\mathrm{th} = \sqrt{\frac{\hat{k_B T}}{A m_\mathrm{H}}}\,, \qquad
            \hat{\rho} = A m_\mathrm{H} \hat{n}\,, \qquad \hat{p} = \hat{\rho}\, \hat{v}_\mathrm{th}^2 = \hat{n} \hat{k_B T}\,,
            \qquad \hat{t} = \frac{\hat{x}}{\hat{v}_\mathrm{th}}\,.

        The equations then have no dimensionless parameters."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy (perturbation): :math:`E_U = \frac{1}{2} \int \rho_0 |\tilde{\mathbf{U}}|^2 \, \mathrm{d}V` (``en_U``)
        - Mean pressure perturbation: :math:`E_p = \int \frac{\tilde{p}}{\gamma - 1} \, \mathrm{d}V` (``en_p``). This is a linear diagnostic, not an energy.
        - Thermal perturbation energy: :math:`E_{\mathrm{thermal}} = \frac{1}{2}\int \frac{\tilde{p}^2}{\gamma p_0} \, \mathrm{d}V` (``en_thermal``), with :math:`\gamma = 5/3`
        - Equilibrium internal energy: :math:`E_{p0} = \int \frac{p_0}{\gamma - 1} \, \mathrm{d}V` (``en_p_eq``)
        - Total perturbed energy: :math:`E_{\mathrm{tot}} = E_U + E_{\mathrm{thermal}}` (``en_tot``). The sound wave
          step conserves it up to the solver tolerances; the background pressure step changes it only for
          :math:`\nabla p_0 \neq 0`."""

    @classmethod
    def doc_long_description(cls):
        r"""This model simulates small-amplitude perturbations of a compressible, inviscid fluid around a static
        equilibrium. It solves the linearized Euler equations for the density, velocity and pressure perturbations.
        Its only waves are sound waves; with a non-uniform equilibrium density they are refracted.

        This is the parent model of the LinearEuler family: :class:`~struphy.models.LinearMHD` adds the magnetic
        field (shear-Alfvén waves and the background current), and inherits the species, propagators and scalars
        of this model.

        The time step is split into the sound wave step, which conserves the perturbed energy
        :math:`\frac 12 \int \rho_0 |\tilde{\mathbf U}|^2 \,\textnormal d \mathbf x
        + \frac{1}{2\gamma} \int \tilde p^2 / p_0 \,\textnormal d \mathbf x` exactly (Crank-Nicolson with the
        :math:`1/p_0`-weighted pressure equation), and the exact-in-time step of the term
        :math:`- \tilde{\mathbf U} \cdot \nabla p_0`, which vanishes for an equilibrium in force balance.

        The equilibrium is given by an MHD or fluid equilibrium (``equil``) providing :math:`n_0` and :math:`p_0`; a
        magnetic field of the equilibrium is not used. With free boundary conditions the pressure term imposes
        :math:`\tilde p = 0` weakly at the boundary; with essential boundary conditions on the velocity (Dirichlet in
        the de Rham sequence), :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` holds strongly (rigid wall)."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a linear Euler model:

        .. code-block:: python

            from struphy import BaseUnits
            from struphy.models import LinearEuler

            model = LinearEuler(base_units=BaseUnits(kBT=1.0))

            # Access fields
            model.mhd.density
            model.mhd.velocity
            model.mhd.pressure

            # Access tracked scalar quantities
            model.scalars["en_U"]
            model.scalars["en_thermal"]
            model.scalars["en_tot"]
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for studying linear sound waves in a static compressible fluid.

        Typical use cases include:

        - verification of the sound wave dispersion relation and of eigenmodes in curvilinear geometries
        - sound waves in non-uniform equilibrium densities (refraction)
        - testing the fluid part of the linear MHD models of the family"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - nonlinear fluid dynamics with finite-amplitude perturbations (shocks)
        - equilibria with non-zero background flow
        - dissipative effects such as viscosity or heat conduction
        - magnetized plasmas (see :class:`~struphy.models.LinearMHD`)"""
