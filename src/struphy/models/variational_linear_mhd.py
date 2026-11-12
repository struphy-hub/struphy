from struphy.feec.mass import L2Projector
from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC
from struphy.models.species import (
    DiagnosticSpecies,
    FieldSpecies,
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.variational_density_evolve import VariationalDensityEvolve
from struphy.propagators.variational_linear_alfven_evolve import VariationalLinearAlfvenEvolve
from struphy.propagators.variational_linear_sound_evolve import VariationalLinearSoundEvolve
from struphy.propagators.variational_resistivity import VariationalResistivity
from struphy.propagators.variational_viscosity import VariationalViscosity


class VariationalLinearMHD(StruphyModel):
    """Linear visco-resistive MHD equations around a zero-flow equilibrium, with the pressure variable, discretized
    with a variational method.

    The models of the variational MHD family inherit from this model
    (:class:`~struphy.models.variational_perturbative_mhd.VariationalPerturbativeMHD` adds the terms that are nonlinear
    in the perturbations): they call ``super().__init__()`` and extend the hooks of
    :meth:`~struphy.models.base.StruphyModel._setup`.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mass_number: float
        Mass number (in units of Proton mass) of the fluid species (default: 1.0)
    with_viscosity: bool
        Whether to include viscous dissipation (default: True)
    with_resistivity: bool
        Whether to include resistive dissipation (default: True)
    u_space: str
        Space of the velocity, ``"H1vec"`` (vector field) or ``"Hdiv"`` (2-form, as in
        :class:`~struphy.models.linear_mhd.LinearMHD`) (default: ``"H1vec"``)
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    class MHD(FluidSpecies):
        def __init__(self, mass_number: float = 1.0, u_space: str = "H1vec"):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space=u_space)
            self.pressure = FEECVariable(space="L2")
            self.init_variables(mass_number=mass_number)

    class Diagnostics(DiagnosticSpecies):
        def __init__(self):
            self.div_u = FEECVariable(space="L2")
            self.u2 = FEECVariable(space="Hdiv")
            self.pt3 = FEECVariable(space="L2")
            self.bt2 = FEECVariable(space="Hdiv")
            self.init_variables()

    ## propagators

    class Propagators:
        """Propagators of the linear model: the density, the coupling of the velocity and the magnetic field
        (shear-Alfvén and fast waves) and of the velocity and the pressure (sound waves), one linear solve each, and the
        dissipation (which heats the second-order pressure ``pt3``).

        Parameters
        ----------
        div_u, u2, pt3, bt2 : FEECVariable, optional
            Diagnostics: the divergence and the 2-form of the velocity, the second-order pressure and magnetic field.
        rho : FEECVariable, optional
            The density perturbation (for the mass matrix of the dissipative propagators).
        with_viscosity, with_resistivity : bool
            Whether to include the dissipative propagators.
        """

        def __init__(
            self,
            div_u: FEECVariable = None,
            u2: FEECVariable = None,
            pt3: FEECVariable = None,
            bt2: FEECVariable = None,
            rho: FEECVariable = None,
            with_viscosity: bool = True,
            with_resistivity: bool = True,
        ):
            self.variat_dens = VariationalDensityEvolve()
            self.variat_lin_alfven = VariationalLinearAlfvenEvolve(bt2=bt2)
            self.variat_lin_sound = VariationalLinearSoundEvolve(div_u=div_u, u2=u2, pt3=pt3)
            # the linear model heats the second-order pressure p-hat
            if with_viscosity:
                self.variat_viscous = VariationalViscosity(rho=rho, pt3=pt3)
            if with_resistivity:
                self.variat_resist = VariationalResistivity(rho=rho, pt3=pt3)

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    em_fields: EMFields
    mhd: MHD
    diagnostics: Diagnostics
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mass_number: float = 1.0,
        with_viscosity: bool = True,
        with_resistivity: bool = True,
        u_space: str = "H1vec",
    ):
        assert u_space in ("H1vec", "Hdiv"), f'u_space must be "H1vec" or "Hdiv", got {u_space!r}.'
        # 0. store input parameters (a model inheriting from this one stores its own before calling super().__init__)
        self._store_params(locals())
        self._mass_number = mass_number
        self._with_viscosity = with_viscosity
        self._with_resistivity = with_resistivity
        self._u_space = u_space

        # 1.-5. species, units, propagators, variable links and scalars, see StruphyModel._setup
        self._setup(base_units)

    def _create_species(self):
        """:meta private:"""
        self.em_fields = self.EMFields()
        self.mhd = self.MHD(mass_number=self._mass_number, u_space=self._u_space)
        self.diagnostics = self.Diagnostics()

    def _propagator_args(self):
        """:meta private:"""
        return {
            "div_u": self.diagnostics.div_u,
            "u2": self.diagnostics.u2,
            "pt3": self.diagnostics.pt3,
            "bt2": self.diagnostics.bt2,
            "rho": self.mhd.density,
            "with_viscosity": self._with_viscosity,
            "with_resistivity": self._with_resistivity,
        }

    def _link_propagator_variables(self):
        """:meta private:"""
        props = self.propagators
        props.variat_dens.variables.rho = self.mhd.density
        props.variat_dens.variables.u = self.mhd.velocity
        props.variat_lin_alfven.variables.u = self.mhd.velocity
        props.variat_lin_alfven.variables.b = self.em_fields.b_field
        props.variat_lin_sound.variables.u = self.mhd.velocity
        props.variat_lin_sound.variables.s = self.mhd.pressure
        if self._with_viscosity:
            props.variat_viscous.variables.s = self.mhd.pressure
            props.variat_viscous.variables.u = self.mhd.velocity
        if self._with_resistivity:
            props.variat_resist.variables.s = self.mhd.pressure
            props.variat_resist.variables.b = self.em_fields.b_field

    def _kinetic_energy(self):
        """The kinetic energy with the velocity mass matrix of the propagators (density-weighted)."""
        return BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="M2n" if self._u_space == "Hdiv" else "WMMnew")

    def _create_scalars(self):
        """:meta private:"""
        kinetic_energy = self._kinetic_energy()
        magnetic_energy_1 = BilinearEnergyFEEC(self.em_fields.b_field)
        # second-order magnetic energy b0^T M2 b-hat
        magnetic_energy_2 = BilinearEnergyFEEC(self.diagnostics.bt2, right_variable="b2", normalization=2.0)
        thermo_energy = FunctionScalarFEEC(self._compute_en_thermo)
        thermo_energy_l1 = FunctionScalarFEEC(self._compute_en_thermo_l1)
        magnetic_energy_l1 = BilinearEnergyFEEC(self.em_fields.b_field, right_variable="b2", normalization=2.0)
        return {
            "en_U": kinetic_energy,
            "en_thermo": thermo_energy,
            "en_mag_1": magnetic_energy_1,
            "en_mag_2": magnetic_energy_2,
            "en_tot": kinetic_energy + thermo_energy + magnetic_energy_1 + magnetic_energy_2,
            "en_tot_l1": thermo_energy_l1 + magnetic_energy_l1,
            "en_thermo_l1": thermo_energy_l1,
            "en_mag_l1": magnetic_energy_l1,
        }

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    @property
    def propagator_variants(self) -> dict:
        """The values of the option ``model`` of the variational propagators of this model."""
        return {
            "variat_dens": "linear",
            "variat_lin_sound": "linear_p",
            "variat_viscous": "linear_p",
            "variat_resist": "linear_p",
        }

    def post_allocate(self):
        """:meta private:"""
        # the propagator options must match the model
        for name, variant in self.propagator_variants.items():
            prop = getattr(self.propagators, name, None)
            if prop is not None and prop.options.model != variant:
                raise ValueError(
                    f"{self.__class__.__name__} needs {name}.Options(model='{variant}'), got model='{prop.options.model}'."
                )

        # the dissipative heating uses the gamma of the pressure propagator
        gamma = self.propagators.variat_lin_sound.options.gamma
        for name in ("variat_viscous", "variat_resist"):
            prop = getattr(self.propagators, name, None)
            if prop is not None and prop.options.gamma != gamma:
                raise ValueError(
                    f"{name}.Options(gamma={prop.options.gamma}) must match the gamma={gamma} of the pressure."
                )

        # L2 dofs of sqrt(g), the linear form of the integral over the domain of a 3-form: p . vol_dofs = int p dx
        self._vol_dofs = L2Projector("L2", Propagator.mass_ops).get_dofs(Propagator.domain.jacobian_det)

    def _compute_en_thermo(self):
        pt3 = self.diagnostics.pt3.spline.vector
        gamma = self.propagators.variat_lin_sound.options.gamma
        return pt3.inner(self._vol_dofs) / (gamma - 1.0)

    def _compute_en_thermo_l1(self):
        p = self.mhd.pressure.spline.vector
        gamma = self.propagators.variat_lin_sound.options.gamma
        return p.inner(self._vol_dofs) / (gamma - 1.0)

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        with open(params_path, "r") as f:
            lines = f.readlines()
        with open(params_path, "w") as f:
            for line in self._edit_default_parameter_lines(lines):
                f.write(line)

    def _edit_default_parameter_lines(self, lines: list) -> list:
        """The options ``model`` of the variational propagators, and the density background."""
        variants = self.propagator_variants
        new_lines = []
        for line in lines:
            name = next((n for n in variants if f"{n}.Options" in line), None)
            if name is not None:
                new_lines += [
                    f"model.propagators.{name}.options = model.propagators.{name}.Options(model='{variants[name]}')\n",
                ]
            elif "pressure.add_background" in line:
                new_lines += ["model.mhd.density.add_background(FieldsBackground())\n", line]
            else:
                new_lines += [line]
        return new_lines

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        The unknowns are the perturbations :math:`(\tilde\rho, \tilde p, \tilde{\mathbf B})` with respect to a
        zero-flow MHD equilibrium :math:`(\rho_0, p_0, \mathbf B_0)`, and the velocity :math:`\tilde{\mathbf u}`:

        .. math::

            &\partial_t \tilde{\rho} + \nabla \cdot (\rho_0 \tilde{\mathbf{u}}) = 0\,,
            \\[2mm]
            &\partial_t (\rho_0 \tilde{\mathbf{u}}) + \nabla \tilde{p} + \mathbf{B}_0 \times \nabla \times \tilde{\mathbf{B}} + \tilde{\mathbf{B}} \times \nabla \times \mathbf{B}_0 - \nabla \cdot \left( (\mu + \mu_a(\mathbf{x})) \nabla \tilde{\mathbf{u}} \right) = 0\,,
            \\[2mm]
            &\partial_t \tilde{p} + \tilde{\mathbf{u}} \cdot \nabla p_0 + \gamma p_0 \nabla \cdot \tilde{\mathbf{u}} = 0\,,
            \\[2mm]
            &\partial_t \tilde{\mathbf{B}} + \nabla \times (\mathbf{B}_0 \times \tilde{\mathbf{u}}) + \nabla \times (\eta + \eta_a(\mathbf{x})) \nabla \times \tilde{\mathbf{B}} = 0\,.

        Here :math:`\mu_a(\mathbf{x})` and :math:`\eta_a(\mathbf{x})` are artificial viscosity and resistivity
        coefficients.

        **Conserved quantities:** the energy

        .. math::

            \frac12 \int \rho_0 |\tilde{\mathbf u}|^2 + \frac12 \int |\tilde{\mathbf B}|^2
            + \int \mathbf B_0 \cdot \hat{\mathbf B} + \int \frac{\hat p}{\gamma - 1} \,,

        with the second-order magnetic field :math:`\hat{\mathbf B}` and pressure :math:`\hat p`, advected with the
        perturbations as weights, :math:`\partial_t \hat{\mathbf B} + \nabla \times (\tilde{\mathbf B} \times
        \tilde{\mathbf u}) = 0`, :math:`\partial_t \hat p + \nabla \cdot (\tilde p\, \tilde{\mathbf u}) + (\gamma - 1)
        \tilde p\, \nabla \cdot \tilde{\mathbf u} = 0`, for :math:`\tilde{\mathbf u} \cdot \mathbf n = 0` and
        :math:`\mathbf B_0 \cdot \mathbf n = 0` on the boundary. The viscous and resistive heating, quadratic in the
        perturbations, goes into :math:`\hat p` (diagnostic ``pt3``), so the energy is conserved also with
        dissipation. The forces of the second-order terms are the variations of the discrete Lagrangian (see
        :class:`~struphy.propagators.variational_linear_alfven_evolve.VariationalLinearAlfvenEvolve` and
        :class:`~struphy.propagators.variational_linear_sound_evolve.VariationalLinearSoundEvolve`).

        **Velocity in** :math:`H(\textnormal{div})` (``u_space="Hdiv"``): the velocity is a 2-form, as in
        :class:`~struphy.models.linear_mhd.LinearMHD`, with the boundary condition :math:`\tilde{\mathbf u} \cdot
        \mathbf n = 0` instead of :math:`\tilde{\mathbf u} = 0`. The viscous term is then discretized with the
        divergence and the (weak) curl of :math:`\tilde{\mathbf u}`,
        :math:`\int \mu (\nabla \cdot \mathbf u\, \nabla \cdot \mathbf v + \nabla \times \mathbf u \cdot \nabla \times
        \mathbf v)`, which equals :math:`\int \mu \nabla \mathbf u : \nabla \mathbf v` up to boundary terms; the wall
        condition is perfect slip.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The velocity scale is Alfvénic:

        .. math::

            \hat u = \hat v_A.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy: ``en_U``
        - Thermal energies: ``en_thermo`` (of the second-order pressure), ``en_thermo_l1``
        - Magnetic energies: ``en_mag_1``, ``en_mag_2`` (second-order term), ``en_mag_l1``
        - Total energies: ``en_tot`` (conserved), ``en_tot_l1``"""

    @classmethod
    def doc_long_description(cls):
        r"""VariationalLinearMHD solves linear visco-resistive MHD around a zero-flow equilibrium, with the pressure as
        thermodynamic variable. The discretization is variational: the forces are the variations of a discrete
        Lagrangian (magnetic energy with the Galerkin Hodge star, interior products with the commuting projections),
        and the time stepping (one linear solve per substep, implicit midpoint rule) conserves the energy exactly. It is
        the parent of :class:`~struphy.models.variational_perturbative_mhd.VariationalPerturbativeMHD`."""

    @classmethod
    def doc_examples(cls):
        r"""Create the linear model, with the velocity in :math:`(H^1)^3` or in :math:`H(\textnormal{div})`:

        .. code-block:: python

            from struphy.models import VariationalLinearMHD

            model = VariationalLinearMHD()
            model_hdiv = VariationalLinearMHD(u_space="Hdiv", with_viscosity=False, with_resistivity=False)
            model.mhd.pressure
            model.em_fields.b_field
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - linear MHD waves and instabilities around a prescribed equilibrium
        - comparing the variational linear MHD with the velocity in :math:`H(\textnormal{div})`
          (``u_space="Hdiv"``) with :class:`~struphy.models.linear_mhd.LinearMHD`"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - nonlinear MHD (see :class:`~struphy.models.variational_perturbative_mhd.VariationalPerturbativeMHD`)
        - kinetic or particle-resolved plasma effects
        - entropy- or q-based thermodynamic formulations
        - exact energy balance of the dissipation with polar splines (the heating of the auxiliary pressure is exact
          only if the constants are in the space of 3-forms, which is not the case for polar splines)"""
