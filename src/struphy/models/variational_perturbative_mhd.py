import copy

from struphy.feec.mass import L2Projector
from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC, Scalars
from struphy.models.species import (
    DiagnosticSpecies,
    FieldSpecies,
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.propagators.utils import VelocityMass
from struphy.propagators.variational_density_evolve import VariationalDensityEvolve
from struphy.propagators.variational_linear_alfven_evolve import VariationalLinearAlfvenEvolve
from struphy.propagators.variational_linear_sound_evolve import VariationalLinearSoundEvolve
from struphy.propagators.variational_momentum_advection import VariationalMomentumAdvection
from struphy.propagators.variational_pb_evolve import VariationalPBEvolve
from struphy.propagators.variational_resistivity import VariationalResistivity
from struphy.propagators.variational_viscosity import VariationalViscosity


class VariationalPerturbativeMHD(StruphyModel):
    """Perturbative (delta-f) or linear visco-resistive MHD equations, with the pressure variable, discretized with
    a variational method.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mass_number: float
        Mass number (in units of Proton mass) of the fluid species (default: 1.0)
    linear: bool
        Whether to solve the linearized equations, i.e. to drop the terms that are nonlinear in the perturbations
        (default: False)
    with_viscosity: bool
        Whether to include viscous dissipation (default: True)
    with_resistivity: bool
        Whether to include resistive dissipation (default: True)
    crank_nicolson: bool
        Only for ``linear=True``: whether to advance the velocity, the magnetic field and the pressure with one linear
        solve per substep (``variat_lin_alfven`` and ``variat_lin_sound``, energy conserving, by default with midpoint
        weights of the second-order terms, option ``weights``) instead of the Picard iteration of ``variat_pb``
        (default: False)
    u_space: str
        Space of the velocity, ``"H1vec"`` (vector field) or ``"Hdiv"`` (2-form, as in
        :class:`~struphy.models.linear_mhd.LinearMHD`; for the nonlinear model, and for the linear model with one
        linear solve per substep, ``linear=True``, ``crank_nicolson=True``) (default: ``"H1vec"``)
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
        def __init__(
            self,
            div_u: FEECVariable = None,
            u2: FEECVariable = None,
            pt3: FEECVariable = None,
            bt2: FEECVariable = None,
            rho: FEECVariable = None,
            linear: bool = False,
            with_viscosity: bool = True,
            with_resistivity: bool = True,
            crank_nicolson: bool = False,
        ):
            self.variat_dens = VariationalDensityEvolve()
            if crank_nicolson:
                self.variat_lin_alfven = VariationalLinearAlfvenEvolve(bt2=bt2)
                self.variat_lin_sound = VariationalLinearSoundEvolve(div_u=div_u, u2=u2, pt3=pt3)
            elif linear:
                self.variat_pb = VariationalPBEvolve(div_u=div_u, u2=u2, pt3=pt3, bt2=bt2)
            else:
                # rho: the density perturbation, for the mass matrix of the total density of a velocity in Hdiv
                self.variat_mom = VariationalMomentumAdvection(rho=rho)
                self.variat_pb = VariationalPBEvolve(pt3=pt3, bt2=bt2, rho=rho)
            # the linear model heats the auxiliary pressure p-hat; the nonlinear one heats the pressure (full_p)
            if with_viscosity:
                self.variat_viscous = VariationalViscosity(rho=rho, pt3=pt3 if linear else None)
            if with_resistivity:
                self.variat_resist = VariationalResistivity(rho=rho, pt3=pt3 if linear else None)

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mass_number: float = 1.0,
        linear: bool = False,
        with_viscosity: bool = True,
        with_resistivity: bool = True,
        crank_nicolson: bool = False,
        u_space: str = "H1vec",
    ):
        if crank_nicolson and not linear:
            raise ValueError(f"{self.__class__.__name__}: crank_nicolson=True needs linear=True.")
        assert u_space in ("H1vec", "Hdiv"), f'u_space must be "H1vec" or "Hdiv", got {u_space!r}.'
        assert u_space == "H1vec" or not linear or crank_nicolson, (
            'u_space="Hdiv" is available for the nonlinear model and for the linear model with one linear solve per '
            "substep (linear=True, crank_nicolson=True)."
        )

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD(mass_number=mass_number, u_space=u_space)
        self.diagnostics = self.Diagnostics()

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(
            div_u=self.diagnostics.div_u,
            u2=self.diagnostics.u2,
            pt3=self.diagnostics.pt3,
            bt2=self.diagnostics.bt2,
            rho=self.mhd.density,
            linear=linear,
            with_viscosity=with_viscosity,
            with_resistivity=with_resistivity,
            crank_nicolson=crank_nicolson,
        )

        # 4. assign variables to propagators
        self.propagators.variat_dens.variables.rho = self.mhd.density
        self.propagators.variat_dens.variables.u = self.mhd.velocity
        if not linear:
            self.propagators.variat_mom.variables.u = self.mhd.velocity
        if crank_nicolson:
            self.propagators.variat_lin_alfven.variables.u = self.mhd.velocity
            self.propagators.variat_lin_alfven.variables.b = self.em_fields.b_field
            self.propagators.variat_lin_sound.variables.u = self.mhd.velocity
            self.propagators.variat_lin_sound.variables.s = self.mhd.pressure
        else:
            self.propagators.variat_pb.variables.u = self.mhd.velocity
            self.propagators.variat_pb.variables.p = self.mhd.pressure
            self.propagators.variat_pb.variables.b = self.em_fields.b_field
        if with_viscosity:
            self.propagators.variat_viscous.variables.s = self.mhd.pressure
            self.propagators.variat_viscous.variables.u = self.mhd.velocity
        if with_resistivity:
            self.propagators.variat_resist.variables.s = self.mhd.pressure
            self.propagators.variat_resist.variables.b = self.em_fields.b_field

        # 5. define scalars to be tracked during simulation
        # the velocity mass matrix of the propagators (density-weighted): M2n for the linear model with a velocity in
        # Hdiv, the mass matrix of the total density (rho_0 + rho) for the nonlinear one
        if u_space == "Hdiv" and not linear:
            kinetic_energy = FunctionScalarFEEC(self._compute_en_U_hdiv)
        else:
            kinetic_energy = BilinearEnergyFEEC(
                self.mhd.velocity, bilinear_form_name="M2n" if u_space == "Hdiv" else "WMMnew"
            )
        magnetic_energy_1 = BilinearEnergyFEEC(self.em_fields.b_field)
        # second-order magnetic energy b0^T M2 b-hat
        magnetic_energy_2 = BilinearEnergyFEEC(self.diagnostics.bt2, right_variable="b2", normalization=2.0)
        thermo_energy = FunctionScalarFEEC(self._compute_en_thermo)
        thermo_energy_l1 = FunctionScalarFEEC(self._compute_en_thermo_l1)
        magnetic_energy_l1 = BilinearEnergyFEEC(self.em_fields.b_field, right_variable="b2", normalization=2.0)
        self.scalars = Scalars(
            en_U=kinetic_energy,
            en_thermo=thermo_energy,
            en_mag_1=magnetic_energy_1,
            en_mag_2=magnetic_energy_2,
            en_tot=kinetic_energy + thermo_energy + magnetic_energy_1 + magnetic_energy_2,
            en_tot_l1=thermo_energy_l1 + magnetic_energy_l1,
            en_thermo_l1=thermo_energy_l1,
            en_mag_l1=magnetic_energy_l1,
        )

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    @property
    def propagator_variants(self) -> dict:
        """The values of the option ``model`` of the variational propagators that match ``linear``
        (``variat_lin_sound`` with ``crank_nicolson``, else ``variat_pb``)."""
        if self.params["crank_nicolson"]:
            return {
                "variat_dens": "linear",
                "variat_lin_sound": "linear_p",
                "variat_viscous": "linear_p",
                "variat_resist": "linear_p",
            }
        if self.params["linear"]:
            return {
                "variat_dens": "linear",
                "variat_pb": "linear",
                "variat_viscous": "linear_p",
                "variat_resist": "linear_p",
            }
        return {
            "variat_dens": "perturbative",
            "variat_pb": "perturbative",
            "variat_viscous": "full_p",
            "variat_resist": "full_p",
        }

    def post_allocate(self):
        # the propagator options must match the flag `linear`
        for name, variant in self.propagator_variants.items():
            prop = getattr(self.propagators, name, None)
            if prop is not None and prop.options.model != variant:
                raise ValueError(
                    f"{self.__class__.__name__}(linear={self.params['linear']}) needs "
                    f"{name}.Options(model='{variant}'), got model='{prop.options.model}'."
                )

        # the dissipative heating of the linear model uses the gamma of its own propagator options
        if self.params["linear"]:
            gamma = self._thermo_propagator.options.gamma
            for name in ("variat_viscous", "variat_resist"):
                prop = getattr(self.propagators, name, None)
                if prop is not None and prop.options.gamma != gamma:
                    raise ValueError(
                        f"{name}.Options(gamma={prop.options.gamma}) must match the gamma={gamma} of the pressure."
                    )

        # L2 dofs of sqrt(g), the linear form of the integral over the domain of a 3-form:
        # p . vol_dofs = int p^3 d eta = int p dx (the pressure linear form of VariationalPBEvolve, up to -1/(gamma-1))
        projV3 = L2Projector("L2", Propagator.mass_ops)
        self._vol_dofs = projV3.get_dofs(Propagator.domain.jacobian_det)

        self._ones = Propagator.derham.V3pol.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        self._tmp_div_B = Propagator.derham.V3pol.zeros()

        # nonlinear model with a velocity in Hdiv: mass matrix of the total density for the kinetic energy
        if self.params["u_space"] == "Hdiv" and not self.params["linear"]:
            self._velocity_mass = VelocityMass(
                Propagator.derham,
                Propagator.domain,
                Propagator.mass_ops,
                Propagator.projected_equil,
                rho_field=self.mhd.density,
            )
            self._tmp_M_u = Propagator.derham.V2pol.zeros()

    @property
    def _thermo_propagator(self):
        """The propagator that advances the pressure (its options hold gamma)."""
        if self.params["crank_nicolson"]:
            return self.propagators.variat_lin_sound
        return self.propagators.variat_pb

    def _compute_en_U_hdiv(self):
        """Kinetic energy 1/2 u^T M^2_rho u with the total density rho_0 + rho (velocity in Hdiv)."""
        self._velocity_mass.update()
        u = self.mhd.velocity.spline.vector
        return 0.5 * u.inner(self._velocity_mass.M_rho.dot(u, out=self._tmp_M_u))

    def _compute_en_thermo(self):
        pt3 = self.diagnostics.pt3.spline.vector
        gamma = self._thermo_propagator.options.gamma
        return pt3.inner(self._vol_dofs) / (gamma - 1.0)

    def _compute_en_thermo_l1(self):
        p = self.mhd.pressure.spline.vector
        gamma = self._thermo_propagator.options.gamma
        return p.inner(self._vol_dofs) / (gamma - 1.0)

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        variants = self.propagator_variants
        linear = self.params["linear"]
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                name = next((n for n in variants if f"{n}.Options" in line), None)
                if name is not None:
                    new_file += [
                        f"model.propagators.{name}.options = model.propagators.{name}.Options(model='{variants[name]}')\n",
                    ]
                elif "pressure.add_background" in line and linear:
                    new_file += ["model.mhd.density.add_background(FieldsBackground())\n"]
                    new_file += [line]
                elif (
                    line.startswith("derham_opts = DerhamOptions()") and self.params["u_space"] == "Hdiv" and not linear
                ):
                    # the bracket of the momentum advection with a velocity in Hdiv needs the degree p >= 2 (except in
                    # periodic directions with one element)
                    new_file += [
                        "# velocity in Hdiv: the momentum advection needs degree >= 2 (except in periodic directions "
                        "with one element)\n",
                        "derham_opts = DerhamOptions(degree=(2, 2, 1))\n",
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        The unknowns are the perturbations :math:`(\tilde\rho, \tilde p, \tilde{\mathbf B})` with respect to a
        zero-flow MHD equilibrium :math:`(\rho_0, p_0, \mathbf B_0)`, and the velocity :math:`\mathbf u`.

        Continuity:

        .. math::

            \partial_t \tilde{\rho} + \nabla \cdot \left( (\tilde{\rho} + \rho_0) \tilde{\mathbf{u}} \right) = 0

        Momentum:

        .. math::

            \partial_t \left( (\tilde{\rho} + \rho_0) \tilde{\mathbf{u}} \right) + \nabla \cdot \left( (\tilde{\rho} + \rho_0) \tilde{\mathbf{u}} \otimes \tilde{\mathbf{u}} \right) + \nabla \tilde{p} + \mathbf{B}_0 \times \nabla \times \tilde{\mathbf{B}} + \tilde{\mathbf{B}} \times \nabla \times \mathbf{B}_0 + \tilde{\mathbf{B}} \times \nabla \times \tilde{\mathbf{B}} - \nabla \cdot \left( (\mu + \mu_a(\mathbf{x})) \nabla \tilde{\mathbf{u}} \right) = 0

        Pressure:

        .. math::

            \partial_t \tilde{p} + \tilde{\mathbf{u}} \cdot \nabla (\tilde{p} + p_0) + \gamma (\tilde{p} + p_0) \nabla \cdot \tilde{\mathbf{u}} = (\gamma - 1) \left( (\mu + \mu_a(\mathbf{x})) |\nabla \tilde{\mathbf{u}}|^2 + (\eta + \eta_a(\mathbf{x})) |\nabla \times \tilde{\mathbf{B}}|^2 \right)

        Induction:

        .. math::

            \partial_t \tilde{\mathbf{B}} + \nabla \times \left( (\tilde{\mathbf{B}} + \mathbf{B}_0) \times \tilde{\mathbf{u}} \right) + \nabla \times (\eta + \eta_a(\mathbf{x})) \nabla \times \tilde{\mathbf{B}} = 0

        Here :math:`\mu_a(\mathbf{x})` and :math:`\eta_a(\mathbf{x})` are artificial viscosity and resistivity coefficients.

        **Linear model** (``linear=True``): the terms that are nonlinear in the perturbations are dropped,

        .. math::

            &\partial_t \tilde{\rho} + \nabla \cdot (\rho_0 \tilde{\mathbf{u}}) = 0\,,
            \\[2mm]
            &\partial_t (\rho_0 \tilde{\mathbf{u}}) + \nabla \tilde{p} + \mathbf{B}_0 \times \nabla \times \tilde{\mathbf{B}} + \tilde{\mathbf{B}} \times \nabla \times \mathbf{B}_0 - \nabla \cdot \left( (\mu + \mu_a(\mathbf{x})) \nabla \tilde{\mathbf{u}} \right) = 0\,,
            \\[2mm]
            &\partial_t \tilde{p} + \tilde{\mathbf{u}} \cdot \nabla p_0 + \gamma p_0 \nabla \cdot \tilde{\mathbf{u}} = 0\,,
            \\[2mm]
            &\partial_t \tilde{\mathbf{B}} + \nabla \times (\mathbf{B}_0 \times \tilde{\mathbf{u}}) + \nabla \times (\eta + \eta_a(\mathbf{x})) \nabla \times \tilde{\mathbf{B}} = 0\,.

        In the linear model the viscous and resistive heating, which is quadratic in the perturbations, is not fed
        back into the pressure perturbation :math:`\tilde p`: it goes into the second-order auxiliary pressure
        :math:`\hat p` (diagnostic ``pt3``), whose energy :math:`\int \hat p/(\gamma - 1)` is part of the conserved
        total energy ``en_tot``. So with dissipation the total energy is still conserved up to the solver
        tolerances.

        **Second-order magnetic energy**: the term :math:`\int \mathbf B_0 \cdot \hat{\mathbf B} = \mathbf b_0^\top
        \mathbb M^2 \hat{\mathbf b}` of the second-order field :math:`\hat{\mathbf B}` (``diagnostics.bt2``), advected
        with the perturbation as weight, :math:`\partial_t \hat{\mathbf B} + \nabla \times (\tilde{\mathbf B} \times
        \mathbf u) = 0`, with the same commuting projection :math:`\hat\Pi^1` as the field itself. The force of the
        equilibrium current is the variation of this term (the adjoint of the transport, Galerkin Hodge star
        :math:`\mathbb M^2`), without integration by parts or auxiliary variables (see
        :class:`~struphy.propagators.variational_linear_alfven_evolve.VariationalLinearAlfvenEvolve`).

        **Weights of the nonlinear model**: in ``variat_pb`` the terms with the perturbations (the pressure force, the
        force of the equilibrium current and the nonlinear magnetic terms) use the midpoint values
        :math:`\tilde p^{n+1/2}`, :math:`\tilde{\mathbf B}^{n+1/2}` by default (``Options(weights="midpoint")``), so
        that at small amplitude the step reduces to the linear model with midpoint weights, with implicit sound waves.
        With ``weights="frozen"`` (values at :math:`t^n`) these forces are explicit and the sound waves are unstable for
        :math:`\Delta t\, \omega_s \gtrsim 1`.

        **Velocity in** :math:`H(\textnormal{div})` (``linear=True, crank_nicolson=True, u_space="Hdiv"``): the linear model can also be
        discretized with the velocity as a 2-form, as in :class:`~struphy.models.linear_mhd.LinearMHD`; this is possible
        because the linear model has no advection of the momentum (no commutator), the only term that needs
        :math:`\mathbf u \in (H^1)^3`. The variational structure is the same, so the energy is conserved up to the
        solver tolerances; the boundary condition is :math:`\mathbf u \cdot \mathbf n = 0` instead of
        :math:`\mathbf u = 0`. It is available with one linear solve per substep only (``crank_nicolson=True``). The
        viscous term is then discretized with the divergence and the curl of :math:`\mathbf u`,
        :math:`\int \mu (\nabla \cdot \mathbf u\, \nabla \cdot \mathbf v + \nabla \times \mathbf u \cdot \nabla \times \mathbf v)`,
        which equals :math:`\int \mu \nabla \mathbf u : \nabla \mathbf v` up to boundary terms.
        The curl is the weak curl, as the current of the resistivity, so the wall condition is perfect slip,
        :math:`(\nabla \times \mathbf u) \times \mathbf n = 0`, and the tangential velocity at the wall stays free.

        **Velocity in** :math:`H(\textnormal{div})` **for the nonlinear model** (``u_space="Hdiv"``): the advection of
        the momentum (the commutator of the Euler-Poincaré equations) is discretized with the vector fields acting as
        derivations on an auxiliary space that contains the contravariant velocity components, so that the
        bracket needs no projection (:class:`~struphy.feec.variational_utilities.HdivBracketOperator`). It needs the degree :math:`p \geq 2`,
        except in periodic directions with one element, and no polar splines. The velocity mass matrix is the 2-form
        mass matrix of the total density :math:`\rho_0 + \tilde\rho`, and the energy is conserved up to the solver
        tolerances, as with the velocity in :math:`(H^1)^3`.
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
        - Thermal energies: ``en_thermo``, ``en_thermo_l1``
        - Magnetic energies: ``en_mag_1``, ``en_mag_2``, ``en_mag_l1``
        - Total energies: ``en_tot``, ``en_tot_l1``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.variational_density_evolve.VariationalDensityEvolve`
        2. :class:`~struphy.propagators.variational_momentum_advection.VariationalMomentumAdvection` (if :attr:`linear` is False)
        3. :class:`~struphy.propagators.variational_pb_evolve.VariationalPBEvolve`, or, if :attr:`crank_nicolson` is
           True (linear model only),
           :class:`~struphy.propagators.variational_linear_alfven_evolve.VariationalLinearAlfvenEvolve` and
           :class:`~struphy.propagators.variational_linear_sound_evolve.VariationalLinearSoundEvolve` (one linear
           solve each, energy conserving, with midpoint weights of the second-order terms by default)
        4. :class:`~struphy.propagators.variational_viscosity.VariationalViscosity` (if :attr:`with_viscosity` is True)
        5. :class:`~struphy.propagators.variational_resistivity.VariationalResistivity` (if :attr:`with_resistivity` is True)

        The option ``model`` of the variational propagators is ``"perturbative"`` (``"full_p"`` for the dissipative
        ones), or ``"linear"`` (``"linear_p"``) if :attr:`linear` is True. With ``u_space="Hdiv"`` the propagators of the
        linear model use the velocity mass matrix :attr:`~struphy.feec.mass.WeightedMassOperators.M2n` and, for the
        equilibrium terms, the basis projection operators of
        :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator` and
        :class:`~struphy.propagators.magnetosonic.Magnetosonic`; those of the nonlinear model use the 2-form mass
        matrix of the total density (:class:`~struphy.propagators.utils.VelocityMass`) and the transport operators of
        :mod:`~struphy.feec.variational_utilities` with ``u_space="Hdiv"``.
        """
        doc = rf"""**1. VariationalDensityEvolve:**

{VariationalDensityEvolve.__doc__}

**2. VariationalMomentumAdvection:**

{VariationalMomentumAdvection.__doc__}

**3. VariationalPBEvolve:**

{VariationalPBEvolve.__doc__}

**3. (crank_nicolson=True) VariationalLinearAlfvenEvolve:**

{VariationalLinearAlfvenEvolve.__doc__}

**3. (crank_nicolson=True) VariationalLinearSoundEvolve:**

{VariationalLinearSoundEvolve.__doc__}

**4. VariationalViscosity:**

{VariationalViscosity.__doc__}

**5. VariationalResistivity:**

{VariationalResistivity.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""VariationalPerturbativeMHD is the dissipative perturbative MHD model using pressure as thermodynamic
        variable. It is intended for departures from a background equilibrium that are not strictly linear but still
        naturally formulated as perturbations; with ``linear=True`` it solves linear MHD around the equilibrium, with
        ``crank_nicolson=True`` in addition without Picard iteration (one linear solve per substep, still energy
        conserving)."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the visco-resistive perturbative MHD model and its linear version:

        .. code-block:: python

            from struphy.models import VariationalPerturbativeMHD

            model = VariationalPerturbativeMHD()
            linear_model = VariationalPerturbativeMHD(linear=True)
            linear_cn_model = VariationalPerturbativeMHD(linear=True, crank_nicolson=True)
            linear_model_hdiv = VariationalPerturbativeMHD(linear=True, crank_nicolson=True, u_space="Hdiv")
            model_hdiv = VariationalPerturbativeMHD(u_space="Hdiv")
            model.mhd.pressure
            model.em_fields.b_field
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - perturbative nonlinear MHD around a prescribed equilibrium
        - linear MHD waves and instabilities (``linear=True``)
        - comparing linear and perturbative variational formulations
        - comparing the variational linear MHD with the velocity in :math:`H(\textnormal{div})`
          (``u_space="Hdiv"``) with :class:`~struphy.models.linear_mhd.LinearMHD`
        - perturbative nonlinear MHD in toroidal geometries with the velocity in :math:`H(\textnormal{div})`
          (``u_space="Hdiv"``), whose wall condition is only :math:`\mathbf u \cdot \mathbf n = 0`"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - full-f MHD evolution without a reference equilibrium
        - kinetic or particle-resolved plasma effects
        - entropy- or q-based thermodynamic formulations"""
