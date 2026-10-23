import copy

import cunumpy as xp

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
from struphy.propagators.variational_density_evolve import VariationalDensityEvolve
from struphy.propagators.variational_momentum_advection import VariationalMomentumAdvection
from struphy.propagators.variational_qb_evolve import VariationalQBEvolve
from struphy.propagators.variational_resistivity import VariationalResistivity
from struphy.propagators.variational_viscosity import VariationalViscosity


class VariationalPerturbativeMHD_with_q(StruphyModel):
    """Perturbative (delta-f) or linear visco-resistive MHD equations, with the q variable (square root of the
    pressure), discretized with a variational method.

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
        def __init__(self, mass_number: float = 1.0):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="H1vec")
            self.sqrt_p = FEECVariable(space="L2")
            self.init_variables(mass_number=mass_number)

    class Diagnostics(DiagnosticSpecies):
        def __init__(self):
            self.div_u = FEECVariable(space="L2")
            self.u2 = FEECVariable(space="Hdiv")
            self.qt3 = FEECVariable(space="L2")
            self.bt2 = FEECVariable(space="Hdiv")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(
            self,
            div_u: FEECVariable = None,
            u2: FEECVariable = None,
            qt3: FEECVariable = None,
            bt2: FEECVariable = None,
            rho: FEECVariable = None,
            linear: bool = False,
            with_viscosity: bool = True,
            with_resistivity: bool = True,
        ):
            self.variat_dens = VariationalDensityEvolve()
            if not linear:
                self.variat_mom = VariationalMomentumAdvection()
            self.variat_qb = VariationalQBEvolve(div_u=div_u, u2=u2, qt3=qt3, bt2=bt2)
            if with_viscosity:
                self.variat_viscous = VariationalViscosity(rho=rho, pt3=qt3)
            if with_resistivity:
                self.variat_resist = VariationalResistivity(rho=rho, pt3=qt3)

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mass_number: float = 1.0,
        linear: bool = False,
        with_viscosity: bool = True,
        with_resistivity: bool = True,
    ):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD(mass_number=mass_number)
        self.diagnostics = self.Diagnostics()

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(
            div_u=self.diagnostics.div_u,
            u2=self.diagnostics.u2,
            qt3=self.diagnostics.qt3,
            bt2=self.diagnostics.bt2,
            rho=self.mhd.density,
            linear=linear,
            with_viscosity=with_viscosity,
            with_resistivity=with_resistivity,
        )

        # 4. assign variables to propagators
        self.propagators.variat_dens.variables.rho = self.mhd.density
        self.propagators.variat_dens.variables.u = self.mhd.velocity
        if not linear:
            self.propagators.variat_mom.variables.u = self.mhd.velocity
        self.propagators.variat_qb.variables.u = self.mhd.velocity
        self.propagators.variat_qb.variables.q = self.mhd.sqrt_p
        self.propagators.variat_qb.variables.b = self.em_fields.b_field
        if with_viscosity:
            self.propagators.variat_viscous.variables.s = self.mhd.sqrt_p
            self.propagators.variat_viscous.variables.u = self.mhd.velocity
        if with_resistivity:
            self.propagators.variat_resist.variables.s = self.mhd.sqrt_p
            self.propagators.variat_resist.variables.b = self.em_fields.b_field

        # 5. define scalars to be tracked during simulation
        kinetic_energy = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="WMMnew")
        magnetic_energy_1 = BilinearEnergyFEEC(self.em_fields.b_field)
        magnetic_energy_2 = BilinearEnergyFEEC(self.diagnostics.bt2, right_variable="b2", normalization=2.0)
        thermo_energy_1 = FunctionScalarFEEC(self._compute_en_thermo_1)
        thermo_energy_2 = FunctionScalarFEEC(self._compute_en_thermo_2)
        self.scalars = Scalars(
            en_U=kinetic_energy,
            en_mag_1=magnetic_energy_1,
            en_mag_2=magnetic_energy_2,
            en_thermo_1=thermo_energy_1,
            en_thermo_2=thermo_energy_2,
            en_tot=kinetic_energy + magnetic_energy_1 + magnetic_energy_2 + thermo_energy_1 + thermo_energy_2,
        )

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    @property
    def propagator_variants(self) -> dict:
        """The values of the option ``model`` of the variational propagators that match ``linear``."""
        variant = "linear_q" if self.params["linear"] else "perturbative_q"
        return {name: variant for name in ("variat_dens", "variat_qb", "variat_viscous", "variat_resist")}

    def post_allocate(self):
        # the propagator options must match the flag `linear`
        for name, variant in self.propagator_variants.items():
            prop = getattr(self.propagators, name, None)
            if prop is not None and prop.options.model != variant:
                raise ValueError(
                    f"{self.__class__.__name__}(linear={self.params['linear']}) needs "
                    f"{name}.Options(model='{variant}'), got model='{prop.options.model}'."
                )

        projV3 = L2Projector("L2", Propagator.mass_ops)

        def f(e1, e2, e3):
            return 1

        f = xp.vectorize(f)
        self._integrator = projV3(f)

        self._ones = Propagator.derham.V3pol.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        self._tmp_div_B = Propagator.derham.V3pol.zeros()

    def _compute_en_thermo_1(self):
        q = self.mhd.sqrt_p.spline.vector
        gamma = self.propagators.variat_qb.options.gamma
        return 1.0 / (gamma - 1.0) * Propagator.mass_ops.M3.dot_inner(q, q)

    def _compute_en_thermo_2(self):
        qt3 = self.propagators.variat_qb.qt3.spline.vector
        gamma = self.propagators.variat_qb.options.gamma
        return 2.0 / (gamma - 1.0) * Propagator.mass_ops.M3.dot_inner(qt3, Propagator.projected_equil.q3)

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        variant = self.propagator_variants["variat_qb"]
        linear = self.params["linear"]
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if line.startswith("model = ") and linear:
                    new_file += [line.replace("(base_units=base_units)", "(base_units=base_units, linear=True)")]
                elif "variat_dens.Options" in line:
                    new_file += [
                        f"model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model='{variant}')\n",
                    ]
                elif "variat_qb.Options" in line:
                    new_file += [
                        f"model.propagators.variat_qb.options = model.propagators.variat_qb.Options(model='{variant}')\n",
                    ]
                elif "variat_viscous.Options" in line:
                    new_file += [
                        f"model.propagators.variat_viscous.options = model.propagators.variat_viscous.Options(model='{variant}')\n",
                    ]
                elif "variat_resist.Options" in line:
                    new_file += [
                        f"model.propagators.variat_resist.options = model.propagators.variat_resist.Options(model='{variant}')\n",
                    ]
                elif "sqrt_p.add_background" in line and linear:
                    new_file += ["model.mhd.density.add_background(FieldsBackground())\n"]
                    new_file += [line]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        The unknowns are the perturbations :math:`(\tilde\rho, \tilde q, \tilde{\mathbf B})` with respect to a
        zero-flow MHD equilibrium :math:`(\rho_0, q_0, \mathbf B_0)` with :math:`q_0 = \sqrt{p_0}`, and the
        velocity :math:`\mathbf u`. The total quantities are :math:`\rho = \rho_0 + \tilde\rho`,
        :math:`q = q_0 + \tilde q = \sqrt{p}` and :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}`.

        Continuity:

        .. math::

            \partial_t \tilde{\rho} + \nabla \cdot \left( (\rho_0 + \tilde{\rho}) \mathbf{u} \right) = 0

        Momentum:

        .. math::

            \partial_t (\rho \mathbf{u}) + \nabla \cdot (\rho\, \mathbf{u} \otimes \mathbf{u})
            + \nabla \left( 2 q_0 \tilde{q} + \tilde{q}^2 \right)
            = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B}_0 + (\nabla \times \mathbf{B}_0) \times \tilde{\mathbf{B}}
            + (\nabla \times \tilde{\mathbf{B}}) \times \tilde{\mathbf{B}}
            + \nabla \cdot \left( (\mu + \mu_a(\mathbf{x})) \nabla \mathbf{u} \right)

        Square root of the pressure:

        .. math::

            \partial_t \tilde{q} + \mathbf{u} \cdot \nabla (q_0 + \tilde{q}) + \frac{\gamma}{2} (q_0 + \tilde{q}) \nabla \cdot \mathbf{u} = 0

        Induction:

        .. math::

            \partial_t \tilde{\mathbf{B}} - \nabla \times \left( \mathbf{u} \times (\mathbf{B}_0 + \tilde{\mathbf{B}}) \right)
            + \nabla \times \left( (\eta + \eta_a(\mathbf{x})) \nabla \times \tilde{\mathbf{B}} \right) = 0

        Here :math:`\mu_a(\mathbf{x})` and :math:`\eta_a(\mathbf{x})` are artificial viscosity and resistivity
        coefficients. Since :math:`\nabla(2 q_0 \tilde q + \tilde q^2) = \nabla \tilde p` with
        :math:`\tilde p = p - p_0`, the ideal part (:math:`\mu = \eta = 0`) is the perturbative ideal MHD system
        written in :math:`q` instead of :math:`p`.

        **Linear model** (``linear=True``): the terms that are nonlinear in the perturbations are dropped, i.e.
        :math:`\rho \to \rho_0` in the continuity and momentum equations (no :math:`\mathbf u \cdot \nabla \mathbf u`),
        the pressure force is :math:`\nabla (2 q_0 \tilde q) = 2 q_0 \nabla \tilde q + 2 \tilde q \nabla q_0`, the
        Lorentz force has no :math:`(\nabla \times \tilde{\mathbf B}) \times \tilde{\mathbf B}`, and :math:`\tilde q`,
        :math:`\tilde{\mathbf B}` are advected with :math:`q_0` and :math:`\mathbf B_0` only:

        .. math::

            \partial_t \tilde{q} + \mathbf{u} \cdot \nabla q_0 + \frac{\gamma}{2} q_0 \nabla \cdot \mathbf{u} = 0\,,
            \qquad
            \partial_t \tilde{\mathbf{B}} - \nabla \times (\mathbf{u} \times \mathbf{B}_0)
            + \nabla \times \left( (\eta + \eta_a(\mathbf{x})) \nabla \times \tilde{\mathbf{B}} \right) = 0\,.

        The equations derive from the Lagrangian (with :math:`\rho_0` instead of :math:`\rho` in the linear model)

        .. math::

            L = \int_\Omega \left( \rho \frac{|\mathbf{u}|^2}{2} - \frac{\tilde{q}^2 + 2 q_0 \hat{q}}{\gamma - 1}
            - \frac{|\tilde{\mathbf{B}}|^2}{2} - \mathbf{B}_0 \cdot \hat{\mathbf{B}} \right) \textrm d \mathbf{x}\,,

        with the auxiliary variables :math:`\hat q` (``diagnostics.qt3``) and :math:`\hat{\mathbf B}`
        (``diagnostics.bt2``) advected by the perturbations only (in both models),

        .. math::

            \partial_t \hat{q} + \mathbf{u} \cdot \nabla \tilde{q} + \frac{\gamma}{2} \tilde{q} \nabla \cdot \mathbf{u}
            = \frac{\gamma - 1}{2 q_0} \left( (\mu + \mu_a(\mathbf{x})) |\nabla \mathbf{u}|^2
            + (\eta + \eta_a(\mathbf{x})) |\nabla \times \tilde{\mathbf{B}}|^2 \right)\,,
            \qquad
            \partial_t \hat{\mathbf{B}} - \nabla \times (\mathbf{u} \times \tilde{\mathbf{B}}) = 0\,.

        They do not feed back on the other unknowns, but they make the energy

        .. math::

            H = \int_\Omega \left( \rho \frac{|\mathbf{u}|^2}{2} + \frac{\tilde{q}^2 + 2 q_0 \hat{q}}{\gamma - 1}
            + \frac{|\tilde{\mathbf{B}}|^2}{2} + \mathbf{B}_0 \cdot \hat{\mathbf{B}} \right) \textrm d \mathbf{x}

        a conserved quantity. Note that the heat produced by viscosity and resistivity goes into :math:`\hat q`,
        not into :math:`\tilde q` (in the linear model, the heating is skipped with the option ``fast=True`` of the
        dissipative propagators).
        """

    @classmethod
    def doc_normalization(cls):
        r"""The model uses Alfvén-speed normalization:

        .. math::

            \hat u = \hat v_A.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy: ``en_U`` :math:`= \int \rho |\mathbf u|^2/2` (:math:`\rho_0` in the linear model)
        - Magnetic energies: ``en_mag_1`` :math:`= \int |\tilde{\mathbf B}|^2/2`, ``en_mag_2`` :math:`= \int \mathbf B_0 \cdot \hat{\mathbf B}`
        - Thermodynamic energies: ``en_thermo_1`` :math:`= \int \tilde q^2/(\gamma - 1)`, ``en_thermo_2`` :math:`= \int 2 q_0 \hat q/(\gamma - 1)`
        - Total energy: ``en_tot`` (conserved)"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.variational_density_evolve.VariationalDensityEvolve`
        2. :class:`~struphy.propagators.variational_momentum_advection.VariationalMomentumAdvection` (if :attr:`linear` is False)
        3. :class:`~struphy.propagators.variational_qb_evolve.VariationalQBEvolve`
        4. :class:`~struphy.propagators.variational_viscosity.VariationalViscosity` (if :attr:`with_viscosity` is True)
        5. :class:`~struphy.propagators.variational_resistivity.VariationalResistivity` (if :attr:`with_resistivity` is True)

        The option ``model`` of the variational propagators is ``"perturbative_q"``, or ``"linear_q"`` if
        :attr:`linear` is True.
        """
        doc = rf"""**1. VariationalDensityEvolve:**

{VariationalDensityEvolve.__doc__}

**2. VariationalMomentumAdvection:**

{VariationalMomentumAdvection.__doc__}

**3. VariationalQBEvolve:**

{VariationalQBEvolve.__doc__}

**4. VariationalViscosity:**

{VariationalViscosity.__doc__}

**5. VariationalResistivity:**

{VariationalResistivity.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""MHD written for the perturbations with respect to a zero-flow equilibrium, in the variable
        :math:`q = \sqrt{p}`, with optional viscosity and resistivity. By default all nonlinear terms are kept, so
        the model is equivalent to full MHD, but the equilibrium force balance is subtracted analytically: a zero
        perturbation stays exactly zero, also when the discrete equilibrium is not in exact force balance. With
        ``linear=True`` the model solves linear MHD around the equilibrium.
        Using :math:`q` makes the Lagrangian quadratic in the perturbed variables, which allows the
        variational discretization with the auxiliary variables :math:`\hat q` and :math:`\hat{\mathbf B}`.
        Set ``with_viscosity=False`` and ``with_resistivity=False`` for ideal MHD."""

    @classmethod
    def doc_examples(cls):
        r"""Create the ideal perturbative q-MHD model and its linear version:

        .. code-block:: python

            from struphy.models import VariationalPerturbativeMHD_with_q

            model = VariationalPerturbativeMHD_with_q(with_viscosity=False, with_resistivity=False)
            linear_model = VariationalPerturbativeMHD_with_q(linear=True, with_viscosity=False, with_resistivity=False)
            model.mhd.sqrt_p
            model.em_fields.b_field
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - nonlinear MHD dynamics around a given (possibly non-uniform) zero-flow equilibrium
        - linear MHD waves and instabilities (``linear=True``)
        - instabilities and waves where the discrete equilibrium force must not drive spurious flows
        - comparison with :class:`~struphy.models.variational_mhd_with_q.VariationalMHD_with_q` (full-f), with
          :class:`~struphy.models.variational_perturbative_mhd.VariationalPerturbativeMHD` (pressure) and with
          :class:`~struphy.models.linear_mhd.LinearMHD`"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - equilibria with flow
        - dissipative runs where the viscous and resistive heating must raise the pressure (it goes into
          :math:`\hat q`, see ``doc_pde``)
        - entropy-based thermodynamic evolution
        - kinetic or hybrid particle coupling"""
