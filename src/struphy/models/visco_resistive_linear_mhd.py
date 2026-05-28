import copy

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

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
from struphy.propagators.variational_pb_evolve import VariationalPBEvolve
from struphy.propagators.variational_resistivity import VariationalResistivity
from struphy.propagators.variational_viscosity import VariationalViscosity

rank = MPI.COMM_WORLD.Get_rank()


class ViscoResistiveLinearMHD(StruphyModel):
    r"""Linear visco-resistive MHD equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \tilde{\rho} + \nabla \cdot ( \rho_0 \tilde{\mathbf u} ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho_0 \tilde{\mathbf u}) + \frac{1}{\gamma -1} \nabla \tilde{p} + \mathbf B_0 \times \nabla \times \tilde{\mathbf B} + \tilde{\mathbf B} \times \nabla \times \mathbf B_0 - \nabla \cdot \left((\mu+\mu_a(\mathbf x)) \nabla \tilde{\mathbf u} \right) = 0 \,,
        \\[4mm]
        &\partial_t \tilde{p} + \tilde{\mathbf u} \cdot \nabla p_0 + \gamma p_0 \nabla \cdot \tilde{\mathbf u} = \frac{1}{(\gamma -1)}\left((\mu+\mu_a(\mathbf x)) |\nabla \tilde{\mathbf u}|^2 + (\eta + \eta_a(\mathbf x)) |\nabla \times \tilde{\mathbf B}|^2\right) \,,
        \\[4mm]
        &\partial_t \tilde{\mathbf B} + \nabla \times ( \mathbf B_0 \times \tilde{\mathbf u} ) + \nabla \times (\eta + \eta_a(\mathbf x)) \nabla \times \tilde{\mathbf B} = 0 \,,

    and :math:`\mu_a(\mathbf x)` and :math:`\eta_a(\mathbf x)` are artificial viscosity and resistivity coefficients.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.variational_density_evolve.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.variational_pb_evolve.VariationalPBEvolve`
    3. :class:`~struphy.propagators.variational_viscosity.VariationalViscosity`
    4. :class:`~struphy.propagators.variational_resistivity.VariationalResistivity`

    :ref:`Model info <add_model>`:
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
            with_viscosity: bool = True,
            with_resistivity: bool = True,
        ):
            self.variat_dens = VariationalDensityEvolve()
            self.variat_pb = VariationalPBEvolve()
            if with_viscosity:
                self.variat_viscous = VariationalViscosity()
            if with_resistivity:
                self.variat_resist = VariationalResistivity()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mass_number: float = 1.0,
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
            with_viscosity=with_viscosity,
            with_resistivity=with_resistivity,
        )

        # 4. assign variables to propagators
        self.propagators.variat_dens.variables.rho = self.mhd.density
        self.propagators.variat_dens.variables.u = self.mhd.velocity
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
        kinetic_energy = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="WMMnew")
        magnetic_energy_1 = BilinearEnergyFEEC(self.em_fields.b_field)
        magnetic_energy_2 = BilinearEnergyFEEC(self.diagnostics.bt2, right_variable="b2")
        thermo_energy = FunctionScalarFEEC(self._compute_en_thermo)
        thermo_energy_l1 = FunctionScalarFEEC(self._compute_en_thermo_l1)
        magnetic_energy_l1 = BilinearEnergyFEEC(self.em_fields.b_field, right_variable="b2")
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

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity:

        .. math::

            \partial_t \tilde{\rho} + \nabla \cdot (\rho_0 \tilde{\mathbf{u}}) = 0

        Momentum:

        .. math::

            \partial_t (\rho_0 \tilde{\mathbf{u}}) + \frac{1}{\gamma - 1} \nabla \tilde{p} + \mathbf{B}_0 \times \nabla \times \tilde{\mathbf{B}} + \tilde{\mathbf{B}} \times \nabla \times \mathbf{B}_0 - \nabla \cdot \left( (\mu + \mu_a(\mathbf{x})) \nabla \tilde{\mathbf{u}} \right) = 0

        Pressure:

        .. math::

            \partial_t \tilde{p} + \tilde{\mathbf{u}} \cdot \nabla p_0 + \gamma p_0 \nabla \cdot \tilde{\mathbf{u}} = \frac{1}{\gamma - 1} \left( (\mu + \mu_a(\mathbf{x})) |\nabla \tilde{\mathbf{u}}|^2 + (\eta + \eta_a(\mathbf{x})) |\nabla \times \tilde{\mathbf{B}}|^2 \right)

        Induction:

        .. math::

            \partial_t \tilde{\mathbf{B}} + \nabla \times (\mathbf{B}_0 \times \tilde{\mathbf{u}}) + \nabla \times (\eta + \eta_a(\mathbf{x})) \nabla \times \tilde{\mathbf{B}} = 0

        Here :math:`\mu_a(\mathbf{x})` and :math:`\eta_a(\mathbf{x})` are artificial viscosity and resistivity coefficients.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The characteristic velocity is the Alfvén speed,

        .. math::

            \hat u = \hat v_A.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy: ``en_U``
        - Thermal perturbation energy: ``en_thermo``
        - Magnetic energies: ``en_mag_1``, ``en_mag_2``
        - Total energy diagnostics: ``en_tot``, ``en_tot_l1``
        - Auxiliary linearized diagnostics: ``en_thermo_l1``, ``en_mag_l1``"""

    @classmethod
    def doc_discretization(cls):
        doc = rf"""**1. VariationalDensityEvolve:**

{VariationalDensityEvolve.__doc__}

**2. VariationalPBEvolve:**

{VariationalPBEvolve.__doc__}

**3. VariationalViscosity:**

{VariationalViscosity.__doc__}

**4. VariationalResistivity:**

{VariationalResistivity.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""This is the linear dissipative MHD model with pressure as primitive
        thermodynamic variable. It is intended for small-amplitude perturbations
        around an equilibrium while retaining explicit viscosity and
        resistivity."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a linear visco-resistive MHD model:

        .. code-block:: python

            from struphy.models import ViscoResistiveLinearMHD

            model = ViscoResistiveLinearMHD()
            model.em_fields.b_field
            model.mhd.velocity
            model.mhd.pressure
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - linear dissipative MHD wave studies
        - equilibrium perturbation benchmarks with viscosity and resistivity
        - verification of linear variational MHD operators"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - strongly nonlinear MHD dynamics
        - fully kinetic plasma effects
        - entropy- or q-based thermodynamic formulations
        - ideal MHD benchmarks where dissipation must be absent by construction"""

    def allocate_helpers(self):
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

    def _compute_en_thermo(self):
        pt3 = self.propagators.variat_pb.options.pt3.spline.vector
        gamma = self.propagators.variat_pb.options.gamma
        return Propagator.mass_ops.M3.dot_inner(pt3, self._integrator) / (gamma - 1.0)

    def _compute_en_thermo_l1(self):
        p = self.mhd.pressure.spline.vector
        gamma = self.propagators.variat_pb.options.gamma
        return Propagator.mass_ops.M3.dot_inner(p, self._integrator) / (gamma - 1.0)

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "variat_dens.Options" in line:
                    new_file += [
                        "model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model='linear')\n",
                    ]
                elif "variat_pb.Options" in line:
                    new_file += [
                        "model.propagators.variat_pb.options = model.propagators.variat_pb.Options(model='linear',\n",
                    ]
                    new_file += [
                        "                                                                          div_u=model.diagnostics.div_u,\n",
                    ]
                    new_file += [
                        "                                                                          u2=model.diagnostics.u2,\n",
                    ]
                    new_file += [
                        "                                                                          pt3=model.diagnostics.pt3,\n",
                    ]
                    new_file += [
                        "                                                                          bt2=model.diagnostics.bt2)\n",
                    ]
                elif "variat_viscous.Options" in line:
                    new_file += [
                        "model.propagators.variat_viscous.options = model.propagators.variat_viscous.Options(model='linear_p',\n",
                    ]
                    new_file += [
                        "                                                                                    rho=model.mhd.density)\n",
                    ]
                elif "variat_resist.Options" in line:
                    new_file += [
                        "model.propagators.variat_resist.options = model.propagators.variat_resist.Options(model='linear_p',\n",
                    ]
                    new_file += [
                        "                                                                                  rho=model.mhd.density,\n",
                    ]
                    new_file += [
                        "                                                                                  pt3=model.diagnostics.pt3)\n",
                    ]
                elif "pressure.add_background" in line:
                    new_file += ["model.mhd.density.add_background(FieldsBackground())\n"]
                    new_file += [line]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
