import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.feec.projectors import L2Projector
from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalar, Scalars
from struphy.models.species import (
    DiagnosticSpecies,
    FieldSpecies,
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.polar.basic import PolarVector
from struphy.propagators import (
    propagators_fields,
)
from struphy.propagators.base import Propagator

rank = MPI.COMM_WORLD.Get_rank()


class ViscoResistiveDeltafMHD(StruphyModel):
    r""":math:`\delta f` visco-resistive MHD equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \tilde{\rho} + \nabla \cdot ( (\tilde{\rho}+\rho_0) \tilde{\mathbf u} ) = 0 \,,
        \\[4mm]
        &\partial_t ((\tilde{\rho}+\rho_0) \tilde{\mathbf u}) + \nabla \cdot ((\tilde{\rho}+\rho_0) \tilde{\mathbf u} \otimes \tilde{\mathbf u}) + \frac{1}{\gamma -1} \nabla \tilde{p} + \mathbf B_0 \times \nabla \times \tilde{\mathbf B} + \tilde{\mathbf B} \times \nabla \times \mathbf B_0 +  \tilde{\mathbf B} \times \nabla \times \tilde{\mathbf B} - \nabla \cdot \left((\mu+\mu_a(\mathbf x)) \nabla \tilde{\mathbf u} \right) = 0 \,,
        \\[4mm]
        &\partial_t \tilde{p} + \tilde{\mathbf u} \cdot \nabla (\tilde{p} + p_0) + \gamma (\tilde{p} + p_0) \nabla \cdot \tilde{\mathbf u} = \frac{1}{(\gamma -1)}\left((\mu+\mu_a(\mathbf x)) |\nabla \tilde{\mathbf u}|^2 + (\eta + \eta_a(\mathbf x)) |\nabla \times \tilde{\mathbf B}|^2\right) \,,
        \\[4mm]
        &\partial_t \tilde{\mathbf B} + \nabla \times ( (\tilde{\mathbf B} + \mathbf B_0) \times \tilde{\mathbf u} ) + \nabla \times (\eta + \eta_a(\mathbf x)) \nabla \times \tilde{\mathbf B} = 0 \,,

    and :math:`\mu_a(\mathbf x)` and :math:`\eta_a(\mathbf x)` are artificial viscosity and resistivity coefficients.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.propagators_fields.VariationalPBEvolve`
    4. :class:`~struphy.propagators.propagators_fields.VariationalViscosity`
    5. :class:`~struphy.propagators.propagators_fields.VariationalResistivity`

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
            self.variat_dens = propagators_fields.VariationalDensityEvolve()
            self.variat_mom = propagators_fields.VariationalMomentumAdvection()
            self.variat_pb = propagators_fields.VariationalPBEvolve()
            if with_viscosity:
                self.variat_viscous = propagators_fields.VariationalViscosity()
            if with_resistivity:
                self.variat_resist = propagators_fields.VariationalResistivity()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mass_number: float = 1.0,
        with_viscosity: bool = True,
        with_resistivity: bool = True,
    ):

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
        self.propagators.variat_mom.variables.u = self.mhd.velocity
        self.propagators.variat_pb.variables.u = self.mhd.velocity
        self.propagators.variat_pb.variables.p = self.mhd.pressure
        self.propagators.variat_pb.variables.b = self.em_fields.b_field
        if with_viscosity:
            self.propagators.variat_viscous.variables.s = self.mhd.pressure
            self.propagators.variat_viscous.variables.u = self.mhd.velocity
        if with_resistivity:
            self.propagators.variat_resist.variables.s = self.mhd.pressure
            self.propagators.variat_resist.variables.b = self.em_fields.b_field

        kinetic_energy = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="WMM", normalization=0.5)
        magnetic_energy_1 = BilinearEnergyFEEC(self.em_fields.b_field, bilinear_form_name="M2", normalization=0.5)
        magnetic_energy_2 = BilinearEnergyFEEC(self.diagnostics.bt2, right_variable="b2", bilinear_form_name="M2")
        thermo_energy = FunctionScalar(self._compute_en_thermo)
        thermo_energy_l1 = FunctionScalar(self._compute_en_thermo_l1)
        magnetic_energy_l1 = BilinearEnergyFEEC(self.em_fields.b_field, right_variable="b2", bilinear_form_name="M2")
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

    def allocate_helpers(self, verbose: bool = False):
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
                        "model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model='deltaf')\n",
                    ]
                elif "variat_pb.Options" in line:
                    new_file += [
                        "model.propagators.variat_pb.options = model.propagators.variat_pb.Options(model='deltaf',\n",
                    ]
                    new_file += [
                        "                                                                          pt3=model.diagnostics.pt3,\n",
                    ]
                    new_file += [
                        "                                                                          bt2=model.diagnostics.bt2)\n",
                    ]
                elif "variat_viscous.Options" in line:
                    new_file += [
                        "model.propagators.variat_viscous.options = model.propagators.variat_viscous.Options(model='full_p',\n",
                    ]
                    new_file += [
                        "                                                                                    rho=model.mhd.density)\n",
                    ]
                elif "variat_resist.Options" in line:
                    new_file += [
                        "model.propagators.variat_resist.options = model.propagators.variat_resist.Options(model='full_p',\n",
                    ]
                    new_file += [
                        "                                                                                  rho=model.mhd.density)\n",
                    ]
                elif "pressure.add_background" in line:
                    new_file += ["model.mhd.density.add_background(FieldsBackground())\n"]
                    new_file += [line]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
