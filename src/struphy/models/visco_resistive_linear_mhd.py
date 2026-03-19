import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.feec.projectors import L2Projector
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
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

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalPBEvolve`
    3. :class:`~struphy.propagators.propagators_fields.VariationalViscosity`
    4. :class:`~struphy.propagators.propagators_fields.VariationalResistivity`

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
        def __init__(self):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="H1vec")
            self.pressure = FEECVariable(space="L2")
            self.init_variables()

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
            self.variat_pb = propagators_fields.VariationalPBEvolve()
            if with_viscosity:
                self.variat_viscous = propagators_fields.VariationalViscosity()
            if with_resistivity:
                self.variat_resist = propagators_fields.VariationalResistivity()

    ## abstract methods

    def __init__(
        self,
        with_viscosity: bool = True,
        with_resistivity: bool = True,
    ):

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD()
        self.diagnostics = self.Diagnostics()

        # 2. instantiate all propagators
        self.propagators = self.Propagators(
            with_viscosity=with_viscosity,
            with_resistivity=with_resistivity,
        )

        # 3. assign variables to propagators
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

        # define scalars for update_scalar_quantities
        self.add_scalar("en_U")
        self.add_scalar("en_thermo")
        self.add_scalar("en_mag_1")
        self.add_scalar("en_mag_2")
        self.add_scalar("en_tot")

        self.add_scalar("en_tot_l1")
        self.add_scalar("en_thermo_l1")
        self.add_scalar("en_mag_l1")

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

    def update_scalar_quantities(self):
        rho = self.mhd.density.spline.vector
        u = self.mhd.velocity.spline.vector
        p = self.mhd.pressure.spline.vector
        b = self.em_fields.b_field.spline.vector
        bt2 = self.propagators.variat_pb.options.bt2.spline.vector
        pt3 = self.propagators.variat_pb.options.pt3.spline.vector

        gamma = self.propagators.variat_pb.options.gamma

        en_U = 0.5 * Propagator.mass_ops.WMM.massop.dot_inner(u, u)
        self.update_scalar("en_U", en_U)

        en_mag1 = 0.5 * Propagator.mass_ops.M2.dot_inner(b, b)
        self.update_scalar("en_mag_1", en_mag1)

        en_mag2 = Propagator.mass_ops.M2.dot_inner(bt2, Propagator.projected_equil.b2)
        self.update_scalar("en_mag_2", en_mag2)

        en_thermo = Propagator.mass_ops.M3.dot_inner(pt3, self._integrator) / (gamma - 1.0)
        self.update_scalar("en_thermo", en_thermo)

        en_tot = en_U + en_thermo + en_mag1 + en_mag2
        self.update_scalar("en_tot", en_tot)

        # dens_tot = self._ones.inner(rho)
        # self.update_scalar("dens_tot", dens_tot)

        # div_B = Propagator.derham.div.dot(b, out=self._tmp_div_B)
        # L2_div_B = Propagator.mass_ops.M3.dot_inner(div_B, div_B)
        # self.update_scalar("tot_div_B", L2_div_B)

        en_thermo_l1 = Propagator.mass_ops.M3.dot_inner(p, self._integrator) / (gamma - 1.0)
        self.update_scalar("en_thermo_l1", en_thermo_l1)

        en_mag_l1 = Propagator.mass_ops.M2.dot_inner(b, Propagator.projected_equil.b2)
        self.update_scalar("en_mag_l1", en_mag_l1)

        en_tot_l1 = en_thermo_l1 + en_mag_l1
        self.update_scalar("en_tot_l1", en_tot_l1)

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
