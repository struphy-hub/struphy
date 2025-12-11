import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.block import BlockVector
from feectools.linalg.stencil import StencilVector

from struphy.feec.projectors import L2Projector
from struphy.feec.variational_utilities import (
    H1vecMassMatrix_density,
    InternalEnergyEvaluator,
)
from struphy.kinetic_background.base import KineticBackground
from struphy.kinetic_background.maxwellians import Maxwellian3D
from struphy.models.base import StruphyModel
from struphy.models.species import (
    DiagnosticSpecies,
    FieldSpecies,
    FluidSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.polar.basic import PolarVector
from struphy.propagators import (
    propagators_coupling,
    propagators_fields,
    propagators_markers,
)
from struphy.utils.pyccel import Pyccelkernel

rank = MPI.COMM_WORLD.Get_rank()


class ViscoResistiveMHD_with_p(StruphyModel):
    r"""Full (non-linear) visco-resistive MHD equations, with the pressure variable discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \frac{1}{\gamma -1} \nabla p + \mathbf B \times \nabla \times \mathbf B - \nabla \cdot \left((\mu+\mu_a(\mathbf x)) \nabla \mathbf u \right) = 0 \,,
        \\[4mm]
        &\partial_t p + u \cdot \nabla p + \gamma p \nabla \cdot u = \frac{1}{(\gamma -1)}\left((\mu+\mu_a(\mathbf x)) |\nabla \mathbf u|^2 + (\eta + \eta_a(\mathbf x)) |\nabla \times \mathbf B|^2\right) \,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \times ( \mathbf B \times \mathbf u ) + \nabla \times (\eta + \eta_a(\mathbf x)) \nabla \times \mathbf B = 0 \,,

    and :math:`\mu_a(\mathbf x)` and :math:`\eta_a(\mathbf x)` are artificial viscosity and resistivity coefficients.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.propagators_fields.VariationalPBEvolve`
    4. :class:`~struphy.propagators.propagators_fields.VariationalViscosity`
    5. :class:`~struphy.propagators.propagators_fields.VariationalResistivity`

    :ref:`Model info <add_model>`:
    """

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
        with_viscosity: bool = True,
        with_resistivity: bool = True,
    ):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

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

        # define scalars for update_scalar_quantities
        self.add_scalar("en_U")
        self.add_scalar("en_thermo")
        self.add_scalar("en_mag")
        self.add_scalar("en_tot")
        self.add_scalar("dens_tot")
        self.add_scalar("tot_div_B")

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self):
        projV3 = L2Projector("L2", self._mass_ops)

        def f(e1, e2, e3):
            return 1

        f = xp.vectorize(f)
        self._integrator = projV3(f)

        self._ones = self.derham.Vh_pol["3"].zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        self._tmp_div_B = self.derham.Vh_pol["3"].zeros()

    def update_scalar_quantities(self):
        rho = self.mhd.density.spline.vector
        u = self.mhd.velocity.spline.vector
        p = self.mhd.pressure.spline.vector
        b = self.em_fields.b_field.spline.vector

        gamma = self.propagators.variat_pb.options.gamma

        en_U = 0.5 * self.mass_ops.WMM.massop.dot_inner(u, u)
        self.update_scalar("en_U", en_U)

        en_mag = 0.5 * self.mass_ops.M2.dot_inner(b, b)
        self.update_scalar("en_mag", en_mag)

        en_thermo = self.mass_ops.M3.dot_inner(p, self._integrator) / (gamma - 1.0)
        self.update_scalar("en_thermo", en_thermo)

        en_tot = en_U + en_thermo + en_mag
        self.update_scalar("en_tot", en_tot)

        dens_tot = self._ones.inner(rho)
        self.update_scalar("dens_tot", dens_tot)

        div_B = self.derham.div.dot(b, out=self._tmp_div_B)
        L2_div_B = self._mass_ops.M3.dot_inner(div_B, div_B)
        self.update_scalar("tot_div_B", L2_div_B)

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "variat_pb.Options" in line:
                    new_file += [
                        "model.propagators.variat_pb.options = model.propagators.variat_pb.Options(div_u=model.diagnostics.div_u,\n",
                    ]
                    new_file += [
                        "                                                                          u2=model.diagnostics.u2)\n",
                    ]
                elif "variat_viscous.Options" in line:
                    new_file += [
                        "model.propagators.variat_viscous.options = model.propagators.variat_viscous.Options(rho=model.mhd.density)\n",
                    ]
                elif "variat_resist.Options" in line:
                    new_file += [
                        "model.propagators.variat_resist.options = model.propagators.variat_resist.Options(rho=model.mhd.density)\n",
                    ]
                elif "pressure.add_background" in line:
                    new_file += ["model.mhd.density.add_background(FieldsBackground())\n"]
                    new_file += [line]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
