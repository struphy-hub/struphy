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


class LinearExtendedMHDuniform(StruphyModel):
    r"""Linear extended MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).
    For uniform background conditions only.

    :ref:`normalization`:

    .. math::

        \hat U = \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,,
        \\[2mm]
        \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 \,,
        \\[2mm]
        &\frac{\partial \tilde p}{\partial t} + \frac{5}{3}\,p_{0}\nabla\cdot \tilde{\mathbf{U}}=0\,,
        \\[2mm]
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times \left( \tilde{\mathbf{U}} \times \mathbf{B}_0 - \frac{1}{\varepsilon} \frac{\nabla\times \tilde{\mathbf{B}}}{\rho_0}\times \mathbf{B}_0 \right)
        = 0\,.

    where

    .. math::

        \varepsilon = \frac{1}{\hat \Omega_{\textnormal{c}} \hat t}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.ShearAlfvenB1`
    2. :class:`~struphy.propagators.propagators_fields.Hall`
    3. :class:`~struphy.propagators.propagators_fields.MagnetosonicUniform`

    :ref:`Model info <add_model>`:
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.b_field = FEECVariable(space="Hcurl")
            self.init_variables()

    class MHD(FluidSpecies):
        def __init__(self):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="Hdiv")
            self.pressure = FEECVariable(space="L2")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.shear_alf = propagators_fields.ShearAlfvenB1()
            self.hall = propagators_fields.Hall()
            self.mag_sonic = propagators_fields.MagnetosonicUniform()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.shear_alf.variables.u = self.mhd.velocity
        self.propagators.shear_alf.variables.b = self.em_fields.b_field

        self.propagators.hall.variables.b = self.em_fields.b_field

        self.propagators.mag_sonic.variables.n = self.mhd.density
        self.propagators.mag_sonic.variables.u = self.mhd.velocity
        self.propagators.mag_sonic.variables.p = self.mhd.pressure

        # define scalars for update_scalar_quantities
        self.add_scalar("en_U")
        self.add_scalar("en_p")
        self.add_scalar("en_B")
        self.add_scalar("en_p_eq")
        self.add_scalar("en_B_eq")
        self.add_scalar("en_B_tot")
        self.add_scalar("en_tot")
        self.add_scalar("helicity")

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self):
        self._b_eq = self.projected_equil.b1
        self._a_eq = self.projected_equil.a1
        self._p_eq = self.projected_equil.p3

        self._ones = self.projected_equil.p3.space.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        self._tmp_b1: BlockVector = self.derham.Vh["1"].zeros()  # TODO: replace derham.Vh dict by class
        self._tmp_b2: BlockVector = self.derham.Vh["1"].zeros()

        # adjust coupling parameters
        epsilon = self.mhd.equation_params.epsilon

        if abs(epsilon - 1) < 1e-6:
            self.mhd.equation_params.epsilon = 1.0

    def update_scalar_quantities(self):
        # perturbed fields
        u = self.mhd.velocity.spline.vector
        p = self.mhd.pressure.spline.vector
        b = self.em_fields.b_field.spline.vector

        en_U = 0.5 * self.mass_ops.M2n.dot_inner(u, u)
        b1 = self.mass_ops.M1.dot(b, out=self._tmp_b1)
        en_B = 0.5 * b.inner(b1)
        helicity = 2.0 * self._a_eq.inner(b1)
        en_p_i = p.inner(self._ones) / (5.0 / 3.0 - 1.0)

        self.update_scalar("en_U", en_U)
        self.update_scalar("en_B", en_B)
        self.update_scalar("en_p", en_p_i)
        self.update_scalar("helicity", helicity)
        self.update_scalar("en_tot", en_U + en_B + en_p_i)

        # background fields
        b1 = self.mass_ops.M1.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)
        en_B0 = self._b_eq.inner(b1) / 2.0
        en_p0 = self._p_eq.inner(self._ones) / (5.0 / 3.0 - 1.0)

        self.update_scalar("en_B_eq", en_B0)
        self.update_scalar("en_p_eq", en_p0)

        # total magnetic field
        b1 = self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += b

        b2 = self.mass_ops.M1.dot(b1, apply_bc=False, out=self._tmp_b2)
        en_Btot = b1.inner(b2) / 2.0

        self.update_scalar("en_B_tot", en_Btot)

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "hall.Options" in line:
                    new_file += [
                        "model.propagators.hall.options = model.propagators.hall.Options(epsilon_from=model.mhd)\n",
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
