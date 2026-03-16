import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
    FluidSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.polar.basic import PolarVector
from struphy.propagators import (
    propagators_coupling,
    propagators_fields,
    propagators_markers,
)
from struphy.propagators.base import Propagator

rank = MPI.COMM_WORLD.Get_rank()


class LinearMHDDriftkineticCC(StruphyModel):
    r"""Hybrid linear ideal MHD + energetic ions (5D Driftkinetic) with **current coupling scheme**. 

    :ref:`normalization`: 

    .. math::

        \hat U = \hat v =: \hat v_\textnormal{A, bulk} \,, \qquad
        \hat f_\textnormal{h} = \frac{\hat n}{\hat v_\textnormal{h} \hat \mu \hat B} \,,\qquad 
        \hat \mu = \frac{A_\textnormal{h} m_\textnormal{H} \hat v_\textnormal{h}^2}{\hat B} \,.

    :ref:`Equations <gempic>`:

    .. math::

        \begin{align}
        \textnormal{MHD} &\left\{
        \begin{aligned}
        &\frac{\partial \tilde{\rho}}{\partial t}+\nabla\cdot(\rho_{0} \tilde{\mathbf{U}})=0\,, 
        \\
        \rho_{0} &\frac{\partial \tilde{\mathbf{U}}}{\partial t} - \tilde p\, \nabla
        = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B} + (\nabla \times \mathbf B_0) \times \tilde{\mathbf{B}}
        + \frac{A_\textnormal{h}}{A_\textnormal{b}} \left[ \frac{1}{\epsilon} n_\textnormal{gc} \tilde{\mathbf{U}} - \frac{1}{\epsilon} \mathbf{J}_\textnormal{gc} - \nabla \times \mathbf{M}_\textnormal{gc} \right] \times \mathbf{B} \,,
        \\
        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,, 
        \\
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B})
        = 0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{EPs}\,\, &\left\{\,\,
        \begin{aligned}
        \quad &\frac{\partial f_\textnormal{h}}{\partial t} + \frac{1}{B_\parallel^*}(v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^*)\cdot\nabla f_\textnormal{h}
        + \frac{1}{\epsilon} \frac{1}{B_\parallel^*} (\mathbf{B}^* \cdot \mathbf{E}^*) \frac{\partial f_\textnormal{h}}{\partial v_\parallel}
        = 0\,,
        \\
        & n_\textnormal{gc} = \int f_\textnormal{h} B_\parallel^* \,\textnormal dv_\parallel \textnormal d\mu \,,
        \\
        & \mathbf{J}_\textnormal{gc} = \int \frac{f_\textnormal{h}}{B_\parallel^*}(v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^*) \,\textnormal dv_\parallel \textnormal d\mu \,,
        \\
        & \mathbf{M}_\textnormal{gc} = - \int f_\textnormal{h} B_\parallel^* \mu \mathbf{b}_0 \,\textnormal dv_\parallel \textnormal d\mu \,,
        \end{aligned}
        \right.
        \end{align}

    where 

    .. math::

        \begin{align}
        B^*_\parallel = \mathbf{b}_0 \cdot \mathbf{B}^*\,,
        \\[2mm]
        \mathbf{B}^* &= \mathbf{B} + \epsilon v_\parallel \nabla \times \mathbf{b}_0 \,,
        \\[2mm]
        \mathbf{E}^* &= - \tilde{\mathbf{U}} \times \mathbf{B} - \epsilon \mu \nabla (\mathbf{b}_0 \cdot \mathbf{B}) \,,
        \end{align}

    with the normalization parameter 

    .. math::

        \epsilon = \frac{1}{\hat \Omega_\textnormal{c,hot} \hat t} \,, \qquad \hat \Omega_\textnormal{c,hot} = \frac{Z_\textnormal{h} e \hat B}{A_\textnormal{h} m_\textnormal{H}} \,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushGuidingCenterBxEstar`
    2. :class:`~struphy.propagators.propagators_markers.PushGuidingCenterParallel`
    3. :class:`~struphy.propagators.propagators_coupling.CurrentCoupling5DGradB`
    4. :class:`~struphy.propagators.propagators_coupling.CurrentCoupling5DCurlb`
    5. :class:`~struphy.propagators.propagators_fields.CurrentCoupling5DDensity`
    6. :class:`~struphy.propagators.propagators_fields.ShearAlfvenCurrentCoupling5D`
    7. :class:`~struphy.propagators.propagators_fields.Magnetosonic`

    :ref:`Model info <add_model>`:
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Hybrid"

    ## species
    class EnergeticIons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles5D")
            self.init_variables()

    class EMFields(FieldSpecies):
        def __init__(self):
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    class MHD(FluidSpecies):
        def __init__(self):
            self.density = FEECVariable(space="L2")
            self.pressure = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="Hdiv")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self, turn_off: tuple[str, ...] = (None,)):
            if "PushGuidingCenterBxEstar" not in turn_off:
                self.push_bxe = propagators_markers.PushGuidingCenterBxEstar()
            if "PushGuidingCenterParallel" not in turn_off:
                self.push_parallel = propagators_markers.PushGuidingCenterParallel()
            if "ShearAlfvenCurrentCoupling5D" not in turn_off:
                self.shearalfen_cc5d = propagators_fields.ShearAlfvenCurrentCoupling5D()
            if "Magnetosonic" not in turn_off:
                self.magnetosonic = propagators_fields.Magnetosonic()
            if "CurrentCoupling5DDensity" not in turn_off:
                self.cc5d_density = propagators_fields.CurrentCoupling5DDensity()
            if "CurrentCoupling5DGradB" not in turn_off:
                self.cc5d_gradb = propagators_coupling.CurrentCoupling5DGradB()
            if "CurrentCoupling5DCurlb" not in turn_off:
                self.cc5d_curlb = propagators_coupling.CurrentCoupling5DCurlb()

    def __init__(self, turn_off: tuple[str, ...] = (None,)):
        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD()
        self.energetic_ions = self.EnergeticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators(turn_off)

        # 3. assign variables to propagators
        if "ShearAlfvenCurrentCoupling5D" not in turn_off:
            self.propagators.shearalfen_cc5d.variables.u = self.mhd.velocity
            self.propagators.shearalfen_cc5d.variables.b = self.em_fields.b_field
        if "Magnetosonic" not in turn_off:
            self.propagators.magnetosonic.variables.n = self.mhd.density
            self.propagators.magnetosonic.variables.u = self.mhd.velocity
            self.propagators.magnetosonic.variables.p = self.mhd.pressure
        if "CurrentCoupling5DDensity" not in turn_off:
            self.propagators.cc5d_density.variables.u = self.mhd.velocity
        if "CurrentCoupling5DGradB" not in turn_off:
            self.propagators.cc5d_gradb.variables.u = self.mhd.velocity
            self.propagators.cc5d_gradb.variables.energetic_ions = self.energetic_ions.var
        if "CurrentCoupling5DCurlb" not in turn_off:
            self.propagators.cc5d_curlb.variables.u = self.mhd.velocity
            self.propagators.cc5d_curlb.variables.energetic_ions = self.energetic_ions.var
        if "PushGuidingCenterBxEstar" not in turn_off:
            self.propagators.push_bxe.variables.ions = self.energetic_ions.var
        if "PushGuidingCenterParallel" not in turn_off:
            self.propagators.push_parallel.variables.ions = self.energetic_ions.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_U")
        self.add_scalar("en_p")
        self.add_scalar("en_B")
        self.add_scalar("en_fv", compute="from_particles", variable=self.energetic_ions.var)
        self.add_scalar("en_fB", compute="from_particles", variable=self.energetic_ions.var)
        self.add_scalar("en_tot", summands=["en_U", "en_p", "en_B", "en_fv", "en_fB"])

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self, verbose: bool = False):
        self._ones = Propagator.projected_equil.p3.space.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        self._en_fv = xp.empty(1, dtype=float)
        self._en_fB = xp.empty(1, dtype=float)
        self._en_tot = xp.empty(1, dtype=float)
        self._n_lost_particles = xp.empty(1, dtype=float)

        self._PB = getattr(Propagator.basis_ops, "PB")
        self._PBb = self._PB.codomain.zeros()

    def update_scalar_quantities(self):
        # scaling factor
        Ab = self.mhd.mass_number
        Ah = self.energetic_ions.var.species.mass_number

        # perturbed fields
        en_U = 0.5 * Propagator.mass_ops.M2n.dot_inner(
            self.mhd.velocity.spline.vector,
            self.mhd.velocity.spline.vector,
        )
        en_B = 0.5 * Propagator.mass_ops.M2.dot_inner(
            self.em_fields.b_field.spline.vector,
            self.em_fields.b_field.spline.vector,
        )
        en_p = self.mhd.pressure.spline.vector.inner(self._ones) / (5 / 3 - 1)

        self.update_scalar("en_U", en_U)
        self.update_scalar("en_B", en_B)
        self.update_scalar("en_p", en_p)

        # particles' energy
        particles = self.energetic_ions.var.particles

        self._en_fv[0] = (
            particles.markers[~particles.holes, 5].dot(
                particles.markers[~particles.holes, 3] ** 2,
            )
            / (2.0)
            * Ah
            / Ab
        )

        self._PBb = self._PB.dot(self.em_fields.b_field.spline.vector)
        particles.save_magnetic_energy(self._PBb)

        self._en_fB[0] = (
            particles.markers[~particles.holes, 5].dot(
                particles.markers[~particles.holes, 8],
            )
            * Ah
            / Ab
        )

        self.update_scalar("en_fv", self._en_fv[0])
        self.update_scalar("en_fB", self._en_fB[0])
        self.update_scalar("en_tot")

        # print number of lost particles
        n_lost_markers = xp.array(particles.n_lost_markers)

        if Propagator.derham.comm is not None:
            Propagator.derham.comm.Allreduce(
                MPI.IN_PLACE,
                n_lost_markers,
                op=MPI.SUM,
            )

        if self.clone_config is not None:
            self.clone_config.inter_comm.Allreduce(
                MPI.IN_PLACE,
                n_lost_markers,
                op=MPI.SUM,
            )

        if rank == 0:
            print(
                "Lost particle ratio: ",
                n_lost_markers / particles.Np * 100,
                "% \n",
            )

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "shearalfen_cc5d.Options" in line:
                    new_file += [
                        """model.propagators.shearalfen_cc5d.options = model.propagators.shearalfen_cc5d.Options(
                        energetic_ions = model.energetic_ions.var,)\n""",
                    ]

                elif "magnetosonic.Options" in line:
                    new_file += [
                        """model.propagators.magnetosonic.options = model.propagators.magnetosonic.Options(
                        b_field=model.em_fields.b_field,)\n""",
                    ]

                elif "cc5d_density.Options" in line:
                    new_file += [
                        """model.propagators.cc5d_density.options = model.propagators.cc5d_density.Options(
                        energetic_ions = model.energetic_ions.var,
                        b_tilde = model.em_fields.b_field,)\n""",
                    ]

                elif "cc5d_curlb.Options" in line:
                    new_file += [
                        """model.propagators.cc5d_curlb.options = model.propagators.cc5d_curlb.Options(
                        b_tilde = model.em_fields.b_field,)\n""",
                    ]

                elif "cc5d_gradb.Options" in line:
                    new_file += [
                        """model.propagators.cc5d_gradb.options = model.propagators.cc5d_gradb.Options(
                        b_tilde = model.em_fields.b_field,)\n""",
                    ]

                elif "push_bxe.Options" in line:
                    new_file += [
                        """model.propagators.push_bxe.options = model.propagators.push_bxe.Options(
                        b_tilde = model.em_fields.b_field,)\n""",
                    ]

                elif "push_parallel.Options" in line:
                    new_file += [
                        """model.propagators.push_parallel.options = model.propagators.push_parallel.Options(
                        b_tilde = model.em_fields.b_field,)\n""",
                    ]

                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
