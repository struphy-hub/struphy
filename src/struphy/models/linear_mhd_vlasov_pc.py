import logging

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalar, Scalars, VolumeFormEnergyFEEC
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

logger = logging.getLogger("struphy")
rank = MPI.COMM_WORLD.Get_rank()


class LinearMHDVlasovPC(StruphyModel):
    r"""
    Hybrid linear MHD + energetic ions (6D Vlasov) with **pressure coupling scheme**.

    :ref:`normalization`:

    .. math::

        \hat U = \hat v =: \hat v_\textnormal{A, bulk} \,, \qquad
        \hat f_\textnormal{h} = \frac{\hat n}{\hat v_\textnormal{A}^3} \,,\qquad 
        \hat{\mathbb{P}}_\textnormal{h} = A_\textnormal{h}m_\textnormal{H}\hat n \hat v_\textnormal{A}^2\,,

    Implemented equations:

    .. math::

        \begin{align}
        \textnormal{MHD} &\left\{
        \begin{aligned}
        &\frac{\partial \tilde{\rho}}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,, 
        \\
        \rho_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p + \frac{A_\textnormal{h}}{A_\textnormal{b}} \nabla\cdot \tilde{\mathbb{P}}_{\textnormal{h},\perp}
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}}
        \,, \qquad
        \mathbf{J}_0 = \nabla\times\mathbf{B}_0\,, 
        \\
        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) 
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,, 
        \\
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{EPs}\,\, &\left\{\,\,
        \begin{aligned}
        &\quad\,\,\frac{\partial f_\textnormal{h}}{\partial t} + (\mathbf{v} + \tilde{\mathbf{U}}_\perp)\cdot \nabla f_\textnormal{h}
        + \left[\frac{1}{\epsilon}\, \mathbf{v}\times(\mathbf{B}_0 + \tilde{\mathbf{B}}) - \nabla \tilde{\mathbf{U}}_\perp\cdot \mathbf{v} \right]\cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}}
        = 0\,,
        \\
        &\quad\,\,\tilde{\mathbb{P}}_{\textnormal{h},\perp} = \int \mathbf{v}_\perp\mathbf{v}^\top_\perp f_\textnormal{h} d\mathbf{v} \,,
        \end{aligned}
        \right.
        \end{align}

    where 

    .. math::

        \epsilon = \frac{\hat \omega}{2 \pi \, \hat \Omega_{\textnormal{c,hot}}} \,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{c,hot}} = \frac{Z_\textnormal{h}e \hat B}{A_\textnormal{h} m_\textnormal{H}}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushEtaPC`
    2. :class:`~struphy.propagators.propagators_markers.PushVxB`
    3. :class:`~struphy.propagators.propagators_coupling.PressureCoupling6D`
    4. :class:`~struphy.propagators.propagators_fields.ShearAlfven`
    5. :class:`~struphy.propagators.propagators_fields.Magnetosonic`

    :ref:`Model info <add_model>`:
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Hybrid"

    ## species
    class EnergeticIons(ParticleSpecies):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            epsilon: float = None,
        ):
            self.var = PICVariable(space="Particles6D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                epsilon=epsilon,
            )

    class EMFields(FieldSpecies):
        def __init__(self):
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    class MHD(FluidSpecies):
        def __init__(self, mass_number: float = 1.0):
            self.density = FEECVariable(space="L2")
            self.pressure = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="Hdiv")
            self.init_variables(mass_number=mass_number)

    ## propagators

    class Propagators:
        def __init__(self, turn_off: tuple[str, ...] = (None,)):
            if "PushEtaPC" not in turn_off:
                self.push_eta_pc = propagators_markers.PushEtaPC()
            if "PushVxB" not in turn_off:
                self.push_vxb = propagators_markers.PushVxB()
            if "PressureCoupling6D" not in turn_off:
                self.pc6d = propagators_coupling.PressureCoupling6D()
            if "ShearAlfven" not in turn_off:
                self.shearalfven = propagators_fields.ShearAlfven()
            if "Magnetosonic" not in turn_off:
                self.magnetosonic = propagators_fields.Magnetosonic()

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mhd_mass_number: float = 1.0,
        hot_charge_number: int = 1,
        hot_mass_number: float = 1.0,
        hot_epsilon: float = None,
        turn_off: tuple[str, ...] = (None,),
    ):

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD(mhd_mass_number)
        self.energetic_ions = self.EnergeticIons(
            hot_charge_number,
            hot_mass_number,
            hot_epsilon,
        )

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(turn_off)

        # 4. assign variables to propagators
        if "ShearAlfven" not in turn_off:
            self.propagators.shearalfven.variables.u = self.mhd.velocity
            self.propagators.shearalfven.variables.b = self.em_fields.b_field
        if "Magnetosonic" not in turn_off:
            self.propagators.magnetosonic.variables.n = self.mhd.density
            self.propagators.magnetosonic.variables.u = self.mhd.velocity
            self.propagators.magnetosonic.variables.p = self.mhd.pressure
        if "PressureCoupling6D" not in turn_off:
            self.propagators.pc6d.variables.u = self.mhd.velocity
            self.propagators.pc6d.variables.energetic_ions = self.energetic_ions.var
        if "PushEtaPC" not in turn_off:
            self.propagators.push_eta_pc.variables.var = self.energetic_ions.var
        if "PushVxB" not in turn_off:
            self.propagators.push_vxb.variables.ions = self.energetic_ions.var

        kinetic_energy = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="M2n", normalization=0.5)
        pressure_energy = VolumeFormEnergyFEEC(self.mhd.pressure, normalization=1.0 / (5 / 3 - 1))
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field, bilinear_form_name="M2", normalization=0.5)
        particle_energy = FunctionScalar(self._compute_en_f, self.energetic_ions.var)
        self.scalars = Scalars(
            en_U=kinetic_energy,
            en_p=pressure_energy,
            en_B=magnetic_energy,
            en_f=particle_energy,
            en_tot=kinetic_energy + pressure_energy + magnetic_energy + particle_energy,
        )

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

        self._n_lost_particles = xp.empty(1, dtype=float)

    def _compute_en_f(self):
        Ab = self.mhd.mass_number
        Ah = self.energetic_ions.var.species.mass_number
        particles = self.energetic_ions.var.particles
        return (
            particles.markers[~particles.holes, 6].dot(
                particles.markers[~particles.holes, 3] ** 2
                + particles.markers[~particles.holes, 4] ** 2
                + particles.markers[~particles.holes, 5] ** 2,
            )
            / 2.0
            * Ah
            / Ab
        )

    def update_scalar_quantities(self):
        self.scalars.update()
        particles = self.energetic_ions.var.particles
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
            logger.info(
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
                if "magnetosonic.Options" in line:
                    new_file += [
                        """model.propagators.magnetosonic.options = model.propagators.magnetosonic.Options(
                        b_field=model.em_fields.b_field,)\n""",
                    ]

                elif "push_eta_pc.Options" in line:
                    new_file += [
                        """model.propagators.push_eta_pc.options = model.propagators.push_eta_pc.Options(
                        u_tilde = model.mhd.velocity,)\n""",
                    ]

                elif "push_vxb.Options" in line:
                    new_file += [
                        """model.propagators.push_vxb.options = model.propagators.push_vxb.Options(
                        b2_var = model.em_fields.b_field,)\n""",
                    ]

                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
