import numpy as np
from mpi4py import MPI
from psydac.linalg.block import BlockVector

from struphy.models.base import StruphyModel
from struphy.models.species import FieldSpecies, FluidSpecies, ParticleSpecies
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.polar.basic import PolarVector
from struphy.propagators import propagators_coupling, propagators_fields, propagators_markers
from struphy.utils.pyccel import Pyccelkernel

rank = MPI.COMM_WORLD.Get_rank()


rank = MPI.COMM_WORLD.Get_rank()


class LinearMHDVlasovCC(StruphyModel):
    r"""
    Hybrid linear MHD + energetic ions (6D Vlasov) with **current coupling scheme**.

    :ref:`normalization`:

    .. math::

        \hat U = \hat v = \hat v_\textnormal{A} \,, \qquad \hat f_\textnormal{h} = \frac{\hat n}{\hat v_\textnormal{A}^3} \,.

    :ref:`Equations <gempic>`:

    .. math::

        \begin{align}
        \textnormal{MHD}\,\, &\left\{\,\,
        \begin{aligned}
        &\frac{\partial \tilde{\rho}}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,,
        \\[2mm]
        \rho_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 + \mathbf{J}_0\times \tilde{\mathbf{B}} \color{blue} + \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon} \left(n_\textnormal{h}\tilde{\mathbf{U}}-n_\textnormal{h}\mathbf{u}_\textnormal{h}\right)\times(\mathbf{B}_0+\tilde{\mathbf{B}}) \color{black}\,,
        \\[2mm]
        &\frac{\partial \tilde p}{\partial t} + (\gamma-1)\nabla\cdot(p_0 \tilde{\mathbf{U}})
        + p_0\nabla\cdot \tilde{\mathbf{U}}=0\,,
        \\[2mm]
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} = \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)\,,\qquad \nabla\cdot\tilde{\mathbf{B}}=0\,,
        \end{aligned}
        \right.
        \\[2mm]
        \textnormal{EPs}\,\, &\left\{\,\,
        \begin{aligned}
        &\quad\,\,\frac{\partial f_\textnormal{h}}{\partial t}+\mathbf{v}\cdot\nabla f_\textnormal{h} + \frac{1}{\varepsilon} \left[\color{blue} (\mathbf{B}_0+\tilde{\mathbf{B}})\times\tilde{\mathbf{U}} \color{black} + \mathbf{v}\times(\mathbf{B}_0+\tilde{\mathbf{B}})\right]\cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} =0\,,
        \\[2mm]
        &\quad\,\,n_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\,\textnormal{d}^3 \mathbf v\,,\qquad n_\textnormal{h}\mathbf{u}_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\mathbf{v}\,\textnormal{d}^3 \mathbf v\,,
        \end{aligned}
        \right.
        \end{align}

    where :math:`\mathbf{J}_0 = \nabla\times\mathbf{B}_0` and

    .. math::

        \varepsilon = \frac{1}{\hat \Omega_{\textnormal{c,hot}} \hat t}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{c,hot}} = \frac{Z_\textnormal{h}e \hat B}{A_\textnormal{h} m_\textnormal{H}}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.CurrentCoupling6DDensity`
    2. :class:`~struphy.propagators.propagators_fields.ShearAlfven`
    3. :class:`~struphy.propagators.propagators_coupling.CurrentCoupling6DCurrent`
    4. :class:`~struphy.propagators.propagators_markers.PushEta`
    5. :class:`~struphy.propagators.propagators_markers.PushVxB`
    6. :class:`~struphy.propagators.propagators_fields.Magnetosonic`

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
            self.velocity = FEECVariable(space="Hdiv")
            self.pressure = FEECVariable(space="L2")
            self.init_variables()

    class EnergeticIons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles6D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.couple_dens = propagators_fields.CurrentCoupling6DDensity()
            self.shear_alf = propagators_fields.ShearAlfven()
            self.couple_curr = propagators_coupling.CurrentCoupling6DCurrent()
            self.push_eta = propagators_markers.PushEta()
            self.push_vxb = propagators_markers.PushVxB()
            self.mag_sonic = propagators_fields.Magnetosonic()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD()
        self.energetic_ions = self.EnergeticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.couple_dens.variables.u = self.mhd.velocity

        self.propagators.shear_alf.variables.u = self.mhd.velocity
        self.propagators.shear_alf.variables.b = self.em_fields.b_field

        self.propagators.couple_curr.variables.ions = self.energetic_ions.var
        self.propagators.couple_curr.variables.u = self.mhd.velocity

        self.propagators.push_eta.variables.var = self.energetic_ions.var
        self.propagators.push_vxb.variables.ions = self.energetic_ions.var

        self.propagators.mag_sonic.variables.n = self.mhd.density
        self.propagators.mag_sonic.variables.u = self.mhd.velocity
        self.propagators.mag_sonic.variables.p = self.mhd.pressure

        # define scalars for update_scalar_quantities
        self.add_scalar("en_U", compute="from_field")
        self.add_scalar("en_p", compute="from_field")
        self.add_scalar("en_B", compute="from_field")
        self.add_scalar("en_f", compute="from_particles", variable=self.energetic_ions.var)
        self.add_scalar("en_tot", summands=["en_U", "en_p", "en_B", "en_f"])
        self.add_scalar("n_lost_particles", compute="from_particles", variable=self.energetic_ions.var)

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self):
        self._ones = self.projected_equil.p3.space.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        self._tmp = xp.empty(1, dtype=float)
        self._n_lost_particles = xp.empty(1, dtype=float)

        # add control variate to mass_ops object
        if self.energetic_ions.var.particles.control_variate:
            self.mass_ops.weights["f0"] = self.energetic_ions.var.particles.f0

        self._Ah = self.energetic_ions.mass_number
        self._Ab = self.mhd.mass_number

    def update_scalar_quantities(self):
        # perturbed fields
        u = self.mhd.velocity.spline.vector
        p = self.mhd.pressure.spline.vector
        b = self.em_fields.b_field.spline.vector
        particles = self.energetic_ions.var.particles

        en_U = 0.5 * self.mass_ops.M2n.dot_inner(u, u)
        en_B = 0.5 * self.mass_ops.M2.dot_inner(b, b)
        en_p = p.inner(self._ones) / (5 / 3 - 1)

        self.update_scalar("en_U", en_U)
        self.update_scalar("en_B", en_B)
        self.update_scalar("en_p", en_p)

        # particles
        self._tmp[0] = (
            self._Ah
            / self._Ab
            * particles.markers_wo_holes[:, 6].dot(
                particles.markers_wo_holes[:, 3] ** 2
                + particles.markers_wo_holes[:, 4] ** 2
                + particles.markers_wo_holes[:, 5] ** 2,
            )
            / (2)
        )

        self.update_scalar("en_f", self._tmp[0])
        self.update_scalar("en_tot", en_U + en_B + en_p + self._tmp[0])

        # Print number of lost ions
        self._n_lost_particles[0] = particles.n_lost_markers
        self.update_scalar("n_lost_particles", self._n_lost_particles[0])

        if rank == 0:
            print(
                "ratio of lost particles: ",
                self._n_lost_particles[0] / particles.Np * 100,
                "%",
            )

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "mag_sonic.Options" in line:
                    new_file += [
                        "model.propagators.mag_sonic.options = model.propagators.mag_sonic.Options(b_field=model.em_fields.b_field)\n"
                    ]
                elif "couple_dens.Options" in line:
                    new_file += [
                        "model.propagators.couple_dens.options = model.propagators.couple_dens.Options(energetic_ions=model.energetic_ions.var,\n"
                    ]
                    new_file += [
                        "                                                                              b_tilde=model.em_fields.b_field)\n"
                    ]
                elif "couple_curr.Options" in line:
                    new_file += [
                        "model.propagators.couple_curr.options = model.propagators.couple_curr.Options(b_tilde=model.em_fields.b_field)\n"
                    ]
                elif "set_save_data" in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["model.energetic_ions.set_save_data(binning_plots=(binplot,))\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)


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

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["em_fields"]["b_field"] = "Hdiv"
        dct["fluid"]["mhd"] = {
            "density": "L2",
            "velocity": "Hdiv",
            "pressure": "L2",
        }
        dct["kinetic"]["energetic_ions"] = "Particles6D"
        return dct

    @staticmethod
    def bulk_species():
        return "mhd"

    @staticmethod
    def velocity_scale():
        return "alfvén"

    @staticmethod
    def propagators_dct():
        return {
            propagators_markers.PushEtaPC: ["energetic_ions"],
            propagators_markers.PushVxB: ["energetic_ions"],
            propagators_coupling.PressureCoupling6D: ["energetic_ions", "mhd_velocity"],
            propagators_fields.ShearAlfven: ["mhd_velocity", "b_field"],
            propagators_fields.Magnetosonic: ["mhd_density", "mhd_velocity", "mhd_pressure"],
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    # add special options
    @classmethod
    def options(cls):
        dct = super().options()
        cls.add_option(
            species=["fluid", "mhd"],
            key="u_space",
            option="Hdiv",
            dct=dct,
        )
        return dct

    def __init__(self, params, comm, clone_config=None):
        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        from struphy.polar.basic import PolarVector

        # extract necessary parameters
        u_space = params["fluid"]["mhd"]["options"]["u_space"]
        params_alfven = params["fluid"]["mhd"]["options"]["ShearAlfven"]
        params_sonic = params["fluid"]["mhd"]["options"]["Magnetosonic"]
        params_vxb = params["kinetic"]["energetic_ions"]["options"]["PushVxB"]
        params_pressure = params["kinetic"]["energetic_ions"]["options"]["PressureCoupling6D"]

        # use perp model
        assert (
            params["kinetic"]["energetic_ions"]["options"]["PressureCoupling6D"]["use_perp_model"]
            == params["kinetic"]["energetic_ions"]["options"]["PressureCoupling6D"]["use_perp_model"]
        )
        use_perp_model = params["kinetic"]["energetic_ions"]["options"]["PressureCoupling6D"]["use_perp_model"]

        # compute coupling parameters
        Ab = params["fluid"]["mhd"]["phys_params"]["A"]
        Ah = params["kinetic"]["energetic_ions"]["phys_params"]["A"]
        epsilon = self.equation_params["energetic_ions"]["epsilon"]

        if abs(epsilon - 1) < 1e-6:
            epsilon = 1.0

        self._coupling_params = {}
        self._coupling_params["Ab"] = Ab
        self._coupling_params["Ah"] = Ah
        self._coupling_params["epsilon"] = epsilon

        # add control variate to mass_ops object
        if self.pointer["energetic_ions"].control_variate:
            self.mass_ops.weights["f0"] = self.pointer["energetic_ions"].f0

        # Project magnetic field
        self._b_eq = self.derham.P["2"](
            [
                self.equil.b2_1,
                self.equil.b2_2,
                self.equil.b2_3,
            ]
        )
        self._p_eq = self.derham.P["3"](self.equil.p3)
        self._ones = self._p_eq.space.zeros()

        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        # set keyword arguments for propagators
        self._kwargs[propagators_markers.PushEtaPC] = {
            "u": self.pointer["mhd_velocity"],
            "use_perp_model": use_perp_model,
            "u_space": u_space,
        }

        self._kwargs[propagators_markers.PushVxB] = {
            "algo": params_vxb["algo"],
            "kappa": epsilon,
            "b2": self.pointer["b_field"],
            "b2_add": self._b_eq,
        }

        if params_pressure["turn_off"]:
            self._kwargs[propagators_coupling.PressureCoupling6D] = None
        else:
            self._kwargs[propagators_coupling.PressureCoupling6D] = {
                "use_perp_model": use_perp_model,
                "u_space": u_space,
                "solver": params_pressure["solver"],
                "coupling_params": self._coupling_params,
                "filter": params_pressure["filter"],
                "boundary_cut": params_pressure["boundary_cut"],
            }

        if params_alfven["turn_off"]:
            self._kwargs[propagators_fields.ShearAlfven] = None
        else:
            self._kwargs[propagators_fields.ShearAlfven] = {
                "u_space": u_space,
                "solver": params_alfven["solver"],
            }

        if params_sonic["turn_off"]:
            self._kwargs[propagators_fields.Magnetosonic] = None
        else:
            self._kwargs[propagators_fields.Magnetosonic] = {
                "b": self.pointer["b_field"],
                "u_space": u_space,
                "solver": params_sonic["solver"],
            }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation:
        self.add_scalar("en_U", compute="from_field")
        self.add_scalar("en_p", compute="from_field")
        self.add_scalar("en_B", compute="from_field")
        self.add_scalar("en_f", compute="from_particles", species="energetic_ions")
        self.add_scalar("en_tot", summands=["en_U", "en_p", "en_B", "en_f"])
        self.add_scalar("n_lost_particles", compute="from_particles", species="energetic_ions")

        # temporary vectors for scalar quantities
        self._tmp_u = self.derham.Vh["2"].zeros()
        self._tmp_b1 = self.derham.Vh["2"].zeros()
        self._tmp = xp.empty(1, dtype=float)
        self._n_lost_particles = xp.empty(1, dtype=float)

    def update_scalar_quantities(self):
        # perturbed fields
        if "Hdiv" == "Hdiv":
            en_U = 0.5 * self.mass_ops.M2n.dot_inner(self.pointer["mhd_velocity"], self.pointer["mhd_velocity"])
        else:
            en_U = 0.5 * self.mass_ops.Mvn.dot_inner(self.pointer["mhd_velocity"], self.pointer["mhd_velocity"])
        en_B = 0.5 * self.mass_ops.M2.dot_inner(self.pointer["b_field"], self.pointer["b_field"])
        en_p = self.pointer["mhd_pressure"].inner(self._ones) / (5 / 3 - 1)

        self.update_scalar("en_U", en_U)
        self.update_scalar("en_B", en_B)
        self.update_scalar("en_p", en_p)

        # particles
        self._tmp[0] = (
            self._coupling_params["Ah"]
            / self._coupling_params["Ab"]
            * self.pointer["energetic_ions"]
            .markers_wo_holes[:, 6]
            .dot(
                self.pointer["energetic_ions"].markers_wo_holes[:, 3] ** 2
                + self.pointer["energetic_ions"].markers_wo_holes[:, 4] ** 2
                + self.pointer["energetic_ions"].markers_wo_holes[:, 5] ** 2,
            )
            / (2.0)
        )

        self.update_scalar("en_f", self._tmp[0])
        self.update_scalar("en_tot", en_U + en_B + en_p + self._tmp[0])

        # Print number of lost ions
        self._n_lost_particles[0] = self.pointer["energetic_ions"].n_lost_markers
        self.update_scalar("n_lost_particles", self._n_lost_particles[0])
        if self.rank_world == 0:
            print(
                "ratio of lost particles: ",
                self._n_lost_particles[0] / self.pointer["energetic_ions"].Np * 100,
                "%",
            )


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
            if not "PushGuidingCenterBxEstar" in turn_off:
                self.push_bxe = propagators_markers.PushGuidingCenterBxEstar()
            if not "PushGuidingCenterParallel" in turn_off:
                self.push_parallel = propagators_markers.PushGuidingCenterParallel()
            if not "ShearAlfvenCurrentCoupling5D" in turn_off:
                self.shearalfen_cc5d = propagators_fields.ShearAlfvenCurrentCoupling5D()
            if not "Magnetosonic" in turn_off:
                self.magnetosonic = propagators_fields.Magnetosonic()
            if not "CurrentCoupling5DDensity" in turn_off:
                self.cc5d_density = propagators_fields.CurrentCoupling5DDensity()
            if not "CurrentCoupling5DGradB" in turn_off:
                self.cc5d_gradb = propagators_coupling.CurrentCoupling5DGradB()
            if not "CurrentCoupling5DCurlb" in turn_off:
                self.cc5d_curlb = propagators_coupling.CurrentCoupling5DCurlb()

    def __init__(self, turn_off: tuple[str, ...] = (None,)):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD()
        self.energetic_ions = self.EnergeticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators(turn_off)

        # 3. assign variables to propagators
        if not "ShearAlfvenCurrentCoupling5D" in turn_off:
            self.propagators.shearalfen_cc5d.variables.u = self.mhd.velocity
            self.propagators.shearalfen_cc5d.variables.b = self.em_fields.b_field
        if not "Magnetosonic" in turn_off:
            self.propagators.magnetosonic.variables.n = self.mhd.density
            self.propagators.magnetosonic.variables.u = self.mhd.velocity
            self.propagators.magnetosonic.variables.p = self.mhd.pressure
        if not "CurrentCoupling5DDensity" in turn_off:
            self.propagators.cc5d_density.variables.u = self.mhd.velocity
        if not "CurrentCoupling5DGradB" in turn_off:
            self.propagators.cc5d_gradb.variables.u = self.mhd.velocity
            self.propagators.cc5d_gradb.variables.energetic_ions = self.energetic_ions.var
        if not "CurrentCoupling5DCurlb" in turn_off:
            self.propagators.cc5d_curlb.variables.u = self.mhd.velocity
            self.propagators.cc5d_curlb.variables.energetic_ions = self.energetic_ions.var
        if not "PushGuidingCenterBxEstar" in turn_off:
            self.propagators.push_bxe.variables.ions = self.energetic_ions.var
        if not "PushGuidingCenterParallel" in turn_off:
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

    def allocate_helpers(self):
        self._ones = self.projected_equil.p3.space.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        self._en_fv = xp.empty(1, dtype=float)
        self._en_fB = xp.empty(1, dtype=float)
        self._en_tot = xp.empty(1, dtype=float)
        self._n_lost_particles = xp.empty(1, dtype=float)

        self._PB = getattr(self.basis_ops, "PB")
        self._PBb = self._PB.codomain.zeros()

    def update_scalar_quantities(self):
        # scaling factor
        Ab = self.mhd.mass_number
        Ah = self.energetic_ions.var.species.mass_number

        # perturbed fields
        en_U = 0.5 * self.mass_ops.M2n.dot_inner(
            self.mhd.velocity.spline.vector,
            self.mhd.velocity.spline.vector,
        )
        en_B = 0.5 * self.mass_ops.M2.dot_inner(
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

        self.derham.comm.Allreduce(
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
                        energetic_ions = model.energetic_ions.var,)\n"""
                    ]

                elif "magnetosonic.Options" in line:
                    new_file += [
                        """model.propagators.magnetosonic.options = model.propagators.magnetosonic.Options(
                        b_field=model.em_fields.b_field,)\n"""
                    ]

                elif "cc5d_density.Options" in line:
                    new_file += [
                        """model.propagators.cc5d_density.options = model.propagators.cc5d_density.Options(
                        energetic_ions = model.energetic_ions.var,
                        b_tilde = model.em_fields.b_field,)\n"""
                    ]

                elif "cc5d_curlb.Options" in line:
                    new_file += [
                        """model.propagators.cc5d_curlb.options = model.propagators.cc5d_curlb.Options(
                        b_tilde = model.em_fields.b_field,)\n"""
                    ]

                elif "cc5d_gradb.Options" in line:
                    new_file += [
                        """model.propagators.cc5d_gradb.options = model.propagators.cc5d_gradb.Options(
                        b_tilde = model.em_fields.b_field,)\n"""
                    ]

                elif "push_bxe.Options" in line:
                    new_file += [
                        """model.propagators.push_bxe.options = model.propagators.push_bxe.Options(
                        b_tilde = model.em_fields.b_field,)\n"""
                    ]

                elif "push_parallel.Options" in line:
                    new_file += [
                        """model.propagators.push_parallel.options = model.propagators.push_parallel.Options(
                        b_tilde = model.em_fields.b_field,)\n"""
                    ]

                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)


class ColdPlasmaVlasov(StruphyModel):
    r"""Cold plasma hybrid model.

    :ref:`normalization`:

    .. math::

        \hat v = c\,,\qquad \hat E = c \hat B \,,\qquad \hat f = \frac{\hat n}{c^3} \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon_\textnormal{h}}\Big[ \mathbf{E} + \mathbf{v} \times \left( \mathbf{B} + \mathbf{B}_0 \right) \Big]
            \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
        \\[2mm]
        \frac{1}{n_0} &\frac{\partial \mathbf j_\textnormal{c}}{\partial t} = \frac{1}{\varepsilon_\textnormal{c}} \mathbf E + \frac{1}{\varepsilon_\textnormal{c} n_0} \mathbf j_\textnormal{c} \times \mathbf B_0\,,
        \\[2mm]
        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,,
        \\[2mm]
        -&\frac{\partial \mathbf E}{\partial t} + \nabla\times\mathbf B =
        \frac{\alpha^2}{\varepsilon_\textnormal{h}} \left( \mathbf j_\textnormal{c} + \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v} \right) \,,

    where :math:`(n_0,\mathbf B_0)` denotes a (inhomogeneous) background and

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p,cold}}{\hat \Omega_\textnormal{c,cold}}\,, \qquad \varepsilon_\textnormal{c} = \frac{1}{\hat \Omega_\textnormal{c,cold} \hat t}\,, \qquad \varepsilon_\textnormal{h} = \frac{1}{\hat \Omega_\textnormal{c,hot} \hat t} \,.

    At initial time the Poisson equation is solved once to weakly satisfy the Gauss law:

    .. math::

        \begin{align}
            \nabla \cdot \mathbf{E} & = \nu \frac{\alpha^2}{\varepsilon_\textnormal{h}} \int_{\mathbb{R}^3} f \, \text{d}^3 \mathbf{v}\,.
        \end{align}

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.Maxwell`
    2. :class:`~struphy.propagators.propagators_fields.OhmCold`
    3. :class:`~struphy.propagators.propagators_fields.JxBCold`
    4. :class:`~struphy.propagators.propagators_markers.PushVxB`
    5. :class:`~struphy.propagators.propagators_markers.PushEta`
    6. :class:`~struphy.propagators.propagators_coupling.VlasovAmpere`
    """

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class ThermalElectrons(FluidSpecies):
        def __init__(self):
            self.current = FEECVariable(space="Hcurl")
            self.init_variables()

    class HotElectrons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles6D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.maxwell = propagators_fields.Maxwell()
            self.ohm = propagators_fields.OhmCold()
            self.jxb = propagators_fields.JxBCold()
            self.push_eta = propagators_markers.PushEta()
            self.push_vxb = propagators_markers.PushVxB()
            self.coupling_va = propagators_coupling.VlasovAmpere()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.thermal_elec = self.ThermalElectrons()
        self.hot_elec = self.HotElectrons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field

        self.propagators.ohm.variables.j = self.thermal_elec.current
        self.propagators.ohm.variables.e = self.em_fields.e_field

        self.propagators.jxb.variables.j = self.thermal_elec.current

        self.propagators.push_eta.variables.var = self.hot_elec.var
        self.propagators.push_vxb.variables.ions = self.hot_elec.var

        self.propagators.coupling_va.variables.e = self.em_fields.e_field
        self.propagators.coupling_va.variables.ions = self.hot_elec.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_E")
        self.add_scalar("en_B")
        self.add_scalar("en_J")
        self.add_scalar("en_f", compute="from_particles", variable=self.hot_elec.var)
        self.add_scalar("en_tot")

        # initial Poisson (not a propagator used in time stepping)
        self.initial_poisson = propagators_fields.Poisson()
        self.initial_poisson.variables.phi = self.em_fields.phi

    @property
    def bulk_species(self):
        return self.thermal_elec

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self):
        self._tmp = xp.empty(1, dtype=float)

    def update_scalar_quantities(self):
        # e*M1*e/2
        e = self.em_fields.e_field.spline.vector
        en_E = 0.5 * self.mass_ops.M1.dot_inner(e, e)
        self.update_scalar("en_E", en_E)

        # alpha^2 / 2 / N * sum_p w_p v_p^2
        particles = self.hot_elec.var.particles
        alpha = self.hot_elec.equation_params.alpha
        self._tmp[0] = (
            alpha**2
            / (2 * particles.Np)
            * xp.dot(
                particles.markers_wo_holes[:, 3] ** 2
                + particles.markers_wo_holes[:, 4] ** 2
                + particles.markers_wo_holes[:, 5] ** 2,
                particles.markers_wo_holes[:, 6],
            )
        )
        self.update_scalar("en_f", self._tmp[0])

        # en_tot = en_w + en_e
        self.update_scalar("en_tot", en_E + self._tmp[0])

    def allocate_propagators(self):
        """Solve initial Poisson equation.

        :meta private:
        """

        # initialize fields and particles
        super().allocate_propagators()

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nINITIAL POISSON SOLVE:")

        # use control variate method
        particles = self.hot_elec.var.particles
        particles.update_weights()

        # sanity check
        # self.pointer['species1'].show_distribution_function(
        #     [True] + [False]*5, [xp.linspace(0, 1, 32)])

        # accumulate charge density
        charge_accum = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels.charge_density_0form),
            self.mass_ops,
            self.domain.args_domain,
        )

        # another sanity check: compute FE coeffs of density
        # charge_accum.show_accumulated_spline_field(self.mass_ops)

        alpha = self.hot_elec.equation_params.alpha
        epsilon = self.hot_elec.equation_params.epsilon

        self.initial_poisson.options.rho = charge_accum
        self.initial_poisson.options.rho_coeffs = alpha**2 / epsilon
        self.initial_poisson.allocate()

        # Solve with dt=1. and compute electric field
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nSolving initial Poisson problem...")
        self.initial_poisson(1.0)

        phi = self.initial_poisson.variables.phi.spline.vector
        self.derham.grad.dot(-phi, out=self.em_fields.e_field.spline.vector)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Done.")

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "coupling_va.Options" in line:
                    new_file += [line]
                    new_file += ["model.initial_poisson.options = model.initial_poisson.Options()\n"]
                elif "set_save_data" in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["model.hot_elec.set_save_data(binning_plots=(binplot,))\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
