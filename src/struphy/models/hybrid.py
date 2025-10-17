from struphy.models.base import StruphyModel
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc
from struphy.propagators import propagators_coupling, propagators_fields, propagators_markers
from struphy.utils.arrays import xp as np
from struphy.utils.pyccel import Pyccelkernel


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

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["em_fields"]["b_field"] = "Hdiv"
        dct["fluid"]["mhd"] = {"density": "L2", "velocity": "Hdiv", "pressure": "L2"}
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
            propagators_fields.CurrentCoupling6DDensity: ["mhd_velocity"],
            propagators_fields.ShearAlfven: ["mhd_velocity", "b_field"],
            propagators_coupling.CurrentCoupling6DCurrent: ["energetic_ions", "mhd_velocity"],
            propagators_markers.PushEta: ["energetic_ions"],
            propagators_markers.PushVxB: ["energetic_ions"],
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

        # prelim
        e_ions_params = self.kinetic["energetic_ions"]["params"]

        # extract necessary parameters
        u_space = params["fluid"]["mhd"]["options"]["u_space"]
        params_alfven = params["fluid"]["mhd"]["options"]["ShearAlfven"]
        params_sonic = params["fluid"]["mhd"]["options"]["Magnetosonic"]
        params_eta = params["kinetic"]["energetic_ions"]["options"]["PushEta"]
        params_vxb = params["kinetic"]["energetic_ions"]["options"]["PushVxB"]
        params_density = params["fluid"]["mhd"]["options"]["CurrentCoupling6DDensity"]
        params_current = params["kinetic"]["energetic_ions"]["options"]["CurrentCoupling6DCurrent"]

        # compute coupling parameters
        Ab = params["fluid"]["mhd"]["phys_params"]["A"]
        Ah = params["kinetic"]["energetic_ions"]["phys_params"]["A"]
        epsilon = self.equation_params["energetic_ions"]["epsilon"]

        if abs(epsilon - 1) < 1e-6:
            epsilon = 1.0

        self._Ab = Ab
        self._Ah = Ah

        # add control variate to mass_ops object
        if self.pointer["energetic_ions"].control_variate:
            self.mass_ops.weights["f0"] = self.pointer["energetic_ions"].f0

        # project background magnetic field (2-form) and background pressure (3-form)
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
        if params_density["turn_off"]:
            self._kwargs[propagators_fields.CurrentCoupling6DDensity] = None
        else:
            self._kwargs[propagators_fields.CurrentCoupling6DDensity] = {
                "particles": self.pointer["energetic_ions"],
                "u_space": u_space,
                "b_eq": self._b_eq,
                "b_tilde": self.pointer["b_field"],
                "Ab": Ab,
                "Ah": Ah,
                "epsilon": epsilon,
                "solver": params_density["solver"],
                "filter": params_density["filter"],
                "boundary_cut": params_density["boundary_cut"],
            }

        if params_alfven["turn_off"]:
            self._kwargs[propagators_fields.ShearAlfven] = None
        else:
            self._kwargs[propagators_fields.ShearAlfven] = {
                "u_space": u_space,
                "solver": params_alfven["solver"],
            }

        if params_current["turn_off"]:
            self._kwargs[propagators_coupling.CurrentCoupling6DCurrent] = None
        else:
            self._kwargs[propagators_coupling.CurrentCoupling6DCurrent] = {
                "u_space": u_space,
                "b_eq": self._b_eq,
                "b_tilde": self.pointer["b_field"],
                "Ab": Ab,
                "Ah": Ah,
                "epsilon": epsilon,
                "solver": params_current["solver"],
                "filter": params_current["filter"],
                "boundary_cut": params_current["boundary_cut"],
            }

        self._kwargs[propagators_markers.PushEta] = {
            "algo": params_eta["algo"],
        }

        self._kwargs[propagators_markers.PushVxB] = {
            "algo": params_vxb["algo"],
            "kappa": 1.0 / epsilon,
            "b2": self.pointer["b_field"],
            "b2_add": self._b_eq,
        }

        if params_sonic["turn_off"]:
            self._kwargs[propagators_fields.Magnetosonic] = None
        else:
            self._kwargs[propagators_fields.Magnetosonic] = {
                "u_space": u_space,
                "b": self.pointer["b_field"],
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

        # temporary vectors for scalar quantities:
        self._tmp = np.empty(1, dtype=float)
        self._n_lost_particles = np.empty(1, dtype=float)

    def update_scalar_quantities(self):
        # perturbed fields
        en_U = 0.5 * self.mass_ops.M2n.dot_inner(self.pointer["mhd_velocity"], self.pointer["mhd_velocity"])
        en_B = 0.5 * self.mass_ops.M2.dot_inner(self.pointer["b_field"], self.pointer["b_field"])
        en_p = self.pointer["mhd_pressure"].inner(self._ones) / (5 / 3 - 1)

        self.update_scalar("en_U", en_U)
        self.update_scalar("en_B", en_B)
        self.update_scalar("en_p", en_p)

        # particles
        self._tmp[0] = (
            self._Ah
            / self._Ab
            * self.pointer["energetic_ions"]
            .markers_wo_holes[:, 6]
            .dot(
                self.pointer["energetic_ions"].markers_wo_holes[:, 3] ** 2
                + self.pointer["energetic_ions"].markers_wo_holes[:, 4] ** 2
                + self.pointer["energetic_ions"].markers_wo_holes[:, 5] ** 2,
            )
            / (2)
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
        self._tmp = np.empty(1, dtype=float)
        self._n_lost_particles = np.empty(1, dtype=float)

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
        = (\nabla \times \tilde{\mathbf{B}}) \times (\mathbf{B}_0 + (\nabla \times \mathbf B_0) \times \tilde{\mathbf{B}}
        + \frac{A_\textnormal{h}}{A_\textnormal{b}} \left[ \frac{1}{\epsilon} n_\textnormal{gc} \tilde{\mathbf{U}} - \frac{1}{\epsilon} \mathbf{J}_\textnormal{gc} - \nabla \times \mathbf{M}_\textnormal{gc} \right] \times \mathbf{B} \,,
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
        \quad &\frac{\partial f_\textnormal{h}}{\partial t} + \frac{1}{B_\parallel^*}(v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^*)\cdot\nabla f_\textnormal{h}
        + \frac{1}{\epsilon} \frac{1}{B_\parallel^*} (\mathbf{B}^* \cdot \mathbf{E}^*) \frac{\partial f_\textnormal{h}}{\partial v_\parallel}
        = 0\,,
        \\
        & n_\textnormal{gc} = \int f_\textnormal{h} B_\parallel^* \,\textnormal dv_\parallel \textnormal d\mu \,,
        \\
        & \mathbf{J}_\textnormal{gc} = \int f_\textnormal{h}(v_\parallel \mathbf{B}^* - \mathbf{b}_0 \times \mathbf{E}^*) \,\textnormal dv_\parallel \textnormal d\mu \,,
        \\
        & \mathbf{M}_\textnormal{gc} = - \int f_\textnormal{h} B_\parallel^* \mu \mathbf{b}_0 \,\textnormal dv_\parallel \textnormal d\mu \,,
        \end{aligned}
        \right.
        \end{align}

    where 

    .. math::

        \begin{align}
        \mathbf{B}^* &= \mathbf{B} + \epsilon v_\parallel \nabla \times \mathbf{b}_0 \,,\qquad B^*_\parallel = \mathbf{b}_0 \cdot \mathbf{B}^*\,,
        \\[2mm]
        \mathbf{E}^* &= - \tilde{\mathbf{U}} \times \mathbf{B} - \epsilon \mu \nabla B_\parallel \,,
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
    7. :class:`~struphy.propagators.propagators_fields.MagnetosonicCurrentCoupling5D`

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
        dct["kinetic"]["energetic_ions"] = "Particles5D"
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
            propagators_markers.PushGuidingCenterBxEstar: ["energetic_ions"],
            propagators_markers.PushGuidingCenterParallel: ["energetic_ions"],
            propagators_coupling.CurrentCoupling5DGradB: ["energetic_ions", "mhd_velocity"],
            propagators_coupling.CurrentCoupling5DCurlb: ["energetic_ions", "mhd_velocity"],
            propagators_fields.CurrentCoupling5DDensity: ["mhd_velocity"],
            propagators_fields.ShearAlfvenCurrentCoupling5D: ["mhd_velocity", "b_field"],
            propagators_fields.MagnetosonicCurrentCoupling5D: ["mhd_density", "mhd_velocity", "mhd_pressure"],
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
        params_alfven = params["fluid"]["mhd"]["options"]["ShearAlfvenCurrentCoupling5D"]
        params_sonic = params["fluid"]["mhd"]["options"]["MagnetosonicCurrentCoupling5D"]
        params_density = params["fluid"]["mhd"]["options"]["CurrentCoupling5DDensity"]

        params_bxE = params["kinetic"]["energetic_ions"]["options"]["PushGuidingCenterBxEstar"]
        params_parallel = params["kinetic"]["energetic_ions"]["options"]["PushGuidingCenterParallel"]
        params_cc_gradB = params["kinetic"]["energetic_ions"]["options"]["CurrentCoupling5DGradB"]
        params_cc_curlb = params["kinetic"]["energetic_ions"]["options"]["CurrentCoupling5DCurlb"]
        params_cc_gradB = params["kinetic"]["energetic_ions"]["options"]["CurrentCoupling5DGradB"]

        # compute coupling parameters
        Ab = params["fluid"]["mhd"]["phys_params"]["A"]
        Ah = params["kinetic"]["energetic_ions"]["phys_params"]["A"]
        epsilon = self.equation_params["energetic_ions"]["epsilon"]

        self._coupling_params = {}
        self._coupling_params["Ab"] = Ab
        self._coupling_params["Ah"] = Ah

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

        self._absB0 = self.derham.P["0"](self.equil.absB0)

        self._unit_b1 = self.derham.P["1"](
            [
                self.equil.unit_b1_1,
                self.equil.unit_b1_2,
                self.equil.unit_b1_3,
            ]
        )

        self._unit_b2 = self.derham.P["2"](
            [
                self.equil.unit_b2_1,
                self.equil.unit_b2_2,
                self.equil.unit_b2_3,
            ]
        )

        self._gradB1 = self.derham.P["1"](
            [
                self.equil.gradB1_1,
                self.equil.gradB1_2,
                self.equil.gradB1_3,
            ]
        )

        self._curl_unit_b2 = self.derham.P["2"](
            [
                self.equil.curl_unit_b2_1,
                self.equil.curl_unit_b2_2,
                self.equil.curl_unit_b2_3,
            ]
        )

        self._p_eq = self.derham.P["3"](self.equil.p3)
        self._ones = self._p_eq.space.zeros()

        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        # set keyword arguments for propagators
        self._kwargs[propagators_markers.PushGuidingCenterBxEstar] = {
            "b_tilde": self.pointer["b_field"],
            "algo": params_bxE["algo"],
            "epsilon": epsilon,
        }

        self._kwargs[propagators_markers.PushGuidingCenterParallel] = {
            "b_tilde": self.pointer["b_field"],
            "algo": params_parallel["algo"],
            "epsilon": epsilon,
        }

        if params_cc_gradB["turn_off"]:
            self._kwargs[propagators_coupling.CurrentCoupling5DGradB] = None
        else:
            self._kwargs[propagators_coupling.CurrentCoupling5DGradB] = {
                "b": self.pointer["b_field"],
                "b_eq": self._b_eq,
                "unit_b1": self._unit_b1,
                "unit_b2": self._unit_b2,
                "absB0": self._absB0,
                "gradB1": self._gradB1,
                "curl_unit_b2": self._curl_unit_b2,
                "u_space": u_space,
                "solver": params_cc_gradB["solver"],
                "algo": params_cc_gradB["algo"],
                "filter": params_cc_gradB["filter"],
                "coupling_params": self._coupling_params,
                "epsilon": epsilon,
                "boundary_cut": params_cc_gradB["boundary_cut"],
            }

        if params_cc_curlb["turn_off"]:
            self._kwargs[propagators_coupling.CurrentCoupling5DCurlb] = None
        else:
            self._kwargs[propagators_coupling.CurrentCoupling5DCurlb] = {
                "b": self.pointer["b_field"],
                "b_eq": self._b_eq,
                "unit_b1": self._unit_b1,
                "absB0": self._absB0,
                "gradB1": self._gradB1,
                "curl_unit_b2": self._curl_unit_b2,
                "u_space": u_space,
                "solver": params_cc_curlb["solver"],
                "filter": params_cc_curlb["filter"],
                "coupling_params": self._coupling_params,
                "epsilon": epsilon,
                "boundary_cut": params_cc_curlb["boundary_cut"],
            }

        if params_density["turn_off"]:
            self._kwargs[propagators_fields.CurrentCoupling5DDensity] = None
        else:
            self._kwargs[propagators_fields.CurrentCoupling5DDensity] = {
                "particles": self.pointer["energetic_ions"],
                "b": self.pointer["b_field"],
                "b_eq": self._b_eq,
                "unit_b1": self._unit_b1,
                "curl_unit_b2": self._curl_unit_b2,
                "u_space": u_space,
                "solver": params_density["solver"],
                "coupling_params": self._coupling_params,
                "epsilon": epsilon,
                "boundary_cut": params_density["boundary_cut"],
            }

        if params_alfven["turn_off"]:
            self._kwargs[propagators_fields.ShearAlfvenCurrentCoupling5D] = None
        else:
            self._kwargs[propagators_fields.ShearAlfvenCurrentCoupling5D] = {
                "particles": self.pointer["energetic_ions"],
                "unit_b1": self._unit_b1,
                "absB0": self._absB0,
                "u_space": u_space,
                "solver": params_alfven["solver"],
                "filter": params_alfven["filter"],
                "coupling_params": self._coupling_params,
                "accumulated_magnetization": self.pointer["accumulated_magnetization"],
                "boundary_cut": params_alfven["boundary_cut"],
            }

        if params_sonic["turn_off"]:
            self._kwargs[propagators_fields.MagnetosonicCurrentCoupling5D] = None
        else:
            self._kwargs[propagators_fields.MagnetosonicCurrentCoupling5D] = {
                "particles": self.pointer["energetic_ions"],
                "b": self.pointer["b_field"],
                "unit_b1": self._unit_b1,
                "absB0": self._absB0,
                "u_space": u_space,
                "solver": params_sonic["solver"],
                "filter": params_sonic["filter"],
                "coupling_params": self._coupling_params,
                "boundary_cut": params_sonic["boundary_cut"],
            }

        # Initialize propagators used in splitting substeps
        self.init_propagators()
        # Scalar variables to be saved during simulation
        self.add_scalar("en_U", compute="from_field")
        self.add_scalar("en_p", compute="from_field")
        self.add_scalar("en_B", compute="from_field")
        self.add_scalar("en_fv", compute="from_particles", species="energetic_ions")
        self.add_scalar("en_fB", compute="from_particles", species="energetic_ions")
        # self.add_scalar('en_fv_lost', compute = 'from_particles', species='energetic_ions')
        # self.add_scalar('en_fB_lost', compute = 'from_particles', species='energetic_ions')
        # self.add_scalar('en_tot',summands = ['en_U','en_p','en_B','en_fv','en_fB','en_fv_lost','en_fB_lost'])
        self.add_scalar("en_tot", summands=["en_U", "en_p", "en_B", "en_fv", "en_fB"])
        self.add_scalar("n_lost_particles", compute="from_particles", species="energetic_ions")

        # temporaries
        self._b_full1 = self._b_eq.space.zeros()
        self._PBb = self._absB0.space.zeros()

        self._en_fv = np.empty(1, dtype=float)
        self._en_fB = np.empty(1, dtype=float)
        # self._en_fv_lost = np.empty(1, dtype=float)
        # self._en_fB_lost = np.empty(1, dtype=float)
        self._n_lost_particles = np.empty(1, dtype=float)

    def update_scalar_quantities(self):
        en_U = 0.5 * self.mass_ops.M2n.dot_inner(self.pointer["mhd_velocity"], self.pointer["mhd_velocity"])
        en_B = 0.5 * self.mass_ops.M2.dot_inner(self.pointer["b_field"], self.pointer["b_field"])
        en_p = self.pointer["mhd_pressure"].inner(self._ones) / (5 / 3 - 1)

        self.update_scalar("en_U", en_U)
        self.update_scalar("en_p", en_p)
        self.update_scalar("en_B", en_B)

        self._en_fv[0] = (
            self.pointer["energetic_ions"]
            .markers[~self.pointer["energetic_ions"].holes, 5]
            .dot(
                self.pointer["energetic_ions"].markers[~self.pointer["energetic_ions"].holes, 3] ** 2,
            )
            / (2.0)
            * self._coupling_params["Ah"]
            / self._coupling_params["Ab"]
        )

        self.update_scalar("en_fv", self._en_fv[0])

        # self._en_fv_lost[0] = self.pointer['energetic_ions'].lost_markers[:self.pointer['energetic_ions'].n_lost_markers, 5].dot(
        #     self.pointer['energetic_ions'].lost_markers[:self.pointer['energetic_ions'].n_lost_markers, 3]**2) / (2.0) * self._coupling_params['Ah']/self._coupling_params['Ab']

        # self.update_scalar('en_fv_lost', self._en_fv_lost[0])

        # calculate particle magnetic energy
        self.pointer["energetic_ions"].save_magnetic_energy(
            self.pointer["b_field"],
        )

        self._en_fB[0] = (
            self.pointer["energetic_ions"]
            .markers[~self.pointer["energetic_ions"].holes, 5]
            .dot(
                self.pointer["energetic_ions"].markers[~self.pointer["energetic_ions"].holes, 8],
            )
            * self._coupling_params["Ah"]
            / self._coupling_params["Ab"]
        )

        self.update_scalar("en_fB", self._en_fB[0])

        # self._en_fB_lost[0] = self.pointer['energetic_ions'].lost_markers[:self.pointer['energetic_ions'].n_lost_markers, 5].dot(
        #     self.pointer['energetic_ions']  .lost_markers[:self.pointer['energetic_ions'].n_lost_markers, 8]) * self._coupling_params['Ah']/self._coupling_params['Ab']

        # self.update_scalar('en_fB_lost', self._en_fB_lost[0])

        self.update_scalar("en_tot")

        # Print number of lost ions
        self._n_lost_particles[0] = self.pointer["energetic_ions"].n_lost_markers
        self.update_scalar("n_lost_particles", self._n_lost_particles[0])
        if self.rank_world == 0:
            print(
                "ratio of lost particles: ",
                self._n_lost_particles[0] / self.pointer["energetic_ions"].Np * 100,
                "%",
            )

    @staticmethod
    def diagnostics_dct():
        dct = {}

        dct["accumulated_magnetization"] = "Hdiv"
        return dct

    __diagnostics__ = diagnostics_dct()


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
        \frac{\alpha^2}{\varepsilon_\textnormal{c}} \left( \mathbf j_\textnormal{c} + \nu  \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v} \right) \,,

    where :math:`(n_0,\mathbf B_0)` denotes a (inhomogeneous) background and

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p,cold}}{\hat \Omega_\textnormal{c,cold}}\,, \qquad \varepsilon_\textnormal{c} = \frac{1}{\hat \Omega_\textnormal{c,cold} \hat t}\,, \qquad \varepsilon_\textnormal{h} = \frac{1}{\hat \Omega_\textnormal{c,hot} \hat t} \,, \qquad \nu = \frac{Z_\textnormal{h}}{Z_\textnormal{c}}\,.

    At initial time the Poisson equation is solved once to weakly satisfy the Gauss law:

    .. math::

        \begin{align}
            \nabla \cdot \mathbf{E} & = \nu \frac{\alpha^2}{\varepsilon_\textnormal{c}} \int_{\mathbb{R}^3} f \, \text{d}^3 \mathbf{v}\,.
        \end{align}

    Note
    ----------
    If hot and cold particles are of the same species (:math:`Z_\textnormal{c} = Z_\textnormal{h} \,, A_\textnormal{c} = A_\textnormal{h}`) then :math:`\varepsilon_\textnormal{c} = \varepsilon_\textnormal{h}` and :math:`\nu = 1`.


    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.Maxwell`
    2. :class:`~struphy.propagators.propagators_fields.OhmCold`
    3. :class:`~struphy.propagators.propagators_fields.JxBCold`
    4. :class:`~struphy.propagators.propagators_markers.PushVxB`
    5. :class:`~struphy.propagators.propagators_markers.PushEta`
    6. :class:`~struphy.propagators.propagators_coupling.VlasovAmpere`

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["em_fields"]["e_field"] = "Hcurl"
        dct["em_fields"]["b_field"] = "Hdiv"
        dct["fluid"]["cold_electrons"] = {"j": "Hcurl"}
        dct["kinetic"]["hot_electrons"] = "Particles6D"
        return dct

    @staticmethod
    def bulk_species():
        return "cold_electrons"

    @staticmethod
    def velocity_scale():
        return "light"

    @staticmethod
    def propagators_dct():
        return {
            propagators_fields.Maxwell: ["e_field", "b_field"],
            propagators_fields.OhmCold: ["cold_electrons_j", "e_field"],
            propagators_fields.JxBCold: ["cold_electrons_j"],
            propagators_markers.PushEta: ["hot_electrons"],
            propagators_markers.PushVxB: ["hot_electrons"],
            propagators_coupling.VlasovAmpere: ["e_field", "hot_electrons"],
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
            species=["em_fields"],
            option=propagators_fields.ImplicitDiffusion,
            dct=dct,
        )
        return dct

    def __init__(self, params, comm, clone_config=None):
        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        # Get rank and size
        self._rank = self.rank_world

        # prelim
        hot_params = params["kinetic"]["hot_electrons"]

        # model parameters
        self._alpha = np.abs(
            self.equation_params["cold_electrons"]["alpha"],
        )
        self._epsilon_cold = self.equation_params["cold_electrons"]["epsilon"]
        self._epsilon_hot = self.equation_params["hot_electrons"]["epsilon"]

        self._nu = hot_params["phys_params"]["Z"] / params["fluid"]["cold_electrons"]["phys_params"]["Z"]

        # Initialize background magnetic field from MHD equilibrium
        self._b_background = self.derham.P["2"](
            [
                self.equil.b2_1,
                self.equil.b2_2,
                self.equil.b2_3,
            ]
        )

        # propagator parameters
        params_maxwell = params["em_fields"]["options"]["Maxwell"]["solver"]
        params_ohmcold = params["fluid"]["cold_electrons"]["options"]["OhmCold"]["solver"]
        params_jxbcold = params["fluid"]["cold_electrons"]["options"]["JxBCold"]["solver"]
        algo_eta = params["kinetic"]["hot_electrons"]["options"]["PushEta"]["algo"]
        algo_vxb = params["kinetic"]["hot_electrons"]["options"]["PushVxB"]["algo"]
        params_coupling = params["em_fields"]["options"]["VlasovAmpere"]["solver"]
        self._poisson_params = params["em_fields"]["options"]["ImplicitDiffusion"]["solver"]

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.Maxwell] = {"solver": params_maxwell}

        self._kwargs[propagators_fields.OhmCold] = {
            "alpha": self._alpha,
            "epsilon": self._epsilon_cold,
            "solver": params_ohmcold,
        }

        self._kwargs[propagators_fields.JxBCold] = {
            "epsilon": self._epsilon_cold,
            "solver": params_jxbcold,
        }

        self._kwargs[propagators_markers.PushEta] = {"algo": algo_eta}

        self._kwargs[propagators_markers.PushVxB] = {
            "algo": algo_vxb,
            "kappa": 1.0 / self._epsilon_cold,
            "b2": self.pointer["b_field"],
            "b2_add": self._b_background,
        }

        self._kwargs[propagators_coupling.VlasovAmpere] = {
            "c1": self._nu * self._alpha**2 / self._epsilon_cold,
            "c2": 1.0 / self._epsilon_hot,
            "solver": params_coupling,
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during simulation
        self.add_scalar("en_E")
        self.add_scalar("en_B")
        self.add_scalar("en_J")
        self.add_scalar("en_f", compute="from_particles", species="hot_electrons")
        self.add_scalar("en_tot")

        # temporaries
        self._tmp = np.empty(1, dtype=float)

    def initialize_from_params(self):
        """:meta private:"""
        from psydac.linalg.stencil import StencilVector

        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector

        # Initialize fields and particles
        super().initialize_from_params()

        # Accumulate charge density
        charge_accum = AccumulatorVector(
            self.pointer["hot_electrons"],
            "H1",
            Pyccelkernel(accum_kernels.vlasov_maxwell_poisson),
            self.mass_ops,
            self.domain.args_domain,
        )
        charge_accum()

        # Locally subtract mean charge for solvability with periodic bc
        if np.all(charge_accum.vectors[0].space.periods):
            charge_accum._vectors[0][:] -= np.mean(
                charge_accum.vectors[0].toarray()[charge_accum.vectors[0].toarray() != 0],
            )

        # Instantiate Poisson solver
        _phi = StencilVector(self.derham.Vh["0"])
        poisson_solver = propagators_fields.ImplicitDiffusion(
            _phi,
            sigma_1=0,
            rho=self._nu * self._alpha**2 / self._epsilon_cold * charge_accum.vectors[0],
            x0=self._nu * self._alpha**2 / self._epsilon_cold * charge_accum.vectors[0],
            solver=self._poisson_params,
        )

        # Solve with dt=1. and compute electric field
        poisson_solver(1.0)
        self.derham.grad.dot(-_phi, out=self.pointer["e_field"])

    def update_scalar_quantities(self):
        en_E = 0.5 * self.mass_ops.M1.dot_inner(self.pointer["e_field"], self.pointer["e_field"])
        en_B = 0.5 * self.mass_ops.M2.dot_inner(self.pointer["b_field"], self.pointer["b_field"])
        en_J = (
            0.5
            * self._alpha**2
            * self.mass_ops.M1ninv.dot_inner(self.pointer["cold_electrons_j"], self.pointer["cold_electrons_j"])
        )
        self.update_scalar("en_E", en_E)
        self.update_scalar("en_B", en_B)
        self.update_scalar("en_J", en_J)

        # nu alpha^2 eps_h / eps_c / 2 / N * sum_p w_p v_p^2
        self._tmp[0] = (
            self._nu
            * self._alpha**2
            * self._epsilon_hot
            / self._epsilon_cold
            / (2 * self.pointer["hot_electrons"].Np)
            * np.dot(
                self.pointer["hot_electrons"].markers_wo_holes[:, 3] ** 2
                + self.pointer["hot_electrons"].markers_wo_holes[:, 4] ** 2
                + self.pointer["hot_electrons"].markers_wo_holes[:, 5] ** 2,
                self.pointer["hot_electrons"].markers_wo_holes[:, 6],
            )
        )

        self.update_scalar("en_f", self._tmp[0])

        # en_tot = en_E + en_B + en_J + en_w
        self.update_scalar("en_tot", en_E + en_B + en_J + self._tmp[0])
