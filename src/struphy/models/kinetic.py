import numpy as np
from mpi4py import MPI

from struphy.kinetic_background.base import KineticBackground
from struphy.models.base import StruphyModel
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc
from struphy.propagators import propagators_coupling, propagators_fields, propagators_markers
from struphy.models.species import KineticSpecies, FluidSpecies, FieldSpecies
from struphy.models.variables import Variable, FEECVariable, PICVariable, SPHVariable
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector

rank = MPI.COMM_WORLD.Get_rank()


class VlasovAmpereOneSpecies(StruphyModel):
    r"""Vlasov-Ampère equations for one species.

    :ref:`normalization`:

    .. math::

        \begin{align}
            \hat v  = c \,, \qquad \hat E = \hat B \hat v\,,\qquad  \hat \phi = \hat E \hat x \,.
        \end{align}

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B}_0 \right)
            \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
        \\[2mm]
        -&\frac{\partial \mathbf{E}}{\partial t} =
        \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v}\,,

    with the normalization parameter

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 (A m_\textnormal{H})}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{(Ze) \hat B}{(A m_\textnormal{H})}\,,

    where :math:`Z=-1` and :math:`A=1/1836` for electrons.
    At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

    .. math::

            \begin{align}
            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \,\textrm d \mathbf x &= \frac{\alpha^2}{\varepsilon}  \int_\Omega \int_{\mathbb{R}^3} \psi\, (f - f_0) \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,
            \\[2mm]
            \mathbf{E}(t=0) &= -\nabla \phi(t=0)\,.
            \end{align}

    Moreover, it is assumed that

    .. math::

        \nabla \times \mathbf B_0 = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v}\,,

    where :math:`\mathbf B_0` is the static equilibirum magnetic field.

    Notes
    -----

    * The :ref:`control_var` for Ampère's law is optional; in case it is enabled via the parameter file, the following system is solved:
    Find :math:`(\mathbf E, f) \in H(\textnormal{curl}) \times C^\infty` such that

    .. math::

        \begin{align}
            -\int_\Omega \mathbf F\, \cdot \, &\frac{\partial \mathbf{E}}{\partial t}\,\textrm d \mathbf x =
            \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} \mathbf F \cdot \mathbf{v} (f - f_0) \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \mathbf F \in H(\textnormal{curl}) \,,
            \\[2mm]
            &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B}_0 \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,.
        \end{align}


    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushEta`
    2. :class:`~struphy.propagators.propagators_coupling.VlasovAmpere`
    3. :class:`~struphy.propagators.propagators_markers.PushVxB`
    """

    ## species
    
    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.phi = FEECVariable(space="H1")
            self.init_variables()
    
    class KineticIons(KineticSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles6D")
            self.init_variables()
        
    ## propagators
    
    class Propagators:
        def __init__(self):
            self.push_eta = propagators_markers.PushEta()
            self.push_vxb = propagators_markers.PushVxB()
            self.coupling_va = propagators_coupling.VlasovAmpere()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")
            
        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()
        
        # 3. assign variables to propagators
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        self.propagators.push_vxb.variables.ions = self.kinetic_ions.var
        self.propagators.coupling_va.variables.e = self.em_fields.e_field
        self.propagators.coupling_va.variables.ions = self.kinetic_ions.var
        
        # define scalars for update_scalar_quantities
        self.add_scalar("en_E")
        self.add_scalar("en_f", compute="from_particles", variable=self.kinetic_ions.var)
        self.add_scalar("en_tot")
        
        # initial Poisson (not a propagator used in time stepping)
        self.initial_poisson = propagators_fields.Poisson()
        self.initial_poisson.variables.phi = self.em_fields.phi

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self):
        self._tmp = np.empty(1, dtype=float)

    def update_scalar_quantities(self):
        # e*M1*e/2
        e = self.em_fields.e_field.spline.vector
        en_E = 0.5 * self.mass_ops.M1.dot_inner(e, e)
        self.update_scalar("en_E", en_E)

        # alpha^2 / 2 / N * sum_p w_p v_p^2
        particles = self.kinetic_ions.var.particles
        alpha = self.kinetic_ions.equation_params.alpha
        self._tmp[0] = (
            alpha**2
            / (2 * particles.Np)
            * np.dot(
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
        particles = self.kinetic_ions.var.particles
        particles.update_weights()

        # sanity check
        # self.pointer['species1'].show_distribution_function(
        #     [True] + [False]*5, [np.linspace(0, 1, 32)])

        # accumulate charge density
        charge_accum = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels.charge_density_0form),
            self.mass_ops,
            self.domain.args_domain,
        )

        charge_accum(particles.vdim)

        # another sanity check: compute FE coeffs of density
        # charge_accum.show_accumulated_spline_field(self.mass_ops)
        
        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon
        
        self.initial_poisson.options.rho = alpha**2 / epsilon * charge_accum.vectors[0]
        # self.initial_poisson.variables.phi.allocate(self.derham, domain=self.domain)
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
    def generate_default_parameter_file(self, path = None, prompt = True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "coupling_va.Options" in line:
                    new_file += [line]
                    new_file += ["model.initial_poisson.options = model.initial_poisson.Options()\n"]
                else:
                    new_file += [line]
                    
        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)

class VlasovMaxwellOneSpecies(StruphyModel):
    r"""Vlasov-Maxwell equations for one species.

    :ref:`normalization`:

    .. math::

        \begin{align}
            \hat v  = c \,, \qquad \hat E = \hat B \hat v\,,\qquad  \hat \phi = \hat E \hat x \,.
        \end{align}

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon} \left( \mathbf{E} + \mathbf{v} \times \left( \mathbf{B} + \mathbf{B}_0 \right) \right)
        \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
        \\[2mm]
        -&\frac{\partial \mathbf{E}}{\partial t} + \nabla \times \mathbf B =
        \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3}  \mathbf{v} f \, \text{d}^3 \mathbf{v}\,,
        \\[2mm]
        &\frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0 \,,

    with the normalization parameters

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 (A m_\textnormal{H})}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{(Ze) \hat B}{(A m_\textnormal{H})}\,,

    where :math:`Z=-1` and :math:`A=1/1836` for electrons.
    At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

    .. math::

            \begin{align}
            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \,\textrm d \mathbf x &= \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} \psi\, (f - f_0) \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,
            \\[2mm]
            \mathbf{E}(t=0) &= -\nabla \phi(t=0)\,.
            \end{align}

    Moreover, it is assumed that

    .. math::

        \nabla \times \mathbf B_0 = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v}\,,

    where :math:`\mathbf B_0` is the static equilibirum magnetic field.

    Notes
    -----

    * The :ref:`control_var` for Ampère's law is optional; in case it is enabled via the parameter file, the following system is solved:
    Find :math:`(\mathbf E, \tilde{\mathbf B}, f) \in H(\textnormal{curl}) \times H(\textnormal{div}) \times C^\infty` such that

    .. math::

        \begin{align}
            -\int_\Omega \mathbf F\, \cdot \, &\frac{\partial \mathbf{E}}{\partial t}\,\textrm d \mathbf x + \int_\Omega \nabla \times \mathbf{F} \cdot \tilde{\mathbf B}\,\textrm d \mathbf x =
            \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} \mathbf F \cdot \mathbf{v} (f - f_0) \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \mathbf F \in H(\textnormal{curl}) \,,
            \\[2mm]
            &\frac{\partial \tilde{\mathbf B}}{\partial t} + \nabla \times \mathbf{E} = 0 \,,
            \\[2mm]
            &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon}\Big[ \mathbf{E} + \mathbf{v} \times (\mathbf{B}_0 + \tilde{\mathbf B}) \Big]
            \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
        \end{align}

    where :math:`\tilde{\mathbf B} = \mathbf B - \mathbf B_0` denotes the magnetic perturbation.


    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.Maxwell`
    2. :class:`~struphy.propagators.propagators_markers.PushEta`
    3. :class:`~struphy.propagators.propagators_markers.PushVxB`
    4. :class:`~struphy.propagators.propagators_coupling.VlasovAmpere`

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["em_fields"]["e_field"] = "Hcurl"
        dct["em_fields"]["b_field"] = "Hdiv"
        dct["kinetic"]["species1"] = "Particles6D"
        return dct

    @staticmethod
    def bulk_species():
        return "species1"

    @staticmethod
    def velocity_scale():
        return "light"

    @staticmethod
    def propagators_dct():
        return {
            propagators_fields.Maxwell: ["e_field", "b_field"],
            propagators_markers.PushEta: ["species1"],
            propagators_markers.PushVxB: ["species1"],
            propagators_coupling.VlasovAmpere: ["e_field", "species1"],
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
        cls.add_option(
            species=["kinetic", "species1"],
            key="override_eq_params",
            option=[False, {"alpha": 1.0, "epsilon": -1.0}],
            dct=dct,
        )
        return dct

    def __init__(self, params, comm, clone_config=None):
        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        # get species paramaters
        species1_params = params["kinetic"]["species1"]

        # equation parameters
        if species1_params["options"]["override_eq_params"]:
            self._alpha = species1_params["options"]["override_eq_params"]["alpha"]
            self._epsilon = species1_params["options"]["override_eq_params"]["epsilon"]
            print(
                f"\n!!! Override equation parameters: {self._alpha = } and {self._epsilon = }.",
            )
        else:
            self._alpha = self.equation_params["species1"]["alpha"]
            self._epsilon = self.equation_params["species1"]["epsilon"]

        # set background density and mean velocity factors
        self.pointer["species1"].f0.moment_factors["u"] = [
            self._epsilon / self._alpha**2,
        ] * 3

        # Initialize background magnetic field from MHD equilibrium
        if self.projected_equil:
            self._b_background = self.projected_equil.b2
        else:
            self._b_background = None

        # propagator parameters
        params_maxwell = params["em_fields"]["options"]["Maxwell"]["solver"]
        algo_eta = params["kinetic"]["species1"]["options"]["PushEta"]["algo"]
        algo_vxb = params["kinetic"]["species1"]["options"]["PushVxB"]["algo"]
        params_coupling = params["em_fields"]["options"]["VlasovAmpere"]["solver"]
        self._poisson_params = params["em_fields"]["options"]["ImplicitDiffusion"]["solver"]

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.Maxwell] = {"solver": params_maxwell}

        self._kwargs[propagators_markers.PushEta] = {"algo": algo_eta}

        self._kwargs[propagators_markers.PushVxB] = {
            "algo": algo_vxb,
            "kappa": 1.0 / self._epsilon,
            "b2": self.pointer["b_field"],
            "b2_add": self._b_background,
        }

        self._kwargs[propagators_coupling.VlasovAmpere] = {
            "c1": self._alpha**2 / self._epsilon,
            "c2": 1.0 / self._epsilon,
            "solver": params_coupling,
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # Scalar variables to be saved during the simulation
        self.add_scalar("en_E")
        self.add_scalar("en_B")
        self.add_scalar("en_f", compute="from_particles", species="species1")
        self.add_scalar("en_tot")

        # temporaries
        self._tmp = np.empty(1, dtype=float)

    def initialize_from_params(self):
        """:meta private:"""

        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector

        # initialize fields and particles
        super().initialize_from_params()

        if self.rank_world == 0:
            print("\nINITIAL POISSON SOLVE:")

        # use control variate method
        self.pointer["species1"].update_weights()

        # sanity check
        # self.pointer['species1'].show_distribution_function(
        #     [True] + [False]*5, [np.linspace(0, 1, 32)])

        # accumulate charge density
        charge_accum = AccumulatorVector(
            self.pointer["species1"],
            "H1",
            Pyccelkernel(accum_kernels.charge_density_0form),
            self.mass_ops,
            self.domain.args_domain,
        )

        charge_accum(self.pointer["species1"].vdim)

        # another sanity check: compute FE coeffs of density
        # charge_accum.show_accumulated_spline_field(self.mass_ops)

        # Instantiate Poisson solver
        _phi = self.derham.Vh["0"].zeros()
        poisson_solver = propagators_fields.ImplicitDiffusion(
            _phi,
            sigma_1=0.0,
            sigma_2=0.0,
            sigma_3=1.0,
            rho=self._alpha**2 / self._epsilon * charge_accum.vectors[0],
            solver=self._poisson_params,
        )

        # Solve with dt=1. and compute electric field
        if self.rank_world == 0:
            print("\nSolving initial Poisson problem...")
        poisson_solver(1.0)

        self.derham.grad.dot(-_phi, out=self.pointer["e_field"])
        if self.rank_world == 0:
            print("Done.")

    def update_scalar_quantities(self):
        # e*M1*e and b*M2*b
        en_E = 0.5 * self.mass_ops.M1.dot_inner(self.pointer["e_field"], self.pointer["e_field"])
        en_B = 0.5 * self.mass_ops.M2.dot_inner(self.pointer["b_field"], self.pointer["b_field"])
        self.update_scalar("en_E", en_E)
        self.update_scalar("en_B", en_B)

        # alpha^2 / 2 / N * sum_p w_p v_p^2
        self._tmp[0] = (
            self._alpha**2
            / (2 * self.pointer["species1"].Np)
            * np.dot(
                self.pointer["species1"].markers_wo_holes[:, 3] ** 2
                + self.pointer["species1"].markers_wo_holes[:, 4] ** 2
                + self.pointer["species1"].markers_wo_holes[:, 5] ** 2,
                self.pointer["species1"].markers_wo_holes[:, 6],
            )
        )

        self.update_scalar("en_f", self._tmp[0])

        # en_tot = en_w + en_e + en_b
        self.update_scalar("en_tot", en_E + en_B + self._tmp[0])


class LinearVlasovAmpereOneSpecies(StruphyModel):
    r"""Linearized Vlasov-Ampère equations for one species.

    :ref:`normalization`:

    .. math::

        \begin{align}
            \hat v  = c \,, \qquad \hat E = \hat B \hat v\,,\qquad  \hat \phi = \hat E \hat x \,.
        \end{align}

    :ref:`Equations <gempic>`:

    .. math::

        \begin{align}
            & \frac{\partial \tilde{\mathbf E}}{\partial t} = - \frac{\alpha^2}{\varepsilon} \int_{\mathbb R^3} \mathbf{v} \tilde f\, \textrm d^3 \mathbf v \,,
            \\[2mm]
            & \frac{\partial \tilde f}{\partial t} + \mathbf{v} \cdot \, \nabla \tilde f + \frac{1}{\varepsilon} \left( \mathbf{E}_0 + \mathbf{v} \times \mathbf{B}_0 \right)
            \cdot \frac{\partial \tilde f}{\partial \mathbf{v}} = \frac{1}{v_{\text{th}}^2 \varepsilon} \, \tilde{\mathbf E} \cdot \mathbf{v} f_0 \,,
        \end{align}

    with the normalization parameter

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 (A m_\textnormal{H})}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{(Ze) \hat B}{(A m_\textnormal{H})}\,,

    where :math:`Z=-1` and :math:`A=1/1836` for electrons. The background distribution function :math:`f_0` is a uniform Maxwellian

    .. math::

        f_0 = \frac{n_0(\mathbf{x})}{\left( \sqrt{2 \pi} v_{\text{th}} \right)^3}
        \exp \left( - \frac{|\mathbf{v}|^2}{2 v_{\text{th}}^2} \right) \,,

    and the background electric field has to verify the following compatibility condition between with background density

    .. math::

        \nabla_{\mathbf{x}} \ln (n_0(\mathbf{x})) = \frac{1}{v_{\text{th}}^2 \varepsilon} \mathbf{E}_0 \,.

    At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

    .. math::

            \begin{align}
            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \,\textrm d \mathbf x &= \frac{\alpha^2}{\varepsilon}  \int_\Omega \int_{\mathbb{R}^3} \psi\, \tilde f \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,
            \\[2mm]
            \tilde{\mathbf{E}}(t=0) &= -\nabla \phi(t=0) \,.
            \end{align}

    Moreover, it is assumed that

    .. math::

        \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v} = 0 \,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushEta`
    2. :class:`~struphy.propagators.propagators_markers.PushVinEfield`
    3. :class:`~struphy.propagators.propagators_coupling.EfieldWeights`
    4. :class:`~struphy.propagators.propagators_markers.PushVxB`

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["em_fields"]["e_field"] = "Hcurl"
        dct["kinetic"]["species1"] = "DeltaFParticles6D"
        return dct

    @staticmethod
    def bulk_species():
        return "species1"

    @staticmethod
    def velocity_scale():
        return "light"

    @staticmethod
    def propagators_dct():
        return {
            propagators_markers.PushEta: ["species1"],
            propagators_markers.PushVinEfield: ["species1"],
            propagators_coupling.EfieldWeights: ["e_field", "species1"],
            propagators_markers.PushVxB: ["species1"],
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    @classmethod
    def options(cls):
        dct = super().options()
        cls.add_option(
            species=["em_fields"],
            option=propagators_fields.ImplicitDiffusion,
            dct=dct,
        )
        cls.add_option(
            species=["kinetic", "species1"],
            key="override_eq_params",
            option=[False, {"epsilon": -1.0, "alpha": 1.0}],
            dct=dct,
        )
        return dct

    def __init__(self, params, comm, clone_config=None, baseclass=False):
        """Initializes the model either as the full model or as a baseclass to inherit from.
        In case of being a baseclass, the propagators will not be initialized in the __init__ which allows other propagators to be added.

        Parameters
        ----------
        baseclass : Boolean [optional]
            If this model should be used as a baseclass. Default value is False.
        """

        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        from struphy.kinetic_background import maxwellians

        # if model is used as a baseclass
        self._baseclass = baseclass

        # kinetic parameters
        self._species_params = params["kinetic"]["species1"]

        # Assert Maxwellian background (if list, the first entry is taken)
        bckgr_params = self._species_params["background"]
        li_bp = list(bckgr_params)
        assert li_bp[0] == "Maxwellian3D", "The background distribution function must be a uniform Maxwellian!"
        if len(li_bp) > 1:
            # overwrite f0 with single Maxwellian
            self._f0 = getattr(maxwellians, li_bp[0][:-2])(
                maxw_params=bckgr_params[li_bp[0]],
            )
        else:
            # keep allocated background
            self._f0 = self.pointer["species1"].f0

        # Assert uniformity of the Maxwellian background
        assert self._f0.maxw_params["u1"] == 0.0, "The background Maxwellian cannot have shifts in velocity space!"
        assert self._f0.maxw_params["u2"] == 0.0, "The background Maxwellian cannot have shifts in velocity space!"
        assert self._f0.maxw_params["u3"] == 0.0, "The background Maxwellian cannot have shifts in velocity space!"
        assert self._f0.maxw_params["vth1"] == self._f0.maxw_params["vth2"] == self._f0.maxw_params["vth3"], (
            "The background Maxwellian must be isotropic in velocity space!"
        )
        self.vth = self._f0.maxw_params["vth1"]

        # Get coupling strength
        if self._species_params["options"]["override_eq_params"]:
            self.epsilon = self._species_params["options"]["override_eq_params"]["epsilon"]
            self.alpha = self._species_params["options"]["override_eq_params"]["alpha"]
            if self.rank_world == 0:
                print(
                    f"\n!!! Override equation parameters: {self.epsilon = }, {self.alpha = }.\n",
                )
        else:
            self.epsilon = self.equation_params["species1"]["epsilon"]
            self.alpha = self.equation_params["species1"]["alpha"]

        # allocate memory for evaluating f0 in energy computation
        self._f0_values = np.zeros(
            self.pointer["species1"].markers.shape[0],
            dtype=float,
        )

        # ====================================================================================
        # Create pointers to background electric potential and field
        self._has_background_e = False
        if "external_E0" in self.params["em_fields"]["options"].keys():
            e0 = self.params["em_fields"]["options"]["external_E0"]
            if e0 != 0.0:
                self._has_background_e = True
                self._e_background = self.derham.Vh["1"].zeros()
                for block in self._e_background._blocks:
                    block._data[:, :, :] += e0

        # Get parameters of the background magnetic field
        if self.projected_equil:
            self._b_background = self.projected_equil.b2
        else:
            self._b_background = None
        # ====================================================================================

        # propagator parameters
        self._poisson_params = params["em_fields"]["options"]["ImplicitDiffusion"]["solver"]
        algo_eta = params["kinetic"]["species1"]["options"]["PushEta"]["algo"]
        params_coupling = params["em_fields"]["options"]["EfieldWeights"]["solver"]

        # Initialize propagators/integrators used in splitting substeps
        self._kwargs[propagators_markers.PushEta] = {
            "algo": algo_eta,
        }

        # Only add PushVinEfield if e-field is non-zero, otherwise it is more expensive
        if self._has_background_e:
            self._kwargs[propagators_markers.PushVinEfield] = {
                "e_field": self._e_background,
                "kappa": 1.0 / self.epsilon,
            }
        else:
            self._kwargs[propagators_markers.PushVinEfield] = None

        self._kwargs[propagators_coupling.EfieldWeights] = {
            "alpha": self.alpha,
            "kappa": 1.0 / self.epsilon,
            "f0": self._f0,
            "solver": params_coupling,
        }

        # Only add PushVxB if magnetic field is not zero
        self._kwargs[propagators_markers.PushVxB] = None
        if self._b_background:
            self._kwargs[propagators_markers.PushVxB] = {
                "kappa": 1.0 / self.epsilon,
                "b2": self._b_background,
            }

        # Initialize propagators used in splitting substeps
        if not self._baseclass:
            self.init_propagators()

        # Scalar variables to be saved during the simulation
        self.add_scalar("en_E")
        self.add_scalar("en_w", compute="from_particles", species="species1")
        self.add_scalar("en_tot")

        # temporaries
        self._tmp = np.empty(1, dtype=float)
        self.en_E = 0.0

    def initialize_from_params(self):
        """Solve initial Poisson equation.

        :meta private:
        """
        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector

        # Initialize fields and particles
        super().initialize_from_params()

        # Accumulate charge density
        charge_accum = AccumulatorVector(
            self.pointer["species1"],
            "H1",
            Pyccelkernel(accum_kernels.charge_density_0form),
            self.mass_ops,
            self.domain.args_domain,
        )

        charge_accum(self.pointer["species1"].vdim)

        # Instantiate Poisson solver
        _phi = self.derham.Vh["0"].zeros()
        poisson_solver = propagators_fields.ImplicitDiffusion(
            _phi,
            sigma_1=0.0,
            sigma_2=0.0,
            sigma_3=1.0,
            rho=self.alpha**2 / self.epsilon * charge_accum.vectors[0],
            solver=self._poisson_params,
        )

        # Solve with dt=1. and compute electric field
        if self.rank_world == 0:
            print("\nSolving initial Poisson problem...")
        poisson_solver(1.0)
        self.derham.grad.dot(-_phi, out=self.pointer["e_field"])
        if self.rank_world == 0:
            print("Done.")

    def update_scalar_quantities(self):
        # 0.5 * e^T * M_1 * e
        self.en_E = 0.5 * self.mass_ops.M1.dot_inner(self.pointer["e_field"], self.pointer["e_field"])
        self.update_scalar("en_E", self.en_E)

        # evaluate f0
        self._f0_values[self.pointer["species1"].valid_mks] = self._f0(*self.pointer["species1"].phasespace_coords.T)

        # alpha^2 * v_th^2 / (2*N) * sum_p s_0 * w_p^2 / f_{0,p}
        self._tmp[0] = (
            self.alpha**2
            * self.vth**2
            / (2 * self.pointer["species1"].Np)
            * np.dot(
                self.pointer["species1"].weights ** 2,  # w_p^2
                self.pointer["species1"].sampling_density
                / self._f0_values[self.pointer["species1"].valid_mks],  # s_{0,p} / f_{0,p}
            )
        )

        self.update_scalar("en_w", self._tmp[0])

        # en_tot = en_w + en_e
        if not self._baseclass:
            self.update_scalar("en_tot", self._tmp[0] + self.en_E)


class LinearVlasovMaxwellOneSpecies(LinearVlasovAmpereOneSpecies):
    r"""Linearized Vlasov-Ampère equations for one species.

    :ref:`normalization`:

    .. math::

        \begin{align}
            \hat v  = c \,, \qquad \hat E = \hat B \hat v\,,\qquad  \hat \phi = \hat E \hat x \,.
        \end{align}

    :ref:`Equations <gempic>`:

    .. math::

        \begin{align}
            & \frac{\partial \tilde{\mathbf E}}{\partial t} = \nabla \times \tilde{\mathbf B} - \frac{\alpha^2}{\varepsilon} \int_{\mathbb R^3}\mathbf{v} \tilde f\, \textrm d^3 \mathbf v \,,
            \\[2mm]
            & \frac{\partial \tilde{\mathbf B}}{\partial t} = - \nabla \times \tilde{\mathbf E} \,,
            \\[2mm]
            & \frac{\partial \tilde f}{\partial t} + \mathbf{v} \cdot \, \nabla \tilde f + \frac{1}{\varepsilon} \left( \mathbf{E}_0 + \mathbf{v} \times \mathbf{B}_0 \right)
            \cdot \frac{\partial \tilde f}{\partial \mathbf{v}} = \frac{1}{v_{\text{th}}^2 \varepsilon} \, \tilde{\mathbf E} \cdot \mathbf{v} f_0 \,,
        \end{align}

    with the normalization parameter

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,,\qquad \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t} \,,\qquad \textnormal{with} \qquad \hat\Omega_\textnormal{p} = \sqrt{\frac{\hat n (Ze)^2}{\epsilon_0 (A m_\textnormal{H})}} \,,\qquad \hat \Omega_{\textnormal{c}} = \frac{(Ze) \hat B}{(A m_\textnormal{H})}\,,

    where :math:`Z=-1` and :math:`A=1/1836` for electrons. The background distribution function :math:`f_0` is a uniform Maxwellian

    .. math::

        f_0 = \frac{n_0(\mathbf{x})}{\left( \sqrt{2 \pi} v_{\text{th}} \right)^3}
        \exp \left( - \frac{|\mathbf{v}|^2}{2 v_{\text{th}}^2} \right) \,,

    and the background electric field has to verify the following compatibility condition between with background density

    .. math::

        \nabla_{\mathbf{x}} \ln (n_0(\mathbf{x})) = \frac{1}{v_{\text{th}}^2 \varepsilon} \mathbf{E}_0 \,.

    At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

    .. math::

            \begin{align}
            \int_\Omega \nabla \psi^\top \cdot \nabla \phi \,\textrm d \mathbf x &= \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} \psi\, \tilde f \, \text{d}^3 \mathbf{v}\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,
            \\[2mm]
            \tilde{\mathbf{E}(t=0)} &= -\nabla \phi(t=0) \,.
            \end{align}

    Moreover, it is assumed that

    .. math::

        \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v} = 0 \,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushEta`
    2. :class:`~struphy.propagators.propagators_markers.PushVinEfield`
    3. :class:`~struphy.propagators.propagators_coupling.EfieldWeights`
    4. :class:`~struphy.propagators.propagators_markers.PushVxB`
    5. :class:`~struphy.propagators.propagators_fields.Maxwell`

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["em_fields"]["e_field"] = "Hcurl"
        dct["em_fields"]["b_field"] = "Hdiv"
        dct["kinetic"]["species1"] = "DeltaFParticles6D"
        return dct

    @staticmethod
    def bulk_species():
        return "species1"

    @staticmethod
    def velocity_scale():
        return "light"

    @staticmethod
    def propagators_dct():
        return {
            propagators_markers.PushEta: ["species1"],
            propagators_markers.PushVinEfield: ["species1"],
            propagators_coupling.EfieldWeights: ["e_field", "species1"],
            propagators_markers.PushVxB: ["species1"],
            propagators_fields.Maxwell: ["e_field", "b_field"],
        }

    __em_fields__ = species()["em_fields"]
    __fluid_species__ = species()["fluid"]
    __kinetic_species__ = species()["kinetic"]
    __bulk_species__ = bulk_species()
    __velocity_scale__ = velocity_scale()
    __propagators__ = [prop.__name__ for prop in propagators_dct()]

    @classmethod
    def options(cls):
        dct = super().options()
        cls.add_option(
            species=["em_fields"],
            option=propagators_fields.ImplicitDiffusion,
            dct=dct,
        )
        cls.add_option(
            species=["kinetic", "species1"],
            key="override_eq_params",
            option=[False, {"epsilon": -1.0, "alpha": 1.0}],
            dct=dct,
        )
        return dct

    def __init__(self, params, comm, clone_config=None):
        super().__init__(params=params, comm=comm, clone_config=clone_config, baseclass=True)

        # propagator parameters
        params_maxwell = params["em_fields"]["options"]["Maxwell"]["solver"]

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.Maxwell] = {"solver": params_maxwell}

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # magnetic energy
        self.add_scalar("en_b")

    def initialize_from_params(self):
        super().initialize_from_params()

    def update_scalar_quantities(self):
        super().update_scalar_quantities()

        # 0.5 * b^T * M_2 * b
        en_B = 0.5 * self._mass_ops.M2.dot_inner(self.pointer["b_field"], self.pointer["b_field"])
        self.update_scalar("en_tot", self._tmp[0] + self.en_E + en_B)


class DriftKineticElectrostaticAdiabatic(StruphyModel):
    r"""Drift-kinetic equation for one ion species in static background magnetic field,
    coupled to quasi-neutrality equation with adiabatic electrons.

    :ref:`normalization`:

    .. math::

       \hat v = \hat v_\textrm{i} = \sqrt{\frac{k_B \hat T_\textrm{i}}{m_\textrm{i}}}\,,\qquad  \hat E = \hat v_\textrm{i}\hat B\,,\qquad \hat \phi = \hat E \hat x \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial f}{\partial t} + \left[ v_\parallel \frac{\mathbf{B}^*}{B^*_\parallel} + \frac{\mathbf{E}^* \times \mathbf{b}_0}{B^*_\parallel}\right] \cdot \frac{\partial f}{\partial \mathbf{X}} + \left[\frac{1}{\varepsilon} \frac{\mathbf{B}^*}{B^*_\parallel} \cdot \mathbf{E}^*\right] \cdot \frac{\partial f}{\partial v_\parallel} = 0\,.
        \\[2mm]
        - &\nabla_\perp \cdot \left( \frac{n_0}{|B_0|^2} \nabla_\perp \phi \right) + \frac{1}{\varepsilon} n_0 \left(1 + \frac{1}{Z \varepsilon} \frac{1}{T_{0}} \phi \right) = \frac 1 \varepsilon \int f B^*_\parallel \,\textnormal d v_\parallel \textnormal d \mu \,.

    where :math:`f(\mathbf{X}, v_\parallel, \mu, t)` is the guiding center distribution and

    .. math::
        \mathbf{E}^* = - \nabla \phi - \varepsilon \mu \nabla |B_0| \,,  \qquad \mathbf{B}^* = \mathbf{B}_0 + \varepsilon v_\parallel \nabla \times \mathbf{b}_0 \,,\qquad B^*_\parallel = \mathbf B^* \cdot \mathbf b_0  \,,

    and with the normalization parameters

    .. math::

        \varepsilon := \frac{1}{\hat \Omega_\textrm{c} \hat t}\,,\qquad \hat \Omega_\textrm{c} = \frac{q_\textrm{i} \hat B}{m_\textrm{i}} \,.

    Notes
    -----

    * The :ref:`control_var` in the Poisson equation is optional; in case it is enabled via the parameter file, the following Poisson equation is solved:
    Find :math:`\phi \in H^1` such that

    .. math::

        \int \frac{n_0}{|B_0|^2} \nabla_\perp \psi \cdot \nabla_\perp \phi\,\textrm d \mathbf x + \frac{1}{Z\varepsilon^2} \int  \frac{n_0}{T_{0}} \psi \phi \,\textrm d \mathbf x  = \frac 1 \varepsilon \int \int \psi \, (f - f_0) B^*_\parallel \,\textrm d \mathbf x\,\textnormal d v_\parallel \textnormal d \mu \qquad \forall \ \psi \in H^1\,.


    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.ImplicitDiffusion`
    2. :class:`~struphy.propagators.propagators_markers.PushGuidingCenterBxEstar`
    3. :class:`~struphy.propagators.propagators_markers.PushGuidingCenterParallel`

    :ref:`Model info <add_model>`:
    """

    @staticmethod
    def species():
        dct = {"em_fields": {}, "fluid": {}, "kinetic": {}}

        dct["em_fields"]["phi"] = "H1"
        dct["kinetic"]["ions"] = "Particles5D"
        return dct

    @staticmethod
    def bulk_species():
        return "ions"

    @staticmethod
    def velocity_scale():
        return "thermal"

    @staticmethod
    def propagators_dct():
        return {
            propagators_fields.ImplicitDiffusion: ["phi"],
            propagators_markers.PushGuidingCenterBxEstar: ["ions"],
            propagators_markers.PushGuidingCenterParallel: ["ions"],
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
            species=["kinetic", "ions"],
            key="override_eq_params",
            option=[False, {"epsilon": 1.0}],
            dct=dct,
        )
        return dct

    def __init__(self, params, comm, clone_config=None):
        # initialize base class
        super().__init__(params, comm=comm, clone_config=clone_config)

        from struphy.feec.projectors import L2Projector
        from struphy.pic.accumulation.particles_to_grid import AccumulatorVector

        # prelim
        solver_params = params["em_fields"]["options"]["ImplicitDiffusion"]["solver"]
        ions_params = params["kinetic"]["ions"]

        Z = ions_params["phys_params"]["Z"]
        assert Z > 0  # must be positive ions

        # Poisson right-hand side
        charge_accum = AccumulatorVector(
            self.pointer["ions"],
            "H1",
            Pyccelkernel(accum_kernels_gc.gc_density_0form),
            self.mass_ops,
            self.domain.args_domain,
        )

        rho = (charge_accum, self.pointer["ions"])

        # get neutralizing background density
        if not self.pointer["ions"].control_variate:
            l2_proj = L2Projector("H1", self.mass_ops)
            f0e = Z * self.pointer["ions"].f0
            assert isinstance(f0e, KineticBackground)
            rho_eh = l2_proj.get_dofs(f0e.n)
            rho = [rho]
            rho += [rho_eh]

        # Get coupling strength
        if ions_params["options"]["override_eq_params"]:
            self.epsilon = ions_params["options"]["override_eq_params"]["epsilon"]
            print(
                f"\n!!! Override equation parameters: {self.epsilon = }.",
            )
        else:
            self.epsilon = self.equation_params["ions"]["epsilon"]

        # set keyword arguments for propagators
        self._kwargs[propagators_fields.ImplicitDiffusion] = {
            "sigma_1": 1.0 / self.epsilon**2 / Z,  # set to zero for Landau damping test
            "sigma_2": 0.0,
            "sigma_3": 1.0 / self.epsilon,
            "stab_mat": "M0ad",
            "diffusion_mat": "M1gyro",
            "rho": rho,
            "solver": solver_params,
        }

        self._kwargs[propagators_markers.PushGuidingCenterBxEstar] = {
            "phi": self.pointer["phi"],
            "evaluate_e_field": True,
            "epsilon": self.epsilon / Z,
            "algo": ions_params["options"]["PushGuidingCenterBxEstar"]["algo"],
        }

        self._kwargs[propagators_markers.PushGuidingCenterParallel] = {
            "phi": self.pointer["phi"],
            "evaluate_e_field": True,
            "epsilon": self.epsilon / Z,
            "algo": ions_params["options"]["PushGuidingCenterParallel"]["algo"],
        }

        # Initialize propagators used in splitting substeps
        self.init_propagators()

        # scalar quantities
        self.add_scalar("en_phi")
        self.add_scalar("en_particles", compute="from_particles", species="ions")
        self.add_scalar("en_tot")

        # MPI operations needed for scalar variables
        self._tmp3 = np.empty(1, dtype=float)
        self._e_field = self.derham.Vh["1"].zeros()

    def update_scalar_quantities(self):
        # energy from polarization
        e1 = self.derham.grad.dot(-self.pointer["phi"], out=self._e_field)
        en_phi1 = 0.5 * self.mass_ops.M1gyro.dot_inner(e1, e1)

        # energy from adiabatic electrons
        en_phi = 0.5 / self.epsilon**2 * self.mass_ops.M0ad.dot_inner(self.pointer["phi"], self.pointer["phi"])

        # for Landau damping test
        # en_phi = 0.

        # mu_p * |B0(eta_p)|
        self.pointer["ions"].save_magnetic_background_energy()

        # 1/N sum_p (w_p v_p^2/2 + mu_p |B0|_p)
        self._tmp3[0] = (
            1
            / self.pointer["ions"].Np
            * np.sum(
                self.pointer["ions"].weights * self.pointer["ions"].velocities[:, 0] ** 2 / 2.0
                + self.pointer["ions"].markers_wo_holes_and_ghost[:, 8],
            )
        )

        self.update_scalar("en_phi", en_phi + en_phi1)
        self.update_scalar("en_particles", self._tmp3[0])
        self.update_scalar("en_tot", en_phi + en_phi1 + self._tmp3[0])
