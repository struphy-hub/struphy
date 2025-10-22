from psydac.ddm.mpi import mpi as MPI

from struphy.feec.projectors import L2Projector
from struphy.kinetic_background.base import KineticBackground
from struphy.kinetic_background.maxwellians import Maxwellian3D
from struphy.models.base import StruphyModel
from struphy.models.species import FieldSpecies, FluidSpecies, ParticleSpecies
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.propagators import propagators_coupling, propagators_fields, propagators_markers
from struphy.utils.arrays import xp
from struphy.utils.pyccel import Pyccelkernel

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

    class KineticIons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles6D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self, with_B0: bool = True):
            self.push_eta = propagators_markers.PushEta()
            if with_B0:
                self.push_vxb = propagators_markers.PushVxB()
            self.coupling_va = propagators_coupling.VlasovAmpere()

    ## abstract methods

    def __init__(self, with_B0: bool = True):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        self.with_B0 = with_B0

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators(with_B0=with_B0)

        # 3. assign variables to propagators
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        if with_B0:
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
        self._tmp = xp.empty(1, dtype=float)

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
        particles = self.kinetic_ions.var.particles
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

        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon

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
                elif "push_vxb.Options" in line:
                    new_file += ["if model.with_B0:\n"]
                    new_file += ["    " + line]
                elif "set_save_data" in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["model.kinetic_ions.set_save_data(binning_plots=(binplot,))\n"]
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

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class KineticIons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles6D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.maxwell = propagators_fields.Maxwell()
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
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        self.propagators.push_vxb.variables.ions = self.kinetic_ions.var
        self.propagators.coupling_va.variables.e = self.em_fields.e_field
        self.propagators.coupling_va.variables.ions = self.kinetic_ions.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_E")
        self.add_scalar("en_B")
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
        self._tmp = xp.empty(1, dtype=float)

    def update_scalar_quantities(self):
        # e*M1*e/2
        e = self.em_fields.e_field.spline.vector
        b = self.em_fields.b_field.spline.vector

        en_E = 0.5 * self.mass_ops.M1.dot_inner(e, e)
        self.update_scalar("en_E", en_E)

        en_B = 0.5 * self.mass_ops.M2.dot_inner(b, b)
        self.update_scalar("en_B", en_B)

        # alpha^2 / 2 / N * sum_p w_p v_p^2
        particles = self.kinetic_ions.var.particles
        alpha = self.kinetic_ions.equation_params.alpha
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
        particles = self.kinetic_ions.var.particles
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

        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon

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
                elif "push_vxb.Options" in line:
                    new_file += [
                        "model.propagators.push_vxb.options = model.propagators.push_vxb.Options(b2_var=model.em_fields.b_field)\n"
                    ]
                elif "set_save_data" in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["model.kinetic_ions.set_save_data(binning_plots=(binplot,))\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)


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

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class KineticIons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="DeltaFParticles6D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(
            self,
            with_B0: bool = True,
            with_E0: bool = True,
        ):
            self.push_eta = propagators_markers.PushEta()
            if with_E0:
                self.push_vinE = propagators_markers.PushVinEfield()
            self.coupling_Eweights = propagators_coupling.EfieldWeights()
            if with_B0:
                self.push_vxb = propagators_markers.PushVxB()

    ## abstract methods

    def __init__(
        self,
        with_B0: bool = True,
        with_E0: bool = True,
    ):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators(with_B0=with_B0, with_E0=with_E0)

        # 3. assign variables to propagators
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        if with_E0:
            self.propagators.push_vinE.variables.var = self.kinetic_ions.var
        self.propagators.coupling_Eweights.variables.e = self.em_fields.e_field
        self.propagators.coupling_Eweights.variables.ions = self.kinetic_ions.var
        if with_B0:
            self.propagators.push_vxb.variables.ions = self.kinetic_ions.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_E")
        self.add_scalar("en_w", compute="from_particles", variable=self.kinetic_ions.var)
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
        self._tmp = xp.empty(1, dtype=float)

    def update_scalar_quantities(self):
        # e*M1*e/2
        e = self.em_fields.e_field.spline.vector
        particles = self.kinetic_ions.var.particles

        en_E = 0.5 * self.mass_ops.M1.dot_inner(e, e)
        self.update_scalar("en_E", en_E)

        # evaluate f0
        if not hasattr(self, "_f0"):
            backgrounds = self.kinetic_ions.var.backgrounds
            if isinstance(backgrounds, list):
                self._f0 = backgrounds[0]
            else:
                self._f0 = backgrounds
            self._f0_values = xp.zeros(
                self.kinetic_ions.var.particles.markers.shape[0],
                dtype=float,
            )
            assert isinstance(self._f0, Maxwellian3D)

        self._f0_values[particles.valid_mks] = self._f0(*particles.phasespace_coords.T)

        # alpha^2 * v_th^2 / (2*N) * sum_p s_0 * w_p^2 / f_{0,p}
        alpha = self.kinetic_ions.equation_params.alpha
        vth = self._f0.maxw_params["vth1"][0]

        self._tmp[0] = (
            alpha**2
            * vth**2
            / (2 * particles.Np)
            * xp.dot(
                particles.weights**2,  # w_p^2
                particles.sampling_density / self._f0_values[particles.valid_mks],  # s_{0,p} / f_{0,p}
            )
        )

        self.update_scalar("en_w", self._tmp[0])
        self.update_scalar("en_tot", self._tmp[0] + en_E)

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

        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon

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
                if "maxwellian_1 + maxwellian_2" in line:
                    new_file += ["background = maxwellian_1\n"]
                elif "maxwellian_1pt =" in line:
                    new_file += ["maxwellian_1pt = maxwellians.Maxwellian3D(n=(0.0, perturbation))\n"]
                elif "set_save_data" in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["model.kinetic_ions.set_save_data(binning_plots=(binplot,))\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)


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

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class KineticIons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="DeltaFParticles6D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(
            self,
            with_B0: bool = True,
            with_E0: bool = True,
        ):
            self.push_eta = propagators_markers.PushEta()
            if with_E0:
                self.push_vinE = propagators_markers.PushVinEfield()
            self.coupling_Eweights = propagators_coupling.EfieldWeights()
            if with_B0:
                self.push_vxb = propagators_markers.PushVxB()
            self.maxwell = propagators_fields.Maxwell()

    ## abstract methods

    def __init__(
        self,
        with_B0: bool = True,
        with_E0: bool = True,
    ):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators(with_B0=with_B0, with_E0=with_E0)

        # 3. assign variables to propagators
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        if with_E0:
            self.propagators.push_vinE.variables.var = self.kinetic_ions.var
        self.propagators.coupling_Eweights.variables.e = self.em_fields.e_field
        self.propagators.coupling_Eweights.variables.ions = self.kinetic_ions.var
        if with_B0:
            self.propagators.push_vxb.variables.ions = self.kinetic_ions.var
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field

        # define scalars for update_scalar_quantities
        self.add_scalar("en_E")
        self.add_scalar("en_B")
        self.add_scalar("en_w", compute="from_particles", variable=self.kinetic_ions.var)
        self.add_scalar("en_tot")

        # initial Poisson (not a propagator used in time stepping)
        self.initial_poisson = propagators_fields.Poisson()
        self.initial_poisson.variables.phi = self.em_fields.phi

    def update_scalar_quantities(self):
        super().update_scalar_quantities()

        # 0.5 * b^T * M_2 * b
        b = self.em_fields.b_field.spline.vector

        en_B = 0.5 * self._mass_ops.M2.dot_inner(b, b)
        self.update_scalar("en_tot", self.scalar_quantities["en_tot"]["value"][0] + en_B)


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

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class KineticIons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles5D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.gc_poisson = propagators_fields.ImplicitDiffusion()
            self.push_gc_bxe = propagators_markers.PushGuidingCenterBxEstar()
            self.push_gc_para = propagators_markers.PushGuidingCenterParallel()

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
        self.propagators.gc_poisson.variables.phi = self.em_fields.phi
        self.propagators.push_gc_bxe.variables.ions = self.kinetic_ions.var
        self.propagators.push_gc_para.variables.ions = self.kinetic_ions.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_phi")
        self.add_scalar("en_particles", compute="from_particles", variable=self.kinetic_ions.var)
        self.add_scalar("en_tot")

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "thermal"

    def allocate_helpers(self):
        self._tmp3 = xp.empty(1, dtype=float)
        self._e_field = self.derham.Vh["1"].zeros()

        assert self.kinetic_ions.charge_number > 0, "Model written only for positive ions."

    def allocate_propagators(self):
        """Solve initial Poisson equation.

        :meta private:
        """

        # initialize fields and particles
        super().allocate_propagators()

        # Poisson right-hand side
        particles = self.kinetic_ions.var.particles
        Z = self.kinetic_ions.charge_number
        epsilon = self.kinetic_ions.equation_params.epsilon

        charge_accum = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels_gc.gc_density_0form),
            self.mass_ops,
            self.domain.args_domain,
        )

        rho = charge_accum

        # get neutralizing background density
        if not particles.control_variate:
            l2_proj = L2Projector("H1", self.mass_ops)
            f0e = Z * particles.f0
            assert isinstance(f0e, KineticBackground)
            rho_eh = FEECVariable(space="H1")
            rho_eh.allocate(derham=self.derham, domain=self.domain)
            rho_eh.spline.vector = l2_proj.get_dofs(f0e.n)
            rho = [rho]
            rho += [rho_eh]

        self.propagators.gc_poisson.options.sigma_1 = 1.0 / epsilon**2 / Z
        self.propagators.gc_poisson.options.sigma_2 = 0.0
        self.propagators.gc_poisson.options.sigma_3 = 1.0 / epsilon
        self.propagators.gc_poisson.options.stab_mat = "M0ad"
        self.propagators.gc_poisson.options.diffusion_mat = "M1perp"
        self.propagators.gc_poisson.options.rho = rho
        self.propagators.gc_poisson.allocate()

    def update_scalar_quantities(self):
        phi = self.em_fields.phi.spline.vector
        particles = self.kinetic_ions.var.particles
        epsilon = self.kinetic_ions.equation_params.epsilon

        # energy from polarization
        e1 = self.derham.grad.dot(-phi, out=self._e_field)
        en_phi1 = 0.5 * self.mass_ops.M1gyro.dot_inner(e1, e1)

        # energy from adiabatic electrons
        en_phi = 0.5 / epsilon**2 * self.mass_ops.M0ad.dot_inner(phi, phi)

        # for Landau damping test
        # en_phi = 0.

        # mu_p * |B0(eta_p)|
        particles.save_magnetic_background_energy()

        # 1/N sum_p (w_p v_p^2/2 + mu_p |B0|_p)
        self._tmp3[0] = (
            1
            / particles.Np
            * xp.sum(
                particles.weights * particles.velocities[:, 0] ** 2 / 2.0 + particles.markers_wo_holes_and_ghost[:, 8],
            )
        )

        self.update_scalar("en_phi", en_phi + en_phi1)
        self.update_scalar("en_particles", self._tmp3[0])
        self.update_scalar("en_tot", en_phi + en_phi1 + self._tmp3[0])

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "BaseUnits(" in line:
                    new_file += ["base_units = BaseUnits(kBT=1.0)\n"]
                elif "push_gc_bxe.Options" in line:
                    new_file += [
                        "model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(phi=model.em_fields.phi)\n"
                    ]
                elif "push_gc_para.Options" in line:
                    new_file += [
                        "model.propagators.push_gc_para.options = model.propagators.push_gc_para.Options(phi=model.em_fields.phi)\n"
                    ]
                elif "set_save_data" in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["model.kinetic_ions.set_save_data(binning_plots=(binplot,))\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
