import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation import accum_kernels
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.propagators import (
    propagators_coupling,
    propagators_fields,
    propagators_markers,
)
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel

rank = MPI.COMM_WORLD.Get_rank()


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

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

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

        # property to measure violation of gauss law from control variate
        self.measure_gauss = False
    
    def measure_gauss_error(self, measure: bool = False):
        """
        Record gauss law violation during simulaiton:

        .. math::
            \text{error(t)} = \max(|\mathbf{G}^T\mathbf{M}^1\mathbf{e} -  \mathbf{f}^p|)

        :params measure: if True, measure gauss error throughout the simulation
        """
        if measure: 
            self.measure_gauss = True
            self.add_scalar("gauss_error")

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self, verbose: bool = False):
        """Solve initial Poisson equation.

        :meta private:
        """
        self._tmp = xp.empty(1, dtype=float)

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
            Propagator.mass_ops,
            Propagator.domain.args_domain,
        )

        # another sanity check: compute FE coeffs of density
        # charge_accum.show_accumulated_spline_field(Propagator.mass_ops)

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
        Propagator.derham.grad.dot(-phi, out=self.em_fields.e_field.spline.vector)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("... Done.")

    def calculate_gauss_error(self, e):
        # control variate method
        particles = self.kinetic_ions.var.particles
        particles.update_weights()
        charge_accum0 = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels.charge_density_0form),
            Propagator.mass_ops,
            Propagator.domain.args_domain,
        )
        charge_accum0 = charge_accum0.to_array()

        # non control variate method
        op = Propagator.derham.grad.T @ Propagator.mass_ops.M1
        charge_accum1 = op.dot(e)
        charge_accum1 = charge_accum1.to_array()

        residual = xp.max(xp.abs(charge_accum0 - charge_accum1))
        return residual

    def update_scalar_quantities(self):
        # e*M1*e/2
        e = self.em_fields.e_field.spline.vector
        b = self.em_fields.b_field.spline.vector

        en_E = 0.5 * Propagator.mass_ops.M1.dot_inner(e, e)
        self.update_scalar("en_E", en_E)

        en_B = 0.5 * Propagator.mass_ops.M2.dot_inner(b, b)
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

        if self.measure_gauss:
            res = self.calculate_gauss_error(e)
            self.update_scalar("gauss_error", res)

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
                        "model.propagators.push_vxb.options = model.propagators.push_vxb.Options(b2_var=model.em_fields.b_field)\n",
                    ]
                elif "set_save_data" in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["model.kinetic_ions.set_save_data(binning_plots=(binplot,))\n"]
                    new_file += ["model.measure_gauss_error(measure=False)\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
