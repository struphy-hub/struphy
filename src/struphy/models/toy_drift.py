import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.feec.projectors import L2Projector
from struphy.io.options import LiteralOptions
from struphy.kinetic_background.base import KineticBackground
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation import accum_kernels_gc
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.propagators import (
    propagators_fields,
    propagators_markers,
)
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel

rank = MPI.COMM_WORLD.Get_rank()


class ToyDrift(StruphyModel):
    r"""Drift-kinetic equation for one ion species in static background magnetic field.

    :ref:`normalization`:

    .. math::

       \hat v = \hat v_\textrm{i} = \sqrt{\frac{k_B \hat T_\textrm{i}}{m_\textrm{i}}}\,,\qquad  \hat E = \hat v_\textrm{i}\hat B\,,\qquad \hat \phi = \hat E \hat x \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial f}{\partial t} + \frac{\mathbf{E} \times \mathbf{b}_0}{B^*_\parallel} \cdot \frac{\partial f}{\partial \mathbf{X}} = 0\,.
        \\[2mm]
        - &\nabla_\perp \cdot \left( \frac{n_0}{|B_0|^2} \nabla_\perp \phi \right) + \frac{1}{\varepsilon} n_0 \left(1 + \frac{1}{Z \varepsilon} \frac{1}{T_{0}} \phi \right) = \frac 1 \varepsilon \int f B^*_\parallel \,\textnormal d v_\parallel \textnormal d \mu \,.

    where :math:`f(\mathbf{X}, v_\parallel, \mu, t)` is the guiding center distribution and

    .. math::
        \mathbf{E} = - \nabla \phi \,,  \qquad \mathbf{B}^* = \mathbf{B}_0 + \varepsilon v_\parallel \nabla \times \mathbf{b}_0 \,,\qquad B^*_\parallel = \mathbf B^* \cdot \mathbf b_0  \,,

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

    :ref:`Model info <add_model>`:
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

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

    ## abstract methods

    def __init__(self):

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.gc_poisson.variables.phi = self.em_fields.phi
        self.propagators.push_gc_bxe.variables.ions = self.kinetic_ions.var

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

    def allocate_helpers(self, verbose: bool = False):
        """Solve initial Poisson equation.

        :meta private:
        """
        self._tmp3 = xp.empty(1, dtype=float)
        self._e_field = Propagator.derham.Vh["1"].zeros()

        assert self.kinetic_ions.charge_number > 0, "Model written only for positive ions."

        # Poisson right-hand side
        particles = self.kinetic_ions.var.particles
        Z = self.kinetic_ions.charge_number
        epsilon = self.kinetic_ions.equation_params.epsilon

        charge_accum = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels_gc.gc_density_0form),
            Propagator.mass_ops,
            Propagator.domain.args_domain,
        )

        rho = charge_accum

        # sanity check: compute FE coeffs of density
        # rho()
        # rho.show_accumulated_spline_field(Propagator.mass_ops, eta_direction=(True,True,False))

        self.propagators.gc_poisson.options.sigma_1 = 1e-8
        self.propagators.gc_poisson.options.sigma_2 = 0.0
        self.propagators.gc_poisson.options.sigma_3 = 1.0
        self.propagators.gc_poisson.options.stab_mat = "M0ad"
        self.propagators.gc_poisson.options.diffusion_mat = "M1"
        self.propagators.gc_poisson.options.rho = rho
        self.propagators.gc_poisson.allocate()

    def update_scalar_quantities(self):
        phi = self.em_fields.phi.spline.vector
        particles = self.kinetic_ions.var.particles
        epsilon = self.kinetic_ions.equation_params.epsilon

        # energy from polarization
        e1 = Propagator.derham.grad.dot(-phi, out=self._e_field)
        en_phi1 = 0.5 * Propagator.mass_ops.M1gyro.dot_inner(e1, e1)

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

        self.update_scalar("en_phi", en_phi1)
        self.update_scalar("en_particles", self._tmp3[0])
        self.update_scalar("en_tot", en_phi1 + self._tmp3[0])

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
                        "model.propagators.push_gc_bxe.options = model.propagators.push_gc_bxe.Options(phi=model.em_fields.phi)\n",
                    ]
                elif "set_save_data" in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["model.kinetic_ions.set_save_data(binning_plots=(binplot,))\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
