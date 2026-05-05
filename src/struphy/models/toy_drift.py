import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import FunctionScalarFEEC, KineticEnergyPIC, Scalars
from struphy.models.species import (
    FieldSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation import accum_kernels_gc
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.propagators.base import Propagator
from struphy.propagators.poisson_field_solve import PoissonFieldSolve
from struphy.propagators.push_guiding_center_bx_estar import PushGuidingCenterBxEstar
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

    1. :class:`~struphy.propagators.implicit_diffusion.ImplicitDiffusion`
    2. :class:`~struphy.propagators.push_guiding_center_bx_estar.PushGuidingCenterBxEstar`

    :ref:`Model info <add_model>`:
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Toy"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class KineticIons(ParticleSpecies):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            epsilon: float = None,
            alpha: float = None,
        ):
            self.var = PICVariable(space="Particles5D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                epsilon=epsilon,
                alpha=alpha,
            )

    ## propagators

    class Propagators:
        def __init__(self):
            self.gc_poisson = PoissonFieldSolve()
            self.push_gc_bxe = PushGuidingCenterBxEstar()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        charge_number: int = 1,
        mass_number: float = 1.0,
        epsilon: float = None,
        alpha: float = None,
    ):

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons(
            charge_number,
            mass_number,
            epsilon,
            alpha,
        )

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.gc_poisson.variables.phi = self.em_fields.phi
        self.propagators.push_gc_bxe.variables.ions = self.kinetic_ions.var

        # 5. define scalars to be tracked during simulation
        field_energy = FunctionScalarFEEC(self._compute_en_phi)
        particle_energy = KineticEnergyPIC(self.kinetic_ions.var)
        self.scalars = Scalars(
            en_phi=field_energy,
            en_particles=particle_energy,
            en_tot=field_energy + particle_energy,
        )

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "thermal"

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Drift equation:

        .. math::

            \frac{\partial f}{\partial t} + \frac{\mathbf{E} \times \mathbf{b}_0}{B^*_\parallel} \cdot \frac{\partial f}{\partial \mathbf{X}} = 0

        Poisson equation:

        .. math::

            -\nabla_\perp \cdot \left( \frac{n_0}{|B_0|^2} \nabla_\perp \phi \right) + \frac{1}{\varepsilon} n_0 \left( 1 + \frac{1}{Z \varepsilon} \frac{1}{T_0} \phi \right) = \frac{1}{\varepsilon} \int f B^*_\parallel \, \textnormal{d} v_\parallel \textnormal{d} \mu

        where :math:`f(\mathbf{X}, v_\parallel, \mu, t)` is the guiding center distribution and

        .. math::

            \mathbf{E} = -\nabla \phi, \qquad \mathbf{B}^* = \mathbf{B}_0 + \varepsilon v_\parallel \nabla \times \mathbf{b}_0, \qquad B^*_\parallel = \mathbf{B}^* \cdot \mathbf{b}_0

        Notes
        -----

        * The ``control_var`` in the Poisson equation is optional; in case it is enabled via the parameter file, the following Poisson equation is solved:
        Find :math:`\phi \in H^1` such that

        .. math::

            \int \frac{n_0}{|B_0|^2} \nabla_\perp \psi \cdot \nabla_\perp \phi \, \textrm{d} \mathbf{x} + \frac{1}{Z \varepsilon^2} \int \frac{n_0}{T_0} \psi \phi \, \textrm{d} \mathbf{x} = \frac{1}{\varepsilon} \int \int \psi \, (f - f_0) B^*_\parallel \, \textrm{d} \mathbf{x} \, \textnormal{d} v_\parallel \textnormal{d} \mu \qquad \forall \ \psi \in H^1
        """

    @classmethod
    def doc_normalization(cls):
        r"""The reference speed is the ion thermal velocity:

        .. math::

            \hat v = \hat v_i,\qquad \hat E = \hat v_i \hat B,\qquad \hat\phi = \hat E \hat x.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Electrostatic field energy: ``en_phi``
        - Particle kinetic energy: ``en_particles``
        - Total energy: ``en_tot``"""

    @classmethod
    def doc_discretization(cls):
        doc = rf"""**1. PoissonFieldSolve:**

{PoissonFieldSolve.__doc__}

**2. push_guiding_center_bx_estar.PushGuidingCenterBxEstar:**

{PushGuidingCenterBxEstar.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""ToyDrift is a stripped-down guiding-center model used to isolate the
        electrostatic drift part of the dynamics. It is intended for algorithm
        prototyping and reduced verification problems rather than production
        drift-kinetic studies."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the toy drift model:

        .. code-block:: python

            from struphy.models import ToyDrift

            model = ToyDrift()
            model.em_fields.phi
            model.kinetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - reduced electrostatic guiding-center benchmarks
        - testing the field solve plus :math:`\mathbf E\times\mathbf B` pusher
        - algorithm prototyping before moving to the full drift-kinetic model"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - full drift-kinetic dynamics with parallel streaming
        - electromagnetic perturbations
        - high-fidelity turbulence studies
        - full-orbit kinetic physics"""

    def allocate_helpers(self, verbose: bool = False):
        """Solve initial Poisson equation.

        :meta private:
        """
        self._tmp3 = xp.empty(1, dtype=float)
        self._e_field = Propagator.derham.V1.zeros()

        assert self.kinetic_ions.charge_number > 0, "Model written only for positive ions."

        # Poisson right-hand side
        particles = self.kinetic_ions.var.particles
        particles.weights = particles.weights_at_t0.copy()

        alpha = self.kinetic_ions.equation_params.alpha
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

        self.propagators.gc_poisson.options.stab_eps = 0.0
        self.propagators.gc_poisson.options.stab_mat = "M0ad"
        self.propagators.gc_poisson.options.rho = rho
        self.propagators.gc_poisson.options.rho_coeffs = alpha**2 / epsilon
        self.propagators.gc_poisson.allocate()

        if particles.control_variate:
            particles.update_weights()

    def _compute_en_phi(self):
        phi = self.em_fields.phi.spline.vector
        e1 = Propagator.derham.grad.dot(-phi, out=self._e_field)
        return 0.5 * Propagator.mass_ops.M1.dot_inner(e1, e1)

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
