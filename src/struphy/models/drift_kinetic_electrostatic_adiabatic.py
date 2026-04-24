import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy import BaseUnits
from struphy.feec.mass import L2Projector
from struphy.io.options import LiteralOptions
from struphy.kinetic_background.base import KineticBackground
from struphy.models.base import StruphyModel
from struphy.models.scalars import FunctionScalarFEEC, FunctionScalarPIC, KineticEnergyPIC, Scalars
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

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

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
        ):
            self.var = PICVariable(space="Particles5D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                epsilon=epsilon,
            )

    ## propagators

    class Propagators:
        def __init__(self):
            self.gc_poisson = propagators_fields.ImplicitDiffusion()
            self.push_gc_bxe = propagators_markers.PushGuidingCenterBxEstar()
            self.push_gc_para = propagators_markers.PushGuidingCenterParallel()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        charge_number: int = 1,
        mass_number: float = 1.0,
        epsilon: float = None,
    ):

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons(
            charge_number,
            mass_number,
            epsilon,
        )

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.gc_poisson.variables.phi = self.em_fields.phi
        self.propagators.push_gc_bxe.variables.ions = self.kinetic_ions.var
        self.propagators.push_gc_para.variables.ions = self.kinetic_ions.var

        # 5. define scalars to be tracked during simulation
        field_energy = FunctionScalarFEEC(self._compute_en_phi)
        particle_kinetic = KineticEnergyPIC(self.kinetic_ions.var)
        particle_magnetic = FunctionScalarPIC(self._compute_en_particle_magnetic, self.kinetic_ions.var)
        particle_energy = particle_kinetic + particle_magnetic
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

        The model advances a 5D guiding-center distribution for one ion species
        together with an electrostatic quasi-neutrality equation with adiabatic
        electrons. The phase-space dynamics contain parallel streaming,
        :math:`\mathbf E\times\mathbf B` drift, magnetic drifts, and parallel
        acceleration in the background magnetic geometry."""

    @classmethod
    def doc_normalization(cls):
        r"""The reference speed is the ion thermal speed and the electrostatic
        fields are scaled accordingly:

        .. math::

            \hat v = \hat v_i,\qquad \hat E = \hat v_i \hat B,\qquad \hat\phi = \hat E \hat x.

        The small parameter is :math:`\varepsilon = 1/(\hat\Omega_c\hat t)`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Field energy: ``en_phi``
        - Guiding-center particle energy: ``en_particles``
        - Total energy: ``en_tot``"""

    @classmethod
    def doc_discretization(cls):
        doc = rf"""**1. propagators_fields.ImplicitDiffusion:**

{propagators_fields.ImplicitDiffusion.__doc__}

**2. propagators_markers.PushGuidingCenterBxEstar:**

{propagators_markers.PushGuidingCenterBxEstar.__doc__}

**3. propagators_markers.PushGuidingCenterParallel:**

{propagators_markers.PushGuidingCenterParallel.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""This model is an electrostatic drift-kinetic reduction for strongly
        magnetized ions in a fixed magnetic equilibrium. Electrons are not
        evolved kinetically; instead they enter through the adiabatic response
        in the quasi-neutrality solve. The implementation supports control
        variates for the field solve."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the drift-kinetic adiabatic-electron model:

        .. code-block:: python

            from struphy.models import DriftKineticElectrostaticAdiabatic

            model = DriftKineticElectrostaticAdiabatic()
            model.em_fields.phi
            model.kinetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - electrostatic drift-kinetic ion turbulence studies
        - strongly magnetized plasmas with adiabatic electrons
        - guiding-center PIC verification in realistic magnetic geometry
        - low-frequency regimes where full gyrophase resolution is unnecessary"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - fully electromagnetic dynamics with evolving magnetic perturbations
        - electron kinetic effects beyond the adiabatic closure
        - problems that require resolving full cyclotron motion
        - multi-species kinetic coupling without extending the model"""

    def allocate_helpers(self, verbose: bool = False):
        """Solve initial Poisson equation.

        :meta private:
        """
        self._tmp3 = xp.empty(1, dtype=float)
        self._e_field = Propagator.derham.V1.zeros()

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

        # get neutralizing background density
        if not particles.control_variate:
            l2_proj = L2Projector("H1", Propagator.mass_ops)
            f0e = Z * particles.f0
            assert isinstance(f0e, KineticBackground)
            rho_eh = FEECVariable(space="H1")
            rho_eh.allocate(derham=Propagator.derham, domain=Propagator.domain)
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

    def _compute_en_phi(self):
        phi = self.em_fields.phi.spline.vector
        epsilon = self.kinetic_ions.equation_params.epsilon

        e1 = Propagator.derham.grad.dot(-phi, out=self._e_field)
        en_phi1 = 0.5 * Propagator.mass_ops.M1gyro.dot_inner(e1, e1)
        en_phi = 0.5 / epsilon**2 * Propagator.mass_ops.M0ad.dot_inner(phi, phi)
        return en_phi + en_phi1

    def _compute_en_particle_magnetic(self):
        particles = self.kinetic_ions.var.particles
        particles.save_magnetic_background_energy()
        return 1 / particles.Np * xp.sum(particles.markers_wo_holes_and_ghost[:, 8])

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
                elif "push_gc_para.Options" in line:
                    new_file += [
                        "model.propagators.push_gc_para.options = model.propagators.push_gc_para.Options(phi=model.em_fields.phi)\n",
                    ]
                elif "set_save_data" in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["model.kinetic_ions.set_save_data(binning_plots=(binplot,))\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
