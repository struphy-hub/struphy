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


class VlasovAmpereOneSpecies(StruphyModel):
    """
    Vlasov-Ampère system for a single kinetic species in an electric field.

    <p>This model couples the Vlasov equation for the particle distribution function with Ampère's law
    for the electric field evolution. It includes the effect of a static background magnetic field <code>𝐁₀</code>
    and solves the initial Poisson equation to satisfy Gauss's law at t=0. The model uses a particle-in-cell (PIC)
    method with finite element exterior calculus (FEEC) for the electromagnetic fields.</p>

    <h3>Governing Equations</h3>

    <p><strong>Vlasov equation:</strong></p>
    <p><code>∂f/∂t + 𝐯·∇f + (1/ε)(𝐄 + 𝐯×𝐁₀)·∂f/∂𝐯 = 0</code></p>

    <p><strong>Ampère's law:</strong></p>
    <p><code>-∂𝐄/∂t = (α²/ε) ∫ 𝐯 f d³𝐯</code></p>

    <p><strong>Initial Poisson equation:</strong> At t=0, solve weakly for the electric potential φ:</p>
    <p><code>∫ ∇ψ·∇φ dx = (α²/ε) ∫∫ ψ(f - f₀) d³𝐯 dx</code></p>
    <p><code>𝐄(0) = -∇φ(0)</code></p>

    <h3>Normalization</h3>

    <p>The model uses the following normalizations:</p>
    <ul>
        <li>Velocity: <code>v̂ = c</code></li>
        <li>Electric field: <code>Ê = B̂ v̂</code></li>
        <li>Potential: <code>φ̂ = Ê x̂</code></li>
    </ul>

    <p><strong>Dimensionless parameters:</strong></p>
    <ul>
        <li><code>α = Ω̂ₚ/Ω̂ₓ</code> (ratio of plasma to cyclotron frequency)</li>
        <li><code>ε = 1/(Ω̂ₓ t̂)</code> (inverse cyclotron period)</li>
    </ul>

    <p>where <code>Ω̂ₚ = √(n̂(Ze)²/(ε₀Am_H))</code> and <code>Ω̂ₓ = (Ze)B̂/(Am_H)</code></p>
    <p>For electrons: <code>Z = -1</code>, <code>A = 1/1836</code></p>

    <h3>Species</h3>

    <ul>
        <li><code>em_fields.e_field</code> - Electric field (H(curl) space)</li>
        <li><code>em_fields.phi</code> - Electric potential (H¹ space)</li>
        <li><code>kinetic_ions.var</code> - Particle distribution function (6D phase space)</li>
    </ul>

    <h3>Propagators</h3>

    <p>Time integration is performed by the following propagators (in sequence):</p>
    <ol>
        <li><code>PushEta</code> - Push particles in configuration space</li>
        <li><code>PushVxB</code> - Push particles in velocity space (v × B₀ term)</li>
        <li><code>VlasovAmpere</code> - Couple Vlasov and Ampère equations</li>
    </ol>

    <h3>Initial Condition</h3>

    <p>The initial electric field is computed by solving the weak Poisson equation to ensure
    the charge density from the particle distribution satisfies Gauss's law. The background
    magnetic field <code>𝐁₀</code> must satisfy:</p>
    <p><code>∇×𝐁₀ = (α²/ε) ∫ 𝐯 f₀ d³𝐯</code></p>

    <h3>Control Variate Method</h3>

    <p>An optional control variate technique can be enabled to reduce numerical noise in
    Ampère's law by subtracting the equilibrium distribution <code>f₀</code>. When enabled,
    the weak form becomes:</p>
    <p><code>-∫ 𝐅·∂𝐄/∂t dx = (α²/ε) ∫∫ 𝐅·𝐯(f - f₀) d³𝐯 dx</code></p>
    <p>for all test functions <code>𝐅 ∈ H(curl)</code>.</p>

    <h3>Scalar Quantities</h3>

    <p>The following energies are tracked during simulation:</p>
    <ul>
        <li>Electric field energy: <code>E<sub>E</sub> = ½ ∫ |𝐄|² dV</code></li>
        <li>Kinetic energy: <code>E<sub>f</sub> = (α²/2N) Σ<sub>p</sub> w<sub>p</sub> |𝐯<sub>p</sub>|²</code></li>
        <li>Total energy: <code>E<sub>tot</sub> = E<sub>E</sub> + E<sub>f</sub></code></li>
    </ul>

    <h3>Model Properties</h3>

    <ul>
        <li><strong>Model type:</strong> Kinetic</li>
        <li><strong>Velocity scale:</strong> Speed of light</li>
        <li><strong>Bulk species:</strong> kinetic_ions</li>
    </ul>

    <h3>Parameters</h3>

    <ul>
        <li><code>with_B0</code> (bool) - Include background magnetic field effects (default: True)</li>
    </ul>
    """

    __doc_rst__ = r"""
Vlasov-Ampère system for a single kinetic species in an electric field.

This model couples the Vlasov equation for the particle distribution function with Ampère's law
for the electric field evolution. It includes the effect of a static background magnetic field :math:`\mathbf{B}_0`
and solves the initial Poisson equation to satisfy Gauss's law at :math:`t=0`. The model uses a particle-in-cell (PIC)
method with finite element exterior calculus (FEEC) for the electromagnetic fields.

**Governing Equations**

**Vlasov equation:**

.. math::

    \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \frac{1}{\varepsilon} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B}_0 \right)
    \cdot \frac{\partial f}{\partial \mathbf{v}} = 0

**Ampère's law:**

.. math::

    -\frac{\partial \mathbf{E}}{\partial t} = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f \, \mathrm{d}^3 \mathbf{v}

**Initial Poisson equation:** At :math:`t=0`, solve weakly for the electric potential :math:`\phi`:

.. math::

    \int_\Omega \nabla \psi^\top \cdot \nabla \phi \,\mathrm{d} \mathbf{x} &= \frac{\alpha^2}{\varepsilon}  \int_\Omega \int_{\mathbb{R}^3} \psi\, (f - f_0) \, \mathrm{d}^3 \mathbf{v}\,\mathrm{d} \mathbf{x} \qquad \forall \ \psi \in H^1
    
    \mathbf{E}(t=0) &= -\nabla \phi(t=0)

**Normalization**

The model uses the following normalizations:

.. math::

    \hat{v} = c, \qquad \hat{E} = \hat{B} \hat{v}, \qquad \hat{\phi} = \hat{E} \hat{x}

**Dimensionless parameters:**

.. math::

    \alpha = \frac{\hat{\Omega}_\mathrm{p}}{\hat{\Omega}_\mathrm{c}}, \qquad 
    \varepsilon = \frac{1}{\hat{\Omega}_\mathrm{c} \hat{t}}

where

.. math::

    \hat{\Omega}_\mathrm{p} = \sqrt{\frac{\hat{n} (Ze)^2}{\epsilon_0 (A m_\mathrm{H})}}, \qquad
    \hat{\Omega}_\mathrm{c} = \frac{(Ze) \hat{B}}{(A m_\mathrm{H})}

For electrons: :math:`Z = -1`, :math:`A = 1/1836`.

**Species**

- ``em_fields.e_field`` - Electric field (H(curl) space)
- ``em_fields.phi`` - Electric potential (H¹ space)
- ``kinetic_ions.var`` - Particle distribution function (6D phase space)

**Propagators**

Time integration is performed by the following propagators (in sequence):

1. :class:`~struphy.propagators.propagators_markers.PushEta` - Push particles in configuration space
2. :class:`~struphy.propagators.propagators_markers.PushVxB` - Push particles in velocity space (:math:`\mathbf{v} \times \mathbf{B}_0` term)
3. :class:`~struphy.propagators.propagators_coupling.VlasovAmpere` - Couple Vlasov and Ampère equations

**Initial Condition**

The initial electric field is computed by solving the weak Poisson equation to ensure
the charge density from the particle distribution satisfies Gauss's law. The background
magnetic field :math:`\mathbf{B}_0` must satisfy:

.. math::

    \nabla \times \mathbf{B}_0 = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \mathrm{d}^3 \mathbf{v}

**Control Variate Method**

An optional control variate technique can be enabled to reduce numerical noise in
Ampère's law by subtracting the equilibrium distribution :math:`f_0`. When enabled,
the weak form becomes:

Find :math:`(\mathbf{E}, f) \in H(\mathrm{curl}) \times C^\infty` such that

.. math::

    -\int_\Omega \mathbf{F} \cdot \frac{\partial \mathbf{E}}{\partial t}\,\mathrm{d} \mathbf{x} &= 
    \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} \mathbf{F} \cdot \mathbf{v} (f - f_0) \, \mathrm{d}^3 \mathbf{v}\,\mathrm{d} \mathbf{x} 
    \qquad \forall \ \mathbf{F} \in H(\mathrm{curl})
    
    \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \frac{1}{\varepsilon} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B}_0 \right) \cdot \frac{\partial f}{\partial \mathbf{v}} &= 0

**Scalar Quantities**

The following energies are tracked during simulation:

- Electric field energy: :math:`E_E = \frac{1}{2} \int |\mathbf{E}|^2 \, \mathrm{d}V`
- Kinetic energy: :math:`E_f = \frac{\alpha^2}{2N} \sum_p w_p |\mathbf{v}_p|^2`
- Total energy: :math:`E_\mathrm{tot} = E_E + E_f`

**Model Properties**

- **Model type:** Kinetic
- **Velocity scale:** Speed of light
- **Bulk species:** kinetic_ions

**Parameters**

- ``with_B0`` (bool) - Include background magnetic field effects (default: True)

**See Also**

- :class:`~struphy.models.base.StruphyModel` - Base class for all Struphy models
- :class:`~struphy.propagators.propagators_coupling.VlasovAmpere` - Vlasov-Ampère coupling propagator
- :class:`~struphy.propagators.propagators_markers.PushEta` - Configuration space particle pusher
- :class:`~struphy.propagators.propagators_markers.PushVxB` - Velocity space particle pusher

**Examples**

Create and initialize a Vlasov-Ampère model:

.. code-block:: python

    from struphy.models.vlasov_ampere_one_species import VlasovAmpereOneSpecies
    
    # Create model with background magnetic field
    model = VlasovAmpereOneSpecies(with_B0=True)
    
    # Access fields and particles
    # model.em_fields.e_field  - Electric field
    # model.kinetic_ions.var   - Particle distribution
    
    # After initialization, allocate_helpers() solves the initial Poisson equation
"""

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

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

    def update_scalar_quantities(self):
        # e*M1*e/2
        e = self.em_fields.e_field.spline.vector
        en_E = 0.5 * Propagator.mass_ops.M1.dot_inner(e, e)
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
