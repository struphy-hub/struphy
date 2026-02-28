from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.block import BlockVector

from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.polar.basic import PolarVector
from struphy.propagators import (
    propagators_fields,
)
from struphy.propagators.base import Propagator

rank = MPI.COMM_WORLD.Get_rank()


class LinearMHD(StruphyModel):
    """
    Linear ideal MHD with zero-flow equilibrium for magnetohydrodynamic wave propagation.

    <p>This model simulates small-amplitude perturbations in a magnetized plasma with a static
    equilibrium magnetic field <code>𝐁₀</code> and zero background flow. The model solves the linearized
    ideal magnetohydrodynamic (MHD) equations, which couple fluid dynamics (density, velocity, pressure)
    with magnetic field evolution. The system supports both Alfvén waves (incompressible shear) and
    magnetosonic waves (compressible fast and slow modes).</p>

    <h3>Governing Equations</h3>

    <p><strong>Continuity (mass conservation):</strong></p>
    <p><code>∂ρ̃/∂t + ∇·(ρ₀ Ũ) = 0</code></p>

    <p><strong>Momentum (Lorentz force):</strong></p>
    <p><code>ρ₀ ∂Ũ/∂t + ∇p̃ = (∇×B̃)×𝐁₀ + (∇×𝐁₀)×B̃</code></p>

    <p><strong>Energy (adiabatic process):</strong></p>
    <p><code>∂p̃/∂t + ∇·(p₀ Ũ) + (2/3)p₀∇·Ũ = 0</code></p>

    <p><strong>Induction (Faraday's law):</strong></p>
    <p><code>∂B̃/∂t - ∇×(Ũ×𝐁₀) = 0</code></p>

    <h3>Normalization</h3>

    <p>All velocities are normalized by the Alfvén velocity:</p>
    <p><code>Û = v̂ₐ = B̂₀/√(μ₀ρ₀)</code></p>

    <h3>Perturbation Variables</h3>

    <p>All quantities in the equations represent <strong>perturbations</strong> around equilibrium:</p>
    <ul>
        <li><code>ρ̃</code> - Density perturbation</li>
        <li><code>Ũ</code> - Velocity perturbation</li>
        <li><code>p̃</code> - Pressure perturbation</li>
        <li><code>B̃</code> - Magnetic field perturbation</li>
    </ul>

    <p>Equilibrium quantities (with subscript 0) are stationary and satisfy:</p>
    <ul>
        <li><code>ρ₀</code> - Static background density</li>
        <li><code>p₀</code> - Static background pressure</li>
        <li><code>𝐁₀</code> - Static background magnetic field</li>
    </ul>

    <h3>Species</h3>

    <ul>
        <li><code>em_fields.b_field</code> - Magnetic field perturbation (H(div) space)</li>
        <li><code>mhd.density</code> - Density perturbation (L² space)</li>
        <li><code>mhd.velocity</code> - Velocity perturbation (H(div) space)</li>
        <li><code>mhd.pressure</code> - Pressure perturbation (L² space)</li>
    </ul>

    <h3>Wave Modes</h3>

    <p>The linear MHD system supports three wave types:</p>
    <ul>
        <li><strong>Alfvén waves:</strong> Incompressible shear waves propagating along <code>𝐁₀</code> with velocity <code>v_A</code></li>
        <li><strong>Fast magnetosonic wave:</strong> Compressible wave with phase velocity <code>v_fast > v_A</code></li>
        <li><strong>Slow magnetosonic wave:</strong> Compressible wave with phase velocity <code>v_slow < v_A</code></li>
    </ul>

    <h3>Propagators</h3>

    <p>Time integration is performed by the following propagators (in sequence):</p>
    <ol>
        <li><code>ShearAlfven</code> - Evolves Alfvén waves (velocity and magnetic field coupling)</li>
        <li><code>Magnetosonic</code> - Evolves compressible modes (density, velocity, pressure)</li>
    </ol>

    <h3>Scalar Quantities</h3>

    <p>The following energies are tracked during simulation:</p>
    <ul>
        <li>Kinetic energy (perturbation): <code>E_U = ½ ∫ ρ₀|Ũ|² dV</code></li>
        <li>Magnetic energy (perturbation): <code>E_B = ½ ∫ |B̃|²/μ₀ dV</code></li>
        <li>Internal energy (perturbation): <code>E_p = ∫ p̃/(γ-1) dV</code> (γ = 5/3)</li>
        <li>Total perturbed energy: <code>E_tot = E_U + E_B + E_p</code></li>
        <li>Equilibrium magnetic energy: <code>E_B₀ = ½ ∫ |𝐁₀|²/μ₀ dV</code></li>
        <li>Equilibrium internal energy: <code>E_p₀ = ∫ p₀/(γ-1) dV</code></li>
        <li>Total magnetic energy: <code>E_B_tot = ½ ∫ |𝐁₀ + B̃|²/μ₀ dV</code></li>
    </ul>

    <h3>Model Properties</h3>

    <ul>
        <li><strong>Model type:</strong> Fluid</li>
        <li><strong>Velocity scale:</strong> Alfvén velocity</li>
        <li><strong>Bulk species:</strong> mhd</li>
        <li><strong>Assumptions:</strong> Zero-flow equilibrium, linear perturbations, ideal MHD</li>
    </ul>

    <h3>Key Assumptions</h3>

    <ul>
        <li>Perturbations are small (linear theory valid)</li>
        <li>Equilibrium is static: <code>𝐔₀ = 0</code></li>
        <li>Ideal MHD: infinite conductivity, no dissipation</li>
        <li>Adiabatic process: polytropic index γ = 5/3</li>
    </ul>
    """

    __doc_rst__ = r"""
Linear ideal MHD with zero-flow equilibrium for magnetohydrodynamic wave propagation.

This model simulates small-amplitude perturbations in a magnetized plasma with a static
equilibrium magnetic field :math:`\mathbf{B}_0` and zero background flow. The model solves the linearized
ideal magnetohydrodynamic (MHD) equations, which couple fluid dynamics (density, velocity, pressure)
with magnetic field evolution. The system supports both Alfvén waves (incompressible shear) and
magnetosonic waves (compressible fast and slow modes).

**Governing Equations**

**Continuity (mass conservation):**

.. math::

    \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot (\rho_0 \tilde{\mathbf{U}}) = 0

**Momentum (Lorentz force):**

.. math::

    \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde{p} = 
    (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B}_0 + (\nabla \times \mathbf{B}_0) \times \tilde{\mathbf{B}}

**Energy (adiabatic process):**

.. math::

    \frac{\partial \tilde{p}}{\partial t} + \nabla \cdot (p_0 \tilde{\mathbf{U}}) + \frac{2}{3} p_0 \nabla \cdot \tilde{\mathbf{U}} = 0

**Induction (Faraday's law):**

.. math::

    \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times (\tilde{\mathbf{U}} \times \mathbf{B}_0) = 0

**Normalization**

All velocities are normalized by the Alfvén velocity:

.. math::

    \hat{U} = \hat{v}_\mathrm{A} = \frac{\hat{B}_0}{\sqrt{\mu_0 \rho_0}}

**Perturbation Variables**

All quantities in the equations represent perturbations around equilibrium:

- :math:`\tilde{\rho}` - Density perturbation
- :math:`\tilde{\mathbf{U}}` - Velocity perturbation
- :math:`\tilde{p}` - Pressure perturbation
- :math:`\tilde{\mathbf{B}}` - Magnetic field perturbation

Equilibrium quantities (with subscript 0) are stationary:

- :math:`\rho_0` - Static background density
- :math:`p_0` - Static background pressure
- :math:`\mathbf{B}_0` - Static background magnetic field

**Species**

- ``em_fields.b_field`` - Magnetic field perturbation (H(div) space)
- ``mhd.density`` - Density perturbation (L² space)
- ``mhd.velocity`` - Velocity perturbation (H(div) space)
- ``mhd.pressure`` - Pressure perturbation (L² space)

**Wave Modes**

The linear MHD system supports three wave types:

- **Alfvén waves:** Incompressible shear waves propagating along :math:`\mathbf{B}_0` with velocity :math:`v_A`
- **Fast magnetosonic wave:** Compressible wave with phase velocity :math:`v_\mathrm{fast} > v_A`
- **Slow magnetosonic wave:** Compressible wave with phase velocity :math:`v_\mathrm{slow} < v_A`

**Propagators**

Time integration is performed by the following propagators (in sequence):

1. :class:`~struphy.propagators.propagators_fields.ShearAlfven` - Evolves Alfvén waves (velocity and magnetic field coupling)
2. :class:`~struphy.propagators.propagators_fields.Magnetosonic` - Evolves compressible modes (density, velocity, pressure)

**Scalar Quantities**

The following energies are tracked during simulation:

- Kinetic energy (perturbation): :math:`E_U = \frac{1}{2} \int \rho_0 |\tilde{\mathbf{U}}|^2 \, \mathrm{d}V`
- Magnetic energy (perturbation): :math:`E_B = \frac{1}{2} \int \frac{|\tilde{\mathbf{B}}|^2}{\mu_0} \, \mathrm{d}V`
- Internal energy (perturbation): :math:`E_p = \int \frac{\tilde{p}}{\gamma - 1} \, \mathrm{d}V` (:math:`\gamma = 5/3`)
- Total perturbed energy: :math:`E_\mathrm{tot} = E_U + E_B + E_p`
- Equilibrium magnetic energy: :math:`E_{B0} = \frac{1}{2} \int \frac{|\mathbf{B}_0|^2}{\mu_0} \, \mathrm{d}V`
- Equilibrium internal energy: :math:`E_{p0} = \int \frac{p_0}{\gamma - 1} \, \mathrm{d}V`
- Total magnetic energy: :math:`E_{B,\mathrm{tot}} = \frac{1}{2} \int \frac{|\mathbf{B}_0 + \tilde{\mathbf{B}}|^2}{\mu_0} \, \mathrm{d}V`

**Model Properties**

- **Model type:** Fluid
- **Velocity scale:** Alfvén velocity
- **Bulk species:** mhd
- **Assumptions:** Zero-flow equilibrium, linear perturbations, ideal MHD

**Key Assumptions**

- Perturbations are small (linear theory valid)
- Equilibrium is static: :math:`\mathbf{U}_0 = 0`
- Ideal MHD: infinite conductivity, no dissipation
- Adiabatic process: polytropic index :math:`\gamma = 5/3`

**See Also**

- :class:`~struphy.models.base.StruphyModel` - Base class for all Struphy models
- :class:`~struphy.propagators.propagators_fields.ShearAlfven` - Alfvén wave propagator
- :class:`~struphy.propagators.propagators_fields.Magnetosonic` - Magnetosonic wave propagator

**References**

- Boyd, T. J. M., & Sanderson, J. J. (2003). The physics of plasmas. Cambridge University Press.
- Goedbloed, J. P., Keppens, R., & Poedts, S. (2019). Magnetohydrodynamics of laboratory and astrophysical plasmas. Cambridge University Press.

**Examples**

Create and initialize a linear MHD model:

.. code-block:: python

    from struphy.models.linear_mhd import LinearMHD
    
    model = LinearMHD()
    
    # Access fields
    # model.em_fields.b_field     - Magnetic field perturbation
    # model.mhd.density           - Density perturbation
    # model.mhd.velocity          - Velocity perturbation
    # model.mhd.pressure          - Pressure perturbation
    
    # Track energies during simulation
    # model.scalar_quantities["en_U"]  - Kinetic energy
    # model.scalar_quantities["en_B"]  - Magnetic energy
    # model.scalar_quantities["en_p"]  - Internal energy
"""

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

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

    ## propagators

    class Propagators:
        def __init__(self):
            self.shear_alf = propagators_fields.ShearAlfven()
            self.mag_sonic = propagators_fields.Magnetosonic()

    ## abstract methods

    def __init__(self):

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.shear_alf.variables.u = self.mhd.velocity
        self.propagators.shear_alf.variables.b = self.em_fields.b_field

        self.propagators.mag_sonic.variables.n = self.mhd.density
        self.propagators.mag_sonic.variables.u = self.mhd.velocity
        self.propagators.mag_sonic.variables.p = self.mhd.pressure

        # define scalars for update_scalar_quantities
        self.add_scalar("en_U")
        self.add_scalar("en_p")
        self.add_scalar("en_B")
        self.add_scalar("en_p_eq")
        self.add_scalar("en_B_eq")
        self.add_scalar("en_B_tot")
        self.add_scalar("en_tot")

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

        self._tmp_b1: BlockVector = Propagator.derham.Vh["2"].zeros()  # TODO: replace derham.Vh dict by class
        self._tmp_b2: BlockVector = Propagator.derham.Vh["2"].zeros()

    def update_scalar_quantities(self):
        # perturbed fields
        en_U = 0.5 * Propagator.mass_ops.M2n.dot_inner(
            self.mhd.velocity.spline.vector,
            self.mhd.velocity.spline.vector,
        )
        en_B = 0.5 * Propagator.mass_ops.M2.dot_inner(
            self.em_fields.b_field.spline.vector,
            self.em_fields.b_field.spline.vector,
        )
        en_p = self.mhd.pressure.spline.vector.inner(self._ones) / (5 / 3 - 1)

        self.update_scalar("en_U", en_U)
        self.update_scalar("en_B", en_B)
        self.update_scalar("en_p", en_p)
        self.update_scalar("en_tot", en_U + en_B + en_p)

        # background fields
        Propagator.mass_ops.M2.dot(Propagator.projected_equil.b2, apply_bc=False, out=self._tmp_b1)

        en_B0 = Propagator.projected_equil.b2.inner(self._tmp_b1) / 2
        en_p0 = Propagator.projected_equil.p3.inner(self._ones) / (5 / 3 - 1)

        self.update_scalar("en_B_eq", en_B0)
        self.update_scalar("en_p_eq", en_p0)

        # total magnetic field
        Propagator.projected_equil.b2.copy(out=self._tmp_b1)
        self._tmp_b1 += self.em_fields.b_field.spline.vector

        Propagator.mass_ops.M2.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)

        en_Btot = self._tmp_b1.inner(self._tmp_b2) / 2

        self.update_scalar("en_B_tot", en_Btot)

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "mag_sonic.Options" in line:
                    new_file += [
                        "model.propagators.mag_sonic.options = model.propagators.mag_sonic.Options(b_field=model.em_fields.b_field)\n",
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
