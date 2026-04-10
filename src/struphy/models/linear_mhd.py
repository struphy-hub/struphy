from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.block import BlockVector

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC, Scalars, VolumeFormEnergyFEEC
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
from struphy.utils.docstring_converter import auto_convert_docstring

rank = MPI.COMM_WORLD.Get_rank()


@auto_convert_docstring
class LinearMHD(StruphyModel):
    """Linear ideal MHD with zero-flow equilibrium for magnetohydrodynamic wave propagation.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mass_number: float
        Mass number (in units of Proton mass) of the ion species (default: 1.0)
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
        def __init__(
            self,
            mass_number: float = 1.0,
        ):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="Hdiv")
            self.pressure = FEECVariable(space="L2")
            self.init_variables(mass_number=mass_number)

    ## propagators

    class Propagators:
        def __init__(self):
            self.shear_alf = propagators_fields.ShearAlfven()
            self.mag_sonic = propagators_fields.Magnetosonic()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mass_number: float = 1.0,
    ):

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD(mass_number)

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.shear_alf.variables.u = self.mhd.velocity
        self.propagators.shear_alf.variables.b = self.em_fields.b_field

        self.propagators.mag_sonic.variables.n = self.mhd.density
        self.propagators.mag_sonic.variables.u = self.mhd.velocity
        self.propagators.mag_sonic.variables.p = self.mhd.pressure

        # 5. define scalars to be tracked during simulation
        kinetic_energy = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="M2n")
        pressure_energy = VolumeFormEnergyFEEC(self.mhd.pressure, normalization=1.0 / (5 / 3 - 1))
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        background_pressure = FunctionScalarFEEC(self._compute_en_p_eq)
        background_magnetic = FunctionScalarFEEC(self._compute_en_B_eq)
        total_magnetic = FunctionScalarFEEC(self._compute_en_B_tot)
        self.scalars = Scalars(
            en_U=kinetic_energy,
            en_p=pressure_energy,
            en_B=magnetic_energy,
            en_p_eq=background_pressure,
            en_B_eq=background_magnetic,
            en_B_tot=total_magnetic,
            en_tot=kinetic_energy + pressure_energy + magnetic_energy,
        )

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

        self._tmp_b1: BlockVector = Propagator.derham.V2.zeros()
        self._tmp_b2: BlockVector = Propagator.derham.V2.zeros()

    def _compute_en_B_eq(self):
        Propagator.mass_ops.M2.dot(Propagator.projected_equil.b2, apply_bc=False, out=self._tmp_b1)
        return Propagator.projected_equil.b2.inner(self._tmp_b1) / 2

    def _compute_en_p_eq(self):
        return Propagator.projected_equil.p3.inner(self._ones) / (5 / 3 - 1)

    def _compute_en_B_tot(self):
        Propagator.projected_equil.b2.copy(out=self._tmp_b1)
        self._tmp_b1 += self.em_fields.b_field.spline.vector

        Propagator.mass_ops.M2.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)
        return self._tmp_b1.inner(self._tmp_b2) / 2

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

    __doc_rst__ = r"""
Linear ideal MHD with zero-flow equilibrium for magnetohydrodynamic wave propagation.

This model simulates small-amplitude perturbations in a magnetized plasma with a static
equilibrium magnetic field :math:`\mathbf{B}_0` and zero background flow. The model solves the linearized
ideal magnetohydrodynamic (MHD) equations, which couple fluid dynamics (density, velocity, pressure)
with magnetic field evolution. The system supports both Alfvén waves (incompressible shear) and
magnetosonic waves (compressible fast and slow modes).

**Governing Equations**

Continuity (mass conservation):

.. math::

    \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot (\rho_0 \tilde{\mathbf{U}}) = 0

Momentum (Lorentz force):

.. math::

    \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde{p} = 
    (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B}_0 + (\nabla \times \mathbf{B}_0) \times \tilde{\mathbf{B}}

Energy (adiabatic process):

.. math::

    \frac{\partial \tilde{p}}{\partial t} + \nabla \cdot (p_0 \tilde{\mathbf{U}}) + \frac{2}{3} p_0 \nabla \cdot \tilde{\mathbf{U}} = 0

Induction (Faraday's law):

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
