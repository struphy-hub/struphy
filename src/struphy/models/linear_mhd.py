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
from struphy.propagators.base import Propagator
from struphy.propagators.magnetosonic import Magnetosonic
from struphy.propagators.shear_alfven_propagator import ShearAlfvenPropagator

rank = MPI.COMM_WORLD.Get_rank()


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
            self.shear_alf = ShearAlfvenPropagator()
            self.mag_sonic = Magnetosonic()

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

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity (mass conservation):

        .. math::

            \frac{\partial \tilde{\rho}}{\partial t} + \nabla \cdot (\rho_0 \tilde{\mathbf{U}}) = 0

        Momentum (Lorentz force):

        .. math::

            \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde{p} = (\nabla \times \tilde{\mathbf{B}}) \times \mathbf{B}_0 + (\nabla \times \mathbf{B}_0) \times \tilde{\mathbf{B}}

        Energy (adiabatic process):

        .. math::

            \frac{\partial \tilde{p}}{\partial t} + \nabla \cdot (p_0 \tilde{\mathbf{U}}) + \frac{2}{3} p_0 \nabla \cdot \tilde{\mathbf{U}} = 0

        Induction (Faraday's law):

        .. math::

            \frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla \times (\tilde{\mathbf{U}} \times \mathbf{B}_0) = 0.
        """

    @classmethod
    def doc_normalization(cls):
        r"""All velocities are normalized by the Alfvén velocity:

        .. math::

            \hat{U} = \hat{v}_\mathrm{A} = \frac{\hat{B}_0}{\sqrt{\mu_0 \rho_0}}

        The model therefore uses the Alfvén velocity as its characteristic speed scale."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy (perturbation): :math:`E_U = \frac{1}{2} \int \rho_0 |\tilde{\mathbf{U}}|^2 \, \mathrm{d}V`
        - Magnetic energy (perturbation): :math:`E_B = \frac{1}{2} \int \frac{|\tilde{\mathbf{B}}|^2}{\mu_0} \, \mathrm{d}V`
        - Internal energy (perturbation): :math:`E_p = \int \frac{\tilde{p}}{\gamma - 1} \, \mathrm{d}V` with :math:`\gamma = 5/3`
        - Total perturbed energy: :math:`E_{\mathrm{tot}} = E_U + E_B + E_p`
        - Equilibrium magnetic energy: :math:`E_{B0} = \frac{1}{2} \int \frac{|\mathbf{B}_0|^2}{\mu_0} \, \mathrm{d}V`
        - Equilibrium internal energy: :math:`E_{p0} = \int \frac{p_0}{\gamma - 1} \, \mathrm{d}V`
        - Total magnetic energy: :math:`E_{B,\mathrm{tot}} = \frac{1}{2} \int \frac{|\mathbf{B}_0 + \tilde{\mathbf{B}}|^2}{\mu_0} \, \mathrm{d}V`"""

    @classmethod
    def doc_discretization(cls):
        """**Propagators:**

        1. :class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator`
        2. :class:`~struphy.propagators.magnetosonic.Magnetosonic`"""
        doc = rf"""**1. ShearAlfvenPropagator:**

{ShearAlfvenPropagator.__doc__}

**2. Magnetosonic:**

{Magnetosonic.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""This model simulates small-amplitude perturbations in a magnetized plasma with a static
        equilibrium magnetic field :math:`\mathbf{B}_0` and zero background flow. The model solves the linearized
        ideal magnetohydrodynamic equations, which couple fluid dynamics (density, velocity, pressure)
        with magnetic field evolution.

        The linear MHD system supports three wave families:

        - **Alfvén waves:** incompressible shear waves propagating along :math:`\mathbf{B}_0`
        - **Fast magnetosonic waves:** compressible waves with phase velocity above the Alfvén speed
        - **Slow magnetosonic waves:** compressible waves with phase velocity below the Alfvén speed

        All evolved quantities are perturbations around a stationary equilibrium:

        - :math:`\tilde{\rho}` - density perturbation
        - :math:`\tilde{\mathbf{U}}` - velocity perturbation
        - :math:`\tilde{p}` - pressure perturbation
        - :math:`\tilde{\mathbf{B}}` - magnetic field perturbation

        The corresponding equilibrium quantities are:

        - :math:`\rho_0` - background density
        - :math:`p_0` - background pressure
        - :math:`\mathbf{B}_0` - background magnetic field"""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a linear MHD model:

        .. code-block:: python

            from struphy.models import LinearMHD

            model = LinearMHD()

            # Access fields
            model.em_fields.b_field
            model.mhd.density
            model.mhd.velocity
            model.mhd.pressure

            # Access tracked scalar quantities
            model.scalars["en_U"]
            model.scalars["en_B"]
            model.scalars["en_p"]
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for studying linear wave propagation in a static magnetized plasma,
        including Alfvén and magnetosonic dynamics around a prescribed equilibrium.

        Typical use cases include:

        - verification of linear MHD wave dispersion and mode structure
        - perturbative studies around static equilibria
        - benchmark problems for ideal-MHD field-fluid coupling"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - nonlinear MHD dynamics with finite-amplitude perturbations
        - equilibria with non-zero background flow
        - dissipative effects such as resistivity or viscosity
        - kinetic or particle-field effects beyond the fluid MHD description"""

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
