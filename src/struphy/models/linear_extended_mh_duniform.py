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

rank = MPI.COMM_WORLD.Get_rank()


class LinearExtendedMHDuniform(StruphyModel):
    r"""Linear extended MHD with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).
    For uniform background conditions only.

    :ref:`normalization`:

    .. math::

        \hat U = \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,,
        \\[2mm]
        \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} + \nabla \tilde p
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0 \,,
        \\[2mm]
        &\frac{\partial \tilde p}{\partial t} + \frac{5}{3}\,p_{0}\nabla\cdot \tilde{\mathbf{U}}=0\,,
        \\[2mm]
        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times \left( \tilde{\mathbf{U}} \times \mathbf{B}_0 - \frac{1}{\varepsilon} \frac{\nabla\times \tilde{\mathbf{B}}}{\rho_0}\times \mathbf{B}_0 \right)
        = 0\,.

    where

    .. math::

        \varepsilon = \frac{1}{\hat \Omega_{\textnormal{c}} \hat t}\,,\qquad \textnormal{with} \qquad\hat \Omega_{\textnormal{c}} = \frac{Ze \hat B}{A m_\textnormal{H}}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.ShearAlfvenB1`
    2. :class:`~struphy.propagators.propagators_fields.Hall`
    3. :class:`~struphy.propagators.propagators_fields.MagnetosonicUniform`

    :ref:`Model info <add_model>`:
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.b_field = FEECVariable(space="Hcurl")
            self.init_variables()

    class MHD(FluidSpecies):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            epsilon: float = None,
        ):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="Hdiv")
            self.pressure = FEECVariable(space="L2")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                epsilon=epsilon,
            )

    ## propagators

    class Propagators:
        def __init__(self):
            self.shear_alf = propagators_fields.ShearAlfvenB1()
            self.hall = propagators_fields.Hall()
            self.mag_sonic = propagators_fields.MagnetosonicUniform()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        epsilon: float = None,
    ):

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD(
            charge_number=charge_number,
            mass_number=mass_number,
            epsilon=epsilon,
        )

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.shear_alf.variables.u = self.mhd.velocity
        self.propagators.shear_alf.variables.b = self.em_fields.b_field

        self.propagators.hall.variables.b = self.em_fields.b_field

        self.propagators.mag_sonic.variables.n = self.mhd.density
        self.propagators.mag_sonic.variables.u = self.mhd.velocity
        self.propagators.mag_sonic.variables.p = self.mhd.pressure

        # 5. define scalars to be tracked during simulation
        kinetic_energy = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="M2n")
        pressure_energy = VolumeFormEnergyFEEC(self.mhd.pressure, normalization=1.0 / (5.0 / 3.0 - 1.0))
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        background_pressure = FunctionScalarFEEC(self._compute_en_p_eq)
        background_magnetic = FunctionScalarFEEC(self._compute_en_B_eq)
        total_magnetic = FunctionScalarFEEC(self._compute_en_B_tot)
        helicity = FunctionScalarFEEC(self._compute_helicity)
        self.scalars = Scalars(
            en_U=kinetic_energy,
            en_p=pressure_energy,
            en_B=magnetic_energy,
            en_p_eq=background_pressure,
            en_B_eq=background_magnetic,
            en_B_tot=total_magnetic,
            en_tot=kinetic_energy + pressure_energy + magnetic_energy,
            helicity=helicity,
        )

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        This model solves linear extended-MHD perturbations around a uniform
        equilibrium. It contains:

        - density, velocity, and pressure perturbations
        - magnetic perturbation evolution
        - Hall physics through the :math:`1/\varepsilon` correction

        The implementation assumes zero equilibrium flow and uniform
        background conditions."""

    @classmethod
    def doc_normalization(cls):
        r"""All velocities are normalized by the Alfvén speed,

        .. math::

            \hat U = \hat v_A,

        and Hall effects enter through
        :math:`\varepsilon = 1/(\hat\Omega_c\hat t)`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Flow kinetic energy: ``en_U``
        - Thermal pressure energy: ``en_p``
        - Magnetic perturbation energy: ``en_B``
        - Background pressure energy: ``en_p_eq``
        - Background magnetic energy: ``en_B_eq``
        - Total magnetic energy: ``en_B_tot``
        - Total perturbation energy: ``en_tot``
        - Magnetic helicity-like invariant: ``helicity``"""

    @classmethod
    def doc_discretization(cls):
        doc = rf"""**1. propagators_fields.ShearAlfvenB1:**

{propagators_fields.ShearAlfvenB1.__doc__}

**2. propagators_fields.Hall:**

{propagators_fields.Hall.__doc__}

**3. propagators_fields.MagnetosonicUniform:**

{propagators_fields.MagnetosonicUniform.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""LinearExtendedMHDuniform extends the linear MHD model with Hall
        dynamics for a uniform equilibrium. It is targeted at wave propagation
        and mode studies where the equilibrium is simple but finite Hall
        corrections are important."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a linear extended-MHD model:

        .. code-block:: python

            from struphy.models import LinearExtendedMHDuniform

            model = LinearExtendedMHDuniform()
            model.em_fields.b_field
            model.mhd.density
            model.mhd.velocity
            model.mhd.pressure
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - linear Hall-MHD wave studies
        - uniform-equilibrium benchmarks for extended MHD
        - comparison against ideal-MHD dispersion in the Hall-corrected regime"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - nonlinear extended-MHD turbulence
        - non-uniform equilibrium configurations
        - kinetic ion/electron effects beyond the Hall correction
        - dissipation-dominated problems with explicit viscosity or resistivity"""

    def allocate_helpers(self, verbose: bool = False):
        self._b_eq = Propagator.projected_equil.b1
        self._a_eq = Propagator.projected_equil.a1
        self._p_eq = Propagator.projected_equil.p3

        self._ones = Propagator.projected_equil.p3.space.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        self._tmp_b1: BlockVector = Propagator.derham.V1.zeros()
        self._tmp_b2: BlockVector = Propagator.derham.V1.zeros()

        # adjust coupling parameters
        epsilon = self.mhd.equation_params.epsilon

        if abs(epsilon - 1) < 1e-6:
            self.mhd.equation_params.epsilon = 1.0

    def _compute_helicity(self):
        u = self.mhd.velocity.spline.vector
        p = self.mhd.pressure.spline.vector
        b = self.em_fields.b_field.spline.vector

        b1 = Propagator.mass_ops.M1.dot(b, out=self._tmp_b1)
        return 2.0 * self._a_eq.inner(b1)

    def _compute_en_B_eq(self):
        b1 = Propagator.mass_ops.M1.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)
        return self._b_eq.inner(b1) / 2.0

    def _compute_en_p_eq(self):
        return self._p_eq.inner(self._ones) / (5.0 / 3.0 - 1.0)

    def _compute_en_B_tot(self):
        b = self.em_fields.b_field.spline.vector
        b1 = self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += b

        b2 = Propagator.mass_ops.M1.dot(b1, apply_bc=False, out=self._tmp_b2)
        return b1.inner(b2) / 2.0

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "hall.Options" in line:
                    new_file += [
                        "model.propagators.hall.options = model.propagators.hall.Options(epsilon_from=model.mhd)\n",
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
