import copy

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.feec.mass import L2Projector
from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC, Scalars
from struphy.models.species import (
    DiagnosticSpecies,
    FieldSpecies,
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.propagators.variational_density_evolve import VariationalDensityEvolve
from struphy.propagators.variational_qb_evolve import VariationalQBEvolve
from struphy.propagators.variational_resistivity import VariationalResistivity
from struphy.propagators.variational_viscosity import VariationalViscosity

rank = MPI.COMM_WORLD.Get_rank()


class ViscoResistiveLinearMHD_with_q(StruphyModel):
    """Linear visco-resistive MHD equations, with the q variable (square root of the pressure), discretized with a variational method.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mass_number: float
        Mass number (in units of Proton mass) of the fluid species (default: 1.0)
    with_viscosity: bool
        Whether to include viscous dissipation (default: True)
    with_resistivity: bool
        Whether to include resistive dissipation (default: True)
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
        def __init__(self, mass_number: float = 1.0):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="H1vec")
            self.sqrt_p = FEECVariable(space="L2")
            self.init_variables(mass_number=mass_number)

    class Diagnostics(DiagnosticSpecies):
        def __init__(self):
            self.div_u = FEECVariable(space="L2")
            self.u2 = FEECVariable(space="Hdiv")
            self.qt3 = FEECVariable(space="L2")
            self.bt2 = FEECVariable(space="Hdiv")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(
            self,
            div_u: FEECVariable = None,
            u2: FEECVariable = None,
            qt3: FEECVariable = None,
            bt2: FEECVariable = None,
            rho: FEECVariable = None,
            with_viscosity: bool = True,
            with_resistivity: bool = True,
        ):
            self.variat_dens = VariationalDensityEvolve()
            self.variat_qb = VariationalQBEvolve(div_u=div_u, u2=u2, qt3=qt3, bt2=bt2)
            if with_viscosity:
                self.variat_viscous = VariationalViscosity(rho=rho, pt3=qt3)
            if with_resistivity:
                self.variat_resist = VariationalResistivity(rho=rho, pt3=qt3)

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mass_number: float = 1.0,
        with_viscosity: bool = True,
        with_resistivity: bool = True,
    ):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD(mass_number=mass_number)
        self.diagnostics = self.Diagnostics()

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(
            div_u=self.diagnostics.div_u,
            u2=self.diagnostics.u2,
            qt3=self.diagnostics.qt3,
            bt2=self.diagnostics.bt2,
            rho=self.mhd.density,
            with_viscosity=with_viscosity,
            with_resistivity=with_resistivity,
        )

        # 4. assign variables to propagators
        self.propagators.variat_dens.variables.rho = self.mhd.density
        self.propagators.variat_dens.variables.u = self.mhd.velocity
        self.propagators.variat_qb.variables.u = self.mhd.velocity
        self.propagators.variat_qb.variables.q = self.mhd.sqrt_p
        self.propagators.variat_qb.variables.b = self.em_fields.b_field
        if with_viscosity:
            self.propagators.variat_viscous.variables.s = self.mhd.sqrt_p
            self.propagators.variat_viscous.variables.u = self.mhd.velocity
        if with_resistivity:
            self.propagators.variat_resist.variables.s = self.mhd.sqrt_p
            self.propagators.variat_resist.variables.b = self.em_fields.b_field

        # 5. define scalars to be tracked during simulation
        kinetic_energy = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="WMMnew")
        magnetic_energy_1 = BilinearEnergyFEEC(self.em_fields.b_field)
        magnetic_energy_2 = BilinearEnergyFEEC(self.diagnostics.bt2, right_variable="b2")
        thermo_energy_1 = FunctionScalarFEEC(self._compute_en_thermo_1)
        thermo_energy_2 = FunctionScalarFEEC(self._compute_en_thermo_2)
        self.scalars = Scalars(
            en_U=kinetic_energy,
            en_mag_1=magnetic_energy_1,
            en_mag_2=magnetic_energy_2,
            en_thermo_1=thermo_energy_1,
            en_thermo_2=thermo_energy_2,
            en_tot=kinetic_energy + magnetic_energy_1 + magnetic_energy_2 + thermo_energy_1 + thermo_energy_2,
        )

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self):
        projV3 = L2Projector("L2", Propagator.mass_ops)

        def f(e1, e2, e3):
            return 1

        f = xp.vectorize(f)
        self._integrator = projV3(f)

        self._ones = Propagator.derham.V3pol.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

        self._tmp_div_B = Propagator.derham.V3pol.zeros()

    def _compute_en_thermo_1(self):
        q = self.mhd.sqrt_p.spline.vector
        gamma = self.propagators.variat_qb.options.gamma
        return 1.0 / (gamma - 1.0) * Propagator.mass_ops.M3.dot_inner(q, q)

    def _compute_en_thermo_2(self):
        qt3 = self.propagators.variat_qb.qt3.spline.vector
        gamma = self.propagators.variat_qb.options.gamma
        return 2.0 / (gamma - 1.0) * Propagator.mass_ops.M3.dot_inner(qt3, Propagator.projected_equil.q3)

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "variat_dens.Options" in line:
                    new_file += [
                        "model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model='linear_q')\n",
                    ]
                elif "variat_qb.Options" in line:
                    new_file += [
                        "model.propagators.variat_qb.options = model.propagators.variat_qb.Options(model='linear_q')\n",
                    ]
                elif "variat_viscous.Options" in line:
                    new_file += [
                        "model.propagators.variat_viscous.options = model.propagators.variat_viscous.Options(model='linear_q')\n",
                    ]
                elif "variat_resist.Options" in line:
                    new_file += [
                        "model.propagators.variat_resist.options = model.propagators.variat_resist.Options(model='linear_q')\n",
                    ]
                elif "sqrt_p.add_background" in line:
                    new_file += ["model.mhd.density.add_background(FieldsBackground())\n"]
                    new_file += [line]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity:

        .. math::

            \partial_t \tilde{\rho} + \nabla \cdot (\rho_0 \tilde{\mathbf{u}}) = 0

        Momentum:

        .. math::

            \partial_t (\rho_0 \tilde{\mathbf{u}}) + \frac{2 q_0}{\gamma - 1} \nabla \tilde{q} + \frac{2 \tilde{q}}{\gamma - 1} \nabla q_0 + \mathbf{B}_0 \times \nabla \times \tilde{\mathbf{B}} + \tilde{\mathbf{B}} \times \nabla \times \mathbf{B}_0 - \nabla \cdot \left( (\mu + \mu_a(\mathbf{x})) \nabla \tilde{\mathbf{u}} \right) = 0

        Energy-like variable:

        .. math::

            \partial_t \tilde{q} + \nabla q_0 \cdot \tilde{\mathbf{u}} + \left( \frac{\gamma}{2} - 1 \right) q_0 \nabla \cdot \tilde{\mathbf{u}} = 0

        Induction:

        .. math::

            \partial_t \tilde{\mathbf{B}} + \nabla \times (\mathbf{B}_0 \times \tilde{\mathbf{u}}) + \nabla \times (\eta + \eta_a(\mathbf{x})) \nabla \times \tilde{\mathbf{B}} = 0

        Here :math:`\mu_a(\mathbf{x})` and :math:`\eta_a(\mathbf{x})` are artificial viscosity and resistivity coefficients.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The flow normalization is Alfvénic:

        .. math::

            \hat u = \hat v_A.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy: ``en_U``
        - Magnetic energies: ``en_mag_1``, ``en_mag_2``
        - Thermodynamic q-energies: ``en_thermo_1``, ``en_thermo_2``
        - Total energy: ``en_tot``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.variational_density_evolve.VariationalDensityEvolve`
        2. :class:`~struphy.propagators.variational_qb_evolve.VariationalQBEvolve`
        3. :class:`~struphy.propagators.variational_viscosity.VariationalViscosity` (if :attr:`with_viscosity` is True)
        4. :class:`~struphy.propagators.variational_resistivity.VariationalResistivity` (if :attr:`with_resistivity` is True)
        """
        doc = rf"""**1. VariationalDensityEvolve:**

{VariationalDensityEvolve.__doc__}

**2. VariationalQBEvolve:**

{VariationalQBEvolve.__doc__}

**3. VariationalViscosity:**

{VariationalViscosity.__doc__}

**4. VariationalResistivity:**

{VariationalResistivity.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""This variant of the linear dissipative MHD model uses the square root
        of the pressure as primary thermodynamic variable, which is convenient
        for the corresponding variational discretization."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the linear visco-resistive q-MHD model:

        .. code-block:: python

            from struphy.models import ViscoResistiveLinearMHD_with_q

            model = ViscoResistiveLinearMHD_with_q()
            model.mhd.sqrt_p
            model.em_fields.b_field
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - linear dissipative MHD in the q-formulation
        - comparing p- and q-based variational discretizations
        - verification of linear q/B propagators"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - nonlinear MHD dynamics
        - entropy-based thermodynamics
        - kinetic plasma coupling
        - ideal nondissipative studies"""
