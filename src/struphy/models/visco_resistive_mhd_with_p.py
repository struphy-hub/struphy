import copy

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.feec.mass import L2Projector
from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC, Scalars, VolumeFormEnergyFEEC
from struphy.models.species import (
    DiagnosticSpecies,
    FieldSpecies,
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.propagators.variational_density_evolve import VariationalDensityEvolve
from struphy.propagators.variational_momentum_advection import VariationalMomentumAdvection
from struphy.propagators.variational_pb_evolve import VariationalPBEvolve
from struphy.propagators.variational_resistivity import VariationalResistivity
from struphy.propagators.variational_viscosity import VariationalViscosity

rank = MPI.COMM_WORLD.Get_rank()


class ViscoResistiveMHD_with_p(StruphyModel):
    r"""Full (non-linear) visco-resistive MHD equations, with the pressure variable discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \frac{1}{\gamma -1} \nabla p + \mathbf B \times \nabla \times \mathbf B - \nabla \cdot \left((\mu+\mu_a(\mathbf x)) \nabla \mathbf u \right) = 0 \,,
        \\[4mm]
        &\partial_t p + u \cdot \nabla p + \gamma p \nabla \cdot u = \frac{1}{(\gamma -1)}\left((\mu+\mu_a(\mathbf x)) |\nabla \mathbf u|^2 + (\eta + \eta_a(\mathbf x)) |\nabla \times \mathbf B|^2\right) \,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \times ( \mathbf B \times \mathbf u ) + \nabla \times (\eta + \eta_a(\mathbf x)) \nabla \times \mathbf B = 0 \,,

    and :math:`\mu_a(\mathbf x)` and :math:`\eta_a(\mathbf x)` are artificial viscosity and resistivity coefficients.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.variational_density_evolve.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.variational_momentum_advection.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.variational_pb_evolve.VariationalPBEvolve`
    4. :class:`~struphy.propagators.variational_viscosity.VariationalViscosity`
    5. :class:`~struphy.propagators.variational_resistivity.VariationalResistivity`

    :ref:`Model info <add_model>`:
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
            self.pressure = FEECVariable(space="L2")
            self.init_variables(mass_number=mass_number)

    class Diagnostics(DiagnosticSpecies):
        def __init__(self):
            self.div_u = FEECVariable(space="L2")
            self.u2 = FEECVariable(space="Hdiv")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(
            self,
            with_viscosity: bool = True,
            with_resistivity: bool = True,
        ):
            self.variat_dens = VariationalDensityEvolve()
            self.variat_mom = VariationalMomentumAdvection()
            self.variat_pb = VariationalPBEvolve()
            if with_viscosity:
                self.variat_viscous = VariationalViscosity()
            if with_resistivity:
                self.variat_resist = VariationalResistivity()

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
            with_viscosity=with_viscosity,
            with_resistivity=with_resistivity,
        )

        # 4. assign variables to propagators
        self.propagators.variat_dens.variables.rho = self.mhd.density
        self.propagators.variat_dens.variables.u = self.mhd.velocity
        self.propagators.variat_mom.variables.u = self.mhd.velocity
        self.propagators.variat_pb.variables.u = self.mhd.velocity
        self.propagators.variat_pb.variables.p = self.mhd.pressure
        self.propagators.variat_pb.variables.b = self.em_fields.b_field
        if with_viscosity:
            self.propagators.variat_viscous.variables.s = self.mhd.pressure
            self.propagators.variat_viscous.variables.u = self.mhd.velocity
        if with_resistivity:
            self.propagators.variat_resist.variables.s = self.mhd.pressure
            self.propagators.variat_resist.variables.b = self.em_fields.b_field

        # 5. define scalars to be tracked during simulation
        kinetic_energy = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="WMMnew")
        thermo_energy = FunctionScalarFEEC(self._compute_en_thermo)
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        density_total = VolumeFormEnergyFEEC(self.mhd.density)
        div_b_total = FunctionScalarFEEC(self._compute_tot_div_B)
        self.scalars = Scalars(
            en_U=kinetic_energy,
            en_thermo=thermo_energy,
            en_mag=magnetic_energy,
            en_tot=kinetic_energy + thermo_energy + magnetic_energy,
            dens_tot=density_total,
            tot_div_B=div_b_total,
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

        Continuity:

        .. math::

            \partial_t \rho + \nabla \cdot (\rho \mathbf{u}) = 0

        Momentum:

        .. math::

            \partial_t (\rho \mathbf{u}) + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) + \frac{1}{\gamma - 1} \nabla p + \mathbf{B} \times \nabla \times \mathbf{B} - \nabla \cdot \left( (\mu + \mu_a(\mathbf{x})) \nabla \mathbf{u} \right) = 0

        Pressure:

        .. math::

            \partial_t p + u \cdot \nabla p + \gamma p \nabla \cdot u = \frac{1}{\gamma - 1} \left( (\mu + \mu_a(\mathbf{x})) |\nabla \mathbf{u}|^2 + (\eta + \eta_a(\mathbf{x})) |\nabla \times \mathbf{B}|^2 \right)

        Induction:

        .. math::

            \partial_t \mathbf{B} + \nabla \times (\mathbf{B} \times \mathbf{u}) + \nabla \times (\eta + \eta_a(\mathbf{x})) \nabla \times \mathbf{B} = 0

        Here :math:`\mu_a(\mathbf{x})` and :math:`\eta_a(\mathbf{x})` are artificial viscosity and resistivity coefficients.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The characteristic flow speed is the Alfvén speed,

        .. math::

            \hat u = \hat v_A.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy: ``en_U``
        - Thermodynamic energy: ``en_thermo``
        - Magnetic energy: ``en_mag``
        - Total energy: ``en_tot``
        - Total density: ``dens_tot``
        - Divergence diagnostic: ``tot_div_B``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.variational_density_evolve.VariationalDensityEvolve`
        2. :class:`~struphy.propagators.variational_momentum_advection.VariationalMomentumAdvection`
        3. :class:`~struphy.propagators.variational_pb_evolve.VariationalPBEvolve`
        4. :class:`~struphy.propagators.variational_viscosity.VariationalViscosity` (if :attr:`with_viscosity` is True)
        5. :class:`~struphy.propagators.variational_resistivity.VariationalResistivity` (if :attr:`with_resistivity` is True)
        """
        doc = rf"""**1. VariationalDensityEvolve:**

{VariationalDensityEvolve.__doc__}

**2. VariationalMomentumAdvection:**

{VariationalMomentumAdvection.__doc__}

**3. VariationalPBEvolve:**

{VariationalPBEvolve.__doc__}

**4. VariationalViscosity:**

{VariationalViscosity.__doc__}

**5. VariationalResistivity:**

{VariationalResistivity.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""This is the pressure-based full nonlinear dissipative MHD model. It is
        the natural counterpart to the entropy- and q-based full MHD variants
        when pressure is preferred as primitive thermodynamic variable."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the pressure-based visco-resistive MHD model:

        .. code-block:: python

            from struphy.models import ViscoResistiveMHD_with_p

            model = ViscoResistiveMHD_with_p()
            model.mhd.pressure
            model.em_fields.b_field
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - full nonlinear dissipative MHD in p-formulation
        - comparison against entropy- and q-based full MHD models
        - variational resistive/viscous MHD benchmarks"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - ideal MHD without dissipation
        - kinetic plasma physics
        - reduced linear perturbation studies
        - thermodynamic formulations that require entropy or q as primitive variables"""

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

    def _compute_en_thermo(self):
        p = self.mhd.pressure.spline.vector
        gamma = self.propagators.variat_pb.options.gamma
        return Propagator.mass_ops.M3.dot_inner(p, self._integrator) / (gamma - 1.0)

    def _compute_tot_div_B(self):
        b = self.em_fields.b_field.spline.vector
        div_B = Propagator.derham.div.dot(b, out=self._tmp_div_B)
        return Propagator.mass_ops.M3.dot_inner(div_B, div_B)

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "variat_pb.Options" in line:
                    new_file += [
                        "model.propagators.variat_pb.options = model.propagators.variat_pb.Options(div_u=model.diagnostics.div_u,\n",
                    ]
                    new_file += [
                        "                                                                          u2=model.diagnostics.u2)\n",
                    ]
                elif "variat_viscous.Options" in line:
                    new_file += [
                        "model.propagators.variat_viscous.options = model.propagators.variat_viscous.Options(rho=model.mhd.density)\n",
                    ]
                elif "variat_resist.Options" in line:
                    new_file += [
                        "model.propagators.variat_resist.options = model.propagators.variat_resist.Options(rho=model.mhd.density)\n",
                    ]
                elif "pressure.add_background" in line:
                    new_file += ["model.mhd.density.add_background(FieldsBackground())\n"]
                    new_file += [line]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
