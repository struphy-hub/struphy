import copy

from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, Scalars
from struphy.models.species import (
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators.variational_density_evolve import VariationalDensityEvolve
from struphy.propagators.variational_momentum_advection import VariationalMomentumAdvection

rank = MPI.COMM_WORLD.Get_rank()


class VariationalPressurelessFluid(StruphyModel):
    r"""Pressure-less fluid equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) = 0 \,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.variational_density_evolve.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.variational_momentum_advection.VariationalMomentumAdvection`

    :ref:`Model info <add_model>`:
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class Fluid(FluidSpecies):
        def __init__(self, mass_number: float = 1.0):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="H1vec")
            self.init_variables(mass_number=mass_number)

    ## propagators

    class Propagators:
        def __init__(self):
            self.variat_dens = VariationalDensityEvolve()
            self.variat_mom = VariationalMomentumAdvection()

    ## abstract methods

    def __init__(self, base_units: BaseUnits = BaseUnits(), mass_number: float = 1.0):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.fluid = self.Fluid(mass_number=mass_number)

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.variat_dens.variables.rho = self.fluid.density
        self.propagators.variat_dens.variables.u = self.fluid.velocity
        self.propagators.variat_mom.variables.u = self.fluid.velocity

        # 5. define scalars to be tracked during simulation
        kinetic_energy = BilinearEnergyFEEC(self.fluid.velocity, bilinear_form_name="WMMnew")
        self.scalars = Scalars(kinetic_energy=kinetic_energy)

    @property
    def bulk_species(self):
        return self.fluid

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self):
        pass

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "variat_dens.Options" in line:
                    new_file += [
                        "model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model='pressureless')\n",
                    ]
                elif "velocity.add_background" in line:
                    new_file += ["model.fluid.density.add_background(FieldsBackground())\n"]
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

            \partial_t \rho + \nabla \cdot (\rho \mathbf{u}) = 0

        Momentum:

        .. math::

            \partial_t (\rho \mathbf{u}) + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""The flow speed is normalized with the Alfvén speed:

        .. math::

            \hat u = \hat v_A.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy: ``kinetic_energy``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.variational_density_evolve.VariationalDensityEvolve`
        2. :class:`~struphy.propagators.variational_momentum_advection.VariationalMomentumAdvection`
        """
        doc = rf"""**1. VariationalDensityEvolve:**

{VariationalDensityEvolve.__doc__}

**2. VariationalMomentumAdvection:**

{VariationalMomentumAdvection.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""This is the pressureless limit of the variational fluid hierarchy. It
        is intended as a reduced benchmark and as a simple transport model with
        conservative density and momentum updates."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a pressureless variational-fluid model:

        .. code-block:: python

            from struphy.models import VariationalPressurelessFluid

            model = VariationalPressurelessFluid()
            model.fluid.density
            model.fluid.velocity
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - pressureless compressible benchmarks
        - testing the minimal variational fluid update chain
        - reduced transport problems without thermodynamics"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - pressure- or entropy-driven flow
        - magnetic-field coupling
        - viscous/resistive dissipation
        - kinetic particle physics"""
