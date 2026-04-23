from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, Scalars
from struphy.models.species import (
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators import (
    propagators_fields,
)

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

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`

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
        def __init__(self, rho: FEECVariable):
            self.variat_dens = propagators_fields.VariationalDensityEvolve()
            self.variat_mom = propagators_fields.VariationalMomentumAdvection(rho=rho)

    ## abstract methods

    def __init__(self, base_units: BaseUnits = BaseUnits(), mass_number: float = 1.0):

        # 1. instantiate all species
        self.fluid = self.Fluid(mass_number=mass_number)

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(rho=self.fluid.density)

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

    def allocate_helpers(self, verbose: bool = False):
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
                    new_file += [
                        "model.fluid.density.add_background(FieldsBackground())\n"
                    ]
                    new_file += [line]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
