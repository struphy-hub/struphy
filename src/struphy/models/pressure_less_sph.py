from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import KineticEnergySPH, Scalars
from struphy.models.species import (
    ParticleSpecies,
)
from struphy.models.variables import SPHVariable
from struphy.propagators import (
    propagators_markers,
)

rank = MPI.COMM_WORLD.Get_rank()


class PressureLessSPH(StruphyModel):
    r"""Pressureless fluid discretized with smoothed particle hydrodynamics

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) = - \nabla \phi_0 \,,

    where :math:`\phi_0` is a static external potential.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushEta`

    This is discretized by particles going in straight lines.
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class ColdFluid(ParticleSpecies):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            epsilon: float = None,
        ):
            self.var = SPHVariable()
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                epsilon=epsilon,
            )

    ## propagators

    class Propagators:
        def __init__(self):
            self.push_eta = propagators_markers.PushEta()
            self.push_v = propagators_markers.PushVinEfield()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        epsilon: float = None,
    ):

        # 1. instantiate all species
        self.cold_fluid = self.ColdFluid(
            charge_number=charge_number,
            mass_number=mass_number,
            epsilon=epsilon,
        )

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.push_eta.variables.var = self.cold_fluid.var
        self.propagators.push_v.variables.var = self.cold_fluid.var

        # 5. define scalars to be tracked during simulation
        self.scalars = Scalars(
            en_kin=KineticEnergySPH(self.cold_fluid.var),
        )

    @property
    def bulk_species(self):
        return self.cold_fluid

    @property
    def velocity_scale(self):
        return None

    # @staticmethod
    # def diagnostics_dct():
    #     dct = {}
    #     dct["projected_density"] = "L2"
    #     return dct

    def allocate_helpers(self, verbose: bool = False):
        pass

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "push_v.Options" in line:
                    new_file += ["phi = equil.p0\n"]
                    new_file += ["model.propagators.push_v.options = model.propagators.push_v.Options(phi=phi)\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
