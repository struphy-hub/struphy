import copy

from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import KineticEnergySPH, Scalars
from struphy.models.species import (
    ParticleSpecies,
)
from struphy.models.variables import SPHVariable
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_vin_efield import PushVinEfield

rank = MPI.COMM_WORLD.Get_rank()


class PressureLessSPH(StruphyModel):
    r"""Particle discretization of pressureless Euler flow with external forcing.

    Parameters
    ----------
    base_units : BaseUnits, optional
        Reference units used to derive model normalization constants.
    charge_number : int, optional
        Species charge number (default is 1).
    mass_number : float, optional
        Species mass number (default is 1.0).
    epsilon : float, optional
        Scaling parameter for force field (default is None).
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
            self.push_eta = PushEta()
            self.push_v = PushVinEfield()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        epsilon: float = None,
    ):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

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

    def allocate_helpers(self):
        pass

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "push_v.Options" in line:
                    new_file += ["phi = equil.p0\n"]
                    new_file += ["model.propagators.push_v.phi = phi\n"]
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

            \partial_t (\rho \mathbf{u}) + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) = \mathbf{F}

        where :math:`\mathbf{F}` is an external force.
        """

    @classmethod
    def doc_normalization(cls):
        r"""Velocity and field normalizations:

        .. math::

            \hat{u} = 1\,\textrm{m/s}, \qquad \hat{F} = \frac{m\hat{n}\hat{u}}{\hat{t}}\,.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - SPH kinetic energy: ``en_kin``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.push_eta.PushEta`
        2. :class:`~struphy.propagators.push_vin_efield.PushVinEfield`
        """
        doc = rf"""**1. PushEta:**

    {PushEta.__doc__}

    **2. PushVinEfield:**

    {PushVinEfield.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""PressureLessSPH is a meshfree particle model for a pressureless fluid.
        It is primarily useful as a simple SPH benchmark or as a reduced
        particle transport model in a prescribed force field or potential."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a pressureless SPH model:

        .. code-block:: python

            from struphy.models import PressureLessSPH

            model = PressureLessSPH()
            model.cold_fluid.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - meshfree pressureless-flow benchmarks
        - straight-line particle transport with external forcing
        - testing SPH particle loading and diagnostics"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - compressible fluids with pressure forces
        - FEEC-based grid discretizations
        - electromagnetic plasma dynamics
        - viscous or thermal closures"""
