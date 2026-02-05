from feectools.ddm.mpi import mpi as MPI
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
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
        def __init__(self):
            self.var = SPHVariable()
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.push_eta = propagators_markers.PushEta()
            self.push_v = propagators_markers.PushVinEfield()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.cold_fluid = self.ColdFluid()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.push_eta.variables.var = self.cold_fluid.var
        self.propagators.push_v.variables.var = self.cold_fluid.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_kin", compute="from_particles", variable=self.cold_fluid.var)

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

    def update_scalar_quantities(self):
        particles = self.cold_fluid.var.particles
        valid_parts = particles.markers_wo_holes_and_ghost
        en_kin = valid_parts[:, 6].dot(valid_parts[:, 3] ** 2 + valid_parts[:, 4] ** 2 + valid_parts[:, 5] ** 2) / (
            2.0 * particles.Np
        )

        self.update_scalar("en_kin", en_kin)

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
