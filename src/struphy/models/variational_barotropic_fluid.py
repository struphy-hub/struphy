from feectools.ddm.mpi import mpi as MPI
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators import (
    propagators_fields,
)
from struphy.propagators.base import Propagator

rank = MPI.COMM_WORLD.Get_rank()


class VariationalBarotropicFluid(StruphyModel):
    r"""Barotropic fluid equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A} \qquad \hat{\mathcal U} = \frac{\hat \rho}{2} \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \rho \nabla \frac{(\rho \mathcal U (\rho))}{\partial \rho} = 0 \,.

    where the internal energy per unit mass is :math:`\mathcal U(\rho) = \rho/2`.

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
        def __init__(self):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="H1vec")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.variat_dens = propagators_fields.VariationalDensityEvolve()
            self.variat_mom = propagators_fields.VariationalMomentumAdvection()

    ## abstract methods

    def __init__(self):

        # 1. instantiate all species
        self.fluid = self.Fluid()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.variat_dens.variables.rho = self.fluid.density
        self.propagators.variat_dens.variables.u = self.fluid.velocity
        self.propagators.variat_mom.variables.u = self.fluid.velocity

        # define scalars for update_scalar_quantities
        self.add_scalar("en_U")
        self.add_scalar("en_thermo")
        self.add_scalar("en_tot")

    @property
    def bulk_species(self):
        return self.fluid

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self, verbose: bool = False):
        pass

    def update_scalar_quantities(self):
        rho = self.fluid.density.spline.vector
        u = self.fluid.velocity.spline.vector

        en_U = 0.5 * Propagator.mass_ops.WMM.massop.dot_inner(u, u)
        self.update_scalar("en_U", en_U)

        en_thermo = 0.5 * Propagator.mass_ops.M3.dot_inner(rho, rho)
        self.update_scalar("en_thermo", en_thermo)

        en_tot = en_U + en_thermo
        self.update_scalar("en_tot", en_tot)

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "variat_dens.Options" in line:
                    new_file += [
                        "model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model='barotropic')\n",
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
