from feectools.ddm.mpi import mpi as MPI
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators import (
    propagators_fields,
)
from struphy.propagators.base import Propagator

rank = MPI.COMM_WORLD.Get_rank()


class Poisson(StruphyModel):
    r"""Weak discretization of Poisson's equation with diffusion matrix, stabilization
    and time-depedent right-hand side.

    :ref:`normalization`:

    .. math::

        \hat D = \frac{\hat n}{\hat x^2}\,,\qquad \hat \rho = \hat n \,.

    :ref:`Equations <gempic>`: Find :math:`\phi \in H^1` such that

    .. math::

        - \nabla \cdot D_0(\mathbf x) \nabla \phi + n_0(\mathbf x) \phi =  \rho(t, \mathbf x)\,,

    where :math:`n_0, \rho(t):\Omega \to \mathbb R` are real-valued functions, :math:`\rho(t)` parametrized with time :math:`t`,
    and :math:`D_0:\Omega \to \mathbb R^{3\times 3}` is a positive matrix.
    Boundary terms from integration by parts are assumed to vanish.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.TimeDependentSource`
    2. :class:`~struphy.propagators.propagators_fields.ImplicitDiffusion`

    :ref:`Model info <add_model>`:
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.phi = FEECVariable(space="H1")
            self.source = FEECVariable(space="H1")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.source = propagators_fields.TimeDependentSource()
            self.poisson = propagators_fields.Poisson()

    ## abstract methods

    def __init__(self):

        # 1. instantiate all species
        self.em_fields = self.EMFields()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.source.variables.source = self.em_fields.source
        self.propagators.poisson.variables.phi = self.em_fields.phi

    @property
    def bulk_species(self):
        return None

    @property
    def velocity_scale(self):
        return None

    def allocate_helpers(self, verbose: bool = False):
        """Solve initial Poisson equation.

        :meta private:
        """
        # # use setter to assign source
        # self.propagators.poisson.rho = Propagator.mass_ops.M0.dot(self.em_fields.source.spline.vector)

        # Solve with dt=1. and compute electric field
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nSolving initial Poisson problem...")

        self.propagators.poisson(1.0)

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("... Done.")

    def update_scalar_quantities(self):
        pass

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "poisson.Options" in line:
                    new_file += [
                        "model.propagators.poisson.options = model.propagators.poisson.Options(rho=model.em_fields.source)\n",
                    ]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
