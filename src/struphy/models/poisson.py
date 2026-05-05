import logging

from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.implicit_diffusion import ImplicitDiffusion
from struphy.propagators.poisson_field_solve import PoissonFieldSolve
from struphy.propagators.time_dependent_source import TimeDependentSource

logger = logging.getLogger("struphy")
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

    1. :class:`~struphy.propagators.time_dependent_source.TimeDependentSource`
    2. :class:`~struphy.propagators.implicit_diffusion.ImplicitDiffusion`

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
        def __init__(self, with_t_dep_source=False):
            if with_t_dep_source:
                self.source = TimeDependentSource()
            self.poisson = PoissonFieldSolve()

    ## abstract methods

    def __init__(self, base_units: BaseUnits = BaseUnits(), with_t_dep_source=False):

        self.with_t_dep_source = with_t_dep_source

        # 1. instantiate all species
        self.em_fields = self.EMFields()

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(with_t_dep_source=with_t_dep_source)

        # 4. assign variables to propagators
        if with_t_dep_source:
            self.propagators.source.variables.source = self.em_fields.source
        self.propagators.poisson.variables.phi = self.em_fields.phi

        # 5. define scalars to be tracked during simulation

    @property
    def bulk_species(self):
        return None

    @property
    def velocity_scale(self):
        return None

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Find :math:`\phi \in H^1` such that

        .. math::

            -\nabla \cdot D_0(\mathbf{x}) \nabla \phi + n_0(\mathbf{x}) \phi = \rho(t, \mathbf{x})

        where :math:`n_0, \rho(t) : \Omega \to \mathbb{R}` are real-valued functions, :math:`\rho(t)` is
        parametrized by time :math:`t`, and :math:`D_0 : \Omega \to \mathbb{R}^{3 \times 3}` is a positive matrix.
        Boundary terms from integration by parts are assumed to vanish.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The coefficient scaling is

        .. math::

            \hat D = \hat n / \hat x^2,\qquad \hat\rho = \hat n.

        No dedicated velocity normalization is used."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - No default scalar diagnostics are defined by this model."""

    @classmethod
    def doc_discretization(cls):
        doc = rf"""**1. TimeDependentSource:**

{TimeDependentSource.__doc__}

**2. PoissonFieldSolve:**

{PoissonFieldSolve.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""Poisson is the standalone elliptic field-solve model used for weak
        diffusion/Poisson problems. It is also the building block reused by
        other models for initial electrostatic solves."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a Poisson model:

        .. code-block:: python

            from struphy.models import Poisson

            model = Poisson()
            model.em_fields.phi
            model.em_fields.source
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - elliptic benchmark problems
        - electrostatic field solves with prescribed source terms
        - testing weak Poisson discretizations and boundary handling"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - hyperbolic time-dependent wave propagation
        - self-consistent kinetic plasma evolution on its own
        - magnetic-field dynamics or full Maxwell coupling"""

    def allocate_helpers(self, verbose: bool = False):
        """Solve initial Poisson equation.

        :meta private:
        """
        # # use setter to assign source
        # self.propagators.poisson.rho = Propagator.mass_ops.M0.dot(self.em_fields.source.spline.vector)

        # Solve with dt=1. and compute electric field
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.info("\nSolving initial Poisson problem...")

        self.propagators.poisson(1.0)

        if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
            logger.info("... Done.")

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
