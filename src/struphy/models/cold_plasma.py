from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators import (
    propagators_fields,
)
from struphy.propagators.base import Propagator

rank = MPI.COMM_WORLD.Get_rank()


class ColdPlasma(StruphyModel):
    r"""Cold plasma model.

    :ref:`normalization`:

    .. math::

        \hat v = c\,,\qquad \hat E = c \hat B \,.

    :ref:`Equations <gempic>`:

    .. math::

        \frac{1}{n_0} &\frac{\partial \mathbf j}{\partial t} = \frac{1}{\varepsilon} \mathbf E + \frac{1}{\varepsilon n_0} \mathbf j \times \mathbf B_0\,,
        \\[2mm]
        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,,
        \\[2mm]
        -&\frac{\partial \mathbf E}{\partial t} + \nabla\times\mathbf B =
        \frac{\alpha^2}{\varepsilon} \mathbf j \,,

    where :math:`(n_0,\mathbf B_0)` denotes a (inhomogeneous) background and

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p}}{\hat \Omega_\textnormal{c}}\,, \qquad \varepsilon = \frac{1}{\hat \Omega_\textnormal{c} \hat t}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.Maxwell`
    2. :class:`~struphy.propagators.propagators_fields.OhmCold`
    3. :class:`~struphy.propagators.propagators_fields.JxBCold`

    :ref:`Model info <add_model>`:
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    class Electrons(FluidSpecies):
        def __init__(self):
            self.current = FEECVariable(space="Hcurl")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.maxwell = propagators_fields.Maxwell()
            self.ohm = propagators_fields.OhmCold()
            self.jxb = propagators_fields.JxBCold()

    ## abstract methods

    def __init__(self):
        if rank == 0:
            print(f"Creating light-weight instance of model {self.__class__.__name__} ...")

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.electrons = self.Electrons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field

        self.propagators.ohm.variables.j = self.electrons.current
        self.propagators.ohm.variables.e = self.em_fields.e_field

        self.propagators.jxb.variables.j = self.electrons.current

        # define scalars for update_scalar_quantities
        self.add_scalar("electric energy")
        self.add_scalar("magnetic energy")
        self.add_scalar("kinetic energy")
        self.add_scalar("total energy")

    @property
    def bulk_species(self):
        return self.electrons

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self, verbose: bool = False):
        self._alpha = self.electrons.equation_params.alpha

    def update_scalar_quantities(self):
        e = self.em_fields.e_field.spline.vector
        b = self.em_fields.b_field.spline.vector
        j = self.electrons.current.spline.vector

        en_E = 0.5 * Propagator.mass_ops.M1.dot_inner(e, e)
        en_B = 0.5 * Propagator.mass_ops.M2.dot_inner(b, b)
        en_J = 0.5 * self._alpha**2 * Propagator.mass_ops.M1ninv.dot_inner(j, j)

        self.update_scalar("electric energy", en_E)
        self.update_scalar("magnetic energy", en_B)
        self.update_scalar("kinetic energy", en_J)
        self.update_scalar("total energy", en_E + en_B + en_J)
