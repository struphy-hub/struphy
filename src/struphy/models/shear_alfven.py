from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators import (
    propagators_fields,
)

rank = MPI.COMM_WORLD.Get_rank()


class ShearAlfven(StruphyModel):
    r"""ShearAlfven propagator from :class:`~struphy.models.linear_mhd.LinearMHD` with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).

    :ref:`normalization`:

    .. math::

        \hat U =  \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t}
        =(\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0\,,

        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.ShearAlfven`

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
        def __init__(self):
            self.velocity = FEECVariable(space="Hdiv")
            self.init_variables()

    class Propagators:
        def __init__(self) -> None:
            self.shear_alf = propagators_fields.ShearAlfven()

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self):
        # project background magnetic field (2-form) and pressure (3-form)
        self._b_eq = Propagator.projected_equil.b2

        # temporary vectors for scalar quantities
        self._tmp_b1 = Propagator.derham.Vh["2"].zeros()
        self._tmp_b2 = Propagator.derham.Vh["2"].zeros()

    def __init__(self):
        if rank == 0:
            print(f"\n*** Creating light-weight instance of model '{self.__class__.__name__}':")

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.shear_alf.variables.u = self.mhd.velocity
        self.propagators.shear_alf.variables.b = self.em_fields.b_field

        # Scalar variables to be saved during simulation
        self.add_scalar("en_tot")

        self.add_scalar("en_U", compute="from_field")
        self.add_scalar("en_B", compute="from_field")
        self.add_scalar("en_B_eq", compute="from_field")
        self.add_scalar("en_B_tot", compute="from_field")
        self.add_scalar("en_tot2", summands=["en_U", "en_B", "en_B_eq"])

    def update_scalar_quantities(self):
        # perturbed fields
        en_U = 0.5 * Propagator.mass_ops.M2n.dot_inner(self.mhd.velocity.spline.vector, self.mhd.velocity.spline.vector)
        en_B = 0.5 * Propagator.mass_ops.M2.dot_inner(
            self.em_fields.b_field.spline.vector,
            self.em_fields.b_field.spline.vector,
        )

        self.update_scalar("en_U", en_U)
        self.update_scalar("en_B", en_B)
        self.update_scalar("en_tot", en_U + en_B)

        # background fields
        Propagator.mass_ops.M2.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)
        en_B0 = self._b_eq.inner(self._tmp_b1) / 2
        self.update_scalar("en_B_eq", en_B0)

        # total magnetic field
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.em_fields.b_field.spline.vector

        Propagator.mass_ops.M2.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)
        en_Btot = self._tmp_b1.inner(self._tmp_b2) / 2

        self.update_scalar("en_B_tot", en_Btot)
