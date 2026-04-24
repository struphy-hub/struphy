from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC, Scalars
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


class ShearAlfven(StruphyModel):
    r"""ShearAlfven propagator from :class:`~struphy.models.linear_mhd.LinearMHD` with zero-flow equilibrium (:math:`\mathbf U_0 = 0`).

    :ref:`normalization`:

    .. math::

        \hat U =  \hat v_\textnormal{A} \,.

    :ref:`Equations <gempic>`:

    .. math::

        \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} = (\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0\,,

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
        def __init__(self, mass_number: float = 1.0):
            self.velocity = FEECVariable(space="Hdiv")
            self.init_variables(mass_number=mass_number)

    class Propagators:
        def __init__(self) -> None:
            self.shear_alf = propagators_fields.ShearAlfven()

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

    @classmethod
    def doc_pde(cls):
        r""":ref:`Equations <gempic>`:

        .. math::

            \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} = (\nabla\times \tilde{\mathbf{B}})\times\mathbf{B}_0\,,

            &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
            = 0\,.
        """

    @classmethod
    def doc_normalization(cls):
        r"""All velocities are normalized by the Alfvén speed,

        .. math::

            \hat U = \hat v_A.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Perturbed kinetic energy: ``en_U``
        - Perturbed magnetic energy: ``en_B``
        - Background magnetic energy: ``en_B_eq``
        - Total magnetic energy including background: ``en_B_tot``
        - Perturbation total energy: ``en_tot``
        - Total energy including background contribution: ``en_tot2``"""

    @classmethod
    def doc_discretization(cls):
        doc = rf"""**1. propagators_fields.ShearAlfven:**

{propagators_fields.ShearAlfven.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""ShearAlfven is the incompressible magnetic-tension subsystem extracted
        from linear MHD. It is useful when one wants to isolate Alfvénic
        propagation from the full linear MHD model."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a shear-Alfvén model:

        .. code-block:: python

            from struphy.models import ShearAlfven

            model = ShearAlfven()
            model.em_fields.b_field
            model.mhd.velocity
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - isolated shear-Alfvén wave propagation
        - verification of the ShearAlfven propagator
        - reduced linear-MHD studies without compressibility"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - magnetosonic or compressible wave physics
        - nonlinear MHD dynamics
        - kinetic particle coupling
        - resistive, viscous, or Hall-MHD effects"""

    def allocate_helpers(self, verbose: bool = False):
        # project background magnetic field (2-form) and pressure (3-form)
        self._b_eq = Propagator.projected_equil.b2

        # temporary vectors for scalar quantities
        self._tmp_b1 = Propagator.derham.V2.zeros()
        self._tmp_b2 = Propagator.derham.V2.zeros()

    def __init__(self, base_units: BaseUnits = BaseUnits(), mass_number: float = 1.0):

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD(mass_number=mass_number)

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.shear_alf.variables.u = self.mhd.velocity
        self.propagators.shear_alf.variables.b = self.em_fields.b_field

        # 5. define scalars to be tracked during simulation
        kinetic_energy = BilinearEnergyFEEC(self.mhd.velocity, bilinear_form_name="M2n")
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        background_magnetic = FunctionScalarFEEC(self._compute_en_B_eq)
        total_magnetic = FunctionScalarFEEC(self._compute_en_B_tot)
        self.scalars = Scalars(
            en_tot=kinetic_energy + magnetic_energy,
            en_U=kinetic_energy,
            en_B=magnetic_energy,
            en_B_eq=background_magnetic,
            en_B_tot=total_magnetic,
            en_tot2=kinetic_energy + magnetic_energy + background_magnetic,
        )

    def _compute_en_B_eq(self):
        Propagator.mass_ops.M2.dot(self._b_eq, apply_bc=False, out=self._tmp_b1)
        return self._b_eq.inner(self._tmp_b1) / 2

    def _compute_en_B_tot(self):
        self._b_eq.copy(out=self._tmp_b1)
        self._tmp_b1 += self.em_fields.b_field.spline.vector
        Propagator.mass_ops.M2.dot(self._tmp_b1, apply_bc=False, out=self._tmp_b2)
        return self._tmp_b1.inner(self._tmp_b2) / 2
