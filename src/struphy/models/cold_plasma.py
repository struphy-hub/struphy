import copy

from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, Scalars
from struphy.models.species import (
    FieldSpecies,
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators.jxb_cold import JxBCold
from struphy.propagators.maxwell_weak_ampere import MaxwellWeakAmpere
from struphy.propagators.ohm_cold import OhmCold

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

    1. :class:`~struphy.propagators.maxwell.Maxwell`
    2. :class:`~struphy.propagators.ohm_cold.OhmCold`
    3. :class:`~struphy.propagators.jxb_cold.JxBCold`

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
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            alpha: float = None,
            epsilon: float = None,
        ):
            self.current = FEECVariable(space="Hcurl")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                alpha=alpha,
                epsilon=epsilon,
            )

    ## propagators

    class Propagators:
        def __init__(self):
            self.maxwell = MaxwellWeakAmpere()
            self.ohm = OhmCold()
            self.jxb = JxBCold()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = -1,
        mass_number: float = 1 / 1836,
        alpha: float = None,
        epsilon: float = None,
    ):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.electrons = self.Electrons(
            charge_number=charge_number,
            mass_number=mass_number,
            alpha=alpha,
            epsilon=epsilon,
        )

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators()

        # 4. assign variables to propagators
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field

        self.propagators.ohm.variables.j = self.electrons.current
        self.propagators.ohm.variables.e = self.em_fields.e_field

        self.propagators.jxb.variables.j = self.electrons.current

        # 5. define scalars to be tracked during simulation
        electric_energy = BilinearEnergyFEEC(self.em_fields.e_field)
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        kinetic_energy = BilinearEnergyFEEC(
            self.electrons.current,
            bilinear_form_name="M1ninv",
            normalization=self.electrons.equation_params.alpha**2,
        )
        total_energy = electric_energy + magnetic_energy + kinetic_energy

        self.scalars = Scalars(
            electric_energy=electric_energy,
            magnetic_energy=magnetic_energy,
            kinetic_energy=kinetic_energy,
            total_energy=total_energy,
        )

    @property
    def bulk_species(self):
        return self.electrons

    @property
    def velocity_scale(self):
        return "light"

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Cold-plasma current:

        .. math::

            \frac{1}{n_0} \frac{\partial \mathbf{j}}{\partial t} = \frac{1}{\varepsilon} \mathbf{E} + \frac{1}{\varepsilon n_0} \mathbf{j} \times \mathbf{B}_0

        Faraday's law:

        .. math::

            \frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0

        Ampère's law:

        .. math::

            -\frac{\partial \mathbf{E}}{\partial t} + \nabla \times \mathbf{B} = \frac{\alpha^2}{\varepsilon} \mathbf{j}

        where :math:`(n_0, \mathbf{B}_0)` denotes an inhomogeneous background.
        """

    @classmethod
    def doc_normalization(cls):
        r"""Velocities are normalized with the speed of light and the fields satisfy

        .. math::

            \hat v = c,\qquad \hat E = c \hat B.

        The dimensionless plasma parameters are the cold-species
        :math:`\alpha = \hat\Omega_\mathrm{p}/\hat\Omega_\mathrm{c}` and
        :math:`\varepsilon = 1/(\hat\Omega_\mathrm{c}\hat t)`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Electric field energy: ``electric_energy``
        - Magnetic field energy: ``magnetic_energy``
        - Cold-current energy: ``kinetic_energy``
        - Total energy: ``total_energy``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.maxwell_weak_ampere.MaxwellWeakAmpere`
        2. :class:`~struphy.propagators.ohm_cold.OhmCold`
        3. :class:`~struphy.propagators.jxb_cold.JxBCold`
        """
        doc = rf"""**1. propagators.maxwell.Maxwell:**

{MaxwellWeakAmpere.__doc__}

**2. OhmCold:**

{OhmCold.__doc__}

**3. JxBCold:**

{JxBCold.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""This is a fluid electromagnetic model for a cold plasma response. The
        electron fluid is represented only through its current, so thermal
        pressure and kinetic velocity-space effects are omitted. It is useful as
        a reduced model between vacuum Maxwell and fully kinetic Vlasov-Maxwell
        dynamics."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a cold-plasma model:

        .. code-block:: python

            from struphy.models import ColdPlasma

            model = ColdPlasma()
            model.em_fields.e_field
            model.em_fields.b_field
            model.electrons.current
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - cold-plasma wave propagation studies
        - electromagnetic benchmarks with a fluid current response
        - regimes where thermal pressure can be neglected
        - algorithm verification for Maxwell plus current coupling"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - finite-temperature or pressure-driven plasma dynamics
        - kinetic resonances or velocity-space instabilities
        - multi-species hybrid or fully kinetic problems
        - collisional closures beyond the built-in cold-plasma approximation"""

    def allocate_helpers(self):
        pass
