from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC
from struphy.models.species import (
    FieldSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators.maxwell_weak_ampere import MaxwellWeakAmpere


class Maxwell(StruphyModel):
    """Maxwell's equations in vacuum for electromagnetic field evolution.

    The models of the Maxwell family inherit from this model: they call ``super().__init__()`` and extend the hooks of
    :meth:`~struphy.models.base.StruphyModel._setup` (species, propagators, variable links, scalars).

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    """

    ## species

    class EMFields(FieldSpecies):
        """Electromagnetic fields of the Maxwell family (named ``em_fields`` in all models of the family). Models
        with more field variables declare them before calling ``super().__init__()``."""

        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    ## propagators

    class Propagators:
        """Propagators of Maxwell's equations: ``maxwell`` (Faraday's and Ampère's law)."""

        def __init__(self):
            self.maxwell = MaxwellWeakAmpere()

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    em_fields: EMFields
    propagators: Propagators

    ## abstract methods

    def __init__(self, base_units: BaseUnits = BaseUnits()):
        # 0. store input parameters (a model inheriting from Maxwell stores its own before calling super().__init__)
        self._store_params(locals())

        # 1.-5. species, units, propagators, variable links and scalars, see StruphyModel._setup
        self._setup(base_units)

    def _create_species(self):
        """:meta private:"""
        self.em_fields = self.EMFields()

    def _link_propagator_variables(self):
        """:meta private:"""
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field

    def _create_scalars(self):
        """:meta private:"""
        en_E = BilinearEnergyFEEC(self.em_fields.e_field)
        en_B = BilinearEnergyFEEC(self.em_fields.b_field)
        return {"en_E": en_E, "en_B": en_B, "en_tot": en_E + en_B}

    @property
    def bulk_species(self):
        return None

    @property
    def velocity_scale(self):
        return "light"

    def post_allocate(self):
        pass

    ## abstract methods for documentation

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Toy"

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Ampère's law (no current):

        .. math::

            \frac{\partial \mathbf{E}}{\partial t} - \nabla \times \mathbf{B} = 0

        Faraday's law:

        .. math::

            \frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0

        **Conserved quantities** (for :math:`\mathbf E \times \mathbf n = 0` on the boundary, or periodic
        boundaries):

        Electromagnetic energy:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \frac 12 \int \left( |\mathbf E|^2 + |\mathbf B|^2 \right)
            \textnormal d \mathbf x = 0
        """

    @classmethod
    def doc_normalization(cls):
        r"""Velocity and fields are normalized as:

        .. math::

            \hat v = c\,,\qquad \hat E = c \hat B

        where :math:`c` is the speed of light."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Electric energy: :math:`E_E = \frac{1}{2} \int |\mathbf E|^2 \, dV` (``en_E``)
        - Magnetic energy: :math:`E_B = \frac{1}{2} \int |\mathbf B|^2 \, dV` (``en_B``)
        - Total energy: :math:`E_{tot} = E_E + E_B` (``en_tot``)"""

    @classmethod
    def doc_long_description(cls):
        r"""This model simulates the propagation of electromagnetic waves in vacuum
        using Maxwell's equations without sources.
        It uses a finite element exterior calculus (FEEC) formulation
        with the electric field in H(curl) and the magnetic field in H(div) spaces.

        This is the parent model of the Maxwell family: :class:`~struphy.models.ColdPlasma` adds the current of a
        cold electron fluid, and :class:`~struphy.models.ColdPlasmaVlasov` adds a hot kinetic species; both inherit
        the species, propagators and scalars of this model."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a Maxwell model:

        .. code-block:: python

            from struphy.models import Maxwell

            model = Maxwell()
            # Fields are accessible via:
            model.em_fields.e_field
            model.em_fields.b_field
        """

    @classmethod
    def doc_use_cases(cls):
        """Propagation of electromagnetic waves in vacuum."""

    @classmethod
    def doc_cannot_be_used_for(cls):
        """Plasma dynamics, plasma-field interactions, or any scenario involving charged particles.
        This model does not include any particle species or coupling to matter."""
