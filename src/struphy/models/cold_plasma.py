from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.maxwell import Maxwell
from struphy.models.scalars import BilinearEnergyFEEC
from struphy.models.species import FluidSpecies
from struphy.models.variables import FEECVariable
from struphy.propagators.jxb_cold import JxBCold
from struphy.propagators.ohm_cold import OhmCold


class ColdPlasma(Maxwell):
    """Cold plasma model: electron-fluid current coupled with Maxwell's equations via a cold-plasma Ohm's law.

    Extends :class:`~struphy.models.Maxwell` by the current of a cold electron fluid.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    charge_number: int
        Charge number (in units of the positive elementary charge) of the electron species (default: -1)
    mass_number: float
        Mass number (in units of Proton mass) of the electron species (default: 1/1836)
    alpha: float, optional
        Dimensionless parameter: plasma frequency / cyclotron frequency. If None, computed from units and charge/mass numbers.
    epsilon: float, optional
        Normalized cyclotron period: 1 / (cyclotron frequency × time unit). If None, computed from units and charge/mass numbers.
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## additional species

    class Electrons(FluidSpecies):
        """Cold electron fluid, represented by its current (named ``electrons`` in all models of the family)."""

        def __init__(
            self,
            charge_number: int = -1,
            mass_number: float = 1 / 1836,
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

    ## additional propagators

    class Propagators(Maxwell.Propagators):
        """Propagators of :class:`~struphy.models.Maxwell` plus ``ohm`` (cold-plasma Ohm's law) and ``jxb``
        (rotation of the current in the background magnetic field).

        Parameters
        ----------
        **kwargs
            Passed to :class:`Maxwell.Propagators <struphy.models.Maxwell.Propagators>`.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ohm = OhmCold()
            self.jxb = JxBCold()

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    electrons: Electrons
    propagators: Propagators

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
        self._store_params(locals())
        # for the hooks: the arguments of this constructor
        self._electron_args = {
            "charge_number": charge_number,
            "mass_number": mass_number,
            "alpha": alpha,
            "epsilon": epsilon,
        }

        # 1.-5. see Maxwell
        super().__init__(base_units)

    def _create_species(self):
        """:meta private:"""
        super()._create_species()
        self.electrons = self.Electrons(**self._electron_args)

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.ohm.variables.j = self.electrons.current
        self.propagators.ohm.variables.e = self.em_fields.e_field

        self.propagators.jxb.variables.j = self.electrons.current

    def _create_scalars(self):
        """:meta private:"""
        scalars = super()._create_scalars()
        # en_tot of Maxwell is replaced by the one including the current energy (and kept last)
        en_tot = scalars.pop("en_tot")
        en_J = BilinearEnergyFEEC(
            self.electrons.current,
            bilinear_form_name="M1ninv",
            normalization=self.electrons.equation_params.alpha**2,
        )
        scalars["en_J"] = en_J
        scalars["en_tot"] = en_tot + en_J
        return scalars

    @property
    def bulk_species(self):
        return self.electrons

    @property
    def velocity_scale(self):
        return "light"

    def post_allocate(self):
        pass

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

        **Conserved quantities** (for :math:`\mathbf E \times \mathbf n = 0` on the boundary, or periodic
        boundaries):

        Energy:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \frac 12 \int \left( |\mathbf E|^2 + |\mathbf B|^2
            + \frac{\alpha^2}{n_0} |\mathbf j|^2 \right) \textnormal d \mathbf x = 0
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

        - Electric field energy: :math:`\frac 12 \int |\mathbf E|^2 \, dV` (``en_E``)
        - Magnetic field energy: :math:`\frac 12 \int |\mathbf B|^2 \, dV` (``en_B``)
        - Cold-current energy: :math:`\frac{\alpha^2}{2} \int |\mathbf j|^2 / n_0 \, dV` (``en_J``)
        - Total energy: ``en_tot`` (sum of the above)"""

    @classmethod
    def doc_long_description(cls):
        r"""This is a fluid electromagnetic model for a cold plasma response. The
        electron fluid is represented only through its current, so thermal
        pressure and kinetic velocity-space effects are omitted. It is useful as
        a reduced model between vacuum Maxwell and fully kinetic Vlasov-Maxwell
        dynamics.

        In the Maxwell family it extends :class:`~struphy.models.Maxwell` by the cold electron current; its child
        :class:`~struphy.models.ColdPlasmaVlasov` adds a hot kinetic species."""

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
