from struphy.io.options import BaseUnits
from struphy.models.euler_sph import EulerSPH
from struphy.propagators.push_v_in_viscous_potential import PushVinViscousPotential


class ViscousEulerSPH(EulerSPH):
    """Euler equations with viscosity discretized with smoothed particle hydrodynamics (SPH).

    Extends :class:`~struphy.models.EulerSPH` by the viscous force. Without pressure force (viscous diffusion of a
    pressureless fluid), set ``kappa=0.0`` in the options of ``push_sph_p``.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits(kBT=1.0))
    charge_number: int
        Charge number (in units of the positive elementary charge) of the fluid species (default: 1)
    mass_number: float
        Mass number (in units of Proton mass) of the fluid species (default: 1.0)
    """

    ## additional propagators

    class Propagators(EulerSPH.Propagators):
        """Propagators of :class:`~struphy.models.EulerSPH` plus ``push_viscous`` (viscous force).

        Parameters
        ----------
        **kwargs
            Passed to :class:`EulerSPH.Propagators <struphy.models.EulerSPH.Propagators>`.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.push_viscous = PushVinViscousPotential()

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        charge_number: int = 1,
        mass_number: float = 1.0,
    ):
        # 0. store input parameters
        self._store_params(locals())

        # 1.-5. see ColdEulerSPH
        super().__init__(base_units, charge_number, mass_number)

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.push_viscous.variables.fluid = self.fluid.var

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity:

        .. math::

            \partial_t \rho + \nabla \cdot (\rho \mathbf{u}) = 0

        Momentum:

        .. math::

            \rho (\partial_t \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u}) = -\nabla \left( \rho^2 \frac{\partial \mathcal{U}(\rho, S)}{\partial \rho} \right) - \nabla \cdot \boldsymbol{\pi}
            + \rho \mathbf g + \frac{\rho}{\varepsilon} \left( \mathbf{F} + \mathbf{u} \times \mathbf{B}_0 \right)

        Entropy:

        .. math::

            \partial_t S + \mathbf{u} \cdot \nabla S = 0

        where :math:`S` denotes the entropy per unit mass, :math:`\boldsymbol{\pi}` is the viscous stress tensor and
        :math:`\mathbf g`, :math:`\mathbf F`, :math:`\mathbf B_0`, :math:`\varepsilon` are as in
        :class:`~struphy.models.EulerSPH`.

        The viscous stress tensor for a Newtonian fluid is given by:

        .. math::

            \boldsymbol{\pi} = -\mu \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T - \frac{2}{3} (\nabla \cdot \mathbf{u}) \mathbf{I} \right)

        where :math:`\mu` is the dynamic (shear) viscosity and :math:`\mathbf{I}` is the identity tensor.

        The internal energy per unit mass can be defined in two ways:

        .. math::

            \mathrm{isothermal:} \qquad \mathcal{U}(\rho, S) = \kappa(S) \log \rho

        .. math::

            \mathrm{polytropic:} \qquad \mathcal{U}(\rho, S) = \kappa(S) \frac{\rho^{\gamma - 1}}{\gamma - 1}

        **Conserved quantities** (periodic boundaries or no outflow):

        Mass:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \rho \, \textnormal d \mathbf x = 0

        The energy of :class:`~struphy.models.EulerSPH` is dissipated by the viscosity.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The characteristic speed is thermal:

        .. math::

            \hat u = \hat v_\mathrm{th}.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - SPH kinetic energy: :math:`\frac 12 \int \rho |\mathbf u|^2 \, dV` (``en_kin``)"""

    @classmethod
    def doc_long_description(cls):
        r"""ViscousEulerSPH is the particle-based SPH fluid model with pressure and
        viscosity. It is intended for meshfree fluid experiments rather than FEEC-based field simulations.

        In the SPH family it extends :class:`~struphy.models.EulerSPH` by the viscous force and inherits its
        species, propagators and scalars."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the viscous Euler SPH model:

        .. code-block:: python

            from struphy.models import ViscousEulerSPH

            model = ViscousEulerSPH()
            model.fluid.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - SPH verification with pressure and viscosity
        - meshfree compressible-fluid experiments
        - testing particle-based viscosity and pressure pushers"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - FEEC grid-based fluid or MHD formulations
        - entropy-resolved thermodynamic evolution
        - kinetic plasma physics
        - studies that require exact field-based conservation structures"""
