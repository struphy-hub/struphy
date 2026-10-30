from struphy.io.options import BaseUnits
from struphy.models.cold_euler_sph import ColdEulerSPH
from struphy.propagators.push_v_in_sph_pressure import PushVinSPHpressure


class EulerSPH(ColdEulerSPH):
    r"""Euler equations (compressible, inviscid) discretized with smoothed particle hydrodynamics (SPH).

    Extends :class:`~struphy.models.ColdEulerSPH` by the pressure force.

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

    class Propagators(ColdEulerSPH.Propagators):
        """Propagators of :class:`~struphy.models.ColdEulerSPH` plus ``push_sph_p`` (pressure force of the SPH
        density, and an optional constant gravity).

        Parameters
        ----------
        **kwargs
            Passed to :class:`ColdEulerSPH.Propagators <struphy.models.ColdEulerSPH.Propagators>`.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.push_sph_p = PushVinSPHpressure()

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    propagators: Propagators

    # names before the SPH family (old run metadata of ViscousEulerSPH)
    _renamed_species = {"euler_fluid": "fluid"}

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
        self.propagators.push_sph_p.variables.fluid = self.fluid.var

    @property
    def velocity_scale(self):
        return "thermal"

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "saving_params = " in line:
                    new_file += ["\nkd_plot = KernelDensityPlot()\n"]
                    new_file += ["saving_params = SavingParameters(kernel_density_plots=(kd_plot,))\n\n"]
                elif "sorting_params = " in line:
                    new_file += ["sorting_params = SortingParameters(boxes_per_dim=(12, 12, 1))\n\n"]
                elif "base_units = BaseUnits" in line:
                    new_file += ["base_units = BaseUnits(kBT=1.0)\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
        return params_path

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity:

        .. math::

            \partial_t \rho + \nabla \cdot (\rho \mathbf{u}) = 0

        Momentum:

        .. math::

            \rho (\partial_t \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u}) = -\nabla \left( \rho^2 \frac{\partial \mathcal{U}(\rho, S)}{\partial \rho} \right)
            + \rho \mathbf g + \frac{\rho}{\varepsilon} \left( \mathbf{F} + \mathbf{u} \times \mathbf{B}_0 \right)

        Entropy:

        .. math::

            \partial_t S + \mathbf{u} \cdot \nabla S = 0

        where :math:`S` denotes the entropy per unit mass, :math:`\mathbf g` a constant gravity,
        :math:`\mathbf{F}` an external force field per unit charge, :math:`\mathbf{B}_0` the magnetic field of the equilibrium (if
        any) and :math:`\varepsilon` the species parameter ``epsilon`` (see :class:`~struphy.models.ColdEulerSPH`).
        The internal energy per unit mass can be defined in two ways:

        .. math::

            \mathrm{isothermal:} \qquad \mathcal{U}(\rho, S) = \kappa(S) \log \rho

        .. math::

            \mathrm{polytropic:} \qquad \mathcal{U}(\rho, S) = \kappa(S) \frac{\rho^{\gamma - 1}}{\gamma - 1}

        **Conserved quantities** (periodic boundaries or no outflow):

        Mass:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \rho \, \textnormal d \mathbf x = 0

        Energy, for :math:`\mathbf F = 0` and :math:`\mathbf g = 0` (the Lorentz force does no work):

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \rho \left( \frac 12 |\mathbf u|^2 + \mathcal U(\rho, S)
            \right) \textnormal d \mathbf x = 0
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
        r"""EulerSPH is the particle-based SPH model of a compressible, inviscid fluid. The pressure force follows
        from the internal energy of the SPH density (isothermal or polytropic closure, selected with the options of
        ``push_sph_p``). It is intended for meshfree fluid experiments rather than FEEC-based field simulations.

        In the SPH family it extends :class:`~struphy.models.ColdEulerSPH` by the pressure force; its child
        :class:`~struphy.models.ViscousEulerSPH` adds the viscous force."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the Euler SPH model:

        .. code-block:: python

            from struphy.models import EulerSPH

            model = EulerSPH()
            model.fluid.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - SPH verification with pressure forces: sound waves, gas expansion
        - meshfree compressible-fluid experiments
        - testing the SPH pressure pusher"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - viscous flows (see :class:`~struphy.models.ViscousEulerSPH`)
        - FEEC grid-based fluid or MHD formulations
        - entropy-resolved thermodynamic evolution
        - kinetic plasma physics"""
