from struphy.io.options import BaseUnits
from struphy.models.cold_euler_sph import ColdEulerSPH
from struphy.models.species import FieldSpecies
from struphy.models.variables import FEECVariable
from struphy.pic.accumulation.kernels.div_u_weak_1form import div_u_weak_1form
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.propagators.poisson_solve import PoissonSolve
from struphy.propagators.push_v_in_force_field import PushVinForceField
from struphy.propagators.push_v_in_viscous_potential import PushVinViscousPotential


class IncompressibleNavierStokesSPH(ColdEulerSPH):
    """Incompressible Navier-Stokes equations discretized with smoothed particle hydrodynamics (SPH).

    Extends :class:`~struphy.models.ColdEulerSPH` by the viscous force and the incompressibility constraint, whose
    Lagrange multiplier is the pressure. Without viscosity (incompressible Euler equations), set ``mu=0.0`` in the
    options of ``push_viscous``.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits(kBT=1.0))
    charge_number: int
        Charge number (in units of the positive elementary charge) of the fluid species (default: 1)
    mass_number: float
        Mass number (in units of Proton mass) of the fluid species (default: 1.0)
    """

    ## additional species

    class LagrangeMultiplier(FieldSpecies):
        def __init__(self):
            self.pressure = FEECVariable(space="H1")
            self.init_variables()

    ## additional propagators

    class Propagators(ColdEulerSPH.Propagators):
        """Propagators of :class:`~struphy.models.ColdEulerSPH` plus ``push_viscous`` (viscous force),
        ``pressure_poisson`` (pressure from the divergence of the velocity) and ``chorin_projection`` (pressure
        force, which makes the velocity divergence free).

        Parameters
        ----------
        ptg : ParticlesToGrid, optional
            Deposit of the weak velocity divergence, right-hand side of ``pressure_poisson`` (default: None).
        pressure : FEECVariable, optional
            Pressure, the potential of ``chorin_projection`` (default: None).
        ptg_coeff : float
            Coefficient of the right-hand side of ``pressure_poisson`` (default: -1.0).
        **kwargs
            Passed to :class:`ColdEulerSPH.Propagators <struphy.models.ColdEulerSPH.Propagators>`.
        """

        def __init__(
            self,
            ptg: ParticlesToGrid = None,
            pressure: FEECVariable = None,
            ptg_coeff: float = -1.0,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.push_viscous = PushVinViscousPotential()
            self.pressure_poisson = PoissonSolve(rho=ptg, rho_coeffs=ptg_coeff)
            self.chorin_projection = PushVinForceField(potential=pressure)

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    lagrange_multiplier: LagrangeMultiplier
    propagators: Propagators

    # the species was already named fluid (its variable density was renamed var: the save_data flag of old run
    # metadata is not restored)
    _renamed_species = {}

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

    def _create_species(self):
        """:meta private:"""
        # epsilon follows from the normalization (thermal velocity)
        self.fluid = self.Fluid(**self._particle_species_args(), epsilon=1.0)
        self.lagrange_multiplier = self.LagrangeMultiplier()

    def _propagator_args(self):
        """:meta private:"""
        ptg = ParticlesToGrid(
            self.fluid.var,
            "Hcurl",
            div_u_weak_1form,
        )
        return {
            **super()._propagator_args(),
            "ptg": ptg,
            "pressure": self.lagrange_multiplier.pressure,
        }

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.push_viscous.variables.fluid = self.fluid.var
        self.propagators.pressure_poisson.variables.phi = self.lagrange_multiplier.pressure
        self.propagators.chorin_projection.variables.var = self.fluid.var

    @property
    def velocity_scale(self):
        return "thermal"

    def post_allocate(self):
        # specific for Chorin projection
        self.propagators.pressure_poisson._divide_by_dt = True

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "time_opts = " in line:
                    new_file += [line]
                    new_file += ["assert time_opts.split_algo == 'LieTrotter'\n"]
                elif "saving_params = " in line:
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

        Incompressible Navier-Stokes:

        .. math::

            \rho (\partial_t \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u}) = -\nabla p - \nabla \cdot \boldsymbol{\sigma}
            + \frac{\rho}{\varepsilon} \left( \mathbf{F} + \mathbf{u} \times \mathbf{B}_0 \right)
            \\[2mm]
            \nabla \cdot \mathbf{u} = 0\,,

        where the pressure :math:`p` is the Lagrange multiplier of the incompressibility constraint,
        :math:`\boldsymbol{\sigma}` is the viscous stress tensor, and :math:`\mathbf F`, :math:`\mathbf B_0`,
        :math:`\varepsilon` are as in :class:`~struphy.models.ColdEulerSPH` (here :math:`\varepsilon = 1`).

        The viscous stress tensor for a Newtonian fluid is given by:

        .. math::

            \boldsymbol{\sigma} = -\mu \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T - \frac{2}{3} (\nabla \cdot \mathbf{u}) \mathbf{I} \right)

        where :math:`\mu` is the dynamic (shear) viscosity and :math:`\mathbf{I}` is the identity tensor.

        **Conserved quantities** (periodic boundaries or no outflow):

        Mass (volume):

        .. math::

            \frac{\textnormal d}{\textnormal d t} \int \rho \, \textnormal d \mathbf x = 0

        Kinetic energy for :math:`\mu = 0`, :math:`\mathbf F = 0` (the Lorentz force does no work); it is
        dissipated by the viscosity:

        .. math::

            \frac{\textnormal d}{\textnormal d t} \frac 12 \int \rho |\mathbf u|^2 \, \textnormal d \mathbf x
            = \int \boldsymbol \sigma : \nabla \mathbf u \, \textnormal d \mathbf x \leq 0
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
    def doc_discretization(cls) -> str:
        """Time integration is performed by the following propagators (in sequence). The pressure is computed with
        Chorin's projection method: ``pressure_poisson`` solves a Poisson equation for the pressure with the
        divergence of the velocity (divided by the time step) on the right-hand side, and ``chorin_projection``
        pushes the velocities with the pressure force, which makes them divergence free. This needs
        ``Time(split_algo="LieTrotter")``."""
        return cls.propagators_doc()

    @classmethod
    def doc_long_description(cls):
        r"""IncompressibleNavierStokesSPH is the particle-based SPH model of an incompressible, viscous fluid. The
        pressure is not given by an equation of state but is the Lagrange multiplier of the incompressibility
        constraint, computed on the FEEC grid with Chorin's projection method.

        In the SPH family it extends :class:`~struphy.models.ColdEulerSPH` by the viscous force and the
        incompressibility constraint, and inherits its species, propagators and scalars."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the incompressible Navier-Stokes SPH model:

        .. code-block:: python

            from struphy.models import IncompressibleNavierStokesSPH

            model = IncompressibleNavierStokesSPH()
            model.fluid.var
            model.lagrange_multiplier.pressure
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - incompressible flows, e.g. the relaxation of a shear flow in a channel with no-slip walls
        - SPH verification with a projection method for the pressure
        - testing the coupling of SPH particles to FEEC Poisson solves"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - compressible flows (see :class:`~struphy.models.EulerSPH`)
        - FEEC grid-based fluid or MHD formulations
        - Strang splitting (the projection needs ``LieTrotter``)
        - kinetic plasma physics"""
