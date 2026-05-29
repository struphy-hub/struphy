import copy

from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import KineticEnergySPH, Scalars
from struphy.models.species import (
    ParticleSpecies,
)
from struphy.models.variables import SPHVariable
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_vin_sph_pressure import PushVinSPHpressure
from struphy.propagators.push_vin_viscous_potential import PushVinViscousPotential
from struphy.propagators.push_vxb import PushVxB

rank = MPI.COMM_WORLD.Get_rank()


class ViscousEulerSPH(StruphyModel):
    r"""Euler equations with viscosity discretized with smoothed particle hydrodynamics (SPH)."""

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class EulerFluid(ParticleSpecies):
        def __init__(self, charge_number: int = 1, mass_number: float = 1.0):
            self.var = SPHVariable()
            self.init_variables(charge_number=charge_number, mass_number=mass_number)

    ## propagators

    class Propagators:
        def __init__(self, with_B0: bool = True, with_p: bool = True, with_viscosity: bool = True):
            self.push_eta = PushEta()
            if with_B0:
                self.push_vxb = PushVxB()
            if with_p:
                self.push_sph_p = PushVinSPHpressure()
            if with_viscosity:
                self.push_viscous = PushVinViscousPotential()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        charge_number: int = 1,
        mass_number: float = 1.0,
        with_B0: bool = True,
        with_p: bool = True,
        with_viscosity: bool = True,
    ):

        self.with_B0 = with_B0
        self.with_p = with_p
        self.with_viscosity = with_viscosity

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.euler_fluid = self.EulerFluid(charge_number=charge_number, mass_number=mass_number)

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(with_B0=with_B0, with_p=with_p, with_viscosity=with_viscosity)

        # 4. assign variables to propagators
        self.propagators.push_eta.variables.var = self.euler_fluid.var
        if with_B0:
            self.propagators.push_vxb.variables.ions = self.euler_fluid.var
        if with_p:
            self.propagators.push_sph_p.variables.fluid = self.euler_fluid.var
        if with_viscosity:
            self.propagators.push_viscous.variables.fluid = self.euler_fluid.var

        # 5. define scalars to be tracked during simulation
        self.scalars = Scalars(
            en_kin=KineticEnergySPH(self.euler_fluid.var),
        )

    @property
    def bulk_species(self):
        return self.euler_fluid

    @property
    def velocity_scale(self):
        return "thermal"

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity:

        .. math::

            \partial_t \rho + \nabla \cdot (\rho \mathbf{u}) = 0

        Momentum:

        .. math::

            \rho (\partial_t \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u}) = -\nabla \left( \rho^2 \frac{\partial \mathcal{U}(\rho, S)}{\partial \rho} \right) - \nabla \cdot \boldsymbol{\pi}

        Entropy:

        .. math::

            \partial_t S + \mathbf{u} \cdot \nabla S = 0

        where :math:`S` denotes the entropy per unit mass and :math:`\boldsymbol{\pi}` is the viscous stress tensor.

        The viscous stress tensor for a Newtonian fluid is given by:

        .. math::

            \boldsymbol{\sigma} = -\mu \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T - \frac{2}{3} (\nabla \cdot \mathbf{u}) \mathbf{I} \right)

        where :math:`\mu` is the dynamic (shear) viscosity and :math:`\mathbf{I}` is the identity tensor.

        The internal energy per unit mass can be defined in two ways:

        .. math::

            \mathrm{isothermal:} \qquad \mathcal{U}(\rho, S) = \kappa(S) \log \rho

        .. math::

            \mathrm{polytropic:} \qquad \mathcal{U}(\rho, S) = \kappa(S) \frac{\rho^{\gamma - 1}}{\gamma - 1}
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

        - SPH kinetic energy: ``en_kin``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.push_eta.PushEta`
        2. :class:`~struphy.propagators.push_vxb.PushVxB` (if :attr:`with_B0` is True)
        3. :class:`~struphy.propagators.push_vin_sph_pressure.PushVinSPHpressure` (if :attr:`with_p` is True)
        4. :class:`~struphy.propagators.push_vin_viscous_potential.PushVinViscousPotential` (if :attr:`with_viscosity` is True)
        """
        doc = rf"""**1. PushEta:**

    {PushEta.__doc__}

    **2. PushVxB:**

    {PushVxB.__doc__}

    **3. PushVinSPHpressure:**

    {PushVinSPHpressure.__doc__}

    **4. PushVinViscousPotential:**

    {PushVinViscousPotential.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""ViscousEulerSPH is the particle-based SPH fluid model with optional
        pressure and viscosity contributions. It is intended for meshfree fluid
        experiments rather than FEEC-based field simulations."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the viscous Euler SPH model:

        .. code-block:: python

            from struphy.models import ViscousEulerSPH

            model = ViscousEulerSPH()
            model.euler_fluid.var
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

    def allocate_helpers(self):
        pass

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "push_vxb.Options" in line:
                    new_file += ["if model.with_B0:\n"]
                    new_file += ["    " + line]
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
