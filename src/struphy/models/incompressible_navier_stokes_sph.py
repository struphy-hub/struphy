import copy

from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import KineticEnergySPH, Scalars
from struphy.models.species import (
    FieldSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, SPHVariable
from struphy.pic.accumulation import accum_kernels
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.propagators.poisson_solve import PoissonSolve
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_vin_efield import PushVinEfield
from struphy.propagators.push_vin_viscous_potential import PushVinViscousPotential
from struphy.propagators.push_vxb import PushVxB
from struphy.utils.pyccel import Pyccelkernel


class IncompressibleNavierStokesSPH(StruphyModel):
    """Incompressible Navier-Stokes equations discretized with smoothed particle hydrodynamics (SPH).

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits(kBT=1.0))
    charge_number: int
        Charge number (in units of the positive elementary charge) of the fluid species (default: 1)
    mass_number: float
        Mass number (in units of Proton mass) of the fluid species (default: 1.0)
    with_B0: bool
        Whether to include the effect of a background magnetic field B0 (default: True)
    with_viscosity: bool
        Whether to include viscous dissipation (default: True)
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class Fluid(ParticleSpecies):
        def __init__(self, charge_number: int = 1, mass_number: float = 1.0):
            self.density = SPHVariable()
            self.init_variables(charge_number=charge_number, mass_number=mass_number)

    class LagrangeMultiplier(FieldSpecies):
        def __init__(self):
            self.pressure = FEECVariable(space="H1")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(
            self,
            ptg: ParticlesToGrid,
            pressure: FEECVariable,
            ptg_coeff: float = 1.0,
            with_B0: bool = True,
            with_viscosity: bool = True,
        ):
            self.push_eta = PushEta()
            if with_B0:
                self.push_vxb = PushVxB()
            if with_viscosity:
                self.push_viscous = PushVinViscousPotential()
            self.pressure_poisson = PoissonSolve(rho=ptg, rho_coeffs=ptg_coeff)
            self.chorin_projection = PushVinEfield(phi=pressure)

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        charge_number: int = 1,
        mass_number: float = 1.0,
        with_B0: bool = True,
        with_viscosity: bool = True,
    ):

        self.with_B0 = with_B0
        self.with_viscosity = with_viscosity

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.fluid = self.Fluid(charge_number=charge_number, mass_number=mass_number)
        self.lagrange_multiplier = self.LagrangeMultiplier()

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        ptg = ParticlesToGrid(
            self.fluid.density,
            "Hcurl",
            Pyccelkernel(accum_kernels.div_u_weak_1form),
        )
        self.propagators = self.Propagators(
            ptg=ptg,
            pressure=self.lagrange_multiplier.pressure,
            with_B0=with_B0,
            with_viscosity=with_viscosity,
        )

        # 4. assign variables to propagators
        self.propagators.push_eta.variables.var = self.fluid.density
        if with_B0:
            self.propagators.push_vxb.variables.ions = self.fluid.density
        if with_viscosity:
            self.propagators.push_viscous.variables.fluid = self.fluid.density
        self.propagators.pressure_poisson.variables.phi = self.lagrange_multiplier.pressure
        self.propagators.chorin_projection.variables.var = self.fluid.density

        # 5. define scalars to be tracked during simulation
        self.scalars = Scalars(
            en_kin=KineticEnergySPH(self.fluid.density),
        )

    @property
    def bulk_species(self):
        return self.fluid

    @property
    def velocity_scale(self):
        return "thermal"

    def allocate_helpers(self):
        pass

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "time_opts = " in line:
                    new_file += [line]
                    new_file += ["assert time_opts.split_algo == 'LieTrotter'\n"]
                elif "push_vxb.Options" in line:
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

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Incompressible Navier-Stokes:

        .. math::

            \rho (\partial_t \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u}) = -\nabla p - \nabla \cdot \boldsymbol{\sigma}
            \\[2mm]
            \nabla \cdot \mathbf{u} = 0\,.

        where :math:`\boldsymbol{\sigma}` is the viscous stress tensor.

        The viscous stress tensor for a Newtonian fluid is given by:

        .. math::

            \boldsymbol{\sigma} = -\mu \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T - \frac{2}{3} (\nabla \cdot \mathbf{u}) \mathbf{I} \right)

        where :math:`\mu` is the dynamic (shear) viscosity and :math:`\mathbf{I}` is the identity tensor.
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
        3. :class:`~struphy.propagators.push_vin_viscous_potential.PushVinViscousPotential` (if :attr:`with_viscosity` is True)
        4. :class:`~struphy.propagators.poisson_solve.PoissonSolve`
        5. :class:`~struphy.propagators.push_vin_efield.PushVinEfield`
        """
        doc = rf"""**1. PushEta:**

    {PushEta.__doc__}

    **2. PushVxB:**

    {PushVxB.__doc__}

    **3. PushVinViscousPotential:**

    {PushVinViscousPotential.__doc__}
    
    **4. PoissonSolve:**
    
    {PoissonSolve.__doc__}
    
    **5. PushVinEfield:**
    
    {PushVinEfield.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""IncompressibleNavierStokesSPH is the particle-based SPH fluid model..."""

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
