from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import KineticEnergySPH, Scalars
from struphy.models.species import (
    ParticleSpecies,
)
from struphy.models.variables import SPHVariable
from struphy.propagators import (
    propagators_markers,
)

rank = MPI.COMM_WORLD.Get_rank()


class ViscousEulerSPH(StruphyModel):
    r"""Euler equations with viscosity discretized with smoothed particle hydrodynamics (SPH).

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{th} \,.

    :ref:`Equations <gempic>`:

    .. math::

        \begin{align}
        \partial_t \rho + \nabla \cdot (\rho \mathbf u) &= 0\,,
        \\[2mm]
        \rho(\partial_t \mathbf u + \mathbf u \cdot \nabla \mathbf u) &= - \nabla \left(\rho^2 \frac{\partial \mathcal U(\rho, S)}{\partial \rho} \right) - \nabla \cdot \boldsymbol{\pi}\,,
        \\[2mm]
        \partial_t S + \mathbf u \cdot \nabla S &= 0\,,
        \end{align}

    where :math:`S` denotes the entropy per unit mass and :math:`\boldsymbol{\pi}` is the viscous stress tensor.

    The viscous stress tensor for a Newtonian fluid is given by:

    .. math::

        \boldsymbol{\sigma} = -\mu \left( \nabla \mathbf u + (\nabla \mathbf u)^T - \frac{2}{3}(\nabla \cdot \mathbf u)\mathbf{I} \right)\,,

    where :math:`\mu` is the dynamic (shear) viscosity and :math:`\mathbf{I}` is the identity tensor.

    The internal energy per unit mass can be defined in two ways:

    .. math::

        \mathrm{isothermal:}\qquad &\mathcal U(\rho, S) = \kappa(S) \log \rho\,.

        \mathrm{polytropic:}\qquad &\mathcal U(\rho, S) = \kappa(S) \frac{\rho^{\gamma - 1}}{\gamma - 1}\,.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_markers.PushEta`
    2. :class:`~struphy.propagators.propagators_markers.PushVxB`
    3. :class:`~struphy.propagators.propagators_markers.PushVinSPHpressure`
    4. :class:`~struphy.propagators.propagators_markers.PushVinViscousPotential`
    """

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
            self.push_eta = propagators_markers.PushEta()
            if with_B0:
                self.push_vxb = propagators_markers.PushVxB()
            if with_p:
                self.push_sph_p = propagators_markers.PushVinSPHpressure()
            if with_viscosity:
                self.push_viscous = propagators_markers.PushVinViscousPotential()

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
        r""":ref:`Equations <gempic>`:

        .. math::

            \begin{align}
            \partial_t \rho + \nabla \cdot (\rho \mathbf u) &= 0\,,
            \\[2mm]
            \rho(\partial_t \mathbf u + \mathbf u \cdot \nabla \mathbf u) &= - \nabla \left(\rho^2 \frac{\partial \mathcal U(\rho, S)}{\partial \rho} \right) - \nabla \cdot \boldsymbol{\pi}\,,
            \\[2mm]
            \partial_t S + \mathbf u \cdot \nabla S &= 0\,,
            \end{align}

        where :math:`S` denotes the entropy per unit mass and :math:`\boldsymbol{\pi}` is the viscous stress tensor.

        The viscous stress tensor for a Newtonian fluid is given by:

        .. math::

            \boldsymbol{\sigma} = -\mu \left( \nabla \mathbf u + (\nabla \mathbf u)^T - \frac{2}{3}(\nabla \cdot \mathbf u)\mathbf{I} \right)\,,

        where :math:`\mu` is the dynamic (shear) viscosity and :math:`\mathbf{I}` is the identity tensor.

        The internal energy per unit mass can be defined in two ways:

        .. math::

            \mathrm{isothermal:}\qquad &\mathcal U(\rho, S) = \kappa(S) \log \rho\,.

            \mathrm{polytropic:}\qquad &\mathcal U(\rho, S) = \kappa(S) \frac{\rho^{\gamma - 1}}{\gamma - 1}\,.
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
        doc = rf"""**1. propagators_markers.PushEta:**

{propagators_markers.PushEta.__doc__}

**2. propagators_markers.PushVxB:**

{propagators_markers.PushVxB.__doc__}

**3. propagators_markers.PushVinSPHpressure:**

{propagators_markers.PushVinSPHpressure.__doc__}

**4. propagators_markers.PushVinViscousPotential:**

{propagators_markers.PushVinViscousPotential.__doc__}
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

    def allocate_helpers(self, verbose: bool = False):
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
                elif "set_save_data" in line:
                    new_file += ["\nkd_plot = KernelDensityPlot()\n"]
                    new_file += ["model.euler_fluid.set_save_data(kernel_density_plots=(kd_plot,))\n"]
                elif "base_units = BaseUnits" in line:
                    new_file += ["base_units = BaseUnits(kBT=1.0)\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
