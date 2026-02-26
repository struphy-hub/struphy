from feectools.ddm.mpi import mpi as MPI
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
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
        def __init__(self):
            self.var = SPHVariable()
            self.init_variables()

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

    def __init__(self, with_B0: bool = True, with_p: bool = True, with_viscosity: bool = True):

        self.with_B0 = with_B0
        self.with_p = with_p
        self.with_viscosity = with_viscosity

        # 1. instantiate all species
        self.euler_fluid = self.EulerFluid()

        # 2. instantiate all propagators
        self.propagators = self.Propagators(with_B0=with_B0, with_p=with_p, with_viscosity=with_viscosity)

        # 3. assign variables to propagators
        self.propagators.push_eta.variables.var = self.euler_fluid.var
        if with_B0:
            self.propagators.push_vxb.variables.ions = self.euler_fluid.var
        if with_p:
            self.propagators.push_sph_p.variables.fluid = self.euler_fluid.var
        if with_viscosity:
            self.propagators.push_viscous.variables.fluid = self.euler_fluid.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_kin", compute="from_sph", variable=self.euler_fluid.var)

    @property
    def bulk_species(self):
        return self.euler_fluid

    @property
    def velocity_scale(self):
        return "thermal"

    def allocate_helpers(self, verbose: bool = False):
        pass

    def update_scalar_quantities(self):
        particles = self.euler_fluid.var.particles
        valid_markers = particles.markers_wo_holes_and_ghost
        en_kin = valid_markers[:, 6].dot(
            valid_markers[:, 3] ** 2 + valid_markers[:, 4] ** 2 + valid_markers[:, 5] ** 2,
        ) / (2.0 * particles.Np)
        self.update_scalar("en_kin", en_kin)

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
