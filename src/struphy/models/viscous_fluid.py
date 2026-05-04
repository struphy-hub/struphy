import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from struphy.propagators.variational_density_evolve import VariationalDensityEvolve
from struphy.propagators.variational_entropy_evolve import VariationalEntropyEvolve
from struphy.propagators.variational_momentum_advection import VariationalMomentumAdvection
from struphy.propagators.variational_viscosity import VariationalViscosity

from struphy.feec.mass import L2Projector
from struphy.feec.variational_utilities import (
    InternalEnergyEvaluator,
)
from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC, Scalars, VolumeFormEnergyFEEC
from struphy.models.species import (
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator

rank = MPI.COMM_WORLD.Get_rank()

class ViscousFluid(StruphyModel):
    r"""Full (non-linear) viscous Navier-Stokes equations discretized with a variational method.

    :ref:`normalization`:

    .. math::

        \hat u =  \hat v_\textnormal{A}\,, \qquad \hat{\mathcal U} = \frac{\hat{\mathbf B}^2}{\hat \rho \mu_0 (\gamma-1)} \,,\qquad \hat s = \hat \rho\ \textrm{ln}\left(\frac{\hat{\mathbf B}^2}{\mu_0 (\gamma -1) \hat{\rho}}\right) \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\partial_t \rho + \nabla \cdot ( \rho \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t (\rho \mathbf u) + \nabla \cdot (\rho \mathbf u \otimes \mathbf u) + \rho \nabla \frac{(\rho \mathcal U (\rho, s))}{\partial \rho} + s \nabla \frac{(\rho \mathcal U (\rho, s))}{\partial s} - \nabla \cdot \left((\mu +\mu_a(\mathbf x)) \nabla \mathbf u\right) = 0 \,,
        \\[4mm]
        &\partial_t s + \nabla \cdot ( s \mathbf u ) = \frac{1}{T}\left((\mu+\mu_a(\mathbf x)) |\nabla \mathbf u|^2 \right) \,,

    where the internal energy per unit mass is :math:`\mathcal U(\rho) = \rho^{\gamma-1} \exp(s / \rho)`.
    and :math:`\mu_a(\mathbf x)` is an artificial viscosity coefficient.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.variational_density_evolve.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.variational_momentum_advection.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.variational_entropy_evolve.VariationalEntropyEvolve`
    4. :class:`~struphy.propagators.variational_viscosity.VariationalViscosity`

    :ref:`Model info <add_model>`:
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class Fluid(FluidSpecies):
        def __init__(self, mass_number: float = 1.0):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="H1vec")
            self.entropy = FEECVariable(space="L2")
            self.init_variables(mass_number=mass_number)

    ## propagators

    class Propagators:
        def __init__(self, with_viscosity: bool = True):
            self.variat_dens = VariationalDensityEvolve()
            self.variat_mom = VariationalMomentumAdvection()
            self.variat_ent = VariationalEntropyEvolve()
            if with_viscosity:
                self.variat_viscous = VariationalViscosity()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mass_number: float = 1.0,
        with_viscosity: bool = True,
    ):

        # 1. instantiate all species
        self.fluid = self.Fluid(mass_number=mass_number)

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(with_viscosity=with_viscosity)

        # 4. assign variables to propagators
        self.propagators.variat_dens.variables.rho = self.fluid.density
        self.propagators.variat_dens.variables.u = self.fluid.velocity
        self.propagators.variat_mom.variables.u = self.fluid.velocity
        self.propagators.variat_ent.variables.s = self.fluid.entropy
        self.propagators.variat_ent.variables.u = self.fluid.velocity
        if with_viscosity:
            self.propagators.variat_viscous.variables.s = self.fluid.entropy
            self.propagators.variat_viscous.variables.u = self.fluid.velocity

        # 5. define scalars to be tracked during simulation
        kinetic_energy = BilinearEnergyFEEC(self.fluid.velocity, bilinear_form_name="WMMnew")
        thermo_energy = FunctionScalarFEEC(self.update_thermo_energy)
        total_energy = kinetic_energy + thermo_energy
        self.scalars = Scalars(
            en_U=kinetic_energy,
            en_thermo=thermo_energy,
            en_tot=total_energy,
            dens_tot=VolumeFormEnergyFEEC(self.fluid.density),
            entr_tot=VolumeFormEnergyFEEC(self.fluid.entropy),
        )

    @property
    def bulk_species(self):
        return self.fluid

    @property
    def velocity_scale(self):
        return "alfvén"

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity:

        .. math::

            \partial_t \rho + \nabla \cdot (\rho \mathbf{u}) = 0

        Momentum:

        .. math::

            \partial_t (\rho \mathbf{u}) + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) + \rho \nabla \frac{(\rho \mathcal{U}(\rho, s))}{\partial \rho} + s \nabla \frac{(\rho \mathcal{U}(\rho, s))}{\partial s} - \nabla \cdot \left( (\mu + \mu_a(\mathbf{x})) \nabla \mathbf{u} \right) = 0

        Entropy:

        .. math::

            \partial_t s + \nabla \cdot (s \mathbf{u}) = \frac{1}{T} \left( (\mu + \mu_a(\mathbf{x})) |\nabla \mathbf{u}|^2 \right)

        where the internal energy per unit mass is :math:`\mathcal U(\rho) = \rho^{\gamma-1} \exp(s / \rho)`.
        and :math:`\mu_a(\mathbf{x})` is an artificial viscosity coefficient.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The model uses Alfvén-speed scaling for the velocity and entropy-based
        thermodynamic units for the internal-energy closure."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy: ``en_U``
        - Thermodynamic energy: ``en_thermo``
        - Total energy: ``en_tot``
        - Total density / entropy: ``dens_tot``, ``entr_tot``"""

    @classmethod
    def doc_discretization(cls):
        doc = rf"""**1. VariationalDensityEvolve:**

{VariationalDensityEvolve.__doc__}

**2. VariationalMomentumAdvection:**

{VariationalMomentumAdvection.__doc__}

**3. VariationalEntropyEvolve:**

{VariationalEntropyEvolve.__doc__}

**4. VariationalViscosity:**

{VariationalViscosity.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""ViscousFluid is the non-magnetic viscous member of the variational
        fluid family. It is suited for compressible hydrodynamics with entropy
        transport and viscous heating but without magnetic effects."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a viscous-fluid model:

        .. code-block:: python

            from struphy.models import ViscousFluid

            model = ViscousFluid()
            model.fluid.density
            model.fluid.velocity
            model.fluid.entropy
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - nonlinear viscous compressible hydrodynamics
        - entropy-based Navier-Stokes benchmarks
        - testing variational viscous closures without magnetism"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - magnetic or MHD dynamics
        - inviscid strictly conservative benchmarks
        - kinetic particle effects"""

    def allocate_helpers(self, verbose: bool = False):
        projV3 = L2Projector("L2", Propagator.mass_ops)

        def f(e1, e2, e3):
            return 1

        f = xp.vectorize(f)
        self._integrator = projV3(f)

        self._energy_evaluator = InternalEnergyEvaluator(Propagator.derham, self.propagators.variat_ent.options.gamma)

        self._ones = Propagator.derham.V3pol.zeros()
        if isinstance(self._ones, PolarVector):
            self._ones.tp[:] = 1.0
        else:
            self._ones[:] = 1.0

    def update_thermo_energy(self):
        """Reuse tmp used in VariationalEntropyEvolve to compute the thermodynamical energy.

        :meta private:
        """
        rho = self.fluid.density.spline.vector
        s = self.fluid.entropy.spline.vector
        en_prop = self.propagators.variat_dens

        self._energy_evaluator.sf.vector = s
        self._energy_evaluator.rhof.vector = rho
        sf_values = self._energy_evaluator.sf.eval_tp_fixed_loc(
            self._energy_evaluator.integration_grid_spans,
            self._energy_evaluator.integration_grid_bd,
            out=self._energy_evaluator._sf_values,
        )
        rhof_values = self._energy_evaluator.rhof.eval_tp_fixed_loc(
            self._energy_evaluator.integration_grid_spans,
            self._energy_evaluator.integration_grid_bd,
            out=self._energy_evaluator._rhof_values,
        )
        e = self._energy_evaluator.ener
        ener_values = en_prop._proj_rho2_metric_term * e(rhof_values, sf_values)
        en_prop._get_L2dofs_V3(ener_values, dofs=en_prop._linear_form_dl_drho)
        en_thermo = self._integrator.inner(en_prop._linear_form_dl_drho)
        return en_thermo

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "variat_dens.Options" in line:
                    new_file += [
                        "model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model='full',\n",
                    ]
                    new_file += [
                        "                                                                              s=model.fluid.entropy)\n",
                    ]
                elif "variat_ent.Options" in line:
                    new_file += [
                        "model.propagators.variat_ent.options = model.propagators.variat_ent.Options(model='full',\n",
                    ]
                    new_file += [
                        "                                                                            rho=model.fluid.density)\n",
                    ]
                elif "variat_viscous.Options" in line:
                    new_file += [
                        "model.propagators.variat_viscous.options = model.propagators.variat_viscous.Options(rho=model.fluid.density)\n",
                    ]
                elif "entropy.add_background" in line:
                    new_file += ["model.fluid.density.add_background(FieldsBackground())\n"]
                    new_file += [line]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
