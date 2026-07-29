import copy

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.feec.mass import L2Projector
from struphy.feec.variational_utilities import (
    InternalEnergyEvaluator,
)
from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC, Scalars
from struphy.models.species import (
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.propagators.variational_density_evolve import VariationalDensityEvolve
from struphy.propagators.variational_entropy_evolve import VariationalEntropyEvolve
from struphy.propagators.variational_momentum_advection import VariationalMomentumAdvection




class VariationalCompressibleFluid(StruphyModel):
    """Fully compressible fluid equations discretized with a variational method.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    mass_number: float
        Mass number (in units of Proton mass) of the fluid species (default: 1.0)
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
        def __init__(
            self,
            s: FEECVariable = None,
            rho: FEECVariable = None,
        ):
            self.variat_dens = VariationalDensityEvolve(s=s)
            self.variat_mom = VariationalMomentumAdvection()
            self.variat_ent = VariationalEntropyEvolve(rho=rho)

    ## abstract methods

    def __init__(self, base_units: BaseUnits = BaseUnits(), mass_number: float = 1.0):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.fluid = self.Fluid(mass_number=mass_number)

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(s=self.fluid.entropy, rho=self.fluid.density)

        # 4. assign variables to propagators
        self.propagators.variat_dens.variables.rho = self.fluid.density
        self.propagators.variat_dens.variables.u = self.fluid.velocity
        self.propagators.variat_mom.variables.u = self.fluid.velocity
        self.propagators.variat_ent.variables.s = self.fluid.entropy
        self.propagators.variat_ent.variables.u = self.fluid.velocity

        # 5. define scalars to be tracked during simulation
        kinetic_energy = BilinearEnergyFEEC(self.fluid.velocity, bilinear_form_name="WMMnew")
        thermo_energy = FunctionScalarFEEC(self.update_thermo_energy)
        total_energy = kinetic_energy + thermo_energy
        self.scalars = Scalars(
            en_U=kinetic_energy,
            en_thermo=thermo_energy,
            en_tot=total_energy,
        )

    @property
    def bulk_species(self):
        return self.fluid

    @property
    def velocity_scale(self):
        return "alfvén"

    def allocate_helpers(self):
        projV3 = L2Projector("L2", Propagator.mass_ops)

        def f(e1, e2, e3):
            return 1

        f = xp.vectorize(f)
        self._integrator = projV3(f)

        self._energy_evaluator = InternalEnergyEvaluator(Propagator.derham, self.propagators.variat_ent.options.gamma)

    # default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "variat_dens.Options" in line:
                    new_file += [
                        "model.propagators.variat_dens.options = model.propagators.variat_dens.Options(model='full')\n",
                    ]
                elif "entropy.add_background" in line:
                    new_file += ["model.fluid.density.add_background(FieldsBackground())\n"]
                    new_file += [line]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)

    def update_thermo_energy(self):
        """Reuse tmp used in VariationalEntropyEvolve to compute the thermodynamical energy.

        :meta private:
        """
        en_prop = self.propagators.variat_ent

        self._energy_evaluator.sf.vector = self.fluid.entropy.spline.vector
        self._energy_evaluator.rhof.vector = self.fluid.density.spline.vector
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
        e = self.__ener
        ener_values = en_prop._proj_rho2_metric_term * e(rhof_values, sf_values)
        en_prop._get_L2dofs_V3(ener_values, dofs=en_prop._linear_form_dl_ds)
        en_thermo = self._integrator.inner(en_prop._linear_form_dl_ds)
        return en_thermo

    def __ener(self, rho, s):
        """Themodynamical energy as a function of rho and s, usign the perfect gaz hypothesis
        E(rho, s) = rho^gamma*exp(s/rho)"""
        return xp.power(rho, self.propagators.variat_ent.options.gamma) * xp.exp(s / rho)

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Continuity:

        .. math::

            \partial_t \rho + \nabla \cdot (\rho \mathbf{u}) = 0

        Momentum:

        .. math::

            \partial_t (\rho \mathbf{u}) + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) + \rho \nabla \frac{(\rho \mathcal{U}(\rho, s))}{\partial \rho} + s \nabla \frac{(\rho \mathcal{U}(\rho, s))}{\partial s} = 0

        Entropy:

        .. math::

            \partial_t s + \nabla \cdot (s \mathbf{u}) = 0

        where the internal energy per unit mass is :math:`\mathcal U(\rho) = \rho^{\gamma-1} \exp(s / \rho)`.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The model uses Alfvén-speed scaling for the flow variables together
        with separate units for internal energy and entropy:

        .. math::

            \hat u = \hat v_A,\qquad \hat{\mathcal U}=K,\qquad \hat s=\hat\rho C_v.
        """

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Kinetic energy: ``en_U``
        - Thermodynamic energy: ``en_thermo``
        - Total energy: ``en_tot``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.variational_density_evolve.VariationalDensityEvolve`
        2. :class:`~struphy.propagators.variational_momentum_advection.VariationalMomentumAdvection`
        3. :class:`~struphy.propagators.variational_entropy_evolve.VariationalEntropyEvolve`
        """
        doc = rf"""**1. VariationalDensityEvolve:**

{VariationalDensityEvolve.__doc__}

**2. VariationalMomentumAdvection:**

{VariationalMomentumAdvection.__doc__}

**3. VariationalEntropyEvolve:**

{VariationalEntropyEvolve.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""VariationalCompressibleFluid is the entropy-based compressible fluid
        model in the variational family. It is the natural non-magnetic
        counterpart to the full variational MHD models."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a variational compressible-fluid model:

        .. code-block:: python

            from struphy.models import VariationalCompressibleFluid

            model = VariationalCompressibleFluid()
            model.fluid.density
            model.fluid.velocity
            model.fluid.entropy
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - compressible fluid benchmarks with entropy transport
        - testing the variational fluid propagator stack
        - non-magnetic hydrodynamics with conservative thermodynamics"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - magnetic-field evolution or MHD coupling
        - pressureless or barotropic-only reductions
        - kinetic or particle-based transport physics"""
