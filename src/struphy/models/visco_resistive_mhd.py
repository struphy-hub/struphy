import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.feec.projectors import L2Projector
from struphy.feec.variational_utilities import (
    InternalEnergyEvaluator,
)
from struphy.io.options import BaseUnits, LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
    FluidSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.polar.basic import PolarVector
from struphy.propagators import (
    propagators_fields,
)
from struphy.propagators.base import Propagator

rank = MPI.COMM_WORLD.Get_rank()


class ViscoResistiveMHD(StruphyModel):
    r"""Full (non-linear) visco-resistive MHD equations discretized with a variational method.

    where the internal energy per unit mass is :math:`\mathcal U(\rho) = \rho^{\gamma-1} \exp(s / \rho)`,
    and :math:`\mu_a(\mathbf x)` and :math:`\eta_a(\mathbf x)` are artificial viscosity and resistivity coefficients.

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.VariationalDensityEvolve`
    2. :class:`~struphy.propagators.propagators_fields.VariationalMomentumAdvection`
    3. :class:`~struphy.propagators.propagators_fields.VariationalEntropyEvolve`
    4. :class:`~struphy.propagators.propagators_fields.VariationalMagFieldEvolve`
    5. :class:`~struphy.propagators.propagators_fields.VariationalViscosity`
    6. :class:`~struphy.propagators.propagators_fields.VariationalResistivity`

    :ref:`Model info <add_model>`:
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Fluid"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.b_field = FEECVariable(space="Hdiv")
            self.init_variables()

    class MHD(FluidSpecies):
        def __init__(self, mass_number: float = 1.0):
            self.density = FEECVariable(space="L2")
            self.velocity = FEECVariable(space="H1vec")
            self.entropy = FEECVariable(space="L2")
            self.init_variables(mass_number=mass_number)

    ## propagators

    class Propagators:
        def __init__(
            self,
            with_viscosity: bool = True,
            with_resistivity: bool = True,
        ):
            self.variat_dens = propagators_fields.VariationalDensityEvolve()
            self.variat_mom = propagators_fields.VariationalMomentumAdvection()
            self.variat_ent = propagators_fields.VariationalEntropyEvolve()
            self.variat_mag = propagators_fields.VariationalMagFieldEvolve()
            if with_viscosity:
                self.variat_viscous = propagators_fields.VariationalViscosity()
            if with_resistivity:
                self.variat_resist = propagators_fields.VariationalResistivity()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        mass_number: float = 1.0,
        with_viscosity: bool = True,
        with_resistivity: bool = True,
    ):

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.mhd = self.MHD(mass_number=mass_number)

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(
            with_viscosity=with_viscosity,
            with_resistivity=with_resistivity,
        )

        # 4. assign variables to propagators
        self.propagators.variat_dens.variables.rho = self.mhd.density
        self.propagators.variat_dens.variables.u = self.mhd.velocity
        self.propagators.variat_mom.variables.u = self.mhd.velocity
        self.propagators.variat_ent.variables.s = self.mhd.entropy
        self.propagators.variat_ent.variables.u = self.mhd.velocity
        self.propagators.variat_mag.variables.u = self.mhd.velocity
        self.propagators.variat_mag.variables.b = self.em_fields.b_field
        if with_viscosity:
            self.propagators.variat_viscous.variables.s = self.mhd.entropy
            self.propagators.variat_viscous.variables.u = self.mhd.velocity
        if with_resistivity:
            self.propagators.variat_resist.variables.s = self.mhd.entropy
            self.propagators.variat_resist.variables.b = self.em_fields.b_field

        # define scalars for update_scalar_quantities
        self.add_scalar("en_U")
        self.add_scalar("en_thermo")
        self.add_scalar("en_mag")
        self.add_scalar("en_tot")
        self.add_scalar("dens_tot")
        self.add_scalar("entr_tot")
        self.add_scalar("tot_div_B")

    @property
    def bulk_species(self):
        return self.mhd

    @property
    def velocity_scale(self):
        return "alfvén"

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

        self._tmp_div_B = Propagator.derham.V3pol.zeros()

    def update_scalar_quantities(self):
        rho = self.mhd.density.spline.vector
        u = self.mhd.velocity.spline.vector
        s = self.mhd.entropy.spline.vector
        b = self.em_fields.b_field.spline.vector

        en_U = 0.5 * Propagator.mass_ops.WMM.massop.dot_inner(u, u)
        self.update_scalar("en_U", en_U)

        en_mag = 0.5 * Propagator.mass_ops.M2.dot_inner(b, b)
        self.update_scalar("en_mag", en_mag)

        en_thermo = self.update_thermo_energy()

        en_tot = en_U + en_thermo + en_mag
        self.update_scalar("en_tot", en_tot)

        dens_tot = self._ones.inner(rho)
        self.update_scalar("dens_tot", dens_tot)
        entr_tot = self._ones.inner(s)
        self.update_scalar("entr_tot", entr_tot)

        div_B = Propagator.derham.div.dot(b, out=self._tmp_div_B)
        L2_div_B = Propagator.mass_ops.M3.dot_inner(div_B, div_B)
        self.update_scalar("tot_div_B", L2_div_B)

    def update_thermo_energy(self):
        """Reuse tmp used in VariationalEntropyEvolve to compute the thermodynamical energy.

        :meta private:
        """
        rho = self.mhd.density.spline.vector
        s = self.mhd.entropy.spline.vector
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
        self.update_scalar("en_thermo", en_thermo)
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
                        "                                                                              s=model.mhd.entropy)\n",
                    ]
                elif "variat_ent.Options" in line:
                    new_file += [
                        "model.propagators.variat_ent.options = model.propagators.variat_ent.Options(model='full',\n",
                    ]
                    new_file += [
                        "                                                                            rho=model.mhd.density)\n",
                    ]
                elif "variat_viscous.Options" in line:
                    new_file += [
                        "model.propagators.variat_viscous.options = model.propagators.variat_viscous.Options(rho=model.mhd.density)\n",
                    ]
                elif "variat_resist.Options" in line:
                    new_file += [
                        "model.propagators.variat_resist.options = model.propagators.variat_resist.Options(rho=model.mhd.density)\n",
                    ]
                elif "entropy.add_background" in line:
                    new_file += ["model.mhd.density.add_background(FieldsBackground())\n"]
                    new_file += [line]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
