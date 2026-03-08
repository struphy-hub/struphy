import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.species import (
    FieldSpecies,
    FluidSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation import accum_kernels
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.propagators import (
    propagators_coupling,
    propagators_fields,
    propagators_markers,
)
from struphy.propagators.base import Propagator
from struphy.utils.pyccel_utils import Pyccelkernel

rank = MPI.COMM_WORLD.Get_rank()


class ColdPlasmaVlasov(StruphyModel):
    r"""Cold plasma hybrid model.

    :ref:`normalization`:

    .. math::

        \hat v = c\,,\qquad \hat E = c \hat B \,,\qquad \hat f = \frac{\hat n}{c^3} \,.

    :ref:`Equations <gempic>`:

    .. math::

        &\frac{\partial f}{\partial t} + \mathbf{v} \cdot \, \nabla f + \frac{1}{\varepsilon_\textnormal{h}}\Big[ \mathbf{E} + \mathbf{v} \times \left( \mathbf{B} + \mathbf{B}_0 \right) \Big]
            \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,,
        \\[2mm]
        \frac{1}{n_0} &\frac{\partial \mathbf j_\textnormal{c}}{\partial t} = \frac{1}{\varepsilon_\textnormal{c}} \mathbf E + \frac{1}{\varepsilon_\textnormal{c} n_0} \mathbf j_\textnormal{c} \times \mathbf B_0\,,
        \\[2mm]
        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,,
        \\[2mm]
        -&\frac{\partial \mathbf E}{\partial t} + \nabla\times\mathbf B =
        \frac{\alpha^2}{\varepsilon_\textnormal{h}} \left( \mathbf j_\textnormal{c} + \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v} \right) \,,

    where :math:`(n_0,\mathbf B_0)` denotes a (inhomogeneous) background and

    .. math::

        \alpha = \frac{\hat \Omega_\textnormal{p,cold}}{\hat \Omega_\textnormal{c,cold}}\,, \qquad \varepsilon_\textnormal{c} = \frac{1}{\hat \Omega_\textnormal{c,cold} \hat t}\,, \qquad \varepsilon_\textnormal{h} = \frac{1}{\hat \Omega_\textnormal{c,hot} \hat t} \,.

    At initial time the Poisson equation is solved once to weakly satisfy the Gauss law:

    .. math::

        \begin{align}
            \nabla \cdot \mathbf{E} & = \nu \frac{\alpha^2}{\varepsilon_\textnormal{h}} \int_{\mathbb{R}^3} f \, \text{d}^3 \mathbf{v}\,.
        \end{align}

    :ref:`propagators` (called in sequence):

    1. :class:`~struphy.propagators.propagators_fields.Maxwell`
    2. :class:`~struphy.propagators.propagators_fields.OhmCold`
    3. :class:`~struphy.propagators.propagators_fields.JxBCold`
    4. :class:`~struphy.propagators.propagators_markers.PushVxB`
    5. :class:`~struphy.propagators.propagators_markers.PushEta`
    6. :class:`~struphy.propagators.propagators_coupling.VlasovAmpere`
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Hybrid"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class ThermalElectrons(FluidSpecies):
        def __init__(self):
            self.current = FEECVariable(space="Hcurl")
            self.init_variables()

    class HotElectrons(ParticleSpecies):
        def __init__(self):
            self.var = PICVariable(space="Particles6D")
            self.init_variables()

    ## propagators

    class Propagators:
        def __init__(self):
            self.maxwell = propagators_fields.Maxwell()
            self.ohm = propagators_fields.OhmCold()
            self.jxb = propagators_fields.JxBCold()
            self.push_eta = propagators_markers.PushEta()
            self.push_vxb = propagators_markers.PushVxB()
            self.coupling_va = propagators_coupling.VlasovAmpere()

    ## abstract methods

    def __init__(self):

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.thermal_elec = self.ThermalElectrons()
        self.hot_elec = self.HotElectrons()

        # 2. instantiate all propagators
        self.propagators = self.Propagators()

        # 3. assign variables to propagators
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field

        self.propagators.ohm.variables.j = self.thermal_elec.current
        self.propagators.ohm.variables.e = self.em_fields.e_field

        self.propagators.jxb.variables.j = self.thermal_elec.current

        self.propagators.push_eta.variables.var = self.hot_elec.var
        self.propagators.push_vxb.variables.ions = self.hot_elec.var

        self.propagators.coupling_va.variables.e = self.em_fields.e_field
        self.propagators.coupling_va.variables.ions = self.hot_elec.var

        # define scalars for update_scalar_quantities
        self.add_scalar("en_E")
        self.add_scalar("en_B")
        self.add_scalar("en_J")
        self.add_scalar("en_f", compute="from_particles", variable=self.hot_elec.var)
        self.add_scalar("en_tot")

        # initial Poisson (not a propagator used in time stepping)
        self.initial_poisson = propagators_fields.Poisson()
        self.initial_poisson.variables.phi = self.em_fields.phi

    @property
    def bulk_species(self):
        return self.thermal_elec

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self, verbose: bool = False):
        """Solve initial Poisson equation.

        :meta private:
        """
        # helper fields
        self._tmp = xp.empty(1, dtype=float)

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\nINITIAL POISSON SOLVE:")

        # use control variate method
        particles = self.hot_elec.var.particles
        particles.update_weights()

        # sanity check
        # self.pointer['species1'].show_distribution_function(
        #     [True] + [False]*5, [xp.linspace(0, 1, 32)])

        # accumulate charge density
        charge_accum = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels.charge_density_0form),
            Propagator.mass_ops,
            Propagator.domain.args_domain,
        )

        # another sanity check: compute FE coeffs of density
        # charge_accum.show_accumulated_spline_field(Propagator.mass_ops)

        alpha = self.hot_elec.equation_params.alpha
        epsilon = self.hot_elec.equation_params.epsilon

        self.initial_poisson.options.rho = charge_accum
        self.initial_poisson.options.rho_coeffs = alpha**2 / epsilon
        self.initial_poisson.allocate()

        # Solve with dt=1. and compute electric field
        if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
            print("\nSolving initial Poisson problem...")
        self.initial_poisson(1.0)

        phi = self.initial_poisson.variables.phi.spline.vector
        Propagator.derham.grad.dot(-phi, out=self.em_fields.e_field.spline.vector)
        if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
            print("... Done.")

    def update_scalar_quantities(self):
        # e*M1*e/2
        e = self.em_fields.e_field.spline.vector
        en_E = 0.5 * Propagator.mass_ops.M1.dot_inner(e, e)
        self.update_scalar("en_E", en_E)

        # alpha^2 / 2 / N * sum_p w_p v_p^2
        particles = self.hot_elec.var.particles
        alpha = self.hot_elec.equation_params.alpha
        self._tmp[0] = (
            alpha**2
            / (2 * particles.Np)
            * xp.dot(
                particles.markers_wo_holes[:, 3] ** 2
                + particles.markers_wo_holes[:, 4] ** 2
                + particles.markers_wo_holes[:, 5] ** 2,
                particles.markers_wo_holes[:, 6],
            )
        )
        self.update_scalar("en_f", self._tmp[0])

        # en_tot = en_w + en_e
        self.update_scalar("en_tot", en_E + self._tmp[0])

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "coupling_va.Options" in line:
                    new_file += [line]
                    new_file += ["model.initial_poisson.options = model.initial_poisson.Options()\n"]
                elif "set_save_data" in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["model.hot_elec.set_save_data(binning_plots=(binplot,))\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
