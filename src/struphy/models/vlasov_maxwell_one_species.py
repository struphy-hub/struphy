import copy
import logging

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarPIC, KineticEnergyPIC, Scalars
from struphy.models.species import (
    FieldSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation import accum_kernels
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.propagators.base import Propagator
from struphy.propagators.maxwell_weak_ampere import MaxwellWeakAmpere
from struphy.propagators.poisson_solve import PoissonSolve
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_vxb import PushVxB
from struphy.propagators.vlasov_ampere_coupling import VlasovAmpereCoupling
from struphy.utils.pyccel import Pyccelkernel

logger = logging.getLogger("struphy")

rank = MPI.COMM_WORLD.Get_rank()


class VlasovMaxwellOneSpecies(StruphyModel):
    """Vlasov-Maxwell equations for one kinetic species.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    charge_number: int
        Charge number (in units of the positive elementary charge) of the species (default: 1)
    mass_number: float
        Mass number (in units of Proton mass) of the species (default: 1.0)
    alpha: float, optional
        Dimensionless parameter: plasma frequency / cyclotron frequency. If None, computed from units and charge/mass numbers.
    epsilon: float, optional
        Normalized cyclotron period: 1 / (cyclotron frequency × time unit). If None, computed from units and charge/mass numbers.
    measure_gauss_law: bool
        Whether to track the Gauss-law error as a scalar quantity (default: False)
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

    ## species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.e_field = FEECVariable(space="Hcurl")
            self.b_field = FEECVariable(space="Hdiv")
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class KineticIons(ParticleSpecies):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            alpha: float = None,
            epsilon: float = None,
        ):
            self.var = PICVariable(space="Particles6D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                alpha=alpha,
                epsilon=epsilon,
            )

    ## propagators

    class Propagators:
        def __init__(
            self,
            b2_var: FEECVariable = None,
        ):
            self.maxwell = MaxwellWeakAmpere()
            self.push_eta = PushEta()
            self.push_vxb = PushVxB(b2_var=b2_var)
            self.coupling_va = VlasovAmpereCoupling()

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        alpha: float = None,
        epsilon: float = None,
        measure_gauss_law: bool = False,
    ):

        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons(
            charge_number,
            mass_number,
            alpha,
            epsilon,
        )

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(b2_var=self.em_fields.b_field)

        # 4. assign variables to propagators
        self.propagators.maxwell.variables.e = self.em_fields.e_field
        self.propagators.maxwell.variables.b = self.em_fields.b_field
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        self.propagators.push_vxb.variables.ions = self.kinetic_ions.var
        self.propagators.coupling_va.variables.e = self.em_fields.e_field
        self.propagators.coupling_va.variables.ions = self.kinetic_ions.var

        # 5. define scalars to be tracked during simulation
        electric_energy = BilinearEnergyFEEC(self.em_fields.e_field)
        magnetic_energy = BilinearEnergyFEEC(self.em_fields.b_field)
        particle_energy = KineticEnergyPIC(
            self.kinetic_ions.var,
            normalization=self.kinetic_ions.equation_params.alpha**2,
        )
        scalars_dict = {
            "en_E": electric_energy,
            "en_B": magnetic_energy,
            "en_f": particle_energy,
            "en_tot": electric_energy + magnetic_energy + particle_energy,
        }
        if measure_gauss_law:
            scalars_dict["gauss_error"] = FunctionScalarPIC(self.calculate_gauss_error, self.kinetic_ions.var)
        self.scalars = Scalars(**scalars_dict)

        # initial Poisson (not a propagator used in time stepping)
        self.initial_poisson = PoissonSolve()
        self.initial_poisson.variables.phi = self.em_fields.phi

        # property to measure violation of gauss law from control variate
        self.measure_gauss_law = measure_gauss_law

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "light"

    def allocate_helpers(self):
        """Solve initial Poisson equation.

        :meta private:
        """
        self._tmp = xp.empty(1, dtype=float)

        particles = self.kinetic_ions.var.particles

        if self.measure_gauss_law:
            self.op = Propagator.derham.grad.T @ Propagator.mass_ops.M1
            self.subcom_residual = xp.empty(shape=particles.mpi_size, dtype=float)
            self.intercom_residual = xp.empty(shape=particles.num_clones, dtype=float)

        logger.info("\nINITIAL POISSON SOLVE:")

        # use control variate method (reset weights after Poisson solve)
        particles.update_weights()

        # sanity check
        # particles.show_distribution_function([True] + [False]*5, [xp.linspace(0, 1, 32)])

        # accumulate charge density
        self.charge_accum = AccumulatorVector(
            particles,
            "H1",
            Pyccelkernel(accum_kernels.charge_density_0form),
            Propagator.mass_ops,
            Propagator.domain.args_domain,
        )

        # another sanity check: compute FE coeffs of density
        # charge_accum.show_accumulated_spline_field(Propagator.mass_ops)

        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon

        self.initial_poisson.rho = self.charge_accum
        self.initial_poisson.rho_coeffs = alpha**2 / epsilon
        self.initial_poisson.allocate()

        # Solve with dt=1. and compute electric field
        logger.info("Solving initial Poisson problem...")
        self.initial_poisson(1.0)

        phi = self.initial_poisson.variables.phi.spline.vector
        Propagator.derham.grad.dot(-phi, out=self.em_fields.e_field.spline.vector)
        logger.info("... Done.")

        # reset particle weights
        particles.weights = particles.weights_at_t0.copy()

    def calculate_gauss_error(self):
        # control variate method
        particles = self.kinetic_ions.var.particles
        particles.update_weights()
        self.charge_accum()
        rhs = self.charge_accum.vectors[0]
        # reset particle weights
        particles.weights = particles.weights_at_t0.copy()

        # non control variate method
        e = self.em_fields.e_field.spline.vector
        lhs = self.op.dot(e)

        # calculate local residual of local MPI rank
        loc_residual = xp.max(xp.abs(lhs.toarray() - rhs.toarray()))

        # logger.info(f"{MPI.COMM_WORLD.Get_rank() = }, {xp.max(xp.abs(lhs.toarray())) = }")
        # logger.info(f"{MPI.COMM_WORLD.Get_rank() = }, {xp.max(xp.abs(rhs.toarray())) = }")
        # logger.info(f"{loc_residual = }")

        # return the maximum residual across all MPI rank
        particles._gather_scalar_in_subcomm_array(scalar=loc_residual, out=self.subcom_residual)
        particles._gather_scalar_in_intercomm_array(scalar=loc_residual, out=self.intercom_residual)

        return xp.max([xp.max(self.subcom_residual), xp.max(self.intercom_residual)])

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "coupling_va.Options" in line:
                    new_file += [line]
                    new_file += ["model.initial_poisson.options = model.initial_poisson.Options()\n"]
                elif "saving_params = " in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["saving_params = SavingParameters(binning_plots=(binplot,))\n\n"]
                elif "VlasovMaxwellOneSpecies()" in line:
                    new_file += ["\nmodel = VlasovMaxwellOneSpecies(measure_gauss_law=True)\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Vlasov equation:

        .. math::

            \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \frac{1}{\varepsilon} \left( \mathbf{E} + \mathbf{v} \times \left( \mathbf{B} + \mathbf{B}_0 \right) \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0

        Ampère's law:

        .. math::

            -\frac{\partial \mathbf{E}}{\partial t} + \nabla \times \mathbf{B} = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v}

        Faraday's law:

        .. math::

            \frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0

        where :math:`Z=-1` and :math:`A=1/1836` for electrons.

        At initial time the weak Poisson equation is solved once to weakly satisfy Gauss' law,

        .. math::

            \int_{\Omega} \nabla \psi^{\top} \cdot \nabla \phi \, \textrm{d} \mathbf{x} &= \frac{\alpha^2}{\varepsilon} \int_{\Omega} \int_{\mathbb{R}^3} \psi \, (f - f_0) \, \text{d}^3 \mathbf{v} \, \textrm{d} \mathbf{x} \qquad \forall \ \psi \in H^1
            \\[2mm]
            \mathbf{E}(t=0) &= -\nabla \phi(t=0)

        Moreover, it is assumed that

        .. math::

            \nabla \times \mathbf{B}_0 = \frac{\alpha^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v}

        where :math:`\mathbf{B}_0` is the static equilibrium magnetic field.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The model uses the light speed as reference velocity:

        .. math::

            \hat v = c,\qquad \hat E = \hat B \hat v,\qquad \hat\phi = \hat E \hat x.

        The species parameters are :math:`\alpha=\hat\Omega_p/\hat\Omega_c` and
        :math:`\varepsilon=1/(\hat\Omega_c\hat t)`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Electric field energy: ``en_E``
        - Magnetic field energy: ``en_B``
        - Particle kinetic energy: ``en_f``
        - Total energy: ``en_tot``
        - Optional Gauss-law diagnostic: ``gauss_error``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.maxwell_weak_ampere.MaxwellWeakAmpere`
        2. :class:`~struphy.propagators.push_eta.PushEta`
        3. :class:`~struphy.propagators.push_vxb.PushVxB`
        4. :class:`~struphy.propagators.vlasov_ampere_coupling.VlasovAmpereCoupling`
        """
        doc = rf"""**1. propagators.maxwell.Maxwell:**

{MaxwellWeakAmpere.__doc__}

**2. PushEta:**

{PushEta.__doc__}

**3. PushVxB:**

{PushVxB.__doc__}

**4. VlasovAmpereCoupling:**

{VlasovAmpereCoupling.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""VlasovMaxwellOneSpecies is the fully electromagnetic one-species PIC
        model in Struphy. It evolves particles and fields self-consistently and
        supports an optional control-variate formulation for the field coupling."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize a Vlasov-Maxwell model:

        .. code-block:: python

            from struphy.models import VlasovMaxwellOneSpecies

            model = VlasovMaxwellOneSpecies()
            model.em_fields.e_field
            model.em_fields.b_field
            model.kinetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - self-consistent electromagnetic kinetic simulations
        - one-species PIC benchmarks
        - wave-particle interaction studies with evolving magnetic fields
        - verification of the full Vlasov-Maxwell splitting"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - multi-species plasma dynamics without extension
        - collisional kinetic closures
        - reduced electrostatic-only models where magnetic evolution is unnecessary
        - linearized delta-f studies that should use the dedicated linear models"""
