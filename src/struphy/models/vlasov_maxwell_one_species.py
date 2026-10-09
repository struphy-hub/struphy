import copy
import logging

import cunumpy as xp
from maybempi import MPI

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.base import StruphyModel
from struphy.models.scalars import BilinearEnergyFEEC, FunctionScalarFEEC, KineticEnergyPIC, Scalars
from struphy.models.species import (
    FieldSpecies,
    ParticleSpecies,
)
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation.kernels.charge_density_0form import charge_density_0form
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.propagators.base import Propagator
from struphy.propagators.maxwell_weak_ampere import MaxwellWeakAmpere
from struphy.propagators.poisson_solve import PoissonSolve
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_vxb import PushVxB
from struphy.propagators.vlasov_ampere_coupling import VlasovAmpereCoupling

logger = logging.getLogger("struphy")


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
            # the MPI reduction (max over ranks) is done in calculate_gauss_error, not by summing the scalar
            scalars_dict["gauss_error"] = FunctionScalarFEEC(self.calculate_gauss_error)
        self.scalars = Scalars(**scalars_dict)

        # initial Poisson (not a propagator used in time stepping)
        alpha = self.kinetic_ions.equation_params.alpha
        epsilon = self.kinetic_ions.equation_params.epsilon
        particles_to_grid = ParticlesToGrid(
            self.kinetic_ions.var,
            "H1",
            charge_density_0form,
        )

        self.initial_poisson = PoissonSolve(
            rho=particles_to_grid,
            rho_coeffs=alpha**2 / epsilon,
        )
        self.initial_poisson.variables.phi = self.em_fields.phi

        # property to measure violation of gauss law from control variate
        self.measure_gauss_law = measure_gauss_law

    @property
    def bulk_species(self):
        return self.kinetic_ions

    @property
    def velocity_scale(self):
        return "light"

    def post_allocate(self):
        """Solve initial Poisson equation.

        :meta private:
        """
        self._tmp = xp.empty(1, dtype=float)

        particles = self.kinetic_ions.var.particles

        if self.measure_gauss_law:
            self.op = Propagator.derham.grad.T @ Propagator.mass_ops.M1

        logger.info("\nINITIAL POISSON SOLVE:")

        # charge of f - f0 for the Poisson right-hand side (see below for the weights of the time stepping)
        particles.update_weights()

        self.initial_poisson.allocate()

        # keep the AccumulatorVector built by the propagator for the Gauss-law diagnostic below
        self.charge_accum = self.initial_poisson.sources[0]

        # Solve with dt=1. and compute electric field
        logger.info("Solving initial Poisson problem...")
        self.initial_poisson(1.0)

        phi = self.initial_poisson.variables.phi.spline.vector
        Propagator.derham.grad.dot(-phi, out=self.em_fields.e_field.spline.vector)
        # The pushers evaluate e at the markers, including in the ghost cells: grad.dot does not fill them,
        # so without this sync the first time step pushes with a wrong field (and breaks energy conservation).
        self.em_fields.e_field.spline.vector.update_ghost_regions()
        logger.info("... Done.")

        # The Poisson right-hand side is the charge of f - f0 in any case; the time stepping uses the delta-f
        # weights only with the control variate (and the scalars must be computed with the same weights at t=0).
        if not particles.control_variate:
            particles.weights = particles.weights0.copy()

    def calculate_gauss_error(self):
        r"""Maximum norm of the weak Gauss-law residual

        .. math::

            \mathbb G^\top \mathbb M^1 \mathbf e + \frac{\alpha^2}{\varepsilon} \boldsymbol \rho\,,

        where :math:`\boldsymbol \rho` is the charge density of :math:`f - f_0` deposited as in the initial
        Poisson solve. Since :math:`\mathbf e = -\mathbb G \boldsymbol \phi` it vanishes at :math:`t=0`,
        up to the net charge (the kernel of the periodic Poisson problem)."""
        # control variate method
        particles = self.kinetic_ions.var.particles
        particles.update_weights()
        self.charge_accum()
        rho = self.charge_accum.vectors[0]
        # restore the weights of the time stepping (full-f without the control variate)
        if not particles.control_variate:
            particles.weights = particles.weights0.copy()

        e = self.em_fields.e_field.spline.vector
        residual = self.op.dot(e)
        residual += self.initial_poisson.coeffs[0] * rho

        # maximum residual over the local MPI rank, then over all ranks of the domain decomposition
        # (all clones hold the same accumulated charge density)
        self._tmp[0] = xp.max(xp.abs(residual.toarray()))
        if Propagator.derham.comm is not None:
            Propagator.derham.comm.Allreduce(MPI.IN_PLACE, self._tmp, op=MPI.MAX)

        return self._tmp[0]

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

            \int_{\Omega} \nabla \psi^{\top} \cdot \nabla \phi \, \textrm{d} \mathbf{x} = \frac{\alpha^2}{\varepsilon} \int_{\Omega} \int_{\mathbb{R}^3} \psi \, (f - f_0) \, \text{d}^3 \mathbf{v} \, \textrm{d} \mathbf{x} \qquad \forall \ \psi \in H^1
            \\[2mm]
            \mathbf{E}(t=0) = -\nabla \phi(t=0)

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
