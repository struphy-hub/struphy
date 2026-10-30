import cunumpy as xp

from struphy import BaseUnits
from struphy.feec.mass import AverageOperator
from struphy.models.guiding_center import GuidingCenter
from struphy.models.scalars import FunctionScalarFEEC
from struphy.models.species import (
    DiagnosticSpecies,
    FieldSpecies,
)
from struphy.models.variables import FEECVariable
from struphy.pic.accumulation.kernels.gc_density_0form import gc_density_0form
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.propagators.base import Propagator
from struphy.propagators.gyrokinetic_poisson_solve import GyrokineticPoissonSolve


class DriftKineticElectrostaticAdiabatic(GuidingCenter):
    r"""Electrostatic drift-kinetic model for a single ion species with adiabatic electrons.

    Evolves the guiding-center distribution :math:`f(\mathbf{X}, v_\parallel, \mu, t)` under
    both the :math:`\mathbf{E}^* \times \mathbf{B}` drift and parallel streaming along the
    background magnetic field. Electrons are not evolved kinetically; instead they contribute
    through an adiabatic response term in the quasi-neutrality (Poisson) equation. The
    optional control-variate method can be activated for the field solve. Extends
    :class:`~struphy.models.GuidingCenter` by the electrostatic potential (in :math:`\mathbf E^*`) and the
    quasi-neutrality equation.

    Parameters
    ----------
    base_units : BaseUnits
        Base units for normalization (default: ``BaseUnits(kBT=1.0)``).
    charge_number : int
        Charge number (in units of the positive elementary charge) of the ion species (default: 1).
    mass_number : float
        Mass number (in units of the proton mass) of the ion species (default: 1.0).
    epsilon : float, optional
        Normalized inverse cyclotron frequency: :math:`1 / (\hat{\Omega}_\mathrm{c} \hat{t})`.
        If ``None``, computed from ``base_units`` and the charge/mass numbers.
    """

    ## additional species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    class Diagnostics(DiagnosticSpecies):
        def __init__(
            self,
        ):
            self.rho = FEECVariable(space="H1")
            self.init_variables()

    ## additional propagators

    class Propagators(GuidingCenter.Propagators):
        """Propagators of :class:`~struphy.models.GuidingCenter` (with the potential ``phi``) plus ``gc_poisson``
        (quasi-neutrality equation with adiabatic electrons).

        Parameters
        ----------
        phi : FEECVariable, optional
            Electric potential, solved by ``gc_poisson`` and used by the pushers.
        rho : ParticlesToGrid, optional
            Guiding-center charge deposition (right-hand side of the quasi-neutrality equation).
        epsilon : float
            Normalized cyclotron period (equation parameter of the ions).
        Z : int
            Charge number of the ions.
        diagnostic : FEECVariable, optional
            Variable for the charge density of the diagnostic Poisson solve.
        **kwargs
            Passed to :class:`GuidingCenter.Propagators <struphy.models.GuidingCenter.Propagators>`.
        """

        # both pushers move the guiding centers; with the Poisson solve between them, a Strang step
        # bxe(dt/2), poisson, para(dt), poisson, bxe(dt/2) pushes with the potential of the current positions
        # (the two bxe(dt/2) of consecutive steps act as one bxe(dt) with the potential frozen)
        splitting_order = ("push_bxe", "gc_poisson", "push_parallel")

        def __init__(
            self,
            phi: FEECVariable = None,
            rho: ParticlesToGrid = None,
            epsilon: float = 1.0,
            Z: int = 1,
            diagnostic: FEECVariable | None = None,
            **kwargs,
        ):
            # the pushers of GuidingCenter with the electric field -grad phi in E*
            super().__init__(phi=phi, **kwargs)
            self.gc_poisson = GyrokineticPoissonSolve(rho=rho, epsilon=epsilon, Z=Z, diagnostic=diagnostic)

    # propagators renamed to the names of GuidingCenter (old name -> new name), for the options in old run metadata
    _renamed_propagators = {"push_gc_bxe": "push_bxe", "push_gc_para": "push_parallel"}

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    em_fields: EMFields
    diagnostics: Diagnostics
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        charge_number: int = 1,
        mass_number: float = 1.0,
        epsilon: float = None,
        alpha: float = None,
        use_diagnostic_poisson: bool = False,
    ):
        # 0. store input parameters
        self._store_params(locals())

        # 1.-5. see GuidingCenter
        super().__init__(base_units, charge_number, mass_number, epsilon)

    def _create_species(self):
        """:meta private:"""
        self.em_fields = self.EMFields()
        super()._create_species()
        if self.params["use_diagnostic_poisson"]:
            self.diagnostics = self.Diagnostics()

    def _propagator_args(self):
        """:meta private:"""
        rho = ParticlesToGrid(
            self.kinetic_ions.var,
            "H1",
            gc_density_0form,
        )
        diagnostic = self.diagnostics.rho if self.params["use_diagnostic_poisson"] else None
        return {
            **super()._propagator_args(),
            "phi": self.em_fields.phi,
            "rho": rho,
            "epsilon": self.kinetic_ions.equation_params.epsilon,
            "Z": self.kinetic_ions.charge_number,
            "diagnostic": diagnostic,
        }

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.gc_poisson.variables.phi = self.em_fields.phi

    def _create_scalars(self):
        """:meta private:"""
        scalars = super()._create_scalars()
        en_phi = FunctionScalarFEEC(self._compute_en_phi)
        en_tot = en_phi + scalars.pop("en_tot")
        return {
            "phi_integral": FunctionScalarFEEC(self._compute_phi_integral_ITGtest),
            "en_phi": en_phi,
            **scalars,
            "en_tot": en_tot,
        }

    @property
    def velocity_scale(self):
        return "thermal"

    def post_allocate(self):
        """Solve initial Poisson equation and allocate helper arrays.

        :meta private:
        """
        self._tmp3 = xp.empty(1, dtype=float)
        self._e_field = Propagator.derham.V1.zeros()

        self.propagators.gc_poisson(1.0)

        # allocate temporary tabs for scalars computation
        derham = self.em_fields.phi.spline.derham
        self.phi_squared_averaged_feec = FEECVariable("H1")
        self.phi_squared_averaged_feec.allocate(derham, self.em_fields.phi.spline.domain)
        self.phi_squared = self.em_fields.phi.spline.vector.space.zeros()
        self.phi_squared_average = self.em_fields.phi.spline.vector.space.zeros()
        average_matrix_z = AverageOperator(derham, direction=2)
        average_matrix_teta = AverageOperator(derham, direction=1)
        self.average_matrix = average_matrix_z @ average_matrix_teta

    def _compute_en_phi(self):
        phi = self.em_fields.phi.spline.vector
        epsilon = self.kinetic_ions.equation_params.epsilon

        e1 = Propagator.derham.grad.dot(-phi, out=self._e_field)
        en_phi1 = 0.5 * Propagator.mass_ops.M1gyro.dot_inner(e1, e1)
        en_phi = 0.5 / epsilon**2 * Propagator.mass_ops.M0ad.dot_inner(phi, phi)
        return en_phi + en_phi1

    def _compute_phi_integral_ITGtest(self):
        phi = self.em_fields.phi.spline.vector
        self.phi_squared._data = phi._data**2
        self.average_matrix.dot(self.phi_squared, out=self.phi_squared_average)
        self.phi_squared_averaged_feec.spline.vector = self.phi_squared_average
        return self.phi_squared_averaged_feec.spline(0.5, 0.5, 0.5)[0, 0, 0]

    ## default parameters
    def generate_default_parameter_file(self, path=None, prompt=True):
        params_path = super().generate_default_parameter_file(path=path, prompt=prompt)
        new_file = []
        with open(params_path, "r") as f:
            for line in f:
                if "BaseUnits(" in line:
                    new_file += ["base_units = BaseUnits(kBT=1.0)\n"]
                elif "saving_params = " in line:
                    new_file += ["\nbinplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))\n"]
                    new_file += ["saving_params = SavingParameters(binning_plots=(binplot,))\n\n"]
                else:
                    new_file += [line]

        with open(params_path, "w") as f:
            for line in new_file:
                f.write(line)
        return params_path

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Drift-kinetic equation:

        .. math::

            \frac{\partial f}{\partial t} + \left[ v_\parallel \frac{\mathbf{B}^*}{B^*_\parallel} + \frac{\mathbf{E}^* \times \mathbf{b}_0}{B^*_\parallel} \right] \cdot \frac{\partial f}{\partial \mathbf{X}} + \left[ \frac{1}{\varepsilon} \frac{\mathbf{B}^*}{B^*_\parallel} \cdot \mathbf{E}^* \right] \cdot \frac{\partial f}{\partial v_\parallel} = 0

        Poisson equation:

        .. math::

            -\nabla_\perp \cdot \left( \frac{n_0}{|B_0|^2} \nabla_\perp \phi \right) + \frac{1}{\varepsilon} n_0 \left( 1 + \frac{1}{Z \varepsilon} \frac{1}{T_0} \phi \right) = \frac{1}{\varepsilon} \int f B^*_\parallel \, \textnormal{d} v_\parallel \textnormal{d} \mu

        where :math:`f(\mathbf{X}, v_\parallel, \mu, t)` is the guiding center distribution and

        .. math::

            \mathbf{E}^* = -\nabla \phi - \varepsilon \mu \nabla |B_0|, \qquad \mathbf{B}^* = \mathbf{B}_0 + \varepsilon v_\parallel \nabla \times \mathbf{b}_0, \qquad B^*_\parallel = \mathbf{B}^* \cdot \mathbf{b}_0

        The control variate method can be activated in the Poisson equation; if enabled, the following Poisson equation is solved:

        .. math::

            -\nabla_\perp \cdot \left( \frac{n_0}{|B_0|^2} \nabla_\perp \phi \right) +  \frac{1}{Z \varepsilon^2} \frac{n_0}{T_0} \phi = \frac{1}{\varepsilon} \int (f - f_0) B^*_\parallel \, \textnormal{d} v_\parallel \textnormal{d} \mu
        """

    @classmethod
    def doc_normalization(cls):
        r"""The reference speed is the ion thermal speed and the electrostatic
        fields are scaled accordingly:

        .. math::

            \hat v = \hat v_i,\qquad \hat E = \hat v_i \hat B,\qquad \hat\phi = \hat E \hat x.

        The small parameter is :math:`\varepsilon = 1/(\hat\Omega_c\hat t)`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Integral of the averaged squared potential (ITG diagnostic): ``phi_integral``
        - Field energy: ``en_phi``
        - Parallel kinetic energy: ``en_f``
        - Magnetic-moment energy: ``en_mu``
        - Total energy: ``en_tot`` = ``en_phi + en_f + en_mu``
        - Lost markers: ``n_lost_particles``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence).
        The Poisson solve sits between the two pushers, which both move the guiding centers. With Strang
        splitting, bxe(dt/2), poisson, para(dt), poisson, bxe(dt/2), the parallel push and the closing bxe(dt/2) use a
        freshly solved potential; the two bxe(dt/2) at a step boundary act as one bxe(dt) with the potential solved
        before it. With Lie-Trotter splitting the parallel push uses a fresh potential and the E* x b push the one of
        the previous step. Since a time step ends with a push, the saved potential and ``en_phi`` belong to the marker
        positions before that last push."""
        return cls.propagators_doc()

    @classmethod
    def doc_long_description(cls):
        r"""This model is an electrostatic drift-kinetic reduction for strongly
        magnetized ions in a fixed magnetic equilibrium. Electrons are not
        evolved kinetically; instead they enter through the adiabatic response
        in the quasi-neutrality solve. The implementation supports control
        variates for the field solve."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the drift-kinetic adiabatic-electron model:

        .. code-block:: python

            from struphy.models import DriftKineticElectrostaticAdiabatic

            model = DriftKineticElectrostaticAdiabatic()
            model.em_fields.phi
            model.kinetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - electrostatic drift-kinetic ion turbulence studies
        - strongly magnetized plasmas with adiabatic electrons
        - guiding-center PIC verification in realistic magnetic geometry
        - low-frequency regimes where full gyrophase resolution is unnecessary"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - fully electromagnetic dynamics with evolving magnetic perturbations
        - electron kinetic effects beyond the adiabatic closure
        - problems that require resolving full cyclotron motion
        - multi-species kinetic coupling without extending the model"""
