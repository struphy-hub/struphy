import logging

from struphy import BaseUnits
from struphy.models.scalars import FunctionScalarFEEC
from struphy.models.species import FieldSpecies
from struphy.models.variables import FEECVariable
from struphy.models.vlasov import Vlasov
from struphy.pic.accumulation.kernels.charge_density_0form import charge_density_0form
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.propagators.adiabatic_poisson_solve import AdiabaticPoissonSolve
from struphy.propagators.base import Propagator
from struphy.propagators.push_v_in_force_field import PushVinForceField

logger = logging.getLogger("struphy")


class VlasovPoissonAdiabatic(Vlasov):
    r"""Electrostatic full-orbit (6D) model for a single ion species with adiabatic electrons.

    The full-orbit counterpart of :class:`~struphy.models.DriftKineticElectrostaticAdiabatic`:
    the ions are not reduced to guiding centers, so the polarization density is not modelled
    but comes out of the ion orbits. The electrons enter through the adiabatic (Boltzmann) response
    in the Poisson equation, optionally with the same flux-surface average as in the drift-kinetic model.
    Extends :class:`~struphy.models.Vlasov` by the electrostatic force and the Poisson equation.

    Parameters
    ----------
    base_units : BaseUnits
        Base units for normalization (default: ``BaseUnits(kBT=1.0)``).
    charge_number : int
        Charge number (in units of the positive elementary charge) of the ion species (default: 1).
    mass_number : float
        Mass number (in units of the proton mass) of the ion species (default: 1.0).
    alpha : float, optional
        Ratio of plasma to cyclotron frequency :math:`\hat \omega_p / \hat \Omega_c`.
        If ``None``, computed from ``base_units`` and the charge/mass numbers.
    epsilon : float, optional
        Normalized inverse cyclotron frequency :math:`1 / (\hat{\Omega}_\mathrm{c} \hat{t})`.
        If ``None``, computed from ``base_units`` and the charge/mass numbers.
    """

    ## additional species

    class EMFields(FieldSpecies):
        def __init__(self):
            self.phi = FEECVariable(space="H1")
            self.init_variables()

    ## additional propagators

    class Propagators(Vlasov.Propagators):
        """Propagators of :class:`~struphy.models.Vlasov` plus ``poisson`` (Poisson equation with adiabatic electrons)
        and ``push_v_E`` (electric force).

        Parameters
        ----------
        phi : FEECVariable, optional
            Electric potential, solved by ``poisson`` and used by ``push_v_E``.
        rho : ParticlesToGrid, optional
            Charge deposition of the ions (right-hand side of the Poisson equation).
        epsilon : float
            Normalized cyclotron period (equation parameter of the ions).
        alpha : float
            Ratio of plasma to cyclotron frequency (equation parameter of the ions).
        Z : int
            Charge number of the ions.
        magnetic_moment : bool
            Whether ``push_v_E`` also applies the magnetic-moment force (classical Pauli particles).
        **kwargs
            Passed to :class:`Vlasov.Propagators <struphy.models.Vlasov.Propagators>`.
        """

        # Poisson right after the drift: both E-kicks of a Strang step see the potential
        # of the current marker positions (drift-kick-kick-drift, i.e. leapfrog)
        splitting_order = ("push_eta", "poisson", "push_v_E", "push_vxb")

        def __init__(
            self,
            phi: FEECVariable = None,
            rho: ParticlesToGrid = None,
            epsilon: float = 1.0,
            alpha: float = 1.0,
            Z: int = 1,
            magnetic_moment: bool = False,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.poisson = AdiabaticPoissonSolve(rho=rho, epsilon=epsilon, alpha=alpha, Z=Z)
            self.push_v_E = PushVinForceField(potential=phi, magnetic_moment=magnetic_moment)

    # attribute types for IDEs and type checkers: the classes of this model, not those of the parent
    em_fields: EMFields
    propagators: Propagators

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(kBT=1.0),
        charge_number: int = 1,
        mass_number: float = 1.0,
        alpha: float = None,
        epsilon: float = None,
    ):
        # 0. store input parameters
        self._store_params(locals())

        # 1.-5. see Vlasov
        super().__init__(base_units, charge_number, mass_number)

    def _create_species(self):
        """:meta private:"""
        self.em_fields = self.EMFields()
        super()._create_species()

    def _propagator_args(self):
        """:meta private:"""
        rho = ParticlesToGrid(
            self.kinetic_ions.var,
            "H1",
            charge_density_0form,
        )
        return {
            "phi": self.em_fields.phi,
            "rho": rho,
            "epsilon": self.kinetic_ions.equation_params.epsilon,
            "alpha": self.kinetic_ions.equation_params.alpha,
            "Z": self.kinetic_ions.charge_number,
        }

    def _link_propagator_variables(self):
        """:meta private:"""
        super()._link_propagator_variables()
        self.propagators.poisson.variables.phi = self.em_fields.phi
        self.propagators.push_v_E.variables.var = self.kinetic_ions.var

    def _create_scalars(self):
        """:meta private:"""
        scalars = super()._create_scalars()
        en_phi = FunctionScalarFEEC(self._compute_en_phi)
        return {"en_phi": en_phi, **scalars, "en_tot": en_phi + scalars["en_f"]}

    @property
    def velocity_scale(self):
        return "thermal"

    def post_allocate(self):
        """Solve the initial Poisson equation and allocate helper arrays.

        :meta private:
        """
        logger.info("\nINITIAL POISSON SOLVE:")
        self.propagators.poisson(1.0)

        phi = self.em_fields.phi.spline.vector
        self._tmp_phi = phi.space.zeros()
        self._grad_phi = Propagator.derham.grad.codomain.zeros()

    def _compute_en_phi(self):
        r"""Field energy :math:`\frac{1}{2\alpha^2}\int |\nabla \phi|^2 + \frac{1}{2Z\varepsilon^2}\int \frac{n_0}{T_0}\phi(\phi - \langle\phi\rangle)`
        (the vacuum part only if ``quasi_neutral=False``)."""
        poisson = self.propagators.poisson
        phi = self.em_fields.phi.spline.vector
        epsilon = self.kinetic_ions.equation_params.epsilon
        alpha = self.kinetic_ions.equation_params.alpha
        Z = self.kinetic_ions.var.species.charge_number

        # the adiabatic term with the same operator as in the solve, stab_mat @ (I - A)
        en = 0.5 / (Z * epsilon**2) * phi.inner(poisson._stab_mat.dot(phi, out=self._tmp_phi))
        if not poisson.options.quasi_neutral:
            e = Propagator.derham.grad.dot(phi, out=self._grad_phi)
            en += 0.5 / alpha**2 * Propagator.mass_ops.M1.dot_inner(e, e)
        return en

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

    ## abstract methods for documentation

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Vlasov equation:

        .. math::

            \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \frac{1}{\varepsilon} \left( -\nabla \phi + \mathbf{v} \times \mathbf{B}_0 \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0

        Poisson equation with adiabatic electrons:

        .. math::

            -\frac{\varepsilon}{\alpha^2} \Delta \phi + \frac{1}{Z \varepsilon} \frac{n_0}{T_0} \left( \phi - \langle \phi \rangle \right) = \int_{\mathbb{R}^3} f \, \mathrm{d}^3 \mathbf{v} - n_0

        where :math:`f(\mathbf{x}, \mathbf{v}, t)` is the ion distribution function, :math:`n_0` the equilibrium
        ion density, :math:`T_0` the electron temperature and :math:`Z` the ion charge number.
        The vacuum term :math:`-\varepsilon \alpha^{-2} \Delta \phi` is dropped in the quasi-neutral limit
        :math:`\alpha \to \infty` (option ``quasi_neutral``). The flux-surface average
        :math:`\langle \phi \rangle` (zonal correction, as in
        :class:`~struphy.models.DriftKineticElectrostaticAdiabatic`) is optional; by default
        :math:`\langle \phi \rangle = 0`.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The reference speed is the ion thermal speed and the electrostatic
        fields are scaled accordingly:

        .. math::

            \hat v = \sqrt{k_B \hat T / m_i},\qquad \hat E = \hat v \hat B,\qquad \hat\phi = \hat E \hat x,

        with temperatures in units of :math:`k_B \hat T`. The dimensionless parameters are

        .. math::

            \varepsilon = \frac{1}{\hat\Omega_c\hat t} = \frac{\hat \rho}{\hat x}, \qquad \alpha = \frac{\hat{\omega}_\mathrm{p}}{\hat{\Omega}_\mathrm{c}},

        such that :math:`\varepsilon / \alpha` is the normalized Debye length."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation** (their sum is conserved by the model equations):

        - Field energy: ``en_phi``
          :math:`= \frac{1}{2\alpha^2} \int |\nabla \phi|^2 \, \mathrm d \mathbf x + \frac{1}{2 Z \varepsilon^2} \int \frac{n_0}{T_0} \phi (\phi - \langle \phi \rangle) \, \mathrm d \mathbf x`
          (the first term only if ``quasi_neutral=False``)
        - Kinetic energy: ``en_f`` :math:`= \frac{1}{2} \sum_p w_p |\mathbf{v}_p|^2`
          (of :math:`f - f_0` with the control variate)
        - Total energy: ``en_tot``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence).
        The Poisson solve comes right after the drift, such that with Strang splitting both velocity kicks use the
        potential of the current marker positions (drift-kick-kick-drift). Since a Strang step ends with the half drift,
        the saved potential and ``en_phi`` belong to the marker positions half a step earlier."""
        return cls.propagators_doc()

    @classmethod
    def doc_long_description(cls):
        r"""This model is the full-orbit (6D) counterpart of the electrostatic drift-kinetic model
        :class:`~struphy.models.DriftKineticElectrostaticAdiabatic`. The ions follow full Lorentz-force
        orbits in the self-consistent electrostatic field and a static background magnetic field. Electrons
        are not evolved; they enter through a linearized Boltzmann response in the Poisson equation, optionally
        without response to the flux-surface-averaged potential (zonal correction), as in the drift-kinetic model.
        The potential is recomputed from the marker positions at each time step.

        In the strong-field limit :math:`\varepsilon \to 0` with drift ordering, the linear dispersion relation
        of this model reduces to that of the drift-kinetic model (up to finite-Larmor-radius corrections of order
        :math:`k_\perp^2 \rho_i^2`), which makes it a reference for verifying guiding-center models. The time step must
        resolve the cyclotron motion, :math:`\Delta t \lesssim \varepsilon`. In the quasi-neutral limit it must also
        resolve the ion-acoustic frequency of the smallest cells, :math:`\Delta t\, c_s \pi / h \lesssim 2`
        (there is no Debye cutoff)."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the full-orbit adiabatic-electron model:

        .. code-block:: python

            from struphy.models import VlasovPoissonAdiabatic

            model = VlasovPoissonAdiabatic()
            model.em_fields.phi
            model.kinetic_ions.var

            # quasi-neutral (default) or with the vacuum term, optionally with the zonal correction
            model.propagators.poisson.options = model.propagators.poisson.Options(
                quasi_neutral=False, which_geometry="cylindrical"
            )
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - electrostatic ion dynamics with adiabatic electrons, including cyclotron and finite-Larmor-radius effects
        - ion-acoustic waves, ion Bernstein waves, drift waves and slab ITG modes in the full-orbit description
        - verification of drift-kinetic and gyrokinetic models in the strong-field limit"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - electromagnetic dynamics with evolving magnetic perturbations
        - electron kinetic effects beyond the adiabatic closure
        - long simulations in strongly magnetized plasmas, where resolving the cyclotron motion is too expensive
        - multi-species kinetic coupling without extending the model"""
