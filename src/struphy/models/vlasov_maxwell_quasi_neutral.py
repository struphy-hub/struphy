import copy
import logging

from struphy import BaseUnits
from struphy.models.scalars import FunctionScalarFEEC, Scalars
from struphy.models.species import ParticleSpecies
from struphy.models.variables import PICVariable
from struphy.models.vlasov_maxwell_one_species import VlasovMaxwellOneSpecies
from struphy.pic.accumulation.kernels.charge_density_0form import charge_density_0form
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.propagators.poisson_solve import PoissonSolve
from struphy.propagators.push_eta import PushEta
from struphy.propagators.push_vxb import PushVxB
from struphy.propagators.vlasov_maxwell_coupling import VlasovMaxwellCoupling

logger = logging.getLogger("struphy")


class VlasovMaxwellQuasiNeutral(VlasovMaxwellOneSpecies):
    """Vlasov-Maxwell equations for one kinetic species in the cyclotron normalization, with an implicit, energy
    conserving field-particle coupling that also holds in the quasi-neutral limit (no displacement current).

    Same equations as :class:`~struphy.models.vlasov_maxwell_one_species.VlasovMaxwellOneSpecies`, normalized with the
    cyclotron frequency (velocity scale ``"cyclotron"``), so that the coefficients :math:`\\lambda^2` of the
    displacement current and :math:`\\kappa^2` of the plasma current appear in Ampère's law. Ampère's law, Faraday's
    law and the electric force on the particles are solved together in one Crank-Nicolson step
    (:class:`~struphy.propagators.vlasov_maxwell_coupling.VlasovMaxwellCoupling`); ``lam=0.0`` is the quasi-neutral
    limit.

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
        It does not enter the equations of this model (which use lam and kappa = alpha lam); when overriding lam and
        kappa, alpha is not adapted.
    epsilon: float, optional
        Normalized cyclotron period: 1 / (cyclotron frequency × time unit). If None, computed from units and charge/mass
        numbers (equal to 1 for this model's cyclotron velocity scale; set -1 for electrons with the default charge
        number).
    lam: float, optional
        Cyclotron frequency × length unit / speed of light (coefficient lam^2 of the displacement current; 0 is the
        quasi-neutral limit). If None, computed from units and charge/mass numbers.
    kappa: float, optional
        Plasma frequency × length unit / speed of light (coefficient kappa^2 of the plasma current). If None, computed
        from units and charge/mass numbers.
    measure_gauss_law: bool
        Whether to track the Gauss-law error as a scalar quantity (default: False; not available for lam = 0)
    """

    ## species

    class KineticIons(ParticleSpecies):
        def __init__(
            self,
            charge_number: int = 1,
            mass_number: float = 1.0,
            alpha: float = None,
            epsilon: float = None,
            lam: float = None,
            kappa: float = None,
        ):
            self.var = PICVariable(space="Particles6D")
            self.init_variables(
                charge_number=charge_number,
                mass_number=mass_number,
                alpha=alpha,
                epsilon=epsilon,
                lam=lam,
                kappa=kappa,
            )

    ## propagators

    class Propagators:
        def __init__(
            self,
            b2_var=None,
        ):
            self.coupling = VlasovMaxwellCoupling()
            self.push_eta = PushEta()
            self.push_vxb = PushVxB(b2_var=b2_var)

    ## abstract methods

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        charge_number: int = 1,
        mass_number: float = 1.0,
        alpha: float = None,
        epsilon: float = None,
        lam: float = None,
        kappa: float = None,
        measure_gauss_law: bool = False,
    ):
        # 0. store input parameters
        self.params = copy.deepcopy(locals())

        # 1. instantiate all species
        self.em_fields = self.EMFields()
        self.kinetic_ions = self.KineticIons(charge_number, mass_number, alpha, epsilon, lam, kappa)

        # 2. derive units (must be done after instantiating species to access charge and mass numbers)
        self.setup_equation_params(base_units=base_units)

        # 3. instantiate all propagators
        self.propagators = self.Propagators(b2_var=self.em_fields.b_field)

        # 4. assign variables to propagators
        self._link_propagator_variables()

        # 5. define scalars to be tracked during simulation: the parts of the energy conserved by the coupling step
        # (with the coefficients lam^2, c1, c2, c3 of the propagator, known after its allocation)
        coupling = self.propagators.coupling
        electric_energy = FunctionScalarFEEC(lambda: coupling.energies()[0])
        magnetic_energy = FunctionScalarFEEC(lambda: coupling.energies()[1])
        particle_energy = FunctionScalarFEEC(lambda: coupling.energies()[2])
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

        # initial Poisson (not a propagator used in time stepping); its charge coefficient c2 / lam^2 is set in
        # post_allocate, from the coefficients of the coupling propagator
        particles_to_grid = ParticlesToGrid(
            self.kinetic_ions.var,
            "H1",
            charge_density_0form,
        )
        self.initial_poisson = PoissonSolve(
            rho=particles_to_grid,
            rho_coeffs=None,
            delta_f_source=True,
        )
        self.initial_poisson.variables.phi = self.em_fields.phi

        # property to measure violation of gauss law from control variate
        self.measure_gauss_law = measure_gauss_law

    @property
    def velocity_scale(self):
        return "cyclotron"

    def _link_propagator_variables(self):
        """Assign the species variables to the propagators.

        :meta private:
        """
        self.propagators.coupling.variables.e = self.em_fields.e_field
        self.propagators.coupling.variables.b = self.em_fields.b_field
        self.propagators.coupling.variables.ions = self.kinetic_ions.var
        self.propagators.push_eta.variables.var = self.kinetic_ions.var
        self.propagators.push_vxb.variables.ions = self.kinetic_ions.var

    def post_allocate(self):
        """Solve the initial Poisson equation lam^2 div E = c2 rho (Gauss' law consistent with the coupling step),
        except in the quasi-neutral limit lam = 0, where the initial electric field (zero) does not enter the steps.

        :meta private:
        """
        coupling = self.propagators.coupling
        self._quasi_neutral = coupling._lam2 == 0.0
        if self._quasi_neutral:
            logger.info(
                "\nlam = 0 (quasi-neutral limit): no initial Poisson solve, the Gauss-law error is not measured."
            )
            if self.measure_gauss_law:
                logger.warning("measure_gauss_law: Gauss' law is not defined for lam = 0; gauss_error is NaN.")
            return

        self.initial_poisson.rho_coeffs = coupling._c2 / coupling._lam2
        super().post_allocate()

    def calculate_gauss_error(self):
        r"""Maximum norm of the weak Gauss-law residual
        :math:`\mathbb G^\top \mathbb M^1 \mathbf e + \frac{c_2}{\lambda^2} \boldsymbol \rho`, see
        :meth:`~struphy.models.vlasov_maxwell_one_species.VlasovMaxwellOneSpecies.calculate_gauss_error`
        (NaN for :math:`\lambda = 0`)."""
        if self._quasi_neutral:
            return float("nan")
        return super().calculate_gauss_error()

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        Vlasov equation:

        .. math::

            \frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla f + \frac{1}{\varepsilon} \left( \mathbf{E} + \mathbf{v} \times \left( \mathbf{B} + \mathbf{B}_0 \right) \right) \cdot \frac{\partial f}{\partial \mathbf{v}} = 0

        Ampère's law:

        .. math::

            -\lambda^2 \frac{\partial \mathbf{E}}{\partial t} + \nabla \times \mathbf{B} = \frac{\kappa^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f \, \text{d}^3 \mathbf{v}

        Faraday's law:

        .. math::

            \frac{\partial \mathbf{B}}{\partial t} + \nabla \times \mathbf{E} = 0

        with :math:`\varepsilon = 1` in the cyclotron normalization (set :math:`\varepsilon = -1` for electrons with the
        default charge number; then :math:`\kappa^2/\varepsilon` carries the sign of the charge). These forms assume
        :math:`\varepsilon = \pm 1`; for an overridden :math:`\varepsilon` the coefficients of the displacement and
        the plasma current are :math:`\varepsilon^2 \lambda^2` and :math:`\kappa^2 \varepsilon` (see
        :ref:`equation_params`). For
        :math:`\lambda = 0` (quasi-neutral limit) Ampère's law becomes :math:`\nabla \times \mathbf B =
        \frac{\kappa^2}{\varepsilon} \int \mathbf v f \, \textrm d^3 \mathbf v`, which implies :math:`\nabla \cdot \mathbf J = 0`; the electric field is then determined by this
        constraint, not by its own evolution. The coefficients of the coupling terms can be changed by the options
        ``lam``, ``c1``, ``c2``, ``c3`` of the propagator
        :class:`~struphy.propagators.vlasov_maxwell_coupling.VlasovMaxwellCoupling`.

        At initial time the weak Poisson equation (Gauss' law :math:`\lambda^2 \nabla \cdot \mathbf E = \kappa^2 \rho`)
        (with the charge density :math:`\rho = \frac{1}{\varepsilon} \int (f - f_0) \, \textrm d^3 \mathbf v`) is
        solved once for :math:`\lambda > 0`,

        .. math::

            \lambda^2 \int_{\Omega} \nabla \psi^{\top} \cdot \nabla \phi \, \textrm{d} \mathbf{x} = \frac{\kappa^2}{\varepsilon} \int_{\Omega} \int_{\mathbb{R}^3} \psi \, (f - f_0) \, \text{d}^3 \mathbf{v} \, \textrm{d} \mathbf{x} \qquad \forall \ \psi \in H^1
            \\[2mm]
            \mathbf{E}(t=0) = -\nabla \phi(t=0)

        (for :math:`\lambda = 0` the initial electric field does not enter the time stepping). Moreover, it is assumed
        that :math:`\nabla \times \mathbf{B}_0 = \frac{\kappa^2}{\varepsilon} \int_{\mathbb{R}^3} \mathbf{v} f_0 \, \text{d}^3 \mathbf{v}`,
        where :math:`\mathbf{B}_0` is the static equilibrium magnetic field.
        """

    @classmethod
    def doc_normalization(cls):
        r"""The model uses the cyclotron velocity of the species as reference velocity:

        .. math::

            \hat v = \hat\Omega_c \hat x,\qquad \hat t = \frac{1}{\hat\Omega_c},\qquad \hat E = \hat B \hat v,\qquad \hat\phi = \hat E \hat x.

        The species parameters (see :ref:`equation_params`) are

        .. math::

            \lambda = \frac{\hat\Omega_c \hat x}{c}\,,\qquad \kappa = \frac{\hat\Omega_p \hat x}{c} = \alpha \lambda\,,
            \qquad \varepsilon = \frac{1}{\hat\Omega_c \hat t} = 1\,,

        where :math:`c/\hat\Omega_c` and :math:`c/\hat\Omega_p` are the inertial lengths (:math:`\lambda` carries the
        sign of the charge, like :math:`\hat\Omega_c`; only :math:`\lambda^2` enters). The Vlasov equation and
        Faraday's law then have no coefficients, and Ampère's law
        :math:`\frac{1}{c^2} \partial_t \mathbf E = \nabla \times \mathbf B - \mu_0 \mathbf J` becomes
        :math:`\lambda^2 \partial_t \mathbf E = \nabla \times \mathbf B - \kappa^2 \mathbf J`. The quasi-neutral limit is
        :math:`\lambda \to 0` at fixed :math:`\kappa`. With :math:`\lambda = \kappa = 1` (and :math:`\varepsilon = -1` for
        electrons) the equations are those of :class:`~struphy.models.vlasov_maxwell_one_species.VlasovMaxwellOneSpecies`
        with :math:`\alpha = 1`, i.e. lengths in units of :math:`c/\hat\Omega_p` and :math:`\hat\Omega_c = \hat\Omega_p`."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation** (the parts of the energy conserved by the coupling
        step, with the coefficients :math:`\lambda^2, c_1, c_2, c_3` of
        :class:`~struphy.propagators.vlasov_maxwell_coupling.VlasovMaxwellCoupling`):

        - Electric field energy :math:`\frac{\lambda^2}{2} \int |\mathbf E|^2 \textrm d \mathbf x`: ``en_E``
        - Magnetic field energy :math:`\frac{c_1}{2} \int |\mathbf B|^2 \textrm d \mathbf x`: ``en_B``
        - Particle kinetic energy :math:`\frac{c_2}{2 c_3} \sum_p w_p |\mathbf v_p|^2`: ``en_f``
          (:math:`\frac{\kappa^2}{2} \sum_p w_p |\mathbf v_p|^2` for the default coefficients)
        - Total energy: ``en_tot`` (conserved by the coupling step up to the solver tolerance)
        - Optional Gauss-law diagnostic: ``gauss_error`` (NaN for :math:`\lambda = 0`)"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.vlasov_maxwell_coupling.VlasovMaxwellCoupling`
        2. :class:`~struphy.propagators.push_eta.PushEta`
        3. :class:`~struphy.propagators.push_vxb.PushVxB`
        """
        doc = rf"""**1. VlasovMaxwellCoupling:**

{VlasovMaxwellCoupling.__doc__}

**2. PushEta:**

{PushEta.__doc__}

**3. PushVxB:**

{PushVxB.__doc__}
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""VlasovMaxwellQuasiNeutral solves the Vlasov-Maxwell system in the cyclotron normalization, with one
        implicit step for Maxwell's equations and the electric force, which conserves the energy exactly and allows time
        steps larger than the inverse plasma frequency and the light-wave CFL limit. The coefficient :math:`\lambda^2`
        of the displacement current can be set to zero (``lam=0``), which gives the quasi-neutral (Darwin-type) limit,
        in which the electric field is the Lagrange multiplier of a divergence-free current."""

    @classmethod
    def doc_examples(cls):
        r"""Create a quasi-neutral Vlasov-Maxwell model (electrons, lengths in units of the electron skin depth):

        .. code-block:: python

            from struphy.models import VlasovMaxwellQuasiNeutral

            model = VlasovMaxwellQuasiNeutral(epsilon=-1.0, lam=0.0, kappa=1.0)
            model.em_fields.b_field
            model.kinetic_ions.var
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - electromagnetic kinetic simulations with time steps above the plasma period
        - the quasi-neutral limit of the Vlasov-Maxwell system (``lam=0.0``)
        - comparison with :class:`~struphy.models.vlasov_maxwell_one_species.VlasovMaxwellOneSpecies`"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - the quasi-neutral limit with control-variate (delta-f) weights or with vacuum regions
        - multi-species plasma dynamics without extension
        - collisional kinetic closures"""
