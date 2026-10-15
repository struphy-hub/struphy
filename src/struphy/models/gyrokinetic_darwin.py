import logging

from struphy import BaseUnits
from struphy.io.options import LiteralOptions
from struphy.models.gyrokinetic_maxwell import GyrokineticMaxwell, _LazyBackground
from struphy.models.scalars import FunctionScalarFEEC
from struphy.models.variables import FEECVariable, PICVariable
from struphy.propagators.base import Propagator
from struphy.propagators.gyrokinetic_darwin import DarwinFieldCoupling
from struphy.propagators.gyrokinetic_maxwell import (
    BackgroundWeights,
    GyroAverage,
    GyroParallelPush,
    PushGuidingCenterParallelFullB,
)

logger = logging.getLogger("struphy")


class GyrokineticDarwin(GyrokineticMaxwell):
    """Quasi-neutral gyrokinetic Darwin model of Remmerswaal, Hatzky and Sonnendrücker (2025), Sec. 6: the
    gauge-invariant gyrokinetic Maxwell model in the limit of quasi-neutrality and without the transversal
    displacement current, which removes the light wave, the Langmuir wave and the compressional Alfvén wave.

    Parameters
    ----------
    base_units: BaseUnits
        Base units for normalization (default: BaseUnits())
    species: tuple[tuple[str, int, float]]
        The gyrocentre species as ``(name, charge_number, mass_number)``; the species are the attributes
        ``kinetic_<name>`` of the model (default: ions and electrons)
    with_drifts: bool
        Whether to include the perpendicular (E x B and grad-B) drifts and their currents, default: True
    kinetic_backgrounds: bool
        Whether the background moments of the field equations are taken from the kinetic backgrounds of the
        species (default), or the polarization from the fluid equilibrium density
    background_current: bool
        Whether the equilibrium current :math:`\\nabla \\times \\mathbf B_0` enters Ampère's law (default: False)
    n_gyro: int
        Number of points of the gyro-average; 0 is the zero-Larmor-radius limit (default)
    n_disc: int
        Number of radial Gauss-Legendre points of the disc average (default: 2)
    solve_initial_poisson: bool
        Whether to solve the quasi-neutrality equation for the initial electric field (default: True); the part
        of the potential that is constant in the perpendicular directions (the kernel of the perpendicular
        Laplacian, which the polarization cannot balance) is removed
    """

    @classmethod
    def model_type(cls) -> LiteralOptions.ModelTypes:
        return "Kinetic"

    ## propagators

    class Propagators:
        def __init__(
            self,
            species: dict[str, PICVariable],
            b_field: FEECVariable = None,
            background: BackgroundWeights = None,
            polarization=0.0,
            background_current: bool = False,
            gyro: GyroAverage = None,
            with_drifts: bool = True,
        ):
            variables = tuple(species.values())
            self.fields = DarwinFieldCoupling(
                species=variables,
                background=background,
                polarization=polarization,
                background_current=background_current,
                gyro=gyro,
                with_drifts=with_drifts,
            )
            for name in species:
                if gyro is not None and gyro.flr:
                    setattr(self, f"push_para_{name}", GyroParallelPush(b_tilde=b_field, gyro=gyro))
                else:
                    setattr(self, f"push_para_{name}", PushGuidingCenterParallelFullB(b_tilde=b_field))

    def __init__(
        self,
        base_units: BaseUnits = BaseUnits(),
        species: tuple = GyrokineticMaxwell.DEFAULT_SPECIES,
        with_drifts: bool = True,
        kinetic_backgrounds: bool = True,
        background_current: bool = False,
        n_gyro: int = 0,
        n_disc: int = 2,
        solve_initial_poisson: bool = True,
    ):
        super().__init__(
            base_units=base_units,
            species=species,
            with_drifts=with_drifts,
            kinetic_backgrounds=kinetic_backgrounds,
            background_current=background_current,
            n_gyro=n_gyro,
            n_disc=n_disc,
            solve_initial_poisson=solve_initial_poisson,
            measure_gauss_law=False,
        )
        self.params.pop("measure_gauss_law", None)

    def _make_propagators(self, variables: dict[str, PICVariable]):
        self.propagators = self.Propagators(
            species=variables,
            b_field=self.em_fields.b_field,
            background=_LazyBackground(self),
            polarization=self._polarization,
            background_current=self.background_current,
            gyro=self.gyro,
            with_drifts=self.with_drifts,
        )

    def _assign_variables(self, variables: dict[str, PICVariable]):
        self.propagators.fields.variables.e = self.em_fields.e_field
        self.propagators.fields.variables.b = self.em_fields.b_field
        for name, var in variables.items():
            getattr(self.propagators, f"push_para_{name}").variables.ions = var

    def _field_energies(self) -> dict:
        return {
            "en_E": FunctionScalarFEEC(self._compute_en_E),
            "en_B": FunctionScalarFEEC(self._compute_en_B),
        }

    def _setup_gauss_law(self):
        """The quasi-neutrality equation (with the polarization matrix) for the initial electric field."""
        PL = self.propagators.fields.longitudinal_operator
        self._PL = PL
        self._e_tmp = self.em_fields.e_field.spline.vector.space.zeros()
        if not self.solve_initial_poisson:
            return

        logger.info("\nINITIAL QUASI-NEUTRALITY SOLVE (with the polarization matrix):")
        grad = Propagator.derham.grad
        self._charge_accums = {}
        self._charge_coeffs = {}
        super()._setup_gauss_law_accumulators()
        rhs = self._charge_density(self._gauss_rhs)
        phi = PL.solve_laplacian(rhs)
        logger.info(f"... {PL.inverse_laplacian._info}")
        # the kernel of the perpendicular Laplacian (the charge density averaged over the perpendicular
        # directions, k_perp = 0) cannot be balanced by the perpendicular polarization: that part of the
        # potential is only set by the regularization and is removed (it is marker noise in a quasi-neutral plasma)
        removed = PL.remove_average(phi)
        logger.info(f"... removed the perpendicular average of the potential (norm {removed:.3e})")
        e = self.em_fields.e_field.spline.vector
        grad.dot(-phi, out=e)
        e.update_ghost_regions()

    def _compute_en_E(self):
        r"""Longitudinal polarization energy :math:`\frac12 \mathbf e^\top \mathbb P_L \mathbf e`
        (:math:`\frac12 \mathbf D^{\textnormal{Dar}} \cdot \mathbf E` of Eq. (6.18))."""
        e = self.em_fields.e_field.spline.vector
        self._PL.dot(e, out=self._e_tmp)
        return 0.5 * self._e_tmp.inner(e)

    def calculate_gauss_error(self):
        raise NotImplementedError("The Darwin model enforces the quasi-neutrality equation implicitly.")

    @classmethod
    def doc_pde(cls):
        r"""**PDEs solved by model:**

        The gyrocentre equations of motion of :class:`~struphy.models.gyrokinetic_maxwell.GyrokineticMaxwell`,
        coupled to the quasi-neutral Darwin field equations (Sec. 6 of the paper) in the gauge-invariant form

        .. math::

            \frac{\partial \mathbf D^{\textnormal{Dar}}}{\partial t} &= \nabla \times \left(\mathbf B - \frac{p_{0,\parallel} - p_{0,\perp}}{B_0^2} \mathbf B_\perp
            + \sum_s \alpha_s^2 \int f_s\, \mu\, \mathbf b_0 \,\textrm d v_\parallel \textrm d \mu\right)
            - \sum_s \frac{\alpha_s^2}{\varepsilon_s} \int f_s\, \dot{\mathbf R} \,\textrm d v_\parallel \textrm d \mu\,,
            \\
            \frac{\partial \mathbf B}{\partial t} &= - \nabla \times \mathbf E\,,
            \qquad \mathbf D^{\textnormal{Dar}} = \Pi_{L,\perp}\left(\sum_s \alpha_s^2 \frac{n_{0,s}}{B_0^2} \mathbf E_\perp\right)\,,

        where :math:`\Pi_{L,\perp}` is the gyrokinetic longitudinal projection (Eq. (6.6)): compared with the
        gyrokinetic Maxwell model, the vacuum displacement current is dropped (quasi-neutrality,
        :math:`\epsilon_0 \to 0`) and only the longitudinal part of the polarization current is kept (Darwin
        approximation of the second-order Hamiltonian, Eq. (6.5)); for the modes constant in the perpendicular
        directions (:math:`k_\perp = 0`), where the projection is degenerate, the full polarization is retained
        (:class:`~struphy.propagators.gyrokinetic_darwin.DarwinLongitudinalOperator`). The divergence of Ampère's law is the
        quasi-neutrality equation (6.15), its transversal part the constraint of the differential-algebraic
        system (Eqs. (6.16)); the light, Langmuir and compressional Alfvén waves are removed, the shear Alfvén
        wave is retained with :math:`\omega = k_\parallel / \sqrt{\sum_s \alpha_s^2 n_{0,s}/B_0^2}` (light units).

        Reference: R. Remmerswaal, R. Hatzky, E. Sonnendrücker, *Gauge-invariant variational formulations of
        electromagnetic gyrokinetic theory*, J. Plasma Phys. 91, E127 (2025), `doi:10.1017/S0022377825100688 <https://doi.org/10.1017/S0022377825100688>`_.
        """

    @classmethod
    def doc_normalization(cls):
        r"""As :class:`~struphy.models.gyrokinetic_maxwell.GyrokineticMaxwell` (light units); the vacuum
        permittivity is dropped from the field equations (quasi-neutrality)."""

    @classmethod
    def doc_scalar_quantities(cls):
        r"""**The following scalars are tracked during simulation:**

        - Longitudinal polarization energy :math:`\frac12 \mathbf e^\top \mathbb P_L \mathbf e`: ``en_E``
        - Magnetic field energy (with the magnetizing matrix): ``en_B``
        - Species energies :math:`\alpha_s^2 \sum_p w_p (v_\parallel^2/2 + \mu B_\parallel)`: ``en_<name>``
        - Total energy: ``en_tot``
        - Lost markers: ``n_lost_<name>``"""

    @classmethod
    def doc_discretization(cls):
        """Time integration is performed by the following propagators (in sequence):

        1. :class:`~struphy.propagators.gyrokinetic_darwin.DarwinFieldCoupling` (the fields, the parallel
           velocities and the drifts of all species)
        2. :class:`~struphy.propagators.gyrokinetic_maxwell.PushGuidingCenterParallelFullB` (one per species, ZLR) or
           :class:`~struphy.propagators.gyrokinetic_maxwell.GyroParallelPush` (finite Larmor radius)
        """
        doc = rf"""**1. DarwinFieldCoupling:**

{DarwinFieldCoupling.__doc__}

**2. PushGuidingCenterParallelFullB / GyroParallelPush (one per species):**

{GyroParallelPush.__doc__}

The field equations are a differential-algebraic system: all couplings are solved in one implicit mid-point
step, which conserves the Darwin energy exactly for the parallel and E x B couplings. The system is
preconditioned by the weighted mass matrix of the polarization plus the continuum limit of the skin term
(:func:`~struphy.propagators.gyrokinetic_maxwell.skin_preconditioner_matrix`); the solver tolerances are
relative to the right-hand side.
"""
        return doc

    @classmethod
    def doc_long_description(cls):
        r"""GyrokineticDarwin is the quasi-neutral gyrokinetic Darwin approximation of the gauge-invariant
        gyrokinetic Maxwell model (Remmerswaal, Hatzky, Sonnendrücker, J. Plasma Phys. 91, E127, 2025, Sec. 6).
        The fast light, Langmuir and compressional Alfvén waves are removed, the model is still gauge invariant
        and compatible, and the shear Alfvén wave is retained. Since the model is gauge invariant it is solved in
        the fields :math:`\mathbf E`, :math:`\mathbf B` (no potentials and no gauge multiplier): the longitudinal
        polarization operator replaces the dielectric matrix of the Maxwell model and the field equations become
        a differential-algebraic system, solved implicitly together with the particle couplings. Same species,
        background and finite-Larmor-radius options as the Maxwell model."""

    @classmethod
    def doc_examples(cls):
        r"""Create and initialize the model:

        .. code-block:: python

            from struphy.models import GyrokineticDarwin

            model = GyrokineticDarwin(species=(("ions", 1, 1.0), ("electrons", -1, 1 / 100)))
            model.em_fields.e_field
            model.em_fields.b_field
            model.kinetic_ions.var

        ``b_field`` is the perturbation of the equilibrium field; the initial electric field follows from the
        quasi-neutrality equation (``solve_initial_poisson``).
        """

    @classmethod
    def doc_use_cases(cls):
        r"""This model is appropriate for:

        - low-frequency electromagnetic gyrokinetic simulations without the fast waves (large time steps)
        - comparison of the Darwin approximation with the gyrokinetic Maxwell model
        - backgrounds with parallel flow and pressure anisotropy, finite Larmor radius effects"""

    @classmethod
    def doc_cannot_be_used_for(cls):
        r"""This model is not suitable for:

        - the compressional Alfvén, light or Langmuir waves (removed by the approximation)
        - delta-f (control variate) simulations and polar-spline domains
        - backgrounds with parallel flow (the flow terms of Eq. (2.3) are not yet included in the Darwin solve)"""
