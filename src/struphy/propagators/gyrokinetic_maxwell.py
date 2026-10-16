"""Propagators of the gauge-invariant gyrokinetic Maxwell model of Remmerswaal, Hatzky and Sonnendrücker
(J. Plasma Phys. 91, E127, 2025), see :class:`~struphy.models.gyrokinetic_maxwell.GyrokineticMaxwell`.

The field unknowns are :math:`\\mathbf E \\in H(\\textnormal{curl})` and :math:`\\mathbf B \\in H(\\textnormal{div})`
(the model is gauge invariant, hence formulated in the fields rather than in the potentials).
All propagators share the dielectric matrix :func:`dielectric_matrix` (vacuum permittivity plus polarization),
the magnetizing matrix :func:`magnetizing_matrix` (pressure anisotropy of the background) and, for a background
with parallel flow, the flow operator :func:`flow_operator`. The background moments of each species
(:class:`SpeciesBackground`) are taken from its kinetic background.

Finite Larmor radius effects (:class:`GyroAverage`) replace the point evaluations of the fields at the gyrocentres by
gyro-averages (ring) and disc averages, with the couplings to the electric field applied matrix-free
(:class:`GyroParticleOperator`), since the gyro-averaged particle matrices are not stencil matrices.

The propagators work with tensor-product FE coefficients (no polar splines).
"""

import logging
from dataclasses import dataclass
from typing import Callable

import cunumpy as xp
import numpy as np
from feectools.linalg.basic import LinearOperator
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI
from scope_profiler import ProfileManager

from struphy.feec import preconditioner
from struphy.feec.mass import WeightedMassOperator, WeightedMassOperators
from struphy.feec.utilities import LocalRotationMatrix
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.kinetic_background.base import KineticBackground, SumKineticBackground
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.ode.utils import ButcherTableau
from struphy.pic.accumulation.kernels.gc_mag_density_0form import gc_mag_density_0form
from struphy.pic.accumulation.kernels.gk_drift_coupling import gk_drift_coupling
from struphy.pic.accumulation.kernels.gk_drift_gyro_matvec import gk_drift_gyro_matvec
from struphy.pic.accumulation.kernels.gk_drift_gyro_vector import gk_drift_gyro_vector
from struphy.pic.accumulation.kernels.gk_epar_coupling import gk_epar_coupling
from struphy.pic.accumulation.kernels.gk_epar_gyro_matvec import gk_epar_gyro_matvec
from struphy.pic.accumulation.kernels.gk_epar_gyro_vector import gk_epar_gyro_vector
from struphy.pic.accumulation.kernels.gk_mag_density_gyro import gk_mag_density_gyro
from struphy.pic.accumulation.particles_to_grid import Accumulator, AccumulatorVector
from struphy.pic.pushing.kernel_setup import KernelSetup
from struphy.pic.pushing.kernels.gk_bstar_gyro import gk_bstar_gyro
from struphy.pic.pushing.kernels.gk_bstar_parallel_gyro import gk_bstar_parallel_gyro
from struphy.pic.pushing.kernels.gk_hamiltonian_gyro import gk_hamiltonian_gyro
from struphy.pic.pushing.kernels.gk_push_drift_gyro import gk_push_drift_gyro
from struphy.pic.pushing.kernels.gk_push_parallel_dg1_gyro import gk_push_parallel_dg1_gyro
from struphy.pic.pushing.kernels.gk_push_vpar_efield import gk_push_vpar_efield
from struphy.pic.pushing.kernels.gk_push_vpar_gyro import gk_push_vpar_gyro
from struphy.pic.pushing.kernels.push_gc_bxEstar_explicit_multistage import push_gc_bxEstar_explicit_multistage
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.propagators.push_guiding_center_parallel import PushGuidingCenterParallel
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


########################
# Species backgrounds
########################


class SpeciesBackground:
    r"""The background moments :math:`n_{0,s}`, :math:`u_{0,\parallel,s}`, :math:`v_{\textnormal{th},\parallel,s}`,
    :math:`v_{\textnormal{th},\perp,s}` of one gyrocentre species on the logical cube, from its kinetic background
    (a :class:`~struphy.kinetic_background.maxwellians.GyroMaxwellian2D` or a sum of them), or from the fluid
    equilibrium density (``kinetic=False``: :math:`n_0`, no flow, isotropic).

    The normalized species pressures are :math:`p_{\parallel,s} = \alpha_s^2 n_s v_{\textnormal{th},\parallel,s}^2`
    and :math:`p_{\perp,s} = \alpha_s^2 n_s v_{\textnormal{th},\perp,s}^2` (light units, Eq. (5.56) of the paper).

    Parameters
    ----------
    var : PICVariable
        The species' variable (``"Particles5D"``), with its kinetic background(s) added.
    kinetic : bool
        Whether to take the moments from the kinetic background (default) or the density from the fluid equilibrium.
    """

    def __init__(self, var: PICVariable, kinetic: bool = True):
        assert isinstance(var, PICVariable)
        self.var = var
        self.alpha = var.species.equation_params.alpha
        self.epsilon = var.species.equation_params.epsilon
        self.kinetic = kinetic

        backgrounds = var.backgrounds
        if backgrounds is None:
            backgrounds = []
        elif not isinstance(backgrounds, list):
            backgrounds = [backgrounds]
        self.backgrounds = []
        for bg in backgrounds:
            self.backgrounds += self._flatten(bg)
        if kinetic and len(self.backgrounds) == 0:
            raise ValueError(
                f"The species {var.species.__class__.__name__} has no kinetic background; "
                "add one with var.add_background(...) or use the fluid equilibrium (kinetic=False)."
            )

    @staticmethod
    def _flatten(bg) -> list:
        """The Maxwellians of a (sum of) kinetic background(s), which provide the moments ``n``, ``u``, ``vth``."""
        if isinstance(bg, SumKineticBackground):
            return SpeciesBackground._flatten(bg._f1) + SpeciesBackground._flatten(bg._f2)
        if isinstance(bg, KineticBackground) and all(hasattr(bg, m) for m in ("n", "u", "vth")):
            return [bg]
        return []

    @property
    def name(self) -> str:
        return self.var.species.__class__.__name__

    def _moments(self, *etas):
        """Density, parallel flow and squared thermal speeds (sum over the backgrounds, flow weighted by density)."""
        n = 0.0
        nu = 0.0
        n_vpar2 = 0.0
        n_vperp2 = 0.0
        for bg in self.backgrounds:
            n_i = bg.n(*etas)
            u_i = bg.u(*etas)[0]
            vth_i = bg.vth(*etas)
            n = n + n_i
            nu = nu + n_i * u_i
            # the parallel pressure of the sum includes the spread of the mean velocities
            n_vpar2 = n_vpar2 + n_i * (vth_i[0] ** 2 + u_i**2)
            n_vperp2 = n_vperp2 + n_i * vth_i[1] ** 2
        n_safe = xp.where(n > 0.0, n, 1.0)
        u = nu / n_safe
        vth_par2 = n_vpar2 / n_safe - u**2
        vth_perp2 = n_vperp2 / n_safe
        return n, u, vth_par2, vth_perp2

    def n(self, *etas, equil=None):
        """Background density :math:`n_{0,s}` (units of the density unit)."""
        if not self.kinetic:
            return equil.n0(*etas)
        return self._moments(*etas)[0]

    @staticmethod
    def _zeros(*etas):
        return 0.0 * xp.asarray(etas[0] if len(etas) == 3 else etas[0][:, 0])

    def u_par(self, *etas):
        """Background parallel flow :math:`u_{0,\\parallel,s}` (units of c)."""
        if not self.kinetic:
            return self._zeros(*etas)
        return self._moments(*etas)[1]

    def vth_par2(self, *etas):
        """Squared parallel thermal speed."""
        if not self.kinetic:
            return self._zeros(*etas)
        return self._moments(*etas)[2]

    def vth_perp2(self, *etas):
        """Squared perpendicular thermal speed."""
        if not self.kinetic:
            return self._zeros(*etas)
        return self._moments(*etas)[3]


class BackgroundWeights:
    r"""The coefficients of the field operators from the species backgrounds (Eq. (2.2a) of the paper, light units):

    .. math::

        \textnormal{polarization} = \sum_s \alpha_s^2 \frac{n_{0,s}}{B_0^2}\,, \qquad
        \textnormal{flow} = \sum_s \alpha_s^2 \frac{n_{0,s} u_{0,\parallel,s}}{B_0^2}\,, \qquad
        \textnormal{anisotropy} = \frac{p_{0,\parallel} - p_{0,\perp}}{B_0^2} = \sum_s \alpha_s^2 \frac{n_{0,s} (v_{\textnormal{th},\parallel,s}^2 - v_{\textnormal{th},\perp,s}^2)}{B_0^2}\,,

    as callables of the logical coordinates (for :meth:`~struphy.feec.mass.WeightedMassOperators.create_weighted_mass`).

    Parameters
    ----------
    backgrounds : tuple[SpeciesBackground]
        The species backgrounds.
    equil : FluidEquilibriumWithB
        The fluid equilibrium (for :math:`B_0` and, with ``kinetic=False`` backgrounds, :math:`n_0`).
    """

    def __init__(self, backgrounds: tuple[SpeciesBackground, ...], equil):
        self.backgrounds = tuple(backgrounds)
        self.equil = equil

    @staticmethod
    def _sample_grid():
        e = np.linspace(0.05, 0.95, 4)
        return np.meshgrid(e, e, e, indexing="ij")

    @property
    def has_flow(self) -> bool:
        """Whether any kinetic species background has a parallel flow (``u_para != 0`` on a sample grid)."""
        e1, e2, e3 = self._sample_grid()
        for bg in self.backgrounds:
            if bg.kinetic and np.any(np.asarray(bg.u_par(e1, e2, e3)) != 0.0):
                return True
        return False

    @property
    def has_anisotropy(self) -> bool:
        """Whether any kinetic species background is anisotropic (``vth_para != vth_perp`` on a sample grid)."""
        e1, e2, e3 = self._sample_grid()
        for bg in self.backgrounds:
            if bg.kinetic and np.any(np.asarray(bg.vth_par2(e1, e2, e3)) != np.asarray(bg.vth_perp2(e1, e2, e3))):
                return True
        return False

    def polarization(self, e1, e2, e3):
        out = 0.0
        for bg in self.backgrounds:
            out = out + bg.alpha**2 * bg.n(e1, e2, e3, equil=self.equil)
        return out / self.equil.absB0(e1, e2, e3) ** 2

    def flow(self, e1, e2, e3):
        out = 0.0
        for bg in self.backgrounds:
            out = out + bg.alpha**2 * bg.n(e1, e2, e3, equil=self.equil) * bg.u_par(e1, e2, e3)
        return out / self.equil.absB0(e1, e2, e3) ** 2

    def anisotropy(self, e1, e2, e3):
        out = 0.0
        for bg in self.backgrounds:
            n = bg.n(e1, e2, e3, equil=self.equil)
            out = out + bg.alpha**2 * n * (bg.vth_par2(e1, e2, e3) - bg.vth_perp2(e1, e2, e3))
        return out / self.equil.absB0(e1, e2, e3) ** 2

    def skin(self, e1, e2, e3):
        r"""The continuum limit :math:`\sum_s \alpha_s^2 n_{0,s}/\varepsilon_s^2` of the skin term of the implicit parallel couplings."""
        out = 0.0
        for bg in self.backgrounds:
            out = out + bg.alpha**2 / bg.epsilon**2 * bg.n(e1, e2, e3, equil=self.equil)
        return out


def parallel_flow_from_equilibrium(equil, alpha: float, epsilon: float, density: Callable, fraction: float = 1.0):
    r"""The parallel flow :math:`u_{0,\parallel,s}` a species needs to carry a fraction of the equilibrium current,
    Eq. (5.62) of the paper with isotropic background pressure:

    .. math::

        \frac{\alpha_s^2}{\varepsilon_s} n_{0,s}\, u_{0,\parallel,s} = f_s\, \mathbf b_0 \cdot (\nabla \times \mathbf B_0)\,,

    such that, with :math:`\sum_s f_s = 1`, the gyrocentre currents of the backgrounds balance :math:`\nabla \times \mathbf B_0`
    in Ampère's law (``background_current=True`` of the model).

    Parameters
    ----------
    equil : FluidEquilibriumWithB
        The equilibrium (provides :math:`\nabla \times \mathbf B_0` and :math:`\mathbf b_0`).
    alpha, epsilon : float
        The species parameters.
    density : Callable
        The background density :math:`n_{0,s}` of the species on the logical cube.
    fraction : float
        The fraction :math:`f_s` of the parallel current carried by the species.

    Returns
    -------
    Callable
        ``u_par(e1, e2, e3)``, to be used as ``u_para=(u_par, None)`` of a ``GyroMaxwellian2D``.
    """

    def u_par(*etas):
        """Called with ``(eta1, eta2, eta3)`` or with one ``(n, 3)`` array (as the kinetic backgrounds do)."""
        j, _ = equil.j_cart(*etas)
        b, _ = equil.unit_b_cart(*etas)
        j_par = j[0] * b[0] + j[1] * b[1] + j[2] * b[2]
        return fraction * j_par * epsilon / (alpha**2 * density(*etas))

    return u_par


########################
# Field operators
########################


class _DielectricWeight:
    r"""Matrix-valued weight :math:`(1 + s)\, G^{-1} - s\, \hat{\mathbf b}_0^v (\hat{\mathbf b}_0^v)^\top`
    of the dielectric matrix on the logical cube, with :math:`s = \sum_s \alpha_s^2 n_{0,s}/B_0^2` and
    :math:`\hat{\mathbf b}_0^v` the contra-variant components of the equilibrium unit magnetic field
    (such that :math:`\hat{\mathbf e}^{1\top} \hat{\mathbf b}^v_0 = \mathbf E \cdot \mathbf b_0`).

    Callable in the sense of :meth:`~struphy.feec.mass.WeightedMassOperators.create_weighted_mass`
    (returns an array of shape ``(n1, n2, n3, 3, 3)``).
    """

    def __init__(self, domain, equil, polarization, vacuum: bool = True):
        self._domain = domain
        self._equil = equil
        self._polarization = polarization
        self._vacuum = 1.0 if vacuum else 0.0

    def __call__(self, e1, e2, e3):
        ginv = self._domain.metric_inv(e1, e2, e3, change_out_order=True)
        bv = xp.array(
            [
                self._equil.unit_bv_1(e1, e2, e3),
                self._equil.unit_bv_2(e1, e2, e3),
                self._equil.unit_bv_3(e1, e2, e3),
            ]
        )
        bb = xp.moveaxis(bv[:, None] * bv[None, :], (0, 1), (-2, -1))

        if callable(self._polarization):
            pol = self._polarization(e1, e2, e3)
        else:
            pol = self._polarization * self._equil.n0(e1, e2, e3) / self._equil.absB0(e1, e2, e3) ** 2
        pol = xp.asarray(pol)[..., None, None]
        return (self._vacuum + pol) * ginv - pol * bb


class _MagnetizingWeight:
    r"""Matrix-valued weight :math:`(1 - c)\, G + c\, \hat{\mathbf b}^1_0 (\hat{\mathbf b}^1_0)^\top` (2-forms) of the
    magnetizing matrix, with :math:`c = (p_{0,\parallel} - p_{0,\perp})/B_0^2`: the weak form of
    :math:`\mathbf B - c\, \mathbf B_\perp`."""

    def __init__(self, domain, equil, anisotropy):
        self._domain = domain
        self._equil = equil
        self._anisotropy = anisotropy

    def __call__(self, e1, e2, e3):
        g = self._domain.metric(e1, e2, e3, change_out_order=True)
        b1 = xp.array(
            [
                self._equil.unit_b1_1(e1, e2, e3),
                self._equil.unit_b1_2(e1, e2, e3),
                self._equil.unit_b1_3(e1, e2, e3),
            ]
        )
        bb = xp.moveaxis(b1[:, None] * b1[None, :], (0, 1), (-2, -1))
        c = xp.asarray(self._anisotropy(e1, e2, e3))[..., None, None]
        return (1.0 - c) * g + c * bb


def _cache(mass_ops: WeightedMassOperators, name: str, key, build):
    cache = mass_ops.__dict__.setdefault("_gyrokinetic_maxwell_operators", {})
    if (name, key) not in cache:
        cache[(name, key)] = build()
    return cache[(name, key)]


def dielectric_matrix(mass_ops: WeightedMassOperators, polarization) -> WeightedMassOperator:
    r"""Dielectric matrix of the gyrokinetic Maxwell model,

    .. math::

        \mathbb A = \int \vec{\Lambda}^1 \left[ \left(1 + \sum_s \alpha_s^2 \frac{n_{0,s}}{B_0^2}\right) G^{-1}
        - \sum_s \alpha_s^2 \frac{n_{0,s}}{B_0^2} \hat{\mathbf b}_0^v (\hat{\mathbf b}_0^v)^\top \right] \vec{\Lambda}^{1\top} \sqrt g\, \textnormal d \boldsymbol \eta\,,

    i.e. the weak form of :math:`\mathbf I + \sum_s \alpha_s^2 \frac{n_{0,s}}{B_0^2} \Pi_\perp` (vacuum permittivity
    plus the polarization of the second-order Hamiltonian), assembled as one
    :class:`~struphy.feec.mass.WeightedMassOperator`.

    Parameters
    ----------
    mass_ops : WeightedMassOperators
        The weighted mass operators of the simulation (provide domain and equilibrium).

    polarization : float | Callable
        Either the constant :math:`\sum_s \alpha_s^2` (the density is then the fluid equilibrium :math:`n_0`
        for every species), or the callable :math:`\sum_s \alpha_s^2 n_{0,s}/B_0^2` of the logical coordinates
        (:meth:`BackgroundWeights.polarization`).

    Returns
    -------
    The assembled operator; it is cached in ``mass_ops`` and shared between the propagators.
    """
    key = id(polarization) if callable(polarization) else float(polarization)

    def build():
        if not callable(polarization) and float(polarization) == 0.0:
            return mass_ops.M1
        weight = _DielectricWeight(mass_ops.domain, mass_ops.eq_mhd, polarization)
        return mass_ops.create_weighted_mass(
            "Hcurl", "Hcurl", weights=(weight, "sqrt_g"), name="M1_dielectric_gk", assemble=True
        )

    return _cache(mass_ops, "dielectric", key, build)


def polarization_matrix(mass_ops: WeightedMassOperators, polarization) -> WeightedMassOperator:
    r"""Polarization matrix :math:`\mathbb P = \mathbb A - \mathbb M^1`, the weak form of
    :math:`\sum_s \alpha_s^2 \frac{n_{0,s}}{B_0^2} \Pi_\perp` on 1-forms (the coefficient :math:`\mathcal C(1) \Pi_\perp`
    of Eq. (5.47) of the paper), assembled as one :class:`~struphy.feec.mass.WeightedMassOperator`;
    same ``polarization`` argument as :func:`dielectric_matrix`.
    """
    key = id(polarization) if callable(polarization) else float(polarization)

    def build():
        weight = _DielectricWeight(mass_ops.domain, mass_ops.eq_mhd, polarization, vacuum=False)
        return mass_ops.create_weighted_mass(
            "Hcurl", "Hcurl", weights=(weight, "sqrt_g"), name="M1_polarization_gk", assemble=True
        )

    return _cache(mass_ops, "polarization", key, build)


class _SkinWeight:
    r"""Matrix-valued weight :math:`v\, G^{-1} + s\, G^{-1} + \kappa\, \hat{\mathbf b}_0^v (\hat{\mathbf b}_0^v)^\top - s\, \hat{\mathbf b}_0^v (\hat{\mathbf b}_0^v)^\top`
    of the preconditioner mass matrix of the implicit parallel couplings: the dielectric weight (vacuum :math:`v`,
    polarization :math:`s`) plus the continuum limit :math:`\kappa\, \mathbf b_0 \mathbf b_0^\top` of the skin term
    :math:`\sum_s \frac{\alpha_s^2}{\varepsilon_s^2} \sum_p w_p \mathbf c_p \mathbf c_p^\top \to \sum_s \frac{\alpha_s^2}{\varepsilon_s^2} n_{0,s}\, \mathbf b_0 \mathbf b_0^\top`."""

    def __init__(self, domain, equil, polarization, skin, vacuum: float):
        self._dielectric = _DielectricWeight(domain, equil, polarization, vacuum=vacuum > 0.0)
        self._domain = domain
        self._equil = equil
        self._skin = skin
        self._vacuum = vacuum

    def __call__(self, e1, e2, e3):
        out = self._dielectric(e1, e2, e3)
        bv = xp.array(
            [
                self._equil.unit_bv_1(e1, e2, e3),
                self._equil.unit_bv_2(e1, e2, e3),
                self._equil.unit_bv_3(e1, e2, e3),
            ]
        )
        bb = xp.moveaxis(bv[:, None] * bv[None, :], (0, 1), (-2, -1))
        kappa = xp.asarray(self._skin(e1, e2, e3))[..., None, None]
        ginv = self._domain.metric_inv(e1, e2, e3, change_out_order=True)
        # without vacuum permittivity the weight is singular along b_0 where the skin term vanishes: add a floor
        floor = 0.0 if self._vacuum > 0.0 else 1e-8 * ginv
        return out + kappa * bb + floor


def skin_preconditioner_matrix(mass_ops: WeightedMassOperators, polarization, skin, scale: float, vacuum: bool):
    r"""A weighted mass matrix approximating the operator of the implicit parallel coupling,
    :math:`\mathbb A + c\, \mathbb S` (or :math:`\mathbb P_L + c\, \mathbb S` of the Darwin model), by the weight
    :math:`[v + s(\boldsymbol \eta)] G^{-1} - s(\boldsymbol \eta)\, \hat{\mathbf b}_0^v (\hat{\mathbf b}_0^v)^\top + c\, \kappa(\boldsymbol \eta)\, \hat{\mathbf b}_0^v (\hat{\mathbf b}_0^v)^\top`
    with :math:`\kappa = \sum_s \frac{\alpha_s^2}{\varepsilon_s^2} n_{0,s}` (the continuum limit of the skin term),
    for the :class:`~struphy.feec.preconditioner.MassMatrixPreconditioner`.

    Parameters
    ----------
    mass_ops : WeightedMassOperators
    polarization : float | Callable
        As in :func:`dielectric_matrix`.
    skin : Callable
        :math:`\kappa(\boldsymbol \eta)` on the logical cube.
    scale : float
        The constant :math:`c` (e.g. :math:`\Delta t^2/4`).
    vacuum : bool
        Whether the vacuum permittivity is included (Maxwell) or not (Darwin).
    """
    key = (id(polarization) if callable(polarization) else float(polarization), id(skin), float(scale), bool(vacuum))

    def build():
        def kappa(e1, e2, e3):
            return scale * xp.asarray(skin(e1, e2, e3))

        weight = _SkinWeight(mass_ops.domain, mass_ops.eq_mhd, polarization, kappa, 1.0 if vacuum else 0.0)
        return mass_ops.create_weighted_mass(
            "Hcurl", "Hcurl", weights=(weight, "sqrt_g"), name="M1_skin_pc_gk", assemble=True
        )

    return _cache(mass_ops, "skin_pc", key, build)


def magnetizing_matrix(mass_ops: WeightedMassOperators, anisotropy=None) -> WeightedMassOperator:
    r"""Magnetizing matrix of the gyrokinetic Maxwell model: the weak form of
    :math:`\mathbf H = \mathbf B - \frac{p_{0,\parallel} - p_{0,\perp}}{B_0^2} \mathbf B_\perp` on 2-forms,

    .. math::

        \mathbb M^2_H = \int \vec{\Lambda}^2 \left[(1 - c)\, G + c\, \hat{\mathbf b}^1_0 (\hat{\mathbf b}^1_0)^\top\right] \vec{\Lambda}^{2\top} \frac{1}{\sqrt g}\, \textnormal d \boldsymbol \eta\,,
        \qquad c = \frac{p_{0,\parallel} - p_{0,\perp}}{B_0^2}\,,

    which is :math:`\mathbb M^2` for an isotropic background (``anisotropy=None``).
    """
    if anisotropy is None:
        return mass_ops.M2

    def build():
        weight = _MagnetizingWeight(mass_ops.domain, mass_ops.eq_mhd, anisotropy)
        return mass_ops.create_weighted_mass(
            "Hdiv", "Hdiv", weights=(weight, "1/sqrt_g"), name="M2_magnetizing_gk", assemble=True
        )

    return _cache(mass_ops, "magnetizing", id(anisotropy), build)


def flow_operator(mass_ops: WeightedMassOperators, flow) -> WeightedMassOperator:
    r"""The weighted mass operator :math:`\mathbb R: V_h^1 \to V_h^2` of the background parallel flow,

    .. math::

        \int \mathbf F \cdot \left(s\, \mathbf b_0 \times \mathbf E\right) \textnormal d \mathbf x
        = \mathbf f^\top \mathbb R\, \mathbf e\,, \qquad s = \sum_s \alpha_s^2 \frac{n_{0,s} u_{0,\parallel,s}}{B_0^2}\,,

    for a 2-form test function :math:`\mathbf F`. The flow terms of Ampère's law (2.2a), the magnetisation
    :math:`\nabla \times (s\, \mathbf b_0 \times \mathbf E)` and the polarization current :math:`s\, \mathbf b_0 \times (\nabla \times \mathbf E)`,
    are the anti-symmetric operator :math:`\mathbb K = \mathbb C^\top \mathbb R - \mathbb R^\top \mathbb C`.
    """

    def build():
        rot_b = LocalRotationMatrix(mass_ops.eq_mhd.unit_b1_1, mass_ops.eq_mhd.unit_b1_2, mass_ops.eq_mhd.unit_b1_3)
        return mass_ops.create_weighted_mass(
            "Hcurl", "Hdiv", weights=(flow, "G", rot_b, "1/sqrt_g"), name="R_flow_gk", assemble=True
        )

    return _cache(mass_ops, "flow", id(flow), build)


########################
# Gyro-averaging
########################


@dataclass
class GyroAverage:
    """Discretization of the gyro-average (ring) and the disc average of the fields at the gyrocentres.

    Parameters
    ----------
    n_gyro : int
        Number of points on the gyro-ring; ``0`` is the zero-Larmor-radius limit (point evaluation).
    n_disc : int
        Number of Gauss-Legendre points in the radial direction of the disc average; ``0`` approximates the disc
        average by the gyro-average.
    """

    n_gyro: int = 0
    n_disc: int = 2

    def __post_init__(self):
        assert self.n_gyro >= 0 and self.n_disc >= 0
        self.ring_s = np.array([1.0])
        self.ring_w = np.array([1.0])
        if self.n_disc > 0:
            x, w = np.polynomial.legendre.leggauss(self.n_disc)
            s = (x + 1.0) / 2.0
            self.disc_s = s
            # the disc average is 2 int_0^1 s <Q^s> ds
            self.disc_w = w / 2.0 * 2.0 * s
        else:
            self.disc_s = self.ring_s.copy()
            self.disc_w = self.ring_w.copy()

    @property
    def flr(self) -> bool:
        """Whether finite-Larmor-radius averages are used."""
        return self.n_gyro > 0


class GyroParticleOperator(LinearOperator):
    r"""Matrix-free linear operator :math:`V^1_h \to V^1_h` of the gyro-averaged coupling of the markers of one
    species to a 1-form field,

    .. math::

        \mathbf e \mapsto c \sum_p w_p \langle \vec\Lambda^1 \rangle_p\, \mathbb K_p\, \langle \vec\Lambda^{1\top} \rangle_p \mathbf e\,,

    with a per-marker :math:`3 \times 3` matrix :math:`\mathbb K_p` defined by the kernel (e.g. :math:`\mathbf c_p \mathbf c_p^\top`
    for the parallel coupling, the E x B rotation for the drift coupling). Each product evaluates the field
    at the gyro-points of every marker and deposits there again, so the operator is applied inside an iterative
    solver (the gyro-averaged particle matrices are not stencil matrices).

    Parameters
    ----------
    particles : Particles
        The markers.
    derham : Derham
        The discrete de Rham complex (tensor-product spaces).
    args_domain : DomainArguments
        Mapping arguments.
    kernel : Kernel
        A kernel with the signature ``(args_markers, args_derham, args_domain, vec1, vec2, vec3, e1, e2, e3, *args)``.
    args : tuple
        The remaining kernel arguments.
    scale : float
        The constant :math:`c`.
    symmetric : bool
        Whether the operator is symmetric (else it is anti-symmetric), for :meth:`transpose`.
    """

    def __init__(self, particles, derham, args_domain, kernel, args: tuple, scale: float = 1.0, symmetric=True):
        self._particles = particles
        self._derham = derham
        self._args_domain = args_domain
        self._kernel = kernel
        self._args = tuple(args)
        self._scale = scale
        self._symmetric = symmetric
        self._space = derham.V1
        self._tmp = self._space.zeros()

    @property
    def domain(self):
        return self._space

    @property
    def codomain(self):
        return self._space

    @property
    def dtype(self):
        return float

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, new: float):
        self._scale = float(new)

    def kernels(self):
        return (self._kernel,)

    def transpose(self, conjugate=False):
        if self._symmetric:
            return self
        return -1.0 * self

    def tosparse(self):
        raise NotImplementedError("matrix-free operator")

    def toarray(self, **kwargs):
        return LinearOperator.toarray(self, **kwargs)

    def dot(self, v, out=None):
        assert v.space == self._space
        if out is None:
            out = self._space.zeros()
        else:
            assert out.space == self._space
            out *= 0.0

        # the kernel reads the ghost regions of the input and deposits into the ghost regions of the output
        v.update_ghost_regions()
        with ProfileManager.profile_region("kernel: " + self._kernel.name):
            self._kernel(
                self._particles.args_markers,
                self._derham.args_derham,
                self._args_domain,
                out[0]._data,
                out[1]._data,
                out[2]._data,
                v[0]._data,
                v[1]._data,
                v[2]._data,
                *self._args,
            )

        clone_config = self._particles.clone_config
        if clone_config is not None and clone_config.num_clones > 1:
            for block in out.blocks:
                clone_config.inter_comm.Allreduce(MPI.IN_PLACE, block._data, op=MPI.SUM)

        out.exchange_assembly_data()
        out.update_ghost_regions()
        out *= self._scale
        return out


####################################
# Perturbed magnetic field (B_tilde)
####################################


class _PerturbedFieldMixin:
    r"""Tensor-product coefficients of the magnetic perturbation for the kernels: the parallel perturbation
    :math:`\tilde B_\parallel = \Pi^0(\mathbf b_0 \cdot \tilde{\mathbf B})` (0-form), its gradient (1-form) and the
    perturbation itself (2-form), each also added to the equilibrium (``_B_dot_b``, ``_grad_b_full``, ``_b2_full``).

    The vectors are persistent (the kernels hold their data arrays) and updated in place by :meth:`_update_b_tilde`.
    If ``b_tilde`` is None the perturbation is zero.
    """

    def _allocate_b_tilde(self, b_tilde: FEECVariable | None):
        self._b_tilde_var = b_tilde

        self._gradB1 = self.projected_equil.gradB1
        self._absB0 = self.projected_equil.absB0
        self._b2_eq = self.projected_equil.b2
        self._unit_b1 = self.projected_equil.unit_b1
        self._curl_unit_b2 = self.projected_equil.curl_unit_b2
        self._curl_unit_b_dot_b0 = self.projected_equil.curl_unit_b_dot_b0

        # the kernels need tensor-product coefficients: transposed extraction operators
        # PolarVector --> Stencil-/BlockVector (identity maps in case of no polar splines)
        self._E0T = self.derham.extraction_ops["0"].transpose()
        self._E1T = self.derham.extraction_ops["1"].transpose()
        self._E2T = self.derham.extraction_ops["2"].transpose()

        self._B_tilde_par = self.derham.V0.zeros()
        self._grad_B_tilde_par = self.derham.V1.zeros()
        self._b2_tilde = self.derham.V2.zeros()
        self._B_dot_b = self.derham.V0.zeros()
        self._grad_b_full = self.derham.V1.zeros()
        self._b2_full = self.derham.V2.zeros()

        self._PB = getattr(self.basis_ops, "PB")
        self._B_tilde_par_pol = self._PB.codomain.zeros()
        self._grad_B_tilde_par_pol = self.derham.grad.codomain.zeros()

        self._update_b_tilde()

    def _update_b_tilde(self):
        """Parallel magnetic perturbation, its gradient and the perturbation in the tensor-product spaces,
        alone and added to the equilibrium."""
        if self._b_tilde_var is not None:
            b_tilde = self._b_tilde_var.spline.vector

            B_par_pol = self._PB.dot(b_tilde, out=self._B_tilde_par_pol)
            self._E0T.dot(B_par_pol, out=self._B_tilde_par)
            grad_pol = self.derham.grad.dot(B_par_pol, out=self._grad_B_tilde_par_pol)
            self._E1T.dot(grad_pol, out=self._grad_B_tilde_par)
            self._E2T.dot(b_tilde, out=self._b2_tilde)

        for tilde, eq, full in (
            (self._B_tilde_par, self._absB0, self._B_dot_b),
            (self._grad_B_tilde_par, self._gradB1, self._grad_b_full),
            (self._b2_tilde, self._b2_eq, self._b2_full),
        ):
            tilde.update_ghost_regions()
            tilde.copy(out=full)
            full += eq
            full.update_ghost_regions()

    def _gyro_args(self, gyro: GyroAverage, epsilon: float, disc: bool):
        """The common gyro-averaging arguments of the kernels: ``epsilon, n_gyro, s, w, unit_b1 (3), absB0``
        with the ring (``disc=False``) or disc nodes and weights."""
        s, w = (gyro.disc_s, gyro.disc_w) if disc else (gyro.ring_s, gyro.ring_w)
        return (
            epsilon,
            gyro.n_gyro,
            s,
            w,
            self._unit_b1[0]._data,
            self._unit_b1[1]._data,
            self._unit_b1[2]._data,
            self._absB0._data,
        )

    def _bstar_args(self):
        """The field arguments of the gyro-averaged B* kernels: ``b2_eq (3), b2_tilde (3), curl_unit_b2 (3),
        B_tilde_par, curl_unit_b_dot_b0``."""
        return (
            self._b2_eq[0]._data,
            self._b2_eq[1]._data,
            self._b2_eq[2]._data,
            self._b2_tilde[0]._data,
            self._b2_tilde[1]._data,
            self._b2_tilde[2]._data,
            self._curl_unit_b2[0]._data,
            self._curl_unit_b2[1]._data,
            self._curl_unit_b2[2]._data,
            self._B_tilde_par._data,
            self._curl_unit_b_dot_b0._data,
        )

    def _gradb_args(self):
        """The field arguments of the gyro-averaged drift kernels: ``gradB1 (3), grad_B_tilde_par (3), B_tilde_par,
        curl_unit_b_dot_b0``."""
        return (
            self._gradB1[0]._data,
            self._gradB1[1]._data,
            self._gradB1[2]._data,
            self._grad_B_tilde_par[0]._data,
            self._grad_B_tilde_par[1]._data,
            self._grad_B_tilde_par[2]._data,
            self._B_tilde_par._data,
            self._curl_unit_b_dot_b0._data,
        )


def _preconditioner(name, operator):
    if name is None:
        return None
    return getattr(preconditioner, name)(operator)


##############
# Propagators
##############


class MaxwellGyroMagnetisation(Propagator, _PerturbedFieldMixin):
    r"""FEEC discretization of Maxwell's equations of the gyrokinetic Maxwell model (Eq. (2.2) of the paper):
    find :math:`\mathbf E \in H(\textnormal{curl})` and :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        \int_{\Omega} \left(\mathbf I + \sum_s \alpha_s^2 \frac{n_{0,s}}{B_0^2} \Pi_\perp \right)\frac{\partial \mathbf E}{\partial t} \cdot \mathbf F \, \textrm d \mathbf x
        - \int_{\Omega} \left(\mathbf B - \frac{p_{0,\parallel} - p_{0,\perp}}{B_0^2} \mathbf B_\perp
        + \sum_s \alpha_s^2 \frac{n_{0,s} u_{0,\parallel,s}}{B_0^2} \mathbf b_0 \times \mathbf E
        + \sum_s \alpha_s^2 \int f_s\, \mu\, \mathbf b_0 \,\textrm d v_\parallel \textrm d \mu \right) \cdot \nabla \times \mathbf F \,\textrm d \mathbf x
        - \int_\Omega \sum_s \alpha_s^2 \frac{n_{0,s} u_{0,\parallel,s}}{B_0^2} \mathbf b_0 \times (\nabla \times \mathbf E) \cdot \mathbf F \,\textrm d \mathbf x &= 0\,, \qquad \forall \, \mathbf F \in H(\textnormal{curl})
        \\[2mm]
        \frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E &= 0\,,

    where :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}` (with ``background_current``, else :math:`\tilde{\mathbf B}` only,
    which assumes that the gyrocentre currents of the backgrounds balance :math:`\nabla \times \mathbf B_0`).
    The curl term with :math:`\mu \mathbf b_0` is the curl of the intrinsic magnetisation :math:`-\mu \mathbf b_0`
    of the gyrocentres. In matrix form (:func:`dielectric_matrix` :math:`\mathbb A`, :func:`magnetizing_matrix`
    :math:`\mathbb M^2_H`, :func:`flow_operator` :math:`\mathbb K = \mathbb C^\top \mathbb R - \mathbb R^\top \mathbb C`):

    .. math::

        \mathbb A \dot{\mathbf e} = \mathbb C^\top \mathbb M^2_H (\mathbf b + \mathbf b_0) + \mathbb K \mathbf e
        + \mathbb C^\top (\mathcal P^b)^\top \sum_s \alpha_s^2 \mathbf m_s\,,
        \qquad \dot{\mathbf b} = - \mathbb C \mathbf e\,,
        \qquad \mathbf m_s = \sum_p w_p \mu_p \langle\langle \vec \Lambda^0 \rangle\rangle_p\,,

    where :math:`\mathcal P^b` is the basis projection ``PB`` (:math:`\mathbf b \mapsto \Pi^0(\mathbf b_0 \cdot \mathbf B)`)
    and :math:`\langle\langle \cdot \rangle\rangle` the disc average (a point evaluation in the ZLR limit),
    such that the magnetisation source is the exact transpose of the map
    :math:`\mathbf b \mapsto \sum_p w_p \mu_p \langle\langle B_\parallel \rangle\rangle(\boldsymbol \eta_p)`
    of the particle energy: the system is Hamiltonian and the implicit mid-point rule conserves the energy exactly
    (for fixed markers); :math:`\mathbb K` is anti-symmetric and exchanges no energy.

    Time discretization: Crank-Nicolson (implicit mid-point),

    .. math::

        \left(\mathbb A + \frac{\Delta t^2}{4} \mathbb C^\top \mathbb M^2_H \mathbb C - \frac{\Delta t}{2} \mathbb K\right) \mathbf e^{n+1}
        = \left(\mathbb A - \frac{\Delta t^2}{4} \mathbb C^\top \mathbb M^2_H \mathbb C + \frac{\Delta t}{2} \mathbb K\right) \mathbf e^{n}
        + \Delta t\, \mathbb C^\top \mathbb M^2_H \mathbf b^n + \Delta t\, \mathbf s\,,
        \qquad \mathbf b^{n+1} = \mathbf b^n - \frac{\Delta t}{2} \mathbb C (\mathbf e^n + \mathbf e^{n+1})\,,

    solved with ``pcg`` (``pbicgstab`` with the flow operator).
    """

    class Variables:
        """Container for the variables advanced by :class:`MaxwellGyroMagnetisation`.

        Attributes
        ----------
        e : FEECVariable
            Electric field in H(curl) space.
        b : FEECVariable
            Magnetic field (perturbation) in H(div) space.
        """

        def __init__(self):
            self._e: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(
        self,
        species: tuple[PICVariable, ...] = (),
        background: BackgroundWeights = None,
        polarization: float = None,
        background_current: bool = False,
        gyro: GyroAverage = None,
    ):
        """
        Parameters
        ----------
        species : tuple[PICVariable]
            The gyrocentre species (``"Particles5D"``) whose magnetisation enters Ampère's law.
        background : BackgroundWeights, default=None
            The background moments of the species (polarization, flow, anisotropy). If ``None``, the polarization
            is ``polarization`` times the fluid equilibrium density, without flow and anisotropy.
        polarization : float, default=None
            The coefficient :math:`\\sum_s \\alpha_s^2` with ``background=None``; the sum over ``species`` if ``None``.
        background_current : bool
            Whether :math:`\\nabla \\times \\mathbf B_0` enters Ampère's law (the species backgrounds must then carry the
            equilibrium current, see :func:`parallel_flow_from_equilibrium`).
        gyro : GyroAverage, default=None
            The gyro-averaging (``None``: zero-Larmor-radius limit).
        """
        self.variables = self.Variables()
        for var in species:
            assert isinstance(var, PICVariable)
            assert var.space == "Particles5D"
        self.species = tuple(species)
        self.background = background
        if polarization is None:
            polarization = float(sum(var.species.equation_params.alpha**2 for var in species))
        self.polarization = polarization
        self.background_current = background_current
        self.gyro = gyro if gyro is not None else GyroAverage()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`MaxwellGyroMagnetisation`.

        Parameters
        ----------
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver (without background flow).

        solver_flow : LiteralOptions.OptsGenSolver, default="pbicgstab"
            General iterative solver (with background flow, the system is not symmetric).

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the dielectric matrix.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to ``SolverParameters()``.
        """

        solver: LiteralOptions.OptsSymmSolver = "pcg"
        solver_flow: LiteralOptions.OptsGenSolver = "pbicgstab"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.solver_flow, LiteralOptions.OptsGenSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        self._info = self.options.solver_params.info

        # field operators
        if self.background is None:
            self._A = dielectric_matrix(self.mass_ops, self.polarization)
            self._M2H = magnetizing_matrix(self.mass_ops, None)
            self._K = None
        else:
            self._A = dielectric_matrix(self.mass_ops, self.background.polarization)
            aniso = self.background.anisotropy if self.background.has_anisotropy else None
            self._M2H = magnetizing_matrix(self.mass_ops, aniso)
            if self.background.has_flow:
                R = flow_operator(self.mass_ops, self.background.flow)
                self._K = self.derham.curl.T @ R - R.T @ self.derham.curl
            else:
                self._K = None

        curl = self.derham.curl
        self._curlT = curl.T
        self._CTMC = curl.T @ self._M2H @ curl

        # magnetisation: accumulated 0-form per species, then C^T PB^T
        self._allocate_b_tilde(self.variables.b)
        self._accums = []
        self._accum_args = []
        for var in self.species:
            alpha2 = var.species.equation_params.alpha**2
            epsilon = var.species.equation_params.epsilon
            if self.gyro.flr:
                kernel, args = gk_mag_density_gyro, (alpha2, *self._gyro_args(self.gyro, epsilon, disc=True))
            else:
                kernel, args = gc_mag_density_0form, (alpha2,)
            self._accums += [AccumulatorVector(var.particles, "H1", kernel, self.mass_ops, self.domain.args_domain)]
            self._accum_args += [args]

        self._PBT = self.basis_ops.PB.T

        # pre-allocate arrays
        e_space = self.variables.e.spline.vector.space
        self._m_pol = self._PBT.domain.zeros()
        self._s2 = self._PBT.codomain.zeros()
        self._s1 = self._curlT.codomain.zeros()
        self._rhs = e_space.zeros()
        self._tmp = e_space.zeros()
        self._e_new = e_space.zeros()
        self._e_sum = e_space.zeros()
        self._b_new = self.variables.b.spline.vector.space.zeros()

        # the constant background current source C^T M2H b_0
        if self.background_current:
            self._s0 = self._curlT.dot(self._M2H.dot(self.projected_equil.b2))
        else:
            self._s0 = None

        # linear solver (the operator is set at each call, it depends on dt)
        pc = _preconditioner(self.options.precond, self._A)
        solver = self.options.solver if self._K is None else self.options.solver_flow
        self._A_inv = inverse(
            self._A,
            solver,
            pc=pc,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
        )

    @profile
    def __call__(self, dt):
        en = self.variables.e.spline.vector
        bn = self.variables.b.spline.vector

        # left- and right-hand side operators of the mid-point rule
        lhs = self._A + dt**2 / 4.0 * self._CTMC
        rhs_op = self._A - dt**2 / 4.0 * self._CTMC
        if self._K is not None:
            lhs = lhs - dt / 2.0 * self._K
            rhs_op = rhs_op + dt / 2.0 * self._K

        rhs = rhs_op.dot(en, out=self._rhs)

        # dt C^T M2H b^n
        self._M2H.dot(bn, out=self._s2)
        self._curlT.dot(self._s2, out=self._s1)
        self._s1 *= dt
        rhs += self._s1

        # background current
        if self._s0 is not None:
            self._s1 *= 0.0
            self._s1 += self._s0
            self._s1 *= dt
            rhs += self._s1

        # magnetisation source: dt C^T PB^T sum_s alpha_s^2 m_s
        if len(self._accums) > 0:
            self._m_pol *= 0.0
            for accum, args in zip(self._accums, self._accum_args):
                accum(*args)
                self._m_pol += accum.vectors[0]
            self._PBT.dot(self._m_pol, out=self._s2)
            self._curlT.dot(self._s2, out=self._s1)
            self._s1 *= dt
            rhs += self._s1

        # solve for new e coeffs
        self._A_inv.linop = lhs
        with ProfileManager.profile_region(self._solve_region, functions=[self._A_inv.solve]):
            en1 = self._A_inv.solve(rhs, out=self._e_new)
        info = self._A_inv._info

        # new b coeffs
        e_sum = en.copy(out=self._e_sum)
        e_sum += en1
        bn1 = self.derham.curl.dot(e_sum, out=self._b_new)
        bn1 *= -dt / 2.0
        bn1 += bn

        diffs = self.update_feec_variables(e=en1, b=bn1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for MaxwellGyroMagnetisation: {info['success']}")
            logger.info(f"Iterations for MaxwellGyroMagnetisation: {info['niter']}")
            logger.info(f"Maxdiff e for MaxwellGyroMagnetisation: {diffs['e']}")
            logger.info(f"Maxdiff b for MaxwellGyroMagnetisation: {diffs['b']}")
            logger.info("")


class GyroEparCoupling(Propagator, _PerturbedFieldMixin):
    r"""Parallel coupling of one gyrocentre species to the electric field:
    find :math:`\mathbf E \in H(\textnormal{curl})` and :math:`f` such that

    .. math::

        \int_{\Omega} \left(\mathbf I + \sum_s \alpha_s^2 \frac{n_{0,s}}{B_0^2} \Pi_\perp \right)\frac{\partial \mathbf{E}}{\partial t} \cdot \mathbf{F}\,\textrm d \mathbf x
        &= - \frac{\alpha^2}{\varepsilon} \int_{\Omega} \int f\, v_\parallel \frac{\mathbf B^*}{B^*_\parallel} \cdot \langle \mathbf{F} \rangle \,\textrm d v_\parallel \textrm d \mu \,\textrm d \mathbf x \qquad \forall \, \mathbf{F} \in H(\textnormal{curl})
        \\[2mm]
        \frac{\partial f}{\partial t} + \frac{1}{\varepsilon}\, \frac{\mathbf B^*}{B^*_\parallel} \cdot \langle \mathbf{E} \rangle\, \frac{\partial f}{\partial v_\parallel} &= 0\,,

    where :math:`\mathbf B^* = \mathbf B_0 + \langle \tilde{\mathbf B} \rangle + \varepsilon v_\parallel \nabla \times \mathbf b_0`
    (the FLR-corrected field :math:`\mathbf B_1^\star = \langle \mathbf B_1 \rangle` of the paper, Eq. (7.24)) and
    :math:`B^*_\parallel = \mathbf b_0 \cdot \mathbf B^*`; :math:`\langle \cdot \rangle` is the gyro-average (a point
    evaluation in the ZLR limit). For each marker, in logical coordinates,

    .. math::

        \mathbb A \dot{\mathbf e} = -\frac{\alpha^2}{\varepsilon} \sum_p w_p v_{\parallel,p}\, \langle \vec \Lambda^1 \rangle_p \cdot \mathbf c_p\,,
        \qquad \dot v_{\parallel,p} = \frac{1}{\varepsilon}\, \mathbf c_p \cdot \langle \vec \Lambda^{1\top} \rangle_p\, \mathbf e\,,
        \qquad \mathbf c_p = \frac{\hat{\mathbf B}^{*2}}{\sqrt g\, \hat B^*_\parallel}(\boldsymbol \eta_p)\,,

    with :math:`\mathbf c_p` frozen at :math:`v_{\parallel,p}^n`, such that the system is linear in :math:`(\mathbf e, v_\parallel)`.

    Time discretization: Crank-Nicolson (implicit mid-point), system size reduction via the Schur complement

    .. math::

        \left(\mathbb A + \frac{\Delta t^2 \alpha^2}{4 \varepsilon^2} \mathbb S\right) \mathbf e^{n+1}
        = \left(\mathbb A - \frac{\Delta t^2 \alpha^2}{4 \varepsilon^2} \mathbb S\right) \mathbf e^{n} - \Delta t \frac{\alpha^2}{\varepsilon} \mathbf j\,,
        \qquad \mathbb S = \sum_p w_p \langle \vec \Lambda^1 \rangle_p \mathbf c_p \mathbf c_p^\top \langle \vec \Lambda^{1\top} \rangle_p\,,
        \qquad \mathbf j = \sum_p w_p v^n_{\parallel,p} \mathbf c_p \langle \vec \Lambda^1 \rangle_p\,,

    with the stencil matrix :math:`\mathbb S` and :class:`~struphy.linear_algebra.schur_solver.SchurSolver` in
    the ZLR limit (same structure as :class:`~struphy.propagators.vlasov_ampere_coupling.VlasovAmpereCoupling`),
    and the matrix-free :class:`GyroParticleOperator` with finite Larmor radius.
    The accumulation and push kernels evaluate the identical :math:`\mathbf c_p`, which gives exact energy conservation.
    """

    class Variables:
        """Container for the variables advanced by :class:`GyroEparCoupling`.

        Attributes
        ----------
        e : FEECVariable
            Electric field in H(curl) space.
        var : PICVariable
            Gyrocentre particle variable in ``"Particles5D"`` space.
        """

        def __init__(self):
            self._e: FEECVariable = None
            self._var: PICVariable = None

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

        @property
        def var(self) -> PICVariable:
            return self._var

        @var.setter
        def var(self, new):
            assert isinstance(new, PICVariable)
            assert new.space == "Particles5D"
            self._var = new

    def __init__(
        self,
        b_tilde: FEECVariable = None,
        polarization=0.0,
        gyro: GyroAverage = None,
    ):
        """
        Parameters
        ----------
        b_tilde : FEECVariable, default=None
            Magnetic perturbation in ``"Hdiv"`` space entering :math:`\\mathbf B^*`.
            If ``None``, only the equilibrium field is used.
        polarization : float | Callable
            The polarization of the dielectric matrix, see :func:`dielectric_matrix`.
        gyro : GyroAverage, default=None
            The gyro-averaging (``None``: zero-Larmor-radius limit).
        """
        self.variables = self.Variables()
        self.b_tilde = b_tilde
        self.polarization = polarization
        self.gyro = gyro if gyro is not None else GyroAverage()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`GyroEparCoupling`.

        Parameters
        ----------
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the dielectric matrix.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to ``SolverParameters()``.
        """

        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        self._info = self.options.solver_params.info

        # scaling factors
        alpha = self.variables.var.species.equation_params.alpha
        self._epsilon = self.variables.var.species.equation_params.epsilon
        self._c_mat = alpha**2 / (4.0 * self._epsilon**2)
        self._c_vec = alpha**2 / self._epsilon

        # dielectric matrix
        self._A = dielectric_matrix(self.mass_ops, self.polarization)

        # magnetic field (B_0 + B_tilde)
        self._allocate_b_tilde(self.b_tilde)
        particles = self.variables.var.particles

        # buffers
        e_space = self.variables.e.spline.vector.space
        self._e_new = e_space.zeros()
        self._e_mid = e_space.zeros()
        self._rhs = e_space.zeros()
        self._e_mid_tp = self.derham.V1.zeros()

        pc = _preconditioner(self.options.precond, self._A)

        if self.gyro.flr:
            # matrix-free Schur complement with gyro-averages
            gyro_args = self._gyro_args(self.gyro, self._epsilon, disc=False)
            self._S = GyroParticleOperator(
                particles,
                self.derham,
                self.domain.args_domain,
                gk_epar_gyro_matvec,
                (*gyro_args, *self._bstar_args()),
                scale=1.0,
                symmetric=True,
            )
            self._accum = AccumulatorVector(
                particles, "Hcurl", gk_epar_gyro_vector, self.mass_ops, self.domain.args_domain
            )
            self._args_accum = (*gyro_args, *self._bstar_args())
            self._A_inv = inverse(
                self._A,
                self.options.solver,
                pc=pc,
                tol=self.options.solver_params.tol,
                maxiter=self.options.solver_params.maxiter,
                verbose=self.options.solver_params.verbose,
            )
            args_kernel = (
                self.derham.args_derham,
                *gyro_args,
                *self._bstar_args(),
                self._e_mid_tp[0]._data,
                self._e_mid_tp[1]._data,
                self._e_mid_tp[2]._data,
            )
            kernel = gk_push_vpar_gyro
        else:
            # stencil matrix and Schur solver
            self._accum = Accumulator(
                particles,
                "Hcurl",
                gk_epar_coupling,
                self.mass_ops,
                self.domain.args_domain,
                add_vector=True,
                symmetry="symm",
            )
            self._args_accum = (
                self._epsilon,
                self._b2_full[0]._data,
                self._b2_full[1]._data,
                self._b2_full[2]._data,
                self._curl_unit_b2[0]._data,
                self._curl_unit_b2[1]._data,
                self._curl_unit_b2[2]._data,
                self._B_dot_b._data,
                self._curl_unit_b_dot_b0._data,
            )
            self._schur_solver = SchurSolver(
                self._A,
                self._accum.operators[0],
                self.options.solver,
                precond=pc,
                solver_params=self.options.solver_params,
            )
            args_kernel = (
                self.derham.args_derham,
                self._epsilon,
                self._b2_full[0]._data,
                self._b2_full[1]._data,
                self._b2_full[2]._data,
                self._curl_unit_b2[0]._data,
                self._curl_unit_b2[1]._data,
                self._curl_unit_b2[2]._data,
                self._B_dot_b._data,
                self._curl_unit_b_dot_b0._data,
                self._e_mid_tp[0]._data,
                self._e_mid_tp[1]._data,
                self._e_mid_tp[2]._data,
            )
            kernel = gk_push_vpar_efield

        # pusher of the parallel velocity in the mid-point electric field
        self._pusher = Pusher(
            particles,
            kernel,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
            pushes_eta=False,
        )

    @profile
    def __call__(self, dt):
        en = self.variables.e.spline.vector

        # magnetic field and accumulation at v_parallel^n
        self._update_b_tilde()
        self._accum(*self._args_accum)

        if self.gyro.flr:
            # (A + dt^2 c S) e^{n+1} = (A - dt^2 c S) e^n - dt alpha^2/eps j
            lhs = self._A + dt**2 * self._c_mat * self._S
            rhs_op = self._A - dt**2 * self._c_mat * self._S
            rhs = rhs_op.dot(en, out=self._rhs)
            self._e_mid *= 0.0
            self._e_mid += self._accum.vectors[0]
            self._e_mid *= dt * self._c_vec
            rhs -= self._e_mid

            self._A_inv.linop = lhs
            with ProfileManager.profile_region(self._solve_region, functions=[self._A_inv.solve]):
                en1 = self._A_inv.solve(rhs, out=self._e_new)
            info = self._A_inv._info
        else:
            # update Schur solver: BC = -alpha^2/(4 eps^2) * sum_p w_p Lambda c c^T Lambda^T
            self._schur_solver.BC = self._accum.operators[0]
            self._schur_solver.BC *= -self._c_mat

            # Byn = alpha^2/(2 eps) * sum_p w_p v_p c_p Lambda
            self._rhs *= 0.0
            self._rhs += self._accum.vectors[0]
            self._rhs *= self._c_vec / 2.0

            en1, info = self._schur_solver(en, self._rhs, dt, out=self._e_new)

        # mid-point e-field in the tensor-product space (the pusher kernel reads the ghost regions)
        e_mid = en.copy(out=self._e_mid)
        e_mid += en1
        e_mid *= 0.5
        e_mid_tp = self._E1T.dot(e_mid, out=self._e_mid_tp)
        e_mid_tp.update_ghost_regions()

        # update parallel velocities
        self._pusher(dt)

        # update weights in case of control variate
        if self.variables.var.species.weights_params.control_variate:
            self.variables.var.particles.update_weights()

        # write new coeffs into self.variables
        max_de = self.update_feec_variables(e=en1)["e"]

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for GyroEparCoupling: {info['success']}")
            logger.info(f"Iterations for GyroEparCoupling: {info['niter']}")
            logger.info(f"Maxdiff e  for GyroEparCoupling: {max_de}")
            logger.info("")


class _EquilibriumWithFullB:
    """Proxy of a projected equilibrium whose ``b2`` is a given (persistent) vector holding :math:`\\mathbf B_0 + \\tilde{\\mathbf B}`;
    all other attributes are those of the projected equilibrium."""

    def __init__(self, projected_equil, b2_full):
        self._projected_equil = projected_equil
        self._b2_full = b2_full

    @property
    def b2(self):
        return self._b2_full

    def __getattr__(self, name):
        return getattr(self._projected_equil, name)


class PushGuidingCenterParallelFullB(PushGuidingCenterParallel):
    r"""Same as :class:`~struphy.propagators.push_guiding_center_parallel.PushGuidingCenterParallel`
    (without electric potential), but with the magnetic perturbation in the full :math:`\mathbf B^*`:

    .. math::

        \mathbf B^* = \mathbf B_0 + \tilde{\mathbf B} + \varepsilon v_\parallel \nabla \times \mathbf b_0\,,\qquad
        B^*_\parallel = |B_0| + \Pi^0(\mathbf b_0 \cdot \tilde{\mathbf B}) + \varepsilon v_\parallel\, \mathbf b_0 \cdot \nabla \times \mathbf b_0\,,

    which adds the magnetic flutter (streaming along the perturbed field lines) to the parallel motion
    (the parent class uses :math:`\tilde{\mathbf B}` only in :math:`B^*_\parallel` and in :math:`\mathbf E^* = -\varepsilon \mu \nabla B_\parallel`).
    The vector ``b2`` of the projected equilibrium seen by the kernels is replaced by a persistent vector
    holding :math:`\mathbf B_0 + \tilde{\mathbf B}` (tensor-product coefficients), updated before every push.
    Zero-Larmor-radius limit; see :class:`GyroParallelPush` for finite Larmor radius.
    """

    def __init__(self, b_tilde: FEECVariable):
        """
        Parameters
        ----------
        b_tilde : FEECVariable
            Magnetic perturbation in ``"Hdiv"`` space.
        """
        assert isinstance(b_tilde, FEECVariable)
        assert b_tilde.space == "Hdiv"
        super().__init__(phi=None, b_tilde=b_tilde)

    @property
    def projected_equil(self):
        equil = super().projected_equil
        if getattr(self, "_b2_full", None) is None:
            return equil
        return _EquilibriumWithFullB(equil, self._b2_full)

    @profile
    def allocate(self):
        self._E2T_full = self.derham.extraction_ops["2"].transpose()
        self._b2_eq_full = super().projected_equil.b2
        self._b2_full = self.derham.V2.zeros()
        self._update_b2_full()

        super().allocate()

    def _update_b2_full(self):
        """Full magnetic field (equilibrium + perturbation) as 2-form in the tensor-product space."""
        b2_full = self._E2T_full.dot(self.b_tilde.spline.vector, out=self._b2_full)
        b2_full += self._b2_eq_full
        b2_full.update_ghost_regions()

    @profile
    def __call__(self, dt):
        self._update_b2_full()
        super().__call__(dt)


class GyroParallelPush(Propagator, _PerturbedFieldMixin):
    r"""Parallel gyrocentre motion with finite Larmor radius: for each marker :math:`p`, solves

    .. math::

        \left\{
            \begin{aligned}
                \frac{\textnormal d \mathbf X_p(t)}{\textnormal d t} &= v_{\parallel,p}(t) \frac{\mathbf B^*}{B^*_\parallel}(\mathbf X_p(t)) \,,
                \\
                \frac{\textnormal d v_{\parallel,p}(t)}{\textnormal d t} &= -\mu_p \frac{\mathbf B^*}{B^*_\parallel} \cdot \nabla \left(B_0 + \langle\langle \tilde B_\parallel \rangle\rangle\right) (\mathbf X_p(t)) \,,
            \end{aligned}
        \right.

    with :math:`\mathbf B^* = \mathbf B_0 + \langle \tilde{\mathbf B} \rangle + \varepsilon v_\parallel \nabla \times \mathbf b_0`,
    :math:`B^*_\parallel = |B_0| + \langle \tilde B_\parallel \rangle + \varepsilon v_\parallel \mathbf b_0 \cdot \nabla \times \mathbf b_0`
    (Eqs. (5.23)-(5.24) of the paper with :math:`\mathbf B_1^\star = \langle \mathbf B_1 \rangle` and the disc-averaged
    parallel perturbation in the mirror force), by the first-order discrete-gradient (Gonzalez) scheme of
    :class:`~struphy.propagators.push_guiding_center_parallel.PushGuidingCenterParallel`, which conserves the
    energy :math:`v_\parallel^2/2 + \mu (B_0 + \langle\langle \tilde B_\parallel \rangle\rangle)` of each marker exactly.
    """

    class Variables:
        """Container for the variables advanced by :class:`GyroParallelPush`.

        Attributes
        ----------
        ions : PICVariable
            Gyrocentre particle variable in ``"Particles5D"`` space.
        """

        def __init__(self):
            self._ions: PICVariable = None

        @property
        def ions(self) -> PICVariable:
            return self._ions

        @ions.setter
        def ions(self, new):
            assert isinstance(new, PICVariable)
            assert new.space == "Particles5D"
            self._ions = new

    def __init__(self, b_tilde: FEECVariable = None, gyro: GyroAverage = None):
        """
        Parameters
        ----------
        b_tilde : FEECVariable, default=None
            Magnetic perturbation in ``"Hdiv"`` space; ``None`` for the equilibrium field only.
        gyro : GyroAverage, default=None
            The gyro-averaging (``None``: zero-Larmor-radius limit).
        """
        self.variables = self.Variables()
        self.b_tilde = b_tilde
        self.gyro = gyro if gyro is not None else GyroAverage()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`GyroParallelPush`.

        Parameters
        ----------
        maxiter : int, default=20
            Maximum number of fixed-point iterations of the discrete-gradient scheme.

        tol : float, default=1e-7
            Convergence tolerance of the iterations.

        mpi_sort : LiteralOptions.OptsMPIsort, default="each"
            MPI sorting policy for particle exchange.
        """

        maxiter: int = 20
        tol: float = 1e-7
        mpi_sort: LiteralOptions.OptsMPIsort = "each"

        def __post_init__(self):
            check_option(self.mpi_sort, LiteralOptions.OptsMPIsort)

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        self._epsilon = self.variables.ions.species.equation_params.epsilon
        self._allocate_b_tilde(self.b_tilde)

        particles = self.variables.ions.particles
        first_free_idx = particles.args_markers.first_free_idx

        ring_args = self._gyro_args(self.gyro, self._epsilon, disc=False)
        disc_args = self._gyro_args(self.gyro, self._epsilon, disc=True)
        hamiltonian_args = (self.derham.args_derham, *disc_args, self._B_tilde_par._data)

        # init kernels (at eta^n): Hamiltonian, B*_parallel and B*
        self.add_init_kernel(
            KernelSetup(kernel=gk_hamiltonian_gyro, output_indices=(first_free_idx,), args=hamiltonian_args)
        )
        bstar_args = (self.derham.args_derham, *ring_args, *self._bstar_args())
        self.add_init_kernel(
            KernelSetup(kernel=gk_bstar_parallel_gyro, output_indices=(first_free_idx + 1,), args=bstar_args)
        )
        self.add_init_kernel(
            KernelSetup(
                kernel=gk_bstar_gyro,
                output_indices=(first_free_idx + 2, first_free_idx + 3, first_free_idx + 4),
                args=bstar_args,
            )
        )
        # eval kernel (at eta^{n+1,k}): Hamiltonian
        self.add_eval_kernel(
            KernelSetup(
                kernel=gk_hamiltonian_gyro, output_indices=(first_free_idx + 5,), args=hamiltonian_args, alpha=1.0
            )
        )

        args_kernel = (
            self.derham.args_derham,
            *disc_args,
            self._gradB1[0]._data,
            self._gradB1[1]._data,
            self._gradB1[2]._data,
            self._grad_B_tilde_par[0]._data,
            self._grad_B_tilde_par[1]._data,
            self._grad_B_tilde_par[2]._data,
        )

        self._pusher = Pusher(
            particles,
            gk_push_parallel_dg1_gyro,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=0.5,
            init_kernels=self.init_kernels,
            eval_kernels=self.eval_kernels,
            maxiter=self.options.maxiter,
            tol=self.options.tol,
            mpi_sort=self.options.mpi_sort,
            pushes_eta=True,
        )

    @profile
    def __call__(self, dt):
        self._update_b_tilde()
        self._pusher(dt)

        if self.variables.ions.species.weights_params.control_variate:
            self.variables.ions.particles.update_weights()


class GyroDriftCoupling(Propagator, _PerturbedFieldMixin):
    r"""Perpendicular (drift) coupling of all gyrocentre species to the electric field:
    find :math:`\mathbf E \in H(\textnormal{curl})` and :math:`f_s` such that

    .. math::

        \int_{\Omega} \left(\mathbf I + \sum_s \alpha_s^2 \frac{n_{0,s}}{B_0^2} \Pi_\perp \right)\frac{\partial \mathbf{E}}{\partial t} \cdot \mathbf{F}\,\textrm d \mathbf x
        &= - \sum_s \frac{\alpha_s^2}{\varepsilon_s} \int_{\Omega} \int f_s\, \frac{\mathbf E^* \times \mathbf b_0}{B^*_\parallel} \cdot \langle \mathbf{F} \rangle \,\textrm d v_\parallel \textrm d \mu \,\textrm d \mathbf x \qquad \forall \, \mathbf{F} \in H(\textnormal{curl})
        \\[2mm]
        \frac{\partial f_s}{\partial t} + \frac{\mathbf E^* \times \mathbf b_0}{B^*_\parallel} \cdot \nabla f_s &= 0\,,
        \qquad \mathbf E^* = \langle \mathbf E \rangle - \varepsilon_s \mu \nabla \left(B_0 + \langle\langle \tilde B_\parallel \rangle\rangle\right)\,,

    (the gyro- and disc averages are point evaluations in the ZLR limit).
    With the markers at :math:`\boldsymbol \eta^n`, the :math:`E \times B` current is an anti-symmetric operator
    :math:`\mathbb Q = \sum_s \frac{\alpha_s^2}{\varepsilon_s} \sum_p w_p \langle \vec \Lambda^1 \rangle_p \frac{\epsilon_{\mu\nu\lambda} \hat b_{0,\lambda}}{\sqrt g \hat B^*_\parallel} \langle \vec \Lambda^{1\top} \rangle_p`
    (hence exchanges no energy, :math:`\mathbf E \cdot (\mathbf E \times \mathbf b_0) = 0`) and the
    :math:`\nabla B` current a vector :math:`\mathbf j = \sum_s \frac{\alpha_s^2}{\varepsilon_s} \sum_p w_p \dot{\boldsymbol \eta}_{\nabla B,p} \langle \vec \Lambda^1 \rangle_p`:

    .. math::

        \left(\mathbb A + \frac{\Delta t}{2} \mathbb Q \right) \mathbf e^{n+1} = \left(\mathbb A - \frac{\Delta t}{2} \mathbb Q \right) \mathbf e^{n} - \Delta t\, \mathbf j\,,

    with a stencil matrix :math:`\mathbb Q` in the ZLR limit and the matrix-free :class:`GyroParticleOperator` with
    finite Larmor radius. The markers are then pushed with an explicit Runge-Kutta scheme
    in the mid-point field :math:`(\mathbf e^n + \mathbf e^{n+1})/2`.
    The energy exchange of the :math:`\nabla B` current with the particle energy :math:`\mu B_\parallel` is consistent to second order.

    All species are handled in one propagator: a single species' :math:`E \times B` current, balanced only by the
    polarization, gives a spurious stiff mode near the cyclotron frequency.
    """

    class Variables:
        """Container for the variables advanced by :class:`GyroDriftCoupling`.

        Attributes
        ----------
        e : FEECVariable
            Electric field in H(curl) space.
        """

        def __init__(self):
            self._e: FEECVariable = None

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

    def __init__(
        self,
        species: tuple[PICVariable, ...] = (),
        b_tilde: FEECVariable = None,
        polarization=0.0,
        gyro: GyroAverage = None,
    ):
        """
        Parameters
        ----------
        species : tuple[PICVariable]
            The gyrocentre species (``"Particles5D"``) that are pushed and whose drift currents enter Ampère's law.
        b_tilde : FEECVariable, default=None
            Magnetic perturbation in ``"Hdiv"`` space entering :math:`B_\\parallel`.
            If ``None``, only the equilibrium field is used.
        polarization : float | Callable
            The polarization of the dielectric matrix, see :func:`dielectric_matrix`.
        gyro : GyroAverage, default=None
            The gyro-averaging (``None``: zero-Larmor-radius limit).
        """
        self.variables = self.Variables()
        for var in species:
            assert isinstance(var, PICVariable)
            assert var.space == "Particles5D"
        self.species = tuple(species)
        self.b_tilde = b_tilde
        self.polarization = polarization
        self.gyro = gyro if gyro is not None else GyroAverage()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`GyroDriftCoupling`.

        Parameters
        ----------
        solver : LiteralOptions.OptsGenSolver, default="pbicgstab"
            General iterative solver for the (non-symmetric) linear system.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the dielectric matrix.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to ``SolverParameters()``.

        butcher : ButcherTableau, default=None
            Butcher tableau of the explicit marker push. If ``None``, defaults to ``ButcherTableau(algo="rk2")``.

        mpi_sort : LiteralOptions.OptsMPIsort, default="last"
            MPI sorting policy for particle exchange (the markers are sorted after the full step).
        """

        solver: LiteralOptions.OptsGenSolver = "pbicgstab"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        butcher: ButcherTableau = None
        mpi_sort: LiteralOptions.OptsMPIsort = "last"

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsGenSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
            check_option(self.mpi_sort, LiteralOptions.OptsMPIsort)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.butcher is None:
                self.butcher = ButcherTableau(algo="rk2")

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        self._info = self.options.solver_params.info

        # dielectric matrix
        self._A = dielectric_matrix(self.mass_ops, self.polarization)

        # magnetic field (B_0 + B_tilde)
        self._allocate_b_tilde(self.b_tilde)

        # buffers
        e_space = self.variables.e.spline.vector.space
        self._e_new = e_space.zeros()
        self._e_mid = e_space.zeros()
        self._rhs = e_space.zeros()
        self._j = e_space.zeros()
        self._e_mid_tp = self.derham.V1.zeros()

        # accumulators (matrix and vector, or matrix-free operator and vector) and pushers, one per species
        butcher = self.options.butcher
        self._accums = []
        self._args_accum = []
        self._operators = []
        self._pushers = []
        for var in self.species:
            particles = var.particles
            alpha = var.species.equation_params.alpha
            epsilon = var.species.equation_params.epsilon
            scale = alpha**2 / epsilon

            if self.gyro.flr:
                ring_args = self._gyro_args(self.gyro, epsilon, disc=False)
                disc_args = self._gyro_args(self.gyro, epsilon, disc=True)
                self._operators += [
                    GyroParticleOperator(
                        particles,
                        self.derham,
                        self.domain.args_domain,
                        gk_drift_gyro_matvec,
                        (scale, *ring_args, self._B_tilde_par._data, self._curl_unit_b_dot_b0._data),
                        scale=1.0,
                        symmetric=False,
                    )
                ]
                self._accums += [
                    AccumulatorVector(particles, "Hcurl", gk_drift_gyro_vector, self.mass_ops, self.domain.args_domain)
                ]
                # the drift kernels take the ring and the disc nodes: epsilon, n_gyro, ring, disc, unit_b1, absB0
                self._args_accum += [
                    (
                        scale,
                        epsilon,
                        self.gyro.n_gyro,
                        self.gyro.ring_s,
                        self.gyro.ring_w,
                        self.gyro.disc_s,
                        self.gyro.disc_w,
                        *ring_args[4:],
                        *self._gradb_args(),
                    )
                ]
                args_kernel = (
                    self.derham.args_derham,
                    epsilon,
                    self.gyro.n_gyro,
                    self.gyro.ring_s,
                    self.gyro.ring_w,
                    self.gyro.disc_s,
                    self.gyro.disc_w,
                    *ring_args[4:],
                    *self._gradb_args(),
                    self._e_mid_tp[0]._data,
                    self._e_mid_tp[1]._data,
                    self._e_mid_tp[2]._data,
                    butcher.a_stage,
                    butcher.b,
                    butcher.c,
                )
                kernel = gk_push_drift_gyro
            else:
                self._accums += [
                    Accumulator(
                        particles,
                        "Hcurl",
                        gk_drift_coupling,
                        self.mass_ops,
                        self.domain.args_domain,
                        add_vector=True,
                        symmetry="asym",
                    )
                ]
                self._args_accum += [
                    (
                        epsilon,
                        scale,
                        self._unit_b1[0]._data,
                        self._unit_b1[1]._data,
                        self._unit_b1[2]._data,
                        self._grad_b_full[0]._data,
                        self._grad_b_full[1]._data,
                        self._grad_b_full[2]._data,
                        self._B_dot_b._data,
                        self._curl_unit_b_dot_b0._data,
                    )
                ]
                args_kernel = (
                    self.derham.args_derham,
                    epsilon,
                    self._unit_b1[0]._data,
                    self._unit_b1[1]._data,
                    self._unit_b1[2]._data,
                    self._grad_b_full[0]._data,
                    self._grad_b_full[1]._data,
                    self._grad_b_full[2]._data,
                    self._B_dot_b._data,
                    self._curl_unit_b_dot_b0._data,
                    self._e_mid_tp[0]._data,
                    self._e_mid_tp[1]._data,
                    self._e_mid_tp[2]._data,
                    True,
                    butcher.a_stage,
                    butcher.b,
                    butcher.c,
                )
                kernel = push_gc_bxEstar_explicit_multistage

            self._pushers += [
                Pusher(
                    particles,
                    kernel,
                    args_kernel,
                    self.domain.args_domain,
                    alpha_in_kernel=1.0,
                    n_stages=butcher.n_stages,
                    mpi_sort=self.options.mpi_sort,
                    pushes_eta=True,
                )
            ]

        # preconditioner and linear solver
        pc = _preconditioner(self.options.precond, self._A)
        self._A_inv = inverse(
            self._A,
            self.options.solver,
            pc=pc,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
        )

    @profile
    def __call__(self, dt):
        en = self.variables.e.spline.vector

        # magnetic field and accumulation at eta^n
        self._update_b_tilde()
        for accum, args in zip(self._accums, self._args_accum):
            accum(*args)

        # (A + dt/2 Q) e^{n+1} = (A - dt/2 Q) e^n - dt j
        lhs = self._A
        rhs_op = self._A
        self._j *= 0.0
        operators = self._operators if self.gyro.flr else [accum.operators[0] for accum in self._accums]
        for accum, Q in zip(self._accums, operators):
            lhs = lhs + dt / 2 * Q
            rhs_op = rhs_op - dt / 2 * Q
            self._j += accum.vectors[0]
        self._j *= dt

        rhs = rhs_op.dot(en, out=self._rhs)
        rhs -= self._j

        self._A_inv.linop = lhs
        with ProfileManager.profile_region(self._solve_region, functions=[self._A_inv.solve]):
            en1 = self._A_inv.solve(rhs, out=self._e_new)
        info = self._A_inv._info

        # mid-point e-field in the tensor-product space (the pusher kernel reads the ghost regions)
        e_mid = en.copy(out=self._e_mid)
        e_mid += en1
        e_mid *= 0.5
        e_mid_tp = self._E1T.dot(e_mid, out=self._e_mid_tp)
        e_mid_tp.update_ghost_regions()

        # push markers
        for var, pusher in zip(self.species, self._pushers):
            pusher(dt)
            if var.species.weights_params.control_variate:
                var.particles.update_weights()

        # write new coeffs into self.variables
        max_de = self.update_feec_variables(e=en1)["e"]

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for GyroDriftCoupling: {info['success']}")
            logger.info(f"Iterations for GyroDriftCoupling: {info['niter']}")
            logger.info(f"Maxdiff e  for GyroDriftCoupling: {max_de}")
            logger.info("")
