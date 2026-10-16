"""Propagators of the quasi-neutral gyrokinetic Darwin model of Remmerswaal, Hatzky and Sonnendrücker
(J. Plasma Phys. 91, E127, 2025, Sec. 6), see :class:`~struphy.models.gyrokinetic_darwin.GyrokineticDarwin`.

The unknowns are the potentials :math:`\\phi_1 \\in H^1` and :math:`\\mathbf A_1 \\in H(\\textnormal{curl})` in the
perpendicular Coulomb gauge (Eq. (6.12)), with the Lagrange multiplier :math:`\\lambda \\in H^1` of the gauge
condition. The field operators (polarization, magnetizing and flow operators, the gyro-averaged particle
couplings) are those of :mod:`struphy.propagators.gyrokinetic_maxwell`.
"""

import logging
from dataclasses import dataclass

import cunumpy as xp
from feectools.linalg.basic import LinearOperator
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI
from scope_profiler import ProfileManager

from struphy.feec import preconditioner
from struphy.feec.mass import WeightedAverageProjection
from struphy.io.options import LiteralOptions, OptionsBase
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
from struphy.pic.pushing.kernels.gk_push_drift_gyro import gk_push_drift_gyro
from struphy.pic.pushing.kernels.gk_push_vpar_efield import gk_push_vpar_efield
from struphy.pic.pushing.kernels.gk_push_vpar_gyro import gk_push_vpar_gyro
from struphy.pic.pushing.kernels.push_gc_bxEstar_explicit_multistage import push_gc_bxEstar_explicit_multistage
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.propagators.gyrokinetic_maxwell import (
    BackgroundWeights,
    GyroAverage,
    GyroParticleOperator,
    _PerturbedFieldMixin,
    dielectric_matrix,
    magnetizing_matrix,
    polarization_matrix,
    skin_preconditioner_matrix,
)
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


#####################################
# Particle couplings of one species
#####################################


class _ParallelCoupling:
    r"""The parallel coupling of one species: the matrix :math:`\mathbb S = \sum_p w_p \langle \vec \Lambda^1 \rangle_p \mathbf c_p \mathbf c_p^\top \langle \vec \Lambda^{1\top} \rangle_p`
    (stencil matrix in the ZLR limit, matrix-free with finite Larmor radius), the current
    :math:`\mathbf j = \sum_p w_p v_{\parallel,p} \mathbf c_p \langle \vec \Lambda^1 \rangle_p` and the pusher of
    :math:`v_\parallel` in a given 1-form field (tensor-product coefficients ``e_tp``)."""

    def __init__(self, prop: _PerturbedFieldMixin, var: PICVariable, gyro: GyroAverage, e_tp):
        self.var = var
        self.gyro = gyro
        particles = var.particles
        epsilon = var.species.equation_params.epsilon
        self.epsilon = epsilon
        e_args = (e_tp[0]._data, e_tp[1]._data, e_tp[2]._data)

        if gyro.flr:
            gyro_args = prop._gyro_args(gyro, epsilon, disc=False)
            self._operator = GyroParticleOperator(
                particles,
                prop.derham,
                prop.domain.args_domain,
                gk_epar_gyro_matvec,
                (*gyro_args, *prop._bstar_args()),
                scale=1.0,
                symmetric=True,
            )
            self._accum = AccumulatorVector(
                particles, "Hcurl", gk_epar_gyro_vector, prop.mass_ops, prop.domain.args_domain
            )
            self._args = (*gyro_args, *prop._bstar_args())
            args_kernel = (prop.derham.args_derham, *gyro_args, *prop._bstar_args(), *e_args)
            kernel = gk_push_vpar_gyro
        else:
            self._accum = Accumulator(
                particles,
                "Hcurl",
                gk_epar_coupling,
                prop.mass_ops,
                prop.domain.args_domain,
                add_vector=True,
                symmetry="symm",
            )
            self._args = (
                epsilon,
                prop._b2_full[0]._data,
                prop._b2_full[1]._data,
                prop._b2_full[2]._data,
                prop._curl_unit_b2[0]._data,
                prop._curl_unit_b2[1]._data,
                prop._curl_unit_b2[2]._data,
                prop._B_dot_b._data,
                prop._curl_unit_b_dot_b0._data,
            )
            self._operator = None
            args_kernel = (prop.derham.args_derham, *self._args, *e_args)
            kernel = gk_push_vpar_efield

        self.pusher = Pusher(
            particles, kernel, args_kernel, prop.domain.args_domain, alpha_in_kernel=1.0, pushes_eta=False
        )

    def accumulate(self):
        self._accum(*self._args)

    @property
    def operator(self):
        """The (unscaled) matrix :math:`\\mathbb S`."""
        return self._operator if self.gyro.flr else self._accum.operators[0]

    @property
    def vector(self):
        """The (unscaled) current :math:`\\mathbf j`."""
        return self._accum.vectors[0]


class _DriftCoupling:
    r"""The drift coupling of one species: the anti-symmetric E x B operator :math:`\mathbb Q_s` and the grad-B
    current :math:`\mathbf j_s` (both scaled by :math:`\alpha_s^2/\varepsilon_s`), and the explicit pusher of the
    drifts in a given 1-form field (tensor-product coefficients ``e_tp``)."""

    def __init__(self, prop: _PerturbedFieldMixin, var: PICVariable, gyro: GyroAverage, e_tp, butcher, mpi_sort):
        self.var = var
        self.gyro = gyro
        particles = var.particles
        alpha = var.species.equation_params.alpha
        epsilon = var.species.equation_params.epsilon
        scale = alpha**2 / epsilon
        e_args = (e_tp[0]._data, e_tp[1]._data, e_tp[2]._data)

        if gyro.flr:
            ring_args = prop._gyro_args(gyro, epsilon, disc=False)
            self._operator = GyroParticleOperator(
                particles,
                prop.derham,
                prop.domain.args_domain,
                gk_drift_gyro_matvec,
                (scale, *ring_args, prop._B_tilde_par._data, prop._curl_unit_b_dot_b0._data),
                scale=1.0,
                symmetric=False,
            )
            self._accum = AccumulatorVector(
                particles, "Hcurl", gk_drift_gyro_vector, prop.mass_ops, prop.domain.args_domain
            )
            flr_args = (
                epsilon,
                gyro.n_gyro,
                gyro.ring_s,
                gyro.ring_w,
                gyro.disc_s,
                gyro.disc_w,
                *ring_args[4:],
                *prop._gradb_args(),
            )
            self._args = (scale, *flr_args)
            args_kernel = (prop.derham.args_derham, *flr_args, *e_args, butcher.a_stage, butcher.b, butcher.c)
            kernel = gk_push_drift_gyro
        else:
            self._accum = Accumulator(
                particles,
                "Hcurl",
                gk_drift_coupling,
                prop.mass_ops,
                prop.domain.args_domain,
                add_vector=True,
                symmetry="asym",
            )
            field_args = (
                prop._unit_b1[0]._data,
                prop._unit_b1[1]._data,
                prop._unit_b1[2]._data,
                prop._grad_b_full[0]._data,
                prop._grad_b_full[1]._data,
                prop._grad_b_full[2]._data,
                prop._B_dot_b._data,
                prop._curl_unit_b_dot_b0._data,
            )
            self._args = (epsilon, scale, *field_args)
            self._operator = None
            args_kernel = (
                prop.derham.args_derham,
                epsilon,
                *field_args,
                *e_args,
                True,
                butcher.a_stage,
                butcher.b,
                butcher.c,
            )
            kernel = push_gc_bxEstar_explicit_multistage

        self.pusher = Pusher(
            particles,
            kernel,
            args_kernel,
            prop.domain.args_domain,
            alpha_in_kernel=1.0,
            n_stages=butcher.n_stages,
            mpi_sort=mpi_sort,
            pushes_eta=True,
        )

    def accumulate(self):
        self._accum(*self._args)

    @property
    def operator(self):
        """The operator :math:`\\mathbb Q_s` (scaled)."""
        return self._operator if self.gyro.flr else self._accum.operators[0]

    @property
    def vector(self):
        """The grad-B current :math:`\\mathbf j_s` (scaled)."""
        return self._accum.vectors[0]


##############
# Propagators
##############


class DarwinLongitudinalOperator(LinearOperator):
    r"""The longitudinal polarization operator of the gyrokinetic Darwin model on 1-forms,

    .. math::

        \mathbb P_L = \mathbb P \mathbb G \left(\mathbb G^\top \mathbb P \mathbb G + \sigma \mathbb M^0 \Pi_{\textnormal{avg}}\right)^{-1} \mathbb G^\top \mathbb P\,,

    the weak form of the Darwin displacement :math:`\mathbf D^{\textnormal{Dar}} = \Pi_{L,\perp}(\mathcal C(1) \mathbf E_\perp)`
    (Eqs. (6.6), (6.9a) of the paper): the :math:`\mathbb P`-orthogonal projection of a 1-form on gradients, applied
    matrix-free with an inner (preconditioned) conjugate-gradient solve of the perpendicular Laplacian.
    The average projection :math:`\Pi_{\textnormal{avg}}` over the perpendicular directions regularizes the
    kernel of the perpendicular Laplacian (the :math:`k_\perp = 0` modes, which carry no longitudinal
    displacement). The operator is symmetric and positive semi-definite.

    For the perpendicular fields that are constant in the perpendicular directions (:math:`k_\perp = 0`, including
    the uniform field) the longitudinal projection is degenerate: they are both curl- and divergence-free in the
    perpendicular plane, Eq. (6.6) assigns them no longitudinal part and the Darwin model gives them no inertia (a
    :math:`k_\perp = 0` shear Alfvén wave would have no polarization current, and the net E x B current of a
    non-neutral marker ensemble no equation at all). With ``perp_polarization=True`` the full (transversal)
    polarization of these modes is retained, :math:`\mathbb P_L + \Pi_\perp \mathbb P \Pi_\perp` with
    :math:`\Pi_\perp` the mean of the coefficients of each component over the averaged directions (a symmetric
    projection, so the operator stays symmetric positive semi-definite and the energy
    :math:`\frac12 \mathbf e^\top \mathbb P_L \mathbf e` includes these modes); their shear Alfvén frequency is
    then the :math:`k_\perp \to 0` limit :math:`\omega = k_\parallel / \sqrt{\sum_s \alpha_s^2 n_{0,s}/B_0^2}`.
    """

    def __init__(
        self,
        derham,
        mass_ops,
        P,
        directions: tuple[int, ...],
        stab_eps: float,
        tol: float,
        maxiter: int,
        perp_polarization: bool = True,
    ):
        self._derham = derham
        self._P = P
        self._grad = derham.grad
        M0 = mass_ops.M0
        self._avg = WeightedAverageProjection(derham, M0, directions)
        self._laplace = self._grad.T @ P @ self._grad + stab_eps * (M0 @ self._avg)
        pc = preconditioner.StiffnessPreconditioner(mass_ops, "grad", mid=P, sigma=stab_eps)
        self._tol = tol
        self._inv = inverse(self._laplace, "pcg", pc=pc, tol=tol, maxiter=maxiter)
        self._space = P.domain
        self._v0 = self._grad.domain.zeros()
        self._phi = self._grad.domain.zeros()
        self._v1 = self._space.zeros()
        self._perp_polarization = perp_polarization
        self._u = self._space.zeros()
        comm = derham.comm
        self._comm = comm if (comm is not None and comm.Get_size() > 1) else None
        # the averaged and the kept directions, local interior slices and global sizes of the blocks
        self._directions = tuple(sorted(set(directions)))
        self._kept = tuple(d for d in range(3) if d not in self._directions)
        self._interior = []
        self._n_avg = []
        self._kept_shape = []
        self._kept_slices = []
        for V in self._space.spaces:
            self._interior += [
                tuple(slice(p * m, p * m + e - s + 1) for p, m, s, e in zip(V.pads, V.shifts, V.starts, V.ends))
            ]
            self._n_avg += [float(xp.prod(xp.asarray([V.npts[d] for d in self._directions])))]
            self._kept_shape += [tuple(V.npts[d] for d in self._kept)]
            self._kept_slices += [tuple(slice(V.starts[d], V.ends[d] + 1) for d in self._kept)]

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
    def laplacian(self):
        """The regularized perpendicular Laplacian :math:`\\mathbb G^\\top \\mathbb P \\mathbb G + \\sigma \\mathbb M^0 \\Pi_{\\textnormal{avg}}`."""
        return self._laplace

    @property
    def inverse_laplacian(self):
        """Its inverse (an iterative solver)."""
        return self._inv

    def transpose(self, conjugate=False):
        return self

    def tosparse(self):
        raise NotImplementedError("matrix-free operator")

    def potential(self, v, out=None):
        """The potential :math:`\\phi = (\\mathbb G^\\top \\mathbb P \\mathbb G + \\ldots)^{-1} \\mathbb G^\\top \\mathbb P \\mathbf v` of a 1-form ``v``."""
        self._P.dot(v, out=self._v1)
        self._grad.T.dot(self._v1, out=self._v0)
        return self.solve_laplacian(self._v0, out=out)

    def solve_laplacian(self, rhs, out=None):
        """Solve the regularized perpendicular Laplacian with the tolerance relative to the norm of ``rhs``."""
        # relative tolerance with an absolute floor (a vanishing right-hand side gives the zero solution at once)
        self._inv._options["tol"] = max(self._tol * rhs.inner(rhs) ** 0.5, 1e-150)
        return self._inv.solve(rhs, out=out)

    def remove_average(self, phi):
        r"""Remove (in place) the part of the 0-form ``phi`` that is constant in the averaged directions, i.e.
        :math:`\phi \leftarrow (\mathbb I - \Pi_{\textnormal{avg}}) \phi`: the kernel of the perpendicular
        Laplacian, whose gradient carries no longitudinal displacement. Returns the norm of the removed part."""
        self._avg.dot(phi, out=self._v0)
        phi -= self._v0
        return self._v0.inner(self._v0) ** 0.5

    def perpendicular_average(self, v, out=None):
        r"""The projection :math:`\Pi_\perp \mathbf v` on the modes constant in the averaged directions: each
        component is replaced by the mean of its coefficients over these directions (an orthogonal projection)."""
        if out is None:
            out = self._space.zeros()
        for b, ob, sl, n, shape, ksl in zip(
            v.blocks, out.blocks, self._interior, self._n_avg, self._kept_shape, self._kept_slices
        ):
            partial = xp.sum(b._data[sl], axis=self._directions)
            # the sums over the averaged directions on the global grid of the kept directions
            glob = xp.zeros(shape)
            glob[ksl] = partial
            if self._comm is not None:
                self._comm.Allreduce(MPI.IN_PLACE, glob, op=MPI.SUM)
            glob /= n
            mean = glob[ksl]
            for d in self._directions:
                mean = xp.expand_dims(mean, d)
            ob._data[sl] = mean
        out.update_ghost_regions()
        return out

    def longitudinal_part(self, v, out=None):
        r"""The longitudinal part :math:`\Pi_L \mathbf v = \mathbb G \phi(\mathbf v)` of a 1-form (the gradient of its
        potential, :math:`\mathbb P \Pi_L = \mathbb P_L` without the retained :math:`k_\perp = 0` polarization)."""
        phi = self.potential(v, out=self._phi)
        return self._grad.dot(phi, out=out)

    def longitudinal_transpose(self, w, out=None):
        r"""The transpose :math:`\Pi_L^\top \mathbf w = \mathbb P \mathbb G (\ldots)^{-1} \mathbb G^\top \mathbf w`
        on a dual vector (a current)."""
        self._grad.T.dot(w, out=self._v0)
        phi = self.solve_laplacian(self._v0, out=self._phi)
        self._grad.dot(phi, out=self._v1)
        if out is None:
            out = self._space.zeros()
        return self._P.dot(self._v1, out=out)

    def dot(self, v, out=None):
        assert v.space == self._space
        phi = self.potential(v, out=self._phi)
        self._grad.dot(phi, out=self._v1)
        if out is None:
            out = self._space.zeros()
        self._P.dot(self._v1, out=out)
        if self._perp_polarization:
            # Pi_perp P Pi_perp v
            self.perpendicular_average(v, out=self._u)
            self._P.dot(self._u, out=self._v1)
            self.perpendicular_average(self._v1, out=self._u)
            out += self._u
        return out


class _LongitudinalDriftOperator(LinearOperator):
    r"""The E x B operator acting on the longitudinal field only, :math:`\Pi_L^\top \mathbb Q \Pi_L` (anti-symmetric)."""

    def __init__(self, PL: DarwinLongitudinalOperator, Q):
        self._PL = PL
        self._Q = Q
        self._space = PL.domain
        self._tmp1 = self._space.zeros()
        self._tmp2 = self._space.zeros()

    @property
    def domain(self):
        return self._space

    @property
    def codomain(self):
        return self._space

    @property
    def dtype(self):
        return float

    def transpose(self, conjugate=False):
        return -1.0 * self

    def tosparse(self):
        raise NotImplementedError("matrix-free operator")

    def dot(self, v, out=None):
        self._PL.longitudinal_part(v, out=self._tmp1)
        self._tmp1.update_ghost_regions()
        self._Q.dot(self._tmp1, out=self._tmp2)
        return self._PL.longitudinal_transpose(self._tmp2, out=out)


class DarwinFieldCoupling(Propagator, _PerturbedFieldMixin):
    r"""The quasi-neutral gyrokinetic Darwin model (Sec. 6 of the paper) in the gauge-invariant form, coupled to
    the gyrocentres of all species: find :math:`\mathbf E \in H(\textnormal{curl})`, :math:`\mathbf B \in H(\textnormal{div})`
    and :math:`f_s` such that

    .. math::

        \frac{\partial \mathbf D^{\textnormal{Dar}}}{\partial t} &= \nabla \times \left(\mathbf B - \frac{p_{0,\parallel} - p_{0,\perp}}{B_0^2} \mathbf B_\perp
        + \sum_s \alpha_s^2 \int f_s\, \mu\, \mathbf b_0 \right) - \sum_s \frac{\alpha_s^2}{\varepsilon_s} \int f_s\, \dot{\mathbf R}\,,
        \qquad \mathbf D^{\textnormal{Dar}} = \Pi_{L,\perp}\left(\sum_s \alpha_s^2 \frac{n_{0,s}}{B_0^2} \mathbf E_\perp\right)\,,
        \\
        \frac{\partial \mathbf B}{\partial t} &= - \nabla \times \mathbf E\,,

    with :math:`\mathbf B = \mathbf B_0 + \tilde{\mathbf B}` (``background_current``) and the gyro-averaged
    currents of the gyrocentre motion (parallel streaming along :math:`\mathbf B^*`, E x B and grad-B drifts) of
    :class:`~struphy.propagators.gyrokinetic_maxwell.GyroEparCoupling` and
    :class:`~struphy.propagators.gyrokinetic_maxwell.GyroDriftCoupling`. The quasi-neutrality
    :math:`\nabla \cdot \mathbf D^{\textnormal{Dar}} = \rho_f` is the divergence of the first equation, and the
    transversal part of the first equation (no displacement current) is the constraint of the
    differential-algebraic system: the fields are the gauge-invariant :math:`\mathbf E = -\nabla \phi - \partial_t \mathbf A`
    and :math:`\tilde{\mathbf B} = \nabla \times \mathbf A` of Eqs. (6.15)-(6.16), without the potentials.

    In matrix form, with the longitudinal polarization operator :class:`DarwinLongitudinalOperator` :math:`\mathbb P_L`
    in place of the dielectric matrix :math:`\mathbb A` of :class:`~struphy.propagators.gyrokinetic_maxwell.MaxwellGyroMagnetisation`:

    .. math::

        \mathbb P_L \dot{\mathbf e} = \mathbb C^\top \mathbb M^2_H (\mathbf b + \mathbf b_0) + \mathbb C^\top (\mathcal P^b)^\top \sum_s \alpha_s^2 \mathbf m_s
        - \sum_s \frac{\alpha_s^2}{\varepsilon_s} \mathbf j_s(v_\parallel) - \sum_s \mathbb Q_s \mathbf e - \mathbf j_{\nabla B}\,,
        \qquad \dot{\mathbf b} = -\mathbb C \mathbf e\,,
        \qquad \dot v_{\parallel,p} = \frac{1}{\varepsilon_s} \mathbf c_p \cdot \langle \vec \Lambda^{1\top} \rangle_p \mathbf e\,,

    discretized with the implicit mid-point rule in :math:`(\mathbf e, \mathbf b, v_\parallel)`, all couplings included
    (:math:`\mathbf c_p` frozen at :math:`v_\parallel^n`, the markers at :math:`\boldsymbol \eta^n`):

    .. math::

        \left(\mathbb P_L + \frac{\Delta t^2}{4} (\mathbb K + \mathbb S) + \frac{\Delta t}{2} \mathbb Q\right) \mathbf e^{n+1}
        = \left(\mathbb P_L - \frac{\Delta t^2}{4} (\mathbb K + \mathbb S) - \frac{\Delta t}{2} \mathbb Q\right) \mathbf e^{n}
        + \Delta t \left(\mathbb C^\top \mathbb M^2_H (\mathbf b^n + \mathbf b_0) + \mathbf s_m - \mathbf j^n_\parallel - \mathbf j_{\nabla B}\right)\,,

    with :math:`\mathbb K = \mathbb C^\top \mathbb M^2_H \mathbb C`, :math:`\mathbb S = \sum_s \frac{\alpha_s^2}{\varepsilon_s^2} \mathbb S_s`,
    :math:`\mathbb Q = \sum_s \mathbb Q_s` (stencil matrices in the ZLR limit, matrix-free with finite Larmor radius) and
    the magnetisation source :math:`\mathbf s_m`; then :math:`\mathbf b^{n+1} = \mathbf b^n - \Delta t\, \mathbb C \bar{\mathbf e}`,
    the parallel velocities are pushed with :math:`\bar{\mathbf e} = (\mathbf e^n + \mathbf e^{n+1})/2`,
    the drifts with an explicit Runge-Kutta scheme in :math:`\mathbf b^n` and the longitudinal field
    :math:`\Pi_L \mathbf e^n = -\mathbb G \boldsymbol \phi^n` before the field solve (the couplings are accumulated
    at the new positions). The E x B operator and the grad-B current act on the longitudinal field as well,
    :math:`\mathbb Q \to \Pi_L^\top \mathbb Q \Pi_L` and :math:`\mathbf j_{\nabla B} \to \Pi_L^\top \mathbf j_{\nabla B}`:
    the transversal part of the Darwin field is the multiplier of the quasi-static transversal Ampère law, not a
    dynamical field, and does not drive the drifts (as in the :math:`A_\parallel` gyrokinetic models, where the
    E x B drift is electrostatic). The operator is positive definite
    (longitudinal part from :math:`\mathbb P_L`, transversal part from :math:`\mathbb K + \mathbb S`), the system is solved
    with ``pbicgstab`` (``pcg`` without drifts) preconditioned by the weighted mass matrix of the polarization
    plus the continuum limit of the skin term. The mid-point rule conserves the Darwin energy
    :math:`\frac12 \mathbf e^\top \mathbb P_L \mathbf e + \frac12 \mathbf b^\top \mathbb M^2_H \mathbf b + \sum_s \alpha_s^2 \sum_p w_p (v_\parallel^2/2 + \mu B_\parallel)`
    exactly for the parallel and E x B couplings (the grad-B current exchange is second order).
    """

    class Variables:
        """Container for the variables advanced by :class:`DarwinFieldCoupling`.

        Attributes
        ----------
        e : FEECVariable
            Electric field in H(curl) space.
        b : FEECVariable
            Magnetic perturbation in H(div) space.
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
        polarization=0.0,
        background_current: bool = False,
        gyro: GyroAverage = None,
        with_drifts: bool = True,
    ):
        """
        Parameters
        ----------
        species : tuple[PICVariable]
            The gyrocentre species (``"Particles5D"``), whose currents and magnetisation enter the field equations
            and whose parallel velocities and drifts are pushed.
        background : BackgroundWeights, default=None
            The background moments of the species; ``None`` for the polarization ``polarization`` times the fluid
            equilibrium density.
        polarization : float | Callable
            The polarization with ``background=None``.
        background_current : bool
            Whether :math:`\\nabla \\times \\mathbf B_0` enters Ampère's law.
        gyro : GyroAverage, default=None
            The gyro-averaging (``None``: zero-Larmor-radius limit).
        with_drifts : bool
            Whether the E x B and grad-B drifts and their currents are included.
        """
        self.variables = self.Variables()
        for var in species:
            assert isinstance(var, PICVariable)
            assert var.space == "Particles5D"
        self.species = tuple(species)
        self.background = background
        self.polarization = polarization
        self.background_current = background_current
        self.gyro = gyro if gyro is not None else GyroAverage()
        self.with_drifts = with_drifts

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`DarwinFieldCoupling`.

        Parameters
        ----------
        solver : str, default=None
            Iterative solver: a general one (``LiteralOptions.OptsGenSolver``) solves the full system including the
            anti-symmetric E x B operator, a symmetric one (``LiteralOptions.OptsSymmSolver``) solves the symmetric
            part with Picard iterations for the E x B operator (``picard_maxiter``, ``picard_tol``; they converge
            only for small time steps). ``None``: ``"pbicgstab"`` with drifts, ``"pcg"`` without.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner, built from the weighted mass matrix of the polarization plus the continuum limit
            of the skin term (:func:`~struphy.propagators.gyrokinetic_maxwell.skin_preconditioner_matrix`).

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to ``SolverParameters()``. The tolerance is
            relative to the norm of the right-hand side (the system has a kernel, the uniform perpendicular
            field, and its scale is set by the skin term).

        inner_tol : float, default=1e-12
            Tolerance of the inner solves of the perpendicular Laplacian (relative to the right-hand side).

        inner_maxiter : int, default=1000
            Maximum iterations of the inner solves.

        kernel_average : str, default="eta1 eta2"
            The directions of the average projection that regularizes the kernel of the perpendicular Laplacian
            (the directions perpendicular to the background field; ``""`` for the global mean only).

        stab_eps : float, default=1.0
            The regularization :math:`\\sigma` of that kernel.

        perp_polarization : bool, default=True
            Whether the (transversal) polarization of the fields constant in the averaged directions
            (:math:`k_\\perp = 0`, where the longitudinal projection is degenerate and the Darwin model has no
            inertia) is retained, see :class:`DarwinLongitudinalOperator`.

        picard_maxiter : int, default=20
            Maximum number of Picard iterations for the E x B operator (symmetric solvers only).

        picard_tol : float, default=1e-10
            Relative tolerance of the Picard iterations.

        butcher : ButcherTableau, default=None
            Butcher tableau of the explicit drift push, default ``rk2``.

        mpi_sort : LiteralOptions.OptsMPIsort, default="last"
            MPI sorting policy of the drift push.
        """

        solver: str = None
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        inner_tol: float = 1e-12
        inner_maxiter: int = 1000
        kernel_average: str = "eta1 eta2"
        stab_eps: float = 1.0
        perp_polarization: bool = True
        picard_maxiter: int = 20
        picard_tol: float = 1e-10
        butcher: ButcherTableau = None
        mpi_sort: LiteralOptions.OptsMPIsort = "last"

        def __post_init__(self):
            if self.solver is not None:
                assert self.solver in LiteralOptions.OptsSymmSolver.__args__ + LiteralOptions.OptsGenSolver.__args__, (
                    f"Unknown solver {self.solver!r}."
                )
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
            check_option(self.mpi_sort, LiteralOptions.OptsMPIsort)
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
        curl = self.derham.curl

        polarization = self.background.polarization if self.background is not None else self.polarization
        self._P = polarization_matrix(self.mass_ops, polarization)
        aniso = None
        if self.background is not None and self.background.has_anisotropy:
            aniso = self.background.anisotropy
        self._M2H = magnetizing_matrix(self.mass_ops, aniso)
        self._CTMC = curl.T @ self._M2H @ curl
        self._curlT = curl.T

        directions = preconditioner._parse_directions(self.options.kernel_average)
        if not directions:
            directions = (0, 1, 2)
        self._PL = DarwinLongitudinalOperator(
            self.derham,
            self.mass_ops,
            self._P,
            directions,
            self.options.stab_eps,
            self.options.inner_tol,
            self.options.inner_maxiter,
            perp_polarization=self.options.perp_polarization,
        )

        # perturbed field, couplings and magnetisation
        self._allocate_b_tilde(self.variables.b)
        e_space = self.variables.e.spline.vector.space
        self._e_mid_tp = self.derham.V1.zeros()
        self._parallel = []
        self._drifts = []
        self._skin_scales = []
        self._accums = []
        self._accum_args = []
        for var in self.species:
            params = var.species.equation_params
            self._parallel += [_ParallelCoupling(self, var, self.gyro, self._e_mid_tp)]
            self._skin_scales += [params.alpha**2 / params.epsilon**2]
            if self.with_drifts:
                self._drifts += [
                    _DriftCoupling(self, var, self.gyro, self._e_mid_tp, self.options.butcher, self.options.mpi_sort)
                ]
            alpha2 = params.alpha**2
            if self.gyro.flr:
                kernel, args = gk_mag_density_gyro, (alpha2, *self._gyro_args(self.gyro, params.epsilon, disc=True))
            else:
                kernel, args = gc_mag_density_0form, (alpha2,)
            self._accums += [AccumulatorVector(var.particles, "H1", kernel, self.mass_ops, self.domain.args_domain)]
            self._accum_args += [args]
        self._PBT = self.basis_ops.PB.T

        # the constant background current C^T M2H b_0
        if self.background_current:
            self._s0 = self._curlT.dot(self._M2H.dot(self.projected_equil.b2))
        else:
            self._s0 = None

        # buffers
        self._m_pol = self._PBT.domain.zeros()
        self._s2 = self._PBT.codomain.zeros()
        self._s1 = e_space.zeros()
        self._rhs = e_space.zeros()
        self._tmp = e_space.zeros()
        self._e_new = e_space.zeros()
        self._e_picard = e_space.zeros()
        self._e_L = e_space.zeros()
        self._rhs2 = e_space.zeros()
        self._e_mid = e_space.zeros()
        self._b_new = self.variables.b.spline.vector.space.zeros()

        # the solver (the operator and the preconditioner are set at each call, they depend on dt)
        self._polarization_weight = polarization
        self._A = dielectric_matrix(self.mass_ops, polarization)
        self._pc_dt = None
        solver = self.options.solver
        if solver is None:
            solver = "pbicgstab" if self.with_drifts else "pcg"
        self._solver = solver
        # a symmetric solver treats the E x B operator by Picard iterations
        self._picard = self.with_drifts and solver in LiteralOptions.OptsSymmSolver.__args__
        self._A_inv = inverse(
            self._A,
            solver,
            pc=self._preconditioner(0.0),
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
        )

    def _preconditioner(self, dt: float):
        """The mass-matrix preconditioner of the implicit system at time step ``dt``: the polarization plus the
        continuum limit of the skin term (``skin_preconditioner_matrix``), or the dielectric matrix without
        species backgrounds."""
        if self.options.precond is None:
            return None
        pc_class = getattr(preconditioner, self.options.precond)
        if self.background is None or dt == 0.0:
            return pc_class(self._A)
        M = skin_preconditioner_matrix(
            self.mass_ops, self._polarization_weight, self.background.skin, dt**2 / 4.0, vacuum=False
        )
        return pc_class(M)

    @property
    def longitudinal_operator(self) -> DarwinLongitudinalOperator:
        """The longitudinal polarization operator (available after :meth:`allocate`)."""
        return self._PL

    @profile
    def __call__(self, dt):
        en = self.variables.e.spline.vector
        bn = self.variables.b.spline.vector

        # perturbed magnetic field and particle couplings at eta^n, v^n
        self._update_b_tilde()

        # the drifts are pushed first, explicitly in the fields (e^n, b^n): the E x B motion does no work and the
        # grad-B work mu (B(eta^{n+1}) - B(eta^n)) takes place in the frozen field b^n, which the explicit grad-B
        # current (evaluated below at eta^{n+1}, in the same field) balances to second order. The magnetisation
        # and the parallel coupling are then accumulated at eta^{n+1}, where the energies are evaluated, so that
        # the magnetisation and parallel exchanges remain exact.
        # The drifts see the longitudinal (electrostatic) field E_L = -grad phi only: the transversal part of the
        # Darwin field is the multiplier of the quasi-static transversal Ampère law (it scales like 1/dt in the
        # discrete constraint) and drives no E x B motion, as in the A_parallel gyrokinetic models.
        if self._drifts:
            self._PL.longitudinal_part(en, out=self._e_L)
            e_tp = self._E1T.dot(self._e_L, out=self._e_mid_tp)
            e_tp.update_ghost_regions()
            for coupling in self._drifts:
                coupling.pusher(dt)
                if coupling.var.species.weights_params.control_variate:
                    coupling.var.particles.update_weights()

        for coupling in self._parallel + self._drifts:
            coupling.accumulate()

        # left- and right-hand side operators of the mid-point rule (symmetric part)
        lhs = self._PL + dt**2 / 4.0 * self._CTMC
        rhs_op = self._PL - dt**2 / 4.0 * self._CTMC
        for coupling, scale in zip(self._parallel, self._skin_scales):
            S = scale * coupling.operator
            lhs = lhs + dt**2 / 4.0 * S
            rhs_op = rhs_op - dt**2 / 4.0 * S
        Q = None
        for coupling in self._drifts:
            Q = coupling.operator if Q is None else Q + coupling.operator
        if Q is not None:
            Q = _LongitudinalDriftOperator(self._PL, Q)

        rhs = rhs_op.dot(en, out=self._rhs)

        # dt C^T M2H (b^n + b_0)
        self._M2H.dot(bn, out=self._s2)
        self._curlT.dot(self._s2, out=self._s1)
        if self._s0 is not None:
            self._s1 += self._s0
        self._s1 *= dt
        rhs += self._s1

        # magnetisation source: dt C^T PB^T sum_s alpha_s^2 m_s
        self._m_pol *= 0.0
        for accum, args in zip(self._accums, self._accum_args):
            accum(*args)
            self._m_pol += accum.vectors[0]
        self._PBT.dot(self._m_pol, out=self._s2)
        self._curlT.dot(self._s2, out=self._s1)
        self._s1 *= dt
        rhs += self._s1

        # - dt (parallel currents at v^n and grad-B currents) - dt/2 Q e^n
        self._tmp *= 0.0
        for coupling, scale in zip(self._parallel, self._skin_scales):
            self._s1 *= 0.0
            self._s1 += coupling.vector
            self._s1 *= scale * coupling.epsilon
            self._tmp += self._s1
        if self._drifts:
            self._s1 *= 0.0
            for coupling in self._drifts:
                self._s1 += coupling.vector
            self._PL.longitudinal_transpose(self._s1, out=self._e_L)
            self._tmp += self._e_L
        self._tmp *= dt
        rhs -= self._tmp
        if Q is not None:
            Q.dot(en, out=self._tmp)
            self._tmp *= dt / 2.0
            rhs -= self._tmp

        # solve the full system, or the symmetric part with Picard iterations for the E x B operator:
        # lhs e = rhs - dt/2 Q e
        if Q is not None and not self._picard:
            lhs = lhs + dt / 2.0 * Q
        if self._pc_dt != dt:
            pc = self._preconditioner(dt)
            if pc is not None:
                self._A_inv._options["pc"] = pc
            self._pc_dt = dt
        self._A_inv.linop = lhs
        self._A_inv._options["tol"] = max(self.options.solver_params.tol * rhs.inner(rhs) ** 0.5, 1e-150)
        en1 = self._e_new
        en1 *= 0.0
        en1 += en
        n_picard = 0
        with ProfileManager.profile_region(self._solve_region, functions=[self._A_inv.solve]):
            while True:
                self._rhs2 *= 0.0
                self._rhs2 += rhs
                if self._picard:
                    Q.dot(en1, out=self._tmp)
                    self._tmp *= dt / 2.0
                    self._rhs2 -= self._tmp
                self._A_inv.solve(self._rhs2, out=self._e_picard)
                n_picard += 1
                self._tmp *= 0.0
                self._tmp += self._e_picard
                self._tmp -= en1
                diff = self._tmp.inner(self._tmp) ** 0.5
                norm = max(self._e_picard.inner(self._e_picard) ** 0.5, 1e-300)
                en1 *= 0.0
                en1 += self._e_picard
                if (
                    not self._picard
                    or diff <= self.options.picard_tol * norm
                    or n_picard >= self.options.picard_maxiter
                ):
                    break
        info = dict(self._A_inv._info)
        info["picard"] = n_picard

        # mid-point field (tensor-product coefficients for the kernels) and new b coeffs
        e_mid = en.copy(out=self._e_mid)
        e_mid += en1
        e_mid *= 0.5
        e_mid_tp = self._E1T.dot(e_mid, out=self._e_mid_tp)
        e_mid_tp.update_ghost_regions()

        bn1 = self.derham.curl.dot(e_mid, out=self._b_new)
        bn1 *= -dt
        bn1 += bn

        # parallel velocities in the mid-point field
        for coupling in self._parallel:
            coupling.pusher(dt)
            if coupling.var.species.weights_params.control_variate:
                coupling.var.particles.update_weights()

        diffs = self.update_feec_variables(e=en1, b=bn1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for DarwinFieldCoupling: {info['success']}")
            logger.info(f"Iterations for DarwinFieldCoupling: {info['niter']} (Picard: {info['picard']})")
            logger.info(f"Maxdiff e  for DarwinFieldCoupling: {diffs['e']}")
            logger.info(f"Maxdiff b  for DarwinFieldCoupling: {diffs['b']}")
            logger.info("")
