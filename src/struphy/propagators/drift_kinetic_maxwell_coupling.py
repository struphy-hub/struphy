"Particle and FEEC variables are updated."

import logging
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from feectools.linalg.basic import LinearOperator
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI, SerialComm

from struphy.feec import preconditioner
from struphy.feec.mass import WeightedAverageProjection
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation.kernels.dk_maxwell import dk_maxwell
from struphy.pic.accumulation.kernels.dk_maxwell_mf import dk_maxwell_mf
from struphy.pic.accumulation.kernels.marker_operator_asym_dot import marker_operator_asym_dot
from struphy.pic.accumulation.kernels.marker_operator_rank1_dot import marker_operator_rank1_dot
from struphy.pic.accumulation.particles_to_grid import Accumulator, AccumulatorVector, MarkerOperator
from struphy.pic.pushing.kernels.push_vpar_with_efield import push_vpar_with_efield
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator
from struphy.propagators.utils import extracted_ones
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class _PerpProjection:
    r"""The :math:`\mathbb P`-orthogonal projection onto the 1-forms that are constant in the given directions.

    .. math::

        \mathbb P_W = \mathbb E \, (\mathbb E^\top \mathbb P \mathbb E)^+ \, \mathbb E^\top \mathbb P\,,

    where :math:`\mathbb E` embeds the coefficients that are constant in the averaged directions (all three
    components of the 1-form together, so that the off-diagonal blocks of :math:`\mathbb P` are included). The
    reduced matrix :math:`\mathbb E^\top \mathbb P \mathbb E` is assembled once by probing :math:`\mathbb P` with
    colored vectors and pseudo-inverted on every process (it is singular where :math:`\mathbb P` has a kernel, e.g.
    the component parallel to :math:`\mathbf b_0`). The projection itself is a reduced matrix-vector product.

    This generalizes :class:`~struphy.feec.mass.WeightedAverageProjection` to the components of a 1-form, whose
    spline spaces are not those of the 0-forms. The reduced matrix has size ``3 * prod(npts[kept])``; its dense
    pseudo-inverse is computed once on every rank (memory quadratic and compute cubic in this size; sizes above
    4096 are rejected), and one application costs one application of :math:`\mathbb P` and a reduced
    matrix-vector product.
    """

    def __init__(self, space, P, directions: tuple[int, ...], comm):
        import numpy as np
        from scipy import sparse

        directions = tuple(sorted(set(directions)))
        assert directions and all(d in (0, 1, 2) for d in directions)
        kept = tuple(d for d in range(3) if d not in directions)

        self._space = space
        self._P = P
        self._directions = directions
        self._kept = kept
        self._comm = None if isinstance(comm, (SerialComm, type(None))) else comm
        if self._comm is not None and self._comm.Get_size() == 1:
            self._comm = None

        # layout of each component; the components must share the kept directions
        self._blocks = []
        self._red_shape = None
        self._kept_bounds = None
        for V in space.spaces:
            red_shape = tuple(V.npts[d] for d in kept)
            kept_bounds = tuple((V.starts[d], V.ends[d]) for d in kept)
            if self._red_shape is None:
                self._red_shape = red_shape
                self._kept_bounds = kept_bounds
            assert red_shape == self._red_shape, (
                "DarwinLongitudinalOperator: the components of the 1-form must have the same number of spline "
                f"coefficients in the kept directions, got {red_shape} and {self._red_shape}; periodic boundaries "
                "in the kept directions are required."
            )
            assert kept_bounds == self._kept_bounds, (
                "DarwinLongitudinalOperator: the components of the 1-form must have the same decomposition in the "
                f"kept directions, got {kept_bounds} and {self._kept_bounds}."
            )
            assert all(V.periods[d] == space.spaces[0].periods[d] for d in kept), (
                "DarwinLongitudinalOperator: all components of the 1-form must have the same periodicity in the "
                "kept directions."
            )
            local = tuple(slice(p * m, p * m + e - s + 1) for p, m, s, e in zip(V.pads, V.shifts, V.starts, V.ends))
            red_local = tuple(slice(V.starts[d], V.ends[d] + 1) for d in kept)
            self._blocks += [(V, local, red_local)]
        self._n = int(np.prod(self._red_shape)) if kept else 1
        self._n_red = 3 * self._n
        if self._n_red > 4096:
            raise NotImplementedError(
                "Darwin polarization: the retained-polarization reduced matrix has size "
                f"{self._n_red} (3 * the number of spline coefficients in the kept directions); the dense "
                "pseudo-inverse is limited to 4096 (memory O(n^2), compute O(n^3), on every rank). Use a coarser "
                "grid in the kept direction."
            )
        self._full_in = space.zeros()
        self._full_out = space.zeros()

        # colored probing in the kept directions: the columns of one color are at least 2 p + 1 apart (also
        # across a periodic boundary), so their mass-matrix bands do not overlap
        V0 = self._blocks[0][0]
        strides = []
        for d in kept:
            n, p = V0.npts[d], V0.pads[d]
            stride = n
            if not V0.periods[d]:
                stride = min(n, 2 * p + 1)
            else:
                for s in range(2 * p + 1, n + 1):
                    if n % s == 0:
                        stride = s
                        break
            strides.append(stride)
        band = {d: max(V.pads[d] for V, _, _ in self._blocks) for d in kept}

        index = np.arange(self._n).reshape(self._red_shape)
        rows, cols, vals = [], [], []
        for i in range(3):
            for color in np.ndindex(*strides):
                probe = np.zeros(self._red_shape)
                sel = tuple(slice(c, None, s) for c, s in zip(color, strides))
                probe[sel] = 1.0
                self._full_in *= 0.0
                self._E(i, probe, self._full_in.blocks[i])
                self._full_in.update_ghost_regions()
                self._P.dot(self._full_in, out=self._full_out)
                columns = [self._ET(j, self._full_out.blocks[j]) for j in range(3)]
                for a in index[sel].ravel():
                    a_idx = np.unravel_index(a, self._red_shape)
                    ranges = []
                    for k, d in enumerate(kept):
                        n = V0.npts[d]
                        p = band[d]
                        r = np.arange(a_idx[k] - p, a_idx[k] + p + 1)
                        r = r % n if V0.periods[d] else r[(r >= 0) & (r < n)]
                        ranges.append(np.unique(r))
                    for b_idx in np.ndindex(*[len(r) for r in ranges]):
                        b = tuple(r[i] for r, i in zip(ranges, b_idx))
                        bi = index[b]
                        for j in range(3):
                            rows.append(j * self._n + bi)
                            cols.append(i * self._n + a)
                            vals.append(columns[j][b])
        K = sparse.coo_matrix((vals, (rows, cols)), shape=(self._n_red, self._n_red)).tocsc()
        K.sum_duplicates()
        K = np.asarray(K.toarray())
        self._K_pinv = np.linalg.pinv(K, rcond=1e-12, hermitian=True)

    def _ET(self, i, v):
        """E^T v for component ``i``: sum over the averaged directions, as global array over the kept directions."""
        import numpy as np

        _, local, red_local = self._blocks[i]
        loc = xp.to_numpy(v._data[local]).sum(axis=self._directions)
        out = np.zeros(self._red_shape, dtype=loc.dtype)
        out[red_local] = loc
        if self._comm is not None:
            out = self._comm.allreduce(out, op=MPI.SUM)
        return out

    def _E(self, i, red, out):
        """E red for component ``i``: the global array over the kept directions, constant along the averaged ones."""
        _, local, red_local = self._blocks[i]
        red = xp.asarray(red)
        if self._kept:
            loc = red[red_local]
            shape = [1, 1, 1]
            for k, d in enumerate(self._kept):
                shape[d] = loc.shape[k]
            out._data[local] = xp.asarray(loc.reshape(shape))
        else:
            out._data[local] = red.reshape(())

    def ET(self, v):
        """E^T v, concatenated over the three components (a global array on all processes)."""
        import numpy as np

        return np.concatenate([np.atleast_1d(self._ET(i, v.blocks[i])) for i in range(3)]).ravel()

    def E(self, red, out):
        """E red into the 1-form ``out`` (ghost regions updated)."""
        import numpy as np

        red = np.asarray(red).reshape((3,) + self._red_shape)
        for i in range(3):
            self._E(i, red[i], out.blocks[i])
        out.update_ghost_regions()
        return out

    def project(self, v, out=None):
        r"""The :math:`\mathbb P`-orthogonal projection :math:`\mathbb P_W \mathbf v` (a primal 1-form)."""
        if out is None:
            out = self._space.zeros()
        self._P.dot(v, out=self._full_out)
        return self.E(self._K_pinv @ self.ET(self._full_out), out)


class DarwinLongitudinalOperator(LinearOperator):
    r"""The Darwin longitudinal polarization operator on 1-forms,

    .. math::

        \mathbb P_L = \mathbb P \Pi_G
        + (\mathbb I - \Pi_G)^\top \mathbb P \mathbb P_W (\mathbb I - \Pi_G)\,,
        \qquad
        \Pi_G = \mathbb G \, \mathbb L^{-1} \mathbb G^\top \mathbb P\,,
        \qquad
        \mathbb L = \mathbb G^\top \mathbb P \mathbb G + \sigma\, \mathbb M^0 \Pi_{\textnormal{avg}}\,,

    with :math:`\mathbb P = \mathbb M^1_\perp` (:attr:`~struphy.feec.mass.WeightedMassOperators.M1gyro`, the
    mass matrix weighted with :math:`n_0/|B_0|^2` on the perpendicular components),
    :math:`\mathbb G` the discrete gradient, :math:`\Pi_{\textnormal{avg}}` the :math:`\mathbb M^0`-weighted
    average over :math:`\eta_1, \eta_2` and
    :math:`\mathbb P_W = \mathbb E (\mathbb E^\top \mathbb P \mathbb E)^+ \mathbb E^\top \mathbb P` the
    :math:`\mathbb P`-orthogonal projection on the 1-forms that are constant in these directions
    (:class:`_PerpProjection`; :math:`\mathbb E` embeds all three components). The first term is the
    :math:`\mathbb P`-orthogonal projection of the electric field on the :math:`\mathbb P`-visible perpendicular
    gradients (gradients parallel to :math:`\mathbf b_0` are annihilated), the second the polarization of the
    :math:`\mathbb P`-orthogonal complement of that gradient part within the modes constant in the averaged
    directions. Unlike the naive sum :math:`\mathbb P \Pi_G + \mathbb P_W`, the two ranges are not double-counted
    when the weight or the metric varies in :math:`\eta_1, \eta_2`; the operator is symmetric and positive
    semi-definite by construction (each term is of the form :math:`\mathbb A^\top \mathbb P \mathbb A` with the
    :math:`\mathbb P`-self-adjoint projections :math:`\Pi_G` and :math:`\mathbb I-\Pi_G`; :math:`\Pi_G` acts as the
    identity on the :math:`\mathbb P`-visible gradients and annihilates the kernel). With a weight or metric
    varying in the averaged directions this is a :math:`\mathbb P`-orthogonal (not the Euclidean) decomposition:
    fields that are transverse in the Euclidean sense can have a small longitudinal :math:`\mathbb P`-component,
    and :math:`\mathbb P_L \mathbf u \neq \mathbb P \mathbf u` for :math:`k_\perp = 0` fields that are not
    :math:`\mathbb P`-orthogonal to gradients; :math:`\mathbb P_L \mathbb G = \mathbb P \mathbb G` holds for any
    weight and metric.

    This is the matrix-free weak form of the Darwin displacement
    :math:`\mathbf D^\textnormal{Dar} = \Pi_{L,\perp}(\chi\, n_0/|B_0|^2 \mathbf E_\perp)` of Remmerswaal, Hatzky
    and Sonnendrücker (J. Plasma Phys. 91, E127, 2025, Eq. (6.6)): only the perpendicular-gradient part
    :math:`\mathbf E_L = -\nabla_\perp \phi` of :math:`\mathbf E_\perp` carries the longitudinal polarization. The
    gradient part is applied with an inner preconditioned conjugate-gradient solve of :math:`\mathbb L`; its
    tolerance is relative to the norm of the right-hand side.

    **Kernel of the perpendicular Laplacian.** :math:`\mathbb G^\top \mathbb P \mathbb G` is singular on the 0-forms
    that are constant in the averaged directions (:math:`k_\perp = 0`): their gradient is parallel to
    :math:`\mathbf b_0` (or vanishes), so it carries no longitudinal polarization. The term
    :math:`\sigma\, \mathbb M^0 \Pi_{\textnormal{avg}}` (:math:`\sigma =` ``stab_eps``) regularizes that kernel, and
    the second term retains the transversal polarization of the :math:`\mathbb P`-complement of the gradient part
    in these modes. This is the choice of the reference implementation on branch
    ``remmerswaal-gyrokinetic-maxwell-zlr``: the modes have the polarization of the :math:`k_\perp \to 0` limit
    instead of no inertia (with no inertia, a :math:`k_\perp = 0` shear Alfvén wave and the net
    :math:`E \times B` current of the markers would have no equation in the quasi-neutral Darwin model). For a
    background field along :math:`\eta_3` the average over :math:`\eta_1, \eta_2` is exactly the kernel of
    :math:`\mathbb G^\top \mathbb P \mathbb G`. The regularized operator :math:`\Pi_G` is the
    :math:`\mathbb P`-orthogonal projector on the :math:`\mathbb P`-visible gradient subspace (modulo the kernel
    directions annihilated by :math:`\mathbb P`; it maps the kernel to zero), and :math:`\mathbb I-\Pi_G` is the
    complementary projector; :math:`\mathbb P_L \mathbb G = \mathbb P \mathbb G` holds off the kernel, and
    :math:`\mathbb P_L = \mathbb P` on the :math:`k_\perp = 0` fields that are :math:`\mathbb P`-orthogonal to
    gradients (exactly for a uniform perpendicular weight and metric). With ``perp_polarization=False`` the second
    term is dropped (the pure pseudo-inverse form, in which the kernel modes have no polarization). The operator is
    symmetric and positive semi-definite; polar splines are not supported (the average over :math:`\eta_1` is not
    implemented), the background field must have no :math:`\eta_1` and :math:`\eta_2` components, and the average
    is fixed to :math:`\eta_1, \eta_2` (all checked in the constructor).

    The energy :math:`\frac{1}{2}\mathbf e^\top \mathbb P_L \mathbf e` is conserved by the implicit mid-point coupling
    step only up to the inner-solve tolerance, approximately :math:`0.04 \times` ``inner_tol`` (measured); the
    default ``inner_tol = 1e-12`` gives a relative drift of about :math:`5\cdot 10^{-14}` for three steps. A larger
    ``inner_tol`` makes the outer solver's recursively computed residual differ from the true residual.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence.
    mass_ops : WeightedMassOperators
        The mass operators, with the equilibrium :math:`n_0` and :math:`\mathbf B_0` of
        :attr:`~struphy.feec.mass.WeightedMassOperators.M1gyro`.
    stab_eps : float
        The regularization :math:`\sigma > 0` of the kernel.
    inner_tol : float
        Relative tolerance of the inner solves of the perpendicular Laplacian (relative to the norm of the
        right-hand side).
    inner_maxiter : int
        Maximum number of iterations of the inner solves.
    perp_polarization : bool
        Whether the polarization of the :math:`\mathbb P`-complement of the gradient part in the modes constant in
        the averaged directions is retained (default True).
    """

    def __init__(
        self,
        derham,
        mass_ops,
        stab_eps: float = 1.0,
        inner_tol: float = 1e-12,
        inner_maxiter: int = 1000,
        perp_polarization: bool = True,
    ):
        assert not derham.polar_splines, (
            "DarwinLongitudinalOperator does not support polar splines: the average over eta1 across the pole is "
            "not implemented. Use polarization='full' or a tensor-product grid."
        )
        assert stab_eps > 0.0, f"stab_eps must be > 0, got {stab_eps}."
        assert inner_tol > 0.0, f"inner_tol must be > 0, got {inner_tol}."
        assert inner_maxiter > 0, f"inner_maxiter must be > 0, got {inner_maxiter}."

        # the kernel average and the retained polarization assume b0 along eta3 (no eta1/eta2 components)
        equil = mass_ops.eq_mhd
        if equil is not None:
            eta = xp.linspace(0.0, 1.0, 5)
            e1, e2, e3 = xp.meshgrid(eta, eta, eta, indexing="ij")
            b1 = xp.max(xp.abs(equil.unit_bv_1(e1, e2, e3)))
            b2 = xp.max(xp.abs(equil.unit_bv_2(e1, e2, e3)))
            assert float(b1) < 1e-12 and float(b2) < 1e-12, (
                "DarwinLongitudinalOperator assumes a background field along eta3 (no eta1/eta2 components): "
                f"got max|b0.eta1|={float(b1):.3e}, max|b0.eta2|={float(b2):.3e}. Use polarization='full'."
            )

        self._derham = derham
        self._P = mass_ops.M1gyro
        self._grad = derham.grad
        self._space = self._P.domain

        # kernel regularization: the average over eta1, eta2 is the kernel for b0 along eta3
        directions = (0, 1)
        self._directions = directions
        M0 = mass_ops.M0
        self._avg = WeightedAverageProjection(derham, M0, directions)
        self._laplace = self._grad.T @ self._P @ self._grad + stab_eps * (M0 @ self._avg)
        pc = preconditioner.StiffnessPreconditioner(mass_ops, "grad", mid=self._P, sigma=stab_eps)
        self._tol = inner_tol
        self._inv = inverse(self._laplace, "pcg", pc=pc, tol=inner_tol, maxiter=inner_maxiter)

        # buffers
        self._phi = self._grad.domain.zeros()
        self._v0 = self._grad.domain.zeros()
        self._v1 = self._space.zeros()
        self._g_primal = self._space.zeros()
        self._g_dual = self._space.zeros()
        self._h = self._space.zeros()
        self._w = self._space.zeros()
        self._Pw = self._space.zeros()
        self._gw = self._space.zeros()

        # projection on the P-orthogonal complement of the gradient part within the constant-in-eta1/eta2 fields
        self._perp_polarization = perp_polarization
        self._perp = None
        if perp_polarization:
            self._perp = _PerpProjection(self._space, self._P, directions, derham.comm)

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
        return self

    def tosparse(self):
        raise NotImplementedError("matrix-free operator")

    def solve_laplacian(self, rhs, out=None):
        """Solve the regularized perpendicular Laplacian with the tolerance relative to the norm of ``rhs``."""
        # absolute tolerance with a floor (a vanishing right-hand side gives the zero solution at once)
        self._inv._options["tol"] = max(self._tol * rhs.inner(rhs) ** 0.5, 1e-150)
        return self._inv.solve(rhs, out=out)

    def _gradient_potential(self, v):
        """The potential :math:`\\phi = \\mathbb L^{-1} \\mathbb G^\\top \\mathbb P \\mathbf v` of the 1-form ``v``."""
        self._P.dot(v, out=self._v1)
        self._grad.T.dot(self._v1, out=self._v0)
        return self.solve_laplacian(self._v0, out=self._phi)

    def _gradient(self, v, out):
        """The gradient part :math:`\\mathbb P \\Pi_G \\mathbf v` into ``out``."""
        phi = self._gradient_potential(v)
        self._grad.dot(phi, out=self._g_primal)
        return self._P.dot(self._g_primal, out=out)

    def remove_average(self, phi):
        r"""Remove (in place) the part of the 0-form ``phi`` that is constant in the averaged directions, i.e.
        :math:`\phi \leftarrow (\mathbb I - \Pi_{\textnormal{avg}}) \phi`: the kernel of the regularized
        perpendicular Laplacian. Returns the norm of the removed part."""
        self._avg.dot(phi, out=self._v0)
        phi -= self._v0
        return self._v0.inner(self._v0) ** 0.5

    def dot(self, v, out=None):
        assert v.space == self._space
        if out is None:
            out = self._space.zeros()

        # gradient part P Pi_G v = P G phi, and the primal Pi_G v = G phi for the complement below
        phi = self._gradient_potential(v)
        self._grad.dot(phi, out=self._g_primal)
        self._P.dot(self._g_primal, out=self._g_dual)

        if self._perp_polarization:
            # (I - Pi_G) v
            v.copy(out=self._h)
            self._h -= self._g_primal
            # (I - Pi_G)^T P P_W (I - Pi_G) v = P w - P Pi_G w, with w = P_W (I - Pi_G) v
            w = self._perp.project(self._h, out=self._w)
            self._P.dot(w, out=self._Pw)
            self._gradient(w, out=self._gw)
            self._Pw -= self._gw
            self._g_dual += self._Pw

        self._g_dual.update_ghost_regions()
        return self._g_dual.copy(out=out)


class DriftKineticMaxwellCoupling(Propagator):
    r"""PIC-FEEC discretization of the following equations:
    find :math:`\mathbf E \in H(\textnormal{curl})`, :math:`\tilde{\mathbf B} \in H(\textnormal{div})` and the
    guiding-center distribution :math:`f(\mathbf X, v_\parallel, \mu)` (positions fixed in this step) such that

    .. math::

        \lambda^2 \int_{\Omega} \frac{\partial \mathbf E}{\partial t} \cdot \mathbf F \,\textrm d \mathbf x
        + \chi \int_{\Omega} \frac{n_0}{|B_0|^2} \frac{\partial \mathbf E_\perp}{\partial t} \cdot \mathbf F_\perp
        \,\textrm d \mathbf x
        &= c_1 \int_{\Omega} \tilde{\mathbf B} \cdot \nabla \times \mathbf F \,\textrm d \mathbf x
        - c_2 \int_{\Omega} \iint f^\textnormal{vol} \left( v_\parallel \frac{\mathbf B^*}{B^*_\parallel}
        + \frac{\mathbf E \times \mathbf b_0}{B^*_\parallel} \right) \cdot \mathbf F
        \,\textrm d v_\parallel \textrm d \mu \,\textrm d \mathbf x
        \qquad \forall \, \mathbf F \in H(\textnormal{curl})\,,
        \\[2mm]
        \frac{\partial \tilde{\mathbf B}}{\partial t} + \nabla\times\mathbf E &= 0\,,
        \\[2mm]
        \frac{\partial f}{\partial t} + c_3\, \frac{\mathbf B^*}{B^*_\parallel} \cdot \mathbf E\,
        \frac{\partial f}{\partial v_\parallel} &= 0\,,

    with :math:`\mathbf E_\perp = \mathbf E - (\mathbf b_0 \cdot \mathbf E) \mathbf b_0`, the phase-space
    density :math:`f^\textnormal{vol} = B^*_\parallel f` (the marker weights) and, as in
    :class:`~struphy.propagators.push_guiding_center_parallel.PushGuidingCenterParallel`,

    .. math::

        \mathbf B^* = \mathbf B_0 + \tilde{\mathbf B} + \varepsilon v_\parallel \nabla \times \mathbf b_0\,,\qquad
        B^*_\parallel = \mathbf b_0 \cdot \mathbf B^*\,,

    with the total magnetic field in the first term (parallel streaming along the perturbed field lines) and the
    curvature term of the equilibrium.

    The current is the guiding-center current :math:`\iint f^\textnormal{vol} \dot{\mathbf X}` without the
    magnetic drifts (which move the positions): parallel streaming, including the curvature drift in
    :math:`\mathbf B^*`, and (option ``exb_current``) the :math:`E \times B` drift. The coefficients are
    :math:`\lambda \geq 0`, :math:`\chi \geq 0` with :math:`\lambda^2 + \chi > 0`, :math:`c_1 > 0` and
    :math:`c_2 c_3 > 0`. The defaults are the coefficients of the species' :ref:`equation parameters <equation_params>`,
    :math:`\lambda = \varepsilon \lambda_\textrm{s}`, :math:`\chi = (\varepsilon \kappa_\textrm{s})^2`,
    :math:`c_1 = 1`, :math:`c_2 = \kappa_\textrm{s}^2 \varepsilon`, :math:`c_3 = 1/\varepsilon`, i.e. for the thermal
    velocity scale of :class:`~struphy.models.drift_kinetic_maxwell.DriftKineticMaxwell`
    :math:`\lambda = \hat v_\textrm{th}/c`, :math:`\chi = \hat v_\textrm{th}^2/\hat v_A^2`. The :math:`\chi` term is the
    (linearized, long-wavelength) polarization current of the guiding centers, which the drift-kinetic ordering
    removes from :math:`\dot{\mathbf X}`: without it, the perpendicular electrostatic field is undetermined for
    :math:`\lambda = 0` (flute modes), and with only the displacement current the shear Alfvén wave travels at the
    speed of light.

    With the FE coefficients :math:`\mathbf e \in \mathbb R^{N_1}` of :math:`\mathbf E \in V^1_h` and
    :math:`\mathbf b \in \mathbb R^{N_2}` of :math:`\tilde{\mathbf B} \in V^2_h`, and :math:`N` markers with logical
    positions :math:`\boldsymbol \eta_p` (fixed), parallel velocities :math:`v_{\parallel,p}` and weights :math:`w_p`,
    the semi-discrete system reads

    .. math::

        \mathbb M_E \dot{\mathbf e} = c_1\, \mathbb C^\top \mathbb M_2 \mathbf b
        - c_2 \left( \mathbf j(\mathbf V) + \mathbb S\, \mathbf e \right) \,,
        \qquad
        \dot{\mathbf b} = - \mathbb C \mathbf e\,,
        \qquad
        \dot v_{\parallel,p} = c_3\, \mathbf a_p \cdot \mathbf \Lambda^1_p \mathbf e\,,

    with :math:`\mathbb M_E = \lambda^2\, \mathbb M_1 + \chi\, \mathbb M_1^\perp` (:math:`\mathbb M_1^\perp` =
    :attr:`~struphy.feec.mass.WeightedMassOperators.M1gyro`; with ``polarization="darwin"``, :math:`\lambda = 0`
    and :math:`\mathbb M_E = \chi\, \mathbb P_L`, see below), :math:`\mathbf a_p =
    \hat{\mathbf B}^{*2} / \hat B^{*3}_\parallel (\boldsymbol \eta_p)`, :math:`\mathbf \Lambda^1_p` the basis of
    :math:`V^1_h` at :math:`\boldsymbol \eta_p`, and

    .. math::

        \mathbf j(\mathbf V) = \sum_p w_p v_{\parallel,p}\, (\mathbf \Lambda^1_p)^\top \mathbf a_p\,,
        \qquad
        \mathbb A_\parallel = \sum_p w_p\, (\mathbf \Lambda^1_p)^\top \mathbf a_p \mathbf a_p^\top \mathbf \Lambda^1_p\,,
        \qquad
        \mathbf f^\top \mathbb S\, \mathbf e = \sum_p \frac{w_p}{\hat B^{*3}_\parallel(\boldsymbol \eta_p)}\,
        (\mathbf \Lambda^1_p \mathbf f) \cdot \big( (\mathbf \Lambda^1_p \mathbf e) \times \hat{\mathbf b}^1_0
        (\boldsymbol \eta_p) \big)\,.

    :math:`\mathbb A_\parallel` is symmetric positive semi-definite (the parallel response), :math:`\mathbb S` is
    antisymmetric (the :math:`E \times B` current does no work). The system conserves the energy

    .. math::

        H = \frac{1}{2}\, \mathbf e^\top \mathbb M_E \mathbf e + \frac{c_1}{2}\, \mathbf b^\top \mathbb M_2 \mathbf b
        + \frac{c_2}{c_3} \sum_p w_p \left( \frac{v_{\parallel,p}^2}{2} + \mu_p |B_0|(\boldsymbol \eta_p) \right)
        \qquad (:meth:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling.energy`)\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point), with :math:`\mathbf a_p` and :math:`\mathbb S`
    evaluated at the beginning of the step, which conserves :math:`H` exactly (up to the solver tolerance; for
    constant weights). Eliminating :math:`\mathbf b^\textnormal{mid}` and :math:`\mathbf V^\textnormal{mid}` gives the
    Schur system for the mid-point field

    .. math::

        \left( \mathbb M_E + \frac{\Delta t^2}{4} \left( c_1\, \mathbb C^\top \mathbb M_2 \mathbb C
        + c_2 c_3\, \mathbb A_\parallel \right) + \frac{\Delta t}{2}\, c_2\, \mathbb S \right) \mathbf e^\textnormal{mid}
        = \mathbb M_E \mathbf e^n + \frac{\Delta t}{2} \left( c_1\, \mathbb C^\top \mathbb M_2 \mathbf b^n
        - c_2\, \mathbf j(\mathbf V^n) \right)\,,

    accumulated by :func:`~struphy.pic.accumulation.kernels.dk_maxwell.dk_maxwell_kernels.dk_maxwell` (or
    matrix-free) and solved with ``solver`` and ``precond`` (non-symmetric with the :math:`E \times B` current).
    Then

    .. math::

        \mathbf b^{n+1} = \mathbf b^n - \Delta t\, \mathbb C \mathbf e^\textnormal{mid}\,,
        \qquad
        v_{\parallel,p}^{n+1} = v_{\parallel,p}^n + \Delta t\, c_3\, \mathbf a_p \cdot \mathbf \Lambda^1_p
        \mathbf e^\textnormal{mid}\,,
        \qquad
        \mathbf e^{n+1} = 2 \mathbf e^\textnormal{mid} - \mathbf e^n\,,

    with the velocity update by
    :func:`~struphy.pic.pushing.kernels.push_vpar_with_efield.push_vpar_with_efield_kernels.push_vpar_with_efield`.

    **Quasi-neutral limit** :math:`\lambda = 0`. The parallel part of :math:`\mathbf e` is then determined by
    :math:`\mathbb A_\parallel` alone (which needs positive weights, i.e. full-f, and markers everywhere), the
    perpendicular part by :math:`\chi\, \mathbb M_1^\perp` and the curl-curl operator. The perpendicular field
    still carries the energy :math:`\frac 12 \mathbf e^\top \mathbb M_E \mathbf e`, so the Crank-Nicolson update is
    kept (storing :math:`\mathbf e^\textnormal{mid}` would dissipate it). The parallel part of the stored
    :math:`\mathbf e^{n+1}` is not determined by the scheme and may oscillate from step to step; it does not feed
    back, since the steps see :math:`\mathbf e^n` only through :math:`\mathbb M_E \mathbf e^n` and the
    :math:`E \times B` push only through :math:`\mathbf E_\perp`. With ``initial_projection``
    (default), the setup replaces :math:`(\mathbf b^0, \mathbf V^0)` by the mid-point state of one step, which
    removes the undamped period-2 oscillation of the constraint residual.

    **Darwin polarization** (option ``polarization="darwin"``). The displacement current is set to zero
    (:math:`\lambda = 0`, enforced) and the full perpendicular polarization :math:`\chi\, \mathbb M_1^\perp` is
    replaced by the longitudinal (Darwin) polarization

    .. math::

        \mathbb P_L = \mathbb P \Pi_G + (\mathbb I - \Pi_G)^\top \mathbb P \mathbb P_W (\mathbb I - \Pi_G)\,,
        \qquad
        \Pi_G = \mathbb G\, \mathbb L^{-1} \mathbb G^\top \mathbb P\,,
        \qquad
        \mathbb L = \mathbb G^\top \mathbb P \mathbb G + \sigma\, \mathbb M^0 \Pi_\textnormal{avg}\,,
        \qquad
        \mathbb P_W = \mathbb E (\mathbb E^\top \mathbb P \mathbb E)^+ \mathbb E^\top \mathbb P\,,

    with :math:`\mathbb P = \mathbb M_1^\perp = \mathbb M^1_{n_0/|B_0|^2}(\mathbb I - \mathbf b_0 \mathbf b_0^\top)`.
    The first term is the :math:`\mathbb P`-orthogonal projection on perpendicular gradients, the second the
    polarization of their :math:`\mathbb P`-orthogonal complement within the modes constant in the averaged
    directions (:math:`\mathbb P_W` is the projection on those modes, :math:`\mathbb E` embeds all three components).
    This is the weak form of :math:`\mathbf D^\textnormal{Dar} = \Pi_{L,\perp}(\chi\, n_0/|B_0|^2
    \mathbf E_\perp)`, Eq. (6.6) of Remmerswaal, Hatzky and Sonnendrücker, J. Plasma Phys. 91, E127 (2025): only the
    perpendicular-gradient part :math:`\mathbf E_L = -\nabla_\perp \phi` of :math:`\mathbf E_\perp` carries
    inertia, the transversal part is the multiplier of the quasi-static Ampère constraint. The Schur system above is
    unchanged with :math:`\mathbb M_E = \chi\, \mathbb P_L` (and :math:`\lambda = 0`), its transversal part is
    controlled by the curl-curl term, and the energy :math:`\frac{1}{2} \mathbf e^\top \mathbb M_E \mathbf e` is
    conserved as before, up to a drift of about :math:`0.04 \times` ``inner_tol``. :math:`\mathbb P_L` is applied
    matrix-free by :class:`~struphy.propagators.drift_kinetic_maxwell_coupling.DarwinLongitudinalOperator`, with an
    inner PCG solve of :math:`\mathbb L` (``inner_tol``, ``inner_maxiter``). The kernel of :math:`\mathbb G^\top
    \mathbb P \mathbb G` (:math:`k_\perp = 0`, functions constant in :math:`\eta_1, \eta_2`) is
    regularized by :math:`\sigma\, \mathbb M^0 \Pi_\textnormal{avg}` (``stab_eps``) and the polarization of its
    transverse complement is retained (``perp_polarization``); for a uniform perpendicular weight and metric this is
    the :math:`k_\perp = 0` shear Alfvén branch. The compressional branch (:math:`\mathbf E` transverse to both
    :math:`\mathbf k` and :math:`\mathbf b_0`) has no polarization and no oscillation. For the preconditioner,
    :math:`\mathbb P_L` is replaced by :math:`\mathbb M_1^\perp` in the stabilization (a :math:`\mathbb M_1^\perp`
    upper bound). That bound is loose for the inertia-free transverse modes, so the outer iteration count grows at
    small time steps (about 5x the full model in a small aligned test); the preconditioner remains valid but is not
    effective there. The Darwin variant requires a background field along :math:`\eta_3` (no :math:`\eta_1`,
    :math:`\eta_2` components; checked in :meth:`allocate`) and does not support polar splines. The transverse part
    of :math:`\mathbf e` in the kernel of :math:`\mathbb P_L` has no inertia: it is not determined by the energy and
    the update :math:`\mathbf e^{n+1} = 2\mathbf e^\textnormal{mid} - \mathbf e^n` flips its sign every step. It is
    nevertheless stored and passed to the :math:`E \times B` push by the model (a documented limitation of the
    Darwin variant); it stays zero for a longitudinal or zero initial field.

    **Preconditioner.** With :math:`\mathbb A_\parallel \approx \bar n\, \mathbb M_1^\parallel`
    (:attr:`~struphy.feec.mass.WeightedMassOperators.M1para`), the symmetric part of the Schur operator is
    approximately :math:`\mathbb C^\top \left(\frac{\Delta t^2 c_1}{4} \mathbb M_2\right) \mathbb C + \mathbb M_E
    + \frac{\Delta t^2 c_2 c_3 \bar n}{4} \mathbb M_1^\parallel`, preconditioned (default) by
    :class:`~struphy.feec.preconditioner.StiffnessPreconditioner`, built for the current time step
    (:meth:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling.prepare_time_step`).
    """

    class Variables:
        """Container for variables advanced by :class:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling`.

        Attributes
        ----------
        e : FEECVariable
            Electric-field variable in ``"Hcurl"`` space.
        b : FEECVariable
            Magnetic-field perturbation in ``"Hdiv"`` space.
        ions : PICVariable
            Guiding-center particle variable in ``"Particles5D"`` space.
        """

        def __init__(self):
            self._e: FEECVariable = None
            self._b: FEECVariable = None
            self._ions: PICVariable = None

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

        @property
        def ions(self) -> PICVariable:
            return self._ions

        @ions.setter
        def ions(self, new):
            assert isinstance(new, PICVariable)
            assert new.space == "Particles5D"
            self._ions = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`~struphy.propagators.drift_kinetic_maxwell_coupling.DriftKineticMaxwellCoupling`.

        Parameters
        ----------
        lam : float, default=None
            The displacement current has the coefficient :math:`\lambda^2` (``lam`` :math:`= \lambda \geq 0`);
            ``lam=0.0`` is the quasi-neutral limit (needs ``chi > 0`` and full-f weights, no control variate). If
            ``None``, :math:`\varepsilon \lambda_\textrm{s}` of the species (``lam`` and ``epsilon`` of
            its equation parameters).

        chi : float, default=None
            Coefficient :math:`\chi \geq 0` of the polarization current
            :math:`\frac{n_0}{|B_0|^2} \partial_t \mathbf E_\perp`. If ``None``,
            :math:`(\varepsilon \kappa_\textrm{s})^2` of the species (the polarization of the
            guiding centers themselves); a larger value mimics the polarization of a heavier species. ``lam**2 + chi``
            must be positive. The density :math:`n_0` is the one of the MHD equilibrium
            (:attr:`~struphy.feec.mass.WeightedMassOperators.M1gyro`), not of the kinetic background: choose them
            consistently.

        c1 : float, default=1.0
            Coefficient :math:`c_1 > 0` of :math:`\nabla \times \tilde{\mathbf B}` in Ampère's law.

        c2 : float, default=None
            Coefficient of the current density in Ampère's law. If ``None``, :math:`\kappa_\textrm{s}^2 \varepsilon`
            of the species (``kappa`` and ``epsilon`` of its equation parameters).

        c3 : float, default=None
            Coefficient of the parallel electric force (:math:`c_2 c_3 > 0`). If ``None``, :math:`1/\varepsilon` of
            the species.

        exb_current : bool, default=True
            Include the :math:`E \times B` current (antisymmetric operator :math:`\mathbb S`). The Schur system is then
            non-symmetric and needs a general solver. Switch it off if the :math:`E \times B` current is compensated
            (e.g. by other species).

        solver : OptsSolver, default="pbicgstab"
            Iterative solver of the Schur system, passed to ``feectools.linalg.solvers.inverse``. The symmetric
            solvers ``"pcg"`` and ``"cg"`` need ``exb_current=False``.

        precond : LiteralOptions.OptsMaxwellPrecond, default="StiffnessPreconditioner"
            Preconditioner of the Schur system (only for ``"pcg"`` and ``"pbicgstab"``; set to ``None`` otherwise).

            - ``"StiffnessPreconditioner"``: :class:`~struphy.feec.preconditioner.StiffnessPreconditioner` with
              ``derivative="curl"`` for the stabilized curl-curl approximation of the Schur operator (see the class
              docstring), rebuilt when the time step changes.
            - ``"MassMatrixPreconditioner"``, ``"PolarMassMatrixPreconditioner"``: preconditioner of
              :math:`\mathbb M_1` (with polar splines, ``"MassMatrixPreconditioner"`` is replaced by
              ``"PolarMassMatrixPreconditioner"``).
            - ``None``: no preconditioner.

        solver_params : SolverParameters, default=None
            Solver controls. If ``None``, defaults to ``SolverParameters()``.

        n_ref : float, default=None
            Reference density :math:`\bar n` in the stiffness preconditioner. If ``None``, the Rayleigh quotient
            :math:`\mathbf 1^\top \mathbb A_\parallel \mathbf 1 / \mathbf 1^\top \mathbb M_1^\parallel \mathbf 1` at
            the setup (a density average).

        polarization : LiteralOptions.OptsPolarization, default="full"
            The polarization current in Ampère's law.

            - ``"full"`` (default): :math:`\chi\, \mathbb M_1^\perp` on the whole perpendicular field, as in the
              equations of the class docstring (the current behaviour).
            - ``"darwin"``: the longitudinal (Darwin) polarization
              :math:`\chi\, \mathbb P_L = \chi\, [\mathbb P \Pi_G + (\mathbb I - \Pi_G)^\top \mathbb P \mathbb P_W
              (\mathbb I - \Pi_G)]`, with the :math:`\mathbb P`-orthogonal gradient projection
              :math:`\Pi_G = \mathbb G (\mathbb G^\top \mathbb M_1^\perp \mathbb G + \sigma \mathbb M^0
              \Pi_\textnormal{avg})^{-1} \mathbb G^\top \mathbb M_1^\perp` and the projection
              :math:`\mathbb P_W` on the modes constant in :math:`\eta_1, \eta_2` (see
              :class:`~struphy.propagators.drift_kinetic_maxwell_coupling.DarwinLongitudinalOperator`), i.e. only
              the perpendicular-gradient part of :math:`\mathbf E_\perp` plus the polarization of its
              :math:`\mathbb P`-orthogonal complement in the :math:`k_\perp = 0` modes (Eq. (6.6) of Remmerswaal,
              Hatzky and Sonnendrücker, J. Plasma Phys. 91, E127 (2025)), applied matrix-free with an inner solve
              (``inner_tol``, ``inner_maxiter``). This is the Darwin model: it needs ``lam=0`` (no displacement
              current; a positive ``lam`` raises an error) and a background field along :math:`\eta_3` (no
              :math:`\eta_1`, :math:`\eta_2` components; checked in :meth:`allocate`). Not available with polar
              splines. The transverse part of :math:`\mathbf e` in the kernel of :math:`\mathbb P_L` has no inertia
              and flips sign every step (see the class docstring).

        inner_tol : float, default=1e-12
            Only for ``polarization="darwin"``: relative tolerance of the inner solves of the regularized
            perpendicular Laplacian (relative to the norm of the right-hand side). The energy conservation of the
            coupling step is limited by it (about :math:`0.04 \times` ``inner_tol``; the default gives
            :math:`\sim 5\cdot 10^{-14}` for three steps). A larger value makes the outer solver's recursive
            residual differ from the true residual.

        inner_maxiter : int, default=1000
            Only for ``polarization="darwin"``: maximum number of iterations of the inner solves.

        stab_eps : float, default=1.0
            Only for ``polarization="darwin"``: the regularization :math:`\sigma > 0` of the kernel of the
            perpendicular Laplacian. For :math:`\mathbf b_0 \parallel \eta_3` the operator is independent of
            :math:`\sigma` (the kernel component of the potential does not enter the gradient).

        perp_polarization : bool, default=True
            Only for ``polarization="darwin"``: retain the polarization of the :math:`\mathbb P`-orthogonal
            complement of the gradient part within the modes constant in the averaged directions
            (:math:`k_\perp = 0`). Without it (the pure pseudo-inverse form) these modes have no polarization and the
            quasi-neutral Schur system is singular for them.

        matrix_free : bool, default=False
            Apply the particle matrices :math:`\mathbb A_\parallel` and :math:`\mathbb S` matrix-free, with their
            fillings stored per marker (:class:`~struphy.pic.accumulation.particles_to_grid.MarkerOperator`), instead
            of assembling them. Needs 7 auxiliary marker columns.

        initial_projection : bool, default=True
            Only for ``lam=0`` (also with ``polarization="darwin"``): in the setup (first
            :meth:`prepare_time_step`), replace :math:`(\mathbf b, \mathbf V)` by the mid-point state of one step,
            which satisfies the constraint of the quasi-neutral limit exactly. Not applied to a restored (restart)
            state.
        """

        # specific literals
        OptsSolver = Literal["pbicgstab", "bicgstab", "gmres", "pcg", "cg"]

        lam: float = None
        chi: float = None
        c1: float = 1.0
        c2: float = None
        c3: float = None
        exb_current: bool = True
        solver: OptsSolver = "pbicgstab"
        precond: LiteralOptions.OptsMaxwellPrecond = "StiffnessPreconditioner"
        solver_params: SolverParameters = None
        n_ref: float = None
        polarization: LiteralOptions.OptsPolarization = "full"
        inner_tol: float = 1e-12
        inner_maxiter: int = 1000
        stab_eps: float = 1.0
        perp_polarization: bool = True
        matrix_free: bool = False
        initial_projection: bool = True

        def __post_init__(self):
            # checks
            check_option(self.solver, self.OptsSolver)
            check_option(self.precond, LiteralOptions.OptsMaxwellPrecond)
            check_option(self.polarization, LiteralOptions.OptsPolarization)
            if self.lam is not None:
                assert self.lam >= 0.0, f"lam must be >= 0, got {self.lam}."
            if self.chi is not None:
                assert self.chi >= 0.0, f"chi must be >= 0, got {self.chi}."
            if self.lam is not None and self.chi is not None:
                assert self.lam**2 + self.chi > 0.0, (
                    "lam**2 + chi must be > 0: without displacement and polarization current, the perpendicular "
                    "electric field is undetermined."
                )
            assert self.c1 > 0.0, f"c1 must be > 0, got {self.c1}."
            if self.c2 is not None and self.c3 is not None:
                assert self.c2 * self.c3 > 0.0, f"c2 * c3 must be > 0, got c2={self.c2}, c3={self.c3}."
            if self.n_ref is not None:
                assert self.n_ref > 0.0, f"n_ref must be > 0, got {self.n_ref}."
            assert isinstance(self.exb_current, bool)
            assert isinstance(self.matrix_free, bool)
            assert isinstance(self.initial_projection, bool)
            assert isinstance(self.perp_polarization, bool)
            assert self.inner_tol > 0.0, f"inner_tol must be > 0, got {self.inner_tol}."
            assert self.inner_maxiter > 0, f"inner_maxiter must be > 0, got {self.inner_maxiter}."
            assert self.stab_eps > 0.0, f"stab_eps must be > 0, got {self.stab_eps}."
            if self.polarization == "darwin":
                if self.lam is not None:
                    assert self.lam == 0.0, (
                        'polarization="darwin" is the quasi-neutral Darwin model (no displacement current): lam=0.'
                    )
            if self.exb_current:
                assert self.solver not in ("pcg", "cg"), (
                    f'solver="{self.solver}" needs a symmetric system: use a general solver or exb_current=False.'
                )
            if self.solver not in ("pcg", "pbicgstab") and self.precond is not None:
                logger.warning(f'solver="{self.solver}" takes no preconditioner; ignoring precond="{self.precond}".')
                self.precond = None

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
        # coefficients
        equation_params = self.variables.ions.species.equation_params
        epsilon = equation_params.epsilon
        self._epsilon = epsilon
        # lam^2 = (epsilon lam_s)^2, chi = (epsilon kappa_s)^2, c2 = kappa_s^2 epsilon, c3 = 1 / epsilon
        # (see the equation parameters)
        self._lam2 = (epsilon * equation_params.lam) ** 2 if self.options.lam is None else self.options.lam**2
        self._chi = (epsilon * equation_params.kappa) ** 2 if self.options.chi is None else self.options.chi
        self._c1 = self.options.c1
        self._c2 = equation_params.kappa**2 * epsilon if self.options.c2 is None else self.options.c2
        self._c3 = 1.0 / epsilon if self.options.c3 is None else self.options.c3
        self._exb = self.options.exb_current
        assert self._c2 * self._c3 > 0.0, f"c2 * c3 must be > 0, got c2={self._c2}, c3={self._c3}."
        assert self._lam2 + self._chi > 0.0, f"lam**2 + chi must be > 0, got lam**2={self._lam2}, chi={self._chi}."

        # polarization: full perpendicular (default) or Darwin longitudinal (quasi-neutral, lam = 0)
        self._polarization = self.options.polarization
        if self._polarization == "darwin":
            assert self._lam2 == 0.0, (
                "polarization='darwin' is the quasi-neutral Darwin model (no displacement current): set lam=0.0 "
                f"(resolved lam**2={self._lam2})."
            )

        particles = self.variables.ions.particles
        if particles.control_variate:
            assert self._lam2 > 0.0, (
                "The quasi-neutral limit lam=0 needs full-f weights: with the control variate the accumulated "
                "matrix is indefinite and the Schur system is not well-posed."
            )
            logger.warning(
                "DriftKineticMaxwellCoupling with control variate: the energy is not conserved exactly "
                "(the weights change), and the response of the background f0 (parallel response and E x B current) "
                "is missing in the particle matrices."
            )

        self._info = self.options.solver_params.info

        # magnetic field: B* (2-form) and B*_parallel with the total field B0 + b (b frozen in a step)
        self._b2_eq = self.projected_equil.b2
        self._b2 = self._b2_eq.space.zeros()
        self._E2T = self.derham.extraction_ops["2"].transpose()
        self._curl_unit_b2 = self.projected_equil.curl_unit_b2
        self._unit_b1 = self.projected_equil.unit_b1
        self._absB0 = self.projected_equil.absB0
        self._curl_unit_b_dot_b0 = self.projected_equil.curl_unit_b_dot_b0

        self._E0T = self.derham.extraction_ops["0"].transpose()
        self._PB = getattr(self.basis_ops, "PB")
        self._B_dot_b = self.derham.V0.zeros()
        self._B_dot_b_pol = self._PB.codomain.zeros()
        self._update_b_star()

        args_fields = (
            self._b2[0]._data,
            self._b2[1]._data,
            self._b2[2]._data,
            self._curl_unit_b2[0]._data,
            self._curl_unit_b2[1]._data,
            self._curl_unit_b2[2]._data,
        )
        args_unit_b = (
            self._unit_b1[0]._data,
            self._unit_b1[1]._data,
            self._unit_b1[2]._data,
        )
        args_b_par = (
            self._B_dot_b._data,
            self._curl_unit_b_dot_b0._data,
        )

        # accumulation of A_par and S (assembled, or matrix-free) and the vector j
        assert xp.get_backend() != "cupy", "DriftKineticMaxwellCoupling has no CUDA kernels yet."
        self._matrix_free = self.options.matrix_free
        if self._matrix_free:
            # the vector, and the fillings of the matrices per marker (in the first free columns of the marker array)
            column = particles.args_markers.first_free_idx
            assert particles.n_cols_aux >= 7, "matrix_free needs 7 free (auxiliary) marker columns."
            self._args_accum_pre = (epsilon,)
            self._args_accum_post = args_fields + args_unit_b + args_b_par + (column,)
            self._accum = AccumulatorVector(
                particles,
                "Hcurl",
                dk_maxwell_mf,
                self.mass_ops,
                self.domain.args_domain,
            )
            self._A_par = MarkerOperator(
                particles,
                "Hcurl",
                marker_operator_rank1_dot,
                self.mass_ops,
                self.domain.args_domain,
                (1, column),
                symmetry="symm",
            )
            self._S = MarkerOperator(
                particles,
                "Hcurl",
                marker_operator_asym_dot,
                self.mass_ops,
                self.domain.args_domain,
                (1, column + 4),
                symmetry="asym",
            )
        else:
            # one matrix A_par + skew * S, skew = 2 / (dt * c3) (set in prepare_time_step)
            self._args_accum_pre = (epsilon,)
            self._args_accum_post = args_fields + args_unit_b + args_b_par
            self._accum = Accumulator(
                particles,
                "Hcurl",
                dk_maxwell,
                self.mass_ops,
                self.domain.args_domain,
                add_vector=True,
                symmetry=None,
            )
            self._A_lam = self._accum.operators[0]
        self._skew = 0.0

        # operators
        self._M1 = self.mass_ops.M1
        self._M2 = self.mass_ops.M2
        self._curl = self.derham.curl
        self._curl_curl = self._curl.T @ self._M2 @ self._curl
        if self._polarization == "darwin":
            # M_E = chi P_L, P_L matrix-free (longitudinal polarization); lam = 0 checked above
            self._PL = DarwinLongitudinalOperator(
                self.derham,
                self.mass_ops,
                stab_eps=self.options.stab_eps,
                inner_tol=self.options.inner_tol,
                inner_maxiter=self.options.inner_maxiter,
                perp_polarization=self.options.perp_polarization,
            )
            self._ME = self._chi * self._PL
        elif self._lam2 > 0.0 and self._chi > 0.0:
            self._ME = self._lam2 * self._M1 + self._chi * self.mass_ops.M1gyro
        elif self._lam2 > 0.0:
            self._ME = self._lam2 * self._M1
        else:
            self._ME = self._chi * self.mass_ops.M1gyro

        # reference density for the stiffness preconditioner: density average of the parallel particle matrix
        if self.options.n_ref is None:
            self._accumulate()
            ones = extracted_ones(self.derham, "1")
            A_par = self._A_par if self._matrix_free else self._A_lam
            ones_A = ones.inner(A_par.dot(ones))
            ones_M = ones.inner(self.mass_ops.M1para.dot(ones))
            if ones_A > 0.0 and ones_M > 0.0:
                self._n_ref = ones_A / ones_M
            else:
                self._n_ref = 1.0
                logger.warning(
                    "DriftKineticMaxwellCoupling: no positive density average of the markers, using n_ref=1."
                )
        else:
            self._n_ref = self.options.n_ref

        # preconditioner (the stiffness preconditioner depends on dt and is built in prepare_time_step)
        self._precond = self.options.precond
        if self._precond == "MassMatrixPreconditioner" and self.derham.polar_splines:
            self._precond = "PolarMassMatrixPreconditioner"
        if self._precond in (None, "StiffnessPreconditioner"):
            self._mass_pc = None
        else:
            pc_class = getattr(preconditioner, self._precond)
            self._mass_pc = pc_class(self._M1)

        # solver kwargs (the Schur operator and its solver are built in prepare_time_step)
        kwargs = self.options.solver_params.__dict__.copy()
        kwargs.pop("info")
        self._solver_kwargs = kwargs
        self._solver = None
        self._schur_dt = None
        self._projected = False

        # buffers in the spaces of e and b (polar with polar splines)
        e_space = self.variables.e.spline.vector.space
        b_space = self.variables.b.spline.vector.space
        self._e_mid = e_space.zeros()
        self._e_new = e_space.zeros()
        self._e_rhs = e_space.zeros()
        self._e_tmp = e_space.zeros()
        self._b_new = b_space.zeros()
        self._b_tmp = b_space.zeros()

        # the mid-point field on the tensor-product space, read by the pusher kernel: E1^T e_mid (e_mid itself
        # without polar splines)
        self._E1T = self.derham.extraction_ops["1"].transpose()
        self._e_mid_tp = self.derham.V1.zeros() if self.derham.polar_splines else self._e_mid

        # particle pusher: v^{n+1} = v^n + dt * c3 * a_p . Lambda^1 e_mid
        args_kernel = (
            (
                self.derham.args_derham,
                epsilon,
                self._e_mid_tp.blocks[0]._data,
                self._e_mid_tp.blocks[1]._data,
                self._e_mid_tp.blocks[2]._data,
            )
            + args_fields
            + args_b_par
            + (self._c3,)
        )

        self._pusher = Pusher(
            particles,
            push_vpar_with_efield,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
            pushes_eta=False,
        )

    def _update_b_star(self):
        """Total magnetic field B0 + b (2-form, in B*) and parallel field b0 . (B0 + b) (0-form, in B*_parallel) in the
        tensor-product spaces."""
        b = self.variables.b.spline.vector
        b2 = self._E2T.dot(b, out=self._b2)
        b2 += self._b2_eq
        b2.update_ghost_regions()

        B_dot_b_pol = self._PB.dot(b, out=self._B_dot_b_pol)
        B_dot_b = self._E0T.dot(B_dot_b_pol, out=self._B_dot_b)
        B_dot_b += self._absB0
        B_dot_b.update_ghost_regions()

    def _accumulate(self):
        """Accumulate the particle matrices (or their fillings per marker) and the vector j at the current markers."""
        if self._matrix_free:
            self._accum(*self._args_accum_pre, *self._args_accum_post)
        else:
            self._accum(*self._args_accum_pre, self._skew, *self._args_accum_post)

    def prepare_time_step(self, dt: float):
        """Schur operator, its preconditioner and solver for the time step ``dt``: built in the setup, rebuilt when dt
        changes. The particle matrices enter by reference (assembled, or matrix-free reading the fillings stored in
        the markers), so they are updated by the accumulation in every step. For ``lam=0`` (and
        ``initial_projection``), the first call also projects the state onto the constraint."""
        if dt == self._schur_dt:
            return

        schur = self._ME + (dt**2 * self._c1 / 4) * self._curl_curl
        if self._matrix_free:
            schur = schur + (dt**2 * self._c2 * self._c3 / 4) * self._A_par
            if self._exb:
                schur = schur + (dt * self._c2 / 2) * self._S
        else:
            # A_lam = A_par + skew S with skew = 2 / (dt c3): dt^2 c2 c3 / 4 A_lam = dt^2 c2 c3 / 4 A_par + dt c2 / 2 S
            self._skew = 2.0 / (dt * self._c3) if self._exb else 0.0
            schur = schur + (dt**2 * self._c2 * self._c3 / 4) * self._A_lam

        if self._precond == "StiffnessPreconditioner":
            # For Darwin, P_L is matrix-free; the preconditioner uses chi * M1gyro (the full perpendicular
            # polarization), which is an upper bound of chi * P_L in the M1gyro inner product and matches the
            # operator for the modes that dominate the stiffness.
            ME_stab = self._chi * self.mass_ops.M1gyro if self._polarization == "darwin" else self._ME
            stab = ME_stab + (dt**2 * self._c2 * self._c3 * self._n_ref / 4) * self.mass_ops.M1para
            pc = preconditioner.StiffnessPreconditioner(
                self.mass_ops,
                "curl",
                sigma=1.0,
                mid=(dt**2 * self._c1 / 4) * self._M2,
                stab=stab,
            )
        else:
            pc = self._mass_pc

        kwargs = dict(self._solver_kwargs)
        if pc is not None:
            kwargs["pc"] = pc
        self._solver = inverse(schur, self.options.solver, **kwargs)
        self._schur_dt = dt

        # lam = 0: consistent initial state, once (in the setup, before the initial data are saved)
        if self._lam2 == 0.0 and self.options.initial_projection and not self._projected:
            self._project_onto_constraint(dt)
            self._projected = True

    def energies(self) -> tuple[float, float, float]:
        r"""The three parts :math:`\frac{1}{2} \mathbf e^\top \mathbb M_E \mathbf e` (displacement and polarization),
        :math:`\frac{c_1}{2} \mathbf b^\top \mathbb M_2 \mathbf b` and
        :math:`\frac{c_2}{c_3} \sum_p w_p (v_{\parallel,p}^2/2 + \mu_p |B_0|(\boldsymbol \eta_p))` of the energy
        :meth:`energy` (summed over all MPI ranks; a collective operation)."""
        e = self.variables.e.spline.vector
        b = self.variables.b.spline.vector
        particles = self.variables.ions.particles

        particles.save_magnetic_background_energy()
        valid = particles.valid_mks
        markers = particles.markers
        kinetic = xp.array(
            [
                xp.sum(
                    markers[5, valid] * (0.5 * markers[3, valid] ** 2 + markers[particles.first_diagnostics_idx, valid])
                )
            ]
        )
        if self.derham.comm is not None:
            self.derham.comm.Allreduce(MPI.IN_PLACE, kinetic, op=MPI.SUM)
        if particles.clone_config is not None:
            particles.clone_config.inter_comm.Allreduce(MPI.IN_PLACE, kinetic, op=MPI.SUM)

        energy_e = 0.5 * e.inner(self._ME.dot(e, out=self._e_tmp))
        energy_b = self._c1 / 2 * b.inner(self._M2.dot(b, out=self._b_tmp))
        energy_f = self._c2 / self._c3 * float(kinetic[0])
        return energy_e, energy_b, energy_f

    def energy(self) -> float:
        r"""The energy :math:`H = \frac{1}{2} \mathbf e^\top \mathbb M_E \mathbf e + \frac{c_1}{2} \mathbf b^\top
        \mathbb M_2 \mathbf b + \frac{c_2}{c_3} \sum_p w_p (v_{\parallel,p}^2/2 + \mu_p |B_0|(\boldsymbol \eta_p))`
        conserved by the propagator (summed over all MPI ranks; a collective operation)."""
        return sum(self.energies())

    def _mid_point_field(self, dt: float):
        """Accumulate at the current markers and solve the Schur system for the mid-point field (into ``_e_mid``,
        with updated ghost regions, which the pusher kernel reads)."""
        en = self.variables.e.spline.vector
        bn = self.variables.b.spline.vector

        # B*_parallel with b^n, then A_par, S (or their fillings per marker) and j
        self._update_b_star()
        self._accumulate()

        # rhs = M_E e^n + dt/2 (c1 C^T M2 b^n - c2 j)
        rhs = self._curl.T.dot(self._M2.dot(bn, out=self._b_tmp), out=self._e_rhs)
        rhs *= self._c1
        j = self._accum.vectors[0]
        j.copy(out=self._e_tmp)
        self._e_tmp *= self._c2
        rhs -= self._e_tmp
        rhs *= dt / 2
        rhs += self._ME.dot(en, out=self._e_tmp)

        e_mid = self._solver.dot(rhs, out=self._e_mid)
        e_mid.update_ghost_regions()
        if self._e_mid_tp is not e_mid:
            self._E1T.dot(e_mid, out=self._e_mid_tp)
            self._e_mid_tp.update_ghost_regions()
        return e_mid, self._solver._info

    def _project_onto_constraint(self, dt: float):
        """For lam = 0: replace (b, V) by the mid-point state of one step, which satisfies the constraint of the
        quasi-neutral limit exactly. The electric field is kept."""
        bn = self.variables.b.spline.vector
        e_mid, _ = self._mid_point_field(dt)

        # b_mid = b^n - dt/2 C e_mid, v_mid = v^n + dt/2 c3 a_p . Lambda^1 e_mid
        b_new = self._curl.dot(e_mid, out=self._b_new)
        b_new *= -dt / 2
        b_new += bn
        self._pusher(dt / 2)
        self.update_feec_variables(b=b_new)
        logger.info("DriftKineticMaxwellCoupling: projected the initial state onto the constraint (lam = 0).")

    @profile
    def __call__(self, dt):
        en = self.variables.e.spline.vector
        bn = self.variables.b.spline.vector

        # Schur operator for the current time step (no-op unless dt changed since the setup)
        self.prepare_time_step(dt)

        # mid-point field
        e_mid, info = self._mid_point_field(dt)

        # new b coeffs: b^{n+1} = b^n - dt C e_mid
        b_new = self._curl.dot(e_mid, out=self._b_new)
        b_new *= -dt
        b_new += bn

        # new e coeffs: Crank-Nicolson (M_E = lam^2 M1 + chi M1gyro is nonzero, so e carries energy also for lam = 0)
        e_new = e_mid.copy(out=self._e_new)
        e_new *= 2.0
        e_new -= en

        # update parallel velocities with e_mid
        self._pusher(dt)

        # update weights
        if self.variables.ions.species.weights_params.control_variate:
            self.variables.ions.particles.update_weights()

        # write new coeffs into self.variables
        diffs = self.update_feec_variables(e=e_new, b=b_new)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for DriftKineticMaxwellCoupling: {info['success']}")
            logger.info(f"Iterations for DriftKineticMaxwellCoupling: {info['niter']}")
            logger.info(f"Maxdiff e  for DriftKineticMaxwellCoupling: {diffs['e']}")
            logger.info(f"Maxdiff b  for DriftKineticMaxwellCoupling: {diffs['b']}")
            logger.info("")
