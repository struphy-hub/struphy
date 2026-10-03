r"""Smoothers for geometric multigrid.

A smoother approximately solves :math:`A x = b` by the update :math:`x \leftarrow x + S (b - A x)`.
:class:`ChebyshevSmoother` and :class:`JacobiSmoother` are linear and :math:`A`-symmetric, hence a V-cycle
with the same smoother before and after the coarse-grid correction is a symmetric preconditioner, suitable
for the conjugate gradient method. :class:`KrylovSmoother` is non-linear (use with care).
"""

from abc import ABC, abstractmethod

import numpy as np
from feectools.linalg.basic import (
    IdentityOperator,
    LinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
    Vector,
    ZeroOperator,
)
from feectools.linalg.block import BlockVector
from feectools.linalg.solvers import inverse
from feectools.linalg.stencil import StencilVector


class Smoother(ABC):
    """Base class of multigrid smoothers for the operator ``A``."""

    def __init__(self, A: LinearOperator):
        assert A.domain is A.codomain
        self._A = A
        self._r = A.codomain.zeros()

    @property
    def A(self) -> LinearOperator:
        return self._A

    @property
    def is_symmetric(self) -> bool:
        """Whether the smoother is linear and A-symmetric (needed for a symmetric V-cycle)."""
        return True

    def residual(self, b: Vector, x: Vector, out: Vector) -> Vector:
        """``out = b - A x``."""
        self._A.dot(x, out=out)
        out *= -1.0
        out += b
        return out

    @abstractmethod
    def smooth(self, b: Vector, x: Vector) -> None:
        """Improve the approximate solution ``x`` of ``A x = b`` in place."""


class JacobiSmoother(Smoother):
    r"""Damped Jacobi, :math:`x \leftarrow x + \omega D^{-1}(b - A x)`, repeated ``sweeps`` times.

    Parameters
    ----------
    A : LinearOperator
        System operator.

    diag_inv : LinearOperator
        Inverse diagonal :math:`D^{-1}` of ``A`` (e.g. from :func:`inverse_diagonal`).

    omega : float
        Damping factor.

    sweeps : int
        Number of iterations per call.
    """

    def __init__(self, A: LinearOperator, diag_inv: LinearOperator, *, omega: float = 2.0 / 3.0, sweeps: int = 1):
        super().__init__(A)
        self._D_inv = diag_inv
        self._omega = omega
        self._sweeps = sweeps
        self._z = A.domain.zeros()

    def smooth(self, b: Vector, x: Vector) -> None:
        for _ in range(self._sweeps):
            self.residual(b, x, self._r)
            self._D_inv.dot(self._r, out=self._z)
            x.mul_iadd(self._omega, self._z)


class ChebyshevSmoother(Smoother):
    r"""Chebyshev polynomial smoother of degree ``degree`` for :math:`M^{-1} A`.

    The polynomial damps the eigenmodes of :math:`M^{-1} A` in the interval
    :math:`[\alpha \lambda_\max, \beta \lambda_\max]` (``bounds = (alpha, beta)``), where
    :math:`\lambda_\max` is estimated with a few Lanczos (PCG) steps. Each call costs ``degree``
    applications of ``A`` and of ``M_inv``. No inner products are computed.

    Parameters
    ----------
    A : LinearOperator
        Symmetric positive (semi-)definite system operator.

    M_inv : LinearOperator
        Symmetric positive definite preconditioner, e.g. an approximate mass-matrix inverse or
        an inverse diagonal of ``A``.

    degree : int
        Polynomial degree.

    bounds : tuple[float, float]
        Lower and upper end of the smoothing interval relative to the estimated :math:`\lambda_\max`.

    eig_iter : int
        Number of Lanczos steps for estimating :math:`\lambda_\max`.

    lambda_max : float | None
        Largest eigenvalue of :math:`M^{-1} A`, estimated if None.
    """

    def __init__(
        self,
        A: LinearOperator,
        M_inv: LinearOperator,
        *,
        degree: int = 3,
        bounds: tuple[float, float] = (0.1, 1.1),
        eig_iter: int = 15,
        lambda_max: float | None = None,
    ):
        super().__init__(A)
        assert degree >= 1
        assert 0.0 < bounds[0] < bounds[1]
        self._M_inv = M_inv
        self._degree = degree

        if lambda_max is None:
            lambda_max = estimate_lambda_max(A, M_inv, n_iter=eig_iter)
        self._lambda_max = lambda_max
        a, b = bounds[0] * lambda_max, bounds[1] * lambda_max
        self._theta = 0.5 * (b + a)
        self._delta = 0.5 * (b - a)

        self._d = A.domain.zeros()
        self._z = A.domain.zeros()

    @property
    def lambda_max(self) -> float:
        return self._lambda_max

    def smooth(self, b: Vector, x: Vector) -> None:
        # Saad, Iterative Methods for Sparse Linear Systems, Alg. 12.1 (preconditioned)
        theta, delta = self._theta, self._delta
        sigma = theta / delta
        rho = 1.0 / sigma
        r, d, z = self._r, self._d, self._z

        self.residual(b, x, r)
        self._M_inv.dot(r, out=d)
        d *= 1.0 / theta
        for k in range(self._degree):
            x += d
            if k == self._degree - 1:
                break
            self._A.dot(d, out=z)
            r -= z
            rho_new = 1.0 / (2.0 * sigma - rho)
            self._M_inv.dot(r, out=z)
            d *= rho_new * rho
            d.mul_iadd(2.0 * rho_new / delta, z)
            rho = rho_new


class KrylovSmoother(Smoother):
    """A fixed number of (preconditioned) conjugate gradient iterations, warm-started from ``x``.

    Note that this smoother is non-linear; a V-cycle using it is not a fixed linear preconditioner.
    """

    def __init__(self, A: LinearOperator, M_inv: LinearOperator | None = None, *, iterations: int = 3):
        super().__init__(A)
        self._solver = inverse(A, "pcg", pc=M_inv, maxiter=iterations, tol=1e-300, recycle=False)

    @property
    def is_symmetric(self) -> bool:
        return False

    def smooth(self, b: Vector, x: Vector) -> None:
        self._solver._options["x0"] = x.copy()
        self._solver.dot(b, out=x)


def estimate_lambda_max(A: LinearOperator, M_inv: LinearOperator, *, n_iter: int = 15, seed: int = 1234) -> float:
    r"""Estimate the largest eigenvalue of :math:`M^{-1} A` with ``n_iter`` Lanczos steps (via PCG coefficients).

    The estimate is a lower bound that converges quickly to :math:`\lambda_\max`.
    """
    b = A.domain.zeros()
    _fill_random(b, seed)

    x_r = b.copy()
    z = M_inv.dot(x_r)
    p = z.copy()
    q = A.domain.zeros()
    rz = x_r.inner(z)
    alphas, betas = [], []
    for _ in range(n_iter):
        A.dot(p, out=q)
        pq = p.inner(q)
        if pq <= 0.0 or rz <= 0.0:
            break
        alpha = rz / pq
        x_r.mul_iadd(-alpha, q)
        M_inv.dot(x_r, out=z)
        rz_new = x_r.inner(z)
        beta = rz_new / rz
        alphas.append(alpha)
        betas.append(beta)
        if rz_new <= 1e-30 * rz:
            break
        p *= beta
        p += z
        rz = rz_new

    k = len(alphas)
    assert k > 0, "Lanczos breakdown in the eigenvalue estimate (is A positive semi-definite?)."
    T = np.zeros((k, k))
    for j in range(k):
        T[j, j] = 1.0 / alphas[j] + (betas[j - 1] / alphas[j - 1] if j > 0 else 0.0)
        if j + 1 < k:
            T[j, j + 1] = T[j + 1, j] = np.sqrt(betas[j]) / alphas[j]
    return float(np.linalg.eigvalsh(T).max())


def _stencil_blocks(v: Vector) -> list[StencilVector]:
    return list(v.blocks) if isinstance(v, BlockVector) else [v]


def _fill_random(v: Vector, seed: int) -> None:
    """Fill the owned entries of ``v`` with uniform random numbers in [-1, 1] (different on each process)."""
    for n, blk in enumerate(_stencil_blocks(v)):
        V = blk.space
        rng = np.random.default_rng([seed, n] + [int(s) for s in V.starts])
        idx = _owned_slice(blk)
        blk._data[idx] = rng.uniform(-1.0, 1.0, size=blk._data[idx].shape)
        blk.ghost_regions_in_sync = False


def _owned_slice(v: StencilVector) -> tuple[slice, ...]:
    V = v.space
    return tuple(slice(p * m, p * m + e - s + 1) for p, m, s, e in zip(V.pads, V.shifts, V.starts, V.ends))


# ----------------------------------------------------------------------------------------------------
# diagonal of composite operators
# ----------------------------------------------------------------------------------------------------
class DiagonalOperator(LinearOperator):
    """Pointwise multiplication by the entries of a vector."""

    def __init__(self, d: Vector):
        self._d = d
        self._space = d.space

    @property
    def domain(self):
        return self._space

    @property
    def codomain(self):
        return self._space

    @property
    def dtype(self):
        return self._space.dtype

    @property
    def vector(self) -> Vector:
        return self._d

    def dot(self, v: Vector, out: Vector | None = None) -> Vector:
        if out is None:
            out = self._space.zeros()
        for vb, db, ob in zip(_stencil_blocks(v), _stencil_blocks(self._d), _stencil_blocks(out)):
            idx = _owned_slice(ob)
            ob._data[idx] = db._data[idx] * vb._data[idx]
            ob.ghost_regions_in_sync = False
        return out

    def transpose(self, conjugate: bool = False) -> "DiagonalOperator":
        return self


class DiagonalComputer:
    r"""Exact diagonal of (composite) operators, cached per operator object.

    Sums and scalings are combined from the diagonals of their parts; identity and zero operators are
    trivial; every other operator (a leaf, or a product such as :math:`G^\top M G`) is probed with
    colored unit vectors: dofs with equal index modulo :math:`c_d \geq 2 w_d + 1` in every direction do
    not couple, so one application of the operator per color gives the diagonal entries of all dofs of
    that color. This costs :math:`\prod_d c_d` operator applications (times the number of components).

    Parameters
    ----------
    widths : tuple[int, int, int]
        Upper bound for the coupling distance (in index units) of the operators in each direction,
        e.g. the spline degrees for mass and stiffness matrices.
    """

    def __init__(self, widths: tuple[int, int, int]):
        self._widths = tuple(widths)
        self._cache: dict[int, tuple[LinearOperator, Vector]] = {}

    def __call__(self, A: LinearOperator) -> Vector:
        """Diagonal of ``A`` as a vector in ``A.domain``."""
        assert A.domain is A.codomain
        if isinstance(A, SumLinearOperator):
            d = A.domain.zeros()
            for a in A.addends:
                d += self(a)
            return d
        if isinstance(A, ScaledLinearOperator):
            return self(A.operator) * A.scalar
        if isinstance(A, IdentityOperator):
            d = A.domain.zeros()
            for blk in _stencil_blocks(d):
                blk._data[...] = 1.0
            return d
        if isinstance(A, ZeroOperator):
            return A.domain.zeros()

        cached = self._cache.get(id(A))
        if cached is not None and cached[0] is A:
            return cached[1]
        d = self._probe(A)
        self._cache[id(A)] = (A, d)
        return d

    def _probe(self, A: LinearOperator) -> Vector:
        d = A.domain.zeros()
        e = A.domain.zeros()
        y = A.domain.zeros()
        d_blocks, e_blocks = _stencil_blocks(d), _stencil_blocks(e)

        for n, (db, eb) in enumerate(zip(d_blocks, e_blocks)):
            V = eb.space
            colors = [
                _n_colors(int(npts), 2 * w + 1, bool(per)) for npts, w, per in zip(V.npts, self._widths, V.periods)
            ]
            glob = [np.arange(s, e + 1) for s, e in zip(V.starts, V.ends)]
            idx = _owned_slice(eb)
            for color in np.ndindex(*colors):
                mask = np.ones([len(g) for g in glob], dtype=bool)
                for axis, (g, c, k) in enumerate(zip(glob, colors, color)):
                    shape = [1, 1, 1]
                    shape[axis] = len(g)
                    mask = mask & (g % c == k).reshape(shape)
                for blk in e_blocks:
                    blk._data[...] = 0.0
                    blk.ghost_regions_in_sync = False
                eb._data[idx] = mask.astype(float)
                A.dot(e, out=y)
                yb = _stencil_blocks(y)[n]
                db._data[idx][mask] = yb._data[idx][mask]
        return d


def _n_colors(n: int, c: int, periodic: bool) -> int:
    """Number of colors in one direction with ``n`` dofs such that equal colors are at least ``c`` apart."""
    if c >= n:
        return n
    if not periodic:
        return c
    # periodic: c must divide n to avoid coupling across the periodic boundary
    for k in range(c, n + 1):
        if n % k == 0:
            return k
    return n


def inverse_diagonal(diag: Vector) -> DiagonalOperator:
    """Inverse of a diagonal (entries equal to zero, e.g. Dirichlet dofs, are mapped to zero)."""
    inv = diag.copy()
    for blk in _stencil_blocks(inv):
        idx = _owned_slice(blk)
        data = blk._data[idx]
        out = np.zeros_like(data)
        np.divide(1.0, data, out=out, where=data != 0.0)
        blk._data[idx] = out
    return DiagonalOperator(inv)
