"""Two-dimensional (disk-like) coordinate transformations of the logical unit square.

Port of the disk-like mappings of ``src/mappings.jl`` of PolarFEEC.jl. Derivatives are computed with the complex-step
method, which is exact to rounding for analytic functions written with NumPy ufuncs (the Julia code uses
ForwardDiff). All methods are vectorized in the logical coordinates ``(x, y)``: ``x`` is the radial and ``y`` the
periodic angular coordinate.
"""

import numpy as np

_STEP = 1e-30


def complex_step(f, x, *args, argnum=0):
    """Derivative of the analytic function ``f`` in its argument ``argnum`` by the complex-step method."""
    arguments = [np.asarray(x, dtype=float)] + [np.asarray(a, dtype=float) for a in args]
    arguments[argnum] = arguments[argnum] + 1j * _STEP
    return np.imag(f(*arguments)) / _STEP


class Mapping2D:
    """Base class of 2D coordinate transformations ``F(x, y) = (X, Y)``."""

    has_polar_axis = False

    def __call__(self, x, y):
        raise NotImplementedError

    def jacobian_matrix(self, x, y):
        """Jacobian ``J[..., i, j] = dF_i/dx_j`` of shape ``broadcast(x, y).shape + (2, 2)``."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        J = np.empty(x.shape + (2, 2))
        for j in range(2):
            columns = complex_step(lambda a, b: np.stack(self(a, b), axis=-1), x, y, argnum=j)
            J[..., :, j] = columns
        return J

    def jacobian(self, x, y):
        J = self.jacobian_matrix(x, y)
        return J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]


class DiskMapping(Mapping2D):
    """The disk-like map ``F(x, y) = (R(x, y) C(y), R(x, y) S(y))`` of the paper.

    Parameters
    ----------
    radius : float or callable
        Either the disk radius ``R1`` (``R = R1 x``) or a function ``R(x, y)`` with ``R(0, y) = 0`` and ``dR/dx > 0``.
    angular : tuple of callables, optional
        Periodic functions ``(C, S)`` with ``C S' - S C' > 0``; the default is ``(cos(2 pi y), sin(2 pi y))``.

    Example: the perturbed disk of the paper is ``DiskMapping(radius=lambda x, y: x * (1 + 0.2 * x * cos(6 pi y)))``.
    """

    has_polar_axis = True

    def __init__(self, radius=1.0, angular=None):
        if callable(radius):
            self.radius = radius
            self.R1 = None
        else:
            if radius <= 0:
                raise ValueError("radius must be positive")
            R1 = float(radius)
            self.R1 = R1
            self.radius = lambda x, y: R1 * x
        if angular is None:
            self.cosine = lambda y: np.cos(2 * np.pi * y)
            self.sine = lambda y: np.sin(2 * np.pi * y)
            self.circular = True
        else:
            self.cosine, self.sine = angular
            self.circular = False

    def __call__(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        R = self.radius(x, y)
        return R * self.cosine(y), R * self.sine(y)

    def _derivatives(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        R = self.radius(x, y) * np.ones_like(x)
        Rx = complex_step(self.radius, x, y, argnum=0) * np.ones_like(x)
        Ry = complex_step(self.radius, x, y, argnum=1) * np.ones_like(x)
        C, S = self.cosine(y), self.sine(y)
        Cy, Sy = complex_step(self.cosine, y), complex_step(self.sine, y)
        return R, Rx, Ry, C, S, Cy, Sy

    def jacobian_matrix(self, x, y):
        R, Rx, Ry, C, S, Cy, Sy = self._derivatives(x, y)
        J = np.empty(R.shape + (2, 2))
        J[..., 0, 0] = Rx * C
        J[..., 0, 1] = Ry * C + R * Cy
        J[..., 1, 0] = Rx * S
        J[..., 1, 1] = Ry * S + R * Sy
        return J

    def jacobian(self, x, y):
        R, Rx, _, C, S, Cy, Sy = self._derivatives(x, y)
        return R * Rx * (C * Sy - S * Cy)

    def inverse(self, X, Y, iterations=60):
        """Logical coordinates of physical points (polar angle and a safeguarded radial Newton/bisection)."""
        X, Y = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(Y, dtype=float))
        if not self.circular:
            raise NotImplementedError("the inverse is implemented for the default angular functions")
        rho = np.hypot(X, Y)
        y = (np.arctan2(Y, X) / (2 * np.pi)) % 1.0
        outer = self.radius(np.ones_like(y), y)
        lower, upper = np.zeros_like(rho), np.ones_like(rho)
        x = np.clip(rho / outer, 0.0, 1.0)
        for _ in range(iterations):
            residual = self.radius(x, y) - rho
            upper = np.where(residual > 0, x, upper)
            lower = np.where(residual <= 0, x, lower)
            slope = complex_step(self.radius, x, y, argnum=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                candidate = x - residual / slope
            x = np.where((candidate > lower) & (candidate < upper), candidate, (lower + upper) / 2)
        return x, y


class CoordinateTransformation2D(Mapping2D):
    """A general map ``F(x, y) -> (X, Y)`` given by an analytic, vectorized function."""

    def __init__(self, function, has_polar_axis=False):
        self.function = function
        self.has_polar_axis = has_polar_axis

    def __call__(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        X, Y = self.function(x, y)
        return X * np.ones_like(x), Y * np.ones_like(x)


def perturbed_disk(R1=1.0, eps=0.2, k=3):
    """The paper's disk ``R = R1 x (1 + eps x cos(2 pi k y))`` (three petals for ``k = 3``)."""
    return DiskMapping(radius=lambda x, y: R1 * x * (1 + eps * x * np.cos(2 * np.pi * k * y)))
