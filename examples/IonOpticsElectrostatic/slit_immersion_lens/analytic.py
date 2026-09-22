"""Analytic potential of a two-electrode slit immersion lens.

The lens is a slit of half-width ``h`` (plates at ``y = ±h``, invariant in z).
Upstream of the gap both plates are at ``v1``, downstream at ``v2``; across a gap
of width ``g`` centred at ``xc`` the plate potential rises linearly. For a
zero-width gap, the potential inside the strip is

    w(x, y) = sgn(x)/2 * (1 - 4/pi * Re arctan(exp(-pi(|x| + i y)/(2h)))),

which follows from the Fourier series of the step solution and
``sum (-1)^n z^(2n+1)/(2n+1) = arctan z``. The linear gap is the average of
``w`` over the gap, computed with Gauss-Legendre quadrature; ``w`` is analytic
inside the strip, so this is accurate to roundoff.
"""

import numpy as np
from scipy.integrate import solve_ivp


class SlitImmersionLens:
    """Exact potential and field of the slit lens, in normalized units."""

    def __init__(self, h, g, xc, v1, v2, n_quad=64):
        self.h, self.g, self.xc, self.v1, self.v2 = h, g, xc, v1, v2
        self._s, self._w = np.polynomial.legendre.leggauss(n_quad)

    def _u(self, x, y):
        return np.exp(-np.pi * (np.abs(x) + 1j * y) / (2 * self.h))

    def _step(self, x, y):
        return 0.5 * np.sign(x) * (1.0 - 4.0 / np.pi * np.arctan(self._u(x, y)).real)

    def _step_dy(self, x, y):
        u = self._u(x, y)
        return -2.0 / np.pi * np.sign(x) * (-1j * np.pi / (2 * self.h) * u / (1 + u**2)).real

    def _gap_average(self, f, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float) - self.xc, np.asarray(y, dtype=float))
        s = 0.5 * self.g * self._s
        vals = f(x[..., None] - s, y[..., None])
        return 0.5 * vals @ self._w

    def phi(self, x, y):
        return self.v1 + (self.v2 - self.v1) * (0.5 + self._gap_average(self._step, x, y))

    def efield(self, x, y):
        """Return ``(Ex, Ey) = -grad(phi)``."""
        dv = self.v2 - self.v1
        xr = np.asarray(x, dtype=float) - self.xc
        dwdx = (self._step(xr + self.g / 2, y) - self._step(xr - self.g / 2, y)) / self.g
        return -dv * dwdx, -dv * self._gap_average(self._step_dy, x, y)

    def plate_potential(self, x):
        """Linear-gap electrode potential on the plates, ``phi(x, ±h)``."""
        return np.interp(x, self.plate_nodes, (self.v1, self.v2))

    @property
    def plate_nodes(self):
        return (self.xc - self.g / 2, self.xc + self.g / 2)


def reference_trajectories(lens, markers, t_end, epsilon=1.0, rtol=1e-11, atol=1e-12):
    """Integrate ``dv/dt = E/epsilon`` in the exact field with a high-order RK method.

    ``markers`` has rows ``(x, y, vx, vy)``; returns a callable ``t -> (N, 4)``.
    """
    markers = np.asarray(markers, dtype=float)

    def rhs(_, state):
        x, y, vx, vy = state.reshape(4, -1)
        ex, ey = lens.efield(x, y)
        return np.concatenate([vx, vy, ex / epsilon, ey / epsilon])

    sol = solve_ivp(rhs, (0.0, t_end), markers.T.ravel(), method="DOP853", rtol=rtol, atol=atol, dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    return lambda t: sol.sol(t).reshape(4, len(markers)).T
