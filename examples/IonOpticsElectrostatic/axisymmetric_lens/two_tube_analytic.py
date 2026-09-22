"""Analytic potential of the two-tube (cylinder) immersion lens.

Two coaxial tubes of radius R meet at z = zc; upstream the wall is at v1, downstream
at v2, and across a gap of width g the wall potential rises linearly. For a zero gap,

    w(r, z) = sgn(z)/2 * (1 - sum_n 2 J0(j_n r/R) exp(-j_n |z| / R) / (j_n J1(j_n))),

where j_n are the zeros of J0. The linear gap is the average of w over the gap
(Gauss–Legendre), exactly as for the slit lens.
"""

import numpy as np
from scipy.special import j0, j1, jn_zeros


class TwoTubeLens:
    def __init__(self, radius, gap, zc, v1, v2, n_terms=400, n_quad=64):
        self.R, self.g, self.zc, self.v1, self.v2 = radius, gap, zc, v1, v2
        self._zeros = jn_zeros(0, n_terms)
        self._coef = 2.0 / (self._zeros * j1(self._zeros))
        self._s, self._w = np.polynomial.legendre.leggauss(n_quad)

    def _series(self, r, z, derivative=None):
        k = self._zeros / self.R
        rz = np.abs(z)[..., None]
        decay = np.exp(-k * rz)
        radial = j0(k * np.asarray(r)[..., None])
        if derivative is None:
            return 0.5 * np.sign(z) * (1.0 - np.sum(self._coef * radial * decay, axis=-1))
        if derivative == "r":
            return 0.5 * np.sign(z) * np.sum(self._coef * k * j1(k * np.asarray(r)[..., None]) * decay, axis=-1)
        raise ValueError(derivative)

    def _gap_average(self, r, z, derivative=None):
        r, z = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(z, dtype=float) - self.zc)
        s = 0.5 * self.g * self._s
        vals = self._series(r[..., None], z[..., None] - s, derivative)
        return 0.5 * vals @ self._w

    def phi(self, r, z):
        return self.v1 + (self.v2 - self.v1) * (0.5 + self._gap_average(r, z))

    def efield(self, r, z):
        """(E_r, E_z) = -grad(phi)."""
        dv = self.v2 - self.v1
        zr = np.asarray(z, dtype=float) - self.zc
        dwdz = (self._series(r, zr + self.g / 2) - self._series(r, zr - self.g / 2)) / self.g
        return -dv * self._gap_average(r, z, "r"), -dv * dwdz

    def wall_potential(self, z):
        return np.interp(z, (self.zc - self.g / 2, self.zc + self.g / 2), (self.v1, self.v2))
