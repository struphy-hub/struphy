"""Small axisymmetric flux providers for constructing geometry without an MHD solver."""

from pathlib import Path
import re

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root


class CircularFlux:
    """Analytic circular flux surfaces: psi = ((R - R0)**2 + (Z - Z0)**2) / a**2.

    Implements the flux interface accepted by ``Tokamak``. This is a geometry
    example, not a force-balanced MHD equilibrium.
    """

    def __init__(self, R0=3.0, Z0=0.0, a=1.0):
        if not np.isfinite([R0, Z0, a]).all() or a <= 0 or R0 <= a:
            raise ValueError("Require finite parameters and R0 > a > 0")
        self.R0, self.Z0, self.a = float(R0), float(Z0), float(a)
        self.psi_axis_RZ = (self.R0, self.Z0)
        self.psi_range = (0.0, 1.0)

    def psi(self, R, Z, dR=0, dZ=0):
        R, Z = np.broadcast_arrays(np.asarray(R), np.asarray(Z))
        if dR < 0 or dZ < 0 or int(dR) != dR or int(dZ) != dZ:
            raise ValueError("Derivative orders must be non-negative integers")
        if dR == dZ == 0:
            out = ((R - self.R0) ** 2 + (Z - self.Z0) ** 2) / self.a**2
        elif (dR, dZ) == (1, 0):
            out = 2 * (R - self.R0) / self.a**2
        elif (dR, dZ) == (0, 1):
            out = 2 * (Z - self.Z0) / self.a**2
        elif (dR, dZ) in ((2, 0), (0, 2)):
            out = np.full(R.shape, 2 / self.a**2)
        else:
            out = np.zeros(R.shape)
        return out.item() if out.ndim == 0 else out


class EQDSKFlux:
    """Poloidal flux from a G-EQDSK file, with no unit rescaling or smoothing.

    Use ``from_text`` with an uploaded browser file. The reader accepts E/D
    scientific notation and reads the rectangular flux grid and axis metadata;
    pressure, current, and boundary contour data are not interpreted. The
    interpolated magnetic axis is refined by default, for either flux sign.
    """

    def __init__(self, filename, *, refine_axis=True):
        self._read(Path(filename).read_text(), refine_axis=refine_axis)

    @classmethod
    def from_text(cls, text, *, refine_axis=True):
        obj = cls.__new__(cls)
        obj._read(text, refine_axis=refine_axis)
        return obj

    def _read(self, text, *, refine_axis):
        lines = text.splitlines()
        try:
            nR, nZ = map(int, lines[0].split()[-2:])
        except (IndexError, ValueError) as exc:
            raise ValueError("Invalid G-EQDSK header") from exc
        if nR < 4 or nZ < 4:
            raise ValueError("G-EQDSK flux grid must have at least four points per axis")
        values = np.array(
            [
                float(s.replace("D", "E").replace("d", "e"))
                for s in re.findall(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)[EeDd][+-]?\d+", "\n".join(lines[1:]))
            ]
        )
        required = 20 + 4 * nR + nR * nZ
        if values.size < required or not np.isfinite(values[:required]).all():
            raise ValueError("Incomplete or non-finite G-EQDSK flux data")
        rdim, zdim, _, rleft, zmid = values[:5]
        Raxis, Zaxis, _, psi_edge = values[5:9]
        if rdim <= 0 or zdim <= 0:
            raise ValueError("G-EQDSK grid dimensions must be positive")
        self.R = np.linspace(rleft, rleft + rdim, nR)
        self.Z = np.linspace(zmid - zdim / 2, zmid + zdim / 2, nZ)
        grid = values[20 + 4 * nR : required].reshape(nZ, nR).T
        self._spline = RectBivariateSpline(self.R, self.Z, grid, kx=3, ky=3, s=0)
        axis = np.array([Raxis, Zaxis])
        if refine_axis:
            solution = root(lambda q: [self.psi(*q, dR=1), self.psi(*q, dZ=1)], axis)
            if not solution.success:
                raise ValueError("Could not locate the interpolated magnetic axis")
            axis = solution.x
        if not (self.R[0] < axis[0] < self.R[-1] and self.Z[0] < axis[1] < self.Z[-1]):
            raise ValueError("Magnetic axis is outside the flux grid")
        self.psi_axis_RZ = tuple(float(x) for x in axis)
        self.psi_range = (float(self.psi(*axis)), float(psi_edge))
        if self.psi_range[0] == self.psi_range[1]:
            raise ValueError("Axis and edge flux must differ")

    def psi(self, R, Z, dR=0, dZ=0):
        """Evaluate flux or derivatives at broadcast-compatible R/Z coordinates."""
        R, Z = np.broadcast_arrays(np.asarray(R), np.asarray(Z))
        # FITPACK wrappers differ in scalar output shape across SciPy versions.
        # Evaluate flat point lists and restore our explicit broadcasting contract.
        out = np.asarray(self._spline.ev(R.ravel(), Z.ravel(), dx=dR, dy=dZ)).reshape(R.shape)
        return out.item() if out.ndim == 0 else out
