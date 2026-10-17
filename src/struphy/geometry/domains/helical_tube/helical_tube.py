import copy

import cunumpy as xp
import numpy as np

from struphy.geometry.base import Spline, interp_mapping


def helical_axis(zeta, R0, delta, delta_z, periods):
    """Centre curve ``((R0 + delta cos 2 pi m zeta) cos 2 pi zeta, ... sin 2 pi zeta, delta_z sin 2 pi m zeta)``."""
    rho = R0 + delta * np.cos(2 * np.pi * periods * zeta)
    return np.stack(
        [rho * np.cos(2 * np.pi * zeta), rho * np.sin(2 * np.pi * zeta), delta_z * np.sin(2 * np.pi * periods * zeta)]
    )


def axis_derivative(zeta, R0, delta, delta_z, periods):
    m = periods
    rho = R0 + delta * np.cos(2 * np.pi * m * zeta)
    drho = -2 * np.pi * m * delta * np.sin(2 * np.pi * m * zeta)
    c, s = np.cos(2 * np.pi * zeta), np.sin(2 * np.pi * zeta)
    return np.stack(
        [
            drho * c - 2 * np.pi * rho * s,
            drho * s + 2 * np.pi * rho * c,
            2 * np.pi * m * delta_z * np.cos(2 * np.pi * m * zeta),
        ]
    )


def rotating_frame(zeta, R0, delta, delta_z, periods, rotation):
    """The vertical normal frame ``N1 = T x e_z / |T x e_z|``, ``N2 = T x N1``, rotated by ``pi * rotation * zeta``."""
    T = axis_derivative(zeta, R0, delta, delta_z, periods)
    T = T / np.linalg.norm(T, axis=0)
    N1 = np.stack([T[1], -T[0], np.zeros_like(T[0])])
    N1 = N1 / np.linalg.norm(N1, axis=0)
    N2 = np.cross(T, N1, axis=0)
    c, s = np.cos(np.pi * rotation * zeta), np.sin(np.pi * rotation * zeta)
    return c * N1 + s * N2, c * N2 - s * N1


class HelicalTube(Spline):
    r"""Stellarator-like tube :math:`F(\eta_1, \eta_2, \eta_3) = \gamma(\eta_3) + N(\eta_3)\,\Phi_\textnormal{2D}(\eta_1, \eta_2)`
    around a helical axis, the 3D domain of the high-order polar-spline studies of R. Remmerswaal (PolarFEEC.jl
    ``TubeMapping``).

    * axis :math:`\gamma(\zeta) = ((R_0 + \delta\cos 2\pi m\zeta)\cos 2\pi\zeta, (R_0 + \delta\cos 2\pi m\zeta)\sin 2\pi\zeta,
      \delta_z \sin 2\pi m\zeta)` with ``m = periods``,
    * frame: the vertical normal frame :math:`N_1 = T \times e_z / |T \times e_z|`, :math:`N_2 = T \times N_1`, rotated by
      the angle :math:`\pi\,\textnormal{rotation}\,\zeta` (``rotation`` must be even for a periodic frame),
    * cross-section :math:`\Phi_\textnormal{2D} = R\,(a\cos 2\pi\eta_2, b\sin 2\pi\eta_2)`,
      :math:`R = \eta_1 (1 + \varepsilon\,\eta_1 \cos(2\pi k \eta_2))`.

    The presets of the talk: ``"stellarator"`` (the 3D wave and curl-curl studies: three periods, three-petal section of
    radius 0.3 turning once) and ``"islands"`` (PolarFEEC.jl example 07: four periods, ellipse 0.28 x 0.14 turning by
    half a turn per period). The mapping is interpolated by splines of the given degree on the given grid.

    Parameters
    ----------
    preset : str or None
        ``"stellarator"``, ``"islands"`` or ``None`` (use the remaining parameters).
    num_elements, degree : tuple[int, int, int]
        Spline grid of the interpolated mapping.
    R0, delta, delta_z, periods, rotation, a, b, eps, k : float
        Geometry parameters (ignored for a preset).
    """

    PRESETS = {
        "stellarator": dict(R0=1.0, delta=0.1, delta_z=0.1, periods=3, rotation=2, a=0.3, b=0.3, eps=0.2, k=3),
        "islands": dict(R0=1.0, delta=0.12, delta_z=0.12, periods=4, rotation=4, a=0.28, b=0.14, eps=0.0, k=0),
    }

    def __init__(
        self,
        preset: str | None = "stellarator",
        num_elements: tuple[int, int, int] = (16, 48, 96),
        degree: tuple[int, int, int] = (4, 4, 4),
        R0: float = 1.0,
        delta: float = 0.1,
        delta_z: float = 0.1,
        periods: int = 3,
        rotation: int = 2,
        a: float = 0.3,
        b: float = 0.3,
        eps: float = 0.2,
        k: int = 3,
    ):
        geometry = dict(R0=R0, delta=delta, delta_z=delta_z, periods=periods, rotation=rotation, a=a, b=b, eps=eps, k=k)
        if preset is not None:
            geometry = dict(self.PRESETS[preset])
        self.geometry = geometry
        g = geometry
        axis = (g["R0"], g["delta"], g["delta_z"], g["periods"])

        def section(e1, e2):
            radius = e1 * (1 + g["eps"] * e1 * np.cos(2 * np.pi * g["k"] * e2))
            return radius * g["a"] * np.cos(2 * np.pi * e2), radius * g["b"] * np.sin(2 * np.pi * e2)

        def component(i):
            def X(e1, e2, e3):
                if np.ndim(e1) == np.ndim(e2) == np.ndim(e3) == 1:
                    e1, e2, e3 = np.meshgrid(e1, e2, e3, indexing="ij")
                x2, y2 = section(e1, e2)
                N1, N2 = rotating_frame(e3, *axis, g["rotation"])
                return helical_axis(e3, *axis)[i] + N1[i] * x2 + N2[i] * y2

            return X

        spl_kind = (False, True, True)
        cx, cy, cz = interp_mapping(num_elements, degree, spl_kind, component(0), component(1), component(2))
        # the first ring of control points is the axis: make it exactly the same point for all eta2
        for c in (cx, cy, cz):
            c[0] = xp.mean(c[0], axis=0)[None, :]

        self._preset_params = copy.deepcopy(dict(preset=preset, **geometry))
        super().__init__(num_elements=num_elements, degree=degree, spl_kind=spl_kind, cx=cx, cy=cy, cz=cz)
