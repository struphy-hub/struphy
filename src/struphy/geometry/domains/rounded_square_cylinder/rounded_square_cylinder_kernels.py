"""Pyccel kernels for the mapping and Jacobian matrix of :class:`struphy.geometry.domains.RoundedSquareCylinder`."""

from numpy import cos, pi, sin, sqrt
from pyccel.decorators import pure


@pure
def rounded_square(eta1: float, eta2: float, eta3: float, a: float, eps: float, lz: float, f_out: "float[:]"):
    r"""Point-wise evaluation of

    .. math::

        F_x &= a\,\eta_1\,\rho(\eta_2) \cos(2\pi\,\eta_2)\,, \quad F_y = a\,\eta_1\,\rho(\eta_2) \sin(2\pi\,\eta_2)\,,
        \quad F_z = L_z\,\eta_3\,,

        \rho &= \sqrt{2 / (1 + \sqrt{1 + 4\varepsilon \cos(8\pi\,\eta_2)})}\,,

    the rounded square :math:`1 - x^2 - y^2 - \varepsilon (x^4 - 6 x^2 y^2 + y^4) = 0` (for :math:`a = 1`).

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    a : float
        Scale of the cross-section.

    eps : float
        Squareness of the cross-section.

    lz : float
        Length in third direction.

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    q = sqrt(1.0 + 4.0 * eps * cos(8 * pi * eta2))
    rho = sqrt(2.0 / (1.0 + q))

    f_out[0] = a * eta1 * rho * cos(2 * pi * eta2)
    f_out[1] = a * eta1 * rho * sin(2 * pi * eta2)
    f_out[2] = lz * eta3


@pure
def rounded_square_df(eta1: float, eta2: float, a: float, eps: float, lz: float, df_out: "float[:,:]"):
    """Jacobian matrix for :meth:`rounded_square_cylinder_kernels.rounded_square`."""

    q = sqrt(1.0 + 4.0 * eps * cos(8 * pi * eta2))
    rho = sqrt(2.0 / (1.0 + q))
    drho = 16 * pi * eps * sin(8 * pi * eta2) / (rho * (1.0 + q) ** 2 * q)
    radius = a * eta1 * rho
    dr1 = a * rho
    dr2 = a * eta1 * drho
    c = cos(2 * pi * eta2)
    s = sin(2 * pi * eta2)

    df_out[0, 0] = dr1 * c
    df_out[0, 1] = dr2 * c - 2 * pi * radius * s
    df_out[0, 2] = 0.0
    df_out[1, 0] = dr1 * s
    df_out[1, 1] = dr2 * s + 2 * pi * radius * c
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = lz
