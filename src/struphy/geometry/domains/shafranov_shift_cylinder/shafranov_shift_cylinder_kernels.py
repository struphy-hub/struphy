"""Pyccel kernels for the mapping and Jacobian matrix of :class:`struphy.geometry.domains.ShafranovShiftCylinder`."""

from numpy import cos, pi, sin
from pyccel.decorators import pure


@pure
def shafranov_shift(
    eta1: float,
    eta2: float,
    eta3: float,
    rx: float,
    ry: float,
    lz: float,
    de: float,
    f_out: "float[:]",
):
    r"""
    Point-wise evaluation of

    .. math::

        F_x &= r_x\,\eta_1\cos(2\pi\,\eta_2)+(1-\eta_1^2)r_x\Delta\,,

        F_y &= r_y\,\eta_1\sin(2\pi\,\eta_2)\,,

        F_z &= L_z\,\eta_3\,.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    rx, ry : float
        Axes lengths.

    lz : float
        Length in third direction.

    de : float
        Shift factor, should be in [0, 0.1].

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    f_out[0] = (eta1 * rx) * cos(2 * pi * eta2) + (1 - eta1**2) * rx * de
    f_out[1] = (eta1 * ry) * sin(2 * pi * eta2)
    f_out[2] = eta3 * lz


@pure
def shafranov_shift_df(
    eta1: float,
    eta2: float,
    eta3: float,
    rx: float,
    ry: float,
    lz: float,
    de: float,
    df_out: "float[:,:]",
):
    """Jacobian matrix for :meth:`struphy.geometry.domains.shafranov_shift_cylinder.shafranov_shift_cylinder_kernels.shafranov_shift`."""

    df_out[0, 0] = rx * cos(2 * pi * eta2) - 2 * eta1 * rx * de
    df_out[0, 1] = -2 * pi * (eta1 * rx) * sin(2 * pi * eta2)
    df_out[0, 2] = 0.0
    df_out[1, 0] = ry * sin(2 * pi * eta2)
    df_out[1, 1] = 2 * pi * (eta1 * ry) * cos(2 * pi * eta2)
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = lz
