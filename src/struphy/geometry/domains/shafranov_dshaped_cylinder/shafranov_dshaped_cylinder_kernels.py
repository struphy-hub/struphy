"""Pyccel kernels for the mapping and Jacobian matrix of :class:`struphy.geometry.domains.ShafranovDshapedCylinder`."""

from numpy import arcsin, cos, pi, sin
from pyccel.decorators import pure


@pure
def shafranov_dshaped(
    eta1: float,
    eta2: float,
    eta3: float,
    r0: float,
    lz: float,
    dx: float,
    dy: float,
    dg: float,
    eg: float,
    kg: float,
    f_out: "float[:]",
):
    r"""
    Point-wise evaluation of

    .. math::

        x &= R_0\left[1 + (1 - \eta_1^2)\Delta_x + \eta_1\epsilon\cos(2\pi\,\eta_2 + \\arcsin(\delta)\eta_1\sin(2\pi\,\eta_2)) \\right]\,,

        y &= R_0\left[    (1 - \eta_1^2)\Delta_y + \eta_1\epsilon\kappa\sin(2\pi\,\eta_2)\\right]\,,

        z &= L_z\,\eta_3\,.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    r0 : float
        Base radius.

    lz : float
        Length in third direction.

    dx : float
        Shafranov shift in x-direction.

    dy : float
        Shafranov shift in y-direction.

    dg : float
        Delta = sin(alpha): Triangularity, shift of high point.

    eg : float
        Epsilon: Inverse aspect ratio a/r0.

    kg : float
        Kappa: Ellipticity (elongation).

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    f_out[0] = r0 * (1 + (1 - eta1**2) * dx + eg * eta1 * cos(2 * pi * eta2 + arcsin(dg) * eta1 * sin(2 * pi * eta2)))
    f_out[1] = r0 * ((1 - eta1**2) * dy + eg * kg * eta1 * sin(2 * pi * eta2))
    f_out[2] = eta3 * lz


@pure
def shafranov_dshaped_df(
    eta1: float,
    eta2: float,
    eta3: float,
    r0: float,
    lz: float,
    dx: float,
    dy: float,
    dg: float,
    eg: float,
    kg: float,
    df_out: "float[:,:]",
):
    """Jacobian matrix for :meth:`struphy.geometry.domains.shafranov_dshaped_cylinder.shafranov_dshaped_cylinder_kernels.shafranov_dshaped`."""

    df_out[0, 0] = r0 * (
        -2 * dx * eta1
        - eg * eta1 * sin(2 * pi * eta2) * arcsin(dg) * sin(eta1 * sin(2 * pi * eta2) * arcsin(dg) + 2 * pi * eta2)
        + eg * cos(eta1 * sin(2 * pi * eta2) * arcsin(dg) + 2 * pi * eta2)
    )
    df_out[0, 1] = (
        -r0
        * eg
        * eta1
        * (2 * pi * eta1 * cos(2 * pi * eta2) * arcsin(dg) + 2 * pi)
        * sin(eta1 * sin(2 * pi * eta2) * arcsin(dg) + 2 * pi * eta2)
    )
    df_out[0, 2] = 0.0
    df_out[1, 0] = r0 * (-2 * dy * eta1 + eg * kg * sin(2 * pi * eta2))
    df_out[1, 1] = 2 * pi * r0 * eg * eta1 * kg * cos(2 * pi * eta2)
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = lz
