"""Pyccel kernels for the mapping and Jacobian matrix of :class:`struphy.geometry.domains.HollowCylinder`."""

from numpy import cos, pi, sin
from pyccel.decorators import pure


@pure
def hollow_cyl(eta1: float, eta2: float, eta3: float, a1: float, a2: float, lz: float, poc: float, f_out: "float[:]"):
    r"""Point-wise evaluation of

    .. math::

        F_x &= \left[\,a_1 + (a_2-a_1)\,\eta_1\,\\right]\cos(2\pi\,\eta_2 / poc)\,,

        F_y &= \left[\,a_1 + (a_2-a_1)\,\eta_1\,\\right]\sin(2\pi\,\eta_2 / poc)\,,

        F_z &= L_z\,\eta_3\,.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    a1 : float
        Inner radius.

    a2 : float
        Outer radius.

    lz : float
        Length in third direction.

    poc : int
        periodicity in second direction.

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    da = a2 - a1

    f_out[0] = (a1 + eta1 * da) * cos(2 * pi * eta2 / poc)
    f_out[1] = (a1 + eta1 * da) * sin(2 * pi * eta2 / poc)
    f_out[2] = lz * eta3


@pure
def hollow_cyl_df(eta1: float, eta2: float, a1: float, a2: float, lz: float, poc: float, df_out: "float[:,:]"):
    """Jacobian matrix for :meth:`struphy.geometry.domains.hollow_cylinder.hollow_cylinder_kernels.hollow_cyl`."""

    da = a2 - a1

    df_out[0, 0] = da * cos(2 * pi * eta2 / poc)
    df_out[0, 1] = -2 * pi / poc * (a1 + eta1 * da) * sin(2 * pi * eta2 / poc)
    df_out[0, 2] = 0.0
    df_out[1, 0] = da * sin(2 * pi * eta2 / poc)
    df_out[1, 1] = 2 * pi / poc * (a1 + eta1 * da) * cos(2 * pi * eta2 / poc)
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = lz
