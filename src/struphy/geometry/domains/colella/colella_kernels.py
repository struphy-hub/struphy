"""Pyccel kernels for the mapping and Jacobian matrix of :class:`struphy.geometry.domains.Colella`."""

from numpy import cos, pi, sin
from pyccel.decorators import pure


@pure
def colella(eta1: float, eta2: float, eta3: float, lx: float, ly: float, alpha: float, lz: float, f_out: "float[:]"):
    r"""
    Point-wise evaluation of

    .. math::

        F_x &= L_x\,\left[\,\eta_1 + \\alpha\sin(2\pi\,\eta_1)\sin(2\pi\,\eta_2)\,\\right]\,,

        F_y &= L_y\,\left[\,\eta_2 + \\alpha\sin(2\pi\,\eta_2)\sin(2\pi\,\eta_1)\,\\right]\,,

        F_z &= L_z\,\eta_3\,.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    lx : float
        Length in x-direction.

    ly : float
        Length in y-direction.

    alpha : float
        Distortion factor.

    lz : float
        Length in z-direction.

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    f_out[0] = lx * (eta1 + alpha * sin(2 * pi * eta1) * sin(2 * pi * eta2))
    f_out[1] = ly * (eta2 + alpha * sin(2 * pi * eta1) * sin(2 * pi * eta2))
    f_out[2] = lz * eta3


@pure
def colella_df(eta1: float, eta2: float, lx: float, ly: float, alpha: float, lz: float, df_out: "float[:,:]"):
    """Jacobian matrix for :meth:`struphy.geometry.domains.colella.colella_kernels.colella`."""

    df_out[0, 0] = lx * (1 + alpha * cos(2 * pi * eta1) * sin(2 * pi * eta2) * 2 * pi)
    df_out[0, 1] = lx * alpha * sin(2 * pi * eta1) * cos(2 * pi * eta2) * 2 * pi
    df_out[0, 2] = 0.0
    df_out[1, 0] = ly * alpha * cos(2 * pi * eta1) * sin(2 * pi * eta2) * 2 * pi
    df_out[1, 1] = ly * (1 + alpha * sin(2 * pi * eta1) * cos(2 * pi * eta2) * 2 * pi)
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = lz
