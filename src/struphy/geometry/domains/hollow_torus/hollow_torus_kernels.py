"""Pyccel kernels for the mapping and Jacobian matrix of :class:`struphy.geometry.domains.HollowTorus`."""

from numpy import arctan, cos, pi, sin, sqrt, tan
from pyccel.decorators import pure


@pure
def hollow_torus(
    eta1: float,
    eta2: float,
    eta3: float,
    a1: float,
    a2: float,
    r0: float,
    sfl: float,
    pol_period: float,
    tor_period: float,
    f_out: "float[:]",
):
    r"""
    Point-wise evaluation of

    .. math::

        F_x &= \lbrace\left[\,a_1 + (a_2-a_1)\,\eta_1\,\\right]\cos(\theta(\eta_1,\eta_2))+R_0\\rbrace\cos(2\pi\,\eta_3)\,,

        F_y &= \lbrace\left[\,a_1 + (a_2-a_1)\,\eta_1\,\\right]\cos(\theta(\eta_1,\eta_2))+R_0\\rbrace\sin(2\pi\,\eta_3) \,,

        F_z &= \,\,\,\left[\,a_1 + (a_2-a_1)\,\eta_1\,\\right]\sin(\theta(\eta_1,\eta_2)) \,,

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    a1 : float
        Inner radius.

    a2 : float
        Outer radius.

    r0 : float
        Major radius.

    sfl : float
        Whether to use straight field line angular parametrization (yes: 1., no: 0.).

    pol_period: float
        periodicity of theta used in the mapping: theta = 2*pi * eta2 / pol_period (if not sfl)

    tor_period : int
        Toroidal periodicity built into the mapping: phi = 2*pi * eta3 / tor_period

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    # straight field lines coordinates
    if sfl == 1.0:
        da = a2 - a1

        r = a1 + eta1 * da
        theta = 2 * arctan(sqrt((1 + r / r0) / (1 - r / r0)) * tan(pi * eta2))

        f_out[0] = (r * cos(theta) + r0) * cos(2 * pi * eta3 / tor_period)
        f_out[1] = (r * cos(theta) + r0) * (-1) * sin(2 * pi * eta3 / tor_period)
        f_out[2] = r * sin(theta)

    # equal angle coordinates
    else:
        da = a2 - a1

        f_out[0] = ((a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) + r0) * cos(2 * pi * eta3 / tor_period)
        f_out[1] = ((a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) + r0) * (-1) * sin(2 * pi * eta3 / tor_period)
        f_out[2] = (a1 + eta1 * da) * sin(2 * pi * eta2 / pol_period)


@pure
def hollow_torus_df(
    eta1: float,
    eta2: float,
    eta3: float,
    a1: float,
    a2: float,
    r0: float,
    sfl: float,
    pol_period: float,
    tor_period: float,
    df_out: "float[:,:]",
):
    """Jacobian matrix for :meth:`struphy.geometry.domains.hollow_torus.hollow_torus_kernels.hollow_torus`."""

    # straight field lines coordinates
    if sfl == 1.0:
        da = a2 - a1

        r = a1 + da * eta1

        eps = r / r0
        eps_p = da / r0

        tpe = tan(pi * eta2)
        tpe_p = pi / cos(pi * eta2) ** 2

        g = sqrt((1 + eps) / (1 - eps))
        g_p = 1 / (2 * g) * (eps_p * (1 - eps) + (1 + eps) * eps_p) / (1 - eps) ** 2

        theta = 2 * arctan(g * tpe)

        dtheta_deta1 = 2 / (1 + (g * tpe) ** 2) * g_p * tpe
        dtheta_deta2 = 2 / (1 + (g * tpe) ** 2) * g * tpe_p

        df_out[0, 0] = (da * cos(theta) - r * sin(theta) * dtheta_deta1) * cos(2 * pi * eta3 / tor_period)
        df_out[0, 1] = -r * sin(theta) * dtheta_deta2 * cos(2 * pi * eta3 / tor_period)
        df_out[0, 2] = -2 * pi / tor_period * (r * cos(theta) + r0) * sin(2 * pi * eta3 / tor_period)

        df_out[1, 0] = (da * cos(theta) - r * sin(theta) * dtheta_deta1) * (-1) * sin(2 * pi * eta3 / tor_period)
        df_out[1, 1] = -r * sin(theta) * dtheta_deta2 * (-1) * sin(2 * pi * eta3 / tor_period)
        df_out[1, 2] = 2 * pi / tor_period * (r * cos(theta) + r0) * (-1) * cos(2 * pi * eta3 / tor_period)

        df_out[2, 0] = da * sin(theta) + r * cos(theta) * dtheta_deta1
        df_out[2, 1] = r * cos(theta) * dtheta_deta2
        df_out[2, 2] = 0.0

    # equal angle coordinates
    else:
        da = a2 - a1

        df_out[0, 0] = da * cos(2 * pi * eta2 / pol_period) * cos(2 * pi * eta3 / tor_period)
        df_out[0, 1] = (
            -2 * pi / pol_period * (a1 + eta1 * da) * sin(2 * pi * eta2 / pol_period) * cos(2 * pi * eta3 / tor_period)
        )
        df_out[0, 2] = (
            -2
            * pi
            / tor_period
            * ((a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) + r0)
            * sin(2 * pi * eta3 / tor_period)
        )
        df_out[1, 0] = da * cos(2 * pi * eta2 / pol_period) * (-1) * sin(2 * pi * eta3 / tor_period)
        df_out[1, 1] = (
            -2
            * pi
            / pol_period
            * (a1 + eta1 * da)
            * sin(2 * pi * eta2 / pol_period)
            * (-1)
            * sin(2 * pi * eta3 / tor_period)
        )
        df_out[1, 2] = (
            ((a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) + r0)
            * (-1)
            * cos(2 * pi * eta3 / tor_period)
            * 2
            * pi
            / tor_period
        )
        df_out[2, 0] = da * sin(2 * pi * eta2 / pol_period)
        df_out[2, 1] = (a1 + eta1 * da) * cos(2 * pi * eta2 / pol_period) * 2 * pi / pol_period
        df_out[2, 2] = 0.0
