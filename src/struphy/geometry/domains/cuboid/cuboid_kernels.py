"""Pyccel kernels for the mapping and Jacobian matrix of :class:`struphy.geometry.domains.Cuboid`."""

from pyccel.decorators import pure


@pure
def cuboid(
    eta1: float,
    eta2: float,
    eta3: float,
    l1: float,
    r1: float,
    l2: float,
    r2: float,
    l3: float,
    r3: float,
    f_out: "float[:]",
):
    r"""
    Point-wise evaluation of

    .. math::

        F_x &= l_1 + (r_1 - l_1)\,\eta_1\,,

        F_y &= l_2 + (r_2 - l_2)\,\eta_2\,,

        F_z &= l_3 + (r_3 - l_3)\,\eta_3\,.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    l1, l2, l3 : float
        Left domain boundary.

    r1, r2, r3 : float
        Right domain boundary.

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    # value =  begin + (end - begin) * eta
    f_out[0] = l1 + (r1 - l1) * eta1
    f_out[1] = l2 + (r2 - l2) * eta2
    f_out[2] = l3 + (r3 - l3) * eta3


@pure
def cuboid_df(l1: float, r1: float, l2: float, r2: float, l3: float, r3: float, df_out: "float[:,:]"):
    """Jacobian matrix for :meth:`struphy.geometry.domains.cuboid.cuboid_kernels.cuboid`."""

    df_out[0, 0] = r1 - l1
    df_out[0, 1] = 0.0
    df_out[0, 2] = 0.0
    df_out[1, 0] = 0.0
    df_out[1, 1] = r2 - l2
    df_out[1, 2] = 0.0
    df_out[2, 0] = 0.0
    df_out[2, 1] = 0.0
    df_out[2, 2] = r3 - l3
