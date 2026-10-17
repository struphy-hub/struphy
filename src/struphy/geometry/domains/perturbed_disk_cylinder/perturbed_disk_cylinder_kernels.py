"""Pyccel kernels for the mapping and Jacobian matrix of :class:`struphy.geometry.domains.PerturbedDiskCylinder`."""

from numpy import cos, pi, sin
from pyccel.decorators import pure


@pure
def perturbed_disk(
    eta1: float, eta2: float, eta3: float, r1: float, eps: float, k: float, lz: float, q: float, f_out: "float[:]"
):
    r"""Point-wise evaluation of

    .. math::

        F_x &= R(\eta_1, \eta_2) \cos(2\pi\,\eta_2)\,, \quad F_y = R(\eta_1, \eta_2) \sin(2\pi\,\eta_2)\,,
        \quad F_z = L_z\,\eta_3\,,

        R &= R_1\,\eta_1\,\left(1 + \varepsilon\,\eta_1^q \cos(2\pi k\,\eta_2)\right)\,.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical coordinate in [0, 1].

    r1 : float
        Radius of the unperturbed disk.

    eps : float
        Amplitude of the perturbation.

    k : float
        Number of petals.

    lz : float
        Length in third direction.

    q : float
        Power of eta1 in the perturbation (1: the perturbed disk of the paper, 0: a star-shaped disk).

    f_out : array[float]
        Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    radius = r1 * eta1 * (1.0 + eps * eta1**q * cos(2 * pi * k * eta2))

    f_out[0] = radius * cos(2 * pi * eta2)
    f_out[1] = radius * sin(2 * pi * eta2)
    f_out[2] = lz * eta3


@pure
def perturbed_disk_df(
    eta1: float, eta2: float, r1: float, eps: float, k: float, lz: float, q: float, df_out: "float[:,:]"
):
    """Jacobian matrix for :meth:`perturbed_disk_cylinder_kernels.perturbed_disk`."""

    ck = cos(2 * pi * k * eta2)
    sk = sin(2 * pi * k * eta2)
    radius = r1 * eta1 * (1.0 + eps * eta1**q * ck)
    dr1 = r1 * (1.0 + (q + 1.0) * eps * eta1**q * ck)
    dr2 = -2 * pi * k * r1 * eps * eta1 ** (q + 1.0) * sk
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
