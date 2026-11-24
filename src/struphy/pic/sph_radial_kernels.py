r"""Radial SPH smoothing kernels in physical space.

Each kernel is a function of the physical distance :math:`r = |\mathbf x - \mathbf x'|` with compact support of radius
:math:`h`, normalized in :math:`d = 1, 2, 3` dimensions of physical space,

.. math::

    W_h(r) = \frac{\sigma_d}{h^d}\, w(q)\,, \qquad q = \frac rh \in [0, 1]\,, \qquad
    \int_{\mathbb R^d} W_h(|\mathbf x|)\, \textrm d \mathbf x = 1\,.

Kernel ids (``kernel``):

* ``1``, ``"gaussian"``: :math:`w(q) = e^{-9 q^2}` (standard deviation :math:`h/(3\sqrt 2)`), truncated at
  :math:`q = 1` and renormalized over the ball;
* ``2``, ``"wendland_c2"``: :math:`w(q) = (1 - q)^3 (1 + 3q)` for :math:`d = 1`, :math:`(1 - q)^4 (1 + 4q)` for
  :math:`d = 2, 3`;
* ``3``, ``"cubic_spline"`` (M4): :math:`w(q) = 1 - 6q^2 + 6q^3` for :math:`q \leq 1/2`, :math:`2(1 - q)^3` for
  :math:`1/2 < q \leq 1`.

The normalization constants :math:`\sigma_d` were checked by numerical integration (see the unit tests).
"""

from numpy import exp


def sigma(kernel: int, dim: int) -> float:
    """Normalization constant :math:`\\sigma_d` of the kernel shape for support radius 1.

    Parameters
    ----------
    kernel : int
        Kernel id (1: gaussian, 2: wendland_c2, 3: cubic_spline).

    dim : int
        Dimension of the physical space (1, 2 or 3).

    Returns
    -------
    float
        :math:`\\sigma_d`, such that :math:`\\sigma_d w(|\\mathbf x|)` integrates to 1 over the unit ball.
    """
    out = 0.0
    if kernel == 1:
        if dim == 1:
            out = 1.6926061411541502  # 3 / (sqrt(pi) erf(3))
        elif dim == 2:
            out = 2.865142562336415  # 9 / (pi (1 - exp(-9)))
        else:
            out = 4.850986001888355  # 1 / (4 pi (sqrt(pi) erf(3) / 108 - exp(-9) / 18))
    elif kernel == 2:
        if dim == 1:
            out = 1.25  # 5 / 4
        elif dim == 2:
            out = 2.228169203286535  # 7 / pi
        else:
            out = 3.342253804929802  # 21 / (2 pi)
    elif kernel == 3:
        if dim == 1:
            out = 1.3333333333333333  # 4 / 3
        elif dim == 2:
            out = 1.818913635335947  # 40 / (7 pi)
        else:
            out = 2.546479089470326  # 8 / pi
    return out


def shape(kernel: int, dim: int, q: float) -> float:
    """Kernel shape :math:`w(q)` for :math:`0 \\leq q \\leq 1` (zero for :math:`q > 1`).

    Parameters
    ----------
    kernel : int
        Kernel id (1: gaussian, 2: wendland_c2, 3: cubic_spline).

    dim : int
        Dimension of the physical space (1, 2 or 3).

    q : float
        Normalized distance :math:`r/h \\geq 0`.

    Returns
    -------
    float
        :math:`w(q)`.
    """
    out = 0.0
    if q > 1.0:
        return out
    if kernel == 1:
        out = exp(-9.0 * q * q)
    elif kernel == 2:
        if dim == 1:
            out = (1.0 - q) ** 3 * (1.0 + 3.0 * q)
        else:
            out = (1.0 - q) ** 4 * (1.0 + 4.0 * q)
    elif kernel == 3:
        if q <= 0.5:
            out = 1.0 - 6.0 * q * q + 6.0 * q * q * q
        else:
            out = 2.0 * (1.0 - q) ** 3
    return out


def dshape(kernel: int, dim: int, q: float) -> float:
    """Derivative :math:`w'(q)` of the kernel shape for :math:`0 \\leq q \\leq 1` (zero for :math:`q > 1`).

    Parameters
    ----------
    kernel : int
        Kernel id (1: gaussian, 2: wendland_c2, 3: cubic_spline).

    dim : int
        Dimension of the physical space (1, 2 or 3).

    q : float
        Normalized distance :math:`r/h \\geq 0`.

    Returns
    -------
    float
        :math:`w'(q)`.
    """
    out = 0.0
    if q > 1.0:
        return out
    if kernel == 1:
        out = -18.0 * q * exp(-9.0 * q * q)
    elif kernel == 2:
        if dim == 1:
            out = -12.0 * q * (1.0 - q) ** 2
        else:
            out = -20.0 * q * (1.0 - q) ** 3
    elif kernel == 3:
        if q <= 0.5:
            out = -12.0 * q + 18.0 * q * q
        else:
            out = -6.0 * (1.0 - q) ** 2
    return out


def kernel_value(kernel: int, dim: int, r: float, h: float) -> float:
    """Kernel value :math:`W_h(r) = \\sigma_d\\, w(r/h) / h^d`.

    Parameters
    ----------
    kernel : int
        Kernel id (1: gaussian, 2: wendland_c2, 3: cubic_spline).

    dim : int
        Dimension of the physical space (1, 2 or 3).

    r : float
        Physical distance :math:`\\geq 0`.

    h : float
        Kernel width (support radius) in physical units.

    Returns
    -------
    float
        :math:`W_h(r)`.
    """
    q = r / h
    if q > 1.0:
        return 0.0
    return sigma(kernel, dim) / h**dim * shape(kernel, dim, q)


def kernel_dr(kernel: int, dim: int, r: float, h: float) -> float:
    """Radial derivative :math:`\\partial_r W_h(r) = \\sigma_d\\, w'(r/h) / h^{d+1}`; the gradient is
    :math:`\\nabla W_h(\\mathbf x) = \\partial_r W_h(r)\\, \\mathbf x / r` (zero at :math:`r = 0`).

    Parameters
    ----------
    kernel : int
        Kernel id (1: gaussian, 2: wendland_c2, 3: cubic_spline).

    dim : int
        Dimension of the physical space (1, 2 or 3).

    r : float
        Physical distance :math:`\\geq 0`.

    h : float
        Kernel width (support radius) in physical units.

    Returns
    -------
    float
        :math:`\\partial_r W_h(r)`.
    """
    q = r / h
    if q > 1.0:
        return 0.0
    return sigma(kernel, dim) / h ** (dim + 1) * dshape(kernel, dim, q)
