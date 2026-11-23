"""Metric helpers for SPH kernels: the Jacobian determinant :math:`\\sqrt g = |\\det DF|` of the mapping at marker
positions and the gradient of :math:`\\ln \\sqrt g` in logical coordinates.

Ghost particles of mirror-type boundary conditions sit outside the unit cube (at :math:`-\\eta` or :math:`2 - \\eta`);
they are folded back into the unit cube before the mapping is evaluated, which gives the value at the mirrored
marker."""

from numpy import log, zeros
from pyccel.decorators import stack_array

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments


def fold(eta: float) -> float:
    """Map a logical coordinate of a mirrored ghost particle back into the unit interval.

    Parameters
    ----------
    eta : float
        Logical coordinate, in (-1, 2).

    Returns
    -------
    float
        ``-eta`` if ``eta < 0``, ``2 - eta`` if ``eta > 1``, else ``eta``.
    """
    if eta < 0.0:
        return -eta
    elif eta > 1.0:
        return 2.0 - eta
    return eta


@stack_array("tmp")
def sqrt_g(eta1: float, eta2: float, eta3: float, args_domain: "DomainArguments") -> float:
    """Jacobian determinant :math:`\\sqrt g = |\\det DF|` at the (folded) logical position.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical position (ghost positions outside the unit cube are folded back).

    args_domain : DomainArguments
        Arguments of the mapping.

    Returns
    -------
    float
        :math:`|\\det DF(\\boldsymbol \\eta)|`.
    """
    tmp = zeros((3, 3), dtype=float)
    det = evaluation_kernels.det_df(fold(eta1), fold(eta2), fold(eta3), args_domain, tmp)
    if det < 0.0:
        det = -det
    return det


@stack_array("e", "ep", "em")
def grad_log_sqrt_g(eta1: float, eta2: float, eta3: float, args_domain: "DomainArguments", out: "float[:]"):
    """Gradient of :math:`\\ln \\sqrt g` with respect to the logical coordinates, by central finite differences
    (one-sided at the boundaries of the unit cube), written to ``out``.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical position (ghost positions outside the unit cube are folded back).

    args_domain : DomainArguments
        Arguments of the mapping.

    out : array[float]
        Output array of length 3.
    """
    delta = 1e-6

    e = zeros(3, dtype=float)
    ep = zeros(3, dtype=float)
    em = zeros(3, dtype=float)

    e[0] = fold(eta1)
    e[1] = fold(eta2)
    e[2] = fold(eta3)

    for k in range(3):
        ep[:] = e
        em[:] = e
        ep[k] = e[k] + delta
        em[k] = e[k] - delta
        if ep[k] > 1.0:
            ep[k] = 1.0
        if em[k] < 0.0:
            em[k] = 0.0
        out[k] = (log(sqrt_g(ep[0], ep[1], ep[2], args_domain)) - log(sqrt_g(em[0], em[1], em[2], args_domain))) / (
            ep[k] - em[k]
        )
