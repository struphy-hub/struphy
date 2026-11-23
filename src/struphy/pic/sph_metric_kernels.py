"""Metric helpers for SPH kernels: the Jacobian determinant :math:`\\sqrt g = |\\det DF|` of the mapping at marker
positions and the gradient of :math:`\\ln \\sqrt g` in logical coordinates.

The SPH kernels call these helpers for the markers inside the unit cube. As a safeguard, positions of mirrored ghost
particles (at :math:`-\\eta` or :math:`2 - \\eta`) are folded back into the unit cube before the mapping is
evaluated. Mappings with a singular axis (:math:`\\det DF = 0` at :math:`\\eta_1 = 0`) are not supported."""

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
def grad_log_sqrt_g(
    eta1: float, eta2: float, eta3: float, args_domain: "DomainArguments", n_dirs: int, out: "float[:]"
):
    """Gradient of :math:`\\ln \\sqrt g` with respect to the first ``n_dirs`` logical coordinates, written to ``out``
    (the other components are set to zero). Central finite differences are used inside the unit cube and one-sided
    differences starting at :math:`\\eta` within ``delta`` of its boundary, so that the mapping is never evaluated
    on the boundary itself.

    Parameters
    ----------
    eta1, eta2, eta3 : float
        Logical position (ghost positions outside the unit cube are folded back).

    args_domain : DomainArguments
        Arguments of the mapping.

    n_dirs : int
        Number of directions (1, 2 or 3) in which the gradient is computed, i.e. those resolved by the SPH kernel.

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

    out[:] = 0.0
    for k in range(n_dirs):
        ep[:] = e
        em[:] = e
        if e[k] - delta < 0.0:
            ep[k] = e[k] + delta
        elif e[k] + delta > 1.0:
            em[k] = e[k] - delta
        else:
            ep[k] = e[k] + delta
            em[k] = e[k] - delta
        out[k] = (log(sqrt_g(ep[0], ep[1], ep[2], args_domain)) - log(sqrt_g(em[0], em[1], em[2], args_domain))) / (
            ep[k] - em[k]
        )
