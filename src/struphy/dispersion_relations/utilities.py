import numpy as np
from scipy.special import erfi


def Zplasma(xi, der=0):
    """
    The plasma dispersion function and its first derivative.

    Parameters
    ----------
    xi : array_like
        Evaluation points.

    der : int, optional
        Whether to evaluate the plasma dispersion function (der = 0) or its first derivative (der = 1).

    Returns
    -------
    z : array_like
        Complex values of the plasma dispersion function (or its derivative).
    """

    assert der == 0 or der == 1, 'Parameter "der" must be either 0 or 1'

    if der == 0:
        z = np.sqrt(np.pi) * np.exp(-(xi**2)) * (1j - erfi(xi))
    else:
        z = -2 * (1 + xi * Zplasma(xi, 0))

    return z
