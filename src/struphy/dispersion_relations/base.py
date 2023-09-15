'Base classes for dispersion relations.'


from abc import ABCMeta, abstractmethod

import numpy as np

from scipy.special import erfi


class DispersionRelations1D(metaclass=ABCMeta):
    """
    The base class for analytic 1d dispersion relations.

    Parameters
    ----------
    *branch_names
        Branche names (str) of the spectrum.

    **params
        Physical parameters necessary to compute the dispersion relation.

    Notes
    -----
    Analytic Struphy dispersion relations are subclasses of ``DispersionRelations1D`` and should be added to ``struphy/models/dispersion_relations/analytic.py``. 
    """

    def __init__(self, *branch_names, **params):

        self._branches = branch_names
        self._nbranches = len(branch_names)
        self._params = params

    @property
    def branches(self):
        """ List of branch names in the spectrum.
        """
        return self._branches

    @property
    def nbranches(self):
        """ Integer : number of branches.
        """
        return self._nbranches

    @property
    def params(self):
        """ Dictionary of parameters necessary to compute the dispersion relation.
        """
        return self._params

    @abstractmethod
    def __call__(self, k):
        """
        The evaluation of all branches of a 1d dispersion relation.

        Parameters
        ----------
        k : array_like
            Evaluation wave numbers.

        Returns
        -------
        omegas : dict
            A dictionary with key=branch_name and value=omega(k) (complex ndarray).
        """


class ContinuousSpectra1D(metaclass=ABCMeta):
    """
    The base class for analytical continuous spectra in one spatial dimension.

    Parameters
    ----------
    *branch_names
        Names (str) of the continuous spectra.

    **params
        Physical parameters necessary to compute the continuous spectra (e.g. profiles for magnetic field components).
    """

    def __init__(self, *branch_names, **params):

        self._branches = branch_names
        self._nbranches = len(branch_names)
        self._params = params

    @property
    def branches(self):
        """ List of branch names in the spectrum.
        """
        return self._branches

    @property
    def nbranches(self):
        """ Integer : number of branches.
        """
        return self._nbranches

    @property
    def params(self):
        """ Dictionary of parameters necessary to compute the continuous spectra.
        """
        return self._params

    @abstractmethod
    def __call__(self, x, *mode_numbers):
        """ 
        The evaluation of all continuous spectra.

        Parameters
        ----------
        x : array_like
            The spatial coordinates at which the continuous spectra shall be evaluated.

        mode_number1, mode_number2, ... : int
            Mode numbers of a Fourier representation of other spatial directions (e.g. poloidal and toroidal).

        Returns
        -------
        specs : dict
            A dictionary with key=branch_name and value=omega(x) (ndarray).
        """


# somer helper function used in different dispersion relations:
# -------------------------------------------------------------
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
        z = np.sqrt(np.pi)*np.exp(-xi**2)*(1j - erfi(xi))
    else:
        z = -2*(1 + xi*Zplasma(xi, 0))

    return z
