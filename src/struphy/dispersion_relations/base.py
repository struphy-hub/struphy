"Base classes for dispersion relations."

from abc import ABCMeta, abstractmethod

from matplotlib import pyplot as plt

from struphy.utils.arrays import xp as np


class DispersionRelations1D(metaclass=ABCMeta):
    r"""
    The base class for analytic 1d dispersion relations of the form
    :math:`\omega_i(k) \in \mathbb C`, where :math:`i=1,\ldots,n`
    denote the differnt branches of the spectrum.

    Parameters
    ----------
    branch_names : str
        Branche names (str) of the spectrum.

    velocity_scale : str
        Determines the unit of :math:`\omega`.
        Must be one of ``alfvén``, ``cyclotron`` or ``light``.

    Notes
    -----
    Analytic dispersion relations should be added to
    :mod:`~struphy.dispersion_relations.analytic`.
    """

    def __init__(self, *branch_names, velocity_scale="alfvén", **params):
        self._branches = {}
        for name in branch_names:
            self._branches[name] = None

        self._velocity_scale = velocity_scale
        self._params = params

        # critical k-values
        self._k_crit = {}

    @property
    def branches(self):
        r"""Dictionary of branch names holding the numpy arrays of :math:`\omega_i(k) \in \mathbb C`."""
        return self._branches

    @property
    def params(self):
        """Dictionary of parameters necessary to compute the dispersion relation."""
        return self._params

    @property
    def velocity_scale(self):
        r"""Determines the unit of :math:`\omega`.
        Must be one of ``alfvén``, ``cyclotron`` or ``light``.
        """
        return self._velocity_scale

    @property
    def k_crit(self):
        """Dictionary of critical k-values (plotted as vertical lines with self.plot())."""
        return self._k_crit

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

    def plot(self, k):
        self(k)

        if self.velocity_scale == "alfvén":
            unit_om = r"$v_A / \hat x$"
        elif self.velocity_scale == "light":
            unit_om = r"$c / \hat x$"
        elif self.velocity_scale == "cyclotron":
            unit_om = r"$\Omega_c$"
        elif self.velocity_scale == "thermal":
            unit_om = r"$v_{th} / \hat x$"

        plt.figure(figsize=(5, 8))
        plt.subplot(2, 1, 1)
        # plt.title('Real part')
        plt.xlabel(r"k [$1 / \hat x$]")
        plt.ylabel(rf"Re($\omega$) [{unit_om}]")
        plt.subplot(2, 1, 2)
        # plt.title('Imaginary part')
        plt.xlabel(r"k [$1 / \hat x$]")
        plt.ylabel(rf"Im($\omega$) [{unit_om}]")
        for name, omega in self.branches.items():
            plt.subplot(2, 1, 1)
            plt.plot(k, np.real(omega), label=name)
            plt.subplot(2, 1, 2)
            plt.plot(k, np.imag(omega), label=name)

        plt.subplot(2, 1, 1)
        for lab, kc in self.k_crit.items():
            if kc > np.min(k) and kc < np.max(k):
                plt.axvline(kc, color="k", linestyle="--", linewidth=0.5, label=lab)
        plt.legend()
        plt.subplot(2, 1, 2)
        for lab, kc in self.k_crit.items():
            if kc > np.min(k) and kc < np.max(k):
                plt.axvline(kc, color="k", linestyle="--", linewidth=0.5, label=lab)


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
        """List of branch names in the spectrum."""
        return self._branches

    @property
    def nbranches(self):
        """Integer : number of branches."""
        return self._nbranches

    @property
    def params(self):
        """Dictionary of parameters necessary to compute the continuous spectra."""
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
