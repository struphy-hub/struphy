'Base classes for kinetic backgrounds.'


from abc import ABCMeta, abstractmethod
import numpy as np


class Maxwellian(metaclass=ABCMeta):
    r"""
    Base class for a Maxwellian distribution function defined on :math:`[0, 1]^3 \times \mathbb R^n, n \geq 1,` 
    with logical position coordinates :math:`\boldsymbol{\eta} \in [0, 1]^3`:

    .. math::

        f(\boldsymbol{\eta}, v_1,\ldots,v_n) = n(\boldsymbol{\eta}) \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\,v_{\mathrm{th},i}(\boldsymbol{\eta})}
        \exp\left[-\frac{(v_i-u_i(\boldsymbol{\eta}))^2}{2\,v_{\mathrm{th},i}(\boldsymbol{\eta})^2}\right],

    defined by its velocity moments: the density :math:`n(\boldsymbol{\eta})`,
    the mean-velocities :math:`u_i(\boldsymbol{\eta})`,
    and the thermal velocities :math:`v_{\mathrm{th},i}(\boldsymbol{\eta})`.
    """

    @property
    @abstractmethod
    def vdim(self):
        """Dimension of the velocity space (vdim = n).
        """
        pass

    @property
    @abstractmethod
    def is_polar(self):
        """List of booleans. True if the velocity coordinates are polar coordinates.
        """
        pass

    @abstractmethod
    def n(self, *etas):
        """ Number density (0-form). 

        Parameters
        ----------
        etas : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the density evaluated at evaluation points (same shape as etas).
        """
        pass

    @abstractmethod
    def u(self, *etas):
        """ Mean velocities (Cartesian components evaluated at x = F(eta)).

        Parameters
        ----------
        etas : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the mean velocity evaluated at evaluation points (one dimension more than etas).
        The additional dimension is in the first index.
        """
        pass

    @abstractmethod
    def vth(self, *etas):
        """ Thermal velocities (0-forms).

        Parameters
        ----------
        etas : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the thermal velocity evaluated at evaluation points (one dimension more than etas).
        The additional dimension is in the first index.
        """
        pass

    def gaussian(self, v, u=0., vth=1., is_polar=False):
        """n-D normal distribution, to which array-valued mean- and thermal velocities can be passed.

        Parameters
        ----------
        v : float | array-like
            Velocity coordinate(s).

        u : float | array-like
            Mean velocity evaluated at position array.

        vth : float | array-like
            Thermal velocity evaluated at position array, same shape as u.

        is_polar : boolean
            True if the velocity coordinates are polar coordinates.

        Returns
        -------
        An array of size(u).
        """

        if isinstance(v, np.ndarray) and isinstance(u, np.ndarray):
            assert v.shape == u.shape

        if not is_polar:
            return 1./(np.sqrt(2.*np.pi) * vth) * np.exp(-(v - u)**2/(2.*vth**2))
        else:
            return 1./vth**2 * v * np.exp(-(v - u)**2/(2.*vth**2))


    def __call__(self, *args):
        """
        Evaluates the Maxwellian distribution function M(etas, v1, ..., vn).

        Parameters
        ----------
        *args : array_like
            Position-velocity arguments in the order etas, v1, ..., vn.

        Returns
        -------
        f : np.ndarray
            The evaluated Maxwellian.
        """

        res = self.n(*args[:-self.vdim])
        us = self.u(*args[:-self.vdim])
        vths = self.vth(*args[:-self.vdim])

        for i, v in enumerate(args[-self.vdim:]):
            res *= self.gaussian(v, u=us[i], vth=vths[i], is_polar=self.is_polar[i])

        return res