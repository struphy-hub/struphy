#!/usr/bin/env python3
"Analytical moment functions."

import cunumpy as xp


class ITPA_density:
    r"""ITPA radial density profile in `A. Könies et al. 2018  <https://iopscience.iop.org/article/10.1088/1741-4326/aae4e6>`_

    .. math::

        n(\eta_1) = n_0*c_3\exp\left[-\frac{c_2}{c_1}\tanh\left(\frac{\eta_1 - c_0}{c_2}\right)\right]\,.

    Note
    ----
    In the parameter .yml, use the following template in the section ``kinetic/<species>``::

        ITPA_density :
            given_in_basis : '0'
            n0 : 0.00720655
            c : [0.491230, 0.298228, 0.198739, 0.521298]
    """

    def __init__(self, n0=0.00720655, c=(0.491230, 0.298228, 0.198739, 0.521298)):
        """
        Parameters
        ----------
        n0 : float
            ITPA profile density

        c : tuple | list
            4 ITPA profile coefficients
        """

        assert len(c) == 4

        self._n0 = n0
        self._c = c

    def __call__(self, eta1, eta2=None, eta3=None):
        val = 0.0

        if self._c[2] == 0.0:
            val = self._c[3] - 0 * eta1
        else:
            val = (
                self._n0
                * self._c[3]
                * xp.exp(
                    -self._c[2] / self._c[1] * xp.tanh((eta1 - self._c[0]) / self._c[2]),
                )
            )

        return val
