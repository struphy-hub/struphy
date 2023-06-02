from numpy import exp, sqrt, pi

import struphy.kinetic_background.moments_kernels as moments_kernels


def maxwellian_6d(x: 'float[:]', v: 'float[:]', moms_spec: 'int[:]', params: 'float[:]') -> float:
    r"""
    Point-wise evaluation of Maxwellian:

    .. math::

        f_0(\mathbf x, \mathbf v) = \frac{n_0(\mathbf x)}{(\pi)^{3/2}\, \hat v_{0,1}(\mathbf x) \hat v_{0,2}(\mathbf x) \hat v_{0,3}(\mathbf x)} \,
        \exp{\left[ - \frac{(v_1 - u_{0,1}(\mathbf x))^2}{\hat v_{0,1}^2(\mathbf x)}
                      \frac{(v_2 - u_{0,2}(\mathbf x))^2}{\hat v_{0,2}^2(\mathbf x)}
                      \frac{(v_3 - u_{0,3}(\mathbf x))^2}{\hat v_{0,3}^2(\mathbf x)} \right]}

    Parameters
    ----------
        x : array[float]
            Position at which to evaluate the Maxwellian.

        v : array[float]
            Velocity at which to evaluate the Maxwellian.

        moms_spec : array[int]
            Specifier for the seven moments n0, u01, u02, u03, vth01, vth02, vth03 (in this order).
            Is 0 for constant moment, for more see :meth:`kinetic_moments`.

        params : array[float]
            Parameters needed to specify the moments; the order is specified in :ref:`kinetic_moments` for the respective functions available.

    Notes
    -----
        See :meth:`kinetic_moments` for available moment functions.

        parameters.yml specifier: 0
    """

    n0, u01, u02, u03, vth01, vth02, vth03 = moments_kernels.moments(
        x, moms_spec, params)

    g1 = exp(-(v[0] - u01)**2 / (vth01**2)) / (sqrt(pi) * vth01)
    g2 = exp(-(v[1] - u02)**2 / (vth02**2)) / (sqrt(pi) * vth02)
    g3 = exp(-(v[2] - u03)**2 / (vth03**2)) / (sqrt(pi) * vth03)

    return n0 * g1 * g2 * g3
