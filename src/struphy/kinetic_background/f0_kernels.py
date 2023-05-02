from numpy import exp, sqrt, pi

import struphy.kinetic_background.moments_kernels as moments_kernels


def maxwellian_6d(x: 'float[:]', v: 'float[:]', moms_spec: 'int[:]', params: 'float[:]') -> float:
    r"""
    Point-wise evaluation of Maxwellian:

    .. math::

        f_0(\mathbf x, \mathbf v) = \frac{n_0(\mathbf x)}{(\pi)^{3/2}\, \hat v_{0,x}(\mathbf x) \hat v_{0,y}(\mathbf x) \hat v_{0,z}(\mathbf x)}\,\exp{\left[ - \frac{(v_x - u_{0,x}(\mathbf x))^2}{\hat v_{0,x}^2(\mathbf x)} \frac{(v_y - u_{0,y}(\mathbf x))^2}{\hat v_{0,y}^2(\mathbf x)} \frac{(v_z - u_{0,z}(\mathbf x))^2}{\hat v_{0,z}^2(\mathbf x)} \right]}

    Parameters
    ----------
        x : array[float]
            Position at which to evaluate the Maxwellian.

        v : array[float]
            Velocity at which to evaluate the Maxwellian.

        moms_spec : array[int]
            Specifier for the seven moments n0, u0x, u0y, u0z, vth0x, vth0y, vth0z (in this order).
            Is 0 for constant moment, for more see :meth:`kinetic_moments`.

        params : array[float]
            Parameters needed to specify the moments; the order is specified in :ref:`kinetic_moments` for the respective functions available.

    Notes
    -----
        See :meth:`kinetic_moments` for available moment functions.

        parameters.yml specifier: 0
    """

    n0, u0x, u0y, u0z, vth0x, vth0y, vth0z = moments_kernels.moments(
        x, moms_spec, params)

    Gx = exp(-(v[0] - u0x)**2 / (vth0x**2)) / (sqrt(pi)*vth0x)
    Gy = exp(-(v[1] - u0y)**2 / (vth0y**2)) / (sqrt(pi)*vth0y)
    Gz = exp(-(v[2] - u0z)**2 / (vth0z**2)) / (sqrt(pi)*vth0z)

    return n0*Gx*Gy*Gz
