import struphy.kinetic_background.f0_kernels as f0_kernels


def f0(x: 'float[:]', v: 'float[:]', f0_spec: 'int', moms_spec: 'int[:]', params: 'float[:]') -> float:
    """
    Point-wise evaluation of a static kinetic background in 6d phase space.

    Parameters
    ----------
        x : array[float]
            Position at which to evaluate f0.

        v : array[float]
            Velocity at which to evaluate f0.

        f0_spec : int
            Specifier for kinetic background, 0 -> maxwellian_6d. See Notes.

        moms_spec : array[int]
            Specifier for the seven moments n0, u01, u02, u03, vth01, vth02, vth03 (in this order).
            Is 0 for constant moment, for more see :math:`struphy.kinetic_background.moments_kernels.moments`.

        params : array[float]
            Parameters needed to specify the moments; the order is specified in :ref:`struphy.kinetic_background.moments_kernels` for the respective functions available.

    Notes
    -----
        See :ref:`struphy.kinetic_background.f0_kernels` for available backgrounds.
    """

    if f0_spec == 0:
        value = f0_kernels.maxwellian_6d(x, v, moms_spec, params)
    else:
        print('Invalid f0_spec', f0_spec)

    return value
