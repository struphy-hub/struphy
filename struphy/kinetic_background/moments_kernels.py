from numpy import sin, cos, pi


def _docstring():
    '''MODULE DOCSTRING for :ref:`struphy.kinetic_background.moments_kernels`.

    The module contains pyccelized functions that can be used to specify moments of background distribution functions f0.
    
    The main method is :meth:`struphy.kinetic_background.moments_kernels.moments` at the end of the file, which calls the functions defined before.
    
    New functions must be added at the top of the file, right after the docstring. 
    In the "Note" of the new function you must state a **moment specifier**; this is an integer that identifies the function in 
    the if-clause of :meth:`struphy.kinetic_background.moments_kernels.moments`.'''

    print('This is just the docstring function.')


def modes_sin_cos(x: 'float', y: 'float', z: 'float', n_modes: 'int', kxs: 'float[:]', kys: 'float[:]', kzs: 'float[:]', amps_sin: 'float[:]', amps_cos: 'float[:]') -> float:
    '''
    Point-wise evaluation of  
    
    ..math::
    
        u(x, y, z) = \sum_{i=0}^N \left[ A_i*\sin(k_{x,i}*x + k_{y,i}*y + k_{z,i}*z) + B_i*\cos(k_{x,i}*x + k_{y,i}*y + k_{z,i}*z) \\right]\,.

    Parameters
    ----------
        x,y,z : float
            Evaluation point.

        n_modes: int
            Number of modes, >0. 

        kxs : array[float]
            Mode numbers in x-direction, kx = o*2*pi/Lx.

        kys : array[float]
            Mode numbers in y-direction, ky = m*2*pi/Ly.

        kzs : array[float]
            Mode numbers in z-direction, kz = n*2*pi/Lz.

        amps_sin : array[float]
            Amplitude of sine function for each mode k = (kx, ky, kz).

        amps_cos : array[float]
            Amplitude of cosine function for each mode k = (kx, ky, kz).

    Returns
    -------
        The function value at (x, y, z).

    Notes
    -----
        Specifier for use in :meth:`struphy.kinetic_background.f0_kernels`: 1
    '''

    value = 0.
    for k in range(n_modes):
        value += amps_sin[k]*sin( kxs[k]*x + kys[k]*y + kzs[k]*z ) + amps_cos[k]*cos( kxs[k]*x + kys[k]*y + kzs[k]*z )

    return value


def moments(x : 'float[:]', moms_spec : 'int[:]', params: 'float[:]'):
    """
    Point-wise evaluation of the moments density, mean velocity and thermal velocity:

    .. math::

        n_0(\mathbf x) &= \int f_0(\mathbf x, \mathbf v)\,\\textnormal d \mathbf v

        \mathbf u_0(\mathbf x) &= \\frac{1}{n_0(\mathbf x)} \int \mathbf v\, f_0(\mathbf x, \mathbf v)\,\\textnormal d \mathbf v

        v_{\\textnormal{th},0,i}(\mathbf x) &= \sqrt{\\frac{1}{n_0(\mathbf x)} \int \mathbf |v_i - u_{0,i}|^2\, f_0(\mathbf x, \mathbf v)\,\\textnormal d \mathbf v }

    Parameters
    ----------
        x : array[float]
            Position at which to evaluate the moments.
        
        moms_spec : array[int]
            Specifier for the seven moments n0, u0x, u0y, u0z, vth0x, vth0y, vth0z (in this order).
            Is 0 for constant moment, for more see Notes.

        params : array[float]
            Parameters needed to specify the moments; the order is specified in :ref:`struphy.kinetic_background.moments_kernels` in the function's docstrings.
            In case that moms_spec[i]=0 (constant value of moment i), the value is given in params[i].

    Returns
    -------
        The function value at (x, y, z, vx, vy, vz).

    Notes
    -----
        See :ref:`struphy.kinetic_background.moments_kernels` for available moment functions.
    """

    ind = 0 # helps you count through params

    # density
    if moms_spec[0] == 0: # constant value
        n0 = params[ind]
        ind += 1
    elif moms_spec[0] == 1: # modes_sin_cos
        n_modes = int(params[ind])
        ind += 1
        kxs = params[ind : ind + n_modes]
        ind += n_modes
        kys = params[ind : ind + n_modes]
        ind += n_modes
        kzs = params[ind : ind + n_modes]
        ind += n_modes
        amps_sin = params[ind : ind + n_modes]
        ind += n_modes
        amps_cos = params[ind : ind + n_modes]
        ind += n_modes
        n0 = modes_sin_cos(x[0], x[1], x[2], n_modes, kxs, kys, kzs, amps_sin, amps_cos)
    else:
        print('Invalid moms_spec[0]', moms_spec[0])

    # x-momentum
    if moms_spec[1] == 0: # constant value
        u0x = params[ind]
        ind += 1
    elif moms_spec[1] == 1: # modes_sin_cos
        n_modes = int(params[ind])
        ind += 1
        kxs = params[ind : ind + n_modes]
        ind += n_modes
        kys = params[ind : ind + n_modes]
        ind += n_modes
        kzs = params[ind : ind + n_modes]
        ind += n_modes
        amps_sin = params[ind : ind + n_modes]
        ind += n_modes
        amps_cos = params[ind : ind + n_modes]
        ind += n_modes
        u0x = modes_sin_cos(x[0], x[1], x[2], n_modes, kxs, kys, kzs, amps_sin, amps_cos)
    else:
        print('Invalid moms_spec[1]', moms_spec[1])

    # y-momentum
    if moms_spec[2] == 0: # constant value
        u0y = params[ind]
        ind += 1
    elif moms_spec[2] == 1: # modes_sin_cos
        n_modes = int(params[ind])
        ind += 1
        kxs = params[ind : ind + n_modes]
        ind += n_modes
        kys = params[ind : ind + n_modes]
        ind += n_modes
        kzs = params[ind : ind + n_modes]
        ind += n_modes
        amps_sin = params[ind : ind + n_modes]
        ind += n_modes
        amps_cos = params[ind : ind + n_modes]
        ind += n_modes
        u0y = modes_sin_cos(x[0], x[1], x[2], n_modes, kxs, kys, kzs, amps_sin, amps_cos)
    else:
        print('Invalid moms_spec[2]', moms_spec[2])

    # z-momentum
    if moms_spec[3] == 0: # constant value
        u0z = params[ind]
        ind += 1
    elif moms_spec[3] == 1: # modes_sin_cos
        n_modes = int(params[ind])
        ind += 1
        kxs = params[ind : ind + n_modes]
        ind += n_modes
        kys = params[ind : ind + n_modes]
        ind += n_modes
        kzs = params[ind : ind + n_modes]
        ind += n_modes
        amps_sin = params[ind : ind + n_modes]
        ind += n_modes
        amps_cos = params[ind : ind + n_modes]
        ind += n_modes
        u0z = modes_sin_cos(x[0], x[1], x[2], n_modes, kxs, kys, kzs, amps_sin, amps_cos)
    else:
        print('Invalid moms_spec[3]', moms_spec[3])

    # x-thermal velocity
    if moms_spec[4] == 0: # constant value
        vth0x = params[ind]
        ind += 1
    elif moms_spec[4] == 1: # modes_sin_cos
        n_modes = int(params[ind])
        ind += 1
        kxs = params[ind : ind + n_modes]
        ind += n_modes
        kys = params[ind : ind + n_modes]
        ind += n_modes
        kzs = params[ind : ind + n_modes]
        ind += n_modes
        amps_sin = params[ind : ind + n_modes]
        ind += n_modes
        amps_cos = params[ind : ind + n_modes]
        ind += n_modes
        vth0x = modes_sin_cos(x[0], x[1], x[2], n_modes, kxs, kys, kzs, amps_sin, amps_cos)
    else:
        print('Invalid moms_spec[4]', moms_spec[4])

    # y-thermal velocity
    if moms_spec[5] == 0: # constant value
        vth0y = params[ind]
        ind += 1
    elif moms_spec[5] == 1: # modes_sin_cos
        n_modes = int(params[ind])
        ind += 1
        kxs = params[ind : ind + n_modes]
        ind += n_modes
        kys = params[ind : ind + n_modes]
        ind += n_modes
        kzs = params[ind : ind + n_modes]
        ind += n_modes
        amps_sin = params[ind : ind + n_modes]
        ind += n_modes
        amps_cos = params[ind : ind + n_modes]
        ind += n_modes
        vth0y = modes_sin_cos(x[0], x[1], x[2], n_modes, kxs, kys, kzs, amps_sin, amps_cos)
    else:
        print('Invalid moms_spec[5]', moms_spec[5])

    # z-thermal velocity
    if moms_spec[6] == 0: # constant value
        vth0z = params[ind]
        ind += 1
    elif moms_spec[6] == 1: # modes_sin_cos
        n_modes = int(params[ind])
        ind += 1
        kxs = params[ind : ind + n_modes]
        ind += n_modes
        kys = params[ind : ind + n_modes]
        ind += n_modes
        kzs = params[ind : ind + n_modes]
        ind += n_modes
        amps_sin = params[ind : ind + n_modes]
        ind += n_modes
        amps_cos = params[ind : ind + n_modes]
        ind += n_modes
        vth0z = modes_sin_cos(x[0], x[1], x[2], n_modes, kxs, kys, kzs, amps_sin, amps_cos)
    else:
        print('Invalid moms_spec[6]', moms_spec[6])

    return n0, u0x, u0y, u0z, vth0x, vth0y, vth0z


