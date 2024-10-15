#!/usr/bin/env python3
'Analytical perturbations (modes).'


import numpy as np


class ModesSin:
    r'''Sinusoidal function in 3D.

    .. math::

        u(x, y, z) =  \sum_{s} \chi_s(z) A_s \sin \left(l_s \frac{2\pi}{L_x} x + m_s \frac{2\pi}{L_y} y + n_s \frac{2\pi}{L_z} z + \theta_s \right) \,.

    where :math:`\chi_s(z)` is one of

    .. math::

        \chi_s(z) = \left\{ 
        \begin{aligned}
        1\,,
        \\[2mm]
         \tanh((z - 0.5)/\delta)/\cosh((z - 0.5)/\delta)\,, 
        \end{aligned}
        \right.

    Can be used in logical space, where :math:`x \to \eta_1,\, y\to \eta_2,\, z \to \eta_3` 
    and :math:`L_x=L_y=L_z=1.0` (default).

    Note
    ----
    Example of use in a ``.yml`` parameter file::

        perturbations :
            type : ModesSin
            ModesSin :
                comps :
                    scalar_name : '0' # choices: null, 'physical', '0', '3'
                    vector_name : [null , 'v', '2']  # choices: null, 'physical', '1', '2', 'v', 'norm'
                ls : 
                    scalar_name: [1, 3] # two x-modes for scalar variable
                    vector_name: [null, [0, 1], [4]] # two x-modes for 2nd comp. and one x-mode for third component of vector-valued variable            
                theta :
                    scalar_name: [0, 3.1415] 
                    vector_name: [null, [0, 0], [1.5708]]
                pfuns :
                    vector_name: [null, ['localize'], ['Id']]
                pfuns_params
                    vector_name: [null, ['0.1'], [0.]]
                Lx : 7.853981633974483 
                Ly : 1.                
                Lz : 1.               
    '''

    def __init__(self, ls=None, ms=None, ns=None, amps=[1e-4], theta=None, pfuns=['Id'], pfuns_params = [0.], Lx=1., Ly=1., Lz=1.):
        '''
        Parameters
        ----------
        ls : list
            Mode numbers in x-direction (kx = l*2*pi/Lx).

        ms : list
            Mode numbers in y-direction (ky = m*2*pi/Ly).

        ns : list
            Mode numbers in z-direction (kz = n*2*pi/Lz).

        amps : list
            Amplitude of each mode.

        theta : list
            Phase of each mode

        pfuns : list[str]
            "Id" or "localize" define the profile functions.
            localize multiply the sinus by :math: `tanh((\eta_3 - 0.5)/\delta)/cosh((\eta_3 - 0.5)/\delta)`
            to localize it around 0.5. :math: `\delta` is given by the input parameter pfuns_params

        pfuns_params : list
            The parameter needed by the profile function

        Lx, Ly, Lz : float
            Domain lengths.
        '''

        if ls is not None:
            n_modes = len(ls)
        elif ms is not None:
            n_modes = len(ms)
            ls = [0]*n_modes
        elif ns is not None:
            n_modes = len(ns)
            ls = [0]*n_modes
            ms = [0]*n_modes
        else:
            n_modes = 1
            ls = [0]
            ms = [0]
            ns = [0]
            
        if ms is None:
            ms = [0]*n_modes
        else:
            assert len(ms) == n_modes
            
        if ns is None:
            ns = [0]*n_modes
        else:
            assert len(ns) == n_modes
            
        if len(amps) == 1:
            amps = [amps[0]]*n_modes
        else:
            assert len(amps) == n_modes

        if theta is None:
            theta = [0]*n_modes

        if len(theta) == 1:
            theta = [theta[0]]*n_modes
        else:
            assert len(theta) == n_modes

        if len(pfuns) ==1:
            pfuns = [pfuns[0]]*n_modes
        else:
            assert len(pfuns) == n_modes

        if len(pfuns_params) ==1:
            pfuns_params = [pfuns_params[0]]*n_modes
        else:
            assert len(pfuns_params) == n_modes

        self._ls = ls
        self._ms = ms
        self._ns = ns
        self._amps = amps
        self._Lx = Lx
        self._Ly = Ly
        self._Lz = Lz
        self._theta = theta

        self._pfuns = []
        for pfun, params in zip(pfuns, pfuns_params):
            if pfun == 'Id':
                self._pfuns += [lambda eta3: 1.]
            elif pfun == 'localize':
                self._pfuns += [lambda eta3:
                                np.tanh((eta3 - 0.5)/params)/np.cosh((eta3 - 0.5)/params)]
            else:
                raise ValueError(f'Profile function {pfun} is not defined..')

    def __call__(self, x, y, z):

        val = 0.

        for amp, l, m, n, t, pfun in zip(self._amps, self._ls, self._ms, self._ns, self._theta, self._pfuns):
            val += amp*pfun(z)*np.sin(l*2.*np.pi/self._Lx*x + m*2. *
                              np.pi/self._Ly*y + n*2.*np.pi/self._Lz*z + t)

        return val


class ModesCos:
    r'''Cosinusoidal function in 3D.

    .. math::

        u(x, y, z) = \sum_{s} A_s \cos \left(l_s \frac{2\pi}{L_x} x + m_s \frac{2\pi}{L_y} y + n_s \frac{2\pi}{L_z} z \right) \,.

    Can be used in logical space, where :math:`x \to \eta_1,\, y\to \eta_2,\, z \to \eta_3` 
    and :math:`L_x=L_y=L_z=1.0` (default).

    Note
    ----
    Example of use in a ``.yml`` parameter file::

        perturbations :
            type : ModesCos
            ModesCos :
                comps :
                    scalar_name : '0' # choices: null, 'physical', '0', '3'
                    vector_name : [null , 'v', '2']  # choices: null, 'physical', '1', '2', 'v', 'norm'
                ls : 
                    scalar_name: [1, 3] # two x-modes for scalar variable
                    vector_name: [null, [0, 1], [4]] # two x-modes for 2nd comp. and one x-mode for third component of vector-valued variable            
                Lx : 7.853981633974483 
                Ly : 1.                
                Lz : 1. 
    '''

    def __init__(self, ls=None, ms=None, ns=None, amps=[1e-4], Lx=1., Ly=1., Lz=1.):
        '''
        Parameters
        ----------
        ls : list
            Mode numbers in x-direction (kx = l*2*pi/Lx).

        ms : list
            Mode numbers in y-direction (ky = m*2*pi/Ly).

        ns : list
            Mode numbers in z-direction (kz = n*2*pi/Lz).

        amps : list
            Amplitude of each mode.

        Lx, Ly, Lz : float
            Domain lengths.
        '''

        if ls is not None:
            n_modes = len(ls)
        elif ms is not None:
            n_modes = len(ms)
            ls = [0]*n_modes
        elif ns is not None:
            n_modes = len(ns)
            ls = [0]*n_modes
            ms = [0]*n_modes
        else:
            n_modes = 1
            ls = [0]
            ms = [0]
            ns = [0]
            
        if ms is None:
            ms = [0]*n_modes
        else:
            assert len(ms) == n_modes
            
        if ns is None:
            ns = [0]*n_modes
        else:
            assert len(ns) == n_modes
            
        if len(amps) == 1:
            amps = [amps[0]]*n_modes
        else:
            assert len(amps) == n_modes

        self._ls = ls
        self._ms = ms
        self._ns = ns
        self._amps = amps
        self._Lx = Lx
        self._Ly = Ly
        self._Lz = Lz

    def __call__(self, x, y, z):

        val = 0.

        for amp, l, m, n in zip(self._amps, self._ls, self._ms, self._ns):
            val += amp * np.cos(l * 2.*np.pi / self._Lx * x
                                + m * 2.*np.pi / self._Ly * y
                                + n * 2.*np.pi / self._Lz * z)

        return val


class TorusModesSin:
    r'''Sinusoidal function in the periodic coordinates of a Torus.

    .. math::

        u(\eta_1, \eta_2, \eta_3) = \sum_{s} \chi_s(\eta_1) A_s \sin(m_s\,2\pi \eta_2 + n_s\,2\pi \eta_3) \,,

    where :math:`\chi_s(\eta_1)` is one of

    .. math::

        \chi_s(\eta_1) = \left\{ 
        \begin{aligned}
        &\sin(l_s\pi\eta_1)\,,
        \\[2mm]
        &\exp^{-(\eta_1 - r_0)^2/\sigma} \,, 
        \\[2mm]
        & -2(\eta_1 - r_0)/\sigma)\exp^{-(\eta_1 - r_0)^2/\sigma} \,.
        \end{aligned}
        \right.

    Can only be defined in logical coordinates.

    Note
    ----
    In the parameter .yml, use the following template in the section ``fluid/<species>``::

        perturbations :
            type : TorusModesSin
            TorusModesSin :
                comps :
                    n3 : null                     # choices: null, 'physical', '0', '3'
                    u2 : ['physical', 'v', '2']   # choices: null, 'physical', '1', '2', 'v', 'norm'
                    p3 : '0'                      # choices: null, 'physical', '0', '3'
                ms : 
                    n3: null            # poloidal mode numbers
                    u2: [[0], [0], [0]] # poloidal mode numbers
                    p3: [0]             # poloidal mode numbers
                ns :
                    n3: null            # toroidal mode numbers
                    u2: [[1], [1], [1]] # toroidal mode numbers
                    p3: [1]             # toroidal mode numbers
                amps :
                    n3: null                        # amplitudes of each mode
                    u2: [[0.001], [0.001], [0.001]] # amplitudes of each mode
                    p3: [0.01]                      # amplitudes of each mode
                pfuns :
                    n3: null                        # profile function in eta1-direction ('sin' or 'cos' or 'exp' or 'd_exp')
                    u2: [['sin'], ['sin'], ['exp']] # profile function in eta1-direction ('sin' or 'cos' or 'exp' or 'd_exp')
                    p3: [0.01]                      # profile function in eta1-direction ('sin' or 'cos' or 'exp' or 'd_exp')
                pfun_params :
                    n3: null                      # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and l_s for "sin" and "cos"
                    u2: [2, null, [[0.5, 1.]]]    # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and l_s for "sin" and "cos"
                    p3: [0.01]                    # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and l_s for "sin" and "cos"  
    '''

    def __init__(self, ms=None, ns=None, amps=[1e-4], pfuns=['sin'], pfun_params=None):
        r'''
        Parameters
        ----------
        ms : list[int]
            Poloidal mode numbers.

        ns : list[int]
            Toroidal mode numbers.

        pfuns : list[str]
            "sin" or "cos" or "exp" to define the profile functions.

        amps : list[float]
            Amplitudes of each mode (m_i, n_i).

        pfun_params : list
            Provides :math:`[r_0, \sigma]` parameters for each "exp" profile fucntion, and l_s for "sin" and "cos".
        '''

        if ms is not None:
            n_modes = len(ms)
        elif ns is not None:
            n_modes = len(ns)
            ms = [0]*n_modes
        else:
            n_modes = 1
            ms = [1]
            ns = [0]
            
        if ns is None:
            ns = [0]*n_modes
        else:
            assert len(ns) == n_modes
            
        if len(amps) == 1:
            amps = [amps[0]]*n_modes
        else:
            assert len(amps) == n_modes

        if len(pfuns) == 1:
            pfuns = [pfuns[0]]*n_modes
        else:
            assert len(pfuns) == n_modes
            
        if pfun_params is None:
            pfun_params = [None]*n_modes

        self._ms = ms
        self._ns = ns
        self._amps = amps

        self._pfuns = []
        for pfun, params in zip(pfuns, pfun_params):
            if pfun == 'sin':
                if params is None :
                    ls = 1
                else :
                    ls = params
                self._pfuns += [lambda eta1: np.sin(ls*np.pi*eta1)]
            elif pfun == 'exp':
                self._pfuns += [lambda eta1:
                                np.exp(-(eta1 - params[0])**2/params[1])]
            elif pfun == 'd_exp':
                self._pfuns += [lambda eta1:
                                -2*(eta1 - params[0])/params[1]*np.exp(-(eta1 - params[0])**2/params[1])]
            else:
                raise ValueError(f'Profile function {pfun} is not defined..')

    def __call__(self, eta1, eta2, eta3):

        val = 0.
        for mi, ni, pfun, amp in zip(self._ms, self._ns, self._pfuns, self._amps):
            val += amp * pfun(eta1) * np.sin(mi*2.*np.pi *
                                             eta2 + ni*2.*np.pi*eta3)

        return val


class TorusModesCos:
    r'''Cosinusoidal function in the periodic coordinates of a Torus.

    .. math::

        u(\eta_1, \eta_2, \eta_3) = \sum_{s} \chi_s(\eta_1) A_s \cos(m_s\,2\pi \eta_2 + n_s\,2\pi \eta_3) \,,

    where :math:`\chi_s(\eta_1)` is one of

    .. math::

        \chi_s(\eta_1) = \left\{ 
        \begin{aligned}
        &\sin(\pi\eta_1)\,,
        \\[2mm]
        &\exp^{-(\eta_1 - r_0)^2/\sigma} \,, 
        \\[2mm]
        & -2(\eta_1 - r_0)/\sigma)\exp^{-(\eta_1 - r_0)^2/\sigma} \,.
        \end{aligned}
        \right.

    Can only be defined in logical coordinates.

    Note
    ----
    In the parameter .yml, use the following template in the section ``fluid/<species>``::

        perturbations :
            type : TorusModesCos
            TorusModesCos :
                comps :
                    n3 : null                     # choices: null, 'physical', '0', '3'
                    u2 : ['physical', 'v', '2']   # choices: null, 'physical', '1', '2', 'v', 'norm'
                    p3 : H1                       # choices: null, 'physical', '0', '3'
                ms : 
                    n3: null            # poloidal mode numbers
                    u2: [[0], [0], [0]] # poloidal mode numbers
                    p3: [0]             # poloidal mode numbers
                ns :
                    n3: null            # toroidal mode numbers
                    u2: [[1], [1], [1]] # toroidal mode numbers
                    p3: [1]             # toroidal mode numbers
                amps :
                    n3: null                        # amplitudes of each mode
                    u2: [[0.001], [0.001], [0.001]] # amplitudes of each mode
                    p3: [0.01]                      # amplitudes of each mode
                pfuns :
                    n3: null                        # profile function in eta1-direction ('sin' or 'cos' or 'exp' or 'd_exp')
                    u2: [['sin'], ['sin'], ['exp']] # profile function in eta1-direction ('sin' or 'cos' or 'exp' or 'd_exp')
                    p3: [0.01]                      # profile function in eta1-direction ('sin' or 'cos' or 'exp' or 'd_exp')
                pfun_params :
                    n3: null                      # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and l_s for "sin" and "cos".
                    u2: [2, null, [[0.5, 1.]]]    # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and l_s for "sin" and "cos".
                    p3: [0.01]                    # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and l_s for "sin" and "cos".     
    '''

    def __init__(self, ms=None, ns=None, amps=[1e-4], pfuns=['sin'], pfun_params=None):
        r'''
        Parameters
        ----------
        ms : list[int]
            Poloidal mode numbers.

        ns : list[int]
            Toroidal mode numbers.

        pfuns : list[str]
            "sin" or "cos" or "exp" to define the profile functions.

        amps : list[float]
            Amplitudes of each mode (m_i, n_i).

        pfun_params : list
            Provides :math:`[r_0, \sigma]` parameters for each "exp" profile fucntion, and l_s for "sin" and "cos".
        '''

        if ms is not None:
            n_modes = len(ms)
        elif ns is not None:
            n_modes = len(ns)
            ms = [0]*n_modes
        else:
            n_modes = 1
            ms = [1]
            ns = [0]
            
        if ns is None:
            ns = [0]*n_modes
        else:
            assert len(ns) == n_modes
            
        if len(amps) == 1:
            amps = [amps[0]]*n_modes
        else:
            assert len(amps) == n_modes

        if len(pfuns) == 1:
            pfuns = [pfuns[0]]*n_modes
        else:
            assert len(pfuns) == n_modes
            
        if pfun_params is None:
            pfun_params = [None]*n_modes

        self._ms = ms
        self._ns = ns
        self._amps = amps

        self._pfuns = []
        for pfun, params in zip(pfuns, pfun_params):
            if pfun == 'sin':
                if params is None :
                    ls = 1
                else :
                    ls = params
                self._pfuns += [lambda eta1: np.sin(ls*np.pi*eta1)]
            elif pfun == 'cos':
                self._pfuns += [lambda eta1: np.cos(np.pi*eta1)]
            elif pfun == 'exp':
                self._pfuns += [lambda eta1:
                                np.exp(-(eta1 - params[0])**2/params[1])]
            elif pfun == 'd_exp':
                self._pfuns += [lambda eta1:
                                -2*(eta1 - params[0])/params[1]*np.exp(-(eta1 - params[0])**2/params[1])]
            else:
                raise ValueError(
                    'Profile function must be "sin" or "cos" or "exp".')

    def __call__(self, eta1, eta2, eta3):

        val = 0.
        for mi, ni, pfun, amp in zip(self._ms, self._ns, self._pfuns, self._amps):
            val += amp * pfun(eta1) * np.cos(mi*2.*np.pi *
                                             eta2 + ni*2.*np.pi*eta3)

        return val


class Shear_x:
    r'''Double shear layer in eta1 (-1 in outer regions, 1 in inner regions).

    .. math::

        u(\eta_1, \eta_2, \eta_3) = A(-\tanh((\eta_1 - 0.25)/\delta)+\tanh((\eta_1 - 0.75)/\delta) - 1) \,. 

    Can only be used in logical space.

    Note
    ----
    In the parameter .yml, use the following in the section ``fluid/<species>``::

        perturbations :
            type : Shear_x
            Shear_x :
                comps :
                    rho3 : null                   # choices: null, 'physical', '0', '3'
                    uv : ['physical', 'v', '2']   # choices: null, 'physical', '1', '2', 'v', 'norm'
                    s3 : H1                       # choices: null, 'physical', '0', '3'
                amp : 0.001 # amplitudes of each mode
                delta : 0.03333 # characteristic size of the shear layer
    '''

    def __init__(self, amp=1e-4, delta=1/15):
        '''
        Parameters
        ----------
        amps : float
            Amplitude of the velocity on each side.

        delta : float
            Characteristic size of the shear layer
        '''

        self._amp = amp
        self._delta = delta

    def __call__(self, e1, e2, e3):

        val = self._amp*(-np.tanh((e1 - 0.75)/self._delta) +
                         np.tanh((e1 - 0.25)/self._delta) - 1)

        return val


class Shear_y:
    r'''Double shear layer in eta2 (-1 in outer regions, 1 in inner regions).

    .. math::

        u(\eta_1, \eta_2, \eta_3) = A(-\tanh((\eta_2 - 0.25)/\delta) + \tanh((\eta_2 - 0.75)/\delta) - 1) \,.

    Can only be used in logical space.

    Note
    ----
    In the parameter .yml, use the following in the section ``fluid/<species>``::

        perturbations :
            type : Shear_y
            Shear_y :
                comps :
                    rho3 : null                   # choices: null, 'physical', '0', '3'
                    uv : ['physical', 'v', '2']   # choices: null, 'physical', '1', '2', 'v', 'norm'
                    s3 : H1                       # choices: null, 'physical', '0', '3'
                amp : 0.001 # amplitudes of each mode
                delta : 0.03333 # characteristic size of the shear layer
    '''

    def __init__(self, amp=1e-4, delta=1/15):
        '''
        Parameters
        ----------
        amps : float
            Amplitude of the velocity on each side.

        delta : float
            Characteristic size of the shear layer
        '''

        self._amp = amp
        self._delta = delta

    def __call__(self, e1, e2, e3):

        val = self._amp*(-np.tanh((e2 - 0.75)/self._delta) +
                         np.tanh((e2 - 0.25)/self._delta) - 1)

        return val


class Shear_z:
    r'''Double shear layer in eta3 (-1 in outer regions, 1 in inner regions).

    .. math::

        u(\eta_1, \eta_2, \eta_3) = A(-\tanh((\eta_3 - 0.25)/\delta) + \tanh((\eta_3 - 0.75)/\delta) - 1) \,. 

    Can only be used in logical space.

    Note
    ----
    In the parameter .yml, use the following in the section ``fluid/<species>``::

        perturbations :
            type : Shear_y
            Shear_y :
                comps :
                    rho3 : null                   # choices: null, 'physical', '0', '3'
                    uv : ['physical', 'v', '2']   # choices: null, 'physical', '1', '2', 'v', 'norm'
                    s3 : H1                       # choices: null, 'physical', '0', '3'
                amp : 0.001 # amplitudes of each mode
                delta : 0.03333 # characteristic size of the shear layer
    '''

    def __init__(self, amp=1e-4, delta=1/15):
        '''
        Parameters
        ----------
        amps : float
            Amplitude of the velocity on each side.

        delta : float
            Characteristic size of the shear layer
        '''

        self._amp = amp
        self._delta = delta

    def __call__(self, e1, e2, e3):

        val = self._amp*(-np.tanh((e3 - 0.75)/self._delta) +
                         np.tanh((e3 - 0.25)/self._delta) - 1)

        return val


class ITPA_density:
    r'''ITPA radial density profile in `A. Könies et al. 2018  <https://iopscience.iop.org/article/10.1088/1741-4326/aae4e6>`_

    .. math::

        n(\eta_1) = n_0*c_3\exp\left[-\frac{c_2}{c_1}\tanh\left(\frac{\eta_1 - c_0}{c_2}\right)\right]\,.

    Note
    ----
    In the parameter .yml, use the following template in the section ``kinetic/<species>``::

        perturbation :
            type : ITPA_density
            ITPA_density :
                comps :
                    n : '0'
                n0 :
                    n : 0.00720655
                c :
                    n : [0.491230, 0.298228, 0.198739, 0.521298]
    '''

    def __init__(self, n0=0.00720655, c=[0.491230, 0.298228, 0.198739, 0.521298]):
        '''
        Parameters
        ----------
        n0 : float
            ITPA profile density

        c : list
            4 ITPA profile coefficients
        '''

        assert len(c) == 4

        self._n0 = n0
        self._c = c

    def __call__(self, eta1, eta2=None, eta3=None):

        val = 0.

        if self._c[2] == 0.:
            val = self._c[3] - 0*eta1
        else:
            val = self._n0 * \
                self._c[3]*np.exp(-self._c[2]/self._c[1] *
                                  np.tanh((eta1 - self._c[0])/self._c[2]))

        return val
    
class Erf_z:
    r'''Shear layer in eta3 (-1 in lower regions, 1 in upper regions).

    .. math::

        u(\eta_1, \eta_2, \eta_3) = A \, erf((\eta_3 - 0.5)/\delta) \,.

    Can only be used in logical space.

    Note
    ----
    In the parameter .yml, use the following in the section ``fluid/<species>``::

        perturbations :
            type : Erf_z
        Erf_z :
            comps :
                b2 : ['2', null, null] # choices: null, 'physical', '0', '3'
            amp : 
                b2 : [0.001] # amplitudes of each mode
            delta :
                b2 : [0.02] # characteristic size of the shear layer
    '''

    def __init__(self, amp=1e-4, delta=1/15):
        '''
        Parameters
        ----------
        amp : float
            Amplitude of the velocity on each side.

        delta : float
            Characteristic size of the shear layer
        '''

        self._amp = amp
        self._delta = delta

    def __call__(self, e1, e2, e3):

        from scipy.special import erf
        val = self._amp*erf((e3 - 0.5)/self._delta)

        return val
