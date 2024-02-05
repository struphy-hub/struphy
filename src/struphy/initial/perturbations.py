#!/usr/bin/env python3
'Analytical perturbations (modes).'


import numpy as np


class ModesSin:
    r'''Sinusoidal function in 3D.

    .. math::

        u(x, y, z) = \sum_{i} A_i \sin \left(l_i \frac{2\pi}{L_x} x + m_i \frac{2\pi}{L_y} y + n_i \frac{2\pi}{L_z} z \right) \,.

    Can be used in logical space with Lx=Ly=Lz=1.0 (default).

    Note
    ----
    In the parameter .yml, use the following template in the section ``fluid/<species>``::

        init :
            type : ModesSin
            ModesSin :
                comps :
                    n3 : null                    # choices: null, 'physical', '0', '3'
                    u2 : ['physical', 'v', '2']  # choices: null, 'physical', '1', '2', 'v', 'norm'
                    p3 : H1                      # choices: null, 'physical', '0', '3'
                ls :
                    n3: null            # Integer mode numbers in x or eta_1
                    u2: [[0], [0], [0]] # Integer mode numbers in x or eta_1
                    p3: [0]             # Integer mode numbers in x or eta_1
                ms : 
                    n3: null            # Integer mode numbers in y or eta_2
                    u2: [[0], [0], [0]] # Integer mode numbers in y or eta_2
                    p3: [0]             # Integer mode numbers in y or eta_2
                ns :
                    n3: null            # Integer mode numbers in z or eta_3
                    u2: [[1], [1], [1]] # Integer mode numbers in z or eta_3
                    p3: [1]             # Integer mode numbers in z or eta_3
                amps :
                    n3: null                        # amplitudes of each mode
                    u2: [[0.001], [0.001], [0.001]] # amplitudes of each mode
                    p3: [0.01]                      # amplitudes of each mode
                Lx : 7.853981633974483 # domain length in x
                Ly : 1.                # domain length in y
                Lz : 1.                # domain length in z
    '''

    def __init__(self, ls=[0], ms=[0], ns=[0], amps=[1e-4], Lx=1., Ly=1., Lz=1.):
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

        assert len(ls) == len(ms)
        assert len(ls) == len(ns)
        assert len(ls) == len(amps)

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
            val += amp*np.sin(l*2.*np.pi/self._Lx*x + m*2. *
                              np.pi/self._Ly*y + n*2.*np.pi/self._Lz*z)

        return val


class ModesCos:
    r'''Cosinusoidal function in 3D.

    .. math::

        u(x, y, z) = \sum_{i} A_i \cos \left(l_i \frac{2\pi}{L_x} x + m_i \frac{2\pi}{L_y} y + n_i \frac{2\pi}{L_z} z \right) \,.

    Can be used in logical space with Lx=Ly=Lz=1.0 (default).

    Note
    ----
    In the parameter .yml, use the following template in the section ``fluid/<species>``::

        init :
            type : ModesCos 
            ModesCos :
                comps :
                    n3 : null                     # choices: null, 'physical', '0', '3'
                    u2 : ['physical', 'v', '2']   # choices: null, 'physical', '1', '2', 'v', 'norm'
                    p3 : H1                       # choices: null, 'physical', '0', '3'
                ls :
                    n3: null            # Integer mode numbers in x or eta_1
                    u2: [[0], [0], [0]] # Integer mode numbers in x or eta_1
                    p3: [0]             # Integer mode numbers in x or eta_1
                ms : 
                    n3: null            # Integer mode numbers in y or eta_2
                    u2: [[0], [0], [0]] # Integer mode numbers in y or eta_2
                    p3: [0]             # Integer mode numbers in y or eta_2
                ns :
                    n3: null            # Integer mode numbers in z or eta_3
                    u2: [[1], [1], [1]] # Integer mode numbers in z or eta_3
                    p3: [1]             # Integer mode numbers in z or eta_3
                amps :
                    n3: null                        # amplitudes of each mode
                    u2: [[0.001], [0.001], [0.001]] # amplitudes of each mode
                    p3: [0.01]                      # amplitudes of each mode
                Lx : 7.853981633974483 # domain length in x
                Ly : 1.                # domain length in y
                Lz : 1.                # domain length in z
    '''

    def __init__(self, ls=[0], ms=[0], ns=[0], amps=[1e-4], Lx=1., Ly=1., Lz=1.):
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

        assert len(ls) == len(ms)
        assert len(ls) == len(ns)
        assert len(ls) == len(amps)

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
            val += amp*np.cos(l*2.*np.pi/self._Lx*x + m*2. *
                              np.pi/self._Ly*y + n*2.*np.pi/self._Lz*z)

        return val


class TorusModesSin:
    r'''Sinusoidal function in the periodic coordinates of a Torus.

    .. math::

        u(\eta_1, \eta_2, \eta_3) = \sum_{i=0}^N \chi_i(\eta_1) A_i \sin(m_i\,2\pi \eta_2 + n_i\,2\pi \eta_3) \,,

    where :math:`\chi_i(\eta_1)` is one of

    .. math::

        \chi_i(\eta_1) = \left\{ 
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

        init :
            type : TorusModesSin
            TorusModesSin :
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
                    n3: null                      # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and null for "sin" and "cos"
                    u2: [null, null, [[0.5, 1.]]] # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and null for "sin" and "cos"
                    p3: [0.01]                    # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and null for "sin" and "cos"        
    '''

    def __init__(self, ms=[0], ns=[0], amps=[1e-4], pfuns=['sin'], pfun_params=None):
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
                Provides :math:`[r_0, \sigma]` parameters for each "exp" profile fucntion, and None for "sin" and "cos".
        '''

        assert len(ms) == len(ns)
        assert len(ms) == len(pfuns)
        assert len(ms) == len(amps)

        if pfun_params is None:
            pfun_params = [None]*len(ms)

        assert len(ms) == len(pfun_params)

        self._ms = ms
        self._ns = ns
        self._amps = amps

        self._pfuns = []
        for pfun, params in zip(pfuns, pfun_params):
            if pfun == 'sin':
                self._pfuns += [lambda eta1: np.sin(np.pi*eta1)]
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

        u(\eta_1, \eta_2, \eta_3) = \sum_{i=0}^N \chi_i(\eta_1) A_i \cos(m_i\,2\pi \eta_2 + n_i\,2\pi \eta_3) \,,

    where :math:`\chi_i(\eta_1)` is one of

    .. math::

        \chi_i(\eta_1) = \left\{ 
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

        init :
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
                    n3: null                      # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and null for "sin" and "cos"
                    u2: [null, null, [[0.5, 1.]]] # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and null for "sin" and "cos"
                    p3: [0.01]                    # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and null for "sin" and "cos"        
    '''

    def __init__(self, ms=[0], ns=[0], amps=[1e-4], pfuns=['cos'], pfun_params=None):
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
                Provides :math:`[r_0, \sigma]` parameters for each "exp" profile fucntion, and None for "sin" and "cos".
        '''

        assert len(ms) == len(ns)
        assert len(ms) == len(pfuns)
        assert len(ms) == len(amps)

        if pfun_params is None:
            pfun_params = [None]*len(ms)

        assert len(ms) == len(pfun_params)

        self._ms = ms
        self._ns = ns
        self._amps = amps

        self._pfuns = []
        for pfun, params in zip(pfuns, pfun_params):
            if pfun == 'sin':
                self._pfuns += [lambda eta1: np.sin(np.pi*eta1)]
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
