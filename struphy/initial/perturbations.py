#!/usr/bin/env python3

import numpy as np

class ModesSin:
    r'''Defines the callable
    
    .. math::
    
        u(x, y, z) = \sum_{i} A_i \sin \left(l_i \frac{2\pi}{L_x} x + m_i \frac{2\pi}{L_y} y + n_i \frac{2\pi}{L_z} z \right) \,.
    '''

    def __init__(self, ls, ms, ns, amps, Lx=1., Ly=1., Lz=1.):
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
            val += amp*np.sin(l*2.*np.pi/self._Lx*x + m*2.*np.pi/self._Ly*y + n*2.*np.pi/self._Lz*z)

        return val


class ModesCos:
    r'''Defines the callable
    
    .. math::
    
        u(x, y, z) = \sum_{i} A_i \cos \left(l_i \frac{2\pi}{L_x} x + m_i \frac{2\pi}{L_y} y + n_i \frac{2\pi}{L_z} z \right) \,.
    '''

    def __init__(self, ls, ms, ns, amps, Lx=1., Ly=1., Lz=1.):
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
            val += amp*np.cos(l*2.*np.pi/self._Lx*x + m*2.*np.pi/self._Ly*y + n*2.*np.pi/self._Lz*z)

        return val


class TorusModesSin:
    r'''Defines the callable
    
    .. math::
    
        u(\eta_1, \eta_2, \eta_3) = \sum_{i=0}^N \chi_i(\eta_1) \sin(m_i\,2\pi \eta_2 + n_i\,2\pi \eta_3) 

    where :math:`\chi_i(\eta_1)` is one of

    .. math::

        \chi_i(\eta_1) = A_i\sin(\pi\eta_1)\,,\qquad\quad \chi_i(\eta_1) = A_i\exp^{-(\eta_1 - r_0)^2/\sigma}
    '''

    def __init__(self, ms, ns, amps, pfuns='sin', pfun_params=None):
        r'''
        Parameters
        ----------
            ms : list[int]
                Poloidal mode numbers.

            ns : list[int]
                Toroidal mode numbers.

            pfuns : list[str]
                "sin" or "exp" to define the profile functions.

            amps : list[float]
                Amplitudes of each mode (m_i, n_i).

            pfun_params : list
                Provides :math:`[r_0, \sigma]` parameters for each "exp" profile fucntion, and None for "sin".
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
                self._pfuns += [lambda eta1 : np.sin(np.pi*eta1)]
            elif pfun == 'exp':
                self._pfuns += [lambda eta1 : np.exp(-(eta1 - params[0])**2/params[1])]
            else:
                raise ValueError('Profile function must be "sin" or "exp".')

    def __call__(self, eta1, eta2, eta3):

        val = 0.
        for mi, ni, pfun, amp in zip(self._ms, self._ns, self._pfuns, self._amps):
            val += amp * pfun(eta1) * np.sin(mi*2.*np.pi*eta2 + ni*2.*np.pi*eta3)

        return val
