#!/usr/bin/env python3

import numpy as np

class ModesSin:
    r'''Defines the callable
    
    .. math::
    
        u(x, y, z) = \sum_{o=1}^{N_x}\sum_{m=1}^{N_y}\sum_{n=1}^{N_z} A_{omn} \sin(k_{x,o} x + k_{y,m} y + k_{z,n} z) \,.
    '''

    def __init__(self, k1s, k2s, k3s, amps):
        '''
        Parameters
        ----------
            k1s : list
                Mode numbers in x-direction, k1 = o*2*pi/Lx.

            k2s : list
                Mode numbers in y-direction, k2 = m*2*pi/Ly.

            k3s : list
                Mode numbers in z-direction, k3 = n*2*pi/Lz.

            amps : list
                Amplitude of each mode k = (k1, k2, k3), must be a 3d list such that amps[o][m][n] is the amplitude of mode (o,m,m).
        '''

        self._k1s = k1s
        self._k2s = k2s
        self._k3s = k3s
        self._amps = amps

    def __call__(self, x, y, z):
        
        val = 0.
        
        for o, k1 in enumerate(self._k1s):
            for m, k2 in enumerate(self._k2s):
                for n, k3 in enumerate(self._k3s):
                    val += self._amps[o][m][n]*np.sin(k1*x + k2*y + k3*z)

        return val


class ModesCos:
    r'''Defines the callable
    
    .. math::
    
        u(x, y, z) = \sum_{o=1}^{N_x}\sum_{m=1}^{N_y}\sum_{n=1}^{N_z} A_{omn} \cos(k_{x,o} x + k_{y,m} y + k_{z,n} z) \,.
    '''

    def __init__(self, k1s, k2s, k3s, amps):
        '''
        Parameters
        ----------
            k1s : list
                Mode numbers in x-direction, k1 = 2*pi/Lx.

            k2s : list
                Mode numbers in y-direction, k2 = 2*pi/Ly.

            k3s : list
                Mode numbers in z-direction, k3 = 2*pi/Lz.

            amps : list
                Amplitude of each mode k = (k1, k2, k3), must be a 3d list such that amps[o][m][n] is the amplitude of mode (o,m,m).
        '''

        self._k1s = k1s
        self._k2s = k2s
        self._k3s = k3s
        self._amps = amps

    def __call__(self, x, y, z):

        val = 0.
        
        for o, k1 in enumerate(self._k1s):
            for m, k2 in enumerate(self._k2s):
                for n, k3 in enumerate(self._k3s):
                    val += self._amps[o][m][n]*np.cos(k1*x + k2*y + k3*z)

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
