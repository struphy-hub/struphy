#!/usr/bin/env python3

import numpy as np

class Modes_sin:
    '''Defines the callable
    
    .. math::
    
        u(x, y, z) = \sum_{i=0}^N A_i \sin(k_{x,i} x + k_{y,i} y + k_{z,i} z) \,.
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
                Amplitude of each mode k = (k1, k2, k3).
        '''

        assert len(k1s) == len(k2s)
        assert len(k1s) == len(k3s)
        assert len(k1s) == len(amps)

        self._k1s = k1s
        self._k2s = k2s
        self._k3s = k3s
        self._amps = amps

    def __call__(self, x, y, z):

        val = 0.
        for k1, k2, k3, amp in zip(self._k1s, self._k2s, self._k3s, self._amps):
            val += amp*np.sin(k1*x + k2*y + k3*z)

        return val


class Modes_cos:
    '''Defines the callable
    
    .. math::
    
        u(x, y, z) = \sum_{i=0}^N B_i \cos(k_{x,i} x + k_{y,i} y + k_{z,i} z) \,.

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
                Amplitude of each mode k = (k1, k2, k3).
        '''

        assert len(k1s) == len(k2s)
        assert len(k1s) == len(k3s)
        assert len(k1s) == len(amps)

        self._k1s = k1s
        self._k2s = k2s
        self._k3s = k3s
        self._amps = amps

    def __call__(self, x, y, z):

        val = 0.
        for k1, k2, k3, amp in zip(self._k1s, self._k2s, self._k3s, self._amps):
            val += amp*np.cos(k1*x + k2*y + k3*z)

        return val
