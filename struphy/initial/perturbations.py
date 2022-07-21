#!/usr/bin/env python3

import numpy as np

class ModesSin:
    '''Defines the callable
    
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
    '''Defines the callable
    
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
