# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Analytical coordinate transformation theta(eta1, eta2) for circular torus correspondong to ad hoc magnetic field (straight field line or equal arc).
"""

import numpy as np

def theta(eta1, eta2, a, R0, kind):
    """Analytical coordinate transformations theta(eta1, eta2) for circular torus with ad hoc magnetic field.
    
    Parameters
    ----------
    eta1 : 2d np.array
        1st logical (radial) coordinate
        
    eta2 : 2d np.array
        2nd logical (angular) coordinate
        
    a : float
        minor radius of torus
        
    R0 : float
        major radius of torus
        
    kind : string
        which transformation : 'straight' or 'equal arc'
        
    Returns
    -------
    th : 2d np.array
        the geometrical poloidal angle of classical toroidal coordinates
    """
    
    th = np.zeros(eta1.shape, dtype=float)
    
    chi = 2*np.pi*eta2
    r   = a*eta1
    
    assert kind == 'straight' or kind == 'equal arc'
    
    if kind == 'straight':
        
        for i in range(th.shape[0]):
            for j in range(th.shape[1]):
                
                if chi[i, j] == np.pi:
                
                    th[i, j] = np.pi
                    
                elif chi[i, j] > np.pi:
                    
                    eps = r[i, j]/R0
                    
                    th[i, j] = 2*np.arctan(np.sqrt((1 + eps)/(1 - eps))*np.tan(chi[i, j]/2)) + 2*np.pi
                    
                else:
                    
                    eps = r[i, j]/R0
                    
                    th[i, j] = 2*np.arctan(np.sqrt((1 + eps)/(1 - eps))*np.tan(chi[i, j]/2))
                    
    elif kind == 'equal arc':
        
        th[:, :] = chi
                    
    return th