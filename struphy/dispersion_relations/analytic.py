import numpy as np
import scipy.special as sp

from struphy.dispersion_relations.base import DispersionRelations1D


class Maxwell1D(DispersionRelations1D):
    r'''Dispersion relation for Maxwell's equation in vacuum in Struphy units (see ``Maxwell`` in :ref:`models`):
    
    .. math::
    
        \omega^2 = k^2 \,.
    '''

    def __init__(self, **params):
        super().__init__('light wave') 

    def __call__(self, kvec, kperp=None):

        # One complex array for each branch
        tmps = []
        for n in range(self.nbranches):
            tmps += [np.zeros_like(kvec, dtype=complex)]

        ########### Model specific part ##############################
        # first branch
        tmps[0][:] = kvec
        ##############################################################

        # fill output dictionary
        dict_disp = {}
        for name, tmp in zip(self.branches, tmps):
            dict_disp[name] = tmp

        return dict_disp
    
    
class Mhd1D(DispersionRelations1D):
    r'''Dispersion relation for linear MHD equations for homogeneous background :math:`(n_0,p_0,\mathbf B_0)` and wave propagation along z-axis in Struphy units (see ``LinearMHD`` in :ref:`models`):
    
    .. math::
    
        \textnormal{shear Alfvén}:\quad &\omega^2 = c_\textnormal{A}^2 k^2\frac{B_{0z}^2}{|\mathbf B_0|^2}\,,
        
        \textnormal{fast (+) and slow (-) magnetosonic}:\quad &\omega^2 =\frac{1}{2}(c_\textnormal{S}^2+c_\textnormal{A}^2)k^2(1\pm\sqrt{1-\delta}\,)\,,\quad\delta=\frac{4B_{0z}^2c_\textnormal{S}^2c_\textnormal{A}^2}{(c_\textnormal{S}^2+c_\textnormal{A}^2)^2|\mathbf B_0|^2}\,,
        
    where :math:`c_\textnormal{A}^2=|\mathbf B_0|^2/n_0` is the Alfvén velocity and :math:`c_\textnormal{S}^2=\gamma\,p_0/n_0` is the speed of sound.
    '''

    def __init__(self, **params):
        super().__init__('shear Alfvén', 'slow magnetosonic', 'fast magnetosonic', **params) 

    def __call__(self, kvec, kperp=None):

        # One complex array for each branch
        tmps = []
        for n in range(self.nbranches):
            tmps += [np.zeros_like(kvec, dtype=complex)]

        ########### Model specific part ##############################
        
        # Alfvén velocity and speed of sound
        cA = np.sqrt((self.params['B0x']**2 + self.params['B0y']**2 + self.params['B0z']**2)/self.params['n0'])
        cS = np.sqrt(self.params['gamma']*self.params['p0']/self.params['n0']) 
        
        # shear Alfvén branch
        tmps[0][:] = cA * kvec * self.params['B0z']/np.sqrt(self.params['B0x']**2 + self.params['B0y']**2 + self.params['B0z']**2)
        
        # slow/fast magnetosonic branch
        delta = (4*self.params['B0z']**2*cS**2*cA**2)/((cS**2 + cA**2)**2*(self.params['B0x']**2 + self.params['B0y']**2 + self.params['B0z']**2))
        
        tmps[1][:] = np.sqrt(1/2*kvec**2*(cS**2 + cA**2)*(1 - np.sqrt(1 - delta)))
        tmps[2][:] = np.sqrt(1/2*kvec**2*(cS**2 + cA**2)*(1 + np.sqrt(1 - delta)))
        
        ##############################################################

        # fill output dictionary
        dict_disp = {}
        for name, tmp in zip(self.branches, tmps):
            dict_disp[name] = tmp

        return dict_disp