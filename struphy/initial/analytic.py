import numpy as np

from struphy.initial.base import InitialMHD

class InitialMHDSlab(InitialMHD):
    """
    Defines the initial condition
    
    .. math::
    
        &\\textnormal{if}\,\,\, m\\neq 0:
        
        &U_x(t=0,x,y,z) = U\sin\left(\\frac{\pi x}{a}\\right)\sin\left(\\frac{my}{a} + \\frac{nz}{R_0}\\right)\,,

        &U_y(t=0,x,y,z) = U\cos\left(\\frac{\pi x}{a}\\right)\cos\left(\\frac{my}{a} + \\frac{nz}{R_0}\\right)\\frac{\pi}{m}\,,
        
        &U_z(t=0,x,y,z) = A\cos\left(\\frac{my}{a}\\right)\,,
        
        &p(t=0,x,y,z)=n(t=0,x,y,z)=\mathbf{B}(t=0,x,y,z)=0\,,
        
        &
        
        &\\textnormal{else}:
        
        &U_x(t=0,x,y,z) = 0\,,

        &U_y(t=0,x,y,z) = U\cos\left(\\frac{nz}{R_0}\\right)\,,
        
        &U_z(t=0,x,y,z) = A\cos\left(\\frac{my}{a}\\right)\,,
        
        &p(t=0,x,y,z)=n(t=0,x,y,z)=\mathbf{B}(t=0,x,y,z)=0\,,
    
    Parameters
    ----------
        params: dict
            Parameters that characterize the initial condition.

                * a  : minor radius (Lx=a, Ly=2*pi*a)
                * R0 : major radius (Lz=2*pi*R0)
                * m  : poloidal (y) mode number
                * n  : toroidal (z) mode number
                * U  : amplitude of Ux/Uy
                * A  : amplitude of Uz
            
        domain: struphy.geometry.domain_3d.Domain
            All things mapping. Enables pull-backs if set.       
    """
    
    def __init__(self, params=None, domain=None):
        
        # set default parameters
        if params is None:
            params_default = {'a'  : 1.,
                              'R0' : 3.,
                              'm'  : 0,
                              'n'  : 1,
                              'U'  : 1.,
                              'A' : 0.}
            
            super().__init__(params_default, domain)
            
        # or check if given parameter dictionary is complete
        else:
            assert 'a'  in params
            assert 'R0' in params
            assert 'm'  in params
            assert 'n'  in params
            assert 'U'  in params
            assert 'A'  in params
            
            super().__init__(params, domain)
            
        # set vector-valued functions
        self.u = [self.u_x, self.u_y, self.u_z]
        self.b = [self.b_x, self.b_y, self.b_z]

    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================
    
    # initial bulk pressure
    def p(self, x, y, z):

        pp = 0*x

        return pp
    
    # initial velocity field (x - component)
    def u_x(self, x, y, z):
        
        if self.params['m'] != 0:
            ux = self.params['U'] * np.sin(np.pi*x/self.params['a']) * np.sin(self.params['m']*y/self.params['a'] + self.params['n']*z/self.params['R0'])
        else:
            ux = 0*x

        return ux

    # initial velocity field (y - component)
    def u_y(self, x, y, z):
        
        if self.params['m'] != 0:
            uy = self.params['U']*np.pi/self.params['m'] * np.cos(np.pi*x/self.params['a']) * np.cos(self.params['m']*y/self.params['a'] + self.params['n']*z/self.params['R0'])
        else:
            uy = self.params['U'] * np.cos(self.params['n']*z/self.params['R0'])

        return uy

    # initial velocity field (z - component)
    def u_z(self, x, y, z):

        uz = self.params['A'] * np.cos(self.params['m']*y/self.params['a'])

        return uz
    
    # initial magnetic field (x - component)
    def b_x(self, x, y, z):
        
        bx = 0*x

        return bx

    # initial magnetic field (y - component)
    def b_y(self, x, y, z):
        
        by = 0*x

        return by

    # initial magnetic field (z - component)
    def b_z(self, x, y, z):

        bz = 0*x

        return bz
    
    # initial bulk number density
    def n(self, x, y, z):

        nn = 0*x

        return nn