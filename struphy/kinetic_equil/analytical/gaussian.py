import numpy as np

from struphy.geometry import domain_3d

class Gaussian_3d:
    '''Gaussian with three velocity shifts and temperatures. Flat evaluation.
    
    Parameters
    ----------
    params : dict
        Parameters from parameters.yml

    Notes
    -----
    * | G(x, y, z, vx, vy, vz) = exp( -(vx - v0_x(x, y, z))^2/vth_x(x, y, z)^2 ) / sqrt(pi) / vth_x(x, y, z)
      |                        * exp( -(vy - v0_y(x, y, z))^2/vth_y(x, y, z)^2 ) / sqrt(pi) / vth_y(x, y, z)
      |                        * exp( -(vz - v0_z(x, y, z))^2/vth_z(x, y, z)^2 ) / sqrt(pi) / vth_z(x, y, z)                    
    '''

    def __init__(self, params):

        # parameters for homogeneous shift and temperature
        self.vth_x = params['vth_x']
        self.vth_y = params['vth_y']
        self.vth_z = params['vth_z']
        self.v0_x  = params['v0_x']
        self.v0_y  = params['v0_y']
        self.v0_z  = params['v0_z']

    def velocity_distribution(self, x, y, z, vx, vy, vz):
        '''Gaussian, normalized to 1. Flat evaluation.'''

        assert x.shape == y.shape == z.shape == vx.shape == vy.shape == vz.shape

        Gx = np.exp(-(vx - self.uh_x(x, y, z))**2 / self.sig_x(x, y, z)**2) / self.sig_x(x, y, z) / np.sqrt(np.pi)
        Gy = np.exp(-(vy - self.uh_y(x, y, z))**2 / self.sig_y(x, y, z)**2) / self.sig_y(x, y, z) / np.sqrt(np.pi)
        Gz = np.exp(-(vz - self.uh_z(x, y, z))**2 / self.sig_z(x, y, z)**2) / self.sig_z(x, y, z) / np.sqrt(np.pi)
        
        value = Gx * Gy * Gz
    
        return value

    def uh_x(self, x, y, z):
        return self.v0_x - 0.*x

    def uh_y(self, x, y, z):
        return self.v0_y - 0.*x

    def uh_z(self, x, y, z):
        return self.v0_z - 0.*x

    def sig_x(self, x, y, z):
        return self.vth_x - 0.*x

    def sig_y(self, x, y, z):
        return self.vth_y - 0.*x

    def sig_z(self, x, y, z):
        return self.vth_z - 0.*x

    


