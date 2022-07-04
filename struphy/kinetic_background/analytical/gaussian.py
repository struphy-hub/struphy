import numpy as np

from struphy.geometry import domain_3d

class Gaussian_3d:
    '''Gaussian with three velocity shifts and temperatures. Flat evaluation.
    
    Parameters
    ----------
        MOMENTS : obj
            Object holding the velocity moments obtained from "kinetic_background/analytical/moments".

    Notes
    -----
        * | G(x, y, z, vx, vy, vz) = exp( -(vx - v0_x(x, y, z))^2/vth_x(x, y, z)^2 ) / sqrt(pi) / vth_x(x, y, z)
          |                        * exp( -(vy - v0_y(x, y, z))^2/vth_y(x, y, z)^2 ) / sqrt(pi) / vth_y(x, y, z)
          |                        * exp( -(vz - v0_z(x, y, z))^2/vth_z(x, y, z)^2 ) / sqrt(pi) / vth_z(x, y, z)                    
    '''

    def __init__(self, MOMENTS):

        self.MOMENTS = MOMENTS


    def velocity_distribution(self, x, y, z, vx, vy, vz):
        '''Gaussian, normalized to 1. Flat evaluation.'''

        assert x.shape == y.shape == z.shape == vx.shape == vy.shape == vz.shape

        Gx = np.exp(-(vx - self.uh_eq_x(x, y, z))**2 / self.sig_x(x, y, z)**2) / self.sig_x(x, y, z) / np.sqrt(np.pi)
        Gy = np.exp(-(vy - self.uh_eq_y(x, y, z))**2 / self.sig_y(x, y, z)**2) / self.sig_y(x, y, z) / np.sqrt(np.pi)
        Gz = np.exp(-(vz - self.uh_eq_z(x, y, z))**2 / self.sig_z(x, y, z)**2) / self.sig_z(x, y, z) / np.sqrt(np.pi)
        
        value = Gx * Gy * Gz
    
        return value

    def uh_eq_x(self, x, y, z):
        return self.MOMENTS.uh_eq_x(x, y, z)

    def uh_eq_y(self, x, y, z):
        return self.MOMENTS.uh_eq_y(x, y, z)

    def uh_eq_z(self, x, y, z):
        return self.MOMENTS.uh_eq_z(x, y, z)

    def sig_x(self, x, y, z):
        return self.MOMENTS.sig_x(x, y, z)

    def sig_y(self, x, y, z):
        return self.MOMENTS.sig_y(x, y, z)

    def sig_z(self, x, y, z):
        return self.MOMENTS.sig_z(x, y, z)

    


