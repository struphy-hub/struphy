import numpy as np

from struphy.geometry import domain_3d

class Maxwell_pitchangle:
    '''Maxwellian in (x, y, z)-coordinates. Point-wise evaluation.
    
    Parameters
    ----------
    params : dict
        Parameters from parameters.yml

    Notes
    -----
    * | M0(x, y, z, vx, vy, vz) = n0(x, y, z)
      |                         / sqrt(pi)/vth_x(x, y, z) * exp( -(vx - v0_x(x, y, z))^2/vth_x(x, y, z)^2 )
      |                         / sqrt(pi)/vth_y(x, y, z) * exp( -(vy - v0_y(x, y, z))^2/vth_y(x, y, z)^2 )
      |                         / sqrt(pi)/vth_z(x, y, z) * exp( -(vz - v0_z(x, y, z))^2/vth_z(x, y, z)^2 )                     
    '''

    # TODO: everything (this is the Maxwell_xyz case)

    def __init__(self, params):

        # parameters for homogeneous shift and temperature
        self.vth_x = params['vth_x']
        self.vth_y = params['vth_y']
        self.vth_z = params['vth_z']
        self.v0_x  = params['v0_x']
        self.v0_y  = params['v0_y']
        self.v0_z  = params['v0_z']

    def nh(self, x, y, z):
        #E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(x, y, z)
        return 1. #- 0.*E1

    def uh_x(self, x, y, z):
        #E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(x, y, z)
        return self.v0_x #- 0.*E1

    def uh_y(self, x, y, z):
        #E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(x, y, z)
        return self.v0_y #- 0.*E1

    def uh_z(self, x, y, z):
        #E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(x, y, z)
        return self.v0_z #- 0.*E1

    def sig_x(self, x, y, z):
        #E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(x, y, z)
        return self.vth_x #- 0.*E1

    def sig_y(self, x, y, z):
        #E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(x, y, z)
        return self.vth_y #- 0.*E1

    def sig_z(self, x, y, z):
        #E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(x, y, z)
        return self.vth_z #- 0.*E1

    def fh(self, x, y, z, vx, vy, vz):
        '''Hot particle Maxwellian. Point-wise evaluation.'''

        Gx = np.exp(-(vx - self.uh_x(x, y, z))**2 / self.sig_x(x, y, z)**2) / self.sig_x(x, y, z) / np.sqrt(np.pi)
        Gy = np.exp(-(vy - self.uh_y(x, y, z))**2 / self.sig_y(x, y, z)**2) / self.sig_y(x, y, z) / np.sqrt(np.pi)
        Gz = np.exp(-(vz - self.uh_z(x, y, z))**2 / self.sig_z(x, y, z)**2) / self.sig_z(x, y, z) / np.sqrt(np.pi)
        
        value = self.nh(x, y, z) * Gx * Gy * Gz
    
        return value


