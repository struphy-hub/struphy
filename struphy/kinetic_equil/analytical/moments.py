import numpy as np

class Kinetic_homogen_slab:
    """
    TODO
    """
    
    def __init__(self, params):
        
        # uniform density
        self.nh0 = params['nh0']

        # uniform flow velocity
        self.v0_x = params['v0_x']
        self.v0_y = params['v0_y']
        self.v0_z = params['v0_z']
        
        # uniform thermal velocities
        self.vth_x = params['vth_x']
        self.vth_y = params['vth_y']
        self.vth_z = params['vth_z']


    # equilibrium bulk pressure
    def nh_eq(self, x, y, z):
        return self.nh0 - 0*x

    def uh_eq_x(self, x, y, z):
        return self.v0_x - 0.*x

    def uh_eq_y(self, x, y, z):
        return self.v0_y - 0.*x

    def uh_eq_z(self, x, y, z):
        return self.v0_z - 0.*x

    def sig_x(self, x, y, z):
        return self.vth_x - 0.*x

    def sig_y(self, x, y, z):
        return self.vth_y - 0.*x

    def sig_z(self, x, y, z):
        return self.vth_z - 0.*x