import numpy as np

class EquilibriumMhdSlab:
    """
    TODO
    """
    
    def __init__(self, params):
        
        # uniform magnetic field
        self.b0x = params['B0x']
        self.b0y = params['B0y']
        self.b0z = params['B0z']
        
        # plasma beta
        self.beta = params['beta']

    
    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================
    
    # equilibrium magnetic field (x - component)
    def b_eq_x(self, x, y, z):
        
        bx = self.b0x - 0*x

        return bx

    # equilibrium magnetic field (y - component)
    def b_eq_y(self, x, y, z):
        
        by = self.b0y - 0*x

        return by

    # equilibrium magnetic field (z - component)
    def b_eq_z(self, x, y, z):

        bz = self.b0z - 0*x

        return bz

    
    # equilibrium current (x - component, curl of equilibrium magnetic field)
    def j_eq_x(self, x, y, z):

        jx = 0*x

        return jx

    # equilibrium current (y - component, curl of equilibrium magnetic field)
    def j_eq_y(self, x, y, z):

        jy = 0*x

        return jy

    # equilibrium current (z - component, curl of equilibrium magnetic field)
    def j_eq_z(self, x, y, z):
        
        jz = 0*x

        return jz
    
    
    # equilibrium bulk pressure
    def p_eq(self, x, y, z):
        
        p = self.beta/200*(self.b0x**2 + self.b0y**2 + self.b0z**2) - 0*x

        return p

    
    # equilibrium bulk number density
    def n_eq(self, x, y, z):
        
        n = 1 - 0*x

        return n
