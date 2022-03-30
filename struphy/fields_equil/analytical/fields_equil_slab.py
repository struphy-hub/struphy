import numpy as np

class Equilibrium_fields_slab:
    """
    Gives the analytical form of the slab solution to the Maxwell equations
    """

    def __init__(self, params):
        # it's zero what do you want?
        self.params = params

        # uniform magnetic field
        self.b0x = params['B0x']
        self.b0y = params['B0y']
        self.b0z = params['B0z']

        # uniform electric field
        self.e0x = params['E0x']
        self.e0y = params['E0y']
        self.e0z = params['E0z']
    
    def e_eq_x(self, x, y, z):
        return self.e0x - 0*x
    
    def e_eq_y(self, x, y, z):
        return self.e0y - 0*x
    
    def e_eq_z(self, x, y, z):
        return self.e0z - 0*x
    
    def b_eq_x(self, x, y, z):
        return self.b0x - 0*x
    
    def b_eq_y(self, x, y, z):
        return self.b0y - 0*x
    
    def b_eq_z(self, x, y, z):
        return self.b0z - 0*x
    