import numpy as np

from struphy.geometry import domain_3d

import struphy.fields_equil.analytical.fields_equil_slab as slab


class Equilibrium_fields_physical:
    """
    Returns an equilibrium solution to the Maxwell equations for the E-field and B-field

    Parameters
    ----------
    params_fields : dict
        Keys are "type", 
    """

    def __init__(self, type, params_fields):
        
        self.type       = type
        self.params     = params_fields

        if self.type == 'slab':
            
            self.EQ = slab.Equilibrium_fields_slab(self.params)

        else:
            raise ValueError('This type of solution has not been implemented.')
    
    def e_eq_x(self, x, y, z):
        '''X-component of the equilibrium electric field'''
        return self.EQ.e_eq_x(x, y, z)
    
    def e_eq_y(self, x, y, z):
        '''Y-component of the equilibrium electric field'''
        return self.EQ.e_eq_y(x, y, z)
    
    def e_eq_z(self, x, y, z):
        '''Z-component of the equilibrium electric field'''
        return self.EQ.e_eq_z(x, y, z)
    


    def b_eq_x(self, x, y, z):
        '''X-component of the equilibrium magnetic field'''
        return self.EQ.b_eq_x(x, y, z)
    
    def b_eq_y(self, x, y, z):
        '''Y-component of the equilibrium magnetic field'''
        return self.EQ.b_eq_y(x, y, z)
    
    def b_eq_z(self, x, y, z):
        '''Z-component of the equilibrium magnetic field'''
        return self.EQ.b_eq_z(x, y, z)
    

    
    def e_eq(self, x, y, z):
        '''Absolute value of the equilibrium electric field'''

        ex = self.EQ.e_eq_x(x, y, z)
        ey = self.EQ.e_eq_y(x, y, z)
        ez = self.EQ.e_eq_z(x, y, z)

        return np.sqrt(ex**2 + ey**2 + ez**2)
    
    def b_eq(self, x, y, z):
        '''Absolute value of the equilibrium magnetic field'''

        bx = self.EQ.b_eq_x(x, y, z)
        by = self.EQ.b_eq_y(x, y, z)
        bz = self.EQ.b_eq_z(x, y, z)

        return np.sqrt(bx**2 + by**2 + bz**2)
