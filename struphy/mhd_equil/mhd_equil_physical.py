import numpy as np

import struphy.mhd_equil.analytical.mhd_equil_slab     as mhd_slab
import struphy.mhd_equil.analytical.mhd_equil_cylinder as mhd_cylinder
import struphy.mhd_equil.analytical.mhd_equil_torus    as mhd_torus


class Equilibrium_mhd_physical:
    """
    Specifies an analytical ideal MHD equilibrium (p, B, J and rho) in (x, y, z) - coordinates

    Parameters
    ----------

        kind : str
            Type of equilibrium (see Notes)

        params : dict
            Equilibrium parameters from parameters.yml (key 'mhd_equilibrium')

    Notes
    -----
        The following types of MHD equilibria are available:

            * "slab"
            * "cylinder"
            * "torus"

    """
    
    def __init__(self, kind, params):
          
        self.kind   = kind
        self.params = params
        
        # create equilibrium object of given type
        if   self.kind == 'slab':
            self.EQ = mhd_slab.Equilibrium_mhd_slab(self.params)
        elif self.kind == 'cylinder':
            self.EQ = mhd_cylinder.Equilibrium_mhd_cylinder(self.params)
        elif self.kind == 'torus':
            self.EQ = mhd_torus.Equilibrium_mhd_torus(self.params)
        
         
    def p_eq(self, x, y, z):
        '''Equilibrium bulk pressure in physical space.'''
        return self.EQ.p_eq(x, y, z)
    
    def b_eq_x(self, x, y, z):
        '''Equilibrium magnetic field (x - component) in physical space.'''
        return self.EQ.b_eq_x(x, y, z)

    def b_eq_y(self, x, y, z):
        '''Equilibrium magnetic field (y - component) in physical space.'''
        return self.EQ.b_eq_y(x, y, z)
 
    def b_eq_z(self, x, y, z):
        '''Equilibrium magnetic field (z - component) in physical space.'''
        return self.EQ.b_eq_z(x, y, z)
    
    def b_eq(self, x, y, z):
        '''Equilibrium magnetic field (absolute value) in physical space.'''
        
        bx = self.EQ.b_eq_x(x, y, z)
        by = self.EQ.b_eq_y(x, y, z)
        bz = self.EQ.b_eq_z(x, y, z)
        
        return np.sqrt(bx**2 + by**2 + bz**2)
    
    def j_eq_x(self, x, y, z):
        '''Equilibrium current (x - component, curl of equilibrium magnetic field) in physical space.'''
        return self.EQ.j_eq_x(x, y, z)
 
    def j_eq_y(self, x, y, z):
        '''Equilibrium current (y - component, curl of equilibrium magnetic field) in physical space.'''
        return self.EQ.j_eq_y(x, y, z)

    def j_eq_z(self, x, y, z):
        '''Equilibrium current (z - component, curl of equilibrium magnetic field) in physical space.'''
        return self.EQ.j_eq_z(x, y, z)
    
    def r_eq(self, x, y, z):
        '''Equilibrium bulk density in physical space.'''
        return self.EQ.r_eq(x, y, z)