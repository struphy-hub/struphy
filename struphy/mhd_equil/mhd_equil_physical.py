import numpy as np

import struphy.mhd_equil.analytical.mhd_equil_slab     as mhd_slab
import struphy.mhd_equil.analytical.mhd_equil_cylinder as mhd_cylinder
import struphy.mhd_equil.analytical.mhd_equil_torus    as mhd_torus


class Equilibrium_mhd_physical:
    """
    Specifies an analytical ideal MHD equilibrium (p, B, J and rho) in (x, y, z) - coordinates

    Parameters
    ----------

        kind : string
            Available equilibira (choices for kind) are
                - "slab"
                - "cylinder"
                - "torus"

        params : dict
            Equilibrium parameters
                - "slab" : TODO
                - "cylinder" : TODO
                - "torus" : TODO

    Methods
    -------
    p_eq
    b_eq_x
    b_eq_y
    b_eq_z
    b_eq
    j_eq_x
    j_eq_y
    j_eq_z
    r_eq

    """
    
    def __init__(self, kind, params):
          
        self.kind   = kind
        self.params = params
        
        # create equilibrium object of given type
        if   self.kind == 'slab':
            self.EQ = mhd_slab.equilibrium_mhd_slab(self.params)
        elif self.kind == 'cylinder':
            self.EQ = mhd_cylinder.equilibrium_mhd_cylinder(self.params)
        elif self.kind == 'torus':
            self.EQ = mhd_torus.equilibrium_mhd_torus(self.params)
        
         
    # equilibrium bulk pressure
    def p_eq(self, x, y, z):
        return self.EQ.p_eq(x, y, z)
    
    # equilibrium magnetic field (x - component)
    def b_eq_x(self, x, y, z):
        return self.EQ.b_eq_x(x, y, z)

    # equilibrium magnetic field (y - component)
    def b_eq_y(self, x, y, z):
        return self.EQ.b_eq_y(x, y, z)

    # equilibrium magnetic field (z - component)
    def b_eq_z(self, x, y, z):
        return self.EQ.b_eq_z(x, y, z)
    
    # equilibrium magnetic field (absolute value)
    def b_eq(self, x, y, z):
        
        bx = self.EQ.b_eq_x(x, y, z)
        by = self.EQ.b_eq_y(x, y, z)
        bz = self.EQ.b_eq_z(x, y, z)
        
        return np.sqrt(bx**2 + by**2 + bz**2)
    
    # equilibrium current (x - component, curl of equilibrium magnetic field)
    def j_eq_x(self, x, y, z):
        return self.EQ.j_eq_x(x, y, z)

    # equilibrium current (y - component, curl of equilibrium magnetic field)
    def j_eq_y(self, x, y, z):
        return self.EQ.j_eq_y(x, y, z)

    # equilibrium current (z - component, curl of equilibrium magnetic field)
    def j_eq_z(self, x, y, z):
        return self.EQ.j_eq_z(x, y, z)
    
    # equilibrium bulk density
    def r_eq(self, x, y, z):
        return self.EQ.r_eq(x, y, z)