import numpy as np

from struphy.mhd_equil.analytical.mhd_equil_slab        import EquilibriumMhdSlab
from struphy.mhd_equil.analytical.mhd_equil_shearedslab import EquilibriumMhdShearedSlab
from struphy.mhd_equil.analytical.mhd_equil_cylinder    import EquilibriumMhdCylinder
from struphy.mhd_equil.analytical.mhd_equil_torus       import EquilibriumMhdTorus


class EquilibriumMhdPhysical:
    """
    Specifies an analytical ideal MHD equilibrium (p, B, J and n) in (x, y, z) - coordinates

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
            * "sheared slab"
            * "cylinder"
            * "torus"

    """
    
    def __init__(self, kind, params):
          
        self.kind   = kind
        self.params = params
        
        # create equilibrium object of given type
        if   self.kind == 'slab':
            self.eq = EquilibriumMhdSlab(self.params)
        elif self.kind == 'sheared slab':
            self.eq = EquilibriumMhdShearedSlab(self.params)
        elif self.kind == 'cylinder':
            self.eq = EquilibriumMhdCylinder(self.params)
        elif self.kind == 'torus':
            self.eq = EquilibriumMhdTorus(self.params)
        
         
    def b_eq_x(self, x, y, z):
        """Equilibrium magnetic field (x - component) in physical space."""
        return self.eq.b_eq_x(x, y, z)     
    
    def b_eq_y(self, x, y, z):
        """Equilibrium magnetic field (y - component) in physical space."""
        return self.eq.b_eq_y(x, y, z)
 
    def b_eq_z(self, x, y, z):
        """Equilibrium magnetic field (z - component) in physical space."""
        return self.eq.b_eq_z(x, y, z)
    
    def b_eq(self, x, y, z):
        """Equilibrium magnetic field (absolute value) in physical space."""
        
        bx = self.eq.b_eq_x(x, y, z)
        by = self.eq.b_eq_y(x, y, z)
        bz = self.eq.b_eq_z(x, y, z)
        
        return np.sqrt(bx**2 + by**2 + bz**2)
    
    
    def j_eq_x(self, x, y, z):
        """Equilibrium current (x - component, curl of equilibrium magnetic field) in physical space."""
        return self.eq.j_eq_x(x, y, z)
 
    def j_eq_y(self, x, y, z):
        """Equilibrium current (y - component, curl of equilibrium magnetic field) in physical space."""
        return self.eq.j_eq_y(x, y, z)

    def j_eq_z(self, x, y, z):
        """Equilibrium current (z - component, curl of equilibrium magnetic field) in physical space."""
        return self.eq.j_eq_z(x, y, z)
    
    
    def p_eq(self, x, y, z):
        """Equilibrium bulk pressure in physical space."""
        return self.eq.p_eq(x, y, z)
    
    
    def n_eq(self, x, y, z):
        """Equilibrium bulk density in physical space."""
        return self.eq.n_eq(x, y, z)