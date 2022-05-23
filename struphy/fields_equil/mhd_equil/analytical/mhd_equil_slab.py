import numpy as np

from struphy.fields_equil.mhd_equil.mhd_equils import EquilibriumMHD

class EquilibriumMHDSlab(EquilibriumMHD):
    """
    Homogeneous MHD equilibrium in slab geometry.
    
    The profiles in cartesian coordinates (x, y, z) are:
    
    B = B0x*e_x + B0y*e_y + B0z*e_z,
    p = beta/2*( B0x^2 + B0y^2 + B0z^2 ),
    n = 1.
    
    Parameters
    ..........
        params: dictionary
            Parameters that characterize the MHD equilibrium.
                * B0x  : magnetic field in x-direction
                * B0y  : magnetic field in y-direction
                * B0z  : magnetic field in z-direction
                * beta : plasma beta in % (ratio of kinetic pressure to magnetic pressure)
            
        DOMAIN: Domain obj (optional)
            From struphy.geometry.domain_3d.Domain.       
    """
    
    def __init__(self, params={'B0x' : 0., 'B0y' : 0., 'B0z' : 1., 'beta' : 100.}, DOMAIN=None):
        
        # check that parameter dicitionary is complete
        assert 'B0x' in params
        assert 'B0y' in params
        assert 'B0z' in params
        
        assert 'beta' in params
        
        super().__init__(params, DOMAIN)
    
    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================
    
    # equilibrium magnetic field (x-component)
    def b_eq_x(self, x, y, z):
        """Equilibrium magnetic field (x-component)."""
        bx = self.params['B0x'] - 0*x

        return bx

    # equilibrium magnetic field (y-component)
    def b_eq_y(self, x, y, z):
        """Equilibrium magnetic field (y-component)."""
        by = self.params['B0y'] - 0*x

        return by

    # equilibrium magnetic field (z-component)
    def b_eq_z(self, x, y, z):
        """Equilibrium magnetic field (z-component)."""
        bz = self.params['B0z'] - 0*x

        return bz
    
    # equilibrium magnetic field (absolute value)
    def b_eq(self, x, y, z):
        """Equilibrium magnetic field (absolute value)."""
        bx = self.b_eq_x(x, y, z)
        by = self.b_eq_y(x, y, z)
        bz = self.b_eq_z(x, y, z)
        
        return np.sqrt(bx**2 + by**2 + bz**2)
    
    # equilibrium current (x-component, curl of equilibrium magnetic field)
    def j_eq_x(self, x, y, z):
        """Equilibrium current (x-component)."""
        jx = 0*x

        return jx

    # equilibrium current (y-component, curl of equilibrium magnetic field)
    def j_eq_y(self, x, y, z):
        """Equilibrium current (y-component)."""
        jy = 0*x

        return jy

    # equilibrium current (z-component, curl of equilibrium magnetic field)
    def j_eq_z(self, x, y, z):
        """Equilibrium current (z-component)."""
        jz = 0*x

        return jz
       
    # equilibrium pressure
    def p_eq(self, x, y, z):
        """Equilibrium pressure."""
        p = self.params['beta']/200*(self.params['B0x']**2 + self.params['B0y']**2 + self.params['B0z']**2) - 0*x

        return p
 
    # equilibrium number density
    def n_eq(self, x, y, z):
        """Equilibrium number density."""
        n = 1 - 0*x

        return n
