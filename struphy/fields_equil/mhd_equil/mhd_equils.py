from abc import ABCMeta, abstractmethod

class EquilibriumMHD(metaclass=ABCMeta):
    
    def __init__(self):
        pass
    
    
    @abstractmethod
    def b_eq_x(self, x, y, z):
        """Equilibrium magnetic field (x - component) in physical space."""
        return    
    
    @abstractmethod
    def b_eq_y(self, x, y, z):
        """Equilibrium magnetic field (y - component) in physical space."""
        return
 
    @abstractmethod
    def b_eq_z(self, x, y, z):
        """Equilibrium magnetic field (z - component) in physical space."""
        return
    
    @abstractmethod
    def b_eq(self, x, y, z):
        """Equilibrium magnetic field (absolute value) in physical space."""
        return
    
    
    @abstractmethod
    def j_eq_x(self, x, y, z):
        """Equilibrium current (x - component, curl of equilibrium magnetic field) in physical space."""
        return
 
    @abstractmethod
    def j_eq_y(self, x, y, z):
        """Equilibrium current (y - component, curl of equilibrium magnetic field) in physical space."""
        return

    @abstractmethod
    def j_eq_z(self, x, y, z):
        """Equilibrium current (z - component, curl of equilibrium magnetic field) in physical space."""
        return
    
    
    @abstractmethod
    def p_eq(self, x, y, z):
        """Equilibrium bulk pressure in physical space."""
        return
    
    @abstractmethod
    def n_eq(self, x, y, z):
        """Equilibrium bulk density in physical space."""
        return
        
    
    
    # call this function to enable pullbacks of fields!
    def enable_pullbacks(self, DOMAIN):
        self.DOMAIN = DOMAIN
    
    
    
    # equilibrium magnetic field (0-form absolute value on logical domain)
    def b0_eq(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull(self.b_eq, s, chi, phi, '0_form')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
      

    # equilibrium magnetic field (1-form on logical domain, 1-component)
    def b1_eq_1(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, '1_form_1')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
        
 
    # equilibrium magnetic field (1-form on logical domain, 2-component)
    def b1_eq_2(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, '1_form_2')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
        
    
    # equilibrium magnetic field (1-form on logical domain, 3-component)
    def b1_eq_3(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, '1_form_3')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
    
    
    # equilibrium magnetic field (2-form on logical domain, 1-component)
    def b2_eq_1(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, '2_form_1')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
 
    
    # equilibrium magnetic field (2-form on logical domain, 2-component)
    def b2_eq_2(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, '2_form_2')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
        
    
    # equilibrium magnetic field (2-form on logical domain, 3-component)
    def b2_eq_3(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, '2_form_3')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
    
    
    # equilibrium magnetic field (vector on logical domain, 1-component)
    def bv_eq_1(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, 'vector_1')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
 
    
    # equilibrium magnetic field (vector on logical domain, 2-component)
    def bv_eq_2(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, 'vector_2')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
        
    
    # equilibrium magnetic field (vector on logical domain, 3-component)
    def bv_eq_3(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, 'vector_3')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
    
    
    
    # equilibrium current (2-form on logical domain, 1-component)
    def j2_eq_1(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], s, chi, phi, '2_form_1')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
        
        
    # equilibrium current (2-form on logical domain, 2-component)
    def j2_eq_2(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], s, chi, phi, '2_form_2')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
        
        
    # equilibrium current (2-form on logical domain, 3-component)
    def j2_eq_3(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], s, chi, phi, '2_form_3')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
    
    
    # equilibrium current (vector on logical domain, 1-component)
    def jv_eq_1(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], s, chi, phi, 'vector_1')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
        
        
    # equilibrium current (vector on logical domain, 2-component)
    def jv_eq_2(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], s, chi, phi, 'vector_2')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
        
        
    # equilibrium current (vector on logical domain, 3-component)
    def jv_eq_3(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], s, chi, phi, 'vector_3')
    
    
    
    # equilibrium bulk pressure (0-form on logical domain)
    def p0_eq(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull(self.p_eq, s, chi, phi, '0_form')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
           
    
    # equilibrium bulk pressure (3-form on logical domain)
    def p3_eq(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull(self.p_eq, s, chi, phi, '3_form')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
    
    
    
    # equilibrium bulk density (0-form on logical domain)
    def n0_eq(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull(self.n_eq, s, chi, phi, '0_form')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')
            

    # equilibrium bulk density (3-form on logical domain)
    def n3_eq(self, s, chi, phi):
        
        if hasattr(self, 'DOMAIN'):
            return self.DOMAIN.pull(self.n_eq, s, chi, phi, '3_form')
        else:
            print('enable_pullbacks(DOMAIN) must be called first!')