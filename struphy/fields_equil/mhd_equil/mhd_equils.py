from abc import ABCMeta, abstractmethod

class EquilibriumMHD(metaclass=ABCMeta):
    """
    Base class for MHD equilibria in Struphy.
    
    Parameters
    ----------
        params: dictionary
            Parameters that characterize the MHD equilibrium.
            
        DOMAIN: Domain obj, optional
            From struphy.geometry.domain_3d.Domain.        
    """
    
    def __init__(self, params, DOMAIN=None):
        
        # set parameters
        self._params = params
        
        # set domain object
        if DOMAIN is not None:
            self.DOMAIN = DOMAIN
    
    @property
    def params(self):
        """Parameters that characterize the MHD equilibrium."""
        return self._params
    
    @property
    def DOMAIN(self):
        """Domain object that characterizes the mapping from the logical to the physical domain."""
        return self._DOMAIN
    
    @DOMAIN.setter
    def DOMAIN(self, domain):
        """Domain object that characterizes the mapping from the logical to the physical domain."""
        self._DOMAIN = domain
    
    @abstractmethod
    def b_eq_x(self, x, y, z):
        """Equilibrium magnetic field (x-component) in physical space."""
        return    
    
    @abstractmethod
    def b_eq_y(self, x, y, z):
        """Equilibrium magnetic field (y-component) in physical space."""
        return
 
    @abstractmethod
    def b_eq_z(self, x, y, z):
        """Equilibrium magnetic field (z-component) in physical space."""
        return
    
    @abstractmethod
    def b_eq(self, x, y, z):
        """Equilibrium magnetic field (absolute value) in physical space."""
        return
    
    @abstractmethod
    def j_eq_x(self, x, y, z):
        """Equilibrium current (x-component, curl of equilibrium magnetic field) in physical space."""
        return
 
    @abstractmethod
    def j_eq_y(self, x, y, z):
        """Equilibrium current (y-component, curl of equilibrium magnetic field) in physical space."""
        return

    @abstractmethod
    def j_eq_z(self, x, y, z):
        """Equilibrium current (z-component, curl of equilibrium magnetic field) in physical space."""
        return
    
    @abstractmethod
    def p_eq(self, x, y, z):
        """Equilibrium pressure in physical space."""
        return
    
    @abstractmethod
    def n_eq(self, x, y, z):
        """Equilibrium number density in physical space."""
        return
    
    # equilibrium magnetic field (0-form absolute value on logical domain)
    def b0_eq(self, s, chi, phi):
        """0-form absolute value of equilibrium magnetic field in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull(self.b_eq, s, chi, phi, '0_form')
      
    # equilibrium magnetic field (1-form on logical domain, 1-component)
    def b1_eq_1(self, s, chi, phi):
        """1-form equilibrium magnetic field (1-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, '1_form_1')
        
    # equilibrium magnetic field (1-form on logical domain, 2-component)
    def b1_eq_2(self, s, chi, phi):
        """1-form equilibrium magnetic field (2-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, '1_form_2')

    # equilibrium magnetic field (1-form on logical domain, 3-component)
    def b1_eq_3(self, s, chi, phi):
        """1-form equilibrium magnetic field (3-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, '1_form_3')
    
    # equilibrium magnetic field (2-form on logical domain, 1-component)
    def b2_eq_1(self, s, chi, phi):
        """2-form equilibrium magnetic field (1-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, '2_form_1')
    
    # equilibrium magnetic field (2-form on logical domain, 2-component)
    def b2_eq_2(self, s, chi, phi):
        """2-form equilibrium magnetic field (2-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, '2_form_2')
        
    # equilibrium magnetic field (2-form on logical domain, 3-component)
    def b2_eq_3(self, s, chi, phi):
        """2-form equilibrium magnetic field (3-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, '2_form_3')

    # equilibrium magnetic field (vector on logical domain, 1-component)
    def bv_eq_1(self, s, chi, phi):
        """Vector equilibrium magnetic field (1-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, 'vector_1')

    # equilibrium magnetic field (vector on logical domain, 2-component)
    def bv_eq_2(self, s, chi, phi):
        """Vector equilibrium magnetic field (2-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, 'vector_2')
 
    # equilibrium magnetic field (vector on logical domain, 3-component)
    def bv_eq_3(self, s, chi, phi):
        """Vector equilibrium magnetic field (3-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], s, chi, phi, 'vector_3')
 
    # equilibrium current (2-form on logical domain, 1-component)
    def j2_eq_1(self, s, chi, phi):
        """2-form equilibrium current (1-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], s, chi, phi, '2_form_1')
   
    # equilibrium current (2-form on logical domain, 2-component)
    def j2_eq_2(self, s, chi, phi):
        """2-form equilibrium current (2-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], s, chi, phi, '2_form_2')
   
    # equilibrium current (2-form on logical domain, 3-component)
    def j2_eq_3(self, s, chi, phi):
        """2-form equilibrium current (3-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], s, chi, phi, '2_form_3')
      
    # equilibrium current (vector on logical domain, 1-component)
    def jv_eq_1(self, s, chi, phi):
        """Vector equilibrium current (1-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], s, chi, phi, 'vector_1')
     
    # equilibrium current (vector on logical domain, 2-component)
    def jv_eq_2(self, s, chi, phi):
        """Vector equilibrium current (2-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], s, chi, phi, 'vector_2')
       
    # equilibrium current (vector on logical domain, 3-component)
    def jv_eq_3(self, s, chi, phi):
        """Vector equilibrium current (3-component) in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], s, chi, phi, 'vector_3')
    
    # equilibrium pressure (0-form on logical domain)
    def p0_eq(self, s, chi, phi):
        """0-form equilibrium pressure in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull(self.p_eq, s, chi, phi, '0_form')
      
    # equilibrium pressure (3-form on logical domain)
    def p3_eq(self, s, chi, phi):
        """3-form equilibrium pressure in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull(self.p_eq, s, chi, phi, '3_form')
   
    # equilibrium density (0-form on logical domain)
    def n0_eq(self, s, chi, phi):
        """0-form equilibrium number density in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull(self.n_eq, s, chi, phi, '0_form')
     
    # equilibrium density (3-form on logical domain)
    def n3_eq(self, s, chi, phi):
        """3-form equilibrium number density in logical space."""
        assert hasattr(self, 'DOMAIN')
        return self.DOMAIN.pull(self.n_eq, s, chi, phi, '3_form')