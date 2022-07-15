import numpy as np

class InitialMHD:
    """
    Base class for MHD initial conditions.
    
    Parameters
    ----------
        params: dictionary
            Parameters that characterize the MHD equilibrium.
            
        domain: Domain, optional
            From struphy.geometry.domain_3d. Enables pull-backs if set.        
    """
    
    def __init__(self, params, domain=None):
        
        # set parameters
        self._params = params
        
        # set domain object
        if domain is not None:
            self._domain = domain
            
        # set vector-valued functions as list of callables
        self.u1 = [self.u1_1, self.u1_2, self.u1_3]
        self.u2 = [self.u2_1, self.u2_2, self.u2_3]
        self.uv = [self.uv_1, self.uv_2, self.uv_3]
        self.b2 = [self.b2_1, self.b2_2, self.b2_3]
    
    @property
    def params(self):
        """ Parameters that characterize the MHD equilibrium.
        """
        return self._params
    
    @property
    def domain(self):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        return self._domain
    
    @domain.setter
    def domain(self, domain):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        self._domain = domain

    def u1_1(self, s, chi, phi):
        """ 2-form initial magnetic field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], s, chi, phi, '1_form_1')
    
    def u1_2(self, s, chi, phi):
        """ 2-form initial magnetic field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], s, chi, phi, '1_form_2')
        
    def u1_3(self, s, chi, phi):
        """ 2-form initial magnetic field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], s, chi, phi, '1_form_3')
    
    def u2_1(self, s, chi, phi):
        """ 2-form initial magnetic field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], s, chi, phi, '2_form_1')
    
    def u2_2(self, s, chi, phi):
        """ 2-form initial magnetic field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], s, chi, phi, '2_form_2')
        
    def u2_3(self, s, chi, phi):
        """ 2-form initial magnetic field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], s, chi, phi, '2_form_3')
    
    def uv_1(self, s, chi, phi):
        """ 2-form initial magnetic field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], s, chi, phi, 'vector_1')
    
    def uv_2(self, s, chi, phi):
        """ 2-form initial magnetic field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], s, chi, phi, 'vector_2')
        
    def uv_3(self, s, chi, phi):
        """ 2-form initial magnetic field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], s, chi, phi, 'vector_3')
    
    def b2_1(self, s, chi, phi):
        """ 2-form initial magnetic field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], s, chi, phi, '2_form_1')
    
    def b2_2(self, s, chi, phi):
        """ 2-form initial magnetic field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], s, chi, phi, '2_form_2')
        
    def b2_3(self, s, chi, phi):
        """ 2-form initial magnetic field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], s, chi, phi, '2_form_3')
    
    def p0(self, s, chi, phi):
        """ 0-form initial pressure in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull(self.p, s, chi, phi, '0_form')
      
    def p3(self, s, chi, phi):
        """ 3-form initial pressure in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull(self.p, s, chi, phi, '3_form')
   
    def n0(self, s, chi, phi):
        """ 0-form initial number density in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull(self.n, s, chi, phi, '0_form')
     
    def n3(self, s, chi, phi):
        """ 3-form initial number density in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull(self.n, s, chi, phi, '3_form')