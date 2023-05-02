import numpy as np


class InitialMaxwell:
    """
    Base class for Maxwell initial conditions.
    
    Parameters
    ----------
        params: dictionary
            Parameters that characterize possible profile functions.
            
        domain: struphy.geometry.domains
            Enables pull-backs if set.        
    """
    
    def __init__(self, params, domain=None):
        
        # set parameters
        self._params = params
        
        # set domain object
        if domain is not None:
            self._domain = domain
            
        # set vector-valued functions as list of callables
        self.e1 = [self.e1_1, self.e1_2, self.e1_3]
        self.b2 = [self.b2_1, self.b2_2, self.b2_3]
    
    @property
    def params(self):
        """ Parameters that characterize possible profile functions.
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

    def e1_1(self, *etas):
        """ 1-form initial electrid field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.e_x, self.e_y, self.e_z], *etas, kind='1_form')[0]
    
    def e1_2(self, *etas):
        """ 1-form initial electric field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.e_x, self.e_y, self.e_z], *etas, kind='1_form')[1]
        
    def e1_3(self, *etas):
        """ 1-form initial electric field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.e_x, self.e_y, self.e_z], *etas, kind='1_form')[2]
    
    def b2_1(self, *etas):
        """ 2-form initial magnetic field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], *etas, kind='2_form')[0]
    
    def b2_2(self, *etas):
        """ 2-form initial magnetic field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], *etas, kind='2_form')[1]
        
    def b2_3(self, *etas):
        """ 2-form initial magnetic field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], *etas, kind='2_form')[2]


class InitialMHD:
    """
    Base class for MHD initial conditions.
    
    Parameters
    ----------
        params: dictionary
            Parameters that characterize the MHD equilibrium.
            
        domain: struphy.geometry.domains
            Enables pull-backs if set.        
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

    def u1_1(self, *etas):
        """ 2-form initial magnetic field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], *etas, kind='1_form')[0]
    
    def u1_2(self, *etas):
        """ 2-form initial magnetic field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], *etas, kind='1_form')[1]
        
    def u1_3(self, *etas):
        """ 2-form initial magnetic field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], *etas, kind='1_form')[2]
    
    def u2_1(self, *etas):
        """ 2-form initial magnetic field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], *etas, kind='2_form')[0]
    
    def u2_2(self, *etas):
        """ 2-form initial magnetic field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], *etas, kind='2_form')[1]
        
    def u2_3(self, *etas):
        """ 2-form initial magnetic field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], *etas, kind='2_form')[2]
    
    def uv_1(self, *etas):
        """ 2-form initial magnetic field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], *etas, kind='vector')[0]
    
    def uv_2(self, *etas):
        """ 2-form initial magnetic field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], *etas, kind='vector')[1]
        
    def uv_3(self, *etas):
        """ 2-form initial magnetic field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.u_x, self.u_y, self.u_z], *etas, kind='vector')[2]
    
    def b2_1(self, *etas):
        """ 2-form initial magnetic field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], *etas, kind='2_form')[0]
    
    def b2_2(self, *etas):
        """ 2-form initial magnetic field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], *etas, kind='2_form')[1]
        
    def b2_3(self, *etas):
        """ 2-form initial magnetic field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], *etas, kind='2_form')[2]
    
    def p0(self, *etas):
        """ 0-form initial pressure in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.p], *etas, kind='0_form')
      
    def p3(self, *etas):
        """ 3-form initial pressure in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.p], *etas, kind='3_form')
   
    def n0(self, *etas):
        """ 0-form initial number density in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.n], *etas, kind='0_form')
     
    def n3(self, *etas):
        """ 3-form initial number density in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.n], *etas, kind='3_form')