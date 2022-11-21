from abc import ABCMeta, abstractmethod
import numpy as np


class EquilibriumMHD( metaclass=ABCMeta ):
    """
    Base class for MHD equilibria in Struphy.
    
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
    
    @abstractmethod
    def b_x(self, x, y, z):
        """ Equilibrium magnetic field (x-component) in physical space.
        """
        return    
    
    @abstractmethod
    def b_y(self, x, y, z):
        """ Equilibrium magnetic field (y-component) in physical space.
        """
        return
 
    @abstractmethod
    def b_z(self, x, y, z):
        """ Equilibrium magnetic field (z-component) in physical space.
        """
        return 
    
    @abstractmethod
    def b(self, x, y, z):
        """ Equilibrium magnetic field (absolute value) in physical space.
        """
        return
    
    @abstractmethod
    def j_x(self, x, y, z):
        """ Equilibrium current (x-component, curl of equilibrium magnetic field) in physical space.
        """
        return
 
    @abstractmethod
    def j_y(self, x, y, z):
        """ Equilibrium current (y-component, curl of equilibrium magnetic field) in physical space.
        """
        return

    @abstractmethod
    def j_z(self, x, y, z):
        """ Equilibrium current (z-component, curl of equilibrium magnetic field) in physical space.
        """
        return
    
    @abstractmethod
    def p(self, x, y, z):
        """ Equilibrium pressure in physical space.
        """
        return
    
    @abstractmethod
    def n(self, x, y, z):
        """ Equilibrium number density in physical space.
        """
        return
    
    def b(self, x, y, z):
        """ Equilibrium magnetic field (absolute value).
        """
        bx = self.b_x(x, y, z)
        by = self.b_y(x, y, z)
        bz = self.b_z(x, y, z)
        
        return np.sqrt(bx**2 + by**2 + bz**2)

    def norm_b_x(self, x, y, z):
        """ Normalized equilibrium magnetic field (x-component) in physical space.
        """
        return self.b_x(x, y, z) / self.b(x, y, z)   
    
    def norm_b_y(self, x, y, z):
        """ Normalized equilibrium magnetic field (y-component) in physical space.
        """
        return self.b_y(x, y, z) / self.b(x, y, z)
 
    def norm_b_z(self, x, y, z):
        """ Normalized equilibrium magnetic field (z-component) in physical space.
        """
        return self.b_z(x, y, z) / self.b(x, y, z)
    
    def b0(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 0-form absolute value of equilibrium magnetic field in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b], s, chi, phi, '0_form', flat_eval, squeeze_output)
      
    def b1_1(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 1-form equilibrium magnetic field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], s, chi, phi, '1_form_1', flat_eval, squeeze_output)
        
    def b1_2(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 1-form equilibrium magnetic field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], s, chi, phi, '1_form_2', flat_eval, squeeze_output)

    def b1_3(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 1-form equilibrium magnetic field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], s, chi, phi, '1_form_3', flat_eval, squeeze_output)
    
    def b2_1(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 2-form equilibrium magnetic field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], s, chi, phi, '2_form_1', flat_eval, squeeze_output)
    
    def b2_2(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 2-form equilibrium magnetic field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], s, chi, phi, '2_form_2', flat_eval, squeeze_output)
        
    def b2_3(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 2-form equilibrium magnetic field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], s, chi, phi, '2_form_3', flat_eval, squeeze_output)

    def norm_b1_1(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 1-form equilibrium magnetic field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.norm_b_x, self.norm_b_y, self.norm_b_z], s, chi, phi, '1_form_1', flat_eval, squeeze_output)
        
    def norm_b1_2(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 1-form equilibrium magnetic field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.norm_b_x, self.norm_b_y, self.norm_b_z], s, chi, phi, '1_form_2', flat_eval, squeeze_output)

    def norm_b1_3(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 1-form equilibrium magnetic field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.norm_b_x, self.norm_b_y, self.norm_b_z], s, chi, phi, '1_form_3', flat_eval, squeeze_output)
    
    def norm_b2_1(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 2-form equilibrium magnetic field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.norm_b_x, self.norm_b_y, self.norm_b_z], s, chi, phi, '2_form_1', flat_eval, squeeze_output)
    
    def norm_b2_2(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 2-form equilibrium magnetic field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.norm_b_x, self.norm_b_y, self.norm_b_z], s, chi, phi, '2_form_2', flat_eval, squeeze_output)
        
    def norm_b2_3(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 2-form equilibrium magnetic field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.norm_b_x, self.norm_b_y, self.norm_b_z], s, chi, phi, '2_form_3', flat_eval, squeeze_output)

    def bv_1(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ Vector equilibrium magnetic field (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], s, chi, phi, 'vector_1', flat_eval, squeeze_output)

    def bv_2(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ Vector equilibrium magnetic field (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], s, chi, phi, 'vector_2', flat_eval, squeeze_output)
 
    def bv_3(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ Vector equilibrium magnetic field (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.b_x, self.b_y, self.b_z], s, chi, phi, 'vector_3', flat_eval, squeeze_output)
 
    def j2_1(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 2-form equilibrium current (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.j_x, self.j_y, self.j_z], s, chi, phi, '2_form_1', flat_eval, squeeze_output)
   
    def j2_2(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 2-form equilibrium current (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.j_x, self.j_y, self.j_z], s, chi, phi, '2_form_2', flat_eval, squeeze_output)
   
    def j2_3(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 2-form equilibrium current (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.j_x, self.j_y, self.j_z], s, chi, phi, '2_form_3', flat_eval, squeeze_output)
      
    def jv_1(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ Vector equilibrium current (1-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.j_x, self.j_y, self.j_z], s, chi, phi, 'vector_1', flat_eval, squeeze_output)
  
    def jv_2(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ Vector equilibrium current (2-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.j_x, self.j_y, self.j_z], s, chi, phi, 'vector_2', flat_eval, squeeze_output)
       
    def jv_3(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ Vector equilibrium current (3-component) in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.j_x, self.j_y, self.j_z], s, chi, phi, 'vector_3', flat_eval, squeeze_output)
    
    def p0(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 0-form equilibrium pressure in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.p], s, chi, phi, '0_form', flat_eval, squeeze_output)
      
    def p3(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 3-form equilibrium pressure in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.p], s, chi, phi, '3_form', flat_eval, squeeze_output)
   
    def n0(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 0-form equilibrium number density in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.n], s, chi, phi, '0_form', flat_eval, squeeze_output)
     
    def n3(self, s, chi, phi, flat_eval=False, squeeze_output=True):
        """ 3-form equilibrium number density in logical space.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull([self.n], s, chi, phi, '3_form', flat_eval, squeeze_output)