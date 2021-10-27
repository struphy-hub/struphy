import numpy as np
import scipy as sc


class equilibrium_mhd:
    
    def __init__(self, domain, b0x=0., b0y=0., b0z=1., rho0=1., beta=100.):
        
        # geometric parameters
        self.domain = domain
        
        # magnetic field
        self.b0x = b0x
        self.b0y = b0y
        self.b0z = b0z
        
        # density
        self.rho0 = rho0
        
        # pressure
        self.gamma = 5/3
        self.beta  = beta

    # ===============================================================
    #                       physical domain
    # ===============================================================

    # equilibrium bulk pressure
    def p_eq(self, x, y, z):
        
        p = np.sqrt(self.b0x**2 + self.b0y**2 + self.b0z**2)*self.beta/200 - 0*x

        return p

    # equilibrium magnetic field (x - component)
    def b_eq_x(self, x, y, z):
        
        bx = self.b0x - 0*x

        return bx

    # equilibrium magnetic field (y - component)
    def b_eq_y(self, x, y, z):
        
        by = self.b0y - 0*x

        return by

    # equilibrium magnetic field (z - component)
    def b_eq_z(self, x, y, z):

        bz = self.b0z - 0*x

        return bz
    
    # equilibrium magnetic field (absolute value)
    def b_eq(self, x, y, z):
        
        b_abs = np.sqrt(self.b_eq_x(x, y, z)**2 + self.b_eq_y(x, y, z)**2 + self.b_eq_z(x, y, z)**2)
        
        return b_abs

    # equilibrium current (x - component, curl of equilibrium magnetic field)
    def j_eq_x(self, x, y, z):

        jx = 0*x

        return jx

    # equilibrium current (y - component, curl of equilibrium magnetic field)
    def j_eq_y(self, x, y, z):

        jy = 0*x

        return jy

    # equilibrium current (z - component, curl of equilibrium magnetic field)
    def j_eq_z(self, x, y, z):
        
        jz = 0*x

        return jz

    # equilibrium bulk density
    def r_eq(self, x, y, z):
        
        rho = self.rho0 - 0*x

        return rho

    
    # ===============================================================
    #                  pull-backs to logical domain
    # ===============================================================

    # equilibrium bulk pressure (0-form on logical domain)
    def p0_eq(self, eta1, eta2, eta3):
        return self.domain.pull(self.p_eq, eta1, eta2, eta3, '0_form')
           
    
    # equilibrium bulk pressure (3-form on logical domain)
    def p3_eq(self, eta1, eta2, eta3):
        return self.domain.pull(self.p_eq, eta1, eta2, eta3, '3_form')        
    
    
    # equilibrium bulk density (0-form on logical domain)
    def r0_eq(self, eta1, eta2, eta3):
        return self.domain.pull(self.r_eq, eta1, eta2, eta3, '0_form')
            

    # equilibrium bulk density (3-form on logical domain)
    def r3_eq(self, eta1, eta2, eta3):
        return self.domain.pull(self.r_eq, eta1, eta2, eta3, '3_form')
    
    
    # equilibrium magnetic field (0-form absolute value on logical domain)
    def b0_eq(self, eta1, eta2, eta3):
        return self.domain.pull(self.b_eq, eta1, eta2, eta3, '0_form')
      

    # equilibrium magnetic field (1-form on logical domain, 1-component)
    def b1_eq_1(self, eta1, eta2, eta3):
        return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, eta3, '1_form_1')
        
 
    # equilibrium magnetic field (1-form on logical domain, 2-component)
    def b1_eq_2(self, eta1, eta2, eta3):
        return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, eta3, '1_form_2')
        
    
    # equilibrium magnetic field (1-form on logical domain, 3-component)
    def b1_eq_3(self, eta1, eta2, eta3):
        return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, eta3, '1_form_3')
    
    
    # equilibrium magnetic field (2-form on logical domain, 1-component)
    def b2_eq_1(self, eta1, eta2, eta3):
        return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, eta3, '2_form_1')
 
    
    # equilibrium magnetic field (2-form on logical domain, 2-component)
    def b2_eq_2(self, eta1, eta2, eta3):
        return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, eta3, '2_form_2')
        
    
    # equilibrium magnetic field (2-form on logical domain, 3-component)
    def b2_eq_3(self, eta1, eta2, eta3):
        return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, eta3, '2_form_3')
    
    
    # equilibrium magnetic field (vector on logical domain, 1-component)
    def bv_eq_1(self, eta1, eta2, eta3):
        return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, eta3, 'vector_1')
    
    
    # equilibrium magnetic field (vector on logical domain, 2-component)
    def bv_eq_2(self, eta1, eta2, eta3):
        return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, eta3, 'vector_2')
        
    
    # equilibrium magnetic field (vector on logical domain, 3-component)
    def bv_eq_3(self, eta1, eta2, eta3):
        return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, eta3, 'vector_3')
        
            
    # equilibrium current (2-form on logical domain, 1-component)
    def j2_eq_1(self, eta1, eta2, eta3):
        return self.domain.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], eta1, eta2, eta3, '2_form_1')
        
        
    # equilibrium current (2-form on logical domain, 2-component)
    def j2_eq_2(self, eta1, eta2, eta3):
        return self.domain.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], eta1, eta2, eta3, '2_form_2')
        
        
    # equilibrium current (2-form on logical domain, 3-component)
    def j2_eq_3(self, eta1, eta2, eta3):
        return self.domain.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], eta1, eta2, eta3, '2_form_3')
        