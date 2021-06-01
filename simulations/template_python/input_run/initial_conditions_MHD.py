import numpy as np
import scipy as sc


class initial_mhd:
    
    def __init__(self, domain):
        self.domain = domain

    # ===============================================================
    #                       physical domain
    # ===============================================================

    # initial bulk pressure
    def p_ini(self, x, y, z):

        p = 0*x

        return p
    
    # initial velocity (x - component)
    def u_ini_x(self, x, y, z):

        ux = 0*x

        return ux

    # initial velocity (y - component)
    def u_ini_y(self, x, y, z):

        uy = 0*x

        return uy

    # initial velocity (z - component)
    def u_ini_z(self, x, y, z):

        uz = 0*x

        return uz
    
    # initial magnetic field (x - component)
    def b_ini_x(self, x, y, z):
        
        bx = 0*x

        return bx

    # initial magnetic field (y - component)
    def b_ini_y(self, x, y, z):
        
        amp = 1e-4
        
        kx = 0.
        ky = 0.
        kz = 0.8
        
        by = amp*np.sin(kx*x + ky*y + kz*z)

        return by

    # initial magnetic field (z - component)
    def b_ini_z(self, x, y, z):

        bz = 0*x

        return bz

    # initial bulk density
    def r_ini(self, x, y, z):

        rho = 0*x

        return rho


    # ===============================================================
    #                       logical domain
    # ===============================================================

    # equilibrium bulk pressure (3-form on logical domain)
    def p3_ini(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull(self.p_ini, eta1, eta2, eta3, '3_form')
        else:
            if isinstance(eta1, float):
                return self.domain.pull(self.p_ini, eta1, eta2, 0., '3_form')
            else:
                return self.domain.pull(self.p_ini, eta1, eta2, np.array([0.]), '3_form')[:, :, 0]
  

    # equilibrium bulk density (3-form on logical domain)
    def r3_ini(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull(self.r_ini, eta1, eta2, eta3, '3_form')
        else:
            if isinstance(eta1, float):
                return self.domain.pull(self.r_ini, eta1, eta2, 0., '3_form')
            else:
                return self.domain.pull(self.r_ini, eta1, eta2, np.array([0.]), '3_form')[:, :, 0]
            

    # equilibrium current (2-form on logical domain, 1-component)
    def u2_ini_1(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, eta3, '2_form_1')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, 0., '2_form_1')
            else:
                return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, np.array([0.]), '2_form_1')[:, :, 0]
            
        
    # equilibrium current (2-form on logical domain, 2-component)
    def u2_ini_2(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, eta3, '2_form_2')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, 0., '2_form_2')
            else:
                return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, np.array([0.]), '2_form_2')[:, :, 0]
            
        
    # equilibrium current (2-form on logical domain, 3-component)
    def u2_ini_3(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, eta3, '2_form_3')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, 0., '2_form_3')
            else:
                return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, np.array([0.]), '2_form_3')[:, :, 0]
    
    
    # equilibrium magnetic field (2-form on logical domain, 1-component)
    def b2_ini_1(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.b_ini_x, self.b_ini_y, self.b_ini_z], eta1, eta2, eta3, '2_form_1')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.b_ini_x, self.b_ini_y, self.b_ini_z], eta1, eta2, 0., '2_form_1')
            else:
                return self.domain.pull([self.b_ini_x, self.b_ini_y, self.b_ini_z], eta1, eta2, np.array([0.]), '2_form_1')[:, :, 0]
            
        
    # equilibrium magnetic field (2-form on logical domain, 2-component)
    def b2_ini_2(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.b_ini_x, self.b_ini_y, self.b_ini_z], eta1, eta2, eta3, '2_form_2')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.b_ini_x, self.b_ini_y, self.b_ini_z], eta1, eta2, 0., '2_form_2')
            else:
                return self.domain.pull([self.b_ini_x, self.b_ini_y, self.b_ini_z], eta1, eta2, np.array([0.]), '2_form_2')[:, :, 0]
            

    # equilibrium magnetic field (2-form on logical domain, 3-component)
    def b2_ini_3(self, eta1, eta2, eta3=None):

        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.b_ini_x, self.b_ini_y, self.b_ini_z], eta1, eta2, eta3, '2_form_3')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.b_ini_x, self.b_ini_y, self.b_ini_z], eta1, eta2, 0., '2_form_3')
            else:
                return self.domain.pull([self.b_ini_x, self.b_ini_y, self.b_ini_z], eta1, eta2, np.array([0.]), '2_form_3')[:, :, 0]
            
    
    # equilibrium current (2-form on logical domain, 1-component)
    def u_ini_1(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, eta3, 'vector_1')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, 0., 'vector_1')
            else:
                return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, np.array([0.]), 'vector_1')[:, :, 0]
            
        
    # equilibrium current (2-form on logical domain, 2-component)
    def u_ini_2(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, eta3, 'vector_2')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, 0., 'vector_2')
            else:
                return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, np.array([0.]), 'vector_2')[:, :, 0]
            
        
    # equilibrium current (2-form on logical domain, 3-component)
    def u_ini_3(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, eta3, 'vector_3')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, 0., 'vector_3')
            else:
                return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, np.array([0.]), 'vector_3')[:, :, 0]