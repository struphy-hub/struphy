import numpy as np
import scipy as sc


class equilibrium_mhd:
    
    def __init__(self, domain):
        self.domain = domain

    # ===============================================================
    #                       physical domain
    # ===============================================================

    # equilibrium bulk pressure
    def p_eq(self, x, y, z):
        
        p = 0*x

        return p

    # equilibrium magnetic field (x - component)
    def b_eq_x(self, x, y, z):
        
        bx = 0*x

        return bx

    # equilibrium magnetic field (y - component)
    def b_eq_y(self, x, y, z):
        
        by = 0*x

        return by

    # equilibrium magnetic field (z - component)
    def b_eq_z(self, x, y, z):

        b0 = 1.
        bz = b0 - 0*x

        return bz

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

        rho0 = 1.
        rho  = rho0 - 0*x

        return rho



    # ===============================================================
    #                       logical domain
    # ===============================================================

    # equilibrium bulk pressure (0-form on logical domain)
    def p0_eq(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull(self.p_eq, eta1, eta2, eta3, '0_form')
        else:
            if isinstance(eta1, float):
                return self.domain.pull(self.p_eq, eta1, eta2, 0., '0_form')
            else:
                return self.domain.pull(self.p_eq, eta1, eta2, np.array([0.]), '0_form')[:, :, 0]
            

    # equilibrium bulk pressure (3-form on logical domain)
    def p3_eq(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull(self.p_eq, eta1, eta2, eta3, '3_form')
        else:
            if isinstance(eta1, float):
                return self.domain.pull(self.p_eq, eta1, eta2, 0., '3_form')
            else:
                return self.domain.pull(self.p_eq, eta1, eta2, np.array([0.]), '3_form')[:, :, 0]
            
    
    # equilibrium bulk density (0-form on logical domain)
    def r0_eq(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull(self.r_eq, eta1, eta2, eta3, '0_form')
        else:
            if isinstance(eta1, float):
                return self.domain.pull(self.r_eq, eta1, eta2, 0., '0_form')
            else:
                return self.domain.pull(self.r_eq, eta1, eta2, np.array([0.]), '0_form')[:, :, 0]
            

    # equilibrium bulk density (3-form on logical domain)
    def r3_eq(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull(self.r_eq, eta1, eta2, eta3, '3_form')
        else:
            if isinstance(eta1, float):
                return self.domain.pull(self.r_eq, eta1, eta2, 0., '3_form')
            else:
                return self.domain.pull(self.r_eq, eta1, eta2, np.array([0.]), '3_form')[:, :, 0]
            

    # equilibrium magnetic field (2-form on logical domain, 1-component)
    def b2_eq_1(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, eta3, '2_form_1')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, 0., '2_form_1')
            else:
                return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, np.array([0.]), '2_form_1')[:, :, 0]
            
        
    # equilibrium magnetic field (2-form on logical domain, 2-component)
    def b2_eq_2(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, eta3, '2_form_2')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, 0., '2_form_2')
            else:
                return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, np.array([0.]), '2_form_2')[:, :, 0]
            

    # equilibrium magnetic field (2-form on logical domain, 3-component)
    def b2_eq_3(self, eta1, eta2, eta3=None):

        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, eta3, '2_form_3')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, 0., '2_form_3')
            else:
                return self.domain.pull([self.b_eq_x, self.b_eq_y, self.b_eq_z], eta1, eta2, np.array([0.]), '2_form_3')[:, :, 0]
            
            
    # equilibrium current (2-form on logical domain, 1-component)
    def j2_eq_1(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], eta1, eta2, eta3, '2_form_1')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], eta1, eta2, 0., '2_form_1')
            else:
                return self.domain.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], eta1, eta2, np.array([0.]), '2_form_1')[:, :, 0]
            
        
    # equilibrium current (2-form on logical domain, 2-component)
    def j2_eq_2(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], eta1, eta2, eta3, '2_form_2')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], eta1, eta2, 0., '2_form_2')
            else:
                return self.domain.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], eta1, eta2, np.array([0.]), '2_form_2')[:, :, 0]
            
        
    # equilibrium current (2-form on logical domain, 3-component)
    def j2_eq_3(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], eta1, eta2, eta3, '2_form_3')
        else:
            if isinstance(eta1, float):
                return self.domain.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], eta1, eta2, 0., '2_form_3')
            else:
                return self.domain.pull([self.j_eq_x, self.j_eq_y, self.j_eq_z], eta1, eta2, np.array([0.]), '2_form_3')[:, :, 0]