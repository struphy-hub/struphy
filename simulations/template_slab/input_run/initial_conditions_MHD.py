import numpy as np
import scipy as sc


class initial_mhd:
    
    def __init__(self, domain, kx=0., ky=0., kz=0.8, bix=0., biy=1e-4, biz=0.):
        
        # geometry
        self.domain = domain
        
        # parameters
        self.kx = kx
        self.ky = ky
        self.kz = kz
        
        self.bix = bix
        self.biy = biy
        self.biz = biz

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
        
        bx = np.cos(self.kx*x + self.ky*y)*np.sin(self.kz*z) + np.sin(self.kx*x + self.ky*y)*np.cos(self.kz*z)
        bx = self.bix*bx

        return bx

    # initial magnetic field (y - component)
    def b_ini_y(self, x, y, z):
        
        by = np.cos(self.kx*x + self.ky*y)*np.sin(self.kz*z) + np.sin(self.kx*x + self.ky*y)*np.cos(self.kz*z)
        by = self.biy*by

        return by

    # initial magnetic field (z - component)
    def b_ini_z(self, x, y, z):

        bz = np.cos(self.kx*x + self.ky*y)*np.sin(self.kz*z) + np.sin(self.kx*x + self.ky*y)*np.cos(self.kz*z)
        bz = self.biz*bz

        return bz
    # initial bulk density
    def r_ini(self, x, y, z):

        rho = 0*x

        return rho


    # ===============================================================
    #                   pull-back to logical domain
    # ===============================================================

    # equilibrium bulk pressure (3-form on logical domain)
    def p3_ini(self, eta1, eta2, eta3):
        return self.domain.pull(self.p_ini, eta1, eta2, eta3, '3_form')
  

    # equilibrium bulk density (3-form on logical domain)
    def r3_ini(self, eta1, eta2, eta3):
        return self.domain.pull(self.r_ini, eta1, eta2, eta3, '3_form')
            

    # equilibrium current (2-form on logical domain, 1-component)
    def u2_ini_1(self, eta1, eta2, eta3):
        return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, eta3, '2_form_1')
            
        
    # equilibrium current (2-form on logical domain, 2-component)
    def u2_ini_2(self, eta1, eta2, eta3):
        return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, eta3, '2_form_2')
            
        
    # equilibrium current (2-form on logical domain, 3-component)
    def u2_ini_3(self, eta1, eta2, eta3):
        return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, eta3, '2_form_3')
    
    
    # equilibrium magnetic field (2-form on logical domain, 1-component)
    def b2_ini_1(self, eta1, eta2, eta3):
        return self.domain.pull([self.b_ini_x, self.b_ini_y, self.b_ini_z], eta1, eta2, eta3, '2_form_1')
            
        
    # equilibrium magnetic field (2-form on logical domain, 2-component)
    def b2_ini_2(self, eta1, eta2, eta3):
        return self.domain.pull([self.b_ini_x, self.b_ini_y, self.b_ini_z], eta1, eta2, eta3, '2_form_2')
            

    # equilibrium magnetic field (2-form on logical domain, 3-component)
    def b2_ini_3(self, eta1, eta2, eta3):
        return self.domain.pull([self.b_ini_x, self.b_ini_y, self.b_ini_z], eta1, eta2, eta3, '2_form_3')
            
    
    # equilibrium current (vector on logical domain, 1-component)
    def uv_ini_1(self, eta1, eta2, eta3):
        return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, eta3, 'vector_1')
            
        
    # equilibrium current (vector on logical domain, 2-component)
    def uv_ini_2(self, eta1, eta2, eta3):
        return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, eta3, 'vector_2')
            
        
    # equilibrium current (vector on logical domain, 3-component)
    def uv_ini_3(self, eta1, eta2, eta3):
        return self.domain.pull([self.u_ini_x, self.u_ini_y, self.u_ini_z], eta1, eta2, eta3, 'vector_3')