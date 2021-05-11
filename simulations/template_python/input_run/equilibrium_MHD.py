import numpy as np
import scipy as sc


class equilibrium_mhd:
    
    def __init__(self, domain):
        self.domain = domain

    # ===============================================================
    #                       physical domain
    # ===============================================================
    
    # q -profile
    def q(self, r, der=0):
        
        # on-axis q-factor
        q0 = 1.1
        
        # edge q-factor
        qe = 1.85

        q  = q0 + (qe - q0)*r**2
        dq = 2*(qe - q0)*r
        
        if der == 0:
            return q
        else:
            return dq
        
    # derivative of bulk pressure
    def dp(self, r, phi):
        
        b0  = 1.
        r0  = self.domain.params_map[0]/(2*np.pi)
        
        return b0**2*r/(r0**2*self.q(r)**3)*(-2*self.q(r) + r*self.q(r, 1))

    # equilibrium bulk pressure
    def p_eq(self, x, y, z):
        
        r   = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        a   = 1.

        p = np.zeros(x.shape, dtype=float)
        
    
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):

                integrand = lambda r : self.dp(r, phi[i, j, 0])

                p[i, j, :] = quad(integrand, 0., r[i, j, 0])[0] - quad(integrand, 0., a)[0]
    
        #p = 0*x

        return p

    # equilibrium magnetic field (x - component)
    def b_eq_x(self, x, y, z):
        
        r   = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        b0  = 1.
        r0  = self.domain.params_map[0]/(2*np.pi)
        
        b_phi = r*b0/(r0*self.q(r))

        bx = -b_phi*np.sin(phi)
        
        #bx = 0*x

        return bx

    # equilibrium magnetic field (y - component)
    def b_eq_y(self, x, y, z):
        
        r   = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        b0  = 1.
        r0  = self.domain.params_map[0]/(2*np.pi)
        
        b_phi = r*b0/(r0*self.q(r))

        by = b_phi*np.cos(phi)
        
        #by = 0*x

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

        r   = np.sqrt(x**2 + y**2)
        b0  = 1.
        r0  = self.domain.params_map[0]/(2*np.pi)
        
        jz  = b0/r0*(2*self.q(r) - r*self.q(r, 1))/self.q(r)**2
        
        #jz = 0*x

        return jz

    # equilibrium bulk density
    def rho_eq(self, x, y, z):

        r   = np.sqrt(x**2 + y**2)
        eps = 0.
        
        #rho = 1. - eps*r**2
        rho = (1 - eps*r**4)**3

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
    def rho0_eq(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull(self.rho_eq, eta1, eta2, eta3, '0_form')
        else:
            if isinstance(eta1, float):
                return self.domain.pull(self.rho_eq, eta1, eta2, 0., '0_form')
            else:
                return self.domain.pull(self.rho_eq, eta1, eta2, np.array([0.]), '0_form')[:, :, 0]
            

    # equilibrium bulk density (3-form on logical domain)
    def rho3_eq(self, eta1, eta2, eta3=None):
        
        if isinstance(eta3, float) or isinstance(eta3, np.ndarray):
            return self.domain.pull(self.rho_eq, eta1, eta2, eta3, '3_form')
        else:
            if isinstance(eta1, float):
                return self.domain.pull(self.rho_eq, eta1, eta2, 0., '3_form')
            else:
                return self.domain.pull(self.rho_eq, eta1, eta2, np.array([0.]), '3_form')[:, :, 0]
            

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