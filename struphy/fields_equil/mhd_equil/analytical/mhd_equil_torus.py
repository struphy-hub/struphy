import numpy as np

from struphy.fields_equil.mhd_equil.mhd_equils import EquilibriumMHD

class EquilibriumMHDTorus(EquilibriumMHD):
    """
    TODO
    """
    
    def __init__(self, params):
        
        super().__init__()
        
        # geometric parameters
        self.a  = params['a']
        self.r0 = params['R0']
        
        # on-axis toroidal magnetic field
        self.b0 = params['B0']
        
        # safety factor profile
        self.q0 = params['q0']
        self.q1 = params['q1']
        
        # bulk number density profile: n(r) = (1 - na)*(1 - (r/a)^n1)^n2 + na
        self.n1 = params['n1']
        self.n2 = params['n2']
        self.na = params['na']
        
        # pressure profile (plasma beta is on-axis)
        # p_kind = 0 : cylindrical presssure profile
        # p_kind = 1 : p(r) = B0^2*beta/2*(1 - p1*(r/a)^2 - p2*(r/a)^4)
        self.p_kind = params['p_kind']
        self.beta   = params['beta']
        self.p1     = params['p1']
        self.p2     = params['p2']
        
        # inverse toroidal coordinate transformation (x, y, z) --> (r, theta, phi)
        self.r     = lambda x, y, z : np.sqrt((np.sqrt(x**2 + z**2) - self.r0)**2 + y**2)
        self.theta = lambda x, y, z : np.arctan2(y, np.sqrt(x**2 + z**2) - self.r0)
        self.phi   = lambda x, y, z : np.arctan2(z, x)
        
        # inverse aspect ratio
        self.eps = self.a/self.r0
        
        # local inverse aspect ratio
        self.eps_loc = lambda r : r/self.r0
        
        # distance from axis of symmetry
        self.R = lambda r, theta : self.r0*(1 + self.eps_loc(r)*np.cos(theta))
        
        
        
    # ===============================================================
    #           profiles for a straight tokamak geometry
    # ===============================================================
    def n(self, r):
        
        nout = (1 - self.na)*(1 - (r/self.a)**self.n1)**self.n2 + self.na
        
        return nout
    
    def q(self, r):
        
        qout = self.q0 + (self.q1 - self.q0)*(r/self.a)**2
        
        return qout
    
    def q_p(self, r):
        
        qout = 2*(self.q1 - self.q0)*r/self.a**2
        
        return qout
    
    def p(self, r):
        
        if self.p_kind == 0:

            if self.q0 == self.q1:
                pout = self.b0**2*self.beta/200 - 0*r
            else:
                pout = self.b0**2*self.eps**2*self.q0/(2*(self.q1 - self.q0))*(1/self.q(r)**2 - 1/self.q1**2)
                
        else:
            
            pout = self.b0**2*self.beta/200*(1 - self.p1*r**2/self.a**2 - self.p2*r**4/self.a**4)
               
        return pout
        
    
    
    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================
    
    # equilibrium magnetic field (x - component)
    def b_eq_x(self, x, y, z):
        
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi   = self.phi(x, y, z)
        
        q = self.q(r)
        q_bar = q*np.sqrt(1 - self.eps_loc(r)**2)
        
        # poloidal component
        if np.all(q >= 100.):
            b_theta = 0*r
        else:
            b_theta = self.b0*r/(self.R(r, theta)*q_bar)
            
        # toroidal component
        b_phi = self.b0*self.r0/self.R(r, theta)

        # cartesian x-component
        bx = -b_theta*np.sin(theta)*np.cos(phi) - b_phi*np.sin(phi)

        return bx
    
    # equilibrium magnetic field (y - component)
    def b_eq_y(self, x, y, z):
        
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi   = self.phi(x, y, z)
        
        q = self.q(r)
        q_bar = q*np.sqrt(1 - self.eps_loc(r)**2)
        
        # poloidal component
        if np.all(q >= 100.):
            b_theta = 0*r
        else:
            b_theta = self.b0*r/(self.R(r, theta)*q_bar)

        # cartesian y-component
        by = b_theta*np.cos(theta)

        return by
    
    # equilibrium magnetic field (z - component)
    def b_eq_z(self, x, y, z):
        
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi   = self.phi(x, y, z)
        
        q = self.q(r)
        q_bar = q*np.sqrt(1 - self.eps_loc(r)**2)
        
        # poloidal component
        if np.all(q >= 100.):
            b_theta = 0*r
        else:
            b_theta = self.b0*r/(self.R(r, theta)*q_bar)
            
        # toroidal component
        b_phi = self.b0*self.r0/self.R(r, theta)

        # cartesian x-component
        bz = -b_theta*np.sin(theta)*np.sin(phi) + b_phi*np.cos(phi)

        return bz
    
    
    # equilibrium current (x - component, curl of equilibrium magnetic field)
    def j_eq_x(self, x, y, z):

        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi   = self.phi(x, y, z)
        
        q = self.q(r)
        q_p = self.q_p(r)
        
        q_bar = q*np.sqrt(1 - self.eps_loc(r)**2)
        q_bar_p = q_p*np.sqrt(1 - self.eps_loc(r)**2) - q*self.eps_loc(r)/(self.r0*np.sqrt(1 - self.eps_loc(r)**2))
        
        # toroidal component
        if np.all(q >= 100.):
            j_phi = 0*r
        else:
            j_phi = self.b0/(self.R(r, theta)*q_bar**2)*(2*q_bar - r*q_bar_p - r/self.R(r, theta)*q_bar*np.cos(theta))
        
        # cartesian x-component
        jx = -j_phi*np.sin(phi)

        return jx

    # equilibrium current (y - component, curl of equilibrium magnetic field)
    def j_eq_y(self, x, y, z):

        jy = 0*x

        return jy

    # equilibrium current (z - component, curl of equilibrium magnetic field)
    def j_eq_z(self, x, y, z):

        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi   = self.phi(x, y, z)
        
        q = self.q(r)
        q_p = self.q_p(r)
        
        q_bar = q*np.sqrt(1 - self.eps_loc(r)**2)
        q_bar_p = q_p*np.sqrt(1 - self.eps_loc(r)**2) - q*self.eps_loc(r)/(self.r0*np.sqrt(1 - self.eps_loc(r)**2))
        
        # toroidal component
        if np.all(q >= 100.):
            j_phi = 0*r
        else:
            j_phi = self.b0/(self.R(r, theta)*q_bar**2)*(2*q_bar - r*q_bar_p - r/self.R(r, theta)*q_bar*np.cos(theta))
        
        # cartesian x-component
        jz = j_phi*np.cos(phi)
        
        return jz

    
    # equilibrium bulk pressure
    def p_eq(self, x, y, z):
        
        p = self.p(self.r(x, y, z))

        return p
    
    
    # equilibrium bulk number density
    def n_eq(self, x, y, z):
        
        n = self.n(self.r(x, y, z))

        return n