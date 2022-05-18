import numpy as np

from struphy.fields_equil.mhd_equil.mhd_equils import EquilibriumMHD

class EquilibriumMHDCylinder(EquilibriumMHD):
    """
    TODO
    """
    
    def __init__(self, params):
        
        super().__init__()
        
        # geometric parameters
        self.a  = params['a']
        self.r0 = params['R0']
        
        # uniform axial magnetic field (Bz)
        self.b0 = params['B0']
        
        # safety factor profile
        self.q0 = params['q0']
        self.q1 = params['q1']
        
        # bulk number density profile: n(r) = (1 - na)*(1 - (r/a)^n1)^n2 + na
        self.n1 = params['n1']
        self.n2 = params['n2']
        self.na = params['na']
        
        # plasma beta in case of pure axial magnetic field
        self.beta = params['beta']
        
        # inverse aspect ratio
        self.eps = self.a/self.r0
        
        # inverse cylindrical coordinate transformation (x, y, z) --> (r, theta, phi)
        self.r     = lambda x, y, z : np.sqrt((x - self.r0)**2 + y**2)
        self.theta = lambda x, y, z : np.arctan2(y, x - self.r0)
        self.z     = lambda x, y, z : 1*z
        
    
    
    
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
        
        if self.q0 == self.q1:
            pout = self.b0**2*self.beta/200 - 0*r
        else:
            pout = self.b0**2*self.eps**2*self.q0/(2*(self.q1 - self.q0))*(1/self.q(r)**2 - 1/self.q1**2)
               
        return pout

    
    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================
    
    # equilibrium magnetic field (x - component)
    def b_eq_x(self, x, y, z):
        
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        
        q = self.q(r)
        
        # azimuthal component
        if np.all(q >= 100.):
            b_theta = 0*r
        else:
            b_theta = self.b0*r/(self.r0*q)

        # cartesian x-component
        bx = -b_theta*np.sin(theta)

        return bx

    # equilibrium magnetic field (y - component)
    def b_eq_y(self, x, y, z):
        
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        
        q = self.q(r)
        
        # azimuthal component
        if np.all(q >= 100.):
            b_theta = 0*r
        else:
            b_theta = self.b0*r/(self.r0*q)

        # cartesian y-component
        by = b_theta*np.cos(theta)

        return by

    # equilibrium magnetic field (z - component)
    def b_eq_z(self, x, y, z):
        
        bz = self.b0 - 0*x

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

        r = self.r(x, y, z)
        
        q = self.q(r)
        q_p = self.q_p(r)
        
        if np.all(q >= 100.):
            jz = 0*x
        else:
            jz = self.b0/(self.r0*q**2)*(2*q - r*q_p)

        return jz
    

    # equilibrium bulk pressure
    def p_eq(self, x, y, z):
        
        p = self.p(self.r(x, y, z))

        return p

    
    # equilibrium bulk number density
    def n_eq(self, x, y, z):
        
        n = self.n(self.r(x, y, z))
        
        return n
