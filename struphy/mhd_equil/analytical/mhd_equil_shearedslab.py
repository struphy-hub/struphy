import numpy as np

class EquilibriumMhdShearedSlab:
    """
    TODO
    """
    
    def __init__(self, params):
        
        # geometric parameters
        self.a  = params['a']
        self.r0 = params['R0']
        
        # constant toroidal magnetic field (Bz), if q = 0, constant poloidal magnetic field (By)
        self.b0 = params['B0']
        
        # safety factor profile: q(x) = q0 + (q1 - q0)*(x/a)^2
        self.q0 = params['q0']
        self.q1 = params['q1']
        
        # bulk number density profile: n(x) = (1 - na)*(1 - (x/a)^n1)^n2 + na
        self.n1 = params['n1']
        self.n2 = params['n2']
        self.na = params['na']
        
        # plasma beta at x = 0
        self.beta = params['beta']
        
        # inverse aspect ratio
        self.eps = self.a/self.r0
        
        
    
    # ===============================================================
    #             profiles for a sheared slab geometry
    # ===============================================================
    def n(self, x):
        
        nout = (1 - self.na)*(1 - (x/self.a)**self.n1)**self.n2 + self.na
        
        return nout
    
    def q(self, x):
        
        qout = self.q0 + (self.q1 - self.q0)*(x/self.a)**2
        
        return qout
    
    def q_p(self, x):
        
        qout = 2*(self.q1 - self.q0)*x/self.a**2
        
        return qout
    
    def p(self, x):
        
        q = self.q(x)
        
        if np.all(q >= 100.) or np.all(q == 0.):
            pout = self.b0**2*self.beta/200 - 0*x
        else:
            pout = self.b0**2*self.beta/200*(1 + self.eps**2/q**2) + self.b0**2*self.eps**2*(1/self.q0**2 - 1/q**2)
               
        return pout


    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================
    
    # equilibrium magnetic field (x - component)
    def b_eq_x(self, x, y, z):
        
        bx = 0*x

        return bx

    # equilibrium magnetic field (y - component)
    def b_eq_y(self, x, y, z):
        
        q = self.q(x)
        
        if   np.all(q >= 100.):
            by = 0*x
        elif np.all(q == 0.):
            by = self.b0 - 0*x
        else:
            by = self.b0*self.eps/q

        return by

    # equilibrium magnetic field (z - component)
    def b_eq_z(self, x, y, z):
        
        q = self.q(x)
        
        if np.all(q == 0.):
            bz = 0*x
        else:
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
        
        q = self.q(x)
        
        if   np.all(q >= 100.):
            jz = 0*x
        elif np.all(q == 0.):
            jz = 0*x
        else:
            jz = -self.b0*self.eps*self.q_p(x)/q**2

        return jz

    
    # equilibrium bulk pressure
    def p_eq(self, x, y, z):
        
        p = self.p(x)

        return p
    
    
    # equilibrium bulk number density
    def n_eq(self, x, y, z):
        
        n = self.n(x)

        return n