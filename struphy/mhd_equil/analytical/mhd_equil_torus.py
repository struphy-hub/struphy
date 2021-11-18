import numpy as np

from scipy.integrate import quad

import struphy.feec.spline_space as spl


class equilibrium_mhd_torus:
    """
    Define an analytical ad hoc equilibrium for a torus with circular cross section.
    """
    
    def __init__(self, params):
        
         
        # minor and major radius
        self.a  = params['a']
        self.r0 = params['R0']
        
        # on-axis toroidal magnetic field
        self.b0 = params['B0']
        
        # safety factor profile
        self.q0 = params['q0']
        self.q1 = params['q1']
        self.rl = params['rl']
        
        # toroidal correction factor
        self.q_add = params['q_add']
        
        # density profile
        self.r1 = params['r1']
        self.r2 = params['r2']
        self.ra = params['ra']
        
        # pressure profile (plasma beta is on-axis)
        self.beta = params['beta']
        self.p1   = params['p1']
        self.p2   = params['p2']
        
        # inverse toroidal coordinate transformation (x, y, z) --> (r, theta, phi)
        self.r     = lambda x, y, z : np.sqrt((np.sqrt(x**2 + z**2) - self.r0)**2 + y**2)
        self.theta = lambda x, y, z : np.arctan2(y, np.sqrt(x**2 + z**2) - self.r0)
        self.phi   = lambda x, y, z : np.arctan2(z, x)
        
        # radial coordiante normalized to major
        self.eps = lambda r : r/self.r0
        
        # distance from axis of symmetry
        self.R = lambda r, theta : self.r0*(1 + self.eps(r)*np.cos(theta))
        
        # calculate rp parameter in safety factor profile
        if self.q0 == self.q1:
            self.rp = np.inf
        else:
            self.rp = ((self.q1/self.q0)**self.rl - 1)**(-1/(2*self.rl))
            
        # calculate scaling of current profile such that q_min = q0 (for monotonic profiles j0 = 1)
        self.j0 = 1
        
        
    # ===============================================================
    #                   toroidal coordinates
    # ===============================================================
    
    
    # -----------------------------------------------------------------------
    # axial current profile (determines b_theta)
    # -----------------------------------------------------------------------
    def j_phi(self, r):
        
        rn = r/self.a

        jout = 2/self.q0*1/((1 + (rn/self.rp)**(2*self.rl))**(1/self.rl + 1))

        return self.j0*self.b0/self.r0*jout
    
    
    # -----------------------------------------------------------------------
    # poloidal magnetic field component and its derivative in cylindrical limit
    # -----------------------------------------------------------------------
    def b_theta(self, r):
        
        # analytical limit for r --> 0
        limit = 0.
        
        if isinstance(r, float):
            if r == 0.:
                return limit  
        else:
            r_shape = r.shape
            r = r.flatten()
            r_zeros = np.where(r == 0.)[0]
            r[r_zeros] += 1e-10
        
        bout = (1/self.q0)*r*(1 + (r/(self.a*self.rp))**(2*self.rl))**(-1/self.rl)
            
        bout = self.j0*self.b0/self.r0*bout
        
        if not isinstance(r, float):
            bout[r_zeros] = limit
            bout = bout.reshape(r_shape)
            
        return bout
    
    def b_theta_p(self, r):
        
        # analytical limit for r --> 0
        limit = self.j0*self.b0/(self.r0*self.q0)
        
        if isinstance(r, float):
            if r == 0.:
                return limit
        else:
            r_shape = r.shape
            r = r.flatten()
            r_zeros = np.where(r == 0.)[0]
            r[r_zeros] += 1e-10
        
        temp = 1 + (r/(self.a*self.rp))**(2*self.rl)
        
        bout = 1/self.q0*temp**(-1/self.rl) - r/(self.q0*self.rl)*temp**(-1/self.rl - 1)*2*self.rl*(r/(self.a*self.rp))**(2*self.rl - 1)*1/(self.a*self.rp)
        
        bout = self.j0*self.b0/self.r0*bout
        
        if not isinstance(r, float):
            bout[r_zeros] = limit
            bout = bout.reshape(r_shape)
            
        return bout
    
    
    # -----------------------------------------------------------------------
    # safety factor and its derivative in cylindrical limit
    # -----------------------------------------------------------------------
    def q(self, r, cor=0):
        
        # analytical limit for r --> 0
        limit = self.b0/(self.r0*self.b_theta_p(0.))
        
        if isinstance(r, float):
            if r == 0.:
                return limit
        else:
            r_shape = r.shape
            r = r.flatten()
            r_zeros = np.where(r == 0.)[0]
            r[r_zeros] += 1e-10
            
        qout = r*self.b0/(self.r0*self.b_theta(r))*np.sqrt(1 - cor*self.eps(r)**2) 
        
        if not isinstance(r, float):
            qout[r_zeros] = limit
            qout = qout.reshape(r_shape)
        
        return qout
    
    def q_p(self, r, cor=0):
        
        # analytical limit for r --> 0
        limit = 0.
        
        if isinstance(r, float):
            if r == 0.:
                return limit
        else:
            r_shape = r.shape
            r = r.flatten()
            r_zeros = np.where(r == 0.)[0]
            r[r_zeros] += 1e-10
            
        qout = (self.b0/self.r0*(self.b_theta(r) - r*self.b_theta_p(r))/self.b_theta(r)**2)*np.sqrt(1 - cor*self.eps(r)**2) + self.q(r, cor)*cor*self.eps(r)/(self.r0*np.sqrt(1 - cor*self.eps(r)**2))
        
        if not isinstance(r, float):
            qout[r_zeros] = limit
            qout = qout.reshape(r_shape)
        
        return qout
    
    
    # -----------------------------------------------------------------------
    # poloidal magnetic flux function in cylindrical limit
    # -----------------------------------------------------------------------
    def psi0(self, r):
        
        if isinstance(r, float):
            r_shape = 0.
            r = np.array([r])
        else:
            r_shape = r.shape
            r = r.flatten()
        
        psiout = np.zeros(r.size, dtype=float)
        
        psi_a = -quad(self.b_theta, 0., self.a)[0]
        
        for i in range(r.size):
            psiout[i] = -quad(self.b_theta, 0., r[i])[0] - psi_a
            
        if r_shape == 0:
            psiout = psiout[0]
        else:
            psiout = psiout.reshape(r_shape)
            
        return psiout
    
    
    # -----------------------------------------------------------------------
    # pressure profile
    # -----------------------------------------------------------------------
    def p(self, r):
        return self.b0**2*self.beta/200*(1 - self.p1*r**2/self.a**2 - self.p2*r**4/self.a**4)
    
    
    # -----------------------------------------------------------------------
    # density profile
    # -----------------------------------------------------------------------
    def rho(self, r):
        return (1 - self.ra)*(1 - (r/self.a)**self.r1)**self.r2 + self.ra
    
    
    # ===============================================================
    #                       physical domain
    # ===============================================================
    
    
    # equilibrium bulk pressure
    def p_eq(self, x, y, z):
        
        r = self.r(x, y, z)

        return self.p(r)
    
    # equilibrium magnetic field (x - component)
    def b_eq_x(self, x, y, z):
        
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi   = self.phi(x, y, z)
        
        # toroidal components
        b_theta = r*self.b0/(self.R(r, theta)*self.q(r, self.q_add))
        
        if b_theta.max() < 1e-10:
            b_theta = np.zeros(b_theta.shape, dtype=float)
            
        b_phi = self.b0*self.r0/self.R(r, theta)

        # cartesian component
        bx = -b_theta*np.sin(theta)*np.cos(phi) - b_phi*np.sin(phi)

        return bx

    # equilibrium magnetic field (y - component)
    def b_eq_y(self, x, y, z):
        
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi   = self.phi(x, y, z)
        
        # toroidal components
        b_theta = r*self.b0/(self.R(r, theta)*self.q(r, self.q_add))
        
        if b_theta.max() < 1e-10:
            b_theta = np.zeros(b_theta.shape, dtype=float)
            
        b_phi = self.b0*self.r0/self.R(r, theta)

        # cartesian component
        by = b_theta*np.cos(theta)

        return by

    # equilibrium magnetic field (z - component)
    def b_eq_z(self, x, y, z):
        
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi   = self.phi(x, y, z)
        
        # toroidal components
        b_theta = r*self.b0/(self.R(r, theta)*self.q(r, self.q_add))
        
        if b_theta.max() < 1e-10:
            b_theta = np.zeros(b_theta.shape, dtype=float)
            
        b_phi = self.b0*self.r0/self.R(r, theta)
        
        # cartesian component
        bz = -b_theta*np.sin(theta)*np.sin(phi) + b_phi*np.cos(phi)

        return bz

    # equilibrium current (x - component, curl of equilibrium magnetic field)
    def j_eq_x(self, x, y, z):
        
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi   = self.phi(x, y, z)

        # toroidal components
        j_phi = self.b0/(self.R(r, theta)*self.q(r, self.q_add)**2)*(2*self.q(r, self.q_add) - r*self.q_p(r, self.q_add) - r/self.R(r, theta)*self.q(r, self.q_add)*np.cos(theta))
        
        if j_phi.max() < 1e-10:
            j_phi = np.zeros(j_phi.shape, dtype=float)
        
        # cartesian components
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

        # toroidal components
        j_phi = self.b0/(self.R(r, theta)*self.q(r, self.q_add)**2)*(2*self.q(r, self.q_add) - r*self.q_p(r, self.q_add) - r/self.R(r, theta)*self.q(r, self.q_add)*np.cos(theta))
        
        if j_phi.max() < 1e-10:
            j_phi = np.zeros(j_phi.shape, dtype=float)
        
        # cartesian components
        jz = j_phi*np.cos(phi)

        return jz

    # equilibrium bulk density
    def r_eq(self, x, y, z):

        r = self.r(x, y, z)

        return self.rho(r)
