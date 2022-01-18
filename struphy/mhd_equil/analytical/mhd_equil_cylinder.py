import numpy as np

from scipy.integrate import quad

import struphy.feec.spline_space as spl


class Equilibrium_mhd_cylinder:
    """
    TODO
    """
    
    def __init__(self, params):
        
         
        # minor and major radius
        self.a  = params['a']
        self.r0 = params['R0']
        
        # uniform axial magnetic field
        self.b0 = params['B0']
        
        # safety factor profile
        self.q0 = params['q0']
        self.q1 = params['q1']
        self.rl = params['rl']
        
        # density profile
        self.r1 = params['r1']
        self.r2 = params['r2']
        self.ra = params['ra']
        
        # inverse cylindrical coordinate transformation (x, y, z) --> (r, theta, phi)
        self.r     = lambda x, y, z : np.sqrt((x - self.r0)**2 + y**2)
        self.theta = lambda x, y, z : np.arctan2(y, x - self.r0)
        self.phi   = lambda x, y, z : z/(2*np.pi*self.r0)
        
        # calculate rp parameter in safety factor profile
        if self.q0 == self.q1:
            self.rp = np.inf
        else:
            self.rp = ((self.q1/self.q0)**self.rl - 1)**(-1/(2*self.rl))
            
        # calculate scaling of current profile such that q_min = q0 (for monotonic profiles j0 = 1)
        self.j0 = 1.
        
        # discrete pressure profile is based on q-profile
        self.SPLINES = spl.Spline_space_1d(128, 3, False, 10)
        
        self.SPLINES.set_projectors(10)
        
        self.p_coeff = self.SPLINES.projectors.pi_0(lambda eta1 : self.p_ana(eta1*self.a))
        self.p       = lambda r : self.SPLINES.evaluate_N(r/self.a, self.p_coeff)
        
    
    # ===============================================================
    #                   cylindrical coordinates
    # ===============================================================
    
    
    # -----------------------------------------------------------------------
    # axial current profile (determines b_theta)
    # -----------------------------------------------------------------------
    def j_phi(self, r):
        
        rn = r/self.a

        jout = 2/self.q0*1/((1 + (rn/self.rp)**(2*self.rl))**(1/self.rl + 1))

        return self.j0*self.b0/self.r0*jout
    
    # -----------------------------------------------------------------------
    # poloidal magnetic field component and its derivative
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
    # safety factor and its derivative
    # -----------------------------------------------------------------------
    def q(self, r):
        
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
            
        qout = r*self.b0/(self.r0*self.b_theta(r))
        
        if not isinstance(r, float):
            qout[r_zeros] = limit
            qout = qout.reshape(r_shape)
        
        return qout
    
    def q_p(self, r):
        
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
            
        qout = (self.b0/self.r0*(self.b_theta(r) - r*self.b_theta_p(r))/self.b_theta(r)**2)
        
        if not isinstance(r, float):
            qout[r_zeros] = limit
            qout = qout.reshape(r_shape)
        
        return qout
        
    
    # -----------------------------------------------------------------------
    # poloidal magnetic flux function
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
    # pressure profile and its derivative (integrated from MHD force balance with p(a) = 0)
    # -----------------------------------------------------------------------
    def p_ana(self, r):
            
        if isinstance(r, float):
            r_shape = 0.
            r = np.array([r])
        else:
            r_shape = r.shape
            r = r.flatten()

        pout = np.zeros(r.size, dtype=float)

        p_a = quad(self.p_p, 0., self.a)[0]

        for i in range(r.size):
            pout[i] = quad(self.p_p, 0., r[i])[0] - p_a

        if r_shape == 0:
            pout = pout[0]
        else:
            pout = pout.reshape(r_shape)
                
        return pout
    
    def p_p(self, r):
        
        pout = self.b0**2/(self.r0**2*self.q(r)**3)*r*(-2*self.q(r) + r*self.q_p(r))
       
        return pout
    
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
        
        pout = self.p(r)
        
        if pout.max() < 1e-12:
            pout = np.ones(pout.shape, dtype=float)*self.b0**2*self.beta/200

        return pout
    
    # equilibrium magnetic field (x - component)
    def b_eq_x(self, x, y, z):
        
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        
        # cylindrical component
        b_theta = self.b_theta(r)
        
        if b_theta.max() < 1e-10:
            b_theta = np.zeros(b_theta.shape, dtype=float)

        # cartesian components
        bx = -b_theta*np.sin(theta)

        return bx

    # equilibrium magnetic field (y - component)
    def b_eq_y(self, x, y, z):
        
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        
        # cylindrical component
        b_theta = self.b_theta(r)
        
        if b_theta.max() < 1e-10:
            b_theta = np.zeros(b_theta.shape, dtype=float)

        # cartesian components
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
        
        # toroidal components
        j_z = self.b0/(self.r0*self.q(r)**2)*(2*self.q(r) - r*self.q_p(r))
        
        if j_z.max() < 1e-10:
            j_z = np.zeros(j_z.shape, dtype=float)

        return jz

    # equilibrium bulk density
    def r_eq(self, x, y, z):

        r = self.r(x, y, z)

        return self.rho(r)
