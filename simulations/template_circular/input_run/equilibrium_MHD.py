import numpy as np
from scipy.integrate import quad
from scipy.integrate import fixed_quad

import scipy.special as sp

import hylife.utilitis_FEEC.spline_space as spl

import hylife.utilitis_FEEC.basics.mass_matrices_1d  as mass
import hylife.utilitis_FEEC.basics.inner_products_1d as inner



class equilibrium_mhd:
    """
    Specifies an analytical ideal MHD equilibrium specifying the axial (toroidal) current profile.
    Either cylindrical or circular toroidal geometry can be used.
    """
    
    def __init__(self, domain, a=1., r0=10., b0=1., q0=1.4, q1=4.5, rl=1.5, q_add=0, r1=1.5, r2=0.9, rho_a=0.22, p_kind=0, beta=0.2, p1=0.95, p2=0.05):
        
         
        # geometric parameters
        self.domain = domain  # domain object defining the mapping
        self.a      = a       # minor radius in m
        self.r0     = r0      # major radius in m
        
        # numerical parameters for discrete profiles
        self.num_params = [128, 3, False, 10, 10]
        
        # inverse cylindrical coordinate transformation
        if self.domain.kind_map == 1 or self.domain.kind_map == 11:
            self.r         = lambda x, y, z : np.sqrt((x - self.r0)**2 + y**2)
            self.phi       = lambda x, y, z : np.arctan2(y, x - self.r0)
            self.tor       = lambda x, y, z : 0*x
            self.curvature = 0
            self.eps       = lambda r : r/self.r0
            
            assert q_add == 0
        
        # inverse torus coordinate transformation
        elif self.domain.kind_map == 2 or self.domain.kind_map == 14:
            self.r         = lambda x, y, z : np.sqrt((np.sqrt(x**2 + z**2) - self.r0)**2 + y**2)
            self.phi       = lambda x, y, z : np.arctan2(y, np.sqrt(x**2 + z**2) - self.r0)
            self.tor       = lambda x, y, z : np.arctan2(z, x)
            self.curvature = 1
            self.eps       = lambda r : r/self.r0
        
        self.R = lambda r, phi : self.r0*(1 + self.curvature*self.eps(r)*np.cos(phi))
        
        # on-axis toroidal magnetic field in T
        self.b0 = b0
        
        # parameters for safety factor profile: q(r) = q0*(1 + (r/(a*rp))^(2*rl))^(1/rl), rp = ((q1/q0)^rl - 1)^(-1/(2*rl))
        self.q0 = q0
        self.q1 = q1
        self.rl = rl
        
        # calculate rp
        if self.q0 == self.q1:
            self.rp = np.inf
        else:
            self.rp = ((self.q1/self.q0)**self.rl - 1)**(-1/(2*self.rl))
            
        # add toroidal correction to q-profile: q(r) = q(r)*sqrt(1 - (r/R0)^2)
        self.q_add = q_add
        
        # parameters for bumps in current profile (not fully supported yet) 
        self.bmp0 = 0.
        self.bmp1 = 0.
        self.bmp2 = 0.
        self.cg0  = 0.2
        self.cg1  = 0.2
        self.cg2  = 0.2
        self.wg0  = 0.3
        self.wg1  = 0.3
        self.wg2  = 0.3
        
        # add order-eps Shafranov shift in case of toroidal geometry (not fully supported yet)
        self.shafranov = 0
        
        # parameters for bulk density profile: rho(r) = (1 - rho_a)*(1 - (r/a)^r1)^r2 + rho_a
        self.r1    = r1
        self.r2    = r2
        self.rho_a = rho_a
        
        # parameters for pressure profile:
        # p_kind = 0 : dp/dr(r) = B0^2/(R0^2*q(r)^3)*r*(-2*q(r) + r*dq/dr(r)) --> integrated with p(a) = 0
        # p_kind = 1 :     p(r) = B0^2*beta/200*(1 - p1*(r/a)^2 - p2*(r/a)^4), beta = on-axis plasma beta in %
        self.gamma  = 5/3
        self.p_kind = p_kind
        self.beta   = beta
        self.p1     = p1
        self.p2     = p2
        
        # calculate scaling of current profile such that q_min = q0
        self.j0 = 1.
        
        r_grid     = np.linspace(0., self.a, 2001)
        self.r_min = r_grid[np.argmin(self.q(r_grid))]
        self.q_min = self.q(self.r_min)
        self.j0    = self.q_min/self.q0
        
        # calculate Shafranov shift with boundary conditions delta_p(0) = 0 and delta(a) = 0
        self.spl_space_r = spl.spline_space_1d(self.num_params[0], self.num_params[1], self.num_params[2], self.num_params[3], ['f', 'd'])
        
        self.spl_space_r.set_projectors(self.num_params[4])
        
        jac = lambda eta1 : self.a*np.ones(eta1.shape, dtype=float)
        
        # LHS (stiffness matrix)
        LHS = mass.get_M_gen(self.spl_space_r, 1, 1, lambda eta1 : self.a*eta1*self.b_phi(eta1)**2, jac)
        LHS = self.spl_space_r.E0.dot(LHS.dot(self.spl_space_r.E0.T))

        # RHS
        RHS = inner.inner_prod_V0(self.spl_space_r, lambda eta1 : self.a*eta1/self.r0*self.b_phi(eta1)**2, jac)
        RHS = self.spl_space_r.E0.dot(RHS)
        
        # solve system to get FEM coefficients and create callables
        self.delta_coeff = np.linalg.solve(LHS.toarray(), RHS)*self.curvature*self.shafranov
        
        self.delta   = lambda r : self.spl_space_r.evaluate_N(r/self.a, self.delta_coeff, 0)
        self.delta_p = lambda r : self.spl_space_r.evaluate_N(r/self.a, self.delta_coeff, 2)/self.a
        
        # radial magnetic field due to Shafranov shift
        self.Br = lambda r, phi : self.b0*self.delta(r)/(self.R(r, phi)*self.q(r))*np.sin(phi)
        
        # discrete pressure profile if based on q-profile
        if self.p_kind == 0:
            
            self.p_coeff = self.spl_space_r.projectors.pi_0(lambda eta1 : self.p_ana(eta1*self.a))
            self.p = lambda r : self.spl_space_r.evaluate_N(r/self.a, self.p_coeff, 0)
            
        else:
            
            self.p = self.p_ana
        
    # -----------------------------------------------------------------------
    # bulk density profile
    # -----------------------------------------------------------------------
    def rho(self, r):
        return (1 - self.rho_a)*(1 - (r/self.a)**self.r1)**self.r2 + self.rho_a
    
    # -----------------------------------------------------------------------
    # toroidal current profile in cylindrical limit
    # -----------------------------------------------------------------------
    def j_tor(self, r):
        
        rn = r/self.a

        jout = 2/self.q0*1/((1 + (rn/self.rp)**(2*self.rl))**(1/self.rl + 1))
        
        # add contributions from bumps
        if self.bmp0 != 0.:
            jout += self.bmp0*np.exp(-(rn**2 - self.cg0)**2/self.wg0**2)
        if self.bmp1 != 0.:
            jout += self.bmp1*np.exp(-(rn**2 - self.cg1)**2/self.wg1**2)
        if self.bmp2 != 0.:
            jout += self.bmp2*np.exp(-(rn**2 - self.cg2)**2/self.wg2**2)

        return self.j0*self.b0/self.r0*jout
    
    # -----------------------------------------------------------------------
    # poloidal magnetic field component and its derivative in cylindrical limit
    # -----------------------------------------------------------------------
    def b_phi(self, r):
        
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
        
        # add contributions from bumps
        if self.bmp0 != 0.:
            bout += 1/(4*r)*self.bmp0*np.sqrt(np.pi)*self.a**2*self.wg0*sp.erf(self.cg0/self.wg0)*(1 - sp.erf((self.cg0 - r**2/self.a**2)/self.wg0)/sp.erf(self.cg0/self.wg0))
        if self.bmp1 != 0.:
            bout += 1/(4*r)*self.bmp1*np.sqrt(np.pi)*self.a**2*self.wg1*sp.erf(self.cg1/self.wg1)*(1 - sp.erf((self.cg1 - r**2/self.a**2)/self.wg1)/sp.erf(self.cg1/self.wg1))
        if self.bmp2 != 0.:
            bout += 1/(4*r)*self.bmp2*np.sqrt(np.pi)*self.a**2*self.wg2*sp.erf(self.cg2/self.wg2)*(1 - sp.erf((self.cg2 - r**2/self.a**2)/self.wg2)/sp.erf(self.cg2/self.wg2))
            
        bout = self.j0*self.b0/self.r0*bout
        
        if not isinstance(r, float):
            bout[r_zeros] = limit
            bout = bout.reshape(r_shape)
            
        return bout
    
    def b_phi_p(self, r):
        
        # analytical limit for r --> 0
        limit = self.j0*self.b0/(self.r0*self.q0)
        
        # add contributions from bumps
        if self.bmp0 != 0.:
            limit += self.j0*self.b0/(2*self.r0)*self.bmp0*np.exp(-self.cg0**2/self.wg0**2)
        if self.bmp1 != 0.:
            limit += self.j0*self.b0/(2*self.r0)*self.bmp1*np.exp(-self.cg1**2/self.wg1**2)
        if self.bmp2 != 0.:
            limit += self.j0*self.b0/(2*self.r0)*self.bmp2*np.exp(-self.cg2**2/self.wg2**2)
        
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
        
        # add contributions from bumps
        if self.bmp0 != 0.:
            bout += self.bmp0*np.sqrt(np.pi)/4*self.a**2*self.wg0*sp.erf(self.cg0/self.wg0)*(-1/r**2 + 1/sp.erf(self.cg0/self.wg0)*(1/r**2*sp.erf((self.cg0 - r**2/self.a**2)/self.wg0) + 4/(np.sqrt(np.pi)*self.a**2*self.wg0)*np.exp(-(self.cg0 - r**2/self.a**2)**2/self.wg0**2)))
        if self.bmp1 != 0.:
            bout += self.bmp1*np.sqrt(np.pi)/4*self.a**2*self.wg1*sp.erf(self.cg1/self.wg1)*(-1/r**2 + 1/sp.erf(self.cg1/self.wg1)*(1/r**2*sp.erf((self.cg1 - r**2/self.a**2)/self.wg1) + 4/(np.sqrt(np.pi)*self.a**2*self.wg1)*np.exp(-(self.cg1 - r**2/self.a**2)**2/self.wg1**2)))
        if self.bmp2 != 0.:
            bout += self.bmp2*np.sqrt(np.pi)/4*self.a**2*self.wg2*sp.erf(self.cg2/self.wg2)*(-1/r**2 + 1/sp.erf(self.cg2/self.wg2)*(1/r**2*sp.erf((self.cg2 - r**2/self.a**2)/self.wg2) + 4/(np.sqrt(np.pi)*self.a**2*self.wg2)*np.exp(-(self.cg2 - r**2/self.a**2)**2/self.wg2**2)))
            
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
        limit = self.b0/(self.r0*self.b_phi_p(0.))
        
        if isinstance(r, float):
            if r == 0.:
                return limit
        else:
            r_shape = r.shape
            r = r.flatten()
            r_zeros = np.where(r == 0.)[0]
            r[r_zeros] += 1e-10
            
        qout = r*self.b0/(self.r0*self.b_phi(r))*np.sqrt(1 - cor*self.eps(r)**2) 
        
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
            
        qout = (self.b0/self.r0*(self.b_phi(r) - r*self.b_phi_p(r))/self.b_phi(r)**2)*np.sqrt(1 - cor*self.eps(r)**2) + self.q(r, cor)*cor*self.eps(r)/(self.r0*np.sqrt(1 - cor*self.eps(r)**2))
        
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
        
        psi_a = -quad(self.b_phi, 0., self.a)[0]
        
        for i in range(r.size):
            psiout[i] = -quad(self.b_phi, 0., r[i])[0] - psi_a
            
        if r_shape == 0:
            psiout = psiout[0]
        else:
            psiout = psiout.reshape(r_shape)
            
        return psiout
    
    # -----------------------------------------------------------------------
    # order-eps poloidal magnetic flux correction
    # -----------------------------------------------------------------------
    def psi1(self, r, phi):
        return self.delta(r)*self.b_phi(r)*np.cos(phi)
    
    # -----------------------------------------------------------------------
    # pressure profile and its derivative (cylinder --> integrated pressure with p(a) = 0, torus --> free pressure profile)
    # -----------------------------------------------------------------------
    def p_ana(self, r):
        
        if self.p_kind == 0:
            
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
                
        else:
            
            pout = self.b0**2*self.beta/200*(1 - self.p1*r**2/self.a**2 - self.p2*r**4/self.a**4)
            
        return pout
    
    def p_p(self, r):
        
        if self.p_kind == 0:
            pout = self.b0**2/(self.r0**2*self.q(r)**3)*r*(-2*self.q(r) + r*self.q_p(r))
            
        else:
            pout = self.b0**2*self.beta/200*(-2*self.p1*r/self.a**2 - 4*self.p2*r**3/self.a**4)
        
        return pout
     
    
    # ===============================================================
    #                       physical domain
    # ===============================================================

    # equilibrium bulk pressure
    def p_eq(self, x, y, z):
        
        r = self.r(x, y, z)
        
        pout = self.p(r)
        
        if self.curvature == 0 and pout.max() < 1e-12:
            pout = np.ones(pout.shape, dtype=float)*self.b0**2*self.beta/200

        return pout
    
    # equilibrium magnetic field (x - component)
    def b_eq_x(self, x, y, z):
        
        r   = self.r(x, y, z)
        phi = self.phi(x, y, z)
        tor = self.tor(x, y, z)
        
        # toroidal components
        b_r   = self.Br(r, phi)
        
        b_phi = self.r0/self.R(r, phi)*(r*self.b0/(self.r0*self.q(r, self.q_add)) + (self.delta_p(r)*self.b_phi(r) + self.delta(r)*self.b_phi_p(r))*np.cos(phi))
        
        if b_phi.max() < 1e-10:
            b_phi = np.zeros(b_phi.shape, dtype=float)
        
        b_tor = self.b0*self.r0/self.R(r, phi)

        # cartesian components
        bx = b_r*np.cos(phi)*np.cos(tor) - b_phi*np.sin(phi)*np.cos(tor) - b_tor*np.sin(tor)

        return bx

    # equilibrium magnetic field (y - component)
    def b_eq_y(self, x, y, z):
        
        r   = self.r(x, y, z)
        phi = self.phi(x, y, z)
        tor = self.tor(x, y, z)
        
        # toroidal components
        b_r   = self.Br(r, phi)
        
        b_phi = self.r0/self.R(r, phi)*(r*self.b0/(self.r0*self.q(r, self.q_add)) + (self.delta_p(r)*self.b_phi(r) + self.delta(r)*self.b_phi_p(r))*np.cos(phi))
        
        if b_phi.max() < 1e-10:
            b_phi = np.zeros(b_phi.shape, dtype=float)
        
        b_tor = self.b0*self.r0/self.R(r, phi)

        # cartesian components
        by = b_r*np.sin(phi) + b_phi*np.cos(phi)

        return by

    # equilibrium magnetic field (z - component)
    def b_eq_z(self, x, y, z):
        
        r   = self.r(x, y, z)
        phi = self.phi(x, y, z)
        tor = self.tor(x, y, z)
        
        # toroidal components
        b_r   = self.Br(r, phi)
        
        b_phi = self.r0/self.R(r, phi)*(r*self.b0/(self.r0*self.q(r, self.q_add)) + (self.delta_p(r)*self.b_phi(r) + self.delta(r)*self.b_phi_p(r))*np.cos(phi))
        
        if b_phi.max() < 1e-10:
            b_phi = np.zeros(b_phi.shape, dtype=float)
        
        b_tor = self.b0*self.r0/self.R(r, phi)
        
        # cartesian components
        bz = b_r*np.cos(phi)*np.sin(tor) - b_phi*np.sin(phi)*np.sin(tor) + b_tor*np.cos(tor)

        return bz
    
    # equilibrium magnetic field (absolute value)
    def b_eq(self, x, y, z):
        
        b_abs = np.sqrt(self.b_eq_x(x, y, z)**2 + self.b_eq_y(x, y, z)**2 + self.b_eq_z(x, y, z)**2)
        
        return b_abs

    # equilibrium current (x - component, curl of equilibrium magnetic field)
    def j_eq_x(self, x, y, z):

        r   = self.r(x, y, z)
        phi = self.phi(x, y, z)
        tor = self.tor(x, y, z)
        
        # toroidal components
        j_tor = self.b0/(self.R(r, phi)*self.q(r, self.q_add)**2)*(2*self.q(r, self.q_add) - r*self.q_p(r, self.q_add) - self.curvature*r/self.R(r, phi)*self.q(r, self.q_add)*np.cos(phi))
        
        if j_tor.max() < 1e-10:
            j_tor = np.zeros(j_tor.shape, dtype=float)
        
        # cartesian components
        jx = -j_tor*np.sin(tor)

        return jx

    # equilibrium current (y - component, curl of equilibrium magnetic field)
    def j_eq_y(self, x, y, z):

        jy = 0*x

        return jy

    # equilibrium current (z - component, curl of equilibrium magnetic field)
    def j_eq_z(self, x, y, z):

        r   = self.r(x, y, z)
        phi = self.phi(x, y, z)
        tor = self.tor(x, y, z)
        
        # toroidal components
        j_tor = self.b0/(self.R(r, phi)*self.q(r, self.q_add)**2)*(2*self.q(r, self.q_add) - r*self.q_p(r, self.q_add) - self.curvature*r/self.R(r, phi)*self.q(r, self.q_add)*np.cos(phi))
        
        if j_tor.max() < 1e-10:
            j_tor = np.zeros(j_tor.shape, dtype=float)
        
        # cartesian components
        jz = j_tor*np.cos(tor)

        return jz

    # equilibrium bulk density
    def r_eq(self, x, y, z):

        r = self.r(x, y, z)

        return self.rho(r)



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
        