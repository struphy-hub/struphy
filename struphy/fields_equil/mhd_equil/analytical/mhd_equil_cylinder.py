import numpy as np

from struphy.fields_equil.mhd_equil.mhd_equils import EquilibriumMHD

class EquilibriumMHDCylinder(EquilibriumMHD):
    """
    Straight tokamak (screw pinch) MHD equilibrium.
    
    The profiles in cylindrical coordinates (r, theta, z) are:
    
    B(r) = B0*( e_z + r/(q(r)*R0)*e_theta ), q(r) = q0 + ( q1 - q0 )*r^2/a^2,
    p(r) = B0^2*a^2*q0/( 2*R0^2*(q1 - q0) )*( 1/q(r)^2 - 1/q1^2) if q1 not equal q0, p(r) = beta*B0^2/2 else,
    n(r) = ( 1 - na )*( 1 - (r/a)^n1 )^n2 + na.
    
    Parameters
    ..........
        params: dictionary
            Parameters that characterize the MHD equilibrium.
                * a    : minor radius (radius of cylinder)
                * R0   : major radius (Lz = 2*pi*R0)
                * B0   : magnetic field in z-direction
                * q0   : safety factor at r=0
                * q1   : safety factor at r=a
                * n1   : shape factor for number density profile 
                * n2   : shape factor for number density profile 
                * na   : number density at r=a
                * beta : plasma beta in % for flat safety factor (ratio of kinetic pressure to magnetic pressure)
            
        DOMAIN: Domain obj (optional)
            From struphy.geometry.domain_3d.Domain.       
    """
    
    def __init__(self, params={'a' : 1., 'R0' : 5., 'B0' : 1., 'q0' : 1.05, 'q1' : 1.80, 'n1' : 0., 'n2' : 0., 'na' : 1.}, DOMAIN=None):
        
        # check that parameter dicitionary is complete
        assert 'a'  in params
        assert 'R0' in params
        
        assert 'B0' in params
        
        assert 'q0' in params
        assert 'q1' in params
        
        assert 'n1' in params
        assert 'n2' in params
        assert 'na' in params
        
        if params['q0'] == params['q1']:
            assert 'beta' in params
        
        super().__init__(params, DOMAIN)
        
        # inverse cylindrical coordinate transformation (x, y, z) --> (r, theta, phi)
        self.r     = lambda x, y, z : np.sqrt((x - self.params['R0'])**2 + y**2)
        self.theta = lambda x, y, z : np.arctan2(y, x - self.params['R0'])
        self.z     = lambda x, y, z : 1*z
     
    
    # ===============================================================
    #           profiles for a straight tokamak equilibrium
    # ===============================================================
    def n(self, r):
        """Equilibrium number density."""
        nout = (1 - self.params['na'])*(1 - (r/self.params['a'])**self.params['n1'])**self.params['n2'] + self.params['na']
        
        return nout
    
    def q(self, r):
        """Safety factor."""
        qout = self.params['q0'] + (self.params['q1'] - self.params['q0'])*(r/self.params['a'])**2
        
        return qout
    
    def q_p(self, r):
        """Derivative of safety factor."""
        qout = 2*(self.params['q1'] - self.params['q0'])*r/self.params['a']**2
        
        return qout
    
    def p(self, r):
        """Equilibrium pressure."""
        eps = self.params['a']/self.params['R0']
        
        if self.params['q0'] == self.params['q1']:
            pout = self.params['B0']**2*self.params['beta']/200 - 0*r
        else:
            pout = self.params['B0']**2*eps**2*self.params['q0']/(2*(self.params['q1'] - self.params['q0']))*(1/self.q(r)**2 - 1/self.params['q1']**2)
               
        return pout
    
    def plot_profiles(self, n_pts=501):
        """Plots radial profiles."""
        
        import matplotlib.pyplot as plt
        
        r = np.linspace(0., self.params['a'], n_pts)
        
        fig, ax = plt.subplots(1, 3)
        
        fig.set_figheight(3)
        fig.set_figwidth(12)
        
        ax[0].plot(r, self.q(r))
        ax[0].set_xlabel('r')
        ax[0].set_ylabel('q')
        
        ax[0].plot(r, np.ones(r.size), 'k--')
        
        ax[1].plot(r, self.p(r))
        ax[1].set_xlabel('r')
        ax[1].set_ylabel('p')
        
        ax[2].plot(r, self.n(r))
        ax[2].set_xlabel('r')
        ax[2].set_ylabel('n')
        
        plt.subplots_adjust(wspace=0.4)
        
        plt.show()


    
    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================
    
    # equilibrium magnetic field (x-component)
    def b_eq_x(self, x, y, z):
        """Equilibrium magnetic field (x-component)."""
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        
        q = self.q(r)
        
        # azimuthal component
        if np.all(q >= 100.):
            b_theta = 0*r
        else:
            b_theta = self.params['B0']*r/(self.params['R0']*q)

        # cartesian x-component
        bx = -b_theta*np.sin(theta)

        return bx

    # equilibrium magnetic field (y-component)
    def b_eq_y(self, x, y, z):
        """Equilibrium magnetic field (y-component)."""
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        
        q = self.q(r)
        
        # azimuthal component
        if np.all(q >= 100.):
            b_theta = 0*r
        else:
            b_theta = self.params['B0']*r/(self.params['R0']*q)

        # cartesian y-component
        by = b_theta*np.cos(theta)

        return by

    # equilibrium magnetic field (z-component)
    def b_eq_z(self, x, y, z):
        """Equilibrium magnetic field (z-component)."""
        bz = self.params['B0'] - 0*x

        return bz
    
    # equilibrium magnetic field (absolute value)
    def b_eq(self, x, y, z):
        """Equilibrium magnetic field (absolute value)."""
        bx = self.b_eq_x(x, y, z)
        by = self.b_eq_y(x, y, z)
        bz = self.b_eq_z(x, y, z)
        
        return np.sqrt(bx**2 + by**2 + bz**2)
    
    # equilibrium current (x-component, curl of equilibrium magnetic field)
    def j_eq_x(self, x, y, z):
        """Equilibrium current (x-component)."""
        jx = 0*x

        return jx

    # equilibrium current (y-component, curl of equilibrium magnetic field)
    def j_eq_y(self, x, y, z):
        """Equilibrium current (y-component)."""
        jy = 0*x

        return jy

    # equilibrium current (z-component, curl of equilibrium magnetic field)
    def j_eq_z(self, x, y, z):
        """Equilibrium current (z-component)."""
        r = self.r(x, y, z)
        
        q = self.q(r)
        q_p = self.q_p(r)
        
        if np.all(q >= 100.):
            jz = 0*x
        else:
            jz = self.params['B0']/(self.params['R0']*q**2)*(2*q - r*q_p)

        return jz

    # equilibrium pressure
    def p_eq(self, x, y, z):
        """Equilibrium pressure."""
        p = self.p(self.r(x, y, z))

        return p

    # equilibrium number density
    def n_eq(self, x, y, z):
        """Equilibrium number density."""
        n = self.n(self.r(x, y, z))
        
        return n
