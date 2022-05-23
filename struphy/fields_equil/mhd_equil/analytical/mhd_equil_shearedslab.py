import numpy as np

from struphy.fields_equil.mhd_equil.mhd_equils import EquilibriumMHD

class EquilibriumMHDShearedSlab(EquilibriumMHD):
    """
    Sheared slab MHD equilibrium.
    
    The profiles in cartesian coordinates (x, y, z) are:
    
    B(x) = B0*( e_z + a/(q(x)*R0)*e_y ), q(x) = q0 + ( q1 - q0 )*x^2/a^2,
    p(x) = beta*B0^2/2*( 1 + a^2/(q(x)^2*R0^2) ) + B0^2*a^2/R0^2*( 1/q0^2 - 1/q(x)^2 ),
    n(x) = ( 1 - na )*( 1 - (x/a)^n1 )^n2 + na.
    
    Parameters
    ..........
        params: dictionary
            Parameters that characterize the MHD equilibrium.
                * a    : minor radius (Lx = a, Ly = 2*pi*a)
                * R0   : major radius (Lz = 2*pi*R0)
                * B0   : magnetic field in z-direction
                * q0   : safety factor at x=0
                * q1   : safety factor at x=a
                * n1   : shape factor for number density profile 
                * n2   : shape factor for number density profile 
                * na   : number density at x=a
                * beta : plasma beta in % at x=0 (ratio of kinetic pressure to magnetic pressure)
            
        DOMAIN: Domain obj (optional)
            From struphy.geometry.domain_3d.Domain.       
    """
    
    def __init__(self, params={'a' : 1., 'R0' : 3., 'B0' : 1., 'q0' : 1.05, 'q1' : 1.80, 'n1' : 0., 'n2' : 0., 'na' : 1., 'beta' : 10.}, DOMAIN=None):
        
        # check that parameter dicitionary is complete
        assert 'a'  in params
        assert 'R0' in params
        
        assert 'B0' in params
        
        assert 'q0' in params
        assert 'q1' in params
        
        assert 'n1' in params
        assert 'n2' in params
        assert 'na' in params
        
        assert 'beta' in params
        
        super().__init__(params, DOMAIN)
        
    
    # ===============================================================
    #             profiles for a sheared slab geometry
    # ===============================================================
    def n(self, x):
        """Equilibrium number density."""
        nout = (1 - self.params['na'])*(1 - (x/self.params['a'])**self.params['n1'])**self.params['n2'] + self.params['na']
        
        return nout
    
    def q(self, x):
        """Safety factor."""
        qout = self.params['q0'] + (self.params['q1'] - self.params['q0'])*(x/self.params['a'])**2
        
        return qout
    
    def q_p(self, x):
        """Derivative of safety factor."""
        qout = 2*(self.params['q1'] - self.params['q0'])*x/self.params['a']**2
        
        return qout
    
    def p(self, x):
        """Equilibrium pressure."""
        q = self.q(x)
        
        eps = self.params['a']/self.params['R0']
        
        if np.all(q >= 100.) or np.all(q == 0.):
            pout = self.params['B0']**2*self.params['beta']/200 - 0*x
        else:
            pout = self.params['B0']**2*self.params['beta']/200*(1 + eps**2/q**2) + self.params['B0']**2*eps**2*(1/self.params['q0']**2 - 1/q**2)
               
        return pout
    
    def plot_profiles(self, n_pts=501):
        """Plots radial profiles."""
        
        import matplotlib.pyplot as plt
        
        x = np.linspace(0., self.params['a'], n_pts)
        
        fig, ax = plt.subplots(1, 3)
        
        fig.set_figheight(3)
        fig.set_figwidth(12)
        
        ax[0].plot(x, self.q(x))
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('q')
        
        ax[1].plot(x, self.p(x))
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('p')
        
        ax[2].plot(x, self.n(x))
        ax[2].set_xlabel('x')
        ax[2].set_ylabel('n')
        
        plt.subplots_adjust(wspace=0.4)
        
        plt.show()


    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================
    
    # equilibrium magnetic field (x-component)
    def b_eq_x(self, x, y, z):
        """Equilibrium magnetic field (x-component)."""
        bx = 0*x

        return bx

    # equilibrium magnetic field (y-component)
    def b_eq_y(self, x, y, z):
        """Equilibrium magnetic field (y-component)."""
        q = self.q(x)
        
        eps = self.params['a']/self.params['R0']
        
        if   np.all(q >= 100.):
            by = 0*x
        elif np.all(q == 0.):
            by = self.params['B0'] - 0*x
        else:
            by = self.params['B0']*eps/q

        return by

    # equilibrium magnetic field (z-component)
    def b_eq_z(self, x, y, z):
        """Equilibrium magnetic field (z-component)."""
        q = self.q(x)
        
        if np.all(q == 0.):
            bz = 0*x
        else:
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
        q = self.q(x)
        
        eps = self.params['a']/self.params['R0']
        
        if   np.all(q >= 100.):
            jz = 0*x
        elif np.all(q == 0.):
            jz = 0*x
        else:
            jz = -self.params['B0']*eps*self.q_p(x)/q**2

        return jz

    # equilibrium pressure
    def p_eq(self, x, y, z):
        """Equilibrium pressure."""
        p = self.p(x)

        return p
    
    # equilibrium number density
    def n_eq(self, x, y, z):
        """Equilibrium number density."""
        n = self.n(x)

        return n