import numpy as np

from struphy.fields_background.mhd_equil.base import EquilibriumMHD

# =======================================================================================
class HomogenSlab(EquilibriumMHD):
    """
    Homogeneous MHD equilibrium in slab geometry.
    
    .. math::

        \mathbf B_0 = \\begin{pmatrix} B_{0,x} \\ B_{0,y} \\ B_{0,z} \end{pmatrix} = const.\,,
        \qquad p_0 = \\beta \\frac{|\mathbf B_0|^2}{2}\,,\qquad n_0 = 1\,.
    
    Parameters
    ----------
        params: dict
            Parameters that characterize the MHD equilibrium.

                * B0x  : magnetic field in x-direction
                * B0y  : magnetic field in y-direction
                * B0z  : magnetic field in z-direction
                * beta : plasma beta in % (ratio of kinetic pressure to magnetic pressure)
            
        domain: struphy.geometry.domain_3d.Domain
            All things mapping. Enables pull-backs if set.             
    """
    
    def __init__(self, params=None, domain=None):
        
        # set default parameters
        if params is None:
            params_default = {'B0x'  : 0., 
                              'B0y'  : 0., 
                              'B0z'  : 1., 
                              'beta' : 100.}
            
            super().__init__(params_default, domain)
        
        # or check if given parameter dicitionary is complete
        else:
            assert 'B0x' in params
            assert 'B0y' in params
            assert 'B0z' in params

            assert 'beta' in params

            super().__init__(params, domain)
    
    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================
    
    # equilibrium magnetic field (x-component)
    def b_x(self, x, y, z):
        """ Equilibrium magnetic field (x-component).
        """
        bx = self.params['B0x'] - 0*x

        return bx

    # equilibrium magnetic field (y-component)
    def b_y(self, x, y, z):
        """ Equilibrium magnetic field (y-component).
        """
        by = self.params['B0y'] - 0*x

        return by

    # equilibrium magnetic field (z-component)
    def b_z(self, x, y, z):
        """ Equilibrium magnetic field (z-component).
        """
        bz = self.params['B0z'] - 0*x

        return bz
    
    # equilibrium current (x-component, curl of equilibrium magnetic field)
    def j_x(self, x, y, z):
        """ Equilibrium current (x-component).
        """
        jx = 0*x

        return jx

    # equilibrium current (y-component, curl of equilibrium magnetic field)
    def j_y(self, x, y, z):
        """ Equilibrium current (y-component).
        """
        jy = 0*x

        return jy

    # equilibrium current (z-component, curl of equilibrium magnetic field)
    def j_z(self, x, y, z):
        """ Equilibrium current (z-component).
        """
        jz = 0*x

        return jz
       
    # equilibrium pressure
    def p(self, x, y, z):
        """ Equilibrium pressure.
        """
        pp = self.params['beta']/200*(self.params['B0x']**2 + self.params['B0y']**2 + self.params['B0z']**2) - 0*x

        return pp
 
    # equilibrium number density
    def n(self, x, y, z):
        """ Equilibrium number density.
        """
        nn = 1 - 0*x

        return nn

    
# =======================================================================================    
class ShearedSlab(EquilibriumMHD):
    """
    Sheared slab MHD equilibrium in Cartesian space (x, y, z). Profiles depend on x solely. 
    
    .. math::
    
        \mathbf B_0(x) &= B_{0,z} \\begin{pmatrix} 0 \\ a/(q(x)R_0) )  \\ 1 \end{pmatrix}\,,\qquad q(x) = q_0 + ( q_1 - q_0 )\\frac{x^2}{a^2}\,,

        p_0(x) &= \\beta\\frac{B_{0,z}^2}{2} \left( 1 + \\frac{a^2}{q(x)^2 R_0^2} \\right) + B_{0,z}^2 \\frac{a^2}{R_0^2} \left( \\frac{1}{q_0^2} - \\frac{1}{q(x)^2} \\right)\,,

        n_0(x) &= n_a + ( 1 - n_a ) ( 1 - (x/a)^{n_1} )^{n_2} \,.
    
    Parameters
    ----------
        params: dict
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
            
        domain: struphy.geometry.domain_3d.Domain
            All things mapping. Enables pull-backs if set.             
    """
    
    def __init__(self, params=None, domain=None):
        
        # set default parameters
        if params is None:
            params_default = {'a'    : 1., 
                              'R0'   : 3., 
                              'B0'   : 1., 
                              'q0'   : 1.05, 
                              'q1'   : 1.80, 
                              'n1'   : 0., 
                              'n2'   : 0., 
                              'na'   : 1., 
                              'beta' : 10.}
            
            super().__init__(params_default, domain)
        
        # or check if given parameter dicitionary is complete
        else:
            assert 'a'  in params
            assert 'R0' in params

            assert 'B0' in params

            assert 'q0' in params
            assert 'q1' in params

            assert 'n1' in params
            assert 'n2' in params
            assert 'na' in params

            assert 'beta' in params

            super().__init__(params, domain)
        
    
    # ===============================================================
    #             profiles for a sheared slab geometry
    # ===============================================================
    def nx(self, x):
        """ Radial (x) number density profile.
        """
        nout = (1 - self.params['na'])*(1 - (x/self.params['a'])**self.params['n1'])**self.params['n2'] + self.params['na']
        
        return nout
    
    def q(self, x):
        """ Radial (x) safety factor profile.
        """
        qout = self.params['q0'] + (self.params['q1'] - self.params['q0'])*(x/self.params['a'])**2
        
        return qout
    
    def q_p(self, x):
        """ Radial (x) derivative of safety factor profile.
        """
        qout = 2*(self.params['q1'] - self.params['q0'])*x/self.params['a']**2
        
        return qout
    
    def px(self, x):
        """ Radial pressure profile.
        """
        q = self.q(x)
        
        eps = self.params['a']/self.params['R0']
        
        if np.all(q >= 100.) or np.all(q == 0.):
            pout = self.params['B0']**2*self.params['beta']/200 - 0*x
        else:
            pout = self.params['B0']**2*self.params['beta']/200*(1 + eps**2/q**2) + self.params['B0']**2*eps**2*(1/self.params['q0']**2 - 1/q**2)
               
        return pout
    
    def plot_profiles(self, n_pts=501):
        """ Plots radial profiles.
        """
        
        import matplotlib.pyplot as plt
        
        x = np.linspace(0., self.params['a'], n_pts)
        
        fig, ax = plt.subplots(1, 3)
        
        fig.set_figheight(3)
        fig.set_figwidth(12)
        
        ax[0].plot(x, self.q(x))
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('q')
        
        ax[1].plot(x, self.px(x))
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('p')
        
        ax[2].plot(x, self.nx(x))
        ax[2].set_xlabel('x')
        ax[2].set_ylabel('n')
        
        plt.subplots_adjust(wspace=0.4)
        
        plt.show()


    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================
    
    # equilibrium magnetic field (x-component)
    def b_x(self, x, y, z):
        """ Equilibrium magnetic field (x-component).
        """
        bx = 0*x

        return bx

    # equilibrium magnetic field (y-component)
    def b_y(self, x, y, z):
        """ Equilibrium magnetic field (y-component).
        """
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
    def b_z(self, x, y, z):
        """ Equilibrium magnetic field (z-component).
        """
        q = self.q(x)
        
        if np.all(q == 0.):
            bz = 0*x
        else:
            bz = self.params['B0'] - 0*x

        return bz
     
    # equilibrium current (x-component, curl of equilibrium magnetic field)
    def j_x(self, x, y, z):
        """ Equilibrium current (x-component).
        """
        jx = 0*x

        return jx

    # equilibrium current (y-component, curl of equilibrium magnetic field)
    def j_y(self, x, y, z):
        """ Equilibrium current (y-component).
        """
        jy = 0*x

        return jy

    # equilibrium current (z-component, curl of equilibrium magnetic field)
    def j_z(self, x, y, z):
        """ Equilibrium current (z-component).
        """
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
    def p(self, x, y, z):
        """ Equilibrium pressure.
        """
        pp = self.px(x)

        return pp
    
    # equilibrium number density
    def n(self, x, y, z):
        """ Equilibrium number density.
        """
        nn = self.nx(x)

        return nn
    
    
# =======================================================================================        
class ScrewPinch(EquilibriumMHD):
    """
    Straight tokamak (screw pinch) MHD equilibrium.
    
    The profiles in cylindrical coordinates :math:`(r, \\theta, z)` are:
    
    .. math::
    
        \mathbf B_0(r) &= B_{0,z}*\left( \mathbf e_z + \\frac{r}{q(r) R_0} e_\\theta \\right)\,,\qquad q(r) = q_0 + ( q_1 - q_0 )\\frac{r^2}{a^2}\,,

        p_0(r) &= \\frac{B_{0,z}^2 a^2 q_0}{ 2 R_0^2(q_1 - q_0) } \left( \\frac{1}{q(r)^2} - \\frac{1}{q_1^2} \\right) \quad \\textnormal{if $q_1$ not equal $q_0$}\,,\quad p_0(r) = \\beta \\frac{B_{0,z}^2}{2} \quad \\textnormal{else}\,,

        n_0(r) &= n_a + ( 1 - n_a )( 1 - (r/a)^{n_1} )^{n_2}\,.
    
    Parameters
    ----------
        params: dict
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
            
        domain: struphy.geometry.domain_3d.Domain
            All things mapping. Enables pull-backs if set.  
    """
    
    def __init__(self, params=None, domain=None):
        
        # set default parameters
        if params is None:
            params_default = {'a'  : 1., 
                              'R0' : 5., 
                              'B0' : 1., 
                              'q0' : 1.05, 
                              'q1' : 1.80, 
                              'n1' : 0., 
                              'n2' : 0., 
                              'na' : 1.}
            
            super().__init__(params_default, domain)
        
        # or check if given parameter dicitionary is complete
        else:
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

            super().__init__(params, domain)
        
        # inverse cylindrical coordinate transformation (x, y, z) --> (r, theta, phi)
        self.r     = lambda x, y, z : np.sqrt((x - self.params['R0'])**2 + y**2)
        self.theta = lambda x, y, z : np.arctan2(y, x - self.params['R0'])
        self.z     = lambda x, y, z : 1*z
     
    
    # ===============================================================
    #           profiles for a straight tokamak equilibrium
    # ===============================================================
    def nr(self, r):
        """ Radial number density profile.
        """
        nout = (1 - self.params['na'])*(1 - (r/self.params['a'])**self.params['n1'])**self.params['n2'] + self.params['na']
        
        return nout
    
    def q(self, r):
        """ Radial safety factor profile.
        """
        qout = self.params['q0'] + (self.params['q1'] - self.params['q0'])*(r/self.params['a'])**2
        
        return qout
    
    def q_p(self, r):
        """ Radial derivative of safety factor profile.
        """
        qout = 2*(self.params['q1'] - self.params['q0'])*r/self.params['a']**2
        
        return qout
    
    def pr(self, r):
        """ Radial pressure profile.
        """
        eps = self.params['a']/self.params['R0']
        
        if self.params['q0'] == self.params['q1']:
            pout = self.params['B0']**2*self.params['beta']/200 - 0*r
        else:
            pout = self.params['B0']**2*eps**2*self.params['q0']/(2*(self.params['q1'] - self.params['q0']))*(1/self.q(r)**2 - 1/self.params['q1']**2)
               
        return pout
    
    def plot_profiles(self, n_pts=501):
        """ Plots radial profiles.
        """
        
        import matplotlib.pyplot as plt
        
        r = np.linspace(0., self.params['a'], n_pts)
        
        fig, ax = plt.subplots(1, 3)
        
        fig.set_figheight(3)
        fig.set_figwidth(12)
        
        ax[0].plot(r, self.q(r))
        ax[0].set_xlabel('r')
        ax[0].set_ylabel('q')
        
        ax[0].plot(r, np.ones(r.size), 'k--')
        
        ax[1].plot(r, self.pr(r))
        ax[1].set_xlabel('r')
        ax[1].set_ylabel('p')
        
        ax[2].plot(r, self.nr(r))
        ax[2].set_xlabel('r')
        ax[2].set_ylabel('n')
        
        plt.subplots_adjust(wspace=0.4)
        
        plt.show()


    
    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================
    
    # equilibrium magnetic field (x-component)
    def b_x(self, x, y, z):
        """ Equilibrium magnetic field (x-component).
        """
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
    def b_y(self, x, y, z):
        """ Equilibrium magnetic field (y-component).
        """
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
    def b_z(self, x, y, z):
        """ Equilibrium magnetic field (z-component).
        """
        bz = self.params['B0'] - 0*x

        return bz
    
    # equilibrium current (x-component, curl of equilibrium magnetic field)
    def j_x(self, x, y, z):
        """ Equilibrium current (x-component).
        """
        jx = 0*x

        return jx

    # equilibrium current (y-component, curl of equilibrium magnetic field)
    def j_y(self, x, y, z):
        """ Equilibrium current (y-component).
        """
        jy = 0*x

        return jy

    # equilibrium current (z-component, curl of equilibrium magnetic field)
    def j_z(self, x, y, z):
        """ Equilibrium current (z-component).
        """
        r = self.r(x, y, z)
        
        q = self.q(r)
        q_p = self.q_p(r)
        
        if np.all(q >= 100.):
            jz = 0*x
        else:
            jz = self.params['B0']/(self.params['R0']*q**2)*(2*q - r*q_p)

        return jz

    # equilibrium pressure
    def p(self, x, y, z):
        """ Equilibrium pressure.
        """
        pp = self.pr(self.r(x, y, z))

        return pp

    # equilibrium number density
    def n(self, x, y, z):
        """ Equilibrium number density.
        """
        nn = self.nr(self.r(x, y, z))
        
        return nn
    
    
# =======================================================================================
class AdhocTorus(EquilibriumMHD):
    """
    Ad hoc tokamak MHD equilibrium with circular concentric flux surfaces.
    
    The profiles in toroidal coordinates (r, theta, phi) are:
    
    .. math::
    
        \mathbf B_0(r) &= \\frac{B_{0,\phi}R_0}{R} \left( \mathbf e_{\phi} + \\frac{r}{\\bar q(r) R_0} \mathbf e_{\\theta} \\right)\,,\qquad \\bar q(r) = q(r) \sqrt{1 - r^2/R_0^2}\,, \qquad q(r) = q_0 + ( q_1 - q_0 )\\frac{r^2}{a^2}\,,

        R &= R_0 + r \cos(\\theta)
    
        p(r) &= B_{0,\phi}^2\, a^2 \\frac{q_0}{ 2 R_0^2 (q1 - q0) } \left( \\frac{1}{q(r)^2} - \\frac{1}{q_1^2} \\right) \quad \\textnormal{ if $q_1$ not equal $q_0$},\quad p(r) = \\beta \\frac{B_{0,\phi}^2}{2} \quad \\textnormal{else} \,,\qquad \\textnormal{(cylindrical limit)}
        
        p(r) &= \\beta \\frac{B_{0,\phi}^2}{2} \left( 1 - p_1 \\frac{ r^2}{a^2} - p_2 \\frac{r^4}{a^4} \\right)\,,\qquad \\textnormal{(ad hoc profile)}
            
        n(r) &= n_a + ( 1 - n_a ) ( 1 - (r/a)^{n_1} )^{n_2}\,.
    
    Parameters
    ----------
        params: dict
            Parameters that characterize the MHD equilibrium.

                * a      : minor radius of torus
                * R0     : major radius of torus
                * B0     : on-axis toroidal magnetic field
                * q0     : safety factor at r=0
                * q1     : safety factor at r=a
                * n1     : shape factor for number density profile 
                * n2     : shape factor for number density profile 
                * na     : number density at r=a
                * p_kind : kind of pressure profile (0 : cylindrical limit, 1 : ad hoc)
                * p1     : shape factor for ad hoc pressure profile
                * p2     : shape factor for ad hoc pressure profile
                * beta   : on-axis plasma beta in % (ratio of kinetic pressure to magnetic pressure)
            
        domain: struphy.geometry.domain_3d.Domain
            All things mapping. Enables pull-backs if set.   
    """
    
    def __init__(self, params=None, domain=None):
        
        # set default parameters
        if params is None:
            params_default = {'a'      : 1., 
                              'R0'     : 10., 
                              'B0'     : 3., 
                              'q0'     : 1.71, 
                              'q1'     : 1.87, 
                              'n1'     : 0., 
                              'n2'     : 0., 
                              'na'     : 1., 
                              'p_kind' : 1, 
                              'p1'     : 0., 
                              'p2'     : 0., 
                              'beta'   : 0.179}
            
            super().__init__(params_default, domain)
        
        # or check if given parameter dicitionary is complete
        else:
            assert 'a'  in params
            assert 'R0' in params

            assert 'B0' in params

            assert 'q0' in params
            assert 'q1' in params

            assert 'n1' in params
            assert 'n2' in params
            assert 'na' in params

            assert 'p_kind' in params

            if params['p_kind'] == 1:
                assert 'p1' in params
                assert 'p2' in params
                assert 'beta' in params
            else:
                if params['q0'] == params['q1']:
                    assert 'beta' in params

            super().__init__(params, domain)
        
        # inverse toroidal coordinate transformation (x, y, z) --> (r, theta, phi)
        self.r     = lambda x, y, z : np.sqrt((np.sqrt(x**2 + z**2) - self.params['R0'])**2 + y**2)
        self.theta = lambda x, y, z : np.arctan2(y, np.sqrt(x**2 + z**2) - self.params['R0'])
        self.phi   = lambda x, y, z : np.arctan2(z, x)
        
        # local inverse aspect ratio
        self.eps_loc = lambda r : r/self.params['R0']
        
        # distance from axis of symmetry
        self.R = lambda r, theta : self.params['R0']*(1 + self.eps_loc(r)*np.cos(theta))
        
        
    # ===============================================================
    #           profiles for an ad hoc tokamak equilibrium
    # ===============================================================
    def nr(self, r):
        """ Radial number density profile.
        """
        nout = (1 - self.params['na'])*(1 - (r/self.params['a'])**self.params['n1'])**self.params['n2'] + self.params['na']
        
        return nout
    
    def q(self, r):
        """ Radial safety factor profile.
        """
        qout = self.params['q0'] + (self.params['q1'] - self.params['q0'])*(r/self.params['a'])**2
        
        return qout
    
    def q_p(self, r):
        """ Radial derivative of safety factor profile.
        """
        qout = 2*(self.params['q1'] - self.params['q0'])*r/self.params['a']**2
        
        return qout
    
    def pr(self, r):
        """ Radial pressure profile.
        """
        eps = self.params['a']/self.params['R0']
        
        if self.params['p_kind'] == 0:

            if self.params['q0'] == self.params['q1']:
                pout = self.params['B0']**2*self.params['beta']/200 - 0*r
            else:
                pout = self.params['B0']**2*eps**2*self.params['q0']/(2*(self.params['q1'] - self.params['q0']))*(1/self.q(r)**2 - 1/self.params['q1']**2)
                
        else:
            
            pout = self.params['B0']**2*self.params['beta']/200*(1 - self.params['p1']*r**2/self.params['a']**2 - self.params['p2']*r**4/self.params['a']**4)
               
        return pout
    
    def plot_profiles(self, n_pts=501):
        """ Plots radial profiles.
        """
        
        import matplotlib.pyplot as plt
        
        r = np.linspace(0., self.params['a'], n_pts)
        
        fig, ax = plt.subplots(1, 3)
        
        fig.set_figheight(3)
        fig.set_figwidth(12)
        
        ax[0].plot(r, self.q(r))
        ax[0].set_xlabel('r')
        ax[0].set_ylabel('q')
        
        ax[0].plot(r, np.ones(r.size), 'k--')
        
        ax[1].plot(r, self.pr(r))
        ax[1].set_xlabel('r')
        ax[1].set_ylabel('p')
        
        ax[2].plot(r, self.nr(r))
        ax[2].set_xlabel('r')
        ax[2].set_ylabel('n')
        
        plt.subplots_adjust(wspace=0.4)
        
        plt.show()
        
    
    
    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================
    
    # equilibrium magnetic field (x-component)
    def b_x(self, x, y, z):
        """Equilibrium magnetic field (x-component)."""
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi   = self.phi(x, y, z)
        
        q = self.q(r)
        q_bar = q*np.sqrt(1 - self.eps_loc(r)**2)
        
        # poloidal component
        if np.all(q >= 100.):
            b_theta = 0*r
        else:
            b_theta = self.params['B0']*r/(self.R(r, theta)*q_bar)
            
        # toroidal component
        b_phi = self.params['B0']*self.params['R0']/self.R(r, theta)

        # cartesian x-component
        bx = -b_theta*np.sin(theta)*np.cos(phi) - b_phi*np.sin(phi)

        return bx
    
    # equilibrium magnetic field (y-component)
    def b_y(self, x, y, z):
        """Equilibrium magnetic field (y-component)."""
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi   = self.phi(x, y, z)
        
        q = self.q(r)
        q_bar = q*np.sqrt(1 - self.eps_loc(r)**2)
        
        # poloidal component
        if np.all(q >= 100.):
            b_theta = 0*r
        else:
            b_theta = self.params['B0']*r/(self.R(r, theta)*q_bar)

        # cartesian y-component
        by = b_theta*np.cos(theta)

        return by
    
    # equilibrium magnetic field (z-component)
    def b_z(self, x, y, z):
        """Equilibrium magnetic field (z-component)."""
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi   = self.phi(x, y, z)
        
        q = self.q(r)
        q_bar = q*np.sqrt(1 - self.eps_loc(r)**2)
        
        # poloidal component
        if np.all(q >= 100.):
            b_theta = 0*r
        else:
            b_theta = self.params['B0']*r/(self.R(r, theta)*q_bar)
            
        # toroidal component
        b_phi = self.params['B0']*self.params['R0']/self.R(r, theta)

        # cartesian x-component
        bz = -b_theta*np.sin(theta)*np.sin(phi) + b_phi*np.cos(phi)

        return bz
    
    # equilibrium current (x-component, curl of equilibrium magnetic field)
    def j_x(self, x, y, z):
        """Equilibrium current (x-component)."""
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi   = self.phi(x, y, z)
        
        q = self.q(r)
        q_p = self.q_p(r)
        
        q_bar = q*np.sqrt(1 - self.eps_loc(r)**2)
        q_bar_p = q_p*np.sqrt(1 - self.eps_loc(r)**2) - q*self.eps_loc(r)/(self.params['R0']*np.sqrt(1 - self.eps_loc(r)**2))
        
        # toroidal component
        if np.all(q >= 100.):
            j_phi = 0*r
        else:
            j_phi = self.params['B0']/(self.R(r, theta)*q_bar**2)*(2*q_bar - r*q_bar_p - r/self.R(r, theta)*q_bar*np.cos(theta))
        
        # cartesian x-component
        jx = -j_phi*np.sin(phi)

        return jx

    # equilibrium current (y-component, curl of equilibrium magnetic field)
    def j_y(self, x, y, z):
        """Equilibrium current (y-component)."""
        jy = 0*x

        return jy

    # equilibrium current (z-component, curl of equilibrium magnetic field)
    def j_z(self, x, y, z):
        """Equilibrium current (z-component)."""
        r     = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi   = self.phi(x, y, z)
        
        q = self.q(r)
        q_p = self.q_p(r)
        
        q_bar = q*np.sqrt(1 - self.eps_loc(r)**2)
        q_bar_p = q_p*np.sqrt(1 - self.eps_loc(r)**2) - q*self.eps_loc(r)/(self.params['R0']*np.sqrt(1 - self.eps_loc(r)**2))
        
        # toroidal component
        if np.all(q >= 100.):
            j_phi = 0*r
        else:
            j_phi = self.params['B0']/(self.R(r, theta)*q_bar**2)*(2*q_bar - r*q_bar_p - r/self.R(r, theta)*q_bar*np.cos(theta))
        
        # cartesian x-component
        jz = j_phi*np.cos(phi)
        
        return jz

    # equilibrium pressure
    def p(self, x, y, z):
        """Equilibrium pressure."""
        pp = self.pr(self.r(x, y, z))

        return pp

    # equilibrium number density
    def n(self, x, y, z):
        """Equilibrium number density."""
        nn = self.nr(self.r(x, y, z))

        return nn