import numpy as np

from struphy.fields_background.mhd_equil.base import CartesianMHDequilibrium, LogicalMHDequilibrium


class HomogenSlab(CartesianMHDequilibrium):
    r"""
    Homogeneous MHD equilibrium in slab geometry.

    .. math::

        \mathbf B_0 = B_{0x}\,\mathbf e_x + B_{0y}\,\mathbf e_y + B_{0z}\,\mathbf e_z = const.\,,
        \qquad p_0 = \beta \frac{|\mathbf B_0|^2}{2}\,,\qquad n_0 = const.\,.

    Parameters
    ----------
        params: dict
            Parameters that characterize the MHD equilibrium.

                * B0x  : magnetic field in x-direction
                * B0y  : magnetic field in y-direction
                * B0z  : magnetic field in z-direction
                * beta : plasma beta in % (ratio of kinetic pressure to magnetic pressure)
                * n0   : number density            
    """

    def __init__(self, params=None):

        # set default parameters
        if params is None:
            params = {'B0x': 0.,
                      'B0y': 0.,
                      'B0z': 1.,
                      'beta': 100.,
                      'n0': 1.}
        # or check if given parameter dictionary is complete
        else:
            assert 'B0x' in params
            assert 'B0y' in params
            assert 'B0z' in params

            assert 'beta' in params

            assert 'n0' in params

        self._params = params

    @property
    def params(self):
        '''Parameters describing the equilibrium.'''
        return self._params

    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================

    # equilibrium magnetic field 
    def b_xyz(self, x, y, z):
        """ Equilibrium magnetic field (x-component).
        """
        bx = self.params['B0x'] - 0*x
        by = self.params['B0y'] - 0*x
        bz = self.params['B0z'] - 0*x

        return bx, by, bz

    # equilibrium current (curl of equilibrium magnetic field)
    def j_xyz(self, x, y, z):
        """ Equilibrium current.
        """
        jx = 0*x
        jy = 0*x
        jz = 0*x

        return jx, jy, jz

    # equilibrium pressure
    def p_xyz(self, x, y, z):
        """ Equilibrium pressure.
        """
        pp = self.params['beta']/200*(self.params['B0x']**2 +
                                      self.params['B0y']**2 + self.params['B0z']**2) - 0*x

        return pp

    # equilibrium number density
    def n_xyz(self, x, y, z):
        """ Equilibrium number density.
        """
        nn = self.params['n0'] - 0*x

        return nn


class ShearedSlab(CartesianMHDequilibrium):
    r"""
    Sheared slab MHD equilibrium in Cartesian space :math:`(x, y, z)`. Profiles depend on :math:`x` solely. 

    .. math::

        \mathbf B_0(x) &= B_{0z} \left( \mathbf e_z + \frac{a}{q(x)R_0}\mathbf e_y\right)\,,\qquad q(x) = q_0 + ( q_1 - q_0 )\frac{x^2}{a^2}\,,

        p_0(x) &= \beta\frac{B_{0z}^2}{2} \left( 1 + \frac{a^2}{q(x)^2 R_0^2} \right) + B_{0z}^2 \frac{a^2}{R_0^2} \left( \frac{1}{q_0^2} - \frac{1}{q(x)^2} \right)\,,

        n_0(x) &= n_a + ( 1 - n_a ) \left( 1 - \left(\frac{x}{a}\right)^{n_1} \right)^{n_2} \,.

    Parameters
    ----------
        params: dict
            Parameters that characterize the MHD equilibrium.

                * a    : minor radius (Lx = a, Ly = 2*pi*a)
                * R0   : major radius (Lz = 2*pi*R0)
                * B0   : magnetic field in z-direction
                * q0   : safety factor at x=0
                * q1   : safety factor at x=a
                * n1   : 1st shape factor for number density profile 
                * n2   : 2nd shape factor for number density profile 
                * na   : number density at x=a
                * beta : plasma beta in % at x=0 (ratio of kinetic pressure to magnetic pressure)            
    """

    def __init__(self, params=None):

        # set default parameters
        if params is None:
            params = {'a': 1.,
                      'R0': 3.,
                      'B0': 1.,
                      'q0': 1.05,
                      'q1': 1.80,
                      'n1': 0.,
                      'n2': 0.,
                      'na': 1.,
                      'beta': 10.}
        # or check if given parameter dictionary is complete
        else:
            assert 'a' in params
            assert 'R0' in params

            assert 'B0' in params

            assert 'q0' in params
            assert 'q1' in params

            assert 'n1' in params
            assert 'n2' in params
            assert 'na' in params

            assert 'beta' in params

        self._params = params

    @property
    def params(self):
        '''Parameters describing the equilibrium.'''
        return self._params

    # ===============================================================
    #             profiles for a sheared slab geometry
    # ===============================================================

    def nx(self, x):
        """ Radial (x) number density profile.
        """
        nout = (1 - self.params['na'])*(1 - (x/self.params['a']) **
                                        self.params['n1'])**self.params['n2'] + self.params['na']

        return nout

    def q(self, x):
        """ Radial (x) safety factor profile.
        """
        qout = self.params['q0'] + (self.params['q1'] -
                                    self.params['q0'])*(x/self.params['a'])**2

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
            pout = self.params['B0']**2*self.params['beta']/200*(
                1 + eps**2/q**2) + self.params['B0']**2*eps**2*(1/self.params['q0']**2 - 1/q**2)

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

    # equilibrium magnetic field 
    def b_xyz(self, x, y, z):
        """ Equilibrium magnetic field.
        """
        bx = 0*x

        q = self.q(x)
        eps = self.params['a']/self.params['R0']
        if np.all(q >= 100.):
            by = 0*x
            bz = self.params['B0'] - 0*x
        elif np.all(q == 0.):
            by = self.params['B0'] - 0*x
            bz = 0*x
        else:
            by = self.params['B0']*eps/q
            bz = self.params['B0'] - 0*x

        return bx, by, bz

    # equilibrium current (curl of equilibrium magnetic field)
    def j_xyz(self, x, y, z):
        """ Equilibrium current (x-component).
        """
        jx = 0*x
        jy = 0*x

        q = self.q(x)
        eps = self.params['a']/self.params['R0']
        if np.all(q >= 100.):
            jz = 0*x
        elif np.all(q == 0.):
            jz = 0*x
        else:
            jz = -self.params['B0']*eps*self.q_p(x)/q**2

        return jx, jy, jz

    # equilibrium pressure
    def p_xyz(self, x, y, z):
        """ Equilibrium pressure.
        """
        pp = self.px(x)

        return pp

    # equilibrium number density
    def n_xyz(self, x, y, z):
        """ Equilibrium number density.
        """
        nn = self.nx(x)

        return nn


class ScrewPinch(CartesianMHDequilibrium):
    r"""
    Straight tokamak (screw pinch) MHD equilibrium.

    The profiles in cylindrical coordinates :math:`(r, \theta, z)` are:

    .. math::

        \mathbf B_0(r) &= B_{0z}\left( \mathbf e_z + \frac{r}{q(r) R_0}\mathbf e_\theta \right)\,,\qquad q(r) = q_0 + ( q_1 - q_0 )\frac{r^2}{a^2}\,,

        p_0(r) &= \left\{\begin{aligned}
        &\frac{B_{0z}^2 a^2 q_0}{ 2 R_0^2(q_1 - q_0) } \left( \frac{1}{q(r)^2} - \frac{1}{q_1^2} \right) \quad \textnormal{if}\quad q_1\neq q_0\,, 

        &\beta\frac{B_{0z}^2}{2} \quad \textnormal{else}\,,
        \end{aligned}\right.

        n_0(r) &= n_a + ( 1 - n_a )\left( 1 - \left(\frac{r}{a}\right)^{n_1} \right)^{n_2}\,.

    Parameters
    ----------
        params: dict
            Parameters that characterize the MHD equilibrium.

                * a    : minor radius (radius of cylinder)
                * R0   : major radius (Lz = 2*pi*R0)
                * B0   : magnetic field in z-direction
                * q0   : safety factor at r=0
                * q1   : safety factor at r=a
                * n1   : 1st shape factor for number density profile 
                * n2   : 2nd shape factor for number density profile 
                * na   : number density at r=a
                * beta : plasma beta in % for flat safety factor (ratio of kinetic pressure to magnetic pressure)  
    """

    def __init__(self, params=None):

        # set default parameters
        if params is None:
            params = {'a': 1.,
                      'R0': 5.,
                      'B0': 1.,
                      'q0': 1.05,
                      'q1': 1.80,
                      'n1': 0.,
                      'n2': 0.,
                      'na': 1.}
        # or check if given parameter dictionary is complete
        else:
            assert 'a' in params
            assert 'R0' in params

            assert 'B0' in params

            assert 'q0' in params
            assert 'q1' in params

            assert 'n1' in params
            assert 'n2' in params
            assert 'na' in params

            if params['q0'] == params['q1']:
                assert 'beta' in params

        self._params = params

        # inverse cylindrical coordinate transformation (x, y, z) --> (r, theta, phi)
        self.r = lambda x, y, z: np.sqrt(x**2 + y**2)
        self.theta = lambda x, y, z: np.arctan2(y, x)
        self.z = lambda x, y, z: 1*z

    @property
    def params(self):
        '''Parameters describing the equilibrium.'''
        return self._params

    # ===============================================================
    #           profiles for a straight tokamak equilibrium
    # ===============================================================

    def nr(self, r):
        """ Radial number density profile.
        """
        nout = (1 - self.params['na'])*(1 - (r/self.params['a']) **
                                        self.params['n1'])**self.params['n2'] + self.params['na']

        return nout

    def q(self, r):
        """ Radial safety factor profile.
        """
        qout = self.params['q0'] + (self.params['q1'] -
                                    self.params['q0'])*(r/self.params['a'])**2

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
            pout = self.params['B0']**2*eps**2*self.params['q0']/(
                2*(self.params['q1'] - self.params['q0']))*(1/self.q(r)**2 - 1/self.params['q1']**2)

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

    # equilibrium magnetic field 
    def b_xyz(self, x, y, z):
        """ Equilibrium magnetic field (x-component).
        """
        r = self.r(x, y, z)
        theta = self.theta(x, y, z)
        q = self.q(r)
        # azimuthal component
        if np.all(q >= 100.):
            b_theta = 0*r
        else:
            b_theta = self.params['B0']*r/(self.params['R0']*q)
        # cartesian x-component
        bx = -b_theta*np.sin(theta)
        by = b_theta*np.cos(theta)
        bz = self.params['B0'] - 0*x

        return bx, by, bz

    # equilibrium current (curl of equilibrium magnetic field)
    def j_xyz(self, x, y, z):
        """ Equilibrium current (x-component).
        """
        jx = 0*x
        jy = 0*x

        r = self.r(x, y, z)
        q = self.q(r)
        q_p = self.q_p(r)
        if np.all(q >= 100.):
            jz = 0*x
        else:
            jz = self.params['B0']/(self.params['R0']*q**2)*(2*q - r*q_p)

        return jx, jy, jz

    # equilibrium pressure
    def p_xyz(self, x, y, z):
        """ Equilibrium pressure.
        """
        pp = self.pr(self.r(x, y, z))

        return pp

    # equilibrium number density
    def n_xyz(self, x, y, z):
        """ Equilibrium number density.
        """
        nn = self.nr(self.r(x, y, z))

        return nn


class AdhocTorus(CartesianMHDequilibrium):
    r"""
    Ad hoc tokamak MHD equilibrium with circular concentric flux surfaces.

    The profiles in toroidal coordinates :math:`(r, \theta, \phi)` with :math:`R=R_0+r\cos(\theta)` are:

    .. math::

        \mathbf B_0(r) &= \frac{B_{0\phi}R_0}{R} \left( \mathbf e_{\phi} + \frac{r}{\bar q(r) R_0} \mathbf e_{\theta} \right)\,,\qquad \bar q(r) = q(r) \sqrt{1 - \frac{r^2}{R_0^2}}\,, \qquad q(r) = q_0 + ( q_1 - q_0 )\frac{r^2}{a^2}\,,

        p_0(r) &= \left\{\begin{aligned}
        &\frac{B_{0\phi}^2\, a^2 q_0}{ 2 R_0^2 (q_1 - q_0) } \left( \frac{1}{q(r)^2} - \frac{1}{q_1^2} \right) \quad \textnormal{if} \quad p_\textnormal{kind}=0 \quad \textnormal{and} \quad q_1\neq q_0\,, 

        &\beta \frac{B_{0\phi}^2}{2} \quad \textnormal{if} \quad p_\textnormal{kind}=0 \quad \textnormal{and} \quad q_1= q_0 \,,

        &\beta \frac{B_{0\phi}^2}{2} \left( 1 - p_1 \frac{r^2}{a^2} - p_2 \frac{r^4}{a^4} \right) \quad \textnormal{if} \quad p_\textnormal{kind}=1\,,
        \end{aligned}\right.

        n_0(r) &= n_a + ( 1 - n_a ) \left( 1 - \left(\frac{r}{a}\right)^{n_1} \right)^{n_2}\,.

    Parameters
    ----------
        params: dict
            Parameters that characterize the MHD equilibrium.

                * a      : minor radius of torus
                * R0     : major radius of torus
                * B0     : on-axis toroidal magnetic field
                * q0     : safety factor at r=0
                * q1     : safety factor at r=a
                * n1     : 1st shape factor for number density profile 
                * n2     : 2nd shape factor for number density profile 
                * na     : number density at r=a
                * p_kind : kind of pressure profile (0 : cylindrical limit, 1 : ad hoc)
                * p1     : 1st shape factor for ad hoc pressure profile
                * p2     : 2nd shape factor for ad hoc pressure profile
                * beta   : on-axis plasma beta in % (ratio of kinetic pressure to magnetic pressure)   
    """

    def __init__(self, params=None):

        # set default parameters
        if params is None:
            params = {'a': 1.,
                      'R0': 10.,
                      'B0': 3.,
                      'q0': 1.71,
                      'q1': 1.87,
                      'n1': 0.,
                      'n2': 0.,
                      'na': 1.,
                      'p_kind': 1,
                      'p1': 0.,
                      'p2': 0.,
                      'beta': 0.179}
        # or check if given parameter dictionary is complete
        else:
            assert 'a' in params
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

        self._params = params

        # inverse toroidal coordinate transformation (x, y, z) --> (r, theta, phi)
        self.r = lambda x, y, z: np.sqrt(
            (np.sqrt(x**2 + y**2) - self.params['R0'])**2 + z**2)
        self.theta = lambda x, y, z: -np.arctan2(
            z, np.sqrt(x**2 + y**2) - self.params['R0'])
        self.phi = lambda x, y, z: np.arctan2(y, x)

        # local inverse aspect ratio
        self.eps_loc = lambda r: r/self.params['R0']

        # distance from axis of symmetry
        self.R = lambda r, theta: self.params['R0'] * \
            (1 + self.eps_loc(r)*np.cos(theta))

    @property
    def params(self):
        '''Parameters describing the equilibrium.'''
        return self._params

    # ===============================================================
    #           profiles for an ad hoc tokamak equilibrium
    # ===============================================================

    def nr(self, r):
        """ Radial number density profile.
        """
        nout = (1 - self.params['na'])*(1 - (r/self.params['a']) **
                                        self.params['n1'])**self.params['n2'] + self.params['na']

        return nout

    def q(self, r):
        """ Radial safety factor profile.
        """
        qout = self.params['q0'] + (self.params['q1'] -
                                    self.params['q0'])*(r/self.params['a'])**2

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
                pout = self.params['B0']**2*eps**2*self.params['q0']/(
                    2*(self.params['q1'] - self.params['q0']))*(1/self.q(r)**2 - 1/self.params['q1']**2)

        else:

            pout = self.params['B0']**2*self.params['beta']/200*(
                1 - self.params['p1']*r**2/self.params['a']**2 - self.params['p2']*r**4/self.params['a']**4)

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

    # equilibrium magnetic field 
    def b_xyz(self, x, y, z):
        """ Equilibrium magnetic field.
        """
        r = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi = self.phi(x, y, z)

        q = self.q(r)
        q_bar = q*np.sqrt(1 - self.eps_loc(r)**2)

        # poloidal component
        if np.all(q >= 100.):
            b_theta = 0*r
        else:
            b_theta = self.params['B0']*r/(self.R(r, theta)*q_bar)

        # toroidal component
        b_phi = self.params['B0']*self.params['R0']/self.R(r, theta)

        # Cartesian components
        bx = -b_theta*np.sin(theta)*np.cos(phi) - b_phi*np.sin(phi)
        by = -b_theta*np.sin(theta)*np.sin(phi) + b_phi*np.cos(phi)
        bz = -b_theta*np.cos(theta)

        return bx, by, bz

    # equilibrium current (curl of equilibrium magnetic field)
    def j_xyz(self, x, y, z):
        """ Equilibrium current.
        """
        r = self.r(x, y, z)
        theta = self.theta(x, y, z)
        phi = self.phi(x, y, z)

        q = self.q(r)
        q_p = self.q_p(r)

        q_bar = q*np.sqrt(1 - self.eps_loc(r)**2)
        q_bar_p = q_p*np.sqrt(1 - self.eps_loc(r)**2) - q*self.eps_loc(r) / \
            (self.params['R0']*np.sqrt(1 - self.eps_loc(r)**2))

        # toroidal component
        if np.all(q >= 100.):
            j_phi = 0*r
        else:
            j_phi = self.params['B0']/(self.R(r, theta)*q_bar**2)*(
                2*q_bar - r*q_bar_p - r/self.R(r, theta)*q_bar*np.cos(theta))

        # Cartesian x-components
        jx = -j_phi*np.sin(phi)
        jy = j_phi*np.cos(phi)
        jz = 0*x

        return jx, jy, jz

    # equilibrium pressure
    def p_xyz(self, x, y, z):
        """ Equilibrium pressure.
        """
        pp = self.pr(self.r(x, y, z))

        return pp

    # equilibrium number density
    def n_xyz(self, x, y, z):
        """ Equilibrium number density.
        """
        nn = self.nr(self.r(x, y, z))

        return nn


class EQDSKequilibrium(CartesianMHDequilibrium):
    '''Interface to `EQDSK file format <https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf>`_.

    Parameters
    ----------
    params: dict
        Parameters that characterize the MHD equilibrium.

        * rel_path : str
            Whether file is relative to "<struphy_path>/fields_background/mhd_equil/gvec", or is an absolute path.
        * file : str
            Path to eqdsk file.
        * data_type : int
            0: there is no space between data, 1: there is space between data.
        * p_for_psi : list[int]
            Spline degree in each direction used for interpolation of psi data.
        * Nel : tuple[int]
            Number of cells in each direction used for field line tracing.   
        * p : tuple[int]
            Spline degree in each direction used for field line tracing.
        * theta : str
            Choose theta parametrization: 'equal_angle', NOT YET: 'equal_arc_length' or 'sfl' (PEST).
        * tor_period : int
            Toroidal periodicity built into the mapping: phi = 2*pi * eta3 / tor_period.
    '''

    def __init__(self, params=None, show=False):

        from struphy.geometry.base import spline_interpolation_nd, PoloidalSplineTorus
        from struphy.b_splines.bspline_evaluation_2d import evaluate
        from struphy.fields_background.mhd_equil.eqdsk import readeqdsk
        import struphy
        import numpy as np
        from matplotlib import pyplot as plt

        if params is None:
            params = {'rel_path': True,
                      'file': 'AUGNLED_g031213.00830.high',
                      'data_type': 0,
                      'flux_resolution': 16,
                      'psi_resolution': [32, 32],
                      'p_for_psi': [3, 3],
                      'Nel': [16, 32],
                      'p': [3, 3],
                      'theta': 'equal_angle',
                      'tor_period': 3
                      }
        else:
            assert 'rel_path' in params
            assert 'file' in params
            assert 'data_type' in params
            assert 'flux_resolution' in params
            assert 'psi_resolution' in params
            assert 'p_for_psi' in params
            assert 'Nel' in params
            assert 'p' in params
            assert 'theta' in params
            assert 'tor_period' in params

        self._params = params

        if params['rel_path']:
            _path = struphy.__path__[0] + \
                '/fields_background/mhd_equil/eqdsk/data/' + params['file']
        else:
            _path = params['file']

        eqdsk = readeqdsk.Geqdsk()
        eqdsk.openFile(_path, data_type=params['data_type'])

        # Number of horizontal R grid points
        nR = eqdsk.data['nw'][0]
        # Number of vertical Z grid points
        nZ = eqdsk.data['nh'][0]
        # toroidal field function in m-T on flux grid, g = B^1_phi
        g_profile = eqdsk.data['fpol'][0]
        # plasma pressure in Nt/m^2 on uniform flux grid
        pres_profile = eqdsk.data['pres'][0]
        # poloidal flux in Weber/rad on the rectangular grid points
        psi = eqdsk.data['psirz'][0].T
        # poloidal flux in Weber/rad at the plasma boundary
        psi_edge = eqdsk.data['sibry'][0]
        # q values on uniform flux grid from axis to boundary
        q_profile = eqdsk.data['qpsi'][0]
        # Horizontal dimension in meter of computational box
        self._rdim = eqdsk.data['rdim'][0]
        # Vertical dimension in meter of computational box
        self._zdim = eqdsk.data['zdim'][0]
        # Minimum R in meter of rectangular computational box
        rleft = eqdsk.data['rleft'][0]
        # Z of center of computational box in meter
        zmid = eqdsk.data['zmid'][0]
        # R of magnetic axis in meter
        R_at_axis = eqdsk.data['rmaxis'][0]
        # Z of magnetic axis in meter
        Z_at_axis = eqdsk.data['zmaxis'][0] 

        assert g_profile.size == pres_profile.size
        assert g_profile.size == q_profile.size
        assert psi.shape == (nR, nZ)

        p = params['p_for_psi']

        # interpolate toroidal field function and pressure profile from smoothed data
        self._psimin = psi.min()
        self._psidim = psi.max() - self._psimin

        flux_grid = np.linspace(self._psimin, psi.max(), g_profile.size)

        smoothing = g_profile.size // params['flux_resolution']
        g_smoothed = g_profile[::smoothing]
        pres_smoothed = pres_profile[::smoothing]
        q_smoothed = q_profile[::smoothing]
        flux_grid_smoothed = flux_grid[::smoothing]
        
        i_grid = (flux_grid_smoothed - self._psimin) / self._psidim

        self._cg, self._Tg, self._indNg = spline_interpolation_nd(
            [p[0]], [False], [i_grid], g_smoothed)

        self._cpres, self._Tpres, self._indNpres = spline_interpolation_nd(
            [p[0]], [False], [i_grid], pres_smoothed)

        self._cq, self._Tq, self._indNq = spline_interpolation_nd(
            [p[0]], [False], [i_grid], q_smoothed)

        # interpolate psi and s (= normalized psi) from smoothed point data
        R = np.linspace(rleft, rleft + self._rdim, nR)
        Z = np.linspace(zmid - self._zdim/2, zmid + self._zdim/2, nZ)

        self._rmin = R.min()
        self._zmin = Z.min()

        R1 = (R - self._rmin) / self._rdim
        Z1 = (Z - self._zmin) / self._zdim

        s_mat = np.sqrt((psi - self._psimin)/(psi_edge - self._psimin))

        smoothing_r = psi.shape[0] // params['psi_resolution'][0]
        smoothing_z = psi.shape[1] // params['psi_resolution'][1]
        s_mat_smoothed = s_mat[::smoothing_r, ::smoothing_z]
        psi_smoothed = psi[::smoothing_r, ::smoothing_z]
        R1_smoothed = R1[::smoothing_r]
        Z1_smoothed = Z1[::smoothing_z]

        cs, knots, indN = spline_interpolation_nd(
            p, [False, False], [R1_smoothed, Z1_smoothed], s_mat_smoothed)

        self._cpsi, self._Tpsi, self._indNpsi = spline_interpolation_nd(
            p, [False, False], [R1_smoothed, Z1_smoothed], psi_smoothed)

        def s_fun(r, z):
            '''Single point evaluation of .'''
            return evaluate(1, 1, *knots, *p,
                            *indN, cs, (r - self._rmin) / self._rdim, (z - self._zmin) / self._zdim)

        # do the field line tracing
        cx, cy = self.field_line_tracing(
            s_fun, R_at_axis, Z_at_axis, self._rdim, params['Nel'], params['p'], theta=params['theta'])

        # Instantiate Torus domain
        params_map = {'cx': cx,
                      'cy': cy,
                      'Nel': params['Nel'],
                      'p': params['p'],
                      'spl_kind': [False, True],
                      }

        params_map['tor_period'] = params['tor_period']
        self._domain = PoloidalSplineTorus(params_map)

        # plot for testing
        if show:
            print('file: ', params['file'])

            # for testing: compute finite difference derivatives from data points
            dR = self._rdim / (nR - 1)
            dZ = self._zdim / (nZ - 1)
            dflux = flux_grid[1] - flux_grid[0]

            psi_r = np.zeros_like(psi)
            psi_r[1:-1, :] = (np.roll(psi, -1, axis=0)[1:-1, :] - np.roll(psi, 1, axis=0)[1:-1, :]) / (2*dR)
            psi_r[0, :] = (psi[1, :] - psi[0, :]) / dR
            psi_r[-1,:] = (psi[-1, :] - psi[-2, :]) / dR

            psi_z = np.zeros_like(psi)
            psi_z[:, 1:-1] = (np.roll(psi, -1, axis=1)[:, 1:-1] - np.roll(psi, 1, axis=1)[:, 1:-1]) / (2*dZ)
            psi_z[:, 0] = (psi[:, 1] - psi[:, 0]) / dZ
            psi_z[:, -1] = (psi[:, -1] - psi[:, -2]) / dZ

            psi_rr = np.zeros_like(psi)
            psi_rr[1:-1, :] = (np.roll(psi, -1, axis=0)[1:-1] - 2*psi[1:-1] + np.roll(psi, 1, axis=0)[1:-1]) / dR**2

            psi_zz = np.zeros_like(psi)
            psi_zz[:, 1:-1] = (np.roll(psi, -1, axis=1)[:, 1:-1] - 2*psi[:, 1:-1] + np.roll(psi, 1, axis=1)[:, 1:-1]) / dZ**2

            g_psi = np.zeros_like(g_profile)
            g_psi[1:-1] = (np.roll(g_profile, -1)[1:-1] - np.roll(g_profile, 1)[1:-1]) / (2*dflux)
            g_psi[0] = (g_profile[1] - g_profile[0]) / dflux 
            g_psi[-1] = (g_profile[-1] - g_profile[-2]) / dflux

            pres_psi = np.zeros_like(pres_profile)
            pres_psi[1:-1] = (np.roll(pres_profile, -1)[1:-1] - np.roll(pres_profile, 1)[1:-1]) / (2*dflux)
            pres_psi[0] = (pres_profile[1] - pres_profile[0]) / dflux
            pres_psi[-1] = (pres_profile[-1] - pres_profile[-2]) / dflux

            q_psi = np.zeros_like(q_profile)
            q_psi[1:-1] = (np.roll(q_profile, -1)[1:-1] - np.roll(q_profile, 1)[1:-1]) / (2*dflux)
            q_psi[0] = (q_profile[1] - q_profile[0]) / dflux
            q_psi[-1] = (q_profile[-1] - q_profile[-2]) / dflux

            boundary_ind = np.argmin(np.abs(flux_grid - psi_edge))

            plt.figure(figsize=(13, 8))
            plt.subplot(2, 2, 1)
            plt.plot(flux_grid[:boundary_ind], g_profile[:boundary_ind], 'b', label='point data, size=' + str(g_profile.size))
            plt.plot(flux_grid[:boundary_ind], self.g_1d(flux_grid[:boundary_ind]), 'r--', label='smoothed with pts=' + str(flux_grid_smoothed.size))
            #plt.plot([psi_edge, psi_edge], [g_profile.min(), g_profile.max()], 'k--', label='plasma boundary')
            plt.legend()
            plt.xlim(self._psimin, psi_edge)
            plt.xlabel('$\psi$')
            plt.ylabel('g [m-T]')
            plt.title('toroidal field function g')
            plt.subplot(2, 2, 2)
            plt.plot(flux_grid[:boundary_ind], pres_profile[:boundary_ind], 'b', label='point data, size=' + str(pres_profile.size))
            plt.plot(flux_grid[:boundary_ind], self.pres_1d(flux_grid[:boundary_ind]), 'r--', label='smoothed with pts=' + str(flux_grid_smoothed.size))
            plt.legend()
            plt.xlim(self._psimin, psi_edge)
            plt.xlabel('$\psi$')
            plt.ylabel('p [Pascal]')
            plt.title('pressure profile')
            plt.subplot(2, 2, 3)
            plt.plot(flux_grid[:boundary_ind], g_psi[:boundary_ind], 'b', label='point data, size=' + str(g_psi.size))
            plt.plot(flux_grid[:boundary_ind], self.g_1d(flux_grid[:boundary_ind], der=1), 'r--', label='smoothed with pts=' + str(flux_grid_smoothed.size))
            plt.legend()
            plt.xlim(self._psimin, psi_edge)
            plt.xlabel('$\psi$')
            plt.ylabel('dg/d$\psi$')
            plt.subplot(2, 2, 4)
            plt.plot(flux_grid[:boundary_ind], pres_psi[:boundary_ind], 'b', label='point data, size=' + str(pres_psi.size))
            plt.plot(flux_grid[:boundary_ind], self.pres_1d(flux_grid[:boundary_ind], der=1), 'r--', label='smoothed with pts=' + str(flux_grid_smoothed.size))
            plt.legend()
            plt.xlim(self._psimin, psi_edge)
            plt.xlabel('$\psi$')
            plt.ylabel('dp/d$\psi$')

            plt.figure(figsize=(6.5, 8))
            plt.subplot(2, 1, 1)
            plt.plot(flux_grid[:boundary_ind], q_profile[:boundary_ind], 'b', label='point data, size=' + str(g_profile.size))
            plt.plot(flux_grid[:boundary_ind], self.q_1d(flux_grid[:boundary_ind]), 'r--', label='smoothed with pts=' + str(flux_grid_smoothed.size))
            plt.legend()
            plt.xlim(self._psimin, psi_edge)
            plt.xlabel('$\psi$')
            plt.ylabel('q')
            plt.title('safety factor q')
            plt.subplot(2, 1, 2)
            plt.plot(flux_grid[:boundary_ind], q_psi[:boundary_ind], 'b', label='point data, size=' + str(g_psi.size))
            plt.plot(flux_grid[:boundary_ind], self.q_1d(flux_grid[:boundary_ind], der=1), 'r--', label='smoothed with pts=' + str(flux_grid_smoothed.size))
            plt.legend()
            plt.xlim(self._psimin, psi_edge)
            plt.xlabel('$\psi$')
            plt.ylabel('dq/d$\psi$')

            plt.figure(figsize=(13, 6.5))
            plt.subplot(1, 2, 1)
            RR, ZZ = np.meshgrid(R, Z, indexing='ij')
            plt.contourf(RR, ZZ, psi, levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, psi, levels=50, colors='red', linewidths=.5)
            plt.contour(RR, ZZ, psi, levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.scatter(R_at_axis, Z_at_axis, 20, 'red', zorder=10)
            plt.axis('equal')
            plt.xlabel('R')
            plt.ylabel('Z')
            plt.title('eqdsk point data $\psi$, shape=' + str(psi.shape))
            plt.subplot(1, 2, 2)
            plt.contourf(RR, ZZ, self.psi_fun(RR, ZZ), levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=50,
                        colors='red', linewidths=.5)
            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.scatter(R_at_axis, Z_at_axis, 20, 'red', zorder=10)
            plt.axis('equal')
            plt.xlabel('R')
            plt.ylabel('Z')
            plt.title('interpolated $\psi$, smoothed with pts=' + str(psi_smoothed.shape))

            plt.figure(figsize=(13, 6.5))
            plt.subplot(1, 2, 1)
            _s_fun = np.vectorize(s_fun)
            plt.contourf(RR, ZZ, _s_fun(RR, ZZ), levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, _s_fun(RR, ZZ), levels=50,
                        colors='red', linewidths=.5)
            plt.contour(RR, ZZ, _s_fun(RR, ZZ), levels=[
                        1.], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('R')
            plt.ylabel('Z')
            plt.title(
                'normalized, interpolated data $s=\sqrt{(\psi - \psi_a) / (\psi_e - \psi_a)}$')
            plt.subplot(1, 2, 2)
            plt.contourf(RR, ZZ, self.g_fun(RR, ZZ), levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, self.g_fun(RR, ZZ), levels=50,
                        colors='red', linewidths=.5)
            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('R')
            plt.ylabel('Z')
            plt.title('$g(\psi(R, Z))$, smoothed')

            plt.figure(figsize=(13, 13))
            plt.subplot(2, 2, 1)
            plt.contourf(RR, ZZ, self.psi_fun(RR, ZZ, der='r'), levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ, der='r'), levels=50,
                        colors='red', linewidths=.5)
            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('R')
            plt.ylabel('Z')
            plt.title('r-derivative of interpolated $\psi$, smoothed')
            plt.subplot(2, 2, 2)
            plt.contourf(RR, ZZ, self.psi_fun(RR, ZZ, der='z'), levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ, der='z'), levels=50,
                        colors='red', linewidths=.5)
            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('R')
            plt.ylabel('Z')
            plt.title('z-derivative of interpolated $\psi$, smoothed')
            plt.subplot(2, 2, 3)
            plt.contourf(RR, ZZ, psi_r, levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, psi_r, levels=50,
                        colors='red', linewidths=.5)
            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('R')
            plt.ylabel('Z')
            plt.title('r-derivative $\psi$ data, FD')
            plt.subplot(2, 2, 4)
            plt.contourf(RR, ZZ, psi_z, levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, psi_z, levels=50,
                        colors='red', linewidths=.5)
            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('R')
            plt.ylabel('Z')
            plt.title('z-derivative $\psi$ data, FD')

            plt.figure(figsize=(13, 13))
            plt.subplot(2, 2, 1)
            plt.contourf(RR, ZZ, self.psi_fun(RR, ZZ, der='rr'), levels=50)
            plt.colorbar()
            # plt.contour(RR, ZZ, self.psi_fun(RR, ZZ, der='rr'), levels=50,
            #             colors='red', linewidths=.5)
            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('R')
            plt.ylabel('Z')
            plt.title('rr-derivative of interpolated $\psi$, smoothed')
            plt.subplot(2, 2, 2)
            plt.contourf(RR, ZZ, self.psi_fun(RR, ZZ, der='zz'), levels=50)
            plt.colorbar()
            # plt.contour(RR, ZZ, self.psi_fun(RR, ZZ, der='zz'), levels=50,
            #             colors='red', linewidths=.5)
            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('R')
            plt.ylabel('Z')
            plt.title('zz-derivative of interpolated $\psi$, smoothed')
            plt.subplot(2, 2, 3)
            plt.contourf(RR, ZZ, psi_rr, levels=50)
            plt.colorbar()
            # plt.contour(RR, ZZ, self.psi_fun(RR, ZZ, der='rr'), levels=50,
            #             colors='red', linewidths=.5)
            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('R')
            plt.ylabel('Z')
            plt.title('rr-derivative of point data')
            plt.subplot(2, 2, 4)
            plt.contourf(RR, ZZ, psi_zz, levels=50)
            plt.colorbar()
            # plt.contour(RR, ZZ, self.psi_fun(RR, ZZ, der='rr'), levels=50,
            #             colors='red', linewidths=.5)
            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('R')
            plt.ylabel('Z')
            plt.title('zz-derivative of point data')

            plt.figure(figsize=(13, 13))
            Y = 0.
            XX, YY, ZZ2 = np.meshgrid(R, Y, Z, indexing='ij')
            bx = self.b_xyz(XX, YY, ZZ2)[0].squeeze()
            by = self.b_xyz(XX, YY, ZZ2)[1].squeeze()
            bz = self.b_xyz(XX, YY, ZZ2)[2].squeeze()
            plt.subplot(2, 2, 1)
            plt.contourf(XX.squeeze(), ZZ2.squeeze(), bx, levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, psi, levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('bx')
            plt.subplot(2, 2, 2)
            plt.contourf(XX.squeeze(), ZZ2.squeeze(), by, levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, psi, levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('by')
            plt.subplot(2, 2, 3)
            plt.contourf(XX.squeeze(), ZZ2.squeeze(), bz, levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, psi, levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('bz')
            plt.subplot(2, 2, 4)
            plt.contourf(XX.squeeze(), ZZ2.squeeze(), np.sqrt(bx**2 + by**2 + bz**2), levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, psi, levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('abs(b)')

            plt.figure(figsize=(13, 13))
            jx = self.j_xyz(XX, YY, ZZ2)[0].squeeze()
            jy = self.j_xyz(XX, YY, ZZ2)[1].squeeze()
            jz = self.j_xyz(XX, YY, ZZ2)[2].squeeze()
            plt.subplot(2, 2, 1)
            plt.contourf(XX.squeeze(), ZZ2.squeeze(), jx, levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, psi, levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('jx')
            plt.subplot(2, 2, 2)
            plt.contourf(XX.squeeze(), ZZ2.squeeze(), jy, levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, psi, levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('jy')
            plt.subplot(2, 2, 3)
            plt.contourf(XX.squeeze(), ZZ2.squeeze(), jz, levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, psi, levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('jz')
            plt.subplot(2, 2, 4)
            plt.contourf(XX.squeeze(), ZZ2.squeeze(), np.sqrt(jx**2 + jy**2 + jz**2), levels=50)
            plt.colorbar()
            plt.contour(RR, ZZ, psi, levels=[
                        psi_edge], colors='black', linewidths=2.)
            plt.axis('equal')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('abs(j)')

    @property
    def domain(self):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        return self._domain

    @property
    def params(self):
        '''Parameters describing the equilibrium.'''
        return self._params

    def b_xyz(self, x, y, z):
        '''B-field in Cartesian coordinates.'''

        from struphy.geometry.base import Domain

        x, y, z, is_sparse_meshgrid = Domain.prepare_eval_pts(x, y, z)

        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        z = z + 0*r # broadcasting happens here

        r2 = r[:, 0, :]
        z2 = z[:, 0, :]

        # B as 2-form (second component is already multiplied by r)
        # TODO: remove is_sparse_meshgrid (not necessary anymore)
        b2_1_tmp = self.psi_fun(r2, z2, 'z', is_sparse_meshgrid)
        b2_2_tmp = self.g_fun(r2, z2, None, is_sparse_meshgrid)
        b2_3_tmp = - self.psi_fun(r2, z2, 'r', is_sparse_meshgrid)

        if is_sparse_meshgrid:
            shp = (r.shape[0], phi.shape[1], z.shape[2])
        else:
            shp = r.shape

        b2_1 = np.empty(shp)
        b2_2 = np.empty(shp)
        b2_3 = np.empty(shp)

        b2_1[:] = b2_1_tmp[:, None, :]
        b2_2[:] = b2_2_tmp[:, None, :]
        b2_3[:] = b2_3_tmp[:, None, :]

        # push-forward of b2
        b_x = (np.cos(phi)*b2_1 - np.sin(phi)*b2_2) / r
        b_y = (np.sin(phi)*b2_1 + np.cos(phi)*b2_2) / r
        b_z = b2_3 / r

        return b_x, b_y, b_z

    def j_xyz(self, x, y, z):
        '''Current density in Cartesian coordinates.'''

        from struphy.geometry.base import Domain

        x, y, z, is_sparse_meshgrid = Domain.prepare_eval_pts(x, y, z)

        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        z = z + 0*r # broadcasting happens here if sparse_meshgrid

        r2 = r[:, 0, :]
        z2 = z[:, 0, :]

        # J as 2-form (second component is already multiplied by r)
        # TODO: remove is_sparse_meshgrid (not necessary anymore)
        j2_1_tmp = - self.g_fun(r2, z2, 'z', is_sparse_meshgrid)
        j2_2_tmp = self.psi_fun(r2, z2, 'rr', is_sparse_meshgrid) + self.psi_fun(r2, z2, 'zz', is_sparse_meshgrid)
        j2_3_tmp = self.g_fun(r2, z2, 'r', is_sparse_meshgrid)

        if is_sparse_meshgrid:
            shp = (r.shape[0], phi.shape[1], z.shape[2])
        else:
            shp = r.shape

        j2_1 = np.empty(shp)
        j2_2 = np.empty(shp)
        j2_3 = np.empty(shp)

        j2_1[:] = j2_1_tmp[:, None, :]
        j2_2[:] = j2_2_tmp[:, None, :]
        j2_3[:] = j2_3_tmp[:, None, :]

        # push-forward of b2
        j_x = (np.cos(phi)*j2_1 - np.sin(phi)*j2_2) / r
        j_y = (np.sin(phi)*j2_1 + np.cos(phi)*j2_2) / r
        j_z = j2_3 / r

        return j_x, j_y, j_z

    def p_xyz(self, x, y, z):
        '''Pressure in Cartesian coordinates.'''

        from struphy.geometry.base import Domain

        x, y, z, is_sparse_meshgrid = Domain.prepare_eval_pts(x, y, z)

        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        z = z + 0*r # broadcasting happens here

        r2 = r[:, 0, :]
        z2 = z[:, 0, :]

        psi_tmp = self.psi_fun(r2, z2, None, is_sparse_meshgrid) 

        if is_sparse_meshgrid:
            shp = (r.shape[0], phi.shape[1], z.shape[2])
        else:
            shp = r.shape

        psi = np.empty(shp)

        psi[:] = psi_tmp[:, None, :]

        return self.pres_1d(psi.flatten()).reshape(psi.shape)

    def n_xyz(self, x, y, z):
        """ Equilibrium number density in physical space.
        """
        return 0*x + 0*y + 0*z

    ##################
    # Helper functions
    ##################

    @staticmethod
    def field_line_tracing(s_fun, x_at_axis, y_at_axis, xdim, Nel, p, theta='equal_angle'):
        '''Given a poloidal flux function s(x, y), computes a mapping (x, y) = F(s, theta).
        Three different theta parametrizations can be chosen: 'equal_angle', 'equal_arc_length' or 'sfl' (PEST).

        Parameters
        ----------
        s_fun : callable
            The normalized flux function s(x, y). The range of s must be [0, r] with r>=1.

        x_at_axis : float
            x-coordinate of the pole (magnetic axis).

        y_at_axis : float
            y-coordinate of the pole (magnetic axis).

        xdim : float
            Length of x-domain.

        Nel : list[int]
            Number of elements to be used for spline inerpolation.

        p : list[int]
            Spline degrees for spline interpolation.

        theta: str
            WHich theta parametrization to use: 'equal_angle', NOT YET: 'equal_arc_length' or 'sfl' (PEST)'''

        import numpy as np
        from struphy.b_splines import bsplines as bsp
        from scipy.optimize import newton
        from struphy.geometry.base import spline_interpolation_nd

        assert callable(s_fun)
        assert len(Nel) == 2
        assert len(p) == 2

        # avoid theta = 0.5 angle
        while True:
            el_b = [np.linspace(0., 1., Nel + 1) for Nel in Nel]

            spl_kind = [False, True]
            T = [bsp.make_knots(el_b, p, kind)
                 for el_b, p, kind in zip(el_b, p, spl_kind)]

            s_grev, th_grev = [bsp.greville(T, p, kind)
                               for T, p, kind in zip(T, p, spl_kind)]

            if (not np.any(np.abs(th_grev - .5) < 1e-14)):
                break
            else:
                Nel[1] += 1

        X = np.empty((s_grev.size, th_grev.size))
        Y = np.empty_like(X)
        for j, thj in enumerate(th_grev):
            def xj(r): return x_at_axis + r*np.cos(2*np.pi*thj)
            def yj(r): return y_at_axis + r*np.sin(2*np.pi*thj)
            ri = 0.
            for i, si in enumerate(s_grev):
                if i > 0:
                    if i == 1:
                        ri = xdim/20.

                    def f(r): return s_fun(xj(r), yj(r)) - si
                    ri = newton(f, ri)
                X[i, j] = xj(ri)
                Y[i, j] = yj(ri)

        cx, knots, indN = spline_interpolation_nd(
            p, spl_kind, [s_grev, th_grev], X)
        cy, knots, indN = spline_interpolation_nd(
            p, spl_kind, [s_grev, th_grev], Y)

        return cx, cy

    def psi_fun(self, r, z, der=None, is_sparse_meshgrid=False):
        '''Interpolated flux function psi(r, z), and its first derivatives. 
        
        Parameters
        ----------
        r, z : array
            Must stem from meshgrid.

        der : str
            Which derivative to evaluate (None, 'r', 'z', 'rr' or 'zz').

        is_sparse_meshgrid : bool
            Refers to the shapes of r, z.

        Returns
        -------
            A 2d array of dense meshgrid shape.
        '''

        from struphy.b_splines.bspline_evaluation_2d import evaluate_matrix, evaluate_sparse

        if is_sparse_meshgrid:
            _func = evaluate_sparse
            out = np.zeros((r.shape[0], z.shape[1]))
        else:
            _func = evaluate_matrix
            out = np.zeros(r.shape)

        if der is None:
            _func(*self._Tpsi, *self.params['p_for_psi'], *self._indNpsi, self._cpsi, (
                r - self._rmin) / self._rdim, (z - self._zmin) / self._zdim, out, 0)
            fac = 1. 
        elif der == 'r':
            _func(*self._Tpsi, *self.params['p_for_psi'], *self._indNpsi, self._cpsi, (
                r - self._rmin) / self._rdim, (z - self._zmin) / self._zdim, out, 31) 
            fac = 1 / self._rdim
        elif der == 'z':
            _func(*self._Tpsi, *self.params['p_for_psi'], *self._indNpsi, self._cpsi, (
                r - self._rmin) / self._rdim, (z - self._zmin) / self._zdim, out, 32)
            fac = 1 / self._zdim
        elif der == 'rr':
            _func(*self._Tpsi, *self.params['p_for_psi'], *self._indNpsi, self._cpsi, (
                r - self._rmin) / self._rdim, (z - self._zmin) / self._zdim, out, 41)
            fac = 1 / self._rdim**2
        elif der == 'zz':
            _func(*self._Tpsi, *self.params['p_for_psi'], *self._indNpsi, self._cpsi, (
                r - self._rmin) / self._rdim, (z - self._zmin) / self._zdim, out, 42)
            fac = 1 / self._zdim**2
        else:
            raise ValueError(f'der = {der} not implemented.')

        return out * fac

    def g_1d(self, psi, der=None):
        '''Toroidal field function g(psi). Argument must be 1d array.'''

        from struphy.b_splines.bspline_evaluation_1d import evaluate_vector

        assert psi.ndim == 1
        out = np.zeros(psi.shape)

        if der is None:
            kind = 0
            fac = 1.
        elif der == 1:
            kind = 2
            fac = 1 / self._psidim

        evaluate_vector(self._Tg[0], self.params['p_for_psi'][0], self._indNg[0], self._cg,
                        (psi - self._psimin) / self._psidim, out, kind)

        return out * fac

    def pres_1d(self, psi, der=None):
        '''Pressure profile p(psi). Argument must be 1d array.'''

        from struphy.b_splines.bspline_evaluation_1d import evaluate_vector

        assert psi.ndim == 1
        out = np.zeros(psi.shape)

        if der is None:
            kind = 0
            fac = 1.
        elif der == 1:
            kind = 2
            fac = 1 / self._psidim

        evaluate_vector(self._Tpres[0], self.params['p_for_psi'][0], self._indNpres[0], self._cpres,
                        (psi - self._psimin) / self._psidim, out, kind)

        return out * fac

    def q_1d(self, psi, der=None):
        '''Safety factor q(psi). Argument must be 1d array.'''

        from struphy.b_splines.bspline_evaluation_1d import evaluate_vector

        assert psi.ndim == 1
        out = np.zeros(psi.shape)

        if der is None:
            kind = 0
            fac = 1.
        elif der == 1:
            kind = 2
            fac = 1 / self._psidim

        evaluate_vector(self._Tg[0], self.params['p_for_psi'][0], self._indNg[0], self._cq,
                        (psi - self._psimin) / self._psidim, out, kind)

        return out * fac

    def g_fun(self, r, z, der=None, is_sparse_meshgrid=False):
        '''Toroidal field function g(psi(r, z)). Arguments must stem from meshgrid.'''

        if der is None:
            out = self.g_1d(self.psi_fun(r, z, None, is_sparse_meshgrid).flatten()).reshape(r.shape)
        elif der == 'r':
            psi_r = self.psi_fun(r, z, 'r', is_sparse_meshgrid)
            out = self.g_1d(self.psi_fun(r, z, None, is_sparse_meshgrid).flatten(), der=1).reshape(r.shape) * psi_r
        elif der == 'z':
            psi_z = self.psi_fun(r, z, 'z', is_sparse_meshgrid)
            out = self.g_1d(self.psi_fun(r, z, None, is_sparse_meshgrid).flatten(), der=1).reshape(r.shape) * psi_z

        return out


class GVECequilibrium(LogicalMHDequilibrium):
    '''Interface to `gvec_to_python <https://gitlab.mpcdf.mpg.de/spossann/gvec_to_python>`_.

    Parameters
    ----------
    params: dict
        Parameters that characterize the MHD equilibrium.

        * rel_path : str
            Whether dat_file (json_file) are relative to "<struphy_path>/fields_background/mhd_equil/gvec", or are absolute paths.
        * dat_file : str
            Path to .dat file.    
        * json_file : str
            Path to .json file.
        * use_pest : bool
            Whether to use straigh-field line coordinates (PEST).
        * use_nfp : bool
            Whether the field periods of the stellarator should be used in the mapping, i.e. phi = 2*pi*eta3 / nfp (piece of cake).
        * Nel : tuple[int]
            Number of cells in each direction used for interpolation of the mapping.   
        * p : tuple[int]
            Spline degree in each direction used for interpolation of the mapping.
    '''

    def __init__(self, params=None):

        from struphy.geometry.base import interp_mapping
        from struphy.geometry.domains import Spline

        from gvec_to_python.reader.gvec_reader import create_GVEC_json
        from gvec_to_python import GVEC

        import struphy

        # set default parameters
        if params is None:
            params = {'rel_path': True,
                      'dat_file': '/ellipstell_v2/newBC_E1D6_M6N6/GVEC_ELLIPSTELL_V2_State_0000_00200000.dat',
                      'json_file': None,
                      'use_pest': False,
                      'use_nfp': True,
                      'Nel': (16, 16, 16),
                      'p': (3, 3, 3), }
        # or check if given parameter dictionary is complete
        else:
            assert 'rel_path' in params
            assert 'dat_file' in params
            assert 'json_file' in params
            assert 'use_pest' in params
            assert 'use_nfp' in params
            assert 'Nel' in params
            assert 'p' in params

        if params['dat_file'] is None:

            assert params['json_file'] is not None
            assert params['json_file'][-5:] == '.json'

            if params['rel_path']:
                json_file = struphy.__path__[
                    0] + '/fields_background/mhd_equil/gvec' + params['json_file']
            else:
                json_file = params['json_file']

        else:

            assert params['dat_file'][-4:] == '.dat'

            if params['rel_path']:
                dat_file = struphy.__path__[
                    0] + '/fields_background/mhd_equil/gvec' + params['dat_file']
            else:
                dat_file = params['dat_file']

            json_file = dat_file[:-4] + '.json'
            create_GVEC_json(dat_file, json_file)

        if params['use_pest']:
            mapping = 'unit_pest'
        else:
            mapping = 'unit'

        if params['use_nfp']:
            unit_tor_domain = "one-fp"
            spl_kind = (False, True, False)
        else:
            unit_tor_domain = "full"
            spl_kind = (False, True, True)

        # gvec object
        self._gvec = GVEC(json_file, mapping=mapping,
                          unit_tor_domain=unit_tor_domain, use_pyccel=True)

        # project mapping to splines
        def X(e1, e2, e3): return self.gvec.f(e1, e2, e3)[0]
        def Y(e1, e2, e3): return self.gvec.f(e1, e2, e3)[1]
        def Z(e1, e2, e3): return self.gvec.f(e1, e2, e3)[2]

        cx, cy, cz = interp_mapping(
            params['Nel'], params['p'], spl_kind, X, Y, Z)

        # struphy domain object
        params_map = {'cx': cx, 'cy': cy, 'cz': cz,
                      'Nel': params['Nel'], 'p': params['p'], 'spl_kind': spl_kind}
        self._domain = Spline(params_map)

        self._params = params

    @property
    def domain(self):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        return self._domain

    @property
    def gvec(self):
        """ GVEC object.
        """
        return self._gvec

    @property
    def params(self):
        '''Parameters describing the equilibrium.'''
        return self._params

    def b2(self, eta1, eta2, eta3, squeeze_out=True):
        """2-form magnetic field on logical cube [0, 1]^3.
        """
        return self.gvec.b2(eta1, eta2, eta3)

    def j2(self, eta1, eta2, eta3, squeeze_out=True):
        """2-form current density (=curl B) on logical cube [0, 1]^3.
        """
        return self.gvec.j2(eta1, eta2, eta3)

    def p0(self, eta1, eta2, eta3, squeeze_out=True):
        """0-form equilibrium pressure on logical cube [0, 1]^3.
        """
        return self.gvec.p0(eta1, eta2, eta3)

    def n0(self, eta1, eta2, eta3, squeeze_out=True):
        """0-form equilibrium density on logical cube [0, 1]^3.
        """
        # TODO: which density to set?
        return self.gvec.p0(eta1, eta2, eta3) * 0
