import numpy as np

from struphy.fields_background.mhd_equil.base import CartesianMHDequilibrium, LogicalMHDequilibrium, AxisymmMHDequilibrium


class HomogenSlab(CartesianMHDequilibrium):
    r"""
    Homogeneous MHD equilibrium in slab geometry.

    .. math::

        \mathbf B &= B_{0x}\,\mathbf e_x + B_{0y}\,\mathbf e_y + B_{0z}\,\mathbf e_z = const.\,,

        p &= \beta \frac{|\mathbf B|^2}{2}=const.\,,

        n &= n_0 = const.\,.

    Parameters
    ----------
    **params
        Parameters that characterize the MHD equilibrium. Possible keys are

            * B0x  : magnetic field in x-direction
            * B0y  : magnetic field in y-direction
            * B0z  : magnetic field in z-direction
            * beta : plasma beta in % (ratio kinetic/magnetic pressure)
            * n0   : number density            
    """

    def __init__(self, **params):

        params_default = {'B0x': 0.,
                          'B0y': 0.,
                          'B0z': 1.,
                          'beta': 100.,
                          'n0': 1.}

        self._params = set_defaults(params, params_default)

    @property
    def params(self):
        """ Parameters dictionary.
        """
        return self._params

    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================

    # equilibrium magnetic field
    def b_xyz(self, x, y, z):
        """ Magnetic field.
        """
        bx = self.params['B0x'] - 0*x
        by = self.params['B0y'] - 0*x
        bz = self.params['B0z'] - 0*x

        return bx, by, bz

    # equilibrium current (curl of equilibrium magnetic field)
    def j_xyz(self, x, y, z):
        """ Current density.
        """
        jx = 0*x
        jy = 0*x
        jz = 0*x

        return jx, jy, jz

    # equilibrium pressure
    def p_xyz(self, x, y, z):
        """ Pressure.
        """
        pp = self.params['beta']/200*(self.params['B0x']**2 +
                                      self.params['B0y']**2 + self.params['B0z']**2) - 0*x

        return pp

    # equilibrium number density
    def n_xyz(self, x, y, z):
        """ Number density.
        """
        nn = self.params['n0'] - 0*x

        return nn


class ShearedSlab(CartesianMHDequilibrium):
    r"""
    Sheared slab MHD equilibrium in a cube with side lengths :math:`L_x=a,\,L_y=2\pi a,\,L_z=2\pi R_0`. Profiles depend on :math:`x` solely. 

    .. math::

        \mathbf B(x) &= B_{0} \left( \mathbf e_z + \frac{a}{q(x)R_0}\mathbf e_y\right)\,,\qquad q(x) = q_0 + ( q_1 - q_0 )\frac{x^2}{a^2}\,,

        p(x) &= \beta\frac{B_{0}^2}{2} \left( 1 + \frac{a^2}{q(x)^2 R_0^2} \right) + B_{0}^2 \frac{a^2}{R_0^2} \left( \frac{1}{q_0^2} - \frac{1}{q(x)^2} \right)\,,

        n(x) &= n_a + ( 1 - n_a ) \left( 1 - \left(\frac{x}{a}\right)^{n_1} \right)^{n_2} \,.

    Parameters
    ----------
    **params
        Parameters that characterize the MHD equilibrium. Possible keys are

            * a    : "minor" radius (Lx = a, Ly = 2*pi*a)
            * R0   : "major" radius (Lz = 2*pi*R0)
            * B0   : magnetic field in z-direction
            * q0   : safety factor at x=0
            * q1   : safety factor at x=a
            * n1   : 1st shape factor for number density profile 
            * n2   : 2nd shape factor for number density profile 
            * na   : number density at x=a
            * beta : plasma beta in % at x=0 (ratio of kinetic pressure to magnetic pressure)            
    """

    def __init__(self, **params):

        params_default = {'a': 1.,
                          'R0': 3.,
                          'B0': 1.,
                          'q0': 1.05,
                          'q1': 1.80,
                          'n1': 0.,
                          'n2': 0.,
                          'na': 1.,
                          'beta': 10.}

        self._params = set_defaults(params, params_default)

    @property
    def params(self):
        """ Parameters dictionary.
        """
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
        """ Magnetic field.
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
        """ Current density.
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
        """ Pressure.
        """
        pp = self.px(x)

        return pp

    # equilibrium number density
    def n_xyz(self, x, y, z):
        """ Number density.
        """
        nn = self.nx(x)

        return nn


class ScrewPinch(CartesianMHDequilibrium):
    r"""
    Straight tokamak (screw pinch) MHD equilibrium for a cylinder or radius :math:`a` and length :math:`L_z=2\pi R_0`.

    The profiles in cylindrical coordinates :math:`(r, \theta, z)` with transformation formulae 

    .. math::

        x &= r\cos(\theta)\,,

        y &= r\sin(\theta)\,,

    are:

    .. math::

        \mathbf B(r) &= B_{0}\left( \mathbf e_z + \frac{r}{q(r) R_0}\mathbf e_\theta \right)\,,\qquad q(r) = q_0 + ( q_1 - q_0 )\frac{r^2}{a^2}\,,

        p(r) &= \left\{\begin{aligned}
        &\frac{B_{0}^2 a^2 q_0}{ 2 R_0^2(q_1 - q_0) } \left( \frac{1}{q(r)^2} - \frac{1}{q_1^2} \right) \quad &&\textnormal{if}\quad q_1\neq q_0\,, 

        &\beta\frac{B_{0}^2}{2} \quad &&\textnormal{else}\,,
        \end{aligned}\right.

        n(r) &= n_a + ( 1 - n_a )\left( 1 - \left(\frac{r}{a}\right)^{n_1} \right)^{n_2}\,.

    Parameters
    ----------
    **params
        Parameters that characterize the MHD equilibrium. Possible keys are

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

    def __init__(self, **params):

        params_default = {'a': 1.,
                          'R0': 5.,
                          'B0': 1.,
                          'q0': 1.05,
                          'q1': 1.80,
                          'n1': 0.,
                          'n2': 0.,
                          'na': 1.,
                          'beta': 10.}

        self._params = set_defaults(params, params_default)

        # inverse cylindrical coordinate transformation (x, y, z) --> (r, theta, phi)
        self.r = lambda x, y, z: np.sqrt(x**2 + y**2)
        self.theta = lambda x, y, z: np.arctan2(y, x)
        self.z = lambda x, y, z: 1*z

    @property
    def params(self):
        """ Parameters dictionary.
        """
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
        """ Magnetic field.
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
        """ Current density.
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
        """ Pressure.
        """
        pp = self.pr(self.r(x, y, z))

        return pp

    # equilibrium number density
    def n_xyz(self, x, y, z):
        """ Number density.
        """
        nn = self.nr(self.r(x, y, z))

        return nn


class AdhocTorus(AxisymmMHDequilibrium):
    r"""
    Ad hoc tokamak MHD equilibrium with circular concentric flux surfaces.

    For a cylindrical coordinate system :math:`(R, \phi, Z)` with transformation formulae

    .. math::

        x &= R\cos(\phi)\,,     &&R = \sqrt{x^2 + y^2}\,,

        y &= R\sin(\phi)\,,  &&\phi = \arctan(y/x)\,,

        z &= Z\,,               &&Z = z\,,

    the magnetic field is given by

    .. math::

        \mathbf B = \nabla\psi\times\nabla\phi+g\nabla\phi\,,

    where :math:`g=g(R, Z)=-B_0R_0=const.` is the toroidal field function, :math:`R_0` the major radius of the torus and :math:`B_0` the on-axis magnetic field. The ad hoc poloidal flux function :math:`\psi=\psi(r)` is the solution of

    .. math::

        \frac{\textnormal{d}\psi}{\textnormal{d}r}=\frac{B_0r}{q(r)\sqrt{1 - r^2/R_0^2}}\,,\qquad r=\sqrt{Z^2+(R-R_0)^2}\,,

    for a parabolic safety factor profile :math:`q=q(r)=q_0 + ( q_1 - q_0 )r^2/a^2` (:math:`a` is the minor radius of the torus).

    The pressure and number density profiles are chosen as

    .. math::

        p(r) &= \left\{\begin{aligned}
        &\frac{B_{0}^2\, a^2 q_0}{ 2 R_0^2 (q_1 - q_0) } \left( \frac{1}{q(r)^2} - \frac{1}{q_1^2} \right) \quad &&\textnormal{if} \quad p_\textnormal{kind}=0 \quad \textnormal{and} \quad q_1\neq q_0\,, 

        &\beta \frac{B_{0}^2}{2} \quad &&\textnormal{if} \quad p_\textnormal{kind}=0 \quad \textnormal{and} \quad q_1= q_0 \,,

        &\beta \frac{B_{0}^2}{2} \left( 1 - p_1 \frac{r^2}{a^2} - p_2 \frac{r^4}{a^4} \right) \quad &&\textnormal{if} \quad p_\textnormal{kind}=1\,,
        \end{aligned}\right.

        n(r) &= n_a + ( 1 - n_a ) \left( 1 - \left(\frac{r}{a}\right)^{n_1} \right)^{n_2}\,.

    Parameters
    ----------
    **params
        Parameters that characterize the MHD equilibrium. Possible keys are

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

    def __init__(self, **params):

        # parameters
        params_default = {'a': 1.,
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

        self._params = set_defaults(params, params_default)

        # plasma boundary contour
        ths = np.linspace(0., 2*np.pi, 201)

        self._rbs = self.params['R0'] * \
            (1 + self.params['a']/self.params['R0']*np.cos(ths))
        self._zbs = self.params['a']*np.sin(ths)

        # set on-axis and boundary fluxes
        self._psi0 = self.psi(self.params['R0'], 0.)
        self._psi1 = self.psi(self.params['R0'] + self.params['a'], 0.)

    @property
    def params(self):
        """ Parameters dictionary.
        """
        return self._params

    @property
    def boundary_pts_R(self):
        """ R-coordinates of plasma boundary contour.
        """
        return self._rbs

    @property
    def boundary_pts_Z(self):
        """ Z-coordinates of plasma boundary contour.
        """
        return self._zbs

    # ===============================================================
    #           abstract properties
    # ===============================================================

    @property
    def psi_range(self):
        """ Psi on-axis and at plasma boundary.
        """
        return [self._psi0, self._psi1]

    @property
    def psi_axis_RZ(self):
        """ Location of magnetic axis in R-Z-coordinates.
        """
        return [self.params['R0'], 0.]

    # ===============================================================
    #           radial profiles for an ad hoc tokamak equilibrium
    # ===============================================================

    def psi_r(self, r, der=0):
        """ Ad hoc poloidal flux function psi = psi(r).
        """

        assert der >= 0 and der <= 2, 'Only first and second derivative available!'

        eps = self.params['a']/self.params['R0']

        q0 = self.params['q0']
        q1 = self.params['q1']
        dq = q1 - q0

        # geometric correction factor and its first derivative
        gf_0 = np.sqrt(1 - (r/self.params['R0'])**2)
        gf_1 = -r/(self.params['R0']**2*gf_0)

        # safety factors
        q_0 = self.q_r(r, der=0)
        q_1 = self.q_r(r, der=1)

        q_bar_0 = q_0*gf_0
        q_bar_1 = q_1*gf_0 + q_0*gf_1

        if der == 0:
            out = -self.params['B0']*self.params['a']**2 / \
                np.sqrt(dq*q0*eps**2 + dq**2)
            out *= np.arctanh(np.sqrt((dq - dq*(r/self.params['R0'])
                              ** 2)/(q0*eps**2 + dq)))
        elif der == 1:
            out = self.params['B0']*r/q_bar_0
        elif der == 2:
            out = self.params['B0']*(q_bar_0 - r*q_bar_1)/q_bar_0**2
        else:
            raise NotImplementedError(
                'Only first and second derivatives available')

        return out

    def q_r(self, r, der=0):
        """ Radial safety factor profile (and first derivative).
        """

        assert der >= 0 and der <= 1, 'Only first derivative available!'

        if der == 0:
            qout = self.params['q0'] + (self.params['q1'] -
                                        self.params['q0'])*(r/self.params['a'])**2
        else:
            qout = 2*(self.params['q1'] -
                      self.params['q0'])*r/self.params['a']**2

        return qout

    def p_r(self, r):
        """ Radial pressure profile p = p(r).
        """
        eps = self.params['a']/self.params['R0']

        if self.params['p_kind'] == 0:

            if self.params['q0'] == self.params['q1']:
                pout = self.params['B0']**2*self.params['beta']/200 - 0*r
            else:
                pout = self.params['B0']**2*eps**2*self.params['q0']/(
                    2*(self.params['q1'] - self.params['q0']))*(1/self.q_r(r)**2 - 1/self.params['q1']**2)

        else:

            pout = self.params['B0']**2*self.params['beta']/200*(
                1 - self.params['p1']*r**2/self.params['a']**2 - self.params['p2']*r**4/self.params['a']**4)

        return pout

    def n_r(self, r):
        """ Radial number density profile n = n(r).
        """
        nout = (1 - self.params['na'])*(1 - (r/self.params['a']) **
                                        self.params['n1'])**self.params['n2'] + self.params['na']

        return nout

    def plot_profiles(self, n_pts=501):
        """ Plots 1d profiles.
        """

        import matplotlib.pyplot as plt

        r = np.linspace(0., self.params['a'], n_pts)

        fig, ax = plt.subplots(2, 2)

        fig.set_figheight(5)
        fig.set_figwidth(6)

        ax[0, 0].plot(r, self.psi_r(r))
        ax[0, 0].set_xlabel('$r$')
        ax[0, 0].set_ylabel('$\psi$')

        ax[0, 1].plot(r, self.q_r(r))
        ax[0, 1].set_xlabel('$r$')
        ax[0, 1].set_ylabel('$q$')

        ax[1, 0].plot(r, self.p_r(r))
        ax[1, 0].set_xlabel('$r$')
        ax[1, 0].set_ylabel('$p$')

        ax[1, 1].plot(r, self.n_r(r))
        ax[1, 1].set_xlabel('$r$')
        ax[1, 1].set_ylabel('$n$')

        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        plt.show()

    # ===============================================================
    #           abstract methods
    # ===============================================================

    def psi(self, R, Z, dR=0, dZ=0):
        """ Poloidal flux function psi = psi(R, Z).
        """

        r = np.sqrt(Z**2 + (R - self.params['R0'])**2)

        if dR == 0 and dZ == 0:
            out = self.psi_r(r, der=0)
        else:

            dr_dR = (R - self.params['R0'])/r
            dr_dZ = Z/r

            d2r_dR2 = (r - (R - self.params['R0'])*dr_dR)/r**2
            d2r_dZ2 = (r - Z*dr_dZ)/r**2

            if dR == 1 and dZ == 0:
                out = self.psi_r(r, der=1) * dr_dR
            elif dR == 0 and dZ == 1:
                out = self.psi_r(r, der=1) * dr_dZ
            elif dR == 2 and dZ == 0:
                out = self.psi_r(r, der=2) * dr_dR**2 + \
                    self.psi_r(r, der=1) * d2r_dR2
            elif dR == 0 and dZ == 2:
                out = self.psi_r(r, der=2) * dr_dZ**2 + \
                    self.psi_r(r, der=1) * d2r_dZ2
            else:
                raise NotImplementedError(
                    'Only combinations (dR=0, dZ=0), (dR=1, dZ=0), (dR=0, dZ=1), (dR=2, dZ=0) and (dR=0, dZ=2) possible!')

        return out

    def g_tor(self, R, Z, dR=0, dZ=0):
        """ Toroidal field function g = g(R, Z).
        """

        if dR == 0 and dZ == 0:
            out = -self._params['B0']*self._params['R0'] - 0*R
        elif dR == 1 and dZ == 0:
            out = 0*R
        elif dR == 0 and dZ == 1:
            out = 0*Z
        else:
            raise NotImplementedError(
                'Only combinations (dR=0, dZ=0), (dR=1, dZ=0) and (dR=0, dZ=1) possible!')

        return out

    def p_xyz(self, x, y, z):
        """ Pressure p = p(x, y, z).
        """
        r = np.sqrt((np.sqrt(x**2 + y**2) - self._params['R0'])**2 + z**2)

        pp = self.p_r(r)

        return pp

    def n_xyz(self, x, y, z):
        """ Number density n = n(x, y, z).
        """
        r = np.sqrt((np.sqrt(x**2 + y**2) - self._params['R0'])**2 + z**2)

        nn = self.n_r(r)

        return nn


class AdhocTorusQPsi(AxisymmMHDequilibrium):
    r"""
    Ad hoc tokamak MHD equilibrium with circular concentric flux surfaces.

    For a cylindrical coordinate system :math:`(R, \phi, Z)` with transformation formulae

    .. math::

        x &= R\cos(\phi)\,,     &&R = \sqrt{x^2 + y^2}\,,

        y &= R\sin(\phi)\,,  &&\phi = \arctan(y/x)\,,

        z &= Z\,,               &&Z = z\,,

    the magnetic field is given by

    .. math::

        \mathbf B = \nabla\psi\times\nabla\phi+g\nabla\phi\,,

    where :math:`g=g(R, Z)=-B_0R_0=const.` is the toroidal field function, :math:`R_0` the major radius of the torus and :math:`B_0` the on-axis magnetic field. The ad hoc poloidal flux function :math:`\psi=\psi(r)` is the solution of

    .. math::

        \frac{\textnormal{d}\psi}{\textnormal{d}r}=\frac{B_0r}{q(\psi(r))\sqrt{1 - r^2/R_0^2}}\,,\qquad r=\sqrt{Z^2+(R-R_0)^2}\,,

    for a safety factor profile

    .. math::

        q(\psi) &= q_0 + \psi_{\textnormal{norm}}\left[ q_1-q_0+(q_1^\prime-q_1+q_0)\frac{(1-\psi_s)(\psi_{\textnormal{norm}}-1)}{\psi_{\textnormal{norm}}-\psi_s} \right]\,,

        \psi_{\textnormal{norm}} &= \frac{\psi-\psi(0)}{\psi(a)-\psi(0)}\,,

        \psi_s &= (q_1^\prime-q_1+q_0)/(q_0^\prime+q_1^\prime-2q_1+2q_0)\,,

    where :math:`a` is the minor radius of the torus.

    The pressure and number density profiles are chosen as

    .. math::

        p(\psi) &= \frac{\beta B_0^2}{2}\exp\left(-\frac{\psi_{\textnormal{norm}}}{p_1}\right)\,,

        n(\psi) &= n_a + ( 1 - n_a ) \left( 1 - \psi_{\textnormal{norm}}^{n_1} \right)^{n_2}\,.

    Parameters
    ----------
    **params
        Parameters that characterize the MHD equilibrium. Possible keys are

            * a       : minor radius of torus
            * R0      : major radius of torus
            * B0      : on-axis toroidal magnetic field
            * q0      : safety factor at r=0
            * q1      : safety factor at r=a
            * q0p     : derivative of safety factor at r=0 (w.r.t. poloidal flux function)
            * q1p     : derivative of safety factor at r=a (w.r.t. poloidal flux function)
            * n1      : 1st shape factor for number density profile 
            * n2      : 2nd shape factor for number density profile 
            * na      : number density at r=a
            * beta    : on-axis plasma beta in % (ratio of kinetic pressure to magnetic pressure)   
            * p1      : shape factor of pressure profile
            * psi_k   : spline degree to be used for interpolation of poloidal flux function
            * psi_nel : number of cells to be used for interpolation of poloidal flux function
    """

    def __init__(self, **params):

        from scipy.optimize import fsolve
        from scipy.integrate import odeint
        from scipy.interpolate import UnivariateSpline

        # parameters
        params_default = {'a': 0.361925,
                          'R0': 1.,
                          'B0': 1.,
                          'q0': 0.6,
                          'q1': 2.5,
                          'q0p': 0.78,
                          'q1p': 5.00,
                          'n1': 0.,
                          'n2': 0.,
                          'na': 1.,
                          'beta': 4.,
                          'p1': 0.25,
                          'psi_k': 3,
                          'psi_nel': 50}

        self._params = set_defaults(params, params_default)

        # plasma boundary contour
        ths = np.linspace(0., 2*np.pi, 201)

        self._rbs = self.params['R0'] * \
            (1 + self.params['a']/self.params['R0']*np.cos(ths))
        self._zbs = self.params['a']*np.sin(ths)

        # on-axis flux (arbitrary value)
        self._psi0 = -10.

        # poloidal flux function differential equation: dpsi_dr(r) = B0*r/(q(psi(r))*sqrt(1 - r**2/R0**2))
        def dpsi_dr(psi, r, psi1):

            q0 = self.params['q0']
            q1 = self.params['q1']

            q0p = self.params['q0p']
            q1p = self.params['q1p']

            B0 = self.params['B0']
            R0 = self.params['R0']

            psi_norm = (psi - self._psi0)/(psi1 - self._psi0)
            psi_s = (q1p - q1 + q0)/(q0p + q1p - 2*q1 + 2*q0)

            q = q0 + psi_norm*(q1 - q0 + (q1p - q1 + q0) *
                               (1 - psi_s)*(psi_norm - 1)/(psi_norm - psi_s))

            out = B0*r/(q*np.sqrt(1 - r**2/R0**2))

            return out

        # solve differential equation and fix boundary flux
        r_i = np.linspace(0., self.params['a'], self.params['psi_nel'] + 1)

        def fun(psi1):

            out = odeint(dpsi_dr, self._psi0, r_i, args=(psi1,)).flatten()

            return out[-1] - psi1

        self._psi1 = fsolve(fun, -9.5)[0]

        # interpolate flux function
        self._psi_i = UnivariateSpline(r_i, odeint(dpsi_dr, self._psi0, r_i, args=(self._psi1,)).flatten(),
                                       k=self.params['psi_k'], s=0., ext=3)

    @property
    def params(self):
        """ Parameters dictionary describing the equilibrium.
        """
        return self._params

    @property
    def boundary_pts_R(self):
        """ R-coordinates of plasma boundary contour.
        """
        return self._rbs

    @property
    def boundary_pts_Z(self):
        """ Z-coordinates of plasma boundary contour.
        """
        return self._zbs

    # ===============================================================
    #           abstract properties
    # ===============================================================

    @property
    def psi_range(self):
        """ Psi on-axis and at plasma boundary.
        """
        return [self._psi0, self._psi1]

    @property
    def psi_axis_RZ(self):
        """ Location of magnetic axis in R-Z-coordinates.
        """
        return [self.params['R0'], 0.]

    # ===============================================================
    #       1d profiles for an ad hoc tokamak equilibrium
    # ===============================================================

    def psi_r(self, r, der=0):
        """ Ad hoc poloidal flux function psi = psi(r).
        """

        assert der >= 0 and der <= 2, 'Only first and second derivatives available!'

        out = self._psi_i(r, nu=der)

        # remove all "dimensions" for point-wise evaluation
        if isinstance(r, (int, float)):
            assert out.ndim == 0
            out = out.item()

        return out

    def q_psi(self, psi):
        """ Safety factor profile q = q(psi).
        """

        q0 = self.params['q0']
        q1 = self.params['q1']

        q0p = self.params['q0p']
        q1p = self.params['q1p']

        psi_s = (q1p - q1 + q0)/(q0p + q1p - 2*q1 + 2*q0)

        psi_norm = (psi - self._psi0)/(self._psi1 - self._psi0)

        q = q0 + psi_norm*(q1 - q0 + (q1p - q1 + q0)*(1 - psi_s)
                           * (psi_norm - 1)/(psi_norm - psi_s))

        return q

    def p_psi(self, psi, der=0):
        """ Pressure profile p = p(psi).
        """

        assert der >= 0 and der <= 1, 'Only first derivative available!'

        beta, p1, B0 = self.params['beta'], self.params['p1'], self.params['B0']

        psi_norm = (psi - self._psi0)/(self._psi1 - self._psi0)

        if der == 0:
            out = self.params['beta'] * \
                self.params['B0']**2/200*np.exp(-psi_norm/p1)
        else:
            out = -self.params['beta']*self.params['B0']**2/200 * \
                np.exp(-psi_norm/p1)/(p1*(self._psi1 - self._psi0))

        return out

    def n_psi(self, psi, der=0):
        """ Number density profile n = n(psi).
        """

        assert der >= 0 and der <= 1, 'Only first derivative available!'

        n1, n2, na = self.params['n1'], self.params['n2'], self.params['na']

        psi_norm = (psi - self._psi0)/(self._psi1 - self._psi0)

        if der == 0:
            out = (1 - na)*(1 - psi_norm**n1)**n2 + na
        else:
            out = -(1 - na)*n1*n2/(self._psi1 - self._psi0) * \
                (1 - psi_norm**n1)**(n2 - 1)*psi_norm**(n1 - 1)

        return out

    def plot_profiles(self, n_pts=501):
        """ Plots 1d profiles.
        """

        import matplotlib.pyplot as plt

        r = np.linspace(0., self.params['a'], n_pts)
        psi = np.linspace(self._psi0, self._psi1, n_pts)

        fig, ax = plt.subplots(2, 2)

        fig.set_figheight(5)
        fig.set_figwidth(6)

        ax[0, 0].plot(r, self.psi_r(r))
        ax[0, 0].set_xlabel('$r$')
        ax[0, 0].set_ylabel('$\psi$')

        ax[0, 1].plot(psi, self.q_psi(psi))
        ax[0, 1].set_xlabel('$\psi$')
        ax[0, 1].set_ylabel('$q$')

        ax[1, 0].plot(psi, self.p_psi(psi))
        ax[1, 0].set_xlabel('$\psi$')
        ax[1, 0].set_ylabel('$p$')

        ax[1, 1].plot(psi, self.n_psi(psi))
        ax[1, 1].set_xlabel('$\psi$')
        ax[1, 1].set_ylabel('$n$')

        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        plt.show()

    # ===============================================================
    #           abstract methods
    # ===============================================================

    def psi(self, R, Z, dR=0, dZ=0):
        """ Poloidal flux function psi = psi(R, Z).
        """

        r = np.sqrt(Z**2 + (R - self.params['R0'])**2)

        if dR == 0 and dZ == 0:
            out = self.psi_r(r, der=0)
        else:

            dr_dR = (R - self.params['R0'])/r
            dr_dZ = Z/r

            d2r_dR2 = (r - (R - self.params['R0'])*dr_dR)/r**2
            d2r_dZ2 = (r - Z*dr_dZ)/r**2

            if dR == 1 and dZ == 0:
                out = self.psi_r(r, der=1) * dr_dR
            elif dR == 0 and dZ == 1:
                out = self.psi_r(r, der=1) * dr_dZ
            elif dR == 2 and dZ == 0:
                out = self.psi_r(r, der=2) * dr_dR**2 + \
                    self.psi_r(r, der=1) * d2r_dR2
            elif dR == 0 and dZ == 2:
                out = self.psi_r(r, der=2) * dr_dZ**2 + \
                    self.psi_r(r, der=1) * d2r_dZ2
            else:
                raise NotImplementedError(
                    'Only combinations (dR=0, dZ=0), (dR=1, dZ=0), (dR=0, dZ=1), (dR=2, dZ=0) and (dR=0, dZ=2) possible!')

        return out

    def g_tor(self, R, Z, dR=0, dZ=0):
        """ Toroidal field function g = g(R, Z).
        """

        if dR == 0 and dZ == 0:
            out = -self._params['B0']*self._params['R0'] - 0*R
        elif dR == 1 and dZ == 0:
            out = 0*R
        elif dR == 0 and dZ == 1:
            out = 0*Z
        else:
            raise NotImplementedError(
                'Only combinations (dR=0, dZ=0), (dR=1, dZ=0) and (dR=0, dZ=1) possible!')

        return out

    def p_xyz(self, x, y, z):
        """ Pressure p = p(x, y, z).
        """
        r = np.sqrt((np.sqrt(x**2 + y**2) - self._params['R0'])**2 + z**2)

        return self.p_psi(self.psi_r(r))

    def n_xyz(self, x, y, z):
        """ Number density n = n(x, y, z).
        """
        r = np.sqrt((np.sqrt(x**2 + y**2) - self._params['R0'])**2 + z**2)

        return self.n_psi(self.psi_r(r))


class EQDSKequilibrium(AxisymmMHDequilibrium):
    """
    Interface to `EQDSK file format <https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf>`_.

    Parameters
    ----------
    **params
        Parameters that characterize the MHD equilibrium. Possible keys are

            * rel_path : str
                Whether file is relative to "<struphy_path>/fields_background/mhd_equil/gvec", or is an absolute path.
            * file : str
                Path to eqdsk file.
            * data_type : int
                0: there is no space between data, 1: there is space between data.
            * p_for_psi : list[int]
                Spline degrees in (R, Z) directions used for interpolation of psi data.
            * psi_resolution : list[float]
                Resolution of psi data in (R, Z) directions in %, e.g. [50., 50.] uses every second psi data point.
            * p_for_flux : int
                Spline degree in psi direction used for interpolation of 1d functions that depend on psi: f=f(psi).
            * flux_resolution : float
                Resolution of 1d f=f(psi) data in %, e.g. 25. uses every forth data point.
            * n1 : float
                1st shape factor for number density profile n = n(psi).
            * n2 : float
                2nd shape factor for number density profile n = n(psi).
            * na : float
                Number density at plasma boundary.
    """

    def __init__(self, **params):

        from scipy.interpolate import UnivariateSpline, RectBivariateSpline
        from scipy.optimize import minimize

        from struphy.fields_background.mhd_equil.eqdsk import readeqdsk

        import struphy

        params_default = {'rel_path': True,
                          'file': 'AUGNLED_g031213.00830.high',
                          'data_type': 0,
                          'p_for_psi': [3, 3],
                          'psi_resolution': [25., 6.25],
                          'p_for_flux': 3,
                          'flux_resolution': 50.,
                          'n1': 0.,
                          'n2': 0.,
                          'na': 1.,
                          }

        self._params = set_defaults(params, params_default)

        if self._params['rel_path']:
            _path = struphy.__path__[0] + \
                '/fields_background/mhd_equil/eqdsk/data/' + \
                self._params['file']
        else:
            _path = self._params['file']

        eqdsk = readeqdsk.Geqdsk()
        eqdsk.openFile(_path, data_type=self._params['data_type'])

        # Number of horizontal R grid points
        nR = eqdsk.data['nw'][0]
        # Number of vertical Z grid points
        nZ = eqdsk.data['nh'][0]
        # toroidal field function in m-T on flux grid, g = B^1_phi
        g_profile = eqdsk.data['fpol'][0]
        # plasma pressure in Nt/m^2 on uniform flux grid
        p_profile = eqdsk.data['pres'][0]
        # poloidal flux in Weber/rad on the rectangular grid points
        psi = eqdsk.data['psirz'][0].T
        # poloidal flux in Weber/rad at the plasma boundary
        psi_edge = eqdsk.data['sibry'][0]
        # q values on uniform flux grid from axis to boundary
        q_profile = eqdsk.data['qpsi'][0]
        # Horizontal dimension in meter of computational box
        rdim = eqdsk.data['rdim'][0]
        # Vertical dimension in meter of computational box
        zdim = eqdsk.data['zdim'][0]
        # Minimum R in meter of rectangular computational box
        rleft = eqdsk.data['rleft'][0]
        # Z of center of computational box in meter
        zmid = eqdsk.data['zmid'][0]
        # R of magnetic axis in meter
        R_at_axis = eqdsk.data['rmaxis'][0]
        # Z of magnetic axis in meter
        Z_at_axis = eqdsk.data['zmaxis'][0]
        # R of boundary points in meter
        self._rbs = eqdsk.data['rbbbs'][0]
        # Z of boundary points in meter
        self._zbs = eqdsk.data['zbbbs'][0]
        # R of limiter contour in meter
        self._rlims = eqdsk.data['rlim'][0]
        # Z of limiter contour in meter
        self._zlims = eqdsk.data['zlim'][0]

        assert g_profile.size == p_profile.size
        assert g_profile.size == q_profile.size
        assert psi.shape == (nR, nZ)

        # normalize pressure profile to pressure unit 1 Tesla/mu_0
        p_profile *= 1.25663706212e-6

        # spline interpolation of smoothed flux function
        self._r_range = [rleft, rleft + rdim]
        self._z_range = [zmid - zdim/2, zmid + zdim/2]

        R = np.linspace(self._r_range[0], self._r_range[1], nR)
        Z = np.linspace(self._z_range[0], self._z_range[1], nZ)

        smooth_steps = [int(1/(self._params['psi_resolution'][0]*0.01)),
                        int(1/(self._params['psi_resolution'][1]*0.01))]

        self._psi_i = RectBivariateSpline(R[::smooth_steps[0]], Z[::smooth_steps[1]], psi[::smooth_steps[0], ::smooth_steps[1]],
                                          kx=self._params['p_for_psi'][0], ky=self._params['p_for_psi'][1],
                                          s=0.)

        # find minimum of interpolated flux function (is not the same as (R_at_axis, Z_at_axis) and psi.min()!)
        self._psi_i_min = minimize(lambda x: self.psi(
            x[0], x[1]), x0=[R_at_axis, Z_at_axis])

        # set on-axis and boundary fluxes
        self._psi0 = self._psi_i_min['fun']
        self._psi1 = psi_edge

        # interpolate toroidal field function, pressure profile and q-profile on unifrom flux grid from axis to boundary
        flux_grid = np.linspace(self._psi0, self._psi1, g_profile.size)

        smooth_step = int(1/(self._params['flux_resolution']*0.01))

        self._g_i = UnivariateSpline(flux_grid[::smooth_step], g_profile[::smooth_step],
                                     k=self._params['p_for_flux'], s=0., ext=3)
        self._p_i = UnivariateSpline(flux_grid[::smooth_step], p_profile[::smooth_step],
                                     k=self._params['p_for_flux'], s=0., ext=3)
        self._q_i = UnivariateSpline(flux_grid[::smooth_step], q_profile[::smooth_step],
                                     k=self._params['p_for_flux'], s=0., ext=3)

    @property
    def params(self):
        """ Parameters describing the equilibrium.
        """
        return self._params

    @property
    def boundary_pts_R(self):
        """ R-coordinates of plasma boundary contour.
        """
        return self._rbs

    @property
    def boundary_pts_Z(self):
        """ Z-coordinates of plasma boundary contour.
        """
        return self._zbs

    @property
    def limiter_pts_R(self):
        """ R-coordinates of limiter contour.
        """
        return self._rlims

    @property
    def limiter_pts_Z(self):
        """ Z-coordinates of limiter contour.
        """
        return self._zlims

    @property
    def range_R(self):
        """ range of R of flux data.
        """
        return self._r_range

    @property
    def range_Z(self):
        """ range of Z of flux data.
        """
        return self._z_range

    # ===============================================================
    #           abstract properties
    # ===============================================================

    @property
    def psi_range(self):
        """ Psi on-axis and at plasma boundary.
        """
        return [self._psi0, self._psi1]

    @property
    def psi_axis_RZ(self):
        """ Location of magnetic axis in R-Z-coordinates.
        """
        return list(self._psi_i_min['x'])

    # ===============================================================
    #           1d flux function profiles f = f(psi)
    # ===============================================================

    def q_psi(self, psi, der=0):
        """ Safety factor q = q(psi).
        """
        out = self._q_i(psi, nu=der)

        # remove all "dimensions" for point-wise evaluation
        if isinstance(psi, (int, float)):
            assert out.ndim == 0
            out = out.item()

        return out

    def g_psi(self, psi, der=0):
        """ Toroidal field function g = g(psi).
        """
        out = self._g_i(psi, nu=der)

        # remove all "dimensions" for point-wise evaluation
        if isinstance(psi, (int, float)):
            assert out.ndim == 0
            out = out.item()

        return out

    def p_psi(self, psi, der=0):
        """ Pressure profile g = g(psi).
        """
        out = self._p_i(psi, nu=der)

        # remove all "dimensions" for point-wise evaluation
        if isinstance(psi, (int, float)):
            assert out.ndim == 0
            out = out.item()

        return out

    def n_psi(self, psi, der=0):
        """ Number density profile n = n(psi).
        """

        assert der >= 0 and der <= 1, 'Only first derivative available!'

        n1, n2, na = self._params['n1'], self._params['n2'], self._params['na']

        psi_norm = (psi - self._psi0)/(self._psi1 - self._psi0)

        if der == 0:
            out = (1 - na)*(1 - psi_norm**n1)**n2 + na
        else:
            out = -(1 - na)*n1*n2/(self._psi1 - self._psi0) * \
                (1 - psi_norm**n1)**(n2 - 1)*psi_norm**(n1 - 1)

        return out

    # ===============================================================
    #           abstract methods
    # ===============================================================

    def psi(self, R, Z, dR=0, dZ=0):
        """ Poloidal flux function psi = psi(R, Z).
        """

        is_float = all(isinstance(v, (int, float)) for v in [R, Z])

        out = self._psi_i(R, Z, dx=dR, dy=dZ, grid=False)

        # remove all "dimensions" for point-wise evaluation
        if is_float:
            assert out.ndim == 0
            out = out.item()

        return out

    def g_tor(self, R, Z, dR=0, dZ=0):
        """ Toroidal field function g = g(R, Z).
        """

        if dR == 0 and dZ == 0:
            out = self.g_psi(self.psi(R, Z, dR=0, dZ=0), der=0)
        elif dR == 1 and dZ == 0:
            out = self.g_psi(self.psi(R, Z, dR=0, dZ=0), der=1) * \
                self.psi(R, Z, dR=1, dZ=0)
        elif dR == 0 and dZ == 1:
            out = self.g_psi(self.psi(R, Z, dR=0, dZ=0), der=1) * \
                self.psi(R, Z, dR=0, dZ=1)

        return out

    def p_xyz(self, x, y, z):
        """ Pressure p = p(x, y, z).
        """

        R = np.sqrt(x**2 + y**2)
        Z = 1*z

        out = self.p_psi(self.psi(R, Z))

        return out

    def n_xyz(self, x, y, z):
        """ Number density in physical space.
        """

        R = np.sqrt(x**2 + y**2)
        Z = 1*z

        out = self.n_psi(self.psi(R, Z))

        return out


# class EQDSKequilibriumWithDomain(CartesianMHDequilibrium):
#    '''Interface to `EQDSK file format <https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf>`_.
#
#    Parameters
#    ----------
#    **params
#        Parameters that characterize the MHD equilibrium. Possible keys are
#
#        * rel_path : str
#            Whether file is relative to "<struphy_path>/fields_background/mhd_equil/gvec", or is an absolute path.
#        * file : str
#            Path to eqdsk file.
#        * data_type : int
#            0: there is no space between data, 1: there is space between data.
#        * p_for_psi : list[int]
#            Spline degree in each direction used for interpolation of psi data.
#        * Nel : tuple[int]
#            Number of cells in each direction used for field line tracing.
#        * p : tuple[int]
#            Spline degree in each direction used for field line tracing.
#        * theta : str
#            Choose theta parametrization: 'equal_angle', NOT YET: 'equal_arc_length' or 'sfl' (PEST).
#        * tor_period : int
#            Toroidal periodicity built into the mapping: phi = 2*pi * eta3 / tor_period.
#    '''
#
#    def __init__(self, show=False, **params):
#
#        from struphy.geometry.base import spline_interpolation_nd, PoloidalSplineTorus
#        from struphy.b_splines.bspline_evaluation_2d import evaluate
#        from struphy.fields_background.mhd_equil.eqdsk import readeqdsk
#        import struphy
#        import numpy as np
#        from matplotlib import pyplot as plt
#
#        params_default = {'rel_path': True,
#                          'file': 'AUGNLED_g031213.00830.high',
#                          'data_type': 0,
#                          'flux_resolution': 16,
#                          'psi_resolution': [32, 32],
#                          'p_for_psi': [3, 3],
#                          'Nel': [16, 32],
#                          'p': [3, 3],
#                          'theta': 'equal_angle',
#                          'tor_period': 3
#                          }
#
#        self._params = set_defaults(params, params_default)
#
#        if self._params['rel_path']:
#            _path = struphy.__path__[0] + \
#                '/fields_background/mhd_equil/eqdsk/data/' + \
#                self._params['file']
#        else:
#            _path = self._params['file']
#
#        eqdsk = readeqdsk.Geqdsk()
#        eqdsk.openFile(_path, data_type=self._params['data_type'])
#
#        # Number of horizontal R grid points
#        nR = eqdsk.data['nw'][0]
#        # Number of vertical Z grid points
#        nZ = eqdsk.data['nh'][0]
#        # toroidal field function in m-T on flux grid, g = B^1_phi
#        g_profile = eqdsk.data['fpol'][0]
#        # plasma pressure in Nt/m^2 on uniform flux grid
#        pres_profile = eqdsk.data['pres'][0]
#        # poloidal flux in Weber/rad on the rectangular grid points
#        psi = eqdsk.data['psirz'][0].T
#        # poloidal flux in Weber/rad at the plasma boundary
#        psi_edge = eqdsk.data['sibry'][0]
#        # q values on uniform flux grid from axis to boundary
#        q_profile = eqdsk.data['qpsi'][0]
#        # Horizontal dimension in meter of computational box
#        self._rdim = eqdsk.data['rdim'][0]
#        # Vertical dimension in meter of computational box
#        self._zdim = eqdsk.data['zdim'][0]
#        # Minimum R in meter of rectangular computational box
#        rleft = eqdsk.data['rleft'][0]
#        # Z of center of computational box in meter
#        zmid = eqdsk.data['zmid'][0]
#        # R of magnetic axis in meter
#        R_at_axis = eqdsk.data['rmaxis'][0]
#        # Z of magnetic axis in meter
#        Z_at_axis = eqdsk.data['zmaxis'][0]
#
#        assert g_profile.size == pres_profile.size
#        assert g_profile.size == q_profile.size
#        assert psi.shape == (nR, nZ)
#
#        p = self._params['p_for_psi']
#
#        # interpolate toroidal field function and pressure profile from smoothed data
#        self._psimin = psi.min()
#        self._psidim = psi.max() - self._psimin
#
#        flux_grid = np.linspace(self._psimin, psi.max(), g_profile.size)
#
#        smoothing = g_profile.size // self._params['flux_resolution']
#        g_smoothed = g_profile[::smoothing]
#        pres_smoothed = pres_profile[::smoothing]
#        q_smoothed = q_profile[::smoothing]
#        flux_grid_smoothed = flux_grid[::smoothing]
#
#        i_grid = (flux_grid_smoothed - self._psimin) / self._psidim
#
#        self._cg, self._Tg, self._indNg = spline_interpolation_nd(
#            [p[0]], [False], [i_grid], g_smoothed)
#
#        self._cpres, self._Tpres, self._indNpres = spline_interpolation_nd(
#            [p[0]], [False], [i_grid], pres_smoothed)
#
#        self._cq, self._Tq, self._indNq = spline_interpolation_nd(
#            [p[0]], [False], [i_grid], q_smoothed)
#
#        # interpolate psi and s (= normalized psi) from smoothed point data
#        R = np.linspace(rleft, rleft + self._rdim, nR)
#        Z = np.linspace(zmid - self._zdim/2, zmid + self._zdim/2, nZ)
#
#        self._rmin = R.min()
#        self._zmin = Z.min()
#
#        R1 = (R - self._rmin) / self._rdim
#        Z1 = (Z - self._zmin) / self._zdim
#
#        s_mat = np.sqrt((psi - self._psimin)/(psi_edge - self._psimin))
#
#        smoothing_r = psi.shape[0] // self._params['psi_resolution'][0]
#        smoothing_z = psi.shape[1] // self._params['psi_resolution'][1]
#        s_mat_smoothed = s_mat[::smoothing_r, ::smoothing_z]
#        psi_smoothed = psi[::smoothing_r, ::smoothing_z]
#        R1_smoothed = R1[::smoothing_r]
#        Z1_smoothed = Z1[::smoothing_z]
#
#        cs, knots, indN = spline_interpolation_nd(
#            p, [False, False], [R1_smoothed, Z1_smoothed], s_mat_smoothed)
#
#        self._cpsi, self._Tpsi, self._indNpsi = spline_interpolation_nd(
#            p, [False, False], [R1_smoothed, Z1_smoothed], psi_smoothed)
#
#        def s_fun(r, z):
#            '''Single point evaluation of .'''
#            return evaluate(1, 1, *knots, *p,
#                            *indN, cs, (r - self._rmin) / self._rdim, (z - self._zmin) / self._zdim)
#
#        # do the field line tracing
#        cx, cy = self.field_line_tracing(
#            s_fun, R_at_axis, Z_at_axis, self._rdim, self._params['Nel'], self._params['p'], theta=self._params['theta'])
#
#        # remove round-off errors at magnetic axis
#        cx[0, :] = cx[0, 0]
#        cy[0, :] = cy[0, 0]
#
#        # Instantiate Torus domain
#        params_map = {'cx': cx,
#                      'cy': cy,
#                      'Nel': self._params['Nel'],
#                      'p': self._params['p'],
#                      'spl_kind': [False, True],
#                      }
#
#        params_map['tor_period'] = self._params['tor_period']
#        self._domain = PoloidalSplineTorus(**params_map)
#
#        # plot for testing
#        if show:
#            print('file: ', self._params['file'])
#
#            # for testing: compute finite difference derivatives from data points
#            dR = self._rdim / (nR - 1)
#            dZ = self._zdim / (nZ - 1)
#            dflux = flux_grid[1] - flux_grid[0]
#
#            psi_r = np.zeros_like(psi)
#            psi_r[1:-1, :] = (np.roll(psi, -1, axis=0)[1:-1, :] -
#                              np.roll(psi, 1, axis=0)[1:-1, :]) / (2*dR)
#            psi_r[0, :] = (psi[1, :] - psi[0, :]) / dR
#            psi_r[-1, :] = (psi[-1, :] - psi[-2, :]) / dR
#
#            psi_z = np.zeros_like(psi)
#            psi_z[:, 1:-1] = (np.roll(psi, -1, axis=1)[:, 1:-1] -
#                              np.roll(psi, 1, axis=1)[:, 1:-1]) / (2*dZ)
#            psi_z[:, 0] = (psi[:, 1] - psi[:, 0]) / dZ
#            psi_z[:, -1] = (psi[:, -1] - psi[:, -2]) / dZ
#
#            psi_rr = np.zeros_like(psi)
#            psi_rr[1:-1, :] = (np.roll(psi, -1, axis=0)[1:-1] -
#                               2*psi[1:-1] + np.roll(psi, 1, axis=0)[1:-1]) / dR**2
#
#            psi_zz = np.zeros_like(psi)
#            psi_zz[:, 1:-1] = (np.roll(psi, -1, axis=1)[:, 1:-1] - 2 *
#                               psi[:, 1:-1] + np.roll(psi, 1, axis=1)[:, 1:-1]) / dZ**2
#
#            g_psi = np.zeros_like(g_profile)
#            g_psi[1:-1] = (np.roll(g_profile, -1)[1:-1] -
#                           np.roll(g_profile, 1)[1:-1]) / (2*dflux)
#            g_psi[0] = (g_profile[1] - g_profile[0]) / dflux
#            g_psi[-1] = (g_profile[-1] - g_profile[-2]) / dflux
#
#            pres_psi = np.zeros_like(pres_profile)
#            pres_psi[1:-1] = (np.roll(pres_profile, -1)[1:-1] -
#                              np.roll(pres_profile, 1)[1:-1]) / (2*dflux)
#            pres_psi[0] = (pres_profile[1] - pres_profile[0]) / dflux
#            pres_psi[-1] = (pres_profile[-1] - pres_profile[-2]) / dflux
#
#            q_psi = np.zeros_like(q_profile)
#            q_psi[1:-1] = (np.roll(q_profile, -1)[1:-1] -
#                           np.roll(q_profile, 1)[1:-1]) / (2*dflux)
#            q_psi[0] = (q_profile[1] - q_profile[0]) / dflux
#            q_psi[-1] = (q_profile[-1] - q_profile[-2]) / dflux
#
#            boundary_ind = np.argmin(np.abs(flux_grid - psi_edge))
#
#            plt.figure(figsize=(13, 8))
#            plt.subplot(2, 2, 1)
#            plt.plot(flux_grid[:boundary_ind], g_profile[:boundary_ind],
#                     'b', label='point data, size=' + str(g_profile.size))
#            plt.plot(flux_grid[:boundary_ind], self.g_1d(flux_grid[:boundary_ind]),
#                     'r--', label='smoothed with pts=' + str(flux_grid_smoothed.size))
#            #plt.plot([psi_edge, psi_edge], [g_profile.min(), g_profile.max()], 'k--', label='plasma boundary')
#            plt.legend()
#            plt.xlim(self._psimin, psi_edge)
#            plt.xlabel('$\psi$')
#            plt.ylabel('g [m-T]')
#            plt.title('toroidal field function g')
#            plt.subplot(2, 2, 2)
#            plt.plot(flux_grid[:boundary_ind], pres_profile[:boundary_ind],
#                     'b', label='point data, size=' + str(pres_profile.size))
#            plt.plot(flux_grid[:boundary_ind], self.pres_1d(
#                flux_grid[:boundary_ind]), 'r--', label='smoothed with pts=' + str(flux_grid_smoothed.size))
#            plt.legend()
#            plt.xlim(self._psimin, psi_edge)
#            plt.xlabel('$\psi$')
#            plt.ylabel('p [Pascal]')
#            plt.title('pressure profile')
#            plt.subplot(2, 2, 3)
#            plt.plot(flux_grid[:boundary_ind], g_psi[:boundary_ind],
#                     'b', label='point data, size=' + str(g_psi.size))
#            plt.plot(flux_grid[:boundary_ind], self.g_1d(flux_grid[:boundary_ind], der=1),
#                     'r--', label='smoothed with pts=' + str(flux_grid_smoothed.size))
#            plt.legend()
#            plt.xlim(self._psimin, psi_edge)
#            plt.xlabel('$\psi$')
#            plt.ylabel('dg/d$\psi$')
#            plt.subplot(2, 2, 4)
#            plt.plot(flux_grid[:boundary_ind], pres_psi[:boundary_ind],
#                     'b', label='point data, size=' + str(pres_psi.size))
#            plt.plot(flux_grid[:boundary_ind], self.pres_1d(flux_grid[:boundary_ind],
#                     der=1), 'r--', label='smoothed with pts=' + str(flux_grid_smoothed.size))
#            plt.legend()
#            plt.xlim(self._psimin, psi_edge)
#            plt.xlabel('$\psi$')
#            plt.ylabel('dp/d$\psi$')
#
#            plt.figure(figsize=(6.5, 8))
#            plt.subplot(2, 1, 1)
#            plt.plot(flux_grid[:boundary_ind], q_profile[:boundary_ind],
#                     'b', label='point data, size=' + str(g_profile.size))
#            plt.plot(flux_grid[:boundary_ind], self.q_1d(flux_grid[:boundary_ind]),
#                     'r--', label='smoothed with pts=' + str(flux_grid_smoothed.size))
#            plt.legend()
#            plt.xlim(self._psimin, psi_edge)
#            plt.xlabel('$\psi$')
#            plt.ylabel('q')
#            plt.title('safety factor q')
#            plt.subplot(2, 1, 2)
#            plt.plot(flux_grid[:boundary_ind], q_psi[:boundary_ind],
#                     'b', label='point data, size=' + str(g_psi.size))
#            plt.plot(flux_grid[:boundary_ind], self.q_1d(flux_grid[:boundary_ind], der=1),
#                     'r--', label='smoothed with pts=' + str(flux_grid_smoothed.size))
#            plt.legend()
#            plt.xlim(self._psimin, psi_edge)
#            plt.xlabel('$\psi$')
#            plt.ylabel('dq/d$\psi$')
#
#            plt.figure(figsize=(13, 6.5))
#            plt.subplot(1, 2, 1)
#            RR, ZZ = np.meshgrid(R, Z, indexing='ij')
#            plt.contourf(RR, ZZ, psi, levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, psi, levels=50, colors='red', linewidths=.5)
#            plt.contour(RR, ZZ, psi, levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.scatter(R_at_axis, Z_at_axis, 20, 'red', zorder=10)
#            plt.axis('equal')
#            plt.xlabel('R')
#            plt.ylabel('Z')
#            plt.title('eqdsk point data $\psi$, shape=' + str(psi.shape))
#            plt.subplot(1, 2, 2)
#            plt.contourf(RR, ZZ, self.psi_fun(RR, ZZ), levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=50,
#                        colors='red', linewidths=.5)
#            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.scatter(R_at_axis, Z_at_axis, 20, 'red', zorder=10)
#            plt.axis('equal')
#            plt.xlabel('R')
#            plt.ylabel('Z')
#            plt.title('interpolated $\psi$, smoothed with pts=' +
#                      str(psi_smoothed.shape))
#
#            plt.figure(figsize=(13, 6.5))
#            plt.subplot(1, 2, 1)
#            _s_fun = np.vectorize(s_fun)
#            plt.contourf(RR, ZZ, _s_fun(RR, ZZ), levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, _s_fun(RR, ZZ), levels=50,
#                        colors='red', linewidths=.5)
#            plt.contour(RR, ZZ, _s_fun(RR, ZZ), levels=[
#                        1.], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('R')
#            plt.ylabel('Z')
#            plt.title(
#                'normalized, interpolated data $s=\sqrt{(\psi - \psi_a) / (\psi_e - \psi_a)}$')
#            plt.subplot(1, 2, 2)
#            plt.contourf(RR, ZZ, self.g_fun(RR, ZZ), levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, self.g_fun(RR, ZZ), levels=50,
#                        colors='red', linewidths=.5)
#            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('R')
#            plt.ylabel('Z')
#            plt.title('$g(\psi(R, Z))$, smoothed')
#
#            plt.figure(figsize=(13, 13))
#            plt.subplot(2, 2, 1)
#            plt.contourf(RR, ZZ, self.psi_fun(RR, ZZ, der='r'), levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ, der='r'), levels=50,
#                        colors='red', linewidths=.5)
#            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('R')
#            plt.ylabel('Z')
#            plt.title('r-derivative of interpolated $\psi$, smoothed')
#            plt.subplot(2, 2, 2)
#            plt.contourf(RR, ZZ, self.psi_fun(RR, ZZ, der='z'), levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ, der='z'), levels=50,
#                        colors='red', linewidths=.5)
#            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('R')
#            plt.ylabel('Z')
#            plt.title('z-derivative of interpolated $\psi$, smoothed')
#            plt.subplot(2, 2, 3)
#            plt.contourf(RR, ZZ, psi_r, levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, psi_r, levels=50,
#                        colors='red', linewidths=.5)
#            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('R')
#            plt.ylabel('Z')
#            plt.title('r-derivative $\psi$ data, FD')
#            plt.subplot(2, 2, 4)
#            plt.contourf(RR, ZZ, psi_z, levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, psi_z, levels=50,
#                        colors='red', linewidths=.5)
#            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('R')
#            plt.ylabel('Z')
#            plt.title('z-derivative $\psi$ data, FD')
#
#            plt.figure(figsize=(13, 13))
#            plt.subplot(2, 2, 1)
#            plt.contourf(RR, ZZ, self.psi_fun(RR, ZZ, der='rr'), levels=50)
#            plt.colorbar()
#            # plt.contour(RR, ZZ, self.psi_fun(RR, ZZ, der='rr'), levels=50,
#            #             colors='red', linewidths=.5)
#            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('R')
#            plt.ylabel('Z')
#            plt.title('rr-derivative of interpolated $\psi$, smoothed')
#            plt.subplot(2, 2, 2)
#            plt.contourf(RR, ZZ, self.psi_fun(RR, ZZ, der='zz'), levels=50)
#            plt.colorbar()
#            # plt.contour(RR, ZZ, self.psi_fun(RR, ZZ, der='zz'), levels=50,
#            #             colors='red', linewidths=.5)
#            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('R')
#            plt.ylabel('Z')
#            plt.title('zz-derivative of interpolated $\psi$, smoothed')
#            plt.subplot(2, 2, 3)
#            plt.contourf(RR, ZZ, psi_rr, levels=50)
#            plt.colorbar()
#            # plt.contour(RR, ZZ, self.psi_fun(RR, ZZ, der='rr'), levels=50,
#            #             colors='red', linewidths=.5)
#            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('R')
#            plt.ylabel('Z')
#            plt.title('rr-derivative of point data')
#            plt.subplot(2, 2, 4)
#            plt.contourf(RR, ZZ, psi_zz, levels=50)
#            plt.colorbar()
#            # plt.contour(RR, ZZ, self.psi_fun(RR, ZZ, der='rr'), levels=50,
#            #             colors='red', linewidths=.5)
#            plt.contour(RR, ZZ, self.psi_fun(RR, ZZ), levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('R')
#            plt.ylabel('Z')
#            plt.title('zz-derivative of point data')
#
#            plt.figure(figsize=(13, 13))
#            Y = 0.
#            XX, YY, ZZ2 = np.meshgrid(R, Y, Z, indexing='ij')
#            bx = self.b_xyz(XX, YY, ZZ2)[0].squeeze()
#            by = self.b_xyz(XX, YY, ZZ2)[1].squeeze()
#            bz = self.b_xyz(XX, YY, ZZ2)[2].squeeze()
#            plt.subplot(2, 2, 1)
#            plt.contourf(XX.squeeze(), ZZ2.squeeze(), bx, levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, psi, levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('x')
#            plt.ylabel('z')
#            plt.title('bx')
#            plt.subplot(2, 2, 2)
#            plt.contourf(XX.squeeze(), ZZ2.squeeze(), by, levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, psi, levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('x')
#            plt.ylabel('z')
#            plt.title('by')
#            plt.subplot(2, 2, 3)
#            plt.contourf(XX.squeeze(), ZZ2.squeeze(), bz, levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, psi, levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('x')
#            plt.ylabel('z')
#            plt.title('bz')
#            plt.subplot(2, 2, 4)
#            plt.contourf(XX.squeeze(), ZZ2.squeeze(), np.sqrt(
#                bx**2 + by**2 + bz**2), levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, psi, levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('x')
#            plt.ylabel('z')
#            plt.title('abs(b)')
#
#            plt.figure(figsize=(13, 13))
#            jx = self.j_xyz(XX, YY, ZZ2)[0].squeeze()
#            jy = self.j_xyz(XX, YY, ZZ2)[1].squeeze()
#            jz = self.j_xyz(XX, YY, ZZ2)[2].squeeze()
#            plt.subplot(2, 2, 1)
#            plt.contourf(XX.squeeze(), ZZ2.squeeze(), jx, levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, psi, levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('x')
#            plt.ylabel('z')
#            plt.title('jx')
#            plt.subplot(2, 2, 2)
#            plt.contourf(XX.squeeze(), ZZ2.squeeze(), jy, levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, psi, levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('x')
#            plt.ylabel('z')
#            plt.title('jy')
#            plt.subplot(2, 2, 3)
#            plt.contourf(XX.squeeze(), ZZ2.squeeze(), jz, levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, psi, levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('x')
#            plt.ylabel('z')
#            plt.title('jz')
#            plt.subplot(2, 2, 4)
#            plt.contourf(XX.squeeze(), ZZ2.squeeze(), np.sqrt(
#                jx**2 + jy**2 + jz**2), levels=50)
#            plt.colorbar()
#            plt.contour(RR, ZZ, psi, levels=[
#                        psi_edge], colors='black', linewidths=2.)
#            plt.axis('equal')
#            plt.xlabel('x')
#            plt.ylabel('z')
#            plt.title('abs(j)')
#
#    @property
#    def domain(self):
#        """ Domain object that characterizes the mapping from the logical to the physical domain.
#        """
#        return self._domain
#
#    @property
#    def params(self):
#        '''Parameters describing the equilibrium.'''
#        return self._params
#
#    def b_xyz(self, x, y, z):
#        '''B-field in Cartesian coordinates.'''
#
#        from struphy.geometry.base import Domain
#
#        x, y, z, is_sparse_meshgrid = Domain.prepare_eval_pts(x, y, z)
#
#        r = np.sqrt(x**2 + y**2)
#        phi = np.arctan2(y, x)
#        z = z + 0*r  # broadcasting happens here
#
#        #r2 = r[:, 0, :]
#        #z2 = z[:, 0, :]
#        r2 = r[:, :, 0]
#        z2 = z[:, :, 0]
#
#        # B as 2-form (second component is already multiplied by r)
#        # TODO: remove is_sparse_meshgrid (not necessary anymore)
#        b2_1_tmp = self.psi_fun(r2, z2, 'z', is_sparse_meshgrid)
#        b2_2_tmp = -self.g_fun(r2, z2, None, is_sparse_meshgrid)
#        b2_3_tmp = -self.psi_fun(r2, z2, 'r', is_sparse_meshgrid)
#
#        if is_sparse_meshgrid:
#            shp = (r.shape[0], phi.shape[1], z.shape[2])
#        else:
#            shp = r.shape
#
#        b2_1 = np.empty(shp)
#        b2_2 = np.empty(shp)
#        b2_3 = np.empty(shp)
#
#        #b2_1[:] = b2_1_tmp[:, None, :]
#        #b2_2[:] = b2_2_tmp[:, None, :]
#        #b2_3[:] = b2_3_tmp[:, None, :]
#        b2_1[:] = b2_1_tmp[:, :, None]
#        b2_2[:] = b2_2_tmp[:, :, None]
#        b2_3[:] = b2_3_tmp[:, :, None]
#
#        # push-forward of b2
#        b_x = (np.cos(phi)*b2_1 - np.sin(phi)*b2_2) / r
#        b_y = (np.sin(phi)*b2_1 + np.cos(phi)*b2_2) / r
#        b_z = b2_3 / r
#
#        return b_x, b_y, b_z
#
#    def j_xyz(self, x, y, z):
#        '''Current density in Cartesian coordinates.'''
#
#        from struphy.geometry.base import Domain
#
#        x, y, z, is_sparse_meshgrid = Domain.prepare_eval_pts(x, y, z)
#
#        r = np.sqrt(x**2 + y**2)
#        phi = np.arctan2(y, x)
#        z = z + 0*r  # broadcasting happens here if sparse_meshgrid
#
#        r2 = r[:, 0, :]
#        z2 = z[:, 0, :]
#
#        # J as 2-form (second component is already multiplied by r)
#        # TODO: remove is_sparse_meshgrid (not necessary anymore)
#        j2_1_tmp = - self.g_fun(r2, z2, 'z', is_sparse_meshgrid)
#        j2_2_tmp = self.psi_fun(r2, z2, 'rr', is_sparse_meshgrid) + \
#            self.psi_fun(r2, z2, 'zz', is_sparse_meshgrid)
#        j2_3_tmp = self.g_fun(r2, z2, 'r', is_sparse_meshgrid)
#
#        if is_sparse_meshgrid:
#            shp = (r.shape[0], phi.shape[1], z.shape[2])
#        else:
#            shp = r.shape
#
#        j2_1 = np.empty(shp)
#        j2_2 = np.empty(shp)
#        j2_3 = np.empty(shp)
#
#        j2_1[:] = j2_1_tmp[:, None, :]
#        j2_2[:] = j2_2_tmp[:, None, :]
#        j2_3[:] = j2_3_tmp[:, None, :]
#
#        # push-forward of b2
#        j_x = (np.cos(phi)*j2_1 - np.sin(phi)*j2_2) / r
#        j_y = (np.sin(phi)*j2_1 + np.cos(phi)*j2_2) / r
#        j_z = j2_3 / r
#
#        return j_x, j_y, j_z
#
#    def p_xyz(self, x, y, z):
#        '''Pressure in Cartesian coordinates.'''
#
#        from struphy.geometry.base import Domain
#
#        x, y, z, is_sparse_meshgrid = Domain.prepare_eval_pts(x, y, z)
#
#        r = np.sqrt(x**2 + y**2)
#        phi = np.arctan2(y, x)
#        z = z + 0*r  # broadcasting happens here
#
#        r2 = r[:, 0, :]
#        z2 = z[:, 0, :]
#
#        psi_tmp = self.psi_fun(r2, z2, None, is_sparse_meshgrid)
#
#        if is_sparse_meshgrid:
#            shp = (r.shape[0], phi.shape[1], z.shape[2])
#        else:
#            shp = r.shape
#
#        psi = np.empty(shp)
#
#        psi[:] = psi_tmp[:, None, :]
#
#        return self.pres_1d(psi.flatten()).reshape(psi.shape)
#
#    def n_xyz(self, x, y, z):
#        """ Equilibrium number density in physical space.
#        """
#        return 0*x + 0*y + 0*z
#
#    ##################
#    # Helper functions
#    ##################
#
#    @staticmethod
#    def field_line_tracing(s_fun, x_at_axis, y_at_axis, xdim, Nel, p, theta='equal_angle'):
#        '''Given a poloidal flux function s(x, y), computes a mapping (x, y) = F(s, theta).
#        Three different theta parametrizations can be chosen: 'equal_angle', 'equal_arc_length' or 'sfl' (PEST).
#
#        Parameters
#        ----------
#        s_fun : callable
#            The normalized flux function s(x, y). The range of s must be [0, r] with r>=1.
#
#        x_at_axis : float
#            x-coordinate of the pole (magnetic axis).
#
#        y_at_axis : float
#            y-coordinate of the pole (magnetic axis).
#
#        xdim : float
#            Length of x-domain.
#
#        Nel : list[int]
#            Number of elements to be used for spline inerpolation.
#
#        p : list[int]
#            Spline degrees for spline interpolation.
#
#        theta: str
#            WHich theta parametrization to use: 'equal_angle', NOT YET: 'equal_arc_length' or 'sfl' (PEST)'''
#
#        import numpy as np
#        from struphy.b_splines import bsplines as bsp
#        from scipy.optimize import newton
#        from struphy.geometry.base import spline_interpolation_nd
#
#        assert callable(s_fun)
#        assert len(Nel) == 2
#        assert len(p) == 2
#
#        # avoid theta = 0.5 angle
#        while True:
#            el_b = [np.linspace(0., 1., Nel + 1) for Nel in Nel]
#
#            spl_kind = [False, True]
#            T = [bsp.make_knots(el_b, p, kind)
#                 for el_b, p, kind in zip(el_b, p, spl_kind)]
#
#            s_grev, th_grev = [bsp.greville(T, p, kind)
#                               for T, p, kind in zip(T, p, spl_kind)]
#
#            if (not np.any(np.abs(th_grev - .5) < 1e-14)):
#                break
#            else:
#                Nel[1] += 1
#
#        X = np.empty((s_grev.size, th_grev.size))
#        Y = np.empty_like(X)
#        for j, thj in enumerate(th_grev):
#            def xj(r): return x_at_axis + r*np.cos(2*np.pi*thj)
#            def yj(r): return y_at_axis + r*np.sin(2*np.pi*thj)
#            ri = 0.
#            for i, si in enumerate(s_grev):
#                if i > 0:
#                    if i == 1:
#                        ri = xdim/20.
#
#                    def f(r): return s_fun(xj(r), yj(r)) - si
#                    ri = newton(f, ri)
#                X[i, j] = xj(ri)
#                Y[i, j] = yj(ri)
#
#        cx, knots, indN = spline_interpolation_nd(
#            p, spl_kind, [s_grev, th_grev], X)
#        cy, knots, indN = spline_interpolation_nd(
#            p, spl_kind, [s_grev, th_grev], Y)
#
#        return cx, cy
#
#    def psi_fun(self, r, z, der=None, is_sparse_meshgrid=False):
#        '''Interpolated flux function psi(r, z), and its first derivatives.
#
#        Parameters
#        ----------
#        r, z : array
#            Must stem from meshgrid.
#
#        der : str
#            Which derivative to evaluate (None, 'r', 'z', 'rr' or 'zz').
#
#        is_sparse_meshgrid : bool
#            Refers to the shapes of r, z.
#
#        Returns
#        -------
#            A 2d array of dense meshgrid shape.
#        '''
#
#        from struphy.b_splines.bspline_evaluation_2d import evaluate_matrix, evaluate_sparse
#
#        if is_sparse_meshgrid:
#            _func = evaluate_sparse
#            out = np.zeros((r.shape[0], z.shape[1]))
#        else:
#            _func = evaluate_matrix
#            out = np.zeros(r.shape)
#
#        if der is None:
#            _func(*self._Tpsi, *self.params['p_for_psi'], *self._indNpsi, self._cpsi, (
#                r - self._rmin) / self._rdim, (z - self._zmin) / self._zdim, out, 0)
#            fac = 1.
#        elif der == 'r':
#            _func(*self._Tpsi, *self.params['p_for_psi'], *self._indNpsi, self._cpsi, (
#                r - self._rmin) / self._rdim, (z - self._zmin) / self._zdim, out, 31)
#            fac = 1 / self._rdim
#        elif der == 'z':
#            _func(*self._Tpsi, *self.params['p_for_psi'], *self._indNpsi, self._cpsi, (
#                r - self._rmin) / self._rdim, (z - self._zmin) / self._zdim, out, 32)
#            fac = 1 / self._zdim
#        elif der == 'rr':
#            _func(*self._Tpsi, *self.params['p_for_psi'], *self._indNpsi, self._cpsi, (
#                r - self._rmin) / self._rdim, (z - self._zmin) / self._zdim, out, 41)
#            fac = 1 / self._rdim**2
#        elif der == 'zz':
#            _func(*self._Tpsi, *self.params['p_for_psi'], *self._indNpsi, self._cpsi, (
#                r - self._rmin) / self._rdim, (z - self._zmin) / self._zdim, out, 42)
#            fac = 1 / self._zdim**2
#        else:
#            raise ValueError(f'der = {der} not implemented.')
#
#        return out * fac
#
#    def g_1d(self, psi, der=None):
#        '''Toroidal field function g(psi). Argument must be 1d array.'''
#
#        from struphy.b_splines.bspline_evaluation_1d import evaluate_vector
#
#        assert psi.ndim == 1
#        out = np.zeros(psi.shape)
#
#        if der is None:
#            kind = 0
#            fac = 1.
#        elif der == 1:
#            kind = 2
#            fac = 1 / self._psidim
#
#        evaluate_vector(self._Tg[0], self.params['p_for_psi'][0], self._indNg[0], self._cg,
#                        (psi - self._psimin) / self._psidim, out, kind)
#
#        return out * fac
#
#    def pres_1d(self, psi, der=None):
#        '''Pressure profile p(psi). Argument must be 1d array.'''
#
#        from struphy.b_splines.bspline_evaluation_1d import evaluate_vector
#
#        assert psi.ndim == 1
#        out = np.zeros(psi.shape)
#
#        if der is None:
#            kind = 0
#            fac = 1.
#        elif der == 1:
#            kind = 2
#            fac = 1 / self._psidim
#
#        evaluate_vector(self._Tpres[0], self.params['p_for_psi'][0], self._indNpres[0], self._cpres,
#                        (psi - self._psimin) / self._psidim, out, kind)
#
#        return out * fac
#
#    def q_1d(self, psi, der=None):
#        '''Safety factor q(psi). Argument must be 1d array.'''
#
#        from struphy.b_splines.bspline_evaluation_1d import evaluate_vector
#
#        assert psi.ndim == 1
#        out = np.zeros(psi.shape)
#
#        if der is None:
#            kind = 0
#            fac = 1.
#        elif der == 1:
#            kind = 2
#            fac = 1 / self._psidim
#
#        evaluate_vector(self._Tg[0], self.params['p_for_psi'][0], self._indNg[0], self._cq,
#                        (psi - self._psimin) / self._psidim, out, kind)
#
#        return out * fac
#
#    def g_fun(self, r, z, der=None, is_sparse_meshgrid=False):
#        '''Toroidal field function g(psi(r, z)). Arguments must stem from meshgrid.'''
#
#        if der is None:
#            out = self.g_1d(self.psi_fun(
#                r, z, None, is_sparse_meshgrid).flatten()).reshape(r.shape)
#        elif der == 'r':
#            psi_r = self.psi_fun(r, z, 'r', is_sparse_meshgrid)
#            out = self.g_1d(self.psi_fun(
#                r, z, None, is_sparse_meshgrid).flatten(), der=1).reshape(r.shape) * psi_r
#        elif der == 'z':
#            psi_z = self.psi_fun(r, z, 'z', is_sparse_meshgrid)
#            out = self.g_1d(self.psi_fun(
#                r, z, None, is_sparse_meshgrid).flatten(), der=1).reshape(r.shape) * psi_z
#
#        return out


class GVECequilibrium(LogicalMHDequilibrium):
    '''Interface to `gvec_to_python <https://gitlab.mpcdf.mpg.de/spossann/gvec_to_python>`_.

    Parameters
    ----------
    **params
        Parameters that characterize the MHD equilibrium. Possible keys are

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

    def __init__(self, show=False, **params):

        from struphy.geometry.base import interp_mapping
        from struphy.geometry.domains import Spline

        from gvec_to_python.reader.gvec_reader import create_GVEC_json
        from gvec_to_python import GVEC

        import struphy

        params_default = {'rel_path': True,
                          'dat_file': '/ellipstell_v2/newBC_E1D6_M6N6/GVEC_ELLIPSTELL_V2_State_0000_00200000.dat',
                          'json_file': None,
                          'use_pest': False,
                          'use_nfp': True,
                          'Nel': (16, 16, 16),
                          'p': (3, 3, 3), }

        self._params = set_defaults(params, params_default)

        if self._params['dat_file'] is None:

            assert self._params['json_file'] is not None
            assert self._params['json_file'][-5:] == '.json'

            if self._params['rel_path']:
                json_file = struphy.__path__[
                    0] + '/fields_background/mhd_equil/gvec' + self._params['json_file']
            else:
                json_file = self._params['json_file']

        else:

            assert self._params['dat_file'][-4:] == '.dat'

            if self._params['rel_path']:
                dat_file = struphy.__path__[
                    0] + '/fields_background/mhd_equil/gvec' + self._params['dat_file']
            else:
                dat_file = params['dat_file']

            json_file = dat_file[:-4] + '.json'
            create_GVEC_json(dat_file, json_file)

        if self._params['use_pest']:
            mapping = 'unit_pest'
        else:
            mapping = 'unit'

        if self._params['use_nfp']:
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
            self._params['Nel'], self._params['p'], spl_kind, X, Y, Z)

        # struphy domain object
        params_map = {'cx': cx, 'cy': cy, 'cz': cz,
                      'Nel': self._params['Nel'], 'p': self._params['p'], 'spl_kind': spl_kind}
        self._domain = Spline(**params_map)

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


def set_defaults(params_in, params_default):
    """
    Sets missing default key-value pairs in dictionary "params_in" according to "params_default".

    Parameters
    ----------
    params_in : dict
        Dictionary which is compared to the dictionary "params_default" and to which missing defaults are added.

    params_default : dict
        Dictionary with default values.

    Returns
    -------
    params : dict
        Dictionary with same keys as "params_default" and default values for missing keys.
    """

    # check for correct keys in params_in
    for key in params_in:
        assert key in params_default, f'Unknown key "{key}". Please choose one of {[*params_default]}.'

    # set default values if key is missing
    params = params_in

    for key, val in params_default.items():
        params.setdefault(key, val)

    return params
