'MHD equilibria.'


import numpy as np

from struphy.fields_background.mhd_equil.base import CartesianMHDequilibrium, LogicalMHDequilibrium, AxisymmMHDequilibrium


class HomogenSlab(CartesianMHDequilibrium):
    r"""
    Homogeneous MHD equilibrium:

    .. math::

        \mathbf B &= B_{0x}\,\mathbf e_x + B_{0y}\,\mathbf e_y + B_{0z}\,\mathbf e_z = const.\,,

        p &= \beta \frac{|\mathbf B|^2}{2}=const.\,,

        n &= n_0 = const.\,.

    Units are those defned in the parameter file (:code:`struphy units -h`).

    Parameters
    ----------
    B0x : float  
        x-component of magnetic field (default: 0.).
    B0y : float  
        y-component of magnetic field (default: 0.).
    B0z : float  
        z-component of magnetic field (default: 1.).
    beta : float
        Plasma beta (ratio of kinematic pressure to B^2/(2*mu0), default: 0.1).
    n0 : float 
        Ion number density (default: 1.).

    Note
    ----
    In the parameter .yml, use the following in the section `mhd_equilibrium`::

        mhd_equilibrium :
            type : HomogenSlab
            HomogenSlab :
                B0x  : 0. # magnetic field in x
                B0y  : 0. # magnetic field in y
                B0z  : 1. # magnetic field in z
                beta : .1 # plasma beta = p*(2*mu_0)/B^2
                n0   : 1. # number density     
    """

    def __init__(self, **params):

        params_default = {'B0x': 0.,
                          'B0y': 0.,
                          'B0z': 1.,
                          'beta': .1,
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

    # equilibrium magnetic field (curl of equilibrium vector potential)
    def b_xyz(self, x, y, z):
        """ Magnetic field.
        """
        bx = self.params['B0x'] - 0*x
        by = self.params['B0y'] - 0*x
        bz = self.params['B0z'] - 0*x

        return bx, by, bz

    # equilibrium vector potential
    def a_xyz(self, x, y, z):
        """ Vector potential.
        """
        bx = self.params['B0x'] - 0*x
        by = self.params['B0y'] - 0*x
        bz = self.params['B0z'] - 0*x

        ax = by*z
        ay = bz*x
        az = bx*y

        return ax, ay, az

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
        """ Plasma pressure.
        """
        pp = self.params['beta'] * (self.params['B0x']**2 +
                                    self.params['B0y']**2 + self.params['B0z']**2) / 2. - 0*x

        return pp

    # equilibrium number density
    def n_xyz(self, x, y, z):
        """ Number density.
        """
        nn = self.params['n0'] - 0*x

        return nn

    # equilibrium current (curl of equilibrium magnetic field)
    def gradB_xyz(self, x, y, z):
        """ Current density.
        """
        gradBx = 0*x
        gradBy = 0*x
        gradBz = 0*x

        return gradBx, gradBy, gradBz


class ShearedSlab(CartesianMHDequilibrium):
    r"""
    Sheared slab MHD equilibrium in a cube with side lengths :math:`L_x=a,\,L_y=2\pi a,\,L_z=2\pi R_0`. Profiles depend on :math:`x` solely: 

    .. math::

        \mathbf B(x) &= B_{0} \left( \mathbf e_z + \frac{a}{q(x)R_0}\mathbf e_y\right)\,,\qquad q(x) = q_0 + ( q_1 - q_0 )\frac{x^2}{a^2}\,,

        p(x) &= \beta\frac{B_{0}^2}{2} \left( 1 + \frac{a^2}{q(x)^2 R_0^2} \right) + B_{0}^2 \frac{a^2}{R_0^2} \left( \frac{1}{q_0^2} - \frac{1}{q(x)^2} \right)\,,

        n(x) &= n_a + ( 1 - n_a ) \left( 1 - \left(\frac{x}{a}\right)^{n_1} \right)^{n_2} \,.

    Units are those defned in the parameter file (:code:`struphy units -h`).

    Parameters
    ---------- 
    a : float 
        "Minor" radius (must be compatible with :math:`L_x=a` and :math:`L_y=2\pi a`, default: 1.).
    R0 : float
        "Major" radius (must be compatible with :math:`L_z=2\pi R_0`, default: 3.).
    B0 : float
        z-component of magnetic field (constant) (default: 1.).
    q0 : float
        Safety factor at x=0 (default: 1.05).
    q1 : float
        Safety factor at x=a (default: 1.80).
    n1 : float
        1st shape factor for ion number density profile (default: 0.).
    n2 : float
        2nd shape factor for ion number density profile (default: 0.). 
    na : float
        Ion number density at x=a (default: 1.).
    beta : float
        Plasma beta (ratio of kinematic pressure to B^2/(2*mu0), default: 0.1).

    Note
    ----
    In the parameter .yml, use the following in the section `mhd_equilibrium`::

        mhd_equilibrium :
            type : ShearedSlab
            ShearedSlab :
                a    : 1.   # minor radius (Lx=a, Ly=2*pi*a) 
                R0   : 3.   # major radius (Lz=2*pi*R0)
                B0   : 1.   # magnetic field in z-direction    
                q0   : 1.05 # safety factor at x = 0
                q1   : 1.80 # safety factor at x = a
                n1   : 0.   # 1st shape factor for ion number density profile
                n2   : 0.   # 2nd shape factor for ion number density profile
                na   : 1.   # number density at r=a
                beta : .1   # plasma beta = p*(2*mu_0)/B^2
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
                          'beta': .1}

        self._params = set_defaults(params, params_default)

    @property
    def params(self):
        """ Parameters dictionary.
        """
        return self._params

    # ===============================================================
    #             profiles for a sheared slab geometry
    # ===============================================================

    def q_x(self, x, der=0):
        """ Safety factor profile q = q(x) (or its first derivative if der=1).
        """

        assert der >= 0 and der <= 1, 'Only first derivative available!'

        if self.params['q0'] == 'inf' and self.params['q1'] == 'inf':
            if der == 0:
                qout = 101. - 0*x
            else:
                qout = 0*x

        else:
            if der == 0:
                qout = self.params['q0'] + (self.params['q1'] -
                                            self.params['q0'])*(x/self.params['a'])**2
            else:
                qout = 2*(self.params['q1'] -
                          self.params['q0'])*x/self.params['a']**2

        return qout

    def p_x(self, x):
        """ Pressure profile p = p(x).
        """
        q = self.q_x(x)

        eps = self.params['a']/self.params['R0']

        if np.all(q >= 100.):
            pout = self.params['B0']**2*self.params['beta'] / 2. - 0*x
        else:
            pout = self.params['B0']**2*self.params['beta'] / 2. * (
                1 + eps**2/q**2) + self.params['B0']**2*eps**2*(1/self.params['q0']**2 - 1/q**2)

        return pout

    def n_x(self, x):
        """ Ion number density profile n = n(x).
        """
        nout = (1 - self.params['na'])*(1 - (x/self.params['a']) **
                                        self.params['n1'])**self.params['n2'] + self.params['na']

        return nout

    def plot_profiles(self, n_pts=501):
        """ Plots radial profiles.
        """

        import matplotlib.pyplot as plt

        x = np.linspace(0., self.params['a'], n_pts)

        fig, ax = plt.subplots(1, 3)

        fig.set_figheight(3)
        fig.set_figwidth(12)

        ax[0].plot(x, self.q_x(x))
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('q')

        ax[1].plot(x, self.p_x(x))
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('p')

        ax[2].plot(x, self.n_x(x))
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

        q = self.q_x(x)
        eps = self.params['a']/self.params['R0']
        if np.all(q >= 100.):
            by = 0*x
            bz = self.params['B0'] - 0*x
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

        q = self.q_x(x)
        eps = self.params['a']/self.params['R0']
        if np.all(q >= 100.):
            jz = 0*x
        else:
            jz = -self.params['B0']*eps*self.q_x(x, der=1)/q**2

        return jx, jy, jz

    # equilibrium pressure
    def p_xyz(self, x, y, z):
        """ Pressure.
        """
        pp = self.p_x(x)

        return pp

    # equilibrium number density
    def n_xyz(self, x, y, z):
        """ Number density.
        """
        nn = self.n_x(x)

        return nn

    # gradient of equilibrium magnetic field (grad of equilibrium magnetic field)
    def gradB_xyz(self, x, y, z):
        """ Gradient of magnetic field.
        """
        gradBy = 0*x
        gradBz = 0*x

        q = self.q_x(x)
        eps = self.params['a']/self.params['R0']
        if np.all(q >= 100.):
            gradBx = 0*x
        else:
            gradBx = -self.params['B0']*eps**2 / \
                np.sqrt(1+eps**2/self.q_x(x)**2) * \
                self.q_x(x, der=1)/self.q_x(x)**3

        return gradBx, gradBy, gradBz


class ScrewPinch(CartesianMHDequilibrium):
    r"""
    Straight tokamak (screw pinch) MHD equilibrium for a cylindrical geometry of radius :math:`a` and length :math:`L_z=2\pi R_0`.

    The profiles in cylindrical coordinates :math:`(r, \theta, z)` with transformation formulae 

    .. math::

        x &= r\cos\theta\,,

        y &= r\sin\theta\,,

        z &= z\,,

    are:

    .. math::

        \mathbf B(r) &= B_{0}\left( \mathbf e_z + \frac{r}{q(r) R_0}\mathbf e_\theta \right)\,,\qquad q(r) = q_0 + ( q_1 - q_0 )\frac{r^2}{a^2}\,,

        p(r) &= \left\{\begin{aligned}
        &\frac{B_{0}^2 a^2 q_0}{ 2 R_0^2(q_1 - q_0) } \left( \frac{1}{q(r)^2} - \frac{1}{q_1^2} \right) \quad &&\textnormal{if}\quad q_1\neq q_0\neq\infty\,, 

        &\frac{B_{0}^2 a^2}{R_0^2q_0^2} \left(1 - \frac{r^2}{a^2} \right) \quad &&\textnormal{if}\quad q_1= q_0\neq\infty\,, 

        &\beta\frac{B_{0}^2}{2} \quad &&\textnormal{if}\quad q_0= q_1=\infty\,, 
        \end{aligned}\right.

        n(r) &= n_a + ( 1 - n_a )\left( 1 - \left(\frac{r}{a}\right)^{n_1} \right)^{n_2}\,.

    Units are those defned in the parameter file (:code:`struphy units -h`).

    Parameters
    ----------
    a : float 
        "Minor" radius (radius of cylinder, default: 1.).
    R0 : float 
        "Major" radius (must be compatible with :math:`L_z=2\pi R_0`, default: 5.).
    B0 : float
        z-component of magnetic field (constant) (default: 1.).
    q0 : float, str
        Safety factor at r=0 (use the string "inf" for infinity, default: 1.05).
    q1 : float, str
        Safety factor at r=a (use the string "inf" for infinity, default: 1.80).
    n1 : float
        1st shape factor for ion number density profile (default: 0.).
    n2 : float
        2nd shape factor for ion number density profile (default: 0.). 
    na : float
        Ion nnumber density at r=a (default: 1.).
    beta : float
        Plasma beta for :math:`q_0=q_1=\infty` (ratio of kinematic pressure to B^2/(2*mu0), default: 0.1).

    Note
    ----
    In the parameter .yml, use the following in the section `mhd_equilibrium`::

        mhd_equilibrium :
            type : ScrewPinch
            ScrewPinch :
                a    : 1.   # minor radius (radius of cylinder)
                R0   : 3.   # major radius (length of pinch Lz=2*pi*R0)
                B0   : 1.   # magnetic field in z-direction
                q0   : 1.05 # safety factor at r=0
                q1   : 1.80 # safety factor at r=a
                n1   : 0.   # 1st shape factor for ion number density profile 
                n2   : 0.   # 2nd shape factor for ion number density profile 
                na   : 1.   # ion number density at r=a
                beta : 0.1  # plasma beta = p*(2*mu_0)/B^2 for q0=q1=inf (pure axial field).
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
                          'beta': .1}

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

    def q_r(self, r, der=0):
        """ Radial safety factor profile q = q(r) (and first derivative).
        """

        assert der >= 0 and der <= 1, 'Only first derivative available!'

        if self.params['q0'] == 'inf' and self.params['q1'] == 'inf':
            if der == 0:
                qout = 101. - 0*r
            else:
                qout = 0*r

        else:
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

        q0 = self.params['q0']
        q1 = self.params['q1']
        B0 = self.params['B0']

        if q0 == 'inf' and q1 == 'inf':
            pout = B0**2 * self.params['beta'] / 2. - 0*r

        else:

            if q0 == q1:
                pout = (B0**2*eps**2/q0**2)*(1 - r**2/self.params['a']**2)
            else:
                pout = B0**2*eps**2*q0/(2*(q1 - q0)) * \
                    (1/self.q_r(r)**2 - 1/q1**2)

        return pout

    def n_r(self, r):
        """ Radial ion number density profile n = n(r).
        """
        nout = (1 - self.params['na'])*(1 - (r/self.params['a']) **
                                        self.params['n1'])**self.params['n2'] + self.params['na']

        return nout

    def plot_profiles(self, n_pts=501):
        """ Plots radial profiles.
        """

        import matplotlib.pyplot as plt

        r = np.linspace(0., self.params['a'], n_pts)

        fig, ax = plt.subplots(1, 3)

        fig.set_figheight(3)
        fig.set_figwidth(12)

        ax[0].plot(r, self.q_r(r))
        ax[0].set_xlabel('r')
        ax[0].set_ylabel('q')

        ax[0].plot(r, np.ones(r.size), 'k--')

        ax[1].plot(r, self.p_r(r))
        ax[1].set_xlabel('r')
        ax[1].set_ylabel('p')

        ax[2].plot(r, self.n_r(r))
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
        q = self.q_r(r)
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
        q = self.q_r(r)
        q_p = self.q_r(r, der=1)
        if np.all(q >= 100.):
            jz = 0*x
        else:
            jz = self.params['B0']/(self.params['R0']*q**2)*(2*q - r*q_p)

        return jx, jy, jz

    # equilibrium pressure
    def p_xyz(self, x, y, z):
        """ Pressure.
        """
        pp = self.p_r(self.r(x, y, z))

        return pp

    # equilibrium number density
    def n_xyz(self, x, y, z):
        """ Number density.
        """
        nn = self.n_r(self.r(x, y, z))

        return nn

    # gradient of equilibrium magnetic field (grad of equilibrium magnetic field)
    def gradB_xyz(self, x, y, z):
        """ Gradient of magnetic field.
        """
        r = self.r(x, y, z)
        theta = self.theta(x, y, z)
        q = self.q_r(r)
        if np.all(q >= 100.):
            gradBr = 0*x
        else:
            gradBr = self.params['B0']/self.params['R0']**2/np.sqrt(1+r**2/self.q_r(
                r)**2/self.params['R0']**2)*(r/self.q_r(r)**2 - r**2/self.q_r(r)**3*self.q_r(r, der=1))
        gradBx = gradBr*np.cos(theta)
        gradBy = gradBr*np.sin(theta)
        gradBz = 0*x

        return gradBx, gradBy, gradBz


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

    for some given safety factor profile. Two profiles in terms of the on-axis :math:`q_0\equiv q(r=0)` and edge :math:`q_1\equiv q(r=a)` safety factor values are available (:math:`a` is the minor radius of the torus):

    .. math::

        q(r) &= \left\{\begin{aligned}
        &q_0 + ( q_1 - q_0 )\frac{r^2}{a^2} \quad &&\textnormal{if} \quad q_\textnormal{kind}=0\,,

        &\frac{q_0}{1-\left(1-\frac{r^2}{a^2}\right)^{\frac{q_1}{q_0}}}\frac{r^2}{a^2} \quad &&\textnormal{if} \quad q_\textnormal{kind}=1\,.
        \end{aligned}\right.

    The pressure profile

    .. math::

        p^\prime(r) &= -\frac{B_0^2}{R_0^2}\frac{r\left[2q(r)-rq^\prime(r)\right]}{q(r)^3} \quad &&\textnormal{if} \quad p_\textnormal{kind}=0\,,

        p(r) &= \beta \frac{B_{0}^2}{2} \left( 1 - p_1 \frac{r^2}{a^2} - p_2 \frac{r^4}{a^4} \right) \quad &&\textnormal{if} \quad p_\textnormal{kind}=1\,,

    is either the exact solution of the MHD equilibrium condition in the cylindrical limit (:math:`p_\textnormal{kind}=0`) or an monotonically decreasing adhoc profile for some given on-axis plasma beta (:math:`p_\textnormal{kind}=1`). Finally, the number density profile is chosen as

    .. math::

        n(r) = n_a + ( 1 - n_a ) \left( 1 - \left(\frac{r}{a}\right)^{n_1} \right)^{n_2}\,.

    Units are those defned in the parameter file (:code:`struphy units -h`).

    Parameters
    ----------
    a : float
        Minor radius of torus (default: 1.).
    R0 : float
        Major radius of torus (default: 10.).
    B0 : float
        On-axis (r=0) toroidal magnetic field (default: 3.).
    q_kind : int 
        Which safety factor profile, see docstring (0 or 1, default: 0).
    q0 : float
        Safety factor at r=0 (default: 1.71).
    q1 : float
        Safety factor at r=a (default: 1.87).
    n1 : float
        1st shape factor for ion number density profile (default: 0.).
    n2 : float
        2nd shape factor for ion number density profile (default: 0.).
    na : float
        Ion number density at r=a (default: 1.).
    p_kind : int 
        Kind of pressure profile, see docstring (0 or 1, default: 1).
    p1 : float
        1st shape factor for ad hoc pressure profile (default: 0.).
    p2 : float
        2nd shape factor for ad hoc pressure profile (default: 0.).
    beta : float
        On-axis (r=0) plasma beta if p_kind=1 (ratio of kinematic pressure to B^2/(2*mu0), default: 0.179).
    psi_k : int
        Spline degree to be used for interpolation of poloidal flux function (if q_kind=1, default=3).
    psi_nel : int
        Number of cells to be used for interpolation of poloidal flux function (if q_kind=1, default=50).

    Note
    ----
    In the parameter .yml, use the following in the section `mhd_equilibrium`::

        mhd_equilibrium :
            type : AdhocTorus
            AdhocTorus :
                a       : 1.   # minor radius
                R0      : 3.   # major radius
                B0      : 1.   # on-axis toroidal magnetic field
                q_kind  : 0    # which profile (0 : parabolic, 1 : other, see documentation)
                q0      : 1.05 # safety factor at r=0
                q1      : 1.80 # safety factor at r=a
                n1      : 0.   # 1st shape factor for number density profile 
                n2      : 0.   # 2nd shape factor for number density profile 
                na      : 1.   # number density at r=a
                p_kind  : 1    # kind of pressure profile (0 : cylindrical limit, 1 : ad hoc)
                p1      : .1   # 1st shape factor for ad hoc pressure profile
                p2      : .1   # 2nd shape factor for ad hoc pressure profile
                beta    : .01  # plasma beta = p*(2*mu_0)/B^2 for flat safety factor 
                psi_k   : 3    # spline degree to be used for interpolation of poloidal flux function (only needed if q_kind=1)
                psi_nel : 50   # number of cells to be used for interpolation of poloidal flux function (only needed if q_kind=1)
    """

    def __init__(self, **params):

        from scipy.interpolate import UnivariateSpline
        from scipy.integrate import quad

        # parameters
        params_default = {'a': 1.,
                          'R0': 3.,
                          'B0': 2.,
                          'q_kind': 0,
                          'q0': 1.71,
                          'q1': 1.87,
                          'n1': 0.,
                          'n2': 0.,
                          'na': 1.,
                          'p_kind': 1,
                          'p1': 0.,
                          'p2': 0.,
                          'beta': 0.179,
                          'psi_k': 3,
                          'psi_nel': 50}

        self._params = set_defaults(params, params_default)

        # plasma boundary contour
        ths = np.linspace(0., 2*np.pi, 201)

        self._rbs = self.params['R0'] * \
            (1 + self.params['a']/self.params['R0']*np.cos(ths))
        self._zbs = self.params['a']*np.sin(ths)

        # set on-axis and boundary fluxes
        if self.params['q_kind'] == 0:

            self._psi0 = self.psi(self.params['R0'], 0.)
            self._psi1 = self.psi(self.params['R0'] + self.params['a'], 0.)

            self._psi_i = None
            self._p_i = None

        else:

            r_i = np.linspace(0., self.params['a'], self.params['psi_nel'] + 1)

            def dpsi_dr(r):
                return self.params['B0']*r/(self.q_r(r)*np.sqrt(1 - r**2/self.params['R0']**2))

            psis = np.zeros_like(r_i)

            for i, rr in enumerate(r_i):
                psis[i] = quad(dpsi_dr, 0., rr)[0]

            self._psi_i = UnivariateSpline(
                r_i, psis, k=self.params['psi_k'], s=0., ext=3)

            self._psi0 = 0.
            self._psi1 = self.psi(self.params['R0'] + self.params['a'], 0.)

            def dp_dr(r):
                return -(self.params['B0']**2*r)/(self.params['R0']**2*self.q_r(r)**3)*(2*self.q_r(r) - r*self.q_r(r, der=1))

            ps = np.zeros_like(r_i)

            for i, rr in enumerate(r_i):
                ps[i] = quad(dp_dr, 0., rr)[0]

            self._p_i = UnivariateSpline(
                r_i, ps - ps[-1], k=self.params['psi_k'], s=0., ext=3)

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

        # parabolic profile (analytical)
        if self.params['q_kind'] == 0:

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

        # alternative profile (interpolated)
        else:

            out = self._psi_i(r, nu=der)

            # remove all "dimensions" for point-wise evaluation
            if isinstance(r, (int, float)):
                assert out.ndim == 0
                out = out.item()

        return out

    def q_r(self, r, der=0):
        """ Radial safety factor profile q = q(r) (and first derivative).
        """

        assert der >= 0 and der <= 1, 'Only first derivative available!'

        q0 = self.params['q0']
        q1 = self.params['q1']

        a = self.params['a']

        # parabolic profile
        if self.params['q_kind'] == 0:

            if der == 0:
                qout = q0 + (q1 - q0)*(r/a)**2
            else:
                qout = 2*(q1 - q0)*r/a**2

        # alternative profile
        else:

            # int/float input
            if isinstance(r, (int, float)):
                if r == 0:
                    if der == 0:
                        qout = 1*q0
                    else:
                        qout = 0*r
                else:
                    if der == 0:
                        if self.params['q0'] == self.params['q1']:
                            qout = 1*q0
                        else:
                            qout = q1*(r/a)**2/(1 - (1 - (r/a)**2)**(q1/q0))
                    else:
                        if self.params['q0'] == self.params['q1']:
                            qout = 0*r
                        else:
                            qout = (2*r*q1/a**2)*(1 - (1 - (r/a)**2)**(q1/q0) - (r/a)**2*(q1/q0)
                                                  * (1 - (r/a)**2)**(q1/q0 - 1))/(1 - (1 - (r/a)**2)**(q1/q0))**2

            # vector input
            else:
                sh = r.shape

                r_flat = r.flatten()

                r_zeros = np.where(r_flat == 0.)[0]
                r_nzero = np.where(r_flat != 0.)[0]

                qout = np.zeros(r_flat.size, dtype=float)

                if der == 0:
                    if self.params['q0'] == self.params['q1']:
                        qout[:] = 1*q0
                    else:
                        qout[r_zeros] = 1*q0
                        qout[r_nzero] = q1*(r_flat[r_nzero]/a)**2 / \
                            (1 - (1 - (r_flat[r_nzero]/a)**2)**(q1/q0))
                else:
                    if self.params['q0'] == self.params['q1']:
                        qout[:] = 0.
                    else:
                        qout[r_zeros] = 0*r_zeros
                        qout[r_nzero] = (2*r_flat[r_nzero]*q1/a**2)*(1 - (1 - (r_flat[r_nzero]/a)**2)**(q1/q0) - (r_flat[r_nzero]/a)**2*(
                            q1/q0)*(1 - (r_flat[r_nzero]/a)**2)**(q1/q0 - 1))/(1 - (1 - (r_flat[r_nzero]/a)**2)**(q1/q0))**2

                qout = qout.reshape(sh).copy()

        return qout

    def p_r(self, r):
        """ Radial pressure profile p = p(r).
        """

        eps = self.params['a']/self.params['R0']

        # profile in cylindrical limit
        if self.params['p_kind'] == 0:

            # parabolic q-profile
            if self.params['q_kind'] == 0:

                if self.params['q0'] == self.params['q1']:
                    pout = self.params['B0']**2*self.params['a']**2/(
                        self.params['R0']**2*self.params['q0']**2)*(1 - r**2/self.params['a']**2)
                else:
                    pout = self.params['B0']**2*eps**2*self.params['q0']/(
                        2*(self.params['q1'] - self.params['q0']))*(1/self.q_r(r)**2 - 1/self.params['q1']**2)

            # alternative profile
            else:

                pout = self._p_i(r)

                # remove all "dimensions" for point-wise evaluation
                if isinstance(r, (int, float)):
                    assert pout.ndim == 0
                    pout = pout.item()

        # ad-hoc profile
        else:

            pout = self.params['B0']**2 * self.params['beta'] / 2. * (
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

            d2r_dRdZ = -Z*(R - self.params['R0'])/r**3

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
            elif dR == 1 and dZ == 1:
                out = self.psi_r(r, der=2) * dr_dR * dr_dZ + \
                    self.psi_r(r, der=1) * d2r_dRdZ
            else:
                raise NotImplementedError(
                    'Only combinations (dR=0, dZ=0), (dR=1, dZ=0), (dR=0, dZ=1), (dR=2, dZ=0), (dR=0, dZ=2) and (dR=1, dZ=1) possible!')

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

    Units are those defned in the parameter file (:code:`struphy units -h`).

    Parameters
    ----------
    a : float
        Minor radius of torus (default: 0.361925).
    R0 : float
        Major radius of torus (default: 1.).
    B0 : float
        On-axis (r=0) toroidal magnetic field (default: 1.).
    q0 : float
        Safety factor at r=0 (default: 0.6).
    q1 : float
        Safety factor at r=a (default: 2.5).
    q0p : float
        Derivative of safety factor at r=0 (w.r.t. poloidal flux function, default: 0.78).
    q1p : float
        Derivative of safety factor at r=a (w.r.t. poloidal flux function, default: 5.00).
    n1 : float
        1st shape factor for ion number density profile (default: 0.).
    n2 : float
        2nd shape factor for ion number density profile (default: 0.).
    na : float
        Ion number density at r=a (default: 1.).
    beta : float
        On-axis (r=0) plasma beta (ratio of kinematic pressure to B^2/(2*mu0), default: 0.1).
    p1 : float
        Shape factor for pressure profile, see docstring (default: 0.25).
    psi_k : int
        Spline degree to be used for interpolation of poloidal flux function (default=3).
    psi_nel : int
        Number of cells to be used for interpolation of poloidal flux function (default=50).

    Note
    ----
    In the parameter .yml, use the following in the section `mhd_equilibrium`::

        mhd_equilibrium :
            type : AdhocTorusQPsi
            AdhocTorusQPsi :
                a       : 0.361925 # minor radius
                R0      : 1.   # major radius
                B0      : 1.   # on-axis toroidal magnetic field
                q0      : 0.6  # safety factor at r=0
                q1      : 2.5  # safety factor at r=a
                q0p     : 0.78 # derivative of safety factor at r=0 (w.r.t. to poloidal flux function)
                q1p     : 5.00 # derivative of safety factor at r=a (w.r.t. to poloidal flux function)
                n1      : 0.   # shape factor for number density profile 
                n2      : 0.   # shape factor for number density profile 
                na      : 1.   # number density at r=a
                beta    : .1   # plasma beta = p*(2*mu_0)/B^2 for flat safety factor
                p1      : 0.25 # shape factor of pressure profile
                psi_k   : 3    # spline degree to be used for interpolation of poloidal flux function
                psi_nel : 50   # number of cells to be used for interpolation of poloidal flux functionq_kind=1)
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
                self.params['B0']**2 / 2. * np.exp(-psi_norm/p1)
        else:
            out = -self.params['beta']*self.params['B0']**2 / 2. * \
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
    units : dict
        All Struphy units. If None, no rescaling of EQDSK output is performed.
    rel_path : bool
        Whether file is relative to "<struphy_path>/fields_background/mhd_equil/eqdsk/data/", or is an absolute path (default: True).
    file : str
        Path to eqdsk file (default: "AUGNLED_g031213.00830.high").
    data_type : int
        0: there is no space between data, 1: there is space between data (default: 0).
    p_for_psi : list[int]
        Spline degrees in (R, Z) directions used for interpolation of psi data (default: [3, 3]).
    psi_resolution : list[float]
        Resolution of psi data in (R, Z) directions in %, e.g. [50., 50.] uses every second psi data point (default: [25., 6.25]).
    p_for_flux : int
        Spline degree in psi direction used for interpolation of 1d functions that depend on psi: f=f(psi) (default: 3).
    flux_resolution : float
        Resolution of 1d f=f(psi) data in %, e.g. 25. uses every forth data point (default: 50.).
    n1 : float
        1st shape factor for ion number density profile n = n(psi) (default: 0.).
    n2 : float
        2nd shape factor for ion number density profile n = n(psi) (default: 0.).
    na : float
        Ion number density at plasma boundary (default: 1.).

    Note
    ----
    In the parameter .yml, use the following in the section `mhd_equilibrium`::

        mhd_equilibrium :
            type : EQDSKequilibrium
            EQDSKequilibrium :
                rel_path        : True # whether eqdsk file path relative to "<struphy_path>/fields_background/mhd_equil/eqdsk/data/", or the absolute path
                file            : 'AUGNLED_g031213.00830.high' # path to eqdsk file
                data_type       : 0 # 0: there is no space between data, 1: there is space between data
                p_for_psi       : [3, 3]      # spline degrees used in interpolation of poloidal flux function grid data
                psi_resolution  : [25., 6.25] # resolution used in interpolation of poloidal flux function grid data in %, i.e. [100., 100.] uses all grid points
                p_for_flux      : 3   # spline degree used in interpolation of 1d functions f=f(psi) (e.g. toroidal field function)
                flux_resolution : 50. # resolution used in interpolation of of 1d functions f=f(psi) in %
                n1              : 0.  # 1st shape factor for number density profile n(psi) = (1-na)*(1 - psi_norm^n1)^n2 + na
                n2              : 0.  # 2nd shape factor for number density profile n(psi) = (1-na)*(1 - psi_norm^n1)^n2 + na
                na              : 1.  # number density at last closed flux surface
    """

    def __init__(self, units=None, **params):

        from scipy.interpolate import UnivariateSpline, RectBivariateSpline
        from scipy.optimize import minimize

        from struphy.fields_background.mhd_equil.eqdsk import readeqdsk

        import struphy

        # no rescaling if units are not provided
        if units is None:
            units = {}
            units['x'] = 1.
            units['B'] = 1.
            units['p'] = 1 / 1.25663706212e-6
            print(
                '+++WARNING+++: "units" was passed as None, no rescaling performed in EQDSK output.')

        self._units = units

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

        # normalize pressure profile to pressure unit 1 Tesla^2/mu_0
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
    def units(self):
        """ All Struphy units.
        """
        return self._units

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
        """ Poloidal flux function psi = psi(R, Z) in units T*m^2.
        """

        is_float = all(isinstance(v, (int, float)) for v in [R, Z])

        out = self._psi_i(R, Z, dx=dR, dy=dZ, grid=False)

        # remove all "dimensions" for point-wise evaluation
        if is_float:
            assert out.ndim == 0
            out = out.item()

        # rescale to Struphy units
        out /= self.units['B'] * self.units['x']**2

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

        # rescale to Struphy units
        out /= self.units['B'] * self.units['x']

        return out

    def p_xyz(self, x, y, z):
        """ Pressure p = p(x, y, z) in units 1 Tesla^2/mu_0.
        """

        R = np.sqrt(x**2 + y**2)
        Z = 1*z

        out = self.p_psi(self.psi(R, Z))

        # rescale to Struphy units
        out /= 1.25663706212e-6 * self.units['p']

        return out

    def n_xyz(self, x, y, z):
        """ Number density in physical space. Units from parameter file. 
        """

        R = np.sqrt(x**2 + y**2)
        Z = 1*z

        out = self.n_psi(self.psi(R, Z))

        return out


class GVECequilibrium(LogicalMHDequilibrium):
    """
    Numerical equilibrium via an interface to `gvec_to_python <https://gitlab.mpcdf.mpg.de/spossann/gvec_to_python>`_.

    Parameters
    ----------
    rel_path : bool
        Whether dat_file (json_file) are relative to "<struphy_path>/fields_background/mhd_equil/gvec/", or are absolute paths (default: True).
    dat_file : str
        Path to .dat file (default: "/ellipstell_v2/newBC_E1D6_M6N6/GVEC_ELLIPSTELL_V2_State_0000_00200000.dat").    
    json_file : str
        Path to .json file (default: None).
    use_pest : bool
        Whether to use straigh-field line coordinates (PEST) (default: False).
    use_nfp : bool
        Whether the field periods of the stellarator should be used in the mapping, i.e. phi = 2*pi*eta3 / nfp (piece of cake) (default: True).
    rmin : float
        Between [0, 1), radius (in logical space) of the domian hole around the magnetic axis (default: rmin=0.0).
    Nel : tuple[int]
        Number of cells in each direction used for interpolation of the mapping (default: (16, 16, 16)).   
    p : tuple[int]
        Spline degree in each direction used for interpolation of the mapping (default: (3, 3, 3)).

    Note
    ----
    In the parameter .yml, use the following in the section `mhd_equilibrium`::

        mhd_equilibrium :
            type : GVECequilibrium
            GVECequilibrium : 
                rel_path : True # whether file path is relative to "<struphy_path>/fields_background/mhd_equil/gvec/", or the absolute path
                dat_file : '/ellipstell_v2/newBC_E1D6_M6N6/GVEC_ELLIPSTELL_V2_State_0000_00200000.dat' # path to gvec .dat output file 
                json_file : null # give directly the parsed json file, if it exists (then dat_file is not used)
                use_pest : False # whether to use straight-field line coordinates (PEST)
                use_nfp : True # whether to use the field periods of the stellarator in the mapping, i.e. phi = 2*pi*eta3 / nfp (piece of cake).
                rmin : 0.0 # radius of domain hole around magnetic axis.
                Nel : [32, 32, 32] # number of cells in each direction used for interpolation of the mapping.
                p : [3, 3, 3] # spline degree in each direction used for interpolation of the mapping.
    """

    def __init__(self, show=False, **params):

        from struphy.geometry.domains import GVECunit

        from gvec_to_python.reader.gvec_reader import create_GVEC_json
        from gvec_to_python import GVEC

        import struphy
        import os
        from mpi4py import MPI
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        params_default = {'rel_path': True,
                          'dat_file': 'ellipstell_v2/newBC_E1D6_M6N6/GVEC_ELLIPSTELL_V2_State_0000_00200000.dat',
                          'json_file': None,
                          'use_pest': False,
                          'use_nfp': True,
                          'rmin': 0.0,
                          'Nel': (16, 16, 16),
                          'p': (3, 3, 3), }

        self._params = set_defaults(params, params_default)

        if self._params['dat_file'] is None:

            assert self._params['json_file'] is not None
            assert self._params['json_file'][-5:] == '.json'

            if self._params['rel_path']:
                json_file = os.path.join(struphy.__path__[0], 'fields_background/mhd_equil/gvec', self._params['json_file'])
            else:
                json_file = self._params['json_file']

        else:

            assert self._params['dat_file'][-4:] == '.dat'

            if self._params['rel_path']:
                dat_file = os.path.join(struphy.__path__[0], 'fields_background/mhd_equil/gvec', self._params['dat_file'])
            else:
                dat_file = params['dat_file']

            json_file = dat_file[:-4] + '.json'
            #TODO: better read/write for MPI
            if rank == 0:
                create_GVEC_json(dat_file, json_file)
            comm.Barrier()

        if self._params['use_pest']:
            mapping = 'unit_pest'
        else:
            mapping = 'unit'

        if self._params['use_nfp']:
            unit_tor_domain = "one-fp"
        else:
            unit_tor_domain = "full"

        # gvec object
        self._gvec = GVEC(json_file, mapping=mapping,
                          unit_tor_domain=unit_tor_domain, use_pyccel=True)

        # struphy domain object
        self._domain = GVECunit(self)

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

    def gradB1(self, eta1, eta2, eta3, squeeze_out=True):
        """1-form gradient of magnetic field on logical cube [0, 1]^3.
        """
        raise NotImplementedError(
            '1-form gradient of magnetic field of GVECequilibrium is not implemented')


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
