"Analytical callables needed for the simulation of the Two-Fluid Quasi-Neutral Model by Restelli."

import numpy as np


class RestelliForcingTerm:
    r"""Non-zero Force term :math:`f` on the right-hand-side of:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,

    defined as follows:

    .. math::

        f = \nu \omega \,,
        \\[2mm]
        \omega = \left[0, \alpha \frac{R_0 - 4R}{a R_0 R} - \beta \frac{B_p}{B_0}\frac{R_0^2}{a R^3}, 0 \right] \,,
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.

    Can only be defined in Cartesian coordinates. This class returns only the non-zero term in the second component.
    The system and solution were taken from Finite element discretization of a Stokes-like model arising in plasma physics

    Note
    ----
    In the parameter .yml, use the following template in the section ``fluid/<mhd>``::

        options:
            TwoFluidQuasiNeutralFull:
                nu: 1.      # viscosity
                nu_e: 0.01  # viscosity electrons
                a: 1.       # minor radius
                R0: 2.      # major radius
                B0: 10.     # on-axis toroidal magnetic field
                Bp: 12.5    # poloidal magnetic field
                alpha: 0.1
                beta: 1.
    References
    ----------
    [1] Juan Vicente Gutiérrez-Santacreu, Omar Maj, Marco Restelli: Finite element discretization of a Stokes-like model arising
    in plasma physics, Journal of Computational Physics 2018.
    """

    def __init__(self, nu=1.0, R0=2.0, a=1.0, B0=10.0, Bp=12.5, alpha=0.1, beta=1.0, eps=1.0):
        r"""
        Parameters
        ----------
        nu  : 1.    # viscosity

        a   : 1.    # minor radius

        R0  : 2.    # major radius

        B0  : 10.   # on-axis toroidal magnetic field

        Bp  : 12.5  # poloidal magnetic field

        alpha: 0.1

        beta: 1.

        eps: 1.     # Normalization
        """

        self._nu = nu
        self._R0 = R0
        self._a = a
        self._B0 = B0
        self._Bp = Bp
        self._alpha = alpha
        self._beta = beta
        self._eps_norm = eps

    def __call__(self, x, y, z):
        R = np.sqrt(x**2 + y**2)
        R = np.where(R == 0.0, 1e-9, R)
        phi = np.arctan2(-y, x)
        force_Z = self._nu * (
            self._alpha * (self._R0 - 4 * R) / (self._a * self._R0 * R)
            - self._beta * self._Bp * self._R0**2 / (self._B0 * self._a * R**3)
        )

        return force_Z


class ManufacturedSolutionForceterm:
    r"""Right-hand-side force terms :math:`f` and :math:`f_e` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    In 1D it is defined as follows: 

    .. math::

        f = \left[\begin{array}{c} 2 \pi cos(2 \pi x) + \nu 4 \pi^2 sin(2\pi x) + \frac{sin(2 \pi x) + 1.0}{dt} \\ \frac{B_0}{\epsilon} (sin(2 \pi x) + 1.9) \\ 0 \end{array} \right] \,,
        \\[2mm]
        f_e = \left[\begin{array}{c} -2 \pi cos(2 \pi x) + \nu_e 4 \pi^2 sin(2\pi x) - \sigma sin(2 \pi x) \\ -\frac{B_0}{\epsilon} sin(2 \pi x) \\ 0 \end{array} \right] \,.

    In 2D it is defined as follows: 

    .. math::

        f = \left[\begin{array}{c} -2\pi sin(2\pi x) + \frac{B_0}{\epsilon} cos(2\pi x)cos(2\pi y) - \nu 8 \pi^2 sin(2\pi x)sin(2\pi y) \\ 2\pi cos(2\pi y) - \frac{B_0}{\epsilon} sin(2\pi x)sin(2\pi y) - \nu 8 \pi^2 cos(2\pi x)cos(2\pi y) \\ 0 \end{array} \right] \,,
        \\[2mm]
        f_e = \left[\begin{array}{c} 2\pi sin(2\pi x) -\frac{B_0}{\epsilon} cos(4\pi x)cos(4\pi y) - \nu_e 32 \pi^2 sin(4\pi x)sin(4\pi y) + \sigma sin(4\pi x) sin(4 \pi y) \\ -2\pi cos(2\pi y) +\frac{B_0}{\epsilon} sin(4\pi x)sin(4\pi y) - \nu_e 32 \pi^2 cos(4\pi x)cos(4\pi y) + \sigma cos(4\pi x) cos(4 \pi y) \\ 0 \end{array} \right] \,.

    In Tokamak geometry it is defined as follows: 

    .. math::

        f = \left[\begin{array}{c} \alpha \frac{B_0}{a}(R-R_0) - \alpha \frac{1}{a R_0} \frac{R_0 B_0 Z}{R} + \nu \alpha \frac{1}{a R_0} \frac{R_0}{R^2}   \\
                 \alpha \frac{1}{a R_0} (R-R_0) \frac{R_0 B_0}{R} + \alpha \frac{B_0Z}{a} \\
                \alpha \frac{1}{a R_0} \frac{R_0 B_p}{a R^2} \left( (R-R_0)^2 + Z^2\right)  \end{array} \right] \,, 
        \\[2mm]
        f = \left[\begin{array}{c} -\alpha \frac{B_0}{a}(R-R_0) + \alpha \frac{1}{a R_0} \frac{R_0 B_0 Z}{R} + \nu_e \alpha \frac{1}{a R_0} \frac{R_0}{R^2}   \\
                 -\alpha \frac{1}{a R_0} (R-R_0) \frac{R_0 B_0}{R} - \alpha \frac{B_0 Z}{a} \\
                -\alpha \frac{1}{a R_0} \frac{ R_0 B_p}{a R^2} \left( (R-R_0)^2 + Z^2\right)  \end{array} \right] \,, 
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.


    Can only be defined in Cartesian coordinates. 
    """

    def __init__(
        self,
        species,
        comp,
        dimension,
        stab_sigma,
        eps,
        dt,
        b0=1.0,
        nu=1.0,
        nu_e=0.01,
        R0=2.0,
        a=1.0,
        Bp=12.5,
        alpha=0.1,
        beta=1.0,
    ):
        """
            Parameters
        ----------
        species : string
            'Ions' or 'Electrons'.
        comp : string
            Which component of the solution ('0', '1' or '2').
        dimension: string
            Defines the manufactured solution to be selected ('1D' or '2D').
        stab_sigma : float
            Stabilization parameter.
        eps : float
            Normalization parameter.
        dt : float
            Time step.
        b0 : float
            Magnetic field (default: 1.0).
        nu  : float
            Viscosity of ions (default: 1.0).
        nu_e  : float
            Viscosity of electrons (default: 0.01).
        R0 : float
            Major radius of torus (default: 2.).
        a : float
            Minor radius of torus (default: 1.).
        Bp : float
            Poloidal magnetic field (default: 12.5).
        alpha : float
            (default: 0.1)
        beta : float
            (default: 1.0)
        """

        self._B0 = b0
        self._nu = nu
        self._nu_e = nu_e
        self._comp = comp
        self._species = species
        self._dimension = dimension
        self._eps_norm = eps
        self._stab_sigma = stab_sigma
        self._dt = dt
        self._R0 = R0
        self._Bp = Bp
        self._alpha = alpha
        self._beta = beta
        self._a = a

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        A = self._alpha / (self._a * self._R0)
        C = self._beta * self._Bp * self._R0 / (self._B0 * self._a)
        R = np.sqrt(x**2 + y**2)
        R = np.where(R == 0.0, 1e-9, R)
        phi = np.arctan2(-y, x)
        if self._species == "Ions":
            """Forceterm for ions on the right hand side."""
            if self._dimension == "2D":
                fx = (
                    -2.0 * np.pi * np.sin(2 * np.pi * x)
                    + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * self._B0 / self._eps_norm
                    - self._nu * 8.0 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
                )
                fy = (
                    2.0 * np.pi * np.cos(2 * np.pi * y)
                    - np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * self._B0 / self._eps_norm
                    - self._nu * 8.0 * np.pi**2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
                )
                fz = 0.0 * x

            elif self._dimension == "1D":
                fx = (
                    2.0 * np.pi * np.cos(2 * np.pi * x)
                    + self._nu * 4.0 * np.pi**2 * np.sin(2 * np.pi * x)
                    + (np.sin(2 * np.pi * x) + 1.0) / self._dt
                )
                fy = (np.sin(2 * np.pi * x) + 1.0) * self._B0 / self._eps_norm
                fz = 0.0 * x

            elif self._dimension == "Tokamak":
                # Covariant basis with transfo DF u withoud phi dependency
                fR = (
                    self._alpha * self._B0 / self._a * (R - self._R0)
                    - A * self._R0 / R * (z * self._B0)
                    + self._nu * A / R**2 * self._R0
                )
                fZ = self._alpha * self._B0 * z / self._a + A * self._R0 / R * ((R - self._R0) * self._B0)
                fphi = A * self._R0 * self._Bp / (self._a * R**2) * ((R - self._R0) ** 2 + z**2)

                fx = np.cos(phi) * fR - R * np.sin(phi) * fphi
                fy = -np.sin(phi) * fR - R * np.cos(phi) * fphi
                fz = fZ

            if self._comp == "0":
                return fx
            elif self._comp == "1":
                return fy
            elif self._comp == "2":
                return fz
            else:
                raise ValueError(f"Invalid component '{self._comp}'. Must be '0', '1', or '2'.")

        elif self._species == "Electrons":
            """Forceterm for electrons on the right hand side."""
            if self._dimension == "2D":
                fx = (
                    2.0 * np.pi * np.sin(2 * np.pi * x)
                    - np.cos(4 * np.pi * x) * np.cos(4 * np.pi * y) * self._B0 / self._eps_norm
                    - self._nu_e * 32.0 * np.pi**2 * np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y)
                    - self._stab_sigma * (-np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y))
                )
                fy = (
                    -2.0 * np.pi * np.cos(2 * np.pi * y)
                    + np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y) * self._B0 / self._eps_norm
                    - self._nu_e * 32.0 * np.pi**2 * np.cos(4 * np.pi * x) * np.cos(4 * np.pi * y)
                    - self._stab_sigma * (-np.cos(4 * np.pi * x) * np.cos(4 * np.pi * y))
                )
                fz = 0.0 * x

            elif self._dimension == "1D":
                fx = (
                    -2.0 * np.pi * np.cos(2 * np.pi * x)
                    + self._nu_e * 4.0 * np.pi**2 * np.sin(2 * np.pi * x)
                    - self._stab_sigma * np.sin(2 * np.pi * x)
                )
                fy = -np.sin(2 * np.pi * x) * self._B0 / self._eps_norm
                fz = 0.0 * x

            elif self._dimension == "Tokamak":
                # Covariant basis with transfo DF u Solution without phi dependency
                fR = (
                    -self._alpha * self._B0 / self._a * (R - self._R0)
                    + A * self._R0 / R * (z * self._B0)
                    + self._nu_e * A * self._R0 / R**2
                )
                fZ = -self._alpha * self._B0 * z / self._a - A * self._R0 / R * ((R - self._R0) * self._B0)
                fphi = -A * self._R0 * self._Bp / (self._a * R**2) * ((R - self._R0) ** 2 + z**2)

                fx = np.cos(phi) * fR - R * np.sin(phi) * fphi
                fy = -np.sin(phi) * fR - R * np.cos(phi) * fphi
                fz = fZ

            if self._comp == "0":
                return fx
            elif self._comp == "1":
                return fy
            elif self._comp == "2":
                return fz
            else:
                raise ValueError(f"Invalid component '{self._comp}'. Must be '0', '1', or '2'.")

        else:
            raise ValueError(f"Invalid species '{self._species}'. Must be 'Ions' or 'Electrons'.")
