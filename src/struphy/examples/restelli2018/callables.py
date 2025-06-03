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
            Stokes:
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

    def __init__(self, nu=1.0, R0=2.0, a=1.0, B0=10.0, Bp=12.5, alpha=0.1, beta=1.0):
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
        """

        self._nu = nu
        self._R0 = R0
        self._a = a
        self._B0 = B0
        self._Bp = Bp
        self._alpha = alpha
        self._beta = beta

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

    defined as follows: 

    .. math::

        f = \left[1 - b_0 cos(x) - \nu sin(y), 1 - b_0 sin(y) + \nu cos(x) , 0 \right] \,, 
        \\[2mm]
        f_e = \left[-1 + 0.5 b_0 cos(x) - \nu_e 0.5 sin(y), -1 + 0.5 b_0 sin(y) + \nu_e cos(x) , 0 \right] \,.

    Can only be defined in Cartesian coordinates. 
    """

    def __init__(self, species, comp, b0=1.0, nu=1.0, nu_e=0.01):
        """
            Parameters
        ----------
        species : string
            'Ions' or 'Electrons'.
        comp : string
            Which component of the solution ('0', '1' or '2').
        b0 : float
            Magnetic field (default: 1.0).
        nu  : float
            Viscosity of ions (default: 1.0)
        nu_e  : float
            Viscosity of electrons (default: 0.01)
        """

        self._b = b0
        self._nu = nu
        self._nu_e = nu_e
        self._comp = comp
        self._species = species

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        if self._species == "Ions":
            """Forceterm for ions on the right hand side."""
            """x component"""
            # fx = (
            #     -2.0 * np.pi * np.sin(2 * np.pi * x)
            #     + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * self._b
            #     - self._nu * 8.0 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
            # )
            fx = (
                2.0 * np.pi * np.cos(2 * np.pi * x)
                + self._nu * 4.0 * np.pi**2 * np.sin(2 * np.pi * x)  # + (np.sin(2 * np.pi * x)) / 0.1
            )
            # fx = (
            #     2.0 * np.pi * np.cos(2 * np.pi * x)
            #     - self._b * np.cos(2 * np.pi * x)
            #     + self._nu * 4.0 * np.pi**2 * np.sin(2 * np.pi * x)
            # )

            # fx = 2.0 * np.pi * np.cos(2 * np.pi * x) + (np.sin(2 * np.pi * x) + 1.0) / 0.1

            """y component"""
            # fy = (
            #     2.0 * np.pi * np.cos(2 * np.pi * y)
            #     - np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * self._b
            #     - self._nu * 8.0 * np.pi**2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
            # )
            fy = (
                 (np.sin(2 * np.pi * x) + 1.0)  * self._b
            )
            # fy = (
            #     + np.sin(2 * np.pi * x)  * self._b
            #     + self._nu * 4.0 * np.pi**2 * np.cos(2 * np.pi * x)
            # )

            # fy = 0.0 * x

            """z component"""
            fz = 0.0 * x

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
            """x component"""
            # fx = (
            #     2.0 * np.pi * np.sin(2 * np.pi * x)
            #     - np.cos(4 * np.pi * x) * np.cos(4 * np.pi * y) * self._b
            #     - self._nu_e * 32.0 * np.pi**2 * np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y)
            # )
            fx = (
                -2.0 * np.pi * np.cos(2 * np.pi * x)
                + self._nu_e * 4.0 * np.pi**2 * np.sin(2 * np.pi * x) - 0.001*np.sin(2 * np.pi * x)
            )
            # fx = (
            #     -2.0 * np.pi * np.cos(2 * np.pi * x)
            #     + np.cos(2 * np.pi * x) * self._b
            #     + self._nu_e * 4.0 * np.pi**2 * np.sin(2 * np.pi * x)
            # )

            # fx = -2.0 * np.pi * np.cos(2 * np.pi * x) - 1e-3 * np.sin(2 * np.pi * x)

            """y component"""
            # fy = (
            #     -2.0 * np.pi * np.cos(2 * np.pi * y)
            #     + np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y) * self._b
            #     - self._nu_e * 32.0 * np.pi**2 * np.cos(4 * np.pi * x) * np.cos(4 * np.pi * y)
            # )
            fy = (
                - np.sin(2 * np.pi * x) * self._b
            )
            # fy = (
            #     - np.sin(2 * np.pi * x) * self._b
            #     + 4.0 * np.pi**2 * np.cos(2 * np.pi * x) * self._nu_e
            # )

            # fy = 0.0 * x

            """z component"""
            fz = 0.0 * x

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
