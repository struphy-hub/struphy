#!/usr/bin/env python3
"Analytical perturbations (modes)."

import scipy
import scipy.special

from struphy.utils.arrays import xp as np


class ModesSin:
    r"""Sinusoidal function in 3D.

    .. math::

        u(x, y, z) =  \sum_{s} \chi_s(z) A_s \sin \left(l_s \frac{2\pi}{L_x} x + m_s \frac{2\pi}{L_y} y + n_s \frac{2\pi}{L_z} z + \theta_s \right) \,.

    where :math:`\chi_s(z)` is one of

    .. math::

        \chi_s(z) = \left\{
        \begin{aligned}
        1\,,
        \\[2mm]
         \tanh((z - 0.5)/\delta)/\cosh((z - 0.5)/\delta)\,,
        \end{aligned}
        \right.

    Can be used in logical space, where :math:`x \to \eta_1,\, y\to \eta_2,\, z \to \eta_3`
    and :math:`L_x=L_y=L_z=1.0` (default).

    Parameters
    ----------
    ls : tuple | list
        Mode numbers in x-direction (kx = l*2*pi/Lx).

    ms : tuple | list
        Mode numbers in y-direction (ky = m*2*pi/Ly).

    ns : tuple | list
        Mode numbers in z-direction (kz = n*2*pi/Lz).

    amps : tuple | list
        Amplitude of each mode.

    theta : tuple | list
        Phase of each mode

    pfuns : tuple | list[str]
        "Id" or "localize" define the profile functions.
        localize multiply the sinus by :math: `tanh((\eta_3 - 0.5)/\delta)/cosh((\eta_3 - 0.5)/\delta)`
        to localize it around 0.5. :math: `\delta` is given by the input parameter pfuns_params

    pfuns_params : tuple | list
        The parameter needed by the profile function

    Lx, Ly, Lz : float
        Domain lengths.

    Note
    ----
    Example of use in a ``.yml`` parameter file::

        perturbations :
            type : ModesSin
            ModesSin :
                comps :
                    scalar_name : '0' # choices: null, 'physical', '0', '3'
                    vector_name : [null , 'v', '2']  # choices: null, 'physical', '1', '2', 'v', 'norm'
                ls :
                    scalar_name: [1, 3] # two x-modes for scalar variable
                    vector_name: [null, [0, 1], [4]] # two x-modes for 2nd comp. and one x-mode for third component of vector-valued variable
                theta :
                    scalar_name: [0, 3.1415]
                    vector_name: [null, [0, 0], [1.5708]]
                pfuns :
                    vector_name: [null, ['localize'], ['Id']]
                pfuns_params
                    vector_name: [null, ['0.1'], [0.]]
                Lx : 7.853981633974483
                Ly : 1.
                Lz : 1.
    """

    def __init__(
        self,
        ls=None,
        ms=None,
        ns=None,
        amps=(1e-4,),
        theta=None,
        pfuns=("Id",),
        pfuns_params=(0.0,),
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
    ):
        if ls is not None:
            n_modes = len(ls)
        elif ms is not None:
            n_modes = len(ms)
            ls = [0] * n_modes
        elif ns is not None:
            n_modes = len(ns)
            ls = [0] * n_modes
            ms = [0] * n_modes
        else:
            n_modes = 1
            ls = [0]
            ms = [0]
            ns = [0]

        if ms is None:
            ms = [0] * n_modes
        else:
            assert len(ms) == n_modes

        if ns is None:
            ns = [0] * n_modes
        else:
            assert len(ns) == n_modes

        if len(amps) == 1:
            amps = [amps[0]] * n_modes
        else:
            assert len(amps) == n_modes

        if theta is None:
            theta = [0] * n_modes

        if len(theta) == 1:
            theta = [theta[0]] * n_modes
        else:
            assert len(theta) == n_modes

        if len(pfuns) == 1:
            pfuns = [pfuns[0]] * n_modes
        else:
            assert len(pfuns) == n_modes

        if len(pfuns_params) == 1:
            pfuns_params = [pfuns_params[0]] * n_modes
        else:
            assert len(pfuns_params) == n_modes

        self._ls = ls
        self._ms = ms
        self._ns = ns
        self._amps = amps
        self._Lx = Lx
        self._Ly = Ly
        self._Lz = Lz
        self._theta = theta

        self._pfuns = []
        for pfun, params in zip(pfuns, pfuns_params):
            if pfun == "Id":
                self._pfuns += [lambda eta3: 1.0]
            elif pfun == "localize":
                self._pfuns += [
                    lambda eta3: np.tanh((eta3 - 0.5) / params) / np.cosh((eta3 - 0.5) / params),
                ]
            else:
                raise ValueError(f"Profile function {pfun} is not defined..")

    def __call__(self, x, y, z):
        val = 0.0

        for amp, l, m, n, t, pfun in zip(self._amps, self._ls, self._ms, self._ns, self._theta, self._pfuns):
            val += (
                amp
                * pfun(z)
                * np.sin(
                    l * 2.0 * np.pi / self._Lx * x
                    + m * 2.0 * np.pi / self._Ly * y
                    + n * 2.0 * np.pi / self._Lz * z
                    + t,
                )
            )

        return val


class ModesCos:
    r"""Cosinusoidal function in 3D.

    .. math::

        u(x, y, z) = \sum_{s} A_s \cos \left(l_s \frac{2\pi}{L_x} x + m_s \frac{2\pi}{L_y} y + n_s \frac{2\pi}{L_z} z \right) \,.

    Can be used in logical space, where :math:`x \to \eta_1,\, y\to \eta_2,\, z \to \eta_3`
    and :math:`L_x=L_y=L_z=1.0` (default).

    Parameters
    ----------
    ls : tuple | list
        Mode numbers in x-direction (kx = l*2*pi/Lx).

    ms : tuple | list
        Mode numbers in y-direction (ky = m*2*pi/Ly).

    ns : tuple | list
        Mode numbers in z-direction (kz = n*2*pi/Lz).

    amps : tuple | list
        Amplitude of each mode.

    Lx, Ly, Lz : float
        Domain lengths.

    Note
    ----
    Example of use in a ``.yml`` parameter file::

        perturbations :
            type : ModesCos
            ModesCos :
                comps :
                    scalar_name : '0' # choices: null, 'physical', '0', '3'
                    vector_name : [null , 'v', '2']  # choices: null, 'physical', '1', '2', 'v', 'norm'
                ls :
                    scalar_name: [1, 3] # two x-modes for scalar variable
                    vector_name: [null, [0, 1], [4]] # two x-modes for 2nd comp. and one x-mode for third component of vector-valued variable
                Lx : 7.853981633974483
                Ly : 1.
                Lz : 1.
    """

    def __init__(self, ls=None, ms=None, ns=None, amps=(1e-4,), Lx=1.0, Ly=1.0, Lz=1.0):
        if ls is not None:
            n_modes = len(ls)
        elif ms is not None:
            n_modes = len(ms)
            ls = [0] * n_modes
        elif ns is not None:
            n_modes = len(ns)
            ls = [0] * n_modes
            ms = [0] * n_modes
        else:
            n_modes = 1
            ls = [0]
            ms = [0]
            ns = [0]

        if ms is None:
            ms = [0] * n_modes
        else:
            assert len(ms) == n_modes

        if ns is None:
            ns = [0] * n_modes
        else:
            assert len(ns) == n_modes

        if len(amps) == 1:
            amps = [amps[0]] * n_modes
        else:
            assert len(amps) == n_modes

        self._ls = ls
        self._ms = ms
        self._ns = ns
        self._amps = amps
        self._Lx = Lx
        self._Ly = Ly
        self._Lz = Lz

    def __call__(self, x, y, z):
        val = 0.0

        for amp, l, m, n in zip(self._amps, self._ls, self._ms, self._ns):
            val += amp * np.cos(
                l * 2.0 * np.pi / self._Lx * x + m * 2.0 * np.pi / self._Ly * y + n * 2.0 * np.pi / self._Lz * z,
            )
        # print( "Cos max value", val.max())
        return val


class CoaxialWaveguideElectric_r:
    r"""Initializes function for Coaxial Waveguide electric field in radial direction.

    Solutions taken from TUM master thesis of Alicia Robles Pérez:
    "Development of a Geometric Particle-in-Cell Method for Cylindrical Coordinate Systems", 2024

    Parameters
    ----------
    m : int
        Number of Modes
    a1, a2 : float
        inner and outer radius of Hollow Cylinder
    a, b : float
        Parameters of Electric field
    """

    def __init__(self, m=1, a1=1.0, a2=2.0, a=1, b=-0.28):
        self._m = m
        self._r1 = a1
        self._r2 = a2
        self._a = a
        self._b = b

    def __call__(self, eta1, eta2, eta3):
        val = 0.0
        r = eta1 * (self._r2 - self._r1) + self._r1
        theta = eta2 * 2.0 * np.pi

        val += (
            -self._m
            / r
            * np.cos(self._m * theta)
            * (self._a * scipy.special.jv(self._m, r) + self._b * scipy.special.yn(self._m, r))
        )
        return val


class CoaxialWaveguideElectric_theta:
    r"""
    Initializes funtion for Coaxial Waveguide electric field in the azimuthal direction.

    Solutions taken from TUM master thesis of Alicia Robles Pérez:
    "Development of a Geometric Particle-in-Cell Method for Cylindrical Coordinate Systems", 2024

    Parameters
    ----------
    m : int
        Number of Modes
    a1, a2 : float
        inner and outer radius of Hollow Cylinder
    a, b : float
        Parameters of Electric field
    """

    def __init__(self, m=1, a1=1.0, a2=2.0, a=1, b=-0.28):
        self._m = m
        self._r1 = a1
        self._r2 = a2
        self._a = a
        self._b = b

    def __call__(self, eta1, eta2, eta3):
        val = 0.0
        r = eta1 * (self._r2 - self._r1) + self._r1
        theta = eta2 * 2.0 * np.pi

        val += (
            self._a * ((self._m / r) * scipy.special.jv(self._m, r) - scipy.special.jv(self._m + 1, r))
            + (self._b * ((self._m / r) * scipy.special.yn(self._m, r) - scipy.special.yn(self._m + 1, r)))
        ) * np.sin(self._m * theta)
        return val


class CoaxialWaveguideMagnetic:
    r"""Initializes funtion for Coaxial Waveguide magnetic field in $z$-direction.

    Solutions taken from TUM master thesis of Alicia Robles Pérez:
    "Development of a Geometric Particle-in-Cell Method for Cylindrical Coordinate Systems", 2024

    Parameters
    ----------
    m : int
        Number of Modes
    a1, a2 : float
        inner and outer radius of Hollow Cylinder
    a, b : float
        Parameters of Electric field
    """

    def __init__(self, m=1, a1=1.0, a2=2.0, a=1, b=-0.28):
        self._m = m
        self._r1 = a1
        self._r2 = a2
        self._a = a
        self._b = b

    def __call__(self, eta1, eta2, eta3):
        val = 0.0
        r = eta1 * (self._r2 - self._r1) + self._r1
        theta = eta2 * 2.0 * np.pi
        z = eta3

        val += (self._a * scipy.special.jv(self._m, r) + self._b * scipy.special.yn(self._m, r)) * np.cos(
            self._m * theta
        )
        return val


class ModesCosCos:
    r"""

    .. math::

        u(x, y, z) = \sum_s A_s \, \chi_s(z)
        \cos \!\left(l_s \tfrac{2\pi}{L_x} x + \theta_{x,s}\right)
        \cos \!\left(m_s \tfrac{2\pi}{L_y} y + \theta_{y,s}\right)

    where :math:`\chi_s(z)` can be either 1 or localized in z.

    """

    def __init__(
        self,
        ls=None,
        ms=None,
        amps=(1e-4,),
        theta_x=None,
        theta_y=None,
        pfuns=("Id",),
        pfuns_params=(0.0,),
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
    ):
        if ls is not None:
            n_modes = len(ls)
        elif ms is not None:
            n_modes = len(ms)
            ls = [0] * n_modes
        else:
            n_modes = 1
            ls = [0]
            ms = [0]

        if ms is None:
            ms = [0] * n_modes
        else:
            assert len(ms) == n_modes

        if len(amps) == 1:
            amps = [amps[0]] * n_modes
        else:
            assert len(amps) == n_modes

        if theta_x is None:
            theta_x = [0] * n_modes
        elif len(theta_x) == 1:
            theta_x = [theta_x[0]] * n_modes
        else:
            assert len(theta_x) == n_modes

        if theta_y is None:
            theta_y = [0] * n_modes
        elif len(theta_y) == 1:
            theta_y = [theta_y[0]] * n_modes
        else:
            assert len(theta_y) == n_modes

        if len(pfuns) == 1:
            pfuns = [pfuns[0]] * n_modes
        else:
            assert len(pfuns) == n_modes

        if len(pfuns_params) == 1:
            pfuns_params = [pfuns_params[0]] * n_modes
        else:
            assert len(pfuns_params) == n_modes

        self._ls = ls
        self._ms = ms
        self._amps = amps
        self._theta_x = theta_x
        self._theta_y = theta_y
        self._Lx = Lx
        self._Ly = Ly
        self._Lz = Lz

        self._pfuns = []
        for pfun, params in zip(pfuns, pfuns_params):
            if pfun == "Id":
                self._pfuns += [lambda z: 1.0]
            elif pfun == "localize":
                self._pfuns += [lambda z, p=params: np.tanh((z - 0.5) / p) / np.cosh((z - 0.5) / p)]
            else:
                raise ValueError(f"Profile function {pfun} is not defined..")

    def __call__(self, x, y, z):
        val = 0.0
        for amp, l, m, thx, thy, pfun in zip(self._amps, self._ls, self._ms, self._theta_x, self._theta_y, self._pfuns):
            val += (
                amp
                * pfun(z)
                * np.cos(l * 2.0 * np.pi / self._Lx * x + thx)
                * np.cos(m * 2.0 * np.pi / self._Ly * y + thy)
            )
        return val


class ModesSinSin:
    r"""

    .. math::

        u(x, y, z) = \sum_s A_s \, \chi_s(z)
        \sin \!\left(l_s \tfrac{2\pi}{L_x} x + \theta_{x,s}\right)
        \sin \!\left(m_s \tfrac{2\pi}{L_y} y + \theta_{y,s}\right)

    where :math:`\chi_s(z)` can be either 1 or localized in z.
    """

    def __init__(
        self,
        ls=None,
        ms=None,
        amps=(1e-4,),
        theta_x=None,
        theta_y=None,
        pfuns=("Id",),
        pfuns_params=(0.0,),
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
    ):
        if ls is not None:
            n_modes = len(ls)
        elif ms is not None:
            n_modes = len(ms)
            ls = [0] * n_modes
        else:
            n_modes = 1
            ls = [0]
            ms = [0]

        if ms is None:
            ms = [0] * n_modes
        else:
            assert len(ms) == n_modes

        if len(amps) == 1:
            amps = [amps[0]] * n_modes
        else:
            assert len(amps) == n_modes

        if theta_x is None:
            theta_x = [0] * n_modes
        elif len(theta_x) == 1:
            theta_x = [theta_x[0]] * n_modes
        else:
            assert len(theta_x) == n_modes

        if theta_y is None:
            theta_y = [0] * n_modes
        elif len(theta_y) == 1:
            theta_y = [theta_y[0]] * n_modes
        else:
            assert len(theta_y) == n_modes

        if len(pfuns) == 1:
            pfuns = [pfuns[0]] * n_modes
        else:
            assert len(pfuns) == n_modes

        if len(pfuns_params) == 1:
            pfuns_params = [pfuns_params[0]] * n_modes
        else:
            assert len(pfuns_params) == n_modes

        self._ls = ls
        self._ms = ms
        self._amps = amps
        self._theta_x = theta_x
        self._theta_y = theta_y
        self._Lx = Lx
        self._Ly = Ly
        self._Lz = Lz

        self._pfuns = []
        for pfun, params in zip(pfuns, pfuns_params):
            if pfun == "Id":
                self._pfuns += [lambda z: 1.0]
            elif pfun == "localize":
                self._pfuns += [lambda z, p=params: np.tanh((z - 0.5) / p) / np.cosh((z - 0.5) / p)]
            else:
                raise ValueError(f"Profile function {pfun} is not defined..")

    def __call__(self, x, y, z):
        val = 0.0
        for amp, l, m, thx, thy, pfun in zip(self._amps, self._ls, self._ms, self._theta_x, self._theta_y, self._pfuns):
            val += (
                amp
                * pfun(z)
                * np.sin(l * 2.0 * np.pi / self._Lx * x + thx)
                * np.sin(m * 2.0 * np.pi / self._Ly * y + thy)
            )
        return val


class ModesSinCos:
    r"""

    .. math::

        u(x, y, z) = \sum_s A_s \, \chi_s(z)
        \sin \!\left(l_s \tfrac{2\pi}{L_x} x + \theta_{x,s}\right)
        \cos \!\left(m_s \tfrac{2\pi}{L_y} y + \theta_{y,s}\right)

    where :math:`\chi_s(z)` can be either 1 or localized in z.
    """

    def __init__(
        self,
        ls=None,
        ms=None,
        amps=(1e-4,),
        theta_x=None,
        theta_y=None,
        pfuns=("Id",),
        pfuns_params=(0.0,),
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
    ):
        # number of modes
        if ls is not None:
            n_modes = len(ls)
        elif ms is not None:
            n_modes = len(ms)
            ls = [0] * n_modes
        else:
            n_modes = 1
            ls = [0]
            ms = [0]

        if ms is None:
            ms = [0] * n_modes
        else:
            assert len(ms) == n_modes

        if len(amps) == 1:
            amps = [amps[0]] * n_modes
        else:
            assert len(amps) == n_modes

        if theta_x is None:
            theta_x = [0] * n_modes
        elif len(theta_x) == 1:
            theta_x = [theta_x[0]] * n_modes
        else:
            assert len(theta_x) == n_modes

        if theta_y is None:
            theta_y = [0] * n_modes
        elif len(theta_y) == 1:
            theta_y = [theta_y[0]] * n_modes
        else:
            assert len(theta_y) == n_modes

        if len(pfuns) == 1:
            pfuns = [pfuns[0]] * n_modes
        else:
            assert len(pfuns) == n_modes

        if len(pfuns_params) == 1:
            pfuns_params = [pfuns_params[0]] * n_modes
        else:
            assert len(pfuns_params) == n_modes

        # store
        self._ls = ls
        self._ms = ms
        self._amps = amps
        self._theta_x = theta_x
        self._theta_y = theta_y
        self._Lx = Lx
        self._Ly = Ly
        self._Lz = Lz

        self._pfuns = []
        for pfun, params in zip(pfuns, pfuns_params):
            if pfun == "Id":
                self._pfuns += [lambda z: 1.0]
            elif pfun == "localize":
                self._pfuns += [lambda z, p=params: np.tanh((z - 0.5) / p) / np.cosh((z - 0.5) / p)]
            else:
                raise ValueError(f"Profile function {pfun} is not defined..")

    def __call__(self, x, y, z):
        val = 0.0
        for amp, l, m, thx, thy, pfun in zip(self._amps, self._ls, self._ms, self._theta_x, self._theta_y, self._pfuns):
            val += (
                amp
                * pfun(z)
                * np.sin(l * 2.0 * np.pi / self._Lx * x + thx)
                * np.cos(m * 2.0 * np.pi / self._Ly * y + thy)
            )
        return val


class ModesCosSin:
    r"""

    .. math::

        u(x, y, z) = \sum_s A_s \, \chi_s(z)
        \cos \!\left(l_s \tfrac{2\pi}{L_x} x + \theta_{x,s}\right)
        \sin \!\left(m_s \tfrac{2\pi}{L_y} y + \theta_{y,s}\right)

    where :math:`\chi_s(z)` can be either 1 or localized in z.
    """

    def __init__(
        self,
        ls=None,
        ms=None,
        amps=(1e-4,),
        theta_x=None,
        theta_y=None,
        pfuns=("Id",),
        pfuns_params=(0.0,),
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
    ):
        # number of modes
        if ls is not None:
            n_modes = len(ls)
        elif ms is not None:
            n_modes = len(ms)
            ls = [0] * n_modes
        else:
            n_modes = 1
            ls = [0]
            ms = [0]

        if ms is None:
            ms = [0] * n_modes
        else:
            assert len(ms) == n_modes

        if len(amps) == 1:
            amps = [amps[0]] * n_modes
        else:
            assert len(amps) == n_modes

        if theta_x is None:
            theta_x = [0] * n_modes
        elif len(theta_x) == 1:
            theta_x = [theta_x[0]] * n_modes
        else:
            assert len(theta_x) == n_modes

        if theta_y is None:
            theta_y = [0] * n_modes
        elif len(theta_y) == 1:
            theta_y = [theta_y[0]] * n_modes
        else:
            assert len(theta_y) == n_modes

        if len(pfuns) == 1:
            pfuns = [pfuns[0]] * n_modes
        else:
            assert len(pfuns) == n_modes

        if len(pfuns_params) == 1:
            pfuns_params = [pfuns_params[0]] * n_modes
        else:
            assert len(pfuns_params) == n_modes

        # store
        self._ls = ls
        self._ms = ms
        self._amps = amps
        self._theta_x = theta_x
        self._theta_y = theta_y
        self._Lx = Lx
        self._Ly = Ly
        self._Lz = Lz

        self._pfuns = []
        for pfun, params in zip(pfuns, pfuns_params):
            if pfun == "Id":
                self._pfuns += [lambda z: 1.0]
            elif pfun == "localize":
                self._pfuns += [lambda z, p=params: np.tanh((z - 0.5) / p) / np.cosh((z - 0.5) / p)]
            else:
                raise ValueError(f"Profile function {pfun} is not defined..")

    def __call__(self, x, y, z):
        val = 0.0
        for amp, l, m, thx, thy, pfun in zip(self._amps, self._ls, self._ms, self._theta_x, self._theta_y, self._pfuns):
            val += (
                amp
                * pfun(z)
                * np.cos(l * 2.0 * np.pi / self._Lx * x + thx)
                * np.sin(m * 2.0 * np.pi / self._Ly * y + thy)
            )
        return val


class TorusModesSin:
    r"""Sinusoidal function in the periodic coordinates of a Torus.

    .. math::

        u(\eta_1, \eta_2, \eta_3) = \sum_{s} \chi_s(\eta_1) A_s \sin(m_s\,2\pi \eta_2 + n_s\,2\pi \eta_3) \,,

    where :math:`\chi_s(\eta_1)` is one of

    .. math::

        \chi_s(\eta_1) = \left\{
        \begin{aligned}
        &\sin(l_s\pi\eta_1)\,,
        \\[2mm]
        &\exp \left(- \frac{(\eta_1 - r_0)^2}{2 \sigma^2} \right) \,,
        \\[2mm]
        & - \frac{\eta_1 - r_0}{\sigma} \exp \left(- \frac{(\eta_1 - r_0)^2}{2 \sigma^2} \right) \,.
        \end{aligned}
        \right.

    Can only be defined in logical coordinates.

    Parameters
    ----------
    ms : tuple | list[int]
        Poloidal mode numbers.

    ns : tuple | list[int]
        Toroidal mode numbers.

    pfuns : tuple | list[str]
        "sin" or "cos" or "exp" to define the profile functions.

    amps : tuple | list[float]
        Amplitudes of each mode (m_i, n_i).

    pfun_params : tuple | list
        Provides :math:`[r_0, \sigma]` parameters for each "exp" profile fucntion, and l_s for "sin" and "cos".

    Note
    ----
    In the parameter .yml, use the following template in the section ``fluid/<species>``::

        perturbations :
            type : TorusModesSin
            TorusModesSin :
                comps :
                    n3 : null                     # choices: null, 'physical', '0', '3'
                    u2 : ['physical', 'v', '2']   # choices: null, 'physical', '1', '2', 'v', 'norm'
                    p3 : '0'                      # choices: null, 'physical', '0', '3'
                ms :
                    n3: null            # poloidal mode numbers
                    u2: [[0], [0], [0]] # poloidal mode numbers
                    p3: [0]             # poloidal mode numbers
                ns :
                    n3: null            # toroidal mode numbers
                    u2: [[1], [1], [1]] # toroidal mode numbers
                    p3: [1]             # toroidal mode numbers
                amps :
                    n3: null                        # amplitudes of each mode
                    u2: [[0.001], [0.001], [0.001]] # amplitudes of each mode
                    p3: [0.01]                      # amplitudes of each mode
                pfuns :
                    n3: null                        # profile function in eta1-direction ('sin' or 'cos' or 'exp' or 'd_exp')
                    u2: [['sin'], ['sin'], ['exp']] # profile function in eta1-direction ('sin' or 'cos' or 'exp' or 'd_exp')
                    p3: [0.01]                      # profile function in eta1-direction ('sin' or 'cos' or 'exp' or 'd_exp')
                pfun_params :
                    n3: null                      # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and l_s for "sin" and "cos"
                    u2: [2, null, [[0.5, 1.]]]    # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and l_s for "sin" and "cos"
                    p3: [0.01]                    # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and l_s for "sin" and "cos"
    """

    def __init__(self, ms=None, ns=None, amps=(1e-4,), pfuns=("sin",), pfun_params=None):
        if ms is not None:
            n_modes = len(ms)
        elif ns is not None:
            n_modes = len(ns)
            ms = [0] * n_modes
        else:
            n_modes = 1
            ms = [1]
            ns = [0]

        if ns is None:
            ns = [0] * n_modes
        else:
            assert len(ns) == n_modes

        if len(amps) == 1:
            amps = [amps[0]] * n_modes
        else:
            assert len(amps) == n_modes

        if len(pfuns) == 1:
            pfuns = [pfuns[0]] * n_modes
        else:
            assert len(pfuns) == n_modes

        if pfun_params is None:
            pfun_params = [None] * n_modes

        self._ms = ms
        self._ns = ns
        self._amps = amps

        self._pfuns = []
        for pfun, params in zip(pfuns, pfun_params):
            if pfun == "sin":
                if params is None:
                    ls = 1
                else:
                    ls = params
                self._pfuns += [lambda eta1: np.sin(ls * np.pi * eta1)]
            elif pfun == "exp":
                self._pfuns += [
                    lambda eta1: np.exp(-((eta1 - params[0]) ** 2) / (2 * params[1] ** 2))
                    / np.sqrt(2 * np.pi * params[1] ** 2),
                ]
            elif pfun == "d_exp":
                self._pfuns += [
                    lambda eta1: -(eta1 - params[0])
                    / params[1] ** 2
                    * np.exp(-((eta1 - params[0]) ** 2) / (2 * params[1] ** 2))
                    / np.sqrt(2 * np.pi * params[1] ** 2),
                ]
            else:
                raise ValueError(f"Profile function {pfun} is not defined..")

    def __call__(self, eta1, eta2, eta3):
        val = 0.0
        for mi, ni, pfun, amp in zip(self._ms, self._ns, self._pfuns, self._amps):
            val += (
                amp
                * pfun(eta1)
                * np.sin(
                    mi * 2.0 * np.pi * eta2 + ni * 2.0 * np.pi * eta3,
                )
            )

        return val


class TorusModesCos:
    r"""Cosinusoidal function in the periodic coordinates of a Torus.

    .. math::

        u(\eta_1, \eta_2, \eta_3) = \sum_{s} \chi_s(\eta_1) A_s \cos(m_s\,2\pi \eta_2 + n_s\,2\pi \eta_3) \,,

    where :math:`\chi_s(\eta_1)` is one of

    .. math::

        \chi_s(\eta_1) = \left\{
        \begin{aligned}
        &\sin(\pi\eta_1)\,,
        \\[2mm]
        &\exp \left(- \frac{(\eta_1 - r_0)^2}{2 \sigma^2} \right) \,,
        \\[2mm]
        & - \frac{\eta_1 - r_0}{\sigma} \exp \left(- \frac{(\eta_1 - r_0)^2}{2 \sigma^2} \right) \,.
        \end{aligned}
        \right.

    Can only be defined in logical coordinates.

    Parameters
    ----------
    ms : tuple | list[int]
        Poloidal mode numbers.

    ns : tuple | list[int]
        Toroidal mode numbers.

    pfuns : tuple | list[str]
        "sin" or "cos" or "exp" to define the profile functions.

    amps : tuple | list[float]
        Amplitudes of each mode (m_i, n_i).

    pfun_params : tuple | list
        Provides :math:`[r_0, \sigma]` parameters for each "exp" profile fucntion, and l_s for "sin" and "cos".

    Note
    ----
    In the parameter .yml, use the following template in the section ``fluid/<species>``::

        perturbations :
            type : TorusModesCos
            TorusModesCos :
                comps :
                    n3 : null                     # choices: null, 'physical', '0', '3'
                    u2 : ['physical', 'v', '2']   # choices: null, 'physical', '1', '2', 'v', 'norm'
                    p3 : H1                       # choices: null, 'physical', '0', '3'
                ms :
                    n3: null            # poloidal mode numbers
                    u2: [[0], [0], [0]] # poloidal mode numbers
                    p3: [0]             # poloidal mode numbers
                ns :
                    n3: null            # toroidal mode numbers
                    u2: [[1], [1], [1]] # toroidal mode numbers
                    p3: [1]             # toroidal mode numbers
                amps :
                    n3: null                        # amplitudes of each mode
                    u2: [[0.001], [0.001], [0.001]] # amplitudes of each mode
                    p3: [0.01]                      # amplitudes of each mode
                pfuns :
                    n3: null                        # profile function in eta1-direction ('sin' or 'cos' or 'exp' or 'd_exp')
                    u2: [['sin'], ['sin'], ['exp']] # profile function in eta1-direction ('sin' or 'cos' or 'exp' or 'd_exp')
                    p3: [0.01]                      # profile function in eta1-direction ('sin' or 'cos' or 'exp' or 'd_exp')
                pfun_params :
                    n3: null                      # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and l_s for "sin" and "cos".
                    u2: [2, null, [[0.5, 1.]]]    # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and l_s for "sin" and "cos".
                    p3: [0.01]                    # Provides [r_0, sigma] parameters for each "exp" and "d_exp" profile fucntion, and l_s for "sin" and "cos".
    """

    def __init__(self, ms=None, ns=None, amps=(1e-4,), pfuns=("sin",), pfun_params=None):
        if ms is not None:
            n_modes = len(ms)
        elif ns is not None:
            n_modes = len(ns)
            ms = [0] * n_modes
        else:
            n_modes = 1
            ms = [1]
            ns = [0]

        if ns is None:
            ns = [0] * n_modes
        else:
            assert len(ns) == n_modes

        if len(amps) == 1:
            amps = [amps[0]] * n_modes
        else:
            assert len(amps) == n_modes

        if len(pfuns) == 1:
            pfuns = [pfuns[0]] * n_modes
        else:
            assert len(pfuns) == n_modes

        if pfun_params is None:
            pfun_params = [None] * n_modes

        self._ms = ms
        self._ns = ns
        self._amps = amps

        self._pfuns = []
        for pfun, params in zip(pfuns, pfun_params):
            if pfun == "sin":
                if params is None:
                    ls = 1
                else:
                    ls = params
                self._pfuns += [lambda eta1: np.sin(ls * np.pi * eta1)]
            elif pfun == "cos":
                self._pfuns += [lambda eta1: np.cos(np.pi * eta1)]
            elif pfun == "exp":
                self._pfuns += [
                    lambda eta1: np.exp(-((eta1 - params[0]) ** 2) / (2 * params[1] ** 2))
                    / np.sqrt(2 * np.pi * params[1] ** 2),
                ]
            elif pfun == "d_exp":
                self._pfuns += [
                    lambda eta1: -(eta1 - params[0])
                    / params[1] ** 2
                    * np.exp(-((eta1 - params[0]) ** 2) / (2 * params[1] ** 2))
                    / np.sqrt(2 * np.pi * params[1] ** 2),
                ]
            else:
                raise ValueError(
                    'Profile function must be "sin" or "cos" or "exp".',
                )

    def __call__(self, eta1, eta2, eta3):
        val = 0.0
        for mi, ni, pfun, amp in zip(self._ms, self._ns, self._pfuns, self._amps):
            val += (
                amp
                * pfun(eta1)
                * np.cos(
                    mi * 2.0 * np.pi * eta2 + ni * 2.0 * np.pi * eta3,
                )
            )

        return val


class Shear_x:
    r"""Double shear layer in eta1 (-1 in outer regions, 1 in inner regions).

    .. math::

        u(\eta_1, \eta_2, \eta_3) = A(-\tanh((\eta_1 - 0.25)/\delta)+\tanh((\eta_1 - 0.75)/\delta) - 1) \,.

    Can only be used in logical space.

    Parameters
    ----------
    amps : float
        Amplitude of the velocity on each side.

    delta : float
        Characteristic size of the shear layer

    Note
    ----
    In the parameter .yml, use the following in the section ``fluid/<species>``::

        perturbations :
            type : Shear_x
            Shear_x :
                comps :
                    rho3 : null                   # choices: null, 'physical', '0', '3'
                    uv : ['physical', 'v', '2']   # choices: null, 'physical', '1', '2', 'v', 'norm'
                    s3 : H1                       # choices: null, 'physical', '0', '3'
                amp : 0.001 # amplitudes of each mode
                delta : 0.03333 # characteristic size of the shear layer
    """

    def __init__(self, amp=1e-4, delta=1 / 15):
        self._amp = amp
        self._delta = delta

    def __call__(self, e1, e2, e3):
        val = self._amp * (-np.tanh((e1 - 0.75) / self._delta) + np.tanh((e1 - 0.25) / self._delta) - 1)

        return val


class Shear_y:
    r"""Double shear layer in eta2 (-1 in outer regions, 1 in inner regions).

    .. math::

        u(\eta_1, \eta_2, \eta_3) = A(-\tanh((\eta_2 - 0.25)/\delta) + \tanh((\eta_2 - 0.75)/\delta) - 1) \,.

    Can only be used in logical space.

    Parameters
    ----------
    amps : float
        Amplitude of the velocity on each side.

    delta : float
        Characteristic size of the shear layer

    Note
    ----
    In the parameter .yml, use the following in the section ``fluid/<species>``::

        perturbations :
            type : Shear_y
            Shear_y :
                comps :
                    rho3 : null                   # choices: null, 'physical', '0', '3'
                    uv : ['physical', 'v', '2']   # choices: null, 'physical', '1', '2', 'v', 'norm'
                    s3 : H1                       # choices: null, 'physical', '0', '3'
                amp : 0.001 # amplitudes of each mode
                delta : 0.03333 # characteristic size of the shear layer
    """

    def __init__(self, amp=1e-4, delta=1 / 15):
        self._amp = amp
        self._delta = delta

    def __call__(self, e1, e2, e3):
        val = self._amp * (-np.tanh((e2 - 0.75) / self._delta) + np.tanh((e2 - 0.25) / self._delta) - 1)

        return val


class Shear_z:
    r"""Double shear layer in eta3 (-1 in outer regions, 1 in inner regions).

    .. math::

        u(\eta_1, \eta_2, \eta_3) = A(-\tanh((\eta_3 - 0.25)/\delta) + \tanh((\eta_3 - 0.75)/\delta) - 1) \,.

    Can only be used in logical space.

    Parameters
    ----------
    amps : float
        Amplitude of the velocity on each side.

    delta : float
        Characteristic size of the shear layer

    Note
    ----
    In the parameter .yml, use the following in the section ``fluid/<species>``::

        perturbations :
            type : Shear_y
            Shear_y :
                comps :
                    rho3 : null                   # choices: null, 'physical', '0', '3'
                    uv : ['physical', 'v', '2']   # choices: null, 'physical', '1', '2', 'v', 'norm'
                    s3 : H1                       # choices: null, 'physical', '0', '3'
                amp : 0.001 # amplitudes of each mode
                delta : 0.03333 # characteristic size of the shear layer
    """

    def __init__(self, amp=1e-4, delta=1 / 15):
        self._amp = amp
        self._delta = delta

    def __call__(self, e1, e2, e3):
        val = self._amp * (-np.tanh((e3 - 0.75) / self._delta) + np.tanh((e3 - 0.25) / self._delta) - 1)

        return val


class Erf_z:
    r"""Shear layer in eta3 (-1 in lower regions, 1 in upper regions).

    .. math::

        u(\eta_1, \eta_2, \eta_3) = A \, erf((\eta_3 - 0.5)/\delta) \,.

    Can only be used in logical space.

    Parameters
    ----------
    amp : float
        Amplitude of the velocity on each side.

    delta : float
        Characteristic size of the shear layer

    Note
    ----
    In the parameter .yml, use the following in the section ``fluid/<species>``::

        perturbations :
            type : Erf_z
        Erf_z :
            comps :
                b2 : ['2', null, null] # choices: null, 'physical', '0', '3'
            amp :
                b2 : [0.001] # amplitudes of each mode
            delta :
                b2 : [0.02] # characteristic size of the shear layer
    """

    def __init__(self, amp=1e-4, delta=1 / 15):
        self._amp = amp
        self._delta = delta

    def __call__(self, e1, e2, e3):
        from scipy.special import erf

        val = self._amp * erf((e3 - 0.5) / self._delta)

        return val


class RestelliAnalyticSolutionVelocity:
    r"""Analytic solution :math:`u=u_e` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    where :math:`f` is defined as follows: 

    .. math::

        f = \nu \omega \,, 
        \\[2mm]
        \omega = \left[0, \alpha \frac{R_0 - 4R}{a R_0 R} - \beta \frac{B_p}{B_0}\frac{R_0^2}{a R^3}, 0 \right] \,, 
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.

    Can only be defined in Cartesian coordinates. 
    The solution is given by:

    .. math::
        \alpha \frac{R}{a R_0} \left[\begin{array}{c} -z \\ R-R_0 \\ 0 \end{array} \right] + \beta \frac{B_p}{B_0} \frac{R_0}{aR} \left[\begin{array}{c} z \\ -(R-R_0) \\ \frac{B_0}{B_p} a \end{array} \right] \,,
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.

    Parameters
    ----------
    comp : string
        Which component of the solution ('0', '1' or '2').
    a : float
        Minor radius of torus (default: 1.).
    R0 : float
        Major radius of torus (default: 2.).
    B0 : float
        On-axis (r=0) toroidal magnetic field (default: 10.).
    Bp : float
        Poloidal magnetic field (default: 12.5).
    alpha : float
        (default: 0.1)
    beta : float
        (default: 1.0)

    References
    ----------
    [1] Juan Vicente Gutiérrez-Santacreu, Omar Maj, Marco Restelli: Finite element discretization of a Stokes-like model arising
    in plasma physics, Journal of Computational Physics 2018.
    """

    def __init__(self, comp="0", a=1.0, R0=2.0, B0=10.0, Bp=12.5, alpha=0.1, beta=1.0):
        self._comp = comp
        self._a = a
        self._R0 = R0
        self._B0 = B0
        self._Bp = Bp
        self._alpha = alpha
        self._beta = beta

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        R = np.sqrt(x**2 + y**2)
        R = np.where(R == 0.0, 1e-9, R)
        phi = np.arctan2(-y, x)
        ustarR = (
            self._alpha * R / (self._a * self._R0) * (-z)
            + self._beta * self._Bp * self._R0 / (self._B0 * self._a * R) * z
        )
        ustarZ = self._alpha * R / (self._a * self._R0) * (R - self._R0) + self._beta * self._Bp * self._R0 / (
            self._B0 * self._a * R
        ) * (-(R - self._R0))
        ustarphi = self._beta * self._Bp * self._R0 / (self._B0 * self._a * R) * self._B0 * self._a / self._Bp

        # form normalized to cylindrical coordinates:
        uR = ustarR
        uphi = ustarphi / R
        uZ = ustarZ

        # from cylindrical to cartesian:

        if self._comp == "0":
            ux = np.cos(phi) * uR - R * np.sin(phi) * uphi
            return ux
        elif self._comp == "1":
            uy = -np.sin(phi) * uR - R * np.cos(phi) * uphi
            return uy
        elif self._comp == "2":
            uz = uZ
            return uz
        else:
            raise ValueError(f"Invalid component '{self._comp}'. Must be '0', '1', or '2'.")


class RestelliAnalyticSolutionVelocity_2:
    r"""Analytic solution :math:`u=u_e` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    where :math:`f` is defined as follows: 

    .. math::

        f = \nu \omega \,, 
        \\[2mm]
        \omega = \left[0, \alpha \frac{R_0 - 4R}{a R_0 R} - \beta \frac{B_p}{B_0}\frac{R_0^2}{a R^3}, 0 \right] \,, 
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.

    Can only be defined in Cartesian coordinates. 
    The solution is given by:

    .. math::
        \alpha \frac{R}{a R_0} \left[\begin{array}{c} -z \\ R-R_0 \\ 0 \end{array} \right] + \beta \frac{B_p}{B_0} \frac{R_0}{aR} \left[\begin{array}{c} z \\ -(R-R_0) \\ \frac{B_0}{B_p} a \end{array} \right] \,,
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.

    Parameters
    ----------
    comp : string
        Which component of the solution ('0', '1' or '2').
    a : float
        Minor radius of torus (default: 1.).
    R0 : float
        Major radius of torus (default: 2.).
    B0 : float
        On-axis (r=0) toroidal magnetic field (default: 10.).
    Bp : float
        Poloidal magnetic field (default: 12.5).
    alpha : float
        (default: 0.1)
    beta : float
        (default: 1.0)

    References
    ----------
    [1] Juan Vicente Gutiérrez-Santacreu, Omar Maj, Marco Restelli: Finite element discretization of a Stokes-like model arising
    in plasma physics, Journal of Computational Physics 2018.
    """

    def __init__(self, comp="0", a=1.0, R0=2.0, B0=10.0, Bp=12.5, alpha=0.1, beta=1.0):
        self._comp = comp
        self._a = a
        self._R0 = R0
        self._B0 = B0
        self._Bp = Bp
        self._alpha = alpha
        self._beta = beta

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        R = np.sqrt(x**2 + y**2)
        R = np.where(R == 0.0, 1e-9, R)
        phi = np.arctan2(-y, x)
        ustarR = (
            self._alpha * R / (self._a * self._R0) * (-z)
            + self._beta * self._Bp * self._R0 / (self._B0 * self._a * R) * z
        )
        ustarZ = self._alpha * R / (self._a * self._R0) * (R - self._R0) + self._beta * self._Bp * self._R0 / (
            self._B0 * self._a * R
        ) * (-(R - self._R0))
        ustarphi = self._beta * self._Bp * self._R0 / (self._B0 * self._a * R) * self._B0 * self._a / self._Bp

        # form normalized to cylindrical coordinates:
        uR = ustarR
        uphi = ustarphi / R
        uZ = ustarZ

        # from cylindrical to cartesian:

        if self._comp == "0":
            ux = np.cos(phi) * uR - R * np.sin(phi) * uphi
            return ux
        elif self._comp == "1":
            uy = -np.sin(phi) * uR - R * np.cos(phi) * uphi
            return uy
        elif self._comp == "2":
            uz = uZ
            return uz
        else:
            raise ValueError(f"Invalid component '{self._comp}'. Must be '0', '1', or '2'.")


class RestelliAnalyticSolutionVelocity_3:
    r"""Analytic solution :math:`u=u_e` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    where :math:`f` is defined as follows: 

    .. math::

        f = \nu \omega \,, 
        \\[2mm]
        \omega = \left[0, \alpha \frac{R_0 - 4R}{a R_0 R} - \beta \frac{B_p}{B_0}\frac{R_0^2}{a R^3}, 0 \right] \,, 
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.

    Can only be defined in Cartesian coordinates. 
    The solution is given by:

    .. math::
        \alpha \frac{R}{a R_0} \left[\begin{array}{c} -z \\ R-R_0 \\ 0 \end{array} \right] + \beta \frac{B_p}{B_0} \frac{R_0}{aR} \left[\begin{array}{c} z \\ -(R-R_0) \\ \frac{B_0}{B_p} a \end{array} \right] \,,
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.

    Parameters
    ----------
    comp : string
        Which component of the solution ('0', '1' or '2').
    a : float
        Minor radius of torus (default: 1.).
    R0 : float
        Major radius of torus (default: 2.).
    B0 : float
        On-axis (r=0) toroidal magnetic field (default: 10.).
    Bp : float
        Poloidal magnetic field (default: 12.5).
    alpha : float
        (default: 0.1)
    beta : float
        (default: 1.0)

    References
    ----------
    [1] Juan Vicente Gutiérrez-Santacreu, Omar Maj, Marco Restelli: Finite element discretization of a Stokes-like model arising
    in plasma physics, Journal of Computational Physics 2018.
    """

    def __init__(self, comp="0", a=1.0, R0=2.0, B0=10.0, Bp=12.5, alpha=0.1, beta=1.0):
        self._comp = comp
        self._a = a
        self._R0 = R0
        self._B0 = B0
        self._Bp = Bp
        self._alpha = alpha
        self._beta = beta

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        R = np.sqrt(x**2 + y**2)
        R = np.where(R == 0.0, 1e-9, R)
        phi = np.arctan2(-y, x)
        ustarR = (
            self._alpha * R / (self._a * self._R0) * (-z)
            + self._beta * self._Bp * self._R0 / (self._B0 * self._a * R) * z
        )
        ustarZ = self._alpha * R / (self._a * self._R0) * (R - self._R0) + self._beta * self._Bp * self._R0 / (
            self._B0 * self._a * R
        ) * (-(R - self._R0))
        ustarphi = self._beta * self._Bp * self._R0 / (self._B0 * self._a * R) * self._B0 * self._a / self._Bp

        # form normalized to cylindrical coordinates:
        uR = ustarR
        uphi = ustarphi / R
        uZ = ustarZ

        # from cylindrical to cartesian:

        if self._comp == "0":
            ux = np.cos(phi) * uR - R * np.sin(phi) * uphi
            return ux
        elif self._comp == "1":
            uy = -np.sin(phi) * uR - R * np.cos(phi) * uphi
            return uy
        elif self._comp == "2":
            uz = uZ
            return uz
        else:
            raise ValueError(f"Invalid component '{self._comp}'. Must be '0', '1', or '2'.")


class RestelliAnalyticSolutionPotential:
    r"""Analytic solution :math:`\phi` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    where :math:`f` is defined as follows: 

    .. math::

        f = \nu \omega \,, 
        \\[2mm]
        \omega = \left[0, \alpha \frac{R_0 - 4R}{a R_0 R} - \beta \frac{B_p}{B_0}\frac{R_0^2}{a R^3}, 0 \right] \,, 
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.

    Can only be defined in Cartesian coordinates. 
    The solution is given by:

    .. math::
        \phi = \frac{1}{2} a B_0 \alpha \left( \frac{(R-R_0)^2+z^2}{a^2} - \frac{2}{3} \right)
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.

    Parameters
    ----------
    a : float
        Minor radius of torus (default: 1.).
    R0 : float
        Major radius of torus (default: 2.).
    B0 : float
        On-axis (r=0) toroidal magnetic field (default: 10.).
    Bp : float
        Poloidal magnetic field (default: 12.5).
    alpha : float
        (default: 0.1)
    beta : float
        (default: 1.0)

    References
    ----------
    [1] Juan Vicente Gutiérrez-Santacreu, Omar Maj, Marco Restelli: Finite element discretization of a Stokes-like model arising
    in plasma physics, Journal of Computational Physics 2018.
    """

    def __init__(self, a=1.0, R0=2.0, B0=10.0, Bp=12.5, alpha=0.1, beta=1.0):
        self._a = a
        self._R0 = R0
        self._B0 = B0
        self._Bp = Bp
        self._alpha = alpha
        self._beta = beta

    # equilibrium potential
    def __call__(self, x, y, z):
        """Equilibrium potential."""
        R = np.sqrt(x**2 + y**2)
        pp = 0.5 * self._a * self._B0 * self._alpha * (((R - self._R0) ** 2 + z**2) / self._a**2 - 2.0 / 3.0)

        return pp


class ManufacturedSolutionVelocity:
    r"""Analytic solutions :math:`u` and :math:`u_e` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    Can only be defined in Cartesian coordinates. 
    The solution in 1D is given by:

    .. math::
        u =  \left[\begin{array}{c} sin(2 \pi x) + 1.0 \\ 0 \\ 0 \end{array} \right] \,,
        u_e =  \left[\begin{array}{c} sin(2 \pi x) \\ 0 \\ 0 \end{array} \right] \,.
    
    The solution in 2D is given by:

    .. math::
        u =  \left[\begin{array}{c} -sin(2 \pi x) sin(2 \pi y) \\ -cos(2 \pi y) cos(2 \pi y) \\ 0 \end{array} \right] \,,
        u_e =  \left[\begin{array}{c} -sin(4 \pi x) sin(4 \pi y) \\ -cos(4 \pi y) cos(4 \pi y) \\ 0 \end{array} \right] \,.
        
    Parameters
    ----------
    species : string
        'Ions' or 'Electrons'.
    comp : string
        Which component of the solution ('0', '1' or '2').
    dimension: string
        Defines the manufactured solution to be selected ('1D' or '2D').
    b0 : float
        Magnetic field (default: 1.0).
    """

    def __init__(self, species="Ions", comp="0", dimension="1D", b0=1.0):
        self._b = b0
        self._species = species
        self._comp = comp
        self._dimension = dimension

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        if self._species == "Ions":
            """Velocity of ions."""
            """x component"""
            if self._dimension == "2D":
                ux = -np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
            elif self._dimension == "1D":
                ux = np.sin(2 * np.pi * x) + 1.0

            """y component"""
            if self._dimension == "2D":
                uy = -np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
            elif self._dimension == "1D":
                uy = np.cos(2 * np.pi * x)

            """z component"""
            uz = 0.0 * x

            if self._comp == "0":
                return ux
            elif self._comp == "1":
                return uy
            elif self._comp == "2":
                return uz
            else:
                raise ValueError(f"Invalid component '{self._comp}'. Must be '0', '1', or '2'.")

        elif self._species == "Electrons":
            """Velocity of electrons."""
            """x component"""
            if self._dimension == "2D":
                ux = -np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y)
            elif self._dimension == "1D":
                ux = np.sin(2.0 * np.pi * x)

            """y component"""
            if self._dimension == "2D":
                uy = -np.cos(4 * np.pi * x) * np.cos(4 * np.pi * y)
            elif self._dimension == "1D":
                uy = np.cos(2 * np.pi * x)

            """z component"""
            uz = 0.0 * x

            if self._comp == "0":
                return ux
            if self._comp == "1":
                return uy
            if self._comp == "2":
                return uz
            else:
                raise ValueError(f"Invalid component '{self._comp}'. Must be '0', '1', or '2'.")

        else:
            raise ValueError(f"Invalid species '{self._species}'. Must be 'Ions' or 'Electrons'.")


class ManufacturedSolutionPotential:
    r"""Analytic solution :math:`\phi` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    where :math:`f` is defined as follows: 

    .. math::

        f = \left[1 - b_0 cos(x) - \nu sin(y), 1 - b_0 sin(y) + \nu cos(x) , 0 \right] \,, 
        \\[2mm]
        f_e = \left[-1 + 0.5 b_0 cos(x) - \nu_e 0.5 sin(y), -1 + 0.5 b_0 sin(y) + \nu_e cos(x) , 0 \right] \,.

    Can only be defined in Cartesian coordinates. 
    The solution in 1D is given by:

    .. math::
        \phi =  sin(2\pi x) \,.
        
    The solution in 2D is given by:

    .. math::
        \phi =  cos(2\pi x) + sin(2\pi y) \,.
        
    Parameters
    ----------
    dimension: string
        Defines the manufactured solution to be selected ('1D' or '2D').
    b0 : float
        Magnetic field (default: 1.0).
    """

    def __init__(self, dimension="1D", b0=1.0):
        self._ab = b0
        self._dimension = dimension

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Potential."""
        if self._dimension == "2D":
            phi = np.cos(2 * np.pi * x) + np.sin(2 * np.pi * y)
        elif self._dimension == "1D":
            phi = np.sin(2.0 * np.pi * x)

        return phi


class ManufacturedSolutionVelocity_2:
    r"""Analytic solutions :math:`u` and :math:`u_e` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    Can only be defined in Cartesian coordinates. 
    The solution in 1D is given by:

    .. math::
        u =  \left[\begin{array}{c} sin(2 \pi x) + 1.0 \\ 0 \\ 0 \end{array} \right] \,,
        u_e =  \left[\begin{array}{c} sin(2 \pi x) \\ 0 \\ 0 \end{array} \right] \,.
    
    The solution in 2D is given by:

    .. math::
        u =  \left[\begin{array}{c} -sin(2 \pi x) sin(2 \pi y) \\ -cos(2 \pi y) cos(2 \pi y) \\ 0 \end{array} \right] \,,
        u_e =  \left[\begin{array}{c} -sin(4 \pi x) sin(4 \pi y) \\ -cos(4 \pi y) cos(4 \pi y) \\ 0 \end{array} \right] \,.
        
    Parameters
    ----------
    species : string
        'Ions' or 'Electrons'.
    comp : string
        Which component of the solution ('0', '1' or '2').
    dimension: string
        Defines the manufactured solution to be selected ('1D' or '2D').
    b0 : float
        Magnetic field (default: 1.0).
    """

    def __init__(self, species="Ions", comp="0", dimension="1D", b0=1.0):
        self._b = b0
        self._species = species
        self._comp = comp
        self._dimension = dimension

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        if self._species == "Ions":
            """Velocity of ions."""
            """x component"""
            if self._dimension == "2D":
                ux = -np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
            elif self._dimension == "1D":
                ux = np.sin(2 * np.pi * x) + 1.0

            """y component"""
            if self._dimension == "2D":
                uy = -np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
            elif self._dimension == "1D":
                uy = np.cos(2 * np.pi * x)

            """z component"""
            uz = 0.0 * x

            if self._comp == "0":
                return ux
            elif self._comp == "1":
                return uy
            elif self._comp == "2":
                return uz
            else:
                raise ValueError(f"Invalid component '{self._comp}'. Must be '0', '1', or '2'.")

        elif self._species == "Electrons":
            """Velocity of electrons."""
            """x component"""
            if self._dimension == "2D":
                ux = -np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y)
            elif self._dimension == "1D":
                ux = np.sin(2.0 * np.pi * x)

            """y component"""
            if self._dimension == "2D":
                uy = -np.cos(4 * np.pi * x) * np.cos(4 * np.pi * y)
            elif self._dimension == "1D":
                uy = np.cos(2 * np.pi * x)

            """z component"""
            uz = 0.0 * x

            if self._comp == "0":
                return ux
            if self._comp == "1":
                return uy
            if self._comp == "2":
                return uz
            else:
                raise ValueError(f"Invalid component '{self._comp}'. Must be '0', '1', or '2'.")

        else:
            raise ValueError(f"Invalid species '{self._species}'. Must be 'Ions' or 'Electrons'.")


class TokamakManufacturedSolutionVelocity:
    r"""Analytic solution :math:`u=u_e` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    where :math:`f` is defined as follows: 

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
    The solution is given by:

    .. math::
        \mathbf{u} = \alpha \frac{1}{a R_0} \left[\begin{array}{c} R-R_0 \\ z \\ 0 \end{array} \right]  \,,
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.

    Parameters
    ----------
    comp : string
        Which component of the solution ('0', '1' or '2').
    a : float
        Minor radius of torus (default: 1.).
    R0 : float
        Major radius of torus (default: 2.).
    B0 : float
        On-axis (r=0) toroidal magnetic field (default: 10.).
    Bp : float
        Poloidal magnetic field (default: 12.5).
    alpha : float
        (default: 0.1)
    beta : float
        (default: 1.0)

    References
    ----------
    [1] Juan Vicente Gutiérrez-Santacreu, Omar Maj, Marco Restelli: Finite element discretization of a Stokes-like model arising
    in plasma physics, Journal of Computational Physics 2018.
    """

    def __init__(self, comp="0", a=1.0, R0=2.0, B0=10.0, Bp=12.5, alpha=0.1, beta=1.0):
        self._comp = comp
        self._a = a
        self._R0 = R0
        self._B0 = B0
        self._Bp = Bp
        self._alpha = alpha
        self._beta = beta

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        R = np.sqrt(x**2 + y**2)
        R = np.where(R == 0.0, 1e-9, R)
        phi = np.arctan2(-y, x)
        A = self._alpha / (self._a * self._R0)
        C = self._beta * self._Bp * self._R0 / (self._B0 * self._a)

        uR = A * (R - self._R0)
        uZ = A * z
        uphi = 0

        # from cylindrical to cartesian:

        if self._comp == "0":
            ux = np.cos(phi) * uR - R * np.sin(phi) * uphi
            return ux
        elif self._comp == "1":
            uy = -np.sin(phi) * uR - R * np.cos(phi) * uphi
            return uy
        elif self._comp == "2":
            uz = uZ
            return uz
        else:
            raise ValueError(f"Invalid component '{self._comp}'. Must be '0', '1', or '2'.")


class TokamakManufacturedSolutionVelocity_1:
    r"""Analytic solution :math:`u=u_e` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    where :math:`f` is defined as follows: 

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
    The solution is given by:

    .. math::
        \mathbf{u} = \alpha \frac{1}{a R_0} \left[\begin{array}{c} R-R_0 \\ z \\ 0 \end{array} \right]  \,,
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.

    Parameters
    ----------
    comp : string
        Which component of the solution ('0', '1' or '2').
    a : float
        Minor radius of torus (default: 1.).
    R0 : float
        Major radius of torus (default: 2.).
    B0 : float
        On-axis (r=0) toroidal magnetic field (default: 10.).
    Bp : float
        Poloidal magnetic field (default: 12.5).
    alpha : float
        (default: 0.1)
    beta : float
        (default: 1.0)

    References
    ----------
    [1] Juan Vicente Gutiérrez-Santacreu, Omar Maj, Marco Restelli: Finite element discretization of a Stokes-like model arising
    in plasma physics, Journal of Computational Physics 2018.
    """

    def __init__(self, comp="0", a=1.0, R0=2.0, B0=10.0, Bp=12.5, alpha=0.1, beta=1.0):
        self._comp = comp
        self._a = a
        self._R0 = R0
        self._B0 = B0
        self._Bp = Bp
        self._alpha = alpha
        self._beta = beta

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        R = np.sqrt(x**2 + y**2)
        R = np.where(R == 0.0, 1e-9, R)
        phi = np.arctan2(-y, x)
        A = self._alpha / (self._a * self._R0)
        C = self._beta * self._Bp * self._R0 / (self._B0 * self._a)

        uR = A * (R - self._R0)
        uZ = A * z
        uphi = 0

        # from cylindrical to cartesian:

        if self._comp == "0":
            ux = np.cos(phi) * uR - R * np.sin(phi) * uphi
            return ux
        elif self._comp == "1":
            uy = -np.sin(phi) * uR - R * np.cos(phi) * uphi
            return uy
        elif self._comp == "2":
            uz = uZ
            return uz
        else:
            raise ValueError(f"Invalid component '{self._comp}'. Must be '0', '1', or '2'.")


class TokamakManufacturedSolutionVelocity_2:
    r"""Analytic solution :math:`u=u_e` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    where :math:`f` is defined as follows: 

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
    The solution is given by:

    .. math::
        \mathbf{u} = \alpha \frac{1}{a R_0} \left[\begin{array}{c} R-R_0 \\ z \\ 0 \end{array} \right]  \,,
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.

    Parameters
    ----------
    comp : string
        Which component of the solution ('0', '1' or '2').
    a : float
        Minor radius of torus (default: 1.).
    R0 : float
        Major radius of torus (default: 2.).
    B0 : float
        On-axis (r=0) toroidal magnetic field (default: 10.).
    Bp : float
        Poloidal magnetic field (default: 12.5).
    alpha : float
        (default: 0.1)
    beta : float
        (default: 1.0)

    References
    ----------
    [1] Juan Vicente Gutiérrez-Santacreu, Omar Maj, Marco Restelli: Finite element discretization of a Stokes-like model arising
    in plasma physics, Journal of Computational Physics 2018.
    """

    def __init__(self, comp="0", a=1.0, R0=2.0, B0=10.0, Bp=12.5, alpha=0.1, beta=1.0):
        self._comp = comp
        self._a = a
        self._R0 = R0
        self._B0 = B0
        self._Bp = Bp
        self._alpha = alpha
        self._beta = beta

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        R = np.sqrt(x**2 + y**2)
        R = np.where(R == 0.0, 1e-9, R)
        phi = np.arctan2(-y, x)
        A = self._alpha / (self._a * self._R0)
        C = self._beta * self._Bp * self._R0 / (self._B0 * self._a)

        uR = A * (R - self._R0)
        uZ = A * z
        uphi = 0

        # from cylindrical to cartesian:

        if self._comp == "0":
            ux = np.cos(phi) * uR - R * np.sin(phi) * uphi
            return ux
        elif self._comp == "1":
            uy = -np.sin(phi) * uR - R * np.cos(phi) * uphi
            return uy
        elif self._comp == "2":
            uz = uZ
            return uz
        else:
            raise ValueError(f"Invalid component '{self._comp}'. Must be '0', '1', or '2'.")
