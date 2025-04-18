#!/usr/bin/env python3
"Analytical perturbations (modes)."

import numpy as np


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
        r"""
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
        """

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
        """
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
        """

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
        r"""
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
        """

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
        r"""
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
        """

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
        """
        Parameters
        ----------
        amps : float
            Amplitude of the velocity on each side.

        delta : float
            Characteristic size of the shear layer
        """

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
        """
        Parameters
        ----------
        amps : float
            Amplitude of the velocity on each side.

        delta : float
            Characteristic size of the shear layer
        """

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
        """
        Parameters
        ----------
        amps : float
            Amplitude of the velocity on each side.

        delta : float
            Characteristic size of the shear layer
        """

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
        """
        Parameters
        ----------
        amp : float
            Amplitude of the velocity on each side.

        delta : float
            Characteristic size of the shear layer
        """

        self._amp = amp
        self._delta = delta

    def __call__(self, e1, e2, e3):
        from scipy.special import erf

        val = self._amp * erf((e3 - 0.5) / self._delta)

        return val


class forcingterm:
    r"""Force term :math:`\chi_s(\eta_1)` on the right-hand-side of:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,

    where :math:`f` is defined as follows:

    .. math::

        f = \nu \omega \,,
        \\[2mm]
        \omega = \left[0, \alpha \frac{R_0 - 4R}{a R_0 R} - \beta \frac{B_p}{B_0}\frac{R_0^2}{a R^3}, 0 \right] \,,
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.

    Can only be defined in carthesian coordinates.

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
        R = np.sqrt(x**2+y**2)
        R = np.where(R == 0.0, 1e-9, R)
        phi = np.arctan2(-y, x)
        force_Z = self._nu * (self._alpha * (self._R0 - 4 * R) / (
            self._a * self._R0 * R
        ) - self._beta * self._Bp * self._R0**2 / (self._B0 * self._a * R**3))

        return force_Z


class AnalyticSolutionRestelliVelocity_x:
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

    Can only be defined in carthesian coordinates. 
    The solution is given by:

    .. math::
        \alpha \frac{R}{a R_0} \left[\begin{array}{c} -z \\ R-R_0 \\ 0 \end{array} \right] + \beta \frac{B_p}{B_0} \frac{R_0}{aR} \left[\begin{array}{c} z \\ -(R-R_0) \\ \frac{B_0}{B_p} a \end{array} \right] \,,
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.
    """

    def __init__(self, a=1.0, R0=2.0, B0=10.0, Bp=12.5, alpha=0.1, beta=1.0):
        """
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
        """

        self._a = a
        self._R0 = R0
        self._B0 = B0
        self._Bp = Bp
        self._alpha = alpha
        self._beta = beta

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        R = np.sqrt(x**2+y**2)
        R = np.where(R == 0.0, 1e-9, R)
        phi = np.arctan2(-y, x)
        uR = self._alpha*R/(self._a*self._R0)*(-z) + self._beta*self._Bp*self._R0/(self._B0*self._a*R)*z
        uZ = self._alpha*R/(self._a*self._R0)*(R-self._R0) + self._beta * \
            self._Bp*self._R0/(self._B0*self._a*R)*(-(R-self._R0))
        uphi = self._beta*self._Bp*self._R0/(self._B0*self._a*R)*self._B0*self._a/self._Bp

        ux = np.cos(phi)*uR - R*np.sin(phi)*uphi  # signs changed??

        return ux


class AnalyticSolutionRestelliVelocity_y:
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

    Can only be defined in carthesian coordinates. 
    The solution is given by:

    .. math::
        \alpha \frac{R}{a R_0} \left[\begin{array}{c} -z \\ R-R_0 \\ 0 \end{array} \right] + \beta \frac{B_p}{B_0} \frac{R_0}{aR} \left[\begin{array}{c} z \\ -(R-R_0) \\ \frac{B_0}{B_p} a \end{array} \right] \,,
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.
    """

    def __init__(self, a=1.0, R0=2.0, B0=10.0, Bp=12.5, alpha=0.1, beta=1.0):
        """
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
        """

        self._a = a
        self._R0 = R0
        self._B0 = B0
        self._Bp = Bp
        self._alpha = alpha
        self._beta = beta

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        R = np.sqrt(x**2+y**2)
        R = np.where(R == 0.0, 1e-9, R)
        phi = np.arctan2(-y, x)
        uR = self._alpha*R/(self._a*self._R0)*(-z) + self._beta*self._Bp*self._R0/(self._B0*self._a*R)*z
        uZ = self._alpha*R/(self._a*self._R0)*(R-self._R0) + self._beta * \
            self._Bp*self._R0/(self._B0*self._a*R)*(-(R-self._R0))
        uphi = self._beta*self._Bp*self._R0/(self._B0*self._a*R)*self._B0*self._a/self._Bp

        uy = -np.sin(phi)*uR - R*np.cos(phi)*uphi

        return uy


class AnalyticSolutionRestelliVelocity_z:
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

    Can only be defined in carthesian coordinates. 
    The solution is given by:

    .. math::
        \alpha \frac{R}{a R_0} \left[\begin{array}{c} -z \\ R-R_0 \\ 0 \end{array} \right] + \beta \frac{B_p}{B_0} \frac{R_0}{aR} \left[\begin{array}{c} z \\ -(R-R_0) \\ \frac{B_0}{B_p} a \end{array} \right] \,,
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.
    """

    def __init__(self, a=1.0, R0=2.0, B0=10.0, Bp=12.5, alpha=0.1, beta=1.0):
        """
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
        """

        self._a = a
        self._R0 = R0
        self._B0 = B0
        self._Bp = Bp
        self._alpha = alpha
        self._beta = beta

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        R = np.sqrt(x**2+y**2)
        R = np.where(R == 0.0, 1e-9, R)
        phi = np.arctan2(-y, x)
        uR = self._alpha*R/(self._a*self._R0)*(-z) + self._beta*self._Bp*self._R0/(self._B0*self._a*R)*z
        uZ = self._alpha*R/(self._a*self._R0)*(R-self._R0) + self._beta * \
            self._Bp*self._R0/(self._B0*self._a*R)*(-(R-self._R0))
        uphi = self._beta*self._Bp*self._R0/(self._B0*self._a*R)*self._B0*self._a/self._Bp

        uz = uZ

        return uz


class AnalyticSolutionRestelliPotential:
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

    Can only be defined in carthesian coordinates. 
    The solution is given by:

    .. math::
        \phi = \frac{1}{2} a B_0 \alpha \left( \frac{(R-R_0)^2+z^2}{a^2} - \frac{2}{3} \right)
        \\[2mm]
        R = \sqrt{x^2 + y^2} \,.
    """

    def __init__(self, a=1.0, R0=2.0, B0=10.0, Bp=12.5, alpha=0.1, beta=1.0):
        """
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
        """

        self._a = a
        self._R0 = R0
        self._B0 = B0
        self._Bp = Bp
        self._alpha = alpha
        self._beta = beta

    # equilibrium potential
    def __call__(self, x, y, z):
        """Equilibrium potential."""
        R = np.sqrt(x**2+y**2)
        pp = 0.5*self._a*self._B0*self._alpha*(((R-self._R0)**2 + z**2)/self._a**2-2.0/3.0)

        return pp


class ManufacturedSolutionVelocity_x:
    r"""Analytic solution :math:`u` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    where :math:`f` is defined as follows: 

    .. math::

        f = \left[1 - b_0 cos(x) - \nu sin(y), 1 - b_0 sin(y) + \nu cos(x) , 0 \right] \,, 
        \\[2mm]
        f_e = \left[-1 + 0.5 b_0 cos(x) - \nu_e 0.5 sin(y), -1 + 0.5 b_0 sin(y) + \nu_e cos(x) , 0 \right] \,.

    Can only be defined in carthesian coordinates. 
    The solution is given by:

    .. math::
        u =  \left[\begin{array}{c} -sin(y) \\ cos(x) \\ 0 \end{array} \right] \,.
    """

    def __init__(self, b0=1.0):
        """
            Parameters
        ----------
        b0 : float
            Magnetic field (default: 1.0).
        """

        self._b = b0

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        #ux = -np.sin(y)
        #ux = -y**2
        ux = -np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

        return ux


class ManufacturedSolutionVelocity_y:
    r"""Analytic solution :math:`u` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    where :math:`f` is defined as follows: 

    .. math::

        f = \left[1 - b_0 cos(x) - \nu sin(y), 1 - b_0 sin(y) + \nu cos(x) , 0 \right] \,, 
        \\[2mm]
        f_e = \left[-1 + 0.5 b_0 cos(x) - \nu_e 0.5 sin(y), -1 + 0.5 b_0 sin(y) + \nu_e cos(x) , 0 \right] \,.

    Can only be defined in carthesian coordinates. 
    The solution is given by:

    .. math::
        u =  \left[\begin{array}{c} -sin(y) \\ cos(x) \\ 0 \end{array} \right] \,.
    """

    def __init__(self, b0=1.0):
        """
            Parameters
        ----------
        b0 : float
            Magnetic field (default: 1.0).
        """

        self._b = b0

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        #uy = np.cos(x)
        #uy = x**2
        uy = -np.cos(2*np.pi*x)*np.cos(2*np.pi*y)

        return uy


class ManufacturedSolutionVelocityElectrons_x:
    r"""Analytic solution :math:`u` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    where :math:`f` is defined as follows: 

    .. math::

        f = \left[1 - b_0 cos(x) - \nu sin(y), 1 - b_0 sin(y) + \nu cos(x) , 0 \right] \,, 
        \\[2mm]
        f_e = \left[-1 + 0.5 b_0 cos(x) - \nu_e 0.5 sin(y), -1 + 0.5 b_0 sin(y) + \nu_e cos(x) , 0 \right] \,.

    Can only be defined in carthesian coordinates. 
    The solution is given by:

    .. math::
        u =  \left[\begin{array}{c} -sin(y) \\ cos(x) \\ 0 \end{array} \right] \,.
    """

    def __init__(self, b0=1.0):
        """
            Parameters
        ----------
        b0 : float
            Magnetic field (default: 1.0).
        """

        self._b = b0

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        #ux = -0.5*np.sin(y)
        #ux = -y**3
        ux = -np.sin(4*np.pi*x)*np.sin(4*np.pi*y)

        return ux


class ManufacturedSolutionVelocityElectrons_y:
    r"""Analytic solution :math:`u` of the system:

    .. math::

        \partial_t u = - \nabla \phi + u \times B + \nu \Delta u + f \,,\\
        0 = \nabla \phi- u_e \times B + \nu_e \Delta u_e + f_e \,, \\
        \nabla \cdot (u-u_e) = 0 \,.

    where :math:`f` is defined as follows: 

    .. math::

        f = \left[1 - b_0 cos(x) - \nu sin(y), 1 - b_0 sin(y) + \nu cos(x) , 0 \right] \,, 
        \\[2mm]
        f_e = \left[-1 + 0.5 b_0 cos(x) - \nu_e 0.5 sin(y), -1 + 0.5 b_0 sin(y) + \nu_e cos(x) , 0 \right] \,.

    Can only be defined in carthesian coordinates. 
    The solution is given by:

    .. math::
        u =  \left[\begin{array}{c} -sin(y) \\ cos(x) \\ 0 \end{array} \right] \,.
    """

    def __init__(self, b0=1.0):
        """
            Parameters
        ----------
        b0 : float
            Magnetic field (default: 1.0).
        """

        self._b = b0

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        #uy = 0.5*np.cos(x)
        #uy = x**3
        uy = -np.cos(4*np.pi*x)*np.cos(4*np.pi*y)

        return uy


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

    Can only be defined in carthesian coordinates. 
    The solution is given by:

    .. math::
        \phi =  x+y \,.
    """

    def __init__(self, b0=1.0):
        """
            Parameters
        ----------
        b0 : float
            Magnetic field (default: 1.0).
        """

        self._ab = b0

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        #phi = x+y
        #phi = x**2+y**2
        phi = np.cos(2*np.pi*x)+np.sin(2*np.pi*y)

        return phi


class ManufacturedSolutionForceterm_x:
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

    Can only be defined in carthesian coordinates. 
    The solution is given by:

    .. math::
        \phi =  x+y \,.
    """

    def __init__(self, b0=1.0, nu=1.0):
        """
            Parameters
        ----------
        b0 : float
            Magnetic field (default: 1.0).
        nu  : float
            Viscosity (default: 1.0)
        """

        self._b = b0
        self._nu = nu

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        #fx = 1.0 - self._nu*np.sin(y)-self._b* np.cos(x)
        #fx = 2.0*x - self._b*x**2+2.0*self._nu
        fx = -2.0*np.pi*np.sin(2*np.pi*x)+np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*self._b - \
            self._nu*8.0*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

        return fx


class ManufacturedSolutionForceterm_y:
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

    Can only be defined in carthesian coordinates. 
    The solution is given by:

    .. math::
        \phi =  x+y \,.
    """

    def __init__(self, b0=1.0, nu=1.0):
        """
            Parameters
        ----------
        b0 : float
            Magnetic field (default: 1.0).
        nu  : float
            Viscosity (default: 1.0)
        """

        self._b = b0
        self._nu = nu

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        #fy = 1.0 + self._nu*np.cos(x)-self._b* np.sin(y)
        #fy = 2*y - self._b*y**2 -2.0*self._nu
        fy = 2.0*np.pi*np.cos(2*np.pi*y)-np.sin(2*np.pi*x)*np.sin(2*np.pi*y)*self._b - \
            self._nu*8.0*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)

        return fy


class ManufacturedSolutionForcetermElectrons_x:
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

    Can only be defined in carthesian coordinates. 
    The solution is given by:

    .. math::
        \phi =  x+y \,.
    """

    def __init__(self, b0=1.0, nu_e=0.01):
        """
            Parameters
        ----------
        b0 : float
            Magnetic field (default: 1.0).
        nu  : float
            Viscosity (default: 1.0)
        """

        self._b = b0
        self._nu_e = nu_e

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        #fx = -1.0  - self._nu_e*0.5*np.sin(y)+ 0.5*self._b* np.cos(x)
        #fx = -2*x+self._b*x**3+6.0*y*self._nu_e
        fx = 2.0*np.pi*np.sin(2*np.pi*x)-np.cos(4*np.pi*x)*np.cos(4*np.pi*y)*self._b - \
            self._nu_e*32.0*np.pi**2*np.sin(4*np.pi*x)*np.sin(4*np.pi*y)

        return fx


class ManufacturedSolutionForcetermElectrons_y:
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

    Can only be defined in carthesian coordinates. 
    The solution is given by:

    .. math::
        \phi =  x+y \,.
    """

    def __init__(self, b0=1.0, nu_e=0.01):
        """
            Parameters
        ----------
        b0 : float
            Magnetic field (default: 1.0).
        nu  : float
            Viscosity (default: 1.0)
        """

        self._b = b0
        self._nu_e = nu_e

    # equilibrium ion velocity
    def __call__(self, x, y, z):
        """Velocity of ions and electrons."""
        #fy = -1.0 + self._nu_e*0.5*np.cos(x)+ 0.5*self._b* np.sin(y)
        #fy = -2*y + self._b * y**3 - 6.0*x*self._nu_e
        fy = -2.0*np.pi*np.cos(2*np.pi*y)+np.sin(4*np.pi*x)*np.sin(4*np.pi*y)*self._b - \
            self._nu_e*32.0*np.pi**2*np.cos(4*np.pi*x)*np.cos(4*np.pi*y)

        return fy
