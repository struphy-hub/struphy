"Analytic dispersion relations."

import cunumpy as xp
from numpy.polynomial import Polynomial
from scipy.optimize import fsolve

from struphy.dispersion_relations.base import ContinuousSpectra1D, DispersionRelations1D
from struphy.dispersion_relations.utilities import Zplasma
from struphy.fields_background.equils import set_defaults


class Maxwell1D(DispersionRelations1D):
    r"""
    Dispersion relation for Maxwell's equation in vacuum:

    .. math::

        \omega = c k \,.

    Parameters
    ----------
    c : float
        Speed of light. Remark that :math:`c=1.0` in Struphy units, see :class:`~struphy.models.toy.Maxwell`.
    """

    def __init__(self, c=1.0):
        super().__init__("light wave", velocity_scale="light", c=c)

    def __call__(self, k):
        """
        The evaluation of all branches of the 1d dispersion relation.

        Parameters
        ----------
        k : array_like
            Evaluation wave numbers.

        Returns
        -------
        omegas : dict
            A dictionary with key=branch_name and value=omega(k) (complex numpy.ndarray).
        """

        self._branches["light wave"] = self.params["c"] * k

        return self.branches


class MHDhomogenSlab(DispersionRelations1D):
    r"""
    Dispersion relation for linear MHD equations in homogeneous background :math:`(n_0,p_0,\mathbf B_0)`
    and wave propagation along :math:`z`-axis in Struphy units (see :class:`~struphy.models.fluid.LinearMHD`):

    .. math::

        \textnormal{shear Alfvén}:\quad &\omega = v_\textnormal{A} k\frac{B_{0z}}{|\mathbf B_0|}\,,

        \textnormal{fast (+) and slow (-) magnetosonic}:\quad &\omega = k\sqrt{\frac{1}{2}(c_\textnormal{S}^2+v_\textnormal{A}^2)(1\pm\sqrt{1-\delta})}\,,\quad\delta=\frac{4B_{0z}^2c_\textnormal{S}^2v_\textnormal{A}^2}{(c_\textnormal{S}^2+v_\textnormal{A}^2)^2|\mathbf B_0|^2}\,,

    where :math:`v_\textnormal{A}^2=|\mathbf B_0|^2/n_0` is the Alfvén velocity
    and :math:`c_\textnormal{S}^2=\gamma\,p_0/n_0` is the speed of sound.

    Parameters
    ----------
    B0x : float
        x-component of magnetic field (default: 0.).
    B0y : float
        y-component of magnetic field (default: 1.).
    B0z : float
        z-component of magnetic field (default: 1.).
    p0 : float
        Plasma pressure (default: 0.1).
    n0 : float
        Ion number density (default: 1.).
    gamma : float
        Adiabatic index (default: 5/3).
    """

    def __init__(self, B0x=0.0, B0y=1.0, B0z=1.0, p0=0.1, n0=1.0, gamma=5 / 3):
        super().__init__(
            "shear Alfvén",
            "slow magnetosonic",
            "fast magnetosonic",
            velocity_scale="alfvén",
            B0x=B0x,
            B0y=B0y,
            B0z=B0z,
            p0=p0,
            n0=n0,
            gamma=gamma,
        )

    def __call__(self, k):
        """
        The evaluation of all branches of the 1d dispersion relation.

        Parameters
        ----------
        k : array_like
            Evaluation wave numbers.

        Returns
        -------
        omegas : dict
            A dictionary with key=branch_name and value=omega(k) (complex numpy.ndarray).
        """

        Bsquare = self.params["B0x"] ** 2 + self.params["B0y"] ** 2 + self.params["B0z"] ** 2

        # Alfvén velocity and speed of sound
        vA = xp.sqrt(Bsquare / self.params["n0"])

        cS = xp.sqrt(self.params["gamma"] * self.params["p0"] / self.params["n0"])

        # shear Alfvén branch
        self._branches["shear Alfvén"] = vA * k * self.params["B0z"] / xp.sqrt(Bsquare)

        # slow/fast magnetosonic branch
        delta = (4 * self.params["B0z"] ** 2 * cS**2 * vA**2) / ((cS**2 + vA**2) ** 2 * Bsquare)

        self._branches["slow magnetosonic"] = xp.sqrt(1 / 2 * k**2 * (cS**2 + vA**2) * (1 - xp.sqrt(1 - delta)))
        self._branches["fast magnetosonic"] = xp.sqrt(1 / 2 * k**2 * (cS**2 + vA**2) * (1 + xp.sqrt(1 - delta)))

        return self.branches


class ExtendedMHDhomogenSlab(DispersionRelations1D):
    r"""
    Dispersion relation for linear extended MHD equations in homogeneous background :math:`(n_0,p_0,\mathbf B_0)` 
    and wave propagation along :math:`z`-axis in Struphy units (see :class:`~struphy.models.fluid.LinearExtendedMHD`).
    The linear mode analysis is performed on the following system:

    .. math::

        \begin{align}
        -i \omega m n_0 \tilde{\mathbf U} + i \mathbf k \tilde p &= i\frac{B_0}{\mu_0}(\mathbf k \times \tilde{\mathbf B}) \times \mathbf e_z\,,
        \\
        -i\omega \tilde{\mathbf B} - B_0 i  \mathbf k \times ( \tilde{\mathbf U} \times \mathbf e_z ) \color{red} + i \mathbf k \times \left[\frac{B_0}{q n_0\mu_0} (i\mathbf k \times \tilde{\mathbf B}  ) \times \mathbf e_z \right] \color{default} &= 0\,,
        \\[3mm]
        -i\omega \tilde p + i \gamma p_0 \mathbf k \cdot \tilde{\mathbf U} &= 0\,.
        \end{align}

    Computed are the four roots of

    .. math::

        (x - \omega_0^2) \Big\{ x^3 - x^2\Big[(c_s^2 + v_A^2)k^2 + v_A^2 k_z^2 + \omega_0^2\Big] + x\Big[c_s^2 k^2 (2 v_A^2 k_z^2 + \omega_0^2) + v_A^4 k_z^2 k^2 \Big] - c_s^2 v_A^4 k_z^4 k^2\Big\} = 0

    where :math:`x = \omega^2`, :math:`v_\textnormal{A}^2=|\mathbf B_0|^2/n_0` denotes the Alfvén velocity, 
    :math:`c_s^2=\gamma\,p_0/n_0` is the speed of sound, and :math:`\omega_0 = v_A^2 k_z k/\Omega_i`
    stands for the first root where 
    :math:`\Omega_i = |\mathbf B_0|/\epsilon` is the cyclotron frequency 
    with :math:`\epsilon = 1/(\hat \Omega_i \hat t)`.
    """

    def __init__(self, B0x=0.0, B0y=0.0, B0z=1.0, p0=0.1, n0=1.0, gamma=5 / 3, eps=0.1):
        super().__init__(
            "shear Alfvén",
            "slow magnetosonic",
            "fast magnetosonic",
            "compressional Alfvén",
            velocity_scale="alfvén",
            B0x=B0x,
            B0y=B0y,
            B0z=B0z,
            p0=p0,
            n0=n0,
            gamma=gamma,
            eps=eps,
        )

    def __call__(self, k):
        """
        The evaluation of all branches of the 1d dispersion relation.

        Parameters
        ----------
        k : array_like
            Evaluation wave numbers.

        Returns
        -------
        omegas : dict
            A dictionary with key=branch_name and value=omega(k) (complex numpy.ndarray).
        """

        Bsquare = self.params["B0x"] ** 2 + self.params["B0y"] ** 2 + self.params["B0z"] ** 2

        cos_theta = self.params["B0z"] / xp.sqrt(Bsquare)

        # Alfvén velocity, speed of sound and cyclotron frequency
        vA = xp.sqrt(Bsquare / self.params["n0"])

        cS = xp.sqrt(self.params["gamma"] * self.params["p0"] / self.params["n0"])

        Omega_i = xp.sqrt(Bsquare) / self.params["eps"]

        # auxiliary functions
        def omega_0(k):
            return vA**2 * k**2 * cos_theta

        def b(k):
            return -(cS**2 + vA**2) * k**2 - vA**2 * k**2 * cos_theta**2 - omega_0(k) ** 2

        def c(k):
            return cS**2 * k**2 * (2.0 * vA**2 * k**2 * cos_theta**2 + omega_0(k) ** 2) + vA**4 * k**4 * cos_theta**2

        def d(k):
            return -(cS**2) * vA**4 * k**6 * cos_theta**4

        def discriminant(k):
            return (
                18.0 * b(k) * c(k) * d(k)
                - 4.0 * b(k) ** 3 * d(k)
                + b(k) ** 2 * c(k) ** 2
                - 4.0 * c(k) ** 3
                - 27.0 * d(k) ** 2
            )

        # solve
        out = xp.zeros((k.size, 4), dtype=complex)
        for i, ki in enumerate(k):
            p0 = Polynomial([-(omega_0(ki) ** 2), 1.0])
            p1 = Polynomial([d(ki), c(ki), b(ki), 1.0])
            poly = p0 * p1
            out[i] = poly.roots()

        self._branches["slow magnetosonic"] = out[:, 0]
        self._branches["shear Alfvén"] = out[:, 1]
        self._branches["fast magnetosonic"] = out[:, 2]
        self._branches["compressional Alfvén"] = out[:, 3]

        return self.branches


class FluidSlabITG(DispersionRelations1D):
    r"""
    Dispersion relation for ion fluid equations with adiabatic electrons
    in homogeneous background :math:`(n_0,p_0,\mathbf B_0)`, with a constant density and a
    temperature gradient in :math:`x`-direction. The Braginskii closure for drift cancellations
    is taken into account (strong magentic field).
    The dispersion relation is

    .. math::

        \omega^3 - (Z + \gamma)v_i^2 k_z^2 \omega + Z \frac{v_i^4}{v^*} k_z^2 k_y = 0\,,

    where :math:`v_i^2=k_B T_0/m` denotes the ion thermal velocity and

    .. math::

        v^* = \frac{\Omega_i}{\partial_x T_0/T_0}\,,\qquad \Omega_i = \frac{Ze B_0}{m}\,.

    The dispersion relation is calculated as a function of :math:`k_y` for fixed :math:`k_z`.
    Instabilites occur when

    .. math::

        \frac{k_y}{k_z} > \frac{2}{3^{3/2}}\frac{\sqrt{Z + \gamma}^3}{Z^2} \frac{v^*}{v_i} \,, \qquad \frac{v^*}{v_i} = \frac{\partial_x T_0/T_0}{\rho_i}\,.
    """

    def __init__(self, vstar=10.0, vi=1.0, Z=1.0, kz=1.0, gamma=5 / 3):
        super().__init__(
            "wave 1", "wave 2", "wave 3", velocity_scale="thermal", vstar=vstar, vi=vi, Z=Z, kz=kz, gamma=gamma
        )

    def __call__(self, k):
        """
        The evaluation of all branches of the 1d dispersion relation.

        Parameters
        ----------
        k : array_like
            Evaluation wave numbers.

        Returns
        -------
        omegas : dict
            A dictionary with key=branch_name and value=omega(k) (complex numpy.ndarray).
        """

        # helper
        p = -(self.params["Z"] + self.params["gamma"]) * self.params["vi"] ** 2 * self.params["kz"] ** 2

        # stability threshold
        ky_crit = (
            2.0
            / 3.0 ** (3 / 2)
            * (self.params["Z"] + self.params["gamma"]) ** (3 / 2)
            / self.params["Z"]
            * self.params["vstar"]
            / self.params["vi"]
        )
        print(f"{ky_crit =}")
        self._k_crit["analytic threshold"] = ky_crit

        # auxiliary functions
        def q(k):
            return self.params["Z"] * self.params["vi"] ** 4 / self.params["vstar"] * self.params["kz"] ** 2 * k

        def discriminant(k):
            return -4.0 * p**3 - 27.0 * q(k) ** 2

        # solve
        out = xp.zeros((k.size, 3), dtype=complex)
        for i, ki in enumerate(k):
            poly = Polynomial([q(ki), p, 0.0, 1.0])
            out[i] = poly.roots()

        self._branches["wave 1"] = out[:, 0]
        self._branches["wave 2"] = out[:, 1]
        self._branches["wave 3"] = out[:, 2]

        return self.branches


class ColdPlasma1D(DispersionRelations1D):
    r"""Dispersion relation for cold plasma model for homogeneous background :math:`(n_0,\mathbf B_0)`
    and wave propagation along z-axis :math:`(\mathbf k = k \mathbf e_z)` in Struphy units
    (see ``ColdPlasma`` in :ref:`models`):

    .. math::

        \left[ \left( \omega^2 - |k|^2 \right) \mathbb I + \mathbf k \otimes \mathbf k + i \frac{\alpha^2}{\varepsilon_c} \omega \sigma_c \right] \mathbf E = 0\,,

    where :math:`\left( \omega^2 - |k|^2 \right) \mathbb I + \mathbf k \otimes \mathbf k + i \frac{\alpha^2}{\varepsilon_c} \omega \sigma_c = \epsilon`
    is the dielectric tensor, :math:`\alpha` is the plasma frequency in units of the electron cyclotron frequency,
    :math:`1/\varepsilon_c` is the electron cyclotron frequency in struphy units,
    and :math:`\sigma_c = \left( \mathbb I - i Q / \varepsilon_c \omega \right)^{-1} i n_0 / \varepsilon_c \omega`,
    with :math:`Q` being an operator which, if applied to vector :math:`\mathbf v`, returns :math:`\mathbf v \times \mathbf B_0`.
    """

    def __init__(self, **params):
        # set default parameters
        params_default = {"B0x": 0.0, "B0y": 0.0, "B0z": 1.0, "n0": 1.0, "alpha": 1.0, "epsilon": 1.0}

        params_all = set_defaults(params, params_default)

        super().__init__("ion-cyclotron wave", "electron-cyclotron wave", "L-wave", "R-wave", **params)

    def __call__(self, kvec):
        # One complex array for each branch
        tmps = []
        for n in range(self.nbranches):
            tmps += [xp.zeros_like(kvec, dtype=complex)]

        ########### Model specific part ##############################

        # angle between k and magnetic field
        if self.params["B0z"] == 0:
            theta = xp.pi / 2
        else:
            theta = xp.arctan(xp.sqrt(self.params["B0x"] ** 2 + self.params["B0y"] ** 2) / self.params["B0z"])
        print(theta)
        cos2 = xp.cos(theta) ** 2

        neq = self.params["n0"]

        # powers of parameters
        B2 = self.params["B0x"] ** 2 + self.params["B0y"] ** 2 + self.params["B0z"] ** 2
        alpha2 = self.params["alpha"] ** 2
        alpha4 = self.params["alpha"] ** 4
        alpha6 = self.params["alpha"] ** 6
        eps2 = self.params["epsilon"] ** 2
        eps4 = self.params["epsilon"] ** 4
        eps6 = self.params["epsilon"] ** 6
        k2vec = kvec**2

        for n, k2 in enumerate(k2vec):
            # polynomial coefficients in order of increasing degree
            # 0th degree
            a = B2 * k2**2 * neq * alpha2 * eps2 * cos2
            # 1st degree in omega^2
            b = (
                -(neq**3) * alpha6
                - B2 * k2 * neq * alpha2 * eps2
                - 2 * k2 * neq**2 * alpha4 * eps2
                - B2 * k2 * neq * alpha2 * eps2 * cos2
                - B2 * k2**2 * eps4
                - k2**2 * neq * alpha2 * eps4
            )
            # 2nc degree in omega^2
            c = (
                B2 * neq * alpha2 * eps2
                + 3 * neq**2 * alpha4 * eps2
                + 2 * B2 * k2 * eps4
                + 4 * k2 * neq * alpha2 * eps4
                + k2**2 * eps6
            )
            # 3rd degree in omega^2
            d = -B2 * eps4 - 3 * neq * alpha2 * eps4 - 2 * k2 * eps6
            # 4th degree in omega^2
            e = eps6

            # determinant in polynomial form
            det = xp.polynomial.Polynomial([a, b, c, d, e])

            # solutions
            sol = xp.sqrt(xp.abs(det.roots()))
            # Ion-cyclotron branch
            tmps[0][n] = sol[0]
            # Electron-cyclotron branch
            tmps[1][n] = sol[1]
            # L-branch
            tmps[2][n] = sol[2]
            # R- branch
            tmps[3][n] = sol[3]

        ##############################################################

        # fill output dictionary
        dict_disp = {}
        for name, tmp in zip(self.branches, tmps):
            dict_disp[name] = tmp

        return dict_disp


class CurrentCoupling6DParallel(DispersionRelations1D):
    r"""
    Dispersion relation for linearized hybrid MHD-Vlasov model (current coupling scheme) in Struphy units for homogeneous background :math:`(n_0=1,p_0,\mathbf B_0=B_0\mathbf e_z)`, wave propagation along z-axis and EP distribution function

    .. math::

        f_0=\frac{1}{\pi^{3/2}v_\textnormal{th}^3}\exp\left[-\frac{(v_\parallel-v_0)^2+v_\perp^2}{v_\textnormal{th}^2}\right]\,.

    The two branches of the dispersion relation are circularly polarized shear Alfvén waves (R-wave and L-wave)

    .. math::

        D_\textnormal{R/L}(\omega,k)=\omega^2-B_0^2k^2\pm\nu_\textnormal{h}\omega\frac{Z_\textnormal{h}B_0}{A_\textnormal{b}}\kappa\mp\nu_\textnormal{h}\frac{Z_\textnormal{h}B_0}{A_\textnormal{b}}\kappa v_0 k+\nu_\textnormal{h}\frac{Z_\textnormal{h}^2B_0^2}{A_\textnormal{h}A_\textnormal{b}}\kappa^2\frac{\omega-kv_0}{kv_\textnormal{th}}Z(\xi^\pm)\,,

    and standard sound waves

    .. math::

        \omega^2=\gamma p_0 k^2\,,

    where :math:`\xi^\pm=(\omega-kv_0\pm Z_\textnormal{h}B_0\kappa/A_\textnormal{h})/kv_\textnormal{th}` and :math:`Z(\xi)=\sqrt{\pi}\exp(-\xi^2)[i-\textnormal{erfi}(\xi)]` is the plasma dispersion function.

    Parameters
    ----------
    **params
        Keyword arguments that characterize the dispersion relation.
            * B0 : float
                Magnetic field strength (default: 1.).
            * p0 : float
                Plasma pressure (default: 0.5).
            * gamma : float
                Adiabatic index (default: 5/3).
            * Ab : int
                Bulk species mass number (default: 1).
            * Ah : int
                Energetic species mass number (default: 1).
            * Zh : int
                Energetic species charge number (default: 1).
            * vth : float
                Energetic species thermal velocity (default: 1.).
            * v0 : float
                Energetic species shift of Maxwellian (default: 2.).
            * nuh : float
                Ratio of energetic/bulk number densities (default: 0.5).
            * nb : float
                Bulk species number density in units of 1e20 / m^3 (default: 0.0005185219355).
    """

    def __init__(self, **params):
        # set default parameters
        params_default = {
            "B0": 1.0,
            "p0": 0.5,
            "gamma": 5 / 3,
            "Ab": 1,
            "Ah": 1,
            "Zh": 1,
            "vth": 1.0,
            "v0": 2.0,
            "nuh": 0.05,
            "nb": 0.0005185219355,
        }

        params_all = set_defaults(params, params_default)

        super().__init__("shear_Alfvén_R", "shear_Alfvén_L", "sound", **params_all)

        # some constants
        mp = 1.672621924e-27
        mu = 1.256637062e-6
        ee = 1.602176634e-19

        # calculate coupling parameter alpha_c from bulk number density and mass number
        self._kappa = ee * xp.sqrt(mu * self.params["Ab"] * self.params["nb"] * 1e20 / mp)

    def __call__(self, k, method="newton", tol=1e-10, max_it=100):
        """
        Solves the dispersion relation for given wave numbers.

        Parameters
        ----------
        k : array_like
            The wave numbers for which to evaluate the dispersion relation.

        method : str, optional
            Which numerical method/solver to be used (either "newton" or "fsolve").

        tol : float, optional
            Stop tolerance in numerical solution of dispersion relation.

        max_it : int, optional
            Maximum number of iterations in case of Newton solver.

        Returns
        -------
        omegas : dict
            A dictionary with key=branch_name and value=omega(k) (complex numpy.ndarray).
        """

        # One complex array for each branch
        tmps = []
        for _ in range(self.nbranches):
            tmps += [xp.zeros_like(k, dtype=complex)]

        ########### Model specific part ##############################

        # sound waves
        tmps[2][:] = self.params["gamma"] * self.params["p0"] * k

        # solve dispersion relation for R-/L-waves for fixed k with Newton method
        for i, ki in enumerate(k):
            # choose initial guess w = B0*k for first iteration and result from last k otherwise
            if i == 0:
                wR = [self.params["B0"] * ki, 0.0]
                wL = [self.params["B0"] * ki, 0.0]
            else:
                wR = [xp.real(tmps[0][i - 1]), xp.imag(tmps[0][i - 1])]
                wL = [xp.real(tmps[1][i - 1]), xp.imag(tmps[1][i - 1])]

            # apply solver
            if method == "newton":
                # R -wave
                counter = 0

                Dr, Di = self.D_RL(wR, ki, +1)

                while xp.abs(Dr + Di * 1j) > tol or counter == max_it:
                    # derivative
                    Drp, Dip = self.D_RL(wR, ki, +1, 1)

                    # update
                    wR[0] = wR[0] - xp.real((Dr + Di * 1j) / (Drp + Dip * 1j))
                    wR[1] = wR[1] - xp.imag((Dr + Di * 1j) / (Drp + Dip * 1j))

                    Dr, Di = self.D_RL(wR, ki, +1)
                    counter += 1

                # L -wave
                counter = 0

                Dr, Di = self.D_RL(wL, ki, -1)

                while xp.abs(Dr + Di * 1j) > tol or counter == max_it:
                    # derivative
                    Drp, Dip = self.D_RL(wL, ki, -1, 1)

                    # update
                    wL[0] = wL[0] - xp.real((Dr + Di * 1j) / (Drp + Dip * 1j))
                    wL[1] = wL[1] - xp.imag((Dr + Di * 1j) / (Drp + Dip * 1j))

                    Dr, Di = self.D_RL(wL, ki, -1)
                    counter += 1

            elif method == "fsolve":
                wR = fsolve(self.D_RL, x0=wR, args=(ki, +1, 0), xtol=tol)
                wL = fsolve(self.D_RL, x0=wL, args=(ki, -1, 0), xtol=tol)

            else:
                raise NotImplementedError("Only methods newton and fsolve available!")

            tmps[0][i] = wR[0] + 1j * wR[1]
            tmps[1][i] = wL[0] + 1j * wL[1]

        ##############################################################

        # fill output dictionary
        omegas = {}
        for name, tmp in zip(self.branches, tmps):
            omegas[name] = tmp

        return omegas

    def D_RL(self, w, k, pol, der=0):
        r"""
        Dispersion relation :math:`D_\mathrm{R/L}(\omega,k)=0` (or its first derivative with respect to :math:`\omega`) for R- and L- shear Alfvén waves.

        Parameters
        ----------
        w : list
            The complex frequencies at which to evaluate the dispersion relation. w[0] is the real part, w[1] the imaginary part.

        k : array_like
            The real wave numbers at which to evaluate the dispersion relation.

        pol : int
            The polarization of the wave (+1 : R-wave, -1 : L-wave).

        der : int, optional
            Whether to evaluate the dispersion relation (der = 0) or its first derivative with respect to w (der = 1).

        Returns
        -------
        d_real : ndarray
            The real part of the evaluated dispersion relation.

        d_imag : ndarray
            The imaginary part of the evaluated dispersion relation.
        """

        assert der == 0 or der == 1, 'Parameter "der" must be either 0 or 1.'
        assert pol == 1 or pol == -1, "Polarization must be either +1 (R) or -1 (L)."

        w = w[0] + 1j * w[1]

        vth = self.params["vth"]
        v0 = self.params["v0"]
        B0 = self.params["B0"]
        Zh = self.params["Zh"]
        Ah = self.params["Ah"]
        Ab = self.params["Ab"]
        nuh = self.params["nuh"]

        xi = (w - k * v0 + pol * B0 * Zh * self._kappa / Ah) / (k * vth)
        xip = 1 / (k * vth)

        if der == 0:
            out = w**2
            out -= B0**2 * k**2
            out += pol * nuh * w * Zh * B0 / Ab * self._kappa
            out -= pol * nuh * Zh * B0 / Ab * self._kappa * v0 * k
            out += nuh * Zh**2 * B0**2 / (Ah * Ab) * self._kappa**2 * (w - k * v0) / (k * vth) * Zplasma(xi, 0)

        else:
            out = 2 * w
            out += pol * nuh * Zh * B0 / Ab * self._kappa
            out += (
                nuh
                * Zh**2
                * B0**2
                / (Ah * Ab)
                * self._kappa**2
                / (k * vth)
                * (Zplasma(xi, 0) + (w - k * v0) * Zplasma(xi, 1) * xip)
            )

        return xp.real(out), xp.imag(out)


class PressureCouplingFull6DParallel(DispersionRelations1D):
    r"""Dispersion relation for linear MHD equations coupled to the Vlasov equation with Full Pressure Coupling scheme
    for homogeneous background :math:`(n_0,p_0,\mathbf B_0)`, wave propagation along z-axis in Struphy units and space-homogeneous shifted Maxwellian energetic particles distribution :math:`f_h = f_{h,0} + \tilde{f_h}`
    where :math:`f_{h,0}(v_{\parallel}, v_{\perp}) = n_0 \frac{1}{\sqrt{\pi}} \frac{1}{\hat{v_{\parallel}}} e^{- (v_{\parallel} - u_0)^2 / \hat{v}^2_{\parallel} } \frac{1}{\pi} \frac{1}{\hat{v^2_{\perp}}} e^{- v^2_{\perp} / \hat{v}^2_{\perp}}`
    here, :math:`u_0` is a velocity shift in the parallel direction (see ``PC_LinMHD_6d_full`` in :ref:`models`):

    :math:`\textnormal{shear Alfvén (R) and (L) wave}` :

    .. math::

        \omega^2 = v_\textnormal{A}^2 k^2\frac{B_{0z}^2}{|\mathbf B_0|^2} + \omega k \nu_h &\left[ \frac{\omega_c}{\omega} \left\{ \left( 1 - \frac{\hat{v}^2_\perp}{\hat{v}^2_\parallel}\right) \hat{v}_\parallel \left( \pm Y_3 \mp \frac{\omega - \omega_c}{\hat{v}_\parallel k_\parallel} Y_2 \right) + u_0 \frac{\hat{v}^2_\perp}{\hat{v}^2_\parallel} \left( \pm Y_2 \mp \frac{\omega - \omega_c}{\hat{v}_\parallel k_\parallel} Y_2 \right) \right\} \right.

        &- \left. \frac{\hat{v}^2_\perp}{\hat{v}^2_\parallel} \left( Y_3 - \frac{\omega \mp \omega_c}{\hat{v}_\parallel k_\parallel} Y_2 - \frac{u_0}{\hat{v}_\parallel} Y_2 + \frac{\omega \mp \omega_c}{\hat{v}^2_\parallel k_\parallel} u_0 Y_1 \right)\right]\,,

    :math:`\textnormal{sonic wave}` :

    .. math::

        \omega^2 =v_\textnormal{A}^2 k^2 - 2 \omega k_\parallel \nu_h \hat{v}_\parallel X_4 \,

    where :math:`v_\textnormal{A}^2=|\mathbf B_0|^2/n_0` is the Alfvén velocity and :math:`c_\textnormal{S}^2=\gamma\,p_0/n_0` is the speed of sound.

    Variaous integrals are defined as follows

    .. math::

        X_4(\xi_0, a_0):= \frac{1}{\sqrt{\pi}} \int^\infty_\infty \frac{(t+a)^3 t }{t - \xi_0} e^{- t^2} dt \, \quad \qquad &= \frac{5}{4} \xi_0 + \frac{3}{2} a_0 + (\xi_0 + a_0)^3 [1 + \xi_0 Z(\xi_0)] \,,

        Y_1(\xi_-, \xi_+, a_0) := \frac{1}{\sqrt{\pi}} \int^\infty_\infty \frac{t+a_0}{(t-\xi_-)(t-\xi_+)} e^{-t^2} dt &= Z(\xi_-) + (\xi_+ + a_0) \frac{Z(\xi_-) - Z(\xi_+)}{\xi_- - \xi_+} \,,

        Y_2(\xi_-, \xi_+, a_0) := \frac{1}{\sqrt{\pi}} \int^\infty_\infty \frac{(t+a)^2}{(t-\xi_-)(t-\xi_+)} e^{-t^2} dt &= 1 + (\xi_- + \xi_+ + 2a_0) Z(\xi_-)

        &+ (\xi_+ + a_0)^2 \frac{Z(\xi_-) - Z(\xi_+)}{\xi_- - \xi_+} \,,

        Y_3(\xi_-, \xi_+, a_0) := \frac{1}{\sqrt{\pi}} \int^\infty_\infty \frac{(t+a)^3}{(t-\xi_-)(t-\xi_+)} e^{-t^2} dt &= \xi_- + \xi_+ + 3a_0

        &+ [\xi_-^2 + \xi_- \xi_+ + \xi_+^2 + 3a_0(\xi_- + \xi_+) + 3a_0^2] Z(\xi_-)

        &+ (\xi_+ + a_0)^3 \frac{Z(\xi_-) - Z(\xi_+)}{\xi_- - \xi_+} \,,

    where :math:`\xi_0 = \frac{\omega / k_\parallel - u_0}{\hat{v}_\parallel}, \quad \xi_\pm = \frac{(\omega \pm \omega_c) / k_\parallel - u_0}{\hat{v}_\parallel}, \quad a_0 = \frac{u_0}{\hat{v}_\parallel}`
    and :math:`Z(\xi) = \frac{1}{\sqrt{\pi}} \int^\infty_\infty \frac{e^{- t^2}}{t - \xi} dt = i \sqrt{\pi} e^{- \xi^2} ( 1 + \text{erf}(i\xi))` is the plasma dispersion function.

    """

    def __init__(self, params):
        super().__init__("shear Alfvén_R", "shear Alfvén_L", "sonic", **params)

    def __call__(self, k, tol=1e-10):
        """
        Solves the dispersion relation for given wave numbers.

        Parameters
        ----------
        k : array_like
            The wave numbers for which to evaluate the dispersion relation.

        tol : float, optional
            Stop tolerance in numerical solution of dispersion relation.

        Returns
        -------
        omegas : dict
            A dictionary with key=branch_name and value=omega(k) (complex ndarray).
        """

        # One complex array for each branch
        tmps = []
        for n in range(self.nbranches):
            tmps += [xp.zeros_like(k, dtype=complex)]

        ########### Model specific part ##############################

        # solve omega
        for i, ki in enumerate(k):
            # choose initial guess wRL = vA*k, wS = cS*k for first iteration and result from last k otherwise
            if i == 0:
                wR = [1 * ki, 0.0]  # TODO: use vA
                wL = [1 * ki, 0.0]  # TODO: use vA
                wS = [1 * ki, 0.0]  # TODO: use cS
            else:
                wR = [xp.real(tmps[0][i - 1]), xp.imag(tmps[0][i - 1])]
                wL = [xp.real(tmps[1][i - 1]), xp.imag(tmps[1][i - 1])]
                wS = [xp.real(tmps[2][i - 1]), xp.imag(tmps[2][i - 1])]

            # R/L shear Alfvén wave
            sol_R = fsolve(self.D_RL, x0=wR, args=(ki, +1), xtol=tol)
            sol_L = fsolve(self.D_RL, x0=wL, args=(ki, -1), xtol=tol)

            tmps[0][i] = sol_R[0] + 1j * sol_R[1]
            tmps[1][i] = sol_L[0] + 1j * sol_L[1]

            # sonic wave
            sol_S = fsolve(self.D_sonic, x0=wS, args=(ki,), xtol=tol)

            tmps[2][i] = sol_S[0] + 1j * sol_S[1]

        ##############################################################

        # fill output dictionary
        omegas = {}
        for name, tmp in zip(self.branches, tmps):
            omegas[name] = tmp

        return omegas

    def D_RL(self, w, k, pol):
        r"""
        Dispersion relation :math:`D_\mathrm{R/L}(\omega,k)=0` for R- and L- shear Alfvén waves.

        Parameters
        ----------
        w : list
            The complex frequencies at which to evaluate the dispersion relation. w[0] is the real part, w[1] the imaginary part.

        k : array_like
            The real wave numbers at which to evaluate the dispersion relation.

        pol : int
            The polarization of the wave (+1 : R-wave, -1 : L-wave).

        Returns
        -------
        d_real : ndarray
            The real part of the evaluated dispersion relation.

        d_imag : ndarray
            The imaginary part of the evaluated dispersion relation.
        """

        assert pol == 1 or pol == -1, "Polarization must be either +1 (R) or -1 (L)."

        w = w[0] + 1j * w[1]

        # Alfvén velocity and speed of sound
        # TODO: call the parameters from the yml file.
        wc = 1.0
        u0 = 2.5  # TODO
        vpara = 1.0  # TODO
        vperp = 1.0  # TODO
        vth = 1.0

        vA = xp.sqrt((self.params["B0x"] ** 2 + self.params["B0y"] ** 2 + self.params["B0z"] ** 2) / self.params["n0"])
        # cS = xp.sqrt(self.params['beta']*vA)
        cS = 1.0

        a0 = u0 / vpara  # TODO
        nu = 0.05  # TODO

        zp = self._zetap(w, k, u0, vpara, wc)
        zm = self._zetam(w, k, u0, vpara, wc)

        y1 = self._Y1(zm, zp, a0)
        y2 = self._Y2(zm, zp, a0)
        y3 = self._Y3(zm, zp, a0)

        # R-wave
        if pol == 1:
            c1 = (
                w**2
                - vA**2 * k**2
                - w
                * nu
                * k
                * (
                    wc / w * u0 * (+y2 - (w - wc) / k / vpara * y1)
                    - (vperp**2 / vpara)
                    * (y3 - (w - wc) / k / vpara * y2 - u0 / vpara * y2 + (w - wc) / k / vpara**2 * u0 * y1)
                )
            )

        # L-wave:
        else:
            c1 = (
                w**2
                - vA**2 * k**2
                - w
                * nu
                * k
                * (
                    wc / w * u0 * (-y2 + (w + wc) / k / vpara * y1)
                    - (vperp**2 / vpara)
                    * (y3 - (w + wc) / k / vpara * y2 - u0 / vpara * y2 + (w + wc) / k / vpara**2 * u0 * y1)
                )
            )

        return xp.real(c1), xp.imag(c1)

    def D_sonic(self, w, k):
        r"""
        Dispersion relation :math:`D_\mathrm{sonic}(\omega,k)=0` for sonic waves.

        Parameters
        ----------
        w : list
            The complex frequencies at which to evaluate the dispersion relation. w[0] is the real part, w[1] the imaginary part.

        k : array_like
            The real wave numbers at which to evaluate the dispersion relation.

        Returns
        -------
        d_real : ndarray
            The real part of the evaluated dispersion relation.

        d_imag : ndarray
            The imaginary part of the evaluated dispersion relation.
        """

        w = w[0] + 1j * w[1]

        # Alfvén velocity and speed of sound
        # TODO: call the parameters from the yml file.
        wc = 1.0
        u0 = 2.5  # TODO
        vpara = 1.0  # TODO
        vperp = 1.0  # TODO
        vth = 1.0

        vA = xp.sqrt((self.params["B0x"] ** 2 + self.params["B0y"] ** 2 + self.params["B0z"] ** 2) / self.params["n0"])
        # cS = xp.sqrt(self.params['beta']*vA)
        cS = 1.0

        a0 = u0 / vpara  # TODO
        nu = 0.05  # TODO

        z0 = self._zeta0(w, k, u0, vpara)
        x4 = self._X4(z0, a0)

        c1 = w**2 - k**2 * cS**2 + 2 * w * k * nu * vpara * x4

        return xp.real(c1), xp.imag(c1)

    # private methods:
    # ----------------

    # define integrals and functions

    def _Y1(self, xi, eta, a):
        y1 = Zplasma(xi)
        y1 += (eta + a) * (Zplasma(xi) - Zplasma(eta)) / (xi - eta)

        return y1

    def _Y2(self, xi, eta, a):
        y2 = 1.0
        y2 += (xi + eta + 2 * a) * Zplasma(xi)
        y2 += (eta + a) ** 2 * (Zplasma(xi) - Zplasma(eta)) / (xi - eta)

        return y2

    def _Y3(self, xi, eta, a):
        y3 = xi + eta + 3 * a
        y3 += (xi**2 + xi * eta + eta**2 + 3 * a * (xi + eta) + 3 * a**2) * Zplasma(xi)
        y3 += (eta + a) ** 3 * (Zplasma(xi) - Zplasma(eta)) / (xi - eta)

        return y3

    def _X4(self, xi, a):
        return 5 / 4 * xi + 3 / 2 * a + (xi + a) ** 3 * (1 + xi * Zplasma(xi))

    def _zeta0(self, w, k, u0, vpara):
        return (w / k - u0) / vpara

    def _zetap(self, w, k, u0, vpara, wc):
        return ((w + wc) / k - u0) / vpara

    def _zetam(self, w, k, u0, vpara, wc):
        return ((w - wc) / k - u0) / vpara


class MhdContinousSpectraShearedSlab(ContinuousSpectra1D):
    r"""
    Continuous shear Alfvén and slow sound spectra along x-direction in slab geometry with side lengths :math:`L_x=a,L_y=2\pi a, L_z=2\pi\,R_0` in Struphy units.

    The profiles in Cartesian coordinates :math:`(x, y, z)` are

    .. math::

        \mathbf B_0 &= \mathbf B_0(x) = B_{0z}(x)\left( \mathbf e_z + \frac{a}{q(x) R_0}\mathbf e_y \right)\,,

        p_0 &= p_0(x)

        n_0 &= n_0(x)\,.

    The continuous spectra are then given by

    .. math::

        \textnormal{shear Alfvén}:\quad & \omega^2(x)=\frac{B_{0z}(x)^2}{n_0(x)}\frac{1}{R_0^2}\left(n+\frac{m}{q(x)}\right)^2\,

        \textnormal{slow sound}:\quad & \omega^2(x)=\frac{\gamma p_0(x)B_{0z}(x)^2}{n_0(x)\,[\gamma p_0(x) + B_{0y}(x)^2 + B_{0z}(x)^2]}\frac{1}{R_0^2}\left(n+\frac{m}{q(x)}\right)^2\,.

    Parameters
    ----------
    **params
        Keyword arguments that characterize the dispersion relation.
            * a : float
                "Minor" radius (must be compatible with :math:`L_x=a` and :math:`L_y=2\pi a`, default: 1.).
            * R0 : float
                "Major" radius (must be compatible with :math:`L_z=2\pi R_0`, default: 3.).
            * gamma : float
                Adiabatic index (default: 5/3).
            * Bz : callable
                Profile of axial magnetic field Bz=Bz(x) (default: 1. - 0*x).
            * p : callable
                Pressure profile p=p(x) (default: 0.5 - 0*x).
            * rho : callable
                Profile of mass density rho=rho(x) (default: 1. - 0*x).
            * q : callable
                Safety factor profile q=q(x) (default: 1.1 + 0.7*x**2).
    """

    def __init__(self, **params):
        # set default parameters
        params_default = {
            "a": 1.0,
            "R0": 3.0,
            "gamma": 5 / 3,
            "Bz": lambda x: 1.0 - 0 * x,
            "p": lambda x: 0.5 - 0 * x,
            "rho": lambda x: 1.0 - 0 * x,
            "q": lambda x: 1.1 + 0.7 * x**2,
        }

        params_all = set_defaults(params, params_default)

        super().__init__("shear_Alfvén", "slow_sound", **params_all)

    def __call__(self, x, m, n):
        """
        The calculation of all continuous spectra.

        Parameters
        ----------
        x : array_like
            The x points at which the continuous spectra shall be evaluated.

        m, n : int
            Mode numbers in y- (m) and z- (n) direction.

        Returns
        -------
        specs : dict
            A dictionary with key=branch_name and value=omega(x) (ndarray).
        """

        # radial profiles
        Bz = self.params["Bz"]
        By = self._By
        p = self.params["p"]
        rho = self.params["rho"]
        F = self._F

        # other parameters
        gamma = self.params["gamma"]

        specs = {}

        # shear Alfvén continuum
        specs["shear_Alfvén"] = xp.sqrt(F(x, m, n) ** 2 / rho(x))

        # slow sound continuum
        specs["slow_sound"] = xp.sqrt(
            gamma * p(x) * F(x, m, n) ** 2 / (rho(x) * (gamma * p(x) + By(x) ** 2 + Bz(x) ** 2))
        )

        return specs

    # private methods:
    # ----------------

    def _By(self, x):
        """Poloidal magnetic field."""
        return self.params["a"] * self.params["Bz"](x) / (self.params["R0"] * self.params["q"](x))

    def _F(self, x, m, n):
        """Dot product of magnetic field with wavenumber k.B."""
        return m / self.params["a"] * self._By(x) + n / self.params["R0"] * self.params["Bz"](x)


class MhdContinousSpectraCylinder(ContinuousSpectra1D):
    r"""
    Continuous shear Alfvén and slow sound spectra along radial direction in cylindrical geometry with radius :math:`a` and length :math:`2\pi\,R_0` in Struphy units.

    The profiles in cylindrical coordinates :math:`(r, \theta, z)` are

    .. math::

        \mathbf B_0 &= \mathbf B_0(r) = B_{0z}(r)\left( \mathbf e_z + \frac{r}{q(r) R_0}\mathbf e_\theta \right)\,,

        p_0 &= p_0(r)

        n_0 &= n_0(r)\,.

    The continuous spectra are then given by

    .. math::

        \textnormal{shear Alfvén}:\quad & \omega^2(r)=\frac{B_{0z}(r)^2}{n_0(r)}\frac{1}{R_0^2}\left(n+\frac{m}{q(r)}\right)^2\,

        \textnormal{slow sound}:\quad & \omega^2(r)=\frac{\gamma p_0(r)B_{0z}(r)^2}{n_0(r)\,[\gamma p_0(r) + B_{0\theta}(r)^2 + B_{0z}(r)^2]}\frac{1}{R_0^2}\left(n+\frac{m}{q(r)}\right)^2\,.

    Parameters
    ----------
    **params
        Keyword arguments that characterize the dispersion relation.
            * R0 : float
                "Major" radius (must be compatible with :math:`L_z=2\pi R_0`, default: 3.).
            * gamma : float
                Adiabatic index (default: 5/3).
            * Bz : callable
                Profile of axial magnetic field Bz=Bz(x) (default: 1. - 0*r).
            * p : callable
                Pressure profile p=p(x) (default: 0.5 - 0*r).
            * rho : callable
                Profile of mass density rho=rho(x) (default: 1. - 0*r).
            * q : callable
                Safety factor profile q=q(x) (default: 1.1 + 0.7*r**2).
    """

    def __init__(self, **params):
        # set default parameters
        params_default = {
            "R0": 3.0,
            "gamma": 5 / 3,
            "Bz": lambda r: 1.0 - 0 * r,
            "p": lambda r: 0.5 - 0 * r,
            "rho": lambda r: 1.0 - 0 * r,
            "q": lambda r: 1.1 + 0.7 * r**2,
        }

        params_all = set_defaults(params, params_default)

        super().__init__("shear_Alfvén", "slow_sound", **params_all)

    def __call__(self, r, m, n):
        """
        The evaluation of all continuous spectra.

        Parameters
        ----------
        r : array_like
            The radial points at which the continuous spectra shall be evaluated.

        m, n : int
            Mode numbers in theta- (m) and z- (n) direction.

        Returns
        -------
        specs : dict
            A dictionary with key=branch_name and value=omega(r) (ndarray).
        """

        # radial profiles
        Bz = self.params["Bz"]
        Bt = self._Bt
        p = self.params["p"]
        rho = self.params["rho"]
        F = self._F

        # other parameters
        gamma = self.params["gamma"]

        specs = {}

        # shear Alfvén continuum
        specs["shear_Alfvén"] = xp.sqrt(F(r, m, n) ** 2 / rho(r))

        # slow sound continuum
        specs["slow_sound"] = xp.sqrt(
            gamma * p(r) * F(r, m, n) ** 2 / (rho(r) * (gamma * p(r) + Bt(r) ** 2 + Bz(r) ** 2))
        )

        return specs

    # private methods:
    # ----------------

    def _Bt(self, r):
        """Poloidal magnetic field."""
        return r * self.params["Bz"](r) / (self.params["R0"] * self.params["q"](r))

    def _F(self, r, m, n):
        """Dot product of magnetic field with wavenumber k.B."""
        return m / r * self._Bt(r) + n / self.params["R0"] * self.params["Bz"](r)
