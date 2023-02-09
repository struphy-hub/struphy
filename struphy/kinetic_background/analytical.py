from abc import ABCMeta, abstractmethod
import numpy as np


class Maxwellian6D(metaclass=ABCMeta):
    r"""
    Base class for a 6d Maxwellian distribution function defined on [0, 1]^3 x R^3, with logical position and Cartesian velocity coordinates, defined by its velocity moments.

    .. math::

        f(\boldsymbol{\eta}, \mathbf v) = n(\boldsymbol{\eta})\frac{1}{\pi^{3/2}\,v_{\mathrm{th},x}(\boldsymbol{\eta})v_{\mathrm{th},y}(\boldsymbol{\eta})v_{\mathrm{th},z}(\boldsymbol{\eta})}\exp\left[-\frac{(v_x-u_x(\boldsymbol{\eta}))^2}{v_{\mathrm{th},x}(\boldsymbol{\eta})^2}-\frac{(v_y-u_y(\boldsymbol{\eta}))^2}{v_{\mathrm{th},y}(\boldsymbol{\eta})^2}-\frac{(v_z-u_z(\boldsymbol{\eta}))^2}{v_{\mathrm{th},z}(\boldsymbol{\eta})^2}\right]. 

    Parameters
    ----------
    **params
        Paramters defining the moments of the 6d Maxwellian.
    """

    def __init__(self, params):

        self._params = params

    @property
    def params(self):
        """ Parameters dictionary defining the moments of the 6d Maxwellian.
        """
        return self._params

    @abstractmethod
    def n(self, eta1, eta2, eta3):
        """ Number density (0-form).
        """

    @abstractmethod
    def ux(self, eta1, eta2, eta3):
        """ Mean velocity (Cartesian x-component, but dependent on logical coordinates).
        """

    @abstractmethod
    def uy(self, eta1, eta2, eta3):
        """ Mean velocity (Cartesian y-component, but dependent on logical coordinates).
        """

    @abstractmethod
    def uz(self, eta1, eta2, eta3):
        """ Mean velocity (Cartesian z-component, but dependent on logical coordinates).
        """

    @abstractmethod
    def vthx(self, eta1, eta2, eta3):
        """ Thermal velocity (Cartesian x-component, but dependent on logical coordinates).
        """

    @abstractmethod
    def vthy(self, eta1, eta2, eta3):
        """ Thermal velocity (Cartesian y-component, but dependent on logical coordinates).
        """

    @abstractmethod
    def vthz(self, eta1, eta2, eta3):
        """ Thermal velocity (Cartesian z-component, but dependent on logical coordinates).
        """

    def mx(self, eta1, eta2, eta3, vx):
        """ The Maxwellian integrated over vy-vz space.
        """
        return 1/(np.sqrt(np.pi)*self.vthx(eta1, eta2, eta3))*np.exp(-(vx - self.ux(eta1, eta2, eta3))**2/self.vthx(eta1, eta2, eta3)**2)

    def my(self, eta1, eta2, eta3, vy):
        """ The Maxwellian integrated over vx-vz space.
        """
        return 1/(np.sqrt(np.pi)*self.vthy(eta1, eta2, eta3))*np.exp(-(vy - self.uy(eta1, eta2, eta3))**2/self.vthy(eta1, eta2, eta3)**2)

    def mz(self, eta1, eta2, eta3, vz):
        """ The Maxwellian integrated over vx-vy space.
        """
        return 1/(np.sqrt(np.pi)*self.vthz(eta1, eta2, eta3))*np.exp(-(vz - self.uz(eta1, eta2, eta3))**2/self.vthz(eta1, eta2, eta3)**2)

    def __call__(self, eta1, eta2, eta3, vx, vy, vz):
        """
        Evaluates the 6d Maxwellian.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical coordinates.

        vx, vy, vz : array_like
            Velocity coordinates

        Returns
        -------
        f : np.ndarray
            The evaluated Maxwellian.
        """

        f = self.n(eta1, eta2, eta3)

        f *= self.mx(eta1, eta2, eta3, vx)
        f *= self.my(eta1, eta2, eta3, vy)
        f *= self.mz(eta1, eta2, eta3, vz)

        return f


class Maxwellian6DUniform(Maxwellian6D):
    r"""
    6d Maxwellian distribution function defined on [0, 1]^3 x R^3, with logical position and Cartesian velocity coordinates, with uniform velocity moments.

    .. math::

        f(\boldsymbol{\eta}, \mathbf v) = n\,\frac{1}{\pi^{3/2}\,v_{\mathrm{th},x}\,v_{\mathrm{th},y}\,v_{\mathrm{th},z}}\,\exp\left[-\frac{(v_x-u_x)^2}{v_{\mathrm{th},x}^2}-\frac{(v_y-u_y)^2}{v_{\mathrm{th},y}^2}-\frac{(v_z-u_z)^2}{v_{\mathrm{th},z}^2}\right].

    Parameters
    ----------
    **params
        Keyword arguments (n= , ux=, etc.) defining the moments of the 6d Maxwellian.
    """

    def __init__(self, **params):

        # set default parameters if a key is missing
        keys = ['n', 'ux', 'uy', 'uz', 'vthx', 'vthy', 'vthz']

        params[keys[0]] = 1. if keys[0] not in params.keys() else params[keys[0]]

        params[keys[1]] = 0. if keys[1] not in params.keys() else params[keys[1]]
        params[keys[2]] = 0. if keys[2] not in params.keys() else params[keys[2]]
        params[keys[3]] = 0. if keys[3] not in params.keys() else params[keys[3]]

        params[keys[4]] = 1. if keys[4] not in params.keys() else params[keys[4]]
        params[keys[5]] = 1. if keys[5] not in params.keys() else params[keys[5]]
        params[keys[6]] = 1. if keys[6] not in params.keys() else params[keys[6]]

        super().__init__(params)

    def n(self, eta1, eta2, eta3):
        """ Number density (0-form).
        """
        return self.params['n'] - 0*eta1

    def ux(self, eta1, eta2, eta3):
        """ Mean velocity (Cartesian x-component, but dependent on logical coordinates).
        """
        return self.params['ux'] - 0*eta1

    def uy(self, eta1, eta2, eta3):
        """ Mean velocity (Cartesian y-component, but dependent on logical coordinates).
        """
        return self.params['uy'] - 0*eta1

    def uz(self, eta1, eta2, eta3):
        """ Mean velocity (Cartesian z-component, but dependent on logical coordinates).
        """
        return self.params['uz'] - 0*eta1

    def vthx(self, eta1, eta2, eta3):
        """ Thermal velocity (Cartesian x-component, but dependent on logical coordinates).
        """
        return self.params['vthx'] - 0*eta1

    def vthy(self, eta1, eta2, eta3):
        """ Thermal velocity (Cartesian y-component, but dependent on logical coordinates).
        """
        return self.params['vthy'] - 0*eta1

    def vthz(self, eta1, eta2, eta3):
        """ Thermal velocity (Cartesian z-component, but dependent on logical coordinates).
        """
        return self.params['vthz'] - 0*eta1


class Maxwellian6DPerturbed(Maxwellian6D):
    r"""
    6d Maxwellian distribution function defined on [0, 1]^3 x R^3, with logical position and Cartesian velocity coordinates, with sin/cos perturbed velocity moments.

    .. math::

        f(\boldsymbol{\eta}, \mathbf v) = n(\boldsymbol{\eta})&\frac{1}{\pi^{3/2}\,v_{\mathrm{th},x}(\boldsymbol{\eta})\,v_{\mathrm{th},y}(\boldsymbol{\eta})\,v_{\mathrm{th},z}(\boldsymbol{\eta})}\,\exp\left[-\frac{(v_x-u_x(\boldsymbol{\eta}))^2}{v_{\mathrm{th},x}(\boldsymbol{\eta})^2}-\frac{(v_y-u_y(\boldsymbol{\eta}))^2}{v_{\mathrm{th},y}(\boldsymbol{\eta})^2}-\frac{(v_z-u_z(\boldsymbol{\eta}))^2}{v_{\mathrm{th},z}(\boldsymbol{\eta})^2}\right]\,,

        n(\boldsymbol{\eta})&= n_0 + \sum_i\left\lbrace A_i\sin\left[2\pi(l_i\,\eta_1+m_i\,\eta_2+n_i\,\eta_3)\right] + B_i\cos\left[2\pi(l_i\,\eta_1+m_i\,\eta_2+n_i\,\eta_3)\right] \right\rbrace\,,

    and similarly for the other moments :math:`u_x(\boldsymbol{\eta}),u_y(\boldsymbol{\eta})`, etc..

    Parameters
    ----------
    **params
        Keyword arguments defining the moments of the 6d Maxwellian. For each moment, a dictionary of the form {'n0' : float, 'perturbation' : {'l' : list, 'm' : list, 'n' : list, 'amps_sin' : list, 'amps_cos' : list}} must be passed.
    """

    def __init__(self, **params):

        moment_keys = ['n', 'ux', 'uy', 'uz', 'vthx', 'vthy', 'vthz']

        backgr_keys = ['n0', 'ux0', 'uy0', 'uz0', 'vthx0', 'vthy0', 'vthz0']

        # set default background, mode numbers and amplitudes if no perturbation of a moment in given
        for moment_key, backgr_key in zip(moment_keys, backgr_keys):

            # add moment key if not there
            if moment_key not in params.keys():
                params[moment_key] = {}

            if not backgr_key in params[moment_key].keys():

                if len(backgr_key) == 2:
                    params[moment_key][backgr_key] = 1.
                elif len(backgr_key) == 3:
                    params[moment_key][backgr_key] = 0.
                else:
                    params[moment_key][backgr_key] = 1.

            if not 'perturbation' in params[moment_key].keys():
                params[moment_key]['perturbation'] = {}

                params[moment_key]['perturbation']['l'] = [0]
                params[moment_key]['perturbation']['m'] = [0]
                params[moment_key]['perturbation']['n'] = [0]

                params[moment_key]['perturbation']['amps_sin'] = [0]
                params[moment_key]['perturbation']['amps_cos'] = [0]

        super().__init__(params)

    def modes_sin(self, eta1, eta2, eta3, l, m, n, amps):
        """
        Superposition of sine modes.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Coordinates at which to evaluate.

        l, m, n : array_like
            List of modes numbers in certain spatial direction. Must be of equal length.

        amps : array_like
            List of modes amplitudes. Must have same length as lists of mode numbers.

        Returns
        -------
        value : ndarray
            Superposition of sine modes evaluated at given coordinates.
        """

        value = 0.
        for i in range(len(amps)):
            value += amps[i]*np.sin(2*np.pi*l[i]*eta1 +
                                    2*np.pi*m[i]*eta2 +
                                    2*np.pi*n[i]*eta3)

        return value

    def modes_cos(self, eta1, eta2, eta3, l, m, n, amps):
        """
        Superposition of cosine modes.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Coordinates at which to evaluate.

        l, m, n : array_like
            List of modes numbers in certain spatial direction. Must be of equal length.

        amps : array_like
            List of modes amplitudes. Must have same length as lists of mode numbers.

        Returns
        -------
        value : ndarray
            Superposition of cosine modes evaluated at given coordinates.
        """

        value = 0.
        for i in range(len(amps)):
            value += amps[i]*np.cos(2*np.pi*l[i]*eta1 +
                                    2*np.pi*m[i]*eta2 +
                                    2*np.pi*n[i]*eta3)

        return value

    def n(self, eta1, eta2, eta3):
        """ Number density (0-form).
        """

        ls = self.params['n']['perturbation']['l']
        ms = self.params['n']['perturbation']['m']
        ns = self.params['n']['perturbation']['n']

        amps_sin = self.params['n']['perturbation']['amps_sin']
        amps_cos = self.params['n']['perturbation']['amps_cos']

        res = self.params['n']['n0']
        res += self.modes_sin(eta1, eta2, eta3, ls, ms, ns, amps_sin)
        res += self.modes_cos(eta1, eta2, eta3, ls, ms, ns, amps_cos)

        return res

    def ux(self, eta1, eta2, eta3):
        """ Mean velocity (Cartesian x-component, but dependent on logical coordinates).
        """

        ls = self.params['ux']['perturbation']['l']
        ms = self.params['ux']['perturbation']['m']
        ns = self.params['ux']['perturbation']['n']

        amps_sin = self.params['ux']['perturbation']['amps_sin']
        amps_cos = self.params['ux']['perturbation']['amps_cos']

        res = self.params['ux']['ux0']
        res += self.modes_sin(eta1, eta2, eta3, ls, ms, ns, amps_sin)
        res += self.modes_cos(eta1, eta2, eta3, ls, ms, ns, amps_cos)

        return res

    def uy(self, eta1, eta2, eta3):
        """ Mean velocity (Cartesian y-component, but dependent on logical coordinates).
        """

        ls = self.params['uy']['perturbation']['l']
        ms = self.params['uy']['perturbation']['m']
        ns = self.params['uy']['perturbation']['n']

        amps_sin = self.params['uy']['perturbation']['amps_sin']
        amps_cos = self.params['uy']['perturbation']['amps_cos']

        res = self.params['uy']['uy0']
        res += self.modes_sin(eta1, eta2, eta3, ls, ms, ns, amps_sin)
        res += self.modes_cos(eta1, eta2, eta3, ls, ms, ns, amps_cos)

        return res

    def uz(self, eta1, eta2, eta3):
        """ Mean velocity (Cartesian z-component, but dependent on logical coordinates).
        """

        ls = self.params['uz']['perturbation']['l']
        ms = self.params['uz']['perturbation']['m']
        ns = self.params['uz']['perturbation']['n']

        amps_sin = self.params['uz']['perturbation']['amps_sin']
        amps_cos = self.params['uz']['perturbation']['amps_cos']

        res = self.params['uz']['uz0']
        res += self.modes_sin(eta1, eta2, eta3, ls, ms, ns, amps_sin)
        res += self.modes_cos(eta1, eta2, eta3, ls, ms, ns, amps_cos)

        return res

    def vthx(self, eta1, eta2, eta3):
        """ Thermal velocity (Cartesian x-component, but dependent on logical coordinates).
        """

        ls = self.params['vthx']['perturbation']['l']
        ms = self.params['vthx']['perturbation']['m']
        ns = self.params['vthx']['perturbation']['n']

        amps_sin = self.params['vthx']['perturbation']['amps_sin']
        amps_cos = self.params['vthx']['perturbation']['amps_cos']

        res = self.params['vthx']['vthx0']
        res += self.modes_sin(eta1, eta2, eta3, ls, ms, ns, amps_sin)
        res += self.modes_cos(eta1, eta2, eta3, ls, ms, ns, amps_cos)

        return res

    def vthy(self, eta1, eta2, eta3):
        """ Thermal velocity (Cartesian y-component, but dependent on logical coordinates).
        """

        ls = self.params['vthy']['perturbation']['l']
        ms = self.params['vthy']['perturbation']['m']
        ns = self.params['vthy']['perturbation']['n']

        amps_sin = self.params['vthy']['perturbation']['amps_sin']
        amps_cos = self.params['vthy']['perturbation']['amps_cos']

        res = self.params['vthy']['vthy0']
        res += self.modes_sin(eta1, eta2, eta3, ls, ms, ns, amps_sin)
        res += self.modes_cos(eta1, eta2, eta3, ls, ms, ns, amps_cos)

        return res

    def vthz(self, eta1, eta2, eta3):
        """ Thermal velocity (Cartesian z-component, but dependent on logical coordinates).
        """

        ls = self.params['vthz']['perturbation']['l']
        ms = self.params['vthz']['perturbation']['m']
        ns = self.params['vthz']['perturbation']['n']

        amps_sin = self.params['vthz']['perturbation']['amps_sin']
        amps_cos = self.params['vthz']['perturbation']['amps_cos']

        res = self.params['vthz']['vthz0']
        res += self.modes_sin(eta1, eta2, eta3, ls, ms, ns, amps_sin)
        res += self.modes_cos(eta1, eta2, eta3, ls, ms, ns, amps_cos)

        return res


class Maxwellian6DITPA(Maxwellian6D):
    r"""
    6d Maxwellian distribution function defined on [0, 1]^3 x R^3, with logical position and Cartesian velocity coordinates, with isotropic, shifted distribution in velocity space and 1d density variation in first direction.

    .. math::

        f(\boldsymbol{\eta}, \mathbf v) = n(\eta_1)&\,\frac{1}{\pi^{3/2}\,v_{\mathrm{th}}^3}\,\exp\left[-\frac{(v_x-u_x)^2+(v_y-u_y)^2+(v_z-u_z)^2}{v_{\mathrm{th}}^2}\right]\,,

        n(\eta_1)& = c_3\exp\left[-\frac{c_2}{c_1}\tanh\left(\frac{\eta_1 - c_0}{c_2}\right)\right]\,.

    Parameters
    ----------
    **params
        Keyword arguments defining the moments of the 6d Maxwellian. For the density profile a dictionary of the form {'c0' : float, 'c1' : float, 'c2' : float, 'c3' : float} must be passed.
    """

    def __init__(self, **params):

        # set default ITPA default parameters if not given
        if 'n' not in params.keys():
            params['n'] = {}

            params['n']['c0'] = 0.491230
            params['n']['c1'] = 0.298228
            params['n']['c2'] = 0.198739
            params['n']['c3'] = 0.521298

        if 'vth' not in params.keys():
            params['vth'] = 1.

        self._params = params

    def n(self, eta1, eta2, eta3):
        """ Number density (0-form).
        """

        c0 = self.params['n']['c0']
        c1 = self.params['n']['c1']
        c2 = self.params['n']['c2']
        c3 = self.params['n']['c3']

        if c2 == 0.:
            res = c3 - 0*eta1
        else:
            res = c3*np.exp(-c2/c1*np.tanh((eta1 - c0)/c2))

        return res

    def ux(self, eta1, eta2, eta3):
        """ Mean velocity (Cartesian x-component, but dependent on logical coordinates).
        """
        return 0*eta1

    def uy(self, eta1, eta2, eta3):
        """ Mean velocity (Cartesian y-component, but dependent on logical coordinates).
        """
        return 0*eta1

    def uz(self, eta1, eta2, eta3):
        """ Mean velocity (Cartesian z-component, but dependent on logical coordinates).
        """
        return 0*eta1

    def vthx(self, eta1, eta2, eta3):
        """ Thermal velocity (Cartesian x-component, but dependent on logical coordinates).
        """
        return self.params['vth'] - 0*eta1

    def vthy(self, eta1, eta2, eta3):
        """ Thermal velocity (Cartesian y-component, but dependent on logical coordinates).
        """
        return self.params['vth'] - 0*eta1

    def vthz(self, eta1, eta2, eta3):
        """ Thermal velocity (Cartesian z-component, but dependent on logical coordinates).
        """
        return self.params['vth'] - 0*eta1
