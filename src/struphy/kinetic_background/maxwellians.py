'Maxwellian (Gaussian) distributions in velocity space.'


import numpy as np

from struphy.kinetic_background.base import Maxwellian
from struphy.fields_background.mhd_equil.equils import set_defaults


class Maxwellian6DUniform(Maxwellian):
    r"""
    6d Maxwellian distribution function defined on :math:`[0, 1]^3 \times \mathbb R^3`, 
    with logical position and Cartesian velocity coordinates and uniform velocity moments.

    .. math::

        f(\mathbf v) = \frac{n}{(2\pi)^{3/2} \, v_{\mathrm{th},1} \, v_{\mathrm{th},2} \, v_{\mathrm{th},3}} \,
            \exp\left[- \frac{(v_1 - u_1)^2}{2v_{\mathrm{th},1}^2}
                      - \frac{(v_2 - u_2)^2}{2v_{\mathrm{th},2}^2}
                      - \frac{(v_3 - u_3)^2}{2v_{\mathrm{th},3}^2}\right].

    Parameters
    ----------
    **params
        Keyword arguments (n= , u1=, etc.) defining the moments of the 6d Maxwellian.

    Note
    ----
    In the parameter .yml, use the following in the section ``kinetic/<species>``::

        init :
            type : Maxwellian6DUniform
            Maxwellian6DUniform :
                n : 1.0
                u1 : 0.0
                u2 : 0.0
                u3 : 0.0
                vth1 : 1.0
                vth2 : 1.0
                vth3 : 1.0

    Can use ``background :`` instead of ``init :``.
    """

    def __init__(self, **params):

        # default parameters
        params_default = {'n': 1.,
                          'u1': 0.,
                          'u2': 0.,
                          'u3': 0.,
                          'vth1': 1.,
                          'vth2': 1.,
                          'vth3': 1.}

        self._params = set_defaults(params, params_default)

    @property
    def params(self):
        """ Parameters dictionary defining the moments of the Maxwellian.
        """
        return self._params

    @property
    def vdim(self):
        """Dimension of the velocity space.
        """
        return 3
    
    @property
    def is_polar(self):
        """List of booleans. True if the velocity coordinates are polar coordinates.
        """
        return [False, False, False]

    def n(self, eta1, eta2, eta3):
        """ Number density (0-form). 

        Parameters
        ----------
        eta1, eta2, eta3 : numpy.array
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the density evaluated at evaluation points (same shape as etas).
        """
        return self.params['n'] - 0*eta1

    def vth(self, eta1, eta2, eta3):
        """ Thermal velocities (0-forms).

        Parameters
        ----------
        etas : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the thermal velocity evaluated at evaluation points (one dimension more than etas).
        The additional dimension is in the first index.
        """
        res_list = []

        res_list += [self.params['vth1'] - 0*eta1]
        res_list += [self.params['vth2'] - 0*eta1]
        res_list += [self.params['vth3'] - 0*eta1]

        return np.array(res_list)

    def u(self, eta1, eta2, eta3):
        """ Mean velocities (Cartesian components evaluated at x = F(eta)).

        Parameters
        ----------
        eta1, eta2, eta3  : numpy.array
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the mean velocity evaluated at evaluation points (one dimension more than etas).
        The additional dimension is in the first index.
        """
        res_list = []

        res_list += [self.params['u1'] - 0*eta1]
        res_list += [self.params['u2'] - 0*eta1]
        res_list += [self.params['u3'] - 0*eta1]

        return np.array(res_list)


class Maxwellian6DPerturbed(Maxwellian):
    r"""
    6d Maxwellian distribution function defined on :math:`[0, 1]^3 \times \mathbb R^3`, 
    with logical position and Cartesian velocity coordinates, with sin/cos perturbed velocity moments.

    .. math::

        f(\boldsymbol{\eta}, \mathbf v) = \frac{n(\boldsymbol{\eta})}{(2\pi)^{3/2}(v_{\mathrm{th},x}\,v_{\mathrm{th},y}\,v_{\mathrm{th},z})(\boldsymbol{\eta})}\,\exp\left[-\frac{(v_x-u_x(\boldsymbol{\eta}))^2}{2v_{\mathrm{th},x}(\boldsymbol{\eta})^2}-\frac{(v_y-u_y(\boldsymbol{\eta}))^2}{2v_{\mathrm{th},y}(\boldsymbol{\eta})^2}-\frac{(v_z-u_z(\boldsymbol{\eta}))^2}{2v_{\mathrm{th},z}(\boldsymbol{\eta})^2}\right]\,, 

    with perturbations of the form

    .. math::

        n(\boldsymbol{\eta})= n_0 + \sum_i\left\lbrace A_i\sin\left[2\pi(l_i\,\eta_1+m_i\,\eta_2+n_i\,\eta_3)\right] + B_i\cos\left[2\pi(l_i\,\eta_1+m_i\,\eta_2+n_i\,\eta_3)\right] \right\rbrace\,,

    and similarly for the other moments :math:`u_x(\boldsymbol{\eta}),u_y(\boldsymbol{\eta})`, etc.

    Parameters
    ----------
    **params
        Keyword arguments defining the moments of the 6d Maxwellian. For each moment, a dictionary of the form {'n0' : float, 'perturbation' : {'l' : list, 'm' : list, 'n' : list, 'amps_sin' : list, 'amps_cos' : list}} must be passed.

    Note
    ----
    In the parameter .yml, use the following in the section ``kinetic/<species>``::

        init :
            type : Maxwellian6DPerturbed
            Maxwellian6DPerturbed :
                n :
                    n0 : 1.
                    perturbation :
                        l : [0]
                        m : [0]
                        n : [0]
                        amps_sin : [0.]
                        amps_cos : [0.]
                u1 :
                    u10 : 0.
                u2 :
                    u20 : 0.
                u3 :
                    u30 : 0.
                vth1 :
                    vth10 : 1.
                vth2 :
                    vth20 : 1.
                vth3 :
                    vth30 : 1.

    Can use ``background :`` instead of ``init :``.
    """

    def __init__(self, **params):

        moment_keys = ['n', 'u1', 'u2', 'u3', 'vth1', 'vth2', 'vth3']

        backgr_keys = ['n0', 'u01', 'u02', 'u03', 'vth01', 'vth02', 'vth03']

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

        self._params = params

    @property
    def params(self):
        """Parameters dictionary defining the moments of the Maxwellian.
        """
        return self._params

    @property
    def vdim(self):
        """Dimension of the velocity space.
        """
        return 3

    @property
    def is_polar(self):
        """List of booleans. True if the velocity coordinates are polar coordinates.
        """
        return [False, False, False]

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
        """ Mean velocities (Cartesian components evaluated at x = F(eta)).

        Parameters
        ----------
        eta1, eta2, eta3 : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the mean velocity evaluated at evaluation points (one dimension more than etas).
        The additional dimension is in the first index.
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

    def u(self, eta1, eta2, eta3):
        """ Mean velocities (Cartesian components evaluated at x = F(eta)).

        Parameters
        ----------
        eta1, eta2, eta3  : numpy.array
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the mean velocity evaluated at evaluation points (one dimension more than etas).
        The additional dimension is in the first index.
        """

        res_list = []

        ls = self.params['u1']['perturbation']['l']
        ms = self.params['u1']['perturbation']['m']
        ns = self.params['u1']['perturbation']['n']

        amps_sin = self.params['u1']['perturbation']['amps_sin']
        amps_cos = self.params['u1']['perturbation']['amps_cos']

        res = self.params['u1']['u01']
        res += self.modes_sin(eta1, eta2, eta3, ls, ms, ns, amps_sin)
        res += self.modes_cos(eta1, eta2, eta3, ls, ms, ns, amps_cos)

        res_list += [res]

        ls = self.params['u2']['perturbation']['l']
        ms = self.params['u2']['perturbation']['m']
        ns = self.params['u2']['perturbation']['n']

        amps_sin = self.params['u2']['perturbation']['amps_sin']
        amps_cos = self.params['u2']['perturbation']['amps_cos']

        res = self.params['u2']['u02']
        res += self.modes_sin(eta1, eta2, eta3, ls, ms, ns, amps_sin)
        res += self.modes_cos(eta1, eta2, eta3, ls, ms, ns, amps_cos)

        res_list += [res]

        ls = self.params['u3']['perturbation']['l']
        ms = self.params['u3']['perturbation']['m']
        ns = self.params['u3']['perturbation']['n']

        amps_sin = self.params['u3']['perturbation']['amps_sin']
        amps_cos = self.params['u3']['perturbation']['amps_cos']

        res = self.params['u3']['u03']
        res += self.modes_sin(eta1, eta2, eta3, ls, ms, ns, amps_sin)
        res += self.modes_cos(eta1, eta2, eta3, ls, ms, ns, amps_cos)

        res_list += [res]

        return np.array(res_list)

    def vth(self, eta1, eta2, eta3):
        """ Thermal velocities (0-forms).

        Parameters
        ----------
        etas : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the thermal velocity evaluated at evaluation points (one dimension more than etas).
        The additional dimension is in the first index.
        """

        res_list = []

        ls = self.params['vth1']['perturbation']['l']
        ms = self.params['vth1']['perturbation']['m']
        ns = self.params['vth1']['perturbation']['n']

        amps_sin = self.params['vth1']['perturbation']['amps_sin']
        amps_cos = self.params['vth1']['perturbation']['amps_cos']

        res = self.params['vth1']['vth01']
        res += self.modes_sin(eta1, eta2, eta3, ls, ms, ns, amps_sin)
        res += self.modes_cos(eta1, eta2, eta3, ls, ms, ns, amps_cos)

        res_list += [res]

        ls = self.params['vth2']['perturbation']['l']
        ms = self.params['vth2']['perturbation']['m']
        ns = self.params['vth2']['perturbation']['n']

        amps_sin = self.params['vth2']['perturbation']['amps_sin']
        amps_cos = self.params['vth2']['perturbation']['amps_cos']

        res = self.params['vth2']['vth02']
        res += self.modes_sin(eta1, eta2, eta3, ls, ms, ns, amps_sin)
        res += self.modes_cos(eta1, eta2, eta3, ls, ms, ns, amps_cos)

        res_list += [res]

        ls = self.params['vth3']['perturbation']['l']
        ms = self.params['vth3']['perturbation']['m']
        ns = self.params['vth3']['perturbation']['n']

        amps_sin = self.params['vth3']['perturbation']['amps_sin']
        amps_cos = self.params['vth3']['perturbation']['amps_cos']

        res = self.params['vth3']['vth03']
        res += self.modes_sin(eta1, eta2, eta3, ls, ms, ns, amps_sin)
        res += self.modes_cos(eta1, eta2, eta3, ls, ms, ns, amps_cos)

        res_list += [res]

        return np.array(res_list)


class Maxwellian6DITPA(Maxwellian):
    r"""
    6d Maxwellian distribution function defined on :math:`[0, 1]^3 \times \mathbb R^3`, 
    with logical position and Cartesian velocity coordinates, with isotropic, shifted distribution in velocity space and 1d density variation in first direction.

    .. math::

        f(\eta_1, \mathbf v) = \,\frac{n(\eta_1)}{(2\pi)^{3/2}\,v_{\mathrm{th}}^3}\,\exp\left[-\frac{(v_x-u_x)^2+(v_y-u_y)^2+(v_z-u_z)^2}{2v_{\mathrm{th}}^2}\right]\,,

    with the density profile

    .. math::

        n(\eta_1) = c_3\exp\left[-\frac{c_2}{c_1}\tanh\left(\frac{\eta_1 - c_0}{c_2}\right)\right]\,.

    Parameters
    ----------
    **params
        Keyword arguments defining the moments of the 6d Maxwellian. For the density profile a dictionary of the form {'c0' : float, 'c1' : float, 'c2' : float, 'c3' : float} must be passed.

    Note
    ----
    In the parameter .yml, use the following in the section ``kinetic/<species>``::

        init :
            type : Maxwellian6DITPA
            Maxwellian6DITPA :
                n : 
                    c0: 0.5
                    c1: 0.5
                    c2: 0.5
                    c3: 0.5
                vth : 1.0

    Can use ``background :`` instead of ``init :``.
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

    @property
    def params(self):
        """Parameters dictionary defining the moments of the Maxwellian.
        """
        return self._params

    @property
    def vdim(self):
        """Dimension of the velocity space.
        """
        return 3

    @property
    def is_polar(self):
        """List of booleans. True if the velocity coordinates are polar coordinates.
        """
        return [False, False, False]

    def n(self, eta1, eta2, eta3):
        """ Number density (0-form). 

        Parameters
        ----------
        eta1, eta2, eta3 : numpy.array
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the density evaluated at evaluation points (same shape as etas).
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

    def vth(self, eta1, eta2, eta3):
        """ Thermal velocities (0-forms).

        Parameters
        ----------
        etas : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the thermal velocity evaluated at evaluation points (one dimension more than etas).
        The additional dimension is in the first index.
        """

        res_list = []

        res_list += [self.params['vth1'] - 0*eta1]
        res_list += [self.params['vth2'] - 0*eta1]
        res_list += [self.params['vth3'] - 0*eta1]

        return np.array(res_list)

    def u(self, eta1, eta2, eta3):
        """ Mean velocities (Cartesian components evaluated at x = F(eta)).

        Parameters
        ----------
        eta1, eta2, eta3  : numpy.array
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the mean velocity evaluated at evaluation points (one dimension more than etas).
        The additional dimension is in the first index.
        """
        res_list = []

        res_list += [0*eta1]
        res_list += [0*eta1]
        res_list += [0*eta1]

        return np.array(res_list)


class Maxwellian5DUniform(Maxwellian):
    r"""
    5d Maxwellian distribution function defined on :math:`[0, 1]^3 \times \mathbb R^2`, 
    with logical position and Cartesian velocity coordinates, with uniform velocity moments.

    .. math::

        f(v_\parallel, v_\perp) = \frac{n}{2\pi\,v_{\mathrm{th},\parallel}\,v_{\mathrm{th},\perp}}\exp\left[-\frac{(v_\parallel-u_\parallel)^2}{2v_{\mathrm{th},\parallel}^2} - \frac{(v_\perp-u_\perp)^2}{2v_{\mathrm{th},\perp}^2}\right]\,.

    Parameters
    ----------
    **params
        Keyword arguments (n= , u_parallel=, etc.) defining the moments of the 6d Maxwellian.

    Note
    ----
    In the parameter .yml, use the following in the section ``kinetic/<species>``::

        init :
            type : Maxwellian5DUniform
            Maxwellian5DUniform :
                n : 1.0
                u_parallel : 0.0
                u_perp : 0.0
                vth_parallel : 1.0
                vth_perp : 1.0

    Can use ``background :`` instead of ``init :``.
    """

    def __init__(self, **params):

        # default parameters
        params_default = {'n': 1.,
                          'u_parallel': 0.,
                          'u_perp': 0.,
                          'vth_parallel': 1.,
                          'vth_perp': 1.}

        self._params = set_defaults(params, params_default)

    @property
    def params(self):
        """Parameters dictionary defining the moments of the Maxwellian.
        """
        return self._params

    @property
    def vdim(self):
        """Dimension of the velocity space.
        """
        return 2
    
    @property
    def is_polar(self):
        """List of booleans. True if the velocity coordinates are polar coordinates.
        """
        return [False, True]

    def n(self, eta1, eta2, eta3):
        """ Number density (0-form). 

        Parameters
        ----------
        eta1, eta2, eta3 : numpy.array
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the density evaluated at evaluation points (same shape as etas).
        """
        return self.params['n'] - 0*eta1

    def vth(self, eta1, eta2, eta3):
        """ Thermal velocities (0-forms).

        Parameters
        ----------
        etas : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the thermal velocity evaluated at evaluation points (one dimension more than etas).
        The additional dimension is in the first index.
        """
        res_list = []

        res_list += [self.params['vth_parallel'] - 0*eta1]
        res_list += [self.params['vth_perp'] - 0*eta1]

        return np.array(res_list)

    def u(self, eta1, eta2, eta3):
        """ Mean velocities (Cartesian components evaluated at x = F(eta)).

        Parameters
        ----------
        eta1, eta2, eta3  : numpy.array
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the mean velocity evaluated at evaluation points (one dimension more than etas).
        The additional dimension is in the first index.
        """
        res_list = []

        res_list += [self.params['u_parallel'] - 0*eta1]
        res_list += [self.params['u_perp'] - 0*eta1]

        return np.array(res_list)


class Maxwellian5DITPA(Maxwellian):
    r"""
    5d Maxwellian distribution function defined on :math:`[0, 1]^3 \times \mathbb R^3`, 
    with logical position and Cartesian velocity coordinates, with isotropic, shifted distribution in velocity space and 1d density variation in first direction.

    .. math::

        f(\eta_1, v_\parallel) &= \,\frac{n(\eta_1)}{\sqrt{2\pi}\,v_\mathrm{th}}\,\exp\left[-\frac{(v_\parallel-u_\parallel)^2}{2v_{\mathrm{th}}^2}\right]\,,
        \\
        f(\eta_1, v_\perp) &= \,\frac{n(\eta_1)}{v^2_\mathrm{th}} v_\perp \,\exp\left[-\frac{(v_\perp-u_\perp)^2}{2v_{\mathrm{th}}^2}\right]\,,

    with the density profile

    .. math::

        n(\eta_1) = c_3\exp\left[-\frac{c_2}{c_1}\tanh\left(\frac{\eta_1 - c_0}{c_2}\right)\right]\,.

    Parameters
    ----------
    **params
        Keyword arguments defining the moments of the 6d Maxwellian. For the density profile a dictionary of the form {'c0' : float, 'c1' : float, 'c2' : float, 'c3' : float} must be passed.

    Note
    ----
    In the parameter .yml, use the following in the section ``kinetic/<species>``::

        init :
            type : Maxwellian5DITPA
            Maxwellian5DITPA :
                n : 
                    n0: 0.00720655
                    c0: 0.49123
                    c1: 0.298228
                    c2: 0.198739
                    c3: 0.521298
                vth : 1.0

    Can use ``background :`` instead of ``init :``.
    """

    def __init__(self, **params):

        # set default ITPA default parameters if not given
        if 'n' not in params.keys():
            params['n'] = {}
            params['n']['n0'] = 0.00720655
            params['n']['c0'] = 0.491230
            params['n']['c1'] = 0.298228
            params['n']['c2'] = 0.198739
            params['n']['c3'] = 0.521298

        if 'vth' not in params.keys():
            params['vth'] = 1.

        self._params = params

    @property
    def params(self):
        """Parameters dictionary defining the moments of the Maxwellian.
        """
        return self._params

    @property
    def vdim(self):
        """Dimension of the velocity space.
        """
        return 2

    @property
    def is_polar(self):
        """List of booleans. True if the velocity coordinates are polar coordinates.
        """
        return [False, True]

    def n(self, eta1, eta2, eta3):
        """ Number density (0-form). 

        Parameters
        ----------
        eta1, eta2, eta3 : numpy.array
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the density evaluated at evaluation points (same shape as etas).
        """

        n0 = self.params['n']['n0']
        c0 = self.params['n']['c0']
        c1 = self.params['n']['c1']
        c2 = self.params['n']['c2']
        c3 = self.params['n']['c3']

        if c2 == 0.:
            res = n0*c3 - 0*eta1
        else:
            res = n0*c3*np.exp(-c2/c1*np.tanh((eta1 - c0)/c2))

        return res

    def vth(self, eta1, eta2, eta3):
        """ Thermal velocities (0-forms).

        Parameters
        ----------
        etas : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the thermal velocity evaluated at evaluation points (one dimension more than etas).
        The additional dimension is in the first index.
        """

        res_list = []

        res_list += [self.params['vth'] - 0*eta1]
        res_list += [self.params['vth'] - 0*eta1]

        return np.array(res_list)

    def u(self, eta1, eta2, eta3):
        """ Mean velocities (Cartesian components evaluated at x = F(eta)).

        Parameters
        ----------
        eta1, eta2, eta3  : numpy.array
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the mean velocity evaluated at evaluation points (one dimension more than etas).
        The additional dimension is in the first index.
        """
        res_list = []

        res_list += [0*eta1]
        res_list += [0*eta1]

        return np.array(res_list)