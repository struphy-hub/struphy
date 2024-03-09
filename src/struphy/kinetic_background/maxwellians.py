'Maxwellian (Gaussian) distributions in velocity space.'


import numpy as np

from struphy.kinetic_background.base import Maxwellian
from struphy.fields_background.mhd_equil.equils import set_defaults
from struphy.initial import perturbations


class Maxwellian6D(Maxwellian):
    r""" A :class:`~struphy.kinetic_background.base.Maxwellian` with velocity dimension :math:`n=3`.

    Parameters
    ----------
    **params : 
        Consist of background and perturbation parameters.
    """

    @classmethod
    def default_bckgr_params(cls):
        """ Default parameters dictionary defining constant moments of the Maxwellian.
        """
        return {
            'n': 1.,
            'u1': 0.,
            'u2': 0.,
            'u3': 0.,
            'vth1': 1.,
            'vth2': 1.,
            'vth3': 1.,
        }

    def __init__(self, **params):

        # Set background parameters as the default, then overwrite below if given
        self._bckgr_params = self.default_bckgr_params()

        if 'background' in params:
            assert params['background']['type'] == 'Maxwellian6D', \
                f"Background must be 'Maxwellian6D' but is {params['background']['type']}"

            if 'Maxwellian6D' in params['background'].keys():
                bckgr_params = params['background']['Maxwellian6D']
                self._bckgr_params = set_defaults(
                    bckgr_params, self.default_bckgr_params()
                )

        # Set parameters for perturbation
        self._pert_params = params['perturbation'] if 'perturbation' in params \
            else {'type': None}

    @property
    def bckgr_params(self):
        """ Parameters dictionary defining constant moments of the Maxwellian.
        """
        return self._bckgr_params

    @property
    def pert_params(self):
        """ Parameters dictionary defining the perturbations of the :meth:`~Maxwellian6D.bckgr_params`.
        """
        return self._pert_params

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
        """ Density as background + perturbation.

        Parameters
        ----------
        eta1, eta2, eta3 : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        Without perturbation a float (background value); with perturbation a numpy.array of the evaluated density.
        """

        assert np.shape(eta1) == np.shape(eta2) == np.shape(eta3)

        res = self.bckgr_params['n']

        # Add perturbation if parameters are given and if density is to be perturbed
        pert_type = self.pert_params['type']
        if pert_type is not None and pert_type in self.pert_params and 'n' in self.pert_params[pert_type]['comps']:

            n_pert_params = {}
            for key, item in self.pert_params[pert_type].items():
                # Skip the comps entry
                if key == 'comps':
                    continue
                n_pert_params[key] = item['n']

            perturbation = getattr(perturbations, pert_type)(
                **n_pert_params
            )

            res += perturbation(eta1, eta2, eta3)

        return res

    def u(self, eta1, eta2, eta3):
        """ Mean velocities as background + perturbation.

        Parameters
        ----------
        eta1, eta2, eta3  : numpy.array
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        Without perturbation a list[float] (background value), with perturbation a list[numpy.array] of the evaluated velocities.
        """

        assert np.shape(eta1) == np.shape(eta2) == np.shape(eta3)

        res = [
            self.bckgr_params['u1'],
            self.bckgr_params['u2'],
            self.bckgr_params['u3'],
        ]

        # Add perturbation if parameters are given
        pert_type = self.pert_params['type']
        if pert_type is not None and pert_type in self.pert_params:
            for k in range(3):
                label = 'u' + str(k+1)

                # Add perturbation if it is in comps list
                if label in self.pert_params[pert_type]['comps']:

                    u_pert_params = {}
                    for key, item in self.pert_params[pert_type].items():
                        # Skip the comps entry
                        if key == 'comps':
                            continue
                        u_pert_params[key] = item[label]

                    perturbation = getattr(perturbations, pert_type)(
                        **u_pert_params
                    )

                    res[k] += perturbation(eta1, eta2, eta3)
        return res

    def vth(self, eta1, eta2, eta3):
        """ Thermal velocities as background + perturbation.

        Parameters
        ----------
        etas : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        Without perturbation a list[float] (background value), with perturbation a list[numpy.array] of the evaluated thermal velocities.
        """

        assert np.shape(eta1) == np.shape(eta2) == np.shape(eta3)

        res = [
            self.bckgr_params['vth1'],
            self.bckgr_params['vth2'],
            self.bckgr_params['vth3'],
        ]

        # Add perturbation if parameters are given
        pert_type = self.pert_params['type']
        if pert_type is not None and pert_type in self.pert_params:
            for k in range(3):
                label = 'vth' + str(k+1)

                # Add perturbation if it is in comps list
                if label in self.pert_params[pert_type]['comps']:

                    vth_pert_params = {}
                    for key, item in self.pert_params[pert_type].items():
                        # Skip the comps entry
                        if key == 'comps':
                            continue
                        vth_pert_params[key] = item[label]

                    perturbation = getattr(perturbations, pert_type)(
                        **vth_pert_params
                    )

                    res[k] += perturbation(eta1, eta2, eta3)

        return res


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

        bckgr_params = params['background'] if 'background' in params.keys() else {}

        # set default ITPA default parameters if not given
        if 'n' not in bckgr_params.keys():
            bckgr_params['n'] = {}

            bckgr_params['n']['c0'] = 0.491230
            bckgr_params['n']['c1'] = 0.298228
            bckgr_params['n']['c2'] = 0.198739
            bckgr_params['n']['c3'] = 0.521298

        if 'vth' not in bckgr_params.keys():
            bckgr_params['vth'] = 1.

        self._params = bckgr_params

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

        res_list += [self.params['vth'] - 0*eta1]
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

        # Get background type if given in params
        bckgr_type = None
        if 'background' in params.keys():
            if 'type' in params['background'].keys():
                bckgr_type = params['background']['type']

        # Get background params if given in params
        bckgr_params = {}
        if bckgr_type is not None:
            if bckgr_type in params['background'].keys():
                bckgr_params = params['background'][bckgr_type]

        # default parameters
        params_default = {'n': 1.,
                          'u_parallel': 0.,
                          'u_perp': 0.,
                          'vth_parallel': 1.,
                          'vth_perp': 1.}

        self._params = set_defaults(bckgr_params, params_default)

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

        bckgr_params = params['background'] if 'background' in params.keys() else {}

        # set default ITPA default parameters if not given
        if 'n' not in bckgr_params.keys():
            bckgr_params['n'] = {}
            bckgr_params['n']['n0'] = 0.00720655
            bckgr_params['n']['c0'] = 0.491230
            bckgr_params['n']['c1'] = 0.298228
            bckgr_params['n']['c2'] = 0.198739
            bckgr_params['n']['c3'] = 0.521298

        if 'vth' not in bckgr_params.keys():
            bckgr_params['vth'] = 1.

        self._params = bckgr_params

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
