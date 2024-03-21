'Maxwellian (Gaussian) distributions in velocity space.'


import numpy as np

from struphy.kinetic_background.base import Maxwellian
from struphy.fields_background.mhd_equil.equils import set_defaults
from struphy.initial import perturbations
from struphy.fields_background.mhd_equil.base import MHDequilibrium


class Maxwellian6D(Maxwellian):
    r""" A :class:`~struphy.kinetic_background.base.Maxwellian` with velocity dimension :math:`n=3`.

    Parameters
    ----------
    maxw_params : dict
        Parameters for the kinetic background.

    pert_params : dict
        Parameters for the kinetic perturbation added to the background.

    mhd_equil : MHDequilibrium
        One of :mod:`~struphy.fields_background.mhd_equil.equils`.
    """

    @classmethod
    def default_maxw_params(cls):
        """ Default parameters dictionary defining constant moments of the Maxwellian.
        """
        return {
            'n': 1.,
            'u1': 0.,
            'u2': 0.,
            'u3': 0.,
            'vth1': 1.,
            'vth2': 1.,
            'vth3': 1.
        }

    def __init__(self, maxw_params=None, pert_params=None, mhd_equil=None):

        # Set background parameters
        self._maxw_params = self.default_maxw_params()

        if maxw_params is not None:
            assert isinstance(maxw_params, dict)
            self._maxw_params = set_defaults(
                maxw_params, self.default_maxw_params())

        # check if mhd is needed
        for key, val in self.maxw_params.items():
            if val == 'mhd':
                assert isinstance(
                    mhd_equil, MHDequilibrium), f'MHD equilibrium must be passed to compute {key}.'

        # Set parameters for perturbation
        self._pert_params = pert_params

        if self.pert_params is not None:
            assert isinstance(pert_params, dict)
            assert 'type' in self.pert_params, '"type" is mandatory in perturbation dictionary.'
            ptype = self.pert_params['type']
            assert ptype in self.pert_params, f'{ptype} is mandatory in perturbation dictionary.'
            self._pert_type = ptype

        # MHD equilibrium
        self._mhd_equil = mhd_equil

        # factors multiplied onto the defined moments n, u and vth (can be set via setter)
        self._moment_factors = {'n': 1.,
                                'u': [1., 1., 1.],
                                'vth': [1., 1., 1.]}

    @property
    def maxw_params(self):
        """ Parameters dictionary defining constant moments of the Maxwellian.
        """
        return self._maxw_params

    @property
    def pert_params(self):
        """ Parameters dictionary defining the perturbations of the :meth:`~Maxwellian6D.maxw_params`.
        """
        return self._pert_params

    @property
    def mhd_equil(self):
        """ One of :mod:`~struphy.fields_background.mhd_equil.equils` 
        in case that moments are to be set in that way, None otherwise.
        """
        return self._mhd_equil

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

    @property
    def moment_factors(self):
        """Collection of factors multiplied onto the defined moments n, u, and vth.
        """
        return self._moment_factors

    @moment_factors.setter
    def moment_factors(self, **kwargs):
        for kw, arg in kwargs:
            if kw in {'u', 'vth'}:
                assert len(arg) == 3
            self._moment_factors[kw] = arg

    def n(self, eta1, eta2, eta3):
        """ Density as background + perturbation.

        Parameters
        ----------
        eta1, eta2, eta3 : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A float (background value) or a numpy.array of the evaluated density.
        """

        # collect arguments
        assert isinstance(eta1, np.ndarray)
        assert isinstance(eta2, np.ndarray)
        assert isinstance(eta3, np.ndarray)
        assert eta1.shape == eta2.shape == eta3.shape

        # flat evaluation for markers
        if eta1.ndim == 1:
            etas = [np.concatenate(
                (eta1[:, None], eta2[:, None], eta3[:, None]), axis=1)]
        # assuming that input comes from meshgrid.
        elif eta1.ndim == 6:
            etas = (eta1[:, :, :, 0, 0, 0],
                    eta2[:, :, :, 0, 0, 0],
                    eta3[:, :, :, 0, 0, 0])
        else:
            etas = (eta1, eta2, eta3)

        # set background density
        res = self.maxw_params['n']

        if self.maxw_params['n'] == 'mhd':
            res = self.mhd_equil.n0(*etas)

            assert np.all(res > 0.), 'Number density must be positive!'

        # Add perturbation if parameters are given and if density is to be perturbed
        if self.pert_params is not None and 'n' in self.pert_params[self._pert_type]['comps']:
            n_pert_params = {}
            for key, item in self.pert_params[self._pert_type].items():
                # Skip the comps entry
                if key == 'comps':
                    continue
                n_pert_params[key] = item['n']

            perturbation = getattr(perturbations, self._pert_type)(
                **n_pert_params)

            if eta1.ndim == 1:
                res += perturbation(eta1, eta2, eta3)
            else:
                res += perturbation(*etas)

        return res * self.moment_factors['n']

    def u(self, eta1, eta2, eta3):
        """ Mean velocities as background + perturbation.

        Parameters
        ----------
        eta1, eta2, eta3  : numpy.array
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A list[float] (background values) or a list[numpy.array] of the evaluated velocities.
        """

        # collect arguments
        assert isinstance(eta1, np.ndarray)
        assert isinstance(eta2, np.ndarray)
        assert isinstance(eta3, np.ndarray)
        assert eta1.shape == eta2.shape == eta3.shape

        # flat evaluation for markers
        if eta1.ndim == 1:
            etas = [np.concatenate(
                (eta1[:, None], eta2[:, None], eta3[:, None]), axis=1)]
        # assuming that input comes from meshgrid.
        elif eta1.ndim == 6:
            etas = (eta1[:, :, :, 0, 0, 0],
                    eta2[:, :, :, 0, 0, 0],
                    eta3[:, :, :, 0, 0, 0])
        else:
            etas = (eta1, eta2, eta3)

        # set background velocity
        res = [
            self.maxw_params['u1'],
            self.maxw_params['u2'],
            self.maxw_params['u3'],
        ]

        if (self.maxw_params['u1'] == 'mhd' or
            self.maxw_params['u2'] == 'mhd' or
                self.maxw_params['u3'] == 'mhd'):

            tmp = self.mhd_equil.j_cart(*etas)[0] / self.mhd_equil.n0(*etas)

        if self.maxw_params['u1'] == 'mhd':
            res[0] = tmp[0]

        if self.maxw_params['u2'] == 'mhd':
            res[1] = tmp[1]

        if self.maxw_params['u3'] == 'mhd':
            res[2] = tmp[2]

        # Add perturbation if parameters are given
        if self.pert_params is not None:
            comps = ['u1', 'u2', 'u3']
            for i, comp in enumerate(comps):
                if comp in self.pert_params[self._pert_type]['comps']:
                    # Add perturbation if it is in comps list
                    u_pert_params = {}
                    for key, item in self.pert_params[self._pert_type].items():
                        # Skip the comps entry
                        if key == 'comps':
                            continue
                        u_pert_params[key] = item[comp]

                    perturbation = getattr(perturbations, self._pert_type)(
                        **u_pert_params)

                    if eta1.ndim == 1:
                        res[i] += perturbation(eta1, eta2, eta3)
                    else:
                        res[i] += perturbation(*etas)

        return [re * mom_fac for re, mom_fac in zip(res, self.moment_factors['u'])]

    def vth(self, eta1, eta2, eta3):
        """ Thermal velocities as background + perturbation.

        Parameters
        ----------
        etas : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A list[float] (background value) or a list[numpy.array] of the evaluated thermal velocities.
        """

        # collect arguments
        assert isinstance(eta1, np.ndarray)
        assert isinstance(eta2, np.ndarray)
        assert isinstance(eta3, np.ndarray)
        assert eta1.shape == eta2.shape == eta3.shape

        # flat evaluation for markers
        if eta1.ndim == 1:
            etas = [np.concatenate(
                (eta1[:, None], eta2[:, None], eta3[:, None]), axis=1)]
        # assuming that input comes from meshgrid.
        elif eta1.ndim == 6:
            etas = (eta1[:, :, :, 0, 0, 0],
                    eta2[:, :, :, 0, 0, 0],
                    eta3[:, :, :, 0, 0, 0])
        else:
            etas = (eta1, eta2, eta3)

        # set background thermal velocity
        res = [
            self.maxw_params['vth1'],
            self.maxw_params['vth2'],
            self.maxw_params['vth3'],
        ]

        if (self.maxw_params['vth1'] == 'mhd' or
            self.maxw_params['vth2'] == 'mhd' or
                self.maxw_params['vth3'] == 'mhd'):

            tmp = np.sqrt(self.mhd_equil.p0(*etas) / self.mhd_equil.n0(*etas))
            assert np.all(tmp > 0.), 'Thermal velocity must be positive!'

        if self.maxw_params['vth1'] == 'mhd':
            res[0] = tmp

        if self.maxw_params['vth2'] == 'mhd':
            res[1] = tmp

        if self.maxw_params['vth3'] == 'mhd':
            res[2] = tmp

        # Add perturbation if parameters are given
        if self.pert_params is not None:
            comps = ['vth1', 'vth2', 'vth3']
            for i, comp in enumerate(comps):
                if comp in self.pert_params[self._pert_type]['comps']:
                    # Add perturbation if it is in comps list
                    vth_pert_params = {}
                    for key, item in self.pert_params[self._pert_type].items():
                        # Skip the comps entry
                        if key == 'comps':
                            continue
                        vth_pert_params[key] = item[comp]

                    perturbation = getattr(perturbations, self._pert_type)(
                        **vth_pert_params)

                    if eta1.ndim == 1:
                        res[i] += perturbation(eta1, eta2, eta3)
                    else:
                        res[i] += perturbation(*etas)

        return [re * mom_fac for re, mom_fac in zip(res, self.moment_factors['vth'])]


class Maxwellian5D(Maxwellian):
    r""" A :class:`~struphy.kinetic_background.base.Maxwellian` with velocity dimension :math:`n=2`.

    Parameters
    ----------
    maxw_params : dict
        Parameters for the kinetic background.

    pert_params : dict
        Parameters for the kinetic perturbation added to the background.

    mhd_equil : MHDequilibrium
        One of :mod:`~struphy.fields_background.mhd_equil.equils`.
    """

    @classmethod
    def default_maxw_params(cls):
        """ Default parameters dictionary defining constant moments of the Maxwellian.
        """
        return {
            'n': 1.,
            'u_para': 0.,
            'u_perp': 0.,
            'vth_para': 1.,
            'vth_perp': 1.,
        }

    def __init__(self, maxw_params=None, pert_params=None, mhd_equil=None):

        # Set background parameters
        self._maxw_params = self.default_maxw_params()

        if maxw_params is not None:
            assert isinstance(maxw_params, dict)
            self._maxw_params = set_defaults(
                maxw_params, self.default_maxw_params())

        # check if mhd is needed
        for key, val in self.maxw_params.items():
            if val == 'mhd':
                assert isinstance(
                    mhd_equil, MHDequilibrium), f'MHD equilibrium must be passed to compute {key}.'

        # Set parameters for perturbation
        self._pert_params = pert_params

        if self.pert_params is not None:
            assert isinstance(pert_params, dict)
            assert 'type' in self.pert_params, '"type" is mandatory in perturbation dictionary.'
            ptype = self.pert_params['type']
            assert ptype in self.pert_params, f'{ptype} is mandatory in perturbation dictionary.'
            self._pert_type = ptype

        # MHD equilibrium
        self._mhd_equil = mhd_equil

        # factors multiplied onto the defined moments n, u and vth (can be set via setter)
        self._moment_factors = {'n': 1.,
                                'u': [1., 1.],
                                'vth': [1., 1.]}

    @property
    def maxw_params(self):
        """ Parameters dictionary defining constant moments of the Maxwellian.
        """
        return self._maxw_params

    @property
    def pert_params(self):
        """ Parameters dictionary defining the perturbations of the :meth:`~Maxwellian5D.maxw_params`.
        """
        return self._pert_params

    @property
    def mhd_equil(self):
        """ One of :mod:`~struphy.fields_background.mhd_equil.equils` 
        in case that moments are to be set in that way, None otherwise.
        """
        return self._mhd_equil

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

    @property
    def moment_factors(self):
        """Collection of factors multiplied onto the defined moments n, u, and vth.
        """
        return self._moment_factors

    @moment_factors.setter
    def moment_factors(self, **kwargs):
        for kw, arg in kwargs:
            if kw in {'u', 'vth'}:
                assert len(arg) == 2
            self._moment_factors[kw] = arg

    def n(self, eta1, eta2, eta3):
        """ Density as background + perturbation.

        Parameters
        ----------
        eta1, eta2, eta3 : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A float (background value) or a numpy.array of the evaluated density.
        """

        # collect arguments
        assert isinstance(eta1, np.ndarray)
        assert isinstance(eta2, np.ndarray)
        assert isinstance(eta3, np.ndarray)
        assert eta1.shape == eta2.shape == eta3.shape

        # flat evaluation for markers
        if eta1.ndim == 1:
            etas = [np.concatenate(
                (eta1[:, None], eta2[:, None], eta3[:, None]), axis=1)]
        # assuming that input comes from meshgrid.
        elif eta1.ndim == 5:
            etas = (eta1[:, :, :, 0, 0],
                    eta2[:, :, :, 0, 0],
                    eta3[:, :, :, 0, 0])
        else:
            etas = (eta1, eta2, eta3)

        # set background density
        res = self.maxw_params['n']

        if self.maxw_params['n'] == 'mhd':
            res = self.mhd_equil.n0(*etas)

            assert np.all(res > 0.), 'Number density must be positive!'

        # Add perturbation if parameters are given and if density is to be perturbed
        if self.pert_params is not None and 'n' in self.pert_params[self._pert_type]['comps']:
            n_pert_params = {}
            for key, item in self.pert_params[self._pert_type].items():
                # Skip the comps entry
                if key == 'comps':
                    continue
                n_pert_params[key] = item['n']

            perturbation = getattr(perturbations, self._pert_type)(
                **n_pert_params)

            if eta1.ndim == 1:
                res += perturbation(eta1, eta2, eta3)
            else:
                res += perturbation(*etas)

        return res * self.moment_factors['n']

    def u(self, eta1, eta2, eta3):
        """ Mean velocities as background + perturbation.

        Parameters
        ----------
        eta1, eta2, eta3  : numpy.array
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A list[float] (background value) or a list[numpy.array] of the evaluated velocities.
        """

        # collect arguments
        assert isinstance(eta1, np.ndarray)
        assert isinstance(eta2, np.ndarray)
        assert isinstance(eta3, np.ndarray)
        assert eta1.shape == eta2.shape == eta3.shape

        # flat evaluation for markers
        if eta1.ndim == 1:
            etas = [np.concatenate(
                (eta1[:, None], eta2[:, None], eta3[:, None]), axis=1)]
        # assuming that input comes from meshgrid.
        elif eta1.ndim == 5:
            etas = (eta1[:, :, :, 0, 0],
                    eta2[:, :, :, 0, 0],
                    eta3[:, :, :, 0, 0])
        else:
            etas = (eta1, eta2, eta3)

        # set background velocity
        res = [
            self.maxw_params['u_para'],
            self.maxw_params['u_perp']
        ]

        if (self.maxw_params['u_para'] == 'mhd' or
                self.maxw_params['u_perp'] == 'mhd'):

            tmp_jv = self.mhd_equil.jv(*etas) / self.mhd_equil.n0(*etas)
            tmp_unit_b1 = self.mhd_equil.unit_b1(*etas)
            # j_parallel = jv.b1
            j_para = sum([ji * bi for ji, bi in zip(tmp_jv, tmp_unit_b1)])

        if self.maxw_params['u_para'] == 'mhd':
            res[0] = j_para

        if self.maxw_params['u_perp'] == 'mhd':
            raise NotImplementedError(
                'A shift in v_perp is not yet implemented.')

        # Add perturbation if parameters are given
        if self.pert_params is not None:
            comps = ['u_para', 'u_perp']
            for i, comp in enumerate(comps):
                if comp in self.pert_params[self._pert_type]['comps']:
                    # Add perturbation if it is in comps list
                    u_pert_params = {}
                    for key, item in self.pert_params[self._pert_type].items():
                        # Skip the comps entry
                        if key == 'comps':
                            continue
                        u_pert_params[key] = item[comp]

                    perturbation = getattr(perturbations, self._pert_type)(
                        **u_pert_params)

                    if eta1.ndim == 1:
                        res[i] += perturbation(eta1, eta2, eta3)
                    else:
                        res[i] += perturbation(*etas)

        return [re * mom_fac for re, mom_fac in zip(res, self.moment_factors['u'])]

    def vth(self, eta1, eta2, eta3):
        """ Thermal velocities as background + perturbation.

        Parameters
        ----------
        etas : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A list[float] (background value) or a list[numpy.array] of the evaluated thermal velocities.
        """

        # collect arguments
        assert isinstance(eta1, np.ndarray)
        assert isinstance(eta2, np.ndarray)
        assert isinstance(eta3, np.ndarray)
        assert eta1.shape == eta2.shape == eta3.shape

        # flat evaluation for markers
        if eta1.ndim == 1:
            etas = [np.concatenate(
                (eta1[:, None], eta2[:, None], eta3[:, None]), axis=1)]
        # assuming that input comes from meshgrid.
        elif eta1.ndim == 5:
            etas = (eta1[:, :, :, 0, 0],
                    eta2[:, :, :, 0, 0],
                    eta3[:, :, :, 0, 0])
        else:
            etas = (eta1, eta2, eta3)

        # set background thermal velocity
        res = [
            self.maxw_params['vth_para'],
            self.maxw_params['vth_perp']
        ]

        if (self.maxw_params['vth_para'] == 'mhd' or
                self.maxw_params['vth_perp'] == 'mhd'):

            tmp = np.sqrt(self.mhd_equil.p0(*etas) / self.mhd_equil.n0(*etas))
            assert np.all(tmp > 0.), 'Thermal velocity must be positive!'

        if self.maxw_params['vth_para'] == 'mhd':
            res[0] = tmp

        if self.maxw_params['vth_perp'] == 'mhd':
            res[1] = tmp

        # Add perturbation if parameters are given
        if self.pert_params is not None:
            comps = ['vth_para', 'vth_perp']
            for i, comp in enumerate(comps):
                if comp in self.pert_params[self._pert_type]['comps']:
                    # Add perturbation if it is in comps list
                    vth_pert_params = {}
                    for key, item in self.pert_params[self._pert_type].items():
                        # Skip the comps entry
                        if key == 'comps':
                            continue
                        vth_pert_params[key] = item[comp]

                    perturbation = getattr(perturbations, self._pert_type)(
                        **vth_pert_params)

                    if eta1.ndim == 1:
                        res[i] += perturbation(eta1, eta2, eta3)
                    else:
                        res[i] += perturbation(*etas)

        return [re * mom_fac for re, mom_fac in zip(res, self.moment_factors['vth'])]
