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


class Maxwellian5D(Maxwellian):
    r""" A :class:`~struphy.kinetic_background.base.Maxwellian` with velocity dimension :math:`n=2`.

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
            'u_para': 0.,
            'u_perp': 0.,
            'vth_para': 1.,
            'vth_perp': 1.,
        }

    def __init__(self, **params):

        # Set background parameters as the default, then overwrite below if given
        self._bckgr_params = self.default_bckgr_params()

        if 'background' in params:
            assert params['background']['type'] == 'Maxwellian5D', \
                f"Background must be 'Maxwellian5D' but is {params['background']['type']}"

            if 'Maxwellian5D' in params['background'].keys():
                bckgr_params = params['background']['Maxwellian5D']
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
        return 2

    @property
    def is_polar(self):
        """List of booleans. True if the velocity coordinates are polar coordinates.
        """
        return [False, True]

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
            self.bckgr_params['u_para'],
            self.bckgr_params['u_perp']
        ]

        strings = ['_para', '_perp']

        # Add perturbation if parameters are given
        pert_type = self.pert_params['type']
        if pert_type is not None and pert_type in self.pert_params:
            for k in range(2):
                label = 'u' + strings[k]

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
            self.bckgr_params['vth_para'],
            self.bckgr_params['vth_perp']
        ]

        strings = ['_para', '_perp']

        # Add perturbation if parameters are given
        pert_type = self.pert_params['type']
        if pert_type is not None and pert_type in self.pert_params:
            for k in range(2):
                label = 'vth' + strings[k]

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
