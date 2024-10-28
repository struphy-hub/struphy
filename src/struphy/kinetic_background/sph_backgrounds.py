'SPH velocity distributions.'


import numpy as np

from struphy.kinetic_background.base import Maxwellian, CanonicalMaxwellian, KineticBackground
from struphy.fields_background.mhd_equil.equils import set_defaults
from struphy.initial import perturbations
from struphy.fields_background.mhd_equil.base import MHDequilibrium
from struphy.fields_background.braginskii_equil.base import BraginskiiEquilibrium

class Constant6D(KineticBackground):
    r""" Base class for a constant distribution function on the unit cube. 
    The Background does not depend on the velocity

    """

    @classmethod
    def default_maxw_params(cls):
        """ Default parameters dictionary defining the constant value of the constant background.
        """
        return {
            'density_profile' : 'constant',
            'n0': 5.,
            'n1': 0.,
            'u1': 0.,
            'u2': 0.,
            'u3': 0.,
        }

    def __init__(self, maxw_params=None, pert_params=None, mhd_equil=None, braginskii_equil=None):

        # Set background parameters
        self._maxw_params = self.default_maxw_params()

        if maxw_params is not None:
            assert isinstance(maxw_params, dict)
            self._maxw_params = set_defaults(
                maxw_params, self.default_maxw_params())

        assert self._maxw_params['density_profile'] in ['constant', 'affine']

        # Set parameters for perturbation
        self._pert_params = pert_params

        if self.pert_params is not None:
            assert isinstance(pert_params, dict)
            assert 'type' in self.pert_params, '"type" is mandatory in perturbation dictionary.'
            ptype = self.pert_params['type']
            assert ptype in self.pert_params, f'{ptype} is mandatory in perturbation dictionary.'
            self._pert_type = ptype

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
    def coords(self):
        """ Coordinates of the constant background.
        """
        return None

    @property
    def vdim(self):
        """ Dimension of the velocity space (vdim = 0).
        """
        return 0

    @property
    def is_polar(self):
        """ List of booleans of length vdim. True for a velocity coordinate that is a radial polar coordinate (v_perp).
        """
        return []

    @property
    def volume_form(self):
        """ Boolean. True if the background is represented as a volume form (thus including the velocity Jacobian).
        """
        return False

    @property
    def velocity_jacobian_det(self, eta1, eta2, eta3, *v):
        """ Jacobian determinant of the velocity coordinate transformation.
        """
        return 1.

    def n(self, *etas):
        """ Number density (0-form). 

        Parameters
        ----------
        etas : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the density evaluated at evaluation points (same shape as etas).
        """
        if self._maxw_params['density_profile']=='constant':
            return self.maxw_params['n0'] + 0 * etas[0]
        elif self._maxw_params['density_profile']=='affine':
            return self.maxw_params['n0'] + self.maxw_params['n1'] * etas[0]

    def u(self, *etas):
        """ Mean velocities (Cartesian components evaluated at x = F(eta)).

        Parameters
        ----------
        etas : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A list[float] (background values) or a list[numpy.array] of the evaluated velocities.
        """
        return np.array([0.*etas[0]+self._maxw_params['u1'],
                         0.*etas[0]+self._maxw_params['u2'],
                         0.*etas[0]+self._maxw_params['u3']
                         ])

    def __call__(self, eta1, eta2, eta3, v1, v2, v3):
        """ Evaluates the constant function.

        There are two use-cases for this function in the code:
            1.) Evaluating for particles (inputs are all of length N_p)
            2.) Evaluating the function on a meshgrid
        Hence all arguments must always have the same shape.


        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Position arguments.

        Returns
        -------
        f : np.ndarray
            The evaluated constant function.
        """

        # Check that all args have the same shape
        assert np.shape(eta1) == np.shape(eta2) == np.shape(eta3)

        # set background density
        res = self.n(eta1, eta2, eta3)

        assert np.all(res > 0.), 'Number density must be positive!'

        # Add perturbation if parameters are given and if density is to be perturbed
        if self.pert_params is not None and 'density' in self.pert_params[self._pert_type]['comps']:
            n_pert_params = {}
            for key, item in self.pert_params[self._pert_type].items():
                # Skip the comps entry
                if key == 'comps':
                    continue
                n_pert_params[key] = item['density']

            perturbation = getattr(perturbations, self._pert_type)(
                **n_pert_params)

            res += perturbation(eta1, eta2, eta3)
        return res + 0.*eta1