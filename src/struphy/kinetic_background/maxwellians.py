'Maxwellian (Gaussian) distributions in velocity space.'


import numpy as np

from struphy.kinetic_background.base import Maxwellian, CanonicalMaxwellian, KineticBackground
from struphy.fields_background.mhd_equil.equils import set_defaults
from struphy.initial import perturbations
from struphy.fields_background.mhd_equil.base import MHDequilibrium
from struphy.fields_background.braginskii_equil.base import BraginskiiEquilibrium


class Maxwellian3D(Maxwellian):
    r""" A :class:`~struphy.kinetic_background.base.Maxwellian` depending on three (:math:`n=3`) Cartesian velocities.

    Parameters
    ----------
    maxw_params : dict
        Parameters for the kinetic background.

    pert_params : dict
        Parameters for the kinetic perturbation added to the background.

    mhd_equil : MHDequilibrium
        One of :mod:`~struphy.fields_background.mhd_equil.equils`.

    braginskii_equil : BraginskiiEquilibrium
        One of :mod:`~struphy.fields_background.braginskii_equil.equils`.
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

    def __init__(self,
                 maxw_params: dict = {'n': 1.,
                                      'u1': 0.,
                                      'u2': 0.,
                                      'u3': 0.,
                                      'vth1': 1.,
                                      'vth2': 1.,
                                      'vth3': 1.},
                 pert_params: dict = None,
                 mhd_equil: MHDequilibrium = None,
                 braginskii_equil: BraginskiiEquilibrium = None):

        # Set background parameters
        self._maxw_params = self.default_maxw_params()

        if maxw_params is not None:
            assert isinstance(maxw_params, dict)
            self._maxw_params = set_defaults(
                maxw_params, self.default_maxw_params())

        # check if mhd or braginskii is needed
        for key, val in self.maxw_params.items():
            if val == 'mhd':
                assert isinstance(
                    mhd_equil, MHDequilibrium), f'MHD equilibrium must be passed to compute {key}.'

            if val == 'braginskii':
                assert isinstance(
                    braginskii_equil, BraginskiiEquilibrium), f'Braginskii equilibrium must be passed to compute {key}.'

        # Set parameters for perturbation
        self._pert_params = pert_params

        if self.pert_params is not None:
            assert isinstance(pert_params, dict)
            assert 'type' in self.pert_params, '"type" is mandatory in perturbation dictionary.'
            ptype = self.pert_params['type']
            assert ptype in self.pert_params, f'{ptype} is mandatory in perturbation dictionary.'
            self._pert_type = ptype

        # MHD and Braginskii equilibrium
        self._mhd_equil = mhd_equil
        self._braginskii_equil = braginskii_equil

        # factors multiplied onto the defined moments n, u and vth (can be set via setter)
        self._moment_factors = {'n': 1.,
                                'u': [1., 1., 1.],
                                'vth': [1., 1., 1.]}

    @property
    def coords(self):
        """ Coordinates of the Maxwellian6D, :math:`(v_1, v_2, v_3)`.
        """
        return 'cartesian'

    @property
    def maxw_params(self):
        """ Parameters dictionary defining constant moments of the Maxwellian.
        """
        return self._maxw_params

    @property
    def pert_params(self):
        """ Parameters dictionary defining the perturbations of the :meth:`~Maxwellian3D.maxw_params`.
        """
        return self._pert_params

    @property
    def mhd_equil(self):
        """ One of :mod:`~struphy.fields_background.mhd_equil.equils` 
        in case that moments are to be set in that way, None otherwise.
        """
        return self._mhd_equil

    @property
    def braginskii_equil(self):
        """ One of :mod:`~struphy.fields_background.braginskii_equil.equils` 
        in case that moments are to be set in that way, None otherwise.
        """
        return self._braginskii_equil

    @property
    def vdim(self):
        """Dimension of the velocity space.
        """
        return 3

    @property
    def is_polar(self):
        """List of booleans of length vdim. True for a velocity coordinate that is a radial polar coordinate (v_perp).
        """
        return [False, False, False]

    def velocity_jacobian_det(self, eta1, eta2, eta3, *v):
        """ Jacobian determinant of the velocity coordinate transformation from Maxwellian6D('cartesian') to Particles6D('cartesian').

        Input parameters should be slice of 2d numpy marker array. (i.e. *self.phasespace_coords.T)

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The Jacobian determinant evaluated at given logical coordinates.
        -------
        """

        assert eta1.ndim == 1
        assert eta2.ndim == 1
        assert eta3.ndim == 1
        assert len(v) == 3

        return 1. + 0*eta1

    @property
    def volume_form(self):
        """ Boolean. True if the background is represented as a volume form (thus including the velocity Jacobian).
        """
        return False

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
        if self.maxw_params['n'] == 'mhd':
            res = self.mhd_equil.n0(*etas)

            assert np.all(res > 0.), 'Number density must be positive!'

        elif self.maxw_params['n'] == 'braginskii':
            res = self.braginskii_equil.n0(*etas)

            assert np.all(res > 0.), 'Number density must be positive!'

        elif isinstance(self.maxw_params['n'], dict):
            type = self.maxw_params['n']['type']
            params = self.maxw_params['n'][type]
            nfun = getattr(perturbations, type)(**params)

            if eta1.ndim == 1:
                res = nfun(eta1, eta2, eta3)
            else:
                res = nfun(*etas)

        else:
            if eta1.ndim == 1:
                res = self.maxw_params['n'] + 0.*eta1
            else:
                res = self.maxw_params['n'] + 0.*etas[0]

        # Add perturbation if parameters are given and if density is to be perturbed
        if self.pert_params is not None and 'n' in self.pert_params[self._pert_type]['comps']:
            n_pert_params = {}
            for key, item in self.pert_params[self._pert_type].items():
                # Skip the comps entry
                if key == 'comps':
                    assert item['n'] == '0', 'Moment perturbations must be passed as 0-forms to Maxwellians.'
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
        if (self.maxw_params['u1'] == 'mhd' or
            self.maxw_params['u2'] == 'mhd' or
                self.maxw_params['u3'] == 'mhd'):

            tmp = self.mhd_equil.j_cart(*etas)[0] / self.mhd_equil.n0(*etas)

        if (self.maxw_params['u1'] == 'braginskii' or
            self.maxw_params['u2'] == 'braginskii' or
                self.maxw_params['u3'] == 'braginskii'):

            tmp2 = self.braginskii_equil.u_cart(*etas)

        res = [None, None, None]

        if self.maxw_params['u1'] == 'mhd':
            res[0] = tmp[0]

        elif self.maxw_params['u1'] == 'braginskii':
            res[0] = tmp2[0]

        else:
            if eta1.ndim == 1:
                res[0] = self.maxw_params['u1'] + 0.*eta1
            else:
                res[0] = self.maxw_params['u1'] + 0.*etas[0]

        if self.maxw_params['u2'] == 'mhd':
            res[1] = tmp[1]

        elif self.maxw_params['u2'] == 'braginskii':
            res[1] = tmp2[1]

        else:
            if eta1.ndim == 1:
                res[1] = self.maxw_params['u2'] + 0.*eta1
            else:
                res[1] = self.maxw_params['u2'] + 0.*etas[0]

        if self.maxw_params['u3'] == 'mhd':
            res[2] = tmp[2]

        elif self.maxw_params['u3'] == 'braginskii':
            res[2] = tmp2[2]

        else:
            if eta1.ndim == 1:
                res[2] = self.maxw_params['u3'] + 0.*eta1
            else:
                res[2] = self.maxw_params['u3'] + 0.*etas[0]

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
                            assert item[comp] == '0', 'Moment perturbations must be passed as 0-forms to Maxwellians.'
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
        if (self.maxw_params['vth1'] == 'mhd' or
            self.maxw_params['vth2'] == 'mhd' or
                self.maxw_params['vth3'] == 'mhd'):

            tmp = np.sqrt(self.mhd_equil.p0(*etas) / self.mhd_equil.n0(*etas))
            assert np.all(tmp > 0.), 'Thermal velocity must be positive!'

        if (self.maxw_params['vth1'] == 'brginskii' or
            self.maxw_params['vth2'] == 'braginskii' or
                self.maxw_params['vth3'] == 'braginskii'):

            tmp2 = np.sqrt(self.braginskii_equil.p0(*etas) /
                           self.braginskii_equil.n0(*etas))
            assert np.all(tmp2 > 0.), 'Thermal velocity must be positive!'

        res = [None, None, None]

        if self.maxw_params['vth1'] == 'mhd':
            res[0] = tmp

        elif self.maxw_params['vth1'] == 'braginskii':
            res[0] = tmp2

        else:
            if eta1.ndim == 1:
                res[0] = self.maxw_params['vth1'] + 0.*eta1
            else:
                res[0] = self.maxw_params['vth1'] + 0.*etas[0]

        if self.maxw_params['vth2'] == 'mhd':
            res[1] = tmp

        elif self.maxw_params['vth2'] == 'braginskii':
            res[1] = tmp2

        else:
            if eta1.ndim == 1:
                res[1] = self.maxw_params['vth2'] + 0.*eta1
            else:
                res[1] = self.maxw_params['vth2'] + 0.*etas[0]

        if self.maxw_params['vth3'] == 'mhd':
            res[2] = tmp

        elif self.maxw_params['vth3'] == 'braginskii':
            res[2] = tmp2

        else:
            if eta1.ndim == 1:
                res[2] = self.maxw_params['vth3'] + 0.*eta1
            else:
                res[2] = self.maxw_params['vth3'] + 0.*etas[0]

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
                            assert item[comp] == '0', 'Moment perturbations must be passed as 0-forms to Maxwellians.'
                            continue
                        vth_pert_params[key] = item[comp]

                    perturbation = getattr(perturbations, self._pert_type)(
                        **vth_pert_params)

                    if eta1.ndim == 1:
                        res[i] += perturbation(eta1, eta2, eta3)
                    else:
                        res[i] += perturbation(*etas)

        return [re * mom_fac for re, mom_fac in zip(res, self.moment_factors['vth'])]


class GyroMaxwellian2D(Maxwellian):
    r""" A gyrotropic :class:`~struphy.kinetic_background.base.Maxwellian` depending on
    two velocities :math:`(v_\parallel, v_\perp)`, :math:`n=2`, 
    where :math:`v_\parallel = \matbf v \cdot \mathbf b_0` and :math:`v_\perp`
    is the radial component of a polar coordinate system perpendicular
    to the magentic direction :math:`\mathbf b_0`.

    Parameters
    ----------    
    maxw_params : dict
        Parameters for the kinetic background.

    pert_params : dict
        Parameters for the kinetic perturbation added to the background.

    volume_form : bool
        Whether to represent the Maxwellian as a volume form; 
        if True it is multiplied by the Jacobian determinant |v_perp|
        of the polar coordinate transofrmation (default = False).

    mhd_equil : MHDequilibrium
        One of :mod:`~struphy.fields_background.mhd_equil.equils`.

    braginskii_equil : BraginskiiEquilibrium
        One of :mod:`~struphy.fields_background.braginskii_equil.equils`.
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

    def __init__(self,
                 maxw_params: dict = {'n': 1.,
                                      'u_para': 0.,
                                      'u_perp': 0.,
                                      'vth_para': 1.,
                                      'vth_perp': 1.},
                 pert_params: dict = None,
                 volume_form: bool = True,
                 mhd_equil: MHDequilibrium = None,
                 braginskii_equil: BraginskiiEquilibrium = None):

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

            if val == 'braginskii':
                assert isinstance(
                    braginskii_equil, BraginskiiEquilibrium), f'Braginskii equilibrium must be passed to compute {key}.'

        # Set parameters for perturbation
        self._pert_params = pert_params

        if self.pert_params is not None:
            assert isinstance(pert_params, dict)
            assert 'type' in self.pert_params, '"type" is mandatory in perturbation dictionary.'
            ptype = self.pert_params['type']
            assert ptype in self.pert_params, f'{ptype} is mandatory in perturbation dictionary.'
            self._pert_type = ptype

        # volume form represenation
        self._volume_form = volume_form

        # MHD and Braginskii equilibrium
        self._mhd_equil = mhd_equil
        self._braginskii_equil = braginskii_equil

        # factors multiplied onto the defined moments n, u and vth (can be set via setter)
        self._moment_factors = {'n': 1.,
                                'u': [1., 1.],
                                'vth': [1., 1.]}

    @property
    def coords(self):
        """ Coordinates of the Maxwellian5D, :math:`(v_\parallel, v_\perp)`.
        """
        return 'vpara_vperp'

    @property
    def maxw_params(self):
        """ Parameters dictionary defining constant moments of the Maxwellian.
        """
        return self._maxw_params

    @property
    def pert_params(self):
        """ Parameters dictionary defining the perturbations of the :meth:`~GyroMaxwellian2D.maxw_params`.
        """
        return self._pert_params

    @property
    def mhd_equil(self):
        """ One of :mod:`~struphy.fields_background.mhd_equil.equils` 
        in case that moments are to be set in that way, None otherwise.
        """
        return self._mhd_equil

    @property
    def braginskii_equil(self):
        """ One of :mod:`~struphy.fields_background.braginskii_equil.equils` 
        in case that moments are to be set in that way, None otherwise.
        """
        return self._braginskii_equil

    @property
    def vdim(self):
        """Dimension of the velocity space.
        """
        return 2

    @property
    def is_polar(self):
        """List of booleans of length vdim. True for a velocity coordinate that is a radial polar coordinate (v_perp).
        """
        return [False, True]

    def velocity_jacobian_det(self, eta1, eta2, eta3, *v):
        r"""Jacobian determinant of the velocity coordinate transformation from Maxwellian5D('vpara_vperp') to Particles5D('vpara_mu').

        .. math::

            \begin{aligned}
            F &: (v_\parallel, v_\perp) \to (v_\parallel, \mu) \,,
            \\[3mm]
            DF &= \begin{bmatrix} \frac{\partial v_\parallel}{\partial v_\parallel} & \frac{\partial v_\parallel}{\partial v_\perp} \\
                 \frac{\partial \mu}{\partial v_\parallel} & \frac{\partial \mu}{\partial v_\perp}  \end{bmatrix} =
                 \begin{bmatrix} 1 & 0 \\
                 0 & \frac{v_\perp}{B}  \end{bmatrix} \,,
            \\[3mm]
            J_F &= \frac{v_\perp}{B} \,,
            \end{aligned}

        where :math:`\mu = \frac{v_\perp^2}{2B}`.

        Input parameters should be slice of 2d numpy marker array. (i.e. *self.phasespace_coords.T)

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The Jacobian determinant evaluated at given logical coordinates.
        -------
        """

        assert eta1.ndim == 1
        assert eta2.ndim == 1
        assert eta3.ndim == 1
        assert len(v) == 2

        # call equilibrium
        etas = (np.vstack((eta1, eta2, eta3)).T).copy()
        absB0 = self.mhd_equil.absB0(etas)

        # J = v_perp/B
        jacobian_det = v[1]/absB0

        return jacobian_det

    @property
    def volume_form(self):
        """ Boolean. True if the background is represented as a volume form (thus including the velocity Jacobian |v_perp|).
        """
        return self._volume_form

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
        if self.maxw_params['n'] == 'mhd':
            res = self.mhd_equil.n0(*etas)

            assert np.all(res > 0.), 'Number density must be positive!'

        elif self.maxw_params['n'] == 'braginskii':
            res = self.braginskii_equil.n0(*etas)

            assert np.all(res > 0.), 'Number density must be positive!'

        elif isinstance(self.maxw_params['n'], dict):
            type = self.maxw_params['n']['type']
            params = self.maxw_params['n'][type]
            nfun = getattr(perturbations, type)(**params)

            if eta1.ndim == 1:
                res = nfun(eta1, eta2, eta3)
            else:
                res = nfun(*etas)

        else:
            if eta1.ndim == 1:
                res = self.maxw_params['n'] + 0.*eta1

            else:
                res = self.maxw_params['n'] + 0.*etas[0]

        # Add perturbation if parameters are given and if density is to be perturbed
        if self.pert_params is not None:
            if 'n' in self.pert_params[self._pert_type]['comps']:
                n_pert_params = {}
                for key, item in self.pert_params[self._pert_type].items():
                    # Skip the comps entry
                    if key == 'comps':
                        assert item['n'] == '0', 'Moment perturbations must be passed as 0-forms to Maxwellians.'
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
        if (self.maxw_params['u_para'] == 'mhd' or
                self.maxw_params['u_perp'] == 'mhd'):

            tmp_jv = self.mhd_equil.jv(*etas) / self.mhd_equil.n0(*etas)
            tmp_unit_b1 = self.mhd_equil.unit_b1(*etas)
            # j_parallel = jv.b1
            j_para = sum([ji * bi for ji, bi in zip(tmp_jv, tmp_unit_b1)])

        if (self.maxw_params['u_para'] == 'braginskii' or
                self.maxw_params['u_perp'] == 'braginskii'):

            tmp_uv = self.braginskii_equil.uv(
                *etas) / self.braginskii_equil.n0(*etas)
            tmp_unit_b1 = self.braginskii_equil.unit_b1(*etas)
            # u_parallel = uv.b1
            u_para = sum([ji * bi for ji, bi in zip(tmp_uv, tmp_unit_b1)])

        res = [None, None]

        if self.maxw_params['u_para'] == 'mhd':
            res[0] = j_para
        elif self.maxw_params['u_para'] == 'braginskii':
            res[0] = u_para
        else:
            if eta1.ndim == 1:
                res[0] = self.maxw_params['u_para'] + 0.*eta1
            else:
                res[0] = self.maxw_params['u_para'] + 0.*etas[0]

        if self.maxw_params['u_perp'] == 'mhd':
            raise NotImplementedError(
                'A shift in v_perp is not yet implemented.')
        elif self.maxw_params['u_perp'] == 'braginskii':
            raise NotImplementedError(
                'A shift in v_perp is not yet implemented.')
        else:
            if eta1.ndim == 1:
                res[1] = self.maxw_params['u_perp'] + 0.*eta1
            else:
                res[1] = self.maxw_params['u_perp'] + 0.*etas[0]

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
                            assert item[comp] == '0', 'Moment perturbations must be passed as 0-forms to Maxwellians.'
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
        if (self.maxw_params['vth_para'] == 'mhd' or
                self.maxw_params['vth_perp'] == 'mhd'):

            tmp = np.sqrt(self.mhd_equil.p0(*etas) / self.mhd_equil.n0(*etas))
            assert np.all(tmp > 0.), 'Thermal velocity must be positive!'

        if (self.maxw_params['vth_para'] == 'braginskii' or
                self.maxw_params['vth_perp'] == 'braginskii'):

            tmp2 = np.sqrt(self.braginskii_equil.p0(*etas) /
                           self.braginskii_equil.n0(*etas))
            assert np.all(tmp2 > 0.), 'Thermal velocity must be positive!'

        res = [None, None]

        if self.maxw_params['vth_para'] == 'mhd':
            res[0] = tmp
        elif self.maxw_params['vth_para'] == 'braginskii':
            res[0] = tmp2
        else:
            if eta1.ndim == 1:
                res[0] = self.maxw_params['vth_para'] + 0.*eta1
            else:
                res[0] = self.maxw_params['vth_para'] + 0.*etas[0]

        if self.maxw_params['vth_perp'] == 'mhd':
            res[1] = tmp
        elif self.maxw_params['vth_perp'] == 'braginskii':
            res[1] = tmp2
        else:
            if eta1.ndim == 1:
                res[1] = self.maxw_params['vth_perp'] + 0.*eta1
            else:
                res[1] = self.maxw_params['vth_perp'] + 0.*etas[0]

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
                            assert item[comp] == '0', 'Moment perturbations must be passed as 0-forms to Maxwellians.'
                            continue
                        vth_pert_params[key] = item[comp]

                    perturbation = getattr(perturbations, self._pert_type)(
                        **vth_pert_params)

                    if eta1.ndim == 1:
                        res[i] += perturbation(eta1, eta2, eta3)
                    else:
                        res[i] += perturbation(*etas)

        return [re * mom_fac for re, mom_fac in zip(res, self.moment_factors['vth'])]


class CanonicalMaxwellian(CanonicalMaxwellian):
    r""" A :class:`~struphy.kinetic_background.base.CanonicalMaxwellian`.

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
            'vth': 1.,
        }

    def __init__(self,
                 maxw_params: dict = {'n': 1.,
                                      'vth': 1., },
                 pert_params: dict = None,
                 volume_form: bool = True,
                 mhd_equil: MHDequilibrium = None,
                 braginskii_equil: BraginskiiEquilibrium = None):

        # Set background parameters
        self._maxw_params = self.default_maxw_params()

        if maxw_params is not None:
            assert isinstance(maxw_params, dict)
            self._maxw_params = set_defaults(
                maxw_params, self.default_maxw_params())

        # Set parameters for perturbation
        self._pert_params = pert_params

        if self.pert_params is not None:
            assert isinstance(pert_params, dict)
            assert 'type' in self.pert_params, '"type" is mandatory in perturbation dictionary.'
            ptype = self.pert_params['type']
            assert ptype in self.pert_params, f'{ptype} is mandatory in perturbation dictionary.'
            self._pert_type = ptype

        self._mhd_equil = mhd_equil

        # volume form represenation
        self._volume_form = volume_form

        # factors multiplied onto the defined moments n and vth (can be set via setter)
        self._moment_factors = {'n': 1.,
                                'vth': 1.}

    @property
    def coords(self):
        """ Coordinates of the CanonicalMaxwellian, :math:`(\epsilon, \mu, \psi_c)`.
        """
        return 'constants_of_motion'

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
    def braginskii_equil(self):
        """ One of :mod:`~struphy.fields_background.braginskii_equil.equils` 
        in case that moments are to be set in that way, None otherwise.
        """
        return self._braginskii_equil

    def velocity_jacobian_det(self, eta1, eta2, eta3, energy):
        r"""TODO
        """

        assert eta1.ndim == 1
        assert eta2.ndim == 1
        assert eta3.ndim == 1

        # call equilibrium
        etas = (np.vstack((eta1, eta2, eta3)).T).copy()
        absB0 = self.mhd_equil.absB0(etas)

        jacobian_det = np.sqrt(energy) * 2. * np.sqrt(2.) / absB0

        return jacobian_det

    @property
    def volume_form(self):
        """ Boolean. True if the background is represented as a volume form (thus including the velocity Jacobian |v_perp|).
        """
        return self._volume_form

    @property
    def moment_factors(self):
        """Collection of factors multiplied onto the defined moments n, u, and vth.
        """
        return self._moment_factors

    @moment_factors.setter
    def moment_factors(self, **kwargs):
        for kw, arg in kwargs:

            self._moment_factors[kw] = arg

    def rc(self, psic):
        r""" Square root of radially normalized canonical toroidal momentum.

        .. math::
            \begin{aligned}
            r_c^2 &= \frac{\psi_c - \psi_\text{axis}}{\psi_\text{edge} - \psi_\text{axis}} \,,
            \\[3mm]
            r_c &= \begin{cases}
            \sqrt{\frac{\psi_c - \psi_\text{axis}}{\psi_\text{edge} - \psi_\text{axis}}} & \text{if} \quad \frac{\psi_c - \psi_\text{axis}}{\psi_\text{edge} - \psi_\text{axis}} \geq 0 \,, \\
            -\sqrt{\frac{\psi_c - \psi_\text{axis}}{\psi_\text{edge} - \psi_\text{axis}}} & \text{if} \quad \frac{\psi_c - \psi_\text{axis}}{\psi_\text{edge} - \psi_\text{axis}} < 0 \,,
            \end{cases}
            \end{aligned}

        where :math:`\psi_\text{axis}` and :math:`\psi_\text{edge}` are poloidal magnetic flux function at the center and edge of poloidal plane respectively.

        Parameters
        ----------
        psic : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array of the evaluated :math:`r_c`.

        """

        # calculate rc²
        rc_squared = (psic - self.mhd_equil.psi_range[0])/(
            self.mhd_equil.psi_range[1] - self.mhd_equil.psi_range[0])

        # sorting out indices of negative rc²
        neg_index = np.logical_not(rc_squared >= 0)

        # make them positive
        rc_squared[neg_index] *= -1

        # calculate rc
        rc = np.sqrt(rc_squared)
        rc[neg_index] *= -1

        return rc

    def n(self, psic):
        """ Density as background + perturbation.

        Parameters
        ----------
        psic : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A float (background value) or a numpy.array of the evaluated density.
        """

        # collect arguments
        assert isinstance(psic, np.ndarray)

        # assuming that input comes from meshgrid.
        if psic.ndim == 3:
            psic = psic[0, 0, :]

        # set background density
        if isinstance(self.maxw_params['n'], dict):
            type = self.maxw_params['n']['type']
            params = self.maxw_params['n'][type]
            nfun = getattr(perturbations, type)(**params)

            res = nfun(eta1=self.rc(psic))

        else:
            res = self.maxw_params['n'] + 0.*psic

        # Add perturbation if parameters are given and if density is to be perturbed
        if self.pert_params is not None:
            if 'n' in self.pert_params[self._pert_type]['comps']:
                n_pert_params = {}
                for key, item in self.pert_params[self._pert_type].items():
                    # Skip the comps entry
                    if key == 'comps':
                        assert item['n'] == '0', 'Moment perturbations must be passed as 0-forms to Maxwellians.'
                        continue
                    n_pert_params[key] = item['n']

                perturbation = getattr(perturbations, self._pert_type)(
                    **n_pert_params)

                res += perturbation(psic)

        return res * self.moment_factors['n']

    def vth(self, psic):
        """ Thermal velocities as background + perturbation.

        Parameters
        ----------
        psic : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A list[float] (background value) or a list[numpy.array] of the evaluated thermal velocities.
        """

        # collect arguments
        assert isinstance(psic, np.ndarray)

        # assuming that input comes from meshgrid.
        if psic.ndim == 3:
            psic = psic[0, 0, :]

        res = self.maxw_params['vth'] + 0.*psic

        # Add perturbation if parameters are given
        if self.pert_params is not None:
            comps = ['vth']
            for i, comp in enumerate(comps):
                if comp in self.pert_params[self._pert_type]['comps']:
                    # Add perturbation if it is in comps list
                    vth_pert_params = {}
                    for key, item in self.pert_params[self._pert_type].items():
                        # Skip the comps entry
                        if key == 'comps':
                            assert item[comp] == '0', 'Moment perturbations must be passed as 0-forms to Maxwellians.'
                            continue
                        vth_pert_params[key] = item[comp]

                    perturbation = getattr(perturbations, self._pert_type)(
                        **vth_pert_params)

                    res += perturbation(psic)

        return res * self.moment_factors['vth']


class Constant(KineticBackground):
    r""" Base class for a constant distribution function on the unit cube. 

    """

    @classmethod
    def default_maxw_params(cls):
        """ Default parameters dictionary defining the constant value of the constant background.
        """
        return {
            'n': 5.
        }

    def __init__(self, maxw_params=None, pert_params=None, mhd_equil=None, braginskii_equil=None):

        # Set background parameters
        self._maxw_params = self.default_maxw_params()

        if maxw_params is not None:
            assert isinstance(maxw_params, dict)
            self._maxw_params = set_defaults(
                maxw_params, self.default_maxw_params())

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

    @property
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
        return self.maxw_params['n'] + 0 * etas[0]

    @property
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
        return [0.*etas[0], 0.*etas[0], 0.*etas[0]]

    def __call__(self, eta1, eta2, eta3):
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
        res = self.maxw_params['n']

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
