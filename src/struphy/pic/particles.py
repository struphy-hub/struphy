import numpy as np
from struphy.pic.base import Particles
from struphy.pic import utilities_kernels
from struphy.kinetic_background import maxwellians
from struphy.fields_background.mhd_equil.equils import set_defaults

from struphy.feec.psydac_derham import Derham
from struphy.geometry.base import Domain
from struphy.fields_background.mhd_equil.base import MHDequilibrium
from struphy.fields_background.braginskii_equil.base import BraginskiiEquilibrium


class Particles6D(Particles):
    """
    A class for initializing particles in models that use the full 6D phase space.

    The numpy marker array is as follows:

    ===== ============== ======================= ======= ====== ====== ==========
    index  | 0 | 1 | 2 | | 3 | 4 | 5           |  6       7       8    >=9
    ===== ============== ======================= ======= ====== ====== ==========
    value position (eta)    velocities           weight   s0     w0    buffer
    ===== ============== ======================= ======= ====== ====== ==========

    Parameters
    ----------
    name : str
        Name of the particle species.

    **params : dict
        Parameters for markers, see :class:`~struphy.pic.base.Particles`.
    """

    @classmethod
    def default_bckgr_params(cls):
        return {'type': 'Maxwellian3D',
                'Maxwellian3D': {}}

    def __init__(self,
                 name: str,
                 derham: Derham,
                 *,
                 domain: Domain = None,
                 mhd_equil: MHDequilibrium = None,
                 braginskii_equil: BraginskiiEquilibrium = None,
                 bckgr_params: dict = None,
                 pert_params: dict = None,
                 **marker_params):

        if bckgr_params is None:
            bckgr_params = self.default_bckgr_params()

        super().__init__(name,
                         derham,
                         domain=domain,
                         mhd_equil=mhd_equil,
                         braginskii_equil=braginskii_equil,
                         bckgr_params=bckgr_params,
                         pert_params=pert_params,
                         **marker_params)

    @property
    def n_cols(self):
        """ Number of the columns at each markers.
        """
        return 23

    @property
    def vdim(self):
        """ Dimension of the velocity space.
        """
        return 3

    @property
    def bufferindex(self):
        """Starting buffer marker index number
        """
        return 9

    @property
    def coords(self):
        """ Coordinates of the Particles6D, :math:`(v_1, v_2, v_3)`.
        """
        return 'cartesian'

    def svol(self, eta1, eta2, eta3, *v):
        """ Sampling density function as volume form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The volume-form sampling density.
        -------
        """
        # load sampling density svol (normalized to 1 in logical space)
        maxw_params = {'n': 1.,
                       'u1': self.marker_params['loading']['moments'][0],
                       'u2': self.marker_params['loading']['moments'][1],
                       'u3': self.marker_params['loading']['moments'][2],
                       'vth1': self.marker_params['loading']['moments'][3],
                       'vth2': self.marker_params['loading']['moments'][4],
                       'vth3': self.marker_params['loading']['moments'][5]}

        fun = maxwellians.Maxwellian3D(maxw_params=maxw_params)

        if self.spatial == 'uniform':
            return fun(eta1, eta2, eta3, *v)

        elif self.spatial == 'disc':
            return fun(eta1, eta2, eta3, *v)*2*eta1

        else:
            raise NotImplementedError(
                f'Spatial drawing must be "uniform" or "disc", is {self._spatial}.')

    def s0(self, eta1, eta2, eta3, *v, remove_holes=True):
        """ Sampling density function as 0 form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        remove_holes : bool
            If True, holes are removed from the returned array. If False, holes are evaluated to -1.

        Returns
        -------
        out : array-like
            The 0-form sampling density.
        -------
        """

        return self.domain.transform(self.svol(eta1, eta2, eta3, *v), self.markers, kind='3_to_0', remove_outside=remove_holes)


class Particles5D(Particles):
    """
    A class for initializing particles in guiding-center, drift-kinetic or gyro-kinetic models that use the 5D phase space.

    The numpy marker array is as follows:

    ===== ============== ========== ====== ======= ====== ====== ====== ============ ================= ==========
    index  | 0 | 1 | 2 |     3        4       5      6      7      8          9             10            >=11
    ===== ============== ========== ====== ======= ====== ====== ====== ============ ================= ==========
    value position (eta) v_parallel v_perp  weight   s0     w0   energy magn. moment toro. can. moment buffer
    ===== ============== ========== ====== ======= ====== ====== ====== ============ ================= ==========   

    Parameters
    ----------
    name : str
        Name of the particle species.

    **params : dict
        Parameters for markers, see :class:`~struphy.pic.base.Particles`.
    """

    @classmethod
    def default_bckgr_params(cls):
        return {'type': 'GyroMaxwellian2D',
                'GyroMaxwellian2D': {}}

    def __init__(self,
                 name: str,
                 derham: Derham,
                 *,
                 domain: Domain = None,
                 mhd_equil: MHDequilibrium = None,
                 braginskii_equil: BraginskiiEquilibrium = None,
                 bckgr_params: dict = None,
                 pert_params: dict = None,
                 **marker_params):

        if bckgr_params is None:
            bckgr_params = self.default_bckgr_params()

        super().__init__(name,
                         derham,
                         domain=domain,
                         mhd_equil=mhd_equil,
                         braginskii_equil=braginskii_equil,
                         bckgr_params=bckgr_params,
                         pert_params=pert_params,
                         **marker_params)

        # magnetic background
        if self.mhd_equil is not None:
            self._magn_bckgr = self.mhd_equil
        else:
            self._magn_bckgr = self.braginskii_equil

        self._absB0_h = self.derham.P['0'](self.magn_bckgr.absB0)

        self._unit_b1_h = self.derham.P['1']([self.magn_bckgr.unit_b1_1,
                                              self.magn_bckgr.unit_b1_2,
                                              self.magn_bckgr.unit_b1_3])

        E0T = self.derham.extraction_ops['0'].transpose()
        E1T = self.derham.extraction_ops['1'].transpose()
        self._absB0_h = E0T.dot(self._absB0_h)
        self._unit_b1_h = E1T.dot(self._unit_b1_h)

        self._tmp2 = self.derham.Vh['2'].zeros()

    @property
    def n_cols(self):
        """Number of columns in markers array, i.e. the attributes of each marker.
        """
        return 31

    @property
    def vdim(self):
        """Dimension of the velocity space.
        """
        return 2

    @property
    def bufferindex(self):
        """Starting buffer marker index number
        """
        return 11

    @property
    def magn_bckgr(self):
        """ Either mhd_equil or braginskii_equil.
        """
        return self._magn_bckgr

    @property
    def absB0_h(self):
        '''Discrete 0-form coefficients of |B_0|.'''
        return self._absB0_h

    @property
    def unit_b1_h(self):
        '''Discrete 1-form coefficients of B/|B|.'''
        return self._unit_b1_h

    @property
    def coords(self):
        """ Coordinates of the Particles5D, :math:`(v_\parallel, \mu)`.
        """
        return 'vpara_mu'

    def svol(self, eta1, eta2, eta3, *v):
        """ 
        Sampling density function as volume-form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The volume-form sampling density.
        -------
        """
        # load sampling density svol (normalized to 1 in logical space)
        maxw_params = {'n': 1.,
                       'u_para': self.marker_params['loading']['moments'][0],
                       'u_perp': self.marker_params['loading']['moments'][1],
                       'vth_para': self.marker_params['loading']['moments'][2],
                       'vth_perp': self.marker_params['loading']['moments'][3]}

        self._svol = maxwellians.GyroMaxwellian2D(
            maxw_params=maxw_params, volume_form=True, mhd_equil=self._magn_bckgr)

        if self.spatial == 'uniform':
            out = self._svol(eta1, eta2, eta3, *v)

        elif self.spatial == 'disc':
            out = 2 * eta1 * self._svol(eta1, eta2, eta3, *v)

        else:
            raise NotImplementedError(
                f'Spatial drawing must be "uniform" or "disc", is {self._spatial}.')

        return out

    def s3(self, eta1, eta2, eta3, *v):
        """
        Sampling density function as 3-form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The 3-form sampling density.
        -------
        """

        return self.svol(eta1, eta2, eta3, *v)/self._svol.velocity_jacobian_det(eta1, eta2, eta3, *v)

    def s0(self, eta1, eta2, eta3, *v, remove_holes=True):
        """ 
        Sampling density function as 0-form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        v_parallel, v_perp : array_like
            Velocity evaluation points.

        remove_holes : bool
            If True, holes are removed from the returned array. If False, holes are evaluated to -1.

        Returns
        -------
        out : array-like
            The 0-form sampling density.
        -------
        """

        return self.domain.transform(self.s3(eta1, eta2, eta3, *v), self.markers, kind='3_to_0', remove_outside=remove_holes)

    def save_constants_of_motion(self, epsilon, abs_B0=None, f_coords='constants_of_motion', initial=False):
        """
        Calculate each markers' constants of motion and assign them into markers[:,8:11].
        Only equilibrium magnetic field is considered.

        Parameters
        ----------
        epsilon : float
            Guiding center scaling factor.

        abs_B0 : BlockVector
            FE coeffs of equilibrium magnetic field magnitude.

        f_coords : str
            Coordinates of the distribution function f0.

        initial : bool
            If True, magnetic moment is also calculated and saved.
        """

        if abs_B0 is None:
            abs_B0 = self.derham.P['0'](self.mhd_equil.absB0)

        E0T = self.derham.extraction_ops['0'].transpose()
        abs_B0 = E0T.dot(abs_B0)

        if initial:
            utilities_kernels.eval_magnetic_moment_5d(self.markers,
                                                      self.derham.args_derham,
                                                      abs_B0._data)

        utilities_kernels.eval_energy_5d(self.markers,
                                         self.derham.args_derham,
                                         abs_B0._data)

        if f_coords == 'constants_of_motion':

            # eval psi at etas
            a1 = self.mhd_equil.domain.params_map['a1']
            R0 = self.mhd_equil.params['R0']
            B0 = self.mhd_equil.params['B0']

            r = self.markers[~self.holes, 0]*(1 - a1) + a1
            self.markers[~self.holes, 10] = self.mhd_equil.psi_r(r)

            utilities_kernels.eval_canonical_toroidal_moment_5d(self.markers,
                                                                self.derham.args_derham,
                                                                epsilon,
                                                                B0,
                                                                R0,
                                                                abs_B0._data)

    def save_magnetic_energy(self, b2):
        r"""
        Calculate magnetic field energy at each particles' position and assign it into markers[:,8].

        Parameters
        ----------

        b2 : BlockVector
            Finite element coefficients of the time-dependent magnetic field.
        """

        E2T = self.derham.extraction_ops['2'].transpose()
        b2t = E2T.dot(b2, out=self._tmp2)
        b2t.update_ghost_regions()

        utilities_kernels.eval_magnetic_energy(self.markers,
                                               self.derham.args_derham,
                                               self.domain.args_domain,
                                               self.absB0_h._data,
                                               self.unit_b1_h[0]._data,
                                               self.unit_b1_h[1]._data,
                                               self.unit_b1_h[2]._data,
                                               b2t[0]._data,
                                               b2t[1]._data,
                                               b2t[2]._data)

    def save_magnetic_background_energy(self):
        r"""
        Evaluate :math:`mu_p |B_0(\boldsymbol \eta_p)|` for each marker.
        The result is stored at markers[:, 8].
        """

        E0T = self.derham.extraction_ops['0'].transpose()

        abs_B0 = E0T.dot(self.absB0_h)
        abs_B0.update_ghost_regions()

        utilities_kernels.eval_magnetic_background_energy(self.markers,
                                                          self.derham.args_derham,
                                                          self.domain.args_domain,
                                                          abs_B0._data)


class Particles3D(Particles):
    """
    A class for initializing particles in 3D configuration space.

    The numpy marker array is as follows:

    ===== ============== ====== ====== ====== ======  
    index  | 0 | 1 | 2 |   3       4     5      >=6       
    ===== ============== ====== ====== ====== ======  
    value position (eta) weight   s0     w0   buffer    
    ===== ============== ====== ====== ====== ======   

    See :class:`~struphy.pic.base.Particles` for more info on parameters.
    """

    @classmethod
    def default_bckgr_params(cls):
        return {'type': 'Constant',
                'Constant': {}}

    def __init__(self,
                 name: str,
                 derham: Derham,
                 *,
                 domain: Domain = None,
                 mhd_equil: MHDequilibrium = None,
                 braginskii_equil: BraginskiiEquilibrium = None,
                 bckgr_params: dict = None,
                 pert_params: dict = None,
                 **marker_params):

        if bckgr_params is None:
            bckgr_params = self.default_bckgr_params()

        super().__init__(name,
                         derham,
                         domain=domain,
                         mhd_equil=mhd_equil,
                         braginskii_equil=braginskii_equil,
                         bckgr_params=bckgr_params,
                         pert_params=pert_params,
                         **marker_params)

    @property
    def n_cols(self):
        """ Number of the columns at each markers.
        """
        return 17

    @property
    def vdim(self):
        """ Dimension of the velocity space.
        """
        return 0

    @property
    def bufferindex(self):
        """Starting buffer marker index number
        """
        return 6

    @property
    def coords(self):
        """ Coordinates of the Particles3D.
        """
        return 'cartesian'

    def svol(self, eta1, eta2, eta3):
        """ Sampling density function as volume form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The volume-form sampling density.
        -------
        """

        if self.spatial == 'uniform':
            return 1. + 0.*eta1

        elif self.spatial == 'disc':
            return 2.*eta1

        else:
            raise NotImplementedError(
                f'Spatial drawing must be "uniform" or "disc", is {self._spatial}.')

    def s0(self, eta1, eta2, eta3, remove_holes=True):
        """ Sampling density function as 0 form.

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        remove_holes : bool
            If True, holes are removed from the returned array. If False, holes are evaluated to -1.

        Returns
        -------
        out : array-like
            The 0-form sampling density.
        -------
        """
        return self.domain.transform(self.svol(eta1, eta2, eta3), self.markers, kind='3_to_0', remove_outside=remove_holes)
