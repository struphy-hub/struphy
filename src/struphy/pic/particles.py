import numpy as np
from struphy.pic.base import Particles
from struphy.pic import utilities_kernels
from struphy.kinetic_background import maxwellians
from struphy.fields_background.mhd_equil.equils import set_defaults


class Particles6D(Particles):
    """
    A class for initializing particles in models that use the full 6D phase space.

    The numpy marker array is as follows:

    ===== ============== ======================= ======= ====== ====== ==========
    index  | 0 | 1 | 2 | | 3 | 4 | 5           |  6       7       8    >=9
    ===== ============== ======================= ======= ====== ====== ==========
    value position (eta)    velocities           weight   s0     w0    additional
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
        return {'type': 'Maxwellian6D',
                'Maxwellian6D': {}}

    def __init__(self, name, **params):

        assert 'bckgr_params' in params
        if params['bckgr_params'] is None:
            params['bckgr_params'] = self.default_bckgr_params()

        super().__init__(name, **params)

    @property
    def n_cols(self):
        """ Number of the columns at each markers.
        """
        return 16

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
        maxwellian6D = getattr(maxwellians, 'Maxwellian6D')

        maxw_params = {'n': 1.,
                       'u1': self.marker_params['loading']['moments'][0],
                       'u2': self.marker_params['loading']['moments'][1],
                       'u3': self.marker_params['loading']['moments'][2],
                       'vth1': self.marker_params['loading']['moments'][3],
                       'vth2': self.marker_params['loading']['moments'][4],
                       'vth3': self.marker_params['loading']['moments'][5]}

        fun = maxwellian6D(maxw_params=maxw_params)

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

    ===== ============== ========== ====== ======= ====== ====== ====== ============ ================ ===========
    index  | 0 | 1 | 2 |     3        4       5      6      7      8          9             10            >=11
    ===== ============== ========== ====== ======= ====== ====== ====== ============ ================= ==========
    value position (eta) v_parallel v_perp  weight   s0     w0   energy magn. moment toro. can. moment additional
    ===== ============== ========== ====== ======= ====== ====== ====== ============ ================= ==========   

    Parameters
    ----------
    name : str
        Name of the particle species.

    **params : dict
        Parameters for markers, see :class:`~struphy.pic.base.Particles`.
    """

    @classmethod
    def default_bckgr_params():
        return {'type': 'Maxwellian5D',
                'Maxwellian5D': {}}

    def __init__(self, name, **params):

        assert 'bckgr_params' in params
        if params['bckgr_params'] is None:
            params['bckgr_params'] = self.default_bckgr_params()

        super().__init__(name, **params)

    @property
    def n_cols(self):
        """Number of the columns at each markers.
        """
        return 25

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
        maxwellian5D = getattr(maxwellians, 'Maxwellian5D')

        maxw_params = {'n': 1.,
                       'u_para': self.marker_params['loading']['moments'][0],
                       'u_perp': self.marker_params['loading']['moments'][1],
                       'vth_para': self.marker_params['loading']['moments'][2],
                       'vth_perp': self.marker_params['loading']['moments'][3]}

        self._svol = maxwellian5D(maxw_params=maxw_params,
                                  mhd_equil=self.mhd_equil)

        if self.spatial == 'uniform':
            return self._svol(eta1, eta2, eta3, *v)

        elif self.spatial == 'disc':
            return self._svol(eta1, eta2, eta3, *v)*2*eta1

        else:
            raise NotImplementedError(
                f'Spatial drawing must be "uniform" or "disc", is {self._spatial}.')

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

    def save_constants_of_motion(self, epsilon, abs_B0=None, initial=False):
        """
        Calculate each markers' constants of motion and assign them into markers[:,8:11].
        Only equilibrium magnetic field is considered.

        Parameters
        ----------
        epsilon : float
            Guiding center scaling factor.

        abs_B0 : BlockVector
            FE coeffs of equilibrium magnetic field magnitude.

        initial : bool
            If True, magnetic moment is also calculated and saved.
        """
        # fixed FEM arguments for the accumulator kernel
        args_fem = (np.array(self.derham.p),
                    self.derham.Vh_fem['0'].knots[0],
                    self.derham.Vh_fem['0'].knots[1],
                    self.derham.Vh_fem['0'].knots[2],
                    np.array(self.derham.Vh['0'].starts))

        if abs_B0 is None:
            abs_B0 = self.derham.P['0'](self.mhd_equil.absB0)

        E0T = self.derham.extraction_ops['0'].transpose()
        abs_B0 = E0T.dot(abs_B0)

        if initial:
            utilities_kernels.eval_magnetic_moment_5d(self.markers,
                                                      *args_fem,
                                                      abs_B0._data)

        utilities_kernels.eval_energy_5d(self.markers,
                                         *args_fem,
                                         abs_B0._data)

        # eval psi at etas
        a1 = self.mhd_equil.domain.params_map['a1']
        R0 = self.mhd_equil.params['R0']
        B0 = self.mhd_equil.params['B0']

        r = self.markers[~self.holes, 0]*(1 - a1) + a1
        self.markers[~self.holes, 10] = self.mhd_equil.psi_r(r)

        utilities_kernels.eval_canonical_toroidal_moment_5d(self.markers,
                                                            *args_fem,
                                                            epsilon, B0, R0,
                                                            abs_B0._data)

    def save_magnetic_energy(self, abs_B0, unit_b1, b):
        """
        Calculate magnetic field energy at each particles' position and assign it into markers[:,8].
        Non-equilibrium magnetic field can be included.

        Parameters
        ----------
        abs_B0 : BlockVector
            FE coeffs of equilibrium magnetic field magnitude.

        unit_b1 : BlockVector
            FE coeffs of 1-form unit equilibrium magnetic field.

        b : BlockVector
            FE coeffs of perturbed magnetic field.
        """

        # fixed FEM arguments for the accumulator kernel
        args_fem = (np.array(self.derham.p),
                    self.derham.Vh_fem['0'].knots[0],
                    self.derham.Vh_fem['0'].knots[1],
                    self.derham.Vh_fem['0'].knots[2],
                    np.array(self.derham.Vh['0'].starts))

        E0T = self.derham.extraction_ops['0'].transpose()
        E1T = self.derham.extraction_ops['1'].transpose()
        E2T = self.derham.extraction_ops['2'].transpose()

        abs_B0 = E0T.dot(abs_B0)
        unit_b1 = E1T.dot(unit_b1)
        b = E2T.dot(b)

        utilities_kernels.eval_magnetic_energy(self.markers,
                                               *args_fem, *self._domain.args_map,
                                               abs_B0._data,
                                               unit_b1[0]._data, unit_b1[1]._data, unit_b1[2]._data,
                                               b[0]._data, b[1]._data, b[2]._data)
