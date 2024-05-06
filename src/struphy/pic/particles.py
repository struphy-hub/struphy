import numpy as np
from struphy.pic.base import Particles
from struphy.pic.utilities_kernels import eval_magnetic_energy, eval_magnetic_moment_5d
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

    def velocity_jacobian_det(self, eta1, eta2, eta3, *v):
        """ Jacobian determinant of the velocity coordinate transformation.

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
        assert len(v) == self.vdim

        return 1. + 0*eta1

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
    value position (eta) v_parallel v_perp  weight   s0     w0   E_perp magn. moment toro. can. moment additional
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

    def velocity_jacobian_det(self, eta1, eta2, eta3, *v):
        """
        Jacobian determinant of the velocity coordinate transformation.

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
        assert len(v) == self.vdim

        # call equilibrium arrays
        etas = (np.vstack((eta1, eta2, eta3)).T).copy()
        bv = self.mhd_equil.bv(etas)
        unit_b1 = self.mhd_equil.unit_b1(etas)

        # B*_parallel = b0 . B*
        jacobian_det = np.einsum('ij,ij->j', unit_b1, bv)/np.abs(v[1])

        return jacobian_det

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

        fun = maxwellian5D(maxw_params=maxw_params)

        if self.spatial == 'uniform':
            return fun(eta1, eta2, eta3, *v)

        elif self.spatial == 'disc':
            return fun(eta1, eta2, eta3, *v)*2*eta1

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

        return self.svol(eta1, eta2, eta3, *v)/self.velocity_jacobian_det(eta1, eta2, eta3, *v)

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

    def save_magnetic_moment(self):
        r"""
        Calculate magnetic moment of each particles :math:`\mu = \frac{m v_\perp^2}{2B}` and asign it into markers[:,9].
        """
        T1, T2, T3 = self.derham.Vh_fem['0'].knots

        absB = self.derham.P['0'](self._mhd_equil.absB0)

        E0T = self.derham.extraction_ops['0'].transpose()

        absB = E0T.dot(absB)

        eval_magnetic_moment_5d(self.markers,
                                np.array(self.derham.p), T1, T2, T3,
                                np.array(self.derham.Vh['0'].starts),
                                absB._data)

    def save_magnetic_energy(self, abs_B0, unit_b1, b):
        r"""
        Calculate magnetic field energy at each particles' position and asign it into markers[:,8].
        """

        # fixed FEM arguments for the accumulator kernel
        self._args_fem = (np.array(self.derham.p),
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

        eval_magnetic_energy(self.markers,
                             *self._args_fem, *self._domain.args_map,
                             abs_B0._data,
                             unit_b1[0]._data, unit_b1[1]._data, unit_b1[2]._data,
                             b[0]._data, b[1]._data, b[2]._data)
