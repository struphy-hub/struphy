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
        Parameters for markers.
    """

    def __init__(self, name, **params):

        # base class params
        base_params = {}

        list_base_params = ['type', 'ppc', 'Np', 'eps',
                            'bc', 'loading', 'derham', 'domain',
                            'f0_params']

        for key, val in params.items():
            if key in list_base_params:
                base_params[key] = val

        super().__init__(name, **base_params)

    @property
    def n_cols(self):
        """Number of the columns at each markers.
        """
        return 16

    @property
    def vdim(self):
        """Dimension of the velocity space.
        """
        return 3

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

        return 1. + 0*eta1

    def svol(self, eta1, eta2, eta3, *v):
        """ 
        Sampling density function as volume form.

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
        # load sampling density svol = s6 = s3 (normalized to 1 in logical space!)
        Maxwellian6DUniform = getattr(maxwellians, 'Maxwellian6DUniform')

        s3 = Maxwellian6DUniform(n=1.,
                                 u1=self._params['loading']['moments'][0],
                                 u2=self._params['loading']['moments'][1],
                                 u3=self._params['loading']['moments'][2],
                                 vth1=self._params['loading']['moments'][3],
                                 vth2=self._params['loading']['moments'][4],
                                 vth3=self._params['loading']['moments'][5])

        return s3(eta1, eta2, eta3, *v)

    def s0(self, eta1, eta2, eta3, *v, remove_holes=True):
        """ 
        Sampling density function as 0 form.

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

    ===== ============== ========== ============= ======= ====== ====== ==========
    index  | 0 | 1 | 2 |  3            4            5       6       7    >=8
    ===== ============== ========== ============= ======= ====== ====== ==========
    value position (eta) v_parallel magn. moment  weight   s0     w0    additional
    ===== ============== ========== ============= ======= ====== ====== ==========        

    Parameters
    ----------
    name : str
        Name of the particle species.

    **params : dict
        Parameters for markers.
    """

    def __init__(self, name, **params):

        # base class params
        base_params = {}

        list_base_params = ['type', 'ppc', 'Np', 'eps',
                            'bc', 'loading', 'derham', 'domain']

        for key, val in params.items():
            if key in list_base_params:
                base_params[key] = val

        super().__init__(name, **base_params)

        # child class params
        child_params = {}

        list_child_params = ['mhd_equil', 'epsilon']

        for key, val in params.items():
            if key in list_child_params:
                child_params[key] = val

        params_default = {'mhd_equil': None,
                          'epsilon': 1.
                          }

        child_params = set_defaults(child_params, params_default)

        self._mhd_equil = params['mhd_equil']
        self._epsilon = child_params['epsilon']

    @property
    def n_cols(self):
        """Number of the columns at each markers.
        """
        return 23

    @property
    def vdim(self):
        """Dimension of the velocity space.
        """
        return 2
    
    @property
    def mhd_equil(self):
        """Class of MHD equilibrium
        """
        return self._mhd_equil
    
    @property
    def epsilon(self):
        """Epsilon unit, 1 / (cyclotron freq * time_unit)
        """
        return self._epsilon

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
        # load sampling density svol = s5 (normalized to 1 in logical space!)
        Maxwellian5DUniform = getattr(maxwellians, 'Maxwellian5DUniform')

        s5 = Maxwellian5DUniform(n=1.,
                                 u_parallel=self.params['loading']['moments'][0],
                                 u_perp=self.params['loading']['moments'][1],
                                 vth_parallel=self.params['loading']['moments'][2],
                                 vth_perp=self.params['loading']['moments'][3])

        return s5(eta1, eta2, eta3, *v)

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
        Calculate magnetic moment of each particles :math:`\mu = \frac{m v_\perp^2}{2B}` and asign it into markers[:,4].
        """
        T1, T2, T3 = self.derham.Vh_fem['0'].knots

        absB = self.derham.P['0'](self._mhd_equil.absB0)

        E0T = self.derham.extraction_ops['0'].transpose()

        absB = E0T.dot(absB)

        eval_magnetic_moment_5d(self.markers,
                                np.array(self.derham.p), T1, T2, T3,
                                np.array(self.derham.Vh['0'].starts),
                                absB._data)

    def save_magnetic_energy(self, PB):
        r"""
        Calculate magnetic field energy at each particles' position and asign it into markers[:,8].
        """
        T1, T2, T3 = self.derham.Vh_fem['0'].knots

        E0T = self.derham.extraction_ops['0'].transpose()

        PB = E0T.dot(PB)

        eval_magnetic_energy(self.markers,
                             np.array(self.derham.p), T1, T2, T3,
                             np.array(self.derham.Vh['0'].starts),
                             PB._data)
