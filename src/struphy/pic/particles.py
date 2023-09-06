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
                            'bc', 'loading', 'derham', 'domain']

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

        list_child_params = ['A', 'Z', 'mhd_equil', 'units_basic']

        for key, val in params.items():
            if key in list_child_params:
                child_params[key] = val

        params_default = {'A': 1,
                          'Z': 1,
                          'mhd_equil': None,
                          'units_basic': None
                          }

        child_params = set_defaults(child_params, params_default)

        self._mhd_equil = params['mhd_equil']

        # compute kappa
        ee = 1.602176634e-19  # elementary charge (C)
        mH = 1.67262192369e-27  # proton mass (kg)

        Ah = child_params['A']
        Zh = child_params['Z']

        omega_ch = (Zh*ee*child_params['units_basic']['B'])/(Ah*mH)
        self._kappa = omega_ch*child_params['units_basic']['t']

    @property
    def n_cols(self):
        """Number of the columns at each markers.
        """
        return 26

    @property
    def vdim(self):
        """Dimension of the velocity space.
        """
        return 2

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
        # call equilibrium arrays
        etas = (np.vstack((eta1, eta2, eta3)).T).copy()
        bv = self._mhd_equil.bv(etas)
        curlb = self._mhd_equil.jv(etas)/self._mhd_equil.absB0(etas)
        unit_b1 = self._mhd_equil.unit_b1(etas)

        # contra-variant components of B* = B + 1/kappa*v_parallel*curlb0
        bstar = bv + 1/self._kappa*v[0]*curlb

        # B*_parallel = b0 . B*
        jacobian_det = np.einsum('ij,ij->j', unit_b1, bstar)/v[1]

        return self.svol(eta1, eta2, eta3, *v)/np.abs(jacobian_det)

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

    def save_magnetic_moment(self, derham):
        r"""
        Calculate magnetic moment of each particles :math:`\mu = \frac{m v_\perp^2}{2B}` and asign it into markers[:,4].
        """
        T1, T2, T3 = derham.Vh_fem['0'].knots

        absB = derham.P['0'](self._mhd_equil.absB0)

        E0T = derham.E['0'].transpose()

        absB = E0T.dot(absB)

        eval_magnetic_moment_5d(self._markers,
                                np.array(derham.p), T1, T2, T3,
                                np.array(derham.Vh['0'].starts),
                                absB._data)

    def save_magnetic_energy(self, derham, PB):
        r"""
        Calculate magnetic field energy at each particles' position and asign it into markers[:,8].
        """
        T1, T2, T3 = derham.Vh_fem['0'].knots

        eval_magnetic_energy(self._markers,
                             np.array(derham.p), T1, T2, T3,
                             np.array(derham.Vh['0'].starts),
                             PB._data)
