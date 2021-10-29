# import logging
from gvec_to_python.util.logger import logger
# logger = logging.getLogger(__name__)

import warnings

import numpy as np

from gvec_to_python.base.sbase import sBase
from gvec_to_python.base.fbase import fBase



class Base:
    """Evaluate the values of a B-spline x Fourier basis at logical coordinates."""

    def __init__(self, Nel, el_b, p, bc, NFP, modes, sin_cos, excl_mn_zero, mn, mn_max, modes_sin, modes_cos, range_sin, range_cos, use_cache=False):

        logger.debug('Base.__init__()')

        # sBase parameters.
        self.Nel  = Nel   # Number of elements.
        self.el_b = el_b  # Element boundaries.
        self.p    = p     # Spline degree.
        self.bc   = bc    # Periodic boundary conditions (use 'False' if clamped).

        # fBase parameters.
        self.NFP          = NFP          # Number of field periods (symmetry).
        self.modes        = modes        # Number of all m-n mode combinations.
        self.sin_cos      = sin_cos      # Whether the data has only sine, only cosine, or both sine and cosine basis. 1=sin, 2=cos, 3=sin&cos.
        self.excl_mn_zero = excl_mn_zero # 
        self.mn           = mn           # mn-mode number, with NFP premultiplied into the n-modes.
        self.mn_max       = mn_max       # Maximum m-mode and n-mode numbers, without NFP.
        self.modes_sin    = modes_sin    # Number of sine modes.
        self.modes_cos    = modes_cos    # Number of cosine modes.
        self.range_sin    = range_sin    # Index range of sine modes in `mn` list.
        self.range_cos    = range_cos    # Index range of cosine modes in `mn` list.

        self.sbase = sBase(Nel, el_b, p, bc)
        self.fbase = fBase(NFP, modes, sin_cos, excl_mn_zero, mn, mn_max, modes_sin, modes_cos, range_sin, range_cos)

        # Cache evaluations in memory.
        self.use_cache = use_cache
        self.cache = {
            'eval_stz': {},
            'eval_suv': {},
            'eval_stz_ds': {},
            'eval_stz_dtheta': {},
            'eval_stz_dzeta': {},
            'eval_suv_ds': {},
            'eval_suv_du': {},
            'eval_suv_dv': {},
        }
        if self.use_cache:
            warnings.warn("PerformanceWarning: Hashing `coef` will take far longer than the actual evaluation! Do not enable cache in this class!")

    def eval_stz(self, s, theta, zeta, coef):
        """Evaluate the value of a B-spline x Fourier basis at `s`, `theta`, `zeta`.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logical coordinate in radial direction.

        theta : float or meshgrid numpy.ndarray
            Angle theta in Tokamak coordinate along poloidal direction.

        zeta : float or meshgrid numpy.ndarray
            Angle zeta in Tokamak coordinate along toroidal direction.

        coef : array_like
            A list of B-spline control points (coefficients) for each (m, n) Fourier mode.

        Returns
        -------
        float or meshgrid numpy.ndarray
            Evaluated coordinate.
        """

        # logger.debug('Base.eval_stz()')

        if self.use_cache:
            param_hash = hash(str(s) + str(theta) + str(zeta) + str(coef))
            if param_hash in self.cache['eval_stz']:
                return self.cache['eval_stz'][param_hash]

        if isinstance(s, np.ndarray):

            assert isinstance(    s, np.ndarray), '1st argument should be of type `np.ndarray`. Got {} instead.'.format(type(s))
            assert isinstance(theta, np.ndarray), '2nd argument should be of type `np.ndarray`. Got {} instead.'.format(type(theta))
            assert isinstance( zeta, np.ndarray), '3rd argument should be of type `np.ndarray`. Got {} instead.'.format(type(zeta))

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == theta.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, theta.ndim)
                assert s.ndim ==  zeta.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim,  zeta.ndim)
                s, theta, zeta = np.meshgrid(s, theta, zeta, indexing='ij', sparse=True)

            # Input coordinates are from a sparse meshgrid, with two of the dimensions == 1.
            if max(s.shape) == s.size:

                assert max(theta.shape) == theta.size, '`s` is a sparse meshgrid, but `theta` is not.'
                assert max( zeta.shape) ==  zeta.size, '`s` is a sparse meshgrid, but `zeta` is not.'
                point = np.zeros(shape=(max(s.shape), max(theta.shape), max(zeta.shape)))

            # Input coordinates are from a dense meshgrid.
            else:

                assert s.shape == theta.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 2nd argument is {}.'.format(s.shape, theta.shape)
                assert s.shape ==  zeta.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 3rd argument is {}.'.format(s.shape,  zeta.shape)
                point = np.zeros_like(s)

        # Input coordinates are of a single point.
        else:

            point = 0

        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        for idx, (m, n) in enumerate(self.mn):
            eval_Bspline_at_s    = self.sbase.eval_s_struphy(s, coef[idx])
            eval_Fourier_at_mtnz = self.fbase.eval_f(idx, m, theta, n, zeta)
            # If (s,u,v) are sparse meshgrids, they should be able to broadcast automatically into a full grid below:
            point = point + eval_Bspline_at_s * eval_Fourier_at_mtnz

        # logger.debug('Transformed coordinate is evaluated to be {}.'.format(point))

        if self.use_cache:
            self.cache['eval_stz'][param_hash] = point
        return point

    def eval_suv(self, s, u, v, coef):
        """Evaluate the value of a B-spline x Fourier basis at `s`, `u`, `v`.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logical coordinate in radial direction.

        u : float or meshgrid numpy.ndarray
            Logical coordinate along poloidal direction. u = theta/(2*PI).

        v : float or meshgrid numpy.ndarray
            Logical coordinate along toroidal direction. v = zeta/(2*PI).

        coef : array_like
            A list of B-spline control points (coefficients) for each (m, n) Fourier mode.

        Returns
        -------
        float or meshgrid numpy.ndarray
            Evaluated coordinate.
        """

        # logger.debug('Base.eval_suv()')

        if self.use_cache:
            param_hash = hash(str(s) + str(u) + str(v) + str(coef))
            if param_hash in self.cache['eval_suv']:
                return self.cache['eval_suv'][param_hash]

        if isinstance(s, np.ndarray):

            assert isinstance(s, np.ndarray), '1st argument should be of type `np.ndarray`. Got {} instead.'.format(type(s))
            assert isinstance(u, np.ndarray), '2nd argument should be of type `np.ndarray`. Got {} instead.'.format(type(u))
            assert isinstance(v, np.ndarray), '3rd argument should be of type `np.ndarray`. Got {} instead.'.format(type(v))

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == u.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, u.ndim)
                assert s.ndim == v.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, v.ndim)
                s, u, v = np.meshgrid(s, u, v, indexing='ij', sparse=True)

            # Input coordinates are from a sparse meshgrid, with two of the dimensions == 1.
            if max(s.shape) == s.size:

                assert max(u.shape) == u.size, '`s` is a sparse meshgrid, but `u` is not.'
                assert max(v.shape) == v.size, '`s` is a sparse meshgrid, but `v` is not.'
                point = np.zeros(shape=(max(s.shape), max(u.shape), max(v.shape)))

            # Input coordinates are from a dense meshgrid.
            else:

                assert s.shape == u.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 2nd argument is {}.'.format(s.shape, u.shape)
                assert s.shape == v.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 3rd argument is {}.'.format(s.shape, v.shape)
                point = np.zeros_like(s)

        # Input coordinates are of a single point.
        else:

            point = 0

        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        for idx, (m, n) in enumerate(self.mn):
            eval_Bspline_at_s    = self.sbase.eval_s_struphy(s, coef[idx])
            eval_Fourier_at_munv = self.fbase.eval_f(idx, m, 2 * np.pi * u, n, 2 * np.pi * v)
            # If (s,u,v) are sparse meshgrids, they should be able to broadcast automatically into a full grid below:
            point = point + eval_Bspline_at_s * eval_Fourier_at_munv

        # logger.debug('Transformed coordinate is evaluated to be {}.'.format(point))

        if self.use_cache:
            self.cache['eval_suv'][param_hash] = point
        return point



    def eval_profile(self, s, profile_coef):
        """Evaluate profiles from coefficients at Greville points.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logical coordinate in radial direction.

        profile_coef : array_like
            A list of B-spline control points (coefficients) of the profile evaluated at Greville points.
            e.g. Coefficients of `phi`, `chi`, `iota`, `pres`, `spos`, found in GVEC data's `profiles` field.

        Returns
        -------
        float or meshgrid numpy.ndarray
            Evaluated value of the given profile coefficients at s-coordinate.
        """

        interpolate_profile_at_s = self.sbase.eval_s_struphy(s, profile_coef)
        return interpolate_profile_at_s

    def eval_dprofile(self, s, profile_coef):
        """Evaluate first s-derivative of profiles from coefficients at Greville points.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logical coordinate in radial direction.

        profile_coef : array_like
            A list of B-spline control points (coefficients) of the profile evaluated at Greville points.
            e.g. Coefficients of `phi`, `chi`, `iota`, `pres`, `spos`, found in GVEC data's `profiles` field.

        Returns
        -------
        float or meshgrid numpy.ndarray
            Evaluated first derivative of the given profile coefficients at s-coordinate.
        """

        interpolate_dprofile_ds_at_s = self.sbase.eval_s_1st_der_struphy(s, profile_coef)
        return interpolate_dprofile_ds_at_s



    def eval_stz_ds(self, s, theta, zeta, coef):
        """Evaluate partial derivative w.r.t. `s`, of a B-spline x Fourier basis.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logical coordinate in radial direction.

        theta : float or meshgrid numpy.ndarray
            Angle theta in Tokamak coordinate along poloidal direction.

        zeta : float or meshgrid numpy.ndarray
            Angle zeta in Tokamak coordinate along toroidal direction.

        coef : array_like
            A list of B-spline control points (coefficients) for each (m, n) Fourier mode.

        Returns
        -------
        float or meshgrid numpy.ndarray
            Partial derivative w.r.t. `s`.
        """

        # logger.debug('Base.eval_stz_ds()')

        if isinstance(s, np.ndarray):

            assert isinstance(    s, np.ndarray), '1st argument should be of type `np.ndarray`. Got {} instead.'.format(type(s))
            assert isinstance(theta, np.ndarray), '2nd argument should be of type `np.ndarray`. Got {} instead.'.format(type(theta))
            assert isinstance( zeta, np.ndarray), '3rd argument should be of type `np.ndarray`. Got {} instead.'.format(type(zeta))

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == theta.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, theta.ndim)
                assert s.ndim ==  zeta.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim,  zeta.ndim)
                s, theta, zeta = np.meshgrid(s, theta, zeta, indexing='ij', sparse=True)

            # Input coordinates are from a sparse meshgrid, with two of the dimensions == 1.
            if max(s.shape) == s.size:

                assert max(theta.shape) == theta.size, '`s` is a sparse meshgrid, but `theta` is not.'
                assert max( zeta.shape) ==  zeta.size, '`s` is a sparse meshgrid, but `zeta` is not.'
                derivative = np.zeros(shape=(max(s.shape), max(theta.shape), max(zeta.shape)))

            # Input coordinates are from a dense meshgrid.
            else:

                assert s.shape == theta.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 2nd argument is {}.'.format(s.shape, theta.shape)
                assert s.shape ==  zeta.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 3rd argument is {}.'.format(s.shape,  zeta.shape)
                derivative = np.zeros_like(s)

        # Input coordinates are of a single point.
        else:

            derivative = 0

        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        for idx, (m, n) in enumerate(self.mn):
            eval_Bspline_at_s    = self.sbase.eval_s_1st_der_struphy(s, coef[idx])
            eval_Fourier_at_mtnz = self.fbase.eval_f(idx, m, theta, n, zeta)
            derivative = derivative + eval_Bspline_at_s * eval_Fourier_at_mtnz

        # logger.debug('Derivative is evaluated to be {}.'.format(derivative))

        return derivative

    def eval_stz_dtheta(self, s, theta, zeta, coef):
        """Evaluate partial derivative w.r.t. `theta`, of a B-spline x Fourier basis.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logical coordinate in radial direction.

        theta : float or meshgrid numpy.ndarray
            Angle theta in Tokamak coordinate along poloidal direction.

        zeta : float or meshgrid numpy.ndarray
            Angle zeta in Tokamak coordinate along toroidal direction.

        coef : array_like
            A list of B-spline control points (coefficients) for each (m, n) Fourier mode.

        Returns
        -------
        float or meshgrid numpy.ndarray
            Partial derivative w.r.t. `theta`.
        """

        # logger.debug('Base.eval_stz_dtheta()')

        if isinstance(s, np.ndarray):

            assert isinstance(    s, np.ndarray), '1st argument should be of type `np.ndarray`. Got {} instead.'.format(type(s))
            assert isinstance(theta, np.ndarray), '2nd argument should be of type `np.ndarray`. Got {} instead.'.format(type(theta))
            assert isinstance( zeta, np.ndarray), '3rd argument should be of type `np.ndarray`. Got {} instead.'.format(type(zeta))

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == theta.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, theta.ndim)
                assert s.ndim ==  zeta.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim,  zeta.ndim)
                s, theta, zeta = np.meshgrid(s, theta, zeta, indexing='ij', sparse=True)

            # Input coordinates are from a sparse meshgrid, with two of the dimensions == 1.
            if max(s.shape) == s.size:

                assert max(theta.shape) == theta.size, '`s` is a sparse meshgrid, but `theta` is not.'
                assert max( zeta.shape) ==  zeta.size, '`s` is a sparse meshgrid, but `zeta` is not.'
                derivative = np.zeros(shape=(max(s.shape), max(theta.shape), max(zeta.shape)))

            # Input coordinates are from a dense meshgrid.
            else:

                assert s.shape == theta.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 2nd argument is {}.'.format(s.shape, theta.shape)
                assert s.shape ==  zeta.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 3rd argument is {}.'.format(s.shape,  zeta.shape)
                derivative = np.zeros_like(s)

        # Input coordinates are of a single point.
        else:

            derivative = 0

        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        for idx, (m, n) in enumerate(self.mn):
            eval_Bspline_at_s    = self.sbase.eval_s_struphy(s, coef[idx])
            eval_Fourier_at_mtnz = self.fbase.eval_f_dtheta(idx, m, theta, n, zeta)
            derivative = derivative + eval_Bspline_at_s * eval_Fourier_at_mtnz

        # logger.debug('Derivative is evaluated to be {}.'.format(derivative))

        return derivative

    def eval_stz_dzeta(self, s, theta, zeta, coef):
        """Evaluate partial derivative w.r.t. `zeta`, of a B-spline x Fourier basis.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logical coordinate in radial direction.

        theta : float or meshgrid numpy.ndarray
            Angle theta in Tokamak coordinate along poloidal direction.

        zeta : float or meshgrid numpy.ndarray
            Angle zeta in Tokamak coordinate along toroidal direction.

        coef : array_like
            A list of B-spline control points (coefficients) for each (m, n) Fourier mode.

        Returns
        -------
        float or meshgrid numpy.ndarray
            Partial derivative w.r.t. `zeta`.
        """

        # logger.debug('Base.eval_stz_dzeta()')

        if isinstance(s, np.ndarray):

            assert isinstance(    s, np.ndarray), '1st argument should be of type `np.ndarray`. Got {} instead.'.format(type(s))
            assert isinstance(theta, np.ndarray), '2nd argument should be of type `np.ndarray`. Got {} instead.'.format(type(theta))
            assert isinstance( zeta, np.ndarray), '3rd argument should be of type `np.ndarray`. Got {} instead.'.format(type(zeta))

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == theta.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, theta.ndim)
                assert s.ndim ==  zeta.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim,  zeta.ndim)
                s, theta, zeta = np.meshgrid(s, theta, zeta, indexing='ij', sparse=True)

            # Input coordinates are from a sparse meshgrid, with two of the dimensions == 1.
            if max(s.shape) == s.size:

                assert max(theta.shape) == theta.size, '`s` is a sparse meshgrid, but `theta` is not.'
                assert max( zeta.shape) ==  zeta.size, '`s` is a sparse meshgrid, but `zeta` is not.'
                derivative = np.zeros(shape=(max(s.shape), max(theta.shape), max(zeta.shape)))

            # Input coordinates are from a dense meshgrid.
            else:

                assert s.shape == theta.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 2nd argument is {}.'.format(s.shape, theta.shape)
                assert s.shape ==  zeta.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 3rd argument is {}.'.format(s.shape,  zeta.shape)
                derivative = np.zeros_like(s)

        # Input coordinates are of a single point.
        else:

            derivative = 0

        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        for idx, (m, n) in enumerate(self.mn):
            eval_Bspline_at_s    = self.sbase.eval_s_struphy(s, coef[idx])
            eval_Fourier_at_mtnz = self.fbase.eval_f_dzeta(idx, m, theta, n, zeta)
            derivative = derivative + eval_Bspline_at_s * eval_Fourier_at_mtnz

        # logger.debug('Derivative is evaluated to be {}.'.format(derivative))

        return derivative



    def eval_suv_ds(self, s, u, v, coef):
        """Evaluate partial derivative w.r.t. `s`, of a B-spline x Fourier basis.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logical coordinate in radial direction.

        u : float or meshgrid numpy.ndarray
            Logical coordinate along poloidal direction. u = theta/(2*PI).

        v : float or meshgrid numpy.ndarray
            Logical coordinate along toroidal direction. v = zeta/(2*PI).

        coef : array_like
            A list of B-spline control points (coefficients) for each (m, n) Fourier mode.

        Returns
        -------
        float or meshgrid numpy.ndarray
            Partial derivative w.r.t. `s`.
        """

        # logger.debug('Base.eval_suv_ds()')

        if isinstance(s, np.ndarray):

            assert isinstance(s, np.ndarray), '1st argument should be of type `np.ndarray`. Got {} instead.'.format(type(s))
            assert isinstance(u, np.ndarray), '2nd argument should be of type `np.ndarray`. Got {} instead.'.format(type(u))
            assert isinstance(v, np.ndarray), '3rd argument should be of type `np.ndarray`. Got {} instead.'.format(type(v))

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == u.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, u.ndim)
                assert s.ndim == v.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, v.ndim)
                s, u, v = np.meshgrid(s, u, v, indexing='ij', sparse=True)

            # Input coordinates are from a sparse meshgrid, with two of the dimensions == 1.
            if max(s.shape) == s.size:

                assert max(u.shape) == u.size, '`s` is a sparse meshgrid, but `u` is not.'
                assert max(v.shape) == v.size, '`s` is a sparse meshgrid, but `v` is not.'
                derivative = np.zeros(shape=(max(s.shape), max(u.shape), max(v.shape)))

            # Input coordinates are from a dense meshgrid.
            else:

                assert s.shape == u.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 2nd argument is {}.'.format(s.shape, u.shape)
                assert s.shape == v.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 3rd argument is {}.'.format(s.shape, v.shape)
                derivative = np.zeros_like(s)

        # Input coordinates are of a single point.
        else:

            derivative = 0

        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        for idx, (m, n) in enumerate(self.mn):
            eval_Bspline_at_s    = self.sbase.eval_s_1st_der_struphy(s, coef[idx])
            eval_Fourier_at_munv = self.fbase.eval_f(idx, m, 2 * np.pi * u, n, 2 * np.pi * v)
            derivative = derivative + eval_Bspline_at_s * eval_Fourier_at_munv

        # logger.debug('Derivative is evaluated to be {}.'.format(derivative))

        return derivative

    def eval_suv_du(self, s, u, v, coef):
        """Evaluate partial derivative w.r.t. `u`, of a B-spline x Fourier basis.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logical coordinate in radial direction.

        u : float or meshgrid numpy.ndarray
            Logical coordinate along poloidal direction. u = theta/(2*PI).

        v : float or meshgrid numpy.ndarray
            Logical coordinate along toroidal direction. v = zeta/(2*PI).

        coef : array_like
            A list of B-spline control points (coefficients) for each (m, n) Fourier mode.

        Returns
        -------
        float or meshgrid numpy.ndarray
            Partial derivative w.r.t. `theta`.
        """

        # logger.debug('Base.eval_suv_dtheta()')

        if isinstance(s, np.ndarray):

            assert isinstance(s, np.ndarray), '1st argument should be of type `np.ndarray`. Got {} instead.'.format(type(s))
            assert isinstance(u, np.ndarray), '2nd argument should be of type `np.ndarray`. Got {} instead.'.format(type(u))
            assert isinstance(v, np.ndarray), '3rd argument should be of type `np.ndarray`. Got {} instead.'.format(type(v))

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == u.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, u.ndim)
                assert s.ndim == v.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, v.ndim)
                s, u, v = np.meshgrid(s, u, v, indexing='ij', sparse=True)

            # Input coordinates are from a sparse meshgrid, with two of the dimensions == 1.
            if max(s.shape) == s.size:

                assert max(u.shape) == u.size, '`s` is a sparse meshgrid, but `u` is not.'
                assert max(v.shape) == v.size, '`s` is a sparse meshgrid, but `v` is not.'
                derivative = np.zeros(shape=(max(s.shape), max(u.shape), max(v.shape)))

            # Input coordinates are from a dense meshgrid.
            else:

                assert s.shape == u.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 2nd argument is {}.'.format(s.shape, u.shape)
                assert s.shape == v.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 3rd argument is {}.'.format(s.shape, v.shape)
                derivative = np.zeros_like(s)

        # Input coordinates are of a single point.
        else:

            derivative = 0

        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        for idx, (m, n) in enumerate(self.mn):
            eval_Bspline_at_s    = self.sbase.eval_s_struphy(s, coef[idx])
            eval_Fourier_at_munv = self.fbase.eval_f_dtheta(idx, m, 2 * np.pi * u, n, 2 * np.pi * v) * 2 * np.pi
            derivative = derivative + eval_Bspline_at_s * eval_Fourier_at_munv

        # logger.debug('Derivative is evaluated to be {}.'.format(derivative))

        return derivative

    def eval_suv_dv(self, s, u, v, coef):
        """Evaluate partial derivative w.r.t. `v`, of a B-spline x Fourier basis.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logical coordinate in radial direction.

        u : float or meshgrid numpy.ndarray
            Logical coordinate along poloidal direction. u = theta/(2*PI).

        v : float or meshgrid numpy.ndarray
            Logical coordinate along toroidal direction. v = zeta/(2*PI).

        coef : array_like
            A list of B-spline control points (coefficients) for each (m, n) Fourier mode.

        Returns
        -------
        float or meshgrid numpy.ndarray
            Partial derivative w.r.t. `zeta`.
        """

        # logger.debug('Base.eval_suv_dzeta()')

        if isinstance(s, np.ndarray):

            assert isinstance(s, np.ndarray), '1st argument should be of type `np.ndarray`. Got {} instead.'.format(type(s))
            assert isinstance(u, np.ndarray), '2nd argument should be of type `np.ndarray`. Got {} instead.'.format(type(u))
            assert isinstance(v, np.ndarray), '3rd argument should be of type `np.ndarray`. Got {} instead.'.format(type(v))

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == u.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, u.ndim)
                assert s.ndim == v.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, v.ndim)
                s, u, v = np.meshgrid(s, u, v, indexing='ij', sparse=True)

            # Input coordinates are from a sparse meshgrid, with two of the dimensions == 1.
            if max(s.shape) == s.size:

                assert max(u.shape) == u.size, '`s` is a sparse meshgrid, but `u` is not.'
                assert max(v.shape) == v.size, '`s` is a sparse meshgrid, but `v` is not.'
                derivative = np.zeros(shape=(max(s.shape), max(u.shape), max(v.shape)))

            # Input coordinates are from a dense meshgrid.
            else:

                assert s.shape == u.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 2nd argument is {}.'.format(s.shape, u.shape)
                assert s.shape == v.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 3rd argument is {}.'.format(s.shape, v.shape)
                derivative = np.zeros_like(s)

        # Input coordinates are of a single point.
        else:

            derivative = 0

        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        for idx, (m, n) in enumerate(self.mn):
            eval_Bspline_at_s    = self.sbase.eval_s_struphy(s, coef[idx])
            eval_Fourier_at_munv = self.fbase.eval_f_dzeta(idx, m, 2 * np.pi * u, n, 2 * np.pi * v) * 2 * np.pi
            derivative = derivative + eval_Bspline_at_s * eval_Fourier_at_munv

        # logger.debug('Derivative is evaluated to be {}.'.format(derivative))

        return derivative
