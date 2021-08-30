import numpy as np
from numba import jit, njit

from gvec_to_python.hmap.base_map import BaseMap

class suv_to_xyz(BaseMap):

    def __init__(self, X1_base, X1_coef, X2_base, X2_coef, LA_base, LA_coef, component='all', use_cache=True):
        """Map (s, u, v) coordinate to Cartesian (x, y, z).

        Parameters
        ----------
        component : str
            Default: 'all'. Specify whether to return a vector (tuple) of all Cartesian coordinates simultaneously evaluated {'all'}, or only return a scalar of a specific Cartesian axis {'x', 'y', 'z'}.
        """
        (self.X1_base, self.X1_coef, self.X2_base, self.X2_coef, self.LA_base, self.LA_coef) = (X1_base, X1_coef, X2_base, X2_coef, LA_base, LA_coef)

        if component == 'all':
            self.mapto = self.mapto_all
        elif component.lower() == 'x':
            self.mapto = self.mapto_x
        elif component.lower() == 'y':
            self.mapto = self.mapto_y
        elif component.lower() == 'z':
            self.mapto = self.mapto_z
        else:
            pass # Error.

        # Cache evaluations in memory.
        # Roughly 4x speed up.
        self.use_cache = use_cache
        self.cache = {
            'mapto_all': {},
            'df': {},
        }

        self.f = self.mapto
        self.F = self.mapto
        self.J  = self.df
        self.dF = self.df
        self.DF = self.df
        self.J_det  = self.df_det
        self.dF_det = self.df_det
        self.DF_det = self.df_det
        self.J_inv  = self.df_inv
        self.dF_inv = self.df_inv
        self.DF_inv = self.df_inv



    def mapto_all(self, s, u, v):
        """Map (s, u, v) coordinate to Cartesian (x, y, z).

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        u : float or meshgrid numpy.ndarray
            Logical coordinate along poloidal direction. u = theta/(2*PI).

        v : float or meshgrid numpy.ndarray
            Logical coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        3-tuple or 3-tuple of meshgrid numpy.ndarray
            Transformed coordinates."""

        if self.use_cache:
            param_hash = hash(str(s) + str(u) + str(v))
            if param_hash in self.cache['mapto_all']:
                return self.cache['mapto_all'][param_hash]

        if isinstance(s, np.ndarray):

            suv_to_xyz.assert_array_input(s, u, v)

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == u.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, u.ndim)
                assert s.ndim == v.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, v.ndim)
                s, u, v = np.meshgrid(s, u, v, indexing='ij', sparse=True)

        X1 = self.X1_base.eval_suv(s, u, v, self.X1_coef)
        X2 = self.X2_base.eval_suv(s, u, v, self.X2_coef)
        # LA = self.LA_base.eval_suv(s, u, v, self.LA_coef) # Not necessary to compute lambda.

        # x =   X1 * np.cos(2 * np.pi * v)
        # y = - X1 * np.sin(2 * np.pi * v)
        # z =   X2

        F = np.array(suv_to_xyz.X1X2v_to_xyz(X1, X2, v))
        if self.use_cache:
            self.cache['mapto_all'][param_hash] = F
        return F

    @staticmethod
    @njit
    def X1X2v_to_xyz(X1, X2, v):

        x =   X1 * np.cos(2 * np.pi * v)
        y = - X1 * np.sin(2 * np.pi * v)
        z =   X2

        return (x, y, z)

    def mapto_x(self, s, u, v):
        """Map (s, u, v) coordinate to Cartesian x.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        u : float or meshgrid numpy.ndarray
            Logical coordinate along poloidal direction. u = theta/(2*PI).

        v : float or meshgrid numpy.ndarray
            Logical coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        float or meshgrid numpy.ndarray
            Transformed x coordinates."""

        if isinstance(s, np.ndarray):

            suv_to_xyz.assert_array_input(s, u, v)

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == u.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, u.ndim)
                assert s.ndim == v.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, v.ndim)
                s, u, v = np.meshgrid(s, u, v, indexing='ij', sparse=True)

        X1 = self.X1_base.eval_suv(s, u, v, self.X1_coef)
        # X2 = self.X2_base.eval_suv(s, u, v, self.X2_coef)
        # LA = self.LA_base.eval_suv(s, u, v, self.LA_coef) # Not necessary to compute lambda.

        # x =   X1 * np.cos(2 * np.pi * v)
        # y = - X1 * np.sin(2 * np.pi * v)
        # z =   X2

        return np.array(suv_to_xyz.X1X2v_to_x(X1, v))

    @staticmethod
    @njit
    def X1X2v_to_x(X1, v):

        return X1 * np.cos(2 * np.pi * v)

    def mapto_y(self, s, u, v):
        """Map (s, u, v) coordinate to Cartesian y.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        u : float or meshgrid numpy.ndarray
            Logical coordinate along poloidal direction. u = theta/(2*PI).

        v : float or meshgrid numpy.ndarray
            Logical coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        float or meshgrid numpy.ndarray
            Transformed y coordinates."""

        if isinstance(s, np.ndarray):

            suv_to_xyz.assert_array_input(s, u, v)

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == u.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, u.ndim)
                assert s.ndim == v.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, v.ndim)
                s, u, v = np.meshgrid(s, u, v, indexing='ij', sparse=True)

        X1 = self.X1_base.eval_suv(s, u, v, self.X1_coef)
        # X2 = self.X2_base.eval_suv(s, u, v, self.X2_coef)
        # LA = self.LA_base.eval_suv(s, u, v, self.LA_coef) # Not necessary to compute lambda.

        # x =   X1 * np.cos(2 * np.pi * v)
        # y = - X1 * np.sin(2 * np.pi * v)
        # z =   X2

        return np.array(suv_to_xyz.X1X2v_to_y(X1, v))

    @staticmethod
    @njit
    def X1X2v_to_y(X1, v):

        return - X1 * np.sin(2 * np.pi * v)

    def mapto_z(self, s, u, v):
        """Map (s, u, v) coordinate to Cartesian z.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        u : float or meshgrid numpy.ndarray
            Logical coordinate along poloidal direction. u = theta/(2*PI).

        v : float or meshgrid numpy.ndarray
            Logical coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        float or meshgrid numpy.ndarray
            Transformed z coordinates."""

        if isinstance(s, np.ndarray):

            suv_to_xyz.assert_array_input(s, u, v)

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == u.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, u.ndim)
                assert s.ndim == v.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, v.ndim)
                s, u, v = np.meshgrid(s, u, v, indexing='ij', sparse=True)

        # X1 = self.X1_base.eval_suv(s, u, v, self.X1_coef)
        X2 = self.X2_base.eval_suv(s, u, v, self.X2_coef)
        # LA = self.LA_base.eval_suv(s, u, v, self.LA_coef) # Not necessary to compute lambda.

        # x =   X1 * np.cos(2 * np.pi * v)
        # y = - X1 * np.sin(2 * np.pi * v)
        # z =   X2

        return np.array(X2)



    def df(self, s, u, v):
        """Point-wise evaluation of Jacobian df.
        i.e. First derivative of the mapping to Cartesian (x, y, z).

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        u : float or meshgrid numpy.ndarray
            Logical coordinate along poloidal direction. u = theta/(2*PI).

        v : float or meshgrid numpy.ndarray
            Logical coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Jacobian of coordinate transform df, evaluated at (s, u, v).
        """

        if self.use_cache:
            param_hash = hash(str(s) + str(u) + str(v))
            if param_hash in self.cache['df']:
                return self.cache['df'][param_hash]

        if isinstance(s, np.ndarray):

            suv_to_xyz.assert_array_input(s, u, v)

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == u.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, u.ndim)
                assert s.ndim == v.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, v.ndim)
                s, u, v = np.meshgrid(s, u, v, indexing='ij', sparse=True)

        X1    = self.X1_base.eval_suv(s, u, v, self.X1_coef)

        dX1ds = self.X1_base.eval_suv_ds(s, u, v, self.X1_coef)
        dX1du = self.X1_base.eval_suv_du(s, u, v, self.X1_coef)
        dX1dv = self.X1_base.eval_suv_dv(s, u, v, self.X1_coef)

        dX2ds = self.X2_base.eval_suv_ds(s, u, v, self.X2_coef)
        dX2du = self.X2_base.eval_suv_du(s, u, v, self.X2_coef)
        dX2dv = self.X2_base.eval_suv_dv(s, u, v, self.X2_coef)

        # dxds = dX1ds * (  np.cos(2 * np.pi * v))
        # dxdu = dX1du * (  np.cos(2 * np.pi * v))
        # dxdv = dX1dv * (  np.cos(2 * np.pi * v)) + ( X1) * (- 2 * np.pi * np.sin(2 * np.pi * v))

        # dyds = dX1ds * (- np.sin(2 * np.pi * v))
        # dydu = dX1du * (- np.sin(2 * np.pi * v))
        # dydv = dX1dv * (- np.sin(2 * np.pi * v)) + (-X1) * (  2 * np.pi * np.cos(2 * np.pi * v))

        # dzds = dX2ds
        # dzdu = dX2du
        # dzdv = dX2dv

        J = suv_to_xyz.df_sin_cos(X1, dX1ds, dX1du, dX1dv, dX2ds, dX2du, dX2dv, v)
        J = np.array(J, dtype=float)
        J = suv_to_xyz.swap_J_axes(J)
        if self.use_cache:
            self.cache['df'][param_hash] = J
        return J

    @staticmethod
    @njit
    def df_sin_cos(X1, dX1ds, dX1du, dX1dv, dX2ds, dX2du, dX2dv, v):

        dxds = dX1ds * (  np.cos(2 * np.pi * v))
        dxdu = dX1du * (  np.cos(2 * np.pi * v))
        dxdv = dX1dv * (  np.cos(2 * np.pi * v)) + ( X1) * (- 2 * np.pi * np.sin(2 * np.pi * v))

        dyds = dX1ds * (- np.sin(2 * np.pi * v))
        dydu = dX1du * (- np.sin(2 * np.pi * v))
        dydv = dX1dv * (- np.sin(2 * np.pi * v)) + (-X1) * (  2 * np.pi * np.cos(2 * np.pi * v))

        dzds = dX2ds
        dzdu = dX2du
        dzdv = dX2dv

        return (
            (dxds, dxdu, dxdv),
            (dyds, dydu, dydv),
            (dzds, dzdu, dzdv),
        )

    def df_det(self, s, u, v):
        """Point-wise evaluation of Jacobian determinant det(df) = df/ds dot (df/du x df/dv).

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        u : float or meshgrid numpy.ndarray
            Logical coordinate along poloidal direction. u = theta/(2*PI).

        v : float or meshgrid numpy.ndarray
            Logical coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Jacobian determinant det(df) = sqrt(det(G)), evaluated at (s, u, v).
        """
        J = self.df(s, u, v)
        return np.linalg.det(J)

    def df_inv(self, s, u, v):
        """Point-wise evaluation of the inverse Jacobian matrix df^(-1)_ij (i,j=1,2,3).

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        u : float or meshgrid numpy.ndarray
            Logical coordinate along poloidal direction. u = theta/(2*PI).

        v : float or meshgrid numpy.ndarray
            Logical coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Inverse Jacobian df^(-1), evaluated at (s, u, v).
        """
        J = self.df(s, u, v)
        return np.linalg.inv(J)

    def G(self, s, u, v):
        """Point-wise evaluation of metric tensor G_ij = sum_k (df^T)_ik (df)_kj (i,j,k=1,2,3).

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        u : float or meshgrid numpy.ndarray
            Logical coordinate along poloidal direction. u = theta/(2*PI).

        v : float or meshgrid numpy.ndarray
            Logical coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Metric tensor G, evaluated at (s, u, v).
        """

        J = self.df(s, u, v)

        # If `J` is a batch of Jacobians in a meshgrid.
        if J.ndim == 5:
            G = np.empty_like(J)
            for i in range(J.shape[0]):
                for j in range(J.shape[1]):
                    for k in range(J.shape[2]):
                        G[i, j, k] = J[i, j, k].T @ J[i, j, k]
        # If `J` is one Jacobian.
        else:
            G = J.T @ J
        return G

    def G_inv(self, s, u, v):
        """Point-wise evaluation of inverse metric tensor G^(-1)_ij = sum_k (df^-1)_ik (df^-T)_kj (i,j,k=1,2,3).

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        u : float or meshgrid numpy.ndarray
            Logical coordinate along poloidal direction. u = theta/(2*PI).

        v : float or meshgrid numpy.ndarray
            Logical coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Inverse metric tensor G_inv, evaluated at (s, u, v).
        """

        J = self.df(s, u, v)

        # If `J` is a batch of Jacobians in a meshgrid.
        if J.ndim == 5:
            out = np.empty_like(J)
            for i in range(J.shape[0]):
                for j in range(J.shape[1]):
                    for k in range(J.shape[2]):
                        G = J[i, j, k].T @ J[i, j, k]
                        out[i, j, k] = np.linalg.inv(G)
        # If `J` is one Jacobian.
        else:
            G = J.T @ J
            out = np.linalg.inv(G)
        return out



    def mapto_nth_der(self, s, u, v, n): # pragma: no cover
        """Compute n-th derivative of the mapping to Cartesian (x, y, z) for n > 1.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        u : float or meshgrid numpy.ndarray
            Logical coordinate along poloidal direction. u = theta/(2*PI).

        v : float or meshgrid numpy.ndarray
            Logical coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        (N+1)D-tuple or (N+1)D-tuple of numpy.ndarray
            n-th derivative after coordinate transform, evaluated at (s, u, v).
        """

        raise NotImplementedError('{}-th derivative not implemented.'.format(n))

    def clear_cache(self):
        for key in self.cache:
            self.cache[key] = {}
