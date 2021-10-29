import logging
import numpy as np
from numba import jit, njit

from gvec_to_python.hmap.base_map import BaseMap

class suv_to_sthetazeta(BaseMap):

    def __init__(self):

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



    @staticmethod
    def mapto(s, u, v):
        """Map (s, u, v) coordinate to (s, theta, zeta).

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

        if isinstance(s, np.ndarray):

            suv_to_sthetazeta.assert_array_input(s, u, v)

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == u.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, u.ndim)
                assert s.ndim == v.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, v.ndim)
                s, u, v = np.meshgrid(s, u, v, indexing='ij', sparse=True)

        return suv_to_sthetazeta.mapto_njit(s, u, v)

    @staticmethod
    @njit
    def mapto_njit(s, u, v):

        return (s, 2 * np.pi * u, 2 * np.pi * v)



    @staticmethod
    def df(s, u, v):
        """Point-wise evaluation of Jacobian df.
        i.e. First derivative of the mapping to (s, theta, zeta).

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

        # To indirectly test for ndarray without calling `type` or `isinstance`, because Numba does not support these built-in functions.
        # if len(np.shape(s)) > 0: # Nah. Numba broke anyway.
        if isinstance(s, np.ndarray):

            suv_to_sthetazeta.assert_array_input(s, u, v)

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == u.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, u.ndim)
                assert s.ndim == v.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, v.ndim)
                s, u, v = np.meshgrid(s, u, v, indexing='ij', sparse=True)

            J = suv_to_sthetazeta.df_array(s, u, v)

        else:

            J = suv_to_sthetazeta.df_float(s, u, v)

        J = np.array(J, dtype=float)
        J = suv_to_sthetazeta.swap_J_axes(J)
        return J

    @staticmethod
    @njit
    def df_array(s, u, v):

        dummy_broadcast = np.zeros_like(s) * np.zeros_like(u) * np.zeros_like(v)
        return (
            ( np.ones_like(dummy_broadcast),            np.zeros_like(dummy_broadcast),            np.zeros_like(dummy_broadcast)),
            (np.zeros_like(dummy_broadcast), 2 * np.pi * np.ones_like(dummy_broadcast),            np.zeros_like(dummy_broadcast)),
            (np.zeros_like(dummy_broadcast),            np.zeros_like(dummy_broadcast), 2 * np.pi * np.ones_like(dummy_broadcast)),
        )

    @staticmethod
    @njit
    def df_float(s, u, v):

        return (
            (1,         0,         0),
            (0, 2 * np.pi,         0),
            (0,         0, 2 * np.pi),
        )

    @staticmethod
    def df_det(s, u, v):
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
        J = suv_to_sthetazeta.df(s, u, v)
        return np.linalg.det(J)

    @staticmethod
    def df_inv(s, u, v):
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
        J = suv_to_sthetazeta.df(s, u, v)
        return np.linalg.inv(J)

    @staticmethod
    def G(s, u, v):
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

        J = suv_to_sthetazeta.df(s, u, v)

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

    @staticmethod
    def G_inv(s, u, v):
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

        J = suv_to_sthetazeta.df(s, u, v)

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



    @staticmethod
    # @njit
    def mapto_nth_der(s, u, v, n): # pragma: no cover
        """Compute n-th derivative of the mapping to (s, theta, zeta) for n > 1.

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

        # To indirectly test for ndarray without calling `type` or `isinstance`, because Numba does not support these built-in functions.
        if len(np.shape(s)) > 0:

            return np.zeros([3 for i in range(n + 1)] + [s.shape[0]])

        else:

            return np.zeros([3 for i in range(n + 1)])
