import numpy as np
from numba import jit, njit

from gvec_to_python.hmap.base_map import BaseMap

class RphiZ_to_xyz(BaseMap):

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
    def mapto(R, phi, Z):
        """Map right-handed (R, phi, Z) coordinate to (x, y, z).

        Parameters
        ----------
        R : float or meshgrid numpy.ndarray
            Radial direction in cylindrical-coordinate.

        phi : float or meshgrid numpy.ndarray
            Polar direction in cylindrical-coordinate.

        Z : float or meshgrid numpy.ndarray
            Vertical direction in cylindrical-coordinate.

        Returns
        -------
        3-tuple or 3-tuple of meshgrid numpy.ndarray
            Transformed coordinates."""

        if isinstance(R, np.ndarray):

            RphiZ_to_xyz.assert_array_input(R, phi, Z)

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if R.ndim == 1:
                assert R.ndim == phi.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(R.ndim, phi.ndim)
                assert R.ndim ==   Z.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(R.ndim,   Z.ndim)
                R, phi, Z = np.meshgrid(R, phi, Z, indexing='ij', sparse=True)

        return np.array(RphiZ_to_xyz.mapto_njit(R, phi, Z))

    @staticmethod
    @njit
    def mapto_njit(R, phi, Z):

        return (R * np.cos(phi), R * np.sin(phi), Z)



    @staticmethod
    def df(R, phi, Z):
        """Point-wise evaluation of Jacobian df.
        i.e. First derivative of the mapping to Cartesian (x, y, z).

        Parameters
        ----------
        R : float or meshgrid numpy.ndarray
            Radial direction in cylindrical-coordinate.

        phi : float or meshgrid numpy.ndarray
            Polar direction in cylindrical-coordinate.

        Z : float or meshgrid numpy.ndarray
            Vertical direction in cylindrical-coordinate.

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Jacobian of coordinate transform df, evaluated at (R, phi, Z).
        """

        # To indirectly test for ndarray without calling `type` or `isinstance`, because Numba does not support these built-in functions.
        # if len(np.shape(R)) > 0: # Nah. Numba broke anyway.
        if isinstance(R, np.ndarray):

            RphiZ_to_xyz.assert_array_input(R, phi, Z)

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if R.ndim == 1:
                assert R.ndim == phi.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(R.ndim, phi.ndim)
                assert R.ndim ==   Z.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(R.ndim,   Z.ndim)
                R, phi, Z = np.meshgrid(R, phi, Z, indexing='ij', sparse=True)

            J = RphiZ_to_xyz.df_array(R, phi, Z)

        else:

            J = RphiZ_to_xyz.df_float(R, phi, Z)

        J = np.array(J, dtype=float)
        J = RphiZ_to_xyz.swap_J_axes(J)
        return J

    @staticmethod
    @njit
    def df_array(R, phi, Z):

        dummy_broadcast = np.zeros_like(R) * np.zeros_like(phi) * np.zeros_like(Z)
        return (
            (np.cos(phi) *  np.ones_like(dummy_broadcast),      - R * np.sin(phi) *  np.ones_like(dummy_broadcast), np.zeros_like(dummy_broadcast)),
            (np.sin(phi) *  np.ones_like(dummy_broadcast),        R * np.cos(phi) *  np.ones_like(dummy_broadcast), np.zeros_like(dummy_broadcast)),
            (              np.zeros_like(dummy_broadcast),                          np.zeros_like(dummy_broadcast),  np.ones_like(dummy_broadcast)),
        )

    @staticmethod
    @njit
    def df_float(R, phi, Z):

        return (
            (np.cos(phi), - R * np.sin(phi), 0),
            (np.sin(phi),   R * np.cos(phi), 0),
            (          0,                 0, 1),
        )

    @staticmethod
    def df_det(R, phi, Z):
        """Point-wise evaluation of Jacobian determinant det(df) = df/ds dot (df/du x df/dv).

        Parameters
        ----------
        R : float or meshgrid numpy.ndarray
            Radial direction in cylindrical-coordinate.

        phi : float or meshgrid numpy.ndarray
            Polar direction in cylindrical-coordinate.

        Z : float or meshgrid numpy.ndarray
            Vertical direction in cylindrical-coordinate.

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Jacobian determinant det(df) = sqrt(det(G)), evaluated at (R, phi, Z).
        """
        J = RphiZ_to_xyz.df(R, phi, Z)
        return np.linalg.det(J)

    @staticmethod
    def df_inv(R, phi, Z):
        """Point-wise evaluation of the inverse Jacobian matrix df^(-1)_ij (i,j=1,2,3).

        Parameters
        ----------
        R : float or meshgrid numpy.ndarray
            Radial direction in cylindrical-coordinate.

        phi : float or meshgrid numpy.ndarray
            Polar direction in cylindrical-coordinate.

        Z : float or meshgrid numpy.ndarray
            Vertical direction in cylindrical-coordinate.

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Inverse Jacobian df^(-1), evaluated at (R, phi, Z).
        """
        J = RphiZ_to_xyz.df(R, phi, Z)
        return np.linalg.inv(J)

    @staticmethod
    def G(R, phi, Z):
        """Point-wise evaluation of metric tensor G_ij = sum_k (df^T)_ik (df)_kj (i,j,k=1,2,3).

        Parameters
        ----------
        R : float or meshgrid numpy.ndarray
            Radial direction in cylindrical-coordinate.

        phi : float or meshgrid numpy.ndarray
            Polar direction in cylindrical-coordinate.

        Z : float or meshgrid numpy.ndarray
            Vertical direction in cylindrical-coordinate.

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Metric tensor G, evaluated at (R, phi, Z).
        """

        J = RphiZ_to_xyz.df(R, phi, Z)

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
    def G_inv(R, phi, Z):
        """Point-wise evaluation of inverse metric tensor G^(-1)_ij = sum_k (df^-1)_ik (df^-T)_kj (i,j,k=1,2,3).

        Parameters
        ----------
        R : float or meshgrid numpy.ndarray
            Radial direction in cylindrical-coordinate.

        phi : float or meshgrid numpy.ndarray
            Polar direction in cylindrical-coordinate.

        Z : float or meshgrid numpy.ndarray
            Vertical direction in cylindrical-coordinate.

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Inverse metric tensor G_inv, evaluated at (R, phi, Z).
        """

        J = RphiZ_to_xyz.df(R, phi, Z)

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
    @njit
    def mapto_nth_der(R, phi, Z, n): # pragma: no cover
        """Compute n-th derivative of the mapping to (x, y, z).

        Parameters
        ----------
        R : float or meshgrid numpy.ndarray
            Radial direction in cylindrical-coordinate.

        phi : float or meshgrid numpy.ndarray
            Polar direction in cylindrical-coordinate.

        Z : float or meshgrid numpy.ndarray
            Vertical direction in cylindrical-coordinate.

        Returns
        -------
        (N+1)D-tuple or (N+1)D-tuple of numpy.ndarray
            n-th derivative after coordinate transform, evaluated at (R, phi, Z).
        """

        raise NotImplementedError('{}-th derivative not implemented.'.format(n))
