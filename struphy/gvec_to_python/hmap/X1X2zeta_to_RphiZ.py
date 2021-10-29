import numpy as np
from numba import jit, njit

from gvec_to_python.hmap.base_map import BaseMap

class X1X2zeta_to_RphiZ(BaseMap):

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
    def mapto(X1, X2, zeta):
        """Map left-handed (X1, X2, zeta) coordinate to right-handed (R, phi, Z).

        Parameters
        ----------
        X1 : float or meshgrid numpy.ndarray
            Solution variables of (s, theta, zeta). X1 = R in (R, phi, Z) coodordinate

        X2 : float or meshgrid numpy.ndarray
            Solution variables of (s, theta, zeta). X2 = Z in (R, phi, Z) coodordinate

        zeta : float or meshgrid numpy.ndarray
            Angular-coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        3-tuple or 3-tuple of meshgrid numpy.ndarray
            Transformed coordinates."""

        if isinstance(X1, np.ndarray):

            X1X2zeta_to_RphiZ.assert_array_input(X1, X2, zeta)

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if X1.ndim == 1:
                assert X1.ndim ==   X2.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(X1.ndim,   X2.ndim)
                assert X1.ndim == zeta.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(X1.ndim, zeta.ndim)
                X1, X2, zeta = np.meshgrid(X1, X2, zeta, indexing='ij', sparse=True)

        return X1X2zeta_to_RphiZ.mapto_njit(X1, X2, zeta)

    @staticmethod
    @njit
    def mapto_njit(X1, X2, zeta):

        return (X1, -zeta, X2)



    @staticmethod
    def df(X1, X2, zeta):
        """Point-wise evaluation of Jacobian df.
        i.e. First derivative of the mapping to (R, phi, Z).

        Parameters
        ----------
        X1 : float or meshgrid numpy.ndarray
            Solution variables of (s, theta, zeta). X1 = R in (R, phi, Z) coodordinate

        X2 : float or meshgrid numpy.ndarray
            Solution variables of (s, theta, zeta). X2 = Z in (R, phi, Z) coodordinate

        zeta : float or meshgrid numpy.ndarray
            Angular-coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Jacobian of coordinate transform df, evaluated at (X1, X2, zeta).
        """

        # To indirectly test for ndarray without calling `type` or `isinstance`, because Numba does not support these built-in functions.
        # if len(np.shape(X1)) > 0: # Nah. Numba broke anyway.
        if isinstance(X1, np.ndarray):

            X1X2zeta_to_RphiZ.assert_array_input(X1, X2, zeta)

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if X1.ndim == 1:
                assert X1.ndim ==   X2.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(X1.ndim,   X2.ndim)
                assert X1.ndim == zeta.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(X1.ndim, zeta.ndim)
                X1, X2, zeta = np.meshgrid(X1, X2, zeta, indexing='ij', sparse=True)

            J = X1X2zeta_to_RphiZ.df_array(X1, X2, zeta)

        else:

            J = X1X2zeta_to_RphiZ.df_float(X1, X2, zeta)

        J = np.array(J, dtype=float)
        J = X1X2zeta_to_RphiZ.swap_J_axes(J)
        return J

    @staticmethod
    @njit
    def df_array(X1, X2, zeta):

        dummy_broadcast = np.zeros_like(X1) * np.zeros_like(X2) * np.zeros_like(zeta)
        return (
            ( np.ones_like(dummy_broadcast), np.zeros_like(dummy_broadcast),  np.zeros_like(dummy_broadcast)),
            (np.zeros_like(dummy_broadcast), np.zeros_like(dummy_broadcast), - np.ones_like(dummy_broadcast)),
            (np.zeros_like(dummy_broadcast),  np.ones_like(dummy_broadcast),  np.zeros_like(dummy_broadcast)),
        )

    @staticmethod
    @njit
    def df_float(X1, X2, zeta):

        return (
            (1, 0,  0),
            (0, 0, -1),
            (0, 1,  0),
        )

    @staticmethod
    def df_det(X1, X2, zeta):
        """Point-wise evaluation of Jacobian determinant det(df) = df/ds dot (df/du x df/dv).

        Parameters
        ----------
        X1 : float or meshgrid numpy.ndarray
            Solution variables of (s, theta, zeta). X1 = R in (R, phi, Z) coodordinate

        X2 : float or meshgrid numpy.ndarray
            Solution variables of (s, theta, zeta). X2 = Z in (R, phi, Z) coodordinate

        zeta : float or meshgrid numpy.ndarray
            Angular-coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Jacobian determinant det(df) = sqrt(det(G)), evaluated at (X1, X2, zeta).
        """
        J = X1X2zeta_to_RphiZ.df(X1, X2, zeta)
        return np.linalg.det(J)

    @staticmethod
    def df_inv(X1, X2, zeta):
        """Point-wise evaluation of the inverse Jacobian matrix df^(-1)_ij (i,j=1,2,3).

        Parameters
        ----------
        X1 : float or meshgrid numpy.ndarray
            Solution variables of (s, theta, zeta). X1 = R in (R, phi, Z) coodordinate

        X2 : float or meshgrid numpy.ndarray
            Solution variables of (s, theta, zeta). X2 = Z in (R, phi, Z) coodordinate

        zeta : float or meshgrid numpy.ndarray
            Angular-coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Inverse Jacobian df^(-1), evaluated at (X1, X2, zeta).
        """
        J = X1X2zeta_to_RphiZ.df(X1, X2, zeta)
        return np.linalg.inv(J)

    @staticmethod
    def G(X1, X2, zeta):
        """Point-wise evaluation of metric tensor G_ij = sum_k (df^T)_ik (df)_kj (i,j,k=1,2,3).

        Parameters
        ----------
        X1 : float or meshgrid numpy.ndarray
            Solution variables of (s, theta, zeta). X1 = R in (R, phi, Z) coodordinate

        X2 : float or meshgrid numpy.ndarray
            Solution variables of (s, theta, zeta). X2 = Z in (R, phi, Z) coodordinate

        zeta : float or meshgrid numpy.ndarray
            Angular-coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Metric tensor G, evaluated at (X1, X2, zeta).
        """

        J = X1X2zeta_to_RphiZ.df(X1, X2, zeta)

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
    def G_inv(X1, X2, zeta):
        """Point-wise evaluation of inverse metric tensor G^(-1)_ij = sum_k (df^-1)_ik (df^-T)_kj (i,j,k=1,2,3).

        Parameters
        ----------
        X1 : float or meshgrid numpy.ndarray
            Solution variables of (s, theta, zeta). X1 = R in (R, phi, Z) coodordinate

        X2 : float or meshgrid numpy.ndarray
            Solution variables of (s, theta, zeta). X2 = Z in (R, phi, Z) coodordinate

        zeta : float or meshgrid numpy.ndarray
            Angular-coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Inverse metric tensor G_inv, evaluated at (X1, X2, zeta).
        """

        J = X1X2zeta_to_RphiZ.df(X1, X2, zeta)

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
    def mapto_nth_der(X1, X2, zeta, n): # pragma: no cover
        """Compute n-th derivative of the mapping to (R, phi, Z) for n > 1.

        Parameters
        ----------
        X1 : float or meshgrid numpy.ndarray
            Solution variables of (s, theta, zeta). X1 = R in (R, phi, Z) coodordinate

        X2 : float or meshgrid numpy.ndarray
            Solution variables of (s, theta, zeta). X2 = Z in (R, phi, Z) coodordinate

        zeta : float or meshgrid numpy.ndarray
            Angular-coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        (N+1)D-tuple or (N+1)D-tuple of numpy.ndarray
            n-th derivative after coordinate transform, evaluated at (X1, X2, zeta).
        """

        raise NotImplementedError('{}-th derivative not implemented.'.format(n))

        # To indirectly test for ndarray without calling `type` or `isinstance`, because Numba does not support these built-in functions.
        if len(np.shape(X1)) > 0:

            return np.zeros([3 for i in range(n + 1)] + [X1.shape[0]])

        else:

            return np.zeros([3 for i in range(n + 1)])
