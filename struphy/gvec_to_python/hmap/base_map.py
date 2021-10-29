import numpy as np
from numba import jit, njit

import warnings
warnings.filterwarnings("ignore", message="faster on contiguous arrays,")
# warnings.filterwarnings("ignore", message="NumbaPerformanceWarning: '@' is faster on contiguous arrays,")



class BaseMap:
    """Contain basic functions that are shared across all mapping classes.

    There are only static methods in this class. There is no need to call `super().__init__()`."""

    @staticmethod
    def assert_array_input(a1, a2, a3): # pragma: no cover

            assert isinstance(a1, np.ndarray), '1st argument should be of type `np.ndarray`. Got {} instead.'.format(type(a1))
            assert isinstance(a2, np.ndarray), '2nd argument should be of type `np.ndarray`. Got {} instead.'.format(type(a2))
            assert isinstance(a3, np.ndarray), '3rd argument should be of type `np.ndarray`. Got {} instead.'.format(type(a3))

            # assert a1.shape == a2.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 2nd argument is {}.'.format(a1.shape, a2.shape)
            # assert a1.shape == a3.shape, 'All inputs should have the same shape, because they are created from a dense meshgrid. Instead, 1st argument is {}, while 3rd argument is {}.'.format(a1.shape, a3.shape)

            return True

    @staticmethod
    def swap_J_axes(J):
        """Swap axes of a batch of Jacobians, such that it is compatible with numpy's batch processing.

        When the inputs are 1D arrays or 3D arrays of meshgrids, the Jacobian dimensions by default will be (3, 3, eta1, eta2, eta3).  
        However, all of numpy's matrix operations expect the 3x3 part to be the last two dimensions, i.e. (eta1, eta2, eta3, 3, 3).  
        This function will first check if the Jacobian has dimensions > 2 (there is no point swapping axis of a scalar input).
        Then it will check if the 3x3 portion is at the beginning of the `shape` tuple. 
        If the conditions are met, it will move the first two axes of 5D Jacobian to the last two, such that it is compatible with numpy's batch processing.

        Parameters
        ----------
        J : numpy.ndarray of shape (3, 3) or (3, 3, ...)
            A batch of Jacobians.

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            A batch of Jacobians.
        """

        if J.ndim > 2 and J.shape[:2] == (3, 3):
            J = np.moveaxis(J, 0, -1)
            J = np.moveaxis(J, 0, -1)
        return J

    @staticmethod
    @njit
    def df_det_from_J(J):
        """Batch evaluation of Jacobian determinant det(df) = df/ds dot (df/du x df/dv).

        Parameters
        ----------
        J : numpy.ndarray of shape (3, 3) or (..., 3, 3)
            A batch of Jacobians.

        Returns
        -------
        numpy.ndarray of shape () or (...,)
            A batch of Jacobian determinants det(df) = sqrt(det(G)).
        """

        # If `J` is a batch of Jacobians in a meshgrid.
        if J.ndim == 5:
            out = np.empty(J.shape[:3])
            for i in range(J.shape[0]):
                for j in range(J.shape[1]):
                    for k in range(J.shape[2]):
                        out[i, j, k] = np.linalg.det(J[i, j, k])
        # If `J` is one Jacobian.
        else:
            out = np.linalg.det(J)
        return out

    @staticmethod
    @njit
    def df_inv_from_J(J):
        """Batch evaluation of the inverse Jacobian matrix df^(-1)_ij (i,j=1,2,3).

        Parameters
        ----------
        J : numpy.ndarray of shape (3, 3) or (..., 3, 3)
            A batch of Jacobians.

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            A batch of inverse Jacobians df^(-1).
        """

        # If `J` is a batch of Jacobians in a meshgrid.
        if J.ndim == 5:
            out = np.empty_like(J)
            for i in range(J.shape[0]):
                for j in range(J.shape[1]):
                    for k in range(J.shape[2]):
                        out[i, j, k] = np.linalg.inv(J[i, j, k])
        # If `J` is one Jacobian.
        else:
            out = np.linalg.inv(J)
        return out

    @staticmethod
    @njit
    def G_from_J(J):
        """Batch evaluation of metric tensor G_ij = sum_k (df^T)_ik (df)_kj (i,j,k=1,2,3).

        Parameters
        ----------
        J : numpy.ndarray of shape (3, 3) or (..., 3, 3)
            A batch of Jacobians.

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            A batch of metric tensors G.
        """

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
    @njit
    def G_inv_from_J(J):
        """Batch evaluation of inverse metric tensor G^(-1)_ij = sum_k (df^-1)_ik (df^-T)_kj (i,j,k=1,2,3).

        Parameters
        ----------
        J : numpy.ndarray of shape (3, 3) or (..., 3, 3)
            A batch of Jacobians.

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            A batch of inverse metric tensors G_inv.
        """

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
    def G_inv_from_G(G):
        """Batch evaluation of inverse metric tensor G^(-1)_ij = sum_k (df^-1)_ik (df^-T)_kj (i,j,k=1,2,3).

        Parameters
        ----------
        G : numpy.ndarray of shape (3, 3) or (..., 3, 3)
            A batch of metric tensors G.

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            A batch of inverse metric tensors G_inv.
        """

        # If `G` is a batch of metric tensors in a meshgrid.
        if G.ndim == 5:
            out = np.empty_like(G)
            for i in range(G.shape[0]):
                for j in range(G.shape[1]):
                    for k in range(G.shape[2]):
                        out[i, j, k] = np.linalg.inv(G[i, j, k])
        # If `G` is one metric tensor.
        else:
            out = np.linalg.inv(G)
        return out
