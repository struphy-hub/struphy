import numpy as np

from gvec_to_python.hmap.base_map import BaseMap

class sthetazeta_to_X1X2zeta(BaseMap):

    def __init__(self, X1_base, X1_coef, X2_base, X2_coef, LA_base, LA_coef):

        (self.X1_base, self.X1_coef, self.X2_base, self.X2_coef, self.LA_base, self.LA_coef) = (X1_base, X1_coef, X2_base, X2_coef, LA_base, LA_coef)

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



    def mapto(self, s, theta, zeta):
        """Map (s, theta, zeta) coordinate to left-handed (X1, X2, zeta).

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        theta : float or meshgrid numpy.ndarray
            Angular-coordinate along poloidal direction. u = theta/(2*PI).

        zeta : float or meshgrid numpy.ndarray
            Angular-coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        3-tuple or 3-tuple of meshgrid numpy.ndarray
            Transformed coordinates."""

        if isinstance(s, np.ndarray):

            sthetazeta_to_X1X2zeta.assert_array_input(s, theta, zeta)

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == theta.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, theta.ndim)
                assert s.ndim ==  zeta.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim,  zeta.ndim)
                s, theta, zeta = np.meshgrid(s, theta, zeta, indexing='ij', sparse=True)

        X1 = self.X1_base.eval_stz(s, theta, zeta, self.X1_coef)
        X2 = self.X2_base.eval_stz(s, theta, zeta, self.X2_coef)
        # LA = self.LA_base.eval_stz(s, theta, zeta, self.LA_coef) # Not necessary to compute lambda.

        return (X1, X2, zeta)



    def df(self, s, theta, zeta):
        """Point-wise evaluation of Jacobian df.
        i.e. First derivative of the mapping to left-handed (X1, X2, zeta).

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        theta : float or meshgrid numpy.ndarray
            Angular-coordinate along poloidal direction. u = theta/(2*PI).

        zeta : float or meshgrid numpy.ndarray
            Angular-coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Jacobian of coordinate transform df, evaluated at (s, theta, zeta).
        """

        if isinstance(s, np.ndarray):

            sthetazeta_to_X1X2zeta.assert_array_input(s, theta, zeta)

            # If input coordinates are simple 1D arrays, turn them into a sparse meshgrid.
            # The output will fallthrough to the logic below, which expects a meshgrid input.
            if s.ndim == 1:
                assert s.ndim == theta.ndim, '2nd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim, theta.ndim)
                assert s.ndim ==  zeta.ndim, '3rd argument has different dimensions than the 1st. Expected {}, got {} instead.'.format(s.ndim,  zeta.ndim)
                s, theta, zeta = np.meshgrid(s, theta, zeta, indexing='ij', sparse=True)

        dX1ds       = self.X1_base.eval_stz_ds    (s, theta, zeta, self.X1_coef)
        dX1dtheta   = self.X1_base.eval_stz_dtheta(s, theta, zeta, self.X1_coef)
        dX1dzeta    = self.X1_base.eval_stz_dzeta (s, theta, zeta, self.X1_coef)

        dX2ds       = self.X2_base.eval_stz_ds    (s, theta, zeta, self.X2_coef)
        dX2dtheta   = self.X2_base.eval_stz_dtheta(s, theta, zeta, self.X2_coef)
        dX2dzeta    = self.X2_base.eval_stz_dzeta (s, theta, zeta, self.X2_coef)

        if isinstance(s, np.ndarray):

            dummy_broadcast = np.zeros_like(s) * np.zeros_like(theta) * np.zeros_like(zeta)
            dzetads     = np.zeros(dummy_broadcast.shape)
            dzetadtheta = np.zeros(dummy_broadcast.shape)
            dzetadzeta  =  np.ones(dummy_broadcast.shape)

        else:

            dzetads     = 0
            dzetadtheta = 0
            dzetadzeta  = 1

        J = (
            (dX1ds      , dX1dtheta  , dX1dzeta   ),
            (dX2ds      , dX2dtheta  , dX2dzeta   ),
            (dzetads    , dzetadtheta, dzetadzeta ),
        )

        J = np.array(J, dtype=float)
        J = sthetazeta_to_X1X2zeta.swap_J_axes(J)
        return J

    def df_det(self, s, theta, zeta):
        """Point-wise evaluation of Jacobian determinant det(df) = df/ds dot (df/du x df/dv).

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        theta : float or meshgrid numpy.ndarray
            Angular-coordinate along poloidal direction. u = theta/(2*PI).

        zeta : float or meshgrid numpy.ndarray
            Angular-coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Jacobian determinant det(df) = sqrt(det(G)), evaluated at (s, theta, zeta).
        """
        J = self.df(s, theta, zeta)
        return np.linalg.det(J)

    def df_inv(self, s, theta, zeta):
        """Point-wise evaluation of the inverse Jacobian matrix df^(-1)_ij (i,j=1,2,3).

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        theta : float or meshgrid numpy.ndarray
            Angular-coordinate along poloidal direction. u = theta/(2*PI).

        zeta : float or meshgrid numpy.ndarray
            Angular-coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Inverse Jacobian df^(-1), evaluated at (s, theta, zeta).
        """
        J = self.df(s, theta, zeta)
        return np.linalg.inv(J)

    def G(self, s, theta, zeta):
        """Point-wise evaluation of metric tensor G_ij = sum_k (df^T)_ik (df)_kj (i,j,k=1,2,3).

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        theta : float or meshgrid numpy.ndarray
            Angular-coordinate along poloidal direction. u = theta/(2*PI).

        zeta : float or meshgrid numpy.ndarray
            Angular-coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Metric tensor G, evaluated at (s, theta, zeta).
        """

        J = self.df(s, theta, zeta)

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

    def G_inv(self, s, theta, zeta):
        """Point-wise evaluation of inverse metric tensor G^(-1)_ij = sum_k (df^-1)_ik (df^-T)_kj (i,j,k=1,2,3).

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        theta : float or meshgrid numpy.ndarray
            Angular-coordinate along poloidal direction. u = theta/(2*PI).

        zeta : float or meshgrid numpy.ndarray
            Angular-coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        numpy.ndarray of shape (3, 3) or (..., 3, 3)
            Inverse metric tensor G_inv, evaluated at (s, theta, zeta).
        """

        J = self.df(s, theta, zeta)

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



    def mapto_nth_der(self, s, theta, zeta, n): # pragma: no cover
        """Compute n-th derivative of the mapping to left-handed (X1, X2, zeta) for n > 1.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logial radial-coordinate from toroidal axis.

        theta : float or meshgrid numpy.ndarray
            Angular-coordinate along poloidal direction. u = theta/(2*PI).

        zeta : float or meshgrid numpy.ndarray
            Angular-coordinate along toroidal direction. v = zeta/(2*PI).

        Returns
        -------
        (N+1)D-tuple or (N+1)D-tuple of numpy.ndarray
            n-th derivative after coordinate transform, evaluated at (s, theta, zeta).
        """

        raise NotImplementedError('{}-th derivative not implemented.'.format(n))
