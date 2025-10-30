# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Classes for commuting projectors in 1D, 2D and 3D based on global spline interpolation and histopolation.
"""

import cunumpy as xp
import scipy.sparse as spa

import struphy.bsplines.bsplines as bsp
from struphy.linear_algebra.linalg_kron import kron_lusolve_2d, kron_lusolve_3d, kron_matvec_2d, kron_matvec_3d


# ======================= 1d ====================================
class Projectors_global_1d:
    """
    Global commuting projectors pi_0 and pi_1 in 1d based on interpolation and histopolation.

    Parameters
    -----------
    spline_space : Spline_space_1d
        A 1d space of B-splines and corresponding D-splines.

    n_quad : int
        Number of Gauss-Legendre quadrature points per integration interval for histopolation.

    Attributes
    -----------
    n_quad : int
        The input number of quadrature points per integration interval.

    pts_loc : 1d array
        Gauss-Legendre quadrature points in (-1, 1).

    wts_loc : 1d array
        Gauss-Legendre quadrature weights in (-1, 1).

    x_int : 1d array
        Interpolation points in [0, 1] = Greville points of space.

    x_his : 1d array
        Integration boundaries for histopolation (including breakpoints if there is one between two Greville points).

    subs : 1d array
        Number of sub-integration intervals per histopolation interval to achieve exact integration of splines (1 or 2).

    subs_cum : list
        Cumulative sum of subs, starting with 0.

    pts : 2d array (float)
        Gauss-Legendre quadrature points in format (integration interval, local quadrature point)

    wts : 2d array (float)
        Gauss-Legendre quadrature weights in format (integration interval, local quadrature weight)

    ptsG : 2d array (float)
        Gauss-Legendre quadrature points in format (integration interval, local quadrature point),
        ignoring subs (less accurate integration for even degree)

    wtsG : 2d array (float)
        Gauss-Legendre quadrature weights in format (integration interval, local quadrature weight),
        ignoring subs (less accurate integration for even degree)

    span_x_int_N : 2d float array
        Knot span indices of B-splines at Greville points in format (greville, 0)

    span_x_int_D : 2d float array
        Knot span indices of M-splines at Greville points in format (greville, 0)

    span_ptsG_N : 2d float array
        Knot span indices of B-splines at quadrature points in format of ptsG.

    span_ptsG_D : 2d float array
        Knot span indices of M-splines at quadrature points in format of ptsG.

    basis_x_int_N : 3d float array
        Values of p + 1 non-zero B-spline basis functions at Greville points in format (greville, 0, basis function)

    basis_x_int_D : 3d float array
        Values of p non-zero M-spline basis functions at Greville points in format (greville, 0, basis function)

    basis_ptsG_N : 3d float array
        Values of p + 1 non-zero B-spline basis functions at ptsG in format (i, iq, basis function)

    basis_ptsG_D : 3d float array
        Values of p non-zero M-spline basis functions at ptsG in format (i, iq, basis function)

    Q : sparse csr matrix
        Quadrature matrix that performs quadrature integrations as matrix-vector product

    QG : sparse csr matrix
        Quadrature matrix that performs quadrature integrations as matrix-vector product,
        ignoring subs (less accurate integration for even degree)

    N_int : sparse csr matrix
        Collocation matrix for B-splines at interpolation points

    D_int : sparse csr matrix
        Collocation matrix for M-splines at interpolation points

    N_pts : sparse csr matrix
        Collocation matrix for B-splines at quadrature points

    D_pts : sparse csr matrix
        Collocation matrix for M-splines at quadrature points

    I : sparse csr matrix
        Interpolation matrix N_j(x_i).

    ID : sparse csr matrix
        Interpolation-like matrix D_j(x_i)

    I0 : sparse csr matrix
        Interpolation matrix B0 * I * B0^T including boundary operators.

    H : sparse csr matrix
        Histopolation matrix int_(x_i)^(x_i + 1) D_j dx.

    HN : sparse csr matrix
        Histopolation-like matrix int_(x_i)^(x_i + 1) N_j dx.

    H0 : sparse csr matrix
        Histopolation matrix B1 * H * B1^T including boundary operators.

    I_LU : Super LU
        LU decompositions of I.

    H_LU : Super LU
        LU decompositions of H.

    I0_LU : Super LU
        LU decompositions of I0.

    H0_LU : Super LU
        LU decompositions of H0.

    I_T_LU : Super LU
        LU decompositions of transpose I.

    H_T_LU : Super LU
        LU decompositions of transpose H.

    I0_T_LU : Super LU
        LU decompositions of transpose I0.

    H0_T_LU : Super LU
        LU decompositions of transpose H0.
    """

    def __init__(self, spline_space, n_quad=6):
        self.space = spline_space

        # number of quadrature points per integration interval
        self.n_quad = n_quad

        # Gauss - Legendre quadrature points and weights in (-1, 1)
        self.pts_loc = xp.polynomial.legendre.leggauss(self.n_quad)[0]
        self.wts_loc = xp.polynomial.legendre.leggauss(self.n_quad)[1]

        # set interpolation points (Greville points)
        self.x_int = spline_space.greville.copy()

        # set number of sub-intervals per integration interval between Greville points and integration boundaries
        self.subs = xp.ones(spline_space.NbaseD, dtype=int)
        self.x_his = xp.array([self.x_int[0]])

        for i in range(spline_space.NbaseD):
            for br in spline_space.el_b:
                # left and right integration boundaries
                if not spline_space.spl_kind:
                    xl = self.x_int[i]
                    xr = self.x_int[i + 1]
                else:
                    xl = self.x_int[i]
                    xr = self.x_int[(i + 1) % spline_space.NbaseD]
                    if i == spline_space.NbaseD - 1:
                        xr += spline_space.el_b[-1]

                # compute subs and x_his
                if (br > xl + 1e-10) and (br < xr - 1e-10):
                    self.subs[i] += 1
                    self.x_his = xp.append(self.x_his, br)
                elif br >= xr - 1e-10:
                    self.x_his = xp.append(self.x_his, xr)
                    break

        if spline_space.spl_kind == True and spline_space.p % 2 == 0:
            self.x_his = xp.append(self.x_his, spline_space.el_b[-1] + self.x_his[0])

        # cumulative number of sub-intervals for conversion local interval --> global interval
        self.subs_cum = xp.append(0, xp.cumsum(self.subs - 1)[:-1])

        # quadrature points and weights
        self.pts, self.wts = bsp.quadrature_grid(self.x_his, self.pts_loc, self.wts_loc)
        self.pts = self.pts % spline_space.el_b[-1]

        # quadrature points and weights, ignoring subs (less accurate integration for even degree)
        self.x_hisG = self.x_int
        if spline_space.spl_kind:
            if spline_space.p % 2 == 0:
                self.x_hisG = xp.append(self.x_hisG, spline_space.el_b[-1] + self.x_hisG[0])
            else:
                self.x_hisG = xp.append(self.x_hisG, spline_space.el_b[-1])

        self.ptsG, self.wtsG = bsp.quadrature_grid(self.x_hisG, self.pts_loc, self.wts_loc)
        self.ptsG = self.ptsG % spline_space.el_b[-1]

        # Knot span indices at interpolation points in format (greville, 0)
        self.span_x_int_N = xp.zeros(self.x_int[:, None].shape, dtype=int)
        self.span_x_int_D = xp.zeros(self.x_int[:, None].shape, dtype=int)
        for i in range(self.x_int.shape[0]):
            self.span_x_int_N[i, 0] = bsp.find_span(self.space.T, self.space.p, self.x_int[i])
            self.span_x_int_D[i, 0] = bsp.find_span(self.space.t, self.space.p - 1, self.x_int[i])

        # Knot span indices at quadrature points between x_int in format (i, iq)
        self.span_ptsG_N = xp.zeros(self.ptsG.shape, dtype=int)
        self.span_ptsG_D = xp.zeros(self.ptsG.shape, dtype=int)
        for i in range(self.ptsG.shape[0]):
            for iq in range(self.ptsG.shape[1]):
                self.span_ptsG_N[i, iq] = bsp.find_span(self.space.T, self.space.p, self.ptsG[i, iq])
                self.span_ptsG_D[i, iq] = bsp.find_span(self.space.t, self.space.p - 1, self.ptsG[i, iq])

        # Values of p + 1 non-zero basis functions at Greville points in format (greville, 0, basis function)
        self.basis_x_int_N = xp.zeros((*self.x_int[:, None].shape, self.space.p + 1), dtype=float)
        self.basis_x_int_D = xp.zeros((*self.x_int[:, None].shape, self.space.p), dtype=float)

        N_temp = bsp.basis_ders_on_quad_grid(self.space.T, self.space.p, self.x_int[:, None], 0, normalize=False)
        D_temp = bsp.basis_ders_on_quad_grid(self.space.t, self.space.p - 1, self.x_int[:, None], 0, normalize=True)

        for i in range(self.x_int.shape[0]):
            for b in range(self.space.p + 1):
                self.basis_x_int_N[i, 0, b] = N_temp[i, b, 0, 0]
            for b in range(self.space.p):
                self.basis_x_int_D[i, 0, b] = D_temp[i, b, 0, 0]

        # Values of p + 1 non-zero basis functions at quadrature points points between x_int in format (i, iq, basis function)
        self.basis_ptsG_N = xp.zeros((*self.ptsG.shape, self.space.p + 1), dtype=float)
        self.basis_ptsG_D = xp.zeros((*self.ptsG.shape, self.space.p), dtype=float)

        N_temp = bsp.basis_ders_on_quad_grid(self.space.T, self.space.p, self.ptsG, 0, normalize=False)
        D_temp = bsp.basis_ders_on_quad_grid(self.space.t, self.space.p - 1, self.ptsG, 0, normalize=True)

        for i in range(self.ptsG.shape[0]):
            for iq in range(self.ptsG.shape[1]):
                for b in range(self.space.p + 1):
                    self.basis_ptsG_N[i, iq, b] = N_temp[i, b, 0, iq]
                for b in range(self.space.p):
                    self.basis_ptsG_D[i, iq, b] = D_temp[i, b, 0, iq]

        # quadrature matrix for performing integrations as matrix-vector products
        self.Q = xp.zeros((spline_space.NbaseD, self.wts.shape[0] * self.n_quad), dtype=float)

        for i in range(spline_space.NbaseD):
            for j in range(self.subs[i]):
                ie = i + j + self.subs_cum[i]
                self.Q[i, self.n_quad * ie : self.n_quad * (ie + 1)] = self.wts[ie]

        self.Q = spa.csr_matrix(self.Q)

        # quadrature matrix for performing integrations as matrix-vector products, ignoring subs (less accurate integration for even degree)
        self.QG = xp.zeros((spline_space.NbaseD, self.wtsG.shape[0] * self.n_quad), dtype=float)

        for i in range(spline_space.NbaseD):
            self.QG[i, self.n_quad * i : self.n_quad * (i + 1)] = self.wtsG[i]

        self.QG = spa.csr_matrix(self.QG)

        # collocation matrices for B-/M-splines at interpolation/quadrature points
        BM_splines = [False, True]

        self.N_int = bsp.collocation_matrix(
            spline_space.T,
            spline_space.p - 0,
            self.x_int,
            spline_space.spl_kind,
            BM_splines[0],
        )
        self.D_int = bsp.collocation_matrix(
            spline_space.t,
            spline_space.p - 1,
            self.x_int,
            spline_space.spl_kind,
            BM_splines[1],
        )

        self.N_int[self.N_int < 1e-12] = 0.0
        self.D_int[self.D_int < 1e-12] = 0.0

        self.N_int = spa.csr_matrix(self.N_int)
        self.D_int = spa.csr_matrix(self.D_int)

        self.N_pts = bsp.collocation_matrix(
            spline_space.T,
            spline_space.p - 0,
            self.pts.flatten(),
            spline_space.spl_kind,
            BM_splines[0],
        )
        self.D_pts = bsp.collocation_matrix(
            spline_space.t,
            spline_space.p - 1,
            self.pts.flatten(),
            spline_space.spl_kind,
            BM_splines[1],
        )

        self.N_pts = spa.csr_matrix(self.N_pts)
        self.D_pts = spa.csr_matrix(self.D_pts)

        # interpolation matrices
        self.I = self.N_int.copy()
        self.ID = self.D_int.copy()
        self.I0 = self.space.B0.dot(self.I.dot(self.space.B0.T)).tocsr()

        # histopolation matrices
        self.H = self.Q.dot(self.D_pts)
        self.HN = self.Q.dot(self.N_pts)
        self.H0 = self.space.B1.dot(self.H.dot(self.space.B1.T)).tocsr()

        # LU decompositions
        self.I_LU = spa.linalg.splu(self.I.tocsc())
        self.H_LU = spa.linalg.splu(self.H.tocsc())

        self.I0_LU = spa.linalg.splu(self.I0.tocsc())
        self.H0_LU = spa.linalg.splu(self.H0.tocsc())

        self.I_T_LU = spa.linalg.splu(self.I.T.tocsc())
        self.H_T_LU = spa.linalg.splu(self.H.T.tocsc())

        self.I0_T_LU = spa.linalg.splu(self.I0.T.tocsc())
        self.H0_T_LU = spa.linalg.splu(self.H0.T.tocsc())

    # degrees of freedoms: V_0 --> R^n

    def dofs_0(self, fun):
        """
        Returns the degrees of freedom for functions in V_0: dofs_0[i] = fun(x_i).
        """

        dofs = fun(self.x_int)

        return dofs

    # degrees of freedom: V_1 --> R^n
    def dofs_1(self, fun, with_subs=True):
        """
        Returns the degrees of freedom for functions in V_1: dofs_1[i] = int_(eta_i)^(eta_i + 1) fun(eta) deta.
        """

        if with_subs:
            dofs = self.Q.dot(fun(self.pts.flatten()))
        else:
            dofs = self.QG.dot(fun(self.ptsG.flatten()))

        return dofs

    # projector pi_0: V_0 --> R^n (callable in V_0 as input)
    def pi_0(self, fun):
        """
        Returns the solution of the interpolation problem I.coeffs = dofs_0 (spline coefficients).
        """

        coeffs = self.I_LU.solve(self.dofs_0(fun))

        return coeffs

    # projector pi_1: V_1 --> R^n (callable in V_1 as input)
    def pi_1(self, fun, with_subs=True):
        """
        Returns the solution of the interpolation problem H.coeffs = dofs_1 (spline coefficients).
        """

        coeffs = self.H_LU.solve(self.dofs_1(fun, with_subs))

        return coeffs

    # projector pi_0: R^n --> R^n (dofs_0 as input)
    def pi_0_mat(self, dofs_0):
        """
        Returns the solution of the interpolation problem I.coeffs = dofs_0 (spline coefficients).
        """

        coeffs = self.I_LU.solve(dofs_0)

        return coeffs

    # projector pi_1: R^n --> R^n (dofs_1 as input)
    def pi_1_mat(self, dofs_1):
        """
        Returns the solution of the interpolation problem H.coeffs = dofs_1 (spline coefficients).
        """

        coeffs = self.H_LU.solve(dofs_1)

        return coeffs

    # degrees of freedom of products of basis functions

    def dofs_1d_bases_products(self, space):
        """
        DISCLAIMER: this routine is not finished and should not be used.

        Computes indices of non-vanishing degrees of freedom of products of basis functions:

        dofs_0_i(N_j*N_k),
        dofs_0_i(D_j*N_k),
        dofs_0_i(N_j*D_k),
        dofs_0_i(D_j*D_k),

        dofs_1_i(N_j*N_k),
        dofs_1_i(D_j*N_k),
        dofs_1_i(N_j*D_k),
        dofs_1_i(D_j*D_k).
        """

        dofs_0_NN = xp.empty((space.NbaseN, space.NbaseN, space.NbaseN), dtype=float)
        dofs_0_DN = xp.empty((space.NbaseN, space.NbaseD, space.NbaseN), dtype=float)
        dofs_0_DD = xp.empty((space.NbaseN, space.NbaseD, space.NbaseD), dtype=float)

        dofs_1_NN = xp.empty((space.NbaseD, space.NbaseN, space.NbaseN), dtype=float)
        dofs_1_DN = xp.empty((space.NbaseD, space.NbaseD, space.NbaseN), dtype=float)
        dofs_1_DD = xp.empty((space.NbaseD, space.NbaseD, space.NbaseD), dtype=float)

        # ========= dofs_0_NN and dofs_1_NN ==============
        cj = xp.zeros(space.NbaseN, dtype=float)
        ck = xp.zeros(space.NbaseN, dtype=float)

        for j in range(space.NbaseN):
            for k in range(space.NbaseN):
                cj[:] = 0.0
                ck[:] = 0.0

                cj[j] = 1.0
                ck[k] = 1.0

                def N_jN_k(eta):
                    return space.evaluate_N(eta, cj) * space.evaluate_N(eta, ck)

                dofs_0_NN[:, j, k] = self.dofs_0(N_jN_k)
                dofs_1_NN[:, j, k] = self.dofs_1(N_jN_k)

        # ========= dofs_0_DN and dofs_1_DN ==============
        cj = xp.zeros(space.NbaseD, dtype=float)
        ck = xp.zeros(space.NbaseN, dtype=float)

        for j in range(space.NbaseD):
            for k in range(space.NbaseN):
                cj[:] = 0.0
                ck[:] = 0.0

                cj[j] = 1.0
                ck[k] = 1.0

                def D_jN_k(eta):
                    return space.evaluate_D(eta, cj) * space.evaluate_N(eta, ck)

                dofs_0_DN[:, j, k] = self.dofs_0(D_jN_k)
                dofs_1_DN[:, j, k] = self.dofs_1(D_jN_k)

        # ========= dofs_0_DD and dofs_1_DD =============
        cj = xp.zeros(space.NbaseD, dtype=float)
        ck = xp.zeros(space.NbaseD, dtype=float)

        for j in range(space.NbaseD):
            for k in range(space.NbaseD):
                cj[:] = 0.0
                ck[:] = 0.0

                cj[j] = 1.0
                ck[k] = 1.0

                def D_jD_k(eta):
                    return space.evaluate_D(eta, cj) * space.evaluate_D(eta, ck)

                dofs_0_DD[:, j, k] = self.dofs_0(D_jD_k)
                dofs_1_DD[:, j, k] = self.dofs_1(D_jD_k)

        dofs_0_ND = xp.transpose(dofs_0_DN, (0, 2, 1))
        dofs_1_ND = xp.transpose(dofs_1_DN, (0, 2, 1))

        # find non-zero entries
        dofs_0_NN_indices = xp.nonzero(dofs_0_NN)
        dofs_0_DN_indices = xp.nonzero(dofs_0_DN)
        dofs_0_ND_indices = xp.nonzero(dofs_0_ND)
        dofs_0_DD_indices = xp.nonzero(dofs_0_DD)

        dofs_1_NN_indices = xp.nonzero(dofs_1_NN)
        dofs_1_DN_indices = xp.nonzero(dofs_1_DN)
        dofs_1_ND_indices = xp.nonzero(dofs_1_ND)
        dofs_1_DD_indices = xp.nonzero(dofs_1_DD)

        dofs_0_NN_i_red = xp.empty(dofs_0_NN_indices[0].size, dtype=int)
        dofs_0_DN_i_red = xp.empty(dofs_0_DN_indices[0].size, dtype=int)
        dofs_0_ND_i_red = xp.empty(dofs_0_ND_indices[0].size, dtype=int)
        dofs_0_DD_i_red = xp.empty(dofs_0_DD_indices[0].size, dtype=int)

        dofs_1_NN_i_red = xp.empty(dofs_1_NN_indices[0].size, dtype=int)
        dofs_1_DN_i_red = xp.empty(dofs_1_DN_indices[0].size, dtype=int)
        dofs_1_ND_i_red = xp.empty(dofs_1_ND_indices[0].size, dtype=int)
        dofs_1_DD_i_red = xp.empty(dofs_1_DD_indices[0].size, dtype=int)

        # ================================
        nv = space.NbaseN * dofs_0_NN_indices[1] + dofs_0_NN_indices[2]
        un = xp.unique(nv)

        for i in range(dofs_0_NN_indices[0].size):
            dofs_0_NN_i_red[i] = xp.nonzero(un == nv[i])[0]

        # ================================
        nv = space.NbaseN * dofs_0_DN_indices[1] + dofs_0_DN_indices[2]
        un = xp.unique(nv)

        for i in range(dofs_0_DN_indices[0].size):
            dofs_0_DN_i_red[i] = xp.nonzero(un == nv[i])[0]

        # ================================
        nv = space.NbaseD * dofs_0_ND_indices[1] + dofs_0_ND_indices[2]
        un = xp.unique(nv)

        for i in range(dofs_0_ND_indices[0].size):
            dofs_0_ND_i_red[i] = xp.nonzero(un == nv[i])[0]

        # ================================
        nv = space.NbaseD * dofs_0_DD_indices[1] + dofs_0_DD_indices[2]
        un = xp.unique(nv)

        for i in range(dofs_0_DD_indices[0].size):
            dofs_0_DD_i_red[i] = xp.nonzero(un == nv[i])[0]

        # ================================
        nv = space.NbaseN * dofs_1_NN_indices[1] + dofs_1_NN_indices[2]
        un = xp.unique(nv)

        for i in range(dofs_1_NN_indices[0].size):
            dofs_1_NN_i_red[i] = xp.nonzero(un == nv[i])[0]

        # ================================
        nv = space.NbaseN * dofs_1_DN_indices[1] + dofs_1_DN_indices[2]
        un = xp.unique(nv)

        for i in range(dofs_1_DN_indices[0].size):
            dofs_1_DN_i_red[i] = xp.nonzero(un == nv[i])[0]

        # ================================
        nv = space.NbaseD * dofs_1_ND_indices[1] + dofs_1_ND_indices[2]
        un = xp.unique(nv)

        for i in range(dofs_1_ND_indices[0].size):
            dofs_1_ND_i_red[i] = xp.nonzero(un == nv[i])[0]

        # ================================
        nv = space.NbaseD * dofs_1_DD_indices[1] + dofs_1_DD_indices[2]
        un = xp.unique(nv)

        for i in range(dofs_1_DD_indices[0].size):
            dofs_1_DD_i_red[i] = xp.nonzero(un == nv[i])[0]

        dofs_0_NN_indices = xp.vstack(
            (dofs_0_NN_indices[0], dofs_0_NN_indices[1], dofs_0_NN_indices[2], dofs_0_NN_i_red),
        )
        dofs_0_DN_indices = xp.vstack(
            (dofs_0_DN_indices[0], dofs_0_DN_indices[1], dofs_0_DN_indices[2], dofs_0_DN_i_red),
        )
        dofs_0_ND_indices = xp.vstack(
            (dofs_0_ND_indices[0], dofs_0_ND_indices[1], dofs_0_ND_indices[2], dofs_0_ND_i_red),
        )
        dofs_0_DD_indices = xp.vstack(
            (dofs_0_DD_indices[0], dofs_0_DD_indices[1], dofs_0_DD_indices[2], dofs_0_DD_i_red),
        )

        dofs_1_NN_indices = xp.vstack(
            (dofs_1_NN_indices[0], dofs_1_NN_indices[1], dofs_1_NN_indices[2], dofs_1_NN_i_red),
        )
        dofs_1_DN_indices = xp.vstack(
            (dofs_1_DN_indices[0], dofs_1_DN_indices[1], dofs_1_DN_indices[2], dofs_1_DN_i_red),
        )
        dofs_1_ND_indices = xp.vstack(
            (dofs_1_ND_indices[0], dofs_1_ND_indices[1], dofs_1_ND_indices[2], dofs_1_ND_i_red),
        )
        dofs_1_DD_indices = xp.vstack(
            (dofs_1_DD_indices[0], dofs_1_DD_indices[1], dofs_1_DD_indices[2], dofs_1_DD_i_red),
        )

        return (
            dofs_0_NN_indices,
            dofs_0_DN_indices,
            dofs_0_ND_indices,
            dofs_0_DD_indices,
            dofs_1_NN_indices,
            dofs_1_DN_indices,
            dofs_1_ND_indices,
            dofs_1_DD_indices,
        )


# ============= 2d for pure tensor product splines ========================
class Projectors_tensor_2d:
    """
    Global commuting projectors pi_0, pi_1, pi_2 in 2d corresponding to the sequence grad = [d_1 f, d2_f] --> curl = [d_1 f_2 - d_2 f_1].

    Parameters
    ----------
    proj_1d : list of two "Projectors_global_1d" objects

    Attributes
    ----------
    TODO
    """

    def __init__(self, proj_1d):
        self.pts_PI = {"0": None, "11": None, "12": None, "2": None}

        # collection of the point sets for different 2D projectors
        self.pts_PI["0"] = [proj_1d[0].x_int, proj_1d[1].x_int]

        self.pts_PI["11"] = [proj_1d[0].pts.flatten(), proj_1d[1].x_int]

        self.pts_PI["12"] = [proj_1d[0].x_int, proj_1d[1].pts.flatten()]

        self.pts_PI["2"] = [proj_1d[0].pts.flatten(), proj_1d[1].pts.flatten()]

        self.Q1 = proj_1d[0].Q
        self.Q2 = proj_1d[1].Q

        self.n1 = proj_1d[0].I.shape[1]
        self.n2 = proj_1d[1].I.shape[1]

        self.d1 = proj_1d[0].H.shape[1]
        self.d2 = proj_1d[1].H.shape[1]

        self.I_LU1 = proj_1d[0].I_LU
        self.I_LU2 = proj_1d[1].I_LU

        self.H_LU1 = proj_1d[0].H_LU
        self.H_LU2 = proj_1d[1].H_LU

    # ======================================

    def eval_for_PI(self, comp, fun):
        """
        Evaluate the callable fun at the points corresponding to the projector comp.

        Parameters
        ----------
        comp : string
            Which projector: '0', '11', '12' or '2'.

        fun : callable
            fun(eta1, eta2).

        Returns
        -------
        fun(pts1, pts2) : 2d numpy array
            Function evaluated at point set needed for the chosen projector.
        """

        pts_PI = self.pts_PI[comp]

        pts1, pts2 = xp.meshgrid(pts_PI[0], pts_PI[1], indexing="ij")
        # pts1, pts2 = xp.meshgrid(pts_PI[0], pts_PI[1], indexing='ij', sparse=True) # numpy >1.7

        return fun(pts1, pts2)

    # ======================================

    def dofs(self, comp, mat_f):
        """
        Compute the degrees of freedom for the projector comp.

        Parameters
        ----------
        comp: string
            Which projector: '0', '11', '12' or '2'.

        mat_f : 2d numpy array
            Function values f(eta1_i, eta2_j) at the points set of the projector (from eval_for_PI).

        Returns
        -------
        dofs : 2d numpy array
            The degrees of freedom sigma_ij.
        """

        assert mat_f.shape == (self.pts_PI[comp][0].size, self.pts_PI[comp][1].size)

        if comp == "0":
            dofs = kron_matvec_2d([spa.identity(mat_f.shape[0]), spa.identity(mat_f.shape[1])], mat_f)

        elif comp == "11":
            dofs = kron_matvec_2d([self.Q1, spa.identity(mat_f.shape[1])], mat_f)
        elif comp == "12":
            dofs = kron_matvec_2d([spa.identity(mat_f.shape[0]), self.Q2], mat_f)

        elif comp == "2":
            dofs = kron_matvec_2d([self.Q1, self.Q2], mat_f)
        else:
            raise ValueError("wrong projector specified")

        return dofs

    # ======================================

    def PI_mat(self, comp, dofs):
        """
        Kronecker solve of the projection problem I.coeffs = dofs.

        Parameters
        ----------
        comp : string
            Which projector: '0', '11', '12' or '2'.

        dofs : 2d numpy array
            The degrees of freedom sigma_ij.

        Returns
        -------
        coeffs : 2d numpy array
            The spline coefficients c_ij obtained by projection.
        """

        if comp == "0":
            assert dofs.shape == (self.n1, self.n2)
            coeffs = kron_lusolve_2d([self.I_LU1, self.I_LU2], dofs)
        elif comp == "11":
            assert dofs.shape == (self.d1, self.n2)
            coeffs = kron_lusolve_2d([self.H_LU1, self.I_LU2], dofs)
        elif comp == "12":
            assert dofs.shape == (self.n1, self.d2)
            coeffs = kron_lusolve_2d([self.I_LU1, self.H_LU2], dofs)
        elif comp == "2":
            assert dofs.shape == (self.d1, self.d2)
            coeffs = kron_lusolve_2d([self.H_LU1, self.H_LU2], dofs)
        else:
            raise ValueError("wrong projector specified")

        return coeffs

    # ======================================

    def PI(self, comp, fun):
        """
        De Rham commuting projectors.

        Parameters
        ----------
        comp : string
            Which projector: '0', '11', '12' or '2'.

        fun : callable
            fun(eta1, eta2).

        Returns
        -------
        coeffs : 2d numpy array
            The spline coefficients c_ij obtained by projection.
        """

        mat_f = self.eval_for_PI(comp, fun)
        dofs = self.dofs(comp, mat_f)

        if comp == "0":
            assert dofs.shape == (self.n1, self.n2)
            coeffs = kron_lusolve_2d([self.I_LU1, self.I_LU2], dofs)
        elif comp == "11":
            assert dofs.shape == (self.d1, self.n2)
            coeffs = kron_lusolve_2d([self.H_LU1, self.I_LU2], dofs)
        elif comp == "12":
            assert dofs.shape == (self.n1, self.d2)
            coeffs = kron_lusolve_2d([self.I_LU1, self.H_LU2], dofs)
        elif comp == "2":
            assert dofs.shape == (self.d1, self.d2)
            coeffs = kron_lusolve_2d([self.H_LU1, self.H_LU2], dofs)
        else:
            raise ValueError("wrong projector specified")

        return coeffs

    # ======================================

    def PI_0(self, fun):
        """
        De Rham commuting projector Pi_0.

        Parameters
        ----------
        fun : callable
            Element in V_0 continuous space, fun(eta1, eta2).

        Returns
        -------
        coeffs : 2d numpy array
            The spline coefficients c_ij obtained by projection.
        """

        coeffs = self.PI("0", fun)

        return coeffs

    # ======================================

    def PI_1(self, fun1, fun2):
        """
        De Rham commuting projector Pi_1 acting on fun = (fun1, fun2) in V_1.

        Parameters
        ----------
        fun1 : callable
            First component of element in V_1 continuous space, fun1(eta1, eta2).
        fun2 : callable
            Second component of element in V_1 continuous space, fun2(eta1, eta2).

        Returns
        -------
        coeffs1 : 2d numpy array
            The spline coefficients c_ij obtained by projection of fun1 on DN.
        coeffs2 : 2d numpy array
            The spline coefficients c_ij obtained by projection of fun2 on ND.
        """

        coeffs1 = self.PI("11", fun1)
        coeffs2 = self.PI("12", fun2)

        return coeffs1, coeffs2

    # ======================================

    def PI_2(self, fun):
        """
        De Rham commuting projector Pi_2.

        Parameters
        ----------
        fun : callable
            Element in V_2 continuous space, fun(eta1, eta2).

        Returns
        -------
        coeffs : 2d numpy array
            The spline coefficients c_ij obtained by projection.
        """

        coeffs = self.PI("2", fun)

        return coeffs


# ============== 3d for pure tensor product splines =======================
class Projectors_tensor_3d:
    """
    Global commuting projectors pi_0, pi_1, pi_2, pi_3 in 3d corresponding to the sequence grad --> curl --> div.

    Parameters
    ----------
    proj_1d : list of three "projectors_global_1d" objects

    Attributes
    ----------
    TODO
    """

    def __init__(self, proj_1d):
        self.pts_PI = {"0": None, "11": None, "12": None, "13": None, "21": None, "22": None, "23": None, "3": None}

        # collection of the point sets for different 2D projectors
        self.pts_PI["0"] = [proj_1d[0].x_int, proj_1d[1].x_int, proj_1d[2].x_int]

        self.pts_PI["11"] = [proj_1d[0].pts.flatten(), proj_1d[1].x_int, proj_1d[2].x_int]

        self.pts_PI["12"] = [proj_1d[0].x_int, proj_1d[1].pts.flatten(), proj_1d[2].x_int]

        self.pts_PI["13"] = [proj_1d[0].x_int, proj_1d[1].x_int, proj_1d[2].pts.flatten()]

        self.pts_PI["21"] = [proj_1d[0].x_int, proj_1d[1].pts.flatten(), proj_1d[2].pts.flatten()]

        self.pts_PI["22"] = [proj_1d[0].pts.flatten(), proj_1d[1].x_int, proj_1d[2].pts.flatten()]

        self.pts_PI["23"] = [proj_1d[0].pts.flatten(), proj_1d[1].pts.flatten(), proj_1d[2].x_int]

        self.pts_PI["3"] = [proj_1d[0].pts.flatten(), proj_1d[1].pts.flatten(), proj_1d[2].pts.flatten()]

        self.Q1 = proj_1d[0].Q
        self.Q2 = proj_1d[1].Q
        self.Q3 = proj_1d[2].Q

        self.n1 = proj_1d[0].I.shape[1]
        self.n2 = proj_1d[1].I.shape[1]
        self.n3 = proj_1d[2].I.shape[1]

        self.d1 = proj_1d[0].H.shape[1]
        self.d2 = proj_1d[1].H.shape[1]
        self.d3 = proj_1d[2].H.shape[1]

        self.I_LU1 = proj_1d[0].I_LU
        self.I_LU2 = proj_1d[1].I_LU
        self.I_LU3 = proj_1d[2].I_LU

        self.H_LU1 = proj_1d[0].H_LU
        self.H_LU2 = proj_1d[1].H_LU
        self.H_LU3 = proj_1d[2].H_LU

    # ======================================

    def eval_for_PI(self, comp, fun):
        """
        Evaluate the callable fun at the points corresponding to the projector comp.

        Parameters
        ----------
        comp: string
            Which projector: '0', '11', '12', '13', '21', '22', '23' or '3'.

        fun : callable
            fun(eta1, eta2, eta3)

        Returns
        -------
        fun(pts1, pts2, pts3) : 3d numpy array
            Function evaluated at point set needed for chosen projector.
        """

        pts_PI = self.pts_PI[comp]

        pts1, pts2, pts3 = xp.meshgrid(pts_PI[0], pts_PI[1], pts_PI[2], indexing="ij")
        # pts1, pts2, pts3 = xp.meshgrid(pts_PI[0], pts_PI[1], pts_PI[2], indexing='ij', sparse=True) # numpy >1.7

        return fun(pts1, pts2, pts3)

    # ======================================
    # def dofs_kernel(self, comp, mat_f):
    #    """
    #    Compute the degrees of freedom (rhs) for the projector comp.
    #
    #    Parameters
    #    ----------
    #    comp: str
    #        Which projector: '0', '11', '12', '13', '21', '22', '23' or '3'.
    #
    #    mat_f : 3d numpy array
    #        Function values f(eta1_i, eta2_j, eta3_k) at the points set of the projector (from eval_for_PI).
    #
    #    Returns
    #    -------
    #    rhs : 3d numpy array
    #        The degrees of freedom sigma_ijk.
    #    """
    #
    #    assert mat_f.shape==(self.pts_PI[comp][0].size,
    #                         self.pts_PI[comp][1].size,
    #                         self.pts_PI[comp][2].size
    #                         )
    #
    #    if comp=='0':
    #        rhs = mat_f
    #
    #    elif comp=='11':
    #        rhs = xp.empty( (self.d1, self.n2, self.n3) )
    #
    #        ker_glob.kernel_int_3d_eta1(self.subs1, self.subs_cum1, self.wts1,
    #                                    mat_f.reshape(self.ne1, self.nq1, self.n2, self.n3), rhs
    #                                    )
    #    elif comp=='12':
    #        rhs = xp.empty( (self.n1, self.d2, self.n3) )
    #
    #        ker_glob.kernel_int_3d_eta2(self.subs2, self.subs_cum2, self.wts2,
    #                                    mat_f.reshape(self.n1, self.ne2, self.nq2, self.n3), rhs
    #                                    )
    #    elif comp=='13':
    #        rhs = xp.empty( (self.n1, self.n2, self.d3) )
    #
    #        ker_glob.kernel_int_3d_eta3(self.subs3, self.subs_cum3, self.wts3,
    #                                    mat_f.reshape(self.n1, self.n2, self.ne3, self.nq3), rhs
    #                                    )
    #    elif comp=='21':
    #        rhs = xp.empty( (self.n1, self.d2, self.d3) )
    #
    #        ker_glob.kernel_int_3d_eta2_eta3(self.subs2, self.subs3,
    #                                         self.subs_cum2, self.subs_cum3,
    #                                         self.wts2, self.wts3,
    #              mat_f.reshape(self.n1, self.ne2, self.nq2, self.ne3, self.nq3), rhs
    #                                             )
    #    elif comp=='22':
    #        rhs = xp.empty( (self.d1, self.n2, self.d3) )
    #
    #        ker_glob.kernel_int_3d_eta1_eta3(self.subs1, self.subs3,
    #                                         self.subs_cum1, self.subs_cum3,
    #                                         self.wts1, self.wts3,
    #              mat_f.reshape(self.ne1, self.nq1, self.n2, self.ne3, self.nq3), rhs
    #                                             )
    #    elif comp=='23':
    #        rhs = xp.empty( (self.d1, self.d2, self.n3) )
    #
    #        ker_glob.kernel_int_3d_eta1_eta2(self.subs1, self.subs2,
    #                                         self.subs_cum1, self.subs_cum2,
    #                                         self.wts1, self.wts2,
    #              mat_f.reshape(self.ne1, self.nq1, self.ne2, self.nq2, self.n3), rhs
    #                                             )
    #    elif comp=='3':
    #        rhs = xp.empty( (self.d1, self.d2, self.d3) )
    #
    #        ker_glob.kernel_int_3d_eta1_eta2_eta3(self.subs1, self.subs2, self.subs3,
    #                                              self.subs_cum1, self.subs_cum2, self.subs_cum3,
    #                                              self.wts1, self.wts2, self.wts3,
    #              mat_f.reshape(self.ne1, self.nq1, self.ne2, self.nq2, self.ne3, self.nq3), rhs
    #                                             )
    #    else:
    #        raise ValueError ("wrong projector specified")
    #
    #    return rhs

    # ======================================
    # def dofs_T_kernel(self, comp, mat_dofs):
    #    """
    #    Transpose of dofs
    #
    #    Parameters
    #    ----------
    #    comp: str
    #        Which projector: '0', '11', '12', '13', '21', '22', '23' or '3'.
    #
    #    mat_dofs : 3d numpy array
    #        Degrees of freedom.
    #
    #    Returns
    #    -------
    #    mat_pts : numpy array
    #        comp == '0' 3d of the form(n1, n2, n3)
    #        comp == '11' 4d of the form(ne1, nq1, n2, n3)
    #        comp == '12' 4d of the form(n1, n2, nq2, n3)
    #        comp == '13' 4d of the form(n1, n2, n3, nq3)
    #        comp == '21' 5d of the form(n1, ne2, nq2, ne3, nq3)
    #        comp == '22' 5d of the form(ne1, nq1, n2, ne3, nq3)
    #        comp == '23' 5d of the form(ne1, nq1, ne2, nq2, n3)
    #        comp == '3' 6d of the form(ne1, nq1, ne2, nq2, n3, nq3)
    #
    #    '''
    #
    #    if comp=='0':
    #        rhs = mat_dofs
    #
    #    elif comp=='11':
    #        assert mat_dofs.shape == (self.d1, self.n2, self.n3)
    #        rhs = xp.empty( (self.ne1, self.nq1, self.n2, self.n3) )
    #
    #        ker_glob.kernel_int_3d_eta1_transpose(self.subs1, self.subs_cum1, self.wts1,
    #                                              mat_dofs, rhs)
    #
    #        rhs = rhs.reshape(self.ne1 * self.nq1, self.n2, self.n3)
    #
    #    elif comp=='12':
    #        assert mat_dofs.shape == (self.n1, self.d2, self.n3)
    #        rhs = xp.empty( (self.n1, self.ne2, self.nq2, self.n3) )
    #
    #        ker_glob.kernel_int_3d_eta2_transpose(self.subs2, self.subs_cum2, self.wts2,
    #                                              mat_dofs, rhs)
    #
    #        rhs = rhs.reshape(self.n1, self.ne2 * self.nq2, self.n3)
    #
    #    elif comp=='13':
    #        assert mat_dofs.shape == (self.n1, self.n2, self.d3)
    #        rhs = xp.empty( (self.n1, self.n2, self.ne3, self.nq3) )
    #
    #        ker_glob.kernel_int_3d_eta3_transpose(self.subs3, self.subs_cum3, self.wts3,
    #                                              mat_dofs, rhs)
    #
    #        rhs = rhs.reshape(self.n1, self.n2, self.ne3 * self.nq3)
    #
    #    elif comp=='21':
    #        assert mat_dofs.shape == (self.n1, self.d2, self.d3)
    #        rhs = xp.empty( (self.n1, self.ne2, self.nq2, self.ne3, self.nq3) )
    #
    #        ker_glob.kernel_int_3d_eta2_eta3_transpose(self.subs2, self.subs3,
    #                                         self.subs_cum2, self.subs_cum3,
    #                                         self.wts2, self.wts3,
    #                                         mat_dofs, rhs)
    #        rhs = rhs.reshape(self.n1, self.ne2 * self.nq2, self.ne3 * self.nq3)
    #
    #    elif comp=='22':
    #        assert mat_dofs.shape == (self.d1, self.n2, self.d3)
    #        rhs = xp.empty( (self.ne1, self.nq1, self.n2, self.ne3, self.nq3) )
    #
    #        ker_glob.kernel_int_3d_eta1_eta3_transpose(self.subs1, self.subs3,
    #                                         self.subs_cum1, self.subs_cum3,
    #                                         self.wts1, self.wts3,
    #                                         mat_dofs, rhs)
    #        rhs = rhs.reshape(self.ne1 * self.nq1, self.n2, self.ne3 * self.nq3)
    #
    #    elif comp=='23':
    #        assert mat_dofs.shape == (self.d1, self.d2, self.n3)
    #        rhs = xp.empty( (self.ne1, self.nq1, self.ne2, self.nq2, self.n3) )
    #
    #        ker_glob.kernel_int_3d_eta1_eta2_transpose(self.subs1, self.subs2,
    #                                         self.subs_cum1, self.subs_cum2,
    #                                         self.wts1, self.wts2,
    #                                         mat_dofs, rhs)
    #        rhs = rhs.reshape(self.ne1 * self.nq1, self.ne2 * self.nq2, self.n3)
    #
    #    elif comp=='3':
    #        assert mat_dofs.shape == (self.d1, self.d2, self.d3)
    #        rhs = xp.empty( (self.ne1, self.nq1, self.ne2, self.nq2, self.ne3, self.nq3) )
    #
    #        ker_glob.kernel_int_3d_eta1_eta2_eta3_transpose(self.subs1, self.subs2, self.subs3,
    #                                              self.subs_cum1, self.subs_cum2, self.subs_cum3,
    #                                              self.wts1, self.wts2, self.wts3,
    #                                              mat_dofs, rhs)
    #        rhs = rhs.reshape(self.ne1 * self.nq1, self.ne2 * self.nq2, self.ne3 * self.nq3)
    #
    #    else:
    #        raise ValueError ("wrong projector specified")
    #
    #    return rhs

    # ======================================

    def dofs(self, comp, mat_f):
        """
        Compute the degrees of freedom for the projector comp.

        Parameters
        ----------
        comp: string
            Which projector: '0', '11', '12', '13', '21', '22', '23' or '3'.

        mat_f : 3d numpy array
            Function values f(eta1_i, eta2_j, eta3_k) at the points set of the projector (from eval_for_PI).

        Returns
        -------
        dofs : 3d numpy array
            The degrees of freedom sigma_ijk.
        """

        assert mat_f.shape == (self.pts_PI[comp][0].size, self.pts_PI[comp][1].size, self.pts_PI[comp][2].size)

        if comp == "0":
            dofs = kron_matvec_3d(
                [spa.identity(mat_f.shape[0]), spa.identity(mat_f.shape[1]), spa.identity(mat_f.shape[2])],
                mat_f,
            )

        elif comp == "11":
            dofs = kron_matvec_3d([self.Q1, spa.identity(mat_f.shape[1]), spa.identity(mat_f.shape[2])], mat_f)
        elif comp == "12":
            dofs = kron_matvec_3d([spa.identity(mat_f.shape[0]), self.Q2, spa.identity(mat_f.shape[2])], mat_f)
        elif comp == "13":
            dofs = kron_matvec_3d([spa.identity(mat_f.shape[0]), spa.identity(mat_f.shape[1]), self.Q3], mat_f)

        elif comp == "21":
            dofs = kron_matvec_3d([spa.identity(mat_f.shape[0]), self.Q2, self.Q3], mat_f)
        elif comp == "22":
            dofs = kron_matvec_3d([self.Q1, spa.identity(mat_f.shape[1]), self.Q3], mat_f)
        elif comp == "23":
            dofs = kron_matvec_3d([self.Q1, self.Q2, spa.identity(mat_f.shape[2])], mat_f)

        elif comp == "3":
            dofs = kron_matvec_3d([self.Q1, self.Q2, self.Q3], mat_f)

        else:
            raise ValueError("wrong projector specified")

        return dofs

    # ======================================

    def dofs_T(self, comp, mat_dofs):
        """
        Transpose of degrees of freedom.

        Parameters
        ----------
        comp: str
            Which projector: '0', '11', '12', '13', '21', '22', '23' or '3'.

        mat_dofs : 3d numpy array
            Degrees of freedom.

        Returns
        -------
        rhs : 3d numpy array
            The degrees of freedom sigma_ijk.
        """

        if comp == "0":
            rhs = kron_matvec_3d(
                [spa.identity(mat_dofs.shape[0]), spa.identity(mat_dofs.shape[1]), spa.identity(mat_dofs.shape[2])],
                mat_dofs,
            )

        elif comp == "11":
            rhs = kron_matvec_3d(
                [self.Q1.T, spa.identity(mat_dofs.shape[1]), spa.identity(mat_dofs.shape[2])],
                mat_dofs,
            )
        elif comp == "12":
            rhs = kron_matvec_3d(
                [spa.identity(mat_dofs.shape[0]), self.Q2.T, spa.identity(mat_dofs.shape[2])],
                mat_dofs,
            )
        elif comp == "13":
            rhs = kron_matvec_3d(
                [spa.identity(mat_dofs.shape[0]), spa.identity(mat_dofs.shape[1]), self.Q3.T],
                mat_dofs,
            )

        elif comp == "21":
            rhs = kron_matvec_3d([spa.identity(mat_dofs.shape[0]), self.Q2.T, self.Q3.T], mat_dofs)
        elif comp == "22":
            rhs = kron_matvec_3d([self.Q1.T, spa.identity(mat_dofs.shape[1]), self.Q3.T], mat_dofs)
        elif comp == "23":
            rhs = kron_matvec_3d([self.Q1.T, self.Q2.T, spa.identity(mat_dofs.shape[2])], mat_dofs)

        elif comp == "3":
            rhs = kron_matvec_3d([self.Q1.T, self.Q2.T, self.Q3.T], mat_dofs)

        else:
            raise ValueError("wrong projector specified")

        return rhs

    # ======================================

    def PI_mat(self, comp, dofs):
        """
        Kronecker solve of the projection problem I.coeffs = dofs.

        Parameters
        ----------
        comp : string
            Which projector: '0', '11', '12', '13', '21', '22', '23' or '3'.

        dofs : 3d numpy array
            The degrees of freedom sigma_ijk.

        Returns
        -------
        coeffs : 3d numpy array
            The spline coefficients c_ijk obtained by projection.
        """

        if comp == "0":
            assert dofs.shape == (self.n1, self.n2, self.n3)
            coeffs = kron_lusolve_3d([self.I_LU1, self.I_LU2, self.I_LU3], dofs)

        elif comp == "11":
            assert dofs.shape == (self.d1, self.n2, self.n3)
            coeffs = kron_lusolve_3d([self.H_LU1, self.I_LU2, self.I_LU3], dofs)
        elif comp == "12":
            assert dofs.shape == (self.n1, self.d2, self.n3)
            coeffs = kron_lusolve_3d([self.I_LU1, self.H_LU2, self.I_LU3], dofs)
        elif comp == "13":
            assert dofs.shape == (self.n1, self.n2, self.d3)
            coeffs = kron_lusolve_3d([self.I_LU1, self.I_LU2, self.H_LU3], dofs)

        elif comp == "21":
            assert dofs.shape == (self.n1, self.d2, self.d3)
            coeffs = kron_lusolve_3d([self.I_LU1, self.H_LU2, self.H_LU3], dofs)
        elif comp == "22":
            assert dofs.shape == (self.d1, self.n2, self.d3)
            coeffs = kron_lusolve_3d([self.H_LU1, self.I_LU2, self.H_LU3], dofs)
        elif comp == "23":
            assert dofs.shape == (self.d1, self.d2, self.n3)
            coeffs = kron_lusolve_3d([self.H_LU1, self.H_LU2, self.I_LU3], dofs)

        elif comp == "3":
            assert dofs.shape == (self.d1, self.d2, self.d3)
            coeffs = kron_lusolve_3d([self.H_LU1, self.H_LU2, self.H_LU3], dofs)

        else:
            raise ValueError("wrong projector specified")

        return coeffs

    # ======================================

    def PI(self, comp, fun):
        """
        De Rham commuting projectors.

        Parameters
        ----------
        comp : string
            Which projector: '0', '11', '12', '13', '21', '22', '23' or '3'.

        fun : callable
            f(eta1, eta2, eta3).

        Returns
        -------
        coeffs : 3d numpy array
            The spline coefficients c_ijk obtained by projection.
        """

        mat_f = self.eval_for_PI(comp, fun)
        dofs = self.dofs(comp, mat_f)

        if comp == "0":
            assert dofs.shape == (self.n1, self.n2, self.n3)
            coeffs = kron_lusolve_3d([self.I_LU1, self.I_LU2, self.I_LU3], dofs)

        elif comp == "11":
            assert dofs.shape == (self.d1, self.n2, self.n3)
            coeffs = kron_lusolve_3d([self.H_LU1, self.I_LU2, self.I_LU3], dofs)
        elif comp == "12":
            assert dofs.shape == (self.n1, self.d2, self.n3)
            coeffs = kron_lusolve_3d([self.I_LU1, self.H_LU2, self.I_LU3], dofs)
        elif comp == "13":
            assert dofs.shape == (self.n1, self.n2, self.d3)
            coeffs = kron_lusolve_3d([self.I_LU1, self.I_LU2, self.H_LU3], dofs)

        elif comp == "21":
            assert dofs.shape == (self.n1, self.d2, self.d3)
            coeffs = kron_lusolve_3d([self.I_LU1, self.H_LU2, self.H_LU3], dofs)
        elif comp == "22":
            assert dofs.shape == (self.d1, self.n2, self.d3)
            coeffs = kron_lusolve_3d([self.H_LU1, self.I_LU2, self.H_LU3], dofs)
        elif comp == "23":
            assert dofs.shape == (self.d1, self.d2, self.n3)
            coeffs = kron_lusolve_3d([self.H_LU1, self.H_LU2, self.I_LU3], dofs)

        elif comp == "3":
            assert dofs.shape == (self.d1, self.d2, self.d3)
            coeffs = kron_lusolve_3d([self.H_LU1, self.H_LU2, self.H_LU3], dofs)

        else:
            raise ValueError("wrong projector specified")

        return coeffs

    # ======================================

    def PI_0(self, fun):
        """
        De Rham commuting projector Pi_0.

        Parameters
        ----------
        fun : callable
            Element in V_0 continuous space, f(eta1, eta2, eta3).

        Returns
        -------
        coeffs : 3d numpy array
            The spline coefficients c_ijk obtained by projection.
        """

        coeffs = self.PI("0", fun)

        return coeffs

    # ======================================

    def PI_1(self, fun1, fun2, fun3):
        """
        De Rham commuting projector Pi_1 acting on fun = (fun1, fun2, fun3) in V_1.

        Parameters
        ----------
        fun1 : callable
            1st component of element in V_1 continuous space, fun1(eta1, eta2, eta3).
        fun2 : callable
            2nd component of element in V_1 continuous space, fun2(eta1, eta2, eta3).
        fun3 : callable
            3rd component of element in V_1 continuous space, fun3(eta1, eta2, eta3).

        Returns
        -------
        coeffs1 : 3d numpy array
            The spline coefficients c_ijk obtained by projection of fun1 on DNN.
        coeffs2 : 3d numpy array
            The spline coefficients c_ijk obtained by projection of fun2 on NDN.
        coeffs3 : 3d numpy array
            The spline coefficients c_ijk obtained by projection of fun3 on NND.
        """

        coeffs1 = self.PI("11", fun1)
        coeffs2 = self.PI("12", fun2)
        coeffs3 = self.PI("13", fun3)

        return coeffs1, coeffs2, coeffs3

    # ======================================

    def PI_2(self, fun1, fun2, fun3):
        """
        De Rham commuting projector Pi_2 acting on fun = (fun1, fun2, fun3) in V_2.

        Parameters
        ----------
        fun1 : callable
            1st component of element in V_2 continuous space, fun1(eta1, eta2, eta3).
        fun2 : callable
            2nd component of element in V_2 continuous space, fun2(eta1, eta2, eta3).
        fun3 : callable
            3rd component of element in V_2 continuous space, fun3(eta1, eta2, eta3).

        Returns
        -------
        coeffs1 : 3d numpy array
            The spline coefficients c_ijk obtained by projection of fun1 on NDD.
        coeffs2 : 3d numpy array
            The spline coefficients c_ijk obtained by projection of fun2 on DND.
        coeffs3 : 3d numpy array
            The spline coefficients c_ijk obtained by projection of fun3 on DDN.
        """

        coeffs1 = self.PI("21", fun1)
        coeffs2 = self.PI("22", fun2)
        coeffs3 = self.PI("23", fun3)

        return coeffs1, coeffs2, coeffs3

    # ======================================

    def PI_3(self, fun):
        """
        De Rham commuting projector Pi_3.

        Parameters
        ----------
        fun : callable
            Element in V_3 continuous space, f(eta1, eta2, eta3).

        Returns
        -------
        coeffs : 3d numpy array
            The spline coefficients c_ijk obtained by projection.
        """

        coeffs = self.PI("3", fun)

        return coeffs


# ===============================================================
class ProjectorsGlobal3D:
    """
    Global commuting projectors in 3 dimensions.

    Parameters
    ----------
        tensor_space : Tensor_spline_space
            the 3d or (2d x Fourier) B-spline finite element space

    """

    def __init__(self, tensor_space):
        # assemble extraction operators P^k for degrees of freedom

        # ----------- standard tensor-product splines in eta_1 x eta_2 plane -----------
        if tensor_space.ck == -1:
            n1, n2 = tensor_space.NbaseN[:2]
            d1, d2 = tensor_space.NbaseD[:2]

            # with boundary dofs
            self.P0_pol = spa.identity(n1 * n2, dtype=float, format="csr")
            self.P1_pol = spa.identity(d1 * n2 + n1 * d2, dtype=float, format="csr")
            self.P2_pol = spa.identity(n1 * d2 + d1 * n2, dtype=float, format="csr")
            self.P3_pol = spa.identity(d1 * d2, dtype=float, format="csr")

            # without boundary dofs
            self.P0_pol_0 = tensor_space.B0_pol.dot(self.P0_pol).tocsr()
            self.P1_pol_0 = tensor_space.B1_pol.dot(self.P1_pol).tocsr()
            self.P2_pol_0 = tensor_space.B2_pol.dot(self.P2_pol).tocsr()
            self.P3_pol_0 = tensor_space.B3_pol.dot(self.P3_pol).tocsr()
        # ---------------------------------------------------------------------------------

        # ----------------- C^k polar splines in eta_1 x eta_2 plane ----------------------
        else:
            # with boundary dofs
            self.P0_pol = tensor_space.polar_splines.P0.copy()
            self.P1_pol = tensor_space.polar_splines.P1C.copy()
            self.P2_pol = tensor_space.polar_splines.P1D.copy()
            self.P3_pol = tensor_space.polar_splines.P2.copy()

            # without boundary dofs
            self.P0_pol_0 = tensor_space.B0_pol.dot(self.P0_pol).tocsr()
            self.P1_pol_0 = tensor_space.B1_pol.dot(self.P1_pol).tocsr()
            self.P2_pol_0 = tensor_space.B2_pol.dot(self.P2_pol).tocsr()
            self.P3_pol_0 = tensor_space.B3_pol.dot(self.P3_pol).tocsr()
        # ---------------------------------------------------------------------------------

        # 3D operators: with boundary dofs
        if tensor_space.dim == 2:
            self.P0 = self.P0_pol.copy()
            self.P1 = spa.bmat([[self.P1_pol, None], [None, self.P0_pol]], format="csr")
            self.P2 = spa.bmat([[self.P2_pol, None], [None, self.P3_pol]], format="csr")
            self.P3 = self.P3_pol.copy()

            self.P0 = spa.kron(self.P0, spa.identity(tensor_space.NbaseN[2]), format="csr")
            self.P1 = spa.kron(self.P1, spa.identity(tensor_space.NbaseN[2]), format="csr")
            self.P2 = spa.kron(self.P2, spa.identity(tensor_space.NbaseN[2]), format="csr")
            self.P3 = spa.kron(self.P3, spa.identity(tensor_space.NbaseN[2]), format="csr")

        else:
            n3 = tensor_space.NbaseN[2]
            d3 = tensor_space.NbaseD[2]

            self.P0 = spa.kron(self.P0_pol, spa.identity(n3), format="csr")
            self.P1 = spa.bmat(
                [[spa.kron(self.P1_pol, spa.identity(n3)), None], [None, spa.kron(self.P0_pol, spa.identity(d3))]],
                format="csr",
            )
            self.P2 = spa.bmat(
                [[spa.kron(self.P2_pol, spa.identity(d3)), None], [None, spa.kron(self.P3_pol, spa.identity(n3))]],
                format="csr",
            )
            self.P3 = spa.kron(self.P3_pol, spa.identity(d3), format="csr")

        # 3D operators: without boundary dofs
        self.P0_0 = tensor_space.B0.dot(self.P0).tocsr()
        self.P1_0 = tensor_space.B1.dot(self.P1).tocsr()
        self.P2_0 = tensor_space.B2.dot(self.P2).tocsr()
        self.P3_0 = tensor_space.B3.dot(self.P3).tocsr()

        # if tensor_space.ck == 1:
        #
        #    # blocks of I0 matrix
        #    self.I0_11 = spa.kron(self.projectors_1d[0].N[:2, :2], self.projectors_1d[1].N)
        #    self.I0_11 = tensor_space.polar_splines.P0_11.dot(self.I0_11.dot(tensor_space.polar_splines.E0_11.T)).tocsr()
        #
        #    self.I0_12 = spa.kron(self.projectors_1d[0].N[:2, 2:], self.projectors_1d[1].N)
        #    self.I0_12 = tensor_space.polar_splines.P0_11.dot(self.I0_12).tocsr()
        #
        #    self.I0_21 = spa.kron(self.projectors_1d[0].N[2:, :2], self.projectors_1d[1].N)
        #    self.I0_21 = self.I0_21.dot(tensor_space.polar_splines.E0_11.T).tocsr()
        #
        #    self.I0_22 = spa.kron(self.projectors_1d[0].N[2:, 2:], self.projectors_1d[1].N, format='csr')
        #
        #    self.I0_22_LUs = [spa.linalg.splu(self.projectors_1d[0].N[2:, 2:].tocsc()), self.projectors_1d[1].N_LU]

        # 2D interpolation/histopolation matrices in poloidal plane
        II = spa.kron(tensor_space.spaces[0].projectors.I, tensor_space.spaces[1].projectors.I, format="csr")
        HI = spa.kron(tensor_space.spaces[0].projectors.H, tensor_space.spaces[1].projectors.I, format="csr")
        IH = spa.kron(tensor_space.spaces[0].projectors.I, tensor_space.spaces[1].projectors.H, format="csr")
        HH = spa.kron(tensor_space.spaces[0].projectors.H, tensor_space.spaces[1].projectors.H, format="csr")

        HI_IH = spa.bmat([[HI, None], [None, IH]], format="csr")
        IH_HI = spa.bmat([[IH, None], [None, HI]], format="csr")

        # including boundary splines
        self.I0_pol = self.P0_pol.dot(II.dot(tensor_space.E0_pol.T)).tocsr()
        self.I1_pol = self.P1_pol.dot(HI_IH.dot(tensor_space.E1_pol.T)).tocsr()
        self.I2_pol = self.P2_pol.dot(IH_HI.dot(tensor_space.E2_pol.T)).tocsr()
        self.I3_pol = self.P3_pol.dot(HH.dot(tensor_space.E3_pol.T)).tocsr()

        # without boundary splines
        self.I0_pol_0 = self.P0_pol_0.dot(II.dot(tensor_space.E0_pol_0.T)).tocsr()
        self.I1_pol_0 = self.P1_pol_0.dot(HI_IH.dot(tensor_space.E1_pol_0.T)).tocsr()
        self.I2_pol_0 = self.P2_pol_0.dot(IH_HI.dot(tensor_space.E2_pol_0.T)).tocsr()
        self.I3_pol_0 = self.P3_pol_0.dot(HH.dot(tensor_space.E3_pol_0.T)).tocsr()

        # LU decompositions in poloidal plane (including boundary splines)
        self.I0_pol_LU = spa.linalg.splu(self.I0_pol.tocsc())
        self.I1_pol_LU = spa.linalg.splu(self.I1_pol.tocsc())
        self.I2_pol_LU = spa.linalg.splu(self.I2_pol.tocsc())
        self.I3_pol_LU = spa.linalg.splu(self.I3_pol.tocsc())

        # LU decompositions in poloidal plane (without boundary splines)
        self.I0_pol_0_LU = spa.linalg.splu(self.I0_pol_0.tocsc())
        self.I1_pol_0_LU = spa.linalg.splu(self.I1_pol_0.tocsc())
        self.I2_pol_0_LU = spa.linalg.splu(self.I2_pol_0.tocsc())
        self.I3_pol_0_LU = spa.linalg.splu(self.I3_pol_0.tocsc())

        self.I0_pol_0_T_LU = spa.linalg.splu(self.I0_pol_0.T.tocsc())
        self.I1_pol_0_T_LU = spa.linalg.splu(self.I1_pol_0.T.tocsc())
        self.I2_pol_0_T_LU = spa.linalg.splu(self.I2_pol_0.T.tocsc())
        self.I3_pol_0_T_LU = spa.linalg.splu(self.I3_pol_0.T.tocsc())

        # whether approximate inverse interpolation matrices were computed already
        self.approx_Ik_0_inv = False
        self.approx_Ik_0_tol = -1.0

        # get 1D interpolation points
        x_i1 = tensor_space.spaces[0].projectors.x_int.copy()
        x_i2 = tensor_space.spaces[1].projectors.x_int.copy()

        # get 1D quadrature points
        x_q1 = tensor_space.spaces[0].projectors.pts.flatten()
        x_q2 = tensor_space.spaces[1].projectors.pts.flatten()

        x_q1G = tensor_space.spaces[0].projectors.ptsG.flatten()
        x_q2G = tensor_space.spaces[1].projectors.ptsG.flatten()

        # get 1D quadrature weight matrices
        self.Q1 = tensor_space.spaces[0].projectors.Q
        self.Q2 = tensor_space.spaces[1].projectors.Q

        self.Q1G = tensor_space.spaces[0].projectors.QG
        self.Q2G = tensor_space.spaces[1].projectors.QG

        # 1D interpolation/histopolation points and matrices in third direction
        if tensor_space.dim == 3:
            x_i3 = tensor_space.spaces[2].projectors.x_int
            x_q3 = tensor_space.spaces[2].projectors.pts.flatten()
            x_q3G = tensor_space.spaces[2].projectors.ptsG.flatten()

            self.Q3 = tensor_space.spaces[2].projectors.Q
            self.Q3G = tensor_space.spaces[2].projectors.QG

            self.I_tor = tensor_space.spaces[2].projectors.I
            self.H_tor = tensor_space.spaces[2].projectors.H

            self.I0_tor = tensor_space.spaces[2].projectors.I0
            self.H0_tor = tensor_space.spaces[2].projectors.H0

            self.I_tor_LU = tensor_space.spaces[2].projectors.I_LU
            self.H_tor_LU = tensor_space.spaces[2].projectors.H_LU

            self.I0_tor_LU = tensor_space.spaces[2].projectors.I0_LU
            self.H0_tor_LU = tensor_space.spaces[2].projectors.H0_LU

            self.I0_tor_T_LU = tensor_space.spaces[2].projectors.I0_T_LU
            self.H0_tor_T_LU = tensor_space.spaces[2].projectors.H0_T_LU

        else:
            if tensor_space.n_tor == 0:
                x_i3 = xp.array([0.0])
                x_q3 = xp.array([0.0])
                x_q3G = xp.array([0.0])

            else:
                if tensor_space.basis_tor == "r":
                    if tensor_space.n_tor > 0:
                        x_i3 = xp.array([1.0, 0.25 / tensor_space.n_tor])
                        x_q3 = xp.array([1.0, 0.25 / tensor_space.n_tor])
                        x_q3G = xp.array([1.0, 0.25 / tensor_space.n_tor])

                    else:
                        x_i3 = xp.array([1.0, 0.75 / (-tensor_space.n_tor)])
                        x_q3 = xp.array([1.0, 0.75 / (-tensor_space.n_tor)])
                        x_q3G = xp.array([1.0, 0.75 / (-tensor_space.n_tor)])

                else:
                    x_i3 = xp.array([0.0])
                    x_q3 = xp.array([0.0])
                    x_q3G = xp.array([0.0])

            self.Q3 = spa.identity(tensor_space.NbaseN[2], format="csr")
            self.Q3G = spa.identity(tensor_space.NbaseN[2], format="csr")

            self.I_tor = spa.identity(tensor_space.NbaseN[2], format="csr")
            self.H_tor = spa.identity(tensor_space.NbaseN[2], format="csr")

            self.I0_tor = spa.identity(tensor_space.NbaseN[2], format="csr")
            self.H0_tor = spa.identity(tensor_space.NbaseN[2], format="csr")

            self.I_tor_LU = spa.linalg.splu(self.I_tor.tocsc())
            self.H_tor_LU = spa.linalg.splu(self.H_tor.tocsc())

            self.I0_tor_LU = spa.linalg.splu(self.I0_tor.tocsc())
            self.H0_tor_LU = spa.linalg.splu(self.H0_tor.tocsc())

            self.I0_tor_T_LU = spa.linalg.splu(self.I0_tor.T.tocsc())
            self.H0_tor_T_LU = spa.linalg.splu(self.H0_tor.T.tocsc())

        # collection of the point sets for different projectors in poloidal plane
        self.pts_PI_0 = [x_i1, x_i2, x_i3]

        self.pts_PI_11 = [x_q1, x_i2, x_i3]
        self.pts_PI_12 = [x_i1, x_q2, x_i3]
        self.pts_PI_13 = [x_i1, x_i2, x_q3]

        self.pts_PI_21 = [x_i1, x_q2, x_q3]
        self.pts_PI_22 = [x_q1, x_i2, x_q3]
        self.pts_PI_23 = [x_q1, x_q2, x_i3]

        self.pts_PI_3 = [x_q1, x_q2, x_q3]

        # without subs
        self.pts_PI_0G = [x_i1, x_i2, x_i3]

        self.pts_PI_11G = [x_q1G, x_i2, x_i3]
        self.pts_PI_12G = [x_i1, x_q2G, x_i3]
        self.pts_PI_13G = [x_i1, x_i2, x_q3G]

        self.pts_PI_21G = [x_i1, x_q2G, x_q3G]
        self.pts_PI_22G = [x_q1G, x_i2, x_q3G]
        self.pts_PI_23G = [x_q1G, x_q2G, x_i3]

        self.pts_PI_3G = [x_q1G, x_q2G, x_q3G]

    # ========================================

    def getpts_for_PI(self, comp, with_subs=True):
        """
        Get the needed point sets for a given projector.

        Parameters
        ----------
        comp: int
            which projector, one of (0, 11, 12, 13, 21, 22, 23, 3).

        Returns
        -------
        pts_PI : list of 1d arrays
            the 1D point sets.
        """

        if with_subs:
            if comp == 0:
                pts_PI = self.pts_PI_0

            elif comp == 11:
                pts_PI = self.pts_PI_11
            elif comp == 12:
                pts_PI = self.pts_PI_12
            elif comp == 13:
                pts_PI = self.pts_PI_13

            elif comp == 21:
                pts_PI = self.pts_PI_21
            elif comp == 22:
                pts_PI = self.pts_PI_22
            elif comp == 23:
                pts_PI = self.pts_PI_23

            elif comp == 3:
                pts_PI = self.pts_PI_3

            else:
                raise ValueError("wrong projector specified")

        else:
            if comp == 0:
                pts_PI = self.pts_PI_0G

            elif comp == 11:
                pts_PI = self.pts_PI_11G
            elif comp == 12:
                pts_PI = self.pts_PI_12G
            elif comp == 13:
                pts_PI = self.pts_PI_13G

            elif comp == 21:
                pts_PI = self.pts_PI_21G
            elif comp == 22:
                pts_PI = self.pts_PI_22G
            elif comp == 23:
                pts_PI = self.pts_PI_23G

            elif comp == 3:
                pts_PI = self.pts_PI_3G

            else:
                raise ValueError("wrong projector specified")

        return pts_PI

    # ======================================

    def eval_for_PI(self, comp, fun, eval_kind, with_subs=True):
        """
        Evaluates the callable "fun" at the points corresponding to the projector, and returns the result as 3d array "mat_f".

        Parameters
        ----------
        comp: int
            which projector, one of (0, 11, 12, 13, 21, 22, 23, 3).

        fun : callable
            the function fun(eta1, eta2, eta3) to project

        eval_kind : string
            kind of function evaluation at interpolation/quadrature points ('meshgrid' or 'tensor_product', point-wise else)

        Returns
        -------
        mat_f : 3d array
            function evaluated on a 3d meshgrid contstructed from the 1d point sets.
        """

        assert callable(fun)

        # get intepolation and quadrature points
        pts_PI = self.getpts_for_PI(comp, with_subs)

        # array of evaluated function
        mat_f = xp.empty((pts_PI[0].size, pts_PI[1].size, pts_PI[2].size), dtype=float)

        # create a meshgrid and evaluate function on point set
        if eval_kind == "meshgrid":
            pts1, pts2, pts3 = xp.meshgrid(pts_PI[0], pts_PI[1], pts_PI[2], indexing="ij")
            mat_f[:, :, :] = fun(pts1, pts2, pts3)

        # tensor-product evaluation is done by input function
        elif eval_kind == "tensor_product":
            mat_f[:, :, :] = fun(pts_PI[0], pts_PI[1], pts_PI[2])

        # point-wise evaluation
        else:
            for i1 in range(pts_PI[0].size):
                for i2 in range(pts_PI[1].size):
                    for i3 in range(pts_PI[2].size):
                        mat_f[i1, i2, i3] = fun(pts_PI[0][i1], pts_PI[1][i2], pts_PI[2][i3])

        return mat_f

    # ======================================
    # def assemble_Schur0_inv(self):
    #
    #    n1 = self.pts_PI_0[0].size
    #    n2 = self.pts_PI_0[1].size
    #
    #    # apply (I0_22) to each column
    #    self.S0 = xp.zeros(((n1 - 2)*n2, 3), dtype=float)
    #
    #    for i in range(3):
    #        self.S0[:, i] = kron_lusolve_2d(self.I0_22_LUs, self.I0_21[:, i].toarray().reshape(n1 - 2, n2)).flatten()
    #
    #    # 3 x 3 matrix
    #    self.S0 = xp.linalg.inv(self.I0_11.toarray() - self.I0_12.toarray().dot(self.S0))
    #
    #
    # ======================================
    # def I0_inv(self, rhs, include_bc):
    #
    #    n1 = self.pts_PI_0[0].size
    #    n2 = self.pts_PI_0[1].size
    #
    #    if include_bc:
    #        rhs1 = rhs[:3]
    #        rhs2 = rhs[3:].reshape(n1 - 2, n2)
    #
    #        # solve pure 3x3 polar contribution
    #        out1 = self.S0.dot(rhs1)
    #
    #        # solve pure tensor-product contribution I0_22^(-1)*rhs2
    #        out2 = kron_lusolve_2d(self.I0_22_LUs, rhs2)
    #
    #        # solve for polar coefficients
    #        out1 -= self.S0.dot(self.I0_12.dot(out2.flatten()))
    #
    #        # solve for tensor-product coefficients
    #        out2  = out2 - kron_lusolve_2d(self.I0_22_LUs, self.I0_21.dot(self.S0.dot(rhs1)).reshape(n1 - 2, n2)) + kron_lusolve_2d(self.I0_22_LUs, self.I0_21.dot(self.S0.dot(self.I0_12.dot(out2.flatten()))).reshape(n1 - 2, n2))
    #
    #    return xp.concatenate((out1, out2.flatten()))

    # ======================================

    def solve_V0(self, dofs_0, include_bc):
        # with boundary splines
        if include_bc:
            dofs_0 = dofs_0.reshape(self.P0_pol.shape[0], self.I_tor.shape[0])
            coeffs = self.I_tor_LU.solve(self.I0_pol_LU.solve(dofs_0).T).T

        # without boundary splines
        else:
            dofs_0 = dofs_0.reshape(self.P0_pol_0.shape[0], self.I0_tor.shape[0])
            coeffs = self.I0_tor_LU.solve(self.I0_pol_0_LU.solve(dofs_0).T).T

        return coeffs.flatten()

    # ======================================
    def solve_V1(self, dofs_1, include_bc):
        # with boundary splines
        if include_bc:
            dofs_11 = dofs_1[: self.P1_pol.shape[0] * self.I_tor.shape[0]].reshape(
                self.P1_pol.shape[0],
                self.I_tor.shape[0],
            )
            dofs_12 = dofs_1[self.P1_pol.shape[0] * self.I_tor.shape[0] :].reshape(
                self.P0_pol.shape[0],
                self.H_tor.shape[0],
            )

            coeffs1 = self.I_tor_LU.solve(self.I1_pol_LU.solve(dofs_11).T).T
            coeffs2 = self.H_tor_LU.solve(self.I0_pol_LU.solve(dofs_12).T).T

        # without boundary splines
        else:
            dofs_11 = dofs_1[: self.P1_pol_0.shape[0] * self.I0_tor.shape[0]].reshape(
                self.P1_pol_0.shape[0],
                self.I0_tor.shape[0],
            )
            dofs_12 = dofs_1[self.P1_pol_0.shape[0] * self.I0_tor.shape[0] :].reshape(
                self.P0_pol_0.shape[0],
                self.H0_tor.shape[0],
            )

            coeffs1 = self.I0_tor_LU.solve(self.I1_pol_0_LU.solve(dofs_11).T).T
            coeffs2 = self.H0_tor_LU.solve(self.I0_pol_0_LU.solve(dofs_12).T).T

        return xp.concatenate((coeffs1.flatten(), coeffs2.flatten()))

    # ======================================
    def solve_V2(self, dofs_2, include_bc):
        # with boundary splines
        if include_bc:
            dofs_21 = dofs_2[: self.P2_pol.shape[0] * self.H_tor.shape[0]].reshape(
                self.P2_pol.shape[0],
                self.H_tor.shape[0],
            )
            dofs_22 = dofs_2[self.P2_pol.shape[0] * self.H_tor.shape[0] :].reshape(
                self.P3_pol.shape[0],
                self.I_tor.shape[0],
            )

            coeffs1 = self.H_tor_LU.solve(self.I2_pol_LU.solve(dofs_21).T).T
            coeffs2 = self.I_tor_LU.solve(self.I3_pol_LU.solve(dofs_22).T).T

        # without boundary splines
        else:
            dofs_21 = dofs_2[: self.P2_pol_0.shape[0] * self.H0_tor.shape[0]].reshape(
                self.P2_pol_0.shape[0],
                self.H0_tor.shape[0],
            )
            dofs_22 = dofs_2[self.P2_pol_0.shape[0] * self.H0_tor.shape[0] :].reshape(
                self.P3_pol_0.shape[0],
                self.I0_tor.shape[0],
            )

            coeffs1 = self.H0_tor_LU.solve(self.I2_pol_0_LU.solve(dofs_21).T).T
            coeffs2 = self.I0_tor_LU.solve(self.I3_pol_0_LU.solve(dofs_22).T).T

        return xp.concatenate((coeffs1.flatten(), coeffs2.flatten()))

    # ======================================
    def solve_V3(self, dofs_3, include_bc):
        # with boundary splines
        if include_bc:
            dofs_3 = dofs_3.reshape(self.P3_pol.shape[0], self.H_tor.shape[0])
            coeffs = self.H_tor_LU.solve(self.I3_pol_LU.solve(dofs_3).T).T

        # without boundary splines
        else:
            dofs_3 = dofs_3.reshape(self.P3_pol_0.shape[0], self.H0_tor.shape[0])
            coeffs = self.H0_tor_LU.solve(self.I3_pol_0_LU.solve(dofs_3).T).T

        return coeffs.flatten()

    # ======================================

    def apply_IinvT_V0(self, rhs, include_bc=False):
        # with boundary splines
        if include_bc:
            if not hasattr(self, "I0_pol_T_LU"):
                self.I0_pol_T_LU = spa.linalg.splu(self.I0_pol.T.tocsc())

            rhs = rhs.reshape(self.P0_pol.shape[0], self.I_tor.shape[0])
            rhs = self.I0_pol_T_LU.solve(self.I_tor_T_LU.solve(rhs.T).T)

        # without boundary splines
        else:
            rhs = rhs.reshape(self.P0_pol_0.shape[0], self.I0_tor.shape[0])
            rhs = self.I0_pol_0_T_LU.solve(self.I0_tor_T_LU.solve(rhs.T).T)

        return rhs.flatten()

    # ======================================
    def apply_IinvT_V1(self, rhs, include_bc=False):
        # with boundary splines
        if include_bc:
            if not hasattr(self, "I0_pol_T_LU"):
                self.I0_pol_T_LU = spa.linalg.splu(self.I0_pol.T.tocsc())

            if not hasattr(self, "I1_pol_T_LU"):
                self.I1_pol_T_LU = spa.linalg.splu(self.I1_pol.T.tocsc())

            rhs1 = rhs[: self.P1_pol.shape[0] * self.I_tor.shape[0]].reshape(self.P1_pol.shape[0], self.I_tor.shape[0])
            rhs2 = rhs[self.P1_pol.shape[0] * self.I_tor.shape[0] :].reshape(self.P0_pol.shape[0], self.H_tor.shape[0])

            rhs1 = self.I1_pol_T_LU.solve(self.I_tor_T_LU.solve(rhs1.T).T)
            rhs2 = self.I0_pol_T_LU.solve(self.H_tor_T_LU.solve(rhs2.T).T)

        # without boundary splines
        else:
            rhs1 = rhs[: self.P1_pol_0.shape[0] * self.I0_tor.shape[0]].reshape(
                self.P1_pol_0.shape[0],
                self.I0_tor.shape[0],
            )
            rhs2 = rhs[self.P1_pol_0.shape[0] * self.I0_tor.shape[0] :].reshape(
                self.P0_pol_0.shape[0],
                self.H0_tor.shape[0],
            )

            rhs1 = self.I1_pol_0_T_LU.solve(self.I0_tor_T_LU.solve(rhs1.T).T)
            rhs2 = self.I0_pol_0_T_LU.solve(self.H0_tor_T_LU.solve(rhs2.T).T)

        return xp.concatenate((rhs1.flatten(), rhs2.flatten()))

    # ======================================
    def apply_IinvT_V2(self, rhs, include_bc=False):
        # with boundary splines
        if include_bc:
            if not hasattr(self, "I2_pol_T_LU"):
                self.I2_pol_T_LU = spa.linalg.splu(self.I2_pol.T.tocsc())

            if not hasattr(self, "I3_pol_T_LU"):
                self.I3_pol_T_LU = spa.linalg.splu(self.I3_pol.T.tocsc())

            rhs1 = rhs[: self.P2_pol.shape[0] * self.H_tor.shape[0]].reshape(self.P2_pol.shape[0], self.H_tor.shape[0])
            rhs2 = rhs[self.P2_pol.shape[0] * self.H_tor.shape[0] :].reshape(self.P3_pol.shape[0], self.I_tor.shape[0])

            rhs1 = self.I2_pol_T_LU.solve(self.H_tor_T_LU.solve(rhs1.T).T)
            rhs2 = self.I3_pol_T_LU.solve(self.I_tor_T_LU.solve(rhs2.T).T)

        # without boundary splines
        else:
            rhs1 = rhs[: self.P2_pol_0.shape[0] * self.H0_tor.shape[0]].reshape(
                self.P2_pol_0.shape[0],
                self.H0_tor.shape[0],
            )
            rhs2 = rhs[self.P2_pol_0.shape[0] * self.H0_tor.shape[0] :].reshape(
                self.P3_pol_0.shape[0],
                self.I0_tor.shape[0],
            )

            rhs1 = self.I2_pol_0_T_LU.solve(self.H0_tor_T_LU.solve(rhs1.T).T)
            rhs2 = self.I3_pol_0_T_LU.solve(self.I0_tor_T_LU.solve(rhs2.T).T)

        return xp.concatenate((rhs1.flatten(), rhs2.flatten()))

    # ======================================
    def apply_IinvT_V3(self, rhs, include_bc=False):
        # with boundary splines
        if include_bc:
            if not hasattr(self, "I3_pol_T_LU"):
                self.I3_pol_T_LU = spa.linalg.splu(self.I3_pol.T.tocsc())

            rhs = rhs.reshape(self.P3_pol.shape[0], self.H_tor.shape[0])
            rhs = self.I3_pol_T_LU.solve(self.H_tor_T_LU.solve(rhs.T).T)

        # without boundary splines
        else:
            rhs = rhs.reshape(self.P3_pol_0.shape[0], self.H0_tor.shape[0])
            rhs = self.I3_pol_0_T_LU.solve(self.H0_tor_T_LU.solve(rhs.T).T)

        return rhs.flatten()

    # ======================================

    def dofs_0(self, fun, include_bc=True, eval_kind="meshgrid"):
        # get function values at point sets
        dofs = self.eval_for_PI(0, fun, eval_kind)

        # get dofs on tensor-product grid
        dofs = kron_matvec_3d(
            [spa.identity(dofs.shape[0]), spa.identity(dofs.shape[1]), spa.identity(dofs.shape[2])],
            dofs,
        )

        # apply extraction operator for dofs
        if include_bc:
            dofs = self.P0.dot(dofs.flatten())
        else:
            dofs = self.P0_0.dot(dofs.flatten())

        return dofs

    # ======================================
    def dofs_1(self, fun, include_bc=True, eval_kind="meshgrid", with_subs=True):
        # get function values at point sets
        dofs_1 = self.eval_for_PI(11, fun[0], eval_kind, with_subs)
        dofs_2 = self.eval_for_PI(12, fun[1], eval_kind, with_subs)
        dofs_3 = self.eval_for_PI(13, fun[2], eval_kind, with_subs)

        # get dofs_1 on tensor-product grid: integrate along 1-direction
        if with_subs:
            dofs_1 = kron_matvec_3d([self.Q1, spa.identity(dofs_1.shape[1]), spa.identity(dofs_1.shape[2])], dofs_1)
        else:
            dofs_1 = kron_matvec_3d([self.Q1G, spa.identity(dofs_1.shape[1]), spa.identity(dofs_1.shape[2])], dofs_1)

        # get dofs_2 on tensor-product grid: integrate along 2-direction
        if with_subs:
            dofs_2 = kron_matvec_3d([spa.identity(dofs_2.shape[0]), self.Q2, spa.identity(dofs_2.shape[2])], dofs_2)
        else:
            dofs_2 = kron_matvec_3d([spa.identity(dofs_2.shape[0]), self.Q2G, spa.identity(dofs_2.shape[2])], dofs_2)

        # get dofs_3 on tensor-product grid: integrate along 3-direction
        if with_subs:
            dofs_3 = kron_matvec_3d([spa.identity(dofs_3.shape[0]), spa.identity(dofs_3.shape[1]), self.Q3], dofs_3)
        else:
            dofs_3 = kron_matvec_3d([spa.identity(dofs_3.shape[0]), spa.identity(dofs_3.shape[1]), self.Q3G], dofs_3)

        # apply extraction operator for dofs
        if include_bc:
            dofs = self.P1.dot(xp.concatenate((dofs_1.flatten(), dofs_2.flatten(), dofs_3.flatten())))
        else:
            dofs = self.P1_0.dot(xp.concatenate((dofs_1.flatten(), dofs_2.flatten(), dofs_3.flatten())))

        return dofs

    # ======================================
    def dofs_2(self, fun, include_bc=True, eval_kind="meshgrid", with_subs=True):
        # get function values at point sets
        dofs_1 = self.eval_for_PI(21, fun[0], eval_kind, with_subs)
        dofs_2 = self.eval_for_PI(22, fun[1], eval_kind, with_subs)
        dofs_3 = self.eval_for_PI(23, fun[2], eval_kind, with_subs)

        # get dofs_1 on tensor-product grid: integrate in 2-3-plane
        if with_subs:
            dofs_1 = kron_matvec_3d([spa.identity(dofs_1.shape[0]), self.Q2, self.Q3], dofs_1)
        else:
            dofs_1 = kron_matvec_3d([spa.identity(dofs_1.shape[0]), self.Q2G, self.Q3G], dofs_1)

        # get dofs_2 on tensor-product grid: integrate in 1-3-plane
        if with_subs:
            dofs_2 = kron_matvec_3d([self.Q1, spa.identity(dofs_2.shape[1]), self.Q3], dofs_2)
        else:
            dofs_2 = kron_matvec_3d([self.Q1G, spa.identity(dofs_2.shape[1]), self.Q3G], dofs_2)

        # get dofs_3 on tensor-product grid: integrate in 1-2-plane
        if with_subs:
            dofs_3 = kron_matvec_3d([self.Q1, self.Q2, spa.identity(dofs_3.shape[2])], dofs_3)
        else:
            dofs_3 = kron_matvec_3d([self.Q1G, self.Q2G, spa.identity(dofs_3.shape[2])], dofs_3)

        # apply extraction operator for dofs
        if include_bc:
            dofs = self.P2.dot(xp.concatenate((dofs_1.flatten(), dofs_2.flatten(), dofs_3.flatten())))
        else:
            dofs = self.P2_0.dot(xp.concatenate((dofs_1.flatten(), dofs_2.flatten(), dofs_3.flatten())))

        return dofs

    # ======================================
    def dofs_3(self, fun, include_bc=True, eval_kind="meshgrid", with_subs=True):
        # get function values at point sets
        dofs = self.eval_for_PI(3, fun, eval_kind, with_subs)

        # get dofs on tensor-product grid: integrate in 1-2-3-cell
        if with_subs:
            dofs = kron_matvec_3d([self.Q1, self.Q2, self.Q3], dofs)
        else:
            dofs = kron_matvec_3d([self.Q1G, self.Q2G, self.Q3G], dofs)

        # apply extraction operator for dofs
        if include_bc:
            dofs = self.P3.dot(dofs.flatten())
        else:
            dofs = self.P3_0.dot(dofs.flatten())

        return dofs

    # ======================================

    def pi_0(self, fun, include_bc=True, eval_kind="meshgrid"):
        return self.solve_V0(self.dofs_0(fun, include_bc, eval_kind), include_bc)

    # ======================================
    def pi_1(self, fun, include_bc=True, eval_kind="meshgrid", with_subs=True):
        return self.solve_V1(self.dofs_1(fun, include_bc, eval_kind, with_subs), include_bc)

    # ======================================
    def pi_2(self, fun, include_bc=True, eval_kind="meshgrid", with_subs=True):
        return self.solve_V2(self.dofs_2(fun, include_bc, eval_kind, with_subs), include_bc)

    # ======================================
    def pi_3(self, fun, include_bc=True, eval_kind="meshgrid", with_subs=True):
        return self.solve_V3(self.dofs_3(fun, include_bc, eval_kind, with_subs), include_bc)

    # ========================================

    def assemble_approx_inv(self, tol):
        if not self.approx_Ik_0_inv or (self.approx_Ik_0_inv and self.approx_Ik_0_tol != tol):
            # poloidal plane
            I0_pol_0_inv_approx = xp.linalg.inv(self.I0_pol_0.toarray())
            I1_pol_0_inv_approx = xp.linalg.inv(self.I1_pol_0.toarray())
            I2_pol_0_inv_approx = xp.linalg.inv(self.I2_pol_0.toarray())
            I3_pol_0_inv_approx = xp.linalg.inv(self.I3_pol_0.toarray())
            I0_pol_inv_approx = xp.linalg.inv(self.I0_pol.toarray())

            if tol > 1e-14:
                I0_pol_0_inv_approx[xp.abs(I0_pol_0_inv_approx) < tol] = 0.0
                I1_pol_0_inv_approx[xp.abs(I1_pol_0_inv_approx) < tol] = 0.0
                I2_pol_0_inv_approx[xp.abs(I2_pol_0_inv_approx) < tol] = 0.0
                I3_pol_0_inv_approx[xp.abs(I3_pol_0_inv_approx) < tol] = 0.0
                I0_pol_inv_approx[xp.abs(I0_pol_inv_approx) < tol] = 0.0

            I0_pol_0_inv_approx = spa.csr_matrix(I0_pol_0_inv_approx)
            I1_pol_0_inv_approx = spa.csr_matrix(I1_pol_0_inv_approx)
            I2_pol_0_inv_approx = spa.csr_matrix(I2_pol_0_inv_approx)
            I3_pol_0_inv_approx = spa.csr_matrix(I3_pol_0_inv_approx)
            I0_pol_inv_approx = spa.csr_matrix(I0_pol_inv_approx)

            # toroidal direction
            I_inv_tor_approx = xp.linalg.inv(self.I_tor.toarray())
            H_inv_tor_approx = xp.linalg.inv(self.H_tor.toarray())

            if tol > 1e-14:
                I_inv_tor_approx[xp.abs(I_inv_tor_approx) < tol] = 0.0
                H_inv_tor_approx[xp.abs(H_inv_tor_approx) < tol] = 0.0

            I_inv_tor_approx = spa.csr_matrix(I_inv_tor_approx)
            H_inv_tor_approx = spa.csr_matrix(H_inv_tor_approx)

            # tensor-product poloidal x toroidal
            self.I0_0_inv_approx = spa.kron(I0_pol_0_inv_approx, I_inv_tor_approx, format="csr")

            self.I1_0_inv_approx = spa.bmat(
                [
                    [spa.kron(I1_pol_0_inv_approx, I_inv_tor_approx), None],
                    [None, spa.kron(I0_pol_0_inv_approx, H_inv_tor_approx)],
                ],
                format="csr",
            )

            self.I2_0_inv_approx = spa.bmat(
                [
                    [spa.kron(I2_pol_0_inv_approx, H_inv_tor_approx), None],
                    [None, spa.kron(I3_pol_0_inv_approx, I_inv_tor_approx)],
                ],
                format="csr",
            )

            self.I3_0_inv_approx = spa.kron(I3_pol_0_inv_approx, H_inv_tor_approx, format="csr")

            self.I0_inv_approx = spa.kron(I0_pol_inv_approx, I_inv_tor_approx, format="csr")

            self.approx_Ik_0_inv = True
            self.approx_Ik_0_tol = tol

        else:
            print("Approximations for inverse interpolation matrices already exist!")
