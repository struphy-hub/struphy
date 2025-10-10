# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic modules to create tensor-product finite element spaces of univariate B-splines.
"""

import matplotlib
import scipy.sparse as spa

from struphy.utils.arrays import xp as np

matplotlib.rcParams.update({"font.size": 16})
import matplotlib.pyplot as plt

import struphy.bsplines.bsplines as bsp
import struphy.bsplines.evaluation_kernels_1d as eva_1d
import struphy.bsplines.evaluation_kernels_2d as eva_2d
import struphy.bsplines.evaluation_kernels_3d as eva_3d
import struphy.eigenvalue_solvers.derivatives as der
import struphy.eigenvalue_solvers.mass_matrices_1d as mass_1d
import struphy.eigenvalue_solvers.mass_matrices_2d as mass_2d
import struphy.eigenvalue_solvers.mass_matrices_3d as mass_3d
import struphy.eigenvalue_solvers.projectors_global as pro
import struphy.polar.extraction_operators as pol


# =============== 1d B-spline space ======================
class Spline_space_1d:
    """
    Defines a 1d space of B-splines.

    Parameters
    ----------
        Nel : int
            number of elements of discretized 1D domain [0, 1]

        p : int
            spline degree

        spl_kind : boolean
            kind of spline space (True = periodic, False = clamped)

        n_quad : int
            number of Gauss-Legendre quadrature points per grid cell (defined by break points)

        bc : [str, str]
            boundary conditions at eta1=0.0 and eta1=1.0, 'f' free, 'd' dirichlet (remove boundary spline)

    Attributes
    ----------
        el_b : np.array
            Element boundaries, equally spaced.

        delta : float
            Uniform grid spacing

        T : np.array
            Knot vector of 0-space.

        t : np.arrray
            Knot vector of 1-space.

        greville : np.array
            Greville points.

        NbaseN : int
            Dimension of 0-space.

        NbaseD : int
            Dimension of 1-space.

        indN : np.array
            Global indices of non-vanishing B-splines in each element in format (element, local basis function)

        indD : np.array
            Global indices of non-vanishing M-splines in each element in format (element, local basis function)

        pts : np.array
            Global GL quadrature points in format (element, local point).

        wts : np.array
            Global GL quadrature weights in format (element, local point).

        basisN : np.array
            N-basis functions evaluated at quadrature points in format (element, local basis function, derivative, local point)

        basisD : np.array
            D-basis functions evaluated at quadrature points in format (element, local basis function, derivative, local point)

        E0 : csr_matrix
            Identity matrix of rank NbaseN.

        E1 : csr_matrix
            Identity matrix of rank NbaseD.

        B0 : csr_matrix
            Boundary operator for 0-space: removes interpolatory B-spline at eta=0 if bc[0]='d', at eta=1 if bc[1]='d'.

        B1 : csr_matrix
            Boundary operator for 1-space, same as E1 (identity).

        E0_0 : csr_matrix
            Extraction operator for 0-space, with bc applied: E0_0 = B0 * E0

        E1_0 : csr_matrix
            Extraction operator for 1-space, same as E1 (identity).

        M0 : csr_matrix
            NN-mass-matrix with boundary conditions, M0 = E0_0 * M0_full * E0_0^T.

        M01 : csr_matrix
            ND-mass-matrix with boundary conditions, M01_0 = E0_0 * M0.

        M10 : csr_matrix
            DN-mass-matrix (csr) with boundary conditions, M10_0 = M0 * E0_0^T.

        M1 : csr_matrix
            DD-mass-matrix.

        G : csr_matrix
            Derivative matrix.

        G0 : csr_matrix
            Derivative matrix with boundary conditions, G0 = B1 * G * B0^T.

        projectors : object
            1D projectors object from struphy.feec.projectors.pro_global.projectors_global.Projectors_global_1d
    """

    def __init__(self, Nel, p, spl_kind, n_quad=6, bc=["f", "f"]):
        self.Nel = Nel  # number of elements
        self.p = p  # spline degree
        self.spl_kind = spl_kind  # kind of spline space (periodic or clamped)

        # boundary conditions at eta=0. and eta=1. in case of clamped splines
        if spl_kind:
            self.bc = [None, None]
        else:
            self.bc = bc

        self.el_b = np.linspace(0.0, 1.0, Nel + 1)  # element boundaries
        self.delta = 1 / self.Nel  # element length

        self.T = bsp.make_knots(self.el_b, self.p, self.spl_kind)  # spline knot vector for B-splines (N)
        self.t = self.T[1:-1]  # spline knot vector for M-splines (D)

        self.greville = bsp.greville(self.T, self.p, self.spl_kind)  # greville points

        self.NbaseN = len(self.T) - self.p - 1 - self.spl_kind * self.p  # total number of B-splines (N)
        self.NbaseD = self.NbaseN - 1 + self.spl_kind  # total number of M-splines (D)

        # global indices of non-vanishing splines in each element in format (Nel, p + 1)
        self.indN = (np.indices((self.Nel, self.p + 1 - 0))[1] + np.arange(self.Nel)[:, None]) % self.NbaseN
        self.indD = (np.indices((self.Nel, self.p + 1 - 1))[1] + np.arange(self.Nel)[:, None]) % self.NbaseD

        self.n_quad = n_quad  # number of Gauss-Legendre points per grid cell (defined by break points)

        self.pts_loc = np.polynomial.legendre.leggauss(self.n_quad)[0]  # Gauss-Legendre points  (GLQP) in (-1, 1)
        self.wts_loc = np.polynomial.legendre.leggauss(self.n_quad)[1]  # Gauss-Legendre weights (GLQW) in (-1, 1)

        # global GLQP in format (element, local point) and total number of GLQP
        self.pts = bsp.quadrature_grid(self.el_b, self.pts_loc, self.wts_loc)[0]
        self.n_pts = self.pts.flatten().size

        # global GLQW in format (element, local point)
        self.wts = bsp.quadrature_grid(self.el_b, self.pts_loc, self.wts_loc)[1]

        # basis functions evaluated at quadrature points in format (element, local basis function, derivative, local point)
        self.basisN = bsp.basis_ders_on_quad_grid(self.T, self.p, self.pts, 0, normalize=False)
        self.basisD = bsp.basis_ders_on_quad_grid(self.t, self.p - 1, self.pts, 0, normalize=True)

        # -------------------------------------------------
        # Set extraction operators for boundary conditions:
        # -------------------------------------------------
        n1 = self.NbaseN
        d1 = self.NbaseD

        # boundary operators
        self.B0 = np.identity(n1, dtype=float)
        self.B1 = np.identity(d1, dtype=float)

        # extraction operators without boundary conditions
        self.E0 = spa.csr_matrix(self.B0.copy())
        self.E1 = spa.csr_matrix(self.B1.copy())

        # remove contributions from N-splines at eta = 0
        if self.bc[0] == "d":
            self.B0 = self.B0[1:, :]

        # remove contributions from N-splines at eta = 1
        if self.bc[1] == "d":
            self.B0 = self.B0[:-1, :]

        self.B0 = spa.csr_matrix(self.B0)
        self.B1 = spa.csr_matrix(self.B1)

        self.E0_0 = self.B0.dot(self.E0)
        self.E1_0 = self.B1.dot(self.E1)

        # -------------------------------------------------
        # Set discrete derivatives:
        # -------------------------------------------------
        self.G, self.G0 = der.discrete_derivatives_1d(self)

        # print('Spline space set up (1d) done.')

    # functions for setting mass matrices:
    # =================================================
    def assemble_M0(self, weight=None):
        """Assembles NN-mass-matrix (csr) with boundary conditions, M0_0 = E0_0 * M0 * E0_0^T."""
        self.M0 = self.E0_0.dot(mass_1d.get_M(self, 0, 0, weight).dot(self.E0_0.T))
        print("Assembly of M0 (1d) done.")

    # =================================================
    def assemble_M1(self, weight=None):
        """Assembles DD-mass-matrix (csr)."""
        self.M1 = self.E1_0.dot(mass_1d.get_M(self, 1, 1, weight).dot(self.E1_0.T))
        print("Assembly of M1 (1d) done.")

    # =================================================
    def assemble_M01(self, weight=None):
        """Assembles ND-mass-matrix (csr) with boundary conditions, M01_0 = E0_0 * M0."""
        self.M01 = self.E0_0.dot(mass_1d.get_M(self, 0, 1, weight).dot(self.E1_0.T))
        print("Assembly of M01 (1d) done.")

    # =================================================
    def assemble_M10(self, weight=None):
        """Assembles DN-mass-matrix (csr) with boundary conditions, M10_0 = M0 * E0_0^T."""
        self.M10 = self.E1_0.dot(mass_1d.get_M(self, 1, 0, weight).dot(self.E0_0.T))
        print("Assembly of M10 (1d) done.")

    # functions for setting projectors:
    # =================================================
    def set_projectors(self, nq=6):
        """Initialize 1d projectors object."""
        self.projectors = pro.Projectors_global_1d(self, nq)
        # print('Set projectors (1d) done.')

    # spline evaluation and plotting:
    # =================================================
    def evaluate_N(self, eta, coeff, kind=0):
        """
        Evaluates the spline space (N) at the point(s) eta for given coefficients coeff.

        Parameters
        ----------
        eta : double or array_like
            evaluation point(s)

        coeff : array_like
            FEM coefficients

        kind : int
            kind of evaluation (0: N, 2: dN/deta, 3: ddN/deta^2)

        Returns
        -------
        value : double or array_like
            evaluated FEM field at the point(s) eta
        """

        assert (coeff.size == self.E0.shape[0]) or (coeff.size == self.E0_0.shape[0])
        assert (kind == 0) or (kind == 2) or (kind == 3)

        if coeff.size == self.E0_0.shape[0]:
            coeff = self.E0_0.T.dot(coeff)

        if isinstance(eta, float):
            pts = np.array([eta])
        elif isinstance(eta, np.ndarray):
            pts = eta.flatten()

        values = np.empty(pts.size, dtype=float)
        eva_1d.evaluate_vector(self.T, self.p, self.indN, coeff, pts, values, kind)

        if isinstance(eta, float):
            values = values[0]
        elif isinstance(eta, np.ndarray):
            values = values.reshape(eta.shape)

        return values

    # =================================================
    def evaluate_D(self, eta, coeff):
        """
        Evaluates the spline space (D) at the point(s) eta for given coefficients coeff.

        Parameters
        ----------
        eta : double or array_like
            evaluation point(s)

        coeff : array_like
            FEM coefficients

        Returns
        -------
        value : double or array_like
            evaluated FEM field at the point(s) eta
        """

        assert coeff.size == self.E1.shape[0]

        if isinstance(eta, float):
            pts = np.array([eta])
        elif isinstance(eta, np.ndarray):
            pts = eta.flatten()

        values = np.empty(pts.size, dtype=float)
        eva_1d.evaluate_vector(self.t, self.p - 1, self.indD, coeff, pts, values, 1)

        if isinstance(eta, float):
            values = values[0]
        elif isinstance(eta, np.ndarray):
            values = values.reshape(eta.shape)

        return values

    # =================================================
    def plot_splines(self, n_pts=500, which="N"):
        """
        Plots all basis functions.

        Parameters
        ----------
        n_pts : int
            number of points for plotting (optinal, default=500)

        which : string
            which basis to plot. 'N', 'D' or 'dN' (optional, default='N')
        """

        etaplot = np.linspace(0.0, 1.0, n_pts)

        degree = self.p

        if which == "N":
            coeff = np.zeros(self.NbaseN, dtype=float)

            for i in range(self.NbaseN):
                coeff[:] = 0.0
                coeff[i] = 1.0
                plt.plot(etaplot, self.evaluate_N(etaplot, coeff), label=str(i))

        elif which == "D":
            coeff = np.zeros(self.NbaseD, dtype=float)

            for i in range(self.NbaseD):
                coeff[:] = 0.0
                coeff[i] = 1.0
                plt.plot(etaplot, self.evaluate_D(etaplot, coeff), label=str(i))

            degree = self.p - 1

        elif which == "dN":
            coeff = np.zeros(self.NbaseN, dtype=float)

            for i in range(self.NbaseN):
                coeff[:] = 0.0
                coeff[i] = 1.0
                plt.plot(etaplot, self.evaluate_N(etaplot, coeff, 2), label=str(i))

        else:
            print("Only N, D and dN available")

        if self.spl_kind:
            bcs = "periodic"
        else:
            bcs = "clamped"

        (greville,) = plt.plot(self.greville, np.zeros(self.greville.shape), "ro", label="greville")
        (breaks,) = plt.plot(self.el_b, np.zeros(self.el_b.shape), "k+", label="breaks")
        plt.title(which + f"$^{degree}$-splines, " + bcs + f", Nel={self.Nel}")
        plt.legend(handles=[greville, breaks])


# =============== 2d/3d tensor-product B-spline space ======================
class Tensor_spline_space:
    """
    Defines a tensor product space of 1d B-spline spaces in higher dimensions (2d and 3d).

    Parameters
    ----------
        spline_spaces : list of spline_space_1d
            1d B-spline spaces from which the tensor_product B-spline space is built

        ck : int
            smoothness contraint at eta_1=0 (pole): -1 (no constraints), 0 or 1 (polar splines)

        cx, cy : 2D arrays
            control points for spline mapping in case of polar splines

        n_tor : int
            mode number in third direction for a 2D spline space (default n_tor = 0)

        basis_tor : string
            basis in third direction for a 2D spline space if |n_tor| > 0 (r: real sin/cos, i: complex Fourier)


    Attributes
    ----------
        E0_0 : csr_matrix
            3D Extraction operator for 0-space with boundary conditions, E0_0 = B0 * E0.

        E1_0 : csr_matrix
            3D Extraction operator for 1-space (as block matrix) with boundary conditions, E1_0 = B1 * E1.

        E2_0 : csr_matrix
            3D Extraction operator for 2-space (as block matrix) with boundary conditions, E2_0 = B2 * E2.

        E3_0 : csr_matrix
            3D Extraction operator for 3-space with boundary conditions, E3_0 = B3 * E3.

        Ev_0 : csr_matrix
            3D Extraction operator for vector-feild-space (as block matrix) with boundary conditions, Ev_0 = Bv * Ev.

        M0 : lin. operator
            (NNN)-(NNN)-|detDF|-mass-matrix with extraction, M0 = E0_0 * M0_ * E0_0^T.

        M0_mat : csr_matrix
            (NNN)-(NNN)-|detDF|-mass-matrix with extraction, M0 = E0_0 * M0_ * E0_0^T.

        M1 : lin. operator
            V1-mass-matrix with extraction, M1 = E1_0 * M1_ * E1_0^T, in format M1_11 = (DNN)-Ginv_11-(DNN)-|detDF|,
            M1_12 = (DNN)-Ginv_12-(NDN)-|detDF|, etc.

        M1_mat : csr_matrix
            V1-mass-matrix with extraction, M1 = E1_0 * M1_ * E1_0^T, in format M1_11 = (DNN)-Ginv_11-(DNN)-|detDF|,
            M1_12 = (DNN)-Ginv_12-(NDN)-|detDF|, etc.

        M2 : linear operator
            V2-mass-matrix with extraction, M2 = E2_0 * M2_ * E2_0^T, in format M2_11 = (NDD)-G_11-(NDD)-|detDFinv|,
            M2_12 = (NDD)-G_12-(DND)-|detDFinv|, etc.

        M2_mat : csr_matrix
            V2-mass-matrix with extraction, M2 = E2_0 * M2_ * E2_0^T, in format M2_11 = (NDD)-G_11-(NDD)-|detDFinv|,
            M2_12 = (NDD)-G_12-(DND)-|detDFinv|, etc.

        M3 : linear operator
            (DDD)-(DDD)-|detDFinv|-mass-matrix with extraction, M3 = E3_0 * M3_ * E3_0^T.

        M3_mat : csr_matrix
            (DDD)-(DDD)-|detDFinv|-mass-matrix with extraction, M3 = E3_0 * M3_ * E3_0^T.

        Mv : linear operator
            Vector-field-mass-matrix in format Mv_ij = (NNN)-G_ij-(NNN)-|detDF|.

        Mv_mat : csr_matrix
            Vector-field-mass-matrix in format Mv_ij = (NNN)-G_ij-(NNN)-|detDF|.

        G : csr_matrix
            Gradient (block) matrix.

        G0 : csr_matrix
            Gradient (block) matrix with boundary conditions, G0 = B1 * G * B0^T.

        C : csr_matrix
            Curl (block) matrix.

        C0 : csr_matrix
            Curl (block) matrix with boundary conditions, C0 = B2 * C * B1^T.

        D : csr_matrix
            Divergence (block) matrix.

        D0 : csr_matrix
            Divergence (block) matrix with boundary conditions, D0 = B3 * D * B2^T.

        projectors : object
            3D projectors object from struphy.feec.projectors.pro_global.projectors_global.Projectors_global_3d.
    """

    def __init__(self, spline_spaces, ck=-1, cx=None, cy=None, n_tor=0, basis_tor="r"):
        # 1D B-spline spaces
        assert len(spline_spaces) == 2 or len(spline_spaces) == 3

        self.spaces = spline_spaces
        self.dim = len(self.spaces)

        # set basis in 3rd dimension if |n_tor| > 0 (2D space): sin(2*pi*n*eta_3)/cos(2*pi*n*eta_3) or exp(i*2*pi*n*eta_3)
        if self.dim == 2:
            self.n_tor = 0

            if abs(n_tor) > 0:
                assert isinstance(n_tor, int)
                self.n_tor = n_tor

                assert basis_tor == "r" or basis_tor == "i"
                self.basis_tor = basis_tor

        # C^k smoothness constraint at eta_1 = 0
        assert ck == -1 or ck == 0 or ck == 1

        self.ck = ck

        if self.ck == 1:
            assert cx.ndim == 2
            assert cy.ndim == 2

        # input from 1d spaces
        # ====================
        self.Nel = [spl.Nel for spl in self.spaces]  # number of elements
        self.p = [spl.p for spl in self.spaces]  # spline degree
        self.spl_kind = [spl.spl_kind for spl in self.spaces]  # kind of spline space (periodic or clamped)

        self.bc = [spl.bc for spl in self.spaces]  # boundary conditions at eta = 0 and eta = 1

        self.el_b = [spl.el_b for spl in self.spaces]  # element boundaries
        self.delta = [spl.delta for spl in self.spaces]  # element lengths

        self.T = [spl.T for spl in self.spaces]  # spline knot vector for B-splines (N)
        self.t = [spl.t for spl in self.spaces]  # spline knot vector for M-splines (D)

        self.NbaseN = [spl.NbaseN for spl in self.spaces]  # total number of B-splines (N)
        self.NbaseD = [spl.NbaseD for spl in self.spaces]  # total number of M-splines (D)

        self.indN = [spl.indN for spl in self.spaces]  # global indices of non-vanishing B-splines (N) per element
        self.indD = [spl.indD for spl in self.spaces]  # global indices of non-vanishing M-splines (D) per element

        self.n_quad = [spl.n_quad for spl in self.spaces]  # number of Gauss-Legendre quadrature points per element

        self.pts_loc = [spl.pts_loc for spl in self.spaces]  # Gauss-Legendre quadrature points  (GLQP) in (-1, 1)
        self.wts_loc = [spl.wts_loc for spl in self.spaces]  # Gauss-Legendre quadrature weights (GLQW) in (-1, 1)

        self.pts = [spl.pts for spl in self.spaces]  # global GLQP in format (element, local point)
        self.wts = [spl.wts for spl in self.spaces]  # global GLQW in format (element, local weight)

        self.n_pts = [spl.n_pts for spl in self.spaces]  # total number of quadrature points

        # basis functions evaluated at quadrature points in format (element, local basis function, derivative, local point)
        self.basisN = [spl.basisN for spl in self.spaces]
        self.basisD = [spl.basisD for spl in self.spaces]

        # set number of basis function in 3rd direction for 2D space
        if self.dim == 2:
            if self.n_tor == 0:
                self.NbaseN = self.NbaseN + [1]
                self.NbaseD = self.NbaseD + [1]

            else:
                if self.basis_tor == "r":
                    self.NbaseN = self.NbaseN + [2]
                    self.NbaseD = self.NbaseD + [2]

                else:
                    self.NbaseN = self.NbaseN + [1]
                    self.NbaseD = self.NbaseD + [1]

        # set mass matrices in 3rd direction
        if self.dim == 2:
            if self.n_tor == 0 or self.basis_tor == "i":
                self.M0_tor = spa.identity(1, format="csr")
                self.M1_tor = spa.identity(1, format="csr")

            else:
                self.M0_tor = spa.csr_matrix(np.identity(2) / 2)
                self.M1_tor = spa.csr_matrix(np.identity(2) / 2)

        else:
            self.M0_tor = mass_1d.get_M(self.spaces[2], 0, 0)
            self.M1_tor = mass_1d.get_M(self.spaces[2], 1, 1)

        # number of basis functions of discrete tensor-product p-forms in 2D x analytical third dimension
        self.Nbase_0form = [self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]]

        self.Nbase_1form = [
            [self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]],
            [self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]],
            [self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]],
        ]

        self.Nbase_2form = [
            [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]],
            [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]],
            [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]],
        ]

        self.Nbase_3form = [self.NbaseD[0], self.NbaseD[1], self.NbaseD[2]]

        # total number of basis functions
        self.Ntot_0form = self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2]

        self.Ntot_1form = [
            self.NbaseD[0] * self.NbaseN[1] * self.NbaseN[2],
            self.NbaseN[0] * self.NbaseD[1] * self.NbaseN[2],
            self.NbaseN[0] * self.NbaseN[1] * self.NbaseD[2],
        ]

        self.Ntot_2form = [
            self.NbaseN[0] * self.NbaseD[1] * self.NbaseD[2],
            self.NbaseD[0] * self.NbaseN[1] * self.NbaseD[2],
            self.NbaseD[0] * self.NbaseD[1] * self.NbaseN[2],
        ]

        self.Ntot_3form = self.NbaseD[0] * self.NbaseD[1] * self.NbaseD[2]

        # cumulative number of basis functions for vector-valued spaces
        self.Ntot_1form_cum = [
            self.Ntot_1form[0],
            self.Ntot_1form[0] + self.Ntot_1form[1],
            self.Ntot_1form[0] + self.Ntot_1form[1] + self.Ntot_1form[2],
        ]

        self.Ntot_2form_cum = [
            self.Ntot_2form[0],
            self.Ntot_2form[0] + self.Ntot_2form[1],
            self.Ntot_2form[0] + self.Ntot_2form[1] + self.Ntot_2form[2],
        ]

        # -------------------------------------------------
        # Set extraction operators for boundary conditions:
        # -------------------------------------------------

        # standard tensor-product splines in eta_1-eta_2 plane
        if self.ck == -1:
            # extraction operators without boundary conditions
            ENN = spa.kron(self.spaces[0].E0, self.spaces[1].E0, format="csr")
            EDN = spa.kron(self.spaces[0].E1, self.spaces[1].E0, format="csr")
            END = spa.kron(self.spaces[0].E0, self.spaces[1].E1, format="csr")
            EDD = spa.kron(self.spaces[0].E1, self.spaces[1].E1, format="csr")

            self.E0_pol = ENN
            self.E3_pol = EDD

            self.E1_pol = spa.bmat([[EDN, None], [None, END]], format="csr")
            self.E2_pol = spa.bmat([[END, None], [None, EDN]], format="csr")

            self.Ev_pol = spa.bmat([[ENN, None], [None, ENN]], format="csr")

            # boundary operators
            BNN = spa.kron(self.spaces[0].B0, self.spaces[1].B0, format="csr")
            BDN = spa.kron(self.spaces[0].B1, self.spaces[1].B0, format="csr")
            BND = spa.kron(self.spaces[0].B0, self.spaces[1].B1, format="csr")
            BDD = spa.kron(self.spaces[0].B1, self.spaces[1].B1, format="csr")

            self.B0_pol = BNN
            self.B3_pol = BDD

            self.B1_pol = spa.bmat([[BDN, None], [None, BND]], format="csr")
            self.B2_pol = spa.bmat([[BND, None], [None, BDN]], format="csr")

            Bv1 = spa.kron(self.spaces[0].B0, spa.identity(self.spaces[1].NbaseN), format="csr")
            Bv2 = spa.kron(spa.identity(self.spaces[0].NbaseN), self.spaces[1].B0, format="csr")
            Bv3 = spa.kron(spa.identity(self.spaces[0].NbaseN), spa.identity(self.spaces[1].NbaseN), format="csr")

            self.Bv_pol = spa.bmat([[Bv1, None], [None, Bv2]], format="csr")

        # C^k polar splines in eta_1-eta_2 plane
        else:
            if self.ck == 0:
                self.polar_splines = pol.PolarSplines_C0_2D(self.spaces[0].NbaseN, self.spaces[1].NbaseN)
            elif self.ck == 1:
                self.polar_splines = pol.PolarSplines_C1_2D(cx, cy)

            # extraction operators without boundary conditions
            self.E0_pol = self.polar_splines.E0.copy()
            self.E1_pol = self.polar_splines.E1C.copy()
            self.E2_pol = self.polar_splines.E1D.copy()
            self.E3_pol = self.polar_splines.E2.copy()

            self.Ev_pol = spa.bmat([[self.E0_pol, None], [None, self.E0_pol]], format="csr")

            # boundary operators
            BNN = spa.identity(self.polar_splines.Nbase0, format="lil")
            BDD = spa.identity(self.polar_splines.Nbase2, format="lil")

            BDN = spa.identity(self.polar_splines.Nbase1C_1, format="lil")
            BND = spa.identity(self.polar_splines.Nbase1C_2, format="lil")

            if self.bc[0][1] == "d":
                BNN = BNN[: -self.spaces[1].NbaseN, :]
                BND = BND[: -self.spaces[1].NbaseD, :]

            self.B0_pol = BNN.tocsr()
            self.B3_pol = BDD.tocsr()

            self.B1_pol = spa.bmat([[BDN, None], [None, BND]], format="csr")
            self.B2_pol = spa.bmat([[BND, None], [None, BDN]], format="csr")

            Bv1 = self.B0_pol
            Bv2 = spa.identity(self.polar_splines.Nbase0, format="csr")
            Bv3 = spa.identity(self.polar_splines.Nbase0, format="csr")

            self.Bv_pol = spa.bmat([[Bv1, None], [None, Bv2]], format="csr")

        # extraction operators with boundary conditions
        self.E0_pol_0 = self.B0_pol.dot(self.E0_pol).tocsr()
        self.E1_pol_0 = self.B1_pol.dot(self.E1_pol).tocsr()
        self.E2_pol_0 = self.B2_pol.dot(self.E2_pol).tocsr()
        self.E3_pol_0 = self.B3_pol.dot(self.E3_pol).tocsr()
        self.Ev_pol_0 = self.Bv_pol.dot(self.Ev_pol).tocsr()

        # toroidal eta_3 direction
        if self.dim == 2:
            self.E0_tor = spa.identity(self.NbaseN[2])
            self.E1_tor = spa.identity(self.NbaseD[2])

            self.B0_tor = spa.identity(self.NbaseN[2])
            self.B1_tor = spa.identity(self.NbaseD[2])

        else:
            self.E0_tor = self.spaces[2].E0
            self.E1_tor = self.spaces[2].E1

            self.B0_tor = self.spaces[2].B0
            self.B1_tor = self.spaces[2].B1

        self.E0_tor_0 = self.B0_tor.dot(self.E0_tor).tocsr()
        self.E1_tor_0 = self.B1_tor.dot(self.E1_tor).tocsr()

        # extraction operators for 3D diagram: without boundary conditions
        self.E0 = spa.kron(self.E0_pol, self.E0_tor, format="csr")
        self.E1 = spa.bmat(
            [[spa.kron(self.E1_pol, self.E0_tor), None], [None, spa.kron(self.E0_pol, self.E1_tor)]], format="csr"
        )
        self.E2 = spa.bmat(
            [[spa.kron(self.E2_pol, self.E1_tor), None], [None, spa.kron(self.E3_pol, self.E0_tor)]], format="csr"
        )
        self.E3 = spa.kron(self.E3_pol, self.E1_tor, format="csr")
        self.Ev = spa.bmat(
            [[spa.kron(self.Ev_pol, self.E0_tor), None], [None, spa.kron(self.E0_pol, self.E0_tor)]], format="csr"
        )

        # boundary operators for 3D diagram
        self.B0 = spa.kron(self.B0_pol, self.B0_tor, format="csr")
        self.B1 = spa.bmat(
            [[spa.kron(self.B1_pol, self.B0_tor), None], [None, spa.kron(self.B0_pol, self.B1_tor)]], format="csr"
        )
        self.B2 = spa.bmat(
            [[spa.kron(self.B2_pol, self.B1_tor), None], [None, spa.kron(self.B3_pol, self.B0_tor)]], format="csr"
        )
        self.B3 = spa.kron(self.B3_pol, self.B1_tor, format="csr")
        self.Bv = spa.bmat(
            [[spa.kron(self.Bv_pol, self.E0_tor), None], [None, spa.kron(Bv3, self.B0_tor)]], format="csr"
        )

        # extraction operators for 3D diagram: with boundary conditions
        self.E0_0 = self.B0.dot(self.E0).tocsr()
        self.E1_0 = self.B1.dot(self.E1).tocsr()
        self.E2_0 = self.B2.dot(self.E2).tocsr()
        self.E3_0 = self.B3.dot(self.E3).tocsr()
        self.Ev_0 = self.Bv.dot(self.Ev).tocsr()

        # -------------------------------------------------
        # Set discrete derivatives:
        # -------------------------------------------------
        self.G, self.G0, self.C, self.C0, self.D, self.D0 = der.discrete_derivatives_3d(self)

    # function for setting projectors:
    # =================================================
    def set_projectors(self, which="tensor"):
        # tensor-product projectors (no polar splines possible)
        if which == "tensor":
            if self.dim == 2:
                self.projectors = pro.Projectors_tensor_2d([space.projectors for space in self.spaces])
            elif self.dim == 3:
                self.projectors = pro.Projectors_tensor_3d([space.projectors for space in self.spaces])
            else:
                raise NotImplementedError("Only 2d and 3d supported.")

        # general projectors (polar splines possible)
        elif which == "general":
            self.projectors = pro.ProjectorsGlobal3D(self)

    # ============== mass matrices =======================
    def apply_M0_ten(self, x, mats):
        """
        TODO
        """

        x = self.reshape_pol_0(x)

        out = mats[1].dot(mats[0].dot(x).T).T

        return out.flatten()

    def apply_M1_ten(self, x, mats):
        """
        TODO
        """

        x1, x2 = self.reshape_pol_1(x)

        out1 = mats[0][1].dot(mats[0][0].dot(x1).T).T
        out2 = mats[1][1].dot(mats[1][0].dot(x2).T).T

        return np.concatenate((out1.flatten(), out2.flatten()))

    def apply_M2_ten(self, x, mats):
        """
        TODO
        """

        x1, x2 = self.reshape_pol_2(x)

        out1 = mats[0][1].dot(mats[0][0].dot(x1).T).T
        out2 = mats[1][1].dot(mats[1][0].dot(x2).T).T

        return np.concatenate((out1.flatten(), out2.flatten()))

    def apply_M3_ten(self, x, mats):
        """
        TODO
        """

        x = self.reshape_pol_3(x)

        out = mats[1].dot(mats[0].dot(x).T).T

        return out.flatten()

    def apply_Mv_ten(self, x, mats):
        """
        TODO
        """

        x1, x2 = self.reshape_pol_v(x)

        out1 = mats[0][1].dot(mats[0][0].dot(x1).T).T
        out2 = mats[1][1].dot(mats[1][0].dot(x2).T).T

        return np.concatenate((out1.flatten(), out2.flatten()))

    def apply_M0_0_ten(self, x, mats):
        """
        TODO
        """

        x = self.reshape_pol_0(x)

        out = self.B0_tor.dot(mats[1].dot(self.B0_tor.T.dot(self.B0_pol.dot(mats[0].dot(self.B0_pol.T.dot(x))).T))).T

        return out.flatten()

    def apply_M1_0_ten(self, x, mats):
        """
        TODO
        """

        x1, x2 = self.reshape_pol_1(x)

        out1 = self.B0_tor.dot(
            mats[0][1].dot(self.B0_tor.T.dot(self.B1_pol.dot(mats[0][0].dot(self.B1_pol.T.dot(x1))).T))
        ).T
        out2 = self.B1_tor.dot(
            mats[1][1].dot(self.B1_tor.T.dot(self.B0_pol.dot(mats[1][0].dot(self.B0_pol.T.dot(x2))).T))
        ).T

        return np.concatenate((out1.flatten(), out2.flatten()))

    def apply_M2_0_ten(self, x, mats):
        """
        TODO
        """

        x1, x2 = self.reshape_pol_2(x)

        out1 = self.B1_tor.dot(
            mats[0][1].dot(self.B1_tor.T.dot(self.B2_pol.dot(mats[0][0].dot(self.B2_pol.T.dot(x1))).T))
        ).T
        out2 = self.B0_tor.dot(
            mats[1][1].dot(self.B0_tor.T.dot(self.B3_pol.dot(mats[1][0].dot(self.B3_pol.T.dot(x2))).T))
        ).T

        return np.concatenate((out1.flatten(), out2.flatten()))

    def apply_M3_0_ten(self, x, mats):
        """
        TODO
        """

        x = self.reshape_pol_3(x)

        out = self.B1_tor.dot(mats[1].dot(self.B1_tor.T.dot(self.B3_pol.dot(mats[0].dot(self.B3_pol.T.dot(x))).T))).T

        return out.flatten()

    def apply_Mv_0_ten(self, x, mats):
        """
        TODO
        """

        x1, x2 = self.reshape_pol_v(x)

        out1 = mats[0][1].dot(self.Bv_pol.dot(mats[0][0].dot(self.Bv_pol.T.dot(x1))).T).T
        out2 = self.B0_tor.dot(mats[1][1].dot(self.B0_tor.T.dot(mats[1][0].dot(x2).T))).T

        return np.concatenate((out1.flatten(), out2.flatten()))

    def __assemble_M0(self, domain, as_tensor=False):
        """
        TODO
        """

        self.M0_as_tensor = as_tensor

        # tensor product 2D poloidal x 1D toroidal
        if as_tensor:
            self.M0_pol_mat = mass_2d.get_M0(self, domain)

            matvec = lambda x: self.apply_M0_ten(x, [self.M0_pol_mat, self.M0_tor])
            matvec_0 = lambda x: self.apply_M0_0_ten(x, [self.M0_pol_mat, self.M0_tor])

        # 3D
        else:
            if self.dim == 2:
                self.M0_mat = spa.kron(mass_2d.get_M0(self, domain), self.M0_tor, format="csr")
            else:
                self.M0_mat = mass_3d.get_M0(self, domain)

            matvec = lambda x: self.M0_mat.dot(x)
            matvec_0 = lambda x: self.B0.dot(self.M0_mat.dot(self.B0.T.dot(x)))

        # linear operators
        self.M0 = spa.linalg.LinearOperator((self.E0.shape[0], self.E0.shape[0]), matvec=matvec)
        self.M0_0 = spa.linalg.LinearOperator((self.E0_0.shape[0], self.E0_0.shape[0]), matvec=matvec_0)

    def __assemble_M1(self, domain, as_tensor=False):
        """
        TODO
        """

        self.M1_as_tensor = as_tensor

        # tensor product 2D poloidal x 1D toroidal
        if as_tensor:
            self.M1_pol_mat = mass_2d.get_M1(self, domain)

            matvec = lambda x: self.apply_M1_ten(
                x, [[self.M1_pol_mat[0], self.M0_tor], [self.M1_pol_mat[1], self.M1_tor]]
            )
            matvec_0 = lambda x: self.apply_M1_0_ten(
                x, [[self.M1_pol_mat[0], self.M0_tor], [self.M1_pol_mat[1], self.M1_tor]]
            )

        # 3D
        else:
            if self.dim == 2:
                M11, M22 = mass_2d.get_M1(self, domain)
                self.M1_mat = spa.bmat(
                    [[spa.kron(M11, self.M0_tor), None], [None, spa.kron(M22, self.M1_tor)]], format="csr"
                )
            else:
                self.M1_mat = mass_3d.get_M1(self, domain)

            matvec = lambda x: self.M1_mat.dot(x)
            matvec_0 = lambda x: self.B1.dot(self.M1_mat.dot(self.B1.T.dot(x)))

        # linaer operators
        self.M1 = spa.linalg.LinearOperator((self.E1.shape[0], self.E1.shape[0]), matvec=matvec)
        self.M1_0 = spa.linalg.LinearOperator((self.E1_0.shape[0], self.E1_0.shape[0]), matvec=matvec_0)

    def __assemble_M2(self, domain, as_tensor=False):
        """
        TODO
        """

        self.M2_as_tensor = as_tensor

        # tensor product 2D poloidal x 1D toroidal
        if as_tensor:
            self.M2_pol_mat = mass_2d.get_M2(self, domain)

            matvec = lambda x: self.apply_M2_ten(
                x, [[self.M2_pol_mat[0], self.M1_tor], [self.M2_pol_mat[1], self.M0_tor]]
            )
            matvec_0 = lambda x: self.apply_M2_0_ten(
                x, [[self.M2_pol_mat[0], self.M1_tor], [self.M2_pol_mat[1], self.M0_tor]]
            )

        # 3D
        else:
            if self.dim == 2:
                M11, M22 = mass_2d.get_M2(self, domain)
                self.M2_mat = spa.bmat(
                    [[spa.kron(M11, self.M1_tor), None], [None, spa.kron(M22, self.M0_tor)]], format="csr"
                )
            else:
                self.M2_mat = mass_3d.get_M2(self, domain)

            matvec = lambda x: self.M2_mat.dot(x)
            matvec_0 = lambda x: self.B2.dot(self.M2_mat.dot(self.B2.T.dot(x)))

        # linear operators
        self.M2 = spa.linalg.LinearOperator((self.E2.shape[0], self.E2.shape[0]), matvec=matvec)
        self.M2_0 = spa.linalg.LinearOperator((self.E2_0.shape[0], self.E2_0.shape[0]), matvec=matvec_0)

    def __assemble_M3(self, domain, as_tensor=False):
        """
        TODO
        """

        self.M3_as_tensor = as_tensor

        # tensor product 2D poloidal x 1D toroidal
        if as_tensor:
            self.M3_pol_mat = mass_2d.get_M3(self, domain)

            matvec = lambda x: self.apply_M3_ten(x, [self.M3_pol_mat, self.M1_tor])
            matvec_0 = lambda x: self.apply_M3_0_ten(x, [self.M3_pol_mat, self.M1_tor])

        # 3D
        else:
            if self.dim == 2:
                self.M3_mat = spa.kron(mass_2d.get_M3(self, domain), self.M1_tor, format="csr")
            else:
                self.M3_mat = mass_3d.get_M3(self, domain)

            matvec = lambda x: self.M3_mat.dot(x)
            matvec_0 = lambda x: self.B3.dot(self.M3_mat.dot(self.B3.T.dot(x)))

        # linear operators
        self.M3 = spa.linalg.LinearOperator((self.E3.shape[0], self.E3.shape[0]), matvec=matvec)
        self.M3_0 = spa.linalg.LinearOperator((self.E3_0.shape[0], self.E3_0.shape[0]), matvec=matvec_0)

    def __assemble_Mv(self, domain, as_tensor=False):
        """
        TODO
        """

        self.Mv_as_tensor = as_tensor

        # tensor product 2D poloidal x 1D toroidal
        if as_tensor:
            self.Mv_pol_mat = mass_2d.get_Mv(self, domain)

            matvec = lambda x: self.apply_Mv_ten(
                x, [[self.Mv_pol_mat[0], self.M0_tor], [self.Mv_pol_mat[1], self.M0_tor]]
            )
            matvec_0 = lambda x: self.apply_Mv_0_ten(
                x, [[self.Mv_pol_mat[0], self.M0_tor], [self.Mv_pol_mat[1], self.M0_tor]]
            )

        # 3D
        else:
            if self.dim == 2:
                M11, M22 = mass_2d.get_Mv(self, domain)
                self.Mv_mat = spa.bmat(
                    [[spa.kron(M11, self.M0_tor), None], [None, spa.kron(M22, self.M0_tor)]], format="csr"
                )
            else:
                self.Mv_mat = mass_3d.get_Mv(self, domain)

            matvec = lambda x: self.Mv_mat.dot(x)
            matvec_0 = lambda x: self.Bv.dot(self.Mv_mat.dot(self.Bv.T.dot(x)))

        # linear operators
        self.Mv = spa.linalg.LinearOperator((self.Ev.shape[0], self.Ev.shape[0]), matvec=matvec)
        self.Mv_0 = spa.linalg.LinearOperator((self.Ev_0.shape[0], self.Ev_0.shape[0]), matvec=matvec_0)

    def assemble_Mk(self, domain, space, as_tensor=False):
        """
        TODO
        """

        if space == "V0":
            self.__assemble_M0(domain, as_tensor)
        elif space == "V1":
            self.__assemble_M1(domain, as_tensor)
        elif space == "V2":
            self.__assemble_M2(domain, as_tensor)
        elif space == "V3":
            self.__assemble_M3(domain, as_tensor)
        elif space == "Vv":
            self.__assemble_Mv(domain, as_tensor)

    # == reshape of flattened 3D coefficients to structure (2D poloidal x 1D toroidal) ===
    def reshape_pol_0(self, coeff):
        """
        TODO
        """

        c_size = coeff.size

        assert c_size == self.E0.shape[0] or c_size == self.E0_0.shape[0]

        if c_size == self.E0.shape[0]:
            coeff0_pol = coeff.reshape(self.E0_pol.shape[0], self.E0_tor.shape[0])
        else:
            coeff0_pol = coeff.reshape(self.E0_pol_0.shape[0], self.E0_tor_0.shape[0])

        return coeff0_pol

    def reshape_pol_1(self, coeff):
        """
        TODO
        """

        c_size = coeff.size

        assert c_size == self.E1.shape[0] or c_size == self.E1_0.shape[0]

        if c_size == self.E1.shape[0]:
            coeff1_pol_1 = coeff[: self.E1_pol.shape[0] * self.E0_tor.shape[0]].reshape(
                self.E1_pol.shape[0], self.E0_tor.shape[0]
            )
            coeff1_pol_3 = coeff[self.E1_pol.shape[0] * self.E0_tor.shape[0] :].reshape(
                self.E0_pol.shape[0], self.E1_tor.shape[0]
            )
        else:
            coeff1_pol_1 = coeff[: self.E1_pol_0.shape[0] * self.E0_tor_0.shape[0]].reshape(
                self.E1_pol_0.shape[0], self.E0_tor_0.shape[0]
            )
            coeff1_pol_3 = coeff[self.E1_pol_0.shape[0] * self.E0_tor_0.shape[0] :].reshape(
                self.E0_pol_0.shape[0], self.E1_tor_0.shape[0]
            )

        return coeff1_pol_1, coeff1_pol_3

    def reshape_pol_2(self, coeff):
        """
        TODO
        """

        c_size = coeff.size

        assert c_size == self.E2.shape[0] or c_size == self.E2_0.shape[0]

        if c_size == self.E2.shape[0]:
            coeff2_pol_1 = coeff[: self.E2_pol.shape[0] * self.E1_tor.shape[0]].reshape(
                self.E2_pol.shape[0], self.E1_tor.shape[0]
            )
            coeff2_pol_3 = coeff[self.E2_pol.shape[0] * self.E1_tor.shape[0] :].reshape(
                self.E3_pol.shape[0], self.E0_tor.shape[0]
            )
        else:
            coeff2_pol_1 = coeff[: self.E2_pol_0.shape[0] * self.E1_tor_0.shape[0]].reshape(
                self.E2_pol_0.shape[0], self.E1_tor_0.shape[0]
            )
            coeff2_pol_3 = coeff[self.E2_pol_0.shape[0] * self.E1_tor_0.shape[0] :].reshape(
                self.E3_pol_0.shape[0], self.E0_tor_0.shape[0]
            )

        return coeff2_pol_1, coeff2_pol_3

    def reshape_pol_3(self, coeff):
        """
        TODO
        """

        c_size = coeff.size

        assert c_size == self.E3.shape[0] or c_size == self.E3_0.shape[0]

        if c_size == self.E3.shape[0]:
            coeff3_pol = coeff.reshape(self.E3_pol.shape[0], self.E1_tor.shape[0])
        else:
            coeff3_pol = coeff.reshape(self.E3_pol_0.shape[0], self.E1_tor_0.shape[0])

        return coeff3_pol

    def reshape_pol_v(self, coeff):
        """
        TODO
        """

        c_size = coeff.size

        assert c_size == self.Ev.shape[0] or c_size == self.Ev_0.shape[0]

        if c_size == self.Ev.shape[0]:
            coeffv_pol_1 = coeff[: self.Ev_pol.shape[0] * self.E0_tor.shape[0]].reshape(
                self.Ev_pol.shape[0], self.E0_tor.shape[0]
            )
            coeffv_pol_3 = coeff[self.Ev_pol.shape[0] * self.E0_tor.shape[0] :].reshape(
                self.E0_pol.shape[0], self.E0_tor.shape[0]
            )

        else:
            coeffv_pol_1 = coeff[: self.Ev_pol_0.shape[0] * self.E0_tor.shape[0]].reshape(
                self.Ev_pol_0.shape[0], self.E0_tor.shape[0]
            )
            coeffv_pol_3 = coeff[self.Ev_pol_0.shape[0] * self.E0_tor.shape[0] :].reshape(
                self.E0_pol.shape[0], self.E0_tor_0.shape[0]
            )

        return coeffv_pol_1, coeffv_pol_3

    # ========= extraction of flattened 3D coefficients to tensor-product space =========
    def extract_0(self, coeff):
        """Reshape flattened 3D 0-form coefficients to tensor-product space.

        Parameters
        ----------
        coeff : numpy.ndarray
            Flattened 3D coefficients.

        Returns
        -------
        coeff0 : numpy.ndarray
            Coefficients in tensor-produce space.
        """

        c_size = coeff.size

        assert c_size == self.E0.shape[0] or c_size == self.E0_0.shape[0]

        if c_size == self.E0.shape[0]:
            coeff0 = self.E0.T.dot(coeff)
        else:
            coeff0 = self.E0_0.T.dot(coeff)

        coeff0 = coeff0.reshape(self.Nbase_0form)

        return coeff0

    def extract_1(self, coeff):
        """Reshape flattened 3D 1-form coefficients to tensor-product space.

        Parameters
        ----------
        coeff : numpy.ndarray
            Flattened 3D coefficients.

        Returns
        -------
        (coeff1_1, coeff1_2, coeff1_3) : tuple of numpy.ndarray
            Coefficients in tensor-produce space.
        """

        c_size = coeff.size

        assert c_size == self.E1.shape[0] or c_size == self.E1_0.shape[0]

        if c_size == self.E1.shape[0]:
            coeff1 = self.E1.T.dot(coeff)
        else:
            coeff1 = self.E1_0.T.dot(coeff)

        coeff1_1, coeff1_2, coeff1_3 = np.split(coeff1, [self.Ntot_1form_cum[0], self.Ntot_1form_cum[1]])

        coeff1_1 = coeff1_1.reshape(self.Nbase_1form[0])
        coeff1_2 = coeff1_2.reshape(self.Nbase_1form[1])
        coeff1_3 = coeff1_3.reshape(self.Nbase_1form[2])

        return coeff1_1, coeff1_2, coeff1_3

    def extract_2(self, coeff):
        """Reshape flattened 3D 2-form coefficients to tensor-product space.

        Parameters
        ----------
        coeff : numpy.ndarray
            Flattened 3D coefficients.

        Returns
        -------
        (coeff2_1, coeff2_2, coeff2_3) : tuple of numpy.ndarray
            Coefficients in tensor-produce space.
        """

        c_size = coeff.size

        assert c_size == self.E2.shape[0] or c_size == self.E2_0.shape[0]

        if c_size == self.E2.shape[0]:
            coeff2 = self.E2.T.dot(coeff)
        else:
            coeff2 = self.E2_0.T.dot(coeff)

        coeff2_1, coeff2_2, coeff2_3 = np.split(coeff2, [self.Ntot_2form_cum[0], self.Ntot_2form_cum[1]])

        coeff2_1 = coeff2_1.reshape(self.Nbase_2form[0])
        coeff2_2 = coeff2_2.reshape(self.Nbase_2form[1])
        coeff2_3 = coeff2_3.reshape(self.Nbase_2form[2])

        return coeff2_1, coeff2_2, coeff2_3

    def extract_3(self, coeff):
        """Reshape flattened 3D 3-form coefficients to tensor-product space.

        Parameters
        ----------
        coeff : numpy.ndarray
            Flattened 3D coefficients.

        Returns
        -------
        coeff3 : numpy.ndarray
            Coefficients in tensor-produce space.
        """

        c_size = coeff.size

        assert c_size == self.E3.shape[0] or c_size == self.E3_0.shape[0]

        if c_size == self.E3.shape[0]:
            coeff3 = self.E3.T.dot(coeff)
        else:
            coeff3 = self.E3_0.T.dot(coeff)

        coeff3 = coeff3.reshape(self.Nbase_3form)

        return coeff3

    def extract_v(self, coeff):
        c_size = coeff.size

        assert c_size == self.Ev.shape[0] or c_size == self.Ev_0.shape[0]

        if c_size == self.Ev.shape[0]:
            coeffv = self.Ev.T.dot(coeff)
        else:
            coeffv = self.Ev_0.T.dot(coeff)

        coeffv_1, coeffv_2, coeffv_3 = np.split(coeffv, [self.Ntot_0form, 2 * self.Ntot_0form])

        coeffv_1 = coeffv_1.reshape(self.Nbase_0form)
        coeffv_2 = coeffv_2.reshape(self.Nbase_0form)
        coeffv_3 = coeffv_3.reshape(self.Nbase_0form)

        return coeffv_1, coeffv_2, coeffv_3

    # =================================================
    def evaluate_NN(self, eta1, eta2, eta3, coeff, which="V0", part="r"):
        """
        Evaluates the spline space [(NN) x Fourier] with coefficients 'coeff' at the point(s) eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or array_like
            1st component of logical evaluation point(s)

        eta2 : double or array_like
            2nd component of logical evaluation point(s)

        eta3 : double or array_like
            3rd component of logical evaluation point(s)

        coeff : array_like
            FEM coefficients

        which : string
            which space (V0 or V1)

        part : string
            real (r) or imaginary (i) part to return

        Returns
        -------
        out : double or array_like
            evaluated FEM field at the point(s) eta = (eta1, eta2, eta3)
        """

        assert part == "r" or part == "i"
        assert which == "V0" or which == "V1"

        # extract coefficients if flattened
        if coeff.ndim == 1:
            if which == "V0":
                coeff = self.extract_0(coeff)
            else:
                coeff = self.extract_1(coeff)[2]

        # check if coefficients have correct shape
        assert coeff.shape[:2] == (self.NbaseN[0], self.NbaseN[1])

        # get real and imaginary part
        coeff_r = np.real(coeff)
        coeff_i = np.imag(coeff)

        # ------ evaluate FEM field at given points --------
        if isinstance(eta1, np.ndarray):
            # tensor-product evaluation
            if eta1.ndim == 1:
                values_r_1 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)
                values_i_1 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)

                eva_2d.evaluate_tensor_product_2d(
                    self.T[0],
                    self.T[1],
                    self.p[0],
                    self.p[1],
                    self.indN[0],
                    self.indN[1],
                    coeff_r[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_r_1,
                    0,
                )
                eva_2d.evaluate_tensor_product_2d(
                    self.T[0],
                    self.T[1],
                    self.p[0],
                    self.p[1],
                    self.indN[0],
                    self.indN[1],
                    coeff_i[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_i_1,
                    0,
                )

                if self.n_tor != 0 and self.basis_tor == "r":
                    values_r_2 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)
                    values_i_2 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)

                    eva_2d.evaluate_tensor_product_2d(
                        self.T[0],
                        self.T[1],
                        self.p[0],
                        self.p[1],
                        self.indN[0],
                        self.indN[1],
                        coeff_r[:, :, 1],
                        eta1,
                        eta2,
                        values_r_2,
                        0,
                    )
                    eva_2d.evaluate_tensor_product_2d(
                        self.T[0],
                        self.T[1],
                        self.p[0],
                        self.p[1],
                        self.indN[0],
                        self.indN[1],
                        coeff_i[:, :, 1],
                        eta1,
                        eta2,
                        values_i_2,
                        0,
                    )

            # matrix evaluation
            else:
                values_r_1 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)
                values_i_1 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)

                eva_2d.evaluate_matrix_2d(
                    self.T[0],
                    self.T[1],
                    self.p[0],
                    self.p[1],
                    self.indN[0],
                    self.indN[1],
                    coeff_r[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_r_1,
                    0,
                )
                eva_2d.evaluate_matrix_2d(
                    self.T[0],
                    self.T[1],
                    self.p[0],
                    self.p[1],
                    self.indN[0],
                    self.indN[1],
                    coeff_i[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_i_1,
                    0,
                )

                if self.n_tor != 0 and self.basis_tor == "r":
                    values_r_2 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)
                    values_i_2 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)

                    eva_2d.evaluate_matrix_2d(
                        self.T[0],
                        self.T[1],
                        self.p[0],
                        self.p[1],
                        self.indN[0],
                        self.indN[1],
                        coeff_r[:, :, 1],
                        eta1,
                        eta2,
                        values_r_2,
                        0,
                    )
                    eva_2d.evaluate_matrix_2d(
                        self.T[0],
                        self.T[1],
                        self.p[0],
                        self.p[1],
                        self.indN[0],
                        self.indN[1],
                        coeff_i[:, :, 1],
                        eta1,
                        eta2,
                        values_i_2,
                        0,
                    )

            # multiply with Fourier basis in third direction
            if self.n_tor == 0:
                out = (values_r_1 + 1j * values_i_1)[:, :, None] * np.ones(eta3.shape, dtype=float)

            else:
                if self.basis_tor == "r":
                    out = (values_r_1 + 1j * values_i_1)[:, :, None] * np.cos(2 * np.pi * self.n_tor * eta3)
                    out += (values_r_2 + 1j * values_i_2)[:, :, None] * np.sin(2 * np.pi * self.n_tor * eta3)

                else:
                    out = (values_r_1 + 1j * values_i_1)[:, :, None] * np.exp(1j * 2 * np.pi * self.n_tor * eta3)

        # --------- evaluate FEM field at given point -------
        else:
            real_1 = eva_2d.evaluate_n_n(
                self.T[0],
                self.T[1],
                self.p[0],
                self.p[1],
                self.indN[0],
                self.indN[1],
                coeff_r[:, :, 0].copy(),
                eta1,
                eta2,
            )
            imag_1 = eva_2d.evaluate_n_n(
                self.T[0],
                self.T[1],
                self.p[0],
                self.p[1],
                self.indN[0],
                self.indN[1],
                coeff_i[:, :, 0].copy(),
                eta1,
                eta2,
            )

            if self.n_tor != 0 and self.basis_tor == "r":
                real_2 = eva_2d.evaluate_n_n(
                    self.T[0], self.T[1], self.p[0], self.p[1], self.indN[0], self.indN[1], coeff_r[:, :, 1], eta1, eta2
                )
                imag_2 = eva_2d.evaluate_n_n(
                    self.T[0], self.T[1], self.p[0], self.p[1], self.indN[0], self.indN[1], coeff_i[:, :, 1], eta1, eta2
                )

            # multiply with Fourier basis in third direction if |n_tor| > 0
            if self.n_tor == 0:
                out = real_1 + 1j * imag_1

            else:
                if self.basis_tor == "r":
                    out = (real_1 + 1j * imag_1) * np.cos(2 * np.pi * self.n_tor * eta3)
                    out += (real_2 + 1j * imag_2) * np.sin(2 * np.pi * self.n_tor * eta3)

                else:
                    out = (real_1 + 1j * imag_1) * np.exp(1j * 2 * np.pi * self.n_tor * eta3)

        # return real or imaginary part
        if part == "r":
            out = np.real(out)
        else:
            out = np.imag(out)

        return out

    # =================================================
    def evaluate_DN(self, eta1, eta2, eta3, coeff, which="V1", part="r"):
        """
        Evaluates the spline space [(DN) x Fourier] with coefficients 'coeff' at the point(s) eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or array_like
            1st component of logical evaluation point(s)

        eta2 : double or array_like
            2nd component of logical evaluation point(s)

        eta3 : double or array_like
            3rd component of logical evaluation point(s)

        coeff : array_like
            FEM coefficients

        which : string
            which space (V1 or V2)

        part : string
            real (r) or imaginary (i) part to return

        Returns
        -------
        out : double or array_like
            evaluated FEM field at the point(s) eta = (eta1, eta2, eta3)
        """

        assert part == "r" or part == "i"
        assert which == "V1" or which == "V2"

        # extract coefficients if flattened
        if coeff.ndim == 1:
            if which == "V1":
                coeff = self.extract_1(coeff)[0]
            else:
                coeff = self.extract_2(coeff)[1]

        # check if coefficients have correct shape
        assert coeff.shape[:2] == (self.NbaseD[0], self.NbaseN[1])

        # get real and imaginary part
        coeff_r = np.real(coeff)
        coeff_i = np.imag(coeff)

        # ------ evaluate FEM field at given points --------
        if isinstance(eta1, np.ndarray):
            # tensor-product evaluation
            if eta1.ndim == 1:
                values_r_1 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)
                values_i_1 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)

                eva_2d.evaluate_tensor_product_2d(
                    self.t[0],
                    self.T[1],
                    self.p[0] - 1,
                    self.p[1],
                    self.indD[0],
                    self.indN[1],
                    coeff_r[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_r_1,
                    11,
                )
                eva_2d.evaluate_tensor_product_2d(
                    self.t[0],
                    self.T[1],
                    self.p[0] - 1,
                    self.p[1],
                    self.indD[0],
                    self.indN[1],
                    coeff_i[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_i_1,
                    11,
                )

                if self.n_tor != 0 and self.basis_tor == "r":
                    values_r_2 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)
                    values_i_2 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)

                    eva_2d.evaluate_tensor_product_2d(
                        self.t[0],
                        self.T[1],
                        self.p[0] - 1,
                        self.p[1],
                        self.indD[0],
                        self.indN[1],
                        coeff_r[:, :, 1],
                        eta1,
                        eta2,
                        values_r_2,
                        11,
                    )
                    eva_2d.evaluate_tensor_product_2d(
                        self.t[0],
                        self.T[1],
                        self.p[0] - 1,
                        self.p[1],
                        self.indD[0],
                        self.indN[1],
                        coeff_i[:, :, 1],
                        eta1,
                        eta2,
                        values_i_2,
                        11,
                    )

            # matrix evaluation
            else:
                values_r_1 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)
                values_i_1 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)

                eva_2d.evaluate_matrix_2d(
                    self.t[0],
                    self.T[1],
                    self.p[0] - 1,
                    self.p[1],
                    self.indD[0],
                    self.indN[1],
                    coeff_r[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_r_1,
                    11,
                )
                eva_2d.evaluate_matrix_2d(
                    self.t[0],
                    self.T[1],
                    self.p[0] - 1,
                    self.p[1],
                    self.indD[0],
                    self.indN[1],
                    coeff_i[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_i_1,
                    11,
                )

                if self.n_tor != 0 and self.basis_tor == "r":
                    values_r_2 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)
                    values_i_2 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)

                    eva_2d.evaluate_matrix_2d(
                        self.t[0],
                        self.T[1],
                        self.p[0] - 1,
                        self.p[1],
                        self.indD[0],
                        self.indN[1],
                        coeff_r[:, :, 1],
                        eta1,
                        eta2,
                        values_r_2,
                        11,
                    )
                    eva_2d.evaluate_matrix_2d(
                        self.t[0],
                        self.T[1],
                        self.p[0] - 1,
                        self.p[1],
                        self.indD[0],
                        self.indN[1],
                        coeff_i[:, :, 1],
                        eta1,
                        eta2,
                        values_i_2,
                        11,
                    )

            # multiply with Fourier basis in third direction
            if self.n_tor == 0:
                out = (values_r_1 + 1j * values_i_1)[:, :, None] * np.ones(eta3.shape, dtype=float)

            else:
                if self.basis_tor == "r":
                    out = (values_r_1 + 1j * values_i_1)[:, :, None] * np.cos(2 * np.pi * self.n_tor * eta3)
                    out += (values_r_2 + 1j * values_i_2)[:, :, None] * np.sin(2 * np.pi * self.n_tor * eta3)

                else:
                    out = (values_r_1 + 1j * values_i_1)[:, :, None] * np.exp(1j * 2 * np.pi * self.n_tor * eta3)

        # --------- evaluate FEM field at given point -------
        else:
            real_1 = eva_2d.evaluate_d_n(
                self.t[0],
                self.T[1],
                self.p[0] - 1,
                self.p[1],
                self.indD[0],
                self.indN[1],
                coeff_r[:, :, 0].copy(),
                eta1,
                eta2,
            )
            imag_1 = eva_2d.evaluate_d_n(
                self.t[0],
                self.T[1],
                self.p[0] - 1,
                self.p[1],
                self.indD[0],
                self.indN[1],
                coeff_i[:, :, 0].copy(),
                eta1,
                eta2,
            )

            if self.n_tor != 0 and self.basis_tor == "r":
                real_2 = eva_2d.evaluate_d_n(
                    self.t[0],
                    self.T[1],
                    self.p[0] - 1,
                    self.p[1],
                    self.indD[0],
                    self.indN[1],
                    coeff_r[:, :, 1],
                    eta1,
                    eta2,
                )
                imag_2 = eva_2d.evaluate_d_n(
                    self.t[0],
                    self.T[1],
                    self.p[0] - 1,
                    self.p[1],
                    self.indD[0],
                    self.indN[1],
                    coeff_i[:, :, 1],
                    eta1,
                    eta2,
                )

            # multiply with Fourier basis in third direction if |n_tor| > 0
            if self.n_tor == 0:
                out = real_1 + 1j * imag_1

            else:
                if self.basis_tor == "r":
                    out = (real_1 + 1j * imag_1) * np.cos(2 * np.pi * self.n_tor * eta3)
                    out += (real_2 + 1j * imag_2) * np.sin(2 * np.pi * self.n_tor * eta3)

                else:
                    out = (real_1 + 1j * imag_1) * np.exp(1j * 2 * np.pi * self.n_tor * eta3)

        # return real or imaginary part
        if part == "r":
            out = np.real(out)
        else:
            out = np.imag(out)

        return out

    # =================================================
    def evaluate_ND(self, eta1, eta2, eta3, coeff, which="V2", part="r"):
        """
        Evaluates the spline space [(ND) x Fourier] with coefficients 'coeff' at the point(s) eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or array_like
            1st component of logical evaluation point(s)

        eta2 : double or array_like
            2nd component of logical evaluation point(s)

        eta3 : double or array_like
            3rd component of logical evaluation point(s)

        coeff : array_like
            FEM coefficients

        which : string
            which space (V1 or V2)

        part : string
            real (r) or imaginary (i) part to return

        Returns
        -------
        out : double or array_like
            evaluated FEM field at the point(s) eta = (eta1, eta2, eta3)
        """

        assert part == "r" or part == "i"
        assert which == "V1" or which == "V2"

        # extract coefficients if flattened
        if coeff.ndim == 1:
            if which == "V1":
                coeff = self.extract_1(coeff)[1]
            else:
                coeff = self.extract_2(coeff)[0]

        # check if coefficients have correct shape
        assert coeff.shape[:2] == (self.NbaseN[0], self.NbaseD[1])

        # get real and imaginary part
        coeff_r = np.real(coeff)
        coeff_i = np.imag(coeff)

        # ------ evaluate FEM field at given points --------
        if isinstance(eta1, np.ndarray):
            # tensor-product evaluation
            if eta1.ndim == 1:
                values_r_1 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)
                values_i_1 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)

                eva_2d.evaluate_tensor_product_2d(
                    self.T[0],
                    self.t[1],
                    self.p[0],
                    self.p[1] - 1,
                    self.indN[0],
                    self.indD[1],
                    coeff_r[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_r_1,
                    12,
                )
                eva_2d.evaluate_tensor_product_2d(
                    self.T[0],
                    self.t[1],
                    self.p[0],
                    self.p[1] - 1,
                    self.indN[0],
                    self.indD[1],
                    coeff_i[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_i_1,
                    12,
                )

                if self.n_tor != 0 and self.basis_tor == "r":
                    values_r_2 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)
                    values_i_2 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)

                    eva_2d.evaluate_tensor_product_2d(
                        self.T[0],
                        self.t[1],
                        self.p[0],
                        self.p[1] - 1,
                        self.indN[0],
                        self.indD[1],
                        coeff_r[:, :, 1].copy(),
                        eta1,
                        eta2,
                        values_r_2,
                        12,
                    )
                    eva_2d.evaluate_tensor_product_2d(
                        self.T[0],
                        self.t[1],
                        self.p[0],
                        self.p[1] - 1,
                        self.indN[0],
                        self.indD[1],
                        coeff_i[:, :, 1].copy(),
                        eta1,
                        eta2,
                        values_i_2,
                        12,
                    )

            # matrix evaluation
            else:
                values_r_1 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)
                values_i_1 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)

                eva_2d.evaluate_matrix_2d(
                    self.T[0],
                    self.t[1],
                    self.p[0],
                    self.p[1] - 1,
                    self.indN[0],
                    self.indD[1],
                    coeff_r[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_r_1,
                    12,
                )
                eva_2d.evaluate_matrix_2d(
                    self.T[0],
                    self.t[1],
                    self.p[0],
                    self.p[1] - 1,
                    self.indN[0],
                    self.indD[1],
                    coeff_i[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_i_1,
                    12,
                )

                if self.n_tor != 0 and self.basis_tor == "r":
                    values_r_2 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)
                    values_i_2 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)

                    eva_2d.evaluate_matrix_2d(
                        self.T[0],
                        self.t[1],
                        self.p[0],
                        self.p[1] - 1,
                        self.indN[0],
                        self.indD[1],
                        coeff_r[:, :, 1].copy(),
                        eta1,
                        eta2,
                        values_r_2,
                        12,
                    )
                    eva_2d.evaluate_matrix_2d(
                        self.T[0],
                        self.t[1],
                        self.p[0],
                        self.p[1] - 1,
                        self.indN[0],
                        self.indD[1],
                        coeff_i[:, :, 1].copy(),
                        eta1,
                        eta2,
                        values_i_2,
                        12,
                    )

            # multiply with Fourier basis in third direction
            if self.n_tor == 0:
                out = (values_r_1 + 1j * values_i_1)[:, :, None] * np.ones(eta3.shape, dtype=float)

            else:
                if self.basis_tor == "r":
                    out = (values_r_1 + 1j * values_i_1)[:, :, None] * np.cos(2 * np.pi * self.n_tor * eta3)
                    out += (values_r_2 + 1j * values_i_2)[:, :, None] * np.sin(2 * np.pi * self.n_tor * eta3)

                else:
                    out = (values_r_1 + 1j * values_i_1)[:, :, None] * np.exp(1j * 2 * np.pi * self.n_tor * eta3)

        # --------- evaluate FEM field at given point -------
        else:
            real_1 = eva_2d.evaluate_n_d(
                self.T[0],
                self.t[1],
                self.p[0],
                self.p[1] - 1,
                self.indN[0],
                self.indD[1],
                coeff_r[:, :, 0].copy(),
                eta1,
                eta2,
            )
            imag_1 = eva_2d.evaluate_n_d(
                self.T[0],
                self.t[1],
                self.p[0],
                self.p[1] - 1,
                self.indN[0],
                self.indD[1],
                coeff_i[:, :, 0].copy(),
                eta1,
                eta2,
            )

            if self.n_tor != 0 and self.basis_tor == "r":
                real_2 = eva_2d.evaluate_n_d(
                    self.T[0],
                    self.t[1],
                    self.p[0],
                    self.p[1] - 1,
                    self.indN[0],
                    self.indD[1],
                    coeff_r[:, :, 1].copy(),
                    eta1,
                    eta2,
                )
                imag_2 = eva_2d.evaluate_n_d(
                    self.T[0],
                    self.t[1],
                    self.p[0],
                    self.p[1] - 1,
                    self.indN[0],
                    self.indD[1],
                    coeff_i[:, :, 1].copy(),
                    eta1,
                    eta2,
                )

            # multiply with Fourier basis in third direction if |n_tor| > 0
            if self.n_tor == 0:
                out = real_1 + 1j * imag_1

            else:
                if self.basis_tor == "r":
                    out = (real_1 + 1j * imag_1) * np.cos(2 * np.pi * self.n_tor * eta3)
                    out += (real_2 + 1j * imag_2) * np.sin(2 * np.pi * self.n_tor * eta3)

                else:
                    out = (real_1 + 1j * imag_1) * np.exp(1j * 2 * np.pi * self.n_tor * eta3)

        # return real or imaginary part
        if part == "r":
            out = np.real(out)
        else:
            out = np.imag(out)

        return out

    # =================================================
    def evaluate_DD(self, eta1, eta2, eta3, coeff, which="V3", part="r"):
        """
        Evaluates the spline space [(DD) x Fourier] with coefficients 'coeff' at the point(s) eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or array_like
            1st component of logical evaluation point(s)

        eta2 : double or array_like
            2nd component of logical evaluation point(s)

        eta3 : double or array_like
            3rd component of logical evaluation point(s)

        coeff : array_like
            FEM coefficients

        part : string
            real (r) or imaginary (i) part to return

        which : string
            which space (V2 or V3)

        part : string
            real (r) or imaginary (i) part to return

        Returns
        -------
        out : double or array_like
            evaluated FEM field at the point(s) eta = (eta1, eta2, eta3)
        """

        assert part == "r" or part == "i"
        assert which == "V2" or which == "V3"

        # extract coefficients if flattened
        if coeff.ndim == 1:
            if which == "V2":
                coeff = self.extract_2(coeff)[2]
            else:
                coeff = self.extract_3(coeff)

        # check if coefficients have correct shape
        assert coeff.shape[:2] == (self.NbaseD[0], self.NbaseD[1])

        # get real and imaginary part
        coeff_r = np.real(coeff)
        coeff_i = np.imag(coeff)

        # ------ evaluate FEM field at given points --------
        if isinstance(eta1, np.ndarray):
            # tensor-product evaluation
            if eta1.ndim == 1:
                values_r_1 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)
                values_i_1 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)

                eva_2d.evaluate_tensor_product_2d(
                    self.t[0],
                    self.t[1],
                    self.p[0] - 1,
                    self.p[1] - 1,
                    self.indD[0],
                    self.indD[1],
                    coeff_r[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_r_1,
                    2,
                )
                eva_2d.evaluate_tensor_product_2d(
                    self.t[0],
                    self.t[1],
                    self.p[0] - 1,
                    self.p[1] - 1,
                    self.indD[0],
                    self.indD[1],
                    coeff_i[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_i_1,
                    2,
                )

                if self.n_tor != 0 and self.basis_tor == "r":
                    values_r_2 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)
                    values_i_2 = np.empty((eta1.shape[0], eta2.shape[0]), dtype=float)

                    eva_2d.evaluate_tensor_product_2d(
                        self.t[0],
                        self.t[1],
                        self.p[0] - 1,
                        self.p[1] - 1,
                        self.indD[0],
                        self.indD[1],
                        coeff_r[:, :, 1],
                        eta1,
                        eta2,
                        values_r_2,
                        2,
                    )
                    eva_2d.evaluate_tensor_product_2d(
                        self.t[0],
                        self.t[1],
                        self.p[0] - 1,
                        self.p[1] - 1,
                        self.indD[0],
                        self.indD[1],
                        coeff_i[:, :, 1],
                        eta1,
                        eta2,
                        values_i_2,
                        2,
                    )

            # matrix evaluation
            else:
                values_r_1 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)
                values_i_1 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)

                eva_2d.evaluate_matrix_2d(
                    self.t[0],
                    self.t[1],
                    self.p[0] - 1,
                    self.p[1] - 1,
                    self.indD[0],
                    self.indD[1],
                    coeff_r[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_r_1,
                    2,
                )
                eva_2d.evaluate_matrix_2d(
                    self.t[0],
                    self.t[1],
                    self.p[0] - 1,
                    self.p[1] - 1,
                    self.indD[0],
                    self.indD[1],
                    coeff_i[:, :, 0].copy(),
                    eta1,
                    eta2,
                    values_i_1,
                    2,
                )

                if self.n_tor != 0 and self.basis_tor == "r":
                    values_r_2 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)
                    values_i_2 = np.empty((eta1.shape[0], eta2.shape[1]), dtype=float)

                    eva_2d.evaluate_matrix_2d(
                        self.t[0],
                        self.t[1],
                        self.p[0] - 1,
                        self.p[1] - 1,
                        self.indD[0],
                        self.indD[1],
                        coeff_r[:, :, 1],
                        eta1,
                        eta2,
                        values_r_2,
                        2,
                    )
                    eva_2d.evaluate_matrix_2d(
                        self.t[0],
                        self.t[1],
                        self.p[0] - 1,
                        self.p[1] - 1,
                        self.indD[0],
                        self.indD[1],
                        coeff_i[:, :, 1],
                        eta1,
                        eta2,
                        values_i_2,
                        2,
                    )

            # multiply with Fourier basis in third direction
            if self.n_tor == 0:
                out = (values_r_1 + 1j * values_i_1)[:, :, None] * np.ones(eta3.shape, dtype=float)

            else:
                if self.basis_tor == "r":
                    out = (values_r_1 + 1j * values_i_1)[:, :, None] * np.cos(2 * np.pi * self.n_tor * eta3)
                    out += (values_r_2 + 1j * values_i_2)[:, :, None] * np.sin(2 * np.pi * self.n_tor * eta3)

                else:
                    out = (values_r_1 + 1j * values_i_1)[:, :, None] * np.exp(1j * 2 * np.pi * self.n_tor * eta3)

        # --------- evaluate FEM field at given point -------
        else:
            real_1 = eva_2d.evaluate_d_d(
                self.t[0],
                self.t[1],
                self.p[0] - 1,
                self.p[1] - 1,
                self.indD[0],
                self.indD[1],
                coeff_r[:, :, 0].copy(),
                eta1,
                eta2,
            )
            imag_1 = eva_2d.evaluate_d_d(
                self.t[0],
                self.t[1],
                self.p[0] - 1,
                self.p[1] - 1,
                self.indD[0],
                self.indD[1],
                coeff_i[:, :, 0].copy(),
                eta1,
                eta2,
            )

            if self.n_tor != 0 and self.basis_tor == "r":
                real_2 = eva_2d.evaluate_d_d(
                    self.t[0],
                    self.t[1],
                    self.p[0] - 1,
                    self.p[1] - 1,
                    self.indD[0],
                    self.indD[1],
                    coeff_r[:, :, 1],
                    eta1,
                    eta2,
                )
                imag_2 = eva_2d.evaluate_d_d(
                    self.t[0],
                    self.t[1],
                    self.p[0] - 1,
                    self.p[1] - 1,
                    self.indD[0],
                    self.indD[1],
                    coeff_i[:, :, 1],
                    eta1,
                    eta2,
                )

            # multiply with Fourier basis in third direction if |n_tor| > 0
            if self.n_tor == 0:
                out = real_1 + 1j * imag_1

            else:
                if self.basis_tor == "r":
                    out = (real_1 + 1j * imag_1) * np.cos(2 * np.pi * self.n_tor * eta3)
                    out += (real_2 + 1j * imag_2) * np.sin(2 * np.pi * self.n_tor * eta3)

                else:
                    out = (real_1 + 1j * imag_1) * np.exp(1j * 2 * np.pi * self.n_tor * eta3)

        # return real or imaginary part
        if part == "r":
            out = np.real(out)
        else:
            out = np.imag(out)

        return out

    # =================================================
    def evaluate_NNN(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (NNN) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point

        eta2 : double or np.ndarray
            2nd component of logical evaluation point

        eta3 : double or np.ndarray
            3rd component of logical evaluation point

        coeff : array_like
            FEM coefficients

        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """

        if coeff.ndim == 1:
            coeff = self.extract_0(coeff)

        if isinstance(eta1, np.ndarray):
            # tensor-product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(
                    self.T[0],
                    self.T[1],
                    self.T[2],
                    self.p[0],
                    self.p[1],
                    self.p[2],
                    self.indN[0],
                    self.indN[1],
                    self.indN[2],
                    coeff,
                    eta1,
                    eta2,
                    eta3,
                    values,
                    0,
                )

            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(
                        self.T[0],
                        self.T[1],
                        self.T[2],
                        self.p[0],
                        self.p[1],
                        self.p[2],
                        self.indN[0],
                        self.indN[1],
                        self.indN[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        eta1.shape[0],
                        eta2.shape[1],
                        eta3.shape[2],
                        values,
                        0,
                    )

                # `eta1` is a dense meshgrid. Process each point as default.
                else:
                    eva_3d.evaluate_matrix(
                        self.T[0],
                        self.T[1],
                        self.T[2],
                        self.p[0],
                        self.p[1],
                        self.p[2],
                        self.indN[0],
                        self.indN[1],
                        self.indN[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        values,
                        0,
                    )

            return values

        else:
            return eva_3d.evaluate_n_n_n(
                self.T[0],
                self.T[1],
                self.T[2],
                self.p[0],
                self.p[1],
                self.p[2],
                self.indN[0],
                self.indN[1],
                self.indN[2],
                coeff,
                eta1,
                eta2,
                eta3,
            )

    # =================================================
    def evaluate_DNN(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (DNN) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point

        eta2 : double or np.ndarray
            2nd component of logical evaluation point

        eta3 : double or np.ndarray
            3rd component of logical evaluation point

        coeff : array_like
            FEM coefficients

        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """

        if coeff.ndim == 1:
            coeff = self.extract_1(coeff)[0]

        if isinstance(eta1, np.ndarray):
            # tensor product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(
                    self.t[0],
                    self.T[1],
                    self.T[2],
                    self.p[0] - 1,
                    self.p[1],
                    self.p[2],
                    self.indD[0],
                    self.indN[1],
                    self.indN[2],
                    coeff,
                    eta1,
                    eta2,
                    eta3,
                    values,
                    11,
                )

            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(
                        self.t[0],
                        self.T[1],
                        self.T[2],
                        self.p[0] - 1,
                        self.p[1],
                        self.p[2],
                        self.indD[0],
                        self.indN[1],
                        self.indN[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        eta1.shape[0],
                        eta2.shape[1],
                        eta3.shape[2],
                        values,
                        11,
                    )
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
                    eva_3d.evaluate_matrix(
                        self.t[0],
                        self.T[1],
                        self.T[2],
                        self.p[0] - 1,
                        self.p[1],
                        self.p[2],
                        self.indD[0],
                        self.indN[1],
                        self.indN[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        values,
                        11,
                    )

            return values

        else:
            return eva_3d.evaluate_d_n_n(
                self.t[0],
                self.T[1],
                self.T[2],
                self.p[0] - 1,
                self.p[1],
                self.p[2],
                self.indD[0],
                self.indN[1],
                self.indN[2],
                coeff,
                eta1,
                eta2,
                eta3,
            )

    # =================================================
    def evaluate_NDN(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (NDN) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point

        eta2 : double or np.ndarray
            2nd component of logical evaluation point

        eta3 : double or np.ndarray
            3rd component of logical evaluation point

        coeff : array_like
            FEM coefficients

        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """

        if coeff.ndim == 1:
            coeff = self.extract_1(coeff)[1]

        if isinstance(eta1, np.ndarray):
            # tensor product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(
                    self.T[0],
                    self.t[1],
                    self.T[2],
                    self.p[0],
                    self.p[1] - 1,
                    self.p[2],
                    self.indN[0],
                    self.indD[1],
                    self.indN[2],
                    coeff,
                    eta1,
                    eta2,
                    eta3,
                    values,
                    12,
                )

            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(
                        self.T[0],
                        self.t[1],
                        self.T[2],
                        self.p[0],
                        self.p[1] - 1,
                        self.p[2],
                        self.indN[0],
                        self.indD[1],
                        self.indN[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        eta1.shape[0],
                        eta2.shape[1],
                        eta3.shape[2],
                        values,
                        12,
                    )
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
                    eva_3d.evaluate_matrix(
                        self.T[0],
                        self.t[1],
                        self.T[2],
                        self.p[0],
                        self.p[1] - 1,
                        self.p[2],
                        self.indN[0],
                        self.indD[1],
                        self.indN[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        values,
                        12,
                    )

            return values

        else:
            return eva_3d.evaluate_n_d_n(
                self.T[0],
                self.t[1],
                self.T[2],
                self.p[0],
                self.p[1] - 1,
                self.p[2],
                self.indN[0],
                self.indD[1],
                self.indN[2],
                coeff,
                eta1,
                eta2,
                eta3,
            )

    # =================================================
    def evaluate_NND(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (NND) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point

        eta2 : double or np.ndarray
            2nd component of logical evaluation point

        eta3 : double or np.ndarray
            3rd component of logical evaluation point

        coeff : array_like
            FEM coefficients

        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """

        if coeff.ndim == 1:
            coeff = self.extract_1(coeff)[2]

        if isinstance(eta1, np.ndarray):
            # tensor product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(
                    self.T[0],
                    self.T[1],
                    self.t[2],
                    self.p[0],
                    self.p[1],
                    self.p[2] - 1,
                    self.indN[0],
                    self.indN[1],
                    self.indD[2],
                    coeff,
                    eta1,
                    eta2,
                    eta3,
                    values,
                    13,
                )

            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(
                        self.T[0],
                        self.T[1],
                        self.t[2],
                        self.p[0],
                        self.p[1],
                        self.p[2] - 1,
                        self.indN[0],
                        self.indN[1],
                        self.indD[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        eta1.shape[0],
                        eta2.shape[1],
                        eta3.shape[2],
                        values,
                        13,
                    )
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
                    eva_3d.evaluate_matrix(
                        self.T[0],
                        self.T[1],
                        self.t[2],
                        self.p[0],
                        self.p[1],
                        self.p[2] - 1,
                        self.indN[0],
                        self.indN[1],
                        self.indD[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        values,
                        13,
                    )

            return values

        else:
            return eva_3d.evaluate_n_n_d(
                self.T[0],
                self.T[1],
                self.t[2],
                self.p[0],
                self.p[1],
                self.p[2] - 1,
                self.indN[0],
                self.indN[1],
                self.indD[2],
                coeff,
                eta1,
                eta2,
                eta3,
            )

    # =================================================
    def evaluate_NDD(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (NDD) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point

        eta2 : double or np.ndarray
            2nd component of logical evaluation point

        eta3 : double or np.ndarray
            3rd component of logical evaluation point

        coeff : array_like
            FEM coefficients

        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """

        if coeff.ndim == 1:
            coeff = self.extract_2(coeff)[0]

        if isinstance(eta1, np.ndarray):
            # tensor product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(
                    self.T[0],
                    self.t[1],
                    self.t[2],
                    self.p[0],
                    self.p[1] - 1,
                    self.p[2] - 1,
                    self.indN[0],
                    self.indD[1],
                    self.indD[2],
                    coeff,
                    eta1,
                    eta2,
                    eta3,
                    values,
                    21,
                )

            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(
                        self.T[0],
                        self.t[1],
                        self.t[2],
                        self.p[0],
                        self.p[1] - 1,
                        self.p[2] - 1,
                        self.indN[0],
                        self.indD[1],
                        self.indD[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        eta1.shape[0],
                        eta2.shape[1],
                        eta3.shape[2],
                        values,
                        21,
                    )
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
                    eva_3d.evaluate_matrix(
                        self.T[0],
                        self.t[1],
                        self.t[2],
                        self.p[0],
                        self.p[1] - 1,
                        self.p[2] - 1,
                        self.indN[0],
                        self.indD[1],
                        self.indD[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        values,
                        21,
                    )

            return values

        else:
            return eva_3d.evaluate_n_d_d(
                self.T[0],
                self.t[1],
                self.t[2],
                self.p[0],
                self.p[1] - 1,
                self.p[2] - 1,
                self.indN[0],
                self.indD[1],
                self.indD[2],
                coeff,
                eta1,
                eta2,
                eta3,
            )

    # =================================================
    def evaluate_DND(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (DND) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point

        eta2 : double or np.ndarray
            2nd component of logical evaluation point

        eta3 : double or np.ndarray
            3rd component of logical evaluation point

        coeff : array_like
            FEM coefficients

        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """

        if coeff.ndim == 1:
            coeff = self.extract_2(coeff)[1]

        if isinstance(eta1, np.ndarray):
            # tensor product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(
                    self.t[0],
                    self.T[1],
                    self.t[2],
                    self.p[0] - 1,
                    self.p[1],
                    self.p[2] - 1,
                    self.indD[0],
                    self.indN[1],
                    self.indD[2],
                    coeff,
                    eta1,
                    eta2,
                    eta3,
                    values,
                    22,
                )

            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(
                        self.t[0],
                        self.T[1],
                        self.t[2],
                        self.p[0] - 1,
                        self.p[1],
                        self.p[2] - 1,
                        self.indD[0],
                        self.indN[1],
                        self.indD[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        eta1.shape[0],
                        eta2.shape[1],
                        eta3.shape[2],
                        values,
                        22,
                    )
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
                    eva_3d.evaluate_matrix(
                        self.t[0],
                        self.T[1],
                        self.t[2],
                        self.p[0] - 1,
                        self.p[1],
                        self.p[2] - 1,
                        self.indD[0],
                        self.indN[1],
                        self.indD[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        values,
                        22,
                    )

            return values

        else:
            return eva_3d.evaluate_d_n_d(
                self.t[0],
                self.T[1],
                self.t[2],
                self.p[0] - 1,
                self.p[1],
                self.p[2] - 1,
                self.indD[0],
                self.indN[1],
                self.indD[2],
                coeff,
                eta1,
                eta2,
                eta3,
            )

    # =================================================
    def evaluate_DDN(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (DDN) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point

        eta2 : double or np.ndarray
            2nd component of logical evaluation point

        eta3 : double or np.ndarray
            3rd component of logical evaluation point

        coeff : array_like
            FEM coefficients

        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """

        if coeff.ndim == 1:
            coeff = self.extract_2(coeff)[2]

        if isinstance(eta1, np.ndarray):
            # tensor product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(
                    self.t[0],
                    self.t[1],
                    self.T[2],
                    self.p[0] - 1,
                    self.p[1] - 1,
                    self.p[2],
                    self.indD[0],
                    self.indD[1],
                    self.indN[2],
                    coeff,
                    eta1,
                    eta2,
                    eta3,
                    values,
                    23,
                )

            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(
                        self.t[0],
                        self.t[1],
                        self.T[2],
                        self.p[0] - 1,
                        self.p[1] - 1,
                        self.p[2],
                        self.indD[0],
                        self.indD[1],
                        self.indN[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        eta1.shape[0],
                        eta2.shape[1],
                        eta3.shape[2],
                        values,
                        23,
                    )
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
                    eva_3d.evaluate_matrix(
                        self.t[0],
                        self.t[1],
                        self.T[2],
                        self.p[0] - 1,
                        self.p[1] - 1,
                        self.p[2],
                        self.indD[0],
                        self.indD[1],
                        self.indN[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        values,
                        23,
                    )

            return values

        else:
            return eva_3d.evaluate_d_d_n(
                self.t[0],
                self.t[1],
                self.T[2],
                self.p[0] - 1,
                self.p[1] - 1,
                self.p[2],
                self.indD[0],
                self.indD[1],
                self.indN[2],
                coeff,
                eta1,
                eta2,
                eta3,
            )

    # =================================================
    def evaluate_DDD(self, eta1, eta2, eta3, coeff):
        """
        Evaluates the spline space (DDD) with coefficients 'coeff' at the point eta = (eta1, eta2, eta3).

        Parameters
        ----------
        eta1 : double or np.ndarray
            1st component of logical evaluation point

        eta2 : double or np.ndarray
            2nd component of logical evaluation point

        eta3 : double or np.ndarray
            3rd component of logical evaluation point

        coeff : array_like
            FEM coefficients

        Returns
        -------
        value : double
            evaluated FEM field at the point eta = (eta1, eta2, eta3)
        """

        if coeff.ndim == 1:
            coeff = self.extract_3(coeff)

        if isinstance(eta1, np.ndarray):
            # tensor product evaluation
            if eta1.ndim == 1:
                values = np.empty((eta1.size, eta2.size, eta3.size), dtype=float)
                eva_3d.evaluate_tensor_product(
                    self.t[0],
                    self.t[1],
                    self.t[2],
                    self.p[0] - 1,
                    self.p[1] - 1,
                    self.p[2] - 1,
                    self.indD[0],
                    self.indD[1],
                    self.indD[2],
                    coeff,
                    eta1,
                    eta2,
                    eta3,
                    values,
                    3,
                )

            # matrix evaluation
            else:
                values = np.empty((eta1.shape[0], eta2.shape[1], eta3.shape[2]), dtype=float)
                # `eta1` is a sparse meshgrid.
                if max(eta1.shape) == eta1.size:
                    eva_3d.evaluate_sparse(
                        self.t[0],
                        self.t[1],
                        self.t[2],
                        self.p[0] - 1,
                        self.p[1] - 1,
                        self.p[2] - 1,
                        self.indD[0],
                        self.indD[1],
                        self.indD[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        eta1.shape[0],
                        eta2.shape[1],
                        eta3.shape[2],
                        values,
                        3,
                    )
                # `eta1` is a dense meshgrid. Process each point as default.
                else:
                    eva_3d.evaluate_matrix(
                        self.t[0],
                        self.t[1],
                        self.t[2],
                        self.p[0] - 1,
                        self.p[1] - 1,
                        self.p[2] - 1,
                        self.indD[0],
                        self.indD[1],
                        self.indD[2],
                        coeff,
                        eta1,
                        eta2,
                        eta3,
                        values,
                        3,
                    )

            return values

        else:
            return eva_3d.evaluate_d_d_d(
                self.t[0],
                self.t[1],
                self.t[2],
                self.p[0] - 1,
                self.p[1] - 1,
                self.p[2] - 1,
                self.indD[0],
                self.indD[1],
                self.indD[2],
                coeff,
                eta1,
                eta2,
                eta3,
            )
