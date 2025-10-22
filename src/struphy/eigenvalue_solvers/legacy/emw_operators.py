# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Class for 2D/3D linear MHD projection operators.
"""

import scipy.sparse as spa

import struphy.eigenvalue_solvers.kernels_3d as ker
import struphy.eigenvalue_solvers.legacy.mass_matrices_3d_pre as mass_3d_pre
from struphy.utils.arrays import xp


class EMW_operators:
    """
    Define the needed operator for the ceratain model

    Parameters
    ----------
        domain : obj
            Domain object.

        space : obj
            Tensor_spline_space object.

        eq_MHD : obj
            Equilibrium_mhd object.


    Notes
    -----
        Implemented operators
        ===================================================== =========================== =====================
        operator                                              dim of matrix               verification method
        ===================================================== =========================== =====================
        R1  = G B2 eps_ijk lambda_j^1 lambda_k^1 1/sqrt(g)     R^{N^1 x N^1} r
        ===================================================== =========================== =====================

    """

    # def __init__(self, space, equilibrium, domain, basis_e):
    def __init__(self, DOMAIN, SPACES, EQUILIBRIUM=None):
        # create objects
        self.DOMAIN = DOMAIN
        self.SPACES = SPACES
        self.EQUILIBRIUM = EQUILIBRIUM

        self.dim_V1 = self.SPACES.Ntot_1form_cum[-1]
        self.dim_V2 = self.SPACES.Ntot_2form_cum[-1]

        # Build the rotation operator
        weight = [self.__weight_1, self.__weight_2, self.__weight_3]
        self.__assemble_M1_cross(weight)

    # ================ Build the R1 operator =====================
    def __weight_1(self, eta1, eta2, eta3):
        det_g = 1.0 / abs(self.DOMAIN.evaluate(eta1, eta2, eta3, "det_df"))
        G_11 = self.DOMAIN.evaluate(eta1, eta2, eta3, "g_11")
        G_12 = self.DOMAIN.evaluate(eta1, eta2, eta3, "g_12")
        G_13 = self.DOMAIN.evaluate(eta1, eta2, eta3, "g_13")
        B0_1 = self.EQUILIBRIUM.b2_eq_1(eta1, eta2, eta3)
        B0_2 = self.EQUILIBRIUM.b2_eq_2(eta1, eta2, eta3)
        B0_3 = self.EQUILIBRIUM.b2_eq_3(eta1, eta2, eta3)
        return det_g * (G_11 * B0_1 + G_12 * B0_2 + G_13 * B0_3)

    def __weight_2(self, eta1, eta2, eta3):
        det_g = 1.0 / abs(self.DOMAIN.evaluate(eta1, eta2, eta3, "det_df"))
        G_21 = self.DOMAIN.evaluate(eta1, eta2, eta3, "g_21")
        G_22 = self.DOMAIN.evaluate(eta1, eta2, eta3, "g_22")
        G_23 = self.DOMAIN.evaluate(eta1, eta2, eta3, "g_23")
        B0_1 = self.EQUILIBRIUM.b2_eq_1(eta1, eta2, eta3)
        B0_2 = self.EQUILIBRIUM.b2_eq_2(eta1, eta2, eta3)
        B0_3 = self.EQUILIBRIUM.b2_eq_3(eta1, eta2, eta3)
        return det_g * (G_21 * B0_1 + G_22 * B0_2 + G_23 * B0_3)

    def __weight_3(self, eta1, eta2, eta3):
        det_g = 1.0 / abs(self.DOMAIN.evaluate(eta1, eta2, eta3, "det_df"))
        G_31 = self.DOMAIN.evaluate(eta1, eta2, eta3, "g_31")
        G_32 = self.DOMAIN.evaluate(eta1, eta2, eta3, "g_32")
        G_33 = self.DOMAIN.evaluate(eta1, eta2, eta3, "g_33")
        B0_1 = self.EQUILIBRIUM.b2_eq_1(eta1, eta2, eta3)
        B0_2 = self.EQUILIBRIUM.b2_eq_2(eta1, eta2, eta3)
        B0_3 = self.EQUILIBRIUM.b2_eq_3(eta1, eta2, eta3)
        return det_g * (G_31 * B0_1 + G_32 * B0_2 + G_33 * B0_3)

    def __assemble_M1_cross(self, weight):
        """
        Assembles the 3D mass matrix with integrand
        Lambda_1 x Lambda_1 * weight.

        Parameters
        ----------
        self.SPACES : Tensor_spline_space
            tensor product B-spline space for finite element spaces

        weight : callable
            optional additional weight functions
        """

        p = self.SPACES.p  # spline degrees
        Nel = self.SPACES.Nel  # number of elements
        indN = self.SPACES.indN  # global indices of non-vanishing basis functions (N) in format (element, global index)
        indD = self.SPACES.indD  # global indices of non-vanishing basis functions (D) in format (element, global index)

        n_quad = self.SPACES.n_quad  # number of quadrature points per element
        pts = self.SPACES.pts  # global quadrature points
        wts = self.SPACES.wts  # global quadrature weights

        basisN = self.SPACES.basisN  # evaluated basis functions at quadrature points (N)
        basisD = self.SPACES.basisD  # evaluated basis functions at quadrature points (D)

        # indices and basis functions of components of a 1-form
        ind = [
            [indD[0], indN[1], indN[2]],  # DNN
            [indN[0], indD[1], indN[2]],  # NDN
            [indN[0], indN[1], indD[2]],
        ]  # NND

        basis = [
            [basisD[0], basisN[1], basisN[2]],  # DNN
            [basisN[0], basisD[1], basisN[2]],  # DNN
            [basisN[0], basisN[1], basisD[2]],
        ]  # NND

        ns = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # blocks of global mass matrix
        M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        for a in range(3):
            for b in range(3):
                Ni = self.SPACES.Nbase_1form[a]
                Nj = self.SPACES.Nbase_1form[b]

                M[a][b] = xp.zeros((Ni[0], Ni[1], Ni[2], 2 * p[0] + 1, 2 * p[1] + 1, 2 * p[2] + 1), dtype=float)

                # evaluate metric tensor at quadrature points
                if a == 1 and b == 2:
                    mat_w = weight[0](pts[0].flatten(), pts[1].flatten(), pts[2].flatten())
                elif a == 2 and b == 1:
                    mat_w = -weight[0](pts[0].flatten(), pts[1].flatten(), pts[2].flatten())

                elif a == 2 and b == 0:
                    mat_w = weight[1](pts[0].flatten(), pts[1].flatten(), pts[2].flatten())
                elif a == 0 and b == 2:
                    mat_w = -weight[1](pts[0].flatten(), pts[1].flatten(), pts[2].flatten())
                elif a == 0 and b == 1:
                    mat_w = weight[2](pts[0].flatten(), pts[1].flatten(), pts[2].flatten())
                elif a == 1 and b == 0:
                    mat_w = -weight[2](pts[0].flatten(), pts[1].flatten(), pts[2].flatten())

                if a != b:
                    mat_w = mat_w.reshape(Nel[0], n_quad[0], Nel[1], n_quad[1], Nel[2], n_quad[2])
                    ker.kernel_mass(
                        Nel[0],
                        Nel[1],
                        Nel[2],
                        p[0],
                        p[1],
                        p[2],
                        n_quad[0],
                        n_quad[1],
                        n_quad[2],
                        ns[a][0],
                        ns[a][1],
                        ns[a][2],
                        ns[b][0],
                        ns[b][1],
                        ns[b][2],
                        wts[0],
                        wts[1],
                        wts[2],
                        basis[a][0],
                        basis[a][1],
                        basis[a][2],
                        basis[b][0],
                        basis[b][1],
                        basis[b][2],
                        ind[a][0],
                        ind[a][1],
                        ind[a][2],
                        M[a][b],
                        mat_w,
                    )
                # convert to sparse matrix
                indices = xp.indices((Ni[0], Ni[1], Ni[2], 2 * p[0] + 1, 2 * p[1] + 1, 2 * p[2] + 1))

                shift = [xp.arange(Ni) - p for Ni, p in zip(Ni, p)]

                row = (Ni[1] * Ni[2] * indices[0] + Ni[2] * indices[1] + indices[2]).flatten()

                col1 = (indices[3] + shift[0][:, None, None, None, None, None]) % Nj[0]
                col2 = (indices[4] + shift[1][None, :, None, None, None, None]) % Nj[1]
                col3 = (indices[5] + shift[2][None, None, :, None, None, None]) % Nj[2]

                col = Nj[1] * Nj[2] * col1 + Nj[2] * col2 + col3

                M[a][b] = spa.csr_matrix(
                    (M[a][b].flatten(), (row, col.flatten())), shape=(Ni[0] * Ni[1] * Ni[2], Nj[0] * Nj[1] * Nj[2])
                )
                M[a][b].eliminate_zeros()

        M = spa.bmat(
            [[M[0][0], M[0][1], M[0][2]], [M[1][0], M[1][1], M[1][2]], [M[2][0], M[2][1], M[2][2]]], format="csr"
        )

        self.R1_mat = -self.SPACES.E1_0.dot(M.dot(self.SPACES.E1_0.T)).tocsr()

    # ================ Set Operator ==============================
    def set_Operators(self):
        self.R1 = spa.linalg.LinearOperator((self.dim_V1, self.dim_V1), matvec=lambda x: self.R1_mat.dot(x))

    # ================ Set Preconditioners =======================
    def set_Preconditioners(self, solver, drop_tol=1e-4, fill_fac=10.0):
        """
        TODO
        """

        assert solver == "FFT", "Until now, only the FFT preconditioner is implemented!"

        if solver == "FFT":
            self.M1_inv = mass_3d_pre.get_M1_PRE(self.SPACES, self.DOMAIN)
