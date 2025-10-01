# coding: utf-8
#
# Copyright 2020 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Class for control variates in delta-f method for current coupling scheme.
"""

import numpy as np
import scipy.sparse as spa

import struphy.feec.basics.kernels_3d as ker
import struphy.feec.control_variates.kernels_control_variate as ker_cv


class terms_control_variate:
    """
    Contains method for computing the terms (B x jh_eq) and -(rhoh_eq * (B x U)).

    Parameters
    ----------
    tensor_space_FEM : Tensor_spline_space
        3D tensor product B-spline space

    domain : domain
        domain object defining the geometry

    basis_u : int
        representation of MHD bulk velocity
    """

    def __init__(self, tensor_space_FEM, domain, basis_u):
        self.space = tensor_space_FEM  # 3D B-spline space
        self.basis_u = basis_u  # representation of MHD bulk velocity

        if self.basis_u == 0:
            kind_fun_eq = [1, 2, 3, 4]

        elif self.basis_u == 2:
            kind_fun_eq = [11, 12, 13, 14]

        # ========= evaluation of DF^(-1) * jh_eq_phys * |det(DF)| at quadrature points =========
        self.mat_jh1 = np.empty(
            (
                self.space.Nel[0],
                self.space.n_quad[0],
                self.space.Nel[1],
                self.space.n_quad[1],
                self.space.Nel[2],
                self.space.n_quad[2],
            ),
            dtype=float,
        )
        self.mat_jh2 = np.empty(
            (
                self.space.Nel[0],
                self.space.n_quad[0],
                self.space.Nel[1],
                self.space.n_quad[1],
                self.space.Nel[2],
                self.space.n_quad[2],
            ),
            dtype=float,
        )
        self.mat_jh3 = np.empty(
            (
                self.space.Nel[0],
                self.space.n_quad[0],
                self.space.Nel[1],
                self.space.n_quad[1],
                self.space.Nel[2],
                self.space.n_quad[2],
            ),
            dtype=float,
        )

        ker_cv.kernel_evaluation_quad(
            self.space.Nel,
            self.space.n_quad,
            self.space.pts[0],
            self.space.pts[1],
            self.space.pts[2],
            self.mat_jh1,
            kind_fun_eq[0],
            domain.kind_map,
            domain.params,
            domain.T[0],
            domain.T[1],
            domain.T[2],
            domain.p,
            domain.NbaseN,
            domain.cx,
            domain.cy,
            domain.cz,
        )
        ker_cv.kernel_evaluation_quad(
            self.space.Nel,
            self.space.n_quad,
            self.space.pts[0],
            self.space.pts[1],
            self.space.pts[2],
            self.mat_jh2,
            kind_fun_eq[1],
            domain.kind_map,
            domain.params,
            domain.T[0],
            domain.T[1],
            domain.T[2],
            domain.p,
            domain.NbaseN,
            domain.cx,
            domain.cy,
            domain.cz,
        )
        ker_cv.kernel_evaluation_quad(
            self.space.Nel,
            self.space.n_quad,
            self.space.pts[0],
            self.space.pts[1],
            self.space.pts[2],
            self.mat_jh3,
            kind_fun_eq[2],
            domain.kind_map,
            domain.params,
            domain.T[0],
            domain.T[1],
            domain.T[2],
            domain.p,
            domain.NbaseN,
            domain.cx,
            domain.cy,
            domain.cz,
        )

        # ========= evaluation of nh_eq_phys * |det(DF)| at quadrature points ===================
        self.mat_nh = np.empty(
            (
                self.space.Nel[0],
                self.space.n_quad[0],
                self.space.Nel[1],
                self.space.n_quad[1],
                self.space.Nel[2],
                self.space.n_quad[2],
            ),
            dtype=float,
        )

        ker_cv.kernel_evaluation_quad(
            self.space.Nel,
            self.space.n_quad,
            self.space.pts[0],
            self.space.pts[1],
            self.space.pts[2],
            self.mat_nh,
            kind_fun_eq[3],
            domain.kind_map,
            domain.params,
            domain.T[0],
            domain.T[1],
            domain.T[2],
            domain.p,
            domain.NbaseN,
            domain.cx,
            domain.cy,
            domain.cz,
        )

        # =========== 2-form magnetic field at quadrature points =================================
        self.B2_1 = np.empty(
            (
                self.space.Nel[0],
                self.space.n_quad[0],
                self.space.Nel[1],
                self.space.n_quad[1],
                self.space.Nel[2],
                self.space.n_quad[2],
            ),
            dtype=float,
        )
        self.B2_2 = np.empty(
            (
                self.space.Nel[0],
                self.space.n_quad[0],
                self.space.Nel[1],
                self.space.n_quad[1],
                self.space.Nel[2],
                self.space.n_quad[2],
            ),
            dtype=float,
        )
        self.B2_3 = np.empty(
            (
                self.space.Nel[0],
                self.space.n_quad[0],
                self.space.Nel[1],
                self.space.n_quad[1],
                self.space.Nel[2],
                self.space.n_quad[2],
            ),
            dtype=float,
        )

        # ================== correction matrices in step 1 ========================
        if self.basis_u == 0:
            self.M12 = np.empty(
                (
                    self.space.NbaseN[0],
                    self.space.NbaseN[1],
                    self.space.NbaseN[2],
                    2 * self.space.p[0] + 1,
                    2 * self.space.p[1] + 1,
                    2 * self.space.p[2] + 1,
                ),
                dtype=float,
            )
            self.M13 = np.empty(
                (
                    self.space.NbaseN[0],
                    self.space.NbaseN[1],
                    self.space.NbaseN[2],
                    2 * self.space.p[0] + 1,
                    2 * self.space.p[1] + 1,
                    2 * self.space.p[2] + 1,
                ),
                dtype=float,
            )
            self.M23 = np.empty(
                (
                    self.space.NbaseN[0],
                    self.space.NbaseN[1],
                    self.space.NbaseN[2],
                    2 * self.space.p[0] + 1,
                    2 * self.space.p[1] + 1,
                    2 * self.space.p[2] + 1,
                ),
                dtype=float,
            )

        elif self.basis_u == 2:
            self.M12 = np.empty(
                (
                    self.space.NbaseN[0],
                    self.space.NbaseD[1],
                    self.space.NbaseD[2],
                    2 * self.space.p[0] + 1,
                    2 * self.space.p[1] + 1,
                    2 * self.space.p[2] + 1,
                ),
                dtype=float,
            )
            self.M13 = np.empty(
                (
                    self.space.NbaseN[0],
                    self.space.NbaseD[1],
                    self.space.NbaseD[2],
                    2 * self.space.p[0] + 1,
                    2 * self.space.p[1] + 1,
                    2 * self.space.p[2] + 1,
                ),
                dtype=float,
            )
            self.M23 = np.empty(
                (
                    self.space.NbaseD[0],
                    self.space.NbaseN[1],
                    self.space.NbaseD[2],
                    2 * self.space.p[0] + 1,
                    2 * self.space.p[1] + 1,
                    2 * self.space.p[2] + 1,
                ),
                dtype=float,
            )

        # ==================== correction vectors in step 3 =======================
        if self.basis_u == 0:
            self.F1 = np.empty((self.space.NbaseN[0], self.space.NbaseN[1], self.space.NbaseN[2]), dtype=float)
            self.F2 = np.empty((self.space.NbaseN[0], self.space.NbaseN[1], self.space.NbaseN[2]), dtype=float)
            self.F3 = np.empty((self.space.NbaseN[0], self.space.NbaseN[1], self.space.NbaseN[2]), dtype=float)

        elif self.basis_u == 2:
            self.F1 = np.empty((self.space.NbaseN[0], self.space.NbaseD[1], self.space.NbaseD[2]), dtype=float)
            self.F2 = np.empty((self.space.NbaseD[0], self.space.NbaseN[1], self.space.NbaseD[2]), dtype=float)
            self.F3 = np.empty((self.space.NbaseD[0], self.space.NbaseD[1], self.space.NbaseN[2]), dtype=float)

    # ===== inner product in V0^3 resp. V2 of (B x jh_eq) - term ==========
    def inner_prod_jh_eq(self, b1, b2, b3):
        """
        Computes the inner product of the term

        (B x (DF^(-1) * jh_eq_phys) * |det(DF)|) (if MHD bulk velocity is a vector field)
        (B x (DF^(-1) * jh_eq_phys)            ) (if MHD bulk velocity is a 2-form)

        with each basis function in V0^3, respectively V2.

        Parameters
        ----------
        b1 : array_like
            the B-field FEM coefficients (1-component)

        b2 : array_like
            the B-field FEM coefficients (2-component)

        b3 : array_like
            the B-field FEM coefficients (3-component)

        Returns
        -------
        F : array_like
            inner products with each basis function in V0^3 resp. V2
        """

        # evaluation of magnetic field at quadrature points
        ker.kernel_evaluate_2form(
            self.space.Nel,
            self.space.p,
            [0, 1, 1],
            self.space.n_quad,
            b1,
            self.space.Nbase_2form[0],
            self.space.basisN[0],
            self.space.basisD[1],
            self.space.basisD[2],
            self.B2_1,
        )
        ker.kernel_evaluate_2form(
            self.space.Nel,
            self.space.p,
            [1, 0, 1],
            self.space.n_quad,
            b2,
            self.space.Nbase_2form[1],
            self.space.basisD[0],
            self.space.basisN[1],
            self.space.basisD[2],
            self.B2_2,
        )
        ker.kernel_evaluate_2form(
            self.space.Nel,
            self.space.p,
            [1, 1, 0],
            self.space.n_quad,
            b3,
            self.space.Nbase_2form[2],
            self.space.basisD[0],
            self.space.basisD[1],
            self.space.basisN[2],
            self.B2_3,
        )

        if self.basis_u == 0:
            # assembly of F (1-component)
            ker.kernel_inner(
                self.space.Nel[0],
                self.space.Nel[1],
                self.space.Nel[2],
                self.space.p[0],
                self.space.p[1],
                self.space.p[2],
                self.space.n_quad[0],
                self.space.n_quad[1],
                self.space.n_quad[2],
                0,
                0,
                0,
                self.space.wts[0],
                self.space.wts[1],
                self.space.wts[2],
                self.space.basisN[0],
                self.space.basisN[1],
                self.space.basisN[2],
                self.space.NbaseN[0],
                self.space.NbaseN[1],
                self.space.NbaseN[2],
                self.F1,
                self.B2_2 * self.mat_jh3 - self.B2_3 * self.mat_jh2,
            )

            # assembly of F (2-component)
            ker.kernel_inner(
                self.space.Nel[0],
                self.space.Nel[1],
                self.space.Nel[2],
                self.space.p[0],
                self.space.p[1],
                self.space.p[2],
                self.space.n_quad[0],
                self.space.n_quad[1],
                self.space.n_quad[2],
                0,
                0,
                0,
                self.space.wts[0],
                self.space.wts[1],
                self.space.wts[2],
                self.space.basisN[0],
                self.space.basisN[1],
                self.space.basisN[2],
                self.space.NbaseN[0],
                self.space.NbaseN[1],
                self.space.NbaseN[2],
                self.F2,
                self.B2_3 * self.mat_jh1 - self.B2_1 * self.mat_jh3,
            )

            # assembly of F (3-component)
            ker.kernel_inner(
                self.space.Nel[0],
                self.space.Nel[1],
                self.space.Nel[2],
                self.space.p[0],
                self.space.p[1],
                self.space.p[2],
                self.space.n_quad[0],
                self.space.n_quad[1],
                self.space.n_quad[2],
                0,
                0,
                0,
                self.space.wts[0],
                self.space.wts[1],
                self.space.wts[2],
                self.space.basisN[0],
                self.space.basisN[1],
                self.space.basisN[2],
                self.space.NbaseN[0],
                self.space.NbaseN[1],
                self.space.NbaseN[2],
                self.F3,
                self.B2_1 * self.mat_jh2 - self.B2_2 * self.mat_jh1,
            )

        elif self.basis_u == 2:
            # assembly of F (1-component)
            ker.kernel_inner(
                self.space.Nel[0],
                self.space.Nel[1],
                self.space.Nel[2],
                self.space.p[0],
                self.space.p[1],
                self.space.p[2],
                self.space.n_quad[0],
                self.space.n_quad[1],
                self.space.n_quad[2],
                0,
                1,
                1,
                self.space.wts[0],
                self.space.wts[1],
                self.space.wts[2],
                self.space.basisN[0],
                self.space.basisD[1],
                self.space.basisD[2],
                self.space.NbaseN[0],
                self.space.NbaseD[1],
                self.space.NbaseD[2],
                self.F1,
                self.B2_2 * self.mat_jh3 - self.B2_3 * self.mat_jh2,
            )

            # assembly of F (2-component)
            ker.kernel_inner(
                self.space.Nel[0],
                self.space.Nel[1],
                self.space.Nel[2],
                self.space.p[0],
                self.space.p[1],
                self.space.p[2],
                self.space.n_quad[0],
                self.space.n_quad[1],
                self.space.n_quad[2],
                1,
                0,
                1,
                self.space.wts[0],
                self.space.wts[1],
                self.space.wts[2],
                self.space.basisD[0],
                self.space.basisN[1],
                self.space.basisD[2],
                self.space.NbaseD[0],
                self.space.NbaseN[1],
                self.space.NbaseD[2],
                self.F2,
                self.B2_3 * self.mat_jh1 - self.B2_1 * self.mat_jh3,
            )

            # assembly of F (3-component)
            ker.kernel_inner(
                self.space.Nel[0],
                self.space.Nel[1],
                self.space.Nel[2],
                self.space.p[0],
                self.space.p[1],
                self.space.p[2],
                self.space.n_quad[0],
                self.space.n_quad[1],
                self.space.n_quad[2],
                1,
                1,
                0,
                self.space.wts[0],
                self.space.wts[1],
                self.space.wts[2],
                self.space.basisD[0],
                self.space.basisD[1],
                self.space.basisN[2],
                self.space.NbaseD[0],
                self.space.NbaseD[1],
                self.space.NbaseN[2],
                self.F3,
                self.B2_1 * self.mat_jh2 - self.B2_2 * self.mat_jh1,
            )

        return np.concatenate((self.F1.flatten(), self.F2.flatten(), self.F3.flatten()))

    # ===== mass matrix in V0^3 resp. V2 of -(rhoh_eq * (B x U)) - term =======
    def mass_nh_eq(self, b1, b2, b3):
        """
        Computes the mass matrix in V0^3 respectively V2 weighted with the term

        -(rhoh_eq_phys * |det(DF)| B x) (if MHD bulk velocity is a vector field)
        -(rhoh_eq_phys / |det(DF)| B x) (if MHD bulk velocity is a 2-form)


        Parameters
        ----------
        b1 : array_like
            the B-field FEM coefficients (1-component)

        b2 : array_like
            the B-field FEM coefficients (2-component)

        b3 : array_like
            the B-field FEM coefficients (3-component)

        Returns
        -------
        M12 : 6D array
            12 block of  weighted mass matrix

        M13 : 6D array
            13 block of  weighted mass matrix

        M23 : 6D array
            23 block of  weighted mass matrix
        """

        # evaluation of magnetic field at quadrature points
        ker.kernel_evaluate_2form(
            self.space.Nel,
            self.space.p,
            [0, 1, 1],
            self.space.n_quad,
            b1,
            self.space.Nbase_2form[0],
            self.space.basisN[0],
            self.space.basisD[1],
            self.space.basisD[2],
            self.B2_1,
        )
        ker.kernel_evaluate_2form(
            self.space.Nel,
            self.space.p,
            [1, 0, 1],
            self.space.n_quad,
            b2,
            self.space.Nbase_2form[1],
            self.space.basisD[0],
            self.space.basisN[1],
            self.space.basisD[2],
            self.B2_2,
        )
        ker.kernel_evaluate_2form(
            self.space.Nel,
            self.space.p,
            [1, 1, 0],
            self.space.n_quad,
            b3,
            self.space.Nbase_2form[2],
            self.space.basisD[0],
            self.space.basisD[1],
            self.space.basisN[2],
            self.B2_3,
        )

        if self.basis_u == 0:
            # assembly of M12
            ker.kernel_mass(
                self.space.Nel[0],
                self.space.Nel[1],
                self.space.Nel[2],
                self.space.p[0],
                self.space.p[1],
                self.space.p[2],
                self.space.n_quad[0],
                self.space.n_quad[1],
                self.space.n_quad[2],
                0,
                0,
                0,
                0,
                0,
                0,
                self.space.wts[0],
                self.space.wts[1],
                self.space.wts[2],
                self.space.basisN[0],
                self.space.basisN[1],
                self.space.basisN[2],
                self.space.basisN[0],
                self.space.basisN[1],
                self.space.basisN[2],
                self.space.indN[0],
                self.space.indN[1],
                self.space.indN[2],
                self.M12,
                +self.mat_nh * self.B2_3,
            )

            # assembly of M13
            ker.kernel_mass(
                self.space.Nel[0],
                self.space.Nel[1],
                self.space.Nel[2],
                self.space.p[0],
                self.space.p[1],
                self.space.p[2],
                self.space.n_quad[0],
                self.space.n_quad[1],
                self.space.n_quad[2],
                0,
                0,
                0,
                0,
                0,
                0,
                self.space.wts[0],
                self.space.wts[1],
                self.space.wts[2],
                self.space.basisN[0],
                self.space.basisN[1],
                self.space.basisN[2],
                self.space.basisN[0],
                self.space.basisN[1],
                self.space.basisN[2],
                self.space.indN[0],
                self.space.indN[1],
                self.space.indN[2],
                self.M13,
                -self.mat_nh * self.B2_2,
            )

            # assembly of M23
            ker.kernel_mass(
                self.space.Nel[0],
                self.space.Nel[1],
                self.space.Nel[2],
                self.space.p[0],
                self.space.p[1],
                self.space.p[2],
                self.space.n_quad[0],
                self.space.n_quad[1],
                self.space.n_quad[2],
                0,
                0,
                0,
                0,
                0,
                0,
                self.space.wts[0],
                self.space.wts[1],
                self.space.wts[2],
                self.space.basisN[0],
                self.space.basisN[1],
                self.space.basisN[2],
                self.space.basisN[0],
                self.space.basisN[1],
                self.space.basisN[2],
                self.space.indN[0],
                self.space.indN[1],
                self.space.indN[2],
                self.M23,
                +self.mat_nh * self.B2_1,
            )

        elif self.basis_u == 2:
            # assembly of M12
            ker.kernel_mass(
                self.space.Nel[0],
                self.space.Nel[1],
                self.space.Nel[2],
                self.space.p[0],
                self.space.p[1],
                self.space.p[2],
                self.space.n_quad[0],
                self.space.n_quad[1],
                self.space.n_quad[2],
                0,
                1,
                1,
                1,
                0,
                1,
                self.space.wts[0],
                self.space.wts[1],
                self.space.wts[2],
                self.space.basisN[0],
                self.space.basisD[1],
                self.space.basisD[2],
                self.space.basisD[0],
                self.space.basisN[1],
                self.space.basisD[2],
                self.space.indN[0],
                self.space.indD[1],
                self.space.indD[2],
                self.M12,
                +self.mat_nh * self.B2_3,
            )

            # assembly of M13
            ker.kernel_mass(
                self.space.Nel[0],
                self.space.Nel[1],
                self.space.Nel[2],
                self.space.p[0],
                self.space.p[1],
                self.space.p[2],
                self.space.n_quad[0],
                self.space.n_quad[1],
                self.space.n_quad[2],
                0,
                1,
                1,
                1,
                1,
                0,
                self.space.wts[0],
                self.space.wts[1],
                self.space.wts[2],
                self.space.basisN[0],
                self.space.basisD[1],
                self.space.basisD[2],
                self.space.basisD[0],
                self.space.basisD[1],
                self.space.basisN[2],
                self.space.indN[0],
                self.space.indD[1],
                self.space.indD[2],
                self.M13,
                -self.mat_nh * self.B2_2,
            )

            # assembly of M23
            ker.kernel_mass(
                self.space.Nel[0],
                self.space.Nel[1],
                self.space.Nel[2],
                self.space.p[0],
                self.space.p[1],
                self.space.p[2],
                self.space.n_quad[0],
                self.space.n_quad[1],
                self.space.n_quad[2],
                1,
                0,
                1,
                1,
                1,
                0,
                self.space.wts[0],
                self.space.wts[1],
                self.space.wts[2],
                self.space.basisD[0],
                self.space.basisN[1],
                self.space.basisD[2],
                self.space.basisD[0],
                self.space.basisD[1],
                self.space.basisN[2],
                self.space.indD[0],
                self.space.indN[1],
                self.space.indD[2],
                self.M23,
                +self.mat_nh * self.B2_1,
            )

        # conversion to sparse matrix and return
        return self.M12, self.M13, self.M23
