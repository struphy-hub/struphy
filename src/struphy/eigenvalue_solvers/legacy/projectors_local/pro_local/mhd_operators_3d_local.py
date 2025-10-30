# coding: utf-8
#
# Copyright 2020 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Class for local projections for linear ideal mhd in 3d based on quasi-interpolation
"""

import sys

import cunumpy as xp
import scipy.sparse as spa
import source_run.kernels_projectors_evaluation as ker_eva

import struphy.feec.basics.kernels_3d as ker_loc_3d
import struphy.feec.bsplines as bsp
import struphy.feec.projectors.pro_local.kernels_projectors_local_mhd as ker_loc


class projectors_local_mhd:
    """
    Local commuting projections of various terms in linear ideal MHD.

    Parameters
    ----------
    tensor_space : Tensor_spline_space
        a 3d tensor product space of B-splines

    n_quad : list of ints
        number of quadrature points per integration interval for histopolations
    """

    def __init__(self, tensor_space, n_quad):
        self.tensor_space = tensor_space

        self.T = tensor_space.T  # knot vector
        self.p = tensor_space.p  # spline degree
        self.bc = tensor_space.bc  # boundary conditions

        self.Nel = tensor_space.Nel  # number of elements
        self.NbaseN = tensor_space.NbaseN  # number of basis functions (N)
        self.NbaseD = tensor_space.NbaseD  # number of basis functions (D)

        self.n_quad = n_quad  # number of quadrature point per integration interval

        # Gauss - Legendre quadrature points and weights in (-1, 1)
        self.pts_loc = [xp.polynomial.legendre.leggauss(n_quad)[0] for n_quad in self.n_quad]
        self.wts_loc = [xp.polynomial.legendre.leggauss(n_quad)[1] for n_quad in self.n_quad]

        # set interpolation and histopolation coefficients
        self.coeff_i = [0, 0, 0]
        self.coeff_h = [0, 0, 0]

        for a in range(3):
            if self.bc[a] == True:
                self.coeff_i[a] = xp.zeros((1, 2 * self.p[a] - 1), dtype=float)
                self.coeff_h[a] = xp.zeros((1, 2 * self.p[a]), dtype=float)

                if self.p[a] == 1:
                    self.coeff_i[a][0, :] = xp.array([1.0])
                    self.coeff_h[a][0, :] = xp.array([1.0, 1.0])

                elif self.p[a] == 2:
                    self.coeff_i[a][0, :] = 1 / 2 * xp.array([-1.0, 4.0, -1.0])
                    self.coeff_h[a][0, :] = 1 / 2 * xp.array([-1.0, 3.0, 3.0, -1.0])

                elif self.p[a] == 3:
                    self.coeff_i[a][0, :] = 1 / 6 * xp.array([1.0, -8.0, 20.0, -8.0, 1.0])
                    self.coeff_h[a][0, :] = 1 / 6 * xp.array([1.0, -7.0, 12.0, 12.0, -7.0, 1.0])

                elif self.p[a] == 4:
                    self.coeff_i[a][0, :] = 2 / 45 * xp.array([-1.0, 16.0, -295 / 4, 140.0, -295 / 4, 16.0, -1.0])
                    self.coeff_h[a][0, :] = (
                        2 / 45 * xp.array([-1.0, 15.0, -231 / 4, 265 / 4, 265 / 4, -231 / 4, 15.0, -1.0])
                    )

                else:
                    print("degree > 4 not implemented!")

            else:
                self.coeff_i[a] = xp.zeros((2 * self.p[a] - 1, 2 * self.p[a] - 1), dtype=float)
                self.coeff_h[a] = xp.zeros((2 * self.p[a] - 1, 2 * self.p[a]), dtype=float)

                if self.p[a] == 1:
                    self.coeff_i[a][0, :] = xp.array([1.0])
                    self.coeff_h[a][0, :] = xp.array([1.0, 1.0])

                elif self.p[a] == 2:
                    self.coeff_i[a][0, :] = 1 / 2 * xp.array([2.0, 0.0, 0.0])
                    self.coeff_i[a][1, :] = 1 / 2 * xp.array([-1.0, 4.0, -1.0])
                    self.coeff_i[a][2, :] = 1 / 2 * xp.array([0.0, 0.0, 2.0])

                    self.coeff_h[a][0, :] = 1 / 2 * xp.array([3.0, -1.0, 0.0, 0.0])
                    self.coeff_h[a][1, :] = 1 / 2 * xp.array([-1.0, 3.0, 3.0, -1.0])
                    self.coeff_h[a][2, :] = 1 / 2 * xp.array([0.0, 0.0, -1.0, 3.0])

                elif self.p[a] == 3:
                    self.coeff_i[a][0, :] = 1 / 18 * xp.array([18.0, 0.0, 0.0, 0.0, 0.0])
                    self.coeff_i[a][1, :] = 1 / 18 * xp.array([-5.0, 40.0, -24.0, 8.0, -1.0])
                    self.coeff_i[a][2, :] = 1 / 18 * xp.array([3.0, -24.0, 60.0, -24.0, 3.0])
                    self.coeff_i[a][3, :] = 1 / 18 * xp.array([-1.0, 8.0, -24.0, 40.0, -5.0])
                    self.coeff_i[a][4, :] = 1 / 18 * xp.array([0.0, 0.0, 0.0, 0.0, 18.0])

                    self.coeff_h[a][0, :] = 1 / 18 * xp.array([23.0, -17.0, 7.0, -1.0, 0.0, 0.0])
                    self.coeff_h[a][1, :] = 1 / 18 * xp.array([-8.0, 56.0, -28.0, 4.0, 0.0, 0.0])
                    self.coeff_h[a][2, :] = 1 / 18 * xp.array([3.0, -21.0, 36.0, 36.0, -21.0, 3.0])
                    self.coeff_h[a][3, :] = 1 / 18 * xp.array([0.0, 0.0, 4.0, -28.0, 56.0, -8.0])
                    self.coeff_h[a][4, :] = 1 / 18 * xp.array([0.0, 0.0, -1.0, 7.0, -17.0, 23.0])

                elif self.p[a] == 4:
                    self.coeff_i[a][0, :] = 1 / 360 * xp.array([360.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    self.coeff_i[a][1, :] = 1 / 360 * xp.array([-59.0, 944.0, -1000.0, 720.0, -305.0, 64.0, -4.0])
                    self.coeff_i[a][2, :] = 1 / 360 * xp.array([23.0, -368.0, 1580.0, -1360.0, 605.0, -128.0, 8.0])
                    self.coeff_i[a][3, :] = 1 / 360 * xp.array([-16.0, 256.0, -1180.0, 2240.0, -1180.0, 256.0, -16.0])
                    self.coeff_i[a][4, :] = 1 / 360 * xp.array([8.0, -128.0, 605.0, -1360.0, 1580.0, -368.0, 23.0])
                    self.coeff_i[a][5, :] = 1 / 360 * xp.array([-4.0, 64.0, -305.0, 720.0, -1000.0, 944.0, -59.0])
                    self.coeff_i[a][6, :] = 1 / 360 * xp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 360.0])

                    self.coeff_h[a][0, :] = 1 / 360 * xp.array([419.0, -525.0, 475.0, -245.0, 60.0, -4.0, 0.0, 0.0])
                    self.coeff_h[a][1, :] = 1 / 360 * xp.array([-82.0, 1230.0, -1350.0, 730.0, -180.0, 12.0, 0.0, 0.0])
                    self.coeff_h[a][2, :] = 1 / 360 * xp.array([39.0, -585.0, 2175.0, -1425.0, 360.0, -24.0, 0.0, 0.0])
                    self.coeff_h[a][3, :] = (
                        1 / 360 * xp.array([-16.0, 240.0, -924.0, 1060.0, 1060.0, -924.0, 240.0, -16.0])
                    )
                    self.coeff_h[a][4, :] = 1 / 360 * xp.array([0.0, 0.0, -24.0, 360.0, -1425.0, 2175.0, -585.0, 39.0])
                    self.coeff_h[a][5, :] = 1 / 360 * xp.array([0.0, 0.0, 12.0, -180.0, 730.0, -1350.0, 1230.0, -82.0])
                    self.coeff_h[a][6, :] = 1 / 360 * xp.array([0.0, 0.0, -4.0, 60.0, -245.0, 475.0, -525.0, 419.0])

                else:
                    print("degree > 4 not implemented!")

        # set interpolation points
        n_lambda_int = [NbaseN for NbaseN in self.NbaseN]  # number of coefficients in space V0
        self.n_int = [2 * p - 1 for p in self.p]  # number of interpolation points (1, 3, 5, 7, ...)

        self.n_int_locbf_N = [0, 0, 0]
        self.n_int_locbf_D = [0, 0, 0]

        for a in range(3):
            if self.p[a] == 1:
                self.n_int_locbf_N[a] = 2  # number of non-vanishing N bf in interpolation interval (2, 3, 5, 7)
                self.n_int_locbf_D[a] = 1  # number of non-vanishing D bf in interpolation interval (1, 2, 4, 6)

            else:
                self.n_int_locbf_N[a] = (
                    2 * self.p[a] - 1
                )  # number of non-vanishing N bf in interpolation interval (2, 3, 5, 7)
                self.n_int_locbf_D[a] = (
                    2 * self.p[a] - 2
                )  # number of non-vanishing D bf in interpolation interval (1, 2, 4, 6)

        self.x_int = [
            xp.zeros((n_lambda_int, n_int), dtype=float) for n_lambda_int, n_int in zip(n_lambda_int, self.n_int)
        ]

        self.int_global_N = [
            xp.zeros((n_lambda_int, n_int_locbf_N), dtype=int)
            for n_lambda_int, n_int_locbf_N in zip(n_lambda_int, self.n_int_locbf_N)
        ]
        self.int_global_D = [
            xp.zeros((n_lambda_int, n_int_locbf_D), dtype=int)
            for n_lambda_int, n_int_locbf_D in zip(n_lambda_int, self.n_int_locbf_D)
        ]

        self.int_loccof_N = [
            xp.zeros((n_lambda_int, n_int_locbf_N), dtype=int)
            for n_lambda_int, n_int_locbf_N in zip(n_lambda_int, self.n_int_locbf_N)
        ]
        self.int_loccof_D = [
            xp.zeros((n_lambda_int, n_int_locbf_D), dtype=int)
            for n_lambda_int, n_int_locbf_D in zip(n_lambda_int, self.n_int_locbf_D)
        ]

        self.x_int_indices = [
            xp.zeros((n_lambda_int, n_int), dtype=int) for n_lambda_int, n_int in zip(n_lambda_int, self.n_int)
        ]
        self.coeffi_indices = [xp.zeros(n_lambda_int, dtype=int) for n_lambda_int in n_lambda_int]

        self.n_int_nvcof_D = [None, None, None]
        self.n_int_nvcof_N = [None, None, None]

        self.int_add_D = [None, None, None]
        self.int_add_N = [None, None, None]

        self.int_shift_D = [0, 0, 0]
        self.int_shift_N = [0, 0, 0]

        for a in range(3):
            if not self.bc[a]:
                # maximum number of non-vanishing coefficients
                if self.p[a] == 1:
                    self.n_int_nvcof_D[a] = 2
                    self.n_int_nvcof_N[a] = 2

                else:
                    self.n_int_nvcof_D[a] = 3 * self.p[a] - 3
                    self.n_int_nvcof_N[a] = 3 * self.p[a] - 2

                # shift in local coefficient indices at right boundary (only for non-periodic boundary conditions)
                self.int_add_D[a] = xp.arange(self.n_int[a] - 2) + 1
                self.int_add_N[a] = xp.arange(self.n_int[a] - 1) + 1

                counter_D = 0
                counter_N = 0

                # shift local coefficients --> global coefficients (D)
                if self.p[a] == 1:
                    self.int_shift_D[a] = xp.arange(self.NbaseD[a])
                else:
                    self.int_shift_D[a] = xp.arange(self.NbaseD[a]) - (self.p[a] - 2)
                    self.int_shift_D[a][: 2 * self.p[a] - 2] = 0
                    self.int_shift_D[a][-(2 * self.p[a] - 2) :] = self.int_shift_D[a][-(2 * self.p[a] - 2)]

                # shift local coefficients --> global coefficients (N)
                if self.p[a] == 1:
                    self.int_shift_N[a] = xp.arange(self.NbaseN[a])
                    self.int_shift_N[a][-1] = self.int_shift_N[a][-2]

                else:
                    self.int_shift_N[a] = xp.arange(self.NbaseN[a]) - (self.p[a] - 1)
                    self.int_shift_N[a][: 2 * self.p[a] - 1] = 0
                    self.int_shift_N[a][-(2 * self.p[a] - 1) :] = self.int_shift_N[a][-(2 * self.p[a] - 1)]

                counter_coeffi = xp.copy(self.p[a])

                for i in range(n_lambda_int[a]):
                    # left boundary region
                    if i < self.p[a] - 1:
                        self.int_global_N[a][i] = xp.arange(self.n_int_locbf_N[a])
                        self.int_global_D[a][i] = xp.arange(self.n_int_locbf_D[a])

                        self.x_int_indices[a][i] = xp.arange(self.n_int[a])
                        self.coeffi_indices[a][i] = i
                        for j in range(2 * (self.p[a] - 1) + 1):
                            xi = self.p[a] - 1
                            self.x_int[a][i, j] = (
                                self.T[a][xi + 1 + int(j / 2)] + self.T[a][xi + 1 + int((j + 1) / 2)]
                            ) / 2

                    # right boundary region
                    elif i > n_lambda_int[a] - self.p[a]:
                        self.int_global_N[a][i] = (
                            xp.arange(self.n_int_locbf_N[a]) + n_lambda_int[a] - self.p[a] - (self.p[a] - 1)
                        )
                        self.int_global_D[a][i] = (
                            xp.arange(self.n_int_locbf_D[a]) + n_lambda_int[a] - self.p[a] - (self.p[a] - 1)
                        )

                        self.x_int_indices[a][i] = xp.arange(self.n_int[a]) + 2 * (
                            n_lambda_int[a] - self.p[a] - (self.p[a] - 1)
                        )
                        self.coeffi_indices[a][i] = counter_coeffi
                        counter_coeffi += 1
                        for j in range(2 * (self.p[a] - 1) + 1):
                            xi = n_lambda_int[a] - self.p[a]
                            self.x_int[a][i, j] = (
                                self.T[a][xi + 1 + int(j / 2)] + self.T[a][xi + 1 + int((j + 1) / 2)]
                            ) / 2

                    # interior
                    else:
                        if self.p[a] == 1:
                            self.int_global_N[a][i] = xp.arange(self.n_int_locbf_N[a]) + i
                            self.int_global_D[a][i] = xp.arange(self.n_int_locbf_D[a]) + i

                            self.int_global_N[a][-1] = self.int_global_N[a][-2]
                            self.int_global_D[a][-1] = self.int_global_D[a][-2]

                        else:
                            self.int_global_N[a][i] = xp.arange(self.n_int_locbf_N[a]) + i - (self.p[a] - 1)
                            self.int_global_D[a][i] = xp.arange(self.n_int_locbf_D[a]) + i - (self.p[a] - 1)

                        if self.p[a] == 1:
                            self.x_int_indices[a][i] = i
                        else:
                            self.x_int_indices[a][i] = xp.arange(self.n_int[a]) + 2 * (i - (self.p[a] - 1))

                        self.coeffi_indices[a][i] = self.p[a] - 1

                        for j in range(2 * (self.p[a] - 1) + 1):
                            self.x_int[a][i, j] = (
                                self.T[a][i + 1 + int(j / 2)] + self.T[a][i + 1 + int((j + 1) / 2)]
                            ) / 2

                    # local coefficient index
                    if self.p[a] == 1:
                        self.int_loccof_N[a][i] = xp.array([0, 1])
                        self.int_loccof_D[a][-1] = xp.array([1])

                    else:
                        if i > 0:
                            for il in range(self.n_int_locbf_D[a]):
                                k_glob_new = self.int_global_D[a][i, il]
                                bol = k_glob_new == self.int_global_D[a][i - 1]

                                if xp.any(bol):
                                    self.int_loccof_D[a][i, il] = self.int_loccof_D[a][i - 1, xp.where(bol)[0][0]] + 1

                                if (k_glob_new >= n_lambda_int[a] - self.p[a] - (self.p[a] - 2)) and (
                                    self.int_loccof_D[a][i, il] == 0
                                ):
                                    self.int_loccof_D[a][i, il] = self.int_add_D[a][counter_D]
                                    counter_D += 1

                            for il in range(self.n_int_locbf_N[a]):
                                k_glob_new = self.int_global_N[a][i, il]
                                bol = k_glob_new == self.int_global_N[a][i - 1]

                                if xp.any(bol):
                                    self.int_loccof_N[a][i, il] = self.int_loccof_N[a][i - 1, xp.where(bol)[0][0]] + 1

                                if (k_glob_new >= n_lambda_int[a] - self.p[a] - (self.p[a] - 2)) and (
                                    self.int_loccof_N[a][i, il] == 0
                                ):
                                    self.int_loccof_N[a][i, il] = self.int_add_N[a][counter_N]
                                    counter_N += 1

            else:
                # maximum number of non-vanishing coefficients
                if self.p[a] == 1:
                    self.n_int_nvcof_D[a] = 2 * self.p[a] - 1
                    self.n_int_nvcof_N[a] = 2 * self.p[a]

                else:
                    self.n_int_nvcof_D[a] = 2 * self.p[a] - 2
                    self.n_int_nvcof_N[a] = 2 * self.p[a] - 1

                # shift local coefficients --> global coefficients
                if self.p[a] == 1:
                    self.int_shift_D[a] = xp.arange(self.NbaseN[a]) - (self.p[a] - 1)
                    self.int_shift_N[a] = xp.arange(self.NbaseN[a]) - (self.p[a])
                else:
                    self.int_shift_D[a] = xp.arange(self.NbaseN[a]) - (self.p[a] - 2)
                    self.int_shift_N[a] = xp.arange(self.NbaseN[a]) - (self.p[a] - 1)

                for i in range(n_lambda_int[a]):
                    # global indices of non-vanishing basis functions and position of coefficients in final matrix
                    self.int_global_N[a][i] = (xp.arange(self.n_int_locbf_N[a]) + i - (self.p[a] - 1)) % self.NbaseN[a]
                    self.int_global_D[a][i] = (xp.arange(self.n_int_locbf_D[a]) + i - (self.p[a] - 1)) % self.NbaseD[a]

                    self.int_loccof_N[a][i] = xp.arange(self.n_int_locbf_N[a] - 1, -1, -1)
                    self.int_loccof_D[a][i] = xp.arange(self.n_int_locbf_D[a] - 1, -1, -1)

                    if self.p[a] == 1:
                        self.x_int_indices[a][i] = i
                    else:
                        self.x_int_indices[a][i] = (xp.arange(self.n_int[a]) + 2 * (i - (self.p[a] - 1))) % (
                            2 * self.Nel[a]
                        )

                    self.coeffi_indices[a][i] = 0

                    for j in range(2 * (self.p[a] - 1) + 1):
                        self.x_int[a][i, j] = (
                            (self.T[a][i + 1 + int(j / 2)] + self.T[a][i + 1 + int((j + 1) / 2)]) / 2
                        ) % 1.0

            # identify unique interpolation points to save memory
            self.x_int[a] = xp.unique(self.x_int[a].flatten())

        # set histopolation points, quadrature points and weights
        n_lambda_his = [xp.copy(NbaseD) for NbaseD in self.NbaseD]  # number of coefficients in space V1

        self.n_his = [2 * p for p in self.p]  # number of histopolation intervals
        self.n_his_locbf_N = [2 * p for p in self.p]  # number of non-vanishing N bf in histopolation interval
        self.n_his_locbf_D = [2 * p - 1 for p in self.p]  # number of non-vanishing D bf in histopolation interval

        self.x_his = [
            xp.zeros((n_lambda_his, n_his + 1), dtype=float) for n_lambda_his, n_his in zip(n_lambda_his, self.n_his)
        ]

        self.his_global_N = [
            xp.zeros((n_lambda_his, n_his_locbf_N), dtype=int)
            for n_lambda_his, n_his_locbf_N in zip(n_lambda_his, self.n_his_locbf_N)
        ]
        self.his_global_D = [
            xp.zeros((n_lambda_his, n_his_locbf_D), dtype=int)
            for n_lambda_his, n_his_locbf_D in zip(n_lambda_his, self.n_his_locbf_D)
        ]

        self.his_loccof_N = [
            xp.zeros((n_lambda_his, n_his_locbf_N), dtype=int)
            for n_lambda_his, n_his_locbf_N in zip(n_lambda_his, self.n_his_locbf_N)
        ]
        self.his_loccof_D = [
            xp.zeros((n_lambda_his, n_his_locbf_D), dtype=int)
            for n_lambda_his, n_his_locbf_D in zip(n_lambda_his, self.n_his_locbf_D)
        ]

        self.x_his_indices = [
            xp.zeros((n_lambda_his, n_his), dtype=int) for n_lambda_his, n_his in zip(n_lambda_his, self.n_his)
        ]
        self.coeffh_indices = [xp.zeros(n_lambda_his, dtype=int) for n_lambda_his in n_lambda_his]

        self.pts = [0, 0, 0]
        self.wts = [0, 0, 0]

        self.n_his_nvcof_D = [None, None, None]
        self.n_his_nvcof_N = [None, None, None]

        self.his_add_D = [None, None, None]
        self.his_add_N = [None, None, None]

        self.his_shift_D = [0, 0, 0]
        self.his_shift_N = [0, 0, 0]

        for a in range(3):
            if not self.bc[a]:
                # maximum number of non-vanishing coefficients
                self.n_his_nvcof_D[a] = 3 * self.p[a] - 2
                self.n_his_nvcof_N[a] = 3 * self.p[a] - 1

                # shift in local coefficient indices at right boundary (only for non-periodic boundary conditions)
                self.his_add_D[a] = xp.arange(self.n_his[a] - 2) + 1
                self.his_add_N[a] = xp.arange(self.n_his[a] - 1) + 1

                counter_D = 0
                counter_N = 0

                # shift local coefficients --> global coefficients (D)
                self.his_shift_D[a] = xp.arange(self.NbaseD[a]) - (self.p[a] - 1)
                self.his_shift_D[a][: 2 * self.p[a] - 1] = 0
                self.his_shift_D[a][-(2 * self.p[a] - 1) :] = self.his_shift_D[a][-(2 * self.p[a] - 1)]

                # shift local coefficients --> global coefficients (N)
                self.his_shift_N[a] = xp.arange(self.NbaseN[a]) - self.p[a]
                self.his_shift_N[a][: 2 * self.p[a]] = 0
                self.his_shift_N[a][-2 * self.p[a] :] = self.his_shift_N[a][-2 * self.p[a]]

                counter_coeffh = xp.copy(self.p[a])

                for i in range(n_lambda_his[a]):
                    # left boundary region
                    if i < self.p[a] - 1:
                        self.his_global_N[a][i] = xp.arange(self.n_his_locbf_N[a])
                        self.his_global_D[a][i] = xp.arange(self.n_his_locbf_D[a])

                        self.x_his_indices[a][i] = xp.arange(self.n_his[a])
                        self.coeffh_indices[a][i] = i
                        for j in range(2 * self.p[a] + 1):
                            xi = self.p[a] - 1
                            self.x_his[a][i, j] = (
                                self.T[a][xi + 1 + int(j / 2)] + self.T[a][xi + 1 + int((j + 1) / 2)]
                            ) / 2

                    # right boundary region
                    elif i > n_lambda_his[a] - self.p[a]:
                        self.his_global_N[a][i] = (
                            xp.arange(self.n_his_locbf_N[a]) + n_lambda_his[a] - self.p[a] - (self.p[a] - 1)
                        )
                        self.his_global_D[a][i] = (
                            xp.arange(self.n_his_locbf_D[a]) + n_lambda_his[a] - self.p[a] - (self.p[a] - 1)
                        )

                        self.x_his_indices[a][i] = xp.arange(self.n_his[a]) + 2 * (
                            n_lambda_his[a] - self.p[a] - (self.p[a] - 1)
                        )
                        self.coeffh_indices[a][i] = counter_coeffh
                        counter_coeffh += 1
                        for j in range(2 * self.p[a] + 1):
                            xi = n_lambda_his[a] - self.p[a]
                            self.x_his[a][i, j] = (
                                self.T[a][xi + 1 + int(j / 2)] + self.T[a][xi + 1 + int((j + 1) / 2)]
                            ) / 2

                    # interior
                    else:
                        self.his_global_N[a][i] = xp.arange(self.n_his_locbf_N[a]) + i - (self.p[a] - 1)
                        self.his_global_D[a][i] = xp.arange(self.n_his_locbf_D[a]) + i - (self.p[a] - 1)

                        self.x_his_indices[a][i] = xp.arange(self.n_his[a]) + 2 * (i - (self.p[a] - 1))
                        self.coeffh_indices[a][i] = self.p[a] - 1
                        for j in range(2 * self.p[a] + 1):
                            self.x_his[a][i, j] = (
                                self.T[a][i + 1 + int(j / 2)] + self.T[a][i + 1 + int((j + 1) / 2)]
                            ) / 2

                    # local coefficient index
                    if i > 0:
                        for il in range(self.n_his_locbf_D[a]):
                            k_glob_new = self.his_global_D[a][i, il]
                            bol = k_glob_new == self.his_global_D[a][i - 1]

                            if xp.any(bol):
                                self.his_loccof_D[a][i, il] = self.his_loccof_D[a][i - 1, xp.where(bol)[0][0]] + 1

                            if (k_glob_new >= n_lambda_his[a] - self.p[a] - (self.p[a] - 2)) and (
                                self.his_loccof_D[a][i, il] == 0
                            ):
                                self.his_loccof_D[a][i, il] = self.his_add_D[a][counter_D]
                                counter_D += 1

                        for il in range(self.n_his_locbf_N[a]):
                            k_glob_new = self.his_global_N[a][i, il]
                            bol = k_glob_new == self.his_global_N[a][i - 1]

                            if xp.any(bol):
                                self.his_loccof_N[a][i, il] = self.his_loccof_N[a][i - 1, xp.where(bol)[0][0]] + 1

                            if (k_glob_new >= n_lambda_his[a] - self.p[a] - (self.p[a] - 2)) and (
                                self.his_loccof_N[a][i, il] == 0
                            ):
                                self.his_loccof_N[a][i, il] = self.his_add_N[a][counter_N]
                                counter_N += 1

                # quadrature points and weights
                self.pts[a], self.wts[a] = bsp.quadrature_grid(
                    xp.unique(self.x_his[a].flatten()),
                    self.pts_loc[a],
                    self.wts_loc[a],
                )

            else:
                # maximum number of non-vanishing coefficients
                self.n_his_nvcof_D[a] = 2 * self.p[a] - 1
                self.n_his_nvcof_N[a] = 2 * self.p[a]

                # shift local coefficients --> global coefficients (D)
                self.his_shift_D[a] = xp.arange(self.NbaseD[a]) - (self.p[a] - 1)

                # shift local coefficients --> global coefficients (N)
                self.his_shift_N[a] = xp.arange(self.NbaseD[a]) - self.p[a]

                for i in range(n_lambda_his[a]):
                    self.his_global_N[a][i] = (xp.arange(self.n_his_locbf_N[a]) + i - (self.p[a] - 1)) % self.NbaseN[a]
                    self.his_global_D[a][i] = (xp.arange(self.n_his_locbf_D[a]) + i - (self.p[a] - 1)) % self.NbaseD[a]
                    self.his_loccof_N[a][i] = xp.arange(self.n_his_locbf_N[a] - 1, -1, -1)
                    self.his_loccof_D[a][i] = xp.arange(self.n_his_locbf_D[a] - 1, -1, -1)

                    self.x_his_indices[a][i] = (xp.arange(self.n_his[a]) + 2 * (i - (self.p[a] - 1))) % (
                        2 * self.Nel[a]
                    )
                    self.coeffh_indices[a][i] = 0

                    for j in range(2 * self.p[a] + 1):
                        self.x_his[a][i, j] = (self.T[a][i + 1 + int(j / 2)] + self.T[a][i + 1 + int((j + 1) / 2)]) / 2

                # quadrature points and weights
                self.pts[a], self.wts[a] = bsp.quadrature_grid(
                    xp.append(xp.unique(self.x_his[a].flatten() % 1.0), 1.0),
                    self.pts_loc[a],
                    self.wts_loc[a],
                )

        # evaluate N basis functions at interpolation and quadrature points
        self.basisN_int = [
            bsp.collocation_matrix(T, p, x_int, bc) for T, p, x_int, bc in zip(self.T, self.p, self.x_int, self.bc)
        ]

        self.basisN_his = [
            bsp.collocation_matrix(T, p, pts.flatten(), bc).reshape(pts[:, 0].size, pts[0, :].size, NbaseN)
            for T, p, pts, bc, NbaseN in zip(self.T, self.p, self.pts, self.bc, self.NbaseN)
        ]

        # evaluate D basis functions at interpolation and quadrature points
        self.basisD_int = [
            bsp.collocation_matrix(T[1:-1], p - 1, x_int, bc, normalize=True)
            for T, p, x_int, bc in zip(self.T, self.p, self.x_int, self.bc)
        ]

        self.basisD_his = [
            bsp.collocation_matrix(T[1:-1], p - 1, pts.flatten(), bc, normalize=True).reshape(
                pts[:, 0].size,
                pts[0, :].size,
                NbaseD,
            )
            for T, p, pts, bc, NbaseD in zip(self.T, self.p, self.pts, self.bc, self.NbaseD)
        ]

    # ========================================================================
    def projection_Q_0form(self, domain):
        """
        Computes the sparse matrix of the expression pi_2(rho3_eq * lambda^0) with the output (coefficients, basis_fun of lambda^2).

        The following blocks need to be computed:

        1 - component [int, his, his] : (N, N, N)*rho3_eq, None             , None
        2 - component [his, int, his] : None             , (N, N, N)*rho3_eq, None
        3 - component [his, his, int] : None             , None             , (N, N, N)*rho3_eq

        An analytical mapping is called from struphy.geometry.mappings_analytical.

        Parameters
        ----------
        domain : domain
            domain object created with struphy.geometry.domain that defines the geometry

        Returns
        -------
        Q : sparse matrix in csc-format
            the projection of each basis function in V0 on V2 weighted with rho3_eq
        """

        # non-vanishing coefficients
        Q11 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
            dtype=float,
        )
        Q22 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
            dtype=float,
        )
        Q33 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )

        # size of interpolation/quadrature points of the 3 components
        n_unique1 = [self.x_int[0].size, self.pts[1].flatten().size, self.pts[2].flatten().size]
        n_unique2 = [self.pts[0].flatten().size, self.x_int[1].size, self.pts[2].flatten().size]
        n_unique3 = [self.pts[0].flatten().size, self.pts[1].flatten().size, self.x_int[2].size]

        # ========= assembly of 1 - component (pi2_1 : int, his, his) ============
        mat_eq = xp.empty((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float)

        ker_eva.kernel_eva(
            self.x_int[0],
            self.pts[1].flatten(),
            self.pts[2].flatten(),
            mat_eq,
            11,
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

        ker_loc.kernel_pi2_1(
            [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]],
            [self.n_quad[1], self.n_quad[2]],
            [self.n_int[0], self.n_his[1], self.n_his[2]],
            [self.n_int_locbf_N[0], self.n_his_locbf_N[1], self.n_his_locbf_N[2]],
            self.int_global_N[0],
            self.his_global_N[1],
            self.his_global_N[2],
            self.int_loccof_N[0],
            self.his_loccof_N[1],
            self.his_loccof_N[2],
            self.coeff_i[0],
            self.coeff_h[1],
            self.coeff_h[2],
            self.coeffi_indices[0],
            self.coeffh_indices[1],
            self.coeffh_indices[2],
            self.basisN_int[0],
            self.basisN_his[1],
            self.basisN_his[2],
            self.x_int_indices[0],
            self.x_his_indices[1],
            self.x_his_indices[2],
            self.wts[1],
            self.wts[2],
            Q11,
            mat_eq.reshape(
                n_unique1[0],
                self.pts[1][:, 0].size,
                self.pts[1][0, :].size,
                self.pts[2][:, 0].size,
                self.pts[2][0, :].size,
            ),
        )

        # ========= assembly of 2 - component (pi2_2 : his, int, his) ============
        mat_eq = xp.empty((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.x_int[1],
            self.pts[2].flatten(),
            mat_eq,
            11,
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

        ker_loc.kernel_pi2_2(
            [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]],
            [self.n_quad[0], self.n_quad[2]],
            [self.n_his[0], self.n_int[1], self.n_his[2]],
            [self.n_his_locbf_N[0], self.n_int_locbf_N[1], self.n_his_locbf_N[2]],
            self.his_global_N[0],
            self.int_global_N[1],
            self.his_global_N[2],
            self.his_loccof_N[0],
            self.int_loccof_N[1],
            self.his_loccof_N[2],
            self.coeff_h[0],
            self.coeff_i[1],
            self.coeff_h[2],
            self.coeffh_indices[0],
            self.coeffi_indices[1],
            self.coeffh_indices[2],
            self.basisN_his[0],
            self.basisN_int[1],
            self.basisN_his[2],
            self.x_his_indices[0],
            self.x_int_indices[1],
            self.x_his_indices[2],
            self.wts[0],
            self.wts[2],
            Q22,
            mat_eq.reshape(
                self.pts[0][:, 0].size,
                self.pts[0][0, :].size,
                n_unique2[1],
                self.pts[2][:, 0].size,
                self.pts[2][0, :].size,
            ),
        )

        # ========= assembly of 3 - component (pi2_3 : his, his, int) ============
        mat_eq = xp.empty((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.pts[1].flatten(),
            self.x_int[2],
            mat_eq,
            11,
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

        ker_loc.kernel_pi2_3(
            [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]],
            [self.n_quad[0], self.n_quad[1]],
            [self.n_his[0], self.n_his[1], self.n_int[2]],
            [self.n_his_locbf_N[0], self.n_his_locbf_N[1], self.n_int_locbf_N[2]],
            self.his_global_N[0],
            self.his_global_N[1],
            self.int_global_N[2],
            self.his_loccof_N[0],
            self.his_loccof_N[1],
            self.int_loccof_N[2],
            self.coeff_h[0],
            self.coeff_h[1],
            self.coeff_i[2],
            self.coeffh_indices[0],
            self.coeffh_indices[1],
            self.coeffi_indices[2],
            self.basisN_his[0],
            self.basisN_his[1],
            self.basisN_int[2],
            self.x_his_indices[0],
            self.x_his_indices[1],
            self.x_int_indices[2],
            self.wts[0],
            self.wts[1],
            Q33,
            mat_eq.reshape(
                self.pts[0][:, 0].size,
                self.pts[0][0, :].size,
                self.pts[1][:, 0].size,
                self.pts[1][0, :].size,
                n_unique3[2],
            ),
        )

        # ========= conversion to sparse matrices (1 - component) =================
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseD[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        Q11 = spa.csc_matrix(
            (Q11.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseN[0] * self.NbaseD[1] * self.NbaseD[2]),
        )
        Q11.eliminate_zeros()

        # ========= conversion to sparse matrices (2 - component) =================
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseN[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        Q22 = spa.csc_matrix(
            (Q22.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseD[0] * self.NbaseN[1] * self.NbaseD[2]),
        )
        Q22.eliminate_zeros()

        # ========= conversion to sparse matrices (3 - component) =================
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseD[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        Q33 = spa.csc_matrix(
            (Q33.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseD[0] * self.NbaseD[1] * self.NbaseN[2]),
        )
        Q33.eliminate_zeros()

        self.Q = spa.bmat([[Q11.T, None, None], [None, Q22.T, None], [None, None, Q33.T]], format="csc")

    # ========================================================================
    def projection_Q_2form(self, domain):
        """
        Computes the sparse matrix of the expression pi_2(rho3_eq * lambda^2) with the output (coefficients, basis_fun of lambda^2).

        The following blocks need to be computed:

        1 - component [int, his, his] : (N, D, D)*rho3_eq, None             , None
        2 - component [his, int, his] : None             , (D, N, D)*rho3_eq, None
        3 - component [his, his, int] : None             , None             , (D, D, N)*rho3_eq

        An analytical mapping is called from struphy.geometry.mappings_analytical.

        Parameters
        ----------
        domain : domain
            domain object created with struphy.geometry.domain that defines the geometry

        Returns
        -------
        Q : sparse matrix in csc-format
            the projection of each basis function in V2 on V2 weighted with rho3_eq
        """

        # non-vanishing coefficients
        Q11 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseD[1],
                self.NbaseD[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_D[1],
                self.n_his_nvcof_D[2],
            ),
            dtype=float,
        )
        Q22 = xp.empty(
            (
                self.NbaseD[0],
                self.NbaseN[1],
                self.NbaseD[2],
                self.n_his_nvcof_D[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_D[2],
            ),
            dtype=float,
        )
        Q33 = xp.empty(
            (
                self.NbaseD[0],
                self.NbaseD[1],
                self.NbaseN[2],
                self.n_his_nvcof_D[0],
                self.n_his_nvcof_D[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )

        # size of interpolation/quadrature points of the 3 components
        n_unique1 = [self.x_int[0].size, self.pts[1].flatten().size, self.pts[2].flatten().size]
        n_unique2 = [self.pts[0].flatten().size, self.x_int[1].size, self.pts[2].flatten().size]
        n_unique3 = [self.pts[0].flatten().size, self.pts[1].flatten().size, self.x_int[2].size]

        # ========= assembly of 1 - component (pi2_1 : int, his, his) ============
        mat_eq = xp.empty((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float)

        ker_eva.kernel_eva(
            self.x_int[0],
            self.pts[1].flatten(),
            self.pts[2].flatten(),
            mat_eq,
            11,
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

        ker_loc.kernel_pi2_1(
            [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]],
            [self.n_quad[1], self.n_quad[2]],
            [self.n_int[0], self.n_his[1], self.n_his[2]],
            [self.n_int_locbf_N[0], self.n_his_locbf_D[1], self.n_his_locbf_D[2]],
            self.int_global_N[0],
            self.his_global_D[1],
            self.his_global_D[2],
            self.int_loccof_N[0],
            self.his_loccof_D[1],
            self.his_loccof_D[2],
            self.coeff_i[0],
            self.coeff_h[1],
            self.coeff_h[2],
            self.coeffi_indices[0],
            self.coeffh_indices[1],
            self.coeffh_indices[2],
            self.basisN_int[0],
            self.basisD_his[1],
            self.basisD_his[2],
            self.x_int_indices[0],
            self.x_his_indices[1],
            self.x_his_indices[2],
            self.wts[1],
            self.wts[2],
            Q11,
            mat_eq.reshape(
                n_unique1[0],
                self.pts[1][:, 0].size,
                self.pts[1][0, :].size,
                self.pts[2][:, 0].size,
                self.pts[2][0, :].size,
            ),
        )

        # ========= assembly of 2 - component (pi2_2 : his, int, his) ============
        mat_eq = xp.empty((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.x_int[1],
            self.pts[2].flatten(),
            mat_eq,
            11,
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

        ker_loc.kernel_pi2_2(
            [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]],
            [self.n_quad[0], self.n_quad[2]],
            [self.n_his[0], self.n_int[1], self.n_his[2]],
            [self.n_his_locbf_D[0], self.n_int_locbf_N[1], self.n_his_locbf_D[2]],
            self.his_global_D[0],
            self.int_global_N[1],
            self.his_global_D[2],
            self.his_loccof_D[0],
            self.int_loccof_N[1],
            self.his_loccof_D[2],
            self.coeff_h[0],
            self.coeff_i[1],
            self.coeff_h[2],
            self.coeffh_indices[0],
            self.coeffi_indices[1],
            self.coeffh_indices[2],
            self.basisD_his[0],
            self.basisN_int[1],
            self.basisD_his[2],
            self.x_his_indices[0],
            self.x_int_indices[1],
            self.x_his_indices[2],
            self.wts[0],
            self.wts[2],
            Q22,
            mat_eq.reshape(
                self.pts[0][:, 0].size,
                self.pts[0][0, :].size,
                n_unique2[1],
                self.pts[2][:, 0].size,
                self.pts[2][0, :].size,
            ),
        )

        # ========= assembly of 3 - component (pi2_3 : his, his, int) ============
        mat_eq = xp.empty((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.pts[1].flatten(),
            self.x_int[2],
            mat_eq,
            11,
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

        ker_loc.kernel_pi2_3(
            [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]],
            [self.n_quad[0], self.n_quad[1]],
            [self.n_his[0], self.n_his[1], self.n_int[2]],
            [self.n_his_locbf_D[0], self.n_his_locbf_D[1], self.n_int_locbf_N[2]],
            self.his_global_D[0],
            self.his_global_D[1],
            self.int_global_N[2],
            self.his_loccof_D[0],
            self.his_loccof_D[1],
            self.int_loccof_N[2],
            self.coeff_h[0],
            self.coeff_h[1],
            self.coeff_i[2],
            self.coeffh_indices[0],
            self.coeffh_indices[1],
            self.coeffi_indices[2],
            self.basisD_his[0],
            self.basisD_his[1],
            self.basisN_int[2],
            self.x_his_indices[0],
            self.x_his_indices[1],
            self.x_int_indices[2],
            self.wts[0],
            self.wts[1],
            Q33,
            mat_eq.reshape(
                self.pts[0][:, 0].size,
                self.pts[0][0, :].size,
                self.pts[1][:, 0].size,
                self.pts[1][0, :].size,
                n_unique3[2],
            ),
        )

        # ========= conversion to sparse matrices (1 - component) =================
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseD[1],
                self.NbaseD[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_D[1],
                self.n_his_nvcof_D[2],
            ),
        )
        row = self.NbaseD[1] * self.NbaseD[2] * indices[0] + self.NbaseD[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseD[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        Q11 = spa.csc_matrix(
            (Q11.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseD[1] * self.NbaseD[2], self.NbaseN[0] * self.NbaseD[1] * self.NbaseD[2]),
        )
        Q11.eliminate_zeros()

        # ========= conversion to sparse matrices (2 - component) =================
        indices = xp.indices(
            (
                self.NbaseD[0],
                self.NbaseN[1],
                self.NbaseD[2],
                self.n_his_nvcof_D[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_D[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseD[2] * indices[0] + self.NbaseD[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseN[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        Q22 = spa.csc_matrix(
            (Q22.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseD[0] * self.NbaseN[1] * self.NbaseD[2], self.NbaseD[0] * self.NbaseN[1] * self.NbaseD[2]),
        )
        Q22.eliminate_zeros()

        # ========= conversion to sparse matrices (3 - component) =================
        indices = xp.indices(
            (
                self.NbaseD[0],
                self.NbaseD[1],
                self.NbaseN[2],
                self.n_his_nvcof_D[0],
                self.n_his_nvcof_D[1],
                self.n_int_nvcof_N[2],
            ),
        )
        row = self.NbaseD[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseD[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        Q33 = spa.csc_matrix(
            (Q33.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseD[0] * self.NbaseD[1] * self.NbaseN[2], self.NbaseD[0] * self.NbaseD[1] * self.NbaseN[2]),
        )
        Q33.eliminate_zeros()

        self.Q = spa.bmat([[Q11.T, None, None], [None, Q22.T, None], [None, None, Q33.T]], format="csc")

    # ========================================================================
    def projection_W_0form(self, domain):
        """
        Computes the sparse matrix of the expression pi_0(rho0_eq * lambda^0) with the output (coefficients, basis_fun of lambda^2).

        The following blocks need to be computed:

        1 - component [int, int, int] : (N, N, N)*rho0_eq, None             , None
        2 - component [int, int, int] : None             , (N, N, N)*rho0_eq, None
        3 - component [int, int, int] : None             , None             , (N, N, N)*rho0_eq

        An analytical mapping is called from struphy.geometry.mappings_analytical.

        Parameters
        ----------
        domain : domain
            domain object created with struphy.geometry.domain that defines the geometry

        Returns
        -------
        W : sparse matrix in csc-format
            the projection of each basis function in V0 on V0 weighted with rho0_eq
        """

        # non-vanishing coefficients
        W1 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )
        # W2 = xp.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.n_int_nvcof_N[0], self.n_int_nvcof_N[1], self.n_int_nvcof_N[2]), dtype=float)
        # W3 = xp.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.n_int_nvcof_N[0], self.n_int_nvcof_N[1], self.n_int_nvcof_N[2]), dtype=float)

        # size of interpolation/quadrature points of the 3 components
        n_unique = [self.x_int[0].size, self.x_int[1].size, self.x_int[2].size]

        # assembly
        mat_eq = xp.empty((n_unique[0], n_unique[1], n_unique[2]), dtype=float)

        ker_eva.kernel_eva(
            self.x_int[0],
            self.x_int[1],
            self.x_int[2],
            mat_eq,
            12,
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

        ker_loc.kernel_pi0(
            self.NbaseN,
            self.n_int,
            self.n_int_locbf_N,
            self.int_global_N[0],
            self.int_global_N[1],
            self.int_global_N[2],
            self.int_loccof_N[0],
            self.int_loccof_N[1],
            self.int_loccof_N[2],
            self.coeff_i[0],
            self.coeff_i[1],
            self.coeff_i[2],
            self.coeffi_indices[0],
            self.coeffi_indices[1],
            self.coeffi_indices[2],
            self.basisN_int[0],
            self.basisN_int[1],
            self.basisN_int[2],
            self.x_int_indices[0],
            self.x_int_indices[1],
            self.x_int_indices[2],
            W1,
            mat_eq,
        )

        # ker_loc.kernel_pi0(self.NbaseN, self.n_int, self.n_int_locbf_N, self.int_global_N[0], self.int_global_N[1], self.int_global_N[2], self.int_loccof_N[0], self.int_loccof_N[1], self.int_loccof_N[2], self.coeff_i[0], self.coeff_i[1], self.coeff_i[2], self.coeffi_indices[0], self.coeffi_indices[1], self.coeffi_indices[2], self.basisN_int[0], self.basisN_int[1], self.basisN_int[2], self.x_int_indices[0], self.x_int_indices[1], self.x_int_indices[2], W2, mat_eq)

        # ker_loc.kernel_pi0(self.NbaseN, self.n_int, self.n_int_locbf_N, self.int_global_N[0], self.int_global_N[1], self.int_global_N[2], self.int_loccof_N[0], self.int_loccof_N[1], self.int_loccof_N[2], self.coeff_i[0], self.coeff_i[1], self.coeff_i[2], self.coeffi_indices[0], self.coeffi_indices[1], self.coeffi_indices[2], self.basisN_int[0], self.basisN_int[1], self.basisN_int[2], self.x_int_indices[0], self.x_int_indices[1], self.x_int_indices[2], W3, mat_eq)

        """
        if self.bc[0] == False:
            # apply Dirichlet boundary conditions for u1 at eta1 = 0
            if bc_u1[0][0] == 'dirichlet':
                W1[0]  = 0.

            # apply Dirichlet boundary conditions for u1 at eta1 = 1
            if bc_u1[0][1] == 'dirichlet':
                W1[-1] = 0.
        """

        # conversion to sparse matrix
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
        )

        # row indices
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        # column indices
        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseN[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        # create sparse matrices
        W1 = spa.csc_matrix(
            (W1.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2]),
        )
        W1.eliminate_zeros()

        # W2 = spa.csc_matrix((W2.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2]))
        # W2.eliminate_zeros()

        # W3 = spa.csc_matrix((W3.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2]))
        # W3.eliminate_zeros()

        self.W = spa.bmat([[W1.T, None, None], [None, W1.T, None], [None, None, W1.T]], format="csc")

    # =========================================================================
    def projection_T_0form(self, domain):
        """
        Computes the matrix of the expression pi_1(b2_eq * lambda^0) with the output (coefficients, basis_fun of lambda^0).

        The following blocks need to be computed:

        1 - component [his, int, int] :   None       , -(N, N, N)*B3,  (N, N, N)*B2
        2 - component [int, his, int] :  (N, N, N)*B3,   None       , -(N, N, N)*B1
        3 - component [int, int, his] : -(N, N, N)*B2,  (N, N, N)*B1,   None

        An analytical mapping is called from struphy.geometry.mappings_analytical.

        Parameters
        ----------
        domain : domain
            domain object created with struphy.geometry.domain that defines the geometry

        Returns
        -------
        T : sparse matrix in csc-format
            the projection of each basis function in V2 on V1 weighted with b2_eq
        """

        # non-vanishing coefficients
        T12 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )
        T13 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )

        T21 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )
        T23 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )

        T31 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
            dtype=float,
        )
        T32 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
            dtype=float,
        )

        # unique interpolation points
        n_unique1 = [self.pts[0].flatten().size, self.x_int[1].size, self.x_int[2].size]
        n_unique2 = [self.x_int[0].size, self.pts[1].flatten().size, self.x_int[2].size]
        n_unique3 = [self.x_int[0].size, self.x_int[1].size, self.pts[2].flatten().size]

        # ================= assembly of 1 - component (pi1_1 : his, int, int) ============
        mat_eq = xp.empty((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.x_int[1],
            self.x_int[2],
            mat_eq,
            23,
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

        ker_loc.kernel_pi1_1(
            [self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]],
            self.n_quad[0],
            [self.n_his[0], self.n_int[1], self.n_int[2]],
            [self.n_his_locbf_N[0], self.n_int_locbf_N[1], self.n_int_locbf_N[2]],
            self.his_global_N[0],
            self.int_global_N[1],
            self.int_global_N[2],
            self.his_loccof_N[0],
            self.int_loccof_N[1],
            self.int_loccof_N[2],
            self.coeff_h[0],
            self.coeff_i[1],
            self.coeff_i[2],
            self.coeffh_indices[0],
            self.coeffi_indices[1],
            self.coeffi_indices[2],
            self.basisN_his[0],
            self.basisN_int[1],
            self.basisN_int[2],
            self.x_his_indices[0],
            self.x_int_indices[1],
            self.x_int_indices[2],
            self.wts[0],
            T12,
            mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique1[1], n_unique1[2]),
        )

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.x_int[1],
            self.x_int[2],
            mat_eq,
            22,
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

        ker_loc.kernel_pi1_1(
            [self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]],
            self.n_quad[0],
            [self.n_his[0], self.n_int[1], self.n_int[2]],
            [self.n_his_locbf_N[0], self.n_int_locbf_N[1], self.n_int_locbf_N[2]],
            self.his_global_N[0],
            self.int_global_N[1],
            self.int_global_N[2],
            self.his_loccof_N[0],
            self.int_loccof_N[1],
            self.int_loccof_N[2],
            self.coeff_h[0],
            self.coeff_i[1],
            self.coeff_i[2],
            self.coeffh_indices[0],
            self.coeffi_indices[1],
            self.coeffi_indices[2],
            self.basisN_his[0],
            self.basisN_int[1],
            self.basisN_int[2],
            self.x_his_indices[0],
            self.x_int_indices[1],
            self.x_int_indices[2],
            self.wts[0],
            T13,
            mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique1[1], n_unique1[2]),
        )

        # ================= assembly of 2 - component (PI_1_2 : int, his, int) ============
        mat_eq = xp.empty((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float)

        ker_eva.kernel_eva(
            self.x_int[0],
            self.pts[1].flatten(),
            self.x_int[2],
            mat_eq,
            23,
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

        ker_loc.kernel_pi1_2(
            [self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]],
            self.n_quad[1],
            [self.n_int[0], self.n_his[1], self.n_int[2]],
            [self.n_int_locbf_N[0], self.n_his_locbf_N[1], self.n_int_locbf_N[2]],
            self.int_global_N[0],
            self.his_global_N[1],
            self.int_global_N[2],
            self.int_loccof_N[0],
            self.his_loccof_N[1],
            self.int_loccof_N[2],
            self.coeff_i[0],
            self.coeff_h[1],
            self.coeff_i[2],
            self.coeffi_indices[0],
            self.coeffh_indices[1],
            self.coeffi_indices[2],
            self.basisN_int[0],
            self.basisN_his[1],
            self.basisN_int[2],
            self.x_int_indices[0],
            self.x_his_indices[1],
            self.x_int_indices[2],
            self.wts[1],
            T21,
            mat_eq.reshape(n_unique2[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique2[2]),
        )

        """
        if self.bc[0] == False:
            # apply Dirichlet boundary conditions for u1 at eta1 = 0
            if bc_u1[0][0] == 'dirichlet':
                T21[0]  = 0.

            # apply Dirichlet boundary conditions for u1 at eta1 = 1
            if bc_u1[0][1] == 'dirichlet':
                T21[-1] = 0.
        """

        ker_eva.kernel_eva(
            self.x_int[0],
            self.pts[1].flatten(),
            self.x_int[2],
            mat_eq,
            21,
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

        ker_loc.kernel_pi1_2(
            [self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]],
            self.n_quad[1],
            [self.n_int[0], self.n_his[1], self.n_int[2]],
            [self.n_int_locbf_N[0], self.n_his_locbf_N[1], self.n_int_locbf_N[2]],
            self.int_global_N[0],
            self.his_global_N[1],
            self.int_global_N[2],
            self.int_loccof_N[0],
            self.his_loccof_N[1],
            self.int_loccof_N[2],
            self.coeff_i[0],
            self.coeff_h[1],
            self.coeff_i[2],
            self.coeffi_indices[0],
            self.coeffh_indices[1],
            self.coeffi_indices[2],
            self.basisN_int[0],
            self.basisN_his[1],
            self.basisN_int[2],
            self.x_int_indices[0],
            self.x_his_indices[1],
            self.x_int_indices[2],
            self.wts[1],
            T23,
            mat_eq.reshape(n_unique2[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique2[2]),
        )

        # ================= assembly of 3 - component (PI_1_3 : int, int, his) ============
        mat_eq = xp.empty((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float)

        ker_eva.kernel_eva(
            self.x_int[0],
            self.x_int[1],
            self.pts[2].flatten(),
            mat_eq,
            22,
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

        ker_loc.kernel_pi1_3(
            [self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]],
            self.n_quad[2],
            [self.n_int[0], self.n_int[1], self.n_his[2]],
            [self.n_int_locbf_N[0], self.n_int_locbf_N[1], self.n_his_locbf_N[2]],
            self.int_global_N[0],
            self.int_global_N[1],
            self.his_global_N[2],
            self.int_loccof_N[0],
            self.int_loccof_N[1],
            self.his_loccof_N[2],
            self.coeff_i[0],
            self.coeff_i[1],
            self.coeff_h[2],
            self.coeffi_indices[0],
            self.coeffi_indices[1],
            self.coeffh_indices[2],
            self.basisN_int[0],
            self.basisN_int[1],
            self.basisN_his[2],
            self.x_int_indices[0],
            self.x_int_indices[1],
            self.x_his_indices[2],
            self.wts[2],
            T31,
            mat_eq.reshape(n_unique3[0], n_unique3[1], self.pts[2][:, 0].size, self.pts[2][0, :].size),
        )

        """
        if self.bc[0] == False:
            # apply Dirichlet boundary conditions for u1 at eta1 = 0
            if bc_u1[0][0] == 'dirichlet':
                T31[0]  = 0.

            # apply Dirichlet boundary conditions for u1 at eta1 = 1
            if bc_u1[0][1] == 'dirichlet':
                T31[-1] = 0.
        """

        ker_eva.kernel_eva(
            self.x_int[0],
            self.x_int[1],
            self.pts[2].flatten(),
            mat_eq,
            21,
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

        ker_loc.kernel_pi1_3(
            [self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]],
            self.n_quad[2],
            [self.n_int[0], self.n_int[1], self.n_his[2]],
            [self.n_int_locbf_N[0], self.n_int_locbf_N[1], self.n_his_locbf_N[2]],
            self.int_global_N[0],
            self.int_global_N[1],
            self.his_global_N[2],
            self.int_loccof_N[0],
            self.int_loccof_N[1],
            self.his_loccof_N[2],
            self.coeff_i[0],
            self.coeff_i[1],
            self.coeff_h[2],
            self.coeffi_indices[0],
            self.coeffi_indices[1],
            self.coeffh_indices[2],
            self.basisN_int[0],
            self.basisN_int[1],
            self.basisN_his[2],
            self.x_int_indices[0],
            self.x_int_indices[1],
            self.x_his_indices[2],
            self.wts[2],
            T32,
            mat_eq.reshape(n_unique3[0], n_unique3[1], self.pts[2][:, 0].size, self.pts[2][0, :].size),
        )

        # conversion to sparse matrices (1 - component)
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseN[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        T12 = spa.csc_matrix(
            (T12.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseD[0] * self.NbaseN[1] * self.NbaseN[2]),
        )
        T12.eliminate_zeros()

        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseN[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        T13 = spa.csc_matrix(
            (T13.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseD[0] * self.NbaseN[1] * self.NbaseN[2]),
        )
        T13.eliminate_zeros()

        # conversion to sparse matrices (2 - component)
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseD[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        T21 = spa.csc_matrix(
            (T21.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseN[0] * self.NbaseD[1] * self.NbaseN[2]),
        )
        T21.eliminate_zeros()

        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseD[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        T23 = spa.csc_matrix(
            (T23.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseN[0] * self.NbaseD[1] * self.NbaseN[2]),
        )
        T23.eliminate_zeros()

        # conversion to sparse matrices (3 - component)
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseN[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        T31 = spa.csc_matrix(
            (T31.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseN[0] * self.NbaseN[1] * self.NbaseD[2]),
        )
        T31.eliminate_zeros()

        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseN[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        T32 = spa.csc_matrix(
            (T32.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseN[0] * self.NbaseN[1] * self.NbaseD[2]),
        )
        T32.eliminate_zeros()

        self.TAU = spa.bmat([[None, -T12.T, T13.T], [T21.T, None, -T23.T], [-T31.T, T32.T, None]], format="csc")

    # =========================================================================
    def projection_T_1form(self, domain):
        """
        Computes the matrix of the expression pi_1(b2_eq * lambda^1) with the output (coefficients, basis_fun of lambda^1).

        The following blocks need to be computed:

        1 - component [his, int, int] :   None       , -(N, D, N)*B3,  (N, N, D)*B2
        2 - component [int, his, int] :  (D, N, N)*B3,   None       , -(N, N, D)*B1
        3 - component [int, int, his] : -(D, N, N)*B2,  (N, D, N)*B1,   None

        An analytical mapping is called from struphy.geometry.mappings_analytical.

        Parameters
        ----------
        domain : domain
            domain object created with struphy.geometry.domain that defines the geometry

        Returns
        -------
        T : sparse matrix in csc-format
            the projection of each basis function in V2 on V1 weighted with b2_eq
        """

        # non-vanishing coefficients
        T12 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseD[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_int_nvcof_D[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )
        T13 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseD[2],
                self.n_his_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_int_nvcof_D[2],
            ),
            dtype=float,
        )

        T21 = xp.empty(
            (
                self.NbaseD[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_D[0],
                self.n_his_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )
        T23 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseD[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_int_nvcof_D[2],
            ),
            dtype=float,
        )

        T31 = xp.empty(
            (
                self.NbaseD[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_D[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
            dtype=float,
        )
        T32 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseD[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_int_nvcof_D[1],
                self.n_his_nvcof_N[2],
            ),
            dtype=float,
        )

        # unique interpolation points
        n_unique1 = [self.pts[0].flatten().size, self.x_int[1].size, self.x_int[2].size]
        n_unique2 = [self.x_int[0].size, self.pts[1].flatten().size, self.x_int[2].size]
        n_unique3 = [self.x_int[0].size, self.x_int[1].size, self.pts[2].flatten().size]

        # ================= assembly of 1 - component (pi1_1 : his, int, int) ============
        mat_eq = xp.empty((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.x_int[1],
            self.x_int[2],
            mat_eq,
            23,
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

        ker_loc.kernel_pi1_1(
            [self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]],
            self.n_quad[0],
            [self.n_his[0], self.n_int[1], self.n_int[2]],
            [self.n_his_locbf_N[0], self.n_int_locbf_D[1], self.n_int_locbf_N[2]],
            self.his_global_N[0],
            self.int_global_D[1],
            self.int_global_N[2],
            self.his_loccof_N[0],
            self.int_loccof_D[1],
            self.int_loccof_N[2],
            self.coeff_h[0],
            self.coeff_i[1],
            self.coeff_i[2],
            self.coeffh_indices[0],
            self.coeffi_indices[1],
            self.coeffi_indices[2],
            self.basisN_his[0],
            self.basisD_int[1],
            self.basisN_int[2],
            self.x_his_indices[0],
            self.x_int_indices[1],
            self.x_int_indices[2],
            self.wts[0],
            T12,
            mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique1[1], n_unique1[2]),
        )

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.x_int[1],
            self.x_int[2],
            mat_eq,
            22,
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

        ker_loc.kernel_pi1_1(
            [self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]],
            self.n_quad[0],
            [self.n_his[0], self.n_int[1], self.n_int[2]],
            [self.n_his_locbf_N[0], self.n_int_locbf_N[1], self.n_int_locbf_D[2]],
            self.his_global_N[0],
            self.int_global_N[1],
            self.int_global_D[2],
            self.his_loccof_N[0],
            self.int_loccof_N[1],
            self.int_loccof_D[2],
            self.coeff_h[0],
            self.coeff_i[1],
            self.coeff_i[2],
            self.coeffh_indices[0],
            self.coeffi_indices[1],
            self.coeffi_indices[2],
            self.basisN_his[0],
            self.basisN_int[1],
            self.basisD_int[2],
            self.x_his_indices[0],
            self.x_int_indices[1],
            self.x_int_indices[2],
            self.wts[0],
            T13,
            mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique1[1], n_unique1[2]),
        )

        # ================= assembly of 2 - component (PI_1_2 : int, his, int) ============
        mat_eq = xp.empty((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float)

        ker_eva.kernel_eva(
            self.x_int[0],
            self.pts[1].flatten(),
            self.x_int[2],
            mat_eq,
            23,
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

        ker_loc.kernel_pi1_2(
            [self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]],
            self.n_quad[1],
            [self.n_int[0], self.n_his[1], self.n_int[2]],
            [self.n_int_locbf_D[0], self.n_his_locbf_N[1], self.n_int_locbf_N[2]],
            self.int_global_D[0],
            self.his_global_N[1],
            self.int_global_N[2],
            self.int_loccof_D[0],
            self.his_loccof_N[1],
            self.int_loccof_N[2],
            self.coeff_i[0],
            self.coeff_h[1],
            self.coeff_i[2],
            self.coeffi_indices[0],
            self.coeffh_indices[1],
            self.coeffi_indices[2],
            self.basisD_int[0],
            self.basisN_his[1],
            self.basisN_int[2],
            self.x_int_indices[0],
            self.x_his_indices[1],
            self.x_int_indices[2],
            self.wts[1],
            T21,
            mat_eq.reshape(n_unique2[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique2[2]),
        )

        ker_eva.kernel_eva(
            self.x_int[0],
            self.pts[1].flatten(),
            self.x_int[2],
            mat_eq,
            21,
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

        ker_loc.kernel_pi1_2(
            [self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]],
            self.n_quad[1],
            [self.n_int[0], self.n_his[1], self.n_int[2]],
            [self.n_int_locbf_N[0], self.n_his_locbf_N[1], self.n_int_locbf_D[2]],
            self.int_global_N[0],
            self.his_global_N[1],
            self.int_global_D[2],
            self.int_loccof_N[0],
            self.his_loccof_N[1],
            self.int_loccof_D[2],
            self.coeff_i[0],
            self.coeff_h[1],
            self.coeff_i[2],
            self.coeffi_indices[0],
            self.coeffh_indices[1],
            self.coeffi_indices[2],
            self.basisN_int[0],
            self.basisN_his[1],
            self.basisD_int[2],
            self.x_int_indices[0],
            self.x_his_indices[1],
            self.x_int_indices[2],
            self.wts[1],
            T23,
            mat_eq.reshape(n_unique2[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique2[2]),
        )

        # ================= assembly of 3 - component (PI_1_3 : int, int, his) ============
        mat_eq = xp.empty((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float)

        ker_eva.kernel_eva(
            self.x_int[0],
            self.x_int[1],
            self.pts[2].flatten(),
            mat_eq,
            22,
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

        ker_loc.kernel_pi1_3(
            [self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]],
            self.n_quad[2],
            [self.n_int[0], self.n_int[1], self.n_his[2]],
            [self.n_int_locbf_D[0], self.n_int_locbf_N[1], self.n_his_locbf_N[2]],
            self.int_global_D[0],
            self.int_global_N[1],
            self.his_global_N[2],
            self.int_loccof_D[0],
            self.int_loccof_N[1],
            self.his_loccof_N[2],
            self.coeff_i[0],
            self.coeff_i[1],
            self.coeff_h[2],
            self.coeffi_indices[0],
            self.coeffi_indices[1],
            self.coeffh_indices[2],
            self.basisD_int[0],
            self.basisN_int[1],
            self.basisN_his[2],
            self.x_int_indices[0],
            self.x_int_indices[1],
            self.x_his_indices[2],
            self.wts[2],
            T31,
            mat_eq.reshape(n_unique3[0], n_unique3[1], self.pts[2][:, 0].size, self.pts[2][0, :].size),
        )

        ker_eva.kernel_eva(
            self.x_int[0],
            self.x_int[1],
            self.pts[2].flatten(),
            mat_eq,
            21,
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

        ker_loc.kernel_pi1_3(
            [self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]],
            self.n_quad[2],
            [self.n_int[0], self.n_int[1], self.n_his[2]],
            [self.n_int_locbf_N[0], self.n_int_locbf_D[1], self.n_his_locbf_N[2]],
            self.int_global_N[0],
            self.int_global_D[1],
            self.his_global_N[2],
            self.int_loccof_N[0],
            self.int_loccof_D[1],
            self.his_loccof_N[2],
            self.coeff_i[0],
            self.coeff_i[1],
            self.coeff_h[2],
            self.coeffi_indices[0],
            self.coeffi_indices[1],
            self.coeffh_indices[2],
            self.basisN_int[0],
            self.basisD_int[1],
            self.basisN_his[2],
            self.x_int_indices[0],
            self.x_int_indices[1],
            self.x_his_indices[2],
            self.wts[2],
            T32,
            mat_eq.reshape(n_unique3[0], n_unique3[1], self.pts[2][:, 0].size, self.pts[2][0, :].size),
        )

        # conversion to sparse matrices (1 - component)
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseD[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_int_nvcof_D[1],
                self.n_int_nvcof_N[2],
            ),
        )
        row = self.NbaseD[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_D[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseN[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        T12 = spa.csc_matrix(
            (T12.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseD[1] * self.NbaseN[2], self.NbaseD[0] * self.NbaseN[1] * self.NbaseN[2]),
        )
        T12.eliminate_zeros()

        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseD[2],
                self.n_his_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_int_nvcof_D[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseD[2] * indices[0] + self.NbaseD[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_D[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseN[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        T13 = spa.csc_matrix(
            (T13.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseD[2], self.NbaseD[0] * self.NbaseN[1] * self.NbaseN[2]),
        )
        T13.eliminate_zeros()

        # conversion to sparse matrices (2 - component)
        indices = xp.indices(
            (
                self.NbaseD[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_D[0],
                self.n_his_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_D[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseD[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        T21 = spa.csc_matrix(
            (T21.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseD[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseN[0] * self.NbaseD[1] * self.NbaseN[2]),
        )
        T21.eliminate_zeros()

        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseD[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_int_nvcof_D[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseD[2] * indices[0] + self.NbaseD[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_D[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseD[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        T23 = spa.csc_matrix(
            (T23.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseD[2], self.NbaseN[0] * self.NbaseD[1] * self.NbaseN[2]),
        )
        T23.eliminate_zeros()

        # conversion to sparse matrices (3 - component)
        indices = xp.indices(
            (
                self.NbaseD[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_D[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_D[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseN[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        T31 = spa.csc_matrix(
            (T31.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseD[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseN[0] * self.NbaseN[1] * self.NbaseD[2]),
        )
        T31.eliminate_zeros()

        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseD[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_int_nvcof_D[1],
                self.n_his_nvcof_N[2],
            ),
        )
        row = self.NbaseD[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_D[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseN[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        T32 = spa.csc_matrix(
            (T32.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseD[1] * self.NbaseN[2], self.NbaseN[0] * self.NbaseN[1] * self.NbaseD[2]),
        )
        T32.eliminate_zeros()

        self.TAU = spa.bmat([[None, -T12.T, T13.T], [T21.T, None, -T23.T], [-T31.T, T32.T, None]], format="csc")

    # =========================================================================
    def projection_T_2form(self, domain):
        """
        Computes the matrix of the expression pi_1(b2_eq * lambda^2) with the output (coefficients, basis_fun of lambda^2).

        The following blocks need to be computed:

        1 - component [his, int, int] :   None       , -(D, N, D)*B3,  (D, D, N)*B2
        2 - component [int, his, int] :  (N, D, D)*B3,   None       , -(D, D, N)*B1
        3 - component [int, int, his] : -(N, D, D)*B2,  (D, N, D)*B1,   None

        An analytical mapping is called from struphy.geometry.mappings_analytical.

        Parameters
        ----------
        domain : domain
            domain object created with struphy.geometry.domain that defines the geometry

        Returns
        -------
        T : sparse matrix in csc-format
            the projection of each basis function in V2 on V1 weighted with b2_eq
        """

        # non-vanishing coefficients
        T12 = xp.empty(
            (
                self.NbaseD[0],
                self.NbaseN[1],
                self.NbaseD[2],
                self.n_his_nvcof_D[0],
                self.n_int_nvcof_N[1],
                self.n_int_nvcof_D[2],
            ),
            dtype=float,
        )
        T13 = xp.empty(
            (
                self.NbaseD[0],
                self.NbaseD[1],
                self.NbaseN[2],
                self.n_his_nvcof_D[0],
                self.n_int_nvcof_D[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )

        T21 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseD[1],
                self.NbaseD[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_D[1],
                self.n_int_nvcof_D[2],
            ),
            dtype=float,
        )
        T23 = xp.empty(
            (
                self.NbaseD[0],
                self.NbaseD[1],
                self.NbaseN[2],
                self.n_int_nvcof_D[0],
                self.n_his_nvcof_D[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )

        T31 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseD[1],
                self.NbaseD[2],
                self.n_int_nvcof_N[0],
                self.n_int_nvcof_D[1],
                self.n_his_nvcof_D[2],
            ),
            dtype=float,
        )
        T32 = xp.empty(
            (
                self.NbaseD[0],
                self.NbaseN[1],
                self.NbaseD[2],
                self.n_int_nvcof_D[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_D[2],
            ),
            dtype=float,
        )

        # unique interpolation points
        n_unique1 = [self.pts[0].flatten().size, self.x_int[1].size, self.x_int[2].size]
        n_unique2 = [self.x_int[0].size, self.pts[1].flatten().size, self.x_int[2].size]
        n_unique3 = [self.x_int[0].size, self.x_int[1].size, self.pts[2].flatten().size]

        # ================= assembly of 1 - component (pi1_1 : his, int, int) ============
        mat_eq = xp.empty((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.x_int[1],
            self.x_int[2],
            mat_eq,
            23,
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

        ker_loc.kernel_pi1_1(
            [self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]],
            self.n_quad[0],
            [self.n_his[0], self.n_int[1], self.n_int[2]],
            [self.n_his_locbf_D[0], self.n_int_locbf_N[1], self.n_int_locbf_D[2]],
            self.his_global_D[0],
            self.int_global_N[1],
            self.int_global_D[2],
            self.his_loccof_D[0],
            self.int_loccof_N[1],
            self.int_loccof_D[2],
            self.coeff_h[0],
            self.coeff_i[1],
            self.coeff_i[2],
            self.coeffh_indices[0],
            self.coeffi_indices[1],
            self.coeffi_indices[2],
            self.basisD_his[0],
            self.basisN_int[1],
            self.basisD_int[2],
            self.x_his_indices[0],
            self.x_int_indices[1],
            self.x_int_indices[2],
            self.wts[0],
            T12,
            mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique1[1], n_unique1[2]),
        )

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.x_int[1],
            self.x_int[2],
            mat_eq,
            22,
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

        ker_loc.kernel_pi1_1(
            [self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]],
            self.n_quad[0],
            [self.n_his[0], self.n_int[1], self.n_int[2]],
            [self.n_his_locbf_D[0], self.n_int_locbf_D[1], self.n_int_locbf_N[2]],
            self.his_global_D[0],
            self.int_global_D[1],
            self.int_global_N[2],
            self.his_loccof_D[0],
            self.int_loccof_D[1],
            self.int_loccof_N[2],
            self.coeff_h[0],
            self.coeff_i[1],
            self.coeff_i[2],
            self.coeffh_indices[0],
            self.coeffi_indices[1],
            self.coeffi_indices[2],
            self.basisD_his[0],
            self.basisD_int[1],
            self.basisN_int[2],
            self.x_his_indices[0],
            self.x_int_indices[1],
            self.x_int_indices[2],
            self.wts[0],
            T13,
            mat_eq.reshape(self.pts[0][:, 0].size, self.pts[0][0, :].size, n_unique1[1], n_unique1[2]),
        )

        # ================= assembly of 2 - component (PI_1_2 : int, his, int) ============
        mat_eq = xp.empty((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float)

        ker_eva.kernel_eva(
            self.x_int[0],
            self.pts[1].flatten(),
            self.x_int[2],
            mat_eq,
            23,
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

        ker_loc.kernel_pi1_2(
            [self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]],
            self.n_quad[1],
            [self.n_int[0], self.n_his[1], self.n_int[2]],
            [self.n_int_locbf_N[0], self.n_his_locbf_D[1], self.n_int_locbf_D[2]],
            self.int_global_N[0],
            self.his_global_D[1],
            self.int_global_D[2],
            self.int_loccof_N[0],
            self.his_loccof_D[1],
            self.int_loccof_D[2],
            self.coeff_i[0],
            self.coeff_h[1],
            self.coeff_i[2],
            self.coeffi_indices[0],
            self.coeffh_indices[1],
            self.coeffi_indices[2],
            self.basisN_int[0],
            self.basisD_his[1],
            self.basisD_int[2],
            self.x_int_indices[0],
            self.x_his_indices[1],
            self.x_int_indices[2],
            self.wts[1],
            T21,
            mat_eq.reshape(n_unique2[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique2[2]),
        )

        ker_eva.kernel_eva(
            self.x_int[0],
            self.pts[1].flatten(),
            self.x_int[2],
            mat_eq,
            21,
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

        ker_loc.kernel_pi1_2(
            [self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]],
            self.n_quad[1],
            [self.n_int[0], self.n_his[1], self.n_int[2]],
            [self.n_int_locbf_D[0], self.n_his_locbf_D[1], self.n_int_locbf_N[2]],
            self.int_global_D[0],
            self.his_global_D[1],
            self.int_global_N[2],
            self.int_loccof_D[0],
            self.his_loccof_D[1],
            self.int_loccof_N[2],
            self.coeff_i[0],
            self.coeff_h[1],
            self.coeff_i[2],
            self.coeffi_indices[0],
            self.coeffh_indices[1],
            self.coeffi_indices[2],
            self.basisD_int[0],
            self.basisD_his[1],
            self.basisN_int[2],
            self.x_int_indices[0],
            self.x_his_indices[1],
            self.x_int_indices[2],
            self.wts[1],
            T23,
            mat_eq.reshape(n_unique2[0], self.pts[1][:, 0].size, self.pts[1][0, :].size, n_unique2[2]),
        )

        # ================= assembly of 3 - component (PI_1_3 : int, int, his) ============
        mat_eq = xp.empty((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float)

        ker_eva.kernel_eva(
            self.x_int[0],
            self.x_int[1],
            self.pts[2].flatten(),
            mat_eq,
            22,
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

        ker_loc.kernel_pi1_3(
            [self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]],
            self.n_quad[2],
            [self.n_int[0], self.n_int[1], self.n_his[2]],
            [self.n_int_locbf_N[0], self.n_int_locbf_D[1], self.n_his_locbf_D[2]],
            self.int_global_N[0],
            self.int_global_D[1],
            self.his_global_D[2],
            self.int_loccof_N[0],
            self.int_loccof_D[1],
            self.his_loccof_D[2],
            self.coeff_i[0],
            self.coeff_i[1],
            self.coeff_h[2],
            self.coeffi_indices[0],
            self.coeffi_indices[1],
            self.coeffh_indices[2],
            self.basisN_int[0],
            self.basisD_int[1],
            self.basisD_his[2],
            self.x_int_indices[0],
            self.x_int_indices[1],
            self.x_his_indices[2],
            self.wts[2],
            T31,
            mat_eq.reshape(n_unique3[0], n_unique3[1], self.pts[2][:, 0].size, self.pts[2][0, :].size),
        )

        ker_eva.kernel_eva(
            self.x_int[0],
            self.x_int[1],
            self.pts[2].flatten(),
            mat_eq,
            21,
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

        ker_loc.kernel_pi1_3(
            [self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]],
            self.n_quad[2],
            [self.n_int[0], self.n_int[1], self.n_his[2]],
            [self.n_int_locbf_D[0], self.n_int_locbf_N[1], self.n_his_locbf_D[2]],
            self.int_global_D[0],
            self.int_global_N[1],
            self.his_global_D[2],
            self.int_loccof_D[0],
            self.int_loccof_N[1],
            self.his_loccof_D[2],
            self.coeff_i[0],
            self.coeff_i[1],
            self.coeff_h[2],
            self.coeffi_indices[0],
            self.coeffi_indices[1],
            self.coeffh_indices[2],
            self.basisD_int[0],
            self.basisN_int[1],
            self.basisD_his[2],
            self.x_int_indices[0],
            self.x_int_indices[1],
            self.x_his_indices[2],
            self.wts[2],
            T32,
            mat_eq.reshape(n_unique3[0], n_unique3[1], self.pts[2][:, 0].size, self.pts[2][0, :].size),
        )

        # ============== conversion to sparse matrices (1 - component) ==============
        indices = xp.indices(
            (
                self.NbaseD[0],
                self.NbaseN[1],
                self.NbaseD[2],
                self.n_his_nvcof_D[0],
                self.n_int_nvcof_N[1],
                self.n_int_nvcof_D[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseD[2] * indices[0] + self.NbaseD[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_D[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseN[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        T12 = spa.csc_matrix(
            (T12.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseD[0] * self.NbaseN[1] * self.NbaseD[2], self.NbaseD[0] * self.NbaseN[1] * self.NbaseN[2]),
        )
        T12.eliminate_zeros()

        indices = xp.indices(
            (
                self.NbaseD[0],
                self.NbaseD[1],
                self.NbaseN[2],
                self.n_his_nvcof_D[0],
                self.n_int_nvcof_D[1],
                self.n_int_nvcof_N[2],
            ),
        )
        row = self.NbaseD[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_D[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseN[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        T13 = spa.csc_matrix(
            (T13.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseD[0] * self.NbaseD[1] * self.NbaseN[2], self.NbaseD[0] * self.NbaseN[1] * self.NbaseN[2]),
        )
        T13.eliminate_zeros()

        # ============== conversion to sparse matrices (2 - component) ==============
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseD[1],
                self.NbaseD[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_D[1],
                self.n_int_nvcof_D[2],
            ),
        )
        row = self.NbaseD[1] * self.NbaseD[2] * indices[0] + self.NbaseD[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_D[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseD[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        T21 = spa.csc_matrix(
            (T21.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseD[1] * self.NbaseD[2], self.NbaseN[0] * self.NbaseD[1] * self.NbaseN[2]),
        )
        T21.eliminate_zeros()

        indices = xp.indices(
            (
                self.NbaseD[0],
                self.NbaseD[1],
                self.NbaseN[2],
                self.n_int_nvcof_D[0],
                self.n_his_nvcof_D[1],
                self.n_int_nvcof_N[2],
            ),
        )
        row = self.NbaseD[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_D[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseD[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        T23 = spa.csc_matrix(
            (T23.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseD[0] * self.NbaseD[1] * self.NbaseN[2], self.NbaseN[0] * self.NbaseD[1] * self.NbaseN[2]),
        )
        T23.eliminate_zeros()

        # ============== conversion to sparse matrices (3 - component) ==============
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseD[1],
                self.NbaseD[2],
                self.n_int_nvcof_N[0],
                self.n_int_nvcof_D[1],
                self.n_his_nvcof_D[2],
            ),
        )
        row = self.NbaseD[1] * self.NbaseD[2] * indices[0] + self.NbaseD[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_D[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseN[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        T31 = spa.csc_matrix(
            (T31.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseD[1] * self.NbaseD[2], self.NbaseN[0] * self.NbaseN[1] * self.NbaseD[2]),
        )
        T31.eliminate_zeros()

        indices = xp.indices(
            (
                self.NbaseD[0],
                self.NbaseN[1],
                self.NbaseD[2],
                self.n_int_nvcof_D[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_D[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseD[2] * indices[0] + self.NbaseD[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_D[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseN[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        T32 = spa.csc_matrix(
            (T32.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseD[0] * self.NbaseN[1] * self.NbaseD[2], self.NbaseN[0] * self.NbaseN[1] * self.NbaseD[2]),
        )
        T32.eliminate_zeros()

        self.TAU = spa.bmat([[None, -T12.T, T13.T], [T21.T, None, -T23.T], [-T31.T, T32.T, None]], format="csc")

    # ========================================================================
    def projection_S_0form(self, domain):
        """
        Computes the sparse matrix of the expression pi_2(p3_eq * lambda^0) with the output (coefficients, basis_fun of lambda^0).

        The following blocks need to be computed:

        1 - component [int, his, his] : (N, N, N)*p3_eq, None             , None
        2 - component [his, int, his] : None             , (N, N, N)*p3_eq, None
        3 - component [his, his, int] : None             , None             , (N, N, N)*p3_eq

        An analytical mapping is called from struphy.geometry.mappings_analytical.

        Parameters
        ----------
        domain : domain
            domain object created with struphy.geometry.domain that defines the geometry

        Returns
        -------
        S : sparse matrix in csc-format
            the projection of each basis function in V0 on V2 weighted with p3_eq
        """

        # non-vanishing coefficients
        S11 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
            dtype=float,
        )
        S22 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
            dtype=float,
        )
        S33 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )

        # size of interpolation/quadrature points of the 3 components
        n_unique1 = [self.x_int[0].size, self.pts[1].flatten().size, self.pts[2].flatten().size]
        n_unique2 = [self.pts[0].flatten().size, self.x_int[1].size, self.pts[2].flatten().size]
        n_unique3 = [self.pts[0].flatten().size, self.pts[1].flatten().size, self.x_int[2].size]

        # ========= assembly of 1 - component (pi2_1 : int, his, his) ============
        mat_eq = xp.empty((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float)

        ker_eva.kernel_eva(
            self.x_int[0],
            self.pts[1].flatten(),
            self.pts[2].flatten(),
            mat_eq,
            31,
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

        ker_loc.kernel_pi2_1(
            [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]],
            [self.n_quad[1], self.n_quad[2]],
            [self.n_int[0], self.n_his[1], self.n_his[2]],
            [self.n_int_locbf_N[0], self.n_his_locbf_N[1], self.n_his_locbf_N[2]],
            self.int_global_N[0],
            self.his_global_N[1],
            self.his_global_N[2],
            self.int_loccof_N[0],
            self.his_loccof_N[1],
            self.his_loccof_N[2],
            self.coeff_i[0],
            self.coeff_h[1],
            self.coeff_h[2],
            self.coeffi_indices[0],
            self.coeffh_indices[1],
            self.coeffh_indices[2],
            self.basisN_int[0],
            self.basisN_his[1],
            self.basisN_his[2],
            self.x_int_indices[0],
            self.x_his_indices[1],
            self.x_his_indices[2],
            self.wts[1],
            self.wts[2],
            S11,
            mat_eq.reshape(
                n_unique1[0],
                self.pts[1][:, 0].size,
                self.pts[1][0, :].size,
                self.pts[2][:, 0].size,
                self.pts[2][0, :].size,
            ),
        )

        # ========= assembly of 2 - component (pi2_2 : his, int, his) ============
        mat_eq = xp.empty((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.x_int[1],
            self.pts[2].flatten(),
            mat_eq,
            31,
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

        ker_loc.kernel_pi2_2(
            [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]],
            [self.n_quad[0], self.n_quad[2]],
            [self.n_his[0], self.n_int[1], self.n_his[2]],
            [self.n_his_locbf_N[0], self.n_int_locbf_N[1], self.n_his_locbf_N[2]],
            self.his_global_N[0],
            self.int_global_N[1],
            self.his_global_N[2],
            self.his_loccof_N[0],
            self.int_loccof_N[1],
            self.his_loccof_N[2],
            self.coeff_h[0],
            self.coeff_i[1],
            self.coeff_h[2],
            self.coeffh_indices[0],
            self.coeffi_indices[1],
            self.coeffh_indices[2],
            self.basisN_his[0],
            self.basisN_int[1],
            self.basisN_his[2],
            self.x_his_indices[0],
            self.x_int_indices[1],
            self.x_his_indices[2],
            self.wts[0],
            self.wts[2],
            S22,
            mat_eq.reshape(
                self.pts[0][:, 0].size,
                self.pts[0][0, :].size,
                n_unique2[1],
                self.pts[2][:, 0].size,
                self.pts[2][0, :].size,
            ),
        )

        # ========= assembly of 3 - component (pi2_3 : his, his, int) ============
        mat_eq = xp.empty((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.pts[1].flatten(),
            self.x_int[2],
            mat_eq,
            31,
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

        ker_loc.kernel_pi2_3(
            [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]],
            [self.n_quad[0], self.n_quad[1]],
            [self.n_his[0], self.n_his[1], self.n_int[2]],
            [self.n_his_locbf_N[0], self.n_his_locbf_N[1], self.n_int_locbf_N[2]],
            self.his_global_N[0],
            self.his_global_N[1],
            self.int_global_N[2],
            self.his_loccof_N[0],
            self.his_loccof_N[1],
            self.int_loccof_N[2],
            self.coeff_h[0],
            self.coeff_h[1],
            self.coeff_i[2],
            self.coeffh_indices[0],
            self.coeffh_indices[1],
            self.coeffi_indices[2],
            self.basisN_his[0],
            self.basisN_his[1],
            self.basisN_int[2],
            self.x_his_indices[0],
            self.x_his_indices[1],
            self.x_int_indices[2],
            self.wts[0],
            self.wts[1],
            S33,
            mat_eq.reshape(
                self.pts[0][:, 0].size,
                self.pts[0][0, :].size,
                self.pts[1][:, 0].size,
                self.pts[1][0, :].size,
                n_unique3[2],
            ),
        )

        # ========= conversion to sparse matrices (1 - component) =================
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseD[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        S11 = spa.csc_matrix(
            (S11.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseN[0] * self.NbaseD[1] * self.NbaseD[2]),
        )
        S11.eliminate_zeros()

        # ========= conversion to sparse matrices (2 - component) =================
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseN[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        S22 = spa.csc_matrix(
            (S22.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseD[0] * self.NbaseN[1] * self.NbaseD[2]),
        )
        S22.eliminate_zeros()

        # ========= conversion to sparse matrices (3 - component) =================
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseD[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        S33 = spa.csc_matrix(
            (S33.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseD[0] * self.NbaseD[1] * self.NbaseN[2]),
        )
        S33.eliminate_zeros()

        self.S = spa.bmat([[S11.T, None, None], [None, S22.T, None], [None, None, S33.T]], format="csc")

    # ========================================================================
    def projection_S_2form(self, domain):
        """
        Computes the sparse matrix of the expression pi_2(p3_eq * lambda^2) with the output (coefficients, basis_fun of lambda^2).

        The following blocks need to be computed:

        1 - component [int, his, his] : (N, D, D)*p3_eq, None           , None
        2 - component [his, int, his] : None           , (D, N, D)*p3_eq, None
        3 - component [his, his, int] : None           , None           , (D, D, N)*p3_eq

        An analytical mapping is called from struphy.geometry.mappings_analytical.

        Parameters
        ----------
        domain : domain
            domain object created with struphy.geometry.domain that defines the geometry

        Returns
        -------
        S : sparse matrix in csc-format
            the projection of each basis function in V2 on V2 weighted with rho3_eq
        """

        # non-vanishing coefficients
        S11 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseD[1],
                self.NbaseD[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_D[1],
                self.n_his_nvcof_D[2],
            ),
            dtype=float,
        )
        S22 = xp.empty(
            (
                self.NbaseD[0],
                self.NbaseN[1],
                self.NbaseD[2],
                self.n_his_nvcof_D[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_D[2],
            ),
            dtype=float,
        )
        S33 = xp.empty(
            (
                self.NbaseD[0],
                self.NbaseD[1],
                self.NbaseN[2],
                self.n_his_nvcof_D[0],
                self.n_his_nvcof_D[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )

        # size of interpolation/quadrature points of the 3 components
        n_unique1 = [self.x_int[0].size, self.pts[1].flatten().size, self.pts[2].flatten().size]
        n_unique2 = [self.pts[0].flatten().size, self.x_int[1].size, self.pts[2].flatten().size]
        n_unique3 = [self.pts[0].flatten().size, self.pts[1].flatten().size, self.x_int[2].size]

        # ========= assembly of 1 - component (pi2_1 : int, his, his) ============
        mat_eq = xp.empty((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float)

        ker_eva.kernel_eva(
            self.x_int[0],
            self.pts[1].flatten(),
            self.pts[2].flatten(),
            mat_eq,
            31,
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

        ker_loc.kernel_pi2_1(
            [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]],
            [self.n_quad[1], self.n_quad[2]],
            [self.n_int[0], self.n_his[1], self.n_his[2]],
            [self.n_int_locbf_N[0], self.n_his_locbf_D[1], self.n_his_locbf_D[2]],
            self.int_global_N[0],
            self.his_global_D[1],
            self.his_global_D[2],
            self.int_loccof_N[0],
            self.his_loccof_D[1],
            self.his_loccof_D[2],
            self.coeff_i[0],
            self.coeff_h[1],
            self.coeff_h[2],
            self.coeffi_indices[0],
            self.coeffh_indices[1],
            self.coeffh_indices[2],
            self.basisN_int[0],
            self.basisD_his[1],
            self.basisD_his[2],
            self.x_int_indices[0],
            self.x_his_indices[1],
            self.x_his_indices[2],
            self.wts[1],
            self.wts[2],
            S11,
            mat_eq.reshape(
                n_unique1[0],
                self.pts[1][:, 0].size,
                self.pts[1][0, :].size,
                self.pts[2][:, 0].size,
                self.pts[2][0, :].size,
            ),
        )

        # ========= assembly of 2 - component (pi2_2 : his, int, his) ============
        mat_eq = xp.empty((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.x_int[1],
            self.pts[2].flatten(),
            mat_eq,
            31,
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

        ker_loc.kernel_pi2_2(
            [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]],
            [self.n_quad[0], self.n_quad[2]],
            [self.n_his[0], self.n_int[1], self.n_his[2]],
            [self.n_his_locbf_D[0], self.n_int_locbf_N[1], self.n_his_locbf_D[2]],
            self.his_global_D[0],
            self.int_global_N[1],
            self.his_global_D[2],
            self.his_loccof_D[0],
            self.int_loccof_N[1],
            self.his_loccof_D[2],
            self.coeff_h[0],
            self.coeff_i[1],
            self.coeff_h[2],
            self.coeffh_indices[0],
            self.coeffi_indices[1],
            self.coeffh_indices[2],
            self.basisD_his[0],
            self.basisN_int[1],
            self.basisD_his[2],
            self.x_his_indices[0],
            self.x_int_indices[1],
            self.x_his_indices[2],
            self.wts[0],
            self.wts[2],
            S22,
            mat_eq.reshape(
                self.pts[0][:, 0].size,
                self.pts[0][0, :].size,
                n_unique2[1],
                self.pts[2][:, 0].size,
                self.pts[2][0, :].size,
            ),
        )

        # ========= assembly of 3 - component (pi2_3 : his, his, int) ============
        mat_eq = xp.empty((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.pts[1].flatten(),
            self.x_int[2],
            mat_eq,
            31,
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

        ker_loc.kernel_pi2_3(
            [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]],
            [self.n_quad[0], self.n_quad[1]],
            [self.n_his[0], self.n_his[1], self.n_int[2]],
            [self.n_his_locbf_D[0], self.n_his_locbf_D[1], self.n_int_locbf_N[2]],
            self.his_global_D[0],
            self.his_global_D[1],
            self.int_global_N[2],
            self.his_loccof_D[0],
            self.his_loccof_D[1],
            self.int_loccof_N[2],
            self.coeff_h[0],
            self.coeff_h[1],
            self.coeff_i[2],
            self.coeffh_indices[0],
            self.coeffh_indices[1],
            self.coeffi_indices[2],
            self.basisD_his[0],
            self.basisD_his[1],
            self.basisN_int[2],
            self.x_his_indices[0],
            self.x_his_indices[1],
            self.x_int_indices[2],
            self.wts[0],
            self.wts[1],
            S33,
            mat_eq.reshape(
                self.pts[0][:, 0].size,
                self.pts[0][0, :].size,
                self.pts[1][:, 0].size,
                self.pts[1][0, :].size,
                n_unique3[2],
            ),
        )

        # ========= conversion to sparse matrices (1 - component) =================
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseD[1],
                self.NbaseD[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_D[1],
                self.n_his_nvcof_D[2],
            ),
        )
        row = self.NbaseD[1] * self.NbaseD[2] * indices[0] + self.NbaseD[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseD[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        S11 = spa.csc_matrix(
            (S11.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseD[1] * self.NbaseD[2], self.NbaseN[0] * self.NbaseD[1] * self.NbaseD[2]),
        )
        S11.eliminate_zeros()

        # ========= conversion to sparse matrices (2 - component) =================
        indices = xp.indices(
            (
                self.NbaseD[0],
                self.NbaseN[1],
                self.NbaseD[2],
                self.n_his_nvcof_D[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_D[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseD[2] * indices[0] + self.NbaseD[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseN[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        S22 = spa.csc_matrix(
            (S22.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseD[0] * self.NbaseN[1] * self.NbaseD[2], self.NbaseD[0] * self.NbaseN[1] * self.NbaseD[2]),
        )
        S22.eliminate_zeros()

        # ========= conversion to sparse matrices (3 - component) =================
        indices = xp.indices(
            (
                self.NbaseD[0],
                self.NbaseD[1],
                self.NbaseN[2],
                self.n_his_nvcof_D[0],
                self.n_his_nvcof_D[1],
                self.n_int_nvcof_N[2],
            ),
        )
        row = self.NbaseD[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseD[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        S33 = spa.csc_matrix(
            (S33.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseD[0] * self.NbaseD[1] * self.NbaseN[2], self.NbaseD[0] * self.NbaseD[1] * self.NbaseN[2]),
        )
        S33.eliminate_zeros()

        self.S = spa.bmat([[S11.T, None, None], [None, S22.T, None], [None, None, S33.T]], format="csc")

    # ========================================================================
    def projection_K_3form(self, domain):
        """
        Computes the sparse matrix of the expression pi_3(p0_eq * lambda^3) with the output (coefficients, basis_fun of lambda^2).

        The following block needs to be computed:

        [his, his, his] : (D, D, D)*p0_eq

        An analytical mapping is called from struphy.geometry.mappings_analytical.

        Parameters
        ----------
        domain : domain
            domain object created with struphy.geometry.domain that defines the geometry

        Returns
        -------
        K : sparse matrix in csc-format
            the projection of each basis function in V3 on V3 weighted with p0_eq
        """

        # non-vanishing coefficients
        K = xp.zeros(
            (
                self.NbaseD[0],
                self.NbaseD[1],
                self.NbaseD[2],
                self.n_his_nvcof_D[0],
                self.n_his_nvcof_D[1],
                self.n_his_nvcof_D[2],
            ),
            dtype=float,
        )

        # evaluation of equilibrium pressure at interpolation points
        n_unique = [self.pts[0].flatten().size, self.pts[1].flatten().size, self.pts[2].flatten().size]

        mat_eq = xp.zeros((n_unique[0], n_unique[1], n_unique[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.pts[1].flatten(),
            self.pts[2].flatten(),
            mat_eq,
            41,
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

        # assembly of K
        ker_loc.kernel_pi3(
            self.NbaseD,
            self.n_quad,
            self.n_his,
            self.n_his_locbf_D,
            self.his_global_D[0],
            self.his_global_D[1],
            self.his_global_D[2],
            self.his_loccof_D[0],
            self.his_loccof_D[1],
            self.his_loccof_D[2],
            self.coeff_h[0],
            self.coeff_h[1],
            self.coeff_h[2],
            self.coeffh_indices[0],
            self.coeffh_indices[1],
            self.coeffh_indices[2],
            self.basisD_his[0],
            self.basisD_his[1],
            self.basisD_his[2],
            self.x_his_indices[0],
            self.x_his_indices[1],
            self.x_his_indices[2],
            self.wts[0],
            self.wts[1],
            self.wts[2],
            K,
            mat_eq.reshape(
                self.pts[0][:, 0].size,
                self.pts[0][0, :].size,
                self.pts[1][:, 0].size,
                self.pts[1][0, :].size,
                self.pts[2][:, 0].size,
                self.pts[2][0, :].size,
            ),
        )

        # conversion to sparse matrix
        indices = xp.indices(
            (
                self.NbaseD[0],
                self.NbaseD[1],
                self.NbaseD[2],
                self.n_his_nvcof_D[0],
                self.n_his_nvcof_D[1],
                self.n_his_nvcof_D[2],
            ),
        )

        # row indices
        row = self.NbaseD[1] * self.NbaseD[2] * indices[0] + self.NbaseD[2] * indices[1] + indices[2]

        # column indices
        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseD[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        # create sparse matrix
        K = spa.csc_matrix(
            (K.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseD[0] * self.NbaseD[1] * self.NbaseD[2], self.NbaseD[0] * self.NbaseD[1] * self.NbaseD[2]),
        )
        K.eliminate_zeros()

        self.K = K.T

    # ========================================================================
    def projection_N_0form(self, domain):
        """
        Computes the sparse matrix of the expression pi_2(g_sqrt * lambda^0) with the output (coefficients, basis_fun of lambda^0).

        The following blocks need to be computed:

        1 - component [int, his, his] : (N, N, N)*g_sqrt, None             , None
        2 - component [his, int, his] : None             , (N, N, N)*g_sqrt, None
        3 - component [his, his, int] : None             , None             , (N, N, N)*g_sqrt

        An analytical mapping is called from struphy.geometry.mappings_analytical.

        Parameters
        ----------
        domain : domain
            domain object created with struphy.geometry.domain that defines the geometry

        Returns
        -------
        N : sparse matrix in csc-format
            the projection of each basis function in V0 on V2 weighted with g_sqrt
        """

        # non-vanishing coefficients
        N11 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
            dtype=float,
        )
        N22 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
            dtype=float,
        )
        N33 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )

        # size of interpolation/quadrature points of the 3 components
        n_unique1 = [self.x_int[0].size, self.pts[1].flatten().size, self.pts[2].flatten().size]
        n_unique2 = [self.pts[0].flatten().size, self.x_int[1].size, self.pts[2].flatten().size]
        n_unique3 = [self.pts[0].flatten().size, self.pts[1].flatten().size, self.x_int[2].size]

        # ========= assembly of 1 - component (pi2_1 : int, his, his) ============
        mat_eq = xp.empty((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float)

        ker_eva.kernel_eva(
            self.x_int[0],
            self.pts[1].flatten(),
            self.pts[2].flatten(),
            mat_eq,
            51,
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

        ker_loc.kernel_pi2_1(
            [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]],
            [self.n_quad[1], self.n_quad[2]],
            [self.n_int[0], self.n_his[1], self.n_his[2]],
            [self.n_int_locbf_N[0], self.n_his_locbf_N[1], self.n_his_locbf_N[2]],
            self.int_global_N[0],
            self.his_global_N[1],
            self.his_global_N[2],
            self.int_loccof_N[0],
            self.his_loccof_N[1],
            self.his_loccof_N[2],
            self.coeff_i[0],
            self.coeff_h[1],
            self.coeff_h[2],
            self.coeffi_indices[0],
            self.coeffh_indices[1],
            self.coeffh_indices[2],
            self.basisN_int[0],
            self.basisN_his[1],
            self.basisN_his[2],
            self.x_int_indices[0],
            self.x_his_indices[1],
            self.x_his_indices[2],
            self.wts[1],
            self.wts[2],
            N11,
            mat_eq.reshape(
                n_unique1[0],
                self.pts[1][:, 0].size,
                self.pts[1][0, :].size,
                self.pts[2][:, 0].size,
                self.pts[2][0, :].size,
            ),
        )

        # ========= assembly of 2 - component (pi2_2 : his, int, his) ============
        mat_eq = xp.empty((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.x_int[1],
            self.pts[2].flatten(),
            mat_eq,
            51,
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

        ker_loc.kernel_pi2_2(
            [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]],
            [self.n_quad[0], self.n_quad[2]],
            [self.n_his[0], self.n_int[1], self.n_his[2]],
            [self.n_his_locbf_N[0], self.n_int_locbf_N[1], self.n_his_locbf_N[2]],
            self.his_global_N[0],
            self.int_global_N[1],
            self.his_global_N[2],
            self.his_loccof_N[0],
            self.int_loccof_N[1],
            self.his_loccof_N[2],
            self.coeff_h[0],
            self.coeff_i[1],
            self.coeff_h[2],
            self.coeffh_indices[0],
            self.coeffi_indices[1],
            self.coeffh_indices[2],
            self.basisN_his[0],
            self.basisN_int[1],
            self.basisN_his[2],
            self.x_his_indices[0],
            self.x_int_indices[1],
            self.x_his_indices[2],
            self.wts[0],
            self.wts[2],
            N22,
            mat_eq.reshape(
                self.pts[0][:, 0].size,
                self.pts[0][0, :].size,
                n_unique2[1],
                self.pts[2][:, 0].size,
                self.pts[2][0, :].size,
            ),
        )

        # ========= assembly of 3 - component (pi2_3 : his, his, int) ============
        mat_eq = xp.empty((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.pts[1].flatten(),
            self.x_int[2],
            mat_eq,
            51,
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

        ker_loc.kernel_pi2_3(
            [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]],
            [self.n_quad[0], self.n_quad[1]],
            [self.n_his[0], self.n_his[1], self.n_int[2]],
            [self.n_his_locbf_N[0], self.n_his_locbf_N[1], self.n_int_locbf_N[2]],
            self.his_global_N[0],
            self.his_global_N[1],
            self.int_global_N[2],
            self.his_loccof_N[0],
            self.his_loccof_N[1],
            self.int_loccof_N[2],
            self.coeff_h[0],
            self.coeff_h[1],
            self.coeff_i[2],
            self.coeffh_indices[0],
            self.coeffh_indices[1],
            self.coeffi_indices[2],
            self.basisN_his[0],
            self.basisN_his[1],
            self.basisN_int[2],
            self.x_his_indices[0],
            self.x_his_indices[1],
            self.x_int_indices[2],
            self.wts[0],
            self.wts[1],
            N33,
            mat_eq.reshape(
                self.pts[0][:, 0].size,
                self.pts[0][0, :].size,
                self.pts[1][:, 0].size,
                self.pts[1][0, :].size,
                n_unique3[2],
            ),
        )

        # ========= conversion to sparse matrices (1 - component) =================
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseD[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        N11 = spa.csc_matrix(
            (N11.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseN[0] * self.NbaseD[1] * self.NbaseD[2]),
        )
        N11.eliminate_zeros()

        # ========= conversion to sparse matrices (2 - component) =================
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_N[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseN[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        N22 = spa.csc_matrix(
            (N22.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseD[0] * self.NbaseN[1] * self.NbaseD[2]),
        )
        N22.eliminate_zeros()

        # ========= conversion to sparse matrices (3 - component) =================
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseN[1],
                self.NbaseN[2],
                self.n_his_nvcof_N[0],
                self.n_his_nvcof_N[1],
                self.n_int_nvcof_N[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_N[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.his_shift_N[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseD[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        N33 = spa.csc_matrix(
            (N33.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseN[1] * self.NbaseN[2], self.NbaseD[0] * self.NbaseD[1] * self.NbaseN[2]),
        )
        N33.eliminate_zeros()

        self.N = spa.bmat([[N11.T, None, None], [None, N22.T, None], [None, None, N33.T]], format="csc")

    # ========================================================================
    def projection_N_2form(self, domain):
        """
        Computes the sparse matrix of the expression pi_2(g_sqrt * lambda^2) with the output (coefficients, basis_fun of lambda^2).

        The following blocks need to be computed:

        1 - component [int, his, his] : (N, D, D)*g_sqrt, None            , None
        2 - component [his, int, his] : None            , (D, N, D)*g_sqrt, None
        3 - component [his, his, int] : None            , None            , (D, D, N)*g_sqrt

        An analytical mapping is called from struphy.geometry.mappings_analytical.

        Parameters
        ----------
        domain : domain
            domain object created with struphy.geometry.domain that defines the geometry

        Returns
        -------
        N : sparse matrix in csc-format
            the projection of each basis function in V2 on V2 weighted with rho3_eq
        """

        # non-vanishing coefficients
        N11 = xp.empty(
            (
                self.NbaseN[0],
                self.NbaseD[1],
                self.NbaseD[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_D[1],
                self.n_his_nvcof_D[2],
            ),
            dtype=float,
        )
        N22 = xp.empty(
            (
                self.NbaseD[0],
                self.NbaseN[1],
                self.NbaseD[2],
                self.n_his_nvcof_D[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_D[2],
            ),
            dtype=float,
        )
        N33 = xp.empty(
            (
                self.NbaseD[0],
                self.NbaseD[1],
                self.NbaseN[2],
                self.n_his_nvcof_D[0],
                self.n_his_nvcof_D[1],
                self.n_int_nvcof_N[2],
            ),
            dtype=float,
        )

        # size of interpolation/quadrature points of the 3 components
        n_unique1 = [self.x_int[0].size, self.pts[1].flatten().size, self.pts[2].flatten().size]
        n_unique2 = [self.pts[0].flatten().size, self.x_int[1].size, self.pts[2].flatten().size]
        n_unique3 = [self.pts[0].flatten().size, self.pts[1].flatten().size, self.x_int[2].size]

        # ========= assembly of 1 - component (pi2_1 : int, his, his) ============
        mat_eq = xp.empty((n_unique1[0], n_unique1[1], n_unique1[2]), dtype=float)

        ker_eva.kernel_eva(
            self.x_int[0],
            self.pts[1].flatten(),
            self.pts[2].flatten(),
            mat_eq,
            51,
            kind_map=kind_map,
            params_map=params_map,
        )

        ker_loc.kernel_pi2_1(
            [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]],
            [self.n_quad[1], self.n_quad[2]],
            [self.n_int[0], self.n_his[1], self.n_his[2]],
            [self.n_int_locbf_N[0], self.n_his_locbf_D[1], self.n_his_locbf_D[2]],
            self.int_global_N[0],
            self.his_global_D[1],
            self.his_global_D[2],
            self.int_loccof_N[0],
            self.his_loccof_D[1],
            self.his_loccof_D[2],
            self.coeff_i[0],
            self.coeff_h[1],
            self.coeff_h[2],
            self.coeffi_indices[0],
            self.coeffh_indices[1],
            self.coeffh_indices[2],
            self.basisN_int[0],
            self.basisD_his[1],
            self.basisD_his[2],
            self.x_int_indices[0],
            self.x_his_indices[1],
            self.x_his_indices[2],
            self.wts[1],
            self.wts[2],
            N11,
            mat_eq.reshape(
                n_unique1[0],
                self.pts[1][:, 0].size,
                self.pts[1][0, :].size,
                self.pts[2][:, 0].size,
                self.pts[2][0, :].size,
            ),
        )

        # ========= assembly of 2 - component (pi2_2 : his, int, his) ============
        mat_eq = xp.empty((n_unique2[0], n_unique2[1], n_unique2[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.x_int[1],
            self.pts[2].flatten(),
            mat_eq,
            51,
            kind_map=kind_map,
            params_map=params_map,
        )

        ker_loc.kernel_pi2_2(
            [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]],
            [self.n_quad[0], self.n_quad[2]],
            [self.n_his[0], self.n_int[1], self.n_his[2]],
            [self.n_his_locbf_D[0], self.n_int_locbf_N[1], self.n_his_locbf_D[2]],
            self.his_global_D[0],
            self.int_global_N[1],
            self.his_global_D[2],
            self.his_loccof_D[0],
            self.int_loccof_N[1],
            self.his_loccof_D[2],
            self.coeff_h[0],
            self.coeff_i[1],
            self.coeff_h[2],
            self.coeffh_indices[0],
            self.coeffi_indices[1],
            self.coeffh_indices[2],
            self.basisD_his[0],
            self.basisN_int[1],
            self.basisD_his[2],
            self.x_his_indices[0],
            self.x_int_indices[1],
            self.x_his_indices[2],
            self.wts[0],
            self.wts[2],
            N22,
            mat_eq.reshape(
                self.pts[0][:, 0].size,
                self.pts[0][0, :].size,
                n_unique2[1],
                self.pts[2][:, 0].size,
                self.pts[2][0, :].size,
            ),
        )

        # ========= assembly of 3 - component (pi2_3 : his, his, int) ============
        mat_eq = xp.empty((n_unique3[0], n_unique3[1], n_unique3[2]), dtype=float)

        ker_eva.kernel_eva(
            self.pts[0].flatten(),
            self.pts[1].flatten(),
            self.x_int[2],
            mat_eq,
            51,
            kind_map=kind_map,
            params_map=params_map,
        )

        ker_loc.kernel_pi2_3(
            [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]],
            [self.n_quad[0], self.n_quad[1]],
            [self.n_his[0], self.n_his[1], self.n_int[2]],
            [self.n_his_locbf_D[0], self.n_his_locbf_D[1], self.n_int_locbf_N[2]],
            self.his_global_D[0],
            self.his_global_D[1],
            self.int_global_N[2],
            self.his_loccof_D[0],
            self.his_loccof_D[1],
            self.int_loccof_N[2],
            self.coeff_h[0],
            self.coeff_h[1],
            self.coeff_i[2],
            self.coeffh_indices[0],
            self.coeffh_indices[1],
            self.coeffi_indices[2],
            self.basisD_his[0],
            self.basisD_his[1],
            self.basisN_int[2],
            self.x_his_indices[0],
            self.x_his_indices[1],
            self.x_int_indices[2],
            self.wts[0],
            self.wts[1],
            N33,
            mat_eq.reshape(
                self.pts[0][:, 0].size,
                self.pts[0][0, :].size,
                self.pts[1][:, 0].size,
                self.pts[1][0, :].size,
                n_unique3[2],
            ),
        )

        # ========= conversion to sparse matrices (1 - component) =================
        indices = xp.indices(
            (
                self.NbaseN[0],
                self.NbaseD[1],
                self.NbaseD[2],
                self.n_int_nvcof_N[0],
                self.n_his_nvcof_D[1],
                self.n_his_nvcof_D[2],
            ),
        )
        row = self.NbaseD[1] * self.NbaseD[2] * indices[0] + self.NbaseD[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.int_shift_N[0][:, None, None, None, None, None]) % self.NbaseN[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseD[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        N11 = spa.csc_matrix(
            (N11.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseN[0] * self.NbaseD[1] * self.NbaseD[2], self.NbaseN[0] * self.NbaseD[1] * self.NbaseD[2]),
        )
        N11.eliminate_zeros()

        # ========= conversion to sparse matrices (2 - component) =================
        indices = xp.indices(
            (
                self.NbaseD[0],
                self.NbaseN[1],
                self.NbaseD[2],
                self.n_his_nvcof_D[0],
                self.n_int_nvcof_N[1],
                self.n_his_nvcof_D[2],
            ),
        )
        row = self.NbaseN[1] * self.NbaseD[2] * indices[0] + self.NbaseD[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.int_shift_N[1][None, :, None, None, None, None]) % self.NbaseN[1]
        col3 = (indices[5] + self.his_shift_D[2][None, None, :, None, None, None]) % self.NbaseD[2]

        col = self.NbaseN[1] * self.NbaseD[2] * col1 + self.NbaseD[2] * col2 + col3

        N22 = spa.csc_matrix(
            (N22.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseD[0] * self.NbaseN[1] * self.NbaseD[2], self.NbaseD[0] * self.NbaseN[1] * self.NbaseD[2]),
        )
        N22.eliminate_zeros()

        # ========= conversion to sparse matrices (3 - component) =================
        indices = xp.indices(
            (
                self.NbaseD[0],
                self.NbaseD[1],
                self.NbaseN[2],
                self.n_his_nvcof_D[0],
                self.n_his_nvcof_D[1],
                self.n_int_nvcof_N[2],
            ),
        )
        row = self.NbaseD[1] * self.NbaseN[2] * indices[0] + self.NbaseN[2] * indices[1] + indices[2]

        col1 = (indices[3] + self.his_shift_D[0][:, None, None, None, None, None]) % self.NbaseD[0]
        col2 = (indices[4] + self.his_shift_D[1][None, :, None, None, None, None]) % self.NbaseD[1]
        col3 = (indices[5] + self.int_shift_N[2][None, None, :, None, None, None]) % self.NbaseN[2]

        col = self.NbaseD[1] * self.NbaseN[2] * col1 + self.NbaseN[2] * col2 + col3

        N33 = spa.csc_matrix(
            (N33.flatten(), (row.flatten(), col.flatten())),
            shape=(self.NbaseD[0] * self.NbaseD[1] * self.NbaseN[2], self.NbaseD[0] * self.NbaseD[1] * self.NbaseN[2]),
        )
        N33.eliminate_zeros()

        self.N = spa.bmat([[N11.T, None, None], [None, N22.T, None], [None, None, N33.T]], format="csc")

    # =====================================
    def setOperators(self, gamma, dt, drop_tol_S6, fill_fac_S6):
        A = (1 / 2 * (self.W.T.dot(self.tensor_space.Mv) + self.tensor_space.Mv.dot(self.W))).tocsr()
        self.A = spa.linalg.LinearOperator(A.shape, matvec=lambda x: A.dot(x), rmatvec=lambda x: A.T.dot(x))

        L = (-self.tensor_space.DIV.dot(self.S) - (gamma - 1) * self.K.dot(self.tensor_space.DIV.dot(self.N))).tocsr()
        self.L = spa.linalg.LinearOperator(L.shape, matvec=lambda x: L.dot(x), rmatvec=lambda x: L.T.dot(x))

        S6 = (A - dt**2 / 4 * self.N.T.dot(self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(L)))).tocsr()
        self.S6 = spa.linalg.LinearOperator(S6.shape, matvec=lambda x: S6.dot(x), rmatvec=lambda x: S6.T.dot(x))

        S6_ILU = spa.linalg.spilu(S6.tocsc(), drop_tol=drop_tol_S6, fill_factor=fill_fac_S6)
        self.S6_PRE = spa.linalg.LinearOperator(self.S6.shape, lambda x: S6_ILU.solve(x))

    # =====================================
    def RHS6(self, u, p, b, dt):
        out = (
            self.A(u)
            + dt**2 / 4 * self.N.T.dot(self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(self.L(u))))
            + dt * self.N.T.dot(self.tensor_space.DIV.T.dot(self.tensor_space.M3.dot(p)))
        )

        return out


# ===========================================================
class term_curl_beq:
    """
    Computes the inner product of the term [nabla x (G * Beq)] x B2 = [nabla x (DF^T * B_eq_phys)] x B2 with each basis function in V2.

    Parameters
    ----------
    tensor_space : Tensor_spline_space
        tensor product B-spline space

    kind_map : int
        type of mapping

    params_map : list of doubles
        parameters for the mapping
    """

    def __init__(
        self,
        tensor_space,
        mapping,
        kind_map=None,
        params_map=None,
        tensor_space_F=None,
        cx=None,
        cy=None,
        cz=None,
    ):
        self.p = tensor_space.p  # spline degrees
        self.Nel = tensor_space.Nel  # number of elements
        self.NbaseN = tensor_space.NbaseN  # total number of basis functions (N)
        self.NbaseD = tensor_space.NbaseD  # total number of basis functions (D)

        self.n_quad = tensor_space.n_quad  # number of quadrature points per element
        self.wts = tensor_space.wts  # quadrature weights in format (element, local point)
        self.pts = tensor_space.pts  # quadrature points in format (element, local point)
        self.n_pts = tensor_space.n_pts  # total number of quadrature points

        # basis functions evaluated at quadrature points in format (element, local basis function, derivative, local point)
        self.basisN = tensor_space.basisN
        self.basisD = tensor_space.basisD

        # mapping
        self.mapping = mapping

        if mapping == 0:
            self.kind_map = kind_map
            self.params_map = params_map

        elif mapping == 1:
            self.T_F = tensor_space_F.T
            self.p_F = tensor_space_F.p
            self.NbaseN_F = tensor_space_F.NbaseN

            self.cx = cx
            self.cy = cy
            self.cz = cz

        # ============= evaluation of background magnetic field at quadrature points =========
        self.mat_curl_beq_1 = xp.empty(
            (self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]),
            dtype=float,
        )
        self.mat_curl_beq_2 = xp.empty(
            (self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]),
            dtype=float,
        )
        self.mat_curl_beq_3 = xp.empty(
            (self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]),
            dtype=float,
        )

        if mapping == 0:
            ker_eva.kernel_eva_quad(
                self.Nel,
                self.n_quad,
                self.pts[0],
                self.pts[1],
                self.pts[2],
                self.mat_curl_beq_1,
                61,
                kind_map,
                params_map,
            )
            ker_eva.kernel_eva_quad(
                self.Nel,
                self.n_quad,
                self.pts[0],
                self.pts[1],
                self.pts[2],
                self.mat_curl_beq_2,
                62,
                kind_map,
                params_map,
            )
            ker_eva.kernel_eva_quad(
                self.Nel,
                self.n_quad,
                self.pts[0],
                self.pts[1],
                self.pts[2],
                self.mat_curl_beq_3,
                63,
                kind_map,
                params_map,
            )
        elif mapping == 1:
            ker_eva.kernel_eva_quad(
                self.Nel,
                self.n_quad,
                self.pts[0],
                self.pts[1],
                self.pts[2],
                self.mat_curl_beq_1,
                61,
                self.T_F[0],
                self.T_F[1],
                self.T_F[2],
                self.p_F,
                self.NbaseN_F,
                cx,
                cy,
                cz,
            )
            ker_eva.kernel_eva_quad(
                self.Nel,
                self.n_quad,
                self.pts[0],
                self.pts[1],
                self.pts[2],
                self.mat_curl_beq_2,
                62,
                self.T_F[0],
                self.T_F[1],
                self.T_F[2],
                self.p_F,
                self.NbaseN_F,
                cx,
                cy,
                cz,
            )
            ker_eva.kernel_eva_quad(
                self.Nel,
                self.n_quad,
                self.pts[0],
                self.pts[1],
                self.pts[2],
                self.mat_curl_beq_3,
                63,
                self.T_F[0],
                self.T_F[1],
                self.T_F[2],
                self.p_F,
                self.NbaseN_F,
                cx,
                cy,
                cz,
            )

        # ====================== perturbed magnetic field at quadrature points ==========
        self.B1 = xp.empty(
            (self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]),
            dtype=float,
        )
        self.B2 = xp.empty(
            (self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]),
            dtype=float,
        )
        self.B3 = xp.empty(
            (self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]),
            dtype=float,
        )

        # ========================== inner products =====================================
        self.F1 = xp.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
        self.F2 = xp.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
        self.F3 = xp.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)

    # ============================================================
    def inner_curl_beq(self, b1, b2, b3):
        """
        Computes the inner product of the term [nabla x (G * Beq)] x B2 = [nabla x (DF^T * B_eq_phys)] x B2 with each basis function in V2.
        """

        # evaluation of perturbed magnetic field at quadrature points
        ker_loc_3d.kernel_evaluate_2form(
            self.Nel,
            self.p,
            [0, 1, 1],
            self.n_quad,
            b1,
            [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]],
            self.basisN[0],
            self.basisD[1],
            self.basisD[2],
            self.B1,
        )
        ker_loc_3d.kernel_evaluate_2form(
            self.Nel,
            self.p,
            [1, 0, 1],
            self.n_quad,
            b2,
            [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]],
            self.basisD[0],
            self.basisN[1],
            self.basisD[2],
            self.B2,
        )
        ker_loc_3d.kernel_evaluate_2form(
            self.Nel,
            self.p,
            [1, 1, 0],
            self.n_quad,
            b3,
            [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]],
            self.basisD[0],
            self.basisD[1],
            self.basisN[2],
            self.B3,
        )

        # assembly of F (1 - component)
        ker_loc_3d.kernel_inner_2(
            self.Nel[0],
            self.Nel[1],
            self.Nel[2],
            self.p[0],
            self.p[1],
            self.p[2],
            self.n_quad[0],
            self.n_quad[1],
            self.n_quad[2],
            0,
            0,
            0,
            self.wts[0],
            self.wts[1],
            self.wts[2],
            self.basisN[0],
            self.basisN[1],
            self.basisN[2],
            self.NbaseN[0],
            self.NbaseN[1],
            self.NbaseN[2],
            self.F1,
            self.mat_curl_beq_2 * self.B3 - self.mat_curl_beq_3 * self.B2,
        )
        # ker_loc_3d.kernel_inner_2(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisN[1], self.basisN[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.F1, self.mat_curl_beq_1)

        # assembly of F (2 - component)
        ker_loc_3d.kernel_inner_2(
            self.Nel[0],
            self.Nel[1],
            self.Nel[2],
            self.p[0],
            self.p[1],
            self.p[2],
            self.n_quad[0],
            self.n_quad[1],
            self.n_quad[2],
            0,
            0,
            0,
            self.wts[0],
            self.wts[1],
            self.wts[2],
            self.basisN[0],
            self.basisN[1],
            self.basisN[2],
            self.NbaseN[0],
            self.NbaseN[1],
            self.NbaseN[2],
            self.F2,
            self.mat_curl_beq_3 * self.B1 - self.mat_curl_beq_1 * self.B3,
        )
        # ker_loc_3d.kernel_inner_2(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisN[1], self.basisN[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.F2, self.mat_curl_beq_2)

        # assembly of F (3 - component)
        ker_loc_3d.kernel_inner_2(
            self.Nel[0],
            self.Nel[1],
            self.Nel[2],
            self.p[0],
            self.p[1],
            self.p[2],
            self.n_quad[0],
            self.n_quad[1],
            self.n_quad[2],
            0,
            0,
            0,
            self.wts[0],
            self.wts[1],
            self.wts[2],
            self.basisN[0],
            self.basisN[1],
            self.basisN[2],
            self.NbaseN[0],
            self.NbaseN[1],
            self.NbaseN[2],
            self.F3,
            self.mat_curl_beq_1 * self.B2 - self.mat_curl_beq_2 * self.B1,
        )
        # ker_loc_3d.kernel_inner_2(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 0, self.wts[0], self.wts[1], self.wts[2], self.basisN[0], self.basisN[1], self.basisN[2], self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.F3, self.mat_curl_beq_3)

        # convert to 1d array and return
        return xp.concatenate((self.F1.flatten(), self.F2.flatten(), self.F3.flatten()))


# ================ mass matrix in V1 ===========================
def mass_curl(tensor_space, kind_map, params_map):
    """

    Parameters
    ----------
    tensor_space : Tensor_spline_space
        tensor product B-spline space for finite element spaces

    kind_map : int
        type of mapping in case of analytical mapping

    params_map : list of doubles
        parameters for the mapping in case of analytical mapping
    """

    p = tensor_space.p  # spline degrees
    Nel = tensor_space.Nel  # number of elements
    bc = tensor_space.bc  # boundary conditions (periodic vs. clamped)
    NbaseN = tensor_space.NbaseN  # total number of basis functions (N)
    NbaseD = tensor_space.NbaseD  # total number of basis functions (D)

    n_quad = tensor_space.n_quad  # number of quadrature points per element
    pts = tensor_space.pts  # global quadrature points
    wts = tensor_space.wts  # global quadrature weights

    basisN = tensor_space.basisN  # evaluated basis functions at quadrature points (N)
    basisD = tensor_space.basisD  # evaluated basis functions at quadrature points (D)

    # number of basis functions
    # blocks   12         13         21         23         31          32
    Nbi1 = [NbaseN[0], NbaseN[0], NbaseN[0], NbaseN[0], NbaseN[0], NbaseN[0]]
    Nbi2 = [NbaseN[1], NbaseN[1], NbaseN[1], NbaseN[1], NbaseN[1], NbaseN[1]]
    Nbi3 = [NbaseN[2], NbaseN[2], NbaseN[2], NbaseN[2], NbaseN[2], NbaseN[2]]

    Nbj1 = [NbaseD[0], NbaseD[0], NbaseN[0], NbaseD[0], NbaseN[0], NbaseD[0]]
    Nbj2 = [NbaseN[1], NbaseD[1], NbaseD[1], NbaseD[1], NbaseD[1], NbaseN[1]]
    Nbj3 = [NbaseD[2], NbaseN[2], NbaseD[2], NbaseN[2], NbaseD[2], NbaseD[2]]

    # ============= evaluation of background magnetic field at quadrature points =========
    mat_curl_beq_1 = xp.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    mat_curl_beq_2 = xp.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)
    mat_curl_beq_3 = xp.empty((Nel[0], Nel[1], Nel[2], n_quad[0], n_quad[1], n_quad[2]), dtype=float)

    ker_eva.kernel_eva_quad(Nel, n_quad, pts[0], pts[1], pts[2], mat_curl_beq_1, 61, kind_map, params_map)
    ker_eva.kernel_eva_quad(Nel, n_quad, pts[0], pts[1], pts[2], mat_curl_beq_2, 62, kind_map, params_map)
    ker_eva.kernel_eva_quad(Nel, n_quad, pts[0], pts[1], pts[2], mat_curl_beq_3, 63, kind_map, params_map)
    # =====================================================================================

    # blocks of global mass matrix
    M = [
        xp.zeros((Nbi1, Nbi2, Nbi3, 2 * p[0] + 1, 2 * p[1] + 1, 2 * p[2] + 1), dtype=float)
        for Nbi1, Nbi2, Nbi3 in zip(Nbi1, Nbi2, Nbi3)
    ]

    # assembly of block 12
    ker_loc_3d.kernel_mass(
        Nel[0],
        Nel[1],
        Nel[2],
        p[0],
        p[1],
        p[2],
        n_quad[0],
        n_quad[1],
        n_quad[2],
        0,
        0,
        0,
        1,
        0,
        1,
        wts[0],
        wts[1],
        wts[2],
        basisN[0],
        basisN[1],
        basisN[2],
        basisD[0],
        basisN[1],
        basisD[2],
        NbaseN[0],
        NbaseN[1],
        NbaseN[2],
        M[0],
        mat_curl_beq_3,
    )

    # assembly of block 13
    ker_loc_3d.kernel_mass(
        Nel[0],
        Nel[1],
        Nel[2],
        p[0],
        p[1],
        p[2],
        n_quad[0],
        n_quad[1],
        n_quad[2],
        0,
        0,
        0,
        1,
        1,
        0,
        wts[0],
        wts[1],
        wts[2],
        basisN[0],
        basisN[1],
        basisN[2],
        basisD[0],
        basisD[1],
        basisN[2],
        NbaseN[0],
        NbaseN[1],
        NbaseN[2],
        M[1],
        mat_curl_beq_2,
    )

    # assembly of block 21
    ker_loc_3d.kernel_mass(
        Nel[0],
        Nel[1],
        Nel[2],
        p[0],
        p[1],
        p[2],
        n_quad[0],
        n_quad[1],
        n_quad[2],
        0,
        0,
        0,
        0,
        1,
        1,
        wts[0],
        wts[1],
        wts[2],
        basisN[0],
        basisN[1],
        basisN[2],
        basisN[0],
        basisD[1],
        basisD[2],
        NbaseN[0],
        NbaseN[1],
        NbaseN[2],
        M[2],
        mat_curl_beq_3,
    )

    # assembly of block 23
    ker_loc_3d.kernel_mass(
        Nel[0],
        Nel[1],
        Nel[2],
        p[0],
        p[1],
        p[2],
        n_quad[0],
        n_quad[1],
        n_quad[2],
        0,
        0,
        0,
        1,
        1,
        0,
        wts[0],
        wts[1],
        wts[2],
        basisN[0],
        basisN[1],
        basisN[2],
        basisD[0],
        basisD[1],
        basisN[2],
        NbaseN[0],
        NbaseN[1],
        NbaseN[2],
        M[3],
        mat_curl_beq_1,
    )

    # assembly of block 31
    ker_loc_3d.kernel_mass(
        Nel[0],
        Nel[1],
        Nel[2],
        p[0],
        p[1],
        p[2],
        n_quad[0],
        n_quad[1],
        n_quad[2],
        0,
        0,
        0,
        0,
        1,
        1,
        wts[0],
        wts[1],
        wts[2],
        basisN[0],
        basisN[1],
        basisN[2],
        basisN[0],
        basisD[1],
        basisD[2],
        NbaseN[0],
        NbaseN[1],
        NbaseN[2],
        M[4],
        mat_curl_beq_2,
    )

    # assembly of block 32
    ker_loc_3d.kernel_mass(
        Nel[0],
        Nel[1],
        Nel[2],
        p[0],
        p[1],
        p[2],
        n_quad[0],
        n_quad[1],
        n_quad[2],
        0,
        0,
        0,
        1,
        0,
        1,
        wts[0],
        wts[1],
        wts[2],
        basisN[0],
        basisN[1],
        basisN[2],
        basisD[0],
        basisN[1],
        basisD[2],
        NbaseN[0],
        NbaseN[1],
        NbaseN[2],
        M[5],
        mat_curl_beq_1,
    )

    # global indices
    counter = 0

    for i in range(6):
        indices = xp.indices((Nbi1[counter], Nbi2[counter], Nbi3[counter], 2 * p[0] + 1, 2 * p[1] + 1, 2 * p[2] + 1))

        shift1 = xp.arange(Nbi1[counter]) - p[0]
        shift2 = xp.arange(Nbi2[counter]) - p[1]
        shift3 = xp.arange(Nbi3[counter]) - p[2]

        row = (Nbi2[counter] * Nbi3[counter] * indices[0] + Nbi3[counter] * indices[1] + indices[2]).flatten()

        col1 = (indices[3] + shift1[:, None, None, None, None, None]) % Nbj1[counter]
        col2 = (indices[4] + shift2[None, :, None, None, None, None]) % Nbj2[counter]
        col3 = (indices[5] + shift3[None, None, :, None, None, None]) % Nbj3[counter]

        col = Nbj2[counter] * Nbj3[counter] * col1 + Nbj3[counter] * col2 + col3

        M[counter] = spa.csc_matrix(
            (M[counter].flatten(), (row, col.flatten())),
            shape=(Nbi1[counter] * Nbi2[counter] * Nbi3[counter], Nbj1[counter] * Nbj2[counter] * Nbj3[counter]),
        )
        M[counter].eliminate_zeros()

        counter += 1

    M = spa.bmat([[None, -M[0], M[1]], [M[2], None, -M[3]], [-M[4], M[5], None]], format="csc")

    return M
