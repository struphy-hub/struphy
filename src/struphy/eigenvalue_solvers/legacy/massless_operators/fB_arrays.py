import time
import timeit

import numpy as np
import scipy.sparse as spa
from mpi4py import MPI

import struphy.geometry.mappings_3d as mapping3d
import struphy.geometry.mappings_3d_fast as mapping_fast
import struphy.linear_algebra.linalg_kernels as linalg


class Temp_arrays:
    """
    Class for arrays predefined before the time loops

    Parameters
    ---------
    TENSOR_SPACE_FEM : tensor_spline_space-tensor product B-spline space
        DOMAIN :         : class of Domain
        control          : control variate or not
        mpi_comm         : environment of mpi processes
    """

    def __init__(self, TENSOR_SPACE_FEM, DOMAIN, control, mpi_comm):
        self.Nel = TENSOR_SPACE_FEM.Nel
        self.NbaseN = TENSOR_SPACE_FEM.NbaseN  # number of basis functions (N)
        self.NbaseD = TENSOR_SPACE_FEM.NbaseD  # number of basis functions (D)
        self.DOMAIN = DOMAIN
        self.TENSOR_SPACE_FEM = TENSOR_SPACE_FEM
        self.n_quad = TENSOR_SPACE_FEM.n_quad
        self.mpi_rank = mpi_comm.Get_rank()
        self.N_0form = TENSOR_SPACE_FEM.Nbase_0form
        self.N_1form = TENSOR_SPACE_FEM.Nbase_1form
        self.N_2form = TENSOR_SPACE_FEM.Nbase_2form
        self.N_3form = TENSOR_SPACE_FEM.Nbase_3form

        self.Ntot_0form = TENSOR_SPACE_FEM.Ntot_0form
        self.Ntot_1form = TENSOR_SPACE_FEM.Ntot_1form
        self.Ntot_2form = TENSOR_SPACE_FEM.Ntot_2form

        self.b1_old = np.empty(TENSOR_SPACE_FEM.Nbase_1form[0], dtype=float)
        self.b2_old = np.empty(TENSOR_SPACE_FEM.Nbase_1form[1], dtype=float)
        self.b3_old = np.empty(TENSOR_SPACE_FEM.Nbase_1form[2], dtype=float)

        self.b1_iter = np.empty(TENSOR_SPACE_FEM.Nbase_1form[0], dtype=float)
        self.b2_iter = np.empty(TENSOR_SPACE_FEM.Nbase_1form[1], dtype=float)
        self.b3_iter = np.empty(TENSOR_SPACE_FEM.Nbase_1form[2], dtype=float)

        self.temp_dft = np.empty((3, 3), dtype=float)
        self.temp_generate_weight1 = np.empty(3, dtype=float)
        self.temp_generate_weight2 = np.empty(3, dtype=float)
        self.temp_generate_weight3 = np.empty(3, dtype=float)

        self.zerosform_temp_long = np.empty(TENSOR_SPACE_FEM.Ntot_0form, dtype=float)
        self.oneform_temp1_long = np.empty(TENSOR_SPACE_FEM.Ntot_1form[0], dtype=float)
        self.oneform_temp2_long = np.empty(TENSOR_SPACE_FEM.Ntot_1form[1], dtype=float)
        self.oneform_temp3_long = np.empty(TENSOR_SPACE_FEM.Ntot_1form[2], dtype=float)

        self.oneform_temp_long = np.empty(
            TENSOR_SPACE_FEM.Ntot_1form[0] + TENSOR_SPACE_FEM.Ntot_1form[1] + TENSOR_SPACE_FEM.Ntot_1form[2],
            dtype=float,
        )

        self.twoform_temp1_long = np.empty(TENSOR_SPACE_FEM.Ntot_2form[0], dtype=float)
        self.twoform_temp2_long = np.empty(TENSOR_SPACE_FEM.Ntot_2form[1], dtype=float)
        self.twoform_temp3_long = np.empty(TENSOR_SPACE_FEM.Ntot_2form[2], dtype=float)

        self.twoform_temp_long = np.empty(
            TENSOR_SPACE_FEM.Ntot_2form[0] + TENSOR_SPACE_FEM.Ntot_2form[1] + TENSOR_SPACE_FEM.Ntot_2form[2],
            dtype=float,
        )

        self.temp_twoform1 = np.empty(TENSOR_SPACE_FEM.Nbase_2form[0], dtype=float)
        self.temp_twoform2 = np.empty(TENSOR_SPACE_FEM.Nbase_2form[1], dtype=float)
        self.temp_twoform3 = np.empty(TENSOR_SPACE_FEM.Nbase_2form[2], dtype=float)

        # arrays used to store intermidaite values
        self.form_0_flatten = np.empty(self.Ntot_0form, dtype=float)

        self.form_1_1_flatten = np.empty(self.Ntot_1form[0], dtype=float)
        self.form_1_2_flatten = np.empty(self.Ntot_1form[1], dtype=float)
        self.form_1_3_flatten = np.empty(self.Ntot_1form[2], dtype=float)

        self.form_1_tot_flatten = np.empty(self.Ntot_1form[0] + self.Ntot_1form[1] + self.Ntot_1form[2], dtype=float)

        self.form_2_1_flatten = np.empty(self.Ntot_2form[0], dtype=float)
        self.form_2_2_flatten = np.empty(self.Ntot_2form[1], dtype=float)
        self.form_2_3_flatten = np.empty(self.Ntot_2form[2], dtype=float)

        self.form_2_tot_flatten = np.empty(self.Ntot_2form[0] + self.Ntot_2form[1] + self.Ntot_2form[2], dtype=float)

        self.bulkspeed_loc = np.zeros((3, self.Nel[0], self.Nel[1], self.Nel[2]), dtype=float)
        self.temperature_loc = np.zeros((3, self.Nel[0], self.Nel[1], self.Nel[2]), dtype=float)
        self.bulkspeed = np.zeros((3, self.Nel[0], self.Nel[1], self.Nel[2]), dtype=float)
        if self.mpi_rank == 0:
            temperature = np.zeros((3, self.Nel[0], self.Nel[1], self.Nel[2]), dtype=float)
        else:
            temperature = None

        # values of magnetic fields at all quadrature points
        self.LO_inv = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )

        self.LO_b1 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.LO_b2 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.LO_b3 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        # values of weights (used in the linear operators)
        self.LO_w1 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.LO_w2 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.LO_w3 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        # values of a function (given its finite element coefficients) at all quadrature points
        self.LO_r1 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.LO_r2 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.LO_r3 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        # values of determinant of Jacobi matrix of the map at all quadrature points
        self.df_det = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        # when using delta f method, the values of current equilibrium at all quadrature points
        if control == True:
            self.Jeqx = np.empty(
                (
                    self.Nel[0],
                    self.Nel[1],
                    self.Nel[2],
                    TENSOR_SPACE_FEM.n_quad[0],
                    TENSOR_SPACE_FEM.n_quad[1],
                    TENSOR_SPACE_FEM.n_quad[2],
                ),
                dtype=float,
            )
            self.Jeqy = np.empty(
                (
                    self.Nel[0],
                    self.Nel[1],
                    self.Nel[2],
                    TENSOR_SPACE_FEM.n_quad[0],
                    TENSOR_SPACE_FEM.n_quad[1],
                    TENSOR_SPACE_FEM.n_quad[2],
                ),
                dtype=float,
            )
            self.Jeqz = np.empty(
                (
                    self.Nel[0],
                    self.Nel[1],
                    self.Nel[2],
                    TENSOR_SPACE_FEM.n_quad[0],
                    TENSOR_SPACE_FEM.n_quad[1],
                    TENSOR_SPACE_FEM.n_quad[2],
                ),
                dtype=float,
            )
        # values of DF and inverse of DF at all quadrature points
        self.DF_11 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DF_12 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DF_13 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DF_21 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DF_22 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DF_23 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DF_31 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DF_32 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DF_33 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )

        self.DFI_11 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFI_12 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFI_13 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFI_21 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFI_22 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFI_23 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFI_31 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFI_32 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFI_33 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )

        self.DFIT_11 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFIT_12 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFIT_13 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFIT_21 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFIT_22 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFIT_23 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFIT_31 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFIT_32 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.DFIT_33 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )

        self.G_inv_11 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.G_inv_12 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.G_inv_13 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )

        self.G_inv_22 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )
        self.G_inv_23 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )

        self.G_inv_33 = np.empty(
            (
                self.Nel[0],
                self.Nel[1],
                self.Nel[2],
                TENSOR_SPACE_FEM.n_quad[0],
                TENSOR_SPACE_FEM.n_quad[1],
                TENSOR_SPACE_FEM.n_quad[2],
            ),
            dtype=float,
        )

        self.temp_particle = np.empty(3, dtype=float)
        # initialization of DF and its inverse
        # ================ for mapping evaluation ==================
        # spline degrees
        pf1 = DOMAIN.p[0]
        pf2 = DOMAIN.p[1]
        pf3 = DOMAIN.p[2]

        # pf + 1 non-vanishing basis functions up tp degree pf
        b1f = np.empty((pf1 + 1, pf1 + 1), dtype=float)
        b2f = np.empty((pf2 + 1, pf2 + 1), dtype=float)
        b3f = np.empty((pf3 + 1, pf3 + 1), dtype=float)

        # left and right values for spline evaluation
        l1f = np.empty(pf1, dtype=float)
        l2f = np.empty(pf2, dtype=float)
        l3f = np.empty(pf3, dtype=float)

        r1f = np.empty(pf1, dtype=float)
        r2f = np.empty(pf2, dtype=float)
        r3f = np.empty(pf3, dtype=float)

        # scaling arrays for M-splines
        d1f = np.empty(pf1, dtype=float)
        d2f = np.empty(pf2, dtype=float)
        d3f = np.empty(pf3, dtype=float)

        # pf + 1 derivatives
        der1f = np.empty(pf1 + 1, dtype=float)
        der2f = np.empty(pf2 + 1, dtype=float)
        der3f = np.empty(pf3 + 1, dtype=float)

        # needed mapping quantities
        df = np.empty((3, 3), dtype=float)
        fx = np.empty(3, dtype=float)
        ginv = np.empty((3, 3), dtype=float)
        dfinv = np.empty((3, 3), dtype=float)

        for ie1 in range(self.Nel[0]):
            for ie2 in range(self.Nel[1]):
                for ie3 in range(self.Nel[2]):
                    for q1 in range(TENSOR_SPACE_FEM.n_quad[0]):
                        for q2 in range(TENSOR_SPACE_FEM.n_quad[1]):
                            for q3 in range(TENSOR_SPACE_FEM.n_quad[2]):
                                span1f = int(TENSOR_SPACE_FEM.pts[0][ie1, q1] * DOMAIN.NbaseN[0]) + pf1
                                span2f = int(TENSOR_SPACE_FEM.pts[1][ie2, q2] * DOMAIN.NbaseN[1]) + pf2
                                span3f = int(TENSOR_SPACE_FEM.pts[2][ie3, q3] * DOMAIN.NbaseN[2]) + pf3
                                # evaluate Jacobian matrix
                                mapping_fast.df_all(
                                    DOMAIN.kind_map,
                                    DOMAIN.params,
                                    DOMAIN.T[0],
                                    DOMAIN.T[1],
                                    DOMAIN.T[2],
                                    DOMAIN.p,
                                    DOMAIN.NbaseN,
                                    span1f,
                                    span2f,
                                    span3f,
                                    DOMAIN.cx,
                                    DOMAIN.cy,
                                    DOMAIN.cz,
                                    l1f,
                                    l2f,
                                    l3f,
                                    r1f,
                                    r2f,
                                    r3f,
                                    b1f,
                                    b2f,
                                    b3f,
                                    d1f,
                                    d2f,
                                    d3f,
                                    der1f,
                                    der2f,
                                    der3f,
                                    self.TENSOR_SPACE_FEM.pts[0][ie1, q1],
                                    self.TENSOR_SPACE_FEM.pts[1][ie2, q2],
                                    self.TENSOR_SPACE_FEM.pts[2][ie3, q3],
                                    df,
                                    fx,
                                    0,
                                )
                                # evaluate inverse Jacobian matrix
                                mapping_fast.df_inv_all(df, dfinv)
                                # evaluate metric tensor
                                mapping_fast.g_inv_all(dfinv, ginv)
                                # evaluate Jacobian determinant
                                det_number = abs(linalg.det(df))

                                self.DFI_11[ie1, ie2, ie3, q1, q2, q3] = dfinv[0, 0]
                                self.DFI_12[ie1, ie2, ie3, q1, q2, q3] = dfinv[0, 1]
                                self.DFI_13[ie1, ie2, ie3, q1, q2, q3] = dfinv[0, 2]
                                self.DFI_21[ie1, ie2, ie3, q1, q2, q3] = dfinv[1, 0]
                                self.DFI_22[ie1, ie2, ie3, q1, q2, q3] = dfinv[1, 1]
                                self.DFI_23[ie1, ie2, ie3, q1, q2, q3] = dfinv[1, 2]
                                self.DFI_31[ie1, ie2, ie3, q1, q2, q3] = dfinv[2, 0]
                                self.DFI_32[ie1, ie2, ie3, q1, q2, q3] = dfinv[2, 1]
                                self.DFI_33[ie1, ie2, ie3, q1, q2, q3] = dfinv[2, 2]

                                self.DFIT_11[ie1, ie2, ie3, q1, q2, q3] = dfinv[0, 0]
                                self.DFIT_12[ie1, ie2, ie3, q1, q2, q3] = dfinv[1, 0]
                                self.DFIT_13[ie1, ie2, ie3, q1, q2, q3] = dfinv[2, 0]
                                self.DFIT_21[ie1, ie2, ie3, q1, q2, q3] = dfinv[0, 1]
                                self.DFIT_22[ie1, ie2, ie3, q1, q2, q3] = dfinv[1, 1]
                                self.DFIT_23[ie1, ie2, ie3, q1, q2, q3] = dfinv[2, 1]
                                self.DFIT_31[ie1, ie2, ie3, q1, q2, q3] = dfinv[0, 2]
                                self.DFIT_32[ie1, ie2, ie3, q1, q2, q3] = dfinv[1, 2]
                                self.DFIT_33[ie1, ie2, ie3, q1, q2, q3] = dfinv[2, 2]

                                self.DF_11[ie1, ie2, ie3, q1, q2, q3] = df[0, 0]
                                self.DF_12[ie1, ie2, ie3, q1, q2, q3] = df[0, 1]
                                self.DF_13[ie1, ie2, ie3, q1, q2, q3] = df[0, 2]
                                self.DF_21[ie1, ie2, ie3, q1, q2, q3] = df[1, 0]
                                self.DF_22[ie1, ie2, ie3, q1, q2, q3] = df[1, 1]
                                self.DF_23[ie1, ie2, ie3, q1, q2, q3] = df[1, 2]
                                self.DF_31[ie1, ie2, ie3, q1, q2, q3] = df[2, 0]
                                self.DF_32[ie1, ie2, ie3, q1, q2, q3] = df[2, 1]
                                self.DF_33[ie1, ie2, ie3, q1, q2, q3] = df[2, 2]

                                self.G_inv_11[ie1, ie2, ie3, q1, q2, q3] = ginv[0, 0]
                                self.G_inv_12[ie1, ie2, ie3, q1, q2, q3] = ginv[0, 1]
                                self.G_inv_13[ie1, ie2, ie3, q1, q2, q3] = ginv[0, 2]

                                self.G_inv_22[ie1, ie2, ie3, q1, q2, q3] = ginv[1, 1]
                                self.G_inv_23[ie1, ie2, ie3, q1, q2, q3] = ginv[1, 2]

                                self.G_inv_33[ie1, ie2, ie3, q1, q2, q3] = ginv[2, 2]

                                self.df_det[ie1, ie2, ie3, q1, q2, q3] = det_number

                                if control == True:
                                    x1 = mapping3d.f(
                                        TENSOR_SPACE_FEM.pts[0][ie1, q1],
                                        TENSOR_SPACE_FEM.pts[1][ie2, q2],
                                        TENSOR_SPACE_FEM.pts[2][ie3, q3],
                                        1,
                                        DOMAIN.kind_map,
                                        DOMAIN.params,
                                        DOMAIN.T[0],
                                        DOMAIN.T[1],
                                        DOMAIN.T[2],
                                        DOMAIN.p,
                                        DOMAIN.NbaseN,
                                        DOMAIN.cx,
                                        DOMAIN.cy,
                                        DOMAIN.cz,
                                    )
                                    x2 = mapping3d.f(
                                        TENSOR_SPACE_FEM.pts[0][ie1, q1],
                                        TENSOR_SPACE_FEM.pts[1][ie2, q2],
                                        TENSOR_SPACE_FEM.pts[2][ie3, q3],
                                        2,
                                        DOMAIN.kind_map,
                                        DOMAIN.params,
                                        DOMAIN.T[0],
                                        DOMAIN.T[1],
                                        DOMAIN.T[2],
                                        DOMAIN.p,
                                        DOMAIN.NbaseN,
                                        DOMAIN.cx,
                                        DOMAIN.cy,
                                        DOMAIN.cz,
                                    )
                                    x3 = mapping3d.f(
                                        TENSOR_SPACE_FEM.pts[0][ie1, q1],
                                        TENSOR_SPACE_FEM.pts[1][ie2, q2],
                                        TENSOR_SPACE_FEM.pts[2][ie3, q3],
                                        3,
                                        DOMAIN.kind_map,
                                        DOMAIN.params,
                                        DOMAIN.T[0],
                                        DOMAIN.T[1],
                                        DOMAIN.T[2],
                                        DOMAIN.p,
                                        DOMAIN.NbaseN,
                                        DOMAIN.cx,
                                        DOMAIN.cy,
                                        DOMAIN.cz,
                                    )
                                    Jeqx[ie1, ie2, ie3, q1, q2, q3] = equ_PIC.jhx_eq(x1, x2, x3)
                                    Jeqy[ie1, ie2, ie3, q1, q2, q3] = equ_PIC.jhy_eq(x1, x2, x3)
                                    Jeqz[ie1, ie2, ie3, q1, q2, q3] = equ_PIC.jhz_eq(x1, x2, x3)
