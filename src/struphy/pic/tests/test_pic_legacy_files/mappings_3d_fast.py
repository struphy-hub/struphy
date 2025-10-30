# coding: utf-8


"""
Efficient modules for point-wise evaluation of a 3d analytical (kind_map >= 10) or discrete (kind_map < 10) B-spline mapping.
Especially suited for PIC routines since it avoids computing the Jacobian matrix multiple times.
"""

from numpy import cos, empty, pi, sin

import struphy.bsplines.bsplines_kernels as bsp
import struphy.pic.tests.test_pic_legacy_files.mappings_3d as mapping
from struphy.pic.tests.test_pic_legacy_files.spline_evaluation_2d import evaluation_kernel_2d
from struphy.pic.tests.test_pic_legacy_files.spline_evaluation_3d import evaluation_kernel_3d


# ==========================================================================
def df_all(
    kind_map: "int",
    params_map: "float[:]",
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    pn: "int[:]",
    nbase_n: "int[:]",
    span_n1: "int",
    span_n2: "int",
    span_n3: "int",
    cx: "float[:,:,:]",
    cy: "float[:,:,:]",
    cz: "float[:,:,:]",
    l1: "float[:]",
    l2: "float[:]",
    l3: "float[:]",
    r1: "float[:]",
    r2: "float[:]",
    r3: "float[:]",
    b1: "float[:,:]",
    b2: "float[:,:]",
    b3: "float[:,:]",
    d1: "float[:]",
    d2: "float[:]",
    d3: "float[:]",
    der1: "float[:]",
    der2: "float[:]",
    der3: "float[:]",
    eta1: "float",
    eta2: "float",
    eta3: "float",
    mat_out: "float[:,:]",
    vec_out: "float[:]",
    mat_or_vec: "int",
):
    """
    TODO: write documentation, implement faster eval_kernels (with list of global indices, not modulo-operation)
    """
    # 3d discrete mapping
    if kind_map == 0:
        # evaluate non-vanishing basis functions and its derivatives
        bsp.basis_funs_and_der(tn1, pn[0], eta1, span_n1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(tn2, pn[1], eta2, span_n2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(tn3, pn[2], eta3, span_n3, l3, r3, b3, d3, der3)

        # evaluate Jacobian matrix
        if mat_or_vec == 0 or mat_or_vec == 2:
            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            mat_out[0, 0] = evaluation_kernel_3d(
                pn[0],
                pn[1],
                pn[2],
                der1,
                b2[pn[1]],
                b3[pn[2]],
                span_n1,
                span_n2,
                span_n3,
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cx,
            )
            mat_out[0, 1] = evaluation_kernel_3d(
                pn[0],
                pn[1],
                pn[2],
                b1[pn[0]],
                der2,
                b3[pn[2]],
                span_n1,
                span_n2,
                span_n3,
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cx,
            )
            mat_out[0, 2] = evaluation_kernel_3d(
                pn[0],
                pn[1],
                pn[2],
                b1[pn[0]],
                b2[pn[1]],
                der3,
                span_n1,
                span_n2,
                span_n3,
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cx,
            )

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            mat_out[1, 0] = evaluation_kernel_3d(
                pn[0],
                pn[1],
                pn[2],
                der1,
                b2[pn[1]],
                b3[pn[2]],
                span_n1,
                span_n2,
                span_n3,
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cy,
            )
            mat_out[1, 1] = evaluation_kernel_3d(
                pn[0],
                pn[1],
                pn[2],
                b1[pn[0]],
                der2,
                b3[pn[2]],
                span_n1,
                span_n2,
                span_n3,
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cy,
            )
            mat_out[1, 2] = evaluation_kernel_3d(
                pn[0],
                pn[1],
                pn[2],
                b1[pn[0]],
                b2[pn[1]],
                der3,
                span_n1,
                span_n2,
                span_n3,
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cy,
            )

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            mat_out[2, 0] = evaluation_kernel_3d(
                pn[0],
                pn[1],
                pn[2],
                der1,
                b2[pn[1]],
                b3[pn[2]],
                span_n1,
                span_n2,
                span_n3,
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cz,
            )
            mat_out[2, 1] = evaluation_kernel_3d(
                pn[0],
                pn[1],
                pn[2],
                b1[pn[0]],
                der2,
                b3[pn[2]],
                span_n1,
                span_n2,
                span_n3,
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cz,
            )
            mat_out[2, 2] = evaluation_kernel_3d(
                pn[0],
                pn[1],
                pn[2],
                b1[pn[0]],
                b2[pn[1]],
                der3,
                span_n1,
                span_n2,
                span_n3,
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cz,
            )

        # evaluate mapping
        if mat_or_vec == 1 or mat_or_vec == 2:
            vec_out[0] = evaluation_kernel_3d(
                pn[0],
                pn[1],
                pn[2],
                b1[pn[0]],
                b2[pn[1]],
                b3[pn[2]],
                span_n1,
                span_n2,
                span_n3,
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cx,
            )
            vec_out[1] = evaluation_kernel_3d(
                pn[0],
                pn[1],
                pn[2],
                b1[pn[0]],
                b2[pn[1]],
                b3[pn[2]],
                span_n1,
                span_n2,
                span_n3,
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cy,
            )
            vec_out[2] = evaluation_kernel_3d(
                pn[0],
                pn[1],
                pn[2],
                b1[pn[0]],
                b2[pn[1]],
                b3[pn[2]],
                span_n1,
                span_n2,
                span_n3,
                nbase_n[0],
                nbase_n[1],
                nbase_n[2],
                cz,
            )

    # discrete cylinder
    elif kind_map == 1:
        lz = 2 * pi * cx[0, 0, 0]

        # evaluate non-vanishing basis functions and its derivatives
        bsp.basis_funs_and_der(tn1, pn[0], eta1, span_n1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(tn2, pn[1], eta2, span_n2, l2, r2, b2, d2, der2)

        # evaluate Jacobian matrix
        if mat_or_vec == 0 or mat_or_vec == 2:
            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            mat_out[0, 0] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                der1,
                b2[pn[1]],
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cx[:, :, 0],
            )
            mat_out[0, 1] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                b1[pn[0]],
                der2,
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cx[:, :, 0],
            )
            mat_out[0, 2] = 0.0

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            mat_out[1, 0] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                der1,
                b2[pn[1]],
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cy[:, :, 0],
            )
            mat_out[1, 1] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                b1[pn[0]],
                der2,
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cy[:, :, 0],
            )
            mat_out[1, 2] = 0.0

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            mat_out[2, 0] = 0.0
            mat_out[2, 1] = 0.0
            mat_out[2, 2] = lz

        # evaluate mapping
        if mat_or_vec == 1 or mat_or_vec == 2:
            vec_out[0] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                b1[pn[0]],
                b2[pn[1]],
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cx[:, :, 0],
            )
            vec_out[1] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                b1[pn[0]],
                b2[pn[1]],
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cy[:, :, 0],
            )
            vec_out[2] = lz * eta3

    # discrete torus
    elif kind_map == 2:
        # evaluate non-vanishing basis functions and its derivatives
        bsp.basis_funs_and_der(tn1, pn[0], eta1, span_n1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(tn2, pn[1], eta2, span_n2, l2, r2, b2, d2, der2)

        # evaluate Jacobian matrix
        if mat_or_vec == 0 or mat_or_vec == 2:
            # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
            mat_out[0, 0] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                der1,
                b2[pn[1]],
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cx[:, :, 0],
            ) * cos(2 * pi * eta3)
            mat_out[0, 1] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                b1[pn[0]],
                der2,
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cx[:, :, 0],
            ) * cos(2 * pi * eta3)
            mat_out[0, 2] = (
                evaluation_kernel_2d(
                    pn[0],
                    pn[1],
                    b1[pn[0]],
                    b2[pn[1]],
                    span_n1,
                    span_n2,
                    nbase_n[0],
                    nbase_n[1],
                    cx[:, :, 0],
                )
                * sin(2 * pi * eta3)
                * (-2 * pi)
            )

            # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
            mat_out[1, 0] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                der1,
                b2[pn[1]],
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cy[:, :, 0],
            )
            mat_out[1, 1] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                b1[pn[0]],
                der2,
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cy[:, :, 0],
            )
            mat_out[1, 2] = 0.0

            # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
            mat_out[2, 0] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                der1,
                b2[pn[1]],
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cx[:, :, 0],
            ) * sin(2 * pi * eta3)
            mat_out[2, 1] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                b1[pn[0]],
                der2,
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cx[:, :, 0],
            ) * sin(2 * pi * eta3)
            mat_out[2, 2] = (
                evaluation_kernel_2d(
                    pn[0],
                    pn[1],
                    b1[pn[0]],
                    b2[pn[1]],
                    span_n1,
                    span_n2,
                    nbase_n[0],
                    nbase_n[1],
                    cx[:, :, 0],
                )
                * cos(2 * pi * eta3)
                * 2
                * pi
            )

        # evaluate mapping
        if mat_or_vec == 1 or mat_or_vec == 2:
            vec_out[0] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                b1[pn[0]],
                b2[pn[1]],
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cx[:, :, 0],
            ) * cos(2 * pi * eta3)
            vec_out[1] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                b1[pn[0]],
                b2[pn[1]],
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cy[:, :, 0],
            )
            vec_out[2] = evaluation_kernel_2d(
                pn[0],
                pn[1],
                b1[pn[0]],
                b2[pn[1]],
                span_n1,
                span_n2,
                nbase_n[0],
                nbase_n[1],
                cx[:, :, 0],
            ) * sin(2 * pi * eta3)

    # analytical mapping
    else:
        # evaluate Jacobian matrix
        if mat_or_vec == 0 or mat_or_vec == 2:
            mat_out[0, 0] = mapping.df(
                eta1,
                eta2,
                eta3,
                11,
                kind_map,
                params_map,
                tn1,
                tn2,
                tn3,
                pn,
                nbase_n,
                cx,
                cy,
                cz,
            )
            mat_out[0, 1] = mapping.df(
                eta1,
                eta2,
                eta3,
                12,
                kind_map,
                params_map,
                tn1,
                tn2,
                tn3,
                pn,
                nbase_n,
                cx,
                cy,
                cz,
            )
            mat_out[0, 2] = mapping.df(
                eta1,
                eta2,
                eta3,
                13,
                kind_map,
                params_map,
                tn1,
                tn2,
                tn3,
                pn,
                nbase_n,
                cx,
                cy,
                cz,
            )

            mat_out[1, 0] = mapping.df(
                eta1,
                eta2,
                eta3,
                21,
                kind_map,
                params_map,
                tn1,
                tn2,
                tn3,
                pn,
                nbase_n,
                cx,
                cy,
                cz,
            )
            mat_out[1, 1] = mapping.df(
                eta1,
                eta2,
                eta3,
                22,
                kind_map,
                params_map,
                tn1,
                tn2,
                tn3,
                pn,
                nbase_n,
                cx,
                cy,
                cz,
            )
            mat_out[1, 2] = mapping.df(
                eta1,
                eta2,
                eta3,
                23,
                kind_map,
                params_map,
                tn1,
                tn2,
                tn3,
                pn,
                nbase_n,
                cx,
                cy,
                cz,
            )

            mat_out[2, 0] = mapping.df(
                eta1,
                eta2,
                eta3,
                31,
                kind_map,
                params_map,
                tn1,
                tn2,
                tn3,
                pn,
                nbase_n,
                cx,
                cy,
                cz,
            )
            mat_out[2, 1] = mapping.df(
                eta1,
                eta2,
                eta3,
                32,
                kind_map,
                params_map,
                tn1,
                tn2,
                tn3,
                pn,
                nbase_n,
                cx,
                cy,
                cz,
            )
            mat_out[2, 2] = mapping.df(
                eta1,
                eta2,
                eta3,
                33,
                kind_map,
                params_map,
                tn1,
                tn2,
                tn3,
                pn,
                nbase_n,
                cx,
                cy,
                cz,
            )

        # evaluate mapping
        if mat_or_vec == 1 or mat_or_vec == 2:
            vec_out[0] = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
            vec_out[1] = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
            vec_out[2] = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)


# ===========================================================================
def df_inv_all(mat_in: "float[:,:]", mat_out: "float[:,:]"):
    """
    Inverts the Jacobain matrix (mat_in) and writes it to mat_out

    Parameters:
    -----------
        mat_in : array
            Jacobian matrix

        mat_out : array
            emtpy array where the inverse Jacobian matrix will be written
    """

    # inverse Jacobian determinant computed from Jacobian matrix (mat_in)
    over_det_df = 1.0 / (
        mat_in[0, 0] * (mat_in[1, 1] * mat_in[2, 2] - mat_in[2, 1] * mat_in[1, 2])
        + mat_in[1, 0] * (mat_in[2, 1] * mat_in[0, 2] - mat_in[0, 1] * mat_in[2, 2])
        + mat_in[2, 0] * (mat_in[0, 1] * mat_in[1, 2] - mat_in[1, 1] * mat_in[0, 2])
    )

    # inverse Jacobian matrix computed from Jacobian matrix (mat_in)
    mat_out[0, 0] = (mat_in[1, 1] * mat_in[2, 2] - mat_in[2, 1] * mat_in[1, 2]) * over_det_df
    mat_out[0, 1] = (mat_in[2, 1] * mat_in[0, 2] - mat_in[0, 1] * mat_in[2, 2]) * over_det_df
    mat_out[0, 2] = (mat_in[0, 1] * mat_in[1, 2] - mat_in[1, 1] * mat_in[0, 2]) * over_det_df

    mat_out[1, 0] = (mat_in[1, 2] * mat_in[2, 0] - mat_in[2, 2] * mat_in[1, 0]) * over_det_df
    mat_out[1, 1] = (mat_in[2, 2] * mat_in[0, 0] - mat_in[0, 2] * mat_in[2, 0]) * over_det_df
    mat_out[1, 2] = (mat_in[0, 2] * mat_in[1, 0] - mat_in[1, 2] * mat_in[0, 0]) * over_det_df

    mat_out[2, 0] = (mat_in[1, 0] * mat_in[2, 1] - mat_in[2, 0] * mat_in[1, 1]) * over_det_df
    mat_out[2, 1] = (mat_in[2, 0] * mat_in[0, 1] - mat_in[0, 0] * mat_in[2, 1]) * over_det_df
    mat_out[2, 2] = (mat_in[0, 0] * mat_in[1, 1] - mat_in[1, 0] * mat_in[0, 1]) * over_det_df


# ===========================================================================
def g_all(mat_in: "float[:,:]", mat_out: "float[:,:]"):
    """
    Compute the metric tensor (mat_out) from Jacobian matrix (mat_in)

    Parameters:
    -----------
        mat_in : array
            Jacobian matrix

        mat_out : array
            array where metric tensor will be written to
    """
    mat_out[0, 0] = mat_in[0, 0] * mat_in[0, 0] + mat_in[1, 0] * mat_in[1, 0] + mat_in[2, 0] * mat_in[2, 0]
    mat_out[0, 1] = mat_in[0, 0] * mat_in[0, 1] + mat_in[1, 0] * mat_in[1, 1] + mat_in[2, 0] * mat_in[2, 1]
    mat_out[0, 2] = mat_in[0, 0] * mat_in[0, 2] + mat_in[1, 2] * mat_in[1, 2] + mat_in[2, 0] * mat_in[2, 2]

    mat_out[1, 0] = mat_out[0, 1]
    mat_out[1, 1] = mat_in[0, 1] * mat_in[0, 1] + mat_in[1, 1] * mat_in[1, 1] + mat_in[2, 1] * mat_in[2, 1]
    mat_out[1, 2] = mat_in[0, 1] * mat_in[0, 2] + mat_in[1, 0] * mat_in[1, 2] + mat_in[2, 0] * mat_in[2, 2]

    mat_out[2, 0] = mat_out[0, 2]
    mat_out[2, 1] = mat_out[1, 2]
    mat_out[2, 2] = mat_in[0, 2] * mat_in[0, 2] + mat_in[1, 2] * mat_in[1, 2] + mat_in[2, 2] * mat_in[2, 2]


# ===========================================================================
def g_inv_all(mat_in: "float[:,:]", mat_out: "float[:,:]"):
    """
    Compute the inverse metric tensor (mat_out) from inverse Jacobian matrix (mat_in)

    Parameters:
    -----------
        mat_in : array
            inverse Jacobian matrix

        mat_out : array
            array where inverse metric tensor will be written to
    """
    mat_out[0, 0] = mat_in[0, 0] * mat_in[0, 0] + mat_in[0, 1] * mat_in[0, 1] + mat_in[0, 2] * mat_in[0, 2]
    mat_out[0, 1] = mat_in[0, 0] * mat_in[1, 0] + mat_in[0, 1] * mat_in[1, 1] + mat_in[0, 2] * mat_in[1, 2]
    mat_out[0, 2] = mat_in[0, 0] * mat_in[2, 0] + mat_in[0, 1] * mat_in[2, 1] + mat_in[0, 2] * mat_in[2, 2]

    mat_out[1, 0] = mat_out[0, 1]
    mat_out[1, 1] = mat_in[1, 0] * mat_in[1, 0] + mat_in[1, 1] * mat_in[1, 1] + mat_in[1, 2] * mat_in[1, 2]
    mat_out[1, 2] = mat_in[1, 0] * mat_in[2, 0] + mat_in[1, 1] * mat_in[2, 1] + mat_in[1, 2] * mat_in[2, 2]

    mat_out[2, 0] = mat_out[0, 2]
    mat_out[2, 1] = mat_out[1, 2]
    mat_out[2, 2] = mat_in[2, 0] * mat_in[2, 0] + mat_in[2, 1] * mat_in[2, 1] + mat_in[2, 2] * mat_in[2, 2]
