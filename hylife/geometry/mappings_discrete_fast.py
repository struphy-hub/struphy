# coding: utf-8
#
# Copnyright 2020 Florian Holderied

"""
Basic functions for point-wise evaluation of a 3d discrete B-spline mapping.
"""

from pyccel.decorators import types

import hylife.utilitis_FEEC.bsplines_kernels as bsp
import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva

from numpy import empty


# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double','double[:,:]')
def df(tn1, tn2, tn3, pn, nbase_n, span_n, cx, cy, cz, eta1, eta2, eta3, mat):
    
    # evaluate non-vanishing basis functions and its derivatives
    b1 = empty((2, pn[0] + 1), dtype=float)
    b2 = empty((2, pn[1] + 1), dtype=float)
    b3 = empty((2, pn[2] + 1), dtype=float)
    
    bsp.basis_funs_all_ders(tn1, pn[0], eta1, span_n[0], 1, b1)
    bsp.basis_funs_all_ders(tn2, pn[1], eta2, span_n[1], 1, b2)
    bsp.basis_funs_all_ders(tn3, pn[2], eta3, span_n[2], 1, b3)

    # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
    mat[0, 0] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    mat[0, 1] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    mat[0, 2] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    
    # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
    mat[1, 0] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    mat[1, 1] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    mat[1, 2] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    
    # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
    mat[2, 0] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    mat[2, 1] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    mat[2, 2] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def det_df(tn1, tn2, tn3, pn, nbase_n, span_n, cx, cy, cz, eta1, eta2, eta3):
    
    # evaluate non-vanishing basis functions and its derivatives
    b1 = empty((2, pn[0] + 1), dtype=float)
    b2 = empty((2, pn[1] + 1), dtype=float)
    b3 = empty((2, pn[2] + 1), dtype=float)
    
    bsp.basis_funs_all_ders(tn1, pn[0], eta1, span_n[0], 1, b1)
    bsp.basis_funs_all_ders(tn2, pn[1], eta2, span_n[1], 1, b2)
    bsp.basis_funs_all_ders(tn3, pn[2], eta3, span_n[2], 1, b3)

    # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
    df_11 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    df_12 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    df_13 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    
    # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
    df_21 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    df_22 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    df_23 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    
    # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
    df_31 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    df_32 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    df_33 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    
    # compute determinant
    value = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
    
    return value
    
    
# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double','double[:,:]')
def df_inv(tn1, tn2, tn3, pn, nbase_n, span_n, cx, cy, cz, eta1, eta2, eta3, mat):
    
    # evaluate non-vanishing basis functions and its derivatives
    b1 = empty((2, pn[0] + 1), dtype=float)
    b2 = empty((2, pn[1] + 1), dtype=float)
    b3 = empty((2, pn[2] + 1), dtype=float)
    
    bsp.basis_funs_all_ders(tn1, pn[0], eta1, span_n[0], 1, b1)
    bsp.basis_funs_all_ders(tn2, pn[1], eta2, span_n[1], 1, b2)
    bsp.basis_funs_all_ders(tn3, pn[2], eta3, span_n[2], 1, b3)

    # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
    df_11 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    df_12 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    df_13 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    
    # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
    df_21 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    df_22 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    df_23 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    
    # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
    df_31 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    df_32 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    df_33 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    
    # inverte Jabobian matrix
    over_det_df = 1. / (df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13))
    
    mat[0, 0] = (df_22*df_33 - df_23*df_32) * over_det_df
    mat[0, 1] = (df_13*df_32 - df_12*df_33) * over_det_df
    mat[0, 2] = (df_12*df_23 - df_13*df_22) * over_det_df
    
    mat[1, 0] = (df_23*df_31 - df_21*df_33) * over_det_df
    mat[1, 1] = (df_11*df_33 - df_13*df_31) * over_det_df
    mat[1, 2] = (df_13*df_21 - df_11*df_23) * over_det_df
    
    mat[2, 0] = (df_21*df_32 - df_22*df_31) * over_det_df
    mat[2, 1] = (df_12*df_31 - df_11*df_32) * over_det_df
    mat[2, 2] = (df_11*df_22 - df_12*df_21) * over_det_df
    
    
# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double','double[:,:]')
def g(tn1, tn2, tn3, pn, nbase_n, span_n, cx, cy, cz, eta1, eta2, eta3, mat):
    
    # evaluate non-vanishing basis functions and its derivatives
    b1 = empty((2, pn[0] + 1), dtype=float)
    b2 = empty((2, pn[1] + 1), dtype=float)
    b3 = empty((2, pn[2] + 1), dtype=float)
    
    bsp.basis_funs_all_ders(tn1, pn[0], eta1, span_n[0], 1, b1)
    bsp.basis_funs_all_ders(tn2, pn[1], eta2, span_n[1], 1, b2)
    bsp.basis_funs_all_ders(tn3, pn[2], eta3, span_n[2], 1, b3)

    # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
    df_11 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    df_12 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    df_13 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    
    # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
    df_21 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    df_22 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    df_23 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    
    # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
    df_31 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    df_32 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    df_33 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    
    # assemble g
    mat[0, 0] = df_11*df_11 + df_21*df_21 + df_31*df_31
    mat[0, 1] = df_11*df_12 + df_21*df_22 + df_31*df_32
    mat[0, 2] = df_11*df_13 + df_21*df_23 + df_31*df_33
    
    mat[1, 0] = mat[0, 1]
    mat[1, 1] = df_12*df_12 + df_22*df_22 + df_32*df_32
    mat[1, 2] = df_12*df_13 + df_22*df_23 + df_32*df_33
    
    mat[2, 0] = mat[0, 2]
    mat[2, 1] = mat[1, 2]
    mat[2, 2] = df_13*df_13 + df_23*df_23 + df_33*df_33
    
    
# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double','double[:,:]')
def g_inv(tn1, tn2, tn3, pn, nbase_n, span_n, cx, cy, cz, eta1, eta2, eta3, mat):
    
    # evaluate non-vanishing basis functions and its derivatives
    b1 = empty((2, pn[0] + 1), dtype=float)
    b2 = empty((2, pn[1] + 1), dtype=float)
    b3 = empty((2, pn[2] + 1), dtype=float)
    
    bsp.basis_funs_all_ders(tn1, pn[0], eta1, span_n[0], 1, b1)
    bsp.basis_funs_all_ders(tn2, pn[1], eta2, span_n[1], 1, b2)
    bsp.basis_funs_all_ders(tn3, pn[2], eta3, span_n[2], 1, b3)

    # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
    df_11 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    df_12 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    df_13 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cx)
    
    # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
    df_21 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    df_22 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    df_23 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cy)
    
    # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
    df_31 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[1], b2[0], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    df_32 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[1], b3[0], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    df_33 = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[0], b2[0], b3[1], span_n[0], span_n[1], span_n[2], nbase_n[0], nbase_n[1], nbase_n[2], cz)
    
    # inverte Jabobian matrix
    over_det_df = 1. / (df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13))
    
    dfinv_11 = (df_22*df_33 - df_23*df_32) * over_det_df
    dfinv_12 = (df_13*df_32 - df_12*df_33) * over_det_df
    dfinv_13 = (df_12*df_23 - df_13*df_22) * over_det_df
    
    dfinv_21 = (df_23*df_31 - df_21*df_33) * over_det_df
    dfinv_22 = (df_11*df_33 - df_13*df_31) * over_det_df
    dfinv_23 = (df_13*df_21 - df_11*df_23) * over_det_df
    
    dfinv_31 = (df_21*df_32 - df_22*df_31) * over_det_df
    dfinv_32 = (df_12*df_31 - df_11*df_32) * over_det_df
    dfinv_33 = (df_11*df_22 - df_12*df_21) * over_det_df
    
    # assemble g_inv
    mat[0, 0] = dfinv_11*dfinv_11 + dfinv_12*dfinv_12 + dfinv_13*dfinv_13
    mat[0, 1] = dfinv_11*dfinv_21 + dfinv_12*dfinv_22 + dfinv_13*dfinv_23
    mat[0, 2] = dfinv_11*dfinv_31 + dfinv_12*dfinv_32 + dfinv_13*dfinv_33
    
    mat[1, 0] = mat[0, 1]
    mat[1, 1] = dfinv_21*dfinv_21 + dfinv_22*dfinv_22 + dfinv_23*dfinv_23
    mat[1, 2] = dfinv_21*dfinv_31 + dfinv_22*dfinv_32 + dfinv_23*dfinv_33
    
    mat[2, 0] = mat[0, 2]
    mat[2, 1] = mat[1, 2]
    mat[2, 2] = dfinv_31*dfinv_31 + dfinv_32*dfinv_32 + dfinv_33*dfinv_33