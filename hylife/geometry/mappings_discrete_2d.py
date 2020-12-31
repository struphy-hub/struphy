# coding: utf-8
#
# Copnyright 2020 Florian Holderied

"""
Basic functions for point-wise evaluation of a 2d discrete B-spline mapping.
"""

from pyccel.decorators import types

import hylife.utilitis_FEEC.basics.spline_evaluation_2d as eva

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double','double')
def f(tn1, tn2, pn, nbase_n, control, eta1, eta2):
    return eva.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], control, eta1, eta2)

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double','double')
def df_1(tn1, tn2, pn, nbase_n, control, eta1, eta2):
    return eva.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], control, eta1, eta2)

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double','double')
def df_2(tn1, tn2, pn, nbase_n, control, eta1, eta2):
    return eva.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], control, eta1, eta2)

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double','double')
def det_df(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2):
    
    df_11 = df_1(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    df_12 = df_2(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    
    df_21 = df_1(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    df_22 = df_2(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    
    return df_11*df_22 - df_12*df_21

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double','double')
def dfinv_11(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2):
    
    df_11  = df_1(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    df_12  = df_2(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    
    df_21  = df_1(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    df_22  = df_2(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    
    det_df = df_11*df_22 - df_12*df_21
    
    return df_22/det_df

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double','double')
def dfinv_12(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2):
    
    df_11  = df_1(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    df_12  = df_2(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    
    df_21  = df_1(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    df_22  = df_2(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    
    det_df = df_11*df_22 - df_12*df_21
    
    return -df_12/det_df

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double','double')
def dfinv_21(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2):
    
    df_11  = df_1(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    df_12  = df_2(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    
    df_21  = df_1(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    df_22  = df_2(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    
    det_df = df_11*df_22 - df_12*df_21
    
    return -df_21/det_df

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double','double')
def dfinv_22(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2):
    
    df_11  = df_1(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    df_12  = df_2(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    
    df_21  = df_1(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    df_22  = df_2(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    
    det_df = df_11*df_22 - df_12*df_21
    
    return df_11/det_df

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double','double')
def g_11(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2):
    
    df_11  = df_1(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    df_12  = df_2(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    
    return df_11*df_11 + df_12*df_12

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double','double')
def g_12(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2):
    
    df_11  = df_1(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    df_12  = df_2(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    
    df_21  = df_1(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    df_22  = df_2(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    
    return df_11*df_12 + df_21*df_22

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double','double')
def g_21(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2):
    
    df_11  = df_1(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    df_12  = df_2(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    
    df_21  = df_1(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    df_22  = df_2(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    
    return df_12*df_11 + df_22*df_21

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double','double')
def g_22(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2):
    
    df_12  = df_2(tn1, tn2, pn, nbase_n, cx, eta1, eta2)
    df_22  = df_2(tn1, tn2, pn, nbase_n, cy, eta1, eta2)
    
    return df_12*df_12 + df_22*df_22

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double','double')
def ginv_11(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2):
    
    dfinv11  = dfinv_11(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2)
    dfinv12  = dfinv_12(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2)
    
    return dfinv11*dfinv11 + dfinv12*dfinv12

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double','double')
def ginv_12(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2):
    
    dfinv11  = dfinv_11(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2)
    dfinv12  = dfinv_12(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2)
    
    dfinv21  = dfinv_21(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2)
    dfinv22  = dfinv_22(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2)
    
    return dfinv11*dfinv21 + dfinv12*dfinv22

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double','double')
def ginv_21(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2):
    
    dfinv11  = dfinv_11(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2)
    dfinv12  = dfinv_12(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2)
    
    dfinv21  = dfinv_21(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2)
    dfinv22  = dfinv_22(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2)
    
    return dfinv21*dfinv11 + dfinv22*dfinv12

# ==========================================================================
@types('double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double','double')
def ginv_22(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2):
    
    dfinv21  = dfinv_21(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2)
    dfinv22  = dfinv_22(tn1, tn2, pn, nbase_n, cx, cy, eta1, eta2)
    
    return dfinv21*dfinv21 + dfinv22*dfinv22