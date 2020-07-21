# coding: utf-8
#
# Copnyright 2020 Florian Holderied

"""
Basic functions for point-wise evaluation of a 3d discrete B-spline mapping.
"""

from pyccel.decorators import types

import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double','double','double')
def f(tn1, tn2, tn3, pn, nbase_n, control, eta1, eta2, eta3):
    return eva.evaluate_n_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], control, eta1, eta2, eta3)

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double','double','double')
def df_1(tn1, tn2, tn3, pn, nbase_n, control, eta1, eta2, eta3):
    return eva.evaluate_diffn_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], control, eta1, eta2, eta3)

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double','double','double')
def df_2(tn1, tn2, tn3, pn, nbase_n, control, eta1, eta2, eta3):
    return eva.evaluate_n_diffn_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], control, eta1, eta2, eta3)

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double','double','double')
def df_3(tn1, tn2, tn3, pn, nbase_n, control, eta1, eta2, eta3):
    return eva.evaluate_n_n_diffn(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], control, eta1, eta2, eta3)

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def det_df(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11 = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_12 = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_13 = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21 = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_22 = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_23 = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31 = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_32 = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_33 = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    return df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def dfinv_11(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_12  = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_13  = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_22  = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_23  = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_32  = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_33  = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    det_df = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
    
    return (df_22*df_33 - df_23*df_32)/det_df

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def dfinv_12(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_12  = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_13  = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_22  = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_23  = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_32  = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_33  = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    det_df = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
    
    return (df_13*df_32 - df_12*df_33)/det_df

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def dfinv_13(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_12  = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_13  = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_22  = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_23  = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_32  = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_33  = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    det_df = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
    
    return (df_12*df_23 - df_13*df_22)/det_df

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def dfinv_21(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_12  = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_13  = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_22  = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_23  = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_32  = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_33  = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    det_df = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
    
    return (df_23*df_31 - df_21*df_33)/det_df

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def dfinv_22(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_12  = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_13  = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_22  = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_23  = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_32  = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_33  = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    det_df = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
    
    return (df_11*df_33 - df_13*df_31)/det_df

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def dfinv_23(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_12  = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_13  = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_22  = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_23  = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_32  = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_33  = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    det_df = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
    
    return (df_13*df_21 - df_11*df_23)/det_df

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def dfinv_31(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_12  = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_13  = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_22  = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_23  = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_32  = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_33  = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    det_df = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
    
    return (df_21*df_32 - df_22*df_31)/det_df

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def dfinv_32(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_12  = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_13  = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_22  = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_23  = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_32  = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_33  = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    det_df = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
    
    return (df_12*df_31 - df_11*df_32)/det_df

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def dfinv_33(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_12  = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_13  = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_22  = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_23  = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_32  = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_33  = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    det_df = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
    
    return (df_11*df_22 - df_12*df_21)/det_df

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def g_11(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    return df_11*df_11 + df_21*df_21 + df_31*df_31

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def g_12(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_12  = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_22  = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_32  = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    return df_11*df_12 + df_21*df_22 + df_31*df_32

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def g_13(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_13  = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_23  = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_33  = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    return df_11*df_13 + df_21*df_23 + df_31*df_33

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def g_21(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_12  = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_22  = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_32  = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    return df_12*df_11 + df_22*df_21 + df_32*df_31

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def g_22(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_12  = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_22  = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_32  = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    return df_12*df_12 + df_22*df_22 + df_32*df_32

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def g_23(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_12  = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_13  = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_22  = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_23  = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_32  = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_33  = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    return df_12*df_13 + df_22*df_23 + df_32*df_33

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def g_31(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_11  = df_1(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_13  = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_21  = df_1(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_23  = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_31  = df_1(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_33  = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    return df_13*df_11 + df_23*df_21 + df_33*df_31

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def g_32(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_12  = df_2(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_13  = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    
    df_22  = df_2(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_23  = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    
    df_32  = df_2(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    df_33  = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    return df_13*df_12 + df_23*df_22 + df_33*df_32

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def g_33(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    df_13  = df_3(tn1, tn2, tn3, pn, nbase_n, cx, eta1, eta2, eta3)
    df_23  = df_3(tn1, tn2, tn3, pn, nbase_n, cy, eta1, eta2, eta3)
    df_33  = df_3(tn1, tn2, tn3, pn, nbase_n, cz, eta1, eta2, eta3)
    
    return df_13*df_13 + df_23*df_23 + df_33*df_33

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def ginv_11(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    dfinv11  = dfinv_11(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv12  = dfinv_12(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv13  = dfinv_13(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    return dfinv11*dfinv11 + dfinv12*dfinv12 + dfinv13*dfinv13

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def ginv_12(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    dfinv11  = dfinv_11(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv12  = dfinv_12(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv13  = dfinv_13(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    dfinv21  = dfinv_21(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv22  = dfinv_22(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv23  = dfinv_23(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    return dfinv11*dfinv21 + dfinv12*dfinv22 + dfinv13*dfinv23

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def ginv_13(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    dfinv11  = dfinv_11(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv12  = dfinv_12(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv13  = dfinv_13(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    dfinv31  = dfinv_31(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv32  = dfinv_32(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv33  = dfinv_33(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    return dfinv11*dfinv31 + dfinv12*dfinv32 + dfinv13*dfinv33

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def ginv_21(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    dfinv11  = dfinv_11(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv12  = dfinv_12(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv13  = dfinv_13(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    dfinv21  = dfinv_21(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv22  = dfinv_22(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv23  = dfinv_23(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    return dfinv21*dfinv11 + dfinv22*dfinv12 + dfinv23*dfinv13

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def ginv_22(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    dfinv21  = dfinv_21(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv22  = dfinv_22(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv23  = dfinv_23(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    return dfinv21*dfinv21 + dfinv22*dfinv22 + dfinv23*dfinv23

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def ginv_23(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    dfinv21  = dfinv_21(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv22  = dfinv_22(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv23  = dfinv_23(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    dfinv31  = dfinv_31(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv32  = dfinv_32(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv33  = dfinv_33(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    return dfinv21*dfinv31 + dfinv22*dfinv32 + dfinv23*dfinv33

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def ginv_31(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    dfinv11  = dfinv_11(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv12  = dfinv_12(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv13  = dfinv_13(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    dfinv31  = dfinv_31(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv32  = dfinv_32(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv33  = dfinv_33(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    return dfinv11*dfinv31 + dfinv12*dfinv32 + dfinv13*dfinv33

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def ginv_32(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    dfinv21  = dfinv_21(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv22  = dfinv_22(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv23  = dfinv_23(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    dfinv31  = dfinv_31(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv32  = dfinv_32(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv33  = dfinv_33(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    return dfinv31*dfinv21 + dfinv32*dfinv22 + dfinv33*dfinv23

# ==========================================================================
@types('double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double','double')
def ginv_33(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3):
    
    dfinv31  = dfinv_31(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv32  = dfinv_32(tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, eta1, eta2, eta3)
    dfinv33  = dfinv_33(tn1, tn2, tn3, pn, nbase_n, cx, cz, cz, eta1, eta2, eta3)
    
    return dfinv31*dfinv31 + dfinv32*dfinv32 + dfinv33*dfinv33