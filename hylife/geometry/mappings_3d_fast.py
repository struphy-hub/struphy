# coding: utf-8
#
# Copnyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Efficient modules for point-wise evaluation of a 3d analytical (kind_map >= 1) or discrete (kind_map = 0) B-spline mapping.
Especially suited for PIC routines since it avoids computing the Jacobian matrix multiple times.
"""

from pyccel.decorators import types

import hylife.geometry.mappings_3d as mapping

import hylife.utilitis_FEEC.bsplines_kernels as bsp
import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva

from numpy import empty, zeros


# ==========================================================================
@types('int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int','int','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:]','double[:]','double[:]','double[:]','double[:]','double[:]','double[:,:]','double[:,:]','double[:,:]','double[:]','double[:]','double[:]','double[:]','double[:]','double[:]','double','double','double','double[:,:]')
def df_all(kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, span_n1, span_n2, span_n3, cx, cy, cz, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, der1, der2, der3, eta1, eta2, eta3, mat_out):
    
    # discrete mapping
    if kind_map == 0:
        
        # evaluate non-vanishing basis functions and its derivatives
        bsp.basis_funs_and_der(tn1, pn[0], eta1, span_n1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(tn2, pn[1], eta2, span_n2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(tn3, pn[2], eta3, span_n3, l3, r3, b3, d3, der3)

        # sum-up non-vanishing contributions (line 1: df_11, df_12 and df_13)
        mat_out[0, 0] = eva.evaluation_kernel(pn[0], pn[1], pn[2], der1, b2[pn[1]], b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cx)
        mat_out[0, 1] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[pn[0]], der2, b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cx)
        mat_out[0, 2] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], der3, span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cx)

        # sum-up non-vanishing contributions (line 2: df_21, df_22 and df_23)
        mat_out[1, 0] = eva.evaluation_kernel(pn[0], pn[1], pn[2], der1, b2[pn[1]], b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cy)
        mat_out[1, 1] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[pn[0]], der2, b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cy)
        mat_out[1, 2] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], der3, span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cy)

        # sum-up non-vanishing contributions (line 3: df_31, df_32 and df_33)
        mat_out[2, 0] = eva.evaluation_kernel(pn[0], pn[1], pn[2], der1, b2[pn[1]], b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cz)
        mat_out[2, 1] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[pn[0]], der2, b3[pn[2]], span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cz)
        mat_out[2, 2] = eva.evaluation_kernel(pn[0], pn[1], pn[2], b1[pn[0]], b2[pn[1]], der3, span_n1, span_n2, span_n3, nbase_n[0], nbase_n[1], nbase_n[2], cz)
        
    # analytical mapping
    else:
        mat_out[0, 0] = mapping.df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        mat_out[0, 1] = mapping.df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        mat_out[0, 2] = mapping.df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        mat_out[1, 0] = mapping.df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        mat_out[1, 1] = mapping.df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        mat_out[1, 2] = mapping.df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        mat_out[2, 0] = mapping.df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        mat_out[2, 1] = mapping.df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        mat_out[2, 2] = mapping.df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        
# ===========================================================================
@types('double[:,:]','double[:,:]')
def df_inv_all(mat_in, mat_out):
    
    # inverse Jacobian determinant computed from Jacobian matrix
    over_det_df = 1. / (mat_in[0, 0]*(mat_in[1, 1]*mat_in[2, 2] - mat_in[2, 1]*mat_in[1, 2]) + mat_in[1, 0]*(mat_in[2, 1]*mat_in[0, 2] - mat_in[0, 1]*mat_in[2, 2]) + mat_in[2, 0]*(mat_in[0, 1]*mat_in[1, 2] - mat_in[1, 1]*mat_in[0, 2]))

    # inverse Jacobian matrix computed from Jacobian matrix
    mat_out[0, 0] = (mat_in[1, 1]*mat_in[2, 2] - mat_in[1, 2]*mat_in[2, 1]) * over_det_df
    mat_out[0, 1] = (mat_in[0, 2]*mat_in[2, 1] - mat_in[0, 1]*mat_in[2, 2]) * over_det_df
    mat_out[0, 2] = (mat_in[0, 1]*mat_in[1, 2] - mat_in[0, 2]*mat_in[1, 1]) * over_det_df

    mat_out[1, 0] = (mat_in[1, 2]*mat_in[2, 0] - mat_in[1, 0]*mat_in[2, 2]) * over_det_df
    mat_out[1, 1] = (mat_in[0, 0]*mat_in[2, 2] - mat_in[0, 2]*mat_in[2, 0]) * over_det_df
    mat_out[1, 2] = (mat_in[0, 2]*mat_in[1, 0] - mat_in[0, 0]*mat_in[1, 2]) * over_det_df

    mat_out[2, 0] = (mat_in[1, 0]*mat_in[2, 1] - mat_in[1, 1]*mat_in[2, 0]) * over_det_df
    mat_out[2, 1] = (mat_in[0, 1]*mat_in[0, 2] - mat_in[0, 0]*mat_in[2, 1]) * over_det_df
    mat_out[2, 2] = (mat_in[0, 0]*mat_in[1, 1] - mat_in[0, 1]*mat_in[1, 0]) * over_det_df
    
    
# ===========================================================================
@types('double[:,:]','double[:,:]')
def g_all(mat_in, mat_out):
    
    # metric tensor computed from Jacobian matrix
    mat_out[0, 0] = mat_in[0, 0]*mat_in[0, 0] + mat_in[1, 0]*mat_in[1, 0] + mat_in[2, 0]*mat_in[2, 0]
    mat_out[0, 1] = mat_in[0, 0]*mat_in[0, 1] + mat_in[1, 0]*mat_in[1, 1] + mat_in[2, 0]*mat_in[2, 1]
    mat_out[0, 2] = mat_in[0, 0]*mat_in[0, 2] + mat_in[1, 2]*mat_in[1, 2] + mat_in[2, 0]*mat_in[2, 2]

    mat_out[1, 0] = mat_out[0, 1]
    mat_out[1, 1] = mat_in[0, 1]*mat_in[0, 1] + mat_in[1, 1]*mat_in[1, 1] + mat_in[2, 1]*mat_in[2, 1]
    mat_out[1, 2] = mat_in[0, 1]*mat_in[0, 2] + mat_in[1, 0]*mat_in[1, 2] + mat_in[2, 0]*mat_in[2, 2]

    mat_out[2, 0] = mat_out[0, 2]
    mat_out[2, 1] = mat_out[1, 2]
    mat_out[2, 2] = mat_in[0, 2]*mat_in[0, 2] + mat_in[1, 2]*mat_in[1, 2] + mat_in[2, 2]*mat_in[2, 2]
    
    
# ===========================================================================
@types('double[:,:]','double[:,:]')
def g_inv_all(mat_in, mat_out):
    
    # inverse metric tensor computed from inverse Jacobian matrix
    mat_out[0, 0] = mat_in[0, 0]*mat_in[0, 0] + mat_in[0, 1]*mat_in[0, 1] + mat_in[0, 2]*mat_in[0, 2]
    mat_out[0, 1] = mat_in[0, 0]*mat_in[1, 0] + mat_in[0, 1]*mat_in[1, 1] + mat_in[0, 2]*mat_in[1, 2]
    mat_out[0, 2] = mat_in[0, 0]*mat_in[2, 0] + mat_in[0, 1]*mat_in[2, 1] + mat_in[0, 2]*mat_in[2, 2]

    mat_out[1, 0] = mat_out[0, 1]
    mat_out[1, 1] = mat_in[1, 0]*mat_in[1, 0] + mat_in[1, 1]*mat_in[1, 1] + mat_in[1, 2]*mat_in[1, 2]
    mat_out[1, 2] = mat_in[1, 0]*mat_in[2, 0] + mat_in[1, 1]*mat_in[2, 1] + mat_in[1, 2]*mat_in[2, 2]

    mat_out[2, 0] = mat_out[0, 2]
    mat_out[2, 1] = mat_out[1, 2]
    mat_out[2, 2] = mat_in[2, 0]*mat_in[2, 0] + mat_in[2, 1]*mat_in[2, 1] + mat_in[2, 2]*mat_in[2, 2]