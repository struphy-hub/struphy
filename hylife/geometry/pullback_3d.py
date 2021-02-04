# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Basic pull-back (physical --> logical) operations between scalar fields, vector fields and differential p-forms:

- 0-form: a^0 = a
- 1-form: (a^1_1, a^1_2, a^1_3) = DF^T (ax, ay, az)
- 2-form: (a^2_1, a^2_2, a^2_3) = |det(DF)| DF^(-1) (ax, ay, az)
- 3-form: a^3 = |det(DF)| a
- vector: (a_1, a_2, a_3) = DF^(-1) (ax, ay, az)
"""

from numpy import shape

from pyccel.decorators import types

import hylife.geometry.mappings_3d as mapping


# ==============================================================================
@types('double','double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pull_0_form(a, eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    a0 = a
    
    return a0


# ==============================================================================
@types('double','double','double','double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pull_1_form(ax, ay, az, eta1, eta2, eta3, component, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    if   component == 1:
        
        df_11 = mapping.df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_21 = mapping.df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_31 = mapping.df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = df_11*ax + df_21*ay + df_31*az
    
    elif component == 2:
        
        df_12 = mapping.df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_22 = mapping.df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_32 = mapping.df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = df_12*ax + df_22*ay + df_32*az
        
    elif component == 3:
        
        df_13 = mapping.df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_23 = mapping.df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_33 = mapping.df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = df_13*ax + df_23*ay + df_33*az
        
    return a


# ==============================================================================
@types('double','double','double','double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pull_2_form(ax, ay, az, eta1, eta2, eta3, component, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    if   component == 1:
        
        dfinv_11 = mapping.df_inv(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_12 = mapping.df_inv(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_13 = mapping.df_inv(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (dfinv_11*ax + dfinv_12*ay + dfinv_13*az) * abs(detdf)
    
    elif component == 2:
        
        dfinv_21 = mapping.df_inv(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_22 = mapping.df_inv(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_23 = mapping.df_inv(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (dfinv_21*ax + dfinv_22*ay + dfinv_23*az) * abs(detdf)
        
    elif component == 3:
        
        dfinv_31 = mapping.df_inv(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_32 = mapping.df_inv(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_33 = mapping.df_inv(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (dfinv_31*ax + dfinv_32*ay + dfinv_33*az) * abs(detdf)
        
    return a


# ==============================================================================
@types('double','double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pull_3_form(a, eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    a3 = a * abs(detdf)
    
    return a3



# ==============================================================================
@types('double','double','double','double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pull_vector(ax, ay, az, eta1, eta2, eta3, component, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    if   component == 1:
        
        dfinv_11 = mapping.df_inv(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_12 = mapping.df_inv(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_13 = mapping.df_inv(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = dfinv_11*ax + dfinv_12*ay + dfinv_13*az
    
    elif component == 2:
        
        dfinv_21 = mapping.df_inv(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_22 = mapping.df_inv(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_23 = mapping.df_inv(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = dfinv_21*ax + dfinv_22*ay + dfinv_23*az
        
    elif component == 3:
        
        dfinv_31 = mapping.df_inv(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_32 = mapping.df_inv(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_33 = mapping.df_inv(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = dfinv_31*ax + dfinv_32*ay + dfinv_33*az
        
    return a


# ==============================================================================
@types('double[:,:,:]','double[:]','double[:]','double[:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def kernel_evaluate_tensor_scalar(a, eta1, eta2, eta3, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, values):
    
    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            for i3 in range(len(eta3)):
                
                # 0-form
                if   kind_fun == 1:
                    values[i1, i2, i3] = pull_0_form(a[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                # 3-form
                elif kind_fun == 2:
                    values[i1, i2, i3] = pull_3_form(a[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                    
                    
# ==============================================================================
@types('double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def kernel_evaluate_general_scalar(a, eta1, eta2, eta3, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, values):
    
    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]
    
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                
                # 0-form
                if   kind_fun == 1:
                    values[i1, i2, i3] = pull_0_form(a[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                # 3-form
                elif kind_fun == 2:
                    values[i1, i2, i3] = pull_3_form(a[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                    
                    
# ==============================================================================
@types('double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:]','double[:]','double[:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def kernel_evaluate_tensor_vector(ax, ay, az, eta1, eta2, eta3, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, values):
    
    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            for i3 in range(len(eta3)):
                
                # 1-form
                if   kind_fun == 1:
                    values[i1, i2, i3] = pull_1_form(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                elif kind_fun == 2:
                    values[i1, i2, i3] = pull_1_form(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                elif kind_fun == 3:
                    values[i1, i2, i3] = pull_1_form(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                    
                # 2-form
                elif kind_fun == 4:
                    values[i1, i2, i3] = pull_2_form(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                elif kind_fun == 5:
                    values[i1, i2, i3] = pull_2_form(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                elif kind_fun == 6:
                    values[i1, i2, i3] = pull_2_form(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                    
                # vector
                elif kind_fun == 7:
                    values[i1, i2, i3] = pull_vector(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                elif kind_fun == 8:
                    values[i1, i2, i3] = pull_vector(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                elif kind_fun == 9:
                    values[i1, i2, i3] = pull_vector(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                    
                    
# ==============================================================================
@types('double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def kernel_evaluate_general_vector(ax, ay, az, eta1, eta2, eta3, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, values):
    
    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]
    
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                
                # 1-form
                if   kind_fun == 1:
                    values[i1, i2, i3] = pull_1_form(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                elif kind_fun == 2:
                    values[i1, i2, i3] = pull_1_form(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                elif kind_fun == 3:
                    values[i1, i2, i3] = pull_1_form(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                    
                # 2-form
                elif kind_fun == 4:
                    values[i1, i2, i3] = pull_2_form(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                elif kind_fun == 5:
                    values[i1, i2, i3] = pull_2_form(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                elif kind_fun == 6:
                    values[i1, i2, i3] = pull_2_form(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                    
                # vector
                elif kind_fun == 7:
                    values[i1, i2, i3] = pull_vector(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                elif kind_fun == 8:
                    values[i1, i2, i3] = pull_vector(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                elif kind_fun == 9:
                    values[i1, i2, i3] = pull_vector(ax[i1, i2, i3], ay[i1, i2, i3], az[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)