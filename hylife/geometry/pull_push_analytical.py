# coding: utf-8
#
# Copyright 2020 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Basic functions for analytical pull-back (physical --> logical) and push-forward (logical -- > physical) operations between scalar fields, vector fields and differential p-forms.
"""

from pyccel.decorators import types

import hylife.geometry.mappings_analytical as mapping


# ==============================================================================
@types('double','double','double','double','int','double[:]')
def pull_0_form(a, eta1, eta2, eta3, kind_map, params_map):
    
    a0 = a
    
    return a0


# ==============================================================================
@types('double','double','double','double','double','double','int','double[:]','int')
def pull_1_form(ax, ay, az, eta1, eta2, eta3, kind_map, params_map, component):
    
    if   component == 1:
        
        df_11 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 11)
        df_21 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 21)
        df_31 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 31)
        
        a = df_11 * ax + df_21 * ay + df_31 * az
    
    elif component == 2:
        
        df_12 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 12)
        df_22 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 22)
        df_32 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 32)
        
        a = df_12 * ax + df_22 * ay + df_32 * az
        
    elif component == 3:
        
        df_13 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 13)
        df_23 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 23)
        df_33 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 33)
        
        a = df_13 * ax + df_23 * ay + df_33 * az
        
    return a


# ==============================================================================
@types('double','double','double','double','double','double','int','double[:]','int')
def pull_2_form(ax, ay, az, eta1, eta2, eta3, kind_map, params_map, component):
    
    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map)
    
    if   component == 1:
        
        dfinv_11 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 11)
        dfinv_12 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 12)
        dfinv_13 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 13)
        
        a = (dfinv_11 * ax + dfinv_12 * ay + dfinv_13 * az) * abs(detdf)
    
    elif component == 2:
        
        dfinv_21 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 21)
        dfinv_22 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 22)
        dfinv_23 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 23)
        
        a = (dfinv_21 * ax + dfinv_22 * ay + dfinv_23 * az) * abs(detdf)
        
    elif component == 3:
        
        dfinv_31 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 31)
        dfinv_32 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 32)
        dfinv_33 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 33)
        
        a = (dfinv_31 * ax + dfinv_32 * ay + dfinv_33 * az) * abs(detdf)
        
    return a


# ==============================================================================
@types('double','double','double','double','int','double[:]')
def pull_3_form(a, eta1, eta2, eta3, kind_map, params_map):
    
    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map)
    
    a3 = a * abs(detdf)
    
    return a3



# ==============================================================================
@types('double','double','double','double','double','double','int','double[:]','int')
def pull_vector(ax, ay, az, eta1, eta2, eta3, kind_map, params_map, component):
    
    if   component == 1:
        
        dfinv_11 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 11)
        dfinv_12 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 12)
        dfinv_13 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 13)
        
        a = dfinv_11 * ax + dfinv_12 * ay + dfinv_13 * az
    
    elif component == 2:
        
        dfinv_21 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 21)
        dfinv_22 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 22)
        dfinv_23 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 23)
        
        a = dfinv_21 * ax + dfinv_22 * ay + dfinv_23 * az
        
    elif component == 3:
        
        dfinv_31 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 31)
        dfinv_32 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 32)
        dfinv_33 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 33)
        
        a = dfinv_31 * ax + dfinv_32 * ay + dfinv_33 * az
        
    return a


# ==============================================================================
@types('double','double','double','double','int','double[:]')
def push_0_form(a0, eta1, eta2, eta3, kind_map, params_map):
    
    a = a0
    
    return a


# ==============================================================================
@types('double','double','double','double','double','double','int','double[:]','int')
def push_1_form(a1_1, a1_2, a1_3, eta1, eta2, eta3, kind_map, params_map, component):
    
    if   component == 1:
        
        dfinv_11 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 11)
        dfinv_21 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 21)
        dfinv_31 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 31)
        
        a = dfinv_11 * a1_1 + dfinv_21 * a1_2 + dfinv_31 * a1_3
    
    elif component == 2:
        
        dfinv_12 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 12)
        dfinv_22 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 22)
        dfinv_32 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 32)
        
        a = dfinv_12 * a1_1 + dfinv_22 * a1_2 + dfinv_32 * a1_3
        
    elif component == 3:
        
        dfinv_13 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 13)
        dfinv_23 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 23)
        dfinv_33 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 33)
        
        a = dfinv_13 * a1_1 + dfinv_23 * a1_2 + dfinv_33 * a1_3
        
    return a


# ==============================================================================
@types('double','double','double','double','double','double','int','double[:]','int')
def push_2_form(a2_1, a2_2, a2_3, eta1, eta2, eta3, kind_map, params_map, component):
    
    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map)
    
    if   component == 1:
        
        df_11 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 11)
        df_12 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 12)
        df_13 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 13)
        
        a = (df_11 * a2_1 + df_12 * a2_2 + df_13 * a2_3) / abs(detdf)
    
    elif component == 2:
        
        df_21 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 21)
        df_22 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 22)
        df_23 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 23)
        
        a = (df_21 * a2_1 + df_22 * a2_2 + df_23 * a2_3) / abs(detdf)
        
    elif component == 3:
        
        df_31 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 31)
        df_32 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 32)
        df_33 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 33)
        
        a = (df_31 * a2_1 + df_32 * a2_2 + df_33 * a2_3) / abs(detdf)
        
    return a


# ==============================================================================
@types('double','double','double','double','int','double[:]')
def push_3_form(a3, eta1, eta2, eta3, kind_map, params_map):
    
    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map)
    
    a = a3 / abs(detdf)
    
    return a



# ==============================================================================
@types('double','double','double','double','double','double','int','double[:]','int')
def push_vector(a1, a2, a3, eta1, eta2, eta3, kind_map, params_map, component):
    
    if   component == 1:
        
        df_11 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 11)
        df_12 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 12)
        df_13 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 13)
        
        a = df_11 * a1 + df_12 * a2 + df_13 * a3
    
    elif component == 2:
        
        df_21 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 21)
        df_22 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 22)
        df_23 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 23)
        
        a = df_21 * a1 + df_22 * a2 + df_23 * a3
        
    elif component == 3:
        
        df_31 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 31)
        df_32 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 32)
        df_33 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 33)
        
        a = df_31 * a1 + df_32 * a2 + df_33 * a3
        
    return a


# ==============================================================================
@types('double[:, :, :]','double[:]','double[:]','double[:]','int','int','double[:]','double[:, :, :]')
def kernel_evaluation_scalar(a, eta1, eta2, eta3, kind_fun, kind_map, params_map, values):
    
    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            for i3 in range(len(eta3)):
                
                # pull-back
                if   kind_fun == 1:
                    values[i1, i2, i3] = pull_0_form(a[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map)
                elif kind_fun == 2:
                    values[i1, i2, i3] = pull_3_form(a[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map)
                
                # push-forward
                elif kind_fun == 3:
                    values[i1, i2, i3] = push_0_form(a[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map)
                elif kind_fun == 4:
                    values[i1, i2, i3] = push_3_form(a[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map)
                    
                    
# ==============================================================================
@types('int','int','int','double[:, :, :]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','int','double[:]','double[:, :, :]')
def kernel_evaluation_scalar_mat(n1, n2, n3, a, eta1, eta2, eta3, kind_fun, kind_map, params_map, values):
    
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                
                # pull-back
                if   kind_fun == 1:
                    values[i1, i2, i3] = pull_0_form(a[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map)
                elif kind_fun == 2:
                    values[i1, i2, i3] = pull_3_form(a[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map)
                
                # push-forward
                elif kind_fun == 3:
                    values[i1, i2, i3] = push_0_form(a[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map)
                elif kind_fun == 4:
                    values[i1, i2, i3] = push_3_form(a[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map)
                    
                    
# ==============================================================================
@types('double[:, :, :]','double[:, :, :]','double[:, :, :]','double[:]','double[:]','double[:]','int','int','double[:]','double[:, :, :]')
def kernel_evaluation_vector(a1, a2, a3, eta1, eta2, eta3, kind_fun, kind_map, params_map, values):
    
    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            for i3 in range(len(eta3)):
                
                # ========== pull-back ================
                
                # 1-form
                if   kind_fun == 1:
                    values[i1, i2, i3] = pull_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 1)
                elif kind_fun == 2:
                    values[i1, i2, i3] = pull_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 2)
                elif kind_fun == 3:
                    values[i1, i2, i3] = pull_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 3)
                    
                # 2-form
                elif kind_fun == 4:
                    values[i1, i2, i3] = pull_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 1)
                elif kind_fun == 5:
                    values[i1, i2, i3] = pull_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 2)
                elif kind_fun == 6:
                    values[i1, i2, i3] = pull_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 3)
                    
                # vector
                elif kind_fun == 7:
                    values[i1, i2, i3] = pull_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 1)
                elif kind_fun == 8:
                    values[i1, i2, i3] = pull_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 2)
                elif kind_fun == 9:
                    values[i1, i2, i3] = pull_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 3)
                    
                # ========== push-forward ================
                
                # 1-form
                if   kind_fun == 10:
                    values[i1, i2, i3] = push_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 1)
                elif kind_fun == 11:
                    values[i1, i2, i3] = push_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 2)
                elif kind_fun == 12:
                    values[i1, i2, i3] = push_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 3)
                    
                # 2-form
                elif kind_fun == 13:
                    values[i1, i2, i3] = push_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 1)
                elif kind_fun == 14:
                    values[i1, i2, i3] = push_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 2)
                elif kind_fun == 15:
                    values[i1, i2, i3] = push_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 3)
                    
                # vector
                elif kind_fun == 16:
                    values[i1, i2, i3] = push_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 1)
                elif kind_fun == 17:
                    values[i1, i2, i3] = push_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 2)
                elif kind_fun == 18:
                    values[i1, i2, i3] = push_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], kind_map, params_map, 3)
                    
                    
                    
# ==============================================================================
@types('int','int','int','double[:, :, :]','double[:, :, :]','double[:, :, :]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','int','double[:]','double[:, :, :]')
def kernel_evaluation_vector_mat(n1, n2, n3, a1, a2, a3, eta1, eta2, eta3, kind_fun, kind_map, params_map, values):
    
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                
                # ========== pull-back ================
                
                # 1-form
                if   kind_fun == 1:
                    values[i1, i2, i3] = pull_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 1)
                elif kind_fun == 2:
                    values[i1, i2, i3] = pull_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 2)
                elif kind_fun == 3:
                    values[i1, i2, i3] = pull_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 3)
                    
                # 2-form
                elif kind_fun == 4:
                    values[i1, i2, i3] = pull_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 1)
                elif kind_fun == 5:
                    values[i1, i2, i3] = pull_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 2)
                elif kind_fun == 6:
                    values[i1, i2, i3] = pull_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 3)
                    
                # vector
                elif kind_fun == 7:
                    values[i1, i2, i3] = pull_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 1)
                elif kind_fun == 8:
                    values[i1, i2, i3] = pull_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 2)
                elif kind_fun == 9:
                    values[i1, i2, i3] = pull_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 3)
                    
                # ========== push-forward ================
                
                # 1-form
                if   kind_fun == 10:
                    values[i1, i2, i3] = push_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 1)
                elif kind_fun == 11:
                    values[i1, i2, i3] = push_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 2)
                elif kind_fun == 12:
                    values[i1, i2, i3] = push_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 3)
                    
                # 2-form
                elif kind_fun == 13:
                    values[i1, i2, i3] = push_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 1)
                elif kind_fun == 14:
                    values[i1, i2, i3] = push_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 2)
                elif kind_fun == 15:
                    values[i1, i2, i3] = push_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 3)
                    
                # vector
                elif kind_fun == 16:
                    values[i1, i2, i3] = push_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 1)
                elif kind_fun == 17:
                    values[i1, i2, i3] = push_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 2)
                elif kind_fun == 18:
                    values[i1, i2, i3] = push_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_map, params_map, 3)