# coding: utf-8
#
# Copyright 2020 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Basic functions for discrete (spline mapping) pull-back (physical --> logical) and push-forward (logical -- > physical) operations between scalar fields, vector fields and differential p-forms.
"""

from pyccel.decorators import types

import hylife.geometry.mappings_discrete_3d as mapping


# ==============================================================================
@types('double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pull_0_form(a, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    a0 = a
    
    return a0


# ==============================================================================
@types('double','double','double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int')
def pull_1_form(ax, ay, az, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, component):
    
    if   component == 1:
        
        df_11 = mapping.df_1(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        df_21 = mapping.df_1(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        df_31 = mapping.df_1(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
        
        a = df_11 * ax + df_21 * ay + df_31 * az
    
    elif component == 2:
        
        df_12 = mapping.df_2(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        df_22 = mapping.df_2(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        df_32 = mapping.df_2(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
        
        a = df_12 * ax + df_22 * ay + df_32 * az
        
    elif component == 3:
        
        df_13 = mapping.df_3(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        df_23 = mapping.df_3(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        df_33 = mapping.df_3(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
        
        a = df_13 * ax + df_23 * ay + df_33 * az
        
    return a


# ==============================================================================
@types('double','double','double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int')
def pull_2_form(ax, ay, az, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, component):
    
    detdf = mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    
    if   component == 1:
        
        dfinv11 = mapping.dfinv_11(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv12 = mapping.dfinv_12(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv13 = mapping.dfinv_13(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        a = (dfinv11 * ax + dfinv12 * ay + dfinv13 * az) * abs(detdf)
    
    elif component == 2:
        
        dfinv21 = mapping.dfinv_21(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv22 = mapping.dfinv_22(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv23 = mapping.dfinv_23(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        a = (dfinv21 * ax + dfinv22 * ay + dfinv23 * az) * abs(detdf)
        
    elif component == 3:
        
        dfinv31 = mapping.dfinv_31(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv32 = mapping.dfinv_32(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv33 = mapping.dfinv_33(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        a = (dfinv31 * ax + dfinv32 * ay + dfinv33 * az) * abs(detdf)
        
    return a


# ==============================================================================
@types('double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pull_3_form(a, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    detdf = mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    
    a3 = a * abs(detdf)
    
    return a3


# ==============================================================================
@types('double','double','double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int')
def pull_vector(ax, ay, az, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, component):
    
    if   component == 1:
        
        dfinv11 = mapping.dfinv_11(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv12 = mapping.dfinv_12(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv13 = mapping.dfinv_13(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        a = dfinv11 * ax + dfinv12 * ay + dfinv13 * az
    
    elif component == 2:
        
        dfinv21 = mapping.dfinv_21(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv22 = mapping.dfinv_22(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv23 = mapping.dfinv_23(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        a = dfinv21 * ax + dfinv22 * ay + dfinv23 * az
        
    elif component == 3:
        
        dfinv31 = mapping.dfinv_31(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv32 = mapping.dfinv_32(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv33 = mapping.dfinv_33(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        a = dfinv31 * ax + dfinv32 * ay + dfinv33 * az
        
    return a


# ==============================================================================
@types('double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def push_0_form(a0, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    a = a0
    
    return a


# ==============================================================================
@types('double','double','double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int')
def push_1_form(a1_1, a1_2, a1_3, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, component):
    
    if   component == 1:
        
        dfinv11 = mapping.dfinv_11(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv21 = mapping.dfinv_21(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv31 = mapping.dfinv_31(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        a = dfinv11 * a1_1 + dfinv21 * a1_2 + dfinv31 * a1_3
    
    elif component == 2:
        
        dfinv12 = mapping.dfinv_12(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv22 = mapping.dfinv_22(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv32 = mapping.dfinv_32(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        a = dfinv12 * a1_1 + dfinv22 * a1_2 + dfinv32 * a1_3
        
    elif component == 3:
        
        dfinv13 = mapping.dfinv_13(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv23 = mapping.dfinv_23(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv33 = mapping.dfinv_33(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        a = dfinv13 * a1_1 + dfinv23 * a1_2 + dfinv33 * a1_3
        
    return a


# ==============================================================================
@types('double','double','double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int')
def push_2_form(a2_1, a2_2, a2_3, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, component):
    
    detdf = mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    
    if   component == 1:
        
        df11 = mapping.df_1(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        df12 = mapping.df_2(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        df13 = mapping.df_3(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        
        a = (df11 * a2_1 + df12 * a2_2 + df13 * a2_3) / abs(detdf)
    
    elif component == 2:
        
        df21 = mapping.df_1(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        df22 = mapping.df_2(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        df23 = mapping.df_3(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        
        a = (df21 * a2_1 + df22 * a2_2 + df23 * a2_3) / abs(detdf)
        
    elif component == 3:
        
        df31 = mapping.df_1(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
        df32 = mapping.df_2(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
        df33 = mapping.df_3(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
        
        a = (df31 * a2_1 + df32 * a2_2 + df33 * a2_3) / abs(detdf)
        
    return a


# ==============================================================================
@types('double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def push_3_form(a3, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    detdf = mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    
    a = a3 / abs(detdf)
    
    return a



# ==============================================================================
@types('double','double','double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int')
def push_vector(a1, a2, a3, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, component):
    
    if   component == 1:
        
        df11 = mapping.df_1(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        df12 = mapping.df_2(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        df13 = mapping.df_3(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        
        a = df11 * a1 + df12 * a2 + df13 * a3
    
    elif component == 2:
        
        df21 = mapping.df_1(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        df22 = mapping.df_2(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        df23 = mapping.df_3(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        
        a = df21 * a1 + df22 * a2 + df23 * a3
        
    elif component == 3:
        
        df31 = mapping.df_1(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
        df32 = mapping.df_2(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
        df33 = mapping.df_3(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
        
        a = df31 * a1 + df32 * a2 + df33 * a3
        
    return a


# ==============================================================================
@types('double[:, :, :]','double[:]','double[:]','double[:]','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:, :, :]')
def kernel_evaluation_scalar(a, eta1, eta2, eta3, kind_fun, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, values):
    
    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            for i3 in range(len(eta3)):
                
                # pull-back
                if   kind_fun == 1:
                    values[i1, i2, i3] = pull_0_form(a[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
                elif kind_fun == 2:
                    values[i1, i2, i3] = pull_3_form(a[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
                
                # push-forward
                elif kind_fun == 3:
                    values[i1, i2, i3] = push_0_form(a[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
                elif kind_fun == 4:
                    values[i1, i2, i3] = push_3_form(a[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
                    
                    
# ==============================================================================
@types('int','int','int','double[:, :, :]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:, :, :]')
def kernel_evaluation_scalar_mat(n1, n2, n3, a, eta1, eta2, eta3, kind_fun, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, values):
    
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                
                # pull-back
                if   kind_fun == 1:
                    values[i1, i2, i3] = pull_0_form(a[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
                elif kind_fun == 2:
                    values[i1, i2, i3] = pull_3_form(a[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
                
                # push-forward
                elif kind_fun == 3:
                    values[i1, i2, i3] = push_0_form(a[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
                elif kind_fun == 4:
                    values[i1, i2, i3] = push_3_form(a[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
                    
                    
# ==============================================================================
@types('double[:, :, :]','double[:, :, :]','double[:, :, :]','double[:]','double[:]','double[:]','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:, :, :]')
def kernel_evaluation_vector(a1, a2, a3, eta1, eta2, eta3, kind_fun, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, values):
    
    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            for i3 in range(len(eta3)):
                
                # ========== pull-back ================
                
                # 1-form
                if   kind_fun == 1:
                    values[i1, i2, i3] = pull_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)
                elif kind_fun == 2:
                    values[i1, i2, i3] = pull_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)
                elif kind_fun == 3:
                    values[i1, i2, i3] = pull_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)
                    
                # 2-form
                elif kind_fun == 4:
                    values[i1, i2, i3] = pull_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)
                elif kind_fun == 5:
                    values[i1, i2, i3] = pull_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)
                elif kind_fun == 6:
                    values[i1, i2, i3] = pull_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)
                    
                # vector
                elif kind_fun == 7:
                    values[i1, i2, i3] = pull_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)
                elif kind_fun == 8:
                    values[i1, i2, i3] = pull_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)
                elif kind_fun == 9:
                    values[i1, i2, i3] = pull_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)
                    
                # ========== push-forward ================
                
                # 1-form
                if   kind_fun == 10:
                    values[i1, i2, i3] = push_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)
                elif kind_fun == 11:
                    values[i1, i2, i3] = push_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)
                elif kind_fun == 12:
                    values[i1, i2, i3] = push_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)
                    
                # 2-form
                elif kind_fun == 13:
                    values[i1, i2, i3] = push_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)
                elif kind_fun == 14:
                    values[i1, i2, i3] = push_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)
                elif kind_fun == 15:
                    values[i1, i2, i3] = push_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)
                    
                # vector
                elif kind_fun == 16:
                    values[i1, i2, i3] = push_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)
                elif kind_fun == 17:
                    values[i1, i2, i3] = push_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)
                elif kind_fun == 18:
                    values[i1, i2, i3] = push_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1], eta2[i2], eta3[i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)
                    
                    
                    
# ==============================================================================
@types('int','int','int','double[:, :, :]','double[:, :, :]','double[:, :, :]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:, :, :]')
def kernel_evaluation_vector_mat(n1, n2, n3, a1, a2, a3, eta1, eta2, eta3, kind_fun, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, values):
    
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                
                # ========== pull-back ================
                
                # 1-form
                if   kind_fun == 1:
                    values[i1, i2, i3] = pull_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)
                elif kind_fun == 2:
                    values[i1, i2, i3] = pull_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)
                elif kind_fun == 3:
                    values[i1, i2, i3] = pull_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)
                    
                # 2-form
                elif kind_fun == 4:
                    values[i1, i2, i3] = pull_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)
                elif kind_fun == 5:
                    values[i1, i2, i3] = pull_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)
                elif kind_fun == 6:
                    values[i1, i2, i3] = pull_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)
                    
                # vector
                elif kind_fun == 7:
                    values[i1, i2, i3] = pull_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)
                elif kind_fun == 8:
                    values[i1, i2, i3] = pull_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)
                elif kind_fun == 9:
                    values[i1, i2, i3] = pull_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)
                    
                # ========== push-forward ================
                
                # 1-form
                if   kind_fun == 10:
                    values[i1, i2, i3] = push_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)
                elif kind_fun == 11:
                    values[i1, i2, i3] = push_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)
                elif kind_fun == 12:
                    values[i1, i2, i3] = push_1_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)
                    
                # 2-form
                elif kind_fun == 13:
                    values[i1, i2, i3] = push_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)
                elif kind_fun == 14:
                    values[i1, i2, i3] = push_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)
                elif kind_fun == 15:
                    values[i1, i2, i3] = push_2_form(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)
                    
                # vector
                elif kind_fun == 16:
                    values[i1, i2, i3] = push_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)
                elif kind_fun == 17:
                    values[i1, i2, i3] = push_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)
                elif kind_fun == 18:
                    values[i1, i2, i3] = push_vector(a1[i1, i2, i3], a2[i1, i2, i3], a3[i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)