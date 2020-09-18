# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic functions for pulling-back scalar and vector fields on the physical domain to differential p-forms on the logical domain via a discrete spline mapping.
"""

from pyccel.decorators import types

import hylife.geometry.mappings_discrete as mapping


# ==============================================================================
@types('double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pull_0_form(a, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    value = a
    
    return value


# ==============================================================================
@types('double','double','double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int')
def pull_1_form(ax, ay, az, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, component):
    
    if   component == 1:
        
        df_11 = mapping.df_1(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        df_21 = mapping.df_1(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        df_31 = mapping.df_1(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
        
        value = df_11 * ax + df_21 * ay + df_31 * az
    
    elif component == 2:
        
        df_12 = mapping.df_2(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        df_22 = mapping.df_2(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        df_32 = mapping.df_2(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
        
        value = df_12 * ax + df_22 * ay + df_32 * az
        
    elif component == 3:
        
        df_13 = mapping.df_3(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        df_23 = mapping.df_3(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        df_33 = mapping.df_3(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
        
        value = df_13 * ax + df_23 * ay + df_33 * az
        
    return value


# ==============================================================================
@types('double','double','double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int')
def pull_2_form(ax, ay, az, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, component):
    
    detdf = mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    
    if   component == 1:
        
        dfinv11 = mapping.dfinv_11(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv12 = mapping.dfinv_12(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv13 = mapping.dfinv_13(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        value = (dfinv11 * ax + dfinv12 * ay + dfinv13 * az) * abs(detdf)
    
    elif component == 2:
        
        dfinv21 = mapping.dfinv_21(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv22 = mapping.dfinv_22(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv23 = mapping.dfinv_23(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        value = (dfinv21 * ax + dfinv22 * ay + dfinv23 * az) * abs(detdf)
        
    elif component == 3:
        
        dfinv31 = mapping.dfinv_31(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv32 = mapping.dfinv_32(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv33 = mapping.dfinv_33(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        value = (dfinv31 * ax + dfinv32 * ay + dfinv33 * az) * abs(detdf)
        
    return value


# ==============================================================================
@types('double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pull_3_form(a, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    detdf = mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    
    value  = a * abs(detdf)
    
    return value


# ==============================================================================
@types('double','double','double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int')
def pull_vector(ax, ay, az, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, component):
    
    if   component == 1:
        
        dfinv11 = mapping.dfinv_11(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv12 = mapping.dfinv_12(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv13 = mapping.dfinv_13(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        value = dfinv11 * ax + dfinv12 * ay + dfinv13 * az
    
    elif component == 2:
        
        dfinv21 = mapping.dfinv_21(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv22 = mapping.dfinv_22(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv23 = mapping.dfinv_23(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        value = dfinv21 * ax + dfinv22 * ay + dfinv23 * az
        
    elif component == 3:
        
        dfinv31 = mapping.dfinv_31(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv32 = mapping.dfinv_32(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv33 = mapping.dfinv_33(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        value = dfinv31 * ax + dfinv32 * ay + dfinv33 * az
        
    return value