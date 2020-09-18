# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic functions for pulling-back scalar and vector fields on the physical domain to differential p-forms on the logical domain via an analytical mapping.
"""

from pyccel.decorators import types

import hylife.geometry.mappings_analytical as mapping


# ==============================================================================
@types('double','double','double','double','int','double[:]')
def pull_0_form(a, eta1, eta2, eta3, kind_map, params_map):
    
    value = a
    
    return value


# ==============================================================================
@types('double','double','double','double','double','double','int','double[:]','int')
def pull_1_form(ax, ay, az, eta1, eta2, eta3, kind_map, params_map, component):
    
    if   component == 1:
        
        df_11 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 11)
        df_21 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 21)
        df_31 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 31)
        
        value = df_11 * ax + df_21 * ay + df_31 * az
    
    elif component == 2:
        
        df_12 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 12)
        df_22 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 22)
        df_32 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 32)
        
        value = df_12 * ax + df_22 * ay + df_32 * az
        
    elif component == 3:
        
        df_13 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 13)
        df_23 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 23)
        df_33 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 33)
        
        value = df_13 * ax + df_23 * ay + df_33 * az
        
    return value


# ==============================================================================
@types('double','double','double','double','double','double','int','double[:]','int')
def pull_2_form(ax, ay, az, eta1, eta2, eta3, kind_map, params_map, component):
    
    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map)
    
    if   component == 1:
        
        dfinv_11 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 11)
        dfinv_12 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 12)
        dfinv_13 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 13)
        
        value = (dfinv_11 * ax + dfinv_12 * ay + dfinv_13 * az) * abs(detdf)
    
    elif component == 2:
        
        dfinv_21 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 21)
        dfinv_22 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 22)
        dfinv_23 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 23)
        
        value = (dfinv_21 * ax + dfinv_22 * ay + dfinv_23 * az) * abs(detdf)
        
    elif component == 3:
        
        dfinv_31 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 31)
        dfinv_32 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 32)
        dfinv_33 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 33)
        
        value = (dfinv_31 * ax + dfinv_32 * ay + dfinv_33 * az) * abs(detdf)
        
    return value


# ==============================================================================
@types('double','double','double','double','int','double[:]')
def pull_3_form(a, eta1, eta2, eta3, kind_map, params_map):
    
    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map)
    
    value  = a * abs(detdf)
    
    return value



# ==============================================================================
@types('double','double','double','double','double','double','int','double[:]','int')
def pull_vector(ax, ay, az, eta1, eta2, eta3, kind_map, params_map, component):
    
    if   component == 1:
        
        dfinv_11 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 11)
        dfinv_12 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 12)
        dfinv_13 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 13)
        
        value = dfinv_11 * ax + dfinv_12 * ay + dfinv_13 * az
    
    elif component == 2:
        
        dfinv_21 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 21)
        dfinv_22 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 22)
        dfinv_23 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 23)
        
        value = dfinv_21 * ax + dfinv_22 * ay + dfinv_23 * az
        
    elif component == 3:
        
        dfinv_31 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 31)
        dfinv_32 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 32)
        dfinv_33 = mapping.df_inv(eta1, eta2, eta3, kind_map, params_map, 33)
        
        value = dfinv_31 * ax + dfinv_32 * ay + dfinv_33 * az
        
    return value