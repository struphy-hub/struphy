# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Basic push-forward (logical --> physical) operations between scalar fields, vector fields and differential p-forms:

- 0-form:  a           = a^0
- 1-form: (ax, ay, az) =             DF^(-T) (a^1_1, a^1_2, a^1_3)
- 2-form: (ax, ay, az) = 1/|det(DF)| DF      (a^2_1, a^2_2, a^2_3)
- 3-form:  a           = 1/|det(DF)| a^3

- vector: (ax, ay, az) =             DF      (a_1  , a_2  , a_3  )  
"""

from numpy import shape

from struphy.geometry.mappings_3d import df, df_inv, det_df


# ==============================================================================
def push_0_form(a0 : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    
    a = a0
    
    return a


# ==============================================================================
def push_1_form(a1_1 : 'double', a1_2 : 'double', a1_3 : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    
    if   component == 1:
        
        dfinv_11 = df_inv(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, 1)
        dfinv_21 = df_inv(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, 1)
        dfinv_31 = df_inv(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, 1)
        
        a = dfinv_11*a1_1 + dfinv_21*a1_2 + dfinv_31*a1_3
    
    elif component == 2:
        
        dfinv_12 = df_inv(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, 1)
        dfinv_22 = df_inv(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, 1)
        dfinv_32 = df_inv(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, 1)
        
        a = dfinv_12*a1_1 + dfinv_22*a1_2 + dfinv_32*a1_3
        
    elif component == 3:
        
        dfinv_13 = df_inv(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, 1)
        dfinv_23 = df_inv(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, 1)
        dfinv_33 = df_inv(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, 1)
        
        a = dfinv_13*a1_1 + dfinv_23*a1_2 + dfinv_33*a1_3
        
    return a


# ==============================================================================
def push_2_form(a2_1 : 'double', a2_2 : 'double', a2_3 : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    if   component == 1:
        
        df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (df_11*a2_1 + df_12*a2_2 + df_13*a2_3) / abs(detdf)
    
    elif component == 2:
        
        df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (df_21*a2_1 + df_22*a2_2 + df_23*a2_3) / abs(detdf)
        
    elif component == 3:
        
        df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (df_31*a2_1 + df_32*a2_2 + df_33*a2_3) / abs(detdf)
        
    return a


# ==============================================================================
def push_3_form(a3 : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    a = a3 / abs(detdf)
    
    return a


# ==============================================================================
def push_vector(a1 : 'double', a2 : 'double', a3 : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    
    if   component == 1:
        
        df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = df_11*a1 + df_12*a2 + df_13*a3
    
    elif component == 2:
        
        df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = df_21*a1 + df_22*a2 + df_23*a3
        
    elif component == 3:
        
        df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = df_31*a1 + df_32*a2 + df_33*a3
        
    return a


# ==============================================================================
def push_all(a : 'double[:]', eta1 : 'double', eta2 : 'double', eta3 : 'double', kind_fun : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':

    value = 0.

    # 0-form
    if   kind_fun == 0:
        value = push_0_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    # 3-form
    elif kind_fun == 3:
        value = push_3_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    # 1-form
    elif kind_fun == 11:
        value = push_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 12:
        value = push_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 13:
        value = push_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    # 2-form
    elif kind_fun == 21:
        value = push_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 22:
        value = push_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 23:
        value = push_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    # vector
    elif kind_fun == 31:
        value = push_vector(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 32:
        value = push_vector(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 33:
        value = push_vector(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    return value


# ==============================================================================
def kernel_evaluate(a : 'double[:,:,:,:]', eta1 : 'double[:,:,:]', eta2 : 'double[:,:,:]', eta3 : 'double[:,:,:]', kind_fun : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]', values : 'double[:,:,:]'):

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                values[i1, i2, i3] = push_all(a[:, i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)


# ==============================================================================
def kernel_evaluate_sparse(a  : 'double[:,:,:,:]', eta1 : 'double[:,:,:]', eta2 : 'double[:,:,:]', eta3 : 'double[:,:,:]', kind_fun : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]', values : 'double[:,:,:]'):

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                values[i1, i2, i3] = push_all(a[:, i1, i2, i3], eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)


# ==============================================================================
def kernel_evaluate_flat(a : 'double[:,:]', eta1 : 'double[:]', eta2 : 'double[:]', eta3 : 'double[:]', kind_fun : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]', values : 'double[:]'):
    """Same as `kernel_evaluate`, but for flat evaluation.

    Returns
    -------
        mat_f:  np.array
            1d array [f(x1, y1, z1) f(x2, y2, z2) etc.]
    """

    for i in range(len(eta1)):
        values[i] = push_all(a[:, i], eta1[i], eta2[i], eta3[i], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)