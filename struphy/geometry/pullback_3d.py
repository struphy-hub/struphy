# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Basic pull-back (physical --> logical) operations between scalar fields, vector fields and differential p-forms:

- 0-form:  a^0                  = a
- 1-form: (a^1_1, a^1_2, a^1_3) =           DF^T    (ax, ay, az)
- 2-form: (a^2_1, a^2_2, a^2_3) = |det(DF)| DF^(-1) (ax, ay, az)
- 3-form:  a^3                  = |det(DF)| a

- vector: (a_1  , a_2  , a_3  ) =           DF^(-1) (ax, ay, az)
"""

from numpy import shape

from struphy.geometry.mappings_3d import df, det_df


# ==============================================================================
def pull_0_form(a : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    
    a0 = a
    
    return a0


# ==============================================================================
def pull_1_form(ax: 'double', ay: 'double', az: 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    
    if   component == 1:
        
        df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = df_11*ax + df_21*ay + df_31*az
    
    elif component == 2:
        
        df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = df_12*ax + df_22*ay + df_32*az
        
    elif component == 3:
        
        df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = df_13*ax + df_23*ay + df_33*az
        
    return a


# ==============================================================================
def pull_2_form(ax: 'double', ay: 'double', az: 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    
    df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    detdf = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
    
    if   component == 1:
        
        a = (df_22*df_33 - df_32*df_23)*ax + (df_32*df_13 - df_12*df_33)*ay + (df_12*df_23 - df_22*df_13)*az 
    
    elif component == 2:
        
        a = (df_23*df_31 - df_33*df_21)*ax + (df_33*df_11 - df_13*df_31)*ay + (df_13*df_21 - df_23*df_11)*az 
        
    elif component == 3:
        
        a = (df_21*df_32 - df_31*df_22)*ax + (df_31*df_12 - df_11*df_32)*ay + (df_11*df_22 - df_21*df_12)*az
         
    if detdf < 0.:
        a = -a
        
    return a


# ==============================================================================
def pull_3_form(a: 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    a3 = a * abs(detdf)
    
    return a3

    
# ==============================================================================
def pull_vector(ax : 'double', ay : 'double', az : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    
    df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    detdf = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
    
    if   component == 1:
        
        a = (df_22*df_33 - df_32*df_23)*ax + (df_32*df_13 - df_12*df_33)*ay + (df_12*df_23 - df_22*df_13)*az 
    
    elif component == 2:
        
        a = (df_23*df_31 - df_33*df_21)*ax + (df_33*df_11 - df_13*df_31)*ay + (df_13*df_21 - df_23*df_11)*az 
        
    elif component == 3:
        
        a = (df_21*df_32 - df_31*df_22)*ax + (df_31*df_12 - df_11*df_32)*ay + (df_11*df_22 - df_21*df_12)*az
        
    a = a / detdf
        
    return a


# ==============================================================================
def pull_all(a : 'double[:]', eta1 : 'double', eta2 : 'double', eta3 : 'double', kind_fun : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':

    value = 0.

    # 0-form
    if   kind_fun == 0:
        value = pull_0_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    # 3-form
    elif kind_fun == 3:
        value = pull_3_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    # 1-form
    elif kind_fun == 11:
        value = pull_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 12:
        value = pull_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 13:
        value = pull_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    # 2-form
    elif kind_fun == 21:
        value = pull_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 22:
        value = pull_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 23:
        value = pull_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    # vector
    elif kind_fun == 31:
        value = pull_vector(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 32:
        value = pull_vector(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 33:
        value = pull_vector(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    return value


# ==============================================================================
def kernel_evaluate(a : 'double[:,:,:,:]', eta1 : 'double[:,:,:]', eta2 : 'double[:,:,:]', eta3 : 'double[:,:,:]', kind_fun : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]', values : 'double[:,:,:]'):

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                values[i1, i2, i3] = pull_all(a[:, i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)


# ==============================================================================
def kernel_evaluate_sparse(a : 'double[:,:,:,:]', eta1 : 'double[:,:,:]', eta2 : 'double[:,:,:]', eta3 : 'double[:,:,:]', kind_fun : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]', values : 'double[:,:,:]'):

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                values[i1, i2, i3] = pull_all(a[:, i1, i2, i3], eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)


# ==============================================================================
def kernel_evaluate_flat(a : 'double[:,:]', eta1 : 'double[:]', eta2 : 'double[:]', eta3 : 'double[:]', kind_fun : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]', values : 'double[:]'):
    """Same as `kernel_evaluate`, but for flat evaluation.

    Returns
    -------
        mat_f:  np.array
            1d array [f(x1, y1, z1) f(x2, y2, z2) etc.]
    """

    for i in range(len(eta1)):
        values[i] = pull_all(a[:, i], eta1[i], eta2[i], eta3[i], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
