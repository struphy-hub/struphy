# coding: utf-8

"""Transform between functions which are defined at different domains

Methods
-------
transformation type:
norm scalar to 0-form
norm scalarto 3-form
norm vector to vector
norm vector to 1-form          
norm vector to 2-form
0-form to 3-form         
1-form to 2-form
2-form to 1-form
3-form to 0-form

evaluation type:
kernel_evaluate
kernel_evaluate_sparse
"""

from numpy import shape, sqrt

from struphy.geometry.mappings_3d import df, df_inv, det_df, g, g_inv


# ==============================================================================
def transform_norm_scalar_to_0_form(norm_a : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    scalar
    norm logical to 0-form: a^0 = a
    '''
    a0 = norm_a
    
    return a0


# ==============================================================================
def transform_norm_scalar_to_3_form(norm_a : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    scalar
    norm logical to 3-form:  a^3 = |det(DF)| * a
    '''
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    a3 = norm_a * detdf
    
    return a3


# ==============================================================================
def transform_norm_vector_to_vector(norm_ax : 'double', norm_ay : 'double', norm_az : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    vector
    norm logical to vector: 
    (ax, ay, az)          =   (ax^* / sqrt(DF_11**2 + DF_21**2 + DF_31**2),
                               ay^* / sqrt(DF_12**2 + DF_22**2 + DF_32**2), 
                               az^* / sqrt(DF_13**2 + DF_23**2 + DF_33**2)) 
    '''
    if component == 1:

        DF_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        DF_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        DF_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
        a = norm_ax / sqrt(DF_11**2 + DF_21**2 + DF_31**2)

    elif component == 2:
        DF_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        DF_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        DF_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

        a = norm_ay / sqrt(DF_12**2 + DF_22**2 + DF_32**2)

    elif component == 3:
        DF_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        DF_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        DF_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

        a = norm_az / sqrt(DF_13**2 + DF_23**2 + DF_33**2)
        
    return a

# ==============================================================================
def transform_norm_vector_to_1_form(norm_ax : 'double', norm_ay : 'double', norm_az : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    vector
    norm logical to 1-form: 
    (ax, ay, az)          =   (ax^* / sqrt(DF_11**2 + DF_21**2 + DF_31**2),
                               ay^* / sqrt(DF_12**2 + DF_22**2 + DF_32**2), 
                               az^* / sqrt(DF_13**2 + DF_23**2 + DF_33**2)) 
    (a^1_1, a^1_2, a^1_3) = G (ax, ay, az)
    '''
    ax = transform_norm_vector_to_vector(norm_ax, norm_ay, norm_az, eta1, eta2, eta3, 1,  kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    ay = transform_norm_vector_to_vector(norm_ay, norm_ay, norm_az, eta1, eta2, eta3, 2,  kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    az = transform_norm_vector_to_vector(norm_az, norm_ay, norm_az, eta1, eta2, eta3, 3,  kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    if   component == 1:
        
        g_11 = g(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_12 = g(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_13 = g(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = g_11*ax + g_12*ay + g_13*az
    
    elif component == 2:
        
        g_21 = g(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_22 = g(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_23 = g(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = g_21*ax + g_22*ay + g_23*az
        
    elif component == 3:
        
        g_31 = g(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_32 = g(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_33 = g(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = g_31*ax + g_32*ay + g_33*az
        
    return a


# ==============================================================================
def transform_norm_vector_to_2_form(norm_ax : 'double', norm_ay : 'double', norm_az : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    vector
    norm logical to 2-form: 
    (ax, ay, az)          =             (ax^* / sqrt(DF_11**2 + DF_21**2 + DF_31**2),
                                         ay^* / sqrt(DF_12**2 + DF_22**2 + DF_32**2), 
                                         az^* / sqrt(DF_13**2 + DF_23**2 + DF_33**2)) 
    (a^1_1, a^1_2, a^1_3) = |det(DF)| * (ax, ay, az)
    '''
    ax = transform_norm_vector_to_vector(norm_ax, norm_ay, norm_az, eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    ay = transform_norm_vector_to_vector(norm_ay, norm_ay, norm_az, eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    az = transform_norm_vector_to_vector(norm_az, norm_ay, norm_az, eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    if   component == 1:

        a = ax * detdf
    
    elif component == 2:

        a = ay * detdf
        
    elif component == 3:

        a = az * detdf
        
    return a


# ==============================================================================
def transform_0_form_to_3_form(a0 : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    scalar
    0 form to 3 form
    a^3 = a^0 * |det(DF)|
    '''
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    a3 = a0 * detdf
    
    return a3

# ==============================================================================
def transform_1_form_to_2_form(a1x : 'double', a1y : 'double', a1z : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    vector
    1 form to 2 form:
    (a^2_1, a^2_2, a^2_3) = G_inv (a^1_1, a^1_2, a^1_3) * |det(DF)|
    '''
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    if   component == 1:
        
        ginv_11 = g_inv(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        ginv_12 = g_inv(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        ginv_13 = g_inv(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (ginv_11*a1x + ginv_12*a1y + ginv_13*a1z) * detdf
    
    elif component == 2:
        
        ginv_21 = g_inv(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        ginv_22 = g_inv(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        ginv_23 = g_inv(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (ginv_21*a1x + ginv_22*a1y + ginv_23*a1z) * detdf
        
    elif component == 3:
        
        ginv_31 = g_inv(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        ginv_32 = g_inv(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        ginv_33 = g_inv(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (ginv_31*a1x + ginv_32*a1y + ginv_33*a1z) * detdf
        
    return a 


# ==============================================================================
def transform_2_form_to_1_form(a2x : 'double', a2y : 'double', a2z : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    vector
    2 form to 1 form:
    (a^1_1, a^1_2, a^1_3) = G (a^2_1, a^2_2, a^2_3) / |det(DF)|
    '''
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    if   component == 1:
        
        g_11 = g(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_12 = g(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_13 = g(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (g_11*a2x + g_12*a2y + g_13*a2z) / detdf
    
    elif component == 2:
        
        g_21 = g(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_22 = g(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_23 = g(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (g_21*a2x + g_22*a2y + g_23*a2z) / detdf
        
    elif component == 3:
        
        g_31 = g(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_32 = g(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_33 = g(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (g_31*a2x + g_32*a2y + g_33*a2z) / detdf
        
    return a 


# ==============================================================================
def transform_3_form_to_0_form(a3 : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    scalar
    3 form to 0 form
    a^0 = a^3 / |det(DF)|
    '''
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    a0 = a3 / detdf
    
    return a3


# ==============================================================================
def transform_all(a : 'double[:]', eta1 : 'double', eta2 : 'double', eta3 : 'double', kind_fun : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':

    value = 0.

    # norm scalar to 0 form
    if   kind_fun == 0:
        value = transform_norm_scalar_to_0_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    # norm scalar to 3 form
    elif kind_fun == 3:
        value = transform_norm_scalar_to_3_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    # norm vector to 1 form
    elif kind_fun == 11:
        value = transform_norm_vector_to_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 12:
        value = transform_norm_vector_to_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 13:
        value = transform_norm_vector_to_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    # norm vector to 2 form
    elif kind_fun == 21:
        value = transform_norm_vector_to_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 22:
        value = transform_norm_vector_to_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 23:
        value = transform_norm_vector_to_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    # norm vector to vector
    elif kind_fun == 31:
        value = transform_norm_vector_to_vector(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 32:
        value = transform_norm_vector_to_vector(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 33:
        value = transform_norm_vector_to_vector(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    # 1 from to 2 form
    elif kind_fun == 41:
        value = transform_1_form_to_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 42:
        value = transform_1_form_to_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 43:
        value = transform_1_form_to_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    # 2 from to 1 form
    elif kind_fun == 51:
        value = transform_2_form_to_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 52:
        value = transform_2_form_to_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 53:
        value = transform_2_form_to_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)


    # 0 form to 3 form
    elif kind_fun == 4:
        value = transform_0_form_to_3_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    # 3 form to 0 form
    elif kind_fun == 5:
        value = transform_3_form_to_0_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    return value


# ==============================================================================
def kernel_evaluate(a : 'double[:,:,:,:]', eta1 : 'double[:,:,:]', eta2 : 'double[:,:,:]', eta3 : 'double[:,:,:]', kind_fun : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]', values : 'double[:,:,:]'):

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                values[i1, i2, i3] = transform_all(a[:, i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)


# ==============================================================================
def kernel_evaluate_sparse(a : 'double[:,:,:,:]', eta1 : 'double[:,:,:]', eta2 : 'double[:,:,:]', eta3 : 'double[:,:,:]', kind_fun : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]', values : 'double[:,:,:]'):

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                values[i1, i2, i3] = transform_all(a[:, i1, i2, i3], eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)


# ==============================================================================
def kernel_evaluate_flat(a : 'double[:,:]', eta1 : 'double[:]', eta2 : 'double[:]', eta3 : 'double[:]', kind_fun : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]', values : 'double[:]'):
    """Same as `kernel_evaluate`, but for flat evaluation.

    Returns
    -------
        values : np.array
            1d array [f(x1, y1, z1) f(x2, y2, z2) etc.]
    """

    for i in range(len(eta1)):
        values[i] = transform_all(a[:, i], eta1[i], eta2[i], eta3[i], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
