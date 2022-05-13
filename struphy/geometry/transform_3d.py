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

from struphy.geometry.mappings_3d import df, det_df


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

    a3 = norm_a * abs(detdf)
    
    return a3


# ==============================================================================
def transform_norm_vector_to_vector(norm_a_1 : 'double', norm_a_2 : 'double', norm_a_3 : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    vector
    norm logical to vector: 
    (a_1, a_2, a_3) = (a_1^* / sqrt(DF_11**2 + DF_21**2 + DF_31**2),
                       a_2^* / sqrt(DF_12**2 + DF_22**2 + DF_32**2), 
                       a_3^* / sqrt(DF_13**2 + DF_23**2 + DF_33**2)) 
    '''
    if component == 1:

        df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
        a = norm_a_1 / sqrt(df_11**2 + df_21**2 + df_31**2)

    elif component == 2:
        
        df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

        a = norm_a_2 / sqrt(df_12**2 + df_22**2 + df_32**2)

    elif component == 3:
        
        df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

        a = norm_a_3 / sqrt(df_13**2 + df_23**2 + df_33**2)
        
    return a


# ==============================================================================
def transform_norm_vector_to_1_form(norm_a_1 : 'double', norm_a_2 : 'double', norm_a_3 : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    vector
    norm logical to 1-form: 
    (a_1, a_2, a_3)       =   (a_1^* / sqrt(DF_11**2 + DF_21**2 + DF_31**2),
                               a_2^* / sqrt(DF_12**2 + DF_22**2 + DF_32**2), 
                               a_3^* / sqrt(DF_13**2 + DF_23**2 + DF_33**2)) 
    (a^1_1, a^1_2, a^1_3) = G (a_1, a_2, a_3)
    '''
    
    df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    if   component == 1:
        
        g_11 = df_11*df_11 + df_21*df_21 + df_31*df_31
        g_12 = df_11*df_12 + df_21*df_22 + df_31*df_32
        g_13 = df_11*df_13 + df_21*df_23 + df_31*df_33
        
        a  = g_11*norm_a_1/sqrt(df_11**2 + df_21**2 + df_31**2) 
        a += g_12*norm_a_2/sqrt(df_12**2 + df_22**2 + df_32**2) 
        a += g_13*norm_a_3/sqrt(df_13**2 + df_23**2 + df_33**2)
    
    elif component == 2:
        
        g_21 = df_11*df_12 + df_21*df_22 + df_31*df_32
        g_22 = df_12*df_12 + df_22*df_22 + df_32*df_32
        g_23 = df_12*df_13 + df_22*df_23 + df_32*df_33
        
        a  = g_21*norm_a_1/sqrt(df_11**2 + df_21**2 + df_31**2) 
        a += g_22*norm_a_2/sqrt(df_12**2 + df_22**2 + df_32**2) 
        a += g_23*norm_a_3/sqrt(df_13**2 + df_23**2 + df_33**2)
        
    elif component == 3:
        
        g_31 = df_11*df_13 + df_21*df_23 + df_31*df_33
        g_32 = df_12*df_13 + df_22*df_23 + df_32*df_33
        g_33 = df_13*df_13 + df_23*df_23 + df_33*df_33
        
        a  = g_31*norm_a_1/sqrt(df_11**2 + df_21**2 + df_31**2) 
        a += g_32*norm_a_2/sqrt(df_12**2 + df_22**2 + df_32**2) 
        a += g_33*norm_a_3/sqrt(df_13**2 + df_23**2 + df_33**2)
        
    return a


# ==============================================================================
def transform_norm_vector_to_2_form(norm_a_1 : 'double', norm_a_2 : 'double', norm_a_3 : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    vector
    norm logical to 2-form: 
    (a_1, a_2, a_3)       =             (a_1^* / sqrt(DF_11**2 + DF_21**2 + DF_31**2),
                                         a_2^* / sqrt(DF_12**2 + DF_22**2 + DF_32**2), 
                                         a_3^* / sqrt(DF_13**2 + DF_23**2 + DF_33**2)) 
    (a^1_1, a^1_2, a^1_3) = |det(DF)| * (a_1, a_2, a_3)
    '''
    
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

        a = abs(detdf)*norm_a_1/sqrt(df_11**2 + df_21**2 + df_31**2)
    
    elif component == 2:

        a = abs(detdf)*norm_a_2/sqrt(df_12**2 + df_22**2 + df_32**2)
        
    elif component == 3:

        a = abs(detdf)*norm_a_3/sqrt(df_13**2 + df_23**2 + df_33**2)
        
    return a


# ==============================================================================
def transform_0_form_to_3_form(a0 : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    scalar
    0 form to 3 form
    a^3 = a^0 * |det(DF)|
    '''
    
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    a3 = a0 * abs(detdf)
    
    return a3


# ==============================================================================
def transform_1_form_to_2_form(a1_1 : 'double', a1_2 : 'double', a1_3 : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    vector
    1 form to 2 form:
    (a^2_1, a^2_2, a^2_3) = G_inv (a^1_1, a^1_2, a^1_3) * |det(DF)|
    '''
    
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
        
        ginv_11 = (df_22*df_33 - df_23*df_32)*(df_22*df_33 - df_23*df_32) + (df_13*df_32 - df_12*df_33)*(df_13*df_32 - df_12*df_33) + (df_12*df_23 - df_13*df_33)*(df_12*df_23 - df_13*df_33)
        
        ginv_12 = (df_22*df_33 - df_23*df_32)*(df_23*df_31 - df_21*df_33) + (df_13*df_32 - df_12*df_33)*(df_11*df_33 - df_13*df_31) + (df_12*df_23 - df_13*df_33)*(df_13*df_21 - df_11*df_23)
        
        ginv_13 = (df_22*df_33 - df_23*df_32)*(df_21*df_32 - df_22*df_31) + (df_13*df_32 - df_12*df_33)*(df_12*df_31 - df_11*df_32) + (df_12*df_23 - df_13*df_33)*(df_11*df_22 - df_12*df_21)
        
        a = (ginv_11*a1_1 + ginv_12*a1_2 + ginv_13*a1_3) * abs(detdf)
    
    elif component == 2:
        
        ginv_21 = (df_22*df_33 - df_23*df_32)*(df_23*df_31 - df_21*df_33) + (df_13*df_32 - df_12*df_33)*(df_11*df_33 - df_13*df_31) + (df_12*df_23 - df_13*df_33)*(df_13*df_21 - df_11*df_23)
        
        ginv_22 = (df_23*df_31 - df_21*df_33)*(df_23*df_31 - df_21*df_33) + (df_11*df_33 - df_13*df_31)*(df_11*df_33 - df_13*df_31) + (df_13*df_21 - df_11*df_23)*(df_13*df_21 - df_11*df_23)
        
        ginv_23 = (df_23*df_31 - df_21*df_33)*(df_21*df_32 - df_22*df_31) + (df_11*df_33 - df_13*df_31)*(df_12*df_31 - df_11*df_32) + (df_13*df_21 - df_11*df_23)*(df_11*df_22 - df_12*df_21)
        
        a = (ginv_21*a1_1 + ginv_22*a1_2 + ginv_23*a1_3) * abs(detdf)
        
    elif component == 3:
        
        ginv_31 = (df_22*df_33 - df_23*df_32)*(df_21*df_32 - df_22*df_31) + (df_13*df_32 - df_12*df_33)*(df_12*df_31 - df_11*df_32) + (df_12*df_23 - df_13*df_33)*(df_11*df_22 - df_12*df_21)
        
        ginv_32 = (df_23*df_31 - df_21*df_33)*(df_21*df_32 - df_22*df_31) + (df_11*df_33 - df_13*df_31)*(df_12*df_31 - df_11*df_32) + (df_13*df_21 - df_11*df_23)*(df_11*df_22 - df_12*df_21)
        
        ginv_33 = (df_21*df_32 - df_22*df_31)*(df_21*df_32 - df_22*df_31) + (df_12*df_31 - df_11*df_32)*(df_12*df_31 - df_11*df_32) + (df_11*df_22 - df_12*df_21)*(df_11*df_22 - df_12*df_21)
        
        a = (ginv_31*a1_1 + ginv_32*a1_2 + ginv_33*a1_3) * abs(detdf)
        
    return a 


# ==============================================================================
def transform_2_form_to_1_form(a2_1 : 'double', a2_2 : 'double', a2_3 : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', component : 'int', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    vector
    2 form to 1 form:
    (a^1_1, a^1_2, a^1_3) = G (a^2_1, a^2_2, a^2_3) / |det(DF)|
    '''
    
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
        
        g_11 = df_11*df_11 + df_21*df_21 + df_31*df_31
        g_12 = df_11*df_12 + df_21*df_22 + df_31*df_32
        g_13 = df_11*df_13 + df_21*df_23 + df_31*df_33
        
        a = (g_11*a2_1 + g_12*a2_2 + g_13*a2_3) / abs(detdf)
    
    elif component == 2:
        
        g_21 = df_11*df_12 + df_21*df_22 + df_31*df_32
        g_22 = df_12*df_12 + df_22*df_22 + df_32*df_32
        g_23 = df_12*df_13 + df_22*df_23 + df_32*df_33
        
        a = (g_21*a2_1 + g_22*a2_2 + g_23*a2_3) / abs(detdf)
        
    elif component == 3:
        
        g_31 = df_11*df_13 + df_21*df_23 + df_31*df_33
        g_32 = df_12*df_13 + df_22*df_23 + df_32*df_33
        g_33 = df_13*df_13 + df_23*df_23 + df_33*df_33
        
        a = (g_31*a2_1 + g_32*a2_2 + g_33*a2_3) / abs(detdf)
        
    return a 


# ==============================================================================
def transform_3_form_to_0_form(a3 : 'double', eta1 : 'double', eta2 : 'double', eta3 : 'double', kind_map : 'int', params_map : 'double[:]', tn1 : 'double[:]', tn2 : 'double[:]', tn3 : 'double[:]', pn : 'int[:]', nbase_n : 'int[:]', cx : 'double[:,:,:]', cy : 'double[:,:,:]', cz : 'double[:,:,:]') -> 'double':
    '''
    scalar
    3 form to 0 form
    a^0 = a^3 / |det(DF)|
    '''
    
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    a0 = a3 / abs(detdf)
    
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
