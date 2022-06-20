# coding: utf-8

"""
Transform between functions which are defined at different domains

Methods
-------
transformation types:

- norm scalar to 0-form
- norm scalarto 3-form
- norm vector to vector
- norm vector to 1-form          
- norm vector to 2-form
- 0-form to 3-form         
- 1-form to 2-form
- 2-form to 1-form
- 3-form to 0-form

evaluation types:

- kernel_evaluate
- kernel_evaluate_sparse
- kernel_evaluate_flat
"""

from numpy import shape, empty, sqrt

from struphy.linear_algebra.core import det

from struphy.geometry.mappings_3d import df_ij, df, det_df


# ==============================================================================
def transform_norm_scalar_to_0_form(norm_a : float, eta1 : float, eta2 : float, eta3 : float, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    '''
    scalar
    norm logical to 0-form: a^0 = a
    '''
    a0 = norm_a
    
    return a0


# ==============================================================================
def transform_norm_scalar_to_3_form(norm_a : float, eta1 : float, eta2 : float, eta3 : float, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    '''
    scalar
    norm logical to 3-form:  a^3 = |det(DF)| * a
    '''
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    a3 = norm_a * abs(detdf)
    
    return a3


# ==============================================================================
def transform_norm_vector_to_vector(norm_a_1 : float, norm_a_2 : float, norm_a_3 : float, eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    '''
    vector
    norm logical to vector: 
    (a_1, a_2, a_3) = (a_1^* / sqrt(DF_11**2 + DF_21**2 + DF_31**2),
                       a_2^* / sqrt(DF_12**2 + DF_22**2 + DF_32**2), 
                       a_3^* / sqrt(DF_13**2 + DF_23**2 + DF_33**2)) 
    '''
    
    a = 0.
    
    if component == 1:

        df_11 = df_ij(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_21 = df_ij(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_31 = df_ij(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
        a = norm_a_1 / sqrt(df_11**2 + df_21**2 + df_31**2)

    elif component == 2:
        
        df_12 = df_ij(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_22 = df_ij(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_32 = df_ij(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

        a = norm_a_2 / sqrt(df_12**2 + df_22**2 + df_32**2)

    elif component == 3:
        
        df_13 = df_ij(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_23 = df_ij(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_33 = df_ij(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

        a = norm_a_3 / sqrt(df_13**2 + df_23**2 + df_33**2)
        
    return a


# ==============================================================================
def transform_norm_vector_to_1_form(norm_a_1 : float, norm_a_2 : float, norm_a_3 : float, eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    '''
    vector
    norm logical to 1-form: 
    (a_1, a_2, a_3)       =   (a_1^* / sqrt(DF_11**2 + DF_21**2 + DF_31**2),
                               a_2^* / sqrt(DF_12**2 + DF_22**2 + DF_32**2), 
                               a_3^* / sqrt(DF_13**2 + DF_23**2 + DF_33**2)) 
    (a^1_1, a^1_2, a^1_3) = G (a_1, a_2, a_3)
    '''
    
    df_mat = empty((3, 3), dtype=float)
    
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_mat)

    a = 0.
    
    if   component == 1:
        
        g_11 = df_mat[0, 0]*df_mat[0, 0] + df_mat[1, 0]*df_mat[1, 0] + df_mat[2, 1]*df_mat[2, 1]
        g_12 = df_mat[0, 0]*df_mat[0, 1] + df_mat[1, 0]*df_mat[1, 1] + df_mat[2, 1]*df_mat[2, 1]
        g_13 = df_mat[0, 0]*df_mat[0, 2] + df_mat[1, 0]*df_mat[1, 2] + df_mat[2, 1]*df_mat[2, 2]
        
        a += g_11*norm_a_1/sqrt(df_mat[0, 0]**2 + df_mat[1, 0]**2 + df_mat[2, 1]**2) 
        a += g_12*norm_a_2/sqrt(df_mat[0, 1]**2 + df_mat[1, 1]**2 + df_mat[2, 1]**2) 
        a += g_13*norm_a_3/sqrt(df_mat[0, 2]**2 + df_mat[1, 2]**2 + df_mat[2, 2]**2)
    
    elif component == 2:
        
        g_21 = df_mat[0, 0]*df_mat[0, 1] + df_mat[1, 0]*df_mat[1, 1] + df_mat[2, 1]*df_mat[2, 1]
        g_22 = df_mat[0, 1]*df_mat[0, 1] + df_mat[1, 1]*df_mat[1, 1] + df_mat[2, 1]*df_mat[2, 1]
        g_23 = df_mat[0, 1]*df_mat[0, 2] + df_mat[1, 1]*df_mat[1, 2] + df_mat[2, 1]*df_mat[2, 2]
        
        a += g_21*norm_a_1/sqrt(df_mat[0, 0]**2 + df_mat[1, 0]**2 + df_mat[2, 1]**2) 
        a += g_22*norm_a_2/sqrt(df_mat[0, 1]**2 + df_mat[1, 1]**2 + df_mat[2, 1]**2) 
        a += g_23*norm_a_3/sqrt(df_mat[0, 2]**2 + df_mat[1, 2]**2 + df_mat[2, 2]**2)
        
    elif component == 3:
        
        g_31 = df_mat[0, 0]*df_mat[0, 2] + df_mat[1, 0]*df_mat[1, 2] + df_mat[2, 1]*df_mat[2, 2]
        g_32 = df_mat[0, 1]*df_mat[0, 2] + df_mat[1, 1]*df_mat[1, 2] + df_mat[2, 1]*df_mat[2, 2]
        g_33 = df_mat[0, 2]*df_mat[0, 2] + df_mat[1, 2]*df_mat[1, 2] + df_mat[2, 2]*df_mat[2, 2]
        
        a += g_31*norm_a_1/sqrt(df_mat[0, 0]**2 + df_mat[1, 0]**2 + df_mat[2, 1]**2) 
        a += g_32*norm_a_2/sqrt(df_mat[0, 1]**2 + df_mat[1, 1]**2 + df_mat[2, 1]**2) 
        a += g_33*norm_a_3/sqrt(df_mat[0, 2]**2 + df_mat[1, 2]**2 + df_mat[2, 2]**2)
        
    return a


# ==============================================================================
def transform_norm_vector_to_2_form(norm_a_1 : float, norm_a_2 : float, norm_a_3 : float, eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    '''
    vector
    norm logical to 2-form: 
    (a_1, a_2, a_3)       =             (a_1^* / sqrt(DF_11**2 + DF_21**2 + DF_31**2),
                                         a_2^* / sqrt(DF_12**2 + df_mat[1, 1]**2 + DF_32**2), 
                                         a_3^* / sqrt(DF_13**2 + DF_23**2 + DF_33**2)) 
    (a^1_1, a^1_2, a^1_3) = |det(DF)| * (a_1, a_2, a_3)
    '''
    
    df_mat = empty((3, 3), dtype=float)
    
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_mat)

    detdf = det(df_mat)
    
    a = 0.

    if   component == 1:

        a = abs(detdf)*norm_a_1/sqrt(df_mat[0, 0]**2 + df_mat[1, 0]**2 + df_mat[2, 1]**2)
    
    elif component == 2:

        a = abs(detdf)*norm_a_2/sqrt(df_mat[0, 1]**2 + df_mat[1, 1]**2 + df_mat[2, 1]**2)
        
    elif component == 3:

        a = abs(detdf)*norm_a_3/sqrt(df_mat[0, 2]**2 + df_mat[1, 2]**2 + df_mat[2, 2]**2)
        
    return a


# ==============================================================================
def transform_0_form_to_3_form(a0 : float, eta1 : float, eta2 : float, eta3 : float, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    '''
    scalar
    0 form to 3 form
    a^3 = a^0 * |det(DF)|
    '''
    
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    a3 = a0 * abs(detdf)
    
    return a3


# ==============================================================================
def transform_1_form_to_2_form(a1_1 : float, a1_2 : float, a1_3 : float, eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    '''
    vector
    1 form to 2 form:
    (a^2_1, a^2_2, a^2_3) = G_inv (a^1_1, a^1_2, a^1_3) * |det(DF)|
    '''
    
    df_mat = empty((3, 3), dtype=float)
    
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_mat)

    detdf = det(df_mat)
    
    if   component == 1:
        
        ginv_11 = (df_mat[1, 1]*df_mat[2, 2] - df_mat[1, 2]*df_mat[2, 1])*(df_mat[1, 1]*df_mat[2, 2] - df_mat[1, 2]*df_mat[2, 1]) + (df_mat[0, 2]*df_mat[2, 1] - df_mat[0, 1]*df_mat[2, 2])*(df_mat[0, 2]*df_mat[2, 1] - df_mat[0, 1]*df_mat[2, 2]) + (df_mat[0, 1]*df_mat[1, 2] - df_mat[0, 2]*df_mat[2, 2])*(df_mat[0, 1]*df_mat[1, 2] - df_mat[0, 2]*df_mat[2, 2])
        
        ginv_12 = (df_mat[1, 1]*df_mat[2, 2] - df_mat[1, 2]*df_mat[2, 1])*(df_mat[1, 2]*df_mat[2, 1] - df_mat[1, 0]*df_mat[2, 2]) + (df_mat[0, 2]*df_mat[2, 1] - df_mat[0, 1]*df_mat[2, 2])*(df_mat[0, 0]*df_mat[2, 2] - df_mat[0, 2]*df_mat[2, 1]) + (df_mat[0, 1]*df_mat[1, 2] - df_mat[0, 2]*df_mat[2, 2])*(df_mat[0, 2]*df_mat[1, 0] - df_mat[0, 0]*df_mat[1, 2])
        
        ginv_13 = (df_mat[1, 1]*df_mat[2, 2] - df_mat[1, 2]*df_mat[2, 1])*(df_mat[1, 0]*df_mat[2, 1] - df_mat[1, 1]*df_mat[2, 1]) + (df_mat[0, 2]*df_mat[2, 1] - df_mat[0, 1]*df_mat[2, 2])*(df_mat[0, 1]*df_mat[2, 1] - df_mat[0, 0]*df_mat[2, 1]) + (df_mat[0, 1]*df_mat[1, 2] - df_mat[0, 2]*df_mat[2, 2])*(df_mat[0, 0]*df_mat[1, 1] - df_mat[0, 1]*df_mat[1, 0])
        
        a = (ginv_11*a1_1 + ginv_12*a1_2 + ginv_13*a1_3) * abs(detdf)
    
    elif component == 2:
        
        ginv_21 = (df_mat[1, 1]*df_mat[2, 2] - df_mat[1, 2]*df_mat[2, 1])*(df_mat[1, 2]*df_mat[2, 1] - df_mat[1, 0]*df_mat[2, 2]) + (df_mat[0, 2]*df_mat[2, 1] - df_mat[0, 1]*df_mat[2, 2])*(df_mat[0, 0]*df_mat[2, 2] - df_mat[0, 2]*df_mat[2, 1]) + (df_mat[0, 1]*df_mat[1, 2] - df_mat[0, 2]*df_mat[2, 2])*(df_mat[0, 2]*df_mat[1, 0] - df_mat[0, 0]*df_mat[1, 2])
        
        ginv_22 = (df_mat[1, 2]*df_mat[2, 1] - df_mat[1, 0]*df_mat[2, 2])*(df_mat[1, 2]*df_mat[2, 1] - df_mat[1, 0]*df_mat[2, 2]) + (df_mat[0, 0]*df_mat[2, 2] - df_mat[0, 2]*df_mat[2, 0])*(df_mat[0, 0]*df_mat[2, 2] - df_mat[0, 2]*df_mat[2, 0]) + (df_mat[0, 2]*df_mat[1, 0] - df_mat[0, 0]*df_mat[1, 2])*(df_mat[0, 2]*df_mat[1, 0] - df_mat[0, 0]*df_mat[1, 2])
        
        ginv_23 = (df_mat[1, 2]*df_mat[2, 0] - df_mat[1, 0]*df_mat[2, 2])*(df_mat[1, 0]*df_mat[2, 1] - df_mat[1, 1]*df_mat[2, 0]) + (df_mat[0, 0]*df_mat[2, 2] - df_mat[0, 2]*df_mat[2, 0])*(df_mat[0, 1]*df_mat[2, 0] - df_mat[0, 0]*df_mat[2, 1]) + (df_mat[0, 2]*df_mat[1, 0] - df_mat[0, 0]*df_mat[1, 2])*(df_mat[0, 0]*df_mat[1, 1] - df_mat[0, 1]*df_mat[1, 0])
        
        a = (ginv_21*a1_1 + ginv_22*a1_2 + ginv_23*a1_3) * abs(detdf)
        
    elif component == 3:
        
        ginv_31 = (df_mat[1, 1]*df_mat[2, 2] - df_mat[1, 2]*df_mat[2, 1])*(df_mat[1, 0]*df_mat[2, 1] - df_mat[1, 1]*df_mat[2, 0]) + (df_mat[0, 2]*df_mat[2, 1] - df_mat[0, 1]*df_mat[2, 2])*(df_mat[0, 1]*df_mat[2, 0] - df_mat[0, 0]*df_mat[2, 1]) + (df_mat[0, 1]*df_mat[1, 2] - df_mat[0, 2]*df_mat[2, 2])*(df_mat[0, 0]*df_mat[1, 1] - df_mat[0, 1]*df_mat[1, 0])
        
        ginv_32 = (df_mat[1, 2]*df_mat[2, 0] - df_mat[1, 0]*df_mat[2, 2])*(df_mat[1, 0]*df_mat[2, 1] - df_mat[1, 1]*df_mat[2, 0]) + (df_mat[0, 0]*df_mat[2, 2] - df_mat[0, 2]*df_mat[2, 0])*(df_mat[0, 1]*df_mat[2, 0] - df_mat[0, 0]*df_mat[2, 1]) + (df_mat[0, 2]*df_mat[1, 0] - df_mat[0, 0]*df_mat[1, 2])*(df_mat[0, 0]*df_mat[1, 1] - df_mat[0, 1]*df_mat[1, 0])
        
        ginv_33 = (df_mat[1, 0]*df_mat[2, 1] - df_mat[1, 1]*df_mat[2, 0])*(df_mat[1, 0]*df_mat[2, 1] - df_mat[1, 1]*df_mat[2, 0]) + (df_mat[0, 1]*df_mat[2, 0] - df_mat[0, 0]*df_mat[2, 1])*(df_mat[0, 1]*df_mat[2, 0] - df_mat[0, 0]*df_mat[2, 1]) + (df_mat[0, 0]*df_mat[1, 1] - df_mat[0, 1]*df_mat[1, 0])*(df_mat[0, 0]*df_mat[1, 1] - df_mat[0, 1]*df_mat[1, 0])
        
        a = (ginv_31*a1_1 + ginv_32*a1_2 + ginv_33*a1_3) * abs(detdf)
        
    return a 


# ==============================================================================
def transform_2_form_to_1_form(a2_1 : float, a2_2 : float, a2_3 : float, eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    '''
    vector
    2 form to 1 form:
    (a^1_1, a^1_2, a^1_3) = G (a^2_1, a^2_2, a^2_3) / |det(DF)|
    '''
    
    df_mat = empty((3, 3), dtype=float)
    
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_mat)

    detdf = det(df_mat)
    
    if   component == 1:
        
        g_11 = df_mat[0, 0]*df_mat[0, 0] + df_mat[1, 0]*df_mat[1, 0] + df_mat[2, 0]*df_mat[2, 0]
        g_12 = df_mat[0, 0]*df_mat[0, 1] + df_mat[1, 0]*df_mat[1, 1] + df_mat[2, 0]*df_mat[2, 1]
        g_13 = df_mat[0, 0]*df_mat[0, 2] + df_mat[1, 0]*df_mat[1, 2] + df_mat[2, 0]*df_mat[2, 2]
        
        a = (g_11*a2_1 + g_12*a2_2 + g_13*a2_3) / abs(detdf)
    
    elif component == 2:
        
        g_21 = df_mat[0, 0]*df_mat[0, 1] + df_mat[1, 0]*df_mat[1, 1] + df_mat[2, 0]*df_mat[2, 1]
        g_22 = df_mat[0, 1]*df_mat[0, 1] + df_mat[1, 1]*df_mat[1, 1] + df_mat[2, 1]*df_mat[2, 1]
        g_23 = df_mat[0, 1]*df_mat[0, 2] + df_mat[1, 1]*df_mat[1, 2] + df_mat[2, 1]*df_mat[2, 2]
        
        a = (g_21*a2_1 + g_22*a2_2 + g_23*a2_3) / abs(detdf)
        
    elif component == 3:
        
        g_31 = df_mat[0, 0]*df_mat[0, 2] + df_mat[1, 0]*df_mat[1, 2] + df_mat[2, 0]*df_mat[2, 2]
        g_32 = df_mat[0, 1]*df_mat[0, 2] + df_mat[1, 1]*df_mat[1, 2] + df_mat[2, 1]*df_mat[2, 2]
        g_33 = df_mat[0, 2]*df_mat[0, 2] + df_mat[1, 2]*df_mat[1, 2] + df_mat[2, 2]*df_mat[2, 2]
        
        a = (g_31*a2_1 + g_32*a2_2 + g_33*a2_3) / abs(detdf)
        
    return a 


# ==============================================================================
def transform_3_form_to_0_form(a3 : float, eta1 : float, eta2 : float, eta3 : float, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    '''
    scalar
    3 form to 0 form
    a^0 = a^3 / |det(DF)|
    '''
    
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    a0 = a3 / abs(detdf)
    
    return a0


# ==============================================================================
def transform_all(a : 'float[:]', eta1 : float, eta2 : float, eta3 : float, kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:

    value = 0.

    # norm scalar to 0 form
    if   kind_fun == 0:
        value = transform_norm_scalar_to_0_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # norm scalar to 3 form
    elif kind_fun == 3:
        value = transform_norm_scalar_to_3_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # norm vector to 1 form
    elif kind_fun == 11:
        value = transform_norm_vector_to_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 12:
        value = transform_norm_vector_to_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 13:
        value = transform_norm_vector_to_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # norm vector to 2 form
    elif kind_fun == 21:
        value = transform_norm_vector_to_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 22:
        value = transform_norm_vector_to_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 23:
        value = transform_norm_vector_to_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # norm vector to vector
    elif kind_fun == 31:
        value = transform_norm_vector_to_vector(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 32:
        value = transform_norm_vector_to_vector(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 33:
        value = transform_norm_vector_to_vector(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    # 1 from to 2 form
    elif kind_fun == 41:
        value = transform_1_form_to_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 42:
        value = transform_1_form_to_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 43:
        value = transform_1_form_to_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # 2 from to 1 form
    elif kind_fun == 51:
        value = transform_2_form_to_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 52:
        value = transform_2_form_to_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 53:
        value = transform_2_form_to_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)


    # 0 form to 3 form
    elif kind_fun == 4:
        value = transform_0_form_to_3_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # 3 form to 0 form
    elif kind_fun == 5:
        value = transform_3_form_to_0_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    return value


# ==============================================================================
def kernel_evaluate(a : 'float[:,:,:,:]', eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', values : 'float[:,:,:]'):

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                values[i1, i2, i3] = transform_all(a[:, i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)


# ==============================================================================
def kernel_evaluate_sparse(a : 'float[:,:,:,:]', eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', values : 'float[:,:,:]'):

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                values[i1, i2, i3] = transform_all(a[:, i1, i2, i3], eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)


# ==============================================================================
def kernel_evaluate_flat(a : 'float[:,:]', eta1 : 'float[:]', eta2 : 'float[:]', eta3 : 'float[:]', kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', values : 'float[:]'):
    """Same as `kernel_evaluate`, but for flat evaluation.

    Returns
    -------
        values : np.array
            1d array [f(x1, y1, z1) f(x2, y2, z2) etc.]
    """

    for i in range(len(eta1)):
        values[i] = transform_all(a[:, i], eta1[i], eta2[i], eta3[i], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
