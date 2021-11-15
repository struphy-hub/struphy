# coding: utf-8

from numpy import shape, sqrt
from pyccel.decorators import types
import struphy.geometry.mappings_3d as mapping

"""Transform between functions which are defined at different domains

Methods
-------
transformation type:
norm scalar to 0-form
norm scalarto 3-form
norm vector to vector
norm vectorto 1-form          
norm vectorto 2-form
0-form to 3-form         
1-form to 2-form
2-form to 1-form
3-form to 0-form

evaluation type:
kernel_evaluate
kernel_evaluate_sparse
"""


# ==============================================================================
@types('double','double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def transform_norm_scalar_to_0_form(norm_a, eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    '''
    scalar
    norm logical to 0-form: a^0 = a
    '''
    a0 = norm_a
    
    return a0


# ==============================================================================
@types('double','double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def transform_norm_scalar_to_3_form(norm_a, eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    '''
    scalar
    norm logical to 3-form:  a^3 = |det(DF)| * a
    '''
    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    a3 = norm_a * detdf
    
    return a3


# ==============================================================================
@types('double','double','double','double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def transform_norm_vector_to_vector(norm_ax, norm_ay, norm_az, eta1, eta2, eta3, component, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    '''
    vector
    norm logical to vector: 
    (ax, ay, az)          =   (ax^* / sqrt(DF_11**2 + DF_21**2 + DF_31**2),
                               ay^* / sqrt(DF_12**2 + DF_22**2 + DF_32**2), 
                               az^* / sqrt(DF_13**2 + DF_23**2 + DF_33**2)) 
    '''
    if component == 1:

        DF_11 = mapping.df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        DF_21 = mapping.df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        DF_31 = mapping.df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
        a = norm_ax / sqrt(DF_11**2 + DF_21**2 + DF_31**2)

    elif component == 2:
        DF_12 = mapping.df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        DF_22 = mapping.df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        DF_32 = mapping.df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

        a = norm_ay / sqrt(DF_12**2 + DF_22**2 + DF_32**2)

    elif component == 3:
        DF_13 = mapping.df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        DF_23 = mapping.df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        DF_33 = mapping.df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

        a = norm_az / sqrt(DF_13**2 + DF_23**2 + DF_33**2)
        
    return a

# ==============================================================================
@types('double','double','double','double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def transform_norm_vector_to_1_form(norm_ax, norm_ay, norm_az, eta1, eta2, eta3, component, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
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
        
        g_11 = mapping.g(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_12 = mapping.g(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_13 = mapping.g(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = g_11*ax + g_12*ay + g_13*az
    
    elif component == 2:
        
        g_21 = mapping.g(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_22 = mapping.g(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_23 = mapping.g(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = g_21*ax + g_22*ay + g_23*az
        
    elif component == 3:
        
        g_31 = mapping.g(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_32 = mapping.g(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_33 = mapping.g(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = g_31*ax + g_32*ay + g_33*az
        
    return a


# ==============================================================================
@types('double','double','double','double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def transform_norm_vector_to_2_form(norm_ax, norm_ay, norm_az, eta1, eta2, eta3, component, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
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

    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    if   component == 1:

        a = ax * detdf
    
    elif component == 2:

        a = ay * detdf
        
    elif component == 3:

        a = az * detdf
        
    return a


# ==============================================================================
@types('double','double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def transform_0_form_to_3_form(a0, eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    '''
    scalar
    0 form to 3 form
    a^3 = a^0 * |det(DF)|
    '''
    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    a3 = a0 * detdf
    
    return a3

# ==============================================================================
@types('double','double','double','double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def transform_1_form_to_2_form(a1x, a1y, a1z, eta1, eta2, eta3, component, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    '''
    vector
    1 form to 2 form:
    (a^2_1, a^2_2, a^2_3) = G_inv (a^1_1, a^1_2, a^1_3) * |det(DF)|
    '''
    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    if   component == 1:
        
        ginv_11 = mapping.g_inv(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        ginv_12 = mapping.g_inv(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        ginv_13 = mapping.g_inv(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (ginv_11*a1x + ginv_12*a1y + ginv_13*a1z) * detdf
    
    elif component == 2:
        
        ginv_21 = mapping.g_inv(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        ginv_22 = mapping.g_inv(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        ginv_23 = mapping.g_inv(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (ginv_21*a1x + ginv_22*a1y + ginv_23*a1z) * detdf
        
    elif component == 3:
        
        ginv_31 = mapping.g_inv(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        ginv_32 = mapping.g_inv(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        ginv_33 = mapping.g_inv(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (ginv_31*a1x + ginv_32*a1y + ginv_33*a1z) * detdf
        
    return a 


# ==============================================================================
@types('double','double','double','double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def transform_2_form_to_1_form(a2x, a2y, a2z, eta1, eta2, eta3, component, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    '''
    vector
    2 form to 1 form:
    (a^1_1, a^1_2, a^1_3) = G (a^2_1, a^2_2, a^2_3) / |det(DF)|
    '''
    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    if   component == 1:
        
        g_11 = mapping.g(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_12 = mapping.g(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_13 = mapping.g(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (g_11*a2x + g_12*a2y + g_13*a2z) / detdf
    
    elif component == 2:
        
        g_21 = mapping.g(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_22 = mapping.g(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_23 = mapping.g(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (g_21*a2x + g_22*a2y + g_23*a2z) / detdf
        
    elif component == 3:
        
        g_31 = mapping.g(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_32 = mapping.g(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        g_33 = mapping.g(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        a = (g_31*a2x + g_32*a2y + g_33*a2z) / detdf
        
    return a 


# ==============================================================================
@types('double','double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def transform_3_form_to_0_form(a3, eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    '''
    scalar
    3 form to 0 form
    a^0 = a^3 / |det(DF)|
    '''
    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    a0 = a3 / detdf
    
    return a3


# ==============================================================================
@types('double[:]','double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def transform_all(a, eta1, eta2, eta3, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):

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
@types('double[:,:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def kernel_evaluate(a, eta1, eta2, eta3, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, values):

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                values[i1, i2, i3] = transform_all(a[:, i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)


# ==============================================================================
@types('double[:,:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def kernel_evaluate_sparse(a, eta1, eta2, eta3, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, values):

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                values[i1, i2, i3] = transform_all(a[:, i1, i2, i3], eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
