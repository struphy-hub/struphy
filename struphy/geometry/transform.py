# coding: utf-8

"""
Transform between functions which are defined at different domains

Methods
-------
transformation types:

- norm vector to vector
- norm vector to 1-form          
- norm vector to 2-form
- 0-form to 3-form         
- 1-form to 2-form
- 2-form to 1-form
- 3-form to 0-form

evaluation types:

- kernel_evaluate
"""
from pyccel.decorators import pure, stack_array

from numpy import shape, empty, sqrt, sum

from struphy.linear_algebra.core import det, matrix_vector
from struphy.geometry.map_eval import df, det_df, g, g_inv


@stack_array('df_out')
def transform_norm_vector_to_vector(norm_a: 'float[:]', eta1: float, eta2: float, eta3: float, component: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    '''
    vector
    norm logical to vector: 
    (a_1, a_2, a_3) = (a_1^* / sqrt(DF_11**2 + DF_21**2 + DF_31**2),
                       a_2^* / sqrt(DF_12**2 + DF_22**2 + DF_32**2), 
                       a_3^* / sqrt(DF_13**2 + DF_23**2 + DF_33**2)) 
    '''

    df_out = empty((3, 3), dtype=float)
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2,
       tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_out)

    if component == 1:
        return norm_a[0] / sqrt(sum(df_out[:, 0]**2))
    elif component == 2:
        return norm_a[1] / sqrt(sum(df_out[:, 1]**2))
    elif component == 3:
        return norm_a[2] / sqrt(sum(df_out[:, 2]**2))
    else:
        print('Error: component does not exist')

@stack_array('tmp', 'df_out', 'g_out')
def transform_norm_vector_to_1_form(norm_a: 'float[:]', eta1: float, eta2: float, eta3: float, component: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    '''
    vector
    norm logical to 1-form: 
    (a_1, a_2, a_3)       =   (a_1^* / sqrt(DF_11**2 + DF_21**2 + DF_31**2),
                               a_2^* / sqrt(DF_12**2 + DF_22**2 + DF_32**2), 
                               a_3^* / sqrt(DF_13**2 + DF_23**2 + DF_33**2)) 
    (a^1_1, a^1_2, a^1_3) = G (a_1, a_2, a_3)
    '''

    tmp = empty(3, dtype=float)
    df_out = empty((3, 3), dtype=float)
    g_out = empty((3, 3), dtype=float)

    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2,
       tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_out)
    g(eta1, eta2, eta3, kind_map, params_map, tn1, tn2,
      tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, g_out)

    tmp[0] = norm_a[0] / sqrt(sum(df_out[:, 0]**2))
    tmp[1] = norm_a[1] / sqrt(sum(df_out[:, 1]**2))
    tmp[2] = norm_a[2] / sqrt(sum(df_out[:, 2]**2))

    matrix_vector(g_out, tmp, norm_a)

    if component == 1:
        return norm_a[0]
    elif component == 2:
        return norm_a[1]
    elif component == 3:
        return norm_a[2]
    else:
        print('Error: component does not exist')

@stack_array('df_out')
def transform_norm_vector_to_2_form(norm_a: 'float[:]', eta1: float, eta2: float, eta3: float, component: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    '''
    vector
    norm logical to 2-form: 
    (a_1, a_2, a_3)       =             (a_1^* / sqrt(DF_11**2 + DF_21**2 + DF_31**2),
                                         a_2^* / sqrt(DF_12**2 + DF_22**2 + DF_32**2), 
                                         a_3^* / sqrt(DF_13**2 + DF_23**2 + DF_33**2)) 
    (a^1_1, a^1_2, a^1_3) = |det(DF)| * (a_1, a_2, a_3)
    '''

    df_out = empty((3, 3), dtype=float)
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2,
       tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_out)
    detdf = det(df_out)

    if component == 1:
        return abs(detdf) * norm_a[0] / sqrt(sum(df_out[:, 0]**2))
    elif component == 2:
        return abs(detdf) * norm_a[1] / sqrt(sum(df_out[:, 1]**2))
    elif component == 3:
        return abs(detdf) * norm_a[2] / sqrt(sum(df_out[:, 2]**2))
    else:
        print('Error: component does not exist')

def transform_0_form_to_3_form(a0: float, eta1: float, eta2: float, eta3: float, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    '''
    scalar
    0 form to 3 form
    a^3 = a^0 * |det(DF)|
    '''

    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1,
                   tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    a3 = a0 * abs(detdf)

    return a3

@stack_array('tmp', 'ginv_out')
def transform_1_form_to_2_form(a1: 'float[:]', eta1: float, eta2: float, eta3: float, component: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    '''
    vector
    1 form to 2 form:
    (a^2_1, a^2_2, a^2_3) = G_inv (a^1_1, a^1_2, a^1_3) * |det(DF)|
    '''

    tmp = empty(3, dtype=float)
    ginv_out = empty((3, 3), dtype=float)

    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2,
                   tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    g_inv(eta1, eta2, eta3, kind_map, params_map, tn1, tn2,
          tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, ginv_out)

    matrix_vector(ginv_out, a1, tmp)

    if component == 1:
        return tmp[0] * abs(detdf)
    elif component == 2:
        return tmp[1] * abs(detdf)
    elif component == 3:
        return tmp[2] * abs(detdf)
    else:
        print('Error: component does not exist')

@stack_array('tmp', 'g_out')
def transform_2_form_to_1_form(a2: 'float[:]', eta1: float, eta2: float, eta3: float, component: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    '''
    vector
    2 form to 1 form:
    (a^1_1, a^1_2, a^1_3) = G (a^2_1, a^2_2, a^2_3) / |det(DF)|
    '''

    tmp = empty(3, dtype=float)
    g_out = empty((3, 3), dtype=float)

    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2,
                   tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    g(eta1, eta2, eta3, kind_map, params_map, tn1, tn2,
      tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, g_out)

    matrix_vector(g_out, a2, tmp)

    if component == 1:
        return tmp[0] / abs(detdf)
    elif component == 2:
        return tmp[1] / abs(detdf)
    elif component == 3:
        return tmp[2] / abs(detdf)
    else:
        print('Error: component does not exist')

def transform_3_form_to_0_form(a3: float, eta1: float, eta2: float, eta3: float, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    '''
    scalar
    3 form to 0 form
    a^0 = a^3 / |det(DF)|
    '''

    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1,
                   tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    a0 = a3 / abs(detdf)

    return a0

def kernel_evaluate(a: 'float[:,:,:,:]', eta1: 'float[:,:,:]', eta2: 'float[:,:,:]', eta3: 'float[:,:,:]', kind_fun: int, kind_map: int, params_map: 'float[:]', pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]', values: 'float[:,:,:]', is_sparse_meshgrid: bool):

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    if is_sparse_meshgrid:
        sparse_factor = 0
    else:
        sparse_factor = 1

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):

                e1 = eta1[i1, i2*sparse_factor, i3*sparse_factor]
                e2 = eta2[i1*sparse_factor, i2, i3*sparse_factor]
                e3 = eta3[i1*sparse_factor, i2*sparse_factor, i3]

                # norm vector to vector
                if kind_fun == 11:
                    values[i1, i2, i3] = transform_norm_vector_to_vector(
                        a[:, i1, i2, i3], e1, e2, e3, 1, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 12:
                    values[i1, i2, i3] = transform_norm_vector_to_vector(
                        a[:, i1, i2, i3], e1, e2, e3, 2, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 13:
                    values[i1, i2, i3] = transform_norm_vector_to_vector(
                        a[:, i1, i2, i3], e1, e2, e3, 3, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

                # norm vector to 1 form
                elif kind_fun == 21:
                    values[i1, i2, i3] = transform_norm_vector_to_1_form(
                        a[:, i1, i2, i3], e1, e2, e3, 1, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 22:
                    values[i1, i2, i3] = transform_norm_vector_to_1_form(
                        a[:, i1, i2, i3], e1, e2, e3, 2, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 23:
                    values[i1, i2, i3] = transform_norm_vector_to_1_form(
                        a[:, i1, i2, i3], e1, e2, e3, 3, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

                # norm vector to 2 form
                elif kind_fun == 31:
                    values[i1, i2, i3] = transform_norm_vector_to_2_form(
                        a[:, i1, i2, i3], e1, e2, e3, 1, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 32:
                    values[i1, i2, i3] = transform_norm_vector_to_2_form(
                        a[:, i1, i2, i3], e1, e2, e3, 2, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 33:
                    values[i1, i2, i3] = transform_norm_vector_to_2_form(
                        a[:, i1, i2, i3], e1, e2, e3, 3, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

                # 0 form to 3 form
                elif kind_fun == 0:
                    values[i1, i2, i3] = transform_0_form_to_3_form(
                        a[0, i1, i2, i3], e1, e2, e3, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

                # 1 form to 2 form
                elif kind_fun == 41:
                    values[i1, i2, i3] = transform_1_form_to_2_form(
                        a[:, i1, i2, i3], e1, e2, e3, 1, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 42:
                    values[i1, i2, i3] = transform_1_form_to_2_form(
                        a[:, i1, i2, i3], e1, e2, e3, 2, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 43:
                    values[i1, i2, i3] = transform_1_form_to_2_form(
                        a[:, i1, i2, i3], e1, e2, e3, 3, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

                # 2 form to 1 form
                elif kind_fun == 51:
                    values[i1, i2, i3] = transform_2_form_to_1_form(
                        a[:, i1, i2, i3], e1, e2, e3, 1, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 52:
                    values[i1, i2, i3] = transform_2_form_to_1_form(
                        a[:, i1, i2, i3], e1, e2, e3, 2, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 53:
                    values[i1, i2, i3] = transform_2_form_to_1_form(
                        a[:, i1, i2, i3], e1, e2, e3, 3, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

                # 3 form to 0 form
                elif kind_fun == 1:
                    values[i1, i2, i3] = transform_3_form_to_0_form(
                        a[0, i1, i2, i3], e1, e2, e3, kind_map, params_map,
                        tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
