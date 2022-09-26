# coding: utf-8

"""
Basic push-forward (logical --> physical) operations between scalar fields, vector fields and differential p-forms:

- 0-form:  a           = a^0
- 1-form: (ax, ay, az) =             DF^(-T) (a^1_1, a^1_2, a^1_3)
- 2-form: (ax, ay, az) = 1/|det(DF)| DF      (a^2_1, a^2_2, a^2_3)
- 3-form:  a           = 1/|det(DF)| a^3

- vector: (ax, ay, az) =             DF      (a_1  , a_2  , a_3  )  
"""
from pyccel.decorators import pure, stack_array

from numpy import shape, empty

from struphy.linear_algebra.core import det, matrix_inv, matrix_vector, transpose
from struphy.geometry.map_eval import df, det_df


@pure
def push_0_form(a0: float, eta1: float, eta2: float, eta3: float, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:, :]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    """
    Point-wise push-forward of a differential 0-form to a scalar field.

    Parameters
    ----------
        a0 : float
            Value of 0-form.

        eta1, eta2, eta3 : float
            Evaluation point.

        kind_map : int                 
            Kind of mapping (see module docstring).

        params_map : array[float]
            Parameters for the mapping in a 1d array.

        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.

        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].

        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

    Returns
    -------
        a : float
            Scalar field resulting from the push-forward.
    """

    a = a0

    return a


@stack_array('tmp', 'df_out', 'df_T')
def push_1_form(a1: 'float[:]', eta1: float, eta2: float, eta3: float, component: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:, :]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    """
    Point-wise push-forward of components of a differential 1-form to Cartesian components of a vector field.

    Parameters
    ----------
        a1 : array[float]
            Components of differential 1-form.

        eta1, eta2, eta3 : float
            Evaluation point.

        component : int
            Cartesian component of vector field (1 : ax, 2 : ay, 3 : az).

        kind_map : int                 
            Kind of mapping (see module docstring).

        params_map : array[float]
            Parameters for the mapping in a 1d array.

        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.

        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].

        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

    Returns
    -------
        a : float
            Cartesian Component of vector field resulting from the push-forward.
    """

    tmp = empty(3, dtype=float)
    df_out = empty((3, 3), dtype=float)
    df_T = empty((3, 3), dtype=float)
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2,
       tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_out)
    transpose(df_out, df_T)
    matrix_inv(df_T, df_out)
    matrix_vector(df_out, a1, tmp)

    if component == 1:
        return tmp[0]
    elif component == 2:
        return tmp[1]
    elif component == 3:
        return tmp[2]
    else:
        print('Error: component does not exist')


@stack_array('tmp', 'df_out')
def push_2_form(a2: 'float[:]', eta1: float, eta2: float, eta3: float, component: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:, :]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    """
    Point-wise push-forward of components of a differential 2-form to Cartesian components of a vector field.

    Parameters
    ----------
        a2 : float
            Components of differential 2-form.

        eta1, eta2, eta3 : float
            Evaluation point.

        component : int
            Cartesian component of vector field (1 : ax, 2 : ay, 3 : az).

        kind_map : int                 
            Kind of mapping (see module docstring).

        params_map : array[float]
            Parameters for the mapping in a 1d array.

        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.

        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].

        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

    Returns
    -------
        a : float
            Cartesian Component of vector field resulting from the push-forward.
    """

    tmp = empty(3, dtype=float)
    df_out = empty((3, 3), dtype=float)
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2,
       tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_out)
    matrix_vector(df_out, a2, tmp)
    detdf = det(df_out)

    if component == 1:
        return tmp[0]/abs(detdf)
    elif component == 2:
        return tmp[1]/abs(detdf)
    elif component == 3:
        return tmp[2]/abs(detdf)
    else:
        print('Error: component does not exist')


@pure
def push_3_form(a3: float, eta1: float, eta2: float, eta3: float, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:, :]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    """
    Point-wise push-forward of a differential 3-form to a scalar field.

    Parameters
    ----------
        a3 : float
            Value of 0-form.

        eta1, eta2, eta3 : float
            Evaluation point.

        kind_map : int                 
            Kind of mapping (see module docstring).

        params_map : array[float]
            Parameters for the mapping in a 1d array.

        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.

        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].

        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

    Returns
    -------
        a : float
            Scalar field resulting from the push-forward.
    """

    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1,
                   tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    a = a3 / abs(detdf)

    return a


@stack_array('tmp', 'df_out')
def push_vector(a: 'float[:]', eta1: float, eta2: float, eta3: float, component: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:, :]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    """
    Point-wise push-forward of contravariant components of a vector field to Cartesian components of a vector field.

    Parameters
    ----------
        a : float
            Contravariant components of vector field.

        eta1, eta2, eta3 : float
            Evaluation point.

        component : int
            Cartesian component of vector field (1 : ax, 2 : ay, 3 : az).

        kind_map : int                 
            Kind of mapping (see module docstring).

        params_map : array[float]
            Parameters for the mapping in a 1d array.

        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.

        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].

        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

    Returns
    -------
        a : float
            Cartesian Component of vector field resulting from the push-forward.
    """

    tmp = empty(3, dtype=float)
    df_out = empty((3, 3), dtype=float)
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2,
       tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_out)
    matrix_vector(df_out, a, tmp)

    if component == 1:
        return tmp[0]
    elif component == 2:
        return tmp[1]
    elif component == 3:
        return tmp[2]
    else:
        print('Error: component does not exist')


def push_all(a: 'float[:]', eta1: float, eta2: float, eta3: float, kind_fun: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:, :]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    """
    Point-wise push-forward of a differential k-form to a Cartesian scalar/vector field.

    Parameters
    ----------
        a : array[float]
            Value of differential 0/3-form [a] or values of a differential 1/2-form or contravariant vector field [a1, a2, a3].

        eta1, eta2, eta3 : float
            Evaluation point.

        kind_fun : int
            Which push-forward to be performed.

        kind_map : int                 
            Kind of mapping (see module docstring).

        params_map : array[float]
            Parameters for the mapping in a 1d array.

        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.

        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].

        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

    Returns
    -------
        value : float
            Scalar field or Cartesian component of vector field resulting from the push-forward.
    """

    value = 0.

    # 0-form
    if kind_fun == 0:
        value = push_0_form(a[0], eta1, eta2, eta3, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # 3-form
    elif kind_fun == 3:
        value = push_3_form(a[0], eta1, eta2, eta3, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # 1-form
    elif kind_fun == 11:
        value = push_1_form(a, eta1, eta2, eta3, 1, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 12:
        value = push_1_form(a, eta1, eta2, eta3, 2, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 13:
        value = push_1_form(a, eta1, eta2, eta3, 3, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # 2-form
    elif kind_fun == 21:
        value = push_2_form(a, eta1, eta2, eta3, 1, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 22:
        value = push_2_form(a, eta1, eta2, eta3, 2, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 23:
        value = push_2_form(a, eta1, eta2, eta3, 3, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # vector
    elif kind_fun == 31:
        value = push_vector(a, eta1, eta2, eta3, 1, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 32:
        value = push_vector(a, eta1, eta2, eta3, 2, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 33:
        value = push_vector(a, eta1, eta2, eta3, 3, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    return value


def kernel_evaluate(a: 'float[:,:,:,:]', eta1: 'float[:,:,:]', eta2: 'float[:,:,:]', eta3: 'float[:,:,:]', kind_fun: int, kind_map: int, params_map: 'float[:]', pn: 'int[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:, :]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]', values: 'float[:,:,:]', is_sparse_meshgrid: bool):
    """
    Push-forward of a differential k-form or contravariant vector field to Cartesian scalar/vector field.

    Parameters
    ----------
        a : array[float]
            Values of 0/3-form [a_ijk] or values of 1/2-form or contravariant vector field [a_mu,ijk].

        eta1, eta2, eta3 : array[float]
            Evaluation points in 3d arrays such that shape(eta1) == shape(eta2) == shape(eta3).

        kind_fun : int
            Which push-forward to be performed.

        kind_map : int                 
            Kind of mapping (see module docstring).

        params_map : array[float]
            Parameters for the mapping in a 1d array.

        tn1, tn2, tn3 : array[float]          
            Knot vectors of univariate splines.

        pn : array[int]
            Degrees of univariate splines [pn1, pn2, pn3].

        ind_n1, ind_n2, ind_n3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).

        cx, cy, cz : array[float]     
            Control points of (f_1, f_2, f_3) in case of a IGA mapping.

        is_sparse_meshgrid : bool
            Whether the evaluation points werde obtained from a sparse meshgrid.

    Returns
    -------
        values : float
            Scalar fied or Cartesian component resulting from the push-forward.
    """

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

                # 0-form
                if kind_fun == 0:
                    values[i1, i2, i3] = push_0_form(a[0, i1, i2, i3], e1, e2, e3, kind_map, params_map,
                                                     tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

                # 3-form
                elif kind_fun == 3:
                    values[i1, i2, i3] = push_3_form(a[0, i1, i2, i3], e1, e2, e3, kind_map, params_map,
                                                     tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

                # 1-form
                elif kind_fun == 11:
                    values[i1, i2, i3] = push_1_form(a[:, i1, i2, i3], e1, e2, e3, 1, kind_map, params_map,
                                                     tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 12:
                    values[i1, i2, i3] = push_1_form(a[:, i1, i2, i3], e1, e2, e3, 2, kind_map, params_map,
                                                     tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 13:
                    values[i1, i2, i3] = push_1_form(a[:, i1, i2, i3], e1, e2, e3, 3, kind_map, params_map,
                                                     tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

                # 2-form
                elif kind_fun == 21:
                    values[i1, i2, i3] = push_2_form(a[:, i1, i2, i3], e1, e2, e3, 1, kind_map, params_map,
                                                     tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 22:
                    values[i1, i2, i3] = push_2_form(a[:, i1, i2, i3], e1, e2, e3, 2, kind_map, params_map,
                                                     tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 23:
                    values[i1, i2, i3] = push_2_form(a[:, i1, i2, i3], e1, e2, e3, 3, kind_map, params_map,
                                                     tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

                # vector
                elif kind_fun == 31:
                    values[i1, i2, i3] = push_vector(a[:, i1, i2, i3], e1, e2, e3, 1, kind_map, params_map,
                                                     tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 32:
                    values[i1, i2, i3] = push_vector(a[:, i1, i2, i3], e1, e2, e3, 2, kind_map, params_map,
                                                     tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
                elif kind_fun == 33:
                    values[i1, i2, i3] = push_vector(a[:, i1, i2, i3], e1, e2, e3, 3, kind_map, params_map,
                                                     tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
