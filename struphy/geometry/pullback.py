# coding: utf-8

"""
Basic pull-back (physical --> logical) operations between scalar fields, vector fields and differential p-forms:

- 0-form:  a^0                  = a
- 1-form: (a^1_1, a^1_2, a^1_3) =           DF^T    (ax, ay, az)
- 2-form: (a^2_1, a^2_2, a^2_3) = |det(DF)| DF^(-1) (ax, ay, az)
- 3-form:  a^3                  = |det(DF)| a

- vector: (a_1  , a_2  , a_3  ) =           DF^(-1) (ax, ay, az)
"""

from numpy import shape, empty

from struphy.linear_algebra.core import det, matrix_vector, transpose, matrix_inv_with_det
from struphy.geometry.map_eval import df, det_df, df_inv


def pull_0_form(a: float, eta1: float, eta2: float, eta3: float, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    """
    Point-wise pull-back of a scalar field to a differential 0-form.

    Parameters
    ----------
        a : float
            Value of scalar field.

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
        a0 : float
            Differential 0-form resulting from the pull-back.
    """

    a0 = a

    return a0


def pull_1_form(a: 'float[:]', eta1: float, eta2: float, eta3: float, component: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    """
    Point-wise pull-back of a Cartesian vector field to a differential 1-form.

    Parameters
    ----------
        a : array[float]
            Cartesian components of vector field.

        eta1, eta2, eta3 : float
            Evaluation point.

        component : int
            Component of 1-form (1 : a^1_1, 2 : a^1_2, 3 : a^1_3).

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
            Component of differential 1-form resulting from the pull-back.
    """

    tmp = empty(3, dtype=float)
    df_out = empty((3, 3), dtype=float)
    df_T = empty((3, 3), dtype=float)

    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2,
       tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_out)
    transpose(df_out, df_T)
    matrix_vector(df_T, a, tmp)

    if component == 1:
        return tmp[0]
    elif component == 2:
        return tmp[1]
    elif component == 3:
        return tmp[2]
    else:
        print('Error: component does not exist')


def pull_2_form(a: 'float[:]', eta1: float, eta2: float, eta3: float, component: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    """
    Point-wise pull-back of a Cartesian vector field to a differential 2-form.

    Parameters
    ----------
        a : array[float]
            Cartesian components of vector field.

        eta1, eta2, eta3 : float
            Evaluation point.

        component : int
            Component of 2-form (1 : a^2_1, 2 : a^2_2, 3 : a^2_3).

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
            Component of differential 2-form resulting from the pull-back.
    """

    df_mat = empty((3, 3), dtype=float)
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2,
       tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_mat)

    # pullback computed by cancellation of Jacobian determinant in numerator and denominator
    if component == 1:
        value = (df_mat[1, 1]*df_mat[2, 2] - df_mat[2, 1]*df_mat[1, 2])*a[0] + (df_mat[2, 1]*df_mat[0, 2] -
                                                                                df_mat[0, 1]*df_mat[2, 2])*a[1] + (df_mat[0, 1]*df_mat[1, 2] - df_mat[1, 1]*df_mat[0, 2])*a[2]
    elif component == 2:
        value = (df_mat[1, 2]*df_mat[2, 0] - df_mat[2, 2]*df_mat[1, 0])*a[0] + (df_mat[2, 2]*df_mat[0, 0] -
                                                                                df_mat[0, 2]*df_mat[2, 0])*a[1] + (df_mat[0, 2]*df_mat[1, 0] - df_mat[1, 2]*df_mat[0, 0])*a[2]
    elif component == 3:
        value = (df_mat[1, 0]*df_mat[2, 1] - df_mat[2, 0]*df_mat[1, 1])*a[0] + (df_mat[2, 0]*df_mat[0, 1] -
                                                                                df_mat[0, 0]*df_mat[2, 1])*a[1] + (df_mat[0, 0]*df_mat[1, 1] - df_mat[1, 0]*df_mat[0, 1])*a[2]
    else:
        print('Error: component does not exist')

    if det(df_mat) < 0.:
        value = -value

    return value


def pull_3_form(a: float, eta1: float, eta2: float, eta3: float, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    """
    Point-wise pull-back of a scalar field to component of a differential 3-form.

    Parameters
    ----------
        a : float
            Value of scalar field.

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
        a3 : float
            Component of differential 3-form resulting from the pull-back.
    """

    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1,
                   tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    a3 = a * abs(detdf)

    return a3


def pull_vector(a: 'float[:]', eta1: float, eta2: float, eta3: float, component: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    """
    Point-wise pull-back of a Cartesian vector field to contravariant components.

    Parameters
    ----------
        a : array[float]
            Cartesian components of vector field.

        eta1, eta2, eta3 : float
            Evaluation point.

        component : int
            Component of vector field (1 : a_1, 2 : a_2, 3 : a_3).

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
            Contravariant component resulting from the pull-back.
    """

    tmp = empty(3, dtype=float)
    dfinv_out = empty((3, 3), dtype=float)
    
    df_inv(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3,
           pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, dfinv_out)
    matrix_vector(dfinv_out, a, tmp)

    if component == 1:
        return tmp[0]
    elif component == 2:
        return tmp[1]
    elif component == 3:
        return tmp[2]
    else:
        print('Error: component does not exist')


def pull_all(a: 'float[:]', eta1: float, eta2: float, eta3: float, kind_fun: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> float:
    """
    Point-wise pull-back of a Cartesian scalar/vector field to a differential k-form.

    Parameters
    ----------
        a : array[float]
            Value of scalar field [a] or values of Cartesian components of vector field [ax, ay, az].

        eta1, eta2, eta3 : float
            Evaluation point.

        kind_fun : int
            Which pull-back to be performed.

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
            Component of differential p-form resulting from the pull-back.
    """

    value = 0.

    # 0-form
    if kind_fun == 0:
        value = pull_0_form(a[0], eta1, eta2, eta3, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # 3-form
    elif kind_fun == 3:
        value = pull_3_form(a[0], eta1, eta2, eta3, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # 1-form
    elif kind_fun == 11:
        value = pull_1_form(a, eta1, eta2, eta3, 1, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 12:
        value = pull_1_form(a, eta1, eta2, eta3, 2, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 13:
        value = pull_1_form(a, eta1, eta2, eta3, 3, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # 2-form
    elif kind_fun == 21:
        value = pull_2_form(a, eta1, eta2, eta3, 1, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 22:
        value = pull_2_form(a, eta1, eta2, eta3, 2, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 23:
        value = pull_2_form(a, eta1, eta2, eta3, 3, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # vector
    elif kind_fun == 31:
        value = pull_vector(a, eta1, eta2, eta3, 1, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 32:
        value = pull_vector(a, eta1, eta2, eta3, 2, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 33:
        value = pull_vector(a, eta1, eta2, eta3, 3, kind_map, params_map,
                            tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    return value


def kernel_evaluate(a: 'float[:,:,:,:]', eta1: 'float[:,:,:]', eta2: 'float[:,:,:]', eta3: 'float[:,:,:]', kind_fun: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]', values: 'float[:,:,:]'):
    """
    Pull-back of a Cartesian scalar/vector field to a differential k-form.

    Parameters
    ----------
        a : array[float]
            Values of scalar field [a_ijk] or values of Cartesian components of vector field [a_mu,ijk].

        eta1, eta2, eta3 : array[float]
            Evaluation points in 3d arrays such that shape(eta1) == shape(eta2) == shape(eta3).

        kind_fun : int
            Which pull-back to be performed.

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
        values : float
            Component of differential p-form resulting from the pull-back.
    """

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                values[i1, i2, i3] = pull_all(a[:, i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3],
                                              kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)


def kernel_evaluate_sparse(a: 'float[:,:,:,:]', eta1: 'float[:,:,:]', eta2: 'float[:,:,:]', eta3: 'float[:,:,:]', kind_fun: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]', values: 'float[:,:,:]'):
    """
    Pull-back of a Cartesian scalar/vector field to a differential k-form using sparse meshgrids.

    Parameters
    ----------
        a : array[float]
            Values of scalar field [a_ijk] or values of Cartesian components of vector field [a_mu,ijk].

        eta1, eta2, eta3 : array[float]
            Evaluation points in 3d arrays such that shape(eta1) = (:,1,1), shape(eta2) = (1,:,1), shape(eta3) = (1,1,:).

        kind_fun : int
            Which pull-back to be performed.

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
        values : float
            Component of differential p-form resulting from the pull-back.
    """

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                values[i1, i2, i3] = pull_all(a[:, i1, i2, i3], eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3],
                                              kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)


def kernel_evaluate_flat(a: 'float[:,:]', eta1: 'float[:]', eta2: 'float[:]', eta3: 'float[:]', kind_fun: int, kind_map: int, params_map: 'float[:]', tn1: 'float[:]', tn2: 'float[:]', tn3: 'float[:]', pn: 'int[:]', ind_n1: 'int[:,:]', ind_n2: 'int[:,:]', ind_n3: 'int[:,:]', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]', values: 'float[:]'):
    """
    Pull-back of a Cartesian scalar/vector field to a differential k-form using a flat evaluation.

    Parameters
    ----------
        a : array[float]
            Values of scalar field [a_ijk] or values of Cartesian components of vector field [a_mu,ijk].

        eta1, eta2, eta3 : array[float]
            Evaluation points in 1d arrays such that len(eta1) == len(eta2) == len(eta3).

        kind_fun : int
            Which pull-back to be performed.

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
        values : float
            Component of differential p-form resulting from the pull-back.
    """

    for i in range(len(eta1)):
        values[i] = pull_all(a[:, i], eta1[i], eta2[i], eta3[i], kind_fun, kind_map,
                             params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
