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

from numpy import shape, empty

from struphy.linear_algebra.core import det

from struphy.geometry.mappings_3d import df_ij, df, det_df


# ==============================================================================
def push_0_form(a0 : float, eta1 : float, eta2 : float, eta3 : float, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:, :]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
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


# ==============================================================================
def push_1_form(a1_1 : float, a1_2 : float, a1_3 : float, eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:, :]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    """
    Point-wise push-forward of components of a differential 1-form to Cartesian components of a vector field.
    
    Parameters
    ----------
        a1_1, a1_2, a1_3 : float
            Values of components of differential 1-form.
            
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
    
    a = 0.
    
    df_mat = empty((3, 3), dtype=float)
    
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_mat)
    
    detdf = det(df_mat)
    
    if   component == 1:
        
        a = (df_mat[1, 1]*df_mat[2, 2] - df_mat[2, 1]*df_mat[1, 2])*a1_1 + (df_mat[1, 2]*df_mat[2, 0] - df_mat[2, 2]*df_mat[1, 0])*a1_2 + (df_mat[1, 0]*df_mat[2, 1] - df_mat[2, 0]*df_mat[1, 1])*a1_3 
    
    elif component == 2:
        
        a = (df_mat[2, 1]*df_mat[0, 2] - df_mat[0, 1]*df_mat[2, 2])*a1_1 + (df_mat[2, 2]*df_mat[0, 0] - df_mat[0, 2]*df_mat[2, 0])*a1_2 + (df_mat[2, 0]*df_mat[0, 1] - df_mat[0, 0]*df_mat[2, 1])*a1_3 
        
    elif component == 3:
        
        a = (df_mat[0, 1]*df_mat[1, 2] - df_mat[1, 1]*df_mat[0, 2])*a1_1 + (df_mat[0, 2]*df_mat[1, 0] - df_mat[1, 2]*df_mat[0, 0])*a1_2 + (df_mat[0, 0]*df_mat[1, 1] - df_mat[1, 0]*df_mat[0, 1])*a1_3
        
    a = a / detdf
        
    return a


# ==============================================================================
def push_2_form(a2_1 : float, a2_2 : float, a2_3 : float, eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:, :]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    """
    Point-wise push-forward of components of a differential 2-form to Cartesian components of a vector field.
    
    Parameters
    ----------
        a2_1, a2_2, a2_3 : float
            Values of components of differential 2-form.
            
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
    
    a = 0.
    
    df_mat = empty((3, 3), dtype=float)
    
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_mat)
    
    detdf = det(df_mat)
    
    if   component == 1:
        
        a = (df_mat[0, 0]*a2_1 + df_mat[0, 1]*a2_2 + df_mat[0, 2]*a2_3) / abs(detdf)
    
    elif component == 2:
        
        a = (df_mat[1, 0]*a2_1 + df_mat[1, 1]*a2_2 + df_mat[1, 2]*a2_3) / abs(detdf)
        
    elif component == 3:
        
        a = (df_mat[2, 0]*a2_1 + df_mat[2, 1]*a2_2 + df_mat[2, 2]*a2_3) / abs(detdf)
        
    return a


# ==============================================================================
def push_3_form(a3 : float, eta1 : float, eta2 : float, eta3 : float, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:, :]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
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
    
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    a = a3 / abs(detdf)
    
    return a


# ==============================================================================
def push_vector(a_1 : float, a_2 : float, a_3 : float, eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:, :]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    """
    Point-wise push-forward of contravariant components of a vector field to Cartesian components of a vector field.
    
    Parameters
    ----------
        a_1, a_2, a_3 : float
            Values of contravariant components of vector field.
            
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
    
    a = 0.
    
    if   component == 1:
        
        df_11 = df_ij(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_12 = df_ij(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_13 = df_ij(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        
        a = df_11*a_1 + df_12*a_2 + df_13*a_3
    
    elif component == 2:
        
        df_21 = df_ij(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_22 = df_ij(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_23 = df_ij(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        
        a = df_21*a_1 + df_22*a_2 + df_23*a_3
        
    elif component == 3:
        
        df_31 = df_ij(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_32 = df_ij(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_33 = df_ij(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        
        a = df_31*a_1 + df_32*a_2 + df_33*a_3
        
    return a


# ==============================================================================
def push_all(a : 'float[:]', eta1 : float, eta2 : float, eta3 : float, kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:, :]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
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
    if   kind_fun == 0:
        value = push_0_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # 3-form
    elif kind_fun == 3:
        value = push_3_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # 1-form
    elif kind_fun == 11:
        value = push_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 12:
        value = push_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 13:
        value = push_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # 2-form
    elif kind_fun == 21:
        value = push_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 22:
        value = push_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 23:
        value = push_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # vector
    elif kind_fun == 31:
        value = push_vector(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 32:
        value = push_vector(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 33:
        value = push_vector(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    return value


# ==============================================================================
def kernel_evaluate(a : 'float[:,:,:,:]', eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:, :]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', values : 'float[:,:,:]'):
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
            
    Returns
    -------
        values : float
            Scalar fied or Cartesian component resulting from the push-forward.
    """

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                values[i1, i2, i3] = push_all(a[:, i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)


# ==============================================================================
def kernel_evaluate_sparse(a  : 'float[:,:,:,:]', eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:, :]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', values : 'float[:,:,:]'):
    """
    Push-forward of a differential k-form or contravariant vector field to Cartesian scalar/vector field using sparse meshgrids.
    
    Parameters
    ----------
        a : array[float]
            Values of 0/3-form [a_ijk] or values of 1/2-form or contravariant vector field [a_mu,ijk].
            
        eta1, eta2, eta3 : array[float]
            Evaluation points in 3d arrays such that shape(eta1) = (:,1,1), shape(eta2) = (1,:,1), shape(eta3) = (1,1,:).
            
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
        values : float
            Scalar fied or Cartesian component resulting from the push-forward.
    """

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                values[i1, i2, i3] = push_all(a[:, i1, i2, i3], eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)


# ==============================================================================
def kernel_evaluate_flat(a : 'float[:,:]', eta1 : 'float[:]', eta2 : 'float[:]', eta3 : 'float[:]', kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:, :]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', values : 'float[:]'):
    """
    Push-forward of a differential k-form or contravariant vector field to Cartesian scalar/vector field using flat evaluation.
    
    Parameters
    ----------
        a : array[float]
            Values of 0/3-form [a_ijk] or values of 1/2-form or contravariant vector field [a_mu,ijk].
            
        eta1, eta2, eta3 : array[float]
            Evaluation points in 1d arrays such that len(eta1) == len(eta2) == len(eta3).
            
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
        values : float
            Scalar fied or Cartesian component resulting from the push-forward.
    """

    for i in range(len(eta1)):
        values[i] = push_all(a[:, i], eta1[i], eta2[i], eta3[i], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)