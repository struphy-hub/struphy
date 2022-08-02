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

from numpy import shape, empty

from struphy.linear_algebra.core import det

from struphy.geometry.mappings_3d import df_ij, df, det_df


# ==============================================================================
def pull_0_form(a : float, eta1 : float, eta2 : float, eta3 : float, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
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


# ==============================================================================
def pull_1_form(ax : float, ay : float, az : float, eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    """
    Point-wise pull-back of a Cartesian vector field to a differential 1-form.
    
    Parameters
    ----------
        ax, ay, az : float
            Values of Cartesian components of vector field.
            
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
    
    a = 0.
    
    if   component == 1:
        
        df_11 = df_ij(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_21 = df_ij(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_31 = df_ij(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        
        a = df_11*ax + df_21*ay + df_31*az
    
    elif component == 2:
        
        df_12 = df_ij(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_22 = df_ij(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_32 = df_ij(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        
        a = df_12*ax + df_22*ay + df_32*az
        
    elif component == 3:
        
        df_13 = df_ij(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_23 = df_ij(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        df_33 = df_ij(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
        
        a = df_13*ax + df_23*ay + df_33*az
        
    return a


# ==============================================================================
def pull_2_form(ax : float, ay : float, az : float, eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    """
    Point-wise pull-back of a Cartesian vector field to a differential 2-form.
    
    Parameters
    ----------
        ax, ay, az : float
            Values of Cartesian components of vector field.
            
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
    
    a = 0.
    
    df_mat = empty((3, 3), dtype=float)
    
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_mat)
    
    detdf = det(df_mat)
    
    if   component == 1:
        
        a = (df_mat[1, 1]*df_mat[2, 2] - df_mat[2, 1]*df_mat[1, 2])*ax + (df_mat[2, 1]*df_mat[0, 2] - df_mat[0, 1]*df_mat[2, 2])*ay + (df_mat[0, 1]*df_mat[1, 2] - df_mat[1, 1]*df_mat[0, 2])*az 
    
    elif component == 2:
        
        a = (df_mat[1, 2]*df_mat[2, 0] - df_mat[2, 2]*df_mat[1, 0])*ax + (df_mat[2, 2]*df_mat[0, 0] - df_mat[0, 2]*df_mat[2, 0])*ay + (df_mat[0, 2]*df_mat[1, 0] - df_mat[1, 2]*df_mat[0, 0])*az 
        
    elif component == 3:
        
        a = (df_mat[1, 0]*df_mat[2, 1] - df_mat[2, 0]*df_mat[1, 1])*ax + (df_mat[2, 0]*df_mat[0, 1] - df_mat[0, 0]*df_mat[2, 1])*ay + (df_mat[0, 0]*df_mat[1, 1] - df_mat[1, 0]*df_mat[0, 1])*az
         
    if detdf < 0.:
        a = -a
        
    return a


# ==============================================================================
def pull_3_form(a : float, eta1 : float, eta2 : float, eta3 : float, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
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
    
    detdf = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    
    a3 = a * abs(detdf)
    
    return a3

    
# ==============================================================================
def pull_vector(ax : float, ay : float, az : float, eta1 : float, eta2 : float, eta3 : float, component : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
    """
    Point-wise pull-back of a Cartesian vector field to contravariant components.
    
    Parameters
    ----------
        ax, ay, az : float
            Values of Cartesian components of vector field.
            
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
    
    a = 0.
    
    df_mat = empty((3, 3), dtype=float)
    
    df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz, df_mat)
    
    detdf = det(df_mat)
    
    if   component == 1:
        
        a = (df_mat[1, 1]*df_mat[2, 2] - df_mat[2, 1]*df_mat[1, 2])*ax + (df_mat[2, 1]*df_mat[0, 2] - df_mat[0, 1]*df_mat[2, 2])*ay + (df_mat[0, 1]*df_mat[1, 2] - df_mat[1, 1]*df_mat[0, 2])*az 
    
    elif component == 2:
        
        a = (df_mat[1, 2]*df_mat[2, 0] - df_mat[2, 2]*df_mat[1, 0])*ax + (df_mat[2, 2]*df_mat[0, 0] - df_mat[0, 2]*df_mat[2, 0])*ay + (df_mat[0, 2]*df_mat[1, 0] - df_mat[1, 2]*df_mat[0, 0])*az 
        
    elif component == 3:
        
        a = (df_mat[1, 0]*df_mat[2, 1] - df_mat[2, 0]*df_mat[1, 1])*ax + (df_mat[2, 0]*df_mat[0, 1] - df_mat[0, 0]*df_mat[2, 1])*ay + (df_mat[0, 0]*df_mat[1, 1] - df_mat[1, 0]*df_mat[0, 1])*az
        
    a = a / detdf
        
    return a


# ==============================================================================
def pull_all(a : 'float[:]', eta1 : float, eta2 : float, eta3 : float, kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float:
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
    if   kind_fun == 0:
        value = pull_0_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # 3-form
    elif kind_fun == 3:
        value = pull_3_form(a[0], eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # 1-form
    elif kind_fun == 11:
        value = pull_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 12:
        value = pull_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 13:
        value = pull_1_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # 2-form
    elif kind_fun == 21:
        value = pull_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 22:
        value = pull_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 23:
        value = pull_2_form(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    # vector
    elif kind_fun == 31:
        value = pull_vector(a[0], a[1], a[2], eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 32:
        value = pull_vector(a[0], a[1], a[2], eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
    elif kind_fun == 33:
        value = pull_vector(a[0], a[1], a[2], eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)

    return value


# ==============================================================================
def kernel_evaluate(a : 'float[:,:,:,:]', eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', values : 'float[:,:,:]'):
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
                values[i1, i2, i3] = pull_all(a[:, i1, i2, i3], eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)


# ==============================================================================
def kernel_evaluate_sparse(a : 'float[:,:,:,:]', eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', values : 'float[:,:,:]'):
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
                values[i1, i2, i3] = pull_all(a[:, i1, i2, i3], eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)


# ==============================================================================
def kernel_evaluate_flat(a : 'float[:,:]', eta1 : 'float[:]', eta2 : 'float[:]', eta3 : 'float[:]', kind_fun : int, kind_map : int, params_map : 'float[:]', tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', values : 'float[:]'):
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
        values[i] = pull_all(a[:, i], eta1[i], eta2[i], eta3[i], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, ind_n1, ind_n2, ind_n3, cx, cy, cz)
