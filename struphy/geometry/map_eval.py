"""
Module containing accelerated (pyccelized) functions for evaluation of metric coefficients 
corresponding to mappings (x, y, z) = F(eta_1, eta_2, eta_3).
"""

from numpy import shape, empty, zeros
from numpy import sin, cos, pi, sqrt, arctan2, arcsin

import struphy.geometry.mappings_3d_bis as maps
from struphy.linear_algebra.core import det, matrix_matrix, transpose, matrix_inv


def f(eta1 : float, eta2 : float, eta3 : float, # evaluation point
      kind_map : int, params : 'float[:]', # mapping parameters
      t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', # spline mapping knots and degrees
      ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', # spline index arrays
      cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', # control points (numpy array, cloned to each process)
      f_out: 'float[:]'): # output array
    """
    Point-wise evaluation of (x, y, z) = F(eta1, eta2, eta3). 

    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate B-splines.
        
        p : array[int]
            Degrees of univariate B-splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.

        f_out : array[float]
            Output: (x, y, z) = F(eta1, eta2, eta3).
    """

    if kind_map == 0:
        maps.spline_3d(eta1, eta2, eta3, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, f_out)
    elif kind_map == 1:
        maps.spline_2d_straight(eta1, eta2, eta3, t1, t2, p[:2], ind1, ind2, cx[:, :, 0], cy[:, :, 0], params[0], f_out)
    elif kind_map == 2:
        maps.spline_2d_torus(eta1, eta2, eta3, t1, t2, p[:2], ind1, ind2, cx[:, :, 0], cy[:, :, 0], f_out)
    elif kind_map == 10:
        maps.cuboid(eta1, eta2, eta3, params[0], params[1], params[2], params[3], params[4], params[5], f_out)
    elif kind_map == 11:
        maps.hollow_cyl(eta1, eta2, eta3, params[0], params[1], params[2], params[3], f_out)
    elif kind_map == 12:
        maps.colella(eta1, eta2, eta3, params[0], params[1], params[2], params[3], f_out)
    elif kind_map == 13:
        maps.orthogonal(eta1, eta2, eta3, params[0], params[1], params[2], params[3], f_out)
    elif kind_map == 14:
        maps.hollow_torus(eta1, eta2, eta3, params[0], params[1], params[2], f_out)
    elif kind_map == 15:
        maps.ellipse(eta1, eta2, eta3, params[0], params[1], params[2], params[3], params[4], params[5], f_out)
    elif kind_map == 16:
        maps.rotated_ellipse(eta1, eta2, eta3, params[0], params[1], params[2], params[3], params[4], params[5], params[6], f_out)
    elif kind_map == 17:
        maps.shafranov_shift(eta1, eta2, eta3, params[0], params[1], params[2], params[3], params[4], params[5], params[6], f_out)
    elif kind_map == 18:
        maps.shafranov_sqrt(eta1, eta2, eta3, params[0], params[1], params[2], params[3], params[4], params[5], params[6], f_out)
    elif kind_map == 19:
        maps.shafranov_dshaped(eta1, eta2, eta3, params[0], params[1], params[2], params[3], params[4], 
                                                 params[5], params[6], params[7], params[8], params[9], f_out)

 
def df(eta1 : float, eta2 : float, eta3 : float, # evaluation point
      kind_map : int, params : 'float[:]', # mapping parameters
      t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', # spline mapping knots and degrees
      ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', # spline index arrays
      cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', # control points (numpy array, cloned to each process)
      df_out: 'float[:,:]'): # output array
    """
    Point-wise evaluation of the Jacobian matrix DF = (dF_i/deta_j)_(i,j=1,2,3). 

    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.

        df_out : array[float]
            Output: DF(eta1, eta2, eta3).
    """

    if kind_map == 0:
        maps.spline_3d_df(eta1, eta2, eta3, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_out)
    elif kind_map == 1:
        maps.spline_2d_straight_df(eta1, eta2, t1, t2, p[:2], ind1, ind2, cx[:, :, 0], cy[:, :, 0], params[0], df_out)
    elif kind_map == 2:
        maps.spline_2d_torus_df(eta1, eta2, eta3, t1, t2, p[:2], ind1, ind2, cx[:, :, 0], cy[:, :, 0], df_out)
    elif kind_map == 10:
        maps.cuboid_df(params[0], params[1], params[2], params[3], params[4], params[5], df_out)
    elif kind_map == 11:
        maps.hollow_cyl_df(eta1, eta2, params[0], params[1], params[2], df_out)
    elif kind_map == 12:
        maps.colella_df(eta1, eta2, params[0], params[1], params[2], params[3], df_out)
    elif kind_map == 13:
        maps.orthogonal_df(eta1, eta2, params[0], params[1], params[2], params[3], df_out)
    elif kind_map == 14:
        maps.hollow_torus_df(eta1, eta2, eta3, params[0], params[1], params[2], df_out)
    elif kind_map == 15:
        maps.ellipse_df(eta1, eta2, eta3, params[3], params[4], params[5], df_out)
    elif kind_map == 16:
        maps.rotated_ellipse_df(eta1, eta2, eta3, params[3], params[4], params[5], params[6], df_out)
    elif kind_map == 17:
        maps.shafranov_shift_df(eta1, eta2, eta3, params[3], params[4], params[5], params[6], df_out)
    elif kind_map == 18:
        maps.shafranov_sqrt_df(eta1, eta2, eta3, params[3], params[4], params[5], params[6], df_out)
    elif kind_map == 19:
        maps.shafranov_dshaped_df(eta1, eta2, eta3, params[3], params[4], params[5], params[6], params[7], params[8], params[9], df_out)

  
def det_df(eta1 : float, eta2 : float, eta3 : float, # evaluation point
      kind_map : int, params : 'float[:]', # mapping parameters
      t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', # spline mapping knots and degrees
      ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', # spline index arrays
      cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float: # control points (numpy array, cloned to each process)
    """
    Point-wise evaluation of the Jacobian determinant det(df) = df/deta1.dot(df/deta2 x df/deta3). 
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.

    Returns:
    --------
        detdf : float
            Jacobian determinant det(df)(eta1, eta2, eta3).
    """
    
    df_mat = empty((3, 3), dtype=float)
    df(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
    
    detdf = det(df_mat)
            
    return detdf


def df_inv(eta1 : float, eta2 : float, eta3 : float, # evaluation point
      kind_map : int, params : 'float[:]', # mapping parameters
      t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', # spline mapping knots and degrees
      ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', # spline index arrays
      cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', # control points (numpy array, cloned to each process)
      dfinv_out: 'float[:,:]'): # output array
    """
    Point-wise evaluation of ij-th component of the inverse Jacobian matrix df^(-1)_ij (i,j=1,2,3). 
    
    The 3 x 3 inverse is computed directly from df, using the cross product of the columns of df:

                            | [ (df/deta2) x (df/deta3) ]^T |
    (df)^(-1) = 1/det_df *  | [ (df/deta3) x (df/deta1) ]^T |
                            | [ (df/deta1) x (df/deta2) ]^T |
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.

        dfinv_out : array[float]
            Output: the inverse Jacobian matrix at (eta1, eta2, eta3).
    """
    
    df_mat = empty((3, 3), dtype=float)
    df(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
    
    matrix_inv(df_mat, dfinv_out)
    
    
def g(eta1 : float, eta2 : float, eta3 : float, # evaluation point
      kind_map : int, params : 'float[:]', # mapping parameters
      t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', # spline mapping knots and degrees
      ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', # spline index arrays
      cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', # control points (numpy array, cloned to each process)
      g_out: 'float[:,:]'): # output array
    """
    Point-wise evaluation of the metric tensor g = df^T * df. 
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.

        g_out : array[float]
            Output: g(eta1, eta2, eta3).
    """
    
    df_mat = empty((3, 3), dtype=float)
    df(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)

    df_t = empty((3, 3), dtype=float)
    transpose(df_mat, df_t)

    matrix_matrix(df_t, df_mat, g_out)


def g_inv(eta1 : float, eta2 : float, eta3 : float, # evaluation point
      kind_map : int, params : 'float[:]', # mapping parameters
      t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', # spline mapping knots and degrees
      ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', # spline index arrays
      cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', # control points (numpy array, cloned to each process)
      ginv_out: 'float[:,:]'): # output array
    """
    Point-wise evaluation of the inverse metric tensor g^(-1) = df^(-1) * df^(-T). 
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.

        ginv_out : array[float]
            Output: g^(-1)(eta1, eta2, eta3).
    """
    
    g_mat = empty((3, 3), dtype=float)
    g(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)

    matrix_inv(g_mat, ginv_out)
    
    
def mappings_all(eta1 : float, eta2 : float, eta3 : float, # evaluation point
            kind_fun : int, # metric coefficient key
            kind_map : int, params : 'float[:]', # mapping parameters 
            t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', # spline mapping knots and degrees
            ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', # spline index arrays
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]') -> float: # control points (numpy array, cloned to each process)
    """
    Point-wise evaluation of
        - f_i       : mapping x_i = f_i(eta1, eta2, eta3),
        - df_ij     : Jacobian matrix df_i/deta_j,
        - det_df    : Jacobian determinant det(df),
        - df_inv_ij : inverse Jacobian matrix (df_i/deta_j)^(-1),
        - g_ij      : metric tensor df^T * df,
        - g_inv_ij  : inverse metric tensor df^(-1) * df^(-T).
    
    Parameters
    ----------
        eta1, eta2, eta3 : float              
            Logical coordinates in [0, 1].
            
        kind_fun : int
            Which metric coefficient to evaluate
        
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.

    Returns:
    --------
        value : float
            Point value of metric coefficient at (eta1, eta2, eta3).
    """
    
    value = 0.
    
    # mapping f
    if   kind_fun == 1:
        f_vec = empty(3, dtype=float)
        f(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, f_vec)
        value = f_vec[0]
    elif kind_fun == 2:
        f_vec = empty(3, dtype=float)
        f(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, f_vec)
        value = f_vec[1]
    elif kind_fun == 3:
        f_vec = empty(3, dtype=float)
        f(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, f_vec)
        value = f_vec[2]
    
    # Jacobian matrix df
    elif kind_fun == 11:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[0, 0]
    elif kind_fun == 12:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[0, 1]
    elif kind_fun == 13:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[0, 2]
    elif kind_fun == 14:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[1, 0]
    elif kind_fun == 15:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[1, 1]
    elif kind_fun == 16:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[1, 2]
    elif kind_fun == 17:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[2, 0]
    elif kind_fun == 18:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[2, 1]
    elif kind_fun == 19:
        df_mat = empty((3, 3), dtype=float)
        df(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
        value = df_mat[2, 2]
        
    # Jacobian determinant det_df
    elif kind_fun == 4:
        value = det_df(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz)
        
    # inverse Jacobian matrix df_inv
    elif kind_fun == 21:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[0, 0]
    elif kind_fun == 22:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[0, 1]
    elif kind_fun == 23:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[0, 2]
    elif kind_fun == 24:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[1, 0]
    elif kind_fun == 25:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[1, 1]
    elif kind_fun == 26:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[1, 2]
    elif kind_fun == 27:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[2, 0]
    elif kind_fun == 28:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[2, 1]
    elif kind_fun == 29:
        dfinv_mat = empty((3, 3), dtype=float)
        df_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, dfinv_mat)
        value = dfinv_mat[2, 2]
        
    # metric tensor g
    elif kind_fun == 31:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[0, 0]
    elif kind_fun == 32:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[0, 1]
    elif kind_fun == 33:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[0, 2]
    elif kind_fun == 34:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[1, 0]
    elif kind_fun == 35:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[1, 1]
    elif kind_fun == 36:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[1, 2]
    elif kind_fun == 37:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[2, 0]     
    elif kind_fun == 38:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[2, 1]
    elif kind_fun == 39:
        g_mat = empty((3, 3), dtype=float)
        g(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, g_mat)
        value = g_mat[2, 2]
    
    # metric tensor g_inv
    elif kind_fun == 41:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[0, 0]
    elif kind_fun == 42:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[0, 1]
    elif kind_fun == 43:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[0, 2]  
    elif kind_fun == 44:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[1, 0]
    elif kind_fun == 45:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[1, 1]
    elif kind_fun == 46:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[1, 2]
    elif kind_fun == 47:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[2, 0]
    elif kind_fun == 48:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[2, 1]
    elif kind_fun == 49:
        ginv_mat = empty((3, 3), dtype=float)
        g_inv(eta1, eta2, eta3, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz, ginv_mat)
        value = ginv_mat[2, 2]
    
    return value

   
def kernel_evaluate(eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', kind_fun : int, kind_map : int, params : 'float[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mat_f : 'float[:,:,:]'):
    """
    Matrix-wise evaluation of
        - f_i       : mapping x_i = f_i(eta1, eta2, eta3),
        - df_ij     : Jacobian matrix df_i/deta_j,
        - det_df    : Jacobian determinant det(df),
        - df_inv_ij : inverse Jacobian matrix (df_i/deta_j)^(-1),
        - g_ij      : metric tensor df^T * df,
        - g_inv_ij  : inverse metric tensor df^(-1) * df^(-T).

    Parameters
    ----------
        eta1, eta2, eta3 : array[float]              
            Logical coordinatess in 3d arrays with shape(eta1) == shape(eta2) == shape(eta3).
            
        kind_fun : int
            Which metric coefficient to evaluate
        
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.
            
        mat_f : array[float]
            matrix-valued mapping/metric coefficient evaluated at (eta1, eta2, eta3).
    """

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                mat_f[i1, i2, i3] = mappings_all(eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_fun, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz)

     
def kernel_evaluate_sparse(eta1 : 'float[:,:,:]', eta2 : 'float[:,:,:]', eta3 : 'float[:,:,:]', kind_fun : int, kind_map : int, params : 'float[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mat_f : 'float[:,:,:]'):
    """
    Same as kernel_evaluate, but for sparse meshgrids.
    
    Parameters
    ----------
        eta1, eta2, eta3 : array[float]              
            Logical coordinatess in 3d arrays with shape(eta1) = (:,1,1), shape(eta2) = (1,:,1), shape(eta3) = (1,1,:).
            
        kind_fun : int
            Which metric coefficient to evaluate
        
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.
            
        mat_f : array[float]
            matrix-valued mapping/metric coefficient evaluated at (eta1, eta2, eta3).
    """

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                mat_f[i1, i2, i3] = mappings_all(eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3], kind_fun, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz)

                     
def kernel_evaluate_flat(eta1 : 'float[:]', eta2 : 'float[:]', eta3 : 'float[:]', kind_fun : int, kind_map : int, params : 'float[:]', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mat_f : 'float[:]'):
    """
    Same as kernel_evaluate, but for flat evaluation.
    
    Parameters
    ----------
        eta1, eta2, eta3 : array[float]              
            Logical coordinatess in 1d arrays with len(eta1) == len(eta2) == len(eta3).
            
        kind_fun : int
            Which metric coefficient to evaluate
        
        kind_map : int                 
            Kind of mapping (see module docstring).
        
        params : array[float]
            Parameters for the mapping in a 1d array.
        
        t1, t2, t3 : array[float]          
            Knot vectors of univariate splines.
        
        p : array[int]
            Degrees of univariate splines.
        
        ind1, ind2, ind3 : array[int]                 
            Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
        
        cx, cy, cz : array[float]     
            Control points of (F_x, F_y, F_z) in case of a IGA mapping.
            
        mat_f : array[float]
            matrix-valued mapping/metric coefficient evaluated at (eta1, eta2, eta3).

    Returns
    -------
        mat_f:  np.array
            1d array [f(x1, y1, z1) f(x2, y2, z2) etc.]
    """

    for i in range(len(eta1)):
        mat_f[i] = mappings_all(eta1[i], eta2[i], eta3[i], kind_fun, kind_map, params, t1, t2, t3, p, ind1, ind2, ind3, cx, cy, cz)


def loop_f(kind_map : int, params : 'float[:]', 
            tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', 
            ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', 
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', 
            eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
            f_out : 'float[:,:]'):

    for n in range(len(eta1s)):

        f(eta1s[n], eta2s[n], eta3s[n],
            kind_map, params,
            tn1, tn2, tn3, pn, 
            ind_n1, ind_n2, ind_n3, 
            cx, cy, cz,
            f_out[n, :])


def loop_df(kind_map : int, params : 'float[:]', 
            tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', 
            ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', 
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', 
            eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
            mat_out : 'float[:,:,:]'):

    for n in range(len(eta1s)):

        df(eta1s[n], eta2s[n], eta3s[n],
            kind_map, params,
            tn1, tn2, tn3, pn, 
            ind_n1, ind_n2, ind_n3, 
            cx, cy, cz,
            mat_out[n, :, :])


def loop_f_and_df(kind_map : int, params : 'float[:]', 
            tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', 
            ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', 
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', 
            eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
            f_out : 'float[:,:]', mat_out : 'float[:,:,:]'):

    for n in range(len(eta1s)):

        f(eta1s[n], eta2s[n], eta3s[n],
            kind_map, params,
            tn1, tn2, tn3, pn, 
            ind_n1, ind_n2, ind_n3, 
            cx, cy, cz,
            f_out[n, :])

        df(eta1s[n], eta2s[n], eta3s[n],
            kind_map, params,
            tn1, tn2, tn3, pn, 
            ind_n1, ind_n2, ind_n3, 
            cx, cy, cz,
            mat_out[n, :, :])
        

def loop_detdf(kind_map : int, params : 'float[:]', 
            tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', 
            ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', 
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', 
            eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
            f_out : 'float[:]'):

    for n in range(len(eta1s)):

        f_out[n] = det_df(eta1s[n], eta2s[n], eta3s[n],
                            kind_map, params,
                            tn1, tn2, tn3, pn, 
                            ind_n1, ind_n2, ind_n3, 
                            cx, cy, cz)


def loop_dfinv(kind_map : int, params : 'float[:]', 
            tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', 
            ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', 
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', 
            eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
            mat_out : 'float[:,:,:]'):

    for n in range(len(eta1s)):

        df_inv(eta1s[n], eta2s[n], eta3s[n],
                kind_map, params,
                tn1, tn2, tn3, pn, 
                ind_n1, ind_n2, ind_n3, 
                cx, cy, cz,
                mat_out[n, :, :])


def loop_g(kind_map : int, params : 'float[:]', 
            tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', 
            ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', 
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', 
            eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
            mat_out : 'float[:,:,:]'):

    for n in range(len(eta1s)):

        g(eta1s[n], eta2s[n], eta3s[n],
                kind_map, params,
                tn1, tn2, tn3, pn, 
                ind_n1, ind_n2, ind_n3, 
                cx, cy, cz,
                mat_out[n, :, :])


def loop_ginv(kind_map : int, params : 'float[:]', 
            tn1 : 'float[:]', tn2 : 'float[:]', tn3 : 'float[:]', pn : 'int[:]', 
            ind_n1 : 'int[:,:]', ind_n2 : 'int[:,:]', ind_n3 : 'int[:,:]', 
            cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', 
            eta1s: 'float[:]', eta2s: 'float[:]', eta3s: 'float[:]',
            mat_out : 'float[:,:,:]'):

    for n in range(len(eta1s)):

        g_inv(eta1s[n], eta2s[n], eta3s[n],
                kind_map, params,
                tn1, tn2, tn3, pn, 
                ind_n1, ind_n2, ind_n3, 
                cx, cy, cz,
                mat_out[n, :, :])