# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Basic functions for point-wise evaluation of a 2d analytical or discrete spline mapping mapping and its corresponding geometric quantities:

- Jacobian matrix (df)
- inverse Jacobian matrix (df_inv)
- Jacobian determinant (det_df)
- metric tensor (g)
- inverse metric tensor (g_inv)
"""

from numpy import shape
from numpy import sin, cos, pi, zeros, array, sqrt

from pyccel.decorators import types

import hylife.utilitis_FEEC.basics.spline_evaluation_2d as eva


# =======================================================================
@types('double','double','int','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]')
def f(eta1, eta2, component, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy):
    """
    returns one of the two components of an analytical (kind_map >= 1) or discrete (kind_map = 0) mapping x, y = F(eta1, eta2) in two space dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    component : int
        physical coordinate (1 : x, 2 : y)
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : array_like
        parameters for the mapping (1 : [lx, ly], 2 : [r1, r2], 3 : [lx, ly, alpha], 4 : [lx, ly, alpha])
        
    tn1 : array_like
        spline knot vector in 1-direction
        
    tn2 : array_like
        spline knot vector in 2-direction
        
    pn : array_like
        spline degrees in all directions
        
    nbase_n : array_like
        number of splines in all directions
        
    cx : array_like
        control points of x-component
        
    cy : array_like
        control points of y-component
    """
   
   
    value = 0.
    
    # =========== discrete =================
    if kind_map == 0:
        
        if   component == 1:
            value = eva.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx, eta1, eta2)

        elif component == 2:
            value = eva.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy, eta1, eta2)


    # ============== slab ==================
    elif kind_map == 1:
         
        lx = params[0] 
        ly = params[1] 
        
        if   component == 1:
            value = lx * eta1
        elif component == 2:
            value = ly * eta2
            
    # ============= annulus ================
    elif kind_map == 2:
        
        r1 = params[0]
        r2 = params[1]
        dr = r2 - r1
        
        if   component == 1:
            value = (eta1 * dr + r1) * cos(2*pi*eta2)
        elif component == 2:
            value = (eta1 * dr + r1) * sin(2*pi*eta2)
            
    # ============ colella =================
    elif kind_map == 3:
        
        lx    = params[0]
        ly    = params[1]
        alpha = params[2]
        
        if   component == 1:
            value = lx * (eta1 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        elif component == 2:
            value = ly * (eta2 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
    
    # =========== orthogonal ===============
    elif kind_map == 4:
        
        lx    = params[0]
        ly    = params[1]
        alpha = params[2]
        
        if   component == 1:
            value = lx * (eta1 + alpha * sin(2*pi*eta1))
        elif component == 2:
            value = ly * (eta2 + alpha * sin(2*pi*eta2))
            
    return value


# =======================================================================
@types('double','double','int','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]')
def df(eta1, eta2, component, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy):
    """
    returns one of the four components of the Jacobian matrix DF_ij = dF_i/deta_j of an analytical (kind_map >= 1) or discrete (kind_map = 0) mapping x, y = F(eta1, eta2) in two space dimensions.
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    component : int
        11 : (dx/deta1), 12 : (dx/deta2)
        21 : (dy/deta1), 22 : (dy/deta2)
    
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : array_like
        parameters for the mapping (1 : [lx, ly], 2 : [r1, r2], 3 : [lx, ly, alpha], 4 : [lx, ly, alpha])
        
    tn1 : array_like
        spline knot vector in 1-direction
        
    tn2 : array_like
        spline knot vector in 2-direction
        
    pn : array_like
        spline degrees in all directions
        
    nbase_n : array_like
        number of splines in all directions
        
    cx : array_like
        control points of x-component
        
    cy : array_like
        control points of y-component
    """
    
    value = 0.
    
    # ============ discrete ================
    if kind_map == 0:
        
        if   component == 11:
            value = eva.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx, eta1, eta2)
        elif component == 12:
            value = eva.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx, eta1, eta2)

        elif component == 21:
            value = eva.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy, eta1, eta2)
        elif component == 22:
            value = eva.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy, eta1, eta2)
               
    # ============== slab ==================
    elif kind_map == 1:
         
        lx = params[0] 
        ly = params[1] 
        
        if   component == 11:
            value = lx
        elif component == 12:
            value = 0.
        elif component == 21:
            calue = 0.
        elif component == 22:
            value = ly
            
    # ============ annulus =================
    elif kind_map == 2:
        
        r1 = params[0]
        r2 = params[1]
        dr = r2 - r1
        
        if   component == 11:
            value = dr * cos(2*pi*eta2)
        elif component == 12:
            value = -2*pi * (eta1*dr + r1) * sin(2*pi*eta2)
        elif component == 21:
            value = dr * sin(2*pi*eta2)
        elif component == 22:
            value = 2*pi * (eta1*dr + r1) * cos(2*pi*eta2)
            
    # ============ colella =================
    elif kind_map == 3:
        
        lx    = params[0]
        ly    = params[1]
        alpha = params[2]
        
        if   component == 11:
            value = lx * (1 + alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi)
        elif component == 12:
            value = lx * alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi
        elif component == 21:
            value = ly * alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi
        elif component == 22:
            value = ly * (1 + alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi)
                   
    # =========== orthogonal ================
    elif kind_map == 4:
        
        lx    = params[0]
        ly    = params[1]
        alpha = params[2]
        
        if   component == 11:
            value = lx * (1 + alpha * cos(2*pi*eta1) * 2*pi)
        elif component == 12:
            value = 0.
        elif component == 21:
            value = 0.
        elif component == 22:
            value = ly * (1 + alpha * cos(2*pi*eta2) * 2*pi)
            
    return value


# =======================================================================
@types('double','double','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]')
def det_df(eta1, eta2, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy):
    """
    returns the Jacobian determinant det(DF)=DF_11*DF_22 - DF_12*DF_21 of an analytical (kind_map >= 1) or discrete (kind_map = 0) mapping x, y = F(eta1, eta2) in two space dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : array_like
        parameters for the mapping (1 : [lx, ly], 2 : [r1, r2], 3 : [lx, ly, alpha], 4 : [lx, ly, alpha])
        
    tn1 : array_like
        spline knot vector in 1-direction
        
    tn2 : array_like
        spline knot vector in 2-direction
   
    pn : array_like
        spline degrees in all directions
        
    nbase_n : array_like
        number of splines in all directions
        
    cx : array_like
        control points of x-component
        
    cy : array_like
        control points of y-component
                 
    Returns
    -------
    value : double
        the Jacobian determinant (DF_11*DF_22 - DF_12*DF_21)
    """
    
    value = 0.
    
    df_11 = df(eta1, eta2, 11, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    df_12 = df(eta1, eta2, 12, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    
    df_21 = df(eta1, eta2, 21, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    df_22 = df(eta1, eta2, 22, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    
    value = df_11*df_22 - df_21*df_12
            
    return value


# =======================================================================
@types('double','double','int','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]')
def df_inv(eta1, eta2, component, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy):
    """
    returns one of the four components of the inverse Jacobian matrix of an analytical (kind_map >= 1) or discrete (kind_map = 0) mapping x, y = F(eta1, eta2) in two space dimensions. 
    
    the 2x2 inverse is computed directly from DF:

                            |  DF_22 -DF_12 |
    (DF)^(-1) = 1/det_df *  | -DF_21  DF_11 |
    
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    component : int
        the component of the inverse Jacobian matrix
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : array_like
        parameters for the mapping (1 : [lx, ly], 2 : [r1, r2], 3 : [lx, ly, alpha], 4 : [lx, ly, alpha])
        
    tn1 : array_like
        spline knot vector in 1-direction
        
    tn2 : array_like
        spline knot vector in 2-direction
        
    pn : array_like
        spline degrees in all directions
        
    nbase_n : array_like
        number of splines in all directions
        
    cx : array_like
        control points of x-component
        
    cy : array_like
        control points of y-component
    """
    
    value = 0.

    df_11 = df(eta1, eta2, 11, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    df_12 = df(eta1, eta2, 12, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    
    df_21 = df(eta1, eta2, 21, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    df_22 = df(eta1, eta2, 22, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)

    detdf = df_11*df_22 - df_21*df_12

    if   component == 11:
        value =  df_22/detdf
    elif component == 12:
        value = -df_12/detdf
    elif component == 21:
        value = -df_21/detdf
    elif component == 22:
        value =  df_11/detdf
            
    return value


# =======================================================================
@types('double','double','int','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]')
def g(eta1, eta2, component, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy):
    """
    returns one of the four components of the metric tensor G = DF^T * DF of an analytical (kind_map >= 1) or discrete (kind_map = 0) mapping x, y = F(eta1, eta2) in two space dimensions.
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
    
    component : int
        the component of the metric tensor
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : array_like
        parameters for the mapping (1 : [lx, ly], 2 : [r1, r2], 3 : [lx, ly, alpha], 4 : [lx, ly, alpha])
        
    tn1 : array_like
        spline knot vector in 1-direction
        
    tn2 : array_like
        spline knot vector in 2-direction
        
    pn : array_like
        spline degrees in all directions
        
    nbase_n : array_like
        number of splines in all directions
        
    cx : array_like
        control points of x-component
        
    cy : array_like
        control points of y-component
    """
    
    value = 0.

    if   component == 11:
        df_11 = df(eta1, eta2, 11, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        df_21 = df(eta1, eta2, 21, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        value = df_11*df_11 + df_21*df_21
        
    elif component == 22:                                              
        df_12 = df(eta1, eta2, 12, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        df_22 = df(eta1, eta2, 22, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        value = df_12*df_12 + df_22*df_22
                 
    elif ((component == 12) or (component == 21)) :
        df_11 = df(eta1, eta2, 11, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        df_21 = df(eta1, eta2, 21, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        df_12 = df(eta1, eta2, 12, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        df_22 = df(eta1, eta2, 22, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        value = df_11*df_12 + df_21*df_22
               
    return value


# =======================================================================
@types('double','double','int','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]')
def g_inv(eta1, eta2, component, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy):
    """
    returns one of the four components of the inverse metric tensor G = DF^(-1) * DF^(-T) of an analytical (kind_map >= 1) or discrete (kind_map = 0) mapping x, y = F(eta1, eta2) in thwo space dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    component : int
        the component of the inverse metric tensor
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : array_like
        parameters for the mapping (1 : [lx, ly], 2 : [r1, r2], 3 : [lx, ly, alpha], 4 : [lx, ly, alpha])
        
    tn1 : array_like
        spline knot vector in 1-direction
        
    tn2 : array_like
        spline knot vector in 2-direction
    
    pn : array_like
        spline degrees in all directions
        
    nbase_n : array_like
        number of splines in all directions
        
    cx : array_like
        control points of x-component
        
    cy : array_like
        control points of y-component
    """
    
    value = 0.

    if   component == 11:
        dfinv_11 = df_inv(eta1, eta2, 11, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        dfinv_12 = df_inv(eta1, eta2, 12, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        value    = dfinv_11*dfinv_11 + dfinv_12*dfinv_12
                  
    elif component == 22:                                              
        dfinv_21 = df_inv(eta1, eta2, 21, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        dfinv_22 = df_inv(eta1, eta2, 22, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        value    = dfinv_21*dfinv_21 + dfinv_22*dfinv_22
                  
    elif ((component == 12) or (component == 21)) :
        dfinv_11 = df_inv(eta1, eta2, 11, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        dfinv_12 = df_inv(eta1, eta2, 12, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        dfinv_21 = df_inv(eta1, eta2, 21, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        dfinv_22 = df_inv(eta1, eta2, 22, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        value    = dfinv_11*dfinv_21 + dfinv_12*dfinv_22
    
    return value


# ==========================================================================================
@types('double','double','int','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]')
def all_mappings(eta1, eta2, kind_fun, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy):
    
    value = 0.
    
    # mapping f
    if   kind_fun == 1:
        value = f(eta1, eta2, 1, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    elif kind_fun == 2:
        value = f(eta1, eta2, 2, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    
    # Jacobian matrix df
    elif kind_fun == 11:
        value = df(eta1, eta2, 11, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    elif kind_fun == 12:
        value = df(eta1, eta2, 12, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    elif kind_fun == 13:
        value = df(eta1, eta2, 21, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    elif kind_fun == 14:
        value = df(eta1, eta2, 22, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        
    # Jacobian determinant det_df
    elif kind_fun == 3:
        value = det_df(eta1, eta2, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        
    # inverse Jacobian matrix df_inv
    elif kind_fun == 21:
        value = df_inv(eta1, eta2, 11, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    elif kind_fun == 22:
        value = df_inv(eta1, eta2, 12, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    elif kind_fun == 23:
        value = df_inv(eta1, eta2, 21, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    elif kind_fun == 24:
        value = df_inv(eta1, eta2, 22, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        
    # metric tensor g
    elif kind_fun == 31:
        value = g(eta1, eta2, 11, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    elif kind_fun == 32:
        value = g(eta1, eta2, 12, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    elif kind_fun == 33:
        value = g(eta1, eta2, 21, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    elif kind_fun == 34:
        value = g(eta1, eta2, 22, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
        
    # metric tensor g_inv
    elif kind_fun == 41:
        value = g_inv(eta1, eta2, 11, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    elif kind_fun == 42:
        value = g_inv(eta1, eta2, 12, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    elif kind_fun == 43:
        value = g_inv(eta1, eta2, 21, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    elif kind_fun == 44:
        value = g_inv(eta1, eta2, 22, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
    
    return value


# ==========================================================================================
@types('double[:]','double[:]','int','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]')       
def kernel_evaluate_tensor_product(eta1, eta2, kind_fun, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy, mat_f):
    
    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            mat_f[i1, i2] = all_mappings(eta1[i1], eta2[i2], kind_fun, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)
                
                
# ==========================================================================================
@types('double[:,:]','double[:,:]','int','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]')       
def kernel_evaluate_general(eta1, eta2, kind_fun, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy, mat_f):
    
    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    
    for i1 in range(n1):
        for i2 in range(n2):
            mat_f[i1, i2] = all_mappings(eta1[i1, i2], eta2[i1, i2], kind_fun, kind_map, params, tn1, tn2, pn, nbase_n, cx, cy)