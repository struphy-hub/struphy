# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Basic functions for point-wise evaluation of a 3d analytical or discrete spline mapping mapping and its corresponding geometric quantities:

- Jacobian matrix (df)
- inverse Jacobian matrix (df_inv)
- Jacobian determinant (det_df)
- metric tensor (g)
- inverse metric tensor (g_inv)
"""

from numpy import shape
from numpy import sin, cos, pi, zeros, array, sqrt

from pyccel.decorators import types

import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva


# =======================================================================
@types('double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def f(eta1, eta2, eta3, component, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    """
    returns one of the three components of an analytical (kind_map >= 1) or discrete (kind_map = 0) mapping x, y, z = F(eta1, eta2, eta3) in three space dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    eta3 : double
        3rd logical coordinate in [0, 1]
        
    component : int
        physical coordinate (1 : x, 2 : y, 3 : z)
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : array_like
        parameters for the mapping (1 : [lx, ly, lz], 2 : [r1, r2, lz], 3 : [lx, ly, alpha, lz], 4 : [lx, ly, alpha, lz])
        
    tn1 : array_like
        spline knot vector in 1-direction
        
    tn2 : array_like
        spline knot vector in 2-direction
        
    tn3 : array_like
        spline knot vector in 3-direction
        
    pn : array_like
        spline degrees in all directions
        
    nbase_n : array_like
        number of splines in all directions
        
    cx : array_like
        control points of x-component
        
    cy : array_like
        control points of y-component
        
    cz : array_like
        control points of z-component
    """
   
   
    value = 0.
    
    # =========== discrete =================
    if kind_map == 0:
        
        if   component == 1:
            value = eva.evaluate_n_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cx, eta1, eta2, eta3)

        elif component == 2:
            value = eva.evaluate_n_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cy, eta1, eta2, eta3)

        elif component == 3:
            value = eva.evaluate_n_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cz, eta1, eta2, eta3)


    # ============== slab ==================
    elif kind_map == 1:
         
        lx = params[0] 
        ly = params[1] 
        lz = params[2]
        
        if   component == 1:
            value = lx * eta1
        elif component == 2:
            value = ly * eta2
        elif component == 3:
            value = lz * eta3
            
    # ============= annulus ================
    elif kind_map == 2:
        
        r1 = params[0]
        r2 = params[1]
        lz = params[2]
        dr = r2 - r1
        
        if   component == 1:
            value = (eta1 * dr + r1) * cos(2*pi*eta2)
        elif component == 2:
            value = (eta1 * dr + r1) * sin(2*pi*eta2)
        elif component == 3:
            value = lz * eta3
            
    # ============ colella =================
    elif kind_map == 3:
        
        lx    = params[0]
        ly    = params[1]
        alpha = params[2]
        lz    = params[3]
        
        if   component == 1:
            value = lx * (eta1 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        elif component == 2:
            value = ly * (eta2 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        elif component == 3:
            value = lz * eta3
    
    # =========== orthogonal ===============
    elif kind_map == 4:
        
        lx    = params[0]
        ly    = params[1]
        alpha = params[2]
        lz    = params[3]
        
        if   component == 1:
            value = lx * (eta1 + alpha * sin(2*pi*eta1))
        elif component == 2:
            value = ly * (eta2 + alpha * sin(2*pi*eta2))
        elif component == 3:
            value = lz * eta3
            
    return value


# =======================================================================
@types('double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def df(eta1, eta2, eta3, component, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    """
    returns one of the nine components of the Jacobian matrix DF_ij = dF_i/deta_j of an analytical (kind_map >= 1) or discrete (kind_map = 0) mapping x, y, z = F(eta1, eta2, eta3) in three space dimensions.
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    eta3 : double
        3rd logical coordinate in [0, 1]
        
    component : int
        11 : (dx/deta1), 12 : (dx/deta2), 13 : (dx/deta3)
        21 : (dy/deta1), 22 : (dy/deta2), 23 : (dy/deta3)
        31 : (dz/deta1), 32 : (dz/deta2), 33 : (dz/deta3)
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : array_like
        parameters for the mapping (1 : [lx, ly, lz], 2 : [r1, r2, lz], 3 : [lx, ly, alpha, lz], 4 : [lx, ly, alpha, lz])
        
    tn1 : array_like
        spline knot vector in 1-direction
        
    tn2 : array_like
        spline knot vector in 2-direction
        
    tn3 : array_like
        spline knot vector in 3-direction
        
    pn : array_like
        spline degrees in all directions
        
    nbase_n : array_like
        number of splines in all directions
        
    cx : array_like
        control points of x-component
        
    cy : array_like
        control points of y-component
        
    cz : array_like
        control points of z-component
    """
    
    value = 0.
    
    # ============ discrete ================
    if kind_map == 0:
        
        if   component == 11:
            value = eva.evaluate_diffn_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cx, eta1, eta2, eta3)
        elif component == 12:
            value = eva.evaluate_n_diffn_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cx, eta1, eta2, eta3)
        elif component == 13:
            value = eva.evaluate_n_n_diffn(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cx, eta1, eta2, eta3)
        elif component == 21:
            value = eva.evaluate_diffn_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cy, eta1, eta2, eta3)
        elif component == 22:
            value = eva.evaluate_n_diffn_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cy, eta1, eta2, eta3)
        elif component == 23:
            value = eva.evaluate_n_n_diffn(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cy, eta1, eta2, eta3)
        elif component == 31:
            value = eva.evaluate_diffn_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cz, eta1, eta2, eta3)
        elif component == 32:
            value = eva.evaluate_n_diffn_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cz, eta1, eta2, eta3)
        elif component == 33:
            value = eva.evaluate_n_n_diffn(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cz, eta1, eta2, eta3)
               
    # ============== slab ==================
    elif kind_map == 1:
         
        lx = params[0] 
        ly = params[1] 
        lz = params[2]
        
        if   component == 11:
            value = lx
        elif component == 12:
            value = 0.
        elif component == 13:
            value = 0.
        elif component == 21:
            value = 0.
        elif component == 22:
            value = ly
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = lz
            
    # ============ annulus =================
    elif kind_map == 2:
        
        r1 = params[0]
        r2 = params[1]
        lz = params[2]
        dr = r2 - r1
        
        if   component == 11:
            value = dr * cos(2*pi*eta2)
        elif component == 12:
            value = -2*pi * (eta1*dr + r1) * sin(2*pi*eta2)
        elif component == 13:
            value = 0.
        elif component == 21:
            value = dr * sin(2*pi*eta2)
        elif component == 22:
            value = 2*pi * (eta1*dr + r1) * cos(2*pi*eta2)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = lz
            
    # ============ colella =================
    elif kind_map == 3:
        
        lx    = params[0]
        ly    = params[1]
        alpha = params[2]
        lz    = params[3]
        
        if   component == 11:
            value = lx * (1 + alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi)
        elif component == 12:
            value = lx * alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi
        elif component == 13:
            value = 0.
        elif component == 21:
            value = ly * alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi
        elif component == 22:
            value = ly * (1 + alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = lz
                   
    # =========== orthogonal ================
    elif kind_map == 4:
        
        lx    = params[0]
        ly    = params[1]
        alpha = params[2]
        lz    = params[3]
        
        if   component == 11:
            value = lx * (1 + alpha * cos(2*pi*eta1) * 2*pi)
        elif component == 12:
            value = 0.
        elif component == 13:
            value = 0.
        elif component == 21:
            value = 0.
        elif component == 22:
            value = ly * (1 + alpha * cos(2*pi*eta2) * 2*pi)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = lz
            
    return value



# =======================================================================
@types('double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def det_df(eta1, eta2, eta3, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    """
    returns the Jacobian determinant det(DF)=dF/deta1.( dF/deta2 x dF/deta3) of an analytical (kind_map >= 1) or discrete (kind_map = 0) mapping x, y, z = F(eta1, eta2, eta3) in three space dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    eta3 : double
        3rd logical coordinate in [0, 1]
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : array_like
        parameters for the mapping (1 : [lx, ly, lz], 2 : [r1, r2, lz], 3 : [lx, ly, alpha, lz], 4 : [lx, ly, alpha, lz])
        
    tn1 : array_like
        spline knot vector in 1-direction
        
    tn2 : array_like
        spline knot vector in 2-direction
        
    tn3 : array_like
        spline knot vector in 3-direction
        
    pn : array_like
        spline degrees in all directions
        
    nbase_n : array_like
        number of splines in all directions
        
    cx : array_like
        control points of x-component
        
    cy : array_like
        control points of y-component
        
    cz : array_like
        control points of z-component
                 
    Returns
    -------
    value : double
        the Jacobian determinant dF/deta1 . ( dF/deta2 x dF/deta3)
    """
    
    value = 0.
    
    df_11 = df(eta1, eta2, eta3, 11, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_12 = df(eta1, eta2, eta3, 12, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_13 = df(eta1, eta2, eta3, 13, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_21 = df(eta1, eta2, eta3, 21, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_22 = df(eta1, eta2, eta3, 22, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_23 = df(eta1, eta2, eta3, 23, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_31 = df(eta1, eta2, eta3, 31, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_32 = df(eta1, eta2, eta3, 32, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_33 = df(eta1, eta2, eta3, 33, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    value = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
            
    return value


# =======================================================================
@types('double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def df_inv(eta1, eta2, eta3, component, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    """
    returns one of the nine components of the inverse Jacobian matrix of an analytical (kind_map >= 1) or discrete (kind_map = 0) mapping x, y, z = F(eta1, eta2, eta3) in three space dimensions. 
    
    the 3x3 inverse is computed directly from DF, using the cross product of the columns of DF:

                            | [ (dF/deta2) x (dF/deta3) ]^T |
    (DF)^(-1) = 1/det_df *  | [ (dF/deta3) x (dF/deta1) ]^T |
                            | [ (dF/deta1) x (dF/deta2) ]^T |
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    eta3 : double
        3rd logical coordinate in [0, 1]
        
    component : int
        the component of the inverse Jacobian matrix
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : array_like
        parameters for the mapping (1 : [lx, ly, lz], 2 : [r1, r2, lz], 3 : [lx, ly, alpha, lz], 4 : [lx, ly, alpha, lz])
        
    tn1 : array_like
        spline knot vector in 1-direction
        
    tn2 : array_like
        spline knot vector in 2-direction
        
    tn3 : array_like
        spline knot vector in 3-direction
        
    pn : array_like
        spline degrees in all directions
        
    nbase_n : array_like
        number of splines in all directions
        
    cx : array_like
        control points of x-component
        
    cy : array_like
        control points of y-component
        
    cz : array_like
        control points of z-component
    """
    
    value = 0.

    df_11 = df(eta1, eta2, eta3, 11, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_12 = df(eta1, eta2, eta3, 12, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_13 = df(eta1, eta2, eta3, 13, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_21 = df(eta1, eta2, eta3, 21, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_22 = df(eta1, eta2, eta3, 22, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_23 = df(eta1, eta2, eta3, 23, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_31 = df(eta1, eta2, eta3, 31, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_32 = df(eta1, eta2, eta3, 32, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_33 = df(eta1, eta2, eta3, 33, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    detdf = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)

    if   component == 11:
        value = df_22*df_33 - df_32*df_23
    elif component == 12:
        value = df_32*df_13 - df_12*df_33
    elif component == 13:
        value = df_12*df_23 - df_22*df_13
    elif component == 21:
        value = df_23*df_31 - df_33*df_21
    elif component == 22:
        value = df_33*df_11 - df_13*df_31
    elif component == 23:
        value = df_13*df_21 - df_23*df_11
    elif component == 31:
        value = df_21*df_32 - df_31*df_22
    elif component == 32:
        value = df_31*df_12 - df_11*df_32
    elif component == 33:
        value = df_11*df_22 - df_21*df_12
        
    value = value/detdf
            
    return value


# =======================================================================
@types('double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def g(eta1, eta2, eta3, component, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    """
    returns one of the nine components of the metric tensor G = DF^T * DF of an analytical (kind_map >= 1) or discrete (kind_map = 0) mapping x, y, z = F(eta1, eta2, eta3) in three space dimensions.
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    eta3 : double
        3rd logical coordinate in [0, 1]
        
    component : int
        the component of the metric tensor
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : array_like
        parameters for the mapping (1 : [lx, ly, lz], 2 : [r1, r2, lz], 3 : [lx, ly, alpha, lz], 4 : [lx, ly, alpha, lz])
        
    tn1 : array_like
        spline knot vector in 1-direction
        
    tn2 : array_like
        spline knot vector in 2-direction
        
    tn3 : array_like
        spline knot vector in 3-direction
        
    pn : array_like
        spline degrees in all directions
        
    nbase_n : array_like
        number of splines in all directions
        
    cx : array_like
        control points of x-component
        
    cy : array_like
        control points of y-component
        
    cz : array_like
        control points of z-component 
    """
    
    value = 0.

    if   component == 11:
        df_11 = df(eta1, eta2, eta3, 11, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_21 = df(eta1, eta2, eta3, 21, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_31 = df(eta1, eta2, eta3, 31, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_11*df_11 + df_21*df_21 + df_31*df_31
        
    elif component == 22:                                              
        df_12 = df(eta1, eta2, eta3, 12, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_22 = df(eta1, eta2, eta3, 22, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_32 = df(eta1, eta2, eta3, 32, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_12*df_12 + df_22*df_22 + df_32*df_32
                 
    elif component == 33:                                              
        df_13 = df(eta1, eta2, eta3, 13, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_23 = df(eta1, eta2, eta3, 23, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_33 = df(eta1, eta2, eta3, 33, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_13*df_13 + df_23*df_23 + df_33*df_33
                 
    elif ((component == 12) or (component == 21)) :
        df_11 = df(eta1, eta2, eta3, 11, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_21 = df(eta1, eta2, eta3, 21, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_31 = df(eta1, eta2, eta3, 31, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_12 = df(eta1, eta2, eta3, 12, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_22 = df(eta1, eta2, eta3, 22, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_32 = df(eta1, eta2, eta3, 32, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_11*df_12 + df_21*df_22 + df_31*df_32
                 
    elif ((component == 13) or (component == 31)):
        df_11 = df(eta1, eta2, eta3, 11, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_21 = df(eta1, eta2, eta3, 21, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_31 = df(eta1, eta2, eta3, 31, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_13 = df(eta1, eta2, eta3, 13, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_23 = df(eta1, eta2, eta3, 23, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_33 = df(eta1, eta2, eta3, 33, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_11*df_13 + df_21*df_23 + df_31*df_33
                 
    elif ((component == 23) or (component == 32)):  
        df_12 = df(eta1, eta2, eta3, 12, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_22 = df(eta1, eta2, eta3, 22, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_32 = df(eta1, eta2, eta3, 32, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_13 = df(eta1, eta2, eta3, 13, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_23 = df(eta1, eta2, eta3, 23, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_33 = df(eta1, eta2, eta3, 33, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_12*df_13 + df_22*df_23 + df_32*df_33
               
    return value


# =======================================================================
@types('double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def g_inv(eta1, eta2, eta3, component, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    """
    returns one of the nine components of the inverse metric tensor G = DF^(-1) * DF^(-T) of an analytical (kind_map >= 1) or discrete (kind_map = 0) mapping x, y, z = F(eta1, eta2, eta3) in three space dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    eta3 : double
        3rd logical coordinate in [0, 1]
        
    component : int
        the component of the inverse metric tensor
        
    kind_map : int
        kind of mapping (0 : discrete, 1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : array_like
        parameters for the mapping (1 : [lx, ly, lz], 2 : [r1, r2, lz], 3 : [lx, ly, alpha, lz], 4 : [lx, ly, alpha, lz])
        
    tn1 : array_like
        spline knot vector in 1-direction
        
    tn2 : array_like
        spline knot vector in 2-direction
        
    tn3 : array_like
        spline knot vector in 3-direction
        
    pn : array_like
        spline degrees in all directions
        
    nbase_n : array_like
        number of splines in all directions
        
    cx : array_like
        control points of x-component
        
    cy : array_like
        control points of y-component
        
    cz : array_like
        control points of z-component 
    """
    
    value = 0.

    if   component == 11:
        dfinv_11 = df_inv(eta1, eta2, eta3, 11, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_12 = df_inv(eta1, eta2, eta3, 12, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_13 = df_inv(eta1, eta2, eta3, 13, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value    = dfinv_11*dfinv_11 + dfinv_12*dfinv_12 + dfinv_13*dfinv_13
                  
    elif component == 22:                                              
        dfinv_21 = df_inv(eta1, eta2, eta3, 21, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_22 = df_inv(eta1, eta2, eta3, 22, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_23 = df_inv(eta1, eta2, eta3, 23, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value    = dfinv_21*dfinv_21 + dfinv_22*dfinv_22 + dfinv_23*dfinv_23
                  
    elif component == 33:                                              
        dfinv_31 = df_inv(eta1, eta2, eta3, 31, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_32 = df_inv(eta1, eta2, eta3, 32, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_33 = df_inv(eta1, eta2, eta3, 33, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value    = dfinv_31*dfinv_31 + dfinv_32*dfinv_32 + dfinv_33*dfinv_33
                  
    elif ((component == 12) or (component == 21)) :
        dfinv_11 = df_inv(eta1, eta2, eta3, 11, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_12 = df_inv(eta1, eta2, eta3, 12, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_13 = df_inv(eta1, eta2, eta3, 13, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_21 = df_inv(eta1, eta2, eta3, 21, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_22 = df_inv(eta1, eta2, eta3, 22, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_23 = df_inv(eta1, eta2, eta3, 23, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value    = dfinv_11*dfinv_21 + dfinv_12*dfinv_22 + dfinv_13*dfinv_23
                  
    elif ((component == 13) or (component == 31)):
        dfinv_11 = df_inv(eta1, eta2, eta3, 11, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_12 = df_inv(eta1, eta2, eta3, 12, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_13 = df_inv(eta1, eta2, eta3, 13, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_31 = df_inv(eta1, eta2, eta3, 31, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_32 = df_inv(eta1, eta2, eta3, 32, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_33 = df_inv(eta1, eta2, eta3, 33, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value    = dfinv_11*dfinv_31 + dfinv_12*dfinv_32 + dfinv_13*dfinv_33
                  
    elif ((component == 23) or (component == 32)):  
        dfinv_21 = df_inv(eta1, eta2, eta3, 21, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_22 = df_inv(eta1, eta2, eta3, 22, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_23 = df_inv(eta1, eta2, eta3, 23, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_31 = df_inv(eta1, eta2, eta3, 31, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_32 = df_inv(eta1, eta2, eta3, 32, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_33 = df_inv(eta1, eta2, eta3, 33, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value    = dfinv_21*dfinv_31 + dfinv_22*dfinv_32 + dfinv_23*dfinv_33
    
    return value


# ==========================================================================================
@types('double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def all_mappings(eta1, eta2, eta3, kind_fun, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    value = 0.
    
    # mapping f
    if   kind_fun == 1:
        value = f(eta1, eta2, eta3, 1, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 2:
        value = f(eta1, eta2, eta3, 2, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 3:
        value = f(eta1, eta2, eta3, 3, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    # Jacobian matrix df
    elif kind_fun == 11:
        value = df(eta1, eta2, eta3, 11, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 12:
        value = df(eta1, eta2, eta3, 12, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 13:
        value = df(eta1, eta2, eta3, 13, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 14:
        value = df(eta1, eta2, eta3, 21, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 15:
        value = df(eta1, eta2, eta3, 22, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 16:
        value = df(eta1, eta2, eta3, 23, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 17:
        value = df(eta1, eta2, eta3, 31, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 18:
        value = df(eta1, eta2, eta3, 32, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 19:
        value = df(eta1, eta2, eta3, 33, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # Jacobian determinant det_df
    elif kind_fun == 4:
        value = det_df(eta1, eta2, eta3, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # inverse Jacobian matrix df_inv
    elif kind_fun == 21:
        value = df_inv(eta1, eta2, eta3, 11, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 22:
        value = df_inv(eta1, eta2, eta3, 12, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 23:
        value = df_inv(eta1, eta2, eta3, 13, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 24:
        value = df_inv(eta1, eta2, eta3, 21, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 25:
        value = df_inv(eta1, eta2, eta3, 22, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 26:
        value = df_inv(eta1, eta2, eta3, 23, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 27:
        value = df_inv(eta1, eta2, eta3, 31, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 28:
        value = df_inv(eta1, eta2, eta3, 32, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 29:
        value = df_inv(eta1, eta2, eta3, 33, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # metric tensor g
    elif kind_fun == 31:
        value = g(eta1, eta2, eta3, 11, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 32:
        value = g(eta1, eta2, eta3, 12, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 33:
        value = g(eta1, eta2, eta3, 13, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 34:
        value = g(eta1, eta2, eta3, 21, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 35:
        value = g(eta1, eta2, eta3, 22, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 36:
        value = g(eta1, eta2, eta3, 23, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 37:
        value = g(eta1, eta2, eta3, 31, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 38:
        value = g(eta1, eta2, eta3, 32, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 39:
        value = g(eta1, eta2, eta3, 33, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # metric tensor g_inv
    elif kind_fun == 41:
        value = g_inv(eta1, eta2, eta3, 11, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 42:
        value = g_inv(eta1, eta2, eta3, 12, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 43:
        value = g_inv(eta1, eta2, eta3, 13, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 44:
        value = g_inv(eta1, eta2, eta3, 21, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 45:
        value = g_inv(eta1, eta2, eta3, 22, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 46:
        value = g_inv(eta1, eta2, eta3, 23, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 47:
        value = g_inv(eta1, eta2, eta3, 31, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 48:
        value = g_inv(eta1, eta2, eta3, 32, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 49:
        value = g_inv(eta1, eta2, eta3, 33, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    return value



# ==========================================================================================
@types('double[:]','double[:]','double[:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')       
def kernel_evaluate_tensor_product(eta1, eta2, eta3, kind_fun, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, mat_f):
    
    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            for i3 in range(len(eta3)):
                mat_f[i1, i2, i3] = all_mappings(eta1[i1], eta2[i2], eta3[i3], kind_fun, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
                
                
# ==========================================================================================
@types('double[:,:,:]','double[:,:,:]','double[:,:,:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')       
def kernel_evaluate_general(eta1, eta2, eta3, kind_fun, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, mat_f):
    
    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]
    
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                mat_f[i1, i2, i3] = all_mappings(eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_fun, kind_map, params, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)