# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""Module of accelerated functions for point-wise evaluation of a 3d analytical mapping
or discrete spline mapping, and its corresponding metric coefficients:

- f      : mapping x_i = f_i(eta1, eta2, eta3)
- df     : Jacobian matrix df_i/deta_j
- det_df : Jacobian determinant det(df)
- df_inv : inverse Jacobian matrix (df_i/deta_j)^(-1)
- g      : metric tensor df^T * df 
- g_inv  : inverse metric tensor df^(-1) * df^(-T)

The following geometries are implemented:

- kind_map = 0  : 3d discrete spline mapping. All information is stored in control points cx, cy, cz. params_map = [].
- kind_map = 1  : discrete cylinder. 2d discrete spline mapping in xy-plane and analytical in z. params_map = [lz].
- kind_map = 2  : discrete torus. 2d discrete spline mapping in xy-plane and analytical in phi. params_map = [].

- kind_map = 10 : cuboid. params_map = [lx, ly, lz].
- kind_map = 11 : hollow cylinder. params_map = [a1, a2, lz].
- kind_map = 12 : colella. params_map = [lx, ly, alpha, lz].
- kind_map = 13 : othogonal. params_map = [ly, ly, alpha, lz].
- kind_map = 14 : hollow torus. params_map = [a1, a2, r0].

"""

from numpy import shape
from numpy import sin, cos, pi, zeros, array, sqrt

from pyccel.decorators import types

import hylife.utilitis_FEEC.basics.spline_evaluation_2d as eva_2d
import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva_3d


# =======================================================================
@types('double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def f(eta1, eta2, eta3, component, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    """Point-wise evaluation of Cartesian coordinate x_i = f_i(eta1, eta2, eta3), i=1,2,3. 
    
    Parameters:
    -----------
        eta1, eta2, eta3:       double              logical coordinates in [0, 1]
        component:              int                 Cartesian coordinate (1: x, 2: y, 3: z)
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             double[:]           parameters for the mapping
        tn1, tn2, tn3:          double[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             double[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            Cartesian coordinate x_i = f_i(eta1, eta2, eta3)
    """
   
   
    value = 0.
    
    # =========== 3d discrete =================
    if kind_map == 0:
        
        if   component == 1:
            value = eva_3d.evaluate_n_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cx, eta1, eta2, eta3)

        elif component == 2:
            value = eva_3d.evaluate_n_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cy, eta1, eta2, eta3)

        elif component == 3:
            value = eva_3d.evaluate_n_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cz, eta1, eta2, eta3)
            
    # =========== discrete cylinder =================
    elif kind_map == 1:
        
        lz = params_map[0]
        
        if   component == 1:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2)

        elif component == 2:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)
            
        elif component == 3:
            value = lz * eta3

    # =========== discrete torus ====================
    elif kind_map == 2:
        
        if   component == 1:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3)

        elif component == 2:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)
            
        elif component == 3:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3)
  
    # ============== cuboid ==================
    elif kind_map == 10:
         
        lx = params_map[0] 
        ly = params_map[1] 
        lz = params_map[2]
        
        if   component == 1:
            value = lx * eta1
        elif component == 2:
            value = ly * eta2
        elif component == 3:
            value = lz * eta3
            
    # ========= hollow cylinder ==============
    elif kind_map == 11:
        
        a1 = params_map[0]
        a2 = params_map[1]
        lz = params_map[2]
        
        da = a2 - a1
        
        if   component == 1:
            value = (a1 + eta1 * da) * cos(2*pi*eta2)
        elif component == 2:
            value = (a1 + eta1 * da) * sin(2*pi*eta2)
        elif component == 3:
            value = lz * eta3
            
    # ============ colella ===================
    elif kind_map == 12:
        
        lx    = params_map[0]
        ly    = params_map[1]
        alpha = params_map[2]
        lz    = params_map[3]
        
        if   component == 1:
            value = lx * (eta1 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        elif component == 2:
            value = ly * (eta2 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        elif component == 3:
            value = lz * eta3
    
    # =========== orthogonal ================
    elif kind_map == 13:
        
        lx    = params_map[0]
        ly    = params_map[1]
        alpha = params_map[2]
        lz    = params_map[3]
        
        if   component == 1:
            value = lx * (eta1 + alpha * sin(2*pi*eta1))
        elif component == 2:
            value = ly * (eta2 + alpha * sin(2*pi*eta2))
        elif component == 3:
            value = lz * eta3
            
            
    # ========= hollow torus ================
    elif kind_map == 14:
        
        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]
        
        da = a2 - a1
        
        if   component == 1:
            value = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * cos(2*pi*eta3)
        elif component == 2:
            value =  (a1 + eta1 * da) * sin(2*pi*eta2)
        elif component == 3:
            value = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * sin(2*pi*eta3)
            
    return value


# =======================================================================
@types('double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def df(eta1, eta2, eta3, component, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    """Point-wise evaluation of ij-th component of the Jacobian matrix df_ij = df_i/deta_j (i,j=1,2,3). 
    
    Parameters:
    -----------
        eta1, eta2, eta3:       double              logical coordinates in [0, 1]
        component:              int                 11 : (df1/deta1), 12 : (df1/deta2), 13 : (df1/deta3)
                                                    21 : (df2/deta1), 22 : (df2/deta2), 23 : (df2/deta3)
                                                    31 : (df3/deta1), 32 : (df3/deta2), 33 : (df3/deta3)
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             double[:]           parameters for the mapping
        tn1, tn2, tn3:          double[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             double[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            point value df_ij(eta1, eta2, eta3)
    """
    
    value = 0.
    
    # ============ 3d discrete ================
    if kind_map == 0:
        
        if   component == 11:
            value = eva_3d.evaluate_diffn_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cx, eta1, eta2, eta3)
        elif component == 12:
            value = eva_3d.evaluate_n_diffn_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cx, eta1, eta2, eta3)
        elif component == 13:
            value = eva_3d.evaluate_n_n_diffn(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cx, eta1, eta2, eta3)
        elif component == 21:
            value = eva_3d.evaluate_diffn_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cy, eta1, eta2, eta3)
        elif component == 22:
            value = eva_3d.evaluate_n_diffn_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cy, eta1, eta2, eta3)
        elif component == 23:
            value = eva_3d.evaluate_n_n_diffn(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cy, eta1, eta2, eta3)
        elif component == 31:
            value = eva_3d.evaluate_diffn_n_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cz, eta1, eta2, eta3)
        elif component == 32:
            value = eva_3d.evaluate_n_diffn_n(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cz, eta1, eta2, eta3)
        elif component == 33:
            value = eva_3d.evaluate_n_n_diffn(tn1, tn2, tn3, pn[0], pn[1], pn[2], nbase_n[0], nbase_n[1], nbase_n[2], cz, eta1, eta2, eta3)
               
    # ============ discrete cylinder ================
    elif kind_map == 1:
        
        lz = params_map[0]
        
        if   component == 11:
            value = eva_2d.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2)
        elif component == 12:
            value = eva_2d.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2)
        elif component == 13:
            value = 0.
        elif component == 21:
            value = eva_2d.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)
        elif component == 22:
            value = eva_2d.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = lz
    
    # ============ discrete torus ===================
    elif kind_map == 2:
        
        if   component == 11:
            value = eva_2d.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3)
        elif component == 12:
            value = eva_2d.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3)
        elif component == 13:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3) * (-2*pi)
        elif component == 21:
            value = eva_2d.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)
        elif component == 22:
            value = eva_2d.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cy[:, :, 0], eta1, eta2)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = eva_2d.evaluate_diffn_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3)
        elif component == 32:
            value = eva_2d.evaluate_n_diffn(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * sin(2*pi*eta3)
        elif component == 33:
            value = eva_2d.evaluate_n_n(tn1, tn2, pn[0], pn[1], nbase_n[0], nbase_n[1], cx[:, :, 0], eta1, eta2) * cos(2*pi*eta3) * 2*pi
    
    # ============== cuboid ===================
    elif kind_map == 10:
         
        lx = params_map[0] 
        ly = params_map[1] 
        lz = params_map[2]
        
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
            
    # ======== hollow cylinder =================
    elif kind_map == 11:
        
        a1 = params_map[0]
        a2 = params_map[1]
        lz = params_map[2]
        
        da = a2 - a1
        
        if   component == 11:
            value = da * cos(2*pi*eta2)
        elif component == 12:
            value = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2)
        elif component == 13:
            value = 0.
        elif component == 21:
            value = da * sin(2*pi*eta2)
        elif component == 22:
            value = 2*pi * (a1 + eta1 * da) * cos(2*pi*eta2)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = lz
            
    # ============ colella =================
    elif kind_map == 12:
        
        lx    = params_map[0]
        ly    = params_map[1]
        alpha = params_map[2]
        lz    = params_map[3]
        
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
    elif kind_map == 13:
        
        lx    = params_map[0]
        ly    = params_map[1]
        alpha = params_map[2]
        lz    = params_map[3]
        
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
            
            
    # ========= hollow torus ==================
    elif kind_map == 14:
        
        a1 = params_map[0]
        a2 = params_map[1]
        r0 = params_map[2]
        
        da = a2 - a1
        
        if   component == 11:
            value = da * cos(2*pi*eta2) * cos(2*pi*eta3)
        elif component == 12:
            value = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2) * cos(2*pi*eta3)
        elif component == 13:
            value = -2*pi * ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * sin(2*pi*eta3)
        elif component == 21:
            value = da * sin(2*pi*eta2)
        elif component == 22:
            value = (a1 + eta1 * da) * cos(2*pi*eta2) * 2*pi
        elif component == 23:
            value = 0.
        elif component == 31:
            value = da * cos(2*pi*eta2) * sin(2*pi*eta3)
        elif component == 32:
            value = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2) * sin(2*pi*eta3)
        elif component == 33:
            value = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * cos(2*pi*eta3) * 2*pi
            
    return value



# =======================================================================
@types('double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    """Point-wise evaluation of Jacobian determinant det(df) = df/deta1.(df/deta2 x df/deta3). 
    
    Parameters:
    -----------
        eta1, eta2, eta3:       double              logical coordinates in [0, 1]
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             double[:]           parameters for the mapping
        tn1, tn2, tn3:          double[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             double[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            point value of Jacobian determinant det(df)(eta1, eta2, eta3)
    """
    
    value = 0.
    
    df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

    value = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
            
    return value


# =======================================================================
@types('double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def df_inv(eta1, eta2, eta3, component, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    """Point-wise evaluation of ij-th component of the inverse Jacobian matrix df^(-1)_ij (i,j=1,2,3). 
    
    The 3 x 3 inverse is computed directly from df, using the cross product of the columns of df:

                            | [ (df/deta2) x (df/deta3) ]^T |
    (df)^(-1) = 1/det_df *  | [ (df/deta3) x (df/deta1) ]^T |
                            | [ (df/deta1) x (df/deta2) ]^T |
    
    Parameters:
    -----------
        eta1, eta2, eta3:       double              logical coordinates in [0, 1]
        component:              int                 index ij (11, 12, 13, 21, 22, 23, 31, 32, 33)
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             double[:]           parameters for the mapping
        tn1, tn2, tn3:          double[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             double[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            point value df^(-1)_ij(eta1, eta2, eta3)
    """
    
    value = 0.

    df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

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
def g(eta1, eta2, eta3, component, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    """Point-wise evaluation of ij-th component of metric tensor g_ij = sum_k (df^T)_ik (df)_kj (i,j,k=1,2,3). 
    
    Parameters:
    -----------
        eta1, eta2, eta3:       double              logical coordinates in [0, 1]
        component:              int                 index ij (11, 12, 13, 21, 22, 23, 31, 32, 33)
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             double[:]           parameters for the mapping
        tn1, tn2, tn3:          double[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             double[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            point value g_ij(eta1, eta2, eta3)
    """
    
    value = 0.

    if   component == 11:
        df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_11*df_11 + df_21*df_21 + df_31*df_31
        
    elif component == 22:                                              
        df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_12*df_12 + df_22*df_22 + df_32*df_32
                 
    elif component == 33:                                              
        df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_13*df_13 + df_23*df_23 + df_33*df_33
                 
    elif ((component == 12) or (component == 21)) :
        df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_11*df_12 + df_21*df_22 + df_31*df_32
                 
    elif ((component == 13) or (component == 31)):
        df_11 = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_21 = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_31 = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_11*df_13 + df_21*df_23 + df_31*df_33
                 
    elif ((component == 23) or (component == 32)):  
        df_12 = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_22 = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_32 = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_13 = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_23 = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        df_33 = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value = df_12*df_13 + df_22*df_23 + df_32*df_33
               
    return value


# =======================================================================
@types('double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def g_inv(eta1, eta2, eta3, component, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    """Point-wise evaluation of ij-th component of inverse metric tensor g^(-1)_ij = sum_k (df^-1)_ik (df^-T)_kj (i,j,k=1,2,3). 
    
    Parameters:
    -----------
        eta1, eta2, eta3:       double              logical coordinates in [0, 1]
        component:              int                 index ij (11, 12, 13, 21, 22, 23, 31, 32, 33)
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             double[:]           parameters for the mapping
        tn1, tn2, tn3:          double[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             double[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            point value g^(-1)_ij(eta1, eta2, eta3) 
    """
    
    value = 0.

    if   component == 11:
        dfinv_11 = df_inv(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_12 = df_inv(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_13 = df_inv(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value    = dfinv_11*dfinv_11 + dfinv_12*dfinv_12 + dfinv_13*dfinv_13
                  
    elif component == 22:                                              
        dfinv_21 = df_inv(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_22 = df_inv(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_23 = df_inv(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value    = dfinv_21*dfinv_21 + dfinv_22*dfinv_22 + dfinv_23*dfinv_23
                  
    elif component == 33:                                              
        dfinv_31 = df_inv(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_32 = df_inv(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_33 = df_inv(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value    = dfinv_31*dfinv_31 + dfinv_32*dfinv_32 + dfinv_33*dfinv_33
                  
    elif ((component == 12) or (component == 21)) :
        dfinv_11 = df_inv(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_12 = df_inv(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_13 = df_inv(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_21 = df_inv(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_22 = df_inv(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_23 = df_inv(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value    = dfinv_11*dfinv_21 + dfinv_12*dfinv_22 + dfinv_13*dfinv_23
                  
    elif ((component == 13) or (component == 31)):
        dfinv_11 = df_inv(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_12 = df_inv(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_13 = df_inv(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_31 = df_inv(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_32 = df_inv(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_33 = df_inv(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value    = dfinv_11*dfinv_31 + dfinv_12*dfinv_32 + dfinv_13*dfinv_33
                  
    elif ((component == 23) or (component == 32)):  
        dfinv_21 = df_inv(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_22 = df_inv(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_23 = df_inv(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_31 = df_inv(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_32 = df_inv(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_33 = df_inv(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        value    = dfinv_21*dfinv_31 + dfinv_22*dfinv_32 + dfinv_23*dfinv_33
    
    return value


# ==========================================================================================
@types('double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def mappings_all(eta1, eta2, eta3, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    '''Point-wise evaluation of
        - f      : mapping x_i = f_i(eta1, eta2, eta3)
        - df     : Jacobian matrix df_i/deta_j
        - det_df : Jacobian determinant det(df)
        - df_inv : inverse Jacobian matrix (df_i/deta_j)^(-1)
        - g      : metric tensor df^T * df 
        - g_inv  : inverse metric tensor df^(-1) * df^(-T)  .
    
    Parameters:
    -----------
        eta1, eta2, eta3:       double              logical coordinates in [0, 1]
        kind_fun:               int                 function to evaluate (see keys_map in 'domain_3d.py')
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             double[:]           parameters for the mapping
        tn1, tn2, tn3:          double[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             double[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        value:  float
            point value of mapping/metric coefficient at (eta1, eta2, eta3)
    '''
    
    value = 0.
    
    # mapping f
    if   kind_fun == 1:
        value = f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 2:
        value = f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 3:
        value = f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    # Jacobian matrix df
    elif kind_fun == 11:
        value = df(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 12:
        value = df(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 13:
        value = df(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 14:
        value = df(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 15:
        value = df(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 16:
        value = df(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 17:
        value = df(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 18:
        value = df(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 19:
        value = df(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # Jacobian determinant det_df
    elif kind_fun == 4:
        value = det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # inverse Jacobian matrix df_inv
    elif kind_fun == 21:
        value = df_inv(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 22:
        value = df_inv(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 23:
        value = df_inv(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 24:
        value = df_inv(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 25:
        value = df_inv(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 26:
        value = df_inv(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 27:
        value = df_inv(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 28:
        value = df_inv(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 29:
        value = df_inv(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # metric tensor g
    elif kind_fun == 31:
        value = g(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 32:
        value = g(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 33:
        value = g(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 34:
        value = g(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 35:
        value = g(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 36:
        value = g(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 37:
        value = g(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 38:
        value = g(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 39:
        value = g(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # metric tensor g_inv
    elif kind_fun == 41:
        value = g_inv(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 42:
        value = g_inv(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 43:
        value = g_inv(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 44:
        value = g_inv(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 45:
        value = g_inv(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 46:
        value = g_inv(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 47:
        value = g_inv(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 48:
        value = g_inv(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 49:
        value = g_inv(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    return value


# ==========================================================================================
@types('double[:,:,:]','double[:,:,:]','double[:,:,:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')       
def kernel_evaluate(eta1, eta2, eta3, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, mat_f):
    '''Matrix-wise evaluation of
        - f      : mapping x_i = f_i(eta1, eta2, eta3)
        - df     : Jacobian matrix df_i/deta_j
        - det_df : Jacobian determinant det(df)
        - df_inv : inverse Jacobian matrix (df_i/deta_j)^(-1)
        - g      : metric tensor df^T * df 
        - g_inv  : inverse metric tensor df^(-1) * df^(-T)  .

    Parameters:
    -----------
        eta1, eta2, eta3:       double[:, :, :]     matrices of logical coordinates in [0, 1]
        kind_fun:               int                 function to evaluate (see keys_map in 'domain_3d.py')
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             double[:]           parameters for the mapping
        tn1, tn2, tn3:          double[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             double[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        mat_f:  ndarray
            matrix-valued mapping/metric coefficient evaluated at (eta1, eta2, eta3)
    '''

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                mat_f[i1, i2, i3] = mappings_all(eta1[i1, i2, i3], eta2[i1, i2, i3], eta3[i1, i2, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)


# ==========================================================================================
@types('double[:,:,:]','double[:,:,:]','double[:,:,:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')       
def kernel_evaluate_sparse(eta1, eta2, eta3, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz, mat_f):
    '''Same as `kernel_evluate`, but for sparse meshgrid.
    Matrix-wise evaluation of
        - f      : mapping x_i = f_i(eta1, eta2, eta3)
        - df     : Jacobian matrix df_i/deta_j
        - det_df : Jacobian determinant det(df)
        - df_inv : inverse Jacobian matrix (df_i/deta_j)^(-1)
        - g      : metric tensor df^T * df 
        - g_inv  : inverse metric tensor df^(-1) * df^(-T)  .

    Parameters:
    -----------
        eta1, eta2, eta3:       double[:, :, :]     matrices of logical coordinates in [0, 1] produced from sparse meshgrid
        kind_fun:               int                 function to evaluate (see keys_map in 'domain_3d.py')
        kind_map:               int                 kind of mapping (see module docstring)
        params_map:             double[:]           parameters for the mapping
        tn1, tn2, tn3:          double[:]           knot vectors for mapping
        pn:                     int[:]              spline degrees for mapping
        nbase_n:                int[:]              dimensions of univariate spline spaces for mapping 
        cx, cy, cz:             double[:, :, :]     control points of (f_1, f_2, f_3)

    Returns:
    --------
        mat_f:  ndarray
            matrix-valued mapping/metric coefficient evaluated at (eta1, eta2, eta3)
    '''

    n1 = shape(eta1)[0]
    n2 = shape(eta2)[1]
    n3 = shape(eta3)[2]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                mat_f[i1, i2, i3] = mappings_all(eta1[i1, 0, 0], eta2[0, i2, 0], eta3[0, 0, i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
