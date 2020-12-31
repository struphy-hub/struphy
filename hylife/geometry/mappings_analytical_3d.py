# coding: utf-8
#
# Copnyright 2020 Florian Holderied

"""
Basic functions for point-wise evaluation of a 3d analytical mapping and its corresponding geometric quantities:
"""

from pyccel.decorators import types
from numpy             import sin, cos, pi, zeros, array, sqrt


# =======================================================================
@types('double','double','double','int','double[:]','int')
def f(eta1, eta2, eta3, kind, params, component):
    """
    returns the components of an analytical mapping x, y, z = F(eta1, eta2, eta3) in three dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    eta3 : double
        3rd logical coordinate in [0, 1]
        
    kind : int
        kind of mapping (1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : list of doubles
        parameters for the mapping (1 : [lx, ly, lz], 2 : [r1, r2, lz], 3 : [lx, ly, alpha, lz], 4 : [lx, ly, alpha, lz])
        
    component : int
        physical coordinate (1 : x, 2 : y, 3 : z)
    """
   
    value = 0.

    # ============== slab ==================
    if kind == 1:
         
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
    elif kind == 2:
        
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
    elif kind == 3:
        
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
    elif kind == 4:
        
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
@types('double','double','double','int','double[:]','int')
def df(eta1, eta2, eta3, kind, params, component):
    """
    returns the components of the Jacobian matrix DF_ij = dF_i/deta_j of an analytical mapping x, y, z = F(eta1, eta2, eta3) in three dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    eta3 : double
        3rd logical coordinate in [0, 1]
        
    kind : int
        kind of mapping (1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : list of doubles
        parameters for the mapping (1 : [lx, ly, lz], 2 : [r1, r2, lz], 3 : [lx, ly, alpha, lz], 4 : [lx, ly, alpha, lz])
                 
    component : int 
        11 : (dx/deta1), 12 : (dx/deta2), 13 : (dx/deta3)
        21 : (dy/deta1), 22 : (dy/deta2), 23 : (dy/deta3)
        31 : (dz/deta1), 32 : (dz/deta2), 33 : (dz/deta3)
    """
    
    value = 0.
               
    # ============== slab ==================
    if kind == 1:
         
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
            calue = 0.
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
    elif kind == 2:
        
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
    elif kind == 3:
        
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
    elif kind == 4:
        
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
@types('double','double','double','int','double[:]')
def det_df(eta1, eta2, eta3, kind, params):
    """
    returns the Jacobian determinant det(DF) of an analytical mapping x, y, z = F(eta1, eta2, eta3) in three dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    eta3 : double
        3rd logical coordinate in [0, 1]
        
    kind : int
        kind of mapping (1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : list of doubles
        parameters for the mapping (1 : [lx, ly, lz], 2 : [r1, r2, lz], 3 : [lx, ly, alpha, lz], 4 : [lx, ly, alpha, lz])
                 
    Returns
    -------
    value : double
        the Jacobian determinant dF/deta1 . ( dF/deta2 x dF/deta3)
    """
    
    value = 0.
    
    df_11 = df(eta1, eta2, eta3, kind, params, 11)
    df_12 = df(eta1, eta2, eta3, kind, params, 12)
    df_13 = df(eta1, eta2, eta3, kind, params, 13)
    
    df_21 = df(eta1, eta2, eta3, kind, params, 21)
    df_22 = df(eta1, eta2, eta3, kind, params, 22)
    df_23 = df(eta1, eta2, eta3, kind, params, 23)
    
    df_31 = df(eta1, eta2, eta3, kind, params, 31)
    df_32 = df(eta1, eta2, eta3, kind, params, 32)
    df_33 = df(eta1, eta2, eta3, kind, params, 33)

    value = df_11*(df_22*df_33 - df_32*df_23) + df_21*(df_32*df_13 - df_12*df_33) + df_31*(df_12*df_23 - df_22*df_13)
            
    return value




# =======================================================================
@types('double','double','double','int','double[:]','int')
def df_inv(eta1, eta2, eta3, kind, params, component):
    """
    returns the components of the inverse of the Jacobian matrix DF_ij = dF_i/deta_j of an analytical mapping x, y, z = F(eta1, eta2, eta3) in three dimensions. 
    
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
        
    kind : int
        kind of mapping (1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : list of doubles
        parameters for the mapping (1 : [lx, ly, lz], 2 : [r1, r2, lz], 3 : [lx, ly, alpha, lz], 4 : [lx, ly, alpha, lz])
                 
    component : int 
    """
    
    value = 0.

    df_11 = df(eta1, eta2, eta3, kind, params, 11)
    df_12 = df(eta1, eta2, eta3, kind, params, 12)
    df_13 = df(eta1, eta2, eta3, kind, params, 13)
    
    df_21 = df(eta1, eta2, eta3, kind, params, 21)
    df_22 = df(eta1, eta2, eta3, kind, params, 22)
    df_23 = df(eta1, eta2, eta3, kind, params, 23)
    
    df_31 = df(eta1, eta2, eta3, kind, params, 31)
    df_32 = df(eta1, eta2, eta3, kind, params, 32)
    df_33 = df(eta1, eta2, eta3, kind, params, 33)

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
@types('double','double','double','int','double[:]','int')
def g(eta1, eta2, eta3, kind, params, component):
    """
    returns the components of the metric tensor G = DF^T * DF of an analytical mapping x, y, z = F(eta1, eta2, eta3) in three dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    eta3 : double
        3rd logical coordinate in [0, 1]
        
    kind : int
        kind of mapping (1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : list of doubles
        parameters for the mapping (1 : [lx, ly, lz], 2 : [r1, r2, lz], 3 : [lx, ly, alpha, lz], 4 : [lx, ly, alpha, lz])
                 
    component : int 
    """
    
    value = 0.

    if   component == 11:
        df_11 = df(eta1, eta2, eta3, kind, params, 11)
        df_21 = df(eta1, eta2, eta3, kind, params, 21)
        df_31 = df(eta1, eta2, eta3, kind, params, 31)
        value = df_11*df_11 + df_21*df_21 + df_31*df_31
        
    elif component == 22:                                              
        df_12 = df(eta1, eta2, eta3, kind, params, 12)
        df_22 = df(eta1, eta2, eta3, kind, params, 22)
        df_32 = df(eta1, eta2, eta3, kind, params, 32)
        value = df_12*df_12 + df_22*df_22 + df_32*df_32
                 
    elif component == 33:                                              
        df_13 = df(eta1, eta2, eta3, kind, params, 13)
        df_23 = df(eta1, eta2, eta3, kind, params, 23)
        df_33 = df(eta1, eta2, eta3, kind, params, 33)
        value = df_13*df_13 + df_23*df_23 + df_33*df_33
                 
    elif ((component == 12) or (component == 21)) :
        df_11 = df(eta1, eta2, eta3, kind, params, 11)
        df_21 = df(eta1, eta2, eta3, kind, params, 21)
        df_31 = df(eta1, eta2, eta3, kind, params, 31)
        df_12 = df(eta1, eta2, eta3, kind, params, 12)
        df_22 = df(eta1, eta2, eta3, kind, params, 22)
        df_32 = df(eta1, eta2, eta3, kind, params, 32)
        value = df_11*df_12 + df_21*df_22 + df_31*df_32
                 
    elif ((component == 13) or (component == 31)):
        df_11 = df(eta1, eta2, eta3, kind, params, 11)
        df_21 = df(eta1, eta2, eta3, kind, params, 21)
        df_31 = df(eta1, eta2, eta3, kind, params, 31)
        df_13 = df(eta1, eta2, eta3, kind, params, 13)
        df_23 = df(eta1, eta2, eta3, kind, params, 23)
        df_33 = df(eta1, eta2, eta3, kind, params, 33)
        value = df_11*df_13 + df_21*df_23 + df_31*df_33
                 
    elif ((component == 23) or (component == 32)):  
        df_12 = df(eta1, eta2, eta3, kind, params, 12)
        df_22 = df(eta1, eta2, eta3, kind, params, 22)
        df_32 = df(eta1, eta2, eta3, kind, params, 32)
        df_13 = df(eta1, eta2, eta3, kind, params, 13)
        df_23 = df(eta1, eta2, eta3, kind, params, 23)
        df_33 = df(eta1, eta2, eta3, kind, params, 33)
        value = df_12*df_13 + df_22*df_23 + df_32*df_33
               
    return value


# =======================================================================
@types('double','double','double','int','double[:]','int')
def g_inv(eta1, eta2, eta3, kind, params, component):
    """
    returns the components of the iverse metric tensor G = DF^(-1) * DF^(-T) of an analytical mapping x, y, z = F(eta1, eta2, eta3) in three dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    eta3 : double
        3rd logical coordinate in [0, 1]
        
    kind : int
        kind of mapping (1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : list of doubles
        parameters for the mapping (1 : [lx, ly, lz], 2 : [r1, r2, lz], 3 : [lx, ly, alpha, lz], 4 : [lx, ly, alpha, lz])
                 
    component : int 
    """
    
    value = 0.

    if   component == 11:
        dfinv_11 = df_inv(eta1, eta2, eta3, kind, params, 11)
        dfinv_12 = df_inv(eta1, eta2, eta3, kind, params, 12)
        dfinv_13 = df_inv(eta1, eta2, eta3, kind, params, 13)
        value    = dfinv_11*dfinv_11 + dfinv_12*dfinv_12 + dfinv_13*dfinv_13
                  
    elif component == 22:                                              
        dfinv_21 = df_inv(eta1, eta2, eta3, kind, params, 21)
        dfinv_22 = df_inv(eta1, eta2, eta3, kind, params, 22)
        dfinv_23 = df_inv(eta1, eta2, eta3, kind, params, 23)
        value    = dfinv_21*dfinv_21 + dfinv_22*dfinv_22 + dfinv_23*dfinv_23
                  
    elif component == 33:                                              
        dfinv_31 = df_inv(eta1, eta2, eta3, kind, params, 31)
        dfinv_32 = df_inv(eta1, eta2, eta3, kind, params, 32)
        dfinv_33 = df_inv(eta1, eta2, eta3, kind, params, 33)
        value    = dfinv_31*dfinv_31 + dfinv_32*dfinv_32 + dfinv_33*dfinv_33
                  
    elif ((component == 12) or (component == 21)) :
        dfinv_11 = df_inv(eta1, eta2, eta3, kind, params, 11)
        dfinv_12 = df_inv(eta1, eta2, eta3, kind, params, 12)
        dfinv_13 = df_inv(eta1, eta2, eta3, kind, params, 13)
        dfinv_21 = df_inv(eta1, eta2, eta3, kind, params, 21)
        dfinv_22 = df_inv(eta1, eta2, eta3, kind, params, 22)
        dfinv_23 = df_inv(eta1, eta2, eta3, kind, params, 23)
        value    = dfinv_11*dfinv_21 + dfinv_12*dfinv_22 + dfinv_13*dfinv_23
                  
    elif ((component == 13) or (component == 31)):
        dfinv_11 = df_inv(eta1, eta2, eta3, kind, params, 11)
        dfinv_12 = df_inv(eta1, eta2, eta3, kind, params, 12)
        dfinv_13 = df_inv(eta1, eta2, eta3, kind, params, 13)
        dfinv_31 = df_inv(eta1, eta2, eta3, kind, params, 31)
        dfinv_32 = df_inv(eta1, eta2, eta3, kind, params, 32)
        dfinv_33 = df_inv(eta1, eta2, eta3, kind, params, 33)
        value    = dfinv_11*dfinv_31 + dfinv_12*dfinv_32 + dfinv_13*dfinv_33
                  
    elif ((component == 23) or (component == 32)):  
        dfinv_21 = df_inv(eta1, eta2, eta3, kind, params, 21)
        dfinv_22 = df_inv(eta1, eta2, eta3, kind, params, 22)
        dfinv_23 = df_inv(eta1, eta2, eta3, kind, params, 23)
        dfinv_31 = df_inv(eta1, eta2, eta3, kind, params, 31)
        dfinv_32 = df_inv(eta1, eta2, eta3, kind, params, 32)
        dfinv_33 = df_inv(eta1, eta2, eta3, kind, params, 33)
        value    = dfinv_21*dfinv_31 + dfinv_22*dfinv_32 + dfinv_23*dfinv_33
    
    return value



# ==========================================================================================
@types('double','double','double','int','int','double[:]')        
def fun(eta1, eta2, eta3, kind_fun, kind_map, params):
    
    value = 0.
    
    # mapping f
    if   kind_fun == 1:
        value = f(eta1, eta2, eta3, kind_map, params, 1)
    elif kind_fun == 2:
        value = f(eta1, eta2, eta3, kind_map, params, 2)
    elif kind_fun == 3:
        value = f(eta1, eta2, eta3, kind_map, params, 3)
    
    # Jacobian matrix df
    elif kind_fun == 11:
        value = df(eta1, eta2, eta3, kind_map, params, 11)
    elif kind_fun == 12:
        value = df(eta1, eta2, eta3, kind_map, params, 12)
    elif kind_fun == 13:
        value = df(eta1, eta2, eta3, kind_map, params, 13)
    elif kind_fun == 14:
        value = df(eta1, eta2, eta3, kind_map, params, 21)
    elif kind_fun == 15:
        value = df(eta1, eta2, eta3, kind_map, params, 22)
    elif kind_fun == 16:
        value = df(eta1, eta2, eta3, kind_map, params, 23)
    elif kind_fun == 17:
        value = df(eta1, eta2, eta3, kind_map, params, 31)
    elif kind_fun == 18:
        value = df(eta1, eta2, eta3, kind_map, params, 32)
    elif kind_fun == 19:
        value = df(eta1, eta2, eta3, kind_map, params, 33)
        
    # Jacobian determinant det_df
    elif kind_fun == 4:
        value = det_df(eta1, eta2, eta3, kind_map, params)
        
    # inverse Jacobian matrix df_inv
    elif kind_fun == 21:
        value = df_inv(eta1, eta2, eta3, kind_map, params, 11)
    elif kind_fun == 22:
        value = df_inv(eta1, eta2, eta3, kind_map, params, 12)
    elif kind_fun == 23:
        value = df_inv(eta1, eta2, eta3, kind_map, params, 13)
    elif kind_fun == 24:
        value = df_inv(eta1, eta2, eta3, kind_map, params, 21)
    elif kind_fun == 25:
        value = df_inv(eta1, eta2, eta3, kind_map, params, 22)
    elif kind_fun == 26:
        value = df_inv(eta1, eta2, eta3, kind_map, params, 23)
    elif kind_fun == 27:
        value = df_inv(eta1, eta2, eta3, kind_map, params, 31)
    elif kind_fun == 28:
        value = df_inv(eta1, eta2, eta3, kind_map, params, 32)
    elif kind_fun == 29:
        value = df_inv(eta1, eta2, eta3, kind_map, params, 33)
        
    # metric tensor g
    elif kind_fun == 31:
        value = g(eta1, eta2, eta3, kind_map, params, 11)
    elif kind_fun == 32:
        value = g(eta1, eta2, eta3, kind_map, params, 12)
    elif kind_fun == 33:
        value = g(eta1, eta2, eta3, kind_map, params, 13)
    elif kind_fun == 34:
        value = g(eta1, eta2, eta3, kind_map, params, 21)
    elif kind_fun == 35:
        value = g(eta1, eta2, eta3, kind_map, params, 22)
    elif kind_fun == 36:
        value = g(eta1, eta2, eta3, kind_map, params, 23)
    elif kind_fun == 37:
        value = g(eta1, eta2, eta3, kind_map, params, 31)
    elif kind_fun == 38:
        value = g(eta1, eta2, eta3, kind_map, params, 32)
    elif kind_fun == 39:
        value = g(eta1, eta2, eta3, kind_map, params, 33)
        
    # metric tensor g_inv
    elif kind_fun == 41:
        value = g_inv(eta1, eta2, eta3, kind_map, params, 11)
    elif kind_fun == 42:
        value = g_inv(eta1, eta2, eta3, kind_map, params, 12)
    elif kind_fun == 43:
        value = g_inv(eta1, eta2, eta3, kind_map, params, 13)
    elif kind_fun == 44:
        value = g_inv(eta1, eta2, eta3, kind_map, params, 21)
    elif kind_fun == 45:
        value = g_inv(eta1, eta2, eta3, kind_map, params, 22)
    elif kind_fun == 46:
        value = g_inv(eta1, eta2, eta3, kind_map, params, 23)
    elif kind_fun == 47:
        value = g_inv(eta1, eta2, eta3, kind_map, params, 31)
    elif kind_fun == 48:
        value = g_inv(eta1, eta2, eta3, kind_map, params, 32)
    elif kind_fun == 49:
        value = g_inv(eta1, eta2, eta3, kind_map, params, 33)
    
    return value


# ==========================================================================================
@types('double[:]','double[:]','double[:]','double[:,:,:]','int','int','double[:]')        
def kernel_evaluation_tensor(eta1, eta2, eta3, mat_f, kind_fun, kind_map, params):
    
    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            for i3 in range(len(eta3)):
                mat_f[i1, i2, i3] = fun(eta1[i1], eta2[i2], eta3[i3], kind_fun, kind_map, params)
