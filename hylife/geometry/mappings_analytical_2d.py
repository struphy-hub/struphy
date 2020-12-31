# coding: utf-8
#
# Copnyright 2020 Florian Holderied

"""
Basic functions for point-wise evaluation of a 2d analytical mapping and its corresponding geometric quantities:
"""

from pyccel.decorators import types
from numpy             import sin, cos, pi, zeros, array, sqrt


# =======================================================================
@types('double','double','int','double[:]','int')
def f(eta1, eta2, kind, params, component):
    """
    returns the components of an analytical mapping x, y = F(eta1, eta2) in two dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    kind : int
        kind of mapping (1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : list of doubles
        parameters for the mapping (1 : [lx, ly], 2 : [r1, r2], 3 : [lx, ly, alpha], 4 : [lx, ly, alpha])
        
    component : int
        physical coordinate (1 : x, 2 : y)
    """
   
    value = 0.

    # ============== slab ==================
    if kind == 1:
         
        lx = params[0] 
        ly = params[1] 
        
        if   component == 1:
            value = lx * eta1
        elif component == 2:
            value = ly * eta2
            
    # ============= annulus ================
    elif kind == 2:
        
        r1 = params[0]
        r2 = params[1]
        dr = r2 - r1
        
        if   component == 1:
            value = (eta1 * dr + r1) * cos(2*pi*eta2)
        elif component == 2:
            value = (eta1 * dr + r1) * sin(2*pi*eta2)
            
    # ============ colella =================
    elif kind == 3:
        
        lx    = params[0]
        ly    = params[1]
        alpha = params[2]
        
        if   component == 1:
            value = lx * (eta1 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        elif component == 2:
            value = ly * (eta2 + alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
    
    # =========== orthogonal ===============
    elif kind == 4:
        
        lx    = params[0]
        ly    = params[1]
        alpha = params[2]
        
        if   component == 1:
            value = lx * (eta1 + alpha * sin(2*pi*eta1))
        elif component == 2:
            value = ly * (eta2 + alpha * sin(2*pi*eta2))
                  
    return value


# =======================================================================
@types('double','double','int','double[:]','int')
def df(eta1, eta2, kind, params, component):
    """
    returns the components of the Jacobian matrix DF_ij = dF_i/deta_j of an analytical mapping x, y = F(eta1, eta2) in two dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    kind : int
        kind of mapping (1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : list of doubles
        parameters for the mapping (1 : [lx, ly], 2 : [r1, r2], 3 : [lx, ly, alpha], 4 : [lx, ly, alpha])
                 
    component : int 
        11 : (dx/deta1), 12 : (dx/deta2)
        21 : (dy/deta1), 22 : (dy/deta2)
    """
    
    value = 0.
               
    # ============== slab ==================
    if kind == 1:
         
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
    elif kind == 2:
        
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
    elif kind == 3:
        
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
    elif kind == 4:
        
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
@types('double','double','int','double[:]')
def det_df(eta1, eta2, kind, params):
    """
    returns the Jacobian determinant det(DF) of an analytical mapping x, y = F(eta1, eta2) in three dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    kind : int
        kind of mapping (1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : list of doubles
        parameters for the mapping (1 : [lx, ly], 2 : [r1, r2], 3 : [lx, ly, alpha], 4 : [lx, ly, alpha])
                 
    Returns
    -------
    value : double
        the Jacobian determinant dx/deta1 * dy/deta2 - dx/deta2 * dy/deta1
    """
    
    value = 0.
    
    df_11 = df(eta1, eta2, kind, params, 11)
    df_12 = df(eta1, eta2, kind, params, 12)
    
    df_21 = df(eta1, eta2, kind, params, 21)
    df_22 = df(eta1, eta2, kind, params, 22)

    value = df_11*df_22 - df_12*df_21
            
    return value


# =======================================================================
@types('double','double','int','double[:]','int')
def df_inv(eta1, eta2, kind, params, component):
    """
    returns the components of the inverse of the Jacobian matrix DF_ij = dF_i/deta_j of an analytical mapping x, y = F(eta1, eta2) in two dimensions. 
    
    the 2x2 inverse is computed directly from DF:

                            |  DF_22 -DF_12 |
    (DF)^(-1) = 1/det_df *  | -DF_21  DF_11 |
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    kind : int
        kind of mapping (1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : list of doubles
        parameters for the mapping (1 : [lx, ly], 2 : [r1, r2], 3 : [lx, ly, alpha], 4 : [lx, ly, alpha])
                 
    component : int 
    """
    
    value = 0.

    df_11 = df(eta1, eta2, kind, params, 11)
    df_12 = df(eta1, eta2, kind, params, 12)
    
    df_21 = df(eta1, eta2, kind, params, 21)
    df_22 = df(eta1, eta2, kind, params, 22)
    
    detdf = df_11*df_22 - df_12*df_21

    if   component == 11:
        value = df_22
    elif component == 12:
        value = df_12
    elif component == 21:
        value = df_21
    elif component == 22:
        value = df_11
        
    value = value/detdf
            
    return value


# =======================================================================
@types('double','double','int','double[:]','int')
def g(eta1, eta2, kind, params, component):
    """
    returns the components of the metric tensor G = DF^T * DF of an analytical mapping x, y = F(eta1, eta2) in two dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    kind : int
        kind of mapping (1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : list of doubles
        parameters for the mapping (1 : [lx, ly], 2 : [r1, r2], 3 : [lx, ly, alpha], 4 : [lx, ly, alpha])
                 
    component : int 
    """
    
    value = 0.

    if   component == 11:
        df_11 = df(eta1, eta2, kind, params, 11)
        df_21 = df(eta1, eta2, kind, params, 21)
        value = df_11*df_11 + df_21*df_21
        
    elif component == 22:                                              
        df_12 = df(eta1, eta2, kind, params, 12)
        df_22 = df(eta1, eta2, kind, params, 22)
        value = df_12*df_12 + df_22*df_22
                 
    elif ((component == 12) or (component == 21)) :
        df_11 = df(eta1, eta2, kind, params, 11)
        df_21 = df(eta1, eta2, kind, params, 21)
        df_12 = df(eta1, eta2, kind, params, 12)
        df_22 = df(eta1, eta2, kind, params, 22)
        value = df_11*df_12 + df_21*df_22
               
    return value


# =======================================================================
@types('double','double','int','double[:]','int')
def g_inv(eta1, eta2, kind, params, component):
    """
    returns the components of the iverse metric tensor G = DF^(-1) * DF^(-T) of an analytical mapping x, y = F(eta1, eta2) in two dimensions. 
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    kind : int
        kind of mapping (1 : slab, 2 : annulus, 3 : colella, 4 : orthogonal)
        
    params : list of doubles
        parameters for the mapping (1 : [lx, ly], 2 : [r1, r2], 3 : [lx, ly, alpha], 4 : [lx, ly, alpha])
                 
    component : int 
    """
    
    value = 0.

    if   component == 11:
        dfinv_11 = df_inv(eta1, eta2, kind, params, 11)
        dfinv_12 = df_inv(eta1, eta2, kind, params, 12)
        value    = dfinv_11*dfinv_11 + dfinv_12*dfinv_12
                  
    elif component == 22:                                              
        dfinv_21 = df_inv(eta1, eta2, kind, params, 21)
        dfinv_22 = df_inv(eta1, eta2, kind, params, 22)
        value    = dfinv_21*dfinv_21 + dfinv_22*dfinv_22
                  
    elif ((component == 12) or (component == 21)) :
        dfinv_11 = df_inv(eta1, eta2, kind, params, 11)
        dfinv_12 = df_inv(eta1, eta2, kind, params, 12)
        dfinv_21 = df_inv(eta1, eta2, kind, params, 21)
        dfinv_22 = df_inv(eta1, eta2, kind, params, 22)
        value    = dfinv_11*dfinv_21 + dfinv_12*dfinv_22
    
    return value


# ==========================================================================================
@types('double','double','int','int','double[:]')        
def fun(eta1, eta2, kind_fun, kind_map, params):
    
    value = 0.
    
    # mapping f
    if   kind_fun == 1:
        value = f(eta1, eta2, kind_map, params, 1)
    elif kind_fun == 2:
        value = f(eta1, eta2, kind_map, params, 2)
    
    # Jacobian matrix df
    elif kind_fun == 11:
        value = df(eta1, eta2, kind_map, params, 11)
    elif kind_fun == 12:
        value = df(eta1, eta2, kind_map, params, 12)
    elif kind_fun == 13:
        value = df(eta1, eta2, kind_map, params, 21)
    elif kind_fun == 14:
        value = df(eta1, eta2, kind_map, params, 22)
        
    # Jacobian determinant det_df
    elif kind_fun == 3:
        value = det_df(eta1, eta2, kind_map, params)
        
    # inverse Jacobian matrix df_inv
    elif kind_fun == 21:
        value = df_inv(eta1, eta2, kind_map, params, 11)
    elif kind_fun == 22:
        value = df_inv(eta1, eta2, kind_map, params, 12)
    elif kind_fun == 23:
        value = df_inv(eta1, eta2, kind_map, params, 21)
    elif kind_fun == 24:
        value = df_inv(eta1, eta2, kind_map, params, 22)
        
    # metric tensor g
    elif kind_fun == 31:
        value = g(eta1, eta2, kind_map, params, 11)
    elif kind_fun == 32:
        value = g(eta1, eta2, kind_map, params, 12)
    elif kind_fun == 33:
        value = g(eta1, eta2, kind_map, params, 21)
    elif kind_fun == 34:
        value = g(eta1, eta2, kind_map, params, 22)
    
        
    # metric tensor g_inv
    elif kind_fun == 41:
        value = g_inv(eta1, eta2, kind_map, params, 11)
    elif kind_fun == 42:
        value = g_inv(eta1, eta2, kind_map, params, 12)
    elif kind_fun == 43:
        value = g_inv(eta1, eta2, kind_map, params, 21)
    elif kind_fun == 44:
        value = g_inv(eta1, eta2, kind_map, params, 22)
    
    return value


# ==========================================================================================
@types('double[:]','double[:]','double[:,:]','int','int','double[:]')        
def kernel_evaluation_tensor(eta1, eta2, mat_f, kind_fun, kind_map, params):
    
    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            mat_f[i1, i2] = fun(eta1[i1], eta2[i2], kind_fun, kind_map, params)
