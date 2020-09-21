from pyccel.decorators import types
from numpy             import sin, cos, pi, zeros, array

# =======================================================================
@types('double','double','double','int','double[:]','int')
def f(eta1, eta2, eta3, kind, params, component):
    '''
    defines an analytical mapping X = f(eta) in three dimensions. 
    X=(X1,X2,X3), eta=(eta1,eta2,eta3)
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    eta3 : double
        3rd logical coordinate in [0, 1]
        
    kind : int
        type of mapping (1 = slab, 2 = hollow cylinder, 3 = colella, 4 = orthogonal)
        
    params : list of doubles
        parameters for the mapping (slab : [Lx, Ly, Lz], hollow cylinder : [R1, R2, Lz], colella : [Lx, Ly, alpha, Lz], orthogonal : [Lx, Ly, alpha, Lz])
        
    component : int
        physical coordinate (1 = x, 2 = y, 3 = z)
        
    
    params    : slab            --> Lx, Ly, Lz
              : hollow cylinder --> R1, R2, Lz
              : colella         --> Lx, Ly, alpha, Lz
                 
    component : 1 (x), 2 (y), 3 (z)
    '''
               
    if kind == 1:
         
        Lx = params[0] 
        Ly = params[1] 
        Lz = params[2]
        
        if   component == 1:
            value = Lx * eta1
        elif component == 2:
            value = Ly * eta2
        elif component == 3:
            value = Lz * eta3
            
    elif kind == 2:
        
        R1 = params[0]
        R2 = params[1]
        Lz = params[2]
        dR = R2 - R1
        
        if   component == 1:
            arg   = 2*pi*eta2
            value = (eta1 * dR + R1) * cos(arg)
        elif component == 2:
            arg   = 2*pi*eta2
            value = (eta1 * dR + R1) * sin(arg)
        elif component == 3:
            value = Lz * eta3
            
    elif kind == 3:
        
        Lx    = params[0]
        Ly    = params[1]
        alpha = params[2]
        Lz    = params[3]
        
        if   component == 1:
            arg1  = 2*pi*eta1
            arg2  = 2*pi*eta2
            value = Lx * (eta1 + alpha * sin(arg1) * sin(arg2))
        elif component == 2:
            arg1  = 2*pi*eta1
            arg2  = 2*pi*eta2
            value = Ly * (eta2 + alpha * sin(arg1) * sin(arg2))
        elif component == 3:
            value = Lz * eta3
    
    elif kind == 4:
        
        Lx    = params[0]
        Ly    = params[1]
        alpha = params[2]
        Lz    = params[3]
        
        if   component == 1:
            arg   = 2*pi*eta1
            value = Lx * (eta1 + alpha * sin(arg))
        elif component == 2:
            arg   = 2*pi*eta2
            value = Ly * (eta2 + alpha * sin(arg))
        elif component == 3:
            value = Lz * eta3
    
    else:
        #raise ValueError("kind of mapping unknown")
        value = -99999999.
                  
    return value



# =======================================================================
@types('double','double','double','int','double[:]','int')
def df(eta1, eta2, eta3, kind, params, component):
    '''
    returns the components of the Jacobian matrix of an analytical mapping X = F(eta) in three dimensions. 

    X=(X1,X2,X3), eta=(eta1,eta2,eta3)
    
    Parameters
    ----------
    eta1 : double
        1st logical coordinate in [0, 1]
        
    eta2 : double
        2nd logical coordinate in [0, 1]
        
    eta3 : double
        3rd logical coordinate in [0, 1]
        
    kind : int
        type of mapping (1 = slab, 2 = hollow cylinder, 3 = colella)
        
    params : list of doubles
        parameters for the mapping (slab : [Lx, Ly, Lz], hollow cylinder : [R1, R2, Lz], colella : [Lx, Ly, alpha, Lz], orthogonal : [Lx, Ly, alpha, Lz])
        
    component : int
        component of Jacobian matrix (11 = df1/deta1, 12 = df1/deta2, 13 = df1/deta3, 21 = df2/deta1, 22 = df2/deta2, 23 = df2/deta3, 31 = df3/deta1, 32 = df3/deta2, 33 = df3/deta3)
        
    
    params    : slab            --> Lx, Ly, Lz
              : hollow cylinder --> R1, R2, Lz
              : colella         --> Lx, Ly, alpha, Lz
                 
    component : 11 (dX1/deta1), 12 (dX1/deta2), 13 (dX1/deta3)
                21 (dX2/deta1), 22 (dX2/deta2), 23 (dX2/deta3)
                31 (dX3/deta1), 32 (dX3/deta2), 33 (dX3/deta3)
    '''
    
    value = 0.
               
    if kind == 1:
         
        Lx = params[0] 
        Ly = params[1] 
        Lz = params[2]
        
        if   component == 11:
            value = Lx
        elif component == 12:
            value = 0.
        elif component == 13:
            value = 0.
        elif component == 21:
            calue = 0.
        elif component == 22:
            value = Ly
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = Lz
        else:
            #raise ValueError("df component not correct")
            value = -99999999.
            
    elif kind == 2:
        
        R1 = params[0]
        R2 = params[1]
        Lz = params[2]
        dR = R2 - R1
        
        if   component == 11:
            arg   = 2*pi*eta2
            value = dR * cos(arg)
        elif component == 12:
            arg   = 2*pi*eta2
            value = -2*pi * (eta1*dR + R1) * sin(arg)
        elif component == 13:
            value = 0.
        elif component == 21:
            arg   = 2*pi*eta2
            value = dR * sin(arg)
        elif component == 22:
            arg   = 2*pi*eta2
            value =  2*pi * (eta1*dR + R1) * cos(arg)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = Lz
        else:
            #raise ValueError("df component not correct")
            value = -99999999.
            
    elif kind == 3:
        
        Lx    = params[0]
        Ly    = params[1]
        alpha = params[2]
        Lz    = params[3]
        
        if   component == 11:
            arg1  = 2*pi*eta1
            arg2  = 2*pi*eta2
            value = Lx * (1 + alpha * cos(arg1) * sin(arg2) * 2*pi)
        elif component == 12:
            arg1  = 2*pi*eta1
            arg2  = 2*pi*eta2
            value = Lx * alpha * sin(arg1) * cos(arg2) * 2*pi
        elif component == 13:
            value = 0.
        elif component == 21:
            arg1  = 2*pi*eta1
            arg2  = 2*pi*eta2
            value = Ly * alpha * cos(arg1) * sin(arg2) * 2*pi
        elif component == 22:
            arg1  = 2*pi*eta1
            arg2  = 2*pi*eta2
            value = Ly * (1 + alpha * sin(arg1) * cos(arg2) * 2*pi)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = Lz
        else:
            #raise ValueError("df component not correct")
            value = -99999999.
                   
    elif kind == 4:
        
        Lx    = params[0]
        Ly    = params[1]
        alpha = params[2]
        Lz    = params[3]
        
        if   component == 11:
            arg   = 2*pi*eta1
            value = Lx * (1 + alpha * cos(arg) * 2*pi)
        elif component == 12:
            value = 0.
        elif component == 13:
            value = 0.
        elif component == 21:
            value = 0.
        elif component == 22:
            arg   = 2*pi*eta2
            value = Ly * (1 + alpha * cos(arg) * 2*pi)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = Lz
        else:
            #raise ValueError("df component not correct")
            value = -99999999.
    else:
        #raise ValueError("kind of mapping unknown")
        value = -99999999.
            
    return value



# =======================================================================
@types('double','double','double','int','double[:]')
def det_df(eta1, eta2, eta3, kind, params):
    '''
    returns the jacobian determinant of an analytical mapping X = F(eta) in three dimensions.

    uses df function

    det_df = dX/deta1 . ( dX/deta2 x dX/deta3)
    
    '''
    
    dX1_deta1=df(eta1, eta2, eta3, kind, params, 11)
    dX2_deta1=df(eta1, eta2, eta3, kind, params, 21)
    dX3_deta1=df(eta1, eta2, eta3, kind, params, 31)
    dX1_deta2=df(eta1, eta2, eta3, kind, params, 12)
    dX2_deta2=df(eta1, eta2, eta3, kind, params, 22)
    dX3_deta2=df(eta1, eta2, eta3, kind, params, 32)
    dX1_deta3=df(eta1, eta2, eta3, kind, params, 13)
    dX2_deta3=df(eta1, eta2, eta3, kind, params, 23)
    dX3_deta3=df(eta1, eta2, eta3, kind, params, 33)

    value  = ( dX1_deta1*(dX2_deta2*dX3_deta3 - dX3_deta2* dX2_deta3 ) 
              +dX2_deta1*(dX3_deta2*dX1_deta3 - dX1_deta2* dX3_deta3 )
              +dX3_deta1*(dX1_deta2*dX2_deta3 - dX2_deta2* dX1_deta3 ) )
            
    return value




# =======================================================================
@types('double','double','double','int','double[:]','int')
def df_inv(eta1, eta2, eta3, kind, params, component):
    '''
    returns the components of the Jacobian matrix of an analytical mapping X= F(eta) in three dimensions. 

    X=(X1,X2,X3), eta=(eta1,eta2,eta3)

    the jacobian stems from the chain rule:

    DF_ij = dX^i/deta^j

    | d/deta1 |   | dX1/deta1 dX2/deta1 dX3/deta1 | | d/dX1 |      | d/dX1 |   
    | d/deta2 | = | dX1/deta2 dX2/deta2 dX3/deta2 | | d/dX2 | = DF | d/dX2 |
    | d/deta3 |   | dX1/deta3 dX2/deta3 dX3/deta3 | | d/dX3 |      | d/dX3 |   

    So the inverse gives

    | d/dX1 |    | deta1/dX1 deta2/dX1 deta3/dX1 | | d/deta1 |              | d/deta1 | 
    | d/dX2 |  = | deta1/dX2 deta2/dX2 deta3/dX2 | | d/deta2 | =  (DF)^(-1) | d/deta2 | 
    | d/dX3 |    | deta1/dX3 deta2/dX3 deta3/dX3 | | d/deta3 |              | d/deta3 | 
                 
    component : 11 (deta1/dX1), 12 (deta2/dX1), 13 (deta3/dX1)
                21 (deta1/dX2), 22 (deta2/dX2), 23 (deta3/dX2)
                31 (deta1/dX3), 32 (deta2/dX3), 33 (deta3/dX3)

    We will compute the 3x3 inverse from DF directly, using the cross product of the columns of DF:

                            | [ (dX/deta2) x (dX/deta3) ]^T |
    (DF)^(-1) = 1/det_df *  | [ (dX/deta3) x (dX/deta1) ]^T |
                            | [ (dX/deta1) x (dX/deta2) ]^T |

    '''

    dX1_deta1=df(eta1, eta2, eta3, kind, params, 11)
    dX2_deta1=df(eta1, eta2, eta3, kind, params, 21)
    dX3_deta1=df(eta1, eta2, eta3, kind, params, 31)
    dX1_deta2=df(eta1, eta2, eta3, kind, params, 12)
    dX2_deta2=df(eta1, eta2, eta3, kind, params, 22)
    dX3_deta2=df(eta1, eta2, eta3, kind, params, 32)
    dX1_deta3=df(eta1, eta2, eta3, kind, params, 13)
    dX2_deta3=df(eta1, eta2, eta3, kind, params, 23)
    dX3_deta3=df(eta1, eta2, eta3, kind, params, 33)

    det_df_loc = ( dX1_deta1*(dX2_deta2*dX3_deta3 - dX3_deta2* dX2_deta3 ) 
                  +dX2_deta1*(dX3_deta2*dX1_deta3 - dX1_deta2* dX3_deta3 )
                  +dX3_deta1*(dX1_deta2*dX2_deta3 - dX2_deta2* dX1_deta3 ) )

    if   component == 11:
        value           = (dX2_deta2*dX3_deta3 - dX3_deta2* dX2_deta3 )
    elif component == 12:
        value           = (dX3_deta2*dX1_deta3 - dX1_deta2* dX3_deta3 )
    elif component == 13:
        value           = (dX1_deta2*dX2_deta3 - dX2_deta2* dX1_deta3 )
    elif component == 21:
        value           = (dX2_deta3*dX3_deta1 - dX3_deta3* dX2_deta1 )
    elif component == 22:
        value           = (dX3_deta3*dX1_deta1 - dX1_deta3* dX3_deta1 )
    elif component == 23:
        value           = (dX1_deta3*dX2_deta1 - dX2_deta3* dX1_deta1 )
    elif component == 31:
        value           = (dX2_deta1*dX3_deta2 - dX3_deta1* dX2_deta2 )
    elif component == 32:
        value           = (dX3_deta1*dX1_deta2 - dX1_deta1* dX3_deta2 )
    elif component == 33:
        value           = (dX1_deta1*dX2_deta2 - dX2_deta1* dX1_deta2 )
    else:
        #raise ValueError("kind of mapping unknown")
        value = -99999999.

    value=value/det_df_loc
            
    return value




# =======================================================================
@types('double','double','double','int','double[:]','int')
def g(eta1, eta2, eta3, kind, params, component):
    '''
    returns the components of the symmetric metric tensor (DF)^T*DF 
    of an analytical mapping X = F(eta) in three dimensions. 

    X=(X1,X2,X3), eta=(eta1,eta2,eta3)
                 
    component ij = sum_k (DF_ki DF_kj)
    '''

    if   component == 11:
        df_11 = df(eta1, eta2, eta3, kind, params, 11)
        df_21 = df(eta1, eta2, eta3, kind, params, 21)
        df_31 = df(eta1, eta2, eta3, kind, params, 31)
        value = (df_11*df_11 + df_21*df_21 + df_31*df_31 )  
    elif component == 22:                                              
        df_12 = df(eta1, eta2, eta3, kind, params, 12)
        df_22 = df(eta1, eta2, eta3, kind, params, 22)
        df_32 = df(eta1, eta2, eta3, kind, params, 32)
        value = (df_12*df_12 + df_22*df_22 + df_32*df_32 )
    elif component == 33:                                              
        df_13 = df(eta1, eta2, eta3, kind, params, 13)
        df_23 = df(eta1, eta2, eta3, kind, params, 23)
        df_33 = df(eta1, eta2, eta3, kind, params, 33)
        value = (df_13*df_13 + df_23*df_23 + df_33*df_33 )
    elif ((component == 12) or (component == 21)) :
        df_11 = df(eta1, eta2, eta3, kind, params, 11)
        df_21 = df(eta1, eta2, eta3, kind, params, 21)
        df_31 = df(eta1, eta2, eta3, kind, params, 31)
        df_12 = df(eta1, eta2, eta3, kind, params, 12)
        df_22 = df(eta1, eta2, eta3, kind, params, 22)
        df_32 = df(eta1, eta2, eta3, kind, params, 32)
        value = (df_11*df_12 + df_21*df_22 + df_31*df_32 )
    elif ((component == 13) or (component == 31)):
        df_11 = df(eta1, eta2, eta3, kind, params, 11)
        df_21 = df(eta1, eta2, eta3, kind, params, 21)
        df_31 = df(eta1, eta2, eta3, kind, params, 31)
        df_13 = df(eta1, eta2, eta3, kind, params, 13)
        df_23 = df(eta1, eta2, eta3, kind, params, 23)
        df_33 = df(eta1, eta2, eta3, kind, params, 33)
        value = (df_11*df_13 + df_21*df_23 + df_31*df_33 )
    elif ((component == 23) or (component == 32)):  
        df_12 = df(eta1, eta2, eta3, kind, params, 12)
        df_22 = df(eta1, eta2, eta3, kind, params, 22)
        df_32 = df(eta1, eta2, eta3, kind, params, 32)
        df_13 = df(eta1, eta2, eta3, kind, params, 13)
        df_23 = df(eta1, eta2, eta3, kind, params, 23)
        df_33 = df(eta1, eta2, eta3, kind, params, 33)
        value = (df_12*df_13 + df_22*df_23 + df_32*df_33 )
    else:
        #raise ValueError("kind of mapping unknown")
        value = -99999999.
        
            
    return value


# =======================================================================
@types('double','double','double','int','double[:]','int')
def g_inv(eta1, eta2, eta3, kind, params, component):
    '''
    returns the components of the inverse symmetric metric tensor (DF)^(-1)*DF^(-T) 
    of an analytical mapping X = F(eta) in three dimensions. 
    
    X=(X1,X2,X3), eta=(eta1,eta2,eta3)
                 
    component ij = sum_k ( (DFinv)_ik (DFinv)_jk )
    '''

    if   component == 11:
        dfi_11 = df_inv(eta1, eta2, eta3, kind, params, 11)
        dfi_12 = df_inv(eta1, eta2, eta3, kind, params, 12)
        dfi_13 = df_inv(eta1, eta2, eta3, kind, params, 13)
        value  = (dfi_11*dfi_11 + dfi_12*dfi_12 + dfi_13*dfi_13 )  
    elif component == 22:                                              
        dfi_21 = df_inv(eta1, eta2, eta3, kind, params, 21)
        dfi_22 = df_inv(eta1, eta2, eta3, kind, params, 22)
        dfi_23 = df_inv(eta1, eta2, eta3, kind, params, 23)
        value  = (dfi_21*dfi_21 + dfi_22*dfi_22 + dfi_23*dfi_23 )
    elif component == 33:                                              
        dfi_31 = df_inv(eta1, eta2, eta3, kind, params, 31)
        dfi_32 = df_inv(eta1, eta2, eta3, kind, params, 32)
        dfi_33 = df_inv(eta1, eta2, eta3, kind, params, 33)
        value  = (dfi_31*dfi_31 + dfi_32*dfi_32 + dfi_33*dfi_33 )
    elif ((component == 12) or (component == 21)) :
        dfi_11 = df_inv(eta1, eta2, eta3, kind, params, 11)
        dfi_12 = df_inv(eta1, eta2, eta3, kind, params, 12)
        dfi_13 = df_inv(eta1, eta2, eta3, kind, params, 13)
        dfi_21 = df_inv(eta1, eta2, eta3, kind, params, 21)
        dfi_22 = df_inv(eta1, eta2, eta3, kind, params, 22)
        dfi_23 = df_inv(eta1, eta2, eta3, kind, params, 23)
        value  = (dfi_11*dfi_21 + dfi_12*dfi_22 + dfi_13*dfi_23 )
    elif ((component == 13) or (component == 31)):
        dfi_11 = df_inv(eta1, eta2, eta3, kind, params, 11)
        dfi_12 = df_inv(eta1, eta2, eta3, kind, params, 12)
        dfi_13 = df_inv(eta1, eta2, eta3, kind, params, 13)
        dfi_31 = df_inv(eta1, eta2, eta3, kind, params, 31)
        dfi_32 = df_inv(eta1, eta2, eta3, kind, params, 32)
        dfi_33 = df_inv(eta1, eta2, eta3, kind, params, 33)
        value  = (dfi_11*dfi_31 + dfi_12*dfi_32 + dfi_13*dfi_33 )
    elif ((component == 23) or (component == 32)):  
        dfi_21 = df_inv(eta1, eta2, eta3, kind, params, 21)
        dfi_22 = df_inv(eta1, eta2, eta3, kind, params, 22)
        dfi_23 = df_inv(eta1, eta2, eta3, kind, params, 23)
        dfi_31 = df_inv(eta1, eta2, eta3, kind, params, 31)
        dfi_32 = df_inv(eta1, eta2, eta3, kind, params, 32)
        dfi_33 = df_inv(eta1, eta2, eta3, kind, params, 33)
        value  = (dfi_21*dfi_31 + dfi_22*dfi_32 + dfi_23*dfi_33 )
    else:
        #raise ValueError("kind of mapping unknown")
        value = -99999999.
        
            
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
