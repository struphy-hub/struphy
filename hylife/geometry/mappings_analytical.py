from pyccel.decorators import types
from numpy             import sin, cos, pi, zeros, array

__all__ = ['f',
           'df',
           'det_df',
           'df_inv',
           'g',
           'g_inv',
           'pull_v_pw',
           'pull_v',
           'pull_0_pw',
           'pull_0',
           'pull_1_pw',
           'pull_1',
           'pull_2_pw',
           'pull_2',
           'pull_3_pw',
           'pull_3',
           'push_v_pw',
           'push_v',
           'push_0_pw',
           'push_0',
           'push_1_pw',
           'push_1',
           'push_2_pw',
           'push_2',
           'push_3_pw',
           'push_3']

# =======================================================================
@types('double','double','double','int','double[:]','int')
def f(xi1, xi2, xi3, kind, params, component):

    '''
    defines an analytical mapping X = f(xi) in three dimensions. 
    X=(X1,X2,X3), xi=(xi1,xi2,xi3)
   
    Parameters
    ----------
    xi1 : double
        1st logical coordinate in [0, 1]
        
    xi2 : double
        2nd logical coordinate in [0, 1]
        
    xi3 : double
        3rd logical coordinate in [0, 1]
        
    kind : int
        type of mapping (1 = slab, 2 = hollow cylinder, 3 = colella, 4 = orthogonal)
        
    params : list of doubles
        parameters for the mapping (slab : [Lx, Ly, Lz], hollow cylinder : [R1, R2, Lz], colella : [Lx, Ly, alpha, Lz], orthogonal : [Lx, Ly, alpha, Lz])
        
    component : int
        physical coordinate (1 = x, 2 = y, 3 = z)
        
    
   Returns
    -------
    value : double
        the physical cooordinate corresponding to the logical coordinate (xi1, xi2, xi3)
    """
    
    value = 0.
               
    if kind == 1:
         
        Lx = params[0] 
        Ly = params[1] 
        Lz = params[2]
        
        if   component == 1:
            value = Lx * xi1
        elif component == 2:
            value = Ly * xi2
        elif component == 3:
            value = Lz * xi3
            
    elif kind == 2:
        
        R1 = params[0]
        R2 = params[1]
        Lz = params[2]
        dR = R2 - R1
        
        if   component == 1:
            arg   = 2*pi*xi2
            value = (xi1 * dR + R1) * cos(arg)
        elif component == 2:
            arg   = 2*pi*xi2
            value = (xi1 * dR + R1) * sin(arg)
        elif component == 3:
            value = Lz * xi3
            
    elif kind == 3:
        
        Lx    = params[0]
        Ly    = params[1]
        alpha = params[2]
        Lz    = params[3]
        
        if   component == 1:
            arg1  = 2*pi*xi1
            arg2  = 2*pi*xi2
            value = Lx * (xi1 + alpha * sin(arg1) * sin(arg2))
        elif component == 2:
            arg1  = 2*pi*xi1
            arg2  = 2*pi*xi2
            value = Ly * (xi2 + alpha * sin(arg1) * sin(arg2))
        elif component == 3:
            value = Lz * xi3
    else:
        #raise ValueError("kind of mapping unknown")
        value = -99999999.
            
    elif kind == 4:
        
        Lx    = params[0]
        Ly    = params[1]
        alpha = params[2]
        Lz    = params[3]
        
        if   component == 1:
            arg   = 2*pi*xi1
            value = Lx * (xi1 + alpha * sin(arg))
        elif component == 2:
            arg   = 2*pi*xi2
            value = Ly * (xi2 + alpha * sin(arg))
        elif component == 3:
            value = Lz * xi3
                 
    return value



# =======================================================================
@types('double','double','double','int','double[:]','int')
def df(xi1, xi2, xi3, kind, params, component):

    '''
    returns the components of the Jacobian matrix of an analytical mapping X = F(xi) in three dimensions. 

    X=(X1,X2,X3), xi=(xi1,xi2,xi3)
   
    Parameters
    ----------
    xi1 : double
        1st logical coordinate in [0, 1]
        
    xi2 : double
        2nd logical coordinate in [0, 1]
        
    xi3 : double
        3rd logical coordinate in [0, 1]
        
    kind : int
        type of mapping (1 = slab, 2 = hollow cylinder, 3 = colella)
        
    params : list of doubles
        parameters for the mapping (slab : [Lx, Ly, Lz], hollow cylinder : [R1, R2, Lz], colella : [Lx, Ly, alpha, Lz], orthogonal : [Lx, Ly, alpha, Lz])
        
    component : int
        component of Jacobian matrix (11 = df1/dxi1, 12 = df1/dxi2, 13 = df1/dxi3, 21 = df2/dxi1, 22 = df2/dxi2, 23 = df2/dxi3, 31 = df3/dxi1, 32 = df3/dxi2, 33 = df3/dxi3)
        
    
   Returns
    -------
    value : double
        the component of the Jacobian matrix evaluated at the logical coordinate (xi1, xi2, xi3)
    """
    
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
            arg   = 2*pi*xi2
            value = dR * cos(arg)
        elif component == 12:
            arg   = 2*pi*xi2
            value = -2*pi * (xi1*dR + R1) * sin(arg)
        elif component == 13:
            value = 0.
        elif component == 21:
            arg   = 2*pi*xi2
            value = dR * sin(arg)
        elif component == 22:
            arg   = 2*pi*xi2
            value =  2*pi * (xi1*dR + R1) * cos(arg)
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
            arg1  = 2*pi*xi1
            arg2  = 2*pi*xi2
            value = Lx * (1 + alpha * cos(arg1) * sin(arg2) * 2*pi)
        elif component == 12:
            arg1  = 2*pi*xi1
            arg2  = 2*pi*xi2
            value = Lx * alpha * sin(arg1) * cos(arg2) * 2*pi
        elif component == 13:
            value = 0.
        elif component == 21:
            arg1  = 2*pi*xi1
            arg2  = 2*pi*xi2
            value = Ly * alpha * cos(arg1) * sin(arg2) * 2*pi
        elif component == 22:
            arg1  = 2*pi*xi1
            arg2  = 2*pi*xi2
            value = Ly * (1 + alpha * sin(arg1) * cos(arg2) * 2*pi)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = Lz
            
            
    elif kind == 4:
        
        Lx    = params[0]
        Ly    = params[1]
        alpha = params[2]
        Lz    = params[3]
        
        if   component == 11:
            arg   = 2*pi*xi1
            value = Lx * (1 + alpha * cos(arg) * 2*pi)
        elif component == 12:
            value = 0.
        elif component == 13:
            value = 0.
        elif component == 21:
            value = 0.
        elif component == 22:
            arg   = 2*pi*xi2
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
def det_df(xi1, xi2, xi3, kind, params):

   """
    Returns the determinant of the Jacobian matrix df/dxi corresponding to the mapping f.
    
    Parameters
    ----------
    xi1 : double
        1st logical coordinate in [0, 1]
        
    xi2 : double
        2nd logical coordinate in [0, 1]
        
    xi3 : double
        3rd logical coordinate in [0, 1]
        
    kind : int
        type of mapping (1 = slab, 2 = hollow cylinder, 3 = colella)
        
    params : list of doubles
        parameters for the mapping (slab : [Lx, Ly, Lz], hollow cylinder : [R1, R2, Lz], colella : [Lx, Ly, alpha, Lz], orthogonal : [Lx, Ly, alpha, Lz])
        
        
    Returns
    -------
    value : double
        the Jacobian determinant evaluated at the logical coordinate (xi1, xi2, xi3)
    """
    
    dX1_dxi1=df(xi1, xi2, xi3, kind, params, 11)
    dX2_dxi1=df(xi1, xi2, xi3, kind, params, 21)
    dX3_dxi1=df(xi1, xi2, xi3, kind, params, 31)
    dX1_dxi2=df(xi1, xi2, xi3, kind, params, 12)
    dX2_dxi2=df(xi1, xi2, xi3, kind, params, 22)
    dX3_dxi2=df(xi1, xi2, xi3, kind, params, 32)
    dX1_dxi3=df(xi1, xi2, xi3, kind, params, 13)
    dX2_dxi3=df(xi1, xi2, xi3, kind, params, 23)
    dX3_dxi3=df(xi1, xi2, xi3, kind, params, 33)

    value  = ( dX1_dxi1*(dX2_dxi2*dX3_dxi3 - dX3_dxi2* dX2_dxi3 ) 
              +dX2_dxi1*(dX3_dxi2*dX1_dxi3 - dX1_dxi2* dX3_dxi3 )
              +dX3_dxi1*(dX1_dxi2*dX2_dxi3 - dX2_dxi2* dX1_dxi3 ) )
            
    return value



# =======================================================================
@types('double','double','double','int','double[:]','int')
def df_inv(xi1, xi2, xi3, kind, params, component):
    '''
    returns the components of the Jacobian matrix of an analytical mapping X= F(xi) in three dimensions. 

    X=(X1,X2,X3), xi=(xi1,xi2,xi3)

    the jacobian stems from the chain rule:

    DF_ij = dX^i/dxi^j

    | d/dxi1 |   | dX1/dxi1 dX2/dxi1 dX3/dxi1 | | d/dX1 |      | d/dX1 |   
    | d/dxi2 | = | dX1/dxi2 dX2/dxi2 dX3/dxi2 | | d/dX2 | = DF | d/dX2 |
    | d/dxi3 |   | dX1/dxi3 dX2/dxi3 dX3/dxi3 | | d/dX3 |      | d/dX3 |   

    So the inverse gives

    | d/dX1 |    | dxi1/dX1 dxi2/dX1 dxi3/dX1 | | d/dxi1 |              | d/dxi1 | 
    | d/dX2 |  = | dxi1/dX2 dxi2/dX2 dxi3/dX2 | | d/dxi2 | =  (DF)^(-1) | d/dxi2 | 
    | d/dX3 |    | dxi1/dX3 dxi2/dX3 dxi3/dX3 | | d/dxi3 |              | d/dxi3 | 
                 
    component : 11 (dxi1/dX1), 12 (dxi2/dX1), 13 (dxi3/dX1)
                21 (dxi1/dX2), 22 (dxi2/dX2), 23 (dxi3/dX2)
                31 (dxi1/dX3), 32 (dxi2/dX3), 33 (dxi3/dX3)

    We will compute the 3x3 inverse from DF directly, using the cross product of the columns of DF:

                            | [ (dX/dxi2) x (dX/dxi3) ]^T |
    (DF)^(-1) = 1/det_df *  | [ (dX/dxi3) x (dX/dxi1) ]^T |
                            | [ (dX/dxi1) x (dX/dxi2) ]^T |

    '''

    dX1_dxi1=df(xi1, xi2, xi3, kind, params, 11)
    dX2_dxi1=df(xi1, xi2, xi3, kind, params, 21)
    dX3_dxi1=df(xi1, xi2, xi3, kind, params, 31)
    dX1_dxi2=df(xi1, xi2, xi3, kind, params, 12)
    dX2_dxi2=df(xi1, xi2, xi3, kind, params, 22)
    dX3_dxi2=df(xi1, xi2, xi3, kind, params, 32)
    dX1_dxi3=df(xi1, xi2, xi3, kind, params, 13)
    dX2_dxi3=df(xi1, xi2, xi3, kind, params, 23)
    dX3_dxi3=df(xi1, xi2, xi3, kind, params, 33)

    det_df_loc = ( dX1_dxi1*(dX2_dxi2*dX3_dxi3 - dX3_dxi2* dX2_dxi3 ) 
                  +dX2_dxi1*(dX3_dxi2*dX1_dxi3 - dX1_dxi2* dX3_dxi3 )
                  +dX3_dxi1*(dX1_dxi2*dX2_dxi3 - dX2_dxi2* dX1_dxi3 ) )

    if   component == 11:
        value           = (dX2_dxi2*dX3_dxi3 - dX3_dxi2* dX2_dxi3 )
    elif component == 12:
        value           = (dX3_dxi2*dX1_dxi3 - dX1_dxi2* dX3_dxi3 )
    elif component == 13:
        value           = (dX1_dxi2*dX2_dxi3 - dX2_dxi2* dX1_dxi3 )
    elif component == 21:
        value           = (dX2_dxi3*dX3_dxi1 - dX3_dxi3* dX2_dxi1 )
    elif component == 22:
        value           = (dX3_dxi3*dX1_dxi1 - dX1_dxi3* dX3_dxi1 )
    elif component == 23:
        value           = (dX1_dxi3*dX2_dxi1 - dX2_dxi3* dX1_dxi1 )
    elif component == 31:
        value           = (dX2_dxi1*dX3_dxi2 - dX3_dxi1* dX2_dxi2 )
    elif component == 32:
        value           = (dX3_dxi1*dX1_dxi2 - dX1_dxi1* dX3_dxi2 )
    elif component == 33:
        value           = (dX1_dxi1*dX2_dxi2 - dX2_dxi1* dX1_dxi2 )
    else:
        #raise ValueError("kind of mapping unknown")
        value = -99999999.

    value=value/det_df_loc
            
    return value


# =======================================================================
@types('double','double','double','int','double[:]','int')
def g(xi1, xi2, xi3, kind, params, component):
    '''
    returns the components of the symmetric metric tensor (DF)^T*DF 
    of an analytical mapping X = F(xi) in three dimensions. 

    X=(X1,X2,X3), xi=(xi1,xi2,xi3)
                 
    component ij = sum_k (DF_ki DF_kj)
    '''

    if   component == 11:
        df_11 = df(xi1, xi2, xi3, kind, params, 11)
        df_21 = df(xi1, xi2, xi3, kind, params, 21)
        df_31 = df(xi1, xi2, xi3, kind, params, 31)
        value = (df_11*df_11 + df_21*df_21 + df_31*df_31 )  
    elif component == 22:                                              
        df_12 = df(xi1, xi2, xi3, kind, params, 12)
        df_22 = df(xi1, xi2, xi3, kind, params, 22)
        df_32 = df(xi1, xi2, xi3, kind, params, 32)
        value = (df_12*df_12 + df_22*df_22 + df_32*df_32 )
    elif component == 33:                                              
        df_13 = df(xi1, xi2, xi3, kind, params, 13)
        df_23 = df(xi1, xi2, xi3, kind, params, 23)
        df_33 = df(xi1, xi2, xi3, kind, params, 33)
        value = (df_13*df_13 + df_23*df_23 + df_33*df_33 )
    elif ((component == 12) or (component == 21)) :
        df_11 = df(xi1, xi2, xi3, kind, params, 11)
        df_21 = df(xi1, xi2, xi3, kind, params, 21)
        df_31 = df(xi1, xi2, xi3, kind, params, 31)
        df_12 = df(xi1, xi2, xi3, kind, params, 12)
        df_22 = df(xi1, xi2, xi3, kind, params, 22)
        df_32 = df(xi1, xi2, xi3, kind, params, 32)
        value = (df_11*df_12 + df_21*df_22 + df_31*df_32 )
    elif ((component == 13) or (component == 31)):
        df_11 = df(xi1, xi2, xi3, kind, params, 11)
        df_21 = df(xi1, xi2, xi3, kind, params, 21)
        df_31 = df(xi1, xi2, xi3, kind, params, 31)
        df_13 = df(xi1, xi2, xi3, kind, params, 13)
        df_23 = df(xi1, xi2, xi3, kind, params, 23)
        df_33 = df(xi1, xi2, xi3, kind, params, 33)
        value = (df_11*df_13 + df_21*df_23 + df_31*df_33 )
    elif ((component == 23) or (component == 32)):  
        df_12 = df(xi1, xi2, xi3, kind, params, 12)
        df_22 = df(xi1, xi2, xi3, kind, params, 22)
        df_32 = df(xi1, xi2, xi3, kind, params, 32)
        df_13 = df(xi1, xi2, xi3, kind, params, 13)
        df_23 = df(xi1, xi2, xi3, kind, params, 23)
        df_33 = df(xi1, xi2, xi3, kind, params, 33)
        value = (df_12*df_13 + df_22*df_23 + df_32*df_33 )
    else:
        #raise ValueError("kind of mapping unknown")
        value = -99999999.
        
        arg1  = 2*pi*xi1
        arg2  = 2*pi*xi2
        
        value = Lx*Ly*Lz * (1. + alpha * cos(arg1) * sin(arg2) * 2*pi + alpha * sin(arg1) * cos(arg2) * 2*pi)
        
    elif kind == 4:
        
        Lx    = params[0]
        Ly    = params[1]
        alpha = params[2]
        Lz    = params[3]
        
        arg1  = 2*pi*xi1
        arg2  = 2*pi*xi2
        
        value = Lx*Ly*Lz * (1. + alpha * cos(arg1) * 2*pi) * (1. + alpha * cos(arg2) * 2*pi)
            
    return value


# =======================================================================
@types('double','double','double','int','double[:]','int')
def g_inv(xi1, xi2, xi3, kind, params, component):
    '''
    returns the components of the inverse symmetric metric tensor (DF)^(-1)*DF^(-T) 
    of an analytical mapping X = F(xi) in three dimensions. 
    
    X=(X1,X2,X3), xi=(xi1,xi2,xi3)
                 
    component ij = sum_k ( (DFinv)_ik (DFinv)_jk )
    '''

    if   component == 11:
        dfi_11 = df_inv(xi1, xi2, xi3, kind, params, 11)
        dfi_12 = df_inv(xi1, xi2, xi3, kind, params, 12)
        dfi_13 = df_inv(xi1, xi2, xi3, kind, params, 13)
        value  = (dfi_11*dfi_11 + dfi_12*dfi_12 + dfi_13*dfi_13 )  
    elif component == 22:                                              
        dfi_21 = df_inv(xi1, xi2, xi3, kind, params, 21)
        dfi_22 = df_inv(xi1, xi2, xi3, kind, params, 22)
        dfi_23 = df_inv(xi1, xi2, xi3, kind, params, 23)
        value  = (dfi_21*dfi_21 + dfi_22*dfi_22 + dfi_23*dfi_23 )
    elif component == 33:                                              
        dfi_31 = df_inv(xi1, xi2, xi3, kind, params, 31)
        dfi_32 = df_inv(xi1, xi2, xi3, kind, params, 32)
        dfi_33 = df_inv(xi1, xi2, xi3, kind, params, 33)
        value  = (dfi_31*dfi_31 + dfi_32*dfi_32 + dfi_33*dfi_33 )
    elif ((component == 12) or (component == 21)) :
        dfi_11 = df_inv(xi1, xi2, xi3, kind, params, 11)
        dfi_12 = df_inv(xi1, xi2, xi3, kind, params, 12)
        dfi_13 = df_inv(xi1, xi2, xi3, kind, params, 13)
        dfi_21 = df_inv(xi1, xi2, xi3, kind, params, 21)
        dfi_22 = df_inv(xi1, xi2, xi3, kind, params, 22)
        dfi_23 = df_inv(xi1, xi2, xi3, kind, params, 23)
        value  = (dfi_11*dfi_21 + dfi_12*dfi_22 + dfi_13*dfi_23 )
    elif ((component == 13) or (component == 31)):
        dfi_11 = df_inv(xi1, xi2, xi3, kind, params, 11)
        dfi_12 = df_inv(xi1, xi2, xi3, kind, params, 12)
        dfi_13 = df_inv(xi1, xi2, xi3, kind, params, 13)
        dfi_31 = df_inv(xi1, xi2, xi3, kind, params, 31)
        dfi_32 = df_inv(xi1, xi2, xi3, kind, params, 32)
        dfi_33 = df_inv(xi1, xi2, xi3, kind, params, 33)
        value  = (dfi_11*dfi_31 + dfi_12*dfi_32 + dfi_13*dfi_33 )
    elif ((component == 23) or (component == 32)):  
        dfi_21 = df_inv(xi1, xi2, xi3, kind, params, 21)
        dfi_22 = df_inv(xi1, xi2, xi3, kind, params, 22)
        dfi_23 = df_inv(xi1, xi2, xi3, kind, params, 23)
        dfi_31 = df_inv(xi1, xi2, xi3, kind, params, 31)
        dfi_32 = df_inv(xi1, xi2, xi3, kind, params, 32)
        dfi_33 = df_inv(xi1, xi2, xi3, kind, params, 33)
        value  = (dfi_21*dfi_31 + dfi_22*dfi_32 + dfi_23*dfi_33 )
    else:
        #raise ValueError("kind of mapping unknown")
        value = -99999999.
       
            
    return value

#==============================================================================
def pull_v_pw(a1_phys, a2_phys, a3_phys, k, xi1, xi2, xi3, kind, params):
    
    '''
    Returns k-th component of pulled-back vector field, point-wise, according to a(xi) = DF^(-1) * a_phys(F(xi)).
    
    Parameters
    ----------  
        a1_phys, a2_phys, a2_phys: callable
            Components of vector-field on physical domain, a_phys(x,y,z).
            
        k: int
            Which component a_k(xi) of pulled-back vector-field to compute, k=1,2,3.
    
        xi1, xi2, xi3: float
            Coordinates in logical space.    
            
    Returns
    -------
        value: float
    '''

    x = f(xi1, xi2, xi3, kind, params, 1)
    y = f(xi1, xi2, xi3, kind, params, 2)
    z = f(xi1, xi2, xi3, kind, params, 3)

    value = ( df_inv(xi1, xi2, xi3, kind, params, 10*k + 1)*a1_phys(x, y, z) +
              df_inv(xi1, xi2, xi3, kind, params, 10*k + 2)*a2_phys(x, y, z) +
              df_inv(xi1, xi2, xi3, kind, params, 10*k + 3)*a3_phys(x, y, z) )

    return value

#==============================================================================
def pull_v(a1_phys, a2_phys, a3_phys, k, xi1_vec, xi2_vec, xi3_vec, kind, params):
    
    '''
    Returns k-th component of pulled-back vector field as 3D ndarray for integration.
    See 'pull_v_pw' for other arguments.
    '''
    
    if isinstance(xi1_vec, float):
        xi1_vec = array([xi1_vec])
        
    if isinstance(xi2_vec, float):
        xi2_vec = array([xi2_vec])
        
    if isinstance(xi3_vec, float):
        xi3_vec = array([xi3_vec])
    
    values = zeros( (xi1_vec.size, xi2_vec.size, xi3_vec.size), dtype='float' )
    
    for i1, xi1 in enumerate(xi1_vec):
        for i2, xi2 in enumerate(xi2_vec):
            for i3, xi3 in enumerate(xi3_vec):

                values[i1,i2,i3] = pull_v_pw(a1_phys, a2_phys, a3_phys, k, xi1, xi2, xi3, kind, params)

    return values

#==============================================================================
def pull_0_pw(a_phys, xi1, xi2, xi3, kind, params):
    
    '''
    Returns pullback of 0-form, point-wise, according to a(xi) = a_phys(F(xi)).
    
    Parameters
    ---------- 
        a_phys: callable
            0-form on physical domain, a_phys(x,y,z).
    
        xi1, xi2, xi3: float
            Coordinates in logical space.
           
    Returns
    -------
        value: float
    '''

    x = f(xi1, xi2, xi3, kind, params, 1)
    y = f(xi1, xi2, xi3, kind, params, 2)
    z = f(xi1, xi2, xi3, kind, params, 3)

    value = a_phys(x, y, z)

    return value

#==============================================================================
def pull_0(a_phys, xi1_vec, xi2_vec, xi3_vec, kind, params):
    
    '''
    Returns pulled-back 0-form as 3D ndarray for integration.
    See 'pull_0_pw' for other arguments.
    '''
    
    if isinstance(xi1_vec, float):
        xi1_vec = array([xi1_vec])
        
    if isinstance(xi2_vec, float):
        xi2_vec = array([xi2_vec])
        
    if isinstance(xi3_vec, float):
        xi3_vec = array([xi3_vec])
    
    values = zeros( (xi1_vec.size, xi2_vec.size, xi3_vec.size), dtype='float' )
    
    for i1, xi1 in enumerate(xi1_vec):
        for i2, xi2 in enumerate(xi2_vec):
            for i3, xi3 in enumerate(xi3_vec):
                values[i1,i2,i3] = pull_0_pw(a_phys, xi1, xi2, xi3, kind, params)

    return values

#==============================================================================
def pull_1_pw(a1_phys, a2_phys, a3_phys, k, xi1, xi2, xi3, kind, params):
    
    '''
    Returns k-th component of pulled-back 1-form, point-wise, according to a(xi) = DF^T * a_phys(F(xi)).
    
    Parameters
    ---------- 
        a1_phys, a2_phys, a2_phys: callable
            Components of 1-form on physical domain, a_phys(x,y,z).
            
        k: int
            Which component a_k(xi) of pulled-back 1-form to compute, k=1,2,3.
    
        xi1, xi2, xi3: float
            Coordinates in logical space.
            
    Returns
    -------
        value: float
    '''

    x = f(xi1, xi2, xi3, kind, params, 1)
    y = f(xi1, xi2, xi3, kind, params, 2)
    z = f(xi1, xi2, xi3, kind, params, 3)

    value = ( df(xi1, xi2, xi3, kind, params, 10 + k)*a1_phys(x, y, z) +
              df(xi1, xi2, xi3, kind, params, 20 + k)*a2_phys(x, y, z) +
              df(xi1, xi2, xi3, kind, params, 30 + k)*a3_phys(x, y, z) )

    return value

#==============================================================================
def pull_1(a1_phys, a2_phys, a3_phys, k, xi1_vec, xi2_vec, xi3_vec, kind, params):
   
    '''
    Returns k-th component of pulled-back 1-form as 3D ndarray for integration.
    See 'pull_1_pw' for other arguments.
    '''
    
    if isinstance(xi1_vec, float):
        xi1_vec = array([xi1_vec])
        
    if isinstance(xi2_vec, float):
        xi2_vec = array([xi2_vec])
        
    if isinstance(xi3_vec, float):
        xi3_vec = array([xi3_vec])
    
    values = zeros( (xi1_vec.size, xi2_vec.size, xi3_vec.size), dtype='float' )
    
    for i1, xi1 in enumerate(xi1_vec):
        for i2, xi2 in enumerate(xi2_vec):
            for i3, xi3 in enumerate(xi3_vec):

                values[i1,i2,i3] = pull_1_pw(a1_phys, a2_phys, a3_phys, k, xi1, xi2, xi3, kind, params)

    return values

#==============================================================================
def pull_2_pw(a1_phys, a2_phys, a3_phys, k, xi1, xi2, xi3, kind, params):
    
    '''
    Returns k-th component of pulled-back 2-form, point-wise, according to a(xi) = det(DF) * DF^(-1) * a_phys(F(xi)).
    
    Parameters
    ---------- 
        a1_phys, a2_phys, a2_phys: callable
            Components of 2-form on physical domain, a_phys(x,y,z).
            
        k: int
            Which component a_k(xi) of pulled-back 2-form to compute, k=1,2,3.
    
        xi1, xi2, xi3: float
            Coordinates in logical space.    
            
    Returns
    -------
        value: float
    '''

    x = f(xi1, xi2, xi3, kind, params, 1)
    y = f(xi1, xi2, xi3, kind, params, 2)
    z = f(xi1, xi2, xi3, kind, params, 3)

    value = det_df(xi1, xi2, xi3, kind, params)*( df_inv(xi1, xi2, xi3, kind, params, 10*k + 1)*a1_phys(x, y, z) +
                                                  df_inv(xi1, xi2, xi3, kind, params, 10*k + 2)*a2_phys(x, y, z) +
                                                  df_inv(xi1, xi2, xi3, kind, params, 10*k + 3)*a3_phys(x, y, z) )

    return value

#==============================================================================
def pull_2(a1_phys, a2_phys, a3_phys, k, xi1_vec, xi2_vec, xi3_vec, kind, params):
    
    '''
    Returns k-th component of pulled-back 2-form as 3D ndarray for integration.
    See 'pull_2_pw' for other arguments.
    '''
    
    if isinstance(xi1_vec, float):
        xi1_vec = array([xi1_vec])
        
    if isinstance(xi2_vec, float):
        xi2_vec = array([xi2_vec])
        
    if isinstance(xi3_vec, float):
        xi3_vec = array([xi3_vec])
    
    values = zeros( (xi1_vec.size, xi2_vec.size, xi3_vec.size), dtype='float' )
    
    for i1, xi1 in enumerate(xi1_vec):
        for i2, xi2 in enumerate(xi2_vec):
            for i3, xi3 in enumerate(xi3_vec):

                values[i1,i2,i3] = pull_2_pw(a1_phys, a2_phys, a3_phys, k, xi1, xi2, xi3, kind, params)

    return values

#==============================================================================
def pull_3_pw(a_phys, xi1, xi2, xi3, kind, params):
    
    '''
    Returns pullback of 3-form, point-wise, according to a(xi) = det(DF) * a_phys(F(xi)).
    
    Parameters
    ---------- 
        a_phys: callable
            3-form on physical domain, a_phys(x,y,z).
    
        xi1, xi2, xi3: float
            Coordinates in logical space.
            
    Returns
    -------
        value: float
    '''

    x = f(xi1, xi2, xi3, kind, params, 1)
    y = f(xi1, xi2, xi3, kind, params, 2)
    z = f(xi1, xi2, xi3, kind, params, 3)

    value = det_df(xi1, xi2, xi3, kind, params)*a_phys(x, y, z)

    return value

#==============================================================================
def pull_3(a_phys, xi1_vec, xi2_vec, xi3_vec, kind, params):
    
    '''
    Returns pulled-back 3-form as 3D ndarray for integration.
    See 'pull_3_pw' for other arguments.
    '''
    
    if isinstance(xi1_vec, float):
        xi1_vec = array([xi1_vec])
        
    if isinstance(xi2_vec, float):
        xi2_vec = array([xi2_vec])
        
    if isinstance(xi3_vec, float):
        xi3_vec = array([xi3_vec])
    
    values = zeros( (xi1_vec.size, xi2_vec.size, xi3_vec.size), dtype='float' )
    
    for i1, xi1 in enumerate(xi1_vec):
        for i2, xi2 in enumerate(xi2_vec):
            for i3, xi3 in enumerate(xi3_vec):

                values[i1,i2,i3] = pull_3_pw(a_phys, xi1, xi2, xi3, kind, params)

    return values

#==============================================================================
def push_v_pw(a1, a2, a3, k, xi1, xi2, xi3, kind, params):
    
    '''
    x, y, z, values = push_v_pw(..) returns in values the k-th component
    of pushed-forward vector field, point-wise, 
    according to a_phys(x) = DF * a(xi), and the point of evaluation x=F(xi).
    
    Parameters
    ----------
        a1, a2, a2: callable
            Components of vector-field on logical domain, a(xi1,xi2,xi3).
            
        k: int
            Which component a_phys_k(x) of pushed-forward vector-field to compute, k=1,2,3.
    
        xi1, xi2, xi3: float
            Coordinates in logical space.    
            
    Returns
    -------
        value: float
            Pushed-forward vector-field component.
        x, y, z: float
            Physical coordinates (x,y,z)=F(xi) 
    '''

    x = f(xi1, xi2, xi3, kind, params, 1)
    y = f(xi1, xi2, xi3, kind, params, 2)
    z = f(xi1, xi2, xi3, kind, params, 3)

    value = ( df(xi1, xi2, xi3, kind, params, 10*k + 1)*a1(xi1, xi2, xi3) +
              df(xi1, xi2, xi3, kind, params, 10*k + 2)*a2(xi1, xi2, xi3) +
              df(xi1, xi2, xi3, kind, params, 10*k + 3)*a3(xi1, xi2, xi3) )

    return x, y, z, value

#==============================================================================
def push_v(a1, a2, a3, k, xi1_vec, xi2_vec, xi3_vec, kind, params):
    
    '''
    x, y, z, values = push_v(..) returns in 'values' the k-th component 
    of pushed-forward vector field as 3D ndarray for plotting. 
    The points of evaluation x,y,z are returned as 3D ndarrays as in meshgrid. 
    See 'push_v_pw' for other arguments.
    '''
    
    if isinstance(xi1_vec, float):
        xi1_vec = array([xi1_vec])
       
    if isinstance(xi2_vec, float):
        xi2_vec = array([xi2_vec])
        
    if isinstance(xi3_vec, float):
        xi3_vec = array([xi3_vec])
    
    values = zeros( (xi1_vec.size, xi2_vec.size, xi3_vec.size), dtype='float' )
    x = values.copy()
    y = values.copy()
    z = values.copy()
    
    for i1, xi1 in enumerate(xi1_vec):
        for i2, xi2 in enumerate(xi2_vec):
            for i3, xi3 in enumerate(xi3_vec):

                i = i1, i2, i3
                
                x[i], y[i], z[i], values[i] = push_v_pw(a1, a2, a3, k, xi1, xi2, xi3, kind, params)

    return x, y, z, values

#==============================================================================
def push_0_pw(a, xi1, xi2, xi3, kind, params):
    
    '''
    Returns push-forward of 0-form, point-wise, according to a_phys(x) = a(xi), and x=F(xi).
    
    Parameters
    ----------  
        a: callable
            0-form on logical domain, a(xi1,xi2,xi3).
    
        xi1, xi2, xi3: float
            Coordinates in logical space.    
            
    Returns
    -------
        value: float
            Pushed-forward 0-form.
        x, y, z: float
            Physical coordinates (x,y,z)=F(xi) 
    '''

    x = f(xi1, xi2, xi3, kind, params, 1)
    y = f(xi1, xi2, xi3, kind, params, 2)
    z = f(xi1, xi2, xi3, kind, params, 3)

    value = a(xi1, xi2, xi3) 

    return x, y, z, value

#==============================================================================
def push_0(a, xi1_vec, xi2_vec, xi3_vec, kind, params):
    
    '''
    Returns pushed-forward 0-form as 3D ndarray for plotting. 
    The points of evaluation x,y,z are returned as 3D ndarrays as in meshgrid.
    See 'push_0_pw' for other arguments.
    '''
    
    if isinstance(xi1_vec, float):
        xi1_vec = array([xi1_vec])
        
    if isinstance(xi2_vec, float):
        xi2_vec = array([xi2_vec])
        
    if isinstance(xi3_vec, float):
        xi3_vec = array([xi3_vec])
    
    values = zeros( (xi1_vec.size, xi2_vec.size, xi3_vec.size), dtype='float' )
    x = values.copy()
    y = values.copy()
    z = values.copy()
    
    for i1, xi1 in enumerate(xi1_vec):
        for i2, xi2 in enumerate(xi2_vec):
            for i3, xi3 in enumerate(xi3_vec):

                i = i1, i2, i3
                
                x[i], y[i], z[i], values[i] = push_0_pw(a, xi1, xi2, xi3, kind, params)

    return x, y, z, values

#==============================================================================
def push_1_pw(a1, a2, a3, k, xi1, xi2, xi3, kind, params):
    
    '''
    Returns k-th component of pushed-forward 1-form, point-wise, according to a_phys(x) = DF^(-T) * a(xi), and x=F(xi).
   
    Parameters
    ----------  
        a1, a2, a2: callable
            Components of 1-form on logical domain, a(xi1,xi2,xi3).
            
        k: int
            Which component a_phys_k(x) of pushed-forward 1-form to compute, k=1,2,3.
    
        xi1, xi2, xi3: float
            Coordinates in logical space.    
            
    Returns
    -------
        value: float
            Pushed-forward 1-form component.
        x, y, z: float
            Physical coordinates (x,y,z)=F(xi) 
    '''

    x = f(xi1, xi2, xi3, kind, params, 1)
    y = f(xi1, xi2, xi3, kind, params, 2)
    z = f(xi1, xi2, xi3, kind, params, 3)

    value = ( df_inv(xi1, xi2, xi3, kind, params, 10 + k)*a1(xi1, xi2, xi3) +
              df_inv(xi1, xi2, xi3, kind, params, 20 + k)*a2(xi1, xi2, xi3) +
              df_inv(xi1, xi2, xi3, kind, params, 30 + k)*a3(xi1, xi2, xi3) )

    return x, y, z, value

#==============================================================================
def push_1(a1, a2, a3, k, xi1_vec, xi2_vec, xi3_vec, kind, params):
    
    '''
    Returns k-th component of pushed-forward 1-form as 3D ndarray for plotting. 
    The points of evaluation x,y,z are returned as 3D ndarrays as in meshgrid.
    See 'push_1_pw' for other arguments.
    '''
    
    if isinstance(xi1_vec, float):
        xi1_vec = array([xi1_vec])
        
    if isinstance(xi2_vec, float):
        xi2_vec = array([xi2_vec])
        
    if isinstance(xi3_vec, float):
        xi3_vec = array([xi3_vec])
    
    values = zeros( (xi1_vec.size, xi2_vec.size, xi3_vec.size), dtype='float' )
    x = values.copy()
    y = values.copy()
    z = values.copy()
    
    for i1, xi1 in enumerate(xi1_vec):
        for i2, xi2 in enumerate(xi2_vec):
            for i3, xi3 in enumerate(xi3_vec):

                i = i1, i2, i3
                
                x[i], y[i], z[i], values[i] = push_1_pw(a1, a2, a3, k, xi1, xi2, xi3, kind, params)

    return x, y, z, values

#==============================================================================
def push_2_pw(a1, a2, a3, k, xi1, xi2, xi3, kind, params):
    
    '''
    Returns k-th component of pushed-forward 2-form, point-wise, according to a_phys(x) = 1/det(DF) * DF * a(xi), and x=F(xi).
    
    Parameters
    ----------
        a1, a2, a2: callable
            Components of 2-form on logical domain, a(xi1,xi2,xi3).
            
        k: int
            Which component a_phys_k(x) of pushed-forward 2-form to compute, k=1,2,3.
    
        xi1, xi2, xi3: float
            Coordinates in logical space.    
            
    Returns
    -------
        value: float
            Pushed-forward 2-form component.
        x, y, z: float
            Physical coordinates (x,y,z)=F(xi) 
    '''

    x = f(xi1, xi2, xi3, kind, params, 1)
    y = f(xi1, xi2, xi3, kind, params, 2)
    z = f(xi1, xi2, xi3, kind, params, 3)

    value = ( df(xi1, xi2, xi3, kind, params, 10*k + 1)*a1(xi1, xi2, xi3) +
              df(xi1, xi2, xi3, kind, params, 10*k + 2)*a2(xi1, xi2, xi3) +
              df(xi1, xi2, xi3, kind, params, 10*k + 3)*a3(xi1, xi2, xi3) ) / det_df(xi1, xi2, xi3, kind, params)

    return x, y, z, value

#==============================================================================
def push_2(a1, a2, a3, k, xi1_vec, xi2_vec, xi3_vec, kind, params):
    
    '''
    Returns k-th component of pushed-forward 2-form as 3D ndarray for plotting. 
    The points of evaluation x,y,z are returned as 3D ndarrays as in meshgrid.
    See 'push_2_pw' for other arguments.
    '''
    
    if isinstance(xi1_vec, float):
        xi1_vec = array([xi1_vec])
        
    if isinstance(xi2_vec, float):
        xi2_vec = array([xi2_vec])
        
    if isinstance(xi3_vec, float):
        xi3_vec = array([xi3_vec])
    
    values = zeros( (xi1_vec.size, xi2_vec.size, xi3_vec.size), dtype='float' )
    x = values.copy()
    y = values.copy()
    z = values.copy()
    
    for i1, xi1 in enumerate(xi1_vec):
        for i2, xi2 in enumerate(xi2_vec):
            for i3, xi3 in enumerate(xi3_vec):

                i = i1, i2, i3
                
                x[i], y[i], z[i], values[i] = push_2_pw(a1, a2, a3, k, xi1, xi2, xi3, kind, params)

    return x, y, z, values

#==============================================================================
def push_3_pw(a, xi1, xi2, xi3, kind, params):
    
    '''
    Returns push-forward of 3-form, point-wise, according to a_phys(x) = 1/det(DF) * a(xi), and x=F(xi).
    
    Parameters
    ---------- 
        a: callable
            3-form on logical domain, a(xi1,xi2,xi3).
    
        xi1, xi2, xi3: float
            Coordinates in logical space.    
            
    Returns
    -------
        value: float
            Pushed-forward 3-form.
        x, y, z: float
            Physical coordinates (x,y,z)=F(xi) 
    '''

    x = f(xi1, xi2, xi3, kind, params, 1)
    y = f(xi1, xi2, xi3, kind, params, 2)
    z = f(xi1, xi2, xi3, kind, params, 3)

    value = a(xi1, xi2, xi3) / det_df(xi1, xi2, xi3, kind, params)

    return x, y, z, value

#==============================================================================
def push_3(a, xi1_vec, xi2_vec, xi3_vec, kind, params):
    
    '''
    Returns pushed-forward 3-form as 3D ndarray for plotting. 
    The points of evaluation x,y,z are returned as 3D ndarrays as in meshgrid.
    See 'push_3_pw' for other arguments.
    '''
    
    if isinstance(xi1_vec, float):
        xi1_vec = array([xi1_vec])
        
    if isinstance(xi2_vec, float):
        xi2_vec = array([xi2_vec])
        
    if isinstance(xi3_vec, float):
        xi3_vec = array([xi3_vec])
    
    values = zeros( (xi1_vec.size, xi2_vec.size, xi3_vec.size), dtype='float' )
    x = values.copy()
    y = values.copy()
    z = values.copy()
    
    for i1, xi1 in enumerate(xi1_vec):
        for i2, xi2 in enumerate(xi2_vec):
            for i3, xi3 in enumerate(xi3_vec):

                i = i1, i2, i3
                
                x[i], y[i], z[i], values[i] = push_3_pw(a, xi1, xi2, xi3, kind, params)

    return x, y, z, values

