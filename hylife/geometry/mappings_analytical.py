from pyccel.decorators import types
from numpy             import sin, cos, pi


# =======================================================================
@types('double','double','double','int','double[:]','int')
def f(xi1, xi2, xi3, kind, params, component):
    '''
    defines an analytical mapping x = f(xi) in three dimensions. 
    
    kind      : 1 (slab), 2 (hollow cylinder), 3 (colella) 
    
    params    : slab            --> Lx, Ly, Lz
              : hollow cylinder --> R1, R2, Lz
              : colella         --> Lx, Ly, alpha, Lz
                 
    component : 1 (x), 2 (y), 3 (z)
    '''
    
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
            value = (xi1 * dR + R1) * cos(2*pi*xi2)
        elif component == 2:
            value = (xi1 * dR + R1) * sin(2*pi*xi2)
        elif component == 3:
            value = Lz * xi3
            
    elif kind == 3:
        
        Lx    = params[0]
        Ly    = params[1]
        alpha = params[2]
        Lz    = params[3]
        
        if   component == 1:
            value = Lx * (xi1 + alpha * sin(2*pi*xi1) * sin(2*pi*xi2))
        elif component == 2:
            value = Ly * (xi2 + alpha * sin(2*pi*xi1) * sin(2*pi*xi2))
        elif component == 3:
            value = Lz * xi3
    else
        raise ValueError("kind of mapping unknown")
            
    return value


# =======================================================================
@types('double','double','double','int','double[:]','int')
def df(xi1, xi2, xi3, kind, params, component):
    '''
    returns the components of the Jacobian matrix of an analytical mapping x = f(xi) in three dimensions. 
    
    kind      : 1 (slab), 2 (hollow cylinder), 3 (colella) 
    
    params    : slab            --> Lx, Ly, Lz
              : hollow cylinder --> R1, R2, Lz
              : colella         --> Lx, Ly, alpha, Lz
                 
    component : 11 (dfx/dxi1), 12 (dfx/dxi2), 13 (dfx/dxi3)
                21 (dfy/dxi1), 22 (dfy/dxi2), 23 (dfy/dxi3)
                31 (dfz/dxi1), 32 (dfz/dxi2), 33 (dfz/dxi3)
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
        else
            raise ValueError("df component not correct")
            
    elif kind == 2:
        
        R1 = params[0]
        R2 = params[1]
        Lz = params[2]
        dR = R2 - R1
        
        if   component == 11:
            value = dR * cos(2*pi*xi2)
        elif component == 12:
            value = -2*pi * (xi1*dR + R1) * sin(2*pi*xi2)
        elif component == 13:
            value = 0.
        elif component == 21:
            value = dR * sin(2*pi*xi2)
        elif component == 22:
            value =  2*pi * (xi1*dR + R1) * cos(2*pi*xi2)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = Lz
        else
            raise ValueError("df component not correct")
            
    elif kind == 3:
        
        Lx    = params[0]
        Ly    = params[1]
        alpha = params[2]
        Lz    = params[3]
        
        if   component == 11:
            value = Lx * (1 + alpha * cos(2*pi*xi1) * sin(2*pi*xi2) * 2*pi)
        elif component == 12:
            value = Lx * alpha * sin(2*pi*xi1) * cos(2*pi*xi2) * 2*pi
        elif component == 13:
            value = 0.
        elif component == 21:
            value = Ly * alpha * cos(2*pi*xi1) * sin(2*pi*xi2) * 2*pi
        elif component == 22:
            value = Ly * (1 + alpha * sin(2*pi*xi1) * cos(2*pi*xi2) * 2*pi)
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = Lz
        else
            raise ValueError("df component not correct")
    else
        raise ValueError("kind of mapping unknown")
            
    return value




# =======================================================================
@types('double','double','double','int','double[:]')
def det_df(xi1, xi2, xi3, kind, params):
    '''
    returns the jacobian determinant of an analytical mapping x = f(xi) in three dimensions. 
    
    kind      : 1 (slab), 2 (hollow cylinder), 3 (colella) 
    
    params    : slab            --> Lx, Ly, Lz
              : hollow cylinder --> R1, R2, Lz
              : colella         --> Lx, Ly, alpha, Lz
    '''
    
    value = 0.
               
    if kind == 1:
         
        Lx = params[0] 
        Ly = params[1] 
        Lz = params[2]
        
        value = Lx * Ly * Lz
            
    elif kind == 2:
        
        R1 = params[0]
        R2 = params[1]
        Lz = params[2]
        dR = R2 - R1
        
        value = dR * Lz * 2*pi * (xi1*dR + R1)
            
    elif kind == 3:
        
        Lx    = params[0]
        Ly    = params[1]
        alpha = params[2]
        Lz    = params[3]
        
        value = Lx*Ly*Lz * (1 + alpha * cos(2*pi*xi1) * sin(2*pi*xi2) * 2*pi + alpha * sin(2*pi*xi1) * cos(2*pi*xi2) * 2*pi)
    else
        raise ValueError("kind of mapping unknown")
            
    return value



# =======================================================================
@types('double','double','double','int','double[:]','int')
def df_inv(xi1, xi2, xi3, kind, params, component):
    '''
    returns the components of the Jacobian matrix of an analytical mapping x = f(xi) in three dimensions. 
    
    kind      : 1 (slab), 2 (hollow cylinder), 3 (colella) 
    
    params    : slab            --> Lx, Ly, Lz
              : hollow cylinder --> R1, R2, Lz
              : colella         --> Lx, Ly, alpha, Lz
                 
    component : 11 (dfx/dxi1), 12 (dfx/dxi2), 13 (dfx/dxi3)
                21 (dfy/dxi1), 22 (dfy/dxi2), 23 (dfy/dxi3)
                31 (dfz/dxi1), 32 (dfz/dxi2), 33 (dfz/dxi3)
    '''
    
    value = 0.
               
    if kind == 1:
         
        Lx = params[0] 
        Ly = params[1] 
        Lz = params[2]
        
        if   component == 11:
            value = Ly * Lz
        elif component == 12:
            value = 0.
        elif component == 13:
            value = 0.
        elif component == 21:
            calue = 0.
        elif component == 22:
            value = Lx * Lz
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = Lx * Ly
        else
            raise ValueError("df_inv component not correct")
            
    elif kind == 2:
        
        R1 = params[0]
        R2 = params[1]
        Lz = params[2]
        dR = R2 - R1
        
        if   component == 11:
            value = 2*pi * (xi1*dR + R1) * cos(2*pi*xi2) * Lz
        elif component == 12:
            value = 2*pi * (xi1*dR + R1) * sin(2*pi*xi2) * Lz
        elif component == 13:
            value = 0.
        elif component == 21:
            value = -dR * sin(2*pi*xi2) * Lz
        elif component == 22:
            value =  dR * cos(2*pi*xi2) * Lz
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = dR * 2*pi * (xi1*dR + R1)
        else
            raise ValueError("df_inv component not correct")
            
    elif kind == 3:
        
        Lx    = params[0]
        Ly    = params[1]
        alpha = params[2]
        Lz    = params[3]
        
        if   component == 11:
            value = Ly * (1 + alpha * sin(2*pi*xi1) * cos(2*pi*xi2) * 2*pi) * Lz
        elif component == 12:
            value = -Lx * alpha * sin(2*pi*xi1) * cos(2*pi*xi2) * 2*pi * Lz
        elif component == 13:
            value = 0.
        elif component == 21:
            value = -Ly * alpha * cos(2*pi*xi1) * sin(2*pi*xi2) * 2*pi * Lz
        elif component == 22:
            value = Lx * (1 + alpha * cos(2*pi*xi1) * sin(2*pi*xi2) * 2*pi) * Lz
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = Lx*Ly * (1 + alpha * cos(2*pi*xi1) * sin(2*pi*xi2) * 2*pi + alpha * sin(2*pi*xi1) * cos(2*pi*xi2) * 2*pi)
        else
            raise ValueError("df_inv component not correct")
            
    else
        raise ValueError("kind of mapping unknown")
            
    return value/det_df(xi1, xi2, xi3, kind, params)


# =======================================================================
@types('double','double','double','int','double[:]','int')
def g(xi1, xi2, xi3, kind, params, component):
    '''
    returns the components of the metric tensor (df)^T*df of an analytical mapping x = f(xi) in three dimensions. 
    
    kind      : 1 (slab), 2 (hollow cylinder), 3 (colella) 
    
    params    : slab            --> Lx, Ly, Lz
              : hollow cylinder --> R1, R2, Lz
              : colella         --> Lx, Ly, alpha, Lz
                 
    component : 11 (dfx/dxi1), 12 (dfx/dxi2), 13 (dfx/dxi3)
                21 (dfy/dxi1), 22 (dfy/dxi2), 23 (dfy/dxi3)
                31 (dfz/dxi1), 32 (dfz/dxi2), 33 (dfz/dxi3)
    '''
    
    value = 0.
               
    if kind == 1:
         
        Lx = params[0] 
        Ly = params[1] 
        Lz = params[2]
        
        if   component == 11:
            value = Lx**2
        elif component == 12:
            value = 0.
        elif component == 13:
            value = 0.
        elif component == 21:
            calue = 0.
        elif component == 22:
            value = Ly**2
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = Lz**2
        else
            raise ValueError("g component not correct")
            
    elif kind == 2:
        
        R1 = params[0]
        R2 = params[1]
        Lz = params[2]
        dR = R2 - R1
        
        if   component == 11:
            value = dR**2
        elif component == 12:
            value = 0.
        elif component == 13:
            value = 0.
        elif component == 21:
            value = 0.
        elif component == 22:
            value = (2*pi)**2 * (xi1*dR + R1)**2
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = Lz**2
        else
            raise ValueError("g component not correct")
            
    elif kind == 3:
        
        Lx    = params[0]
        Ly    = params[1]
        alpha = params[2]
        Lz    = params[3]
        
        if   component == 11:
            value = Lx**2 * (1 + alpha * cos(2*pi*xi1) * sin(2*pi*xi2) * 2*pi)**2 + Ly**2 * alpha**2 * cos(2*pi*xi1)**2 * sin(2*pi*xi2)**2 * (2*pi)**2
        elif component == 12:
            value = (Lx**2 + Ly**2) * alpha**2 * cos(2*pi*xi1) * sin(2*pi*xi2) * sin(2*pi*xi1) * cos(2*pi*xi2)* (2*pi)**2 + Lx**2 * alpha * sin(2*pi*xi1) * cos(2*pi*xi2) * 2*pi + Ly**2 * alpha * cos(2*pi*xi1) * sin(2*pi*xi2) * 2*pi 
        elif component == 13:
            value = 0.
        elif component == 21:
            value = (Lx**2 + Ly**2) * alpha**2 * cos(2*pi*xi1) * sin(2*pi*xi2) * sin(2*pi*xi1) * cos(2*pi*xi2)* (2*pi)**2 + Lx**2 * alpha * sin(2*pi*xi1) * cos(2*pi*xi2) * 2*pi + Ly**2 * alpha * cos(2*pi*xi1) * sin(2*pi*xi2) * 2*pi
        elif component == 22:
            value = Ly**2 * (1 + alpha * sin(2*pi*xi1) * cos(2*pi*xi2) * 2*pi)**2 + Lx**2 * alpha**2 * sin(2*pi*xi1)**2 * cos(2*pi*xi2)**2 * (2*pi)**2
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = Lz**2
        else
            raise ValueError("g component not correct")
    else
        raise ValueError("kind of mapping unknown")
        
            
    return value


# =======================================================================
@types('double','double','double','int','double[:]','int')
def g_inv(xi1, xi2, xi3, kind, params, component):
    '''
    returns the components of the inverse metric tensor (df)^(-1)*df^(-T) of an analytical mapping x = f(xi) in three dimensions. 
    
    kind      : 1 (slab), 2 (hollow cylinder), 3 (colella) 
    
    params    : slab            --> Lx, Ly, Lz
              : hollow cylinder --> R1, R2, Lz
              : colella         --> Lx, Ly, alpha, Lz
                 
    component : 11 (dfx/dxi1), 12 (dfx/dxi2), 13 (dfx/dxi3)
                21 (dfy/dxi1), 22 (dfy/dxi2), 23 (dfy/dxi3)
                31 (dfz/dxi1), 32 (dfz/dxi2), 33 (dfz/dxi3)
    '''
    
    value = 0.
               
    if kind == 1:
         
        Lx = params[0] 
        Ly = params[1] 
        Lz = params[2]
        
        if   component == 11:
            value = Ly**2 * Lz**2
        elif component == 12:
            value = 0.
        elif component == 13:
            value = 0.
        elif component == 21:
            calue = 0.
        elif component == 22:
            value = Lx**2 * Lz**2
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = Lx**2 * Ly**2
        else
            raise ValueError("g_inv component not correct")
            
    elif kind == 2:
        
        R1 = params[0]
        R2 = params[1]
        Lz = params[2]
        dR = R2 - R1
        
        if   component == 11:
            value = (2*pi)**2 * (xi1*dR + R1)**2 * Lz**2
        elif component == 12:
            value = 0.
        elif component == 13:
            value = 0.
        elif component == 21:
            value = 0.
        elif component == 22:
            value = dR**2 * Lz**2
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.
        elif component == 33:
            value = dR**2 * (2*pi)**2 * (xi1*dR + R1)**2
        else
            raise ValueError("g_inv component not correct")
            
    elif kind == 3:
        
        Lx    = params[0]
        Ly    = params[1]
        alpha = params[2]
        Lz    = params[3]
        
        if   component == 11:
            value = Ly**2 * Lz**2 * (1 + alpha * sin(2*pi*xi1) * cos(2*pi*xi2) * 2*pi)**2 + Lx**2 * Lz**2 * alpha**2 * sin(2*pi*xi1)**2 * cos(2*pi*xi2)**2 * (2*pi)**2
        elif component == 12:
            value = -((Lx**2 + Ly**2) * alpha**2 * cos(2*pi*xi1) * sin(2*pi*xi2) * sin(2*pi*xi1) * cos(2*pi*xi2)* (2*pi)**2 + Lx**2 * alpha * sin(2*pi*xi1) * cos(2*pi*xi2) * 2*pi + Ly**2 * alpha * cos(2*pi*xi1) * sin(2*pi*xi2) * 2*pi) * Lz**2 
        elif component == 13:
            value = 0.
        elif component == 21:
            value = -((Lx**2 + Ly**2) * alpha**2 * cos(2*pi*xi1) * sin(2*pi*xi2) * sin(2*pi*xi1) * cos(2*pi*xi2)* (2*pi)**2 + Lx**2 * alpha * sin(2*pi*xi1) * cos(2*pi*xi2) * 2*pi + Ly**2 * alpha * cos(2*pi*xi1) * sin(2*pi*xi2) * 2*pi) * Lz**2
        elif component == 22:
            value = Lx**2 * Lz**2 * (1 + alpha * cos(2*pi*xi1) * sin(2*pi*xi2) * 2*pi)**2 + Ly**2 * Lz**2 * alpha**2 * cos(2*pi*xi1)**2 * sin(2*pi*xi2)**2 * (2*pi)**2
        elif component == 23:
            value = 0.
        elif component == 31:
            value = 0.
        elif component == 32:
            value = 0.    
        elif component == 33:
            value = (Lx*Ly * (1 + alpha * cos(2*pi*xi1) * sin(2*pi*xi2) * 2*pi + alpha * sin(2*pi*xi1) * cos(2*pi*xi2) * 2*pi))**2
        else
            raise ValueError("g_inv component not correct")
            
    else
        raise ValueError("kind of mapping unknown")
          
    return value/det_df(xi1, xi2, xi3, kind, params)**2