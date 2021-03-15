from pyccel.decorators import types

import numpy as np

import hylife.geometry.mappings_analytical as mp_a

__all__ = ['f_ini_phys',
          'pull_v',
          'pull_0'
          'pull_1'
          'pull_2',
          'pull_3']

# =======================================================================
@types('double','double','double','int','double[:]','int')
def f_ini_phys(xi1, xi2, xi3, kind, params, component):
    '''
    defines an analytical function X = f(xi) in three dimensionsional physical space,
    which is supposed to serve as initial condition for a simulation.
    X=(X1,X2,X3), xi=(xi1,xi2,xi3)
    
    kind      : 1 (plane wave), 2 (sinusoidal linear form), 3 (?) 
    
    params    : plane wave            --> kx, ky, kz, ax, ay ,az (components of wave vector, amplitude of each component)
              : hollow cylinder --> R1, R2, Lz
              : colella         --> Lx, Ly, alpha, Lz
                 
    component : 1 (x), 2 (y), 3 (z)
    '''
               
    if kind == 1:
         
        kx = params[0] 
        ky = params[1] 
        kz = params[2]
        
        ax = params[3] 
        ay = params[4] 
        az = params[5]
        
        if   component == 1:
            value = ax*np.cos(2*np.pi*(kx*xi1 + ky*xi2 + kz*xi3))
        elif component == 2:
            value = ay*np.cos(2*np.pi*(kx*xi1 + ky*xi2 + kz*xi3))
        elif component == 3:
            value = az*np.cos(2*np.pi*(kx*xi1 + ky*xi2 + kz*xi3))
           
    else:
        #raise ValueError("kind of mapping unknown")
        value = -99999999.
            
    return value


# =======================================================================
@types('double','double','double','int','double[:]','int','double[:]','int')
def pull_v(xi1, xi2, xi3, kind_f, params_f, kind_input, params_input, k):
    
    '''
    Returns k-th component of pulled-back vector field, point-wise, according to a(xi) = DF^(-1) * a_phys(F(xi)).
    
    Parameters
    ----------  
            
    
        xi1, xi2, xi3: float
            Coordinates in logical space.    
            
        kind_f: int
            Type of mapping
            
        params_f: double[:]
            parameters for the mapping
            
        kind_input: int
            Type of input function
            
        params_input: double[:]
            parameters for input function
            
        k: int
            Which component a_k(xi) of pulled-back vector-field to compute, k=1,2,3.
            
    Returns
    -------
        value: float
    '''

    x = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 1)
    y = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 2)
    z = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 3)
    
    a1_phys = f_ini_phys(x, y, z, kind_input, params_input, 1)
    a2_phys = f_ini_phys(x, y, z, kind_input, params_input, 2)
    a3_phys = f_ini_phys(x, y, z, kind_input, params_input, 3)

    value = ( mp_a.df_inv(xi1, xi2, xi3, kind_f, params_f, 10*k + 1)*a1_phys +
              mp_a.df_inv(xi1, xi2, xi3, kind_f, params_f, 10*k + 2)*a2_phys +
              mp_a.df_inv(xi1, xi2, xi3, kind_f, params_f, 10*k + 3)*a3_phys )

    return value


#==============================================================================
@types('double','double','double','int','double[:]','int','double[:]')
def pull_0(xi1, xi2, xi3, kind_f, params_f, kind_input, params_input):
    
    '''
    Returns pullback of 0-form, point-wise, according to a(xi) = a_phys(F(xi)).
    
    Parameters
    ---------- 
    
        xi1, xi2, xi3: float
            Coordinates in logical space.
        
        kind_f: int
            Type of mapping
            
        params_f: double[:]
            parameters for the mapping
            
        kind_input: int
            Type of input function
            
        params_input: double[:]
            parameters for input function
            
    Returns
    -------
        value: float
    '''

    x = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 1)
    y = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 2)
    z = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 3)

    value = f_ini_phys(x, y, z, kind_input, params_input, 1)

    return value


#==============================================================================
@types('double','double','double','int','double[:]','int','double[:]','int')
def pull_1(xi1, xi2, xi3, kind_f, params_f, kind_input, params_input, k):
    
    '''
    Returns k-th component of pulled-back 1-form, point-wise, according to a(xi) = DF^T * a_phys(F(xi)).
    
    Parameters
    ---------- 
        
    
        xi1, xi2, xi3: float
            Coordinates in logical space.
            
        kind_f: int
            Type of mapping
            
        params_f: double[:]
            parameters for the mapping
            
        kind_input: int
            Type of input function
            
        params_input: double[:]
            parameters for input function
            
        k: int
            Which component a_k(xi) of pulled-back 1-form to compute, k=1,2,3.
            
    Returns
    -------
        value: float
    '''

    x = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 1)
    y = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 2)
    z = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 3)
    
    a1_phys = f_ini_phys(x, y, z, kind_input, params_input, 1)
    a2_phys = f_ini_phys(x, y, z, kind_input, params_input, 2)
    a3_phys = f_ini_phys(x, y, z, kind_input, params_input, 3)
    
    value = ( mp_a.df(xi1, xi2, xi3, kind_f, params_f, 10 + k)*a1_phys +
              mp_a.df(xi1, xi2, xi3, kind_f, params_f, 20 + k)*a2_phys +
              mp_a.df(xi1, xi2, xi3, kind_f, params_f, 30 + k)*a3_phys )

    return value


#==============================================================================
@types('double','double','double','int','double[:]','int','double[:]','int')
def pull_2(xi1, xi2, xi3, kind_f, params_f, kind_input, params_input, k):
    
    '''
    Returns k-th component of pulled-back 2-form, point-wise, according to a(xi) = det(DF) * DF^(-1) * a_phys(F(xi)).
    
    Parameters
    ---------- 
            
    
        xi1, xi2, xi3: float
            Coordinates in logical space.  
            
        kind_f: int
            Type of mapping
            
        params_f: double[:]
            parameters for the mapping
            
        kind_input: int
            Type of input function
            
        params_input: double[:]
            parameters for input function
                
         k: int
            Which component a_k(xi) of pulled-back 2-form to compute, k=1,2,3.
            
    Returns
    -------
        value: float
    '''

    x = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 1)
    y = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 2)
    z = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 3)
    
    a1_phys = f_ini_phys(x, y, z, kind_input, params_input, 1)
    a2_phys = f_ini_phys(x, y, z, kind_input, params_input, 2)
    a3_phys = f_ini_phys(x, y, z, kind_input, params_input, 3)
    
    value = mp_a.det_df(xi1, xi2, xi3, kind_f, params_f)*( mp_a.df_inv(xi1, xi2, xi3, kind_f, params_f, 10*k + 1)*a1_phys +
                                                           mp_a.df_inv(xi1, xi2, xi3, kind_f, params_f, 10*k + 2)*a2_phys +
                                                           mp_a.df_inv(xi1, xi2, xi3, kind_f, params_f, 10*k + 3)*a3_phys )



    return value

#==============================================================================
@types('double','double','double','int','double[:]','int','double[:]')
def pull_3(xi1, xi2, xi3, kind_f, params_f, kind_input, params_input):
    
    '''
    Returns pullback of 3-form, point-wise, according to a(xi) = det(DF) * a_phys(F(xi)).
    
    Parameters
    ---------- 
    
        xi1, xi2, xi3: float
            Coordinates in logical space.
            
        kind_f: int
            Type of mapping
            
        params_f: double[:]
            parameters for the mapping
            
        kind_input: int
            Type of input function
            
        params_input: double[:]
            parameters for input function
                  
            
    Returns
    -------
        value: float
    '''

    x = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 1)
    y = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 2)
    z = mp_a.f(xi1, xi2, xi3, kind_f, params_f, 3)
    
    
    value = mp_a.det_df(xi1, xi2, xi3, kind_f, params_f)*f_ini_phys(x, y, z, kind_input, params_input, 1)

    return value
#==============================================================================