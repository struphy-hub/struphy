import hylife.geometry.mappings_analytical as mp_a

#==============================================================================
def pull_v(a1_phys, a2_phys, a3_phys, k, xi1, xi2, xi3, kind, params):
    
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

    x = mp_a.f(xi1, xi2, xi3, kind, params, 1)
    y = mp_a.f(xi1, xi2, xi3, kind, params, 2)
    z = mp_a.f(xi1, xi2, xi3, kind, params, 3)

    value = ( df_inv(xi1, xi2, xi3, kind, params, 10*k + 1)*a1_phys(x, y, z) +
              df_inv(xi1, xi2, xi3, kind, params, 10*k + 2)*a2_phys(x, y, z) +
              df_inv(xi1, xi2, xi3, kind, params, 10*k + 3)*a3_phys(x, y, z) )

    return value

#==============================================================================
def pull_0(a_phys, xi1, xi2, xi3, kind, params):
    
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

    x = mp_a.f(xi1, xi2, xi3, kind, params, 1)
    y = mp_a.f(xi1, xi2, xi3, kind, params, 2)
    z = mp_a.f(xi1, xi2, xi3, kind, params, 3)

    value = a_phys(x, y, z)

    return value


#==============================================================================
def pull_1(a1_phys, a2_phys, a3_phys, k, xi1, xi2, xi3, kind, params):
    
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

    x = mp_a.f(xi1, xi2, xi3, kind, params, 1)
    y = mp_a.f(xi1, xi2, xi3, kind, params, 2)
    z = mp_a.f(xi1, xi2, xi3, kind, params, 3)

    value = (  mp_a.df(xi1, xi2, xi3, kind, params, 10 + k)*a1_phys(x, y, z) +
               mp_a.df(xi1, xi2, xi3, kind, params, 20 + k)*a2_phys(x, y, z) +
               mp_a.df(xi1, xi2, xi3, kind, params, 30 + k)*a3_phys(x, y, z) )

    return value

#==============================================================================
def pull_2(a1_phys, a2_phys, a3_phys, k, xi1, xi2, xi3, kind, params):
    
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

    x = mp_a.f(xi1, xi2, xi3, kind, params, 1)
    y = mp_a.f(xi1, xi2, xi3, kind, params, 2)
    z = mp_a.f(xi1, xi2, xi3, kind, params, 3)

    value = det_df(xi1, xi2, xi3, kind, params)*( mp_a.df_inv(xi1, xi2, xi3, kind, params, 10*k + 1)*a1_phys(x, y, z) +
                                                  mp_a.df_inv(xi1, xi2, xi3, kind, params, 10*k + 2)*a2_phys(x, y, z) +
                                                  mp_a.df_inv(xi1, xi2, xi3, kind, params, 10*k + 3)*a3_phys(x, y, z) )

    return value


#==============================================================================
def pull_3(a_phys, xi1, xi2, xi3, kind, params):
    
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

    x = mp_a.f(xi1, xi2, xi3, kind, params, 1)
    y = mp_a.f(xi1, xi2, xi3, kind, params, 2)
    z = mp_a.f(xi1, xi2, xi3, kind, params, 3)

    value = mp_a.det_df(xi1, xi2, xi3, kind, params)*a_phys(x, y, z)

    return value


#==============================================================================
def push_v(a1, a2, a3, k, xi1, xi2, xi3, kind, params):
    
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

    x = mp_a.f(xi1, xi2, xi3, kind, params, 1)
    y = mp_a.f(xi1, xi2, xi3, kind, params, 2)
    z = mp_a.f(xi1, xi2, xi3, kind, params, 3)

    value = ( mp_a.df(xi1, xi2, xi3, kind, params, 10*k + 1)*a1(xi1, xi2, xi3) +
              mp_a.df(xi1, xi2, xi3, kind, params, 10*k + 2)*a2(xi1, xi2, xi3) +
              mp_a.df(xi1, xi2, xi3, kind, params, 10*k + 3)*a3(xi1, xi2, xi3) )

    return x, y, z, value


#==============================================================================
def push_0(a, xi1, xi2, xi3, kind, params):
    
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

    x = mp_a.f(xi1, xi2, xi3, kind, params, 1)
    y = mp_a.f(xi1, xi2, xi3, kind, params, 2)
    z = mp_a.f(xi1, xi2, xi3, kind, params, 3)

    value = a(xi1, xi2, xi3) 

    return x, y, z, value


#==============================================================================
def push_1(a1, a2, a3, k, xi1, xi2, xi3, kind, params):
    
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

    x = mp_a.f(xi1, xi2, xi3, kind, params, 1)
    y = mp_a.f(xi1, xi2, xi3, kind, params, 2)
    z = mp_a.f(xi1, xi2, xi3, kind, params, 3)

    value = ( mp_a.df_inv(xi1, xi2, xi3, kind, params, 10 + k)*a1(xi1, xi2, xi3) +
              mp_a.df_inv(xi1, xi2, xi3, kind, params, 20 + k)*a2(xi1, xi2, xi3) +
              mp_a.df_inv(xi1, xi2, xi3, kind, params, 30 + k)*a3(xi1, xi2, xi3) )

    return value


#==============================================================================
def push_2(a1, a2, a3, k, xi1, xi2, xi3, kind, params):
    
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

    x = mp_a.f(xi1, xi2, xi3, kind, params, 1)
    y = mp_a.f(xi1, xi2, xi3, kind, params, 2)
    z = mp_a.f(xi1, xi2, xi3, kind, params, 3)

    value = ( mp_a.df(xi1, xi2, xi3, kind, params, 10*k + 1)*a1(xi1, xi2, xi3) +
              mp_a.df(xi1, xi2, xi3, kind, params, 10*k + 2)*a2(xi1, xi2, xi3) +
              mp_a.df(xi1, xi2, xi3, kind, params, 10*k + 3)*a3(xi1, xi2, xi3) ) / mp_a.det_df(xi1, xi2, xi3, kind, params)

    return value

#==============================================================================
def push_3(a, xi1, xi2, xi3, kind, params):
    
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

    x = mp_a.f(xi1, xi2, xi3, kind, params, 1)
    y = mp_a.f(xi1, xi2, xi3, kind, params, 2)
    z = mp_a.f(xi1, xi2, xi3, kind, params, 3)

    value = a(xi1, xi2, xi3) / mp_a.det_df(xi1, xi2, xi3, kind, params)

    return x, y, z, value
#==============================================================================