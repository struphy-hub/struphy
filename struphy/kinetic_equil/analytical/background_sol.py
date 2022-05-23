"""
    Pyccel-ised function to return values of the background solution (Maxwellian, ...)
"""


# ==========================================================================================================
def maxwellian_point(v : 'float[:]', v0 : 'float[:]', vth : 'float[:]', n0 : 'float') -> float:
    """
    Takes single 3-vector for v, v0, v_th and evaluates the Maxwellian function homogeneous in x with constant density in space nh0

    Parameters :
    ------------
        v : np.array
            Shape(3,), contains velocity values for each particle
        
        v0 : np.array
            Shape(3,), contains values of shift velocity
        
        vth : np.array
            Shape(3,), contains values of thermal velocity
        
        n0 : Integer
            density-value (homogeneous in x)
    """

    from  numpy import exp, sqrt, pi

    vx = v[0]
    vy = v[1]
    vz = v[2]

    v0x = v0[0]
    v0y = v0[1]
    v0z = v0[2]

    vth2 = vth[0]**2 + vth[1]**2 + vth[2]**2

    Gx = exp( -(vx - v0x)**2 / (2*vth2) ) / ( sqrt(2*pi*vth2) )
    Gy = exp( -(vy - v0y)**2 / (2*vth2) ) / ( sqrt(2*pi*vth2) )
    Gz = exp( -(vz - v0z)**2 / (2*vth2) ) / ( sqrt(2*pi*vth2) )

    return n0*Gx*Gy*Gz
