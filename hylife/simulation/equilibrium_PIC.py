from pyccel.decorators import types


# ============= 0-th moment of equilibrium distribution function fh0 =============
@types('double','double','double')
def n_eq(x, y, z):
    
    nh0 = 0.05
    
    return nh0

# ============= 1-st moment of equilibrium distribution function fh0 =============

# x - component
@types('double','double','double')
def jhx_eq(x, y, z):
    
    nh0 = 0.05
    vx0 = 2.5
    
    return nh0 * vx0

# y - component
@types('double','double','double')
def jhy_eq(x, y, z):
    
    nh0 = 0.05
    vy0 = 0.
    
    return nh0 * vy0

# z - component
@types('double','double','double')
def jhz_eq(x, y, z):
    
    nh0 = 0.05
    vz0 = 0.
    
    return nh0 * vz0