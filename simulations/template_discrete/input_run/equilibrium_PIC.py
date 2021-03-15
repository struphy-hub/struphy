from pyccel.decorators import types

import hylife.geometry.mappings_discrete as mapping
from numpy import exp, pi

# ============================ physical domain ===========================================================

# ======= equilibrium distribution function (used in delta-f method) =============
@types('double','double','double','double','double','double')
def fh_eq_phys(x, y, z, vx, vy, vz):
    
    v0x = 2.5
    v0y = 0.
    v0z = 0.
    
    vth = 1.
    
    nh0 = 0.05
    
    arg = -(vx - v0x)**2/vth**2 - (vy - v0y)**2/vth**2 - (vz - v0z)**2/vth**2
    
    value = nh0/(pi**(3/2)*vth**3) * exp(arg)
    
    return value


# ============= 0-th moment of equilibrium distribution function fh_eq ===========
@types('double','double','double')
def nh_eq_phys(x, y, z):
    
    nh0 = 0.05
    
    return nh0

# ============= 1-st moment of equilibrium distribution function fh_eq ===========

# x - component
@types('double','double','double')
def jhx_eq(x, y, z):
    
    nh0 = 0.05
    v0x = 2.5
    
    return nh0 * v0x

# y - component
@types('double','double','double')
def jhy_eq(x, y, z):
    
    nh0 = 0.05
    v0y = 0.
    
    return nh0 * v0y

# z - component
@types('double','double','double')
def jhz_eq(x, y, z):
    
    nh0 = 0.05
    v0z = 0.
    
    return nh0 * v0z

# ============= energy of equilibrium distribution function fh_eq ===============
@types('int','double[:]')
def eh_eq(kind_map, params_map):
    
    v0x = 2.5
    v0y = 0.
    v0z = 0.
    
    vth = 1.
    
    nh0 = 0.05
    
    if   kind_map == 1:
        value = nh0/2 * params_map[0] * params_map[1] * params_map[2] * (v0x**2 + v0y**2 + v0z**2 + 3*vth**2/2)
        
    elif kind_map == 3:
        value = nh0/2 * params_map[0] * params_map[1] * params_map[3] * (v0x**2 + v0y**2 + v0z**2 + 3*vth**2/2)
        
    elif kind_map == 4:
        value = nh0/2 * params_map[0] * params_map[1] * params_map[3] * (v0x**2 + v0y**2 + v0z**2 + 3*vth**2/2)
        
    else:
        value = 0.
    
    return value


# ============================ logical domain ===========================================================

# ======= equilibrium distribution function (used in delta-f method) =============
@types('double','double','double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def fh_eq(eta1, eta2, eta3, vx, vy, vz, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    return fh_eq_phys(x, y, z, vx, vy, vz)
