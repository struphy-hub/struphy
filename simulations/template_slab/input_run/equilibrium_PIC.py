from pyccel.decorators import types

import hylife.geometry.mappings_3d as mapping
import hylife.geometry.pullback_3d as pull

from numpy import exp, pi

# ===============================================================
#                       physical domain
# ===============================================================

# ======= equilibrium distribution function (used in delta-f method) =============
@types('double','double','double','double','double','double')
def fh_eq_phys(x, y, z, vx, vy, vz):
    
    v0x = 0.
    v0y = 0.
    v0z = 2.5
    
    vth = 1.
    
    nh0 = 1.
    
    arg = -(vx - v0x)**2/vth**2 - (vy - v0y)**2/vth**2 - (vz - v0z)**2/vth**2
    
    value = nh0/(pi**(3/2)*vth**3) * exp(arg)
    
    return value


# ============= 0-th moment of equilibrium distribution function fh_eq ===========
@types('double','double','double')
def nh_eq_phys(x, y, z):
    
    nh0 = 1.
    
    return nh0

# ============= 1-st moment of equilibrium distribution function fh_eq ===========

# x - component
@types('double','double','double')
def jhx_eq(x, y, z):
    
    nh0 = 1.
    v0x = 0.
    
    return nh0 * v0x

# y - component
@types('double','double','double')
def jhy_eq(x, y, z):
    
    nh0 = 1.
    v0y = 0.
    
    return nh0 * v0y

# z - component
@types('double','double','double')
def jhz_eq(x, y, z):
    
    nh0 = 1.
    v0z = 2.5
    
    return nh0 * v0z

# ============= energy of equilibrium distribution function fh_eq ===============
@types('int','double[:]')
def eh_eq(kind_map, params_map):
    
    v0x = 0.
    v0y = 0.
    v0z = 2.5
    
    vth = 1.
    
    nh0 = 1.
    
    if   kind_map == 10:
        value = nh0/2 * params_map[0] * params_map[1] * params_map[2] * (v0x**2 + v0y**2 + v0z**2 + 3*vth**2/2)
        
    elif kind_map == 12:
        value = nh0/2 * params_map[0] * params_map[1] * params_map[3] * (v0x**2 + v0y**2 + v0z**2 + 3*vth**2/2)
        
    elif kind_map == 13:
        value = nh0/2 * params_map[0] * params_map[1] * params_map[3] * (v0x**2 + v0y**2 + v0z**2 + 3*vth**2/2)
        
    else:
        value = 0.
    
    return value


# ===============================================================
#                       logical domain
# ===============================================================

# ======= equilibrium distribution function (used in delta-f method) =============
@types('double','double','double','double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def fh_eq(eta1, eta2, eta3, vx, vy, vz, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    x = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    y = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    z = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    return pull.pull_0_form(fh_eq_phys(x, y, z, vx, vy, vz), eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)