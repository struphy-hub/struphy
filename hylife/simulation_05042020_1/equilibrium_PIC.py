from pyccel.decorators import types
import ..geometry.mappings_analytical as mapping
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
    
    value = nh0/(pi**(3/2)*vth**3) * exp(-(vx - v0x)**2/vth**2 - (vy - v0y)**2/vth**2 - (vz - v0z)**2/vth**2)
    
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
    
    value = nh0/2 * params_map[0] * params_map[1] * params_map[2] * (v0x**2 + v0y**2 + v0z**2 + 3*vth**2/2)
    
    return value


# ============================ logical domain ===========================================================
# ======= equilibrium distribution function (used in delta-f method) =============
@types('double','double','double','double','double','double','int','double[:]')
def fh_eq(xi1, xi2, xi3, vx, vy, vz, kind_map, params_map):
    
    x   = mapping.f(xi1, xi2, xi3, kind_map, params_map, 1)
    y   = mapping.f(xi1, xi2, xi3, kind_map, params_map, 2)
    z   = mapping.f(xi1, xi2, xi3, kind_map, params_map, 3)
    
    return mapping.det_df(xi1, xi2, xi3, kind_map, params_map) * fh_eq_phys(x, y, z, vx, vy, vz)