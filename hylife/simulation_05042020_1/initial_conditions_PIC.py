from pyccel.decorators import types
import ..geometry.mappings_analytical as mapping
from numpy import exp, pi


# ======= initial distribution function (physical domain) =============
@types('double','double','double','double','double','double')
def fh_ini_phys(x, y, z, vx, vy, vz):
    
    v0x = 2.5
    v0y = 0.
    v0z = 0.
    
    vth = 1.
    
    nh0 = 0.05
    
    value = nh0/(pi**(3/2)*vth**3) * exp(-(vx - v0x)**2/vth**2 - (vy - v0y)**2/vth**2 - (vz - v0z)**2/vth**2)
    
    return value


# ======= initial distribution function (logical domain) ===============
@types('double','double','double','double','double','double','int','double[:]')
def fh_ini(xi1, xi2, xi3, vx, vy, vz, kind_map, params_map):
    
    x   = mapping.f(xi1, xi2, xi3, kind_map, params_map, 1)
    y   = mapping.f(xi1, xi2, xi3, kind_map, params_map, 2)
    z   = mapping.f(xi1, xi2, xi3, kind_map, params_map, 3)
    
    return mapping.det_df(xi1, xi2, xi3, kind_map, params_map) * fh_ini_phys(x, y, z, vx, vy, vz)


# ===== sampling distribution of initial markers on logical domain =====
@types('double','double','double','double','double','double')
def g_sampling(xi1, xi2, xi3, vx, vy, vz):
    
    v0x = 2.5
    v0y = 0.
    v0z = 0.
    
    vth = 1.
    
    return 1/(pi**(3/2)*vth**3)*exp(-(vx - v0x)**2/vth**2 - (vy - v0y)**2/vth**2 - (vz - v0z)**2/vth**2)