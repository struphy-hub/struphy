from pyccel.decorators import types

import hylife.geometry.mappings_discrete as mapping
from numpy import exp, pi


# ======= initial distribution function (physical domain) =============
@types('double','double','double','double','double','double')
def fh_ini_phys(x, y, z, vx, vy, vz):
    
    v0x = 2.5
    v0y = 0.
    v0z = 0.
    
    vth = 1.
    
    nh0 = 0.05
    
    arg = -(vx - v0x)**2/vth**2 - (vy - v0y)**2/vth**2 - (vz - v0z)**2/vth**2
    
    value = nh0/(pi**(3/2)*vth**3) * exp(arg)
    
    return value


# ======= initial distribution function (logical domain) ===============
@types('double','double','double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def fh_ini(eta1, eta2, eta3, vx, vy, vz, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    return fh_ini_phys(x, y, z, vx, vy, vz)


# ===== sampling distribution of initial markers on logical domain =====
@types('double','double','double','double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def sh(eta1, eta2, eta3, vx, vy, vz, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    v0x = 2.5
    v0y = 0.
    v0z = 0.
    
    vth = 1.
    
    det_df = mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    
    arg = -(vx - v0x)**2/vth**2 - (vy - v0y)**2/vth**2 - (vz - v0z)**2/vth**2
    
    return 1/(pi**(3/2)*vth**3*det_df)*exp(arg)
