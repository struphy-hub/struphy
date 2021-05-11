from pyccel.decorators import types

import hylife.geometry.mappings_3d as mapping
import hylife.geometry.pullback_3d as pull

from numpy import exp, pi

# ===============================================================
#                       physical domain
# ===============================================================

# ======= initial distribution function =========================
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


# ===============================================================
#                       logical domain
# ===============================================================


# ========== initial distribution function  =====================
@types('double','double','double','double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def fh_ini(eta1, eta2, eta3, vx, vy, vz, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    x = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    y = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    z = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    return pull.pull_0_form(fh_ini_phys(x, y, z, vx, vy, vz), eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)


# ========= sampling distribution of initial markers =============
@types('double','double','double','double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def sh(eta1, eta2, eta3, vx, vy, vz, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    v0x = 2.5
    v0y = 0.
    v0z = 0.
    
    vth = 1.
    
    detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    arg = -(vx - v0x)**2/vth**2 - (vy - v0y)**2/vth**2 - (vz - v0z)**2/vth**2
    
    return 1/(pi**(3/2)*vth**3*detdf)*exp(arg)