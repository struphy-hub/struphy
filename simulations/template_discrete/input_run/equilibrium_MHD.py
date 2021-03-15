from pyccel.decorators import types

import hylife.geometry.mappings_discrete  as mapping
import hylife.geometry.pull_back_discrete as pull



# ========== physical domain ================================

# equilibrium bulk pressure (x - component)
@types('double','double','double')
def p_eq_phys(x, y, z):
    
    p_phys = 1.
    
    return p_phys

# equilibrium bulk velocity (x - component)
@types('double','double','double')
def ux_eq(x, y, z):
    
    ux = 0.
    
    return ux

# equilibrium bulk velocity (y - component)
@types('double','double','double')
def uy_eq(x, y, z):
    
    uy = 0.
    
    return uy

# equilibrium bulk velocity (z - component)
@types('double','double','double')
def uz_eq(x, y, z):
    
    uz = 0.
    
    return uz

# equilibrium magnetic field (x - component)
@types('double','double','double')
def bx_eq(x, y, z):
    
    bx = 1.
    
    return bx

# equilibrium magnetic field (y - component)
@types('double','double','double')
def by_eq(x, y, z):
    
    by = 0.
    
    return by

# equilibrium magnetic field (z - component)
@types('double','double','double')
def bz_eq(x, y, z):
    
    bz = 0.
    
    return bz

# equilibrium bulk density
@types('double','double','double')
def rho_eq_phys(x, y, z):
    
    rho_phys = 1.
    
    return rho_phys





# ======== pull-back to logical domain ===================

# equilibrium bulk pressure (0-form on logical domain)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def p_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    return pull.pull_0_form(p_eq_phys(x, y, z), eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

# equilibrium bulk velocity (1-form on logical domain, 1-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def u1_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    ux = ux_eq(x, y, z)
    uy = uy_eq(x, y, z)
    uz = uz_eq(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)

# equilibrium bulk velocity (1-form on logical domain, 2-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def u2_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    ux = ux_eq(x, y, z)
    uy = uy_eq(x, y, z)
    uz = uz_eq(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)

# equilibrium bulk velocity (1-form on logical domain, 3-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def u3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    ux = ux_eq(x, y, z)
    uy = uy_eq(x, y, z)
    uz = uz_eq(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)

# equilibrium magnetic field (2-form on logical domain, 1-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b1_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    bx = bx_eq(x, y, z)
    by = by_eq(x, y, z)
    bz = bz_eq(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)

# equilibrium magnetic field (2-form on logical domain, 2-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b2_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    bx = bx_eq(x, y, z)
    by = by_eq(x, y, z)
    bz = bz_eq(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)

# equilibrium magnetic field (3-form on logical domain, 3-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    bx = bx_eq(x, y, z)
    by = by_eq(x, y, z)
    bz = bz_eq(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)

# equilibrium bulk density (3-form on logical domain)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def rho_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    return pull.pull_3_form(rho_eq_phys(x, y, z), eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)





# =====================================================================
# curl of equilibrium magnetic field (nabla x (DF^T * B_phys)) 

# curl of equilibrium magnetic field (1-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def curlb1_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    return 0.

# curl of equilibrium magnetic field (2-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def curlb2_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    return 0.

# curl of equilibrium magnetic field (3-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def curlb3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    return 0.
