from pyccel.decorators import types

import hylife.geometry.mappings_discrete  as mapping
import hylife.geometry.pull_back_discrete as pull
from   numpy import sin, cos

# ============================ physical domain ===========================================================
# initial bulk pressure
@types('double','double','double')
def p_ini_phys(x, y, z):
    
    p_phys = 0.
    
    return p_phys

# initial bulk velocity (x - component)
@types('double','double','double')
def ux_ini(x, y, z):
    
    ux = 0.
    
    return ux

# initial bulk velocity (y - component)
@types('double','double','double')
def uy_ini(x, y, z):
    
    uy = 0.
    
    return uy

# initial bulk velocity (z - component)
@types('double','double','double')
def uz_ini(x, y, z):
    
    uz = 0.
    
    return uz

# initial magnetic field (x - component)
@types('double','double','double')
def bx_ini(x, y, z):
    
    bx = 0.
    
    return bx

# initial magnetic field (y - component)
@types('double','double','double')
def by_ini(x, y, z):
    
    by = 0.
    
    return by

# initial magnetic field (z - component)
@types('double','double','double')
def bz_ini(x, y, z):
    
    amp = 1e-3
    
    kx  = 0.8
    ky  = 0.
    kz  = 0.
    
    bz  = amp * sin(kx * x + ky * y + kz * z)
    
    return bz

# initial bulk density
@types('double','double','double')
def rho_ini_phys(x, y, z):
    
    rho_phys = 0.
    
    return rho_phys





# ======== pull-back to logical domain ===================

# equilibrium bulk pressure (0-form on logical domain)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def p_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    return pull.pull_0_form(p_ini_phys(x, y, z), eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

# equilibrium bulk velocity (1-form on logical domain, 1-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def u1_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    ux = ux_ini(x, y, z)
    uy = uy_ini(x, y, z)
    uz = uz_ini(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)

# equilibrium bulk velocity (1-form on logical domain, 2-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def u2_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    ux = ux_ini(x, y, z)
    uy = uy_ini(x, y, z)
    uz = uz_ini(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)

# equilibrium bulk velocity (1-form on logical domain, 3-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def u3_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    ux = ux_ini(x, y, z)
    uy = uy_ini(x, y, z)
    uz = uz_ini(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)

# equilibrium magnetic field (2-form on logical domain, 1-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b1_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    bx = bx_ini(x, y, z)
    by = by_ini(x, y, z)
    bz = bz_ini(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)

# equilibrium magnetic field (2-form on logical domain, 2-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b2_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    bx = bx_ini(x, y, z)
    by = by_ini(x, y, z)
    bz = bz_ini(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)

# equilibrium magnetic field (3-form on logical domain, 3-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b3_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    bx = bx_ini(x, y, z)
    by = by_ini(x, y, z)
    bz = bz_ini(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)

# equilibrium bulk density (3-form on logical domain)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def rho_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    return pull.pull_3_form(rho_ini_phys(x, y, z), eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
