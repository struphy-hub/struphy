from pyccel.decorators import types

import hylife.geometry.mappings_analytical  as mapping
import hylife.geometry.pull_back_analytical as pull

from numpy import sin, cos, pi

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
    
    arg = kx * x + ky * y + kz * z
    
    bz  = amp * sin(arg)
    
    return bz

# initial bulk density
@types('double','double','double')
def rho_ini_phys(x, y, z):
    
    rho_phys = 0.
    
    return rho_phys





# ============================ logical domain ===========================================================
# initial bulk pressure
@types('double','double','double','int','double[:]')
def p_ini_(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    return pull.pull_0_form(p_ini_phys(x, y, z), xi1, xi2, xi3, kind, params)

# initial bulk velocity (1 - component)
@types('double','double','double','int','double[:]')
def u1_ini_(xi1, xi2, xi3, kind, params):
    
    x  = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y  = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z  = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    ux = ux_ini(x, y, z)
    uy = uy_ini(x, y, z)
    uz = uz_ini(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, xi1, xi2, xi3, kind, params, 1)

# initial bulk velocity (2 - component)
@types('double','double','double','int','double[:]')
def u2_ini_(xi1, xi2, xi3, kind, params):
    
    x  = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y  = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z  = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    ux = ux_ini(x, y, z)
    uy = uy_ini(x, y, z)
    uz = uz_ini(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, xi1, xi2, xi3, kind, params, 2)

# initial bulk velocity (3 - component)
@types('double','double','double','int','double[:]')
def u3_ini_(xi1, xi2, xi3, kind, params):
    
    x  = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y  = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z  = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    ux = ux_ini(x, y, z)
    uy = uy_ini(x, y, z)
    uz = uz_ini(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, xi1, xi2, xi3, kind, params, 3)

# initial magnetic field (1 - component)
@types('double','double','double','int','double[:]')
def b1_ini_(xi1, xi2, xi3, kind, params):
    
    x  = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y  = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z  = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    bx = bx_ini(x, y, z)
    by = by_ini(x, y, z)
    bz = bz_ini(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, xi1, xi2, xi3, kind, params, 1)

# initial magnetic field (2 - component)
@types('double','double','double','int','double[:]')
def b2_ini_(xi1, xi2, xi3, kind, params):
    
    x  = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y  = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z  = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    bx = bx_ini(x, y, z)
    by = by_ini(x, y, z)
    bz = bz_ini(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, xi1, xi2, xi3, kind, params, 2)

# initial magnetic field (3 - component)
@types('double','double','double','int','double[:]')
def b3_ini_(xi1, xi2, xi3, kind, params):
    
    x  = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y  = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z  = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    bx = bx_ini(x, y, z)
    by = by_ini(x, y, z)
    bz = bz_ini(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, xi1, xi2, xi3, kind, params, 3)

# initial bulk density
@types('double','double','double','int','double[:]')
def rho_ini_(xi1, xi2, xi3, kind, params):
    
    x  = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y  = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z  = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    return pull.pull_3_form(rho_ini_phys(x, y, z), xi1, xi2, xi3, kind, params)