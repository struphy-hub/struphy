from pyccel.decorators import types

import hylife.geometry.mappings_analytical  as mapping
import hylife.geometry.pull_back_analytical as pull



# ============================ physical domain ===========================================================
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

# equilibrium velocity (y - component)
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





# ============================ logical domain ===========================================================
# equilibrium bulk pressure (0-form on logical domain)
@types('double','double','double','int','double[:]')
def p_eq_(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    return pull.pull_0_form(p_eq_phys(x, y, z), xi1, xi2, xi3, kind, params)

# equilibrium bulk velocity (1-form on logical domain, 1 - component)
@types('double','double','double','int','double[:]')
def u1_eq_(xi1, xi2, xi3, kind, params):
    
    x  = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y  = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z  = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    ux = ux_eq(x, y, z)
    uy = uy_eq(x, y, z)
    uz = uz_eq(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, xi1, xi2, xi3, kind, params, 1)

# equilibrium bulk velocity (1-form on logical domain, 2 - component)
@types('double','double','double','int','double[:]')
def u2_eq_(xi1, xi2, xi3, kind, params):
    
    x  = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y  = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z  = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    ux = ux_eq(x, y, z)
    uy = uy_eq(x, y, z)
    uz = uz_eq(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, xi1, xi2, xi3, kind, params, 2)

# equilibrium bulk velocity (1-form on logical domain, 3 - component)
@types('double','double','double','int','double[:]')
def u3_eq_(xi1, xi2, xi3, kind, params):
    
    x  = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y  = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z  = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    ux = ux_eq(x, y, z)
    uy = uy_eq(x, y, z)
    uz = uz_eq(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, xi1, xi2, xi3, kind, params, 3)

# equilibrium magnetic field (2-form on logical domain, 1 - component)
@types('double','double','double','int','double[:]')
def b1_eq_(xi1, xi2, xi3, kind, params):
    
    x  = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y  = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z  = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    bx = bx_eq(x, y, z)
    by = by_eq(x, y, z)
    bz = bz_eq(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, xi1, xi2, xi3, kind, params, 1)

# equilibrium magnetic field (2-form on logical domain, 2 - component)
@types('double','double','double','int','double[:]')
def b2_eq_(xi1, xi2, xi3, kind, params):
    
    x  = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y  = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z  = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    bx = bx_eq(x, y, z)
    by = by_eq(x, y, z)
    bz = bz_eq(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, xi1, xi2, xi3, kind, params, 2)

# equilibrium magnetic field (3-form on logical domain, 3 - component)
@types('double','double','double','int','double[:]')
def b3_eq_(xi1, xi2, xi3, kind, params):
    
    x  = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y  = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z  = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    bx = bx_eq(x, y, z)
    by = by_eq(x, y, z)
    bz = bz_eq(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, xi1, xi2, xi3, kind, params, 3)

# equilibrium bulk density (3-form on logical domain)
@types('double','double','double','int','double[:]')
def rho_eq_(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    return pull.pull_3_form(rho_eq_phys(x, y, z), xi1, xi2, xi3, kind, params)





# =====================================================================
# curl of equilibrium magnetic field (nabla x (DF^T * B_phys)) 

# curl of equilibrium magnetic field (1 - component)
@types('double','double','double','int','double[:]')
def curlb1_eq_(xi1, xi2, xi3, kind, params):
    return 0.

# curl of equilibrium magnetic field (2 - component)
@types('double','double','double','int','double[:]')
def curlb2_eq_(xi1, xi2, xi3, kind, params):
    return 0.

# curl of equilibrium magnetic field (3 - component)
@types('double','double','double','int','double[:]')
def curlb3_eq_(xi1, xi2, xi3, kind, params):
    return 0.