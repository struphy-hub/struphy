from pyccel.decorators import types

import hylife.geometry.mappings_3d as mapping
import hylife.geometry.pullback_3d as pull


# ===============================================================
#                       physical domain
# ===============================================================

# equilibrium bulk pressure
@types('double','double','double')
def p_eq(x, y, z):
    
    p = 1.
    
    return p

# equilibrium bulk velocity (x - component)
@types('double','double','double')
def u_eq_x(x, y, z):
    
    ux = 0.
    
    return ux

# equilibrium bulk velocity (y - component)
@types('double','double','double')
def u_eq_y(x, y, z):
    
    uy = 0.
    
    return uy

# equilibrium bulk velocity (z - component)
@types('double','double','double')
def u_eq_z(x, y, z):
    
    uz = 0.
    
    return uz

# equilibrium magnetic field (x - component)
@types('double','double','double')
def b_eq_x(x, y, z):
    
    bx = 1.
    
    return bx

# equilibrium magnetic field (y - component)
@types('double','double','double')
def b_eq_y(x, y, z):
    
    by = 0.
    
    return by

# equilibrium magnetic field (z - component)
@types('double','double','double')
def b_eq_z(x, y, z):
    
    bz = 0.
    
    return bz

# equilibrium bulk density
@types('double','double','double')
def rho_eq(x, y, z):
    
    rho = 1.
    
    return rho



# ===============================================================
#                       logical domain
# ===============================================================

# equilibrium bulk pressure (0-form on logical domain)
@types('double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def p0_eq(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    x = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    y = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    z = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    return pull.pull_0_form(p_eq(x, y, z), eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

# equilibrium bulk pressure (3-form on logical domain)
@types('double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def p3_eq(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    x = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    y = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    z = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    return pull.pull_3_form(p_eq(x, y, z), eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

# equilibrium bulk density (0-form on logical domain)
@types('double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def rho0_eq(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    x = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    y = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    z = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    return pull.pull_0_form(rho_eq(x, y, z), eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

# equilibrium bulk density (3-form on logical domain)
@types('double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def rho3_eq(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    x = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    y = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    z = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    return pull.pull_3_form(rho_eq(x, y, z), eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

# equilibrium magnetic field (2-form on logical domain, 1-component)
@types('double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b2_eq_1(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    x = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    y = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    z = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    bx = b_eq_x(x, y, z)
    by = b_eq_y(x, y, z)
    bz = b_eq_z(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

# equilibrium magnetic field (2-form on logical domain, 2-component)
@types('double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b2_eq_2(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    x = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    y = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    z = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    bx = b_eq_x(x, y, z)
    by = b_eq_y(x, y, z)
    bz = b_eq_z(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

# equilibrium magnetic field (2-form on logical domain, 3-component)
@types('double','double','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b2_eq_3(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    x = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    y = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    z = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    bx = b_eq_x(x, y, z)
    by = b_eq_y(x, y, z)
    bz = b_eq_z(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)