from pyccel.decorators import types

import hylife.geometry.mappings_discrete  as mapping
import hylife.geometry.pull_push_discrete as pull



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
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def p0_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    return pull.pull_0_form(p_eq(x, y, z), eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

# equilibrium bulk pressure (3-form on logical domain)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def p3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    return pull.pull_3_form(p_eq(x, y, z), eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

# equilibrium bulk density (0-form on logical domain)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def rho0_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    return pull.pull_0_form(rho_eq(x, y, z), eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

# equilibrium bulk density (3-form on logical domain)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def rho3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    return pull.pull_3_form(rho_eq(x, y, z), eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

# equilibrium bulk velocity (vector on logical domain, 1-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def u_eq_1(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    ux = u_eq_x(x, y, z)
    uy = u_eq_y(x, y, z)
    uz = u_eq_z(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)

# equilibrium bulk velocity (vector on logical domain, 2-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def u_eq_2(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    ux = u_eq_x(x, y, z)
    uy = u_eq_y(x, y, z)
    uz = u_eq_z(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)

# equilibrium bulk velocity (vector on logical domain, 3-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def u_eq_3(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    ux = u_eq_x(x, y, z)
    uy = u_eq_y(x, y, z)
    uz = u_eq_z(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)

# equilibrium magnetic field (2-form on logical domain, 1-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b2_eq_1(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    bx = b_eq_x(x, y, z)
    by = b_eq_y(x, y, z)
    bz = b_eq_z(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)

# equilibrium magnetic field (2-form on logical domain, 2-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b2_eq_2(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    bx = b_eq_x(x, y, z)
    by = b_eq_y(x, y, z)
    bz = b_eq_z(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)

# equilibrium magnetic field (2-form on logical domain, 3-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b2_eq_3(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    bx = b_eq_x(x, y, z)
    by = b_eq_y(x, y, z)
    bz = b_eq_z(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)



# =====================================================================
# curl of equilibrium magnetic field nabla x (DF^T * B_phys) or nabla x [G * vec(B)] 

# curl of equilibrium magnetic field (1 - component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def curlb_eq_1(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    return 0.

# curl of equilibrium magnetic field (2 - component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def curlb_eq_2(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    return 0.

# curl of equilibrium magnetic field (3 - component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def curlb_eq_3(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    return 0.