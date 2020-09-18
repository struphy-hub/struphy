from pyccel.decorators import types

import hylife.geometry.mappings_discrete  as mapping
import hylife.geometry.pull_push_discrete as pull

from numpy import sin, cos, pi

# ===============================================================
#                       physical domain
# ===============================================================

# initial bulk pressure
@types('double','double','double')
def p_ini(x, y, z):
    
    p = 0.
    
    return p

# initial bulk velocity (x - component)
@types('double','double','double')
def u_ini_x(x, y, z):
    
    ux = 0.
    
    return ux

# initial bulk velocity (y - component)
@types('double','double','double')
def u_ini_y(x, y, z):
    
    uy = 0.
    
    return uy

# initial bulk velocity (z - component)
@types('double','double','double')
def u_ini_z(x, y, z):
    
    uz = 0.
    
    return uz

# initial magnetic field (x - component)
@types('double','double','double')
def b_ini_x(x, y, z):
    
    bx = 0.
    
    return bx

# initial magnetic field (y - component)
@types('double','double','double')
def b_ini_y(x, y, z):
    
    by = 0.
    
    return by

# initial magnetic field (z - component)
@types('double','double','double')
def b_ini_z(x, y, z):
    
    amp = 1e-3
    
    kx  = 0.8
    ky  = 0.
    kz  = 0.
    
    bz  = amp * sin(kx * x + ky * y + kz * z)
    
    return bz

# initial bulk density
@types('double','double','double')
def rho_ini(x, y, z):
    
    rho = 0.
    
    return rho




# ===============================================================
#                       logical domain
# ===============================================================

# initial bulk pressure (3-form)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def p3_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    return pull.pull_0_form(p_ini(x, y, z), eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

# initial bulk velocity (vector, 1-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def u_ini_1(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    ux = u_ini_x(x, y, z)
    uy = u_ini_y(x, y, z)
    uz = u_ini_z(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)

# initial bulk velocity (vector, 2-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def u_ini_2(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    ux = u_ini_x(x, y, z)
    uy = u_ini_y(x, y, z)
    uz = u_ini_z(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)

# initial bulk velocity (vector, 3-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def u_ini_3(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    ux = u_ini_x(x, y, z)
    uy = u_ini_y(x, y, z)
    uz = u_ini_z(x, y, z)
    
    return pull.pull_1_form(ux, uy, uz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)

# initial magnetic field (2-form, 1-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b_ini_1(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    bx = b_ini_x(x, y, z)
    by = b_ini_y(x, y, z)
    bz = b_ini_z(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 1)

# initial magnetic field (2-form, 2-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b_ini_2(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    bx = b_ini_x(x, y, z)
    by = b_ini_y(x, y, z)
    bz = b_ini_z(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 2)

# initial magnetic field (2-form, 3-component)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def b_ini_3(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x  = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y  = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z  = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    bx = b_ini_x(x, y, z)
    by = b_ini_y(x, y, z)
    bz = b_ini_z(x, y, z)
    
    return pull.pull_2_form(bx, by, bz, eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz, 3)

# initial bulk density (3-form)
@types('double','double','double','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def rho3_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    x = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
    y = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
    z = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
    
    return pull.pull_3_form(rho_ini(x, y, z), eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)