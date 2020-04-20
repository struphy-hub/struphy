from pyccel.decorators import types
from numpy             import sin, cos, pi
import hylife.geometry.mappings_analytical as mapping



# ============================ physical domain ===========================================================
# initial bulk pressure
@types('double','double','double')
def p_ini_phys(x, y, z):
    
    p_phys = 0.
    #p_phys = sin(2*pi*xi1) * sin(2*pi*xi2) * sin(2*pi*xi3)
    
    return p_phys

# initial bulk velocity (x - component)
@types('double','double','double')
def ux_ini(x, y, z):
    
    ux = 0.
    #ux = cos(2*pi*xi1) * sin(2*pi*xi2) * sin(2*pi*xi3) * 2*pi
    
    return ux

# initial bulk velocity (y - component)
@types('double','double','double')
def uy_ini(x, y, z):
    
    uy = 0.
    #uy = sin(2*pi*xi1) * cos(2*pi*xi2) * sin(2*pi*xi3) * 2*pi
    
    return uy

# initial bulk velocity (z - component)
@types('double','double','double')
def uz_ini(x, y, z):
    
    uz = 0.
    #uz = sin(2*pi*xi1) * sin(2*pi*xi2) * cos(2*pi*xi3) * 2*pi
    
    return uz

# initial magnetic field (x - component)
@types('double','double','double')
def bx_ini(x, y, z):
    
    bx = 0.
    #bx = sin(2*pi*xi1) * cos(2*pi*xi2) * cos(2*pi*xi3) * (2*pi)**2
    
    return bx

# initial magnetic field (y - component)
@types('double','double','double')
def by_ini(x, y, z):
    
    by = 0.
    #by = cos(2*pi*xi1) * sin(2*pi*xi2) * cos(2*pi*xi3) * (2*pi)**2
    
    return by

# initial magnetic field (z - component)
@types('double','double','double')
def bz_ini(x, y, z):
    
    amp = 1e-3
    
    kx  = 0.75
    ky  = 0.
    kz  = 0.
    
    bz  = amp * sin(kx * x + ky * y)
    
    #bz = cos(2*pi*xi1) * cos(2*pi*xi2) * sin(2*pi*xi3) * (2*pi)**2
    
    return bz

# initial bulk density
@types('double','double','double')
def rho_ini_phys(x, y, z):
    
    rho_phys = 0.
    #rho_phys = cos(2*pi*xi1) * cos(2*pi*xi2) * cos(2*pi*xi3) * (2*pi)**3
    
    return rho_phys





# ============================ logical domain ===========================================================
# initial bulk pressure
@types('double','double','double','int','double[:]')
def p_ini(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    p_phys = p_ini_phys(x, y, z)
    
    return p_phys

# initial bulk velocity (1 - component)
@types('double','double','double','int','double[:]')
def u1_ini(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    df_11 = mapping.df(xi1, xi2, xi3, kind, params, 11)
    df_21 = mapping.df(xi1, xi2, xi3, kind, params, 21)
    df_31 = mapping.df(xi1, xi2, xi3, kind, params, 31)
    
    ux = ux_ini(x, y, z)
    uy = uy_ini(x, y, z)
    uz = uz_ini(x, y, z)
    
    return df_11 * ux + df_21 * uy + df_31 * uz

# initial bulk velocity (2 - component)
@types('double','double','double','int','double[:]')
def u2_ini(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    df_12 = mapping.df(xi1, xi2, xi3, kind, params, 12)
    df_22 = mapping.df(xi1, xi2, xi3, kind, params, 22)
    df_32 = mapping.df(xi1, xi2, xi3, kind, params, 32)
    
    ux = ux_ini(x, y, z)
    uy = uy_ini(x, y, z)
    uz = uz_ini(x, y, z)
    
    return df_12 * ux + df_22 * uy + df_32 * uz

# initial bulk velocity (3 - component)
@types('double','double','double','int','double[:]')
def u3_ini(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    df_13 = mapping.df(xi1, xi2, xi3, kind, params, 13)
    df_23 = mapping.df(xi1, xi2, xi3, kind, params, 23)
    df_33 = mapping.df(xi1, xi2, xi3, kind, params, 33)
    
    ux = ux_ini(x, y, z)
    uy = uy_ini(x, y, z)
    uz = uz_ini(x, y, z)
    
    return df_13 * ux + df_23 * uy + df_33 * uz

# initial magnetic field (1 - component)
@types('double','double','double','int','double[:]')
def b1_ini(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    dfinv_11 = mapping.df_inv(xi1, xi2, xi3, kind, params, 11)
    dfinv_12 = mapping.df_inv(xi1, xi2, xi3, kind, params, 12)
    dfinv_13 = mapping.df_inv(xi1, xi2, xi3, kind, params, 13)
    
    det_df   = mapping.det_df(xi1, xi2, xi3, kind, params)
    
    bx = bx_ini(x, y, z)
    by = by_ini(x, y, z)
    bz = bz_ini(x, y, z)
    
    return (dfinv_11 * bx + dfinv_12 * by + dfinv_13 * bz) * det_df

# initial magnetic field (2 - component)
@types('double','double','double','int','double[:]')
def b2_ini(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    dfinv_21 = mapping.df_inv(xi1, xi2, xi3, kind, params, 21)
    dfinv_22 = mapping.df_inv(xi1, xi2, xi3, kind, params, 22)
    dfinv_23 = mapping.df_inv(xi1, xi2, xi3, kind, params, 23)
    
    det_df   = mapping.det_df(xi1, xi2, xi3, kind, params)
    
    bx = bx_ini(x, y, z)
    by = by_ini(x, y, z)
    bz = bz_ini(x, y, z)
    
    return (dfinv_21 * bx + dfinv_22 * by + dfinv_23 * bz) * det_df

# initial magnetic field (3 - component)
@types('double','double','double','int','double[:]')
def b3_ini(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    dfinv_31 = mapping.df_inv(xi1, xi2, xi3, kind, params, 31)
    dfinv_32 = mapping.df_inv(xi1, xi2, xi3, kind, params, 32)
    dfinv_33 = mapping.df_inv(xi1, xi2, xi3, kind, params, 33)
    
    det_df   = mapping.det_df(xi1, xi2, xi3, kind, params)
    
    bx = bx_ini(x, y, z)
    by = by_ini(x, y, z)
    bz = bz_ini(x, y, z)
    
    return (dfinv_31 * bx + dfinv_32 * by + dfinv_33 * bz) * det_df

# initial bulk density
@types('double','double','double','int','double[:]')
def rho_ini(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    rho_phys = rho_ini_phys(x, y, z)
    
    return rho_phys * mapping.det_df(xi1, xi2, xi3, kind, params)