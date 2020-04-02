from pyccel.decorators import types
import ..geometry.mappings_analytical as mapping




# ============================ physical domain ===========================================================
# equilibrium bulk pressure (x - component)
@types('double','double','double')
def p_eq_phys(x, y, z):
    
    #p_phys = 1.
    p_phys = (1 - x) * (1 - y) * (1 - z) * x * y * z
    
    return p_phys

# equilibrium bulk velocity (x - component)
@types('double','double','double')
def ux_eq(x, y, z):
    
    ux = 0.
    #ux = cos(2*pi*xi1) * sin(2*pi*xi2) * sin(2*pi*xi3) * 2*pi
    
    return ux

# equilibrium velocity (y - component)
@types('double','double','double')
def uy_eq(x, y, z):
    
    uy = 0.
    #uy = sin(2*pi*xi1) * cos(2*pi*xi2) * sin(2*pi*xi3) * 2*pi
    
    return uy

# equilibrium bulk velocity (z - component)
@types('double','double','double')
def uz_eq(x, y, z):
    
    uz = 0.
    #uz = sin(2*pi*xi1) * sin(2*pi*xi2) * cos(2*pi*xi3) * 2*pi
    
    return uz

# equilibrium magnetic field (x - component)
@types('double','double','double')
def bx_eq(x, y, z):
    
    bx = 1.
    #bx = sin(2*pi*xi1) * cos(2*pi*xi2) * cos(2*pi*xi3) * (2*pi)**2
    
    return bx

# equilibrium magnetic field (y - component)
@types('double','double','double')
def by_eq(x, y, z):
    
    by = 0.
    #by = cos(2*pi*xi1) * sin(2*pi*xi2) * cos(2*pi*xi3) * (2*pi)**2
    
    return by

# equilibrium magnetic field (z - component)
@types('double','double','double')
def bz_eq(x, y, z):
    
    bz = 0.
    #bz = cos(2*pi*xi1) * cos(2*pi*xi2) * sin(2*pi*xi3) * (2*pi)**2
    
    return bz

# equilibrium bulk density
@types('double','double','double')
def rho_eq_phys(x, y, z):
    
    rho_phys = 1.
    #rho_phys = cos(2*pi*xi1) * cos(2*pi*xi2) * cos(2*pi*xi3) * (2*pi)**3
    
    return rho_phys







# ============================ logical domain ===========================================================
# equilibrium bulk pressure (0-form on logical domain)
@types('double','double','double','int','double[:]')
def p_eq(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    p_phys = p_eq_phys(x, y, z)
    
    return p_phys

# equilibrium bulk velocity (1-form on logical domain, 1 - component)
@types('double','double','double','int','double[:]')
def u1_eq(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    df_11 = mapping.df(xi1, xi2, xi3, kind, params, 11)
    df_21 = mapping.df(xi1, xi2, xi3, kind, params, 21)
    df_31 = mapping.df(xi1, xi2, xi3, kind, params, 31)
    
    ux = ux_eq(x, y, z)
    uy = uy_eq(x, y, z)
    uz = uz_eq(x, y, z)
    
    return df_11 * ux + df_21 * uy + df_31 * uz

# equilibrium bulk velocity (1-form on logical domain, 2 - component)
@types('double','double','double','int','double[:]')
def u2_eq(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    df_12 = mapping.df(xi1, xi2, xi3, kind, params, 12)
    df_22 = mapping.df(xi1, xi2, xi3, kind, params, 22)
    df_32 = mapping.df(xi1, xi2, xi3, kind, params, 32)
    
    ux = ux_eq(x, y, z)
    uy = uy_eq(x, y, z)
    uz = uz_eq(x, y, z)
    
    return df_12 * ux + df_22 * uy + df_32 * uz

# equilibrium bulk velocity (1-form on logical domain, 3 - component)
@types('double','double','double','int','double[:]')
def u3_eq(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    df_13 = mapping.df(xi1, xi2, xi3, kind, params, 13)
    df_23 = mapping.df(xi1, xi2, xi3, kind, params, 23)
    df_33 = mapping.df(xi1, xi2, xi3, kind, params, 33)
    
    ux = ux_eq(x, y, z)
    uy = uy_eq(x, y, z)
    uz = uz_eq(x, y, z)
    
    return df_13 * ux + df_23 * uy + df_33 * uz

# equilibrium magnetic field (2-form on logical domain, 1 - component)
@types('double','double','double','int','double[:]')
def b1_eq(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    dfinv_11 = mapping.df_inv(xi1, xi2, xi3, kind, params, 11)
    dfinv_12 = mapping.df_inv(xi1, xi2, xi3, kind, params, 12)
    dfinv_13 = mapping.df_inv(xi1, xi2, xi3, kind, params, 13)
    
    det_df   = mapping.det_df(xi1, xi2, xi3, kind, params)
    
    bx = bx_eq(x, y, z)
    by = by_eq(x, y, z)
    bz = bz_eq(x, y, z)
    
    return (dfinv_11 * bx + dfinv_12 * by + dfinv_13 * bz) * det_df

# equilibrium magnetic field (2-form on logical domain, 2 - component)
@types('double','double','double','int','double[:]')
def b2_eq(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    dfinv_21 = mapping.df_inv(xi1, xi2, xi3, kind, params, 21)
    dfinv_22 = mapping.df_inv(xi1, xi2, xi3, kind, params, 22)
    dfinv_23 = mapping.df_inv(xi1, xi2, xi3, kind, params, 23)
    
    det_df   = mapping.det_df(xi1, xi2, xi3, kind, params)
    
    bx = bx_eq(x, y, z)
    by = by_eq(x, y, z)
    bz = bz_eq(x, y, z)
    
    return (dfinv_21 * bx + dfinv_22 * by + dfinv_23 * bz) * det_df

# equilibrium magnetic field (3-form on logical domain, 3 - component)
@types('double','double','double','int','double[:]')
def b3_eq(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    dfinv_31 = mapping.df_inv(xi1, xi2, xi3, kind, params, 31)
    dfinv_32 = mapping.df_inv(xi1, xi2, xi3, kind, params, 32)
    dfinv_33 = mapping.df_inv(xi1, xi2, xi3, kind, params, 33)
    
    det_df   = mapping.det_df(xi1, xi2, xi3, kind, params)
    
    bx = bx_eq(x, y, z)
    by = by_eq(x, y, z)
    bz = bz_eq(x, y, z)
    
    return (dfinv_31 * bx + dfinv_32 * by + dfinv_33 * bz) * det_df

# equilibrium bulk density (3-form on logical domain)
@types('double','double','double','int','double[:]')
def rho_eq(xi1, xi2, xi3, kind, params):
    
    x = mapping.f(xi1, xi2, xi3, kind, params, 1)
    y = mapping.f(xi1, xi2, xi3, kind, params, 2)
    z = mapping.f(xi1, xi2, xi3, kind, params, 3)
    
    rho_phys = rho_eq_phys(x, y, z)
    
    return rho_phys * mapping.det_df(xi1, xi2, xi3, kind, params)





# =====================================================================
# curl of equilibrium magnetic field (nabla x (DF^T * B_phys)) 

# curl of equilibrium magnetic field (1 - component)
@types('double','double','double','int','double[:]')
def curlb1_eq(xi1, xi2, xi3, kind, params):
    return 0.

# curl of equilibrium magnetic field (2 - component)
@types('double','double','double','int','double[:]')
def curlb2_eq(xi1, xi2, xi3, kind, params):
    return 0.

# curl of equilibrium magnetic field (3 - component)
@types('double','double','double','int','double[:]')
def curlb3_eq(xi1, xi2, xi3, kind, params):
    return 0.