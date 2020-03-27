from pyccel.decorators import types
import .mappings_analytical as mapping

# =====================================================================
# equilibrium bulk pressure (0 - form on logical domain)
@types('double','double','double','int','double[:]')
def p_eq(xi1, xi2, xi3, kind, params):
    return (1 - xi1) * (1 - xi2) * (1 - xi3) * xi1 * xi2 * xi3

# equilibrium bulk velocity (1-form on logical domain, 1 - component)
@types('double','double','double','int','double[:]')
def u1_eq(xi1, xi2, xi3, kind, params):
    
    df_11 = mapping.df(xi1, xi2, xi3, kind, params, 11)
    df_21 = mapping.df(xi1, xi2, xi3, kind, params, 21)
    df_31 = mapping.df(xi1, xi2, xi3, kind, params, 31)
    
    ux = 0.
    uy = 0.
    uz = 0.
    
    return  df_11 * ux + df_21 * uy + df_31 * uz

# equilibrium bulk velocity (1-form on logical domain, 2 - component)
@types('double','double','double','int','double[:]')
def u2_eq(xi1, xi2, xi3, kind, params):
    
    df_12 = mapping.df(xi1, xi2, xi3, kind, params, 12)
    df_22 = mapping.df(xi1, xi2, xi3, kind, params, 22)
    df_32 = mapping.df(xi1, xi2, xi3, kind, params, 32)
    
    ux = 0.
    uy = 0.
    uz = 0.
    
    return  df_12 * ux + df_22 * uy + df_32 * uz

# equilibrium bulk velocity (1-form on logical domain, 3 - component)
@types('double','double','double','int','double[:]')
def u3_eq(xi1, xi2, xi3, kind, params):
    
    df_13 = mapping.df(xi1, xi2, xi3, kind, params, 13)
    df_23 = mapping.df(xi1, xi2, xi3, kind, params, 23)
    df_33 = mapping.df(xi1, xi2, xi3, kind, params, 33)
    
    ux = 0.
    uy = 0.
    uz = 0.
    
    return  df_13 * ux + df_23 * uy + df_33 * uz

# equilibrium magnetic field (2-form on logical domain, 1 - component)
@types('double','double','double','int','double[:]')
def b1_eq(xi1, xi2, xi3, kind, params):
    
    dfinv_11 = mapping.df_inv(xi1, xi2, xi3, kind, params, 11)
    dfinv_12 = mapping.df_inv(xi1, xi2, xi3, kind, params, 12)
    dfinv_13 = mapping.df_inv(xi1, xi2, xi3, kind, params, 13)
    
    det_df   = mapping.det_df(xi1, xi2, xi3, kind, params)
    
    bx = (1 - xi1) * (1 - xi2) * (1 - xi3) * xi1 * xi2 * xi3
    by = 0.
    bz = 0.
    
    return (dfinv_11 * bx + dfinv_12 * by + dfinv_13 * bz) * det_df

# equilibrium magnetic field (2-form on logical domain, 2 - component)
@types('double','double','double','int','double[:]')
def b2_eq(xi1, xi2, xi3, kind, params):
    
    dfinv_21 = mapping.df_inv(xi1, xi2, xi3, kind, params, 21)
    dfinv_22 = mapping.df_inv(xi1, xi2, xi3, kind, params, 22)
    dfinv_23 = mapping.df_inv(xi1, xi2, xi3, kind, params, 23)
    
    det_df   = mapping.det_df(xi1, xi2, xi3, kind, params)
    
    bx = 1.
    by = 0.
    bz = 0.
    
    return (dfinv_21 * bx + dfinv_22 * by + dfinv_23 * bz) * det_df

# equilibrium magnetic field (2-form on logical domain, 3 - component)
@types('double','double','double','int','double[:]')
def b3_eq(xi1, xi2, xi3, kind, params):
    
    dfinv_31 = mapping.df_inv(xi1, xi2, xi3, kind, params, 31)
    dfinv_32 = mapping.df_inv(xi1, xi2, xi3, kind, params, 32)
    dfinv_33 = mapping.df_inv(xi1, xi2, xi3, kind, params, 33)
    
    det_df   = mapping.det_df(xi1, xi2, xi3, kind, params)
    
    bx = 1.
    by = 0.
    bz = 0.
    
    return (dfinv_31 * bx + dfinv_32 * by + dfinv_33 * bz) * det_df

# equilibrium bulk density (3 - form on logical domain)
@types('double','double','double','int','double[:]')
def rho_eq(xi1, xi2, xi3, kind, params):
    return (1 - xi1) * (1 - xi2) * (1 - xi3) * xi1 * xi2 * xi3 * mapping.det_df(xi1, xi2, xi3, kind, params)


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