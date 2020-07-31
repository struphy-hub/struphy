from pyccel.decorators import types

# ===================================================
# name input files here
# ===================================================
import simulations.example_analytical.equilibrium_MHD        as eq_mhd
import simulations.example_analytical.equilibrium_PIC        as eq_pic
import simulations.example_analytical.initial_conditions_MHD as ini_mhd
import simulations.example_analytical.initial_conditions_PIC as ini_pic
# ===================================================



# ==================================== equilibrium MHD ====================================================
# equilibrium bulk pressure (0-form on logical domain)
@types('double','double','double','int','double[:]')
def p_eq(xi1, xi2, xi3, kind, params):
    return eq_mhd.p_eq_(xi1, xi2, xi3, kind, params)

# equilibrium bulk velocity (1-form on logical domain, 1 - component)
@types('double','double','double','int','double[:]')
def u1_eq(xi1, xi2, xi3, kind, params):
    return eq_mhd.u1_eq_(xi1, xi2, xi3, kind, params)

# equilibrium bulk velocity (1-form on logical domain, 2 - component)
@types('double','double','double','int','double[:]')
def u2_eq(xi1, xi2, xi3, kind, params):
    return eq_mhd.u2_eq_(xi1, xi2, xi3, kind, params)

# equilibrium bulk velocity (1-form on logical domain, 3 - component)
@types('double','double','double','int','double[:]')
def u3_eq(xi1, xi2, xi3, kind, params):
    return eq_mhd.u3_eq_(xi1, xi2, xi3, kind, params)

# equilibrium magnetic field (2-form on logical domain, 1 - component)
@types('double','double','double','int','double[:]')
def b1_eq(xi1, xi2, xi3, kind, params):
    return eq_mhd.b1_eq_(xi1, xi2, xi3, kind, params)

# equilibrium magnetic field (2-form on logical domain, 2 - component)
@types('double','double','double','int','double[:]')
def b2_eq(xi1, xi2, xi3, kind, params):
    return eq_mhd.b2_eq_(xi1, xi2, xi3, kind, params)

# equilibrium magnetic field (3-form on logical domain, 3 - component)
@types('double','double','double','int','double[:]')
def b3_eq(xi1, xi2, xi3, kind, params):
    return eq_mhd.b3_eq_(xi1, xi2, xi3, kind, params)

# equilibrium bulk density (3-form on logical domain)
@types('double','double','double','int','double[:]')
def rho_eq(xi1, xi2, xi3, kind, params):
    return eq_mhd.rho_eq_(xi1, xi2, xi3, kind, params)

# curl of equilibrium magnetic field (1-form on logical domain, 1 - component)
@types('double','double','double','int','double[:]')
def curlb1_eq(xi1, xi2, xi3, kind, params):
    return eq_mhd.curlb1_eq_(xi1, xi2, xi3, kind, params)

# curl of equilibrium magnetic field (1-form on logical domain, 2 - component)
@types('double','double','double','int','double[:]')
def curlb2_eq(xi1, xi2, xi3, kind, params):
    return eq_mhd.curlb2_eq_(xi1, xi2, xi3, kind, params)

# curl of equilibrium magnetic field (1-form on logical domain, 3 - component)
@types('double','double','double','int','double[:]')
def curlb3_eq(xi1, xi2, xi3, kind, params):
    return eq_mhd.curlb3_eq_(xi1, xi2, xi3, kind, params)


# ==================================== equilibrium PIC ====================================================
# equilibrium energetic ion density (scalar field on physical domain)
@types('double','double','double')
def nh_eq_phys(x, y, z):
    return eq_pic.nh_eq_phys_(x, y, z)

# equilibrium energetic ion current (vector field on physical domain, x - component)
@types('double','double','double')
def jhx_eq(x, y, z):
    return eq_pic.jhx_eq_(x, y, z)

# equilibrium energetic ion current (vector field on physical domain, y - component)
@types('double','double','double')
def jhy_eq(x, y, z):
    return eq_pic.jhy_eq_(x, y, z)

# equilibrium energetic ion current (vector field on physical domain, z - component)
@types('double','double','double')
def jhz_eq(x, y, z):
    return eq_pic.jhz_eq_(x, y, z)

# equilibrium energetic ion distribution function (0-form on logical domain)
@types('double','double','double','double','double','double','int','double[:]')
def fh_eq(xi1, xi2, xi3, vx, vy, vz, kind_map, params_map):
    return eq_pic.fh_eq_(xi1, xi2, xi3, vx, vy, vz, kind_map, params_map)

# equilibrium energetic ion energy
@types('int','double[:]')
def eh_eq(kind_map, params_map):
    return eq_pic.eh_eq_(kind_map, params_map)


# ==================================== initial conditions MHD ================================================
# initial bulk pressure (0-form on logical domain)
@types('double','double','double','int','double[:]')
def p_ini(xi1, xi2, xi3, kind, params):
    return ini_mhd.p_ini_(xi1, xi2, xi3, kind, params)

# initial bulk velocity (1-form on logical domain, 1 - component)
@types('double','double','double','int','double[:]')
def u1_ini(xi1, xi2, xi3, kind, params):
    return ini_mhd.u1_ini_(xi1, xi2, xi3, kind, params)

# initial bulk velocity (1-form on logical domain, 2 - component)
@types('double','double','double','int','double[:]')
def u2_ini(xi1, xi2, xi3, kind, params):
    return ini_mhd.u2_ini_(xi1, xi2, xi3, kind, params)

# initial bulk velocity (1-form on logical domain, 3 - component)
@types('double','double','double','int','double[:]')
def u3_ini(xi1, xi2, xi3, kind, params):
    return ini_mhd.u3_ini_(xi1, xi2, xi3, kind, params)

# initial magnetic field (2-form on logical domain, 1 - component)
@types('double','double','double','int','double[:]')
def b1_ini(xi1, xi2, xi3, kind, params):
    return ini_mhd.b1_ini_(xi1, xi2, xi3, kind, params)

# initial magnetic field (2-form on logical domain, 2 - component)
@types('double','double','double','int','double[:]')
def b2_ini(xi1, xi2, xi3, kind, params):
    return ini_mhd.b2_ini_(xi1, xi2, xi3, kind, params)

# initial magnetic field (3-form on logical domain, 3 - component)
@types('double','double','double','int','double[:]')
def b3_ini(xi1, xi2, xi3, kind, params):
    return ini_mhd.b3_ini_(xi1, xi2, xi3, kind, params)

# initial bulk density (3-form on logical domain)
@types('double','double','double','int','double[:]')
def rho_ini(xi1, xi2, xi3, kind, params):
    return ini_mhd.rho_ini_(xi1, xi2, xi3, kind, params)


# ==================================== initial conditions PIC ================================================
# initial energetic ion distribution function (0-form on logical domain)
@types('double','double','double','double','double','double','int','double[:]')
def fh_ini(xi1, xi2, xi3, vx, vy, vz, kind_map, params_map):
    return ini_pic.fh_ini_(xi1, xi2, xi3, vx, vy, vz, kind_map, params_map)


# ===== sampling distribution of initial markers on logical domain =====
@types('double','double','double','double','double','double','int','double[:]')
def sh(xi1, xi2, xi3, vx, vy, vz, kind_map, params_map):
    return ini_pic.sh_(xi1, xi2, xi3, vx, vy, vz, kind_map, params_map)
