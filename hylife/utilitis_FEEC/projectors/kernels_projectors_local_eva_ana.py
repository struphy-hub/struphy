# import pyccel decorators
from pyccel.decorators import types

# import subroutines
import hylife.geometry.mappings_analytical as mapping

import input_run.equilibrium_MHD        as eq_mhd
import input_run.initial_conditions_MHD as ini_mhd

# ==========================================================================================
@types('double','double','double','int','int','double[:]')        
def fun(xi1, xi2, xi3, kind_fun, kind_map, params):
    
    # initial conditions
    if   kind_fun == 1:
        value = ini_mhd.p_ini(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 2:
        value = ini_mhd.u1_ini(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 3:
        value = ini_mhd.u2_ini(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 4:
        value = ini_mhd.u3_ini(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 5:
        value = ini_mhd.b1_ini(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 6:
        value = ini_mhd.b2_ini(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 7:
        value = ini_mhd.b3_ini(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 8:
        value = ini_mhd.rho_ini(xi1, xi2, xi3, kind_map, params)
    
    # quantities for projection matrix Q
    elif kind_fun == 11:
        value = eq_mhd.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 11)
    elif kind_fun == 12:
        value = eq_mhd.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 12)
    elif kind_fun == 13:
        value = eq_mhd.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 13)
    elif kind_fun == 14:
        value = eq_mhd.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 21)
    elif kind_fun == 15:
        value = eq_mhd.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 22)
    elif kind_fun == 16:
        value = eq_mhd.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 23)
    elif kind_fun == 17:
        value = eq_mhd.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 31)
    elif kind_fun == 18:
        value = eq_mhd.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 32)
    elif kind_fun == 19:
        value = eq_mhd.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 33)
        
    # quantities for projection matrix T
    elif kind_fun == 21:
        value = eq_mhd.b2_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 31) - eq_mhd.b3_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 21)
    elif kind_fun == 22:
        value = eq_mhd.b2_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 32) - eq_mhd.b3_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 22)
    elif kind_fun == 23:
        value = eq_mhd.b2_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 33) - eq_mhd.b3_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 23)
    elif kind_fun == 24:
        value = eq_mhd.b3_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 11) - eq_mhd.b1_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 31)
    elif kind_fun == 25:
        value = eq_mhd.b3_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 12) - eq_mhd.b1_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 32)
    elif kind_fun == 26:
        value = eq_mhd.b3_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 13) - eq_mhd.b1_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 33)
    elif kind_fun == 27:
        value = eq_mhd.b1_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 21) - eq_mhd.b2_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 11)
    elif kind_fun == 28:
        value = eq_mhd.b1_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 22) - eq_mhd.b2_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 12)
    elif kind_fun == 29:
        value = eq_mhd.b1_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 23) - eq_mhd.b2_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 13)
        
    # quantities for projection matrix W
    elif kind_fun == 31:
        value = eq_mhd.rho_eq(xi1, xi2, xi3, kind_map, params) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
        
    # quantities for projection matrix P
    elif kind_fun == 41:
        value = -eq_mhd.curlb3_eq(xi1, xi2, xi3, kind_map, params) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 42:
        value =  eq_mhd.curlb2_eq(xi1, xi2, xi3, kind_map, params) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 43:
        value =  eq_mhd.curlb3_eq(xi1, xi2, xi3, kind_map, params) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 44:
        value = -eq_mhd.curlb1_eq(xi1, xi2, xi3, kind_map, params) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 45:
        value = -eq_mhd.curlb2_eq(xi1, xi2, xi3, kind_map, params) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 46:
        value =  eq_mhd.curlb1_eq(xi1, xi2, xi3, kind_map, params) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    
    # quantities for projection matrices K and S
    elif kind_fun == 91:
        value = eq_mhd.p_eq(xi1, xi2, xi3, kind_map, params)
        
    
    return value

     
# ==========================================================================================
@types('int[:]','double[:]','double[:]','double[:]','double[:,:,:]','int','int','double[:]')        
def kernel_eva_ana(n, xi1, xi2, xi3, mat_f, kind_fun, kind_map, params):
    
    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                mat_f[i1, i2, i3] = fun(xi1[i1], xi2[i2], xi3[i3], kind_fun, kind_map, params)