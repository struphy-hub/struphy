# import pyccel decorators
from pyccel.decorators import types

# relative import of subroutines
import ..geometry.mappings_analytical as mapping


# absolute import of interface for simulation setup
import hylife.interface as inter




# ==========================================================================================
@types('double','double','double','int','int','double[:]')        
def fun(xi1, xi2, xi3, kind_fun, kind_map, params):
    
    value = 0.
    
    # initial conditions
    if   kind_fun == 1:
        value = inter.p_ini(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 2:
        value = inter.u1_ini(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 3:
        value = inter.u2_ini(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 4:
        value = inter.u3_ini(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 5:
        value = inter.b1_ini(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 6:
        value = inter.b2_ini(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 7:
        value = inter.b3_ini(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 8:
        value = inter.rho_ini(xi1, xi2, xi3, kind_map, params)
    
    # quantities for projection matrix Q
    elif kind_fun == 11:
        value = inter.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 11)
    elif kind_fun == 12:
        value = inter.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 12)
    elif kind_fun == 13:
        value = inter.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 13)
    elif kind_fun == 14:
        value = inter.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 21)
    elif kind_fun == 15:
        value = inter.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 22)
    elif kind_fun == 16:
        value = inter.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 23)
    elif kind_fun == 17:
        value = inter.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 31)
    elif kind_fun == 18:
        value = inter.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 32)
    elif kind_fun == 19:
        value = inter.rho_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 33)
        
    # quantities for projection matrix T
    elif kind_fun == 21:
        value = inter.b2_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 31) - inter.b3_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 21)
    elif kind_fun == 22:
        value = inter.b2_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 32) - inter.b3_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 22)
    elif kind_fun == 23:
        value = inter.b2_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 33) - inter.b3_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 23)
    elif kind_fun == 24:
        value = inter.b3_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 11) - inter.b1_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 31)
    elif kind_fun == 25:
        value = inter.b3_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 12) - inter.b1_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 32)
    elif kind_fun == 26:
        value = inter.b3_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 13) - inter.b1_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 33)
    elif kind_fun == 27:
        value = inter.b1_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 21) - inter.b2_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 11)
    elif kind_fun == 28:
        value = inter.b1_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 22) - inter.b2_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 12)
    elif kind_fun == 29:
        value = inter.b1_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 23) - inter.b2_eq(xi1, xi2, xi3, kind_map, params) * mapping.g_inv(xi1, xi2, xi3, kind_map, params, 13)
        
    # quantities for projection matrix W
    elif kind_fun == 31:
        value = inter.rho_eq(xi1, xi2, xi3, kind_map, params) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
        
    # quantities for projection matrix P
    elif kind_fun == 41:
        value = -inter.curlb3_eq(xi1, xi2, xi3, kind_map, params) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 42:
        value =  inter.curlb2_eq(xi1, xi2, xi3, kind_map, params) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 43:
        value =  inter.curlb3_eq(xi1, xi2, xi3, kind_map, params) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 44:
        value = -inter.curlb1_eq(xi1, xi2, xi3, kind_map, params) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 45:
        value = -inter.curlb2_eq(xi1, xi2, xi3, kind_map, params) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    elif kind_fun == 46:
        value =  inter.curlb1_eq(xi1, xi2, xi3, kind_map, params) / mapping.det_df(xi1, xi2, xi3, kind_map, params)
    
    # quantities for projection matrices K and S
    elif kind_fun == 91:
        value = inter.p_eq(xi1, xi2, xi3, kind_map, params)
        
    
    return value

     
# ==========================================================================================
@types('int[:]','double[:]','double[:]','double[:]','double[:,:,:](order=F)','int','int','double[:]')        
def kernel_eva(n, xi1, xi2, xi3, mat_f, kind_fun, kind_map, params):
    
    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                mat_f[i1, i2, i3] = fun(xi1[i1], xi2[i2], xi3[i3], kind_fun, kind_map, params)
                