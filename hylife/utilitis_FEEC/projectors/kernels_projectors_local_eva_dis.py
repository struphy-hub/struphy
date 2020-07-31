# import pyccel decorators
from pyccel.decorators import types

# import subroutines
import hylife.geometry.mappings_discrete as mapping

import hylife.interface_discrete as inter_dis


# ==========================================================================================
@types('double','double','double','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')        
def fun(xi1, xi2, xi3, kind_fun, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    # initial conditions
    if   kind_fun == 1:
        value = inter_dis.p_ini(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 2:
        value = inter_dis.u1_ini(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 3:
        value = inter_dis.u2_ini(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 4:
        value = inter_dis.u3_ini(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 5:
        value = inter_dis.b1_ini(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 6:
        value = inter_dis.b2_ini(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 7:
        value = inter_dis.b3_ini(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 8:
        value = inter_dis.rho_ini(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    
    # quantities for projection matrix Q
    elif kind_fun == 11:
        value = inter_dis.rho_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_11(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 12:
        value = inter_dis.rho_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_12(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 13:
        value = inter_dis.rho_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_13(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 14:
        value = inter_dis.rho_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_21(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 15:
        value = inter_dis.rho_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_22(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 16:
        value = inter_dis.rho_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_23(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 17:
        value = inter_dis.rho_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_31(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 18:
        value = inter_dis.rho_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_32(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 19:
        value = inter_dis.rho_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_33(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
    # quantities for projection matrix T
    elif kind_fun == 21:
        value = inter_dis.b2_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_31(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) - inter_dis.b3_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_21(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 22:
        value = inter_dis.b2_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_32(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) - inter_dis.b3_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_22(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 23:
        value = inter_dis.b2_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_33(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) - inter_dis.b3_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_23(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 24:
        value = inter_dis.b3_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_11(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) - inter_dis.b1_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_31(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 25:
        value = inter_dis.b3_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_12(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) - inter_dis.b1_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_32(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 26:
        value = inter_dis.b3_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_13(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) - inter_dis.b1_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_33(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 27:
        value = inter_dis.b1_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_21(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) - inter_dis.b2_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_11(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 28:
        value = inter_dis.b1_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_22(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) - inter_dis.b2_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_12(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 29:
        value = inter_dis.b1_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_23(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) - inter_dis.b2_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_13(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
    # quantities for projection matrix W
    elif kind_fun == 31:
        value = inter_dis.rho_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) / mapping.det_df(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
    # quantities for projection matrix P
    elif kind_fun == 41:
        value = -inter_dis.curlb3_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) / mapping.det_df(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 42:
        value =  inter_dis.curlb2_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) / mapping.det_df(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 43:
        value =  inter_dis.curlb3_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) / mapping.det_df(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 44:
        value = -inter_dis.curlb1_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) / mapping.det_df(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 45:
        value = -inter_dis.curlb2_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) / mapping.det_df(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 46:
        value =  inter_dis.curlb1_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) / mapping.det_df(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    
    # quantities for projection matrices K and S
    elif kind_fun == 91:
        value = inter_dis.p_eq(xi1, xi2, xi3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
    
    return value

     
# ==========================================================================================
@types('int[:]','double[:]','double[:]','double[:]','double[:,:,:]','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')        
def kernel_eva_dis(n, xi1, xi2, xi3, mat_f, kind_fun, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                mat_f[i1, i2, i3] = fun(xi1[i1], xi2[i2], xi3[i3], kind_fun, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)