# import pyccel decorators
from pyccel.decorators import types

# import subroutines
import hylife.geometry.mappings_discrete as mapping

import input_run.equilibrium_MHD        as eq_mhd
import input_run.initial_conditions_MHD as ini_mhd


# ==========================================================================================
@types('double','double','double','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')        
def fun(eta1, eta2, eta3, kind_fun, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    # initial conditions
    if   kind_fun == 1:
        value = ini_mhd.p_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 2:
        value = ini_mhd.u1_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 3:
        value = ini_mhd.u2_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 4:
        value = ini_mhd.u3_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 5:
        value = ini_mhd.b1_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 6:
        value = ini_mhd.b2_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 7:
        value = ini_mhd.b3_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 8:
        value = ini_mhd.rho_ini(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    
    # quantities for projection matrix Q
    elif kind_fun == 11:
        value = eq_mhd.rho_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_11(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 12:
        value = eq_mhd.rho_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_12(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 13:
        value = eq_mhd.rho_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_13(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 14:
        value = eq_mhd.rho_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_21(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 15:
        value = eq_mhd.rho_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_22(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 16:
        value = eq_mhd.rho_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_23(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 17:
        value = eq_mhd.rho_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_31(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 18:
        value = eq_mhd.rho_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_32(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 19:
        value = eq_mhd.rho_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_33(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
    # quantities for projection matrix T
    elif kind_fun == 21:
        value = eq_mhd.b2_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_31(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3) - eq_mhd.b3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_21(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 22:
        value = eq_mhd.b2_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_32(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3) - eq_mhd.b3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_22(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 23:
        value = eq_mhd.b2_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_33(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3) - eq_mhd.b3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_23(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 24:
        value = eq_mhd.b3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_11(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3) - eq_mhd.b1_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_31(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 25:
        value = eq_mhd.b3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_12(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3) - eq_mhd.b1_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_32(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 26:
        value = eq_mhd.b3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_13(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3) - eq_mhd.b1_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_33(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 27:
        value = eq_mhd.b1_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_21(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3) - eq_mhd.b2_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_11(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 28:
        value = eq_mhd.b1_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_22(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3) - eq_mhd.b2_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_12(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
    elif kind_fun == 29:
        value = eq_mhd.b1_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_23(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3) - eq_mhd.b2_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) * mapping.ginv_13(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
    # quantities for projection matrix W
    elif kind_fun == 31:
        value = eq_mhd.rho_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) / abs(mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3))
        
    # quantities for projection matrix P
    elif kind_fun == 41:
        value = -eq_mhd.curlb3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) / abs(mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3))
    elif kind_fun == 42:
        value =  eq_mhd.curlb2_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) / abs(mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3))
    elif kind_fun == 43:
        value =  eq_mhd.curlb3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) / abs(mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3))
    elif kind_fun == 44:
        value = -eq_mhd.curlb1_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) / abs(mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3))
    elif kind_fun == 45:
        value = -eq_mhd.curlb2_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) / abs(mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3))
    elif kind_fun == 46:
        value =  eq_mhd.curlb1_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz) / abs(mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3))
    
    # quantities for projection matrices K and S
    elif kind_fun == 91:
        value = eq_mhd.p_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
    
    return value

     
# ==========================================================================================
@types('int[:]','double[:]','double[:]','double[:]','double[:,:,:]','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')        
def kernel_eva(n, eta1, eta2, eta3, mat_f, kind_fun, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    #$ omp parallel
    #$ omp do private (i1, i2, i3)
    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                mat_f[i1, i2, i3] = fun(eta1[i1], eta2[i2], eta3[i3], kind_fun, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0