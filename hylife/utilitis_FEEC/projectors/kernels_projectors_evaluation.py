# import pyccel decorators
from pyccel.decorators import types

# import subroutines
import hylife.geometry.mappings_3d as mapping

import input_run.equilibrium_MHD        as eq_mhd
import input_run.initial_conditions_MHD as ini_mhd


# ==========================================================================================
@types('double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')        
def fun(eta1, eta2, eta3, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    # initial conditions
    if   kind_fun == 1:
        value = ini_mhd.p3_ini(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 2:
        value = ini_mhd.u_ini_1(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 3:
        value = ini_mhd.u_ini_2(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 4:
        value = ini_mhd.u_ini_3(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 5:
        value = ini_mhd.b2_ini_1(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 6:
        value = ini_mhd.b2_ini_2(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 7:
        value = ini_mhd.b2_ini_3(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 8:
        value = ini_mhd.rho3_ini(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 61:
        value = ini_mhd.u2_ini_1(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 62:
        value = ini_mhd.u2_ini_2(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 63:
        value = ini_mhd.u2_ini_3(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    
    # quantities for projection matrix Q
    elif kind_fun == 11:
        value = eq_mhd.rho3_eq(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # quantities for projection matrix W
    elif kind_fun == 12:
        value = eq_mhd.rho0_eq(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # quantities for projection matrix T
    elif kind_fun == 21:
        value = eq_mhd.b2_eq_1(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 22:
        value = eq_mhd.b2_eq_2(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    elif kind_fun == 23:
        value = eq_mhd.b2_eq_3(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # quantities for projection matrix S
    elif kind_fun == 31:
        value = eq_mhd.p3_eq(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # quantities for projection matrix K
    elif kind_fun == 41:
        value = eq_mhd.p0_eq(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
    # quantities for projection matrix N
    elif kind_fun == 51:
        value = abs(mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz))
        
    return value

     
# ==========================================================================================
@types('double[:]','double[:]','double[:]','double[:,:,:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')        
def kernel_eva(eta1, eta2, eta3, mat_f, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    #$ omp parallel
    #$ omp do private (i1, i2, i3)
    for i1 in range(len(eta1)):
        for i2 in range(len(eta2)):
            for i3 in range(len(eta3)):
                mat_f[i1, i2, i3] = fun(eta1[i1], eta2[i2], eta3[i3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    

# ==========================================================================================
@types('int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:,:,:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')        
def kernel_eva_quad(nel, nq, eta1, eta2, eta3, mat_f, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3)
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(nel[2]):
    
                for q1 in range(nq[0]):
                    for q2 in range(nq[1]):
                        for q3 in range(nq[2]):
                            mat_f[ie1, ie2, ie3, q1, q2, q3] = fun(eta1[ie1, q1], eta2[ie2, q2], eta3[ie3, q3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0