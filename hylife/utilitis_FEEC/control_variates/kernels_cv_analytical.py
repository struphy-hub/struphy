# import pyccel decorators
from pyccel.decorators import types

import hylife.geometry.mappings_analytical as mapping

import input_run.equilibrium_PIC as eq_pic
import input_run.equilibrium_MHD as eq_mhd


# ==========================================================================================
@types('double','double','double','int','int','double[:]')        
def fun(eta1, eta2, eta3, kind_fun, kind_map, params_map):
    
    if   kind_fun == 1:
        
        x     = mapping.f(eta1, eta2, eta3, kind_map, params_map, 1)
        y     = mapping.f(eta1, eta2, eta3, kind_map, params_map, 2)
        z     = mapping.f(eta1, eta2, eta3, kind_map, params_map, 3)

        jhx   = eq_pic.jhx_eq(x, y, z)
        jhy   = eq_pic.jhy_eq(x, y, z)
        jhz   = eq_pic.jhz_eq(x, y, z)
        
        df_11 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 11)
        df_21 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 21)
        df_31 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 31)
        
        value = df_11 * jhx + df_21 * jhy + df_31 * jhz
        
    elif kind_fun == 2:
        
        x     = mapping.f(eta1, eta2, eta3, kind_map, params_map, 1)
        y     = mapping.f(eta1, eta2, eta3, kind_map, params_map, 2)
        z     = mapping.f(eta1, eta2, eta3, kind_map, params_map, 3)

        jhx   = eq_pic.jhx_eq(x, y, z)
        jhy   = eq_pic.jhy_eq(x, y, z)
        jhz   = eq_pic.jhz_eq(x, y, z)
        
        df_12 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 12)
        df_22 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 22)
        df_32 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 32)
        
        value = df_12 * jhx + df_22 * jhy + df_32 * jhz
        
    elif kind_fun == 3:
        
        x     = mapping.f(eta1, eta2, eta3, kind_map, params_map, 1)
        y     = mapping.f(eta1, eta2, eta3, kind_map, params_map, 2)
        z     = mapping.f(eta1, eta2, eta3, kind_map, params_map, 3)

        jhx   = eq_pic.jhx_eq(x, y, z)
        jhy   = eq_pic.jhy_eq(x, y, z)
        jhz   = eq_pic.jhz_eq(x, y, z)
        
        df_13 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 13)
        df_23 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 23)
        df_33 = mapping.df(eta1, eta2, eta3, kind_map, params_map, 33)
        
        value = df_13 * jhx + df_23 * jhy + df_33 * jhz
    
    elif kind_fun == 4:
        
        x     = mapping.f(eta1, eta2, eta3, kind_map, params_map, 1)
        y     = mapping.f(eta1, eta2, eta3, kind_map, params_map, 2)
        z     = mapping.f(eta1, eta2, eta3, kind_map, params_map, 3)
        
        value = eq_pic.nh_eq_phys(x, y, z)
    
    elif kind_fun == 11:
        value = eq_mhd.b1_eq(eta1, eta2, eta3, kind_map, params_map)
    elif kind_fun == 12:
        value = eq_mhd.b2_eq(eta1, eta2, eta3, kind_map, params_map)
    elif kind_fun == 13:
        value = eq_mhd.b3_eq(eta1, eta2, eta3, kind_map, params_map)
        
    elif kind_fun == 21:
        value = mapping.g(eta1, eta2, eta3, kind_map, params_map, 11) / abs(mapping.det_df(eta1, eta2, eta3, kind_map, params_map))
    elif kind_fun == 22:
        value = mapping.g(eta1, eta2, eta3, kind_map, params_map, 21) / abs(mapping.det_df(eta1, eta2, eta3, kind_map, params_map))
    elif kind_fun == 23:
        value = mapping.g(eta1, eta2, eta3, kind_map, params_map, 22) / abs(mapping.det_df(eta1, eta2, eta3, kind_map, params_map))
    elif kind_fun == 24:
        value = mapping.g(eta1, eta2, eta3, kind_map, params_map, 31) / abs(mapping.det_df(eta1, eta2, eta3, kind_map, params_map))
    elif kind_fun == 25:
        value = mapping.g(eta1, eta2, eta3, kind_map, params_map, 32) / abs(mapping.det_df(eta1, eta2, eta3, kind_map, params_map))
    elif kind_fun == 26:
        value = mapping.g(eta1, eta2, eta3, kind_map, params_map, 33) / abs(mapping.det_df(eta1, eta2, eta3, kind_map, params_map))
    
    return value

     

# ==========================================================================================
@types('int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:,:,:]','int','int','double[:]')        
def kernel_evaluation(nel, nq, eta1, eta2, eta3, mat_f, kind_fun, kind_map, params):
    
    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3)
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(nel[2]):
    
                for q1 in range(nq[0]):
                    for q2 in range(nq[1]):
                        for q3 in range(nq[2]):
                            mat_f[ie1, ie2, ie3, q1, q2, q3] = fun(eta1[ie1, q1], eta2[ie2, q2], eta3[ie3, q3], kind_fun, kind_map, params)
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
                            
                
# ==========================================================================================
@types('int[:]','int[:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:]','int[:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:,:,:]','int','int','double[:]')        
def kernel_evaluate_2form(nel, p, ns, nq, pts1, pts2, pts3, b_coeff, nbase, bi1, bi2, bi3, b_eva, component, kind_map, params_map):
    
    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3, il1, il2, il3)
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(nel[2]):
                
                for q1 in range(nq[0]):
                    for q2 in range(nq[1]):
                        for q3 in range(nq[2]):
                            
                            b_eva[ie1, ie2, ie3, q1, q2, q3] = fun(pts1[ie1, q1], pts2[ie2, q2], pts3[ie3, q3], component, kind_map, params_map)
                
                            for il1 in range(p[0] + 1 - ns[0]):
                                for il2 in range(p[1] + 1 - ns[1]):
                                    for il3 in range(p[2] + 1 - ns[2]):
                                        
                                        b_eva[ie1, ie2, ie3, q1, q2, q3] += b_coeff[(ie1 + il1)%nbase[0], (ie2 + il2)%nbase[1], (ie3 + il3)%nbase[2]] * bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
                                        
                                        

# ==========================================================================================          
@types('int','int','int','int','int','int','int','int','int','int','int','int','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:,:,:]','int','int','int','double[:,:,:]','double[:,:,:,:,:,:]')
def kernel_inner(nel1, nel2, nel3, p1, p2, p3, nq1, nq2, nq3, ni1, ni2, ni3, w1, w2, w3, bi1, bi2, bi3, nbase1, nbase2, nbase3, mat, mat_f):
    
    mat[:, :, :] = 0.
    
    #$ omp parallel
    #$ omp do reduction ( + : mat) private (ie1, ie2, ie3, il1, il2, il3, value, q1, q2, q3, wvol, bi)
    for ie1 in range(nel1):
        for ie2 in range(nel2):
            for ie3 in range(nel3):

                for il1 in range(p1 + 1 - ni1):
                    for il2 in range(p2 + 1 - ni2):
                        for il3 in range(p3 + 1 - ni3):
                            
                            value = 0.

                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):

                                        wvol = w1[ie1, q1] * w2[ie2, q2] * w3[ie3, q3]
                                        bi   = bi1[ie1, il1, 0, q1] * bi2[ie2, il2, 0, q2] * bi3[ie3, il3, 0, q3]

                                        value += wvol * bi * mat_f[ie1, ie2, ie3, q1, q2, q3]

                            mat[(ie1 + il1)%nbase1, (ie2 + il2)%nbase2, (ie3 + il3)%nbase3] += value
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0