# import pyccel decorators
from pyccel.decorators import types

import hylife.geometry.mappings_discrete as mapping

import input_run.equilibrium_PIC as eq_pic
import input_run.equilibrium_MHD as eq_mhd


# ==========================================================================================
@types('double','double','double','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')        
def fun(eta1, eta2, eta3, kind_fun, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    if   kind_fun == 1:
        
        x     = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        y     = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        z     = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)

        jhx   = eq_pic.jhx_eq(x, y, z)
        jhy   = eq_pic.jhy_eq(x, y, z)
        jhz   = eq_pic.jhz_eq(x, y, z)
        
        detdf = mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        dfinv11 = mapping.dfinv_11(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv12 = mapping.dfinv_12(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv13 = mapping.dfinv_13(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        value = (dfinv11 * jhx + dfinv12 * jhy + dfinv13 * jhz) * abs(detdf)
        
    elif kind_fun == 2:
        
        x     = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        y     = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        z     = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)

        jhx   = eq_pic.jhx_eq(x, y, z)
        jhy   = eq_pic.jhy_eq(x, y, z)
        jhz   = eq_pic.jhz_eq(x, y, z)
        
        detdf = mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        dfinv21 = mapping.dfinv_21(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv22 = mapping.dfinv_22(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv23 = mapping.dfinv_23(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        value = (dfinv21 * jhx + dfinv22 * jhy + dfinv23 * jhz) * abs(detdf)
        
    elif kind_fun == 3:
        
        x     = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        y     = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        z     = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)

        jhx   = eq_pic.jhx_eq(x, y, z)
        jhy   = eq_pic.jhy_eq(x, y, z)
        jhz   = eq_pic.jhz_eq(x, y, z)
        
        detdf = mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        dfinv31 = mapping.dfinv_31(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv32 = mapping.dfinv_32(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        dfinv33 = mapping.dfinv_33(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        value = (dfinv31 * jhx + dfinv32 * jhy + dfinv33 * jhz) * abs(detdf)
    
    elif kind_fun == 4:
        
        x     = mapping.f(tf1, tf2, tf3, pf, nbasef, cx, eta1, eta2, eta3)
        y     = mapping.f(tf1, tf2, tf3, pf, nbasef, cy, eta1, eta2, eta3)
        z     = mapping.f(tf1, tf2, tf3, pf, nbasef, cz, eta1, eta2, eta3)
        
        detdf = mapping.det_df(tf1, tf2, tf3, pf, nbasef, cx, cy, cz, eta1, eta2, eta3)
        
        value = eq_pic.nh_eq_phys(x, y, z)* abs(detdf)
    
    elif kind_fun == 11:
        value = eq_mhd.b2_eq_1(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 12:
        value = eq_mhd.b2_eq_2(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    elif kind_fun == 13:
        value = eq_mhd.b2_eq_3(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    
    return value


# ==========================================================================================
@types('int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:,:,:]','int','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')        
def kernel_evaluation_quad(nel, nq, eta1, eta2, eta3, mat_f, kind_fun, tf1, tf2, tf3, pf, nbasef, cx, cy, cz):
    
    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3)
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(nel[2]):
    
                for q1 in range(nq[0]):
                    for q2 in range(nq[1]):
                        for q3 in range(nq[2]):
                            mat_f[ie1, ie2, ie3, q1, q2, q3] = fun(eta1[ie1, q1], eta2[ie2, q2], eta3[ie3, q3], kind_fun, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0