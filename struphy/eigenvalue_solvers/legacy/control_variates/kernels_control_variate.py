from pyccel.decorators import types

import struphy.geometry.mappings_3d as mapping

import struphy.pic.equilibrium_PIC as eq_pic


# ==========================================================================================
@types('double','double','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')        
def fun(eta1, eta2, eta3, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    # bulk velocity is a 0-form
    if   kind_fun == 1:
        
        x   = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        y   = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        z   = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

        jhx = eq_pic.jhx_eq(x, y, z)
        jhy = eq_pic.jhy_eq(x, y, z)
        jhz = eq_pic.jhz_eq(x, y, z)
        
        detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        dfinv_11 = mapping.df_inv(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_12 = mapping.df_inv(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_13 = mapping.df_inv(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        value = (dfinv_11*jhx + dfinv_12*jhy + dfinv_13*jhz) * abs(detdf)
        
    elif kind_fun == 2:
        
        x   = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        y   = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        z   = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

        jhx = eq_pic.jhx_eq(x, y, z)
        jhy = eq_pic.jhy_eq(x, y, z)
        jhz = eq_pic.jhz_eq(x, y, z)
        
        detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        dfinv_21 = mapping.df_inv(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_22 = mapping.df_inv(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_23 = mapping.df_inv(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        value = (dfinv_21*jhx + dfinv_22*jhy + dfinv_23*jhz) * abs(detdf)
        
    elif kind_fun == 3:
        
        x   = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        y   = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        z   = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

        jhx = eq_pic.jhx_eq(x, y, z)
        jhy = eq_pic.jhy_eq(x, y, z)
        jhz = eq_pic.jhz_eq(x, y, z)
        
        detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        dfinv_31 = mapping.df_inv(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_32 = mapping.df_inv(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_33 = mapping.df_inv(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        value = (dfinv_31*jhx + dfinv_32*jhy + dfinv_33*jhz) * abs(detdf)
    
    elif kind_fun == 4:
        
        x = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        y = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        z = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        value = eq_pic.nh_eq_phys(x, y, z) * abs(detdf)
        
    
    # bulk velocity is a 2-form
    if   kind_fun == 11:
        
        x   = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        y   = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        z   = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

        jhx = eq_pic.jhx_eq(x, y, z)
        jhy = eq_pic.jhy_eq(x, y, z)
        jhz = eq_pic.jhz_eq(x, y, z)
        
        dfinv_11 = mapping.df_inv(eta1, eta2, eta3, 11, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_12 = mapping.df_inv(eta1, eta2, eta3, 12, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_13 = mapping.df_inv(eta1, eta2, eta3, 13, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        value = dfinv_11*jhx + dfinv_12*jhy + dfinv_13*jhz
        
    elif kind_fun == 12:
        
        x   = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        y   = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        z   = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

        jhx = eq_pic.jhx_eq(x, y, z)
        jhy = eq_pic.jhy_eq(x, y, z)
        jhz = eq_pic.jhz_eq(x, y, z)
        
        dfinv_21 = mapping.df_inv(eta1, eta2, eta3, 21, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_22 = mapping.df_inv(eta1, eta2, eta3, 22, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_23 = mapping.df_inv(eta1, eta2, eta3, 23, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        value = dfinv_21*jhx + dfinv_22*jhy + dfinv_23*jhz
        
    elif kind_fun == 13:
        
        x   = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        y   = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        z   = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)

        jhx = eq_pic.jhx_eq(x, y, z)
        jhy = eq_pic.jhy_eq(x, y, z)
        jhz = eq_pic.jhz_eq(x, y, z)
        
        dfinv_31 = mapping.df_inv(eta1, eta2, eta3, 31, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_32 = mapping.df_inv(eta1, eta2, eta3, 32, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        dfinv_33 = mapping.df_inv(eta1, eta2, eta3, 33, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        value = dfinv_31*jhx + dfinv_32*jhy + dfinv_33*jhz
    
    elif kind_fun == 14:
        
        x = mapping.f(eta1, eta2, eta3, 1, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        y = mapping.f(eta1, eta2, eta3, 2, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        z = mapping.f(eta1, eta2, eta3, 3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        detdf = mapping.det_df(eta1, eta2, eta3, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
        
        value = eq_pic.nh_eq_phys(x, y, z) / abs(detdf)
        
    
    return value



# ==========================================================================================
@types('int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:,:,:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')        
def kernel_evaluation_quad(nel, nq, eta1, eta2, eta3, mat_f, kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz):
    
    #$ omp parallel
    #$ omp do private (ie1, ie2, ie3, q1, q2, q3)
    for ie1 in range(nel[0]):
        for ie2 in range(nel[1]):
            for ie3 in range(nel[2]):
    
                for q1 in range(nq[0]):
                    for q2 in range(nq[1]):
                        for q3 in range(nq[2]):
                            mat_f[ie1, q1, ie2, q2, ie3, q3] = fun(eta1[ie1, q1], eta2[ie2, q2], eta3[ie3, q3], kind_fun, kind_map, params_map, tn1, tn2, tn3, pn, nbase_n, cx, cy, cz)
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0