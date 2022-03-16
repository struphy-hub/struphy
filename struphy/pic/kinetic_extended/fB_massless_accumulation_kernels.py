# import pyccel decorators
from pyccel.decorators import types

# import module for mapping evaluation
import struphy.geometry.mappings_3d_fast as mapping_fast

# import module for matrix-matrix and matrix-vector multiplications
import struphy.linear_algebra.core as linalg

# import modules for B-spline evaluation
import struphy.feec.bsplines_kernels as bsp
import struphy.feec.basics.spline_evaluation_3d as eva


@types('double','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:]','int[:]','int[:]','int[:]','double[:]','double[:]','double[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:]','double[:,:]','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def accumulate_substepvv(ddt, idnx, idny, idnz, iddx, iddy, iddz, Nel, p, NbaseN, NbaseD, t1, t2, t3, Np_loc, vec1, vec2, vec3, particles, mid_particles, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):

    from numpy import empty, zeros
    vec1[:,:,:] = 0.0
    vec2[:,:,:] = 0.0
    vec3[:,:,:] = 0.0

    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # p + 1 non-vanishing basis functions up tp degree p
    b1  = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2  = empty((pn2 + 1, pn2 + 1), dtype=float)
    b3  = empty((pn3 + 1, pn3 + 1), dtype=float)

    l1  = empty( pn1              , dtype=float)
    l2  = empty( pn2              , dtype=float)
    l3  = empty( pn3              , dtype=float)

    r1  = empty( pn1              , dtype=float)
    r2  = empty( pn2              , dtype=float)
    r3  = empty( pn3              , dtype=float)

    # scaling arrays for M-splines
    d1  = empty( pn1              , dtype=float)
    d2  = empty( pn2              , dtype=float)
    d3  = empty( pn3              , dtype=float)
    
    # non-vanishing N-splines
    bn1 = empty( pn1 + 1          , dtype=float)
    bn2 = empty( pn2 + 1          , dtype=float)
    bn3 = empty( pn3 + 1          , dtype=float)

    # non-vanishing D-splines
    bd1 = empty( pd1 + 1          , dtype=float)
    bd2 = empty( pd2 + 1          , dtype=float)
    bd3 = empty( pd3 + 1          , dtype=float)

    # ================ for mapping evaluation ==================
    # spline degrees
    pf1   = pf[0]
    pf2   = pf[1]
    pf3   = pf[2]
    
    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f   = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f   = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f   = empty((pf3 + 1, pf3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1f   = empty( pf1, dtype=float)
    l2f   = empty( pf2, dtype=float)
    l3f   = empty( pf3, dtype=float)
    
    r1f   = empty( pf1, dtype=float)
    r2f   = empty( pf2, dtype=float)
    r3f   = empty( pf3, dtype=float)
    
    # scaling arrays for M-splines
    d1f   = empty( pf1, dtype=float)
    d2f   = empty( pf2, dtype=float)
    d3f   = empty( pf3, dtype=float)
    
    # pf + 1 derivatives
    der1f = empty( pf1 + 1, dtype=float)
    der2f = empty( pf2 + 1, dtype=float)
    der3f = empty( pf3 + 1, dtype=float)
    
    # needed mapping quantities
    df      = empty((3, 3), dtype=float) 
    dfinv   = empty((3, 3), dtype=float) 
    g       = empty((3, 3), dtype=float) 
    ginv    = empty((3, 3), dtype=float)
    fx      = empty( 3    , dtype=float)
    vel2    = empty( 3    , dtype=float)
    vel     = zeros( 3    , dtype=float)
    # ==========================================================

    #$ omp parallel
    #$ omp do reduction ( + : vec1, vec2, vec3) private (ip, eta1, eta2, eta3, span1, span2, span3, ie1, ie2, ie3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, bn1, bn2, bn3, bd1, bd2, bd3, vel, weight, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, dfinv, fx, det_number, g, ginv, vel2, il1, il2 ,il3)
    for ip in range(Np_loc):
        
        eta1  = particles[0, ip]
        eta2  = particles[1, ip]
        eta3  = particles[2, ip]

        span1 = int(eta1*Nel[0]) + pn1
        span2 = int(eta2*Nel[1]) + pn2
        span3 = int(eta3*Nel[2]) + pn3

        ie1   = span1 - pn1 
        ie2   = span2 - pn2
        ie3   = span3 - pn3

        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)
        
        bn1[:] = b1[pn1]
        bd1[:] = b1[pd1, :pn1]*d1[:]
        bn2[:] = b2[pn2]
        bd2[:] = b2[pd2, :pn2]*d2[:]
        bn3[:] = b3[pn3]
        bd3[:] = b3[pd3, :pn3]*d3[:]

        vel[0]   = particles[3,ip] + ddt * mid_particles[0, ip]
        vel[1]   = particles[4,ip] + ddt * mid_particles[1, ip]
        vel[2]   = particles[5,ip] + ddt * mid_particles[2, ip]
        
        weight   = particles[6,ip]

        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)
        
        mapping_fast.df_inv_all(df, dfinv)
        mapping_fast.g_inv_all(dfinv, ginv)
        linalg.matrix_vector(ginv, vel, vel2)

        for il1 in range(pd1 + 1):
            for il2 in range(pn2 +1):
                for il3 in range(pn3 +1):
                    vec1[iddx[ie1, il1], idny[ie2, il2], idnz[ie3, il3]] += vel2[0] * weight * bd1[il1]*bn2[il2]*bn3[il3]


        for il1 in range(pn1 + 1):
            for il2 in range(pd2 +1):
                for il3 in range(pn3 +1):
                    vec2[idnx[ie1, il1], iddy[ie2, il2], idnz[ie3, il3]] += vel2[1] * weight * bn1[il1]*bd2[il2]*bn3[il3]


        for il1 in range(pn1 + 1):
            for il2 in range(pn2 + 1):
                for il3 in range(pd3 + 1):
                    vec3[idnx[ie1, il1], idny[ie2, il2], iddz[ie3, il3]] += vel2[2] * weight * bn1[il1]*bn2[il2]*bd3[il3]

    #$ omp end do
    #$ omp end parallel
    ierr = 0



# ==============================================================================
@types('int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','int','double[:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def kernel_step4(idnx, idny, idnz, iddx, iddy, iddz, particles, t1, t2, t3, p, nel, nbase_n, nbase_d, np, kind_map, params_map, mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    from numpy import empty, zeros
    # polynomial degrees
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1
    
    # p + 1 non-vanishing basis functions up tp degree p
    b1  = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2  = empty((pn2 + 1, pn2 + 1), dtype=float)
    b3  = empty((pn3 + 1, pn3 + 1), dtype=float)
    
    l1  = empty( pn1              , dtype=float)
    l2  = empty( pn2              , dtype=float)
    l3  = empty( pn3              , dtype=float)
    
    r1  = empty( pn1              , dtype=float)
    r2  = empty( pn2              , dtype=float)
    r3  = empty( pn3              , dtype=float)
    
    # scaling arrays for M-splines
    d1  = empty( pn1              , dtype=float)
    d2  = empty( pn2              , dtype=float)
    d3  = empty( pn3              , dtype=float)
    
    # non-vanishing N-splines
    bn1 = empty( pn1 + 1          , dtype=float)
    bn2 = empty( pn2 + 1          , dtype=float)
    bn3 = empty( pn3 + 1          , dtype=float)
    
    # non-vanishing D-splines
    bd1 = empty( pd1 + 1          , dtype=float)
    bd2 = empty( pd2 + 1          , dtype=float)
    bd3 = empty( pd3 + 1          , dtype=float)
    
    # particle velocity
    v            = zeros( 3    , dtype=float)
    
    # mapping related quantities
    dfinv         = zeros((3, 3), dtype=float)
    ginv         = zeros((3, 3), dtype=float)
    
    temp_mat1    = zeros((3, 3), dtype=float)
    temp_mat2    = zeros((3, 3), dtype=float)
    
    temp_vec     = zeros( 3    , dtype=float)
    
    # reset arrays
    mat11[:, :, :, :, :, :] = 0.
    mat12[:, :, :, :, :, :] = 0.
    mat13[:, :, :, :, :, :] = 0.
    mat22[:, :, :, :, :, :] = 0.
    mat23[:, :, :, :, :, :] = 0.
    mat33[:, :, :, :, :, :] = 0.
    
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.


    # ================ for mapping evaluation ==================
    # spline degrees
    pf1   = pf[0]
    pf2   = pf[1]
    pf3   = pf[2]
    
    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f   = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f   = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f   = empty((pf3 + 1, pf3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1f   = empty( pf1, dtype=float)
    l2f   = empty( pf2, dtype=float)
    l3f   = empty( pf3, dtype=float)
    
    r1f   = empty( pf1, dtype=float)
    r2f   = empty( pf2, dtype=float)
    r3f   = empty( pf3, dtype=float)
    
    # scaling arrays for M-splines
    d1f   = empty( pf1, dtype=float)
    d2f   = empty( pf2, dtype=float)
    d3f   = empty( pf3, dtype=float)
    
    # pf + 1 derivatives
    der1f = empty( pf1 + 1, dtype=float)
    der2f = empty( pf2 + 1, dtype=float)
    der3f = empty( pf3 + 1, dtype=float)
    
    # needed mapping quantities
    df      = empty((3, 3), dtype=float) 
    fx      = empty( 3    , dtype=float)
    # ==========================================================
    
    #$ omp parallel
    #$ omp do reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3) private (ip, eta1, eta2, eta3, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, ie1, ie2, ie3, bn1, bn2, bn3, bd1, bd2, bd3, w, v, ginv, temp_mat2, temp_vec, temp11, temp12, temp13, temp22, temp23, temp33, temp1, temp2, temp3, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, bi1, bi2, bi3, bj1, bj2, bj3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx)
    for ip in range(np):

        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3
        
        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)

        # element indices
        ie1   = span1 - pn1
        ie2   = span2 - pn2
        ie3   = span3 - pn3
        
        # N-splines and D-splines
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # particle weight and velocity
        w    = particles[6, ip]
        v[0] = particles[3, ip]
        v[1] = particles[4, ip]
        v[2] = particles[5, ip]
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
        
        # evaluate inverse metric tensor
        mapping_fast.g_inv_all(dfinv, ginv)
        # ==========================================
        linalg.matrix_vector(ginv, v, temp_vec)
        linalg.matrix_matrix(ginv, ginv, temp_mat2)
        temp11 = w * temp_mat2[0, 0]
        temp12 = w * temp_mat2[0, 1]
        temp13 = w * temp_mat2[0, 2]
        temp22 = w * temp_mat2[1, 1]
        temp23 = w * temp_mat2[1, 2]
        temp33 = w * temp_mat2[2, 2]
        
        temp1  = w * temp_vec[0]
        temp2  = w * temp_vec[1]
        temp3  = w * temp_vec[2]

        # add contribution to 11 component (DNN DNN), 12 component (DNN NDN) and 13 component (DNN NND)
        for il1 in range(pd1 + 1):
            i1  = iddx[ie1, il1]
            bi1 = bd1[il1]
            for il2 in range(pn2 + 1):
                i2  = idny[ie2, il2]
                bi2 = bi1 * bn2[il2]
                for il3 in range(pn3 + 1):
                    i3  = idnz[ie3, il3]
                    bi3 = bi2 * bn3[il3]
                    
                    vec1[i1, i2, i3] += bi3 * temp1 
                    
                    for jl1 in range(pd1 + 1):
                        bj1 = bi3 * bd1[jl1] * temp11
                        for jl2 in range(pn2 + 1):
                            bj2 =  bj1 * bn2[jl2]
                            for jl3 in range(pn3 + 1):
                                bj3 = bj2 * bn3[jl3]
                                
                                mat11[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                
                    for jl1 in range(pn1 + 1):
                        bj1 = bi3 * bn1[jl1] * temp12
                        for jl2 in range(pd2 + 1):
                            bj2 =  bj1 * bd2[jl2]
                            for jl3 in range(pn3 + 1):
                                bj3 = bj2 * bn3[jl3]
                                
                                mat12[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                
                    for jl1 in range(pn1 + 1):
                        bj1 = bi3 * bn1[jl1] * temp13
                        for jl2 in range(pn2 + 1):
                            bj2 =  bj1 * bn2[jl2]
                            for jl3 in range(pd3 + 1):
                                bj3 = bj2 * bd3[jl3]
                                
                                mat13[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                
                                
                                
        # add contribution to 22 component (NDN NDN) and 23 component (NDN NND)
        for il1 in range(pn1 + 1):
            i1  = idnx[ie1, il1]
            bi1 = bn1[il1]
            for il2 in range(pd2 + 1):
                i2  = iddy[ie2, il2]
                bi2 = bi1 * bd2[il2]
                for il3 in range(pn3 + 1):
                    i3  = idnz[ie3, il3]
                    bi3 = bi2 * bn3[il3]
                    
                    vec2[i1, i2, i3] += bi3 * temp2 
                    
                    for jl1 in range(pn1 + 1):
                        bj1 = bi3 * bn1[jl1]
                        
                        for jl2 in range(pd2 + 1):
                            bj2 =  bj1 * bd2[jl2] * temp22
                            for jl3 in range(pn3 + 1):
                                bj3 = bj2 * bn3[jl3]
                                
                                mat22[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                   
                        for jl2 in range(pn2 + 1):
                            bj2 =  bj1 * bn2[jl2] * temp23
                            for jl3 in range(pd3 + 1):
                                bj3 = bj2 * bd3[jl3]
                                
                                mat23[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                
                       
        
        # add contribution to 33 component (NND NND)
        for il1 in range(pn1 + 1):
            i1  = idnx[ie1, il1]
            bi1 = bn1[il1]
            for il2 in range(pn2 + 1):
                i2  = idny[ie2, il2]
                bi2 = bi1 * bn2[il2]
                for il3 in range(pd3 + 1):
                    i3  = iddz[ie3, il3]
                    bi3 = bi2 * bd3[il3]
                    
                    vec3[i1, i2, i3] += bi3 * temp3 
                    
                    for jl1 in range(pn1 + 1):
                        bj1 = bi3 * bn1[jl1] * temp33
                        for jl2 in range(pn2 + 1):
                            bj2 =  bj1 * bn2[jl2]
                            for jl3 in range(pd3 + 1):
                                bj3 = bj2 * bd3[jl3]
                                
                                mat33[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                
        # ==========================================
                                                   
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0



