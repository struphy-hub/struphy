# import pyccel decorators
from pyccel.decorators import types

# import module for matrix-matrix and matrix-vector multiplications
import hylife.linear_algebra.core as linalg

# import module for mapping evaluation
import hylife.geometry.mappings_3d_fast as mapping_fast

# import modules for B-spline evaluation
import hylife.utilitis_FEEC.bsplines_kernels as bsp
import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva

# ==============================================================================
@types('double[:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]')
def kernel_step1(particles, t1, t2, t3, p, nel, nbase_n, nbase_d, np, bb1, bb2, bb3, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz, mat12, mat13, mat23):
    
    from numpy import empty, zeros
    
    # reset arrays
    mat12[:, :, :, :, :, :] = 0.
    mat13[:, :, :, :, :, :] = 0.
    mat23[:, :, :, :, :, :] = 0.
    
    # ============== for magnetic field evaluation ============
    # spline degrees
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
    
    # left and right values for spline evaluation
    l1  = empty( pn1, dtype=float)
    l2  = empty( pn2, dtype=float)
    l3  = empty( pn3, dtype=float)
    
    r1  = empty( pn1, dtype=float)
    r2  = empty( pn2, dtype=float)
    r3  = empty( pn3, dtype=float)
    
    # scaling arrays for M-splines
    d1  = empty( pn1, dtype=float)
    d2  = empty( pn2, dtype=float)
    d3  = empty( pn3, dtype=float)
    
    # magnetic field at particle position
    b   = empty( 3, dtype=float)
    # ==========================================================
    
    
    # =================== for deposition =======================
    # non-vanishing N-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)
    
    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)
    # ==========================================================
    
    
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
    # ==========================================================
    
    
    #$ omp parallel
    #$ omp do reduction ( + : mat12, mat13, mat23) private (ip, eta1, eta2, eta3, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, b, ie1, ie2, ie3, bn1, bn2, bn3, bd1, bd2, bd3, temp12, temp13, temp23, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, bi1, bi2, bi3, bj1, bj2, bj3)
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
        
        b[0] = eva.evaluation_kernel(pn1, pd2, pd3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pd3, :pn3]*d3[:], span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], bb1)
        b[1] = eva.evaluation_kernel(pd1, pn2, pd3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pd3, :pn3]*d3[:], span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], bb2)
        b[2] = eva.evaluation_kernel(pd1, pd2, pn3, b1[pd1, :pn1]*d1[:], b2[pd2, :pn2]*d2[:], b3[pn3], span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], bb3)
        # ==========================================
        
        # ========= charge accumulation ============
        # element indices
        ie1   = span1 - pn1
        ie2   = span2 - pn2
        ie3   = span3 - pn3
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]
        
        # particle weight and magnetic field rotation
        temp12 = -particles[6, ip] * b[2]
        temp13 =  particles[6, ip] * b[1]
        temp23 = -particles[6, ip] * b[0]
        
        for il1 in range(pn1 + 1):
            i1  = (ie1 + il1)%nbase_n[0]
            bi1 = bn1[il1]
            for il2 in range(pn2 + 1):
                i2  = (ie2 + il2)%nbase_n[1]
                bi2 = bi1 * bn2[il2]
                for il3 in range(pn3 + 1):
                    i3  = (ie3 + il3)%nbase_n[2]
                    bi3 = bi2 * bn3[il3]
                    
                    for jl1 in range(pn1 + 1):
                        bj1 = bi3 * bn1[jl1]
                        for jl2 in range(pn2 + 1):
                            bj2 =  bj1 * bn2[jl2]
                            for jl3 in range(pn3 + 1):
                                bj3 = bj2 * bn3[jl3]
                                
                                mat12[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * temp12
                                mat13[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * temp13
                                mat23[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * temp23
        
        
        
        ## add contribution to 12 component (DNN NDN) and 13 component (DNN NND)
        #for il1 in range(pd1 + 1):
        #    i1  = (ie1 + il1)%nbase_d[0]
        #    bi1 = bd1[il1]
        #    for il2 in range(pn2 + 1):
        #        i2  = (ie2 + il2)%nbase_n[1]
        #        bi2 = bi1 * bn2[il2]
        #        for il3 in range(pn3 + 1):
        #            i3  = (ie3 + il3)%nbase_n[2]
        #            bi3 = bi2 * bn3[il3]
        #            
        #            for jl1 in range(pn1 + 1):
        #                bj1 = bi3 * bn1[jl1]
        #                
        #                for jl2 in range(pd2 + 1):
        #                    bj2 =  bj1 * bd2[jl2] * temp12
        #                    for jl3 in range(pn3 + 1):
        #                        bj3 = bj2 * bn3[jl3]
        #                        
        #                        mat12[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
        #                        
        #                for jl2 in range(pn2 + 1):
        #                    bj2 =  bj1 * bn2[jl2] * temp13
        #                    for jl3 in range(pd3 + 1):
        #                        bj3 = bj2 * bd3[jl3]
        #                        
        #                        mat13[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
        #
        #
        #
        ## add contribution to 23 component (NDN NND)
        #for il1 in range(pn1 + 1):
        #    i1  = (ie1 + il1)%nbase_n[0]
        #    bi1 = bn1[il1] * temp23
        #    for il2 in range(pd2 + 1):
        #        i2  = (ie2 + il2)%nbase_d[1]
        #        bi2 = bi1 * bd2[il2]
        #        for il3 in range(pn3 + 1):
        #            i3  = (ie3 + il3)%nbase_n[2]
        #            bi3 = bi2 * bn3[il3]
        #            for jl1 in range(pn1 + 1):
        #                bj1 = bi3 * bn1[jl1]
        #                for jl2 in range(pn2 + 1):
        #                    bj2 =  bj1 * bn2[jl2]
        #                    for jl3 in range(pd3 + 1):
        #                        bj3 = bj2 * bd3[jl3]
        #                        
        #                        mat23[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
        # ==========================================
                                                       
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0

    
    
    
    
    
# ==============================================================================
@types('double[:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def kernel_step3(particles, t1, t2, t3, p, nel, nbase_n, nbase_d, np, bb1, bb2, bb3, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz, mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3):
    
    from numpy import empty, zeros
    
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
    
    # ============== for magnetic field evaluation ============
    # spline degrees
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
    
    # left and right values for spline evaluation
    l1  = empty( pn1, dtype=float)
    l2  = empty( pn2, dtype=float)
    l3  = empty( pn3, dtype=float)
    
    r1  = empty( pn1, dtype=float)
    r2  = empty( pn2, dtype=float)
    r3  = empty( pn3, dtype=float)
    
    # scaling arrays for M-splines
    d1  = empty( pn1, dtype=float)
    d2  = empty( pn2, dtype=float)
    d3  = empty( pn3, dtype=float)
    
    # magnetic field at particle position
    b        = empty( 3    , dtype=float)
    b_prod   = zeros((3, 3), dtype=float)
    b_prod_t = zeros((3, 3), dtype=float)
    # ==========================================================
    
    
    # =================== for deposition =======================
    # non-vanishing N-splines
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)
    
    # non-vanishing D-splines
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)
    # ==========================================================
    
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
    df           = empty((3, 3), dtype=float) 
    dfinv        = empty((3, 3), dtype=float) 
    ginv         = empty((3, 3), dtype=float) 
    
    temp_mat1    = empty((3, 3), dtype=float)
    temp_mat2    = empty((3, 3), dtype=float)
    
    temp_mat_vec = empty((3, 3), dtype=float)
    
    temp_vec     = empty( 3    , dtype=float)
    
    # particle velocity
    v            = empty( 3    , dtype=float)
    # ==========================================================
    
    
    #$ omp parallel
    #$ omp do reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3) private (ip, eta1, eta2, eta3, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, b, b_prod_t, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, dfinv, ginv, ie1, ie2, ie3, bn1, bn2, bn3, bd1, bd2, bd3, v, temp_mat_vec, temp_mat1, temp_mat2, temp_vec, temp11, temp12, temp13, temp22, temp23, temp33, temp1, temp2, temp3, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, bi1, bi2, bi3, bj1, bj2, bj3) firstprivate(b_prod)
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
        
        b[0] = eva.evaluation_kernel(pn1, pd2, pd3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pd3, :pn3]*d3[:], span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], bb1)
        b[1] = eva.evaluation_kernel(pd1, pn2, pd3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pd3, :pn3]*d3[:], span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], bb2)
        b[2] = eva.evaluation_kernel(pd1, pd2, pn3, b1[pd1, :pn1]*d1[:], b2[pd2, :pn2]*d2[:], b3[pn3], span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], bb3)
        
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] =  b[1]

        b_prod[1, 0] =  b[2]
        b_prod[1, 2] = -b[0]

        b_prod[2, 0] = -b[1]
        b_prod[2, 1] =  b[0]
        
        linalg.transpose(b_prod, b_prod_t)
        # ==========================================
        
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
        
        # evaluate inverse metric tensor
        mapping_fast.g_inv_all(dfinv, ginv)
        # ==========================================
        
        
        # ========= current accumulation ===========
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
    
        # particle velocity
        v[:] = particles[3:6, ip]
        
        # perform matrix-matrix multiplications
        linalg.matrix_matrix(b_prod, dfinv, temp_mat_vec)
        linalg.matrix_matrix(b_prod, ginv, temp_mat1)
        linalg.matrix_matrix(temp_mat1, b_prod_t, temp_mat2)
        
        linalg.matrix_vector(temp_mat_vec, v, temp_vec)
        
        temp11 = particles[6, ip] * temp_mat2[0, 0]
        temp12 = particles[6, ip] * temp_mat2[0, 1]
        temp13 = particles[6, ip] * temp_mat2[0, 2]
        temp22 = particles[6, ip] * temp_mat2[1, 1]
        temp23 = particles[6, ip] * temp_mat2[1, 2]
        temp33 = particles[6, ip] * temp_mat2[2, 2]
        
        temp1  = particles[6, ip] * temp_vec[0]
        temp2  = particles[6, ip] * temp_vec[1]
        temp3  = particles[6, ip] * temp_vec[2]
        
        
        
        for il1 in range(pn1 + 1):
            i1  = (ie1 + il1)%nbase_n[0]
            bi1 = bn1[il1]
            for il2 in range(pn2 + 1):
                i2  = (ie2 + il2)%nbase_n[1]
                bi2 = bi1 * bn2[il2]
                for il3 in range(pn3 + 1):
                    i3  = (ie3 + il3)%nbase_n[2]
                    bi3 = bi2 * bn3[il3]
                    
                    vec1[i1, i2, i3] += bi3 * temp1
                    vec2[i1, i2, i3] += bi3 * temp2
                    vec3[i1, i2, i3] += bi3 * temp3
                    
                    for jl1 in range(pn1 + 1):
                        bj1 = bi3 * bn1[jl1]
                        for jl2 in range(pn2 + 1):
                            bj2 =  bj1 * bn2[jl2]
                            for jl3 in range(pn3 + 1):
                                bj3 = bj2 * bn3[jl3]
                                
                                mat11[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * temp11
                                mat12[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * temp12
                                mat13[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * temp13
                                mat22[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * temp22
                                mat23[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * temp23
                                mat33[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3 * temp33
        
        
        
        ## add contribution to 11 component (DNN DNN), 12 component (DNN NDN) and 13 component (DNN NND)
        #for il1 in range(pd1 + 1):
        #    i1  = (ie1 + il1)%nbase_d[0]
        #    bi1 = bd1[il1]
        #    for il2 in range(pn2 + 1):
        #        i2  = (ie2 + il2)%nbase_n[1]
        #        bi2 = bi1 * bn2[il2]
        #        for il3 in range(pn3 + 1):
        #            i3  = (ie3 + il3)%nbase_n[2]
        #            bi3 = bi2 * bn3[il3]
        #            
        #            vec1[i1, i2, i3] += bi3 * temp1 
        #            
        #            for jl1 in range(pd1 + 1):
        #                bj1 = bi3 * bd1[jl1] * temp11
        #                for jl2 in range(pn2 + 1):
        #                    bj2 =  bj1 * bn2[jl2]
        #                    for jl3 in range(pn3 + 1):
        #                        bj3 = bj2 * bn3[jl3]
        #                        
        #                        mat11[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
        #                        
        #            for jl1 in range(pn1 + 1):
        #                bj1 = bi3 * bn1[jl1] * temp12
        #                for jl2 in range(pd2 + 1):
        #                    bj2 =  bj1 * bd2[jl2]
        #                    for jl3 in range(pn3 + 1):
        #                        bj3 = bj2 * bn3[jl3]
        #                        
        #                        mat12[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
        #                        
        #            for jl1 in range(pn1 + 1):
        #                bj1 = bi3 * bn1[jl1] * temp13
        #                for jl2 in range(pn2 + 1):
        #                    bj2 =  bj1 * bn2[jl2]
        #                    for jl3 in range(pd3 + 1):
        #                        bj3 = bj2 * bd3[jl3]
        #                        
        #                        mat13[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
        #                        
        #                        
        #                        
        ## add contribution to 22 component (NDN NDN) and 23 component (NDN NND)
        #for il1 in range(pn1 + 1):
        #    i1  = (ie1 + il1)%nbase_n[0]
        #    bi1 = bn1[il1]
        #    for il2 in range(pd2 + 1):
        #        i2  = (ie2 + il2)%nbase_d[1]
        #        bi2 = bi1 * bd2[il2]
        #        for il3 in range(pn3 + 1):
        #            i3  = (ie3 + il3)%nbase_n[2]
        #            bi3 = bi2 * bn3[il3]
        #            
        #            vec2[i1, i2, i3] += bi3 * temp2 
        #            
        #            for jl1 in range(pn1 + 1):
        #                bj1 = bi3 * bn1[jl1]
        #                
        #                for jl2 in range(pd2 + 1):
        #                    bj2 =  bj1 * bd2[jl2] * temp22
        #                    for jl3 in range(pn3 + 1):
        #                        bj3 = bj2 * bn3[jl3]
        #                        
        #                        mat22[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
        #                           
        #                for jl2 in range(pn2 + 1):
        #                    bj2 =  bj1 * bn2[jl2] * temp23
        #                    for jl3 in range(pd3 + 1):
        #                        bj3 = bj2 * bd3[jl3]
        #                        
        #                        mat23[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
        #                        
        #               
        #
        ## add contribution to 33 component (NND NND)
        #for il1 in range(pn1 + 1):
        #    i1  = (ie1 + il1)%nbase_n[0]
        #    bi1 = bn1[il1]
        #    for il2 in range(pn2 + 1):
        #        i2  = (ie2 + il2)%nbase_n[1]
        #        bi2 = bi1 * bn2[il2]
        #        for il3 in range(pd3 + 1):
        #            i3  = (ie3 + il3)%nbase_d[2]
        #            bi3 = bi2 * bd3[il3]
        #            
        #            vec3[i1, i2, i3] += bi3 * temp3 
        #            
        #            for jl1 in range(pn1 + 1):
        #                bj1 = bi3 * bn1[jl1] * temp33
        #                for jl2 in range(pn2 + 1):
        #                    bj2 =  bj1 * bn2[jl2]
        #                    for jl3 in range(pd3 + 1):
        #                        bj3 = bj2 * bd3[jl3]
        #                        
        #                        mat33[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
        # ==========================================
                                                   
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0