# import pyccel decorators
from pyccel.decorators import types

# import input files for simulation setup
import input_run.equilibrium_MHD as eq_mhd

# import module for matrix-matrix and matrix-vector multiplications
import hylife.linear_algebra.core as linalg

# import modules for B-spline evaluation
import hylife.utilitis_FEEC.bsplines_kernels as bsp
import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva

# ==============================================================================
@types('double[:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]')
def kernel_step1(particles, t1, t2, t3, p, nel, nbase_n, nbase_d, np, bb1, bb2, bb3, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz, mat12, mat13, mat23):
    
    from numpy import empty, zeros
    
    # ====================== for field evaluation =============
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
    
    # magnetic field and velocity field at particle positions
    b      = empty( 3    , dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    # ==========================================================
    
    
    # =================== for deposition =======================
    # non-vanishing N-splines
    bn1 = empty( pn1 + 1          , dtype=float)
    bn2 = empty( pn2 + 1          , dtype=float)
    bn3 = empty( pn3 + 1          , dtype=float)
    
    # non-vanishing D-splines
    bd1 = empty( pd1 + 1          , dtype=float)
    bd2 = empty( pd2 + 1          , dtype=float)
    bd3 = empty( pd3 + 1          , dtype=float)
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
    l1f   = empty( pf1              , dtype=float)
    l2f   = empty( pf2              , dtype=float)
    l3f   = empty( pf3              , dtype=float)
    
    r1f   = empty( pf1              , dtype=float)
    r2f   = empty( pf2              , dtype=float)
    r3f   = empty( pf3              , dtype=float)
    
    # scaling arrays for M-splines
    d1f   = empty( pf1              , dtype=float)
    d2f   = empty( pf2              , dtype=float)
    d3f   = empty( pf3              , dtype=float)
    
    # pf + 1 derivatives
    der1f = empty( pf1 + 1          , dtype=float)
    der2f = empty( pf2 + 1          , dtype=float)
    der3f = empty( pf3 + 1          , dtype=float)
    
    # needed mapping quantities
    df         = empty((3, 3), dtype=float) 
    dfinv      = empty((3, 3), dtype=float) 
    ginv       = empty((3, 3), dtype=float) 
    
    temp_mat1  = empty((3, 3), dtype=float)
    temp_mat2  = empty((3, 3), dtype=float)
    # ==========================================================
    
    
    # reset arrays
    mat12[:, :, :, :, :, :] = 0.
    mat13[:, :, :, :, :, :] = 0.
    mat23[:, :, :, :, :, :] = 0.
    
    #$ omp parallel
    #$ omp do reduction ( + : mat12, mat13, mat23) private (ip, eta1, eta2, eta3, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, b, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, over_det_df, dfinv, ginv, ie1, ie2, ie3, bn1, bn2, bn3, bd1, bd2, bd3, w, temp_mat1, temp_mat2, temp12, temp13, temp23, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, bi1, bi2, bi3, bj1, bj2, bj3) firstprivate(b_prod)
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
        
        b[0] = eva.evaluation_kernel(pn1, pd2, pd3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pd3, :pn3]*d3[:], span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], bb1) + eq_mhd.b1_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        b[1] = eva.evaluation_kernel(pd1, pn2, pd3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pd3, :pn3]*d3[:], span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], bb2) + eq_mhd.b2_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        b[2] = eva.evaluation_kernel(pd1, pd2, pn3, b1[pd1, :pn1]*d1[:], b2[pd2, :pn2]*d2[:], b3[pn3], span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], bb3) + eq_mhd.b3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] =  b[1]

        b_prod[1, 0] =  b[2]
        b_prod[1, 2] = -b[0]

        b_prod[2, 0] = -b[1]
        b_prod[2, 1] =  b[0]
        # ==========================================

        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        bsp.basis_funs_and_der(tf1, pf1, eta1, span1f, l1f, r1f, b1f, d1f, der1f)
        bsp.basis_funs_and_der(tf2, pf2, eta2, span2f, l2f, r2f, b2f, d2f, der2f)
        bsp.basis_funs_and_der(tf3, pf3, eta3, span3f, l3f, r3f, b3f, d3f, der3f)
        
        
        # evaluate components of Jacobian matrix
        df[0, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
    
        df[1, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)

        df[2, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        
        # inverse Jacobian determinant
        over_det_df = 1. / (df[0, 0]*(df[1, 1]*df[2, 2] - df[2, 1]*df[1, 2]) + df[1, 0]*(df[2, 1]*df[0, 2] - df[0, 1]*df[2, 2]) + df[2, 0]*(df[0, 1]*df[1, 2] - df[1, 1]*df[0, 2]))
        
        # inverse Jacobian matrix
        dfinv[0, 0] = (df[1, 1]*df[2, 2] - df[1, 2]*df[2, 1]) * over_det_df
        dfinv[0, 1] = (df[0, 2]*df[2, 1] - df[0, 1]*df[2, 2]) * over_det_df
        dfinv[0, 2] = (df[0, 1]*df[1, 2] - df[0, 2]*df[1, 1]) * over_det_df

        dfinv[1, 0] = (df[1, 2]*df[2, 0] - df[1, 0]*df[2, 2]) * over_det_df
        dfinv[1, 1] = (df[0, 0]*df[2, 2] - df[0, 2]*df[2, 0]) * over_det_df
        dfinv[1, 2] = (df[0, 2]*df[1, 0] - df[0, 0]*df[1, 2]) * over_det_df

        dfinv[2, 0] = (df[1, 0]*df[2, 1] - df[1, 1]*df[2, 0]) * over_det_df
        dfinv[2, 1] = (df[0, 1]*df[0, 2] - df[0, 0]*df[2, 1]) * over_det_df
        dfinv[2, 2] = (df[0, 0]*df[1, 1] - df[0, 1]*df[1, 0]) * over_det_df
        
        # inverse metric tensor
        ginv[0, 0] = dfinv[0, 0]*dfinv[0, 0] + dfinv[0, 1]*dfinv[0, 1] + dfinv[0, 2]*dfinv[0, 2]
        ginv[0, 1] = dfinv[0, 0]*dfinv[1, 0] + dfinv[0, 1]*dfinv[1, 1] + dfinv[0, 2]*dfinv[1, 2]
        ginv[0, 2] = dfinv[0, 0]*dfinv[2, 0] + dfinv[0, 1]*dfinv[2, 1] + dfinv[0, 2]*dfinv[2, 2]

        ginv[1, 0] = ginv[0, 1]
        ginv[1, 1] = dfinv[1, 0]*dfinv[1, 0] + dfinv[1, 1]*dfinv[1, 1] + dfinv[1, 2]*dfinv[1, 2]
        ginv[1, 2] = dfinv[1, 0]*dfinv[2, 0] + dfinv[1, 1]*dfinv[2, 1] + dfinv[1, 2]*dfinv[2, 2]

        ginv[2, 0] = ginv[0, 2]
        ginv[2, 1] = ginv[1, 2]
        ginv[2, 2] = dfinv[2, 0]*dfinv[2, 0] + dfinv[2, 1]*dfinv[2, 1] + dfinv[2, 2]*dfinv[2, 2]
        # ==========================================
        
        
        
        # ========= charge accumulation ============
        
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
        
        # particle weight
        w = particles[6, ip]
        
        # perform matrix-matrix multiplications
        linalg.matrix_matrix(ginv, b_prod, temp_mat1)
        linalg.matrix_matrix(temp_mat1, ginv, temp_mat2)
        
        temp12 = w * temp_mat2[0, 1]
        temp13 = w * temp_mat2[0, 2]
        temp23 = w * temp_mat2[1, 2]
        
        
        # add contribution to 12 component (DNN NDN) and 13 component (DNN NND)
        for il1 in range(pd1 + 1):
            i1  = (ie1 + il1)%nbase_d[0]
            bi1 = bd1[il1]
            for il2 in range(pn2 + 1):
                i2  = (ie2 + il2)%nbase_n[1]
                bi2 = bi1 * bn2[il2]
                for il3 in range(pn3 + 1):
                    i3  = (ie3 + il3)%nbase_n[2]
                    bi3 = bi2 * bn3[il3]
                    
                    for jl1 in range(pn1 + 1):
                        bj1 = bi3 * bn1[jl1]
                        
                        for jl2 in range(pd2 + 1):
                            bj2 =  bj1 * bd2[jl2] * temp12
                            for jl3 in range(pn3 + 1):
                                bj3 = bj2 * bn3[jl3]
                                
                                mat12[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                
                        for jl2 in range(pn2 + 1):
                            bj2 =  bj1 * bn2[jl2] * temp13
                            for jl3 in range(pd3 + 1):
                                bj3 = bj2 * bd3[jl3]
                                
                                mat13[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
        
        
        
        
        # add contribution to 23 component (NDN NND)
        for il1 in range(pn1 + 1):
            i1  = (ie1 + il1)%nbase_n[0]
            bi1 = bn1[il1] * temp23
            for il2 in range(pd2 + 1):
                i2  = (ie2 + il2)%nbase_d[1]
                bi2 = bi1 * bd2[il2]
                for il3 in range(pn3 + 1):
                    i3  = (ie3 + il3)%nbase_n[2]
                    bi3 = bi2 * bn3[il3]
                    for jl1 in range(pn1 + 1):
                        bj1 = bi3 * bn1[jl1]
                        for jl2 in range(pn2 + 1):
                            bj2 =  bj1 * bn2[jl2]
                            for jl3 in range(pd3 + 1):
                                bj3 = bj2 * bd3[jl3]
                                
                                mat23[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                
        # ==========================================
                                                       
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
    
# ==============================================================================
@types('double[:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def kernel_step3(particles, t1, t2, t3, p, nel, nbase_n, nbase_d, np, bb1, bb2, bb3, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz, mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3):
    
    from numpy import empty, zeros
    
    # ====================== for field evaluation =============
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
    
    # magnetic field and velocity field at particle positions
    b        = empty( 3    , dtype=float)
    b_prod   = zeros((3, 3), dtype=float)
    b_prod_t = zeros((3, 3), dtype=float)
    # ==========================================================
    
    
    # =================== for deposition =======================
    # non-vanishing N-splines
    bn1 = empty( pn1 + 1          , dtype=float)
    bn2 = empty( pn2 + 1          , dtype=float)
    bn3 = empty( pn3 + 1          , dtype=float)
    
    # non-vanishing D-splines
    bd1 = empty( pd1 + 1          , dtype=float)
    bd2 = empty( pd2 + 1          , dtype=float)
    bd3 = empty( pd3 + 1          , dtype=float)
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
    l1f   = empty( pf1              , dtype=float)
    l2f   = empty( pf2              , dtype=float)
    l3f   = empty( pf3              , dtype=float)
    
    r1f   = empty( pf1              , dtype=float)
    r2f   = empty( pf2              , dtype=float)
    r3f   = empty( pf3              , dtype=float)
    
    # scaling arrays for M-splines
    d1f   = empty( pf1              , dtype=float)
    d2f   = empty( pf2              , dtype=float)
    d3f   = empty( pf3              , dtype=float)
    
    # pf + 1 derivatives
    der1f = empty( pf1 + 1          , dtype=float)
    der2f = empty( pf2 + 1          , dtype=float)
    der3f = empty( pf3 + 1          , dtype=float)
    
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
    
    
    #$ omp parallel
    #$ omp do reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3) private (ip, eta1, eta2, eta3, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, b, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, over_det_df, dfinv, ginv, ie1, ie2, ie3, bn1, bn2, bn3, bd1, bd2, bd3, w, v, temp_mat1, temp_mat2, temp_mat_vec, temp_vec, b_prod_t, temp11, temp12, temp13, temp22, temp23, temp33, temp1, temp2, temp3, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, bi1, bi2, bi3, bj1, bj2, bj3) firstprivate(b_prod)
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
        
        b[0] = eva.evaluation_kernel(pn1, pd2, pd3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pd3, :pn3]*d3[:], span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], bb1) + eq_mhd.b1_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        b[1] = eva.evaluation_kernel(pd1, pn2, pd3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pd3, :pn3]*d3[:], span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], bb2) + eq_mhd.b2_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        b[2] = eva.evaluation_kernel(pd1, pd2, pn3, b1[pd1, :pn1]*d1[:], b2[pd2, :pn2]*d2[:], b3[pn3], span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], bb3) + eq_mhd.b3_eq(eta1, eta2, eta3, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] =  b[1]

        b_prod[1, 0] =  b[2]
        b_prod[1, 2] = -b[0]

        b_prod[2, 0] = -b[1]
        b_prod[2, 1] =  b[0]
        # ==========================================
        
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        bsp.basis_funs_and_der(tf1, pf1, eta1, span1f, l1f, r1f, b1f, d1f, der1f)
        bsp.basis_funs_and_der(tf2, pf2, eta2, span2f, l2f, r2f, b2f, d2f, der2f)
        bsp.basis_funs_and_der(tf3, pf3, eta3, span3f, l3f, r3f, b3f, d3f, der3f)
        
        
        # evaluate components of Jacobian matrix
        df[0, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
        df[0, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cx)
    
        df[1, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)
        df[1, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cy)

        df[2, 0] = eva.evaluation_kernel(pf1, pf2, pf3, der1f, b2f[pf2], b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 1] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], der2f, b3f[pf3], span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        df[2, 2] = eva.evaluation_kernel(pf1, pf2, pf3, b1f[pf1], b2f[pf2], der3f, span1f, span2f, span3f, nbasef[0], nbasef[1], nbasef[2], cz)
        
        # inverse Jacobian determinant
        over_det_df = 1. / (df[0, 0]*(df[1, 1]*df[2, 2] - df[2, 1]*df[1, 2]) + df[1, 0]*(df[2, 1]*df[0, 2] - df[0, 1]*df[2, 2]) + df[2, 0]*(df[0, 1]*df[1, 2] - df[1, 1]*df[0, 2]))
        
        # inverse Jacobian matrix
        dfinv[0, 0] = (df[1, 1]*df[2, 2] - df[1, 2]*df[2, 1]) * over_det_df
        dfinv[0, 1] = (df[0, 2]*df[2, 1] - df[0, 1]*df[2, 2]) * over_det_df
        dfinv[0, 2] = (df[0, 1]*df[1, 2] - df[0, 2]*df[1, 1]) * over_det_df

        dfinv[1, 0] = (df[1, 2]*df[2, 0] - df[1, 0]*df[2, 2]) * over_det_df
        dfinv[1, 1] = (df[0, 0]*df[2, 2] - df[0, 2]*df[2, 0]) * over_det_df
        dfinv[1, 2] = (df[0, 2]*df[1, 0] - df[0, 0]*df[1, 2]) * over_det_df

        dfinv[2, 0] = (df[1, 0]*df[2, 1] - df[1, 1]*df[2, 0]) * over_det_df
        dfinv[2, 1] = (df[0, 1]*df[0, 2] - df[0, 0]*df[2, 1]) * over_det_df
        dfinv[2, 2] = (df[0, 0]*df[1, 1] - df[0, 1]*df[1, 0]) * over_det_df
        
        # inverse metric tensor
        ginv[0, 0] = dfinv[0, 0]*dfinv[0, 0] + dfinv[0, 1]*dfinv[0, 1] + dfinv[0, 2]*dfinv[0, 2]
        ginv[0, 1] = dfinv[0, 0]*dfinv[1, 0] + dfinv[0, 1]*dfinv[1, 1] + dfinv[0, 2]*dfinv[1, 2]
        ginv[0, 2] = dfinv[0, 0]*dfinv[2, 0] + dfinv[0, 1]*dfinv[2, 1] + dfinv[0, 2]*dfinv[2, 2]

        ginv[1, 0] = ginv[0, 1]
        ginv[1, 1] = dfinv[1, 0]*dfinv[1, 0] + dfinv[1, 1]*dfinv[1, 1] + dfinv[1, 2]*dfinv[1, 2]
        ginv[1, 2] = dfinv[1, 0]*dfinv[2, 0] + dfinv[1, 1]*dfinv[2, 1] + dfinv[1, 2]*dfinv[2, 2]

        ginv[2, 0] = ginv[0, 2]
        ginv[2, 1] = ginv[1, 2]
        ginv[2, 2] = dfinv[2, 0]*dfinv[2, 0] + dfinv[2, 1]*dfinv[2, 1] + dfinv[2, 2]*dfinv[2, 2]
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
    
        # particle weight and velocity
        w    = particles[  6, ip]
        v[:] = particles[3:6, ip]
        
        
        # perform matrix-matrix multiplications
        linalg.matrix_matrix(ginv, b_prod, temp_mat1)
        linalg.matrix_matrix(temp_mat1, dfinv, temp_mat_vec)
        linalg.matrix_vector(temp_mat_vec, v, temp_vec)
        
        linalg.matrix_matrix(temp_mat1, ginv, temp_mat2)
        linalg.transpose(b_prod, b_prod_t)
        linalg.matrix_matrix(temp_mat2, b_prod_t, temp_mat1)
        linalg.matrix_matrix(temp_mat1, ginv, temp_mat2)
        
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
            i1  = (ie1 + il1)%nbase_d[0]
            bi1 = bd1[il1]
            for il2 in range(pn2 + 1):
                i2  = (ie2 + il2)%nbase_n[1]
                bi2 = bi1 * bn2[il2]
                for il3 in range(pn3 + 1):
                    i3  = (ie3 + il3)%nbase_n[2]
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
            i1  = (ie1 + il1)%nbase_n[0]
            bi1 = bn1[il1]
            for il2 in range(pd2 + 1):
                i2  = (ie2 + il2)%nbase_d[1]
                bi2 = bi1 * bd2[il2]
                for il3 in range(pn3 + 1):
                    i3  = (ie3 + il3)%nbase_n[2]
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
            i1  = (ie1 + il1)%nbase_n[0]
            bi1 = bn1[il1]
            for il2 in range(pn2 + 1):
                i2  = (ie2 + il2)%nbase_n[1]
                bi2 = bi1 * bn2[il2]
                for il3 in range(pd3 + 1):
                    i3  = (ie3 + il3)%nbase_d[2]
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