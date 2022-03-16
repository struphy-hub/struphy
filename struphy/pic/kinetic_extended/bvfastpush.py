# import pyccel decorators
from pyccel.decorators import types

# import module for matrix-matrix and matrix-vector multiplications
import struphy.linear_algebra.core as linalg

# import module for mapping evaluation
import struphy.geometry.mappings_3d_fast as mapping_fast

# import modules for B-spline evaluation
import struphy.feec.bsplines_kernels as bsp
import struphy.feec.basics.spline_evaluation_3d as eva







@types('int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double','double[:,:]','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def bvfastpusher(idnx, idny, idnz, iddx, iddy, iddz, t1, t2, t3, nel, p, Np, bb1, bb2, bb3, dt, particles, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    from numpy import empty, zeros
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    pd1 = p[0] -1
    pd2 = p[1] -1
    pd3 = p[2] -1

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

    vec1= zeros( 3    , dtype=float)
    vec2= zeros( 3    , dtype=float)

    ginv    = zeros((3, 3), dtype=float)
    dfinv    = zeros((3, 3), dtype=float)

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
    df    = zeros((3, 3), dtype=float) 
    fx    = zeros( 3    , dtype=float)
    # =========================================

    #$ omp parallel
    #$ omp do private (ip, vec1, eta1, eta2, eta3, span1, span2, span3, ie1, ie2, ie3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, bn1, bn2, bn3, bd1, bd2, bd3, bi1, bi2, bi3, il1, il2, il3, dfinv, ginv, vec2, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx)
    for ip in range(Np):
        vec1[:] = 0.0
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3
        # element indices
        ie1   = span1 - pn1
        ie2   = span2 - pn2
        ie3   = span3 - pn3
        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)
        
        # N-splines and D-splines
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]
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
        for il1 in range(pd1 + 1):
            bi1 = bd1[il1]
            for il2 in range(pn2 + 1):
                bi2 = bi1 * bn2[il2]
                for il3 in range(pn3 + 1):
                    bi3 = bi2 * bn3[il3]
                    vec1[0] += bi3 * bb1[iddx[ie1, il1], idny[ie2, il2], idnz[ie3, il3]]

        for il1 in range(pn1 + 1):
            bi1 = bn1[il1]
            for il2 in range(pd2 + 1):
                bi2 = bi1 * bd2[il2]
                for il3 in range(pn3 + 1):
                    bi3 = bi2 * bn3[il3]
                    vec1[1] += bi3 * bb2[idnx[ie1, il1], iddy[ie2, il2], idnz[ie3, il3]]

        for il1 in range(pn1 + 1):
            bi1 = bn1[il1]
            for il2 in range(pn2 + 1):
                bi2 = bi1 * bn2[il2]
                for il3 in range(pd3 + 1):
                    bi3 = bi2 * bd3[il3]
                    vec1[2] += bi3 * bb3[idnx[ie1, il1], idny[ie2, il2], iddz[ie3, il3]]
        #vec1[0] = eva.evaluation_kernel(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, NbaseD[0], NbaseN[1], NbaseN[2], bb1)
        #vec1[1] = eva.evaluation_kernel(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, NbaseN[0], NbaseD[1], NbaseN[2], bb2)
        #vec1[2] = eva.evaluation_kernel(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, NbaseN[0], NbaseN[1], NbaseD[2], bb3)
        linalg.matrix_vector(ginv, vec1, vec2)
        particles[3, ip] += dt * vec2[0]
        particles[4, ip] += dt * vec2[1]
        particles[5, ip] += dt * vec2[2]

        # ==========================================
                                                   
    #$ omp end do
    #$ omp end parallel

    
    ierr = 0
