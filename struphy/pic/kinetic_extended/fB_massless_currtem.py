# import pyccel decorators
from pyccel.decorators import types

import struphy.feec.bsplines_kernels as bsp

import struphy.feec.basics.spline_evaluation_3d as eva

import struphy.linear_algebra.core as linalg

import struphy.geometry.mappings_3d_fast as mapping_fast
# ============================================================
@types('double','double[:,:,:,:]','double[:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int','int','double[:,:,:]','int[:]','int','double[:]')
def current(tol, bulk_speed, particles_loc, t1, t2, t3, p, NbaseN, NbaseD, np_loc, Np, gathergrid, nel, kind_map, params_map):

    from numpy import empty, zeros, exp

    # spline degrees
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    bulk_speed[:,:,:,:] = 0.0
    cons = (nel[0]*nel[1]*nel[2])/(params_map[0]*params_map[1]*params_map[2])/Np

# ===================================================
    #$ omp parallel
    #$ omp do reduction ( + : bulk_speed) private (ip, eta1, eta2, eta3, v1, v2, v3, span1, span2, span3, ie1, ie2, ie3, w, U_value)
    for ip in range(np_loc):
    
        eta1 = particles_loc[0, ip]
        eta2 = particles_loc[1, ip]
        eta3 = particles_loc[2, ip]

        v1   = particles_loc[3, ip]
        v2   = particles_loc[4, ip]
        v3   = particles_loc[5, ip]
        
        # ========== field evaluation =============
        # element indices
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3

        ie1   = span1 - pn1
        ie2   = span2 - pn2
        ie3   = span3 - pn3

        # particle weight and velocity
        w    = particles_loc[6, ip]

        U_value = gathergrid[ie1, ie2, ie3]

        if abs(U_value) > tol:
            bulk_speed[0, ie1, ie2, ie3] += w * v1 / U_value * cons
            bulk_speed[1, ie1, ie2, ie3] += w * v2 / U_value * cons
            bulk_speed[2, ie1, ie2, ie3] += w * v3 / U_value * cons
        else:
            bulk_speed[0, ie1, ie2, ie3] += 0.0
            bulk_speed[1, ie1, ie2, ie3] += 0.0
            bulk_speed[2, ie1, ie2, ie3] += 0.0

    #$ omp end do
    #$ omp end parallel


    ierr = 0




# ============================================================
@types('int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','double[:,:,:,:]','double[:,:,:,:]','double[:,:]','int','int','int[:]','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def temper(p, bb1, bb2, bb3, t1, t2, t3, NbaseN, NbaseD, temperature, bulk_speed, particles_loc, np_loc, Np, nel, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    
    from numpy import empty, zeros

    temperature[:,:,:,:] = 0.0
    cons = (nel[0]*nel[1]*nel[2])/(params_map[0]*params_map[1]*params_map[2])/Np
    # spline degrees
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # ===================================================
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

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

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
    fx      = empty( 3    , dtype=float)
    df      = empty((3, 3), dtype=float)
    dfinv   = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    # ==========================================================
    b_cur   = zeros(3           , dtype=float)
    b   = zeros(3                 , dtype=float)
    vel = zeros(3                 , dtype=float)
    b_tensor = zeros((3, 3)       , dtype=float)
    iden_b_tensor = zeros((3, 3)  , dtype=float)

    #$ omp parallel
    #$ omp do reduction ( + : temperature) private (ie1, ie2, ie3, ip, eta1, eta2, eta3, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, bn1, bn2, bn3, bd1, bd2, bd3, b, b_cur, w, vel, il1, il2, b_tensor, iden_b_tensor, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f ,d2f, d3f, der1f, der2f, der3f, df ,fx, dfinv, dfinv_t, b_norm)
    for ip in range(np_loc):

        eta1 = particles_loc[0, ip]
        eta2 = particles_loc[1, ip]
        eta3 = particles_loc[2, ip]

        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3

        ie1 = span1 - pn1
        ie2 = span2 - pn2
        ie3 = span3 - pn3
        
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
                   
        b_cur[0] = eva.evaluation_kernel(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, NbaseD[0], NbaseN[1], NbaseN[2], bb1)
        b_cur[1] = eva.evaluation_kernel(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, NbaseN[0], NbaseD[1], NbaseN[2], bb2)
        b_cur[2] = eva.evaluation_kernel(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, NbaseN[0], NbaseN[1], NbaseD[2], bb3)
        
        b[0]       = dfinv_t[0,0] * b_cur[0] + dfinv_t[0,1] * b_cur[1] + dfinv_t[0,2] * b_cur[2]
        b[1]       = dfinv_t[1,0] * b_cur[0] + dfinv_t[1,1] * b_cur[1] + dfinv_t[1,2] * b_cur[2]
        b[2]       = dfinv_t[2,0] * b_cur[0] + dfinv_t[2,1] * b_cur[1] + dfinv_t[2,2] * b_cur[2]

        b_norm     = (b[0]**2.0 + b[1]**2.0 + b[2]**2.0) ** 0.5

        w      = particles_loc[6, ip]

        vel[0] = particles_loc[3, ip]
        vel[1] = particles_loc[4, ip]
        vel[2] = particles_loc[5, ip]

        if b_norm > 10 ** (-12):
            for il1 in range(3):
                for il2 in range(3):
                    b_tensor[il1, il2] = (b[il1]/b_norm) * (b[il2]/b_norm)
        else:
            b_tensor[:, :] = 0.0

        iden_b_tensor[0, 0] = 0.5 - 0.5 * b_tensor[0, 0]
        iden_b_tensor[1, 1] = 0.5 - 0.5 * b_tensor[1, 1]
        iden_b_tensor[2, 2] = 0.5 - 0.5 * b_tensor[2, 2]

        iden_b_tensor[0, 1] =  - 0.5 * b_tensor[0, 1]
        iden_b_tensor[0, 2] =  - 0.5 * b_tensor[0, 2]
        iden_b_tensor[1, 0] =  - 0.5 * b_tensor[1, 0]
        iden_b_tensor[1, 2] =  - 0.5 * b_tensor[1, 2]
        iden_b_tensor[2, 0] =  - 0.5 * b_tensor[2, 0]
        iden_b_tensor[2, 1] =  - 0.5 * b_tensor[2, 1]

        for il1 in range(3):
            for il2 in range(3):
                temperature[0, ie1, ie2, ie3] += w * cons * (vel[il1] - bulk_speed[il1, ie1, ie2, ie3]) * (vel[il2] - bulk_speed[il2, ie1, ie2, ie3]) * b_tensor[il1, il2]

        for il1 in range(3):
            for il2 in range(3):
                temperature[1, ie1, ie2, ie3] += w * cons * (vel[il1] - bulk_speed[il1, ie1, ie2, ie3]) * (vel[il2] - bulk_speed[il2, ie1, ie2, ie3]) * iden_b_tensor[il1, il2]

    #$ omp end do
    #$ omp end parallel

    ierr = 0
