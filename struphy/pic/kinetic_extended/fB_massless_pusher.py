# import pyccel decorators
from pyccel.decorators import types

# import module for mapping evaluation
import struphy.geometry.mappings_3d_fast as mapping_fast
# import module for matrix-matrix and matrix-vector multiplications
import struphy.linear_algebra.core as linalg

# import modules for B-spline evaluation
import struphy.feec.bsplines_kernels as bsp
import struphy.feec.basics.spline_evaluation_3d as eva

# =============================================================================================================================================================================
@types('double[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:]','int','double')
def rkfinal(particles_loc, stage1_out_loc, stage2_out_loc, stage3_out_loc, stage4_out_loc, Np_loc, dt):

    #$ omp parallel
    #$ omp do private (ip)
    for ip in range(Np_loc):
        particles_loc[3,ip] +=  1.0/6.0 * dt * (stage1_out_loc[0, ip] + 2.0*stage2_out_loc[0, ip] + 2.0*stage3_out_loc[0, ip] + stage4_out_loc[0, ip] )
        particles_loc[4,ip] +=  1.0/6.0 * dt * (stage1_out_loc[1, ip] + 2.0*stage2_out_loc[1, ip] + 2.0*stage3_out_loc[1, ip] + stage4_out_loc[1, ip] )
        particles_loc[5,ip] +=  1.0/6.0 * dt * (stage1_out_loc[2, ip] + 2.0*stage2_out_loc[2, ip] + 2.0*stage3_out_loc[2, ip] + stage4_out_loc[2, ip] )

    #$ omp end do
    #$ omp end parallel


    ierr = 0


# =============================================================================================================================================================================
@types('double[:,:]','double[:,:]','int','double')
def rkfinal2(particles_loc, stage2_out_loc, Np_loc, dt):

    #$ omp parallel
    #$ omp do private (ip)
    for ip in range(Np_loc):
        particles_loc[3,ip] +=  dt * stage2_out_loc[0, ip] 
        particles_loc[4,ip] +=  dt * stage2_out_loc[1, ip] 
        particles_loc[5,ip] +=  dt * stage2_out_loc[2, ip] 

    #$ omp end do
    #$ omp end parallel


    ierr = 0


# =============================================================================================================================================================================
@types('int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','int[:]','int[:]','double[:]','double[:]','double[:]','double[:,:]','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pushvv(idnx, idny, idnz, iddx, iddy, iddz, out_vector, coe1, coe2, coe3, Np_loc, Nel, p, t1, t2, t3, particles, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    # can we set matrix as global variable, how data is sent in python?
    from numpy import empty, zeros
    out_vector[:,:] = 0.0

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
    g       = empty((3, 3), dtype=float) 
    fx      = empty( 3    , dtype=float)
    vel2    = empty( 3    , dtype=float)
    vel     = zeros( 3    , dtype=float)
    dfinv   = empty((3, 3), dtype=float) 
    ginv    = empty((3, 3), dtype=float)

    #$ omp parallel
    #$ omp do private (ip, vel, vel2, eta1, eta2, eta3, span1, span2, span3, ie1, ie2, ie3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, bn1, bn2, bn3, bd1, bd2, bd3, il1, il2, il3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, dfinv, fx, det_number, g, ginv)
    for ip in range(Np_loc):

        vel[:]   = 0.0
        vel2[:]  = 0.
        eta1   = particles[0,ip]
        eta2   = particles[1,ip]
        eta3   = particles[2,ip]
        span1 = int(eta1*Nel[0]) + pn1
        span2 = int(eta2*Nel[1]) + pn2
        span3 = int(eta3*Nel[2]) + pn3

        ie1   = span1 - pn1
        ie2   = span2 - pn2
        ie3   = span3 - pn3

        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)

        bn1[:] = b1[pn1, :]
        bd1[:] = b1[pd1, :pn1]*d1[:]
        bn2[:] = b2[pn2, :]
        bd2[:] = b2[pd2, :pn2]*d2[:]
        bn3[:] = b3[pn3, :]
        bd3[:] = b3[pd3, :pn3]*d3[:]
        

        for il1 in range(pn1 + 1):
            for il2 in range(pd2 + 1):
                for il3 in range(pd3 + 1):
                    vel[0] += coe1[idnx[ie1,il1], iddy[ie2, il2], iddz[ie3, il3]] *  bn1[il1]*bd2[il2]*bd3[il3]  

        for il1 in range(pd1 + 1):
            for il2 in range(pn2 + 1):
                for il3 in range(pd3 + 1):
                    vel[1] += coe2[iddx[ie1,il1], idny[ie2, il2], iddz[ie3, il3]] *  bd1[il1]*bn2[il2]*bd3[il3]  

        for il1 in range(pd1 + 1):
            for il2 in range(pd2 + 1):
                for il3 in range(pn3 + 1):
                    vel[2] += coe3[iddx[ie1,il1], iddy[ie2, il2], idnz[ie3, il3]] *  bd1[il1]*bd2[il2]*bn3[il3]  

        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)
            
        
        # ==========================================
        
        mapping_fast.df_inv_all(df, dfinv)
        mapping_fast.g_inv_all(dfinv, ginv)
            
        linalg.matrix_vector(ginv, vel, vel2)
        out_vector[0, ip] = vel2[0]
        out_vector[1, ip] = vel2[1]
        out_vector[2, ip] = vel2[2]
    #$ omp end do
    #$ omp end parallel


    ierr = 0









# ==========================================================================================================
@types('int[:]','int[:]','double[:]','double[:]','double[:]','int[:]','double[:,:]','double','double[:,:,:]','double[:,:,:]','double[:,:,:]','int[:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def rotation(nbase_n, nbase_d, t1, t2, t3, nel, particles_loc, dt, bb1, bb2, bb3, p, Np_loc, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    
    from numpy import empty, zeros, sin, cos
    # ================ for field evaluation ==================
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
    fx      = empty( 3    , dtype=float)
    df      = empty((3, 3), dtype=float)
    dfinv   = empty((3, 3), dtype=float)
    # ===============================================================

    # ============== needed in rotation =============================
    element_rot = zeros((3,3), dtype=float)
    B_ip = zeros(3, dtype=float)
    B_ip2 = zeros(3, dtype=float)
    B_rotip = zeros((3,3), dtype=float)
    B_rotip_s = zeros((3,3), dtype=float) 
    temp = zeros(3, dtype=float) 
    # ===============================================================

    #$ omp parallel
    #$ omp do private (alpha, span1, span2, span3, temp, ip, eta1, eta2, eta3, element_rot, B_ip, B_ip2, B_rotip, b1, b2, b3, l1, l2, l3, r1, r2, r3, d1, d2, d3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f ,der3f, df, fx, dfinv)
    for ip in range(Np_loc):

        eta1   = particles_loc[0,ip]
        eta2   = particles_loc[1,ip]
        eta3   = particles_loc[2,ip]


        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3

        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, element_rot)

        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3
        
        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)
        
        B_ip2[0] = eva.evaluation_kernel(pd1, pn2, pn3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pn3], span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], bb1)
        B_ip2[1] = eva.evaluation_kernel(pn1, pd2, pn3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pn3], span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], bb2)
        B_ip2[2] = eva.evaluation_kernel(pn1, pn2, pd3, b1[pn1], b2[pn2], b3[pd3, :pn3]*d3[:], span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], bb3)
        
        linalg.matrix_vector(element_rot, B_ip2, B_ip)
        #============================================================
        if (B_ip2[0]**2 + B_ip2[1]**2 + B_ip2[2] ** 2)** 0.5 > 10**(-13): 
            B_ip2[0] = B_ip[0] / ((B_ip[0]*B_ip[0] + B_ip[1]*B_ip[1] + B_ip[2]*B_ip[2])**0.5)
            B_ip2[1] = B_ip[1] / ((B_ip[0]*B_ip[0] + B_ip[1]*B_ip[1] + B_ip[2]*B_ip[2])**0.5)
            B_ip2[2] = B_ip[2] / ((B_ip[0]*B_ip[0] + B_ip[1]*B_ip[1] + B_ip[2]*B_ip[2])**0.5)
            alpha    = dt * ((B_ip[0]*B_ip[0] + B_ip[1]*B_ip[1] + B_ip[2]*B_ip[2])**0.5)
            B_rotip[0,0] = B_ip2[0]**2 + (B_ip2[1]**2 + B_ip2[2]**2)*cos(alpha)
            B_rotip[0,1] = B_ip2[2]*sin(alpha) + B_ip2[0]*B_ip2[1]*(1-cos(alpha))
            B_rotip[0,2] = -B_ip2[1]*sin(alpha) + B_ip2[0]*B_ip2[2]*(1-cos(alpha))
            B_rotip[1,0] = -B_ip2[2]*sin(alpha) + B_ip2[0]*B_ip2[1]*(1-cos(alpha))
            B_rotip[1,1] = B_ip2[1]**2 + (B_ip2[0]**2 + B_ip2[2]**2)*cos(alpha)
            B_rotip[1,2] = B_ip2[0]*sin(alpha) + B_ip2[1]*B_ip2[2]*(1-cos(alpha))
            B_rotip[2,0] = B_ip2[1]*sin(alpha) + B_ip2[0]*B_ip2[2]*(1-cos(alpha))
            B_rotip[2,1] = -B_ip2[0]*sin(alpha) + B_ip2[1]*B_ip2[2]*(1-cos(alpha))
            B_rotip[2,2] = B_ip2[2]**2 + (B_ip2[0]**2 + B_ip2[1]**2)*cos(alpha)

            temp[0] = particles_loc[3, ip]
            temp[1] = particles_loc[4, ip]
            temp[2] = particles_loc[5, ip]
       
            particles_loc[3, ip] = B_rotip[0,0]*temp[0] + B_rotip[0,1]*temp[1] + B_rotip[0,2]*temp[2]
            particles_loc[4, ip] = B_rotip[1,0]*temp[0] + B_rotip[1,1]*temp[1] + B_rotip[1,2]*temp[2]
            particles_loc[5, ip] = B_rotip[2,0]*temp[0] + B_rotip[2,1]*temp[1] + B_rotip[2,2]*temp[2]

    #$ omp end do
    #$ omp end parallel
    
    ierr = 0


# ==============================================================================
@types('int[:]','double[:,:]','int','double','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def step1_pushx(Nel, particles_loc, Np_loc, dt, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    from numpy import empty, zeros
    # =======================================
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
    dfinv   = zeros((3, 3), dtype=float)
    
    #$ omp parallel
    #$ omp do private (ip, eta1, eta2, eta3, dfinv, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx)
    for ip in range(Np_loc):

        eta1 = particles_loc[0, ip]
        eta2 = particles_loc[1, ip]
        eta3 = particles_loc[2, ip]
        # ================================
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
        #=========== first substep  update position =============
        particles_loc[0,ip] = particles_loc[0,ip] + dt / 2.0 * (dfinv[0,0]*particles_loc[3,ip] + dfinv[0,1]*particles_loc[4,ip] + dfinv[0,2]*particles_loc[5,ip])
        particles_loc[1,ip] = particles_loc[1,ip] + dt / 2.0 * (dfinv[1,0]*particles_loc[3,ip] + dfinv[1,1]*particles_loc[4,ip] + dfinv[1,2]*particles_loc[5,ip])
        particles_loc[2,ip] = particles_loc[2,ip] + dt / 2.0 * (dfinv[2,0]*particles_loc[3,ip] + dfinv[2,1]*particles_loc[4,ip] + dfinv[2,2]*particles_loc[5,ip])
        particles_loc[0,ip] = particles_loc[0,ip]%1.0
        particles_loc[1,ip] = particles_loc[1,ip]%1.0
        particles_loc[2,ip] = particles_loc[2,ip]%1.0

    #$ omp end do
    #$ omp end parallel

    ierr = 0            





# ==============================================================================
@types('int[:]','int[:]','int[:]','int','int','int','double','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','int[:]','double[:]','double[:,:]','int','int','double','double[:,:,:,:,:,:]','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def step1_pushv(index_shapex, index_shapey, index_shapez, index_diffx, index_diffy, index_diffz, thermal, p_shape, n_quad, pts1, pts2, pts3, Nel, p_size, particles_loc, Np_loc, Np, dt, weight, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    from numpy import empty, zeros, floor
    # this scheme can be verified to be a good approximation for xv equation
    dfinv        = zeros((3, 3), dtype=float)
    dfinv_t      = zeros((3, 3), dtype=float)
    cell_left    = empty(3, dtype=int)
    point_left   = zeros(3, dtype=float)
    point_right  = zeros(3, dtype=float)
    cell_number  = empty(3, dtype=int)

    temp1        = zeros(3, dtype=float)
    temp4        = zeros(3, dtype=float)
    temp5        = zeros(3, dtype=float)
    temp6        = zeros(3, dtype=float)
    value        = zeros(3, dtype=float)

    compact      = zeros(3, dtype=float)
    compact[0]   = (p_shape[0]+1.0)*p_size[0]
    compact[1]   = (p_shape[1]+1.0)*p_size[1]
    compact[2]   = (p_shape[2]+1.0)*p_size[2]

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


    #===========second substep  update velocity============
    #$ omp parallel
    #$ omp do private (ip, eta1, eta2, eta3, dfinv, dfinv_t, weight_p, point_left, point_right, cell_left, cell_number, il1, il2, il3, q1, q2, q3, temp1, temp4, temp6, value, value_x, value_y, value_z, d_value_x, d_value_y, d_value_z, ww, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx)
    for ip in range(Np_loc):

        eta1   = particles_loc[0,ip]
        eta2   = particles_loc[1,ip]
        eta3   = particles_loc[2,ip]

        # ================================
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        linalg.transpose(dfinv, dfinv_t)
        #==========================

        weight_p = 1.0/(p_size[0]*p_size[1]*p_size[2])
        #the points here are still not put in the periodic box [0, 1] x [0, 1] x [0, 1]
        point_left[0]  = eta1 - 0.5*compact[0]
        point_right[0] = eta1 + 0.5*compact[0]
        point_left[1]  = eta2 - 0.5*compact[1]
        point_right[1] = eta2 + 0.5*compact[1]
        point_left[2]  = eta3 - 0.5*compact[2]
        point_right[2] = eta3 + 0.5*compact[2]
        
        cell_left[0] = int(floor(point_left[0]*Nel[0]))
        cell_left[1] = int(floor(point_left[1]*Nel[1]))
        cell_left[2] = int(floor(point_left[2]*Nel[2]))

        cell_number[0] = int(floor(point_right[0]*Nel[0])) - cell_left[0] + 1.0
        cell_number[1] = int(floor(point_right[1]*Nel[1])) - cell_left[1] + 1.0
        cell_number[2] = int(floor(point_right[2]*Nel[2])) - cell_left[2] + 1.0


        #======================================
        for il1 in range(cell_number[0]):
            for il2 in range(cell_number[1]):
                for il3 in range(cell_number[2]):
                    for q1 in range(n_quad[0]):
                        for q2 in range(n_quad[1]):
                            for q3 in range(n_quad[2]):
                                temp1[0] = (cell_left[0] + il1)/Nel[0] + pts1[0,q1] # quadrature points in the cell x direction
                                temp4[0] = abs(temp1[0] - eta1) - compact[0]/2 # if > 0, result is 0

                                temp1[1] = (cell_left[1] + il2)/Nel[1] + pts2[0,q2] 
                                temp4[1] = abs(temp1[1] - eta2) - compact[1]/2 # if > 0, result is 0

                                temp1[2] = (cell_left[2] + il3)/Nel[2] + pts3[0,q3] 
                                temp4[2] = abs(temp1[2] - eta3) - compact[2]/2 # if > 0, result is 0

                                if temp4[0] < 0 and temp4[1] < 0 and temp4[2] < 0:

                                    value_x = bsp.piecewise(p_shape[0], p_size[0], temp1[0] - eta1)
                                    value_y = bsp.piecewise(p_shape[1], p_size[1], temp1[1] - eta2)
                                    value_z = bsp.piecewise(p_shape[2], p_size[2], temp1[2] - eta3)
                                    d_value_x = bsp.piecewise_der(p_shape[0], p_size[0], temp1[0] - eta1)
                                    d_value_y = bsp.piecewise_der(p_shape[1], p_size[1], temp1[1] - eta2)
                                    d_value_z = bsp.piecewise_der(p_shape[2], p_size[2], temp1[2] - eta3)

                                    value[0] = d_value_x * value_y * value_z
                                    value[1] = value_x * d_value_y * value_z
                                    value[2] = value_x * value_y * d_value_z

                                    ww = weight[index_shapex[cell_left[0] + il1 + index_diffx], index_shapey[cell_left[1] + il2 + index_diffy], index_shapez[cell_left[2] + il3 + index_diffz], q1, q2, q3]

                                    temp6[0] = dfinv_t[0,0]*value[0] + dfinv_t[0,1]*value[1] + dfinv_t[0,2]*value[2] 
                                    temp6[1] = dfinv_t[1,0]*value[0] + dfinv_t[1,1]*value[1] + dfinv_t[1,2]*value[2] 
                                    temp6[2] = dfinv_t[2,0]*value[0] + dfinv_t[2,1]*value[1] + dfinv_t[2,2]*value[2] 

        #                            # check weight_123 index
                                    particles_loc[3,ip] += dt * ww * thermal * weight_p * temp6[0]
                                    particles_loc[4,ip] += dt * ww * thermal * weight_p * temp6[1]
                                    particles_loc[5,ip] += dt * ww * thermal * weight_p * temp6[2]

    #$ omp end do
    #$ omp end parallel
    ierr = 0


