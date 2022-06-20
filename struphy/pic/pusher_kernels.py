import struphy.geometry.mappings_3d_fast as mapping_fast

import struphy.feec.bsplines_kernels as bsp
import struphy.feec.basics.spline_evaluation_2d as eva2
import struphy.feec.basics.spline_evaluation_3d as eva3

import struphy.linear_algebra.core as linalg


# ==========================================================================================================
def pusher_v_mhd_electric(particles : 'float[:,:]', dt : 'float', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', nel : 'int[:]', nbase_n : 'int[:]', nbase_d : 'int[:]', np : 'int', b2_1 : 'float[:,:,:]', b2_2 : 'float[:,:,:]', b2_3 : 'float[:,:,:]', b0 : 'float[:,:,:]', u1 : 'float[:,:,:]', u2 : 'float[:,:,:]', u3 : 'float[:,:,:]', basis_u : 'int', kind_map : 'int', params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mu : 'float[:]'):
    
    from numpy import empty, zeros, sin, cos, pi
    
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
    
    # p + 1 non-vanishing derivatives
    der1 = empty(pn1 + 1, dtype=float)
    der2 = empty(pn2 + 1, dtype=float)
    der3 = empty(pn3 + 1, dtype=float)
    
    # non-vanishing N-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)
    
    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)
    
    # magnetic field, velocity field and electric field at particle position
    u           = empty( 3, dtype=float)
    b           = empty( 3, dtype=float)
    b_grad      = empty( 3, dtype=float)
    
    u_cart      = empty( 3, dtype=float)
    b_cart      = empty( 3, dtype=float)
    b_grad_cart = empty( 3, dtype=float)
    
    e_cart      = empty( 3, dtype=float)
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
    dfinv_t = empty((3, 3), dtype=float)
    # ==========================================================
    
    
    #$ omp parallel private(ip, eta1, eta2, eta3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df, dfinv, dfinv_t, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, der1, der2, der3, bn1, bn2, bn3, bd1, bd2, bd3, u, u_cart, b, b_cart, b_grad, b_grad_cart, e_cart)
    #$ omp for 
    for ip in range(np):
        
        # only do something if particle is inside the logical domain (s < 1)
        if particles[0, ip] > 1.0:
            continue
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)
        
        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)
        # ==========================================
        
        
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3
        
        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, eta1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, eta2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, eta3, span3, l3, r3, b3, d3, der3)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]
        
        # velocity field (0-form, push-forward with df)
        if basis_u == 0:
            
            u[0] = eva3.evaluation_kernel_3d(pn1, pn2, pn3, bn1, bn2, bn3, span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pn1, pn2, pn3, bn1, bn2, bn3, span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pn1, pn2, pn3, bn1, bn2, bn3, span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], u3)
            
            linalg.matrix_vector(df, u, u_cart)
        
        # velocity field (1-form, push forward with df^(-T))
        elif basis_u == 1:
            
            u[0] = eva3.evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
            
            linalg.matrix_vector(dfinv_t, u, u_cart)
        
        # velocity field (2-form, push forward with df/|det df|)
        elif basis_u == 2:
            
            u[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)
            
            linalg.matrix_vector(df, u, u_cart)
            
            u_cart[0] = u_cart[0]/det_df
            u_cart[1] = u_cart[1]/det_df
            u_cart[2] = u_cart[2]/det_df
            
        
        # magnetic field (2-form)
        b[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], b2_1)
        b[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], b2_2)
        b[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], b2_3)
        
        # push-forward to physical domain 
        linalg.matrix_vector(df, b, b_cart)
        
        b_cart[0] = b_cart[0]/det_df
        b_cart[1] = b_cart[1]/det_df
        b_cart[2] = b_cart[2]/det_df
        
        
        # gradient of absolute value of magnetic field (1-form)
        b_grad[0] = eva3.evaluation_kernel_3d(pn1, pn2, pn3, der1, bn2, bn3, span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], b0)
        b_grad[1] = eva3.evaluation_kernel_3d(pn1, pn2, pn3, bn1, der2, bn3, span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], b0)
        b_grad[2] = eva3.evaluation_kernel_3d(pn1, pn2, pn3, bn1, bn2, der3, span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], b0)
            
        # push-forward to physical domain 
        linalg.matrix_vector(dfinv_t, b_grad, b_grad_cart)
       
        
        # electric field B x U
        linalg.cross(b_cart, u_cart, e_cart)
        
        # additional artificial electric field if Pauli particles are used
        e_cart[:] = e_cart - mu[ip]*b_grad_cart
        # ==========================================
        
        
        # ======== particle pushing ================
        particles[3, ip] += dt*e_cart[0]
        particles[4, ip] += dt*e_cart[1]
        particles[5, ip] += dt*e_cart[2]
        # ==========================================
        
    #$ omp end parallel
    
    ierr = 0
    
    
    
# ==========================================================================================================
def pusher_vxb_implicit(particles : 'float[:,:]', dt : 'float', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', nel : 'int[:]', nbase_n : 'int[:]', nbase_d : 'int[:]', np : 'int', b2_1 : 'float[:,:,:]', b2_2 : 'float[:,:,:]', b2_3 : 'float[:,:,:]', kind_map : 'int', params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]'):
    
    from numpy import empty, zeros, sin, cos, pi
    
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
    
    # non-vanishing N-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)
    
    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)
    
    # magnetic field at particle position and velocity
    b      = empty( 3    , dtype=float)
    b_prod = zeros((3, 3), dtype=float)
    v      = empty( 3    , dtype=float)
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
    dfinv_t = empty((3, 3), dtype=float)
    # ==========================================================
    
    
    # ============== for solving linear 3 x 3 system ===========
    temp_mat1 = empty((3, 3), dtype=float)
    temp_mat2 = empty((3, 3), dtype=float)
    
    rhs       = empty( 3    , dtype=float)
    lhs       = empty((3, 3), dtype=float)
    lhs1      = empty((3, 3), dtype=float)
    lhs2      = empty((3, 3), dtype=float)
    lhs3      = empty((3, 3), dtype=float)
    
    identity  = zeros((3, 3), dtype=float)
    
    identity[0, 0] = 1.
    identity[1, 1] = 1.
    identity[2, 2] = 1.
    # ===========================================================
    
    
    #$ omp parallel private(ip, eta1, eta2, eta3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df, dfinv, dfinv_t, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, bn1, bn2, bn3, bd1, bd2, bd3, b, v, temp_mat1, temp_mat2, rhs, lhs, det_lhs, lhs1, lhs2, lhs3, det_lhs1, det_lhs2, det_lhs3) firstprivate(b_prod)
    #$ omp for 
    for ip in range(np):
        
        # only do something if particle is inside the logical domain (s < 1)
        if particles[0, ip] > 1.0:
            continue
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)
        
        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)
        # ==========================================
        
         
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3
        
        # evaluation of basis functions
        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]
        
        
        # magnetic field (2-form)
        b[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], b2_1)
        b[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], b2_2)
        b[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], b2_3)
            
        
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] =  b[1]

        b_prod[1, 0] =  b[2]
        b_prod[1, 2] = -b[0]

        b_prod[2, 0] = -b[1]
        b_prod[2, 1] =  b[0]
        # ==========================================
        
        
        # ======== particle pushing ================
        v[:] = particles[3:6, ip]
        
        # perform matrix-matrix and matrix-vector multiplications
        linalg.matrix_matrix(b_prod, dfinv, temp_mat1)
        linalg.matrix_matrix(dfinv_t, temp_mat1, temp_mat2)
        
        # explicit part of update rule
        linalg.matrix_vector(identity - dt/2*temp_mat2, v, rhs)
        
        # implicit part of update rule
        lhs = identity + dt/2*temp_mat2
        
        # solve 3 x 3 system with Cramer's rule
        det_lhs    = linalg.det(lhs)
        
        lhs1[:, 0] = rhs
        lhs1[:, 1] = lhs[:, 1]
        lhs1[:, 2] = lhs[:, 2]
        
        lhs2[:, 0] = lhs[:, 0]
        lhs2[:, 1] = rhs
        lhs2[:, 2] = lhs[:, 2]
        
        lhs3[:, 0] = lhs[:, 0]
        lhs3[:, 1] = lhs[:, 1]
        lhs3[:, 2] = rhs
        
        det_lhs1   = linalg.det(lhs1)
        det_lhs2   = linalg.det(lhs2)
        det_lhs3   = linalg.det(lhs3)
        
        # update particle velocities
        particles[3, ip] = det_lhs1/det_lhs
        particles[4, ip] = det_lhs2/det_lhs
        particles[5, ip] = det_lhs3/det_lhs
        # ==========================================
    
    #$ omp end parallel
    
    ierr = 0
    
    
    
# ==========================================================================================================
def pusher_vxb(particles : 'float[:,:]', dt : 'float', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', nel : 'int[:]', nbase_n : 'int[:]', nbase_d : 'int[:]', np : 'int', b2_1 : 'float[:,:,:]', b2_2 : 'float[:,:,:]', b2_3 : 'float[:,:,:]', kind_map : 'int', params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]'):
    
    from numpy import empty, zeros, sqrt, cos, sin, pi
    
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
    
    # non-vanishing N-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)
    
    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)
    
    # magnetic field at particle position (2-form, cartesian, normalized cartesian)
    b      = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b0     = empty(3, dtype=float)
    
    # particle velocity (cartesian, perpendicular, v x b0, b0 x vperp)
    v        = empty(3, dtype=float)
    vperp    = empty(3, dtype=float)
    vxb0     = empty(3, dtype=float)
    b0xvperp = empty(3, dtype=float)
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
    df = empty((3, 3), dtype=float)
    fx = empty( 3    , dtype=float)
    # ==========================================================
    
    
    #$ omp parallel private (ip, eta1, eta2, eta3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, bn1, bn2, bn3, bd1, bd2, bd3, b, b_cart, b_norm, b0, v, vpar, vxb0, vperp, b0xvperp)
    #$ omp for 
    for ip in range(np):
        
        # only do something if particle is inside the logical domain (s < 1)
        if particles[0, ip] > 1.0:
            continue
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)
        
        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))
        # ==========================================
        
        
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3
        
        # evaluation of basis functions
        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]
        
        # magnetic field (2-form)
        b[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], b2_1)
        b[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], b2_2)
        b[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], b2_3)
        
        # push-forward to physical domain
        linalg.matrix_vector(df, b, b_cart)
        
        b_cart[0] = b_cart[0]/det_df
        b_cart[1] = b_cart[1]/det_df
        b_cart[2] = b_cart[2]/det_df
        
        # absolute value of magnetic field
        b_norm = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)
        
        # normalized magnetic field direction
        b0[0] = b_cart[0]/b_norm
        b0[1] = b_cart[1]/b_norm
        b0[2] = b_cart[2]/b_norm
        # ==========================================
        
        
        # ======== particle pushing ================
        # particle velocity
        v[:] = particles[3:6, ip]
        
        # parallel velocity v . b0
        vpar = v[0]*b0[0] + v[1]*b0[1] + v[2]*b0[2]
        
        # perpendicular velocity b0 x (v x b0)
        linalg.cross(v, b0, vxb0)
        linalg.cross(b0, vxb0, vperp)
        
        # analytical rotation
        linalg.cross(b0, vperp, b0xvperp)
        
        particles[3:6, ip] = vpar*b0 + cos(b_norm*dt)*vperp - sin(b_norm*dt)*b0xvperp
        # ==========================================
    
    #$ omp end parallel
    
    ierr = 0



# ==========================================================================================================
def pusher_v_pressure_full(particles : 'float[:,:]', dt : 'float', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', nel : 'int[:]', nbase_n : 'int[:]', nbase_d : 'int[:]', np : 'int', u11 : 'float[:,:,:]', u12 : 'float[:,:,:]', u13 : 'float[:,:,:]', u21 : 'float[:,:,:]', u22 : 'float[:,:,:]', u23 : 'float[:,:,:]', u31 : 'float[:,:,:]', u32 : 'float[:,:,:]', u33 : 'float[:,:,:]', basis_u : 'int', kind_map : 'int', params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]'):
    
    from numpy import empty, zeros
    
    # ============== for magnetic field evaluation ============
    # spline degrees
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1
    
    # p + 1 non-vanishing basis functions up tp degree p
    b1 = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2 = empty((pn2 + 1, pn2 + 1), dtype=float)
    b3 = empty((pn3 + 1, pn3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1 = empty( pn1, dtype=float)
    l2 = empty( pn2, dtype=float)
    l3 = empty( pn3, dtype=float)
    
    r1 = empty( pn1, dtype=float)
    r2 = empty( pn2, dtype=float)
    r3 = empty( pn3, dtype=float)
    
    # scaling arrays for M-splines
    d1 = empty( pn1, dtype=float)
    d2 = empty( pn2, dtype=float)
    d3 = empty( pn3, dtype=float)
    
    # p + 1 non-vanishing derivatives
    der1 = empty(pn1 + 1, dtype=float)
    der2 = empty(pn2 + 1, dtype=float)
    der3 = empty(pn3 + 1, dtype=float)
    
    # non-vanishing N-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)
    
    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)
    
    # # velocity field at particle position
    u      = empty(3, dtype=float)
    u_cart = empty(3, dtype=float)

    # particle velocity
    v = empty(3, dtype=float)
    # ==========================================================
    
    
    # ================ for mapping evaluation ==================
    # spline degrees
    pf1  = pf[0]
    pf2  = pf[1]
    pf3  = pf[2]
    
    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f = empty((pf3 + 1, pf3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1f = empty( pf1, dtype=float)
    l2f = empty( pf2, dtype=float)
    l3f = empty( pf3, dtype=float)
    
    r1f = empty( pf1, dtype=float)
    r2f = empty( pf2, dtype=float)
    r3f = empty( pf3, dtype=float)
    
    # scaling arrays for M-splines
    d1f = empty( pf1, dtype=float)
    d2f = empty( pf2, dtype=float)
    d3f = empty( pf3, dtype=float)
    
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
    
    
    #$ omp parallel private(ip, eta1, eta2, eta3, v, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dfinv, dfinv_t, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, der1, der2, der3, bn1, bn2, bn3, bd1, bd2, bd3, u, u_cart)
    #$ omp for 
    for ip in range(np):
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        v[:] = particles[3:6, ip]
        
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
        # ==========================================
        
        
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3
        
        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, eta1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, eta2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, eta3, span3, l3, r3, b3, d3, der3)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # u1 = u11 * v[0] + u21 * v[0] + u31 * v[0]
        # u2 = u12 * v[1] + u22 * v[1] + u32 * v[1]
        # u3 = u13 * v[2] + u23 * v[2] + u33 * v[2]

        # Evaluate G.dot(X_dot(u) at the particle positions
        if basis_u == 1:

            u[0] = eva3.evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u11 * v[0] + u21 * v[1] + u31 * v[2])
            u[1] = eva3.evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u12 * v[0] + u22 * v[1] + u32 * v[2])
            u[2] = eva3.evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u13 * v[0] + u23 * v[1] + u33 * v[2])

        elif basis_u == 2:

            u[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2- 1, span3- 1, nbase_n[0], nbase_d[1], nbase_d[2], u11 * v[0] + u21 * v[1] + u31 * v[2])
            u[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1- 1, span2, span3- 1, nbase_d[0], nbase_n[1], nbase_d[2], u12 * v[0] + u22 * v[1] + u32 * v[2])
            u[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1- 1, span2- 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u13 * v[0] + u23 * v[1] + u33 * v[2])
        
        linalg.matrix_vector(dfinv_t, u, u_cart)
        # ==========================================
        
        
        # ======== particle pushing ================
        # particles[3, ip] -= dt*(u_cart[0,0]*v[0] + u_cart[0,1]*v[1] + u_cart[0,2]*v[2])
        # particles[4, ip] -= dt*(u_cart[1,0]*v[0] + u_cart[1,1]*v[1] + u_cart[1,2]*v[2])
        # particles[5, ip] -= dt*(u_cart[2,0]*v[0] + u_cart[2,1]*v[1] + u_cart[2,2]*v[2])
        particles[3, ip] -= dt*u_cart[0]/2
        particles[4, ip] -= dt*u_cart[1]/2
        particles[5, ip] -= dt*u_cart[2]/2
        # ==========================================
        
    #$ omp end parallel
    
    ierr = 0
    
    
# ==========================================================================================================
def pusher_v_pressure_perp(particles : 'float[:,:]', dt : 'float', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', nel : 'int[:]', nbase_n : 'int[:]', nbase_d : 'int[:]', np : 'int', u11 : 'float[:,:,:]', u12 : 'float[:,:,:]', u13 : 'float[:,:,:]', u21 : 'float[:,:,:]', u22 : 'float[:,:,:]', u23 : 'float[:,:,:]', u31 : 'float[:,:,:]', u32 : 'float[:,:,:]', u33 : 'float[:,:,:]', basis_u : 'int', kind_map : 'int', params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]'):
    
    from numpy import empty, zeros
    
    # ============== for magnetic field evaluation ============
    # spline degrees
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1
    
    # p + 1 non-vanishing basis functions up tp degree p
    b1 = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2 = empty((pn2 + 1, pn2 + 1), dtype=float)
    b3 = empty((pn3 + 1, pn3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1 = empty( pn1, dtype=float)
    l2 = empty( pn2, dtype=float)
    l3 = empty( pn3, dtype=float)
    
    r1 = empty( pn1, dtype=float)
    r2 = empty( pn2, dtype=float)
    r3 = empty( pn3, dtype=float)
    
    # scaling arrays for M-splines
    d1 = empty( pn1, dtype=float)
    d2 = empty( pn2, dtype=float)
    d3 = empty( pn3, dtype=float)
    
    # p + 1 non-vanishing derivatives
    der1 = empty(pn1 + 1, dtype=float)
    der2 = empty(pn2 + 1, dtype=float)
    der3 = empty(pn3 + 1, dtype=float)
    
    # non-vanishing N-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)
    
    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)
    
    # # velocity field at particle position
    u      = empty(3, dtype=float)
    u_cart = empty(3, dtype=float)

    # particle velocity
    v = empty(3, dtype=float)
    # ==========================================================
    
    
    # ================ for mapping evaluation ==================
    # spline degrees
    pf1  = pf[0]
    pf2  = pf[1]
    pf3  = pf[2]
    
    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f = empty((pf3 + 1, pf3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1f = empty( pf1, dtype=float)
    l2f = empty( pf2, dtype=float)
    l3f = empty( pf3, dtype=float)
    
    r1f = empty( pf1, dtype=float)
    r2f = empty( pf2, dtype=float)
    r3f = empty( pf3, dtype=float)
    
    # scaling arrays for M-splines
    d1f = empty( pf1, dtype=float)
    d2f = empty( pf2, dtype=float)
    d3f = empty( pf3, dtype=float)
    
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
    
    
    #$ omp parallel private(ip, eta1, eta2, eta3, v, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dfinv, dfinv_t, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, der1, der2, der3, bn1, bn2, bn3, bd1, bd2, bd3, u, u_cart)
    #$ omp for 
    for ip in range(np):
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        v[:] = particles[3:6, ip]
        
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
        # ==========================================
        
        
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3
        
        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, eta1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, eta2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, eta3, span3, l3, r3, b3, d3, der3)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # Evaluate G.dot(X_dot(u) at the particle positions
        if basis_u == 1:

            u[0] = eva3.evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u21 * v[1] + u31 * v[2])
            u[1] = eva3.evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u22 * v[1] + u32 * v[2])
            u[2] = eva3.evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u23 * v[1] + u33 * v[2])

        elif basis_u == 2:

            u[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2- 1, span3- 1, nbase_n[0], nbase_d[1], nbase_d[2], + u21 * v[1] + u31 * v[2])
            u[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1- 1, span2, span3- 1, nbase_d[0], nbase_n[1], nbase_d[2], + u22 * v[1] + u32 * v[2])
            u[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1- 1, span2- 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], + u23 * v[1] + u33 * v[2])
        
        linalg.matrix_vector(dfinv_t, u, u_cart)
        # ==========================================
        
        
        # ======== particle pushing ================
        particles[3, ip] -= dt*u_cart[0]/2
        particles[4, ip] -= dt*u_cart[1]/2
        particles[5, ip] -= dt*u_cart[2]/2
        # ==========================================
        
    #$ omp end parallel
    
    ierr = 0


# ==========================================================================================================
def pusher_v_cold_plasma(particles : 'float[:,:]', dt : 'float', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', nel : 'int[:]', nbase_n : 'int[:]', nbase_d : 'int[:]', np : 'int', e1 : 'float[:,:,:]', e2 : 'float[:,:,:]', e3 : 'float[:,:,:]', basis_u : 'int', kind_map : 'int', params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]'):
    
    from numpy import empty, zeros, sin, cos, pi
    
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
    
    # p + 1 non-vanishing derivatives
    der1 = empty(pn1 + 1, dtype=float)
    der2 = empty(pn2 + 1, dtype=float)
    der3 = empty(pn3 + 1, dtype=float)
    
    # non-vanishing N-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)
    
    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)
    
    # electric field at particle position
    e           = empty( 3, dtype=float)
    e_cart      = empty( 3, dtype=float)
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
    dfinv_t = empty((3, 3), dtype=float)
    # ==========================================================
    
    
    #$ omp parallel private(ip, eta1, eta2, eta3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dfinv, dfinv_t, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, der1, der2, der3, bn1, bn2, bn3, bd1, bd2, bd3, e, e_cart)
    #$ omp for 
    for ip in range(np):
        
        # only do something if particle is inside the logical domain (s < 1)
        if particles[0, ip] > 1.0:
            continue
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)
        
  
        # evaluate inverse transosed Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
        linalg.transpose(dfinv, dfinv_t)
        # ==========================================
        
        
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3
        
        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, eta1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, eta2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, eta3, span3, l3, r3, b3, d3, der3)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]
        
        # velocity field (1-form, push forward with df^(-T))
        if basis_u == 1:
            
            e[0] = eva3.evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], e1)
            e[1] = eva3.evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], e2)
            e[2] = eva3.evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], e3)
            
            linalg.matrix_vector(dfinv_t, e, e_cart)

        # print(dt)
        # print(e_cart[0].max())
        # print(e_cart[1].max())
        # print(e_cart[2].max())
  
        
        # ======== particle pushing ================
        particles[3, ip] += (dt/2.)*e_cart[0]
        particles[4, ip] += (dt/2.)*e_cart[1]
        particles[5, ip] += (dt/2.)*e_cart[2]
        # ==========================================
        
    #$ omp end parallel
    
    ierr = 0


# ==========================================================================================================
def pusher_v_mhd_electric(particles : 'float[:,:]', dt : 'float', t1 : 'float[:]', t2 : 'float[:]', p : 'int[:]', nel : 'int[:]', nbase_n : 'int[:]', nbase_d : 'int[:]', np : 'int', b_eq_1 : 'float[:,:,:]', b_eq_2 : 'float[:,:,:]', b_eq_3 : 'float[:,:,:]', b_p_1 : 'float[:,:,:]', b_p_2 : 'float[:,:,:]', b_p_3 : 'float[:,:,:]', b_norm : 'float[:,:,:]', u1 : 'float[:,:,:]', u2 : 'float[:,:,:]', u3 : 'float[:,:,:]', basis_u : 'int', kind_map : 'int', params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', mu : 'float[:]', n_tor : 'int'):
    
    from numpy import empty, zeros, sin, cos, pi
    
    # ============== for magnetic field evaluation ============
    # spline degrees
    pn1 = p[0]
    pn2 = p[1]
    
    pd1 = pn1 - 1
    pd2 = pn2 - 1
    
    # p + 1 non-vanishing basis functions up tp degree p
    b1  = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2  = empty((pn2 + 1, pn2 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1  = empty( pn1, dtype=float)
    l2  = empty( pn2, dtype=float)
    
    r1  = empty( pn1, dtype=float)
    r2  = empty( pn2, dtype=float)
    
    # scaling arrays for M-splines
    d1  = empty( pn1, dtype=float)
    d2  = empty( pn2, dtype=float)
    
    # p + 1 non-vanishing derivatives
    der1 = empty(pn1 + 1, dtype=float)
    der2 = empty(pn2 + 1, dtype=float)
    
    # non-vanishing N-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    
    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    
    # cos/sin at particle position
    cs = empty(2, dtype=float)
    
    # magnetic field, velocity field and electric field at particle position
    u           = empty(3, dtype=float)
    b           = empty(3, dtype=float)
    b_grad      = empty(3, dtype=float)
    
    u_cart      = empty(3, dtype=float)
    b_cart      = empty(3, dtype=float)
    b_grad_cart = empty(3, dtype=float)
    
    e_cart      = empty(3, dtype=float)
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
    dfinv_t = empty((3, 3), dtype=float)
    # ==========================================================
    
    
    #$ omp parallel private(ip, eta1, eta2, eta3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df, dfinv, dfinv_t, span1, span2, l1, l2, r1, r2, b1, b2, d1, d2, der1, der2, bn1, bn2, bd1, bd2, cs, u, u_cart, b, b_cart, b_grad, b_grad_cart, e_cart)
    #$ omp for 
    for ip in range(np):
        
        # only do something if particle is inside the logical domain (s < 1)
        if particles[0, ip] > 1.0:
            continue
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)
        
        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)
        # ==========================================
        
        
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        
        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, eta1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, eta2, span2, l2, r2, b2, d2, der2)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        
        # cos/sin at particle position
        cs[0] = cos(2*pi*n_tor*eta3)
        cs[1] = sin(2*pi*n_tor*eta3)
        
        
        # velocity field (0-form, push-forward with df)
        if basis_u == 0:
            
            u[:] = 0.
            
            for i in range(nbase_n[2]):
                
                u[0] += eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1, span2, nbase_n[0], nbase_n[1], u1[:, :, i])*cs[i]
                u[1] += eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1, span2, nbase_n[0], nbase_n[1], u2[:, :, i])*cs[i]
                u[2] += eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1, span2, nbase_n[0], nbase_n[1], u3[:, :, i])*cs[i]
            
            linalg.matrix_vector(df, u, u_cart)
        
        # velocity field (1-form, push forward with df^(-T))
        elif basis_u == 1:
            
            u[:] = 0.
            
            for i in range(nbase_n[2]):
            
                u[0] += eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], u1[:, :, i])*cs[i]
                u[1] += eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], u2[:, :, i])*cs[i]
                u[2] += eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1 - 0, span2 - 0, nbase_n[0], nbase_n[1], u3[:, :, i])*cs[i]
            
            linalg.matrix_vector(dfinv_t, u, u_cart)
        
        # velocity field (2-form, push forward with df/|det df|)
        elif basis_u == 2:
            
            u[:] = 0.
            
            for i in range(nbase_n[2]):

                u[0] += eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], u1[:, :, i])*cs[i]
                u[1] += eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], u2[:, :, i])*cs[i]
                u[2] += eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], u3[:, :, i])*cs[i]
            
            linalg.matrix_vector(df, u, u_cart)
            
            u_cart[0] = u_cart[0]/det_df
            u_cart[1] = u_cart[1]/det_df
            u_cart[2] = u_cart[2]/det_df
            
        
        # equilibrium magnetic field (2-form)
        b[0] = eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_eq_1[:, :, 0])
        b[1] = eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_eq_2[:, :, 0])
        b[2] = eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_eq_3[:, :, 0])

        # perturbed magnetic field (2-form)
        for i in range(nbase_n[2]):
            
            b[0] += eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_p_1[:, :, i])*cs[i]
            b[1] += eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_p_2[:, :, i])*cs[i]
            b[2] += eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_p_3[:, :, i])*cs[i]
        
        # push-forward to physical domain 
        linalg.matrix_vector(df, b, b_cart)
        
        b_cart[0] = b_cart[0]/det_df
        b_cart[1] = b_cart[1]/det_df
        b_cart[2] = b_cart[2]/det_df
        
        
        # gradient of absolute value of magnetic field (1-form)
        b_grad[0] = eva2.evaluation_kernel_2d(pn1, pn2, der1, bn2, span1, span2, nbase_n[0], nbase_n[1], b_norm[:, :, 0])
        b_grad[1] = eva2.evaluation_kernel_2d(pn1, pn2, bn1, der2, span1, span2, nbase_n[0], nbase_n[1], b_norm[:, :, 0])
        b_grad[2] = 0.
            
        # push-forward to physical domain 
        linalg.matrix_vector(dfinv_t, b_grad, b_grad_cart)
       
        
        # electric field B x U
        linalg.cross(b_cart, u_cart, e_cart)
        
        # additional artificial electric field if Pauli particles are used
        e_cart[:] = e_cart - mu[ip]*b_grad_cart
        # ==========================================
        
        
        # ======== particle pushing ================
        particles[3, ip] += dt*e_cart[0]
        particles[4, ip] += dt*e_cart[1]
        particles[5, ip] += dt*e_cart[2]
        # ==========================================
        
    #$ omp end parallel
    
    ierr = 0
    
    
    

# ==========================================================================================================
def pusher_vxb(particles : 'float[:,:]', dt : 'float', t1 : 'float[:]', t2 : 'float[:]', p : 'int[:]', nel : 'int[:]', nbase_n : 'int[:]', nbase_d : 'int[:]', np : 'int', b_eq_1 : 'float[:,:,:]', b_eq_2 : 'float[:,:,:]', b_eq_3 : 'float[:,:,:]', b_p_1 : 'float[:,:,:]', b_p_2 : 'float[:,:,:]', b_p_3 : 'float[:,:,:]', kind_map : 'int', params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', n_tor : 'int'):
    
    from numpy import empty, zeros, sqrt, cos, sin, pi
    
    # ============== for magnetic field evaluation ============
    # spline degrees
    pn1 = p[0]
    pn2 = p[1]
    
    pd1 = pn1 - 1
    pd2 = pn2 - 1
    
    # p + 1 non-vanishing basis functions up tp degree p
    b1  = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2  = empty((pn2 + 1, pn2 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1  = empty( pn1, dtype=float)
    l2  = empty( pn2, dtype=float)
    
    r1  = empty( pn1, dtype=float)
    r2  = empty( pn2, dtype=float)
    
    # scaling arrays for M-splines
    d1  = empty( pn1, dtype=float)
    d2  = empty( pn2, dtype=float)
    
    # non-vanishing N-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    
    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    
    # cos/sin at particle position
    cs = empty(2, dtype=float)
    
    # magnetic field at particle position (2-form, cartesian, normalized cartesian)
    b      = empty(3, dtype=float)
    b_cart = empty(3, dtype=float)
    b0     = empty(3, dtype=float)
    
    # particle velocity (cartesian, perpendicular, v x b0, b0 x vperp)
    v        = empty(3, dtype=float)
    vperp    = empty(3, dtype=float)
    vxb0     = empty(3, dtype=float)
    b0xvperp = empty(3, dtype=float)
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
    fx = empty( 3    , dtype=float)
    df = empty((3, 3), dtype=float)
    # ==========================================================
    
    
    #$ omp parallel private(ip, eta1, eta2, eta3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df, span1, span2, l1, l2, r1, r2, b1, b2, d1, d2, bn1, bn2, bd1, bd2, cs, b, b_cart, b_norm, b0, v, vpar, vxb0, vperp, b0xvperp)
    #$ omp for 
    for ip in range(np):
        
        # only do something if particle is inside the logical domain (s < 1)
        if particles[0, ip] > 1.0:
            continue
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)
        
        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))
        # ==========================================
        
        
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        
        # evaluation of basis functions
        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        
        # cos/sin at particle position
        cs[0] = cos(2*pi*n_tor*eta3)
        cs[1] = sin(2*pi*n_tor*eta3)
        
        # equilibrium magnetic field (2-form)
        b[0] = eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_eq_1[:, :, 0])
        b[1] = eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_eq_2[:, :, 0])
        b[2] = eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_eq_3[:, :, 0])

        # perturbed magnetic field (2-form)
        for i in range(nbase_n[2]):
            
            b[0] += eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_p_1[:, :, i])*cs[i]
            b[1] += eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_p_2[:, :, i])*cs[i]
            b[2] += eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_p_3[:, :, i])*cs[i]
        
        # push-forward to physical domain
        linalg.matrix_vector(df, b, b_cart)
        
        b_cart[0] = b_cart[0]/det_df
        b_cart[1] = b_cart[1]/det_df
        b_cart[2] = b_cart[2]/det_df
        
        # absolute value of magnetic field
        b_norm = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)
        
        # normalized magnetic field direction
        b0[0] = b_cart[0]/b_norm
        b0[1] = b_cart[1]/b_norm
        b0[2] = b_cart[2]/b_norm
        # ==========================================
        
        
        # ======== particle pushing ================
        # particle velocity
        v[:] = particles[3:6, ip]
        
        # parallel velocity v . b0
        vpar = v[0]*b0[0] + v[1]*b0[1] + v[2]*b0[2]
        
        # perpendicular velocity b0 x (v x b0)
        linalg.cross(v, b0, vxb0)
        linalg.cross(b0, vxb0, vperp)
        
        # analytical rotation
        linalg.cross(b0, vperp, b0xvperp)
        
        particles[3:6, ip] = vpar*b0 + cos(b_norm*dt)*vperp - sin(b_norm*dt)*b0xvperp
        # ==========================================
    
    #$ omp end parallel
    
    ierr = 0


# ==========================================================================================================
def pusher_rk4(particles : 'float[:,:]', dt : 'float', np : 'int', kind_map : 'int', params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]'):
    
    from numpy import empty, sqrt, arctan2, pi, cos, sin
    
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
    df    = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    fx    = empty( 3    , dtype=float)
    # ========================================================
    
    
    # ======= particle position and velocity =================
    eta = empty(3, dtype=float)
    v   = empty(3, dtype=float)
    # ========================================================
    
    
    # ===== intermediate stps in 4th order Runge-Kutta =======
    k1 = empty(3, dtype=float)  
    k2 = empty(3, dtype=float)  
    k3 = empty(3, dtype=float)  
    k4 = empty(3, dtype=float) 
    # ========================================================
    
    
    #$ omp parallel private(ip, eta, v, pos1, pos2, pos3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dfinv, k1, k2, k3, k4)
    #$ omp for 
    for ip in range(np):
        
        eta[:] = particles[0:3, ip]
        v[:]   = particles[3:6, ip]
        
        # ----------- step 1 in Runge-Kutta method -----------------------
        pos1   = eta[0]
        pos2   = eta[1]
        pos3   = eta[2]
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
                
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k1)
        # ------------------------------------------------------------------
        
        
        # ----------------- step 2 in Runge-Kutta method -------------------
        pos1   = (eta[0] + dt*k1[0]/2)%1.
        pos2   = (eta[1] + dt*k1[1]/2)%1.
        pos3   = (eta[2] + dt*k1[2]/2)%1.
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
                
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k2)
        # ------------------------------------------------------------------
        
        
        # ------------------ step 3 in Runge-Kutta method ------------------
        pos1   = (eta[0] + dt*k2[0]/2)%1.
        pos2   = (eta[1] + dt*k2[1]/2)%1.
        pos3   = (eta[2] + dt*k2[2]/2)%1.
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
                
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k3)
        # ------------------------------------------------------------------
        
        
        # ------------------ step 4 in Runge-Kutta method ------------------
        pos1   = (eta[0] + dt*k3[0])%1.
        pos2   = (eta[1] + dt*k3[1])%1.
        pos3   = (eta[2] + dt*k3[2])%1.
        
        span1f = int(pos1*nelf[0]) + pf1
        span2f = int(pos2*nelf[1]) + pf2
        span3f = int(pos3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
                
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k4)
        # ------------------------------------------------------------------
        
        
        #  ---------------- update logical coordinates ---------------------
        particles[0, ip] = (eta[0] + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6)%1.0
        particles[1, ip] = (eta[1] + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6)%1.0
        particles[2, ip] = (eta[2] + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6)%1.0
        # ------------------------------------------------------------------
    #$ omp end parallel
    
    ierr = 0
    
    
    
# ========================================================================================================    
def reflect(df : 'float[:,:]', df_inv : 'float[:,:]', v : 'float[:]'):
    
    from numpy import empty, sqrt
    
    vg        = empty( 3    , dtype=float)
    
    basis     = empty((3, 3), dtype=float)
    basis_inv = empty((3, 3), dtype=float)
    
    
    # calculate normalized basis vectors
    norm1 = sqrt(df_inv[0, 0]**2 + df_inv[0, 1]**2 + df_inv[0, 2]**2)
    
    norm2 = sqrt(df[0, 1]**2 + df[1, 1]**2 + df[2, 1]**2)
    norm3 = sqrt(df[0, 2]**2 + df[1, 2]**2 + df[2, 2]**2)
    
    basis[:, 0] = df_inv[0, :]/norm1
    
    basis[:, 1] = df[:, 1]/norm2
    basis[:, 2] = df[:, 2]/norm3
    
    linalg.matrix_inv(basis, basis_inv)
    
    linalg.matrix_vector(basis_inv, v, vg)
    
    vg[0] = -vg[0]
    
    linalg.matrix_vector(basis, vg, v)
    
    


# ==========================================================================================================
def pusher_rk4_pseudo(particles : 'float[:,:]', dt : 'float', np : 'int', kind_map : 'int', params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', a : 'float', r0 : 'float'):
    
    from numpy import empty, zeros
    import struphy.geometry.mappings_3d      as mappings_3d
    
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
    df        = empty((3, 3), dtype=float)
    dfinv     = empty((3, 3), dtype=float)
    
    df_old    = empty((3, 3), dtype=float)
    dfinv_old = empty((3, 3), dtype=float)
    
    fx        = empty( 3    , dtype=float)
    
    # needed mapping quantities for pseudo-cartesian coordinates
    df_pseudo     = empty((3, 3), dtype=float)
    
    df_pseudo_old = empty((3, 3), dtype=float)
    fx_pseudo     = empty( 3    , dtype=float)
    
    params_pseudo = empty( 3    , dtype=float)
    
    params_pseudo[0] = 0.
    params_pseudo[1] = a
    params_pseudo[2] = r0
    # ========================================================
    
    
    # ======= particle position and velocity =================
    eta    = empty(3, dtype=float)
    v      = empty(3, dtype=float)
    v_temp = empty(3, dtype=float)
    # ========================================================
    
    
    # ===== intermediate stps in 4th order Runge-Kutta =======
    k1 = empty(3, dtype=float)  
    k2 = empty(3, dtype=float)  
    k3 = empty(3, dtype=float)  
    k4 = empty(3, dtype=float) 
    # ========================================================
    
    
    #$ omp parallel private (ip, eta, v, fx_pseudo, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df_old, fx, dfinv_old, df_pseudo_old, df, dfinv, df_pseudo, v_temp, k1, k2, k3, k4)
    #$ omp for 
    for ip in range(np):
        
        # only do something if particle is inside the logical domain (s < 1)
        if particles[0, ip] > 1.0:
            continue
        
        # old logical coordinates and velocities
        eta[:] = particles[0:3, ip]
        v[:]   = particles[3:6, ip]
        
        # compute old pseudo-cartesian coordinates
        fx_pseudo[0] = mappings_3d.f(eta[0], eta[1], eta[2], 1, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        fx_pseudo[1] = mappings_3d.f(eta[0], eta[1], eta[2], 2, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        fx_pseudo[2] = mappings_3d.f(eta[0], eta[1], eta[2], 3, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
       
        # evaluate old Jacobian matrix of mapping F
        span1f = int(eta[0]*nelf[0]) + pf1
        span2f = int(eta[1]*nelf[1]) + pf2
        span3f = int(eta[2]*nelf[2]) + pf3
        
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta[0], eta[1], eta[2], df_old, fx, 0)

        # evaluate old inverse Jacobian matrix of mapping F
        mapping_fast.df_inv_all(df_old, dfinv_old)
        
        # evaluate old Jacobian matrix of mapping F_pseudo
        df_pseudo_old[0, 0] = mappings_3d.df(eta[0], eta[1], eta[2], 11, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        df_pseudo_old[0, 1] = mappings_3d.df(eta[0], eta[1], eta[2], 12, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        df_pseudo_old[0, 2] = mappings_3d.df(eta[0], eta[1], eta[2], 13, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
        df_pseudo_old[1, 0] = mappings_3d.df(eta[0], eta[1], eta[2], 21, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        df_pseudo_old[1, 1] = mappings_3d.df(eta[0], eta[1], eta[2], 22, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        df_pseudo_old[1, 2] = mappings_3d.df(eta[0], eta[1], eta[2], 23, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
        df_pseudo_old[2, 0] = mappings_3d.df(eta[0], eta[1], eta[2], 31, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        df_pseudo_old[2, 1] = mappings_3d.df(eta[0], eta[1], eta[2], 32, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        df_pseudo_old[2, 2] = mappings_3d.df(eta[0], eta[1], eta[2], 33, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
        
        while True:
            
            # ----------- step 1 in Runge-Kutta method -----------------------
            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv_old, v, v_temp)
            linalg.matrix_vector(df_pseudo_old, v_temp, k1)
            # ------------------------------------------------------------------
            
        
            # ----------------- step 2 in Runge-Kutta method -------------------
            eta[0] = mappings_3d.f_inv(fx_pseudo[0] + dt*k1[0]/2, fx_pseudo[1] + dt*k1[1]/2, fx_pseudo[2] + dt*k1[2]/2, 1, 14, params_pseudo)
            eta[1] = mappings_3d.f_inv(fx_pseudo[0] + dt*k1[0]/2, fx_pseudo[1] + dt*k1[1]/2, fx_pseudo[2] + dt*k1[2]/2, 2, 14, params_pseudo)
            eta[2] = mappings_3d.f_inv(fx_pseudo[0] + dt*k1[0]/2, fx_pseudo[1] + dt*k1[1]/2, fx_pseudo[2] + dt*k1[2]/2, 3, 14, params_pseudo)
            
            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                
                particles[6, ip] = 0.
                particles[0, ip] = 1.5
                
                break

            # evaluate Jacobian matrix of mapping F
            span1f = int(eta[0]*nelf[0]) + pf1
            span2f = int(eta[1]*nelf[1]) + pf2
            span3f = int(eta[2]*nelf[2]) + pf3
            
            mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta[0], eta[1], eta[2], df, fx, 0)
            
            # evaluate inverse Jacobian matrix of mapping F
            mapping_fast.df_inv_all(df, dfinv)

            # evaluate Jacobian matrix of mapping F_pseudo
            df_pseudo[0, 0] = mappings_3d.df(eta[0], eta[1], eta[2], 11, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[0, 1] = mappings_3d.df(eta[0], eta[1], eta[2], 12, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[0, 2] = mappings_3d.df(eta[0], eta[1], eta[2], 13, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            df_pseudo[1, 0] = mappings_3d.df(eta[0], eta[1], eta[2], 21, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[1, 1] = mappings_3d.df(eta[0], eta[1], eta[2], 22, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[1, 2] = mappings_3d.df(eta[0], eta[1], eta[2], 23, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            df_pseudo[2, 0] = mappings_3d.df(eta[0], eta[1], eta[2], 31, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[2, 1] = mappings_3d.df(eta[0], eta[1], eta[2], 32, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[2, 2] = mappings_3d.df(eta[0], eta[1], eta[2], 33, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv, v, v_temp)
            linalg.matrix_vector(df_pseudo, v_temp, k2)
            # ------------------------------------------------------------------


            # ------------------ step 3 in Runge-Kutta method ------------------
            eta[0] = mappings_3d.f_inv(fx_pseudo[0] + dt*k2[0]/2, fx_pseudo[1] + dt*k2[1]/2, fx_pseudo[2] + dt*k2[2]/2, 1, 14, params_pseudo)
            eta[1] = mappings_3d.f_inv(fx_pseudo[0] + dt*k2[0]/2, fx_pseudo[1] + dt*k2[1]/2, fx_pseudo[2] + dt*k2[2]/2, 2, 14, params_pseudo)
            eta[2] = mappings_3d.f_inv(fx_pseudo[0] + dt*k2[0]/2, fx_pseudo[1] + dt*k2[1]/2, fx_pseudo[2] + dt*k2[2]/2, 3, 14, params_pseudo)
            
            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                
                particles[6, ip] = 0.
                particles[0, ip] = 1.5
                
                break
                   
            # evaluate Jacobian matrix of mapping F
            span1f = int(eta[0]*nelf[0]) + pf1
            span2f = int(eta[1]*nelf[1]) + pf2
            span3f = int(eta[2]*nelf[2]) + pf3
            
            mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta[0], eta[1], eta[2], df, fx, 0)

            # evaluate inverse Jacobian matrix of mapping F
            mapping_fast.df_inv_all(df, dfinv)

            # evaluate Jacobian matrix of mapping F_pseudo
            df_pseudo[0, 0] = mappings_3d.df(eta[0], eta[1], eta[2], 11, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[0, 1] = mappings_3d.df(eta[0], eta[1], eta[2], 12, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[0, 2] = mappings_3d.df(eta[0], eta[1], eta[2], 13, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            df_pseudo[1, 0] = mappings_3d.df(eta[0], eta[1], eta[2], 21, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[1, 1] = mappings_3d.df(eta[0], eta[1], eta[2], 22, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[1, 2] = mappings_3d.df(eta[0], eta[1], eta[2], 23, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            df_pseudo[2, 0] = mappings_3d.df(eta[0], eta[1], eta[2], 31, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[2, 1] = mappings_3d.df(eta[0], eta[1], eta[2], 32, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[2, 2] = mappings_3d.df(eta[0], eta[1], eta[2], 33, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv, v, v_temp)
            linalg.matrix_vector(df_pseudo, v_temp, k3)
            # ------------------------------------------------------------------


            # ------------------ step 4 in Runge-Kutta method ------------------
            eta[0] = mappings_3d.f_inv(fx_pseudo[0] + dt*k3[0], fx_pseudo[1] + dt*k3[1], fx_pseudo[2] + dt*k3[2], 1, 14, params_pseudo)
            eta[1] = mappings_3d.f_inv(fx_pseudo[0] + dt*k3[0], fx_pseudo[1] + dt*k3[1], fx_pseudo[2] + dt*k3[2], 2, 14, params_pseudo)
            eta[2] = mappings_3d.f_inv(fx_pseudo[0] + dt*k3[0], fx_pseudo[1] + dt*k3[1], fx_pseudo[2] + dt*k3[2], 3, 14, params_pseudo)
            
            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                
                particles[6, ip] = 0.
                particles[0, ip] = 1.5
                
                break
                
            # evaluate Jacobian matrix of mapping F
            span1f = int(eta[0]*nelf[0]) + pf1
            span2f = int(eta[1]*nelf[1]) + pf2
            span3f = int(eta[2]*nelf[2]) + pf3
            
            mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta[0], eta[1], eta[2], df, fx, 0)

            # evaluate inverse Jacobian matrix of mapping F
            mapping_fast.df_inv_all(df, dfinv)

            # evaluate Jacobian matrix of mapping F_pseudo
            df_pseudo[0, 0] = mappings_3d.df(eta[0], eta[1], eta[2], 11, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[0, 1] = mappings_3d.df(eta[0], eta[1], eta[2], 12, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[0, 2] = mappings_3d.df(eta[0], eta[1], eta[2], 13, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            df_pseudo[1, 0] = mappings_3d.df(eta[0], eta[1], eta[2], 21, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[1, 1] = mappings_3d.df(eta[0], eta[1], eta[2], 22, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[1, 2] = mappings_3d.df(eta[0], eta[1], eta[2], 23, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            df_pseudo[2, 0] = mappings_3d.df(eta[0], eta[1], eta[2], 31, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[2, 1] = mappings_3d.df(eta[0], eta[1], eta[2], 32, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)
            df_pseudo[2, 2] = mappings_3d.df(eta[0], eta[1], eta[2], 33, 14, params_pseudo, tf1, tf2, tf3, pf, nbasef, cx, cy, cz)

            # compute df_pseudo*df_inv*v
            linalg.matrix_vector(dfinv, v, v_temp)
            linalg.matrix_vector(df_pseudo, v_temp, k4)
            # ------------------------------------------------------------------


            #  ---------------- update pseudo-cartesian coordinates ------------
            fx_pseudo[0] = fx_pseudo[0] + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6.0
            fx_pseudo[1] = fx_pseudo[1] + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6.0
            fx_pseudo[2] = fx_pseudo[2] + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6.0
            # ------------------------------------------------------------------

            # compute logical coordinates
            eta[0] = mappings_3d.f_inv(fx_pseudo[0], fx_pseudo[1], fx_pseudo[2], 1, 14, params_pseudo)
            eta[1] = mappings_3d.f_inv(fx_pseudo[0], fx_pseudo[1], fx_pseudo[2], 2, 14, params_pseudo)
            eta[2] = mappings_3d.f_inv(fx_pseudo[0], fx_pseudo[1], fx_pseudo[2], 3, 14, params_pseudo)
            
            # check if particle has left the domain at s = 1: if yes, stop iteration and set weight to zero
            if eta[0] > 1.0:
                
                particles[6, ip] = 0.
                particles[0, ip] = 1.5
                
                break

            particles[0, ip] = eta[0]
            particles[1, ip] = eta[1]
            particles[2, ip] = eta[2]
            
            # set particle velocity (will only change if particle was reflected)
            particles[3, ip] = v[0]
            particles[4, ip] = v[1]
            particles[5, ip] = v[2]
            
            break
    #$ omp end parallel
    
    ierr = 0
    
    

    
# ==========================================================================================================
def pusher_exact(particles : 'float[:,:]', dt : 'float', np : 'int', kind_map : 'int', params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]', tol : 'float'):
    
    from numpy import empty
    
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
    df    = empty((3, 3), dtype=float)
    dfinv = empty((3, 3), dtype=float)
    
    x_old = empty( 3    , dtype=float)
    x_new = empty( 3    , dtype=float)
    
    temp  = empty( 3    , dtype=float)
    # ========================================================
    
    
    # ======= particle position and velocity =================
    e = empty(3, dtype=float)
    v = empty(3, dtype=float)
    # ========================================================
    
    
    #$ omp parallel private(ip, e, v, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, x_old, x_new, dfinv, temp)
    #$ omp for 
    for ip in range(np):
        
        e[:] = particles[0:3, ip]
        v[:] = particles[3:6, ip]
        
        span1f = int(e[0]*nelf[0]) + pf1
        span2f = int(e[1]*nelf[1]) + pf2
        span3f = int(e[2]*nelf[2]) + pf3
        
        # evaluate Jacobian matrix and current Cartesian coordinates
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, e[0], e[1], e[2], df, x_old, 2)
        
        # update cartesian coordinates exactly
        x_new[0] = x_old[0] + dt*v[0]
        x_new[1] = x_old[1] + dt*v[1]
        x_new[2] = x_old[2] + dt*v[2]
        
        # calculate new logical coordinates by solving inverse mapping with Newton-method
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
        
        while True:

            x_old[:] = x_old - x_new
            linalg.matrix_vector(dfinv, x_old, temp)
            
            e[0] =  e[0] - temp[0]
            e[1] = (e[1] - temp[1])%1.0
            e[2] = (e[2] - temp[2])%1.0
            
            span1f = int(e[0]*nelf[0]) + pf1
            span2f = int(e[1]*nelf[1]) + pf2
            span3f = int(e[2]*nelf[2]) + pf3
            
            # evaluate Jacobian matrix and mapping
            mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, e[0], e[1], e[2], df, x_old, 2)
            
            if abs(x_old[0] - x_new[0]) < tol and abs(x_old[1] - x_new[1]) < tol and abs(x_old[2] - x_new[2]) < tol:
                particles[0:3, ip] = e
                break
            
            # evaluate inverse Jacobian matrix
            mapping_fast.df_inv_all(df, dfinv)
    #$ omp end parallel
    
    ierr = 0



# ==========================================================================================================
def pusher_rk4_pc_full(particles : 'float[:,:]', dt : 'float', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', nel : 'int[:]', nbase_n : 'int[:]', nbase_d : 'int[:]', np : 'int', u1 : 'float[:,:,:]', u2 : 'float[:,:,:]', u3 : 'float[:,:,:]', basis_u : 'int', kind_map : 'int', params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]'):
    
    from numpy import empty

    #============== for velocity evaluation ============
    # spline degrees
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1
    
    # p + 1 non-vanishing basis functions up tp degree p
    b1 = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2 = empty((pn2 + 1, pn2 + 1), dtype=float)
    b3 = empty((pn3 + 1, pn3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1 = empty( pn1, dtype=float)
    l2 = empty( pn2, dtype=float)
    l3 = empty( pn3, dtype=float)
    
    r1 = empty( pn1, dtype=float)
    r2 = empty( pn2, dtype=float)
    r3 = empty( pn3, dtype=float)
    
    # scaling arrays for M-splines
    d1 = empty( pn1, dtype=float)
    d2 = empty( pn2, dtype=float)
    d3 = empty( pn3, dtype=float)
    
    # p + 1 non-vanishing derivatives
    der1 = empty(pn1 + 1, dtype=float)
    der2 = empty(pn2 + 1, dtype=float)
    der3 = empty(pn3 + 1, dtype=float)
    
    # non-vanishing N-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)
    
    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # # velocity field at particle position
    u = empty(3, dtype=float)
    # ==========================================================


    # ================ for mapping evaluation ==================
    # spline degrees
    pf1 = pf[0]
    pf2 = pf[1]
    pf3 = pf[2]
    
    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f = empty((pf3 + 1, pf3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1f = empty( pf1, dtype=float)
    l2f = empty( pf2, dtype=float)
    l3f = empty( pf3, dtype=float)
    
    r1f = empty( pf1, dtype=float)
    r2f = empty( pf2, dtype=float)
    r3f = empty( pf3, dtype=float)
    
    # scaling arrays for M-splines
    d1f = empty( pf1, dtype=float)
    d2f = empty( pf2, dtype=float)
    d3f = empty( pf3, dtype=float)
    
    # pf + 1 derivatives
    der1f = empty( pf1 + 1, dtype=float)
    der2f = empty( pf2 + 1, dtype=float)
    der3f = empty( pf3 + 1, dtype=float)
    
    # needed mapping quantities
    df      = empty((3, 3), dtype=float)
    dfinv   = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    Ginv    = empty((3, 3), dtype=float)
    fx      = empty( 3    , dtype=float)
    # ========================================================
    
    
    # ======= particle position and velocity =================
    eta = empty(3, dtype=float)
    v   = empty(3, dtype=float)
    # ========================================================
    
    
    # ===== intermediate stps in 4th order Runge-Kutta =======
    k1   = empty(3, dtype=float)  
    k2   = empty(3, dtype=float)  
    k3   = empty(3, dtype=float)  
    k4   = empty(3, dtype=float)
    k1_u = empty(3, dtype=float)  
    k2_u = empty(3, dtype=float)  
    k3_u = empty(3, dtype=float)  
    k4_u = empty(3, dtype=float) 
    k1_v = empty(3, dtype=float)  
    k2_v = empty(3, dtype=float)  
    k3_v = empty(3, dtype=float)  
    k4_v = empty(3, dtype=float)  
    # ========================================================
    
    
    #$ omp parallel private(ip, eta, v, pos1, pos2, pos3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dfinv, dfinv_t, Ginv, det_df, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, der1, der2, der3, bn1, bn2, bn3, bd1, bd2, bd3, u, k1, k2, k3, k4, k1_u, k2_u, k3_u, k4_u, k1_v, k2_v, k3_v, k4_v)
    #$ omp for 
    for ip in range(np):
        
        eta[:] = particles[0:3, ip]
        v[:]   = particles[3:6, ip]

        # ----------- step 1 in Runge-Kutta method -----------------------
        # ========= mapping evaluation =============
        pos1 = eta[0]
        pos2 = eta[1]
        pos3 = eta[2]

        span1f = int(eta[0]*nelf[0]) + pf1
        span2f = int(eta[1]*nelf[1]) + pf2
        span3f = int(eta[2]*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv) ###########
        # ============================================

        # ========== field evaluation ==============
        span1 = int(pos1*nel[0]) + pn1
        span2 = int(pos2*nel[1]) + pn2
        span3 = int(pos3*nel[2]) + pn3
        
        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
            
            linalg.matrix_vector(Ginv, u, k1_u)
            
        elif basis_u ==2:
            u[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)

            k1_u[:] = u/det_df
        
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k1_v)

        k1[:] = k1_v[:] + k1_u[:]

        # ------------------------------------------------------------------
        
        
        # ----------------- step 2 in Runge-Kutta method -------------------
        pos1   = (eta[0] + dt*k1[0]/2)%1.
        pos2   = (eta[1] + dt*k1[1]/2)%1.
        pos3   = (eta[2] + dt*k1[2]/2)%1.
        
        # ========= mapping evaluation =============
        span1f = int(eta[0]*nelf[0]) + pf1
        span2f = int(eta[1]*nelf[1]) + pf2
        span3f = int(eta[2]*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
        # ============================================

        # ========== field evaluation ==============
        span1 = int(pos1*nel[0]) + pn1
        span2 = int(pos2*nel[1]) + pn2
        span3 = int(pos3*nel[2]) + pn3
        
        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
            
            linalg.matrix_vector(Ginv, u, k2_u)
            
        elif basis_u ==2:
            u[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)

            k2_u[:] = u/det_df
        
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k2_v)

        k2[:] = k2_v[:] + k2_u[:]
        # ------------------------------------------------------------------
        
        
        # ------------------ step 3 in Runge-Kutta method ------------------
        pos1   = (eta[0] + dt*k2[0]/2)%1.
        pos2   = (eta[1] + dt*k2[1]/2)%1.
        pos3   = (eta[2] + dt*k2[2]/2)%1.
        
        # ========= mapping evaluation =============
        span1f = int(eta[0]*nelf[0]) + pf1
        span2f = int(eta[1]*nelf[1]) + pf2
        span3f = int(eta[2]*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
        # ============================================

        # ========== field evaluation ==============
        span1 = int(pos1*nel[0]) + pn1
        span2 = int(pos2*nel[1]) + pn2
        span3 = int(pos3*nel[2]) + pn3
        
        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
            
            linalg.matrix_vector(Ginv, u, k3_u)
            
        elif basis_u ==2:
            u[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)

            k3_u[:] = u/det_df
        
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k3_v)

        k3[:] = k3_v[:] + k3_u[:]
        # ------------------------------------------------------------------
        
        
        # ------------------ step 4 in Runge-Kutta method ------------------
        pos1   = (eta[0] + dt*k3[0])%1.
        pos2   = (eta[1] + dt*k3[1])%1.
        pos3   = (eta[2] + dt*k3[2])%1.
        
        # ========= mapping evaluation =============
        span1f = int(eta[0]*nelf[0]) + pf1
        span2f = int(eta[1]*nelf[1]) + pf2
        span3f = int(eta[2]*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
        # ============================================

        # ========== field evaluation ==============
        span1 = int(pos1*nel[0]) + pn1
        span2 = int(pos2*nel[1]) + pn2
        span3 = int(pos3*nel[2]) + pn3
        
        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
            
            linalg.matrix_vector(Ginv, u, k4_u)
            
        elif basis_u ==2:
            u[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)

            k4_u[:] = u/det_df
        
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k4_v)

        k4[:] = k4_v[:] + k4_u[:]
        # ------------------------------------------------------------------

        #  ---------------- update logical coordinates ---------------------
        particles[0, ip] = (eta[0] + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6)%1.0
        particles[1, ip] = (eta[1] + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6)%1.0
        particles[2, ip] = (eta[2] + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6)%1.0

        # ------------------------------------------------------------------
    #$ omp end parallel
    
    ierr = 0


# ==========================================================================================================
def pusher_rk4_pc_perp(particles : 'float[:,:]', dt : 'float', t1 : 'float[:]', t2 : 'float[:]', t3 : 'float[:]', p : 'int[:]', nel : 'int[:]', nbase_n : 'int[:]', nbase_d : 'int[:]', np : 'int', u1 : 'float[:,:,:]', u2 : 'float[:,:,:]', u3 : 'float[:,:,:]', basis_u : 'int', kind_map : 'int', params_map : 'float[:]', tf1 : 'float[:]', tf2 : 'float[:]', tf3 : 'float[:]', pf : 'int[:]', nelf : 'int[:]', nbasef : 'int[:]', cx : 'float[:,:,:]', cy : 'float[:,:,:]', cz : 'float[:,:,:]'):
    
    from numpy import empty

    #============== for velocity evaluation ============
    # spline degrees
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1
    
    # p + 1 non-vanishing basis functions up tp degree p
    b1 = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2 = empty((pn2 + 1, pn2 + 1), dtype=float)
    b3 = empty((pn3 + 1, pn3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1 = empty( pn1, dtype=float)
    l2 = empty( pn2, dtype=float)
    l3 = empty( pn3, dtype=float)
    
    r1 = empty( pn1, dtype=float)
    r2 = empty( pn2, dtype=float)
    r3 = empty( pn3, dtype=float)
    
    # scaling arrays for M-splines
    d1 = empty( pn1, dtype=float)
    d2 = empty( pn2, dtype=float)
    d3 = empty( pn3, dtype=float)
    
    # p + 1 non-vanishing derivatives
    der1 = empty(pn1 + 1, dtype=float)
    der2 = empty(pn2 + 1, dtype=float)
    der3 = empty(pn3 + 1, dtype=float)
    
    # non-vanishing N-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)
    
    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # # velocity field at particle position
    u = empty(3, dtype=float)
    # ==========================================================


    # ================ for mapping evaluation ==================
    # spline degrees
    pf1 = pf[0]
    pf2 = pf[1]
    pf3 = pf[2]
    
    # pf + 1 non-vanishing basis functions up tp degree pf
    b1f = empty((pf1 + 1, pf1 + 1), dtype=float)
    b2f = empty((pf2 + 1, pf2 + 1), dtype=float)
    b3f = empty((pf3 + 1, pf3 + 1), dtype=float)
    
    # left and right values for spline evaluation
    l1f = empty( pf1, dtype=float)
    l2f = empty( pf2, dtype=float)
    l3f = empty( pf3, dtype=float)
    
    r1f = empty( pf1, dtype=float)
    r2f = empty( pf2, dtype=float)
    r3f = empty( pf3, dtype=float)
    
    # scaling arrays for M-splines
    d1f = empty( pf1, dtype=float)
    d2f = empty( pf2, dtype=float)
    d3f = empty( pf3, dtype=float)
    
    # pf + 1 derivatives
    der1f = empty( pf1 + 1, dtype=float)
    der2f = empty( pf2 + 1, dtype=float)
    der3f = empty( pf3 + 1, dtype=float)
    
    # needed mapping quantities
    df      = empty((3, 3), dtype=float)
    dfinv   = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    Ginv    = empty((3, 3), dtype=float)
    fx      = empty( 3    , dtype=float)
    # ========================================================
    
    
    # ======= particle position and velocity =================
    eta = empty(3, dtype=float)
    v   = empty(3, dtype=float)
    # ========================================================
    
    
    # ===== intermediate stps in 4th order Runge-Kutta =======
    k1   = empty(3, dtype=float)  
    k2   = empty(3, dtype=float)  
    k3   = empty(3, dtype=float)  
    k4   = empty(3, dtype=float)
    k1_u = empty(3, dtype=float)  
    k2_u = empty(3, dtype=float)  
    k3_u = empty(3, dtype=float)  
    k4_u = empty(3, dtype=float) 
    k1_v = empty(3, dtype=float)  
    k2_v = empty(3, dtype=float)  
    k3_v = empty(3, dtype=float)  
    k4_v = empty(3, dtype=float)  
    # ========================================================
    
    
    #$ omp parallel private(ip, eta, v, pos1, pos2, pos3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, dfinv, dfinv_t, Ginv, det_df, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, der1, der2, der3, bn1, bn2, bn3, bd1, bd2, bd3, u, k1, k2, k3, k4, k1_u, k2_u, k3_u, k4_u, k1_v, k2_v, k3_v, k4_v)
    #$ omp for 
    for ip in range(np):
        
        eta[:] = particles[0:3, ip]
        v[:]   = particles[3:6, ip]

        # ----------- step 1 in Runge-Kutta method -----------------------
        # ========= mapping evaluation =============
        pos1 = eta[0]
        pos2 = eta[1]
        pos3 = eta[2]

        span1f = int(eta[0]*nelf[0]) + pf1
        span2f = int(eta[1]*nelf[1]) + pf2
        span3f = int(eta[2]*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv) ###########
        # ============================================

        # ========== field evaluation ==============
        span1 = int(pos1*nel[0]) + pn1
        span2 = int(pos2*nel[1]) + pn2
        span3 = int(pos3*nel[2]) + pn3
        
        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
            
            linalg.matrix_vector(Ginv, u, k1_u)
            
        elif basis_u ==2:
            u[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)

            k1_u[:] = u/det_df
        
        k1_u[0] = 0.
        
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k1_v)

        k1[:] = k1_v[:] + k1_u[:]

        # ------------------------------------------------------------------
        
        
        # ----------------- step 2 in Runge-Kutta method -------------------
        pos1   = (eta[0] + dt*k1[0]/2)%1.
        pos2   = (eta[1] + dt*k1[1]/2)%1.
        pos3   = (eta[2] + dt*k1[2]/2)%1.
        
        # ========= mapping evaluation =============
        span1f = int(eta[0]*nelf[0]) + pf1
        span2f = int(eta[1]*nelf[1]) + pf2
        span3f = int(eta[2]*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
        # ============================================

        # ========== field evaluation ==============
        span1 = int(pos1*nel[0]) + pn1
        span2 = int(pos2*nel[1]) + pn2
        span3 = int(pos3*nel[2]) + pn3
        
        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
            
            linalg.matrix_vector(Ginv, u, k2_u)
            
        elif basis_u ==2:
            u[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)

            k2_u[:] = u/det_df
        
        k2_u[0] = 0.
        
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k2_v)

        k2[:] = k2_v[:] + k2_u[:]
        # ------------------------------------------------------------------
        
        
        # ------------------ step 3 in Runge-Kutta method ------------------
        pos1   = (eta[0] + dt*k2[0]/2)%1.
        pos2   = (eta[1] + dt*k2[1]/2)%1.
        pos3   = (eta[2] + dt*k2[2]/2)%1.
        
        # ========= mapping evaluation =============
        span1f = int(eta[0]*nelf[0]) + pf1
        span2f = int(eta[1]*nelf[1]) + pf2
        span3f = int(eta[2]*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
        # ============================================

        # ========== field evaluation ==============
        span1 = int(pos1*nel[0]) + pn1
        span2 = int(pos2*nel[1]) + pn2
        span3 = int(pos3*nel[2]) + pn3
        
        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
            
            linalg.matrix_vector(Ginv, u, k3_u)
            
        elif basis_u ==2:
            u[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)

            k3_u[:] = u/det_df
            
        k3_u[0] = 0.
        
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k3_v)

        k3[:] = k3_v[:] + k3_u[:]
        # ------------------------------------------------------------------
        
        
        # ------------------ step 4 in Runge-Kutta method ------------------
        pos1   = (eta[0] + dt*k3[0])%1.
        pos2   = (eta[1] + dt*k3[1])%1.
        pos3   = (eta[2] + dt*k3[2])%1.
        
        # ========= mapping evaluation =============
        span1f = int(eta[0]*nelf[0]) + pf1
        span2f = int(eta[1]*nelf[1]) + pf2
        span3f = int(eta[2]*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)

        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))

        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)

        # evaluate Ginv matrix
        linalg.matrix_matrix(dfinv, dfinv_t, Ginv)
        # ============================================

        # ========== field evaluation ==============
        span1 = int(pos1*nel[0]) + pn1
        span2 = int(pos2*nel[1]) + pn2
        span3 = int(pos3*nel[2]) + pn3
        
        # evaluation of basis functions and derivatives
        bsp.basis_funs_and_der(t1, pn1, pos1, span1, l1, r1, b1, d1, der1)
        bsp.basis_funs_and_der(t2, pn2, pos2, span2, l2, r2, b2, d2, der2)
        bsp.basis_funs_and_der(t3, pn3, pos3, span3, l3, r3, b3, d3, der3)
        
        # N-splines and D-splines at particle positions
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]

        # velocity field
        if basis_u == 1:
            u[0] = eva3.evaluation_kernel_3d(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
            
            linalg.matrix_vector(Ginv, u, k4_u)
            
        elif basis_u ==2:
            u[0] = eva3.evaluation_kernel_3d(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
            u[1] = eva3.evaluation_kernel_3d(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
            u[2] = eva3.evaluation_kernel_3d(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)

            k4_u[:] = u/det_df
            
        k4_u[0] = 0.
        
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k4_v)

        k4[:] = k4_v[:] + k4_u[:]
        # ------------------------------------------------------------------

        #  ---------------- update logical coordinates ---------------------
        particles[0, ip] = (eta[0] + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6)%1.0
        particles[1, ip] = (eta[1] + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6)%1.0
        particles[2, ip] = (eta[2] + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6)%1.0

        # ------------------------------------------------------------------
    #$ omp end parallel
    
    ierr = 0



# ==========================================================================================================
def quicksort(a: 'float[:]', lo: 'int', hi: 'int'):
    """
    Implementation of the quicksort sorting algorithm

    Parameters:
    -----------
        a : array
            list that is to be sorted
        
        lo : integer
            lower index from which the sort to start
        
        hi : integer
            upper index until which the sort is to be done
    """
    i = lo
    j = hi
    while i < hi:
        pivot = a[(lo + hi) // 2]
        while i <= j :
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                tmp  = a[i]
                a[i] = a[j]
                a[j] = tmp
                i += 1
                j -= 1
        if lo < j:
            quicksort(a, lo, j)
        lo = i
        j = hi


# ==========================================================================================================
def find_taus(eta: 'float', eta_next: 'float', Nel: 'int', breaks: 'float[:]', uniform: 'int', tau_list: 'float[:]'):
    """
    Find the values of tau for which the particle crosses the cell boundaries while going from eta to eta_next

    Parameters:
    -----------
        eta : float
            old position
 
        eta_next : float
            new position

        Nel : integer
            contains the number of elements in this direction
        
        breaks : array
            break points in this direction

        uniform : integer
            0 if the grid is non-uniform, 1 if the grid is uniform
    """

    from numpy import floor

    if uniform == 1:
        index      = int( floor( eta * Nel ) )
        index_next = int( floor( eta_next * Nel ) )
        length     = int( abs( index_next - index ) )
        
        # break = eta / dx = eta * Nel

        for i in range(length):
            if index_next > index:
                tau_list[i] = (1.0 / Nel * (index + i + 1) - eta) / (eta_next - eta)
            elif index > index_next:
                tau_list[i] = (eta - 1.0 / Nel * (index - i)) / (eta - eta_next)
    
    elif uniform == 0:
        # TODO
        print('Not implemented yet')
    
    else:
        print('ValueError, uniform must be 1 or 0 !')


# ==========================================================================================================
def aux_fun_x_v_stat_e(particle: 'float[:]', kind_map: 'int', params_map: 'float[:]', dt: 'float', p: 'int[:]', Nel: 'int[:]', breaks1: 'float[:]', breaks2: 'float[:]', breaks3: 'float[:]', t1: 'float[:]', t2: 'float[:]', t3: 'float[:]', indN1: 'int[:,:]', indN2: 'int[:,:]', indN3: 'int[:,:]', indD1: 'int[:,:]', indD2: 'int[:,:]', indD3: 'int[:,:]', loc1: 'float[:]', loc2: 'float[:]', loc3: 'float[:]', weight1: 'float[:]', weight2: 'float[:]', weight3: 'float[:]', nbase_n: 'int[:]', nbase_d: 'int[:]', e0_coeffs: 'float[:]', eps: 'float[:]', maxiter: 'int', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]') -> 'int':
    """
    Auxiliary function for the pusher_x_v_static_efield, introduced to enable time-step splitting if scheme does not converge for the standard dt

    Parameters:
    -----------
        particle : array
            shape(7), contains the values for the positions [0:3], velocities [3:6], and weights [6]
        
        dt : float
            time stepping

        p : int array
            contains the degrees of the basis splines in each direction
        
        Nel : int array
            contains the number of elements in each direction
        
        breaks1 : array
            contains the break points in direction 1

        breaks2 : array
            contains the break points in direction 2

        breaks3 : array
            contains the break points in direction 3
        
        t1 : array
            contains the knot vector in direction 1

        t2 : array
            contains the knot vector in direction 2

        t3 : array
            contains the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3
        
        indd1 : array
            indD[0] from TensorSpline class, contains the global indices of non-zero D-splines in direction 1

        indd2 : array
            indD[1] from TensorSpline class, contains the global indices of non-zero D-splines in direction 2

        indd3 : array
            indD[2] from TensorSpline class, contains the global indices of non-zero D-splines in direction 3
        
        loc1 : array
            contains the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in direction 1

        loc2 : array
            contains the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in direction 2

        loc3 : array
            contains the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in direction 3
        
        weight1 : array
            contains the values of the weights for the Legendre-Gauss quadrature in direction 1
        
        weight2 : array
            contains the values of the weights for the Legendre-Gauss quadrature in direction 2
        
        weight3 : array
            contains the values of the weights for the Legendre-Gauss quadrature in direction 3
        
        nbase_n : int array
            contains 3 values for the dimensions of the univariate spline spaces

        nbase_d : int array
            contains 3 values for the dimensions of the univariate spline spaces
        
        e0_coeffs : array
            shape (3*p[0]*p[1]*p[2],) contains the values of the coefficient of the electric field
        
        eps: array
            determines the accuracy for the position (0th element) and velocity (1st element) with which the implicit scheme is executed

        maxiter : integer
            sets the maximum number of iterations for the iterative scheme
    """

    from numpy import empty, abs, floor

    df      = empty( (3,3), dtype=float )
    df_inv  = empty( (3,3), dtype=float )
    fx      = empty( 3    , dtype=float )

    # total number of basis functions : B-splines (pn   ) and D-splines (pd)
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]

    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1

    # non-vanishing B-splines at particle position
    bn1 = empty( pn1 + 1, dtype=float)
    bn2 = empty( pn2 + 1, dtype=float)
    bn3 = empty( pn3 + 1, dtype=float)

    # non-vanishing D-splines at particle position
    bd1 = empty( pd1 + 1, dtype=float)
    bd2 = empty( pd2 + 1, dtype=float)
    bd3 = empty( pd3 + 1, dtype=float)

    # number of quadrature points in direction 1
    n_quad1 = int( floor( pd1*pn2*pn3 / 2 + 1 ) )
    # number of quadrature points in direction 2
    n_quad2 = int( floor( pn1*pd2*pn3 / 2 + 1 ) )
    # number of quadrature points in direction 3
    n_quad3 = int( floor( pn1*pn2*pd3 / 2 + 1 ) )

    eps_pos = eps[0]
    eps_vel = eps[1]

    # position
    eta1 = particle[0]
    eta2 = particle[1]
    eta3 = particle[2]

    # velocities
    v1 = particle[3]
    v2 = particle[4]
    v3 = particle[5]

    # set initial value for x_k^{n+1}
    eta1_curr = eta1
    eta2_curr = eta2
    eta3_curr = eta3

    # set initial value for v_k^{n+1}
    v1_curr = v1
    v2_curr = v2
    v3_curr = v3

    # Use Euler method as a predictor for positions
    mapping_fast.dl_all(kind_map, params_map, t1, t2, t3, p, cx, cy, cz, indN1, indN2, indN3, eta1, eta2, eta3, df, fx, 0)
    mapping_fast.df_inv_all(df, df_inv)

    v1_curv = df_inv[0,0]*(v1_curr + v1) + df_inv[0,1]*(v2_curr + v2) + df_inv[0,2]*(v3_curr + v3)
    v2_curv = df_inv[1,0]*(v1_curr + v1) + df_inv[1,1]*(v2_curr + v2) + df_inv[1,2]*(v3_curr + v3)
    v3_curv = df_inv[2,0]*(v1_curr + v1) + df_inv[2,1]*(v2_curr + v2) + df_inv[2,2]*(v3_curr + v3)

    eta1_next = ( eta1 + dt * v1_curv / 2. )%1
    eta2_next = ( eta2 + dt * v2_curv / 2. )%1
    eta3_next = ( eta3 + dt * v3_curv / 2. )%1

    # set some initial value for v_next
    v1_next = v1_curr
    v2_next = v2_curr
    v3_next = v3_curr

    runs = 0

    while abs(eta1_next - eta1_curr) > eps_pos or abs(eta2_next - eta2_curr) > eps_pos or abs(eta3_next - eta3_curr) > eps_pos or abs(v1_next - v1_curr) > eps_vel or abs(v2_next - v2_curr) > eps_vel or abs(v3_next - v3_curr) > eps_vel:

        # update the positions and velocities
        eta1_curr  = eta1_next
        eta2_curr  = eta2_next
        eta3_curr  = eta3_next

        v1_curr    = v1_next 
        v2_curr    = v2_next 
        v3_curr    = v3_next

        # find Jacobian matrix
        mapping_fast.dl_all(kind_map, params_map, t1, t2, t3, p, cx, cy, cz, indN1, indN2, indN3, (eta1_curr + eta1)/2, (eta2_curr + eta2)/2, (eta3_curr + eta3)/2, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, df_inv)

        # ======================================================================================
        # update the positions and place them back into the computational domain
        v1_curv = df_inv[0,0]*(v1_curr + v1) + df_inv[0,1]*(v2_curr + v2) + df_inv[0,2]*(v3_curr + v3)
        v2_curv = df_inv[1,0]*(v1_curr + v1) + df_inv[1,1]*(v2_curr + v2) + df_inv[1,2]*(v3_curr + v3)
        v3_curv = df_inv[2,0]*(v1_curr + v1) + df_inv[2,1]*(v2_curr + v2) + df_inv[2,2]*(v3_curr + v3)

        # x_{n+1} = x_n + dt/2 * DF^{-1}(x_{n+1}/2 + x_n/2) * (v_{n+1} + v_n)
        eta1_next = ( eta1 + dt * v1_curv / 2. )%1
        eta2_next = ( eta2 + dt * v2_curv / 2. )%1
        eta3_next = ( eta3 + dt * v3_curv / 2. )%1



        # ======================================================================================
        # Compute tau-values in [0,1] for crossings of cell-boundaries

        index1      = int( floor( eta1_curr * Nel[0] ) )
        index1_next = int( floor( eta1_next * Nel[0] ) )
        length1     = int( abs( index1_next - index1 ) )

        index2      = int( floor( eta2_curr * Nel[1] ) )
        index2_next = int( floor( eta2_next * Nel[1] ) )
        length2     = int( abs( index2_next - index2 ) )

        index3      = int( floor( eta3_curr * Nel[2] ) )
        index3_next = int( floor( eta3_next * Nel[2] ) )
        length3     = int( abs( index3_next - index3 ) )

        length = length1 + length2 + length3

        taus = empty( length + 2, dtype=float )

        taus[0]          = 0.0
        taus[length + 1] = 1.0

        find_taus(eta1_curr, eta1_next, Nel[0], breaks1, 1, taus[1:length1+1])
        find_taus(eta2_curr, eta2_next, Nel[1], breaks2, 1, taus[length1+1:length1+length2+1])
        find_taus(eta3_curr, eta3_next, Nel[2], breaks3, 1, taus[length1+length2+1:length+1])

        quicksort(taus, 1, length)


        # ======================================================================================
        # update velocity in direction 1

        temp1 = 0.

        # loop over the cells
        for k in range( len(taus) - 1 ):

            a      = eta1 + taus[k] * ( eta1_curr - eta1 )
            b      = eta1 + taus[k+1] * ( eta1_curr - eta1 )
            factor = (b-a)/2
            adding = (a+b)/2
        
            for n in range(n_quad1):

                quad_pos1 = factor * loc1[n] + adding
                quad_pos2 = factor * loc2[n] + adding
                quad_pos3 = factor * loc3[n] + adding

                # spans (i.e. index for non-vanishing basis functions)
                span1 = bsp.find_span(t1, pn1, quad_pos1)
                span2 = bsp.find_span(t2, pn2, quad_pos2)
                span3 = bsp.find_span(t3, pn3, quad_pos3)

                # compute bn, bd, i.e. values for non-vanishing B-/D-splines at quadrature point
                bsp.b_d_splines_slim(t1, pn1, quad_pos1, span1, bn1, bd1)
                bsp.b_d_splines_slim(t2, pn2, quad_pos2, span2, bn2, bd2)
                bsp.b_d_splines_slim(t3, pn3, quad_pos3, span3, bn3, bd3)

                # find global index where non-zero basis functions begin
                ie1 = span1 - p[0]
                ie2 = span2 - p[1]
                ie3 = span3 - p[2]

                # (DNN)
                for il1 in range(pd1 + 1):
                    i1 = indD1[ie1,il1]
                    # i1 = indN1[ie1-1,il1+1]
                    bi1 = bd1[il1]
                    for il2 in range(pn2 +1):
                        i2 = indN2[ie2,il2]
                        bi2 = bi1 * bn2[il2]
                        for il3 in range(pn3 + 1):
                            i3 = indN3[ie3,il3]
                            bi3 = bi2 * bn3[il3] * e0_coeffs[ nbase_n[1]*nbase_n[2]*i1 + nbase_n[2]*i2 + i3 ]

                            temp1 += bi3 * weight1[n]

        # ======================================================================================
        # update velocity in direction 2

        temp2 = 0.

        # loop over the cells
        for k in range( len(taus) - 1 ):

            a      = eta2 + taus[k] * ( eta2_curr - eta2 )
            b      = eta2 + taus[k+1] * ( eta2_curr - eta2 )
            factor = (b-a)/2
            adding = (a+b)/2
        
            for n in range(n_quad2):

                quad_pos1 = factor * loc1[n] + adding
                quad_pos2 = factor * loc2[n] + adding
                quad_pos3 = factor * loc3[n] + adding

                # spans (i.e. index for non-vanishing basis functions)
                span1 = bsp.find_span(t1, pn1, quad_pos1)
                span2 = bsp.find_span(t2, pn2, quad_pos2)
                span3 = bsp.find_span(t3, pn3, quad_pos3)

                # compute bn, bd, i.e. values for non-vanishing B-/D-splines at quadrature point
                bsp.b_d_splines_slim(t1, pn1, quad_pos1, span1, bn1, bd1)
                bsp.b_d_splines_slim(t2, pn2, quad_pos2, span2, bn2, bd2)
                bsp.b_d_splines_slim(t3, pn3, quad_pos3, span3, bn3, bd3)

                # find global index where non-zero basis functions begin
                ie1 = span1 - p[0]
                ie2 = span2 - p[1]
                ie3 = span3 - p[2]

                # (NDN)
                for il1 in range(pn1 + 1):
                    i1 = indN1[ie1,il1]
                    bi1 = bn1[il1]
                    for il2 in range(pd2 +1):
                        i2 = indD2[ie2,il2]
                        # i2 = indN2[ie2-1,il2+1]
                        bi2 = bi1 * bd2[il2]
                        for il3 in range(pn3 + 1):
                            i3 = indN3[ie3,il3]
                            bi3 = bi2 * bn3[il3] * e0_coeffs[ nbase_d[1]*nbase_n[2]*i1 + nbase_n[2]*i2 + i3 ]

                            temp2 += bi3 * weight2[n]
            
        # ======================================================================================
        # update velocity in direction 3

        temp3 = 0.

        # loop over the cells
        for k in range( len(taus) - 1 ):

            a      = eta3 + taus[k] * ( eta3_curr - eta3 )
            b      = eta3 + taus[k+1] * ( eta3_curr - eta3 )
            factor = (b-a)/2
            adding = (a+b)/2
        
            for n in range(n_quad3):

                quad_pos1 = factor * loc1[n] + adding
                quad_pos2 = factor * loc2[n] + adding
                quad_pos3 = factor * loc3[n] + adding

                # spans (i.e. index for non-vanishing basis functions)
                span1 = bsp.find_span(t1, pn1, quad_pos1)
                span2 = bsp.find_span(t2, pn2, quad_pos2)
                span3 = bsp.find_span(t3, pn3, quad_pos3)

                # compute bn, bd, i.e. values for non-vanishing B-/D-splines at quadrature point
                bsp.b_d_splines_slim(t1, pn1, quad_pos1, span1, bn1, bd1)
                bsp.b_d_splines_slim(t2, pn2, quad_pos2, span2, bn2, bd2)
                bsp.b_d_splines_slim(t3, pn3, quad_pos3, span3, bn3, bd3)

                # find global index where non-zero basis functions begin
                ie1 = span1 - p[0]
                ie2 = span2 - p[1]
                ie3 = span3 - p[2]

                # (NND)
                for il1 in range(pn1 + 1):
                    i1 = indN1[ie1,il1]
                    bi1 = bn1[il1]
                    for il2 in range(pn2 +1):
                        i2 = indN2[ie2,il2]
                        bi2 = bi1 * bn2[il2]
                        for il3 in range(pd3 + 1):
                            i3 = indD3[ie3,il3]
                            # i3 = indN3[ie3-1,il3+1]
                            bi3 = bi2 * bd3[il3] * e0_coeffs[ nbase_n[1]*nbase_d[2]*i1 + nbase_d[2]*i2 + i3 ]

                            temp3 += bi3 * weight3[n]

        # v_{n+1} = v_n + dt * DF^{-T}(x_n) * int_0^1 d tau ( E(x_n + tau*(x_{n+1} - x_n) ) )
        v1_next = v1 + dt * ( df_inv[0,0]*temp1 + df_inv[1,0]*temp2 + df_inv[2,0]*temp3 )
        v2_next = v2 + dt * ( df_inv[0,1]*temp1 + df_inv[1,1]*temp2 + df_inv[2,1]*temp3 )
        v3_next = v3 + dt * ( df_inv[0,2]*temp1 + df_inv[1,2]*temp2 + df_inv[2,2]*temp3 )

        runs += 1
        del(taus)

        if runs == maxiter:
            break

    if runs < maxiter:
        # print('For convergence this took runs:', runs)
        # print()
        runs = 0

    # write the results in the particle array and impose periodic boundary conditions on the particles by taking modulo 1
    particle[0] = eta1_next%1
    particle[1] = eta2_next%1
    particle[2] = eta3_next%1
    particle[3] = v1_next
    particle[4] = v2_next
    particle[5] = v3_next

    return runs


# ==========================================================================================================
def pusher_x_v_static_efield(particles: 'float[:,:]', kind_map: 'int', params_map: 'float[:]', dt: 'float', p: 'int[:]', Nel: 'int[:]', breaks1: 'float[:]', breaks2: 'float[:]', breaks3: 'float[:]', t1: 'float[:]', t2: 'float[:]', t3: 'float[:]', indN1: 'int[:,:]', indN2: 'int[:,:]', indN3: 'int[:,:]', indD1: 'int[:,:]', indD2: 'int[:,:]', indD3: 'int[:,:]', loc1: 'float[:]', loc2: 'float[:]', loc3: 'float[:]', weight1: 'float[:]', weight2: 'float[:]', weight3: 'float[:]', np: 'int', nbase_n: 'int[:]', nbase_d: 'int[:]', e0_coeffs: 'float[:]', eps: 'float[:]', maxiter: 'int', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]'):
    """
    particle pusher for ODE dx/dt = v ; dv/dt = q/m * e_o(x)

    Parameters : 
    ------------
        particles : array
            shape(7, np), contains the values for the positions [:3,], velocities [3:6,], and weights [6,]
        
        dt : float
            time stepping

        p : int array
            contains the degrees of the basis splines in each direction
        
        Nel : int array
            contains the number of elements in each direction

        breaks1 : array
            contains the break points in direction 1

        breaks2 : array
            contains the break points in direction 2

        breaks3 : array
            contains the break points in direction 3
        
        t1 : array
            contains the knot vector in direction 1

        t2 : array
            contains the knot vector in direction 2

        t3 : array
            contains the knot vector in direction 3
        
        indn1 : array
            indN[0] from TensorSpline class, contains the global indices of non-zero B-splines in direction 1

        indn2 : array
            indN[1] from TensorSpline class, contains the global indices of non-zero B-splines in direction 2

        indn3 : array
            indN[2] from TensorSpline class, contains the global indices of non-zero B-splines in direction 3
        
        indd1 : array
            indD[0] from TensorSpline class, contains the global indices of non-zero D-splines in direction 1

        indd2 : array
            indD[1] from TensorSpline class, contains the global indices of non-zero D-splines in direction 2

        indd3 : array
            indD[2] from TensorSpline class, contains the global indices of non-zero D-splines in direction 3
        
        loc1 : array
            contains the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in direction 1

        loc2 : array
            contains the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in direction 2

        loc3 : array
            contains the positions of the Legendre-Gauss quadrature points of necessary order to integrate basis splines exactly in direction 3
        
        weight1 : array
            contains the values of the weights for the Legendre-Gauss quadrature in direction 1
        
        weight2 : array
            contains the values of the weights for the Legendre-Gauss quadrature in direction 2
        
        weight3 : array
            contains the values of the weights for the Legendre-Gauss quadrature in direction 3
        
        np : int
            number of particles
        
        nbase_n : int array
            contains 3 values for the dimensions of the univariate spline spaces

        nbase_d : int array
            contains 3 values for the dimensions of the univariate spline spaces
        
        e0_coeffs : array
            shape (3*p[0]*p[1]*p[2],) contains the values of the coefficient of the electric field
        
        eps: array
            determines the accuracy for the position (0th element) and velocity (1st element) with which the implicit scheme is executed

        maxiter : integer
            sets the maximum number of iterations for the iterative scheme
    """

    from numpy import zeros

    particle = zeros( 7, dtype=float )

    #$ omp parallel private(ip, run, temp, k, m, particle, dt2)
    #$ omp for
    for ip in range(np):

        particle[:] = particles[:,ip]

        run = 1
        k   = 0
        
        while run != 0:
            k += 1
            if k == 5:
                print('Splitting the time steps into 4 has not been enough, aborting the iteration.')
                print()
                break

            run = 0
            
            dt2 = dt/k

            for m in range(k):
                temp = aux_fun_x_v_stat_e(particle, kind_map, params_map, dt2, p, Nel, breaks1, breaks2, breaks3, t1, t2, t3, indN1, indN2, indN3, indD1, indD2, indD3, loc1, loc2, loc3, weight1, weight2, weight3, nbase_n, nbase_d, e0_coeffs, eps, maxiter, cx, cy, cz)
                run = run + temp

        # write the results in the particles array
        particles[:,ip] = particle[:]
    
    #$ omp end parallel

    ierr = 0