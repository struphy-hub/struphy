# import pyccel decorators
from pyccel.decorators import types

# import module for matrix-matrix and matrix-vector multiplications
import struphy.linear_algebra.core as linalg

# import modules for mapping evaluation
import struphy.geometry.mappings_3d_fast as mapping_fast

# import modules for B-spline evaluation
import struphy.feec.bsplines_kernels as bsp

import struphy.feec.basics.spline_evaluation_2d as eva2
import struphy.feec.basics.spline_evaluation_3d as eva3


# ==========================================================================================================
@types('double[:,:]','double','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:]','int')
def pusher_step3(particles, dt, t1, t2, t3, p, nel, nbase_n, nbase_d, np, b_eq_1, b_eq_2, b_eq_3, b_p_1, b_p_2, b_p_3, b_norm, u1, u2, u3, basis_u, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz, mu, n_tor):
    
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
    
    # sin/cos at particle position (only necessary for 2d)
    sc = empty(2, dtype=float)
    
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
    
    
    #$ omp parallel
    #$ omp do private (ip, eta1, eta2, eta3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df, dfinv, dfinv_t, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, der1, der2, der3, bn1, bn2, bn3, bd1, bd2, bd3, sc, u, u_cart, b, b_cart, b_grad, b_grad_cart, e_cart)
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
        
        if nel[2] > 0:
            
            span3 = int(eta3*nel[2]) + pn3
            
            bsp.basis_funs_and_der(t3, pn3, eta3, span3, l3, r3, b3, d3, der3)
            
            bn3[:] = b3[pn3, :]
            bd3[:] = b3[pd3, :pn3] * d3[:]
            
        else:
            
            sc[0] = sin(2*pi*n_tor*eta3)
            sc[1] = cos(2*pi*n_tor*eta3)
        
        # velocity field (0-form, push-forward with df)
        if basis_u == 0:
            
            if nel[2] > 0:
            
                u[0] = eva3.evaluation_kernel(pn1, pn2, pn3, bn1, bn2, bn3, span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], u1)
                u[1] = eva3.evaluation_kernel(pn1, pn2, pn3, bn1, bn2, bn3, span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], u2)
                u[2] = eva3.evaluation_kernel(pn1, pn2, pn3, bn1, bn2, bn3, span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], u3)
                
            else:
                
                u[0]  = eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1, span2, nbase_n[0], nbase_n[1], u1[:, :, 0])*sc[0]
                u[1]  = eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1, span2, nbase_n[0], nbase_n[1], u2[:, :, 0])*sc[0]
                u[2]  = eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1, span2, nbase_n[0], nbase_n[1], u3[:, :, 0])*sc[0]
                
                u[0] += eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1, span2, nbase_n[0], nbase_n[1], u1[:, :, 1])*sc[1]
                u[1] += eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1, span2, nbase_n[0], nbase_n[1], u2[:, :, 1])*sc[1]
                u[2] += eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1, span2, nbase_n[0], nbase_n[1], u3[:, :, 1])*sc[1]
            
            linalg.matrix_vector(df, u, u_cart)
        
        # velocity field (1-form, push forward with df^(-T))
        elif basis_u == 1:
            
            if nel[2] > 0:
            
                u[0] = eva3.evaluation_kernel(pd1, pn2, pn3, bd1, bn2, bn3, span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
                u[1] = eva3.evaluation_kernel(pn1, pd2, pn3, bn1, bd2, bn3, span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
                u[2] = eva3.evaluation_kernel(pn1, pn2, pd3, bn1, bn2, bd3, span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
                
            else:
                
                u[0]  = eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], u1[:, :, 0])*sc[0]
                u[1]  = eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], u2[:, :, 0])*sc[0]
                u[2]  = eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1 - 0, span2 - 0, nbase_n[0], nbase_n[1], u3[:, :, 0])*sc[0]

                u[0] += eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], u1[:, :, 1])*sc[1]
                u[1] += eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], u2[:, :, 1])*sc[1]
                u[2] += eva2.evaluation_kernel_2d(pn1, pn2, bn1, bn2, span1 - 0, span2 - 0, nbase_n[0], nbase_n[1], u3[:, :, 1])*sc[1]
            
            linalg.matrix_vector(dfinv_t, u, u_cart)
        
        # velocity field (2-form, push forward with df/|det df|)
        elif basis_u == 2:
            
            if nel[2] > 0:
            
                u[0] = eva3.evaluation_kernel(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
                u[1] = eva3.evaluation_kernel(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
                u[2] = eva3.evaluation_kernel(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)
                
            else:
                
                u[0]  = eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], u1[:, :, 0])*sc[0]
                u[1]  = eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], u2[:, :, 0])*sc[0]
                u[2]  = eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], u3[:, :, 0])*sc[0]

                u[0] += eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], u1[:, :, 1])*sc[1]
                u[1] += eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], u2[:, :, 1])*sc[1]
                u[2] += eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], u3[:, :, 1])*sc[1]
            
            linalg.matrix_vector(df, u, u_cart)
            u_cart[:] = u_cart/det_df
            
        
        # magnetic field (2-form)
        if nel[2] > 0:
            
            b[0] = eva3.evaluation_kernel(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], b_eq_1 + b_p_1)
            b[1] = eva3.evaluation_kernel(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], b_eq_2 + b_p_2)
            b[2] = eva3.evaluation_kernel(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], b_eq_3 + b_p_3)
            
        else:
            
            b[0]  = eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_eq_1[:, :, 0])
            b[1]  = eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_eq_2[:, :, 0])
            b[2]  = eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_eq_3[:, :, 0])

            # perturbed magnetic field (2-form)
            b[0] += eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_p_1[:, :, 0])*sc[0]
            b[1] += eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_p_2[:, :, 0])*sc[0]
            b[2] += eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_p_3[:, :, 0])*sc[0]

            b[0] += eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_p_1[:, :, 1])*sc[1]
            b[1] += eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_p_2[:, :, 1])*sc[1]
            b[2] += eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_p_3[:, :, 1])*sc[1]
        
        linalg.matrix_vector(df, b, b_cart)
        b_cart[:] = b_cart/det_df
        
        
        # gradient of absolute value of magnetic field (1-form)
        if nel[2] > 2:
            
            b_grad[0] = eva3.evaluation_kernel(pn1, pn2, pn3, der1, bn2, bn3, span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], b_norm)
            b_grad[1] = eva3.evaluation_kernel(pn1, pn2, pn3, bn1, der2, bn3, span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], b_norm)
            b_grad[2] = eva3.evaluation_kernel(pn1, pn2, pn3, bn1, bn2, der3, span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], b_norm)

            
        else:
            
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
        
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
    
# ==========================================================================================================
@types('double[:,:]','double','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int')
def pusher_step5(particles, dt, t1, t2, t3, p, nel, nbase_n, nbase_d, np, b_eq_1, b_eq_2, b_eq_3, b_p_1, b_p_2, b_p_3, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz, n_tor):
    
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
    
    # sin/cos at particle position (only necessary for 2d)
    sc = empty(2, dtype=float)
    
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
    x       = empty( 3    , dtype=float)
    df      = empty((3, 3), dtype=float)
    dfinv   = empty((3, 3), dtype=float)
    dfinv_t = empty((3, 3), dtype=float)
    # ==========================================================
    
    
    # ================ for solving linear 3x3 system ===========
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
    
    
    #$ omp parallel
    #$ omp do private (ip, eta1, eta2, eta3, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, bn1, bn2, bn3, bd1, bd2, bd3, sc, b, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, x, dfinv, dfinv_t, v, temp_mat1, temp_mat2, rhs, lhs, det_lhs, lhs1, lhs2, lhs3, det_lhs1, det_lhs2, det_lhs3) firstprivate(b_prod)
    for ip in range(np):
        
        # only do something if particle is inside the logical domain (s < 1)
        if particles[0, ip] > 1.0:
            continue
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
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
        
        if nel[2] > 0:
            
            span3 = int(eta3*nel[2]) + pn3
            
            bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)
            
            bn3[:] = b3[pn3, :]
            bd3[:] = b3[pd3, :pn3] * d3[:]
            
        else:
            
            sc[0] = sin(2*pi*n_tor*eta3)
            sc[1] = cos(2*pi*n_tor*eta3)
        
        # magnetic field (2-form)
        if nel[2] > 0:
            
            b[0] = eva3.evaluation_kernel(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], b_eq_1 + b_p_1)
            b[1] = eva3.evaluation_kernel(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], b_eq_2 + b_p_2)
            b[2] = eva3.evaluation_kernel(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], b_eq_3 + b_p_3)
            
        else:
            
            b[0]  = eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_eq_1[:, :, 0])
            b[1]  = eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_eq_2[:, :, 0])
            b[2]  = eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_eq_3[:, :, 0])

            # perturbed magnetic field (2-form)
            b[0] += eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_p_1[:, :, 0])*sc[0]
            b[1] += eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_p_2[:, :, 0])*sc[0]
            b[2] += eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_p_3[:, :, 0])*sc[0]

            b[0] += eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_p_1[:, :, 1])*sc[1]
            b[1] += eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_p_2[:, :, 1])*sc[1]
            b[2] += eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_p_3[:, :, 1])*sc[1]
        
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
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, x, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
        
        # evaluate transposed inverse Jacobian matrix
        linalg.transpose(dfinv, dfinv_t)
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
    
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
    
    
    
    
# ==========================================================================================================
@types('double[:,:]','double','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int')
def pusher_step5_ana(particles, dt, t1, t2, t3, p, nel, nbase_n, nbase_d, np, b_eq_1, b_eq_2, b_eq_3, b_p_1, b_p_2, b_p_3, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz, n_tor):
    
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
    
    # sin/cos at particle position (only necessary for 2d)
    sc = empty(2, dtype=float)
    
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
    
    
    #$ omp parallel
    #$ omp do private (ip, eta1, eta2, eta3, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, bn1, bn2, bn3, bd1, bd2, bd3, sc, b, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, fx, det_df, v, b_cart, b_norm, b0, vpar, vxb0, vperp, b0xvperp)
    for ip in range(np):
        
        # only do something if particle is inside the logical domain (s < 1)
        if particles[0, ip] > 1.0:
            continue
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
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
        
        if nel[2] > 0:
            
            span3 = int(eta3*nel[2]) + pn3
            
            bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)
            
            bn3[:] = b3[pn3, :]
            bd3[:] = b3[pd3, :pn3] * d3[:]
            
        else:
            
            sc[0] = sin(2*pi*n_tor*eta3)
            sc[1] = cos(2*pi*n_tor*eta3)
        
        # magnetic field (2-form)
        if nel[2] > 0:
            
            b[0] = eva3.evaluation_kernel(pn1, pd2, pd3, bn1, bd2, bd3, span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], b_eq_1 + b_p_1)
            b[1] = eva3.evaluation_kernel(pd1, pn2, pd3, bd1, bn2, bd3, span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], b_eq_2 + b_p_2)
            b[2] = eva3.evaluation_kernel(pd1, pd2, pn3, bd1, bd2, bn3, span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], b_eq_3 + b_p_3)
            
        else:
            
            b[0]  = eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_eq_1[:, :, 0])
            b[1]  = eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_eq_2[:, :, 0])
            b[2]  = eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_eq_3[:, :, 0])

            # perturbed magnetic field (2-form)
            b[0] += eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_p_1[:, :, 0])*sc[0]
            b[1] += eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_p_2[:, :, 0])*sc[0]
            b[2] += eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_p_3[:, :, 0])*sc[0]

            b[0] += eva2.evaluation_kernel_2d(pn1, pd2, bn1, bd2, span1 - 0, span2 - 1, nbase_n[0], nbase_d[1], b_p_1[:, :, 1])*sc[1]
            b[1] += eva2.evaluation_kernel_2d(pd1, pn2, bd1, bn2, span1 - 1, span2 - 0, nbase_d[0], nbase_n[1], b_p_2[:, :, 1])*sc[1]
            b[2] += eva2.evaluation_kernel_2d(pd1, pd2, bd1, bd2, span1 - 1, span2 - 1, nbase_d[0], nbase_d[1], b_p_3[:, :, 1])*sc[1]
        # ==========================================
        
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df, fx, 0)
        
        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))
        # ==========================================
        
        
        # ======== particle pushing ================
        v[:] = particles[3:6, ip]
        
        # push-forward of magnetic field
        linalg.matrix_vector(df, b, b_cart)
        b_cart[:] = b_cart/det_df
        
        #r =  1.0 + 0.1*eta1*cos(2*pi*eta2)
        
        #x = (1.0 + 0.1*eta1*cos(2*pi*eta2))*cos(2*pi*eta3)
        #y = (1.0 + 0.1*eta1*cos(2*pi*eta2))*sin(2*pi*eta3)
        #z =        0.1*eta1*sin(2*pi*eta2)
        
        #b_phi = 0.1*eta1/(2*r)
        #b_tor = 1.0/r
        
        #b_cart[0] = -b_phi*sin(2*pi*eta2)*cos(2*pi*eta3) - b_tor*sin(2*pi*eta3)
        #b_cart[1] =  b_phi*cos(2*pi*eta2)
        #b_cart[2] = -b_phi*sin(2*pi*eta2)*sin(2*pi*eta3) + b_tor*cos(2*pi*eta3)
        
        #b_cart[0] = -b_phi*sin(2*pi*eta2)*cos(2*pi*eta3) - b_tor*sin(2*pi*eta3)
        #b_cart[1] = -b_phi*sin(2*pi*eta2)*sin(2*pi*eta3) + b_tor*cos(2*pi*eta3)
        #b_cart[2] =  b_phi*cos(2*pi*eta2)
        
        #b_cart[0] = -(2*y + x*z)/(2*r**2)
        #b_cart[1] =  (2*x - y*z)/(2*r**2)
        #b_cart[2] =  (r - 1)/(2*r)
        
        # absolute value of magnetic field
        b_norm = sqrt(b_cart[0]**2 + b_cart[1]**2 + b_cart[2]**2)
        
        # normalized magnetic field direction
        b0[:] = b_cart/b_norm
        
        # parallel velocity v * b0
        vpar = v[0]*b0[0] + v[1]*b0[1] + v[2]*b0[2]
        
        # perpendicular velocity b0 x (v x b0)
        linalg.cross(v, b0, vxb0)
        linalg.cross(b0, vxb0, vperp)
        
        # analytical rotation
        linalg.cross(b0, vperp, b0xvperp)
        
        particles[3:6, ip] = vpar*b0 + cos(b_norm*dt)*vperp - sin(b_norm*dt)*b0xvperp
        # ==========================================
    
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0