# import pyccel decorators
from pyccel.decorators import types

# import module for matrix-matrix and matrix-vector multiplications
import hylife.linear_algebra.core as linalg

# import module for mapping evaluation
import hylife.geometry.mappings_3d_fast as mapping_fast

# import modules for B-spline evaluation
import hylife.utilitis_FEEC.bsplines_kernels as bsp
import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva

# ==========================================================================================================
@types('double[:,:]','double','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:]')
def pusher_step3(particles, dt, t1, t2, t3, p, nel, nbase_n, nbase_d, np, bb1, bb2, bb3, bnorm, u1, u2, u3, basis_u, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz, mu):
    
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
    df        = empty((3, 3), dtype=float)
    dfinv     = empty((3, 3), dtype=float)
    dfinv_t   = empty((3, 3), dtype=float)
    # ==========================================================
    
    
    #$ omp parallel
    #$ omp do private (ip, eta1, eta2, eta3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, det_df, dfinv, dfinv_t, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, der1, der2, der3, u, u_cart, b, b_cart, b_grad, b_grad_cart, e_cart)
    for ip in range(np):
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df)
        
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
        
        
        # velocity field (0-form, push-forward with df)
        if basis_u == 0:
            u[0] = eva.evaluation_kernel(pn1, pn2, pn3, b1[pn1], b2[pn2], b3[pn3], span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], u1)
            u[1] = eva.evaluation_kernel(pn1, pn2, pn3, b1[pn1], b2[pn2], b3[pn3], span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], u2)
            u[2] = eva.evaluation_kernel(pn1, pn2, pn3, b1[pn1], b2[pn2], b3[pn3], span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], u3)
            
            linalg.matrix_vector(df, u, u_cart)
        
        # velocity field (1-form, push forward with df^(-T))
        elif basis_u == 1:
            u[0] = eva.evaluation_kernel(pd1, pn2, pn3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pn3], span1 - 1, span2, span3, nbase_d[0], nbase_n[1], nbase_n[2], u1)
            u[1] = eva.evaluation_kernel(pn1, pd2, pn3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pn3], span1, span2 - 1, span3, nbase_n[0], nbase_d[1], nbase_n[2], u2)
            u[2] = eva.evaluation_kernel(pn1, pn2, pd3, b1[pn1], b2[pn2], b3[pd3, :pn3]*d3[:], span1, span2, span3 - 1, nbase_n[0], nbase_n[1], nbase_d[2], u3)
            
            linalg.matrix_vector(dfinv_t, u, u_cart)
        
        # velocity field (2-form, push forward with df/|det df|)
        elif basis_u == 2:
            u[0] = eva.evaluation_kernel(pn1, pd2, pd3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pd3, :pn3]*d3[:], span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], u1)
            u[1] = eva.evaluation_kernel(pd1, pn2, pd3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pd3, :pn3]*d3[:], span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], u2)
            u[2] = eva.evaluation_kernel(pd1, pd2, pn3, b1[pd1, :pn1]*d1[:], b2[pd2, :pn2]*d2[:], b3[pn3], span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], u3)
            
            linalg.matrix_vector(df, u, u_cart)
            u_cart[:] = u_cart/det_df
            
        
        # magnetic field (2-form)
        b[0] = eva.evaluation_kernel(pn1, pd2, pd3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pd3, :pn3]*d3[:], span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], bb1)
        b[1] = eva.evaluation_kernel(pd1, pn2, pd3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pd3, :pn3]*d3[:], span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], bb2)
        b[2] = eva.evaluation_kernel(pd1, pd2, pn3, b1[pd1, :pn1]*d1[:], b2[pd2, :pn2]*d2[:], b3[pn3], span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], bb3)
        
        linalg.matrix_vector(df, b, b_cart)
        b_cart[:] = b_cart/det_df
        
        # evaluation of grad(B) on logical domain (|B| is a 0-form, then grad(B) a 1-form, , push forward with df^(-T))
        b_grad[0] = eva.evaluation_kernel(pn1, pn2, pn3, der1, b2[pn2], b3[pn3], span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], bnorm)
        b_grad[1] = eva.evaluation_kernel(pn1, pn2, pn3, b1[pn1], der2, b3[pn3], span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], bnorm)
        b_grad[2] = eva.evaluation_kernel(pn1, pn2, pn3, b1[pn1], b2[pn2], der3, span1, span2, span3, nbase_n[0], nbase_n[1], nbase_n[2], bnorm)
        
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
@types('double[:,:]','double','int','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pusher_step4(particles, dt, np, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    
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
    
    
    #$ omp parallel
    #$ omp do private (ip, eta, v, pos1, pos2, pos3, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, dfinv, k1, k2, k3, k4)
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
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df)
        
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
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df)
        
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
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df)
        
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
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, pos1, pos2, pos3, df)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, dfinv)
                
        # pull-back of velocity
        linalg.matrix_vector(dfinv, v, k4)
        # ------------------------------------------------------------------
        
        
        #  ---------------- update logical coordinates ---------------------
        particles[0, ip] = (eta[0] + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6)%1.
        particles[1, ip] = (eta[1] + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6)%1.
        particles[2, ip] = (eta[2] + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6)%1.
        # ------------------------------------------------------------------
    
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0

    
    
    
# ==========================================================================================================
@types('double[:,:]','double','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pusher_step5(particles, dt, t1, t2, t3, p, nel, nbase_n, nbase_d, np, bb1, bb2, bb3, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    
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
    df        = empty((3, 3), dtype=float)
    dfinv     = empty((3, 3), dtype=float)
    dfinv_t   = empty((3, 3), dtype=float)
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
    #$ omp do private (ip, eta1, eta2, eta3, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, b, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, dfinv, dfinv_t, v, temp_mat1, temp_mat2, rhs, lhs, det_lhs, lhs1, lhs2, lhs3, det_lhs1, det_lhs2, det_lhs3) firstprivate(b_prod)
    for ip in range(np):
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3
        
        # evaluation of basis functions
        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)
        
        # magnetic field (2-form)
        b[0] = eva.evaluation_kernel(pn1, pd2, pd3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pd3, :pn3]*d3[:], span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], bb1)
        b[1] = eva.evaluation_kernel(pd1, pn2, pd3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pd3, :pn3]*d3[:], span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], bb2)
        b[2] = eva.evaluation_kernel(pd1, pd2, pn3, b1[pd1, :pn1]*d1[:], b2[pd2, :pn2]*d2[:], b3[pn3], span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], bb3)
        
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
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df)
        
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
@types('double[:,:]','double','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:,:]','double[:,:,:]','double[:,:,:]','int','double[:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def pusher_step5_ana(particles, dt, t1, t2, t3, p, nel, nbase_n, nbase_d, np, bb1, bb2, bb3, kind_map, params_map, tf1, tf2, tf3, pf, nelf, nbasef, cx, cy, cz):
    
    from numpy import empty, zeros, sqrt, cos, sin
    
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
    df    = empty((3, 3), dtype=float)
    # ==========================================================
    
    
    #$ omp parallel
    #$ omp do private (ip, eta1, eta2, eta3, span1, span2, span3, l1, l2, l3, r1, r2, r3, b1, b2, b3, d1, d2, d3, b, span1f, span2f, span3f, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, df, det_df, v, b_cart, b_norm, b0, vpar, vxb0, vperp, b0xvperp)
    for ip in range(np):
        
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]
        
        # ========== field evaluation ==============
        span1 = int(eta1*nel[0]) + pn1
        span2 = int(eta2*nel[1]) + pn2
        span3 = int(eta3*nel[2]) + pn3
        
        # evaluation of basis functions
        bsp.basis_funs_all(t1, pn1, eta1, span1, l1, r1, b1, d1)
        bsp.basis_funs_all(t2, pn2, eta2, span2, l2, r2, b2, d2)
        bsp.basis_funs_all(t3, pn3, eta3, span3, l3, r3, b3, d3)
        
        # magnetic field (2-form)
        b[0] = eva.evaluation_kernel(pn1, pd2, pd3, b1[pn1], b2[pd2, :pn2]*d2[:], b3[pd3, :pn3]*d3[:], span1, span2 - 1, span3 - 1, nbase_n[0], nbase_d[1], nbase_d[2], bb1)
        b[1] = eva.evaluation_kernel(pd1, pn2, pd3, b1[pd1, :pn1]*d1[:], b2[pn2], b3[pd3, :pn3]*d3[:], span1 - 1, span2, span3 - 1, nbase_d[0], nbase_n[1], nbase_d[2], bb2)
        b[2] = eva.evaluation_kernel(pd1, pd2, pn3, b1[pd1, :pn1]*d1[:], b2[pd2, :pn2]*d2[:], b3[pn3], span1 - 1, span2 - 1, span3, nbase_d[0], nbase_d[1], nbase_n[2], bb3)
        # ==========================================
        
        
        # ========= mapping evaluation =============
        span1f = int(eta1*nelf[0]) + pf1
        span2f = int(eta2*nelf[1]) + pf2
        span3f = int(eta3*nelf[2]) + pf3
        
        # evaluate Jacobian matrix
        mapping_fast.df_all(kind_map, params_map, tf1, tf2, tf3, pf, nbasef, span1f, span2f, span3f, cx, cy, cz, l1f, l2f, l3f, r1f, r2f, r3f, b1f, b2f, b3f, d1f, d2f, d3f, der1f, der2f, der3f, eta1, eta2, eta3, df)
        
        # evaluate Jacobian determinant
        det_df = abs(linalg.det(df))
        # ==========================================
        
        
        # ======== particle pushing ================
        v[:] = particles[3:6, ip]
        
        # push-forward of magnetic field
        linalg.matrix_vector(df/det_df, b, b_cart)
        
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